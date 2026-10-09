from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Event

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tests.fakes.fake_components import (  # noqa: E402
    FakeBackendAdapter,
    FakeRunner,
    FakeTaskPlugin,
    FakeVerifier,
)
from true_rl_agent.agent_core.ledger import (  # noqa: E402
    LEDGER_RECORD_SCHEMA_V4,
    LEDGER_RECORD_SCHEMA_VERSION,
    PROVIDER_AUTHORING_MODE,
    load_ledger,
)
from true_rl_agent.agent_core.live_loop import run_live_loop  # noqa: E402
from true_rl_agent.agent_core.session_epoch import (  # noqa: E402
    LEDGER_EPOCH_FILENAME,
    LiveSessionInProgressError,
    prepare_ledger_epoch,
)
from true_rl_agent.agent_core.session_genesis import (  # noqa: E402
    LEGACY_SESSION_GENESIS_SCHEMA_VERSION,
    SESSION_GENESIS_FILENAME,
    SessionGenesisArtifact,
    prepare_session_genesis,
)
from true_rl_agent.candidates.schema import (  # noqa: E402
    CandidateChangeSet,
    CandidatePackageRef,
)
from true_rl_agent.evaluation.acceptance_replay import (  # noqa: E402
    replay_acceptance_ledger,
)
from true_rl_agent.evaluation.binding import (  # noqa: E402
    EvaluationBindingV1,
    canonical_json_hash,
)
from true_rl_agent.json_contract import canonical_json_bytes  # noqa: E402
from true_rl_agent.policy import PolicyInitialization  # noqa: E402
from true_rl_agent.policy_baselines import (  # noqa: E402
    BaselineCompatibilitySpec,
    BaselineManifest,
    CheckpointProvenance,
)
from true_rl_agent.prompt_artifacts import (  # noqa: E402
    AUTHORING_EVIDENCE_SCHEMA_V2,
)
from true_rl_agent.runner.specs import CheckpointRef  # noqa: E402
from true_rl_agent.tasks.base import TaskSpec  # noqa: E402


class _BaselineRequiredTask(FakeTaskPlugin):
    @property
    def task_spec(self) -> TaskSpec:
        return replace(
            super().task_spec,
            supported_policy_initializations=frozenset({"warm_start"}),
            genesis_anchor_policy={"warm_start": "policy_baseline"},
        )


def _denied_predecessor_bytes(session_genesis_hash: str) -> bytes:
    return canonical_json_bytes(
        {
            "schema_version": "true_rl_agent.ledger_record.v2",
            "session_genesis_hash": session_genesis_hash,
            "decision": {"accepted": False},
        },
        trailing_newline=True,
    )


class _CountingRunner(FakeRunner):
    def __init__(self) -> None:
        self.baseline_eval_calls = 0
        self.train_specs = []

    def run_baseline_eval(self, eval_spec):
        self.baseline_eval_calls += 1
        return super().run_baseline_eval(eval_spec)

    def run_train(self, train_spec):
        self.train_specs.append(train_spec)
        return super().run_train(train_spec)


class _SessionBoundBackend(FakeBackendAdapter):
    def __init__(self) -> None:
        self.resolve_allow_create: list[bool] = []

    def _session_binding(self, task_spec: TaskSpec) -> EvaluationBindingV1:
        fixed = super().evaluation_binding_for(task_spec)
        return EvaluationBindingV1(
            eval_protocol_hash=fixed.eval_protocol_hash,
            acceptance_protocol_hash=fixed.acceptance_protocol_hash,
            task_protocol_hash=fixed.task_protocol_hash,
            seed_cohort_hash="9" * 64,
        )

    def resolve_session_evaluation_binding(
        self,
        *,
        session_id,
        session_dir,
        task_spec,
        allow_create,
    ):
        _ = session_id, session_dir
        self.resolve_allow_create.append(allow_create)
        return self._session_binding(task_spec)

    def validate_session_evaluation_binding(
        self,
        *,
        task_spec,
        evaluation_binding,
    ) -> None:
        if evaluation_binding != self._session_binding(task_spec):
            raise ValueError("test session evaluation binding mismatch")


class _DriftedSessionBoundBackend(_SessionBoundBackend):
    def _session_binding(self, task_spec: TaskSpec) -> EvaluationBindingV1:
        fixed = super()._session_binding(task_spec)
        return EvaluationBindingV1(
            eval_protocol_hash=fixed.eval_protocol_hash,
            acceptance_protocol_hash=fixed.acceptance_protocol_hash,
            task_protocol_hash=fixed.task_protocol_hash,
            seed_cohort_hash="8" * 64,
        )


class _ScoreProvider:
    def __init__(self, root: Path, scores: tuple[float, ...]) -> None:
        self.root = root
        self.scores = scores
        self.payloads: list[dict] = []

    def next_candidate(self, *, prompt_payload: dict) -> CandidatePackageRef:
        self.payloads.append(prompt_payload)
        round_number = int(prompt_payload["public_history"]["previous_rounds"]) + 1
        reward_expression = {
            1: "inputs.base_height + inputs.roll",
            2: "inputs.base_height - inputs.roll",
            3: "inputs.base_height + inputs.roll.square()",
        }[round_number]
        candidate_id = f"candidate-{round_number:03d}"
        candidate_dir = self.root / candidate_id
        source_dir = candidate_dir / "training_code"
        source_dir.mkdir(parents=True, exist_ok=True)
        (candidate_dir / "candidate_manifest.yaml").write_text(
            "schema_version: true_rl_agent.candidate.v1\n"
            f"candidate_id: {candidate_id}\n"
            "entrypoint: training_code.components:build_training_components\n",
            encoding="utf-8",
        )
        (source_dir / "components.py").write_text(
            "from true_rl_agent.candidates.components import RewardTerm, TrainingComponents\n\n"
            f"def reward_{round_number}(inputs):\n"
            f"    return {reward_expression}\n\n"
            "def build_training_components():\n"
            "    return TrainingComponents(\n"
            f"        name={candidate_id!r},\n"
            "        reward_terms=[RewardTerm(\n"
            f"            name='reward_{round_number}',\n"
            "            weight=1.0,\n"
            f"            fn=reward_{round_number},\n"
            "        )],\n"
            "    )\n",
            encoding="utf-8",
        )
        (candidate_dir / "rationale.md").write_text("Session candidate.\n", encoding="utf-8")
        (candidate_dir / "operation_diff.json").write_text("{}\n", encoding="utf-8")
        (candidate_dir / "diagnostics.json").write_text("{}\n", encoding="utf-8")
        return CandidatePackageRef(
            candidate_id=candidate_id,
            path=str(candidate_dir),
            change_set=CandidateChangeSet(frozenset({"reward_callable"})),
            metadata={"fake_public_score": self.scores[round_number - 1]},
        )


class _FailIfCalledProvider:
    def __init__(self) -> None:
        self.calls = 0

    def next_candidate(self, *, prompt_payload: dict) -> CandidatePackageRef:
        _ = prompt_payload
        self.calls += 1
        raise AssertionError("provider must not run when target round count is complete")


class _BlockingScoreProvider(_ScoreProvider):
    def __init__(self, root: Path, scores: tuple[float, ...]) -> None:
        super().__init__(root, scores)
        self.started = Event()
        self.release = Event()

    def next_candidate(self, *, prompt_payload: dict) -> CandidatePackageRef:
        self.started.set()
        if not self.release.wait(timeout=10):
            raise RuntimeError("test provider release timed out")
        return super().next_candidate(prompt_payload=prompt_payload)


def _manifest(task_id: str) -> BaselineManifest:
    return BaselineManifest(
        baseline_id="official-fake-baseline",
        task_id=task_id,
        checkpoint=CheckpointRef(
            uri="baseline://fake/policy.pt",
            checksum="sha256:" + "1" * 64,
            role="warm_start",
            sealed=True,
        ),
        provenance=CheckpointProvenance(
            source_run_id="official-run",
            source_task_id=task_id,
            created_by="runner",
        ),
        compatibility=BaselineCompatibilitySpec(
            backend_id="fake_backend",
            robot="fake_robot",
            observation_schema_hash="fake_obs",
            action_schema_hash="fake_action",
            policy_architecture="fake_policy",
        ),
    )


def _catalog_item(payload: dict, logical_id: str) -> dict:
    return next(
        item
        for item in payload["source_catalog"]["items"]
        if item["logical_id"] == logical_id
    )


class SessionGenesisLiveLoopTests(unittest.TestCase):
    def test_legacy_genesis_is_readable_but_cannot_resume_formal_runtime(self) -> None:
        task = FakeTaskPlugin()
        backend = FakeBackendAdapter()
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            current = prepare_session_genesis(
                output_dir=output_dir,
                session_id="legacy-genesis-session",
                task_plugin=task,
                backend_adapter=backend,
                runner=FakeRunner(),
                policy_initialization=PolicyInitialization.random(),
                genesis_identity_snapshot={"fixture": "legacy-read"},
                bootstrap_initial_best_score=0.0,
            )
            self.assertEqual(
                current.artifact.schema_version,
                "true_rl_agent.session_genesis.v3",
            )
            self.assertIsNone(
                current.artifact.backend_execution_environment
            )
            payload = current.artifact.to_dict()
            payload["schema_version"] = LEGACY_SESSION_GENESIS_SCHEMA_VERSION
            payload.pop("backend_execution_environment")
            payload["session_genesis_hash"] = canonical_json_hash(
                {
                    key: value
                    for key, value in payload.items()
                    if key != "session_genesis_hash"
                }
            )
            genesis_path = output_dir / SESSION_GENESIS_FILENAME
            genesis_path.write_bytes(
                canonical_json_bytes(payload, trailing_newline=True)
            )

            historical = SessionGenesisArtifact.from_mapping(payload)
            self.assertEqual(
                historical.schema_version,
                LEGACY_SESSION_GENESIS_SCHEMA_VERSION,
            )
            self.assertIsNone(historical.backend_execution_environment)
            with self.assertRaisesRegex(ValueError, "historical-only"):
                prepare_session_genesis(
                    output_dir=output_dir,
                    session_id="legacy-genesis-session",
                    task_plugin=task,
                    backend_adapter=backend,
                    runner=FakeRunner(),
                    policy_initialization=PolicyInitialization.random(),
                    genesis_identity_snapshot={"fixture": "legacy-read"},
                    bootstrap_initial_best_score=0.0,
                    require_existing=True,
                )

    def test_live_start_rejects_genesis_binding_drift_before_work(self) -> None:
        task = FakeTaskPlugin()
        genesis_backend = _SessionBoundBackend()
        runner = _CountingRunner()
        provider = _FailIfCalledProvider()

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            genesis = prepare_session_genesis(
                output_dir=output_dir,
                session_id="binding-preflight",
                task_plugin=task,
                backend_adapter=genesis_backend,
                runner=runner,
                policy_initialization=PolicyInitialization.random(),
                genesis_identity_snapshot={"fixture": "binding_preflight_v1"},
                bootstrap_initial_best_score=0.10,
            )

            with self.assertRaisesRegex(
                ValueError,
                "evaluation binding mismatch",
            ):
                run_live_loop(
                    output_dir=output_dir,
                    rounds=1,
                    session_id="binding-preflight",
                    task_plugin=task,
                    backend_adapter=_DriftedSessionBoundBackend(),
                    runner=runner,
                    candidate_admission_policy=FakeVerifier(),
                    provider=provider,
                    policy_initialization=PolicyInitialization.random(),
                    session_genesis=genesis,
                )

        self.assertEqual(provider.calls, 0)
        self.assertEqual(runner.train_specs, [])

    def test_existing_v4_v2_evidence_history_resumes_with_current(self) -> None:
        task = FakeTaskPlugin()
        backend = FakeBackendAdapter()
        runner = FakeRunner()

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            genesis = prepare_session_genesis(
                output_dir=output_dir,
                session_id="push-v4-resume",
                task_plugin=task,
                backend_adapter=backend,
                runner=runner,
                policy_initialization=PolicyInitialization.random(),
                genesis_identity_snapshot={"fixture": "push_v4_resume"},
                bootstrap_initial_best_score=0.10,
            )
            run_live_loop(
                output_dir=output_dir,
                rounds=1,
                session_id="push-v4-resume",
                task_plugin=task,
                backend_adapter=backend,
                runner=runner,
                candidate_admission_policy=FakeVerifier(),
                provider=_ScoreProvider(output_dir / "candidates", (0.25,)),
                policy_initialization=PolicyInitialization.random(),
                session_genesis=genesis,
            )

            ledger_path = output_dir / "ledger.jsonl"
            historical = load_ledger(ledger_path)[0]
            evidence_ref = next(
                item
                for item in historical["private_artifacts"]
                if item.get("kind") == "provider_authoring_evidence"
            )
            evidence = json.loads(
                Path(evidence_ref["uri"]).read_text(encoding="utf-8")
            )
            evidence["schema_version"] = AUTHORING_EVIDENCE_SCHEMA_V2
            evidence.pop("budget_contract")
            evidence.pop("terminal_outcome")
            evidence.pop("round_authoring_anchor")
            evidence_path = Path(evidence_ref["uri"]).with_name(
                "authoring_evidence_v2.json"
            )
            evidence_bytes = canonical_json_bytes(
                evidence,
                trailing_newline=True,
            )
            evidence_path.write_bytes(evidence_bytes)
            evidence_ref["uri"] = str(evidence_path)
            evidence_ref["checksum"] = (
                "sha256:" + hashlib.sha256(evidence_bytes).hexdigest()
            )
            evidence_ref["size_bytes"] = len(evidence_bytes)
            historical["schema_version"] = LEDGER_RECORD_SCHEMA_V4
            historical.pop("ledger_epoch_hash")
            ledger_path.write_bytes(
                canonical_json_bytes(historical, trailing_newline=True)
            )

            summary = run_live_loop(
                output_dir=output_dir,
                rounds=2,
                session_id="push-v4-resume",
                task_plugin=task,
                backend_adapter=backend,
                runner=runner,
                candidate_admission_policy=FakeVerifier(),
                provider=_ScoreProvider(
                    output_dir / "resume-candidates",
                    (0.0, 0.35),
                ),
                policy_initialization=PolicyInitialization.random(),
                session_genesis=genesis,
            )
            records = load_ledger(ledger_path)

        self.assertEqual(summary["resumed_from_round_count"], 1)
        self.assertEqual(
            [record["schema_version"] for record in records],
            [LEDGER_RECORD_SCHEMA_V4, LEDGER_RECORD_SCHEMA_VERSION],
        )

    def test_separate_ledger_epoch_preserves_legacy_genesis_root(self) -> None:
        task = FakeTaskPlugin()
        backend = FakeBackendAdapter()
        runner = FakeRunner()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            genesis_root = root / "formal-v2"
            epoch_root = root / "verified-v5"
            genesis = prepare_session_genesis(
                output_dir=genesis_root,
                session_id="ledger-epoch-session",
                task_plugin=task,
                backend_adapter=backend,
                runner=runner,
                policy_initialization=PolicyInitialization.random(),
                genesis_identity_snapshot={"fixture": "ledger_epoch_v1"},
                bootstrap_initial_best_score=0.10,
            )
            genesis_path = genesis_root / SESSION_GENESIS_FILENAME
            genesis_bytes = genesis_path.read_bytes()
            legacy_ledger = genesis_root / "ledger.jsonl"
            legacy_bytes = _denied_predecessor_bytes(
                genesis.session_genesis_hash
            )
            legacy_ledger.write_bytes(legacy_bytes)

            summary = run_live_loop(
                output_dir=epoch_root,
                rounds=2,
                session_id="ledger-epoch-session",
                task_plugin=task,
                backend_adapter=backend,
                runner=runner,
                candidate_admission_policy=FakeVerifier(),
                provider=_ScoreProvider(
                    epoch_root / "candidates",
                    (0.25, 0.35),
                ),
                policy_initialization=PolicyInitialization.random(),
                session_genesis=genesis,
                session_genesis_dir=genesis_root,
                ledger_epoch_id="phase-f-v5-rerun",
            )
            epoch_manifest_bytes = (
                epoch_root / LEDGER_EPOCH_FILENAME
            ).read_bytes()
            resumed = run_live_loop(
                output_dir=epoch_root,
                rounds=2,
                session_id="ledger-epoch-session",
                task_plugin=task,
                backend_adapter=backend,
                runner=runner,
                candidate_admission_policy=FakeVerifier(),
                provider=_FailIfCalledProvider(),
                policy_initialization=PolicyInitialization.random(),
                session_genesis=genesis,
                session_genesis_dir=genesis_root,
                ledger_epoch_id="phase-f-v5-rerun",
            )

            new_ledger = load_ledger(epoch_root / "ledger.jsonl")
            final_genesis_bytes = genesis_path.read_bytes()
            final_legacy_bytes = legacy_ledger.read_bytes()
            final_epoch_manifest_bytes = (
                epoch_root / LEDGER_EPOCH_FILENAME
            ).read_bytes()

        self.assertEqual(genesis_bytes, final_genesis_bytes)
        self.assertEqual(legacy_bytes, final_legacy_bytes)
        self.assertEqual(epoch_manifest_bytes, final_epoch_manifest_bytes)
        self.assertEqual(len(new_ledger), 2)
        self.assertTrue(
            all(
                record["schema_version"] == LEDGER_RECORD_SCHEMA_VERSION
                and record["ledger_epoch_hash"]
                == summary["ledger_epoch"]["manifest_hash"]
                for record in new_ledger
            )
        )
        self.assertEqual(summary["session_genesis_dir"], str(genesis_root))
        self.assertEqual(
            summary["ledger_epoch"]["epoch_id"],
            "phase-f-v5-rerun",
        )
        self.assertEqual(
            summary["ledger_epoch"]["predecessor_ledger_status"],
            "bound",
        )
        self.assertEqual(resumed["rounds_executed_this_invocation"], 0)

    def test_ledger_epoch_rejects_predecessor_drift_before_provider(self) -> None:
        task = FakeTaskPlugin()
        backend = FakeBackendAdapter()
        runner = FakeRunner()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            genesis_root = root / "formal-v2"
            epoch_root = root / "verified-v5"
            genesis = prepare_session_genesis(
                output_dir=genesis_root,
                session_id="ledger-epoch-drift",
                task_plugin=task,
                backend_adapter=backend,
                runner=runner,
                policy_initialization=PolicyInitialization.random(),
                genesis_identity_snapshot={"fixture": "ledger_epoch_v1"},
                bootstrap_initial_best_score=0.10,
            )
            predecessor = genesis_root / "ledger.jsonl"
            predecessor.write_bytes(
                _denied_predecessor_bytes(genesis.session_genesis_hash)
            )
            run_live_loop(
                output_dir=epoch_root,
                rounds=1,
                session_id="ledger-epoch-drift",
                task_plugin=task,
                backend_adapter=backend,
                runner=runner,
                candidate_admission_policy=FakeVerifier(),
                provider=_ScoreProvider(epoch_root / "candidates", (0.25,)),
                policy_initialization=PolicyInitialization.random(),
                session_genesis=genesis,
                session_genesis_dir=genesis_root,
                ledger_epoch_id="phase-f-v5-rerun",
            )
            predecessor.write_bytes(predecessor.read_bytes() + b"{}\n")
            provider = _FailIfCalledProvider()
            with self.assertRaisesRegex(
                ValueError,
                "ledger epoch differs|predecessor ledger|invalid ledger schema",
            ):
                run_live_loop(
                    output_dir=epoch_root,
                    rounds=2,
                    session_id="ledger-epoch-drift",
                    task_plugin=task,
                    backend_adapter=backend,
                    runner=runner,
                    candidate_admission_policy=FakeVerifier(),
                    provider=provider,
                    policy_initialization=PolicyInitialization.random(),
                    session_genesis=genesis,
                    session_genesis_dir=genesis_root,
                    ledger_epoch_id="phase-f-v5-rerun",
                )

        self.assertEqual(provider.calls, 0)

    def test_ledger_epoch_rejects_predecessor_that_advanced_genesis(self) -> None:
        task = FakeTaskPlugin()
        backend = FakeBackendAdapter()
        runner = FakeRunner()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            genesis_root = root / "formal-v2"
            genesis = prepare_session_genesis(
                output_dir=genesis_root,
                session_id="accepted-predecessor",
                task_plugin=task,
                backend_adapter=backend,
                runner=runner,
                policy_initialization=PolicyInitialization.random(),
                genesis_identity_snapshot={"fixture": "ledger_epoch_v1"},
                bootstrap_initial_best_score=0.10,
            )
            predecessor_record = json.loads(
                _denied_predecessor_bytes(
                    genesis.session_genesis_hash
                )
            )
            predecessor_record["decision"]["accepted"] = True
            (genesis_root / "ledger.jsonl").write_bytes(
                canonical_json_bytes(
                    predecessor_record,
                    trailing_newline=True,
                )
            )

            with self.assertRaisesRegex(
                ValueError,
                "advanced the Genesis current-best",
            ):
                prepare_ledger_epoch(
                    output_dir=root / "verified-v5",
                    genesis_dir=genesis_root,
                    epoch_id="phase-f-v5-rerun",
                    session_id="accepted-predecessor",
                    genesis=genesis,
                )

            predecessor_record["decision"]["accepted"] = False
            predecessor_record["session_genesis_hash"] = "0" * 64
            (genesis_root / "ledger.jsonl").write_bytes(
                canonical_json_bytes(
                    predecessor_record,
                    trailing_newline=True,
                )
            )
            with self.assertRaisesRegex(
                ValueError,
                "different session Genesis",
            ):
                prepare_ledger_epoch(
                    output_dir=root / "foreign-predecessor-v5",
                    genesis_dir=genesis_root,
                    epoch_id="phase-f-v5-foreign",
                    session_id="accepted-predecessor",
                    genesis=genesis,
                )

    def test_ledger_epoch_rejects_foreign_ledger_before_provider(self) -> None:
        task = FakeTaskPlugin()
        backend = FakeBackendAdapter()
        runner = FakeRunner()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            genesis_root = root / "genesis"
            epoch_a_root = root / "epoch-a"
            epoch_b_root = root / "epoch-b"
            genesis = prepare_session_genesis(
                output_dir=genesis_root,
                session_id="foreign-ledger",
                task_plugin=task,
                backend_adapter=backend,
                runner=runner,
                policy_initialization=PolicyInitialization.random(),
                genesis_identity_snapshot={"fixture": "ledger_epoch_v1"},
                bootstrap_initial_best_score=0.10,
            )
            summary_a = run_live_loop(
                output_dir=epoch_a_root,
                rounds=1,
                session_id="foreign-ledger",
                task_plugin=task,
                backend_adapter=backend,
                runner=runner,
                candidate_admission_policy=FakeVerifier(),
                provider=_ScoreProvider(epoch_a_root / "candidates", (0.25,)),
                policy_initialization=PolicyInitialization.random(),
                session_genesis=genesis,
                session_genesis_dir=genesis_root,
                ledger_epoch_id="epoch-a",
            )
            epoch_b = prepare_ledger_epoch(
                output_dir=epoch_b_root,
                genesis_dir=genesis_root,
                epoch_id="epoch-b",
                session_id="foreign-ledger",
                genesis=genesis,
            )
            assert epoch_b is not None
            self.assertNotEqual(
                summary_a["ledger_epoch"]["manifest_hash"],
                epoch_b.manifest_hash,
            )
            (epoch_b_root / "ledger.jsonl").write_bytes(
                (epoch_a_root / "ledger.jsonl").read_bytes()
            )

            provider = _FailIfCalledProvider()
            with self.assertRaisesRegex(
                ValueError,
                "ledger_epoch_hash_mismatch",
            ):
                run_live_loop(
                    output_dir=epoch_b_root,
                    rounds=1,
                    session_id="foreign-ledger",
                    task_plugin=task,
                    backend_adapter=backend,
                    runner=runner,
                    candidate_admission_policy=FakeVerifier(),
                    provider=provider,
                    policy_initialization=PolicyInitialization.random(),
                    session_genesis=genesis,
                    session_genesis_dir=genesis_root,
                    ledger_epoch_id="epoch-b",
                )

        self.assertEqual(provider.calls, 0)

    def test_live_session_lock_rejects_second_loop_before_provider(self) -> None:
        task = FakeTaskPlugin()
        backend = FakeBackendAdapter()
        runner = FakeRunner()

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            genesis = prepare_session_genesis(
                output_dir=output_dir,
                session_id="exclusive-live-session",
                task_plugin=task,
                backend_adapter=backend,
                runner=runner,
                policy_initialization=PolicyInitialization.random(),
                genesis_identity_snapshot={"fixture": "live_lock_v1"},
                bootstrap_initial_best_score=0.10,
            )
            first_provider = _BlockingScoreProvider(
                output_dir / "candidates",
                (0.25,),
            )
            second_provider = _FailIfCalledProvider()
            common = {
                "output_dir": output_dir,
                "rounds": 1,
                "session_id": "exclusive-live-session",
                "task_plugin": task,
                "backend_adapter": backend,
                "runner": runner,
                "candidate_admission_policy": FakeVerifier(),
                "policy_initialization": PolicyInitialization.random(),
                "session_genesis": genesis,
            }
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    run_live_loop,
                    provider=first_provider,
                    **common,
                )
                self.assertTrue(first_provider.started.wait(timeout=5))
                try:
                    with self.assertRaises(LiveSessionInProgressError):
                        run_live_loop(provider=second_provider, **common)
                finally:
                    first_provider.release.set()
                first_summary = future.result(timeout=10)
            ledger = load_ledger(output_dir / "ledger.jsonl")

        self.assertEqual(second_provider.calls, 0)
        self.assertEqual(first_summary["rounds_completed"], 1)
        self.assertEqual(len(ledger), 1)

    def test_baseline_live_resume_replay_and_prompt_authority_transition(self) -> None:
        task = _BaselineRequiredTask()
        backend = _SessionBoundBackend()
        runner = _CountingRunner()
        manifest = _manifest(task.task_spec.task_id)
        initialization = manifest.to_policy_initialization()
        genesis_identity_snapshot = {"fixture": "warm_start_session_v1"}
        scores = (0.28, 0.31, 0.30)

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            genesis = prepare_session_genesis(
                output_dir=output_dir,
                session_id="baseline-live",
                task_plugin=task,
                backend_adapter=backend,
                runner=runner,
                policy_initialization=initialization,
                genesis_identity_snapshot=genesis_identity_snapshot,
                baseline_manifest=manifest,
            )
            genesis_bytes = (output_dir / SESSION_GENESIS_FILENAME).read_bytes()
            first_provider = _ScoreProvider(output_dir / "candidates", scores)
            first_summary = run_live_loop(
                output_dir=output_dir,
                rounds=1,
                session_id="baseline-live",
                task_plugin=task,
                backend_adapter=backend,
                runner=runner,
                candidate_admission_policy=FakeVerifier(),
                provider=first_provider,
                policy_initialization=initialization,
                session_genesis=genesis,
            )

            resumed_genesis = prepare_session_genesis(
                output_dir=output_dir,
                session_id="baseline-live",
                task_plugin=task,
                backend_adapter=backend,
                runner=runner,
                policy_initialization=initialization,
                genesis_identity_snapshot=genesis_identity_snapshot,
                baseline_manifest=manifest,
            )
            no_op_provider = _FailIfCalledProvider()
            no_op_summary = run_live_loop(
                output_dir=output_dir,
                rounds=1,
                session_id="baseline-live",
                task_plugin=task,
                backend_adapter=backend,
                runner=runner,
                candidate_admission_policy=FakeVerifier(),
                provider=no_op_provider,
                policy_initialization=initialization,
                session_genesis=resumed_genesis,
            )
            no_op_summary_file = json.loads(
                (output_dir / "live_summary.json").read_text(encoding="utf-8")
            )
            resume_provider = _ScoreProvider(output_dir / "candidates", scores)
            final_summary = run_live_loop(
                output_dir=output_dir,
                rounds=3,
                session_id="baseline-live",
                task_plugin=task,
                backend_adapter=backend,
                runner=runner,
                candidate_admission_policy=FakeVerifier(),
                provider=resume_provider,
                policy_initialization=initialization,
                session_genesis=resumed_genesis,
            )
            ledger = load_ledger(output_dir / "ledger.jsonl")
            replay = replay_acceptance_ledger(
                ledger,
                initial_current_best=resumed_genesis.initial_current_best,
                expected_session_genesis_hash=resumed_genesis.session_genesis_hash,
                acceptance_protocol=task.task_spec.acceptance_protocol,
                expected_authoring_mode=PROVIDER_AUTHORING_MODE,
            )
            final_genesis_bytes = (
                output_dir / SESSION_GENESIS_FILENAME
            ).read_bytes()

        self.assertEqual(runner.baseline_eval_calls, 1)
        self.assertTrue(backend.resolve_allow_create[0])
        self.assertTrue(all(not value for value in backend.resolve_allow_create[1:]))
        self.assertEqual(genesis.evaluation_binding.seed_cohort_hash, "9" * 64)
        self.assertEqual(genesis_bytes, final_genesis_bytes)
        self.assertEqual(first_summary["final_best_eval_stats"]["anchor_kind"], "baseline")
        self.assertEqual(first_summary["decision_outcome_counts"], {"reject": 1})
        self.assertEqual(no_op_provider.calls, 0)
        self.assertEqual(no_op_summary["rounds_executed_this_invocation"], 0)
        self.assertEqual(no_op_summary["provider_attempt_count"], 1)
        self.assertEqual(
            no_op_summary[
                "provider_attempt_count_for_rounds_closed_this_invocation"
            ],
            0,
        )
        self.assertEqual(
            no_op_summary_file["provider_attempt_count"],
            no_op_summary["provider_attempt_count"],
        )
        self.assertEqual(len(no_op_summary["authoring_evidence_paths"]), 1)
        self.assertEqual(final_summary["resumed_from_round_count"], 1)
        self.assertEqual(final_summary["rounds_executed_this_invocation"], 2)
        self.assertEqual(final_summary["provider_attempt_count"], 3)
        self.assertEqual(
            final_summary[
                "provider_attempt_count_for_rounds_closed_this_invocation"
            ],
            2,
        )
        self.assertEqual(len(final_summary["authoring_evidence_paths"]), 3)
        self.assertEqual(
            [record["round_id"] for record in ledger],
            [
                "baseline-live-round-001",
                "baseline-live-round-002",
                "baseline-live-round-003",
            ],
        )
        self.assertEqual(
            [record["decision"]["reason"] for record in ledger],
            [
                "noise_aware_delta_not_met",
                "noise_aware_delta",
                "noise_aware_delta_not_met",
            ],
        )
        self.assertEqual(replay.current_best.anchor_kind, "candidate")
        self.assertEqual(replay.current_best.anchor_id, "candidate-002")
        self.assertEqual(
            replay.current_best.to_public_dict(),
            final_summary["final_best_eval_stats"],
        )
        self.assertEqual(len(runner.train_specs), 3)
        for train_spec in runner.train_specs:
            checkpoint = train_spec.training_start_checkpoint()
            self.assertIsNotNone(checkpoint)
            self.assertEqual(checkpoint.checksum, manifest.checkpoint.checksum)

        initial_payload = first_provider.payloads[0]
        self.assertEqual(initial_payload["candidate_lineage"]["status"], "baseline_current_best")
        self.assertEqual(
            _catalog_item(
                initial_payload,
                "current_best/training_code/components.py",
            )["status"],
            "baseline_anchor",
        )
        before_accept_payload = resume_provider.payloads[0]
        self.assertEqual(
            before_accept_payload["candidate_lineage"]["status"],
            "baseline_current_best",
        )
        self.assertEqual(
            before_accept_payload["candidate_lineage"]["best_observed_experiment"][
                "candidate_id"
            ],
            "candidate-001",
        )
        after_accept_payload = resume_provider.payloads[1]
        self.assertEqual(
            after_accept_payload["candidate_lineage"]["status"],
            "accepted_current_best",
        )
        self.assertEqual(
            after_accept_payload["candidate_lineage"]["current_best"]["candidate_id"],
            "candidate-002",
        )
        self.assertTrue(
            any(
                payload["logical_id"]
                == "current_best/training_code/components.py"
                for payload in after_accept_payload["source_payloads"]
            )
        )

    def test_resume_rejects_ledger_runtime_identity_before_provider(self) -> None:
        task = _BaselineRequiredTask()
        backend = FakeBackendAdapter()
        runner = _CountingRunner()
        manifest = _manifest(task.task_spec.task_id)
        initialization = manifest.to_policy_initialization()
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            genesis = prepare_session_genesis(
                output_dir=output_dir,
                session_id="identity-drift",
                task_plugin=task,
                backend_adapter=backend,
                runner=runner,
                policy_initialization=initialization,
                genesis_identity_snapshot={"fixture": "identity_drift_v1"},
                baseline_manifest=manifest,
            )
            run_live_loop(
                output_dir=output_dir,
                rounds=1,
                session_id="identity-drift",
                task_plugin=task,
                backend_adapter=backend,
                runner=runner,
                candidate_admission_policy=FakeVerifier(),
                provider=_ScoreProvider(output_dir / "candidates", (0.28,)),
                policy_initialization=initialization,
                session_genesis=genesis,
            )
            ledger_path = output_dir / "ledger.jsonl"
            record = json.loads(ledger_path.read_text(encoding="utf-8"))
            record["task"]["task_id"] = "forged-task"
            ledger_path.write_text(
                json.dumps(record, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            provider = _FailIfCalledProvider()

            with self.assertRaisesRegex(ValueError, "task identity mismatch"):
                run_live_loop(
                    output_dir=output_dir,
                    rounds=2,
                    session_id="identity-drift",
                    task_plugin=task,
                    backend_adapter=backend,
                    runner=runner,
                    candidate_admission_policy=FakeVerifier(),
                    provider=provider,
                    policy_initialization=initialization,
                    session_genesis=genesis,
                )

            self.assertEqual(provider.calls, 0)


if __name__ == "__main__":
    unittest.main()
