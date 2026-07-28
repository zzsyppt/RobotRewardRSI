from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

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
from true_rl_agent.agent_core.ledger import load_ledger  # noqa: E402
from true_rl_agent.agent_core.live_loop import run_live_loop  # noqa: E402
from true_rl_agent.agent_core.session_genesis import (  # noqa: E402
    SESSION_GENESIS_FILENAME,
    prepare_session_genesis,
)
from true_rl_agent.candidates.schema import (  # noqa: E402
    CandidateChangeSet,
    CandidatePackageRef,
)
from true_rl_agent.evaluation.acceptance_replay import (  # noqa: E402
    replay_acceptance_ledger,
)
from true_rl_agent.evaluation.binding import EvaluationBindingV1  # noqa: E402
from true_rl_agent.policy_baselines import (  # noqa: E402
    BaselineCompatibilitySpec,
    BaselineManifest,
    CheckpointProvenance,
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


class _ScoreProvider:
    def __init__(self, root: Path, scores: tuple[float, ...]) -> None:
        self.root = root
        self.scores = scores
        self.payloads: list[dict] = []

    def next_candidate(self, *, prompt_payload: dict) -> CandidatePackageRef:
        self.payloads.append(prompt_payload)
        round_number = int(prompt_payload["public_context"]["previous_rounds"]) + 1
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
            f"    return inputs.base_height + {round_number}.0 * inputs.roll\n\n"
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
    def test_baseline_live_resume_replay_and_prompt_authority_transition(self) -> None:
        task = _BaselineRequiredTask()
        backend = _SessionBoundBackend()
        runner = _CountingRunner()
        manifest = _manifest(task.task_spec.task_id)
        initialization = manifest.to_policy_initialization()
        runtime_snapshot = {"fixture": "warm_start_session_v1"}
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
                runtime_snapshot=runtime_snapshot,
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
                verifier=FakeVerifier(),
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
                runtime_snapshot=runtime_snapshot,
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
                verifier=FakeVerifier(),
                provider=no_op_provider,
                policy_initialization=initialization,
                session_genesis=resumed_genesis,
            )
            resume_provider = _ScoreProvider(output_dir / "candidates", scores)
            final_summary = run_live_loop(
                output_dir=output_dir,
                rounds=3,
                session_id="baseline-live",
                task_plugin=task,
                backend_adapter=backend,
                runner=runner,
                verifier=FakeVerifier(),
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
        self.assertEqual(final_summary["resumed_from_round_count"], 1)
        self.assertEqual(final_summary["rounds_executed_this_invocation"], 2)
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
                runtime_snapshot={"fixture": "identity_drift_v1"},
                baseline_manifest=manifest,
            )
            run_live_loop(
                output_dir=output_dir,
                rounds=1,
                session_id="identity-drift",
                task_plugin=task,
                backend_adapter=backend,
                runner=runner,
                verifier=FakeVerifier(),
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
                    verifier=FakeVerifier(),
                    provider=provider,
                    policy_initialization=initialization,
                    session_genesis=genesis,
                )

            self.assertEqual(provider.calls, 0)


if __name__ == "__main__":
    unittest.main()
