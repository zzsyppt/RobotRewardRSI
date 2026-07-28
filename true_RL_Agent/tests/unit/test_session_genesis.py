from __future__ import annotations

import json
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Barrier, Lock

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tests.fakes.fake_components import FakeBackendAdapter, FakeRunner, FakeTaskPlugin
from true_rl_agent.agent_core.session_genesis import (
    BASELINE_EVAL_ARTIFACT_FILENAME,
    SESSION_GENESIS_FILENAME,
    prepare_session_genesis,
)
from true_rl_agent.evaluation.binding import EvaluationBindingV1, canonical_json_hash
from true_rl_agent.json_contract import canonical_json_bytes
from true_rl_agent.memory import PublicContextBuilder
from true_rl_agent.policy import PolicyInitialization
from true_rl_agent.policy_baselines import (
    BaselineCompatibilitySpec,
    BaselineManifest,
    CheckpointProvenance,
)
from true_rl_agent.runner.specs import CheckpointRef
from true_rl_agent.tasks.base import TaskSpec


class BaselineRequiredFakeTask(FakeTaskPlugin):
    @property
    def task_spec(self) -> TaskSpec:
        return replace(
            super().task_spec,
            supported_policy_initializations=frozenset({"warm_start"}),
            genesis_anchor_policy={"warm_start": "policy_baseline"},
        )


class CountingFakeRunner(FakeRunner):
    def __init__(self) -> None:
        self.baseline_eval_calls = 0
        self.train_specs = []

    def run_baseline_eval(self, eval_spec):
        self.baseline_eval_calls += 1
        return super().run_baseline_eval(eval_spec)

    def run_train(self, train_spec):
        self.train_specs.append(train_spec)
        return super().run_train(train_spec)


class BarrierBaselineRunner(CountingFakeRunner):
    def __init__(self, parties: int) -> None:
        super().__init__()
        self._barrier = Barrier(parties)
        self._lock = Lock()

    def run_baseline_eval(self, eval_spec):
        with self._lock:
            self.baseline_eval_calls += 1
        result = FakeRunner.run_baseline_eval(self, eval_spec)
        self._barrier.wait(timeout=5)
        return result


class DriftedBindingBackend(FakeBackendAdapter):
    def evaluation_binding_for(self, task_spec):
        binding = super().evaluation_binding_for(task_spec)
        return EvaluationBindingV1(
            eval_protocol_hash=binding.eval_protocol_hash,
            acceptance_protocol_hash=binding.acceptance_protocol_hash,
            task_protocol_hash=binding.task_protocol_hash,
            seed_cohort_hash="a" * 64,
        )


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


class SessionGenesisTests(unittest.TestCase):
    def test_baseline_genesis_is_full_current_best_and_resume_does_not_rerun(self) -> None:
        task = BaselineRequiredFakeTask()
        manifest = _manifest(task.task_spec.task_id)
        runner = CountingFakeRunner()
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            first = prepare_session_genesis(
                output_dir=output_dir,
                session_id="session-001",
                task_plugin=task,
                backend_adapter=FakeBackendAdapter(),
                runner=runner,
                policy_initialization=manifest.to_policy_initialization(),
                runtime_snapshot={"fixed": True},
                baseline_manifest=manifest,
            )
            first_bytes = (output_dir / SESSION_GENESIS_FILENAME).read_bytes()
            second = prepare_session_genesis(
                output_dir=output_dir,
                session_id="session-001",
                task_plugin=task,
                backend_adapter=FakeBackendAdapter(),
                runner=runner,
                policy_initialization=manifest.to_policy_initialization(),
                runtime_snapshot={"fixed": True},
                baseline_manifest=manifest,
            )

            self.assertEqual(runner.baseline_eval_calls, 1)
            self.assertEqual(first.session_genesis_hash, second.session_genesis_hash)
            self.assertEqual(
                first_bytes,
                (output_dir / SESSION_GENESIS_FILENAME).read_bytes(),
            )
            current_best = second.initial_current_best
            self.assertEqual(current_best.mode, "accepted")
            self.assertEqual(current_best.anchor_kind, "baseline")
            self.assertEqual(current_best.anchor_id, manifest.baseline_id)
            self.assertEqual(current_best.checkpoint_hash, manifest.checkpoint.checksum)
            self.assertTrue(current_best.has_full_eval_stats())
            with self.assertRaises(TypeError):
                current_best.public_accept_stats["mean"] = 99.0
            with self.assertRaises(TypeError):
                second.artifact.anchor.public_result["public_score"] = 99.0
            with self.assertRaises(AttributeError):
                second._artifact = first.artifact

            prompt = PublicContextBuilder(
                acceptance_protocol=task.task_spec.acceptance_protocol,
                initial_current_best=second.initial_current_best,
                session_genesis_hash=second.session_genesis_hash,
            ).build_prompt_payload([])
            current_item = next(
                item
                for item in prompt["source_catalog"]["items"]
                if item["logical_id"]
                == "current_best/training_code/components.py"
            )
            self.assertEqual(current_item["role"], "baseline_policy_anchor")
            self.assertEqual(current_item["status"], "baseline_anchor")
            self.assertFalse(current_item["requestable"])
            self.assertEqual(
                prompt["candidate_lineage"]["status"],
                "baseline_current_best",
            )
            self.assertEqual(prompt["source_payloads"], [])

    def test_required_baseline_task_cannot_fall_back_to_bootstrap(self) -> None:
        task = BaselineRequiredFakeTask()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(
                ValueError,
                "does not define a Genesis anchor",
            ):
                prepare_session_genesis(
                    output_dir=Path(tmp),
                    session_id="session-required",
                    task_plugin=task,
                    backend_adapter=FakeBackendAdapter(),
                    runner=CountingFakeRunner(),
                    policy_initialization=PolicyInitialization.random(),
                    runtime_snapshot={"fixed": True},
                    bootstrap_initial_best_score=0.0,
                )

    def test_bootstrap_genesis_is_explicit_for_random_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            genesis = prepare_session_genesis(
                output_dir=Path(tmp),
                session_id="session-bootstrap",
                task_plugin=FakeTaskPlugin(),
                backend_adapter=FakeBackendAdapter(),
                runner=CountingFakeRunner(),
                policy_initialization=PolicyInitialization.random(),
                runtime_snapshot={"fixed": True},
                bootstrap_initial_best_score=0.1,
            )

            self.assertEqual(genesis.artifact.genesis_kind, "bootstrap")
            self.assertEqual(genesis.initial_current_best.mode, "bootstrap")
            self.assertEqual(genesis.initial_current_best.public_score, 0.1)

    def test_runtime_snapshot_drift_fails_without_overwrite(self) -> None:
        task = FakeTaskPlugin()
        runner = CountingFakeRunner()
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            prepare_session_genesis(
                output_dir=output_dir,
                session_id="session-drift",
                task_plugin=task,
                backend_adapter=FakeBackendAdapter(),
                runner=runner,
                policy_initialization=PolicyInitialization.random(),
                runtime_snapshot={"budget": 128},
                bootstrap_initial_best_score=0.1,
            )
            original = (output_dir / SESSION_GENESIS_FILENAME).read_bytes()

            with self.assertRaisesRegex(ValueError, "runtime snapshot drift"):
                prepare_session_genesis(
                    output_dir=output_dir,
                    session_id="session-drift",
                    task_plugin=task,
                    backend_adapter=FakeBackendAdapter(),
                    runner=runner,
                    policy_initialization=PolicyInitialization.random(),
                    runtime_snapshot={"budget": 129},
                    bootstrap_initial_best_score=0.1,
                )

            self.assertEqual(
                original,
                (output_dir / SESSION_GENESIS_FILENAME).read_bytes(),
            )

    def test_bootstrap_score_drift_fails_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            prepare_session_genesis(
                output_dir=output_dir,
                session_id="session-score-drift",
                task_plugin=FakeTaskPlugin(),
                backend_adapter=FakeBackendAdapter(),
                runner=CountingFakeRunner(),
                policy_initialization=PolicyInitialization.random(),
                runtime_snapshot={"fixed": True},
                bootstrap_initial_best_score=0.1,
            )
            original = (output_dir / SESSION_GENESIS_FILENAME).read_bytes()

            with self.assertRaisesRegex(ValueError, "initial score drift"):
                prepare_session_genesis(
                    output_dir=output_dir,
                    session_id="session-score-drift",
                    task_plugin=FakeTaskPlugin(),
                    backend_adapter=FakeBackendAdapter(),
                    runner=CountingFakeRunner(),
                    policy_initialization=PolicyInitialization.random(),
                    runtime_snapshot={"fixed": True},
                    bootstrap_initial_best_score=0.2,
                )

            self.assertEqual(
                original,
                (output_dir / SESSION_GENESIS_FILENAME).read_bytes(),
            )

    def test_baseline_and_bootstrap_score_are_mutually_exclusive(self) -> None:
        task = BaselineRequiredFakeTask()
        manifest = _manifest(task.task_spec.task_id)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "cannot also declare"):
                prepare_session_genesis(
                    output_dir=Path(tmp),
                    session_id="session-mixed-anchor",
                    task_plugin=task,
                    backend_adapter=FakeBackendAdapter(),
                    runner=CountingFakeRunner(),
                    policy_initialization=manifest.to_policy_initialization(),
                    runtime_snapshot={"fixed": True},
                    baseline_manifest=manifest,
                    bootstrap_initial_best_score=0.0,
                )

    def test_tampered_genesis_hash_is_rejected(self) -> None:
        task = FakeTaskPlugin()
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            prepare_session_genesis(
                output_dir=output_dir,
                session_id="session-tamper",
                task_plugin=task,
                backend_adapter=FakeBackendAdapter(),
                runner=CountingFakeRunner(),
                policy_initialization=PolicyInitialization.random(),
                runtime_snapshot={"fixed": True},
                bootstrap_initial_best_score=0.1,
            )
            path = output_dir / SESSION_GENESIS_FILENAME
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["anchor"]["initial_best_score"] = 99.0
            path.write_bytes(canonical_json_bytes(payload, trailing_newline=True))

            with self.assertRaisesRegex(ValueError, "session genesis hash mismatch"):
                prepare_session_genesis(
                    output_dir=output_dir,
                    session_id="session-tamper",
                    task_plugin=task,
                    backend_adapter=FakeBackendAdapter(),
                    runner=CountingFakeRunner(),
                    policy_initialization=PolicyInitialization.random(),
                    runtime_snapshot={"fixed": True},
                    bootstrap_initial_best_score=0.1,
                )

    def test_tampered_private_baseline_artifact_is_rejected_without_rebuild(self) -> None:
        task = BaselineRequiredFakeTask()
        manifest = _manifest(task.task_spec.task_id)
        runner = CountingFakeRunner()
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            prepare_session_genesis(
                output_dir=output_dir,
                session_id="session-private-tamper",
                task_plugin=task,
                backend_adapter=FakeBackendAdapter(),
                runner=runner,
                policy_initialization=manifest.to_policy_initialization(),
                runtime_snapshot={"fixed": True},
                baseline_manifest=manifest,
            )
            artifact_path = output_dir / BASELINE_EVAL_ARTIFACT_FILENAME
            artifact_path.write_text("{}\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "private artifact checksum mismatch"):
                prepare_session_genesis(
                    output_dir=output_dir,
                    session_id="session-private-tamper",
                    task_plugin=task,
                    backend_adapter=FakeBackendAdapter(),
                    runner=runner,
                    policy_initialization=manifest.to_policy_initialization(),
                    runtime_snapshot={"fixed": True},
                    baseline_manifest=manifest,
                )

            self.assertEqual(runner.baseline_eval_calls, 1)
            self.assertEqual(artifact_path.read_text(encoding="utf-8"), "{}\n")

    def test_private_baseline_artifact_cannot_be_relocated_with_rehashed_genesis(self) -> None:
        task = BaselineRequiredFakeTask()
        manifest = _manifest(task.task_spec.task_id)
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            prepare_session_genesis(
                output_dir=output_dir,
                session_id="session-private-path",
                task_plugin=task,
                backend_adapter=FakeBackendAdapter(),
                runner=CountingFakeRunner(),
                policy_initialization=manifest.to_policy_initialization(),
                runtime_snapshot={"fixed": True},
                baseline_manifest=manifest,
            )
            original_artifact = output_dir / BASELINE_EVAL_ARTIFACT_FILENAME
            relocated_artifact = output_dir / "relocated_baseline_eval.json"
            relocated_artifact.write_bytes(original_artifact.read_bytes())
            genesis_path = output_dir / SESSION_GENESIS_FILENAME
            payload = json.loads(genesis_path.read_text(encoding="utf-8"))
            payload["anchor"]["private_eval_artifact"]["uri"] = str(
                relocated_artifact.resolve()
            )
            payload["session_genesis_hash"] = canonical_json_hash(
                {
                    key: value
                    for key, value in payload.items()
                    if key != "session_genesis_hash"
                }
            )
            genesis_path.write_bytes(
                canonical_json_bytes(payload, trailing_newline=True)
            )

            with self.assertRaisesRegex(ValueError, "private artifact path drift"):
                prepare_session_genesis(
                    output_dir=output_dir,
                    session_id="session-private-path",
                    task_plugin=task,
                    backend_adapter=FakeBackendAdapter(),
                    runner=CountingFakeRunner(),
                    policy_initialization=manifest.to_policy_initialization(),
                    runtime_snapshot={"fixed": True},
                    baseline_manifest=manifest,
                )

    def test_concurrent_baseline_genesis_publish_has_one_canonical_winner(self) -> None:
        task = BaselineRequiredFakeTask()
        manifest = _manifest(task.task_spec.task_id)
        runner = BarrierBaselineRunner(parties=2)
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)

            def prepare():
                return prepare_session_genesis(
                    output_dir=output_dir,
                    session_id="session-concurrent",
                    task_plugin=task,
                    backend_adapter=FakeBackendAdapter(),
                    runner=runner,
                    policy_initialization=manifest.to_policy_initialization(),
                    runtime_snapshot={"fixed": True},
                    baseline_manifest=manifest,
                )

            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(prepare) for _ in range(2)]
                outcomes = []
                for future in futures:
                    try:
                        outcomes.append(future.result(timeout=10))
                    except FileExistsError as exc:
                        outcomes.append(exc)

            winners = [item for item in outcomes if not isinstance(item, Exception)]
            self.assertEqual(len(winners), 1)
            self.assertEqual(runner.baseline_eval_calls, 2)
            loaded = prepare_session_genesis(
                output_dir=output_dir,
                session_id="session-concurrent",
                task_plugin=task,
                backend_adapter=FakeBackendAdapter(),
                runner=runner,
                policy_initialization=manifest.to_policy_initialization(),
                runtime_snapshot={"fixed": True},
                baseline_manifest=manifest,
            )
            self.assertEqual(
                loaded.session_genesis_hash,
                winners[0].session_genesis_hash,
            )

    def test_resume_rejects_manifest_or_evaluation_binding_drift(self) -> None:
        task = BaselineRequiredFakeTask()
        manifest = _manifest(task.task_spec.task_id)
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            prepare_session_genesis(
                output_dir=output_dir,
                session_id="session-context-drift",
                task_plugin=task,
                backend_adapter=FakeBackendAdapter(),
                runner=CountingFakeRunner(),
                policy_initialization=manifest.to_policy_initialization(),
                runtime_snapshot={"fixed": True},
                baseline_manifest=manifest,
            )
            drifted_manifest = replace(
                manifest,
                provenance=CheckpointProvenance(
                    source_run_id="different-run",
                    source_task_id=task.task_spec.task_id,
                    created_by="runner",
                ),
                manifest_hash=None,
            )

            with self.assertRaisesRegex(
                ValueError,
                "warm-start policy initialization differs|baseline manifest hash drift",
            ):
                prepare_session_genesis(
                    output_dir=output_dir,
                    session_id="session-context-drift",
                    task_plugin=task,
                    backend_adapter=FakeBackendAdapter(),
                    runner=CountingFakeRunner(),
                    policy_initialization=drifted_manifest.to_policy_initialization(),
                    runtime_snapshot={"fixed": True},
                    baseline_manifest=drifted_manifest,
                )
            with self.assertRaisesRegex(ValueError, "evaluation binding drift"):
                prepare_session_genesis(
                    output_dir=output_dir,
                    session_id="session-context-drift",
                    task_plugin=task,
                    backend_adapter=DriftedBindingBackend(),
                    runner=CountingFakeRunner(),
                    policy_initialization=manifest.to_policy_initialization(),
                    runtime_snapshot={"fixed": True},
                    baseline_manifest=manifest,
                )

    def test_rehashed_trusted_seal_tamper_is_rejected_contextually(self) -> None:
        task = BaselineRequiredFakeTask()
        manifest = _manifest(task.task_spec.task_id)
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            prepare_session_genesis(
                output_dir=output_dir,
                session_id="session-seal-tamper",
                task_plugin=task,
                backend_adapter=FakeBackendAdapter(),
                runner=CountingFakeRunner(),
                policy_initialization=manifest.to_policy_initialization(),
                runtime_snapshot={"fixed": True},
                baseline_manifest=manifest,
            )
            genesis_path = output_dir / SESSION_GENESIS_FILENAME
            payload = json.loads(genesis_path.read_text(encoding="utf-8"))
            payload["anchor"]["trusted_eval_seal"]["seal_hash"] = "a" * 64
            payload["session_genesis_hash"] = canonical_json_hash(
                {
                    key: value
                    for key, value in payload.items()
                    if key != "session_genesis_hash"
                }
            )
            genesis_path.write_bytes(
                canonical_json_bytes(payload, trailing_newline=True)
            )

            with self.assertRaisesRegex(ValueError, "trusted eval seal mismatch"):
                prepare_session_genesis(
                    output_dir=output_dir,
                    session_id="session-seal-tamper",
                    task_plugin=task,
                    backend_adapter=FakeBackendAdapter(),
                    runner=CountingFakeRunner(),
                    policy_initialization=manifest.to_policy_initialization(),
                    runtime_snapshot={"fixed": True},
                    baseline_manifest=manifest,
                )

    def test_duplicate_genesis_key_changes_bytes_and_is_rejected(self) -> None:
        task = FakeTaskPlugin()
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            genesis = prepare_session_genesis(
                output_dir=output_dir,
                session_id="session-duplicate-key",
                task_plugin=task,
                backend_adapter=FakeBackendAdapter(),
                runner=CountingFakeRunner(),
                policy_initialization=PolicyInitialization.random(),
                runtime_snapshot={"fixed": True},
                bootstrap_initial_best_score=0.1,
            )
            path = output_dir / SESSION_GENESIS_FILENAME
            original = path.read_text(encoding="utf-8")
            path.write_text(
                original.replace("{", '{"task_id":"forged",', 1),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "bytes changed"):
                genesis.assert_file_unchanged(output_dir)
            with self.assertRaisesRegex(ValueError, "failed to read"):
                prepare_session_genesis(
                    output_dir=output_dir,
                    session_id="session-duplicate-key",
                    task_plugin=task,
                    backend_adapter=FakeBackendAdapter(),
                    runner=CountingFakeRunner(),
                    policy_initialization=PolicyInitialization.random(),
                    runtime_snapshot={"fixed": True},
                    bootstrap_initial_best_score=0.1,
                )


if __name__ == "__main__":
    unittest.main()
