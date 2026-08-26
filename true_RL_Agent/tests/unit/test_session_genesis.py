from __future__ import annotations

import json
import os
import subprocess
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

from tests.fakes.fake_components import FakeBackendAdapter, FakeRunner, FakeTaskPlugin
from true_rl_agent.agent_core.generation_contract import (
    GENERATION_SESSION_CONTRACT_V3_SCHEMA_VERSION,
    GenerationSessionContract,
    publish_generation_session_contract,
)
from true_rl_agent.agent_core.session_genesis import (
    BASELINE_EVAL_ARTIFACT_FILENAME,
    GENERATION_SESSION_GENESIS_SCHEMA_VERSION,
    LEGACY_SESSION_GENESIS_SCHEMA_VERSION,
    SESSION_GENESIS_CREATION_LEASE_FILENAME,
    SESSION_GENESIS_FILENAME,
    SessionGenesisArtifact,
    SessionGenesisCreationInProgressError,
    _acquire_session_genesis_creation_lease,
    _parse_pre_genesis_creation_lease,
    _pre_genesis_creation_identity,
    _publish_or_load_pre_genesis_recovery,
    _recover_pre_genesis_execution,
    _scan_pre_genesis_attempts,
    prepare_session_genesis,
    session_genesis_identity_hash,
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
from true_rl_agent.runner.job_attempts import (
    RECOVERED_ATTEMPT_RESULT_SCHEMA_VERSION,
    JobAttemptJournal,
    PreGenesisExecutionScope,
)
from true_rl_agent.runner.specs import CheckpointRef, GpuProcessBinding
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


class BlockingBaselineRunner(CountingFakeRunner):
    def __init__(self) -> None:
        super().__init__()
        self.eval_started = Event()
        self.allow_eval = Event()

    def run_baseline_eval(self, eval_spec):
        self.baseline_eval_calls += 1
        self.eval_started.set()
        if not self.allow_eval.wait(timeout=5):
            raise TimeoutError("test baseline evaluation was not released")
        result = FakeRunner.run_baseline_eval(self, eval_spec)
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


class GpuBoundBaselineBackend(FakeBackendAdapter):
    def __init__(self) -> None:
        self.last_baseline_spec = None

    def materialize_baseline_eval(
        self,
        *,
        gpu_process_binding=None,
        eval_attempt_scope=None,
        **kwargs,
    ):
        self.last_baseline_spec = replace(
            super().materialize_baseline_eval(**kwargs),
            gpu_process_binding=gpu_process_binding,
            attempt_scope=eval_attempt_scope,
        )
        return self.last_baseline_spec


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


def _pre_genesis_recovery_fixture(
    output_dir: Path,
    *,
    role: str,
) -> tuple[PreGenesisExecutionScope, dict, JobAttemptJournal]:
    if role == "reference_train":
        job_kind = "train"
        max_retries = 1
    elif role == "reference_eval":
        job_kind = "eval"
        max_retries = 0
    else:
        raise ValueError("test pre-Genesis role is invalid")
    scope = PreGenesisExecutionScope(
        journal_root=output_dir / "job_attempts",
        session_id="pre-genesis-recovery-session",
        genesis_identity_hash="a" * 64,
        session_contract_hash="b" * 64,
        lane_id="lane-a",
        invocation_segment_id="session-genesis",
        budget_contract_hash="c" * 64,
        attempt_budget_units=max_retries + 1,
        max_infrastructure_retries=max_retries,
        resource_binding_hash="d" * 64,
    )
    identity = _pre_genesis_creation_identity(
        session_id=scope.session_id,
        genesis_identity_hash=scope.genesis_identity_hash,
        session_contract_hash=scope.session_contract_hash,
        budget_contract_hash=scope.budget_contract_hash,
        resource_binding_hash=scope.resource_binding_hash,
        scopes={role: scope},
    )
    journal = JobAttemptJournal(
        scope=scope,
        logical_job_id=f"{role}-job",
        job_kind=job_kind,
        job_input_hash="e" * 64,
        environment_hash="f" * 64,
        execution_device="cuda:0",
        resource_binding_hash=scope.resource_binding_hash,
    )
    return scope, identity, journal


def _leave_unclean_pre_genesis_reservation(output_dir_text: str, role: str) -> None:
    output_dir = Path(output_dir_text)
    scope, identity, journal = _pre_genesis_recovery_fixture(
        output_dir,
        role=role,
    )
    del scope
    lease = _acquire_session_genesis_creation_lease(
        output_dir=output_dir,
        session_id=identity["session_id"],
        pre_genesis_identity=identity,
    )
    lease.activate()
    journal.reserve(0, previous_result=None)
    os._exit(0)


def _spawn_unclean_pre_genesis_reservation(output_dir: Path, role: str) -> None:
    code = (
        "import sys\n"
        "from tests.unit.test_session_genesis import "
        "_leave_unclean_pre_genesis_reservation\n"
        "_leave_unclean_pre_genesis_reservation(sys.argv[1], sys.argv[2])\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code, str(output_dir), role],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
        start_new_session=(os.name != "nt"),
    )
    if completed.returncode != 0:
        raise AssertionError(completed.stderr)


class SessionGenesisTests(unittest.TestCase):
    def test_generation_contract_is_create_once_bound_by_genesis_v4(self) -> None:
        task = FakeTaskPlugin()
        backend = FakeBackendAdapter()
        contract = GenerationSessionContract(
            session_id="generation-session",
            task_id=task.task_spec.task_id,
            backend_id=backend.backend_id,
            max_generations=2,
            max_trials=4,
            max_train_jobs=4,
            bound_contract_hashes={
                "authoring_policy": "1" * 64,
                "continuation_policy": "2" * 64,
                "evaluation_cost": "3" * 64,
                "evaluation_protocol": "4" * 64,
                "exploration_projector": "e" * 64,
                "final_confirmation_plan": "5" * 64,
                "provider_policy": "6" * 64,
                "resource_plan": "7" * 64,
                "retry_policy": "8" * 64,
                "selector": "9" * 64,
            },
        )
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            ref = publish_generation_session_contract(output_dir, contract)
            created = prepare_session_genesis(
                output_dir=output_dir,
                session_id="generation-session",
                task_plugin=task,
                backend_adapter=backend,
                runner=FakeRunner(),
                policy_initialization=PolicyInitialization.random(),
                genesis_identity_snapshot={"fixed": True},
                generation_session_contract_ref=ref,
                bootstrap_initial_best_score=0.10,
            )
            loaded = prepare_session_genesis(
                output_dir=output_dir,
                session_id="generation-session",
                task_plugin=task,
                backend_adapter=backend,
                runner=FakeRunner(),
                policy_initialization=PolicyInitialization.random(),
                genesis_identity_snapshot={"fixed": True},
                generation_session_contract_ref=ref,
                require_existing=True,
                bootstrap_initial_best_score=0.10,
            )

            self.assertEqual(
                created.artifact.schema_version,
                GENERATION_SESSION_GENESIS_SCHEMA_VERSION,
            )
            self.assertEqual(loaded.generation_session_contract_ref, ref)
            with self.assertRaisesRegex(
                ValueError,
                "contract v3 with session_cost",
            ):
                prepare_session_genesis(
                    output_dir=output_dir,
                    session_id="generation-session",
                    task_plugin=task,
                    backend_adapter=backend,
                    runner=FakeRunner(),
                    policy_initialization=PolicyInitialization.random(),
                    genesis_identity_snapshot={"fixed": True},
                    generation_session_contract_ref=ref,
                    gpu_process_binding=GpuProcessBinding(
                        host_execution_device="cuda:0",
                        process_execution_device="cuda:0",
                        resource_binding_hash="f" * 64,
                    ),
                    require_existing=True,
                    bootstrap_initial_best_score=0.10,
                )
            with self.assertRaisesRegex(ValueError, "capability/schema"):
                prepare_session_genesis(
                    output_dir=output_dir,
                    session_id="generation-session",
                    task_plugin=task,
                    backend_adapter=backend,
                    runner=FakeRunner(),
                    policy_initialization=PolicyInitialization.random(),
                    genesis_identity_snapshot={"fixed": True},
                    require_existing=True,
                    bootstrap_initial_best_score=0.10,
                )

    def test_pre_genesis_scope_cannot_drift_from_session_cost_contract(self) -> None:
        task = BaselineRequiredFakeTask()
        backend = FakeBackendAdapter()
        manifest = _manifest(task.task_spec.task_id)
        snapshot = {"fixed": True}
        contract = GenerationSessionContract(
            session_id="generation-session",
            task_id=task.task_spec.task_id,
            backend_id=backend.backend_id,
            max_generations=2,
            max_trials=4,
            max_train_jobs=4,
            bound_contract_hashes={
                "authoring_policy": "1" * 64,
                "continuation_policy": "2" * 64,
                "evaluation_cost": "3" * 64,
                "evaluation_protocol": "4" * 64,
                "exploration_projector": "e" * 64,
                "final_confirmation_plan": "5" * 64,
                "provider_policy": "6" * 64,
                "resource_plan": "7" * 64,
                "retry_policy": "8" * 64,
                "selector": "9" * 64,
                "session_cost": "a" * 64,
            },
            schema_version=GENERATION_SESSION_CONTRACT_V3_SCHEMA_VERSION,
            session_mode="development_generation_pilot",
        )
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            ref = publish_generation_session_contract(output_dir, contract)
            scope = PreGenesisExecutionScope(
                journal_root=output_dir / "job_attempts",
                session_id="generation-session",
                genesis_identity_hash=canonical_json_hash(snapshot),
                session_contract_hash=contract.contract_hash,
                lane_id="lane-a",
                invocation_segment_id="session-genesis",
                budget_contract_hash="b" * 64,
                attempt_budget_units=1,
                max_infrastructure_retries=0,
            )

            with self.assertRaisesRegex(
                ValueError,
                "baseline_eval_attempt_scope identity mismatch",
            ):
                prepare_session_genesis(
                    output_dir=output_dir,
                    session_id="generation-session",
                    task_plugin=task,
                    backend_adapter=backend,
                    runner=FakeRunner(),
                    policy_initialization=manifest.to_policy_initialization(),
                    genesis_identity_snapshot=snapshot,
                    baseline_manifest=manifest,
                    generation_session_contract_ref=ref,
                    baseline_eval_attempt_scope=scope,
                )

    def test_gpu_bound_baseline_private_artifact_preserves_host_binding(self) -> None:
        task = BaselineRequiredFakeTask()
        backend = GpuBoundBaselineBackend()
        runner = CountingFakeRunner()
        manifest = _manifest(task.task_spec.task_id)
        session_id = "gpu-baseline-session"
        snapshot = {"fixture": "gpu-baseline-session-v1"}
        resource_binding_hash = "d" * 64
        session_cost_hash = "a" * 64
        gpu_process_binding = GpuProcessBinding(
            host_execution_device="cuda:1",
            process_execution_device="cuda:0",
            resource_binding_hash=resource_binding_hash,
        )
        contract = GenerationSessionContract(
            schema_version=GENERATION_SESSION_CONTRACT_V3_SCHEMA_VERSION,
            session_mode="development_generation_pilot",
            session_id=session_id,
            task_id=task.task_spec.task_id,
            backend_id=backend.backend_id,
            max_generations=1,
            max_trials=2,
            max_train_jobs=2,
            bound_contract_hashes={
                "authoring_policy": "1" * 64,
                "continuation_policy": "2" * 64,
                "evaluation_cost": "3" * 64,
                "evaluation_protocol": "4" * 64,
                "exploration_projector": "e" * 64,
                "final_confirmation_plan": "5" * 64,
                "provider_policy": "6" * 64,
                "resource_plan": "7" * 64,
                "retry_policy": "8" * 64,
                "selector": "9" * 64,
                "session_cost": session_cost_hash,
            },
        )
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            contract_ref = publish_generation_session_contract(
                output_dir,
                contract,
            )
            eval_scope = PreGenesisExecutionScope(
                journal_root=output_dir / "job_attempts",
                session_id=session_id,
                genesis_identity_hash=session_genesis_identity_hash(snapshot),
                session_contract_hash=contract.contract_hash,
                lane_id="lane-a",
                invocation_segment_id="session-genesis",
                budget_contract_hash=session_cost_hash,
                attempt_budget_units=1,
                max_infrastructure_retries=0,
                resource_binding_hash=resource_binding_hash,
            )
            kwargs = {
                "output_dir": output_dir,
                "session_id": session_id,
                "task_plugin": task,
                "backend_adapter": backend,
                "runner": runner,
                "policy_initialization": manifest.to_policy_initialization(),
                "genesis_identity_snapshot": snapshot,
                "baseline_manifest": manifest,
                "generation_session_contract_ref": contract_ref,
                "gpu_process_binding": gpu_process_binding,
                "baseline_eval_attempt_scope": eval_scope,
            }

            created = prepare_session_genesis(**kwargs)
            private_path = output_dir / BASELINE_EVAL_ARTIFACT_FILENAME
            genesis_path = output_dir / SESSION_GENESIS_FILENAME
            private_bytes = private_path.read_bytes()
            genesis_bytes = genesis_path.read_bytes()
            private_payload = json.loads(private_bytes)
            self.assertEqual(
                private_payload["eval_job"]["gpu_process_binding"],
                gpu_process_binding.to_dict(),
            )
            self.assertEqual(runner.baseline_eval_calls, 1)

            loaded = prepare_session_genesis(**kwargs, require_existing=True)

            self.assertEqual(
                loaded.session_genesis_hash,
                created.session_genesis_hash,
            )
            self.assertEqual(runner.baseline_eval_calls, 1)
            self.assertEqual(private_path.read_bytes(), private_bytes)
            self.assertEqual(genesis_path.read_bytes(), genesis_bytes)
            self.assertEqual(
                backend.last_baseline_spec.gpu_process_binding,
                gpu_process_binding,
            )
            self.assertEqual(backend.last_baseline_spec.execution_device, "cuda:0")
            public_binding = backend.last_baseline_spec.to_public_dict()[
                "gpu_process_binding"
            ]
            self.assertEqual(
                public_binding,
                gpu_process_binding.to_public_dict(),
            )
            self.assertNotIn("host_execution_device", public_binding)

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
                genesis_identity_snapshot={"fixed": True},
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
                genesis_identity_snapshot={"fixed": True},
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
                    genesis_identity_snapshot={"fixed": True},
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
                genesis_identity_snapshot={"fixed": True},
                bootstrap_initial_best_score=0.1,
            )

            self.assertEqual(genesis.artifact.genesis_kind, "bootstrap")
            self.assertEqual(genesis.initial_current_best.mode, "bootstrap")
            self.assertEqual(genesis.initial_current_best.public_score, 0.1)

    def test_bootstrap_genesis_can_bind_initial_policy_state(self) -> None:
        expected = "a" * 64
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            genesis = prepare_session_genesis(
                output_dir=output_dir,
                session_id="session-bootstrap-paired",
                task_plugin=FakeTaskPlugin(),
                backend_adapter=FakeBackendAdapter(),
                runner=CountingFakeRunner(),
                policy_initialization=PolicyInitialization.random(),
                genesis_identity_snapshot={
                    "fixed": True,
                    "bootstrap_initial_policy_state_hash": expected,
                },
                bootstrap_initial_best_score=0.0,
                bootstrap_initial_policy_state_hash=expected,
            )

            self.assertEqual(genesis.expected_initial_policy_state_hash, expected)
            self.assertEqual(
                genesis.artifact.anchor.to_dict()["initial_policy_state_hash"],
                expected,
            )
            original = (output_dir / SESSION_GENESIS_FILENAME).read_bytes()
            with self.assertRaisesRegex(ValueError, "initial policy hash drift"):
                prepare_session_genesis(
                    output_dir=output_dir,
                    session_id="session-bootstrap-paired",
                    task_plugin=FakeTaskPlugin(),
                    backend_adapter=FakeBackendAdapter(),
                    runner=CountingFakeRunner(),
                    policy_initialization=PolicyInitialization.random(),
                    genesis_identity_snapshot={
                        "fixed": True,
                        "bootstrap_initial_policy_state_hash": expected,
                    },
                    bootstrap_initial_best_score=0.0,
                    bootstrap_initial_policy_state_hash="b" * 64,
                )
            self.assertEqual(
                (output_dir / SESSION_GENESIS_FILENAME).read_bytes(),
                original,
            )

    def test_genesis_identity_drift_fails_without_overwrite(self) -> None:
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
                genesis_identity_snapshot={"budget": 128},
                bootstrap_initial_best_score=0.1,
            )
            original = (output_dir / SESSION_GENESIS_FILENAME).read_bytes()

            with self.assertRaisesRegex(ValueError, "Genesis identity drift"):
                prepare_session_genesis(
                    output_dir=output_dir,
                    session_id="session-drift",
                    task_plugin=task,
                    backend_adapter=FakeBackendAdapter(),
                    runner=runner,
                    policy_initialization=PolicyInitialization.random(),
                    genesis_identity_snapshot={"budget": 129},
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
                genesis_identity_snapshot={"fixed": True},
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
                    genesis_identity_snapshot={"fixed": True},
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
                    genesis_identity_snapshot={"fixed": True},
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
                genesis_identity_snapshot={"fixed": True},
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
                    genesis_identity_snapshot={"fixed": True},
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
                genesis_identity_snapshot={"fixed": True},
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
                    genesis_identity_snapshot={"fixed": True},
                    baseline_manifest=manifest,
                )

            self.assertEqual(runner.baseline_eval_calls, 1)
            self.assertEqual(artifact_path.read_text(encoding="utf-8"), "{}\n")

    def test_legacy_baseline_genesis_reads_posix_private_uri_on_any_host(self) -> None:
        task = BaselineRequiredFakeTask()
        manifest = _manifest(task.task_spec.task_id)
        with tempfile.TemporaryDirectory() as tmp:
            current = prepare_session_genesis(
                output_dir=Path(tmp),
                session_id="legacy-baseline-posix-uri",
                task_plugin=task,
                backend_adapter=FakeBackendAdapter(),
                runner=CountingFakeRunner(),
                policy_initialization=manifest.to_policy_initialization(),
                genesis_identity_snapshot={"fixed": True},
                baseline_manifest=manifest,
            )
            payload = current.artifact.to_dict()

        payload["schema_version"] = LEGACY_SESSION_GENESIS_SCHEMA_VERSION
        payload.pop("backend_execution_environment")
        payload["anchor"]["private_eval_artifact"]["uri"] = (
            "/data/yuyangliu/history/baseline_eval_artifact.json"
        )
        payload["session_genesis_hash"] = canonical_json_hash(
            {
                key: value
                for key, value in payload.items()
                if key != "session_genesis_hash"
            }
        )

        historical = SessionGenesisArtifact.from_mapping(payload)
        self.assertEqual(
            historical.anchor.private_eval_artifact.uri,
            "/data/yuyangliu/history/baseline_eval_artifact.json",
        )
        relative = json.loads(json.dumps(payload))
        relative["anchor"]["private_eval_artifact"]["uri"] = (
            "history/baseline_eval_artifact.json"
        )
        with self.assertRaisesRegex(ValueError, "uri must be absolute"):
            SessionGenesisArtifact.from_mapping(relative)

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
                genesis_identity_snapshot={"fixed": True},
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
                    genesis_identity_snapshot={"fixed": True},
                    baseline_manifest=manifest,
                )

    def test_concurrent_baseline_genesis_has_one_writer_before_eval(self) -> None:
        task = BaselineRequiredFakeTask()
        manifest = _manifest(task.task_spec.task_id)
        winner_runner = BlockingBaselineRunner()
        loser_runner = CountingFakeRunner()
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)

            def prepare_winner():
                return prepare_session_genesis(
                    output_dir=output_dir,
                    session_id="session-concurrent",
                    task_plugin=task,
                    backend_adapter=FakeBackendAdapter(),
                    runner=winner_runner,
                    policy_initialization=manifest.to_policy_initialization(),
                    genesis_identity_snapshot={"fixed": True},
                    baseline_manifest=manifest,
                )

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(prepare_winner)
                self.assertTrue(winner_runner.eval_started.wait(timeout=5))
                with self.assertRaises(SessionGenesisCreationInProgressError):
                    prepare_session_genesis(
                        output_dir=output_dir,
                        session_id="session-concurrent",
                        task_plugin=task,
                        backend_adapter=FakeBackendAdapter(),
                        runner=loser_runner,
                        policy_initialization=manifest.to_policy_initialization(),
                        genesis_identity_snapshot={"fixed": True},
                        baseline_manifest=manifest,
                    )
                self.assertEqual(loser_runner.baseline_eval_calls, 0)
                self.assertFalse(
                    (output_dir / SESSION_GENESIS_FILENAME).exists()
                )
                winner_runner.allow_eval.set()
                winner = future.result(timeout=10)

            self.assertEqual(winner_runner.baseline_eval_calls, 1)
            self.assertTrue(
                (output_dir / SESSION_GENESIS_CREATION_LEASE_FILENAME).exists()
            )
            loaded = prepare_session_genesis(
                output_dir=output_dir,
                session_id="session-concurrent",
                task_plugin=task,
                backend_adapter=FakeBackendAdapter(),
                runner=loser_runner,
                policy_initialization=manifest.to_policy_initialization(),
                genesis_identity_snapshot={"fixed": True},
                baseline_manifest=manifest,
            )
            self.assertEqual(
                loaded.session_genesis_hash,
                winner.session_genesis_hash,
            )
            self.assertEqual(loser_runner.baseline_eval_calls, 0)

    def test_hard_crash_releases_os_creation_lock(self) -> None:
        task = BaselineRequiredFakeTask()
        manifest = _manifest(task.task_spec.task_id)
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            code = (
                "import os, sys\n"
                "from pathlib import Path\n"
                f"sys.path.insert(0, {str(SRC_ROOT)!r})\n"
                "from true_rl_agent.agent_core.session_genesis import "
                "_acquire_session_genesis_creation_lease\n"
                "_lease = _acquire_session_genesis_creation_lease("
                "output_dir=Path(sys.argv[1]), session_id='session-crash')\n"
                "os._exit(0)\n"
            )
            completed = subprocess.run(
                [sys.executable, "-c", code, str(output_dir)],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(
                (output_dir / SESSION_GENESIS_CREATION_LEASE_FILENAME).is_file()
            )
            lease_bytes = (
                output_dir / SESSION_GENESIS_CREATION_LEASE_FILENAME
            ).read_bytes()
            lease_payload = json.loads(lease_bytes)
            self.assertEqual(
                set(lease_payload),
                {
                    "schema_version",
                    "session_id",
                    "process_id",
                    "owner_token",
                    "ownership_model",
                },
            )
            self.assertEqual(
                lease_payload["schema_version"],
                "true_rl_agent.session_genesis.creation_lease.v2",
            )
            self.assertEqual(
                lease_bytes,
                canonical_json_bytes(lease_payload, trailing_newline=True),
            )

            recovered = prepare_session_genesis(
                output_dir=output_dir,
                session_id="session-crash",
                task_plugin=task,
                backend_adapter=FakeBackendAdapter(),
                runner=CountingFakeRunner(),
                policy_initialization=manifest.to_policy_initialization(),
                genesis_identity_snapshot={"fixed": True},
                baseline_manifest=manifest,
            )

            self.assertEqual(recovered.artifact.session_id, "session-crash")

    def test_dead_pre_genesis_train_attempt_uses_only_reserved_retry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            _spawn_unclean_pre_genesis_reservation(
                output_dir,
                "reference_train",
            )
            scope, identity, journal = _pre_genesis_recovery_fixture(
                output_dir,
                role="reference_train",
            )
            lease = _acquire_session_genesis_creation_lease(
                output_dir=output_dir,
                session_id=scope.session_id,
                pre_genesis_identity=identity,
            )
            try:
                self.assertTrue(
                    _recover_pre_genesis_execution(
                        output_dir=output_dir,
                        identity=identity,
                        scopes={"reference_train": scope},
                        previous_lease_payload=lease.previous_payload,
                    )
                )
                lease.activate()
                recovered = journal.load_result(0)
                self.assertEqual(
                    recovered.payload["schema_version"],
                    RECOVERED_ATTEMPT_RESULT_SCHEMA_VERSION,
                )
                self.assertEqual(
                    (
                        recovered.payload["outcome"],
                        recovered.payload["failure_kind"],
                        recovered.payload["retryable"],
                    ),
                    ("infrastructure_failure", "abandoned_unclean", True),
                )
                attempt_index, retry, previous = journal.begin_or_resume()
                self.assertEqual(attempt_index, 1)
                self.assertEqual(previous.payload_hash, recovered.payload_hash)
                journal.finish(
                    retry,
                    outcome="infrastructure_failure",
                    failure_kind="test_terminal",
                    retryable=False,
                    output_hash=None,
                    returncode=None,
                )
                with self.assertRaisesRegex(ValueError, "exhausted its retry"):
                    journal.begin_or_resume()
            finally:
                lease.release()

    def test_dead_pre_genesis_eval_attempt_is_terminal_without_retry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            _spawn_unclean_pre_genesis_reservation(
                output_dir,
                "reference_eval",
            )
            scope, identity, journal = _pre_genesis_recovery_fixture(
                output_dir,
                role="reference_eval",
            )
            lease = _acquire_session_genesis_creation_lease(
                output_dir=output_dir,
                session_id=scope.session_id,
                pre_genesis_identity=identity,
            )
            try:
                self.assertTrue(
                    _recover_pre_genesis_execution(
                        output_dir=output_dir,
                        identity=identity,
                        scopes={"reference_eval": scope},
                        previous_lease_payload=lease.previous_payload,
                    )
                )
                lease.activate()
                recovered = journal.load_result(0)
                self.assertFalse(recovered.payload["retryable"])
                with self.assertRaisesRegex(ValueError, "terminal result"):
                    journal.begin_or_resume()
                self.assertFalse(
                    (journal.root / "attempt_01.reservation.json").exists()
                )
            finally:
                lease.release()

    def test_published_pre_genesis_recovery_reconciles_missing_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            _spawn_unclean_pre_genesis_reservation(
                output_dir,
                "reference_train",
            )
            scope, identity, journal = _pre_genesis_recovery_fixture(
                output_dir,
                role="reference_train",
            )
            first_lease = _acquire_session_genesis_creation_lease(
                output_dir=output_dir,
                session_id=scope.session_id,
                pre_genesis_identity=identity,
            )
            try:
                unclean_lease = _parse_pre_genesis_creation_lease(
                    first_lease.previous_payload
                )
                self.assertIsNotNone(unclean_lease)
                scan = _scan_pre_genesis_attempts(
                    identity=identity,
                    scopes={"reference_train": scope},
                    recovery_bindings={},
                )
                recovery = _publish_or_load_pre_genesis_recovery(
                    output_dir=output_dir,
                    identity=identity,
                    scopes={"reference_train": scope},
                    unclean_lease=unclean_lease,
                    observed_incarnation=None,
                    process_id_reused=False,
                    reservations=scan["open_reservations"],
                )
                recovery_hash = canonical_json_hash(recovery)
                self.assertFalse(
                    (journal.root / "attempt_00.result.json").exists()
                )
            finally:
                # Leave the prior active bytes in place, exactly as a crash
                # between receipt publication and result closure would do.
                first_lease.release()

            second_lease = _acquire_session_genesis_creation_lease(
                output_dir=output_dir,
                session_id=scope.session_id,
                pre_genesis_identity=identity,
            )
            try:
                self.assertTrue(
                    _recover_pre_genesis_execution(
                        output_dir=output_dir,
                        identity=identity,
                        scopes={"reference_train": scope},
                        previous_lease_payload=second_lease.previous_payload,
                    )
                )
                result = journal.load_result(0)
                self.assertEqual(
                    result.payload["recovery_receipt_hash"],
                    recovery_hash,
                )
                second_lease.activate()
            finally:
                second_lease.release()

    def test_active_pre_genesis_owner_cannot_be_recovered(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            scope, identity, _ = _pre_genesis_recovery_fixture(
                output_dir,
                role="reference_train",
            )
            owner = _acquire_session_genesis_creation_lease(
                output_dir=output_dir,
                session_id=scope.session_id,
                pre_genesis_identity=identity,
            )
            owner.activate()
            owner.lock.release()
            inspector = _acquire_session_genesis_creation_lease(
                output_dir=output_dir,
                session_id=scope.session_id,
                pre_genesis_identity=identity,
            )
            try:
                with self.assertRaisesRegex(ValueError, "process is still alive"):
                    _recover_pre_genesis_execution(
                        output_dir=output_dir,
                        identity=identity,
                        scopes={"reference_train": scope},
                        previous_lease_payload=inspector.previous_payload,
                    )
            finally:
                inspector.release()

    def test_crash_recovery_rejects_partial_runner_state(self) -> None:
        task = BaselineRequiredFakeTask()
        manifest = _manifest(task.task_spec.task_id)
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            lease_path = output_dir / SESSION_GENESIS_CREATION_LEASE_FILENAME
            lease_path.write_text('{"process_id":999999}\n', encoding="utf-8")
            partial_path = (
                output_dir
                / "runner"
                / "session-partial"
                / "train"
                / "summary.json"
            )
            partial_path.parent.mkdir(parents=True)
            partial_path.write_text("{}\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "runner state"):
                prepare_session_genesis(
                    output_dir=output_dir,
                    session_id="session-partial",
                    task_plugin=task,
                    backend_adapter=FakeBackendAdapter(),
                    runner=CountingFakeRunner(),
                    policy_initialization=manifest.to_policy_initialization(),
                    genesis_identity_snapshot={"fixed": True},
                    baseline_manifest=manifest,
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
                genesis_identity_snapshot={"fixed": True},
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
                    genesis_identity_snapshot={"fixed": True},
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
                    genesis_identity_snapshot={"fixed": True},
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
                genesis_identity_snapshot={"fixed": True},
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
                    genesis_identity_snapshot={"fixed": True},
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
                genesis_identity_snapshot={"fixed": True},
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
                    genesis_identity_snapshot={"fixed": True},
                    bootstrap_initial_best_score=0.1,
                )


if __name__ == "__main__":
    unittest.main()
