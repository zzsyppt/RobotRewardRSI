from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import textwrap
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tests.fakes.backend_environment import (
    TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_MANIFEST_HASH,
    TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_VERIFICATION,
)
from tests.fakes.evaluation import (
    TEST_CAPABILITY_PROFILE,
    TEST_PROGRAM_SOURCE_CHECKSUM,
    TEST_RSL_RL_AGENT_CONFIGURATION,
    TEST_SUBJECT_MANIFEST_HASH,
    make_evaluation_binding,
    make_training_binding,
    make_verified_candidate_artifact,
)
from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.evaluation.phases import (
    EVALUATION_PHASE_SUMMARY_SCHEMA_VERSION,
    EvaluationPhase,
    EvaluationPhaseResult,
)
from true_rl_agent.evaluation.public_result import PublicEvalResult
from true_rl_agent.evaluation.subject import ExperimentSubject
from true_rl_agent.policy import BaselinePolicyRef, PolicyInitialization
from true_rl_agent.runner.base import CandidateRuntimeError, RunnerFailureKind
from true_rl_agent.runner.candidate_failure import (
    CandidateFailureContract,
    CandidateFailureRule,
)
from true_rl_agent.runner.evidence import evidence_artifact_from_json
from true_rl_agent.runner.gpu_lease import GpuDeviceIdentity, GpuLeaseManager
from true_rl_agent.runner.job_attempts import (
    RESOURCE_LEASE_ATTEMPT_RESULT_SCHEMA_VERSION,
    JobAttemptScope,
)
from true_rl_agent.runner.job_argv import validate_train_job_argv
from true_rl_agent.runner.specs import (
    BaselineEvalJobSpec,
    CheckpointOutputPlan,
    CheckpointRef,
    EvalJobSpec,
    EvalPhaseJobSpec,
    EvaluationJobSpec,
    GpuProcessBinding,
    TrainJobSpec,
)
from true_rl_agent.runner.subprocess_runner import (
    RunnerExecutionError,
    SubprocessRunner,
)

_EVAL_CHECKPOINT_BYTES = b"subprocess-runner-eval-checkpoint"
_EVAL_CHECKPOINT_HASH = (
    "sha256:" + hashlib.sha256(_EVAL_CHECKPOINT_BYTES).hexdigest()
)
_CURRENT_CHECKPOINT_HASH = "sha256:" + "2" * 64
_FORGED_CHECKPOINT_HASH = "sha256:" + "6" * 64
_OLD_CHECKPOINT_HASH = "sha256:" + "3" * 64
_OTHER_CHECKPOINT_HASH = "sha256:" + "5" * 64
_TRAIN_CHECKPOINT_HASH = "sha256:" + "1" * 64
_WRONG_CHECKPOINT_HASH = "sha256:" + "4" * 64

_TEST_CANDIDATE_FAILURE_CONTRACT = CandidateFailureContract(
    contract_id="true_rl_agent.test.candidate_failure.v1",
    rules=(
        CandidateFailureRule(
            stage="activation_preflight",
            reason_codes=("activation_rejected",),
            requires_initial_policy_state=False,
        ),
        CandidateFailureRule(
            stage="ppo_reward_execution",
            reason_codes=("runtime_rejected",),
            requires_initial_policy_state=True,
        ),
    ),
)


def _expected_train_summary_binding(
    *,
    job_id: str,
    candidate_id: str,
    budget_steps: int = 128,
) -> dict:
    artifact = _verified_test_candidate(candidate_id)
    training_binding = make_training_binding(
        PolicyInitialization.random(),
        budget_steps=budget_steps,
    )
    return {
        "job_id": job_id,
        "task_id": "unitree_push_recovery",
        "subject_kind": "candidate",
        "subject_id": candidate_id,
        "subject_manifest_hash": artifact.subject_manifest_hash,
        "training_binding_hash": training_binding.binding_hash,
        "policy_architecture_hash": training_binding.policy_architecture_hash,
        "optimizer_training_profile_hash": (
            training_binding.optimizer_training_profile_hash
        ),
        "agent_configuration_hash": training_binding.agent_configuration_hash,
        "initialization_seed_schedule_hash": (
            training_binding.initialization_seed_schedule_hash
        ),
        "capability_profile_hash": training_binding.capability_profile_hash,
        "program_source_checksum": artifact.program_source_checksum,
        "execution_device": "cuda:0",
    }


def _train_job_spec(*, candidate_id: str, **kwargs) -> TrainJobSpec:
    initialization = kwargs["policy_initialization"]
    binding = make_training_binding(
        initialization,
        budget_steps=kwargs["budget_steps"],
    )
    metadata = dict(kwargs.get("metadata", {}))
    artifact = _verified_test_candidate(candidate_id)
    candidate_failure_contract = kwargs.get("candidate_failure_contract")
    argv = metadata.get("cli_argv")
    if isinstance(argv, list):
        metadata["cli_argv"] = _bound_train_argv(
            argv,
            binding,
            job_id=kwargs["job_id"],
            task_id=kwargs["task_id"],
            candidate_id=candidate_id,
            artifact=artifact,
            budget_steps=kwargs["budget_steps"],
            initialization=initialization,
            metadata=metadata,
            expected_initial_policy_state_hash=kwargs.get(
                "expected_initial_policy_state_hash"
            ),
            candidate_failure_contract=candidate_failure_contract,
            gpu_process_binding=kwargs.get("gpu_process_binding"),
        )
    kwargs["metadata"] = metadata
    kwargs.setdefault("execution_device", "cuda:0")
    return TrainJobSpec(
        subject=ExperimentSubject.candidate(candidate_id),
        subject_manifest_hash=artifact.subject_manifest_hash,
        program_source_checksum=artifact.program_source_checksum,
        training_binding=binding,
        candidate_artifact=artifact,
        **kwargs,
    )


def _verified_test_candidate(candidate_id: str):
    return make_verified_candidate_artifact(candidate_id)


def _bound_train_argv(
    argv: list[str],
    binding,
    *,
    job_id: str,
    task_id: str,
    candidate_id: str,
    artifact,
    budget_steps: int,
    initialization: PolicyInitialization,
    metadata: dict,
    expected_initial_policy_state_hash: str | None = None,
    candidate_failure_contract: CandidateFailureContract | None = None,
    gpu_process_binding: GpuProcessBinding | None = None,
) -> list[str]:
    result = [
        *argv,
        "--job-id",
        job_id,
        "--round",
        "test-round",
        "--task-id",
        task_id,
        "--subject-kind",
        "candidate",
        "--subject-id",
        candidate_id,
        "--subject-manifest-hash",
        artifact.subject_manifest_hash,
        "--program-source-checksum",
        artifact.program_source_checksum,
        "--candidate",
        artifact.path,
        "--candidate-artifact-hash",
        artifact.artifact_hash,
        "--budget-steps",
        str(budget_steps),
        "--init",
        initialization.kind,
        "--summary-path",
        str(metadata.get("summary_path") or "missing-summary-path"),
        "--checkpoint-output",
        str(metadata.get("checkpoint_output_path") or "missing-checkpoint-output"),
        "--training-binding-hash",
        str(binding.binding_hash),
        "--capability-profile-hash",
        binding.capability_profile_hash,
        "--policy-architecture-hash",
        binding.policy_architecture_hash,
        "--optimizer-training-profile-hash",
        binding.optimizer_training_profile_hash,
        "--agent-configuration-hash",
        binding.agent_configuration_hash,
        "--agent-configuration-json",
        TEST_RSL_RL_AGENT_CONFIGURATION.to_canonical_json(),
        "--initialization-seed-schedule-hash",
        binding.initialization_seed_schedule_hash,
        "--seed",
        "42",
        "--device",
        "cuda:0",
    ]
    if expected_initial_policy_state_hash is not None:
        result.extend(
            [
                "--expected-initial-policy-state-hash",
                expected_initial_policy_state_hash,
            ]
        )
    if candidate_failure_contract is not None:
        result.extend(
            [
                "--candidate-failure-contract-hash",
                candidate_failure_contract.contract_hash,
            ]
        )
    if gpu_process_binding is not None:
        result.extend(
            [
                "--gpu-host-execution-device",
                gpu_process_binding.host_execution_device,
                "--gpu-process-execution-device",
                gpu_process_binding.process_execution_device,
                "--gpu-resource-binding-hash",
                gpu_process_binding.resource_binding_hash,
            ]
        )
    return result


def _with_train_argv(spec: TrainJobSpec, argv: list[str]) -> TrainJobSpec:
    metadata = dict(spec.metadata)
    metadata["cli_argv"] = _bound_train_argv(
        argv,
        spec.training_binding,
        job_id=spec.job_id,
        task_id=spec.task_id,
        candidate_id=spec.subject.subject_id,
        artifact=spec.candidate_artifact,
        budget_steps=spec.budget_steps,
        initialization=spec.policy_initialization,
        metadata=metadata,
        expected_initial_policy_state_hash=(
            spec.expected_initial_policy_state_hash
        ),
        candidate_failure_contract=spec.candidate_failure_contract,
        gpu_process_binding=spec.gpu_process_binding,
    )
    return replace(spec, metadata=metadata)


def _eval_job_spec(*, candidate_id: str, **kwargs) -> EvalJobSpec:
    kwargs.setdefault("execution_device", "cuda:0")
    spec = EvalJobSpec(
        subject=ExperimentSubject.candidate(candidate_id),
        subject_manifest_hash=TEST_SUBJECT_MANIFEST_HASH,
        program_source_checksum=TEST_PROGRAM_SOURCE_CHECKSUM,
        training_binding_hash="7" * 64,
        capability_profile_hash=str(TEST_CAPABILITY_PROFILE.profile_hash),
        **kwargs,
    )
    return _bind_eval_spec_argv(spec)


def _baseline_eval_job_spec(
    *,
    baseline_id: str,
    baseline_manifest_hash: str,
    **kwargs,
) -> BaselineEvalJobSpec:
    kwargs.setdefault("execution_device", "cuda:0")
    spec = BaselineEvalJobSpec(
        subject=ExperimentSubject.baseline(baseline_id),
        subject_manifest_hash=baseline_manifest_hash,
        capability_profile_hash=str(TEST_CAPABILITY_PROFILE.profile_hash),
        **kwargs,
    )
    return _bind_eval_spec_argv(spec)


def _bind_eval_spec_argv(spec: EvaluationJobSpec) -> EvaluationJobSpec:
    metadata = dict(spec.metadata)
    argv = metadata.get("cli_argv")
    if not isinstance(argv, list):
        return spec
    binding = spec.evaluation_binding
    protocol = spec.evaluation_protocol()
    argv = [
        *argv,
        "--job-id",
        spec.job_id,
        "--round",
        "test-round",
        "--task-id",
        spec.task_id,
        "--subject-kind",
        spec.subject.kind,
        "--subject-id",
        spec.subject.subject_id,
        "--subject-manifest-hash",
        str(spec.subject_manifest_hash),
        "--public-seeds",
        str(protocol.public_seed_count),
        "--public-accept-seeds",
        str(protocol.public_accept_seed_count),
        "--sealed-validation-seeds",
        str(protocol.sealed_validation_seed_count),
        "--eval-protocol-version",
        protocol.eval_protocol_version,
        "--task-protocol-hash",
        binding.task_protocol_hash,
        "--acceptance-protocol-hash",
        binding.acceptance_protocol_hash,
        "--seed-cohort-hash",
        binding.seed_cohort_hash,
        "--evaluation-binding-hash",
        binding.binding_hash,
        "--capability-profile-hash",
        spec.capability_profile_hash,
        "--summary-path",
        str(metadata.get("summary_path") or "missing-summary-path"),
        "--num-envs",
        str(protocol.eval_num_envs),
        "--seed",
        "0",
        "--device",
        spec.execution_device,
    ]
    execution_environment_hash = metadata.get(
        "backend_execution_environment_manifest_hash"
    )
    if execution_environment_hash is not None:
        argv.extend(
            [
                "--backend-execution-environment-manifest-hash",
                str(execution_environment_hash),
            ]
        )
    if isinstance(spec, EvalPhaseJobSpec):
        argv.extend(
            [
                "--program-source-checksum",
                spec.program_source_checksum,
                "--training-binding-hash",
                spec.training_binding_hash,
                "--evaluation-phase",
                spec.phase.value,
                "--evaluation-plan-hash",
                spec.evaluation_plan_hash,
            ]
        )
    elif isinstance(spec, EvalJobSpec):
        argv.extend(
            [
                "--program-source-checksum",
                spec.program_source_checksum,
                "--training-binding-hash",
                spec.training_binding_hash,
            ]
        )
    gpu_process_binding = getattr(spec, "gpu_process_binding", None)
    if gpu_process_binding is not None:
        argv.extend(
            [
                "--gpu-host-execution-device",
                gpu_process_binding.host_execution_device,
                "--gpu-process-execution-device",
                gpu_process_binding.process_execution_device,
                "--gpu-resource-binding-hash",
                gpu_process_binding.resource_binding_hash,
            ]
        )
    metadata["cli_argv"] = argv
    return replace(spec, metadata=metadata)


def _with_eval_argv(
    spec: EvaluationJobSpec,
    argv: list[str],
) -> EvaluationJobSpec:
    metadata = dict(spec.metadata)
    metadata["cli_argv"] = argv
    return _bind_eval_spec_argv(replace(spec, metadata=metadata))


class SubprocessRunnerTests(unittest.TestCase):
    def test_subprocess_runner_executes_and_validates_one_eval_phase(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            phase_summary = root / "phase-summary.json"
            plan = _eval_job_spec(
                job_id="eval-phase-plan",
                task_id="unitree_push_recovery",
                candidate_id="candidate-phase",
                checkpoint=self._eval_checkpoint(root),
                public_seed_count=3,
                public_accept_seed_count=5,
                sealed_validation_seed_count=7,
                evaluation_binding=make_evaluation_binding(
                    public_seed_count=3,
                    public_accept_seed_count=5,
                    sealed_validation_seed_count=7,
                ),
                metadata={
                    "summary_path": str(root / "full-summary.json"),
                    "backend_id": "isaac_lab_rsl_rl",
                    "backend_execution_environment_manifest_hash": (
                        TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_MANIFEST_HASH
                    ),
                },
            )
            phase_spec = EvalPhaseJobSpec(
                job_id="eval-phase-public-dev",
                evaluation_plan=plan,
                phase=EvaluationPhase.PUBLIC_DEV,
                metadata={
                    "summary_path": str(phase_summary),
                    "backend_id": "isaac_lab_rsl_rl",
                    "backend_execution_environment_manifest_hash": (
                        TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_MANIFEST_HASH
                    ),
                },
            )
            task_evidence = {
                "schema_version": "test.task.phase_evidence.v1",
                "phase": "public_dev",
                "score": 0.4,
            }
            summary = {
                **self._eval_summary_binding(phase_spec),
                "schema_version": EVALUATION_PHASE_SUMMARY_SCHEMA_VERSION,
                "evaluation_phase": "public_dev",
                "evaluation_plan_hash": phase_spec.evaluation_plan_hash,
                "status": "pass",
                "task_evidence": task_evidence,
                "task_evidence_hash": canonical_json_hash(task_evidence),
                "evaluation_execution_cost": {
                    "schema_version": "test.phase_cost.v1",
                    "splits": {"public_dev": {"seed_count": 3}},
                },
                "environment_source_preflight": {"status": "pass"},
                "backend_execution_environment_verification": dict(
                    TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_VERIFICATION
                ),
                "checkpoint_safety_report": self._checkpoint_safety_report(),
                "candidate_code_loaded_in_eval": False,
                "candidate_package_loaded": False,
                "candidate_package_loaded_in_eval": False,
                "candidate_path_argument_present": False,
                "errors": [],
            }
            code = textwrap.dedent(
                f"""
                import json
                from pathlib import Path
                Path({str(phase_summary)!r}).write_text({json.dumps(summary)!r})
                """
            )
            phase_spec = _with_eval_argv(
                phase_spec,
                self._checkpoint_cli_argv(
                    plan.execution_checkpoint(),
                    code,
                ),
            )
            runner = SubprocessRunner(
                allowed_executables=(sys.executable,),
                timeout_s=10,
            )

            result = runner.run_eval_phase(phase_spec)

        self.assertEqual(result.phase, EvaluationPhase.PUBLIC_DEV)
        self.assertEqual(result.evaluation_plan_hash, phase_spec.evaluation_plan_hash)
        self.assertEqual(result.task_evidence, task_evidence)
        self.assertFalse(result.to_private_dict()["acceptance_authority"])

    def test_bound_eval_phase_receipt_uses_validated_summary_binding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            phase_summary = root / "bound-phase-summary.json"
            identity = GpuDeviceIdentity("cuda:1", "machine", "GPU-BBB")
            binding = GpuProcessBinding(
                host_execution_device="cuda:1",
                process_execution_device="cuda:0",
                resource_binding_hash=identity.identity_hash,
            )
            scope = JobAttemptScope(
                journal_root=root / "attempts",
                session_genesis_hash="a" * 64,
                session_contract_hash="b" * 64,
                generation_id="generation-001",
                lane_id="lane-b",
                invocation_segment_id="segment-000",
                budget_contract_hash="c" * 64,
                attempt_budget_units=1,
                max_infrastructure_retries=0,
                resource_binding_hash=identity.identity_hash,
            )
            plan = _eval_job_spec(
                job_id="bound-eval-phase-plan",
                task_id="unitree_push_recovery",
                candidate_id="candidate-phase",
                checkpoint=self._eval_checkpoint(root),
                public_seed_count=3,
                public_accept_seed_count=5,
                sealed_validation_seed_count=7,
                evaluation_binding=make_evaluation_binding(
                    public_seed_count=3,
                    public_accept_seed_count=5,
                    sealed_validation_seed_count=7,
                ),
                gpu_process_binding=binding,
                metadata={
                    "summary_path": str(root / "full-summary.json"),
                    "backend_id": "isaac_lab_rsl_rl",
                    "backend_execution_environment_manifest_hash": (
                        TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_MANIFEST_HASH
                    ),
                },
            )
            phase_spec = EvalPhaseJobSpec(
                job_id="bound-eval-phase-public-dev",
                evaluation_plan=plan,
                phase=EvaluationPhase.PUBLIC_DEV,
                attempt_scope=scope,
                metadata={
                    "summary_path": str(phase_summary),
                    "backend_id": "isaac_lab_rsl_rl",
                    "backend_execution_environment_manifest_hash": (
                        TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_MANIFEST_HASH
                    ),
                },
            )
            task_evidence = {
                "schema_version": "test.task.phase_evidence.v1",
                "phase": "public_dev",
                "score": 0.4,
            }
            summary = {
                **self._eval_summary_binding(phase_spec),
                "schema_version": EVALUATION_PHASE_SUMMARY_SCHEMA_VERSION,
                "evaluation_phase": "public_dev",
                "evaluation_plan_hash": phase_spec.evaluation_plan_hash,
                "status": "pass",
                "gpu_process_binding": binding.to_dict(),
                "task_evidence": task_evidence,
                "task_evidence_hash": canonical_json_hash(task_evidence),
                "evaluation_execution_cost": {
                    "schema_version": "test.phase_cost.v1",
                    "splits": {"public_dev": {"seed_count": 3}},
                },
                "environment_source_preflight": {"status": "pass"},
                "backend_execution_environment_verification": dict(
                    TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_VERIFICATION
                ),
                "checkpoint_safety_report": self._checkpoint_safety_report(),
                "candidate_code_loaded_in_eval": False,
                "candidate_package_loaded": False,
                "candidate_package_loaded_in_eval": False,
                "candidate_path_argument_present": False,
                "errors": [],
            }
            code = textwrap.dedent(
                f"""
                import json
                import os
                from pathlib import Path
                summary = json.loads({json.dumps(summary)!r})
                pid = os.getpid()
                summary["owned_process_identity"] = {{
                    "schema_version": "true_rl_agent.runner.owned_process_identity.v1",
                    "process_id": pid,
                    "process_group_id": getattr(os, "getpgrp", lambda: pid)(),
                }}
                Path({str(phase_summary)!r}).write_text(json.dumps(summary))
                """
            )
            phase_spec = _with_eval_argv(
                phase_spec,
                self._checkpoint_cli_argv(
                    plan.execution_checkpoint(),
                    code,
                ),
            )
            runner = SubprocessRunner(
                allowed_executables=(sys.executable,),
                timeout_s=10,
                gpu_lease_manager=GpuLeaseManager(
                    lease_root=root / "leases",
                    devices={"cuda:1": identity},
                ),
            )

            result = runner.run_eval_phase(phase_spec)

        self.assertEqual(result.execution_device, "cuda:0")
        self.assertEqual(result.resource_binding_hash, identity.identity_hash)
        self.assertEqual(result.gpu_process_binding, binding.to_dict())
        self.assertEqual(
            result.to_private_dict()["gpu_process_binding"],
            binding.to_dict(),
        )
        self.assertEqual(
            EvaluationPhaseResult.from_private_mapping(
                result.to_private_dict()
            ).to_private_dict(),
            result.to_private_dict(),
        )

    def test_backend_environment_summary_receipt_is_exact_and_hash_bound(self) -> None:
        spec = SimpleNamespace(
            job_id="isaac-job",
            metadata={
                "backend_id": "isaac_lab_rsl_rl",
                "backend_execution_environment_manifest_hash": (
                    TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_MANIFEST_HASH
                ),
            },
        )
        SubprocessRunner._validate_backend_execution_environment_summary(
            {
                "backend_execution_environment_verification": dict(
                    TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_VERIFICATION
                )
            },
            spec,
        )

        drifted = dict(TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_VERIFICATION)
        drifted["actual_manifest_hash"] = "0" * 64
        with self.assertRaisesRegex(
            RunnerExecutionError,
            "backend environment verification mismatch",
        ):
            SubprocessRunner._validate_backend_execution_environment_summary(
                {"backend_execution_environment_verification": drifted},
                spec,
            )

    def test_fake_backend_remains_neutral_to_environment_receipts(self) -> None:
        spec = SimpleNamespace(
            job_id="fake-job",
            metadata={"backend_id": "fake_backend"},
        )
        SubprocessRunner._validate_backend_execution_environment_summary({}, spec)
        with self.assertRaisesRegex(
            RunnerExecutionError,
            "unexpected backend environment verification",
        ):
            SubprocessRunner._validate_backend_execution_environment_summary(
                {
                    "backend_execution_environment_verification": dict(
                        TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_VERIFICATION
                    )
                },
                spec,
            )

    def test_runner_constructor_rejects_invalid_execution_config(self) -> None:
        for kwargs, error in (
            ({"allowed_executables": ()}, "allowed_executables"),
            ({"allowed_executables": [sys.executable]}, "allowed_executables"),
            (
                {"allowed_executables": (sys.executable,), "timeout_s": True},
                "timeout_s",
            ),
            (
                {"allowed_executables": (sys.executable,), "timeout_s": 1.5},
                "timeout_s",
            ),
            (
                {"allowed_executables": (sys.executable,), "cwd": "."},
                "cwd",
            ),
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex((TypeError, ValueError), error):
                    SubprocessRunner(**kwargs)

    def test_bound_train_uses_host_lease_and_child_only_uuid_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary_path = root / "train-summary.json"
            checkpoint_path = root / "trained.pt"
            environment_path = root / "child-environment.json"
            checkpoint_bytes = b"bound-train-checkpoint"
            checkpoint_hash = (
                "sha256:" + hashlib.sha256(checkpoint_bytes).hexdigest()
            )
            identity = GpuDeviceIdentity(
                "cuda:1",
                "test-machine",
                "GPU-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
            )
            binding = GpuProcessBinding(
                host_execution_device="cuda:1",
                process_execution_device="cuda:0",
                resource_binding_hash=identity.identity_hash,
            )
            scope = JobAttemptScope(
                journal_root=root / "attempts",
                session_genesis_hash="a" * 64,
                session_contract_hash="b" * 64,
                generation_id="generation-001",
                lane_id="lane-b",
                invocation_segment_id="segment-000",
                budget_contract_hash="c" * 64,
                attempt_budget_units=1,
                max_infrastructure_retries=0,
                resource_binding_hash=identity.identity_hash,
            )
            summary = {
                **_expected_train_summary_binding(
                    job_id="bound-train",
                    candidate_id="candidate-code",
                ),
                "status": "pass",
                "gpu_process_binding": binding.to_dict(),
                "checkpoint": {
                    "uri": str(checkpoint_path),
                    "checksum": checkpoint_hash,
                    "role": "train_output",
                    "metadata": {"local_path": str(checkpoint_path)},
                },
            }
            code = textwrap.dedent(
                f"""
                import json
                import os
                from pathlib import Path
                Path({str(environment_path)!r}).write_text(json.dumps({{
                    "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "NVIDIA_VISIBLE_DEVICES": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
                }}))
                Path({str(checkpoint_path)!r}).write_bytes({checkpoint_bytes!r})
                summary = json.loads({json.dumps(summary)!r})
                pid = os.getpid()
                summary["owned_process_identity"] = {{
                    "schema_version": "true_rl_agent.runner.owned_process_identity.v1",
                    "process_id": pid,
                    "process_group_id": getattr(os, "getpgrp", lambda: pid)(),
                }}
                Path({str(summary_path)!r}).write_text(json.dumps(summary))
                """
            )
            spec = _train_job_spec(
                job_id="bound-train",
                task_id="unitree_push_recovery",
                candidate_id="candidate-code",
                policy_initialization=PolicyInitialization.random(),
                budget_steps=128,
                execution_device="cuda:0",
                gpu_process_binding=binding,
                attempt_scope=scope,
                metadata={
                    "cli_argv": [sys.executable, "-c", code],
                    "summary_path": str(summary_path),
                    "checkpoint_output_path": str(checkpoint_path),
                    "sealed_checkpoint_uri": "artifact://bound/sealed/best.pt",
                },
            )
            runner = SubprocessRunner(
                allowed_executables=(sys.executable,),
                timeout_s=10,
                gpu_lease_manager=GpuLeaseManager(
                    lease_root=root / "leases",
                    devices={"cuda:1": identity},
                ),
            )

            with patch.dict(
                os.environ,
                {
                    "CUDA_VISIBLE_DEVICES": "PARENT-CUDA",
                    "NVIDIA_VISIBLE_DEVICES": "PARENT-NVIDIA",
                },
            ):
                result = runner.run_train(spec)
                self.assertEqual(
                    os.environ["CUDA_VISIBLE_DEVICES"],
                    "PARENT-CUDA",
                )
                self.assertEqual(
                    os.environ["NVIDIA_VISIBLE_DEVICES"],
                    "PARENT-NVIDIA",
                )

            child_environment = json.loads(environment_path.read_text())
            self.assertEqual(
                child_environment,
                {
                    "CUDA_VISIBLE_DEVICES": identity.physical_id,
                    "NVIDIA_VISIBLE_DEVICES": None,
                },
            )
            self.assertEqual(result.gpu_process_binding, binding)
            self.assertEqual(
                result.checkpoint.metadata["gpu_process_binding"],
                binding.to_dict(),
            )
            self.assertEqual(
                result.to_public_dict()["gpu_process_binding"],
                binding.to_public_dict(),
            )
            self.assertNotIn(
                "host_execution_device",
                result.to_public_dict()["gpu_process_binding"],
            )
            attempt_results = list(
                (root / "attempts").rglob("attempt_00.result.json")
            )
            self.assertEqual(len(attempt_results), 1)
            attempt_result = json.loads(attempt_results[0].read_text())
            self.assertEqual(
                attempt_result["schema_version"],
                RESOURCE_LEASE_ATTEMPT_RESULT_SCHEMA_VERSION,
            )
            self.assertGreater(
                attempt_result["resource_lease_wall_duration_ns"],
                0,
            )
            with self.assertRaisesRegex(ValueError, "must be absent"):
                validate_train_job_argv(
                    replace(spec, gpu_process_binding=None),
                    spec.metadata["cli_argv"],
                )

            untrusted_hash = "d" * 64
            with self.assertRaisesRegex(
                RunnerExecutionError,
                "GPU resource binding is invalid",
            ):
                runner.run_train(
                    replace(
                        spec,
                        job_id="untrusted-bound-train",
                        gpu_process_binding=replace(
                            binding,
                            resource_binding_hash=untrusted_hash,
                        ),
                        attempt_scope=replace(
                            scope,
                            journal_root=root / "untrusted-attempts",
                            resource_binding_hash=untrusted_hash,
                        ),
                    )
                )

            missing_binding_summary = dict(summary)
            missing_binding_summary.pop("gpu_process_binding")
            summary_path.write_text(json.dumps(missing_binding_summary))
            with self.assertRaises(RunnerExecutionError) as raised:
                runner.run_train(spec)
            self.assertEqual(
                raised.exception.failure_kind,
                RunnerFailureKind.SUMMARY_INVALID,
            )

    def test_baseline_eval_optional_gpu_binding_omits_legacy_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            identity = GpuDeviceIdentity("cuda:1", "machine", "GPU-BBB")
            binding = GpuProcessBinding(
                host_execution_device="cuda:1",
                process_execution_device="cuda:0",
                resource_binding_hash=identity.identity_hash,
            )
            bound = _baseline_eval_job_spec(
                job_id="bound-baseline",
                task_id="unitree_push_recovery",
                baseline_id="baseline-policy",
                baseline_manifest_hash="e" * 64,
                checkpoint=self._eval_checkpoint(root, role="warm_start"),
                public_seed_count=3,
                public_accept_seed_count=5,
                sealed_validation_seed_count=7,
                evaluation_binding=make_evaluation_binding(
                    public_seed_count=3,
                    public_accept_seed_count=5,
                    sealed_validation_seed_count=7,
                ),
                gpu_process_binding=binding,
            )

            self.assertEqual(
                bound.to_public_dict()["gpu_process_binding"],
                binding.to_public_dict(),
            )
            self.assertNotIn(
                "gpu_process_binding",
                replace(bound, gpu_process_binding=None).to_public_dict(),
            )

    def test_summary_missing_uses_exactly_one_reserved_infrastructure_retry(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            counter_path = root / "launch-count.txt"
            identity = GpuDeviceIdentity("cuda:0", "test-machine", "GPU-TEST")
            scope = JobAttemptScope(
                journal_root=root / "attempts",
                session_genesis_hash="a" * 64,
                session_contract_hash="b" * 64,
                generation_id="generation-001",
                lane_id="lane-a",
                invocation_segment_id="segment-000",
                budget_contract_hash="c" * 64,
                attempt_budget_units=2,
                max_infrastructure_retries=1,
                resource_binding_hash=identity.identity_hash,
            )
            code = textwrap.dedent(
                f"""
                from pathlib import Path
                path = Path({str(counter_path)!r})
                count = int(path.read_text()) if path.exists() else 0
                path.write_text(str(count + 1))
                """
            )
            spec = _train_job_spec(
                job_id="summary-missing-retry",
                task_id="unitree_push_recovery",
                candidate_id="candidate-code",
                policy_initialization=PolicyInitialization.random(),
                budget_steps=128,
                attempt_scope=scope,
                metadata={
                    "cli_argv": [sys.executable, "-c", code],
                    "summary_path": str(root / "missing-summary.json"),
                    "checkpoint_output_path": str(root / "missing-checkpoint.pt"),
                },
            )
            runner = SubprocessRunner(
                allowed_executables=(sys.executable,),
                timeout_s=10,
                gpu_lease_manager=GpuLeaseManager(
                    lease_root=root / "leases",
                    devices={"cuda:0": identity},
                ),
            )
            self.assertNotIn("attempt_scope", spec.to_public_dict())

            with self.assertRaises(RunnerExecutionError) as raised:
                runner.run_train(spec)

            self.assertEqual(
                raised.exception.failure_kind.value,
                "summary_missing",
            )
            self.assertEqual(counter_path.read_text(), "2")
            results = sorted((root / "attempts").rglob("attempt_*.result.json"))
            self.assertEqual(len(results), 2)
            payloads = [json.loads(path.read_text()) for path in results]
            self.assertEqual([item["retryable"] for item in payloads], [True, False])

    def test_completed_train_attempt_is_revalidated_without_relaunch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary_path = root / "train-summary.json"
            checkpoint_path = root / "trained.pt"
            launch_count_path = root / "launch-count.txt"
            checkpoint_bytes = b"completed-attempt-checkpoint"
            checkpoint_hash = "sha256:" + hashlib.sha256(checkpoint_bytes).hexdigest()
            identity = GpuDeviceIdentity("cuda:0", "test-machine", "GPU-TEST")
            scope = JobAttemptScope(
                journal_root=root / "attempts",
                session_genesis_hash="a" * 64,
                session_contract_hash="b" * 64,
                generation_id="generation-001",
                lane_id="lane-a",
                invocation_segment_id="segment-000",
                budget_contract_hash="c" * 64,
                attempt_budget_units=2,
                max_infrastructure_retries=1,
                resource_binding_hash=identity.identity_hash,
            )
            summary = {
                **_expected_train_summary_binding(
                    job_id="completed-attempt-train",
                    candidate_id="candidate-code",
                ),
                "status": "pass",
                "checkpoint": {
                    "uri": str(checkpoint_path),
                    "checksum": checkpoint_hash,
                    "role": "train_output",
                    "metadata": {"local_path": str(checkpoint_path)},
                },
            }
            code = textwrap.dedent(
                f"""
                import json
                from pathlib import Path
                counter = Path({str(launch_count_path)!r})
                count = int(counter.read_text()) if counter.exists() else 0
                counter.write_text(str(count + 1))
                Path({str(checkpoint_path)!r}).write_bytes({checkpoint_bytes!r})
                Path({str(summary_path)!r}).write_text({json.dumps(summary)!r})
                """
            )
            spec = _train_job_spec(
                job_id="completed-attempt-train",
                task_id="unitree_push_recovery",
                candidate_id="candidate-code",
                policy_initialization=PolicyInitialization.random(),
                budget_steps=128,
                attempt_scope=scope,
                metadata={
                    "cli_argv": [sys.executable, "-c", code],
                    "summary_path": str(summary_path),
                    "checkpoint_output_path": str(checkpoint_path),
                    "sealed_checkpoint_uri": "artifact://train/sealed/best.pt",
                },
            )
            runner = SubprocessRunner(
                allowed_executables=(sys.executable,),
                timeout_s=10,
                gpu_lease_manager=GpuLeaseManager(
                    lease_root=root / "leases",
                    devices={"cuda:0": identity},
                ),
            )

            first = runner.run_train(spec)
            second = runner.run_train(spec)

            self.assertEqual(launch_count_path.read_text(), "1")
            self.assertEqual(first.to_public_dict(), second.to_public_dict())
            summary_path.unlink()
            with self.assertRaises(RunnerExecutionError) as raised:
                runner.run_train(spec)
            self.assertEqual(
                raised.exception.failure_kind,
                RunnerFailureKind.SUMMARY_INVALID,
            )
            self.assertEqual(launch_count_path.read_text(), "1")

    def _checkpoint_safety_report(
        self,
        *,
        status: str = "pass",
        reason: str = "checkpoint_safe_to_load",
        checkpoint_hash: str = _EVAL_CHECKPOINT_HASH,
    ) -> dict:
        return {
            "schema_version": "true_rl_agent.checkpoint_safety_report.v1",
            "status": status,
            "checkpoint_hash": checkpoint_hash,
            "expected_checkpoint_hash": checkpoint_hash,
            "file_exists": True,
            "byte_read_status": "pass",
            "weights_only_load_status": "pass" if status == "pass" else "fail",
            "checkpoint_hash_match_status": "pass",
            "loader_error_class": None if status == "pass" else "RuntimeError",
            "loader_error_summary": None if status == "pass" else "RuntimeError",
            "reason": reason,
        }

    def _eval_checkpoint(
        self,
        root: Path,
        *,
        role: str = "eval",
        name: str = "eval-input.pt",
    ) -> CheckpointRef:
        path = root / name
        path.write_bytes(_EVAL_CHECKPOINT_BYTES)
        return CheckpointRef(
            f"artifact://test/{name}",
            _EVAL_CHECKPOINT_HASH,
            role,
            sealed=True,
            metadata={"local_path": str(path)},
        )

    def _checkpoint_cli_argv(
        self,
        checkpoint: CheckpointRef,
        code: str,
        *,
        path_option: str = "--checkpoint",
        checksum_option: str = "--checkpoint-sha256",
        extra: tuple[str, ...] = (),
    ) -> list[str]:
        local_path = checkpoint.metadata["local_path"]
        return [
            sys.executable,
            "-c",
            code,
            path_option,
            str(local_path),
            checksum_option,
            checkpoint.checksum,
            *extra,
        ]

    def _sealed_validation_gate(
        self,
        *,
        status: str = "pass",
        candidate_isolation_status: str = "pass",
        checkpoint_safety_status: str = "pass",
        candidate_code_loaded: bool = False,
    ) -> dict:
        return {
            "status": status,
            "candidate_isolation_status": candidate_isolation_status,
            "checkpoint_safety_status": checkpoint_safety_status,
            "checkpoint_hash_match_status": "pass" if checkpoint_safety_status == "pass" else "fail",
            "eval_protocol_hash_match_status": "pass",
            "evaluation_binding_hash_match_status": "pass",
            "holdout_rollout_status": "pass",
            "candidate_code_loaded_in_eval": candidate_code_loaded,
            "candidate_package_loaded_in_eval": False,
            "candidate_path_argument_present": False,
            "reason": (
                "sealed_validation_complete"
                if status == "pass"
                else "sealed_validation_safety_failed"
            ),
        }

    def _sealed_validation_evidence(self, eval_spec: EvaluationJobSpec) -> dict:
        protocol_hash = eval_spec.evaluation_protocol().protocol_hash
        count = int(eval_spec.sealed_validation_seed_count)
        evidence = {
            "schema_version": "true_rl_agent.sealed_validation.evidence.v3",
            "status": "pass",
            "reason": "test_holdout_passed",
            "expected_seed_count": count,
            "observed_seed_count": count,
            "eval_protocol_hash": protocol_hash,
            "evaluation_binding_hash": eval_spec.evaluation_binding.binding_hash,
            "evaluation_subject": eval_spec.evaluation_subject().to_public_dict(),
            "subject_manifest_hash": eval_spec.evaluation_manifest_hash(),
            "program_source_checksum": (
                eval_spec.program_source_checksum
                if isinstance(eval_spec, EvalJobSpec)
                else None
            ),
            "training_binding_hash": (
                eval_spec.training_binding_hash
                if isinstance(eval_spec, EvalJobSpec)
                else None
            ),
            "capability_profile_hash": eval_spec.capability_profile_hash,
            "checkpoint_hash": eval_spec.checkpoint.checksum,
            "score_stats": {
                "count": count,
                "mean": 0.42,
                "std": 0.0,
                "standard_error": 0.0,
                "eval_protocol_hash": protocol_hash,
            },
        }
        return evidence

    def _run_eval_summary(
        self,
        summary: dict,
        *,
        exit_code: int = 0,
        include_sealed_validation_evidence: bool = True,
        task_evidence_hash_override: str | None = None,
    ) -> PublicEvalResult:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eval_summary = root / "eval_summary.json"
            runner = SubprocessRunner(allowed_executables=(sys.executable,), timeout_s=10)
            eval_spec = _eval_job_spec(
                job_id="eval-summary-probe",
                task_id="unitree_push_recovery",
                candidate_id="candidate-code",
                checkpoint=self._eval_checkpoint(root),
                public_seed_count=3,
                sealed_validation_seed_count=7,
                evaluation_binding=make_evaluation_binding(),
                metadata={"summary_path": str(eval_summary)},
            )
            bound_summary = {**self._eval_summary_binding(eval_spec), **summary}
            if include_sealed_validation_evidence:
                sealed_evidence = self._sealed_validation_evidence(eval_spec)
                raw_sealed_summary = bound_summary.get("sealed_validation_summary")
                if isinstance(raw_sealed_summary, dict):
                    raw_sealed_summary.setdefault(
                        "score_stats",
                        dict(sealed_evidence["score_stats"]),
                    )
                    sealed_evidence["schema_version"] = (
                        "true_rl_agent.sealed_validation.evidence.v4"
                    )
                    sealed_evidence["task_evidence_hash"] = (
                        task_evidence_hash_override
                        or canonical_json_hash(raw_sealed_summary)
                    )
                bound_summary.setdefault(
                    "sealed_validation_evidence",
                    sealed_evidence,
                )
            eval_code = textwrap.dedent(
                f"""
                from pathlib import Path
                Path({str(eval_summary)!r}).write_text({json.dumps(bound_summary)!r})
                raise SystemExit({exit_code})
                """
            )
            eval_spec = _with_eval_argv(
                eval_spec,
                self._checkpoint_cli_argv(
                    eval_spec.execution_checkpoint(),
                    eval_code,
                ),
            )

            return runner.run_eval(eval_spec)

    def _run_baseline_eval_summary(self, summary: dict) -> PublicEvalResult:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eval_summary = root / "baseline_eval_summary.json"
            runner = SubprocessRunner(
                allowed_executables=(sys.executable,),
                timeout_s=10,
            )
            eval_spec = _baseline_eval_job_spec(
                job_id="baseline-eval-summary-probe",
                task_id="unitree_push_recovery",
                baseline_id="official-g1-rough",
                baseline_manifest_hash="c" * 64,
                checkpoint=self._eval_checkpoint(
                    root,
                    role="warm_start",
                    name="baseline-input.pt",
                ),
                public_seed_count=3,
                sealed_validation_seed_count=7,
                evaluation_binding=make_evaluation_binding(),
                metadata={"summary_path": str(eval_summary)},
            )
            bound_summary = {**self._eval_summary_binding(eval_spec), **summary}
            bound_summary.setdefault(
                "sealed_validation_evidence",
                self._sealed_validation_evidence(eval_spec),
            )
            eval_code = textwrap.dedent(
                f"""
                from pathlib import Path
                Path({str(eval_summary)!r}).write_text({json.dumps(bound_summary)!r})
                """
            )
            eval_spec = _with_eval_argv(
                eval_spec,
                self._checkpoint_cli_argv(
                    eval_spec.execution_checkpoint(),
                    eval_code,
                ),
            )
            return runner.run_baseline_eval(eval_spec)

    def test_subprocess_runner_rejects_planned_checkpoint_before_launch(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / "subprocess-launched"
            eval_spec = _eval_job_spec(
                job_id="eval-planned",
                task_id="unitree_push_recovery",
                candidate_id="candidate-code",
                checkpoint=None,
                checkpoint_output_plan=CheckpointOutputPlan(
                    "artifact://round/checkpoints/best.pt"
                ),
                public_seed_count=3,
                sealed_validation_seed_count=7,
                evaluation_binding=make_evaluation_binding(),
                metadata={
                    "cli_argv": [
                        sys.executable,
                        "-c",
                        (
                            "from pathlib import Path; "
                            f"Path({str(marker)!r}).write_text('launched')"
                        ),
                    ]
                },
            )
            runner = SubprocessRunner(
                allowed_executables=(sys.executable,),
                timeout_s=10,
            )

            with self.assertRaisesRegex(
                RunnerExecutionError,
                "eval checkpoint is not executable",
            ):
                runner.run_eval(eval_spec)
            self.assertFalse(marker.exists())

    def test_train_rejects_duplicate_agent_configuration_before_launch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            marker = root / "duplicate-agent-config-launched.marker"
            code = (
                "from pathlib import Path; "
                f"Path({str(marker)!r}).write_text('launched')"
            )
            train_spec = _train_job_spec(
                job_id="duplicate-agent-configuration",
                task_id="unitree_push_recovery",
                candidate_id="candidate-code",
                policy_initialization=PolicyInitialization.random(),
                budget_steps=128,
                metadata={
                    "cli_argv": [
                        sys.executable,
                        "-c",
                        code,
                        "--agent-configuration-hash",
                        "9" * 64,
                    ],
                    "summary_path": str(root / "summary.json"),
                    "checkpoint_output_path": str(root / "trained.pt"),
                },
            )

            with self.assertRaisesRegex(
                RunnerExecutionError,
                "--agent-configuration-hash must appear exactly once",
            ):
                SubprocessRunner((sys.executable,), timeout_s=10).run_train(
                    train_spec
                )

            self.assertFalse(marker.exists())

    def test_train_rejects_execution_device_drift_before_launch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            marker = root / "train-device-drift-launched.marker"
            spec = _train_job_spec(
                job_id="train-device-drift",
                task_id="unitree_push_recovery",
                candidate_id="candidate-device-drift",
                policy_initialization=PolicyInitialization.random(),
                budget_steps=128,
                execution_device="cuda:1",
                metadata={
                    "cli_argv": [
                        sys.executable,
                        "-c",
                        f"from pathlib import Path; Path({str(marker)!r}).write_text('launched')",
                    ],
                    "summary_path": str(root / "summary.json"),
                    "checkpoint_output_path": str(root / "trained.pt"),
                },
            )
            metadata = dict(spec.metadata)
            argv = list(metadata["cli_argv"])
            argv[argv.index("--device") + 1] = "cuda:0"
            metadata["cli_argv"] = argv
            spec = replace(spec, metadata=metadata)

            with self.assertRaisesRegex(
                RunnerExecutionError,
                "--device does not match",
            ):
                SubprocessRunner((sys.executable,), timeout_s=10).run_train(spec)
            self.assertFalse(marker.exists())

    def test_eval_rejects_duplicate_execution_device_before_launch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner = SubprocessRunner((sys.executable,), timeout_s=10)
            cases = (
                _eval_job_spec(
                    job_id="candidate-eval-device-duplicate",
                    task_id="unitree_push_recovery",
                    candidate_id="candidate-device-duplicate",
                    checkpoint=self._eval_checkpoint(root),
                    public_seed_count=3,
                    sealed_validation_seed_count=7,
                    evaluation_binding=make_evaluation_binding(),
                    metadata={
                        "cli_argv": [sys.executable, "-c", "pass"],
                        "summary_path": str(root / "candidate-summary.json"),
                    },
                ),
                _baseline_eval_job_spec(
                    job_id="baseline-eval-device-duplicate",
                    task_id="unitree_push_recovery",
                    baseline_id="official-g1-rough",
                    baseline_manifest_hash="c" * 64,
                    checkpoint=self._eval_checkpoint(
                        root,
                        role="warm_start",
                        name="baseline-device-input.pt",
                    ),
                    public_seed_count=3,
                    sealed_validation_seed_count=7,
                    evaluation_binding=make_evaluation_binding(),
                    metadata={
                        "cli_argv": [sys.executable, "-c", "pass"],
                        "summary_path": str(root / "baseline-summary.json"),
                    },
                ),
            )
            for spec in cases:
                with self.subTest(subject_kind=spec.subject.kind):
                    metadata = dict(spec.metadata)
                    metadata["cli_argv"] = [
                        *metadata["cli_argv"],
                        "--device",
                        spec.execution_device,
                    ]
                    drifted = replace(spec, metadata=metadata)
                    with self.assertRaisesRegex(
                        RunnerExecutionError,
                        "--device must appear exactly once",
                    ):
                        if isinstance(drifted, EvalJobSpec):
                            runner.run_eval(drifted)
                        else:
                            runner.run_baseline_eval(drifted)

    def test_train_rejects_mutated_candidate_artifact_before_launch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            marker = root / "candidate-mutation-launched.marker"
            spec = _train_job_spec(
                job_id="candidate-mutation-preflight",
                task_id="unitree_push_recovery",
                candidate_id="candidate-mutation-probe",
                policy_initialization=PolicyInitialization.random(),
                budget_steps=128,
                metadata={
                    "cli_argv": [
                        sys.executable,
                        "-c",
                        f"from pathlib import Path; Path({str(marker)!r}).write_text('launched')",
                    ],
                    "summary_path": str(root / "summary.json"),
                    "checkpoint_output_path": str(root / "best.pt"),
                },
            )
            source = (
                Path(spec.candidate_artifact.path)
                / "training_code"
                / "components.py"
            )
            source.write_text("def build_training_components():\n    return None\n")

            with self.assertRaisesRegex(
                RunnerExecutionError,
                "candidate artifact file evidence mismatch",
            ):
                SubprocessRunner((sys.executable,), timeout_s=10).run_train(spec)
            self.assertFalse(marker.exists())

    def test_train_rejects_forged_program_checksum_argv_before_launch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            marker = root / "source-checksum-launched.marker"
            spec = _train_job_spec(
                job_id="source-checksum-preflight",
                task_id="unitree_push_recovery",
                candidate_id="candidate-source-checksum-probe",
                policy_initialization=PolicyInitialization.random(),
                budget_steps=128,
                metadata={
                    "cli_argv": [
                        sys.executable,
                        "-c",
                        f"from pathlib import Path; Path({str(marker)!r}).write_text('launched')",
                    ],
                    "summary_path": str(root / "summary.json"),
                    "checkpoint_output_path": str(root / "best.pt"),
                },
            )
            metadata = dict(spec.metadata)
            argv = list(metadata["cli_argv"])
            index = argv.index("--program-source-checksum")
            argv[index + 1] = "sha256:" + "0" * 64
            metadata["cli_argv"] = argv
            spec = replace(spec, metadata=metadata)

            with self.assertRaisesRegex(
                RunnerExecutionError,
                "--program-source-checksum does not match",
            ):
                SubprocessRunner((sys.executable,), timeout_s=10).run_train(spec)
            self.assertFalse(marker.exists())

    def test_candidate_eval_requires_program_checksum_before_launch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            marker = root / "eval-source-checksum-launched.marker"
            spec = _eval_job_spec(
                job_id="eval-source-checksum-preflight",
                task_id="unitree_push_recovery",
                candidate_id="candidate-code",
                checkpoint=self._eval_checkpoint(root),
                public_seed_count=3,
                sealed_validation_seed_count=7,
                evaluation_binding=make_evaluation_binding(),
                metadata={
                    "cli_argv": self._checkpoint_cli_argv(
                        self._eval_checkpoint(
                            root,
                            name="second-eval-input.pt",
                        ),
                        f"from pathlib import Path; Path({str(marker)!r}).write_text('launched')",
                    ),
                    "summary_path": str(root / "summary.json"),
                },
            )
            checkpoint = spec.execution_checkpoint()
            argv = self._checkpoint_cli_argv(
                checkpoint,
                f"from pathlib import Path; Path({str(marker)!r}).write_text('launched')",
            )
            spec = _with_eval_argv(spec, argv)
            metadata = dict(spec.metadata)
            argv = list(metadata["cli_argv"])
            index = argv.index("--program-source-checksum")
            del argv[index : index + 2]
            metadata["cli_argv"] = argv
            spec = replace(spec, metadata=metadata)

            with self.assertRaisesRegex(
                RunnerExecutionError,
                "--program-source-checksum must appear exactly once",
            ):
                SubprocessRunner((sys.executable,), timeout_s=10).run_eval(spec)
            self.assertFalse(marker.exists())

    def _run_train_summary(
        self,
        summary: dict,
        *,
        checkpoint_bytes: bytes | None = None,
        exit_code: int = 0,
        expected_initial_policy_state_hash: str | None = None,
        candidate_failure_contract: CandidateFailureContract | None = None,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_summary = root / "train_summary.json"
            checkpoint_output = root / "best.pt"
            runner = SubprocessRunner(
                allowed_executables=(sys.executable,),
                timeout_s=10,
            )
            train_spec = _train_job_spec(
                job_id="train-summary-probe",
                task_id="unitree_push_recovery",
                candidate_id="candidate-code",
                policy_initialization=PolicyInitialization.random(),
                budget_steps=128,
                expected_initial_policy_state_hash=(
                    expected_initial_policy_state_hash
                ),
                candidate_failure_contract=candidate_failure_contract,
                metadata={
                    "summary_path": str(train_summary),
                    "sealed_checkpoint_uri": "artifact://train/sealed/best.pt",
                    "checkpoint_output_path": str(checkpoint_output),
                },
            )
            bound_summary = {**self._train_summary_binding(train_spec), **summary}
            checkpoint_write = ""
            if checkpoint_bytes is not None:
                checksum = "sha256:" + hashlib.sha256(checkpoint_bytes).hexdigest()
                bound_summary["checkpoint"] = {
                    "uri": str(checkpoint_output),
                    "checksum": checksum,
                    "role": "train_output",
                    "metadata": {"local_path": str(checkpoint_output)},
                }
                checkpoint_write = (
                    f"Path({str(checkpoint_output)!r}).write_bytes({checkpoint_bytes!r})"
                )
            train_code = textwrap.dedent(
                f"""
                import json
                from pathlib import Path
                {checkpoint_write}
                Path({str(train_summary)!r}).write_text({json.dumps(bound_summary)!r})
                raise SystemExit({exit_code})
                """
            )
            train_spec = _with_train_argv(
                train_spec,
                [sys.executable, "-c", train_code],
            )
            return runner.run_train(train_spec)

    def _train_summary_binding(self, train_spec: TrainJobSpec) -> dict:
        result = {
            **_expected_train_summary_binding(
                job_id=train_spec.job_id,
                candidate_id=train_spec.subject.subject_id,
                budget_steps=train_spec.budget_steps,
            ),
        }
        if train_spec.candidate_failure_contract is not None:
            result["candidate_failure_contract_hash"] = (
                train_spec.candidate_failure_contract.contract_hash
            )
        return result

    def _eval_summary_binding(self, eval_spec: EvaluationJobSpec) -> dict:
        return {
            **eval_spec.job_identity().to_public_dict(),
            "evaluation_protocol": eval_spec.evaluation_protocol().to_public_dict(),
            "evaluation_binding": eval_spec.evaluation_binding.to_public_dict(),
            "execution_device": eval_spec.execution_device,
        }

    def test_subprocess_runner_executes_baseline_eval_without_candidate_identity(
        self,
    ) -> None:
        result = self._run_baseline_eval_summary(
            {
                "status": "pass",
                "public_score": 0.42,
                "metrics": {"success_rate": 0.42},
                "candidate_code_loaded_in_eval": False,
                "candidate_package_loaded_in_eval": False,
                "checkpoint_safety_report": self._checkpoint_safety_report(),
                "sealed_validation_gate": self._sealed_validation_gate(),
            }
        )

        self.assertEqual(
            result.metrics["evaluation_subject"],
            {
                "schema_version": "true_rl_agent.experiment.subject.v2",
                "kind": "baseline",
                "id": "official-g1-rough",
            },
        )
        self.assertEqual(result.metrics["subject_manifest_hash"], "c" * 64)
        self.assertNotIn("candidate_id", result.to_public_dict())

    def test_subprocess_runner_preserves_task_owned_eval_metrics(self) -> None:
        source_preflight = {
            "status": "pass",
            "opaque_task_evidence": "preserved",
        }
        result = self._run_baseline_eval_summary(
            {
                "status": "pass",
                "public_score": 0.42,
                "metrics": {
                    "environment_source_preflight": source_preflight,
                },
                "candidate_code_loaded_in_eval": False,
                "candidate_package_loaded_in_eval": False,
                "checkpoint_safety_report": self._checkpoint_safety_report(),
                "sealed_validation_gate": self._sealed_validation_gate(),
            }
        )

        self.assertEqual(
            result.metrics["environment_source_preflight"],
            source_preflight,
        )

    def test_subprocess_runner_accepts_existing_summary_without_optional_diagnostics(
        self,
    ) -> None:
        """Task-local additive artifacts must not become a shared summary requirement."""

        result = self._run_eval_summary(
            {
                "status": "pass",
                "public_score": 0.42,
                "metrics": {"success_rate": 0.42},
                "checkpoint_safety_report": self._checkpoint_safety_report(),
                "sealed_validation_gate": self._sealed_validation_gate(),
            }
        )

        self.assertEqual(result.public_score, 0.42)
        self.assertEqual(result.metrics["success_rate"], 0.42)

    def test_unbound_eval_rejects_summary_gpu_process_binding_claim(self) -> None:
        with self.assertRaisesRegex(
            RunnerExecutionError,
            "unbound summary cannot claim a GPU process binding",
        ) as raised:
            self._run_eval_summary(
                {
                    "status": "pass",
                    "public_score": 0.42,
                    "metrics": {"success_rate": 0.42},
                    "gpu_process_binding": {
                        "host_execution_device": "cuda:1",
                        "process_execution_device": "cuda:0",
                        "resource_binding_hash": "a" * 64,
                    },
                    "checkpoint_safety_report": self._checkpoint_safety_report(),
                    "sealed_validation_gate": self._sealed_validation_gate(),
                }
            )
        self.assertEqual(
            raised.exception.failure_kind,
            RunnerFailureKind.SUMMARY_INVALID,
        )

    def test_subprocess_runner_rejects_baseline_manifest_or_subject_drift(
        self,
    ) -> None:
        base_summary = {
            "status": "pass",
            "public_score": 0.42,
            "metrics": {"success_rate": 0.42},
            "candidate_code_loaded_in_eval": False,
            "candidate_package_loaded_in_eval": False,
            "checkpoint_safety_report": self._checkpoint_safety_report(),
            "sealed_validation_gate": self._sealed_validation_gate(),
        }
        for field, value in (
            ("subject_manifest_hash", "d" * 64),
            ("subject_kind", "candidate"),
            ("subject_id", "candidate-001"),
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(
                    RunnerExecutionError,
                    f"binding mismatch for {field}",
                ):
                    self._run_baseline_eval_summary(
                        {**base_summary, field: value}
                    )

    def test_subprocess_runner_rejects_nested_second_eval_identity(self) -> None:
        base_summary = {
            "status": "pass",
            "public_score": 0.42,
            "metrics": {"success_rate": 0.42},
            "candidate_code_loaded_in_eval": False,
            "candidate_package_loaded_in_eval": False,
            "checkpoint_safety_report": self._checkpoint_safety_report(),
            "sealed_validation_gate": self._sealed_validation_gate(),
        }
        forged_subject = {
            "schema_version": "true_rl_agent.experiment.subject.v2",
            "kind": "candidate",
            "id": "forged-candidate",
        }
        for field in ("evaluation_subject", "subject"):
            with self.subTest(field=field):
                with self.assertRaisesRegex(
                    RunnerExecutionError,
                    f"alternate job identity field {field}",
                ):
                    self._run_baseline_eval_summary(
                        {**base_summary, field: forged_subject}
                    )
        for field, value in (
            ("candidate_id", "forged-candidate"),
            ("subject", forged_subject),
            ("evaluation_identity", forged_subject),
        ):
            with self.subTest(metrics_field=field):
                with self.assertRaisesRegex(
                    RunnerExecutionError,
                    f"eval summary metrics must not contain {field}",
                ):
                    self._run_baseline_eval_summary(
                        {
                            **base_summary,
                            "metrics": {
                                "success_rate": 0.42,
                                field: value,
                            },
                        }
                    )

    def test_subprocess_runner_rejects_checkpoint_bytes_before_launch(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint_path = root / "baseline.pt"
            checkpoint_path.write_bytes(b"wrong-checkpoint-bytes")
            marker_path = root / "subprocess-launched.marker"
            checkpoint = CheckpointRef(
                "baseline://official-g1-rough/policy.pt",
                "sha256:" + "0" * 64,
                "warm_start",
                sealed=True,
                metadata={"local_path": str(checkpoint_path)},
            )
            marker_code = (
                "from pathlib import Path; "
                f"Path({str(marker_path)!r}).write_text('launched')"
            )
            eval_spec = _baseline_eval_job_spec(
                job_id="baseline-preflight",
                task_id="unitree_push_recovery",
                baseline_id="official-g1-rough",
                baseline_manifest_hash="c" * 64,
                checkpoint=checkpoint,
                public_seed_count=3,
                sealed_validation_seed_count=7,
                evaluation_binding=make_evaluation_binding(),
                metadata={
                    "cli_argv": self._checkpoint_cli_argv(
                        checkpoint,
                        marker_code,
                    ),
                    "summary_path": str(root / "summary.json"),
                },
            )

            with self.assertRaisesRegex(
                RunnerExecutionError,
                "checkpoint preflight failed: "
                "checkpoint input checksum does not match bytes",
            ):
                SubprocessRunner(
                    allowed_executables=(sys.executable,),
                    timeout_s=10,
                ).run_baseline_eval(eval_spec)

            self.assertFalse(marker_path.exists())

    def test_subprocess_runner_rejects_warm_start_bytes_before_train_launch(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint_path = root / "baseline.pt"
            checkpoint_path.write_bytes(b"wrong-checkpoint-bytes")
            marker_path = root / "subprocess-launched.marker"
            checkpoint = CheckpointRef(
                uri="artifact://baseline/official-g1-rough.pt",
                checksum="sha256:" + hashlib.sha256(b"expected-bytes").hexdigest(),
                role="warm_start",
                sealed=True,
                metadata={"local_path": str(checkpoint_path)},
            )
            marker_code = (
                "from pathlib import Path; "
                f"Path({str(marker_path)!r}).write_text('launched')"
            )
            train_spec = _train_job_spec(
                job_id="warm-start-preflight",
                task_id="unitree_push_recovery",
                candidate_id="candidate-code",
                policy_initialization=PolicyInitialization.warm_start(
                    BaselinePolicyRef(
                        baseline_id="official-g1-rough",
                        task_id="unitree_push_recovery",
                        checkpoint=checkpoint,
                        baseline_manifest_hash="d" * 64,
                    )
                ),
                budget_steps=128,
                metadata={
                    "cli_argv": self._checkpoint_cli_argv(
                        checkpoint,
                        marker_code,
                        path_option="--init-checkpoint",
                        checksum_option="--init-checkpoint-sha256",
                    ),
                    "summary_path": str(root / "summary.json"),
                    "checkpoint_output_path": str(root / "trained.pt"),
                },
            )

            with self.assertRaisesRegex(
                RunnerExecutionError,
                "checkpoint preflight failed: "
                "checkpoint input checksum does not match bytes",
            ):
                SubprocessRunner(
                    allowed_executables=(sys.executable,),
                    timeout_s=10,
                ).run_train(train_spec)

            self.assertFalse(marker_path.exists())

    def test_candidate_eval_rejects_duplicate_checkpoint_path_before_launch(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            marker = root / "candidate-eval-launched.marker"
            checkpoint = self._eval_checkpoint(root)
            code = (
                "from pathlib import Path; "
                f"Path({str(marker)!r}).write_text('launched')"
            )
            eval_spec = _eval_job_spec(
                job_id="candidate-eval-argv-drift",
                task_id="unitree_push_recovery",
                candidate_id="candidate-code",
                checkpoint=checkpoint,
                public_seed_count=3,
                sealed_validation_seed_count=7,
                evaluation_binding=make_evaluation_binding(),
                metadata={
                    "cli_argv": self._checkpoint_cli_argv(
                        checkpoint,
                        code,
                        extra=("--checkpoint", str(root / "wrong.pt")),
                    ),
                    "summary_path": str(root / "summary.json"),
                },
            )

            with self.assertRaisesRegex(
                RunnerExecutionError,
                "--checkpoint must appear exactly once",
            ):
                SubprocessRunner((sys.executable,), timeout_s=10).run_eval(
                    eval_spec
                )

            self.assertFalse(marker.exists())

    def test_baseline_eval_rejects_duplicate_checkpoint_hash_before_launch(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            marker = root / "baseline-eval-launched.marker"
            checkpoint = self._eval_checkpoint(
                root,
                role="warm_start",
                name="baseline-input.pt",
            )
            code = (
                "from pathlib import Path; "
                f"Path({str(marker)!r}).write_text('launched')"
            )
            eval_spec = _baseline_eval_job_spec(
                job_id="baseline-eval-argv-drift",
                task_id="unitree_push_recovery",
                baseline_id="official-g1-rough",
                baseline_manifest_hash="c" * 64,
                checkpoint=checkpoint,
                public_seed_count=3,
                sealed_validation_seed_count=7,
                evaluation_binding=make_evaluation_binding(),
                metadata={
                    "cli_argv": self._checkpoint_cli_argv(
                        checkpoint,
                        code,
                        extra=("--checkpoint-sha256", _WRONG_CHECKPOINT_HASH),
                    ),
                    "summary_path": str(root / "summary.json"),
                },
            )

            with self.assertRaisesRegex(
                RunnerExecutionError,
                "--checkpoint-sha256 must appear exactly once",
            ):
                SubprocessRunner(
                    (sys.executable,),
                    timeout_s=10,
                ).run_baseline_eval(eval_spec)

            self.assertFalse(marker.exists())

    def test_warm_start_train_rejects_checkpoint_path_drift_before_launch(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            marker = root / "warm-start-train-launched.marker"
            checkpoint = self._eval_checkpoint(
                root,
                role="warm_start",
                name="warm-start.pt",
            )
            code = (
                "from pathlib import Path; "
                f"Path({str(marker)!r}).write_text('launched')"
            )
            argv = self._checkpoint_cli_argv(
                checkpoint,
                code,
                path_option="--init-checkpoint",
                checksum_option="--init-checkpoint-sha256",
            )
            argv[argv.index("--init-checkpoint") + 1] = str(root / "wrong.pt")
            train_spec = _train_job_spec(
                job_id="warm-start-train-argv-drift",
                task_id="unitree_push_recovery",
                candidate_id="candidate-code",
                policy_initialization=PolicyInitialization.warm_start(
                    BaselinePolicyRef(
                        baseline_id="official-g1-rough",
                        task_id="unitree_push_recovery",
                        checkpoint=checkpoint,
                        baseline_manifest_hash="d" * 64,
                    )
                ),
                budget_steps=128,
                metadata={
                    "cli_argv": argv,
                    "summary_path": str(root / "summary.json"),
                    "checkpoint_output_path": str(root / "trained.pt"),
                },
            )

            with self.assertRaisesRegex(
                RunnerExecutionError,
                "--init-checkpoint does not match",
            ):
                SubprocessRunner((sys.executable,), timeout_s=10).run_train(
                    train_spec
                )

            self.assertFalse(marker.exists())

    def test_random_train_rejects_hidden_warm_start_args_before_launch(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            marker = root / "random-train-launched.marker"
            code = (
                "from pathlib import Path; "
                f"Path({str(marker)!r}).write_text('launched')"
            )
            train_spec = _train_job_spec(
                job_id="random-train-hidden-warm-start",
                task_id="unitree_push_recovery",
                candidate_id="candidate-code",
                policy_initialization=PolicyInitialization.random(),
                budget_steps=128,
                metadata={
                    "cli_argv": [
                        sys.executable,
                        "-c",
                        code,
                        "--init-checkpoint",
                        str(root / "hidden.pt"),
                        "--init-checkpoint-sha256",
                        _WRONG_CHECKPOINT_HASH,
                    ],
                    "summary_path": str(root / "summary.json"),
                    "checkpoint_output_path": str(root / "trained.pt"),
                },
            )

            with self.assertRaisesRegex(
                RunnerExecutionError,
                "--init-checkpoint must be absent",
            ):
                SubprocessRunner((sys.executable,), timeout_s=10).run_train(
                    train_spec
                )

            self.assertFalse(marker.exists())

    def test_subprocess_runner_rejects_eval_argv_candidate_path_before_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eval_summary = root / "eval_summary.json"
            runner = SubprocessRunner(allowed_executables=(sys.executable,), timeout_s=10)
            eval_spec = _eval_job_spec(
                job_id="eval-forbidden-candidate-path",
                task_id="unitree_push_recovery",
                candidate_id="candidate-code",
                checkpoint=self._eval_checkpoint(root),
                public_seed_count=3,
                sealed_validation_seed_count=7,
                evaluation_binding=make_evaluation_binding(),
                metadata={"summary_path": str(eval_summary)},
            )
            payload = {
                **self._eval_summary_binding(eval_spec),
                "status": "pass",
                "public_score": 0.9,
                "metrics": {"success_rate": 0.9},
                "checkpoint_safety_report": self._checkpoint_safety_report(),
                "sealed_validation_gate": self._sealed_validation_gate(),
            }
            eval_code = textwrap.dedent(
                f"""
                import json
                from pathlib import Path
                Path({str(eval_summary)!r}).write_text(json.dumps({payload!r}))
                """
            )
            eval_spec = _with_eval_argv(
                eval_spec,
                self._checkpoint_cli_argv(
                    eval_spec.execution_checkpoint(),
                    eval_code,
                    extra=("--candidate-dir", "C:/candidate"),
                ),
            )

            with self.assertRaisesRegex(RunnerExecutionError, "candidate path argument is not allowed"):
                runner.run_eval(eval_spec)
            self.assertFalse(eval_summary.exists())

    def test_subprocess_runner_executes_train_and_eval_from_structured_argv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_summary = root / "train_summary.json"
            eval_summary = root / "eval_summary.json"
            checkpoint_output = root / "train" / "best.pt"
            checkpoint_bytes = b"train-checkpoint"
            checkpoint_checksum = "sha256:" + hashlib.sha256(checkpoint_bytes).hexdigest()
            train_code = textwrap.dedent(
                f"""
                import json
                from pathlib import Path
                checkpoint = Path({str(checkpoint_output)!r})
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                checkpoint.write_bytes({checkpoint_bytes!r})
                Path({str(train_summary)!r}).write_text(json.dumps({{
                    **{_expected_train_summary_binding(job_id="train-001", candidate_id="candidate-code")!r},
                    "status": "pass",
                    "checkpoint": {{
                        "uri": {str(checkpoint_output)!r},
                        "checksum": {checkpoint_checksum!r},
                        "role": "train_output",
                        "metadata": {{"local_path": {str(checkpoint_output)!r}}}
                    }},
                    "runtime_activation_summary": {{
                        "reward_term_calls": 3,
                        "nonzero_reward_contribution": True
                    }}
                }}))
                """
            )
            eval_payload = {
                "status": "pass",
                "public_score": 0.42,
                "metrics": {"success_rate": 0.42},
                "candidate_code_loaded_in_eval": False,
                "candidate_package_loaded": False,
                "checkpoint_safety_report": self._checkpoint_safety_report(),
                "sealed_validation_gate": self._sealed_validation_gate(),
            }
            eval_code = textwrap.dedent(
                f"""
                import json
                from pathlib import Path
                Path({str(eval_summary)!r}).write_text({json.dumps(eval_payload)!r})
                """
            )
            runner = SubprocessRunner(allowed_executables=(sys.executable,), timeout_s=10)
            train_spec = _train_job_spec(
                job_id="train-001",
                task_id="unitree_push_recovery",
                candidate_id="candidate-code",
                policy_initialization=PolicyInitialization.random(),
                budget_steps=128,
                metadata={
                    "cli_argv": [sys.executable, "-c", train_code],
                    "summary_path": str(train_summary),
                    "sealed_checkpoint_uri": "artifact://train/sealed/best.pt",
                    "checkpoint_output_path": str(checkpoint_output),
                },
            )
            eval_spec = _eval_job_spec(
                job_id="eval-001",
                task_id="unitree_push_recovery",
                candidate_id="candidate-code",
                checkpoint=self._eval_checkpoint(root),
                public_seed_count=3,
                sealed_validation_seed_count=7,
                evaluation_binding=make_evaluation_binding(),
                metadata={"summary_path": str(eval_summary)},
            )
            eval_payload["sealed_validation_evidence"] = self._sealed_validation_evidence(
                eval_spec
            )
            eval_payload = {**self._eval_summary_binding(eval_spec), **eval_payload}
            eval_code = textwrap.dedent(
                f"""
                import json
                from pathlib import Path
                Path({str(eval_summary)!r}).write_text({json.dumps(eval_payload)!r})
                """
            )
            eval_spec = _with_eval_argv(
                eval_spec,
                self._checkpoint_cli_argv(
                    eval_spec.execution_checkpoint(),
                    eval_code,
                ),
            )

            train_result = runner.run_train(train_spec)
            public_result = runner.run_eval(eval_spec)

        self.assertEqual(train_result.checkpoint.role, "eval")
        self.assertTrue(train_result.checkpoint.sealed)
        self.assertEqual(train_result.checkpoint.uri, "artifact://train/sealed/best.pt")
        self.assertEqual(train_result.summary["runtime_activation_summary"]["reward_term_calls"], 3)
        self.assertIsInstance(public_result, PublicEvalResult)
        self.assertEqual(public_result.public_score, 0.42)
        self.assertEqual(public_result.metrics["success_rate"], 0.42)
        self.assertIs(public_result.metrics["candidate_code_loaded_in_eval"], False)
        self.assertIs(public_result.metrics["candidate_package_loaded"], False)
        self.assertEqual(public_result.metrics["checkpoint_safety_status"], "pass")
        self.assertEqual(public_result.metrics["execution_device"], "cuda:0")

    def test_subprocess_runner_wraps_malformed_eval_scalar_and_status_fields(self) -> None:
        cases = {
            "bool_score": ("pass", True, {}),
            "nan_score": ("pass", float("nan"), {}),
            "infinite_score": ("pass", float("inf"), {}),
            "string_score": ("pass", "0.42", {}),
            "bad_status": ("banana", 0.42, {}),
            "bool_status": (True, 0.42, {}),
            "bad_metrics": ("pass", 0.42, []),
        }
        for label, (status, public_score, metrics) in cases.items():
            with self.subTest(label=label):
                summary = {
                    "status": status,
                    "public_score": public_score,
                    "metrics": metrics,
                    "checkpoint_safety_report": self._checkpoint_safety_report(),
                    "sealed_validation_gate": self._sealed_validation_gate(),
                }

                with self.assertRaises(RunnerExecutionError):
                    self._run_eval_summary(summary)

    def test_subprocess_runner_strictly_validates_train_status(self) -> None:
        checkpoint = {
            "uri": "artifact://train/raw.pt",
            "checksum": _TRAIN_CHECKPOINT_HASH,
            "role": "train_output",
        }
        for status in (None, True, "banana"):
            with self.subTest(status=status):
                summary = {"checkpoint": checkpoint}
                if status is not None:
                    summary["status"] = status
                with self.assertRaisesRegex(
                    RunnerExecutionError,
                    "train summary status must be pass or fail",
                ):
                    self._run_train_summary(summary)

    def test_subprocess_runner_does_not_coerce_checkpoint_fields(self) -> None:
        valid_checkpoint = {
            "uri": "artifact://train/raw.pt",
            "checksum": _TRAIN_CHECKPOINT_HASH,
            "role": "train_output",
            "sealed": False,
            "metadata": {},
        }
        mutations = {
            "none_uri": {"uri": None},
            "bool_checksum": {"checksum": True},
            "wrong_checksum_scheme": {"checksum": "md5:train"},
            "bool_role": {"role": True},
            "wrong_role": {"role": "eval"},
            "string_sealed": {"sealed": "false"},
            "presealed": {"sealed": True},
            "none_metadata": {"metadata": None},
        }
        for label, update in mutations.items():
            with self.subTest(label=label):
                checkpoint = {**valid_checkpoint, **update}
                with self.assertRaisesRegex(
                    RunnerExecutionError,
                    "invalid train summary checkpoint",
                ):
                    self._run_train_summary(
                        {"status": "pass", "checkpoint": checkpoint}
                    )

    def test_subprocess_runner_accepts_train_summary_bound_to_current_job(self) -> None:
        checkpoint_bytes = b"current-checkpoint"
        expected_checksum = "sha256:" + hashlib.sha256(checkpoint_bytes).hexdigest()
        result = self._run_train_summary(
            {
                "status": "pass",
                "checkpoint": {
                    "uri": "artifact://train/current.pt",
                    "checksum": _CURRENT_CHECKPOINT_HASH,
                    "role": "train_output",
                },
                "runtime_activation_summary": {
                    "reward_term_calls": 7,
                    "nonzero_reward_contribution": True,
                },
            },
            checkpoint_bytes=checkpoint_bytes,
        )

        self.assertEqual(result.job_id, "train-summary-probe")
        self.assertEqual(result.checkpoint.checksum, expected_checksum)
        self.assertEqual(result.summary["job_id"], "train-summary-probe")
        self.assertEqual(result.summary["task_id"], "unitree_push_recovery")
        self.assertEqual(result.summary["subject_kind"], "candidate")
        self.assertEqual(result.summary["subject_id"], "candidate-code")
        self.assertEqual(
            result.summary["runtime_activation_summary"]["reward_term_calls"],
            7,
        )

    def test_unbound_train_rejects_summary_gpu_process_binding_claim(self) -> None:
        with self.assertRaisesRegex(
            RunnerExecutionError,
            "unbound summary cannot claim a GPU process binding",
        ) as raised:
            self._run_train_summary(
                {
                    "status": "pass",
                    "gpu_process_binding": {
                        "host_execution_device": "cuda:1",
                        "process_execution_device": "cuda:0",
                        "resource_binding_hash": "a" * 64,
                    },
                },
                checkpoint_bytes=b"reverse-bound-drift",
            )
        self.assertEqual(
            raised.exception.failure_kind,
            RunnerFailureKind.SUMMARY_INVALID,
        )

    def test_initial_policy_state_pairing_is_bound_to_train_result(self) -> None:
        expected = "a" * 64
        result = self._run_train_summary(
            {
                "status": "pass",
                "training": {
                    "expected_initial_policy_state_hash": expected,
                    "initial_policy_state_hash": expected,
                    "initial_policy_state_match": True,
                },
            },
            checkpoint_bytes=b"paired-checkpoint",
            expected_initial_policy_state_hash=expected,
        )

        self.assertEqual(
            result.initial_policy_state_pairing.expected_initial_policy_state_hash,
            expected,
        )
        self.assertEqual(
            result.checkpoint.metadata["initial_policy_state_pairing"],
            result.initial_policy_state_pairing.to_public_dict(),
        )

    def test_initial_policy_state_pairing_rejects_missing_or_mismatched_summary(
        self,
    ) -> None:
        expected = "a" * 64
        cases = {
            "missing": None,
            "mismatch": {
                "expected_initial_policy_state_hash": expected,
                "initial_policy_state_hash": "b" * 64,
                "initial_policy_state_match": False,
            },
            "forged_match": {
                "expected_initial_policy_state_hash": expected,
                "initial_policy_state_hash": "b" * 64,
                "initial_policy_state_match": True,
            },
        }
        for label, training in cases.items():
            with self.subTest(label=label):
                payload = {
                    "status": "fail",
                    "candidate_failure": (
                        _TEST_CANDIDATE_FAILURE_CONTRACT.evidence(
                            stage="ppo_reward_execution",
                            reason_code="runtime_rejected",
                        ).to_public_dict()
                    ),
                    "errors": [],
                }
                if training is not None:
                    payload["training"] = training
                with self.assertRaisesRegex(
                    RunnerExecutionError,
                    "initial policy state pairing failed",
                ):
                    self._run_train_summary(
                        payload,
                        exit_code=2,
                        expected_initial_policy_state_hash=expected,
                        candidate_failure_contract=(
                            _TEST_CANDIDATE_FAILURE_CONTRACT
                        ),
                    )

    def test_subprocess_runner_rejects_train_summary_identity_mismatch_before_checkpoint_parse(self) -> None:
        mismatches = {
            "job_id": "stale-train-job",
            "task_id": "different-task",
            "subject_id": "different-candidate",
        }
        for field, wrong_value in mismatches.items():
            with self.subTest(field=field):
                with self.assertRaisesRegex(
                    RunnerExecutionError,
                    rf"summary binding mismatch.*{field}",
                ) as context:
                    self._run_train_summary(
                        {
                            field: wrong_value,
                            "status": "pass",
                            # Deliberately malformed: identity must be rejected before
                            # checkpoint or runtime evidence can be consumed.
                            "checkpoint": None,
                            "runtime_activation_summary": {
                                "reward_term_calls": 999,
                            },
                        }
                    )

                self.assertNotIn("checkpoint", str(context.exception).lower())

    def test_subprocess_runner_rejects_execution_device_summary_drift(self) -> None:
        with self.assertRaisesRegex(
            RunnerExecutionError,
            "execution_device",
        ):
            self._run_train_summary(
                {
                    "execution_device": "cuda:1",
                    "status": "fail",
                }
            )

        with self.assertRaisesRegex(
            RunnerExecutionError,
            "execution_device",
        ):
            self._run_eval_summary(
                {
                    "execution_device": "cuda:1",
                    "status": "fail",
                    "public_score": 0.0,
                    "metrics": {},
                }
            )

    def test_subprocess_runner_deletes_stale_train_summary_before_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            train_summary = Path(tmp) / "train_summary.json"
            train_summary.write_text(
                json.dumps(
                    {
                        "job_id": "old-train-job",
                        "task_id": "unitree_push_recovery",
                        "candidate_id": "old-candidate",
                        "status": "pass",
                        "checkpoint": {
                            "uri": "artifact://old/raw.pt",
                            "checksum": _OLD_CHECKPOINT_HASH,
                            "role": "train_output",
                        },
                        "runtime_activation_summary": {
                            "reward_term_calls": 999,
                        },
                    }
                ),
                encoding="utf-8",
            )
            runner = SubprocessRunner(
                allowed_executables=(sys.executable,),
                timeout_s=10,
            )
            train_spec = _train_job_spec(
                job_id="new-train-job",
                task_id="unitree_push_recovery",
                candidate_id="new-candidate",
                policy_initialization=PolicyInitialization.random(),
                budget_steps=128,
                metadata={
                    "cli_argv": [sys.executable, "-c", "pass"],
                    "summary_path": str(train_summary),
                    "sealed_checkpoint_uri": "artifact://train/sealed/current.pt",
                    "checkpoint_output_path": str(Path(tmp) / "current.pt"),
                },
            )

            with self.assertRaisesRegex(
                RunnerExecutionError,
                "summary file was not created",
            ):
                runner.run_train(train_spec)

            self.assertFalse(train_summary.exists())

    def test_subprocess_runner_binds_train_checkpoint_to_planned_path_and_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary_path = root / "train_summary.json"
            checkpoint_path = root / "checkpoints" / "best.pt"
            checkpoint_bytes = b"fresh-checkpoint"
            checksum = "sha256:" + hashlib.sha256(checkpoint_bytes).hexdigest()
            payload = {
                **_expected_train_summary_binding(
                    job_id="train-bound-output",
                    candidate_id="candidate-code",
                ),
                "status": "pass",
                "checkpoint": {
                    "uri": str(checkpoint_path),
                    "checksum": checksum,
                    "role": "train_output",
                    "metadata": {"local_path": str(checkpoint_path)},
                },
            }
            train_code = textwrap.dedent(
                f"""
                import json
                from pathlib import Path
                checkpoint = Path({str(checkpoint_path)!r})
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                checkpoint.write_bytes({checkpoint_bytes!r})
                Path({str(summary_path)!r}).write_text(json.dumps({payload!r}))
                """
            )
            spec = _train_job_spec(
                job_id="train-bound-output",
                task_id="unitree_push_recovery",
                candidate_id="candidate-code",
                policy_initialization=PolicyInitialization.random(),
                budget_steps=128,
                metadata={
                    "cli_argv": [sys.executable, "-c", train_code],
                    "summary_path": str(summary_path),
                    "checkpoint_output_path": str(checkpoint_path),
                    "sealed_checkpoint_uri": "artifact://train/sealed/best.pt",
                },
            )

            result = SubprocessRunner(
                allowed_executables=(sys.executable,),
                timeout_s=10,
            ).run_train(spec)

        self.assertEqual(result.checkpoint.checksum, checksum)
        self.assertEqual(result.checkpoint.metadata["local_path"], str(checkpoint_path))

    def test_subprocess_runner_rejects_stale_or_mismatched_train_checkpoint_output(self) -> None:
        cases = {
            "missing_after_stale_clear": "was not created",
            "declared_path_mismatch": "path mismatch",
            "checksum_mismatch": "checksum does not match output bytes",
        }
        for label, expected_error in cases.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                summary_path = root / "train_summary.json"
                checkpoint_path = root / "checkpoints" / "best.pt"
                declared_path = (
                    root / "checkpoints" / "other.pt"
                    if label == "declared_path_mismatch"
                    else checkpoint_path
                )
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                checkpoint_path.write_bytes(b"stale-checkpoint")
                payload = {
                    **_expected_train_summary_binding(
                        job_id="train-output-reject",
                        candidate_id="candidate-code",
                    ),
                    "status": "pass",
                    "checkpoint": {
                        "uri": str(declared_path),
                        "checksum": _WRONG_CHECKPOINT_HASH,
                        "role": "train_output",
                        "metadata": {"local_path": str(declared_path)},
                    },
                }
                writes_output = label != "missing_after_stale_clear"
                train_code = textwrap.dedent(
                    f"""
                    import json
                    from pathlib import Path
                    checkpoint = Path({str(checkpoint_path)!r})
                    if {writes_output!r}:
                        checkpoint.parent.mkdir(parents=True, exist_ok=True)
                        checkpoint.write_bytes(b"fresh-checkpoint")
                    Path({str(summary_path)!r}).write_text(json.dumps({payload!r}))
                    """
                )
                spec = _train_job_spec(
                    job_id="train-output-reject",
                    task_id="unitree_push_recovery",
                    candidate_id="candidate-code",
                    policy_initialization=PolicyInitialization.random(),
                    budget_steps=128,
                    metadata={
                        "cli_argv": [sys.executable, "-c", train_code],
                        "summary_path": str(summary_path),
                        "checkpoint_output_path": str(checkpoint_path),
                    },
                )

                with self.assertRaisesRegex(RunnerExecutionError, expected_error):
                    SubprocessRunner(
                        allowed_executables=(sys.executable,),
                        timeout_s=10,
                    ).run_train(spec)

    def test_subprocess_runner_resolves_outputs_relative_to_subprocess_work_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp) / "work"
            work_dir.mkdir()
            checkpoint_bytes = b"relative-checkpoint"
            checksum = "sha256:" + hashlib.sha256(checkpoint_bytes).hexdigest()
            payload = {
                **_expected_train_summary_binding(
                    job_id="train-relative-output",
                    candidate_id="candidate-code",
                ),
                "status": "pass",
                "checkpoint": {
                    "uri": "checkpoints/best.pt",
                    "checksum": checksum,
                    "role": "train_output",
                    "metadata": {"local_path": "checkpoints/best.pt"},
                },
            }
            train_code = textwrap.dedent(
                f"""
                import json
                from pathlib import Path
                checkpoint = Path("checkpoints/best.pt")
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                checkpoint.write_bytes({checkpoint_bytes!r})
                summary = Path("train/summary.json")
                summary.parent.mkdir(parents=True, exist_ok=True)
                summary.write_text(json.dumps({payload!r}))
                """
            )
            spec = _train_job_spec(
                job_id="train-relative-output",
                task_id="unitree_push_recovery",
                candidate_id="candidate-code",
                policy_initialization=PolicyInitialization.random(),
                budget_steps=128,
                metadata={
                    "cli_argv": [sys.executable, "-c", train_code],
                    "summary_path": "train/summary.json",
                    "checkpoint_output_path": "checkpoints/best.pt",
                    "work_dir": str(work_dir),
                },
            )

            result = SubprocessRunner(
                allowed_executables=(sys.executable,),
                timeout_s=10,
            ).run_train(spec)

        self.assertEqual(result.checkpoint.checksum, checksum)
        self.assertEqual(
            result.checkpoint.metadata["local_path"],
            str((work_dir / "checkpoints" / "best.pt").resolve()),
        )

    def test_subprocess_runner_projects_sealed_validation_gate(self) -> None:
        public_result = self._run_eval_summary(
            {
                "status": "pass",
                "public_score": 0.42,
                "metrics": {"success_rate": 0.42},
                "candidate_code_loaded_in_eval": False,
                "candidate_package_loaded_in_eval": False,
                "checkpoint_safety_report": self._checkpoint_safety_report(),
                "sealed_validation_gate": self._sealed_validation_gate(),
            }
        )

        self.assertEqual(public_result.metrics["sealed_validation_gate"]["candidate_isolation_status"], "pass")
        self.assertEqual(public_result.metrics["sealed_validation_gate"]["checkpoint_safety_status"], "pass")
        self.assertIs(public_result.metrics["candidate_package_loaded_in_eval"], False)
        self.assertEqual(public_result.metrics["checkpoint_safety_report"]["status"], "pass")

    def test_subprocess_runner_rejects_failed_sealed_validation_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eval_summary = root / "eval_summary.json"
            eval_payload = {
                "status": "pass",
                "public_score": 0.42,
                "metrics": {"success_rate": 0.42},
                "checkpoint_safety_report": self._checkpoint_safety_report(),
                "candidate_code_loaded_in_eval": True,
                "sealed_validation_gate": {
                    "status": "fail",
                    "candidate_isolation_status": "fail",
                    "checkpoint_safety_status": "pass",
                    "candidate_code_loaded_in_eval": True,
                    "candidate_package_loaded_in_eval": False,
                    "candidate_path_argument_present": False,
                },
            }
            eval_code = textwrap.dedent(
                f"""
                import json
                from pathlib import Path
                Path({str(eval_summary)!r}).write_text({json.dumps(eval_payload)!r})
                """
            )
            runner = SubprocessRunner(allowed_executables=(sys.executable,), timeout_s=10)
            eval_spec = _eval_job_spec(
                job_id="eval-gate-fail",
                task_id="unitree_push_recovery",
                candidate_id="candidate-code",
                checkpoint=self._eval_checkpoint(root),
                public_seed_count=3,
                sealed_validation_seed_count=7,
                evaluation_binding=make_evaluation_binding(),
                metadata={"summary_path": str(eval_summary)},
            )
            eval_payload = {**self._eval_summary_binding(eval_spec), **eval_payload}
            eval_code = textwrap.dedent(
                f"""
                import json
                from pathlib import Path
                Path({str(eval_summary)!r}).write_text({json.dumps(eval_payload)!r})
                """
            )
            eval_spec = _with_eval_argv(
                eval_spec,
                self._checkpoint_cli_argv(
                    eval_spec.execution_checkpoint(),
                    eval_code,
                ),
            )

            with self.assertRaisesRegex(RunnerExecutionError, "candidate isolation failed"):
                runner.run_eval(eval_spec)

    def test_subprocess_runner_rejects_eval_summary_bound_to_different_job(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eval_summary = root / "eval_summary.json"
            runner = SubprocessRunner(allowed_executables=(sys.executable,), timeout_s=10)
            eval_spec = _eval_job_spec(
                job_id="eval-current",
                task_id="unitree_push_recovery",
                candidate_id="candidate-code",
                checkpoint=self._eval_checkpoint(root),
                public_seed_count=3,
                sealed_validation_seed_count=7,
                evaluation_binding=make_evaluation_binding(),
                metadata={"summary_path": str(eval_summary)},
            )
            payload = {
                **self._eval_summary_binding(eval_spec),
                "job_id": "eval-stale",
                "subject_id": "wrong-candidate",
                "status": "pass",
                "public_score": 0.99,
                "metrics": {"success_rate": 0.99},
                "checkpoint_safety_report": self._checkpoint_safety_report(),
                "sealed_validation_gate": self._sealed_validation_gate(),
            }
            eval_code = textwrap.dedent(
                f"""
                import json
                from pathlib import Path
                Path({str(eval_summary)!r}).write_text(json.dumps({payload!r}))
                """
            )
            eval_spec = _with_eval_argv(
                eval_spec,
                self._checkpoint_cli_argv(
                    eval_spec.execution_checkpoint(),
                    eval_code,
                ),
            )

            with self.assertRaisesRegex(RunnerExecutionError, "eval summary binding mismatch"):
                runner.run_eval(eval_spec)

    def test_subprocess_runner_deletes_stale_eval_summary_before_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eval_summary = root / "eval_summary.json"
            eval_summary.write_text(
                json.dumps(
                    {
                        "job_id": "eval-stale",
                        "task_id": "unitree_push_recovery",
                        "candidate_id": "wrong-candidate",
                        "status": "pass",
                        "public_score": 0.99,
                        "metrics": {"success_rate": 0.99},
                        "checkpoint_safety_report": self._checkpoint_safety_report(),
                        "sealed_validation_gate": self._sealed_validation_gate(),
                    }
                ),
                encoding="utf-8",
            )
            runner = SubprocessRunner(allowed_executables=(sys.executable,), timeout_s=10)
            eval_spec = _eval_job_spec(
                job_id="eval-current",
                task_id="unitree_push_recovery",
                candidate_id="candidate-code",
                checkpoint=self._eval_checkpoint(root),
                public_seed_count=3,
                sealed_validation_seed_count=7,
                evaluation_binding=make_evaluation_binding(),
                metadata={"summary_path": str(eval_summary)},
            )
            eval_spec = _with_eval_argv(
                eval_spec,
                self._checkpoint_cli_argv(
                    eval_spec.execution_checkpoint(),
                    "pass",
                ),
            )

            with self.assertRaisesRegex(RunnerExecutionError, "summary file was not created"):
                runner.run_eval(eval_spec)

    def test_subprocess_runner_rejects_pass_gate_with_contamination_evidence(self) -> None:
        summary = {
            "status": "pass",
            "public_score": 0.42,
            "metrics": {"success_rate": 0.42},
            "checkpoint_safety_report": self._checkpoint_safety_report(),
            "sealed_validation_gate": {
                "status": "pass",
                "candidate_isolation_status": "pass",
                "checkpoint_safety_status": "pass",
                "candidate_code_loaded_in_eval": True,
                "candidate_package_loaded_in_eval": False,
                "candidate_path_argument_present": False,
            },
        }

        with self.assertRaisesRegex(RunnerExecutionError, "runner-derived evidence"):
            self._run_eval_summary(summary)

    def test_subprocess_runner_rejects_top_level_candidate_contamination(self) -> None:
        for key in ("candidate_code_loaded_in_eval", "candidate_package_loaded", "candidate_package_loaded_in_eval"):
            summary = {
                "status": "pass",
                "public_score": 0.42,
                "metrics": {"success_rate": 0.42},
                "checkpoint_safety_report": self._checkpoint_safety_report(),
                key: True,
                "sealed_validation_gate": {
                    "status": "pass",
                    "candidate_isolation_status": "pass",
                    "checkpoint_safety_status": "pass",
                    "candidate_code_loaded_in_eval": False,
                    "candidate_package_loaded_in_eval": False,
                    "candidate_path_argument_present": False,
                },
            }
            with self.subTest(key=key):
                with self.assertRaisesRegex(RunnerExecutionError, "candidate isolation failed"):
                    self._run_eval_summary(summary)

    def test_subprocess_runner_rejects_metrics_candidate_contamination(self) -> None:
        summary = {
            "status": "pass",
            "public_score": 0.42,
            "metrics": {"success_rate": 0.42, "candidate_code_loaded_in_eval": True},
            "checkpoint_safety_report": self._checkpoint_safety_report(),
            "sealed_validation_gate": {
                "status": "pass",
                "candidate_isolation_status": "pass",
                "checkpoint_safety_status": "pass",
                "candidate_code_loaded_in_eval": False,
                "candidate_package_loaded_in_eval": False,
                "candidate_path_argument_present": False,
            },
        }

        with self.assertRaisesRegex(RunnerExecutionError, "candidate isolation failed"):
            self._run_eval_summary(summary)

    def test_subprocess_runner_rejects_missing_checkpoint_safety_report(self) -> None:
        summary = {
            "status": "pass",
            "public_score": 0.42,
            "metrics": {"success_rate": 0.42},
            "sealed_validation_gate": self._sealed_validation_gate(),
        }

        with self.assertRaisesRegex(RunnerExecutionError, "checkpoint_safety_report must be an object"):
            self._run_eval_summary(summary)

    def test_subprocess_runner_rejects_failed_checkpoint_safety_report(self) -> None:
        summary = {
            "status": "pass",
            "public_score": 0.42,
            "metrics": {"success_rate": 0.42},
            "checkpoint_safety_report": self._checkpoint_safety_report(
                status="fail",
                reason="checkpoint_weights_only_load_failed",
            ),
            "sealed_validation_gate": self._sealed_validation_gate(checkpoint_safety_status="fail"),
        }

        with self.assertRaisesRegex(RunnerExecutionError, "checkpoint safety failed"):
            self._run_eval_summary(summary)

    def test_subprocess_runner_rejects_checkpoint_safety_hash_for_different_eval_checkpoint(self) -> None:
        summary = {
            "status": "pass",
            "public_score": 0.42,
            "metrics": {"success_rate": 0.42},
            "checkpoint_safety_report": self._checkpoint_safety_report(
                checkpoint_hash=_OTHER_CHECKPOINT_HASH
            ),
            "sealed_validation_gate": self._sealed_validation_gate(),
        }

        with self.assertRaisesRegex(RunnerExecutionError, "checkpoint safety report hash"):
            self._run_eval_summary(summary)

    def test_subprocess_runner_rejects_gate_checkpoint_safety_failure(self) -> None:
        summary = {
            "status": "pass",
            "public_score": 0.42,
            "metrics": {"success_rate": 0.42},
            "checkpoint_safety_report": self._checkpoint_safety_report(),
            "sealed_validation_gate": self._sealed_validation_gate(checkpoint_safety_status="fail"),
        }

        with self.assertRaisesRegex(RunnerExecutionError, "runner-derived evidence"):
            self._run_eval_summary(summary)

    def test_subprocess_runner_projects_safe_evidence_manifest(self) -> None:
        checkpoint_safety_report = self._checkpoint_safety_report()
        artifact = evidence_artifact_from_json(
            artifact_id="checkpoint-safety-report",
            kind="checkpoint_safety_report",
            payload=checkpoint_safety_report,
            producer="isaaclab_rsl_rl_eval",
            visibility="public_projection_only",
            public_projection_ref="metrics.checkpoint_safety_report",
        )
        summary = {
            "status": "pass",
            "public_score": 0.42,
            "metrics": {"success_rate": 0.42},
            "checkpoint_safety_report": checkpoint_safety_report,
            "sealed_validation_gate": self._sealed_validation_gate(),
            "evidence_manifest": [artifact.to_public_dict()],
        }

        public_result = self._run_eval_summary(summary)

        self.assertEqual(public_result.metrics["evidence_manifest"][0]["kind"], "checkpoint_safety_report")
        self.assertNotIn("path", json.dumps(public_result.metrics["evidence_manifest"]).lower())

    def test_subprocess_runner_rejects_evidence_manifest_not_bound_to_projection(self) -> None:
        checkpoint_safety_report = self._checkpoint_safety_report()
        artifact = evidence_artifact_from_json(
            artifact_id="checkpoint-safety-report",
            kind="checkpoint_safety_report",
            payload=checkpoint_safety_report,
            producer="isaaclab_rsl_rl_eval",
            visibility="public_projection_only",
            public_projection_ref="metrics.checkpoint_safety_report",
        ).to_public_dict()
        for field, value in (
            ("sha256", "sha256:" + "f" * 64),
            ("byte_size", artifact["byte_size"] + 1),
            ("public_projection_ref", "metrics.unknown_projection"),
        ):
            with self.subTest(field=field):
                tampered = {**artifact, field: value}
                summary = {
                    "status": "pass",
                    "public_score": 0.42,
                    "metrics": {"success_rate": 0.42},
                    "checkpoint_safety_report": checkpoint_safety_report,
                    "sealed_validation_gate": self._sealed_validation_gate(),
                    "evidence_manifest": [tampered],
                }

                with self.assertRaisesRegex(RunnerExecutionError, "evidence artifact"):
                    self._run_eval_summary(summary)

    def test_subprocess_runner_rejects_malformed_evidence_manifest_as_runner_error(self) -> None:
        summary = {
            "status": "pass",
            "public_score": 0.42,
            "metrics": {"success_rate": 0.42},
            "checkpoint_safety_report": self._checkpoint_safety_report(),
            "sealed_validation_gate": self._sealed_validation_gate(),
            "evidence_manifest": [
                {
                    "schema_version": "true_rl_agent.evidence.artifact.v1",
                    "artifact_id": "checkpoint-safety-report",
                    "sha256": "sha256:" + "c" * 64,
                    "byte_size": 123,
                    "visibility": "public_projection_only",
                    "producer": "isaaclab_rsl_rl_eval",
                    "mime_type": "application/json",
                    "public_projection_ref": "metrics.checkpoint_safety_report",
                }
            ],
        }

        with self.assertRaisesRegex(RunnerExecutionError, "evidence artifact kind is required"):
            self._run_eval_summary(summary)

    def test_subprocess_runner_rejects_wrong_type_evidence_byte_size_as_runner_error(self) -> None:
        for invalid_value in (None, True, "123"):
            with self.subTest(invalid_value=repr(invalid_value)):
                summary = {
                    "status": "pass",
                    "public_score": 0.42,
                    "metrics": {"success_rate": 0.42},
                    "checkpoint_safety_report": self._checkpoint_safety_report(),
                    "sealed_validation_gate": self._sealed_validation_gate(),
                    "evidence_manifest": [
                        {
                            "schema_version": "true_rl_agent.evidence.artifact.v1",
                            "artifact_id": "checkpoint-safety-report",
                            "kind": "checkpoint_safety_report",
                            "sha256": "sha256:" + "c" * 64,
                            "byte_size": invalid_value,
                            "visibility": "public_projection_only",
                            "producer": "isaaclab_rsl_rl_eval",
                            "mime_type": "application/json",
                            "public_projection_ref": "metrics.checkpoint_safety_report",
                        }
                    ],
                }

                with self.assertRaisesRegex(
                    RunnerExecutionError,
                    "evidence artifact byte_size",
                ):
                    self._run_eval_summary(summary)

    def test_subprocess_runner_rejects_shell_strings_and_unapproved_executables(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner = SubprocessRunner(allowed_executables=(sys.executable,), timeout_s=1)
            common_metadata = {
                "summary_path": str(root / "summary.json"),
                "checkpoint_output_path": str(root / "best.pt"),
            }
            string_command_spec = _train_job_spec(
                job_id="train-string",
                task_id="unitree_push_recovery",
                candidate_id="candidate",
                policy_initialization=PolicyInitialization.random(),
                budget_steps=128,
                metadata={**common_metadata, "cli_argv": "python -c pass"},
            )
            unapproved_spec = _train_job_spec(
                job_id="train-bad-exec",
                task_id="unitree_push_recovery",
                candidate_id="candidate",
                policy_initialization=PolicyInitialization.random(),
                budget_steps=128,
                metadata={**common_metadata, "cli_argv": ["/bin/sh", "-c", "echo bad"]},
            )

            with self.assertRaisesRegex(RunnerExecutionError, "cli_argv must be list"):
                runner.run_train(string_command_spec)
            with self.assertRaisesRegex(RunnerExecutionError, "executable is not allowlisted"):
                runner.run_train(unapproved_spec)

    def test_subprocess_runner_reports_missing_summary_as_execution_error(self) -> None:
        runner = SubprocessRunner(allowed_executables=(sys.executable,), timeout_s=10)
        with tempfile.TemporaryDirectory() as tmp:
            missing_summary = Path(tmp) / "missing.json"
            train_spec = _train_job_spec(
                job_id="train-missing-summary",
                task_id="unitree_push_recovery",
                candidate_id="candidate",
                policy_initialization=PolicyInitialization.random(),
                budget_steps=128,
                metadata={
                    "cli_argv": [sys.executable, "-c", "pass"],
                    "summary_path": str(missing_summary),
                    "checkpoint_output_path": str(Path(tmp) / "best.pt"),
                },
            )

            with self.assertRaisesRegex(RunnerExecutionError, "summary file was not created"):
                runner.run_train(train_spec)

    def test_unclassified_train_failure_is_infrastructure_owned(self) -> None:
        with self.assertRaisesRegex(
            RunnerExecutionError,
            "not bound to a candidate failure contract",
        ):
            self._run_train_summary(
                {
                    "status": "fail",
                    "errors": [{"type": "RuntimeError", "message": "runtime failure"}],
                },
                exit_code=1,
            )

    def test_closed_candidate_failure_is_projected_without_raw_error(self) -> None:
        evidence = _TEST_CANDIDATE_FAILURE_CONTRACT.evidence(
            stage="activation_preflight",
            reason_code="activation_rejected",
        )
        with self.assertRaises(CandidateRuntimeError) as context:
            self._run_train_summary(
                {
                    "status": "fail",
                    "candidate_failure": evidence.to_public_dict(),
                    "errors": [],
                },
                exit_code=1,
                candidate_failure_contract=_TEST_CANDIDATE_FAILURE_CONTRACT,
            )

        self.assertEqual(
            context.exception.public_details,
            {
                "status": "fail",
                "subprocess_returncode": 1,
                "candidate_failure_contract": (
                    _TEST_CANDIDATE_FAILURE_CONTRACT.to_public_dict()
                ),
                "candidate_failure": evidence.to_public_dict(),
            },
        )
        self.assertNotIn("errors", context.exception.to_public_dict()["details"])

    def test_candidate_failure_rejects_raw_errors(self) -> None:
        with self.assertRaisesRegex(
            RunnerExecutionError,
            "cannot contain raw errors",
        ):
            self._run_train_summary(
                {
                    "status": "fail",
                    "candidate_failure": (
                        _TEST_CANDIDATE_FAILURE_CONTRACT.evidence(
                            stage="activation_preflight",
                            reason_code="activation_rejected",
                        ).to_public_dict()
                    ),
                    "errors": [
                        {"type": "RuntimeError", "message": "private path"}
                    ],
                },
                exit_code=1,
                candidate_failure_contract=_TEST_CANDIDATE_FAILURE_CONTRACT,
            )

    def test_candidate_failure_contract_rejects_unknown_or_forged_evidence(self) -> None:
        valid = _TEST_CANDIDATE_FAILURE_CONTRACT.evidence(
            stage="activation_preflight",
            reason_code="activation_rejected",
        ).to_public_dict()
        cases = {
            "owner": {**valid, "failure_owner": "infrastructure"},
            "stage": {**valid, "failure_stage": "unknown_stage"},
            "reason": {**valid, "reason_code": "unknown_reason"},
            "contract": {**valid, "contract_hash": "f" * 64},
            "extra": {**valid, "private_error": "D:/private/path"},
        }
        for label, candidate_failure in cases.items():
            with self.subTest(label=label):
                with self.assertRaisesRegex(
                    RunnerExecutionError,
                    "invalid candidate failure evidence",
                ):
                    self._run_train_summary(
                        {
                            "status": "fail",
                            "candidate_failure": candidate_failure,
                            "errors": [],
                        },
                        exit_code=1,
                        candidate_failure_contract=(
                            _TEST_CANDIDATE_FAILURE_CONTRACT
                        ),
                    )

    def test_candidate_failure_policy_pairing_depends_on_declared_stage(self) -> None:
        expected = "a" * 64
        activation = _TEST_CANDIDATE_FAILURE_CONTRACT.evidence(
            stage="activation_preflight",
            reason_code="activation_rejected",
        )
        with self.assertRaises(CandidateRuntimeError):
            self._run_train_summary(
                {
                    "status": "fail",
                    "candidate_failure": activation.to_public_dict(),
                    "training": {
                        "expected_initial_policy_state_hash": expected,
                    },
                    "errors": [],
                },
                exit_code=1,
                expected_initial_policy_state_hash=expected,
                candidate_failure_contract=_TEST_CANDIDATE_FAILURE_CONTRACT,
            )

        with self.assertRaisesRegex(
            RunnerExecutionError,
            "cannot claim actual policy state",
        ):
            self._run_train_summary(
                {
                    "status": "fail",
                    "candidate_failure": activation.to_public_dict(),
                    "training": {
                        "expected_initial_policy_state_hash": expected,
                        "initial_policy_state_hash": expected,
                        "initial_policy_state_match": True,
                    },
                    "errors": [],
                },
                exit_code=1,
                expected_initial_policy_state_hash=expected,
                candidate_failure_contract=_TEST_CANDIDATE_FAILURE_CONTRACT,
            )

        ppo = _TEST_CANDIDATE_FAILURE_CONTRACT.evidence(
            stage="ppo_reward_execution",
            reason_code="runtime_rejected",
        )
        with self.assertRaises(CandidateRuntimeError):
            self._run_train_summary(
                {
                    "status": "fail",
                    "candidate_failure": ppo.to_public_dict(),
                    "training": {
                        "expected_initial_policy_state_hash": expected,
                        "initial_policy_state_hash": expected,
                        "initial_policy_state_match": True,
                    },
                    "errors": [],
                },
                exit_code=1,
                expected_initial_policy_state_hash=expected,
                candidate_failure_contract=_TEST_CANDIDATE_FAILURE_CONTRACT,
            )

    def test_failed_train_summary_rejects_zero_exit(self) -> None:
        with self.assertRaisesRegex(
            RunnerExecutionError,
            "requires a nonzero subprocess exit",
        ):
            self._run_train_summary(
                {
                    "status": "fail",
                    "candidate_failure": (
                        _TEST_CANDIDATE_FAILURE_CONTRACT.evidence(
                            stage="activation_preflight",
                            reason_code="activation_rejected",
                        ).to_public_dict()
                    ),
                    "errors": [],
                },
                exit_code=0,
                candidate_failure_contract=_TEST_CANDIDATE_FAILURE_CONTRACT,
            )

    def test_nonzero_exit_rejects_a_pass_summary_as_inconsistent(self) -> None:
        with self.assertRaisesRegex(
            RunnerExecutionError,
            "train summary status is pass but subprocess exited nonzero",
        ):
            self._run_train_summary(
                {"status": "pass"},
                checkpoint_bytes=b"should-not-be-consumed",
                exit_code=3,
            )

    def test_nonzero_eval_exit_can_return_a_complete_fail_result(self) -> None:
        result = self._run_eval_summary(
            {
                "status": "fail",
                "public_score": 0.0,
                "metrics": {"success_rate": 0.0},
                "checkpoint_safety_report": self._checkpoint_safety_report(),
                "sealed_validation_gate": self._sealed_validation_gate(),
            },
            exit_code=1,
        )

        self.assertEqual(result.status, "fail")
        self.assertEqual(result.public_score, 0.0)

    def test_sealed_summary_hash_is_checked_and_raw_summary_stays_private(self) -> None:
        raw_summary = {
            "schema_version": "test.task.sealed_summary.v1",
            "seed_macro_scores": [0.42] * 7,
        }
        result = self._run_eval_summary(
            {
                "status": "pass",
                "public_score": 0.42,
                "metrics": {"success_rate": 0.42},
                "checkpoint_safety_report": self._checkpoint_safety_report(),
                "sealed_validation_summary": raw_summary,
                "sealed_validation_gate": self._sealed_validation_gate(),
            },
        )

        self.assertEqual(
            result.metrics["sealed_validation"]["task_evidence_hash"],
            canonical_json_hash(raw_summary),
        )
        self.assertNotIn("sealed_validation_summary", result.metrics)

    def test_sealed_summary_tamper_rejects_hash_mismatch(self) -> None:
        with self.assertRaisesRegex(
            RunnerExecutionError,
            "task_evidence_hash binding mismatch",
        ):
            self._run_eval_summary(
                {
                    "status": "pass",
                    "public_score": 0.42,
                    "metrics": {"success_rate": 0.42},
                    "checkpoint_safety_report": self._checkpoint_safety_report(),
                    "sealed_validation_summary": {
                        "schema_version": "test.task.sealed_summary.v1",
                        "seed_macro_scores": [0.42] * 7,
                    },
                    "sealed_validation_gate": self._sealed_validation_gate(),
                },
                task_evidence_hash_override="0" * 64,
            )

    def test_sealed_summary_score_stats_must_match_public_evidence(self) -> None:
        with self.assertRaisesRegex(
            RunnerExecutionError,
            "task summary score_stats differ from public evidence",
        ):
            self._run_eval_summary(
                {
                    "status": "pass",
                    "public_score": 0.42,
                    "metrics": {"success_rate": 0.42},
                    "checkpoint_safety_report": self._checkpoint_safety_report(),
                    "sealed_validation_summary": {
                        "schema_version": "test.task.sealed_summary.v1",
                        "seed_macro_scores": [0.9] * 7,
                        "score_stats": {"mean": 0.9},
                    },
                    "sealed_validation_gate": self._sealed_validation_gate(),
                },
            )

    def test_nonzero_eval_exit_does_not_hide_incomplete_fail_evidence(self) -> None:
        with self.assertRaisesRegex(
            RunnerExecutionError,
            "sealed validation rollout evidence invalid",
        ):
            self._run_eval_summary(
                {
                    "status": "fail",
                    "public_score": 0.0,
                    "metrics": {"success_rate": 0.0},
                    "checkpoint_safety_report": self._checkpoint_safety_report(),
                    "sealed_validation_gate": self._sealed_validation_gate(),
                },
                exit_code=1,
                include_sealed_validation_evidence=False,
            )

    def test_nonzero_exit_requires_a_valid_identity_bound_fail_summary(self) -> None:
        cases = {
            "missing": ("raise SystemExit(2)", "summary file was not created"),
            "invalid_json": (
                "from pathlib import Path; "
                "Path('summary.json').write_text('{'); raise SystemExit(2)",
                "summary file is not valid JSON",
            ),
            "identity_mismatch": (
                "import json; from pathlib import Path; "
                "Path('summary.json').write_text(json.dumps({"
                "'job_id':'old-job','task_id':'unitree_push_recovery',"
                "'subject_kind':'candidate','subject_id':'candidate',"
                f"'subject_manifest_hash':{TEST_SUBJECT_MANIFEST_HASH!r},"
                f"'training_binding_hash':{make_training_binding(PolicyInitialization.random(), budget_steps=128).binding_hash!r},"
                f"'capability_profile_hash':{str(TEST_CAPABILITY_PROFILE.profile_hash)!r},"
                f"'program_source_checksum':{TEST_PROGRAM_SOURCE_CHECKSUM!r},"
                "'status':'fail'})); raise SystemExit(2)",
                "summary binding mismatch for job_id",
            ),
        }
        for label, (code, expected_error) in cases.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                summary_path = root / "summary.json"
                if label != "missing":
                    code = code.replace("'summary.json'", repr(str(summary_path)))
                spec = _train_job_spec(
                    job_id="current-job",
                    task_id="unitree_push_recovery",
                    candidate_id="candidate",
                    policy_initialization=PolicyInitialization.random(),
                    budget_steps=128,
                    metadata={
                        "cli_argv": [sys.executable, "-c", code],
                        "summary_path": str(summary_path),
                        "checkpoint_output_path": str(root / "best.pt"),
                    },
                )

                with self.assertRaisesRegex(RunnerExecutionError, expected_error):
                    SubprocessRunner(
                        allowed_executables=(sys.executable,),
                        timeout_s=10,
                    ).run_train(spec)

    def test_subprocess_train_requires_checkpoint_output_contract_before_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary_path = root / "summary.json"
            marker_path = root / "executed.marker"
            forged_summary = {
                **_expected_train_summary_binding(
                    job_id="train-missing-output-contract",
                    candidate_id="candidate",
                ),
                "status": "pass",
                "checkpoint": {
                    "uri": "artifact://forged/checkpoint.pt",
                    "checksum": _FORGED_CHECKPOINT_HASH,
                    "role": "train_output",
                },
                "runtime_activation_summary": {"reward_term_calls": 999},
            }
            code = textwrap.dedent(
                f"""
                import json
                from pathlib import Path
                Path({str(marker_path)!r}).write_text("executed")
                Path({str(summary_path)!r}).write_text(json.dumps({forged_summary!r}))
                """
            )
            spec = _train_job_spec(
                job_id="train-missing-output-contract",
                task_id="unitree_push_recovery",
                candidate_id="candidate",
                policy_initialization=PolicyInitialization.random(),
                budget_steps=128,
                metadata={
                    "cli_argv": [sys.executable, "-c", code],
                    "summary_path": str(summary_path),
                },
            )

            with self.assertRaisesRegex(RunnerExecutionError, "checkpoint_output_path is required"):
                SubprocessRunner(
                    allowed_executables=(sys.executable,),
                    timeout_s=10,
                ).run_train(spec)

            self.assertFalse(marker_path.exists())

    def test_subprocess_train_rejects_symlinked_checkpoint_output_without_following_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint_target = root / "outside-target.pt"
            checkpoint_target.write_bytes(b"must-survive")
            checkpoint_output = root / "planned-output.pt"
            try:
                checkpoint_output.symlink_to(checkpoint_target)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"file symlinks unavailable: {type(exc).__name__}")

            marker_path = root / "executed.marker"
            spec = _train_job_spec(
                job_id="train-symlink-output",
                task_id="unitree_push_recovery",
                candidate_id="candidate",
                policy_initialization=PolicyInitialization.random(),
                budget_steps=128,
                metadata={
                    "cli_argv": [
                        sys.executable,
                        "-c",
                        f"from pathlib import Path; Path({str(marker_path)!r}).write_text('executed')",
                    ],
                    "summary_path": str(root / "summary.json"),
                    "checkpoint_output_path": str(checkpoint_output),
                },
            )

            with self.assertRaisesRegex(
                RunnerExecutionError,
                "checkpoint_output_path must not be a symlink",
            ):
                SubprocessRunner(
                    allowed_executables=(sys.executable,),
                    timeout_s=10,
                ).run_train(spec)

            self.assertEqual(checkpoint_target.read_bytes(), b"must-survive")
            self.assertFalse(marker_path.exists())

    def test_subprocess_runner_redacts_secrets_from_failed_process_output(self) -> None:
        runner = SubprocessRunner(allowed_executables=(sys.executable,), timeout_s=10)
        with tempfile.TemporaryDirectory() as tmp:
            missing_summary = Path(tmp) / "summary.json"
            failure_code = textwrap.dedent(
                """
                import sys
                print("Task-neutral task-specific risk-free disk-usage")
                print("sk-lower123 SK-UPPER456")
                print("TOKEN=plainsecretvalue")
                print("MY_TOKEN = namespacedsecretvalue")
                print("NOTOKEN=ordinaryvalue")
                print("Authorization: Bearer plainbearer", file=sys.stderr)
                print("Authorization : Bearer spacedbearer", file=sys.stderr)
                print("XAuthorization: Bearer ordinarybearer", file=sys.stderr)
                print(r'request headers={\"Authorization\":\"Basic escapedbasic\"}', file=sys.stderr)
                print("upstream sent Bearer embeddedbearer", file=sys.stderr)
                raise SystemExit(2)
                """
            )
            train_spec = _train_job_spec(
                job_id="train-secret-output",
                task_id="unitree_push_recovery",
                candidate_id="candidate",
                policy_initialization=PolicyInitialization.random(),
                budget_steps=128,
                metadata={
                    "cli_argv": [sys.executable, "-c", failure_code],
                    "summary_path": str(missing_summary),
                    "checkpoint_output_path": str(Path(tmp) / "best.pt"),
                },
            )

            with self.assertRaises(RunnerExecutionError) as context:
                runner.run_train(train_spec)

        message = str(context.exception)
        self.assertNotIn("plainsecretvalue", message)
        self.assertNotIn("namespacedsecretvalue", message)
        self.assertNotIn("plainbearer", message)
        self.assertNotIn("spacedbearer", message)
        self.assertNotIn("ordinarybearer", message)
        self.assertNotIn("escapedbasic", message)
        self.assertNotIn("embeddedbearer", message)
        self.assertNotIn("sk-lower123", message)
        self.assertNotIn("SK-UPPER456", message)
        self.assertIn("Task-neutral task-specific risk-free disk-usage", message)
        self.assertIn("NOTOKEN=ordinaryvalue", message)
        self.assertEqual(message.count("[redacted]"), 3)
        self.assertIn("TOKEN=REDACTED", message)
        self.assertIn("MY_TOKEN=REDACTED", message)


if __name__ == "__main__":
    unittest.main()
