"""Isaac Lab + RSL-RL backend job materializer."""

from __future__ import annotations

import posixpath
import sys
from dataclasses import InitVar, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from true_rl_agent.backends.base import BackendAdapter, BackendJobBundle
from true_rl_agent.backends.isaac_lab_execution_environment import (
    IsaacLabExecutionEnvironmentManifest,
    build_current_isaac_lab_execution_environment_manifest,
)
from true_rl_agent.backends.rsl_rl_training_config import (
    RslRlAgentConfiguration,
)
from true_rl_agent.backends.specs import BackendSpec
from true_rl_agent.candidates.artifact import (
    VerifiedCandidateArtifactRef,
    verify_verified_candidate_artifact,
)
from true_rl_agent.evaluation.binding import (
    EvaluationBindingV1,
    canonical_json_hash,
    is_sha256_checksum,
    require_sha256_digest,
)
from true_rl_agent.evaluation.protocol import (
    EvaluationProtocol,
)
from true_rl_agent.evaluation.phases import EvaluationPhase
from true_rl_agent.evaluation.subject import (
    ExperimentSubject,
    validate_experiment_subject_id,
)
from true_rl_agent.policy import PolicyInitialization
from true_rl_agent.policy_baselines import (
    BaselineManifest,
    validate_baseline_compatibility,
)
from true_rl_agent.reference_artifact import (
    VerifiedReferenceProgramArtifactRef,
    materialize_reference_program_artifact,
)
from true_rl_agent.runner.budget import build_budget_accounting
from true_rl_agent.runner.candidate_failure import CandidateFailureContract
from true_rl_agent.runner.job_identity import JobIdentity
from true_rl_agent.runner.job_attempts import JobAttemptScope
from true_rl_agent.runner.specs import (
    BaselineEvalJobSpec,
    CheckpointOutputPlan,
    CheckpointRef,
    EvalJobSpec,
    EvalPhaseJobSpec,
    TrainJobSpec,
    require_execution_device,
    require_positive_integer,
)
from true_rl_agent.tasks.base import TaskSpec
from true_rl_agent.training_binding import (
    TrainingBinding,
    fixed_training_seed_resolver_hash,
    initialization_seed_schedule_hash,
)
from true_rl_agent.training_programs import VerifiedReferenceProgramArtifact


@dataclass(frozen=True)
class IsaacLabRslRlBackendConfig:
    isaac_lab_python: str
    train_entrypoint: str
    eval_entrypoint: str
    supported_tasks: frozenset[str]
    default_budget_steps: int
    public_seed_count: int
    public_accept_seed_count: int
    eval_num_envs: int
    eval_protocol_version: str
    evaluation_binding: EvaluationBindingV1
    robot: str
    observation_schema_hash: str
    action_schema_hash: str
    policy_architecture: str
    train_profile: str
    eval_profile: str
    train_num_envs: int
    training_agent_configuration: RslRlAgentConfiguration
    eval_base_seed: int
    execution_device: str
    candidate_admission_policy_id: str
    candidate_admission_policy_hash: str
    candidate_execution_capability_ids: frozenset[str]
    candidate_execution_authorization_schema: str | None = None
    candidate_failure_contract: CandidateFailureContract | None = None
    train_seed: int = 42
    sealed_validation_seed_count: int | None = None
    artifact_root: str = "outputs/runner"
    train_max_iterations: int | None = None
    train_steps_per_iteration: int | None = None
    runtime_config_path: str | None = None
    train_runtime_dynamics_hash: str | None = None
    eval_runtime_dynamics_hash: str | None = None
    work_dir: str = "."
    curriculum_capabilities: Mapping[str, Mapping[str, Any]] | None = None
    trusted_seed_count: InitVar[int | None] = None

    def __post_init__(self, trusted_seed_count: int | None = None) -> None:
        for field_name in (
            "isaac_lab_python",
            "train_entrypoint",
            "eval_entrypoint",
            "eval_protocol_version",
            "robot",
            "observation_schema_hash",
            "action_schema_hash",
            "policy_architecture",
            "train_profile",
            "eval_profile",
            "artifact_root",
            "work_dir",
            "candidate_admission_policy_id",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        sealed_validation_seed_count = self.sealed_validation_seed_count
        if sealed_validation_seed_count is None:
            if trusted_seed_count is None:
                raise ValueError("sealed_validation_seed_count must be provided")
            sealed_validation_seed_count = require_positive_integer(
                trusted_seed_count,
                "trusted_seed_count",
            )
            object.__setattr__(
                self,
                "sealed_validation_seed_count",
                sealed_validation_seed_count,
            )
        else:
            sealed_validation_seed_count = require_positive_integer(
                sealed_validation_seed_count,
                "sealed_validation_seed_count",
            )
            if trusted_seed_count is not None:
                legacy_count = require_positive_integer(
                    trusted_seed_count,
                    "trusted_seed_count",
                )
                if legacy_count != sealed_validation_seed_count:
                    raise ValueError(
                        "trusted_seed_count legacy alias does not match "
                        "sealed_validation_seed_count"
                    )
        if not isinstance(self.supported_tasks, frozenset) or not self.supported_tasks:
            raise ValueError("supported_tasks must be a non-empty frozenset")
        if any(
            not isinstance(task_id, str) or not task_id.strip()
            for task_id in self.supported_tasks
        ):
            raise ValueError("supported_tasks must contain non-empty task IDs")
        require_sha256_digest(
            self.candidate_admission_policy_hash,
            "backend candidate admission policy hash",
        )
        if not isinstance(self.candidate_execution_capability_ids, frozenset):
            raise TypeError("candidate_execution_capability_ids must be a frozenset")
        if any(
            not isinstance(capability_id, str) or not capability_id
            for capability_id in self.candidate_execution_capability_ids
        ):
            raise ValueError(
                "candidate execution capability IDs must be non-empty strings"
            )
        if self.candidate_execution_authorization_schema is not None and (
            not isinstance(self.candidate_execution_authorization_schema, str)
            or not self.candidate_execution_authorization_schema
        ):
            raise ValueError(
                "candidate execution authorization schema must be a non-empty string"
            )
        if self.candidate_failure_contract is not None and not isinstance(
            self.candidate_failure_contract,
            CandidateFailureContract,
        ):
            raise TypeError(
                "candidate_failure_contract must be a CandidateFailureContract"
            )
        for field_name in (
            "default_budget_steps",
            "public_seed_count",
            "public_accept_seed_count",
            "sealed_validation_seed_count",
            "eval_num_envs",
            "train_num_envs",
        ):
            require_positive_integer(getattr(self, field_name), field_name)
        for field_name in (
            "train_max_iterations",
            "train_steps_per_iteration",
        ):
            value = getattr(self, field_name)
            if value is not None:
                require_positive_integer(value, field_name)
        if (
            self.train_max_iterations is not None
            and self.train_steps_per_iteration is not None
        ):
            build_budget_accounting(
                budget_steps=self.default_budget_steps,
                num_envs=self.train_num_envs,
                num_steps_per_env=self.train_steps_per_iteration,
                max_iterations=self.train_max_iterations,
            )
        if self.runtime_config_path is not None and (
            not isinstance(self.runtime_config_path, str)
            or not self.runtime_config_path.strip()
        ):
            raise ValueError(
                "runtime_config_path must be a non-empty string when provided"
            )
        if not isinstance(self.evaluation_binding, EvaluationBindingV1):
            raise TypeError("evaluation_binding must be EvaluationBindingV1")
        if not isinstance(
            self.training_agent_configuration,
            RslRlAgentConfiguration,
        ):
            raise TypeError(
                "training_agent_configuration must be RslRlAgentConfiguration"
            )
        require_execution_device(
            self.execution_device,
            "backend execution_device",
        )
        if (
            self.evaluation_binding.eval_protocol_hash
            != self.evaluation_protocol().protocol_hash
        ):
            raise ValueError(
                "backend evaluation binding does not match evaluation protocol"
            )
        for field_name in ("eval_base_seed", "train_seed"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        if self.curriculum_capabilities is None:
            object.__setattr__(self, "curriculum_capabilities", {})
        elif not isinstance(self.curriculum_capabilities, Mapping):
            raise TypeError("curriculum_capabilities must be an object")

    def evaluation_protocol(self) -> EvaluationProtocol:
        return EvaluationProtocol(
            eval_protocol_version=self.eval_protocol_version,
            public_seed_count=self.public_seed_count,
            public_accept_seed_count=self.public_accept_seed_count,
            sealed_validation_seed_count=require_positive_integer(
                self.sealed_validation_seed_count,
                "sealed_validation_seed_count",
            ),
            eval_num_envs=self.eval_num_envs,
        )


@dataclass(frozen=True)
class BackendCompatibilityReport:
    compatible: bool
    reasons: tuple[str, ...]


class IsaacLabRslRlBackendAdapter(BackendAdapter):
    """Materialize fixed train/eval jobs for the first real backend target."""

    external_dependencies = (
        "Isaac Lab",
        "RSL-RL",
        "Unitree G1/H1 assets",
        "CUDA runtime on the execution server",
    )

    def __init__(
        self,
        config: IsaacLabRslRlBackendConfig,
        *,
        execution_environment_resolver: Callable[
            [], IsaacLabExecutionEnvironmentManifest | Mapping[str, Any]
        ]
        | None = None,
    ) -> None:
        self.config = config
        self._uses_production_execution_environment_resolver = (
            execution_environment_resolver is None
        )
        if execution_environment_resolver is None:
            execution_environment_resolver = (
                build_current_isaac_lab_execution_environment_manifest
            )
        if not callable(execution_environment_resolver):
            raise TypeError("execution_environment_resolver must be callable")
        self._execution_environment_resolver = execution_environment_resolver

    @property
    def future_curriculum_capabilities(self) -> Mapping[str, Mapping[str, Any]]:
        return self.config.curriculum_capabilities

    @property
    def candidate_execution_capability_ids(self) -> frozenset[str]:
        return self.config.candidate_execution_capability_ids

    @property
    def candidate_admission_policy_identity(self) -> tuple[str, str]:
        return (
            self.config.candidate_admission_policy_id,
            self.config.candidate_admission_policy_hash,
        )

    @property
    def candidate_execution_authorization_schema(self) -> str | None:
        return self.config.candidate_execution_authorization_schema

    @property
    def backend_spec(self) -> BackendSpec:
        return BackendSpec(
            backend_id="isaac_lab_rsl_rl",
            supported_tasks=self.config.supported_tasks,
            supported_policy_initializations=frozenset({"random", "warm_start"}),
            curriculum_capabilities=self.config.curriculum_capabilities,
            supports_baseline_eval=True,
        )

    def evaluation_binding_for(self, task_spec: TaskSpec) -> EvaluationBindingV1:
        self._assert_eval_protocol_matches_task(task_spec)
        self._assert_evaluation_binding_matches_task(task_spec)
        return self.config.evaluation_binding

    def resolve_session_execution_environment(
        self,
        *,
        session_id: str,
        session_dir: Path,
        allow_create: bool,
    ) -> Mapping[str, Any]:
        _ = session_id, session_dir, allow_create
        return self._resolve_current_execution_environment()

    def validate_session_execution_environment(
        self,
        *,
        execution_environment: Mapping[str, Any] | None,
    ) -> None:
        if execution_environment is not None:
            IsaacLabExecutionEnvironmentManifest.from_mapping(
                execution_environment
            )

    def validate_runtime_session_execution_environment(
        self,
        *,
        execution_environment: Mapping[str, Any] | None,
    ) -> None:
        if execution_environment is None:
            raise ValueError(
                "formal Isaac Lab execution requires a Genesis-bound "
                "environment manifest"
            )
        expected = IsaacLabExecutionEnvironmentManifest.from_mapping(
            execution_environment
        ).to_dict()
        if self._resolve_current_execution_environment() != expected:
            raise ValueError("backend execution environment manifest drift")

    def _resolve_current_execution_environment(self) -> dict[str, Any]:
        if self._uses_production_execution_environment_resolver:
            controller_python = Path(sys.executable).resolve(strict=True)
            backend_python = Path(self.config.isaac_lab_python).resolve(strict=True)
            if controller_python != backend_python:
                raise ValueError(
                    "backend execution environment must be sampled by "
                    "config.isaac_lab_python: "
                    f"controller={controller_python}:backend={backend_python}"
                )
        resolved = self._execution_environment_resolver()
        manifest = (
            resolved
            if isinstance(resolved, IsaacLabExecutionEnvironmentManifest)
            else IsaacLabExecutionEnvironmentManifest.from_mapping(resolved)
        )
        return manifest.to_dict()

    def check_compatibility(
        self, *, task_spec: TaskSpec, policy_initialization: str
    ) -> BackendCompatibilityReport:
        reasons: list[str] = []
        if task_spec.task_id not in self.backend_spec.supported_tasks:
            reasons.append(f"unsupported_task:{task_spec.task_id}")
        if (
            policy_initialization
            not in self.backend_spec.supported_policy_initializations
        ):
            reasons.append(f"unsupported_policy_initialization:{policy_initialization}")
        return BackendCompatibilityReport(
            compatible=not reasons, reasons=tuple(reasons)
        )

    def materialize_jobs(
        self,
        *,
        round_id: str,
        task_spec: TaskSpec,
        candidate: VerifiedCandidateArtifactRef,
        policy_initialization: PolicyInitialization,
        evaluation_binding: EvaluationBindingV1 | None = None,
        expected_initial_policy_state_hash: str | None = None,
        backend_execution_environment_manifest_hash: str | None = None,
        train_attempt_scope: JobAttemptScope | None = None,
        eval_attempt_scope: JobAttemptScope | None = None,
    ) -> BackendJobBundle:
        candidate = verify_verified_candidate_artifact(candidate)
        self._assert_candidate_admission_policy(candidate)
        return self._materialize_program_jobs(
            run_id=round_id,
            task_spec=task_spec,
            policy_initialization=policy_initialization,
            evaluation_binding=evaluation_binding,
            candidate=candidate,
            reference=None,
            expected_initial_policy_state_hash=(expected_initial_policy_state_hash),
            execution_environment_manifest_hash=(
                backend_execution_environment_manifest_hash
            ),
            train_attempt_scope=train_attempt_scope,
            eval_attempt_scope=eval_attempt_scope,
        )

    def _assert_candidate_admission_policy(
        self,
        candidate: VerifiedCandidateArtifactRef,
    ) -> None:
        if (
            candidate.verification_policy != self.config.candidate_admission_policy_id
            or candidate.verification_policy_hash
            != self.config.candidate_admission_policy_hash
        ):
            raise ValueError(
                "candidate artifact admission policy does not match backend"
            )
        authorization = candidate.execution_authorization
        expected_schema = self.config.candidate_execution_authorization_schema
        if expected_schema is None:
            if authorization is not None:
                raise ValueError(
                    "candidate execution authorization is not supported by backend"
                )
            return
        if authorization is None:
            raise ValueError("candidate execution authorization is required by backend")
        if authorization.payload_schema_version != expected_schema:
            raise ValueError(
                "candidate execution authorization schema does not match backend"
            )

    def materialize_reference_jobs(
        self,
        *,
        session_id: str,
        task_spec: TaskSpec,
        reference: VerifiedReferenceProgramArtifact,
        artifact_root: Path,
        policy_initialization: PolicyInitialization,
        evaluation_binding: EvaluationBindingV1 | None = None,
        backend_execution_environment_manifest_hash: str | None = None,
    ) -> BackendJobBundle:
        session_id = validate_experiment_subject_id(session_id)
        if not isinstance(reference, VerifiedReferenceProgramArtifact):
            raise TypeError(
                "reference materialization requires a verified reference program"
            )
        manifest = reference.manifest
        if manifest.task_id != task_spec.task_id:
            raise ValueError("reference program task_id mismatch")
        if (
            manifest.capability_profile_hash
            != task_spec.candidate_capability_profile.profile_hash
        ):
            raise ValueError("reference program capability profile mismatch")
        artifact = materialize_reference_program_artifact(
            reference,
            artifact_root=artifact_root,
        )
        return self._materialize_program_jobs(
            run_id=session_id,
            task_spec=task_spec,
            policy_initialization=policy_initialization,
            evaluation_binding=evaluation_binding,
            candidate=None,
            reference=artifact,
            expected_initial_policy_state_hash=None,
            execution_environment_manifest_hash=(
                backend_execution_environment_manifest_hash
            ),
        )

    def _materialize_program_jobs(
        self,
        *,
        run_id: str,
        task_spec: TaskSpec,
        policy_initialization: PolicyInitialization,
        evaluation_binding: EvaluationBindingV1 | None,
        candidate: VerifiedCandidateArtifactRef | None,
        reference: VerifiedReferenceProgramArtifactRef | None,
        expected_initial_policy_state_hash: str | None,
        execution_environment_manifest_hash: str | None,
        train_attempt_scope: JobAttemptScope | None = None,
        eval_attempt_scope: JobAttemptScope | None = None,
    ) -> BackendJobBundle:
        if (candidate is None) == (reference is None):
            raise ValueError(
                "program job materialization requires exactly one artifact"
            )
        if candidate is None and expected_initial_policy_state_hash is not None:
            raise ValueError(
                "reference jobs cannot carry an expected initial policy state"
            )
        if (
            candidate is not None
            and task_spec.genesis_anchor_kind_for(policy_initialization.kind)
            == "reference_program"
            and expected_initial_policy_state_hash is None
        ):
            raise ValueError(
                "reference-anchored candidate training requires the Genesis "
                "initial policy state hash"
            )
        report = self.check_compatibility(
            task_spec=task_spec,
            policy_initialization=policy_initialization.kind,
        )
        if not report.compatible:
            raise ValueError(
                f"backend compatibility failed: {', '.join(report.reasons)}"
            )
        self._assert_training_budget_matches_task(task_spec)
        self._assert_runtime_dynamics_matches_task(task_spec)
        evaluation_binding = evaluation_binding or self.evaluation_binding_for(
            task_spec
        )
        self.validate_session_evaluation_binding(
            task_spec=task_spec,
            evaluation_binding=evaluation_binding,
        )
        if execution_environment_manifest_hash is not None:
            execution_environment_manifest_hash = require_sha256_digest(
                execution_environment_manifest_hash,
                "backend execution environment manifest hash",
            )

        planned_checkpoint = CheckpointOutputPlan(
            uri=f"artifact://{run_id}/checkpoints/best.pt",
        )
        train_summary_path = self._summary_path(round_id=run_id, phase="train")
        eval_summary_path = self._summary_path(round_id=run_id, phase="eval")
        checkpoint_output_path = self._checkpoint_output_path(round_id=run_id)
        eval_protocol = self.config.evaluation_protocol()
        if candidate is not None:
            subject = ExperimentSubject.candidate(candidate.candidate_id)
            subject_manifest_hash = candidate.subject_manifest_hash
            program_source_checksum = candidate.program_source_checksum
        else:
            assert reference is not None
            subject = ExperimentSubject.reference(reference.reference_id)
            subject_manifest_hash = reference.subject_manifest_hash
            program_source_checksum = reference.program_source_checksum
        training_binding = self._training_binding(
            task_spec=task_spec,
            policy_initialization=policy_initialization,
        )
        train_job_id = f"{run_id}-train"
        eval_job_id = f"{run_id}-eval"
        train_argv = self._train_argv(
            job_id=train_job_id,
            round_id=run_id,
            task_id=task_spec.task_id,
            subject=subject,
            candidate=candidate,
            reference=reference,
            policy_initialization=policy_initialization,
            training_binding=training_binding,
            subject_manifest_hash=subject_manifest_hash,
            program_source_checksum=program_source_checksum,
            expected_initial_policy_state_hash=(expected_initial_policy_state_hash),
            summary_path=train_summary_path,
            checkpoint_output_path=checkpoint_output_path,
            execution_environment_manifest_hash=(execution_environment_manifest_hash),
        )
        train_spec = TrainJobSpec(
            job_id=train_job_id,
            task_id=task_spec.task_id,
            subject=subject,
            subject_manifest_hash=subject_manifest_hash,
            program_source_checksum=program_source_checksum,
            policy_initialization=policy_initialization,
            training_binding=training_binding,
            budget_steps=self.config.default_budget_steps,
            execution_device=self.config.execution_device,
            expected_initial_policy_state_hash=(expected_initial_policy_state_hash),
            candidate_failure_contract=(
                self.config.candidate_failure_contract
                if candidate is not None
                else None
            ),
            candidate_artifact=candidate,
            reference_artifact=reference,
            attempt_scope=train_attempt_scope,
            metadata={
                "backend_id": self.backend_id,
                **(
                    {
                        "backend_execution_environment_manifest_hash": (
                            execution_environment_manifest_hash
                        )
                    }
                    if execution_environment_manifest_hash is not None
                    else {}
                ),
                **_cli_metadata(train_argv),
                "summary_path": train_summary_path,
                "sealed_checkpoint_uri": planned_checkpoint.uri,
                "checkpoint_output_path": checkpoint_output_path,
                "work_dir": self.config.work_dir,
                "external_dependencies": self.external_dependencies,
                "robot": self.config.robot,
                "observation_schema_hash": self.config.observation_schema_hash,
                "action_schema_hash": self.config.action_schema_hash,
                "policy_architecture": self.config.policy_architecture,
                "policy_architecture_hash": (training_binding.policy_architecture_hash),
                "optimizer_training_profile_hash": (
                    training_binding.optimizer_training_profile_hash
                ),
                "agent_configuration_hash": (training_binding.agent_configuration_hash),
                "fixed_train_protocol": {
                    **dict(task_spec.fixed_train_protocol),
                    "budget_steps": self.config.default_budget_steps,
                },
            },
        )
        eval_argv = self._eval_argv(
            job_id=eval_job_id,
            round_id=run_id,
            task_id=task_spec.task_id,
            subject=subject,
            checkpoint=planned_checkpoint,
            summary_path=eval_summary_path,
            subject_manifest_hash=subject_manifest_hash,
            program_source_checksum=program_source_checksum,
            training_binding_hash=str(training_binding.binding_hash),
            capability_profile_hash=str(
                task_spec.candidate_capability_profile.profile_hash
            ),
            evaluation_binding=evaluation_binding,
            execution_environment_manifest_hash=(execution_environment_manifest_hash),
        )
        eval_spec = EvalJobSpec(
            job_id=eval_job_id,
            task_id=task_spec.task_id,
            subject=subject,
            subject_manifest_hash=subject_manifest_hash,
            program_source_checksum=program_source_checksum,
            training_binding_hash=str(training_binding.binding_hash),
            capability_profile_hash=str(
                task_spec.candidate_capability_profile.profile_hash
            ),
            checkpoint=None,
            public_seed_count=self.config.public_seed_count,
            evaluation_binding=evaluation_binding,
            sealed_validation_seed_count=require_positive_integer(
                self.config.sealed_validation_seed_count,
                "sealed_validation_seed_count",
            ),
            public_accept_seed_count=self.config.public_accept_seed_count,
            eval_num_envs=self.config.eval_num_envs,
            eval_protocol_version=self.config.eval_protocol_version,
            execution_device=self.config.execution_device,
            checkpoint_output_plan=planned_checkpoint,
            attempt_scope=eval_attempt_scope,
            metadata={
                "backend_id": self.backend_id,
                **(
                    {
                        "backend_execution_environment_manifest_hash": (
                            execution_environment_manifest_hash
                        )
                    }
                    if execution_environment_manifest_hash is not None
                    else {}
                ),
                **_cli_metadata(eval_argv),
                "evaluation_subject": subject.to_public_dict(),
                "subject_manifest_hash": subject_manifest_hash,
                "program_source_checksum": program_source_checksum,
                "training_binding_hash": training_binding.binding_hash,
                "capability_profile_hash": (
                    task_spec.candidate_capability_profile.profile_hash
                ),
                "summary_path": eval_summary_path,
                "work_dir": self.config.work_dir,
                "checkpoint_policy": "runner_sealed_train_output_required",
                "fixed_eval_protocol": {
                    "eval_protocol": "fixed_public_accept_and_sealed_validation",
                    **eval_protocol.to_public_dict(),
                    **dict(task_spec.fixed_eval_protocol),
                },
            },
        )
        return BackendJobBundle(
            train_spec=train_spec,
            eval_spec=eval_spec,
            task_invariants=task_spec.invariants,
        )

    def materialize_eval_phase(
        self,
        *,
        eval_plan: EvalJobSpec,
        phase: EvaluationPhase,
        attempt_scope: JobAttemptScope | None = None,
    ) -> EvalPhaseJobSpec:
        """Bind one explicit split to the same full Stair evaluation plan."""

        if not isinstance(eval_plan, EvalJobSpec):
            raise TypeError("phased eval requires an EvalJobSpec plan")
        eval_plan.execution_checkpoint()
        phase = EvaluationPhase.parse(phase)
        metadata = dict(eval_plan.metadata)
        raw_argv = metadata.get("cli_argv")
        if not isinstance(raw_argv, list) or not all(
            isinstance(item, str) for item in raw_argv
        ):
            raise ValueError("phased eval requires structured base eval argv")
        base_summary_path = metadata.get("summary_path")
        if not isinstance(base_summary_path, str) or not base_summary_path:
            raise ValueError("phased eval requires a base summary path")
        phase_job_id = f"{eval_plan.job_id}-{phase.value.replace('_', '-')}"
        phase_summary_path = posixpath.join(
            posixpath.dirname(base_summary_path),
            phase.value,
            "summary.json",
        )
        phase_argv = _replace_separate_cli_option(
            raw_argv,
            "--job-id",
            phase_job_id,
        )
        phase_argv = _replace_separate_cli_option(
            phase_argv,
            "--summary-path",
            phase_summary_path,
        )
        if "--evaluation-phase" in phase_argv or "--evaluation-plan-hash" in phase_argv:
            raise ValueError("base eval argv already declares phased-eval options")
        phase_argv.extend(
            [
                "--evaluation-phase",
                phase.value,
                "--evaluation-plan-hash",
                canonical_json_hash(eval_plan.to_public_dict()),
            ]
        )
        metadata.update(
            {
                "cli_argv": phase_argv,
                "cli_command": " ".join(phase_argv),
                "summary_path": phase_summary_path,
                "evaluation_phase": phase.value,
                "evaluation_plan_hash": canonical_json_hash(
                    eval_plan.to_public_dict()
                ),
            }
        )
        return EvalPhaseJobSpec(
            job_id=phase_job_id,
            evaluation_plan=eval_plan,
            phase=phase,
            metadata=metadata,
            attempt_scope=attempt_scope,
        )

    def materialize_baseline_eval(
        self,
        *,
        session_id: str,
        task_spec: TaskSpec,
        baseline_manifest: BaselineManifest,
        evaluation_binding: EvaluationBindingV1 | None = None,
        backend_execution_environment_manifest_hash: str | None = None,
    ) -> BaselineEvalJobSpec:
        session_id = validate_experiment_subject_id(session_id)
        report = self.check_compatibility(
            task_spec=task_spec,
            policy_initialization="warm_start",
        )
        if not report.compatible:
            raise ValueError(
                f"backend compatibility failed: {', '.join(report.reasons)}"
            )
        compatibility = validate_baseline_compatibility(
            manifest=baseline_manifest,
            task_spec=task_spec,
            backend_spec=self.backend_spec,
            robot=self.config.robot,
            observation_schema_hash=self.config.observation_schema_hash,
            action_schema_hash=self.config.action_schema_hash,
            policy_architecture=self.config.policy_architecture,
        )
        if not compatibility.compatible:
            raise ValueError(
                "baseline compatibility failed: " + ", ".join(compatibility.reasons)
            )
        self._assert_runtime_dynamics_matches_task(task_spec, phases=("eval",))
        evaluation_binding = evaluation_binding or self.evaluation_binding_for(
            task_spec
        )
        self.validate_session_evaluation_binding(
            task_spec=task_spec,
            evaluation_binding=evaluation_binding,
        )
        execution_environment_manifest_hash = (
            require_sha256_digest(
                backend_execution_environment_manifest_hash,
                "backend execution environment manifest hash",
            )
            if backend_execution_environment_manifest_hash is not None
            else None
        )

        subject = ExperimentSubject.baseline(baseline_manifest.baseline_id)
        phase = posixpath.join(
            "baseline",
            baseline_manifest.baseline_id,
            "eval",
        )
        summary_path = self._summary_path(round_id=session_id, phase=phase)
        job_id = f"{session_id}-baseline-eval"
        eval_argv = self._eval_argv(
            job_id=job_id,
            round_id=session_id,
            task_id=task_spec.task_id,
            subject=subject,
            checkpoint=baseline_manifest.checkpoint,
            summary_path=summary_path,
            subject_manifest_hash=str(baseline_manifest.manifest_hash),
            program_source_checksum=None,
            training_binding_hash=None,
            capability_profile_hash=str(
                task_spec.candidate_capability_profile.profile_hash
            ),
            evaluation_binding=evaluation_binding,
            execution_environment_manifest_hash=(execution_environment_manifest_hash),
        )
        metadata = {
            "backend_id": self.backend_id,
            **(
                {
                    "backend_execution_environment_manifest_hash": (
                        execution_environment_manifest_hash
                    )
                }
                if execution_environment_manifest_hash is not None
                else {}
            ),
            **_cli_metadata(eval_argv),
            "evaluation_subject": subject.to_public_dict(),
            "subject_manifest_hash": baseline_manifest.manifest_hash,
            "capability_profile_hash": (
                task_spec.candidate_capability_profile.profile_hash
            ),
            "summary_path": summary_path,
            "work_dir": self.config.work_dir,
            "checkpoint_policy": "sealed_warm_start_baseline_required",
            "fixed_eval_protocol": {
                "eval_protocol": "fixed_public_accept_and_sealed_validation",
                **self.config.evaluation_protocol().to_public_dict(),
                **dict(task_spec.fixed_eval_protocol),
            },
        }
        return BaselineEvalJobSpec(
            job_id=job_id,
            task_id=task_spec.task_id,
            subject=subject,
            subject_manifest_hash=str(baseline_manifest.manifest_hash),
            capability_profile_hash=str(
                task_spec.candidate_capability_profile.profile_hash
            ),
            checkpoint=baseline_manifest.checkpoint,
            public_seed_count=self.config.public_seed_count,
            evaluation_binding=evaluation_binding,
            sealed_validation_seed_count=require_positive_integer(
                self.config.sealed_validation_seed_count,
                "sealed_validation_seed_count",
            ),
            public_accept_seed_count=self.config.public_accept_seed_count,
            eval_num_envs=self.config.eval_num_envs,
            eval_protocol_version=self.config.eval_protocol_version,
            execution_device=self.config.execution_device,
            metadata=metadata,
        )

    def _train_argv(
        self,
        *,
        job_id: str | None = None,
        round_id: str,
        task_id: str,
        subject: ExperimentSubject,
        candidate: VerifiedCandidateArtifactRef | None,
        reference: VerifiedReferenceProgramArtifactRef | None,
        policy_initialization: PolicyInitialization,
        training_binding: TrainingBinding,
        subject_manifest_hash: str,
        program_source_checksum: str,
        expected_initial_policy_state_hash: str | None = None,
        summary_path: str | None = None,
        checkpoint_output_path: str | None = None,
        execution_environment_manifest_hash: str | None,
    ) -> list[str]:
        summary_path = summary_path or self._summary_path(
            round_id=round_id, phase="train"
        )
        checkpoint_output_path = checkpoint_output_path or self._checkpoint_output_path(
            round_id=round_id
        )
        job_id = job_id or f"{round_id}-train"
        if not isinstance(subject, ExperimentSubject) or subject.kind not in {
            "candidate",
            "reference",
        }:
            raise TypeError("train subject must be candidate or reference")
        if (candidate is None) == (reference is None):
            raise ValueError("train argv requires exactly one program artifact")
        if candidate is not None and subject != ExperimentSubject.candidate(
            candidate.candidate_id
        ):
            raise ValueError("candidate train subject mismatch")
        if reference is not None and subject != ExperimentSubject.reference(
            reference.reference_id
        ):
            raise ValueError("reference train subject mismatch")
        JobIdentity(
            job_id=job_id,
            task_id=task_id,
            subject=subject,
            subject_manifest_hash=subject_manifest_hash,
        )
        if execution_environment_manifest_hash is not None:
            require_sha256_digest(
                execution_environment_manifest_hash,
                "backend execution environment manifest hash",
            )
        if training_binding.initialization_mode != policy_initialization.kind:
            raise ValueError(
                "train argv initialization does not match training binding"
            )
        parts = [
            self.config.isaac_lab_python,
            self.config.train_entrypoint,
            "--job-id",
            job_id,
            "--round",
            round_id,
            "--task-id",
            task_id,
            "--subject-kind",
            subject.kind,
            "--subject-id",
            subject.subject_id,
            "--subject-manifest-hash",
            subject_manifest_hash,
            "--program-source-checksum",
            program_source_checksum,
        ]
        if candidate is not None:
            authorization = candidate.execution_authorization
            parts.extend(
                [
                    "--candidate",
                    candidate.path,
                    "--candidate-artifact-hash",
                    candidate.artifact_hash,
                ]
            )
            if authorization is not None:
                parts.extend(
                    [
                        "--candidate-verification-hash",
                        candidate.verification_hash,
                        "--candidate-execution-authorization-json",
                        authorization.canonical_json,
                        "--candidate-execution-authorization-hash",
                        authorization.authorization_hash,
                    ]
                )
            if self.config.candidate_failure_contract is not None:
                parts.extend(
                    [
                        "--candidate-failure-contract-hash",
                        self.config.candidate_failure_contract.contract_hash,
                    ]
                )
        else:
            assert reference is not None
            parts.extend(
                [
                    "--reference-artifact",
                    reference.path,
                    "--reference-artifact-hash",
                    reference.artifact_hash,
                ]
            )
        parts.extend(
            [
                "--budget-steps",
                str(self.config.default_budget_steps),
                "--init",
                policy_initialization.kind,
                "--training-binding-hash",
                str(training_binding.binding_hash),
                "--policy-architecture-hash",
                training_binding.policy_architecture_hash,
                "--optimizer-training-profile-hash",
                training_binding.optimizer_training_profile_hash,
                "--agent-configuration-hash",
                training_binding.agent_configuration_hash,
                "--agent-configuration-json",
                self.config.training_agent_configuration.to_canonical_json(),
                "--initialization-seed-schedule-hash",
                training_binding.initialization_seed_schedule_hash,
                "--capability-profile-hash",
                training_binding.capability_profile_hash,
                "--summary-path",
                summary_path,
                "--checkpoint-output",
                checkpoint_output_path,
                "--profile",
                self.config.train_profile,
                "--runtime-dynamics-hash",
                self._required_runtime_dynamics_hash(phase="train"),
                "--num-envs",
                str(self.config.train_num_envs),
                "--seed",
                str(self.config.train_seed),
                "--device",
                self.config.execution_device,
            ]
        )
        if execution_environment_manifest_hash is not None:
            parts.extend(
                [
                    "--backend-execution-environment-manifest-hash",
                    execution_environment_manifest_hash,
                ]
            )
        if expected_initial_policy_state_hash is not None:
            require_sha256_digest(
                expected_initial_policy_state_hash,
                "expected initial policy state hash",
            )
            parts.extend(
                [
                    "--expected-initial-policy-state-hash",
                    expected_initial_policy_state_hash,
                ]
            )
        if self.config.train_steps_per_iteration is not None:
            parts.extend(
                ["--num-steps-per-env", str(self.config.train_steps_per_iteration)]
            )
        if self.config.train_max_iterations is not None:
            parts.extend(["--max-iterations", str(self.config.train_max_iterations)])
        if self.config.runtime_config_path is not None:
            parts.extend(["--config", self.config.runtime_config_path])
        parts.append("--headless")
        init_checkpoint = None
        if policy_initialization.baseline is not None:
            init_checkpoint = policy_initialization.baseline.checkpoint
        if init_checkpoint is not None:
            init_checkpoint_path = (
                init_checkpoint.metadata.get("local_path") or init_checkpoint.uri
            )
            if not isinstance(init_checkpoint_path, str) or not init_checkpoint_path:
                raise ValueError(
                    "warm-start checkpoint local path must be a non-empty string"
                )
            parts.extend(
                [
                    "--init-checkpoint",
                    init_checkpoint_path,
                    "--init-checkpoint-sha256",
                    init_checkpoint.checksum,
                ]
            )
        return parts

    def _assert_training_budget_matches_task(self, task_spec: TaskSpec) -> None:
        invariants = dict(task_spec.invariants.values)
        checks = {
            "budget_steps": self.config.default_budget_steps,
            "train_num_envs": self.config.train_num_envs,
            "train_max_iterations": self.config.train_max_iterations,
            "train_steps_per_iteration": self.config.train_steps_per_iteration,
        }
        for key, actual in checks.items():
            if key not in invariants or actual is None:
                continue
            expected = require_positive_integer(
                invariants[key],
                f"task invariant {key}",
            )
            if expected != require_positive_integer(actual, key):
                raise ValueError(
                    f"task_budget_mismatch:{key}:expected={expected}:actual={actual}"
                )
        agent_configuration = self.config.training_agent_configuration
        hash_checks = {
            "policy_architecture_hash": (agent_configuration.policy_architecture_hash),
            "optimizer_training_profile_hash": (
                agent_configuration.optimizer_training_profile_hash
            ),
            "agent_configuration_hash": (agent_configuration.agent_configuration_hash),
        }
        for key, actual in hash_checks.items():
            expected = task_spec.fixed_train_protocol.get(key)
            if expected is None:
                raise ValueError(f"task_training_configuration_missing:{key}")
            if expected != actual:
                raise ValueError(
                    "task_training_configuration_mismatch:"
                    f"{key}:expected={expected}:actual={actual}"
                )
        self._assert_eval_protocol_matches_task(task_spec)

    def _assert_runtime_dynamics_matches_task(
        self,
        task_spec: TaskSpec,
        *,
        phases: tuple[str, ...] = ("train", "eval"),
    ) -> None:
        invariant_values = dict(task_spec.invariants.values)
        checks = (
            (
                "train_runtime_profile",
                self.config.train_profile,
                "train",
                self.config.train_runtime_dynamics_hash,
            ),
            (
                "eval_runtime_profile",
                self.config.eval_profile,
                "eval",
                self.config.eval_runtime_dynamics_hash,
            ),
        )
        for profile_key, actual_profile, phase, actual_hash in checks:
            if phase not in phases:
                continue
            expected_profile = invariant_values.get(profile_key)
            expected_hash = invariant_values.get(f"{phase}_runtime_dynamics_hash")
            if expected_profile is None and expected_hash is None:
                continue
            if str(expected_profile) != actual_profile:
                raise ValueError(
                    "task_runtime_profile_mismatch:"
                    f"{phase}:expected={expected_profile}:actual={actual_profile}"
                )
            if not is_sha256_checksum(actual_hash):
                raise ValueError(f"backend_runtime_dynamics_hash_missing:{phase}")
            if str(expected_hash) != actual_hash:
                raise ValueError(
                    "task_runtime_dynamics_hash_mismatch:"
                    f"{phase}:expected={expected_hash}:actual={actual_hash}"
                )

    def _assert_evaluation_binding_matches_task(
        self,
        task_spec: TaskSpec,
    ) -> None:
        binding = self.config.evaluation_binding
        acceptance_protocol = task_spec.acceptance_protocol
        if acceptance_protocol is None:
            raise ValueError(
                "task_evaluation_binding_mismatch:acceptance_protocol_missing"
            )
        checks = {
            "eval_protocol_hash": self.config.evaluation_protocol().protocol_hash,
            "acceptance_protocol_hash": acceptance_protocol.protocol_hash,
            "task_protocol_hash": task_spec.resolved_task_protocol_hash(),
        }
        for field_name, expected in checks.items():
            actual = getattr(binding, field_name)
            if actual != expected:
                raise ValueError(
                    "task_evaluation_binding_mismatch:"
                    f"{field_name}:expected={expected}:actual={actual}"
                )
        for source, values in (
            ("task_invariants", task_spec.invariants.values),
            ("fixed_eval_protocol", task_spec.fixed_eval_protocol),
        ):
            expected_seed_cohort_hash = values.get("seed_cohort_hash")
            if expected_seed_cohort_hash is None:
                raise ValueError(
                    "task_evaluation_binding_mismatch:"
                    f"seed_cohort_hash:source={source}:expected=<missing>:"
                    f"actual={binding.seed_cohort_hash}"
                )
            if str(expected_seed_cohort_hash) != binding.seed_cohort_hash:
                raise ValueError(
                    "task_evaluation_binding_mismatch:"
                    f"seed_cohort_hash:source={source}:"
                    f"expected={expected_seed_cohort_hash}:"
                    f"actual={binding.seed_cohort_hash}"
                )

    def _required_runtime_dynamics_hash(self, *, phase: str) -> str:
        value = (
            self.config.train_runtime_dynamics_hash
            if phase == "train"
            else self.config.eval_runtime_dynamics_hash
        )
        if not is_sha256_checksum(value):
            raise ValueError(f"backend_runtime_dynamics_hash_missing:{phase}")
        assert isinstance(value, str)
        return value

    def _assert_eval_protocol_matches_task(self, task_spec: TaskSpec) -> None:
        actual = self.config.evaluation_protocol().to_public_dict()
        required_keys = (
            "eval_protocol_version",
            "public_seed_count",
            "public_accept_seed_count",
            "sealed_validation_seed_count",
            "eval_num_envs",
            "eval_protocol_hash",
        )
        invariant_values = dict(task_spec.invariants.values)
        for key in required_keys:
            self._assert_protocol_field_matches(
                key=key,
                expected_values=invariant_values,
                actual_value=actual[key],
                source="task_invariants",
            )
        fixed_eval_protocol = dict(task_spec.fixed_eval_protocol)
        for key in required_keys:
            self._assert_protocol_field_matches(
                key=key,
                expected_values=fixed_eval_protocol,
                actual_value=actual[key],
                source="fixed_eval_protocol",
            )

    def _assert_protocol_field_matches(
        self,
        *,
        key: str,
        expected_values: dict,
        actual_value: object,
        source: str,
    ) -> None:
        if key not in expected_values:
            raise ValueError(
                f"task_eval_protocol_mismatch:{key}:source={source}:expected=<missing>:actual={actual_value}"
            )
        expected = expected_values[key]
        if isinstance(actual_value, int) and not isinstance(actual_value, bool):
            expected = require_positive_integer(
                expected,
                f"{source} {key}",
            )
            matches = expected == actual_value
        else:
            matches = isinstance(expected, str) and expected == actual_value
        if not matches:
            raise ValueError(
                f"task_eval_protocol_mismatch:{key}:source={source}:expected={expected}:actual={actual_value}"
            )

    def _eval_argv(
        self,
        *,
        job_id: str | None = None,
        round_id: str,
        task_id: str,
        subject: ExperimentSubject,
        checkpoint: CheckpointRef | CheckpointOutputPlan,
        evaluation_binding: EvaluationBindingV1,
        summary_path: str | None = None,
        subject_manifest_hash: str | None = None,
        program_source_checksum: str | None = None,
        training_binding_hash: str | None = None,
        capability_profile_hash: str,
        execution_environment_manifest_hash: str | None,
    ) -> list[str]:
        summary_path = summary_path or self._summary_path(
            round_id=round_id, phase="eval"
        )
        if not isinstance(subject, ExperimentSubject):
            raise TypeError("eval subject must be ExperimentSubject")
        job_id = job_id or f"{round_id}-eval"
        JobIdentity(
            job_id=job_id,
            task_id=task_id,
            subject=subject,
            subject_manifest_hash=subject_manifest_hash,
        )
        if execution_environment_manifest_hash is not None:
            require_sha256_digest(
                execution_environment_manifest_hash,
                "backend execution environment manifest hash",
            )
        checkpoint_path = (
            checkpoint.metadata.get("local_path") or checkpoint.uri
            if isinstance(checkpoint, CheckpointRef)
            else checkpoint.uri
        )
        if not isinstance(checkpoint_path, str) or not checkpoint_path:
            raise ValueError("eval checkpoint local path must be a non-empty string")
        binding = evaluation_binding
        parts = [
            self.config.isaac_lab_python,
            self.config.eval_entrypoint,
            "--job-id",
            job_id,
            "--round",
            round_id,
            "--task-id",
            task_id,
            "--subject-kind",
            subject.kind,
            "--subject-id",
            subject.subject_id,
            "--checkpoint",
            checkpoint_path,
        ]
        if isinstance(checkpoint, CheckpointRef):
            parts.extend(
                [
                    "--checkpoint-sha256",
                    checkpoint.checksum,
                ]
            )
        parts.extend(
            [
                "--public-seeds",
                str(self.config.public_seed_count),
                "--public-accept-seeds",
                str(self.config.public_accept_seed_count),
                "--sealed-validation-seeds",
                str(self.config.sealed_validation_seed_count),
                "--eval-protocol-version",
                self.config.eval_protocol_version,
                "--task-protocol-hash",
                binding.task_protocol_hash,
                "--acceptance-protocol-hash",
                binding.acceptance_protocol_hash,
                "--seed-cohort-hash",
                binding.seed_cohort_hash,
                "--evaluation-binding-hash",
                binding.binding_hash,
                "--capability-profile-hash",
                capability_profile_hash,
                "--summary-path",
                summary_path,
                "--profile",
                self.config.eval_profile,
                "--runtime-dynamics-hash",
                self._required_runtime_dynamics_hash(phase="eval"),
                "--num-envs",
                str(self.config.eval_num_envs),
                "--seed",
                str(self.config.eval_base_seed),
                "--device",
                self.config.execution_device,
            ]
        )
        if execution_environment_manifest_hash is not None:
            parts.extend(
                [
                    "--backend-execution-environment-manifest-hash",
                    execution_environment_manifest_hash,
                ]
            )
        if subject_manifest_hash is not None:
            parts.extend(["--subject-manifest-hash", subject_manifest_hash])
        if training_binding_hash is not None:
            parts.extend(["--training-binding-hash", training_binding_hash])
        if program_source_checksum is not None:
            parts.extend(["--program-source-checksum", program_source_checksum])
        if self.config.runtime_config_path is not None:
            parts.extend(["--config", self.config.runtime_config_path])
        parts.append("--headless")
        return parts

    def _training_binding(
        self,
        *,
        task_spec: TaskSpec,
        policy_initialization: PolicyInitialization,
    ) -> TrainingBinding:
        checkpoint = policy_initialization.baseline
        checkpoint_hash = (
            checkpoint.checkpoint.checksum if checkpoint is not None else None
        )
        agent_configuration = self.config.training_agent_configuration
        return TrainingBinding(
            initialization_mode=policy_initialization.kind,
            initialization_checkpoint_hash=checkpoint_hash,
            policy_architecture_hash=str(agent_configuration.policy_architecture_hash),
            agent_configuration_hash=str(agent_configuration.agent_configuration_hash),
            initialization_seed_schedule_hash=(
                initialization_seed_schedule_hash(self.config.train_seed)
            ),
            optimizer_training_profile_hash=str(
                agent_configuration.optimizer_training_profile_hash
            ),
            budget_steps=self.config.default_budget_steps,
            private_training_seed_resolver_hash=(
                fixed_training_seed_resolver_hash(self.config.train_seed)
            ),
            capability_profile_hash=str(
                task_spec.candidate_capability_profile.profile_hash
            ),
        )

    def _summary_path(self, *, round_id: str, phase: str) -> str:
        return posixpath.join(
            self.config.artifact_root.rstrip("/"), round_id, phase, "summary.json"
        )

    def _checkpoint_output_path(self, *, round_id: str) -> str:
        return posixpath.join(
            self.config.artifact_root.rstrip("/"), round_id, "checkpoints", "best.pt"
        )


def _cli_metadata(argv: list[str]) -> dict[str, Any]:
    """Keep the human-readable command as a projection of executable argv."""

    return {
        "cli_argv": list(argv),
        "cli_command": " ".join(argv),
    }


def _replace_separate_cli_option(
    argv: list[str],
    option: str,
    value: str,
) -> list[str]:
    positions = [index for index, item in enumerate(argv) if item == option]
    if len(positions) != 1:
        raise ValueError(f"eval argv requires exactly one {option}")
    index = positions[0]
    if index + 1 >= len(argv):
        raise ValueError(f"eval argv {option} is missing its value")
    updated = list(argv)
    updated[index + 1] = value
    return updated
