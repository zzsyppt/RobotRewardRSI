"""Isaac Lab + RSL-RL backend job materializer."""

from __future__ import annotations

import posixpath
from dataclasses import InitVar, dataclass
from typing import Any, Mapping

from true_rl_agent.backends.base import BackendAdapter, BackendJobBundle
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
    is_sha256_checksum,
)
from true_rl_agent.evaluation.protocol import (
    EvaluationProtocol,
)
from true_rl_agent.evaluation.subject import (
    ExperimentSubject,
    validate_experiment_subject_id,
)
from true_rl_agent.policy import PolicyInitialization
from true_rl_agent.policy_baselines import (
    BaselineManifest,
    validate_baseline_compatibility,
)
from true_rl_agent.runner.budget import build_budget_accounting
from true_rl_agent.runner.eval_isolation import validate_candidate_id
from true_rl_agent.runner.job_identity import JobIdentity
from true_rl_agent.runner.specs import (
    BaselineEvalJobSpec,
    CheckpointOutputPlan,
    CheckpointRef,
    EvalJobSpec,
    TrainJobSpec,
    require_positive_integer,
)
from true_rl_agent.tasks.base import TaskSpec
from true_rl_agent.training_binding import (
    TrainingBinding,
    fixed_training_seed_resolver_hash,
    initialization_seed_schedule_hash,
)


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
    def __init__(self, config: IsaacLabRslRlBackendConfig) -> None:
        self.config = config

    @property
    def future_curriculum_capabilities(self) -> Mapping[str, Mapping[str, Any]]:
        return self.config.curriculum_capabilities

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

    def check_compatibility(self, *, task_spec: TaskSpec, policy_initialization: str) -> BackendCompatibilityReport:
        reasons: list[str] = []
        if task_spec.task_id not in self.backend_spec.supported_tasks:
            reasons.append(f"unsupported_task:{task_spec.task_id}")
        if policy_initialization not in self.backend_spec.supported_policy_initializations:
            reasons.append(f"unsupported_policy_initialization:{policy_initialization}")
        return BackendCompatibilityReport(compatible=not reasons, reasons=tuple(reasons))

    def materialize_jobs(
        self,
        *,
        round_id: str,
        task_spec: TaskSpec,
        candidate: VerifiedCandidateArtifactRef,
        policy_initialization: PolicyInitialization,
        evaluation_binding: EvaluationBindingV1 | None = None,
    ) -> BackendJobBundle:
        candidate = verify_verified_candidate_artifact(candidate)
        report = self.check_compatibility(
            task_spec=task_spec,
            policy_initialization=policy_initialization.kind,
        )
        if not report.compatible:
            raise ValueError(f"backend compatibility failed: {', '.join(report.reasons)}")
        self._assert_training_budget_matches_task(task_spec)
        self._assert_runtime_dynamics_matches_task(task_spec)
        evaluation_binding = (
            evaluation_binding or self.evaluation_binding_for(task_spec)
        )
        self.validate_session_evaluation_binding(
            task_spec=task_spec,
            evaluation_binding=evaluation_binding,
        )

        planned_checkpoint = CheckpointOutputPlan(
            uri=f"artifact://{round_id}/checkpoints/best.pt",
        )
        train_summary_path = self._summary_path(round_id=round_id, phase="train")
        eval_summary_path = self._summary_path(round_id=round_id, phase="eval")
        checkpoint_output_path = self._checkpoint_output_path(round_id=round_id)
        eval_protocol = self.config.evaluation_protocol()
        subject = ExperimentSubject.candidate(candidate.candidate_id)
        training_binding = self._training_binding(
            task_spec=task_spec,
            policy_initialization=policy_initialization,
        )
        train_job_id = f"{round_id}-train"
        eval_job_id = f"{round_id}-eval"
        train_argv = self._train_argv(
            job_id=train_job_id,
            round_id=round_id,
            task_id=task_spec.task_id,
            candidate=candidate,
            policy_initialization=policy_initialization,
            training_binding=training_binding,
            subject_manifest_hash=candidate.subject_manifest_hash,
            program_source_checksum=candidate.program_source_checksum,
            summary_path=train_summary_path,
            checkpoint_output_path=checkpoint_output_path,
        )
        train_spec = TrainJobSpec(
            job_id=train_job_id,
            task_id=task_spec.task_id,
            subject=subject,
            subject_manifest_hash=candidate.subject_manifest_hash,
            program_source_checksum=candidate.program_source_checksum,
            policy_initialization=policy_initialization,
            training_binding=training_binding,
            budget_steps=self.config.default_budget_steps,
            candidate_artifact=candidate,
            metadata={
                "backend_id": self.backend_id,
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
                "policy_architecture_hash": (
                    training_binding.policy_architecture_hash
                ),
                "optimizer_training_profile_hash": (
                    training_binding.optimizer_training_profile_hash
                ),
                "agent_configuration_hash": (
                    training_binding.agent_configuration_hash
                ),
                "fixed_train_protocol": {
                    **dict(task_spec.fixed_train_protocol),
                    "budget_steps": self.config.default_budget_steps,
                },
            },
        )
        eval_argv = self._eval_argv(
            job_id=eval_job_id,
            round_id=round_id,
            task_id=task_spec.task_id,
            subject=subject,
            checkpoint=planned_checkpoint,
            summary_path=eval_summary_path,
            subject_manifest_hash=candidate.subject_manifest_hash,
            program_source_checksum=candidate.program_source_checksum,
            training_binding_hash=str(training_binding.binding_hash),
            capability_profile_hash=str(
                task_spec.candidate_capability_profile.profile_hash
            ),
            evaluation_binding=evaluation_binding,
        )
        eval_spec = EvalJobSpec(
            job_id=eval_job_id,
            task_id=task_spec.task_id,
            subject=subject,
            subject_manifest_hash=candidate.subject_manifest_hash,
            program_source_checksum=candidate.program_source_checksum,
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
            checkpoint_output_plan=planned_checkpoint,
            metadata={
                "backend_id": self.backend_id,
                **_cli_metadata(eval_argv),
                "evaluation_subject": subject.to_public_dict(),
                "subject_manifest_hash": candidate.subject_manifest_hash,
                "program_source_checksum": candidate.program_source_checksum,
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

    def materialize_baseline_eval(
        self,
        *,
        session_id: str,
        task_spec: TaskSpec,
        baseline_manifest: BaselineManifest,
        evaluation_binding: EvaluationBindingV1 | None = None,
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
                "baseline compatibility failed: "
                + ", ".join(compatibility.reasons)
            )
        self._assert_runtime_dynamics_matches_task(task_spec, phases=("eval",))
        evaluation_binding = (
            evaluation_binding or self.evaluation_binding_for(task_spec)
        )
        self.validate_session_evaluation_binding(
            task_spec=task_spec,
            evaluation_binding=evaluation_binding,
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
        )
        metadata = {
            "backend_id": self.backend_id,
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
            metadata=metadata,
        )

    def _train_argv(
        self,
        *,
        job_id: str | None = None,
        round_id: str,
        task_id: str,
        candidate: VerifiedCandidateArtifactRef,
        policy_initialization: PolicyInitialization,
        training_binding: TrainingBinding,
        subject_manifest_hash: str,
        program_source_checksum: str,
        summary_path: str | None = None,
        checkpoint_output_path: str | None = None,
    ) -> list[str]:
        summary_path = summary_path or self._summary_path(round_id=round_id, phase="train")
        checkpoint_output_path = checkpoint_output_path or self._checkpoint_output_path(round_id=round_id)
        job_id = job_id or f"{round_id}-train"
        candidate_id = validate_candidate_id(candidate.candidate_id)
        subject = ExperimentSubject.candidate(candidate_id)
        JobIdentity(
            job_id=job_id,
            task_id=task_id,
            subject=subject,
            subject_manifest_hash=subject_manifest_hash,
        )
        if training_binding.initialization_mode != policy_initialization.kind:
            raise ValueError("train argv initialization does not match training binding")
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
            "--candidate",
            candidate.path,
            "--candidate-artifact-hash",
            candidate.artifact_hash,
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
        ]
        if self.config.train_steps_per_iteration is not None:
            parts.extend(["--num-steps-per-env", str(self.config.train_steps_per_iteration)])
        if self.config.train_max_iterations is not None:
            parts.extend(["--max-iterations", str(self.config.train_max_iterations)])
        if self.config.runtime_config_path is not None:
            parts.extend(["--config", self.config.runtime_config_path])
        parts.append("--headless")
        init_checkpoint = None
        if policy_initialization.baseline is not None:
            init_checkpoint = policy_initialization.baseline.checkpoint
        if init_checkpoint is not None:
            init_checkpoint_path = init_checkpoint.metadata.get("local_path") or init_checkpoint.uri
            if not isinstance(init_checkpoint_path, str) or not init_checkpoint_path:
                raise ValueError("warm-start checkpoint local path must be a non-empty string")
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
                raise ValueError(f"task_budget_mismatch:{key}:expected={expected}:actual={actual}")
        agent_configuration = self.config.training_agent_configuration
        hash_checks = {
            "policy_architecture_hash": (
                agent_configuration.policy_architecture_hash
            ),
            "optimizer_training_profile_hash": (
                agent_configuration.optimizer_training_profile_hash
            ),
            "agent_configuration_hash": (
                agent_configuration.agent_configuration_hash
            ),
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
            raise ValueError("task_evaluation_binding_mismatch:acceptance_protocol_missing")
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
            raise ValueError(f"task_eval_protocol_mismatch:{key}:source={source}:expected=<missing>:actual={actual_value}")
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
            raise ValueError(f"task_eval_protocol_mismatch:{key}:source={source}:expected={expected}:actual={actual_value}")

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
    ) -> list[str]:
        summary_path = summary_path or self._summary_path(round_id=round_id, phase="eval")
        if not isinstance(subject, ExperimentSubject):
            raise TypeError("eval subject must be ExperimentSubject")
        job_id = job_id or f"{round_id}-eval"
        JobIdentity(
            job_id=job_id,
            task_id=task_id,
            subject=subject,
            subject_manifest_hash=subject_manifest_hash,
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
            ]
        )
        if subject_manifest_hash is not None:
            parts.extend(
                ["--subject-manifest-hash", subject_manifest_hash]
            )
        if training_binding_hash is not None:
            parts.extend(
                ["--training-binding-hash", training_binding_hash]
            )
        if program_source_checksum is not None:
            parts.extend(
                ["--program-source-checksum", program_source_checksum]
            )
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
            policy_architecture_hash=str(
                agent_configuration.policy_architecture_hash
            ),
            agent_configuration_hash=str(
                agent_configuration.agent_configuration_hash
            ),
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
        return posixpath.join(self.config.artifact_root.rstrip("/"), round_id, phase, "summary.json")

    def _checkpoint_output_path(self, *, round_id: str) -> str:
        return posixpath.join(self.config.artifact_root.rstrip("/"), round_id, "checkpoints", "best.pt")


def _cli_metadata(argv: list[str]) -> dict[str, Any]:
    """Keep the human-readable command as a projection of executable argv."""

    return {
        "cli_argv": list(argv),
        "cli_command": " ".join(argv),
    }
