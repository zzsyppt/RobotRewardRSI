"""Pure preflight contracts for production Stair subprocess entrypoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from true_rl_agent.backends.rsl_rl_training_config import RslRlAgentConfiguration
from true_rl_agent.candidates.artifact import (
    CandidateExecutionAuthorization,
    SealedCandidateArtifactRef,
    load_sealed_candidate_artifact,
)
from true_rl_agent.candidates.schema import CandidateChangeSet
from true_rl_agent.composition.cell_manifest import ResolvedCellManifestV1
from true_rl_agent.evaluation.binding import (
    EvaluationBindingV1,
    require_sha256_checksum,
    require_sha256_digest,
)
from true_rl_agent.evaluation.subject import ExperimentSubject
from true_rl_agent.reference_artifact import (
    VerifiedReferenceProgramArtifactRef,
    load_reference_program_artifact,
)
from true_rl_agent.runner.budget import build_budget_accounting
from true_rl_agent.runner.job_identity import JobIdentity
from true_rl_agent.tasks.stair_climbing.candidate_admission import (
    VerifiedStairCandidateExecutionAuthorization,
    validate_stair_candidate_execution_authorization,
)
from true_rl_agent.tasks.stair_climbing.candidate_failure import (
    STAIR_CANDIDATE_FAILURE_CONTRACT,
)
from true_rl_agent.tasks.stair_climbing.config import (
    STAIR_TASK_ID,
    StairClimbingTaskConfig,
)
from true_rl_agent.tasks.stair_climbing.local_paths import resolve_existing_file
from true_rl_agent.tasks.stair_climbing.seed_policy import (
    STAIR_FINAL_CONFIRMATION_EVAL_NAMESPACE,
    STAIR_FORMAL_EVAL_NAMESPACE,
)
from true_rl_agent.tasks.stair_climbing.row_catalog import stair_robot_row
from true_rl_agent.training_binding import TrainingBinding
from true_rl_agent.tasks.stair_climbing.unified_protocol import (
    STAIR_UNIFIED_PROTOCOL_RELATIVE_PATH,
)


@dataclass(frozen=True)
class StairTrainEntrypointContext:
    config: StairClimbingTaskConfig
    identity: JobIdentity
    program_source_checksum: str
    candidate_artifact: SealedCandidateArtifactRef | None
    candidate_verification_hash: str | None
    candidate_execution_authorization: (
        VerifiedStairCandidateExecutionAuthorization | None
    )
    reference_artifact: VerifiedReferenceProgramArtifactRef | None
    training_binding: TrainingBinding
    agent_configuration: RslRlAgentConfiguration
    budget_accounting: dict[str, int | str]
    expected_initial_policy_state_hash: str | None
    candidate_failure_contract_hash: str | None
    resolved_cell_manifest: ResolvedCellManifestV1 | None


@dataclass(frozen=True)
class StairEvalEntrypointContext:
    config: StairClimbingTaskConfig
    identity: JobIdentity
    evaluation_binding: EvaluationBindingV1
    evaluation_subject: ExperimentSubject
    checkpoint_path: Path
    checkpoint_hash: str
    subject_manifest_hash: str
    program_source_checksum: str
    training_binding_hash: str
    capability_profile_hash: str
    seed_namespace: str


STAIR_EVAL_PURPOSE_FORMAL = "formal_generation"
STAIR_EVAL_PURPOSE_FINAL_CONFIRMATION = "final_confirmation"


def validate_stair_train_invocation(
    args: Any,
    *,
    project_root: Path,
    resolved_cell_manifest: ResolvedCellManifestV1 | None = None,
) -> StairTrainEntrypointContext:
    """Reject any train argv that differs from the task-owned config snapshot."""

    config_path = _canonical_config_path(
        args.config,
        project_root=project_root,
        resolved_cell_manifest=resolved_cell_manifest,
    )
    defaults = _resolved_stair_config(
        config_path,
        resolved_cell_manifest=resolved_cell_manifest,
    )
    if defaults.protocol_state != "active":
        raise ValueError("Stair unified protocol is draft; formal training is disabled")
    config = StairClimbingTaskConfig(
        protocol_v2_path=config_path,
        budget_steps=_positive_int(args.budget_steps, "budget_steps"),
        train_num_envs=_positive_int(args.num_envs, "num_envs"),
        train_max_iterations=_positive_int(args.max_iterations, "max_iterations"),
        train_steps_per_iteration=_positive_int(
            args.num_steps_per_env,
            "num_steps_per_env",
        ),
        eval_num_envs=defaults.eval_num_envs,
        public_seed_count=defaults.public_seed_count,
        public_accept_seed_count=defaults.public_accept_seed_count,
        sealed_validation_seed_count=defaults.sealed_validation_seed_count,
        eval_protocol_version=defaults.eval_protocol_version,
        train_seed=_nonnegative_int(args.seed, "seed"),
        runner_timeout_s=defaults.runner_timeout_s,
        train_profile=_text(args.profile, "profile"),
        eval_profile=defaults.eval_profile,
        cell_variant_id=defaults.cell_variant_id,
        resolved_cell_task_protocol_hash=(
            defaults.resolved_cell_task_protocol_hash
        ),
    )
    if args.task_id != STAIR_TASK_ID:
        raise ValueError("stair train task_id mismatch")
    if args.subject_kind not in {"candidate", "reference"}:
        raise ValueError("stair train subject must be candidate or reference")
    if args.init != "random":
        raise ValueError("stair training requires random initialization")
    if getattr(args, "init_checkpoint", None) is not None or getattr(
        args,
        "init_checkpoint_sha256",
        None,
    ) is not None:
        raise ValueError("stair random training cannot carry an init checkpoint")
    if args.runtime_dynamics_hash != config.train_runtime_profile().runtime_dynamics_hash:
        raise ValueError("stair train runtime dynamics hash mismatch")

    subject = ExperimentSubject(
        kind=args.subject_kind,
        subject_id=_text(args.subject_id, "subject_id"),
    )
    identity = JobIdentity(
        job_id=_text(args.job_id, "job_id"),
        task_id=STAIR_TASK_ID,
        subject=subject,
        subject_manifest_hash=require_sha256_digest(
            args.subject_manifest_hash,
            "stair train subject manifest hash",
        ),
    )
    source_checksum = require_sha256_checksum(
        args.program_source_checksum,
        "stair train program source checksum",
    )
    candidate_artifact: SealedCandidateArtifactRef | None = None
    candidate_verification_hash: str | None = None
    candidate_execution_authorization: (
        VerifiedStairCandidateExecutionAuthorization | None
    ) = None
    reference_artifact: VerifiedReferenceProgramArtifactRef | None = None
    if subject.kind == "candidate":
        _require_absent(args.reference_artifact, "reference_artifact")
        _require_absent(
            args.reference_artifact_hash,
            "reference_artifact_hash",
        )
        artifact_path = _absolute_artifact_path(args.candidate, "candidate")
        candidate_artifact = load_sealed_candidate_artifact(
            artifact_path,
            change_set=CandidateChangeSet(frozenset()),
            expected_candidate_id=identity.subject.subject_id,
            expected_artifact_hash=require_sha256_digest(
                args.candidate_artifact_hash,
                "stair train candidate artifact hash",
            ),
            expected_subject_manifest_hash=identity.subject_manifest_hash,
            expected_program_source_checksum=source_checksum,
        )
        candidate_verification_hash = require_sha256_digest(
            args.candidate_verification_hash,
            "stair train candidate verification hash",
        )
        authorization = CandidateExecutionAuthorization.from_canonical_json(
            args.candidate_execution_authorization_json,
            expected_authorization_hash=require_sha256_digest(
                args.candidate_execution_authorization_hash,
                "stair train candidate execution authorization hash",
            ),
        )
        candidate_execution_authorization = (
            validate_stair_candidate_execution_authorization(
                authorization,
                artifact=candidate_artifact,
                config=config,
                resolved_cell_manifest=resolved_cell_manifest,
            )
        )
        expected_initial_policy_state_hash = require_sha256_digest(
            args.expected_initial_policy_state_hash,
            "stair train expected initial policy state hash",
        )
        candidate_failure_contract_hash = require_sha256_digest(
            args.candidate_failure_contract_hash,
            "stair candidate failure contract hash",
        )
        if (
            candidate_failure_contract_hash
            != STAIR_CANDIDATE_FAILURE_CONTRACT.contract_hash
        ):
            raise ValueError("stair candidate failure contract hash mismatch")
    else:
        _require_absent(args.candidate, "candidate")
        _require_absent(args.candidate_artifact_hash, "candidate_artifact_hash")
        _require_absent(
            args.candidate_verification_hash,
            "candidate_verification_hash",
        )
        _require_absent(
            args.candidate_execution_authorization_json,
            "candidate_execution_authorization_json",
        )
        _require_absent(
            args.candidate_execution_authorization_hash,
            "candidate_execution_authorization_hash",
        )
        artifact_path = _absolute_artifact_path(
            args.reference_artifact,
            "reference_artifact",
        )
        reference_artifact = load_reference_program_artifact(
            artifact_path,
            expected_artifact_hash=require_sha256_digest(
                args.reference_artifact_hash,
                "stair train reference artifact hash",
            ),
            expected_reference_id=identity.subject.subject_id,
            expected_task_id=STAIR_TASK_ID,
            expected_subject_manifest_hash=identity.subject_manifest_hash,
            expected_program_source_checksum=source_checksum,
        )
        if (
            reference_artifact.program.manifest.capability_profile_hash
            != config.candidate_capability_profile.profile_hash
        ):
            raise ValueError("stair train reference capability profile mismatch")
        _require_absent(
            args.expected_initial_policy_state_hash,
            "expected_initial_policy_state_hash",
        )
        expected_initial_policy_state_hash = None
        _require_absent(
            args.candidate_failure_contract_hash,
            "candidate_failure_contract_hash",
        )
        candidate_failure_contract_hash = None

    agent_configuration = RslRlAgentConfiguration.from_json(
        _text(args.agent_configuration_json, "agent_configuration_json")
    )
    expected_agent = config.training_agent_configuration()
    if agent_configuration != expected_agent:
        raise ValueError("stair train agent configuration differs from task config")
    training_binding = TrainingBinding(
        initialization_mode="random",
        initialization_checkpoint_hash=None,
        policy_architecture_hash=require_sha256_digest(
            args.policy_architecture_hash,
            "stair policy architecture hash",
        ),
        agent_configuration_hash=require_sha256_digest(
            args.agent_configuration_hash,
            "stair agent configuration hash",
        ),
        initialization_seed_schedule_hash=require_sha256_digest(
            args.initialization_seed_schedule_hash,
            "stair initialization seed schedule hash",
        ),
        optimizer_training_profile_hash=require_sha256_digest(
            args.optimizer_training_profile_hash,
            "stair optimizer training profile hash",
        ),
        budget_steps=config.budget_steps,
        private_training_seed_resolver_hash=(
            config.private_training_seed_resolver_hash()
        ),
        capability_profile_hash=require_sha256_digest(
            args.capability_profile_hash,
            "stair capability profile hash",
        ),
        task_protocol_hash=(
            config.training_binding_task_protocol_hash
            if config.is_unified_protocol
            else None
        ),
        binding_hash=require_sha256_digest(
            args.training_binding_hash,
            "stair training binding hash",
        ),
    )
    expected_binding = config.formal_training_binding()
    if training_binding != expected_binding:
        raise ValueError("stair train binding differs from task config")

    return StairTrainEntrypointContext(
        config=config,
        identity=identity,
        program_source_checksum=source_checksum,
        candidate_artifact=candidate_artifact,
        candidate_verification_hash=candidate_verification_hash,
        candidate_execution_authorization=candidate_execution_authorization,
        reference_artifact=reference_artifact,
        training_binding=training_binding,
        agent_configuration=agent_configuration,
        budget_accounting=build_budget_accounting(
            budget_steps=config.budget_steps,
            num_envs=config.train_num_envs,
            num_steps_per_env=config.train_steps_per_iteration,
            max_iterations=config.train_max_iterations,
        ),
        expected_initial_policy_state_hash=(
            expected_initial_policy_state_hash
        ),
        candidate_failure_contract_hash=candidate_failure_contract_hash,
        resolved_cell_manifest=resolved_cell_manifest,
    )


def validate_stair_eval_invocation(
    args: Any,
    *,
    project_root: Path,
    resolved_cell_manifest: ResolvedCellManifestV1 | None = None,
) -> StairEvalEntrypointContext:
    """Bind an eval invocation to one immutable Stair comparison domain."""

    config_path = _canonical_config_path(
        args.config,
        project_root=project_root,
        resolved_cell_manifest=resolved_cell_manifest,
    )
    config = _resolved_stair_config(
        config_path,
        resolved_cell_manifest=resolved_cell_manifest,
    )
    if config.protocol_state != "active":
        raise ValueError("Stair unified protocol is draft; formal evaluation is disabled")
    if args.task_id != STAIR_TASK_ID:
        raise ValueError("stair eval task_id mismatch")
    if args.subject_kind not in {"candidate", "reference"}:
        raise ValueError("stair eval subject must be candidate or reference")
    if args.profile != config.eval_profile:
        raise ValueError("stair eval profile mismatch")
    if args.runtime_dynamics_hash != config.eval_runtime_profile().runtime_dynamics_hash:
        raise ValueError("stair eval runtime dynamics hash mismatch")
    checks = {
        "public_seeds": config.public_seed_count,
        "public_accept_seeds": config.public_accept_seed_count,
        "sealed_validation_seeds": config.sealed_validation_seed_count,
        "num_envs": config.eval_num_envs,
        "seed": config.seed_resolver().base_seed,
    }
    for field_name, expected in checks.items():
        if _nonnegative_int(getattr(args, field_name), field_name) != expected:
            raise ValueError(f"stair eval {field_name} mismatch")
    if args.eval_protocol_version != config.eval_protocol_version:
        raise ValueError("stair eval protocol version mismatch")

    subject = ExperimentSubject(
        kind=args.subject_kind,
        subject_id=_text(args.subject_id, "subject_id"),
    )
    subject_manifest_hash = require_sha256_digest(
        args.subject_manifest_hash,
        "stair eval subject manifest hash",
    )
    identity = JobIdentity(
        job_id=_text(args.job_id, "job_id"),
        task_id=STAIR_TASK_ID,
        subject=subject,
        subject_manifest_hash=subject_manifest_hash,
    )
    program_source_checksum = require_sha256_checksum(
        args.program_source_checksum,
        "stair eval program source checksum",
    )
    training_binding_hash = require_sha256_digest(
        args.training_binding_hash,
        "stair eval training binding hash",
    )
    capability_profile_hash = require_sha256_digest(
        args.capability_profile_hash,
        "stair eval capability profile hash",
    )
    if capability_profile_hash != config.candidate_capability_profile.profile_hash:
        raise ValueError("stair eval capability profile mismatch")

    eval_purpose = getattr(
        args,
        "stair_eval_purpose",
        STAIR_EVAL_PURPOSE_FORMAL,
    )
    seed_namespace_by_purpose = {
        STAIR_EVAL_PURPOSE_FORMAL: STAIR_FORMAL_EVAL_NAMESPACE,
        STAIR_EVAL_PURPOSE_FINAL_CONFIRMATION: (
            STAIR_FINAL_CONFIRMATION_EVAL_NAMESPACE
        ),
    }
    try:
        seed_namespace = seed_namespace_by_purpose[eval_purpose]
    except (KeyError, TypeError) as exc:
        raise ValueError("stair eval purpose is invalid") from exc

    expected_binding = EvaluationBindingV1(
        eval_protocol_hash=config.evaluation_protocol().protocol_hash,
        acceptance_protocol_hash=config.acceptance_protocol.protocol_hash,
        task_protocol_hash=config.task_protocol_hash,
        seed_cohort_hash=config.seed_resolver().namespace_cohort_hash(
            seed_namespace
        ),
    )
    declared_binding = EvaluationBindingV1(
        eval_protocol_hash=config.evaluation_protocol().protocol_hash,
        acceptance_protocol_hash=require_sha256_digest(
            args.acceptance_protocol_hash,
            "stair eval acceptance protocol hash",
        ),
        task_protocol_hash=require_sha256_digest(
            args.task_protocol_hash,
            "stair eval task protocol hash",
        ),
        seed_cohort_hash=require_sha256_digest(
            args.seed_cohort_hash,
            "stair eval seed cohort hash",
        ),
    )
    declared_binding_hash = require_sha256_digest(
        args.evaluation_binding_hash,
        "stair evaluation binding hash",
    )
    if declared_binding.binding_hash != declared_binding_hash:
        raise ValueError("stair eval declared binding hash mismatch")
    if declared_binding != expected_binding:
        raise ValueError("stair eval binding differs from task config")

    declared_checkpoint = Path(_text(args.checkpoint, "checkpoint"))
    if "://" in str(declared_checkpoint):
        raise ValueError("stair eval checkpoint must be a local non-symlink path")
    checkpoint_path = resolve_existing_file(
        (
            declared_checkpoint
            if declared_checkpoint.is_absolute()
            else project_root / declared_checkpoint
        ),
        field_name="stair eval checkpoint",
    )
    checkpoint_hash = require_sha256_checksum(
        args.checkpoint_sha256,
        "stair eval checkpoint hash",
    )
    return StairEvalEntrypointContext(
        config=config,
        identity=identity,
        evaluation_binding=declared_binding,
        evaluation_subject=subject,
        checkpoint_path=checkpoint_path,
        checkpoint_hash=checkpoint_hash,
        subject_manifest_hash=subject_manifest_hash,
        program_source_checksum=program_source_checksum,
        training_binding_hash=training_binding_hash,
        capability_profile_hash=capability_profile_hash,
        seed_namespace=seed_namespace,
    )


def _canonical_config_path(
    value: Any,
    *,
    project_root: Path,
    resolved_cell_manifest: ResolvedCellManifestV1 | None,
) -> Path:
    if not isinstance(project_root, Path):
        raise TypeError("project_root must be a Path")
    expected = STAIR_UNIFIED_PROTOCOL_RELATIVE_PATH
    if resolved_cell_manifest is not None:
        if not isinstance(resolved_cell_manifest, ResolvedCellManifestV1):
            raise TypeError("resolved Stair cell manifest is invalid")
        expected = stair_robot_row(
            resolved_cell_manifest.cell_variant_id
        ).protocol_relative_path
    if value != expected:
        raise ValueError("stair entrypoint config path is not canonical")
    return resolve_existing_file(
        project_root / expected,
        field_name="stair entrypoint config",
    )


def _resolved_stair_config(
    config_path: Path,
    *,
    resolved_cell_manifest: ResolvedCellManifestV1 | None,
) -> StairClimbingTaskConfig:
    if resolved_cell_manifest is None:
        return StairClimbingTaskConfig.defaults(config_path)
    if not isinstance(resolved_cell_manifest, ResolvedCellManifestV1):
        raise TypeError("resolved Stair cell manifest is invalid")
    return StairClimbingTaskConfig.defaults(
        config_path,
        cell_variant_id=resolved_cell_manifest.cell_variant_id,
        resolved_cell_task_protocol_hash=(
            resolved_cell_manifest.resolved_cell_task_protocol_hash
        ),
    )


def _text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{field_name} must be a non-empty canonical string")
    return value


def _absolute_artifact_path(value: Any, field_name: str) -> Path:
    path = Path(_text(value, field_name))
    if not path.is_absolute() or path.is_symlink():
        raise ValueError(f"stair train {field_name} path must be absolute")
    return path


def _require_absent(value: Any, field_name: str) -> None:
    if value is not None:
        raise ValueError(f"stair train {field_name} must be absent")


def _positive_int(value: Any, field_name: str) -> int:
    parsed = _nonnegative_int(value, field_name)
    if parsed <= 0:
        raise ValueError(f"{field_name} must be positive")
    return parsed


def _nonnegative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


__all__ = [
    "StairEvalEntrypointContext",
    "STAIR_EVAL_PURPOSE_FINAL_CONFIRMATION",
    "STAIR_EVAL_PURPOSE_FORMAL",
    "StairTrainEntrypointContext",
    "validate_stair_eval_invocation",
    "validate_stair_train_invocation",
]
