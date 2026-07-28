"""Pure preflight contracts for production Stair subprocess entrypoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from true_rl_agent.backends.rsl_rl_training_config import RslRlAgentConfiguration
from true_rl_agent.candidates.artifact import (
    SealedCandidateArtifactRef,
    load_sealed_candidate_artifact,
)
from true_rl_agent.candidates.schema import CandidateChangeSet
from true_rl_agent.evaluation.binding import (
    EvaluationBindingV1,
    require_sha256_checksum,
    require_sha256_digest,
)
from true_rl_agent.evaluation.subject import ExperimentSubject
from true_rl_agent.runner.budget import build_budget_accounting
from true_rl_agent.runner.job_identity import JobIdentity
from true_rl_agent.tasks.stair_climbing.config import (
    STAIR_PROTOCOL_V2_RELATIVE_PATH,
    STAIR_TASK_ID,
    StairClimbingTaskConfig,
)
from true_rl_agent.training_binding import TrainingBinding


@dataclass(frozen=True)
class StairTrainEntrypointContext:
    config: StairClimbingTaskConfig
    identity: JobIdentity
    candidate_artifact: SealedCandidateArtifactRef
    training_binding: TrainingBinding
    agent_configuration: RslRlAgentConfiguration
    budget_accounting: dict[str, int | str]


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


def validate_stair_train_invocation(
    args: Any,
    *,
    project_root: Path,
) -> StairTrainEntrypointContext:
    """Reject any train argv that differs from the task-owned config snapshot."""

    config_path = _canonical_config_path(args.config, project_root=project_root)
    defaults = StairClimbingTaskConfig.defaults(config_path)
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
    )
    if args.task_id != STAIR_TASK_ID:
        raise ValueError("stair train task_id mismatch")
    if args.subject_kind != "candidate":
        raise ValueError("stair Phase B train subject must be candidate")
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

    identity = JobIdentity(
        job_id=_text(args.job_id, "job_id"),
        task_id=STAIR_TASK_ID,
        subject=ExperimentSubject.candidate(_text(args.subject_id, "subject_id")),
        subject_manifest_hash=require_sha256_digest(
            args.subject_manifest_hash,
            "stair train subject manifest hash",
        ),
    )
    source_checksum = require_sha256_checksum(
        args.program_source_checksum,
        "stair train program source checksum",
    )
    artifact_path = Path(_text(args.candidate, "candidate"))
    if not artifact_path.is_absolute() or artifact_path.is_symlink():
        raise ValueError("stair train candidate artifact path must be absolute")
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
        binding_hash=require_sha256_digest(
            args.training_binding_hash,
            "stair training binding hash",
        ),
    )
    expected_binding = TrainingBinding(
        initialization_mode="random",
        initialization_checkpoint_hash=None,
        policy_architecture_hash=expected_agent.policy_architecture_hash,
        agent_configuration_hash=expected_agent.agent_configuration_hash,
        initialization_seed_schedule_hash=config.initialization_seed_schedule_hash(),
        optimizer_training_profile_hash=(
            expected_agent.optimizer_training_profile_hash
        ),
        budget_steps=config.budget_steps,
        private_training_seed_resolver_hash=(
            config.private_training_seed_resolver_hash()
        ),
        capability_profile_hash=config.protocol_v2.capability_profile.profile_hash,
    )
    if training_binding != expected_binding:
        raise ValueError("stair train binding differs from task config")

    return StairTrainEntrypointContext(
        config=config,
        identity=identity,
        candidate_artifact=candidate_artifact,
        training_binding=training_binding,
        agent_configuration=agent_configuration,
        budget_accounting=build_budget_accounting(
            budget_steps=config.budget_steps,
            num_envs=config.train_num_envs,
            num_steps_per_env=config.train_steps_per_iteration,
            max_iterations=config.train_max_iterations,
        ),
    )


def validate_stair_eval_invocation(
    args: Any,
    *,
    project_root: Path,
) -> StairEvalEntrypointContext:
    """Bind an eval invocation to one immutable Stair comparison domain."""

    config_path = _canonical_config_path(args.config, project_root=project_root)
    config = StairClimbingTaskConfig.defaults(config_path)
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
    if capability_profile_hash != config.protocol_v2.capability_profile.profile_hash:
        raise ValueError("stair eval capability profile mismatch")

    expected_binding = EvaluationBindingV1(
        eval_protocol_hash=config.evaluation_protocol().protocol_hash,
        acceptance_protocol_hash=config.protocol_v1.acceptance_protocol.protocol_hash,
        task_protocol_hash=config.protocol_v2.protocol_hash,
        seed_cohort_hash=config.seed_resolver().cohort_hash,
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

    checkpoint_path = Path(_text(args.checkpoint, "checkpoint"))
    if "://" in str(checkpoint_path) or checkpoint_path.is_symlink():
        raise ValueError("stair eval checkpoint must be a local non-symlink path")
    if not checkpoint_path.is_absolute():
        checkpoint_path = project_root / checkpoint_path
    checkpoint_path = checkpoint_path.resolve(strict=True)
    if not checkpoint_path.is_file():
        raise ValueError("stair eval checkpoint must be a regular file")
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
    )


def _canonical_config_path(value: Any, *, project_root: Path) -> Path:
    if not isinstance(project_root, Path):
        raise TypeError("project_root must be a Path")
    if value != STAIR_PROTOCOL_V2_RELATIVE_PATH:
        raise ValueError("stair entrypoint config path is not canonical")
    expected = (project_root / STAIR_PROTOCOL_V2_RELATIVE_PATH).resolve(strict=True)
    if expected.is_symlink() or not expected.is_file():
        raise ValueError("stair entrypoint config must be a regular file")
    return expected


def _text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{field_name} must be a non-empty canonical string")
    return value


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
    "StairTrainEntrypointContext",
    "validate_stair_eval_invocation",
    "validate_stair_train_invocation",
]
