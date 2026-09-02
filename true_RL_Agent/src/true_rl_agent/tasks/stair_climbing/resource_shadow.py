"""Task-owned fixture and fixed workload for the P6-A resource shadow pilot."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from true_rl_agent.backends.rsl_rl_training_config import RslRlAgentConfiguration
from true_rl_agent.evaluation.binding import (
    canonical_json_hash,
    require_sha256_checksum,
    require_sha256_digest,
)
from true_rl_agent.fs_contract import read_bounded_regular_file
from true_rl_agent.json_contract import canonical_json_bytes, strict_json_object_loads
from true_rl_agent.runner.budget import build_budget_accounting
from true_rl_agent.runner.checkpoint import file_sha256
from true_rl_agent.tasks.stair_climbing.config import (
    STAIR_TASK_ID,
    StairClimbingTaskConfig,
)
from true_rl_agent.tasks.stair_climbing.gpu_process_binding import (
    stair_single_gpu_kit_args,
)
from true_rl_agent.tasks.stair_climbing.isaaclab_execution import (
    StairTrainingExecutionResult,
    VerifiedStairExecutionSourceManifest,
    _StairTrainingWorkload,
    _execute_stair_training,
)
from true_rl_agent.tasks.stair_climbing.reference_program import (
    StairOfficialReferenceProgram,
    resolve_official_stair_reference_program,
)
from true_rl_agent.tasks.stair_climbing.reward_program import (
    QualifiedStairRewardProgram,
)

STAIR_RESOURCE_SHADOW_WORKLOAD_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.resource_shadow_workload.v2"
)
STAIR_RESOURCE_SHADOW_FIXTURE_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.resource_shadow_fixture.v1"
)
STAIR_RESOURCE_SHADOW_PRESET_ID = "resource_shadow_v1"
STAIR_RESOURCE_SHADOW_FIXTURE_ID = "official_reference_complete_reward_v1"
STAIR_RESOURCE_SHADOW_ACTIVE_MARKER_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.resource_shadow_active.v2"
)
STAIR_RESOURCE_SHADOW_SUMMARY_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.resource_shadow_summary.v2"
)
_MAX_ACTIVE_MARKER_BYTES = 16 * 1024
_MAX_SUMMARY_BYTES = 4 * 1024 * 1024


@dataclass(frozen=True)
class StairResourceShadowWorkload:
    """One immutable workload; pilot callers cannot tune it at runtime."""

    preset_id: str = STAIR_RESOURCE_SHADOW_PRESET_ID
    num_envs: int = 4096
    num_steps_per_env: int = 24
    max_iterations: int = 64
    budget_steps: int = 6_291_456
    checkpoint_save_interval: int = 50
    process_execution_device: str = "cuda:0"
    cuda_visibility_binding: str = "single_physical_gpu_uuid"
    renderer_multi_gpu: bool = False
    renderer_multi_gpu_auto_enable: bool = False
    renderer_multi_gpu_max_gpu_count: int = 1

    def __post_init__(self) -> None:
        if (
            self.preset_id,
            self.num_envs,
            self.num_steps_per_env,
            self.max_iterations,
            self.budget_steps,
            self.checkpoint_save_interval,
            self.process_execution_device,
            self.cuda_visibility_binding,
            self.renderer_multi_gpu,
            self.renderer_multi_gpu_auto_enable,
            self.renderer_multi_gpu_max_gpu_count,
        ) != (
            STAIR_RESOURCE_SHADOW_PRESET_ID,
            4096,
            24,
            64,
            6_291_456,
            50,
            "cuda:0",
            "single_physical_gpu_uuid",
            False,
            False,
            1,
        ):
            raise ValueError("Stair resource-shadow workload is immutable")
        build_budget_accounting(
            budget_steps=self.budget_steps,
            num_envs=self.num_envs,
            num_steps_per_env=self.num_steps_per_env,
            max_iterations=self.max_iterations,
        )

    def to_contract_dict(self) -> dict[str, Any]:
        return {
            "schema_version": STAIR_RESOURCE_SHADOW_WORKLOAD_SCHEMA_VERSION,
            "preset_id": self.preset_id,
            "num_envs": self.num_envs,
            "num_steps_per_env": self.num_steps_per_env,
            "max_iterations": self.max_iterations,
            "budget_steps": self.budget_steps,
            "checkpoint_save_interval": self.checkpoint_save_interval,
            "process_execution_device": self.process_execution_device,
            "cuda_visibility_binding": self.cuda_visibility_binding,
            "renderer_multi_gpu": self.renderer_multi_gpu,
            "renderer_multi_gpu_auto_enable": (
                self.renderer_multi_gpu_auto_enable
            ),
            "renderer_multi_gpu_max_gpu_count": (
                self.renderer_multi_gpu_max_gpu_count
            ),
            "acceptance_authority": False,
            "ledger_authority": False,
        }

    @property
    def workload_hash(self) -> str:
        return canonical_json_hash(self.to_contract_dict())


STAIR_RESOURCE_SHADOW_WORKLOAD = StairResourceShadowWorkload()


@dataclass(frozen=True)
class StairResourceShadowFixture:
    """Qualified official complete-reward program plus its frozen identity."""

    official_program: StairOfficialReferenceProgram
    contract: Mapping[str, Any]
    fixture_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.official_program, StairOfficialReferenceProgram):
            raise TypeError("Stair resource-shadow official program is invalid")
        normalized = dict(self.contract)
        if canonical_json_hash(normalized) != self.fixture_hash:
            raise ValueError("Stair resource-shadow fixture hash mismatch")
        object.__setattr__(self, "contract", MappingProxyType(normalized))

    @property
    def reward_program(self) -> QualifiedStairRewardProgram:
        return self.official_program.reward_program

    def to_contract_dict(self) -> dict[str, Any]:
        return dict(self.contract)


def stair_resource_shadow_workload_contract() -> dict[str, Any]:
    return STAIR_RESOURCE_SHADOW_WORKLOAD.to_contract_dict()


def stair_resource_shadow_workload_hash() -> str:
    return STAIR_RESOURCE_SHADOW_WORKLOAD.workload_hash


def stair_resource_shadow_kit_args(*, renderer_active_gpu_ordinal: int) -> str:
    """Return the task-owned Kit settings for one UUID-isolated process."""

    renderer_active_gpu_ordinal = _renderer_gpu_ordinal(
        renderer_active_gpu_ordinal,
        "Stair resource-shadow renderer GPU ordinal",
    )
    return stair_single_gpu_kit_args(
        renderer_active_gpu_ordinal=renderer_active_gpu_ordinal
    )


def resolve_stair_resource_shadow_fixture(
    config: StairClimbingTaskConfig,
    *,
    expected_initial_policy_state_hash: str,
    project_root: Path | None = None,
) -> StairResourceShadowFixture:
    """Resolve and qualify the tracked official fixture from current repo bytes."""

    if not isinstance(config, StairClimbingTaskConfig):
        raise TypeError("Stair resource-shadow fixture requires task config")
    expected_initial_policy_state_hash = require_sha256_digest(
        expected_initial_policy_state_hash,
        "Stair resource-shadow fixture initial policy state hash",
    )
    official = resolve_official_stair_reference_program(
        config,
        project_root=project_root,
    )
    manifest = official.artifact.manifest
    qualified = official.reward_program
    contract = {
        "schema_version": STAIR_RESOURCE_SHADOW_FIXTURE_SCHEMA_VERSION,
        "fixture_id": STAIR_RESOURCE_SHADOW_FIXTURE_ID,
        "task_id": STAIR_TASK_ID,
        "task_protocol_hash": config.task_protocol_hash,
        "subject": qualified.subject.to_public_dict(),
        "subject_manifest_hash": qualified.subject_manifest_hash,
        "program_source_checksum": qualified.program_source_checksum,
        "program_source_size_bytes": qualified.program_source_size_bytes,
        "reference_verification_hash": official.artifact.verification_hash,
        "reward_program_qualification_hash": qualified.qualification_hash,
        "capability_profile_hash": manifest.capability_profile_hash,
        "agent_configuration_hash": (
            config.training_agent_configuration().agent_configuration_hash
        ),
        "expected_initial_policy_state_hash": expected_initial_policy_state_hash,
        "acceptance_authority": False,
        "ledger_authority": False,
    }
    return StairResourceShadowFixture(
        official_program=official,
        contract=contract,
        fixture_hash=canonical_json_hash(contract),
    )


def load_stair_resource_shadow_active_marker(
    path: Path,
    *,
    output_dir: Path,
    expected_plan_hash: str,
    expected_job_id: str,
    expected_case_id: str,
    expected_lane_id: str,
    expected_physical_execution_device: str,
    expected_process_execution_device: str,
    expected_physical_gpu_uuid: str,
    expected_renderer_active_gpu_ordinal: int,
) -> dict[str, Any]:
    """Load the small create-once active marker and bind it to one planned job."""

    marker_path = _controlled_existing_file(
        path,
        output_dir=output_dir,
        label="Stair resource-shadow active marker",
    )
    marker = _load_canonical_json(
        marker_path,
        max_bytes=_MAX_ACTIVE_MARKER_BYTES,
        label="Stair resource-shadow active marker",
    )
    expected_keys = {
        "schema_version",
        "plan_hash",
        "job_id",
        "case_id",
        "lane_id",
        "physical_execution_device",
        "process_execution_device",
        "physical_gpu_uuid",
        "renderer_active_gpu_ordinal",
        "process_id",
        "process_group_id",
        "active_start_monotonic_ns",
        "acceptance_authority",
        "ledger_authority",
    }
    if set(marker) != expected_keys:
        raise ValueError("Stair resource-shadow active marker fields mismatch")
    if (
        marker["schema_version"]
        != STAIR_RESOURCE_SHADOW_ACTIVE_MARKER_SCHEMA_VERSION
    ):
        raise ValueError("Stair resource-shadow active marker schema mismatch")
    _renderer_gpu_ordinal(
        marker["renderer_active_gpu_ordinal"],
        "Stair resource-shadow active renderer GPU ordinal",
    )
    expected_renderer_active_gpu_ordinal = _renderer_gpu_ordinal(
        expected_renderer_active_gpu_ordinal,
        "Stair resource-shadow expected renderer GPU ordinal",
    )
    expected = {
        "plan_hash": require_sha256_digest(
            expected_plan_hash,
            "Stair resource-shadow active plan hash",
        ),
        "job_id": expected_job_id,
        "case_id": expected_case_id,
        "lane_id": expected_lane_id,
        "physical_execution_device": expected_physical_execution_device,
        "process_execution_device": expected_process_execution_device,
        "physical_gpu_uuid": expected_physical_gpu_uuid,
        "renderer_active_gpu_ordinal": expected_renderer_active_gpu_ordinal,
    }
    for field_name, expected_value in expected.items():
        if marker[field_name] != expected_value:
            raise ValueError(
                f"Stair resource-shadow active {field_name} mismatch"
            )
    for field_name in (
        "process_id",
        "process_group_id",
        "active_start_monotonic_ns",
    ):
        _positive_int(marker[field_name], f"Stair resource-shadow {field_name}")
    _require_no_authority(marker, "Stair resource-shadow active marker")
    return marker


def load_stair_resource_shadow_summary(
    path: Path,
    *,
    output_dir: Path,
    expected_checkpoint_path: Path,
    expected_plan_hash: str,
    expected_session_genesis_hash: str,
    expected_execution_environment_manifest_hash: str,
    expected_fixture_hash: str,
    expected_workload_hash: str,
    expected_resource_plan_hash: str,
    expected_initial_policy_state_hash: str,
    expected_job_id: str,
    expected_case_id: str,
    expected_lane_id: str,
    expected_physical_execution_device: str,
    expected_process_execution_device: str,
    expected_physical_gpu_uuid: str,
    expected_renderer_active_gpu_ordinal: int,
    expected_status: str | None = None,
) -> dict[str, Any]:
    """Validate only the resource-pilot bindings and successful output bytes."""

    summary_path = _controlled_existing_file(
        path,
        output_dir=output_dir,
        label="Stair resource-shadow summary",
    )
    summary = _load_canonical_json(
        summary_path,
        max_bytes=_MAX_SUMMARY_BYTES,
        label="Stair resource-shadow summary",
    )
    required_keys = {
        "schema_version",
        "status",
        "training_executed",
        "plan_hash",
        "session_genesis_hash",
        "execution_environment_manifest_hash",
        "fixture_hash",
        "workload_hash",
        "resource_plan_hash",
        "job_id",
        "case_id",
        "lane_id",
        "physical_execution_device",
        "process_execution_device",
        "physical_gpu_uuid",
        "renderer_active_gpu_ordinal",
        "process_id",
        "process_group_id",
        "active_start_monotonic_ns",
        "active_end_monotonic_ns",
        "active_duration_ns",
        "expected_initial_policy_state_hash",
        "initial_policy_state_hash",
        "initial_policy_state_match",
        "workload",
        "fixture",
        "environment_source_preflight",
        "execution_environment_verification",
        "created_at_utc",
        "acceptance_authority",
        "ledger_authority",
    }
    allowed_result_keys = {
        "runtime_activation_summary",
        "activation_preflight",
        "errors",
        "candidate_failure",
        "checkpoint",
        "training",
    }
    if not required_keys.issubset(summary) or not set(summary).issubset(
        required_keys | allowed_result_keys
    ):
        raise ValueError("Stair resource-shadow summary fields mismatch")
    if summary["schema_version"] != STAIR_RESOURCE_SHADOW_SUMMARY_SCHEMA_VERSION:
        raise ValueError("Stair resource-shadow summary schema mismatch")
    _renderer_gpu_ordinal(
        summary["renderer_active_gpu_ordinal"],
        "Stair resource-shadow summary renderer GPU ordinal",
    )
    expected_renderer_active_gpu_ordinal = _renderer_gpu_ordinal(
        expected_renderer_active_gpu_ordinal,
        "Stair resource-shadow expected renderer GPU ordinal",
    )
    status = summary["status"]
    if status not in {"pass", "fail"}:
        raise ValueError("Stair resource-shadow summary status is invalid")
    if expected_status is not None and status != expected_status:
        raise ValueError("Stair resource-shadow summary status mismatch")
    if not isinstance(summary["training_executed"], bool):
        raise TypeError("Stair resource-shadow training_executed must be boolean")
    expected_bindings = {
        "plan_hash": require_sha256_digest(
            expected_plan_hash,
            "Stair resource-shadow summary plan hash",
        ),
        "session_genesis_hash": require_sha256_digest(
            expected_session_genesis_hash,
            "Stair resource-shadow summary session Genesis hash",
        ),
        "execution_environment_manifest_hash": require_sha256_digest(
            expected_execution_environment_manifest_hash,
            "Stair resource-shadow summary environment hash",
        ),
        "fixture_hash": require_sha256_digest(
            expected_fixture_hash,
            "Stair resource-shadow summary fixture hash",
        ),
        "workload_hash": require_sha256_digest(
            expected_workload_hash,
            "Stair resource-shadow summary workload hash",
        ),
        "resource_plan_hash": require_sha256_digest(
            expected_resource_plan_hash,
            "Stair resource-shadow summary resource plan hash",
        ),
        "expected_initial_policy_state_hash": require_sha256_digest(
            expected_initial_policy_state_hash,
            "Stair resource-shadow summary initial policy state hash",
        ),
        "job_id": expected_job_id,
        "case_id": expected_case_id,
        "lane_id": expected_lane_id,
        "physical_execution_device": expected_physical_execution_device,
        "process_execution_device": expected_process_execution_device,
        "physical_gpu_uuid": expected_physical_gpu_uuid,
        "renderer_active_gpu_ordinal": expected_renderer_active_gpu_ordinal,
    }
    for field_name, expected_value in expected_bindings.items():
        if summary[field_name] != expected_value:
            raise ValueError(
                f"Stair resource-shadow summary {field_name} mismatch"
            )
    for field_name in (
        "process_id",
        "process_group_id",
        "active_start_monotonic_ns",
        "active_end_monotonic_ns",
        "active_duration_ns",
    ):
        _positive_int(summary[field_name], f"Stair resource-shadow {field_name}")
    if summary["active_end_monotonic_ns"] <= summary["active_start_monotonic_ns"]:
        raise ValueError("Stair resource-shadow active interval is not positive")
    if summary["active_duration_ns"] != (
        summary["active_end_monotonic_ns"]
        - summary["active_start_monotonic_ns"]
    ):
        raise ValueError("Stair resource-shadow active duration mismatch")
    if summary["workload"] != stair_resource_shadow_workload_contract():
        raise ValueError("Stair resource-shadow summary workload contract mismatch")
    fixture = _mapping(summary["fixture"], "Stair resource-shadow fixture")
    if canonical_json_hash(fixture) != expected_bindings["fixture_hash"]:
        raise ValueError("Stair resource-shadow summary fixture contract mismatch")
    if fixture.get("expected_initial_policy_state_hash") != (
        expected_bindings["expected_initial_policy_state_hash"]
    ):
        raise ValueError("Stair resource-shadow fixture initial policy mismatch")
    environment_verification = _mapping(
        summary["execution_environment_verification"],
        "Stair resource-shadow environment verification",
    )
    if (
        environment_verification.get("status") != "pass"
        or environment_verification.get("expected_manifest_hash")
        != expected_bindings["execution_environment_manifest_hash"]
        or environment_verification.get("actual_manifest_hash")
        != expected_bindings["execution_environment_manifest_hash"]
    ):
        raise ValueError("Stair resource-shadow environment verification mismatch")
    _require_no_authority(summary, "Stair resource-shadow summary")

    if status == "pass":
        if summary["training_executed"] is not True:
            raise ValueError("passing Stair resource-shadow summary did not train")
        actual_initial = require_sha256_digest(
            summary["initial_policy_state_hash"],
            "Stair resource-shadow actual initial policy state hash",
        )
        if (
            actual_initial != expected_bindings["expected_initial_policy_state_hash"]
            or summary["initial_policy_state_match"] is not True
        ):
            raise ValueError("Stair resource-shadow initial policy state mismatch")
        training = _mapping(summary.get("training"), "Stair resource-shadow training")
        if (
            training.get("initial_policy_state_hash") != actual_initial
            or training.get("expected_initial_policy_state_hash")
            != expected_bindings["expected_initial_policy_state_hash"]
            or training.get("initial_policy_state_match") is not True
            or training.get("workload_kind") != STAIR_RESOURCE_SHADOW_PRESET_ID
            or training.get("device") != expected_process_execution_device
        ):
            raise ValueError("Stair resource-shadow training binding mismatch")
        _validate_shadow_checkpoint(
            summary.get("checkpoint"),
            output_dir=output_dir,
            expected_checkpoint_path=expected_checkpoint_path,
        )
    elif "checkpoint" in summary:
        raise ValueError("failed Stair resource-shadow summary cannot own checkpoint")
    return summary


def _validate_shadow_checkpoint(
    value: Any,
    *,
    output_dir: Path,
    expected_checkpoint_path: Path,
) -> None:
    checkpoint = _mapping(value, "Stair resource-shadow checkpoint")
    declared_path = Path(checkpoint.get("uri", ""))
    if declared_path.is_symlink() or expected_checkpoint_path.is_symlink():
        raise ValueError("Stair resource-shadow checkpoint cannot be a symlink")
    declared = _controlled_path(
        declared_path,
        output_dir=output_dir,
        label="Stair resource-shadow checkpoint",
    )
    expected = _controlled_path(
        expected_checkpoint_path,
        output_dir=output_dir,
        label="Stair resource-shadow expected checkpoint",
    )
    if declared != expected:
        raise ValueError("Stair resource-shadow checkpoint path mismatch")
    if expected.is_symlink() or not expected.is_file():
        raise ValueError("Stair resource-shadow checkpoint is not a regular file")
    declared_checksum = require_sha256_checksum(
        checkpoint.get("checksum"),
        "Stair resource-shadow checkpoint checksum",
    )
    actual_checksum = file_sha256(expected)
    if actual_checksum is None or actual_checksum != declared_checksum:
        raise ValueError("Stair resource-shadow checkpoint checksum mismatch")
    if checkpoint.get("role") != "train_output":
        raise ValueError("Stair resource-shadow checkpoint role mismatch")


def _load_canonical_json(path: Path, *, max_bytes: int, label: str) -> dict[str, Any]:
    payload = read_bounded_regular_file(path, max_bytes=max_bytes, label=label)
    parsed = strict_json_object_loads(payload, context=label)
    if payload != canonical_json_bytes(parsed, trailing_newline=True):
        raise ValueError(f"{label} is not canonical JSON")
    return parsed


def _controlled_existing_file(path: Path, *, output_dir: Path, label: str) -> Path:
    if isinstance(path, Path) and path.is_symlink():
        raise ValueError(f"{label} cannot be a symlink")
    resolved = _controlled_path(path, output_dir=output_dir, label=label)
    if resolved.is_symlink() or not resolved.is_file():
        raise ValueError(f"{label} must be a regular file")
    return resolved


def _controlled_path(path: Path, *, output_dir: Path, label: str) -> Path:
    if not isinstance(path, Path) or not path.is_absolute():
        raise ValueError(f"{label} path must be absolute")
    if not isinstance(output_dir, Path) or not output_dir.is_absolute():
        raise ValueError(f"{label} output_dir must be absolute")
    root = output_dir.resolve()
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escaped its job output") from exc
    if not relative.parts:
        raise ValueError(f"{label} cannot equal its job output")
    return resolved


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return dict(value)


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _renderer_gpu_ordinal(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in {0, 1}:
        raise ValueError(f"{label} must be 0 or 1")
    return value


def _require_no_authority(value: Mapping[str, Any], label: str) -> None:
    if (
        value.get("acceptance_authority") is not False
        or value.get("ledger_authority") is not False
    ):
        raise ValueError(f"{label} cannot carry acceptance or ledger authority")


def execute_stair_resource_shadow_training(
    *,
    config: StairClimbingTaskConfig,
    source_manifest_preflight: VerifiedStairExecutionSourceManifest,
    agent_configuration: RslRlAgentConfiguration,
    reward_program: QualifiedStairRewardProgram,
    device: str,
    run_dir: Path,
    checkpoint_output: Path,
    run_name: str,
    expected_initial_policy_state_hash: str,
) -> StairTrainingExecutionResult:
    """Use the frozen shadow preset with the unchanged authoritative train body.

    P6-A deliberately depends on these two package-private execution primitives so
    the existing execution-environment manifest hot path remains byte-identical.
    The internal ``production`` kind selects the full 4096-env path; the returned
    task-owned evidence explicitly records the non-authoritative shadow preset.
    """

    if not isinstance(config, StairClimbingTaskConfig):
        raise TypeError("Stair resource-shadow training requires task config")
    if not isinstance(
        source_manifest_preflight,
        VerifiedStairExecutionSourceManifest,
    ):
        raise TypeError("Stair resource-shadow source preflight is invalid")
    source_manifest_preflight.validate_for(config)
    expected_initial_policy_state_hash = require_sha256_digest(
        expected_initial_policy_state_hash,
        "Stair resource-shadow expected initial policy state hash",
    )
    workload = STAIR_RESOURCE_SHADOW_WORKLOAD
    if config.train_num_envs != workload.num_envs:
        raise ValueError(
            "Stair resource-shadow env count must match the production shape"
        )
    if config.train_steps_per_iteration != workload.num_steps_per_env:
        raise ValueError(
            "Stair resource-shadow rollout length must match the production shape"
        )
    result = _execute_stair_training(
        config=config,
        agent_configuration=agent_configuration,
        reward_program=reward_program,
        device=device,
        run_dir=run_dir,
        checkpoint_output=checkpoint_output,
        run_name=run_name,
        expected_initial_policy_state_hash=expected_initial_policy_state_hash,
        resolved_cell_manifest=None,
        workload=_StairTrainingWorkload(
            # Keep the authoritative training module unchanged while selecting
            # its existing full-size production-shape branch.
            kind="production",
            num_envs=workload.num_envs,
            num_steps_per_env=workload.num_steps_per_env,
            max_iterations=workload.max_iterations,
            budget_steps=workload.budget_steps,
            checkpoint_save_interval=workload.checkpoint_save_interval,
        ),
    )
    training = dict(result.training) if result.training is not None else None
    if training is not None:
        if training.get("workload_kind") != "production":
            raise RuntimeError("Stair resource-shadow execution-body branch drifted")
        training["execution_body_workload_kind"] = "production"
        training["workload_kind"] = workload.preset_id
    return StairTrainingExecutionResult(
        status=result.status,
        training_executed=result.training_executed,
        checkpoint=result.checkpoint,
        runtime_activation_summary=result.runtime_activation_summary,
        activation_preflight=result.activation_preflight,
        training=training,
        candidate_failure=result.candidate_failure,
        errors=result.errors,
    )


__all__ = [
    "STAIR_RESOURCE_SHADOW_ACTIVE_MARKER_SCHEMA_VERSION",
    "STAIR_RESOURCE_SHADOW_FIXTURE_ID",
    "STAIR_RESOURCE_SHADOW_FIXTURE_SCHEMA_VERSION",
    "STAIR_RESOURCE_SHADOW_PRESET_ID",
    "STAIR_RESOURCE_SHADOW_SUMMARY_SCHEMA_VERSION",
    "STAIR_RESOURCE_SHADOW_WORKLOAD",
    "STAIR_RESOURCE_SHADOW_WORKLOAD_SCHEMA_VERSION",
    "StairResourceShadowFixture",
    "StairResourceShadowWorkload",
    "execute_stair_resource_shadow_training",
    "load_stair_resource_shadow_active_marker",
    "load_stair_resource_shadow_summary",
    "resolve_stair_resource_shadow_fixture",
    "stair_resource_shadow_kit_args",
    "stair_resource_shadow_workload_contract",
    "stair_resource_shadow_workload_hash",
]
