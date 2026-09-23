"""Single task-owned Isaac Lab/RSL-RL execution core for Stair."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping

from true_rl_agent.backends.base import (
    BACKEND_EXECUTION_ENVIRONMENT_VERIFICATION_SCHEMA_VERSION,
)
from true_rl_agent.backends.isaac_lab_execution_environment import (
    ISAAC_LAB_RSL_RL_BACKEND_ID,
)
from true_rl_agent.backends.rsl_rl_training_config import RslRlAgentConfiguration
from true_rl_agent.backends.rsl_rl_checkpoint import (
    INITIAL_MODEL_STATE_HASH_SCHEMA_VERSION,
    initial_model_state_hash,
    load_inference_checkpoint,
)
from true_rl_agent.composition.cell_manifest import ResolvedCellManifestV1
from true_rl_agent.evaluation.binding import (
    canonical_json_hash,
    require_sha256_checksum,
    require_sha256_digest,
)
from true_rl_agent.runner.budget import build_budget_accounting
from true_rl_agent.runner.candidate_failure import CandidateFailureEvidence
from true_rl_agent.runner.checkpoint import file_sha256
from true_rl_agent.runner.native_video import NativeVideoOptions
from true_rl_agent.robots.isaaclab_locomotion import (
    isaaclab_locomotion_runtime_spec,
)
from true_rl_agent.tasks.stair_climbing.candidate_failure import (
    StairCandidateProgramFailure,
)
from true_rl_agent.tasks.stair_climbing.config import StairClimbingTaskConfig
from true_rl_agent.tasks.stair_climbing.isaaclab_training import (
    apply_stair_rsl_rl_runner_contract,
    configure_stair_rsl_rl_agent,
    verify_stair_rsl_rl_observation_groups,
)
from true_rl_agent.tasks.stair_climbing.local_paths import resolve_existing_file
from true_rl_agent.tasks.stair_climbing.ordered_evaluation import (
    STAIR_ORDERED_FORMAL_COHORTS,
)
from true_rl_agent.tasks.stair_climbing.reward_program import (
    QualifiedStairRewardProgram,
    StairRewardOutputContractError,
    StairRewardProgramError,
)
from true_rl_agent.tasks.stair_climbing.row_catalog import stair_robot_row
from true_rl_agent.tasks.stair_climbing.runtime_profile import (
    STAIR_COMPLETE_REWARD_RUNTIME_STATUS,
    STAIR_EVAL_GYM_ID,
    STAIR_TRAIN_GYM_ID,
)
from true_rl_agent.tasks.stair_climbing.legacy_runtime_projection import (
    build_legacy_stair_runtime_projection,
)
from true_rl_agent.tasks.stair_climbing.runtime_projection import (
    StairRuntimeProjection,
    stair_runtime_projection_from_manifest,
)
from true_rl_agent.tasks.stair_climbing.seed_policy import (
    STAIR_CALIBRATION_EVAL_NAMESPACE,
    STAIR_FINAL_CONFIRMATION_EVAL_NAMESPACE,
)
from true_rl_agent.tasks.stair_climbing.termination_capture import (
    verify_stair_termination_manager_resolution,
    verify_stair_vnext_termination_manager_resolution,
)
from true_rl_agent.tasks.stair_climbing.training_distribution import (
    STAIR_EXECUTION_SOURCE_PREFLIGHT_SCHEMA_VERSION,
    build_stair_execution_source_preflight_summary,
    validate_stair_training_distribution_upstream_source_evidence,
    verify_installed_stair_training_distribution_upstream_sources,
)

STAIR_EVALUATION_EXECUTION_COST_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.eval_execution_cost.v1"
)


@dataclass(frozen=True, init=False)
class VerifiedStairExecutionSourceManifest:
    """Capability proving that the installed Stair environment sources were read."""

    task_protocol_hash: str
    source_manifest_hash: str
    _upstream_source_evidence_json: str

    __TOKEN = object()

    def __init__(
        self,
        *,
        task_protocol_hash: str,
        upstream_source_evidence: Mapping[str, Any],
        _token: object | None = None,
    ) -> None:
        if _token is not self.__TOKEN:
            raise TypeError(
                "verified Stair execution sources must come from the source preflight"
            )
        if not isinstance(upstream_source_evidence, Mapping):
            raise TypeError("Stair upstream source evidence must be an object")
        normalized = dict(upstream_source_evidence)
        manifest_hash = require_sha256_digest(
            normalized.get("source_manifest_hash"),
            "Stair execution source manifest hash",
        )
        object.__setattr__(
            self,
            "task_protocol_hash",
            require_sha256_digest(
                task_protocol_hash,
                "Stair execution task protocol hash",
            ),
        )
        object.__setattr__(self, "source_manifest_hash", manifest_hash)
        object.__setattr__(
            self,
            "_upstream_source_evidence_json",
            json.dumps(
                normalized,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ),
        )

    @classmethod
    def _from_verified_evidence(
        cls,
        *,
        task_protocol_hash: str,
        upstream_source_evidence: Mapping[str, Any],
    ) -> VerifiedStairExecutionSourceManifest:
        return cls(
            task_protocol_hash=task_protocol_hash,
            upstream_source_evidence=upstream_source_evidence,
            _token=cls.__TOKEN,
        )

    def validate_for(self, config: StairClimbingTaskConfig) -> None:
        if not isinstance(config, StairClimbingTaskConfig):
            raise TypeError("Stair source preflight requires StairClimbingTaskConfig")
        if self.task_protocol_hash != config.task_protocol_hash:
            raise ValueError("Stair execution source preflight protocol mismatch")
        if config.is_unified_protocol:
            evidence = _unified_execution_source_evidence(config)
            if (
                self.upstream_source_evidence() != evidence
                or evidence["source_manifest_hash"] != self.source_manifest_hash
            ):
                raise ValueError("Stair unified execution source preflight drifted")
            return
        assert config.protocol_v2 is not None
        contract = config.protocol_v2.runner_owned_training_distribution
        if contract is None:
            raise ValueError("Stair execution requires a training distribution contract")
        evidence = validate_stair_training_distribution_upstream_source_evidence(
            self.upstream_source_evidence(),
            contract=contract,
        )
        if evidence["source_manifest_hash"] != self.source_manifest_hash:
            raise ValueError("Stair execution source preflight hash mismatch")

    def upstream_source_evidence(self) -> dict[str, Any]:
        value = json.loads(self._upstream_source_evidence_json)
        if not isinstance(value, dict):
            raise RuntimeError("Stair execution source evidence is not an object")
        return value

    def to_summary_dict(self) -> dict[str, str]:
        return build_stair_execution_source_preflight_summary(
            task_protocol_hash=self.task_protocol_hash,
            source_manifest_hash=self.source_manifest_hash,
        )


def preflight_stair_execution_source_manifest(
    config: StairClimbingTaskConfig,
) -> VerifiedStairExecutionSourceManifest:
    """Read and verify the installed upstream bytes before launching Isaac."""

    if not isinstance(config, StairClimbingTaskConfig):
        raise TypeError("Stair source preflight requires StairClimbingTaskConfig")
    if config.is_unified_protocol:
        verified = VerifiedStairExecutionSourceManifest._from_verified_evidence(
            task_protocol_hash=config.task_protocol_hash,
            upstream_source_evidence=_unified_execution_source_evidence(config),
        )
        verified.validate_for(config)
        return verified
    assert config.protocol_v2 is not None
    contract = config.protocol_v2.runner_owned_training_distribution
    if contract is None:
        raise ValueError("Stair execution requires a training distribution contract")
    evidence = verify_installed_stair_training_distribution_upstream_sources(contract)
    verified = VerifiedStairExecutionSourceManifest._from_verified_evidence(
        task_protocol_hash=config.task_protocol_hash,
        upstream_source_evidence=evidence,
    )
    verified.validate_for(config)
    return verified


def _unified_execution_source_evidence(
    config: StairClimbingTaskConfig,
) -> dict[str, Any]:
    """Bind task semantics and the selected robot runtime upstream sources."""

    root = Path(__file__).resolve().parent
    names = (
        "isaaclab_reward_runtime.py",
        "isaaclab_runtime_env.py",
        "isaaclab_unified_terrain.py",
        "isaaclab_evaluation_terrain.py",
        "isaaclab_irregular_terrain.py",
        "isaaclab_rough_terrain.py",
        "ordered_evaluation.py",
        "rough_surface.py",
        "terrain_evidence.py",
        "transition_decoder.py",
        "unified_evaluation.py",
    )
    entries = []
    for name in names:
        data = (root / name).read_bytes()
        entries.append(
            {
                "repository_path": (
                    "src/true_rl_agent/tasks/stair_climbing/" + name
                ),
                "sha256": hashlib.sha256(data).hexdigest(),
                "size_bytes": len(data),
            }
        )
    row_binding = stair_robot_row(config.cell_variant_id).load_binding()
    robot_sources = isaaclab_locomotion_runtime_spec(
        row_binding.runtime_adapter_id
    ).installed_source_evidence()
    payload = {
        "schema_version": (
            "true_rl_agent.task.stair_climbing."
            "unified_execution_source_evidence.v3"
        ),
        "runtime_adapter_id": row_binding.runtime_adapter_id,
        "robot_runtime_source_evidence": robot_sources,
        "source_files": entries,
    }
    return {**payload, "source_manifest_hash": canonical_json_hash(payload)}


@dataclass(frozen=True)
class StairTrainingExecutionResult:
    status: str
    training_executed: bool
    checkpoint: Mapping[str, Any] | None
    runtime_activation_summary: Mapping[str, Any] | None
    activation_preflight: Mapping[str, Any] | None
    training: Mapping[str, Any] | None
    errors: tuple[Mapping[str, str], ...]
    candidate_failure: CandidateFailureEvidence | None = None

    def __post_init__(self) -> None:
        if self.status not in {"pass", "fail"}:
            raise ValueError("Stair training execution status is invalid")
        if not isinstance(self.training_executed, bool):
            raise TypeError("Stair training_executed must be boolean")
        if self.status == "pass":
            if self.checkpoint is None or self.training is None:
                raise ValueError("passing Stair training execution is incomplete")
            if self.errors or self.candidate_failure is not None:
                raise ValueError("passing Stair training execution contains failure data")
        elif (self.candidate_failure is None) == (not self.errors):
            raise ValueError(
                "failed Stair training execution requires exactly one failure owner"
            )
        if self.candidate_failure is not None and not isinstance(
            self.candidate_failure,
            CandidateFailureEvidence,
        ):
            raise TypeError("Stair candidate failure evidence is invalid")
        for field_name in (
            "checkpoint",
            "runtime_activation_summary",
            "activation_preflight",
            "training",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, MappingProxyType(dict(value)))
        object.__setattr__(
            self,
            "errors",
            tuple(MappingProxyType(dict(item)) for item in self.errors),
        )

    def to_summary_fields(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "status": self.status,
            "training_executed": self.training_executed,
            "runtime_activation_summary": (
                dict(self.runtime_activation_summary)
                if self.runtime_activation_summary is not None
                else None
            ),
            "activation_preflight": (
                dict(self.activation_preflight)
                if self.activation_preflight is not None
                else None
            ),
            "errors": [dict(item) for item in self.errors],
        }
        if self.candidate_failure is not None:
            result["candidate_failure"] = self.candidate_failure.to_public_dict()
        if self.checkpoint is not None:
            result["checkpoint"] = dict(self.checkpoint)
        if self.training is not None:
            result["training"] = dict(self.training)
        return result


@dataclass(frozen=True)
class StairEvaluationExecutionResult:
    summaries: Mapping[str, Mapping[str, Any]]
    per_seed_outcomes: Mapping[str, tuple[tuple[Any, ...], ...]]
    split_vector_env_step_calls: Mapping[str, int]
    split_wall_times_s: Mapping[str, float]
    checkpoint_hash: str
    seed_namespace: str
    seed_cohort_hash: str
    source_manifest_preflight: Mapping[str, str]
    backend_execution_environment_verification: Mapping[str, str]
    execution_device: str
    checkpoint_loading: Mapping[str, Any] | None = None
    constructed_learner: Mapping[str, Any] | None = None
    numeric_evidence: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.summaries, Mapping) or not self.summaries:
            raise ValueError("Stair evaluation summaries cannot be empty")
        if set(self.summaries) != set(self.per_seed_outcomes):
            raise ValueError("Stair evaluation outcome/split identity mismatch")
        for values in (self.split_vector_env_step_calls, self.split_wall_times_s):
            if set(self.summaries) != set(values):
                raise ValueError("Stair evaluation cost/split identity mismatch")
        require_sha256_checksum(
            self.checkpoint_hash,
            "Stair evaluation execution checkpoint hash",
        )
        if not isinstance(self.seed_namespace, str) or not self.seed_namespace:
            raise ValueError("Stair evaluation seed namespace cannot be empty")
        require_sha256_digest(
            self.seed_cohort_hash,
            "Stair evaluation seed cohort hash",
        )
        if not isinstance(self.execution_device, str) or not self.execution_device:
            raise ValueError("Stair evaluation execution device cannot be empty")
        object.__setattr__(
            self,
            "source_manifest_preflight",
            MappingProxyType(
                _validated_source_manifest_preflight_summary(
                    self.source_manifest_preflight
                )
            ),
        )
        object.__setattr__(
            self,
            "backend_execution_environment_verification",
            MappingProxyType(
                _validated_backend_execution_environment_verification(
                    self.backend_execution_environment_verification
                )
            ),
        )
        object.__setattr__(
            self,
            "summaries",
            _freeze_json_value(dict(self.summaries)),
        )
        object.__setattr__(
            self,
            "per_seed_outcomes",
            MappingProxyType(
                {
                    name: tuple(tuple(seed) for seed in seeds)
                    for name, seeds in self.per_seed_outcomes.items()
                }
            ),
        )
        normalized_vector_steps = dict(self.split_vector_env_step_calls)
        normalized_wall_times = dict(self.split_wall_times_s)
        for split, seed_outcomes in self.per_seed_outcomes.items():
            seed_count = len(seed_outcomes)
            env_counts = {len(seed) for seed in seed_outcomes}
            if seed_count <= 0 or len(env_counts) != 1 or 0 in env_counts:
                raise ValueError("Stair evaluation outcomes must form a fixed non-empty grid")
            wall_time_s = normalized_wall_times[split]
            if (
                isinstance(wall_time_s, bool)
                or not isinstance(wall_time_s, (int, float))
                or not math.isfinite(float(wall_time_s))
                or wall_time_s < 0.0
            ):
                raise ValueError("Stair split wall time must be finite and non-negative")
            vector_steps = normalized_vector_steps[split]
            if (
                isinstance(vector_steps, bool)
                or not isinstance(vector_steps, int)
                or vector_steps <= 0
            ):
                raise ValueError("Stair split vector env step calls must be positive")
        object.__setattr__(
            self,
            "split_vector_env_step_calls",
            MappingProxyType(normalized_vector_steps),
        )
        object.__setattr__(
            self,
            "split_wall_times_s",
            MappingProxyType(normalized_wall_times),
        )

    def public_summaries(self) -> dict[str, dict[str, Any]]:
        """Return detached JSON data for the subprocess evidence boundary."""

        projected = _copy_json_value(self.summaries)
        if not isinstance(projected, dict) or any(
            not isinstance(value, dict) for value in projected.values()
        ):
            raise TypeError("Stair evaluation summaries must project to JSON objects")
        return projected

    def execution_identity(self) -> dict[str, Any]:
        """Return the immutable inputs verified around this actual execution."""

        return {
            "schema_version": (
                "true_rl_agent.task.stair_climbing.eval_execution_identity.v1"
            ),
            "checkpoint_hash": self.checkpoint_hash,
            "seed_namespace": self.seed_namespace,
            "seed_cohort_hash": self.seed_cohort_hash,
            "splits": sorted(self.summaries),
            "execution_device": self.execution_device,
            "environment_source_preflight": dict(
                self.source_manifest_preflight
            ),
            "backend_execution_environment_verification": dict(
                self.backend_execution_environment_verification
            ),
        }

    def execution_cost_artifact(self) -> dict[str, Any]:
        """Return runner-owned diagnostics excluded from score and prompt projection."""

        split_costs = {}
        for split, seed_outcomes in self.per_seed_outcomes.items():
            env_count = len(seed_outcomes[0])
            vector_steps = self.split_vector_env_step_calls[split]
            split_costs[split] = {
                "wall_time_s": float(self.split_wall_times_s[split]),
                "seed_count": len(seed_outcomes),
                "evaluation_env_count_per_seed": env_count,
                "vector_env_step_calls": vector_steps,
                "simulated_env_control_steps": vector_steps * env_count,
                "closed_first_episode_control_steps": sum(
                    int(outcome.episode_steps)
                    for seed in seed_outcomes
                    for outcome in seed
                ),
            }
        return {
            "schema_version": STAIR_EVALUATION_EXECUTION_COST_SCHEMA_VERSION,
            "clock_source": "time.perf_counter_ns",
            "measurement_scope": "split_rollout_and_summary_aggregation",
            "splits": split_costs,
        }


class StairEvaluationRuntimeLease:
    """Keep a successful multi-split env alive until its subprocess exits."""

    def __init__(self) -> None:
        self._owner: Any | None = None

    @property
    def is_bound(self) -> bool:
        return self._owner is not None

    def bind(self, owner: Any) -> None:
        if owner is None:
            raise ValueError("Stair evaluation runtime owner cannot be None")
        if self._owner is not None:
            raise RuntimeError("Stair evaluation runtime lease is already bound")
        self._owner = owner


def _freeze_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("Stair evaluation summary keys must be strings")
        return MappingProxyType(
            {key: _freeze_json_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json_value(item) for item in value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Stair evaluation summaries must contain finite numbers")
        return value
    if value is None or isinstance(value, (str, bool, int)):
        return value
    raise TypeError(
        "Stair evaluation summaries must contain only JSON-compatible values"
    )


def _copy_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("Stair evaluation summary keys must be strings")
        return {key: _copy_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_copy_json_value(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Stair evaluation summaries must contain finite numbers")
        return value
    if value is None or isinstance(value, (str, bool, int)):
        return value
    raise TypeError(
        "Stair evaluation summaries must contain only JSON-compatible values"
    )


@dataclass(frozen=True)
class _StairTrainingWorkload:
    kind: str
    num_envs: int
    num_steps_per_env: int
    max_iterations: int
    budget_steps: int
    checkpoint_save_interval: int

    def __post_init__(self) -> None:
        if self.kind not in {"production", "phase_d_smoke"}:
            raise ValueError("Stair training workload kind is invalid")
        build_budget_accounting(
            budget_steps=self.budget_steps,
            num_envs=self.num_envs,
            num_steps_per_env=self.num_steps_per_env,
            max_iterations=self.max_iterations,
        )
        if (
            isinstance(self.checkpoint_save_interval, bool)
            or not isinstance(self.checkpoint_save_interval, int)
            or self.checkpoint_save_interval <= 0
        ):
            raise ValueError("Stair checkpoint save interval must be positive")


def execute_stair_training(
    *,
    config: StairClimbingTaskConfig,
    source_manifest_preflight: VerifiedStairExecutionSourceManifest,
    agent_configuration: RslRlAgentConfiguration,
    reward_program: QualifiedStairRewardProgram,
    device: str,
    run_dir: Path,
    checkpoint_output: Path,
    run_name: str,
    expected_initial_policy_state_hash: str | None = None,
    resolved_cell_manifest: ResolvedCellManifestV1 | None = None,
    runtime_comparison: Mapping[str, Any] | None = None,
    capture_training_numeric: bool = False,
    numeric_evidence_identity: Mapping[str, Any] | None = None,
) -> StairTrainingExecutionResult:
    """Run the only authoritative Stair train path and return closed evidence."""

    if not isinstance(config, StairClimbingTaskConfig):
        raise TypeError("Stair training requires StairClimbingTaskConfig")
    _validate_source_manifest_preflight(
        config=config,
        source_manifest_preflight=source_manifest_preflight,
    )
    execution = config.training_execution_contract()
    return _execute_stair_training(
        config=config,
        agent_configuration=agent_configuration,
        reward_program=reward_program,
        device=device,
        run_dir=run_dir,
        checkpoint_output=checkpoint_output,
        run_name=run_name,
        expected_initial_policy_state_hash=expected_initial_policy_state_hash,
        resolved_cell_manifest=resolved_cell_manifest,
        runtime_comparison=runtime_comparison,
        capture_training_numeric=capture_training_numeric,
        numeric_evidence_identity={**dict(numeric_evidence_identity or {}),
            "environment_source_preflight": source_manifest_preflight.to_summary_dict()},
        workload=_StairTrainingWorkload(
            kind="production",
            num_envs=config.train_num_envs,
            num_steps_per_env=config.train_steps_per_iteration,
            max_iterations=config.train_max_iterations,
            budget_steps=config.budget_steps,
            checkpoint_save_interval=int(
                execution["checkpoint_save_interval_iterations"]
            ),
        ),
    )


def execute_stair_training_smoke(
    *,
    config: StairClimbingTaskConfig,
    source_manifest_preflight: VerifiedStairExecutionSourceManifest,
    agent_configuration: RslRlAgentConfiguration,
    reward_program: QualifiedStairRewardProgram,
    device: str,
    run_dir: Path,
    checkpoint_output: Path,
    run_name: str,
    resolved_cell_manifest: ResolvedCellManifestV1 | None = None,
    runtime_comparison: Mapping[str, Any] | None = None,
    capture_training_numeric: bool = False,
    numeric_evidence_identity: Mapping[str, Any] | None = None,
) -> StairTrainingExecutionResult:
    """Run one fixed, non-eligible smoke through the production execution body."""

    _validate_source_manifest_preflight(
        config=config,
        source_manifest_preflight=source_manifest_preflight,
    )
    return _execute_stair_training(
        config=config,
        agent_configuration=agent_configuration,
        reward_program=reward_program,
        device=device,
        run_dir=run_dir,
        checkpoint_output=checkpoint_output,
        run_name=run_name,
        expected_initial_policy_state_hash=None,
        resolved_cell_manifest=resolved_cell_manifest,
        runtime_comparison=runtime_comparison,
        capture_training_numeric=capture_training_numeric,
        numeric_evidence_identity={**dict(numeric_evidence_identity or {}),
            "environment_source_preflight": source_manifest_preflight.to_summary_dict()},
        workload=_StairTrainingWorkload(
            kind="phase_d_smoke",
            num_envs=16,
            num_steps_per_env=8,
            max_iterations=1,
            budget_steps=128,
            checkpoint_save_interval=1,
        ),
    )


def _execute_stair_training(
    *,
    config: StairClimbingTaskConfig,
    agent_configuration: RslRlAgentConfiguration,
    reward_program: QualifiedStairRewardProgram,
    device: str,
    run_dir: Path,
    checkpoint_output: Path,
    run_name: str,
    expected_initial_policy_state_hash: str | None,
    resolved_cell_manifest: ResolvedCellManifestV1 | None,
    workload: _StairTrainingWorkload,
    runtime_comparison: Mapping[str, Any] | None = None,
    capture_training_numeric: bool = False,
    numeric_evidence_identity: Mapping[str, Any] | None = None,
) -> StairTrainingExecutionResult:

    _validate_training_inputs(
        config=config,
        agent_configuration=agent_configuration,
        reward_program=reward_program,
        device=device,
        run_dir=run_dir,
        checkpoint_output=checkpoint_output,
        run_name=run_name,
        expected_initial_policy_state_hash=expected_initial_policy_state_hash,
        resolved_cell_manifest=resolved_cell_manifest,
    )
    runtime_projection = _stair_runtime_projection_for_execution(resolved_cell_manifest)
    import gymnasium as gym
    import torch
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    from true_rl_agent.runner.training_numeric_capture import (
        TrainingNumericCapture,
        finish_training_numeric_capture,
    )

    from true_rl_agent.tasks.stair_climbing.isaaclab_reward_runtime import (
        get_stair_complete_reward_term,
        run_stair_reward_activation_preflight,
    )
    from true_rl_agent.tasks.stair_climbing.isaaclab_runtime_env import (
        build_stair_train_env_cfg,
        register_production_stair_envs,
        stair_training_distribution_runtime_evidence,
    )

    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
    env = None
    complete_term = None
    training_capture = None
    training_executed = False
    activation_preflight: dict[str, Any] | None = None
    training_summary: dict[str, Any] | None = None
    try:
        if checkpoint_output.exists():
            raise FileExistsError(
                "checkpoint output must be fresh for this Stair training attempt"
            )
        execution = config.training_execution_contract()
        torch.backends.cuda.matmul.allow_tf32 = bool(
            execution["torch_matmul_allow_tf32"]
        )
        torch.backends.cudnn.allow_tf32 = bool(execution["cudnn_allow_tf32"])
        torch.backends.cudnn.deterministic = bool(
            execution["cudnn_deterministic"]
        )
        torch.backends.cudnn.benchmark = bool(execution["cudnn_benchmark"])

        register_production_stair_envs()
        training_distribution_contract = (
            None
            if config.is_unified_protocol
            else config.protocol_v2.runner_owned_training_distribution
        )
        env_cfg = build_stair_train_env_cfg(
            None if config.is_unified_protocol else config.protocol_v1.to_dict(),
            seed=config.train_seed,
            device=device,
            reward_program=reward_program,
            runner_owned_training_distribution=training_distribution_contract,
            unified_protocol=(
                config.unified_protocol if config.is_unified_protocol else None
            ),
            runtime_projection=runtime_projection,
        )
        training_distribution_evidence = (
            stair_training_distribution_runtime_evidence(
                env_cfg,
                protocol=(
                    None if config.is_unified_protocol else config.protocol_v1.to_dict()
                ),
                contract=training_distribution_contract,
                unified_protocol=(
                    config.unified_protocol if config.is_unified_protocol else None
                ),
            )
        )
        if workload.kind == "phase_d_smoke":
            env_cfg.scene.num_envs = workload.num_envs
        elif int(env_cfg.scene.num_envs) != workload.num_envs:
            raise RuntimeError("Stair production env count drifted")
        if env_cfg.recorders is not None:
            raise RuntimeError("Stair training must not install a recorder")
        agent_cfg = isaaclab_locomotion_runtime_spec(
            runtime_projection.runtime_adapter_id
        ).instantiate_runner_cfg()
        effective_agent_execution = configure_stair_rsl_rl_agent(
            agent_cfg,
            configuration=agent_configuration,
            seed=config.train_seed,
            device=device,
            num_steps_per_env=workload.num_steps_per_env,
            max_iterations=workload.max_iterations,
            save_interval=workload.checkpoint_save_interval,
            run_name=run_name,
            experiment_name="true_rl_agent_stair",
            execution_contract=execution,
        )

        initial_assignment_evidence = None

        def create_and_reset_env() -> Any:
            nonlocal initial_assignment_evidence
            created = gym.make(STAIR_TRAIN_GYM_ID, cfg=env_cfg, render_mode=None)
            if config.is_unified_protocol:
                from true_rl_agent.tasks.stair_climbing.isaaclab_unified_terrain import (
                    apply_unified_initial_terrain_assignment,
                )

                initial_assignment_evidence = apply_unified_initial_terrain_assignment(
                    created.unwrapped,
                    seed=config.train_seed,
                    profile=config.unified_training_profile,
                )
            created.reset()
            return created

        env = _execute_stair_reward_program_boundary(
            create_and_reset_env,
            reward_program=reward_program,
            stage="activation_preflight",
        )
        base_env = env.unwrapped
        if str(base_env.device) != device or str(agent_cfg.device) != device:
            raise RuntimeError(
                "Stair training device does not match the requested execution device"
            )
        termination_manager_resolution = (
            verify_stair_vnext_termination_manager_resolution(
                base_env,
                runtime_projection=runtime_projection,
                force_threshold_n=float(
                    config.unified_protocol.episode_outcomes["signals"][
                        "forbidden_contact"
                    ]["force_threshold_n"]
                ),
            )
            if config.is_unified_protocol
            else verify_stair_termination_manager_resolution(
                base_env,
                protocol=config.protocol_v1.to_dict(),
            )
        )
        if list(base_env.reward_manager.active_terms) != ["complete_reward"]:
            raise RuntimeError("Stair RewardManager contains non-authoritative terms")
        term_cfg = base_env.reward_manager.get_term_cfg("complete_reward")
        if float(term_cfg.weight) != 1.0:
            raise RuntimeError("Stair complete reward manager weight drifted")
        complete_term = get_stair_complete_reward_term(base_env)
        if runtime_comparison is not None:
            complete_term.dispatcher.bind_runtime_comparison(**runtime_comparison)
        activation_preflight = _execute_stair_reward_program_boundary(
            lambda: run_stair_reward_activation_preflight(
                env,
                reward_program=reward_program,
            ),
            reward_program=reward_program,
            stage="activation_preflight",
        )

        wrapped_env = RslRlVecEnvWrapper(
            env,
            clip_actions=agent_cfg.clip_actions,
        )
        log_dir = run_dir / "rsl_rl"
        runner = OnPolicyRunner(
            wrapped_env,
            agent_cfg.to_dict(),
            log_dir=str(log_dir),
            device=agent_cfg.device,
        )
        effective_agent_execution["constructed_learner"] = (
            config.training_agent_configuration().verify_runner(runner)
        )
        resolved_observation_groups = verify_stair_rsl_rl_observation_groups(
            runner,
            execution_contract=execution,
        )
        if resolved_observation_groups is not None:
            effective_agent_execution["resolved_observation_groups"] = (
                resolved_observation_groups
            )
        initial_policy_state_hash = initial_model_state_hash(runner)
        initial_policy_state_match = (
            None
            if expected_initial_policy_state_hash is None
            else initial_policy_state_hash
            == expected_initial_policy_state_hash
        )
        training_summary = {
            "workload_kind": workload.kind,
            "device": str(agent_cfg.device),
            "initial_policy_state_hash": initial_policy_state_hash,
            "initial_policy_state_hash_schema_version": INITIAL_MODEL_STATE_HASH_SCHEMA_VERSION,
            "expected_initial_policy_state_hash": (
                expected_initial_policy_state_hash
            ),
            "initial_policy_state_match": initial_policy_state_match,
        }
        if (
            expected_initial_policy_state_hash is not None
            and initial_policy_state_match is not True
        ):
            raise RuntimeError(
                "Stair initial policy state does not match the paired reference"
            )
        runner.add_git_repo_to_log(__file__)
        if capture_training_numeric:
            # Bind after activation preflight and runner construction: those
            # calls are not PPO rollout steps and must not shift the windows.
            training_capture = TrainingNumericCapture(
                run_dir / "training_numeric_evidence",
                identity={**dict(numeric_evidence_identity or {}),
                          "task": "stair_climbing", "run_name": run_name,
                          "workload_kind": workload.kind, "program": reward_program.to_public_dict(),
                          "task_protocol_hash": config.task_protocol_hash,
                          "resolved_cell_manifest_hash": resolved_cell_manifest.manifest_hash if resolved_cell_manifest else None,
                          "resolved_cell_identity": resolved_cell_manifest.summary_identity() if resolved_cell_manifest else None,
                          "training_seed": config.train_seed, "run_dir": str(run_dir),
                          "initial_policy_state_hash": initial_policy_state_hash},
                semantics=complete_term.dispatcher.training_numeric_semantics(control_dt=float(base_env.step_dt)),
                planned_control_steps=workload.max_iterations * workload.num_steps_per_env,
                num_envs=workload.num_envs, steps_per_iteration=workload.num_steps_per_env,
                source_step_offset=complete_term.dispatcher.step_calls,
                initial_learning_iteration=int(runner.current_learning_iteration),
                reported_iteration=lambda: int(runner.current_learning_iteration),
            )
            complete_term.dispatcher.bind_training_numeric_capture(training_capture)
        training_executed = True
        _execute_stair_reward_program_boundary(
            lambda: runner.learn(
                num_learning_iterations=agent_cfg.max_iterations,
                init_at_random_ep_len=bool(
                    execution["init_at_random_episode_length"]
                ),
            ),
            reward_program=reward_program,
            stage="ppo_reward_execution",
        )
        latest_checkpoint = _find_latest_checkpoint(log_dir)
        if latest_checkpoint is None:
            raise RuntimeError("Stair RSL-RL training produced no checkpoint")
        if execution["checkpoint_selector"] != "final_iteration":
            raise RuntimeError("unsupported Stair checkpoint selector")
        latest_iteration = checkpoint_iteration(latest_checkpoint)
        expected_iteration = workload.max_iterations - 1
        if latest_iteration != expected_iteration:
            raise RuntimeError(
                "Stair final checkpoint iteration mismatch: "
                f"expected={expected_iteration} actual={latest_iteration}"
            )
        shutil.copy2(latest_checkpoint, checkpoint_output)
        source_hash = file_sha256(latest_checkpoint)
        checkpoint_hash = file_sha256(checkpoint_output)
        if source_hash is None or checkpoint_hash is None:
            raise RuntimeError("Stair checkpoint bytes could not be hashed")
        if source_hash != checkpoint_hash:
            raise RuntimeError("Stair checkpoint copy hash mismatch")
        runtime_activation = complete_term.runtime_summary()
        if runtime_activation["dispatcher_calls"] <= 0:
            raise RuntimeError("Stair complete reward dispatcher never executed")
        budget = build_budget_accounting(
            budget_steps=workload.budget_steps,
            num_envs=workload.num_envs,
            num_steps_per_env=workload.num_steps_per_env,
            max_iterations=workload.max_iterations,
        )
        checkpoint_retention = _retain_final_training_checkpoint(
            log_dir,
            keep=latest_checkpoint,
        )
        training_summary = {
            "budget_accounting": budget,
            "workload_kind": workload.kind,
            "num_envs": workload.num_envs,
            "num_steps_per_env": workload.num_steps_per_env,
            "max_iterations": workload.max_iterations,
            "budget_steps": workload.budget_steps,
            "device": str(agent_cfg.device),
            "log_dir": str(log_dir),
            "latest_checkpoint_iteration": latest_iteration,
            "checkpoint_retention": checkpoint_retention,
            "execution_contract": dict(execution),
            "effective_agent_execution": effective_agent_execution,
            "training_distribution": training_distribution_evidence,
            "initial_terrain_assignment": initial_assignment_evidence,
            "initial_policy_state_hash": initial_policy_state_hash,
            "initial_policy_state_hash_schema_version": INITIAL_MODEL_STATE_HASH_SCHEMA_VERSION,
            "expected_initial_policy_state_hash": (
                expected_initial_policy_state_hash
            ),
            "initial_policy_state_match": initial_policy_state_match,
            "numeric_evidence": finish_training_numeric_capture(
                training_capture, training_status="pass", requested=capture_training_numeric),
        }
        return StairTrainingExecutionResult(
            status="pass",
            training_executed=True,
            checkpoint={
                "uri": str(checkpoint_output),
                "checksum": checkpoint_hash,
                "role": "train_output",
                "metadata": {
                    "local_path": str(checkpoint_output),
                    "source_latest_checkpoint": str(latest_checkpoint),
                    "source_latest_checkpoint_sha256": source_hash,
                },
            },
            runtime_activation_summary={
                **runtime_activation,
                "runtime_status": STAIR_COMPLETE_REWARD_RUNTIME_STATUS,
                "activation_preflight": activation_preflight,
                "reward_manager_active_terms": list(
                    base_env.reward_manager.active_terms
                ),
                "reward_manager_term_weight": float(term_cfg.weight),
                "training_recorder_enabled": False,
                "termination_manager_resolution": termination_manager_resolution,
            },
            activation_preflight=activation_preflight,
            training=training_summary,
            candidate_failure=None,
            errors=(),
        )
    except StairCandidateProgramFailure as exc:
        runtime_summary = (
            complete_term.runtime_summary() if complete_term is not None else None
        )
        candidate_training = training_summary
        if candidate_training is None:
            candidate_training = {
                "workload_kind": workload.kind,
                "device": device,
                "expected_initial_policy_state_hash": (
                    expected_initial_policy_state_hash
                ),
            }
        candidate_training["numeric_evidence"] = finish_training_numeric_capture(
            training_capture, training_status="fail", requested=capture_training_numeric)
        return StairTrainingExecutionResult(
            status="fail",
            training_executed=training_executed,
            checkpoint=None,
            runtime_activation_summary=runtime_summary,
            activation_preflight=activation_preflight,
            training=candidate_training,
            candidate_failure=exc.evidence,
            errors=(),
        )
    except Exception as exc:  # noqa: BLE001 - execution evidence is fail closed
        runtime_summary = (
            complete_term.runtime_summary() if complete_term is not None else None
        )
        training_summary = dict(training_summary or {"workload_kind": workload.kind, "device": device})
        training_summary["numeric_evidence"] = finish_training_numeric_capture(
            training_capture, training_status="fail", requested=capture_training_numeric)
        return StairTrainingExecutionResult(
            status="fail",
            training_executed=training_executed,
            checkpoint=None,
            runtime_activation_summary=runtime_summary,
            activation_preflight=activation_preflight,
            training=training_summary,
            candidate_failure=None,
            errors=({"type": type(exc).__name__, "message": str(exc)},),
        )
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def _execute_stair_reward_program_boundary(
    callback: Callable[[], Any],
    *,
    reward_program: QualifiedStairRewardProgram,
    stage: str,
) -> Any:
    """Project only task-owned reward failures at the exact execution boundary."""

    if stage not in {"activation_preflight", "ppo_reward_execution"}:
        raise ValueError("unsupported Stair candidate failure stage")
    try:
        return callback()
    except StairRewardProgramError as exc:
        if reward_program.subject.kind != "candidate":
            raise
        if stage == "activation_preflight":
            failure = StairCandidateProgramFailure.activation_preflight()
        elif isinstance(exc, StairRewardOutputContractError):
            failure = StairCandidateProgramFailure.ppo_reward_output_contract()
        else:
            failure = StairCandidateProgramFailure.ppo_reward_execution()
        raise failure from exc


def execute_stair_evaluation(
    *,
    config: StairClimbingTaskConfig,
    source_manifest_preflight: VerifiedStairExecutionSourceManifest,
    backend_execution_environment_verification: Mapping[str, str],
    checkpoint_path: Path,
    device: str,
    seed_namespace: str,
    splits: tuple[str, ...],
    runtime_lease: StairEvaluationRuntimeLease | None = None,
    resolved_cell_manifest: ResolvedCellManifestV1 | None = None,
    native_video: NativeVideoOptions | None = None,
    numeric_evidence_dir: Path | None = None,
    numeric_evidence_identity: Mapping[str, Any] | None = None,
    numeric_evidence_max_bytes: int = 2 * 1024**3,
) -> StairEvaluationExecutionResult:
    """Load one policy and evaluate explicit splits in one isolated runtime."""

    if native_video is not None and resolved_cell_manifest is None:
        raise ValueError("native recording requires an explicit resolved cell manifest")
    if native_video is not None and splits != ("public_dev",):
        raise ValueError("Stair native recording requires the public_dev evaluation phase")
    if numeric_evidence_dir is not None and (
        splits != ("public_dev",) or resolved_cell_manifest is None or not numeric_evidence_identity
    ):
        raise ValueError("Stair numeric capture requires bound public_dev execution")
    if not isinstance(config, StairClimbingTaskConfig):
        raise TypeError("Stair evaluation requires StairClimbingTaskConfig")
    is_study_final = (
        config.requires_study_holdout
        and seed_namespace == STAIR_FINAL_CONFIRMATION_EVAL_NAMESPACE
    )
    if (config.final_holdout is not None) != is_study_final or (
        is_study_final and splits != ("sealed_validation",)
    ):
        raise ValueError("Stair private study instance is required only for sealed final evaluation")
    _validate_source_manifest_preflight(
        config=config,
        source_manifest_preflight=source_manifest_preflight,
    )
    resolved_checkpoint = resolve_existing_file(
        checkpoint_path,
        field_name="Stair evaluation checkpoint",
    )
    checkpoint_hash = file_sha256(resolved_checkpoint)
    if checkpoint_hash is None:
        raise ValueError("Stair evaluation checkpoint hash could not be computed")
    backend_verification = _validated_backend_execution_environment_verification(
        backend_execution_environment_verification
    )
    if not isinstance(device, str) or not device:
        raise ValueError("Stair evaluation device must be non-empty")
    valid_splits = STAIR_ORDERED_FORMAL_COHORTS
    if config.is_unified_protocol:
        valid_splits = (*valid_splits, "calibration")
    if (
        not isinstance(splits, tuple)
        or not splits
        or len(splits) != len(set(splits))
        or any(split not in valid_splits for split in splits)
    ):
        raise ValueError("Stair evaluation split set is invalid")
    is_calibration = splits == ("calibration",)
    if is_calibration != (seed_namespace == STAIR_CALIBRATION_EVAL_NAMESPACE):
        raise ValueError(
            "Stair C1 calibration split and seed namespace must be paired"
        )
    requires_runtime_lease = len(splits) > 1
    if config.is_unified_protocol and requires_runtime_lease:
        raise ValueError(
            "unified Stair evaluation requires one split-local environment per process"
        )
    if requires_runtime_lease and not isinstance(
        runtime_lease,
        StairEvaluationRuntimeLease,
    ):
        raise TypeError(
            "multi-split Stair evaluation requires a subprocess runtime lease"
        )
    runtime_projection = _stair_runtime_projection_for_execution(resolved_cell_manifest)
    numeric_capture = None
    if numeric_evidence_dir is not None:
        from true_rl_agent.tasks.stair_climbing.numeric_capture import StairNumericCapture
        numeric_capture = StairNumericCapture(
            numeric_evidence_dir, config=config, resolved_cell_manifest=resolved_cell_manifest,
            identity={**numeric_evidence_identity, "checkpoint_hash": checkpoint_hash,
                      "resolved_cell_manifest_hash": resolved_cell_manifest.manifest_hash,
                      "seed_namespace": seed_namespace, "splits": list(splits),
                      "environment_source_preflight": source_manifest_preflight.to_summary_dict(),
                      "backend_execution_environment_verification": dict(backend_execution_environment_verification)},
            max_bytes=numeric_evidence_max_bytes,
        )

    import gymnasium as gym
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    from true_rl_agent.tasks.stair_climbing.evaluation_contract import (
        build_stair_evaluation_summary,
    )
    from true_rl_agent.tasks.stair_climbing.unified_scoring import (
        build_unified_stair_summary,
    )
    from true_rl_agent.tasks.stair_climbing.evaluation_runtime import (
        run_stair_split_execution,
    )
    from true_rl_agent.tasks.stair_climbing.isaaclab_runtime_env import (
        build_stair_eval_env_cfg,
        register_production_stair_envs,
    )

    resolver = config.seed_resolver()
    seed_cohort_hash = resolver.namespace_cohort_hash(seed_namespace)
    register_production_stair_envs()
    env_cfg = build_stair_eval_env_cfg(
        None if config.is_unified_protocol else config.protocol_v1.to_dict(),
        seed=resolver.base_seed,
        device=device,
        evaluation_profile=config.evaluation_profile_config,
        ordered_evaluation_suite=config.ordered_evaluation_suite_config,
        evaluation_split=(splits[0] if config.is_unified_protocol else None),
        runtime_projection=runtime_projection,
        capture_contact_history=numeric_capture is not None,
        unified_protocol=(
            config.unified_protocol if config.is_unified_protocol else None
        ),
    )
    if native_video is not None:
        native_video.configure_viewer(env_cfg)
    base_env = gym.make(
        STAIR_EVAL_GYM_ID, cfg=env_cfg,
        render_mode="rgb_array" if native_video is not None else None,
    )
    recorder = None
    if native_video is not None:
        from true_rl_agent.runner.native_video import NativeVideoRecorder

        recorder = NativeVideoRecorder(base_env, native_video)
        base_env = recorder.env
    if requires_runtime_lease:
        assert runtime_lease is not None
        runtime_lease.bind(base_env)
    env = None
    try:
        if config.is_unified_protocol:
            verify_stair_vnext_termination_manager_resolution(
                base_env.unwrapped,
                runtime_projection=runtime_projection,
                force_threshold_n=float(
                    config.unified_protocol.episode_outcomes["signals"][
                        "forbidden_contact"
                    ]["force_threshold_n"]
                ),
            )
        else:
            verify_stair_termination_manager_resolution(
                base_env.unwrapped,
                protocol=config.protocol_v1.to_dict(),
            )
        agent_cfg = isaaclab_locomotion_runtime_spec(
            runtime_projection.runtime_adapter_id
        ).instantiate_runner_cfg()
        config.training_agent_configuration().apply_to(agent_cfg)
        agent_cfg.seed = config.train_seed
        agent_cfg.device = device
        actual_env_device = str(base_env.unwrapped.device)
        if actual_env_device != device or str(agent_cfg.device) != device:
            raise RuntimeError(
                "Stair evaluation device does not match the requested execution device"
            )
        execution = config.training_execution_contract()
        apply_stair_rsl_rl_runner_contract(
            agent_cfg,
            execution_contract=execution,
        )
        env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(
            env,
            agent_cfg.to_dict(),
            log_dir=None,
            device=agent_cfg.device,
        )
        verify_stair_rsl_rl_observation_groups(
            runner,
            execution_contract=execution,
        )
        split_counts = {
            "public_dev": config.public_seed_count,
            "public_accept": config.public_accept_seed_count,
            "sealed_validation": config.sealed_validation_seed_count,
        }
        if config.is_unified_protocol:
            split_counts["calibration"] = (
                config.ordered_evaluation_suite.calibration_seed_count
            )
        constructed_learner = config.training_agent_configuration().verify_runner(runner)
        checkpoint_loading = load_inference_checkpoint(runner, resolved_checkpoint)
        inference_policy = runner.get_inference_policy(
            device=env.unwrapped.device
        )
        summaries: dict[str, Mapping[str, Any]] = {}
        outcomes: dict[str, tuple[tuple[Any, ...], ...]] = {}
        split_vector_env_step_calls: dict[str, int] = {}
        split_wall_times_s: dict[str, float] = {}
        for split in splits:
            split_start_ns = time.perf_counter_ns()

            def seed_completed(completed: int, total: int) -> None:
                print(f"stair evaluation split={split} seed={completed}/{total}", flush=True)

            def finish_seed_video(seed_outcomes: tuple[Any, ...]) -> None:
                outcome = seed_outcomes[native_video.env_index]
                # Gym's first frame follows control step 1. Exclude the
                # terminal auto-reset frame and all later unscored episodes.
                record = recorder.finish(frame_count=outcome.episode_steps - 1)
                record["split"] = split
                record["episode_outcome"] = outcome.to_public_dict()
                if config.ordered_evaluation_suite_config is not None:
                    course = config.ordered_evaluation_suite_config.cell(outcome.evaluation_cell_id)
                    record["geometry_digest"] = course.geometry_digest

            split_execution = run_stair_split_execution(
                env=env,
                inference_policy=inference_policy,
                protocol=config.protocol_v1,
                semantics=config.evaluation_semantics,
                seed_resolver=resolver,
                split=split,
                seed_count=split_counts[split],
                seed_namespace=seed_namespace,
                runtime_projection=runtime_projection,
                progress_callback=seed_completed,
                on_seed_reset=(
                    (lambda index, seed: recorder.start(f"{split}_seed_{index}", seed))
                    if recorder is not None else None
                ),
                on_seed_completed=(finish_seed_video if recorder is not None else None),
                numeric_capture=numeric_capture,
            )
            per_seed = split_execution.per_seed_outcomes
            print(
                f"stair evaluation split={split} aggregation=started",
                flush=True,
            )
            summaries[split] = (
                build_unified_stair_summary(
                    suite=config.ordered_evaluation_suite,
                    semantics=config.evaluation_semantics,
                    split=split,
                    per_seed_outcomes=per_seed,
                )
                if config.is_unified_protocol
                else build_stair_evaluation_summary(
                    config.protocol_v1,
                    semantics=config.evaluation_semantics,
                    evaluation_protocol=config.evaluation_protocol(),
                    split=split,
                    per_seed_outcomes=per_seed,
                    evaluation_profile=config.evaluation_profile_config,
                )
            )
            outcomes[split] = tuple(tuple(seed) for seed in per_seed)
            wall_time_s = round(
                (time.perf_counter_ns() - split_start_ns) / 1_000_000_000.0,
                6,
            )
            split_vector_env_step_calls[split] = (
                split_execution.vector_env_step_calls
            )
            split_wall_times_s[split] = wall_time_s
            print(
                f"stair evaluation split={split} aggregation=completed",
                flush=True,
            )
        checkpoint_hash_after_execution = file_sha256(resolved_checkpoint)
        if checkpoint_hash_after_execution != checkpoint_hash:
            raise RuntimeError("Stair evaluation checkpoint bytes changed during execution")
        result = StairEvaluationExecutionResult(
            summaries=summaries,
            per_seed_outcomes=outcomes,
            split_vector_env_step_calls=split_vector_env_step_calls,
            split_wall_times_s=split_wall_times_s,
            checkpoint_hash=checkpoint_hash,
            seed_namespace=seed_namespace,
            seed_cohort_hash=seed_cohort_hash,
            source_manifest_preflight=source_manifest_preflight.to_summary_dict(),
            backend_execution_environment_verification=backend_verification,
            execution_device=actual_env_device,
            checkpoint_loading=checkpoint_loading,
            constructed_learner=constructed_learner,
            numeric_evidence=(numeric_capture.complete(summaries["public_dev"]) if numeric_capture is not None else None),
        )
        if recorder is not None:
            sidecar = {
                "schema_version": "true_rl_agent.stair_native_video.v1",
                "reclassified_by_video_exporter": False,
                "recorded_interval": "first_episode_control_step_1_to_terminal_minus_1",
                "execution": result.execution_identity(),
                "checkpoint_loading": dict(checkpoint_loading),
                "constructed_learner": dict(constructed_learner),
                "resolved_cell_manifest": resolved_cell_manifest.to_dict(),
                "evaluation_summaries": result.public_summaries(),
                "recordings": recorder.records,
            }
            (native_video.output_dir / "recordings.json").write_text(
                json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        print("stair evaluation result=ready", flush=True)
        return result
    finally:
        target = env if env is not None else base_env
        if requires_runtime_lease:
            print(
                "stair evaluation environment_close=owned_by_subprocess",
                flush=True,
            )
        else:
            try:
                print("stair evaluation environment_close=started", flush=True)
                target.close()
                print("stair evaluation environment_close=completed", flush=True)
            except Exception:
                pass


def checkpoint_iteration(path: Path) -> int:
    match = re.search(r"model_(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else -1


def _find_latest_checkpoint(log_dir: Path) -> Path | None:
    checkpoints = sorted(
        log_dir.rglob("model_*.pt"),
        key=lambda path: (
            checkpoint_iteration(path),
            path.stat().st_mtime,
        ),
    )
    return checkpoints[-1] if checkpoints else None


def _retain_final_training_checkpoint(
    log_dir: Path,
    *,
    keep: Path,
) -> dict[str, Any]:
    """Remove resumable intermediates after the authoritative final copy exists."""

    keep_resolved = keep.resolve(strict=True)
    removed_count = 0
    removed_bytes = 0
    for checkpoint in log_dir.rglob("model_*.pt"):
        if checkpoint.resolve(strict=True) == keep_resolved:
            continue
        removed_bytes += checkpoint.stat().st_size
        checkpoint.unlink()
        removed_count += 1
    return {
        "policy": "retain_final_iteration_only_after_authoritative_copy",
        "retained_checkpoint": str(keep),
        "removed_checkpoint_count": removed_count,
        "removed_checkpoint_bytes": removed_bytes,
    }


def _validate_source_manifest_preflight(
    *,
    config: StairClimbingTaskConfig,
    source_manifest_preflight: VerifiedStairExecutionSourceManifest,
) -> None:
    if not isinstance(
        source_manifest_preflight,
        VerifiedStairExecutionSourceManifest,
    ):
        raise TypeError(
            "Stair execution requires a verified source manifest preflight"
        )
    source_manifest_preflight.validate_for(config)


def _stair_runtime_projection_for_execution(
    manifest: ResolvedCellManifestV1 | None,
) -> StairRuntimeProjection:
    if manifest is None:
        return build_legacy_stair_runtime_projection()
    return stair_runtime_projection_from_manifest(manifest)


def _validated_source_manifest_preflight_summary(
    value: Mapping[str, str],
) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError("Stair evaluation source preflight must be an object")
    normalized = dict(value)
    if set(normalized) != {
        "schema_version",
        "status",
        "task_protocol_hash",
        "source_manifest_hash",
    }:
        raise ValueError("Stair evaluation source preflight fields mismatch")
    if (
        normalized.get("schema_version")
        != STAIR_EXECUTION_SOURCE_PREFLIGHT_SCHEMA_VERSION
        or normalized.get("status") != "pass"
    ):
        raise ValueError("Stair evaluation source preflight identity mismatch")
    normalized["task_protocol_hash"] = require_sha256_digest(
        normalized.get("task_protocol_hash"),
        "Stair evaluation source preflight task protocol hash",
    )
    normalized["source_manifest_hash"] = require_sha256_digest(
        normalized.get("source_manifest_hash"),
        "Stair evaluation source preflight source manifest hash",
    )
    return normalized


def _validated_backend_execution_environment_verification(
    value: Mapping[str, str],
) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError("Stair backend environment verification must be an object")
    normalized = dict(value)
    if set(normalized) != {
        "schema_version",
        "backend_id",
        "status",
        "expected_manifest_hash",
        "actual_manifest_hash",
    }:
        raise ValueError("Stair backend environment verification fields mismatch")
    expected = require_sha256_digest(
        normalized.get("expected_manifest_hash"),
        "Stair backend expected environment manifest hash",
    )
    actual = require_sha256_digest(
        normalized.get("actual_manifest_hash"),
        "Stair backend actual environment manifest hash",
    )
    if (
        normalized.get("schema_version")
        != BACKEND_EXECUTION_ENVIRONMENT_VERIFICATION_SCHEMA_VERSION
        or normalized.get("backend_id") != ISAAC_LAB_RSL_RL_BACKEND_ID
        or normalized.get("status") != "pass"
        or actual != expected
    ):
        raise ValueError("Stair backend environment verification mismatch")
    normalized["expected_manifest_hash"] = expected
    normalized["actual_manifest_hash"] = actual
    return normalized


def _validate_training_inputs(
    *,
    config: StairClimbingTaskConfig,
    agent_configuration: RslRlAgentConfiguration,
    reward_program: QualifiedStairRewardProgram,
    device: str,
    run_dir: Path,
    checkpoint_output: Path,
    run_name: str,
    expected_initial_policy_state_hash: str | None,
    resolved_cell_manifest: ResolvedCellManifestV1 | None,
) -> None:
    if not isinstance(config, StairClimbingTaskConfig):
        raise TypeError("Stair training requires StairClimbingTaskConfig")
    if not isinstance(agent_configuration, RslRlAgentConfiguration):
        raise TypeError("Stair training agent configuration is invalid")
    if agent_configuration != config.training_agent_configuration():
        raise ValueError("Stair training agent configuration differs from task config")
    if not isinstance(reward_program, QualifiedStairRewardProgram):
        raise TypeError("Stair training reward program is not qualified")
    from true_rl_agent.tasks.stair_climbing.component_smoke import (
        StairClimbingComponentSmokeAdapter,
    )

    active_contract = StairClimbingComponentSmokeAdapter(
        config,
        resolved_cell_manifest=resolved_cell_manifest,
    ).contract
    if reward_program.contract != active_contract:
        raise ValueError("Stair training reward contract differs from task config")
    if expected_initial_policy_state_hash is not None:
        require_sha256_digest(
            expected_initial_policy_state_hash,
            "Stair expected initial policy state hash",
        )
    for value, field_name in (
        (device, "device"),
        (run_name, "run_name"),
    ):
        if not isinstance(value, str) or not value or value != value.strip():
            raise ValueError(f"Stair training {field_name} must be canonical text")
    for value, field_name in (
        (run_dir, "run_dir"),
        (checkpoint_output, "checkpoint_output"),
    ):
        if not isinstance(value, Path):
            raise TypeError(f"Stair training {field_name} must be a Path")
    if run_dir.exists():
        raise FileExistsError("Stair training run_dir must be fresh")


__all__ = [
    "STAIR_EVALUATION_EXECUTION_COST_SCHEMA_VERSION",
    "STAIR_EXECUTION_SOURCE_PREFLIGHT_SCHEMA_VERSION",
    "StairEvaluationExecutionResult",
    "StairTrainingExecutionResult",
    "VerifiedStairExecutionSourceManifest",
    "checkpoint_iteration",
    "execute_stair_evaluation",
    "execute_stair_training",
    "execute_stair_training_smoke",
    "preflight_stair_execution_source_manifest",
]
