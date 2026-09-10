"""Push recovery task-owned fixed configuration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from true_rl_agent.evaluation.acceptance_protocol import (
    PUBLIC_SCORE_BINDING_ACCEPTANCE_MEAN,
    AcceptanceProtocol,
)
from true_rl_agent.evaluation.protocol import (
    DEFAULT_EVAL_PROTOCOL_VERSION,
    EvaluationProtocol,
)
from true_rl_agent.runner.budget import build_budget_accounting

PHASE8B_TRAIN_NUM_ENVS = 4096
PHASE8B_TRAIN_MAX_ITERATIONS = 500
PHASE8B_TRAIN_STEPS_PER_ITERATION = 24
PHASE8B_ENV_STEP_BUDGET = PHASE8B_TRAIN_NUM_ENVS * PHASE8B_TRAIN_MAX_ITERATIONS * PHASE8B_TRAIN_STEPS_PER_ITERATION
PHASE8B_RUNNER_TIMEOUT_S = 7200

@dataclass(frozen=True)
class PushRecoveryTaskConfig:
    success_predicate: str = "no_fall_and_recovers_velocity_after_push"
    eval_seed_visibility: str = "counts_only_no_raw_seed_ids"
    budget_steps: int = PHASE8B_ENV_STEP_BUDGET
    train_num_envs: int = PHASE8B_TRAIN_NUM_ENVS
    train_max_iterations: int = PHASE8B_TRAIN_MAX_ITERATIONS
    train_steps_per_iteration: int = PHASE8B_TRAIN_STEPS_PER_ITERATION
    eval_num_envs: int = 64
    public_seed_count: int = 3
    public_accept_seed_count: int = 3
    sealed_validation_seed_count: int = 7
    eval_protocol_version: str = DEFAULT_EVAL_PROTOCOL_VERSION
    train_seed: int = 42
    eval_base_seed: int = 42
    runner_timeout_s: int = PHASE8B_RUNNER_TIMEOUT_S
    checkpoint_selector: str = "runner_sealed_train_output_best_public_dev"
    acceptance_score_source: str = "public_accept.score_stats.mean"
    absolute_accept_floor: float = 0.05
    min_policy_delta: float = 0.05
    runtime_config_path: str | None = None
    train_profile: str = "strong_push"
    eval_profile: str = "strong_push"
    reference_program_genesis: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.reference_program_genesis, bool):
            raise TypeError("reference_program_genesis must be boolean")
        for field_name in (
            "success_predicate",
            "eval_seed_visibility",
            "eval_protocol_version",
            "checkpoint_selector",
            "acceptance_score_source",
            "train_profile",
            "eval_profile",
        ):
            _require_non_empty_string(getattr(self, field_name), field_name)
        for field_name in (
            "budget_steps",
            "train_num_envs",
            "train_max_iterations",
            "train_steps_per_iteration",
            "eval_num_envs",
            "public_seed_count",
            "public_accept_seed_count",
            "sealed_validation_seed_count",
            "runner_timeout_s",
        ):
            _require_positive_integer(getattr(self, field_name), field_name)
        for field_name in ("train_seed", "eval_base_seed"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        for field_name in ("absolute_accept_floor", "min_policy_delta"):
            _require_finite_number(getattr(self, field_name), field_name)
        if self.runtime_config_path is not None:
            _require_non_empty_string(
                self.runtime_config_path,
                "runtime_config_path",
            )

        build_budget_accounting(
            budget_steps=self.budget_steps,
            num_envs=self.train_num_envs,
            num_steps_per_env=self.train_steps_per_iteration,
            max_iterations=self.train_max_iterations,
        )

        # Reuse the protocol constructors for task-owned relational validation.
        self.evaluation_protocol()
        self.acceptance_protocol()

    def evaluation_protocol(self) -> EvaluationProtocol:
        return EvaluationProtocol(
            eval_protocol_version=self.eval_protocol_version,
            public_seed_count=self.public_seed_count,
            public_accept_seed_count=self.public_accept_seed_count,
            sealed_validation_seed_count=self.sealed_validation_seed_count,
            eval_num_envs=self.eval_num_envs,
        )

    def acceptance_protocol(self) -> AcceptanceProtocol:
        return AcceptanceProtocol(
            protocol_id="unitree_push_recovery_acceptance_v1",
            score_source=self.acceptance_score_source,
            min_score_count=self.public_accept_seed_count,
            absolute_accept_floor=self.absolute_accept_floor,
            min_delta=self.min_policy_delta,
            public_score_binding=PUBLIC_SCORE_BINDING_ACCEPTANCE_MEAN,
        )

    def seed_cohort_hash(self) -> str:
        from true_rl_agent.tasks.push_recovery.seed_policy import (
            build_seed_cohort_hash,
        )

        return build_seed_cohort_hash(
            base_seed=self.eval_base_seed,
            evaluation_protocol=self.evaluation_protocol(),
        )

    def train_runtime_dynamics(self) -> dict:
        from true_rl_agent.tasks.push_recovery.isaac_config import (
            load_public_runtime_dynamics,
        )

        return load_public_runtime_dynamics(self.runtime_config_path, self.train_profile)

    def eval_runtime_dynamics(self) -> dict:
        from true_rl_agent.tasks.push_recovery.isaac_config import (
            load_public_runtime_dynamics,
        )

        return load_public_runtime_dynamics(self.runtime_config_path, self.eval_profile)

    @property
    def accept_threshold(self) -> float:
        """Compatibility projection for prompt text; protocol is authoritative."""

        return self.min_policy_delta


def _require_non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _require_positive_integer(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _require_finite_number(value: Any, field_name: str) -> int | float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{field_name} must be a finite non-boolean number")
    return value
