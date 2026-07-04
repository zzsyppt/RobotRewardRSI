"""Push recovery task-owned fixed configuration."""

from __future__ import annotations

from dataclasses import dataclass


PHASE8B_TRAIN_NUM_ENVS = 4096
PHASE8B_TRAIN_MAX_ITERATIONS = 500
PHASE8B_TRAIN_STEPS_PER_ITERATION = 24
PHASE8B_ENV_STEP_BUDGET = PHASE8B_TRAIN_NUM_ENVS * PHASE8B_TRAIN_MAX_ITERATIONS * PHASE8B_TRAIN_STEPS_PER_ITERATION
PHASE8B_RUNNER_TIMEOUT_S = 7200


@dataclass(frozen=True)
class PushRecoveryTaskConfig:
    fall_threshold: float = 0.30
    episode_length: int = 512
    push_magnitude_xy: tuple[float, float] = (0.5, 1.0)
    push_interval_s: tuple[float, float] = (2.0, 5.0)
    success_predicate: str = "no_fall_and_recovers_velocity_after_push"
    eval_seed_visibility: str = "counts_only_no_raw_seed_ids"
    budget_steps: int = PHASE8B_ENV_STEP_BUDGET
    train_num_envs: int = PHASE8B_TRAIN_NUM_ENVS
    train_max_iterations: int = PHASE8B_TRAIN_MAX_ITERATIONS
    train_steps_per_iteration: int = PHASE8B_TRAIN_STEPS_PER_ITERATION
    eval_num_envs: int = 64
    public_seed_count: int = 3
    public_accept_seed_count: int = 1
    runner_timeout_s: int = PHASE8B_RUNNER_TIMEOUT_S
    checkpoint_selector: str = "runner_sealed_train_output_best_public_dev"
    accept_threshold: float = 0.05
