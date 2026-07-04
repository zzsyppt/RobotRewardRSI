"""Runner-owned fixed-budget checks."""

from __future__ import annotations

from true_rl_agent.runner.specs import TrainJobSpec


class BudgetViolation(ValueError):
    """Raised when a job attempts to drift from the fixed runner budget."""


def enforce_fixed_budget(spec: TrainJobSpec, *, expected_budget_steps: int) -> None:
    if expected_budget_steps <= 0:
        raise BudgetViolation("expected_budget_steps must be positive")
    if spec.budget_steps != expected_budget_steps:
        raise BudgetViolation(
            f"budget mismatch: expected {expected_budget_steps}, got {spec.budget_steps}"
        )


def build_budget_accounting(
    *,
    budget_steps: int,
    num_envs: int,
    num_steps_per_env: int,
    max_iterations: int,
) -> dict[str, int | str]:
    for name, value in (
        ("budget_steps", budget_steps),
        ("num_envs", num_envs),
        ("num_steps_per_env", num_steps_per_env),
        ("max_iterations", max_iterations),
    ):
        if int(value) <= 0:
            raise BudgetViolation(f"{name} must be positive")
    planned_env_steps = int(num_envs) * int(num_steps_per_env) * int(max_iterations)
    if planned_env_steps != int(budget_steps):
        raise BudgetViolation(
            f"planned env steps mismatch: budget_steps={budget_steps}, planned_env_steps={planned_env_steps}"
        )
    return {
        "status": "pass",
        "budget_steps": int(budget_steps),
        "num_envs": int(num_envs),
        "num_steps_per_env": int(num_steps_per_env),
        "max_iterations": int(max_iterations),
        "planned_env_steps": planned_env_steps,
        "config_derived_total_timesteps": planned_env_steps,
    }
