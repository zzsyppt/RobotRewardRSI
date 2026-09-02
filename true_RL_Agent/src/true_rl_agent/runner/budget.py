"""Runner-owned fixed-budget checks."""

from __future__ import annotations

from true_rl_agent.runner.specs import TrainJobSpec, require_positive_integer


class BudgetViolation(ValueError):
    """Raised when a job attempts to drift from the fixed runner budget."""


def enforce_fixed_budget(spec: TrainJobSpec, *, expected_budget_steps: int) -> None:
    expected_budget_steps = _require_budget_integer(
        expected_budget_steps,
        "expected_budget_steps",
    )
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
        _require_budget_integer(value, name)
    planned_env_steps = num_envs * num_steps_per_env * max_iterations
    if planned_env_steps != budget_steps:
        raise BudgetViolation(
            f"planned env steps mismatch: budget_steps={budget_steps}, planned_env_steps={planned_env_steps}"
        )
    return {
        "status": "pass",
        "budget_steps": budget_steps,
        "num_envs": num_envs,
        "num_steps_per_env": num_steps_per_env,
        "max_iterations": max_iterations,
        "planned_env_steps": planned_env_steps,
        "config_derived_total_timesteps": planned_env_steps,
    }


def _require_budget_integer(value: object, field_name: str) -> int:
    try:
        return require_positive_integer(value, field_name)
    except ValueError as exc:
        raise BudgetViolation(str(exc)) from exc
