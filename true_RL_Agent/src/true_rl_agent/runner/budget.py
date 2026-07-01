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

