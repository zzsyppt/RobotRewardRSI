"""Task invariant enforcement helpers."""

from __future__ import annotations

from true_rl_agent.tasks.base import TaskInvariantSpec


def enforce_task_invariants(*, expected: TaskInvariantSpec, actual: TaskInvariantSpec) -> None:
    expected.assert_compatible_with(actual)

