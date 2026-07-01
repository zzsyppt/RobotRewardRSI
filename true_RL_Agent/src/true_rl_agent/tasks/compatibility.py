"""Task/backend compatibility checks."""

from __future__ import annotations

from dataclasses import dataclass

from true_rl_agent.backends.base import BackendAdapter
from true_rl_agent.tasks.base import TaskPlugin


@dataclass(frozen=True)
class CompatibilityReport:
    compatible: bool
    reasons: tuple[str, ...]


def check_task_backend_compatibility(
    *,
    task_plugin: TaskPlugin,
    backend_adapter: BackendAdapter,
    policy_initialization: str,
) -> CompatibilityReport:
    task_spec = task_plugin.task_spec
    backend_spec = backend_adapter.backend_spec
    reasons: list[str] = []

    if task_spec.task_id not in backend_spec.supported_tasks:
        reasons.append(f"backend_does_not_support_task:{task_spec.task_id}")
    if not task_spec.supports_policy_initialization(policy_initialization):
        reasons.append(f"task_does_not_support_policy_initialization:{policy_initialization}")
    if policy_initialization not in backend_spec.supported_policy_initializations:
        reasons.append(f"backend_does_not_support_policy_initialization:{policy_initialization}")

    return CompatibilityReport(compatible=not reasons, reasons=tuple(reasons))

