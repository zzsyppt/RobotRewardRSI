"""Default application composition root.

Concrete task imports are intentionally confined to this wiring module.
"""

from __future__ import annotations

from true_rl_agent.composition.registry import TaskCompositionRegistry
from true_rl_agent.tasks.push_recovery.composition import (
    PUSH_RECOVERY_TASK_ID,
    PushRecoveryCompositionProvider,
)


def build_application_composition_registry() -> TaskCompositionRegistry:
    registry = TaskCompositionRegistry(default_task_id=PUSH_RECOVERY_TASK_ID)
    registry.register(PushRecoveryCompositionProvider())
    return registry
