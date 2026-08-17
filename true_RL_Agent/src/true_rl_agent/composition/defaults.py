"""Default application composition root.

Concrete task imports are intentionally confined to this wiring module.
"""

from __future__ import annotations

from true_rl_agent.composition.base import TaskCompositionProvider
from true_rl_agent.composition.lazy import LazyTaskCompositionProvider
from true_rl_agent.composition.registry import TaskCompositionRegistry
from true_rl_agent.tasks.push_recovery.composition import (
    PUSH_RECOVERY_TASK_ID,
    PushRecoveryCompositionProvider,
)

_STAIR_TASK_ID = "unitree_stair_climbing"


def build_application_composition_registry() -> TaskCompositionRegistry:
    registry = TaskCompositionRegistry(default_task_id=PUSH_RECOVERY_TASK_ID)
    registry.register(PushRecoveryCompositionProvider())
    registry.register(
        LazyTaskCompositionProvider(
            task_id=_STAIR_TASK_ID,
            provider_factory=_build_stair_composition_provider,
        )
    )
    return registry


def _build_stair_composition_provider() -> TaskCompositionProvider:
    from true_rl_agent.tasks.stair_climbing.composition import (
        StairClimbingCompositionProvider,
    )

    return StairClimbingCompositionProvider()
