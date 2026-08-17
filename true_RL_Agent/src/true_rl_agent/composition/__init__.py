"""Task-neutral application composition contracts."""

from true_rl_agent.composition.base import (
    ResolvedTaskComposition,
    TaskCompositionContext,
    TaskCompositionError,
    TaskCompositionProvider,
    resolve_task_composition,
)
from true_rl_agent.composition.lazy import LazyTaskCompositionProvider
from true_rl_agent.composition.registry import TaskCompositionRegistry

__all__ = [
    "ResolvedTaskComposition",
    "LazyTaskCompositionProvider",
    "TaskCompositionContext",
    "TaskCompositionError",
    "TaskCompositionProvider",
    "TaskCompositionRegistry",
    "resolve_task_composition",
]
