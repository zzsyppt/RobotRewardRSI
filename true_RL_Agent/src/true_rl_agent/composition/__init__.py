"""Task-neutral application composition contracts."""

from true_rl_agent.composition.base import (
    ResolvedTaskComposition,
    TaskCompositionContext,
    TaskCompositionError,
    TaskCompositionProvider,
    resolve_task_composition,
)
from true_rl_agent.composition.cell_manifest import (
    CellEnvironmentDescriptor,
    CellResolution,
    CellVariantCatalog,
    CellVariantDefinition,
    ResolvedCellManifestV1,
    RobotTaskBinding,
    compile_cell_manifest,
    grant_legacy_training_binding_v1,
)
from true_rl_agent.composition.lazy import LazyTaskCompositionProvider
from true_rl_agent.composition.registry import TaskCompositionRegistry

__all__ = [
    "CellEnvironmentDescriptor",
    "CellResolution",
    "CellVariantCatalog",
    "CellVariantDefinition",
    "ResolvedTaskComposition",
    "ResolvedCellManifestV1",
    "RobotTaskBinding",
    "LazyTaskCompositionProvider",
    "TaskCompositionContext",
    "TaskCompositionError",
    "TaskCompositionProvider",
    "TaskCompositionRegistry",
    "compile_cell_manifest",
    "grant_legacy_training_binding_v1",
    "resolve_task_composition",
]
