"""Explicit non-default registration root for Stair development phases."""

from __future__ import annotations

from true_rl_agent.composition.defaults import build_application_composition_registry
from true_rl_agent.composition.registry import TaskCompositionRegistry
from true_rl_agent.tasks.stair_climbing.composition import (
    StairClimbingCompositionProvider,
)
from true_rl_agent.training_programs import ReferenceProgramManifest


def build_stair_staging_composition_registry(
    *,
    reference_program_manifest: ReferenceProgramManifest,
) -> TaskCompositionRegistry:
    registry = build_application_composition_registry()
    registry.register(
        StairClimbingCompositionProvider(
            reference_program_manifest=reference_program_manifest
        )
    )
    return registry


__all__ = ["build_stair_staging_composition_registry"]
