"""Stair-climbing ownership of candidate component classifications."""

from __future__ import annotations

from true_rl_agent.candidates.schema import CandidateSurfacePolicy
from true_rl_agent.tasks.stair_climbing.phase0_contract import StairProtocolV1


def build_stair_candidate_surface_policy(
    protocol: StairProtocolV1,
) -> CandidateSurfacePolicy:
    """Derive the task-owned classification policy from the frozen protocol."""

    boundary = protocol.to_dict()["candidate_boundary"]
    return CandidateSurfacePolicy(
        parameter_only_components=frozenset(boundary["parameter_only_components"]),
        diagnostic_components=frozenset(boundary["diagnostic_components"]),
        engineering_components=frozenset(boundary["engineering_components"]),
        code_level_components=frozenset(boundary["code_level_components"]),
    )


__all__ = ["build_stair_candidate_surface_policy"]
