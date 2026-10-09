"""Legacy G1-37 entrypoint kept outside morphology-neutral Stair projection code."""

from __future__ import annotations

from pathlib import Path

from true_rl_agent.composition.cell_manifest import RobotTaskBinding
from true_rl_agent.json_contract import load_strict_json_object
from true_rl_agent.robots.defaults import (
    G1_37DOF_ISAACLAB_ADAPTER_ID,
    G1_37DOF_ISAACLAB_PROFILE_ID,
    build_default_robot_registry,
)
from true_rl_agent.tasks.stair_climbing.runtime_projection import (
    StairRuntimeProjection,
    stair_runtime_projection_from_execution,
)

STAIR_G1_37DOF_LEGACY_REWARD_ADAPTER_ID = "stair_g1_37dof_reward_v2_adapter_v1"

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_LEGACY_BINDING_PATH = (
    _PROJECT_ROOT / "configs/tasks/stair_g1_37dof_clean_legacy_binding_v1.json"
)


def build_legacy_stair_runtime_projection() -> StairRuntimeProjection:
    """Resolve the no-manifest compatibility entry from the frozen G1 binding."""

    registry = build_default_robot_registry()
    profile, adapter = registry.execution(
        robot_profile_id=G1_37DOF_ISAACLAB_PROFILE_ID,
        runtime_adapter_id=G1_37DOF_ISAACLAB_ADAPTER_ID,
    )
    binding = RobotTaskBinding.from_mapping(
        load_strict_json_object(
            _LEGACY_BINDING_PATH,
            context="Stair legacy robot task binding",
            require_canonical=True,
            trailing_newline=True,
        )
    )
    projection = stair_runtime_projection_from_execution(
        profile=profile,
        adapter=adapter,
        binding=binding,
    )
    if (
        projection.legacy_reward_adapter_id
        != STAIR_G1_37DOF_LEGACY_REWARD_ADAPTER_ID
    ):
        raise ValueError("Stair legacy binding does not authorize its reward adapter")
    return projection


__all__ = [
    "STAIR_G1_37DOF_LEGACY_REWARD_ADAPTER_ID",
    "build_legacy_stair_runtime_projection",
]
