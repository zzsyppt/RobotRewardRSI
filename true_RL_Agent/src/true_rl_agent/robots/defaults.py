"""Version-controlled legacy robot profiles used by the composition root."""

from __future__ import annotations

from pathlib import Path

from true_rl_agent.json_contract import strict_json_object_loads
from true_rl_agent.robots.base import RobotProfile, RobotRuntimeAdapter
from true_rl_agent.robots.registry import RobotRegistry

G1_37DOF_ISAACLAB_PROFILE_ID = "unitree_g1_37dof_isaaclab_v1"
G1_29DOF_UNITREE_RL_LAB_PROFILE_ID = "unitree_g1_29dof_unitree_rl_lab_v1"
G1_37DOF_ISAACLAB_ADAPTER_ID = "g1_37dof_isaaclab_adapter_v1"
G1_29DOF_UNITREE_RL_LAB_ADAPTER_ID = "g1_29dof_unitree_rl_lab_adapter_v1"

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_PROFILE_PATHS = (
    _PROJECT_ROOT / "configs/robots/unitree_g1_37dof_isaaclab_v1.json",
    _PROJECT_ROOT / "configs/robots/unitree_g1_29dof_unitree_rl_lab_v1.json",
)


def build_default_robot_registry() -> RobotRegistry:
    registry = RobotRegistry()
    profiles = tuple(_load_profile(path) for path in _PROFILE_PATHS)
    for profile in profiles:
        registry.register_profile(profile)
    by_id = {profile.robot_profile_id: profile for profile in profiles}
    g1_37 = by_id[G1_37DOF_ISAACLAB_PROFILE_ID]
    g1_29 = by_id[G1_29DOF_UNITREE_RL_LAB_PROFILE_ID]
    registry.register_runtime_adapter(
        RobotRuntimeAdapter(
            runtime_adapter_id=G1_37DOF_ISAACLAB_ADAPTER_ID,
            robot_profile_id=g1_37.robot_profile_id,
            backend_id="isaac_lab_rsl_rl",
            runtime_provider=str(g1_37.runtime_facts["runtime_provider"]),
            upstream_revision=str(g1_37.runtime_facts["upstream_revision"]),
            articulation_source="isaaclab_assets:G1_MINIMAL_CFG",
            provided_capabilities=frozenset(
                {"articulation_runtime", "contact_sensor_runtime"}
            ),
        )
    )
    registry.register_runtime_adapter(
        RobotRuntimeAdapter(
            runtime_adapter_id=G1_29DOF_UNITREE_RL_LAB_ADAPTER_ID,
            robot_profile_id=g1_29.robot_profile_id,
            backend_id="isaac_lab_rsl_rl",
            runtime_provider=str(g1_29.runtime_facts["runtime_provider"]),
            upstream_revision=str(g1_29.runtime_facts["upstream_revision"]),
            articulation_source=(
                "unitree_rl_lab.assets.robots.unitree:UNITREE_G1_29DOF_CFG"
            ),
            provided_capabilities=frozenset(
                {"articulation_runtime", "contact_sensor_runtime"}
            ),
        )
    )
    return registry


def _load_profile(path: Path) -> RobotProfile:
    payload = strict_json_object_loads(
        path.read_text(encoding="utf-8"),
        context=f"robot profile {path.name}",
    )
    return RobotProfile.from_mapping(payload)


__all__ = [
    "G1_29DOF_UNITREE_RL_LAB_ADAPTER_ID",
    "G1_29DOF_UNITREE_RL_LAB_PROFILE_ID",
    "G1_37DOF_ISAACLAB_ADAPTER_ID",
    "G1_37DOF_ISAACLAB_PROFILE_ID",
    "build_default_robot_registry",
]
