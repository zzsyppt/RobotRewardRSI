"""G1-37 compatibility projection from Stair VNext inputs to frozen reward ABI V2."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from true_rl_agent.tasks.stair_climbing.protocol_v2 import (
    StairCompleteRewardRuntimeContract,
)
from true_rl_agent.tasks.stair_climbing.reward_inputs import StairRewardInputsVNext
from true_rl_agent.tasks.stair_climbing.legacy_runtime_projection import (
    STAIR_G1_37DOF_LEGACY_REWARD_ADAPTER_ID,
)
from true_rl_agent.tasks.stair_climbing.runtime_projection import (
    StairRuntimeProjection,
)


class StairG137LegacyRewardAdapter:
    """The only module allowed to interpret G1-specific V2 reward groups."""

    def __init__(
        self,
        *,
        projection: StairRuntimeProjection,
        contract: StairCompleteRewardRuntimeContract,
        base_protocol: Mapping[str, Any],
    ) -> None:
        if not isinstance(projection, StairRuntimeProjection):
            raise TypeError("Stair legacy reward projection is invalid")
        if (
            projection.legacy_reward_adapter_id
            != STAIR_G1_37DOF_LEGACY_REWARD_ADAPTER_ID
        ):
            raise ValueError("Stair cell does not authorize the G1 legacy reward ABI")
        if not isinstance(contract, StairCompleteRewardRuntimeContract):
            raise TypeError("Stair legacy reward contract is invalid")
        if not isinstance(base_protocol, Mapping):
            raise TypeError("Stair legacy reward base protocol is invalid")
        expected_joint_names = tuple(
            str(name)
            for name in base_protocol["policy_compatibility"]["robot_schema"][
                "joint_names"
            ]
        )
        if projection.joint_names != expected_joint_names:
            raise ValueError(
                "Stair legacy reward joint schema differs from resolved profile"
            )
        if projection.action_joint_names != expected_joint_names:
            raise ValueError(
                "Stair legacy reward action schema differs from resolved profile"
            )
        self.projection = projection
        self.contract = contract
        self._groups = {
            "ankle": _group_indices(
                expected_joint_names,
                lambda name: (
                    "_ankle_pitch_joint" in name or "_ankle_roll_joint" in name
                ),
                expected_count=4,
                context="ankle joints",
            ),
            "locomotion": _group_indices(
                expected_joint_names,
                lambda name: any(
                    token in name for token in ("_hip_", "_knee_joint", "_ankle_")
                ),
                expected_count=12,
                context="locomotion joints",
            ),
            "hip_knee": _group_indices(
                expected_joint_names,
                lambda name: "_hip_" in name or "_knee_joint" in name,
                expected_count=8,
                context="hip and knee joints",
            ),
            "hip_yaw_roll": _group_indices(
                expected_joint_names,
                lambda name: "_hip_yaw_joint" in name or "_hip_roll_joint" in name,
                expected_count=4,
                context="hip yaw and roll joints",
            ),
            "arm": _group_indices(
                expected_joint_names,
                lambda name: "_shoulder_" in name or "_elbow_" in name,
                expected_count=10,
                context="arm joints",
            ),
            "finger": _group_indices(
                expected_joint_names,
                lambda name: any(
                    f"_{token}_joint" in name
                    for token in (
                        "five",
                        "three",
                        "six",
                        "four",
                        "zero",
                        "one",
                        "two",
                    )
                ),
                expected_count=14,
                context="finger joints",
            ),
            "torso": _group_indices(
                expected_joint_names,
                lambda name: name == "torso_joint",
                expected_count=1,
                context="torso joint",
            ),
        }

    def project(self, inputs: StairRewardInputsVNext) -> dict[str, Any]:
        if not isinstance(inputs, StairRewardInputsVNext):
            raise TypeError("Stair legacy adapter requires VNext reward inputs")
        if inputs.projection_hash != self.projection.projection_hash:
            raise ValueError("Stair VNext input projection identity changed")
        group = self._groups
        values = {
            "base_height": inputs.base_height,
            "roll": inputs.roll,
            "pitch": inputs.pitch,
            "base_pos_xy": inputs.base_pos_xy,
            "base_lin_vel": inputs.base_lin_vel,
            "base_ang_vel": inputs.base_ang_vel,
            "projected_gravity": inputs.projected_gravity,
            "commanded_base_vel": inputs.commanded_base_vel,
            "joint_pos": inputs.joint_pos,
            "joint_vel": inputs.joint_vel,
            "joint_pos_error": inputs.joint_pos_error,
            "last_action": inputs.last_action,
            "prev_action": inputs.prev_action,
            "action_delta": inputs.action_delta,
            "progress_fraction": inputs.progress_fraction,
            "base_lin_vel_yaw": inputs.base_lin_vel_yaw,
            "base_ang_vel_world": inputs.base_ang_vel_world,
            "base_quat_world": inputs.base_quat_world,
            "ankle_joint_pos": inputs.joint_pos[:, list(group["ankle"])],
            "ankle_soft_joint_pos_limits": inputs.soft_joint_pos_limits[
                :, list(group["ankle"]), :
            ],
            "locomotion_joint_torque": inputs.joint_torque[
                :, list(group["locomotion"])
            ],
            "hip_knee_joint_acc": inputs.joint_acc[:, list(group["hip_knee"])],
            "hip_yaw_roll_joint_pos_error": inputs.joint_pos_error[
                :, list(group["hip_yaw_roll"])
            ],
            "arm_joint_pos_error": inputs.joint_pos_error[:, list(group["arm"])],
            "finger_joint_pos_error": inputs.joint_pos_error[:, list(group["finger"])],
            "torso_joint_pos_error": inputs.joint_pos_error[:, list(group["torso"])],
            "foot_air_time": inputs.support_air_time,
            "foot_contact_time": inputs.support_contact_time,
            "foot_contact_force_history": inputs.support_contact_force_history,
            "foot_contact_force": inputs.support_contact_force,
            "foot_contact": inputs.support_contact,
            "foot_lin_vel": inputs.support_lin_vel,
            "foot_pos_relative_to_terrain_origin": (
                inputs.support_pos_relative_to_terrain_origin
            ),
            "base_pos_relative_to_terrain_origin": (
                inputs.base_pos_relative_to_terrain_origin
            ),
            "base_contact_force": inputs.forbidden_contact_force,
            "height_scan": inputs.height_scan,
            "forward_progress_m": inputs.forward_progress_m,
            "remaining_goal_distance_m": inputs.remaining_goal_distance_m,
            "goal_progress_fraction": inputs.goal_progress_fraction,
            "terminated_non_timeout": inputs.terminated_non_timeout,
            "time_out": inputs.time_out,
            "base_contact": inputs.forbidden_contact,
        }
        if set(values) != set(self.contract.field_names):
            raise RuntimeError("Stair legacy reward adapter differs from frozen ABI")
        return values


def _group_indices(
    joint_names: tuple[str, ...],
    predicate: Any,
    *,
    expected_count: int,
    context: str,
) -> tuple[int, ...]:
    indices = tuple(index for index, name in enumerate(joint_names) if predicate(name))
    if len(indices) != expected_count:
        raise RuntimeError(f"Stair legacy {context} schema drifted")
    return indices


__all__ = ["StairG137LegacyRewardAdapter"]
