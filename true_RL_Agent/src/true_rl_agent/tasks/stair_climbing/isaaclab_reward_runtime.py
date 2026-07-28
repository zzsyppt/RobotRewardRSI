"""Isaac Lab binding for the Stair complete-reward dispatcher."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from isaaclab.managers import ManagerTermBase, RewardTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    euler_xyz_from_quat,
    quat_apply_inverse,
    yaw_quat,
)

from true_rl_agent.tasks.stair_climbing.protocol_v2 import (
    StairCompleteRewardRuntimeContract,
)
from true_rl_agent.tasks.stair_climbing.reward_inputs import StairRewardInputsV2
from true_rl_agent.tasks.stair_climbing.reward_program import (
    QualifiedStairRewardProgram,
    StairCompleteRewardDispatcher,
    StairRewardProgramError,
    assert_stair_reward_program_deterministic,
)
from true_rl_agent.tasks.stair_climbing.runtime_profile import (
    STAIR_REWARD_ACTIVATION_PREFLIGHT_SCHEMA_VERSION,
)
from true_rl_agent.tasks.stair_climbing.termination_capture import (
    StairTerminationCaptureError,
    capture_current_termination_masks,
)

STAIR_COMPLETE_REWARD_TERM_ATTRIBUTE = "_true_rl_stair_complete_reward_term"

_FOOT_BODY_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")
_BASE_BODY_NAMES = ("torso_link",)


class StairRewardInputBuilder:
    """Project current pre-reset Isaac buffers into detached public tensors."""

    def __init__(
        self,
        env: Any,
        *,
        contract: StairCompleteRewardRuntimeContract,
        base_protocol: Mapping[str, Any],
    ) -> None:
        if not isinstance(contract, StairCompleteRewardRuntimeContract):
            raise TypeError("Stair reward builder contract is invalid")
        if not isinstance(base_protocol, Mapping):
            raise TypeError("Stair reward builder base protocol is invalid")
        self._env = env
        self.contract = contract
        self._base_protocol = dict(base_protocol)
        robot = env.scene["robot"]
        contact_sensor = env.scene["contact_forces"]
        expected_joint_names = tuple(
            str(name)
            for name in base_protocol["policy_compatibility"]["robot_schema"][
                "joint_names"
            ]
        )
        self._joint_ids = _resolve_joint_ids(
            robot,
            expected_joint_names,
            "all policy joints",
        )
        self._joint_names = expected_joint_names
        self._ankle_ids = self._joint_group_ids(
            lambda name: "_ankle_pitch_joint" in name
            or "_ankle_roll_joint" in name,
            expected_count=4,
            context="ankle joints",
        )
        self._locomotion_ids = self._joint_group_ids(
            lambda name: any(
                token in name for token in ("_hip_", "_knee_joint", "_ankle_")
            ),
            expected_count=12,
            context="locomotion joints",
        )
        self._hip_knee_ids = self._joint_group_ids(
            lambda name: "_hip_" in name or "_knee_joint" in name,
            expected_count=8,
            context="hip and knee joints",
        )
        self._hip_yaw_roll_ids = self._joint_group_ids(
            lambda name: "_hip_yaw_joint" in name or "_hip_roll_joint" in name,
            expected_count=4,
            context="hip yaw and roll joints",
        )
        self._arm_ids = self._joint_group_ids(
            lambda name: "_shoulder_" in name or "_elbow_" in name,
            expected_count=10,
            context="arm joints",
        )
        self._finger_ids = self._joint_group_ids(
            lambda name: any(
                f"_{token}_joint" in name
                for token in ("five", "three", "six", "four", "zero", "one", "two")
            ),
            expected_count=14,
            context="finger joints",
        )
        self._torso_ids = self._joint_group_ids(
            lambda name: name == "torso_joint",
            expected_count=1,
            context="torso joint",
        )
        self._robot_foot_ids = _resolve_body_ids(
            robot,
            _FOOT_BODY_NAMES,
            "robot feet",
        )
        self._contact_foot_ids = _resolve_body_ids(
            contact_sensor,
            _FOOT_BODY_NAMES,
            "contact feet",
        )
        self._contact_base_ids = _resolve_body_ids(
            contact_sensor,
            _BASE_BODY_NAMES,
            "contact base",
        )
        terrain_goal = base_protocol["terrain"]["goal"]
        self._progress_start_m = float(terrain_goal["progress_start_m_from_origin"])
        self._progress_goal_m = float(terrain_goal["progress_goal_m_from_origin"])
        if self._progress_goal_m <= self._progress_start_m:
            raise ValueError("Stair reward progress goal is invalid")
        force_threshold = base_protocol["episode_outcomes"]["signals"]["slip"][
            "contact_force_threshold"
        ]
        self._foot_contact_threshold = float(force_threshold)
        height_scan_transform = contract.source_transform("height_scan")
        if (
            height_scan_transform["source_formula"]
            != "sensor_pos_w_z_minus_ray_hit_w_z_minus_offset"
            or height_scan_transform["corruption"] != "disabled"
            or height_scan_transform["ray_miss_policy"]
            != "saturate_infinity_to_clip_preserve_nan"
        ):
            raise ValueError("Unsupported Stair height scan source transform")
        self._height_scan_offset_m = float(height_scan_transform["offset_m"])
        self._height_scan_clip_min = float(height_scan_transform["clip_min"])
        self._height_scan_clip_max = float(height_scan_transform["clip_max"])
        self._field_names = contract.field_names
        self.last_input_fingerprint: dict[str, Any] | None = None

    def build(self) -> StairRewardInputsV2:
        env = self._env
        robot_data = env.scene["robot"].data
        contact_data = env.scene["contact_forces"].data
        height_data = env.scene["height_scanner"].data
        terrain_origins = env.scene.terrain.env_origins
        root_pos = robot_data.root_pos_w
        root_quat = robot_data.root_quat_w
        root_relative = root_pos - terrain_origins
        foot_pos = robot_data.body_pos_w[:, self._robot_foot_ids, :]
        foot_relative = foot_pos - terrain_origins.unsqueeze(1)
        command = env.command_manager.get_command("base_velocity")
        action = env.action_manager.action
        previous_action = env.action_manager.prev_action
        joint_pos = robot_data.joint_pos[:, self._joint_ids]
        joint_vel = robot_data.joint_vel[:, self._joint_ids]
        default_joint_pos = robot_data.default_joint_pos[:, self._joint_ids]
        joint_error = joint_pos - default_joint_pos
        roll, pitch, _ = euler_xyz_from_quat(root_quat)
        base_lin_vel_yaw = quat_apply_inverse(
            yaw_quat(root_quat),
            robot_data.root_lin_vel_w[:, :3],
        )
        force_history = contact_data.net_forces_w_history[
            :, :, self._contact_foot_ids, :
        ]
        current_forces = contact_data.net_forces_w
        if current_forces is None:
            raise StairRewardProgramError(
                "Stair reward inputs require current contact forces"
            )
        foot_forces = current_forces[:, self._contact_foot_ids, :]
        base_forces = current_forces[:, self._contact_base_ids, :]
        foot_contact = (
            force_history.norm(dim=-1).max(dim=1).values
            > self._foot_contact_threshold
        )
        raw_height_scan = (
            height_data.pos_w[:, 2].unsqueeze(1)
            - height_data.ray_hits_w[..., 2]
            - self._height_scan_offset_m
        )
        # Ray misses are +/-inf in Isaac Lab. Clamp saturates those misses while
        # preserving NaN so the single dispatcher guard still rejects real faults.
        height_scan = raw_height_scan.clamp(
            min=self._height_scan_clip_min,
            max=self._height_scan_clip_max,
        )
        progress = root_relative[:, 0] - self._progress_start_m
        goal_span = self._progress_goal_m - self._progress_start_m
        goal_fraction = (progress / goal_span).clamp(min=0.0, max=1.0)
        episode_fraction = (
            env.episode_length_buf.float() / float(env.max_episode_length)
        ).clamp(min=0.0, max=1.0)
        termination = capture_current_termination_masks(env)
        try:
            base_contact = termination.term_masks["base_contact"]
        except KeyError as exc:
            raise StairTerminationCaptureError(
                "Stair reward runtime requires the base_contact termination term"
            ) from exc

        values = {
            "base_height": root_relative[:, 2],
            "roll": roll,
            "pitch": pitch,
            "base_pos_xy": root_relative[:, :2],
            "base_lin_vel": robot_data.root_lin_vel_b[:, :3],
            "base_ang_vel": robot_data.root_ang_vel_b[:, :3],
            "projected_gravity": robot_data.projected_gravity_b[:, :3],
            "commanded_base_vel": command[:, :3],
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "joint_pos_error": joint_error,
            "last_action": action,
            "prev_action": previous_action,
            "action_delta": action - previous_action,
            "progress_fraction": episode_fraction,
            "base_lin_vel_yaw": base_lin_vel_yaw,
            "base_ang_vel_world": robot_data.root_ang_vel_w[:, :3],
            "base_quat_world": root_quat,
            "ankle_joint_pos": robot_data.joint_pos[:, self._ankle_ids],
            "ankle_soft_joint_pos_limits": robot_data.soft_joint_pos_limits[
                :, self._ankle_ids, :
            ],
            "locomotion_joint_torque": robot_data.applied_torque[
                :, self._locomotion_ids
            ],
            "hip_knee_joint_acc": robot_data.joint_acc[:, self._hip_knee_ids],
            "hip_yaw_roll_joint_pos_error": _joint_error(
                robot_data,
                self._hip_yaw_roll_ids,
            ),
            "arm_joint_pos_error": _joint_error(robot_data, self._arm_ids),
            "finger_joint_pos_error": _joint_error(robot_data, self._finger_ids),
            "torso_joint_pos_error": _joint_error(robot_data, self._torso_ids),
            "foot_air_time": contact_data.current_air_time[
                :, self._contact_foot_ids
            ],
            "foot_contact_time": contact_data.current_contact_time[
                :, self._contact_foot_ids
            ],
            "foot_contact_force_history": force_history,
            "foot_contact_force": foot_forces,
            "foot_contact": foot_contact,
            "foot_lin_vel": robot_data.body_lin_vel_w[
                :, self._robot_foot_ids, :
            ],
            "foot_pos_relative_to_terrain_origin": foot_relative,
            "base_pos_relative_to_terrain_origin": root_relative,
            "base_contact_force": base_forces,
            "height_scan": height_scan,
            "forward_progress_m": progress,
            "remaining_goal_distance_m": self._progress_goal_m
            - root_relative[:, 0],
            "goal_progress_fraction": goal_fraction,
            "terminated_non_timeout": termination.terminated_non_timeout_mask,
            "time_out": termination.time_out_mask,
            "base_contact": base_contact,
        }
        cloned = self._clone_and_validate(values)
        return StairRewardInputsV2._validated(
            contract=self.contract,
            values=cloned,
        )

    def _clone_and_validate(
        self,
        values: Mapping[str, Any],
    ) -> dict[str, Any]:
        if set(values) != set(self._field_names):
            raise StairRewardProgramError(
                "Stair reward input builder differs from the canonical ABI"
            )
        result: dict[str, Any] = {}
        output_storages: set[tuple[str, int]] = set()
        metadata = [] if self.last_input_fingerprint is None else None
        with torch.inference_mode(False):
            for field in self.contract.input_fields:
                source = values[field.name]
                if not torch.is_tensor(source):
                    raise StairRewardProgramError(
                        f"Stair reward input is not a tensor: {field.name}"
                    )
                expected_shape = (int(self._env.num_envs), *field.trailing_shape)
                if tuple(source.shape) != expected_shape:
                    raise StairRewardProgramError(
                        f"Stair reward input shape mismatch: {field.name}: "
                        f"expected={expected_shape!r} actual={tuple(source.shape)!r}"
                    )
                if source.device != torch.device(self._env.device):
                    raise StairRewardProgramError(
                        f"Stair reward input device mismatch: {field.name}"
                    )
                if field.dtype == "bool":
                    if source.dtype != torch.bool:
                        raise StairRewardProgramError(
                            f"Stair reward input dtype mismatch: {field.name}"
                        )
                elif not source.dtype.is_floating_point:
                    raise StairRewardProgramError(
                        f"Stair reward input dtype mismatch: {field.name}"
                    )
                clone = source.detach().clone()
                if clone.is_inference():
                    raise StairRewardProgramError(
                        f"Stair reward input lacks version tracking: {field.name}"
                    )
                storage = _storage_identity(clone)
                if (
                    storage == _storage_identity(source)
                    or storage in output_storages
                ):
                    raise StairRewardProgramError(
                        f"Stair reward input storage alias detected: {field.name}"
                    )
                output_storages.add(storage)
                result[field.name] = clone
                if metadata is not None:
                    metadata.append(
                        {
                            "name": field.name,
                            "shape": list(clone.shape),
                            "dtype": str(clone.dtype),
                            "device": str(clone.device),
                            "detached": clone.grad_fn is None,
                            "storage_unique": True,
                            "version_tracked": True,
                        }
                    )
        if metadata is not None:
            self.last_input_fingerprint = {
                "input_schema_hash": self.contract.input_schema_hash,
                "fields": metadata,
            }
        return result

    def _joint_group_ids(
        self,
        predicate: Any,
        *,
        expected_count: int,
        context: str,
    ) -> list[int]:
        ids = [
            self._joint_ids[index]
            for index, name in enumerate(self._joint_names)
            if predicate(name)
        ]
        if len(ids) != expected_count or len(ids) != len(set(ids)):
            raise RuntimeError(f"Stair {context} schema drifted")
        return ids


class StairCompleteRewardTerm(ManagerTermBase):
    """The sole active Stair RewardManager term."""

    def __init__(self, cfg: RewardTermCfg, env: Any) -> None:
        super().__init__(cfg, env)
        program = cfg.params.get("qualified_program")
        base_protocol = cfg.params.get("base_protocol")
        if not isinstance(program, QualifiedStairRewardProgram):
            raise TypeError("Stair RewardManager term requires a qualified program")
        if not isinstance(base_protocol, Mapping):
            raise TypeError("Stair RewardManager term requires the base protocol")
        self.program = program
        self.input_builder = StairRewardInputBuilder(
            env,
            contract=program.contract,
            base_protocol=base_protocol,
        )
        self.dispatcher = StairCompleteRewardDispatcher(
            program,
            num_envs=int(env.num_envs),
        )
        self.last_inputs: StairRewardInputsV2 | None = None
        if hasattr(env, STAIR_COMPLETE_REWARD_TERM_ATTRIBUTE):
            raise RuntimeError("Stair complete reward term was installed twice")
        setattr(env, STAIR_COMPLETE_REWARD_TERM_ATTRIBUTE, self)

    def __call__(
        self,
        env: Any,
        qualified_program: QualifiedStairRewardProgram,
        base_protocol: Mapping[str, Any],
    ) -> Any:
        if env is not self._env:
            raise RuntimeError("Stair reward term environment identity changed")
        if qualified_program.qualification_hash != self.program.qualification_hash:
            raise RuntimeError("Stair reward program identity changed after install")
        if not isinstance(base_protocol, Mapping):
            raise RuntimeError("Stair reward base protocol changed after install")
        inputs = self.input_builder.build()
        self.last_inputs = inputs
        return self.dispatcher.compute(
            inputs,
            common_step=int(env.common_step_counter),
        )

    def runtime_summary(self) -> dict[str, Any]:
        return {
            **self.dispatcher.runtime_summary(),
            "input_fingerprint": self.input_builder.last_input_fingerprint,
        }

    def latest_inputs(self) -> StairRewardInputsV2:
        if self.last_inputs is None:
            raise RuntimeError("Stair complete reward has not observed a transition")
        return self.last_inputs


@configclass
class StairCompleteRewardManagerCfg:
    """A RewardManager config with no legacy or sentinel reward terms."""

    complete_reward = RewardTermCfg(
        func=StairCompleteRewardTerm,
        weight=1.0,
        params={},
    )


def get_stair_complete_reward_term(env: Any) -> StairCompleteRewardTerm:
    base = getattr(env, "unwrapped", env)
    term = getattr(base, STAIR_COMPLETE_REWARD_TERM_ATTRIBUTE, None)
    if not isinstance(term, StairCompleteRewardTerm):
        raise RuntimeError("Stair complete reward term is not active")
    return term


def run_stair_reward_activation_preflight(
    env: Any,
    *,
    reward_program: QualifiedStairRewardProgram,
) -> dict[str, Any]:
    """Validate one real pre-reset transition before handing the env to PPO."""

    if not isinstance(reward_program, QualifiedStairRewardProgram):
        raise TypeError("Stair reward preflight requires a qualified program")
    base = getattr(env, "unwrapped", env)
    term = get_stair_complete_reward_term(base)
    if term.program.qualification_hash != reward_program.qualification_hash:
        raise RuntimeError("Stair reward preflight program identity changed")
    action = getattr(getattr(base, "action_manager", None), "action", None)
    if (
        not torch.is_tensor(action)
        or action.ndim != 2
        or int(action.shape[0]) != int(base.num_envs)
        or int(action.shape[1]) <= 0
    ):
        raise RuntimeError("Stair reward preflight action schema is invalid")
    calls_before = int(term.runtime_summary()["dispatcher_calls"])
    env.step(torch.zeros_like(action))
    calls_after = int(term.runtime_summary()["dispatcher_calls"])
    if calls_after != calls_before + 1:
        raise RuntimeError("Stair reward preflight did not execute exactly one transition")
    transition_inputs = term.latest_inputs()
    result = assert_stair_reward_program_deterministic(
        reward_program,
        lambda: transition_inputs,
    )
    transition_common_step = int(base.common_step_counter)
    env.reset()
    return {
        **result,
        "schema_version": STAIR_REWARD_ACTIVATION_PREFLIGHT_SCHEMA_VERSION,
        "real_transition_dispatcher_calls": 1,
        "transition_common_step": transition_common_step,
    }


def _resolve_joint_ids(
    robot: Any,
    names: tuple[str, ...],
    context: str,
) -> list[int]:
    ids, resolved_names = robot.find_joints(list(names), preserve_order=True)
    if tuple(resolved_names) != names or len(ids) != len(names):
        raise RuntimeError(f"Stair {context} schema drifted")
    return [int(item) for item in ids]


def _resolve_body_ids(
    asset: Any,
    names: tuple[str, ...],
    context: str,
) -> list[int]:
    ids, resolved_names = asset.find_bodies(list(names), preserve_order=True)
    if tuple(resolved_names) != names or len(ids) != len(names):
        raise RuntimeError(f"Stair {context} body schema drifted")
    return [int(item) for item in ids]


def _joint_error(robot_data: Any, ids: list[int]) -> Any:
    return robot_data.joint_pos[:, ids] - robot_data.default_joint_pos[:, ids]


def _storage_identity(value: Any) -> tuple[str, int]:
    return str(value.device), int(value.untyped_storage().data_ptr())


__all__ = [
    "STAIR_COMPLETE_REWARD_TERM_ATTRIBUTE",
    "StairCompleteRewardManagerCfg",
    "StairCompleteRewardTerm",
    "StairRewardInputBuilder",
    "get_stair_complete_reward_term",
    "run_stair_reward_activation_preflight",
]
