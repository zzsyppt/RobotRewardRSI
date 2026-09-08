"""Isaac Lab binding for the Stair complete-reward dispatcher."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from isaaclab.managers import ManagerTermBase, RewardTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    euler_xyz_from_quat,
    quat_apply_inverse,
    yaw_quat,
)
from true_rl_agent.backends.isaac_lab_arrays import as_torch, quaternion_wxyz

from true_rl_agent.tasks.stair_climbing.protocol_v2 import (
    StairCompleteRewardRuntimeContract,
)
from true_rl_agent.tasks.stair_climbing.legacy_reward_adapter import (
    StairG137LegacyRewardAdapter,
)
from true_rl_agent.tasks.stair_climbing.legacy_runtime_projection import (
    build_legacy_stair_runtime_projection,
)
from true_rl_agent.tasks.stair_climbing.reward_inputs import (
    StairRewardInputsV2,
    StairRewardInputsVNext,
)
from true_rl_agent.tasks.stair_climbing.reward_program import (
    QualifiedStairRewardProgram,
    StairCompleteRewardDispatcher,
    StairRewardProgramError,
    assert_stair_reward_program_deterministic,
)
from true_rl_agent.tasks.stair_climbing.runtime_profile import (
    STAIR_REWARD_ACTIVATION_PREFLIGHT_SCHEMA_VERSION,
)
from true_rl_agent.tasks.stair_climbing.runtime_projection import (
    STAIR_KINEMATIC_BASE_ARTICULATION_ROOT,
    STAIR_KINEMATIC_BASE_BODY_ROLE,
    ResolvedStairRuntimeProjection,
    StairRuntimeProjection,
    resolve_stair_runtime_projection,
)
from true_rl_agent.tasks.stair_climbing.termination_capture import (
    StairTerminationCaptureError,
    capture_current_termination_masks,
)

STAIR_COMPLETE_REWARD_TERM_ATTRIBUTE = "_true_rl_stair_complete_reward_term"


@dataclass(frozen=True)
class StairKinematicBaseState:
    """One explicitly selected kinematic frame shared by reward and evaluation."""

    pos_w: Any
    quat_xyzw: Any
    lin_vel_w: Any
    ang_vel_w: Any
    lin_vel_b: Any
    ang_vel_b: Any
    projected_gravity_b: Any


def stair_kinematic_base_state(
    robot_data: Any,
    *,
    runtime_projection: StairRuntimeProjection,
    resolved_projection: ResolvedStairRuntimeProjection,
) -> StairKinematicBaseState:
    """Read the binding-selected frame; legacy ABI choice never changes physics."""

    source = runtime_projection.kinematic_base_source
    if source == STAIR_KINEMATIC_BASE_ARTICULATION_ROOT:
        return StairKinematicBaseState(
            pos_w=as_torch(robot_data.root_pos_w),
            quat_xyzw=as_torch(robot_data.root_quat_w),
            lin_vel_w=as_torch(robot_data.root_lin_vel_w)[:, :3],
            ang_vel_w=as_torch(robot_data.root_ang_vel_w)[:, :3],
            lin_vel_b=as_torch(robot_data.root_lin_vel_b)[:, :3],
            ang_vel_b=as_torch(robot_data.root_ang_vel_b)[:, :3],
            projected_gravity_b=as_torch(robot_data.projected_gravity_b)[:, :3],
        )
    if source == STAIR_KINEMATIC_BASE_BODY_ROLE:
        base_id = resolved_projection.base_body_ids[0]
        pos_w = as_torch(robot_data.body_pos_w)[:, base_id, :]
        quat_w = as_torch(robot_data.body_quat_w)[:, base_id, :]
        lin_vel_w = as_torch(robot_data.body_lin_vel_w)[:, base_id, :]
        ang_vel_w = as_torch(robot_data.body_ang_vel_w)[:, base_id, :]
        gravity_w = torch.zeros_like(pos_w)
        gravity_w[:, 2] = -1.0
        return StairKinematicBaseState(
            pos_w=pos_w,
            quat_xyzw=quat_w,
            lin_vel_w=lin_vel_w,
            ang_vel_w=ang_vel_w,
            lin_vel_b=quat_apply_inverse(quat_w, lin_vel_w),
            ang_vel_b=quat_apply_inverse(quat_w, ang_vel_w),
            projected_gravity_b=quat_apply_inverse(quat_w, gravity_w),
        )
    raise StairRewardProgramError("Unsupported Stair kinematic base source")


class StairRewardInputBuilder:
    """Project current pre-reset Isaac buffers into detached public tensors."""

    def __init__(
        self,
        env: Any,
        *,
        contract: StairCompleteRewardRuntimeContract,
        base_protocol: Mapping[str, Any],
        runtime_projection: StairRuntimeProjection | None = None,
    ) -> None:
        if not isinstance(contract, StairCompleteRewardRuntimeContract):
            raise TypeError("Stair reward builder contract is invalid")
        if not isinstance(base_protocol, Mapping):
            raise TypeError("Stair reward builder base protocol is invalid")
        self._env = env
        self.contract = contract
        self._base_protocol = dict(base_protocol)
        self.runtime_projection = (
            runtime_projection
            if runtime_projection is not None
            else build_legacy_stair_runtime_projection()
        )
        if not isinstance(self.runtime_projection, StairRuntimeProjection):
            raise TypeError("Stair reward runtime projection is invalid")
        robot = env.scene["robot"]
        contact_sensor = env.scene[self.runtime_projection.contact_sensor_name]
        self._resolved_projection = resolve_stair_runtime_projection(
            self.runtime_projection,
            robot=robot,
            contact_sensor=contact_sensor,
        )
        self._legacy_adapter = (
            StairG137LegacyRewardAdapter(
                projection=self.runtime_projection,
                contract=contract,
                base_protocol=base_protocol,
            )
            if self.runtime_projection.legacy_reward_adapter_id is not None
            else None
        )
        terrain_goal = base_protocol["terrain"]["goal"]
        self._progress_start_m = float(terrain_goal["progress_start_m_from_origin"])
        goal_by_type = terrain_goal.get("progress_goal_m_by_terrain_type")
        if goal_by_type is None:
            self._progress_goal_m = float(terrain_goal["progress_goal_m_from_origin"])
            self._progress_goal_m_by_terrain_type = None
            if self._progress_goal_m <= self._progress_start_m:
                raise ValueError("Stair reward progress goal is invalid")
        else:
            goals = tuple(float(value) for value in goal_by_type)
            terrain = env.scene.terrain
            if (
                not goals
                or min(goals) <= self._progress_start_m
                or len(goals) != int(terrain.terrain_origins.shape[1])
            ):
                raise ValueError("Stair per-terrain reward progress goals are invalid")
            self._progress_goal_m = None
            self._progress_goal_m_by_terrain_type = torch.tensor(
                goals,
                dtype=terrain.env_origins.dtype,
                device=terrain.env_origins.device,
            )
        force_threshold = base_protocol["episode_outcomes"]["signals"]["slip"][
            "contact_force_threshold"
        ]
        self._support_contact_threshold = float(force_threshold)
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
        self.last_input_fingerprint: dict[str, Any] | None = None
        self.last_vnext_inputs: StairRewardInputsVNext | None = None

    def build(self) -> StairRewardInputsV2 | StairRewardInputsVNext:
        env = self._env
        robot_data = env.scene["robot"].data
        contact_sensor = env.scene[self.runtime_projection.contact_sensor_name]
        contact_data = contact_sensor.data
        height_data = env.scene[self.runtime_projection.height_sensor_name].data
        terrain_origins = env.scene.terrain.env_origins
        resolved = self._resolved_projection
        kinematic_base = stair_kinematic_base_state(
            robot_data,
            runtime_projection=self.runtime_projection,
            resolved_projection=resolved,
        )
        base_pos = kinematic_base.pos_w
        base_quat = kinematic_base.quat_xyzw
        base_lin_vel_world = kinematic_base.lin_vel_w
        base_ang_vel_world = kinematic_base.ang_vel_w
        base_lin_vel_body = kinematic_base.lin_vel_b
        base_ang_vel_body = kinematic_base.ang_vel_b
        projected_gravity = kinematic_base.projected_gravity_b
        base_relative = base_pos - terrain_origins
        support_pos = as_torch(robot_data.body_pos_w)[:, resolved.support_body_ids, :]
        support_relative = support_pos - terrain_origins.unsqueeze(1)
        command = as_torch(env.command_manager.get_command("base_velocity"))
        action = as_torch(env.action_manager.action)
        previous_action = as_torch(env.action_manager.prev_action)
        joint_pos = as_torch(robot_data.joint_pos)[:, resolved.joint_ids]
        joint_vel = as_torch(robot_data.joint_vel)[:, resolved.joint_ids]
        default_joint_pos = as_torch(robot_data.default_joint_pos)[:, resolved.joint_ids]
        joint_error = joint_pos - default_joint_pos
        joint_acc = as_torch(robot_data.joint_acc)[:, resolved.joint_ids]
        joint_torque = as_torch(robot_data.applied_torque)[:, resolved.joint_ids]
        soft_joint_pos_limits = as_torch(robot_data.soft_joint_pos_limits)[
            :, resolved.joint_ids, :
        ]
        roll, pitch, _ = euler_xyz_from_quat(base_quat)
        base_lin_vel_yaw = quat_apply_inverse(
            yaw_quat(base_quat),
            base_lin_vel_world,
        )
        force_history = as_torch(contact_data.net_forces_w_history)[
            :, :, resolved.contact_support_body_ids, :
        ]
        current_forces = as_torch(contact_data.net_forces_w)
        if current_forces is None:
            raise StairRewardProgramError(
                "Stair reward inputs require current contact forces"
            )
        support_forces = current_forces[:, resolved.contact_support_body_ids, :]
        forbidden_forces = current_forces[:, resolved.contact_forbidden_body_ids, :]
        support_contact = (
            force_history.norm(dim=-1).max(dim=1).values
            > self._support_contact_threshold
        )
        raw_height_scan = (
            as_torch(height_data.pos_w)[:, 2].unsqueeze(1)
            - as_torch(height_data.ray_hits_w)[..., 2]
            - self._height_scan_offset_m
        )
        # Ray misses are +/-inf in Isaac Lab. Clamp saturates those misses while
        # preserving NaN so the single dispatcher guard still rejects real faults.
        height_scan = raw_height_scan.clamp(
            min=self._height_scan_clip_min,
            max=self._height_scan_clip_max,
        )
        progress = base_relative[:, 0] - self._progress_start_m
        progress_goal_m = (
            self._progress_goal_m_by_terrain_type[
                env.scene.terrain.terrain_types
            ]
            if self._progress_goal_m_by_terrain_type is not None
            else self._progress_goal_m
        )
        goal_span = progress_goal_m - self._progress_start_m
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

        vnext_values = {
            "base_height": base_relative[:, 2],
            "roll": roll,
            "pitch": pitch,
            "base_pos_xy": base_relative[:, :2],
            "base_lin_vel": base_lin_vel_body,
            "base_ang_vel": base_ang_vel_body,
            "projected_gravity": projected_gravity,
            "commanded_base_vel": command[:, :3],
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "joint_pos_error": joint_error,
            "last_action": action,
            "prev_action": previous_action,
            "action_delta": action - previous_action,
            "joint_acc": joint_acc,
            "joint_torque": joint_torque,
            "soft_joint_pos_limits": soft_joint_pos_limits,
            "progress_fraction": episode_fraction,
            "base_lin_vel_yaw": base_lin_vel_yaw,
            "base_ang_vel_world": base_ang_vel_world,
            "base_quat_world": quaternion_wxyz(base_quat),
            "support_air_time": as_torch(contact_data.current_air_time)[
                :, resolved.contact_support_body_ids
            ],
            "support_contact_time": as_torch(contact_data.current_contact_time)[
                :, resolved.contact_support_body_ids
            ],
            "support_contact_force_history": force_history,
            "support_contact_force": support_forces,
            "support_contact": support_contact,
            "support_lin_vel": as_torch(robot_data.body_lin_vel_w)[
                :, resolved.support_body_ids, :
            ],
            "support_pos_relative_to_terrain_origin": support_relative,
            "base_pos_relative_to_terrain_origin": base_relative,
            "forbidden_contact_force": forbidden_forces,
            "height_scan": height_scan,
            "forward_progress_m": progress,
            "remaining_goal_distance_m": progress_goal_m - base_relative[:, 0],
            "goal_progress_fraction": goal_fraction,
            "terminated_non_timeout": termination.terminated_non_timeout_mask,
            "time_out": termination.time_out_mask,
            "forbidden_contact": base_contact,
        }
        if self.runtime_projection.support_transition_inputs:
            vnext_values["support_last_air_time"] = as_torch(contact_data.last_air_time)[
                :, resolved.contact_support_body_ids
            ]
            vnext_values["support_first_contact"] = (
                as_torch(contact_sensor.compute_first_contact(env.step_dt)).bool()[
                    :, resolved.contact_support_body_ids
                ]
            )
        joint_sources = {
            "joint_pos": joint_pos,
            "joint_pos_error": joint_error,
            "joint_acc": joint_acc,
            "joint_torque": joint_torque,
            "soft_joint_pos_limits": soft_joint_pos_limits,
        }
        group_indices = self.runtime_projection.joint_group_indices
        for field_name, role_id, source_name, _, _ in (
            self.runtime_projection.joint_group_input_specs
        ):
            vnext_values[field_name] = joint_sources[source_name][
                :, list(group_indices[role_id]), ...
            ]
        internal_vnext_inputs = StairRewardInputsVNext._validated(
            projection=self.runtime_projection,
            values=vnext_values,
        )
        if self._legacy_adapter is not None:
            values = self._legacy_adapter.project(internal_vnext_inputs)
            cloned = self._clone_and_validate(
                values,
                fields=self.contract.input_fields,
                schema_hash=self.contract.input_schema_hash,
            )
            return StairRewardInputsV2._validated(
                contract=self.contract,
                values=cloned,
            )
        cloned = self._clone_and_validate(
            vnext_values,
            fields=self.runtime_projection.reward_input_fields_vnext,
            schema_hash=self.runtime_projection.reward_input_schema_hash_vnext,
        )
        result = StairRewardInputsVNext._validated(
            projection=self.runtime_projection,
            values=cloned,
        )
        self.last_vnext_inputs = result
        return result

    def _clone_and_validate(
        self,
        values: Mapping[str, Any],
        *,
        fields: Sequence[Any],
        schema_hash: str,
    ) -> dict[str, Any]:
        field_names = tuple(field.name for field in fields)
        if set(values) != set(field_names):
            raise StairRewardProgramError(
                "Stair reward input builder differs from the canonical ABI"
            )
        result: dict[str, Any] = {}
        output_storages: set[tuple[str, int]] = set()
        metadata = [] if self.last_input_fingerprint is None else None
        with torch.inference_mode(False):
            for field in fields:
                source = values[field.name]
                if not torch.is_tensor(source):
                    raise StairRewardProgramError(
                        f"Stair reward input is not a tensor: {field.name}"
                    )
                expected_shape = (int(self._env.num_envs), *field.trailing_shape)
                actual_shape = tuple(source.shape)
                shape_matches = len(actual_shape) == len(expected_shape) and all(
                    actual == expected if isinstance(expected, int) else actual > 0
                    for actual, expected in zip(
                        actual_shape,
                        expected_shape,
                        strict=True,
                    )
                )
                if not shape_matches:
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
                if storage == _storage_identity(source) or storage in output_storages:
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
                "input_schema_hash": schema_hash,
                "fields": metadata,
            }
        return result

    def runtime_projection_summary(self) -> dict[str, Any]:
        return self._resolved_projection.to_summary_dict()


class StairCompleteRewardTerm(ManagerTermBase):
    """The sole active Stair RewardManager term."""

    def __init__(self, cfg: RewardTermCfg, env: Any) -> None:
        super().__init__(cfg, env)
        program = cfg.params.get("qualified_program")
        base_protocol = cfg.params.get("base_protocol")
        runtime_projection = cfg.params.get("runtime_projection")
        if not isinstance(program, QualifiedStairRewardProgram):
            raise TypeError("Stair RewardManager term requires a qualified program")
        if not isinstance(base_protocol, Mapping):
            raise TypeError("Stair RewardManager term requires the base protocol")
        self.program = program
        self.input_builder = StairRewardInputBuilder(
            env,
            contract=program.contract,
            base_protocol=base_protocol,
            runtime_projection=runtime_projection,
        )
        projection = self.input_builder.runtime_projection
        if (
            projection.legacy_reward_adapter_id is None
            and program.contract.input_schema_hash
            != projection.reward_input_schema_hash_vnext
        ):
            raise ValueError(
                "Stair reward program is not qualified for the resolved VNext ABI"
            )
        self.dispatcher = StairCompleteRewardDispatcher(
            program,
            num_envs=int(env.num_envs),
        )
        self.last_inputs: StairRewardInputsV2 | StairRewardInputsVNext | None = None
        if hasattr(env, STAIR_COMPLETE_REWARD_TERM_ATTRIBUTE):
            raise RuntimeError("Stair complete reward term was installed twice")
        setattr(env, STAIR_COMPLETE_REWARD_TERM_ATTRIBUTE, self)

    def __call__(
        self,
        env: Any,
        qualified_program: QualifiedStairRewardProgram,
        base_protocol: Mapping[str, Any],
        runtime_projection: StairRuntimeProjection | None = None,
    ) -> Any:
        if env is not self._env:
            raise RuntimeError("Stair reward term environment identity changed")
        if qualified_program.qualification_hash != self.program.qualification_hash:
            raise RuntimeError("Stair reward program identity changed after install")
        if not isinstance(base_protocol, Mapping):
            raise RuntimeError("Stair reward base protocol changed after install")
        installed_projection = self.input_builder.runtime_projection
        if runtime_projection is None:
            if installed_projection.legacy_reward_adapter_id is None:
                raise RuntimeError("Stair VNext reward projection disappeared after install")
        elif (
            not isinstance(runtime_projection, StairRuntimeProjection)
            or runtime_projection.projection_hash
            != installed_projection.projection_hash
        ):
            raise RuntimeError("Stair reward projection changed after install")
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
            "runtime_projection": self.input_builder.runtime_projection_summary(),
        }

    def latest_inputs(self) -> StairRewardInputsV2 | StairRewardInputsVNext:
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
        raise RuntimeError(
            "Stair reward preflight did not execute exactly one transition"
        )
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


def _storage_identity(value: Any) -> tuple[str, int]:
    return str(value.device), int(value.untyped_storage().data_ptr())


__all__ = [
    "STAIR_COMPLETE_REWARD_TERM_ATTRIBUTE",
    "StairCompleteRewardManagerCfg",
    "StairCompleteRewardTerm",
    "StairKinematicBaseState",
    "StairRewardInputBuilder",
    "get_stair_complete_reward_term",
    "run_stair_reward_activation_preflight",
    "stair_kinematic_base_state",
]
