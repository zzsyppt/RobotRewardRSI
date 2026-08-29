"""Production Isaac Lab environment and recorder for Stair evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import gymnasium as gym
from isaaclab.managers import (
    DatasetExportMode,
    RecorderManagerBaseCfg,
    RecorderTerm,
    RecorderTermCfg,
    SceneEntityCfg,
)
from isaaclab.utils import configclass

from true_rl_agent.robots.isaaclab_locomotion import (
    isaaclab_locomotion_runtime_spec,
)
from true_rl_agent.tasks.stair_climbing.evaluation_profile import (
    StairEvaluationProfile,
)
from true_rl_agent.tasks.stair_climbing.isaaclab_env_cfg import build_stair_env_cfg
from true_rl_agent.tasks.stair_climbing.isaaclab_evaluation_terrain import (
    build_ordered_eval_terrain_generator,
    build_profile_eval_terrain_generator,
)
from true_rl_agent.tasks.stair_climbing.isaaclab_reward_runtime import (
    StairCompleteRewardManagerCfg,
    stair_kinematic_base_state,
)
from true_rl_agent.tasks.stair_climbing.isaaclab_unified_terrain import (
    build_unified_train_terrain_generator,
    goal_progress_m_by_terrain_type,
    stair_goal_progress_levels,
)
from true_rl_agent.tasks.stair_climbing.legacy_runtime_projection import (
    build_legacy_stair_runtime_projection,
)
from true_rl_agent.tasks.stair_climbing.ordered_evaluation import (
    STAIR_ORDERED_RUNNABLE_COHORTS,
    StairOrderedEvaluationSuite,
)
from true_rl_agent.tasks.stair_climbing.reward_program import (
    QualifiedStairRewardProgram,
)
from true_rl_agent.tasks.stair_climbing.runtime_profile import (
    STAIR_EVAL_GYM_ID,
    STAIR_EVAL_TRANSITION_EXTRAS_KEY,
    STAIR_TRAIN_GYM_ID,
    STAIR_TRANSITION_RECORDER_SCHEMA_VERSION,
)
from true_rl_agent.tasks.stair_climbing.runtime_projection import (
    STAIR_TRANSITION_VNEXT_SCHEMA_VERSION,
    StairRuntimeProjection,
    resolve_stair_runtime_projection,
)
from true_rl_agent.tasks.stair_climbing.termination_capture import (
    capture_current_termination_masks,
    install_stair_termination_capture,
)
from true_rl_agent.tasks.stair_climbing.training_distribution import (
    resolved_stair_training_distribution,
)
from true_rl_agent.tasks.stair_climbing.unified_protocol import (
    StairUnifiedProtocol,
)


class StairEvalTransitionRecorder(RecorderTerm):
    """Capture runner-owned task state before done environments are reset."""

    def __init__(self, cfg: RecorderTermCfg, env: Any) -> None:
        super().__init__(cfg, env)
        projection = cfg.runtime_projection
        if projection is None:
            projection = build_legacy_stair_runtime_projection()
        if not isinstance(projection, StairRuntimeProjection):
            raise TypeError("Stair recorder runtime projection is invalid")
        self.runtime_projection = projection
        robot = env.scene["robot"]
        contact_sensor = env.scene[projection.contact_sensor_name]
        self._resolved_projection = resolve_stair_runtime_projection(
            projection,
            robot=robot,
            contact_sensor=contact_sensor,
        )
        terrain = env.scene.terrain
        terrain_types = getattr(terrain, "terrain_types", None)
        if terrain_types is None:
            raise RuntimeError("Stair eval recorder requires terrain type identity")
        self._terrain_env_origins_w = terrain.env_origins.detach().clone()
        self._terrain_types = terrain_types.detach().clone()

    def record_post_step(self) -> tuple[None, None]:
        termination = capture_current_termination_masks(self._env)
        scene = self._env.scene
        robot_data = scene["robot"].data
        contact_data = scene[self.runtime_projection.contact_sensor_name].data
        contact_forces = contact_data.net_forces_w
        if contact_forces is None:
            raise RuntimeError("Stair eval recorder requires current contact forces")

        resolved = self._resolved_projection
        kinematic_base = stair_kinematic_base_state(
            robot_data,
            runtime_projection=self.runtime_projection,
            resolved_projection=resolved,
        )
        base_pos_w = kinematic_base.pos_w
        base_quat_w = kinematic_base.quat_w
        common = {
            "joint_pos": robot_data.joint_pos[:, resolved.joint_ids].detach().clone(),
            "episode_length": self._env.episode_length_buf.detach().clone(),
            "terrain_env_origins_w": self._terrain_env_origins_w,
            "terrain_types": self._terrain_types,
            "termination_term_order": termination.term_order,
            "termination_masks": dict(termination.term_masks),
            "done_mask": termination.done_mask,
        }
        support = {
            "support_pos_w": robot_data.body_pos_w[:, resolved.support_body_ids, :]
            .detach()
            .clone(),
            "support_lin_vel_w": robot_data.body_lin_vel_w[
                :, resolved.support_body_ids, :
            ]
            .detach()
            .clone(),
            "support_contact_forces_w": contact_forces[
                :, resolved.contact_support_body_ids, :
            ]
            .detach()
            .clone(),
            "forbidden_contact_forces_w": contact_forces[
                :, resolved.contact_forbidden_body_ids, :
            ]
            .detach()
            .clone(),
        }
        if self.runtime_projection.legacy_reward_adapter_id is not None:
            payload = {
                **common,
                "schema_version": STAIR_TRANSITION_RECORDER_SCHEMA_VERSION,
                "root_pos_w": base_pos_w.detach().clone(),
                "root_quat_w": base_quat_w.detach().clone(),
                "foot_pos_w": support["support_pos_w"],
                "foot_lin_vel_w": support["support_lin_vel_w"],
                "foot_contact_forces_w": support["support_contact_forces_w"],
                "base_contact_forces_w": support["forbidden_contact_forces_w"],
            }
        else:
            payload = {
                **common,
                **support,
                "schema_version": STAIR_TRANSITION_VNEXT_SCHEMA_VERSION,
                "base_pos_w": base_pos_w.detach().clone(),
                "base_quat_w": base_quat_w.detach().clone(),
                "runtime_projection_hash": self.runtime_projection.projection_hash,
                "reward_input_schema_hash_vnext": (
                    self.runtime_projection.reward_input_schema_hash_vnext
                ),
                "support_count": self.runtime_projection.support_count,
                "forbidden_contact_count": (
                    self.runtime_projection.forbidden_contact_count
                ),
            }
        self._env.extras[STAIR_EVAL_TRANSITION_EXTRAS_KEY] = payload
        return None, None

    def runtime_summary(self) -> dict[str, Any]:
        return self._resolved_projection.to_summary_dict()


@configclass
class StairEvalTransitionRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = StairEvalTransitionRecorder
    runtime_projection: StairRuntimeProjection | None = None


@configclass
class StairEvalRecorderManagerCfg(RecorderManagerBaseCfg):
    transition = StairEvalTransitionRecorderCfg()
    dataset_export_mode = DatasetExportMode.EXPORT_NONE
    export_in_record_pre_reset = False


def build_stair_train_env_cfg(
    protocol: Mapping[str, Any] | None,
    *,
    seed: int,
    device: str,
    reward_program: QualifiedStairRewardProgram,
    runner_owned_training_distribution: Mapping[str, Any] | None = None,
    unified_protocol: StairUnifiedProtocol | None = None,
    runtime_projection: StairRuntimeProjection | None = None,
) -> Any:
    """Build training with one pre-reset complete-reward authority."""

    if not isinstance(reward_program, QualifiedStairRewardProgram):
        raise TypeError("Stair train env requires a qualified reward program")
    if unified_protocol is not None:
        if protocol is not None or runner_owned_training_distribution is not None:
            raise ValueError("unified Stair training cannot consult a legacy protocol")
        cfg = _build_unified_stair_base_env_cfg(
            unified_protocol,
            mode="train",
            seed=seed,
            device=device,
            runtime_projection=runtime_projection,
        )
    else:
        if protocol is None:
            raise ValueError("legacy Stair training requires its protocol")
        cfg = build_stair_env_cfg(
            protocol,
            mode="train",
            seed=seed,
            device=device,
        )
        _apply_runner_owned_training_distribution(
            cfg,
            protocol=protocol,
            contract=runner_owned_training_distribution,
        )
    install_stair_termination_capture(cfg.terminations)
    rewards = StairCompleteRewardManagerCfg()
    rewards.complete_reward.params = {
        "qualified_program": reward_program,
        "base_protocol": (
            unified_protocol.runtime_task_contract()
            if unified_protocol is not None
            else dict(protocol)
        ),
        "runtime_projection": runtime_projection,
    }
    cfg.rewards = rewards
    cfg.recorders = None
    return cfg


def stair_training_distribution_runtime_evidence(
    cfg: Any,
    *,
    protocol: Mapping[str, Any] | None,
    contract: Mapping[str, Any] | None,
    unified_protocol: StairUnifiedProtocol | None = None,
) -> dict[str, Any]:
    if unified_protocol is not None:
        generator = cfg.scene.terrain.terrain_generator
        profile = unified_protocol.training_profile
        columns = profile.column_descriptors()
        curriculum_term = cfg.curriculum.terrain_levels
        if (
            tuple(generator.sub_terrains)
            != tuple(str(item["column_id"]) for item in columns)
            or int(generator.num_rows) != 10
            or int(generator.num_cols) != 10
            or curriculum_term is None
            or curriculum_term.func is not stair_goal_progress_levels
        ):
            raise RuntimeError("effective unified Stair training distribution drifted")
        return {
            "schema_version": (
                "true_rl_agent.task.stair_climbing."
                "training_distribution_evidence.unified.v1"
            ),
            "status": "unified_c0_c1",
            "training_profile_id": profile.profile_id,
            "training_profile_hash": profile.profile_hash,
            "terrain_rows": int(generator.num_rows),
            "terrain_cols": int(generator.num_cols),
            "column_ids": [str(item["column_id"]) for item in columns],
            "regime_allocation": dict(profile.allocation),
            "curriculum": {
                "level_update_term": _callable_identity(curriculum_term.func),
                "terrain_type_is_fixed": True,
                "goal_progress_m_by_terrain_type": list(
                    goal_progress_m_by_terrain_type(profile)
                ),
            },
            "training_exposure": profile.to_dict()["training_exposure"],
        }
    if contract is None:
        return {
            "schema_version": (
                "true_rl_agent.task.stair_climbing."
                "training_distribution_evidence.legacy_v1"
            ),
            "status": "legacy",
        }
    if protocol is None:
        raise ValueError("legacy Stair training evidence requires its protocol")
    expected = dict(
        resolved_stair_training_distribution(
            contract,
            base_protocol=protocol,
        )
    )
    terrain_generator = cfg.scene.terrain.terrain_generator
    sub_terrains = terrain_generator.sub_terrains
    if set(sub_terrains) != {"one_way_stairs"}:
        raise RuntimeError("Stair training terrain source set drifted")
    effective_height_range = list(sub_terrains["one_way_stairs"].step_height_range)
    command = cfg.commands.base_velocity
    curriculum_term = cfg.curriculum.terrain_levels
    if curriculum_term is None:
        raise RuntimeError("Stair runner-owned terrain curriculum is disabled")
    observed = {
        **expected,
        "terrain": {
            **expected["terrain"],
            "step_height_range_m": effective_height_range,
            "generator_curriculum": bool(terrain_generator.curriculum),
            "runtime_level_curriculum_enabled": curriculum_term is not None,
            "initial_max_terrain_level": int(cfg.scene.terrain.max_init_terrain_level),
            "level_update_term": _callable_identity(curriculum_term.func),
        },
        "command": {
            **expected["command"],
            "lin_vel_x_range_mps": list(command.ranges.lin_vel_x),
            "lin_vel_y_range_mps": list(command.ranges.lin_vel_y),
            "ang_vel_z_range_radps": list(command.ranges.ang_vel_z),
            "standing_env_fraction": float(command.rel_standing_envs),
            "heading_command_enabled": bool(command.heading_command),
        },
    }
    if observed != expected:
        raise RuntimeError("effective Stair runner-owned training distribution drifted")
    return observed


def _apply_runner_owned_training_distribution(
    cfg: Any,
    *,
    protocol: Mapping[str, Any],
    contract: Mapping[str, Any] | None,
) -> None:
    if contract is None:
        return
    resolved = resolved_stair_training_distribution(
        contract,
        base_protocol=protocol,
    )
    terrain = resolved["terrain"]
    terrain_generator = cfg.scene.terrain.terrain_generator
    sub_terrains = terrain_generator.sub_terrains
    if set(sub_terrains) != {"one_way_stairs"}:
        raise RuntimeError("Stair training terrain source set drifted")
    sub_terrains["one_way_stairs"].step_height_range = tuple(
        terrain["step_height_range_m"]
    )
    terrain_generator.curriculum = bool(terrain["generator_curriculum"])
    cfg.scene.terrain.max_init_terrain_level = int(terrain["initial_max_terrain_level"])
    default_curriculum = type(cfg.curriculum)()
    if default_curriculum.terrain_levels is None:
        raise RuntimeError("official locomotion terrain-level curriculum is unavailable")
    cfg.curriculum.terrain_levels = default_curriculum.terrain_levels

    command = cfg.commands.base_velocity
    resolved_command = resolved["command"]
    command.rel_standing_envs = float(resolved_command["standing_env_fraction"])
    command.ranges.lin_vel_x = tuple(resolved_command["lin_vel_x_range_mps"])
    command.ranges.lin_vel_y = tuple(resolved_command["lin_vel_y_range_mps"])
    command.ranges.ang_vel_z = tuple(resolved_command["ang_vel_z_range_radps"])
    command.heading_command = bool(resolved_command["heading_command_enabled"])
    stair_training_distribution_runtime_evidence(
        cfg,
        protocol=protocol,
        contract=contract,
    )


def _callable_identity(value: Any) -> str:
    module = getattr(value, "__module__", None)
    name = getattr(value, "__name__", None)
    if (
        not isinstance(module, str)
        or not module
        or not isinstance(name, str)
        or not name
    ):
        raise RuntimeError("Stair training curriculum callable identity is invalid")
    return f"{module}.{name}"


def build_stair_eval_env_cfg(
    protocol: Mapping[str, Any] | None,
    *,
    seed: int,
    device: str,
    evaluation_profile: StairEvaluationProfile | None = None,
    ordered_evaluation_suite: StairOrderedEvaluationSuite | None = None,
    evaluation_split: str | None = None,
    runtime_projection: StairRuntimeProjection | None = None,
    unified_protocol: StairUnifiedProtocol | None = None,
) -> Any:
    """Build production evaluation with pre-reset task evidence."""

    if evaluation_profile is not None and ordered_evaluation_suite is not None:
        raise ValueError("Stair eval cannot combine grid and ordered suites")
    if ordered_evaluation_suite is None and evaluation_split is not None:
        raise ValueError("Stair eval split requires an ordered suite")
    if (
        ordered_evaluation_suite is not None
        and evaluation_split not in STAIR_ORDERED_RUNNABLE_COHORTS
    ):
        raise ValueError("Stair ordered eval split is invalid")
    if unified_protocol is not None:
        if protocol is not None:
            raise ValueError(
                "unified Stair evaluation cannot consult a legacy protocol"
            )
        cfg = _build_unified_stair_base_env_cfg(
            unified_protocol,
            mode="eval",
            seed=seed,
            device=device,
            runtime_projection=runtime_projection,
        )
    else:
        if protocol is None:
            raise ValueError("legacy Stair evaluation requires its protocol")
        cfg = build_stair_env_cfg(protocol, mode="eval", seed=seed, device=device)
    if evaluation_profile is not None:
        cfg.scene.terrain.terrain_generator = build_profile_eval_terrain_generator(
            protocol,
            seed=seed,
            profile=evaluation_profile,
        )
        cfg.scene.num_envs = evaluation_profile.evaluation_protocol.eval_num_envs
        cfg.episode_length_s = evaluation_profile.max_episode_horizon_s
        cfg.commands.base_velocity.resampling_time_range = (
            cfg.episode_length_s,
            cfg.episode_length_s,
        )
    elif ordered_evaluation_suite is not None:
        assert evaluation_split is not None
        runtime_cells = ordered_evaluation_suite.runtime_cells(evaluation_split)
        cfg.scene.terrain.terrain_generator = build_ordered_eval_terrain_generator(
            seed=seed,
            cells=runtime_cells,
        )
        cfg.scene.num_envs = len(
            ordered_evaluation_suite.private_cell_assignment(evaluation_split)
        )
        cfg.episode_length_s = max(cell.episode_horizon_s for cell in runtime_cells)
        cfg.commands.base_velocity.resampling_time_range = (
            cfg.episode_length_s,
            cfg.episode_length_s,
        )
    install_stair_termination_capture(cfg.terminations)
    cfg.recorders = StairEvalRecorderManagerCfg()
    cfg.recorders.transition.runtime_projection = runtime_projection
    return cfg


def _build_unified_stair_base_env_cfg(
    protocol: StairUnifiedProtocol,
    *,
    mode: str,
    seed: int,
    device: str,
    runtime_projection: StairRuntimeProjection | None,
) -> Any:
    """Apply one direct task/runtime contract over the robot's env adapter."""

    if mode not in {"train", "eval"}:
        raise ValueError("unified Stair env mode is invalid")
    if not isinstance(runtime_projection, StairRuntimeProjection):
        raise TypeError("unified Stair env requires a resolved runtime projection")
    cfg = isaaclab_locomotion_runtime_spec(
        runtime_projection.runtime_adapter_id
    ).instantiate_environment_cfg()
    cfg.seed = int(seed)
    cfg.sim.device = str(device)
    cfg.scene.num_envs = (
        int(protocol.training["num_envs"])
        if mode == "train"
        else protocol.evaluation_suite.evaluation_protocol.eval_num_envs
    )
    cfg.episode_length_s = (
        float(protocol.training["episode_horizon_s"])
        if mode == "train"
        else max(
            cell.episode_horizon_s
            for cell in protocol.evaluation_suite.runtime_cells("public_accept")
        )
    )
    cfg.scene.terrain.debug_vis = False
    if mode == "train":
        cfg.scene.terrain.terrain_generator = build_unified_train_terrain_generator(
            protocol.training_profile,
            seed=seed,
        )
        cfg.scene.terrain.max_init_terrain_level = 9

    controls = protocol.runtime_controls
    if controls["external_force_events_enabled"] or controls["push_events_enabled"]:
        raise ValueError("unified Stair runtime events must remain disabled")
    cfg.events.base_external_force_torque = None
    cfg.events.push_robot = None
    spawn = controls["spawn"]
    velocities = controls["reset_velocity_ranges"]
    cfg.events.reset_base.params = {
        "pose_range": {
            "x": tuple(float(value) for value in spawn["x_range_m_from_origin"]),
            "y": tuple(float(value) for value in spawn["y_range_m_from_origin"]),
            "yaw": tuple(float(value) for value in spawn["yaw_range_rad"]),
        },
        "velocity_range": {
            axis: tuple(float(value) for value in values)
            for axis, values in velocities.items()
        },
    }
    command = cfg.commands.base_velocity
    command.resampling_time_range = (cfg.episode_length_s, cfg.episode_length_s)
    if mode == "train":
        declared = protocol.training["command"]
        forward = tuple(float(value) for value in declared["forward_mps_range"])
        lateral = (float(declared["lateral_command_mps"]),) * 2
        yaw = (float(declared["yaw_command_radps"]),) * 2
        command.rel_standing_envs = float(declared["standing_env_fraction"])
        command.heading_command = bool(declared["heading_command_enabled"])
    else:
        declared = controls["evaluation_command"]
        forward = (float(declared["forward_mps"]),) * 2
        lateral = (float(declared["lateral_mps"]),) * 2
        yaw = (float(declared["yaw_radps"]),) * 2
        command.rel_standing_envs = 0.0
        command.heading_command = False
    command.rel_heading_envs = 0.0
    command.debug_vis = False
    command.ranges.lin_vel_x = forward
    command.ranges.lin_vel_y = lateral
    command.ranges.ang_vel_z = yaw
    if hasattr(command, "limit_ranges"):
        command.limit_ranges.lin_vel_x = forward
        command.limit_ranges.lin_vel_y = lateral
        command.limit_ranges.ang_vel_z = yaw
    cfg.observations.policy.enable_corruption = bool(
        controls[
            "train_observation_corruption_enabled"
            if mode == "train"
            else "eval_observation_corruption_enabled"
        ]
    )

    base_contact = cfg.terminations.base_contact
    base_contact.params["sensor_cfg"].body_names = list(
        runtime_projection.forbidden_contact_body_names
    )
    base_contact.params["threshold"] = float(
        protocol.episode_outcomes["signals"]["forbidden_contact"]["force_threshold_n"]
    )
    if mode == "train":
        term = cfg.curriculum.terrain_levels
        if term is None:
            raise RuntimeError("robot adapter has no terrain-level curriculum term")
        term.func = stair_goal_progress_levels
        term.params = {
            **dict(term.params),
            "asset_cfg": SceneEntityCfg("robot"),
            "goal_progress_m_by_terrain_type": goal_progress_m_by_terrain_type(
                protocol.training_profile
            ),
            "move_up_threshold": float(
                protocol.training_profile.curriculum["move_up_threshold"]
            ),
            "move_down_threshold": float(
                protocol.training_profile.curriculum["move_down_threshold"]
            ),
        }
    else:
        cfg.curriculum.terrain_levels = None
    return cfg


def register_production_stair_envs() -> None:
    """Explicitly register production IDs without import-time activation."""

    for gym_id in (STAIR_TRAIN_GYM_ID, STAIR_EVAL_GYM_ID):
        if gym_id in gym.registry:
            continue
        gym.register(
            id=gym_id,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
        )


__all__ = [
    "StairEvalRecorderManagerCfg",
    "StairEvalTransitionRecorder",
    "build_stair_eval_env_cfg",
    "build_stair_train_env_cfg",
    "register_production_stair_envs",
]
