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
)
from isaaclab.utils import configclass

from true_rl_agent.tasks.stair_climbing.evaluation_profile import (
    StairEvaluationProfile,
)
from true_rl_agent.tasks.stair_climbing.isaaclab_env_cfg import (
    StairG1RoughEnvCfg,
    build_stair_env_cfg,
)
from true_rl_agent.tasks.stair_climbing.isaaclab_evaluation_terrain import (
    build_profile_eval_terrain_generator,
)
from true_rl_agent.tasks.stair_climbing.isaaclab_reward_runtime import (
    StairCompleteRewardManagerCfg,
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
from true_rl_agent.tasks.stair_climbing.termination_capture import (
    capture_current_termination_masks,
    install_stair_termination_capture,
)
from true_rl_agent.tasks.stair_climbing.training_distribution import (
    resolved_stair_training_distribution,
)

_FOOT_BODY_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")
_BASE_CONTACT_BODY_NAMES = ("torso_link",)


class StairEvalTransitionRecorder(RecorderTerm):
    """Capture runner-owned task state before done environments are reset."""

    def __init__(self, cfg: RecorderTermCfg, env: Any) -> None:
        super().__init__(cfg, env)
        robot = env.scene["robot"]
        contact_sensor = env.scene["contact_forces"]
        self._robot_foot_ids = _resolve_body_ids(robot, _FOOT_BODY_NAMES, "robot")
        self._contact_foot_ids = _resolve_body_ids(
            contact_sensor,
            _FOOT_BODY_NAMES,
            "contact sensor feet",
        )
        self._contact_base_ids = _resolve_body_ids(
            contact_sensor,
            _BASE_CONTACT_BODY_NAMES,
            "contact sensor base",
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
        contact_data = scene["contact_forces"].data
        contact_forces = contact_data.net_forces_w
        if contact_forces is None:
            raise RuntimeError("Stair eval recorder requires current contact forces")

        self._env.extras[STAIR_EVAL_TRANSITION_EXTRAS_KEY] = {
            "schema_version": STAIR_TRANSITION_RECORDER_SCHEMA_VERSION,
            "root_pos_w": robot_data.root_pos_w.detach().clone(),
            "root_quat_w": robot_data.root_quat_w.detach().clone(),
            "joint_pos": robot_data.joint_pos.detach().clone(),
            "foot_pos_w": robot_data.body_pos_w[:, self._robot_foot_ids, :]
            .detach()
            .clone(),
            "foot_lin_vel_w": robot_data.body_lin_vel_w[:, self._robot_foot_ids, :]
            .detach()
            .clone(),
            "foot_contact_forces_w": contact_forces[:, self._contact_foot_ids, :]
            .detach()
            .clone(),
            "base_contact_forces_w": contact_forces[:, self._contact_base_ids, :]
            .detach()
            .clone(),
            "episode_length": self._env.episode_length_buf.detach().clone(),
            "terrain_env_origins_w": self._terrain_env_origins_w,
            "terrain_types": self._terrain_types,
            "termination_term_order": termination.term_order,
            "termination_masks": dict(termination.term_masks),
            "done_mask": termination.done_mask,
        }
        return None, None


@configclass
class StairEvalTransitionRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = StairEvalTransitionRecorder


@configclass
class StairEvalRecorderManagerCfg(RecorderManagerBaseCfg):
    transition = StairEvalTransitionRecorderCfg()
    dataset_export_mode = DatasetExportMode.EXPORT_NONE
    export_in_record_pre_reset = False


def build_stair_train_env_cfg(
    protocol: Mapping[str, Any],
    *,
    seed: int,
    device: str,
    reward_program: QualifiedStairRewardProgram,
    runner_owned_training_distribution: Mapping[str, Any] | None = None,
) -> StairG1RoughEnvCfg:
    """Build training with one pre-reset complete-reward authority."""

    if not isinstance(reward_program, QualifiedStairRewardProgram):
        raise TypeError("Stair train env requires a qualified reward program")
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
        "base_protocol": dict(protocol),
    }
    cfg.rewards = rewards
    cfg.recorders = None
    return cfg


def stair_training_distribution_runtime_evidence(
    cfg: StairG1RoughEnvCfg,
    *,
    protocol: Mapping[str, Any],
    contract: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if contract is None:
        return {
            "schema_version": (
                "true_rl_agent.task.stair_climbing."
                "training_distribution_evidence.legacy_v1"
            ),
            "status": "legacy",
        }
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
    effective_height_range = list(
        sub_terrains["one_way_stairs"].step_height_range
    )
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
            "initial_max_terrain_level": int(
                cfg.scene.terrain.max_init_terrain_level
            ),
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
    cfg: StairG1RoughEnvCfg,
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
    cfg.scene.terrain.max_init_terrain_level = int(
        terrain["initial_max_terrain_level"]
    )
    default_curriculum = type(cfg.curriculum)()
    if default_curriculum.terrain_levels is None:
        raise RuntimeError("official G1 terrain-level curriculum is unavailable")
    cfg.curriculum.terrain_levels = default_curriculum.terrain_levels

    command = cfg.commands.base_velocity
    resolved_command = resolved["command"]
    command.rel_standing_envs = float(
        resolved_command["standing_env_fraction"]
    )
    command.ranges.lin_vel_x = tuple(resolved_command["lin_vel_x_range_mps"])
    command.ranges.lin_vel_y = tuple(resolved_command["lin_vel_y_range_mps"])
    command.ranges.ang_vel_z = tuple(
        resolved_command["ang_vel_z_range_radps"]
    )
    command.heading_command = bool(
        resolved_command["heading_command_enabled"]
    )
    stair_training_distribution_runtime_evidence(
        cfg,
        protocol=protocol,
        contract=contract,
    )


def _callable_identity(value: Any) -> str:
    module = getattr(value, "__module__", None)
    name = getattr(value, "__name__", None)
    if not isinstance(module, str) or not module or not isinstance(name, str) or not name:
        raise RuntimeError("Stair training curriculum callable identity is invalid")
    return f"{module}.{name}"


def build_stair_eval_env_cfg(
    protocol: Mapping[str, Any],
    *,
    seed: int,
    device: str,
    evaluation_profile: StairEvaluationProfile | None = None,
) -> StairG1RoughEnvCfg:
    """Build production evaluation with pre-reset task evidence."""

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
    install_stair_termination_capture(cfg.terminations)
    cfg.recorders = StairEvalRecorderManagerCfg()
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


def _resolve_body_ids(asset: Any, names: tuple[str, ...], context: str) -> list[int]:
    body_ids, resolved_names = asset.find_bodies(list(names), preserve_order=True)
    if tuple(resolved_names) != names or len(body_ids) != len(names):
        raise RuntimeError(f"Stair {context} body schema drifted")
    return [int(body_id) for body_id in body_ids]


__all__ = [
    "StairEvalRecorderManagerCfg",
    "StairEvalTransitionRecorder",
    "build_stair_eval_env_cfg",
    "build_stair_train_env_cfg",
    "register_production_stair_envs",
]
