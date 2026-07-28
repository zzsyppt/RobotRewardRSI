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

from true_rl_agent.tasks.stair_climbing.isaaclab_env_cfg import (
    StairG1RoughEnvCfg,
    build_stair_env_cfg,
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

    def record_post_step(self) -> tuple[None, None]:
        termination = capture_current_termination_masks(self._env)
        scene = self._env.scene
        robot_data = scene["robot"].data
        contact_data = scene["contact_forces"].data
        terrain = scene.terrain
        contact_forces = contact_data.net_forces_w
        if contact_forces is None:
            raise RuntimeError("Stair eval recorder requires current contact forces")
        terrain_types = getattr(terrain, "terrain_types", None)
        if terrain_types is None:
            raise RuntimeError("Stair eval recorder requires terrain type identity")

        self._env.extras[STAIR_EVAL_TRANSITION_EXTRAS_KEY] = {
            "schema_version": STAIR_TRANSITION_RECORDER_SCHEMA_VERSION,
            "root_pos_w": robot_data.root_pos_w.detach().clone(),
            "root_quat_w": robot_data.root_quat_w.detach().clone(),
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
            "terrain_env_origins_w": terrain.env_origins.detach().clone(),
            "terrain_types": terrain_types.detach().clone(),
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
) -> StairG1RoughEnvCfg:
    """Build training with one pre-reset complete-reward authority."""

    if not isinstance(reward_program, QualifiedStairRewardProgram):
        raise TypeError("Stair train env requires a qualified reward program")
    cfg = build_stair_env_cfg(protocol, mode="train", seed=seed, device=device)
    install_stair_termination_capture(cfg.terminations)
    rewards = StairCompleteRewardManagerCfg()
    rewards.complete_reward.params = {
        "qualified_program": reward_program,
        "base_protocol": dict(protocol),
    }
    cfg.rewards = rewards
    cfg.recorders = None
    return cfg


def build_stair_eval_env_cfg(
    protocol: Mapping[str, Any],
    *,
    seed: int,
    device: str,
) -> StairG1RoughEnvCfg:
    """Build production evaluation with pre-reset task evidence."""

    cfg = build_stair_env_cfg(protocol, mode="eval", seed=seed, device=device)
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
