"""Isaac Lab G1 rough-locomotion overlay for the Phase 0 stair probe."""

from __future__ import annotations

from typing import Any, Literal, Mapping

import gymnasium as gym
from isaaclab.managers import (
    DatasetExportMode,
    RecorderManagerBaseCfg,
    RecorderTerm,
    RecorderTermCfg,
)
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.rough_env_cfg import (
    G1RoughEnvCfg,
)

from true_rl_agent.tasks.stair_climbing.isaaclab_terrain import (
    build_eval_terrain_generator,
    build_train_terrain_generator,
)
from true_rl_agent.tasks.stair_climbing.phase0_contract import (
    PHASE0_STAIR_GYM_ID,
    PHASE0_TRANSITION_EXTRAS_KEY,
)


class StairPhase0TransitionRecorder(RecorderTerm):
    """Publish the post-transition state before Isaac Lab auto-resets done envs."""

    def record_post_step(self) -> tuple[None, None]:
        termination_manager = self._env.termination_manager
        timeout_terms = [
            name
            for name in termination_manager.active_terms
            if termination_manager.get_term_cfg(name).time_out
        ]
        terminated_terms = [
            name
            for name in termination_manager.active_terms
            if not termination_manager.get_term_cfg(name).time_out
        ]
        if len(timeout_terms) != 1 or len(terminated_terms) != 1:
            raise RuntimeError(
                "Phase 0 recorder requires exactly one timeout term and "
                "one non-timeout termination term"
            )

        # With one term in each Gym termination class, these union buffers are
        # exact per-term masks and preserve simultaneous timeout/contact events.
        self._env.extras[PHASE0_TRANSITION_EXTRAS_KEY] = {
            "root_pos_w": self._env.scene["robot"].data.root_pos_w.detach().clone(),
            "episode_length": self._env.episode_length_buf.detach().clone(),
            "termination_masks": {
                timeout_terms[0]: self._env.reset_time_outs.detach().clone(),
                terminated_terms[0]: self._env.reset_terminated.detach().clone(),
            },
        }
        return None, None


@configclass
class StairPhase0TransitionRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = StairPhase0TransitionRecorder


@configclass
class StairPhase0RecorderManagerCfg(RecorderManagerBaseCfg):
    terminal_transition = StairPhase0TransitionRecorderCfg()
    dataset_export_mode = DatasetExportMode.EXPORT_NONE
    export_in_record_pre_reset = False


@configclass
class StairG1RoughEnvCfg(G1RoughEnvCfg):
    """G1 rough config retaining the official policy observation/action contract."""

    def __post_init__(self) -> None:
        super().__post_init__()


def build_stair_env_cfg(
    protocol: Mapping[str, Any],
    *,
    mode: Literal["train", "eval"],
    seed: int,
    device: str,
) -> StairG1RoughEnvCfg:
    """Build an environment solely from the frozen task protocol."""

    cfg = StairG1RoughEnvCfg()
    cfg.seed = int(seed)
    cfg.sim.device = str(device)
    cfg.scene.terrain.terrain_generator = (
        build_eval_terrain_generator(protocol, seed=seed)
        if mode == "eval"
        else build_train_terrain_generator(protocol, seed=seed)
    )
    cfg.scene.terrain.max_init_terrain_level = (
        0
        if mode == "eval"
        else int(protocol["terrain"]["training_distribution"]["terrain_rows"]) - 1
    )
    cfg.scene.terrain.debug_vis = False

    run_cfg = protocol["evaluation"] if mode == "eval" else protocol["training"]
    cfg.scene.num_envs = (
        int(protocol["evaluation"]["protocol"]["eval_num_envs"])
        if mode == "eval"
        else int(protocol["training"]["num_envs"])
    )
    cfg.episode_length_s = float(run_cfg["episode_length_s"])

    cfg.curriculum.terrain_levels = None
    runtime_controls = protocol["runtime_controls"]
    event_controls = runtime_controls["events"]
    if any(bool(value) for value in event_controls.values()):
        raise ValueError("StairProtocolV1 does not support Phase 0 runtime events")
    cfg.events.base_external_force_torque = None
    cfg.events.push_robot = None
    cfg.events.reset_base.params = {
        "pose_range": {
            "x": tuple(float(value) for value in protocol["terrain"]["spawn"]["x_range_m_from_origin"]),
            "y": tuple(float(value) for value in protocol["terrain"]["spawn"]["y_range_m_from_origin"]),
            "yaw": tuple(float(value) for value in protocol["terrain"]["spawn"]["yaw_range_rad"]),
        },
        "velocity_range": {
            axis: tuple(float(value) for value in raw_range)
            for axis, raw_range in runtime_controls["reset_velocity_ranges"].items()
        },
    }

    command = cfg.commands.base_velocity
    command_controls = runtime_controls["command"]
    if command_controls["resampling"] != "episode_length":
        raise ValueError("unsupported stair command resampling mode")
    command.resampling_time_range = (cfg.episode_length_s, cfg.episode_length_s)
    command.rel_standing_envs = float(command_controls["standing_env_fraction"])
    command.rel_heading_envs = float(command_controls["heading_env_fraction"])
    command.heading_command = bool(command_controls["heading_command_enabled"])
    command.debug_vis = False
    forward = float(run_cfg["forward_command_mps"])
    lateral = float(run_cfg["lateral_command_mps"])
    yaw = float(run_cfg["yaw_command_radps"])
    command.ranges.lin_vel_x = (forward, forward)
    command.ranges.lin_vel_y = (lateral, lateral)
    command.ranges.ang_vel_z = (yaw, yaw)
    cfg.observations.policy.enable_corruption = bool(
        run_cfg["observation_corruption_enabled"]
    )
    return cfg


def build_phase0_stair_env_cfg(
    protocol: Mapping[str, Any],
    *,
    seed: int,
    device: str,
) -> StairG1RoughEnvCfg:
    """Build the probe-only eval config with terminal transition evidence."""

    cfg = build_stair_env_cfg(
        protocol,
        mode="eval",
        seed=seed,
        device=device,
    )
    cfg.recorders = StairPhase0RecorderManagerCfg()
    return cfg


def register_phase0_stair_env() -> None:
    """Explicitly register the probe-only Gym ID without package import side effects."""

    if PHASE0_STAIR_GYM_ID in gym.registry:
        return
    gym.register(
        id=PHASE0_STAIR_GYM_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
    )


__all__ = [
    "PHASE0_STAIR_GYM_ID",
    "PHASE0_TRANSITION_EXTRAS_KEY",
    "StairG1RoughEnvCfg",
    "StairPhase0RecorderManagerCfg",
    "StairPhase0TransitionRecorder",
    "build_phase0_stair_env_cfg",
    "build_stair_env_cfg",
    "register_phase0_stair_env",
]
