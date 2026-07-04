"""IsaacLab config overlay for the project-local push recovery task."""

from __future__ import annotations

import argparse
from typing import Any


def as_pair(value: Any) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"expected two-element range, got {value!r}")
    return (float(value[0]), float(value[1]))


def apply_push_recovery_overlay(env_cfg: Any, config: dict[str, Any], args: argparse.Namespace) -> None:
    """Mutate the Unitree velocity task cfg into standing push recovery."""

    env_cfg.seed = args.seed
    env_cfg.scene.num_envs = args.num_envs
    if getattr(args, "device", None) is not None:
        env_cfg.sim.device = args.device

    env_settings = config["environment"]
    env_cfg.episode_length_s = float(env_settings["episode_length_s"])

    terrain_generator = getattr(env_cfg.scene.terrain, "terrain_generator", None)
    if terrain_generator is not None:
        if "terrain_rows" in env_settings:
            terrain_generator.num_rows = int(env_settings["terrain_rows"])
        if "terrain_cols" in env_settings:
            terrain_generator.num_cols = int(env_settings["terrain_cols"])
        if bool(env_settings.get("disable_curriculum", False)):
            terrain_generator.curriculum = False

    if bool(env_settings.get("disable_curriculum", False)) and hasattr(env_cfg, "curriculum"):
        for name in ("terrain_levels", "lin_vel_cmd_levels"):
            if hasattr(env_cfg.curriculum, name):
                setattr(env_cfg.curriculum, name, None)

    command = config["standing_command"]
    base_velocity = env_cfg.commands.base_velocity
    base_velocity.resampling_time_range = as_pair(command["resampling_time_range_s"])
    base_velocity.rel_standing_envs = float(command["rel_standing_envs"])
    base_velocity.rel_heading_envs = float(command["rel_heading_envs"])
    base_velocity.heading_command = bool(command["heading_command"])
    base_velocity.debug_vis = False
    base_velocity.ranges.lin_vel_x = as_pair(command["lin_vel_x_mps"])
    base_velocity.ranges.lin_vel_y = as_pair(command["lin_vel_y_mps"])
    base_velocity.ranges.ang_vel_z = as_pair(command["ang_vel_z_radps"])
    if hasattr(base_velocity, "limit_ranges"):
        base_velocity.limit_ranges.lin_vel_x = as_pair(command["lin_vel_x_mps"])
        base_velocity.limit_ranges.lin_vel_y = as_pair(command["lin_vel_y_mps"])
        base_velocity.limit_ranges.ang_vel_z = as_pair(command["ang_vel_z_radps"])

    if bool(env_settings.get("disable_observation_corruption", False)):
        for group_name in ("policy", "critic"):
            group = getattr(env_cfg.observations, group_name, None)
            if group is not None and hasattr(group, "enable_corruption"):
                group.enable_corruption = False

    push = config["push"]
    push_robot = env_cfg.events.push_robot
    push_robot.interval_range_s = as_pair(push["interval_range_s"])
    push_robot.params["velocity_range"]["x"] = as_pair(push["velocity_range_mps"]["x"])
    push_robot.params["velocity_range"]["y"] = as_pair(push["velocity_range_mps"]["y"])


def apply_eval_overrides(env_cfg: Any, config: dict[str, Any]) -> None:
    evaluation_cfg = config.get("evaluation", {})
    if bool(evaluation_cfg.get("disable_vendor_fall_terminations", False)):
        for name in ("base_height", "bad_orientation"):
            if hasattr(env_cfg.terminations, name):
                setattr(env_cfg.terminations, name, None)
