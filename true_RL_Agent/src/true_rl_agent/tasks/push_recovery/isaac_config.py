"""Project-local runtime config helpers for push recovery IsaacLab jobs."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

DEFAULT_PUSH_RECOVERY_RUNTIME_CONFIG: dict[str, Any] = {
    "schema_version": "true_rl_agent.task.push_recovery.runtime.v1",
    "task": {
        "gym_id": "Unitree-G1-29dof-Velocity",
        "robot": "Unitree G1 29DoF",
        "vendor_env": "unitree_rl_lab.tasks.locomotion.robots.g1.29dof.velocity_env_cfg",
        "implementation_strategy": "true_rl_agent_project_local_config_overlay",
    },
    "environment": {
        "episode_length_s": 4.0,
        "terrain_rows": 2,
        "terrain_cols": 2,
        "disable_curriculum": True,
        "disable_observation_corruption": True,
    },
    "evaluation": {
        "disable_vendor_fall_terminations": True,
    },
    "standing_command": {
        "command_name": "base_velocity",
        "resampling_time_range_s": [10.0, 10.0],
        "lin_vel_x_mps": [0.0, 0.0],
        "lin_vel_y_mps": [0.0, 0.0],
        "ang_vel_z_radps": [0.0, 0.0],
        "rel_standing_envs": 1.0,
        "rel_heading_envs": 0.0,
        "heading_command": False,
    },
    "push": {
        "mechanism": "base_velocity_kick",
        "isaac_event": "push_by_setting_velocity",
        "body": "torso_link",
        "interval_range_s": [1.0, 1.0],
        "velocity_range_mps": {
            "x": [-0.6, 0.6],
            "y": [-0.6, 0.6],
        },
        "evidence": {
            "pre_window_s": 0.75,
            "post_window_s": 1.0,
            "single_step_window_s": [0.9, 1.1],
            "min_observed_delta_mps": 0.12,
            "min_env_fraction_with_delta": 0.20,
            "min_single_step_delta_mps": 0.12,
            "min_env_fraction_with_single_step_delta": 0.20,
        },
    },
    "success_thresholds": {
        "min_base_height_m": 0.45,
        "max_abs_roll_rad": 0.65,
        "max_abs_pitch_rad": 0.65,
        "max_final_xy_drift_m": 0.80,
    },
    "behavior_gate_thresholds": {
        "max_final_planar_speed_mps": 5.0,
    },
    "metrics": {
        "max_per_episode_records": 24,
    },
    "profiles": {
        "strong_push": {
            "push": {
                "interval_range_s": [1.0, 1.0],
                "velocity_range_mps": {"x": [-0.6, 0.6], "y": [-0.6, 0.6]},
            }
        },
        "small_push": {
            "push": {
                "interval_range_s": [1.0, 1.0],
                "velocity_range_mps": {"x": [-0.25, 0.25], "y": [-0.25, 0.25]},
            }
        },
        "no_push": {
            "push": {
                "interval_range_s": [1.0, 1.0],
                "velocity_range_mps": {"x": [0.0, 0.0], "y": [0.0, 0.0]},
            }
        },
    },
    "policy_defaults": {
        "random_action_scale": 1.0,
        "zero_action_scale": 0.0,
    },
}


def load_runtime_config(path: str | Path | None) -> dict[str, Any]:
    """Load project-local YAML and merge it over the built-in runtime defaults."""

    config = copy.deepcopy(DEFAULT_PUSH_RECOVERY_RUNTIME_CONFIG)
    if path is None:
        return config
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"push recovery runtime config does not exist: {config_path}")
    data = _load_yaml_mapping(config_path)
    return deep_merge(config, data)


def config_for_profile(config: dict[str, Any], profile_name: str | None) -> dict[str, Any]:
    if not profile_name:
        return copy.deepcopy(config)
    profiles = config.get("profiles", {})
    if profile_name not in profiles:
        available = ", ".join(sorted(profiles)) or "<none>"
        raise KeyError(f"unknown push recovery profile: {profile_name}; available: {available}")
    merged = deep_merge(config, profiles[profile_name])
    merged["active_profile"] = profile_name
    return merged


def public_runtime_dynamics(config: Mapping[str, Any]) -> dict[str, Any]:
    """Project the runner-applied public task protocol and its dynamics."""

    task = _required_mapping(config, "task")
    environment = _required_mapping(config, "environment")
    evaluation = _required_mapping(config, "evaluation")
    standing = _required_mapping(config, "standing_command")
    push = _required_mapping(config, "push")
    push_evidence = _required_mapping(push, "evidence")
    success_thresholds = _required_mapping(config, "success_thresholds")
    behavior_gate_thresholds = _required_mapping(config, "behavior_gate_thresholds")
    metrics = _required_mapping(config, "metrics")
    policy_defaults = _required_mapping(config, "policy_defaults")
    velocity_ranges = _required_mapping(push, "velocity_range_mps")
    active_profile = _non_empty_string(
        config.get("active_profile", "default"),
        "active_profile",
    )
    episode_length_s = _positive_finite_number(
        environment.get("episode_length_s"),
        "environment.episode_length_s",
    )
    terrain_rows = _positive_integer(
        environment.get("terrain_rows"),
        "environment.terrain_rows",
    )
    terrain_cols = _positive_integer(
        environment.get("terrain_cols"),
        "environment.terrain_cols",
    )
    disable_curriculum = _boolean(
        environment.get("disable_curriculum"),
        "environment.disable_curriculum",
    )
    disable_observation_corruption = _boolean(
        environment.get("disable_observation_corruption"),
        "environment.disable_observation_corruption",
    )
    disable_vendor_fall_terminations = _boolean(
        evaluation.get("disable_vendor_fall_terminations"),
        "evaluation.disable_vendor_fall_terminations",
    )
    resampling_time_range = _finite_pair(
        standing.get("resampling_time_range_s"),
        "standing_command.resampling_time_range_s",
    )
    command_x = _finite_pair(standing.get("lin_vel_x_mps"), "standing_command.lin_vel_x_mps")
    command_y = _finite_pair(standing.get("lin_vel_y_mps"), "standing_command.lin_vel_y_mps")
    command_yaw = _finite_pair(standing.get("ang_vel_z_radps"), "standing_command.ang_vel_z_radps")
    push_x = _finite_pair(velocity_ranges.get("x"), "push.velocity_range_mps.x")
    push_y = _finite_pair(velocity_ranges.get("y"), "push.velocity_range_mps.y")
    interval = _finite_pair(push.get("interval_range_s"), "push.interval_range_s")
    rel_standing_envs = _finite_number(
        standing.get("rel_standing_envs"),
        "standing_command.rel_standing_envs",
    )
    rel_heading_envs = _finite_number(
        standing.get("rel_heading_envs"),
        "standing_command.rel_heading_envs",
    )
    heading_command = _boolean(
        standing.get("heading_command"),
        "standing_command.heading_command",
    )
    push_evidence_projection = {
        "pre_window_s": _non_negative_finite_number(
            push_evidence.get("pre_window_s"),
            "push.evidence.pre_window_s",
        ),
        "post_window_s": _non_negative_finite_number(
            push_evidence.get("post_window_s"),
            "push.evidence.post_window_s",
        ),
        "single_step_window_s": list(
            _finite_pair(
                push_evidence.get("single_step_window_s"),
                "push.evidence.single_step_window_s",
            )
        ),
        "min_observed_delta_mps": _non_negative_finite_number(
            push_evidence.get("min_observed_delta_mps"),
            "push.evidence.min_observed_delta_mps",
        ),
        "min_env_fraction_with_delta": _unit_interval(
            push_evidence.get("min_env_fraction_with_delta"),
            "push.evidence.min_env_fraction_with_delta",
        ),
        "min_single_step_delta_mps": _non_negative_finite_number(
            push_evidence.get("min_single_step_delta_mps"),
            "push.evidence.min_single_step_delta_mps",
        ),
        "min_env_fraction_with_single_step_delta": _unit_interval(
            push_evidence.get("min_env_fraction_with_single_step_delta"),
            "push.evidence.min_env_fraction_with_single_step_delta",
        ),
    }
    projection: dict[str, Any] = {
        "schema_version": "true_rl_agent.task.push_recovery.public_runtime_dynamics.v2",
        "active_profile": active_profile,
        "runner_owned": True,
        "candidate_can_change": False,
        "task": {
            key: _non_empty_string(task.get(key), f"task.{key}")
            for key in (
                "gym_id",
                "robot",
                "vendor_env",
                "implementation_strategy",
            )
        },
        "environment": {
            "episode_length_s": episode_length_s,
            "terrain_rows": terrain_rows,
            "terrain_cols": terrain_cols,
        },
        "standing_command": {
            "resampling_time_range_s": list(resampling_time_range),
            "lin_vel_x_mps": list(command_x),
            "lin_vel_y_mps": list(command_y),
            "ang_vel_z_radps": list(command_yaw),
            "rel_standing_envs": rel_standing_envs,
            "rel_heading_envs": rel_heading_envs,
            "heading_command": heading_command,
            "commanded_locomotion_is_zero": (
                rel_standing_envs == 1.0
                and command_x == (0.0, 0.0)
                and command_y == (0.0, 0.0)
                and command_yaw == (0.0, 0.0)
            ),
        },
        "push": {
            "mechanism": _non_empty_string(
                push.get("mechanism"),
                "push.mechanism",
            ),
            "isaac_event": _non_empty_string(
                push.get("isaac_event"),
                "push.isaac_event",
            ),
            "body": _non_empty_string(push.get("body"), "push.body"),
            "coordinate_frame": "world_xy_velocity",
            "interval_range_s": list(interval),
            "velocity_range_mps": {
                "x": list(push_x),
                "y": list(push_y),
            },
            "active_world_axes": [
                axis
                for axis, values in (("x", push_x), ("y", push_y))
                if values != (0.0, 0.0)
            ],
            "xy_ranges_equal": push_x == push_y,
            "evidence": push_evidence_projection,
        },
        "success_thresholds": {
            key: _finite_number(success_thresholds.get(key), f"success_thresholds.{key}")
            for key in (
                "min_base_height_m",
                "max_abs_roll_rad",
                "max_abs_pitch_rad",
                "max_final_xy_drift_m",
            )
        },
        "behavior_gate_thresholds": {
            "max_final_planar_speed_mps": _finite_number(
                behavior_gate_thresholds.get("max_final_planar_speed_mps"),
                "behavior_gate_thresholds.max_final_planar_speed_mps",
            )
        },
        "metrics": {
            "max_per_episode_records": _positive_integer(
                metrics.get("max_per_episode_records"),
                "metrics.max_per_episode_records",
            )
        },
        "policy_defaults": {
            "random_action_scale": _non_negative_finite_number(
                policy_defaults.get("random_action_scale"),
                "policy_defaults.random_action_scale",
            ),
            "zero_action_scale": _non_negative_finite_number(
                policy_defaults.get("zero_action_scale"),
                "policy_defaults.zero_action_scale",
            ),
        },
    }
    binding = {
        **projection,
        "_runner_controls": {
            "disable_curriculum": disable_curriculum,
            "disable_observation_corruption": disable_observation_corruption,
            "disable_vendor_fall_terminations": (
                disable_vendor_fall_terminations
            ),
        },
    }
    digest = hashlib.sha256(stable_json_dumps(binding).encode("utf-8")).hexdigest()
    projection["runtime_dynamics_hash"] = f"sha256:{digest}"
    return projection


def load_public_runtime_dynamics(
    path: str | Path | None,
    profile_name: str | None,
) -> dict[str, Any]:
    """Load, profile, and project the exact dynamics consumed by the runner."""

    return public_runtime_dynamics(config_for_profile(load_runtime_config(path), profile_name))


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _required_mapping(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = data.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"push recovery runtime config field must be an object: {key}")
    return value


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"push recovery runtime config field must be numeric: {name}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"push recovery runtime config field must be finite: {name}")
    return result


def _finite_pair(value: Any, name: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"push recovery runtime config field must be a two-element range: {name}")
    low = _finite_number(value[0], f"{name}[0]")
    high = _finite_number(value[1], f"{name}[1]")
    if low > high:
        raise ValueError(f"push recovery runtime config range must be ordered: {name}")
    return (low, high)


def _positive_finite_number(value: Any, name: str) -> float:
    result = _finite_number(value, name)
    if result <= 0.0:
        raise ValueError(f"push recovery runtime config field must be positive: {name}")
    return result


def _non_negative_finite_number(value: Any, name: str) -> float:
    result = _finite_number(value, name)
    if result < 0.0:
        raise ValueError(
            f"push recovery runtime config field must be non-negative: {name}"
        )
    return result


def _unit_interval(value: Any, name: str) -> float:
    result = _finite_number(value, name)
    if result < 0.0 or result > 1.0:
        raise ValueError(
            f"push recovery runtime config field must be within [0, 1]: {name}"
        )
    return result


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(
            f"push recovery runtime config field must be a positive integer: {name}"
        )
    return value


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(
            f"push recovery runtime config field must be boolean: {name}"
        )
    return value


def _non_empty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"push recovery runtime config field must be a non-empty string: {name}"
        )
    return value


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    import yaml  # type: ignore[import-not-found]

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"runtime config must be a mapping: {path}")
    return data
