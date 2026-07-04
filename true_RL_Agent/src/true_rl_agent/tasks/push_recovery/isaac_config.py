"""Project-local runtime config helpers for push recovery IsaacLab jobs."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any


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
        return config
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


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    import yaml  # type: ignore[import-not-found]

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"runtime config must be a mapping: {path}")
    return data
