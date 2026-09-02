"""Config-level Phase 2 baseline overrides.

The baseline intentionally reuses existing Unitree/IsaacLab reward functions.
It changes weights and scalar training settings only, which keeps the future
Agent loop compatible with the protected-code boundary in the project plan.
"""

from __future__ import annotations

from typing import Any


def apply_phase2_baseline_overrides(env_cfg: Any, agent_cfg: Any, phase2_cfg: dict[str, Any]) -> dict[str, Any]:
    """Apply manual baseline reward/curriculum/PPO overrides.

    Returns a report describing which requested fields were applied or missing.
    """

    report: dict[str, Any] = {
        "reward_weights_applied": {},
        "reward_weights_missing": [],
        "reward_params_applied": {},
        "reward_params_missing": [],
        "curriculum_applied": {},
        "ppo_applied": {},
        "ppo_missing": [],
    }

    rewards_cfg = getattr(env_cfg, "rewards", None)
    for name, weight in phase2_cfg.get("reward_weights", {}).items():
        term = getattr(rewards_cfg, name, None) if rewards_cfg is not None else None
        if term is None or not hasattr(term, "weight"):
            report["reward_weights_missing"].append(name)
            continue
        term.weight = float(weight)
        report["reward_weights_applied"][name] = float(weight)

    for term_name, params in phase2_cfg.get("reward_params", {}).items():
        term = getattr(rewards_cfg, term_name, None) if rewards_cfg is not None else None
        if term is None or not hasattr(term, "params"):
            report["reward_params_missing"].append(term_name)
            continue
        for key, value in params.items():
            term.params[key] = value
        report["reward_params_applied"][term_name] = dict(params)

    curriculum_cfg = phase2_cfg.get("curriculum", {})
    env_curriculum = getattr(env_cfg, "curriculum", None)
    if env_curriculum is not None and bool(curriculum_cfg.get("disable_terrain_levels", False)):
        if hasattr(env_curriculum, "terrain_levels"):
            setattr(env_curriculum, "terrain_levels", None)
            report["curriculum_applied"]["terrain_levels"] = None
    if env_curriculum is not None and bool(curriculum_cfg.get("disable_velocity_command_levels", False)):
        if hasattr(env_curriculum, "lin_vel_cmd_levels"):
            setattr(env_curriculum, "lin_vel_cmd_levels", None)
            report["curriculum_applied"]["lin_vel_cmd_levels"] = None

    for name, value in phase2_cfg.get("ppo", {}).items():
        if value is None:
            continue
        if hasattr(agent_cfg, name):
            setattr(agent_cfg, name, value)
            report["ppo_applied"][name] = value
        else:
            report["ppo_missing"].append(name)

    return report

