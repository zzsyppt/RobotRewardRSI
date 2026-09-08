"""Thin adapter of Unitree's G1-29 locomotion configuration to Isaac Lab 3."""

from __future__ import annotations

import importlib
from typing import Any


def instantiate_g1_29_environment_cfg(*, device: str, num_envs: int) -> Any:
    from true_rl_agent.backends.isaac_lab_runtime import resolve_physx_environment

    vendor = importlib.import_module(
        "unitree_rl_lab.tasks.locomotion.robots.g1.29dof.velocity_env_cfg"
    )

    config = resolve_physx_environment(vendor.RobotEnvCfg())
    config.sim.device = device
    config.scene.num_envs = num_envs
    return config


def instantiate_g1_29_runner_cfg() -> Any:
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg
    from true_rl_agent.robots.rsl_rl_runtime import (
        G1_29DOF_UNITREE_RL_LAB_AGENT_CONFIGURATION,
    )

    config = RslRlOnPolicyRunnerCfg(
        num_steps_per_env=24,
        max_iterations=50000,
        save_interval=100,
        experiment_name="",
        obs_groups={"actor": ["policy"], "critic": ["critic"]},
    )
    G1_29DOF_UNITREE_RL_LAB_AGENT_CONFIGURATION.apply_to(config)
    return config
