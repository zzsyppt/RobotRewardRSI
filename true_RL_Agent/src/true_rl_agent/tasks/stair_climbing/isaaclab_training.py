"""Task-owned configuration helpers for Stair RSL-RL execution."""

from __future__ import annotations

from typing import Any

from true_rl_agent.backends.rsl_rl_training_config import (
    RslRlAgentConfiguration,
)


def configure_stair_rsl_rl_agent(
    agent_cfg: Any,
    *,
    configuration: RslRlAgentConfiguration,
    seed: int,
    device: str,
    num_steps_per_env: int,
    max_iterations: int,
    save_interval: int,
    run_name: str,
    experiment_name: str,
) -> dict[str, Any]:
    """Apply one validated policy/optimizer payload and execution envelope."""

    if not isinstance(configuration, RslRlAgentConfiguration):
        raise TypeError("Stair RSL-RL configuration is invalid")
    _nonnegative_integer(seed, "seed")
    for field_name, value in (
        ("num_steps_per_env", num_steps_per_env),
        ("max_iterations", max_iterations),
        ("save_interval", save_interval),
    ):
        _positive_integer(value, field_name)
    for field_name, value in (
        ("device", device),
        ("run_name", run_name),
        ("experiment_name", experiment_name),
    ):
        _canonical_text(value, field_name)

    configuration.apply_and_verify(agent_cfg)
    agent_cfg.seed = seed
    agent_cfg.device = device
    agent_cfg.num_steps_per_env = num_steps_per_env
    agent_cfg.max_iterations = max_iterations
    agent_cfg.save_interval = save_interval
    agent_cfg.run_name = run_name
    agent_cfg.experiment_name = experiment_name

    effective = {
        "seed": agent_cfg.seed,
        "device": agent_cfg.device,
        "num_steps_per_env": agent_cfg.num_steps_per_env,
        "max_iterations": agent_cfg.max_iterations,
        "save_interval": agent_cfg.save_interval,
        "run_name": agent_cfg.run_name,
        "experiment_name": agent_cfg.experiment_name,
    }
    expected = {
        "seed": seed,
        "device": device,
        "num_steps_per_env": num_steps_per_env,
        "max_iterations": max_iterations,
        "save_interval": save_interval,
        "run_name": run_name,
        "experiment_name": experiment_name,
    }
    if effective != expected:
        raise RuntimeError("effective Stair RSL-RL execution configuration drifted")
    return {
        **effective,
        "policy_architecture_hash": configuration.policy_architecture_hash,
        "optimizer_training_profile_hash": (
            configuration.optimizer_training_profile_hash
        ),
        "agent_configuration_hash": configuration.agent_configuration_hash,
    }


def _positive_integer(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"Stair RSL-RL {field_name} must be a positive integer")
    return value


def _nonnegative_integer(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"Stair RSL-RL {field_name} must be a non-negative integer")
    return value


def _canonical_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"Stair RSL-RL {field_name} must be canonical text")
    return value


__all__ = ["configure_stair_rsl_rl_agent"]
