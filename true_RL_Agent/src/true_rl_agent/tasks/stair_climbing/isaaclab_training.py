"""Task-owned configuration helpers for Stair RSL-RL execution."""

from __future__ import annotations

from typing import Any, Mapping

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
    execution_contract: Mapping[str, Any],
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
    if not isinstance(execution_contract, Mapping):
        raise TypeError("Stair RSL-RL execution contract must be an object")

    configuration.apply_to(agent_cfg)
    agent_cfg.seed = seed
    agent_cfg.device = device
    agent_cfg.num_steps_per_env = num_steps_per_env
    agent_cfg.max_iterations = max_iterations
    agent_cfg.save_interval = save_interval
    agent_cfg.run_name = run_name
    agent_cfg.experiment_name = experiment_name
    effective_runner = apply_stair_rsl_rl_runner_contract(
        agent_cfg,
        execution_contract=execution_contract,
    )

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
    if effective_runner:
        effective.update(effective_runner)
    return {
        **effective,
        "policy_architecture_hash": configuration.policy_architecture_hash,
        "optimizer_training_profile_hash": (
            configuration.optimizer_training_profile_hash
        ),
        "agent_configuration_hash": configuration.agent_configuration_hash,
        "runner_contract_mode": (
            "explicit" if effective_runner else "legacy_registry_defaults"
        ),
    }


def apply_stair_rsl_rl_runner_contract(
    agent_cfg: Any,
    *,
    execution_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the task-bound RSL-RL runner fields before wrapper construction."""

    if not isinstance(execution_contract, Mapping):
        raise TypeError("Stair RSL-RL execution contract must be an object")
    if "observation_groups" not in execution_contract:
        return {}
    expected = {
        name: execution_contract[name]
        for name in (
            "observation_groups",
            "clip_actions",
            "empirical_normalization",
            "logger",
            "resume",
        )
    }
    if expected["observation_groups"] != {
        "policy": ["policy"],
        "critic": ["policy"],
    }:
        raise ValueError("Stair RSL-RL observation groups are invalid")
    native_groups = {
        "actor": list(expected["observation_groups"]["policy"]),
        "critic": list(expected["observation_groups"]["critic"]),
    }
    agent_cfg.obs_groups = native_groups
    agent_cfg.clip_actions = expected["clip_actions"]
    agent_cfg.empirical_normalization = expected["empirical_normalization"]
    agent_cfg.logger = expected["logger"]
    agent_cfg.resume = expected["resume"]
    effective = {
        "observation_groups": {
            "policy": list(agent_cfg.obs_groups["actor"]),
            "critic": list(agent_cfg.obs_groups["critic"]),
        },
        "clip_actions": agent_cfg.clip_actions,
        "empirical_normalization": agent_cfg.empirical_normalization,
        "logger": agent_cfg.logger,
        "resume": agent_cfg.resume,
    }
    if effective != expected:
        raise RuntimeError("effective Stair RSL-RL runner contract drifted")
    return effective


def verify_stair_rsl_rl_observation_groups(
    runner: Any,
    *,
    execution_contract: Mapping[str, Any],
) -> dict[str, list[str]] | None:
    """Prove the runner resolved exactly the observation sets bound by V2."""

    expected = execution_contract.get("observation_groups")
    if expected is None:
        return None
    expected = {"actor": list(expected["policy"]), "critic": list(expected["critic"])}
    runner_cfg = getattr(runner, "cfg", None)
    if not isinstance(runner_cfg, Mapping):
        raise RuntimeError("Stair RSL-RL runner configuration is unavailable")
    raw_groups = runner_cfg.get("obs_groups")
    if not isinstance(raw_groups, Mapping):
        raise RuntimeError("Stair RSL-RL resolved observation groups are unavailable")
    resolved = {name: list(groups) for name, groups in raw_groups.items()}
    if resolved != expected:
        raise RuntimeError("resolved Stair RSL-RL observation groups drifted")
    # Preserve the protocol's policy/critic vocabulary in its existing evidence
    # fields. Constructed learner evidence records the native actor/critic names.
    return {"policy": resolved["actor"], "critic": resolved["critic"]}


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


__all__ = [
    "apply_stair_rsl_rl_runner_contract",
    "configure_stair_rsl_rl_agent",
    "verify_stair_rsl_rl_observation_groups",
]
