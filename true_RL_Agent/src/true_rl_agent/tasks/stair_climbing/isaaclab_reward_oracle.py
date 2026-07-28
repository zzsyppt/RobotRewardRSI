"""Probe-only Isaac Lab oracle for the canonical Stair R0 program."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import gymnasium as gym
import torch
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
    StairRewardInputBuilder,
)
from true_rl_agent.tasks.stair_climbing.reward_program import (
    QualifiedStairRewardProgram,
    evaluate_stair_reward_program,
)
from true_rl_agent.tasks.stair_climbing.termination_capture import (
    install_stair_termination_capture,
)

STAIR_REWARD_ORACLE_GYM_ID = "TrueRLAgent-Stair-Reward-Oracle-v0"
STAIR_REWARD_ORACLE_EXTRAS_KEY = "true_rl_agent_stair_reward_oracle"
STAIR_REWARD_ORACLE_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.reward_equivalence_transition.v1"
)


class StairRewardEquivalenceRecorder(RecorderTerm):
    """Compare official and R0 rewards on the same pre-reset transition."""

    def __init__(self, cfg: RecorderTermCfg, env: Any) -> None:
        super().__init__(cfg, env)
        program = getattr(cfg, "qualified_program", None)
        base_protocol = getattr(cfg, "base_protocol", None)
        if not isinstance(program, QualifiedStairRewardProgram):
            raise TypeError("Stair reward oracle requires a qualified R0 program")
        if not isinstance(base_protocol, Mapping):
            raise TypeError("Stair reward oracle requires the base task protocol")
        self._program = program
        self._input_builder = StairRewardInputBuilder(
            env,
            contract=program.contract,
            base_protocol=base_protocol,
        )
        equivalence = program.contract.to_public_dict()["reference_equivalence"]
        self._active_names = tuple(equivalence["active_term_names"])
        self._zero_weight_names = tuple(equivalence["zero_weight_term_names"])
        self._disabled_names = tuple(equivalence["disabled_term_names"])
        self._manager_names: tuple[str, ...] | None = None

    def _ensure_manager_contract(self) -> tuple[str, ...]:
        if self._manager_names is not None:
            return self._manager_names
        manager = self._env.reward_manager
        manager_names = tuple(manager.active_terms)
        expected_manager_names = set(self._active_names) | set(
            self._zero_weight_names
        )
        if set(manager_names) != expected_manager_names:
            raise RuntimeError(
                "Official Stair reward term set differs from the V2 contract"
            )
        if set(self._disabled_names) & set(manager_names):
            raise RuntimeError("Disabled official Stair rewards are still active")
        if tuple(self._program.term_names) != self._active_names:
            raise RuntimeError("Canonical R0 term order differs from the V2 contract")
        program_weights = {
            term.name: float(term.weight)
            for term in self._program.reward_terms
        }
        for name in self._active_names:
            manager_weight = float(manager.get_term_cfg(name).weight)
            if manager_weight != program_weights[name]:
                raise RuntimeError(
                    f"Canonical R0 weight differs from the official manager: {name}"
                )
        for name in self._zero_weight_names:
            if float(manager.get_term_cfg(name).weight) != 0.0:
                raise RuntimeError(
                    f"Official zero-weight Stair term is not zero: {name}"
                )
        self._manager_names = manager_names
        return manager_names

    def record_post_step(self) -> tuple[None, None]:
        env = self._env
        manager = env.reward_manager
        manager_names = self._ensure_manager_contract()
        inputs = self._input_builder.build()
        evaluation = evaluate_stair_reward_program(
            self._program,
            inputs,
            num_envs=int(env.num_envs),
        )
        term_payload: dict[str, dict[str, Any]] = {}
        for name in self._active_names:
            term_index = manager_names.index(name)
            term_cfg = manager.get_term_cfg(name)
            official_raw = term_cfg.func(env, **term_cfg.params)
            if not torch.is_tensor(official_raw) or tuple(official_raw.shape) != (
                int(env.num_envs),
            ):
                raise RuntimeError(
                    f"Official Stair reward output shape drifted: {name}"
                )
            official_weighted = official_raw * float(term_cfg.weight)
            term_payload[name] = {
                "official_raw": official_raw.detach().clone(),
                "official_weighted_rate": official_weighted.detach().clone(),
                "manager_weighted_rate": manager._step_reward[  # noqa: SLF001
                    :, term_index
                ]
                .detach()
                .clone(),
                "r0_raw": evaluation.raw_terms[name].detach().clone(),
                "r0_weighted_rate": evaluation.weighted_terms[name]
                .detach()
                .clone(),
            }
        zero_weight_rates = {
            name: manager._step_reward[:, manager_names.index(name)]  # noqa: SLF001
            .detach()
            .clone()
            for name in self._zero_weight_names
        }
        dt = float(env.step_dt)
        env.extras[STAIR_REWARD_ORACLE_EXTRAS_KEY] = {
            "schema_version": STAIR_REWARD_ORACLE_SCHEMA_VERSION,
            "common_step": int(env.common_step_counter),
            "manager_term_order": manager_names,
            "active_term_names": self._active_names,
            "zero_weight_term_names": self._zero_weight_names,
            "disabled_term_names": self._disabled_names,
            "terms": term_payload,
            "zero_weight_manager_rates": zero_weight_rates,
            "totals": {
                "manager_reward_rate": manager._reward_buf.detach().clone()  # noqa: SLF001
                / dt,
                "manager_term_sum_rate": manager._step_reward.detach()  # noqa: SLF001
                .sum(dim=1)
                .clone(),
                "r0_total_rate": evaluation.total_rate.detach().clone(),
            },
            "termination_masks": {
                "time_out": inputs.time_out.detach().clone(),
                "base_contact": inputs.base_contact.detach().clone(),
                "terminated_non_timeout": inputs.terminated_non_timeout.detach().clone(),
            },
        }
        return None, None


@configclass
class StairRewardEquivalenceRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = StairRewardEquivalenceRecorder
    qualified_program: object | None = None
    base_protocol: object | None = None


@configclass
class StairRewardEquivalenceRecorderManagerCfg(RecorderManagerBaseCfg):
    equivalence = StairRewardEquivalenceRecorderCfg()
    dataset_export_mode = DatasetExportMode.EXPORT_NONE
    export_in_record_pre_reset = False


def build_stair_reward_oracle_env_cfg(
    protocol: Mapping[str, Any],
    *,
    seed: int,
    device: str,
    reward_program: QualifiedStairRewardProgram,
) -> StairG1RoughEnvCfg:
    """Build a probe-only env retaining the untouched official RewardManager."""

    if not isinstance(reward_program, QualifiedStairRewardProgram):
        raise TypeError("Stair reward oracle requires a qualified reward program")
    cfg = build_stair_env_cfg(protocol, mode="train", seed=seed, device=device)
    install_stair_termination_capture(cfg.terminations)
    recorder = StairRewardEquivalenceRecorderManagerCfg()
    recorder.equivalence.qualified_program = reward_program
    recorder.equivalence.base_protocol = dict(protocol)
    cfg.recorders = recorder
    return cfg


def register_stair_reward_oracle_env() -> None:
    """Explicitly register the probe-only oracle without package side effects."""

    if STAIR_REWARD_ORACLE_GYM_ID in gym.registry:
        return
    gym.register(
        id=STAIR_REWARD_ORACLE_GYM_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
    )


__all__ = [
    "STAIR_REWARD_ORACLE_EXTRAS_KEY",
    "STAIR_REWARD_ORACLE_GYM_ID",
    "STAIR_REWARD_ORACLE_SCHEMA_VERSION",
    "StairRewardEquivalenceRecorder",
    "StairRewardEquivalenceRecorderManagerCfg",
    "build_stair_reward_oracle_env_cfg",
    "register_stair_reward_oracle_env",
]
