"""Canonical RSL-RL policy and optimizer execution configuration."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from true_rl_agent.training_binding import TrainingBinding

from true_rl_agent.evaluation.binding import (
    canonical_json_hash,
    require_sha256_digest,
)
from true_rl_agent.json_contract import (
    canonical_json_text,
    strict_json_object_loads,
)

RSL_RL_AGENT_CONFIGURATION_SCHEMA_VERSION = (
    "true_rl_agent.rsl_rl_agent_configuration.v1"
)
RSL_RL_POLICY_CONFIGURATION_SCHEMA_VERSION = (
    "true_rl_agent.rsl_rl_policy_configuration.v1"
)
RSL_RL_ALGORITHM_CONFIGURATION_SCHEMA_VERSION = (
    "true_rl_agent.rsl_rl_algorithm_configuration.v1"
)
# In the frozen v1 contract, PPO means RSL-RL 3's hard-coded Adam optimizer.
# Making that implicit choice explicit in RSL-RL 5 does not rewrite old hashes.
RSL_RL_CONFIGURATION_MAPPING_VERSION = "rsl_rl_3_actor_critic_to_rsl_rl_5_mlp.v1"


@dataclass(frozen=True)
class RslRlAgentConfiguration:
    """Complete policy/algorithm payload applied before RSL-RL runner creation."""

    policy: Mapping[str, Any]
    algorithm: Mapping[str, Any]
    policy_architecture_hash: str | None = field(default=None, compare=False)
    optimizer_training_profile_hash: str | None = field(
        default=None,
        compare=False,
    )
    agent_configuration_hash: str | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        policy = _canonical_object(self.policy, "RSL-RL policy configuration")
        algorithm = _canonical_object(
            self.algorithm,
            "RSL-RL algorithm configuration",
        )
        if not policy or not algorithm:
            raise ValueError("RSL-RL policy and algorithm configurations must be non-empty")
        policy_hash = canonical_json_hash(
            {
                "schema_version": RSL_RL_POLICY_CONFIGURATION_SCHEMA_VERSION,
                "policy": policy,
            }
        )
        algorithm_hash = canonical_json_hash(
            {
                "schema_version": RSL_RL_ALGORITHM_CONFIGURATION_SCHEMA_VERSION,
                "algorithm": algorithm,
            }
        )
        payload = {
            "schema_version": RSL_RL_AGENT_CONFIGURATION_SCHEMA_VERSION,
            "policy": policy,
            "algorithm": algorithm,
            "policy_architecture_hash": policy_hash,
            "optimizer_training_profile_hash": algorithm_hash,
        }
        configuration_hash = canonical_json_hash(payload)
        supplied = (
            ("policy_architecture_hash", self.policy_architecture_hash, policy_hash),
            (
                "optimizer_training_profile_hash",
                self.optimizer_training_profile_hash,
                algorithm_hash,
            ),
            (
                "agent_configuration_hash",
                self.agent_configuration_hash,
                configuration_hash,
            ),
        )
        for field_name, actual, expected in supplied:
            if actual is not None:
                require_sha256_digest(actual, field_name)
                if actual != expected:
                    raise ValueError(f"RSL-RL {field_name} mismatch")
        object.__setattr__(self, "policy", _freeze_json(policy))
        object.__setattr__(self, "algorithm", _freeze_json(algorithm))
        object.__setattr__(self, "policy_architecture_hash", policy_hash)
        object.__setattr__(self, "optimizer_training_profile_hash", algorithm_hash)
        object.__setattr__(self, "agent_configuration_hash", configuration_hash)

    @classmethod
    def from_mapping(cls, value: Any) -> "RslRlAgentConfiguration":
        if not isinstance(value, Mapping):
            raise ValueError("RSL-RL agent configuration must be an object")
        expected_fields = {
            "schema_version",
            "policy",
            "algorithm",
            "policy_architecture_hash",
            "optimizer_training_profile_hash",
            "agent_configuration_hash",
        }
        if set(value) != expected_fields:
            raise ValueError("RSL-RL agent configuration fields mismatch")
        if value.get("schema_version") != RSL_RL_AGENT_CONFIGURATION_SCHEMA_VERSION:
            raise ValueError("RSL-RL agent configuration schema_version mismatch")
        return cls(
            policy=value.get("policy"),
            algorithm=value.get("algorithm"),
            policy_architecture_hash=value.get("policy_architecture_hash"),
            optimizer_training_profile_hash=value.get(
                "optimizer_training_profile_hash"
            ),
            agent_configuration_hash=value.get("agent_configuration_hash"),
        )

    @classmethod
    def from_json(cls, value: str) -> "RslRlAgentConfiguration":
        if not isinstance(value, str) or not value:
            raise ValueError("RSL-RL agent configuration JSON must be non-empty")

        parsed = strict_json_object_loads(
            value,
            context="RSL-RL agent configuration JSON",
        )
        return cls.from_mapping(parsed)

    def to_dict(self) -> dict[str, Any]:
        assert self.policy_architecture_hash is not None
        assert self.optimizer_training_profile_hash is not None
        assert self.agent_configuration_hash is not None
        return {
            "schema_version": RSL_RL_AGENT_CONFIGURATION_SCHEMA_VERSION,
            "policy": _thaw_json(self.policy),
            "algorithm": _thaw_json(self.algorithm),
            "policy_architecture_hash": self.policy_architecture_hash,
            "optimizer_training_profile_hash": (
                self.optimizer_training_profile_hash
            ),
            "agent_configuration_hash": self.agent_configuration_hash,
        }

    def to_canonical_json(self) -> str:
        return canonical_json_text(self.to_dict())

    def training_binding(
        self, *, initialization_mode: str, train_seed: int, budget_steps: int,
        capability_profile_hash: str, initialization_checkpoint_hash: str | None = None,
        task_protocol_hash: str | None = None,
    ) -> TrainingBinding:
        """Bind a fixed-seed execution without choosing its task/lineage domain."""
        from true_rl_agent.training_binding import (
            TrainingBinding, fixed_training_seed_resolver_hash, initialization_seed_schedule_hash,
        )

        return TrainingBinding(
            initialization_mode=initialization_mode,
            initialization_checkpoint_hash=initialization_checkpoint_hash,
            policy_architecture_hash=str(self.policy_architecture_hash),
            agent_configuration_hash=str(self.agent_configuration_hash),
            optimizer_training_profile_hash=str(self.optimizer_training_profile_hash),
            initialization_seed_schedule_hash=initialization_seed_schedule_hash(train_seed),
            budget_steps=budget_steps,
            private_training_seed_resolver_hash=fixed_training_seed_resolver_hash(train_seed),
            capability_profile_hash=capability_profile_hash,
            task_protocol_hash=task_protocol_hash,
        )

    def apply_to(self, agent_cfg: Any) -> None:
        """Materialize the bound scientific payload in the RSL-RL 5 config."""

        sections = self.runtime_sections()
        for section_name, expected in sections.items():
            setattr(agent_cfg, section_name, copy.deepcopy(expected))
        # Lab's base config retains this deprecated field for its other clients.
        # It is not another policy source for our runtime.
        if hasattr(agent_cfg, "policy"):
            agent_cfg.policy = None

    def verify_runner(self, runner: Any) -> dict[str, Any]:
        """Inspect a fresh learner before any checkpoint load or learning.

        This checks initial noise and configured learning rate. The returned
        evidence describes construction, not the checkpoint's learned state.
        """
        import torch
        from rsl_rl.utils import resolve_nn_activation

        learner = runner.alg
        models = {}
        for role in ("actor", "critic"):
            model = getattr(learner, role)
            # Sequential reuses activation objects; children() deduplicates them.
            layers = list(model.mlp)
            linear = [layer for layer in layers if isinstance(layer, torch.nn.Linear)]
            hidden = [layer.out_features for layer in linear[:-1]]
            activation = type(resolve_nn_activation(self.policy["activation"]))
            nonlinear = [layer for layer in layers if not isinstance(layer, torch.nn.Linear)]
            if (
                hidden != list(self.policy[role + "_hidden_dims"])
                or len(nonlinear) != len(hidden)
                or any(type(layer) is not activation for layer in nonlinear)
                or model.obs_normalization != self.policy[role + "_obs_normalization"]
            ):
                raise ValueError(f"constructed RSL-RL {role} differs from the bound policy")
            models[role] = {
                "input_dim": linear[0].in_features,
                "output_dim": linear[-1].out_features,
                "hidden_dims": hidden,
                "activation": activation.__name__,
                "obs_normalization": model.obs_normalization,
                "obs_groups": list(model.obs_groups),
            }
        distribution = learner.actor.distribution
        std = (distribution.std_param if distribution.std_type == "scalar"
               else distribution.log_std_param.exp())
        if (type(distribution).__name__ != "GaussianDistribution"
                or distribution.std_type != self.policy["noise_std_type"]
                or not torch.allclose(std, torch.full_like(std, self.policy["init_noise_std"]))
                or learner.critic.distribution is not None):
            raise ValueError("constructed RSL-RL distribution differs from the bound policy")
        actual_algorithm = {
            name: getattr(learner, name)
            for name in self.algorithm
            if name not in {"class_name", "rnd_cfg", "symmetry_cfg"}
        }
        expected_algorithm = {name: self.algorithm[name] for name in actual_algorithm}
        if (type(learner).__name__ != self.algorithm["class_name"]
                or actual_algorithm != expected_algorithm
                or learner.rnd is not None or learner.symmetry is not None
                or type(learner.optimizer) is not torch.optim.Adam
                or any(group["lr"] != self.algorithm["learning_rate"]
                       for group in learner.optimizer.param_groups)):
            raise ValueError("constructed RSL-RL learner differs from the bound PPO")
        return {
            "schema_version": "true_rl_agent.rsl_rl_constructed_learner.v1",
            "configuration_mapping_version": RSL_RL_CONFIGURATION_MAPPING_VERSION,
            "models": models,
            "algorithm": actual_algorithm,
            "optimizer": "Adam",
            "optimizer_source": "frozen_v1_PPO_implicit_Adam",
            "noise_std_type": distribution.std_type,
            "initial_noise_std": self.policy["init_noise_std"],
        }

    def runtime_sections(self) -> dict[str, Any]:
        """Translate the frozen MLP description without changing its hashes."""

        policy = self.policy
        if policy.get("class_name") != "ActorCritic":
            raise ValueError("the RSL-RL runtime supports the bound MLP policy only")
        actor = {
            "class_name": "MLPModel",
            "hidden_dims": _thaw_json(policy["actor_hidden_dims"]),
            "activation": policy["activation"],
            "obs_normalization": policy["actor_obs_normalization"],
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": policy["init_noise_std"],
                "std_type": policy["noise_std_type"],
            },
        }
        critic = {
            "class_name": "MLPModel",
            "hidden_dims": _thaw_json(policy["critic_hidden_dims"]),
            "activation": policy["activation"],
            "obs_normalization": policy["critic_obs_normalization"],
            "distribution_cfg": None,
        }
        return {
            "actor": actor,
            "critic": critic,
            "algorithm": {**_thaw_json(self.algorithm), "optimizer": "adam"},
        }


def _canonical_object(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    normalized = _normalize_json(dict(value), field_name)
    assert isinstance(normalized, dict)
    return normalized


def _normalize_json(value: Any, field_name: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field_name} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{field_name} contains an invalid key")
            result[key] = _normalize_json(item, f"{field_name}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [
            _normalize_json(item, f"{field_name}[{index}]")
            for index, item in enumerate(value)
        ]
    raise ValueError(f"{field_name} contains a non-JSON value")


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


__all__ = [
    "RSL_RL_AGENT_CONFIGURATION_SCHEMA_VERSION",
    "RslRlAgentConfiguration",
]
