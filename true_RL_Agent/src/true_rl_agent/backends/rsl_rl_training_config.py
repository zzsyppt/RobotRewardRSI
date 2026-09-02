"""Canonical RSL-RL policy and optimizer execution configuration."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

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

    def apply_and_verify(self, agent_cfg: Any) -> None:
        """Apply the bound payload and prove the effective runner payload is exact."""

        for section_name, expected in (
            ("policy", self.policy),
            ("algorithm", self.algorithm),
        ):
            target = getattr(agent_cfg, section_name, None)
            if target is None:
                raise ValueError(f"RSL-RL agent_cfg is missing {section_name}")
            for key, value in expected.items():
                if not hasattr(target, key):
                    raise ValueError(
                        f"RSL-RL agent_cfg.{section_name} is missing {key}"
                    )
                setattr(target, key, copy.deepcopy(_thaw_json(value)))
        to_dict = getattr(agent_cfg, "to_dict", None)
        if not callable(to_dict):
            raise TypeError("RSL-RL agent_cfg must provide to_dict()")
        effective = to_dict()
        if not isinstance(effective, Mapping):
            raise ValueError("RSL-RL agent_cfg.to_dict() must return an object")
        for section_name, expected in (
            ("policy", self.policy),
            ("algorithm", self.algorithm),
        ):
            actual = _canonical_object(
                effective.get(section_name),
                f"effective RSL-RL {section_name} configuration",
            )
            if actual != _thaw_json(expected):
                raise ValueError(
                    f"effective RSL-RL {section_name} configuration mismatch"
                )


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
