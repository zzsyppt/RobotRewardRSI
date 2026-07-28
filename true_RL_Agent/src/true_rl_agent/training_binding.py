"""Immutable training comparison and checkpoint-lineage binding."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from true_rl_agent.evaluation.binding import (
    canonical_json_hash,
    require_sha256_checksum,
    require_sha256_digest,
)
from true_rl_agent.json_contract import strict_json_object_loads
from true_rl_agent.runner.argv_contract import unique_option_value

TRAINING_BINDING_SCHEMA_VERSION = "true_rl_agent.training_binding.v1"


@dataclass(frozen=True)
class TrainingBinding:
    initialization_mode: str
    policy_architecture_hash: str
    agent_configuration_hash: str
    initialization_seed_schedule_hash: str
    optimizer_training_profile_hash: str
    budget_steps: int
    private_training_seed_resolver_hash: str
    capability_profile_hash: str
    initialization_checkpoint_hash: str | None = None
    binding_hash: str | None = None

    def __post_init__(self) -> None:
        if self.initialization_mode not in {"random", "warm_start"}:
            raise ValueError("training binding initialization_mode is invalid")
        for field_name in (
            "policy_architecture_hash",
            "agent_configuration_hash",
            "initialization_seed_schedule_hash",
            "optimizer_training_profile_hash",
            "private_training_seed_resolver_hash",
            "capability_profile_hash",
        ):
            require_sha256_digest(getattr(self, field_name), field_name)
        _require_positive_integer(self.budget_steps, "training binding budget_steps")
        if self.initialization_mode == "random":
            if self.initialization_checkpoint_hash is not None:
                raise ValueError(
                    "random training binding cannot carry an initialization checkpoint"
                )
        else:
            require_sha256_checksum(
                self.initialization_checkpoint_hash,
                "training binding initialization checkpoint hash",
            )
        expected = canonical_json_hash(self._hash_payload())
        if self.binding_hash is not None:
            require_sha256_digest(self.binding_hash, "training binding hash")
            if self.binding_hash != expected:
                raise ValueError("training binding hash mismatch")
        object.__setattr__(self, "binding_hash", expected)

    @classmethod
    def from_mapping(cls, value: Any) -> "TrainingBinding":
        if not isinstance(value, Mapping):
            raise ValueError("training binding must be an object")
        expected_fields = {
            "schema_version",
            "initialization_mode",
            "initialization_checkpoint_hash",
            "policy_architecture_hash",
            "agent_configuration_hash",
            "initialization_seed_schedule_hash",
            "optimizer_training_profile_hash",
            "budget_steps",
            "private_training_seed_resolver_hash",
            "capability_profile_hash",
            "binding_hash",
        }
        if set(value) != expected_fields:
            raise ValueError("training binding fields mismatch")
        if value.get("schema_version") != TRAINING_BINDING_SCHEMA_VERSION:
            raise ValueError("training binding schema_version mismatch")
        return cls(
            initialization_mode=value.get("initialization_mode"),
            initialization_checkpoint_hash=value.get(
                "initialization_checkpoint_hash"
            ),
            policy_architecture_hash=value.get("policy_architecture_hash"),
            agent_configuration_hash=value.get("agent_configuration_hash"),
            initialization_seed_schedule_hash=value.get(
                "initialization_seed_schedule_hash"
            ),
            optimizer_training_profile_hash=value.get(
                "optimizer_training_profile_hash"
            ),
            budget_steps=value.get("budget_steps"),
            private_training_seed_resolver_hash=value.get(
                "private_training_seed_resolver_hash"
            ),
            capability_profile_hash=value.get("capability_profile_hash"),
            binding_hash=value.get("binding_hash"),
        )

    def to_public_dict(self) -> dict[str, Any]:
        assert self.binding_hash is not None
        return {
            "schema_version": TRAINING_BINDING_SCHEMA_VERSION,
            "initialization_mode": self.initialization_mode,
            "initialization_checkpoint_hash": self.initialization_checkpoint_hash,
            "policy_architecture_hash": self.policy_architecture_hash,
            "agent_configuration_hash": self.agent_configuration_hash,
            "initialization_seed_schedule_hash": (
                self.initialization_seed_schedule_hash
            ),
            "optimizer_training_profile_hash": (
                self.optimizer_training_profile_hash
            ),
            "budget_steps": self.budget_steps,
            "capability_profile_hash": self.capability_profile_hash,
            "binding_hash": self.binding_hash,
        }

    def to_private_dict(self) -> dict[str, Any]:
        assert self.binding_hash is not None
        return {**self._hash_payload(), "binding_hash": self.binding_hash}

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "schema_version": TRAINING_BINDING_SCHEMA_VERSION,
            "initialization_mode": self.initialization_mode,
            "initialization_checkpoint_hash": self.initialization_checkpoint_hash,
            "policy_architecture_hash": self.policy_architecture_hash,
            "agent_configuration_hash": self.agent_configuration_hash,
            "initialization_seed_schedule_hash": (
                self.initialization_seed_schedule_hash
            ),
            "optimizer_training_profile_hash": self.optimizer_training_profile_hash,
            "budget_steps": self.budget_steps,
            "private_training_seed_resolver_hash": (
                self.private_training_seed_resolver_hash
            ),
            "capability_profile_hash": self.capability_profile_hash,
        }


def initialization_seed_schedule_hash(seed: int) -> str:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("training seed must be a non-negative integer")
    return canonical_json_hash(
        {
            "schema_version": "true_rl_agent.initialization_seed_schedule.v1",
            "seeds": [seed],
        }
    )


def fixed_training_seed_resolver_hash(seed: int) -> str:
    """Bind the runner-owned fixed seed without exposing task semantics."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("training seed must be a non-negative integer")
    return canonical_json_hash(
        {
            "schema_version": "true_rl_agent.private_training_seed_resolver.v1",
            "mode": "runner_fixed_seed",
            "seed": seed,
        }
    )


def validate_training_execution_argv(
    argv: Sequence[str],
    *,
    binding: TrainingBinding,
) -> Mapping[str, Any]:
    """Bind generic structured train argv to its immutable job contract."""

    options = {
        "--training-binding-hash": str(binding.binding_hash),
        "--capability-profile-hash": binding.capability_profile_hash,
        "--policy-architecture-hash": binding.policy_architecture_hash,
        "--optimizer-training-profile-hash": (
            binding.optimizer_training_profile_hash
        ),
        "--agent-configuration-hash": binding.agent_configuration_hash,
        "--initialization-seed-schedule-hash": (
            binding.initialization_seed_schedule_hash
        ),
    }
    for option, expected in options.items():
        actual = unique_option_value(argv, option)
        if actual != expected:
            raise ValueError(f"train argv {option} does not match training binding")

    configuration = _load_agent_configuration_json(
        unique_option_value(argv, "--agent-configuration-json")
    )
    checks = {
        "policy_architecture_hash": binding.policy_architecture_hash,
        "optimizer_training_profile_hash": (
            binding.optimizer_training_profile_hash
        ),
        "agent_configuration_hash": binding.agent_configuration_hash,
    }
    for field_name, expected in checks.items():
        if configuration.get(field_name) != expected:
            raise ValueError(
                f"train argv agent configuration {field_name} mismatch"
            )

    raw_seed = unique_option_value(argv, "--seed")
    if re.fullmatch(r"0|[1-9][0-9]*", raw_seed) is None:
        raise ValueError("train argv --seed must be a canonical non-negative integer")
    if initialization_seed_schedule_hash(int(raw_seed)) != (
        binding.initialization_seed_schedule_hash
    ):
        raise ValueError("train argv --seed does not match training binding")
    return configuration


def _load_agent_configuration_json(value: str) -> dict[str, Any]:
    if not isinstance(value, str) or not value:
        raise ValueError("agent configuration JSON must be non-empty")

    parsed = strict_json_object_loads(
        value,
        context="agent configuration JSON",
    )
    declared_hash = require_sha256_digest(
        parsed.get("agent_configuration_hash"),
        "agent configuration hash",
    )
    hash_payload = {
        key: item
        for key, item in parsed.items()
        if key != "agent_configuration_hash"
    }
    if canonical_json_hash(hash_payload) != declared_hash:
        raise ValueError("agent configuration hash mismatch")
    return parsed

__all__ = [
    "TRAINING_BINDING_SCHEMA_VERSION",
    "TrainingBinding",
    "fixed_training_seed_resolver_hash",
    "initialization_seed_schedule_hash",
    "validate_training_execution_argv",
]


def _require_positive_integer(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value
