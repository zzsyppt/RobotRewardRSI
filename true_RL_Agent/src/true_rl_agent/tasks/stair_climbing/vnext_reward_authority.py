"""Morphology-neutral authority for every active Stair reward program."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.json_contract import load_strict_json_object

STAIR_VNEXT_REWARD_AUTHORITY_RELATIVE_PATH = (
    "configs/tasks/stair_vnext_reward_authority_v1.json"
)
STAIR_VNEXT_REWARD_AUTHORITY_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.vnext_reward_authority.v1"
)


@dataclass(frozen=True)
class StairVNextRewardAuthority:
    _payload: Mapping[str, Any] = field(repr=False)

    @classmethod
    def load(cls, path: str | Path) -> "StairVNextRewardAuthority":
        payload = load_strict_json_object(
            Path(path), context="Stair VNext reward authority"
        )
        expected_fields = {
            "schema_version",
            "authority_id",
            "task_id",
            "common_input_field_names",
            "joint_group_inputs",
            "output_contract",
            "dispatcher",
            "reference_equivalence",
            "performance_gate",
            "authority_hash",
        }
        if set(payload) != expected_fields:
            raise ValueError("Stair VNext reward authority fields mismatch")
        if payload.get("schema_version") != STAIR_VNEXT_REWARD_AUTHORITY_SCHEMA_VERSION:
            raise ValueError("Stair VNext reward authority schema mismatch")
        if payload.get("task_id") != "unitree_stair_climbing":
            raise ValueError("Stair VNext reward authority task mismatch")
        common = payload.get("common_input_field_names")
        if not isinstance(common, list) or not common:
            raise ValueError("Stair VNext reward authority common inputs are empty")
        groups = payload.get("joint_group_inputs")
        if not isinstance(groups, list):
            raise ValueError("Stair VNext reward authority joint groups are invalid")
        if any(not isinstance(item, str) or not item for item in common):
            raise ValueError("Stair VNext common input names are invalid")
        if len(common) != len(set(common)):
            raise ValueError("Stair VNext common input names are duplicated")
        for item in groups:
            if not isinstance(item, Mapping) or set(item) != {"role_id", "source"}:
                raise ValueError("Stair VNext joint group input is invalid")
            if not all(isinstance(value, str) and value for value in item.values()):
                raise ValueError("Stair VNext joint group input is empty")
        expected_hash = canonical_json_hash(
            {key: value for key, value in payload.items() if key != "authority_hash"}
        )
        if payload.get("authority_hash") != expected_hash:
            raise ValueError("Stair VNext reward authority hash mismatch")
        return cls(_payload=_freeze(payload))

    @property
    def authority_hash(self) -> str:
        return str(self._payload["authority_hash"])

    @property
    def common_input_field_names(self) -> tuple[str, ...]:
        return tuple(self._payload["common_input_field_names"])

    @property
    def joint_group_inputs(self) -> tuple[Mapping[str, str], ...]:
        return tuple(self._payload["joint_group_inputs"])

    def runtime_authority_payload(self) -> dict[str, Any]:
        return {
            key: _thaw(self._payload[key])
            for key in (
                "output_contract",
                "dispatcher",
                "reference_equivalence",
                "performance_gate",
            )
        }

    def task_projection(self) -> dict[str, Any]:
        return {
            "schema_version": self._payload["schema_version"],
            "common_input_field_names": list(self.common_input_field_names),
            "joint_group_inputs": _thaw(self._payload["joint_group_inputs"]),
            **self.runtime_authority_payload(),
        }

    def to_dict(self) -> dict[str, Any]:
        return _thaw(self._payload)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


__all__ = [
    "STAIR_VNEXT_REWARD_AUTHORITY_RELATIVE_PATH",
    "STAIR_VNEXT_REWARD_AUTHORITY_SCHEMA_VERSION",
    "StairVNextRewardAuthority",
]
