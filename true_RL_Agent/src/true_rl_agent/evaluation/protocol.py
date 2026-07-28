"""Task-neutral evaluation protocol contract."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

EVALUATION_PROTOCOL_SCHEMA_VERSION = "true_rl_agent.evaluation.protocol.v1"
DEFAULT_EVAL_PROTOCOL_VERSION = "fixed_public_accept_sealed_validation_v1"


@dataclass(frozen=True)
class EvaluationProtocol:
    """Fixed eval protocol fields shared by task specs, backends, runner specs, and seals."""

    eval_protocol_version: str
    public_seed_count: int
    public_accept_seed_count: int
    sealed_validation_seed_count: int
    eval_num_envs: int

    def __post_init__(self) -> None:
        if not isinstance(self.eval_protocol_version, str) or not self.eval_protocol_version:
            raise ValueError("eval_protocol_version must be non-empty")
        for name in (
            "public_seed_count",
            "public_accept_seed_count",
            "sealed_validation_seed_count",
            "eval_num_envs",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be positive")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "EvaluationProtocol":
        validation = validate_evaluation_protocol_mapping(payload)
        if validation.valid and validation.protocol is not None:
            return validation.protocol
        detail = ", ".join(validation.reasons) or "invalid evaluation protocol"
        raise ValueError(detail)

    @property
    def protocol_hash(self) -> str:
        encoded = json.dumps(
            self._hash_payload(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_public_dict(self) -> dict[str, Any]:
        return {
            **self._hash_payload(),
            "eval_protocol_hash": self.protocol_hash,
        }

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "schema_version": EVALUATION_PROTOCOL_SCHEMA_VERSION,
            "eval_protocol_version": self.eval_protocol_version,
            "public_seed_count": int(self.public_seed_count),
            "public_accept_seed_count": int(self.public_accept_seed_count),
            "sealed_validation_seed_count": int(self.sealed_validation_seed_count),
            "eval_num_envs": int(self.eval_num_envs),
        }


@dataclass(frozen=True)
class EvaluationProtocolValidation:
    valid: bool
    reasons: tuple[str, ...]
    protocol: EvaluationProtocol | None = None


def validate_evaluation_protocol_mapping(value: Any) -> EvaluationProtocolValidation:
    """Strictly reconstruct and hash-check a public evaluation protocol."""

    if not isinstance(value, Mapping):
        return EvaluationProtocolValidation(False, ("eval_protocol_must_be_object",))
    reasons: list[str] = []
    if value.get("schema_version") != EVALUATION_PROTOCOL_SCHEMA_VERSION:
        reasons.append("eval_protocol_schema_version_mismatch")
    version = value.get("eval_protocol_version")
    if not isinstance(version, str) or not version:
        reasons.append("eval_protocol_version_invalid")
    counts: dict[str, int] = {}
    for field_name in (
        "public_seed_count",
        "public_accept_seed_count",
        "sealed_validation_seed_count",
        "eval_num_envs",
    ):
        raw = value.get(field_name)
        if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
            reasons.append(f"eval_protocol_{field_name}_invalid")
        else:
            counts[field_name] = raw
    protocol: EvaluationProtocol | None = None
    if not reasons:
        protocol = EvaluationProtocol(
            eval_protocol_version=version,
            public_seed_count=counts["public_seed_count"],
            public_accept_seed_count=counts["public_accept_seed_count"],
            sealed_validation_seed_count=counts["sealed_validation_seed_count"],
            eval_num_envs=counts["eval_num_envs"],
        )
        declared_hash = value.get("eval_protocol_hash")
        if not isinstance(declared_hash, str) or not declared_hash:
            reasons.append("eval_protocol_hash_missing")
        elif declared_hash != protocol.protocol_hash:
            reasons.append("eval_protocol_hash_mismatch")
    return EvaluationProtocolValidation(
        valid=not reasons,
        reasons=tuple(dict.fromkeys(reasons)),
        protocol=protocol,
    )


__all__ = [
    "DEFAULT_EVAL_PROTOCOL_VERSION",
    "EVALUATION_PROTOCOL_SCHEMA_VERSION",
    "EvaluationProtocol",
    "EvaluationProtocolValidation",
    "validate_evaluation_protocol_mapping",
]
