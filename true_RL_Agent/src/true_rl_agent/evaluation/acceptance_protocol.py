"""Task-neutral fixed acceptance policy contract."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping

ACCEPTANCE_PROTOCOL_SCHEMA_VERSION = "true_rl_agent.acceptance.protocol.v2"
NOISE_AWARE_DELTA_RULE = "bootstrap_floor_then_noise_aware_delta_v1"
PUBLIC_SCORE_BINDING_ACCEPTANCE_MEAN = "acceptance_score_mean"


@dataclass(frozen=True)
class AcceptanceProtocol:
    """Runner-owned acceptance constants supplied by a task plugin."""

    protocol_id: str
    score_source: str
    min_score_count: int
    absolute_accept_floor: float
    min_delta: float
    public_score_binding: str
    decision_rule: str = NOISE_AWARE_DELTA_RULE

    def __post_init__(self) -> None:
        if not isinstance(self.protocol_id, str) or not self.protocol_id:
            raise ValueError("acceptance protocol_id must be non-empty")
        if not isinstance(self.score_source, str) or not self.score_source:
            raise ValueError("acceptance score_source must be non-empty")
        if self.public_score_binding != PUBLIC_SCORE_BINDING_ACCEPTANCE_MEAN:
            raise ValueError(
                "unsupported acceptance public_score_binding: "
                f"{self.public_score_binding}"
            )
        if (
            isinstance(self.min_score_count, bool)
            or not isinstance(self.min_score_count, int)
            or self.min_score_count <= 0
        ):
            raise ValueError("acceptance min_score_count must be a positive integer")
        for name in ("absolute_accept_floor", "min_delta"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError(f"acceptance {name} must be a finite non-negative number")
        if self.decision_rule != NOISE_AWARE_DELTA_RULE:
            raise ValueError(f"unsupported acceptance decision_rule: {self.decision_rule}")

    @classmethod
    def from_mapping(cls, payload: Any) -> "AcceptanceProtocol":
        if not isinstance(payload, Mapping):
            raise ValueError("acceptance protocol must be a mapping")
        if payload.get("schema_version") != ACCEPTANCE_PROTOCOL_SCHEMA_VERSION:
            raise ValueError("acceptance protocol schema_version mismatch")
        protocol = cls(
            protocol_id=_required_text(payload, "protocol_id"),
            score_source=_required_text(payload, "score_source"),
            min_score_count=_required_positive_int(payload, "min_score_count"),
            absolute_accept_floor=_required_number(payload, "absolute_accept_floor"),
            min_delta=_required_number(payload, "min_delta"),
            public_score_binding=_required_text(payload, "public_score_binding"),
            decision_rule=_required_text(payload, "decision_rule"),
        )
        declared_hash = _required_text(payload, "acceptance_protocol_hash")
        if declared_hash != protocol.protocol_hash:
            raise ValueError("acceptance protocol hash mismatch")
        return protocol

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
            "acceptance_protocol_hash": self.protocol_hash,
        }

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "schema_version": ACCEPTANCE_PROTOCOL_SCHEMA_VERSION,
            "protocol_id": self.protocol_id,
            "score_source": self.score_source,
            "min_score_count": int(self.min_score_count),
            "absolute_accept_floor": float(self.absolute_accept_floor),
            "min_delta": float(self.min_delta),
            "public_score_binding": self.public_score_binding,
            "decision_rule": self.decision_rule,
        }


def noise_aware_threshold(
    protocol: AcceptanceProtocol,
    *,
    candidate_standard_error: float | None,
    best_standard_error: float | None,
) -> float:
    """Compute the one canonical threshold used by decisions and ledger replay."""

    pooled = None
    if candidate_standard_error is not None and best_standard_error is not None:
        pooled = math.sqrt(candidate_standard_error**2 + best_standard_error**2)
    elif candidate_standard_error is not None:
        pooled = candidate_standard_error
    return max(protocol.min_delta, pooled if pooled is not None else protocol.min_delta)


def _required_text(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"acceptance {key} must be non-empty")
    return value


def _required_positive_int(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"acceptance {key} must be a positive integer")
    return value


def _required_number(payload: Mapping[str, Any], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"acceptance {key} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"acceptance {key} must be finite")
    return parsed


__all__ = [
    "ACCEPTANCE_PROTOCOL_SCHEMA_VERSION",
    "AcceptanceProtocol",
    "NOISE_AWARE_DELTA_RULE",
    "PUBLIC_SCORE_BINDING_ACCEPTANCE_MEAN",
    "noise_aware_threshold",
]
