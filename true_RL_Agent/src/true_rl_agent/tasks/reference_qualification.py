"""Task-neutral carrier for task-owned reference qualification evidence."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from true_rl_agent.evaluation.binding import canonical_json_hash, require_sha256_digest
from true_rl_agent.evaluation.public_result import PublicEvalResult

REFERENCE_QUALIFICATION_GATE_SCHEMA_VERSION = (
    "true_rl_agent.reference_qualification_gate.v1"
)


@dataclass(frozen=True)
class ReferenceQualificationResult:
    public_result: PublicEvalResult
    gate: Mapping[str, Any]
    private_evidence: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.public_result, PublicEvalResult):
            raise TypeError("reference qualification public_result is invalid")
        gate = _validate_gate(self.gate)
        if not isinstance(self.private_evidence, Mapping):
            raise TypeError("reference qualification private evidence must be an object")
        evidence = dict(self.private_evidence)
        declared_hash = require_sha256_digest(
            evidence.get("evidence_hash"),
            "reference qualification private evidence hash",
        )
        if declared_hash != canonical_json_hash(
            {key: value for key, value in evidence.items() if key != "evidence_hash"}
        ):
            raise ValueError("reference qualification private evidence hash mismatch")
        if gate["evidence_hash"] != declared_hash:
            raise ValueError("reference qualification gate evidence hash mismatch")
        projected_gate = self.public_result.metrics.get(
            "reference_capability_gate"
        )
        if not isinstance(projected_gate, Mapping) or dict(projected_gate) != gate:
            raise ValueError("reference qualification public gate mismatch")
        acceptance = self.public_result.metrics.get("acceptance")
        if not isinstance(acceptance, Mapping):
            raise ValueError("reference qualification acceptance evidence is missing")
        required_gates = acceptance.get("required_pass_gates")
        if (
            not isinstance(required_gates, list)
            or required_gates.count("reference_capability_gate") != 1
        ):
            raise ValueError(
                "reference qualification capability gate must be required exactly once"
            )
        object.__setattr__(self, "gate", MappingProxyType(gate))
        object.__setattr__(self, "private_evidence", MappingProxyType(evidence))

    @property
    def passed(self) -> bool:
        return self.gate["status"] == "pass"


def _validate_gate(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "status",
        "protocol_hash",
        "evidence_hash",
    }:
        raise ValueError("reference qualification gate fields mismatch")
    gate = dict(value)
    if gate["schema_version"] != REFERENCE_QUALIFICATION_GATE_SCHEMA_VERSION:
        raise ValueError("reference qualification gate schema mismatch")
    if gate["status"] not in {"pass", "fail"}:
        raise ValueError("reference qualification gate status is invalid")
    require_sha256_digest(
        gate["protocol_hash"],
        "reference qualification protocol hash",
    )
    require_sha256_digest(
        gate["evidence_hash"],
        "reference qualification evidence hash",
    )
    return gate


__all__ = [
    "REFERENCE_QUALIFICATION_GATE_SCHEMA_VERSION",
    "ReferenceQualificationResult",
]
