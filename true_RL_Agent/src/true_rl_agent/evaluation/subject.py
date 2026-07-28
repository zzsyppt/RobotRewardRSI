"""Task-neutral identity for a train or evaluation subject."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

EXPERIMENT_SUBJECT_SCHEMA_VERSION = "true_rl_agent.experiment.subject.v2"
EXPERIMENT_SUBJECT_KINDS = frozenset({"baseline", "candidate", "reference"})
_EVALUATION_SUBJECT_ID_PATTERN = re.compile(
    r"[A-Za-z0-9_-][A-Za-z0-9._-]{0,127}\Z"
)


def validate_experiment_subject_id(subject_id: str) -> str:
    """Reject empty and path-like identities before they reach argv or artifacts."""

    if not isinstance(subject_id, str):
        raise ValueError("experiment subject_id must be a string")
    value = subject_id.strip()
    if not value:
        raise ValueError("experiment subject_id must be non-empty")
    if value != subject_id:
        raise ValueError("experiment subject_id must not contain surrounding whitespace")
    if _EVALUATION_SUBJECT_ID_PATTERN.fullmatch(value) is None:
        raise ValueError(
            "experiment subject_id must be a bounded ASCII identifier"
        )
    return value


@dataclass(frozen=True)
class ExperimentSubject:
    """The sole public identity for candidate, baseline, and reference jobs."""

    kind: str
    subject_id: str

    def __post_init__(self) -> None:
        if self.kind not in EXPERIMENT_SUBJECT_KINDS:
            raise ValueError(f"unsupported experiment subject kind: {self.kind}")
        validate_experiment_subject_id(self.subject_id)

    @classmethod
    def candidate(cls, candidate_id: str) -> "ExperimentSubject":
        return cls(kind="candidate", subject_id=candidate_id)

    @classmethod
    def baseline(cls, baseline_id: str) -> "ExperimentSubject":
        return cls(kind="baseline", subject_id=baseline_id)

    @classmethod
    def reference(cls, reference_id: str) -> "ExperimentSubject":
        return cls(kind="reference", subject_id=reference_id)

    @classmethod
    def from_mapping(cls, value: Any) -> "ExperimentSubject":
        if not isinstance(value, Mapping):
            raise ValueError("experiment subject must be an object")
        expected_keys = {"schema_version", "kind", "id"}
        if set(value) != expected_keys:
            raise ValueError("experiment subject fields mismatch")
        if value.get("schema_version") != EXPERIMENT_SUBJECT_SCHEMA_VERSION:
            raise ValueError("experiment subject schema_version mismatch")
        return cls(kind=value.get("kind"), subject_id=value.get("id"))

    def to_public_dict(self) -> dict[str, str]:
        return {
            "schema_version": EXPERIMENT_SUBJECT_SCHEMA_VERSION,
            "kind": self.kind,
            "id": self.subject_id,
        }


__all__ = [
    "EXPERIMENT_SUBJECT_KINDS",
    "EXPERIMENT_SUBJECT_SCHEMA_VERSION",
    "ExperimentSubject",
    "validate_experiment_subject_id",
]
