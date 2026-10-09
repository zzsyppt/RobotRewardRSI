"""Task-neutral scalar and status contracts for runner result ingestion."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

RESULT_STATUSES = frozenset({"pass", "fail"})


@dataclass(frozen=True)
class PublicEvalResultFields:
    """Canonical fields shared by runtime objects, runners, and ledger replay."""

    status: str
    public_score: float
    metrics: Mapping[str, Any]


def validate_result_status(value: Any, *, field_name: str = "status") -> str:
    """Return a canonical result status without coercing arbitrary values."""

    if not isinstance(value, str) or value not in RESULT_STATUSES:
        raise ValueError(f"{field_name} must be pass or fail")
    return value


def validate_finite_number(value: Any, *, field_name: str) -> float:
    """Return a finite float while rejecting booleans and coercible strings."""

    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field_name} must be a finite non-boolean number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{field_name} must be a finite non-boolean number")
    return parsed


def validate_public_eval_result_fields(
    *,
    status: Any,
    public_score: Any,
    metrics: Any,
    field_name: str = "public eval result",
) -> PublicEvalResultFields:
    """Validate and canonicalize the complete public eval result field set."""

    canonical_status = validate_result_status(
        status,
        field_name=f"{field_name} status",
    )
    canonical_score = validate_finite_number(
        public_score,
        field_name=f"{field_name} public_score",
    )
    if not isinstance(metrics, Mapping):
        raise ValueError(f"{field_name} metrics must be an object")
    return PublicEvalResultFields(
        status=canonical_status,
        public_score=canonical_score,
        metrics=metrics,
    )


def validate_public_eval_result_mapping(
    value: Any,
    *,
    field_name: str = "public eval result",
) -> PublicEvalResultFields:
    """Validate an untrusted mapping without coercing any scalar field."""

    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return validate_public_eval_result_fields(
        status=value.get("status"),
        public_score=value.get("public_score"),
        metrics=value.get("metrics"),
        field_name=field_name,
    )


__all__ = [
    "PublicEvalResultFields",
    "RESULT_STATUSES",
    "validate_finite_number",
    "validate_public_eval_result_fields",
    "validate_public_eval_result_mapping",
    "validate_result_status",
]
