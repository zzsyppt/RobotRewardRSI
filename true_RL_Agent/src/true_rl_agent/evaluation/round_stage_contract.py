"""Task-neutral stage consistency for public ledger records."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

PROVIDER_FAILURE_STAGE = "provider_failure"
STATIC_AUTHORING_FAILURE_STAGE = "static_authoring_failure"
VERIFIER_FAILURE_STAGE = "verifier_failure"
PRE_TRAIN_REJECTION_STAGE = "pre_train_rejection"
TRAIN_FAILURE_STAGE = "train_failure"
POST_TRAIN_FAILURE_STAGE = "post_train_failure"
EVALUATED_STAGE = "evaluated"

_STRUCTURED_FIELDS = (
    "candidate",
    "verifier",
    "jobs",
    "train_result",
    "public_result",
    "runner_error",
    "provider_failure",
    "static_authoring_failure",
)


@dataclass(frozen=True)
class RoundStageValidation:
    stage: str
    valid: bool
    reasons: tuple[str, ...]
    candidate_authored: bool
    trainable_candidate: bool
    train_result_present: bool
    public_result_present: bool


def validate_round_stage(record: Mapping[str, Any]) -> RoundStageValidation:
    """Derive a terminal stage from evidence and reject backward stage edges."""

    present = {
        name: isinstance(record.get(name), Mapping) and bool(record.get(name))
        for name in _STRUCTURED_FIELDS
    }
    reasons = [
        f"round_stage_field_invalid:{name}"
        for name in _STRUCTURED_FIELDS
        if record.get(name) is not None
        and not isinstance(record.get(name), Mapping)
    ]
    reasons.extend(
        f"round_stage_empty_evidence:{name}"
        for name in _STRUCTURED_FIELDS
        if isinstance(record.get(name), Mapping) and not record.get(name)
    )

    if present["provider_failure"]:
        stage = PROVIDER_FAILURE_STAGE
        _require_absent(
            present,
            (
                "static_authoring_failure",
                "candidate",
                "verifier",
                "jobs",
                "train_result",
                "public_result",
                "runner_error",
            ),
            reasons,
        )
    elif present["static_authoring_failure"]:
        stage = STATIC_AUTHORING_FAILURE_STAGE
        _require_present(present, ("candidate",), reasons)
        _require_absent(
            present,
            (
                "verifier",
                "jobs",
                "train_result",
                "public_result",
                "runner_error",
            ),
            reasons,
        )
    else:
        _require_present(present, ("candidate", "verifier"), reasons)
        verifier = _mapping(record.get("verifier"))
        verifier_status = verifier.get("status")
        if present["verifier"] and (
            not isinstance(verifier_status, str) or not verifier_status
        ):
            reasons.append("round_stage_verifier_status_invalid")

        if present["verifier"] and verifier_status != "pass":
            stage = VERIFIER_FAILURE_STAGE
            _require_absent(
                present,
                ("jobs", "train_result", "public_result", "runner_error"),
                reasons,
            )
        else:
            _require_present(present, ("jobs",), reasons)
            if present["public_result"]:
                stage = EVALUATED_STAGE
                _require_present(present, ("train_result",), reasons)
                _require_absent(present, ("runner_error",), reasons)
            elif present["train_result"]:
                stage = POST_TRAIN_FAILURE_STAGE
                _require_present(present, ("runner_error",), reasons)
            elif present["runner_error"]:
                stage = TRAIN_FAILURE_STAGE
            else:
                stage = PRE_TRAIN_REJECTION_STAGE

    deduped = tuple(dict.fromkeys(reasons))
    trainable = stage in {
        TRAIN_FAILURE_STAGE,
        POST_TRAIN_FAILURE_STAGE,
        EVALUATED_STAGE,
    }
    return RoundStageValidation(
        stage=stage,
        valid=not deduped,
        reasons=deduped,
        candidate_authored=present["candidate"],
        trainable_candidate=trainable,
        train_result_present=present["train_result"],
        public_result_present=present["public_result"],
    )


def _require_present(
    present: Mapping[str, bool],
    fields: tuple[str, ...],
    reasons: list[str],
) -> None:
    reasons.extend(
        f"round_stage_missing_evidence:{field}"
        for field in fields
        if not present[field]
    )


def _require_absent(
    present: Mapping[str, bool],
    fields: tuple[str, ...],
    reasons: list[str],
) -> None:
    reasons.extend(
        f"round_stage_unexpected_evidence:{field}"
        for field in fields
        if present[field]
    )


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


__all__ = [
    "EVALUATED_STAGE",
    "POST_TRAIN_FAILURE_STAGE",
    "PRE_TRAIN_REJECTION_STAGE",
    "PROVIDER_FAILURE_STAGE",
    "RoundStageValidation",
    "STATIC_AUTHORING_FAILURE_STAGE",
    "TRAIN_FAILURE_STAGE",
    "VERIFIER_FAILURE_STAGE",
    "validate_round_stage",
]
