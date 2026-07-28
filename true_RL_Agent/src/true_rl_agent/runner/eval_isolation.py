"""Task-neutral eval isolation contract helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from true_rl_agent.evaluation.sealed_validation import (
    CANDIDATE_CONTAMINATION_FIELDS,
    validate_sealed_validation_gate_contract,
)

CANDIDATE_PATH_ARGUMENTS = ("--candidate", "--candidate-path", "--candidate-dir")


def reject_candidate_path_arguments(argv: Sequence[str]) -> None:
    """Reject candidate package path flags at the eval boundary."""

    for part in argv:
        if part in CANDIDATE_PATH_ARGUMENTS or any(part.startswith(f"{flag}=") for flag in CANDIDATE_PATH_ARGUMENTS):
            raise ValueError(f"candidate path argument is not allowed in eval: {part}")


def validate_candidate_id(candidate_id: str) -> str:
    """Validate an exact candidate id, rejecting coercion and path-like values."""

    if not isinstance(candidate_id, str):
        raise ValueError("candidate_id must be a string")
    value = candidate_id.strip()
    if not value:
        raise ValueError("candidate_id must be non-empty")
    if value != candidate_id:
        raise ValueError("candidate_id must not contain surrounding whitespace")
    if "://" in value or "/" in value or "\\" in value:
        raise ValueError("candidate_id must be an identifier, not a path")
    return value


def build_candidate_isolation_gate(
    *,
    candidate_code_loaded: bool = False,
    candidate_package_loaded: bool = False,
    candidate_path_argument_present: bool = False,
) -> dict[str, Any]:
    """Build the public-safe sealed validation candidate-isolation gate."""

    failed = candidate_code_loaded or candidate_package_loaded or candidate_path_argument_present
    return {
        "status": "fail" if failed else "pass",
        "candidate_isolation_status": "fail" if failed else "pass",
        "candidate_code_loaded_in_eval": bool(candidate_code_loaded),
        "candidate_package_loaded_in_eval": bool(candidate_package_loaded),
        "candidate_path_argument_present": bool(candidate_path_argument_present),
        "reason": "candidate_isolation_failed"
        if failed
        else "candidate_path_not_accepted_and_candidate_code_not_loaded",
    }


def reject_candidate_contamination_flags(
    record: Mapping[str, Any],
    *,
    source: str,
    required_fields: Sequence[str] = (),
) -> None:
    """Reject candidate contamination evidence in a public eval record."""

    for key in required_fields:
        if key not in record:
            raise ValueError(f"{source} {key} is required")
    for key in CANDIDATE_CONTAMINATION_FIELDS:
        if key not in record:
            continue
        value = record[key]
        if not isinstance(value, bool):
            raise ValueError(f"{source} {key} must be boolean")
        if value:
            raise ValueError(f"candidate isolation failed: {source} {key} is true")


def validate_sealed_validation_gate(
    gate: Mapping[str, Any],
    *,
    rollout_evidence: Mapping[str, Any],
    eval_protocol_hash: str,
    evaluation_binding_hash: str | None = None,
) -> Mapping[str, Any]:
    """Fail closed when a sealed validation gate reports candidate contamination."""

    validate_sealed_validation_gate_contract(
        gate,
        rollout_evidence=rollout_evidence,
        eval_protocol_hash=eval_protocol_hash,
        evaluation_binding_hash=evaluation_binding_hash,
    ).raise_for_failure()
    return gate
