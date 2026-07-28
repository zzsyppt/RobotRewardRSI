"""Task-neutral sealed-validation rollout and gate contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from true_rl_agent.evaluation.binding import (
    is_sha256_checksum,
    is_sha256_digest,
)
from true_rl_agent.evaluation.score_stats import validate_score_stats
from true_rl_agent.evaluation.subject import ExperimentSubject

SEALED_VALIDATION_EVIDENCE_SCHEMA_VERSION = "true_rl_agent.sealed_validation.evidence.v3"
SEALED_VALIDATION_EVIDENCE_KEY = "sealed_validation"
CANDIDATE_CONTAMINATION_FIELDS = (
    "candidate_code_loaded_in_eval",
    "candidate_package_loaded",
    "candidate_package_loaded_in_eval",
    "candidate_path_argument_present",
)
SEALED_GATE_REQUIRED_CONTAMINATION_FIELDS = (
    "candidate_code_loaded_in_eval",
    "candidate_package_loaded_in_eval",
    "candidate_path_argument_present",
)
SEALED_GATE_REQUIRED_PASS_STATUS_FIELDS = (
    "checkpoint_safety_status",
    "candidate_isolation_status",
    "checkpoint_hash_match_status",
    "eval_protocol_hash_match_status",
    "evaluation_binding_hash_match_status",
    "holdout_rollout_status",
)
_PASS_STATUS_FAILURE_REASONS = {
    "checkpoint_safety_status": "checkpoint safety failed",
    "candidate_isolation_status": "candidate isolation failed",
    "checkpoint_hash_match_status": "checkpoint hash match failed",
    "eval_protocol_hash_match_status": "eval protocol hash match failed",
    "evaluation_binding_hash_match_status": "evaluation binding hash match failed",
    "holdout_rollout_status": "sealed holdout rollout failed",
}


@dataclass(frozen=True)
class SealedRolloutEvidenceValidation:
    complete: bool
    passed: bool
    status: str
    reasons: tuple[str, ...]
    expected_seed_count: int | None = None
    observed_seed_count: int | None = None
    eval_protocol_hash: str | None = None
    evaluation_binding_hash: str | None = None
    evaluation_subject: Mapping[str, str] = field(default_factory=dict)
    subject_manifest_hash: str | None = None
    program_source_checksum: str | None = None
    training_binding_hash: str | None = None
    capability_profile_hash: str | None = None
    checkpoint_hash: str | None = None
    score_stats: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SealedValidationGateValidation:
    valid: bool
    status: str
    reasons: tuple[str, ...]

    def raise_for_failure(self) -> None:
        if self.valid:
            return
        raise ValueError(self.reasons[0] if self.reasons else "sealed validation gate failed")


def build_sealed_rollout_evidence(
    *,
    score_stats: Mapping[str, Any],
    expected_seed_count: int,
    eval_protocol_hash: str,
    evaluation_binding_hash: str,
    evaluation_subject: ExperimentSubject,
    checkpoint_hash: str,
    task_validation_status: str,
    reason: str,
    subject_manifest_hash: str | None = None,
    program_source_checksum: str | None = None,
    training_binding_hash: str | None = None,
    capability_profile_hash: str,
) -> dict[str, Any]:
    """Bind a task-owned aggregate result to runner-owned eval inputs.

    Task code decides the behavior criterion and supplies only its aggregate
    pass/fail status.  This shared layer binds that result to the sealed
    checkpoint, protocol, expected count, and complete finite score stats.
    """

    evidence = {
        "schema_version": SEALED_VALIDATION_EVIDENCE_SCHEMA_VERSION,
        "status": task_validation_status,
        "reason": reason,
        "expected_seed_count": expected_seed_count,
        "observed_seed_count": score_stats.get("count"),
        "eval_protocol_hash": eval_protocol_hash,
        "evaluation_binding_hash": evaluation_binding_hash,
        "evaluation_subject": evaluation_subject.to_public_dict(),
        "training_binding_hash": training_binding_hash,
        "program_source_checksum": program_source_checksum,
        "capability_profile_hash": capability_profile_hash,
        "checkpoint_hash": checkpoint_hash,
        "score_stats": dict(score_stats),
    }
    if subject_manifest_hash is not None:
        evidence["subject_manifest_hash"] = subject_manifest_hash
    validation = validate_sealed_rollout_evidence(
        evidence,
        expected_seed_count=expected_seed_count,
        expected_eval_protocol_hash=eval_protocol_hash,
        expected_evaluation_binding_hash=evaluation_binding_hash,
        expected_evaluation_subject=evaluation_subject,
        expected_subject_manifest_hash=subject_manifest_hash,
        expected_program_source_checksum=program_source_checksum,
        expected_training_binding_hash=training_binding_hash,
        expected_capability_profile_hash=capability_profile_hash,
        expected_checkpoint_hash=checkpoint_hash,
    )
    if not validation.complete:
        detail = ", ".join(validation.reasons) or "invalid sealed rollout evidence"
        raise ValueError(detail)
    return evidence


def validate_sealed_rollout_evidence(
    evidence: Mapping[str, Any] | Any,
    *,
    expected_seed_count: int | None = None,
    expected_eval_protocol_hash: str | None = None,
    expected_evaluation_binding_hash: str | None = None,
    expected_evaluation_subject: ExperimentSubject | None = None,
    expected_subject_manifest_hash: str | None = None,
    expected_program_source_checksum: str | None = None,
    expected_training_binding_hash: str | None = None,
    expected_capability_profile_hash: str | None = None,
    expected_checkpoint_hash: str | None = None,
) -> SealedRolloutEvidenceValidation:
    if not isinstance(evidence, Mapping) or not evidence:
        return SealedRolloutEvidenceValidation(
            complete=False,
            passed=False,
            status="missing",
            reasons=("sealed_validation evidence must be a non-empty object",),
        )

    reasons: list[str] = []
    allowed_fields = {
        "schema_version",
        "status",
        "reason",
        "expected_seed_count",
        "observed_seed_count",
        "eval_protocol_hash",
        "evaluation_binding_hash",
        "evaluation_subject",
        "subject_manifest_hash",
        "program_source_checksum",
        "training_binding_hash",
        "capability_profile_hash",
        "checkpoint_hash",
        "score_stats",
    }
    required_fields = allowed_fields - {"subject_manifest_hash"}
    extra_fields = sorted(set(evidence) - allowed_fields)
    missing_fields = sorted(required_fields - set(evidence))
    if extra_fields:
        reasons.append(
            "sealed_validation unsupported field(s): " + ", ".join(extra_fields)
        )
    if missing_fields:
        reasons.append(
            "sealed_validation missing field(s): " + ", ".join(missing_fields)
        )
    schema_version = evidence.get("schema_version")
    if schema_version != SEALED_VALIDATION_EVIDENCE_SCHEMA_VERSION:
        reasons.append("sealed_validation schema_version mismatch")
    status = evidence.get("status")
    if status not in {"pass", "fail"}:
        reasons.append("sealed_validation status must be pass or fail")
    reason = evidence.get("reason")
    if not isinstance(reason, str) or not reason:
        reasons.append("sealed_validation reason must be non-empty")

    declared_expected = _positive_int(evidence.get("expected_seed_count"))
    observed = _positive_int(evidence.get("observed_seed_count"))
    if declared_expected is None:
        reasons.append("sealed_validation expected_seed_count must be positive")
    if observed is None:
        reasons.append("sealed_validation observed_seed_count must be positive")
    if declared_expected is not None and observed != declared_expected:
        reasons.append("sealed_validation observed_seed_count mismatch")
    if expected_seed_count is not None:
        if _positive_int(expected_seed_count) is None:
            reasons.append("expected sealed_validation seed count must be positive")
        elif declared_expected != expected_seed_count:
            reasons.append("sealed_validation expected_seed_count binding mismatch")

    eval_protocol_hash = _nonempty_text(evidence.get("eval_protocol_hash"))
    evaluation_binding_hash = _nonempty_text(
        evidence.get("evaluation_binding_hash")
    )
    checkpoint_hash = _nonempty_text(evidence.get("checkpoint_hash"))
    if eval_protocol_hash is None:
        reasons.append("sealed_validation eval_protocol_hash is required")
    elif not is_sha256_digest(eval_protocol_hash):
        reasons.append(
            "sealed_validation eval_protocol_hash must be a lowercase SHA-256 digest"
        )
    if evaluation_binding_hash is None:
        reasons.append("sealed_validation evaluation_binding_hash is required")
    elif not is_sha256_digest(evaluation_binding_hash):
        reasons.append(
            "sealed_validation evaluation_binding_hash must be a lowercase "
            "SHA-256 digest"
        )
    if checkpoint_hash is None:
        reasons.append("sealed_validation checkpoint_hash is required")
    elif not is_sha256_checksum(checkpoint_hash):
        reasons.append(
            "sealed_validation checkpoint_hash must use "
            "sha256:<lowercase SHA-256 digest>"
        )
    if expected_eval_protocol_hash is not None:
        if not is_sha256_digest(expected_eval_protocol_hash):
            reasons.append(
                "expected sealed_validation eval protocol hash must be a lowercase "
                "SHA-256 digest"
            )
        elif eval_protocol_hash != expected_eval_protocol_hash:
            reasons.append("sealed_validation eval_protocol_hash binding mismatch")
    if expected_evaluation_binding_hash is not None:
        if not is_sha256_digest(expected_evaluation_binding_hash):
            reasons.append(
                "expected sealed_validation evaluation binding hash must be a "
                "lowercase SHA-256 digest"
            )
        elif evaluation_binding_hash != expected_evaluation_binding_hash:
            reasons.append(
                "sealed_validation evaluation_binding_hash binding mismatch"
            )

    evaluation_subject: ExperimentSubject | None = None
    subject_manifest_hash = _nonempty_text(
        evidence.get("subject_manifest_hash")
    )
    training_binding_hash = _nonempty_text(
        evidence.get("training_binding_hash")
    )
    program_source_checksum = _nonempty_text(
        evidence.get("program_source_checksum")
    )
    capability_profile_hash = _nonempty_text(
        evidence.get("capability_profile_hash")
    )
    try:
        evaluation_subject = ExperimentSubject.from_mapping(
            evidence.get("evaluation_subject")
        )
    except (TypeError, ValueError) as exc:
        reasons.append(f"sealed_validation evaluation_subject invalid: {exc}")
    if evaluation_subject is not None:
        if evaluation_subject.kind in {"baseline", "reference"}:
            if not is_sha256_digest(subject_manifest_hash):
                reasons.append(
                    "sealed_validation subject_manifest_hash is required"
                )
        elif "subject_manifest_hash" in evidence and subject_manifest_hash is None:
            reasons.append("sealed_validation subject_manifest_hash is invalid")
        if evaluation_subject.kind in {"candidate", "reference"}:
            if not is_sha256_checksum(program_source_checksum):
                reasons.append(
                    "sealed_validation program_source_checksum is required"
                )
            if not is_sha256_digest(training_binding_hash):
                reasons.append(
                    "sealed_validation training_binding_hash is required"
                )
        elif training_binding_hash is not None or program_source_checksum is not None:
            reasons.append(
                "sealed_validation baseline subject cannot carry program lineage"
            )
    if not is_sha256_digest(capability_profile_hash):
        reasons.append("sealed_validation capability_profile_hash is required")
    if (
        expected_evaluation_subject is not None
        and evaluation_subject != expected_evaluation_subject
    ):
        reasons.append("sealed_validation evaluation_subject binding mismatch")
    if expected_subject_manifest_hash is not None:
        if not is_sha256_digest(expected_subject_manifest_hash):
            reasons.append(
                "expected sealed_validation subject manifest hash must be SHA-256"
            )
        elif subject_manifest_hash != expected_subject_manifest_hash:
            reasons.append(
                "sealed_validation subject_manifest_hash binding mismatch"
            )
    if expected_training_binding_hash is not None:
        if not is_sha256_digest(expected_training_binding_hash):
            reasons.append("expected training binding hash must be SHA-256")
        elif training_binding_hash != expected_training_binding_hash:
            reasons.append("sealed_validation training_binding_hash binding mismatch")
    if expected_program_source_checksum is not None:
        if not is_sha256_checksum(expected_program_source_checksum):
            reasons.append("expected program source checksum must be SHA-256")
        elif program_source_checksum != expected_program_source_checksum:
            reasons.append("sealed_validation program_source_checksum binding mismatch")
    if expected_capability_profile_hash is not None:
        if not is_sha256_digest(expected_capability_profile_hash):
            reasons.append("expected capability profile hash must be SHA-256")
        elif capability_profile_hash != expected_capability_profile_hash:
            reasons.append("sealed_validation capability_profile_hash binding mismatch")
    if expected_checkpoint_hash is not None:
        if not is_sha256_checksum(expected_checkpoint_hash):
            reasons.append(
                "expected sealed_validation checkpoint hash must use "
                "sha256:<lowercase SHA-256 digest>"
            )
        elif checkpoint_hash != expected_checkpoint_hash:
            reasons.append("sealed_validation checkpoint_hash binding mismatch")

    stats_validation = validate_score_stats(
        evidence.get("score_stats"),
        minimum_count=declared_expected or 1,
        require_std=True,
        require_standard_error=True,
        require_protocol_hash=True,
    )
    reasons.extend(f"sealed_validation {item}" for item in stats_validation.reasons)
    if stats_validation.count is not None and stats_validation.count != observed:
        reasons.append("sealed_validation score_stats count mismatch")
    if stats_validation.eval_protocol_hash != eval_protocol_hash:
        reasons.append("sealed_validation score_stats protocol hash mismatch")

    deduped = tuple(dict.fromkeys(reasons))
    canonical_status = status if status in {"pass", "fail"} else "missing"
    return SealedRolloutEvidenceValidation(
        complete=not deduped,
        passed=not deduped and canonical_status == "pass",
        status=canonical_status,
        reasons=deduped,
        expected_seed_count=declared_expected,
        observed_seed_count=observed,
        eval_protocol_hash=eval_protocol_hash,
        evaluation_binding_hash=evaluation_binding_hash,
        evaluation_subject=(
            evaluation_subject.to_public_dict()
            if evaluation_subject is not None
            else {}
        ),
        subject_manifest_hash=subject_manifest_hash,
        program_source_checksum=program_source_checksum,
        training_binding_hash=training_binding_hash,
        capability_profile_hash=capability_profile_hash,
        checkpoint_hash=checkpoint_hash,
        score_stats=dict(stats_validation.canonical) if stats_validation.valid else {},
    )


def build_sealed_validation_gate(
    *,
    rollout_evidence: Mapping[str, Any],
    checkpoint_safety_status: str,
    checkpoint_hash_match_status: str,
    candidate_isolation_gate: Mapping[str, Any],
    expected_eval_protocol_hash: str,
    expected_evaluation_binding_hash: str,
) -> dict[str, Any]:
    """Derive the public-safe gate from canonical rollout and safety evidence."""

    rollout = validate_sealed_rollout_evidence(
        rollout_evidence,
        expected_eval_protocol_hash=expected_eval_protocol_hash,
        expected_evaluation_binding_hash=expected_evaluation_binding_hash,
    )
    gate = {
        "checkpoint_safety_status": checkpoint_safety_status,
        "candidate_isolation_status": candidate_isolation_gate.get(
            "candidate_isolation_status"
        ),
        "checkpoint_hash_match_status": checkpoint_hash_match_status,
        "eval_protocol_hash_match_status": (
            "pass"
            if (
                is_sha256_digest(expected_eval_protocol_hash)
                and rollout.eval_protocol_hash == expected_eval_protocol_hash
            )
            else "fail"
        ),
        "evaluation_binding_hash_match_status": (
            "pass"
            if (
                is_sha256_digest(expected_evaluation_binding_hash)
                and rollout.evaluation_binding_hash
                == expected_evaluation_binding_hash
            )
            else "fail"
        ),
        "holdout_rollout_status": "pass" if rollout.passed else "fail",
    }
    for key in SEALED_GATE_REQUIRED_CONTAMINATION_FIELDS:
        gate[key] = candidate_isolation_gate.get(key)
    failed = [
        key
        for key in SEALED_GATE_REQUIRED_PASS_STATUS_FIELDS
        if gate.get(key) != "pass"
    ]
    contaminated = any(gate.get(key) is not False for key in SEALED_GATE_REQUIRED_CONTAMINATION_FIELDS)
    gate["status"] = "fail" if failed or contaminated else "pass"
    gate["reason"] = (
        "sealed_validation_complete"
        if gate["status"] == "pass"
        else (
            rollout_evidence.get("reason")
            if gate["holdout_rollout_status"] == "fail"
            else "sealed_validation_safety_failed"
        )
    )
    return gate


def validate_sealed_validation_gate_evidence(
    gate: Mapping[str, Any] | Any,
    *,
    rollout_evidence: Mapping[str, Any] | Any = None,
    eval_protocol_hash: str | None = None,
    evaluation_binding_hash: str | None = None,
    expected_seed_count: int | None = None,
    expected_checkpoint_hash: str | None = None,
) -> SealedValidationGateValidation:
    """Validate complete gate evidence without requiring the gate to pass."""

    required_pass_status_fields = _gate_required_pass_status_fields(
        rollout_evidence=rollout_evidence,
        expected_evaluation_binding_hash=evaluation_binding_hash,
    )
    reasons = _gate_shape_reasons(
        gate,
        required_pass_status_fields=required_pass_status_fields,
    )
    rollout = validate_sealed_rollout_evidence(
        rollout_evidence,
        expected_seed_count=expected_seed_count,
        expected_eval_protocol_hash=eval_protocol_hash,
        expected_evaluation_binding_hash=evaluation_binding_hash,
        expected_checkpoint_hash=expected_checkpoint_hash,
    )
    reasons.extend(rollout.reasons)
    if isinstance(gate, Mapping):
        expected_holdout_status = "pass" if rollout.passed else "fail"
        if gate.get("holdout_rollout_status") != expected_holdout_status:
            reasons.append("sealed_validation_gate holdout_rollout_status mismatch")
    deduped = tuple(dict.fromkeys(reasons))
    status = _gate_status(gate)
    return SealedValidationGateValidation(
        valid=not deduped,
        status=status,
        reasons=deduped,
    )


def validate_sealed_validation_gate_contract(
    gate: Mapping[str, Any] | Any,
    *,
    rollout_evidence: Mapping[str, Any] | Any = None,
    eval_protocol_hash: str | None = None,
    evaluation_binding_hash: str | None = None,
    expected_seed_count: int | None = None,
    expected_checkpoint_hash: str | None = None,
) -> SealedValidationGateValidation:
    required_pass_status_fields = _gate_required_pass_status_fields(
        rollout_evidence=rollout_evidence,
        expected_evaluation_binding_hash=evaluation_binding_hash,
    )
    evidence_validation = validate_sealed_validation_gate_evidence(
        gate,
        rollout_evidence=rollout_evidence,
        eval_protocol_hash=eval_protocol_hash,
        evaluation_binding_hash=evaluation_binding_hash,
        expected_seed_count=expected_seed_count,
        expected_checkpoint_hash=expected_checkpoint_hash,
    )
    reasons = list(evidence_validation.reasons)
    if isinstance(gate, Mapping):
        for key in CANDIDATE_CONTAMINATION_FIELDS:
            if key in SEALED_GATE_REQUIRED_CONTAMINATION_FIELDS or key in gate:
                value = gate.get(key)
            else:
                continue
            if value is not False:
                reasons.append(f"candidate isolation failed: sealed_validation_gate {key} is not false")
        for key in required_pass_status_fields:
            if gate.get(key) != "pass":
                reasons.append(_PASS_STATUS_FAILURE_REASONS[key])
        if gate.get("status") != "pass":
            reasons.append("sealed validation gate failed")
    deduped = tuple(dict.fromkeys(reasons))
    return SealedValidationGateValidation(
        valid=not deduped,
        status="pass" if not deduped else ("fail" if gate else "missing"),
        reasons=deduped,
    )


def sealed_validation_gate_status(
    gate: Mapping[str, Any] | Any,
    *,
    rollout_evidence: Mapping[str, Any] | Any = None,
    eval_protocol_hash: str | None = None,
    evaluation_binding_hash: str | None = None,
) -> str:
    return validate_sealed_validation_gate_contract(
        gate,
        rollout_evidence=rollout_evidence,
        eval_protocol_hash=eval_protocol_hash,
        evaluation_binding_hash=evaluation_binding_hash,
    ).status


def _gate_shape_reasons(
    gate: Mapping[str, Any] | Any,
    *,
    required_pass_status_fields: tuple[str, ...] = (
        SEALED_GATE_REQUIRED_PASS_STATUS_FIELDS
    ),
) -> list[str]:
    if not isinstance(gate, Mapping):
        return ["sealed_validation_gate must be an object"]
    reasons: list[str] = []
    if gate.get("status") not in {"pass", "fail"}:
        reasons.append("sealed_validation_gate status must be pass or fail")
    if not isinstance(gate.get("reason"), str) or not gate.get("reason"):
        reasons.append("sealed_validation_gate reason must be non-empty")
    for key in SEALED_GATE_REQUIRED_CONTAMINATION_FIELDS:
        if not isinstance(gate.get(key), bool):
            reasons.append(f"sealed_validation_gate {key} must be boolean")
    for key in CANDIDATE_CONTAMINATION_FIELDS:
        if key in SEALED_GATE_REQUIRED_CONTAMINATION_FIELDS or key not in gate:
            continue
        if not isinstance(gate.get(key), bool):
            reasons.append(f"sealed_validation_gate {key} must be boolean")
    for key in required_pass_status_fields:
        if gate.get(key) not in {"pass", "fail"}:
            reasons.append(f"sealed_validation_gate {key} must be pass or fail")
    return reasons


def _gate_required_pass_status_fields(
    *,
    rollout_evidence: Mapping[str, Any] | Any,
    expected_evaluation_binding_hash: str | None,
) -> tuple[str, ...]:
    del rollout_evidence, expected_evaluation_binding_hash
    return SEALED_GATE_REQUIRED_PASS_STATUS_FIELDS


def _gate_status(gate: Any) -> str:
    if not isinstance(gate, Mapping):
        return "missing"
    status = gate.get("status")
    return status if status in {"pass", "fail"} else "missing"


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _nonempty_text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


__all__ = [
    "CANDIDATE_CONTAMINATION_FIELDS",
    "SEALED_GATE_REQUIRED_CONTAMINATION_FIELDS",
    "SEALED_GATE_REQUIRED_PASS_STATUS_FIELDS",
    "SEALED_VALIDATION_EVIDENCE_KEY",
    "SEALED_VALIDATION_EVIDENCE_SCHEMA_VERSION",
    "SealedRolloutEvidenceValidation",
    "SealedValidationGateValidation",
    "build_sealed_rollout_evidence",
    "build_sealed_validation_gate",
    "sealed_validation_gate_status",
    "validate_sealed_rollout_evidence",
    "validate_sealed_validation_gate_contract",
    "validate_sealed_validation_gate_evidence",
]
