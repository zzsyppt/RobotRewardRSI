"""Run-level eval audit derived from public ledger records."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from true_rl_agent.evaluation.acceptance_protocol import AcceptanceProtocol
from true_rl_agent.evaluation.acceptance_replay import (
    AcceptanceReplayEntry,
    replay_acceptance_ledger,
)
from true_rl_agent.evaluation.current_best import CurrentBestEvalStats
from true_rl_agent.evaluation.public_projection import assert_public_projection_safe
from true_rl_agent.evaluation.round_stage_contract import (
    EVALUATED_STAGE,
    PRE_TRAIN_REJECTION_STAGE,
)
from true_rl_agent.evaluation.score_stats import validate_score_stats


def build_eval_run_audit(
    ledger: Sequence[Mapping[str, Any]],
    *,
    acceptance_protocol: AcceptanceProtocol,
    initial_current_best: CurrentBestEvalStats,
    expected_session_genesis_hash: str,
) -> dict[str, Any]:
    """Build a compact public-safe audit of train/eval/decision connectivity."""

    replay = replay_acceptance_ledger(
        ledger,
        initial_current_best=initial_current_best,
        expected_session_genesis_hash=expected_session_genesis_hash,
        acceptance_protocol=acceptance_protocol,
    )
    replay.raise_for_invalid_session_bindings()
    rows = [
        _audit_record(record, replay_entry=entry)
        for record, entry in zip(ledger, replay.entries, strict=True)
    ]
    counts = {
        "authored_candidate_count": sum(1 for row in rows if row["authored_candidate"]),
        "trainable_candidate_count": sum(1 for row in rows if row["trainable_candidate"]),
        "trained_candidate_count": sum(1 for row in rows if row["train_executed"]),
        "full_eval_completed_count": sum(1 for row in rows if row["full_eval_completed"]),
        "sealed_validation_completed_count": sum(
            1 for row in rows if row["sealed_validation_complete"]
        ),
        "declared_accepted_count": sum(
            1 for row in rows if row["declared_accepted"]
        ),
        "accepted_count": sum(1 for row in rows if row["valid_policy_accept"]),
        "eval_incomplete_count": sum(1 for row in rows if row["eval_incomplete"]),
    }
    failure_buckets = Counter(
        row["failure_bucket"] for row in rows if row["failure_bucket"] != "none"
    )
    violations = {
        "trainable_rounds_without_train": [
            row["round_id"]
            for row in rows
            if row["trainable_candidate"] and not row["train_executed"]
        ],
        "trainable_rounds_without_full_eval": [
            row["round_id"]
            for row in rows
            if row["trainable_candidate"] and not row["full_eval_completed"]
        ],
        "public_projection_unsafe_rounds": [
            row["round_id"]
            for row in rows
            if row["public_result_present"] and not row["public_projection_safe"]
        ],
        "accepted_rounds_missing_sealed_validation": [
            row["round_id"]
            for row in rows
            if row["declared_accepted"]
            and row["sealed_validation_status"] != "pass"
        ],
        "accepted_rounds_missing_required_gates": [
            row["round_id"]
            for row in rows
            if row["declared_accepted"]
            and (row["missing_required_accept_gates"] or not row["acceptance_evidence_valid"])
        ],
        "accepted_rounds_missing_consumed_gates": [
            row["round_id"]
            for row in rows
            if row["declared_accepted"]
            and row["missing_consumed_accept_gates"]
        ],
        "accepted_rounds_invalid_preconditions": [
            row["round_id"]
            for row in rows
            if row["declared_accepted"]
            and row["accepted_precondition_failures"]
        ],
        "decision_threshold_missing_rounds": [
            row["round_id"]
            for row in rows
            if row["policy_decision_needs_threshold"] and not row["comparison_threshold_present"]
        ],
        "decision_reason_inconsistent_rounds": [
            row["round_id"] for row in rows if not row["decision_reason_consistent"]
        ],
        "decision_record_invalid_rounds": [
            row["round_id"] for row in rows if not row["decision_record_valid"]
        ],
        "acceptance_replay_invalid_rounds": [
            row["round_id"] for row in rows if row["acceptance_replay_reasons"]
        ],
        "round_stage_invalid_rounds": [
            row["round_id"] for row in rows if not row["round_stage_consistent"]
        ],
        "accepted_rounds_invalid_decision": [
            row["round_id"]
            for row in rows
            if row["declared_accepted"] and not row["decision_record_valid"]
        ],
        "decision_gate_status_mismatch_rounds": [
            row["round_id"] for row in rows if row["gate_status_mismatches"]
        ],
    }
    checks = {
        "trainable_rounds_have_train": not violations["trainable_rounds_without_train"],
        "trainable_rounds_have_full_eval": not violations["trainable_rounds_without_full_eval"],
        "public_projection_passed": not violations["public_projection_unsafe_rounds"],
        "accepted_records_have_required_eval_evidence": not (
            violations["accepted_rounds_missing_sealed_validation"]
            or violations["accepted_rounds_missing_required_gates"]
            or violations["accepted_rounds_missing_consumed_gates"]
            or violations["accepted_rounds_invalid_preconditions"]
            or violations["accepted_rounds_invalid_decision"]
            or violations["decision_gate_status_mismatch_rounds"]
        ),
        "accepted_records_have_valid_preconditions": not violations[
            "accepted_rounds_invalid_preconditions"
        ],
        "decision_consumed_required_gates": not violations[
            "accepted_rounds_missing_consumed_gates"
        ],
        "decision_threshold_consumed": not violations["decision_threshold_missing_rounds"],
        "decision_reasons_consistent_with_eval": not violations[
            "decision_reason_inconsistent_rounds"
        ],
        "decision_records_valid": not violations["decision_record_invalid_rounds"],
        "acceptance_replay_consistent": not violations[
            "acceptance_replay_invalid_rounds"
        ],
        "round_stages_consistent": not violations["round_stage_invalid_rounds"],
        "decision_gate_statuses_match_metrics": not violations[
            "decision_gate_status_mismatch_rounds"
        ],
    }
    return {
        "schema_version": "true_rl_agent.eval_run_audit.v1",
        "counts": counts,
        "failure_buckets": dict(sorted(failure_buckets.items())),
        "checks": checks,
        "violations": violations,
        "rounds": [_public_round_row(row) for row in rows],
    }


def _audit_record(
    record: Mapping[str, Any],
    *,
    replay_entry: AcceptanceReplayEntry,
) -> dict[str, Any]:
    round_id = _round_id(record)
    public_result = _mapping(record.get("public_result"))
    metrics = _mapping(public_result.get("metrics")) if public_result else {}
    declared_decision = _mapping(record.get("decision"))
    candidate_surface = _mapping(record.get("candidate_surface"))
    decision_validation = replay_entry.validation
    decision = replay_entry.public_decision
    round_stage = decision_validation.round_stage
    stage_consistent = decision_validation.stage_consistent
    acceptance_validation = decision_validation.acceptance_evidence
    required_gates = set(acceptance_validation.required_gates)
    consumed_gates = _string_set(decision.get("consumed_gates"))
    gate_statuses = _string_mapping(decision.get("gate_statuses"))
    actual_gate_statuses = dict(acceptance_validation.gate_statuses)
    gate_status_mismatches = list(acceptance_validation.gate_status_mismatches)
    declared_accepted = declared_decision.get("accepted") is True
    accepted = decision.get("accepted") is True
    public_result_status = (
        public_result.get("status") if isinstance(public_result.get("status"), str) else "missing"
    )
    public_result_passed = public_result_status == "pass"
    verifier = _mapping(record.get("verifier"))
    verifier_status = verifier.get("status")
    verifier_failed = isinstance(verifier_status, str) and verifier_status != "pass"
    policy_eligible = (
        candidate_surface.get("eligible_for_policy_accept") is True
        or candidate_surface.get("classification") == "code_level"
    )
    public_projection_safe, public_projection_error = _public_projection_status(public_result)
    eval_job = _mapping(_mapping(record.get("jobs")).get("eval"))
    public_dev_minimum_count = _optional_int(eval_job.get("public_seed_count")) or 1
    public_dev_complete = _split_complete(
        metrics.get("public_dev"),
        minimum_count=public_dev_minimum_count,
    )
    public_accept_complete = bool(
        acceptance_validation.score_count is not None
        and acceptance_validation.min_score_count is not None
        and acceptance_validation.score_count
        >= acceptance_validation.min_score_count
        and acceptance_validation.score_stats_binding_valid
    )
    eval_protocol_bound = acceptance_validation.protocol_binding_valid
    sealed_validation_status = acceptance_validation.sealed_validation_status
    sealed_validation_complete = acceptance_validation.sealed_validation_complete
    full_eval_completed = bool(
        stage_consistent
        and round_stage.stage == EVALUATED_STAGE
        and public_result
        and decision_validation.public_result_contract_valid
        and public_projection_safe
        and public_result_passed
        and eval_protocol_bound
        and acceptance_validation.public_score_binding_valid
        and public_dev_complete
        and public_accept_complete
        and sealed_validation_complete
    )
    train_result_present = isinstance(record.get("train_result"), Mapping)
    train_executed = bool(stage_consistent and train_result_present)
    accepted_precondition_failures = list(
        decision_validation.accepted_precondition_failures
    )
    missing_required_accept_gates = list(acceptance_validation.missing_common_gates)
    missing_consumed_accept_gates = list(acceptance_validation.missing_consumed_gates)
    comparison = _mapping(decision.get("comparison"))
    comparison_threshold_present = _number_present(comparison.get("threshold"))
    policy_decision_needs_threshold = bool(
        accepted
        or (
            policy_eligible
            and full_eval_completed
            and _all_required_gates_pass(actual_gate_statuses)
        )
    )
    row = {
        "round_id": round_id,
        "authored_candidate": bool(stage_consistent and round_stage.candidate_authored),
        "trainable_candidate": bool(stage_consistent and round_stage.trainable_candidate),
        "policy_eligible": policy_eligible,
        "verifier_failed": verifier_failed,
        "round_stage": round_stage.stage,
        "round_stage_consistent": stage_consistent,
        "round_stage_reasons": list(
            decision_validation.stage_consistency_failures
        ),
        "train_result_present": train_result_present,
        "train_executed": train_executed,
        "public_result_present": bool(public_result),
        "public_result_status": public_result_status,
        "public_result_passed": public_result_passed,
        "public_projection_safe": public_projection_safe,
        "public_projection_error": public_projection_error,
        "public_dev_complete": public_dev_complete,
        "public_accept_complete": public_accept_complete,
        "eval_protocol_bound": eval_protocol_bound,
        "sealed_validation_complete": sealed_validation_complete,
        "sealed_validation_status": sealed_validation_status,
        "full_eval_completed": full_eval_completed,
        "declared_accepted": declared_accepted,
        "accepted": accepted,
        "valid_policy_accept": replay_entry.valid_policy_accept,
        "decision_record_valid": replay_entry.valid,
        "decision_record_reasons": list(
            (*decision_validation.reasons, *replay_entry.sequence_reasons)
        ),
        "acceptance_replay_reasons": list(replay_entry.sequence_reasons),
        "accepted_precondition_failures": accepted_precondition_failures,
        "decision_reason": str(decision.get("reason", "")),
        "decision_outcome": str(decision.get("outcome", "")),
        "decision_gate_statuses": gate_statuses,
        "actual_gate_statuses": actual_gate_statuses,
        "gate_status_mismatches": gate_status_mismatches,
        "required_gates": sorted(required_gates),
        "consumed_gates": sorted(consumed_gates),
        "missing_required_accept_gates": missing_required_accept_gates,
        "missing_consumed_accept_gates": missing_consumed_accept_gates,
        "acceptance_evidence_valid": acceptance_validation.valid,
        "acceptance_evidence_reasons": list(acceptance_validation.reasons),
        "comparison_threshold_present": comparison_threshold_present,
        "comparison_delta_met": comparison.get("delta_met"),
        "policy_decision_needs_threshold": policy_decision_needs_threshold,
    }
    row["eval_incomplete"] = bool(
        row["trainable_candidate"] and not full_eval_completed
    )
    row["decision_reason_consistent"] = bool(
        decision_validation.reason_consistent
        and not accepted_precondition_failures
        and not replay_entry.sequence_reasons
    )
    row["failure_bucket"] = _failure_bucket(row, record)
    return row


def _public_round_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "round_id": row["round_id"],
        "trainable_candidate": row["trainable_candidate"],
        "policy_eligible": row["policy_eligible"],
        "train_executed": row["train_executed"],
        "full_eval_completed": row["full_eval_completed"],
        "public_result_status": row["public_result_status"],
        "public_dev_complete": row["public_dev_complete"],
        "public_accept_complete": row["public_accept_complete"],
        "eval_protocol_bound": row["eval_protocol_bound"],
        "sealed_validation_status": row["sealed_validation_status"],
        "accepted": row["accepted"],
        "declared_accepted": row["declared_accepted"],
        "valid_policy_accept": row["valid_policy_accept"],
        "decision_record_valid": row["decision_record_valid"],
        "round_stage": row["round_stage"],
        "round_stage_consistent": row["round_stage_consistent"],
        "round_stage_reasons": row["round_stage_reasons"],
        "decision_reason": row["decision_reason"],
        "failure_bucket": row["failure_bucket"],
    }


def _public_projection_status(public_result: Mapping[str, Any]) -> tuple[bool, str | None]:
    if not public_result:
        return False, None
    try:
        assert_public_projection_safe(public_result)
    except ValueError as exc:
        return False, str(exc)
    return True, None


def _split_complete(value: Any, *, minimum_count: int) -> bool:
    split = _mapping(value)
    if not split:
        return False
    stats = _mapping(split.get("score_stats"))
    return validate_score_stats(
        stats,
        minimum_count=minimum_count,
        require_std=True,
        require_standard_error=True,
        require_protocol_hash=False,
    ).valid


def _failure_bucket(row: Mapping[str, Any], record: Mapping[str, Any]) -> str:
    if not row["round_stage_consistent"]:
        return "decision_contract"
    if (
        record.get("provider_failure") is not None
        or record.get("static_authoring_failure") is not None
        or row["verifier_failed"]
    ):
        return "authoring"
    if not row["trainable_candidate"]:
        if row["round_stage"] == PRE_TRAIN_REJECTION_STAGE:
            return "pre_train"
        return "none"
    if not row["train_executed"]:
        return "train"
    if (
        not row["public_result_present"]
        or not row["public_result_passed"]
        or not row["public_dev_complete"]
        or not row["public_accept_complete"]
        or not row["eval_protocol_bound"]
    ):
        return "public_eval"
    if not row["policy_eligible"]:
        return "non_policy_surface"
    if _has_acceptance_reason(
        row,
        "candidate_contamination_detected",
        "candidate_contamination_flag_invalid",
    ):
        return "sealed_validation"
    if row["sealed_validation_status"] != "pass":
        return "sealed_validation"
    actual_gate_statuses = row["actual_gate_statuses"]
    if actual_gate_statuses.get("budget_accounting") not in {None, "pass"}:
        return "budget_gate"
    if any(status != "pass" for status in actual_gate_statuses.values()):
        return "required_gate"
    if (
        row["missing_required_accept_gates"]
        or row["missing_consumed_accept_gates"]
        or row["gate_status_mismatches"]
    ):
        return "required_gate"
    if not row["decision_record_valid"]:
        return "decision_contract"
    if row["accepted"]:
        if (
            row["missing_required_accept_gates"]
            or row["missing_consumed_accept_gates"]
            or row["gate_status_mismatches"]
            or row["accepted_precondition_failures"]
            or not row["acceptance_evidence_valid"]
        ):
            return "required_gate"
        return "accepted"
    if row["decision_reason"] in {
        "absolute_accept_floor_not_met",
        "noise_aware_delta_not_met",
        "public_score_delta_not_met",
        "current_best_eval_stats_missing",
        "candidate_eval_stats_missing",
        "eval_protocol_hash_missing",
        "eval_protocol_hash_mismatch",
        "insufficient_accept_evidence",
        "accept_score_stats_untrusted",
        "required_score_missing",
    }:
        return "decision_threshold"
    return "other_decision_reject"


def _has_acceptance_reason(row: Mapping[str, Any], *prefixes: str) -> bool:
    reasons = row.get("acceptance_evidence_reasons")
    if not isinstance(reasons, list | tuple):
        return False
    return any(
        isinstance(reason, str) and reason.startswith(prefix)
        for reason in reasons
        for prefix in prefixes
    )


def _round_id(record: Mapping[str, Any]) -> str:
    raw = record.get("round_id")
    return raw if isinstance(raw, str) and raw else "<unknown-round>"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _string_set(value: Any) -> set[str]:
    if not isinstance(value, list | tuple | set | frozenset):
        return set()
    return {str(item) for item in value if str(item)}


def _string_mapping(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, str] = {}
    for key, raw_status in value.items():
        if isinstance(raw_status, str):
            result[str(key)] = raw_status
    return result


def _all_required_gates_pass(gate_statuses: Mapping[str, str]) -> bool:
    return bool(gate_statuses) and all(status == "pass" for status in gate_statuses.values())


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _number_present(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)
