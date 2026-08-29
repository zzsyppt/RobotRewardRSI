"""Shared task-neutral validation for public decision records."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from true_rl_agent.agent_core.ledger import is_strict_ledger_record
from true_rl_agent.evaluation.acceptance_contract import (
    AcceptanceEvidenceValidation,
    validate_acceptance_evidence,
)
from true_rl_agent.evaluation.acceptance_phase import acceptance_comparison_entered
from true_rl_agent.evaluation.acceptance_protocol import (
    ACCEPTANCE_DELTA_ACCEPT_REASONS,
    ACCEPTANCE_DELTA_REJECT_REASONS,
    AcceptanceProtocol,
    acceptance_delta_met,
    acceptance_delta_reason,
    guard_regression_threshold,
    named_guard_regression_threshold,
    threshold_components_match,
)
from true_rl_agent.evaluation.binding import (
    EvaluationBindingV1,
    is_sha256_checksum,
    is_sha256_digest,
    require_sha256_digest,
)
from true_rl_agent.evaluation.protocol import validate_evaluation_protocol_mapping
from true_rl_agent.evaluation.result_contract import validate_public_eval_result_mapping
from true_rl_agent.evaluation.round_stage_contract import (
    EVALUATED_STAGE,
    POST_TRAIN_FAILURE_STAGE,
    PRE_TRAIN_REJECTION_STAGE,
    PROVIDER_FAILURE_STAGE,
    STATIC_AUTHORING_FAILURE_STAGE,
    TRAIN_FAILURE_STAGE,
    VERIFIER_FAILURE_STAGE,
    RoundStageValidation,
    validate_round_stage,
)
from true_rl_agent.evaluation.subject import ExperimentSubject
from true_rl_agent.runner.candidate_failure import (
    validate_candidate_runtime_failure_details,
)
from true_rl_agent.runner.initial_policy import InitialPolicyStatePairing
from true_rl_agent.runner.specs import CheckpointRef

POLICY_IMPROVEMENT_ACCEPT_OUTCOME = "policy_improvement_accept"
DIAGNOSTIC_IMPROVEMENT_RECORD_OUTCOME = "diagnostic_improvement_record"
ENGINEERING_IMPROVEMENT_RECORD_OUTCOME = "engineering_improvement_record"
REJECT_OUTCOME = "reject"

_ALLOWED_REASONS_BY_OUTCOME = {
    POLICY_IMPROVEMENT_ACCEPT_OUTCOME: frozenset(
        {
            "absolute_accept_floor",
            "absolute_and_guard_accept_floors",
            "absolute_and_named_guard_accept_floors",
            "public_score_delta_met",
        }
    )
    | ACCEPTANCE_DELTA_ACCEPT_REASONS,
    DIAGNOSTIC_IMPROVEMENT_RECORD_OUTCOME: frozenset({"diagnostic_only_candidate"}),
    ENGINEERING_IMPROVEMENT_RECORD_OUTCOME: frozenset({"engineering_only_candidate"}),
    REJECT_OUTCOME: frozenset(
        {
            "accept_score_stats_untrusted",
            "acceptance_protocol_mismatch",
            "absolute_accept_floor_not_met",
            "absolute_or_guard_accept_floor_not_met",
            "absolute_or_named_guard_accept_floor_not_met",
            "baseline_accept_gate_not_required",
            "candidate_contamination_detected",
            "candidate_eval_stats_missing",
            "candidate_guard_stats_missing",
            "candidate_named_guard_stats_missing",
            "candidate_runtime_failed",
            "candidate_static_authoring_failed",
            "candidate_verifier_failed",
            "current_best_eval_stats_missing",
            "eval_incomplete",
            "eval_protocol_hash_mismatch",
            "eval_protocol_hash_missing",
            "evaluation_binding_mismatch",
            "guard_eval_stats_missing",
            "guard_score_regression",
            "named_guard_eval_stats_missing",
            "named_guard_score_regression",
            "insufficient_accept_evidence",
            "no_candidate_changes",
            "parameter_only_candidate",
            "provider_candidate_generation_failed",
            "public_result_status_not_pass",
            "public_score_binding_mismatch",
            "public_score_delta_not_met",
            "public_score_nonfinite",
            "required_gate_failed",
            "required_score_missing",
            "runner_activation_missing",
            "sealed_validation_gate_not_required",
            "task_invariant_violation",
            "unsupported_candidate_surface",
        }
    )
    | ACCEPTANCE_DELTA_REJECT_REASONS,
}
_DELTA_ACCEPT_REASONS = ACCEPTANCE_DELTA_ACCEPT_REASONS | frozenset(
    {"public_score_delta_met"}
)
_DELTA_REJECT_REASONS = ACCEPTANCE_DELTA_REJECT_REASONS | frozenset(
    {"public_score_delta_not_met"}
)
_SURFACE_REASONS = frozenset(
    {
        "diagnostic_only_candidate",
        "engineering_only_candidate",
        "no_candidate_changes",
        "parameter_only_candidate",
        "unsupported_candidate_surface",
    }
)
_ACCEPTANCE_REJECT_REASONS = frozenset(
    {
        "accept_score_stats_untrusted",
        "acceptance_protocol_mismatch",
        "baseline_accept_gate_not_required",
        "candidate_contamination_detected",
        "eval_protocol_hash_missing",
        "insufficient_accept_evidence",
        "public_score_binding_mismatch",
        "required_gate_failed",
        "required_score_missing",
        "sealed_validation_gate_not_required",
    }
)
_REASON_ALLOWED_ROUND_STAGES = {
    "provider_candidate_generation_failed": frozenset({PROVIDER_FAILURE_STAGE}),
    "candidate_static_authoring_failed": frozenset({STATIC_AUTHORING_FAILURE_STAGE}),
    "candidate_verifier_failed": frozenset({VERIFIER_FAILURE_STAGE}),
    "task_invariant_violation": frozenset({PRE_TRAIN_REJECTION_STAGE}),
    "candidate_runtime_failed": frozenset(
        {TRAIN_FAILURE_STAGE, POST_TRAIN_FAILURE_STAGE}
    ),
    "eval_incomplete": frozenset({POST_TRAIN_FAILURE_STAGE}),
}


@dataclass(frozen=True)
class DecisionRecordValidation:
    valid: bool
    reasons: tuple[str, ...]
    accepted: bool
    outcome: str
    reason: str
    outcome_consistent: bool
    reason_consistent: bool
    stage_consistent: bool
    valid_policy_accept: bool
    accepted_precondition_failures: tuple[str, ...]
    stage_consistency_failures: tuple[str, ...]
    acceptance_evidence: AcceptanceEvidenceValidation
    round_stage: RoundStageValidation
    public_result_contract_valid: bool
    trusted_acceptance_protocol_bound: bool
    trusted_public_score_valid: bool

    def raise_for_failure(self) -> None:
        if self.valid:
            return
        detail = ", ".join(self.reasons) or "unknown decision record failure"
        raise ValueError(f"invalid decision record: {detail}")


def validate_decision_record(
    record: Mapping[str, Any],
    *,
    acceptance_protocol: AcceptanceProtocol | None = None,
    expected_session_genesis_hash: str | None = None,
    expected_evaluation_binding: EvaluationBindingV1 | None = None,
    expected_initial_policy_state_hash: str | None = None,
) -> DecisionRecordValidation:
    """Validate decision shape, semantics, comparison, and acceptance evidence."""

    decision = _mapping(record.get("decision"))
    raw_public_result = record.get("public_result")
    public_result = _mapping(raw_public_result)
    metrics = _mapping(public_result.get("metrics"))
    public_result_fields = None
    public_result_contract_error = False
    if raw_public_result is not None:
        try:
            public_result_fields = validate_public_eval_result_mapping(
                raw_public_result,
                field_name="ledger public_result",
            )
        except ValueError:
            public_result_contract_error = True
        else:
            metrics = public_result_fields.metrics
    candidate_surface = _mapping(record.get("candidate_surface"))
    comparison_entered = acceptance_comparison_entered(
        policy_eligible=candidate_surface.get("eligible_for_policy_accept") is True,
        public_result_status=(
            public_result_fields.status if public_result_fields is not None else None
        ),
    )
    eval_job = _mapping(_mapping(record.get("jobs")).get("eval"))
    eval_checkpoint = _mapping(eval_job.get("checkpoint"))
    eval_job_protocol = validate_evaluation_protocol_mapping(eval_job)
    acceptance_evidence = validate_acceptance_evidence(
        metrics,
        decision=decision if comparison_entered else None,
        acceptance_protocol=acceptance_protocol,
        public_score=(
            public_result.get("public_score") if public_result else None
        ),
        expected_sealed_seed_count=(
            eval_job_protocol.protocol.sealed_validation_seed_count
            if eval_job_protocol.valid and eval_job_protocol.protocol is not None
            else None
        ),
        expected_checkpoint_hash=_nonempty_string(eval_checkpoint.get("checksum")),
    )
    round_stage = validate_round_stage(record)
    reasons: list[str] = []
    reasons.extend(round_stage.reasons)
    reasons.extend(_candidate_runtime_failure_reasons(record))
    reasons.extend(
        _initial_policy_state_pairing_reasons(
            record,
            round_stage=round_stage,
            eval_job=eval_job,
            expected_initial_policy_state_hash=(
                expected_initial_policy_state_hash
            ),
        )
    )
    reasons.extend(
        _session_binding_reasons(
            record,
            round_stage=round_stage,
            metrics=metrics,
            eval_job=eval_job,
            expected_session_genesis_hash=expected_session_genesis_hash,
            expected_evaluation_binding=expected_evaluation_binding,
        )
    )
    if public_result_contract_error:
        reasons.append("public_result_contract_invalid")
    if round_stage.stage == EVALUATED_STAGE and acceptance_protocol is None:
        reasons.append("trusted_acceptance_protocol_required")
    if round_stage.stage == EVALUATED_STAGE and public_result.get("status") == "pass":
        reasons.extend(
            reason
            for reason in acceptance_evidence.reasons
            if reason.startswith(("public_score_", "acceptance_protocol_public_score_binding"))
        )
    job_eval_protocol_hash: str | None = None
    if round_stage.stage == EVALUATED_STAGE:
        reasons.extend(
            f"eval_job_{reason}" for reason in eval_job_protocol.reasons
        )
        job_eval_protocol_hash = (
            eval_job_protocol.protocol.protocol_hash
            if eval_job_protocol.valid and eval_job_protocol.protocol is not None
            else None
        )
        if (
            job_eval_protocol_hash is not None
            and acceptance_evidence.eval_protocol_hash is not None
            and job_eval_protocol_hash != acceptance_evidence.eval_protocol_hash
        ):
            reasons.append("eval_job_protocol_hash_mismatch")
        if not _nonempty_string(eval_checkpoint.get("checksum")):
            reasons.append("eval_job_checkpoint_hash_missing")

    raw_accepted = decision.get("accepted")
    accepted = raw_accepted is True
    if not isinstance(raw_accepted, bool):
        reasons.append("decision_accepted_not_boolean")

    outcome = _nonempty_string(decision.get("outcome"))
    reason = _nonempty_string(decision.get("reason"))
    if outcome not in _ALLOWED_REASONS_BY_OUTCOME:
        reasons.append("decision_outcome_unknown")
    expected_accepted = outcome == POLICY_IMPROVEMENT_ACCEPT_OUTCOME
    outcome_consistent = isinstance(raw_accepted, bool) and accepted == expected_accepted
    if not outcome_consistent:
        reasons.append("decision_acceptance_outcome_mismatch")

    allowed_reasons = _ALLOWED_REASONS_BY_OUTCOME.get(outcome, frozenset())
    allowed_stages = _REASON_ALLOWED_ROUND_STAGES.get(
        reason,
        frozenset({EVALUATED_STAGE}),
    )
    stage_consistency_failures = list(round_stage.reasons)
    if reason not in allowed_reasons:
        reasons.append("decision_reason_not_allowed_for_outcome")
    elif round_stage.stage not in allowed_stages:
        mismatch = f"round_stage_reason_mismatch:{reason}:actual={round_stage.stage}"
        reasons.append(mismatch)
        stage_consistency_failures.append(mismatch)
    stage_consistent = not stage_consistency_failures

    surface_classification = _nonempty_string(
        candidate_surface.get("classification")
    )
    declared_surface = _nonempty_string(decision.get("surface_classification"))
    if surface_classification and declared_surface != surface_classification:
        reasons.append("decision_surface_classification_mismatch")
    if round_stage.stage == EVALUATED_STAGE and comparison_entered:
        reasons.extend(
            f"decision_required_gate_not_consumed:{gate}"
            for gate in acceptance_evidence.missing_consumed_gates
        )
        reasons.extend(
            f"decision_gate_status_mismatch:{gate}"
            for gate in acceptance_evidence.gate_status_mismatches
        )

    public_score = (
        public_result_fields.public_score if public_result_fields is not None else None
    )
    declared_public_score = _finite_number(decision.get("public_score"))
    if public_score is not None:
        if declared_public_score is None or not _numbers_match(
            declared_public_score,
            public_score,
        ):
            reasons.append("decision_public_score_mismatch")
    elif declared_public_score is not None:
        reasons.append("decision_public_score_without_eval_score")

    comparison = _mapping(decision.get("comparison"))
    if comparison_entered and not comparison:
        reasons.append("decision_comparison_missing")
    if not comparison_entered and comparison:
        reasons.append("decision_comparison_without_comparison_entry")
    if not comparison_entered and (
        _string_set(decision.get("consumed_gates"))
        or _mapping(decision.get("gate_statuses"))
    ):
        reasons.append("decision_gate_consumption_without_comparison_entry")
    reasons.extend(
        _comparison_reasons(
            comparison,
            acceptance_evidence=acceptance_evidence,
            public_score=public_score,
            acceptance_protocol=acceptance_protocol,
        )
    )
    reasons.extend(
        _reason_evidence_reasons(
            reason=reason,
            record=record,
            public_result=public_result,
            metrics=metrics,
            comparison=comparison,
            acceptance_evidence=acceptance_evidence,
            acceptance_protocol=acceptance_protocol,
        )
    )

    accepted_precondition_failures: list[str] = []
    if accepted:
        if not isinstance(record.get("candidate"), Mapping):
            accepted_precondition_failures.append("accepted_candidate_missing")
        if record.get("provider_failure") is not None:
            accepted_precondition_failures.append("accepted_provider_failure_present")
        if record.get("static_authoring_failure") is not None:
            accepted_precondition_failures.append(
                "accepted_static_authoring_failure_present"
            )
        verifier = _mapping(record.get("verifier"))
        if verifier.get("status") != "pass":
            accepted_precondition_failures.append(
                "accepted_candidate_verifier_not_pass"
            )
        if not isinstance(record.get("train_result"), Mapping):
            accepted_precondition_failures.append("accepted_candidate_without_train")
        if public_result.get("status") != "pass":
            accepted_precondition_failures.append(
                "accepted_public_result_status_not_pass"
            )
        if declared_surface != "code_level":
            accepted_precondition_failures.append("accepted_surface_not_code_level")
        if surface_classification != "code_level":
            accepted_precondition_failures.append(
                "accepted_candidate_surface_not_code_level"
            )
        if not acceptance_evidence.valid:
            accepted_precondition_failures.append(
                "accepted_acceptance_evidence_invalid"
            )
    reasons.extend(accepted_precondition_failures)

    deduped_reasons = tuple(dict.fromkeys(reasons))
    valid = not deduped_reasons
    reason_consistent = outcome_consistent and not any(
        item.startswith(("decision_", "round_stage_"))
        or item.endswith("_reason_evidence_mismatch")
        for item in deduped_reasons
    ) and (not accepted or valid)
    trusted_public_score_valid = bool(
        public_result_fields is not None
        and public_result_fields.status == "pass"
        and acceptance_protocol is not None
        and acceptance_evidence.acceptance_protocol_binding_valid
        and acceptance_evidence.protocol_binding_valid
        and acceptance_evidence.public_score_binding_valid
        and job_eval_protocol_hash is not None
        and job_eval_protocol_hash == acceptance_evidence.eval_protocol_hash
    )
    return DecisionRecordValidation(
        valid=valid,
        reasons=deduped_reasons,
        accepted=accepted,
        outcome=outcome,
        reason=reason,
        outcome_consistent=outcome_consistent,
        reason_consistent=reason_consistent,
        stage_consistent=stage_consistent,
        valid_policy_accept=bool(
            valid and accepted and outcome == POLICY_IMPROVEMENT_ACCEPT_OUTCOME
        ),
        accepted_precondition_failures=tuple(accepted_precondition_failures),
        stage_consistency_failures=tuple(stage_consistency_failures),
        acceptance_evidence=acceptance_evidence,
        round_stage=round_stage,
        public_result_contract_valid=public_result_fields is not None,
        trusted_acceptance_protocol_bound=acceptance_protocol is not None,
        trusted_public_score_valid=trusted_public_score_valid,
    )


def _session_binding_reasons(
    record: Mapping[str, Any],
    *,
    round_stage: RoundStageValidation,
    metrics: Mapping[str, Any],
    eval_job: Mapping[str, Any],
    expected_session_genesis_hash: str | None,
    expected_evaluation_binding: EvaluationBindingV1 | None,
) -> list[str]:
    reasons: list[str] = []
    raw_genesis_hash = record.get("session_genesis_hash")
    if raw_genesis_hash is not None and not is_sha256_digest(raw_genesis_hash):
        reasons.append("session_genesis_hash_invalid")
    if expected_session_genesis_hash is not None:
        require_sha256_digest(
            expected_session_genesis_hash,
            "expected_session_genesis_hash",
        )
        if raw_genesis_hash is None:
            reasons.append("session_genesis_hash_missing")
        elif raw_genesis_hash != expected_session_genesis_hash:
            reasons.append("session_genesis_hash_mismatch")

    record_binding = _parse_evaluation_binding(
        record.get("evaluation_binding"),
        reasons=reasons,
        reason_prefix="record_evaluation_binding",
        required=expected_evaluation_binding is not None,
    )
    if (
        expected_evaluation_binding is not None
        and record_binding is not None
        and record_binding != expected_evaluation_binding
    ):
        reasons.append("record_evaluation_binding_mismatch")

    if round_stage.stage == EVALUATED_STAGE:
        reasons.extend(
            _program_lineage_reasons(
                record=record,
                eval_job=eval_job,
                metrics=metrics,
            )
        )
        job_binding = _parse_evaluation_binding(
            eval_job.get("evaluation_binding"),
            reasons=reasons,
            reason_prefix="eval_job_evaluation_binding",
            required=expected_evaluation_binding is not None,
        )
        result_binding = _parse_evaluation_binding(
            metrics.get("evaluation_binding"),
            reasons=reasons,
            reason_prefix="public_result_evaluation_binding",
            required=expected_evaluation_binding is not None,
        )
        for source, binding in (
            ("eval_job", job_binding),
            ("public_result", result_binding),
        ):
            if (
                record_binding is not None
                and binding is not None
                and binding != record_binding
            ):
                reasons.append(f"{source}_evaluation_binding_mismatch")
            if (
                expected_evaluation_binding is not None
                and binding is not None
                and binding != expected_evaluation_binding
            ):
                reasons.append(f"{source}_expected_evaluation_binding_mismatch")
    return reasons


def _initial_policy_state_pairing_reasons(
    record: Mapping[str, Any],
    *,
    round_stage: RoundStageValidation,
    eval_job: Mapping[str, Any],
    expected_initial_policy_state_hash: str | None,
) -> list[str]:
    """Validate pairing only after a reference-anchored job is materialized."""

    if expected_initial_policy_state_hash is None:
        return []
    expected = require_sha256_digest(
        expected_initial_policy_state_hash,
        "expected_initial_policy_state_hash",
    )
    reasons: list[str] = []
    train_job = _mapping(_mapping(record.get("jobs")).get("train"))
    if train_job:
        declared = train_job.get("expected_initial_policy_state_hash")
        if not is_sha256_digest(declared):
            reasons.append("train_job_expected_initial_policy_state_hash_missing")
        elif declared != expected:
            reasons.append("train_job_expected_initial_policy_state_hash_mismatch")

    train_result = record.get("train_result")
    pairing: InitialPolicyStatePairing | None = None
    if isinstance(train_result, Mapping):
        try:
            pairing = InitialPolicyStatePairing.from_mapping(
                train_result.get("initial_policy_state_pairing")
            )
        except (TypeError, ValueError):
            reasons.append("train_result_initial_policy_state_pairing_invalid")
        else:
            if pairing.expected_initial_policy_state_hash != expected:
                reasons.append(
                    "train_result_expected_initial_policy_state_hash_mismatch"
                )
            if not pairing.match:
                reasons.append("train_result_initial_policy_state_mismatch")
        if round_stage.stage == EVALUATED_STAGE:
            train_checkpoint = _public_checkpoint_ref(
                train_result.get("checkpoint")
            )
            eval_checkpoint = _public_checkpoint_ref(eval_job.get("checkpoint"))
            if train_checkpoint is None:
                reasons.append(
                    "initial_policy_state_train_result_checkpoint_invalid"
                )
            if eval_checkpoint is None:
                reasons.append("initial_policy_state_eval_job_checkpoint_invalid")
            if (
                train_checkpoint is not None
                and eval_checkpoint is not None
                and train_checkpoint != eval_checkpoint
            ):
                reasons.append("initial_policy_state_checkpoint_lineage_mismatch")
            declared_pairing_hash = eval_job.get(
                "initial_policy_state_pairing_hash"
            )
            if not is_sha256_digest(declared_pairing_hash):
                reasons.append("eval_job_initial_policy_state_pairing_hash_missing")
            elif pairing is not None and declared_pairing_hash != pairing.pairing_hash:
                reasons.append("eval_job_initial_policy_state_pairing_hash_mismatch")
    elif round_stage.stage == EVALUATED_STAGE:
        reasons.append("evaluated_train_result_initial_policy_state_missing")
    return reasons


def _public_checkpoint_ref(value: Any) -> CheckpointRef | None:
    if not isinstance(value, Mapping) or set(value) != {
        "uri",
        "checksum",
        "role",
        "sealed",
    }:
        return None
    try:
        checkpoint = CheckpointRef(
            uri=value.get("uri"),
            checksum=value.get("checksum"),
            role=value.get("role"),
            sealed=value.get("sealed"),
        )
    except (TypeError, ValueError):
        return None
    if checkpoint.role != "eval" or not checkpoint.sealed:
        return None
    return checkpoint


def _candidate_runtime_failure_reasons(
    record: Mapping[str, Any],
) -> list[str]:
    strict = is_strict_ledger_record(record)
    decision = _mapping(record.get("decision"))
    decision_reason = decision.get("reason")
    runner_error = _mapping(record.get("runner_error"))
    error_type = runner_error.get("type")
    if (
        decision_reason != "candidate_runtime_failed"
        and error_type != "CandidateRuntimeError"
    ):
        return []
    reasons: list[str] = []
    if decision_reason != "candidate_runtime_failed":
        reasons.append("candidate_runtime_error_decision_reason_mismatch")
    if set(runner_error) != {"type", "message", "details"}:
        reasons.append("candidate_runtime_error_fields_invalid")
    if error_type != "CandidateRuntimeError":
        reasons.append("candidate_runtime_error_type_invalid")
    if not _nonempty_string(runner_error.get("message")):
        reasons.append("candidate_runtime_error_message_missing")
    try:
        details = validate_candidate_runtime_failure_details(
            runner_error.get("details")
        )
    except (TypeError, ValueError):
        reasons.append("candidate_runtime_error_details_invalid")
        return _versioned_candidate_runtime_reasons(reasons, strict=strict)
    train_job = _mapping(_mapping(record.get("jobs")).get("train"))
    contract_hash = train_job.get("candidate_failure_contract_hash")
    if not is_sha256_digest(contract_hash):
        reasons.append("candidate_failure_contract_hash_missing")
    elif (
        details["candidate_failure_contract"]["contract_hash"]
        != contract_hash
    ):
        reasons.append("candidate_failure_contract_hash_mismatch")
    return _versioned_candidate_runtime_reasons(reasons, strict=strict)


def _versioned_candidate_runtime_reasons(
    reasons: list[str],
    *,
    strict: bool,
) -> list[str]:
    if strict:
        return reasons
    return [f"legacy_{reason}" for reason in reasons]


def _program_lineage_reasons(
    *,
    record: Mapping[str, Any],
    eval_job: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> list[str]:
    reasons: list[str] = []
    train_job = _mapping(_mapping(record.get("jobs")).get("train"))
    candidate = _mapping(record.get("candidate"))
    candidate_id = _nonempty_string(candidate.get("candidate_id"))
    try:
        eval_subject = ExperimentSubject.from_mapping(eval_job.get("subject"))
    except (TypeError, ValueError):
        eval_subject = None
        reasons.append("eval_job_subject_invalid")
    try:
        train_subject = ExperimentSubject.from_mapping(train_job.get("subject"))
    except (TypeError, ValueError):
        train_subject = None
        reasons.append("train_job_subject_invalid")
    try:
        result_subject = ExperimentSubject.from_mapping(
            metrics.get("evaluation_subject")
        )
    except (TypeError, ValueError):
        result_subject = None
        reasons.append("public_result_subject_invalid")
    expected_subject = (
        ExperimentSubject.candidate(candidate_id) if candidate_id else None
    )
    for source_name, subject in (
        ("train_job", train_subject),
        ("eval_job", eval_subject),
        ("public_result", result_subject),
    ):
        if expected_subject is not None and subject != expected_subject:
            reasons.append(f"{source_name}_candidate_subject_mismatch")

    binding = _mapping(train_job.get("training_binding"))
    lineage_fields = {
        "subject_manifest_hash": is_sha256_digest,
        "program_source_checksum": is_sha256_checksum,
        "training_binding_hash": is_sha256_digest,
        "capability_profile_hash": is_sha256_digest,
    }
    for field_name, validator in lineage_fields.items():
        values = {
            "train_job": (
                binding.get("binding_hash")
                if field_name == "training_binding_hash"
                else binding.get("capability_profile_hash")
                if field_name == "capability_profile_hash"
                else train_job.get(field_name)
            ),
            "eval_job": eval_job.get(field_name),
            "public_result": metrics.get(field_name),
        }
        if any(not validator(value) for value in values.values()):
            reasons.append(f"{field_name}_invalid_or_missing")
            continue
        if len(set(values.values())) != 1:
            reasons.append(f"{field_name}_mismatch")
    return reasons


def _parse_evaluation_binding(
    value: Any,
    *,
    reasons: list[str],
    reason_prefix: str,
    required: bool,
) -> EvaluationBindingV1 | None:
    if value is None:
        if required:
            reasons.append(f"{reason_prefix}_missing")
        return None
    try:
        return EvaluationBindingV1.from_mapping(value)
    except (TypeError, ValueError):
        reasons.append(f"{reason_prefix}_invalid")
        return None


def record_has_valid_policy_accept(
    record: Mapping[str, Any],
    *,
    acceptance_protocol: AcceptanceProtocol | None = None,
) -> bool:
    return validate_decision_record(
        record,
        acceptance_protocol=acceptance_protocol,
    ).valid_policy_accept


def project_validated_decision(
    record: Mapping[str, Any],
    *,
    validation: DecisionRecordValidation | None = None,
    acceptance_protocol: AcceptanceProtocol | None = None,
    extra_validation_reasons: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Return the only decision projection allowed on public consumer surfaces.

    Invalid or internally inconsistent ledger decisions are represented as a
    bounded fail-closed rejection.  The projection never inherits a declared
    score, comparison, or acceptance bit that failed the shared contract.
    """

    validation = validation or validate_decision_record(
        record,
        acceptance_protocol=acceptance_protocol,
    )
    public_score = (
        validation.acceptance_evidence.canonical_public_score
        if validation.trusted_public_score_valid
        else None
    )
    surface = _nonempty_string(
        _mapping(record.get("candidate_surface")).get("classification")
    )
    gate_statuses = dict(validation.acceptance_evidence.gate_statuses)
    all_reasons = tuple(dict.fromkeys((*validation.reasons, *extra_validation_reasons)))
    if not validation.valid or extra_validation_reasons:
        reasons = list(all_reasons[:16])
        projection: dict[str, Any] = {
            "validation_status": "invalid",
            "outcome": REJECT_OUTCOME,
            "accepted": False,
            "reason": "decision_record_invalid",
            "surface_classification": surface or None,
            "public_score": public_score,
            "comparison": {},
            "consumed_gates": [],
            "gate_statuses": gate_statuses,
            "validation_reasons": reasons,
        }
        omitted = len(all_reasons) - len(reasons)
        if omitted > 0:
            projection["omitted_validation_reason_count"] = omitted
        return projection

    decision = _mapping(record.get("decision"))
    return {
        "validation_status": "valid",
        "outcome": validation.outcome,
        "accepted": validation.accepted,
        "reason": validation.reason,
        "surface_classification": surface or None,
        "public_score": public_score,
        "comparison": dict(_mapping(decision.get("comparison"))),
        "consumed_gates": sorted(_string_set(decision.get("consumed_gates"))),
        "gate_statuses": gate_statuses,
    }


def _comparison_reasons(
    comparison: Mapping[str, Any],
    *,
    acceptance_evidence: AcceptanceEvidenceValidation,
    public_score: float | None,
    acceptance_protocol: AcceptanceProtocol | None,
) -> list[str]:
    if not comparison:
        return []
    reasons: list[str] = []
    mode = comparison.get("mode")
    if mode not in {"bootstrap", "reference", "accepted", "incomplete"}:
        reasons.append("decision_comparison_mode_invalid")
    candidate_score = _finite_number(comparison.get("candidate_score"))
    if acceptance_evidence.score_mean is None:
        if candidate_score is not None:
            reasons.append("decision_comparison_candidate_score_without_evidence")
    elif candidate_score is None or not _numbers_match(
        candidate_score,
        acceptance_evidence.score_mean,
    ):
        reasons.append("decision_comparison_candidate_score_mismatch")
    candidate_stats = _mapping(comparison.get("candidate_score_stats"))
    if candidate_stats and dict(candidate_stats) != dict(
        acceptance_evidence.score_stats
    ):
        reasons.append("decision_comparison_candidate_stats_mismatch")
    comparison_public_score = _finite_number(comparison.get("public_score"))
    if comparison_public_score is not None and (
        public_score is None
        or not _numbers_match(comparison_public_score, public_score)
    ):
        reasons.append("decision_comparison_public_score_mismatch")
    if comparison.get("score_source") != acceptance_evidence.score_source:
        reasons.append("decision_comparison_score_source_mismatch")
    if (
        comparison.get("acceptance_protocol_hash")
        != acceptance_evidence.acceptance_protocol_hash
    ):
        reasons.append("decision_comparison_acceptance_protocol_mismatch")
    comparison_floor = _finite_number(comparison.get("absolute_accept_floor"))
    if (
        comparison_floor is None
        or acceptance_evidence.absolute_accept_floor is None
        or not _numbers_match(
            comparison_floor,
            acceptance_evidence.absolute_accept_floor,
        )
    ):
        reasons.append("decision_comparison_absolute_floor_mismatch")
    if acceptance_protocol is not None and (
        comparison.get("acceptance_protocol_hash")
        != acceptance_protocol.protocol_hash
    ):
        reasons.append("decision_comparison_expected_protocol_mismatch")
    if acceptance_protocol is not None and acceptance_protocol.uses_guard_score:
        candidate_guard = _finite_number(comparison.get("candidate_guard_score"))
        if (
            acceptance_evidence.guard_score_mean is None
            or candidate_guard is None
            or not _numbers_match(
                candidate_guard,
                acceptance_evidence.guard_score_mean,
            )
        ):
            reasons.append("decision_comparison_candidate_guard_score_mismatch")
        if comparison.get("guard_score_source") != acceptance_evidence.guard_score_source:
            reasons.append("decision_comparison_guard_score_source_mismatch")
        candidate_guard_stats = _mapping(
            comparison.get("candidate_guard_score_stats")
        )
        if dict(candidate_guard_stats) != dict(
            acceptance_evidence.guard_score_stats
        ):
            reasons.append("decision_comparison_candidate_guard_stats_mismatch")
        max_regression = _finite_number(comparison.get("guard_max_regression"))
        best_guard = _finite_number(comparison.get("current_best_guard_score"))
        guard_delta = _finite_number(comparison.get("guard_delta"))
        effective_regression = _finite_number(
            comparison.get("guard_regression_threshold")
        )
        guard_met = comparison.get("guard_met")
        if (
            max_regression is None
            or acceptance_protocol.guard_max_regression is None
            or not _numbers_match(
                max_regression,
                acceptance_protocol.guard_max_regression,
            )
        ):
            reasons.append("decision_comparison_guard_regression_mismatch")
        if (
            candidate_guard is not None
            and best_guard is not None
            and max_regression is not None
        ):
            expected_guard_delta = candidate_guard - best_guard
            if guard_delta is None or not _numbers_match(
                guard_delta,
                expected_guard_delta,
            ):
                reasons.append("decision_comparison_guard_delta_mismatch")
            try:
                threshold_result = guard_regression_threshold(
                    acceptance_protocol,
                    candidate_guard_score=candidate_guard,
                    best_guard_score=best_guard,
                    candidate_guard_score_samples=(
                        acceptance_evidence.guard_score_samples
                    ),
                    best_guard_score_samples=_mapping(
                        comparison.get("best_guard_score_stats")
                    ).get("score_samples"),
                )
            except ValueError:
                reasons.append("decision_comparison_guard_threshold_invalid")
            else:
                if effective_regression is None or not _numbers_match(
                    effective_regression,
                    threshold_result.threshold,
                ):
                    reasons.append("decision_comparison_guard_threshold_mismatch")
                if not threshold_components_match(
                    comparison.get("guard_threshold_components"),
                    threshold_result.to_public_dict(),
                ):
                    reasons.append(
                        "decision_comparison_guard_threshold_components_mismatch"
                    )
                expected_guard_met = (
                    candidate_guard >= best_guard - threshold_result.threshold
                )
                if guard_met is not expected_guard_met:
                    reasons.append("decision_comparison_guard_met_mismatch")
        elif (
            guard_delta is not None
            or effective_regression is not None
            or bool(_mapping(comparison.get("guard_threshold_components")))
            or guard_met is not None
        ):
            reasons.append("decision_comparison_guard_fields_without_bound_scores")

    if acceptance_protocol is not None and acceptance_protocol.uses_named_guards:
        declared_guards = _mapping(comparison.get("named_guards"))
        expected_names = {guard.name for guard in acceptance_protocol.named_guards}
        if set(declared_guards) != expected_names:
            reasons.append("decision_comparison_named_guard_set_mismatch")
        for guard in acceptance_protocol.named_guards:
            declared = _mapping(declared_guards.get(guard.name))
            evidence = _mapping(
                acceptance_evidence.named_guard_evidence.get(guard.name)
            )
            evidence_stats = _mapping(evidence.get("score_stats"))
            candidate_guard = _finite_number(declared.get("candidate_score"))
            evidence_mean = _finite_number(evidence_stats.get("mean"))
            best_guard = _finite_number(declared.get("current_best_score"))
            best_stats = _mapping(declared.get("best_score_stats"))
            if (
                candidate_guard is None
                or evidence_mean is None
                or not _numbers_match(candidate_guard, evidence_mean)
                or dict(_mapping(declared.get("candidate_score_stats")))
                != dict(evidence_stats)
            ):
                reasons.append(
                    f"decision_comparison_named_guard_candidate_mismatch:{guard.name}"
                )
            declared_floor = _finite_number(
                declared.get("absolute_accept_floor")
            )
            declared_cap = _finite_number(declared.get("max_regression"))
            if (
                declared.get("score_source") != guard.score_source
                or declared_floor is None
                or not _numbers_match(
                    declared_floor, guard.absolute_accept_floor
                )
                or declared_cap is None
                or not _numbers_match(declared_cap, guard.max_regression)
            ):
                reasons.append(
                    f"decision_comparison_named_guard_contract_mismatch:{guard.name}"
                )
            threshold = _finite_number(declared.get("regression_threshold"))
            delta = _finite_number(declared.get("delta"))
            met = declared.get("met")
            if candidate_guard is None or best_guard is None:
                if any(value is not None for value in (threshold, delta, met)):
                    reasons.append(
                        f"decision_comparison_named_guard_unbound:{guard.name}"
                    )
                continue
            try:
                threshold_result = named_guard_regression_threshold(
                    acceptance_protocol,
                    guard,
                    candidate_guard_score=candidate_guard,
                    best_guard_score=best_guard,
                    candidate_guard_score_samples=evidence.get("score_samples"),
                    best_guard_score_samples=best_stats.get("score_samples"),
                )
            except ValueError:
                reasons.append(
                    f"decision_comparison_named_guard_threshold_invalid:{guard.name}"
                )
                continue
            expected_delta = candidate_guard - best_guard
            expected_met = (
                candidate_guard >= best_guard - threshold_result.threshold
            )
            if delta is None or not _numbers_match(delta, expected_delta):
                reasons.append(
                    f"decision_comparison_named_guard_delta_mismatch:{guard.name}"
                )
            if threshold is None or not _numbers_match(
                threshold, threshold_result.threshold
            ):
                reasons.append(
                    f"decision_comparison_named_guard_threshold_mismatch:{guard.name}"
                )
            if not threshold_components_match(
                declared.get("threshold_components"),
                threshold_result.to_public_dict(),
            ):
                reasons.append(
                    "decision_comparison_named_guard_threshold_components_mismatch:"
                    f"{guard.name}"
                )
            if met is not expected_met:
                reasons.append(
                    f"decision_comparison_named_guard_met_mismatch:{guard.name}"
                )

    delta_fields_present = any(
        key in comparison for key in ("delta", "threshold", "delta_met")
    )
    if delta_fields_present:
        delta = _finite_number(comparison.get("delta"))
        threshold = _finite_number(comparison.get("threshold"))
        delta_met = comparison.get("delta_met")
        if threshold is None:
            reasons.append("decision_comparison_delta_fields_invalid")
        elif delta is None:
            if delta_met is not None:
                reasons.append("decision_comparison_delta_fields_invalid")
        elif not isinstance(delta_met, bool):
            reasons.append("decision_comparison_delta_fields_invalid")
        elif (
            acceptance_protocol is not None
            and delta_met
            != acceptance_delta_met(acceptance_protocol, delta, threshold)
        ):
            reasons.append("decision_comparison_delta_met_mismatch")
        elif acceptance_protocol is None and delta_met != (delta > threshold):
            reasons.append("decision_comparison_delta_met_mismatch")
        current_best_score = _finite_number(comparison.get("current_best_score"))
        if candidate_score is not None and current_best_score is not None:
            expected_delta = candidate_score - current_best_score
            if delta is None or not _numbers_match(delta, expected_delta):
                reasons.append("decision_comparison_delta_arithmetic_mismatch")
        elif delta is not None:
            reasons.append("decision_comparison_delta_without_bound_scores")
    return reasons


def _reason_evidence_reasons(
    *,
    reason: str,
    record: Mapping[str, Any],
    public_result: Mapping[str, Any],
    metrics: Mapping[str, Any],
    comparison: Mapping[str, Any],
    acceptance_evidence: AcceptanceEvidenceValidation,
    acceptance_protocol: AcceptanceProtocol | None,
) -> list[str]:
    consistent = True
    if reason == "absolute_accept_floor":
        consistent = _absolute_floor_consistent(comparison, accepted=True)
    elif reason == "absolute_and_guard_accept_floors":
        consistent = _guarded_absolute_floor_consistent(comparison, accepted=True)
    elif reason == "absolute_and_named_guard_accept_floors":
        consistent = _named_guarded_absolute_floor_consistent(
            comparison, accepted=True
        )
    elif reason == "absolute_accept_floor_not_met":
        consistent = _absolute_floor_consistent(comparison, accepted=False)
    elif reason == "absolute_or_guard_accept_floor_not_met":
        consistent = _guarded_absolute_floor_consistent(comparison, accepted=False)
    elif reason == "absolute_or_named_guard_accept_floor_not_met":
        consistent = _named_guarded_absolute_floor_consistent(
            comparison, accepted=False
        )
    elif reason in ACCEPTANCE_DELTA_ACCEPT_REASONS:
        consistent = (
            comparison.get("mode") in {"accepted", "reference"}
            and comparison.get("delta_met") is True
            and (
                acceptance_protocol is None
                or reason
                == acceptance_delta_reason(acceptance_protocol, accepted=True)
            )
        )
    elif reason in ACCEPTANCE_DELTA_REJECT_REASONS:
        consistent = (
            comparison.get("mode") in {"accepted", "reference"}
            and comparison.get("delta_met") is False
            and (
                acceptance_protocol is None
                or reason
                == acceptance_delta_reason(acceptance_protocol, accepted=False)
            )
        )
    elif reason in _DELTA_ACCEPT_REASONS:
        consistent = (
            comparison.get("mode") in {"accepted", "reference"}
            and comparison.get("delta_met") is True
        )
    elif reason in _DELTA_REJECT_REASONS:
        consistent = (
            comparison.get("mode") in {"accepted", "reference"}
            and comparison.get("delta_met") is False
        )
    elif reason == "guard_score_regression":
        consistent = (
            comparison.get("mode") in {"accepted", "reference"}
            and comparison.get("guard_met") is False
        )
    elif reason == "named_guard_score_regression":
        consistent = bool(
            comparison.get("mode") in {"accepted", "reference"}
            and any(
                _mapping(item).get("met") is False
                for item in _mapping(comparison.get("named_guards")).values()
            )
        )
    elif reason in {
        "candidate_named_guard_stats_missing",
        "named_guard_eval_stats_missing",
    }:
        consistent = not acceptance_evidence.named_guard_binding_valid
    elif reason in _ACCEPTANCE_REJECT_REASONS:
        consistent = acceptance_evidence.reject_reason == reason
    elif reason == "eval_protocol_hash_mismatch":
        current_best = _mapping(comparison.get("current_best"))
        best_hash = _nonempty_string(current_best.get("eval_protocol_hash"))
        consistent = bool(
            acceptance_evidence.protocol_binding_valid
            and best_hash
            and best_hash != acceptance_evidence.eval_protocol_hash
        )
    elif reason == "evaluation_binding_mismatch":
        current_best = _mapping(comparison.get("current_best"))
        candidate_hash = _nonempty_string(
            comparison.get("candidate_evaluation_binding_hash")
        )
        best_hash = _nonempty_string(
            current_best.get("evaluation_binding_hash")
        )
        consistent = not candidate_hash or not best_hash or candidate_hash != best_hash
    elif reason == "public_result_status_not_pass":
        consistent = public_result.get("status") != "pass"
    elif reason == "public_score_nonfinite":
        consistent = _finite_number(public_result.get("public_score")) is None
    elif reason == "candidate_eval_stats_missing":
        acceptance = _mapping(metrics.get("acceptance"))
        score_stats = _mapping(acceptance.get("score_stats"))
        consistent = _finite_number(score_stats.get("mean")) is None
    elif reason == "current_best_eval_stats_missing":
        consistent = comparison.get("mode") in {"accepted", "incomplete"}
    elif reason == "runner_activation_missing":
        runtime_gate = _mapping(record.get("runtime_activation_gate"))
        consistent = bool(
            runtime_gate.get("status") != "pass"
            and runtime_gate.get("reason") == "runner_activation_missing"
        )
    elif reason in _SURFACE_REASONS:
        surface = _mapping(record.get("candidate_surface"))
        consistent = surface.get("reason") == reason
    if consistent:
        return []
    return [f"{reason or 'missing'}_reason_evidence_mismatch"]


def _absolute_floor_consistent(
    comparison: Mapping[str, Any],
    *,
    accepted: bool,
) -> bool:
    candidate_score = _finite_number(comparison.get("candidate_score"))
    floor = _finite_number(comparison.get("absolute_accept_floor"))
    if comparison.get("mode") != "bootstrap" or candidate_score is None or floor is None:
        return False
    return candidate_score >= floor if accepted else candidate_score < floor


def _guarded_absolute_floor_consistent(
    comparison: Mapping[str, Any],
    *,
    accepted: bool,
) -> bool:
    candidate_score = _finite_number(comparison.get("candidate_score"))
    score_floor = _finite_number(comparison.get("absolute_accept_floor"))
    candidate_guard = _finite_number(comparison.get("candidate_guard_score"))
    guard_floor = _finite_number(comparison.get("guard_absolute_accept_floor"))
    if (
        comparison.get("mode") != "bootstrap"
        or candidate_score is None
        or score_floor is None
        or candidate_guard is None
        or guard_floor is None
    ):
        return False
    both_met = candidate_score >= score_floor and candidate_guard >= guard_floor
    return both_met if accepted else not both_met


def _named_guarded_absolute_floor_consistent(
    comparison: Mapping[str, Any],
    *,
    accepted: bool,
) -> bool:
    candidate_score = _finite_number(comparison.get("candidate_score"))
    score_floor = _finite_number(comparison.get("absolute_accept_floor"))
    guards = _mapping(comparison.get("named_guards"))
    if (
        comparison.get("mode") != "bootstrap"
        or candidate_score is None
        or score_floor is None
        or not guards
    ):
        return False
    guard_results: list[bool] = []
    for item in guards.values():
        guard_item = _mapping(item)
        score = _finite_number(guard_item.get("candidate_score"))
        floor = _finite_number(guard_item.get("absolute_accept_floor"))
        if score is None or floor is None:
            return False
        guard_results.append(score >= floor)
    all_met = candidate_score >= score_floor and all(guard_results)
    return all_met if accepted else not all_met


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _nonempty_string(value: Any) -> str:
    return value if isinstance(value, str) and value else ""


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _numbers_match(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9)


def _string_set(value: Any) -> set[str]:
    if not isinstance(value, list | tuple | set | frozenset):
        return set()
    return {str(item) for item in value if isinstance(item, str) and item}


__all__ = [
    "DIAGNOSTIC_IMPROVEMENT_RECORD_OUTCOME",
    "DecisionRecordValidation",
    "ENGINEERING_IMPROVEMENT_RECORD_OUTCOME",
    "POLICY_IMPROVEMENT_ACCEPT_OUTCOME",
    "REJECT_OUTCOME",
    "project_validated_decision",
    "record_has_valid_policy_accept",
    "validate_decision_record",
]
