"""Stable candidate decision outcome names."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from true_rl_agent.candidates.schema import CandidateSurfaceAssessment
from true_rl_agent.evaluation.acceptance_contract import (
    AcceptanceEvidenceValidation,
    validate_acceptance_evidence,
)
from true_rl_agent.evaluation.acceptance_phase import acceptance_comparison_entered
from true_rl_agent.evaluation.acceptance_protocol import (
    AcceptanceProtocol,
    noise_aware_threshold,
)
from true_rl_agent.evaluation.binding import EvaluationBindingV1
from true_rl_agent.evaluation.current_best import CurrentBestEvalStats
from true_rl_agent.evaluation.decision_record_contract import (
    DIAGNOSTIC_IMPROVEMENT_RECORD_OUTCOME,
    ENGINEERING_IMPROVEMENT_RECORD_OUTCOME,
    POLICY_IMPROVEMENT_ACCEPT_OUTCOME,
    REJECT_OUTCOME,
)
from true_rl_agent.evaluation.public_result import PublicEvalResult


class DecisionOutcome(str, Enum):
    POLICY_IMPROVEMENT_ACCEPT = POLICY_IMPROVEMENT_ACCEPT_OUTCOME
    DIAGNOSTIC_IMPROVEMENT_RECORD = DIAGNOSTIC_IMPROVEMENT_RECORD_OUTCOME
    ENGINEERING_IMPROVEMENT_RECORD = ENGINEERING_IMPROVEMENT_RECORD_OUTCOME
    REJECT = REJECT_OUTCOME


@dataclass(frozen=True)
class DecisionRecord:
    outcome: DecisionOutcome
    accepted: bool
    reason: str
    surface_classification: str
    public_score: float | None
    comparison: Mapping[str, Any] = field(default_factory=dict)
    consumed_gates: tuple[str, ...] = ()
    gate_statuses: Mapping[str, str] = field(default_factory=dict)

    def to_public_dict(self) -> dict[str, object]:
        return {
            "outcome": self.outcome.value,
            "accepted": self.accepted,
            "reason": self.reason,
            "surface_classification": self.surface_classification,
            "public_score": self.public_score,
            "comparison": dict(self.comparison),
            "consumed_gates": list(self.consumed_gates),
            "gate_statuses": dict(self.gate_statuses),
        }


def reject_without_eval(*, reason: str, surface_classification: str) -> DecisionRecord:
    return DecisionRecord(
        outcome=DecisionOutcome.REJECT,
        accepted=False,
        reason=reason,
        surface_classification=surface_classification,
        public_score=None,
    )


def decide_candidate(
    *,
    surface: CandidateSurfaceAssessment,
    public_result: PublicEvalResult,
    current_best: CurrentBestEvalStats,
    acceptance_protocol: AcceptanceProtocol,
) -> DecisionRecord:
    if surface.classification == "parameter_only":
        return DecisionRecord(
            DecisionOutcome.REJECT,
            False,
            "parameter_only_candidate",
            surface.classification,
            public_result.public_score,
        )
    if surface.classification == "diagnostic_only":
        return DecisionRecord(
            DecisionOutcome.DIAGNOSTIC_IMPROVEMENT_RECORD,
            False,
            "diagnostic_only_candidate",
            surface.classification,
            public_result.public_score,
        )
    if surface.classification == "engineering_only":
        return DecisionRecord(
            DecisionOutcome.ENGINEERING_IMPROVEMENT_RECORD,
            False,
            "engineering_only_candidate",
            surface.classification,
            public_result.public_score,
        )
    if not surface.eligible_for_policy_accept:
        return DecisionRecord(
            DecisionOutcome.REJECT,
            False,
            surface.reason,
            surface.classification,
            public_result.public_score,
        )

    if not acceptance_comparison_entered(
        policy_eligible=surface.eligible_for_policy_accept,
        public_result_status=public_result.status,
    ):
        return DecisionRecord(
            DecisionOutcome.REJECT,
            False,
            "public_result_status_not_pass",
            surface.classification,
            public_result.public_score,
        )
    if _optional_numeric(public_result.public_score) is None:
        return DecisionRecord(
            DecisionOutcome.REJECT,
            False,
            "public_score_nonfinite",
            surface.classification,
            public_result.public_score,
        )

    acceptance_validation = validate_acceptance_evidence(
        public_result.metrics,
        acceptance_protocol=acceptance_protocol,
        public_score=public_result.public_score,
    )
    gate_statuses = dict(acceptance_validation.gate_statuses)
    consumed_gates = tuple(gate_statuses)
    comparison = _comparison_summary(
        public_result=public_result,
        acceptance_validation=acceptance_validation,
        current_best=current_best,
        acceptance_protocol=acceptance_protocol,
    )
    if acceptance_validation.reject_reason is not None:
        return DecisionRecord(
            DecisionOutcome.REJECT,
            False,
            acceptance_validation.reject_reason,
            surface.classification,
            public_result.public_score,
            comparison,
            consumed_gates,
            gate_statuses,
        )
    if current_best.mode == "incomplete":
        return DecisionRecord(
            DecisionOutcome.REJECT,
            False,
            "current_best_eval_stats_missing",
            surface.classification,
            public_result.public_score,
            comparison,
            consumed_gates,
            gate_statuses,
        )
    try:
        candidate_binding = EvaluationBindingV1.from_mapping(
            public_result.metrics.get("evaluation_binding")
        )
        current_best_binding = current_best.evaluation_binding()
    except (TypeError, ValueError):
        return DecisionRecord(
            DecisionOutcome.REJECT,
            False,
            "evaluation_binding_mismatch",
            surface.classification,
            public_result.public_score,
            comparison,
            consumed_gates,
            gate_statuses,
        )
    if candidate_binding != current_best_binding:
        return DecisionRecord(
            DecisionOutcome.REJECT,
            False,
            "evaluation_binding_mismatch",
            surface.classification,
            public_result.public_score,
            comparison,
            consumed_gates,
            gate_statuses,
        )

    if current_best.mode == "bootstrap":
        candidate_score = acceptance_validation.score_mean
        if candidate_score is None:
            return DecisionRecord(
                DecisionOutcome.REJECT,
                False,
                "candidate_eval_stats_missing",
                surface.classification,
                public_result.public_score,
                comparison,
                consumed_gates,
                gate_statuses,
            )
        floor = acceptance_protocol.absolute_accept_floor
        if candidate_score >= floor:
            return DecisionRecord(
                DecisionOutcome.POLICY_IMPROVEMENT_ACCEPT,
                True,
                "absolute_accept_floor",
                surface.classification,
                public_result.public_score,
                comparison,
                consumed_gates,
                gate_statuses,
            )
        return DecisionRecord(
            DecisionOutcome.REJECT,
            False,
            "absolute_accept_floor_not_met",
            surface.classification,
            public_result.public_score,
            comparison,
            consumed_gates,
            gate_statuses,
        )

    if not current_best.has_full_eval_stats():
        return DecisionRecord(
            DecisionOutcome.REJECT,
            False,
            "current_best_eval_stats_missing",
            surface.classification,
            public_result.public_score,
            comparison,
            consumed_gates,
            gate_statuses,
        )
    candidate_score = acceptance_validation.score_mean
    current_best_score = current_best.score_for_comparison()
    if candidate_score is None or current_best_score is None:
        return DecisionRecord(
            DecisionOutcome.REJECT,
            False,
            "candidate_eval_stats_missing" if candidate_score is None else "current_best_eval_stats_missing",
            surface.classification,
            public_result.public_score,
            comparison,
            consumed_gates,
            gate_statuses,
        )
    delta = candidate_score - current_best_score
    threshold = _acceptance_threshold(
        acceptance_protocol=acceptance_protocol,
        acceptance_validation=acceptance_validation,
        current_best=current_best,
    )
    delta_met = delta > threshold
    if delta_met:
        return DecisionRecord(
            DecisionOutcome.POLICY_IMPROVEMENT_ACCEPT,
            True,
            "noise_aware_delta",
            surface.classification,
            public_result.public_score,
            comparison,
            consumed_gates,
            gate_statuses,
        )
    return DecisionRecord(
        DecisionOutcome.REJECT,
        False,
        "noise_aware_delta_not_met",
        surface.classification,
        public_result.public_score,
        comparison,
        consumed_gates,
        gate_statuses,
    )


def _acceptance_threshold(
    *,
    acceptance_protocol: AcceptanceProtocol,
    acceptance_validation: AcceptanceEvidenceValidation,
    current_best: CurrentBestEvalStats,
) -> float:
    candidate_se = acceptance_validation.standard_error
    best_se = _standard_error(current_best.public_accept_stats)
    return noise_aware_threshold(
        acceptance_protocol,
        candidate_standard_error=candidate_se,
        best_standard_error=best_se,
    )


def _comparison_summary(
    *,
    public_result: PublicEvalResult,
    acceptance_validation: AcceptanceEvidenceValidation,
    current_best: CurrentBestEvalStats,
    acceptance_protocol: AcceptanceProtocol,
) -> dict[str, Any]:
    candidate_score = acceptance_validation.score_mean
    best_score = current_best.score_for_comparison()
    threshold = _acceptance_threshold(
        acceptance_protocol=acceptance_protocol,
        acceptance_validation=acceptance_validation,
        current_best=current_best,
    )
    delta = (
        candidate_score - best_score
        if candidate_score is not None and best_score is not None
        else None
    )
    return {
        "mode": current_best.mode,
        "score_source": acceptance_validation.score_source,
        "acceptance_protocol_hash": acceptance_protocol.protocol_hash,
        "candidate_score": _round_optional(candidate_score),
        "current_best_score": _round_optional(best_score),
        "delta": _round_optional(delta),
        "threshold": _round_optional(threshold),
        "delta_met": None if delta is None else delta > threshold,
        "absolute_accept_floor": _round_optional(
            acceptance_validation.absolute_accept_floor
        ),
        "candidate_score_stats": dict(acceptance_validation.score_stats),
        "best_score_stats": dict(current_best.public_accept_stats),
        "candidate_evaluation_binding_hash": _candidate_binding_hash(public_result),
        "current_best": current_best.comparison_identity(),
        "public_score": public_result.public_score,
    }


def _round_optional(value: float | None) -> float | None:
    return None if value is None else round(float(value), 6)


def _standard_error(value: Any) -> float | None:
    if not isinstance(value, Mapping):
        return None
    return _optional_numeric(value.get("standard_error"))


def _optional_numeric(value: Any) -> float | None:
    if not isinstance(value, int | float):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _candidate_binding_hash(public_result: PublicEvalResult) -> str | None:
    try:
        return EvaluationBindingV1.from_mapping(
            public_result.metrics.get("evaluation_binding")
        ).binding_hash
    except (TypeError, ValueError):
        return None
