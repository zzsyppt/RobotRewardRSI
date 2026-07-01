"""Stable candidate decision outcome names."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from true_rl_agent.candidates.schema import CandidateSurfaceAssessment
from true_rl_agent.evaluation.public_result import PublicEvalResult


class DecisionOutcome(str, Enum):
    POLICY_IMPROVEMENT_ACCEPT = "policy_improvement_accept"
    DIAGNOSTIC_IMPROVEMENT_RECORD = "diagnostic_improvement_record"
    ENGINEERING_IMPROVEMENT_RECORD = "engineering_improvement_record"
    REJECT = "reject"


@dataclass(frozen=True)
class DecisionRecord:
    outcome: DecisionOutcome
    accepted: bool
    reason: str
    surface_classification: str
    public_score: float | None

    def to_public_dict(self) -> dict[str, object]:
        return {
            "outcome": self.outcome.value,
            "accepted": self.accepted,
            "reason": self.reason,
            "surface_classification": self.surface_classification,
            "public_score": self.public_score,
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
    current_best_score: float,
    policy_accept_delta: float,
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

    required_score = current_best_score + policy_accept_delta
    if public_result.status == "pass" and public_result.public_score >= required_score:
        return DecisionRecord(
            DecisionOutcome.POLICY_IMPROVEMENT_ACCEPT,
            True,
            "public_score_delta_met",
            surface.classification,
            public_result.public_score,
        )
    return DecisionRecord(
        DecisionOutcome.REJECT,
        False,
        "public_score_delta_not_met",
        surface.classification,
        public_result.public_score,
    )

