"""Stable candidate decision outcome names."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

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

    if public_result.status != "pass":
        return DecisionRecord(
            DecisionOutcome.REJECT,
            False,
            "public_result_status_not_pass",
            surface.classification,
            public_result.public_score,
        )
    if not math.isfinite(float(public_result.public_score)):
        return DecisionRecord(
            DecisionOutcome.REJECT,
            False,
            "public_score_nonfinite",
            surface.classification,
            public_result.public_score,
        )

    acceptance = _acceptance_metadata(public_result.metrics)
    failed_gates = _failed_required_gates(public_result.metrics, acceptance)
    if failed_gates:
        return DecisionRecord(
            DecisionOutcome.REJECT,
            False,
            "required_gate_failed",
            surface.classification,
            public_result.public_score,
        )
    if acceptance.get("missing_required_score") is True:
        return DecisionRecord(
            DecisionOutcome.REJECT,
            False,
            "required_score_missing",
            surface.classification,
            public_result.public_score,
        )

    if _absolute_floor_mode(current_best_score, acceptance):
        floor = _numeric(acceptance.get("absolute_accept_floor"), default=0.0)
        if public_result.public_score >= floor:
            return DecisionRecord(
                DecisionOutcome.POLICY_IMPROVEMENT_ACCEPT,
                True,
                "absolute_accept_floor",
                surface.classification,
                public_result.public_score,
            )
        return DecisionRecord(
            DecisionOutcome.REJECT,
            False,
            "absolute_accept_floor_not_met",
            surface.classification,
            public_result.public_score,
        )

    delta = public_result.public_score - current_best_score
    threshold = _acceptance_threshold(
        policy_accept_delta=policy_accept_delta,
        acceptance=acceptance,
    )
    delta_met = delta > threshold
    if delta_met:
        return DecisionRecord(
            DecisionOutcome.POLICY_IMPROVEMENT_ACCEPT,
            True,
            "noise_aware_delta" if acceptance else "public_score_delta_met",
            surface.classification,
            public_result.public_score,
        )
    return DecisionRecord(
        DecisionOutcome.REJECT,
        False,
        "noise_aware_delta_not_met" if acceptance else "public_score_delta_not_met",
        surface.classification,
        public_result.public_score,
    )


def _acceptance_metadata(metrics: Mapping[str, Any]) -> dict[str, Any]:
    raw = metrics.get("acceptance")
    return dict(raw) if isinstance(raw, Mapping) else {}


def _failed_required_gates(metrics: Mapping[str, Any], acceptance: Mapping[str, Any]) -> list[str]:
    raw_gate_names = acceptance.get("required_pass_gates", [])
    if not isinstance(raw_gate_names, list | tuple):
        return []
    failed = []
    for raw_name in raw_gate_names:
        name = str(raw_name)
        gate = metrics.get(name)
        if not isinstance(gate, Mapping) or gate.get("status") != "pass":
            failed.append(name)
    return failed


def _absolute_floor_mode(current_best_score: float, acceptance: Mapping[str, Any]) -> bool:
    if "absolute_accept_floor" not in acceptance:
        return False
    near_zero = _numeric(acceptance.get("near_zero_success_threshold"), default=0.02)
    return current_best_score <= near_zero


def _acceptance_threshold(*, policy_accept_delta: float, acceptance: Mapping[str, Any]) -> float:
    min_delta = max(
        float(policy_accept_delta),
        _numeric(acceptance.get("min_delta"), default=float(policy_accept_delta)),
    )
    candidate_se = _standard_error(acceptance.get("score_stats"))
    best_se = _standard_error(acceptance.get("best_score_stats"))
    pooled = None
    if candidate_se is not None and best_se is not None:
        pooled = math.sqrt(candidate_se**2 + best_se**2)
    elif candidate_se is not None:
        pooled = candidate_se
    return max(min_delta, pooled if pooled is not None else min_delta)


def _standard_error(value: Any) -> float | None:
    if not isinstance(value, Mapping):
        return None
    return _optional_numeric(value.get("standard_error"))


def _numeric(value: Any, *, default: float) -> float:
    parsed = _optional_numeric(value)
    return default if parsed is None else parsed


def _optional_numeric(value: Any) -> float | None:
    if not isinstance(value, int | float):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None
