"""Stair-owned public exploration value projection.

This module classifies public trial evidence for revisit scheduling only.  It
does not accept candidates and never reads sealed rollout detail.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from true_rl_agent.agent_core.exploration import TaskExplorationAssessment
from true_rl_agent.evaluation.acceptance_protocol import (
    AcceptanceProtocol,
    acceptance_delta_met,
    acceptance_threshold,
    guard_regression_threshold,
)
from true_rl_agent.tasks.stair_climbing.config import (
    STAIR_TASK_ID,
    StairClimbingTaskConfig,
)
from true_rl_agent.tasks.stair_climbing.evaluation_contract import (
    validate_stair_split_summary,
)
from true_rl_agent.tasks.stair_climbing.generalization_contract import (
    validate_stair_generalization_summary,
)


STAIR_EXPLORATION_PROJECTOR_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.exploration_projector.v2"
)


class StairComparisonStats(Protocol):
    """Public comparison surface consumed by the Stair exploration projector."""

    def score_for_comparison(self) -> float | None: ...

    def score_samples_for_comparison(
        self,
        acceptance_protocol: AcceptanceProtocol,
    ) -> tuple[float, ...]: ...

    def guard_score_for_comparison(self) -> float | None: ...

    def guard_score_samples_for_comparison(
        self,
        acceptance_protocol: AcceptanceProtocol,
    ) -> tuple[float, ...]: ...


@dataclass(frozen=True)
class StairExplorationProjector:
    config: StairClimbingTaskConfig
    task_id: str = STAIR_TASK_ID

    def contract(self) -> Mapping[str, Any]:
        return {
            "schema_version": STAIR_EXPLORATION_PROJECTOR_SCHEMA_VERSION,
            "task_id": self.task_id,
            "task_protocol_hash": self.config.task_protocol_hash,
            "evaluation_protocol_hash": self.config.evaluation_protocol().protocol_hash,
            "acceptance_protocol_hash": self.config.acceptance_protocol.protocol_hash,
            "rule": "public_accept_headroom_guard_with_public_dev_consistency_v1",
            "revisit_rank_rule": (
                "task_owned_public_objective_delta_then_guard_delta_v1"
            ),
            "effect_labels": [
                "strong_positive",
                "positive",
                "mixed",
                "neutral",
                "negative",
                "unknown",
            ],
            "value_labels": ["high", "medium", "low", "unknown"],
            "sealed_inputs_allowed": False,
            "acceptance_authority": False,
        }

    def assess_trial(
        self,
        *,
        generation: Mapping[str, Any],
        trial: Mapping[str, Any],
        parent: StairComparisonStats,
    ) -> TaskExplorationAssessment:
        public_accept = _phase_summary(trial, "public_accept")
        public_dev = _phase_summary(trial, "public_dev")
        if public_accept is None:
            return TaskExplorationAssessment(
                observed_effect="unknown",
                value_label="unknown",
                public_evidence_summary={
                    "status": "not_public_complete",
                    "trial_status": trial.get("trial_status"),
                },
                reason_codes=("public_accept_evidence_missing",),
            )
        accept_anchor, accept_generalization = self._validated_summary(
            public_accept,
            split="public_accept",
        )
        if public_dev is not None:
            dev_anchor, dev_generalization = self._validated_summary(
                public_dev,
                split="public_dev",
            )
        else:
            dev_anchor = dev_generalization = None

        protocol = self.config.acceptance_protocol
        candidate_scores = tuple(
            float(value) for value in accept_generalization["seed_macro_scores"]
        )
        candidate_score = float(accept_generalization["score_stats"]["mean"])
        parent_score = parent.score_for_comparison()
        parent_scores = parent.score_samples_for_comparison(protocol)
        candidate_guard_scores = tuple(
            float(value) for value in accept_anchor["seed_macro_scores"]
        )
        candidate_guard = float(accept_anchor["score_stats"]["mean"])
        parent_guard = parent.guard_score_for_comparison()
        parent_guard_scores = parent.guard_score_samples_for_comparison(protocol)
        if (
            parent_score is None
            or len(parent_scores) != len(candidate_scores)
            or parent_guard is None
            or len(parent_guard_scores) != len(candidate_guard_scores)
        ):
            return TaskExplorationAssessment(
                observed_effect="unknown",
                value_label="unknown",
                public_evidence_summary={
                    "status": "parent_comparison_evidence_incomplete",
                    "candidate_objective": candidate_score,
                    "candidate_guard": candidate_guard,
                },
                reason_codes=("parent_public_comparison_evidence_missing",),
            )

        threshold = acceptance_threshold(
            protocol,
            candidate_score=candidate_score,
            best_score=parent_score,
            candidate_score_samples=candidate_scores,
            best_score_samples=parent_scores,
        )
        guard_threshold = guard_regression_threshold(
            protocol,
            candidate_guard_score=candidate_guard,
            best_guard_score=parent_guard,
            candidate_guard_score_samples=candidate_guard_scores,
            best_guard_score_samples=parent_guard_scores,
        )
        objective_delta = candidate_score - parent_score
        guard_delta = candidate_guard - parent_guard
        objective_met = acceptance_delta_met(
            protocol,
            objective_delta,
            threshold.threshold,
        )
        guard_met = guard_delta >= -guard_threshold.threshold

        dev_candidate = (
            float(dev_generalization["score_stats"]["mean"])
            if dev_generalization is not None
            else None
        )
        dev_guard = (
            float(dev_anchor["score_stats"]["mean"])
            if dev_anchor is not None
            else None
        )
        reference = generation.get("guard_reference")
        reference_general = (
            _numeric(_mapping(reference).get("generalization_score_stats"), "mean")
        )
        reference_guard = _numeric(
            _mapping(reference).get("anchor_score_stats"),
            "mean",
        )
        dev_reference_missing = public_dev is not None and (
            reference_general is None or reference_guard is None
        )
        dev_delta = (
            dev_candidate - reference_general
            if dev_candidate is not None and reference_general is not None
            else None
        )
        dev_guard_delta = (
            dev_guard - reference_guard
            if dev_guard is not None and reference_guard is not None
            else None
        )
        directions_conflict = (
            dev_delta is not None
            and objective_delta != 0.0
            and dev_delta != 0.0
            and (dev_delta > 0.0) != (objective_delta > 0.0)
        )

        if not guard_met:
            effect, value, reasons = (
                "negative",
                "low",
                ("public_guard_regression_exceeded",),
            )
        elif dev_reference_missing:
            return TaskExplorationAssessment(
                observed_effect="unknown",
                value_label="unknown",
                public_evidence_summary={
                    "status": "parent_public_dev_reference_incomplete",
                    "candidate_objective": candidate_score,
                    "candidate_guard": candidate_guard,
                    "candidate_public_dev_objective": dev_candidate,
                    "candidate_public_dev_guard": dev_guard,
                },
                reason_codes=("parent_public_dev_reference_missing",),
            )
        elif directions_conflict:
            effect, value, reasons = (
                "mixed",
                "medium",
                ("public_dev_accept_direction_conflict",),
            )
        elif objective_met and (dev_delta is None or dev_delta >= 0.0):
            effect, value, reasons = (
                "strong_positive",
                "high",
                ("public_objective_materiality_and_guard_met",),
            )
        elif objective_delta > 0.0:
            effect, value, reasons = (
                "positive",
                "medium",
                ("public_objective_positive_below_materiality",),
            )
        elif objective_delta <= 0.0 and guard_delta <= 0.0:
            effect, value, reasons = (
                "negative",
                "low",
                ("public_objective_and_guard_nonpositive",),
            )
        elif guard_delta > 0.0:
            effect, value, reasons = (
                "mixed",
                "medium",
                ("public_guard_improved_objective_not_improved",),
            )
        else:
            effect, value, reasons = (
                "neutral",
                "low",
                ("public_effect_neutral",),
            )

        summary = {
            "status": "public_complete",
            "objective": {
                "score_source": protocol.score_source,
                "candidate": candidate_score,
                "parent": parent_score,
                "delta": objective_delta,
                "materiality_threshold": threshold.threshold,
                "materiality_met": objective_met,
            },
            "guard": {
                "score_source": protocol.guard_score_source,
                "candidate": candidate_guard,
                "parent": parent_guard,
                "delta": guard_delta,
                "max_regression": guard_threshold.threshold,
                "guard_met": guard_met,
            },
            "public_dev": {
                "candidate_objective": dev_candidate,
                "parent_objective": reference_general,
                "objective_delta": dev_delta,
                "candidate_guard": dev_guard,
                "parent_guard": reference_guard,
                "guard_delta": dev_guard_delta,
                "direction_consistent_with_accept": not directions_conflict,
            },
            "formal_status": (
                "accepted"
                if trial.get("trial_status") == "accepted"
                else "formal_confirmation_failed"
                if trial.get("trial_status") == "formal_rejected"
                else "not_formally_evaluated"
            ),
        }
        return TaskExplorationAssessment(
            observed_effect=effect,
            value_label=value,
            public_evidence_summary=summary,
            reason_codes=reasons,
            revisit_rank_key=(
                round(objective_delta * 1_000_000),
                round(guard_delta * 1_000_000),
            ),
        )

    def _validated_summary(
        self,
        value: Mapping[str, Any],
        *,
        split: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if set(value) != {"anchor", "generalization"}:
            raise ValueError("Stair exploration phase summary fields mismatch")
        anchor = validate_stair_split_summary(
            self.config.protocol_v1,
            semantics=self.config.evaluation_semantics,
            evaluation_protocol=self.config.evaluation_protocol(),
            split=split,
            summary=value["anchor"],
        )
        generalization = validate_stair_generalization_summary(
            profile=self.config.evaluation_profile,
            semantics=self.config.evaluation_semantics,
            evaluation_protocol=self.config.evaluation_protocol(),
            split=split,
            summary=value["generalization"],
        )
        return anchor, generalization


def _phase_summary(
    trial: Mapping[str, Any],
    phase_name: str,
) -> Mapping[str, Any] | None:
    phase = trial.get(phase_name)
    if phase is None:
        return None
    if not isinstance(phase, Mapping):
        raise ValueError(f"Stair exploration {phase_name} projection is invalid")
    summary = phase.get("public_summary")
    if not isinstance(summary, Mapping):
        raise ValueError(f"Stair exploration {phase_name} summary is missing")
    return summary


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _numeric(value: Any, key: str) -> float | None:
    if not isinstance(value, Mapping):
        return None
    raw = value.get(key)
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        return None
    return float(raw)


__all__ = [
    "STAIR_EXPLORATION_PROJECTOR_SCHEMA_VERSION",
    "StairExplorationProjector",
]
