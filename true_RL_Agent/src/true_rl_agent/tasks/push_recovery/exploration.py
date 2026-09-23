"""Minimal Push feedback for the shared two-lane exploration scheduler."""

from dataclasses import dataclass
import math
from typing import Any, Mapping

from true_rl_agent.agent_core.exploration import TaskExplorationAssessment
from true_rl_agent.tasks.push_recovery.config import PushRecoveryTaskConfig


@dataclass(frozen=True)
class PushExplorationProjector:
    config: PushRecoveryTaskConfig
    task_id: str = "unitree_push_recovery"

    def contract(self) -> Mapping[str, Any]:
        return {
            "schema_version": "true_rl_agent.task.push_recovery.exploration.v2",
            "task_id": self.task_id,
            "evaluation_protocol_hash": self.config.evaluation_protocol().protocol_hash,
            "acceptance_protocol_hash": self.config.acceptance_protocol().protocol_hash,
            "rule": "public_dev_only_descriptive_comparison_without_effect_inference",
            "sealed_inputs_allowed": False,
            "acceptance_authority": False,
        }

    def assess_trial(self, *, generation, trial, parent) -> TaskExplorationAssessment:
        evidence = {}
        projection = trial.get("public_dev")
        if isinstance(projection, Mapping):
            stats = projection["public_summary"].get("score_stats", {})
            if stats.get("mean") is not None:
                evidence["public_dev"] = {key: stats[key] for key in
                    ("count", "mean", "standard_error") if key in stats}
        comparison = public_dev_comparison(generation, trial)
        if comparison:
            evidence["objective"] = comparison
        # A descriptive dev delta does not prove a causal mechanism effect.
        # Acceptance and holdout evidence must never enter authoring cards.
        return TaskExplorationAssessment(
            observed_effect="unknown", value_label="unknown",
            public_evidence_summary=evidence,
            reason_codes=("public_dev_only_no_effect_inference",),
        )


def public_dev_parent_reference(generation: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only the measured parent dev score, never its acceptance score."""

    for key in ("public_dev_parent_reference", "guard_reference", "parent_public_dev"):
        reference = generation.get(key)
        if not isinstance(reference, Mapping):
            continue
        stats = reference.get("score_stats")
        if isinstance(stats, Mapping) and _finite_score(stats.get("mean")) is not None:
            return {"score_stats": {"mean": stats["mean"]}}
    return {}


def public_dev_comparison(generation: Mapping[str, Any], trial: Mapping[str, Any]) -> dict[str, Any]:
    phase = trial.get("public_dev")
    summary = phase.get("public_summary", {}) if isinstance(phase, Mapping) else {}
    candidate = _finite_score(summary.get("score_stats", {}).get("mean"))
    reference = public_dev_parent_reference(generation)
    parent = _finite_score(reference.get("score_stats", {}).get("mean"))
    if candidate is None or parent is None:
        return {}
    return {"score_source": "public_dev.score_stats.mean", "candidate_score": candidate,
            "parent_score": parent, "delta": round(candidate - parent, 6)}


def _finite_score(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        return None
    return float(value)
