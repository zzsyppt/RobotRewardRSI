"""Minimal Push feedback for the shared two-lane exploration scheduler."""

from dataclasses import dataclass
from typing import Any, Mapping

from true_rl_agent.agent_core.exploration import TaskExplorationAssessment
from true_rl_agent.tasks.push_recovery.config import PushRecoveryTaskConfig


@dataclass(frozen=True)
class PushExplorationProjector:
    config: PushRecoveryTaskConfig
    task_id: str = "unitree_push_recovery"

    def contract(self) -> Mapping[str, Any]:
        return {
            "schema_version": "true_rl_agent.task.push_recovery.exploration.v1",
            "task_id": self.task_id,
            "evaluation_protocol_hash": self.config.evaluation_protocol().protocol_hash,
            "acceptance_protocol_hash": self.config.acceptance_protocol().protocol_hash,
            "rule": "public_scores_without_cross_cohort_effect_inference",
            "sealed_inputs_allowed": False,
            "acceptance_authority": False,
        }

    def assess_trial(self, *, generation, trial, parent) -> TaskExplorationAssessment:
        evidence = {}
        for phase in ("public_dev", "public_accept"):
            projection = trial.get(phase)
            if isinstance(projection, Mapping):
                summary = projection["public_summary"]
                stats = summary.get("score_stats", {})
                if stats.get("mean") is not None:
                    evidence[phase] = {key: stats[key] for key in
                        ("count", "mean", "standard_error") if key in stats}
        # A dev score alone cannot establish improvement over a parent's accept
        # cohort. Do not create negative suppression from that comparison.
        return TaskExplorationAssessment(
            observed_effect="unknown", value_label="unknown",
            public_evidence_summary=evidence,
            reason_codes=("public_scores_only_no_effect_inference",),
        )
