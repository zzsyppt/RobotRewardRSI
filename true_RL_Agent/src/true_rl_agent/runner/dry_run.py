"""Dry-run runner used by CLI audits without launching training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from true_rl_agent.evaluation.public_result import PublicEvalResult
from true_rl_agent.runner.base import Runner, TrainJobResult
from true_rl_agent.runner.checkpoint import seal_checkpoint
from true_rl_agent.runner.specs import CheckpointRef, EvalJobSpec, TrainJobSpec
from true_rl_agent.tasks.push_recovery.metrics import build_push_public_result


@dataclass(frozen=True)
class DryRunCandidateMetadata:
    changed_components: tuple[str, ...]
    public_score: float
    failure_breakdown: Mapping[str, float] = field(default_factory=dict)
    motion_quality: Mapping[str, Any] = field(default_factory=dict)


class DryRunAuditRunner(Runner):
    """Runner-owned deterministic dry-run evidence for audit loops."""

    def __init__(self, candidate_metadata: Mapping[str, DryRunCandidateMetadata]) -> None:
        self.candidate_metadata = dict(candidate_metadata)

    def run_train(self, train_spec: TrainJobSpec) -> TrainJobResult:
        metadata = self._metadata_for(train_spec.candidate_id)
        components = set(metadata.changed_components)
        code_level_active = bool(components & {"reward_callable", "control_logic", "recovery_logic"})
        diagnostic_active = "diagnostic" in components
        raw_checkpoint = CheckpointRef(
            uri=f"artifact://{train_spec.job_id}/best.pt",
            checksum=f"sha256:{train_spec.candidate_id}",
            role="train_output",
        )
        return TrainJobResult(
            job_id=train_spec.job_id,
            checkpoint=seal_checkpoint(raw_checkpoint, sealed_uri=f"artifact://{train_spec.job_id}/sealed/best.pt"),
            summary={
                "dry_run": True,
                "formal_training_executed": False,
                "budget_steps": train_spec.budget_steps,
                "training": {
                    "budget_accounting": {
                        "status": "pass",
                        "budget_steps": train_spec.budget_steps,
                        "planned_env_steps": train_spec.budget_steps,
                    }
                },
                "runtime_activation_summary": {
                    "reward_term_calls": 4 if code_level_active else 0,
                    "diagnostic_calls": 1 if diagnostic_active else 0,
                    "nonzero_reward_contribution": code_level_active,
                    "reward_output_stats": (
                        {"mean": 0.4, "finite_fraction": 1.0} if code_level_active else {}
                    ),
                    "diagnostic_outputs": (
                        {"push_recovery_margin": 0.2} if diagnostic_active else {}
                    ),
                    "feature_activation_records": [],
                    "capability_activation_records": sorted(components),
                },
            },
        )

    def run_eval(self, eval_spec: EvalJobSpec) -> PublicEvalResult:
        metadata = self._metadata_for(eval_spec.candidate_id)
        return build_push_public_result(
            success_rate=metadata.public_score,
            failure_breakdown=metadata.failure_breakdown,
            motion_quality=metadata.motion_quality,
        )

    def _metadata_for(self, candidate_id: str) -> DryRunCandidateMetadata:
        try:
            return self.candidate_metadata[candidate_id]
        except KeyError as exc:
            raise KeyError(f"missing dry-run metadata for candidate: {candidate_id}") from exc
