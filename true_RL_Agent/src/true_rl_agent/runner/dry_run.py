"""Dry-run runner used by CLI audits without launching training."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Mapping

from true_rl_agent.evaluation.public_result import PublicEvalResult
from true_rl_agent.runner.base import Runner, RunnerExecutionError, TrainJobResult
from true_rl_agent.runner.checkpoint import seal_checkpoint
from true_rl_agent.runner.specs import (
    BaselineEvalJobSpec,
    CheckpointRef,
    EvalJobSpec,
    TrainJobSpec,
)


@dataclass(frozen=True)
class DryRunCandidateMetadata:
    changed_components: tuple[str, ...]
    public_score: float
    failure_breakdown: Mapping[str, float] = field(default_factory=dict)
    task_eval_metadata: Mapping[str, Any] = field(default_factory=dict)


DryRunEvalResultBuilder = Callable[[EvalJobSpec, DryRunCandidateMetadata], PublicEvalResult]
DryRunTrainSummaryBuilder = Callable[[TrainJobSpec, DryRunCandidateMetadata], Mapping[str, Any]]
DryRunBaselineEvalResultBuilder = Callable[[BaselineEvalJobSpec], PublicEvalResult]


class DryRunAuditRunner(Runner):
    """Runner-owned deterministic dry-run evidence for audit loops."""

    def __init__(
        self,
        candidate_metadata: Mapping[str, DryRunCandidateMetadata],
        *,
        eval_result_builder: DryRunEvalResultBuilder,
        train_summary_builder: DryRunTrainSummaryBuilder,
        baseline_eval_result_builder: DryRunBaselineEvalResultBuilder | None = None,
    ) -> None:
        self.candidate_metadata = dict(candidate_metadata)
        self.eval_result_builder = eval_result_builder
        self.train_summary_builder = train_summary_builder
        self.baseline_eval_result_builder = baseline_eval_result_builder

    def run_train(self, train_spec: TrainJobSpec) -> TrainJobResult:
        train_spec.training_start_checkpoint()
        metadata = self._metadata_for(train_spec.subject.subject_id)
        raw_checkpoint = CheckpointRef(
            uri=f"artifact://{train_spec.job_id}/best.pt",
            checksum=(
                "sha256:"
                + hashlib.sha256(
                    train_spec.subject.subject_id.encode("utf-8")
                ).hexdigest()
            ),
            role="train_output",
            metadata={
                "training_subject": train_spec.subject.to_public_dict(),
                "subject_manifest_hash": train_spec.subject_manifest_hash,
                "program_source_checksum": train_spec.program_source_checksum,
                "training_binding_hash": train_spec.training_binding.binding_hash,
                "capability_profile_hash": (
                    train_spec.training_binding.capability_profile_hash
                ),
            },
        )
        return TrainJobResult(
            job_id=train_spec.job_id,
            subject=train_spec.subject,
            subject_manifest_hash=train_spec.subject_manifest_hash,
            program_source_checksum=train_spec.program_source_checksum,
            training_binding_hash=str(train_spec.training_binding.binding_hash),
            capability_profile_hash=(
                train_spec.training_binding.capability_profile_hash
            ),
            checkpoint=seal_checkpoint(raw_checkpoint, sealed_uri=f"artifact://{train_spec.job_id}/sealed/best.pt"),
            summary=dict(self.train_summary_builder(train_spec, metadata)),
        )

    def run_eval(self, eval_spec: EvalJobSpec) -> PublicEvalResult:
        eval_spec.execution_checkpoint()
        metadata = self._metadata_for(eval_spec.subject.subject_id)
        return self.eval_result_builder(eval_spec, metadata)

    def run_baseline_eval(
        self,
        eval_spec: BaselineEvalJobSpec,
    ) -> PublicEvalResult:
        eval_spec.execution_checkpoint()
        if self.baseline_eval_result_builder is None:
            raise RunnerExecutionError(
                "dry-run runner does not define baseline eval evidence"
            )
        return self.baseline_eval_result_builder(eval_spec)

    def _metadata_for(self, candidate_id: str) -> DryRunCandidateMetadata:
        try:
            return self.candidate_metadata[candidate_id]
        except KeyError as exc:
            raise KeyError(f"missing dry-run metadata for candidate: {candidate_id}") from exc
