"""Narrow Stair capability object for phased evaluation and screening."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from true_rl_agent.evaluation.phases import (
    EvaluationAssembly,
    EvaluationPhase,
    EvaluationPhaseResult,
)
from true_rl_agent.runner.base import TrainJobResult
from true_rl_agent.runner.specs import EvalJobSpec
from true_rl_agent.tasks.stair_climbing.config import StairClimbingTaskConfig
from true_rl_agent.tasks.stair_climbing.phased_cost import (
    stair_phased_evaluation_cost_contract,
)
from true_rl_agent.tasks.stair_climbing.phased_evaluation import (
    assemble_stair_eval_phase_results,
    canonical_stair_phase_result_summary,
)
from true_rl_agent.tasks.stair_climbing.selection import (
    StairPublicDevScreenEvidence,
    StairSelectionResult,
    build_stair_public_dev_screen_evidence,
    select_stair_public_dev_winner,
    stair_selection_contract,
)


@dataclass(frozen=True)
class StairPhasedEvaluationAdapter:
    """Expose Stair semantics without teaching the common runner about Stair."""

    config: StairClimbingTaskConfig

    def selection_contract(self) -> dict[str, object]:
        return stair_selection_contract(self.config)

    def cost_contract(self) -> dict[str, object]:
        return stair_phased_evaluation_cost_contract(self.config)

    def project_public_dev_screen(
        self,
        *,
        trial_id: str,
        phase_result: EvaluationPhaseResult,
        screen_role: str,
        train_result: TrainJobResult | None,
        train_receipt_checksum: str | None,
        mechanism_operation_delta: Mapping[str, Any] | None,
    ) -> StairPublicDevScreenEvidence:
        return build_stair_public_dev_screen_evidence(
            config=self.config,
            trial_id=trial_id,
            phase_result=phase_result,
            screen_role=screen_role,
            train_result=train_result,
            train_receipt_checksum=train_receipt_checksum,
            mechanism_operation_delta=mechanism_operation_delta,
        )

    def select(
        self,
        screens: Sequence[StairPublicDevScreenEvidence],
        *,
        guard_reference: StairPublicDevScreenEvidence,
    ) -> StairSelectionResult:
        return select_stair_public_dev_winner(
            config=self.config,
            screens=screens,
            guard_reference=guard_reference,
        )

    def project_public_phase_summary(
        self,
        *,
        phase_result: EvaluationPhaseResult,
    ) -> Mapping[str, Any]:
        """Return the task-owned canonical public summary for one public phase."""

        if phase_result.phase is EvaluationPhase.SEALED_VALIDATION:
            raise ValueError("sealed phase summaries cannot enter generation trial history")
        return canonical_stair_phase_result_summary(
            config=self.config,
            expected_phase=phase_result.phase,
            result=phase_result,
        )

    def assemble_full_summary(
        self,
        *,
        eval_plan: EvalJobSpec,
        trial_id: str,
        selection_result: StairSelectionResult,
        selection_screens: Sequence[StairPublicDevScreenEvidence],
        guard_reference: StairPublicDevScreenEvidence,
        phase_results: Sequence[EvaluationPhaseResult],
        phase_receipt_refs: Mapping[EvaluationPhase | str, Mapping[str, Any]],
    ) -> EvaluationAssembly:
        return assemble_stair_eval_phase_results(
            config=self.config,
            eval_plan=eval_plan,
            trial_id=trial_id,
            selection_result=selection_result,
            selection_screens=selection_screens,
            guard_reference=guard_reference,
            phase_results=phase_results,
            phase_receipt_refs=phase_receipt_refs,
        )


__all__ = ["StairPhasedEvaluationAdapter"]
