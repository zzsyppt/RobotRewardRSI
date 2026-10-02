"""Narrow Stair capability object for phased evaluation and screening."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from true_rl_agent.evaluation.phases import (
    EvaluationAssembly,
    EvaluationPhase,
    EvaluationPhaseResult,
)
from true_rl_agent.evaluation.phased_adapter import GenerationWorkload
from true_rl_agent.composition.cell_manifest import ResolvedCellManifestV1
from true_rl_agent.runner.base import TrainJobResult
from true_rl_agent.runner.specs import EvalJobSpec
from true_rl_agent.tasks.stair_climbing.config import StairClimbingTaskConfig
from true_rl_agent.tasks.stair_climbing.phased_cost import (
    stair_phased_evaluation_cost_contract,
)
from true_rl_agent.tasks.stair_climbing.phased_evaluation import (
    assemble_stair_eval_phase_results,
    assemble_stair_reference_eval_phase_results,
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
    objective_family_mask_provider: (
        Callable[[], Mapping[str, Any] | None] | None
    ) = None

    @property
    def task_protocol_hash(self) -> str:
        return self.config.task_protocol_hash

    @property
    def parent_public_dev_source(self) -> Literal["genesis_receipt", "evaluate"]:
        return "genesis_receipt" if self.config.is_unified_protocol else "evaluate"

    def _objective_family_mask(self) -> Mapping[str, Any] | None:
        if not self.config.is_unified_protocol:
            return None
        if self.objective_family_mask_provider is None:
            raise ValueError("unified Stair adapter has no objective family mask source")
        mask = self.objective_family_mask_provider()
        if mask is None:
            raise ValueError("unified Stair objective family mask is not frozen")
        return mask

    def selection_contract(self) -> dict[str, object]:
        return stair_selection_contract(self.config)

    def cost_contract(self) -> dict[str, object]:
        return stair_phased_evaluation_cost_contract(self.config)

    def generation_workload(self, *, cell_manifest: ResolvedCellManifestV1) -> GenerationWorkload:
        if cell_manifest.base_task_protocol_hash != self.config.base_task_protocol_hash:
            raise ValueError("Stair workload cell differs from its task protocol")
        suite = self.config.ordered_evaluation_suite_config
        if suite is None:
            raise ValueError("new generation sessions require the current Stair suite")
        return GenerationWorkload(
            task_protocol_hash=self.task_protocol_hash,
            phased_evaluation_cost_contract_hash=self.cost_contract()["contract_hash"],
            train_steps_per_job=self.config.budget_steps,
            public_dev_seed_units_per_job=self.config.public_seed_count,
            public_accept_seed_units_per_job=self.config.public_accept_seed_count,
            sealed_validation_seed_units_per_job=self.config.sealed_validation_seed_count,
            max_eval_control_steps_per_seed_unit=(
                len(suite.private_cell_assignment("public_dev"))
                * suite.max_episode_horizon_control_steps
            ),
        )

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
            objective_family_mask=self._objective_family_mask(),
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
        summary = canonical_stair_phase_result_summary(
            config=self.config,
            expected_phase=phase_result.phase,
            result=phase_result,
        )
        if "behavior" in phase_result.task_evidence:
            from true_rl_agent.tasks.stair_climbing.behavior import project_stair_behavior
            summary = {**summary, "behavior": project_stair_behavior(phase_result.task_evidence["behavior"])}
        return summary

    def project_reference_public_dev_gate(self, *, phase_result):
        # Stair qualification retains its existing complete-suite decision.
        return None

    def project_public_accept_scores(self, *, phase_result):
        from true_rl_agent.evaluation.public_accept_scores import PublicAcceptScores
        from true_rl_agent.tasks.stair_climbing.unified_scoring import build_unified_score_projection

        if phase_result.phase is not EvaluationPhase.PUBLIC_ACCEPT:
            raise ValueError("score comparison requires public-accept evidence")
        if not self.config.is_unified_protocol:
            return None
        summary = self.project_public_phase_summary(phase_result=phase_result)
        projection = build_unified_score_projection(summary,
            objective_family_mask=self._objective_family_mask())
        primary = projection["j_primary"]
        return PublicAcceptScores(score_stats=primary["score_stats"],
            score_samples=tuple(primary["seed_macro_scores"]),
            named_guard_evidence={guard.name: {
                "score_stats": projection[guard.name]["score_stats"],
                "score_samples": projection[guard.name]["seed_macro_scores"]}
                for guard in self.config.acceptance_protocol.named_guards})

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
            objective_family_mask=self._objective_family_mask(),
        )

    def assemble_reference_full_summary(
        self,
        *,
        eval_plan: EvalJobSpec,
        phase_results: Sequence[EvaluationPhaseResult],
        phase_receipt_refs: Mapping[EvaluationPhase | str, Mapping[str, Any]],
    ) -> EvaluationAssembly:
        return assemble_stair_reference_eval_phase_results(
            config=self.config,
            eval_plan=eval_plan,
            phase_results=phase_results,
            phase_receipt_refs=phase_receipt_refs,
        )


__all__ = ["StairPhasedEvaluationAdapter"]
