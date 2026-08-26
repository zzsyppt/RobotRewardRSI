"""Bounded public-only memory summaries for Stair evaluation history."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Any, Mapping, Sequence

from true_rl_agent.agent_core.context import sanitize_public_context
from true_rl_agent.agent_core.exploration import (
    GenerationExplorationPlan,
    build_generation_exploration_plan,
)
from true_rl_agent.agent_core.generation_contract import GenerationSessionContract
from true_rl_agent.evaluation.acceptance_replay import ProjectedPublicHistory
from true_rl_agent.tasks.stair_climbing.config import (
    STAIR_TASK_ID,
    StairClimbingTaskConfig,
)
from true_rl_agent.tasks.stair_climbing.exploration import StairExplorationProjector

_MAX_TREND_ROUNDS = 8


@dataclass(frozen=True)
class StairTaskMemoryAdapter:
    config: StairClimbingTaskConfig | None = None
    task_id: str = STAIR_TASK_ID

    def build_exploration_projection(
        self,
        history: ProjectedPublicHistory,
        *,
        generation_session_contract: GenerationSessionContract | None,
    ) -> Mapping[str, Any]:
        has_generation = any(
            isinstance(record.get("generation"), Mapping) for record in history.records
        )
        if not has_generation:
            return {}
        plan = self.build_generation_exploration_plan(
            history,
            generation_session_contract=generation_session_contract,
        )
        if not plan.index.cards:
            return {}
        return plan.prompt_projection

    def build_exploration_projection_from_validated_history(
        self,
        history: ProjectedPublicHistory,
        *,
        generation_session_contract: GenerationSessionContract | None,
    ) -> Mapping[str, Any]:
        """Build the internal index before prompt-only history fields are scrubbed."""

        return self.build_exploration_projection(
            history,
            generation_session_contract=generation_session_contract,
        )

    def build_generation_exploration_plan(
        self,
        history: ProjectedPublicHistory,
        *,
        generation_session_contract: GenerationSessionContract | None,
    ) -> GenerationExplorationPlan:
        if self.config is None:
            raise ValueError(
                "Stair generation history requires the resolved task configuration"
            )
        if (
            generation_session_contract is None
            or generation_session_contract.task_id != self.task_id
        ):
            raise ValueError(
                "Stair generation history requires its bound generation session contract"
            )
        projector = StairExplorationProjector(self.config)
        return build_generation_exploration_plan(
            history,
            task_projector=projector,
            expected_projector_contract_hash=(
                generation_session_contract.bound_contract_hashes[
                    "exploration_projector"
                ]
            ),
        )

    def build_public_summary(
        self,
        public_records: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        objective_summaries: list[dict[str, Any]] = []
        for record in public_records:
            if _record_task_id(record) not in (None, self.task_id):
                continue
            metrics = _metrics(record)
            objective_accept = _mapping(metrics.get("acceptance_objective"))
            objective_dev = _mapping(metrics.get("generalization_public_dev"))
            objective_score = _score(objective_accept)
            if objective_score is not None and objective_dev:
                arrival_rate = _diagnostic_rate(
                    objective_dev,
                    "first_valid_top_arrival_achieved",
                )
                objective_summaries.append(
                    {
                        "stable_top_success_rate": objective_score,
                        "valid_top_arrival_rate": arrival_rate,
                        "top_arrival_without_stable_success_rate": (
                            max(0.0, arrival_rate - objective_score)
                            if arrival_rate is not None
                            else None
                        ),
                        "dominant_failure": _dominant_failure(objective_dev),
                        "mean_max_progress_m": _mean_progress(objective_dev),
                    }
                )
        if not objective_summaries:
            return {}

        bounded = objective_summaries[-_MAX_TREND_ROUNDS:]
        scores = [
            float(item["stable_top_success_rate"]) for item in objective_summaries
        ]
        latest = objective_summaries[-1]
        return sanitize_public_context(
            {
                "schema_version": "true_rl_agent.stair_memory.public.v3",
                "task_id": self.task_id,
                "metric_scope": "stable_top_generalization_objective",
                "score_source": "acceptance_objective.score_stats.mean",
                "diagnostic_source": "generalization_public_dev",
                "failure_summary_source": "generalization_public_dev",
                "rounds_with_validated_public_stair_metrics": len(objective_summaries),
                "stable_top_success": {
                    "latest": scores[-1],
                    "best": max(scores),
                    "mean": fmean(scores),
                },
                "recent_trend": bounded,
                "latest_dominant_failure": latest["dominant_failure"],
                "latest_mean_max_progress_m": latest["mean_max_progress_m"],
            }
        )


def _record_task_id(record: Mapping[str, Any]) -> str | None:
    task = _mapping(record.get("task"))
    value = task.get("task_id")
    return value if isinstance(value, str) else None


def _metrics(record: Mapping[str, Any]) -> Mapping[str, Any]:
    public_result = _mapping(record.get("public_result"))
    return _mapping(public_result.get("metrics"))


def _score(public_accept: Mapping[str, Any]) -> float | None:
    stats = _mapping(public_accept.get("score_stats"))
    value = stats.get("mean")
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _dominant_failure(public_dev: Mapping[str, Any]) -> Mapping[str, Any]:
    breakdown = _mapping(public_dev.get("primary_failure_breakdown"))
    rates = {
        str(reason): float(rate)
        for reason, rate in breakdown.items()
        if isinstance(reason, str)
        and not isinstance(rate, bool)
        and isinstance(rate, int | float)
    }
    if not rates:
        return {"status": "not_available"}
    reason, rate = max(rates.items(), key=lambda item: (item[1], item[0]))
    return {"status": "available", "reason": reason, "rate": rate}


def _mean_progress(public_dev: Mapping[str, Any]) -> float | None:
    progress = _mapping(public_dev.get("progress_summary"))
    value = progress.get("mean_max_progress_m")
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _diagnostic_rate(public_dev: Mapping[str, Any], name: str) -> float | None:
    rates = _mapping(public_dev.get("diagnostic_flag_rates"))
    value = rates.get(name)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


__all__ = ["StairTaskMemoryAdapter"]
