"""Push recovery public memory summaries."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Any, Mapping, Sequence

from true_rl_agent.agent_core.context import sanitize_public_context
from true_rl_agent.tasks.push_recovery.metrics import (
    classify_push_prompt_failure,
    extract_push_prompt_failure_breakdown,
)


@dataclass(frozen=True)
class PushTaskMemoryAdapter:
    """Summarize push recovery public eval history for future prompts."""

    task_id: str = "unitree_push_recovery"

    def build_public_summary(
        self,
        public_records: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        push_records = [
            record
            for record in public_records
            if _task_id(record) in (None, self.task_id)
            and isinstance(record.get("public_result"), Mapping)
            and bool(record.get("public_result"))
        ]
        records_with_metrics = [
            (record, metrics)
            for record in push_records
            for metrics in [_metrics(record)]
            if metrics
        ]
        if not records_with_metrics:
            return {}

        metrics_by_round = [metrics for _record, metrics in records_with_metrics]
        success_rates = [
            score
            for record, _metrics_by_round in records_with_metrics
            for score in [
                _validated_public_score(record)
            ]
            if score is not None
        ]
        latest_metrics = metrics_by_round[-1]
        latest_failure_breakdown = extract_push_prompt_failure_breakdown(latest_metrics)
        aggregate_failure_breakdown = _mean_numeric_mappings(
            [extract_push_prompt_failure_breakdown(metrics) for metrics in metrics_by_round]
        )

        summary = {
            "schema_version": "true_rl_agent.push_memory.public.v1",
            "task_id": self.task_id,
            "rounds_with_public_push_metrics": len(metrics_by_round),
            "success_rate": {
                "latest": success_rates[-1] if success_rates else None,
                "best": max(success_rates) if success_rates else None,
                "mean": fmean(success_rates) if success_rates else None,
            },
            "latest_failure_mode": classify_push_prompt_failure(latest_metrics),
            "latest_failure_breakdown": latest_failure_breakdown,
            "aggregate_failure_breakdown": aggregate_failure_breakdown,
            "latest_motion_quality": latest_metrics.get("motion_quality", {}),
        }
        return sanitize_public_context(summary)


def _metrics(record: Mapping[str, Any]) -> Mapping[str, Any]:
    public_result = record.get("public_result", {})
    if not isinstance(public_result, Mapping):
        return {}
    metrics = public_result.get("metrics", {})
    return metrics if isinstance(metrics, Mapping) else {}


def _validated_public_score(
    record: Mapping[str, Any],
) -> float | None:
    decision = record.get("decision")
    if not isinstance(decision, Mapping):
        return None
    score = decision.get("public_score")
    if isinstance(score, bool) or not isinstance(score, int | float):
        return None
    return float(score)


def _task_id(record: Mapping[str, Any]) -> str | None:
    task = record.get("task", {})
    if not isinstance(task, Mapping):
        return None
    raw_task_id = task.get("task_id")
    return str(raw_task_id) if raw_task_id is not None else None


def _mean_numeric_mappings(items: Sequence[Any]) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    for item in items:
        if not isinstance(item, Mapping):
            continue
        for key, value in item.items():
            if isinstance(value, int | float):
                values.setdefault(str(key), []).append(float(value))
    return {key: fmean(nums) for key, nums in sorted(values.items())}
