"""Trusted public-eval projection for Phase 8E.

This module is deliberately pure Python: it does not import Isaac, torch, or
candidate code.  Trusted eval processes use it to convert runner-owned rollout
summaries into the public_result schema consumed by Phase 8E decisions.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Mapping, Sequence


PUBLIC_RESULT_SCHEMA = "humanoid_agent.phase8e.public_result.v1"

DEFAULT_MOTION_LIMITS: dict[str, float] = {
    "max_mean_action_l2": 1000.0,
    "max_mean_action_rate_l2": 1000.0,
    "max_mean_final_xy_drift_m": 10.0,
    "max_mean_max_abs_pitch_rad": math.pi,
    "max_mean_max_abs_roll_rad": math.pi,
}

FORBIDDEN_SOURCE_TOKENS = (
    "iid_hidden",
    "iid-hidden",
    "phase2_iid_hidden",
    "iid_hidden_eval",
    "hidden_audit",
    "hidden_eval",
)


def load_json_mapping(path: str | Path) -> dict[str, Any]:
    import json

    loaded = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return loaded


def build_phase8e_public_result(
    *,
    public_dev_summaries: Sequence[Mapping[str, Any]],
    public_accept_summaries: Sequence[Mapping[str, Any]] | None = None,
    budget_summary: Mapping[str, Any] | None = None,
    motion_limits: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Build a public-only Phase 8E result from trusted rollout summaries."""

    errors: list[dict[str, Any]] = []
    limits = {**DEFAULT_MOTION_LIMITS, **dict(motion_limits or {})}
    dev = _project_split(public_dev_summaries, split_name="public_dev", errors=errors)
    accept = _project_split(public_accept_summaries or [], split_name="public_accept", errors=errors)
    all_summaries = list(public_dev_summaries) + list(public_accept_summaries or [])
    motion_gate = _motion_quality_gate(all_summaries, limits=limits, prior_errors=errors)
    budget_gate = _budget_accounting_gate(budget_summary)

    result: dict[str, Any] = {
        "schema_version": PUBLIC_RESULT_SCHEMA,
        "status": "fail" if errors else "pass",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "public_dev": dev,
        "budget_accounting": budget_gate,
        "motion_quality_gate": motion_gate,
        "ood_stress_gate": {
            "status": "not_run",
            "required": False,
            "reason": "Phase 8E-6-1 only implements fixed public eval plus minimal motion sanity; OOD stress is scheduled for a later gate.",
        },
        "errors": errors,
    }
    if public_accept_summaries:
        result["public_accept"] = accept
    return result


def _project_split(
    summaries: Sequence[Mapping[str, Any]],
    *,
    split_name: str,
    errors: list[dict[str, Any]],
) -> dict[str, Any]:
    seed_results: list[dict[str, Any]] = []
    success_rates: list[float] = []
    for index, summary in enumerate(summaries, start=1):
        if _has_forbidden_source_token(summary):
            errors.append({"split": split_name, "run_index": index, "reason": "forbidden_source_token"})
            seed_results.append(
                {
                    "run_index": index,
                    "status": "reject",
                    "success_rate": None,
                    "episodes": None,
                    "failure_breakdown": None,
                    "condition_flag_breakdown": None,
                    "secondary_metrics": None,
                    "reason": "forbidden_source_token",
                }
            )
            continue
        status = str(summary.get("status", "missing"))
        aggregate = _mapping(summary.get("aggregate"))
        success_rate = _optional_finite_float(aggregate.get("success_rate"))
        if status != "pass":
            errors.append({"split": split_name, "run_index": index, "reason": "summary_status_not_pass", "status": status})
        if success_rate is None:
            errors.append({"split": split_name, "run_index": index, "reason": "success_rate_missing_or_nonfinite"})
        else:
            success_rates.append(success_rate)
        seed_results.append(
            {
                "run_index": index,
                "status": status,
                "success_rate": success_rate,
                "episodes": _optional_int(aggregate.get("episodes")),
                "failure_breakdown": aggregate.get("failure_breakdown"),
                "condition_flag_breakdown": aggregate.get("condition_flag_breakdown"),
                "secondary_metrics": _project_secondary_metrics(summary.get("secondary_metrics")),
            }
        )
    if not summaries and split_name == "public_dev":
        errors.append({"split": split_name, "reason": "missing_required_split"})
    return {
        "success_rate_stats": _stats(success_rates),
        "seed_results": seed_results,
    }


def _stats(values: Sequence[float]) -> dict[str, Any]:
    rounded = [_round(value) for value in values]
    if not values:
        return {"count": 0, "mean": None, "std": None, "standard_error": None, "values": []}
    if len(values) == 1:
        return {"count": 1, "mean": rounded[0], "std": 0.0, "standard_error": 0.0, "values": rounded}
    sample_std = stdev(values)
    return {
        "count": len(values),
        "mean": _round(mean(values)),
        "std": _round(sample_std),
        "standard_error": _round(sample_std / math.sqrt(len(values))),
        "values": rounded,
    }


def _motion_quality_gate(
    summaries: Sequence[Mapping[str, Any]],
    *,
    limits: Mapping[str, float],
    prior_errors: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    if not summaries:
        failures.append({"reason": "no_rollout_summaries"})
    for index, summary in enumerate(summaries, start=1):
        if summary.get("status") != "pass":
            failures.append({"run_index": index, "reason": "summary_status_not_pass", "status": summary.get("status")})
        secondary = _mapping(summary.get("secondary_metrics"))
        if not secondary:
            failures.append({"run_index": index, "reason": "secondary_metrics_missing"})
            continue
        _check_limit(failures, secondary, index, "mean_action_l2", limits["max_mean_action_l2"])
        _check_limit(failures, secondary, index, "mean_action_rate_l2", limits["max_mean_action_rate_l2"])
        _check_limit(failures, secondary, index, "mean_final_xy_drift_m", limits["max_mean_final_xy_drift_m"])
        _check_limit(failures, secondary, index, "mean_max_abs_pitch_rad", limits["max_mean_max_abs_pitch_rad"])
        _check_limit(failures, secondary, index, "mean_max_abs_roll_rad", limits["max_mean_max_abs_roll_rad"])
    if prior_errors:
        failures.append({"reason": "public_eval_split_errors_present", "count": len(prior_errors)})
    return {
        "status": "fail" if failures else "pass",
        "summary": "minimal_public_motion_sanity",
        "limits": {key: _round(value) for key, value in limits.items()},
        "failures": failures,
    }


def _check_limit(
    failures: list[dict[str, Any]],
    secondary: Mapping[str, Any],
    run_index: int,
    key: str,
    limit: float,
) -> None:
    value = _optional_finite_float(secondary.get(key))
    if value is None:
        failures.append({"run_index": run_index, "metric": key, "reason": "missing_or_nonfinite"})
        return
    if abs(value) > float(limit):
        failures.append({"run_index": run_index, "metric": key, "actual": _round(value), "limit": _round(limit), "reason": "limit_exceeded"})


def _budget_accounting_gate(budget_summary: Mapping[str, Any] | None) -> dict[str, Any]:
    if budget_summary is None:
        return {"status": "fail", "reason": "budget_summary_missing"}
    source = _mapping(budget_summary.get("budget_accounting")) or _mapping(budget_summary.get("training_accounting")) or budget_summary
    projected: dict[str, Any] = {
        "status": str(source.get("status", budget_summary.get("status", "unknown"))),
        "config_derived_total_timesteps": _optional_int(source.get("config_derived_total_timesteps")),
        "planned_env_steps": _optional_int(source.get("planned_env_steps")),
        "num_envs": _optional_int(source.get("num_envs")),
        "num_steps_per_env": _optional_int(source.get("num_steps_per_env")),
        "max_iterations": _optional_int(source.get("max_iterations")),
    }
    if projected["status"] != "pass":
        projected["reason"] = "budget_accounting_status_not_pass"
    return projected


def _project_secondary_metrics(value: Any) -> dict[str, Any] | None:
    metrics = _mapping(value)
    if not metrics:
        return None
    allowed = (
        "mean_episode_steps",
        "mean_return",
        "mean_action_l2",
        "mean_action_rate_l2",
        "mean_min_base_height_m",
        "mean_max_abs_pitch_rad",
        "mean_max_abs_roll_rad",
        "mean_final_xy_drift_m",
    )
    return {key: _round(float(metrics[key])) for key in allowed if _optional_finite_float(metrics.get(key)) is not None}


def _has_forbidden_source_token(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(_has_forbidden_source_token(key) or _has_forbidden_source_token(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return any(_has_forbidden_source_token(item) for item in value)
    if isinstance(value, str):
        lowered = value.lower()
        return any(token in lowered for token in FORBIDDEN_SOURCE_TOKENS)
    return False


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _round(value: float | int | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)
