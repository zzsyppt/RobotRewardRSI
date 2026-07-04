"""Public push recovery metrics and failure classifier."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from statistics import mean, stdev
from typing import Any, Mapping, Sequence

from true_rl_agent.evaluation.public_result import PublicEvalResult


DEFAULT_MOTION_LIMITS = {
    "max_mean_action_l2": 1000.0,
    "max_mean_action_rate_l2": 1000.0,
    "max_mean_final_xy_drift_m": 5.0,
    "max_mean_max_abs_pitch_rad": 1.2,
    "max_mean_max_abs_roll_rad": 1.2,
}
DEFAULT_ACCEPTANCE = {
    "min_delta": 0.05,
    "near_zero_success_threshold": 0.02,
    "absolute_accept_floor": 0.05,
}
FORBIDDEN_SOURCE_TOKENS = ("private", "hidden", "iid_hidden", "trusted_seed", "accept_reject")


def build_push_public_result(
    *,
    success_rate: float,
    failure_breakdown: Mapping[str, float],
    motion_quality: Mapping[str, Any],
) -> PublicEvalResult:
    return PublicEvalResult(
        status="pass",
        public_score=float(success_rate),
        metrics={
            "success_rate": float(success_rate),
            "failure_breakdown": dict(failure_breakdown),
            "motion_quality": dict(motion_quality),
            "acceptance": {
                "score_source": "single_eval.success_rate",
                "score_stats": {
                    "count": 1,
                    "mean": float(success_rate),
                    "std": 0.0,
                    "standard_error": 0.0,
                    "values": [float(success_rate)],
                },
                "required_pass_gates": [],
                **DEFAULT_ACCEPTANCE,
            },
        },
    )


def build_push_public_result_from_summaries(
    *,
    public_dev_summaries: Sequence[Mapping[str, Any]],
    public_accept_summaries: Sequence[Mapping[str, Any]] | None = None,
    budget_summary: Mapping[str, Any] | None = None,
    motion_limits: Mapping[str, float] | None = None,
) -> PublicEvalResult:
    """Project push-recovery rollout summaries into a public-only eval result.

    This function owns push-recovery success aggregation.  It intentionally keeps
    raw seed identifiers out of the public projection while retaining split
    counts and success-rate statistics for shared accept/reject logic.
    """

    errors: list[dict[str, Any]] = []
    limits = {**DEFAULT_MOTION_LIMITS, **dict(motion_limits or {})}
    public_accept_summaries = public_accept_summaries or []
    dev = _project_split(public_dev_summaries, split_name="public_dev", errors=errors)
    accept = _project_split(public_accept_summaries, split_name="public_accept", errors=errors)
    motion_gate = _motion_quality_gate(
        [*public_dev_summaries, *public_accept_summaries],
        limits=limits,
        prior_errors=errors,
    )
    budget_gate = _budget_accounting_gate(budget_summary)
    accept_stats = accept["success_rate_stats"]
    dev_stats = dev["success_rate_stats"]
    public_score = _first_numeric(accept_stats.get("mean"), dev_stats.get("mean"), 0.0)
    acceptance = {
        "score_source": "public_accept.success_rate_stats.mean",
        "score_stats": accept_stats,
        "missing_required_score": accept_stats.get("mean") is None,
        "required_pass_gates": ["budget_accounting", "motion_quality_gate"],
        **DEFAULT_ACCEPTANCE,
    }
    metrics: dict[str, Any] = {
        "success_rate": public_score,
        "public_dev": dev,
        "budget_accounting": budget_gate,
        "motion_quality_gate": motion_gate,
        "ood_stress_gate": {
            "status": "not_run",
            "required": False,
            "reason": "push_recovery_public_eval_only",
        },
        "acceptance": acceptance,
        "errors": errors,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if public_accept_summaries:
        metrics["public_accept"] = accept
    return PublicEvalResult(
        status="fail" if errors else "pass",
        public_score=float(public_score),
        metrics=metrics,
    )


def attach_push_budget_gate(
    public_result: PublicEvalResult,
    *,
    budget_summary: Mapping[str, Any] | None,
) -> PublicEvalResult:
    metrics = dict(public_result.metrics)
    metrics["budget_accounting"] = _budget_accounting_gate(budget_summary)
    acceptance = dict(metrics.get("acceptance", {})) if isinstance(metrics.get("acceptance"), Mapping) else {}
    gates = list(acceptance.get("required_pass_gates", [])) if isinstance(acceptance.get("required_pass_gates"), list) else []
    if "budget_accounting" not in gates:
        gates.append("budget_accounting")
    acceptance["required_pass_gates"] = gates
    metrics["acceptance"] = acceptance
    return PublicEvalResult(
        status=public_result.status,
        public_score=public_result.public_score,
        metrics=metrics,
    )


def classify_push_failure(metrics: Mapping[str, Any]) -> str:
    breakdown = extract_push_failure_breakdown(metrics)
    if not breakdown:
        return "unknown"
    dominant = max(breakdown, key=breakdown.get)
    lowered = dominant.lower()
    if lowered == "fall" or lowered.startswith("fall_") or lowered.endswith("_fall"):
        return "fall_dominated"
    if lowered in {"push_drift", "excess_lateral_velocity"} or "drift" in lowered:
        return "push_recovery_drift"
    if lowered == "timeout" or "timeout" in lowered:
        return "timeout_dominated"
    return f"{dominant}_dominated"


def extract_push_failure_breakdown(metrics: Mapping[str, Any]) -> dict[str, float]:
    direct = _failure_breakdown_from_payload(metrics.get("failure_breakdown"))
    if direct:
        return direct
    for split_name in ("public_accept", "public_dev"):
        split_breakdown = _failure_breakdown_from_split(metrics.get(split_name))
        if split_breakdown:
            return split_breakdown
    return {}


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
            errors.append({"split": split_name, "run_index": index, "reason": "summary_status_not_pass"})
        if success_rate is None or not 0.0 <= success_rate <= 1.0:
            errors.append({"split": split_name, "run_index": index, "reason": "success_rate_missing_nonfinite_or_out_of_range"})
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
            failures.append({"run_index": index, "reason": "summary_status_not_pass"})
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
        "summary": "push_recovery_public_motion_sanity",
        "limits": {key: _round(value) for key, value in limits.items()},
        "failures": failures,
    }


def _budget_accounting_gate(budget_summary: Mapping[str, Any] | None) -> dict[str, Any]:
    if budget_summary is None:
        return {"status": "fail", "reason": "budget_summary_missing"}
    source = _mapping(budget_summary.get("budget_accounting")) or _mapping(budget_summary.get("training_accounting")) or budget_summary
    projected: dict[str, Any] = {
        "status": str(source.get("status", budget_summary.get("status", "unknown"))),
        "budget_steps": _optional_int(source.get("budget_steps")),
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


def _has_forbidden_source_token(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(_has_forbidden_source_token(key) or _has_forbidden_source_token(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return any(_has_forbidden_source_token(item) for item in value)
    if isinstance(value, str):
        lowered = value.lower()
        return any(token in lowered for token in FORBIDDEN_SOURCE_TOKENS)
    return False


def _failure_breakdown_from_split(value: Any) -> dict[str, float]:
    split = _mapping(value)
    seed_results = split.get("seed_results")
    if not isinstance(seed_results, Sequence) or isinstance(seed_results, (str, bytes)):
        return {}
    weighted_breakdowns: list[tuple[dict[str, float], float]] = []
    for seed_result in seed_results:
        seed = _mapping(seed_result)
        rates, weight = _failure_breakdown_with_weight(
            seed.get("failure_breakdown"),
            seed_episodes=_optional_finite_float(seed.get("episodes")),
        )
        if rates or weight > 0.0:
            weighted_breakdowns.append((rates, weight))
    return _weighted_mean_failure_breakdowns(weighted_breakdowns)


def _failure_breakdown_from_payload(value: Any) -> dict[str, float]:
    payload = _mapping(value)
    if not payload:
        return {}
    rates = _numeric_failure_map(_mapping(payload.get("rates")))
    if rates:
        return rates
    counts = _numeric_failure_map(_mapping(payload.get("counts")))
    if counts:
        total = _optional_finite_float(payload.get("total")) or sum(counts.values())
        if total and total > 0.0:
            return {key: _round(value / total) or 0.0 for key, value in counts.items()}
        return counts
    return _numeric_failure_map(payload)


def _failure_breakdown_with_weight(
    value: Any,
    *,
    seed_episodes: float | None = None,
) -> tuple[dict[str, float], float]:
    payload = _mapping(value)
    if not payload:
        return {}, 0.0
    weight = _failure_breakdown_weight(payload, seed_episodes=seed_episodes)
    rates_payload = _mapping(payload.get("rates"))
    rates = _numeric_failure_map(rates_payload)
    if rates or rates_payload:
        return rates, weight
    counts_payload = _mapping(payload.get("counts"))
    counts = _numeric_failure_map(counts_payload)
    if counts or counts_payload:
        total = _failure_breakdown_total(payload, seed_episodes=seed_episodes)
        if total > 0.0:
            return {key: _round(count / total) or 0.0 for key, count in counts.items()}, total
        return counts, weight
    flat = _numeric_failure_map(payload)
    if flat:
        return flat, weight
    return {}, weight


def _failure_breakdown_weight(payload: Mapping[str, Any], *, seed_episodes: float | None) -> float:
    total = _failure_breakdown_total(payload, seed_episodes=seed_episodes)
    return total if total > 0.0 else 1.0


def _failure_breakdown_total(payload: Mapping[str, Any], *, seed_episodes: float | None) -> float:
    total = _optional_finite_float(payload.get("total"))
    if total is not None and total > 0.0:
        return total
    if seed_episodes is not None and seed_episodes > 0.0:
        return seed_episodes
    counts_payload = _mapping(payload.get("counts"))
    count_total = _sum_numeric_values(counts_payload)
    return count_total if count_total > 0.0 else 0.0


def _sum_numeric_values(value: Mapping[str, Any]) -> float:
    total = 0.0
    for raw in value.values():
        parsed = _optional_finite_float(raw)
        if parsed is not None:
            total += parsed
    return total


def _numeric_failure_map(value: Mapping[str, Any]) -> dict[str, float]:
    numeric: dict[str, float] = {}
    for key, raw in value.items():
        name = str(key)
        if name.lower() in {"success", "recovered"}:
            continue
        parsed = _optional_finite_float(raw)
        if parsed is not None:
            numeric[name] = float(parsed)
    return numeric


def _mean_failure_breakdowns(items: Sequence[Mapping[str, float]]) -> dict[str, float]:
    return _weighted_mean_failure_breakdowns([(dict(item), 1.0) for item in items])


def _weighted_mean_failure_breakdowns(items: Sequence[tuple[Mapping[str, float], float]]) -> dict[str, float]:
    labels = sorted({str(key) for item, _weight in items for key in item})
    total_weight = sum(weight for _item, weight in items if weight > 0.0)
    if not labels or total_weight <= 0.0:
        return {}
    return {
        label: _round(
            sum(float(item.get(label, 0.0)) * weight for item, weight in items if weight > 0.0)
            / total_weight
        )
        or 0.0
        for label in labels
    }


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


def _first_numeric(*values: Any) -> float:
    for value in values:
        if isinstance(value, int | float) and math.isfinite(float(value)):
            return float(value)
    return 0.0
