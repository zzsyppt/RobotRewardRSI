"""Public push recovery metrics and failure classifier."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from statistics import mean, stdev
from typing import Any, Mapping, Sequence

from true_rl_agent.evaluation.acceptance_protocol import AcceptanceProtocol
from true_rl_agent.evaluation.protocol import EvaluationProtocol
from true_rl_agent.evaluation.public_result import PublicEvalResult
from true_rl_agent.evaluation.result_contract import validate_finite_number
from true_rl_agent.evaluation.sealed_validation import build_sealed_rollout_evidence
from true_rl_agent.evaluation.subject import ExperimentSubject
from true_rl_agent.tasks.push_recovery.config import PushRecoveryTaskConfig

DEFAULT_MOTION_LIMITS = {
    "max_mean_action_l2": 1000.0,
    "max_mean_action_rate_l2": 1000.0,
    "min_mean_min_base_height_m": 0.30,
    "max_mean_final_xy_drift_m": 5.0,
    "max_mean_max_abs_pitch_rad": 1.2,
    "max_mean_max_abs_roll_rad": 1.2,
    "max_mean_final_planar_speed_mps": 5.0,
    "min_recovery_window_success_rate": 0.0,
}
DEFAULT_MIN_PUBLIC_ACCEPT_COUNT = 3
FORBIDDEN_SOURCE_TOKENS = ("private", "hidden", "iid_hidden", "trusted_seed", "accept_reject")
SUCCESS_FAILURE_LABELS = {"success", "recovered"}
FAILURE_BREAKDOWN_METADATA_KEYS = {
    "diagnostic_only",
    "dominant_failure",
    "nonexclusive",
    "primary_label_priority",
    "total",
}


def build_push_public_result(
    *,
    success_rate: float,
    failure_breakdown: Mapping[str, float],
    motion_quality: Mapping[str, Any],
    acceptance_protocol: AcceptanceProtocol | None = None,
) -> PublicEvalResult:
    protocol = acceptance_protocol or PushRecoveryTaskConfig().acceptance_protocol()
    canonical_success_rate = validate_finite_number(
        success_rate,
        field_name="push public success_rate",
    )
    return PublicEvalResult(
        status="pass",
        public_score=canonical_success_rate,
        metrics={
            "success_rate": canonical_success_rate,
            "failure_breakdown": dict(failure_breakdown),
            "motion_quality": dict(motion_quality),
            "prompt_feedback": _prompt_feedback_from_breakdown(
                _failure_breakdown_from_payload(failure_breakdown),
                source_split="single_eval",
            ),
            "behavior_quality_gate": _missing_behavior_quality_gate("legacy_single_eval_behavior_metrics_missing"),
            "acceptance": {
                "score_source": "single_eval.success_rate",
                "public_score_binding": protocol.public_score_binding,
                "score_stats": {
                    "count": 1,
                    "mean": canonical_success_rate,
                    "std": 0.0,
                    "standard_error": 0.0,
                    "values": [canonical_success_rate],
                },
                "required_pass_gates": ["behavior_quality_gate"],
                "acceptance_protocol_hash": protocol.protocol_hash,
                "absolute_accept_floor": protocol.absolute_accept_floor,
                "min_delta": protocol.min_delta,
            },
        },
    )


def build_push_public_result_from_summaries(
    *,
    public_dev_summaries: Sequence[Mapping[str, Any]],
    public_accept_summaries: Sequence[Mapping[str, Any]] | None = None,
    budget_summary: Mapping[str, Any] | None = None,
    motion_limits: Mapping[str, float] | None = None,
    min_public_accept_count: int = DEFAULT_MIN_PUBLIC_ACCEPT_COUNT,
    acceptance_protocol: AcceptanceProtocol | None = None,
) -> PublicEvalResult:
    """Project push-recovery rollout summaries into a public-only eval result.

    This function owns push-recovery success aggregation.  It intentionally keeps
    raw seed identifiers out of the public projection while retaining split
    counts and success-rate statistics for shared accept/reject logic.
    """

    min_public_accept_count = _require_positive_integer(
        min_public_accept_count,
        "min_public_accept_count",
    )
    protocol = acceptance_protocol or PushRecoveryTaskConfig(
        public_accept_seed_count=min_public_accept_count
    ).acceptance_protocol()
    if min_public_accept_count != protocol.min_score_count:
        raise ValueError(
            "public accept count does not match task acceptance protocol: "
            f"count={min_public_accept_count} protocol={protocol.min_score_count}"
        )
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
    behavior_gate = _behavior_quality_gate(
        public_dev_summaries=public_dev_summaries,
        public_accept_summaries=public_accept_summaries,
        limits=limits,
        prior_errors=errors,
    )
    budget_gate = _budget_accounting_gate(budget_summary)
    accept_stats = accept["score_stats"]
    dev_stats = dev["score_stats"]
    accept_evidence_gate = _accept_evidence_gate(
        accept_stats,
        min_public_accept_count=min_public_accept_count,
    )
    public_score = _first_numeric(accept_stats.get("mean"), dev_stats.get("mean"), 0.0)
    acceptance = {
        "score_source": protocol.score_source,
        "public_score_binding": protocol.public_score_binding,
        "score_stats": accept_stats,
        "min_score_count": min_public_accept_count,
        "missing_required_score": accept_stats.get("mean") is None,
        "required_pass_gates": [
            "accept_evidence_gate",
            "budget_accounting",
            "motion_quality_gate",
            "behavior_quality_gate",
        ],
        "acceptance_protocol_hash": protocol.protocol_hash,
        "absolute_accept_floor": protocol.absolute_accept_floor,
        "min_delta": protocol.min_delta,
    }
    metrics: dict[str, Any] = {
        "success_rate": public_score,
        "public_dev": dev,
        "motion_quality": _aggregate_public_dev_motion_quality(public_dev_summaries),
        "prompt_feedback": _prompt_feedback_from_splits(public_dev=dev),
        "budget_accounting": budget_gate,
        "accept_evidence_gate": accept_evidence_gate,
        "motion_quality_gate": motion_gate,
        "behavior_quality_gate": behavior_gate,
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


def build_push_sealed_rollout_evidence(
    *,
    summaries: Sequence[Mapping[str, Any]],
    evaluation_protocol: EvaluationProtocol,
    checkpoint_hash: str,
    evaluation_binding_hash: str,
    evaluation_subject: ExperimentSubject,
    subject_manifest_hash: str | None = None,
    program_source_checksum: str | None = None,
    training_binding_hash: str | None = None,
    capability_profile_hash: str,
    acceptance_protocol: AcceptanceProtocol | None = None,
) -> dict[str, Any]:
    """Project runner-owned holdout rollouts without exposing per-run detail."""

    protocol = acceptance_protocol or PushRecoveryTaskConfig(
        public_accept_seed_count=evaluation_protocol.public_accept_seed_count,
        sealed_validation_seed_count=evaluation_protocol.sealed_validation_seed_count,
    ).acceptance_protocol()
    errors: list[dict[str, Any]] = []
    projected = _project_split(
        summaries,
        split_name="sealed_validation",
        errors=errors,
    )
    score_stats = dict(_mapping(projected.get("score_stats")))
    score_stats["eval_protocol_hash"] = evaluation_protocol.protocol_hash
    score_mean = _optional_finite_float(score_stats.get("mean"))
    behavior_passed = bool(
        not errors
        and score_mean is not None
        and score_mean >= protocol.absolute_accept_floor
    )
    return build_sealed_rollout_evidence(
        score_stats=score_stats,
        expected_seed_count=evaluation_protocol.sealed_validation_seed_count,
        eval_protocol_hash=evaluation_protocol.protocol_hash,
        evaluation_binding_hash=evaluation_binding_hash,
        evaluation_subject=evaluation_subject,
        checkpoint_hash=checkpoint_hash,
        task_validation_status="pass" if behavior_passed else "fail",
        reason=(
            "push_recovery_holdout_floor_met"
            if behavior_passed
            else "push_recovery_holdout_floor_not_met"
        ),
        subject_manifest_hash=subject_manifest_hash,
        program_source_checksum=program_source_checksum,
        training_binding_hash=training_binding_hash,
        capability_profile_hash=capability_profile_hash,
    )


def attach_push_budget_gate(
    public_result: PublicEvalResult,
    *,
    budget_summary: Mapping[str, Any] | None,
) -> PublicEvalResult:
    metrics = dict(public_result.metrics)
    metrics["budget_accounting"] = _budget_accounting_gate(budget_summary)
    if not _mapping(metrics.get("behavior_quality_gate")):
        metrics["behavior_quality_gate"] = _missing_behavior_quality_gate(
            "behavior_quality_gate_missing_from_public_result"
        )
    acceptance = dict(metrics.get("acceptance", {})) if isinstance(metrics.get("acceptance"), Mapping) else {}
    gates = list(acceptance.get("required_pass_gates", [])) if isinstance(acceptance.get("required_pass_gates"), list) else []
    if "budget_accounting" not in gates:
        gates.append("budget_accounting")
    if "behavior_quality_gate" not in gates:
        gates.append("behavior_quality_gate")
    if "sealed_validation_gate" in metrics and "sealed_validation_gate" not in gates:
        gates.append("sealed_validation_gate")
    acceptance["required_pass_gates"] = gates
    metrics["acceptance"] = acceptance
    return PublicEvalResult(
        status=public_result.status,
        public_score=public_result.public_score,
        metrics=metrics,
    )


def classify_push_failure(metrics: Mapping[str, Any]) -> str:
    breakdown = extract_push_failure_breakdown(metrics)
    return _classify_push_failure_breakdown(breakdown)


def classify_push_prompt_failure(metrics: Mapping[str, Any]) -> str:
    breakdown = extract_push_prompt_failure_breakdown(metrics)
    return _classify_push_failure_breakdown(breakdown)


def _classify_push_failure_breakdown(breakdown: Mapping[str, float]) -> str:
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
    """Return the broad public failure breakdown for non-prompt analysis."""

    direct = _failure_breakdown_from_payload(metrics.get("failure_breakdown"))
    if direct:
        return direct
    for split_name in ("public_accept", "public_dev"):
        split_breakdown = _failure_breakdown_from_split(metrics.get(split_name))
        if split_breakdown:
            return split_breakdown
    return {}


def extract_push_prompt_failure_breakdown(metrics: Mapping[str, Any]) -> dict[str, float]:
    """Return only prompt-visible diagnostic failure labels.

    Public-accept failure labels are intentionally excluded; that split is
    prompt-visible as aggregate stats only.
    """

    direct = _failure_breakdown_from_payload(metrics.get("failure_breakdown"))
    if direct:
        return direct
    return _failure_breakdown_from_split(metrics.get("public_dev"))


def _prompt_feedback_from_splits(
    *,
    public_dev: Mapping[str, Any],
) -> dict[str, Any]:
    dev_breakdown = _failure_breakdown_from_split(public_dev)
    if dev_breakdown:
        return _prompt_feedback_from_breakdown(dev_breakdown, source_split="public_dev")
    return _prompt_feedback_from_breakdown(
        {},
        source_split="public_dev",
        missing_reason="public_dev_failure_breakdown_missing",
    )


def _prompt_feedback_from_breakdown(
    breakdown: Mapping[str, float],
    *,
    source_split: str,
    missing_reason: str = "failure_breakdown_missing",
) -> dict[str, Any]:
    return {
        "schema_version": "true_rl_agent.public_eval_feedback.push_recovery.v1",
        "dominant_failure": _dominant_failure_prompt_feedback(
            breakdown,
            source_split=source_split,
            missing_reason=missing_reason,
        ),
    }


def _dominant_failure_prompt_feedback(
    breakdown: Mapping[str, float],
    *,
    source_split: str,
    missing_reason: str,
) -> dict[str, Any]:
    numeric: dict[str, float] = {}
    for raw_label, raw_rate in breakdown.items():
        label = str(raw_label)
        rate = _optional_finite_float(raw_rate)
        if rate is None or rate < 0.0:
            continue
        numeric[label] = float(rate)
    if not numeric:
        return {
            "status": "not_available",
            "reason": missing_reason,
        }
    label, rate = max(numeric.items(), key=lambda item: (item[1], item[0]))
    return {
        "status": "available",
        "source_split": str(source_split),
        "label": label,
        "rate": _round(rate),
    }


def _project_split(
    summaries: Sequence[Mapping[str, Any]],
    *,
    split_name: str,
    errors: list[dict[str, Any]],
) -> dict[str, Any]:
    seed_results: list[dict[str, Any]] = []
    success_rates: list[float] = []
    episode_counts: list[int] = []
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
            episodes = _optional_int(aggregate.get("episodes"))
            if episodes is not None and episodes > 0:
                episode_counts.append(episodes)
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
        "score_stats": _stats(success_rates, episode_counts=episode_counts),
        "seed_results": seed_results,
    }


def _stats(values: Sequence[float], *, episode_counts: Sequence[int] = ()) -> dict[str, Any]:
    rounded = [_round(value) for value in values]
    episode_count = sum(int(value) for value in episode_counts) if episode_counts else None
    if not values:
        return {"count": 0, "episode_count": episode_count, "mean": None, "std": None, "standard_error": None, "values": []}
    if len(values) == 1:
        return {"count": 1, "episode_count": episode_count, "mean": rounded[0], "std": 0.0, "standard_error": 0.0, "values": rounded}
    sample_std = stdev(values)
    return {
        "count": len(values),
        "episode_count": episode_count,
        "mean": _round(mean(values)),
        "std": _round(sample_std),
        "standard_error": _round(sample_std / math.sqrt(len(values))),
        "values": rounded,
    }


def _accept_evidence_gate(
    score_stats: Mapping[str, Any],
    *,
    min_public_accept_count: int,
) -> dict[str, Any]:
    required_count = _require_positive_integer(
        min_public_accept_count,
        "min_public_accept_count",
    )
    observed_count = _optional_int(score_stats.get("count")) or 0
    gate: dict[str, Any] = {
        "status": "pass",
        "reason": "sufficient_accept_evidence",
        "observed_count": observed_count,
        "required_count": required_count,
        "episode_count": _optional_int(score_stats.get("episode_count")),
    }
    if observed_count < required_count:
        gate["status"] = "fail"
        gate["reason"] = "insufficient_accept_evidence"
        return gate
    if _optional_finite_float(score_stats.get("mean")) is None:
        gate["status"] = "fail"
        gate["reason"] = "accept_score_stats_untrusted"
        return gate
    if _optional_finite_float(score_stats.get("standard_error")) is None:
        gate["status"] = "fail"
        gate["reason"] = "accept_score_stats_untrusted"
        return gate
    return gate


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


def _behavior_quality_gate(
    *,
    public_dev_summaries: Sequence[Mapping[str, Any]],
    public_accept_summaries: Sequence[Mapping[str, Any]],
    limits: Mapping[str, float],
    prior_errors: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    checked_metrics: list[str] = []
    parsed_failure_breakdown = False
    summaries = (
        ("public_dev", public_dev_summaries),
        ("public_accept", public_accept_summaries),
    )
    if not public_dev_summaries and not public_accept_summaries:
        failures.append({"reason": "no_rollout_summaries"})
    for split_name, split_summaries in summaries:
        for run_index, summary in enumerate(split_summaries, start=1):
            failure_breakdown = _failure_breakdown_payload(summary)
            if _failure_breakdown_is_parseable(failure_breakdown):
                parsed_failure_breakdown = True
            else:
                failures.append(
                    {
                        "split": split_name,
                        "run_index": run_index,
                        "reason": "failure_breakdown_unparseable",
                    }
                )
            secondary = _mapping(summary.get("secondary_metrics"))
            if not secondary:
                failures.append({"split": split_name, "run_index": run_index, "reason": "secondary_metrics_missing"})
                continue
            _check_upper_limit(
                failures,
                checked_metrics,
                secondary,
                split_name,
                run_index,
                "mean_action_l2",
                limits["max_mean_action_l2"],
            )
            _check_upper_limit(
                failures,
                checked_metrics,
                secondary,
                split_name,
                run_index,
                "mean_action_rate_l2",
                limits["max_mean_action_rate_l2"],
            )
            _check_lower_limit(
                failures,
                checked_metrics,
                secondary,
                split_name,
                run_index,
                "mean_min_base_height_m",
                limits["min_mean_min_base_height_m"],
            )
            _check_upper_limit(
                failures,
                checked_metrics,
                secondary,
                split_name,
                run_index,
                "mean_max_abs_pitch_rad",
                limits["max_mean_max_abs_pitch_rad"],
            )
            _check_upper_limit(
                failures,
                checked_metrics,
                secondary,
                split_name,
                run_index,
                "mean_max_abs_roll_rad",
                limits["max_mean_max_abs_roll_rad"],
            )
            _check_upper_limit(
                failures,
                checked_metrics,
                secondary,
                split_name,
                run_index,
                "mean_final_xy_drift_m",
                limits["max_mean_final_xy_drift_m"],
            )
            _check_upper_limit(
                failures,
                checked_metrics,
                secondary,
                split_name,
                run_index,
                "mean_final_planar_speed_mps",
                limits["max_mean_final_planar_speed_mps"],
            )
            _check_rate_metric(
                failures,
                checked_metrics,
                secondary,
                split_name,
                run_index,
                "recovery_window_success_rate",
                min_value=limits["min_recovery_window_success_rate"],
            )
    if prior_errors:
        failures.append({"reason": "public_eval_split_errors_present", "count": len(prior_errors)})
    return {
        "status": "fail" if failures else "pass",
        "summary": "push_recovery_public_behavior_required_gate",
        "failure_breakdown_status": "pass" if parsed_failure_breakdown and not any(item.get("reason") == "failure_breakdown_unparseable" for item in failures) else "fail",
        "checked_metrics": sorted(dict.fromkeys(checked_metrics)),
        "limits": {key: _round(value) for key, value in limits.items()},
        "recovery_signal_status": _recovery_signal_status(checked_metrics=checked_metrics, failures=failures),
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


def _missing_behavior_quality_gate(reason: str) -> dict[str, Any]:
    return {
        "status": "fail",
        "reason": reason,
        "summary": "push_recovery_public_behavior_required_gate",
        "failure_breakdown_status": "fail",
        "checked_metrics": [],
        "recovery_signal_status": "missing",
        "failures": [{"reason": reason}],
    }


def _recovery_signal_status(
    *,
    checked_metrics: Sequence[str],
    failures: Sequence[Mapping[str, Any]],
) -> str:
    required_metrics = {"mean_final_planar_speed_mps", "recovery_window_success_rate"}
    if not required_metrics.issubset(set(checked_metrics)):
        return "missing"
    invalid_reasons = {"missing_or_nonfinite", "negative_value", "out_of_range"}
    for failure in failures:
        if failure.get("metric") in required_metrics and failure.get("reason") in invalid_reasons:
            return "missing"
    return "available"


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
        "mean_final_planar_speed_mps",
        "recovery_window_success_rate",
    )
    return {key: _round(float(metrics[key])) for key in allowed if _optional_finite_float(metrics.get(key)) is not None}


def _aggregate_public_dev_motion_quality(
    summaries: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    """Return bounded prompt-visible means from public-dev rollouts only."""

    projected = [
        metrics
        for summary in summaries
        if summary.get("status") == "pass"
        for metrics in [_project_secondary_metrics(summary.get("secondary_metrics"))]
        if metrics
    ]
    keys = sorted({key for metrics in projected for key in metrics})
    aggregated: dict[str, float] = {}
    for key in keys:
        values = [float(metrics[key]) for metrics in projected if key in metrics]
        if values:
            aggregated[key] = _round(mean(values)) or 0.0
    return aggregated


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
    if value < 0.0:
        failures.append({"run_index": run_index, "metric": key, "reason": "negative_value"})
        return
    if value > float(limit):
        failures.append({"run_index": run_index, "metric": key, "actual": _round(value), "limit": _round(limit), "reason": "limit_exceeded"})


def _check_upper_limit(
    failures: list[dict[str, Any]],
    checked_metrics: list[str],
    secondary: Mapping[str, Any],
    split_name: str,
    run_index: int,
    key: str,
    limit: float,
) -> None:
    value = _optional_finite_float(secondary.get(key))
    if value is None:
        failures.append({"split": split_name, "run_index": run_index, "metric": key, "reason": "missing_or_nonfinite"})
        return
    if value < 0.0:
        failures.append({"split": split_name, "run_index": run_index, "metric": key, "reason": "negative_value"})
        return
    checked_metrics.append(key)
    if value > float(limit):
        failures.append(
            {
                "split": split_name,
                "run_index": run_index,
                "metric": key,
                "actual": _round(value),
                "limit": _round(limit),
                "reason": "limit_exceeded",
            }
        )


def _check_rate_metric(
    failures: list[dict[str, Any]],
    checked_metrics: list[str],
    secondary: Mapping[str, Any],
    split_name: str,
    run_index: int,
    key: str,
    *,
    min_value: float,
) -> None:
    value = _optional_finite_float(secondary.get(key))
    if value is None:
        failures.append({"split": split_name, "run_index": run_index, "metric": key, "reason": "missing_or_nonfinite"})
        return
    if value < 0.0 or value > 1.0:
        failures.append({"split": split_name, "run_index": run_index, "metric": key, "reason": "out_of_range"})
        return
    checked_metrics.append(key)
    if value < float(min_value):
        failures.append(
            {
                "split": split_name,
                "run_index": run_index,
                "metric": key,
                "actual": _round(value),
                "limit": _round(min_value),
                "reason": "limit_exceeded",
            }
        )


def _check_lower_limit(
    failures: list[dict[str, Any]],
    checked_metrics: list[str],
    secondary: Mapping[str, Any],
    split_name: str,
    run_index: int,
    key: str,
    limit: float,
) -> None:
    checked_metrics.append(key)
    value = _optional_finite_float(secondary.get(key))
    if value is None:
        failures.append({"split": split_name, "run_index": run_index, "metric": key, "reason": "missing_or_nonfinite"})
        return
    if value < float(limit):
        failures.append(
            {
                "split": split_name,
                "run_index": run_index,
                "metric": key,
                "actual": _round(value),
                "limit": _round(limit),
                "reason": "limit_exceeded",
            }
        )


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


def _failure_breakdown_payload(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    aggregate = _mapping(summary.get("aggregate"))
    return _mapping(aggregate.get("failure_breakdown"))


def _failure_breakdown_from_payload(value: Any) -> dict[str, float]:
    payload = _mapping(value)
    if not payload:
        return {}
    rates_payload = _mapping(payload.get("rates"))
    if rates_payload or "rates" in payload:
        return _strict_numeric_failure_map(rates_payload, allow_zero_total=True) or {}
    counts_payload = _mapping(payload.get("counts"))
    if counts_payload or "counts" in payload:
        counts = _strict_numeric_failure_map(counts_payload, allow_zero_total=False)
        if counts is None:
            return {}
        total = _optional_finite_float(payload.get("total")) or sum(counts.values())
        if total and total > 0.0:
            return {key: _round(value / total) or 0.0 for key, value in counts.items()}
        return counts
    return _strict_numeric_failure_map(payload, allow_zero_total=True, allow_metadata=True) or {}


def _failure_breakdown_is_parseable(payload: Mapping[str, Any]) -> bool:
    if not payload:
        return False
    rates_payload = _mapping(payload.get("rates"))
    if rates_payload or "rates" in payload:
        return _strict_numeric_failure_map(rates_payload, allow_zero_total=True) is not None
    counts_payload = _mapping(payload.get("counts"))
    if counts_payload or "counts" in payload:
        counts = _strict_numeric_failure_map(counts_payload, allow_zero_total=False)
        total = _optional_finite_float(payload.get("total"))
        return counts is not None and (total is None or total > 0.0)
    return _strict_numeric_failure_map(payload, allow_zero_total=True, allow_metadata=True) is not None


def _strict_numeric_failure_map(
    value: Mapping[str, Any],
    *,
    allow_zero_total: bool,
    allow_metadata: bool = False,
) -> dict[str, float] | None:
    if not value:
        return None
    numeric: dict[str, float] = {}
    total = 0.0
    observed_semantic_label = False
    for key, raw in value.items():
        name = str(key)
        lowered = name.lower()
        if allow_metadata and lowered in FAILURE_BREAKDOWN_METADATA_KEYS:
            continue
        parsed = None if isinstance(raw, bool) else _optional_finite_float(raw)
        if parsed is None or parsed < 0.0:
            return None
        observed_semantic_label = True
        total += parsed
        if lowered not in SUCCESS_FAILURE_LABELS:
            numeric[name] = float(parsed)
    if not observed_semantic_label:
        return None
    if total <= 0.0 and not allow_zero_total:
        return None
    return numeric


def _failure_breakdown_with_weight(
    value: Any,
    *,
    seed_episodes: float | None = None,
) -> tuple[dict[str, float], float]:
    payload = _mapping(value)
    if not payload:
        return {}, 0.0
    if not _failure_breakdown_is_parseable(payload):
        return {}, 0.0
    weight = _failure_breakdown_weight(payload, seed_episodes=seed_episodes)
    return _failure_breakdown_from_payload(payload), weight


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
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _require_positive_integer(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _optional_finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return validate_finite_number(value, field_name="push metric")
    except ValueError:
        return None


def _round(value: float | int | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _first_numeric(*values: Any) -> float:
    for value in values:
        parsed = _optional_finite_float(value)
        if parsed is not None:
            return parsed
    return 0.0
