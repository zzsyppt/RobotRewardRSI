"""Hand-coded failure-to-intervention rule baseline for Phase 4."""

from __future__ import annotations

import copy
import difflib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def build_rule_candidate(
    *,
    summary: dict[str, Any],
    phase1_config: dict[str, Any],
    phase2_config: dict[str, Any],
    rule_config: dict[str, Any],
) -> dict[str, Any]:
    """Build a deterministic candidate config pair from evaluation summary."""

    candidate_phase1 = copy.deepcopy(phase1_config)
    candidate_phase2 = copy.deepcopy(phase2_config)

    diagnosis = _diagnose(summary, rule_config)
    interventions = _select_interventions(diagnosis, rule_config)
    applied = _apply_interventions(candidate_phase1, candidate_phase2, interventions, rule_config, diagnosis)

    analysis = {
        "schema_version": "humanoid_agent.phase4.rule_baseline.analysis.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_name": _candidate_name(rule_config),
        "source_summary": _source_summary_info(summary),
        "hypothesis": _hypothesis(diagnosis),
        "evidence": diagnosis,
        "selected_interventions": interventions,
        "applied_changes": applied,
        "expected_effect": _expected_effect(interventions),
        "risk": _risk(interventions),
        "rollback_condition": (
            "Reject if public mean improvement does not exceed max(5pp, pooled SE), "
            "or if IID-hidden degrades, or if action/energy diagnostics increase materially."
        ),
    }
    return {
        "analysis": analysis,
        "phase1_config": candidate_phase1,
        "phase2_config": candidate_phase2,
    }


def render_unified_diff(base_text: str, candidate_text: str, *, fromfile: str, tofile: str) -> str:
    return "".join(
        difflib.unified_diff(
            base_text.splitlines(keepends=True),
            candidate_text.splitlines(keepends=True),
            fromfile=fromfile,
            tofile=tofile,
        )
    )


def dump_yaml_like(data: Any) -> str:
    """Serialize project config data without requiring PyYAML locally."""

    return _dump_yaml_value(data, 0).rstrip() + "\n"


def stable_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _diagnose(summary: dict[str, Any], rule_config: dict[str, Any]) -> dict[str, Any]:
    public_stats = summary.get("public_eval", {}).get("success_rate_stats", {})
    public_success = _as_float(public_stats.get("mean"), 0.0)
    primary = _aggregate_primary_failures(summary)
    condition_rates = summary.get("public_eval", {}).get("condition_flag_breakdown", {}).get("rates", {})
    secondary = _extract_secondary(summary)
    thresholds = rule_config.get("thresholds", {})

    dominant_primary = _dominant(primary["rates"])
    dominant_condition = _dominant({k: float(v) for k, v in condition_rates.items() if k != "terminated"})

    return {
        "public_success_mean": public_success,
        "primary_failure_counts": primary["counts"],
        "primary_failure_rates": primary["rates"],
        "dominant_primary_failure": dominant_primary,
        "condition_flag_rates": {k: _round(v) for k, v in condition_rates.items()},
        "dominant_condition_flag": dominant_condition,
        "secondary_metrics": secondary,
        "thresholds": thresholds,
        "flags": {
            "low_success": public_success < float(thresholds.get("low_success_rate", 0.45)),
            "high_action_rate": secondary.get("mean_action_rate_l2") is not None
            and float(secondary["mean_action_rate_l2"]) > float(thresholds.get("high_action_rate_l2", 0.5)),
            "high_action_l2": secondary.get("mean_action_l2") is not None
            and float(secondary["mean_action_l2"]) > float(thresholds.get("high_action_l2", 2.5)),
            "high_energy": secondary.get("mean_energy_proxy") is not None
            and float(secondary["mean_energy_proxy"]) > float(thresholds.get("high_energy_proxy", 0.002)),
            "high_oscillation": secondary.get("oscillation_rate") is not None
            and float(secondary["oscillation_rate"]) > float(thresholds.get("high_oscillation_rate", 0.20)),
        },
    }


def _aggregate_primary_failures(summary: dict[str, Any]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    for seed_result in summary.get("seed_results", []):
        breakdown = seed_result.get("public_failure_breakdown", {})
        counts.update({str(k): int(v) for k, v in breakdown.get("counts", {}).items()})
    if not counts and "aggregate" in summary:
        breakdown = summary.get("aggregate", {}).get("failure_breakdown", {})
        counts.update({str(k): int(v) for k, v in breakdown.get("counts", {}).items()})
    total = sum(counts.values())
    rates = {label: (count / total if total else 0.0) for label, count in sorted(counts.items())}
    return {"counts": dict(sorted(counts.items())), "rates": rates, "total": total}


def _extract_secondary(summary: dict[str, Any]) -> dict[str, Any]:
    secondary = dict(summary.get("secondary_metrics", {}))
    public_secondary = summary.get("public_eval", {}).get("secondary_metrics", {})
    if isinstance(public_secondary, dict):
        secondary = {**public_secondary, **secondary}
    # Phase 2 aggregate summaries currently do not include secondary metrics.
    # Keep optional fields for future Phase 5/6 summaries.
    return {
        "mean_action_l2": _optional_float(secondary.get("mean_action_l2")),
        "mean_action_rate_l2": _optional_float(secondary.get("mean_action_rate_l2")),
        "mean_energy_proxy": _first_optional_float(secondary, "mean_energy_proxy", "mean_energy", "mean_power"),
        "oscillation_rate": _first_optional_float(
            secondary,
            "oscillation_rate",
            "mean_oscillation",
            "mean_oscillation_rate",
        ),
        "mean_return": _optional_float(secondary.get("mean_return")),
    }


def _select_interventions(diagnosis: dict[str, Any], rule_config: dict[str, Any]) -> list[dict[str, Any]]:
    thresholds = rule_config.get("thresholds", {})
    high_primary = float(thresholds.get("high_primary_failure_rate", 0.25))
    high_condition = float(thresholds.get("high_condition_flag_rate", 0.20))
    primary_rates = diagnosis["primary_failure_rates"]
    condition_rates = diagnosis["condition_flag_rates"]
    flags = diagnosis["flags"]

    interventions: list[dict[str, Any]] = []
    for label in ("fall_forward", "fall_backward", "fall_side"):
        if float(primary_rates.get(label, 0.0)) >= high_primary:
            interventions.append(
                {
                    "type": "primary_failure",
                    "label": label,
                    "rate": _round(primary_rates[label]),
                    "reason": f"{label} primary failure rate exceeds {high_primary}",
                }
            )

    if float(condition_rates.get("height_below_threshold", 0.0)) >= high_condition:
        interventions.append(
            {
                "type": "condition_flag",
                "label": "height_collapse",
                "rate": _round(condition_rates["height_below_threshold"]),
                "reason": "height_below_threshold is high even when masked by pitch/roll labels",
            }
        )

    if float(condition_rates.get("drift_over_threshold", 0.0)) >= high_condition:
        interventions.append(
            {
                "type": "condition_flag",
                "label": "excessive_drift",
                "rate": _round(condition_rates["drift_over_threshold"]),
                "reason": "drift_over_threshold is high in public eval",
            }
        )

    if flags.get("high_action_rate"):
        interventions.append({"type": "secondary_metric", "label": "high_action_rate", "reason": "mean action rate is high"})
    if flags.get("high_energy"):
        interventions.append({"type": "secondary_metric", "label": "high_energy", "reason": "energy proxy is high"})
    if flags.get("high_oscillation"):
        interventions.append({"type": "secondary_metric", "label": "oscillation", "reason": "oscillation diagnostic is high"})

    if not interventions:
        interventions.append({"type": "fallback", "label": "fallback", "reason": "no single dominant rule fired"})
    return interventions


def _apply_interventions(
    phase1_config: dict[str, Any],
    phase2_config: dict[str, Any],
    interventions: list[dict[str, Any]],
    rule_config: dict[str, Any],
    diagnosis: dict[str, Any],
) -> list[dict[str, Any]]:
    applied: list[dict[str, Any]] = []
    labels = _ordered_unique([item["label"] for item in interventions])
    label_set = set(labels)

    push_profile = _select_training_push_profile(label_set, rule_config)
    phase1_config.setdefault("profiles", {}).setdefault("phase2_train", {}).setdefault("push", {})
    phase1_config["profiles"]["phase2_train"]["push"] = copy.deepcopy(push_profile)
    applied.append({"path": "configs/phase1_push_recovery.yaml", "key": "profiles.phase2_train.push", "value": push_profile})

    adjustments = rule_config.get("reward_adjustments", {})
    for label in labels:
        for reward_name, delta in adjustments.get(label, {}).items():
            old_value = float(phase2_config["reward_weights"][reward_name])
            new_value = _bounded_weight(reward_name, old_value + float(delta), rule_config)
            phase2_config["reward_weights"][reward_name] = new_value
            applied.append(
                {
                    "path": "configs/phase2_baseline.yaml",
                    "key": f"reward_weights.{reward_name}",
                    "old": _round(old_value),
                    "new": _round(new_value),
                    "source_rule": label,
                }
            )

    if diagnosis.get("flags", {}).get("low_success"):
        entropy = rule_config.get("ppo_adjustments", {}).get("entropy_coef_when_low_success")
        if entropy is not None:
            phase2_config.setdefault("ppo", {})["entropy_coef"] = float(entropy)
            applied.append(
                {
                    "path": "configs/phase2_baseline.yaml",
                    "key": "ppo.entropy_coef",
                    "new": float(entropy),
                    "source_rule": "low_success",
                }
            )

    phase2_config.setdefault("baseline", {})["name"] = _candidate_name(rule_config)
    phase2_config["baseline"]["description"] = (
        "Hand-coded Phase 4 rule-baseline candidate generated from Phase 2 public failure summary. "
        "It changes training-only push curriculum and existing reward/PPO scalar knobs; evaluation remains fixed."
    )
    return applied


def _select_training_push_profile(labels: set[str], rule_config: dict[str, Any]) -> dict[str, Any]:
    curriculum = rule_config["training_push_curriculum"]
    if "fall_side" in labels:
        velocity_range = curriculum["side_dominant_velocity_range_mps"]
    elif "excessive_drift" in labels:
        velocity_range = curriculum["drift_dominant_velocity_range_mps"]
    elif {"fall_forward", "fall_backward", "height_collapse"} & labels:
        velocity_range = curriculum["pitch_dominant_velocity_range_mps"]
    else:
        velocity_range = curriculum["fallback_velocity_range_mps"]
    return {
        "interval_range_s": list(curriculum["default_interval_range_s"]),
        "velocity_range_mps": copy.deepcopy(velocity_range),
    }


def _bounded_weight(name: str, value: float, rule_config: dict[str, Any]) -> float:
    bounds = rule_config.get("reward_weight_bounds", {}).get(name)
    if bounds is None:
        return value
    low, high = float(bounds[0]), float(bounds[1])
    return min(high, max(low, value))


def _dominant(rates: dict[str, float]) -> str | None:
    candidates = {k: float(v) for k, v in rates.items() if k != "success"}
    if not candidates:
        return None
    return max(candidates.items(), key=lambda item: item[1])[0]


def _candidate_name(rule_config: dict[str, Any]) -> str:
    prefix = rule_config.get("candidate", {}).get("name_prefix", "rule_baseline")
    return f"{prefix}_v1"


def _source_summary_info(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": summary.get("schema_version"),
        "run_dir": summary.get("run_dir"),
        "status": summary.get("status"),
        "public_success_mean": summary.get("public_eval", {}).get("success_rate_stats", {}).get("mean"),
        "public_success_values": summary.get("public_eval", {}).get("success_rate_stats", {}).get("values"),
    }


def _hypothesis(diagnosis: dict[str, Any]) -> str:
    dominant = diagnosis.get("dominant_primary_failure")
    condition = diagnosis.get("dominant_condition_flag")
    return (
        f"Public eval is dominated by {dominant or 'non-success'} labels with "
        f"{condition or 'no single'} co-occurring condition flag. The rule baseline "
        "will add training-only push exposure and increase existing stability/recovery penalties."
    )


def _expected_effect(interventions: list[dict[str, Any]]) -> str:
    labels = ", ".join(item["label"] for item in interventions)
    return f"Reduce {labels} under fixed public eval without changing public/IID difficulty or budget."


def _risk(interventions: list[dict[str, Any]]) -> str:
    labels = {item["label"] for item in interventions}
    risks = []
    if {"fall_forward", "fall_backward", "height_collapse"} & labels:
        risks.append("stronger upright/base-height shaping may make actions stiffer")
    if "excessive_drift" in labels:
        risks.append("higher velocity tracking/penalty may reduce recovery stepping freedom")
    if "high_action_rate" in labels:
        risks.append("more smoothing may under-react to pushes")
    if "oscillation" in labels:
        risks.append("oscillation damping may reduce quick recovery steps")
    if not risks:
        risks.append("small perturbation may be too weak to beat seed noise")
    return "; ".join(risks)


def _dump_yaml_value(value: Any, indent: int) -> str:
    prefix = " " * indent
    if isinstance(value, dict):
        lines = []
        for key, item in value.items():
            if isinstance(item, list) and all(not isinstance(child, (dict, list)) for child in item):
                lines.append(f"{prefix}{key}: [{', '.join(_format_scalar(child) for child in item)}]")
                continue
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.append(_dump_yaml_value(item, indent + 2).rstrip())
            else:
                lines.append(f"{prefix}{key}: {_format_scalar(item)}")
        return "\n".join(lines) + "\n"
    if isinstance(value, list):
        if all(not isinstance(item, (dict, list)) for item in value):
            return prefix + "[" + ", ".join(_format_scalar(item) for item in value) + "]\n"
        lines = []
        for item in value:
            if isinstance(item, dict):
                lines.append(f"{prefix}-")
                lines.append(_dump_yaml_value(item, indent + 2).rstrip())
            else:
                lines.append(f"{prefix}- {_format_scalar(item)}")
        return "\n".join(lines) + "\n"
    return prefix + _format_scalar(value) + "\n"


def _format_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            raise ValueError("Cannot serialize non-finite YAML scalar")
        if isinstance(value, float):
            return _format_float_scalar(value)
        return repr(value)
    text = str(value)
    if not text or any(char in text for char in ":#[]{}\n") or text.strip() != text:
        return json.dumps(text, ensure_ascii=False)
    return text


def _format_float_scalar(value: float) -> str:
    text = repr(value)
    if "e" not in text.lower():
        return text
    mantissa, exponent = text.lower().split("e", 1)
    if "." not in mantissa:
        mantissa = f"{mantissa}.0"
    return f"{mantissa}e{exponent}"


def _as_float(value: Any, default: float) -> float:
    parsed = _optional_float(value)
    return default if parsed is None else parsed


def _optional_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    except (TypeError, ValueError):
        return None


def _first_optional_float(source: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        parsed = _optional_float(source.get(key))
        if parsed is not None:
            return parsed
    return None


def _ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        ordered.append(value)
        seen.add(value)
    return ordered


def _round(value: Any, digits: int = 6) -> float | Any:
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return value
