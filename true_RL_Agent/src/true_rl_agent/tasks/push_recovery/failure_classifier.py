"""Protected push recovery success and failure classification rules."""

from __future__ import annotations

from collections import Counter
from typing import Any

SUCCESS_LABEL = "success"


def classify_episode(metrics: dict[str, Any], thresholds: dict[str, float]) -> dict[str, Any]:
    height_threshold = float(thresholds["min_base_height_m"])
    roll_threshold = float(thresholds["max_abs_roll_rad"])
    pitch_threshold = float(thresholds["max_abs_pitch_rad"])
    drift_threshold = float(thresholds["max_final_xy_drift_m"])

    min_base_height = float(metrics["min_base_height_m"])
    max_pitch = float(metrics["max_pitch_rad"])
    min_pitch = float(metrics["min_pitch_rad"])
    max_abs_roll = float(metrics["max_abs_roll_rad"])
    max_abs_pitch = float(metrics["max_abs_pitch_rad"])
    final_xy_drift = float(metrics["final_xy_drift_m"])
    terminated = bool(metrics.get("terminated", False))

    fallen = (
        terminated
        or min_base_height < height_threshold
        or max_abs_roll > roll_threshold
        or max_abs_pitch > pitch_threshold
    )
    if fallen:
        if max_pitch > pitch_threshold:
            label = "fall_forward"
        elif min_pitch < -pitch_threshold:
            label = "fall_backward"
        elif max_abs_roll > roll_threshold:
            label = "fall_side"
        elif min_base_height < height_threshold:
            label = "height_collapse"
        else:
            label = "fall_unknown"
    elif final_xy_drift > drift_threshold:
        label = "excessive_drift"
    else:
        label = SUCCESS_LABEL

    return {
        "label": label,
        "success": label == SUCCESS_LABEL,
        "ordered_rules": {
            "fallen": fallen,
            "terminated": terminated,
            "height_below_threshold": min_base_height < height_threshold,
            "roll_over_threshold": max_abs_roll > roll_threshold,
            "pitch_over_threshold": max_abs_pitch > pitch_threshold,
            "drift_over_threshold": final_xy_drift > drift_threshold,
        },
    }


def summarize_failures(labels: list[str]) -> dict[str, Any]:
    total = len(labels)
    counts = Counter(labels)
    rates = {label: count / total for label, count in sorted(counts.items())} if total else {}
    return {
        "total": total,
        "counts": dict(sorted(counts.items())),
        "rates": rates,
        "dominant_failure": _dominant_failure(counts),
    }


def summarize_condition_flags(classifications: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(classifications)
    flag_names = [
        "terminated",
        "height_below_threshold",
        "roll_over_threshold",
        "pitch_over_threshold",
        "drift_over_threshold",
    ]
    counts = dict.fromkeys(flag_names, 0)
    for classification in classifications:
        ordered_rules = classification.get("ordered_rules", {})
        for flag_name in flag_names:
            counts[flag_name] += int(bool(ordered_rules.get(flag_name, False)))
    rates = {name: (count / total if total else 0.0) for name, count in counts.items()}
    return {
        "total": total,
        "counts": counts,
        "rates": rates,
        "nonexclusive": True,
        "primary_label_priority": [
            "fall_forward",
            "fall_backward",
            "fall_side",
            "height_collapse",
            "fall_unknown",
            "excessive_drift",
            "success",
        ],
        "diagnostic_only": True,
    }


def _dominant_failure(counts: Counter[str]) -> str | None:
    failure_counts = Counter({key: value for key, value in counts.items() if key != SUCCESS_LABEL})
    if not failure_counts:
        return None
    return failure_counts.most_common(1)[0][0]
