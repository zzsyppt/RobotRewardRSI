"""Public diagnostic hints derived from push recovery eval summaries."""

from __future__ import annotations

from typing import Any


def build_diagnostic_hints(summary: dict[str, Any]) -> list[dict[str, str]]:
    aggregate = summary.get("aggregate", {})
    failures = aggregate.get("failure_breakdown", {})
    condition_flags = aggregate.get("condition_flag_breakdown", {})
    condition_rates = condition_flags.get("rates", {})
    secondary = summary.get("secondary_metrics", {})
    hints: list[dict[str, str]] = []

    dominant = failures.get("dominant_failure")
    success_rate = float(aggregate.get("success_rate", 0.0))
    if dominant == "fall_forward":
        hints.append(
            {
                "condition": "dominant_failure == fall_forward",
                "interpretation": "Forward pitch excursions dominate the protected classifier.",
                "candidate_patch_scope": "Improve post-push pitch/upright reward; keep eval thresholds fixed.",
            }
        )
    elif dominant == "fall_backward":
        hints.append(
            {
                "condition": "dominant_failure == fall_backward",
                "interpretation": "Backward pitch excursions dominate the protected classifier.",
                "candidate_patch_scope": "Improve backward push recovery reward without changing eval protocol.",
            }
        )
    elif dominant == "fall_side":
        hints.append(
            {
                "condition": "dominant_failure == fall_side",
                "interpretation": "Roll instability dominates failures.",
                "candidate_patch_scope": "Improve lateral stability or recovery logic in training code.",
            }
        )
    elif dominant == "height_collapse":
        hints.append(
            {
                "condition": "dominant_failure == height_collapse",
                "interpretation": "Base height falls below the protected threshold.",
                "candidate_patch_scope": "Improve height/upright recovery reward and inspect action saturation.",
            }
        )
    elif dominant == "excessive_drift":
        hints.append(
            {
                "condition": "dominant_failure == excessive_drift",
                "interpretation": "The robot stays upright but violates the final drift gate.",
                "candidate_patch_scope": "Improve recovery-to-origin behavior; do not loosen eval drift threshold.",
            }
        )

    if float(condition_rates.get("height_below_threshold", 0.0)) > 0.0 and dominant != "height_collapse":
        hints.append(
            {
                "condition": "height_below_threshold co-occurs with other labels",
                "interpretation": "Height collapse may be masked by higher-priority pitch/roll labels.",
                "candidate_patch_scope": "Use condition flags, not only dominant failure, for code-level reward design.",
            }
        )
    if float(condition_rates.get("drift_over_threshold", 0.0)) > 0.0 and dominant != "excessive_drift":
        hints.append(
            {
                "condition": "drift_over_threshold co-occurs with other labels",
                "interpretation": "Drift violations may be masked by fall labels.",
                "candidate_patch_scope": "Inspect drift evidence while keeping protected success rules fixed.",
            }
        )

    mean_action_rate = secondary.get("mean_action_rate_l2")
    mean_action_l2 = secondary.get("mean_action_l2")
    if mean_action_rate is not None and float(mean_action_rate) > 0.5:
        hints.append(
            {
                "condition": "mean_action_rate_l2 > 0.5",
                "interpretation": "Actions are changing quickly; instability may be over-reaction.",
                "candidate_patch_scope": "Consider smooth recovery rewards in candidate code.",
            }
        )
    if mean_action_l2 is not None and float(mean_action_l2) < 0.05 and success_rate < 0.2:
        hints.append(
            {
                "condition": "mean_action_l2 < 0.05 and success_rate < 0.2",
                "interpretation": "The policy is barely acting while failures are frequent.",
                "candidate_patch_scope": "Bias candidate reward toward active post-push recovery.",
            }
        )

    push_effect = summary.get("evidence", {}).get("push_effect", {})
    if push_effect and push_effect.get("nonzero_push_configured", True) and not push_effect.get("gate_pass", False):
        hints.append(
            {
                "condition": "push_effect.gate_pass == false",
                "interpretation": "Observed base-speed change did not prove that push took effect.",
                "candidate_patch_scope": "Treat policy labels as suspect until runner-side push evidence is valid.",
            }
        )

    if not hints:
        hints.append(
            {
                "condition": "no dominant diagnostic issue",
                "interpretation": "No single protected metric dominates.",
                "candidate_patch_scope": "Compare public evidence across rounds before changing candidate code.",
            }
        )
    return hints
