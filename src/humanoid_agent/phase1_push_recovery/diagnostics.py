"""LLM-facing diagnostic hints for Phase 1 summaries."""

from __future__ import annotations

from typing import Any


def build_diagnostic_hints(summary: dict[str, Any]) -> list[dict[str, str]]:
    """Generate compact, evidence-linked hints for future agent decisions."""

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
                "candidate_patch_scope": "Increase post-push pitch/upright reward weight or frontal push curriculum in training only.",
            }
        )
    elif dominant == "fall_backward":
        hints.append(
            {
                "condition": "dominant_failure == fall_backward",
                "interpretation": "Backward pitch excursions dominate the protected classifier.",
                "candidate_patch_scope": "Balance pitch recovery reward and inspect backward push sampling.",
            }
        )
    elif dominant == "fall_side":
        hints.append(
            {
                "condition": "dominant_failure == fall_side",
                "interpretation": "Roll instability dominates failures.",
                "candidate_patch_scope": "Increase roll stability reward or lateral push curriculum in training only.",
            }
        )
    elif dominant == "height_collapse":
        hints.append(
            {
                "condition": "dominant_failure == height_collapse",
                "interpretation": "The base height falls below the protected threshold before posture limits dominate.",
                "candidate_patch_scope": "Increase height/upright recovery reward and inspect action saturation.",
            }
        )
    elif dominant == "excessive_drift":
        hints.append(
            {
                "condition": "dominant_failure == excessive_drift",
                "interpretation": "The robot remains upright but fails the protected final-drift gate.",
                "candidate_patch_scope": "Increase drift penalty or recovery-to-origin reward; keep eval drift threshold unchanged.",
            }
        )

    height_rate = float(condition_rates.get("height_below_threshold", 0.0))
    drift_rate = float(condition_rates.get("drift_over_threshold", 0.0))
    if height_rate > 0.0 and dominant != "height_collapse":
        hints.append(
            {
                "condition": "condition_flag_breakdown.rates.height_below_threshold > 0 and dominant_failure != height_collapse",
                "interpretation": "Base-height collapse co-occurs with higher-priority pitch/roll labels and may be masked by the primary classifier.",
                "candidate_patch_scope": "Use height flag rate as diagnostic evidence; do not rely only on dominant_failure.",
            }
        )
    if drift_rate > 0.0 and dominant != "excessive_drift":
        hints.append(
            {
                "condition": "condition_flag_breakdown.rates.drift_over_threshold > 0 and dominant_failure != excessive_drift",
                "interpretation": "Drift violations co-occur with higher-priority failure labels and may be masked by the primary classifier.",
                "candidate_patch_scope": "Use drift flag rate as diagnostic evidence; keep protected eval drift threshold unchanged.",
            }
        )

    mean_action_rate = secondary.get("mean_action_rate_l2")
    mean_action_l2 = secondary.get("mean_action_l2")
    if mean_action_rate is not None and float(mean_action_rate) > 0.5:
        hints.append(
            {
                "condition": "mean_action_rate_l2 > 0.5",
                "interpretation": "Actions are changing quickly; instability may be over-reaction rather than under-reaction.",
                "candidate_patch_scope": "Consider action-rate penalty or lower action scale in training config.",
            }
        )
    if mean_action_l2 is not None and float(mean_action_l2) < 0.05 and success_rate < 0.2:
        hints.append(
            {
                "condition": "mean_action_l2 < 0.05 and success_rate < 0.2",
                "interpretation": "The policy is barely acting while failures are frequent.",
                "candidate_patch_scope": "Bias reward/curriculum toward active post-push recovery instead of only smooth standing.",
            }
        )

    push_effect = summary.get("evidence", {}).get("push_effect", {})
    if push_effect and push_effect.get("nonzero_push_configured", True) and not push_effect.get("gate_pass", False):
        hints.append(
            {
                "condition": "push_effect.gate_pass == false",
                "interpretation": "The observed base-speed change did not clearly prove that the push took effect.",
                "candidate_patch_scope": "Debug event timing/range before trusting policy-performance labels.",
            }
        )

    if not hints:
        hints.append(
            {
                "condition": "no dominant diagnostic issue",
                "interpretation": "No single protected metric dominates; compare multiple seeds before changing config.",
                "candidate_patch_scope": "Keep eval fixed and inspect per-episode evidence.",
            }
        )
    return hints
