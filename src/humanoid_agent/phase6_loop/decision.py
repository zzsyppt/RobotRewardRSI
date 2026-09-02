"""Phase 6 public-only accept/reject and 3-to-5 seed escalation."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from humanoid_agent.phase6_loop.stats import pooled_standard_error


def build_phase6_public_decision(
    *,
    best_summary: dict[str, Any],
    candidate_summary: dict[str, Any],
    best_summary_path: Path,
    candidate_summary_path: Path,
    operation_diff: dict[str, Any] | None = None,
    seed_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    seed_policy = seed_policy or {}
    public_min_delta = float(seed_policy.get("public_min_delta", 0.05))
    pooled_multiplier = float(seed_policy.get("pooled_se_multiplier", 1.0))
    best_stats = best_summary.get("public_eval", {}).get("success_rate_stats", {})
    cand_stats = candidate_summary.get("public_eval", {}).get("success_rate_stats", {})
    public_delta = _mean(cand_stats) - _mean(best_stats) if _mean(cand_stats) is not None and _mean(best_stats) is not None else None
    pooled_se = pooled_standard_error(cand_stats, best_stats)
    threshold = None if pooled_se is None else max(public_min_delta, pooled_multiplier * pooled_se)
    accepted = bool(
        best_summary.get("status") == "pass"
        and candidate_summary.get("status") == "pass"
        and public_delta is not None
        and threshold is not None
        and public_delta > threshold
    )
    seed_count = int(cand_stats.get("count") or 0)
    expand = should_expand_to_five_seeds(public_delta=public_delta, threshold=threshold, pooled_se=pooled_se, seed_count=seed_count, seed_policy=seed_policy)
    return {
        "schema_version": "humanoid_agent.phase6.public_decision.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "accept" if accepted else "reject",
        "accepted": accepted,
        "needs_more_seeds": expand,
        "decision_basis": {
            "public_eval_only": True,
            "heldout_used_for_decision": False,
            "seed_count": seed_count,
            "rule": "accept iff candidate_public_mean - best_public_mean > max(public_min_delta, pooled_multiplier * pooled_standard_error)",
            "public_min_delta": public_min_delta,
            "pooled_se_multiplier": pooled_multiplier,
            "public_delta": public_delta,
            "pooled_standard_error": pooled_se,
            "acceptance_threshold": threshold,
            "margin_over_threshold": None if public_delta is None or threshold is None else public_delta - threshold,
        },
        "best": {
            "summary_file": str(best_summary_path),
            "public_success_rate_stats": best_stats,
        },
        "candidate": {
            "summary_file": str(candidate_summary_path),
            "public_success_rate_stats": cand_stats,
        },
        "public_seed_deltas": _paired_deltas(best_summary, candidate_summary),
        "operation_diff": operation_diff or {},
        "feedback_safe": {
            "can_feed_to_agent": True,
            "contains_heldout_results": False,
            "contains_eval_seed_config": False,
        },
    }


def should_expand_to_five_seeds(
    *,
    public_delta: float | None,
    threshold: float | None,
    pooled_se: float | None,
    seed_count: int,
    seed_policy: dict[str, Any] | None = None,
) -> bool:
    seed_policy = seed_policy or {}
    initial = len(seed_policy.get("initial_train_seeds", [101, 102, 103]))
    expanded = len(seed_policy.get("expanded_train_seeds", [101, 102, 103, 104, 105]))
    if seed_count >= expanded or seed_count != initial:
        return False
    if public_delta is None or threshold is None:
        return True
    buffer_fraction = float(seed_policy.get("expand_if_abs_margin_below_threshold_fraction", 0.5))
    buffer = (pooled_se if pooled_se is not None else threshold) * buffer_fraction
    return abs(public_delta - threshold) <= buffer


def public_feedback_from_decision(decision: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "humanoid_agent.phase6.public_feedback.v1",
        "decision": decision.get("decision"),
        "accepted": decision.get("accepted"),
        "needs_more_seeds": decision.get("needs_more_seeds"),
        "decision_basis": decision.get("decision_basis"),
        "candidate_public_success_rate_stats": decision.get("candidate", {}).get("public_success_rate_stats"),
        "best_public_success_rate_stats": decision.get("best", {}).get("public_success_rate_stats"),
        "public_seed_deltas": _anonymized_seed_deltas(decision.get("public_seed_deltas", [])),
        "operation_diff": decision.get("operation_diff", {}),
        "feedback_safe": {
            "public_only": True,
            "contains_heldout_results": False,
            "contains_eval_seed_config": False,
        },
    }


def _anonymized_seed_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    safe_rows = []
    for index, item in enumerate(rows, start=1):
        safe_rows.append(
            {
                "run_index": index,
                "best_public_success_rate": item.get("best_public_success_rate"),
                "candidate_public_success_rate": item.get("candidate_public_success_rate"),
                "delta": item.get("delta"),
            }
        )
    return safe_rows


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _mean(stats: dict[str, Any]) -> float | None:
    value = stats.get("mean")
    return None if value is None else float(value)


def _paired_deltas(best_summary: dict[str, Any], candidate_summary: dict[str, Any]) -> list[dict[str, Any]]:
    best_by_seed = {item.get("seed"): item for item in best_summary.get("seed_results", [])}
    rows = []
    for cand_item in candidate_summary.get("seed_results", []):
        seed = cand_item.get("seed")
        best_item = best_by_seed.get(seed)
        if not isinstance(best_item, dict):
            continue
        best_rate = best_item.get("public_success_rate")
        cand_rate = cand_item.get("public_success_rate")
        if best_rate is None or cand_rate is None:
            continue
        rows.append(
            {
                "seed": seed,
                "best_public_success_rate": float(best_rate),
                "candidate_public_success_rate": float(cand_rate),
                "delta": float(cand_rate) - float(best_rate),
            }
        )
    return rows
