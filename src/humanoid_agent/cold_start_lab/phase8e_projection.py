"""Public and prompt-facing projections for Phase 8E artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from humanoid_agent.code_sandbox.loader import candidate_package_fingerprint, load_candidate_manifest
from humanoid_agent.cold_start_lab.phase8e_common import scrub_prompt_facing_value


class CandidateOptionalJsonError(ValueError):
    """Raised when a provider-authored optional JSON artifact cannot be projected."""


def latest_accepted_record(ledger: list[dict[str, Any]]) -> dict[str, Any] | None:
    for record in reversed(ledger):
        decision = record.get("decision", {}) if isinstance(record.get("decision"), dict) else {}
        if decision.get("accepted") is True:
            return record
    return None


def project_candidate_package(candidate_dir: Path | None, *, include_public_artifacts: bool = True) -> dict[str, Any] | None:
    if candidate_dir is None:
        return None
    manifest = load_candidate_manifest(candidate_dir)
    fingerprint = candidate_package_fingerprint(manifest)
    return {
        "candidate_dir": str(Path(candidate_dir).resolve()),
        "manifest": {
            "schema_version": manifest.schema_version,
            "candidate_id": manifest.candidate_id,
            "description": manifest.description,
            "author": manifest.author,
            "llm_authored": manifest.llm_authored,
            "entrypoint": manifest.entrypoint,
            "phase": manifest.phase,
            "expected_behavior": manifest.expected_behavior,
        },
        "fingerprint": fingerprint,
        "operation_diff": _load_optional_json(Path(candidate_dir) / "operation_diff.json") if include_public_artifacts else None,
        "diagnostics": _load_optional_json(Path(candidate_dir) / "diagnostics.json") if include_public_artifacts else None,
    }


def prompt_facing_candidate_package(candidate_dir: Path | None) -> dict[str, Any] | None:
    projected = project_candidate_package(candidate_dir) if candidate_dir is not None else None
    if projected is None:
        return None
    manifest = projected.get("manifest", {}) if isinstance(projected.get("manifest"), dict) else {}
    projected["candidate_dir"] = _prompt_facing_candidate_ref(candidate_dir, manifest)
    projected["fingerprint"] = _prompt_facing_fingerprint(projected.get("fingerprint"))
    return scrub_prompt_facing_value(projected)


def project_verifier_summary(summary: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(summary, dict):
        return None
    issue_codes = _verifier_issue_codes(summary)
    checks = summary.get("checks", {})
    capabilities = summary.get("capabilities", {})
    hard_isolation = summary.get("phase8c_hard_isolation")
    if hard_isolation is None and isinstance(capabilities, dict):
        hard_isolation = bool(capabilities.get("landlock_abi_version"))
    return {
        "status": summary.get("status"),
        "fingerprint": _prompt_facing_fingerprint(summary.get("fingerprint")),
        "static_status": summary.get("static_status")
        or (checks.get("static", {}).get("status") if isinstance(checks, dict) else None),
        "dynamic_status": summary.get("dynamic_status")
        or (checks.get("dynamic", {}).get("status") if isinstance(checks, dict) else None),
        "issue_codes": issue_codes,
        "phase8c_hard_isolation": bool(hard_isolation),
    }


def project_public_result(result: dict[str, Any] | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "schema_version": result.get("schema_version", "humanoid_agent.phase8e.public_result.v1"),
        "status": result.get("status"),
        "public_dev": _project_eval_split(result.get("public_dev")),
        "public_accept": _project_eval_split(result.get("public_accept")),
        "ood_stress_gate": _project_gate(result.get("ood_stress_gate")),
        "motion_quality_gate": _project_gate(result.get("motion_quality_gate")),
        "budget_accounting": _project_gate(result.get("budget_accounting")),
    }


def project_ledger_record(record: dict[str, Any]) -> dict[str, Any]:
    decision = record.get("decision", {}) if isinstance(record.get("decision"), dict) else {}
    detail = decision.get("detail", {}) if isinstance(decision.get("detail"), dict) else {}
    candidate = record.get("candidate", {}) if isinstance(record.get("candidate"), dict) else {}
    manifest = candidate.get("manifest", {}) if isinstance(candidate.get("manifest"), dict) else {}
    projected = {
        "created_at_utc": record.get("created_at_utc"),
        "candidate_id": manifest.get("candidate_id"),
        "decision": decision.get("decision"),
        "accepted": decision.get("accepted"),
        "reason": decision.get("reason"),
        "verifier": project_verifier_summary(record.get("verifier")),
        "public_result": _prompt_history_public_result(decision.get("public_result")),
        "operation_diff": candidate.get("operation_diff"),
        "public_failure_feedback": detail.get("public_failure_feedback"),
        "curriculum_lineage": _project_curriculum_lineage(record.get("curriculum_lineage") or detail.get("curriculum_lineage")),
    }
    return scrub_prompt_facing_value(projected)


def build_public_history_summary(public_history: list[dict[str, Any]]) -> dict[str, Any]:
    """Build runner-computed, prompt-facing feedback signals over public history."""

    round_summaries = [
        summary
        for summary in (_round_feedback_summary(item) for item in public_history)
        if summary is not None
    ]
    if not round_summaries:
        return {
            "schema_version": "humanoid_agent.phase8e.public_feedback_summary.v1",
            "status": "no_public_history",
            "round_count": 0,
        }

    latest = round_summaries[-1]
    latest_with_mean = _latest_summary_with_mean(round_summaries)
    best = _best_summary(round_summaries)
    previous_accepted = _previous_accepted_summary(round_summaries, before=latest_with_mean)
    trend = _success_rate_trend(latest_with_mean, previous_accepted)
    return scrub_prompt_facing_value(
        {
            "schema_version": "humanoid_agent.phase8e.public_feedback_summary.v1",
            "status": "available",
            "round_count": len(round_summaries),
            "accepted_count": sum(1 for item in round_summaries if item.get("accepted") is True),
            "rejected_count": sum(1 for item in round_summaries if item.get("accepted") is False),
            "latest_round": latest,
            "latest_public_dev_round": latest_with_mean,
            "best_public_dev_round": best,
            "trend_vs_previous_accepted": trend,
            "curriculum_attribution_summary": _curriculum_attribution_summary(round_summaries),
        }
    )


def _curriculum_attribution_summary(round_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    accepted_with_lineage: list[dict[str, Any]] = []
    gain_counts: dict[str, int] = {}
    for item in round_summaries:
        if item.get("accepted") is not True:
            continue
        lineage = item.get("curriculum_lineage")
        attribution = lineage.get("attribution") if isinstance(lineage, dict) else None
        if not isinstance(attribution, dict) or not attribution:
            continue
        projected = {
            "candidate_id": item.get("candidate_id"),
            "reward_code_changed": bool(attribution.get("reward_code_changed")),
            "curriculum_changed": bool(attribution.get("curriculum_changed")),
            "diagnostics_changed": bool(attribution.get("diagnostics_changed")),
            "capability_changed": bool(attribution.get("capability_changed")),
            "gain_attributable_to": attribution.get("gain_attributable_to"),
            "pure_knob_candidate": bool(attribution.get("pure_knob_candidate")),
        }
        accepted_with_lineage.append(projected)
        gain = projected.get("gain_attributable_to")
        if isinstance(gain, str) and gain:
            gain_counts[gain] = gain_counts.get(gain, 0) + 1

    if not accepted_with_lineage:
        return {
            "status": "no_accepted_curriculum_lineage",
            "accepted_with_curriculum_lineage_count": 0,
            "current_accepted_pure_knob_streak": 0,
        }

    pure_knob_streak = 0
    for item in reversed(accepted_with_lineage):
        if item.get("pure_knob_candidate") is True:
            pure_knob_streak += 1
        else:
            break
    warning = "accepted_curriculum_degenerated_to_pure_knob_streak" if pure_knob_streak >= 2 else None
    latest = accepted_with_lineage[-1]
    return {
        "status": "warning" if warning else "available",
        "warning": warning,
        "accepted_with_curriculum_lineage_count": len(accepted_with_lineage),
        "accepted_pure_knob_count": sum(1 for item in accepted_with_lineage if item.get("pure_knob_candidate") is True),
        "accepted_curriculum_changed_count": sum(1 for item in accepted_with_lineage if item.get("curriculum_changed") is True),
        "current_accepted_pure_knob_streak": pure_knob_streak,
        "gain_attribution_counts": gain_counts,
        "latest_accepted_attribution": latest,
    }


def _verifier_issue_codes(summary: dict[str, Any]) -> dict[str, int]:
    existing = summary.get("issue_codes")
    if isinstance(existing, dict):
        out: dict[str, int] = {}
        for key, value in existing.items():
            try:
                out[str(key)] = int(value)
            except (TypeError, ValueError):
                continue
        return out

    issues = summary.get("issues", [])
    out = {}
    for item in issues if isinstance(issues, list) else []:
        if not isinstance(item, dict):
            continue
        code = str(item.get("code", "unknown"))
        out[code] = out.get(code, 0) + 1
    return out


def _prompt_facing_fingerprint(fingerprint: Any) -> dict[str, Any] | None:
    if not isinstance(fingerprint, dict):
        return None
    return {
        key: value
        for key, value in fingerprint.items()
        if key not in {"path", "manifest_path", "entrypoint_path"}
    }


def _round_feedback_summary(record: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(record, dict):
        return None
    public_result = record.get("public_result")
    public_dev = public_result.get("public_dev") if isinstance(public_result, dict) else None
    stats = public_dev.get("success_rate_stats") if isinstance(public_dev, dict) else None
    seed_results = public_dev.get("seed_results") if isinstance(public_dev, dict) else None
    if not isinstance(stats, dict):
        stats = {}
    if not isinstance(seed_results, list):
        seed_results = []

    summary: dict[str, Any] = {
        "candidate_id": record.get("candidate_id"),
        "decision": record.get("decision"),
        "accepted": record.get("accepted"),
        "reason": record.get("reason"),
    }
    mean = _as_float(stats.get("mean"))
    if mean is not None:
        summary["public_dev_mean_success_rate"] = mean
    seed_count = stats.get("count")
    if seed_count is None and seed_results:
        seed_count = len(seed_results)
    if seed_count is not None:
        summary["public_dev_seed_count"] = seed_count
    dominant_failure = _dominant_failure(seed_results)
    if dominant_failure is not None:
        summary["dominant_failure"] = dominant_failure

    gate_statuses = _public_gate_statuses(public_result)
    if gate_statuses:
        summary["gate_statuses"] = gate_statuses
    curriculum_lineage = record.get("curriculum_lineage")
    if isinstance(curriculum_lineage, dict):
        summary["curriculum_lineage"] = _project_curriculum_lineage(curriculum_lineage)
    return {key: value for key, value in summary.items() if value is not None}


def _project_curriculum_lineage(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    attribution = value.get("attribution")
    return {
        "status": value.get("status"),
        "compiler_version": value.get("compiler_version"),
        "stage_count": value.get("stage_count"),
        "execution_mode": value.get("execution_mode"),
        "capabilities": value.get("capabilities") if isinstance(value.get("capabilities"), list) else [],
        "attribution": dict(attribution) if isinstance(attribution, dict) else {},
    }


def _public_gate_statuses(public_result: Any) -> dict[str, Any]:
    if not isinstance(public_result, dict):
        return {}
    out: dict[str, Any] = {}
    for key in ("ood_stress_gate", "motion_quality_gate", "budget_accounting"):
        gate = public_result.get(key)
        if isinstance(gate, dict) and gate.get("status") is not None:
            out[key] = gate.get("status")
    return out


def _dominant_failure(seed_results: list[Any]) -> dict[str, Any] | None:
    count_totals: dict[str, float] = {}
    rate_totals: dict[str, float] = {}
    count_denominator = 0.0
    count_seed_count = 0
    rate_seed_count = 0
    for item in seed_results:
        if not isinstance(item, dict):
            continue
        breakdown = item.get("failure_breakdown")
        if not isinstance(breakdown, dict):
            continue
        counts = breakdown.get("counts")
        if isinstance(counts, dict):
            seed_counts: dict[str, float] = {}
            seed_total_from_counts = 0.0
            for label, value in counts.items():
                numeric = _as_float(value)
                if numeric is None:
                    continue
                seed_total_from_counts += numeric
                text_label = str(label)
                if text_label == "success":
                    continue
                seed_counts[text_label] = numeric
            seed_total = _as_float(breakdown.get("total"))
            if seed_total is None:
                seed_total = seed_total_from_counts
            if seed_total is not None and seed_total > 0:
                count_denominator += seed_total
                count_seed_count += 1
            if seed_counts:
                for text_label, numeric in seed_counts.items():
                    count_totals[text_label] = count_totals.get(text_label, 0.0) + numeric
            continue
        rates = breakdown.get("rates")
        if isinstance(rates, dict):
            has_rate_value = False
            for label, value in rates.items():
                text_label = str(label)
                numeric = _as_float(value)
                if numeric is None:
                    continue
                has_rate_value = True
                if text_label == "success":
                    continue
                rate_totals[text_label] = rate_totals.get(text_label, 0.0) + numeric
            if has_rate_value:
                rate_seed_count += 1
                continue
        has_rate_value = False
        for label, value in breakdown.items():
            if label in {"counts", "dominant_failure", "rates", "total"}:
                continue
            numeric = _as_float(value)
            if numeric is None:
                continue
            has_rate_value = True
            text_label = str(label)
            if text_label == "success":
                continue
            rate_totals[text_label] = rate_totals.get(text_label, 0.0) + numeric
        if has_rate_value:
            rate_seed_count += 1
            continue
    if count_totals and count_denominator > 0:
        label, count = sorted(count_totals.items(), key=lambda item: (-item[1], item[0]))[0]
        return {
            "label": label,
            "rate": count / count_denominator,
            "count": count,
            "denominator": count_denominator,
            "seed_count": count_seed_count,
        }
    if rate_totals:
        label, total_rate = sorted(rate_totals.items(), key=lambda item: (-item[1], item[0]))[0]
        if rate_seed_count <= 0:
            return None
        return {
            "label": label,
            "rate": total_rate / rate_seed_count,
            "seed_count": rate_seed_count,
        }
    return None


def _latest_summary_with_mean(round_summaries: list[dict[str, Any]]) -> dict[str, Any] | None:
    for item in reversed(round_summaries):
        if _as_float(item.get("public_dev_mean_success_rate")) is not None:
            return item
    return None


def _best_summary(round_summaries: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [
        item
        for item in round_summaries
        if _as_float(item.get("public_dev_mean_success_rate")) is not None
    ]
    if not eligible:
        return None
    return max(eligible, key=lambda item: _as_float(item.get("public_dev_mean_success_rate")) or float("-inf"))


def _previous_accepted_summary(
    round_summaries: list[dict[str, Any]],
    *,
    before: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if before is None:
        return None
    before_index = next(
        (index for index, item in enumerate(round_summaries) if item is before),
        None,
    )
    if before_index is None:
        return None
    for item in reversed(round_summaries[:before_index]):
        if item.get("accepted") is True and _as_float(item.get("public_dev_mean_success_rate")) is not None:
            return item
    return None


def _success_rate_trend(
    latest: dict[str, Any] | None,
    previous_accepted: dict[str, Any] | None,
) -> dict[str, Any]:
    latest_mean = _as_float(latest.get("public_dev_mean_success_rate")) if isinstance(latest, dict) else None
    previous_mean = (
        _as_float(previous_accepted.get("public_dev_mean_success_rate"))
        if isinstance(previous_accepted, dict)
        else None
    )
    if latest_mean is None or previous_mean is None:
        return {
            "direction": "unknown",
            "reason": "no_previous_accepted_public_dev",
        }
    delta = latest_mean - previous_mean
    if delta > 0:
        direction = "up"
    elif delta < 0:
        direction = "down"
    else:
        direction = "flat"
    return {
        "direction": direction,
        "delta": delta,
        "latest_candidate_id": latest.get("candidate_id"),
        "previous_accepted_candidate_id": previous_accepted.get("candidate_id"),
    }


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def prompt_history_public_result(public_result: Any) -> dict[str, Any] | None:
    projected = _prompt_history_public_result(public_result)
    return scrub_prompt_facing_value(projected) if projected is not None else None


def _prompt_history_public_result(public_result: Any) -> dict[str, Any] | None:
    """Project a public result for the next-round prompt.

    ``public_accept`` decides accept/reject and must never become a next-round
    optimization signal, so it is withheld from prompt history.  ``public_dev``
    stays as feedback; gates are reduced to pass/fail status only.
    """

    if not isinstance(public_result, dict):
        return None
    return {
        "public_dev": public_result.get("public_dev"),
        "ood_stress_gate": _gate_status_only(public_result.get("ood_stress_gate")),
        "motion_quality_gate": _gate_status_only(public_result.get("motion_quality_gate")),
        "budget_accounting": _gate_status_only(public_result.get("budget_accounting")),
        "public_accept": "withheld_from_prompt_history",
    }


def _gate_status_only(gate: Any) -> dict[str, Any] | None:
    if not isinstance(gate, dict):
        return None
    out: dict[str, Any] = {"status": gate.get("status")}
    failures = gate.get("failures")
    if gate.get("status") != "pass" and isinstance(failures, list) and failures:
        # Carry a compact, public-safe digest of which metrics failed so the next
        # round can learn the concrete cause (e.g. exploded action magnitude),
        # not just that "a gate failed".
        out["failures"] = failures[:8]
    return out


def _project_eval_split(split: Any) -> dict[str, Any] | None:
    if not isinstance(split, dict):
        return None
    seed_results = []
    for index, item in enumerate(split.get("seed_results", []), start=1):
        if not isinstance(item, dict):
            continue
        seed_results.append(
            {
                "run_index": index,
                "success_rate": item.get("success_rate"),
                "failure_breakdown": item.get("failure_breakdown"),
                "condition_flag_breakdown": item.get("condition_flag_breakdown"),
            }
        )
    return {
        "success_rate_stats": _project_stats(split.get("success_rate_stats")),
        "seed_results": seed_results,
    }


def _project_stats(stats: Any) -> dict[str, Any] | None:
    if not isinstance(stats, dict):
        return None
    allowed = ("mean", "std", "standard_error", "count", "min", "max")
    return {key: stats.get(key) for key in allowed if key in stats}


def _project_gate(gate: Any) -> dict[str, Any] | None:
    if not isinstance(gate, dict):
        return None
    return {
        "status": gate.get("status"),
        "summary": gate.get("summary"),
        "violations": gate.get("violations", []),
        "failures": _gate_failure_digest(gate) if gate.get("status") != "pass" else [],
    }


def _gate_failure_digest(gate: dict[str, Any]) -> list[dict[str, Any]]:
    """Compact, public-safe digest of a failed gate's worst per-metric failures.

    Deduplicates by metric keeping the largest-magnitude offender, so the next
    prompt sees concrete causes (metric, actual, limit) without raw eval seeds or
    runner internals.
    """

    failures = gate.get("failures")
    if not isinstance(failures, list):
        failures = gate.get("violations")
    if not isinstance(failures, list):
        return []
    worst: dict[str, dict[str, Any]] = {}
    for item in failures:
        if not isinstance(item, dict):
            continue
        metric = str(item.get("metric") or item.get("key") or item.get("reason") or "violation")
        actual = item.get("actual")
        try:
            magnitude = abs(float(actual)) if actual is not None else 0.0
        except (TypeError, ValueError):
            magnitude = 0.0
        existing = worst.get(metric)
        if existing is None or magnitude > existing.get("_magnitude", 0.0):
            worst[metric] = {
                "metric": metric,
                "actual": actual,
                "limit": item.get("limit"),
                "reason": item.get("reason"),
                "_magnitude": magnitude,
            }
    digest = sorted(worst.values(), key=lambda entry: entry.get("_magnitude", 0.0), reverse=True)[:8]
    for entry in digest:
        entry.pop("_magnitude", None)
    return digest


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CandidateOptionalJsonError(f"{path.name} invalid JSON: {_json_error_summary(exc)}") from exc
    return data if isinstance(data, dict) else {"value": data}


def _json_error_summary(exc: json.JSONDecodeError) -> str:
    return f"{exc.msg} at line {exc.lineno} column {exc.colno}"


def _prompt_facing_candidate_ref(candidate_dir: Path | None, manifest: dict[str, Any]) -> str:
    candidate_id = manifest.get("candidate_id")
    if isinstance(candidate_id, str) and candidate_id:
        return f"current_recipe/{candidate_id}"
    label = candidate_dir.name if candidate_dir is not None else "unknown_candidate"
    return f"current_recipe/{label}"
