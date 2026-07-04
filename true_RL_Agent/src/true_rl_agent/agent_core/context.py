"""Public context construction for next-round prompts."""

from __future__ import annotations

import json
from typing import Any, Mapping

from true_rl_agent.agent_core.public_safety import (
    FORBIDDEN_PUBLIC_KEY_PARTS,
    contains_forbidden_public_value_marker,
    sanitize_prompt_public_value,
)

PUBLIC_HISTORY_MAX_RECENT_ROUNDS = 8
PUBLIC_HISTORY_MAX_SERIALIZED_BYTES = 8192
_PUBLIC_HISTORY_MAX_LIST_ITEMS = 8
_PUBLIC_HISTORY_TEXT_LIMIT = 500
_PUBLIC_HISTORY_ENTRY_KEYS = (
    "round_id",
    "outcome",
    "accepted",
    "reason",
    "surface_classification",
    "public_score",
    "changed_components",
    "candidate_package_loaded",
    "candidate_code_loaded_in_eval",
    "runtime_activation",
    "mechanism_attribution",
    "runner_error",
    "provider_failure",
    "repair_feedback",
)
_PUBLIC_HISTORY_BLOCKED_KEY_PARTS = (
    *FORBIDDEN_PUBLIC_KEY_PARTS,
    "seed_id",
    "seed_policy",
    "push_active",
    "push_vec_public",
)
_PUBLIC_HISTORY_BLOCKED_TEXT_PARTS = (
    "seed_id",
    "seed_policy",
    "push_active",
    "push_vec_public",
)
_PUBLIC_HISTORY_DETAIL_KEYS = (
    "status",
    "code",
    "category",
    "error_type",
    "exception_type",
    "issue_count",
)


def sanitize_public_context(value: Any) -> Any:
    if isinstance(value, Mapping):
        clean: dict[str, Any] = {}
        for key, item in value.items():
            lowered = str(key).lower()
            if any(part in lowered for part in FORBIDDEN_PUBLIC_KEY_PARTS):
                continue
            clean[str(key)] = sanitize_public_context(item)
        return clean
    if isinstance(value, list):
        return [sanitize_public_context(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_public_context(item) for item in value]
    if isinstance(value, str):
        if contains_forbidden_public_value_marker(value):
            return "[redacted]"
    return value


def sanitize_prompt_public_context(value: Any) -> Any:
    """Sanitize public context immediately before prompt exposure."""

    return sanitize_prompt_public_value(sanitize_public_context(value))


def build_public_context(ledger_records: list[dict[str, Any]]) -> dict[str, Any]:
    accepted_scores = [
        record.get("decision", {}).get("public_score")
        for record in ledger_records
        if record.get("decision", {}).get("accepted") is True
    ]
    numeric_accepted_scores = [float(score) for score in accepted_scores if isinstance(score, int | float)]
    last_record = ledger_records[-1] if ledger_records else {}
    recent_outcomes, omitted_round_count = _bounded_recent_outcomes(ledger_records)
    context = {
        "public_only": True,
        "previous_rounds": len(ledger_records),
        "best_public_score": max(numeric_accepted_scores) if numeric_accepted_scores else None,
        "last_outcome": last_record.get("decision", {}).get("outcome"),
        "pure_knob_streak": _pure_knob_streak(ledger_records),
        "public_history_limits": {
            "max_recent_rounds": PUBLIC_HISTORY_MAX_RECENT_ROUNDS,
            "max_serialized_bytes": PUBLIC_HISTORY_MAX_SERIALIZED_BYTES,
        },
        "public_history_omitted_round_count": omitted_round_count,
        "recent_outcomes": recent_outcomes,
    }
    return sanitize_prompt_public_context(context)


def build_public_history_section(public_context: Mapping[str, Any]) -> dict[str, Any]:
    recent_outcomes = public_context.get("recent_outcomes")
    if not isinstance(recent_outcomes, list):
        recent_outcomes = []
    recent, extra_omitted = _bounded_history_entries(recent_outcomes)
    omitted_count = _nonnegative_int(public_context.get("public_history_omitted_round_count")) + extra_omitted
    return sanitize_prompt_public_context(
        {
            "schema_version": "true_rl_agent.public_history.v1",
            "previous_rounds": public_context.get("previous_rounds", 0),
            "best_public_score": public_context.get("best_public_score"),
            "last_outcome": public_context.get("last_outcome"),
            "pure_knob_streak": public_context.get("pure_knob_streak", 0),
            "limits": {
                "max_recent_rounds": PUBLIC_HISTORY_MAX_RECENT_ROUNDS,
                "max_serialized_bytes": PUBLIC_HISTORY_MAX_SERIALIZED_BYTES,
            },
            "omitted_round_count": omitted_count,
            "recent_outcomes": recent,
        }
    )


def _pure_knob_streak(ledger_records: list[dict[str, Any]]) -> int:
    streak = 0
    for record in reversed(ledger_records):
        if record.get("candidate_surface", {}).get("classification") != "parameter_only":
            break
        streak += 1
    return streak


def _bounded_recent_outcomes(ledger_records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    entries = [_round_history_entry(record) for record in ledger_records]
    return _bounded_history_entries(entries)


def _bounded_history_entries(entries: list[Any]) -> tuple[list[dict[str, Any]], int]:
    recent_entries = [
        _compact_history_entry(entry)
        for entry in entries[-PUBLIC_HISTORY_MAX_RECENT_ROUNDS:]
        if isinstance(entry, Mapping)
    ]
    omitted_count = max(0, len(entries) - len(recent_entries))
    while recent_entries and _serialized_size(recent_entries) > PUBLIC_HISTORY_MAX_SERIALIZED_BYTES:
        recent_entries.pop(0)
        omitted_count += 1
    return recent_entries, omitted_count


def _compact_history_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key in _PUBLIC_HISTORY_ENTRY_KEYS:
        value = entry.get(key)
        if value is None and key == "surface_classification":
            value = entry.get("surface")
        compacted = _compact_history_field(key, value)
        if compacted is not None:
            compact[key] = compacted
    return sanitize_prompt_public_context(compact)


def _compact_history_field(key: str, value: Any) -> Any:
    if value is None:
        return None
    if key in {"round_id", "outcome", "reason", "surface_classification"}:
        return _optional_text(value)
    if key == "accepted":
        return value if isinstance(value, bool) else None
    if key == "public_score":
        return float(value) if isinstance(value, int | float) else None
    if key == "changed_components":
        return sorted(_list_of_strings(value))
    if key in {"candidate_package_loaded", "candidate_code_loaded_in_eval"}:
        return value if isinstance(value, bool) else _optional_text(value)
    if key == "runtime_activation":
        return _runtime_activation_public_summary(value)
    if key == "mechanism_attribution":
        return _status_reason_summary(value)
    if key == "runner_error":
        return _error_summary(value)
    if key == "provider_failure":
        return _provider_failure_summary(value)
    if key == "repair_feedback":
        return _repair_feedback_value_summary(value)
    return _compact_value(value)


def _round_history_entry(record: Mapping[str, Any]) -> dict[str, Any]:
    decision = _mapping(record.get("decision"))
    candidate_surface = _mapping(record.get("candidate_surface"))
    public_result = _mapping(record.get("public_result"))
    entry: dict[str, Any] = {
        "round_id": _optional_text(record.get("round_id")),
        "outcome": _optional_text(decision.get("outcome")),
        "accepted": decision.get("accepted") if isinstance(decision.get("accepted"), bool) else None,
        "reason": _optional_text(decision.get("reason")),
        "surface_classification": _first_present(
            _optional_text(decision.get("surface_classification")),
            _optional_text(candidate_surface.get("classification")),
        ),
        "public_score": _public_score(decision=decision, public_result=public_result),
        "changed_components": _changed_components(record),
        "candidate_package_loaded": _candidate_package_loaded(record),
        "candidate_code_loaded_in_eval": _candidate_code_loaded_in_eval(record),
        "runtime_activation": _runtime_activation_summary(record),
        "mechanism_attribution": _mechanism_attribution(record),
    }
    runner_error = _error_summary(record.get("runner_error"))
    if runner_error is not None:
        entry["runner_error"] = runner_error
    provider_failure = _provider_failure_summary(record.get("provider_failure"))
    if provider_failure is not None:
        entry["provider_failure"] = provider_failure
    repair_feedback = _repair_feedback_summary(record)
    if repair_feedback is not None:
        entry["repair_feedback"] = repair_feedback
    return entry


def _public_score(*, decision: Mapping[str, Any], public_result: Mapping[str, Any]) -> float | None:
    for value in (decision.get("public_score"), public_result.get("public_score")):
        if isinstance(value, int | float):
            return float(value)
    return None


def _changed_components(record: Mapping[str, Any]) -> list[str]:
    candidate = _mapping(record.get("candidate"))
    raw = candidate.get("changed_components")
    if not isinstance(raw, list | tuple):
        verifier = _mapping(record.get("verifier"))
        details = _mapping(verifier.get("details"))
        raw = details.get("derived_change_set")
    return sorted(_list_of_strings(raw))


def _candidate_package_loaded(record: Mapping[str, Any]) -> bool | str:
    train_summary = _train_summary(record)
    candidate_summary = _mapping(train_summary.get("candidate"))
    metrics = _public_metrics(record)
    value = _first_present(
        train_summary.get("candidate_package_loaded"),
        train_summary.get("candidate_runtime_enabled"),
        candidate_summary.get("enabled"),
        metrics.get("candidate_package_loaded"),
    )
    return value if isinstance(value, bool) else "unknown"


def _candidate_code_loaded_in_eval(record: Mapping[str, Any]) -> bool | str:
    metrics = _public_metrics(record)
    train_summary = _train_summary(record)
    value = _first_present(
        metrics.get("candidate_code_loaded_in_eval"),
        train_summary.get("candidate_code_loaded_in_eval"),
    )
    return value if isinstance(value, bool) else "unknown"


def _runtime_activation_summary(record: Mapping[str, Any]) -> dict[str, Any]:
    runtime = _mapping(record.get("runtime_activation_summary"))
    if not runtime:
        return {"status": "not_available"}
    return _runtime_activation_public_summary(runtime)


def _runtime_activation_public_summary(value: Any) -> dict[str, Any]:
    runtime = _mapping(value)
    if not runtime:
        return {"status": "not_available"}
    return {
        "status": "available",
        "reward_term_calls": _int_or_zero(runtime.get("reward_term_calls")),
        "diagnostic_calls": _int_or_zero(runtime.get("diagnostic_calls")),
        "nonzero_reward_contribution": bool(runtime.get("nonzero_reward_contribution", False)),
        "capability_activation_records": _list_of_strings(
            runtime.get("capability_activation_records"),
            max_items=_PUBLIC_HISTORY_MAX_LIST_ITEMS,
        ),
        "feature_activation_records": _list_of_strings(
            runtime.get("feature_activation_records"),
            max_items=_PUBLIC_HISTORY_MAX_LIST_ITEMS,
        ),
        "reward_output_stats": _compact_stats_mapping(runtime.get("reward_output_stats")),
        "diagnostic_outputs": _compact_stats_mapping(runtime.get("diagnostic_outputs")),
    }


def _mechanism_attribution(record: Mapping[str, Any]) -> dict[str, Any]:
    mechanism = _mapping(record.get("mechanism_attribution"))
    if not mechanism:
        return {"status": "not_available"}
    return _status_reason_summary(mechanism)


def _status_reason_summary(value: Any) -> dict[str, Any]:
    mechanism = _mapping(value)
    if not mechanism:
        return {"status": "not_available"}
    return {
        "status": _optional_text(mechanism.get("status")),
        "reason": _optional_text(mechanism.get("reason")),
    }


def _provider_failure_summary(value: Any) -> dict[str, Any] | None:
    failure = _mapping(value)
    if not failure:
        return None
    summary = {
        "category": _optional_text(failure.get("category")),
        "reason": _optional_text(failure.get("reason")),
    }
    details = _compact_detail_mapping(failure.get("details"))
    if details:
        summary["details"] = details
    return summary


def _repair_feedback_summary(record: Mapping[str, Any]) -> dict[str, Any] | None:
    feedback = _mapping(record.get("candidate_repair_feedback"))
    if not feedback:
        public_context = _mapping(record.get("public_context"))
        feedback = _mapping(public_context.get("candidate_repair_feedback"))
    if not feedback:
        return None
    return _repair_feedback_value_summary(feedback)


def _repair_feedback_value_summary(value: Any) -> dict[str, Any] | None:
    feedback = _mapping(value)
    if not feedback:
        return None
    summary = {
        "failure_category": _optional_text(feedback.get("failure_category")),
        "reason": _optional_text(feedback.get("reason")),
    }
    status = _first_present(feedback.get("status"), feedback.get("provider_status"), feedback.get("runtime_status"))
    if status is not None:
        summary["status"] = _optional_text(status)
    return summary


def _error_summary(value: Any) -> dict[str, Any] | None:
    error = _mapping(value)
    if not error:
        return None
    summary = {
        "type": _optional_text(error.get("type")),
        "message": _optional_text(error.get("message")),
    }
    details = _compact_detail_mapping(error.get("details"))
    if details:
        summary["details"] = details
    return summary


def _train_summary(record: Mapping[str, Any]) -> Mapping[str, Any]:
    train_result = _mapping(record.get("train_result"))
    return _mapping(train_result.get("summary"))


def _public_metrics(record: Mapping[str, Any]) -> Mapping[str, Any]:
    public_result = _mapping(record.get("public_result"))
    return _mapping(public_result.get("metrics"))


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _compact_mapping(value: Any, *, max_items: int = 6) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    compact: dict[str, Any] = {}
    for key, item in value.items():
        key_text = str(key)
        if _blocked_public_history_key(key_text):
            continue
        compact[key_text] = _compact_value(item)
        if len(compact) >= max_items:
            break
    return compact


def _compact_detail_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    compact: dict[str, Any] = {}
    for key in _PUBLIC_HISTORY_DETAIL_KEYS:
        if key in value:
            compact[key] = _compact_value(value[key])
    omitted = sum(1 for key in value if str(key) not in compact)
    if omitted:
        compact["omitted_detail_count"] = omitted
    return compact


def _compact_stats_mapping(value: Any) -> dict[str, Any]:
    stats = _mapping(value)
    if not stats:
        return {}
    allowed_keys = (
        "mean",
        "std",
        "min",
        "max",
        "finite_fraction",
        "nonzero_fraction",
        "nan_fraction",
        "inf_fraction",
        "count",
    )
    return {key: _compact_value(stats[key]) for key in allowed_keys if key in stats}


def _compact_value(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, str):
        return _bounded_text(value)
    if isinstance(value, Mapping):
        return _compact_mapping(value, max_items=4)
    if isinstance(value, list | tuple):
        strings = _list_of_strings(value, max_items=4)
        return strings if strings else f"{len(value)} items"
    return _bounded_text(str(value))


def _list_of_strings(value: Any, *, max_items: int = _PUBLIC_HISTORY_MAX_LIST_ITEMS) -> list[str]:
    if not isinstance(value, list | tuple | set | frozenset):
        return []
    strings = [str(item) for item in value if isinstance(item, str)]
    return list(dict.fromkeys(strings))[:max_items]


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    return _bounded_text(str(value))


def _bounded_text(value: str) -> str:
    compact = sanitize_prompt_public_value(" ".join(value.split()))
    if _blocked_public_history_text(compact):
        return "[redacted]"
    if len(compact) <= _PUBLIC_HISTORY_TEXT_LIMIT:
        return compact
    return compact[: _PUBLIC_HISTORY_TEXT_LIMIT - 15] + "...[truncated]"


def _int_or_zero(value: Any) -> int:
    return int(value) if isinstance(value, int | float) else 0


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _blocked_public_history_key(key: str) -> bool:
    lowered = key.lower()
    return any(part in lowered for part in _PUBLIC_HISTORY_BLOCKED_KEY_PARTS)


def _blocked_public_history_text(text: str) -> bool:
    lowered = text.lower()
    return any(part in lowered for part in _PUBLIC_HISTORY_BLOCKED_TEXT_PARTS) or contains_forbidden_public_value_marker(
        lowered
    )


def _nonnegative_int(value: Any) -> int:
    if isinstance(value, int | float) and value > 0:
        return int(value)
    return 0


def _serialized_size(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8"))
