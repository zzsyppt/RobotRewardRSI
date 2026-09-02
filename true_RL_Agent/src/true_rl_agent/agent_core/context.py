"""Public context construction for next-round prompts."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from true_rl_agent.agent_core.experiment_memory import (
    build_experiment_memory,
    recompact_experiment_memory,
)
from true_rl_agent.agent_core.public_safety import (
    contains_forbidden_public_key_marker,
    contains_forbidden_public_value_marker,
    sanitize_prompt_public_value,
)
from true_rl_agent.agent_core.source_request_feedback import (
    project_source_request_round_feedback,
)
from true_rl_agent.evaluation.acceptance_protocol import AcceptanceProtocol
from true_rl_agent.evaluation.acceptance_replay import (
    ProjectedPublicHistory,
    project_public_history,
)
from true_rl_agent.evaluation.current_best import CurrentBestEvalStats
from true_rl_agent.prompt_serialization import canonical_prompt_json_bytes

PUBLIC_HISTORY_MAX_RECENT_ROUNDS = 8
PUBLIC_HISTORY_MAX_SERIALIZED_BYTES = 8192
_PUBLIC_HISTORY_MAX_LIST_ITEMS = 8
_PUBLIC_HISTORY_TEXT_LIMIT = 500
_PUBLIC_HISTORY_ENTRY_KEYS = (
    "round_id",
    "candidate_id",
    "outcome",
    "accepted",
    "reason",
    "surface_classification",
    "public_score",
    "public_score_source",
    "comparison",
    "changed_components",
    "dominant_failure",
    "sealed_validation_gate",
    "candidate_runtime_activated_in_training",
    "candidate_code_loaded_in_eval",
    "failed_gates",
    "runtime_activation",
    "mechanism_attribution",
    "runner_error",
    "provider_failure",
    "repair_feedback",
    "source_request_feedback",
)
_PUBLIC_HISTORY_BLOCKED_KEY_PARTS = (
    "seed_id",
    "seed_policy",
)
_PUBLIC_HISTORY_BLOCKED_TEXT_PARTS = (
    "seed_id",
    "seed_policy",
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
            if contains_forbidden_public_key_marker(key):
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


def sanitize_public_value_with_policy(
    value: Any,
    *,
    blocked_key_parts: Sequence[str] = (),
    blocked_text_parts: Sequence[str] = (),
) -> Any:
    """Apply common plus task-owned scrub policy to a prompt-public value."""

    return _sanitize_public_history_value(
        value,
        extra_key_parts=_normalized_block_parts(blocked_key_parts),
        extra_text_parts=_normalized_block_parts(blocked_text_parts),
    )


def sanitize_public_records_with_policy(
    ledger_records: Sequence[Mapping[str, Any]],
    *,
    blocked_key_parts: Sequence[str] = (),
    blocked_text_parts: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """Apply common plus task-owned scrub policy before prompt construction."""

    extra_key_parts = _normalized_block_parts(blocked_key_parts)
    extra_text_parts = _normalized_block_parts(blocked_text_parts)
    return [
        _sanitize_public_history_value(
            record,
            extra_key_parts=extra_key_parts,
            extra_text_parts=extra_text_parts,
        )
        for record in ledger_records
        if isinstance(record, Mapping)
    ]


def build_public_context(
    ledger_records: Sequence[Mapping[str, Any]],
    *,
    acceptance_protocol: AcceptanceProtocol | None = None,
    initial_current_best: CurrentBestEvalStats | None = None,
    expected_session_genesis_hash: str | None = None,
    history_blocked_key_parts: Sequence[str] = (),
    history_blocked_text_parts: Sequence[str] = (),
) -> dict[str, Any]:
    """Validate and sequentially project raw ledger rows into public context."""

    has_evaluated_history = any(
        isinstance(record.get("public_result"), Mapping)
        and bool(record.get("public_result"))
        for record in ledger_records
    )
    if has_evaluated_history and (
        acceptance_protocol is None
        or initial_current_best is None
        or expected_session_genesis_hash is None
    ):
        raise ValueError(
            "acceptance_protocol, initial_current_best, and session Genesis are required to build "
            "context from evaluated history"
        )
    history = project_public_history(
        ledger_records,
        initial_current_best=initial_current_best,
        expected_session_genesis_hash=expected_session_genesis_hash,
        acceptance_protocol=acceptance_protocol,
    )
    return build_public_context_from_projected(
        history,
        history_blocked_key_parts=history_blocked_key_parts,
        history_blocked_text_parts=history_blocked_text_parts,
    )


def build_public_context_from_projected(
    history: ProjectedPublicHistory,
    *,
    history_blocked_key_parts: Sequence[str] = (),
    history_blocked_text_parts: Sequence[str] = (),
    initial_anchor_public_score: float | None = None,
    exploration_projection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build public context from the shared sequential projection only."""

    if not isinstance(history, ProjectedPublicHistory):
        raise TypeError("history must be a ProjectedPublicHistory")
    sanitized_records = sanitize_public_records_with_policy(
        history.records,
        blocked_key_parts=history_blocked_key_parts,
        blocked_text_parts=history_blocked_text_parts,
    )
    decision_projections = [
        _mapping(record.get("decision")) for record in sanitized_records
    ]
    accepted_scores = [
        decision.get("public_score")
        for decision in decision_projections
        if decision.get("accepted") is True
    ]
    numeric_accepted_scores = [
        float(score)
        for score in accepted_scores
        if isinstance(score, int | float) and not isinstance(score, bool)
    ]
    last_decision = decision_projections[-1] if decision_projections else {}
    recent_outcomes, omitted_round_count = _bounded_recent_outcomes(
        sanitized_records,
        public_score_source=history.public_score_source,
    )
    experiment_memory = build_experiment_memory(
        sanitized_records,
        exploration_projection=exploration_projection,
    )
    parsed_initial_anchor_score = _optional_number(initial_anchor_public_score)
    if initial_anchor_public_score is not None and parsed_initial_anchor_score is None:
        raise ValueError("initial prompt anchor score must be finite")
    initial_anchor_score = (
        float(parsed_initial_anchor_score)
        if parsed_initial_anchor_score is not None
        else history.initial_current_best.score_for_comparison()
        if history.initial_current_best is not None
        and history.initial_current_best.mode == "accepted"
        else None
    )
    comparable_scores = list(numeric_accepted_scores)
    if initial_anchor_score is not None:
        comparable_scores.append(initial_anchor_score)
    context = {
        "public_only": True,
        "previous_rounds": len(sanitized_records),
        "best_public_score": max(comparable_scores) if comparable_scores else None,
        "public_score_source": history.public_score_source,
        "last_outcome": last_decision.get("outcome"),
        "pure_knob_streak": _pure_knob_streak(sanitized_records),
        "public_history_limits": {
            "max_recent_rounds": PUBLIC_HISTORY_MAX_RECENT_ROUNDS,
            "max_recent_outcomes_serialized_bytes": (
                PUBLIC_HISTORY_MAX_SERIALIZED_BYTES
            ),
        },
        "public_history_omitted_round_count": omitted_round_count,
        "recent_outcomes": recent_outcomes,
        "experiment_memory": experiment_memory,
    }
    return sanitize_prompt_public_context(context)


def build_public_history_section(public_context: Mapping[str, Any]) -> dict[str, Any]:
    recent_outcomes = public_context.get("recent_outcomes")
    if not isinstance(recent_outcomes, list):
        recent_outcomes = []
    recent, extra_omitted = _bounded_history_entries(recent_outcomes)
    omitted_count = _nonnegative_int(public_context.get("public_history_omitted_round_count")) + extra_omitted
    experiment_memory = recompact_experiment_memory(
        public_context.get("experiment_memory")
    )
    return sanitize_prompt_public_context(
        {
            "schema_version": "true_rl_agent.public_history.v4",
            "previous_rounds": public_context.get("previous_rounds", 0),
            "best_public_score": public_context.get("best_public_score"),
            "public_score_source": public_context.get("public_score_source"),
            "last_outcome": public_context.get("last_outcome"),
            "pure_knob_streak": public_context.get("pure_knob_streak", 0),
            "limits": {
                "max_recent_rounds": PUBLIC_HISTORY_MAX_RECENT_ROUNDS,
                "max_recent_outcomes_serialized_bytes": (
                    PUBLIC_HISTORY_MAX_SERIALIZED_BYTES
                ),
            },
            "omitted_round_count": omitted_count,
            "recent_outcomes": recent,
            "experiment_memory": experiment_memory,
        }
    )


def _sanitize_public_history_value(
    value: Any,
    *,
    extra_key_parts: tuple[str, ...],
    extra_text_parts: tuple[str, ...],
) -> Any:
    if isinstance(value, Mapping):
        clean: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if _blocked_public_history_key(key_text, extra_key_parts=extra_key_parts):
                continue
            clean[key_text] = _sanitize_public_history_value(
                item,
                extra_key_parts=extra_key_parts,
                extra_text_parts=extra_text_parts,
            )
        return clean
    if isinstance(value, list):
        return [
            _sanitize_public_history_value(
                item,
                extra_key_parts=extra_key_parts,
                extra_text_parts=extra_text_parts,
            )
            for item in value
        ]
    if isinstance(value, tuple):
        return [
            _sanitize_public_history_value(
                item,
                extra_key_parts=extra_key_parts,
                extra_text_parts=extra_text_parts,
            )
            for item in value
        ]
    if isinstance(value, str):
        compact = sanitize_prompt_public_value(value)
        if _blocked_public_history_text(compact, extra_text_parts=extra_text_parts):
            return "[redacted]"
        return compact
    return value


def _pure_knob_streak(ledger_records: list[dict[str, Any]]) -> int:
    streak = 0
    for record in reversed(ledger_records):
        if record.get("candidate_surface", {}).get("classification") != "parameter_only":
            break
        streak += 1
    return streak


def _bounded_recent_outcomes(
    ledger_records: list[dict[str, Any]],
    *,
    public_score_source: str | None = None,
) -> tuple[list[dict[str, Any]], int]:
    entries = [
        _round_history_entry(record, public_score_source=public_score_source)
        for record in ledger_records
    ]
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
    if key in {
        "round_id",
        "candidate_id",
        "outcome",
        "reason",
        "surface_classification",
    }:
        return _optional_text(value)
    if key == "accepted":
        return value if isinstance(value, bool) else None
    if key == "public_score":
        return float(value) if isinstance(value, int | float) else None
    if key == "public_score_source":
        return _optional_text(value)
    if key == "comparison":
        return _comparison_public_summary(value)
    if key == "changed_components":
        return sorted(_list_of_strings(value))
    if key == "dominant_failure":
        return _dominant_failure_public_summary(value)
    if key == "sealed_validation_gate":
        return _sealed_validation_gate_public_summary(value)
    if key in {
        "candidate_runtime_activated_in_training",
        "candidate_code_loaded_in_eval",
    }:
        return value if isinstance(value, bool) else _optional_text(value)
    if key == "failed_gates":
        return _failed_gates_public_summary(value) or None
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
    if key == "source_request_feedback":
        return project_source_request_round_feedback(value)
    return _compact_value(value)


def _round_history_entry(
    record: Mapping[str, Any],
    *,
    public_score_source: str | None = None,
) -> dict[str, Any]:
    decision = _mapping(record.get("decision"))
    candidate_surface = _mapping(record.get("candidate_surface"))
    public_result = _mapping(record.get("public_result"))
    public_metrics = _mapping(public_result.get("metrics"))
    prompt_feedback = _mapping(public_metrics.get("prompt_feedback"))
    public_score = _public_score(decision)
    entry: dict[str, Any] = {
        "round_id": _optional_text(record.get("round_id")),
        "candidate_id": _optional_text(_mapping(record.get("candidate")).get("candidate_id")),
        "outcome": _optional_text(decision.get("outcome")),
        "accepted": decision.get("accepted") if isinstance(decision.get("accepted"), bool) else None,
        "reason": _optional_text(decision.get("reason")),
        "surface_classification": _first_present(
            _optional_text(decision.get("surface_classification")),
            _optional_text(candidate_surface.get("classification")),
        ),
        "public_score": public_score,
        "public_score_source": (
            _optional_text(public_score_source)
            if public_score is not None
            else None
        ),
        "comparison": _comparison_public_summary(
            decision.get("comparison"),
            public_score_source=public_score_source,
        ),
        "changed_components": _changed_components(record),
        "dominant_failure": _dominant_failure_public_summary(prompt_feedback.get("dominant_failure")),
        "sealed_validation_gate": _sealed_validation_gate_public_summary(
            public_metrics.get("sealed_validation_gate")
        ),
        "candidate_runtime_activated_in_training": (
            _candidate_runtime_activated_in_training(record)
        ),
        "candidate_code_loaded_in_eval": _candidate_code_loaded_in_eval(record),
        "failed_gates": _failed_gate_summary(decision.get("gate_statuses")),
        "runtime_activation": _runtime_activation_summary(record),
        "mechanism_attribution": _mechanism_attribution(record),
        "source_request_feedback": project_source_request_round_feedback(
            record.get("source_request_feedback")
        ),
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


def _public_score(decision: Mapping[str, Any]) -> float | None:
    value = decision.get("public_score")
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _changed_components(record: Mapping[str, Any]) -> list[str]:
    candidate = _mapping(record.get("candidate"))
    raw = candidate.get("changed_components")
    if not isinstance(raw, list | tuple):
        verifier = _mapping(record.get("verifier"))
        details = _mapping(verifier.get("details"))
        raw = details.get("derived_change_set")
    return sorted(_list_of_strings(raw))


def _candidate_runtime_activated_in_training(
    record: Mapping[str, Any],
) -> bool | str:
    train_summary = _train_summary(record)
    candidate_summary = _mapping(train_summary.get("candidate"))
    declared = _first_present(
        train_summary.get("candidate_package_loaded"),
        train_summary.get("candidate_runtime_enabled"),
        candidate_summary.get("enabled"),
    )
    activation_gate = _mapping(record.get("runtime_activation_gate"))
    gate_proves_activation = activation_gate.get("status") == "pass"
    if isinstance(declared, bool):
        if declared is False and gate_proves_activation:
            return "inconsistent_evidence"
        return declared
    if gate_proves_activation:
        return True

    runtime = _mapping(record.get("runtime_activation_summary"))
    for key in ("reward_term_calls", "diagnostic_calls"):
        value = runtime.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return True
    for key in (
        "component_activation_records",
        "feature_activation_records",
        "capability_activation_records",
    ):
        value = runtime.get(key)
        if isinstance(value, list | tuple) and value:
            return True
    return "unknown"


def _candidate_code_loaded_in_eval(record: Mapping[str, Any]) -> bool | str:
    metrics = _public_metrics(record)
    train_summary = _train_summary(record)
    value = _first_present(
        metrics.get("candidate_code_loaded_in_eval"),
        train_summary.get("candidate_code_loaded_in_eval"),
    )
    return value if isinstance(value, bool) else "unknown"


def _failed_gate_summary(value: Any) -> list[dict[str, str]]:
    statuses = _mapping(value)
    failed: list[dict[str, str]] = []
    for raw_name, raw_status in statuses.items():
        name = _optional_text(raw_name)
        status = _optional_text(raw_status)
        if not name or not status or status == "pass":
            continue
        failed.append({"name": name, "status": status})
        if len(failed) >= _PUBLIC_HISTORY_MAX_LIST_ITEMS:
            break
    return failed


def _failed_gates_public_summary(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list | tuple):
        return []
    failed: list[dict[str, str]] = []
    for item in value:
        gate = _mapping(item)
        name = _optional_text(gate.get("name"))
        status = _optional_text(gate.get("status"))
        if not name or not status:
            continue
        failed.append({"name": name, "status": status})
        if len(failed) >= _PUBLIC_HISTORY_MAX_LIST_ITEMS:
            break
    return failed


def _comparison_public_summary(
    value: Any,
    *,
    public_score_source: str | None = None,
) -> dict[str, Any] | None:
    comparison = _mapping(value)
    if not comparison:
        return None
    summary: dict[str, Any] = {}
    for key in ("mode", "score_source", "public_score_source"):
        text = _optional_text(comparison.get(key))
        if text is not None:
            summary[key] = text
    for key in ("candidate_score", "current_best_score", "delta", "threshold", "public_score"):
        numeric = _optional_number(comparison.get(key))
        if numeric is not None:
            summary[key] = numeric
    if "public_score" in summary:
        source = _optional_text(public_score_source)
        if source is not None:
            summary["public_score_source"] = source
    if isinstance(comparison.get("delta_met"), bool):
        summary["delta_met"] = comparison["delta_met"]
    candidate_stats = _score_stats_public_summary(comparison.get("candidate_score_stats"))
    if candidate_stats:
        summary["candidate_score_stats"] = candidate_stats
    best_stats = _score_stats_public_summary(comparison.get("best_score_stats"))
    if best_stats:
        summary["best_score_stats"] = best_stats
    current_best = _current_best_public_summary(comparison.get("current_best"))
    if current_best:
        summary["current_best"] = current_best
    return summary or None


def _score_stats_public_summary(value: Any) -> dict[str, Any]:
    stats = _mapping(value)
    if not stats:
        return {}
    summary: dict[str, Any] = {}
    for key in ("count", "episode_count"):
        count = _optional_int(stats.get(key))
        if count is not None:
            summary[key] = count
    for key in ("mean", "std", "standard_error"):
        numeric = _optional_number(stats.get(key))
        if numeric is not None:
            summary[key] = numeric
    return summary


def _current_best_public_summary(value: Any) -> dict[str, Any]:
    current_best = _mapping(value)
    if not current_best:
        return {}
    summary: dict[str, Any] = {}
    for key in ("mode", "accepted_round_id", "accepted_candidate_id"):
        text = _optional_text(current_best.get(key))
        if text is not None:
            summary[key] = text
    return summary


def _dominant_failure_public_summary(value: Any) -> dict[str, Any] | None:
    failure = _mapping(value)
    if not failure:
        return None
    summary: dict[str, Any] = {}
    for key in ("status", "source_split", "label", "reason"):
        text = _optional_text(failure.get(key))
        if text is not None:
            summary[key] = text
    rate = _optional_number(failure.get("rate"))
    if rate is not None:
        summary["rate"] = rate
    return summary or None


def _sealed_validation_gate_public_summary(value: Any) -> dict[str, Any] | None:
    gate = _mapping(value)
    if not gate:
        return None
    summary: dict[str, Any] = {}
    for key in (
        "status",
        "reason",
        "checkpoint_safety_status",
        "candidate_isolation_status",
        "protocol_match_status",
        "checkpoint_hash_match_status",
        "eval_protocol_hash_match_status",
    ):
        text = _optional_text(gate.get(key))
        if text is not None:
            summary[key] = text
    return summary or None


def _runtime_activation_summary(record: Mapping[str, Any]) -> dict[str, Any]:
    runtime = _mapping(record.get("runtime_activation_summary"))
    if not runtime:
        return {"status": "not_available"}
    return _runtime_activation_public_summary(runtime)


def _runtime_activation_public_summary(value: Any) -> dict[str, Any]:
    runtime = _mapping(value)
    if not runtime:
        return {"status": "not_available"}
    if runtime.get("status") == "not_available":
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
    static_failure = _mapping(record.get("static_authoring_failure"))
    feedback = _mapping(static_failure.get("feedback"))
    if not feedback:
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
    prior_omitted = value.get("omitted_detail_count")
    if (
        isinstance(prior_omitted, bool)
        or not isinstance(prior_omitted, int)
        or prior_omitted < 0
    ):
        prior_omitted = 0
    for key in _PUBLIC_HISTORY_DETAIL_KEYS:
        if key in value:
            compact[key] = _compact_value(value[key])
    omitted = prior_omitted + sum(
        1
        for key in value
        if str(key) not in compact and str(key) != "omitted_detail_count"
    )
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


def _optional_number(value: Any) -> float | int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return None


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


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


def _normalized_block_parts(parts: Sequence[str]) -> tuple[str, ...]:
    return tuple(str(part).lower() for part in parts if isinstance(part, str) and part)


def _blocked_public_history_key(key: str, *, extra_key_parts: Sequence[str] = ()) -> bool:
    lowered = key.lower()
    return contains_forbidden_public_key_marker(key) or any(
        part in lowered
        for part in (*_PUBLIC_HISTORY_BLOCKED_KEY_PARTS, *extra_key_parts)
    )


def _blocked_public_history_text(text: str, *, extra_text_parts: Sequence[str] = ()) -> bool:
    lowered = text.lower()
    return any(
        part in lowered for part in (*_PUBLIC_HISTORY_BLOCKED_TEXT_PARTS, *extra_text_parts)
    ) or contains_forbidden_public_value_marker(lowered)


def _nonnegative_int(value: Any) -> int:
    if isinstance(value, int | float) and value > 0:
        return int(value)
    return 0


def _serialized_size(value: Any) -> int:
    return canonical_prompt_json_bytes(value)
