"""Bounded public memory of previously evaluated candidate mechanisms."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from true_rl_agent.agent_core.public_safety import sanitize_prompt_public_value
from true_rl_agent.prompt_serialization import canonical_prompt_json_bytes

EXPERIMENT_MEMORY_MAX_ROUNDS = 16
EXPERIMENT_MEMORY_MAX_SERIALIZED_BYTES = 14 * 1024
EXPERIMENT_MEMORY_MAX_COMPONENTS = 8
_MAX_FAILED_GATES = 8
_TEXT_LIMIT = 500


def build_experiment_memory(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Project validated public records into a compact cross-round index."""

    entries = [
        _record_entry(record, round_index=index)
        for index, record in enumerate(records, start=1)
    ]
    return _bounded_memory(entries)


def recompact_experiment_memory(value: Any) -> dict[str, Any]:
    """Re-apply fixed limits and allowlists to a supplied public projection."""

    memory = _mapping(value)
    raw_entries = memory.get("experiments")
    entries = list(raw_entries) if isinstance(raw_entries, list | tuple) else []
    return _bounded_memory(
        entries,
        prior_omitted_count=_nonnegative_int(memory.get("omitted_experiment_count")),
    )


def _bounded_memory(
    entries: Sequence[Any],
    *,
    prior_omitted_count: int = 0,
) -> dict[str, Any]:
    compact_entries = [
        compact
        for entry in entries
        if isinstance(entry, Mapping)
        for compact in [_compact_entry(entry)]
        if compact
    ]
    shown = compact_entries[-EXPERIMENT_MEMORY_MAX_ROUNDS:]
    omitted_count = prior_omitted_count + max(0, len(compact_entries) - len(shown))
    while shown and canonical_prompt_json_bytes(shown) > EXPERIMENT_MEMORY_MAX_SERIALIZED_BYTES:
        shown.pop(0)
        omitted_count += 1
    return _sanitize_mapping(
        {
            "schema_version": "true_rl_agent.experiment_memory.v1",
            "purpose": (
                "bounded cross-round component/outcome index; consult it before proposing "
                "a candidate so previously evaluated mechanisms are not repeated blindly"
            ),
            "limits": {
                "max_rounds": EXPERIMENT_MEMORY_MAX_ROUNDS,
                "max_serialized_bytes": EXPERIMENT_MEMORY_MAX_SERIALIZED_BYTES,
                "max_components_per_round": EXPERIMENT_MEMORY_MAX_COMPONENTS,
            },
            "component_value_semantics": {
                "weight": "runner-observed effective component weight",
                "last_mean": "last sampled weighted mean contribution",
                "max_abs": (
                    "maximum absolute weighted contribution over bounded training samples"
                ),
            },
            "omitted_experiment_count": omitted_count,
            "experiments": shown,
        }
    )


def _record_entry(
    record: Mapping[str, Any],
    *,
    round_index: int,
) -> dict[str, Any]:
    decision = _mapping(record.get("decision"))
    comparison = _mapping(decision.get("comparison"))
    runtime = _mapping(record.get("runtime_activation_summary"))
    return _compact_entry(
        {
            "round_index": round_index,
            "candidate_id": _mapping(record.get("candidate")).get("candidate_id"),
            "surface": _first_present(
                decision.get("surface_classification"),
                _mapping(record.get("candidate_surface")).get("classification"),
            ),
            "accepted": decision.get("accepted"),
            "score": decision.get("public_score"),
            "reason": decision.get("reason"),
            "delta": comparison.get("delta"),
            "failed_gates": _failed_gate_names(decision.get("gate_statuses")),
            "activated_components": runtime.get("component_activation_records"),
        }
    )


def _compact_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    round_index = _optional_int(entry.get("round_index"))
    if round_index is not None and round_index > 0:
        compact["round_index"] = round_index
    for key in ("candidate_id", "surface", "reason"):
        text = _optional_text(entry.get(key))
        if text is not None:
            compact[key] = text
    if isinstance(entry.get("accepted"), bool):
        compact["accepted"] = entry["accepted"]
    for key in ("score", "delta"):
        numeric = _optional_number(entry.get(key))
        if numeric is not None:
            compact[key] = numeric
    failed_gates = _failed_gate_names(entry.get("failed_gates"))
    if failed_gates:
        compact["failed_gates"] = failed_gates
    components = _compact_components(entry.get("activated_components"))
    if components:
        compact["activated_components"] = components
    return _sanitize_mapping(compact)


def _failed_gate_names(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        raw_items: Sequence[Any] = [
            name
            for name, status in value.items()
            if _optional_text(status) not in {None, "pass"}
        ]
    elif isinstance(value, list | tuple):
        raw_items = value
    else:
        return []
    failed: list[str] = []
    for raw in raw_items:
        name = (
            _optional_text(_mapping(raw).get("name"))
            if isinstance(raw, Mapping)
            else _optional_text(raw)
        )
        if name is not None:
            failed.append(name)
        if len(failed) >= _MAX_FAILED_GATES:
            break
    return failed


def _compact_components(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list | tuple):
        return []
    components: list[dict[str, Any]] = []
    for raw in value:
        if len(components) >= EXPERIMENT_MEMORY_MAX_COMPONENTS:
            break
        item = _mapping(raw)
        if not item:
            continue
        component: dict[str, Any] = {}
        kind = _first_present(
            _optional_text(item.get("kind")),
            _optional_text(item.get("component_kind")),
        )
        name = _optional_text(item.get("name"))
        if kind is not None:
            component["kind"] = kind
        if name is not None:
            component["name"] = name
        for public_key, source_keys in (
            ("weight", ("weight", "effective_weight")),
            ("last_mean", ("last_mean", "last_weighted_mean")),
            ("max_abs", ("max_abs", "max_abs_weighted_contribution")),
        ):
            numeric = _first_present(
                *(_optional_number(item.get(key)) for key in source_keys)
            )
            if numeric is not None:
                component[public_key] = numeric
        if component:
            components.append(component)
    return components


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text[:_TEXT_LIMIT] if text else None


def _optional_number(value: Any) -> float | int | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    parsed = float(value)
    if not math.isfinite(parsed):
        return None
    return value


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed


def _nonnegative_int(value: Any) -> int:
    parsed = _optional_int(value)
    return max(0, parsed) if parsed is not None else 0


def _first_present(*values: Any) -> Any:
    return next((value for value in values if value is not None), None)


def _sanitize_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    clean = sanitize_prompt_public_value(dict(value))
    return dict(clean) if isinstance(clean, Mapping) else {}
