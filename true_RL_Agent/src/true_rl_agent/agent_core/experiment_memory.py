"""Bounded public memory of previously evaluated candidate mechanisms."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from true_rl_agent.agent_core.ledger import (
    PROVIDER_AUTHORING_MODE,
    dispatch_ledger_record_schema,
)
from true_rl_agent.agent_core.exploration_projection import (
    compact_exploration_prompt_projection,
    degrade_exploration_prompt_projection,
)
from true_rl_agent.agent_core.generation_contract import (
    GENERATION_PUBLIC_HISTORY_PROJECTION_SCHEMA_VERSION,
)
from true_rl_agent.agent_core.public_safety import sanitize_prompt_public_value
from true_rl_agent.candidates.authoring_policy import (
    CANDIDATE_MECHANISM_DELTA_SCHEMA_VERSION,
    MAX_CANONICAL_COMPONENT_DELTAS,
    canonical_mechanism_runtime_identity,
    require_canonical_mechanism_delta,
    require_component_name,
)
from true_rl_agent.candidates.mechanism import MECHANISM_ATTRIBUTION_SCHEMA_VERSION
from true_rl_agent.prompt_serialization import canonical_prompt_json_bytes

EXPERIMENT_MEMORY_MAX_ROUNDS = 16
EXPERIMENT_MEMORY_MAX_SERIALIZED_BYTES = 14 * 1024
EXPERIMENT_MEMORY_MAX_COMPONENTS = 8
EXPERIMENT_MEMORY_MAX_CANDIDATE_COMPONENTS = MAX_CANONICAL_COMPONENT_DELTAS
_MAX_FAILED_GATES = 8
_TEXT_LIMIT = 500
_EXPERIMENT_MEMORY_SCHEMA_V3 = "true_rl_agent.experiment_memory.v3"
_EXPERIMENT_MEMORY_SCHEMA_V4 = "true_rl_agent.experiment_memory.v4"


def build_experiment_memory(
    records: Sequence[Mapping[str, Any]],
    *,
    exploration_projection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Project validated public records into a compact cross-round index."""

    entries: list[dict[str, Any]] = []
    accepted_component_ids: frozenset[tuple[str, str]] | None = None
    previous_component_ids: frozenset[tuple[str, str]] | None = None
    for index, record in enumerate(records, start=1):
        runtime_components = _runtime_component_records(record)
        component_ids = _component_identities(runtime_components)
        comparison_ids = (
            accepted_component_ids
            if accepted_component_ids is not None
            else previous_component_ids
        )
        candidate_components = _authoring_candidate_components(
            record,
            runtime_components=runtime_components,
        )
        if candidate_components is None:
            candidate_components = (
                [
                    component
                    for component in runtime_components
                    if _component_identity(component) not in comparison_ids
                ]
                if comparison_ids is not None
                else []
            )
        entries.append(
            _record_entry(
                record,
                round_index=index,
                candidate_components=candidate_components,
            )
        )
        decision = _mapping(record.get("decision"))
        if component_ids:
            previous_component_ids = component_ids
            if decision.get("accepted") is True:
                accepted_component_ids = component_ids
    return _bounded_memory(
        entries,
        exploration_projection=exploration_projection,
    )


def recompact_experiment_memory(value: Any) -> dict[str, Any]:
    """Re-apply fixed limits and allowlists to a supplied public projection."""

    memory = _mapping(value)
    raw_entries = memory.get("experiments")
    entries = list(raw_entries) if isinstance(raw_entries, list | tuple) else []
    exploration_projection = (
        memory.get("exploration")
        if memory.get("schema_version") == _EXPERIMENT_MEMORY_SCHEMA_V4
        and isinstance(memory.get("exploration"), Mapping)
        else None
    )
    return _bounded_memory(
        entries,
        prior_omitted_count=_nonnegative_int(memory.get("omitted_experiment_count")),
        exploration_projection=exploration_projection,
    )


def _bounded_memory(
    entries: Sequence[Any],
    *,
    prior_omitted_count: int = 0,
    exploration_projection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    exploration = (
        compact_exploration_prompt_projection(exploration_projection)
        if exploration_projection is not None
        else None
    )
    compact_entries = [
        compact
        for entry in entries
        if isinstance(entry, Mapping)
        for compact in [_compact_entry(entry)]
        if compact
    ]
    protected_indexes = _protected_entry_indexes(compact_entries)
    selected_indexes = list(
        range(
            max(0, len(compact_entries) - EXPERIMENT_MEMORY_MAX_ROUNDS),
            len(compact_entries),
        )
    )
    for index in sorted(protected_indexes):
        if index not in selected_indexes:
            selected_indexes.append(index)
    selected_indexes.sort()
    while len(selected_indexes) > EXPERIMENT_MEMORY_MAX_ROUNDS:
        drop_index = next(
            index
            for index in selected_indexes
            if index not in protected_indexes
        )
        selected_indexes.remove(drop_index)

    shown = [(index, compact_entries[index]) for index in selected_indexes]
    omitted_count = prior_omitted_count + len(compact_entries) - len(shown)
    memory = _memory_payload(
        [entry for _, entry in shown],
        omitted_count=omitted_count,
        exploration_projection=exploration,
    )
    while (
        canonical_prompt_json_bytes(memory) > EXPERIMENT_MEMORY_MAX_SERIALIZED_BYTES
    ):
        summary_position = next(
            (
                position
                for position, (index, entry) in enumerate(shown)
                if index not in protected_indexes
                and _retained_entry_summary(entry) != entry
            ),
            None,
        )
        if summary_position is not None:
            index, entry = shown[summary_position]
            shown[summary_position] = (index, _retained_entry_summary(entry))
        else:
            drop_position = next(
                (
                    position
                    for position, (index, _) in enumerate(shown)
                    if index not in protected_indexes
                ),
                None,
            )
            protected_summary_position = next(
                (
                    position
                    for position, (index, entry) in enumerate(shown)
                    if index in protected_indexes
                    and _retained_entry_summary(entry) != entry
                ),
                None,
            )
            if drop_position is not None:
                shown.pop(drop_position)
                omitted_count += 1
            elif protected_summary_position is not None:
                index, entry = shown[protected_summary_position]
                shown[protected_summary_position] = (
                    index,
                    _retained_entry_summary(entry),
                )
            elif exploration is not None:
                exploration = degrade_exploration_prompt_projection(exploration)
            else:
                break
        memory = _memory_payload(
            [entry for _, entry in shown],
            omitted_count=omitted_count,
            exploration_projection=exploration,
        )
    if (
        canonical_prompt_json_bytes(memory)
        > EXPERIMENT_MEMORY_MAX_SERIALIZED_BYTES
    ):
        raise ValueError("protected experiment-memory summaries exceed the byte limit")
    return memory


def _protected_entry_indexes(
    entries: Sequence[Mapping[str, Any]],
) -> frozenset[int]:
    if not entries:
        return frozenset()
    protected = {len(entries) - 1}
    accepted_index = next(
        (
            index
            for index in range(len(entries) - 1, -1, -1)
            if entries[index].get("accepted") is True
        ),
        None,
    )
    if accepted_index is not None:
        protected.add(accepted_index)
    return frozenset(protected)


def _retained_entry_summary(entry: Mapping[str, Any]) -> dict[str, Any]:
    return _sanitize_mapping(
        {
            key: entry[key]
            for key in (
                "round_index",
                "candidate_id",
                "accepted",
                "objective_score_source",
                "objective_score",
                "objective_current_best_score",
                "objective_delta",
                "guard_score_source",
                "guard_score",
                "guard_current_best_score",
                "guard_delta",
                "reason",
                "candidate_components",
            )
            if key in entry
        }
    )


def _memory_payload(
    shown: Sequence[Mapping[str, Any]],
    *,
    omitted_count: int,
    exploration_projection: Mapping[str, Any] | None,
) -> dict[str, Any]:
    payload = {
            "schema_version": (
                _EXPERIMENT_MEMORY_SCHEMA_V4
                if exploration_projection is not None
                else _EXPERIMENT_MEMORY_SCHEMA_V3
            ),
            "purpose": (
                "bounded cross-round component/outcome index; consult it before proposing "
                "a candidate so previously evaluated mechanisms are not repeated blindly"
            ),
            "limits": {
                "max_rounds": EXPERIMENT_MEMORY_MAX_ROUNDS,
                "max_serialized_bytes": EXPERIMENT_MEMORY_MAX_SERIALIZED_BYTES,
                "max_components_per_round": EXPERIMENT_MEMORY_MAX_COMPONENTS,
                "max_candidate_components_per_round": (
                    EXPERIMENT_MEMORY_MAX_CANDIDATE_COMPONENTS
                ),
            },
            "candidate_component_semantics": (
                "mechanism-delta-bound provider records use the sealed canonical delta "
                "plus runner activation; all other records use runtime identity difference"
            ),
            "component_value_semantics": {
                "weight": "runner-observed effective component weight",
                "last_mean": (
                    "last sampled weighted mean contribution; one snapshot is not "
                    "evidence of inactivity or causal effect"
                ),
                "max_abs": (
                    "maximum absolute weighted contribution over bounded training samples"
                ),
            },
            "score_semantics": (
                "objective and guard values are copied from decision.comparison with "
                "their authoritative score-source identities; no score is recomputed"
            ),
            "omitted_experiment_count": omitted_count,
            "experiments": list(shown),
        }
    if exploration_projection is not None:
        payload["exploration_semantics"] = (
            "ledger-derived, task-owned public hypotheses for bounded revisit; "
            "no acceptance authority and no checkpoint/source reuse authority"
        )
        payload["exploration"] = dict(exploration_projection)
    return _sanitize_mapping(payload)


def _record_entry(
    record: Mapping[str, Any],
    *,
    round_index: int,
    candidate_components: Sequence[Mapping[str, Any]] = (),
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
            "reason": decision.get("reason"),
            "objective_score_source": comparison.get("score_source"),
            "objective_score": comparison.get("candidate_score"),
            "objective_current_best_score": comparison.get(
                "current_best_score"
            ),
            "objective_delta": comparison.get("delta"),
            "guard_score_source": comparison.get("guard_score_source"),
            "guard_score": comparison.get("candidate_guard_score"),
            "guard_current_best_score": comparison.get(
                "current_best_guard_score"
            ),
            "guard_delta": comparison.get("guard_delta"),
            "failed_gates": _failed_gate_names(decision.get("gate_statuses")),
            "activated_components": runtime.get("component_activation_records"),
            "candidate_components": candidate_components,
        }
    )


def _compact_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    round_index = _optional_int(entry.get("round_index"))
    if round_index is not None and round_index > 0:
        compact["round_index"] = round_index
    for key in (
        "candidate_id",
        "surface",
        "reason",
        "objective_score_source",
        "guard_score_source",
    ):
        text = _optional_text(entry.get(key))
        if text is not None:
            compact[key] = text
    if isinstance(entry.get("accepted"), bool):
        compact["accepted"] = entry["accepted"]
    for key in (
        "objective_score",
        "objective_current_best_score",
        "objective_delta",
        "guard_score",
        "guard_current_best_score",
        "guard_delta",
    ):
        numeric = _optional_number(entry.get(key))
        if numeric is not None:
            compact[key] = numeric
    failed_gates = _failed_gate_names(entry.get("failed_gates"))
    if failed_gates:
        compact["failed_gates"] = failed_gates
    raw_candidate_components = _compact_candidate_component_records(
        entry.get("candidate_components"),
        max_components=EXPERIMENT_MEMORY_MAX_CANDIDATE_COMPONENTS,
    )
    authoritative_delta = any(
        _optional_text(component.get("mechanism")) is not None
        for component in raw_candidate_components
    )
    if authoritative_delta:
        candidate_components = raw_candidate_components
        candidate_ids = _component_identities(candidate_components)
        components = [
            component
            for component in compact_component_activation_records(
                entry.get("activated_components"),
                max_components=(
                    EXPERIMENT_MEMORY_MAX_COMPONENTS + len(candidate_components)
                ),
            )
            if _component_identity(component) not in candidate_ids
        ][:EXPERIMENT_MEMORY_MAX_COMPONENTS]
    else:
        components = compact_component_activation_records(
            entry.get("activated_components")
        )
        activated_ids = _component_identities(components)
        candidate_components = [
            component
            for component in raw_candidate_components
            if _component_identity(component) not in activated_ids
        ]
    if candidate_components:
        compact["candidate_components"] = candidate_components
    if components:
        compact["activated_components"] = components
    return _sanitize_mapping(compact)


def _runtime_component_records(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    value = _mapping(record.get("runtime_activation_summary")).get(
        "component_activation_records"
    )
    if not isinstance(value, list | tuple):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _authoring_candidate_components(
    record: Mapping[str, Any],
    *,
    runtime_components: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]] | None:
    descriptor = dispatch_ledger_record_schema(record)
    if (
        not descriptor.mechanism_delta_bound
        or descriptor.authoring_mode != PROVIDER_AUTHORING_MODE
    ):
        return None
    generation = _mapping(record.get("generation"))
    if (
        descriptor.record_kind == "generation"
        and generation.get("schema_version")
        == GENERATION_PUBLIC_HISTORY_PROJECTION_SCHEMA_VERSION
        and "formal" in generation
        and generation["formal"] is None
    ):
        # A generation without a formal winner has no singular candidate round.
        # Its sibling mechanisms remain available through the generation-owned
        # exploration projection instead of being misattributed here.
        return []
    surface_classification = _mapping(record.get("candidate_surface")).get(
        "classification"
    )
    if surface_classification in {
        "provider_failure",
        "static_authoring_failure",
    }:
        return []
    attribution = _mapping(record.get("mechanism_attribution"))
    if attribution.get("schema_version") != MECHANISM_ATTRIBUTION_SCHEMA_VERSION:
        raise ValueError("current provider record has invalid mechanism attribution")
    delta = _mapping(attribution.get("authoring_mechanism_delta"))
    if not delta:
        return []
    if delta.get("schema_version") != CANDIDATE_MECHANISM_DELTA_SCHEMA_VERSION:
        raise ValueError("current provider record has invalid authoring mechanism delta")
    changed = require_canonical_mechanism_delta(delta.get("changed_mechanisms"))
    activation_by_id = {
        identity: component
        for component in compact_component_activation_records(
            runtime_components,
            max_components=max(1, len(runtime_components)),
        )
        if (identity := _component_identity(component)) is not None
    }
    projected: list[dict[str, Any]] = []
    for mechanism in changed:
        identity = canonical_mechanism_runtime_identity(mechanism)
        if identity is None:
            raise ValueError("current provider record has invalid mechanism identity")
        kind, name = identity
        component = {"mechanism": mechanism}
        activation = activation_by_id.get((kind, name))
        if activation is not None:
            component.update(
                {
                    key: value
                    for key, value in activation.items()
                    if key in {"weight", "last_mean", "max_abs"}
                }
            )
        projected.append(component)
    return projected


def _compact_candidate_component_records(
    value: Any,
    *,
    max_components: int,
) -> list[dict[str, Any]]:
    """Compact candidate identities, treating canonical mechanism as sole identity."""

    if not isinstance(value, list | tuple):
        return []
    canonical: list[dict[str, Any]] = []
    for raw in value:
        item = _mapping(raw)
        if "mechanism" not in item:
            continue
        mechanism = item.get("mechanism")
        try:
            normalized = require_canonical_mechanism_delta([mechanism])[0]
        except ValueError as exc:
            raise ValueError("candidate component mechanism is not canonical") from exc
        component: dict[str, Any] = {"mechanism": normalized}
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
        canonical.append(component)
    if canonical:
        if len(canonical) > max_components:
            raise ValueError("candidate component delta exceeds the canonical limit")
        return canonical
    return compact_component_activation_records(value, max_components=max_components)


def _component_identities(
    components: Sequence[Mapping[str, Any]],
) -> frozenset[tuple[str, str]]:
    return frozenset(
        identity
        for component in components
        if (identity := _component_identity(component)) is not None
    )


def _component_identity(component: Mapping[str, Any]) -> tuple[str, str] | None:
    if "mechanism" in component:
        return canonical_mechanism_runtime_identity(component.get("mechanism"))
    kind = _first_present(
        _optional_text(component.get("kind")),
        _optional_text(component.get("component_kind")),
    )
    name = _optional_text(component.get("name"))
    if kind is None or name is None:
        return None
    return kind, name


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


def compact_component_activation_records(
    value: Any,
    *,
    max_components: int = EXPERIMENT_MEMORY_MAX_COMPONENTS,
) -> list[dict[str, Any]]:
    """Return bounded, public component evidence ordered by runtime salience."""

    if (
        isinstance(max_components, bool)
        or not isinstance(max_components, int)
        or max_components <= 0
    ):
        raise ValueError("max_components must be a positive integer")
    if not isinstance(value, list | tuple):
        return []
    components: list[tuple[int, dict[str, Any]]] = []
    for index, raw in enumerate(value):
        item = _mapping(raw)
        if not item:
            continue
        component: dict[str, Any] = {}
        kind = _first_present(
            _optional_text(item.get("kind")),
            _optional_text(item.get("component_kind")),
        )
        try:
            name = require_component_name(item.get("name"), "runtime component name")
        except ValueError:
            continue
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
            components.append((index, component))
    # Preserve the most actionable bounded evidence instead of whichever
    # components happened to appear first in the runtime's source order.
    components.sort(key=_component_salience_key)
    return [
        component
        for _, component in components[:max_components]
    ]


def _component_salience_key(
    indexed_component: tuple[int, Mapping[str, Any]],
) -> tuple[float, float, int]:
    index, component = indexed_component
    max_abs = _optional_number(component.get("max_abs"))
    last_mean = _optional_number(component.get("last_mean"))
    return (
        -abs(float(max_abs)) if max_abs is not None else 0.0,
        -abs(float(last_mean)) if last_mean is not None else 0.0,
        index,
    )


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
