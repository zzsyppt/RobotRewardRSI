"""Prompt-visible source catalog and history-best lineage projection."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from true_rl_agent.agent_core.provider_contract import SOURCE_REQUEST_SERVICE_ENABLED
from true_rl_agent.agent_core.public_safety import sanitize_prompt_public_value
from true_rl_agent.agent_core.source_request_limits import (
    BEST_OBSERVED_SOURCE_ID,
    CURRENT_BEST_SOURCE_ID,
    LATEST_FAILED_SOURCE_ID,
    PROMPT_SOURCE_LOGICAL_IDS,
)
from true_rl_agent.agent_core.source_request_limits import (
    CANDIDATE_COMPONENTS_SOURCE_ID as _CANDIDATE_COMPONENTS_SOURCE_ID,
)
from true_rl_agent.agent_core.source_request_limits import (
    KNOWN_GOOD_SOURCE_ID as _KNOWN_GOOD_SOURCE_ID,
)
from true_rl_agent.evaluation.acceptance_protocol import AcceptanceProtocol
from true_rl_agent.evaluation.acceptance_replay import (
    ProjectedPublicHistory,
    project_public_history,
)
from true_rl_agent.evaluation.binding import require_sha256_checksum
from true_rl_agent.evaluation.current_best import CurrentBestEvalStats
from true_rl_agent.prompting.public_safety import assert_public_only_prompt_payload

SOURCE_CATALOG_SCHEMA_VERSION = "true_rl_agent.source_catalog.v2"
CANDIDATE_LINEAGE_SCHEMA_VERSION = "true_rl_agent.candidate_lineage.v2"
SOURCE_PAYLOAD_SCHEMA_VERSION = "true_rl_agent.prompt_source_payload.v1"
EXPLORATION_DIRECTIVE_SCHEMA_VERSION = "true_rl_agent.exploration_directive.v2"
MAX_PROMPT_SOURCE_INLINE_BYTES = 24 * 1024
MAX_PROMPT_SOURCE_READ_BYTES = 256 * 1024
_COMPONENTS_RELATIVE_PATH = Path("training_code") / "components.py"
PROMPT_SOURCE_SECTION_KEYS = (
    "source_catalog",
    "candidate_lineage",
    "source_payloads",
    "exploration_directive",
)
_SOURCE_CATALOG_LOGICAL_IDS = PROMPT_SOURCE_LOGICAL_IDS


def build_prompt_source_context(
    records: Sequence[Any],
    *,
    acceptance_protocol: AcceptanceProtocol | None = None,
    initial_current_best: CurrentBestEvalStats | None = None,
    session_genesis_hash: str | None = None,
    known_good_examples: Any = None,
    latest_failed_candidate: Any = None,
    initial_reference_program_source: str | None = None,
) -> dict[str, Any]:
    """Validate raw history, then build source sections from one shared projection."""

    raw_public_records = [_public_record(record) for record in records]
    has_evaluated_history = any(
        isinstance(record.get("public_result"), Mapping)
        and bool(record.get("public_result"))
        for record in raw_public_records
    )
    if has_evaluated_history and (
        acceptance_protocol is None
        or initial_current_best is None
        or session_genesis_hash is None
    ):
        raise ValueError(
            "acceptance_protocol, initial_current_best, and session Genesis are required to build "
            "sources from evaluated history"
        )
    history = project_public_history(
        raw_public_records,
        initial_current_best=initial_current_best,
        expected_session_genesis_hash=session_genesis_hash,
        acceptance_protocol=acceptance_protocol,
    )
    return build_prompt_source_context_from_projected(
        records,
        history,
        known_good_examples=known_good_examples,
        latest_failed_candidate=latest_failed_candidate,
        initial_reference_program_source=initial_reference_program_source,
    )


def build_prompt_source_context_from_projected(
    records: Sequence[Any],
    history: ProjectedPublicHistory,
    *,
    known_good_examples: Any = None,
    latest_failed_candidate: Any = None,
    initial_reference_program_source: str | None = None,
) -> dict[str, Any]:
    """Build source sections without reinterpreting validated decisions."""

    if not isinstance(history, ProjectedPublicHistory):
        raise TypeError("history must be a ProjectedPublicHistory")
    canonical_public_records = history.records
    _assert_projected_history_alignment(records, canonical_public_records)
    accepted_index = _latest_accepted_index(canonical_public_records)
    accepted = records[accepted_index] if accepted_index is not None else None
    accepted_public = (
        canonical_public_records[accepted_index]
        if accepted_index is not None
        else None
    )
    baseline_anchor = _baseline_anchor_from_history(history)
    reference_anchor = _reference_anchor_from_history(
        history,
        initial_reference_program_source=initial_reference_program_source,
    )
    current_best = (
        _reference_anchor_source(reference_anchor)
        if accepted is None and reference_anchor is not None
        else _baseline_anchor_source(baseline_anchor)
        if accepted is None and baseline_anchor is not None
        else _resolve_current_best_source(accepted)
    )
    best_observed_index = (
        _best_observed_nonaccepted_index(canonical_public_records)
        if accepted_index is None
        else None
    )
    best_observed = _best_observed_experiment(
        records=records,
        canonical_public_records=canonical_public_records,
        index=best_observed_index,
    )
    if latest_failed_candidate is not None:
        latest_failed = _resolve_latest_failed_source(latest_failed_candidate)
        latest_failed_index = None
    else:
        latest_failed_index = _immediate_previous_nonaccepted_index(
            canonical_public_records
        )
        latest_failed = _resolve_latest_failed_record_source(
            records[latest_failed_index] if latest_failed_index is not None else None
        )
    best_observed_aliases_latest = (
        best_observed_index is not None
        and latest_failed_index is not None
        and (
            best_observed_index == latest_failed_index
            or _sources_share_content(
                _mapping(best_observed.get("source")), latest_failed
            )
        )
    )
    catalog = _source_catalog(
        current_best=current_best,
        best_observed=best_observed,
        best_observed_aliases_latest=best_observed_aliases_latest,
        latest_failed=latest_failed,
        known_good_examples=known_good_examples,
    )
    source_payloads = _source_payloads(
        current_best=current_best,
        best_observed=best_observed,
        best_observed_aliases_latest=best_observed_aliases_latest,
        latest_failed=latest_failed,
    )
    lineage = _candidate_lineage(
        accepted=accepted,
        accepted_public=accepted_public,
        current_best=current_best,
        baseline_anchor=baseline_anchor,
        reference_anchor=reference_anchor,
        best_observed=best_observed,
        best_observed_aliases_latest=best_observed_aliases_latest,
    )
    directive = _exploration_directive(lineage)
    sections = {
        "source_catalog": catalog,
        "candidate_lineage": lineage,
        "source_payloads": source_payloads,
        "exploration_directive": directive,
    }
    return validate_prompt_source_sections(sections)


def _assert_projected_history_alignment(
    records: Sequence[Any],
    projected_records: Sequence[Mapping[str, Any]],
) -> None:
    if len(records) != len(projected_records):
        raise ValueError("records and projected history must have equal length")
    for index, (record, projected) in enumerate(
        zip(records, projected_records, strict=True)
    ):
        raw_round_id = _public_record(record).get("round_id")
        projected_round_id = projected.get("round_id")
        if raw_round_id != projected_round_id:
            raise ValueError(
                "records and projected history round_id mismatch at "
                f"index {index}"
            )


def default_prompt_source_sections() -> dict[str, Any]:
    return build_prompt_source_context([])


def validate_prompt_source_sections(
    sections: Mapping[str, Any],
    *,
    require_inline_payloads: bool = True,
) -> dict[str, Any]:
    """Validate source catalog, payloads, lineage, and directive as one contract."""

    if not isinstance(sections, Mapping):
        raise TypeError("prompt source sections must be a mapping")
    unknown = sorted(set(sections) - set(PROMPT_SOURCE_SECTION_KEYS))
    missing = [key for key in PROMPT_SOURCE_SECTION_KEYS if key not in sections]
    if unknown:
        raise ValueError(f"unknown prompt source section keys: {unknown}")
    if missing:
        raise ValueError(f"missing prompt source section keys: {missing}")

    clean = sanitize_prompt_public_value(dict(sections))
    assert_public_only_prompt_payload(clean)
    catalog = _validate_source_catalog(clean["source_catalog"])
    payloads = _validate_source_payloads(
        clean["source_payloads"],
        catalog,
        require_inline_payloads=require_inline_payloads,
    )
    lineage = _validate_candidate_lineage(
        clean["candidate_lineage"],
        catalog,
        payloads,
        require_inline_payloads=require_inline_payloads,
    )
    _validate_exploration_directive(clean["exploration_directive"], lineage)
    assert_public_only_prompt_payload(clean)
    return clean


def prompt_source_catalog_logical_ids(prompt_payload: Mapping[str, Any]) -> frozenset[str]:
    """Return the closed source logical-id allowlist from a provider-facing prompt payload."""

    if not isinstance(prompt_payload, Mapping):
        raise TypeError("prompt payload must be a mapping")
    catalog = sanitize_prompt_public_value(prompt_payload.get("source_catalog"))
    assert_public_only_prompt_payload(catalog)
    return frozenset(_validate_source_catalog(catalog))


def is_known_prompt_source_logical_id(logical_id: Any) -> bool:
    """Return whether a value belongs to the closed prompt source catalog."""

    return isinstance(logical_id, str) and logical_id in _SOURCE_CATALOG_LOGICAL_IDS


def _validate_source_catalog(value: Any) -> dict[str, Mapping[str, Any]]:
    catalog = _require_mapping(value, "source_catalog")
    _require_schema(catalog, SOURCE_CATALOG_SCHEMA_VERSION, "source_catalog")
    policy = _require_mapping(catalog.get("policy"), "source_catalog.policy")
    if policy.get("ids_are_closed_allowlist") is not True:
        raise ValueError("source_catalog.policy.ids_are_closed_allowlist must be true")
    if policy.get("provider_filesystem_access") is not False:
        raise ValueError("source_catalog.policy.provider_filesystem_access must be false")
    if policy.get("source_request_enabled") is not SOURCE_REQUEST_SERVICE_ENABLED:
        raise ValueError(
            "source_catalog.policy.source_request_enabled must match source request service"
        )
    items = _require_list(catalog.get("items"), "source_catalog.items")
    by_id: dict[str, Mapping[str, Any]] = {}
    for index, raw_item in enumerate(items):
        item = _require_mapping(raw_item, f"source_catalog.items[{index}]")
        logical_id = _require_string(item.get("logical_id"), f"source_catalog.items[{index}].logical_id")
        if logical_id not in _SOURCE_CATALOG_LOGICAL_IDS:
            raise ValueError(f"source_catalog contains unknown logical_id: {logical_id}")
        if logical_id in by_id:
            raise ValueError(f"source_catalog contains duplicate logical_id: {logical_id}")
        if not isinstance(item.get("requestable"), bool):
            raise ValueError(f"source_catalog item requestable must be boolean: {logical_id}")
        if item.get("requestable") is True and not SOURCE_REQUEST_SERVICE_ENABLED:
            raise ValueError(f"source_catalog item must not be provider-requestable: {logical_id}")
        if _nonnegative_int(item.get("max_bytes")) is None:
            raise ValueError(f"source_catalog item has invalid max_bytes: {logical_id}")
        if item.get("content_sha256") is not None:
            _require_sha256(item.get("content_sha256"), f"source_catalog item {logical_id} content_sha256")
        if item.get("bytes") is not None and _nonnegative_int(item.get("bytes")) is None:
            raise ValueError(f"source_catalog item has invalid bytes: {logical_id}")
        if item.get("status") == "alias":
            if logical_id != BEST_OBSERVED_SOURCE_ID:
                raise ValueError("only best-observed source may use alias delivery")
            if item.get("alias_of") != LATEST_FAILED_SOURCE_ID:
                raise ValueError("best-observed alias must reference latest-failed source")
            if item.get("requestable") is not False or item.get("max_bytes") != 0:
                raise ValueError("best-observed alias must not be independently requestable")
        by_id[logical_id] = item
    if frozenset(by_id) != _SOURCE_CATALOG_LOGICAL_IDS:
        missing = sorted(_SOURCE_CATALOG_LOGICAL_IDS - frozenset(by_id))
        extra = sorted(frozenset(by_id) - _SOURCE_CATALOG_LOGICAL_IDS)
        raise ValueError(f"source_catalog logical_ids must match closed allowlist; missing={missing}; extra={extra}")
    observed_item = by_id[BEST_OBSERVED_SOURCE_ID]
    current_item = by_id[CURRENT_BEST_SOURCE_ID]
    if current_item.get("status") == "baseline_anchor":
        if current_item.get("role") != "baseline_policy_anchor":
            raise ValueError("baseline current-best catalog item has invalid role")
        if (
            current_item.get("requestable") is not False
            or current_item.get("inline_by_default") is not False
            or current_item.get("max_bytes") != 0
        ):
            raise ValueError("baseline current-best source must not be requestable")
        if current_item.get("content_sha256") is not None:
            raise ValueError("baseline current-best source must not declare source content")
    if current_item.get("role") == "reference_program_anchor":
        if current_item.get("source_owner") != "verified_session_genesis":
            raise ValueError("reference current-best source owner mismatch")
        if current_item.get("status") not in {
            "inline",
            "omitted_over_budget",
        }:
            raise ValueError("reference current-best source must be available")
    if observed_item.get("status") == "alias":
        latest_item = by_id[LATEST_FAILED_SOURCE_ID]
        if observed_item.get("content_sha256") != latest_item.get("content_sha256"):
            raise ValueError("best-observed alias hash must match latest-failed source")
        if observed_item.get("bytes") != latest_item.get("bytes"):
            raise ValueError("best-observed alias bytes must match latest-failed source")
    return by_id


def _validate_source_payloads(
    value: Any,
    catalog: Mapping[str, Mapping[str, Any]],
    *,
    require_inline_payloads: bool,
) -> dict[str, Mapping[str, Any]]:
    payloads = _require_list(value, "source_payloads")
    by_id: dict[str, Mapping[str, Any]] = {}
    for index, raw_payload in enumerate(payloads):
        payload = _require_mapping(raw_payload, f"source_payloads[{index}]")
        _require_schema(payload, SOURCE_PAYLOAD_SCHEMA_VERSION, f"source_payloads[{index}]")
        logical_id = _require_string(payload.get("logical_id"), f"source_payloads[{index}].logical_id")
        if logical_id not in catalog:
            raise ValueError(f"source_payload logical_id is not declared in source_catalog: {logical_id}")
        if logical_id in by_id:
            raise ValueError(f"duplicate source_payload logical_id: {logical_id}")
        if payload.get("delivery") != "inline":
            raise ValueError(f"source_payload delivery must be inline: {logical_id}")
        if payload.get("public_safe") is not True:
            raise ValueError(f"source_payload must declare public_safe true: {logical_id}")
        content = _require_string(payload.get("content"), f"source_payloads[{index}].content")
        encoded = content.encode("utf-8")
        payload_bytes = _nonnegative_int(payload.get("bytes"))
        if payload_bytes != len(encoded):
            raise ValueError(f"source_payload bytes mismatch: {logical_id}")
        if payload_bytes > MAX_PROMPT_SOURCE_INLINE_BYTES:
            raise ValueError(f"source_payload exceeds inline budget: {logical_id}")
        digest = hashlib.sha256(encoded).hexdigest()
        if payload.get("content_sha256") != digest:
            raise ValueError(f"source_payload content_sha256 mismatch: {logical_id}")
        catalog_item = catalog[logical_id]
        if catalog_item.get("status") != "inline" or catalog_item.get("inline_by_default") is not True:
            raise ValueError(f"source_payload is not backed by an inline catalog item: {logical_id}")
        if catalog_item.get("content_sha256") != digest:
            raise ValueError(f"source_payload hash differs from source_catalog: {logical_id}")
        if catalog_item.get("bytes") != payload_bytes:
            raise ValueError(f"source_payload bytes differ from source_catalog: {logical_id}")
        by_id[logical_id] = payload
    if require_inline_payloads:
        for logical_id, item in catalog.items():
            if item.get("status") == "inline" or item.get("inline_by_default") is True:
                if logical_id not in by_id:
                    raise ValueError(f"inline source_catalog item has no source_payload: {logical_id}")
    return by_id


def _validate_candidate_lineage(
    value: Any,
    catalog: Mapping[str, Mapping[str, Any]],
    payloads: Mapping[str, Mapping[str, Any]],
    *,
    require_inline_payloads: bool,
) -> Mapping[str, Any]:
    lineage = _require_mapping(value, "candidate_lineage")
    _require_schema(lineage, CANDIDATE_LINEAGE_SCHEMA_VERSION, "candidate_lineage")
    status = lineage.get("status")
    if status == "none":
        if lineage.get("current_best") is not None:
            raise ValueError("candidate_lineage.current_best must be null when status is none")
        if lineage.get("best_observed_experiment") is not None:
            raise ValueError(
                "candidate_lineage.best_observed_experiment must be null when status is none"
            )
        forbidden = {CURRENT_BEST_SOURCE_ID, BEST_OBSERVED_SOURCE_ID} & set(
            payloads
        )
        if forbidden:
            raise ValueError(
                "lineage source_payloads must be empty when candidate_lineage.status is none"
            )
        return lineage
    if status == "bootstrap_observed_anchor":
        if lineage.get("current_best") is not None:
            raise ValueError(
                "candidate_lineage.current_best must be null for a bootstrap anchor"
            )
        observed = _require_mapping(
            lineage.get("best_observed_experiment"),
            "candidate_lineage.best_observed_experiment",
        )
        if observed.get("accepted") is not False:
            raise ValueError("bootstrap anchor must explicitly declare accepted false")
        if observed.get("authority") != "authoring_anchor_only_not_policy_best":
            raise ValueError("bootstrap anchor has invalid authority")
        _require_string(observed.get("round_id"), "bootstrap anchor round_id")
        _require_string(observed.get("candidate_id"), "bootstrap anchor candidate_id")
        score = _public_score(observed)
        if score is None or not math.isfinite(score):
            raise ValueError("bootstrap anchor public_score must be finite")
        if observed.get("source_logical_id") not in {
            BEST_OBSERVED_SOURCE_ID,
            LATEST_FAILED_SOURCE_ID,
        }:
            raise ValueError("bootstrap anchor references an invalid source role")
        _validate_lineage_source(
            observed,
            catalog,
            payloads,
            require_inline_payloads=require_inline_payloads,
            field_name="candidate_lineage.best_observed_experiment",
        )
        if CURRENT_BEST_SOURCE_ID in payloads:
            raise ValueError("bootstrap lineage must not expose a current-best source_payload")
        return lineage
    if status == "baseline_current_best":
        current_best = _require_mapping(
            lineage.get("current_best"),
            "candidate_lineage.current_best",
        )
        if current_best.get("anchor_kind") != "baseline":
            raise ValueError("baseline lineage current-best anchor kind mismatch")
        _require_string(
            current_best.get("baseline_id"),
            "baseline lineage baseline_id",
        )
        score = _public_score(current_best)
        if score is None or not math.isfinite(score):
            raise ValueError("baseline lineage public_score must be finite")
        for field_name in (
            "task_protocol_hash",
            "eval_protocol_hash",
            "acceptance_protocol_hash",
            "evaluation_binding_hash",
        ):
            _require_sha256(
                current_best.get(field_name),
                f"baseline lineage {field_name}",
            )
        if current_best.get("source_delivery") != "baseline_anchor":
            raise ValueError("baseline lineage source delivery mismatch")
        _validate_lineage_source(
            current_best,
            catalog,
            payloads,
            require_inline_payloads=require_inline_payloads,
            field_name="candidate_lineage.current_best",
        )
        observed = lineage.get("best_observed_experiment")
        if observed is not None:
            _validate_observed_lineage_entry(
                _require_mapping(
                    observed,
                    "candidate_lineage.best_observed_experiment",
                ),
                catalog=catalog,
                payloads=payloads,
                require_inline_payloads=require_inline_payloads,
            )
        if CURRENT_BEST_SOURCE_ID in payloads:
            raise ValueError("baseline lineage must not expose current-best source payload")
        return lineage
    if status == "reference_current_best":
        current_best = _require_mapping(
            lineage.get("current_best"),
            "candidate_lineage.current_best",
        )
        if current_best.get("anchor_kind") != "reference":
            raise ValueError("reference lineage current-best anchor kind mismatch")
        _require_string(
            current_best.get("reference_id"),
            "reference lineage reference_id",
        )
        for field_name in (
            "subject_manifest_hash",
            "training_binding_hash",
            "capability_profile_hash",
            "task_protocol_hash",
            "eval_protocol_hash",
            "acceptance_protocol_hash",
            "evaluation_binding_hash",
        ):
            _require_sha256(
                current_best.get(field_name),
                f"reference lineage {field_name}",
            )
        require_sha256_checksum(
            current_best.get("program_source_checksum"),
            "reference lineage program_source_checksum",
        )
        score = _public_score(current_best)
        if score is None or not math.isfinite(score):
            raise ValueError("reference lineage public_score must be finite")
        _validate_lineage_source(
            current_best,
            catalog,
            payloads,
            require_inline_payloads=require_inline_payloads,
            field_name="candidate_lineage.current_best",
        )
        observed = lineage.get("best_observed_experiment")
        if observed is not None:
            _validate_observed_lineage_entry(
                _require_mapping(
                    observed,
                    "candidate_lineage.best_observed_experiment",
                ),
                catalog=catalog,
                payloads=payloads,
                require_inline_payloads=require_inline_payloads,
            )
        return lineage
    if status != "accepted_current_best":
        raise ValueError(f"unknown candidate_lineage.status: {status}")
    if lineage.get("best_observed_experiment") is not None:
        raise ValueError(
            "accepted lineage must not also expose a bootstrap best-observed anchor"
        )
    if BEST_OBSERVED_SOURCE_ID in payloads:
        raise ValueError(
            "accepted lineage must not expose a bootstrap best-observed source_payload"
        )
    current_best = _require_mapping(lineage.get("current_best"), "candidate_lineage.current_best")
    if current_best.get("source_logical_id") != CURRENT_BEST_SOURCE_ID:
        raise ValueError("accepted lineage must reference current-best source")
    _validate_lineage_source(
        current_best,
        catalog,
        payloads,
        require_inline_payloads=require_inline_payloads,
        field_name="candidate_lineage.current_best",
    )
    return lineage


def _validate_lineage_source(
    source: Mapping[str, Any],
    catalog: Mapping[str, Mapping[str, Any]],
    payloads: Mapping[str, Mapping[str, Any]],
    *,
    require_inline_payloads: bool,
    field_name: str,
) -> None:
    logical_id = _require_string(
        source.get("source_logical_id"), f"{field_name}.source_logical_id"
    )
    if logical_id not in catalog:
        raise ValueError(f"candidate_lineage references unknown source logical_id: {logical_id}")
    source_delivery = _require_string(
        source.get("source_delivery"), f"{field_name}.source_delivery"
    )
    catalog_item = catalog[logical_id]
    if source_delivery == "inline":
        if source.get("source_payload_ref") != logical_id:
            raise ValueError("inline candidate_lineage must reference its source_payload")
        if require_inline_payloads and logical_id not in payloads:
            raise ValueError(f"inline candidate_lineage source has no payload: {logical_id}")
        if catalog_item.get("status") != "inline":
            raise ValueError(f"inline candidate_lineage source is not inline in catalog: {logical_id}")
    elif source_delivery in {"omitted_over_budget", "unavailable"}:
        if logical_id in payloads:
            raise ValueError(f"non-inline candidate_lineage source must not have payload: {logical_id}")
        if source_delivery == "omitted_over_budget" and catalog_item.get("status") != "omitted_over_budget":
            raise ValueError(f"omitted candidate_lineage source is not omitted in catalog: {logical_id}")
    elif source_delivery == "baseline_anchor":
        if logical_id != CURRENT_BEST_SOURCE_ID:
            raise ValueError("only current-best may use baseline anchor delivery")
        if logical_id in payloads:
            raise ValueError("baseline anchor must not have a source payload")
        if catalog_item.get("status") != "baseline_anchor":
            raise ValueError("baseline lineage is not backed by baseline catalog state")
    else:
        raise ValueError(f"unsupported candidate_lineage source_delivery: {source_delivery}")
    source_sha = source.get("source_sha256")
    catalog_sha = catalog_item.get("content_sha256")
    if source_sha is not None:
        _require_sha256(source_sha, f"{field_name}.source_sha256")
    if catalog_sha is not None and source_sha != catalog_sha:
        raise ValueError(f"candidate_lineage source hash differs from source_catalog: {logical_id}")


def _validate_exploration_directive(value: Any, lineage: Mapping[str, Any]) -> None:
    directive = _require_mapping(value, "exploration_directive")
    _require_schema(directive, EXPLORATION_DIRECTIVE_SCHEMA_VERSION, "exploration_directive")
    expected_modes = {
        "none": "initial_mechanism_authoring",
        "bootstrap_observed_anchor": "bootstrap_refine",
        "accepted_current_best": "exploit_refine",
        "baseline_current_best": "baseline_anchored_authoring",
        "reference_current_best": "reference_anchored_authoring",
    }
    expected_mode = expected_modes.get(str(lineage.get("status")))
    if expected_mode is None:
        raise ValueError("exploration_directive has unknown candidate lineage status")
    if directive.get("mode") != expected_mode:
        raise ValueError(f"exploration_directive.mode must be {expected_mode}")


def _baseline_anchor_from_history(
    history: ProjectedPublicHistory,
) -> dict[str, Any] | None:
    current_best = history.initial_current_best
    if (
        current_best is None
        or current_best.mode != "accepted"
        or current_best.anchor_kind != "baseline"
    ):
        return None
    if not current_best.has_full_eval_stats():
        raise ValueError("projected baseline current-best is incomplete")
    return {
        "baseline_id": current_best.anchor_id,
        "public_score": current_best.score_for_comparison(),
        "task_protocol_hash": current_best.task_protocol_hash,
        "eval_protocol_hash": current_best.eval_protocol_hash,
        "acceptance_protocol_hash": current_best.acceptance_protocol_hash,
        "evaluation_binding_hash": current_best.evaluation_binding_hash,
    }


def _baseline_anchor_source(
    baseline_anchor: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "status": "baseline_anchor",
        "source_delivery": "baseline_anchor",
        "pending_reason": "baseline policy has no candidate components source",
        "baseline_anchor": dict(baseline_anchor),
    }


def _reference_anchor_from_history(
    history: ProjectedPublicHistory,
    *,
    initial_reference_program_source: str | None,
) -> dict[str, Any] | None:
    current_best = history.initial_current_best
    if (
        current_best is None
        or current_best.mode != "reference"
        or current_best.anchor_kind != "reference"
    ):
        if initial_reference_program_source is not None:
            raise ValueError(
                "reference program source requires a reference current-best"
            )
        return None
    if not current_best.has_full_eval_stats():
        raise ValueError("projected reference current-best is incomplete")
    if not isinstance(initial_reference_program_source, str) or not (
        initial_reference_program_source
    ):
        raise ValueError("reference current-best source is required")
    raw_source = initial_reference_program_source.encode("utf-8")
    checksum = "sha256:" + hashlib.sha256(raw_source).hexdigest()
    if checksum != current_best.program_source_checksum:
        raise ValueError("reference current-best source checksum mismatch")
    return {
        "reference_id": current_best.anchor_id,
        "public_score": current_best.score_for_comparison(),
        "subject_manifest_hash": current_best.subject_manifest_hash,
        "program_source_checksum": current_best.program_source_checksum,
        "training_binding_hash": current_best.training_binding_hash,
        "capability_profile_hash": current_best.capability_profile_hash,
        "task_protocol_hash": current_best.task_protocol_hash,
        "eval_protocol_hash": current_best.eval_protocol_hash,
        "acceptance_protocol_hash": current_best.acceptance_protocol_hash,
        "evaluation_binding_hash": current_best.evaluation_binding_hash,
        "source": _resolve_program_source_content(
            initial_reference_program_source
        ),
    }


def _reference_anchor_source(
    reference_anchor: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        **dict(_mapping(reference_anchor.get("source"))),
        "anchor_kind": "reference",
        "reference_anchor": {
            key: value
            for key, value in reference_anchor.items()
            if key != "source"
        },
        "payload_role": "reference_program_anchor",
    }


def _source_catalog(
    *,
    current_best: Mapping[str, Any],
    best_observed: Mapping[str, Any],
    best_observed_aliases_latest: bool,
    latest_failed: Mapping[str, Any],
    known_good_examples: Any,
) -> dict[str, Any]:
    current_item = _current_best_catalog_item(current_best)
    items = [
        {
            "logical_id": _CANDIDATE_COMPONENTS_SOURCE_ID,
            "role": "candidate_component_api_contract",
            "candidate_use": "Use the canonical component contract already included in required_candidate_package.component_contract.",
            "visibility": "prompt_visible_metadata",
            "inline_by_default": False,
            "requestable": False,
            "max_bytes": 0,
            "status": "available_via_required_candidate_package",
            "content_sha256": None,
            "pending_reason": "contract body is present in required_candidate_package.component_contract",
            "source_owner": "agent_core",
            "scrub_policy": "contract_projection_no_raw_module_source",
        },
        current_item,
        _best_observed_catalog_item(
            best_observed,
            aliases_latest=best_observed_aliases_latest,
        ),
        _known_good_catalog_item(known_good_examples),
        _latest_failed_catalog_item(latest_failed),
    ]
    catalog = {
        "schema_version": SOURCE_CATALOG_SCHEMA_VERSION,
        "policy": {
            "ids_are_closed_allowlist": True,
            "provider_filesystem_access": False,
            "source_request_enabled": SOURCE_REQUEST_SERVICE_ENABLED,
            "source_resolution_owner": "runner_prompt_source_resolver",
        },
        "items": items,
    }
    return catalog


def _latest_failed_catalog_item(latest_failed: Mapping[str, Any]) -> dict[str, Any]:
    item = {
        "logical_id": LATEST_FAILED_SOURCE_ID,
        "role": "latest_nonaccepted_candidate_source",
        "candidate_use": (
            "Use the latest nonaccepted source with its verifier, runtime, and eval "
            "feedback to repair, refute, or replace the attempted mechanism."
        ),
        "visibility": "prompt_visible_metadata",
        "inline_by_default": latest_failed.get("source_delivery") == "inline",
        "requestable": latest_failed.get("source_delivery") == "inline",
        "max_bytes": MAX_PROMPT_SOURCE_INLINE_BYTES,
        "status": latest_failed.get("status", "pending"),
        "content_sha256": latest_failed.get("content_sha256"),
        "bytes": latest_failed.get("bytes"),
        "hash_scope": latest_failed.get("hash_scope"),
        "pending_reason": latest_failed.get("pending_reason"),
        "source_owner": "latest_nonaccepted_candidate_attempt",
        "scrub_policy": "path_secret_and_public_only_guard",
    }
    return {key: value for key, value in item.items() if value is not None}


def _best_observed_catalog_item(
    best_observed: Mapping[str, Any],
    *,
    aliases_latest: bool,
) -> dict[str, Any]:
    source = _mapping(best_observed.get("source"))
    if aliases_latest:
        item = {
            "logical_id": BEST_OBSERVED_SOURCE_ID,
            "role": "best_observed_nonaccepted_candidate_source",
            "candidate_use": (
                "Bootstrap authoring anchor only; this is not an accepted policy best. "
                "The identical source is already delivered through the latest-failed slot."
            ),
            "visibility": "prompt_visible_metadata",
            "inline_by_default": False,
            "requestable": False,
            "max_bytes": 0,
            "status": "alias",
            "alias_of": LATEST_FAILED_SOURCE_ID,
            "content_sha256": source.get("content_sha256"),
            "bytes": source.get("bytes"),
            "pending_reason": source.get("pending_reason"),
            "source_owner": "validated_nonaccepted_experiment_history",
            "scrub_policy": "path_secret_and_public_only_guard",
        }
        return {key: value for key, value in item.items() if value is not None}
    item = {
        "logical_id": BEST_OBSERVED_SOURCE_ID,
        "role": "best_observed_nonaccepted_candidate_source",
        "candidate_use": (
            "Use only as the highest-scoring evaluated bootstrap authoring anchor; "
            "it is not an accepted policy best and grants no acceptance authority."
        ),
        "visibility": "prompt_visible_metadata",
        "inline_by_default": source.get("source_delivery") == "inline",
        "requestable": source.get("source_delivery") == "inline",
        "max_bytes": MAX_PROMPT_SOURCE_INLINE_BYTES,
        "status": source.get("status", "pending"),
        "content_sha256": source.get("content_sha256"),
        "bytes": source.get("bytes"),
        "hash_scope": source.get("hash_scope"),
        "pending_reason": source.get("pending_reason"),
        "source_owner": "validated_nonaccepted_experiment_history",
        "scrub_policy": "path_secret_and_public_only_guard",
    }
    if item["status"] == "pending" and item["pending_reason"] is None:
        item["pending_reason"] = "no_eligible_nonaccepted_bootstrap_anchor"
    return {key: value for key, value in item.items() if value is not None}


def _known_good_catalog_item(known_good_examples: Any) -> dict[str, Any]:
    has_examples = bool(known_good_examples)
    return {
        "logical_id": _KNOWN_GOOD_SOURCE_ID,
        "role": "known_good_examples_slot",
        "candidate_use": "Task provided examples are exposed through known_good_examples when available.",
        "visibility": "prompt_visible_metadata",
        "inline_by_default": False,
        "requestable": False,
        "max_bytes": 0,
        "status": "available_via_known_good_examples" if has_examples else "pending",
        "content_sha256": None,
        "pending_reason": None if has_examples else "task_prompt_adapter_provided_no_known_good_source",
        "source_owner": "task_prompt_adapter",
        "scrub_policy": "task_sections_public_sanitizer",
    }


def _current_best_catalog_item(current_best: Mapping[str, Any]) -> dict[str, Any]:
    if current_best.get("status") == "baseline_anchor":
        return {
            "logical_id": CURRENT_BEST_SOURCE_ID,
            "role": "baseline_policy_anchor",
            "candidate_use": (
                "Use the baseline score and protocol binding as the policy-best "
                "comparison anchor; author candidate code from public contracts because "
                "the baseline has no candidate-owned source."
            ),
            "visibility": "prompt_visible_metadata",
            "inline_by_default": False,
            "requestable": False,
            "max_bytes": 0,
            "status": "baseline_anchor",
            "content_sha256": None,
            "pending_reason": "baseline policy has no candidate components source",
            "source_owner": "verified_session_genesis",
            "scrub_policy": "no_baseline_checkpoint_or_private_artifact_projection",
        }
    is_reference = current_best.get("anchor_kind") == "reference"
    item = {
        "logical_id": CURRENT_BEST_SOURCE_ID,
        "role": (
            "reference_program_anchor"
            if is_reference
            else "latest_accepted_candidate_source"
        ),
        "candidate_use": (
            "Use the verified reference program and public failure evidence to author a complete replacement."
            if is_reference
            else "Inherit, rewrite, recombine, or refute the current best mechanism without copying it blindly."
        ),
        "visibility": "prompt_visible_metadata",
        "inline_by_default": current_best.get("source_delivery") == "inline",
        "requestable": current_best.get("source_delivery") == "inline",
        "max_bytes": MAX_PROMPT_SOURCE_INLINE_BYTES,
        "status": current_best.get("status", "pending"),
        "content_sha256": current_best.get("content_sha256"),
        "bytes": current_best.get("bytes"),
        "hash_scope": current_best.get("hash_scope"),
        "pending_reason": current_best.get("pending_reason"),
        "source_owner": (
            "verified_session_genesis"
            if is_reference
            else "accepted_candidate_lineage"
        ),
        "scrub_policy": "path_secret_and_public_only_guard",
    }
    return {key: value for key, value in item.items() if value is not None}


def _source_payloads(
    *,
    current_best: Mapping[str, Any],
    best_observed: Mapping[str, Any],
    best_observed_aliases_latest: bool,
    latest_failed: Mapping[str, Any],
) -> list[dict[str, Any]]:
    observed_source = _mapping(best_observed.get("source"))
    payloads = [
        _source_payload(
            logical_id=CURRENT_BEST_SOURCE_ID,
            role=str(
                current_best.get(
                    "payload_role",
                    "latest_accepted_candidate_source",
                )
            ),
            source=current_best,
        ),
        (
            None
            if best_observed_aliases_latest
            else _source_payload(
                logical_id=BEST_OBSERVED_SOURCE_ID,
                role="best_observed_nonaccepted_candidate_source",
                source=observed_source,
            )
        ),
        _source_payload(
            logical_id=LATEST_FAILED_SOURCE_ID,
            role="latest_nonaccepted_candidate_source",
            source=latest_failed,
        ),
    ]
    return [payload for payload in payloads if payload is not None]


def _source_payload(*, logical_id: str, role: str, source: Mapping[str, Any]) -> dict[str, Any] | None:
    content = source.get("content")
    if source.get("source_delivery") != "inline" or not isinstance(content, str):
        return None
    payload = {
        "schema_version": SOURCE_PAYLOAD_SCHEMA_VERSION,
        "logical_id": logical_id,
        "role": role,
        "delivery": "inline",
        "content": content,
        "content_sha256": source["content_sha256"],
        "bytes": source["bytes"],
        "summary": source["source_summary"],
        "public_safe": True,
    }
    return payload


def _candidate_lineage(
    *,
    accepted: Any,
    accepted_public: Mapping[str, Any] | None,
    current_best: Mapping[str, Any],
    baseline_anchor: Mapping[str, Any] | None,
    reference_anchor: Mapping[str, Any] | None,
    best_observed: Mapping[str, Any],
    best_observed_aliases_latest: bool,
) -> dict[str, Any]:
    if accepted is None and reference_anchor is not None:
        observed = _observed_lineage_entry(
            best_observed,
            aliases_latest=best_observed_aliases_latest,
        )
        source_delivery = current_best.get(
            "source_delivery",
            "unavailable",
        )
        current = {
            "anchor_kind": "reference",
            "reference_id": reference_anchor["reference_id"],
            "public_score": reference_anchor["public_score"],
            "subject_manifest_hash": reference_anchor[
                "subject_manifest_hash"
            ],
            "program_source_checksum": reference_anchor[
                "program_source_checksum"
            ],
            "training_binding_hash": reference_anchor[
                "training_binding_hash"
            ],
            "capability_profile_hash": reference_anchor[
                "capability_profile_hash"
            ],
            "task_protocol_hash": reference_anchor["task_protocol_hash"],
            "eval_protocol_hash": reference_anchor["eval_protocol_hash"],
            "acceptance_protocol_hash": reference_anchor[
                "acceptance_protocol_hash"
            ],
            "evaluation_binding_hash": reference_anchor[
                "evaluation_binding_hash"
            ],
            "source_logical_id": CURRENT_BEST_SOURCE_ID,
            "source_sha256": current_best.get("content_sha256"),
            "source_hash_scope": current_best.get("hash_scope"),
            "source_summary": current_best.get("source_summary"),
            "source_delivery": source_delivery,
        }
        if source_delivery == "inline":
            current["source_payload_ref"] = CURRENT_BEST_SOURCE_ID
        elif source_delivery == "omitted_over_budget":
            current["rewrite_directive"] = (
                "Use the verified source hash, summary, and public evidence; "
                "do not assume omitted source details."
            )
        return {
            "schema_version": CANDIDATE_LINEAGE_SCHEMA_VERSION,
            "status": "reference_current_best",
            "current_best": current,
            "best_observed_experiment": observed,
            "guidance": [
                "The verified reference program is the initial policy-best and source authority.",
                "Author a complete replacement from the reference source and public failure evidence.",
                "Only an accepted candidate can replace the reference current-best.",
            ],
        }
    if accepted is None and baseline_anchor is not None:
        observed = _observed_lineage_entry(
            best_observed,
            aliases_latest=best_observed_aliases_latest,
        )
        return {
            "schema_version": CANDIDATE_LINEAGE_SCHEMA_VERSION,
            "status": "baseline_current_best",
            "current_best": {
                "anchor_kind": "baseline",
                "baseline_id": baseline_anchor["baseline_id"],
                "public_score": baseline_anchor["public_score"],
                "task_protocol_hash": baseline_anchor["task_protocol_hash"],
                "eval_protocol_hash": baseline_anchor["eval_protocol_hash"],
                "acceptance_protocol_hash": baseline_anchor[
                    "acceptance_protocol_hash"
                ],
                "evaluation_binding_hash": baseline_anchor[
                    "evaluation_binding_hash"
                ],
                "source_logical_id": CURRENT_BEST_SOURCE_ID,
                "source_summary": (
                    "Verified baseline policy anchor; no candidate-owned source exists."
                ),
                "source_delivery": "baseline_anchor",
            },
            "best_observed_experiment": observed,
            "guidance": [
                "The baseline is the policy-best comparison authority but has no candidate source to inherit.",
                "Use public task contracts and evidence to author a complete mechanism.",
                "A best observed rejected candidate is authoring evidence only and cannot replace the baseline anchor.",
            ],
        }
    if accepted is None and not best_observed:
        return {
            "schema_version": CANDIDATE_LINEAGE_SCHEMA_VERSION,
            "status": "none",
            "current_best": None,
            "best_observed_experiment": None,
            "guidance": [
                "No source has policy-best authority until the runner records a valid accept.",
                "Author a complete code-level mechanism from public contracts and evidence.",
            ],
        }
    if accepted is None:
        observed = _observed_lineage_entry(
            best_observed,
            aliases_latest=best_observed_aliases_latest,
        )
        return {
            "schema_version": CANDIDATE_LINEAGE_SCHEMA_VERSION,
            "status": "bootstrap_observed_anchor",
            "current_best": None,
            "best_observed_experiment": observed,
            "guidance": [
                "This anchor is the best observed experiment, not an accepted policy best.",
                "Compare it with newer failed attempts before making one bounded, attributable code change.",
                "Preserve or replace its mechanism according to public score, gate, and runtime evidence.",
            ],
        }
    public_record = accepted_public or _public_record(accepted)
    candidate = _mapping(public_record.get("candidate"))
    decision = _mapping(public_record.get("decision"))
    source_delivery = current_best.get("source_delivery", "unavailable")
    source_summary = current_best.get("source_summary") or current_best.get("pending_reason") or "current best source unavailable"
    current = {
        "round_id": public_record.get("round_id"),
        "candidate_id": candidate.get("candidate_id"),
        "public_score": _public_score(decision),
        "changed_components": _changed_components(public_record),
        "source_logical_id": CURRENT_BEST_SOURCE_ID,
        "source_sha256": current_best.get("content_sha256"),
        "source_hash_scope": current_best.get("hash_scope"),
        "source_summary": source_summary,
        "source_delivery": source_delivery,
    }
    if source_delivery == "inline":
        current["source_payload_ref"] = CURRENT_BEST_SOURCE_ID
    elif source_delivery == "omitted_over_budget":
        current["rewrite_directive"] = (
            "Use the source summary and public feedback to propose a mechanism-level rewrite; "
            "do not assume hidden details beyond the recorded source hash."
        )
    elif current_best.get("pending_reason"):
        current["source_unavailable_reason"] = current_best["pending_reason"]
    return {
        "schema_version": CANDIDATE_LINEAGE_SCHEMA_VERSION,
        "status": "accepted_current_best",
        "current_best": current,
        "best_observed_experiment": None,
        "guidance": [
            "Current best is reusable evidence, not a template that must receive a minimal numeric tweak.",
            "A valid candidate may inherit, rewrite, recombine, or replace the current mechanism when public evidence justifies it.",
            "If reusing the current best, explain the mechanism-level reason for the change rather than only changing weights or names.",
        ],
    }


def _observed_lineage_entry(
    best_observed: Mapping[str, Any],
    *,
    aliases_latest: bool,
) -> dict[str, Any] | None:
    if not best_observed:
        return None
    observed_source = _mapping(best_observed.get("source"))
    logical_id = (
        LATEST_FAILED_SOURCE_ID if aliases_latest else BEST_OBSERVED_SOURCE_ID
    )
    observed = {
        "round_id": best_observed.get("round_id"),
        "candidate_id": best_observed.get("candidate_id"),
        "public_score": best_observed.get("public_score"),
        "accepted": False,
        "decision_reason": best_observed.get("decision_reason"),
        "failed_gates": best_observed.get("failed_gates", []),
        "authority": "authoring_anchor_only_not_policy_best",
        "source_logical_id": logical_id,
        "source_sha256": observed_source.get("content_sha256"),
        "source_hash_scope": observed_source.get("hash_scope"),
        "source_summary": (
            observed_source.get("source_summary")
            or observed_source.get("pending_reason")
            or "best observed source unavailable"
        ),
        "source_delivery": observed_source.get(
            "source_delivery",
            "unavailable",
        ),
    }
    if observed["source_delivery"] == "inline":
        observed["source_payload_ref"] = logical_id
    elif observed["source_delivery"] == "omitted_over_budget":
        observed["rewrite_directive"] = (
            "Use the source summary and public feedback only; do not assume hidden details."
        )
    elif observed_source.get("pending_reason"):
        observed["source_unavailable_reason"] = observed_source["pending_reason"]
    return observed


def _validate_observed_lineage_entry(
    observed: Mapping[str, Any],
    *,
    catalog: Mapping[str, Mapping[str, Any]],
    payloads: Mapping[str, Mapping[str, Any]],
    require_inline_payloads: bool,
) -> None:
    if observed.get("accepted") is not False:
        raise ValueError("best observed lineage must explicitly declare accepted false")
    if observed.get("authority") != "authoring_anchor_only_not_policy_best":
        raise ValueError("best observed lineage has invalid authority")
    _require_string(observed.get("round_id"), "best observed lineage round_id")
    _require_string(
        observed.get("candidate_id"),
        "best observed lineage candidate_id",
    )
    score = _public_score(observed)
    if score is None or not math.isfinite(score):
        raise ValueError("best observed lineage public_score must be finite")
    if observed.get("source_logical_id") not in {
        BEST_OBSERVED_SOURCE_ID,
        LATEST_FAILED_SOURCE_ID,
    }:
        raise ValueError("best observed lineage references an invalid source role")
    _validate_lineage_source(
        observed,
        catalog,
        payloads,
        require_inline_payloads=require_inline_payloads,
        field_name="candidate_lineage.best_observed_experiment",
    )


def _exploration_directive(lineage: Mapping[str, Any]) -> dict[str, Any]:
    status = lineage.get("status")
    has_best = status == "accepted_current_best"
    has_baseline = status == "baseline_current_best"
    has_reference = status == "reference_current_best"
    has_observed_anchor = status == "bootstrap_observed_anchor"
    if has_best:
        mode = "exploit_refine"
        instructions = [
            "Use current-best source and public feedback to make a mechanism-level candidate change.",
            (
                "When the source catalog exposes a latest nonaccepted candidate, compare its "
                "exact code and failure evidence with the current best before reusing it."
            ),
            "Do not submit identical, rationale-only, or numeric-only edits as policy improvements.",
        ]
    elif has_reference:
        mode = "reference_anchored_authoring"
        instructions = [
            "Treat the verified reference program as the current policy-best and source authority.",
            "Use its complete source and public failure evidence to author a complete replacement.",
            "Do not change runner-owned protocol, initialization, budget, seed, or acceptance rules.",
        ]
    elif has_baseline:
        mode = "baseline_anchored_authoring"
        instructions = [
            "Treat the baseline score and protocol binding as the current policy-best authority.",
            "The baseline has no candidate components source; author a complete verifier-safe mechanism from public contracts.",
            "Use rejected candidate source only as bounded authoring evidence, never as an accepted policy anchor.",
        ]
    elif has_observed_anchor:
        mode = "bootstrap_refine"
        instructions = [
            (
                "No accepted current best exists. Start from best_observed_experiment as an "
                "authoring anchor only, never as proof of policy acceptance."
            ),
            (
                "Compare its exact source and evidence with the latest nonaccepted attempt; "
                "preserve supported mechanisms and make one bounded, attributable code change."
            ),
            (
                "Do not inherit a lower-scoring combination merely because it is newer, and do "
                "not combine rejected mechanisms without positive public evidence."
            ),
        ]
    else:
        mode = "initial_mechanism_authoring"
        instructions = [
            "No accepted current best or evaluated bootstrap anchor exists; author a complete verifier-safe candidate from public contracts.",
            (
                "When a latest nonaccepted candidate source is available, inspect its exact "
                "code and failure evidence before revising or replacing the mechanism."
            ),
            "Prefer a clear prompt-visible effective mechanism over parameter-only changes.",
        ]
    return {
        "schema_version": EXPLORATION_DIRECTIVE_SCHEMA_VERSION,
        "mode": mode,
        "role": "guidance_only_not_acceptance_rule",
        "instructions": instructions,
    }


def _resolve_current_best_source(accepted: Any) -> dict[str, Any]:
    if accepted is None:
        return {
            "status": "pending",
            "source_delivery": "unavailable",
            "pending_reason": "no_accepted_current_best",
        }
    root = _candidate_package_root(accepted)
    if root is None:
        return {
            "status": "pending",
            "source_delivery": "unavailable",
            "pending_reason": "accepted_record_has_no_candidate_package_source_ref",
        }
    return _resolve_candidate_components_source(root)


def _resolve_latest_failed_source(candidate: Any) -> dict[str, Any]:
    if candidate is None:
        return {
            "status": "pending",
            "source_delivery": "unavailable",
            "pending_reason": "no_latest_failed_candidate",
        }
    root = _latest_failed_candidate_root(candidate)
    if root is None:
        return {
            "status": "pending",
            "source_delivery": "unavailable",
            "pending_reason": "latest_failed_candidate_has_no_local_source_ref",
        }
    return _resolve_candidate_components_source(root)


def _resolve_latest_failed_record_source(record: Any) -> dict[str, Any]:
    if record is None:
        return {
            "status": "pending",
            "source_delivery": "unavailable",
            "pending_reason": "no_latest_failed_candidate",
        }
    root = _candidate_package_root(record)
    if root is None:
        return {
            "status": "pending",
            "source_delivery": "unavailable",
            "pending_reason": "latest_failed_record_has_no_candidate_package_source_ref",
        }
    return _resolve_candidate_components_source(root)


def resolve_candidate_components_source_from_record(record: Any) -> dict[str, Any]:
    """Resolve one record's candidate source through the prompt-safe source boundary."""

    root = _candidate_package_root(record)
    if root is None:
        return {
            "status": "pending",
            "source_delivery": "unavailable",
            "pending_reason": "record_has_no_candidate_package_source_ref",
        }
    return _resolve_candidate_components_source(root)


def _best_observed_experiment(
    *,
    records: Sequence[Any],
    canonical_public_records: Sequence[Mapping[str, Any]],
    index: int | None,
) -> dict[str, Any]:
    if index is None:
        return {}
    public_record = canonical_public_records[index]
    decision = _mapping(public_record.get("decision"))
    candidate = _mapping(public_record.get("candidate"))
    return {
        "round_id": public_record.get("round_id"),
        "candidate_id": candidate.get("candidate_id"),
        "public_score": _public_score(decision),
        "decision_reason": decision.get("reason"),
        "failed_gates": sorted(
            str(name)
            for name, status in _mapping(decision.get("gate_statuses")).items()
            if status != "pass"
        ),
        "source": _resolve_latest_failed_record_source(records[index]),
    }


def _resolve_candidate_components_source(root: Path) -> dict[str, Any]:
    source = _read_candidate_components(root)
    if source.get("status") != "available":
        return source
    content = str(source["content"])
    encoded = content.encode("utf-8")
    content_hash = hashlib.sha256(encoded).hexdigest()
    summary = _source_summary(content)
    if len(encoded) > MAX_PROMPT_SOURCE_INLINE_BYTES:
        return {
            "status": "omitted_over_budget",
            "source_delivery": "omitted_over_budget",
            "content_sha256": content_hash,
            "bytes": len(encoded),
            "source_summary": (
                "source omitted because it exceeds inline budget; "
                f"{summary}"
            ),
            "hash_scope": "prompt_visible_sanitized_content",
        }
    return {
        "status": "inline",
        "source_delivery": "inline",
        "content": content,
        "content_sha256": content_hash,
        "bytes": len(encoded),
        "source_summary": summary,
        "hash_scope": "prompt_visible_sanitized_content",
    }


def _resolve_program_source_content(raw_source: str) -> dict[str, Any]:
    content = sanitize_prompt_public_value(raw_source)
    if not isinstance(content, str) or not content:
        raise ValueError("reference program source projection is invalid")
    encoded = content.encode("utf-8")
    content_hash = hashlib.sha256(encoded).hexdigest()
    summary = _source_summary(content)
    if len(encoded) > MAX_PROMPT_SOURCE_INLINE_BYTES:
        return {
            "status": "omitted_over_budget",
            "source_delivery": "omitted_over_budget",
            "content_sha256": content_hash,
            "bytes": len(encoded),
            "source_summary": (
                "source omitted because it exceeds inline budget; " + summary
            ),
            "hash_scope": "prompt_visible_sanitized_content",
        }
    return {
        "status": "inline",
        "source_delivery": "inline",
        "content": content,
        "content_sha256": content_hash,
        "bytes": len(encoded),
        "source_summary": summary,
        "hash_scope": "prompt_visible_sanitized_content",
    }


def _read_candidate_components(root: Path) -> dict[str, Any]:
    try:
        root_resolved = root.resolve()
    except OSError:
        return {
            "status": "pending",
            "source_delivery": "unavailable",
            "pending_reason": "candidate_package_source_ref_could_not_be_resolved",
        }
    path_unresolved = root_resolved / _COMPONENTS_RELATIVE_PATH
    if path_unresolved.is_symlink():
        return {
            "status": "pending",
            "source_delivery": "unavailable",
            "pending_reason": "candidate_components_source_is_symlink",
        }
    try:
        path = path_unresolved.resolve()
    except OSError:
        return {
            "status": "pending",
            "source_delivery": "unavailable",
            "pending_reason": "candidate_package_source_ref_could_not_be_resolved",
        }
    if root_resolved != path and root_resolved not in path.parents:
        return {
            "status": "pending",
            "source_delivery": "unavailable",
            "pending_reason": "candidate_components_source_escaped_package_root",
        }
    if not path.is_file():
        return {
            "status": "pending",
            "source_delivery": "unavailable",
            "pending_reason": "candidate_components_source_not_found",
        }
    try:
        size = int(path.stat().st_size)
        if size > MAX_PROMPT_SOURCE_READ_BYTES:
            metadata = _large_source_metadata(path, size)
            return {
                "status": "omitted_over_budget",
                "source_delivery": "omitted_over_budget",
                "pending_reason": "candidate_components_source_exceeds_read_budget",
                **metadata,
            }
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return {
            "status": "pending",
            "source_delivery": "unavailable",
            "pending_reason": "candidate_components_source_read_failed",
        }
    content = sanitize_prompt_public_value(raw)
    if not isinstance(content, str):
        content = ""
    try:
        assert_public_only_prompt_payload(content)
    except ValueError:
        return {
            "status": "omitted_for_public_safety",
            "source_delivery": "unavailable",
            "pending_reason": "candidate_components_source_failed_public_safety_check",
        }
    return {"status": "available", "content": content}


def _large_source_metadata(path: Path, size: int) -> dict[str, Any]:
    digest = hashlib.sha256()
    line_count = 0
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(64 * 1024), b""):
                digest.update(chunk)
                line_count += chunk.count(b"\n")
    except OSError:
        return {
            "bytes": size,
            "source_summary": f"source omitted because it exceeds read budget; {size} bytes",
        }
    return {
        "content_sha256": digest.hexdigest(),
        "bytes": size,
        "source_summary": (
            "source omitted because it exceeds read budget; "
            f"{size} bytes; {line_count} lines; no inline content inspected"
        ),
        "hash_scope": "raw_source_bytes_no_prompt_content",
    }


def _latest_accepted_index(records: Sequence[Mapping[str, Any]]) -> int | None:
    for index in range(len(records) - 1, -1, -1):
        decision = _mapping(records[index].get("decision"))
        if decision.get("accepted") is True:
            return index
    return None


def _best_observed_nonaccepted_index(
    records: Sequence[Mapping[str, Any]],
) -> int | None:
    best: tuple[float, int] | None = None
    for index, record in enumerate(records):
        decision = _mapping(record.get("decision"))
        public_result = _mapping(record.get("public_result"))
        surface = _mapping(record.get("candidate_surface"))
        score = _public_score(decision)
        if (
            decision.get("accepted") is not False
            or decision.get("reason") == "decision_record_invalid"
            or surface.get("classification") != "code_level"
            or public_result.get("status") != "pass"
            or score is None
            or not math.isfinite(score)
        ):
            continue
        candidate = (score, index)
        if best is None or candidate > best:
            best = candidate
    return best[1] if best is not None else None


def _immediate_previous_nonaccepted_index(
    canonical_public_records: Sequence[Mapping[str, Any]],
) -> int | None:
    if not canonical_public_records:
        return None
    decision = _mapping(canonical_public_records[-1].get("decision"))
    if (
        decision.get("accepted") is False
        and decision.get("reason") != "decision_record_invalid"
    ):
        return len(canonical_public_records) - 1
    return None


def _candidate_package_root(record: Any) -> Path | None:
    for artifact in getattr(record, "private_artifacts", ()) or ():
        if getattr(artifact, "kind", None) == "candidate_package":
            uri = getattr(artifact, "uri", None)
            if isinstance(uri, str) and _looks_like_local_path(uri):
                return Path(uri)
    return None


def _latest_failed_candidate_root(candidate: Any) -> Path | None:
    raw = getattr(candidate, "path", None)
    if isinstance(raw, str) and _looks_like_local_path(raw):
        return Path(raw)
    return None


def _public_record(record: Any) -> Mapping[str, Any]:
    if hasattr(record, "ledger_public_record"):
        return record.ledger_public_record()
    return record if isinstance(record, Mapping) else {}


def _public_score(decision: Mapping[str, Any]) -> float | None:
    value = decision.get("public_score")
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def _changed_components(public_record: Mapping[str, Any]) -> list[str]:
    candidate = _mapping(public_record.get("candidate"))
    raw = candidate.get("changed_components")
    if not isinstance(raw, list | tuple):
        verifier = _mapping(public_record.get("verifier"))
        details = _mapping(verifier.get("details"))
        raw = details.get("derived_change_set")
    return sorted(str(item) for item in raw if isinstance(item, str)) if isinstance(raw, list | tuple) else []


def _source_summary(content: str) -> str:
    lines = content.splitlines()
    interesting = [
        line.strip()
        for line in lines
        if line.strip().startswith(("def ", "class ", "from ", "import "))
    ][:8]
    if not interesting:
        return f"{len(lines)} lines; no public definitions detected"
    head = "; ".join(interesting)
    return f"{len(lines)} lines; {head}"


def _sources_share_content(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> bool:
    left_hash = left.get("content_sha256")
    right_hash = right.get("content_sha256")
    return (
        isinstance(left_hash, str)
        and bool(left_hash)
        and left_hash == right_hash
        and left.get("bytes") == right.get("bytes")
    )


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _require_list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value


def _require_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_schema(value: Mapping[str, Any], expected: str, name: str) -> None:
    if value.get("schema_version") != expected:
        raise ValueError(f"{name}.schema_version must be {expected}")


def _require_sha256(value: Any, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value.lower()):
        raise ValueError(f"{name} must be a sha256 hex digest")


def _nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    return None


def _looks_like_local_path(value: str) -> bool:
    lowered = value.lower()
    return not ("://" in lowered and not lowered.startswith("file://"))
