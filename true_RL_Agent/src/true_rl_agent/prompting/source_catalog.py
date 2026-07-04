"""Prompt-visible source catalog and history-best lineage projection."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

from true_rl_agent.agent_core.public_safety import sanitize_prompt_public_value
from true_rl_agent.prompting.public_safety import assert_public_only_prompt_payload


SOURCE_CATALOG_SCHEMA_VERSION = "true_rl_agent.source_catalog.v1"
CANDIDATE_LINEAGE_SCHEMA_VERSION = "true_rl_agent.candidate_lineage.v1"
SOURCE_PAYLOAD_SCHEMA_VERSION = "true_rl_agent.prompt_source_payload.v1"
EXPLORATION_DIRECTIVE_SCHEMA_VERSION = "true_rl_agent.exploration_directive.v1"
MAX_PROMPT_SOURCE_INLINE_BYTES = 24 * 1024
MAX_PROMPT_SOURCE_READ_BYTES = 256 * 1024
CURRENT_BEST_SOURCE_ID = "current_best/training_code/components.py"
LATEST_FAILED_SOURCE_ID = "attempt/latest_failed/training_code/components.py"
_CANDIDATE_COMPONENTS_SOURCE_ID = "api/candidate_components.py"
_KNOWN_GOOD_SOURCE_ID = "examples/known_good_reward_terms.py"
_COMPONENTS_RELATIVE_PATH = Path("training_code") / "components.py"
PROMPT_SOURCE_SECTION_KEYS = (
    "source_catalog",
    "candidate_lineage",
    "source_payloads",
    "exploration_directive",
)
_SOURCE_CATALOG_LOGICAL_IDS = frozenset(
    {
        _CANDIDATE_COMPONENTS_SOURCE_ID,
        CURRENT_BEST_SOURCE_ID,
        _KNOWN_GOOD_SOURCE_ID,
        LATEST_FAILED_SOURCE_ID,
    }
)


def build_prompt_source_context(
    records: Sequence[Any],
    *,
    known_good_examples: Any = None,
    latest_failed_candidate: Any = None,
) -> dict[str, Any]:
    """Build prompt source catalog, source payloads, lineage, and generic directive."""

    accepted = _latest_accepted_record(records)
    current_best = _resolve_current_best_source(accepted)
    if latest_failed_candidate is not None:
        latest_failed = _resolve_latest_failed_source(latest_failed_candidate)
    else:
        latest_failed = _resolve_latest_failed_record_source(_immediate_previous_runtime_failed_record(records))
    catalog = _source_catalog(
        current_best=current_best,
        latest_failed=latest_failed,
        known_good_examples=known_good_examples,
    )
    source_payloads = _source_payloads(current_best=current_best, latest_failed=latest_failed)
    lineage = _candidate_lineage(accepted=accepted, current_best=current_best)
    directive = _exploration_directive(lineage)
    sections = {
        "source_catalog": catalog,
        "candidate_lineage": lineage,
        "source_payloads": source_payloads,
        "exploration_directive": directive,
    }
    return validate_prompt_source_sections(sections)


def default_prompt_source_sections() -> dict[str, Any]:
    return build_prompt_source_context([])


def validate_prompt_source_sections(sections: Mapping[str, Any]) -> dict[str, Any]:
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
    payloads = _validate_source_payloads(clean["source_payloads"], catalog)
    lineage = _validate_candidate_lineage(clean["candidate_lineage"], catalog, payloads)
    _validate_exploration_directive(clean["exploration_directive"], lineage)
    assert_public_only_prompt_payload(clean)
    return clean


def _validate_source_catalog(value: Any) -> dict[str, Mapping[str, Any]]:
    catalog = _require_mapping(value, "source_catalog")
    _require_schema(catalog, SOURCE_CATALOG_SCHEMA_VERSION, "source_catalog")
    policy = _require_mapping(catalog.get("policy"), "source_catalog.policy")
    if policy.get("ids_are_closed_allowlist") is not True:
        raise ValueError("source_catalog.policy.ids_are_closed_allowlist must be true")
    if policy.get("provider_filesystem_access") is not False:
        raise ValueError("source_catalog.policy.provider_filesystem_access must be false")
    if policy.get("source_request_enabled") is not False:
        raise ValueError("source_catalog.policy.source_request_enabled must be false")
    items = _require_list(catalog.get("items"), "source_catalog.items")
    by_id: dict[str, Mapping[str, Any]] = {}
    for index, raw_item in enumerate(items):
        item = _require_mapping(raw_item, f"source_catalog.items[{index}]")
        logical_id = _require_string(item.get("logical_id"), f"source_catalog.items[{index}].logical_id")
        if logical_id not in _SOURCE_CATALOG_LOGICAL_IDS:
            raise ValueError(f"source_catalog contains unknown logical_id: {logical_id}")
        if logical_id in by_id:
            raise ValueError(f"source_catalog contains duplicate logical_id: {logical_id}")
        if item.get("requestable") is not False:
            raise ValueError(f"source_catalog item must not be provider-requestable: {logical_id}")
        if _nonnegative_int(item.get("max_bytes")) is None:
            raise ValueError(f"source_catalog item has invalid max_bytes: {logical_id}")
        if item.get("content_sha256") is not None:
            _require_sha256(item.get("content_sha256"), f"source_catalog item {logical_id} content_sha256")
        if item.get("bytes") is not None and _nonnegative_int(item.get("bytes")) is None:
            raise ValueError(f"source_catalog item has invalid bytes: {logical_id}")
        by_id[logical_id] = item
    if frozenset(by_id) != _SOURCE_CATALOG_LOGICAL_IDS:
        missing = sorted(_SOURCE_CATALOG_LOGICAL_IDS - frozenset(by_id))
        extra = sorted(frozenset(by_id) - _SOURCE_CATALOG_LOGICAL_IDS)
        raise ValueError(f"source_catalog logical_ids must match closed allowlist; missing={missing}; extra={extra}")
    return by_id


def _validate_source_payloads(value: Any, catalog: Mapping[str, Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
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
    for logical_id, item in catalog.items():
        if item.get("status") == "inline" or item.get("inline_by_default") is True:
            if logical_id not in by_id:
                raise ValueError(f"inline source_catalog item has no source_payload: {logical_id}")
    return by_id


def _validate_candidate_lineage(
    value: Any,
    catalog: Mapping[str, Mapping[str, Any]],
    payloads: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    lineage = _require_mapping(value, "candidate_lineage")
    _require_schema(lineage, CANDIDATE_LINEAGE_SCHEMA_VERSION, "candidate_lineage")
    status = lineage.get("status")
    if status == "none":
        if lineage.get("current_best") is not None:
            raise ValueError("candidate_lineage.current_best must be null when status is none")
        if CURRENT_BEST_SOURCE_ID in payloads:
            raise ValueError("current-best source_payload must be empty when candidate_lineage.status is none")
        return lineage
    if status != "accepted_current_best":
        raise ValueError(f"unknown candidate_lineage.status: {status}")
    current_best = _require_mapping(lineage.get("current_best"), "candidate_lineage.current_best")
    logical_id = _require_string(current_best.get("source_logical_id"), "candidate_lineage.current_best.source_logical_id")
    if logical_id not in catalog:
        raise ValueError(f"candidate_lineage references unknown source logical_id: {logical_id}")
    source_delivery = _require_string(
        current_best.get("source_delivery"), "candidate_lineage.current_best.source_delivery"
    )
    catalog_item = catalog[logical_id]
    if source_delivery == "inline":
        if current_best.get("source_payload_ref") != logical_id:
            raise ValueError("inline candidate_lineage must reference its source_payload")
        if logical_id not in payloads:
            raise ValueError(f"inline candidate_lineage source has no payload: {logical_id}")
        if catalog_item.get("status") != "inline":
            raise ValueError(f"inline candidate_lineage source is not inline in catalog: {logical_id}")
    elif source_delivery in {"omitted_over_budget", "unavailable"}:
        if logical_id in payloads:
            raise ValueError(f"non-inline candidate_lineage source must not have payload: {logical_id}")
        if source_delivery == "omitted_over_budget" and catalog_item.get("status") != "omitted_over_budget":
            raise ValueError(f"omitted candidate_lineage source is not omitted in catalog: {logical_id}")
    else:
        raise ValueError(f"unsupported candidate_lineage source_delivery: {source_delivery}")
    source_sha = current_best.get("source_sha256")
    catalog_sha = catalog_item.get("content_sha256")
    if source_sha is not None:
        _require_sha256(source_sha, "candidate_lineage.current_best.source_sha256")
    if catalog_sha is not None and source_sha != catalog_sha:
        raise ValueError(f"candidate_lineage source hash differs from source_catalog: {logical_id}")
    return lineage


def _validate_exploration_directive(value: Any, lineage: Mapping[str, Any]) -> None:
    directive = _require_mapping(value, "exploration_directive")
    _require_schema(directive, EXPLORATION_DIRECTIVE_SCHEMA_VERSION, "exploration_directive")
    expected_mode = "exploit_refine" if lineage.get("status") == "accepted_current_best" else "initial_mechanism_authoring"
    if directive.get("mode") != expected_mode:
        raise ValueError(f"exploration_directive.mode must be {expected_mode}")


def _source_catalog(
    *,
    current_best: Mapping[str, Any],
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
        _known_good_catalog_item(known_good_examples),
        _latest_failed_catalog_item(latest_failed),
    ]
    catalog = {
        "schema_version": SOURCE_CATALOG_SCHEMA_VERSION,
        "policy": {
            "ids_are_closed_allowlist": True,
            "provider_filesystem_access": False,
            "source_request_enabled": False,
            "source_resolution_owner": "runner_prompt_source_resolver",
        },
        "items": items,
    }
    return catalog


def _latest_failed_catalog_item(latest_failed: Mapping[str, Any]) -> dict[str, Any]:
    item = {
        "logical_id": LATEST_FAILED_SOURCE_ID,
        "role": "latest_failed_candidate_source",
        "candidate_use": "Use this failed attempt source only to repair the latest verifier/runtime issue.",
        "visibility": "prompt_visible_metadata",
        "inline_by_default": latest_failed.get("source_delivery") == "inline",
        "requestable": False,
        "max_bytes": MAX_PROMPT_SOURCE_INLINE_BYTES,
        "status": latest_failed.get("status", "pending"),
        "content_sha256": latest_failed.get("content_sha256"),
        "bytes": latest_failed.get("bytes"),
        "hash_scope": latest_failed.get("hash_scope"),
        "pending_reason": latest_failed.get("pending_reason"),
        "source_owner": "latest_failed_candidate_attempt",
        "scrub_policy": "path_secret_and_public_only_guard",
    }
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
    item = {
        "logical_id": CURRENT_BEST_SOURCE_ID,
        "role": "latest_accepted_candidate_source",
        "candidate_use": "Inherit, rewrite, recombine, or refute the current best mechanism without copying it blindly.",
        "visibility": "prompt_visible_metadata",
        "inline_by_default": current_best.get("source_delivery") == "inline",
        "requestable": False,
        "max_bytes": MAX_PROMPT_SOURCE_INLINE_BYTES,
        "status": current_best.get("status", "pending"),
        "content_sha256": current_best.get("content_sha256"),
        "bytes": current_best.get("bytes"),
        "hash_scope": current_best.get("hash_scope"),
        "pending_reason": current_best.get("pending_reason"),
        "source_owner": "accepted_candidate_lineage",
        "scrub_policy": "path_secret_and_public_only_guard",
    }
    return {key: value for key, value in item.items() if value is not None}


def _source_payloads(*, current_best: Mapping[str, Any], latest_failed: Mapping[str, Any]) -> list[dict[str, Any]]:
    payloads = [
        _source_payload(
            logical_id=CURRENT_BEST_SOURCE_ID,
            role="latest_accepted_candidate_source",
            source=current_best,
        ),
        _source_payload(
            logical_id=LATEST_FAILED_SOURCE_ID,
            role="latest_failed_candidate_source",
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


def _candidate_lineage(*, accepted: Any, current_best: Mapping[str, Any]) -> dict[str, Any]:
    guidance = [
        "Current best is reusable evidence, not a template that must receive a minimal numeric tweak.",
        "A valid candidate may inherit, rewrite, recombine, or replace the current mechanism when public evidence justifies it.",
        "If reusing the current best, explain the mechanism-level reason for the change rather than only changing weights or names.",
    ]
    if accepted is None:
        return {
            "schema_version": CANDIDATE_LINEAGE_SCHEMA_VERSION,
            "status": "none",
            "current_best": None,
            "guidance": guidance,
        }
    public_record = _public_record(accepted)
    candidate = _mapping(public_record.get("candidate"))
    decision = _mapping(public_record.get("decision"))
    source_delivery = current_best.get("source_delivery", "unavailable")
    source_summary = current_best.get("source_summary") or current_best.get("pending_reason") or "current best source unavailable"
    current = {
        "round_id": public_record.get("round_id"),
        "candidate_id": candidate.get("candidate_id"),
        "public_score": _public_score(decision, public_record),
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
        "guidance": guidance,
    }


def _exploration_directive(lineage: Mapping[str, Any]) -> dict[str, Any]:
    has_best = lineage.get("status") == "accepted_current_best"
    return {
        "schema_version": EXPLORATION_DIRECTIVE_SCHEMA_VERSION,
        "mode": "exploit_refine" if has_best else "initial_mechanism_authoring",
        "role": "guidance_only_not_acceptance_rule",
        "instructions": (
            [
                "Use current-best source and public feedback to make a mechanism-level candidate change.",
                "Do not submit identical, rationale-only, or numeric-only edits as policy improvements.",
            ]
            if has_best
            else [
                "No accepted current best exists yet; author a complete verifier-safe candidate from public contracts.",
                "Prefer a clear RewardTerm or DiagnosticTerm mechanism over parameter-only changes.",
            ]
        ),
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


def _latest_accepted_record(records: Sequence[Any]) -> Any:
    for record in reversed(list(records)):
        public = _public_record(record)
        decision = _mapping(public.get("decision"))
        if decision.get("accepted") is True:
            return record
    return None


def _immediate_previous_runtime_failed_record(records: Sequence[Any]) -> Any:
    if not records:
        return None
    record = records[-1]
    public = _public_record(record)
    decision = _mapping(public.get("decision"))
    if decision.get("reason") == "candidate_runtime_failed":
        return record
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


def _public_score(decision: Mapping[str, Any], public_record: Mapping[str, Any]) -> float | None:
    value = decision.get("public_score")
    if isinstance(value, int | float):
        return float(value)
    public_result = _mapping(public_record.get("public_result"))
    value = public_result.get("public_score")
    return float(value) if isinstance(value, int | float) else None


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
