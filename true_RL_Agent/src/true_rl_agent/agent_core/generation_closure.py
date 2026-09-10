"""Versioned generation closure evidence shared by current sessions and history.

The v1 payload is a persisted protocol: changing it requires a new wire version,
not changing the historical reader or the active loop independently.
"""
from __future__ import annotations
from collections.abc import Mapping
from typing import Any
from true_rl_agent.agent_core.generation_foundation import GenerationSessionFoundation
from true_rl_agent.agent_core.generation_cost_journal import GenerationActualCostReceiptRef, generation_cost_receipt_projection
from true_rl_agent.agent_core.generation_receipts import GenerationReceiptJournal
from true_rl_agent.agent_core.generation_record import load_bound_generation_manifest
from true_rl_agent.agent_core.generation_resources import validate_stored_resource_closure_evidence
from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.fs_contract import create_once_bytes
from true_rl_agent.json_contract import canonical_json_bytes, load_strict_json_object


def load_existing_generation_execution(
    foundation: GenerationSessionFoundation,
    *,
    record: Mapping[str, Any],
) -> tuple[dict[str, Any], Any]:
    manifest = load_bound_generation_manifest(record)
    journal = GenerationReceiptJournal(
        foundation.config.output_dir / "generation_receipts" / manifest.generation_id,
        manifest=manifest,
    )
    loaded = journal.load_execution()
    if loaded is None or loaded[0] != dict(record):
        raise ValueError("terminal generation execution receipt is missing")
    return loaded



def publish_or_validate_generation_closure(
    foundation: GenerationSessionFoundation,
    *,
    ledger: list[dict[str, Any]],
    generation_record: Mapping[str, Any],
    generation_execution_receipt: Mapping[str, Any],
    actual_cost_receipt: GenerationActualCostReceiptRef,
    resource_evidence: Mapping[str, Any],
    segment_id: str,
    require_existing: bool = False,
) -> dict[str, Any]:
    generation_id = generation_record.get("generation_id")
    generation_index = generation_record.get("generation_index")
    if (
        not isinstance(generation_id, str)
        or not generation_id
        or isinstance(generation_index, bool)
        or not isinstance(generation_index, int)
        or generation_index <= 0
        or len(ledger) < generation_index
        or ledger[generation_index - 1] != dict(generation_record)
    ):
        raise ValueError("generation closure record identity is invalid")
    path = (
        foundation.config.output_dir / "generation_closures" / f"{generation_id}.json"
    )
    loaded: dict[str, Any] | None = None
    if path.exists() or path.is_symlink():
        loaded = load_strict_json_object(path, context="generation closure")
        if path.read_bytes() != canonical_json_bytes(loaded, trailing_newline=True):
            raise ValueError("generation closure is not canonical JSON")
        existing_segment = loaded.get("closed_by_segment_id")
        existing_resource = loaded.get("resource_evidence")
        if not isinstance(existing_segment, str) or not existing_segment:
            raise ValueError("generation closure segment is invalid")
        if not isinstance(existing_resource, Mapping):
            raise ValueError("generation closure resource evidence is invalid")
        segment_id = existing_segment
        resource_evidence = existing_resource
    elif require_existing:
        raise FileNotFoundError(
            f"bounded generation closure is missing: {generation_id}"
        )
    validate_stored_resource_closure_evidence(
        foundation,
        resource_evidence=resource_evidence,
        generation_index=generation_index,
    )
    body = {
        "schema_version": f"{foundation.cost_contract.operational_evidence_namespace}.generation_closure.v1",
        "session_id": foundation.config.session_id,
        "generation_id": generation_id,
        "generation_index": generation_index,
        "generation_record_hash": canonical_json_hash(generation_record),
        "generation_execution_receipt": dict(generation_execution_receipt),
        "generation_execution_receipt_hash": canonical_json_hash(
            generation_execution_receipt
        ),
        "ledger_prefix_hash": canonical_json_hash(ledger[:generation_index]),
        "resulting_current_best_hash": generation_record.get(
            "resulting_current_best_hash"
        ),
        "actual_cost_receipt": generation_cost_receipt_projection(actual_cost_receipt),
        "resource_evidence": dict(resource_evidence),
        "resource_evidence_hash": canonical_json_hash(resource_evidence),
        "closed_by_segment_id": segment_id,
        "acceptance_authority": False,
    }
    payload = {**body, "closure_hash": canonical_json_hash(body)}
    if loaded is not None:
        if loaded != payload:
            raise ValueError("generation closure differs from recomputation")
        return loaded
    path.parent.mkdir(parents=True, exist_ok=True)
    create_once_bytes(path, canonical_json_bytes(payload, trailing_newline=True))
    return payload

