"""Task-neutral receipts for deterministic prompt capacity projections."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from true_rl_agent.evaluation.binding import canonical_json_hash, require_sha256_digest
from true_rl_agent.prompting.serialization import canonical_prompt_json_bytes
from true_rl_agent.prompt_serialization import legacy_indented_prompt_json


PROMPT_CAPACITY_RECEIPT_SCHEMA_VERSION = (
    "true_rl_agent.prompt.capacity_projection_receipt.v3"
)
_LEGACY_CAPACITY_RECEIPT_SCHEMA_VERSION = "true_rl_agent.prompt.capacity_projection_receipt.v2"
_PROMPT_CAPACITY_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "projection_id",
        "mode",
        "original_content_hash",
        "original_bytes",
        "projected_content_without_receipt_hash",
        "projected_content_without_receipt_bytes",
        "omitted",
    }
)


def build_prompt_capacity_receipt(
    *,
    original: Mapping[str, Any],
    projected_content: Mapping[str, Any],
    projection_id: str,
    omitted: Sequence[str],
) -> dict[str, Any]:
    """Describe one task-owned projection without interpreting its semantics."""

    if not isinstance(original, Mapping) or not isinstance(projected_content, Mapping):
        raise TypeError("prompt capacity receipt inputs must be mappings")
    _require_bounded_text(projection_id, "prompt capacity projection_id")
    if isinstance(omitted, str | bytes) or not isinstance(omitted, Sequence):
        raise TypeError("prompt capacity omitted reasons must be a sequence")
    normalized_omissions = sorted(
        {_require_bounded_text(item, "prompt capacity omission") for item in omitted}
    )
    return {
        "schema_version": PROMPT_CAPACITY_RECEIPT_SCHEMA_VERSION,
        "projection_id": projection_id,
        "mode": "minimum_core",
        "original_content_hash": canonical_json_hash(original),
        "original_bytes": canonical_prompt_json_bytes(original),
        "projected_content_without_receipt_hash": canonical_json_hash(
            projected_content
        ),
        "projected_content_without_receipt_bytes": canonical_prompt_json_bytes(
            projected_content
        ),
        "omitted": normalized_omissions,
    }


def verify_prompt_capacity_receipt(
    receipt: Mapping[str, Any],
    *,
    original: Mapping[str, Any],
    projected_content: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute every receipt field and reject drift or tampering."""

    if not isinstance(receipt, Mapping):
        raise TypeError("prompt capacity receipt must be a mapping")
    projection_id = receipt.get("projection_id")
    omitted = receipt.get("omitted")
    if not isinstance(omitted, list):
        raise ValueError("prompt capacity receipt omitted must be a list")
    expected = build_prompt_capacity_receipt(
        original=original,
        projected_content=projected_content,
        projection_id=projection_id,
        omitted=omitted,
    )
    if receipt.get("schema_version") == _LEGACY_CAPACITY_RECEIPT_SCHEMA_VERSION:
        expected.update(schema_version=_LEGACY_CAPACITY_RECEIPT_SCHEMA_VERSION,
            original_bytes=len(legacy_indented_prompt_json(original).encode("utf-8")),
            projected_content_without_receipt_bytes=len(legacy_indented_prompt_json(projected_content).encode("utf-8")))
    if dict(receipt) != expected:
        raise ValueError("prompt capacity receipt does not match its projection")
    return expected


def validate_prompt_capacity_projection(
    receipt: Mapping[str, Any],
    *,
    projected_content: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the common receipt shape and its visible projected content."""

    if not isinstance(receipt, Mapping):
        raise TypeError("prompt capacity receipt must be a mapping")
    if set(receipt) != _PROMPT_CAPACITY_RECEIPT_FIELDS:
        raise ValueError("prompt capacity receipt fields mismatch")
    if receipt.get("schema_version") not in {PROMPT_CAPACITY_RECEIPT_SCHEMA_VERSION, _LEGACY_CAPACITY_RECEIPT_SCHEMA_VERSION}:
        raise ValueError("prompt capacity receipt schema mismatch")
    _require_bounded_text(
        receipt.get("projection_id"),
        "prompt capacity projection_id",
    )
    if receipt.get("mode") != "minimum_core":
        raise ValueError("prompt capacity receipt mode mismatch")
    require_sha256_digest(
        receipt.get("original_content_hash"),
        "prompt capacity original_content_hash",
    )
    require_sha256_digest(
        receipt.get("projected_content_without_receipt_hash"),
        "prompt capacity projected_content_without_receipt_hash",
    )
    for field_name in (
        "original_bytes",
        "projected_content_without_receipt_bytes",
    ):
        value = receipt.get(field_name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"prompt capacity {field_name} must be non-negative")
    omitted = receipt.get("omitted")
    if not isinstance(omitted, list) or omitted != sorted(set(omitted)):
        raise ValueError("prompt capacity omissions must be sorted and unique")
    for item in omitted:
        _require_bounded_text(item, "prompt capacity omission")
    if receipt.get("projected_content_without_receipt_hash") != canonical_json_hash(
        projected_content
    ):
        raise ValueError("prompt capacity projected content hash mismatch")
    expected_bytes = (
        len(legacy_indented_prompt_json(projected_content).encode("utf-8"))
        if receipt["schema_version"] == _LEGACY_CAPACITY_RECEIPT_SCHEMA_VERSION
        else canonical_prompt_json_bytes(projected_content)
    )
    if receipt.get(
        "projected_content_without_receipt_bytes"
    ) != expected_bytes:
        raise ValueError("prompt capacity projected content size mismatch")
    return dict(receipt)


def _require_bounded_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 160:
        raise ValueError(f"{field_name} must be a non-empty string of at most 160 chars")
    return value


__all__ = [
    "PROMPT_CAPACITY_RECEIPT_SCHEMA_VERSION",
    "build_prompt_capacity_receipt",
    "validate_prompt_capacity_projection",
    "verify_prompt_capacity_receipt",
]
