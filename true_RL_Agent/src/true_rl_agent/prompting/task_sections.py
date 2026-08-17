"""Shared task-owned prompt section sanitizer."""

from __future__ import annotations

import copy
from typing import Any, Mapping

from true_rl_agent.agent_core.context import sanitize_prompt_public_context
from true_rl_agent.evaluation.binding import canonical_json_hash, require_sha256_digest
from true_rl_agent.prompting.public_safety import assert_public_only_prompt_payload
from true_rl_agent.prompting.capacity import validate_prompt_capacity_projection
from true_rl_agent.prompting.serialization import canonical_prompt_json_bytes

PROVIDER_TASK_PROMPT_SECTION_KEYS = (
    "task",
    "reward_inputs_contract",
    "public_eval_contract",
    "public_feedback_summary",
    "runtime_feedback_summary",
    "known_good_examples",
    "selection_guidance",
)
_TASK_ONLY_PROMPT_SECTION_KEYS = frozenset({"task_capability_metadata"})
TASK_PROMPT_SECTION_KEYS = frozenset(PROVIDER_TASK_PROMPT_SECTION_KEYS).union(
    _TASK_ONLY_PROMPT_SECTION_KEYS
)
DISALLOWED_TASK_PROMPT_KEY_PARTS = ("curriculum", "stage_list")
MAX_TASK_PROMPT_SECTIONS_BYTES = 32 * 1024
PROVIDER_TASK_PROMPT_RECEIPT_SCHEMA_VERSION = (
    "true_rl_agent.prompt.provider_task_sections_receipt.v1"
)
_PROVIDER_TASK_PROMPT_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "serializer",
        "section_keys",
        "content_hash",
        "serialized_bytes",
        "max_serialized_bytes",
        "headroom_bytes",
    }
)


def sanitize_task_prompt_sections(sections: Mapping[str, Any]) -> dict[str, Any]:
    """Sanitize and validate task-owned prompt sections before prompt insertion."""

    if not isinstance(sections, Mapping):
        raise TypeError("task prompt sections must be a mapping")
    clean = sanitize_prompt_public_context(dict(sections))
    unknown_sections = sorted(set(clean) - TASK_PROMPT_SECTION_KEYS)
    if unknown_sections:
        raise ValueError(f"unknown task prompt section keys: {unknown_sections}")
    _reject_disallowed_task_prompt_keys(clean, path="$")
    assert_public_only_prompt_payload(clean)
    _validate_capacity_projection_if_present(clean)
    size = canonical_prompt_json_bytes(clean)
    if size > MAX_TASK_PROMPT_SECTIONS_BYTES:
        raise ValueError(
            "task prompt sections exceed "
            f"{MAX_TASK_PROMPT_SECTIONS_BYTES} bytes after sanitization: {size}"
        )
    return clean


def provider_task_prompt_sections(
    prompt_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Extract the exact task-owned sections present in a provider payload."""

    if not isinstance(prompt_payload, Mapping):
        raise TypeError("provider prompt payload must be a mapping")
    sections = {
        key: copy.deepcopy(prompt_payload[key])
        for key in PROVIDER_TASK_PROMPT_SECTION_KEYS
        if key in prompt_payload
    }
    return sanitize_task_prompt_sections(sections)


def build_provider_task_prompt_receipt(
    prompt_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Measure the canonical task sections that are actually sent to a provider."""

    sections = provider_task_prompt_sections(prompt_payload)
    serialized_bytes = canonical_prompt_json_bytes(sections)
    if serialized_bytes > MAX_TASK_PROMPT_SECTIONS_BYTES:
        raise ValueError(
            "provider-facing task prompt sections exceed "
            f"{MAX_TASK_PROMPT_SECTIONS_BYTES} bytes: {serialized_bytes}"
        )
    return {
        "schema_version": PROVIDER_TASK_PROMPT_RECEIPT_SCHEMA_VERSION,
        "serializer": "canonical_prompt_json.v1",
        "section_keys": list(sections),
        "content_hash": canonical_json_hash(sections),
        "serialized_bytes": serialized_bytes,
        "max_serialized_bytes": MAX_TASK_PROMPT_SECTIONS_BYTES,
        "headroom_bytes": MAX_TASK_PROMPT_SECTIONS_BYTES - serialized_bytes,
    }


def validate_provider_task_prompt_receipt(
    receipt: Mapping[str, Any],
    *,
    prompt_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute the provider-facing task receipt and reject drift."""

    if not isinstance(receipt, Mapping):
        raise TypeError("provider task prompt receipt must be a mapping")
    if set(receipt) != _PROVIDER_TASK_PROMPT_RECEIPT_FIELDS:
        raise ValueError("provider task prompt receipt fields mismatch")
    if receipt.get("schema_version") != PROVIDER_TASK_PROMPT_RECEIPT_SCHEMA_VERSION:
        raise ValueError("provider task prompt receipt schema mismatch")
    if receipt.get("serializer") != "canonical_prompt_json.v1":
        raise ValueError("provider task prompt receipt serializer mismatch")
    require_sha256_digest(
        receipt.get("content_hash"),
        "provider task prompt content_hash",
    )
    expected = build_provider_task_prompt_receipt(prompt_payload)
    if dict(receipt) != expected:
        raise ValueError("provider task prompt receipt does not match payload")
    return expected


def _validate_capacity_projection_if_present(sections: Mapping[str, Any]) -> None:
    feedback = sections.get("public_feedback_summary")
    if not isinstance(feedback, Mapping):
        return
    receipt = feedback.get("capacity_projection_receipt")
    if receipt is None:
        return
    projected_content = copy.deepcopy(dict(sections))
    projected_feedback = projected_content.get("public_feedback_summary")
    if not isinstance(projected_feedback, dict):
        raise ValueError("prompt capacity receipt requires public_feedback_summary")
    projected_feedback.pop("capacity_projection_receipt", None)
    validate_prompt_capacity_projection(
        receipt,
        projected_content=projected_content,
    )


def _reject_disallowed_task_prompt_keys(value: Any, *, path: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered = str(key).lower()
            if any(part in lowered for part in DISALLOWED_TASK_PROMPT_KEY_PARTS):
                raise ValueError(f"task prompt sections cannot expose curriculum/stage interfaces at {path}.{key}")
            _reject_disallowed_task_prompt_keys(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_disallowed_task_prompt_keys(item, path=f"{path}[{index}]")
