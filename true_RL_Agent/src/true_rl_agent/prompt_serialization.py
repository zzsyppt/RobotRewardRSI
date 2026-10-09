"""Canonical JSON serialization shared by prompt producers and providers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

MAX_AUTHORING_PROMPT_PAYLOAD_BYTES = 256 * 1024


def canonical_prompt_json(value: Any) -> str:
    """Return the exact JSON text used for saved and provider-facing prompts."""

    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def sealed_prompt_json(value: Mapping[str, Any]) -> str:
    """Verify immutable historical artifacts in their original wire format."""

    if value.get("schema_version") in {
        f"true_rl_agent.prompt_payload.v{version}" for version in range(1, 5)
    }:
        return legacy_indented_prompt_json(value)
    return canonical_prompt_json(value)


def legacy_indented_prompt_json(value: Any) -> str:
    """Historical artifact/receipt verification only; never current budgeting."""
    return json.dumps(value, ensure_ascii=False, indent=2)


def canonical_prompt_json_bytes(value: Any) -> int:
    return len(canonical_prompt_json(value).encode("utf-8"))


__all__ = [
    "MAX_AUTHORING_PROMPT_PAYLOAD_BYTES",
    "canonical_prompt_json",
    "canonical_prompt_json_bytes",
    "sealed_prompt_json",
    "legacy_indented_prompt_json",
]
