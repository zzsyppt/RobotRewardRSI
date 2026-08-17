"""Canonical JSON serialization shared by prompt producers and providers."""

from __future__ import annotations

import json
from typing import Any

MAX_AUTHORING_PROMPT_PAYLOAD_BYTES = 256 * 1024


def canonical_prompt_json(value: Any) -> str:
    """Return the exact JSON text used for saved and provider-facing prompts."""

    return json.dumps(value, ensure_ascii=False, indent=2)


def canonical_prompt_json_bytes(value: Any) -> int:
    return len(canonical_prompt_json(value).encode("utf-8"))


__all__ = [
    "MAX_AUTHORING_PROMPT_PAYLOAD_BYTES",
    "canonical_prompt_json",
    "canonical_prompt_json_bytes",
]
