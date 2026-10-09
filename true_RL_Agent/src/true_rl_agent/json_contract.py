"""Strict, task-neutral JSON parsing and canonical serialization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def canonical_json_bytes(
    value: Any,
    *,
    trailing_newline: bool = False,
) -> bytes:
    """Serialize finite JSON data to one deterministic UTF-8 representation."""

    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("value must contain only finite JSON data") from exc
    if trailing_newline:
        text += "\n"
    return text.encode("utf-8")


def canonical_json_text(value: Any) -> str:
    return canonical_json_bytes(value).decode("utf-8")


def strict_json_loads(
    value: str | bytes,
    *,
    context: str = "JSON value",
) -> Any:
    """Parse JSON while rejecting duplicate keys and non-finite constants."""

    if not isinstance(value, str | bytes):
        raise TypeError(f"{context} must be text or UTF-8 bytes")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"{context} contains duplicate key: {key}")
            result[key] = item
        return result

    def reject_non_finite(token: str) -> None:
        raise ValueError(f"{context} contains non-finite number: {token}")

    try:
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        return json.loads(
            value,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_non_finite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{context} is not valid UTF-8 JSON") from exc


def strict_json_object_loads(
    value: str | bytes,
    *,
    context: str = "JSON object",
) -> dict[str, Any]:
    parsed = strict_json_loads(value, context=context)
    if not isinstance(parsed, dict):
        raise ValueError(f"{context} must contain an object")
    return parsed


def load_strict_json_object(
    path: str | Path,
    *,
    context: str | None = None,
    require_canonical: bool = False,
    trailing_newline: bool = False,
) -> dict[str, Any]:
    resolved = Path(path)
    field_context = context or resolved.name
    try:
        payload = resolved.read_bytes()
    except OSError as exc:
        raise ValueError(f"failed to read {field_context}") from exc
    parsed = strict_json_object_loads(payload, context=field_context)
    if require_canonical and payload != canonical_json_bytes(
        parsed,
        trailing_newline=trailing_newline,
    ):
        raise ValueError(f"{field_context} is not canonical JSON")
    return parsed


__all__ = [
    "canonical_json_bytes",
    "canonical_json_text",
    "load_strict_json_object",
    "strict_json_loads",
    "strict_json_object_loads",
]
