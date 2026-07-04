"""Shared public-data safety rules."""

from __future__ import annotations

import re
from typing import Any, Mapping


FORBIDDEN_PUBLIC_KEY_PARTS = (
    "private",
    "hidden",
    "secret",
    "api_key",
    "token",
    "trusted_eval",
    "trusted_seed",
    "checkpoint_selector",
    "raw_seed",
)
FORBIDDEN_PUBLIC_VALUE_PARTS = ("sk-",)
_SECRET_KEY_RE = re.compile(r"(?<![A-Za-z0-9_-])sk-[A-Za-z0-9][A-Za-z0-9_-]*", re.IGNORECASE)
_FORBIDDEN_PUBLIC_VALUE_RE = (
    _SECRET_KEY_RE,
    re.compile(r"\bprivate[_\s-]+seed(?:\b|_)"),
    re.compile(r"\bhidden[_\s-]+eval(?:\b|_)"),
    re.compile(r"\btrusted[_\s-]+eval(?:\b|_)"),
    re.compile(r"\btrusted[_\s-]+seed(?:\b|_)"),
    re.compile(r"\bcheckpoint[_\s-]+selector(?:\b|_)"),
    re.compile(r"\bapi[_\s-]+key(?:\b|_)"),
    re.compile(r"(?<![A-Za-z0-9_])raw[_-]+seed(?:\b|_)"),
    re.compile(r"\braw[_\s-]+seed[_\s-]+id(?:\b|_)"),
    re.compile(r"\bsecret[_\s-]+(?:key|token)(?:\b|_)"),
    re.compile(r"\baccess[_\s-]+token(?:\b|_)"),
)
_WINDOWS_BACKSLASH_PATH_RE = re.compile(r"[A-Za-z]:\\[^\s\"']+")
_WINDOWS_FORWARD_PATH_RE = re.compile(r"[A-Za-z]:/[^\s\"']+")
_ABSOLUTE_PATH_RE = re.compile(r"/(?:data|home|mnt|tmp|Users|root|var|etc)/[^\s\"']+")


def sanitize_prompt_public_value(value: Any) -> Any:
    """Scrub prompt-public values without changing their public schema shape."""

    if isinstance(value, Mapping):
        return {str(key): sanitize_prompt_public_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_prompt_public_value(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_prompt_public_value(item) for item in value]
    if isinstance(value, str):
        compact = _WINDOWS_BACKSLASH_PATH_RE.sub("[path]", value)
        compact = _WINDOWS_FORWARD_PATH_RE.sub("[path]", compact)
        compact = _ABSOLUTE_PATH_RE.sub("[path]", compact)
        return compact
    return value


def contains_forbidden_public_value_marker(value: str) -> bool:
    lowered = value.lower()
    return any(pattern.search(lowered) for pattern in _FORBIDDEN_PUBLIC_VALUE_RE)


def assert_public_only_prompt_payload(payload: Any) -> None:
    """Reject prompt payload content that should never be visible to providers."""

    def visit(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                lowered = str(key).lower()
                if any(part in lowered for part in FORBIDDEN_PUBLIC_KEY_PARTS):
                    raise ValueError(f"prompt payload contains forbidden key: {path}.{key}")
                visit(item, f"{path}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")
        elif isinstance(value, str):
            if contains_forbidden_public_value_marker(value):
                raise ValueError(f"prompt payload contains forbidden value at: {path}")

    visit(payload, "$")
