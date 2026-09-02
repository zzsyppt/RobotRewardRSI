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
# ``FORBIDDEN_PUBLIC_KEY_PARTS`` remains an import-compatibility description.
# Enforcement is token-aware so ordinary identifiers such as ``token_count``
# and ``secretary_score`` are not treated as credentials.
_CAMEL_KEY_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_PUBLIC_KEY_SEPARATOR_RE = re.compile(r"[^A-Za-z0-9]+")
_FORBIDDEN_PUBLIC_KEY_TOKENS = frozenset({"private", "hidden", "secret"})
_FORBIDDEN_PUBLIC_KEY_EXACT_TOKEN_SEQUENCES = frozenset(
    {
        ("authorization",),
        ("proxy", "authorization"),
    }
)
_FORBIDDEN_PUBLIC_KEY_SEQUENCES = (
    ("api", "key"),
    ("trusted", "eval"),
    ("trusted", "seed"),
    ("checkpoint", "selector"),
    ("raw", "seed"),
    ("access", "token"),
    ("auth", "token"),
    ("authorization", "token"),
    ("bearer", "token"),
    ("refresh", "token"),
    ("secret", "token"),
    ("api", "token"),
)
_FORBIDDEN_PUBLIC_KEY_FLATTENED_IDENTIFIERS = frozenset(
    {
        "apikey",
        "clientsecret",
        "hiddeneval",
        "privatekey",
        "privateseed",
        "proxyauthorization",
        "rawseedid",
        "secretkey",
    }
    | {"".join(sequence) for sequence in _FORBIDDEN_PUBLIC_KEY_SEQUENCES}
)
# Legacy compatibility export only; enforcement must use the boundary-aware predicate.
FORBIDDEN_PUBLIC_VALUE_PARTS = ("sk-",)
_SECRET_KEY_RE = re.compile(r"(?<![A-Za-z0-9_-])sk-[A-Za-z0-9][A-Za-z0-9_-]*", re.IGNORECASE)
_AUTHORIZATION_HEADER_VALUE_RE = re.compile(
    r"(?<![a-z0-9_-])(?:proxy[-_\s]*)?authorization(?:\\?[\"'])?\s*:\s*"
    r"(?:\\?[\"'])?\s*[a-z][a-z0-9_-]*\s+[a-z0-9._~+/-]+=*"
)
_WHOLE_BEARER_CREDENTIAL_RE = re.compile(
    r"^\s*bearer\s+[a-z0-9._~+/-]+=*\s*$"
)
_EMBEDDED_BEARER_CREDENTIAL_RE = re.compile(
    r"(?<![a-z0-9])bearer\s+[a-z0-9._~+/-]+=*"
)
_FORBIDDEN_PUBLIC_VALUE_RE = (
    _SECRET_KEY_RE,
    re.compile(r"\bprivate[_\s-]+seed(?:\b|_)"),
    re.compile(r"\bhidden[_\s-]+eval(?:\b|_)"),
    re.compile(r"(?<![A-Za-z0-9_-])iid[_\s-]+hidden(?:\b|_)"),
    re.compile(r"\btrusted[_\s-]+eval(?:\b|_)"),
    re.compile(r"\btrusted[_\s-]+seed(?:\b|_)"),
    re.compile(r"\bcheckpoint[_\s-]+selector(?:\b|_)"),
    re.compile(r"\bapi[_\s-]+key(?:\b|_)"),
    re.compile(r"(?<![A-Za-z0-9_])raw[_-]+seed(?:\b|_)"),
    re.compile(r"\braw[_\s-]+seed[_\s-]+id(?:\b|_)"),
    re.compile(r"\bsecret[_\s-]+(?:key|token)(?:\b|_)"),
    re.compile(r"\baccess[_\s-]+token(?:\b|_)"),
    _AUTHORIZATION_HEADER_VALUE_RE,
    _WHOLE_BEARER_CREDENTIAL_RE,
)
_WINDOWS_BACKSLASH_PATH_RE = re.compile(r"[A-Za-z]:\\[^\s\"']+")
_WINDOWS_FORWARD_PATH_RE = re.compile(r"(?<![A-Za-z0-9+.-])[A-Za-z]:/(?!/)[^\s\"']+")
_POSIX_ABSOLUTE_PATH_RE = re.compile(r"(?<![:/A-Za-z0-9._-])/(?!/)(?:[A-Za-z0-9._-]+/)+[^\s\"']+")


def sanitize_prompt_public_value(value: Any) -> Any:
    """Scrub prompt-public values without changing their public schema shape."""

    if isinstance(value, Mapping):
        return {str(key): sanitize_prompt_public_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_prompt_public_value(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_prompt_public_value(item) for item in value]
    if isinstance(value, str):
        return scrub_public_paths(value)
    return value


def contains_public_path_marker(value: str) -> bool:
    return bool(
        _WINDOWS_BACKSLASH_PATH_RE.search(value)
        or _WINDOWS_FORWARD_PATH_RE.search(value)
        or _POSIX_ABSOLUTE_PATH_RE.search(value)
    )


def scrub_public_paths(value: str) -> str:
    compact = _WINDOWS_BACKSLASH_PATH_RE.sub("[path]", value)
    compact = _WINDOWS_FORWARD_PATH_RE.sub("[path]", compact)
    compact = _POSIX_ABSOLUTE_PATH_RE.sub("[path]", compact)
    return compact


def redact_secret_key_markers(
    value: str,
    *,
    replacement: str = "[redacted]",
) -> str:
    """Redact standalone ``sk-`` key markers without matching word fragments."""

    return _SECRET_KEY_RE.sub(lambda _match: replacement, value)


def redact_authorization_credential_markers(
    value: str,
    *,
    replacement: str = "[redacted]",
) -> str:
    """Redact untrusted diagnostic strings containing auth headers or Bearer tokens."""

    lowered = str(value).lower()
    if (
        _AUTHORIZATION_HEADER_VALUE_RE.search(lowered)
        or _EMBEDDED_BEARER_CREDENTIAL_RE.search(lowered)
    ):
        return replacement
    return value


def sanitize_untrusted_public_diagnostic(value: Any) -> Any:
    """Recursively sanitize disposable, untrusted public diagnostics.

    Authored public text uses the narrower public-value predicate so ordinary
    prose containing ``bearer`` remains usable. Runtime/provider diagnostics
    are not authoritative content, so embedded Bearer credentials are redacted
    and credential-bearing mapping keys are omitted entirely.
    """

    if isinstance(value, Mapping):
        clean: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if (
                contains_forbidden_public_key_marker(key_text)
                or redact_authorization_credential_markers(key_text) != key_text
            ):
                continue
            clean[key_text] = sanitize_untrusted_public_diagnostic(item)
        return clean
    if isinstance(value, list | tuple):
        return [sanitize_untrusted_public_diagnostic(item) for item in value]
    if isinstance(value, str):
        return redact_authorization_credential_markers(value)
    return value


def contains_forbidden_public_value_marker(value: str) -> bool:
    lowered = value.lower()
    return any(pattern.search(lowered) for pattern in _FORBIDDEN_PUBLIC_VALUE_RE)


def public_key_tokens(value: Any) -> tuple[str, ...]:
    """Return stable identifier tokens for task-neutral key safety checks."""

    text = _CAMEL_KEY_BOUNDARY_RE.sub("_", str(value))
    return tuple(
        token.lower()
        for token in _PUBLIC_KEY_SEPARATOR_RE.split(text)
        if token
    )


def contains_forbidden_public_key_marker(value: Any) -> bool:
    """Detect credential/private key names without substring false positives."""

    tokens = public_key_tokens(value)
    if not tokens:
        return False
    if tokens in _FORBIDDEN_PUBLIC_KEY_EXACT_TOKEN_SEQUENCES:
        return True
    if any(token in _FORBIDDEN_PUBLIC_KEY_TOKENS for token in tokens):
        return True
    if any(
        token in _FORBIDDEN_PUBLIC_KEY_FLATTENED_IDENTIFIERS
        for token in tokens
    ):
        return True
    if "".join(tokens) in _FORBIDDEN_PUBLIC_KEY_FLATTENED_IDENTIFIERS:
        return True
    if tokens == ("token",) or tokens[-1] == "token":
        return True
    return any(
        tokens[index : index + len(sequence)] == sequence
        for sequence in _FORBIDDEN_PUBLIC_KEY_SEQUENCES
        for index in range(len(tokens) - len(sequence) + 1)
    )


def assert_public_only_prompt_payload(payload: Any) -> None:
    """Reject prompt payload content that should never be visible to providers."""

    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if contains_forbidden_public_key_marker(key):
                    raise ValueError(f"prompt payload contains forbidden key: {path}.{key}")
                visit(item, f"{path}.{key}")
        elif isinstance(value, list | tuple):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")
        elif isinstance(value, str):
            if contains_forbidden_public_value_marker(value):
                raise ValueError(f"prompt payload contains forbidden value at: {path}")

    visit(payload, "$")
