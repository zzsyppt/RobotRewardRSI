"""Task-neutral candidate authoring attribution policy."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

from true_rl_agent.agent_core.source_request_limits import (
    BEST_OBSERVED_SOURCE_ID,
    CURRENT_BEST_SOURCE_ID,
    LATEST_FAILED_SOURCE_ID,
)
from true_rl_agent.evaluation.binding import (
    canonical_json_hash,
    require_sha256_checksum,
    require_sha256_digest,
)

MAX_CANONICAL_COMPONENT_DELTAS = 1
MAX_CANONICAL_COMPONENT_NAME_CHARS = 64
MAX_CANONICAL_COMPONENT_NAME_UTF8_BYTES = 64
CANDIDATE_MECHANISM_DELTA_SCHEMA_VERSION = (
    "true_rl_agent.candidate_mechanism_delta.v1"
)
CANDIDATE_MECHANISM_OPERATION_DELTA_SCHEMA_VERSION = (
    "true_rl_agent.candidate_mechanism_operation_delta.v2"
)
CANONICAL_MECHANISM_OPERATIONS = ("add", "remove", "modify", "rename")
SINGLE_COMPONENT_VERIFIER_EQUIVALENCE_SCHEMA_VERSION = (
    "true_rl_agent.single_component_verifier_equivalence.v1"
)
VERIFIER_EQUIVALENCE_EXCLUSION_SET_SCHEMA_VERSION = (
    "true_rl_agent.verifier_equivalence_exclusion_set.v1"
)
ANTI_DEGENERATE_ASSESSMENT_SCHEMA_VERSION = (
    "true_rl_agent.anti_degenerate_assessment.v4"
)
CANONICAL_COMPONENT_TERM_KINDS = (
    "RewardTerm",
    "DiagnosticTerm",
    "ScheduleTerm",
)
_RUNTIME_KIND_BY_COMPONENT_TERM = {
    "RewardTerm": "reward_term",
    "DiagnosticTerm": "diagnostic_term",
    "ScheduleTerm": "schedule_term",
}
SINGLE_COMPONENT_DELTA_INSTRUCTION = (
    "When an authoring anchor is present, change exactly one named canonical "
    "prompt-visible component mechanism relative to it and preserve every other "
    "component mechanism; without an anchor, author the complete initial component set."
)


def component_name_contract() -> dict[str, Any]:
    """Return the one prompt- and verifier-visible component identity contract."""

    return {
        "max_chars": MAX_CANONICAL_COMPONENT_NAME_CHARS,
        "max_utf8_bytes": MAX_CANONICAL_COMPONENT_NAME_UTF8_BYTES,
        "must_be_nonempty_and_trimmed": True,
        "forbidden_substrings": [":", "->"],
        "control_characters_allowed": False,
    }


def format_canonical_mechanism_id(
    kind: str,
    name: str,
    renamed_to: str | None = None,
) -> str:
    """Build the one canonical identifier used by authoring, ledger, and memory."""

    if kind not in CANONICAL_COMPONENT_TERM_KINDS:
        raise ValueError("unsupported canonical component term kind")
    old_name = require_component_name(name)
    if renamed_to is None:
        return f"{kind}:{old_name}"
    return f"{kind}:{old_name}->{require_component_name(renamed_to)}"


def parse_canonical_mechanism_id(
    value: Any,
) -> tuple[str, str, str | None] | None:
    """Parse a canonical mechanism identifier without narrowing valid names."""

    if not isinstance(value, str):
        return None
    kind, separator, names = value.partition(":")
    if not separator or kind not in CANONICAL_COMPONENT_TERM_KINDS:
        return None
    if names.count("->") > 1:
        return None
    old_name, rename_separator, new_name = names.partition("->")
    try:
        old_name = require_component_name(old_name)
        renamed_to = (
            require_component_name(new_name) if rename_separator else None
        )
    except ValueError:
        return None
    return kind, old_name, renamed_to


def require_canonical_mechanism_delta(value: Any) -> tuple[str, ...]:
    """Validate the bounded canonical mechanism delta carried across layers."""

    if not isinstance(value, list | tuple):
        raise ValueError("canonical mechanism delta must be a sequence")
    if not value or len(value) > MAX_CANONICAL_COMPONENT_DELTAS:
        raise ValueError("canonical mechanism delta has invalid cardinality")
    normalized = tuple(value)
    if (
        any(not isinstance(item, str) for item in normalized)
        or len(set(normalized)) != len(normalized)
        or any(parse_canonical_mechanism_id(item) is None for item in normalized)
    ):
        raise ValueError("canonical mechanism delta contains an invalid identifier")
    return normalized


def require_mechanism_operation_delta(value: Any) -> dict[str, Any]:
    """Validate the verifier-owned operation-aware single-component delta."""

    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "operation",
        "before_component",
        "after_component",
        "anchor_source_checksum",
        "candidate_source_checksum",
    }:
        raise ValueError("mechanism operation delta fields mismatch")
    if value.get("schema_version") != CANDIDATE_MECHANISM_OPERATION_DELTA_SCHEMA_VERSION:
        raise ValueError("mechanism operation delta schema mismatch")
    operation = value.get("operation")
    if operation not in CANONICAL_MECHANISM_OPERATIONS:
        raise ValueError("mechanism operation delta operation is invalid")
    before = _component_operation_projection(
        value.get("before_component"),
        field_name="before_component",
    )
    after = _component_operation_projection(
        value.get("after_component"),
        field_name="after_component",
    )
    require_sha256_checksum(
        value.get("anchor_source_checksum"),
        "mechanism operation anchor source checksum",
    )
    require_sha256_checksum(
        value.get("candidate_source_checksum"),
        "mechanism operation candidate source checksum",
    )
    if operation == "add":
        valid_shape = before is None and after is not None
    elif operation == "remove":
        valid_shape = before is not None and after is None
    elif operation == "modify":
        valid_shape = (
            before is not None
            and after is not None
            and before["mechanism_id"] == after["mechanism_id"]
        )
    else:
        valid_shape = (
            before is not None
            and after is not None
            and before["mechanism_id"] != after["mechanism_id"]
            and parse_canonical_mechanism_id(before["mechanism_id"])[0]
            == parse_canonical_mechanism_id(after["mechanism_id"])[0]
        )
    if not valid_shape:
        raise ValueError("mechanism operation delta shape does not match operation")
    return {
        "schema_version": CANDIDATE_MECHANISM_OPERATION_DELTA_SCHEMA_VERSION,
        "operation": str(operation),
        "before_component": before,
        "after_component": after,
        "anchor_source_checksum": str(value["anchor_source_checksum"]),
        "candidate_source_checksum": str(value["candidate_source_checksum"]),
    }


def single_component_verifier_equivalence_key(value: Any) -> str:
    """Hash the verifier-owned, name-insensitive meaning of one component delta.

    The operation, component kind, public input dependencies, and parameter-
    normalized structure are identity.  Parent/source hashes and component names
    deliberately are not: they describe a particular trial, not whether another
    trial is an identical or parameter-only mechanism implementation.
    """

    delta = require_mechanism_operation_delta(value)
    component = delta["after_component"] or delta["before_component"]
    if component is None:  # pragma: no cover - guarded by the delta validator
        raise ValueError("mechanism operation delta has no changed component")
    parsed = parse_canonical_mechanism_id(component["mechanism_id"])
    if parsed is None:  # pragma: no cover - guarded by the delta validator
        raise ValueError("mechanism operation component identity is invalid")
    return canonical_json_hash(
        {
            "schema_version": SINGLE_COMPONENT_VERIFIER_EQUIVALENCE_SCHEMA_VERSION,
            "operation": delta["operation"],
            "component_kind": parsed[0],
            "parameter_normalized_structure_hash": component[
                "parameter_normalized_structure_hash"
            ],
            "input_fields": list(component["input_fields"]),
        }
    )


def verifier_equivalence_exclusion_set_hash(value: Any) -> str:
    """Bind the sorted verifier-equivalence keys used by an authoring lane."""

    if not isinstance(value, list | tuple):
        raise ValueError("verifier equivalence exclusions must be a sequence")
    keys = tuple(value)
    for item in keys:
        require_sha256_digest(item, "verifier equivalence exclusion")
    if keys != tuple(sorted(set(keys))):
        raise ValueError("verifier equivalence exclusions must be sorted and unique")
    return canonical_json_hash(
        {
            "schema_version": VERIFIER_EQUIVALENCE_EXCLUSION_SET_SCHEMA_VERSION,
            "keys": list(keys),
        }
    )


def _component_operation_projection(
    value: Any,
    *,
    field_name: str,
) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != {
        "mechanism_id",
        "structure_hash",
        "parameter_normalized_structure_hash",
        "input_fields",
    }:
        raise ValueError(f"mechanism operation {field_name} fields mismatch")
    mechanism_id = value.get("mechanism_id")
    parsed = parse_canonical_mechanism_id(mechanism_id)
    if parsed is None or parsed[2] is not None:
        raise ValueError(f"mechanism operation {field_name} identity is invalid")
    require_sha256_digest(
        value.get("structure_hash"),
        f"mechanism operation {field_name} structure hash",
    )
    require_sha256_digest(
        value.get("parameter_normalized_structure_hash"),
        f"mechanism operation {field_name} normalized structure hash",
    )
    input_fields = value.get("input_fields")
    if (
        not isinstance(input_fields, list)
        or any(not isinstance(item, str) or not item for item in input_fields)
        or input_fields != sorted(set(input_fields))
    ):
        raise ValueError(f"mechanism operation {field_name} input fields are invalid")
    return {
        "mechanism_id": str(mechanism_id),
        "structure_hash": str(value["structure_hash"]),
        "parameter_normalized_structure_hash": str(
            value["parameter_normalized_structure_hash"]
        ),
        "input_fields": list(input_fields),
    }


def canonical_mechanism_runtime_identity(
    value: Any,
) -> tuple[str, str] | None:
    """Map a canonical authoring identity to its runner activation identity."""

    parsed = parse_canonical_mechanism_id(value)
    if parsed is None:
        return None
    kind, old_name, renamed_to = parsed
    return _RUNTIME_KIND_BY_COMPONENT_TERM[kind], renamed_to or old_name


def select_authoring_anchor_source(
    prompt_payload: Mapping[str, Any],
) -> tuple[str | None, str | None, str]:
    """Select the sole source/hash/role used for authoring comparison."""

    current_source, current_hash = _prompt_source_payload(
        prompt_payload,
        CURRENT_BEST_SOURCE_ID,
    )
    if current_source is not None:
        return current_source, current_hash, "current_best"

    lineage = prompt_payload.get("candidate_lineage")
    if not isinstance(lineage, Mapping):
        return None, None, "none"
    observed = lineage.get("best_observed_experiment")
    if not isinstance(observed, Mapping):
        return None, None, "none"
    logical_id = observed.get("source_logical_id")
    if logical_id not in {BEST_OBSERVED_SOURCE_ID, LATEST_FAILED_SOURCE_ID}:
        return None, None, "none"
    source, digest = _prompt_source_payload(prompt_payload, str(logical_id))
    return source, digest, "best_observed_nonaccepted"


def _prompt_source_payload(
    prompt_payload: Mapping[str, Any],
    logical_id: str,
) -> tuple[str | None, str | None]:
    payloads = prompt_payload.get("source_payloads")
    if not isinstance(payloads, list):
        return None, None
    matches = [
        item
        for item in payloads
        if isinstance(item, Mapping) and item.get("logical_id") == logical_id
    ]
    if len(matches) > 1:
        raise ValueError("authoring anchor source payload must be unique")
    for item in matches:
        content = item.get("content")
        declared_digest = item.get("content_sha256")
        if not isinstance(content, str) or not isinstance(declared_digest, str):
            raise ValueError("authoring anchor source payload is invalid")
        actual_digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if declared_digest != actual_digest:
            raise ValueError("authoring anchor source hash mismatch")
        return content, actual_digest
    return None, None


def require_component_name(value: Any, field_name: str = "component name") -> str:
    """Validate the bounded component identity shared by all candidate paths."""

    try:
        encoded = value.encode("utf-8") if isinstance(value, str) else b""
    except UnicodeEncodeError:
        encoded = b""
    if (
        not isinstance(value, str)
        or not value
        or len(value) > MAX_CANONICAL_COMPONENT_NAME_CHARS
        or not encoded
        or len(encoded) > MAX_CANONICAL_COMPONENT_NAME_UTF8_BYTES
        or value != value.strip()
        or ":" in value
        or "->" in value
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError(
            f"{field_name} must be a trimmed non-empty string of at most "
            f"{MAX_CANONICAL_COMPONENT_NAME_CHARS} characters and "
            f"{MAX_CANONICAL_COMPONENT_NAME_UTF8_BYTES} UTF-8 bytes without ':', "
            "'->', or control characters"
        )
    return value

__all__ = [
    "ANTI_DEGENERATE_ASSESSMENT_SCHEMA_VERSION",
    "CANONICAL_COMPONENT_TERM_KINDS",
    "CANDIDATE_MECHANISM_DELTA_SCHEMA_VERSION",
    "CANDIDATE_MECHANISM_OPERATION_DELTA_SCHEMA_VERSION",
    "CANONICAL_MECHANISM_OPERATIONS",
    "MAX_CANONICAL_COMPONENT_NAME_CHARS",
    "MAX_CANONICAL_COMPONENT_NAME_UTF8_BYTES",
    "MAX_CANONICAL_COMPONENT_DELTAS",
    "SINGLE_COMPONENT_DELTA_INSTRUCTION",
    "SINGLE_COMPONENT_VERIFIER_EQUIVALENCE_SCHEMA_VERSION",
    "VERIFIER_EQUIVALENCE_EXCLUSION_SET_SCHEMA_VERSION",
    "canonical_mechanism_runtime_identity",
    "component_name_contract",
    "format_canonical_mechanism_id",
    "parse_canonical_mechanism_id",
    "require_canonical_mechanism_delta",
    "require_mechanism_operation_delta",
    "require_component_name",
    "select_authoring_anchor_source",
    "single_component_verifier_equivalence_key",
    "verifier_equivalence_exclusion_set_hash",
]
