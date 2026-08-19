"""Frozen, author-only plan for the Stair paired-ready provider probe.

The plan is deliberately task-owned.  It binds the fifteen historical Stair
rounds to the exact production provider requests that will be inspected by the
probe, while granting no training, evaluation, GPU, ledger, or head-transition
authority.  This module only creates and validates evidence; it never performs
network I/O.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from true_rl_agent.candidates.sibling_exclusion import (
    validate_single_component_semantic_exclusion_projection,
)
from true_rl_agent.evaluation.binding import (
    canonical_json_hash,
    require_sha256_checksum,
    require_sha256_digest,
)
from true_rl_agent.fs_contract import create_once_bytes, read_bounded_regular_file
from true_rl_agent.json_contract import canonical_json_bytes, strict_json_object_loads
from true_rl_agent.tasks.stair_climbing.config import STAIR_TASK_ID


STAIR_PAIRED_READY_PROBE_PLAN_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.paired_ready_probe_plan.v1"
)
STAIR_PAIRED_READY_PROBE_CASE_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.paired_ready_probe_case.v1"
)
STAIR_PAIRED_READY_PROBE_KIND = "production_exclusion_author_only.v1"
STAIR_PAIRED_READY_PROBE_CASE_COUNT = 15
STAIR_PAIRED_READY_PROBE_MAX_INPUT_BYTES_PER_CASE = 300_000
STAIR_PAIRED_READY_PROBE_MAX_OUTPUT_TOKENS_PER_CASE = 32_768
STAIR_PAIRED_READY_PROBE_TOTAL_OUTPUT_TOKENS = (
    STAIR_PAIRED_READY_PROBE_CASE_COUNT
    * STAIR_PAIRED_READY_PROBE_MAX_OUTPUT_TOKENS_PER_CASE
)
STAIR_PAIRED_READY_PROBE_MONEY_GOVERNOR = "user_waived_2026-08-20"

_MAX_PLAN_BYTES = 32 * 1024 * 1024
_PROVIDER_CONTRACT = {
    "schema_version": (
        "true_rl_agent.task.stair_climbing.paired_ready_probe_provider.v1"
    ),
    "base_url": "https://api.deepseek.com",
    "endpoint": "/chat/completions",
    "model": "deepseek-v4-flash",
    "thinking": {"type": "enabled"},
    "reasoning_effort": "max",
    "response_format": {"type": "json_object"},
    "stream": False,
    "max_tokens": STAIR_PAIRED_READY_PROBE_MAX_OUTPUT_TOKENS_PER_CASE,
    "max_response_bytes": 512 * 1024,
    "timeout_seconds": 300,
    "temperature_policy": "omitted",
    "requests_per_case": 1,
    "retry_policy": "none",
}
_ZERO_AUTHORITY = {
    "train_jobs": 0,
    "eval_jobs": 0,
    "sealed_queries": 0,
    "primary_ledger_writes": 0,
    "current_best_transitions": 0,
    "gpu_jobs": 0,
}
_REQUEST_FIELDS = {
    "model",
    "messages",
    "response_format",
    "stream",
    "max_tokens",
    "thinking",
    "reasoning_effort",
}
_CASE_INPUT_FIELDS = {
    "round_index",
    "source_round_id",
    "prefix_ledger_hash",
    "prefix_ledger_count",
    "source_prompt_checksum",
    "a_candidate_artifact_checksum",
    "a_candidate_source_checksum",
    "a_candidate_anchor_checksum",
    "production_exclusion_projection",
    "provider_request",
}
_CASE_FIELDS = _CASE_INPUT_FIELDS | {
    "schema_version",
    "case_id",
    "production_exclusion_hash",
    "expected_provider_prompt_hash",
    "expected_provider_request_hash",
    "expected_provider_request_utf8_bytes",
}
_PLAN_FIELDS = {
    "schema_version",
    "task_id",
    "probe_kind",
    "source_commit",
    "source_archive_checksum",
    "source_ledger_checksum",
    "source_session_genesis_hash",
    "case_count",
    "provider",
    "budget",
    "money_governor",
    "authority",
    "cases",
    "plan_hash",
}


def build_stair_paired_ready_probe_plan(
    cases: Sequence[Mapping[str, Any]],
    *,
    source_commit: str,
    source_archive_checksum: str,
    source_ledger_checksum: str,
    source_session_genesis_hash: str,
) -> dict[str, Any]:
    """Build the one canonical fifteen-case plan from exact provider requests."""

    if not isinstance(cases, list | tuple):
        raise TypeError("Stair paired-ready probe cases must be a list or tuple")
    if len(cases) != STAIR_PAIRED_READY_PROBE_CASE_COUNT:
        raise ValueError("Stair paired-ready probe requires exactly 15 cases")

    normalized_cases = [_build_case(value) for value in cases]
    body = _plan_body(
        normalized_cases,
        source_commit=source_commit,
        source_archive_checksum=source_archive_checksum,
        source_ledger_checksum=source_ledger_checksum,
        source_session_genesis_hash=source_session_genesis_hash,
    )
    plan = {**body, "plan_hash": canonical_json_hash(body)}
    return validate_stair_paired_ready_probe_plan(plan)


def create_stair_paired_ready_probe_plan(
    path: Path,
    *,
    cases: Sequence[Mapping[str, Any]],
    source_commit: str,
    source_archive_checksum: str,
    source_ledger_checksum: str,
    source_session_genesis_hash: str,
) -> dict[str, Any]:
    """Create the canonical plan once, allowing only byte-identical reuse."""

    plan = build_stair_paired_ready_probe_plan(
        cases,
        source_commit=source_commit,
        source_archive_checksum=source_archive_checksum,
        source_ledger_checksum=source_ledger_checksum,
        source_session_genesis_hash=source_session_genesis_hash,
    )
    payload = canonical_json_bytes(plan, trailing_newline=True)
    if len(payload) > _MAX_PLAN_BYTES:
        raise ValueError("Stair paired-ready probe plan exceeds its byte limit")

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        existing = read_bounded_regular_file(
            target,
            max_bytes=_MAX_PLAN_BYTES,
            label="Stair paired-ready probe plan",
        )
        if existing != payload:
            raise FileExistsError(
                "Stair paired-ready probe plan already exists with different bytes"
            )
    else:
        create_once_bytes(target, payload)
    return load_stair_paired_ready_probe_plan(target)


def load_stair_paired_ready_probe_plan(path: Path) -> dict[str, Any]:
    """Load one bounded canonical plan and verify every derived binding."""

    payload = read_bounded_regular_file(
        Path(path),
        max_bytes=_MAX_PLAN_BYTES,
        label="Stair paired-ready probe plan",
    )
    value = strict_json_object_loads(
        payload,
        context="Stair paired-ready probe plan",
    )
    if payload != canonical_json_bytes(value, trailing_newline=True):
        raise ValueError("Stair paired-ready probe plan is not canonical JSON")
    return validate_stair_paired_ready_probe_plan(value)


def validate_stair_paired_ready_probe_plan(value: Any) -> dict[str, Any]:
    """Validate topology, immutable request policy, budgets, and plan hash."""

    mapping = _exact_mapping(value, _PLAN_FIELDS, "Stair paired-ready probe plan")
    if mapping["schema_version"] != STAIR_PAIRED_READY_PROBE_PLAN_SCHEMA_VERSION:
        raise ValueError("Stair paired-ready probe plan schema mismatch")
    if mapping["task_id"] != STAIR_TASK_ID:
        raise ValueError("Stair paired-ready probe task mismatch")
    if mapping["probe_kind"] != STAIR_PAIRED_READY_PROBE_KIND:
        raise ValueError("Stair paired-ready probe kind mismatch")
    source_commit = _git_commit(mapping["source_commit"])
    source_archive_checksum = require_sha256_checksum(
        mapping["source_archive_checksum"],
        "Stair paired-ready probe source_archive_checksum",
    )
    source_ledger_checksum = require_sha256_checksum(
        mapping["source_ledger_checksum"],
        "Stair paired-ready probe source_ledger_checksum",
    )
    source_session_genesis_hash = require_sha256_digest(
        mapping["source_session_genesis_hash"],
        "Stair paired-ready probe source_session_genesis_hash",
    )
    if mapping["case_count"] != STAIR_PAIRED_READY_PROBE_CASE_COUNT:
        raise ValueError("Stair paired-ready probe case_count mismatch")
    if mapping["provider"] != _PROVIDER_CONTRACT:
        raise ValueError("Stair paired-ready probe provider contract mismatch")
    if mapping["money_governor"] != STAIR_PAIRED_READY_PROBE_MONEY_GOVERNOR:
        raise ValueError("Stair paired-ready probe money governor mismatch")
    if mapping["authority"] != _ZERO_AUTHORITY:
        raise ValueError("Stair paired-ready probe must have zero runtime authority")

    raw_cases = mapping["cases"]
    if not isinstance(raw_cases, list) or len(raw_cases) != (
        STAIR_PAIRED_READY_PROBE_CASE_COUNT
    ):
        raise ValueError("Stair paired-ready probe requires exactly 15 cases")
    cases = [_validate_case(case) for case in raw_cases]
    _validate_case_sequence(cases)

    expected_budget = _budget(cases)
    if mapping["budget"] != expected_budget:
        raise ValueError("Stair paired-ready probe budget mismatch")

    body = _plan_body(
        cases,
        source_commit=source_commit,
        source_archive_checksum=source_archive_checksum,
        source_ledger_checksum=source_ledger_checksum,
        source_session_genesis_hash=source_session_genesis_hash,
    )
    declared_hash = require_sha256_digest(
        mapping["plan_hash"],
        "Stair paired-ready probe plan_hash",
    )
    if declared_hash != canonical_json_hash(body):
        raise ValueError("Stair paired-ready probe plan hash mismatch")
    return {**body, "plan_hash": declared_hash}


def stair_paired_ready_probe_plan_hash(value: Any) -> str:
    """Return the verified canonical plan hash."""

    return str(validate_stair_paired_ready_probe_plan(value)["plan_hash"])


def _build_case(value: Any) -> dict[str, Any]:
    source = _exact_mapping(
        value,
        _CASE_INPUT_FIELDS,
        "Stair paired-ready probe source case",
    )
    request = _normalize_provider_request(source["provider_request"])
    exclusion = validate_single_component_semantic_exclusion_projection(
        source["production_exclusion_projection"]
    )
    _require_exclusion_in_request(request, exclusion)
    request_bytes = _provider_request_utf8_bytes(request)
    if request_bytes > STAIR_PAIRED_READY_PROBE_MAX_INPUT_BYTES_PER_CASE:
        raise ValueError("Stair paired-ready probe provider request exceeds 300000 bytes")

    round_index = _round_index(source["round_index"])
    case = {
        "schema_version": STAIR_PAIRED_READY_PROBE_CASE_SCHEMA_VERSION,
        "case_id": f"round-{round_index:03d}",
        "round_index": round_index,
        "source_round_id": _non_empty_text(
            source["source_round_id"],
            "Stair paired-ready probe source_round_id",
        ),
        "prefix_ledger_hash": require_sha256_digest(
            source["prefix_ledger_hash"],
            "Stair paired-ready probe prefix_ledger_hash",
        ),
        "prefix_ledger_count": _non_negative_int(
            source["prefix_ledger_count"],
            "Stair paired-ready probe prefix_ledger_count",
        ),
        "source_prompt_checksum": require_sha256_checksum(
            source["source_prompt_checksum"],
            "Stair paired-ready probe source_prompt_checksum",
        ),
        "a_candidate_artifact_checksum": require_sha256_checksum(
            source["a_candidate_artifact_checksum"],
            "Stair paired-ready probe A candidate artifact checksum",
        ),
        "a_candidate_source_checksum": require_sha256_checksum(
            source["a_candidate_source_checksum"],
            "Stair paired-ready probe A candidate source checksum",
        ),
        "a_candidate_anchor_checksum": require_sha256_checksum(
            source["a_candidate_anchor_checksum"],
            "Stair paired-ready probe A candidate anchor checksum",
        ),
        "production_exclusion_projection": exclusion,
        "production_exclusion_hash": exclusion["exclusion_hash"],
        "provider_request": request,
        "expected_provider_prompt_hash": canonical_json_hash(
            _provider_prompt_payload(request)
        ),
        "expected_provider_request_hash": canonical_json_hash(request),
        "expected_provider_request_utf8_bytes": request_bytes,
    }
    return _validate_case(case)


def _validate_case(value: Any) -> dict[str, Any]:
    case = _exact_mapping(value, _CASE_FIELDS, "Stair paired-ready probe case")
    if case["schema_version"] != STAIR_PAIRED_READY_PROBE_CASE_SCHEMA_VERSION:
        raise ValueError("Stair paired-ready probe case schema mismatch")
    round_index = _round_index(case["round_index"])
    if case["case_id"] != f"round-{round_index:03d}":
        raise ValueError("Stair paired-ready probe case_id mismatch")
    source_round_id = _non_empty_text(
        case["source_round_id"],
        "Stair paired-ready probe source_round_id",
    )
    prefix_hash = require_sha256_digest(
        case["prefix_ledger_hash"],
        "Stair paired-ready probe prefix_ledger_hash",
    )
    prefix_count = _non_negative_int(
        case["prefix_ledger_count"],
        "Stair paired-ready probe prefix_ledger_count",
    )
    if prefix_count != round_index - 1:
        raise ValueError(
            "Stair paired-ready probe prefix count must equal prior round count"
        )

    checksum_fields = (
        "source_prompt_checksum",
        "a_candidate_artifact_checksum",
        "a_candidate_source_checksum",
        "a_candidate_anchor_checksum",
    )
    checksums = {
        field: require_sha256_checksum(
            case[field],
            f"Stair paired-ready probe {field}",
        )
        for field in checksum_fields
    }
    exclusion = validate_single_component_semantic_exclusion_projection(
        case["production_exclusion_projection"]
    )
    if case["production_exclusion_hash"] != exclusion["exclusion_hash"]:
        raise ValueError("Stair paired-ready probe exclusion hash mismatch")

    request = _normalize_provider_request(case["provider_request"])
    _require_exclusion_in_request(request, exclusion)
    prompt_hash = canonical_json_hash(_provider_prompt_payload(request))
    request_hash = canonical_json_hash(request)
    request_bytes = _provider_request_utf8_bytes(request)
    if request_bytes > STAIR_PAIRED_READY_PROBE_MAX_INPUT_BYTES_PER_CASE:
        raise ValueError("Stair paired-ready probe provider request exceeds 300000 bytes")
    if case["expected_provider_prompt_hash"] != prompt_hash:
        raise ValueError("Stair paired-ready probe provider prompt hash mismatch")
    if case["expected_provider_request_hash"] != request_hash:
        raise ValueError("Stair paired-ready probe provider request hash mismatch")
    if case["expected_provider_request_utf8_bytes"] != request_bytes:
        raise ValueError("Stair paired-ready probe provider request byte count mismatch")

    return {
        "schema_version": STAIR_PAIRED_READY_PROBE_CASE_SCHEMA_VERSION,
        "case_id": f"round-{round_index:03d}",
        "round_index": round_index,
        "source_round_id": source_round_id,
        "prefix_ledger_hash": prefix_hash,
        "prefix_ledger_count": prefix_count,
        **checksums,
        "production_exclusion_projection": exclusion,
        "production_exclusion_hash": exclusion["exclusion_hash"],
        "provider_request": request,
        "expected_provider_prompt_hash": prompt_hash,
        "expected_provider_request_hash": request_hash,
        "expected_provider_request_utf8_bytes": request_bytes,
    }


def _normalize_provider_request(value: Any) -> dict[str, Any]:
    request = _exact_mapping(
        value,
        _REQUEST_FIELDS,
        "Stair paired-ready probe provider request",
    )
    if request["model"] != _PROVIDER_CONTRACT["model"]:
        raise ValueError("Stair paired-ready probe provider model mismatch")
    if request["response_format"] != _PROVIDER_CONTRACT["response_format"]:
        raise ValueError("Stair paired-ready probe response format mismatch")
    if request["stream"] is not False:
        raise ValueError("Stair paired-ready probe request must be non-streaming")
    if request["max_tokens"] != _PROVIDER_CONTRACT["max_tokens"]:
        raise ValueError("Stair paired-ready probe max_tokens mismatch")
    if request["thinking"] != _PROVIDER_CONTRACT["thinking"]:
        raise ValueError("Stair paired-ready probe thinking mode mismatch")
    if request["reasoning_effort"] != _PROVIDER_CONTRACT["reasoning_effort"]:
        raise ValueError("Stair paired-ready probe reasoning effort mismatch")

    messages = request["messages"]
    if not isinstance(messages, list) or len(messages) != 2:
        raise ValueError("Stair paired-ready probe requires system and user messages")
    normalized_messages: list[dict[str, str]] = []
    for expected_role, message in zip(("system", "user"), messages, strict=True):
        item = _exact_mapping(
            message,
            {"role", "content"},
            "Stair paired-ready probe provider message",
        )
        if item["role"] != expected_role:
            raise ValueError("Stair paired-ready probe provider message role mismatch")
        normalized_messages.append(
            {
                "role": expected_role,
                "content": _non_empty_text(
                    item["content"],
                    "Stair paired-ready probe provider message content",
                ),
            }
        )
    strict_json_object_loads(
        normalized_messages[1]["content"],
        context="Stair paired-ready probe provider user prompt",
    )
    return {
        "model": _PROVIDER_CONTRACT["model"],
        "messages": normalized_messages,
        "response_format": dict(_PROVIDER_CONTRACT["response_format"]),
        "stream": False,
        "max_tokens": STAIR_PAIRED_READY_PROBE_MAX_OUTPUT_TOKENS_PER_CASE,
        "thinking": dict(_PROVIDER_CONTRACT["thinking"]),
        "reasoning_effort": "max",
    }


def _require_exclusion_in_request(
    request: Mapping[str, Any],
    exclusion: Mapping[str, Any],
) -> None:
    user_prompt = _provider_prompt_payload(request)
    if _count_equal_mappings(user_prompt, exclusion) != 1:
        raise ValueError(
            "Stair paired-ready probe request must contain its production exclusion exactly once"
        )


def _provider_prompt_payload(request: Mapping[str, Any]) -> dict[str, Any]:
    messages = request["messages"]
    assert isinstance(messages, list)
    return strict_json_object_loads(
        messages[1]["content"],
        context="Stair paired-ready probe provider user prompt",
    )


def _count_equal_mappings(value: Any, expected: Mapping[str, Any]) -> int:
    count = 0
    if isinstance(value, Mapping):
        if dict(value) == dict(expected):
            count += 1
        for item in value.values():
            count += _count_equal_mappings(item, expected)
    elif isinstance(value, list):
        for item in value:
            count += _count_equal_mappings(item, expected)
    return count


def _provider_request_utf8_bytes(request: Mapping[str, Any]) -> int:
    # Match UrllibJsonTransport's actual request-body serialization.
    return len(
        json.dumps(
            dict(request),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    )


def _validate_case_sequence(cases: Sequence[Mapping[str, Any]]) -> None:
    round_indices = [case["round_index"] for case in cases]
    if round_indices != list(range(1, STAIR_PAIRED_READY_PROBE_CASE_COUNT + 1)):
        raise ValueError("Stair paired-ready probe cases must be ordered rounds 1..15")
    source_round_ids = [case["source_round_id"] for case in cases]
    if len(set(source_round_ids)) != len(source_round_ids):
        raise ValueError("Stair paired-ready probe source round ids must be unique")
    request_hashes = [case["expected_provider_request_hash"] for case in cases]
    if len(set(request_hashes)) != len(request_hashes):
        raise ValueError("Stair paired-ready probe provider requests must be unique")


def _plan_body(
    cases: list[dict[str, Any]],
    *,
    source_commit: str,
    source_archive_checksum: str,
    source_ledger_checksum: str,
    source_session_genesis_hash: str,
) -> dict[str, Any]:
    _validate_case_sequence(cases)
    return {
        "schema_version": STAIR_PAIRED_READY_PROBE_PLAN_SCHEMA_VERSION,
        "task_id": STAIR_TASK_ID,
        "probe_kind": STAIR_PAIRED_READY_PROBE_KIND,
        "source_commit": _git_commit(source_commit),
        "source_archive_checksum": require_sha256_checksum(
            source_archive_checksum,
            "Stair paired-ready probe source_archive_checksum",
        ),
        "source_ledger_checksum": require_sha256_checksum(
            source_ledger_checksum,
            "Stair paired-ready probe source_ledger_checksum",
        ),
        "source_session_genesis_hash": require_sha256_digest(
            source_session_genesis_hash,
            "Stair paired-ready probe source_session_genesis_hash",
        ),
        "case_count": STAIR_PAIRED_READY_PROBE_CASE_COUNT,
        "provider": _json_object_copy(_PROVIDER_CONTRACT),
        "budget": _budget(cases),
        "money_governor": STAIR_PAIRED_READY_PROBE_MONEY_GOVERNOR,
        "authority": dict(_ZERO_AUTHORITY),
        "cases": cases,
    }


def _budget(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total_input_bytes = sum(
        int(case["expected_provider_request_utf8_bytes"]) for case in cases
    )
    return {
        "input_unit": "provider_request_utf8_bytes",
        "max_input_bytes_per_case": (
            STAIR_PAIRED_READY_PROBE_MAX_INPUT_BYTES_PER_CASE
        ),
        "max_reserved_input_bytes": (
            STAIR_PAIRED_READY_PROBE_CASE_COUNT
            * STAIR_PAIRED_READY_PROBE_MAX_INPUT_BYTES_PER_CASE
        ),
        "reserved_input_bytes": total_input_bytes,
        "output_unit": "provider_completion_tokens",
        "max_output_tokens_per_case": (
            STAIR_PAIRED_READY_PROBE_MAX_OUTPUT_TOKENS_PER_CASE
        ),
        "reserved_output_tokens": STAIR_PAIRED_READY_PROBE_TOTAL_OUTPUT_TOKENS,
        "max_provider_requests": STAIR_PAIRED_READY_PROBE_CASE_COUNT,
    }


def _exact_mapping(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{label} fields mismatch")
    return dict(value)


def _json_object_copy(value: Mapping[str, Any]) -> dict[str, Any]:
    return strict_json_object_loads(
        canonical_json_bytes(dict(value)),
        context="Stair paired-ready probe JSON object",
    )


def _round_index(value: Any) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 1
        or value > STAIR_PAIRED_READY_PROBE_CASE_COUNT
    ):
        raise ValueError("Stair paired-ready probe round_index must be in 1..15")
    return value


def _git_commit(value: Any) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(
            "Stair paired-ready probe source_commit must be a 40-character "
            "lowercase Git commit"
        )
    return value


def _non_negative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _non_empty_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value


__all__ = [
    "STAIR_PAIRED_READY_PROBE_CASE_COUNT",
    "STAIR_PAIRED_READY_PROBE_MAX_INPUT_BYTES_PER_CASE",
    "STAIR_PAIRED_READY_PROBE_MAX_OUTPUT_TOKENS_PER_CASE",
    "STAIR_PAIRED_READY_PROBE_MONEY_GOVERNOR",
    "STAIR_PAIRED_READY_PROBE_PLAN_SCHEMA_VERSION",
    "STAIR_PAIRED_READY_PROBE_TOTAL_OUTPUT_TOKENS",
    "build_stair_paired_ready_probe_plan",
    "create_stair_paired_ready_probe_plan",
    "load_stair_paired_ready_probe_plan",
    "stair_paired_ready_probe_plan_hash",
    "validate_stair_paired_ready_probe_plan",
]
