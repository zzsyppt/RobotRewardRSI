"""Bounded public cross-round feedback for provider source requests."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from true_rl_agent.agent_core.public_safety import assert_public_only_prompt_payload
from true_rl_agent.agent_core.source_request_limits import (
    MAX_SOURCE_REQUEST_FEEDBACK_BYTES,
    MAX_SOURCE_REQUEST_FEEDBACK_ENTRIES,
    PROMPT_SOURCE_LOGICAL_IDS,
    SOURCE_REQUEST_DENIAL_REASONS,
)
from true_rl_agent.prompt_serialization import canonical_prompt_json_bytes

SOURCE_REQUEST_ROUND_FEEDBACK_SCHEMA_VERSION = (
    "true_rl_agent.source_request_round_feedback.v1"
)
_SOURCE_REQUEST_STATUSES = frozenset(
    {
        "source_payload_returned",
        "source_request_partial",
        "source_request_denied",
    }
)
_BUDGET_KEYS = (
    "source_request_count",
    "max_source_requests_per_round",
    "max_source_ids_per_request",
    "cumulative_returned_bytes",
    "max_source_request_bytes_per_round",
)


def build_source_request_round_feedback(
    source_request_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Aggregate service-produced attempt results into one bounded round summary."""

    valid_results: list[dict[str, Any]] = []
    for result in source_request_results:
        if not isinstance(result, Mapping):
            continue
        status = result.get("status")
        budget = _project_budget(result.get("budget"))
        if status not in _SOURCE_REQUEST_STATUSES or budget is None:
            continue
        returned, _ = _project_returned_ids(result.get("returned_logical_ids"))
        denied, _ = _project_denied_ids(result.get("denied_logical_ids"))
        valid_results.append(
            {
                "status": status,
                "returned_logical_ids": returned,
                "denied_logical_ids": denied,
                "budget": budget,
            }
        )
    if not valid_results:
        return None

    returned, _ = _project_returned_ids(
        [
            logical_id
            for result in valid_results
            for logical_id in result["returned_logical_ids"]
        ]
    )
    denied, _ = _project_denied_ids(
        [
            item
            for result in valid_results
            for item in result["denied_logical_ids"]
        ]
    )
    status = _aggregate_status(returned=returned, denied=denied)
    if status is None:
        return None
    feedback = {
        "schema_version": SOURCE_REQUEST_ROUND_FEEDBACK_SCHEMA_VERSION,
        "status": status,
        "request_count": len(valid_results),
        "returned_logical_ids": returned,
        "denied_logical_ids": denied,
        "budget": valid_results[-1]["budget"],
        "omitted_returned_logical_id_count": 0,
        "omitted_denied_logical_id_count": 0,
    }
    projected = project_source_request_round_feedback(feedback)
    if projected is None:
        raise ValueError("service-produced source request feedback could not be projected")
    return projected


def project_source_request_round_feedback(value: Any) -> dict[str, Any] | None:
    """Fail-closed allowlist projection for ledger-to-history feedback."""

    if not isinstance(value, Mapping):
        return None
    if value.get("schema_version") != SOURCE_REQUEST_ROUND_FEEDBACK_SCHEMA_VERSION:
        return None
    status = value.get("status")
    if status not in _SOURCE_REQUEST_STATUSES:
        return None
    request_count = _nonnegative_int(value.get("request_count"))
    budget = _project_budget(value.get("budget"))
    if request_count is None or request_count < 1 or budget is None:
        return None
    if request_count != budget["source_request_count"]:
        return None

    returned, newly_omitted_returned = _project_returned_ids(
        value.get("returned_logical_ids")
    )
    denied, newly_omitted_denied = _project_denied_ids(value.get("denied_logical_ids"))

    omitted_returned = _nonnegative_int(
        value.get("omitted_returned_logical_id_count")
    )
    omitted_denied = _nonnegative_int(value.get("omitted_denied_logical_id_count"))
    if omitted_returned is None:
        omitted_returned = 0
    if omitted_denied is None:
        omitted_denied = 0
    if newly_omitted_returned and not returned and omitted_returned == 0:
        return None
    if newly_omitted_denied and not denied and omitted_denied == 0:
        return None
    projected_status = _status_from_presence(
        has_returned=bool(returned) or omitted_returned > 0,
        has_denied=bool(denied) or omitted_denied > 0,
    )
    if projected_status is None or status != projected_status:
        return None

    denied, returned, cap_omitted_denied, cap_omitted_returned = _bounded_entries(
        denied=denied,
        returned=returned,
    )
    feedback = {
        "schema_version": SOURCE_REQUEST_ROUND_FEEDBACK_SCHEMA_VERSION,
        "status": status,
        "request_count": request_count,
        "returned_logical_ids": returned,
        "denied_logical_ids": denied,
        "budget": budget,
        "omitted_returned_logical_id_count": (
            omitted_returned + newly_omitted_returned + cap_omitted_returned
        ),
        "omitted_denied_logical_id_count": (
            omitted_denied + newly_omitted_denied + cap_omitted_denied
        ),
    }
    bounded = _bounded_serialized_feedback(feedback)
    if bounded is None:
        return None
    assert_public_only_prompt_payload(bounded)
    return bounded


def serialized_source_request_feedback_bytes(value: Mapping[str, Any]) -> int:
    return canonical_prompt_json_bytes(value)


def _project_returned_ids(value: Any) -> tuple[list[str], int]:
    if not isinstance(value, list | tuple):
        return [], 0
    projected: list[str] = []
    seen: set[str] = set()
    omitted = 0
    for logical_id in value:
        if not _is_known_prompt_source_logical_id(logical_id):
            omitted += 1
            continue
        logical_id = str(logical_id)
        if logical_id in seen:
            continue
        seen.add(logical_id)
        projected.append(logical_id)
    return projected, omitted


def _project_denied_ids(value: Any) -> tuple[list[dict[str, str]], int]:
    if not isinstance(value, list | tuple):
        return [], 0
    last_entries: dict[str, tuple[int, str]] = {}
    omitted = 0
    for index, raw_item in enumerate(value):
        if not isinstance(raw_item, Mapping) or set(raw_item) != {"logical_id", "reason"}:
            omitted += 1
            continue
        logical_id = raw_item.get("logical_id")
        reason = raw_item.get("reason")
        if (
            not _is_known_prompt_source_logical_id(logical_id)
            or not isinstance(reason, str)
            or reason not in SOURCE_REQUEST_DENIAL_REASONS
        ):
            omitted += 1
            continue
        last_entries[str(logical_id)] = (index, reason)
    ordered = sorted(last_entries.items(), key=lambda item: item[1][0])
    return [
        {"logical_id": logical_id, "reason": position_reason[1]}
        for logical_id, position_reason in ordered
    ], omitted


def _project_budget(value: Any) -> dict[str, int] | None:
    if not isinstance(value, Mapping):
        return None
    budget: dict[str, int] = {}
    for key in _BUDGET_KEYS:
        projected = _nonnegative_int(value.get(key))
        if projected is None:
            return None
        budget[key] = projected
    return budget


def _bounded_entries(
    *,
    denied: list[dict[str, str]],
    returned: list[str],
) -> tuple[list[dict[str, str]], list[str], int, int]:
    retained_denied = denied[-MAX_SOURCE_REQUEST_FEEDBACK_ENTRIES:]
    remaining = MAX_SOURCE_REQUEST_FEEDBACK_ENTRIES - len(retained_denied)
    retained_returned = returned[-remaining:] if remaining else []
    return (
        retained_denied,
        retained_returned,
        len(denied) - len(retained_denied),
        len(returned) - len(retained_returned),
    )


def _bounded_serialized_feedback(
    feedback: Mapping[str, Any],
) -> dict[str, Any] | None:
    bounded = {
        **dict(feedback),
        "returned_logical_ids": list(feedback.get("returned_logical_ids", [])),
        "denied_logical_ids": [
            dict(item) for item in feedback.get("denied_logical_ids", [])
        ],
    }
    while serialized_source_request_feedback_bytes(bounded) > MAX_SOURCE_REQUEST_FEEDBACK_BYTES:
        returned = bounded["returned_logical_ids"]
        denied = bounded["denied_logical_ids"]
        if returned:
            returned.pop(0)
            bounded["omitted_returned_logical_id_count"] += 1
            continue
        if denied:
            denied.pop(0)
            bounded["omitted_denied_logical_id_count"] += 1
            continue
        return None
    return bounded


def _aggregate_status(
    *,
    returned: Sequence[str],
    denied: Sequence[Mapping[str, str]],
) -> str | None:
    return _status_from_presence(
        has_returned=bool(returned),
        has_denied=bool(denied),
    )


def _status_from_presence(*, has_returned: bool, has_denied: bool) -> str | None:
    if has_returned and has_denied:
        return "source_request_partial"
    if has_denied:
        return "source_request_denied"
    if has_returned:
        return "source_payload_returned"
    return None


def _nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _is_known_prompt_source_logical_id(value: Any) -> bool:
    return isinstance(value, str) and value in PROMPT_SOURCE_LOGICAL_IDS
