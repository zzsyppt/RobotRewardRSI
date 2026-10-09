"""Bounded provider source-request resolution from prompt-visible source payloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from true_rl_agent.agent_core.authoring_response import (
    SourceRequestResponse,
    validate_source_request_response_against_prompt,
)
from true_rl_agent.agent_core.provider_contract import SOURCE_REQUEST_SERVICE_ENABLED
from true_rl_agent.agent_core.public_safety import (
    assert_public_only_prompt_payload,
    sanitize_prompt_public_value,
)
from true_rl_agent.agent_core.source_request_limits import (
    MAX_SOURCE_IDS_PER_REQUEST,
    MAX_SOURCE_REQUEST_BYTES_PER_ROUND,
    MAX_SOURCE_REQUESTS_PER_ROUND,
)
from true_rl_agent.prompting.serialization import canonical_prompt_json_bytes
from true_rl_agent.prompting.source_catalog import (
    PROMPT_SOURCE_SECTION_KEYS,
    validate_prompt_source_sections,
)


@dataclass(frozen=True)
class SourceRequestResolution:
    """Public result of servicing one provider source_request turn."""

    status: str
    feedback: dict[str, Any]
    source_payloads: list[dict[str, Any]]
    returned_bytes: int

    def attempt_record(self) -> dict[str, Any]:
        budget = self.feedback.get("budget")
        if not isinstance(budget, Mapping):
            raise ValueError("source request resolution feedback is missing budget")
        record = {
            "status": self.status,
            "returned_logical_ids": list(self.feedback.get("returned_logical_ids", [])),
            "denied_logical_ids": list(self.feedback.get("denied_logical_ids", [])),
            "returned_bytes": self.returned_bytes,
            "budget": {
                key: budget.get(key)
                for key in (
                    "source_request_count",
                    "max_source_requests_per_round",
                    "max_source_ids_per_request",
                    "cumulative_returned_bytes",
                    "max_source_request_bytes_per_round",
                )
            },
        }
        assert_public_only_prompt_payload(record)
        return record


def resolve_source_request(
    *,
    prompt_payload: Mapping[str, Any],
    response: SourceRequestResponse,
    source_request_count: int,
    source_request_bytes_returned: int,
    max_source_requests_per_round: int = MAX_SOURCE_REQUESTS_PER_ROUND,
    max_source_ids_per_request: int = MAX_SOURCE_IDS_PER_REQUEST,
    max_source_request_bytes_per_round: int = MAX_SOURCE_REQUEST_BYTES_PER_ROUND,
) -> SourceRequestResolution:
    """Resolve a source_request only from already prompt-visible source sections."""

    validate_source_request_response_against_prompt(response, prompt_payload=prompt_payload)
    catalog_by_id, payload_by_id = _validated_source_maps(prompt_payload)
    requested_ids = _dedupe_preserve_order(response.logical_ids)

    returned_payloads: list[dict[str, Any]] = []
    denied: list[dict[str, str]] = []
    returned_bytes = 0

    if not SOURCE_REQUEST_SERVICE_ENABLED:
        denied = [{"logical_id": logical_id, "reason": "source_request_service_disabled"} for logical_id in requested_ids]
    elif source_request_count >= max_source_requests_per_round:
        denied = [{"logical_id": logical_id, "reason": "source_request_turn_limit_exceeded"} for logical_id in requested_ids]
    elif len(requested_ids) > max_source_ids_per_request:
        denied = [{"logical_id": logical_id, "reason": "source_request_id_limit_exceeded"} for logical_id in requested_ids]
    else:
        budget_remaining = max_source_request_bytes_per_round - source_request_bytes_returned
        for logical_id in requested_ids:
            catalog_item = catalog_by_id.get(logical_id)
            payload = payload_by_id.get(logical_id)
            if not isinstance(catalog_item, Mapping):
                denied.append({"logical_id": logical_id, "reason": "source_not_in_catalog"})
                continue
            if catalog_item.get("requestable") is not True:
                denied.append({"logical_id": logical_id, "reason": "source_not_requestable"})
                continue
            if payload is None:
                denied.append({"logical_id": logical_id, "reason": _missing_payload_reason(catalog_item)})
                continue
            payload_bytes = canonical_prompt_json_bytes(payload)
            if payload_bytes > budget_remaining:
                denied.append({"logical_id": logical_id, "reason": "source_request_byte_budget_exceeded"})
                continue
            returned_payloads.append(dict(payload))
            returned_bytes += payload_bytes
            budget_remaining -= payload_bytes

    status = _resolution_status(returned_payloads, denied)
    feedback = _feedback(
        response=response,
        status=status,
        returned_payloads=returned_payloads,
        denied=denied,
        source_request_count=source_request_count,
        max_source_requests_per_round=max_source_requests_per_round,
        max_source_ids_per_request=max_source_ids_per_request,
        source_request_bytes_returned=source_request_bytes_returned,
        returned_bytes=returned_bytes,
        max_source_request_bytes_per_round=max_source_request_bytes_per_round,
    )
    clean_payloads = [dict(sanitize_prompt_public_value(payload)) for payload in returned_payloads]
    assert_public_only_prompt_payload(clean_payloads)
    return SourceRequestResolution(
        status=status,
        feedback=feedback,
        source_payloads=clean_payloads,
        returned_bytes=returned_bytes,
    )


def _feedback(
    *,
    response: SourceRequestResponse,
    status: str,
    returned_payloads: list[Mapping[str, Any]],
    denied: list[Mapping[str, str]],
    source_request_count: int,
    max_source_requests_per_round: int,
    max_source_ids_per_request: int,
    source_request_bytes_returned: int,
    returned_bytes: int,
    max_source_request_bytes_per_round: int,
) -> dict[str, Any]:
    feedback = {
        "schema_version": "true_rl_agent.source_request_feedback.v1",
        "status": status,
        "failure_category": status,
        "provider_status": "source_request",
        "source_request": response.to_public_dict(),
        "source_request_service_enabled": SOURCE_REQUEST_SERVICE_ENABLED,
        "returned_logical_ids": [str(item["logical_id"]) for item in returned_payloads],
        "denied_logical_ids": [dict(item) for item in denied],
        "budget": {
            "source_request_count": source_request_count + 1,
            "max_source_requests_per_round": max_source_requests_per_round,
            "requested_logical_id_count": len(response.logical_ids),
            "max_source_ids_per_request": max_source_ids_per_request,
            "returned_bytes": returned_bytes,
            "cumulative_returned_bytes": source_request_bytes_returned + returned_bytes,
            "max_source_request_bytes_per_round": max_source_request_bytes_per_round,
        },
        "instructions": [
            "Use returned source_payloads as read-only prompt evidence.",
            "Return candidate_draft when enough information is available.",
            "Do not request filesystem paths, private artifacts, or non-catalog logical ids.",
        ],
    }
    clean = dict(sanitize_prompt_public_value(feedback))
    assert_public_only_prompt_payload(clean)
    return clean


def _validated_source_maps(
    prompt_payload: Mapping[str, Any],
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    sections = validate_prompt_source_sections(
        {
            key: prompt_payload.get(key)
            for key in PROMPT_SOURCE_SECTION_KEYS
        },
        require_inline_payloads=False,
    )
    by_id: dict[str, Mapping[str, Any]] = {}
    for item in sections["source_catalog"]["items"]:
        by_id[str(item["logical_id"])] = item
    payloads_by_id: dict[str, Mapping[str, Any]] = {}
    for item in sections["source_payloads"]:
        payloads_by_id[str(item["logical_id"])] = item
    return by_id, payloads_by_id


def _dedupe_preserve_order(logical_ids: tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for logical_id in logical_ids:
        if logical_id in seen:
            continue
        seen.add(logical_id)
        deduped.append(logical_id)
    return deduped


def _missing_payload_reason(catalog_item: Mapping[str, Any]) -> str:
    status = catalog_item.get("status")
    if status == "omitted_over_budget":
        return "source_omitted_over_budget"
    if status == "pending":
        return "source_not_requestable"
    return "source_payload_not_available"


def _resolution_status(returned_payloads: list[Mapping[str, Any]], denied: list[Mapping[str, str]]) -> str:
    if returned_payloads and not denied:
        return "source_payload_returned"
    if returned_payloads:
        return "source_request_partial"
    return "source_request_denied"
