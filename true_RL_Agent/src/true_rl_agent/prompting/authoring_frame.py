"""Provider authoring context frame metadata."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from true_rl_agent.prompting.public_safety import assert_public_only_prompt_payload


AUTHORING_CONTEXT_FRAME_SCHEMA_VERSION = "true_rl_agent.authoring_context_frame.v1"
AUTHORING_CONTEXT_FRAME_KINDS = (
    "initial_authoring",
    "candidate_draft",
    "static_repair",
    "rewrite",
)
RESERVED_AUTHORING_CONTEXT_FRAME_KINDS = ("source_followup",)
MAX_AUTHORING_PROMPT_PAYLOAD_BYTES = 256 * 1024
MAX_AUTHORING_SECTION_BYTES = 96 * 1024
SOURCE_PAYLOAD_BYTES_LIMIT_PHASE0 = 0
_BYTES_PER_TOKEN_ESTIMATE = 4


@dataclass(frozen=True)
class AuthoringFramePolicy:
    frame_kind: str
    purpose: str
    required_sections: tuple[str, ...]
    optional_sections: tuple[str, ...]
    omitted_recent_history_reason: str | None = None


_STABLE_CONTRACT_REFS = (
    "task",
    "loop_contract",
    "required_candidate_package",
    "reward_inputs_contract",
    "public_eval_contract",
    "control_effectiveness_contract",
)
_SOURCE_CONTEXT_REFS = (
    "source_catalog",
    "candidate_lineage",
    "source_payloads",
    "exploration_directive",
)
_COMMON_OPTIONAL_SECTIONS = (
    "public_feedback_summary",
    "runtime_feedback_summary",
    "public_history",
    "known_good_examples",
    "selection_guidance",
    "provider_or_runtime_repair_feedback",
)
_FRAME_POLICIES = {
    "initial_authoring": AuthoringFramePolicy(
        frame_kind="initial_authoring",
        purpose="classify the first authoring context; V1 provider calls still use candidate_draft",
        required_sections=_STABLE_CONTRACT_REFS,
        optional_sections=_COMMON_OPTIONAL_SECTIONS,
    ),
    "candidate_draft": AuthoringFramePolicy(
        frame_kind="candidate_draft",
        purpose="write a provider-compatible candidate package from the public contract and bounded history",
        required_sections=(
            *_STABLE_CONTRACT_REFS,
            *_SOURCE_CONTEXT_REFS,
            "public_history",
            "selection_guidance",
            "schema_version",
        ),
        optional_sections=(
            "public_feedback_summary",
            "runtime_feedback_summary",
            "known_good_examples",
        ),
    ),
    "static_repair": AuthoringFramePolicy(
        frame_kind="static_repair",
        purpose="repair provider, schema, verifier, or static authoring feedback without replaying full history",
        required_sections=(
            *_STABLE_CONTRACT_REFS,
            "source_catalog",
            "candidate_lineage",
            "provider_or_runtime_repair_feedback",
            "schema_version",
        ),
        optional_sections=(
            "public_feedback_summary",
            "runtime_feedback_summary",
            "public_history",
            "known_good_examples",
            "source_payloads",
            "exploration_directive",
            "selection_guidance",
        ),
        omitted_recent_history_reason="omitted_for_static_repair_frame",
    ),
    "rewrite": AuthoringFramePolicy(
        frame_kind="rewrite",
        purpose="rewrite a draft from bounded public feedback and explicit lineage/source summaries",
        required_sections=(
            *_STABLE_CONTRACT_REFS,
            "source_catalog",
            "candidate_lineage",
            "exploration_directive",
            "public_feedback_summary",
            "runtime_feedback_summary",
            "selection_guidance",
            "schema_version",
        ),
        optional_sections=("public_history", "known_good_examples", "source_payloads"),
        omitted_recent_history_reason="omitted_for_rewrite_frame",
    ),
}


def infer_authoring_frame_kind(public_context: Mapping[str, Any]) -> str:
    if isinstance(public_context.get("candidate_repair_feedback"), Mapping):
        return "static_repair"
    return "candidate_draft"


def normalize_authoring_frame_kind(frame_kind: str | None, public_context: Mapping[str, Any]) -> str:
    kind = frame_kind or infer_authoring_frame_kind(public_context)
    if kind in RESERVED_AUTHORING_CONTEXT_FRAME_KINDS:
        raise ValueError(f"disabled authoring frame kind: {kind}")
    if kind not in _FRAME_POLICIES:
        raise ValueError(f"unknown authoring frame kind: {kind}")
    return kind


def compact_public_history_for_frame(
    public_history: Mapping[str, Any],
    *,
    frame_kind: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    policy = _FRAME_POLICIES[frame_kind]
    history = dict(public_history)
    omitted: list[dict[str, Any]] = []
    if policy.omitted_recent_history_reason is None:
        return history, omitted

    recent = history.get("recent_outcomes")
    if not isinstance(recent, list) or not recent:
        history["recent_outcomes"] = []
        return history, omitted

    omitted_count = len(recent)
    history["recent_outcomes"] = []
    history["omitted_round_count"] = int(history.get("omitted_round_count", 0) or 0) + omitted_count
    omitted.append(
        {
            "section": "public_history.recent_outcomes",
            "reason": policy.omitted_recent_history_reason,
            "omitted_item_count": omitted_count,
            "recoverable_from": "prior ledger public records",
        }
    )
    return history, omitted


def build_authoring_context_frame(
    *,
    frame_kind: str,
    section_values: Mapping[str, Any],
    section_order: Sequence[str],
    omitted_sections: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    policy = _FRAME_POLICIES[frame_kind]
    included_sections = [
        section
        for section in section_order
        if section in section_values
        and section_values[section] is not None
        and section != "authoring_context_frame"
    ]
    _require_sections(policy, included_sections)
    section_bytes = {section: _serialized_size(section_values[section]) for section in included_sections}
    _raise_for_oversized_sections(section_bytes)
    estimated_prompt_bytes = sum(section_bytes.values())
    frame = {
        "schema_version": AUTHORING_CONTEXT_FRAME_SCHEMA_VERSION,
        "frame_id": f"{frame_kind}.v1",
        "frame_kind": frame_kind,
        "purpose": policy.purpose,
        "stable_contract_refs": [section for section in _STABLE_CONTRACT_REFS if section in included_sections],
        "included_sections": included_sections,
        "omitted_sections": [dict(item) for item in omitted_sections],
        "source_payloads": _source_payload_refs(section_values.get("source_payloads")),
        "source_payload_summaries": _source_payload_summaries(section_values.get("source_payloads")),
        "source_hashes": _source_hashes(section_values),
        "feedback_items": _feedback_items(frame_kind, section_values),
        "frame_contract": {
            "required_sections": list(policy.required_sections),
            "optional_sections": list(policy.optional_sections),
            "source_followup_enabled": False,
        },
        "budget_metadata": {
            "section_serialized_bytes": section_bytes,
            "estimated_prompt_bytes": estimated_prompt_bytes,
            "estimated_prompt_context_units": _estimate_context_units(estimated_prompt_bytes),
            "hard_limits": {
                "max_prompt_payload_bytes": MAX_AUTHORING_PROMPT_PAYLOAD_BYTES,
                "max_section_bytes": MAX_AUTHORING_SECTION_BYTES,
                "max_source_payload_bytes_phase0": SOURCE_PAYLOAD_BYTES_LIMIT_PHASE0,
            },
        },
    }
    assert_public_only_prompt_payload(frame)
    return frame


def with_final_prompt_budget(
    frame: Mapping[str, Any],
    *,
    prompt_payload: Mapping[str, Any],
    legacy_alias_sections: Sequence[str] = (),
) -> dict[str, Any]:
    final_frame = dict(frame)
    for _ in range(8):
        visible_sections = [
            section for section in prompt_payload if section != "authoring_context_frame"
        ]
        section_bytes = {
            section: _serialized_size(prompt_payload[section])
            for section in visible_sections
        }
        _raise_for_oversized_sections(section_bytes)
        legacy_aliases = [
            section for section in legacy_alias_sections if section in prompt_payload
        ]
        budget = dict(final_frame.get("budget_metadata", {}))
        budget["section_serialized_bytes"] = section_bytes
        budget["legacy_alias_section_bytes"] = {
            section: section_bytes[section] for section in legacy_aliases
        }
        estimated_prompt_bytes = sum(section_bytes.values())
        budget["estimated_prompt_bytes"] = estimated_prompt_bytes
        budget["estimated_prompt_context_units"] = _estimate_context_units(estimated_prompt_bytes)

        updated_frame = dict(final_frame)
        updated_frame["included_sections"] = visible_sections
        updated_frame["legacy_alias_sections"] = legacy_aliases
        updated_frame["budget_metadata"] = budget

        payload_with_frame = dict(prompt_payload)
        payload_with_frame["authoring_context_frame"] = updated_frame
        actual_bytes = _serialized_size(payload_with_frame)
        if actual_bytes > MAX_AUTHORING_PROMPT_PAYLOAD_BYTES:
            raise ValueError(
                "authoring prompt payload exceeds "
                f"{MAX_AUTHORING_PROMPT_PAYLOAD_BYTES} bytes after frame assembly: {actual_bytes}"
            )
        actual_units = _estimate_context_units(actual_bytes)
        budget["actual_prompt_payload_bytes"] = actual_bytes
        budget["actual_prompt_payload_context_units"] = actual_units
        candidate_frame = dict(updated_frame)
        candidate_frame["budget_metadata"] = budget
        if candidate_frame == final_frame:
            break
        final_frame = candidate_frame
    assert_public_only_prompt_payload(final_frame)
    return final_frame


def _require_sections(policy: AuthoringFramePolicy, included_sections: Sequence[str]) -> None:
    missing = [section for section in policy.required_sections if section not in included_sections]
    if missing:
        raise ValueError(f"{policy.frame_kind} authoring frame missing required sections: {missing}")


def _raise_for_oversized_sections(section_bytes: Mapping[str, int]) -> None:
    oversized = [
        section for section, size in section_bytes.items() if size > MAX_AUTHORING_SECTION_BYTES
    ]
    if oversized:
        raise ValueError(
            "authoring context section exceeds "
            f"{MAX_AUTHORING_SECTION_BYTES} bytes: {', '.join(sorted(oversized))}"
        )


def _feedback_items(frame_kind: str, section_values: Mapping[str, Any]) -> list[dict[str, str]]:
    feedback = section_values.get("provider_or_runtime_repair_feedback")
    if frame_kind != "static_repair" or not isinstance(feedback, Mapping):
        return []
    category = feedback.get("failure_category")
    if not isinstance(category, str):
        category = "unknown"
    return [{"section": "provider_or_runtime_repair_feedback", "failure_category": category}]


def _source_payload_refs(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    refs: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        ref = {
            "logical_id": item.get("logical_id"),
            "content_sha256": item.get("content_sha256"),
            "bytes": item.get("bytes"),
            "delivery": item.get("delivery"),
        }
        refs.append({key: item_value for key, item_value in ref.items() if item_value is not None})
    return refs


def _source_payload_summaries(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    summaries: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        summary = {
            "logical_id": item.get("logical_id"),
            "summary": item.get("summary"),
        }
        summaries.append({key: item_value for key, item_value in summary.items() if item_value is not None})
    return summaries


def _source_hashes(section_values: Mapping[str, Any]) -> list[dict[str, Any]]:
    hashes: list[dict[str, Any]] = []
    catalog = section_values.get("source_catalog")
    items = catalog.get("items") if isinstance(catalog, Mapping) else None
    if not isinstance(items, list):
        return hashes
    for item in items:
        if not isinstance(item, Mapping):
            continue
        digest = item.get("content_sha256")
        if not isinstance(digest, str) or not digest:
            continue
        hashes.append({"logical_id": item.get("logical_id"), "content_sha256": digest})
    return hashes


def _estimate_context_units(serialized_bytes: int) -> int:
    return max(1, (int(serialized_bytes) + _BYTES_PER_TOKEN_ESTIMATE - 1) // _BYTES_PER_TOKEN_ESTIMATE)


def _serialized_size(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8"))
