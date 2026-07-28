"""Stable public prompt payload assembly."""

from __future__ import annotations

from typing import Any, Mapping

from true_rl_agent.agent_core.context import (
    build_public_history_section,
    sanitize_prompt_public_context,
)
from true_rl_agent.agent_core.provider_contract import (
    candidate_package_files_contract,
    provider_json_output_contract,
)
from true_rl_agent.candidates.capability_profile import CandidateCapabilityProfile
from true_rl_agent.prompting.authoring_frame import (
    build_authoring_context_frame,
    compact_public_history_for_frame,
    normalize_authoring_frame_kind,
    with_final_prompt_budget,
)
from true_rl_agent.prompting.capabilities import (
    candidate_authoring_capabilities,
    candidate_component_contract_from_capabilities,
    control_effectiveness_contract_from_capabilities,
    prompt_visible_component_classes,
)
from true_rl_agent.prompting.public_safety import assert_public_only_prompt_payload
from true_rl_agent.prompting.serialization import (
    canonical_prompt_json,
    canonical_prompt_json_bytes,
)
from true_rl_agent.prompting.source_catalog import (
    PROMPT_SOURCE_SECTION_KEYS,
    default_prompt_source_sections,
    validate_prompt_source_sections,
)
from true_rl_agent.prompting.task_sections import sanitize_task_prompt_sections

PROMPT_PAYLOAD_SECTION_ORDER = (
    "task",
    "loop_contract",
    "required_candidate_package",
    "candidate_capabilities",
    "authoring_context_frame",
    "reward_inputs_contract",
    "public_eval_contract",
    "control_effectiveness_contract",
    "source_catalog",
    "candidate_lineage",
    "source_payloads",
    "public_feedback_summary",
    "runtime_feedback_summary",
    "public_history",
    "known_good_examples",
    "exploration_directive",
    "selection_guidance",
    "provider_or_runtime_repair_feedback",
    "schema_version",
)
LEGACY_PROMPT_PAYLOAD_ALIAS_SECTIONS = (
    "public_context",
)
MAX_LEGACY_PUBLIC_CONTEXT_ALIAS_BYTES = 1024


def serialize_prompt_payload(payload: Mapping[str, Any]) -> str:
    """Return the canonical provider-facing JSON text for a public prompt payload."""

    assert_public_only_prompt_payload(payload)
    return canonical_prompt_json(payload)


def reward_inputs_projection() -> dict[str, Any]:
    return {
        "schema_version": "true_rl_agent.reward_inputs.public.v1",
        "status": "legacy_alias_deprecated",
        "canonical_section": "reward_inputs_contract",
        "field_semantics_source": "task_owned_reward_inputs_contract",
        "fields": [],
    }


def candidate_allowed_controls() -> list[str]:
    return prompt_visible_component_classes()


def candidate_component_contract(
    *, candidate_capabilities: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    return candidate_component_contract_from_capabilities(candidate_capabilities)


def required_candidate_package(
    *, candidate_capabilities: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    advertised_components = (
        list(
            candidate_capabilities.get(
                "provider_declared_components_prompt_advertised",
                [],
            )
        )
        if isinstance(candidate_capabilities, Mapping)
        else None
    )
    return {
        "schema_version": "true_rl_agent.required_candidate_package.v1",
        "candidate_role": "candidate_package_only",
        "provider_json_output": provider_json_output_contract(
            advertised_declared_components=advertised_components,
        ),
        "candidate_files": candidate_package_files_contract(),
        "component_contract": candidate_component_contract(
            candidate_capabilities=candidate_capabilities
        ),
        "candidate_must_not_change": [
            "runner code",
            "runner",
            "eval protocol",
            "success metric",
            "training budget",
            "seed policy",
            "checkpoints",
            "PPO",
            "acceptance rules",
        ],
        "operation_diff_contract": {
            "materialized_by": "candidate_materializer",
            "llm_output_key_allowed": False,
            "derived_from": ["declared_changed_components"],
            "authority": "verifier-derived change set remains authoritative",
            "purpose": "public provenance for declared changed components, not a runner patch",
        },
        "diagnostics_contract": {
            "materialized_by": "candidate_materializer",
            "llm_output_key_allowed": False,
            "derived_from": ["provider_response_parse", "candidate_materialization_status"],
            "purpose": "public candidate-package materialization diagnostics; task-owned diagnostics use only prompt-visible capabilities",
        },
    }


def control_effectiveness_contract(
    *, candidate_capabilities: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    return control_effectiveness_contract_from_capabilities(candidate_capabilities)


def build_prompt_payload(
    *,
    public_context: dict[str, Any],
    frame_kind: str | None = None,
    runner_capability_readiness: Mapping[str, Mapping[str, Any]] | None = None,
    candidate_capability_profile: CandidateCapabilityProfile | None = None,
) -> dict[str, Any]:
    safe_context = sanitize_prompt_public_context(dict(public_context))
    authoring_frame_kind = normalize_authoring_frame_kind(frame_kind, safe_context)
    task_sections = _task_prompt_sections(safe_context.get("task_prompt_sections"))
    safe_context.pop("task_prompt_sections", None)
    source_sections = _source_prompt_sections(
        safe_context,
        allow_source_payload_subset=authoring_frame_kind == "source_followup",
    )
    for section in PROMPT_SOURCE_SECTION_KEYS:
        safe_context.pop(section, None)
    public_history, omitted_sections = compact_public_history_for_frame(
        _public_history(safe_context),
        frame_kind=authoring_frame_kind,
    )
    safe_context["recent_outcomes"] = public_history["recent_outcomes"]
    safe_context["public_history_limits"] = public_history["limits"]
    safe_context["public_history_omitted_round_count"] = public_history["omitted_round_count"]

    candidate_capabilities = candidate_authoring_capabilities(
        capability_profile=candidate_capability_profile,
        task_specific_metadata=task_sections.get("task_capability_metadata"),
        runner_capability_readiness=runner_capability_readiness,
    )
    section_values: dict[str, Any] = {
        "task": task_sections.get("task", _generic_task_section()),
        "loop_contract": _loop_contract(candidate_capabilities),
        "public_feedback_summary": task_sections.get(
            "public_feedback_summary", _generic_public_feedback_summary(safe_context)
        ),
        "runtime_feedback_summary": task_sections.get(
            "runtime_feedback_summary", _generic_runtime_feedback_summary(safe_context)
        ),
        "required_candidate_package": required_candidate_package(
            candidate_capabilities=candidate_capabilities
        ),
        "candidate_capabilities": candidate_capabilities,
        "reward_inputs_contract": task_sections.get(
            "reward_inputs_contract", _generic_reward_inputs_contract()
        ),
        "public_eval_contract": task_sections.get("public_eval_contract", _generic_public_eval_contract()),
        "control_effectiveness_contract": control_effectiveness_contract(
            candidate_capabilities=candidate_capabilities
        ),
        "source_catalog": source_sections["source_catalog"],
        "candidate_lineage": source_sections["candidate_lineage"],
        "source_payloads": source_sections["source_payloads"],
        "public_history": public_history,
        "known_good_examples": task_sections.get("known_good_examples", []),
        "exploration_directive": source_sections["exploration_directive"],
        "selection_guidance": _compose_selection_guidance(
            task_sections.get("selection_guidance")
        ),
        "provider_or_runtime_repair_feedback": safe_context.get("candidate_repair_feedback"),
        "schema_version": "true_rl_agent.prompt_payload.v2",
    }
    section_values["authoring_context_frame"] = build_authoring_context_frame(
        frame_kind=authoring_frame_kind,
        section_values=section_values,
        section_order=PROMPT_PAYLOAD_SECTION_ORDER,
        omitted_sections=omitted_sections,
    )
    payload = {
        section: section_values[section]
        for section in PROMPT_PAYLOAD_SECTION_ORDER
        if section != "provider_or_runtime_repair_feedback" or section_values[section] is not None
    }

    # Compatibility ref for older provider/test adapters that still inspect the
    # sanitized public context directly. Canonical contracts stay in the ordered
    # top-level sections above.
    payload["public_context"] = _legacy_public_context_alias(safe_context)
    payload["authoring_context_frame"] = with_final_prompt_budget(
        payload["authoring_context_frame"],
        prompt_payload=payload,
        legacy_alias_sections=LEGACY_PROMPT_PAYLOAD_ALIAS_SECTIONS,
    )
    assert_public_only_prompt_payload(payload)
    return payload


def _legacy_public_context_alias(public_context: Mapping[str, Any]) -> dict[str, Any]:
    alias: dict[str, Any] = {
        "schema_version": "true_rl_agent.public_context_legacy_summary.v1",
        "deprecated": True,
        "public_only": True,
        "previous_rounds": public_context.get("previous_rounds", 0),
        "best_public_score": public_context.get("best_public_score"),
        "last_outcome": public_context.get("last_outcome"),
        "pure_knob_streak": public_context.get("pure_knob_streak", 0),
        "canonical_sections": {
            "history": "public_history",
            "public_feedback": "public_feedback_summary",
            "runtime_feedback": "runtime_feedback_summary",
        },
    }
    if "candidate_repair_feedback" in public_context:
        alias["candidate_repair_feedback_ref"] = {
            "available": True,
            "canonical_section": "provider_or_runtime_repair_feedback",
        }
    clean = sanitize_prompt_public_context(alias)
    size = canonical_prompt_json_bytes(clean)
    if size >= MAX_LEGACY_PUBLIC_CONTEXT_ALIAS_BYTES:
        raise ValueError(
            "legacy public_context alias must stay below "
            f"{MAX_LEGACY_PUBLIC_CONTEXT_ALIAS_BYTES} bytes: {size}"
        )
    return clean


def _task_prompt_sections(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return sanitize_task_prompt_sections(value)


def _source_prompt_sections(
    public_context: Mapping[str, Any],
    *,
    allow_source_payload_subset: bool = False,
) -> dict[str, Any]:
    defaults = default_prompt_source_sections()
    sections = {
        key: public_context.get(key, defaults[key])
        for key in PROMPT_SOURCE_SECTION_KEYS
    }
    return validate_prompt_source_sections(
        sections,
        require_inline_payloads=not allow_source_payload_subset,
    )


def _generic_task_section() -> dict[str, Any]:
    return {
        "schema_version": "true_rl_agent.task_prompt.generic.v1",
        "task_id": "not_provided",
        "name": "not_provided_by_task_prompt_adapter",
        "prompt_adapter": "not_provided",
    }


def _loop_contract(candidate_capabilities: Mapping[str, Any]) -> dict[str, Any]:
    candidate_may: list[str] = []
    visible = candidate_capabilities.get("prompt_visible_capabilities", [])
    if isinstance(visible, list):
        for capability in visible:
            if not isinstance(capability, Mapping):
                continue
            actions = capability.get("candidate_allowed_actions", [])
            if not isinstance(actions, list):
                continue
            for action in actions:
                if isinstance(action, str) and action not in candidate_may:
                    candidate_may.append(action)
    return {
        "schema_version": "true_rl_agent.loop_contract.v1",
        "candidate_role": "Return a candidate package only.",
        "candidate_may": candidate_may,
        "runner_owned": [
            "training budget",
            "eval protocol",
            "seed policy",
            "checkpoint selection",
            "success metric",
            "accept/reject decision",
        ],
        "candidate_must_not_change": [
            "runner",
            "eval protocol",
            "training budget",
            "seed policy",
            "checkpoints",
            "PPO",
            "acceptance rules",
            "task invariants",
        ],
    }


def _generic_reward_inputs_contract() -> dict[str, Any]:
    return {
        "schema_version": "true_rl_agent.reward_inputs_contract.generic.v1",
        "status": "task_owned_contract_not_provided",
        "field_semantics_source": "task_prompt_adapter_required",
    }


def _generic_public_eval_contract() -> dict[str, Any]:
    return {
        "schema_version": "true_rl_agent.public_eval_contract.generic.v1",
        "status": "task_owned_contract_not_provided",
        "decision_authority": "runner_owned_read_only",
    }


def _generic_public_feedback_summary(public_context: Mapping[str, Any]) -> dict[str, Any]:
    summary = {
        "schema_version": "true_rl_agent.public_feedback_summary.generic.v1",
        "previous_rounds": public_context.get("previous_rounds", 0),
        "best_public_score": public_context.get("best_public_score"),
        "last_outcome": public_context.get("last_outcome"),
        "candidate_repair_feedback_available": "candidate_repair_feedback" in public_context,
    }
    task_summary = public_context.get("task_public_summary")
    if isinstance(task_summary, Mapping):
        summary["task_public_summary"] = sanitize_prompt_public_context(dict(task_summary))
    return summary


def _generic_runtime_feedback_summary(public_context: Mapping[str, Any]) -> dict[str, Any]:
    repair_feedback = public_context.get("candidate_repair_feedback")
    return {
        "schema_version": "true_rl_agent.runtime_feedback_summary.generic.v1",
        "status": "not_projected_yet",
        "repair_feedback_category": (
            repair_feedback.get("failure_category") if isinstance(repair_feedback, Mapping) else None
        ),
    }


def _public_history(public_context: Mapping[str, Any]) -> dict[str, Any]:
    return build_public_history_section(public_context)


def _generic_selection_guidance() -> list[str]:
    return [
        "Prefer code-level training mechanisms over parameter-only changes.",
        "Use only public prompt evidence.",
        "Do not modify runner, eval protocol, training budget, seed policy, checkpoints, PPO, or acceptance rules.",
    ]


def _compose_selection_guidance(task_guidance: Any) -> list[str]:
    """Combine task-owned semantics with task-neutral evidence discipline."""

    if task_guidance is None:
        extension = _generic_selection_guidance()
    elif isinstance(task_guidance, list | tuple) and all(
        isinstance(item, str) and item.strip() for item in task_guidance
    ):
        extension = list(task_guidance)
    else:
        raise TypeError("selection_guidance must be a sequence of non-empty strings")

    guidance = [
        (
            "Before authoring, compare the proposed mechanism with public_history.experiment_memory, "
            "recent_outcomes, and current-best evidence."
        ),
        (
            "Treat lower scores, failed gates, and dominating or negligible runtime component "
            "contributions as negative evidence; do not silently repeat or combine those mechanisms."
        ),
        (
            "If reusing a rejected mechanism, name the closest prior candidate in the rationale, "
            "state the exact code-level difference, and explain how it addresses the observed failure."
        ),
        (
            "Prefer the accepted current-best source, or when none exists the explicitly nonaccepted "
            "best-observed bootstrap anchor, plus one bounded, attributable change; combine multiple "
            "rejected mechanisms only when public evidence supports the combination."
        ),
        *extension,
    ]
    return list(dict.fromkeys(guidance))
