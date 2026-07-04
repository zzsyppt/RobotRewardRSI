"""Stable public prompt payload assembly."""

from __future__ import annotations

import json
from typing import Any, Mapping

from true_rl_agent.agent_core.context import build_public_history_section, sanitize_prompt_public_context
from true_rl_agent.agent_core.provider_contract import (
    candidate_package_files_contract,
    provider_json_output_contract,
)
from true_rl_agent.prompting.authoring_frame import (
    build_authoring_context_frame,
    compact_public_history_for_frame,
    normalize_authoring_frame_kind,
    with_final_prompt_budget,
)
from true_rl_agent.prompting.public_safety import assert_public_only_prompt_payload
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
    "reward_inputs",
    "candidate_allowed_controls",
    "candidate_component_contract",
    "anti_tuning_notice",
    "runner_owned_extensions",
)


def serialize_prompt_payload(payload: Mapping[str, Any]) -> str:
    """Return the canonical provider-facing JSON text for a public prompt payload."""

    assert_public_only_prompt_payload(payload)
    return json.dumps(payload, ensure_ascii=False, indent=2)


def reward_inputs_projection() -> dict[str, Any]:
    return {
        "schema_version": "true_rl_agent.reward_inputs.public.v1",
        "status": "legacy_alias_deprecated",
        "canonical_section": "reward_inputs_contract",
        "field_semantics_source": "task_owned_reward_inputs_contract",
        "fields": [],
    }


def candidate_allowed_controls() -> list[str]:
    return ["RewardTerm", "DiagnosticTerm", "ScheduleTerm"]


def candidate_component_contract() -> dict[str, Any]:
    return {
        "schema_version": "true_rl_agent.candidate_component_contract.v1",
        "module": "true_rl_agent.candidates.components",
        "entrypoint": "training_code.components:build_training_components",
        "builder_signature": "def build_training_components() -> TrainingComponents:",
        "allowed_import": (
            "from true_rl_agent.candidates.components import "
            "DiagnosticTerm, RewardInputs, RewardTerm, ScheduleTerm, TrainingComponents"
        ),
        "classes": {
            "RewardTerm": "RewardTerm(name: str, weight: float, fn: Callable[[RewardInputs], Any], clip_min: float | None = None, clip_max: float | None = None, description: str = '')",
            "DiagnosticTerm": "DiagnosticTerm(name: str, fn: Callable[[RewardInputs], Any], description: str = '')",
            "ScheduleTerm": "ScheduleTerm(name: str, key: str, value: float)",
            "TrainingComponents": "TrainingComponents(name: str, reward_terms: list[RewardTerm] = [], diagnostics: list[DiagnosticTerm] = [], schedules: list[ScheduleTerm] = [], metadata: dict = {}, rationale: str = '')",
        },
        "allowed_schedule_keys": ["reward_weight_scale"],
        "minimal_components_py": (
            "from __future__ import annotations\n\n"
            "from true_rl_agent.candidates.components import RewardInputs, RewardTerm, TrainingComponents\n\n\n"
            "def stance_margin(inp: RewardInputs):\n"
            "    return inp.base_height - 0.5 * (inp.roll * inp.roll + inp.pitch * inp.pitch)\n\n\n"
            "def build_training_components() -> TrainingComponents:\n"
            "    return TrainingComponents(\n"
            "        name=\"stance_margin_candidate\",\n"
            "        reward_terms=[RewardTerm(name=\"stance_margin\", weight=1.0, fn=stance_margin)],\n"
            "        diagnostics=[],\n"
            "        schedules=[],\n"
            "        rationale=\"Add a public reward callable over RewardInputs.\",\n"
            "    )\n"
        ),
        "forbidden_surfaces": [
            "runner state",
            "environment objects",
            "optimizer or PPO hyperparameters",
            "eval protocol",
            "success metric",
            "seeds",
            "budgets",
            "checkpoints",
            "filesystem",
        ],
    }


def required_candidate_package() -> dict[str, Any]:
    return {
        "schema_version": "true_rl_agent.required_candidate_package.v1",
        "candidate_role": "candidate_package_only",
        "provider_json_output": provider_json_output_contract(),
        "candidate_files": candidate_package_files_contract(),
        "component_contract": candidate_component_contract(),
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
            "materialized_by": "provider",
            "llm_output_key_allowed": False,
            "purpose": "public provenance for declared changed components, not a runner patch",
        },
        "diagnostics_contract": {
            "materialized_by": "provider",
            "llm_output_key_allowed": False,
            "purpose": "public candidate-package diagnostics placeholder; task diagnostics belong in components.py DiagnosticTerm",
        },
    }


def control_effectiveness_contract() -> dict[str, Any]:
    return {
        "schema_version": "true_rl_agent.control_effectiveness_contract.v1",
        "component_controls": {
            "RewardTerm": {
                "allowed": True,
                "effectiveness": "effective_now",
                "training_effect": "guarded vector output can be added to training reward",
            },
            "DiagnosticTerm": {
                "allowed": True,
                "effectiveness": "public_feedback_only",
                "training_effect": "summarized for future prompt feedback; not a decision rule",
            },
            "ScheduleTerm": {
                "allowed": True,
                "effectiveness": "key_specific",
                "training_effect": "only currently effective schedule keys alter training",
            },
        },
        "schedule_controls": {
            "allowed_keys": ["reward_weight_scale"],
            "effective_keys": ["reward_weight_scale"],
        },
        "runner_owned_extensions": {
            "observation_projection": "not_prompt_writable",
            "feature_projection": "not_prompt_writable",
            "curriculum": "not_prompt_writable_in_this_prompt_upgrade",
        },
    }


def build_prompt_payload(*, public_context: dict[str, Any], frame_kind: str | None = None) -> dict[str, Any]:
    safe_context = sanitize_prompt_public_context(dict(public_context))
    authoring_frame_kind = normalize_authoring_frame_kind(frame_kind, safe_context)
    task_sections = _task_prompt_sections(safe_context.get("task_prompt_sections"))
    safe_context.pop("task_prompt_sections", None)
    source_sections = _source_prompt_sections(safe_context)
    for section in PROMPT_SOURCE_SECTION_KEYS:
        safe_context.pop(section, None)
    public_history, omitted_sections = compact_public_history_for_frame(
        _public_history(safe_context),
        frame_kind=authoring_frame_kind,
    )
    safe_context["recent_outcomes"] = public_history["recent_outcomes"]
    safe_context["public_history_limits"] = public_history["limits"]
    safe_context["public_history_omitted_round_count"] = public_history["omitted_round_count"]

    section_values: dict[str, Any] = {
        "task": task_sections.get("task", _generic_task_section()),
        "loop_contract": _loop_contract(),
        "public_feedback_summary": task_sections.get(
            "public_feedback_summary", _generic_public_feedback_summary(safe_context)
        ),
        "runtime_feedback_summary": task_sections.get(
            "runtime_feedback_summary", _generic_runtime_feedback_summary(safe_context)
        ),
        "required_candidate_package": required_candidate_package(),
        "reward_inputs_contract": task_sections.get(
            "reward_inputs_contract", _generic_reward_inputs_contract()
        ),
        "public_eval_contract": task_sections.get("public_eval_contract", _generic_public_eval_contract()),
        "control_effectiveness_contract": control_effectiveness_contract(),
        "source_catalog": source_sections["source_catalog"],
        "candidate_lineage": source_sections["candidate_lineage"],
        "source_payloads": source_sections["source_payloads"],
        "public_history": public_history,
        "known_good_examples": task_sections.get("known_good_examples", []),
        "exploration_directive": source_sections["exploration_directive"],
        "selection_guidance": task_sections.get("selection_guidance", _generic_selection_guidance()),
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

    # Transitional aliases for current providers/tests while Step5+ consumers move
    # to the canonical sections above.
    payload["public_context"] = safe_context
    payload["reward_inputs"] = reward_inputs_projection()
    payload["candidate_allowed_controls"] = candidate_allowed_controls()
    payload["candidate_component_contract"] = payload["required_candidate_package"]["component_contract"]
    payload["anti_tuning_notice"] = (
        "Changing only reward_weight, terrain_range, threshold, seed, or budget "
        "is parameter_only and cannot be a policy improvement accept."
    )
    payload["runner_owned_extensions"] = payload["control_effectiveness_contract"]["runner_owned_extensions"]
    payload["authoring_context_frame"] = with_final_prompt_budget(
        payload["authoring_context_frame"],
        prompt_payload=payload,
        legacy_alias_sections=LEGACY_PROMPT_PAYLOAD_ALIAS_SECTIONS,
    )
    assert_public_only_prompt_payload(payload)
    return payload


def _task_prompt_sections(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return sanitize_task_prompt_sections(value)


def _source_prompt_sections(public_context: Mapping[str, Any]) -> dict[str, Any]:
    defaults = default_prompt_source_sections()
    sections = {
        key: public_context.get(key, defaults[key])
        for key in PROMPT_SOURCE_SECTION_KEYS
    }
    return validate_prompt_source_sections(sections)


def _generic_task_section() -> dict[str, Any]:
    return {
        "schema_version": "true_rl_agent.task_prompt.generic.v1",
        "task_id": "not_provided",
        "name": "not_provided_by_task_prompt_adapter",
        "prompt_adapter": "not_provided",
    }


def _loop_contract() -> dict[str, Any]:
    return {
        "schema_version": "true_rl_agent.loop_contract.v1",
        "candidate_role": "Return a candidate package only.",
        "candidate_may": [
            "define RewardTerm callables over prompt-visible RewardInputs",
            "define DiagnosticTerm callables over prompt-visible RewardInputs",
            "use ScheduleTerm only with currently effective allowed keys",
        ],
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
    return {
        "schema_version": "true_rl_agent.public_feedback_summary.generic.v1",
        "previous_rounds": public_context.get("previous_rounds", 0),
        "best_public_score": public_context.get("best_public_score"),
        "last_outcome": public_context.get("last_outcome"),
        "candidate_repair_feedback_available": "candidate_repair_feedback" in public_context,
    }


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
