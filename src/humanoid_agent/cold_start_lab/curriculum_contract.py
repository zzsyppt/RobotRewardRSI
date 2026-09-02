"""Prompt-facing curriculum contracts for Phase 8E."""

from __future__ import annotations

import copy
from typing import Any


CURRICULUM_REGISTRY_SCHEMA = "humanoid_agent.phase8e.curriculum_capability_registry.v1"
CURRICULUM_EVIDENCE_SCHEMA = "humanoid_agent.phase8e.public_curriculum_evidence.v1"
CURRICULUM_DESIGN_CONTRACT_SCHEMA = "humanoid_agent.phase8e.curriculum_design_contract.v1"
CURRICULUM_STAGE_CAPABILITY_IDS = (
    "external_push_curriculum",
    "train_only_dr_scale",
    "reward_only_push_signal",
    "recovery_window_diagnostic",
    "public_push_probe",
)
CURRICULUM_ALLOWED_CAPABILITY_IDS = ("reward_weight_scale", *CURRICULUM_STAGE_CAPABILITY_IDS)


class CurriculumContractError(ValueError):
    """Raised when an LLM-authored curriculum artifact violates the V1 contract."""


def build_v1_curriculum_capability_registry() -> dict[str, Any]:
    """Return the fixed V1 registry approved by runner-side code."""

    capabilities: list[dict[str, Any]] = [
        {
            "id": "reward_weight_scale",
            "kind": "currently_effective_schedule_key",
            "candidate_recipe_surface": True,
            "runner_execution_owner": True,
            "status": "available_now",
            "purpose": "allow stage-specific scaling of candidate reward terms",
        },
        {
            "id": "external_push_curriculum",
            "kind": "train_only_intervention",
            "candidate_recipe_surface": False,
            "runner_execution_owner": True,
            "status": "preapproved_v1_runner_capability",
            "purpose": "let curriculum design reason about train-only perturbation progression",
        },
        {
            "id": "train_only_dr_scale",
            "kind": "train_only_intervention",
            "candidate_recipe_surface": False,
            "runner_execution_owner": True,
            "status": "preapproved_v1_runner_capability",
            "purpose": "let curriculum design reason about train-only robustness progression",
        },
        {
            "id": "reward_only_push_signal",
            "kind": "reward_only_signal",
            "candidate_recipe_surface": True,
            "runner_execution_owner": True,
            "status": "available_now",
            "purpose": "represent current-step push information as prompt-visible reward-only training signal",
        },
        {
            "id": "recovery_window_diagnostic",
            "kind": "public_diagnostic",
            "candidate_recipe_surface": False,
            "runner_execution_owner": True,
            "status": "report_only_not_decision_metric",
            "purpose": "surface recovery quality without changing pass/fail metrics",
        },
        {
            "id": "public_push_probe",
            "kind": "public_probe",
            "candidate_recipe_surface": False,
            "runner_execution_owner": True,
            "status": "public_dev_evidence_only",
            "purpose": "provide public training evidence for curriculum reasoning",
        },
    ]
    return {
        "schema_version": CURRICULUM_REGISTRY_SCHEMA,
        "version": "v1-fixed-human-runner-approved",
        "first_version_allows_capability_requests": False,
        "capabilities": capabilities,
    }


def build_public_curriculum_evidence_bundle(
    *,
    task: dict[str, Any],
    public_feedback_summary: dict[str, Any],
    reward_inputs_contract: dict[str, Any],
    control_effectiveness_contract: dict[str, Any],
    current_recipe: dict[str, Any] | None,
) -> dict[str, Any]:
    """Project public evidence the LLM may use for curriculum reasoning."""

    schedule_controls = control_effectiveness_contract.get("schedule_controls", {})
    effective_schedule_keys = schedule_controls.get("effective_keys", {})
    return {
        "schema_version": CURRICULUM_EVIDENCE_SCHEMA,
        "role": "prompt_facing_public_evidence_for_curriculum_design",
        "task_ref": "task",
        "objective_ref": "task.objective" if task else None,
        "public_feedback_summary": copy.deepcopy(public_feedback_summary),
        "prompt_visible_reward_inputs": list(reward_inputs_contract.get("field_order", [])),
        "effective_schedule_keys": list(effective_schedule_keys.keys()) if isinstance(effective_schedule_keys, dict) else [],
        "current_recipe_present": current_recipe is not None,
        "capability_registry": build_v1_curriculum_capability_registry(),
        "probe_visibility_policy": {
            "public_dev_probes": "prompt_visible_when_projected_by_runner",
            "public_accept_probes": "withheld_from_prompt",
            "reserved_audit_probes": "never_prompt_visible",
        },
        "forbidden_evidence": [
            "raw evaluation seed identifiers",
            "private evaluation internals",
            "private server paths",
            "runner decision internals",
            "candidate claimed budget",
        ],
    }


def build_curriculum_design_contract() -> dict[str, Any]:
    """Return the V1 curriculum-design response contract."""

    return {
        "schema_version": CURRICULUM_DESIGN_CONTRACT_SCHEMA,
        "first_version_scope": "design_with_fixed_registry_no_capability_requests",
        "response_location": {
            "curriculum_hypothesis": "diagnostics.json.curriculum_hypothesis",
            "curriculum_design": "operation_diff.json.curriculum_design",
        },
        "artifact_shape": {
            "diagnostics_curriculum_hypothesis_type": "object",
            "operation_diff_curriculum_design_type": "object",
            "stage_list_location": "operation_diff.json.curriculum_design.stage_list",
        },
        "stage_schema": {
            "required_fields": ["name", "progress_range", "capabilities", "rationale"],
            "progress_range": "two numeric values [start, end] inside [0, 1]",
            "progress_coverage": "stage ranges must start at 0.0, be contiguous with no gaps, and end at 1.0",
            "capabilities": {
                "type": "list[string]",
                "must_use_exact_registry_ids": True,
                "must_include_one_of": list(CURRICULUM_STAGE_CAPABILITY_IDS),
                "do_not_invent_ids": True,
                "example": ["external_push_curriculum", "train_only_dr_scale"],
            },
            "forbidden_stage_shape": "do not describe curriculum only in prose; encode stage_list objects with exact capabilities",
        },
        "final_consolidation_schema": {
            "required_fields": [
                "fraction",
                "distribution",
                "includes_target_push",
                "includes_target_robustness_variation",
            ],
            "accepted_aliases": {
                "fraction": ["min_fraction", "min_fraction_of_training"],
                "includes_target_push": ["target_push_included", "must_include_target_push"],
                "includes_target_robustness_variation": [
                    "target_robustness_included",
                    "must_include_target_robustness_variation",
                ],
            },
            "distribution_required_value": "runner_owned_target_distribution",
        },
        "required_sections": [
            "curriculum_hypothesis",
            "curriculum_design",
            "final_consolidation",
            "anti_tuning_rationale",
            "improvement_attribution_intent",
        ],
        "final_consolidation_requirements": {
            "min_fraction_of_training": 0.2,
            "distribution": "runner_owned_target_distribution",
            "must_include_target_push": True,
            "must_include_target_robustness_variation": True,
        },
        "non_degenerate_design_requirements": {
            "min_stage_count": 2,
            "must_use_curriculum_capability_beyond_reward_weight_scale": True,
            "reject_reward_only_attribution": True,
            "response_must_explain_stage_progression": True,
        },
        "degenerate_curriculum_reject_rules": [
            "reject all-no-push or long-running below-floor push curriculum",
            "reject single-stage or no-op curricula that do not change training distribution over progress",
            "reject reward-only attribution that says improvement is not due to curriculum change",
            "reject final stage shorter than required final_consolidation.min_fraction_of_training",
            "reject curricula that never return to runner-owned target distribution",
            "reject unregistered capability usage",
            "reject curriculum changes that alter budget seed evaluation metric or checkpoint selection",
            "reject recovery-window diagnostics as pass/fail metrics in V1",
        ],
        "capability_request_policy": {
            "allowed_in_v1": False,
            "next_version_requires_typed_templates": True,
        },
    }


def validate_v1_curriculum_candidate_artifacts(candidate_projection: dict[str, Any]) -> None:
    """Validate public candidate artifacts against the V1 curriculum contract."""

    operation_diff = _mapping(candidate_projection.get("operation_diff"), "operation_diff.json")
    diagnostics = _mapping(candidate_projection.get("diagnostics"), "diagnostics.json")
    design = _mapping(operation_diff.get("curriculum_design"), "operation_diff.json.curriculum_design")
    hypothesis = _mapping(diagnostics.get("curriculum_hypothesis"), "diagnostics.json.curriculum_hypothesis")

    stages = _stage_list(design)
    if len(stages) < 2:
        raise CurriculumContractError("operation_diff.json.curriculum_design.stage_list must contain at least two stages")
    _validate_progress_ranges(stages)
    _validate_stage_capabilities(stages)
    _validate_final_consolidation(design)
    _validate_curriculum_attribution(design, hypothesis)


def _stage_list(design: dict[str, Any]) -> list[dict[str, Any]]:
    raw = design.get("stage_list", design.get("stages"))
    if not isinstance(raw, list):
        raise CurriculumContractError("operation_diff.json.curriculum_design.stage_list is required")
    stages = [stage for stage in raw if isinstance(stage, dict)]
    if len(stages) != len(raw):
        raise CurriculumContractError("operation_diff.json.curriculum_design.stage_list entries must be objects")
    return stages


def _validate_progress_ranges(stages: list[dict[str, Any]]) -> None:
    previous_end = 0.0
    for index, stage in enumerate(stages):
        raw_range = stage.get("progress_range")
        if not isinstance(raw_range, list | tuple) or len(raw_range) != 2:
            raise CurriculumContractError(f"stage_list[{index}].progress_range must be [start, end]")
        try:
            start = float(raw_range[0])
            end = float(raw_range[1])
        except (TypeError, ValueError) as exc:
            raise CurriculumContractError(f"stage_list[{index}].progress_range must be numeric") from exc
        if start < 0.0 or end > 1.0 or start >= end:
            raise CurriculumContractError(f"stage_list[{index}].progress_range must stay within [0, 1] and increase")
        if abs(start - previous_end) > 1e-6:
            raise CurriculumContractError("stage_list progress ranges must cover [0, 1] contiguously with no gaps or overlaps")
        previous_end = end
    if abs(previous_end - 1.0) > 1e-6:
        raise CurriculumContractError("stage_list progress ranges must end at 1.0")


def _validate_stage_capabilities(stages: list[dict[str, Any]]) -> None:
    allowed = set(CURRICULUM_ALLOWED_CAPABILITY_IDS)
    required_beyond_reward = set(CURRICULUM_STAGE_CAPABILITY_IDS)
    uses_curriculum_capability = False
    for index, stage in enumerate(stages):
        raw = stage.get("capabilities")
        if not isinstance(raw, list):
            raise CurriculumContractError(f"stage_list[{index}].capabilities must be list[str] of fixed registry ids")
        if not raw:
            raise CurriculumContractError(f"stage_list[{index}].capabilities must include at least one fixed registry id")
        for item in raw:
            if not isinstance(item, str):
                raise CurriculumContractError(f"stage_list[{index}].capabilities must be list[str] of fixed registry ids")
            if item not in allowed:
                raise CurriculumContractError(
                    f"stage_list[{index}].capabilities contains unregistered capability id: {item}"
                )
            if item in required_beyond_reward:
                uses_curriculum_capability = True
    if not uses_curriculum_capability:
        raise CurriculumContractError(
            "curriculum_design must use at least one fixed V1 curriculum capability beyond reward_weight_scale"
        )


def _validate_final_consolidation(design: dict[str, Any]) -> None:
    final = _mapping(design.get("final_consolidation"), "curriculum_design.final_consolidation")
    try:
        fraction = float(_first_present(final, ("fraction", "min_fraction", "min_fraction_of_training"), 0.0))
    except (TypeError, ValueError) as exc:
        raise CurriculumContractError("curriculum_design.final_consolidation fraction must be numeric") from exc
    if fraction < 0.2:
        raise CurriculumContractError("curriculum_design.final_consolidation must reserve at least 20 percent")
    if final.get("distribution") != "runner_owned_target_distribution":
        raise CurriculumContractError("curriculum_design.final_consolidation must return to target distribution")
    if _first_present(final, ("includes_target_push", "target_push_included", "must_include_target_push"), False) is not True:
        raise CurriculumContractError("curriculum_design.final_consolidation must include target push")
    if (
        _first_present(
            final,
            (
                "includes_target_robustness_variation",
                "target_robustness_included",
                "must_include_target_robustness_variation",
            ),
            False,
        )
        is not True
    ):
        raise CurriculumContractError("curriculum_design.final_consolidation must include target robustness variation")


def _validate_curriculum_attribution(design: dict[str, Any], hypothesis: dict[str, Any]) -> None:
    text = " ".join(
        str(item)
        for item in (
            design.get("improvement_attribution_intent"),
            design.get("anti_tuning_rationale"),
            hypothesis.get("improvement_attribution_intent"),
            hypothesis.get("anti_tuning_rationale"),
        )
        if item is not None
    ).lower()
    disallowed = (
        "not to any curriculum",
        "not to curriculum",
        "not curriculum",
        "not to any curriculum change",
        "reward terms, not",
        "only reward",
    )
    if any(pattern in text for pattern in disallowed):
        raise CurriculumContractError("curriculum attribution must not disclaim curriculum contribution")


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CurriculumContractError(f"{label} must be an object")
    return value


def _first_present(mapping: dict[str, Any], keys: tuple[str, ...], default: Any) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default
