"""Phase 8E prompt-package construction."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from humanoid_agent.code_sandbox.loader import candidate_manifest_prompt_contract
from humanoid_agent.cold_start_lab.curriculum_contract import (
    build_curriculum_design_contract,
    build_public_curriculum_evidence_bundle,
)
from humanoid_agent.cold_start_lab.phase8e_common import (
    assert_prompt_facing_payload,
    assert_public_only_payload,
    provider_source_read_enabled,
    provider_source_read_turn_budget,
)
from humanoid_agent.cold_start_lab.phase8e_projection import (
    build_public_history_summary,
    latest_accepted_record,
    project_ledger_record,
    prompt_facing_candidate_package,
)
from humanoid_agent.cold_start_lab.phase8e_source_pack import (
    ALLOWED_TOP_LEVEL_CANDIDATE_FILES,
    build_minimal_source_pack,
    build_source_access_policy,
    known_good_reward_examples_metadata,
)
from humanoid_agent.cold_start_lab.training_adapter import (
    control_effectiveness_contract,
    reward_inputs_contract,
    reward_inputs_field_expansion_report,
)


TRAINING_COMPONENT_FIELDS = (
    "name",
    "reward_terms",
    "loss_terms",
    "schedules",
    "train_observation_transforms",
    "diagnostics",
    "metadata",
    "rationale",
)
NAMED_COMPONENT_COLLECTION_FIELDS = (
    "reward_terms",
    "loss_terms",
    "schedules",
    "diagnostics",
)
PROMPT_FACING_TRAINING_COMPONENT_FIELDS = (
    "name",
    "reward_terms",
    "schedules",
    "diagnostics",
    "metadata",
    "rationale",
)
PROMPT_FACING_NAMED_COMPONENT_COLLECTION_FIELDS = (
    "reward_terms",
    "schedules",
    "diagnostics",
)
PROMPT_FACING_COMPONENT_CLASSES_TO_IMPORT = (
    "RewardInputs",
    "RewardTerm",
    "ScheduleTerm",
    "DiagnosticTerm",
    "TrainingComponents",
)
CANDIDATE_ENTRYPOINT = "training_code.components:build_training_components"
CANDIDATE_PHASE = "8E"
PROMPT_PRIMARY_MECHANISM_REQUIRED = (
    "candidate_must_include_runner_compiled_curriculum_design_as_primary_training_surface"
)
MINIMAL_COMPONENT_TEMPLATE = """from __future__ import annotations

from humanoid_agent.code_sandbox.components import RewardInputs, RewardTerm, TrainingComponents


def example_reward(inp: RewardInputs):
    return inp.base_height


def build_training_components(task_spec, base_env_cfg, base_agent_cfg) -> TrainingComponents:
    _ = (task_spec, base_env_cfg, base_agent_cfg)
    return TrainingComponents(
        name="phase8e_candidate",
        reward_terms=[
            RewardTerm(name="example_reward", fn=example_reward, weight=1.0),
        ],
        schedules=[],
        diagnostics=[],
        rationale="Describe the public training-side change here.",
    )
"""


def _reward_input_prompt_visible(field_meta: dict[str, Any]) -> bool:
    return (
        str(field_meta.get("provenance", "")).lower() != "not_instrumented"
        and str(field_meta.get("source_status", "")).lower() != "not_instrumented"
    )


def prompt_facing_reward_inputs_contract(contract: dict[str, Any]) -> dict[str, Any]:
    """Filter the full runner contract to currently usable prompt-facing fields."""

    fields = contract.get("fields", {}) if isinstance(contract.get("fields"), dict) else {}
    visible_names = [
        name
        for name in contract.get("field_order", [])
        if isinstance(fields.get(name), dict) and _reward_input_prompt_visible(fields[name])
    ]
    return {
        **{key: copy.deepcopy(value) for key, value in contract.items() if key not in {"field_order", "fields"}},
        "field_order": visible_names,
        "fields": {name: copy.deepcopy(fields[name]) for name in visible_names},
    }


def prompt_facing_reward_inputs_field_expansion_report(
    report: dict[str, Any],
    *,
    visible_fields: list[str],
) -> dict[str, Any]:
    visible = set(visible_fields)
    return {
        **{
            key: copy.deepcopy(value)
            for key, value in report.items()
            if key
            not in {
                "baseline_fields",
                "first_batch_fields",
                "implemented_public_fields",
                "planned_public_fields",
                "field_shapes",
                "field_status",
            }
        },
        "baseline_fields": [name for name in report.get("baseline_fields", []) if name in visible],
        "first_batch_fields": [name for name in report.get("first_batch_fields", []) if name in visible],
        "implemented_public_fields": [name for name in report.get("implemented_public_fields", []) if name in visible],
        "planned_public_fields": [],
        "field_shapes": {
            name: copy.deepcopy(shape)
            for name, shape in report.get("field_shapes", {}).items()
            if name in visible
        },
        "field_status": {
            name: status
            for name, status in report.get("field_status", {}).items()
            if name in visible
        },
    }


def prompt_facing_control_effectiveness_contract(contract: dict[str, Any]) -> dict[str, Any]:
    """Expose prompt-facing runner controls that are active in this runner."""

    schedule_controls = contract.get("schedule_controls", {}) if isinstance(contract.get("schedule_controls"), dict) else {}
    effective_keys = copy.deepcopy(schedule_controls.get("effective_keys", {}))
    component_controls = copy.deepcopy(contract.get("component_controls", {}))
    component_controls.pop("loss_terms", None)
    return {
        "schema_version": contract.get("schema_version", "humanoid_agent.phase8e.control_effectiveness.v1"),
        "component_controls": component_controls,
        "schedule_controls": {
            "allowed_keys": list(effective_keys),
            "effective_keys": effective_keys,
        },
        "curriculum_controls": {
            "surface": "operation_diff.json.curriculum_design",
            "contract_ref": "curriculum_design_contract",
            "execution_owner": "runner_curriculum_compiler_and_training_adapter",
            "candidate_role": "design_stage_structure_and_select_approved_capability_ids",
            "allowed_capability_source": "public_curriculum_evidence_bundle.capability_registry",
        },
        "primary_mechanism_required": PROMPT_PRIMARY_MECHANISM_REQUIRED,
        "runtime_evidence": {
            "source": "Phase8ETrainingAdapter.public_summary.runtime_contract",
            "reward_weight_scale_behavior_probe": contract.get("runtime_evidence", {}).get(
                "reward_weight_scale_behavior_probe"
            ),
        },
        "guidance": [
            "CurriculumDesign in operation_diff.json is a first-class runner-compiled training surface; design stage structure and approved capability usage before reward repairs.",
            "RewardTerm, ScheduleTerm, and DiagnosticTerm remain allowed component surfaces; reward_weight_scale is auxiliary and not a substitute for curriculum design.",
        ],
    }


def prompt_facing_candidate_allowed_controls(control_contract: dict[str, Any]) -> list[str]:
    schedule_keys = control_contract.get("schedule_controls", {}).get("allowed_keys", [])
    return [
        "CurriculumDesign in operation_diff.json over approved capability_registry ids; runner compiles, clamps, and executes it as the primary train-only curriculum surface",
        "RewardTerm callables over prompt-visible public RewardInputs",
        f"ScheduleTerm callables for auxiliary currently effective runner keys: {', '.join(schedule_keys)}",
        "DiagnosticTerm callables over prompt-visible public RewardInputs returning numeric public summaries",
    ]


def _provider_json_output_contract(
    *,
    source_read_enabled: bool,
    manifest_template: str,
    max_source_read_turns: int,
) -> dict[str, Any]:
    contract: dict[str, Any] = {
        "allowed_first_response_modes": ["candidate_package"],
        "format": {
            "candidate_package": {
                "files": {
                    "candidate_manifest.yaml": manifest_template,
                    "training_code/components.py": "complete Python source text",
                    "rationale.md": "public rationale text",
                    "operation_diff.json": "public semantic diff JSON text",
                    "diagnostics.json": "public diagnostics JSON text",
                }
            }
        },
        "response_skeleton": {
            "candidate_package": {
                "files": {
                    "candidate_manifest.yaml": manifest_template,
                    "training_code/components.py": "...",
                    "rationale.md": "...",
                    "operation_diff.json": "...",
                    "diagnostics.json": "...",
                }
            }
        },
        "rules": [
            "Return one JSON object only; no prose outside JSON.",
            "All file paths must be relative POSIX paths under the allowed top-level entries.",
            "training_code/components.py must define the entrypoint with the exact builder_signature from component_contract.",
            "operation_diff.json must include a curriculum_design object whose stages and capabilities are the primary training proposal.",
            "Do not present reward-only or reward_weight_scale-only tweaks as curriculum improvements.",
            "Do not include credentials, hidden/IID identifiers, eval seed numbers, or runner path requests.",
        ],
    }
    if source_read_enabled:
        contract["allowed_first_response_modes"] = ["source_requests", "candidate_package"]
        contract["source_requests"] = {
            "enabled": True,
            "max_source_read_turns": max_source_read_turns,
            "shape": {
                "source_requests": [
                    {
                        "id": "examples/known_good_reward_terms.py",
                        "reason": "need a runnable import and builder skeleton",
                    }
                ]
            },
            "rules": [
                f"Return a source_requests object only when you need additional audited source text before authoring the candidate; runtime will serve at most {max_source_read_turns} source-read turn(s).",
                "Each requested id must come from source_access_policy.allowed_logical_ids.",
                "Do not mix source_requests and candidate_package in the same JSON object.",
                "Do not spend a source-read turn unless the candidate would be materially better with that audited source.",
            ],
        }
    else:
        contract["source_requests"] = {
            "enabled": False,
            "rules": ["Source-read is disabled in this author call."],
        }
    return contract


def build_candidate_self_audit_checklist(
    *,
    reward_contract: dict[str, Any],
    control_contract: dict[str, Any],
    allowed_top_level_files: list[str],
    current_recipe_present: bool,
    accepted_recipe_available: bool,
    training_component_fields: tuple[str, ...] = TRAINING_COMPONENT_FIELDS,
    named_component_collection_fields: tuple[str, ...] = NAMED_COMPONENT_COLLECTION_FIELDS,
) -> dict[str, Any]:
    if accepted_recipe_available:
        strategy_rule = (
            "Prefer a small, testable change from current best unless public feedback summary "
            "justifies a larger change."
        )
    elif current_recipe_present:
        strategy_rule = (
            "No accepted LLM recipe exists yet. Treat current_recipe as a bootstrap scaffold, not current best. "
            "Use the public contract, inline component template, and known-good examples to produce a complete verifier-safe candidate."
        )
    else:
        strategy_rule = (
            "No current recipe is available. Use the public contract, inline component template, and known-good examples "
            "to produce a complete verifier-safe candidate."
        )
    component_output_contracts = [
        {"component": "RewardTerm", "required_output": "public vector[num_envs]"},
        {
            "component": "DiagnosticTerm",
            "required_output": "public numeric/bool scalar or vector[num_envs] summary",
        },
    ]
    return {
        "schema_version": "humanoid_agent.phase8e.candidate_self_audit_checklist.v1",
        "role": "guidance_only_not_trusted_for_acceptance",
        "enforcement": "runner_verifier_and_sandbox_remain_authoritative",
        "candidate_must_not_return_self_certification": True,
        "items": [
            {
                "id": "curriculum_design_is_primary_surface",
                "derived_from": {
                    "curriculum_contract": "curriculum_design_contract",
                    "operation_diff": "required_candidate_package.provider_json_output.format.candidate_package.files.operation_diff.json",
                },
                "rule": (
                    "operation_diff.json must contain a curriculum_design using approved capability ids; "
                    "component code may support it but must not replace it as the claimed curriculum mechanism."
                ),
            },
            {
                "id": "component_outputs_match_public_contract",
                "derived_from": {
                    "reward_inputs_field_order": (
                        "required_candidate_package.component_contract.reward_inputs_contract.field_order"
                    ),
                    "component_specs": "required_candidate_package.component_contract",
                },
                "reward_inputs_field_order": list(reward_contract.get("field_order", [])),
                "component_output_contracts": component_output_contracts,
            },
            {
                "id": "component_names_unique",
                "derived_from": {
                    "component_families": "required_candidate_package.component_contract.training_components_fields"
                },
                "component_families": [
                    field for field in named_component_collection_fields if field in training_component_fields
                ],
            },
            {
                "id": "no_unsupported_imports_or_runner_hooks",
                "derived_from": {
                    "imports": "required_candidate_package.component_contract.component_classes_to_import",
                    "blocked_surfaces": "required_candidate_package.component_contract.not_available_to_candidate",
                },
                "rule": "Use only public component APIs and do not call runner-owned hooks or objects.",
            },
            {
                "id": "no_private_eval_or_metric_dependency",
                "derived_from": {
                    "blocked_surfaces": "required_candidate_package.component_contract.not_available_to_candidate",
                    "loop_forbidden": "loop_contract.candidate_forbidden",
                },
                "rule": "Do not depend on private evaluation details, benchmark identifiers, or decision metrics.",
            },
            {
                "id": "json_output_matches_allowed_files",
                "derived_from": {
                    "allowed_top_level_files": "required_candidate_package.allowed_top_level_files",
                    "provider_json_output": "required_candidate_package.provider_json_output",
                },
                "allowed_top_level_files": list(allowed_top_level_files),
                "rule": "Return exactly the requested JSON object with candidate package files only.",
            },
            {
                "id": "small_step_from_current_best_when_available",
                "derived_from": {
                    "current_recipe": "current_recipe",
                    "public_feedback_summary": "public_feedback_summary",
                    "known_good_examples": "known_good_reward_examples",
                },
                "current_recipe_present": current_recipe_present,
                "accepted_recipe_available": accepted_recipe_available,
                "rule": strategy_rule,
            },
        ],
    }


def build_phase8e_prompt_package(
    *,
    config: dict[str, Any],
    ledger: list[dict[str, Any]],
    current_recipe_dir: Path | None,
) -> dict[str, Any]:
    prompt_cfg = config.get("prompt", {})
    history_rounds = int(prompt_cfg.get("include_history_rounds", 5))
    source_read_turn_budget = provider_source_read_turn_budget(config)
    source_read_enabled = provider_source_read_enabled(config)
    manifest_contract = candidate_manifest_prompt_contract(
        required_entrypoint=CANDIDATE_ENTRYPOINT,
        phase=CANDIDATE_PHASE,
    )
    accepted_record = latest_accepted_record(ledger)
    accepted_recipe_available = accepted_record is not None
    current_recipe = prompt_facing_candidate_package(current_recipe_dir) if current_recipe_dir is not None else None
    if isinstance(current_recipe, dict):
        current_recipe["source_item_id"] = "current_best/training_code/components.py"
        current_recipe["accepted_history_available"] = accepted_recipe_available
        if accepted_recipe_available:
            current_recipe["baseline_role"] = "accepted_current_best"
        else:
            current_recipe["baseline_role"] = "bootstrap_scaffold_not_current_best"
            current_recipe["baseline_guidance"] = (
                "No accepted LLM recipe yet. This initial recipe is a bootstrap scaffold, not a performance baseline."
            )
    full_reward_contract = reward_inputs_contract()
    reward_contract = prompt_facing_reward_inputs_contract(full_reward_contract)
    full_field_expansion_report = reward_inputs_field_expansion_report()
    field_expansion_report = prompt_facing_reward_inputs_field_expansion_report(
        full_field_expansion_report,
        visible_fields=list(reward_contract["field_order"]),
    )
    full_control_contract = control_effectiveness_contract()
    control_contract = prompt_facing_control_effectiveness_contract(full_control_contract)
    known_good_examples = known_good_reward_examples_metadata(source_read_enabled=source_read_enabled)
    allowed_top_level_files = list(ALLOWED_TOP_LEVEL_CANDIDATE_FILES)
    candidate_checklist = build_candidate_self_audit_checklist(
        reward_contract=reward_contract,
        control_contract=control_contract,
        allowed_top_level_files=allowed_top_level_files,
        current_recipe_present=current_recipe is not None,
        accepted_recipe_available=accepted_recipe_available,
        training_component_fields=PROMPT_FACING_TRAINING_COMPONENT_FIELDS,
        named_component_collection_fields=PROMPT_FACING_NAMED_COMPONENT_COLLECTION_FIELDS,
    )
    source_bundle = build_minimal_source_pack(
        config=config,
        ledger=ledger,
        current_recipe_dir=current_recipe_dir,
        reward_contract=reward_contract,
        field_expansion_report=field_expansion_report,
        control_contract=control_contract,
        source_read_enabled=source_read_enabled,
    )
    source_pack = source_bundle["prompt_source_pack"]
    source_store = source_bundle["source_store"]
    public_history = [project_ledger_record(item) for item in ledger[-history_rounds:]]
    public_feedback_summary = build_public_history_summary(public_history)
    public_curriculum_evidence_bundle = build_public_curriculum_evidence_bundle(
        task=config.get("task", {}),
        public_feedback_summary=public_feedback_summary,
        reward_inputs_contract=reward_contract,
        control_effectiveness_contract=control_contract,
        current_recipe=current_recipe,
    )
    curriculum_design_contract = build_curriculum_design_contract()
    prompt_payload = {
        "task": config.get("task", {}),
        "loop_contract": {
            "policy_initialization": "random_each_candidate",
            "recipe_code_inheritance": "accepted_recipe_only",
            "candidate_allowed_controls": prompt_facing_candidate_allowed_controls(control_contract),
            "candidate_control_effectiveness_ref": "required_candidate_package.component_contract.control_effectiveness_contract",
            "candidate_forbidden": config.get("candidate_capabilities", {}).get("forbidden", []),
            "formal_training_in_this_command": False,
        },
        "public_feedback_summary": public_feedback_summary,
        "public_curriculum_evidence_bundle": public_curriculum_evidence_bundle,
        "curriculum_design_contract": curriculum_design_contract,
        "required_candidate_package": {
            "schema_version": "humanoid_agent.cold_start.candidate.v1",
            "entrypoint": CANDIDATE_ENTRYPOINT,
            "candidate_manifest_contract": manifest_contract,
            "component_contract": {
                "module": "humanoid_agent.code_sandbox.components",
                "builder_signature": "def build_training_components(task_spec, base_env_cfg, base_agent_cfg):",
                "builder_contract": (
                    "The runner calls the entrypoint with exactly three positional arguments. "
                    "Return TrainingComponents or a compatible dict; ignore arguments you do not need."
                ),
                "candidate_must_define": ["build_training_components"],
                "candidate_must_not_import": ["build_training_components"],
                "minimal_component_template": MINIMAL_COMPONENT_TEMPLATE,
                "component_classes_to_import": [
                    *PROMPT_FACING_COMPONENT_CLASSES_TO_IMPORT,
                ],
                "training_components_fields": list(PROMPT_FACING_TRAINING_COMPONENT_FIELDS),
                "reward_inputs_contract": reward_contract,
                "reward_inputs_fields": list(reward_contract["field_order"]),
                "reward_inputs_field_expansion_report": field_expansion_report,
                "reward_inputs_field_contract": (
                    "The reward_inputs_fields list is complete. Do not use fields not listed here; "
                    "for previous-step action history use prev_action and action_delta."
                ),
                "reward_terms": "RewardTerm(name, fn, weight=1.0, clip_min=None, clip_max=None); fn(RewardInputs) -> vector[num_envs]",
                "schedules": (
                    "ScheduleTerm(name, fn, allowed_keys); fn(progress_fraction: float) -> dict[str, numeric_or_bool]. "
                    "Every returned key must be listed in that term's allowed_keys and in runner_schedule_allowed_keys."
                ),
                "diagnostics": "DiagnosticTerm(name, fn); fn(RewardInputs) -> dict[str, numeric_or_bool_or_vector[num_envs]] summarized by the runner",
                "control_effectiveness_contract": control_contract,
                "runner_schedule_allowed_keys": list(control_contract["schedule_controls"]["allowed_keys"]),
                "effective_schedule_keys": control_contract["schedule_controls"]["effective_keys"],
                "not_available_to_candidate": [
                    "env object",
                    "eval config",
                    "success metric implementation",
                    "budget controls",
                    "checkpoint selector",
                    "runner hooks",
                ],
            },
            "allowed_top_level_files": allowed_top_level_files,
            "response_mode": (
                "source_requests_then_candidate_or_direct_candidate_only"
                if source_read_enabled
                else "candidate_only_source_read_disabled"
            ),
            "provider_json_output": _provider_json_output_contract(
                source_read_enabled=source_read_enabled,
                manifest_template=str(manifest_contract["template"]),
                max_source_read_turns=source_read_turn_budget,
            ),
        },
        "known_good_reward_examples": known_good_examples,
        "current_recipe": current_recipe,
        "public_history": public_history,
        "source_pack": source_pack,
        "source_access_policy": build_source_access_policy(
            source_pack,
            enabled=source_read_enabled,
            max_source_read_turns=source_read_turn_budget,
        ),
        "candidate_self_audit_checklist": candidate_checklist,
        "selection_guidance": [
            "Prefer interpretable training-side mechanisms that can pass the verifier.",
            "Write real Python RewardTerm and ScheduleTerm callables; do not return declarative hint dictionaries.",
            "Schedules are pre-evaluated by the runner on public progress points before training.",
            "The candidate must include runner-compiled CurriculumDesign as the primary training proposal.",
            "Use curriculum_design_contract to design curriculum hypothesis and stage structure in operation_diff.json, with diagnostics.json supporting the rationale.",
            "First-version curriculum design must use only the fixed capability_registry and must not request new runner capabilities.",
            "Each curriculum stage must include a capabilities list using exact ids from the fixed capability_registry.",
            "Curriculum proposals must include final_consolidation returning to the runner-owned target distribution.",
            "Do not submit single-stage/no-op curriculum designs or reward-only attribution; those fail the V1 curriculum audit.",
            "Use public feedback and recipe history only.",
            "Do not modify evaluation, success metrics, budget, checkpoint policy, or runner code.",
            "If the recipe changes curriculum, describe the schedule and expected failure mode it targets.",
        ],
        "schema_version": "humanoid_agent.phase8e.prompt_payload.v1",
    }
    assert_public_only_payload(prompt_payload, config=config, context="phase8e_prompt_payload")
    assert_prompt_facing_payload(prompt_payload, context="phase8e_prompt_payload")
    messages = [
        {
            "role": "system",
            "content": (
                "You are a robotics RL training-recipe agent. Output a candidate package, not runner changes. "
                "Use only the public payload. The policy will start from random initialization for each candidate. "
                "Read task, loop_contract, public_feedback_summary, and required_candidate_package first. "
                + (
                    "If source_access_policy.enabled is true and you need audited source text from source_pack, "
                    f"you may output a JSON object with source_requests only; the runtime will serve at most {source_read_turn_budget} source-read turn(s), and each source-read turn must be justified by need. "
                    "After the source-read budget is exhausted, or if you do not need source text: "
                    if source_read_enabled
                    else "Source-read is not available in this author call. "
                )
                + "Output exactly one JSON object with candidate_package.files only."
            ),
        },
        {"role": "user", "content": json.dumps(prompt_payload, ensure_ascii=False, indent=2)},
    ]
    prompt_text = messages_to_text(messages)
    assert_public_only_payload(prompt_text, config=config, context="phase8e_prompt_text")
    assert_prompt_facing_payload(prompt_text, context="phase8e_prompt_text")
    return {
        "schema_version": "humanoid_agent.phase8e.prompt_package.v1",
        "messages": messages,
        "prompt_text": prompt_text,
        "payload": prompt_payload,
        "source_store": source_store,
    }


def messages_to_text(messages: list[dict[str, str]]) -> str:
    return "\n\n".join(f"## {item['role']}\n\n{item['content']}" for item in messages) + "\n"
