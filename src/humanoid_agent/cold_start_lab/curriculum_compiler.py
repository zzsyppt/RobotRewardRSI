"""Runner-owned compiler for Phase 8E LLM curriculum designs."""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from humanoid_agent.code_sandbox.isolated_loader import load_mapping
from humanoid_agent.code_sandbox.loader import candidate_package_fingerprint, load_candidate_manifest

from .curriculum_contract import validate_v1_curriculum_candidate_artifacts


COMPILED_CURRICULUM_SCHEMA = "humanoid_agent.phase8e.compiled_curriculum.v1"
COMPILER_SUMMARY_SCHEMA = "humanoid_agent.phase8e.curriculum_compiler.summary.v1"
IMMUTABLE_CONTEXT_SCHEMA = "humanoid_agent.phase8e.curriculum_immutable_context.v1"
CURRICULUM_LINEAGE_PUBLIC_SCHEMA = "humanoid_agent.phase8e.curriculum_lineage.public.v1"
COMPILER_VERSION = "v1-runner-owned-stage-push-static-dr"
MIN_FINAL_FRACTION = 0.2
EPS = 1e-9
PUBLIC_LINEAGE_REQUIRED_ATTRIBUTION_KEYS = (
    "reward_code_changed",
    "curriculum_changed",
    "diagnostics_changed",
    "capability_changed",
    "gain_attributable_to",
    "pure_knob_candidate",
)
PUBLIC_LINEAGE_OPTIONAL_ATTRIBUTION_KEYS = (
    "absolute_curriculum_present",
    "component_code_changed",
)
PUBLIC_LINEAGE_GAIN_ATTRIBUTION_VALUES = {"curriculum", "reward_code", "both", "unclear"}


class CurriculumCompileError(RuntimeError):
    """Raised when a candidate curriculum cannot be compiled for training."""

    def __init__(self, message: str, *, summary: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.summary = summary or {}


def compile_v1_curriculum_for_training(
    *,
    candidate_dir: Path,
    phase1_config_path: Path,
    phase2_config_path: Path,
    output_dir: Path,
    train_profile: str,
    max_iterations: int,
    steps_per_iteration: int,
) -> dict[str, Any]:
    """Compile a validated LLM curriculum into runner-owned train-only configs."""

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_file = output_dir / "compiled_curriculum_summary.json"
    try:
        candidate_dir = candidate_dir.resolve()
        phase1_config_path = phase1_config_path.resolve()
        phase2_config_path = phase2_config_path.resolve()
        projection = _load_candidate_projection(candidate_dir)
        validate_v1_curriculum_candidate_artifacts(projection)
        design = projection["operation_diff"]["curriculum_design"]
        stages = _stage_list(design)
        _assert_full_progress_coverage(stages)
        final_fraction = _final_fraction(design)
        final_stage = stages[-1]
        final_start, final_end = _progress_range(final_stage)
        if final_end < 1.0 - EPS or (final_end - final_start) + EPS < final_fraction:
            raise CurriculumCompileError("final stage must cover the declared final consolidation fraction")

        phase1_config = _load_config_mapping(phase1_config_path)
        phase2_config = _load_config_mapping(phase2_config_path)
        target_push = _target_push_distribution(phase1_config)
        compiled_stages = [
            _compile_stage(
                stage=stage,
                stage_index=index,
                stage_count=len(stages),
                target_push=target_push,
                force_target=index == len(stages) - 1,
            )
            for index, stage in enumerate(stages)
        ]
        if not _stage_uses_capability(compiled_stages[-1], "external_push_curriculum"):
            compiled_stages[-1]["capabilities"] = sorted(
                set(compiled_stages[-1].get("capabilities", [])) | {"external_push_curriculum"}
            )
        if not _stage_uses_capability(compiled_stages[-1], "train_only_dr_scale"):
            compiled_stages[-1]["capabilities"] = sorted(
                set(compiled_stages[-1].get("capabilities", [])) | {"train_only_dr_scale"}
            )
        compiled_stages[-1]["push"] = _push_payload(target_push, scale=1.0)
        compiled_stages[-1]["robustness"] = _robustness_payload(scale=1.0, mode="train_only_static_observation_corruption")

        reward_code_changed = _operation_diff_has_non_curriculum_ops(projection["operation_diff"])
        compiled_curriculum = {
            "schema_version": COMPILED_CURRICULUM_SCHEMA,
            "compiler_version": COMPILER_VERSION,
            "enabled": True,
            "execution_mode": "env_step_progress",
            "source": "operation_diff.json.curriculum_design",
            "train_profile": train_profile,
            "planned_policy_iterations": int(max_iterations),
            "planned_training_steps": int(max_iterations) * int(steps_per_iteration),
            "capabilities": _unique_capabilities(compiled_stages),
            "target_distribution": {
                "push": target_push,
                "final_consolidation_fraction": final_fraction,
            },
            "dr_execution": {
                "mode": "train_only_static_observation_corruption",
                "reason": "Current Isaac overlay supports observation corruption at env construction; per-stage DR scaling is recorded but not candidate-controlled.",
            },
            "stages": compiled_stages,
            "guards": {
                "eval_config_changed": False,
                "budget_changed": False,
                "seed_changed": False,
                "checkpoint_selector_changed": False,
                "success_metric_changed": False,
            },
        }

        compiled_phase1 = copy.deepcopy(phase1_config)
        compiled_phase2 = copy.deepcopy(phase2_config)
        _install_compiled_curriculum(
            phase1_config=compiled_phase1,
            phase2_config=compiled_phase2,
            train_profile=train_profile,
            compiled_curriculum=compiled_curriculum,
        )

        phase1_out = output_dir / "compiled_phase1_push_recovery.yaml"
        phase2_out = output_dir / "compiled_phase2_baseline.yaml"
        _write_json(phase1_out, compiled_phase1)
        _write_json(phase2_out, compiled_phase2)

        summary = {
            "schema_version": COMPILER_SUMMARY_SCHEMA,
            "status": "pass",
            "created_at_utc": _now(),
            "compiler_version": COMPILER_VERSION,
            "candidate_dir": str(candidate_dir),
            "phase1_config_path": str(phase1_out),
            "phase2_config_path": str(phase2_out),
            "summary_file": str(summary_file),
            "raw_curriculum_design": copy.deepcopy(design),
            "effective_curriculum": compiled_curriculum,
            "clamps": _summarize_clamps(stages, compiled_stages),
            "reject_reason": None,
        }
        lineage_file = output_dir / "curriculum_lineage.json"
        lineage = build_curriculum_lineage_bundle(
            compiler_summary=summary,
            reward_code_changed=reward_code_changed,
            diagnostics_changed=isinstance(projection["diagnostics"].get("curriculum_hypothesis"), dict),
        )
        _write_json(lineage_file, lineage)
        summary["curriculum_lineage_file"] = str(lineage_file)
        _write_json(summary_file, summary)
        return summary
    except CurriculumCompileError as exc:
        summary = exc.summary or _reject_summary(summary_file=summary_file, reason=str(exc), candidate_dir=candidate_dir)
        _write_json(summary_file, summary)
        raise CurriculumCompileError(str(exc), summary=summary) from exc
    except Exception as exc:  # noqa: BLE001 - convert validation/load errors into compiler rejects.
        summary = _reject_summary(summary_file=summary_file, reason=str(exc), candidate_dir=candidate_dir)
        _write_json(summary_file, summary)
        raise CurriculumCompileError(str(exc), summary=summary) from exc


def project_curriculum_compile_summary(summary: dict[str, Any]) -> dict[str, Any]:
    effective = summary.get("effective_curriculum")
    stage_count = len(effective.get("stages", [])) if isinstance(effective, dict) else 0
    return {
        "status": summary.get("status"),
        "schema_version": summary.get("schema_version"),
        "compiler_version": summary.get("compiler_version"),
        "summary_file": summary.get("summary_file"),
        "stage_count": stage_count,
        "execution_mode": effective.get("execution_mode") if isinstance(effective, dict) else None,
        "capabilities": effective.get("capabilities", []) if isinstance(effective, dict) else [],
        "reject_reason": summary.get("reject_reason"),
        "phase1_config_path": summary.get("phase1_config_path"),
        "phase2_config_path": summary.get("phase2_config_path"),
        "curriculum_lineage_file": summary.get("curriculum_lineage_file"),
        "immutable_context_hash": summary.get("immutable_context_hash"),
    }


def build_curriculum_lineage_bundle(
    *,
    compiler_summary: dict[str, Any],
    reward_code_changed: bool,
    diagnostics_changed: bool,
) -> dict[str, Any]:
    effective = compiler_summary.get("effective_curriculum")
    effective_curriculum = effective if isinstance(effective, dict) else {}
    capabilities = list(effective_curriculum.get("capabilities", [])) if isinstance(effective_curriculum.get("capabilities"), list) else []
    stage_count = len(effective_curriculum.get("stages", [])) if isinstance(effective_curriculum.get("stages"), list) else 0
    curriculum_changed = stage_count >= 2 and any(capability != "reward_weight_scale" for capability in capabilities)
    capability_changed = any(capability in {"external_push_curriculum", "train_only_dr_scale"} for capability in capabilities)
    if reward_code_changed and curriculum_changed:
        gain_attributable_to = "both"
    elif curriculum_changed:
        gain_attributable_to = "curriculum"
    elif reward_code_changed:
        gain_attributable_to = "reward_code"
    else:
        gain_attributable_to = "unclear"
    pure_knob_candidate = not curriculum_changed
    public_summary = {
        "schema_version": CURRICULUM_LINEAGE_PUBLIC_SCHEMA,
        "status": compiler_summary.get("status"),
        "compiler_version": compiler_summary.get("compiler_version"),
        "stage_count": stage_count,
        "execution_mode": effective_curriculum.get("execution_mode"),
        "capabilities": capabilities,
        "attribution": {
            "reward_code_changed": bool(reward_code_changed),
            "curriculum_changed": bool(curriculum_changed),
            "diagnostics_changed": bool(diagnostics_changed),
            "capability_changed": bool(capability_changed),
            "gain_attributable_to": gain_attributable_to,
            "pure_knob_candidate": bool(pure_knob_candidate),
        },
        "clamp_count": len(compiler_summary.get("clamps", [])) if isinstance(compiler_summary.get("clamps"), list) else 0,
    }
    return {
        "schema_version": "humanoid_agent.phase8e.curriculum_lineage.v1",
        "public_summary": public_summary,
        "raw_curriculum_design": copy.deepcopy(compiler_summary.get("raw_curriculum_design")),
        "effective_curriculum": copy.deepcopy(effective_curriculum),
        "clamps": copy.deepcopy(compiler_summary.get("clamps", [])),
    }


def project_curriculum_lineage_bundle(bundle: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(bundle, dict):
        return None
    public_summary = bundle.get("public_summary")
    if not isinstance(public_summary, dict):
        return None
    return _project_valid_curriculum_lineage_public_summary(public_summary)


def _project_valid_curriculum_lineage_public_summary(public_summary: Mapping[str, Any]) -> dict[str, Any] | None:
    if public_summary.get("schema_version") != CURRICULUM_LINEAGE_PUBLIC_SCHEMA:
        return None
    status = public_summary.get("status")
    compiler_version = public_summary.get("compiler_version")
    stage_count = public_summary.get("stage_count")
    execution_mode = public_summary.get("execution_mode")
    capabilities = public_summary.get("capabilities")
    attribution = public_summary.get("attribution")
    if status != "pass" or not isinstance(compiler_version, str) or not compiler_version:
        return None
    if isinstance(stage_count, bool) or not isinstance(stage_count, int) or stage_count < 1:
        return None
    if not isinstance(execution_mode, str) or not execution_mode:
        return None
    if not isinstance(capabilities, list) or any(not isinstance(item, str) or not item for item in capabilities):
        return None
    if not isinstance(attribution, Mapping):
        return None
    projected_attribution: dict[str, Any] = {}
    for key in PUBLIC_LINEAGE_REQUIRED_ATTRIBUTION_KEYS:
        value = attribution.get(key)
        if key == "gain_attributable_to":
            if value not in PUBLIC_LINEAGE_GAIN_ATTRIBUTION_VALUES:
                return None
            projected_attribution[key] = value
        else:
            if not isinstance(value, bool):
                return None
            projected_attribution[key] = value
    for key in PUBLIC_LINEAGE_OPTIONAL_ATTRIBUTION_KEYS:
        value = attribution.get(key)
        if isinstance(value, bool):
            projected_attribution[key] = value

    projected: dict[str, Any] = {
        "schema_version": public_summary["schema_version"],
        "status": status,
        "compiler_version": compiler_version,
        "stage_count": stage_count,
        "execution_mode": execution_mode,
        "capabilities": copy.deepcopy(capabilities),
        "attribution": projected_attribution,
    }
    clamp_count = public_summary.get("clamp_count")
    if isinstance(clamp_count, int) and not isinstance(clamp_count, bool) and clamp_count >= 0:
        projected["clamp_count"] = clamp_count
    immutable_context_hash = public_summary.get("immutable_context_hash")
    if immutable_context_hash is not None:
        if not _is_sha256_hex(immutable_context_hash):
            return None
        projected["immutable_context_hash"] = immutable_context_hash
    relative = public_summary.get("relative_to_current_recipe")
    if isinstance(relative, Mapping):
        projected["relative_to_current_recipe"] = copy.deepcopy(dict(relative))
    return projected


def _is_sha256_hex(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value.lower())


def enrich_curriculum_summary_with_immutable_context(
    summary: dict[str, Any],
    *,
    prompt_payload: Mapping[str, Any] | None,
    prompt_payload_path: Path | None,
    artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    """Attach runner-owned immutable context hashes to a compiled curriculum lineage."""

    if summary.get("status") != "pass":
        return summary
    lineage_file = summary.get("curriculum_lineage_file")
    summary_file = summary.get("summary_file")
    if not isinstance(lineage_file, str) or not isinstance(summary_file, str):
        return summary

    lineage_path = Path(lineage_file)
    lineage = load_mapping(lineage_path)
    context = build_curriculum_immutable_context(
        prompt_payload=prompt_payload,
        prompt_payload_path=prompt_payload_path,
        artifacts=artifacts,
    )
    lineage["immutable_context"] = context
    relative = _relative_attribution_to_current_recipe(
        summary=summary,
        lineage=lineage,
        prompt_payload=prompt_payload,
    )
    if relative is not None:
        _apply_relative_attribution(lineage, relative)
    public_summary = lineage.get("public_summary")
    if isinstance(public_summary, dict):
        public_summary["immutable_context_hash"] = context["sha256"]
    _write_json(lineage_path, lineage)

    enriched_summary = copy.deepcopy(summary)
    enriched_summary["immutable_context_hash"] = context["sha256"]
    _write_json(Path(summary_file), enriched_summary)
    return enriched_summary


def _relative_attribution_to_current_recipe(
    *,
    summary: Mapping[str, Any],
    lineage: Mapping[str, Any],
    prompt_payload: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(prompt_payload, Mapping):
        return None
    current_recipe = _mapping_or_none(prompt_payload.get("current_recipe"))
    if current_recipe is None:
        return None
    current_fingerprint = _mapping_or_none(current_recipe.get("fingerprint"))
    if current_fingerprint is None:
        return None

    candidate_fingerprint = _candidate_fingerprint_from_summary(summary)
    candidate_code_hash = _str_or_none(candidate_fingerprint.get("training_code_tree_sha256") if candidate_fingerprint else None)
    current_code_hash = _str_or_none(current_fingerprint.get("training_code_tree_sha256"))
    component_code_changed = (
        candidate_code_hash is not None and current_code_hash is not None and candidate_code_hash != current_code_hash
    )

    current_operation_diff = _mapping_or_none(current_recipe.get("operation_diff"))
    current_design = _mapping_or_none(current_operation_diff.get("curriculum_design") if current_operation_diff else None)
    candidate_design = _mapping_or_none(summary.get("raw_curriculum_design"))
    if candidate_design is not None and current_design is not None:
        candidate_curriculum_hash = _curriculum_design_effect_hash(candidate_design)
        current_curriculum_hash = _curriculum_design_effect_hash(current_design)
        curriculum_changed = candidate_curriculum_hash != current_curriculum_hash
    else:
        candidate_curriculum_hash = None
        current_curriculum_hash = None
        public = _mapping_or_none(lineage.get("public_summary"))
        attribution = _mapping_or_none(public.get("attribution") if public else None)
        curriculum_changed = bool(attribution.get("curriculum_changed")) if attribution else False

    candidate_diagnostics = _candidate_diagnostics_from_summary(summary)
    current_diagnostics = _mapping_or_none(current_recipe.get("diagnostics"))
    diagnostics_changed = (
        candidate_diagnostics is not None
        and current_diagnostics is not None
        and _stable_json_sha256(candidate_diagnostics) != _stable_json_sha256(current_diagnostics)
    )

    candidate_capabilities = _curriculum_capability_signature(candidate_design)
    current_capabilities = _curriculum_capability_signature(current_design)
    capability_changed = (
        candidate_capabilities is not None
        and current_capabilities is not None
        and candidate_capabilities != current_capabilities
    )
    reward_code_changed = component_code_changed
    pure_knob_candidate = not curriculum_changed and not reward_code_changed
    return {
        "schema_version": "humanoid_agent.phase8e.curriculum_relative_attribution.v1",
        "reference": "current_recipe",
        "reference_candidate_id": current_fingerprint.get("candidate_id"),
        "candidate_candidate_id": candidate_fingerprint.get("candidate_id") if candidate_fingerprint else None,
        "comparison_basis": {
            "component_code": "training_code_tree_sha256",
            "curriculum": "normalized_curriculum_design_effect_signature",
            "diagnostics": "diagnostics_json_sha256",
        },
        "candidate_training_code_tree_sha256": candidate_code_hash,
        "reference_training_code_tree_sha256": current_code_hash,
        "candidate_curriculum_effect_sha256": candidate_curriculum_hash,
        "reference_curriculum_effect_sha256": current_curriculum_hash,
        "reward_code_changed": reward_code_changed,
        "component_code_changed": component_code_changed,
        "curriculum_changed": curriculum_changed,
        "diagnostics_changed": diagnostics_changed,
        "capability_changed": capability_changed,
        "gain_attributable_to": _gain_attribution(
            reward_code_changed=reward_code_changed,
            curriculum_changed=curriculum_changed,
        ),
        "pure_knob_candidate": pure_knob_candidate,
    }


def _apply_relative_attribution(lineage: dict[str, Any], relative: Mapping[str, Any]) -> None:
    lineage["relative_to_current_recipe"] = copy.deepcopy(dict(relative))
    public_summary = lineage.get("public_summary")
    if not isinstance(public_summary, dict):
        return
    attribution = public_summary.get("attribution")
    if not isinstance(attribution, dict):
        attribution = {}
        public_summary["attribution"] = attribution
    attribution["absolute_curriculum_present"] = bool(attribution.get("curriculum_changed"))
    attribution["reward_code_changed"] = bool(relative.get("reward_code_changed"))
    attribution["component_code_changed"] = bool(relative.get("component_code_changed"))
    attribution["curriculum_changed"] = bool(relative.get("curriculum_changed"))
    attribution["diagnostics_changed"] = bool(relative.get("diagnostics_changed"))
    attribution["capability_changed"] = bool(relative.get("capability_changed"))
    attribution["gain_attributable_to"] = relative.get("gain_attributable_to")
    attribution["pure_knob_candidate"] = bool(relative.get("pure_knob_candidate"))
    public_summary["relative_to_current_recipe"] = {
        key: copy.deepcopy(relative.get(key))
        for key in (
            "schema_version",
            "reference",
            "reference_candidate_id",
            "candidate_candidate_id",
            "comparison_basis",
            "candidate_training_code_tree_sha256",
            "reference_training_code_tree_sha256",
            "candidate_curriculum_effect_sha256",
            "reference_curriculum_effect_sha256",
        )
    }


def _gain_attribution(*, reward_code_changed: bool, curriculum_changed: bool) -> str:
    if reward_code_changed and curriculum_changed:
        return "both"
    if curriculum_changed:
        return "curriculum"
    if reward_code_changed:
        return "reward_code"
    return "unclear"


def _candidate_fingerprint_from_summary(summary: Mapping[str, Any]) -> dict[str, Any] | None:
    candidate_dir = _path_or_none(summary.get("candidate_dir"))
    if candidate_dir is None or not candidate_dir.is_dir():
        return None
    try:
        return candidate_package_fingerprint(load_candidate_manifest(candidate_dir))
    except Exception:  # noqa: BLE001 - lineage attribution should degrade, not break training.
        return None


def _candidate_diagnostics_from_summary(summary: Mapping[str, Any]) -> Mapping[str, Any] | None:
    candidate_dir = _path_or_none(summary.get("candidate_dir"))
    if candidate_dir is None:
        return None
    diagnostics_path = candidate_dir / "diagnostics.json"
    if not diagnostics_path.is_file():
        return None
    try:
        return load_mapping(diagnostics_path)
    except Exception:  # noqa: BLE001 - attribution metadata should not convert a trained candidate into reject.
        return None


def _curriculum_design_effect_hash(design: Mapping[str, Any]) -> str:
    return _stable_json_sha256(_curriculum_design_effect_signature(design))


def _curriculum_design_effect_signature(design: Mapping[str, Any]) -> dict[str, Any]:
    stages = []
    raw_stages = design.get("stage_list", design.get("stages"))
    if isinstance(raw_stages, list):
        for item in raw_stages:
            if not isinstance(item, Mapping):
                continue
            stages.append(
                {
                    "name": item.get("name"),
                    "progress_range": item.get("progress_range"),
                    "capabilities": sorted(str(capability) for capability in item.get("capabilities", []) if isinstance(capability, str)),
                }
            )
    final = _mapping_or_none(design.get("final_consolidation")) or {}
    return {
        "stage_list": stages,
        "final_consolidation": {
            "fraction": _first_present_value(final, ("fraction", "min_fraction", "min_fraction_of_training")),
            "distribution": final.get("distribution"),
            "includes_target_push": _first_present_value(
                final,
                ("includes_target_push", "target_push_included", "must_include_target_push"),
            ),
            "includes_target_robustness_variation": _first_present_value(
                final,
                (
                    "includes_target_robustness_variation",
                    "target_robustness_included",
                    "must_include_target_robustness_variation",
                ),
            ),
        },
    }


def _curriculum_capability_signature(design: Mapping[str, Any] | None) -> list[list[str]] | None:
    if design is None:
        return None
    raw_stages = design.get("stage_list", design.get("stages"))
    if not isinstance(raw_stages, list):
        return None
    signature: list[list[str]] = []
    for item in raw_stages:
        if not isinstance(item, Mapping):
            continue
        signature.append(sorted(str(capability) for capability in item.get("capabilities", []) if isinstance(capability, str)))
    return signature


def _str_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _first_present_value(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping.get(key)
    return None


def build_curriculum_immutable_context(
    *,
    prompt_payload: Mapping[str, Any] | None,
    prompt_payload_path: Path | None,
    artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    prompt_available = isinstance(prompt_payload, Mapping)
    prompt = prompt_payload if prompt_available else {}
    public_evidence = _mapping_or_none(prompt.get("public_curriculum_evidence_bundle"))
    capability_registry = _mapping_or_none(public_evidence.get("capability_registry") if public_evidence else None)
    prompt_contract = (
        {
            "required_candidate_package": copy.deepcopy(prompt.get("required_candidate_package")),
            "candidate_allowed_controls": copy.deepcopy(prompt.get("candidate_allowed_controls")),
            "curriculum_design_contract": copy.deepcopy(prompt.get("curriculum_design_contract")),
        }
        if prompt_available
        else None
    )

    body = {
        "schema_version": IMMUTABLE_CONTEXT_SCHEMA,
        "capability_registry": _hash_prompt_section(
            value=capability_registry,
            source_path=prompt_payload_path,
            summary=_capability_registry_summary(capability_registry),
        ),
        "curriculum_compiler": _curriculum_compiler_section(),
        "public_evidence_bundle": _hash_prompt_section(
            value=public_evidence,
            source_path=prompt_payload_path,
            summary=_prompt_section_summary(public_evidence),
        ),
        "prompt_facing_contract": _hash_prompt_section(
            value=prompt_contract,
            source_path=prompt_payload_path,
            summary=_prompt_contract_summary(prompt_contract),
        ),
        "training_config": _training_config_section(artifacts),
        "eval_config": _eval_config_section(artifacts),
        "prompt_payload": _file_artifact(prompt_payload_path),
    }
    body["sha256"] = _stable_json_sha256(body)
    return body


def _hash_prompt_section(
    *,
    value: Any,
    source_path: Path | None,
    summary: Mapping[str, Any] | None,
) -> dict[str, Any]:
    present = value is not None
    section: dict[str, Any] = {
        "available": bool(present),
        "source_path": str(source_path.resolve()) if source_path is not None else None,
    }
    if summary is not None:
        section.update(copy.deepcopy(dict(summary)))
    section["sha256"] = _stable_json_sha256(value if present else section)
    return section


def _curriculum_compiler_section() -> dict[str, Any]:
    payload = {
        "version": COMPILER_VERSION,
        "compiled_curriculum_schema": COMPILED_CURRICULUM_SCHEMA,
        "compiler_summary_schema": COMPILER_SUMMARY_SCHEMA,
        "source_file": str(Path(__file__).resolve()),
        "source_file_sha256": _sha256_file(Path(__file__).resolve()),
    }
    payload["sha256"] = _stable_json_sha256(payload)
    return payload


def _training_config_section(artifacts: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "loop_config": _file_artifact(_path_or_none(artifacts.get("loop_config_path"))),
        "budget_config": _file_artifact(_path_or_none(artifacts.get("budget_config_path"))),
        "original_phase1_config": _file_artifact(_path_or_none(artifacts.get("original_phase1_config_path"))),
        "original_phase2_config": _file_artifact(_path_or_none(artifacts.get("original_phase2_config_path"))),
        "compiled_phase1_config": _file_artifact(_path_or_none(artifacts.get("compiled_phase1_config_path"))),
        "compiled_phase2_config": _file_artifact(_path_or_none(artifacts.get("compiled_phase2_config_path"))),
        "training_script": _file_artifact(_path_or_none(artifacts.get("training_script_path"))),
        "train_profile": artifacts.get("train_profile"),
        "planned_policy_iterations": artifacts.get("planned_policy_iterations"),
        "planned_training_steps": artifacts.get("planned_training_steps"),
    }
    payload["sha256"] = _stable_json_sha256(payload)
    return payload


def _eval_config_section(artifacts: Mapping[str, Any]) -> dict[str, Any]:
    dev_seeds = _int_list(artifacts.get("public_dev_seeds"))
    accept_seeds = _int_list(artifacts.get("public_accept_seeds"))
    payload = {
        "eval_script": _file_artifact(_path_or_none(artifacts.get("eval_script_path"))),
        "eval_config": _file_artifact(_path_or_none(artifacts.get("eval_config_path"))),
        "profile": artifacts.get("rollout_profile"),
        "num_envs": artifacts.get("rollout_num_envs"),
        "max_steps": artifacts.get("rollout_max_steps"),
        "seed_visibility": artifacts.get("rollout_seed_visibility") or "runner_owned_not_prompt_facing",
        "seed_counts": {
            "public_dev": len(dev_seeds),
            "public_accept": len(accept_seeds),
        },
        "seed_hashes": {
            "public_dev": _stable_json_sha256(dev_seeds),
            "public_accept": _stable_json_sha256(accept_seeds),
        },
    }
    payload["sha256"] = _stable_json_sha256(payload)
    return payload


def _file_artifact(path: Path | None) -> dict[str, Any]:
    if path is None:
        payload: dict[str, Any] = {"available": False, "path": None, "reason": "not_provided"}
        payload["sha256"] = _stable_json_sha256(payload)
        return payload
    resolved = path.resolve()
    payload = {"available": resolved.is_file(), "path": str(resolved)}
    if resolved.is_file():
        payload["sha256"] = _sha256_file(resolved)
    else:
        payload["reason"] = "file_missing"
        payload["sha256"] = _stable_json_sha256(payload)
    return payload


def _capability_registry_summary(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {"capability_count": 0, "capability_ids": []}
    raw_items = value.get("capabilities", [])
    ids: list[str] = []
    if isinstance(raw_items, list):
        for item in raw_items:
            if isinstance(item, Mapping) and isinstance(item.get("id"), str):
                ids.append(str(item["id"]))
            elif isinstance(item, str):
                ids.append(item)
    return {
        "schema_version": value.get("schema_version"),
        "capability_count": len(ids),
        "capability_ids": sorted(set(ids)),
    }


def _prompt_section_summary(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {"schema_version": None}
    return {
        "schema_version": value.get("schema_version"),
        "top_level_keys": sorted(str(key) for key in value.keys()),
    }


def _prompt_contract_summary(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {"top_level_keys": []}
    keys = [key for key, item in value.items() if item is not None]
    return {"top_level_keys": sorted(str(key) for key in keys)}


def _mapping_or_none(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _path_or_none(value: Any) -> Path | None:
    if value is None:
        return None
    return Path(value)


def _int_list(value: Any) -> list[int]:
    if not isinstance(value, (list, tuple)):
        return []
    return [int(item) for item in value]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_json_sha256(value: Any) -> str:
    payload = json.dumps(_json_ready(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _load_candidate_projection(candidate_dir: Path) -> dict[str, Any]:
    return {
        "operation_diff": load_mapping(candidate_dir / "operation_diff.json"),
        "diagnostics": load_mapping(candidate_dir / "diagnostics.json"),
    }


def _operation_diff_has_non_curriculum_ops(operation_diff: dict[str, Any]) -> bool:
    operations = operation_diff.get("operations")
    return isinstance(operations, list) and len(operations) > 0


def _load_config_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore[import-not-found]  # noqa: PLC0415

            loaded = yaml.safe_load(text)
        except ModuleNotFoundError:
            loaded = _parse_simple_yaml(text)
    if not isinstance(loaded, dict):
        raise CurriculumCompileError(f"{path} must contain a mapping")
    return loaded


def _parse_simple_yaml(text: str) -> Any:
    lines = _preprocess_simple_yaml(text)
    if not lines:
        return None
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any] | list[Any]]] = [(-1, root)]
    for index, (indent, content) in enumerate(lines):
        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not stack:
            raise CurriculumCompileError(f"invalid YAML indentation near line: {content}")
        parent = stack[-1][1]
        if content.startswith("- "):
            if not isinstance(parent, list):
                raise CurriculumCompileError(f"unexpected YAML list item outside a list: {content}")
            parent.append(_parse_simple_yaml_scalar(content[2:].strip()))
            continue
        if ":" not in content:
            raise CurriculumCompileError(f"unsupported YAML line: {content}")
        if not isinstance(parent, dict):
            raise CurriculumCompileError(f"unexpected YAML mapping entry inside a scalar list: {content}")
        key, raw_value = content.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if raw_value in {">", "|", ">-", "|-"}:
            parent[key] = ""
            continue
        if raw_value == "":
            child: dict[str, Any] | list[Any]
            child = [] if _next_child_is_list(lines, index, indent) else {}
            parent[key] = child
            stack.append((indent, child))
            continue
        parent[key] = _parse_simple_yaml_scalar(raw_value)
    return root


def _next_child_is_list(lines: list[tuple[int, str]], index: int, indent: int) -> bool:
    for child_indent, child_content in lines[index + 1 :]:
        if child_indent <= indent:
            return False
        return child_content.startswith("- ")
    return False


def _preprocess_simple_yaml(text: str) -> list[tuple[int, str]]:
    result: list[tuple[int, str]] = []
    skip_block_indent: int | None = None
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        stripped = raw.strip()
        if skip_block_indent is not None:
            if indent > skip_block_indent:
                continue
            skip_block_indent = None
        content = _strip_yaml_comment(stripped)
        if not content:
            continue
        result.append((indent, content))
        if content.endswith(": >") or content.endswith(": |") or content.endswith(": >-") or content.endswith(": |-"):
            skip_block_indent = indent
    return result


def _strip_yaml_comment(value: str) -> str:
    in_single = False
    in_double = False
    for index, char in enumerate(value):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double and (index == 0 or value[index - 1].isspace()):
            return value[:index].rstrip()
    return value


def _parse_simple_yaml_scalar(value: str) -> Any:
    if value in {"null", "Null", "NULL", "~"}:
        return None
    if value in {"true", "True", "TRUE"}:
        return True
    if value in {"false", "False", "FALSE"}:
        return False
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_simple_yaml_scalar(item.strip()) for item in inner.split(",")]
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        if not any(marker in value for marker in (".", "e", "E")):
            return int(value)
        return float(value)
    except ValueError:
        return value


def _stage_list(design: dict[str, Any]) -> list[dict[str, Any]]:
    raw = design.get("stage_list", design.get("stages"))
    if not isinstance(raw, list) or not raw:
        raise CurriculumCompileError("curriculum_design.stage_list is required")
    stages = [stage for stage in raw if isinstance(stage, dict)]
    if len(stages) != len(raw):
        raise CurriculumCompileError("curriculum_design.stage_list entries must be objects")
    return stages


def _assert_full_progress_coverage(stages: list[dict[str, Any]]) -> None:
    expected_start = 0.0
    for index, stage in enumerate(stages):
        start, end = _progress_range(stage)
        if abs(start - expected_start) > 1e-6:
            raise CurriculumCompileError(f"stage_list[{index}] must start at {expected_start:.6f}; got {start:.6f}")
        expected_start = end
    if abs(expected_start - 1.0) > 1e-6:
        raise CurriculumCompileError("curriculum stages must cover the full [0, 1] training progress range")


def _progress_range(stage: dict[str, Any]) -> tuple[float, float]:
    raw = stage.get("progress_range")
    if not isinstance(raw, list | tuple) or len(raw) != 2:
        raise CurriculumCompileError("stage.progress_range must be [start, end]")
    start = float(raw[0])
    end = float(raw[1])
    return start, end


def _final_fraction(design: dict[str, Any]) -> float:
    final = design.get("final_consolidation")
    if not isinstance(final, dict):
        raise CurriculumCompileError("curriculum_design.final_consolidation is required")
    for key in ("fraction", "min_fraction", "min_fraction_of_training"):
        if key in final:
            value = float(final[key])
            break
    else:
        value = 0.0
    if value < MIN_FINAL_FRACTION:
        raise CurriculumCompileError("final_consolidation fraction below runner floor")
    return value


def _target_push_distribution(phase1_config: dict[str, Any]) -> dict[str, Any]:
    profiles = phase1_config.get("profiles")
    public_profile = profiles.get("phase2_public") if isinstance(profiles, dict) else None
    push = public_profile.get("push") if isinstance(public_profile, dict) else None
    if not isinstance(push, dict):
        push = phase1_config.get("push")
    if not isinstance(push, dict):
        raise CurriculumCompileError("phase1 config must define a target push distribution")
    return {
        "interval_range_s": list(_pair(push.get("interval_range_s", [1.0, 1.0]))),
        "velocity_range_mps": {
            "x": list(_pair(push.get("velocity_range_mps", {}).get("x", [0.0, 0.0]))),
            "y": list(_pair(push.get("velocity_range_mps", {}).get("y", [0.0, 0.0]))),
        },
    }


def _compile_stage(
    *,
    stage: dict[str, Any],
    stage_index: int,
    stage_count: int,
    target_push: dict[str, Any],
    force_target: bool,
) -> dict[str, Any]:
    capabilities = _capabilities(stage)
    start, end = _progress_range(stage)
    push_capable = "external_push_curriculum" in capabilities
    robustness_capable = "train_only_dr_scale" in capabilities
    push_scale = 1.0 if force_target else _stage_default_scale(stage_index, stage_count) if push_capable else 0.0
    dr_scale = 1.0 if force_target else _stage_default_scale(stage_index, stage_count) if robustness_capable else 0.0
    return {
        "name": str(stage.get("name", f"stage_{stage_index + 1}")),
        "progress_range": [round(start, 6), round(end, 6)],
        "capabilities": capabilities,
        "rationale": str(stage.get("rationale", ""))[:512],
        "push": _push_payload(target_push, scale=push_scale),
        "robustness": _robustness_payload(scale=dr_scale, mode="train_only_static_observation_corruption"),
    }


def _capabilities(stage: dict[str, Any]) -> list[str]:
    raw = stage.get("capabilities")
    if not isinstance(raw, list):
        return []
    return [str(item) for item in raw if isinstance(item, str)]


def _stage_default_scale(stage_index: int, stage_count: int) -> float:
    if stage_count <= 1:
        return 1.0
    return max(0.35, min(0.75, (stage_index + 1) / stage_count))


def _push_payload(target_push: dict[str, Any], *, scale: float) -> dict[str, Any]:
    interval = list(_pair(target_push.get("interval_range_s", [1.0, 1.0])))
    velocity = target_push.get("velocity_range_mps", {})
    return {
        "enabled": scale > 0.0,
        "scale": round(scale, 6),
        "interval_range_s": interval,
        "velocity_range_mps": {
            "x": _scale_pair(velocity.get("x", [0.0, 0.0]), scale),
            "y": _scale_pair(velocity.get("y", [0.0, 0.0]), scale),
        },
    }


def _robustness_payload(*, scale: float, mode: str) -> dict[str, Any]:
    return {
        "enabled": scale > 0.0,
        "scale": round(scale, 6),
        "mode": mode,
    }


def _install_compiled_curriculum(
    *,
    phase1_config: dict[str, Any],
    phase2_config: dict[str, Any],
    train_profile: str,
    compiled_curriculum: dict[str, Any],
) -> None:
    phase1_config["runner_curriculum"] = copy.deepcopy(compiled_curriculum)
    stages = compiled_curriculum["stages"]
    first_push = stages[0]["push"]
    profiles = phase1_config.setdefault("profiles", {})
    train = profiles.setdefault(train_profile, {})
    train["push"] = {
        "interval_range_s": list(first_push["interval_range_s"]),
        "velocity_range_mps": copy.deepcopy(first_push["velocity_range_mps"]),
    }
    phase1_config["push"] = {
        "interval_range_s": list(first_push["interval_range_s"]),
        "velocity_range_mps": copy.deepcopy(first_push["velocity_range_mps"]),
    }
    if any(stage.get("robustness", {}).get("enabled") for stage in stages):
        env = phase1_config.setdefault("environment", {})
        env["disable_observation_corruption"] = False
        phase2_config.setdefault("curriculum", {})["train_only_dr_scale"] = 1.0


def _unique_capabilities(stages: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for stage in stages:
        for capability in stage.get("capabilities", []):
            if capability not in seen:
                seen.add(capability)
                ordered.append(capability)
    return ordered


def _stage_uses_capability(stage: dict[str, Any], capability: str) -> bool:
    return capability in stage.get("capabilities", [])


def _summarize_clamps(raw_stages: list[dict[str, Any]], compiled_stages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for raw, compiled in zip(raw_stages, compiled_stages, strict=True):
        raw_capabilities = _capabilities(raw)
        effective_capabilities = _capabilities(compiled)
        added_capabilities = sorted(set(effective_capabilities) - set(raw_capabilities))
        removed_capabilities = sorted(set(raw_capabilities) - set(effective_capabilities))
        item = {
            "stage": compiled.get("name"),
            "raw_progress_range": raw.get("progress_range"),
            "compiled_progress_range": compiled.get("progress_range"),
            "raw_capabilities": raw_capabilities,
            "effective_capabilities": effective_capabilities,
            "added_capabilities": added_capabilities,
            "removed_capabilities": removed_capabilities,
            "compiled_push_scale": compiled.get("push", {}).get("scale"),
            "compiled_dr_scale": compiled.get("robustness", {}).get("scale"),
        }
        forced_final_capabilities = sorted(
            set(added_capabilities) & {"external_push_curriculum", "train_only_dr_scale"}
        )
        if forced_final_capabilities and compiled.get("push", {}).get("scale") == 1.0 and compiled.get(
            "robustness", {}
        ).get("scale") == 1.0:
            item["forced_final_consolidation_capabilities"] = forced_final_capabilities
            item["capability_augmentation_reason"] = "final_consolidation_forces_target_push_and_dr"
        result.append(
            item
        )
    return result


def _reject_summary(*, summary_file: Path, reason: str, candidate_dir: Path) -> dict[str, Any]:
    return {
        "schema_version": COMPILER_SUMMARY_SCHEMA,
        "status": "reject",
        "created_at_utc": _now(),
        "compiler_version": COMPILER_VERSION,
        "candidate_dir": str(candidate_dir),
        "summary_file": str(summary_file),
        "reject_reason": reason[:512],
        "effective_curriculum": None,
    }


def _scale_pair(value: Any, scale: float) -> list[float]:
    lo, hi = _pair(value)
    return [round(lo * scale, 6), round(hi * scale, 6)]


def _pair(value: Any) -> tuple[float, float]:
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise CurriculumCompileError(f"expected two-element numeric range, got {value!r}")
    return (float(value[0]), float(value[1]))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
