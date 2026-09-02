#!/usr/bin/env python3
"""Self-test for the Phase 8E cold-start loop scaffold."""

from __future__ import annotations

import copy
import json
import shutil
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from humanoid_agent.code_sandbox.components import RewardInputs  # noqa: E402
from humanoid_agent.code_sandbox.isolated_loader import load_mapping  # noqa: E402
from humanoid_agent.code_sandbox.isolation import probe_isolation_capabilities  # noqa: E402
from humanoid_agent.cold_start_lab.recipe_adapter import evaluate_reward_terms, precompute_schedule_table  # noqa: E402
from humanoid_agent.cold_start_lab.training_adapter import reward_inputs_contract as full_reward_inputs_contract  # noqa: E402
from humanoid_agent.cold_start_lab.phase8e import (  # noqa: E402
    Phase8EError,
    assert_public_only_payload,
    build_phase8e_prompt_package,
    load_ledger,
    project_public_result,
    run_phase8e_round,
)
from humanoid_agent.cold_start_lab.phase8e_provider import resolve_source_requests  # noqa: E402
from humanoid_agent.cold_start_lab.phase8e import _prompt_history_public_result  # noqa: E402
from humanoid_agent.cold_start_lab.phase8e_common import scrub_prompt_facing_value  # noqa: E402
from humanoid_agent.cold_start_lab.phase8e_projection import build_public_history_summary  # noqa: E402


CONFIG_PATH = PROJECT_ROOT / "configs" / "cold_start" / "phase8e_loop.yaml"
VERIFIER_CONFIG_PATH = PROJECT_ROOT / "configs" / "cold_start" / "phase8d_verifier.yaml"
DECLARATIVE_CANDIDATE = PROJECT_ROOT / "examples" / "cold_start_candidates" / "declarative_curriculum_candidate"
INACTIVE_PROMPT_TOKENS = (
    "not_instrumented",
    "accepted_precomputed_not_applied",
    "accepted_but_ineffective",
    "ineffective_schedule_keys_not_primary_mechanism",
    "LossTerm",
    "loss_terms",
    "train_observation_transforms",
    "push_magnitude_scale",
    "push_start_fraction",
    "push_interval_s",
    "domain_randomization_scale",
    "domain randomization",
)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _assert_no_inactive_prompt_tokens(text: str, context: str) -> None:
    for token in INACTIVE_PROMPT_TOKENS:
        _assert(token not in text, f"{context} must not advertise inactive capability: {token}")


def _assert_p2_prompt_order(prompt_text: str, prompt_payload: dict[str, Any], context: str) -> None:
    _assert(
        next(iter(prompt_payload)) == "task",
        f"{context} serialized prompt payload artifact must preserve task-first order: {list(prompt_payload)[:5]}",
    )
    prompt_top_30 = "\n".join(prompt_text.splitlines()[:30])
    _assert(
        "Output exactly one JSON object with candidate_package.files only" in prompt_top_30,
        f"{context} prompt top section must state the delivery format early: {prompt_top_30}",
    )
    _assert(
        '"task": {' in prompt_top_30
        and '"objective": "maximize fixed public standing push-recovery success from random policy initialization"'
        in prompt_top_30,
        f"{context} prompt top section must expose the real task before supporting material: {prompt_top_30}",
    )
    _assert(
        prompt_text.count('"objective":') == 1,
        f"{context} prompt must keep a single objective field instead of duplicating top-level and task objectives",
    )
    ordered_markers = [
        '"task":',
        '"loop_contract":',
        '"public_feedback_summary":',
        '"public_curriculum_evidence_bundle":',
        '"curriculum_design_contract":',
        '"required_candidate_package":',
        '"known_good_reward_examples":',
        '"current_recipe":',
        '"public_history":',
        '"candidate_self_audit_checklist":',
    ]
    marker_positions = {marker: prompt_text.find(marker) for marker in ordered_markers}
    _assert(
        all(position >= 0 for position in marker_positions.values())
        and [marker_positions[marker] for marker in ordered_markers]
        == sorted(marker_positions.values()),
        f"{context} prompt must use stable task-first top-level ordering, got: {marker_positions}",
    )


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text).strip() + "\n", encoding="utf-8")


def test_public_history_summary_tracks_curriculum_attribution_streak() -> None:
    history = [
        _projected_history_round(
            "accepted_reward_knob_1",
            accepted=True,
            pure_knob=True,
            gain_attributable_to="reward_code",
        ),
        _projected_history_round(
            "rejected_curriculum_probe",
            accepted=False,
            pure_knob=False,
            gain_attributable_to="curriculum",
        ),
        _projected_history_round(
            "accepted_reward_knob_2",
            accepted=True,
            pure_knob=True,
            gain_attributable_to="reward_code",
        ),
    ]
    summary = build_public_history_summary(history)
    attribution = summary.get("curriculum_attribution_summary")
    _assert(isinstance(attribution, dict), f"public feedback summary must include curriculum attribution summary: {summary}")
    _assert(
        attribution.get("accepted_with_curriculum_lineage_count") == 2,
        f"attribution summary should count accepted rounds with lineage: {attribution}",
    )
    _assert(
        attribution.get("current_accepted_pure_knob_streak") == 2,
        f"attribution summary should track consecutive accepted pure-knob candidates: {attribution}",
    )
    _assert(
        attribution.get("warning") == "accepted_curriculum_degenerated_to_pure_knob_streak",
        f"pure-knob accepted streak must be explicit prompt-facing feedback: {attribution}",
    )
    latest = attribution.get("latest_accepted_attribution") or {}
    _assert(latest.get("pure_knob_candidate") is True, f"latest accepted attribution missing pure-knob flag: {attribution}")
    _assert(
        latest.get("gain_attributable_to") == "reward_code",
        f"latest accepted attribution should preserve public gain attribution: {attribution}",
    )


def _projected_history_round(
    candidate_id: str,
    *,
    accepted: bool,
    pure_knob: bool,
    gain_attributable_to: str,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "decision": "accept" if accepted else "reject",
        "accepted": accepted,
        "reason": "selftest",
        "public_result": {
            "public_dev": {
                "success_rate_stats": {"mean": 0.1, "count": 3},
                "seed_results": [],
            },
            "public_accept": "withheld_from_prompt_history",
        },
        "curriculum_lineage": {
            "status": "pass",
            "compiler_version": "selftest",
            "stage_count": 1 if pure_knob else 3,
            "execution_mode": "env_step_progress",
            "capabilities": ["reward_weight_scale"] if pure_knob else ["external_push_curriculum"],
            "attribution": {
                "reward_code_changed": pure_knob,
                "curriculum_changed": not pure_knob,
                "diagnostics_changed": False,
                "capability_changed": not pure_knob,
                "gain_attributable_to": gain_attributable_to,
                "pure_knob_candidate": pure_knob,
            },
        },
    }


def _hard_supported() -> bool:
    caps = probe_isolation_capabilities()
    return bool(caps.get("network_namespace_supported") and caps.get("pid_namespace_supported") and caps.get("landlock_abi_version"))


def _local_config(tmp_root: Path, *, require_dynamic: bool) -> Path:
    verifier = copy.deepcopy(load_mapping(VERIFIER_CONFIG_PATH))
    verifier["dynamic"]["require_phase8c_isolation"] = require_dynamic
    verifier_path = tmp_root / "phase8d_verifier_for_8e_selftest.json"
    _write_json(verifier_path, verifier)

    config = copy.deepcopy(load_mapping(CONFIG_PATH))
    config["verifier"]["config_path"] = str(verifier_path)
    config["paths"]["ledger_filename"] = "phase8e_ledger.jsonl"
    config_path = tmp_root / "phase8e_loop_for_selftest.json"
    _write_json(config_path, config)
    return config_path


def _public_result(*, include_accept: bool, mean: float = 0.08) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": "humanoid_agent.phase8e.public_result.v1",
        "status": "pass",
        "public_dev": {
            "success_rate_stats": {"mean": mean, "standard_error": 0.01, "count": 3},
            "seed_results": [
                {"success_rate": mean, "failure_breakdown": {"fall_forward": 0.5}},
                {"success_rate": mean + 0.01, "failure_breakdown": {"fall_forward": 0.4}},
                {"success_rate": 0.0, "failure_breakdown": {"fall_forward": 0.8}},
            ],
        },
        "budget_accounting": {"status": "pass", "summary": "within scaffold budget contract"},
        "ood_stress_gate": {"status": "pass", "summary": "synthetic selftest gate"},
        "motion_quality_gate": {"status": "pass", "summary": "synthetic selftest gate"},
    }
    if include_accept:
        result["public_accept"] = {
            "success_rate_stats": {"mean": mean, "standard_error": 0.01, "count": 3, "max": 0.820012},
            "seed_results": [
                {"success_rate": mean},
                {"success_rate": mean + 0.01},
                {"success_rate": mean + 0.02},
            ],
        }
    return result


def _curriculum_lineage() -> dict[str, Any]:
    return {
        "schema_version": "humanoid_agent.phase8e.curriculum_lineage.v1",
        "public_summary": {
            "schema_version": "humanoid_agent.phase8e.curriculum_lineage.public.v1",
            "status": "pass",
            "compiler_version": "phase8e-selftest",
            "stage_count": 2,
            "execution_mode": "env_step_progress",
            "capabilities": ["external_push_curriculum", "train_only_dr_scale"],
            "attribution": {
                "reward_code_changed": False,
                "curriculum_changed": True,
                "diagnostics_changed": True,
                "capability_changed": True,
                "gain_attributable_to": "curriculum",
                "pure_knob_candidate": False,
            },
        },
        "raw_curriculum_design": {
            "stage_list": [
                {"name": "balance_warmup", "progress_range": [0.0, 0.35]},
                {"name": "target_consolidation", "progress_range": [0.35, 1.0]},
            ]
        },
        "effective_curriculum": {
            "compiler_version": "phase8e-selftest",
            "stages": [
                {"name": "balance_warmup", "progress_range": [0.0, 0.35]},
                {"name": "target_consolidation", "progress_range": [0.35, 1.0]},
            ],
        },
    }


def _make_leaking_candidate(root: Path) -> Path:
    candidate = root / "leaking_candidate"
    shutil.copytree(DECLARATIVE_CANDIDATE, candidate)
    _write_text(candidate / "rationale.md", "This candidate mentions iid_hidden_eval and must be rejected.")
    _write_json(candidate / "operation_diff.json", {"note": "iid_hidden_eval", "value": 2001})
    return candidate


def main() -> int:
    test_public_history_summary_tracks_curriculum_attribution_streak()
    hard_supported = _hard_supported()
    with tempfile.TemporaryDirectory(prefix="phase8e_selftest_") as tmp:
        tmp_root = Path(tmp)
        config_path = _local_config(tmp_root, require_dynamic=hard_supported)
        config = load_mapping(config_path)

        assert_public_only_payload({"metric": 0.820012}, config=config, context="benign_float")
        try:
            assert_public_only_payload({"seed": 1001}, config=config, context="protected_seed")
        except Phase8EError:
            pass
        else:
            raise AssertionError("protected numeric token must be rejected")
        # A forbidden eval seed embedded inside an alphanumeric blob (e.g. a sha256
        # fingerprint) must NOT trip the numeric-token guard, or benign ledger
        # records would spuriously fail the public-only assertion.
        assert_public_only_payload(
            {"fingerprint": "3f9a1001b7c2001de"}, config=config, context="hex_fingerprint_not_a_seed"
        )

        # A failed gate must carry a public-safe failures digest into prompt history
        # so the next round learns the concrete cause (e.g. exploded action magnitude).
        raw_public_result = {
            "schema_version": "humanoid_agent.phase8e.public_result.v1",
            "status": "pass",
            "public_dev": {"success_rate_stats": {"mean": 0.0, "count": 3}, "seed_results": []},
            "motion_quality_gate": {
                "status": "fail",
                "summary": "minimal_public_motion_sanity",
                "failures": [
                    {"run_index": 1, "metric": "mean_action_l2", "actual": 295781120.64, "limit": 1000.0, "reason": "limit_exceeded"},
                    {"run_index": 2, "metric": "mean_action_l2", "actual": 293339796.48, "limit": 1000.0, "reason": "limit_exceeded"},
                ],
            },
            "budget_accounting": {"status": "pass"},
            "ood_stress_gate": {"status": "not_run"},
        }
        projected = project_public_result(raw_public_result)
        digest = projected["motion_quality_gate"].get("failures", [])
        _assert(any(item.get("metric") == "mean_action_l2" for item in digest), "projected motion gate must keep a failures digest")
        _assert(len(digest) == 1, "failures digest must dedup by metric (worst offender only)")
        history = _prompt_history_public_result(projected)
        hist_failures = history["motion_quality_gate"].get("failures", [])
        _assert(any(item.get("metric") == "mean_action_l2" for item in hist_failures), "prompt history must surface the failing metric to the next round")
        duplicate_history_summary = build_public_history_summary(
            [
                {
                    "candidate_id": "duplicate_public_feedback_probe",
                    "accepted": True,
                    "public_result": {
                        "public_dev": {
                            "success_rate_stats": {"mean": 0.1, "count": 1},
                            "seed_results": [{"success_rate": 0.1, "failure_breakdown": {"fall_forward": 0.5}}],
                        },
                    },
                },
                {
                    "candidate_id": "duplicate_public_feedback_probe",
                    "accepted": True,
                    "public_result": {
                        "public_dev": {
                            "success_rate_stats": {"mean": 0.1, "count": 1},
                            "seed_results": [{"success_rate": 0.1, "failure_breakdown": {"fall_forward": 0.5}}],
                        },
                    },
                },
            ]
        )
        _assert(
            duplicate_history_summary.get("trend_vs_previous_accepted", {}).get("direction") == "flat",
            f"duplicate public summaries must compare against the immediately previous accepted round: {duplicate_history_summary}",
        )
        shaped_failure_history_summary = build_public_history_summary(
            [
                {
                    "candidate_id": "shaped_failure_probe",
                    "accepted": False,
                    "decision": "reject",
                    "reason": "absolute_accept_floor_not_met",
                    "public_result": {
                        "public_dev": {
                            "success_rate_stats": {"mean": 0.0, "count": 3},
                            "seed_results": [
                                {
                                    "success_rate": 0.0,
                                    "failure_breakdown": {
                                        "counts": {"fall_forward": 64},
                                        "dominant_failure": "fall_forward",
                                        "rates": {"fall_forward": 1.0},
                                        "total": 64,
                                    },
                                }
                            ],
                        },
                    },
                }
            ]
        )
        dominant_failure = shaped_failure_history_summary["latest_round"].get("dominant_failure")
        _assert(
            dominant_failure and dominant_failure.get("label") == "fall_forward",
            f"public feedback summary must preserve the actionable failure label, not total: {dominant_failure}",
        )
        _assert(
            dominant_failure.get("count") == 64
            and dominant_failure.get("denominator") == 64
            and abs(float(dominant_failure.get("rate")) - 1.0) < 1e-12,
            f"dominant_failure.rate must be a proportion while count stays separate: {dominant_failure}",
        )
        success_only_seed_summary = build_public_history_summary(
            [
                {
                    "candidate_id": "dominant_failure_success_only_seed_probe",
                    "accepted": False,
                    "decision": "reject",
                    "reason": "absolute_accept_floor_not_met",
                    "public_result": {
                        "public_dev": {
                            "success_rate_stats": {"mean": 0.75, "count": 2},
                            "seed_results": [
                                {
                                    "success_rate": 1.0,
                                    "failure_breakdown": {
                                        "counts": {"success": 128},
                                        "total": 128,
                                    },
                                },
                                {
                                    "success_rate": 0.5,
                                    "failure_breakdown": {
                                        "counts": {"fall_forward": 64, "success": 64},
                                        "dominant_failure": "fall_forward",
                                        "rates": {"fall_forward": 0.5, "success": 0.5},
                                        "total": 128,
                                    },
                                },
                            ],
                        },
                    },
                }
            ]
        )
        success_only_dominant = success_only_seed_summary["latest_round"].get("dominant_failure")
        _assert(
            success_only_dominant
            and success_only_dominant.get("label") == "fall_forward"
            and success_only_dominant.get("count") == 64
            and success_only_dominant.get("denominator") == 256
            and success_only_dominant.get("seed_count") == 2
            and abs(float(success_only_dominant.get("rate")) - 0.25) < 1e-12,
            f"dominant_failure denominator must include success-only public seeds: {success_only_dominant}",
        )
        rate_only_summary = build_public_history_summary(
            [
                {
                    "candidate_id": "dominant_failure_rate_only_probe",
                    "accepted": False,
                    "decision": "reject",
                    "reason": "absolute_accept_floor_not_met",
                    "public_result": {
                        "public_dev": {
                            "success_rate_stats": {"mean": 0.5, "count": 2},
                            "seed_results": [
                                {"success_rate": 0.5, "failure_breakdown": {"fall_forward": 0.5}},
                                {"success_rate": 0.5, "failure_breakdown": {"fall_backward": 0.5}},
                            ],
                        },
                    },
                }
            ]
        )
        rate_only_dominant = rate_only_summary["latest_round"].get("dominant_failure")
        _assert(
            rate_only_dominant
            and rate_only_dominant.get("seed_count") == 2
            and abs(float(rate_only_dominant.get("rate")) - 0.25) < 1e-12,
            f"rate-only dominant_failure must average missing labels over all valid public seeds: {rate_only_dominant}",
        )

        exit_code, prepared = run_phase8e_round(config_path=config_path, run_dir=tmp_root / "prepared")
        _assert(exit_code == 0 and prepared["status"] == "prepared", "prompt-only round should prepare successfully")
        _assert(Path(prepared["prompt"]["messages_file"]).is_file(), "prompt messages must be written")
        prepared_payload = json.loads((tmp_root / "prepared" / "phase8e_prompt_payload.json").read_text(encoding="utf-8"))
        prepared_prompt_text = (tmp_root / "prepared" / "phase8e_prompt.txt").read_text(encoding="utf-8")
        prepared_feedback_summary = prepared_payload.get("public_feedback_summary")
        _assert(isinstance(prepared_feedback_summary, dict), "prompt must include runner-computed public feedback summary")
        _assert(
            prepared_feedback_summary.get("status") == "no_public_history"
            and prepared_feedback_summary.get("round_count") == 0,
            f"first round feedback summary should explicitly report no public history: {prepared_feedback_summary}",
        )
        curriculum_evidence = prepared_payload.get("public_curriculum_evidence_bundle")
        _assert(isinstance(curriculum_evidence, dict), "prompt must include public_curriculum_evidence_bundle")
        _assert(
            curriculum_evidence.get("schema_version") == "humanoid_agent.phase8e.public_curriculum_evidence.v1",
            f"unexpected curriculum evidence schema: {curriculum_evidence}",
        )
        registry = curriculum_evidence.get("capability_registry")
        _assert(isinstance(registry, dict), "curriculum evidence must include capability_registry")
        registry_ids = [item.get("id") for item in registry.get("capabilities", []) if isinstance(item, dict)]
        _assert(
            registry_ids
            == [
                "reward_weight_scale",
                "external_push_curriculum",
                "train_only_dr_scale",
                "reward_only_push_signal",
                "recovery_window_diagnostic",
                "public_push_probe",
            ],
            f"V1 registry must be fixed and ordered: {registry_ids}",
        )
        _assert(
            registry.get("first_version_allows_capability_requests") is False,
            f"V1 must not let LLM request new capabilities: {registry}",
        )
        registry_by_id = {item.get("id"): item for item in registry.get("capabilities", []) if isinstance(item, dict)}
        push_signal_capability = registry_by_id.get("reward_only_push_signal")
        _assert(
            push_signal_capability is not None
            and push_signal_capability.get("status") == "available_now"
            and push_signal_capability.get("candidate_recipe_surface") is True,
            f"reward_only_push_signal registry entry must match live prompt-facing push RewardInputs: {push_signal_capability}",
        )
        _assert(
            curriculum_evidence.get("probe_visibility_policy", {}).get("public_accept_probes")
            == "withheld_from_prompt",
            f"public accept probes must not enter the prompt evidence bundle: {curriculum_evidence}",
        )
        effective_schedule_keys = set(curriculum_evidence.get("effective_schedule_keys", []))
        _assert(
            "reward_weight_scale" in effective_schedule_keys,
            f"reward_weight_scale must remain the only prompt-facing effective schedule key: {curriculum_evidence}",
        )
        _assert(
            {"push_magnitude_scale", "push_start_fraction", "push_interval_s", "domain_randomization_scale"}.isdisjoint(
                effective_schedule_keys
            ),
            f"inactive push/DR schedule keys must not be labeled effective: {curriculum_evidence}",
        )

        curriculum_contract = prepared_payload.get("curriculum_design_contract")
        _assert(isinstance(curriculum_contract, dict), "prompt must include curriculum_design_contract")
        _assert(
            curriculum_contract.get("first_version_scope") == "design_with_fixed_registry_no_capability_requests",
            f"V1 curriculum contract must forbid capability requests: {curriculum_contract}",
        )
        _assert(
            "final_consolidation" in curriculum_contract.get("required_sections", []),
            f"curriculum design must require final_consolidation: {curriculum_contract}",
        )
        final_requirements = curriculum_contract.get("final_consolidation_requirements") or {}
        _assert(
            float(final_requirements.get("min_fraction_of_training", 0.0)) >= 0.2,
            f"final consolidation must reserve at least 20 percent of training: {final_requirements}",
        )
        nondegenerate_requirements = curriculum_contract.get("non_degenerate_design_requirements") or {}
        _assert(
            int(nondegenerate_requirements.get("min_stage_count", 0)) >= 2,
            f"curriculum contract must require at least two stages: {nondegenerate_requirements}",
        )
        _assert(
            nondegenerate_requirements.get("must_use_curriculum_capability_beyond_reward_weight_scale") is True,
            f"curriculum contract must not allow pure reward_weight_scale tuning: {nondegenerate_requirements}",
        )
        _assert(
            nondegenerate_requirements.get("reject_reward_only_attribution") is True,
            f"curriculum contract must reject reward-only attribution: {nondegenerate_requirements}",
        )
        artifact_shape = curriculum_contract.get("artifact_shape") or {}
        _assert(
            artifact_shape.get("diagnostics_curriculum_hypothesis_type") == "object",
            f"curriculum contract must require diagnostics curriculum_hypothesis object: {artifact_shape}",
        )
        stage_schema = curriculum_contract.get("stage_schema") or {}
        _assert(
            "capabilities" in stage_schema.get("required_fields", []),
            f"curriculum stage schema must require capabilities exact-id list: {stage_schema}",
        )
        _assert(
            "contiguous" in str(stage_schema.get("progress_coverage", "")).lower(),
            f"curriculum stage schema must require contiguous full-progress coverage: {stage_schema}",
        )
        capability_schema = stage_schema.get("capabilities") or {}
        _assert(
            "external_push_curriculum" in capability_schema.get("must_include_one_of", [])
            and "train_only_dr_scale" in capability_schema.get("must_include_one_of", []),
            f"stage capabilities must list exact fixed registry ids: {capability_schema}",
        )
        final_schema = curriculum_contract.get("final_consolidation_schema") or {}
        _assert(
            {
                "fraction",
                "distribution",
                "includes_target_push",
                "includes_target_robustness_variation",
            }.issubset(set(final_schema.get("required_fields", []))),
            f"final_consolidation schema must name exact required fields: {final_schema}",
        )
        _assert(
            "min_fraction" in final_schema.get("accepted_aliases", {}).get("fraction", []),
            f"final_consolidation schema must document accepted fraction aliases: {final_schema}",
        )
        reject_rules = curriculum_contract.get("degenerate_curriculum_reject_rules") or []
        _assert(
            any("target distribution" in str(rule).lower() for rule in reject_rules),
            f"degenerate rules must reject curricula that never return to target distribution: {reject_rules}",
        )
        _assert(
            any("single-stage" in str(rule).lower() for rule in reject_rules),
            f"degenerate rules must reject single-stage no-op curricula: {reject_rules}",
        )
        _assert(
            any("reward-only" in str(rule).lower() for rule in reject_rules),
            f"degenerate rules must reject reward-only attribution: {reject_rules}",
        )
        _assert_p2_prompt_order(prepared_prompt_text, prepared_payload, "P2 prepared prompt")
        required_package = prepared_payload["required_candidate_package"]
        component_contract = required_package["component_contract"]
        loop_contract = prepared_payload["loop_contract"]
        _assert("candidate_controls" not in loop_contract, "legacy candidate_controls duplicates candidate_allowed_controls")
        _assert(
            loop_contract.get("candidate_control_effectiveness_ref")
            == "required_candidate_package.component_contract.control_effectiveness_contract",
            "loop contract must point to the single control effectiveness contract instead of duplicating labels",
        )
        allowed_controls_text = json.dumps(loop_contract.get("candidate_allowed_controls"), ensure_ascii=False, sort_keys=True)
        _assert_no_inactive_prompt_tokens(allowed_controls_text, "prompt-facing allowed controls")
        _assert(
            "CurriculumDesign" in allowed_controls_text and "operation_diff.json" in allowed_controls_text,
            f"prompt-facing allowed controls must make curriculum design the first-class surface: {allowed_controls_text}",
        )
        _assert(
            "primary train-only curriculum surface" in allowed_controls_text,
            f"prompt-facing allowed controls must identify curriculum as primary, not a reward tweak sidecar: {allowed_controls_text}",
        )
        _assert(
            component_contract.get("builder_signature")
            == "def build_training_components(task_spec, base_env_cfg, base_agent_cfg):",
            "prompt must state the real three-argument candidate builder signature",
        )
        _assert(
            "build_training_components" in component_contract.get("candidate_must_define", []),
            "prompt must say the builder is candidate-defined, not imported from the API module",
        )
        _assert(
            "last_action" in component_contract.get("reward_inputs_fields", [])
            and "action" not in component_contract.get("reward_inputs_fields", []),
            "prompt must expose prompt-visible RewardInputs fields without nonexistent aliases",
        )
        expanded_fields = set(component_contract.get("reward_inputs_fields", []))
        _assert(
            {"base_pos_xy", "projected_gravity", "commanded_base_vel", "prev_action", "action_delta", "joint_pos_error"}.issubset(expanded_fields),
            f"prompt must expose first-batch expanded RewardInputs fields: {expanded_fields}",
        )
        _assert(
            {"push_active", "push_vec_public"}.issubset(expanded_fields),
            f"prompt must expose real runner-instrumented push RewardInputs fields: {expanded_fields}",
        )
        field_expansion = component_contract.get("reward_inputs_field_expansion_report")
        _assert(isinstance(field_expansion, dict), "prompt must include reward input field expansion report")
        _assert(
            field_expansion.get("planned_public_fields") == [],
            f"prompt-facing expansion report must not advertise planned fields before instrumentation: {field_expansion}",
        )
        field_expansion_text = json.dumps(field_expansion, ensure_ascii=False, sort_keys=True)
        _assert_no_inactive_prompt_tokens(field_expansion_text, "prompt field expansion")
        reward_inputs_contract = component_contract.get("reward_inputs_contract")
        _assert(isinstance(reward_inputs_contract, dict), "prompt must expose reward_inputs_contract as structured schema")
        _assert(
            reward_inputs_contract.get("field_order") == component_contract.get("reward_inputs_fields"),
            "reward_inputs_contract field_order must match reward_inputs_fields",
        )
        _assert(
            set(reward_inputs_contract.get("fields", {})) == set(component_contract.get("reward_inputs_fields", [])),
            "reward_inputs_contract fields must exactly match live RewardInputs fields",
        )
        _assert(
            reward_inputs_contract["fields"]["base_pos_xy"]["coordinate_frame"] == "env_origin_relative_world_xy",
            "prompt contract must prevent base_pos_xy world-absolute misuse",
        )
        _assert(
            reward_inputs_contract["fields"]["action_delta"]["provenance"] == "derived_from_action_history_or_current_action_and_zero_previous_action",
            "prompt contract must expose first-step/reset action_delta provenance",
        )
        _assert(
            reward_inputs_contract["fields"]["push_active"]["source_status"] == "runner_owned_push_event_instrumentation",
            f"prompt contract must expose only real push instrumentation metadata: {reward_inputs_contract['fields'].get('push_active')}",
        )
        _assert(
            reward_inputs_contract["fields"]["push_vec_public"]["coordinate_frame"] == "world_xy_velocity_delta",
            f"prompt contract must document push vector coordinates: {reward_inputs_contract['fields'].get('push_vec_public')}",
        )
        full_contract = full_reward_inputs_contract()
        _assert(
            full_contract["fields"]["push_active"]["source_status"] == "runner_owned_push_event_instrumentation",
            "full internal RewardInputs contract must mark push fields as real runner-owned signals",
        )
        _assert(
            {"push_active", "push_vec_public"}.issubset(reward_inputs_contract["fields"]),
            "prompt-facing RewardInputs contract must expose runner-instrumented push signals",
        )
        control_contract = component_contract.get("control_effectiveness_contract")
        _assert(isinstance(control_contract, dict), "prompt must expose control_effectiveness_contract as structured schema")
        _assert(
            control_contract["schedule_controls"]["allowed_keys"] == ["reward_weight_scale"],
            f"prompt-facing schedule keys must include only currently effective controls: {control_contract}",
        )
        _assert(
            control_contract["schedule_controls"]["allowed_keys"] == component_contract["runner_schedule_allowed_keys"],
            "prompt-facing runner schedule keys must have one shared source between runner list and effectiveness contract",
        )
        _assert(
            set(control_contract["schedule_controls"]["effective_keys"]) == {"reward_weight_scale"},
            f"only reward_weight_scale should be currently effective: {control_contract}",
        )
        _assert(
            "accepted_but_ineffective_keys" not in control_contract["schedule_controls"],
            f"prompt-facing control contract must omit ineffective schedule keys instead of advertising them: {control_contract}",
        )
        _assert(
            "loss_terms" not in control_contract["component_controls"],
            f"prompt-facing control contract must not advertise LossTerm until optimizer insertion exists: {control_contract}",
        )
        _assert(
            control_contract["primary_mechanism_required"]
            == "candidate_must_include_runner_compiled_curriculum_design_as_primary_training_surface",
            f"prompt must make runner-compiled curriculum design the primary mechanism: {control_contract}",
        )
        _assert(
            control_contract.get("curriculum_controls", {}).get("surface") == "operation_diff.json.curriculum_design",
            f"prompt-facing control contract must expose curriculum design as a real control surface: {control_contract}",
        )
        _assert(
            control_contract.get("curriculum_controls", {}).get("allowed_capability_source")
            == "public_curriculum_evidence_bundle.capability_registry",
            f"prompt-facing control contract must point at the real capability registry payload path: {control_contract}",
        )
        public_curriculum_evidence = prepared_payload.get("public_curriculum_evidence_bundle")
        _assert(
            isinstance(public_curriculum_evidence, dict)
            and isinstance(public_curriculum_evidence.get("capability_registry"), dict),
            f"prompt payload must expose the referenced capability registry: {public_curriculum_evidence}",
        )
        guidance_text = json.dumps(control_contract.get("guidance"), ensure_ascii=False)
        _assert(
            "first-class runner-compiled training surface" in guidance_text
            and "not a substitute for curriculum design" in guidance_text,
            f"control guidance must demote reward_weight_scale from primary curriculum mechanism: {control_contract}",
        )
        runtime_evidence = control_contract.get("runtime_evidence", {})
        _assert(
            runtime_evidence.get("reward_weight_scale_behavior_probe") == "covered_by_phase8e_training_adapter_selftest",
            f"control effectiveness must be backed by an adapter behavior probe: {control_contract}",
        )
        contract_text = json.dumps(component_contract, ensure_ascii=False, sort_keys=True)
        _assert("safe_usage" not in contract_text and "unsafe_usage" not in contract_text, "contract schema must not contain prompt tutorial fields")
        _assert_no_inactive_prompt_tokens(contract_text, "prompt component contract")
        source_pack = prepared_payload.get("source_pack")
        _assert(isinstance(source_pack, dict), "prompt must include a structured minimal source pack")
        _assert(source_pack.get("schema_version") == "humanoid_agent.phase8e.source_pack.v1", f"unexpected source pack schema: {source_pack}")
        source_access = prepared_payload.get("source_access_policy")
        _assert(isinstance(source_access, dict), "prompt must expose source_access_policy for Layer 2 source-read turns")
        provider_rules_text = json.dumps(
            required_package.get("provider_json_output", {}).get("rules"), ensure_ascii=False
        )
        _assert(
            "curriculum_design" in provider_rules_text
            and "reward-only" in provider_rules_text
            and "reward_weight_scale-only" in provider_rules_text,
            f"provider JSON contract must require curriculum design and reject reward-only attribution: {provider_rules_text}",
        )
        checklist_ids = [
            item.get("id")
            for item in prepared_payload.get("candidate_self_audit_checklist", {}).get("items", [])
            if isinstance(item, dict)
        ]
        _assert(
            "curriculum_design_is_primary_surface" in checklist_ids,
            f"self-audit checklist must include curriculum design as primary surface: {checklist_ids}",
        )
        selection_guidance_text = json.dumps(prepared_payload.get("selection_guidance"), ensure_ascii=False)
        _assert(
            "reward_weight_scale alone" not in selection_guidance_text
            and "runner-compiled CurriculumDesign as the primary training proposal" in selection_guidance_text,
            f"selection guidance must not tell the model reward_weight_scale alone is enough: {selection_guidance_text}",
        )
        _assert(
            source_access.get("enabled") is True
            and source_access.get("ids_are_closed_allowlist") is True
            and source_access.get("request_ids_are_untrusted") is True,
            f"source access policy must be closed allowlist with untrusted request ids: {source_access}",
        )
        _assert(
            source_access.get("max_source_read_turns") == 2,
            f"source access policy must expose the runtime source-read turn budget: {source_access}",
        )
        _assert(
            "source_requests only" in prepared_prompt_text,
            "prompt must explicitly allow a source_requests-only turn when source-read is enabled",
        )
        known_good = prepared_payload.get("known_good_reward_examples")
        _assert(isinstance(known_good, dict), "prompt must include known_good_reward_examples metadata")
        _assert(
            known_good.get("schema_version") == "humanoid_agent.phase8e.known_good_reward_examples.v1",
            f"unexpected known-good examples schema: {known_good}",
        )
        known_examples = known_good.get("examples", [])
        _assert(len(known_examples) == 2, f"prompt should expose exactly two minimal known-good examples: {known_good}")
        known_ids = {item.get("id") for item in known_examples if isinstance(item, dict)}
        _assert(
            {"upright_action_smoothness", "xy_velocity_command_tracking"}.issubset(known_ids),
            f"known-good examples must cover upright/action and xy command tracking: {known_good}",
        )
        _assert(
            all(item.get("source_item_id") == "examples/known_good_reward_terms.py" for item in known_examples),
            f"known-good examples must point at the audited source pack item: {known_good}",
        )
        _assert(
            any("request examples/known_good_reward_terms.py via source_requests" in note for note in known_good.get("notes", [])),
            f"known-good examples must tell the model how to fetch the full runnable example body: {known_good}",
        )
        source_items = source_pack.get("items")
        _assert(isinstance(source_items, dict), f"source pack items must be keyed by logical id: {source_pack}")
        expected_ids = [
            "contract/reward_inputs_contract.json",
            "contract/reward_inputs_field_expansion_report.json",
            "contract/control_effectiveness_contract.json",
            "examples/known_good_reward_terms.py",
            "current_best/training_code/components.py",
            "public_failures/recent_training_errors.json",
        ]
        _assert(list(source_items) == expected_ids, f"source pack logical ids must follow construction order: {source_items.keys()}")
        _assert(
            source_access.get("allowed_logical_ids") == list(source_items),
            f"source access policy must derive allowed ids from source pack insertion order: {source_access}",
        )
        _assert(
            source_pack.get("policy", {}).get("source_requests_service") == "enabled_closed_allowlist",
            f"source pack policy must advertise the enabled closed source-read service: {source_pack.get('policy')}",
        )
        _assert(
            "src/humanoid_agent/cold_start_lab/training_adapter.py" not in source_items,
            "minimal source pack must not expose the full runner adapter source by default",
        )
        for logical_id, item in source_items.items():
            _assert("content" not in item and "parsed" not in item, f"prompt source pack item must be manifest-only: {logical_id}: {item}")
            _assert(
                item.get("content_available") is True and item.get("content_sha256") and item.get("bytes"),
                f"manifest source item must keep retrieval metadata: {logical_id}: {item}",
            )
        for schema in (
            "humanoid_agent.phase8e.reward_inputs_contract.v1",
            "humanoid_agent.phase8e.reward_inputs_field_expansion.v1",
            "humanoid_agent.phase8e.control_effectiveness.v1",
        ):
            _assert(prepared_prompt_text.count(schema) == 1, f"P1 prompt must include {schema} exactly once")
        _assert(
            source_items["contract/reward_inputs_contract.json"].get("see")
            == "required_candidate_package.component_contract.reward_inputs_contract",
            "contract source manifest must point at the single prompt contract authority",
        )
        current_recipe_projection = prepared_payload.get("current_recipe") or {}
        current_fingerprint = current_recipe_projection.get("fingerprint") or {}
        _assert(
            current_recipe_projection.get("candidate_dir") == "current_recipe/phase8a_noop_candidate",
            f"prompt-facing current recipe should expose a logical ref, not a host path: {current_recipe_projection}",
        )
        _assert(
            current_recipe_projection.get("baseline_role") == "bootstrap_scaffold_not_current_best",
            f"initial prompt should mark the no-op current recipe as bootstrap-only, not current best: {current_recipe_projection}",
        )
        _assert(
            current_recipe_projection.get("accepted_history_available") is False,
            f"initial prompt should say there is no accepted recipe history yet: {current_recipe_projection}",
        )
        _assert(
            current_recipe_projection.get("source_item_id") == "current_best/training_code/components.py",
            f"prompt-facing current recipe should point directly at its source item: {current_recipe_projection}",
        )
        _assert(
            {"path", "manifest_path", "entrypoint_path"}.isdisjoint(current_fingerprint),
            f"prompt-facing fingerprint must not expose host paths: {current_fingerprint}",
        )
        provider_json_output = required_package.get("provider_json_output") or {}
        _assert(
            provider_json_output.get("allowed_first_response_modes") == ["source_requests", "candidate_package"],
            f"provider response protocol must explicitly allow source_requests before candidate authoring: {provider_json_output}",
        )
        source_request_shape = provider_json_output.get("source_requests", {})
        _assert(
            source_request_shape.get("enabled") is True
            and source_request_shape.get("max_source_read_turns") == source_access.get("max_source_read_turns")
            and (source_request_shape.get("shape", {}).get("source_requests") or [{}])[0].get("id"),
            f"provider response protocol must include a source_requests shape when enabled: {provider_json_output}",
        )
        response_skeleton = provider_json_output.get("response_skeleton", {})
        _assert(
            set((response_skeleton.get("candidate_package", {}).get("files") or {}))
            == {
                "candidate_manifest.yaml",
                "training_code/components.py",
                "rationale.md",
                "operation_diff.json",
                "diagnostics.json",
            },
            f"provider response skeleton must cover exactly the allowed candidate files: {response_skeleton}",
        )
        minimal_component_template = component_contract.get("minimal_component_template")
        _assert(
            isinstance(minimal_component_template, str)
            and "def build_training_components(task_spec, base_env_cfg, base_agent_cfg)" in minimal_component_template
            and "TrainingComponents(" in minimal_component_template
            and "RewardTerm(" in minimal_component_template,
            f"component contract must include a minimal runnable component template: {minimal_component_template}",
        )
        prompt_package_for_source_read = build_phase8e_prompt_package(
            config=config,
            ledger=[],
            current_recipe_dir=DECLARATIVE_CANDIDATE,
        )
        source_store = prompt_package_for_source_read.get("source_store")
        _assert(isinstance(source_store, dict), "prompt package must keep server-side source_store for source-read")
        _assert(
            "source_store" not in prepared_payload and "source_store" not in prepared_prompt_text,
            "server-side source_store must not be serialized into the prompt payload/text",
        )
        no_source_read_config = copy.deepcopy(config)
        no_source_read_config["provider_inner_loop"] = dict(no_source_read_config.get("provider_inner_loop", {}))
        no_source_read_config["provider_inner_loop"]["enabled"] = False
        no_source_read_prompt = build_phase8e_prompt_package(
            config=no_source_read_config,
            ledger=[],
            current_recipe_dir=DECLARATIVE_CANDIDATE,
        )
        no_source_read_payload = no_source_read_prompt["payload"]
        no_source_access = no_source_read_payload.get("source_access_policy") or {}
        _assert(
            no_source_access.get("enabled") is False,
            f"prompt must disable source_access_policy when provider source-read is disabled: {no_source_access}",
        )
        no_source_protocol = (no_source_read_payload.get("required_candidate_package") or {}).get("provider_json_output", {})
        _assert(
            no_source_protocol.get("allowed_first_response_modes") == ["candidate_package"],
            f"prompt must forbid source_requests when provider source-read is disabled: {no_source_protocol}",
        )
        _assert(
            "Source-read is not available in this author call." in no_source_read_prompt["messages"][0]["content"],
            "system prompt must say source-read is unavailable when the inner loop is disabled",
        )
        no_source_known_good = no_source_read_payload.get("known_good_reward_examples") or {}
        _assert(
            any("Source-read is disabled in this author call" in note for note in no_source_known_good.get("notes", [])),
            f"known-good examples must offer a fallback note when source-read is disabled: {no_source_known_good}",
        )
        zero_budget_source_read_config = copy.deepcopy(config)
        zero_budget_source_read_config["provider_inner_loop"] = dict(
            zero_budget_source_read_config.get("provider_inner_loop", {})
        )
        zero_budget_source_read_config["provider_inner_loop"]["enabled"] = True
        zero_budget_source_read_config["provider_inner_loop"]["max_source_read_turns"] = 0
        zero_budget_prompt = build_phase8e_prompt_package(
            config=zero_budget_source_read_config,
            ledger=[],
            current_recipe_dir=DECLARATIVE_CANDIDATE,
        )
        zero_budget_payload = zero_budget_prompt["payload"]
        zero_budget_access = zero_budget_payload.get("source_access_policy") or {}
        zero_budget_protocol = (zero_budget_payload.get("required_candidate_package") or {}).get(
            "provider_json_output",
            {},
        )
        _assert(
            zero_budget_access.get("enabled") is False
            and zero_budget_access.get("max_source_read_turns") == 0
            and zero_budget_protocol.get("allowed_first_response_modes") == ["candidate_package"],
            f"prompt must disable source-read when runtime source-read turn budget is zero: access={zero_budget_access}, protocol={zero_budget_protocol}",
        )
        source_store_items = source_store.get("items", {})
        _assert(isinstance(source_store_items, dict), f"source_store items must be keyed by logical id: {source_store}")
        _assert(
            all("parsed" not in item for item in source_store_items.values()),
            f"source_store JSON/text items must keep a single text representation: {source_store_items}",
        )
        example_source = source_store["items"]["examples/known_good_reward_terms.py"]["content"]
        _assert(
            "def build_training_components(task_spec, base_env_cfg, base_agent_cfg)" in example_source
            and "RewardTerm(" in example_source
            and "reward_weight_scale" in example_source,
            "known-good source example must be runnable component code, not prose",
        )
        _assert("xy_velocity_command_tracking" in example_source, "known-good source must include a correct xy vector slicing example")
        namespace: dict[str, Any] = {}
        exec(example_source, namespace)  # runner-owned example source, not candidate input.
        example_components = namespace["build_training_components"](None, None, None)
        example_inputs = RewardInputs(
            base_height=[0.8, 0.9],
            roll=[0.1, -0.1],
            pitch=[0.05, 0.02],
            base_pos_xy=[[0.0, 0.0], [0.1, -0.1]],
            base_lin_vel=[[0.2, 0.0, 0.1], [0.0, -0.1, 0.0]],
            base_ang_vel=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            projected_gravity=[[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]],
            commanded_base_vel=[[0.1, 0.0, 0.0], [0.0, -0.1, 0.0]],
            joint_pos=[[0.0, 0.0], [0.0, 0.0]],
            joint_vel=[[0.0, 0.0], [0.0, 0.0]],
            joint_pos_error=[[0.0, 0.0], [0.0, 0.0]],
            last_action=[[0.1, 0.2], [0.0, 0.1]],
            prev_action=[[0.0, 0.1], [0.0, 0.0]],
            action_delta=[[0.1, 0.1], [0.0, 0.1]],
            push_active=[0.0, 0.0],
            push_vec_public=[[0.0, 0.0], [0.0, 0.0]],
            progress_fraction=[0.25, 0.25],
        )
        reward_summaries = evaluate_reward_terms(example_components, example_inputs, num_envs=2)
        _assert(
            [summary.shape for summary in reward_summaries] == [(2,), (2,)],
            f"known-good RewardTerm examples must return vector[num_envs]: {reward_summaries}",
        )
        schedule_table = precompute_schedule_table(example_components, progress_points=[0.0, 0.5, 1.0])
        _assert(
            "reward_weight_scale" in schedule_table["rows"][1]["values"]["reward_weight_scale_after_balance"],
            f"known-good schedule should use only currently effective reward_weight_scale: {schedule_table}",
        )
        source_payload_summary = resolve_source_requests(
            source_requests=[{"id": "examples/known_good_reward_terms.py", "reason": "verify source store retrieval"}],
            source_pack=prompt_package_for_source_read["payload"]["source_pack"],
            source_store=source_store,
            config=config,
            artifact_path=tmp_root / "source_read_payloads.json",
            max_requests=2,
        )
        _assert(
            source_payload_summary["payloads"][0]["content"] == example_source,
            "source-read must return backing source_store content for manifest-only prompt source_pack",
        )
        current_source = source_store["items"]["current_best/training_code/components.py"]["content"]
        _assert(
            current_source,
            "source_store must include current recipe components.py source or explicit prompt-facing omission note",
        )
        _assert(
            all("content_sha256" in item and item.get("public_safe") is True for item in source_items.values()),
            "every source manifest item must be audited and fingerprinted",
        )
        source_text = json.dumps(source_pack, ensure_ascii=False, sort_keys=True)
        _assert_no_inactive_prompt_tokens(source_text, "P0 prepared source pack")
        for forbidden in ("success_metric", "checkpoint_selector", "eval_config", "iid_hidden", "OPENAI_API_KEY", "DEEPSEEK_API_KEY", "api_key", "secret"):
            _assert(forbidden not in source_text, f"source pack leaked forbidden source token: {forbidden}")
        scrubbed_feedback = scrub_prompt_facing_value(
            {
                "seed_results": [
                    {"success_rate": 0.1, "note": "push_magnitude_scale appeared in a public note"},
                    {"success_rate": 0.2, "note": "clean"},
                ],
                "schedule_keys": ["push_magnitude_scale", "reward_weight_scale"],
            }
        )
        _assert(
            len(scrubbed_feedback["seed_results"]) == 2,
            f"prompt-facing scrub must preserve structured feedback records: {scrubbed_feedback}",
        )
        _assert(
            scrubbed_feedback["seed_results"][0]["note"] == "[omitted: non-prompt-facing training surface]",
            f"prompt-facing scrub must redact inactive string fields in-place: {scrubbed_feedback}",
        )
        _assert(
            scrubbed_feedback["schedule_keys"] == ["reward_weight_scale"],
            f"prompt-facing scrub must remove pure inactive string list items: {scrubbed_feedback}",
        )
        accepted_eval_with_inactive_feedback = {
            "candidate": {"manifest": {"candidate_id": "accepted_eval_scrub_probe"}},
            "decision": {
                "decision": "accept",
                "accepted": True,
                "public_result": {
                    "public_dev": {"note": "push_magnitude_scale appeared in a public diagnostic note"},
                    "ood_stress_gate": {"status": "pass"},
                    "motion_quality_gate": {"status": "pass"},
                    "budget_accounting": {"status": "pass"},
                    "public_accept": {"status": "withheld"},
                },
            },
        }
        eval_scrub_prompt = build_phase8e_prompt_package(
            config=config,
            ledger=[accepted_eval_with_inactive_feedback],
            current_recipe_dir=None,
        )
        _assert_no_inactive_prompt_tokens(
            eval_scrub_prompt["prompt_text"],
            "P0 accepted public eval summary source pack",
        )
        _assert(
            "[omitted: non-prompt-facing training surface]" in eval_scrub_prompt["prompt_text"],
            "accepted public eval summary should scrub inactive feedback instead of failing prompt construction",
        )
        _assert(
            component_contract.get("training_components_fields")
            == ["name", "reward_terms", "schedules", "diagnostics", "metadata", "rationale"],
            "prompt-facing TrainingComponents fields must include only active candidate surfaces",
        )
        _assert(
            "LossTerm" not in component_contract.get("component_classes_to_import", []),
            "prompt must not suggest importing LossTerm until optimizer insertion exists",
        )
        _assert_no_inactive_prompt_tokens(prepared_prompt_text, "P0 prepared prompt")
        checklist = prepared_payload.get("candidate_self_audit_checklist")
        _assert(isinstance(checklist, dict), "prompt must include a candidate self-audit checklist")
        _assert(
            checklist.get("role") == "guidance_only_not_trusted_for_acceptance",
            f"checklist must be generation guidance, not an acceptance signal: {checklist}",
        )
        _assert(
            checklist.get("candidate_must_not_return_self_certification") is True,
            f"checklist must explicitly forbid self-certification output fields: {checklist}",
        )
        checklist_items = checklist.get("items")
        _assert(isinstance(checklist_items, list) and checklist_items, f"checklist items missing: {checklist}")
        items_by_id = {item.get("id"): item for item in checklist_items if isinstance(item, dict)}
        expected_checklist_ids = {
            "component_outputs_match_public_contract",
            "component_names_unique",
            "no_unsupported_imports_or_runner_hooks",
            "no_private_eval_or_metric_dependency",
            "json_output_matches_allowed_files",
            "small_step_from_current_best_when_available",
        }
        _assert(
            expected_checklist_ids.issubset(items_by_id),
            f"checklist missing required derived guidance items: {items_by_id}",
        )
        shape_item = items_by_id["component_outputs_match_public_contract"]
        _assert(
            shape_item.get("derived_from", {}).get("reward_inputs_field_order")
            == "required_candidate_package.component_contract.reward_inputs_contract.field_order",
            f"shape checklist must cite reward_inputs_contract as source: {shape_item}",
        )
        _assert(
            shape_item.get("reward_inputs_field_order") == reward_inputs_contract.get("field_order"),
            f"shape checklist must derive live fields from reward_inputs_contract: {shape_item}",
        )
        _assert(
            "ineffective_schedule_keys_not_primary_mechanism" not in items_by_id,
            f"prompt checklist must not mention hidden ineffective schedule keys: {items_by_id}",
        )
        files_item = items_by_id["json_output_matches_allowed_files"]
        _assert(
            files_item.get("allowed_top_level_files") == required_package.get("allowed_top_level_files"),
            f"file-output checklist must derive allowed files from required package: {files_item}",
        )
        strategy_item = items_by_id["small_step_from_current_best_when_available"]
        _assert(
            strategy_item.get("accepted_recipe_available") is False
            and "bootstrap scaffold" in strategy_item.get("rule", ""),
            f"bootstrap-only rounds must not be framed as small deltas from a current best: {strategy_item}",
        )
        provider_json_text = json.dumps(required_package.get("provider_json_output"), ensure_ascii=False, sort_keys=True)
        for unexpected in ("candidate_self_audit_checklist", "self_certification", "self_audit", "checklist.md"):
            _assert(unexpected not in provider_json_text, f"provider JSON output must not request self-certification: {unexpected}")

        exit_code, pending = run_phase8e_round(
            config_path=config_path,
            run_dir=tmp_root / "pending",
            candidate_dir=DECLARATIVE_CANDIDATE,
            append_ledger=True,
            require_server_enforcement=hard_supported,
        )
        _assert(exit_code == 0 and pending["status"] == "pending_training", "verified candidate without public result must be pending")
        _assert(pending["verifier"]["status"] == "pass", "declarative candidate must pass verifier")
        _assert(len(load_ledger(tmp_root / "pending" / "phase8e_ledger.jsonl")) == 1, "pending ledger record missing")

        dev_result = tmp_root / "public_dev_only.json"
        _write_json(dev_result, _public_result(include_accept=False))
        exit_code, promising = run_phase8e_round(
            config_path=config_path,
            run_dir=tmp_root / "promising",
            candidate_dir=DECLARATIVE_CANDIDATE,
            public_result_path=dev_result,
            require_server_enforcement=hard_supported,
        )
        _assert(exit_code == 0 and promising["status"] == "promising", "public_dev floor should only mark promising")
        _assert(promising["decision"]["accepted"] is False, "promising candidate must not be accepted")

        lab = (tmp_root / "lab").resolve()
        ledger_path = lab / "phase8e_ledger.jsonl"
        accept_result = tmp_root / "public_accept.json"
        _write_json(accept_result, _public_result(include_accept=True, mean=0.08))
        try:
            run_phase8e_round(
                config_path=config_path,
                run_dir=tmp_root / "round_accept_without_inherit",
                lab_dir=lab,
                candidate_dir=DECLARATIVE_CANDIDATE,
                public_result_path=accept_result,
                append_ledger=True,
                inherit_if_accepted=False,
                require_server_enforcement=hard_supported,
            )
        except Phase8EError:
            pass
        else:
            raise AssertionError("accepted ledger append must require explicit immutable inheritance")

        try:
            run_phase8e_round(
                config_path=config_path,
                run_dir=tmp_root / "round_accept_without_curriculum_lineage",
                lab_dir=lab,
                candidate_dir=DECLARATIVE_CANDIDATE,
                public_result_path=accept_result,
                append_ledger=True,
                inherit_if_accepted=True,
                require_server_enforcement=hard_supported,
            )
        except Phase8EError:
            pass
        else:
            raise AssertionError("accepted ledger append must require runner curriculum lineage")
        _assert(not ledger_path.exists() or not load_ledger(ledger_path), "missing-lineage accept must not append ledger")

        malformed_lineage_path = tmp_root / "malformed_curriculum_lineage.json"
        _write_json(
            malformed_lineage_path,
            {
                "schema_version": "humanoid_agent.phase8e.curriculum_lineage.v1",
                "effective_curriculum": {"stages": []},
            },
        )
        try:
            run_phase8e_round(
                config_path=config_path,
                run_dir=tmp_root / "round_accept_malformed_curriculum_lineage",
                lab_dir=lab,
                candidate_dir=DECLARATIVE_CANDIDATE,
                public_result_path=accept_result,
                curriculum_lineage_path=malformed_lineage_path,
                append_ledger=True,
                inherit_if_accepted=True,
                require_server_enforcement=hard_supported,
            )
        except Phase8EError:
            pass
        else:
            raise AssertionError("accepted ledger append must reject non-projectable curriculum lineage")
        _assert(not ledger_path.exists() or not load_ledger(ledger_path), "malformed-lineage accept must not append ledger")

        empty_public_lineage_path = tmp_root / "empty_public_curriculum_lineage.json"
        _write_json(
            empty_public_lineage_path,
            {
                "schema_version": "humanoid_agent.phase8e.curriculum_lineage.v1",
                "public_summary": {},
                "effective_curriculum": {"stages": []},
            },
        )
        try:
            run_phase8e_round(
                config_path=config_path,
                run_dir=tmp_root / "round_accept_empty_public_curriculum_lineage",
                lab_dir=lab,
                candidate_dir=DECLARATIVE_CANDIDATE,
                public_result_path=accept_result,
                curriculum_lineage_path=empty_public_lineage_path,
                append_ledger=True,
                inherit_if_accepted=True,
                require_server_enforcement=hard_supported,
            )
        except Phase8EError:
            pass
        else:
            raise AssertionError("accepted ledger append must reject empty public curriculum lineage")
        _assert(not ledger_path.exists() or not load_ledger(ledger_path), "empty-public-lineage accept must not append ledger")

        curriculum_lineage_path = tmp_root / "curriculum_lineage.json"
        _write_json(curriculum_lineage_path, _curriculum_lineage())
        exit_code, accepted = run_phase8e_round(
            config_path=config_path,
            run_dir=tmp_root / "round_accept",
            lab_dir=lab,
            candidate_dir=DECLARATIVE_CANDIDATE,
            public_result_path=accept_result,
            curriculum_lineage_path=curriculum_lineage_path,
            append_ledger=True,
            inherit_if_accepted=True,
            require_server_enforcement=hard_supported,
        )
        _assert(exit_code == 0 and accepted["status"] == "accept", "public_accept plus gates should accept near-zero baseline candidate")
        _assert(accepted["inheritance"]["snapshot_created"] is True, "accepted recipe must be snapshotted for inheritance")
        _assert(
            accepted.get("curriculum_lineage", {}).get("attribution", {}).get("gain_attributable_to") == "curriculum",
            f"accepted summary must preserve public curriculum lineage: {accepted.get('curriculum_lineage')}",
        )
        snapshot_dir = Path(accepted["inheritance"]["accepted_recipe_snapshot_dir"])
        _assert(snapshot_dir.is_dir() and (snapshot_dir / "candidate_manifest.yaml").is_file(), "immutable accepted-recipe snapshot missing")
        _assert(
            (snapshot_dir / "runner_curriculum_snapshot" / "curriculum_lineage.json").is_file(),
            "accepted recipe snapshot must include runner curriculum lineage",
        )
        _assert(lab in snapshot_dir.parents, "snapshot must live under the stable lab_dir, not the per-round run_dir")
        ledger_records = load_ledger(ledger_path)
        _assert(
            ledger_records[-1].get("curriculum_lineage", {}).get("attribution", {}).get("pure_knob_candidate") is False,
            f"accepted ledger record must preserve public curriculum lineage: {ledger_records[-1]}",
        )

        # BUG-1a regression: a later round with a DIFFERENT run_dir but the same
        # lab_dir must inherit the accepted snapshot, not fall back to initial.
        exit_code, next_round = run_phase8e_round(
            config_path=config_path,
            run_dir=tmp_root / "round_next",
            lab_dir=lab,
            require_server_enforcement=hard_supported,
        )
        current_recipe = next_round.get("current_recipe") or {}
        _assert(
            current_recipe.get("manifest", {}).get("candidate_id") == "phase8e_declarative_curriculum_candidate",
            "next round must inherit the accepted recipe across separate run_dirs",
        )
        inherited_dir = Path(current_recipe.get("candidate_dir", ""))
        _assert(lab in inherited_dir.parents, "P1b: inherited recipe must resolve to the immutable lab snapshot, not the original candidate dir")

        # P1a regression: public_accept must not enter the next-round prompt.
        prompt_payload = json.loads((tmp_root / "round_next" / "phase8e_prompt_payload.json").read_text(encoding="utf-8"))
        next_prompt_recipe = prompt_payload.get("current_recipe") or {}
        _assert(
            next_prompt_recipe.get("candidate_dir") == "current_recipe/phase8e_declarative_curriculum_candidate",
            f"next round prompt-facing current recipe should keep logical ref only: {next_prompt_recipe}",
        )
        _assert(
            next_prompt_recipe.get("source_item_id") == "current_best/training_code/components.py",
            f"next round prompt-facing current recipe should point directly at its source item: {next_prompt_recipe}",
        )
        next_prompt_fingerprint = next_prompt_recipe.get("fingerprint") or {}
        _assert(
            {"path", "manifest_path", "entrypoint_path"}.isdisjoint(next_prompt_fingerprint),
            f"next round prompt-facing fingerprint must not expose host paths: {next_prompt_fingerprint}",
        )
        history = prompt_payload.get("public_history", [])
        _assert(bool(history), "next round prompt should include public history")
        for item in history:
            projected = item.get("public_result") or {}
            _assert(projected.get("public_accept") == "withheld_from_prompt_history", "public_accept must be withheld from prompt history")
            public_dev = projected.get("public_dev") or {}
            _assert(
                public_dev.get("seed_results"),
                f"P3 prompt history must keep per-seed public_dev details after adding aggregate summary: {public_dev}",
            )
            verifier = item.get("verifier") or {}
            _assert("status" in verifier, "prompt history should include public verifier status for candidate learning")
        feedback_summary = prompt_payload.get("public_feedback_summary")
        _assert(isinstance(feedback_summary, dict), "next round prompt must include public feedback summary")
        latest_feedback = feedback_summary.get("latest_public_dev_round") or {}
        _assert(
            latest_feedback.get("candidate_id") == "phase8e_declarative_curriculum_candidate",
            f"P3 latest feedback summary should identify the latest evaluated candidate: {feedback_summary}",
        )
        _assert(
            abs(float(latest_feedback.get("public_dev_mean_success_rate")) - 0.08) < 1e-12,
            f"P3 latest feedback summary must surface public_dev mean success: {feedback_summary}",
        )
        dominant_failure = latest_feedback.get("dominant_failure") or {}
        _assert(
            dominant_failure.get("label") == "fall_forward"
            and dominant_failure.get("seed_count") == 3
            and float(dominant_failure.get("rate")) > 0.5,
            f"P3 feedback summary must surface the dominant public failure without dropping details: {feedback_summary}",
        )
        _assert(
            feedback_summary.get("best_public_dev_round", {}).get("candidate_id")
            == "phase8e_declarative_curriculum_candidate",
            f"P3 feedback summary must include a best-public-dev pointer: {feedback_summary}",
        )
        prompt_text = (tmp_root / "round_next" / "phase8e_prompt.txt").read_text(encoding="utf-8")
        _assert_p2_prompt_order(prompt_text, prompt_payload, "P2 next-round prompt")
        _assert("0.820012" not in prompt_text, "public_accept-only detail must not appear in the prompt text")
        _assert_no_inactive_prompt_tokens(prompt_text, "P0 next-round prompt")
        next_source_pack = prompt_payload.get("source_pack", {})
        next_items = next_source_pack.get("items", {})
        _assert(
            all("content" not in item and "parsed" not in item for item in next_items.values()),
            f"P1 next-round source pack must stay manifest-only: {next_items}",
        )
        inherited_source_manifest = next_items.get("current_best/training_code/components.py", {})
        _assert(
            inherited_source_manifest.get("content_available") is True and "content" not in inherited_source_manifest,
            "next-round prompt source pack must expose current-recipe source as manifest-only",
        )
        next_prompt_package = build_phase8e_prompt_package(
            config=config,
            ledger=load_ledger(ledger_path),
            current_recipe_dir=inherited_dir,
        )
        inherited_source = next_prompt_package["source_store"]["items"]["current_best/training_code/components.py"]["content"]
        _assert_no_inactive_prompt_tokens(inherited_source, "P0 next-round current-best source pack item")
        if "source view omitted" in inherited_source:
            _assert(
                "fn=" not in inherited_source and "RewardTerm(" not in inherited_source,
                "omitted source view must not expose partial callable references",
            )
        else:
            _assert(
                "phase8e_declarative_curriculum_candidate" in inherited_source,
                "visible current-recipe source must preserve inherited recipe identity",
            )
            _assert(
                "upright_survival_margin" in inherited_source and "collapse_vs_recovery_margin" in inherited_source,
                "visible current-recipe source must preserve prompt-visible reward and diagnostic logic",
            )
        eval_summary = json.loads(
            next_prompt_package["source_store"]["items"]
            .get("current_best/public_eval_summary.json", {})
            .get("content", "{}")
        )
        _assert(
            eval_summary.get("public_accept") == "withheld_from_prompt_history",
            "current best public eval summary must keep public_accept withheld in prompt source pack",
        )
        _assert(
            "content" not in next_items.get("current_best/public_eval_summary.json", {}),
            "next-round public eval source must also be manifest-only in prompt",
        )

        synthetic_rejected = {
            "schema_version": "humanoid_agent.phase8e.ledger.v1",
            "created_at_utc": "2026-06-30T00:00:00Z",
            "candidate": {
                "manifest": {"candidate_id": "phase8e_inactive_control_history_probe"},
                "operation_diff": {
                    "note": "Rejected candidate attempted push_magnitude_scale and domain randomization schedules.",
                    "schedule_keys": ["push_magnitude_scale", "domain_randomization_scale"],
                },
            },
            "verifier": {
                "status": "pass",
                "fingerprint": {
                    "path": "/data/yuyangliu/runs/phase_8_cold_start_autoresearch/leaky_candidate",
                    "manifest_path": "/data/yuyangliu/runs/phase_8_cold_start_autoresearch/leaky_candidate/candidate_manifest.yaml",
                    "entrypoint_path": "/data/yuyangliu/runs/phase_8_cold_start_autoresearch/leaky_candidate/training_code/components.py",
                    "sha256": "public-history-probe-sha",
                },
            },
            "decision": {
                "decision": "reject",
                "accepted": False,
                "reason": "synthetic_history_probe",
                "detail": {
                    "public_failure_feedback": {
                        "summary": "Do not repeat push_magnitude_scale from this rejected candidate.",
                    }
                },
            },
        }
        with ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(synthetic_rejected, ensure_ascii=False, sort_keys=True) + "\n")
        exit_code, history_probe = run_phase8e_round(
            config_path=config_path,
            run_dir=tmp_root / "history_leak_probe",
            lab_dir=lab,
            require_server_enforcement=hard_supported,
        )
        _assert(exit_code == 0 and history_probe["status"] == "prepared", "history leak probe round should prepare successfully")
        history_prompt_text = (tmp_root / "history_leak_probe" / "phase8e_prompt.txt").read_text(encoding="utf-8")
        _assert_no_inactive_prompt_tokens(history_prompt_text, "P0 history-leak prompt")
        _assert("/data/yuyangliu" not in history_prompt_text, "P4 prompt history must not expose server absolute paths")
        _assert(
            "manifest_path" not in history_prompt_text and "entrypoint_path" not in history_prompt_text,
            "P4 prompt history fingerprint must drop verifier path keys",
        )
        history_payload = json.loads((tmp_root / "history_leak_probe" / "phase8e_prompt_payload.json").read_text(encoding="utf-8"))
        _assert_p2_prompt_order(history_prompt_text, history_payload, "P2 history-leak prompt")
        history_verifiers = [
            item.get("verifier", {})
            for item in history_payload.get("public_history", [])
            if isinstance(item, dict)
        ]
        _assert(
            any((verifier.get("fingerprint") or {}).get("sha256") == "public-history-probe-sha" for verifier in history_verifiers),
            f"P4 verifier projection should keep non-path fingerprint evidence: {history_verifiers}",
        )
        _assert(
            all(
                {"path", "manifest_path", "entrypoint_path"}.isdisjoint((verifier.get("fingerprint") or {}))
                for verifier in history_verifiers
            ),
            f"P4 verifier projection must scrub path fingerprint keys in every history item: {history_verifiers}",
        )
        history_items = history_payload.get("source_pack", {}).get("items", {})
        _assert(
            all("content" not in item and "parsed" not in item for item in history_items.values()),
            f"P1 history-leak source pack must stay manifest-only: {history_items}",
        )
        failures_item = (
            history_payload.get("source_pack", {})
            .get("items", {})
            .get("public_failures/recent_training_errors.json", {})
        )
        _assert_no_inactive_prompt_tokens(
            json.dumps(failures_item, ensure_ascii=False, sort_keys=True),
            "P0 public-failures source pack item",
        )

        leaking_candidate = _make_leaking_candidate(tmp_root)
        exit_code, rejected = run_phase8e_round(
            config_path=config_path,
            run_dir=tmp_root / "rejected",
            candidate_dir=leaking_candidate,
            require_server_enforcement=hard_supported,
        )
        _assert(exit_code == 1 and rejected["status"] == "reject", "forbidden rationale content must reject candidate")
        _assert("forbidden_content_token" in rejected["verifier"]["issue_codes"], "expected forbidden content issue")

    print(json.dumps({"status": "pass", "test": "phase8e_selftest", "hard_isolation": hard_supported}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
