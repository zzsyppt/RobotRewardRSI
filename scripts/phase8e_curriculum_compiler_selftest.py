#!/usr/bin/env python3
"""Selftests for the runner-owned Phase 8E curriculum compiler."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from humanoid_agent.cold_start_lab.curriculum_compiler import (  # noqa: E402
    CurriculumCompileError,
    compile_v1_curriculum_for_training,
    enrich_curriculum_summary_with_immutable_context,
    project_curriculum_lineage_bundle,
)
from humanoid_agent.cold_start_lab.curriculum_runtime import (  # noqa: E402
    Phase8ECompiledCurriculumRuntime,
    Phase8EVecEnvCurriculumWrapper,
)
from humanoid_agent.code_sandbox.loader import candidate_package_fingerprint, load_candidate_manifest  # noqa: E402


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


_BASE_COMPONENT_SOURCE = """from humanoid_agent.code_sandbox.components import TrainingComponents


def build_training_components(task_spec, base_env_cfg, base_agent_cfg):
    _ = (task_spec, base_env_cfg, base_agent_cfg)
    return TrainingComponents(name="test_candidate")
"""


def _write_candidate_shell(candidate: Path, *, candidate_id: str, component_source: str) -> None:
    (candidate / "training_code").mkdir(parents=True, exist_ok=True)
    _write_json(
        candidate / "candidate_manifest.yaml",
        {
            "schema_version": "humanoid_agent.cold_start.candidate.v1",
            "candidate_id": candidate_id,
            "description": "test curriculum candidate",
            "author": "LLM",
            "llm_authored": True,
            "entrypoint": "training_code.components:build_training_components",
            "phase": "8E",
            "expected_behavior": "test candidate",
        },
    )
    (candidate / "training_code" / "components.py").write_text(component_source, encoding="utf-8")


def _candidate_curriculum() -> tuple[dict[str, Any], dict[str, Any]]:
    operation_diff = {
        "operations": [],
        "curriculum_design": {
            "stage_list": [
                {
                    "name": "stand_balance_warmup",
                    "progress_range": [0.0, 0.30],
                    "capabilities": ["reward_weight_scale"],
                    "rationale": "stabilize standing before perturbation.",
                },
                {
                    "name": "mild_push_recovery",
                    "progress_range": [0.30, 0.80],
                    "capabilities": ["external_push_curriculum"],
                    "rationale": "introduce recoverable external pushes.",
                },
                {
                    "name": "target_consolidation",
                    "progress_range": [0.80, 1.0],
                    "capabilities": ["external_push_curriculum", "train_only_dr_scale"],
                    "rationale": "finish on the target push distribution with train-only robustness variation.",
                },
            ],
            "final_consolidation": {
                "fraction": 0.2,
                "distribution": "runner_owned_target_distribution",
                "includes_target_push": True,
                "includes_target_robustness_variation": True,
            },
            "anti_tuning_rationale": "The design changes the training distribution by stage instead of only changing scalar weights.",
            "improvement_attribution_intent": "Improvement should come from staged push recovery plus final target consolidation.",
        },
    }
    diagnostics = {
        "items": [],
        "curriculum_hypothesis": {
            "primary_failure_mode": "public push recovery is weaker than no-push standing.",
            "evidence": ["public push probe under target distribution"],
            "anti_tuning_rationale": "Reward terms alone do not expose recovery perturbations during training.",
            "improvement_attribution_intent": "Curriculum stage distribution is expected to contribute to improvement.",
        },
    }
    return operation_diff, diagnostics


def _phase1_config() -> dict[str, Any]:
    return {
        "schema_version": "humanoid_agent.phase1.push_recovery.config.v1",
        "environment": {
            "episode_length_s": 4.0,
            "terrain_rows": 2,
            "terrain_cols": 2,
            "disable_curriculum": True,
            "disable_observation_corruption": True,
        },
        "standing_command": {
            "resampling_time_range_s": [10.0, 10.0],
            "lin_vel_x_mps": [0.0, 0.0],
            "lin_vel_y_mps": [0.0, 0.0],
            "ang_vel_z_radps": [0.0, 0.0],
            "rel_standing_envs": 1.0,
            "rel_heading_envs": 0.0,
            "heading_command": False,
        },
        "push": {
            "interval_range_s": [1.0, 1.0],
            "velocity_range_mps": {
                "x": [-0.6, 0.6],
                "y": [-0.6, 0.6],
            },
        },
        "profiles": {
            "phase2_train": {
                "push": {
                    "interval_range_s": [1.0, 1.0],
                    "velocity_range_mps": {
                        "x": [0.0, 0.0],
                        "y": [0.0, 0.0],
                    },
                },
            },
            "phase2_public": {
                "push": {
                    "interval_range_s": [1.0, 1.0],
                    "velocity_range_mps": {
                        "x": [-0.25, 0.25],
                        "y": [-0.25, 0.25],
                    },
                },
            },
        },
    }


def _phase2_config() -> dict[str, Any]:
    return {
        "schema_version": "humanoid_agent.phase2.baseline.v1",
        "training": {"profile": "phase2_train"},
        "evaluation": {
            "public": {"profile": "phase2_public", "seed": 1001},
            "iid_hidden": {"profile": "phase2_iid_hidden", "seed": 2001},
        },
        "curriculum": {
            "disable_terrain_levels": True,
            "disable_velocity_command_levels": True,
        },
        "reward_weights": {},
        "reward_params": {},
        "ppo": {},
    }


def _make_candidate(
    root: Path,
    *,
    valid: bool = True,
    dirname: str = "candidate",
    candidate_id: str = "phase8e_test_candidate",
    component_source: str = _BASE_COMPONENT_SOURCE,
) -> Path:
    candidate = root / dirname
    candidate.mkdir(parents=True)
    _write_candidate_shell(candidate, candidate_id=candidate_id, component_source=component_source)
    operation_diff, diagnostics = _candidate_curriculum()
    if not valid:
        operation_diff["curriculum_design"].pop("final_consolidation")
    _write_json(candidate / "operation_diff.json", operation_diff)
    _write_json(candidate / "diagnostics.json", diagnostics)
    return candidate


def test_public_lineage_projection_requires_attribution_schema() -> None:
    valid = {
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
        }
    }
    projected = project_curriculum_lineage_bundle(valid)
    _assert(projected == valid["public_summary"], f"valid public lineage should project: {projected}")
    _assert(project_curriculum_lineage_bundle({"public_summary": {}}) is None, "empty public lineage summary must not project")

    missing_attribution = {"public_summary": {key: value for key, value in valid["public_summary"].items() if key != "attribution"}}
    _assert(
        project_curriculum_lineage_bundle(missing_attribution) is None,
        "public lineage missing attribution must not project",
    )

    incomplete_attribution = json.loads(json.dumps(valid))
    incomplete_attribution["public_summary"]["attribution"].pop("pure_knob_candidate")
    _assert(
        project_curriculum_lineage_bundle(incomplete_attribution) is None,
        "public lineage with incomplete attribution must not project",
    )

    missing_required_field = json.loads(json.dumps(valid))
    missing_required_field["public_summary"].pop("capabilities")
    _assert(
        project_curriculum_lineage_bundle(missing_required_field) is None,
        "public lineage missing required top-level fields must not project",
    )


def test_compile_writes_train_only_configs_and_summary() -> None:
    with tempfile.TemporaryDirectory(prefix="phase8e-curriculum-compiler-") as tmp:
        root = Path(tmp)
        candidate = _make_candidate(root)
        phase1 = root / "phase1.json"
        phase2 = root / "phase2.json"
        _write_json(phase1, _phase1_config())
        _write_json(phase2, _phase2_config())

        summary = compile_v1_curriculum_for_training(
            candidate_dir=candidate,
            phase1_config_path=phase1,
            phase2_config_path=phase2,
            output_dir=root / "compiled",
            train_profile="phase2_train",
            max_iterations=10,
            steps_per_iteration=24,
        )

        _assert(summary["status"] == "pass", f"compiler should pass: {summary}")
        compiled_phase1 = json.loads(Path(summary["phase1_config_path"]).read_text(encoding="utf-8"))
        compiled_phase2 = json.loads(Path(summary["phase2_config_path"]).read_text(encoding="utf-8"))
        curriculum = compiled_phase1.get("runner_curriculum")
        _assert(isinstance(curriculum, dict), "compiled phase1 config must include runner_curriculum")
        _assert(curriculum["execution_mode"] == "env_step_progress", f"unexpected execution mode: {curriculum}")
        _assert(len(curriculum["stages"]) == 3, f"stage structure should survive compilation: {curriculum}")
        final_push = curriculum["stages"][-1]["push"]["velocity_range_mps"]
        _assert(final_push["x"] == [-0.25, 0.25], f"final stage must return to public target push x: {final_push}")
        _assert(final_push["y"] == [-0.25, 0.25], f"final stage must return to public target push y: {final_push}")
        _assert(
            compiled_phase1["environment"]["disable_observation_corruption"] is False,
            "train-only DR capability should enable train-only observation robustness",
        )
        _assert(
            compiled_phase2["evaluation"] == _phase2_config()["evaluation"],
            "compiler must not edit eval profiles or seeds",
        )
        _assert(Path(summary["summary_file"]).is_file(), "compiler must write a durable summary artifact")


def test_final_stage_capability_augmentation_is_reported_in_clamps() -> None:
    with tempfile.TemporaryDirectory(prefix="phase8e-curriculum-capability-clamp-") as tmp:
        root = Path(tmp)
        candidate = _make_candidate(root)
        operation_diff = json.loads((candidate / "operation_diff.json").read_text(encoding="utf-8"))
        final_stage = operation_diff["curriculum_design"]["stage_list"][-1]
        final_stage["capabilities"] = ["reward_only_push_signal"]
        final_stage["rationale"] = "Final stage relies on push signal rewards; runner must still consolidate target train distribution."
        _write_json(candidate / "operation_diff.json", operation_diff)
        phase1 = root / "phase1.json"
        phase2 = root / "phase2.json"
        _write_json(phase1, _phase1_config())
        _write_json(phase2, _phase2_config())

        summary = compile_v1_curriculum_for_training(
            candidate_dir=candidate,
            phase1_config_path=phase1,
            phase2_config_path=phase2,
            output_dir=root / "compiled",
            train_profile="phase2_train",
            max_iterations=10,
            steps_per_iteration=24,
        )

        final_clamp = summary["clamps"][-1]
        _assert(
            final_clamp.get("raw_capabilities") == ["reward_only_push_signal"],
            f"clamp summary must expose raw final-stage capabilities: {final_clamp}",
        )
        _assert(
            final_clamp.get("effective_capabilities")
            == ["external_push_curriculum", "reward_only_push_signal", "train_only_dr_scale"],
            f"clamp summary must expose effective final-stage capabilities: {final_clamp}",
        )
        _assert(
            final_clamp.get("added_capabilities") == ["external_push_curriculum", "train_only_dr_scale"],
            f"clamp summary must record compiler-added final-stage capabilities: {final_clamp}",
        )
        _assert(
            final_clamp.get("forced_final_consolidation_capabilities")
            == ["external_push_curriculum", "train_only_dr_scale"],
            f"clamp summary must name forced final-stage capabilities: {final_clamp}",
        )
        _assert(
            final_clamp.get("capability_augmentation_reason") == "final_consolidation_forces_target_push_and_dr",
            f"clamp summary must explain why capabilities were added: {final_clamp}",
        )

    with tempfile.TemporaryDirectory(prefix="phase8e-curriculum-partial-capability-clamp-") as tmp:
        root = Path(tmp)
        candidate = _make_candidate(root)
        operation_diff = json.loads((candidate / "operation_diff.json").read_text(encoding="utf-8"))
        operation_diff["curriculum_design"]["stage_list"][-1]["capabilities"] = ["external_push_curriculum"]
        _write_json(candidate / "operation_diff.json", operation_diff)
        phase1 = root / "phase1.json"
        phase2 = root / "phase2.json"
        _write_json(phase1, _phase1_config())
        _write_json(phase2, _phase2_config())

        summary = compile_v1_curriculum_for_training(
            candidate_dir=candidate,
            phase1_config_path=phase1,
            phase2_config_path=phase2,
            output_dir=root / "compiled",
            train_profile="phase2_train",
            max_iterations=10,
            steps_per_iteration=24,
        )
        final_clamp = summary["clamps"][-1]
        _assert(
            final_clamp.get("added_capabilities") == ["train_only_dr_scale"],
            f"clamp summary must record one-sided final-stage capability augmentation: {final_clamp}",
        )
        _assert(
            final_clamp.get("forced_final_consolidation_capabilities") == ["train_only_dr_scale"],
            f"clamp summary must name one-sided forced final-stage capability: {final_clamp}",
        )


def test_runtime_wrapper_applies_stage_push_by_progress() -> None:
    with tempfile.TemporaryDirectory(prefix="phase8e-curriculum-runtime-") as tmp:
        root = Path(tmp)
        candidate = _make_candidate(root)
        phase1 = root / "phase1.json"
        phase2 = root / "phase2.json"
        _write_json(phase1, _phase1_config())
        _write_json(phase2, _phase2_config())
        summary = compile_v1_curriculum_for_training(
            candidate_dir=candidate,
            phase1_config_path=phase1,
            phase2_config_path=phase2,
            output_dir=root / "compiled",
            train_profile="phase2_train",
            max_iterations=10,
            steps_per_iteration=1,
        )
        compiled_phase1 = json.loads(Path(summary["phase1_config_path"]).read_text(encoding="utf-8"))
        runtime = Phase8ECompiledCurriculumRuntime.from_training_config(
            compiled_phase1,
            planned_training_steps=10,
        )

        push_term = SimpleNamespace(
            interval_range_s=(1.0, 1.0),
            params={"velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)}},
        )
        base_env = SimpleNamespace(cfg=SimpleNamespace(events=SimpleNamespace(push_robot=push_term)))

        class FakeVecEnv:
            num_envs = 1
            device = "cpu"

            @property
            def unwrapped(self) -> Any:
                return base_env

            def step(self, actions: Any) -> tuple[Any, ...]:
                return "obs", [0.0], [False], {}

        wrapped = Phase8EVecEnvCurriculumWrapper(FakeVecEnv(), runtime)
        wrapped.step(None)
        _assert(push_term.params["velocity_range"]["x"] == (0.0, 0.0), "warmup stage should keep push disabled")
        for _ in range(9):
            wrapped.step(None)
        _assert(
            push_term.params["velocity_range"]["x"] == (-0.25, 0.25),
            f"final stage should apply target push: {push_term.params}",
        )
        runtime_summary = runtime.runtime_summary()
        _assert(
            runtime_summary["last_applied_stage"]["name"] == "target_consolidation",
            f"runtime should report the final active stage: {runtime_summary}",
        )


def test_degenerate_curriculum_rejected_before_training() -> None:
    with tempfile.TemporaryDirectory(prefix="phase8e-curriculum-reject-") as tmp:
        root = Path(tmp)
        candidate = _make_candidate(root, valid=False)
        phase1 = root / "phase1.json"
        phase2 = root / "phase2.json"
        _write_json(phase1, _phase1_config())
        _write_json(phase2, _phase2_config())
        try:
            compile_v1_curriculum_for_training(
                candidate_dir=candidate,
                phase1_config_path=phase1,
                phase2_config_path=phase2,
                output_dir=root / "compiled",
                train_profile="phase2_train",
                max_iterations=10,
                steps_per_iteration=24,
            )
        except CurriculumCompileError:
            return
        raise AssertionError("missing final_consolidation must reject before training")


def test_unregistered_capability_rejected_before_compilation() -> None:
    with tempfile.TemporaryDirectory(prefix="phase8e-curriculum-unknown-capability-") as tmp:
        root = Path(tmp)
        candidate = _make_candidate(root)
        operation_diff = json.loads((candidate / "operation_diff.json").read_text(encoding="utf-8"))
        operation_diff["curriculum_design"]["stage_list"][1]["capabilities"] = [
            "external_push_curriculum",
            "invented_eval_seed_control",
        ]
        _write_json(candidate / "operation_diff.json", operation_diff)
        phase1 = root / "phase1.json"
        phase2 = root / "phase2.json"
        _write_json(phase1, _phase1_config())
        _write_json(phase2, _phase2_config())
        try:
            compile_v1_curriculum_for_training(
                candidate_dir=candidate,
                phase1_config_path=phase1,
                phase2_config_path=phase2,
                output_dir=root / "compiled",
                train_profile="phase2_train",
                max_iterations=10,
                steps_per_iteration=24,
            )
        except CurriculumCompileError as exc:
            _assert("unregistered" in str(exc) and "capabilities" in str(exc), f"rejection must name capability issue: {exc}")
            return
        raise AssertionError("unregistered capability entered curriculum compiler output")


def test_relative_lineage_marks_component_change_without_curriculum_delta() -> None:
    with tempfile.TemporaryDirectory(prefix="phase8e-curriculum-relative-lineage-") as tmp:
        root = Path(tmp)
        current_best = _make_candidate(root, dirname="current_best", candidate_id="phase8e_current_best")
        candidate = _make_candidate(
            root,
            dirname="candidate",
            candidate_id="phase8e_reward_repair",
            component_source=_BASE_COMPONENT_SOURCE + "\n# reward-side repair changes executable candidate code\n",
        )
        phase1 = root / "phase1.json"
        phase2 = root / "phase2.json"
        _write_json(phase1, _phase1_config())
        _write_json(phase2, _phase2_config())

        summary = compile_v1_curriculum_for_training(
            candidate_dir=candidate,
            phase1_config_path=phase1,
            phase2_config_path=phase2,
            output_dir=root / "compiled",
            train_profile="phase2_train",
            max_iterations=10,
            steps_per_iteration=24,
        )
        current_fingerprint = candidate_package_fingerprint(load_candidate_manifest(current_best))
        current_operation_diff = json.loads((current_best / "operation_diff.json").read_text(encoding="utf-8"))
        current_design = current_operation_diff["curriculum_design"]
        current_design["stages"] = current_design.pop("stage_list")
        current_final = current_design["final_consolidation"]
        current_final["min_fraction"] = current_final.pop("fraction")
        current_final["target_push_included"] = current_final.pop("includes_target_push")
        current_final["target_robustness_included"] = current_final.pop("includes_target_robustness_variation")
        enriched = enrich_curriculum_summary_with_immutable_context(
            summary,
            prompt_payload={
                "current_recipe": {
                    "fingerprint": current_fingerprint,
                    "operation_diff": current_operation_diff,
                },
                "public_curriculum_evidence_bundle": {},
                "required_candidate_package": {},
                "candidate_allowed_controls": [],
                "curriculum_design_contract": {},
            },
            prompt_payload_path=None,
            artifacts={},
        )
        lineage = json.loads(Path(enriched["curriculum_lineage_file"]).read_text(encoding="utf-8"))
        attribution = lineage["public_summary"]["attribution"]
        _assert(attribution["reward_code_changed"] is True, f"component code delta must be attributed: {attribution}")
        _assert(attribution["curriculum_changed"] is False, f"unchanged curriculum design must not be credited: {attribution}")
        _assert(
            attribution["gain_attributable_to"] == "reward_code",
            f"reward repair must not be reported as curriculum gain: {attribution}",
        )


def main() -> int:
    test_public_lineage_projection_requires_attribution_schema()
    test_compile_writes_train_only_configs_and_summary()
    test_final_stage_capability_augmentation_is_reported_in_clamps()
    test_runtime_wrapper_applies_stage_push_by_progress()
    test_degenerate_curriculum_rejected_before_training()
    test_unregistered_capability_rejected_before_compilation()
    test_relative_lineage_marks_component_change_without_curriculum_delta()
    print(json.dumps({"status": "pass", "test": "phase8e_curriculum_compiler_selftest"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
