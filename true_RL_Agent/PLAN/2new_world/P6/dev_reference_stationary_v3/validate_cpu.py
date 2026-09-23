"""Bounded CPU validation of the P6 stationary-v3 development hypothesis."""

from __future__ import annotations

import ast
import hashlib
import json
import sys
import tempfile
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from true_rl_agent.candidates.component_smoke_runtime import run_component_smoke_check
from true_rl_agent.candidates.loader import load_training_components_source_bytes
from true_rl_agent.candidates.schema import CandidateChangeSet, CandidatePackageRef
from true_rl_agent.candidates.verifier import StaticCandidateVerifier
from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.tasks.component_smoke import SmokeTensor
from true_rl_agent.tasks.push_recovery.component_smoke import (
    PushRecoveryComponentSmokeAdapter,
)
from true_rl_agent.tasks.push_recovery.config import PushRecoveryTaskConfig
from true_rl_agent.tasks.push_recovery.plugin import PushRecoveryTaskPlugin
from true_rl_agent.tasks.push_recovery.reference_program import (
    RECOVERY_REFERENCE_SOURCE,
)
from true_rl_agent.tasks.push_recovery.reward_inputs import PushRecoveryV2RewardInputs
from true_rl_agent.training_programs import resolve_reference_program_artifact

REFERENCE_ID = "push_standing_recovery_stationary_reference_v3"
CELLS = ("push.g1_37dof.recovery_v2", "push.go2_12dof.recovery_v2")
HERE = Path(__file__).resolve().parent
SOURCE = HERE / "training_code/components.py"
REPORT = HERE.parent / "dev_reference_stationary_v3_validation.json"


def _hash(payload):
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _functions(source):
    return {node.name: ast.get_source_segment(source, node) for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef)}


def _load(payload, name):
    return load_training_components_source_bytes(payload, module_name=name, source_name=name)


def _candidate(directory, source):
    (directory / "training_code").mkdir()
    (directory / "training_code/components.py").write_bytes(source)
    (directory / "candidate_manifest.yaml").write_text(
        "schema_version: true_rl_agent.candidate.v1\ncandidate_id: p6-stationary-v3-cpu-check\n"
        "entrypoint: training_code.components:build_training_components\n", encoding="utf-8")
    (directory / "rationale.md").write_text("CPU verification wrapper for exact development reference bytes.\n", encoding="utf-8")
    for name in ("operation_diff.json", "diagnostics.json"):
        (directory / name).write_text("{}\n", encoding="utf-8")
    return CandidatePackageRef(candidate_id="p6-stationary-v3-cpu-check", path=str(directory),
        change_set=CandidateChangeSet(frozenset({"reward_callable"})))


def _torch_inputs(adapter, torch):
    fixture = adapter.build_component_smoke_fixture()
    values = {name: getattr(fixture.reward_inputs, name) for name in fixture.reward_inputs.field_names()}
    values = {name: torch.tensor(value.values, dtype=torch.float64, device="cpu")
              if isinstance(value, SmokeTensor) else value for name, value in values.items()}
    return PushRecoveryV2RewardInputs(**values)


def _standing(inputs, planar, angular, torch):
    return replace(inputs,
        roll=torch.zeros(4, dtype=torch.float64), pitch=torch.zeros(4, dtype=torch.float64),
        base_height_relative=torch.ones(4, dtype=torch.float64),
        joint_pos_error=torch.zeros_like(inputs.joint_pos_error),
        action_delta=torch.zeros_like(inputs.action_delta),
        terminated_non_timeout=torch.zeros(4, dtype=torch.bool),
        base_lin_vel_scaled=torch.tensor(planar, dtype=torch.float64),
        base_ang_vel_scaled=torch.tensor(angular, dtype=torch.float64))


def main():
    source = SOURCE.read_bytes()
    old_path = ROOT / RECOVERY_REFERENCE_SOURCE
    old_source, old_mtime = old_path.read_bytes(), old_path.stat().st_mtime_ns
    current, old = _load(source, "p6_stationary_v3"), _load(old_source, "p6_stationary_v2")
    old_functions, new_functions = _functions(old_source.decode()), _functions(source.decode())
    unchanged = ("upright_height", "joint_pose", "action_rate", "fall")
    checks = {"other_four_function_sources_identical": all(old_functions[k] == new_functions[k] for k in unchanged),
        "complete_five_term_program": len(current.reward_terms) == 5,
        "weights_and_clips_unchanged": all((a.weight, a.clip_min, a.clip_max) == (b.weight, b.clip_min, b.clip_max)
                                          for a, b in zip(old.reward_terms, current.reward_terms)),
        "only_stationary_term_id_changed": [t.name for t in current.reward_terms] ==
            ["upright_height", "stationary_v3", "joint_pose", "action_rate", "fall"],
        "reference_id_independent": current.name == REFERENCE_ID and current.name != old.name,
        "no_diagnostics_or_schedules_added": not current.diagnostics and not current.schedules,
        "imports_unchanged": [ast.dump(n) for n in ast.parse(source).body if isinstance(n, (ast.Import, ast.ImportFrom))] ==
            [ast.dump(n) for n in ast.parse(old_source).body if isinstance(n, (ast.Import, ast.ImportFrom))]}
    assert all(checks.values()), checks
    reference = None
    rows = []
    # The deployed pure-function verifier forbids torch imports; this program
    # needs only already-supported Tensor methods. Real Torch is used here only.
    import torch

    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix="p6-stationary-v3-cpu-") as temp:
        candidate = _candidate(Path(temp), source)
        static = StaticCandidateVerifier().verify(candidate)
        if static.status != "pass":
            raise RuntimeError("static verification failed: " + repr(static))
        for cell in CELLS:
            config = PushRecoveryTaskConfig.for_cell(cell)
            plugin = PushRecoveryTaskPlugin(config)
            profile_hash = plugin.task_spec.candidate_capability_profile.profile_hash
            verified = resolve_reference_program_artifact(SOURCE, reference_id=REFERENCE_ID,
                task_id=plugin.task_spec.task_id, authorship_origin="llm_generated",
                upstream_project="true_RL_Agent", upstream_revision="standing_recovery_v2_stationary_only_v3",
                upstream_source_path=SOURCE.relative_to(ROOT).as_posix(), capability_profile_hash=profile_hash)
            if reference is not None:
                assert reference.manifest == verified.manifest
            reference = verified
            adapter = PushRecoveryComponentSmokeAdapter(config)
            smoke = run_component_smoke_check(candidate, adapter)
            assert smoke["status"] == "pass", smoke
            base = _torch_inputs(adapter, torch)
            scales = [0.0, 1.0, 10.0, 20.0]
            batch = _standing(base, [[0.08*x, 0.0, 0.0] for x in scales],
                              [[0.5*x, 0.0, 0.0] for x in scales], torch)
            stationary = current.reward_terms[1].fn(batch)
            v = torch.linalg.vector_norm(batch.base_lin_vel_scaled[:, :2], dim=1) / 0.08
            w = torch.linalg.vector_norm(batch.base_ang_vel_scaled, dim=1) / 0.5
            h = lambda x: torch.where(x <= 1.0, 0.5*x*x, x - 0.5)
            expected = 1.0 - 0.25*(h(v) + h(w))
            float32_batch = replace(batch, **{name: value.float()
                for name in batch.field_names()
                if isinstance((value := getattr(batch, name)), torch.Tensor) and value.is_floating_point()})
            float32_stationary = current.reward_terms[1].fn(float32_batch)
            rotation = _standing(base, [[0.08, 0, 0], [0, 0.08, 0], [0.048, 0.064, 0], [-0.064, 0.048, 0]],
                [[0.5, 0, 0], [0, 0.5, 0], [0.3, 0.4, 0], [-0.4, 0.3, 0]], torch)
            rotation_scores = current.reward_terms[1].fn(rotation)
            stable = _standing(base, [[0, 0, 0]]*4, [[0, 0, 0]]*4, torch)
            fallen = replace(stable, terminated_non_timeout=torch.ones(4, dtype=torch.bool))
            total_stable = sum(term.weight * term.fn(stable) for term in current.reward_terms)
            total_fallen = sum(term.weight * term.fn(fallen) for term in current.reward_terms)
            old_scores = old.reward_terms[1].fn(batch)
            unchanged_outputs = {name: torch.equal(next(t.fn(batch) for t in old.reward_terms if t.name == name),
                next(t.fn(batch) for t in current.reward_terms if t.name == name)) for name in unchanged}
            behavior = {
                "requested_piecewise_formula_equal": torch.allclose(stationary, expected, rtol=0, atol=1e-12),
                "zero_above_gate_above_high_speed": bool(torch.all(stationary[:-1] > stationary[1:])),
                "rotation_same_norm_same_score": torch.allclose(rotation_scores, rotation_scores[:1].expand_as(rotation_scores), rtol=0, atol=1e-12),
                "high_speed_tail_distinguishes_10x_and_20x": bool(stationary[2] > stationary[3]),
                "outputs_finite_correct_shape": bool(torch.isfinite(stationary).all()) and tuple(stationary.shape) == (4,),
                "float32_finite_monotonic_and_matches_formula": bool(torch.isfinite(float32_stationary).all())
                    and bool(torch.all(float32_stationary[:-1] > float32_stationary[1:]))
                    and torch.allclose(float32_stationary, expected.float(), rtol=0, atol=1e-6),
                "fall_total_below_stable_and_nonpositive": bool(torch.all(total_fallen < total_stable) and torch.all(total_fallen <= 0)),
                "other_four_outputs_identical": all(unchanged_outputs.values()),
            }
            assert all(behavior.values()), behavior
            rows.append({"cell": cell, "capability_profile_hash": profile_hash,
                "candidate_capability_ids": sorted(adapter.candidate_capability_ids),
                "joint_and_action_dimension": base.joint_pos.shape[1], "smoke": smoke,
                "torch_behavior_checks": behavior, "speed_multiples_of_gate": scales,
                "stationary_rates": stationary.tolist(), "old_stationary_rates_same_inputs": old_scores.tolist(),
                "float32_stationary_rates": float32_stationary.tolist(),
                "rotation_scores": rotation_scores.tolist(), "full_program_stable_rates": total_stable.tolist(),
                "full_program_fall_upper_bound_case_rates": total_fallen.tolist()})
    checks["original_source_bytes_and_mtime_unchanged"] = old_path.read_bytes() == old_source and old_path.stat().st_mtime_ns == old_mtime
    assert all(checks.values())
    manifest = reference.manifest.to_public_dict()
    (HERE / "reference_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = {"schema_version": "true_rl_agent.p6.dev_reference_stationary_v3_cpu_validation.v1",
        "validated_at_utc": datetime.now(timezone.utc).isoformat(), "status": "pass",
        "scope": "Static verifier, task-owned isolated smoke and Torch CPU behavior only; no training, evaluation, GPU, provider or qualification.",
        "qualification_status": "not_qualified_development_hypothesis",
        "reference_manifest": manifest, "reference_verification_hash": reference.verification_hash,
        "source_path": SOURCE.relative_to(ROOT).as_posix(), "source_sha256": _hash(source),
        "original_source_path": RECOVERY_REFERENCE_SOURCE, "original_source_sha256": _hash(old_source),
        "structural_checks": checks, "static_verifier": {"status": static.status, "reason": static.reason,
            "policy": StaticCandidateVerifier().admission_policy_descriptor().to_public_dict()},
        "container_format_change": {"original": type(old.reward_terms).__name__, "development": type(current.reward_terms).__name__,
            "reason": "Existing static verifier requires reward_terms to be a list; term order and weights are unchanged."},
        "tensor_operations": ["square", "sum(dim=1)", "sqrt", "clamp(max=1)", "slicing", "scalar arithmetic"],
        "torch_version": torch.__version__, "torch_device": "cpu", "cells": rows,
        "instantaneous_fall_rate_bound": {"stable": 4.0, "maximum_when_terminated_non_timeout": -16.0,
            "reason": "upright_height <= 1, stationary_v3 <= 1, joint/action penalties <= 0, fall weight = -20",
            "scope": "Single control-step weighted reward rate only. This does not establish that deliberate falling cannot improve full-episode or discounted return."},
        "effectiveness": "Unmeasured development hypothesis; no claim of root-cause repair, learnability, recovery, or episode-return incentive alignment.",
        "train_jobs": 0, "eval_jobs": 0, "provider_requests": 0,
    }
    report["report_hash"] = canonical_json_hash(report)
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "reference_id": REFERENCE_ID,
        "source_sha256": report["source_sha256"], "report": str(REPORT)}, sort_keys=True))


if __name__ == "__main__":
    main()
