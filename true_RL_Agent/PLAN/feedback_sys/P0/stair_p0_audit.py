"""Read-only Stair P0 audit; writes only stair_* evidence in this directory.

No simulator, training, model provider, or remote runner is invoked.
Run: .venv/Scripts/python.exe PLAN/feedback_sys/P0/stair_p0_audit.py
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from true_rl_agent.tasks.stair_climbing.config import StairClimbingTaskConfig
from true_rl_agent.tasks.stair_climbing.plugin import StairClimbingTaskPlugin
from true_rl_agent.tasks.stair_climbing.robot_binding import resolve_stair_cell_manifest
from true_rl_agent.tasks.stair_climbing.row_catalog import stair_robot_row
from true_rl_agent.tasks.stair_climbing.runtime_projection import stair_runtime_projection_from_manifest
from true_rl_agent.tasks.stair_climbing.unified_scoring import validate_unified_stair_summary, build_unified_stair_prompt_feedback


def read_json(rel):
    return json.loads((ROOT / rel).read_text(encoding="utf-8-sig"))


def sha(rel):
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def thaw(value):
    if isinstance(value, Mapping):
        return {k: thaw(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [thaw(v) for v in value]
    return value


def write_json(name, value):
    (OUT / name).write_text(json.dumps(thaw(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


rows = {}
configs = {}
for robot, variant in (("go2", "stair.go2_12dof.unified_v3"), ("g1", "stair.g1_37dof.unified_v3")):
    row = stair_robot_row(variant)
    config = StairClimbingTaskConfig.defaults(ROOT / row.protocol_relative_path, cell_variant_id=variant)
    configs[robot] = config
    manifest = resolve_stair_cell_manifest(cell_variant_id=variant, config=config, plugin=StairClimbingTaskPlugin(config))
    projection = stair_runtime_projection_from_manifest(manifest)
    rows[robot] = {
        "kind": "current_source_cpu_resolution_not_simulator_run",
        "cell_variant_id": variant,
        "protocol_file": row.protocol_relative_path,
        "binding_file": row.binding_relative_path,
        "resolved_manifest": manifest.to_dict(),
        "runtime_projection": projection.to_summary_dict(),
        "reward_input_schema": projection.reward_input_schema_vnext(),
        "evaluation_suite_hash": config.ordered_evaluation_suite.suite_hash,
        "episode_outcomes": config.evaluation_semantics.outcomes,
    }
write_json("stair_current_identity.json", rows)

sources = {
    "go2": "PLAN/2new_world/P3/results/cohort_alignment_20260909/go2_12dof_qualification/evaluation/public_dev/summary.json",
    "g1": "PLAN/2new_world/P3/results/cohort_alignment_20260909/g1_37dof_qualification/evaluation/public_dev/summary.json",
}
evidence = []
table = []
projection_checks = []
for robot, rel in sources.items():
    document = read_json(rel)
    summary = document["task_evidence"]["summary"]
    ordered = summary["ordered_summary"]
    validation = {"status": "pass"}
    try:
        validate_unified_stair_summary(suite=configs[robot].ordered_evaluation_suite,
                                     semantics=configs[robot].evaluation_semantics,
                                     split="public_dev", summary=summary)
    except (ValueError, TypeError, KeyError) as exc:
        validation = {"status": "mismatch", "error": str(exc)}
    if validation["status"] == "pass":
        feedback = build_unified_stair_prompt_feedback(suite=configs[robot].ordered_evaluation_suite,
                        semantics=configs[robot].evaluation_semantics, summary=summary)
        projection_checks.append({"robot": robot, "source": rel,
            "kind": "cpu_current_task_feedback_projection_of_historical_real_public_dev",
            "scope": "task_feedback_only; no_provider_request; no_session_genesis_mask_assertion",
            "feedback": feedback})
    counts = Counter({reason: 0 for reason in ordered["primary_outcome_counts"]})
    flags = Counter()
    all_facts = []
    by_cell = {}
    for seed in ordered["seed_summaries"]:
        for cell, facts in seed["episode_facts_by_cell"].items():
            by_cell.setdefault(cell, []).extend(facts)
            all_facts.extend(facts)
            counts.update(fact["primary_reason"] for fact in facts)
            for fact in facts:
                flags.update(fact["diagnostic_flags"])
    assert len(all_facts) == ordered["total_episode_count"]
    assert all(counts[k] == v for k, v in ordered["primary_outcome_counts"].items())
    for cell, facts in [("__all__", all_facts), *by_cell.items()]:
        for reason in sorted(counts):
            group = [fact for fact in facts if fact["primary_reason"] == reason]
            for flag in ("insufficient_clearance_detected", "slip_detected"):
                numerator = sum(flag in fact["diagnostic_flags"] for fact in group)
                table.append({"robot": robot, "source_sha256": sha(rel), "cell_id": cell,
                              "primary_reason": reason, "diagnostic_flag": flag,
                              "flag_count": numerator, "episode_count": len(group),
                              "rate": numerator / len(group) if group else "missing_no_episodes"})
    evidence.append({
        "robot": robot, "source": rel, "sha256": sha(rel),
        "artifact_kind": "historical_real_simulator_public_dev_result",
        "new_simulator_run_in_p0": False,
        "status": document["status"], "errors": document["errors"],
        "resolved_cell_identity": document["resolved_cell_identity"],
        "resolved_cell_manifest_hash": document["resolved_cell_manifest_hash"],
        "current_resolved_manifest_hash_match": (
            document["resolved_cell_manifest_hash"] == rows[robot]["resolved_manifest"]["manifest_hash"]),
        "evaluation_binding": document["evaluation_binding"],
        "checkpoint_loading": document["checkpoint_loading"],
        "task_evidence_hash": document["task_evidence_hash"],
        "summary_schema": summary["schema_version"],
        "ordered_summary_schema": ordered["schema_version"],
        "suite_hash": summary["ordered_suite_hash"],
        "current_source_summary_validation": validation,
        "fact_count_closure": "pass", "total_episode_count": len(all_facts),
        "seed_summary_count": len(ordered["seed_summaries"]), "cell_count": len(by_cell),
        "facts_per_cell": {cell: len(facts) for cell, facts in by_cell.items()},
        "episode_fact_fields": sorted(all_facts[0]),
        "primary_counts": dict(counts), "diagnostic_counts": dict(flags),
        "cooccurrence_scope": "whole_episode_only_not_preterminal_window_or_causation",
        "limitations": ["no_persistent_step_history_in_this_summary", "no_per_body_event_onset",
                        "no_failure_crossing_anchor", "single_checkpoint_per_robot_not_independent_training_replicates"],
    })
write_json("stair_real_public_dev_audit.json", evidence)
write_json("stair_current_projection_audit.json", projection_checks)
qualification = []
for robot, directory in (("go2", "go2_12dof_qualification"), ("g1", "g1_37dof_qualification")):
    rel = "PLAN/2new_world/P3/results/cohort_alignment_20260909/" + directory + "/result.json"
    q = read_json(rel)
    qualification.append({"robot": robot, "source": rel, "source_sha256": sha(rel),
        "status": q["status"], "reference_gate_status": q["reference_gate"]["status"],
        "reference_gate_evidence_hash": q["reference_gate"]["evidence_hash"],
        "checkpoint_checksum": q["checkpoint"], "training_summary_source": q["training_summary"],
        "scope": "existing qualification identity/status only; no accept/sealed stratified values exported"})
go2_train_rel = "PLAN/2new_world/P2/results/audit3_go2_full_train/runner/audit3_go2_full_train/train/summary.json"
go2_train = read_json(go2_train_rel)
qualification[0].update({"local_training_summary": go2_train_rel,
                         "local_training_summary_sha256": sha(go2_train_rel),
                         "training_status": go2_train["status"], "checkpoint": go2_train["checkpoint"],
                         "checkpoint_remote_existence_checked_by_this_script": False})
assert qualification[0]["checkpoint_checksum"] == go2_train["checkpoint"]["checksum"]
write_json("stair_reference_identity_audit.json", qualification)
with (OUT / "stair_episode_cooccurrence.csv").open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(table[0]))
    writer.writeheader()
    writer.writerows(table)

prompt_rel = "PLAN/2new_world/P6/shared_history_c6_01/generation_13.json"
prompt = read_json(prompt_rel)
prompt_keys = []
def walk(value, path=""):
    if isinstance(value, dict):
        for key, child in value.items():
            full = path + "." + key if path else key
            prompt_keys.append(full)
            walk(child, full)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            walk(child, f"{path}[{index}]")
walk(prompt)
write_json("stair_historical_prompt_shape.json", {
    "source": prompt_rel, "source_sha256": sha(prompt_rel),
    "kind": "historical_unified_v2_prompt_reconstruction_not_new_provider_call",
    "source_run_receipt": "PLAN/2new_world/P6/shared_history_c6_01/result.json",
    "root_keys": list(prompt),
    "feedback_entry_count": len(prompt["public_feedback_summary"]["feedback_entries"]),
    "feedback_entry_keys": [list(entry) for entry in prompt["public_feedback_summary"]["feedback_entries"]],
    "feedback_keys": [list(entry.get("feedback", {})) for entry in prompt["public_feedback_summary"]["feedback_entries"]],
    "runtime_feedback_summary": prompt["runtime_feedback_summary"],
    "exact_key_presence": {key: any(path.rsplit(".", 1)[-1] == key for path in prompt_keys)
                           for key in ["episode_facts_by_cell", "support_progress_m", "support_surface_clearances_m",
                                       "episode_step", "termination_terms", "primary_failure_breakdown", "diagnostic_flag_rates"]},
})

files = [
    "src/true_rl_agent/tasks/stair_climbing/isaaclab_runtime_env.py",
    "src/true_rl_agent/tasks/stair_climbing/evaluation_runtime.py",
    "src/true_rl_agent/tasks/stair_climbing/transition_decoder.py",
    "src/true_rl_agent/tasks/stair_climbing/episode_evidence.py",
    "src/true_rl_agent/tasks/stair_climbing/terrain_evidence.py",
    "src/true_rl_agent/tasks/stair_climbing/ordered_summary.py",
    "src/true_rl_agent/tasks/stair_climbing/unified_scoring.py",
    "src/true_rl_agent/tasks/stair_climbing/public_projection.py",
    "src/true_rl_agent/tasks/stair_climbing/prompt_adapter.py",
    "src/true_rl_agent/tasks/stair_climbing/phased_adapter.py",
    "src/true_rl_agent/tasks/stair_climbing/robot_binding.py",
    "src/true_rl_agent/tasks/stair_climbing/runtime_projection.py",
    "src/true_rl_agent/tasks/stair_climbing/isaaclab_reward_runtime.py",
    "src/true_rl_agent/tasks/stair_climbing/training_diagnostics.py",
    "src/true_rl_agent/tasks/stair_climbing/reward_program.py",
    "src/true_rl_agent/tasks/stair_climbing/termination_capture.py",
    "src/true_rl_agent/tasks/stair_climbing/row_catalog.py",
    "scripts/isaaclab_rsl_rl_train.py", "scripts/isaaclab_rsl_rl_eval.py",
    "configs/tasks/stair_unified_protocol_v3.json",
    "configs/tasks/stair_go2_12dof_unified_protocol_v3.json",
    "configs/tasks/stair_g1_37dof_unified_binding_v3.json",
    "configs/tasks/stair_go2_12dof_unified_binding_v3.json",
    "configs/tasks/stair_unified_evaluation_suite_v2.json",
    "configs/tasks/stair_go2_12dof_unified_evaluation_suite_v2.json",
    "configs/tasks/stair_unified_acceptance_protocol_v7.json",
    "configs/tasks/stair_vnext_reward_authority_v1.json",
    "configs/tasks/stair_go2_12dof_vnext_reward_authority_v1.json",
    "configs/robots/unitree_g1_37dof_isaaclab_v2.json",
    "configs/robots/unitree_go2_12dof_isaaclab_v1.json",
    "tests/unit/test_stair_episode_evidence.py", "tests/unit/test_stair_transition_materialization.py",
    "tests/unit/test_stair_termination_capture.py", "tests/unit/test_stair_morphology_projection.py",
    "tests/unit/test_stair_completion_scoring.py", "tests/unit/test_stair_ordered_evaluation.py",
    "tests/unit/test_stair_public_projection.py", "tests/unit/test_stair_phase4_contract.py",
]
write_json("stair_source_snapshot.json", {
    "captured_utc": datetime.now(timezone.utc).isoformat(),
    "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
    "python": sys.executable,
    "files": {rel: sha(rel) for rel in files},
    "meaning": "actual_working_tree_hashes_not_HEAD_only; CPU resolution and historical artifact audit, not new simulator validation",
})
log = (OUT / "stair_cpu_regression.log").read_text(encoding="utf-8-sig")
write_json("stair_cpu_regression_receipt.json", {
    "kind": "existing_cpu_unit_regression_not_simulator_or_experiment",
    "python": sys.executable, "environment": {"CUDA_VISIBLE_DEVICES": ""},
    "command": ".venv/Scripts/python.exe -m unittest -v " + " ".join([
        "tests.unit.test_stair_episode_evidence", "tests.unit.test_stair_transition_materialization",
        "tests.unit.test_stair_termination_capture", "tests.unit.test_stair_morphology_projection",
        "tests.unit.test_stair_completion_scoring", "tests.unit.test_stair_ordered_evaluation",
        "tests.unit.test_stair_public_projection.StairPublicProjectionTests.test_prompt_projects_latest_validated_complete_reward_runtime",
        "tests.unit.test_stair_public_projection.StairPublicProjectionTests.test_prompt_feedback_exposes_arrival_without_stable_success",
        "tests.unit.test_stair_public_projection.StairPublicProjectionTests.test_runtime_gate_requires_complete_execution_evidence",
        "tests.unit.test_stair_phase4_contract.StairPhase4ContractTests.test_provider_view_uses_public_dev_without_acceptance_result_leak",
        "tests.unit.test_stair_phase4_contract.StairPhase4ContractTests.test_exploration_compares_only_public_dev_to_public_dev_parent",
        "tests.unit.test_stair_phase4_contract.StairPhase4ContractTests.test_measured_dev_without_selection_has_one_exploration_and_prompt_comparison"]),
    "log": "stair_cpu_regression.log", "log_sha256": hashlib.sha256((OUT / "stair_cpu_regression.log").read_bytes()).hexdigest(),
    "terminal_summary": log[log.rfind("----------------------------------------------------------------------"):].strip(),
    "passed_test_lines": sum(line.endswith(" ... ok") for line in log.splitlines()),
    "note": "Initial audit-only JSON exporter hit MappingProxy serialization; fixed locally with recursive thaw. No production defect or test failure; no business source changed.",
})
print(json.dumps({"rows": {k: {"variant": v["cell_variant_id"], "suite": v["evaluation_suite_hash"],
                              "fields": len(v["reward_input_schema"]["fields"])} for k, v in rows.items()},
                  "real_evidence": [{"robot": e["robot"], "n": e["total_episode_count"],
                                     "validation": e["current_source_summary_validation"],
                                     "primary_counts": e["primary_counts"]} for e in evidence],
                  "written_cooccurrence_rows": len(table)}, ensure_ascii=False, indent=2))
