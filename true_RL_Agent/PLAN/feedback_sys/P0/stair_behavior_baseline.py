"""Recompute a descriptive two-policy Stair v3 baseline from saved public-dev facts.

Run from the repository: .venv/Scripts/python.exe PLAN/feedback_sys/P0/stair_behavior_baseline.py
Reads local sources and receipts; writes only stair_behavior_baseline.json. No replay or API.
"""
from __future__ import annotations

import hashlib
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).with_suffix(".json")
sys.path.insert(0, str(ROOT / "src"))

from true_rl_agent.tasks.stair_climbing.config import StairClimbingTaskConfig
from true_rl_agent.tasks.stair_climbing.plugin import StairClimbingTaskPlugin
from true_rl_agent.tasks.stair_climbing.robot_binding import resolve_stair_cell_manifest
from true_rl_agent.tasks.stair_climbing.unified_scoring import validate_unified_stair_summary

FORMAL = "PLAN/feedback_sys/P0/inventory_remote_raw/outputs/formal_go2_stair_15round_20260901_c6ab8ba"
REPLAY = "PLAN/feedback_sys/audit4_replay/gen008a_v3"
SOURCES = {
    "p3_reference": "PLAN/2new_world/P3/results/cohort_alignment_20260909/go2_12dof_qualification/evaluation/public_dev/summary.json",
    "formal_gen008a_v3_replay": f"{REPLAY}/public_dev_summary.json",
}
TRAIN = {
    "p3_reference": "PLAN/2new_world/P2/results/audit3_go2_full_train/runner/audit3_go2_full_train/train/summary.json",
    "formal_gen008a_v3_replay": f"{FORMAL}/runner/formal-go2-stair-15round-20260901-c6ab8ba-generation-008-lane-a/train/summary.json",
}


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8-sig"))


def sha(path):
    return "sha256:" + hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def span(values):
    return {"n": len(values), "min": min(values), "median": statistics.median(values),
            "max": max(values)} if values else {"n": 0, "min": None, "median": None, "max": None}


def summarize(facts, flags):
    primary = Counter(f["primary_reason"] for f in facts)
    groups = {}
    for reason in sorted(primary):
        selected = [f for f in facts if f["primary_reason"] == reason]
        groups[reason] = {
            "n": len(selected),
            "diagnostic_flag_counts": {flag: sum(flag in f["diagnostic_flags"] for f in selected) for flag in flags},
            "diagnostic_flag_rates": {flag: sum(flag in f["diagnostic_flags"] for f in selected) / len(selected) for flag in flags},
            "outcome_closure_control_steps": span([f["episode_steps"] for f in selected]),
            "maximum_normalized_progress": span([f["progress_fraction"] for f in selected]),
        }
    return {"episodes": len(facts), "primary_counts": dict(sorted(primary.items())), "by_primary": groups}


def analyze(label, path, config, current_manifest_hash):
    raw = read(path)
    summary = raw["task_evidence"]["summary"]
    canonical = validate_unified_stair_summary(suite=config.ordered_evaluation_suite,
        semantics=config.evaluation_semantics, split="public_dev", summary=summary)
    assert raw["status"] == "pass" and summary["split"] == "public_dev"
    assert raw["resolved_cell_manifest_hash"] == current_manifest_hash
    ordered = canonical["ordered_summary"]
    flags = sorted(config.evaluation_semantics.outcomes["diagnostic_flags"])
    by_cell = {cell: [] for cell in ordered["cell_ids"]}
    seed_cells = []
    for seed in ordered["seed_summaries"]:
        for cell, facts in seed["episode_facts_by_cell"].items():
            by_cell[cell].extend(facts)
            success = sum(f["primary_reason"] == "success" for f in facts)
            seed_cells.append({"seed_index": seed["seed_index"], "cell": cell,
                               "n": len(facts), "success": success, "failure": len(facts) - success})
    all_facts = [f for facts in by_cell.values() for f in facts]
    assert len(all_facts) == ordered["total_episode_count"] == 864
    aggregate = summarize(all_facts, flags)
    for reason, count in ordered["primary_outcome_counts"].items():
        assert aggregate["primary_counts"].get(reason, 0) == count
    for cell, facts in by_cell.items():
        counts = Counter(f["primary_reason"] for f in facts)
        assert len(facts) == 36
        assert all(counts[k] == n for k, n in ordered["primary_outcome_counts_by_cell"][cell].items())
        assert all(sum(flag in f["diagnostic_flags"] for f in facts) == n
                   for flag, n in ordered["diagnostic_flag_counts_by_cell"][cell].items())
    forbidden = [f for f in all_facts if f["primary_reason"] == "forbidden_contact"]
    progress_bins = Counter(min(int(f["progress_fraction"] * 10), 9) for f in forbidden)
    execution = raw["task_evidence"]["execution_identity"]
    training = read(TRAIN[label])
    checkpoint = raw["checkpoint_safety_report"]["checkpoint_hash"]
    assert checkpoint == training["checkpoint"]["checksum"]
    cell_summaries = {cell: summarize(facts, flags) for cell, facts in sorted(by_cell.items())}
    no_success = [cell for cell, s in cell_summaries.items()
                  if s["primary_counts"].get("success", 0) == 0 and s["episodes"] > 0]
    return {
        "source": {"path": path, "sha256": sha(path), "training_path": TRAIN[label],
                   "training_sha256": sha(TRAIN[label])},
        "validation": {"current_unified_summary_validator": "pass", "facts_count_closure": "pass",
                       "per_cell_primary_and_flag_count_closure": "pass", "checkpoint_receipt_match": "pass"},
        "identity": {"checkpoint_hash": checkpoint, "resolved_cell_manifest_hash": raw["resolved_cell_manifest_hash"],
            "resolved_cell_identity": raw["resolved_cell_identity"], "suite_hash": ordered["ordered_suite_hash"],
            "public_dev_cohort_hash": ordered["split_cohort_hash"], "evaluation_binding": raw["evaluation_binding"],
            "source_manifest_hash": execution["environment_source_preflight"]["source_manifest_hash"],
            "execution_environment_hash": execution["backend_execution_environment_verification"]["actual_manifest_hash"],
            "checkpoint_loading": raw["checkpoint_loading"], "constructed_learner": raw["constructed_learner"],
            "training_binding_hash": training["training_binding_hash"],
            "training_resolved_manifest_hash": training["resolved_cell_manifest_hash"],
            "training_cell": training["resolved_cell_identity"]["cell_variant_id"]},
        "scores": {k: canonical[k] for k in ("j_primary", "j_id", "j_ood", "aggregate_stratum_scores")},
        "aggregate": aggregate, "by_cell": cell_summaries,
        "by_stratum": {stratum: summarize([f for cell in cells for f in by_cell[cell]], flags)
                       for stratum, cells in canonical["stratum_cell_ids"].items()},
        "forbidden_max_progress_deciles": {"meaning": "maximum normalized root-x progress; not layer or collision position",
            "bins": [{"lower_inclusive": i / 10, "upper": (i + 1) / 10,
                      "upper_inclusive": i == 9, "n": progress_bins[i]} for i in range(10)]},
        "control_coverage": {
            "cell_count": len(by_cell), "cells_with_failure": sum(s["primary_counts"].get("success", 0) < s["episodes"] for s in cell_summaries.values()),
            "cells_without_any_success_control": no_success,
            "seed_cell_count": len(seed_cells),
            "seed_cells_with_failure_but_without_success": [s for s in seed_cells if s["failure"] and not s["success"]],
            "note": "An observed success in a cell is only a potential control; time-window matching is not yet available."},
        "saved_cost": raw["evaluation_execution_cost"]["splits"]["public_dev"],
    }


def main():
    config = StairClimbingTaskConfig.defaults(ROOT / "configs/tasks/stair_go2_12dof_unified_protocol_v3.json",
                                              cell_variant_id="stair.go2_12dof.unified_v3")
    manifest = resolve_stair_cell_manifest(cell_variant_id="stair.go2_12dof.unified_v3", config=config,
                                          plugin=StairClimbingTaskPlugin(config)).to_dict()
    policies = {label: analyze(label, path, config, manifest["manifest_hash"]) for label, path in SOURCES.items()}
    left, right = policies.values()
    identity_keys = ("resolved_cell_manifest_hash", "suite_hash", "public_dev_cohort_hash", "evaluation_binding",
                     "source_manifest_hash", "execution_environment_hash", "training_binding_hash", "training_resolved_manifest_hash")
    old_manifest = read(f"{FORMAL}/resolved_cell_manifest.json")
    old, new = old_manifest["contract_hashes"], manifest["contract_hashes"]
    ledger = [json.loads(line) for line in (ROOT / FORMAL / "ledger.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    generation8 = next(row for row in ledger if row["generation_index"] == 8)
    parent = generation8["parent_current_best"]
    formal_parent = parent["checkpoint_hash"]
    training_receipts = []
    for path in sorted((ROOT / FORMAL / "runner").glob("*/train/summary.json")):
        trained = json.loads(path.read_text(encoding="utf-8"))
        training_receipts.append({"trial_id": path.parent.parent.name, "path": path.relative_to(ROOT).as_posix(),
            "status": trained.get("status"), "checkpoint_hash": trained.get("checkpoint", {}).get("checksum"),
            "training_binding_hash": trained.get("training_binding_hash"),
            "resolved_cell_manifest_hash": trained.get("resolved_cell_manifest_hash")})
    replay = read(f"{REPLAY}/result.json")
    assert replay["status"] == "PASS" and replay["checkpoint_bytes_checked"] is True
    assert replay["phases_executed"] == ["public_dev"]
    assert replay["checkpoint_sha256"] == right["identity"]["checkpoint_hash"]
    body_profile = read("configs/robots/unitree_go2_12dof_isaaclab_v1.json")
    binding = read("configs/tasks/stair_go2_12dof_unified_binding_v3.json")
    result = {
        "kind": "offline_descriptive_two_policy_baseline", "schema_version": 1,
        "execution": {"simulator_jobs": 0, "training_jobs": 0, "provider_calls": 0},
        "boundaries": [
            "Facts contain outcome closure step, maximum normalized progress and diagnostic flags; symptom flags are ever-latches, while goal_not_reached is outcome-derived. No event timestamps or collision surface.",
            "Historical v2 scores are not used in v3 numeric differences.",
            "This P3-reference versus Gen8-A comparison is not the historical parent-child pair and has different source/runtime hashes.",
            "Rates are descriptive for saved episodes, not independent training replicates or reward-causal evidence."],
        "policies": policies,
        "comparison": {
            "same_identity_fields": [k for k in identity_keys if left["identity"][k] == right["identity"][k]],
            "different_identity_fields": [k for k in identity_keys if left["identity"][k] != right["identity"][k]],
            "j_primary_delta_gen8_minus_p3": right["scores"]["j_primary"] - left["scores"]["j_primary"],
            "success_count_delta_gen8_minus_p3": right["aggregate"]["primary_counts"]["success"] - left["aggregate"]["primary_counts"]["success"],
            "per_cell_success_count_delta_gen8_minus_p3": {c: right["by_cell"][c]["primary_counts"].get("success", 0) - left["by_cell"][c]["primary_counts"].get("success", 0) for c in left["by_cell"]},
            "historical_parent_child": formal_parent == left["identity"]["checkpoint_hash"],
            "formal_gen8_actual_parent": {k: parent.get(k) for k in ("anchor_id", "anchor_kind", "checkpoint_hash", "program_source_checksum")},
            "actual_parent_is_p3_reference": formal_parent == left["identity"]["checkpoint_hash"],
            "lineage_source": {"path": f"{FORMAL}/ledger.jsonl", "sha256": sha(f"{FORMAL}/ledger.jsonl"), "generation_index": 8},
            "availability": "facts-level cross-policy BehaviorDiff available; same-build temporal Diff requires Phase2 capture"},
        "v2_training_to_v3_eval": {
            "same_contract_hash_keys": sorted(k for k in old if old[k] == new.get(k)),
            "different_contract_hash_keys": sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k)),
            "robot_profile_same": old_manifest["robot_profile"] == manifest["robot_profile"],
            "runtime_adapter_same": old_manifest["runtime_adapter"] == manifest["runtime_adapter"],
            "resolved_roles_same": old_manifest["resolved_roles"] == manifest["resolved_roles"],
            "actual_replay": {"path": f"{REPLAY}/result.json", "sha256": sha(f"{REPLAY}/result.json"),
                "status": replay["status"], "phases": replay["phases_executed"], "checkpoint_bytes_checked": replay["checkpoint_bytes_checked"],
                "wall_time_s": replay["ended_at_unix"] - replay["started_at_unix"]},
            "preflight_checkpoint_bytes_checked": read(f"{REPLAY}/preflight_result.json")["checkpoint_bytes_checked"],
            "scope": "Loading/inference compatibility plus one verified replay; no blanket guarantee for 22 checkpoints or v2-score comparability."},
        "formal_training_pool": {
            "training_receipt_count": len(training_receipts),
            "status_counts": dict(Counter(r["status"] for r in training_receipts)),
            "receipts_with_checkpoint": sum(bool(r["checkpoint_hash"]) for r in training_receipts),
            "distinct_recorded_checkpoint_hashes": len({r["checkpoint_hash"] for r in training_receipts if r["checkpoint_hash"]}),
            "all_training_bindings_match_p3": all(r["training_binding_hash"] == left["identity"]["training_binding_hash"] for r in training_receipts),
            "all_training_manifests_match_p3": all(r["resolved_cell_manifest_hash"] == left["identity"]["training_resolved_manifest_hash"] for r in training_receipts),
            "generation_count": len(ledger), "lane_count": sum(len(r["trials"]) for r in ledger),
            "generation_parents": [{"generation_index": r["generation_index"],
                                    "checkpoint_hash": r["parent_current_best"]["checkpoint_hash"]} for r in ledger],
            "receipts": training_receipts,
            "note": "15 generation records are not 15 independent training lineages; all 22 checkpoints have not been replayed on v3 here."},
        "body_force_observability": {
            "profile_body_names": body_profile["semantic_roles"]["forbidden_contact"]["targets"],
            "task_termination_and_current_recorder_bodies": binding["role_bindings"]["forbidden_contact"],
            "existing_contact_sensor": binding["task_projection"]["phase0_binding_facts"]["sensors"]["contact_forces"],
            "optional_profile": "Save named 15-body current net-force norms, resolve sensor names at runtime; includes base already recorded.",
            "meaning": "net contact load on a body; not contact point, contacted object, riser/tread face or toe collision; leave termination roles unchanged"},
        "storage_example_not_measurement": {
            "assumptions": "288 envs * 640 control steps * 3 seeds; 61 float32/env-step including repeated origin; episode_length and terrain_type int64; 2 term masks + done bool; excludes metadata/container overhead",
            "bytes_per_env_step": 61 * 4 + 2 * 8 + 3,
            "bytes_per_three_seed_run": (61 * 4 + 2 * 8 + 3) * 288 * 640 * 3,
            "base_xyz_force_to_norm_saving_fraction": 8 / (61 * 4 + 2 * 8 + 3),
            "extra_14_body_norms_float32_bytes_per_run": 14 * 4 * 288 * 640 * 3,
            "note": "Array-layout estimate only; float16 needs threshold-error validation, not required for the first two-policy capture."},
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": OUT.relative_to(ROOT).as_posix(),
        "policy_count": len(policies), "real_episode_facts_checked": sum(v["aggregate"]["episodes"] for v in policies.values()),
        "historical_parent_child": result["comparison"]["historical_parent_child"],
        "j_primary_delta": result["comparison"]["j_primary_delta_gen8_minus_p3"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
