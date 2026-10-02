"""Bind the preregistered questions to two complete, freshly measured phases.

CPU only. Full behavior is built once by the independent archive exporter;
this script checks its public projection and never decodes numeric chunks.
"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
from pathlib import Path
import sys

from author_experiment import (bound_read, digest, feedback_anchors, feedback_views, phase_source_bindings,
                               read, reference_context, write)


def file_ref(path):
    return {"path": str(path.resolve()), "sha256": digest(path)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--design", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    for task in ("stair", "push"):
        for artifact in ("context", "phase", "full"):
            p.add_argument(f"--{task}-{artifact}", type=Path, required=True)
    a = p.parse_args()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(a.root / "src"))
    card = copy.deepcopy(read(a.design))
    if any(c["status"] != "selection_pending_no_author_output" for c in card["cases"]):
        raise ValueError("prepare requires the untouched preregistered question design")
    as_of = datetime.now(timezone.utc).isoformat()
    a.output.mkdir(parents=True, exist_ok=True)
    prepared, report = {}, []
    for task in ("stair", "push"):
        spec = read(getattr(a, f"{task}_context"))
        phase_path, full_path = getattr(a, f"{task}_phase"), getattr(a, f"{task}_full")
        spec["diagnostic_public_dev_phase"] = file_ref(phase_path)
        spec["diagnostic_as_of"] = as_of
        comp, best, evidence, _, namespace, _ = reference_context(spec, root=a.root, base=a.output,
            output=a.output / "validation" / task)
        spec["diagnostic_replay_namespace"] = namespace
        phase, full = read(phase_path), read(full_path)
        if task == "stair":
            from true_rl_agent.tasks.stair_climbing.behavior import project_stair_behavior as project
        else:
            from true_rl_agent.tasks.push_recovery.behavior import project_push_behavior as project
        if project(full) != phase["task_evidence"]["behavior"]:
            raise ValueError(f"{task}: independent full truth does not reproduce the sealed phase behavior")
        views = feedback_views(spec, comp, best, evidence, base=a.output)
        context_path = a.output / f"author_context_{task}.json"
        write(context_path, spec)
        view_refs = {}
        for arm, view in views.items():
            path = a.output / f"author_feedback_{task}_{arm}.json"
            write(path, view)
            view_refs[arm] = file_ref(path)
        prepared[task] = {"author_context": file_ref(context_path),
            "feedback_A": view_refs["A"], "feedback_T": view_refs["T"],
            "blind_truth": file_ref(full_path), "source_bindings": phase_source_bindings(spec, base=a.output)}
        prepared[task]["derived_feedback_hash_changes"] = ([{
            "path": "public_feedback_summary.feedback_entries[0].source_public_result_hash",
            "canonical_source_hash_by_arm": {arm: anchor["public_result_hash"] for arm, anchor in
                feedback_anchors(spec, comp, best, evidence, base=a.output).items()},
            "source_objects_differ_only_by_behavior": True}] if task == "stair" else [])
        report.append({"task": task, "checkpoint_hash": best.checkpoint_hash,
            "phase": file_ref(phase_path), "full_truth": file_ref(full_path),
            "full_projection_equals_sealed_phase": True,
            "A_source": "same phase with behavior removed before task phase projection and feedback adapter",
            "T_source": "same sealed phase with production default bounded selector",
            "original_qualification_scores_changed": False})
    for case in card["cases"]:
        case.update(prepared[case["task"]])
        if case["checkpoint_hash"] != case["source_bindings"]["checkpoint_hash"]:
            raise ValueError("preregistered policy differs from measured phase")
        case["as_of_cutoff"] = as_of
        case["status"] = "ready_for_freeze"
        case["T_replacements"] = {"changed_path": ("public_feedback_summary.behavior" if case["task"] == "stair"
            else "public_feedback_summary.public_dev.behavior"), "old_feedback_fields_removed": [],
            "baseline_behavior": "task adapter's original absence or explicit missing marker is preserved in A",
            "case_specific_episode_selection": False,
            "derived_provenance_hash_changes": case["derived_feedback_hash_changes"]}
    card["status"] = "prepared_not_frozen"
    card["design_source"] = file_ref(a.design)
    card["diagnostic_availability_recorded_at"] = as_of
    write(a.output / "case_inputs.json", card)
    write(a.output / "case_preparation.json", {"status": "PASS", "api_calls": 0, "gpu_jobs": 0,
        "as_of_cutoff": as_of, "sources": report, "questions": 12, "policy_clusters": 2,
        "truth_visibility": "withheld from both author arms; public-dev measurements only"})
    print("PASS: two measured packages bound to twelve preregistered focused questions; no GPU/API")


if __name__ == "__main__":
    main()
