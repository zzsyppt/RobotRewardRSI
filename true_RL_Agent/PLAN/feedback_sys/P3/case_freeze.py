"""Freeze populated P3 cards and preflight; this command never calls a model."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from author_experiment import bound_read, digest, read, schedule, write


def freeze(stage_path, cases_path, preflight_path, output, *, release_api=False):
    stage, card, checked = read(stage_path), read(cases_path), read(preflight_path)
    cases = card["cases"]
    if len(cases) != 12 or sorted(c["task"] for c in cases) != ["push"] * 6 + ["stair"] * 6:
        raise ValueError("expected six Stair and six Push focused questions")
    if len({c["case_id"] for c in cases}) != 12:
        raise ValueError("duplicate case IDs")
    if len({c["process_category"] for c in cases}) < 2:
        raise ValueError("at least two preregistered process categories required")
    for case in cases:
        if case["status"] != "ready_for_freeze" or not case["common_focus"] or not case["as_of_cutoff"]:
            raise ValueError("unpopulated case")
        for key in ("author_context", "feedback_A", "feedback_T", "blind_truth"):
            bound_read(case[key], cases_path.parent)
        if case["visibility"] != "public_dev" or case["parent_diff_status"] != "missing_no_measured_child":
            raise ValueError("this registered reference-only experiment has no candidate Diff")
    if checked["status"] != "PASS" or len(checked["prompts"]) != 24:
        raise ValueError("all 24 first prompts must pass real-builder preflight")
    expected = schedule(stage, cases)
    if checked["schedule"] != expected:
        raise ValueError("preflight schedule differs from card")
    for row in checked["prompts"]:
        if digest(Path(row["path"])) != row["sha256"]:
            raise ValueError("preflight prompt bytes changed")
    result = {"schema_version": "true_rl_agent.p3.author_freeze.v1", "status": "FROZEN",
        "root_api_authorized": release_api, "stage_sha256": digest(stage_path), "cases_sha256": digest(cases_path),
        "preflight_sha256": digest(preflight_path), "schedule": expected,
        "production_source_snapshot": checked["production_source_snapshot"], "harness_sha256": checked["harness_sha256"],
        "first_prompts": checked["prompts"], "human_value_gate": "PENDING_REVIEW",
        "case_count": 12, "policy_cluster_count": len({c["cluster_id"] for c in cases}),
        "observation_packages": len({c["observation_package_id"] for c in cases}),
        "warning": "12 focused questions are correlated views of 2 policies, not 12 independent observations."}
    if output.exists():
        if read(output) != result:
            raise ValueError("freeze already exists with different contents; do not overwrite")
    else:
        write(output, result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for field in ("stage", "cases", "preflight", "output"):
        p.add_argument("--" + field, type=Path, required=True)
    p.add_argument("--release-api", action="store_true", help="Root's explicit post-review release, not a model call")
    a = p.parse_args()
    result = freeze(a.stage, a.cases, a.preflight, a.output, release_api=a.release_api)
    print(json.dumps({k: result[k] for k in ("status", "root_api_authorized", "case_count", "policy_cluster_count")}))
