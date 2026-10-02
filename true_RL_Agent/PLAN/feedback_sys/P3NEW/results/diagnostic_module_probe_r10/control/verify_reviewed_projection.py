"""Replay r10's real review delivery boundary; no API, simulator or semantic scoring.

Use the frozen r10 production source and unchanged same-run complete artifacts.
An external review is consumed verbatim. Invalid observer JSON is never repaired.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def require(condition, reason):
    if not condition:
        raise AssertionError(reason)


def sentences(full, task):
    return full["semantic"]["sentences"] if task == "stair" else full["sentences"]


def numerical_material(full, task):
    """Exclude only the documented visual attachment slots, not numerical data."""
    value = deepcopy(full)
    value.pop("observer_evidence", None)
    value.pop("observer_source_binding", None)
    container = value["semantic"] if task == "stair" else value
    container["sentences"] = [row for row in container["sentences"] if row["section"] != "visual"]
    if task == "stair":
        container["sampling"].pop("visual_observed_episode_n", None)
    else:
        value["evidence_catalog"] = {key: item for key, item in value["evidence_catalog"].items()
                                     if key != "observer_evidence" and not key.startswith("observer_evidence/")}
    return value


def nonvisual_report(report):
    value = deepcopy(report["report"])
    value.pop("visual")
    # Capacity receipts may be added, but original gap statements must survive.
    value["gaps"] = [row for row in value["gaps"] if row["id"] != "report.capacity_omissions"]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("stair", "push"), required=True)
    parser.add_argument("--audit", type=Path, required=True, help="Original mechanical_audit.json; never edited")
    parser.add_argument("--review", type=Path, help="Root's complete actual semantic_review.json")
    parser.add_argument("--output", type=Path, required=True, help="Fresh result directory")
    parser.add_argument("--source-root", type=Path, default=ROOT / "source")
    parser.add_argument("--full-source", type=Path)
    parser.add_argument("--observer-version", default="true_rl_agent.visual_observer.v3.r10.development")
    args = parser.parse_args()
    source = args.source_root.resolve()
    sys.path.insert(0, str(source / "src"))
    from true_rl_agent.evaluation.process_report import report_bytes
    from true_rl_agent.evaluation.visual_review import attach_visual_review, visual_review_verdicts
    from true_rl_agent.tasks.stair_climbing.process_report import project_stair_process_report
    from true_rl_agent.tasks.stair_climbing.visual_observation import attach_stair_visual_observations
    from true_rl_agent.tasks.push_recovery.process_report import project_push_process_behavior
    from true_rl_agent.tasks.push_recovery.visual_observation import attach_push_visual_observations

    files = read(ROOT / "source_manifest.json")["files"]
    dependencies = {}
    for module_name in ("evaluation.process_report", "evaluation.visual_observer", "evaluation.visual_review",
                        "tasks.stair_climbing.process_report", "tasks.stair_climbing.visual_observation",
                        "tasks.push_recovery.process_report", "tasks.push_recovery.visual_observation"):
        path = Path(sys.modules["true_rl_agent." + module_name].__file__).resolve()
        relative = path.relative_to(source).as_posix()
        dependencies[relative] = digest(path)
        require(dependencies[relative] == files[relative], "loaded source differs from r10 freeze: " + relative)
    full_path = (args.full_source or ROOT.parent / "native_observer_failure_boundary_actual_r2" /
                 "sources" / (args.task + "_behavior_full.json")).resolve()
    original, audit = read(full_path), read(args.audit)
    require("semantic_review" not in audit, "--audit must be the original unreviewed audit")
    input_paths = [full_path, args.audit.resolve(), ROOT / "source_manifest.json"]
    if args.review:
        input_paths.append(args.review.resolve())
    input_hashes = {str(path): digest(path) for path in input_paths}
    original_hash, audit_hash = canonical(original), canonical(audit)
    attach, project, limit = ((attach_stair_visual_observations, project_stair_process_report, 8192)
                             if args.task == "stair" else
                             (attach_push_visual_observations, project_push_process_behavior, 12288))
    baseline = project(original)
    expected_numeric = canonical(numerical_material(original, args.task))
    expected_nonvisual = nonvisual_report(baseline)
    unreviewed = attach(original, audit, observer_version=args.observer_version)
    unreviewed_report = project(unreviewed)
    require(not any(row["kind"] == "S3" for row in sentences(unreviewed, args.task)
                    if row["section"] == "visual"), "unreviewed visual prose entered S3")
    unreviewed_text = " ".join(row["text"] for row in unreviewed_report["report"]["visual"])
    require("0 shown" in unreviewed_text, "unreviewed coverage must say zero shown")
    require(("Visual output not delivered" in unreviewed_text) if audit["violations"] else
            (f"{len(audit['observations'])} unreviewed" in unreviewed_text),
            "unreviewed/failed output gap is missing")

    if args.review:
        review = read(args.review)
        reviewed_audit = attach_visual_review(audit, review)
    else:
        require(not audit["observations"] or audit["violations"], "readable observations need the actual root review")
        review, reviewed_audit = None, audit
    verdicts = visual_review_verdicts(reviewed_audit)
    reviewed = attach(original, reviewed_audit, observer_version=args.observer_version)
    reviewed_report = project(reviewed)
    require(reviewed["observer_evidence"] == reviewed_audit, "full artifact changed the supplied review or raw audit")
    if review is not None:
        require(reviewed["observer_evidence"]["semantic_review"] == review, "review provenance did not survive")
    retained = []
    for row in sentences(reviewed, args.task):
        if row["section"] != "visual" or row["kind"] != "S3":
            continue
        refs = [ref for ref in row["evidence_refs"] if ref.startswith("observer_evidence/observations/")
                and len(ref.split("/")) == 3]
        require(len(refs) == 1, "S3 does not identify exactly one original observation")
        index = int(refs[0].rsplit("/", 1)[1])
        record = audit["observations"][index]
        require(verdicts[index] == "supported", "unsupported observation reached S3")
        require(record["raw_observation"]["text"] in row["text"]
                and record["raw_observation"]["uncertainty"] in row["text"], "S3 rewrote the original prose")
        if record["numeric_status"] == "conflict":
            require("conflict (predicate only)" in row["text"], "review concealed numeric conflict")
        retained.append(index)
    rejected = set(reviewed["observer_source_binding"]["rejected_observation_indices"])
    eligible = [i for i, record in enumerate(audit["observations"])
                if verdicts[i] == "supported" and not record["violations"] and i not in rejected]
    if audit["violations"]:
        eligible = []
    require(bool(retained) == bool(eligible), "task attachment lost all eligible reviewed prose or admitted failed output")
    for label, full, report in (("unreviewed", unreviewed, unreviewed_report),
                                ("reviewed", reviewed, reviewed_report)):
        require(canonical(numerical_material(full, args.task)) == expected_numeric, label + " changed numerical material")
        require(nonvisual_report(report) == expected_nonvisual, label + " changed a nonvisual report section")
        require(report_bytes(report) <= limit, label + " exceeded report capacity")
        require(set(report["report_projection"]["omitted_statement_ids"]) <= {"visual.observation"},
                label + " capacity removed nonvisual statements")
    require(canonical(original) == original_hash and canonical(audit) == audit_hash, "input objects were mutated")
    require(all(digest(Path(path)) == value for path, value in input_hashes.items()), "input file bytes changed")
    require(not args.output.exists(), "output must be a fresh directory")
    args.output.mkdir(parents=True)
    write(args.output / "unreviewed_report.json", unreviewed_report)
    write(args.output / "reviewed_report.json", reviewed_report)
    write(args.output / "reviewed_full.json", reviewed)
    receipt = {"status": "OFFLINE_REVIEW_PROJECTION_BOUNDARY_PASS_NOT_SEMANTIC_ACCEPTANCE",
        "task": args.task, "new_api_calls": 0, "gpu_jobs": 0, "input_sha256": input_hashes,
        "loaded_frozen_source_sha256": dependencies, "helper_sha256": digest(Path(__file__)),
        "original_mechanical_passed": audit["passed"], "original_violations": audit["violations"],
        "enumerated_observations": len(audit["observations"]), "reviewer": review["reviewer"] if review else None,
        "semantic_verdicts_supplied_not_rejudged": verdicts,
        "pre_capacity_s3_observation_indices": retained,
        "observer_source_binding": reviewed["observer_source_binding"],
        "complete_nonvisual_numerical_material_sha256": expected_numeric,
        "all_nonvisual_report_sections_unchanged": True,
        "capacity_receipt_exception": "Only generated report.capacity_omissions may be added to gaps",
        "report_byte_limit": limit,
        "unreviewed_report_bytes": report_bytes(unreviewed_report), "reviewed_report_bytes": report_bytes(reviewed_report),
        "unreviewed_visual": unreviewed_report["report"]["visual"],
        "reviewed_visual": reviewed_report["report"]["visual"],
        "reviewed_projection": reviewed_report["report_projection"],
        "review_identity_and_original_audit_retained_in_full": True,
        "outputs_sha256": {name: digest(args.output / name) for name in
                           ("unreviewed_report.json", "reviewed_report.json", "reviewed_full.json")}}
    write(args.output / "receipt.json", receipt)
    print(json.dumps({"status": receipt["status"], "task": args.task,
                      "reviewed_report_bytes": receipt["reviewed_report_bytes"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
