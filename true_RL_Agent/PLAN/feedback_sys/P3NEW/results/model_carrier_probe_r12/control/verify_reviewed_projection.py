"""Offline r12 supplied-review projection over the frozen same-source r2 template.

No model call, semantic rescoring, raw-response fabrication or L0 recomputation.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[6]
CONTROL = Path(__file__).resolve().parent
RESULTS = CONTROL.parent.parent
R11 = RESULTS / "body_grounding_observer_r11"
BASE = R11 / "control/push_run01/failure_projection/template_full.json"
SOURCES = R11 / "materials/r2_production"
OUT = CONTROL / "reviewed_projection"


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def require(value, reason):
    if not value:
        raise AssertionError(reason)


def material(full, *, sentences=True):
    result = deepcopy(full)
    result.pop("observer_evidence", None)
    result.pop("observer_source_binding", None)
    result["sentences"] = [r for r in result["sentences"] if r["section"] != "visual"] if sentences else []
    result["evidence_catalog"] = {k: v for k, v in result["evidence_catalog"].items()
                                  if k != "observer_evidence" and not k.startswith("observer_evidence/")}
    return canonical(result)


def nonvisual(report):
    return {k: v for k, v in report["report"].items() if k != "visual"}


def main():
    require(not OUT.exists(), "output must be fresh")
    sys.path.insert(0, str(REPO / "src"))
    from true_rl_agent.evaluation.binding import canonical_json_hash
    from true_rl_agent.evaluation.process_report import report_bytes
    from true_rl_agent.evaluation.visual_review import attach_visual_review, visual_review_verdicts
    from true_rl_agent.tasks.push_recovery.process_report import _recovery_outline, project_push_process_behavior
    from true_rl_agent.tasks.push_recovery.visual_observation import attach_push_visual_observations

    dependencies = {}
    for name in ("evaluation.binding", "evaluation.process_report", "evaluation.visual_observer", "evaluation.visual_review",
                 "tasks.push_recovery.process_report", "tasks.push_recovery.visual_observation"):
        path = Path(sys.modules["true_rl_agent." + name].__file__).resolve()
        require(path.is_relative_to(REPO / "src"), "production module loaded outside current checkout")
        dependencies[path.relative_to(REPO).as_posix()] = sha(path)
    require(dependencies["src/true_rl_agent/tasks/push_recovery/process_report.py"] ==
            "1a4609d679abc9135832c37289022a7d9c77c0dd2a7e6d95d6bb8838f68506cf", "Push source changed")
    expected = {BASE: "d56f791c0db5b2b6ae05b4e0a17bcea7ec7ba8a08a8d62377606d5f73fc8f5b2",
                CONTROL / "mechanical_audit.json": "232fa635f333d251b139ab4e4b6b6941964727fff6167c353ec19c49fb9fa0d2",
                CONTROL / "root_semantic_review.json": "457f9c3e42251c34a07a36984c24602a7f29cc9f14c6fb4b8b366be8c5ecdb9e"}
    require(all(sha(p) == h for p, h in expected.items()), "fixed r12 input hash differs")
    watched = [p for root in (BASE.parent, SOURCES, CONTROL.parent / "inputs", CONTROL.parent / "output")
               for p in root.rglob("*") if p.is_file()]
    watched += [p for p in CONTROL.iterdir() if p.is_file()] + [R11 / "control/verify_reviewed_projection.py"]
    input_hashes = {p.relative_to(REPO).as_posix(): sha(p) for p in watched}
    base, original, phase = read(BASE), read(SOURCES / "behavior_full.json"), read(SOURCES / "summary.json")
    index, window = read(SOURCES / "source_index.json"), read(CONTROL / "actual_window_binding.json")
    for relative, ref in index["files"].items():
        path = SOURCES / relative
        require(sha(path) == ref["sha256"] and path.stat().st_size == ref["bytes"], "r2 source file hash/size differs")
    require(phase["task_evidence"]["behavior_full"] == index["behavior_full_reference"]
            and index["behavior_full_reference"]["sha256"] == sha(SOURCES / "behavior_full.json")
            and phase["task_evidence"]["behavior"]["full_derivation_sha256"] == canonical_json_hash(original), "original phase/full binding differs")
    require(base["binding"] == index["binding"] and base["source_bindings"] == index["source_bindings"]
            and material(base, sentences=False) == material(original, sentences=False), "template differs from same-source r2 numerical full")
    require("observer_evidence" not in base, "baseline already contains observer evidence")
    for key in ("checkpoint_hash", "resolved_protocol_hash", "evaluation_plan_hash", "resolved_cell_manifest_hash",
                "runtime_dynamics_hash", "program_source_checksum"):
        require(window["numeric_identity"][key] == base["binding"][key], "window/full binding differs: " + key)
    source = next(r for r in base["source_bindings"] if r["actual_run_index"] == window["numeric_identity"]["run_index"])
    require(source["manifest_sha256"] == window["numeric_reference"]["manifest_sha256"], "window uses another r2 numeric source")
    audit, review = read(CONTROL / "mechanical_audit.json"), read(CONTROL / "root_semantic_review.json")
    require(audit["passed"] and not audit["violations"] and "semantic_review" not in audit, "expected untouched passing mechanical audit")
    require(read(CONTROL.parent / "output/raw_answer.txt") == audit["raw_output"], "actual SOL answer differs from original audit")
    identities = {r["alias"]: r["identity"] for r in window["prepare_private_sidecar"]}
    require(all(r["media_identity"] == identities[r["alias"]] for r in audit["observations"]), "audit/window identity differs")

    baseline = project_push_process_behavior(base)
    version = "true_rl_agent.visual_observer.v3.r12.sol_development"
    unreviewed = attach_push_visual_observations(base, audit, observer_version=version)
    unreviewed_report = project_push_process_behavior(unreviewed)
    require(not any(r["kind"] == "S3" for r in unreviewed["sentences"] if r["section"] == "visual"), "unreviewed prose reached S3")
    reviewed_audit = attach_visual_review(audit, review)
    full = attach_push_visual_observations(base, reviewed_audit, observer_version=version)
    report = project_push_process_behavior(full)
    require(full["observer_evidence"] == reviewed_audit and reviewed_audit == {**audit, "semantic_review": review}, "original audit/review changed")
    require(not full["observer_source_binding"]["rejected_observation_indices"], "actual r2 source binding rejected")
    verdicts = visual_review_verdicts(reviewed_audit)
    shown = []
    public_visual = {r["id"]: r["text"] for r in report["report"]["visual"]}
    for row in full["sentences"]:
        if row["section"] != "visual" or row["kind"] != "S3" or row["id"] not in public_visual:
            continue
        require(public_visual[row["id"]] == row["text"], "public projection changed S3 text")
        refs = [r for r in row["evidence_refs"] if r.startswith("observer_evidence/observations/") and len(r.split("/")) == 3]
        require(len(refs) == 1, "S3 must bind one original observation")
        i = int(refs[0].rsplit("/", 1)[1])
        raw = audit["observations"][i]["raw_observation"]
        require(verdicts[i] == "supported" and raw["text"] in row["text"] and raw["uncertainty"] in row["text"], "S3 changed original prose or lacks support")
        shown.append(i)
    require(shown == [0] and verdicts == ["supported"] * 3, "actual current one-S3 selection differs")
    selected = base["representative_selection"]["context_episode_id"]
    episode = next(r for r in base["transcripts"] if f"r{r['source'] + 1}.e{r['env_id']}" == selected)
    example = next(r for r in full["sentences"] if r["id"] == "push.process.example")
    require(example["required"] and all(_recovery_outline(episode, unit) in example["text"] for unit in episode["recoveries"]), "complete required sequence outline missing")
    for label, attached, projected in (("unreviewed", unreviewed, unreviewed_report), ("reviewed", full, report)):
        require(material(attached) == material(base), label + " changed numerical/nonvisual full fields")
        require(nonvisual(projected) == nonvisual(baseline), label + " changed nonvisual public sections")
        require(report_bytes(projected) <= 12288 and not projected["report_projection"]["omitted_statement_ids"], label + " exceeded capacity or omitted evidence")
        require(projected["full_derivation_sha256"] == canonical_json_hash(attached), label + " full/public binding differs")
    require(all(sha(REPO / p) == h for p, h in input_hashes.items()), "original r11/r12 inputs changed")
    refreshed_phase = deepcopy(phase)
    refreshed_phase["task_evidence"]["behavior"] = report
    refreshed_phase["task_evidence"]["behavior_full"] = {"schema_version": "true_rl_agent.process_full_ref.v1",
        "file": "behavior_full.json", "sha256": canonical_json_hash(full), "size_bytes": len(canonical(full))}
    OUT.mkdir()
    outputs = {"behavior_full.json": full, "public_report.json": report,
               "unreviewed_report.json": unreviewed_report, "summary.json": refreshed_phase}
    for name, value in outputs.items():
        (OUT / name).write_bytes(canonical(value))
    require(sha(OUT / "behavior_full.json") == refreshed_phase["task_evidence"]["behavior_full"]["sha256"], "saved full/phase hash differs")
    receipt = {"status": "PASS_SUPPLIED_REVIEW_OFFLINE_PROJECTION_NOT_N5", "new_api_calls": 0, "gpu_jobs": 0, "ssh_calls": 0,
        "actual_author_delivery": False, "semantic_review_created_here": False, "semantic_scoring": "not_performed; root supplied nonblind development review unchanged",
        "carrier": "Actual SOL agent raw_answer.txt; no provider raw_response or provider request/response receipt exists or is synthesized here.",
        "derivation": "Direct frozen r11 template_full reuse over same-source r2 persisted numerical v3; no L0/v2 recomputation or new template refresh.",
        "loaded_current_source_sha256": dependencies, "helper_sha256": sha(Path(__file__)), "input_sha256": input_hashes,
        "source_v3_sha256": sha(SOURCES / "behavior_full.json"), "baseline_template_sha256": sha(BASE),
        "numerical_material_sha256": hashlib.sha256(material(base, sentences=False)).hexdigest(),
        "nonvisual_material_sha256": hashlib.sha256(material(base)).hexdigest(),
        "all_numerical_fields_and_transcripts_unchanged": True, "all_nonvisual_public_sections_unchanged": True,
        "counterexample_residual_and_required_sequence_unchanged": True, "original_r11_r12_inputs_unchanged": True,
        "numeric_source_and_full_public_phase_binding_verified": True, "observer_source_binding": full["observer_source_binding"],
        "representative_episode": selected, "outlined_recovery_units": len(episode["recoveries"]),
        "unreviewed_s3_count": 0, "supplied_verdicts_not_rejudged": verdicts, "shown_observation_indices": shown,
        "all_original_observations_and_uncertainties_retained_in_full": True,
        "baseline_report_bytes": report_bytes(baseline), "unreviewed_report_bytes": report_bytes(unreviewed_report),
        "final_report_bytes": report_bytes(report), "report_limit_bytes": 12288,
        "final_visual": report["report"]["visual"], "final_projection": report["report_projection"],
        "outputs_sha256": {name: sha(OUT / name) for name in outputs}}
    (OUT / "receipt.json").write_bytes(canonical(receipt))
    print(json.dumps({"status": receipt["status"], "report_bytes": report_bytes(report), "shown": shown,
                      "visual": report["report"]["visual"], "receipt_sha256": sha(OUT / "receipt.json")}))


if __name__ == "__main__":
    main()
