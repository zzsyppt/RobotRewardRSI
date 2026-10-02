"""Offline r11 Push delivery boundary using the actual same-source r2 full.

Refresh only current task sentence templates over the persisted numerical v3
full. No L0 recomputation, model calls, JSON repair or semantic scoring. A future
complete review may be supplied explicitly; this helper never creates one.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[6]


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def require(value, reason):
    if not value:
        raise AssertionError(reason)


def numerical(full, *, keep_nonvisual_sentences):
    result = deepcopy(full)
    result.pop("observer_evidence", None)
    result.pop("observer_source_binding", None)
    result["sentences"] = ([row for row in result["sentences"] if row["section"] != "visual"]
                           if keep_nonvisual_sentences else [])
    result["evidence_catalog"] = {key: value for key, value in result["evidence_catalog"].items()
        if key != "observer_evidence" and not key.startswith("observer_evidence/")}
    return canonical(result)


def nonvisual(report):
    result = deepcopy(report["report"])
    result.pop("visual")
    result["gaps"] = [row for row in result["gaps"] if row["id"] != "report.capacity_omissions"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source-index", type=Path, required=True, help="Actual downloaded r2 source_index.json")
    parser.add_argument("--run-dir", type=Path, required=True, help="Original prepared r11 push_run01 directory")
    parser.add_argument("--process-source-sha256", required=True, help="Explicit current production template SHA")
    parser.add_argument("--review", type=Path, help="Root's complete external semantic review, never synthesized here")
    parser.add_argument("--output", type=Path, required=True, help="Fresh output directory")
    args = parser.parse_args()
    sys.path.insert(0, str(REPO / "src"))
    from true_rl_agent.evaluation.binding import canonical_json_hash
    from true_rl_agent.evaluation.process_report import report_bytes
    from true_rl_agent.evaluation.visual_review import attach_visual_review, visual_review_verdicts
    from true_rl_agent.tasks.push_recovery.process_report import (
        _evidence_catalog, _recovery_outline, build_push_process_sentences, project_push_process_behavior,
    )
    from true_rl_agent.tasks.push_recovery.visual_observation import attach_push_visual_observations
    dependencies = {}
    for name in ("evaluation.process_report", "evaluation.visual_observer", "evaluation.visual_review",
                 "tasks.push_recovery.process_report", "tasks.push_recovery.visual_observation"):
        path = Path(sys.modules["true_rl_agent." + name].__file__).resolve()
        require(path.is_relative_to(REPO / "src"), "loaded production source is outside current checkout")
        dependencies[path.relative_to(REPO).as_posix()] = sha(path)
    require(dependencies["src/true_rl_agent/tasks/push_recovery/process_report.py"] == args.process_source_sha256,
            "current Push template SHA differs from the explicitly supplied source")

    source_index = args.source_index.resolve()
    source_dir, run = source_index.parent, args.run_dir.resolve()
    index = read(source_index)
    watched = [source_index]
    for relative, reference in index["files"].items():
        path = (source_dir / relative).resolve()
        require(path.is_relative_to(source_dir), "source-index file escaped its directory")
        require(sha(path) == reference["sha256"] and path.stat().st_size == reference["bytes"], "downloaded source hash/size mismatch")
        watched.append(path)
    original, phase = read(source_dir / "behavior_full.json"), read(source_dir / "summary.json")
    reference = phase["task_evidence"]["behavior_full"]
    require(reference == index["behavior_full_reference"], "phase/full source reference differs")
    require(reference["sha256"] == sha(source_dir / "behavior_full.json"), "phase does not bind the actual source full")
    require(phase["task_evidence"]["behavior"]["full_derivation_sha256"] == canonical_json_hash(original), "original public/full binding differs")
    require(original["binding"] == index["binding"] and original["source_bindings"] == index["source_bindings"], "source-index identity differs")
    require(original["schema_version"] == "true_rl_agent.push_recovery.behavior.v3"
            and original["numerical_rule_version"] == "push_process_v2", "expected persisted v3 with v2 numerical authority")
    require("observer_evidence" not in original, "source full must precede visual attachment")
    seed_refs = phase["task_evidence"]["seed_summaries"]
    require({row["numeric_evidence_ref"]["manifest_sha256"] for row in seed_refs} == {
        row["manifest_sha256"] for row in original["source_bindings"]}, "phase seed cohort and full sources differ")
    for binding in original["source_bindings"]:
        manifest_path = source_dir / f"numeric/public_dev-{binding['actual_run_index']:02}/manifest.json"
        manifest = read(manifest_path)
        require(sha(manifest_path) == binding["manifest_sha256"]
                and manifest["identity"]["run_index"] == binding["actual_run_index"], "source/run manifest binding differs")

    audit_path = run / "response/mechanical_audit.json"
    raw_answer_path, raw_response_path = run / "response/raw_answer.txt", run / "response/raw_response.json"
    sidecar_path, result_path = run / "private_sidecar.json", run / "response/result.json"
    audit, sidecar, result = read(audit_path), read(sidecar_path), read(result_path)
    require("semantic_review" not in audit, "use the untouched original mechanical audit")
    require(result["request_sha256"] == sidecar["request_sha256"] and result["raw_response_sha256"] == sha(raw_response_path), "actual response/request binding differs")
    require(read(raw_response_path)["choices"][0]["message"]["content"] == raw_answer_path.read_text(encoding="utf-8"), "raw answer differs from provider response")
    identity = sidecar["numeric_identity"]
    for key in ("checkpoint_hash", "resolved_protocol_hash", "evaluation_plan_hash", "resolved_cell_manifest_hash",
                "runtime_dynamics_hash", "program_source_checksum"):
        require(identity[key] == original["binding"][key], "r11 and full identities differ: " + key)
    binding = next((row for row in original["source_bindings"] if row["actual_run_index"] == identity["run_index"]), None)
    require(binding is not None and binding["manifest_sha256"] == sidecar["numeric_reference"]["manifest_sha256"], "r11 uses another numeric source")
    for selection in sidecar["selection"]["episodes"]:
        require(selection["run_index"] == identity["run_index"] and any(
            row["source"] == binding["source_index"] and row["env_id"] == selection["env_index"]
            for row in original["transcripts"]), "selected episode is absent from the same-source full")
    watched += [audit_path, raw_answer_path, raw_response_path, sidecar_path, result_path]
    if args.review:
        watched.append(args.review.resolve())
    input_hashes = {str(path): sha(path) for path in watched}

    # Only the task sentences are refreshed; numerical fields and references stay intact.
    baseline_full = deepcopy(original)
    baseline_full["sentences"] = build_push_process_sentences(baseline_full)
    require(_evidence_catalog(baseline_full) == original["evidence_catalog"], "evidence catalog requires an explicit separate migration")
    old_numeric = numerical(original, keep_nonvisual_sentences=False)
    require(numerical(baseline_full, keep_nonvisual_sentences=False) == old_numeric, "template refresh changed numerical v3 material")
    baseline = project_push_process_behavior(baseline_full)
    example = next(row for row in baseline_full["sentences"] if row["id"] == "push.process.example")
    selected = baseline_full["representative_selection"]["context_episode_id"]
    episode = next(row for row in baseline_full["transcripts"] if f"r{row['source'] + 1}.e{row['env_id']}" == selected)
    require(example["required"] and all(_recovery_outline(episode, unit) in example["text"] for unit in episode["recoveries"]), "complete required recovery outline is missing")
    expected_numeric = numerical(baseline_full, keep_nonvisual_sentences=True)
    expected_nonvisual = nonvisual(baseline)
    before_audit = canonical(audit)
    unreviewed = attach_push_visual_observations(baseline_full, audit, observer_version="true_rl_agent.visual_observer.v3.r11.development")
    unreviewed_report = project_push_process_behavior(unreviewed)
    require(not any(row["kind"] == "S3" for row in unreviewed["sentences"] if row["section"] == "visual"), "unreviewed prose reached S3")
    require("0 shown" in " ".join(row["text"] for row in unreviewed_report["report"]["visual"]), "unreviewed output must show zero S3")

    if args.review:
        review = read(args.review)
        reviewed_audit = attach_visual_review(audit, review)
    else:
        require(not audit["passed"] and audit["violations"] and not audit["observations"], "readable observations require a complete external root review")
        review, reviewed_audit = None, audit
        if audit.get("error_type") == "JSONDecodeError":
            try:
                json.loads(raw_answer_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
            else:
                raise AssertionError("original answer no longer reproduces JSONDecodeError")
    final_full = attach_push_visual_observations(baseline_full, reviewed_audit, observer_version="true_rl_agent.visual_observer.v3.r11.development")
    final_report = project_push_process_behavior(final_full)
    require(final_full["observer_evidence"] == reviewed_audit, "attachment changed original audit or supplied review")
    verdicts = visual_review_verdicts(reviewed_audit)
    retained = []
    for row in final_full["sentences"]:
        if row["section"] != "visual" or row["kind"] != "S3":
            continue
        refs = [ref for ref in row["evidence_refs"] if ref.startswith("observer_evidence/observations/") and len(ref.split("/")) == 3]
        require(len(refs) == 1, "S3 must reference one immutable observation")
        i = int(refs[0].rsplit("/", 1)[1])
        raw = audit["observations"][i]["raw_observation"]
        require(verdicts[i] == "supported" and raw["text"] in row["text"] and raw["uncertainty"] in row["text"], "unsupported or rewritten S3")
        retained.append(i)
    if review is None:
        text = " ".join(row["text"] for row in final_report["report"]["visual"])
        require(not retained and "Visual output not delivered" in text and "Statement count unavailable" in text,
                "failed output was represented as zero correct observations")
        require(audit["error_type"] in text, "actual parse failure missing from S5")
    for label, full, report in (("unreviewed", unreviewed, unreviewed_report), ("final", final_full, final_report)):
        require(numerical(full, keep_nonvisual_sentences=True) == expected_numeric, label + " altered numerical/nonvisual material")
        require(nonvisual(report) == expected_nonvisual, label + " altered nonvisual report sections")
        require(report_bytes(report) <= 12288, label + " exceeded 12 KiB")
        require(set(report["report_projection"]["omitted_statement_ids"]) <= {"visual.observation"}, label + " removed required process evidence")
        require(report["full_derivation_sha256"] == canonical_json_hash(full), label + " full/public binding differs")
    require(canonical(audit) == before_audit and all(sha(Path(path)) == value for path, value in input_hashes.items()), "original inputs changed")
    require(not args.output.exists(), "output must be a fresh directory")
    args.output.mkdir(parents=True)
    outputs = {"template_full.json": baseline_full, "template_report.json": baseline,
               "unreviewed_report.json": unreviewed_report, "behavior_full.json": final_full, "public_report.json": final_report}
    updated_phase = deepcopy(phase)
    updated_phase["task_evidence"]["behavior"] = final_report
    updated_phase["task_evidence"]["behavior_full"] = {"schema_version": "true_rl_agent.process_full_ref.v1",
        "file": "behavior_full.json", "sha256": canonical_json_hash(final_full), "size_bytes": len(canonical(final_full))}
    outputs["summary.json"] = updated_phase
    for name, value in outputs.items():
        with (args.output / name).open("xb") as stream:
            stream.write(canonical(value))
    for source_path in (raw_answer_path, raw_response_path):
        with (args.output / source_path.name).open("xb") as stream:
            stream.write(source_path.read_bytes())
    receipt = {"status": "PASS_ACTUAL_FAILED_OUTPUT_BOUNDARY_NO_SEMANTIC_SCORE" if review is None else "PASS_SUPPLIED_REVIEW_PROJECTION_BOUNDARY_NOT_SEMANTIC_ACCEPTANCE",
        "new_api_calls": 0, "gpu_jobs": 0, "ssh_calls": 0, "actual_author_delivery": False,
        "derivation": "Same-source r2 persisted numerical v3; only current task sentence templates refreshed. L0 and v2 numerical reduction were NOT recomputed; evidence_catalog unchanged.",
        "loaded_current_source_sha256": dependencies, "helper_sha256": sha(Path(__file__)), "input_sha256": input_hashes,
        "source_v3_sha256": sha(source_dir / "behavior_full.json"), "source_derivation_hash_retained": original["source_derivation_hash"],
        "numerical_v3_without_sentence_templates_sha256": hashlib.sha256(old_numeric).hexdigest(),
        "refreshed_nonvisual_material_sha256": hashlib.sha256(expected_numeric).hexdigest(),
        "numeric_sources_and_phase_binding_verified": True, "all_numerical_fields_unchanged": True,
        "all_nonvisual_report_sections_unchanged": True, "complete_required_recovery_outline_present": True,
        "representative_episode": selected, "recovery_units": len(episode["recoveries"]),
        "original_mechanical_passed": audit["passed"], "original_violations": audit["violations"],
        "original_error_type": audit.get("error_type"), "enumerated_observations": len(audit["observations"]),
        "semantic_scoring": "not_performed", "necessary_fact_or_correctness_denominator": "unavailable after parse failure" if review is None else "not assessed by this helper",
        "external_review_supplied": review is not None, "semantic_review_created_here": False,
        "supplied_verdicts_not_rejudged": verdicts, "pre_capacity_s3_indices": retained,
        "baseline_report_bytes": report_bytes(baseline), "final_report_bytes": report_bytes(final_report), "report_limit": 12288,
        "final_visual": final_report["report"]["visual"], "final_projection": final_report["report_projection"],
        "full_public_phase_binding_verified": True, "original_raw_answer_bytes_unchanged": True,
        "outputs_sha256": {name: sha(args.output / name) for name in [*outputs, "raw_answer.txt", "raw_response.json"]}}
    with (args.output / "receipt.json").open("xb") as stream:
        stream.write(canonical(receipt))
    print(json.dumps({"status": receipt["status"], "report_bytes": receipt["final_report_bytes"],
        "S3": len(retained), "visual": receipt["final_visual"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
