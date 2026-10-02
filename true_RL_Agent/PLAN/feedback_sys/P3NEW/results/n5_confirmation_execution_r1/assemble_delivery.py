"""Assemble two N5 task reports and capture author HTTP bytes without sending.

CLI: --inputs INPUT.json --source-root FROZEN_TREE --material-root OLD_REPO
     --output NEW_DIR
See ASSEMBLE_DELIVERY_INTERFACE.md. Carrier archival and combined-audit review
are inputs, not responsibilities of this driver. No semantic decisions are made.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def require(condition, message):
    if not condition:
        raise ValueError(message)


def ref(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


def checked(reference):
    path = Path(reference["path"])
    require(path.is_absolute(), "input references must use explicit absolute paths")
    actual = ref(path)
    require(actual["sha256"] == reference["sha256"], "input SHA mismatch: " + str(path))
    require("size_bytes" not in reference or actual["size_bytes"] == reference["size_bytes"],
            "input size mismatch: " + str(path))
    return path.resolve()


def read(reference):
    value = json.loads(checked(reference).read_bytes())
    pointer = reference.get("pointer", "")
    for part in pointer.strip("/").split("/") if pointer else ():
        part = part.replace("~1", "/").replace("~0", "~")
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def load(path, name):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def verify_combined(combined, originals, original_refs):
    """Check the frozen aggregate representation; never construct/rewrite it."""
    require(len(originals) == len(combined["calls"]) == 2, "each task needs both original audits")
    require(combined.get("scope") == "n5_confirmation"
            and combined.get("all_raw_observations_retained") is True
            and "semantic_review" not in combined, "use the unchanged, unreviewed combined audit")
    expected, raw = [], []
    for index, (audit, reference, call) in enumerate(zip(originals, original_refs, combined["calls"], strict=True)):
        prefix = f"calls[{index}]"
        if audit is None:
            require(reference is None and call.get("status") not in (None, "received")
                    and call.get("audit") is None and call.get("audit_sha256") is None
                    and prefix + ":" + call["status"] in combined["violations"],
                    "missing audit must retain the original failed/unsubmitted call")
            raw.append(None)
            continue
        require("semantic_review" not in audit and audit["schema_version"] == combined["schema_version"],
                "original audit changed or observer schemas differ")
        require(call.get("audit") == audit and call.get("audit_sha256") == reference["sha256"],
                "combined call does not retain the exact original audit and SHA")
        raw.append(deepcopy(audit["raw_output"]))
        require(all(prefix + ":" + issue in combined["violations"] for issue in audit["violations"]),
                "combined audit dropped a whole-output mechanical violation")
        for record in audit["observations"]:
            row = deepcopy(record)
            row["reference"] = prefix + "." + row["reference"]
            row["source_call_index"] = index
            expected.append(row)
    require(combined["raw_output"] == {"calls": raw} and combined["observations"] == expected,
            "combined audit omitted, reordered or rewrote original observation fields")
    require(combined["passed"] == (not combined["violations"]
            and not any(row["violations"] for row in expected)), "combined mechanical status differs")


def nonvisual(full, task):
    value = deepcopy(full)
    value.pop("observer_evidence", None)
    value.pop("observer_source_binding", None)
    owner = value["semantic"] if task == "stair" else value
    owner["sentences"] = [row for row in owner["sentences"] if row.get("section") != "visual"
                          and row.get("id") not in {"visual-unavailable", "push.visual.missing"}]
    if task == "stair":
        owner["sampling"].pop("visual_observed_episode_n", None)
    else:
        value["evidence_catalog"] = {key: row for key, row in value["evidence_catalog"].items()
                                     if key != "observer_evidence" and not key.startswith("observer_evidence/")}
    return value


def assemble(inputs_path, source_root, material_root, output):
    inputs_path, source_root, material_root, output = (
        path.resolve() for path in (inputs_path, source_root, material_root, output))
    inputs = json.loads(inputs_path.read_bytes())
    require(inputs.get("schema_version") == "p3new.n5_offline_delivery_inputs.v1"
            and set(inputs["tasks"]) == {"stair", "push"}, "two explicit N5 tasks are required")
    require(not output.exists(), "output must be a fresh directory")

    # PYTHONPATH must select the frozen package before the real old reader can
    # prepend its old src. Do not pretend a module originated at another path.
    import true_rl_agent
    require(Path(true_rl_agent.__file__).resolve().is_relative_to(source_root / "src"),
            "PYTHONPATH did not select the explicit frozen production source")
    from true_rl_agent.evaluation.binding import canonical_json_bytes, canonical_json_hash
    from true_rl_agent.evaluation.visual_review import attach_visual_review
    from true_rl_agent.candidates.authoring_policy import select_authoring_anchor_source
    from true_rl_agent.tasks.stair_climbing.visual_observation import attach_stair_visual_observations
    from true_rl_agent.tasks.stair_climbing.process_report import project_stair_process_report, MAX_PUBLIC_BYTES
    from true_rl_agent.tasks.push_recovery.visual_observation import attach_push_visual_observations
    from true_rl_agent.tasks.push_recovery.process_report import project_push_process_behavior, PUSH_PROCESS_MAX_BYTES
    base = source_root / "PLAN/feedback_sys"
    C = load(base / "P3R/confirmation_context.py", "n5_delivery_context")
    R = load(base / "P3R/author_ceiling_probe.py", "n5_delivery_first_prompt")
    D = load(base / "P3NEW/audit_delivery_offline.py", "n5_delivery_capture")
    reader_path = checked(inputs["qualification_reader"])
    require(reader_path == material_root / "PLAN/feedback_sys/P3NEW/audit_delivery_offline.py",
            "qualification reader must be the real original material-repository file")
    Q = load(reader_path, "n5_original_qualification_reader")
    require(checked(inputs["qualification_index"]) == Q.ORIGINALS / "source_index.json",
            "qualification index differs from the unchanged reader's actual source")
    bound_read, qualification_refs = Q.bound_reader()
    sys.path.insert(0, str(source_root / "src"))
    exp = C._p3(source_root)
    exp.bound_read = bound_read
    capture = read(inputs["capture_card"])
    frozen_inputs = [ref(inputs_path), ref(Path(__file__)), inputs["capture_card"],
                     inputs["qualification_reader"], inputs["qualification_index"], *qualification_refs]
    outputs = {}
    output.mkdir(parents=True)

    for task in ("stair", "push"):
        item = inputs["tasks"][task]
        for name in ("phase", "plan", "full", "combined_audit", "review", "qualified_case", "qualified_spec", "as_of_cutoff_ref"):
            checked(item[name])
            frozen_inputs.append(item[name])
        originals = [read(reference) if reference is not None else None for reference in item["audits"]]
        frozen_inputs.extend(reference for reference in item["audits"] if reference is not None)
        combined, review = read(item["combined_audit"]), read(item["review"])
        verify_combined(combined, originals, item["audits"])
        reviewed = attach_visual_review(combined, review)  # validates exact combined binding and all decisions
        phase, full = read(item["phase"]), read(item["full"])
        require("observer_evidence" not in full, "use the original numeric full before visual attachment")
        declared_full = phase["task_evidence"]["behavior_full"]
        require(declared_full["sha256"] == item["full"]["sha256"]
                and declared_full["size_bytes"] == checked(item["full"]).stat().st_size,
                "original phase/full binding differs")
        attach, project = ((attach_stair_visual_observations, project_stair_process_report) if task == "stair"
                           else (attach_push_visual_observations, project_push_process_behavior))
        before = project(full)
        require(before == phase["task_evidence"]["behavior"], "original phase projection differs from full")
        augmented = attach(full, reviewed, observer_version=item["observer_version"])
        public = project(augmented)
        require(nonvisual(augmented, task) == nonvisual(full, task), "visual attachment changed nonvisual full")
        require(all(public["report"][key] == before["report"][key]
                    for key in (*[key for key in D.SECTIONS if key != "visual"], "actionability")),
                "visual attachment changed required numeric/process/counterexample/gap content")
        public_limit = MAX_PUBLIC_BYTES if task == "stair" else PUSH_PROCESS_MAX_BYTES
        require(len(canonical_json_bytes(public)) <= public_limit, "bounded public report exceeds task byte limit")
        # No positive S3 requirement: wholly rejected/invalid observations retain
        # their S5 accounting, original audit and the unchanged numerical report.
        dest = output / task
        full_ref = D.write(dest / "behavior_full.json", augmented)
        public_ref = D.write(dest / "public_report.json", public)
        derived = deepcopy(phase)
        derived["task_evidence"]["behavior"] = public
        derived["task_evidence"]["behavior_full"] = {"schema_version": "true_rl_agent.process_full_ref.v1",
            "file": "behavior_full.json", "sha256": full_ref["sha256"], "size_bytes": full_ref["size_bytes"]}
        derived["task_evidence_hash"] = canonical_json_hash(derived["task_evidence"])
        phase_ref = D.write(dest / "public_dev_result.json", derived)

        case = deepcopy(bound_read(item["qualified_case"], material_root))
        spec = bound_read(item["qualified_spec"], material_root)
        require(case["task"] == spec["task"] == task
                and case["author_context"]["sha256"] == item["qualified_spec"]["sha256"],
                "qualified case/spec identity differs")
        cutoff = read(item["as_of_cutoff_ref"])
        require(isinstance(cutoff, str) and cutoff == item["as_of_cutoff"],
                "cutoff must equal the bound new observation/review sealing receipt field")
        instant = datetime.fromisoformat(cutoff.replace("Z", "+00:00"))
        require(instant.utcoffset() == timezone.utc.utcoffset(instant), "sealing cutoff must carry UTC timezone")
        case["as_of_cutoff"] = cutoff
        expected_binding = capture["tasks"][task]["current_evaluation_binding"]
        require(phase["evaluation_binding_hash"] == expected_binding["evaluation_binding_hash"],
                "new diagnostic phase is not the capture card's registered binding")
        extra = {}
        if task == "push":
            for name in ("source_manifest", "training_summary"):
                checked(item[name])
                frozen_inputs.append(item[name])
            extra = {"diagnostic_eval_base_seed": capture["tasks"][task]["cohort"]["eval_base_seed"],
                     "source_manifest_ref": item["source_manifest"], "training_summary_ref": item["training_summary"]}
        comp, builder, _ = C.build_context(case, "T", spec, root=source_root, base=material_root,
            output=dest / "context", diagnostic_phase_ref=phase_ref, diagnostic_plan_ref=item["plan"],
            allowed_seed_cohort_hash=expected_binding["seed_cohort_hash"], development_fixture=False, exp=exp, **extra)
        payload = R.first_prompt(comp, builder)
        body, user, delivered = D.capture_request(payload, dest)
        require(D.behavior_in(delivered, task) == public, "final serialized request changed the report")
        _, parent_hash, role = select_authoring_anchor_source(payload)
        require(role == "current_best" and parent_hash == spec["reference_program_source_checksum"].removeprefix("sha256:"),
                "serialized qualified parent source changed")
        (dest / "request.http_body.json").write_bytes(body)
        (dest / "user_payload.json").write_bytes(user)
        shown = {row["id"] for row in public["report"]["visual"]}
        outputs[task] = {"full": full_ref, "public": public_ref, "phase": phase_ref,
            "http_request": ref(dest / "request.http_body.json"), "payload": ref(dest / "user_payload.json"),
            "qualified_parent_source_sha256": parent_hash, "public_bytes": len(canonical_json_bytes(public)),
            "public_limit_bytes": public_limit, "as_of_cutoff": cutoff, "as_of_cutoff_ref": item["as_of_cutoff_ref"],
            "cutoff_basis": "explicit new observation/review sealing receipt; old case timestamp is not used",
            "shown_S3": sum(row.get("kind") == "S3" and row["id"] in shown for row in D.sentences(task, augmented)),
            "source_binding": augmented["observer_source_binding"], "nonvisual_unchanged": True,
            "budget": delivered["authoring_context_frame"]["budget_metadata"]}

    modules = {name: ref(Path(module.__file__)) for name, module in
               (("context", C), ("first_prompt", R), ("capture", D), ("qualification_reader", Q), ("qualification_entry", exp))}
    runtime = {name: ref(Path(module.__file__)) for name, module in tuple(sys.modules.items())
               if name.startswith("true_rl_agent") and getattr(module, "__file__", None)}
    require(all(Path(reference["path"]).is_relative_to(source_root / "src") for reference in runtime.values()),
            "mixed production package roots")
    for reference in frozen_inputs:
        checked(reference)
    receipt = {"status": "offline_serialized_not_sent", "assembled_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root), "material_root": str(material_root), "inputs": frozen_inputs,
        "modules": modules, "loaded_production_modules": runtime, "tasks": outputs,
        "development_fixture": False, "new_author_calls": 0, "model_calls": 0, "gpu_calls": 0,
        "semantic_decisions_created": False, "carrier_archive_or_eligibility_verified_here": False,
        "limits": "Consumes root-sealed carrier/combined audits and module reviews; no independent semantic verdict, qualification refresh, or actual author delivery."}
    D.write(output / "assembly_receipt.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("inputs", "source-root", "material-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = assemble(args.inputs, args.source_root, args.material_root, args.output)
    print(json.dumps({"status": result["status"], "tasks": list(result["tasks"]), "model_calls": 0}))
