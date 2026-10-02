"""CPU-only archival of real SOL answers; never submits or reviews a model.

Run only after root authorizes execution and binds the two actual response files.
The legacy combined scope names its representation, not confirmation validity.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

STATUS = "DEVELOPMENT_REPAIR_NOT_ORIGINAL_N5_CONFIRMATION"


def require(value, reason):
    if not value:
        raise ValueError(reason)


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def ref(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return {"path": str(path), "sha256": sha(raw), "size_bytes": len(raw)}


def checked(reference):
    path = Path(reference["path"])
    require(path.is_absolute(), "explicit absolute input path required")
    actual = ref(path)
    require(actual["sha256"] == reference["sha256"], "input hash differs: " + str(path))
    require("size_bytes" not in reference or actual["size_bytes"] == reference["size_bytes"], "input size differs")
    return path.resolve()


def read(path):
    return json.loads(Path(path).read_bytes())


def inside(root, relative):
    path = (root / relative).resolve()
    require(path.is_relative_to(root), "carrier material path escapes its directory")
    return path


def check_access_metadata(access, materials, materials_sha256):
    """Check the two actual receipt formats without adding absent assertions.

    This compares receipt metadata, not an independent replay of desktop tools.
    E1 records one-based view_order and detail; E2 records zero-based order and
    no detail or per-view SHA (its files_read still binds every image SHA).
    """
    require(access.get("schema_version") == "p3new.observer_access_receipt.v1"
            and access.get("materials_sha256") == materials_sha256, "access receipt material binding differs")
    expected = {r["path"]: r["sha256"] for r in materials}
    expected["materials.json"] = materials_sha256
    reads = access.get("files_read")
    require(isinstance(reads, list) and len(reads) == len(expected)
            and {r["path"]: r["sha256"] for r in reads} == expected, "receipt file access does not cover exact original material set")
    images = [r for r in materials if r["role"] == "original_image"]
    e1_shape = "images_viewed" in access and "viewed_images" not in access
    e2_shape = "viewed_images" in access and "images_viewed" not in access
    require(e1_shape or e2_shape, "unsupported actual image-view receipt structure")
    views = access["images_viewed" if e1_shape else "viewed_images"]
    require(isinstance(views, list) and len(views) == len(images), "unsupported or incomplete actual image-view receipt")
    for index, (image, view) in enumerate(zip(images, views, strict=True)):
        require(view["view_order" if e1_shape else "order"] == index + int(e1_shape)
                and view["path"] == image["path"] and view["step"] == image["step"]
                and view["view_id"] == image["view_id"]
                and view["original_dimensions_px"] == image["image_size_px"], "receipt view identity/order/dimensions differ")
        if e1_shape:
            require(view["sha256"] == image["sha256"] and view["detail"] == "original", "recorded e1 view hash/detail differs")
    return {"status": "CHECKED_ACTUAL_RECEIPT_METADATA", "files_read": len(reads), "images_viewed": len(views),
        "materials_sha256": materials_sha256, "scope": "attestation consistency; not independent desktop tool replay",
        "actual_view_list_key": "images_viewed" if e1_shape else "viewed_images",
        "order_base": 1 if e1_shape else 0, "per_view_sha_recorded": e1_shape,
        "render_detail_assertion": "original" if e1_shape else None,
        "e2_missing_detail_not_inferred": e2_shape,
        "reported_actual_carrier": deepcopy(access.get("actual_carrier")),
        "original_access_limitations": deepcopy(access.get("access_limitations"))}


def main(inputs_path):
    started = datetime.now(timezone.utc).isoformat()
    config = read(inputs_path)
    require(config["status"] == STATUS and config["original_n5_status"] == "INCONCLUSIVE", "engineering scope required")
    require(config["execution_authorized_by_root"] is True, "draft is not released for CPU execution")
    require(Path(config["source_root"]).is_absolute(), "explicit source root required")
    source = Path(config["source_root"]).resolve()
    sys.path[:0] = [str(source / "src"), str(source / "PLAN/feedback_sys/P3NEW")]
    import true_rl_agent
    import prepare_native_observer as native
    from true_rl_agent.evaluation.visual_observer import SCHEMA_VERSION, audit_observer_output
    for module in (true_rl_agent, native, sys.modules[audit_observer_output.__module__]):
        require(Path(module.__file__).resolve().is_relative_to(source), "wrong source module origin")
    for relative, expected in config["source_sha256"].items():
        require(ref(source / relative)["sha256"] == expected, "registered audit source changed: " + relative)

    inputs = [ref(inputs_path), config["report"], config["registration"],
              config["desktop_invocations"], config["root_selected_media_quality"]]
    report_path, registration_path = checked(config["report"]), checked(config["registration"])
    report, registration = read(report_path), read(registration_path)
    invocations = read(checked(config["desktop_invocations"]))
    checked(config["root_selected_media_quality"])
    require(invocations["scope"] == STATUS and invocations["original_n5_status"] == "INCONCLUSIVE",
            "desktop invocations need the actual engineering registration")
    root = report_path.parent
    require(report["task"] == "push" and report["scope"] == "n5_confirmation", "wrong prepared template")
    require([p["episode_alias"] for p in report["packets"]] == ["e1", "e2"]
            and [p["directory"] for p in report["packets"]] == ["e1", "e2"]
            and [p["episode_alias"] for p in config["packets"]] == ["e1", "e2"], "frozen packet order differs")
    for name in ("capture_card", "selection_card"):
        checked(report[name])
        inputs.append(report[name])
    combined_path, receipt_path = root / "combined_audit.json", root / "sol_engineering_audit_receipt.json"
    audit_paths = [root / alias / "sol_response/mechanical_audit.json" for alias in ("e1", "e2")]
    require(not any(p.exists() for p in (*audit_paths, combined_path, receipt_path)), "preserve existing audits; outputs must be fresh")
    calls, observations, violations, raw_calls, staged = [], [], [], [], []
    carrier_checks = []
    for index, (binding, supplied) in enumerate(zip(report["packets"], config["packets"], strict=True)):
        alias = binding["episode_alias"]
        packet, prefix = root / alias, f"calls[{index}]"
        require(ref(packet / "card.json")["sha256"] == binding["prepared_card_sha256"]
                and ref(packet / "request.json")["sha256"] == binding["request_sha256"]
                and ref(packet / "selection.json")["sha256"] == report["selection_sha256"], "prepared packet binding differs")
        private = read(packet / "private_sidecar.json")
        built = native.build(Path(private["source_export"]), packet / "selection.json", episode_alias=alias)
        require(built["prepared"]["request_bytes"] == (packet / "request.json").read_bytes()
                and built["card"]["extraction_sha256"] == report["extraction_sha256"], "rebuilt template differs")
        for name in ("card", "private_sidecar", "mechanical_audit", "alignment"):
            require(built[name] == read(packet / (name + ".json")), "rebuilt preparation differs: " + alias + "/" + name)
            inputs.append(ref(packet / (name + ".json")))
        inputs.extend((ref(packet / "request.json"), ref(packet / "selection.json")))

        materials_path = checked(supplied["materials"])
        require(materials_path == packet / "blind_inputs/materials.json", "carrier materials from another packet")
        materials = read(materials_path)["materials"]
        carrier = read(checked(supplied["carrier_preparation"]))
        require(carrier["scope"] == STATUS and carrier["source_request_sha256"] == binding["request_sha256"]
                and carrier["materials_sha256"] == supplied["materials"]["sha256"], "carrier preparation differs")
        inputs.append(supplied["carrier_preparation"])
        invocation, = [r for r in invocations["invocations"] if r["alias"] == alias]
        require(invocation["agent_id"] == supplied["registered_agent"] and invocation["fork_turns"] == "none"
                and invocation["model"] == "gpt-6-sol" and invocation["reasoning_effort"] == "high"
                and invocation["input_manifest_sha256"] == supplied["materials"]["sha256"]
                and invocation["no_retry"] is True, "registered fresh agent/input differs")
        text_rows = [r for r in materials if r["role"] == "request_text"]
        image_rows = [r for r in materials if r["role"] == "original_image"]
        require(len(text_rows) == 1 and len(text_rows) + len(image_rows) == len(materials), "unexpected carrier material roles")
        for row in materials:
            require(ref(inside(materials_path.parent, row["path"]))["sha256"] == row["sha256"], "actual carrier material bytes changed")
        payload = built["prepared"]["payload"]
        expected_text = {key: value for key, value in payload.items() if key != "messages"}
        expected_text["messages"] = [{"role": message["role"], "text_blocks": (
            [message["content"]] if isinstance(message["content"], str) else
            [part["text"] for part in message["content"] if part["type"] == "text"])} for message in payload["messages"]]
        require(read(inside(materials_path.parent, text_rows[0]["path"])) == expected_text, "SOL blind text changed from prepared template")
        images = [image for window in built["windows"] for image in window.images]
        require(len(images) == len(image_rows), "carrier image count differs")
        for image, row in zip(images, image_rows, strict=True):
            require(row["sha256"] == sha(image.data) and row["step"] == image.displayed_steps[0]
                    and row["view_id"] == image.view_id and row["image_size_px"] == [image.width, image.height], "carrier image/order differs")

        raw_path, access_path = checked(supplied["raw_answer"]), checked(supplied["access_receipt"])
        require(raw_path == packet / "sol_response/raw_answer.txt"
                and access_path == packet / "sol_response/access_receipt.json", "actual SOL response path differs")
        access = read(access_path)
        require(isinstance(access, dict), "preserve and inspect malformed access receipt before use")
        access_check = check_access_metadata(access, materials, supplied["materials"]["sha256"])
        access_check["access_receipt_sha256"] = supplied["access_receipt"]["sha256"]
        inputs.extend((supplied["materials"], supplied["raw_answer"], supplied["access_receipt"]))
        carrier_checks.append(access_check)
        answer = None
        try:
            answer = json.loads(raw_path.read_bytes())
            audit = audit_observer_output(answer, built["windows"], numeric_check=built["numeric_check"])
        except Exception as error:
            audit = {"schema_version": SCHEMA_VERSION, "raw_output": answer, "observations": [],
                "violations": ["answer_parse_or_audit_error"], "passed": False,
                "error_type": type(error).__name__, "semantic_support": "not_evaluated",
                "raw_answer_preserved_in": supplied["raw_answer"]}
        audit.update(engineering_status=STATUS, original_n5_status="INCONCLUSIVE")
        audit_bytes = encoded(audit)
        audit_ref = {"path": str(audit_paths[index]), "sha256": sha(audit_bytes), "size_bytes": len(audit_bytes)}
        staged.append((audit_paths[index], audit_bytes))
        raw_calls.append(deepcopy(audit["raw_output"]))
        violations.extend(prefix + ":" + issue for issue in audit["violations"])
        for record in audit["observations"]:
            row = deepcopy(record)
            row["reference"] = prefix + "." + row["reference"]
            row["source_call_index"] = index
            observations.append(row)
        calls.append({"episode_alias": alias, "status": "received", "carrier": "fresh_subagent",
            "request_sha256": binding["request_sha256"], "request_binding_kind": "prepared_template_not_sent_HTTP",
            "receipt_sha256": supplied["access_receipt"]["sha256"], "receipt_kind": "actual_SOL_access_receipt",
            "access_receipt": supplied["access_receipt"], "original_access_receipt": access,
            "raw_answer": supplied["raw_answer"], "materials": supplied["materials"],
            "registered_agent": supplied["registered_agent"], "registered_invocation": deepcopy(invocation),
            "registration": config["registration"],
            "audit": audit, "audit_sha256": audit_ref["sha256"], "raw_response_sha256": None})
    combined = {"schema_version": SCHEMA_VERSION, "scope": "n5_confirmation",
        "scope_note": "legacy aggregate representation only; original confirmation remains INCONCLUSIVE",
        "engineering_status": STATUS, "original_n5_status": "INCONCLUSIVE",
        "report_card_sha256": config["report"]["sha256"], "calls": calls,
        "raw_output": {"calls": raw_calls}, "observations": observations, "violations": violations,
        "passed": not violations and not any(row["violations"] for row in observations),
        "semantic_support": "not_evaluated", "all_raw_observations_retained": True,
        "actual_desktop_calls_archived": 2, "actual_desktop_cost_usd": None,
        "provider_HTTP_receipts_created": 0, "usage_or_fingerprint_invented": False}
    staged.append((combined_path, encoded(combined)))
    modules = {name: ref(module.__file__) for name, module in tuple(sys.modules.items())
        if (name.startswith("true_rl_agent") or name == "prepare_native_observer") and getattr(module, "__file__", None)}
    require(all(Path(value["path"]).is_relative_to(source) for value in modules.values()), "mixed source roots")
    for reference in inputs:
        checked(reference)
    for path, raw in staged:
        with path.open("xb") as stream:
            stream.write(raw)
    receipt = {"status": STATUS, "original_n5_status": "INCONCLUSIVE", "started_at_utc": started,
        "sealed_at_utc": datetime.now(timezone.utc).isoformat(), "inputs": inputs,
        "source_root": str(source), "loaded_modules": modules, "script": ref(__file__),
        "registration_original": registration, "carrier_access_metadata_checks": carrier_checks,
        "outputs": [ref(path) for path, _ in staged], "mechanical_audit_passed": combined["passed"],
        "semantic_support": "not_evaluated", "raw_answer_bytes_unchanged": True,
        "new_model_calls": 0, "new_API_calls": 0, "gpu_calls": 0, "semantic_decisions_created": False}
    with receipt_path.open("xb") as stream:
        stream.write(encoded(receipt))
    print(json.dumps({"status": STATUS, "mechanical_audit_passed": combined["passed"],
        "observations": len(observations), "receipt": ref(receipt_path), "new_model_calls": 0}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--inputs", type=Path, required=True)
    main(parser.parse_args().inputs.resolve())
