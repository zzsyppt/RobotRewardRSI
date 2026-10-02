"""CPU-only hash/equality closure; does not evaluate observer prose or assemble."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.util
import json

ROOT = Path(__file__).resolve().parent
DOWNLOAD = ROOT / "push_sol_audit_r1"
PACKETS = ROOT / "push_engineering_repair/push"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def ref(path):
    data = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


def check(path, expected):
    actual = ref(path)
    assert actual["sha256"] == expected["sha256"], str(path)
    assert actual["size_bytes"] == expected.get("size_bytes", actual["size_bytes"]), str(path)
    return actual


def save(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main():
    result = read(ROOT / "push_sol_audit_remote_execution_result.json")
    assert result["returncode"] == 0
    exit_ref = check(DOWNLOAD / "push_sol_audit_r1.exit.json", {"sha256": result["exit_record_sha256"]})
    exit_record = read(Path(exit_ref["path"]))
    assert exit_record["returncode"] == 0 and exit_record["assemble_executed"] is False
    log_refs = {name: check(DOWNLOAD / f"push_sol_audit_r1.{name}.txt",
                           {"sha256": result[f"{name}_sha256"]}) for name in ("stdout", "stderr")}
    assert log_refs["stderr"]["size_bytes"] == 0
    audit_stdout = json.loads(result["audit_stdout"])
    receipt_ref = check(DOWNLOAD / "sol_engineering_audit_receipt.json", audit_stdout["receipt"])
    receipt = read(Path(receipt_ref["path"]))
    assert receipt["new_model_calls"] == receipt["new_API_calls"] == receipt["gpu_calls"] == 0
    assert receipt["raw_answer_bytes_unchanged"] is True
    ready = read(ROOT / "push_sol_audit_inputs.ready.json")
    check(ROOT / "audit_push_sol_engineering.py", {"sha256": exit_record["script_sha256"]})
    check(ROOT / "push_sol_audit_inputs.ready.json", {"sha256": exit_record["ready_inputs_sha256"]})
    audits = []
    output_refs = []
    original_refs = []
    for index, alias in enumerate(("e1", "e2")):
        audit_path = DOWNLOAD / alias / "mechanical_audit.json"
        output_refs.append(check(audit_path, receipt["outputs"][index]))
        audit = read(audit_path)
        audits.append(audit)
        packet = ready["packets"][index]
        assert packet["episode_alias"] == alias
        for name, filename in (("raw_answer", "raw_answer.txt"), ("access_receipt", "access_receipt.json")):
            original_refs.append(check(PACKETS / alias / "sol_response" / filename, packet[name]))
        assert audit["raw_output"] == read(PACKETS / alias / "sol_response/raw_answer.txt")
    combined_path = DOWNLOAD / "combined_audit.json"
    output_refs.append(check(combined_path, receipt["outputs"][2]))
    combined = read(combined_path)
    specification = importlib.util.spec_from_file_location("assembly_interface_verification", ROOT / "assemble_delivery.py")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    module.verify_combined(combined, audits, receipt["outputs"][:2])
    binding = read(ROOT / "push_visual_review_binding_cpu.json")
    check(combined_path, binding["combined_audit"])
    assert combined["passed"] is True
    assert len(combined["observations"]) == 6
    # Make the already verified results available at the task's actual packet paths.
    copies = [(DOWNLOAD / alias / "mechanical_audit.json", PACKETS / alias / "sol_response/mechanical_audit.json")
              for alias in ("e1", "e2")]
    copies += [(combined_path, PACKETS / "combined_audit.json"),
               (DOWNLOAD / "sol_engineering_audit_receipt.json", PACKETS / "sol_engineering_audit_receipt.json")]
    for original, target in copies:
        assert not target.exists(), str(target)
    for original, target in copies:
        with target.open("xb") as stream:
            stream.write(original.read_bytes())
        check(target, ref(original))
    # Preserve the original draft and leave semantic review/cutoff inputs pending.
    delivery = read(ROOT / "engineering_delivery_inputs.draft.json")
    delivery["tasks"]["push"]["audits"] = receipt["outputs"][:2]
    delivery["tasks"]["push"]["combined_audit"] = receipt["outputs"][2]
    delivery["audit_archival_receipt"] = audit_stdout["receipt"]
    delivery["pending_root_review_and_cutoff"] = True
    save(ROOT / "engineering_delivery_inputs.audit_bound.json", delivery)
    closure = {
        "status": "DEVELOPMENT_REPAIR_NOT_ORIGINAL_N5_CONFIRMATION", "original_n5_status": "INCONCLUSIVE",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(), "remote_exit": exit_ref,
        "invocation": ref(DOWNLOAD / "push_sol_audit_r1.invocation.json"), "logs": log_refs,
        "receipt": receipt_ref, "downloaded_outputs": output_refs, "original_SOL_files_unchanged": original_refs,
        "packet_copies": [ref(target) for _, target in copies],
        "aggregate_interface_checker": ref(ROOT / "assemble_delivery.py"),
        "checks": {"remote_exit_zero": True, "download_hashes_match_remote": True,
                   "original_raw_and_access_hashes_unchanged": True, "raw_output_exactly_matches_original_json": True,
                   "existing_assemble_verify_combined_passed": True, "mechanical_audits_passed": [a["passed"] for a in audits],
                   "all_six_observations_retained_in_e1_e2_order": True, "new_model_calls": 0,
                   "new_API_calls": 0, "new_GPU_jobs": 0, "semantic_review_performed": False, "assemble_executed": False},
        "visual_review_binding_result": binding,
        "delivery_inputs": ref(ROOT / "engineering_delivery_inputs.audit_bound.json"),
        "pending": ["root complete-observation semantic reviews and their real SHA references",
                    "root actual UTC cutoff values, receipt SHA and JSON pointers for both tasks",
                    "explicit root authorization before assemble_delivery execution"],
    }
    save(DOWNLOAD / "download_verification.json", closure)
    print(json.dumps({"status": closure["status"], "mechanical_passed": combined["passed"],
                      "observations": len(combined["observations"]), "combined_sha256": output_refs[-1]["sha256"],
                      "visual_review_binding": binding["visual_review_binding"],
                      "closure": ref(DOWNLOAD / "download_verification.json")}, indent=2))


if __name__ == "__main__":
    main()
