"""Bind the actual root review originals and UTC receipts; no semantic decisions."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent
BASE = "/data/yuyangliu/AGENT/feedback_p3new_20260929"
CONTROL = BASE + "/n5_execution_20260930"
CAPTURE = BASE + "/n5_confirmation_20260930"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ref(path, remote):
    return {"path": remote, "sha256": digest(path), "size_bytes": path.stat().st_size}


def save(path, data):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


ready = json.loads((ROOT / "engineering_delivery_inputs.audit_bound.json").read_bytes())
files = []
for task, directory, remote in (
    ("stair", ROOT / "observer_prepared_snapshot/stair", CAPTURE + "/observers/stair"),
    ("push", ROOT / "push_engineering_repair/push", CAPTURE + "/engineering_repair/observers/push"),
):
    review_path = directory / "root_module_review.json"
    receipt_path = directory / "root_module_review_receipt.json"
    receipt = json.loads(receipt_path.read_bytes())
    review = ref(review_path, remote + "/root_module_review.json")
    seal = ref(receipt_path, remote + "/root_module_review_receipt.json")
    assert receipt["review_sha256"] == review["sha256"]
    assert receipt["combined_audit_sha256"] == ready["tasks"][task]["combined_audit"]["sha256"]
    if task == "push":
        assert review["sha256"] == "e94755505efe574e385aea19b6e811b879bdfabd8475798d704b0ae055f38a92"
        assert receipt["recorded_at_utc"] == "2026-09-30T05:39:21.535673+00:00"
        files.extend([{**review, "local": str(review_path)}, {**seal, "local": str(receipt_path)}])
    ready["tasks"][task]["review"] = review
    ready["tasks"][task]["as_of_cutoff"] = receipt["recorded_at_utc"]
    ready["tasks"][task]["as_of_cutoff_ref"] = {**seal, "pointer": "/recorded_at_utc"}
ready.pop("draft_not_executable", None)
ready.pop("pending_root_review_and_cutoff", None)
ready["execution_authorized_by_root"] = True
ready["execution_scope"] = "DEVELOPMENT_REPAIR_NOT_ORIGINAL_N5_CONFIRMATION"
ready["original_n5_status"] = "INCONCLUSIVE"
ready["output_directory"] = CAPTURE + "/engineering_repair/delivery_offline_r1"
ready_path = ROOT / "engineering_delivery_inputs.ready.json"
save(ready_path, ready)
for path in (ready_path, ROOT / "assemble_delivery.py"):
    files.append({**ref(path, CONTROL + "/" + path.name), "local": str(path)})
manifest = {"status": ready["status"], "original_n5_status": "INCONCLUSIVE",
            "files": files, "output_directory": ready["output_directory"],
            "source_root": BASE + "/n5_selection_repair_source_r1",
            "material_root": BASE + "/text_author_smoke_r1",
            "automatic_retry": False, "model_calls": 0, "GPU_jobs": 0}
save(ROOT / "engineering_delivery_execution_manifest.json", manifest)
print(json.dumps({"ready": ref(ready_path, str(ready_path)),
                  "tasks": {task: {key: row[key] for key in ("review", "as_of_cutoff", "as_of_cutoff_ref")}
                            for task, row in ready["tasks"].items()}}, indent=2))
