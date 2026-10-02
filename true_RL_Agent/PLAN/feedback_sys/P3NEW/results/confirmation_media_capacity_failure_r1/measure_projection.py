"""Record the known CPU capacity failure; keep the real validator in the path."""
import hashlib
import json
from copy import deepcopy
from pathlib import Path
import sys
from unittest.mock import patch

BASE = Path("/data/yuyangliu/AGENT/feedback_p3new_20260929")
SOURCE = BASE / "source_n5_preflight_r1"
CONTROL = BASE / "confirmation_media_control_r1/PLAN/feedback_sys/P3NEW"
OUT = BASE / "confirmation_media_capacity_failure_r1"
sys.path.insert(0, str(SOURCE / "src"))
import true_rl_agent
sys.path.append(str(CONTROL))
import prepare_native_observer as native
sys.path.insert(0, str(SOURCE / "src"))
from true_rl_agent.evaluation import process_report as common
from true_rl_agent.evaluation.visual_observer import audit_observer_output
from true_rl_agent.tasks.stair_climbing.visual_observation import attach_stair_visual_observations
from true_rl_agent.tasks.stair_climbing import process_report as stair


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


OUT.mkdir(exist_ok=False)
selection = BASE / "confirmation_media_capacity_r1/media-01/observer_selection.json"
full_path = BASE / "stair_dual_view_dev_r3/evaluation/public_dev/behavior_full.json"
full = json.loads(full_path.read_bytes())
built = native.build(BASE / "stair_dual_view_dev_r3_export", selection)
window, = built["windows"]
refs = [{"step": image.displayed_steps[0], "view_id": image.view_id} for image in window.images]
raw = {"episodes": [{"alias": "e1", "not_visible": [], "observations": [
    {"body": "root", "evidence_refs": refs, "text": "形" * 200, "uncertainty": "限" * 120,
     "spatial_visibility": "visible", "relation": "temporal_change"} for _ in range(3)]}]}
audit = audit_observer_output(raw, built["windows"], numeric_check=built["numeric_check"])
assert audit["passed"] and len(audit["observations"]) == 3
assert all(len(row["image_evidence"]) == 12 for row in audit["observations"])
save("synthetic_raw.json", {"synthetic_capacity_only": True, "raw": raw})
save("mechanical_audit.json", audit)
real_validate = common.validate_public_report


def measure(value, label):
    captured = {}
    def record(projection, *, max_bytes):
        captured.update(projection=deepcopy(projection), max_bytes=max_bytes,
                        report_bytes=common.report_bytes(projection))
        return real_validate(projection, max_bytes=max_bytes)
    augmented = attach_stair_visual_observations(value, audit,
        observer_version="synthetic_capacity_only_not_observer_response")
    try:
        with patch.object(common, "validate_public_report", record):
            stair.project_stair_process_report(augmented)
    except ValueError as error:
        assert str(error) == "process report exceeds its byte budget"
        status = "PRODUCTION_VALIDATOR_REJECTED_BYTE_BUDGET"
    else:
        status = "PRODUCTION_PROJECTION_PASS_CAPACITY_ONLY"
    save(label + "_candidate_projection.json", captured["projection"])
    return {"status": status, "report_bytes": captured["report_bytes"], "limit_bytes": captured["max_bytes"],
        "over_limit_bytes": max(0, captured["report_bytes"] - captured["max_bytes"]),
        "section_bytes": {k: common.report_bytes(v) for k, v in captured["projection"]["report"].items()}}


frozen_result = measure(full, "frozen_r3_sentences")
regenerated = deepcopy(full)
fresh = stair._sentences(full, full["semantic"]["residuals"], full["semantic"]["sampling"])
old = {r["id"]: r for r in full["semantic"]["sentences"]}
changes = [{"id": row["id"], "frozen_text_bytes": len(old[row["id"]]["text"].encode()),
            "current_text_bytes": len(row["text"].encode())} for row in fresh
           if row["id"] in old and row["text"] != old[row["id"]]["text"]]
regenerated["semantic"]["sentences"] = fresh
current_result = measure(regenerated, "current_template_diagnostic_only")
result = {"status": "CAPACITY_FAILURE_RECORDED_NOT_REPAIRED", "api_calls": 0, "gpu_jobs": 0,
    "source_root": str(SOURCE), "full": str(full_path), "full_sha256": hashlib.sha256(full_path.read_bytes()).hexdigest(),
    "source_file_unchanged": full == json.loads(full_path.read_bytes()), "frozen_r3_sentences": frozen_result,
    "current_template_diagnostic_only": current_result, "changed_cached_sentence_texts": changes,
    "limits": "Current-template in-memory comparison is a CPU capacity diagnostic only, not a rewritten historical report or new observer response. Both paths call the unchanged production validator."}
save("failure.json", result)
print(json.dumps(result))
