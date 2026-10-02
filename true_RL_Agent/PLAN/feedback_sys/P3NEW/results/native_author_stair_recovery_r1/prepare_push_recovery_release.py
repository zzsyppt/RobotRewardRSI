"""One-off preserved-input Push continuation card and CPU check; never send."""
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
BASE = Path("/data/yuyangliu/AGENT/feedback_p3new_20260929")
SOURCE = BASE / "source_observer_v3_r8"
HELPERS = BASE / "text_author_smoke_r1/PLAN/feedback_sys/P3NEW"
OUT = HELPERS / "results/native_author_smoke_r3"
RECOVERY = HELPERS / "results/native_author_stair_recovery_r1"
sys.path.insert(0, str(SOURCE / "src"))
sys.path.append(str(HELPERS))
spec = importlib.util.spec_from_file_location("sender_after_recovery", HELPERS / "send_native_author_smoke_after_recovery.py")
sender = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sender)

if __name__ == "__main__":
    original_hashes = sender.D.read(RECOVERY / "append_original_hashes.json")
    prior_path = OUT / "stage.released.json"
    stage = deepcopy(sender.D.read(prior_path))
    recovery_path = RECOVERY / "recovery.json"
    recovery = sender.D.read(recovery_path)
    stage.update(previous_release_ref=sender.D.ref(prior_path),
                 prior_case_recovery=sender.D.ref(recovery_path),
                 execution_script=sender.D.ref(Path(sender.__file__)))
    for reference in (stage["execution_script"], stage["previous_release_ref"], stage["prior_case_recovery"],
                      recovery["original_terminal_ref"], recovery["transport_receipt_ref"],
                      recovery["appended_attempt_ref"]):
        if reference not in stage["frozen_inputs"]:
            stage["frozen_inputs"].append(reference)
    path = OUT / "stage.runtime_recovered.json"
    with path.open("xb") as stream:
        stream.write((json.dumps(stage, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode())
    sender.validate(stage, path, OUT, released=True)
    result = sender.run(path, OUT, "push-native-dev", check_only=True)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in original_hashes.items())
    result.update(stage_ref=sender.D.ref(path), driver_ref=stage["execution_script"],
                  original_files_unchanged=len(original_hashes), remaining_actual_calls=1)
    sender.D.write(RECOVERY / "push_recovery_check.json", result)
    print(json.dumps({k: result[k] for k in ("status", "api_calls", "stage_ref", "driver_ref", "original_files_unchanged")}))
