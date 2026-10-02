"""One-off source/card freeze for the authorized old-dev body capture."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil
import tarfile

PROJECT = Path(__file__).resolve().parents[6]
CONTROL = Path(__file__).resolve().parent
BASE = "/data/yuyangliu/AGENT/feedback_p3new_20260929"
REMOTE_SOURCE = BASE + "/source_body_grounding_r1"
REMOTE_CONTROL = BASE + "/body_grounding_capture_r1_control"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    old_card_path = PROJECT / "PLAN/feedback_sys/P3NEW/results/multi_env_development_r4/capture_card.json"
    old = json.loads(old_card_path.read_text(encoding="utf-8"))
    source = CONTROL / "source"
    source.mkdir(exist_ok=False)
    helpers = ["PLAN/feedback_sys/P3NEW/" + name for name in (
        "extract_dual_view_evidence.py", "prepare_native_observer.py", "prepare_n5_media_selection.py",
        "verify_multi_env_capture.py", "results/body_grounding_capture_r1/control/verify_capture.py")]
    paths = [p for folder in ("src", "scripts", "configs") for p in (PROJECT / folder).rglob("*")
             if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"]
    paths.extend(PROJECT / p for p in helpers)
    inventory = {}
    for path in sorted(paths):
        relative = path.relative_to(PROJECT)
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        digest = sha(path)
        if sha(target) != digest:
            raise ValueError("source copy mismatch: " + str(relative))
        inventory[relative.as_posix()] = digest
    manifest = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                "purpose": "Sparse old-development Push body/camera capture engineering; not N5; no model requests",
                "files": inventory}
    write(CONTROL / "source_manifest.json", manifest)
    with tarfile.open(CONTROL / "source.tar.gz", "w:gz") as bundle:
        for relative in inventory:
            bundle.add(source / relative, arcname=relative, recursive=False)
    command = [value.replace(old["source_root"], REMOTE_SOURCE) for value in old["commands"]["push"]["run"]]
    def replace_option(flag, values):
        i = command.index(flag)
        j = i + 1
        while j < len(command) and not command[j].startswith("--"):
            j += 1
        command[i + 1:j] = values
    replace_option("--output", [BASE + "/push_body_grounding_dev_r1"])
    replace_option("--video-dir", [BASE + "/push_body_grounding_dev_r1/video"])
    replace_option("--video-snapshot-steps", ["1", "44", "71", "109", "206", "412"])
    i = command.index("--video-secondary-env-index")
    del command[i:i + 2]
    preflight = list(command)
    for flag, value in (("--mode", "preflight"), ("--output", BASE + "/push_body_grounding_dev_r1_preflight"),
                        ("--video-dir", BASE + "/push_body_grounding_dev_r1_preflight/video")):
        preflight[preflight.index(flag) + 1] = value
    card = {"schema_version": "p3new.body_grounding_capture.v1", "frozen_at_utc": manifest["created_at_utc"],
            "scope": "old_development_sparse_engineering_not_N5", "source_root": REMOTE_SOURCE,
            "source_manifest_sha256": sha(CONTROL / "source_manifest.json"),
            "source_bundle_sha256": sha(CONTROL / "source.tar.gz"), "source_file_count": len(inventory),
            "baseline_capture_card_sha256": sha(old_card_path),
            "baseline_root": BASE + "/push_multi_env_dev_r4", "baseline_source": old["source_root"],
            "checkpoint_sha256": old["checkpoint_sha256"]["push"],
            "commands": {"push": {"preflight": preflight, "run": command}},
            "capture_profile": {"env_index": 19, "seed_runs": 3, "eval_base_seed": 1000000000,
                "resolution": [1920, 1080], "views_per_env": 2, "snapshot_steps": [1, 44, 71, 109, 206, 412],
                "same_r4_camera_offsets": True, "raw_rgb": True, "no_resize_or_enhancement": True},
            "review_body_names": ["pelvis", "left_hip_yaw_link", "left_knee_link", "left_ankle_pitch_link", "left_palm_link",
                                  "right_hip_yaw_link", "right_knee_link", "right_ankle_pitch_link", "right_palm_link"],
            "machine_acceptance": ["completed production public_dev runner with three original seeds",
                "all cohort first-episode numeric fields retain dtype, valid bytes and terminal records vs r4",
                "selected-state clocks, root/joint/body poses unchanged across camera-only renders",
                "real RGB render-product camera parameters and body name order are captured",
                "PNG hashes, dimensions and sparse selected steps agree with same-run manifests",
                "root/leg/palm-link projection landing and camera/image sync require independent image review"],
            "observer_scope": "Sparse engineering only; do not relax or invoke full interval-1 observer preparation",
            "numeric_comparison": "reuse verified compare_run from verify_multi_env_capture.py; compare every valid field and all terminals",
            "resource_policy": {"existing_runner_gpu_lease": True, "check_live_gpu_and_process_state_before_run": True,
                                "stop_other_processes": False, "timeout_s": 7200, "attempts": 1},
            "provider_requests": 0, "training_jobs": 0, "approval_state": "awaiting_root_execution_signal",
            "cpu_evidence": {"root_reported_tests": 133, "root_reported_seconds": 7.446,
                "root_reported_command": ".venv/Scripts/python.exe -m unittest tests.unit.test_native_video tests.unit.test_body_projection tests.unit.test_push_body_geometry tests.unit.test_observer_measurement_context tests.unit.test_task_pose_facts tests.unit.test_visual_review tests.unit.test_visual_observer tests.unit.test_task_visual_observation tests.unit.test_process_report tests.unit.test_stair_process_report tests.unit.test_push_process_report",
                "root_reported_result": "OK", "projection_scalar_fix_tests": "5 tests OK, 0.003s",
                "exact_numeric_compare_self_tests": "3 tests OK, 0.001s",
                "sender_fake_cpu_receipt": "PLAN/feedback_sys/P3NEW/results/body_grounding_sender_cpu_r1.json",
                "sender_fake_cpu_receipt_sha256": sha(PROJECT / "PLAN/feedback_sys/P3NEW/results/body_grounding_sender_cpu_r1.json")}}
    write(CONTROL / "capture_card.json", card)
    old_launcher = old_card_path.with_name("launch_capture.py")
    shutil.copyfile(old_launcher, CONTROL / "launch_capture.py")
    for relative, digest in inventory.items():
        if sha(PROJECT / relative) != digest:
            raise ValueError("workspace source changed during freeze: " + relative)
    receipt = {"source_manifest_sha256": sha(CONTROL / "source_manifest.json"),
               "source_bundle_sha256": sha(CONTROL / "source.tar.gz"), "capture_card_sha256": sha(CONTROL / "capture_card.json"),
               "launcher_sha256": sha(CONTROL / "launch_capture.py"), "files": len(inventory),
               "all_workspace_hashes_rechecked": True, "provider_requests": 0, "gpu_jobs": 0,
               "remote_source": REMOTE_SOURCE, "remote_control": REMOTE_CONTROL,
               "run_command": [command[0], REMOTE_CONTROL + "/launch_capture.py", "push", "run", sha(CONTROL / "capture_card.json")]}
    write(CONTROL / "local_freeze_receipt.json", receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
