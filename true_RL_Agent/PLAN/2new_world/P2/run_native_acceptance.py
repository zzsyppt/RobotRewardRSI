"""One-shot P2 acceptance sequence; historical anchors remain diagnostic only."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PHASE = Path(__file__).resolve().parent
ROOT = PHASE.parents[2]
ANCHORS = {
    "go2": "/data/yuyangliu/AGENT/true_RL_Agent/outputs/p5_go2_derived_reference_ef6826e/runner/p5-go2-derived-reference-ef6826e/checkpoints/best.pt",
    "g1": "/data/yuyangliu/AGENT/true_RL_Agent/outputs/phase4_unified_validation_5a30129_20260829/runner/phase4-unified-validation-5a30129-20260829/checkpoints/best.pt",
}
CASES = (
    ("go2_native_frames_c0_02", "go2", ["--video-env-index", "0"]),
    ("go2_native_frames_c1", "go2", ["--video-env-index", "9"]),
    ("go2_native_frames_c2", "go2", ["--video-env-index", "63"]),
    ("g1_native_frames", "g1", ["--video-env-index", "0"]),
    ("go2_video_off_frames", "go2", ["--no-video"]),
    ("g1_video_off_frames", "g1", ["--no-video"]),
)

parser = argparse.ArgumentParser()
parser.add_argument("--run-prefix", default="")
args = parser.parse_args()
records = []
for name, robot, extra in CASES:
    name = args.run_prefix + name
    command = [
        sys.executable, str(PHASE / "native_eval_job_probe.py"),
        "--robot", robot, "--device", "cuda:1", "--checkpoint", ANCHORS[robot],
        "--output", str(PHASE / "results" / name), *extra,
    ]
    record = {"case": name, "argv": command, "started_at_unix": time.time()}
    print(json.dumps(record), flush=True)
    with (PHASE / (name + ".log")).open("w") as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    record.update(returncode=result.returncode, ended_at_unix=time.time())
    records.append(record)
    (PHASE / "results" / (args.run_prefix + "native_frames_sequence.json")).write_text(
        json.dumps(records, indent=2) + "\n"
    )
    print(json.dumps(record), flush=True)
    if result.returncode:
        sys.exit(result.returncode)
