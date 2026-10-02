"""Push medium env19 camera-development replay, based on P3R record_push_probe.py."""
from dataclasses import replace
from pathlib import Path
import hashlib
import sys

BASE = Path("/data/yuyangliu/AGENT/feedback_p3_20260924")
SOURCE = BASE / "source_push_camera_r1"
OLD_SOURCE = BASE / "source_r3"
CAMERA = "src/true_rl_agent/runner/native_video.py"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

if sha(OLD_SOURCE / CAMERA) != "168e156ac541d1ea9acf3d85a98babf75cef86ec74ed60126dd2ff109ea58e6f":
    raise ValueError("old Push renderer source changed")
if sha(SOURCE / CAMERA) != "933e76a4b6f6112c74a87b1e06a1f9c76dc0dcbf9f3de1f146fb5d9881bb4919":
    raise ValueError("staged Push renderer differs from camera candidate")
if sha(SOURCE / "scripts/push_numeric_replay.py") != "c899b06f8f702599b0c72bbaf5b1f692d1afa596a391fff1295f03c121218be9":
    raise ValueError("staged Push replay driver differs from old source")
sys.path[:0] = [str(SOURCE / "src"), str(SOURCE / "scripts")]
import push_numeric_replay as replay

slot, device, output = int(sys.argv[1]), sys.argv[2], Path(sys.argv[3]).resolve()
if slot != 19 or output.exists():
    raise ValueError("one fresh medium env19 development output is required")
prepare = replay.prepare

def with_video(args):
    composition, phase, report = prepare(args)
    metadata = dict(phase.metadata)
    metadata["cli_argv"] = [*metadata["cli_argv"], "--video-dir", str(output / "video"),
                            "--video-env-index", "19", "--video-camera-eye", "3.0", "-3.0", "1.8",
                            "--video-camera-lookat", "0.3", "0.0", "-0.2",
                            "--video-resolution", "1280", "720"]
    metadata["cli_command"] = " ".join(metadata["cli_argv"])
    return composition, replace(phase, metadata=metadata), {**report, "video_env_index": slot,
        "purpose": "P3NEW Push old-medium camera development"}

replay.prepare = with_video
qualified = Path("/data/yuyangliu/AGENT/true_RL_Agent/outputs/p6_g1_stationary_v3_qualification_20260912")
run_id = "p6-g1-stationary-v3-reference-20260912"
raise SystemExit(replay.main([
    "--cell", "push.g1_37dof.recovery_v2", "--source-manifest", str(qualified / "resolved_cell_manifest.json"),
    "--checkpoint-summary", str(qualified / "runner" / run_id / "train/summary.json"),
    "--checkpoint", str(qualified / "runner" / run_id / "checkpoints/best.pt"),
    "--device", device, "--output", str(output), "--mode", "run", "--capture", "off",
    "--timeout-s", "600", "--eval-base-seed", "1000000000",
    "--isaac-python", "/data/yuyangliu/envs/isaaclab6_validation/bin/python",
]))
