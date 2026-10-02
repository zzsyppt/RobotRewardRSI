"""One source-frozen public-dev replay with video, using the existing runner/lease.

Only fills the missing medium development observation; no training or new author.
The frozen P3 source intentionally reproduces the measured controller/eval cohort.
"""
from dataclasses import replace
from pathlib import Path
import sys

SOURCE = Path("/data/yuyangliu/AGENT/feedback_p3_20260924/source_r3")
sys.path[:0] = [str(SOURCE / "src"), str(SOURCE / "scripts")]
import push_numeric_replay as replay

slot, device, output = int(sys.argv[1]), sys.argv[2], Path(sys.argv[3]).resolve()
prepare = replay.prepare


def with_video(args):
    composition, phase, report = prepare(args)
    metadata = dict(phase.metadata)
    metadata["cli_argv"] = [*metadata["cli_argv"], "--video-dir", str(output / "video"),
                            "--video-env-index", str(slot)]
    metadata["cli_command"] = " ".join(metadata["cli_argv"])
    return composition, replace(phase, metadata=metadata), {**report, "video_env_index": slot,
                                                          "purpose": "P3R development visual observation"}


replay.prepare = with_video
qualified = Path("/data/yuyangliu/AGENT/true_RL_Agent/outputs/p6_g1_stationary_v3_qualification_20260912")
run_id = "p6-g1-stationary-v3-reference-20260912"
raise SystemExit(replay.main([
    "--cell", "push.g1_37dof.recovery_v2", "--source-manifest", str(qualified / "resolved_cell_manifest.json"),
    "--checkpoint-summary", str(qualified / "runner" / run_id / "train/summary.json"),
    "--checkpoint", str(qualified / "runner" / run_id / "checkpoints/best.pt"),
    "--device", device, "--output", str(output), "--mode", "run", "--capture", "off",
    "--timeout-s", "1200", "--eval-base-seed", "1000000000",
    "--isaac-python", "/data/yuyangliu/envs/isaaclab6_validation/bin/python",
]))
