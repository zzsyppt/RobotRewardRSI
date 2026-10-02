"""Reviewer-only native video of the P3 Push qualified-cohort public-dev job (capture off, one env index)."""
import sys, os, json, subprocess, argparse
from pathlib import Path
SRC = Path("/data/yuyangliu/AGENT/feedback_p3_20260924/source_r3")
sys.path.insert(0, str(SRC)); sys.path.insert(0, str(SRC / "src")); sys.path.insert(0, str(SRC / "scripts"))
import push_numeric_replay as pnr
env_index, device, out = int(sys.argv[1]), sys.argv[2], Path(sys.argv[3]).resolve()
Q = "/data/yuyangliu/AGENT/true_RL_Agent/outputs/p6_g1_stationary_v3_qualification_20260912"
args = pnr.build_parser().parse_args(["--cell", "push.g1_37dof.recovery_v2",
    "--source-manifest", Q + "/resolved_cell_manifest.json",
    "--checkpoint-summary", Q + "/runner/p6-g1-stationary-v3-reference-20260912/train/summary.json",
    "--checkpoint", Q + "/runner/p6-g1-stationary-v3-reference-20260912/checkpoints/best.pt",
    "--device", device, "--output", str(out), "--mode", "run", "--capture", "off",
    "--timeout-s", "1800", "--eval-base-seed", "1000000000",
    "--isaac-python", "/data/yuyangliu/envs/isaaclab6_validation/bin/python"])
out.mkdir(parents=True, exist_ok=False)
composition, phase, report = pnr.prepare(args)
argv = list(phase.metadata["cli_argv"]) + ["--video-dir", str(out / "video"), "--video-env-index", str(env_index)]
(out / "argv.json").write_text(json.dumps(argv, indent=1))
(out / "prepare_report.json").write_text(json.dumps(report, indent=1, default=str))
env = dict(os.environ); env.pop("CUDA_VISIBLE_DEVICES", None)
sys.exit(subprocess.run(argv, cwd=SRC, env=env).returncode)
