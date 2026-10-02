"""One frozen P3 Stair camera-development replay; run only after source staging."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path("/data/yuyangliu/AGENT/feedback_p3_20260924")
JOB = ROOT / "stair_v3_reference_r5_on/public_dev_job.json"
ENV_INDEX = 282
EYE = (2.5, -2.5, 1.4)
LOOKAT = (0.4, 0.0, -0.15)
RESOLUTION = (1280, 720)
PATCH_SHA = "933e76a4b6f6112c74a87b1e06a1f9c76dc0dcbf9f3de1f146fb5d9881bb4919"
OLD_CAMERA_SHA = "168e156ac541d1ea9acf3d85a98babf75cef86ec74ed60126dd2ff109ea58e6f"
OLD_EVAL_SHA = "904142dd66a2e350d253295f3c6bd7514d52b8237a5c5b615de09759df1bf398"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_arg(argv: list[str], flag: str, value: str) -> None:
    argv[argv.index(flag) + 1] = value


def source_hashes(root: Path) -> dict[str, str]:
    return {path.relative_to(root).as_posix(): sha(path)
            for directory in ("src/true_rl_agent", "scripts", "configs")
            for path in (root / directory).rglob("*")
            if path.is_file() and path.suffix != ".pyc" and "__pycache__" not in path.parts}


def media_complete(output: Path) -> bool:
    try:
        metadata = json.loads((output / "video/recordings.json").read_text())
        rows = metadata["recordings"]
        if len(rows) != 3 or not (output / "evaluation/public_dev/summary.json").is_file():
            return False
        return all(row["frame_count"] > 0 and row["sha256"] == sha(output / "video" / row["file"])
                   for row in rows)
    except (OSError, KeyError, TypeError, ValueError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera-eye", nargs=3, type=float, default=EYE)
    parser.add_argument("--camera-lookat", nargs=3, type=float, default=LOOKAT)
    parser.add_argument("--timeout-s", type=int, default=900)
    parser.add_argument("--run", action="store_true", help="actually launch one GPU replay")
    args = parser.parse_args()
    if not 10 < args.timeout_s <= 900:
        raise ValueError("GPU lease hard timeout must be in 11..900 seconds")
    source, output = args.source.resolve(strict=True), args.output.absolute()
    old_source = ROOT / "source_r5"
    if source == old_source.resolve():
        raise ValueError("staging must not modify the old source_r5")
    if sha(old_source / "src/true_rl_agent/runner/native_video.py") != OLD_CAMERA_SHA:
        raise ValueError("old source_r5 camera module has changed")
    if sha(source / "src/true_rl_agent/runner/native_video.py") != PATCH_SHA:
        raise ValueError("staged camera module differs from reviewed patch")
    if sha(source / "scripts/isaaclab_stair_rsl_rl_eval.py") != OLD_EVAL_SHA:
        raise ValueError("staged Stair eval differs from old P3 video source")
    original_hashes, staged_hashes = source_hashes(old_source), source_hashes(source)
    camera_path = "src/true_rl_agent/runner/native_video.py"
    if set(original_hashes) != set(staged_hashes) or {
        path for path in original_hashes if original_hashes[path] != staged_hashes[path]
    } != {camera_path}:
        raise ValueError("staged source changes more than the native camera module")
    if output.exists():
        raise ValueError("output directory already exists")
    original = list(json.loads(JOB.read_text())["metadata"]["cli_argv"])
    if original[1] != "scripts/isaaclab_stair_rsl_rl_eval.py":
        raise ValueError("old eval entry is not the expected relative Stair script")
    expected_checkpoint = original[original.index("--checkpoint-sha256") + 1]
    checkpoint = Path(original[original.index("--checkpoint") + 1])
    if "sha256:" + sha(checkpoint) != expected_checkpoint:
        raise ValueError("checkpoint bytes differ from frozen old public-dev job")
    env = dict(os.environ)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env["PYTHONPATH"] = str(source / "src") + os.pathsep + env.get("PYTHONPATH", "")
    isaac_python = original[0]
    manifest_cmd = [isaac_python, "-c",
        "from pathlib import Path; import true_rl_agent.runner.native_video as nv; "
        f"assert Path(nv.__file__).resolve() == Path({str(source / 'src/true_rl_agent/runner/native_video.py')!r}); "
        "from true_rl_agent.backends.isaac_lab_execution_environment import "
        "build_current_isaac_lab_execution_environment_manifest as build; print(build().manifest_hash)"]
    manifest_hash = subprocess.check_output(manifest_cmd, cwd=source, env=env, text=True, timeout=120).strip()
    if len(manifest_hash) != 64:
        raise ValueError("invalid staged execution-environment manifest hash")
    argv = []
    skip = False
    for value in original:
        if skip:
            skip = False
        elif value in ("--numeric-evidence-dir", "--numeric-evidence-max-bytes"):
            skip = True
        else:
            argv.append(value)
    replace_arg(argv, "--summary-path", str(output / "evaluation/public_dev/summary.json"))
    replace_arg(argv, "--backend-execution-environment-manifest-hash", manifest_hash)
    argv[1] = str(source / original[1])
    argv += ["--video-dir", str(output / "video"), "--video-env-index", str(ENV_INDEX),
             "--video-camera-eye", *(str(x) for x in args.camera_eye),
             "--video-camera-lookat", *(str(x) for x in args.camera_lookat),
             "--video-resolution", *(str(x) for x in RESOLUTION)]
    print(json.dumps({"status": "READY_TO_RUN" if not args.run else "RUNNING", "source": str(source),
        "output": str(output), "checkpoint_sha256": expected_checkpoint,
        "staged_environment_manifest_hash": manifest_hash, "env_index": ENV_INDEX,
        "camera_eye": args.camera_eye, "camera_lookat": args.camera_lookat, "resolution": RESOLUTION,
        "lease_timeout_s": args.timeout_s, "two_card_budget_gpu_capacity_hours": 2 * args.timeout_s / 3600,
        "public_seeds": original[original.index("--public-seeds") + 1],
        "eval_script": argv[1], "eval_script_sha256": sha(Path(argv[1]))}, indent=2))
    if not args.run:
        return 0
    output.mkdir(parents=True)
    (output / "argv.json").write_text(json.dumps(argv, indent=2) + "\n")
    sys.path.insert(0, str(source / "src"))
    from true_rl_agent.runner.gpu_lease import GpuLeaseManager, PreGenesisGpuLeaseHolder
    from true_rl_agent.runner.resource_shadow import discover_nvidia_gpu_inventory

    device = original[original.index("--device") + 1]
    manager = GpuLeaseManager(
        lease_root=Path("/tmp/true_rl_agent_gpu_leases"),
        devices={item.execution_device: item.identity for item in discover_nvidia_gpu_inventory()},
    )
    identity = hashlib.sha256((expected_checkpoint + manifest_hash + str(output)).encode()).hexdigest()
    holder = PreGenesisGpuLeaseHolder(
        session_id="stair-camera-dev-slot282", genesis_identity_hash=identity,
        session_contract_hash=identity, lane_id=device, invocation_segment_id=identity[:16],
        logical_job_id="stair-camera-dev-slot282",
    )
    lease_started = None
    lease_receipt = None
    exit_code, status = -1, "FAILED_BEFORE_LAUNCH"
    try:
        with manager.acquire(device, holder=holder) as lease:
            lease_started = time.monotonic()
            lease_receipt = dict(lease.receipt)
            with (output / "run.log").open("w") as log:
                process = subprocess.Popen(argv, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                try:
                    exit_code = process.wait(timeout=max(1.0, args.timeout_s - 10.0
                                                           - (time.monotonic() - lease_started)))
                    status = ("FINISHED" if exit_code == 0 and media_complete(output)
                              else "INCOMPLETE_MEDIA" if exit_code == 0 else "FAILED")
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    exit_code, status = -1, "TIMEOUT"
    finally:
        wall_seconds = None if lease_started is None else time.monotonic() - lease_started
        (output / "run_status.json").write_text(json.dumps({"status": status, "exit_code": exit_code,
            "gpu_lease_wall_seconds": wall_seconds, "gpu_lease_receipt": lease_receipt,
            "gpu_capacity_hours_two_cards": None if wall_seconds is None else 2 * wall_seconds / 3600,
            "lease_timeout_s": args.timeout_s,
            "two_card_budget_gpu_capacity_hours": 2 * args.timeout_s / 3600}, indent=2) + "\n")
    return 0 if status == "FINISHED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
