"""Own a P5 diagnostic process and GPU lease using the production primitives."""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.runner.gpu_lease import GpuLeaseManager, PreGenesisGpuLeaseHolder
from true_rl_agent.runner.process_handle import run_owned_process
from true_rl_agent.runner.resource_shadow import discover_nvidia_gpu_inventory

parser = argparse.ArgumentParser()
parser.add_argument("--robot", required=True)
parser.add_argument("--effect", required=True)
parser.add_argument("--device", required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
inventory = discover_nvidia_gpu_inventory()
gpu = next(item for item in inventory if item.execution_device == args.device)
argv = [sys.executable, str(Path(__file__).with_name("runtime_probe.py")), "--robot", args.robot,
        "--effect", args.effect, "--device", "cuda:0", "--output", str(args.output / "run")]
identity = canonical_json_hash({"purpose": "P5 runtime diagnostic", "argv": argv})
holder = PreGenesisGpuLeaseHolder(session_id="p5-diagnostic", genesis_identity_hash=identity,
    session_contract_hash=identity, lane_id=args.device, invocation_segment_id=args.output.name,
    logical_job_id=args.output.name)
manager = GpuLeaseManager(lease_root=Path("/tmp/true_rl_agent_gpu_leases"),
    devices={item.execution_device: item.identity for item in inventory})
report = {"purpose": "P5 diagnostic; no formal acceptance", "argv": argv, "started_at": time.time()}
with manager.acquire(args.device, holder=holder) as lease:
    report["lease"] = dict(lease.receipt)
    result = run_owned_process(argv, cwd=ROOT, log_dir=args.output / "process", timeout_s=600,
                              environment_overrides={"CUDA_VISIBLE_DEVICES": gpu.uuid})
    report.update(returncode=result.returncode, pid=result.pid, ended_at=time.time())
(args.output / "launch.json").write_text(json.dumps(report, indent=2) + "\n")
raise SystemExit(result.returncode)
