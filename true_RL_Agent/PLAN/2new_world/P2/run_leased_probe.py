"""Run a P2 acceptance probe with production GPU lease/process ownership.

The receipt is diagnostic and is never a formal session Genesis or ledger entry.
"""

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
parser.add_argument("--device", required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--timeout", type=int, default=14400)
parser.add_argument("command", nargs=argparse.REMAINDER)
args = parser.parse_args()
command = args.command[1:] if args.command[0] == "--" else args.command
args.output.mkdir(parents=True, exist_ok=True)
inventory = discover_nvidia_gpu_inventory()
devices = {item.execution_device: item.identity for item in inventory}
identity = canonical_json_hash(
    {"purpose": "P2 acceptance diagnostic", "command": command}
)
holder = PreGenesisGpuLeaseHolder(
    session_id="new-world-p2-acceptance",
    genesis_identity_hash=identity,
    session_contract_hash=identity,
    lane_id=args.device,
    invocation_segment_id=args.output.name,
    logical_job_id=args.output.name,
)
manager = GpuLeaseManager(
    lease_root=Path("/tmp/true_rl_agent_gpu_leases"), devices=devices
)
report = {
    "purpose": "P2 acceptance diagnostic",
    "argv": command,
    "started_at_unix": time.time(),
}
with manager.acquire(args.device, holder=holder) as lease:
    report["lease"] = dict(lease.receipt)
    (args.output / "process.json").write_text(json.dumps(report, indent=2) + "\n")
    try:
        result = run_owned_process(
            command, cwd=ROOT, log_dir=args.output, timeout_s=args.timeout
        )
        report.update(returncode=result.returncode, child_pid=result.pid)
    except Exception as exc:
        report.update(error=str(exc), returncode=1)
    report["ended_at_unix"] = time.time()
    (args.output / "process.json").write_text(json.dumps(report, indent=2) + "\n")
sys.exit(report["returncode"])
