"""Run bounded P6 diagnostic commands with the existing GPU lease/owned process."""
import argparse
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.runner.gpu_lease import GpuLeaseManager, PreGenesisGpuLeaseHolder
from true_rl_agent.runner.process_handle import run_owned_process
from true_rl_agent.runner.resource_shadow import discover_nvidia_gpu_inventory
p = argparse.ArgumentParser()
p.add_argument('--device', required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--timeout', type=int, required=True)
p.add_argument('command', nargs=argparse.REMAINDER)
a = p.parse_args()
command = a.command[1:] if a.command[0] == '--' else a.command
a.output.mkdir(parents=True, exist_ok=False)
identity = canonical_json_hash({'purpose': 'P6 diagnostic', 'command': command})
holder = PreGenesisGpuLeaseHolder(session_id='new-world-p6-diagnostic', genesis_identity_hash=identity,
    session_contract_hash=identity, lane_id=a.device, invocation_segment_id=a.output.name, logical_job_id=a.output.name)
manager = GpuLeaseManager(lease_root=Path('/tmp/true_rl_agent_gpu_leases'),
    devices={item.execution_device: item.identity for item in discover_nvidia_gpu_inventory()})
report = {'purpose': 'P6 diagnostic', 'argv': command, 'started_at_unix': time.time()}
with manager.acquire(a.device, holder=holder) as lease:
    report['lease'] = dict(lease.receipt)
    (a.output / 'process.json').write_text(json.dumps(report, indent=2) + '\n')
    try:
        result = run_owned_process(command, cwd=ROOT, log_dir=a.output, timeout_s=a.timeout)
        report.update(returncode=result.returncode, child_pid=result.pid)
    except Exception as exc:
        report.update(error=str(exc), returncode=1)
    report['ended_at_unix'] = time.time()
    (a.output / 'process.json').write_text(json.dumps(report, indent=2) + '\n')
sys.exit(report['returncode'])
