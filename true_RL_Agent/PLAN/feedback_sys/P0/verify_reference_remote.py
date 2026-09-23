"""Bounded final remote read: reference checkpoint hashes and resource snapshot."""
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

prefix = Path('/data/yuyangliu/isolated/new_world_p2_audit_final/true_RL_Agent/PLAN/2new_world/P2/results/audit3_go2_full_train/runner/audit3_go2_full_train')
paths = [prefix / 'checkpoints/best.pt',
         prefix / 'train/run/attempt-8398d445d3c34dc5af179ee71d20c69e/rsl_rl/model_1999.pt']
expected = '7b582643510cc9ddd9ae6755fc551cfc89f6ea1a06fc7b6837dd3b203da11c4c'
rows = []
for path in paths:
    row = {'path': str(path), 'exists': path.is_file(), 'expected_sha256': expected}
    if path.is_file():
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024**2), b''):
                digest.update(chunk)
        row.update(sha256=digest.hexdigest(), bytes=path.stat().st_size, matches_expected=digest.hexdigest() == expected)
    rows.append(row)
commands = {
    'gpu': ['nvidia-smi', '--query-gpu=index,name,memory.used,utilization.gpu', '--format=csv,noheader'],
    'gpu_processes': ['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory', '--format=csv,noheader'],
    'p0_storage': ['du', '-sb', '/data/yuyangliu/feedback_p0_20260923'],
}
report = {'checked_at': datetime.now(timezone.utc).isoformat(), 'reference': rows, 'resources': {}}
for name, command in commands.items():
    result = subprocess.run(command, capture_output=True, text=True)
    report['resources'][name] = {'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}
print(json.dumps(report))
