"""Run existing CPU-only pair checkers for the P2 audit acceptance matrix."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, required=True)
args = parser.parse_args()
root = args.root
tools = root / 'tools'
results = []
for task, name in [('stair', 'stair_parent'), ('stair', 'stair_g1'),
                   ('push', 'push_g1_v3'), ('push', 'push_go2_v2'),
                   *[('training', f'train_{t}_{r}') for t in ('stair', 'push') for r in ('go2', 'g1')]]:
    output = root / (name + '_verification.json')
    source = root / ('source_training' if task == 'training' else 'source')
    if output.exists():
        results.append({'case': name, 'returncode': 0 if json.loads(output.read_text())['status'] == 'PASS' else 1,
                        'report': str(output), 'status': json.loads(output.read_text())['status']})
        continue
    command = [sys.executable, str(tools / f'verify_{task}_pair.py'), '--project-root', str(source)]
    command += ['--on-dir' if task == 'stair' else '--on', str(root / (name+'_on')),
                '--off-dir' if task == 'stair' else '--off', str(root / (name+'_off')), '--output', str(output)]
    result = subprocess.run(command)
    results.append({'case': name, 'returncode': result.returncode, 'report': str(output),
                    'status': json.loads(output.read_text())['status'] if output.exists() else 'NO_REPORT'})
    print(json.dumps(results[-1]), flush=True)
status = 'PASS' if all(r['returncode'] == 0 and r['status'] == 'PASS' for r in results) else 'FAIL'
(root/'verification_index.json').write_text(json.dumps({'status': status, 'results': results}, indent=2)+'\n')
raise SystemExit(0 if status == 'PASS' else 1)
