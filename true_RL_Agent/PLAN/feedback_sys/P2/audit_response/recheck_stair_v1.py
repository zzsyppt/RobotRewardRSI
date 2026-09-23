"""Read original v1 Stair archives with the audit-revised production reader."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--original-root', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.source / 'src'))
import torch
torch.set_num_threads(4)
from true_rl_agent.tasks.stair_climbing.numeric_capture import recompute_stair_numeric_evidence

rows = []
for name in ('r2_parent_on', 'r2_gen8_on', 'r2_g1_on'):
    receipt = args.original_root / name / 'public_dev_result.json'
    evidence = json.loads(receipt.read_text())['task_evidence']
    start = time.monotonic()
    result = recompute_stair_numeric_evidence(evidence['numeric_evidence'])
    row = {'case': name, 'episodes': result['episode_count'], 'seconds': time.monotonic()-start,
           'summary_equal': result['summary'] == evidence['summary'],
           'contact_force_history': result['contact_force_history'],
           'receipt_sha256': hashlib.sha256(receipt.read_bytes()).hexdigest()}
    assert row['summary_equal'] and row['contact_force_history']['status'] == 'not_recorded', row
    rows.append(row)
    print(json.dumps(row), flush=True)
report = {'status': 'PASS', 'source': str(args.source), 'original_root': str(args.original_root), 'rows': rows}
args.output.write_text(json.dumps(report, indent=2)+'\n')
