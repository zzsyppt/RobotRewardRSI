"""Read-only collection of named historical artifacts. Run on humanoid-server."""
from __future__ import annotations

import hashlib
import json
import subprocess
import shutil
from datetime import datetime, timezone
from pathlib import Path

BASE = Path('/data/yuyangliu/AGENT/true_RL_Agent')
OUT = Path(__file__).resolve().parent


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    report = {'collected_at': datetime.now(timezone.utc).isoformat(), 'source_root': str(BASE),
              'files': [], 'checkpoints': [], 'resources': {}}
    for name, cmd in {
        'gpu': ['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,utilization.gpu', '--format=csv,noheader'],
        'gpu_processes': ['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory', '--format=csv,noheader'],
        'disk': ['df', '-B1', '/data'],
        'memory': ['free', '-b'],
    }.items():
        p = subprocess.run(cmd, capture_output=True, text=True)
        report['resources'][name] = {'returncode': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}
    session = BASE / 'outputs/formal_go2_stair_15round_20260901_c6ab8ba'
    reference = BASE / 'outputs/p5_go2_pseudofall_removed_genesis_0a3f08a'
    sources = set()
    for root in (session, reference):
        for name in ('session_genesis.json', 'generation_session_report.json', 'ledger.jsonl',
                     'resolved_cell_manifest.json', 'generation_session_contract.json',
                     'reference_train_job.json', 'reference_public_dev_gate.json', 'reference_eval_artifact.json'):
            path = root / name
            if path.is_file():
                sources.add(path)
        for pattern in ('generation_actual_cost/*settlement.json', 'runner/*/train/summary.json',
                        'runner/*/eval/public_dev/summary.json', '**/reference_qualification.json'):
            sources.update(p for p in root.glob(pattern) if p.is_file())
    # Original authoring receipts bind actual historical prompts to usage.
    for pattern in ('authoring/**/attempt_result.json', 'authoring/**/prompt_payload.json'):
        sources.update(p for p in session.glob(pattern) if p.is_file())
    total = sum(p.stat().st_size for p in sources)
    if total > 150 * 1024**2:
        raise RuntimeError('bounded raw-JSON export exceeds 150 MiB')
    for path in sorted(sources):
        rel = path.relative_to(BASE).as_posix()
        destination = OUT / 'inventory_remote_raw' / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        report['files'].append({'path': rel, 'size': path.stat().st_size, 'sha256': sha(path)})
    historical = json.loads((OUT / 'protected_before.json').read_text())
    for rel, prior in historical.items():
        if not rel.endswith('.pt'):
            continue
        path = session / rel
        exists = path.is_file()
        row = {'path': str(path), 'relative_path': rel, 'exists': exists, 'historical_sha256': prior[2]}
        if exists:
            row.update(size=path.stat().st_size, sha256=sha(path))
            row['matches_historical'] = row['sha256'] == prior[2]
        report['checkpoints'].append(row)
    (OUT / 'remote_baseline.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'raw_files': len(sources), 'raw_bytes': total, 'checkpoints': len(report['checkpoints']),
                      'present': sum(x['exists'] for x in report['checkpoints']),
                      'hash_matches': sum(x.get('matches_historical', False) for x in report['checkpoints'])}))


if __name__ == '__main__':
    main()
