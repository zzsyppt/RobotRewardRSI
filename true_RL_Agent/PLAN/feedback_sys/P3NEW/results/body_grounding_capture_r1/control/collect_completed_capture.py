"""One CPU post-processing pass after the registered capture has terminated."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import tarfile

BASE = Path('/data/yuyangliu/AGENT/feedback_p3new_20260929')
SOURCE = BASE / 'source_body_grounding_r1'
CONTROL = BASE / 'body_grounding_capture_r1_control'
CAPTURE = BASE / 'push_body_grounding_dev_r1'
PYTHON = '/data/yuyangliu/envs/isaaclab6_validation/bin/python'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    actual = json.loads((CAPTURE / 'result.json').read_text())
    print(json.dumps({k: actual.get(k) for k in ('status', 'error', 'gpu_capacity_seconds')}), flush=True)
    if actual['status'] == 'RUNNING':
        return 10
    if actual['status'] != 'PASS':
        print((CONTROL / 'run.log').read_text(errors='replace')[-12000:], flush=True)
        return 11
    card = json.loads((CONTROL / 'capture_card.json').read_text())
    inventory = json.loads((CONTROL / 'source_manifest.json').read_text())['files']
    if any(sha(SOURCE / p) != h for p, h in inventory.items()):
        raise ValueError('frozen source changed')
    env = dict(os.environ, PYTHONPATH=str(SOURCE / 'src'))
    verification = CONTROL / 'verification.json'
    if not verification.exists():
        subprocess.run([PYTHON, str(SOURCE / 'PLAN/feedback_sys/P3NEW/results/body_grounding_capture_r1/control/verify_capture.py'),
            '--source', str(SOURCE), '--baseline', card['baseline_root'], '--candidate', str(CAPTURE),
            '--card', str(CONTROL / 'capture_card.json'), '--output', str(verification)], env=env, check=False)
    export = BASE / 'push_body_grounding_dev_r1_export'
    if not export.exists():
        subprocess.run([PYTHON, str(SOURCE / 'PLAN/feedback_sys/P3NEW/extract_dual_view_evidence.py'),
            '--summary', str(CAPTURE / 'evaluation/public_dev/summary.json'),
            '--recordings', str(CAPTURE / 'video/recordings.json'), '--output', str(export),
            '--resolved-cell-manifest', str(CAPTURE / 'composition/resolved_cell_manifest.json')], env=env, check=True)
    geometry = CONTROL / 'run01_body_geometry.json'
    if not geometry.exists():
        subprocess.run([PYTHON, str(CONTROL / 'measure_geometry.py'), '--source', str(SOURCE),
                        '--capture', str(CAPTURE), '--output', str(geometry)], env=env, check=True)
    extraction = json.loads((export / 'extraction.json').read_text())
    episode, = [row for row in extraction['episodes'] if row['run_index'] == 1 and row['env_index'] == 19]
    files = {f'receipt/{name}': CONTROL / name for name in (
        'remote_source_verification.json', 'resource_precheck.json', 'run_launch.json', 'verification.json', 'run.log')}
    files.update({'receipt/result.json': CAPTURE / 'result.json',
                  'receipt/preflight_result.json': BASE / 'push_body_grounding_dev_r1_preflight/result.json',
                  'materials/run01_body_geometry.json': geometry})
    relative = ['extraction.json', 'source_summary.json', 'resolved_cell_manifest.json', 'video/recordings.json',
                episode['numeric_file'], episode['exported_source_manifest'], episode['snapshot_manifest']]
    relative.extend(row['file'] for row in episode['snapshot_pairs'])
    for name in relative:
        files['materials/push_run01/' + name] = export / name
    index = {'schema_version': 'p3new.body_grounding_delivery.v1', 'scope': 'run1 original media and selected numeric; all runs mechanical receipt',
             'files': {name: {'source_path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}
                       for name, path in files.items()},
             'provider_requests': 0, 'gpu_capture_attempts': 1, 'run1_image_count': len(episode['snapshot_pairs']),
             'verification_status': json.loads(verification.read_text())['status']}
    index_path = CONTROL / 'download_index.json'
    if index_path.exists():
        if json.loads(index_path.read_text()) != index:
            raise ValueError('delivery inputs changed since index')
    else:
        with index_path.open('x') as stream:
            json.dump(index, stream, indent=2)
    archive = CONTROL / 'run01_review_bundle.tar.gz'
    if not archive.exists():
        with tarfile.open(archive, 'w:gz') as bundle:
            for name, path in files.items():
                bundle.add(path, arcname=name, recursive=False)
            bundle.add(index_path, arcname='receipt/download_index.json', recursive=False)
    print(json.dumps({'bundle': str(archive), 'sha256': sha(archive), 'bytes': archive.stat().st_size,
                      'files': len(files) + 1, 'verification_status': index['verification_status'],
                      'run1_image_count': index['run1_image_count']}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
