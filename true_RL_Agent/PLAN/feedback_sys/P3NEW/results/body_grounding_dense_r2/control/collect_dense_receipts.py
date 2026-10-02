"""Collect completed dense-capture receipts; no preparation or provider calls."""
from pathlib import Path
import hashlib
import json
import tarfile

BASE = Path('/data/yuyangliu/AGENT/feedback_p3new_20260929')
CONTROL = BASE / 'body_grounding_dense_r2'
CAPTURE = BASE / 'push_body_grounding_dev_r2'
EXPORT = BASE / 'push_body_grounding_dev_r2_export'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    actual = json.loads((CAPTURE / 'result.json').read_text())
    audit = json.loads((CONTROL / 'verification.json').read_text())
    assert actual['status'] == 'PASS' and audit['status'] == 'pass'
    extraction = json.loads((EXPORT / 'extraction.json').read_text())
    episode, = [e for e in extraction['episodes'] if e['run_index'] == 1 and e['env_index'] == 19]
    files = {'receipt/' + name: CONTROL / name for name in (
        'remote_preparation_receipt.json', 'resource_precheck.json', 'run_launch.json', 'verification.json', 'run.log')}
    files['receipt/result.json'] = CAPTURE / 'result.json'
    files['receipt/preflight_result.json'] = BASE / 'push_body_grounding_dev_r2_preflight/result.json'
    for name in ('extraction.json', 'source_summary.json', 'resolved_cell_manifest.json',
                 episode['numeric_file'], episode['exported_source_manifest'], episode['snapshot_manifest']):
        files['materials/push_run01/' + name] = EXPORT / name
    old_snapshot = json.loads((BASE / 'push_body_grounding_dev_r1/video/public_dev-01.snapshots.json').read_text())
    old_images = {(s['control_step'], v['view_id']): v['sha256'] for s in old_snapshot['samples'] for v in s['views']}
    selected = []
    for row in episode['snapshot_pairs']:
        key = (row['control_step'], row['view_id'])
        if key in old_images:
            assert sha(EXPORT / row['file']) == row['sha256']
            selected.append(dict(row, r1_sha256=old_images[key], same_bytes_as_r1=row['sha256'] == old_images[key]))
    assert len(selected) == 12
    comparison = {'images': selected, 'all_12_same_bytes_as_r1': all(r['same_bytes_as_r1'] for r in selected)}
    comparison_path = CONTROL / 'selected_image_comparison_r1.json'
    if comparison_path.exists():
        assert json.loads(comparison_path.read_text()) == comparison
    else:
        with comparison_path.open('x') as stream:
            json.dump(comparison, stream, indent=2)
    files['materials/selected_image_comparison_r1.json'] = comparison_path
    if not comparison['all_12_same_bytes_as_r1']:
        for row in selected:
            files['materials/push_run01/' + row['file']] = EXPORT / row['file']
    index = {'provider_requests': 0, 'gpu_capture_attempts': 1, 'complete_remote_export': str(EXPORT),
             'verification_status': audit['status'], 'image_count': audit['image_count'],
             'sample_count': audit['sample_count'], 'gpu_capacity_seconds': actual['gpu_capacity_seconds'],
             'selected_images_same_bytes_as_r1': comparison['all_12_same_bytes_as_r1'],
             'files': {name: {'source_path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}
                       for name, path in files.items()}}
    index_path = CONTROL / 'completion_delivery_index.json'
    if index_path.exists():
        assert json.loads(index_path.read_text()) == index
    else:
        with index_path.open('x') as stream:
            json.dump(index, stream, indent=2)
    archive = CONTROL / 'dense_completion_receipts.tar.gz'
    if not archive.exists():
        with tarfile.open(archive, 'w:gz') as bundle:
            for name, path in files.items():
                bundle.add(path, arcname=name, recursive=False)
            bundle.add(index_path, arcname='receipt/completion_delivery_index.json', recursive=False)
    print(json.dumps({'bundle': str(archive), 'sha256': sha(archive), 'bytes': archive.stat().st_size,
                      'files': len(files) + 1, 'image_count': audit['image_count'],
                      'sample_count': audit['sample_count'], 'gpu_capacity_seconds': actual['gpu_capacity_seconds'],
                      'selected_images_same_bytes_as_r1': comparison['all_12_same_bytes_as_r1'],
                      'complete_export': str(EXPORT), 'extraction_sha256': sha(EXPORT / 'extraction.json')}), flush=True)


if __name__ == '__main__':
    main()
