"""Prepare registered R11 input once, with no provider request or GPU work."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import subprocess
import tarfile

BASE = Path('/data/yuyangliu/AGENT/feedback_p3new_20260929')
CONTROL = BASE / 'body_grounding_observer_r11/control'
SOURCE = BASE / 'source_body_grounding_observer_r11'
EXPORT = BASE / 'push_body_grounding_dev_r2_export'
PYTHON = '/data/yuyangliu/envs/isaaclab6_validation/bin/python'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, obj):
    with path.open('x') as stream:
        json.dump(obj, stream, indent=2, allow_nan=False)


def main():
    plan = read(CONTROL / 'preparation_plan.v2.json')
    assert sha(CONTROL / 'preparation_plan.v2.json') == '4e8a5a91a63426c36269e6ee1b8a13707a742a62c18786800ad85b36401e595f'
    assert sha(CONTROL / 'source.tar.gz') == plan['observer_source_bundle_sha256']
    assert sha(CONTROL / 'source_manifest.json') == plan['source_manifest_sha256']
    assert sha(CONTROL / 'R11_REGISTRATION_ZH.md') == plan['registration_sha256']
    assert sha(CONTROL / 'R11_REGISTRATION_AMENDMENT_01_ZH.md') == plan['before_call_registration_revision_sha256']
    assert sha(CONTROL / 'send_native_observer.py') == plan['sender_sha256']
    assert sha(EXPORT / 'extraction.json') == plan['extraction_sha256']
    assert read(BASE / 'body_grounding_dense_r2/verification.json')['status'] == 'pass'
    files = read(CONTROL / 'source_manifest.json')['files']
    assert len(files) == 401 and not SOURCE.exists()
    SOURCE.mkdir()
    with tarfile.open(CONTROL / 'source.tar.gz') as bundle:
        members = bundle.getmembers()
        assert {m.name for m in members if m.isfile()} == set(files)
        assert all(m.isfile() and (SOURCE / m.name).resolve().is_relative_to(SOURCE) for m in members)
        bundle.extractall(SOURCE, filter='data')
    assert all(sha(SOURCE / p) == h for p, h in files.items())
    capture_manifest = BASE / 'body_grounding_capture_r1_control/source_manifest.json'
    assert sha(capture_manifest) == plan['capture_source_manifest_sha256']
    original = read(capture_manifest)['files']
    assert all(sha(Path(plan['capture_source_root']) / p) == h for p, h in original.items())
    assert {p for p in files if files[p] != original[p]} == {'src/true_rl_agent/evaluation/visual_observer.py'}
    write(CONTROL / 'remote_source_verification.json', {'checked_at_utc': datetime.now(timezone.utc).isoformat(),
          'source_manifest_sha256': sha(CONTROL / 'source_manifest.json'), 'observer_files_verified': 401,
          'capture_files_unchanged': 401, 'provider_requests': 0, 'gpu_jobs': 0})
    selection = read(CONTROL / 'selection.draft.json')
    selection['source']['extraction_sha256'] = sha(EXPORT / 'extraction.json')
    selection['source']['description'] = 'same-run interval-1 v4 old-development replay'
    write(CONTROL / 'selection.json', selection)
    output = CONTROL / 'push_run01'
    argv = [PYTHON, '-B', str(SOURCE / 'PLAN/feedback_sys/P3NEW/prepare_native_observer.py'),
            '--export', str(EXPORT), '--selection', str(CONTROL / 'selection.json'), '--output', str(output)]
    write(CONTROL / 'prepare_invocation.json', {'argv': argv, 'started_at_utc': datetime.now(timezone.utc).isoformat(),
          'provider_requests': 0, 'automatic_retry': False})
    with (CONTROL / 'prepare.log').open('xb') as log:
        completed = subprocess.run(argv, env=dict(os.environ, PYTHONPATH=str(SOURCE / 'src')), stdout=log, stderr=subprocess.STDOUT)
    write(CONTROL / 'prepare_receipt.json', {'returncode': completed.returncode,
          'finished_at_utc': datetime.now(timezone.utc).isoformat(), 'provider_requests': 0,
          'gpu_jobs': 0, 'prepare_driver_sha256': sha(Path(argv[2]))})
    if completed.returncode != 0:
        print(json.dumps(read(CONTROL / 'prepare_receipt.json')), flush=True)
        print((CONTROL / 'prepare.log').read_text(errors='replace'), flush=True)
        return 1
    request = read(output / 'request.json')
    blind = {k: v for k, v in request.items() if k != 'messages'}
    blind['messages'] = []
    for message in request['messages']:
        content = message['content']
        texts = [content] if isinstance(content, str) else [item['text'] for item in content if item['type'] == 'text']
        blind['messages'].append({'role': message['role'], 'text_blocks': texts})
    write(output / 'blind_input_text.json', blind)
    selected = read(output / 'private_sidecar.json')['prepare_private_sidecar'][0]['images']
    episode, = [e for e in read(EXPORT / 'extraction.json')['episodes'] if e['run_index'] == 1 and e['env_index'] == 19]
    pairs = {(p['control_step'], p['view_id']): p for p in episode['snapshot_pairs']}
    images = []
    for image in selected:
        step, = image['displayed_steps']
        row = pairs[(step, image['view_id'])]
        assert image['sha256'] == row['sha256'] == sha(EXPORT / row['file'])
        images.append(row)
    assert len(images) == 12
    write(output / 'selected_image_sources.json', {'extraction_sha256': sha(EXPORT / 'extraction.json'),
          'original_images': images, 'local_delivery': 'body_grounding_dense_r2/materials/push_run01',
          'provider_requests': 0})
    delivery = {p.name: p for p in (CONTROL / name for name in (
        'selection.json', 'remote_source_verification.json', 'prepare_invocation.json', 'prepare_receipt.json', 'prepare.log'))}
    delivery.update({'push_run01/' + p.name: p for p in output.iterdir() if p.is_file() and p.name != 'request.json'})
    index = {'provider_requests': 0, 'request_sha256': sha(output / 'request.json'),
             'request_bytes': (output / 'request.json').stat().st_size,
             'prepared_card_sha256': sha(output / 'card.json'),
             'files': {name: {'source_path': str(p), 'sha256': sha(p), 'bytes': p.stat().st_size} for name, p in delivery.items()}}
    write(CONTROL / 'prepared_delivery_index.json', index)
    archive = CONTROL / 'prepared_review_materials.tar.gz'
    with tarfile.open(archive, 'w:gz') as bundle:
        for name, p in delivery.items():
            bundle.add(p, arcname=name, recursive=False)
        bundle.add(CONTROL / 'prepared_delivery_index.json', arcname='prepared_delivery_index.json', recursive=False)
    print(json.dumps({'status': 'PREPARED_NOT_SENT', 'provider_requests': 0, 'bundle': str(archive),
          'bundle_sha256': sha(archive), 'bundle_bytes': archive.stat().st_size,
          'request_sha256': index['request_sha256'], 'request_bytes': index['request_bytes'],
          'prepared_card_sha256': index['prepared_card_sha256']}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
