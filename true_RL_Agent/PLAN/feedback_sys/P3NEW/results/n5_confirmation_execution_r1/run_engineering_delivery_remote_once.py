"""Run the existing offline assembler exactly once, preserving stdout/stderr/exit."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import subprocess

BASE = Path('/data/yuyangliu/AGENT/feedback_p3new_20260929')
CONTROL = BASE / 'n5_execution_20260930'
STAGE = CONTROL / 'engineering_delivery_stage_r1'
PACKET = BASE / 'n5_confirmation_20260930/engineering_repair/observers/push'
SOURCE = BASE / 'n5_selection_repair_source_r1'
MATERIAL = BASE / 'text_author_smoke_r1'
OUTPUT = BASE / 'n5_confirmation_20260930/engineering_repair/delivery_offline_r1'


def ref(path):
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': str(path), 'sha256': digest, 'size_bytes': path.stat().st_size}


def save(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


manifest = json.loads((STAGE / 'engineering_delivery_execution_manifest.json').read_bytes())
allowed = {CONTROL / 'assemble_delivery.py', CONTROL / 'engineering_delivery_inputs.ready.json',
           PACKET / 'root_module_review.json', PACKET / 'root_module_review_receipt.json'}
assert {Path(row['path']) for row in manifest['files']} == allowed
assert manifest['output_directory'] == str(OUTPUT)
assert manifest['source_root'] == str(SOURCE) and manifest['material_root'] == str(MATERIAL)
assert not OUTPUT.exists(), 'fresh output required'
copied, existing = [], []
for row in manifest['files']:
    target = Path(row['path'])
    origin = STAGE / target.name
    assert ref(origin)['sha256'] == row['sha256'] and origin.stat().st_size == row['size_bytes']
    if target.exists():
        assert ref(target)['sha256'] == row['sha256'], 'existing target differs: ' + str(target)
        existing.append(str(target))
for row in manifest['files']:
    target = Path(row['path'])
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write((STAGE / target.name).read_bytes())
        copied.append(str(target))
    assert ref(target)['sha256'] == row['sha256']
inputs = json.loads((CONTROL / 'engineering_delivery_inputs.ready.json').read_bytes())
assert inputs['execution_authorized_by_root'] is True
for task in ('stair', 'push'):
    for key in ('review', 'as_of_cutoff_ref'):
        record = inputs['tasks'][task][key]
        assert ref(Path(record['path']))['sha256'] == record['sha256']

stdout = CONTROL / 'engineering_delivery_r1.stdout.txt'
stderr = CONTROL / 'engineering_delivery_r1.stderr.txt'
invocation = CONTROL / 'engineering_delivery_r1.invocation.json'
exit_path = CONTROL / 'engineering_delivery_r1.exit.json'
assert not any(path.exists() for path in (stdout, stderr, invocation, exit_path))
argv = ['/data/yuyangliu/envs/isaaclab6_validation/bin/python', '-B', str(CONTROL / 'assemble_delivery.py'),
        '--inputs', str(CONTROL / 'engineering_delivery_inputs.ready.json'),
        '--source-root', str(SOURCE), '--material-root', str(MATERIAL), '--output', str(OUTPUT)]
changes = {'PYTHONPATH': str(SOURCE / 'src') + ':' + str(SOURCE / 'PLAN/feedback_sys/P3NEW'),
           'PYTHONDONTWRITEBYTECODE': '1', 'CUDA_VISIBLE_DEVICES': ''}
record = {'status': 'DEVELOPMENT_REPAIR_NOT_ORIGINAL_N5_CONFIRMATION', 'original_n5_status': 'INCONCLUSIVE',
          'started_at_utc': datetime.now(timezone.utc).isoformat(), 'argv': argv, 'cwd': str(CONTROL),
          'environment_set_process_local': changes, 'script': ref(CONTROL / 'assemble_delivery.py'),
          'ready_inputs': ref(CONTROL / 'engineering_delivery_inputs.ready.json'),
          'files_copied_without_rewriting_bytes': copied, 'existing_identical_files': existing,
          'stdout': str(stdout), 'stderr': str(stderr), 'automatic_retry': False,
          'model_calls': 0, 'GPU_jobs': 0, 'training_jobs': 0}
save(invocation, record)
exit_record = dict(record)
try:
    environment = os.environ.copy()
    environment.update(changes)
    with stdout.open('xb') as out, stderr.open('xb') as err:
        completed = subprocess.run(argv, cwd=CONTROL, env=environment, stdout=out, stderr=err, timeout=1800)
    exit_record['returncode'] = completed.returncode
except Exception as error:
    exit_record['returncode'] = None
    exit_record['exception_type'] = type(error).__name__
finally:
    exit_record['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
    exit_record['stdout_ref'] = ref(stdout)
    exit_record['stderr_ref'] = ref(stderr)
    save(exit_path, exit_record)
response = {'status': record['status'], 'returncode': exit_record['returncode'], 'exit_record': ref(exit_path),
            'stdout': ref(stdout), 'stderr': ref(stderr), 'assembly_stdout': stdout.read_text(encoding='utf-8')}
if exit_record['returncode'] == 0:
    receipt_path = OUTPUT / 'assembly_receipt.json'
    receipt = json.loads(receipt_path.read_bytes())
    assert receipt['new_author_calls'] == receipt['model_calls'] == receipt['gpu_calls'] == 0
    for row in manifest['files']:
        assert ref(Path(row['path']))['sha256'] == row['sha256']
    scope = {'status': record['status'], 'original_n5_status': 'INCONCLUSIVE',
             'scope_applies_to': 'all files in this delivery directory, including unchanged-schema production payloads',
             'assembled_not_sent': True, 'formal_confirmation': False, 'formal_scores_created': False,
             'assembly_receipt': ref(receipt_path), 'execution_exit': ref(exit_path),
             'new_model_calls': 0, 'training_jobs': 0, 'GPU_jobs': 0}
    save(OUTPUT / 'ENGINEERING_SCOPE.json', scope)
    response['assembly_receipt'] = ref(receipt_path)
    response['scope'] = ref(OUTPUT / 'ENGINEERING_SCOPE.json')
print(json.dumps(response))
