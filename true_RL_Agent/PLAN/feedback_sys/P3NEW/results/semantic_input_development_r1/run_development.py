"""Freeze current helpers, execute a new remote offline attempt, keep exact logs."""
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

D = Path(__file__).resolve().parent
ROOT = D.parents[4]
attempt = sys.argv[1] if len(sys.argv) == 2 else 'attempt_01'
if not attempt.startswith('attempt_') or not attempt[8:].isdigit():
    raise ValueError('explicit attempt_NN required')
out = D / attempt
out.mkdir(exist_ok=False)
names = [f'PLAN/feedback_sys/P3NEW/{name}' for name in ('prepare_confirmation_materials.py', 'export_confirmation_raw.py',
    'run_confirmation_review.py', 'confirmation_review_coverage.py', 'prepare_review_task_contract.py', 'N5_FACT_REVIEW_PROTOCOL_ZH.md')]
names.append('PLAN/feedback_sys/P3R/confirmation_raw_material.py')
bundle, refs = {}, []
for name in names:
    raw = (ROOT / name).read_bytes()
    dest = out / 'executed_source' / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open('xb') as stream:
        stream.write(raw)
    bundle[name] = base64.b64encode(raw).decode()
    refs.append({'file': str((ROOT / name).resolve()), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)})
source = ('ATTEMPT = ' + repr(attempt) + '\nSOURCE_BUNDLE = ' + repr(bundle) + '\n').encode() + (D / 'rebuild_remote.py').read_bytes()
argv = ['ssh', 'humanoid-server', '/data/yuyangliu/envs/isaaclab6_validation/bin/python', '-B', '-']
started = time.time()
with (out / 'stdout.jsonl').open('xb') as stdout, (out / 'stderr.txt').open('xb') as stderr:
    completed = subprocess.run(argv, input=source, stdout=stdout, stderr=stderr, creationflags=subprocess.CREATE_NO_WINDOW)
execution = {'argv': argv, 'attempt': attempt, 'exit_code': completed.returncode, 'elapsed_seconds': time.time() - started,
    'remote_output': '/data/yuyangliu/AGENT/feedback_p3new_20260929/semantic_input_development_r1/' + attempt,
    'stdin_script_sha256': hashlib.sha256(source).hexdigest(), 'source_refs': refs,
    'model_calls': 0, 'gpu_calls': 0, 'new_capture': False}
with (out / 'execution.json').open('x', encoding='utf-8') as stream:
    json.dump(execution, stream, ensure_ascii=False, indent=2)
print((out / 'stdout.jsonl').read_text(encoding='utf-8'))
print((out / 'stderr.txt').read_text(encoding='utf-8'))
print(json.dumps(execution, ensure_ascii=False))
raise SystemExit(completed.returncode)
