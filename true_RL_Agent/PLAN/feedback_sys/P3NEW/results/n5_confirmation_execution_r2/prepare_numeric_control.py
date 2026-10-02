"""One-shot R2 numeric CPU preparation; no model request, GPU run, or release.

Run only after both frozen captures have completed. Keep all outputs if any
command fails; no implicit retry or replacement is implemented here.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

B = Path('/data/yuyangliu/AGENT/feedback_p3new_20260929')
S, C = B / 'n5_confirmation_source_r2', B / 'n5_execution_20260930_r2'
CAP = B / 'n5_confirmation_20260930_r2'
P = S / 'PLAN/feedback_sys/P3NEW'
RAW, PACK = CAP / 'numeric_raw', CAP / 'numeric_materials'
DEFINITIONS, OUT = C / 'numeric_task_definitions', C / 'postcapture/numeric'
CARD_SHA = '6bea98ed63e1f1e516631e58205e3da93406311d1b87466e445fe754b0c22794'
SOURCE_SHA = '497d14b6a133d1ba3afa97e15c39b00515479692e528aac648970851245f3b76'
CONTROL_SHA = {
    'verify_numeric_arithmetic.py': 'd6e4be5cca7a5363b53d6a2c808aea894320a062dc507171795d3ebbd0ec063b',
    'count_prepared_text.py': '475fa8cf776c335010e6720733f947d25e15329a02df5003f399338dd4d80b52',
}


def now():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def ref(path):
    data = path.read_bytes()
    return {'path': str(path), 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}


def read(path):
    return json.loads(path.read_bytes())


def save(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write('\n')


def main():
    require(ref(C / 'capture_card.json')['sha256'] == CARD_SHA, 'Capture card differs')
    require(ref(C / 'runtime_source_manifest.json')['sha256'] == SOURCE_SHA, 'Source manifest differs')
    card, source = read(C / 'capture_card.json'), read(C / 'runtime_source_manifest.json')
    require(card['status'] == 'FROZEN' and card['output_root'] == str(CAP)
            and card['source']['remote_root'] == str(S), 'Unexpected frozen roots')
    capture_refs = {}
    for task in ('stair', 'push'):
        require(read(C / f'{task}.exit.json')['returncode'] == 0, task + ' capture did not exit successfully')
        require(read(CAP / task / 'result.json')['status'] == 'PASS', task + ' capture wrapper did not pass')
        phase = CAP / task / 'public_dev_result.json'
        require(read(phase)['evaluation_binding_hash'] ==
                card['tasks'][task]['current_evaluation_binding']['evaluation_binding_hash'],
                task + ' phase belongs to another cohort')
        capture_refs[task] = {'exit': ref(C / f'{task}.exit.json'), 'phase': ref(phase),
                              'wrapper': ref(CAP / task / 'result.json')}
    for name, sha in CONTROL_SHA.items():
        require(ref(C / name)['sha256'] == sha, 'Control helper differs: ' + name)
    for name in ('export_confirmation_raw.py', 'prepare_review_task_contract.py',
                 'prepare_confirmation_materials.py', 'run_confirmation_review.py'):
        record = source['PLAN/feedback_sys/P3NEW/' + name]
        actual = ref(P / name)
        require(actual['sha256'] == record['sha256'] and actual['bytes'] == record['size_bytes'],
                'Frozen entrypoint differs: ' + name)
    destinations = (RAW, PACK, DEFINITIONS, OUT, C / 'control/numeric_arithmetic')
    require(not any(path.exists() for path in destinations), 'Preserve previous output; no automatic retry')

    stair = CAP / 'stair/numeric_evidence/manifest.json'
    push = [CAP / f'push/numeric/public_dev-{n:02d}/manifest.json' for n in (1, 2, 3)]
    cell = CAP / 'push/composition/resolved_cell_manifest.json'
    protocol = P / 'N5_FACT_REVIEW_PROTOCOL_ZH.md'
    commands = [('raw_export', [P / 'export_confirmation_raw.py', '--stair', stair, '--push', *push,
        '--output', RAW, '--purpose', 'confirmation', '--capture-card', C / 'capture_card.json',
        '--review-protocol', protocol, '--profile', 'related_v1', '--part-steps', '64'])]
    definitions = [('stair', stair, DEFINITIONS / 'stair.json')]
    definitions.extend(('push', path, DEFINITIONS / f'push_run{n:02d}.json') for n, path in enumerate(push, 1))
    for task, numeric, destination in definitions:
        argv = [P / 'prepare_review_task_contract.py', '--task', task, '--numeric-manifest', numeric,
                '--source-manifest', RAW / '_private/sources.json', '--output', destination]
        if task == 'push':
            argv.extend(['--resolved-cell-manifest', cell])
        commands.append(('definition_' + destination.stem, argv))
    packing = [P / 'prepare_confirmation_materials.py', '--raw-export', RAW, '--output', PACK,
        '--protocol', protocol, '--capture-card', C / 'capture_card.json', '--profile', 'related_v1',
        '--max-output-tokens', '32768', '--push-resolved-cell-manifest', cell]
    for _, _, destination in definitions:
        packing.extend(['--task-definition', destination])
    commands.extend([
        ('numeric_pack', packing),
        ('independent_arithmetic', [C / 'verify_numeric_arithmetic.py']),
        ('static_text_count', [C / 'count_prepared_text.py', '--prepared-root', PACK / 'prepared',
                               '--output', OUT / 'prepared_text_counts.json']),
    ])
    commands = [(name, [sys.executable, '-B', *map(str, argv)]) for name, argv in commands]
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=f'{S}/src:{S}:{P}')
    OUT.mkdir(parents=True, exist_ok=False)
    receipt = {'status': 'NOT_RELEASED', 'execution_status': 'IN_PROGRESS', 'started_at_utc': now(),
        'script': ref(Path(__file__).resolve()), 'capture_card': ref(C / 'capture_card.json'),
        'capture_refs': capture_refs, 'source_manifest': ref(C / 'runtime_source_manifest.json'),
        'commands': commands, 'cwd': str(S), 'steps': [], 'model_calls': 0, 'gpu_runs': 0,
        'semantic_review_performed': False, 'facts_admitted': False, 'automatic_retries': 0,
        'scope': '36 numeric facts units only; four media units prepared separately; no report requests'}
    save(OUT / 'commands.json', receipt)
    try:
        for name, argv in commands:
            step = {'name': name, 'started_at_utc': now(), 'argv': argv}
            save(OUT / f'{name}.started.json', step)
            with (OUT / f'{name}.stdout.log').open('xb') as stdout, (OUT / f'{name}.stderr.log').open('xb') as stderr:
                process = subprocess.Popen(argv, cwd=S, env=env, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr)
                code = process.wait()
            step.update(returncode=code, finished_at_utc=now(), child_pid=process.pid,
                        stdout=ref(OUT / f'{name}.stdout.log'), stderr=ref(OUT / f'{name}.stderr.log'))
            save(OUT / f'{name}.exit.json', step)
            receipt['steps'].append(step)
            print(json.dumps({'step': name, 'returncode': code, 'finished_at_utc': step['finished_at_utc']}), flush=True)
            require(code == 0, name + ' failed; retain evidence and stop downstream')
        packing_receipt = read(PACK / 'packing_receipt.json')
        counts = read(OUT / 'prepared_text_counts.json')
        require(packing_receipt['episode_count'] == 24 and packing_receipt['focus_count'] == 12
                and packing_receipt['actual_calls'] == 0 and len(packing_receipt['packets']) == 36,
                'Numeric unit scope differs')
        require(counts['requests_counted'] == 72 and counts['image_count'] == 0
                and all(row['stage'] == 'facts' for row in counts['records']), 'Expected 72 numeric-only facts requests')
        receipt.update(execution_status='COMPLETED', completed_at_utc=now(),
            artifacts={key: ref(path) for key, path in {
                'raw_manifest': RAW / 'manifest.json', 'packing_receipt': PACK / 'packing_receipt.json',
                'arithmetic_receipt': C / 'control/numeric_arithmetic/arithmetic_receipt.json',
                'prepared_text_counts': OUT / 'prepared_text_counts.json'}.items()})
        save(OUT / 'execution_receipt.json', receipt)
        return 0
    except Exception as error:
        receipt.update(execution_status='ERROR', stopped_at_utc=now(), error_type=type(error).__name__, error=str(error))
        with (OUT / 'exception.traceback.txt').open('x', encoding='utf-8') as stream:
            traceback.print_exc(file=stream)
        save(OUT / 'execution_receipt.json', receipt)
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
