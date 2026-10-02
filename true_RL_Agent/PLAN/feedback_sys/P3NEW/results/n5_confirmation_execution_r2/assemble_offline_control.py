"""Bind actual R2 evidence and run the unchanged frozen offline assembler once."""
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
CAP, MATERIAL = B / 'n5_confirmation_20260930_r2', B / 'text_author_smoke_r1'
P = S / 'PLAN/feedback_sys/P3NEW'
DRIVER = P / 'results/n5_confirmation_execution_r1/assemble_delivery.py'
OUT, DELIVERY = C / 'postcapture/delivery', CAP / 'delivery_offline_r1'
SEALS = {
    'stair': ('2026-09-30T08:46:05.242044+00:00',
              '585d0e37cf6974d770a13edd124a8f5bdf40bd629a94333433e8116356b84919',
              '1e26c260e1e2d7aaef738df099e8a41cbe544064880f19f338bae8f223cbf585'),
    'push': ('2026-09-30T09:31:13.024858+00:00',
             'af3ae05d1a30bf0912e1fa49b191ab4a8ebb8ed9875e97a1b77c04ad967264f2',
             '4c7c9e94b91b25d7ff522b8c58ed843ae29d957f7e5c97c114809da543e3c916'),
}


def read(path):
    return json.loads(path.read_bytes())


def ref(path):
    data = path.read_bytes()
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(data).hexdigest(), 'size_bytes': len(data)}


def checked(reference):
    actual = ref(Path(reference['path']))
    require(actual['sha256'] == reference['sha256'] and
            actual['size_bytes'] == reference.get('size_bytes', actual['size_bytes']), 'Reference differs: ' + reference['path'])
    return dict(reference)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write('\n')


def main():
    require(not OUT.exists() and not DELIVERY.exists() and not (C / 'delivery_inputs.ready.json').exists(),
            'Preserve any previous attempt')
    frozen = read(C / 'runtime_source_manifest.json')
    require(ref(DRIVER)['sha256'] == frozen[DRIVER.relative_to(S).as_posix()]['sha256'], 'Frozen assembler changed')
    require(ref(C / 'capture_card.json')['sha256'] ==
            '6bea98ed63e1f1e516631e58205e3da93406311d1b87466e445fe754b0c22794', 'Capture card changed')
    old_path = B / 'n5_execution_20260930/engineering_delivery_inputs.ready.json'
    old = read(old_path)
    ready = {'schema_version': 'p3new.n5_offline_delivery_inputs.v1',
             'status': 'READY_FOR_OFFLINE_SERIALIZATION_NOT_RELEASED', 'capture_card': ref(C / 'capture_card.json'),
             'qualification_reader': checked(old['qualification_reader']),
             'qualification_index': checked(old['qualification_index']), 'tasks': {}}
    for task in ('stair', 'push'):
        root = CAP / 'observers' / task
        review, seal_path = root / 'root_module_review.json', root / 'root_module_review_receipt.json'
        seal = read(seal_path)
        cutoff, review_sha, seal_sha = SEALS[task]
        require(ref(review)['sha256'] == review_sha and ref(seal_path)['sha256'] == seal_sha
                and seal['recorded_at_utc'] == cutoff and seal['review_sha256'] == review_sha
                and seal['combined_audit_sha256'] == ref(root / 'combined_audit.json')['sha256'],
                'Actual root module review/seal differs: ' + task)
        item = {'phase': ref(CAP / task / 'public_dev_result.json'),
            'plan': ref(CAP / task / 'public_dev_job.json'),
            'full': ref(CAP / task / 'evaluation/public_dev/behavior_full.json'),
            'audits': [ref(root / alias / 'sol_response/mechanical_audit.json') for alias in ('e1', 'e2')],
            'combined_audit': ref(root / 'combined_audit.json'), 'review': ref(review),
            'observer_version': 'true_rl_agent.visual_observer.v3', 'as_of_cutoff': cutoff,
            'as_of_cutoff_ref': {**ref(seal_path), 'pointer': '/recorded_at_utc'},
            'qualified_case': checked(old['tasks'][task]['qualified_case']),
            'qualified_spec': checked(old['tasks'][task]['qualified_spec'])}
        if task == 'push':
            item.update({key: checked(old['tasks'][task][key]) for key in ('source_manifest', 'training_summary')})
        ready['tasks'][task] = item
    OUT.mkdir(parents=True, exist_ok=False)
    save(C / 'delivery_inputs.ready.json', ready)
    argv = [sys.executable, '-B', str(DRIVER), '--inputs', str(C / 'delivery_inputs.ready.json'),
            '--source-root', str(S), '--material-root', str(MATERIAL), '--output', str(DELIVERY)]
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=f'{S}/src:{S}:{P}')
    receipt = {'schema_version': 'p3new.n5_offline_assembly_execution.v1', 'status': 'NOT_RELEASED',
        'execution_status': 'IN_PROGRESS', 'started_at_utc': now(), 'script': ref(Path(__file__)),
        'assembler': ref(DRIVER), 'inputs': ref(C / 'delivery_inputs.ready.json'),
        'old_qualification_reference_registry': ref(old_path), 'argv': argv, 'cwd': str(S),
        'new_model_calls': 0, 'author_calls': 0, 'gpu_calls': 0, 'automatic_retries': 0,
        'formal_judge_reports_created': False, 'semantic_decisions_created': False}
    save(OUT / 'started.json', receipt)
    try:
        with (OUT / 'stdout.log').open('xb') as stdout, (OUT / 'stderr.log').open('xb') as stderr:
            process = subprocess.Popen(argv, cwd=S, env=env, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr)
            code = process.wait()
        receipt.update(returncode=code, child_pid=process.pid, process_finished_at_utc=now(),
                       stdout=ref(OUT / 'stdout.log'), stderr=ref(OUT / 'stderr.log'))
        require(code == 0, 'Frozen assembler failed; keep all output')
        assembly = read(DELIVERY / 'assembly_receipt.json')
        require(assembly['status'] == 'offline_serialized_not_sent' and assembly['model_calls'] == 0
                and assembly['new_author_calls'] == 0 and assembly['gpu_calls'] == 0
                and assembly['semantic_decisions_created'] is False, 'Offline status differs')
        results = {}
        for task, row in assembly['tasks'].items():
            require(row['nonvisual_unchanged'] is True and row['public_bytes'] <= row['public_limit_bytes']
                    and row['public_limit_bytes'] == {'stair': 8192, 'push': 12288}[task]
                    and row['as_of_cutoff'] == SEALS[task][0], 'Report bound/cutoff changed')
            for key in ('public', 'http_request', 'payload', 'full', 'phase'):
                checked(row[key])
            results[task] = {key: row[key] for key in ('public', 'http_request', 'payload', 'public_bytes',
                'public_limit_bytes', 'as_of_cutoff', 'as_of_cutoff_ref', 'shown_S3', 'nonvisual_unchanged',
                'qualified_parent_source_sha256')}
        receipt.update(execution_status='COMPLETED', completed_at_utc=now(),
            assembly_receipt=ref(DELIVERY / 'assembly_receipt.json'), tasks=results,
            engineering_status='OFFLINE_SERIALIZATION_CHECKED_NOT_SENT',
            limit='No actual author delivery, formal semantic review, new qualification or training evidence.')
        save(OUT / 'execution_receipt.json', receipt)
        print(json.dumps({'execution_status': 'COMPLETED', 'assembly_status': assembly['status'],
                          'tasks': {task: {key: row[key] for key in ('public_bytes', 'public_limit_bytes', 'shown_S3',
                              'nonvisual_unchanged', 'as_of_cutoff')} for task, row in results.items()}, 'model_calls': 0}), flush=True)
        return 0
    except Exception as error:
        receipt.update(execution_status='ERROR', stopped_at_utc=now(), error_type=type(error).__name__, error=str(error))
        with (OUT / 'exception.traceback.txt').open('x', encoding='utf-8') as handle:
            traceback.print_exc(file=handle)
        save(OUT / 'execution_receipt.json', receipt)
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
