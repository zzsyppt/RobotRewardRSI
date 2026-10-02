"""Bind original R3 evidence for development/INCONCLUSIVE offline serialization once."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

B = Path('/data/yuyangliu/AGENT/feedback_p3new_20260929')
S, C = B / 'n5_confirmation_source_r3_final', B / 'n5_execution_20260930_r3'
CAP, MATERIAL = B / 'n5_confirmation_20260930_r3', B / 'text_author_smoke_r1'
P = S / 'PLAN/feedback_sys/P3NEW'
DRIVER = P / 'results/n5_confirmation_execution_r1/assemble_delivery.py'
OUT, DELIVERY = C / 'postcapture/delivery_original_r1', CAP / 'delivery_offline_r1'
SEALS = {
    'stair': ('2026-09-30T16:15:15.187796+00:00',
              'ceeab9f071e9afaf80ef698768a9b5915509b3d3b34ebfaf81434604b10080dd',
              '4a58ea02edc234de7626a892353601b6fb3ee22cc9a0ac11295738b6578a1bf6'),
    'push': ('2026-09-30T16:52:32.792132+00:00',
             'b8995ab2c4265da663701de148d4e32312fc89fe8ff3533caba3ca6885aaefdb',
             '41a8a8896f5908add6dc41e2c80764afa2e63f79e3e05e5dfc713d2428e809cd'),
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


def contains_text(value, needle):
    if isinstance(value, str):
        return needle in value
    if isinstance(value, dict):
        return any(contains_text(item, needle) for item in value.values())
    if isinstance(value, list):
        return any(contains_text(item, needle) for item in value)
    return False


def main():
    require(not OUT.exists() and not DELIVERY.exists() and not (C / 'delivery_inputs.ready.json').exists(),
            'Preserve any previous attempt')
    frozen = read(C / 'runtime_source_manifest.json')
    require(ref(DRIVER)['sha256'] == frozen[DRIVER.relative_to(S).as_posix()]['sha256'], 'Frozen assembler changed')
    for relative, expected in frozen.items():
        require(ref(S / relative)['sha256'] == expected['sha256'], 'Frozen source differs: ' + relative)
    require(ref(C / 'capture_card.json')['sha256'] ==
            '5c97844ff53027b80991a7308f991960e013df52fbc268046721e9771d42abc4', 'Capture card changed')
    old_path = B / 'n5_execution_20260930/engineering_delivery_inputs.ready.json'
    old = read(old_path)
    ready = {'schema_version': 'p3new.n5_offline_delivery_inputs.v1',
             'status': 'READY_FOR_OFFLINE_SERIALIZATION_NOT_RELEASED',
             'scope': 'development_INCONCLUSIVE', 'confirmation_admitted': False, 'capture_card': ref(C / 'capture_card.json'),
             'qualification_reader': checked(old['qualification_reader']),
             'qualification_index': checked(old['qualification_index']), 'tasks': {}}
    for task in ('stair', 'push'):
        root = CAP / 'observers' / task
        review, seal_path = root / 'root_module_review.json', root / 'root_module_review_receipt.json'
        seal = read(seal_path)
        cutoff, review_sha, seal_sha = SEALS[task]
        require(ref(review)['sha256'] == review_sha and ref(seal_path)['sha256'] == seal_sha
                and seal['recorded_at_utc'] == cutoff and seal['review']['sha256'] == review_sha
                and seal['combined_audit']['sha256'] == ref(root / 'combined_audit.json')['sha256'],
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
        'scope': 'development_INCONCLUSIVE', 'confirmation_admitted': False,
        'execution_status': 'IN_PROGRESS', 'started_at_utc': now(), 'script': ref(Path(__file__)),
        'assembler': ref(DRIVER), 'inputs': ref(C / 'delivery_inputs.ready.json'),
        'old_qualification_reference_registry': ref(old_path), 'argv': argv, 'cwd': str(S),
        'new_model_calls': 0, 'API_calls': 0, 'author_calls': 0, 'gpu_calls': 0, 'automatic_retries': 0,
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
            public, payload = read(Path(row['public']['path'])), read(Path(row['payload']['path']))
            combined = read(Path(ready['tasks'][task]['combined_audit']['path']))
            decisions = read(Path(ready['tasks'][task]['review']['path']))['decisions']
            rejected = []
            for decision in decisions:
                if decision['verdict'] == 'supported':
                    continue
                observation = combined['observations'][decision['observation_index']]
                text = observation['raw_observation']['text']
                require(not contains_text(public, text) and not contains_text(payload, text),
                        'Non-admitted original observation leaked into serialized author content')
                rejected.append({'observation_index': decision['observation_index'],
                    'reference': observation['reference'], 'original_verdict': decision['verdict'],
                    'original_text_sha256': hashlib.sha256(text.encode('utf-8')).hexdigest(),
                    'absent_from_public_and_author_payload': True})
            require(len(rejected) == {'stair': 2, 'push': 0}[task], 'Original decision accounting differs')
            results[task] = {key: row[key] for key in ('public', 'http_request', 'payload', 'public_bytes',
                'public_limit_bytes', 'as_of_cutoff', 'as_of_cutoff_ref', 'shown_S3', 'nonvisual_unchanged',
                'qualified_parent_source_sha256')}
            results[task]['excluded_original_observations'] = rejected
            results[task]['budget'] = row['budget']
        for name, item in assembly['loaded_production_modules'].items():
            relative = Path(item['path']).relative_to(S).as_posix()
            require(item['sha256'] == frozen[relative]['sha256'], 'Loaded frozen module differs: ' + name)
        for relative, expected in frozen.items():
            require(ref(S / relative)['sha256'] == expected['sha256'], 'Frozen source changed during assembly: ' + relative)
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
