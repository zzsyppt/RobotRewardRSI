"""Bind actual R4 module seals for one offline serialization; never send a request."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

B = Path('/data/yuyangliu/AGENT/feedback_p3new_20260929')
S, C = B / 'n5_confirmation_source_r4_final', B / 'n5_execution_20261001_r4'
CAP, MATERIAL = B / 'n5_confirmation_20261001_r4', B / 'text_author_smoke_r1'
P = S / 'PLAN/feedback_sys/P3NEW'
DRIVER = P / 'results/n5_confirmation_execution_r1/assemble_delivery.py'
OUT, DELIVERY = C / 'postcapture/delivery_original_r1', CAP / 'delivery_offline_r1'
EXPECTED_CARD_SHA256 = '811d7f140a5b3015f66a5093ba516c60bdc4a05c9ac827461d19bb05b8be3041'


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


def matched(reference, path):
    """Bind unchanged desktop-origin receipts to their known staged remote copies."""
    actual = ref(path)
    require(actual['sha256'] == reference['sha256'] and
            actual['size_bytes'] == reference.get('size_bytes', actual['size_bytes']),
            'Staged evidence differs: ' + str(path))
    return actual


def module_inputs(task, seal_path, card_ref, manifest_ref, frozen):
    root = CAP / 'observers' / task
    require(seal_path.resolve() == root / 'root_module_review_receipt.json', 'Unexpected receipt root: ' + task)
    seal, review = read(seal_path), read(root / 'root_module_review.json')
    combined = read(root / 'combined_audit.json')
    controls = [ref(seal_path), matched(seal['review'], root / 'root_module_review.json'),
                matched(seal['combined_audit'], root / 'combined_audit.json'),
                matched(seal['access_receipt'], root / 'root_module_access_receipt.json')]
    require(combined['scope'] == 'n5_confirmation' and combined['task'] == task and
            combined['capture_card_sha256'] == card_ref['sha256'] and
            combined['source_manifest_sha256'] == manifest_ref['sha256'], 'Capture/source binding differs: ' + task)
    module_relative = 'src/true_rl_agent/evaluation/visual_review.py'
    require(seal['visual_review_module']['sha256'] == frozen[module_relative]['sha256'] and
            seal['visual_review_binding'] == review['source_sha256'] and
            seal['criterion'] == review['criterion'] and seal['reviewer'] == review['reviewer'],
            'Root review/source seal differs: ' + task)
    cutoff = seal['recorded_at_utc']
    sealed_at = datetime.fromisoformat(cutoff.replace('Z', '+00:00'))
    require(sealed_at.utcoffset() is not None, 'Review cutoff has no timezone')
    binding_path, access_path = root / 'desktop_invocations.trace_binding.json', root / 'desktop_access_audit.json'
    binding, access = read(binding_path), read(access_path)
    controls.extend([ref(binding_path), ref(access_path),
                     matched(binding['original_registration'], root / 'desktop_invocations.json')])
    matched(combined['desktop_invocations'], root / 'desktop_invocations.json')
    require(binding['schema_version'] == 'p3new.actual_observer_trace_binding.v1' and
            binding['original_registration_unchanged'] is True and
            access['status'] == 'PASS_OBSERVABLE_ACCESS_AND_TRANSPORT_NOT_SEMANTIC', 'Trace audit not complete: ' + task)
    traces, audits, calls = binding['invocations'], access['checks'], combined['calls']
    require([[x[key] for x in rows] for rows, key in
             ((traces, 'alias'), (audits, 'alias'), (calls, 'episode_alias'))] == [['e1', 'e2']] * 3,
            'Actual observer aliases differ: ' + task)
    for trace, audit, call in zip(traces, audits, calls):
        alias = trace['alias']
        require(trace['actual_agent_path'] == trace['registered_agent_path'] == call['registered_agent'] and
                trace['actual_rollout_uuid'] == audit['rollout_uuid'] and
                datetime.fromisoformat(trace['task_completed_at_utc'].replace('Z', '+00:00')) <= sealed_at,
                'Actual observer identity/completion differs: ' + task + '/' + alias)
        require(audit['model_and_effort'] == {'model': 'gpt-6.1-sol', 'effort': 'high'} and
                audit['image_returns_match_original_png_bytes_and_dimensions'] is True and
                audit['current_outputs_match_hashes_returned_at_original_write'] is True and
                audit['semantic_review_performed'] is False, 'Observable transport audit differs')
        for key, relative, call_key in (
                ('raw_answer', 'sol_response/raw_answer.txt', 'raw_answer'),
                ('access_receipt', 'sol_response/access_receipt.json', 'access_receipt'),
                ('input_manifest', 'blind_inputs/materials.json', 'materials')):
            path = root / alias / relative
            controls.append(matched(trace[key], path))
            matched(audit[key], path)
            matched(call[call_key], path)
        for key, filename in (('tool_trace', 'observable_tool_trace.json'),
                              ('tool_outputs', 'observable_tool_outputs.json'),
                              ('returned_images', 'returned_image_metadata.json')):
            controls.append(matched(audit[key], root / alias / 'desktop_trace' / filename))
    # Production attach_visual_review in the unchanged assembler validates all decisions.
    return cutoff, controls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stair-review-receipt', type=Path, required=True)
    parser.add_argument('--push-review-receipt', type=Path, required=True)
    args = parser.parse_args()
    seals = {'stair': args.stair_review_receipt, 'push': args.push_review_receipt}
    require(all(path.is_file() for path in seals.values()), 'Both actual module seals are required; no assembly started')
    require(not OUT.exists() and not DELIVERY.exists() and not (C / 'delivery_inputs.ready.json').exists(),
            'Preserve any previous attempt')
    manifest_ref = ref(C / 'runtime_source_manifest.json')
    frozen = read(C / 'runtime_source_manifest.json')
    require(ref(DRIVER)['sha256'] == frozen[DRIVER.relative_to(S).as_posix()]['sha256'], 'Frozen assembler changed')
    for relative, expected in frozen.items():
        require(ref(S / relative)['sha256'] == expected['sha256'], 'Frozen source differs: ' + relative)
    card_ref = ref(C / 'capture_card.json')
    require(card_ref['sha256'] == EXPECTED_CARD_SHA256, 'Capture card changed')
    card = read(C / 'capture_card.json')
    require(card['source']['remote_root'] == str(S) and card['output_root'] == str(CAP) and
            card['execution_control_root'] == str(C) and card['source']['manifest']['sha256'] == manifest_ref['sha256'] and
            card['source']['file_count'] == len(frozen), 'Card/source/output binding differs')
    require(card['materials']['bounded_request_origin'] == 'offline_serialized_not_sent' and
            card['materials']['new_author_calls'] == 0, 'Offline delivery contract differs')
    old_path = B / 'n5_execution_20260930/engineering_delivery_inputs.ready.json'
    old = read(old_path)
    ready = {'schema_version': 'p3new.n5_offline_delivery_inputs.v1',
             'status': 'READY_FOR_OFFLINE_SERIALIZATION_NOT_RELEASED',
             'scope': 'n5_confirmation', 'independent_semantic_acceptance_claimed': False, 'capture_card': card_ref,
             'qualification_reader': checked(old['qualification_reader']),
             'qualification_index': checked(old['qualification_index']), 'tasks': {}}
    bound_controls = []
    for task in ('stair', 'push'):
        root = CAP / 'observers' / task
        review, seal_path = root / 'root_module_review.json', seals[task]
        cutoff, controls = module_inputs(task, seal_path, card_ref, manifest_ref, frozen)
        bound_controls.extend(controls)
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
        'scope': 'n5_confirmation', 'independent_semantic_acceptance_claimed': False,
        'execution_status': 'IN_PROGRESS', 'started_at_utc': now(), 'script': ref(Path(__file__)),
        'assembler': ref(DRIVER), 'inputs': ref(C / 'delivery_inputs.ready.json'),
        'old_qualification_reference_registry': ref(old_path), 'argv': argv, 'cwd': str(S),
        'capture_card': card_ref, 'source_manifest': manifest_ref, 'actual_module_and_trace_inputs': bound_controls,
        'new_model_calls': 0, 'API_calls': 0, 'author_calls': 0, 'new_author_calls': 0,
        'gpu_calls': 0, 'automatic_retries': 0,
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
                    and row['as_of_cutoff'] == ready['tasks'][task]['as_of_cutoff'], 'Report bound/cutoff changed')
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
            # Rejected count comes only from these actual reviewed observations, never an R3 constant.
            results[task] = {key: row[key] for key in ('public', 'http_request', 'payload', 'public_bytes',
                'public_limit_bytes', 'as_of_cutoff', 'as_of_cutoff_ref', 'shown_S3', 'nonvisual_unchanged',
                'qualified_parent_source_sha256')}
            results[task]['excluded_original_observations'] = rejected
            results[task]['actual_module_decision_counts'] = {
                verdict: sum(item['verdict'] == verdict for item in decisions)
                for verdict in ('supported', 'contradicted', 'insufficient')}
            results[task]['raw_observation_count'] = len(combined['observations'])
            results[task]['budget'] = row['budget']
        for control in bound_controls + [card_ref, manifest_ref]:
            checked(control)
        for name, item in assembly['loaded_production_modules'].items():
            relative = Path(item['path']).relative_to(S).as_posix()
            require(item['sha256'] == frozen[relative]['sha256'], 'Loaded frozen module differs: ' + name)
        for relative, expected in frozen.items():
            require(ref(S / relative)['sha256'] == expected['sha256'], 'Frozen source changed during assembly: ' + relative)
        receipt.update(status='offline_serialized_not_sent', execution_status='COMPLETED', completed_at_utc=now(),
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
