"""R2 four-unit media facts preparation and control-only report ABI routing.

Uses actual frozen selections. No model call, report request, or semantic verdict.
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
CAP, P = B / 'n5_confirmation_20260930_r2', S / 'PLAN/feedback_sys/P3NEW'
OUT, MEDIA = C / 'postcapture/media', CAP / 'media_materials'
CARD_SHA = '6bea98ed63e1f1e516631e58205e3da93406311d1b87466e445fe754b0c22794'


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_bytes())


def ref(path):
    return {'file': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def save(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write('\n')


def main():
    card = C / 'capture_card.json'
    require(ref(card)['sha256'] == CARD_SHA, 'R2 card changed')
    source = read(C / 'runtime_source_manifest.json')
    for filename in ('prepare_confirmation_media.py', 'prepare_native_observer.py', 'export_confirmation_raw.py',
                     'prepare_confirmation_materials.py', 'run_confirmation_review.py', 'prepare_review_task_contract.py'):
        require(ref(P / filename)['sha256'] == source['PLAN/feedback_sys/P3NEW/' + filename]['sha256'],
                'Frozen helper changed: ' + filename)
    require(ref(C / 'count_prepared_text.py')['sha256'] ==
            '475fa8cf776c335010e6720733f947d25e15329a02df5003f399338dd4d80b52', 'Counter changed')
    require(all(not path.exists() for path in (OUT, MEDIA, C / 'media_inputs.json')), 'Preserve previous output')
    config = {'purpose': 'confirmation', 'capture_card': ref(card), 'selections': {}, 'units': []}
    selections, selection_refs = {}, {}
    for task in ('stair', 'push'):
        require(read(C / f'{task}.exit.json')['returncode'] == 0, 'Capture not completed: ' + task)
        path = CAP / 'observer_selection' / task / 'selection.json'
        selection, selection_card = read(path), read(path.with_name('card.json'))
        require(selection_card['observer_profile'] == 'READY_FOR_PREPARATION'
                and selection_card['pool_complete'] is True and not selection_card['missing_classes'],
                'Media selection incomplete: ' + task)
        require(selection_card['selection_sha256'] == ref(path)['sha256']
                and selection['source']['capture_card_sha256'] == CARD_SHA, 'Selection binding differs')
        observer = read(CAP / 'observers' / task / 'report.json')
        require(observer['status'] == 'PREPARED_NOT_RELEASED' and observer['selection_sha256'] == ref(path)['sha256']
                and observer['provider_requests'] == 0, 'Observer preparation not completed: ' + task)
        config['selections'][task] = ref(path)
        selections[task] = selection
        selection_refs[task] = {'selection': ref(path), 'card': ref(path.with_name('card.json')),
                                'actual_selection_driver_sha256': selection_card['driver_sha256']}
        require([row['alias'] for row in selection['episodes']] == ['e1', 'e2'], 'Unexpected aliases')
        for selected in selection['episodes']:
            config['units'].append({'id': f"media-{len(config['units']) + 1:02d}",
                'export': str(CAP / 'exports' / task), 'run_index': selected['run_index'],
                'env_index': selected['env_index'],
                'full': ref(CAP / task / 'evaluation/public_dev/behavior_full.json')})
    OUT.mkdir(parents=True, exist_ok=False)
    save(C / 'media_inputs.json', config)
    commands = [
        ('media_facts', [P / 'prepare_confirmation_media.py', '--inputs', C / 'media_inputs.json',
                         '--output', MEDIA, '--protocol', P / 'N5_FACT_REVIEW_PROTOCOL_ZH.md']),
        ('static_text_count', [C / 'count_prepared_text.py', '--prepared-root', MEDIA,
                               '--output', OUT / 'prepared_text_counts.json']),
    ]
    commands = [(name, [sys.executable, '-B', *map(str, argv)]) for name, argv in commands]
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=f'{S}/src:{S}:{P}')
    receipt = {'status': 'NOT_RELEASED', 'execution_status': 'IN_PROGRESS', 'started_at_utc': now(),
        'script': ref(Path(__file__).resolve()), 'inputs': ref(C / 'media_inputs.json'),
        'selection_sources': selection_refs, 'commands': commands, 'steps': [],
        'model_requests': 0, 'gpu_runs': 0, 'report_requests': 0, 'semantic_review_performed': False,
        'root_or_observer_answers_read': False, 'automatic_retries': 0}
    save(OUT / 'commands.json', receipt)
    try:
        for name, argv in commands:
            step = {'name': name, 'started_at_utc': now(), 'argv': argv}
            save(OUT / f'{name}.started.json', step)
            with (OUT / f'{name}.stdout.log').open('xb') as stdout, (OUT / f'{name}.stderr.log').open('xb') as stderr:
                process = subprocess.Popen(argv, cwd=S, env=env, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr)
                code = process.wait()
            step.update(returncode=code, child_pid=process.pid, finished_at_utc=now(),
                        stdout=ref(OUT / f'{name}.stdout.log'), stderr=ref(OUT / f'{name}.stderr.log'))
            save(OUT / f'{name}.exit.json', step)
            receipt['steps'].append(step)
            print(json.dumps({'step': name, 'returncode': code}), flush=True)
            require(code == 0, name + ' failed; preserve outputs')
        media_receipt = read(MEDIA / 'confirmation_media_receipt.json')
        counts = read(OUT / 'prepared_text_counts.json')
        require(len(media_receipt['units']) == 4 and counts['requests_counted'] == 8
                and all(row['stage'] == 'facts' for row in counts['records']), 'Expected four units / eight facts requests')
        definitions = [(path, read(path)) for path in sorted((C / 'numeric_task_definitions').glob('*.json'))]
        verified_definitions = read(CAP / 'numeric_materials/packing_receipt.json')['task_definitions']['contracts']
        mappings = []
        for unit in media_receipt['units']:
            task, run, env_index = unit['task'], unit['source_run'], unit['source_env']
            selected, = [row for row in selections[task]['episodes']
                         if (row['run_index'], row['env_index']) == (run, env_index)]
            numeric_source_path = MEDIA / unit['unit_id'] / 'raw/source_numeric.json'
            numeric_source = read(numeric_source_path)
            original = Path(numeric_source['manifest']['file'])
            require(ref(original) == numeric_source['manifest'], 'Media numeric original changed')
            numeric = read(original)
            require(numeric['identity'] == numeric_source['identity'], 'Media numeric identity differs')
            if task == 'push':
                require(numeric['identity']['run_index'] == run, 'Push source run differs')
            else:
                require(run in [row['run_index'] for row in numeric['metadata']['seeds']], 'Stair source run absent')
            contract_path, contract = next((path, value) for path, value in definitions
                if value['task'] == task and value['numeric_manifest']['sha256'] == numeric_source['manifest']['sha256'])
            require(sum(value['task'] == task and value['numeric_manifest']['sha256'] == numeric_source['manifest']['sha256']
                        for _, value in definitions) == 1, 'Ambiguous media task definition')
            require(any(item['file'] == str(contract_path) and item['sha256'] == ref(contract_path)['sha256']
                        for item in verified_definitions), 'Definition differs from the verified numeric packing input')
            definition_ref = {'id': f"task-definition-{task}-{numeric_source['manifest']['sha256'][:12]}",
                              'role': 'task_definition', 'mime_type': 'application/json', **ref(contract_path)}
            mappings.append({'packet_id': unit['unit_id'], 'task': task, 'source_run': run, 'source_env': env_index,
                'original_selection_alias': selected['alias'], 'facts_packet_alias': unit['packet_alias'],
                'facts_manifest': unit['facts_manifest'], 'source_numeric_control_ref': ref(numeric_source_path),
                'original_numeric_manifest': numeric_source['manifest'], 'report_task_definition': definition_ref,
                'facts_task_definition_supplied': False, 'report_task_definition_supplied': False,
                'report_mapping_only_not_sent': True})
        mapping_path = OUT / 'report_task_definition_alias_mapping.json'
        save(mapping_path, {'schema_version': 'p3new.n5_media_report_task_definition_mapping.v1',
            'recorded_at_utc': now(), 'status': 'CONTROL_MAPPING_NOT_SENT', 'inputs': ref(C / 'media_inputs.json'),
            'media_receipt': ref(MEDIA / 'confirmation_media_receipt.json'), 'mappings': mappings,
            'stage_boundary': 'Frozen media facts contain measurement definitions and originals, not candidate permission judgment. Later report manifests append this independent same-task/run definition using frozen prepare_report_manifest and retain all facts materials; actual report preparation remains pending.',
            'not_a_permission_verdict': True, 'model_requests': 0, 'root_review_supplied_to_facts': False})
        receipt.update(execution_status='COMPLETED', completed_at_utc=now(),
            media_receipt=ref(MEDIA / 'confirmation_media_receipt.json'),
            prepared_text_counts=ref(OUT / 'prepared_text_counts.json'), report_mapping=ref(mapping_path))
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
