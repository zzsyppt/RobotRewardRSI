"""Execute the reviewed media control correction once, then check actual artifacts."""
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
FIX = C / 'postcapture_media_fix'
OUT, MEDIA = C / 'postcapture/media_control_corrected', CAP / 'media_materials_control_corrected'


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_bytes())


def sha(data):
    return hashlib.sha256(data).hexdigest()


def ref(path):
    return {'file': str(path.resolve()), 'sha256': sha(path.read_bytes())}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def save(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write('\n')


def main():
    require(ref(FIX / 'prepare_confirmation_media.py')['sha256'] ==
        'e3f733a0e8d67be9bd24abce18c9f7ae1ebd2c317452d8c9b5b37eacff5491d3', 'Reviewed correction differs')
    require(ref(P / 'prepare_confirmation_media.py')['sha256'] ==
        '8f757979f28f993e3c1d34ad198765850e9f43f07ef249bf245e3f33df5b3d37', 'Original frozen source changed')
    require(ref(FIX / 'registration.json')['sha256'] ==
        'a041e92ab0ab03070d1ccea60a377598a8a5bbc53f0a7fd49990f003fbbfec52', 'Correction registration differs')
    require(ref(C / 'media_inputs.json') == read(C / 'postcapture/media/commands.json')['inputs'], 'Original media inputs changed')
    require(not OUT.exists() and not MEDIA.exists(), 'New output required; preserve all previous attempts')
    OUT.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=f'{S}/src:{S}:{P}')
    commands = [
        ('media_facts', [FIX / 'prepare_confirmation_media.py', '--inputs', C / 'media_inputs.json',
                        '--output', MEDIA, '--protocol', P / 'N5_FACT_REVIEW_PROTOCOL_ZH.md']),
        ('static_text_count', [C / 'count_prepared_text.py', '--prepared-root', MEDIA,
                              '--output', OUT / 'prepared_text_counts.json']),
    ]
    commands = [(name, [sys.executable, '-B', *map(str, argv)]) for name, argv in commands]
    receipt = {'schema_version': 'p3new.n5_corrected_media_execution.v1',
        'status': 'NOT_RELEASED', 'execution_status': 'IN_PROGRESS', 'started_at_utc': now(),
        'script': ref(Path(__file__)), 'corrected_generator': ref(FIX / 'prepare_confirmation_media.py'),
        'source_registration': ref(FIX / 'registration.json'), 'inputs': ref(C / 'media_inputs.json'),
        'previous_failed_receipt': ref(C / 'postcapture/media/execution_receipt.json'),
        'commands': commands, 'steps': [], 'model_requests': 0, 'gpu_runs': 0, 'automatic_retries': 0,
        'reports_or_facts_seals_created': False, 'observer_or_root_answers_read': False,
        'semantic_or_capacity_release': False}
    save(OUT / 'commands.json', receipt)
    try:
        for name, argv in commands:
            step = {'name': name, 'argv': argv, 'started_at_utc': now()}
            save(OUT / f'{name}.started.json', step)
            with (OUT / f'{name}.stdout.log').open('xb') as stdout, (OUT / f'{name}.stderr.log').open('xb') as stderr:
                process = subprocess.Popen(argv, cwd=S, env=env, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr)
                code = process.wait()
            step.update(returncode=code, child_pid=process.pid, finished_at_utc=now(),
                        stdout=ref(OUT / f'{name}.stdout.log'), stderr=ref(OUT / f'{name}.stderr.log'))
            save(OUT / f'{name}.exit.json', step)
            receipt['steps'].append(step)
            print(json.dumps({'step': name, 'returncode': code}), flush=True)
            require(code == 0, name + ' failed; retain evidence')

        sys.path[:0] = [str(S / 'src'), str(P), str(S)]
        import run_confirmation_review as runner
        media_receipt = read(MEDIA / 'confirmation_media_receipt.json')
        proof = read(FIX / 'native_input_comparison.json')
        proof_rows = {(row['task'], row['alias']): row for row in proof['units']}
        counts = read(OUT / 'prepared_text_counts.json')
        require(len(media_receipt['units']) == 4 and counts['requests_counted'] == 8
                and all(row['stage'] == 'facts' for row in counts['records']), 'Expected four units/eight facts requests')
        contracts = [(path, read(path)) for path in sorted((C / 'numeric_task_definitions').glob('*.json'))]
        packing_definitions = read(CAP / 'numeric_materials/packing_receipt.json')['task_definitions']['contracts']
        mappings, checks = [], []
        for unit in media_receipt['units']:
            task, alias, packet_id = unit['task'], unit['alias'], unit['unit_id']
            dest, actual_observer = MEDIA / packet_id, CAP / 'observers' / task / alias
            prior = proof_rows[task, alias]
            require((unit['source_run'], unit['source_env'], unit['selected_steps']) ==
                    (prior['source_run'], prior['source_env'], prior['selected_steps']), 'Unit identity/timepoints differ')
            require(unit['packet_alias'] == alias, 'Receipt alias differs')
            require((dest / 'observer_selection.json').read_bytes() ==
                    (CAP / 'observer_selection' / task / 'selection.json').read_bytes(), 'Original v2 selection changed')
            public = read(dest / 'observer_blind_input.json')
            encoded = json.dumps(public, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
            require(sha(encoded) == prior['canonical_complete_public_window_sha256'], 'Complete public window differs')
            metadata, raw = read(dest / 'media.json'), read(dest / 'raw/related_numeric.json')
            require(metadata['alias'] == raw['episode']['episode_id'] == read(dest / 'questions.json')['alias'] == alias,
                    'Numeric/media/question alias differs')
            require(metadata['observer_blind_input'] == public and raw['episode']['task'] == task, 'Metadata/identity differs')
            supplied = read(actual_observer / 'blind_inputs/materials.json')['materials']
            originals = [row for row in supplied if row['role'] == 'original_image']
            require(len(metadata['images']) == len(originals) == prior['image_count'], 'Image count differs')
            for row, original in zip(metadata['images'], originals, strict=True):
                require(row['alias'] == alias and (row['step'], row['view_id'], row['sha256']) ==
                        (original['step'], original['view_id'], original['sha256']), 'Image identity/order differs')
                path = dest / f"{alias}-step{row['step']:06d}-{row['view_id']}.png"
                require(path.read_bytes() == (actual_observer / 'blind_inputs' / original['path']).read_bytes(),
                        'Original PNG bytes differ')
            facts_path = dest / 'facts.manifest.json'
            facts = read(facts_path)
            require(facts['stage'] == 'facts' and facts['packet_id'] == packet_id and
                {row['role'] for row in facts['materials']} ==
                {'review_protocol', 'focus_questions', 'raw_numeric', 'raw_media_metadata', 'raw_image'},
                'Unexpected media facts roles or stage')
            common, variants = None, []
            for judge in ('deepseek', 'gpt6-sol'):
                prepared = dest / 'prepared_facts' / judge
                body, _, card = runner.build(facts_path, judge)
                require(body == (prepared / 'request.http_body.json').read_bytes() and card == read(prepared / 'prepare.json'),
                        'Whole prepared facts request differs from frozen rebuild')
                parts = json.loads(body)['messages'][1]['content']
                require(common is None or parts == common, 'Facts evidence bytes/order differ between judges')
                common = parts
                variants.append({'judge': judge, 'request': ref(prepared / 'request.http_body.json'),
                                 'request_bytes': len(body), 'png_count': card['png_count']})
            numeric_source_path = dest / 'raw/source_numeric.json'
            numeric_source = read(numeric_source_path)
            numeric_path = Path(numeric_source['manifest']['file'])
            require(ref(numeric_path) == numeric_source['manifest'], 'Media source manifest changed')
            numeric = read(numeric_path)
            require(numeric['identity'] == numeric_source['identity'], 'Numeric identity differs')
            if task == 'push':
                require(numeric['identity']['run_index'] == unit['source_run'], 'Push run differs')
            else:
                require(unit['source_run'] in [row['run_index'] for row in numeric['metadata']['seeds']], 'Stair run absent')
            matches = [(path, value) for path, value in contracts if value['task'] == task
                       and value['numeric_manifest']['sha256'] == numeric_source['manifest']['sha256']]
            require(len(matches) == 1, 'Task/numeric definition is not unique')
            contract_path, contract = matches[0]
            require(any(item['file'] == str(contract_path) and item['sha256'] == ref(contract_path)['sha256']
                        for item in packing_definitions), 'Definition differs from independently verified packing input')
            mappings.append({'packet_id': packet_id, 'task': task, 'source_run': unit['source_run'],
                'source_env': unit['source_env'], 'original_selection_alias': alias, 'facts_packet_alias': alias,
                'facts_manifest': ref(facts_path), 'numeric_source': ref(numeric_source_path),
                'original_numeric_manifest': numeric_source['manifest'],
                'report_task_definition': {'id': f"task-definition-{task}-{numeric_source['manifest']['sha256'][:12]}",
                    'role': 'task_definition', 'mime_type': 'application/json', **ref(contract_path)},
                'facts_task_definition_supplied': False, 'report_task_definition_supplied': False,
                'report_mapping_only_not_sent': True, 'permission_verified': False})
            checks.append({'packet_id': packet_id, 'task': task, 'alias': alias, 'image_count': len(originals),
                'complete_public_window_matches_actual_observer': True, 'original_pngs_bytes_order_match': True,
                'numeric_media_questions_aliases_match': True, 'source_task_numeric_sha_run_match': True,
                'whole_facts_requests_rebuilt_identically': True, 'facts_variants': variants})
        mapping_path = OUT / 'report_task_definition_alias_mapping.json'
        save(mapping_path, {'schema_version': 'p3new.n5_media_report_task_definition_mapping.v1',
            'recorded_at_utc': now(), 'status': 'CONTROL_MAPPING_NOT_SENT', 'mappings': mappings,
            'stage_boundary': 'Media facts contain measurement definitions and originals, not candidate permission judgment. Later report manifests append this independent same-task/numeric/run definition using frozen prepare_report_manifest and retain all facts materials. Report preparation and permission judgments remain pending.',
            'root_review_supplied_to_facts': False, 'model_requests': 0})
        receipt.update(execution_status='COMPLETED', completed_at_utc=now(), check_status='PASS_CHECKED_ENGINEERING_SCOPE',
            actual_artifact_checks=checks, media_receipt=ref(MEDIA / 'confirmation_media_receipt.json'),
            prepared_text_counts=ref(OUT / 'prepared_text_counts.json'), report_mapping=ref(mapping_path),
            native_input_comparison=ref(FIX / 'native_input_comparison.json'),
            static_text_summary={key: counts[key] for key in ('requests_counted', 'image_count', 'max_text_tokens',
                'max_reserved_text_proxy_total', 'unmeasured')})
        save(OUT / 'execution_receipt.json', receipt)
        print(json.dumps({'execution_status': 'COMPLETED', 'units': len(checks), 'requests': counts['requests_counted'],
                          'max_text_tokens': counts['max_text_tokens'], 'report_mapping': ref(mapping_path)}), flush=True)
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
