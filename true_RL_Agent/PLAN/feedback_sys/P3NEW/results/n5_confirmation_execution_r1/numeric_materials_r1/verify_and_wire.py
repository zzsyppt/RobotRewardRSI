"""Bounded offline N5 numeric checks and explicit task-definition references."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

BASE = Path('/data/yuyangliu/AGENT/feedback_p3new_20260929')
SOURCE = BASE / 'n5_postprocess_source_r1'
CAPTURE = BASE / 'n5_confirmation_20260930'
RAW = CAPTURE / 'numeric_raw'
PACK = CAPTURE / 'numeric_materials'
CONTROL = BASE / 'n5_execution_20260930/control/numeric_materials_r1'
sys.path.insert(0, str(SOURCE / 'src'))
sys.path.insert(0, str(SOURCE / 'PLAN/feedback_sys/P3NEW'))
import run_confirmation_review as runner
import verify_stair_confirmation_arithmetic as stair
import verify_push_confirmation_arithmetic as push


def read(path):
    return json.loads(path.read_bytes())


def ref(path):
    with path.open('rb') as handle:
        checksum = hashlib.file_digest(handle, 'sha256').hexdigest()
    return {'path': str(path.resolve()), 'sha256': checksum, 'bytes': path.stat().st_size}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
        handle.write('\n')


def main():
    preflight = read(CONTROL / 'preflight.json')
    source_manifest = read(Path(preflight['source_manifest']['path']))
    for path, entry in source_manifest['files'].items():
        assert ref(SOURCE / path)['sha256'] == entry['sha256'], path
    fulls = {}
    for task, original in zip(('stair', 'push'), preflight['fulls'], strict=True):
        path = Path(original['path'])
        assert ref(path) == original
        fulls[task] = read(path)
    raw_receipt = read(RAW / 'manifest.json')
    packing = read(PACK / 'packing_receipt.json')
    assert raw_receipt['purpose'] == packing['purpose'] == 'confirmation'
    assert packing['actual_calls'] == 0
    assert packing['source_sha256']['raw_manifest'] == ref(RAW / 'manifest.json')['sha256']
    private = read(RAW / '_private/sources.json')
    aliases = {(r['task'], r['episode_id']): r['original_id'] for r in private['anonymous_mapping']}
    indexed = {
        'stair': {row['episode_id']: (i, row) for i, row in enumerate(fulls['stair']['episodes'])},
        'push': {f"r{row['source']+1}.e{row['env_id']}": (i, row) for i, row in enumerate(fulls['push']['transcripts'])}}
    assert fulls['stair']['domain']['archive_hash'] == preflight['numeric_manifests'][0]['sha256']
    assert {row['manifest_sha256'] for row in fulls['push']['sources']} == {
        row['sha256'] for row in preflight['numeric_manifests'][1:]}
    arithmetic, wiring = [], []
    for unit in packing['packets']:
        task, packet = unit['task'], unit['packet_id']
        manifest_path = PACK / unit['facts_manifest']
        assert ref(manifest_path)['sha256'] == unit['facts_manifest_sha256']
        old = read(manifest_path)
        raw_ref, = [m for m in old['materials'] if m['role'] == 'raw_numeric']
        raw_path = (manifest_path.parent / raw_ref['file']).resolve()
        assert ref(raw_path)['sha256'] == raw_ref['sha256']
        raw = read(raw_path)
        if unit['axis'] == 'episode':
            logical_id = aliases[task, unit['scope']]
            index, episode = indexed[task][logical_id]
            shared = ({'domain': fulls[task]['domain'], 'semantics': fulls[task]['semantics'],
                       'contract': fulls[task]['semantic']['contract']} if task == 'stair' else
                      {key: fulls[task][key] for key in ('binding', 'comparison_domain', 'definitions', 'process_contract', 'source_bindings')})
            derivation = {'source': {'full': preflight['fulls'][0 if task == 'stair' else 1]},
                'source_pointer': f"/{'episodes' if task == 'stair' else 'transcripts'}/{index}",
                'episode_id': unit['scope'], 'original_logical_id': logical_id,
                'shared': shared, 'complete_episode_derivation': episode}
            derivation_path = CONTROL / 'arithmetic_inputs' / (packet + '.json')
            save(derivation_path, derivation)
            result = (stair.verify_stair_episode(raw, derivation) if task == 'stair' else
                      push.verify_push_confirmation_arithmetic(raw, derivation))
            result_path = CONTROL / 'arithmetic' / (packet + '.json')
            save(result_path, result)
            row = {'packet_id': packet, 'task': task, 'logical_id': logical_id,
                'raw_numeric': ref(raw_path), 'full_derivation': ref(derivation_path),
                'result': ref(result_path), 'status': result['status'],
                'checked_count': result['checked_n'] if task == 'stair' else len(result['checked_field_paths']),
                'failed_count': result['failed_n'] if task == 'stair' else len(result['differences'])}
            if task == 'stair':
                link = stair.verify_stair_operand_sources(raw,
                    episode_header_path=RAW / 'episodes' / unit['scope'] / 'episode.json',
                    raw_export_manifest_path=RAW / 'manifest.json', packing_receipt_path=PACK / 'packing_receipt.json')
                link_path = CONTROL / 'source_links' / (packet + '.json')
                save(link_path, link)
                row['source_link'] = {'status': link['status'], 'result': ref(link_path),
                    'failed_count': link.get('failed_n', sum(c['status'] == 'fail' for c in link['checks']))}
            else:
                row['source_link'] = {'status': 'NOT_AVAILABLE_IN_FROZEN_HELPER',
                    'limit': 'Push arithmetic authenticates supplied field bytes against supplied metadata only; raw export roundtrip is separate engineering evidence.'}
            arithmetic.append(row)
        labels = (['stair'] if task == 'stair' else
                  ['push_' + raw['episode']['run_group'].replace('run_', 'run')] if unit['axis'] == 'episode' else
                  ['push_run01', 'push_run02', 'push_run03'])
        added = []
        for label in labels:
            definition_path = PACK / 'task_definitions' / (label + '.json')
            definition_ref = ref(definition_path)
            assert read(definition_path)['task'] == task
            added.append({'id': 'task-definition-' + label, 'role': 'task_definition',
                'file': str(definition_path.resolve()), 'sha256': definition_ref['sha256'], 'mime_type': 'application/json'})
        updated = deepcopy(old)
        updated['materials'].extend(added)
        new_manifest_path = manifest_path.with_name('facts.task_definition.manifest.json')
        save(new_manifest_path, updated)
        variants, common_content = {}, None
        for judge in ('deepseek', 'gpt6-sol'):
            destination = PACK / 'prepared_with_task_definition' / packet / judge
            card = runner.prepare(new_manifest_path, judge, destination)
            request_path = destination / 'request.http_body.json'
            body = read(request_path)
            content = body['messages'][1]['content']
            if common_content is not None:
                assert content == common_content, packet
            common_content = content
            assert body['max_tokens'] == 32768
            size = len(body['messages'][0]['content'].encode('utf-8')) + sum(len(x['text'].encode('utf-8')) for x in content)
            variants[judge] = {'request': ref(request_path), 'card_request_sha256': card['request_sha256'],
                'model_visible_text_utf8_bytes': size, 'input_token_estimate_bytes_plus_1024': size + 1024,
                'output_reserve': 32768, 'additional_margin': 8192,
                'estimated_required_context_tokens': size + 1024 + 32768 + 8192,
                'actual_calls': 0, 'capacity_status': 'NOT_RELEASED_UNVERIFIED_ACTUAL_TOKEN_COUNT'}
        assert updated['materials'][:-len(added)] == old['materials']
        assert ref(manifest_path)['sha256'] == unit['facts_manifest_sha256']
        wiring.append({'packet_id': packet, 'task': task, 'axis': unit['axis'],
            'original_manifest': ref(manifest_path), 'new_manifest': ref(new_manifest_path),
            'added_task_definition_refs': added, 'all_original_material_refs_unchanged_in_original_order': True,
            'reviewer_materials_identical_in_order': True, 'variants': variants})
    assert len(arithmetic) == 24 and len(wiring) == 36
    for path, entry in source_manifest['files'].items():
        assert ref(SOURCE / path)['sha256'] == entry['sha256'], path
    for original in preflight['numeric_manifests'] + preflight['fulls'] + [preflight['capture_card'], preflight['review_protocol']]:
        assert ref(Path(original['path'])) == original
    failures = sum(row['failed_count'] + row['source_link'].get('failed_count', 0) for row in arithmetic)
    receipt = {'schema_version': 'p3new.n5_numeric_wiring_arithmetic.v1',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'FAIL_NOT_RELEASED' if failures else 'PREPARED_CHECKED_WITH_EXPLICIT_LIMITS_NOT_RELEASED',
        'package_confirmation_status': 'INCONCLUSIVE',
        'package_status_basis': 'Coordinator reported the frozen Push media selector excludes success=True/failure_label=success_recovered and failed observer preparation. This receipt does not re-audit media, repair that selection, or release confirmation. Frozen numeric methods and their results remain unchanged.',
        'model_calls': 0, 'gpu_calls': 0, 'facts_admitted': False,
        'facts_answers_or_observer_answers_read': False, 'formal_judgments_read': False,
        'source_files_verified_before_and_after': 445,
        'source_helpers': {module.__name__: ref(Path(module.__file__)) for module in (runner, stair, push)},
        'execution_script': ref(Path(__file__)), 'raw_export_manifest': ref(RAW / 'manifest.json'),
        'packing_receipt': ref(PACK / 'packing_receipt.json'),
        'original_source_manifest': ref(RAW / '_private/sources.json'),
        'arithmetic': arithmetic, 'wiring': wiring,
        'failed_comparisons': failures,
        'scope_limits': ['Arithmetic checks cover the fixed 24 selected episodes; focus summaries have no additional independent verifier here.',
            'Stair source link closes five supplied force/surface columns to archived NPZ; geometry uses the task normative primitive, not independent collision physics.',
            'Push has no frozen independent archive source-link verifier.',
            'Identity meanings, definition prose, physical terminal meanings, semantic facts and N5 admission are not decided by these machine results.',
            'Original prepared requests and packing receipt are retained; only new manifests and prepared_with_task_definition requests contain the task definitions.',
            'Arithmetic full derivations are control-only files and were never attached to facts packets.']}
    save(CONTROL / 'wiring_arithmetic_receipt.json', receipt)
    print(json.dumps({'status': receipt['status'], 'episode_checks': len(arithmetic), 'packets_reprepared': len(wiring),
        'failed_comparisons': failures, 'arithmetic_statuses': {task: sorted({r['status'] for r in arithmetic if r['task'] == task}) for task in ('stair', 'push')},
        'source_link_statuses': sorted({r['source_link']['status'] for r in arithmetic}),
        'max_estimated_context': max(v['estimated_required_context_tokens'] for row in wiring for v in row['variants'].values())}))


if __name__ == '__main__':
    main()
