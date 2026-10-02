"""One new, offline attempt on immutable existing R2 raw sources; no model call."""
import base64
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
import traceback
import zipfile

import numpy as np

B = Path('/data/yuyangliu/AGENT/feedback_p3new_20260929')
OLD = B / 'n5_confirmation_20260930_r2'
OUT = B / 'semantic_input_development_r1' / ATTEMPT
watched, checks, packets = {}, [], []


def sha(data):
    return hashlib.sha256(data).hexdigest()


def ref(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return {'file': str(path), 'sha256': sha(raw), 'bytes': len(raw)}


def watch(path):
    record = ref(path)
    watched[record['file']] = record
    return Path(path).read_bytes()


def read(path):
    return json.loads(watch(path))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2).encode() + b'\n')


def check(name, passed):
    checks.append({'check': name, 'status': 'PASS' if passed else 'FAIL'})
    if not passed:
        raise AssertionError(name)


def watch_tree(path):
    for file in sorted(path.rglob('*')):
        if file.is_file():
            watch(file)


def restore_columns(value, count):
    dtype = np.dtype(value['dtype'])
    shape = [count] + value['component_shape']
    if 'by_c_order_component' in value:
        columns = [[np.nan if field == 'null' else float(field) for field in column.split(',')]
                   for column in value['by_c_order_component']]
    else:
        columns = []
        for intervals in value['inclusive_constant_intervals_by_component']:
            values, next_step = [], 1
            for first, last, scalar in intervals:
                if first != next_step or last < first:
                    raise ValueError('invalid complete intervals')
                values.extend([scalar] * (last - first + 1))
                next_step = last + 1
            if next_step != count + 1:
                raise ValueError('incomplete intervals')
            columns.append(values)
    return np.asarray(columns, dtype=dtype).T.reshape(shape)


def equal_array(actual, expected):
    return actual.dtype == expected.dtype and actual.shape == expected.shape and (
        actual.tobytes() == expected.tobytes() if np.isfinite(expected).all()
        else np.array_equal(actual, expected, equal_nan=True))


def verify_episode(pid, header, parts, npz_path, new, old):
    task = header['episode']['task']
    with np.load(npz_path, allow_pickle=False) as archive:
        raw = {key[5:]: archive[key].copy() for key in archive.files if key.startswith('raw__')}
    count = new['clock']['last_step']
    for name, value in new['raw_columns'].items():
        check(pid + '/complete raw ' + name, equal_array(restore_columns(value, count), raw[name]))
    for name, value in new['raw_discrete'].items():
        restored = np.concatenate([np.repeat(np.asarray(v, dtype=value['dtype'])[None], last-first+1, axis=0)
                                   for first, last, v in value['inclusive_runs']])
        check(pid + '/complete discrete ' + name, equal_array(restored, raw[name]))
    for name, value in new['direct_columns'].items():
        original = np.concatenate([np.asarray(part['direct_curves'][name], dtype=np.float64) for part in parts])
        check(pid + '/complete direct ' + name, equal_array(restore_columns(value, count), original))
    for name in ('raw_columns', 'raw_discrete', 'definitions', 'static_values', 'source_fields'):
        check(pid + '/v1 original retained ' + name, new[name] == old[name])
    check(pid + '/all original direct values retained', all(new['direct_columns'].get(k) == v for k, v in old['direct_columns'].items()))
    added = set(new['direct_columns']) - set(old['direct_columns'])
    check(pid + '/only registered extra direct curves', added == ({'support_planar_speed_mps', 'support_reference_clearance_m'} if task == 'stair' else set()))
    original_episode = header['episode']
    terminal = original_episode['original_terminal_record']
    expected = {k: v for k, v in terminal.items() if k in (exporter.STAIR_FACT_TERMINAL_FIELDS if task == 'stair' else exporter.PUSH_FACT_TERMINAL_FIELDS)}
    if task == 'push' and 'push_events' in expected:
        expected['push_events'] = [{k: v for k, v in event.items() if k in exporter.PUSH_FACT_EVENT_FIELDS} for event in terminal['push_events']]
    check(pid + '/original task terminal values preserved', new['episode']['original_terminal_record'] == expected)
    check(pid + '/diagnostic flags absent from facts terminal', 'diagnostic_flags' not in new['episode']['original_terminal_record'])
    permitted = ('curve_extrema', 'threshold_intervals_by_component', 'raw_termination_intervals', 'root_x_max_drawdown_witness', 'root_first_last_xyz_terrain_m', 'focus_operand_missing_fields') if task == 'stair' else ('curve_extrema', 'events', 'missing_operands')
    measurements = original_episode['direct_measurements']
    check(pid + '/only unchanged neutral raw arithmetic', new['direct_arithmetic'] == {k: v for k, v in measurements.items() if k in permitted})
    check(pid + '/comparison pass not supplied', 'arithmetic_latch_equals_original_record' not in json.dumps(new))
    check(pid + '/availability is explicit', {'reconstructible_direct_fields', 'not_transmitted_direct_fields', 'not_supplied_direct_fields', 'requested_but_not_recorded_fields'} <= set(new))
    check(pid + '/omitted differs from missing', set(new['not_transmitted_direct_fields']) == set(new['reconstructible_direct_fields']) | set(new['not_supplied_direct_fields']))
    return {'task': task, 'steps': count, 'added_direct_fields': sorted(added), 'retained_raw_columns': sorted(new['raw_columns']),
            'retained_discrete_fields': sorted(new['raw_discrete']), 'direct_arithmetic_fields': sorted(new['direct_arithmetic']),
            'removed_terminal_fields': sorted(set(terminal) - set(expected)), 'reconstructible_direct_fields': sorted(new['reconstructible_direct_fields'])}


def verify_requests(pid, manifest, prepared_paths):
    requests, cards = {}, {}
    for judge, path in prepared_paths.items():
        card = json.loads((path / 'prepare.json').read_bytes())
        raw = (path / 'request.http_body.json').read_bytes()
        check(pid + '/' + judge + '/prepared hash bytes', sha(raw) == card['request_sha256'] and len(raw) == card['request_bytes'])
        requests[judge], cards[judge] = json.loads(raw), card
        actual, _ = runner.coverage.request_materials(requests[judge])
        for item in manifest['materials']:
            material_path = Path(item['file'])
            if not material_path.is_absolute():
                material_path = (manifest_paths[pid].parent / material_path).resolve()
            if item['mime_type'] == 'application/json':
                check(pid + '/' + judge + '/lossless ' + item['id'], runner.coverage.canonical(actual[item['id']]['data']) == runner.coverage.canonical(json.loads(material_path.read_bytes())))
    check(pid + '/identical two-reviewer evidence', requests['deepseek']['messages'][1] == requests['gpt6-sol']['messages'][1])
    result = {'packet_id': pid, 'manifest': ref(manifest_paths[pid]), 'requests': {j: {'prepare': ref(p / 'prepare.json'), 'request': ref(p / 'request.http_body.json'), 'png_count': cards[j]['png_count']} for j, p in prepared_paths.items()}, 'reviewer_blocks_identical': True}
    packets.append(result)
    return requests


def measure_text():
    import tokenizers
    from tokenizers import Tokenizer
    token_path = B / 'n5_execution_20260930/control/numeric_materials_r1/token_capacity_r1/deepseek_v4_tokenizer.zip'
    blob = watch(token_path)
    check('cached tokenizer archive hash', sha(blob) == 'e7310d1dafe0a86d8a5629fe78a7c763760f651db9b8682718a1781dcd6fe495')
    expected = '89085f12ef79460ac5f66d1119325ddfc694b4ab209d80bbd81d35f081dc9614'
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        members = [(name, archive.read(name)) for name in archive.namelist() if name.endswith('tokenizer.json') and sha(archive.read(name)) == expected]
    check('cached tokenizer member hash', len(members) == 1)
    tokenizer = Tokenizer.from_str(members[0][1].decode())
    capture = read(capture_path)
    rules = {key: capture['materials'][key] for key in ('input_token_proxy_limit', 'system_tools_reserved', 'output_reserved', 'proxy_margin_reserved', 'total_proxy_reserved')}
    rules.update({key: capture['model_roles']['gpt6_sol_judge'][key] for key in ('context_window', 'auto_compact_limit')})
    reserve = sum(rules[k] for k in ('system_tools_reserved', 'output_reserved', 'proxy_margin_reserved'))
    records = []
    for packet in packets:
        for judge, source in packet['requests'].items():
            body = json.loads(Path(source['request']['file']).read_bytes())
            parts = []
            for mi, message in enumerate(body['messages']):
                blocks = [{'type': 'text', 'text': message['content']}] if isinstance(message['content'], str) else message['content']
                for bi, block in enumerate(blocks):
                    if block['type'] != 'text':
                        continue
                    text = block['text']
                    parts.append({'message_index': mi, 'block_index': bi, 'text_sha256': sha(text.encode()), 'text_utf8_bytes': len(text.encode()), 'tokens': len(tokenizer.encode(text, add_special_tokens=False).ids)})
            total = sum(part['tokens'] for part in parts)
            records.append({'packet_id': packet['packet_id'], 'judge_id': judge, **source, 'text_blocks': parts,
                'serialized_text_tokens_deepseek_v4': total, 'conservative_reserved_text_proxy': total + reserve,
                'text_only_proxy_within_input_limit': total <= rules['input_token_proxy_limit'],
                'text_only_reserved_proxy_below_autocompact': total + reserve < rules['auto_compact_limit'],
                'text_only_reserved_proxy_within_context': total + reserve <= rules['context_window'],
                'body_output_cap': body['max_tokens'], 'image_tokens': None, 'actual_sol_tokens': None, 'provider_usage': None})
    result = {'schema_version': 'p3new.semantic_input_text_capacity.v1', 'status': 'STATIC_TEXT_MEASURED_NOT_RELEASED', 'requests_counted': len(records),
        'tokenizer': ref(token_path), 'tokenizer_json_sha256': expected, 'tokenizers_version': tokenizers.__version__, 'reservation_rules': rules,
        'records': records, 'max_text_tokens': max(row['serialized_text_tokens_deepseek_v4'] for row in records),
        'all_text_only_proxy_comparisons_pass': all(row['text_only_proxy_within_input_limit'] and row['text_only_reserved_proxy_below_autocompact'] and row['text_only_reserved_proxy_within_context'] for row in records),
        'method': 'Complete actual system/user text blocks summed with cached DeepSeek V4 encode(add_special_tokens=False); no PNG/base64 tokenization.',
        'unmeasured': ['image tokens', 'provider framing and special tokens', 'actual SOL tokens', 'runtime peak context'], 'capacity_release': False, 'model_calls': 0, 'gpu_calls': 0}
    save(OUT / 'text_capacity.json', result)
    return {key: result[key] for key in ('status', 'requests_counted', 'max_text_tokens', 'all_text_only_proxy_comparisons_pass', 'capacity_release')}


OUT.mkdir(parents=True, exist_ok=False)
manifest_paths = {}
try:
    old_source = B / 'n5_confirmation_source_r2'
    old_raw = OLD / 'numeric_raw'
    old_numeric = OLD / 'numeric_materials'
    for tree in (old_source, old_raw, old_numeric):
        watch_tree(tree)
    original_packing = read(old_numeric / 'packing_receipt.json')
    snapshot = OUT / 'source_snapshot'
    shutil.copytree(old_source, snapshot)
    for name, data in SOURCE_BUNDLE.items():
        dest = snapshot / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(base64.b64decode(data))
    source_refs = [ref(snapshot / name) for name in SOURCE_BUNDLE]
    save(OUT / 'executed_source_refs.json', source_refs)
    view = OUT / 'raw_development_view'
    shutil.copytree(old_raw, view)
    original_manifest = read(old_raw / 'manifest.json')
    save(OUT / 'original_raw_manifest.json', original_manifest)
    changed = deepcopy(original_manifest)
    changed['purpose'] = 'development'
    changed['development_view_of'] = {'source': ref(old_raw / 'manifest.json'), 'original_purpose': original_manifest['purpose'], 'status': 'EXISTING_R2_OFFLINE_DEVELOPMENT_NOT_A_NEW_CONFIRMATION'}
    (view / 'manifest.json').write_bytes(json.dumps(changed, ensure_ascii=False, indent=2).encode() + b'\n')
    check('raw profile remains related_v1', changed['material_profile'] == 'related_v1')
    check('all copied raw nonmanifest files unchanged', all((view / path.relative_to(old_raw)).read_bytes() == path.read_bytes() for path in old_raw.rglob('*') if path.is_file() and path.name != 'manifest.json'))
    here = snapshot / 'PLAN/feedback_sys/P3NEW'
    sys.path.insert(0, str(here))
    sys.path.insert(0, str(snapshot / 'src'))
    import prepare_confirmation_materials as packing
    import export_confirmation_raw as exporter
    import run_confirmation_review as runner
    capture_path = Path(original_packing['source']['capture_card'])
    watch(capture_path)
    contracts = [Path(row['file']) for row in original_packing['task_definitions']['contracts']]
    for path in contracts:
        watch(path)
    cell_path = OLD / 'push/composition/resolved_cell_manifest.json'
    watch(cell_path)
    numeric = OUT / 'numeric_materials'
    receipt = packing.pack(view, numeric, here / 'N5_FACT_REVIEW_PROTOCOL_ZH.md', capture_path,
        profile='related_v2', max_output_tokens=32768, task_definitions=contracts, push_resolved_cell_manifest=cell_path)
    check('24 episode plus 12 focus real packets', receipt['episode_count'] == 24 and receipt['focus_count'] == 12 and len(receipt['packets']) == 36)
    check('original task numeric bindings unchanged', receipt['task_definitions']['task_run_bindings'] == original_packing['task_definitions']['task_run_bindings'])
    details = []
    for row in receipt['packets']:
        pid = row['packet_id']
        new = json.loads((numeric / 'materials/related' / (pid + '.json')).read_bytes())
        old = read(old_numeric / 'materials/related' / (pid + '.json'))
        if row['axis'] == 'episode':
            folder = old_raw / 'episodes' / row['scope']
            header = read(folder / 'episode.json')
            parts = [read(folder / part['file']) for part in header['parts']]
            detail = verify_episode(pid, header, parts, folder / header['original_values_npz']['file'], new, old)
        else:
            for field in ('complete_pool_groups', 'scope_denominator', 'scope_row_columns', 'scope_rows', 'operands_in_same_row_order'):
                check(pid + '/unchanged complete denominator and operands ' + field, new[field] == old[field])
            detail = {'task': row['task'], 'scope_denominator': new['scope_denominator'], 'all_member_operands_unchanged': True}
        old_row = next(x for x in original_packing['packets'] if x['packet_id'] == pid)
        check(pid + '/original task definition routing unchanged', row['task_definition_ids'] == old_row['task_definition_ids'])
        details.append({'packet_id': pid, 'axis': row['axis'], **detail})
        path = numeric / 'packets' / pid / 'facts.manifest.json'
        manifest_paths[pid] = path
        verify_requests(pid, json.loads(path.read_bytes()), {judge: numeric / 'prepared' / pid / judge for judge in ('deepseek', 'gpt6-sol')})
    save(OUT / 'numeric_verification.json', {'status': 'PASS_ENGINEERING_ONLY', 'packets': details, 'checks': checks, 'model_calls': 0, 'gpu_calls': 0})
    print(json.dumps({'stage': 'numeric', 'packets': 36, 'checks': len(checks), 'status': 'PASS_ENGINEERING_ONLY'}), flush=True)
    media_root = OLD / 'media_materials_control_corrected'
    png_records = []
    for number in range(1, 5):
        pid = f'media-{number:02d}'
        origin = media_root / pid
        original = read(origin / 'facts.manifest.json')
        raw_folder = origin / 'raw/episodes' / ('e1' if number % 2 else 'e2')
        watch_tree(raw_folder)
        header = read(raw_folder / 'episode.json')
        parts = [read(raw_folder / part['file']) for part in header['parts']]
        new = packing.related_episode(header, parts, raw_folder / header['original_values_npz']['file'])
        old = read(origin / 'raw/related_numeric.json')
        media_detail = verify_episode(pid, header, parts, raw_folder / header['original_values_npz']['file'], new, old)
        destination = OUT / 'media_materials' / pid
        destination.mkdir(parents=True)
        shutil.copytree(raw_folder, destination / 'original_raw')
        save(destination / 'related_numeric.json', new)
        manifest = deepcopy(original)
        manifest.update(purpose='old_development_numeric_packing_offline_only', material_profile='related_v2', reference_contract=runner.coverage.BLOCK_CONTRACT,
            development_view_of={'source': ref(origin / 'facts.manifest.json'), 'status': 'EXISTING_R2_MEDIA_NO_NEW_CAPTURE'})
        for item in manifest['materials']:
            original_path = Path(item['file'])
            if not original_path.is_absolute():
                original_path = (origin / original_path).resolve()
            original_bytes = watch(original_path)
            check(pid + '/original material hash ' + item['id'], sha(original_bytes) == item['sha256'])
            if item['role'] == 'review_protocol':
                protocol = {**json.loads(original_bytes), **runner.packet_protocol(here / 'N5_FACT_REVIEW_PROTOCOL_ZH.md'), 'status': 'DRAFT_NOT_FROZEN_NOT_RELEASED'}
                updated = destination / 'protocol.json'
                save(updated, protocol)
            elif item['id'] == 'related-numeric':
                updated = destination / 'related_numeric.json'
            else:
                updated = original_path
            item.update(file=str(updated), sha256=sha(updated.read_bytes()))
        manifest_paths[pid] = destination / 'facts.manifest.json'
        save(manifest_paths[pid], manifest)
        prepared_paths = {judge: destination / judge / 'prepared' for judge in ('deepseek', 'gpt6-sol')}
        for judge, path in prepared_paths.items():
            runner.prepare(manifest_paths[pid], judge, path)
        requests = verify_requests(pid, manifest, prepared_paths)
        content = requests['deepseek']['messages'][1]['content']
        images = [(content[index - 1], block) for index, block in enumerate(content) if block['type'] == 'image_url']
        original_images = [item for item in original['materials'] if item['role'] == 'raw_image']
        check(pid + '/12 original images', len(images) == len(original_images) == 12)
        for (label, block), item in zip(images, original_images):
            raw = base64.b64decode(block['image_url']['url'].split(',', 1)[1])
            check(pid + '/original PNG bytes and label ' + item['id'], raw == Path(item['file']).read_bytes() and json.loads(label['text'])['material_id'] == item['id'])
            png_records.append({'packet_id': pid, 'material_id': item['id'], 'sha256': sha(raw)})
        save(destination / 'verification.json', {'status': 'PASS_ENGINEERING_ONLY', **media_detail, 'png_count': len(images), 'media_metadata_original_bytes_unchanged': True})
    check('40 actual facts packets', len(packets) == 40)
    check('48 original PNGs unchanged in order', len(png_records) == 48)
    print(json.dumps({'stage': 'all_facts', 'packets': 40, 'png_count': 48, 'status': 'PASS_ENGINEERING_ONLY'}), flush=True)
    text_capacity = measure_text()
    check('80 requests measured', text_capacity['requests_counted'] == 80)
    check('all original source files unchanged', all(ref(path)['sha256'] == record['sha256'] for path, record in watched.items()))
    result = {'schema_version': 'p3new.semantic_input_development.v1', 'status': 'PASS_ENGINEERING_ONLY_NOT_RELEASED',
        'recorded_at_utc': datetime.now(timezone.utc).isoformat(), 'attempt': ATTEMPT, 'source_refs': source_refs,
        'original_input_refs': list(watched.values()), 'original_inputs_unchanged': True,
        'raw_development_manifest': ref(view / 'manifest.json'), 'packing_receipt': ref(numeric / 'packing_receipt.json'),
        'episode_count': 24, 'focus_count': 12, 'media_count': 4, 'prepared_requests': 80,
        'packets': packets, 'png_count': len(png_records), 'png_records': png_records,
        'checks': checks, 'text_capacity': text_capacity, 'model_calls': 0, 'gpu_calls': 0, 'new_capture': False,
        'semantic_review_performed': False, 'facts_seal_created': False, 'capacity_release': False,
        'limits': ['Existing R2 evidence only; a development projection is not a new confirmation cohort.',
                   'Engineering equality and static token counts do not establish correct necessary facts, process semantics, observer quality or reference compliance.',
                   'Image tokens, actual SOL tokens, provider framing and runtime peak context remain unmeasured.']}
    save(OUT / 'receipt.json', result)
    print(json.dumps({key: result[key] for key in ('status', 'episode_count', 'focus_count', 'media_count', 'prepared_requests', 'png_count', 'text_capacity')}), flush=True)
except Exception as error:
    unchanged = all(ref(path)['sha256'] == record['sha256'] for path, record in watched.items())
    save(OUT / 'failure.json', {'status': 'FAILED_ATTEMPT_PRESERVED', 'attempt': ATTEMPT, 'error_type': type(error).__name__, 'detail': str(error),
        'traceback': traceback.format_exc(), 'checks': checks, 'original_inputs_unchanged': unchanged, 'original_input_refs': list(watched.values()), 'model_calls': 0, 'gpu_calls': 0})
    traceback.print_exc()
    raise
