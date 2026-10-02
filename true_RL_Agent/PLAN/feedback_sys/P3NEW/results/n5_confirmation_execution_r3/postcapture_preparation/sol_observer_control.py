"""R3 one-shot SOL carrier prepare/audit glue; no model calls or semantic review.

Reuse R1 blind-text extraction, native rebuild, mechanical audit and faithful
e1/e2 aggregation. All production functions load from the unchanged R3 freeze.
"""
from __future__ import annotations
import argparse
import base64
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

B = Path('/data/yuyangliu/AGENT/feedback_p3new_20260929')
S, C, CAP = (B / name for name in ('n5_confirmation_source_r3_final', 'n5_execution_20260930_r3', 'n5_confirmation_20260930_r3'))
P = S / 'PLAN/feedback_sys/P3NEW'
CARD, MANIFEST = C / 'capture_card.json', C / 'runtime_source_manifest.json'
CARD_SHA = None
SOURCE_SHA = '5cb1b36c5ca2e7af40d62c51e7cf1fe9aae4790e080353b8947cc6e14e5902d5'


def require(value, reason):
    if not value:
        raise ValueError(reason)


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def ref(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return {'path': str(path), 'sha256': sha(raw), 'size_bytes': len(raw)}


def read(path):
    return json.loads(Path(path).read_bytes())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle:
        handle.write(encoded(value))


def checked(reference):
    actual = ref(reference['path'])
    require(actual['sha256'] == reference['sha256'] and
            ('size_bytes' not in reference or actual['size_bytes'] == reference['size_bytes']),
            'Input hash/size differs: ' + reference['path'])
    return Path(actual['path'])


def inside(root, relative):
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()), 'Carrier material path escapes its directory')
    return path


def blind_text(payload):
    value = {key: item for key, item in payload.items() if key != 'messages'}
    value['messages'] = [{'role': message['role'], 'text_blocks':
        [message['content']] if isinstance(message['content'], str) else
        [part['text'] for part in message['content'] if part['type'] == 'text']}
        for message in payload['messages']]
    return value


def load_context(task):
    require(ref(CARD)['sha256'] == CARD_SHA and ref(MANIFEST)['sha256'] == SOURCE_SHA, 'R3 freeze differs')
    card, manifest = read(CARD), read(MANIFEST)
    require(card['status'] == 'FROZEN' and card['output_root'] == str(CAP)
            and card['source']['remote_root'] == str(S)
            and card['source']['manifest']['sha256'] == SOURCE_SHA, 'Wrong R3 roots/card')
    sys.path[:0] = [str(S / 'src'), str(P), str(S)]
    native = importlib.import_module('prepare_native_observer')
    observer = importlib.import_module('true_rl_agent.evaluation.visual_observer')
    review = importlib.import_module('true_rl_agent.evaluation.visual_review')
    for module in (native, observer, review):
        path = Path(module.__file__).resolve()
        require(path.is_relative_to(S), 'Wrong frozen module origin')
        require(ref(path)['sha256'] == manifest[path.relative_to(S).as_posix()]['sha256'], 'Frozen helper changed')
    root = CAP / 'observers' / task
    report_path = root / 'report.json'
    report = read(report_path)
    require(report['schema_version'] == 'p3new.native_observer_report.v1'
            and report['status'] == 'PREPARED_NOT_RELEASED' and report['scope'] == 'n5_confirmation'
            and report['task'] == task, 'Wrong prepared observer report')
    require(checked(report['capture_card']) == CARD and report['capture_card']['sha256'] == CARD_SHA,
            'Observer report belongs to another capture')
    selection_dir = CAP / 'observer_selection' / task
    require(checked(report['selection_card']) == selection_dir / 'card.json', 'Wrong selection-card path')
    selection, selection_card = read(selection_dir / 'selection.json'), read(selection_dir / 'card.json')
    require(ref(selection_dir / 'selection.json')['sha256'] == report['selection_sha256'] == selection_card['selection_sha256']
            and selection['source']['capture_card_sha256'] == CARD_SHA
            and selection_card['observer_profile'] == 'READY_FOR_PREPARATION'
            and selection_card['pool_complete'] is True and not selection_card['missing_classes'], 'Selection binding differs')
    require([(row['episode_alias'], row['directory']) for row in report['packets']] == [('e1', 'e1'), ('e2', 'e2')],
            'Prepared packet order differs')
    inputs = [ref(CARD), ref(MANIFEST), ref(report_path), report['selection_card'], ref(selection_dir / 'selection.json')]
    return root, report, inputs, manifest, native, observer, review


def rebuild(root, report, binding, native, inputs):
    packet = root / binding['directory']
    require(ref(packet / 'card.json')['sha256'] == binding['prepared_card_sha256']
            and ref(packet / 'request.json')['sha256'] == binding['request_sha256']
            and ref(packet / 'selection.json')['sha256'] == report['selection_sha256'], 'Packet binding differs')
    private = read(packet / 'private_sidecar.json')
    require(Path(private['source_export']).resolve() == CAP / 'exports' / report['task'], 'Wrong R3 export root')
    built = native.build(Path(private['source_export']), packet / 'selection.json', episode_alias=binding['episode_alias'])
    require(built['prepared']['request_bytes'] == (packet / 'request.json').read_bytes()
            and built['card']['extraction_sha256'] == report['extraction_sha256'], 'Rebuilt request differs')
    for name in ('card', 'private_sidecar', 'mechanical_audit', 'alignment'):
        require(built[name] == read(packet / (name + '.json')), 'Rebuilt preparation differs: ' + name)
        inputs.append(ref(packet / (name + '.json')))
    inputs.extend((ref(packet / 'request.json'), ref(packet / 'selection.json')))
    return packet, built


def check_access_metadata(access, materials, materials_sha256):
    """Reuse the two actual R1 receipt shapes; never synthesize absent claims."""
    require(access.get('schema_version') == 'p3new.observer_access_receipt.v1'
            and access.get('materials_sha256') == materials_sha256, 'Access/material binding differs')
    expected = {row['path']: row['sha256'] for row in materials}
    expected['materials.json'] = materials_sha256
    reads = access.get('files_read')
    require(isinstance(reads, list) and len(reads) == len(expected)
            and {row['path']: row['sha256'] for row in reads} == expected, 'Actual files_read coverage differs')
    images = [row for row in materials if row['role'] == 'original_image']
    one_based = 'images_viewed' in access and 'viewed_images' not in access
    zero_based = 'viewed_images' in access and 'images_viewed' not in access
    require(one_based or zero_based, 'Unsupported actual image-view receipt; preserve it without rewriting')
    views = access['images_viewed' if one_based else 'viewed_images']
    require(isinstance(views, list) and len(views) == len(images), 'Incomplete image-view receipt')
    for index, (image, view) in enumerate(zip(images, views, strict=True)):
        require(view['view_order' if one_based else 'order'] == index + int(one_based)
                and view['path'] == image['path'] and view['step'] == image['step']
                and view['view_id'] == image['view_id'] and view['original_dimensions_px'] == image['image_size_px'],
                'Actual image-view identity/order/dimensions differ')
        if one_based:
            require(view['sha256'] == image['sha256'] and view['detail'] == 'original', 'Recorded view hash/detail differs')
    return {'status': 'CHECKED_ACTUAL_RECEIPT_METADATA', 'files_read': len(reads), 'images_viewed': len(views),
        'scope': 'attestation consistency; not independent desktop tool replay',
        'actual_view_list_key': 'images_viewed' if one_based else 'viewed_images', 'order_base': int(one_based),
        'per_view_sha_recorded': one_based, 'render_detail_assertion': 'original' if one_based else None,
        'reported_actual_carrier': deepcopy(access.get('actual_carrier')),
        'original_access_limitations': deepcopy(access.get('access_limitations'))}


def prepare_packet(packet, built, binding):
    blind = packet / 'blind_inputs'
    require(not blind.exists() and not (packet / 'carrier_preparation.json').exists(), 'Preserve existing carrier preparation')
    assets = [image for window in built['windows'] for image in window.images]
    urls = [block['image_url']['url'] for message in built['prepared']['payload']['messages']
            if isinstance(message['content'], list) for block in message['content'] if block['type'] == 'image_url']
    require(len(urls) == len(assets), 'Request/window image count differs')
    blind.mkdir()
    text_path = blind / 'blind_input_text.json'
    save(text_path, blind_text(built['prepared']['payload']))
    materials = [{'role': 'request_text', 'path': text_path.name, 'sha256': ref(text_path)['sha256']}]
    (blind / 'images').mkdir()
    for index, (asset, url) in enumerate(zip(assets, urls, strict=True)):
        require(url.startswith('data:image/png;base64,') and base64.b64decode(url.split(',', 1)[1], validate=True) == asset.data,
                'Request does not contain the exact original PNG')
        name = f'images/{index:02d}.step{asset.displayed_steps[0]:06d}.{asset.view_id}.png'
        with (blind / name).open('xb') as handle:
            handle.write(asset.data)
        materials.append({'role': 'original_image', 'path': name, 'step': asset.displayed_steps[0],
            'view_id': asset.view_id, 'sha256': sha(asset.data), 'image_size_px': [asset.width, asset.height]})
    save(blind / 'materials.json', {'schema_version': 'p3new.fresh_observer_materials.v1', 'materials': materials})
    manifest_ref = ref(blind / 'materials.json')
    save(packet / 'carrier_preparation.json', {'status': 'PREPARED_NOT_CALLED', 'scope': 'n5_confirmation',
        'source_request_sha256': binding['request_sha256'], 'materials_sha256': manifest_ref['sha256'],
        'capture_card_sha256': CARD_SHA, 'source_manifest_sha256': SOURCE_SHA,
        'image_count': len(assets), 'encoding': 'lossless original PNG bytes; all request text preserved in message order',
        'template_model_is_not_actual_carrier': True, 'requested_carrier': 'gpt-6.1-sol',
        'actual_agent_identity': None, 'provider_calls': 0})
    return {'episode_alias': binding['episode_alias'], 'materials': manifest_ref,
            'carrier_preparation': ref(packet / 'carrier_preparation.json'), 'images': len(assets)}


def audit_packet(packet, built, binding, invocation, observer, inputs):
    materials_ref = ref(packet / 'blind_inputs/materials.json')
    materials = read(materials_ref['path'])['materials']
    carrier = read(packet / 'carrier_preparation.json')
    require(carrier['source_request_sha256'] == binding['request_sha256']
            and carrier['materials_sha256'] == materials_ref['sha256'] and carrier['capture_card_sha256'] == CARD_SHA
            and carrier['source_manifest_sha256'] == SOURCE_SHA, 'Wrong carrier preparation binding')
    require(invocation['alias'] == binding['episode_alias'] and invocation['input_manifest_sha256'] == materials_ref['sha256']
            and invocation['fork_turns'] == 'none' and invocation['model'] == 'gpt-6.1-sol'
            and invocation['reasoning_effort'] == 'high' and invocation['no_retry'] is True
            and isinstance(invocation['agent_id'], str) and invocation['agent_id'].strip(), 'Actual fresh invocation is missing/mismatched')
    text_rows = [row for row in materials if row['role'] == 'request_text']
    image_rows = [row for row in materials if row['role'] == 'original_image']
    require(len(text_rows) == 1 and len(materials) == 1 + len(image_rows), 'Unexpected carrier roles')
    blind = packet / 'blind_inputs'
    for row in materials:
        actual = ref(inside(blind, row['path']))
        require(actual['sha256'] == row['sha256'], 'Carrier bytes changed')
        inputs.append(actual)
    require(read(inside(blind, text_rows[0]['path'])) == blind_text(built['prepared']['payload']), 'Blind text differs')
    assets = [image for window in built['windows'] for image in window.images]
    require(len(assets) == len(image_rows), 'Carrier/window image count differs')
    for asset, row in zip(assets, image_rows, strict=True):
        require(row['sha256'] == sha(asset.data) and row['step'] == asset.displayed_steps[0]
                and row['view_id'] == asset.view_id and row['image_size_px'] == [asset.width, asset.height], 'Carrier image/order differs')
    raw_ref, access_ref = (ref(packet / 'sol_response' / name) for name in ('raw_answer.txt', 'access_receipt.json'))
    access = read(access_ref['path'])
    access_check = check_access_metadata(access, materials, materials_ref['sha256'])
    inputs.extend((materials_ref, ref(packet / 'carrier_preparation.json'), raw_ref, access_ref))
    answer = None
    try:
        answer = read(raw_ref['path'])
        audit = observer.audit_observer_output(answer, built['windows'], numeric_check=built['numeric_check'])
    except Exception as error:
        audit = {'schema_version': observer.SCHEMA_VERSION, 'raw_output': answer, 'observations': [],
            'violations': ['answer_parse_or_audit_error'], 'passed': False,
            'error_type': type(error).__name__, 'semantic_support': 'not_evaluated', 'raw_answer_preserved_in': raw_ref}
    audit_bytes = encoded(audit)
    audit_path = packet / 'sol_response/mechanical_audit.json'
    audit_ref = {'path': str(audit_path), 'sha256': sha(audit_bytes), 'size_bytes': len(audit_bytes)}
    call = {'episode_alias': binding['episode_alias'], 'status': 'received', 'carrier': 'fresh_subagent',
        'request_sha256': binding['request_sha256'], 'request_binding_kind': 'prepared_template_not_sent_HTTP',
        'receipt_sha256': access_ref['sha256'], 'receipt_kind': 'actual_SOL_access_receipt',
        'access_receipt': access_ref, 'original_access_receipt': access, 'carrier_access_metadata_check': access_check,
        'raw_answer': raw_ref, 'materials': materials_ref, 'registered_agent': invocation['agent_id'],
        'registered_invocation': deepcopy(invocation), 'audit': audit, 'audit_sha256': audit_ref['sha256'],
        'raw_response_sha256': None}
    return call, audit_path, audit_bytes


def run(task, mode, invocations_path=None):
    started = datetime.now(timezone.utc).isoformat()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.dont_write_bytecode = True
    root, report, inputs, manifest, native, observer, review = load_context(task)
    receipt_path = root / ('sol_carrier_preparation_receipt.json' if mode == 'prepare' else 'sol_audit_receipt.json')
    require(not receipt_path.exists(), 'Preserve existing receipt')
    packets, staged = [], []
    if mode == 'audit':
        require(invocations_path is not None, 'audit requires the real --invocations file; identity is never generated')
        invocations_path = invocations_path.resolve()
        invocations = read(invocations_path)
        require(isinstance(invocations.get('recorded_at_utc'), str) and invocations['recorded_at_utc'].strip(), 'Actual invocation record needs its own recorded UTC')
        require([row['alias'] for row in invocations['invocations']] == ['e1', 'e2']
                and len({row['agent_id'] for row in invocations['invocations']}) == 2, 'Need the two distinct actual agents in e1/e2 order')
        inputs.append(ref(invocations_path))
        require(not (root / 'combined_audit.json').exists()
                and not any((root / alias / 'sol_response/mechanical_audit.json').exists() for alias in ('e1', 'e2')), 'Preserve existing audits')
    for index, binding in enumerate(report['packets']):
        packet, built = rebuild(root, report, binding, native, inputs)
        if mode == 'prepare':
            packets.append(prepare_packet(packet, built, binding))
        else:
            call, path, data = audit_packet(packet, built, binding, invocations['invocations'][index], observer, inputs)
            packets.append(call)
            staged.append((path, data))
    if mode == 'audit':
        observations, violations, raw_calls = [], [], []
        for index, call in enumerate(packets):
            prefix, audit = f'calls[{index}]', call['audit']
            raw_calls.append(deepcopy(audit['raw_output']))
            violations.extend(prefix + ':' + issue for issue in audit['violations'])
            for original in audit['observations']:
                row = deepcopy(original)
                row['reference'] = prefix + '.' + row['reference']
                row['source_call_index'] = index
                observations.append(row)
        combined = {'schema_version': observer.SCHEMA_VERSION, 'scope': 'n5_confirmation',
            'capture_card_sha256': CARD_SHA, 'source_manifest_sha256': SOURCE_SHA, 'task': task,
            'report_card_sha256': ref(root / 'report.json')['sha256'], 'desktop_invocations': ref(invocations_path),
            'calls': packets, 'raw_output': {'calls': raw_calls}, 'observations': observations, 'violations': violations,
            'passed': not violations and not any(row['violations'] for row in observations),
            'semantic_support': 'not_evaluated', 'all_raw_observations_retained': True,
            'actual_desktop_calls_archived': 2, 'actual_desktop_cost_usd': None,
            'provider_HTTP_receipts_created': 0, 'usage_or_fingerprint_invented': False}
        staged.append((root / 'combined_audit.json', encoded(combined)))
    modules = {name: ref(module.__file__) for name, module in tuple(sys.modules.items())
        if (name.startswith('true_rl_agent') or name == 'prepare_native_observer') and getattr(module, '__file__', None)}
    for item in modules.values():
        relative = Path(item['path']).relative_to(S).as_posix()
        require(item['sha256'] == manifest[relative]['sha256'], 'Loaded frozen module changed: ' + relative)
    for item in inputs:
        checked(item)
    for path, data in staged:
        with path.open('xb') as handle:
            handle.write(data)
    receipt = {'status': 'PREPARED_NOT_CALLED' if mode == 'prepare' else 'AUDITED_NOT_RELEASED',
        'task': task, 'mode': mode, 'scope': 'n5_confirmation', 'capture_root': str(CAP),
        'started_at_utc': started, 'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
        'timestamp_scope': 'this CPU operation only; not a root-review sealing time',
        'script': ref(__file__), 'source_root': str(S), 'loaded_modules': modules, 'inputs': inputs,
        'new_model_calls': 0, 'new_API_calls': 0, 'gpu_calls': 0, 'semantic_decisions_created': False,
        'root_review_created': False, 'confirmation_admitted': False,
        'carrier_identity_limit': 'caller-supplied actual invocation/access records; not independent service/tool attestation'}
    if mode == 'prepare':
        receipt['packets'] = packets
    else:
        receipt.update(outputs=[ref(path) for path, _ in staged], mechanical_audit_passed=combined['passed'],
            visual_review_binding=review.visual_review_binding(combined),
            visual_review_module=ref(Path(review.__file__)), observations=len(combined['observations']),
            raw_answer_bytes_unchanged=True, original_invocations=invocations)
    save(receipt_path, receipt)
    print(json.dumps({'status': receipt['status'], 'task': task, 'receipt': ref(receipt_path), 'model_calls': 0}))
    return 0 if mode == 'prepare' or combined['passed'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--task', choices=('stair', 'push'), required=True)
    parser.add_argument('--mode', choices=('prepare', 'audit'), required=True)
    parser.add_argument('--invocations', type=Path, help='Actual per-task e1/e2 desktop invocation record; required for audit')
    parser.add_argument('--card-sha256', required=True, help='Actual R3 frozen card SHA')
    args = parser.parse_args()
    CARD_SHA = args.card_sha256
    require(len(CARD_SHA) == 64 and all(c in '0123456789abcdef' for c in CARD_SHA), 'Invalid card SHA')
    if args.mode == 'audit' and args.invocations is None:
        parser.error('--mode audit requires --invocations; provide actual identities after the calls')
    if args.mode == 'prepare' and args.invocations is not None:
        parser.error('--invocations applies only to audit')
    raise SystemExit(run(args.task, args.mode, args.invocations))
