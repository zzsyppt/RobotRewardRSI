"""Fixed two-packet CPU audit; preserve original desktop answers and receipts."""
from copy import deepcopy
from datetime import datetime, timezone
import importlib
from pathlib import Path
import sys
from prepare_cpu import C, CAP, D, S, SCOPE, load, read, ref, require, save


def main():
    require(not (D / 'combined_audit.json').exists(), 'Preserve existing audits')
    manifest = read(D / 'development_source_manifest.json')
    inputs = [ref(D / name) for name in ('development_source_manifest.json', 'cpu_preparation_receipt.json',
                                       'desktop_invocations.json', 'probe_card.json', 'audit_upload_manifest.json')]
    for item in read(D / 'audit_upload_manifest.json')['files']:
        actual = ref(D / item['relative_path'])
        require(actual['sha256'] == item['sha256'] and actual['size_bytes'] == item['size_bytes'], 'Uploaded original differs')
        inputs.append(actual)
    for relative, expected in manifest.items():
        require(ref(S / relative)['sha256'] == expected['sha256'], 'Development source changed: ' + relative)
    sys.path[:0] = [str(S / 'src'), str(S / 'PLAN/feedback_sys/P3NEW'), str(S)]
    native = load('prepare_native_observer', S / 'PLAN/feedback_sys/P3NEW/prepare_native_observer.py')
    observer = importlib.import_module('true_rl_agent.evaluation.visual_observer')
    review = importlib.import_module('true_rl_agent.evaluation.visual_review')
    control = load('existing_access_control', C / 'postcapture_preparation/sol_observer_control_access_v1.py')
    inputs.append(ref(Path(control.__file__)))
    invocations = read(D / 'desktop_invocations.json')['invocations']
    require([row['alias'] for row in invocations] == ['e1', 'e2']
            and len({row['agent_id'] for row in invocations}) == 2, 'Two actual independent invocation records required')
    preparation = read(D / 'cpu_preparation_receipt.json')
    require(preparation['scope'] == SCOPE, 'Wrong preparation scope')
    calls, staged = [], []
    for invocation, prepared in zip(invocations, preparation['packets'], strict=True):
        alias, packet = invocation['alias'], D / invocation['alias']
        require(alias == prepared['alias'], 'Packet order differs')
        built = native.build(CAP / 'exports/stair', packet / 'selection.json', episode_alias=alias)
        request_ref = ref(packet / 'request.json')
        require(built['prepared']['request_bytes'] == (packet / 'request.json').read_bytes()
                and request_ref['sha256'] == prepared['new_request']['sha256'], 'Prepared request rebuild differs')
        materials_ref = ref(packet / 'blind_inputs/materials.json')
        materials = read(packet / 'blind_inputs/materials.json')['materials']
        carrier = read(packet / 'carrier_preparation.json')
        require(materials_ref['sha256'] == prepared['materials']['sha256'] == invocation['input_manifest_sha256']
                and carrier['source_request_sha256'] == request_ref['sha256'] and carrier['scope'] == SCOPE
                and carrier['source_manifest_sha256'] == ref(D / 'development_source_manifest.json')['sha256'], 'Carrier binding differs')
        require(invocation['fork_turns'] == 'none' and invocation['model'] == 'gpt-6.1-sol'
                and invocation['reasoning_effort'] == 'high' and invocation['no_retry'] is True, 'Registered invocation differs')
        for row in materials:
            actual = ref(packet / 'blind_inputs' / row['path'])
            require(actual['sha256'] == row['sha256'], 'Original input bytes changed')
            inputs.append(actual)
        require(read(packet / 'blind_inputs/blind_input_text.json') == control.blind_text(built['prepared']['payload']), 'Original text differs')
        assets = [image for window in built['windows'] for image in window.images]
        image_rows = [row for row in materials if row['role'] == 'original_image']
        for asset, row in zip(assets, image_rows, strict=True):
            require(control.sha(asset.data) == row['sha256'] and row['step'] == asset.displayed_steps[0]
                    and row['view_id'] == asset.view_id and row['image_size_px'] == [asset.width, asset.height], 'Original image identity differs')
        raw_ref, access_ref = [ref(packet / 'sol_response' / name) for name in ('raw_answer.txt', 'access_receipt.json')]
        access = read(packet / 'sol_response/access_receipt.json')
        desktop_root = (r'D:\humanoid_agent\true_RL_Agent\PLAN\feedback_sys\P3NEW\results\stair_body_identity_dev_r1'
                        + '\\' + alias + r'\blind_inputs')
        access_check = control.check_access_metadata(access, materials, materials_ref['sha256'],
                                                     desktop_root=desktop_root, raw_answer_sha256=raw_ref['sha256'])
        answer = None
        try:
            answer = read(packet / 'sol_response/raw_answer.txt')
            audit = observer.audit_observer_output(answer, built['windows'], numeric_check=built['numeric_check'])
        except Exception as error:
            audit = {'schema_version': observer.SCHEMA_VERSION, 'raw_output': answer, 'observations': [],
                     'violations': ['answer_parse_or_audit_error'], 'passed': False, 'error_type': type(error).__name__,
                     'semantic_support': 'not_evaluated', 'raw_answer_preserved_in': raw_ref}
        audit.update(scope=SCOPE, confirmation_admitted=False)
        audit_path, audit_bytes = packet / 'sol_response/mechanical_audit.json', control.encoded(audit)
        staged.append((audit_path, audit_bytes))
        calls.append({'episode_alias': alias, 'scope': SCOPE, 'status': 'received', 'carrier': 'fresh_subagent',
            'request_sha256': request_ref['sha256'], 'request_binding_kind': 'prepared_template_not_sent_HTTP',
            'receipt_sha256': access_ref['sha256'], 'receipt_kind': 'actual_SOL_access_receipt',
            'access_receipt': access_ref, 'original_access_receipt': access, 'carrier_access_metadata_check': access_check,
            'raw_answer': raw_ref, 'materials': materials_ref, 'registered_agent': invocation['agent_id'],
            'registered_invocation': deepcopy(invocation), 'audit': audit, 'audit_sha256': control.sha(audit_bytes),
            'raw_response_sha256': None})
        inputs.extend([request_ref, materials_ref, raw_ref, access_ref, ref(packet / 'selection.json'), ref(packet / 'carrier_preparation.json')])
    observations, violations, raw_calls = [], [], []
    for index, call in enumerate(calls):
        audit, prefix = call['audit'], f'calls[{index}]'
        raw_calls.append(deepcopy(audit['raw_output']))
        violations.extend(prefix + ':' + issue for issue in audit['violations'])
        for original in audit['observations']:
            row = deepcopy(original)
            row.update(reference=prefix + '.' + row['reference'], source_call_index=index)
            observations.append(row)
    combined = {'schema_version': observer.SCHEMA_VERSION, 'scope': SCOPE, 'task': 'stair',
        'capture_card_sha256': ref(C / 'capture_card.json')['sha256'],
        'source_manifest_sha256': ref(D / 'development_source_manifest.json')['sha256'],
        'report_card_sha256': ref(D / 'probe_card.json')['sha256'], 'report_card_kind': 'development_probe_card',
        'desktop_invocations': ref(D / 'desktop_invocations.json'), 'calls': calls,
        'raw_output': {'calls': raw_calls}, 'observations': observations, 'violations': violations,
        'passed': not violations and all(call['audit']['passed'] for call in calls) and not any(row['violations'] for row in observations),
        'semantic_support': 'not_evaluated', 'all_raw_observations_retained': True, 'actual_desktop_calls_archived': 2,
        'actual_desktop_cost_usd': None, 'provider_HTTP_receipts_created': 0, 'usage_or_fingerprint_invented': False,
        'confirmation_admitted': False}
    staged.append((D / 'combined_audit.json', control.encoded(combined)))
    modules = {name: ref(Path(module.__file__)) for name, module in tuple(sys.modules.items())
               if (name.startswith('true_rl_agent') or name == 'prepare_native_observer') and getattr(module, '__file__', None)}
    for item in modules.values():
        relative = Path(item['path']).relative_to(S).as_posix()
        require(item['sha256'] == manifest[relative]['sha256'], 'Loaded source differs')
    for item in inputs:
        require(ref(Path(item['path'])) == item, 'Original input changed during audit')
    for path, data in staged:
        with path.open('xb') as stream:
            stream.write(data)
    binding = review.visual_review_binding(combined)
    save(D / 'visual_review_binding.json', {'scope': SCOPE, 'source_sha256': binding,
        'criterion': review.REVIEW_CRITERION, 'combined_audit': ref(D / 'combined_audit.json'), 'semantic_review_created': False})
    save(D / 'sol_audit_receipt.json', {'scope': SCOPE, 'status': 'AUDITED_NOT_RELEASED',
        'recorded_at_utc': datetime.now(timezone.utc).isoformat(), 'script': ref(Path(__file__)),
        'helper': ref(D / 'prepare_cpu.py'), 'loaded_modules': modules, 'inputs': inputs,
        'outputs': [ref(path) for path, _ in staged] + [ref(D / 'visual_review_binding.json')],
        'mechanical_audit_passed': combined['passed'], 'observations': len(observations),
        'visual_review_binding': binding, 'raw_answer_bytes_unchanged': True,
        'new_model_calls': 0, 'new_API_calls': 0, 'gpu_calls': 0,
        'semantic_decisions_created': False, 'confirmation_admitted': False,
        'access_check_limit': 'Recorded metadata consistency; actual tool trace review remains separate.'})
    print(control.encoded({'passed': combined['passed'], 'observations': len(observations),
                          'visual_review_binding': binding, 'scope': SCOPE}).decode())


if __name__ == '__main__':
    main()
