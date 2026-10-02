"""One-shot R5 CPU arithmetic glue. Produces control-only evidence, never release."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import traceback

B = Path('/data/yuyangliu/AGENT/feedback_p3new_20260929')
S = B / 'n5_confirmation_source_r5_final'
C = B / 'n5_execution_20261001_r5'
CAP = B / 'n5_confirmation_20261001_r5'
RAW, PACK = CAP / 'numeric_raw', CAP / 'numeric_materials'
OUT = C / 'control/numeric_arithmetic'
HELPERS = S / 'PLAN/feedback_sys/P3NEW'
SOURCE_MANIFEST = C / 'runtime_source_manifest.json'
SOURCE_SHA = 'ca9e11f079200d26545f75a2550bcf0901afd59c02adee3d418695cb4ccfa2bd'
CARD = C / 'capture_card.json'
CARD_SHA = None
PROTOCOL = HELPERS / 'N5_FACT_REVIEW_PROTOCOL_ZH.md'
NUMERIC = {'stair': [CAP / 'stair/numeric_evidence/manifest.json'],
           'push': [CAP / f'push/numeric/public_dev-{n:02d}/manifest.json' for n in (1, 2, 3)]}
LIMITS = [
    'Only the fixed 24 episode packets are checked; the 12 focus packets have no additional independent verifier here.',
    'Stair operand-source coverage is exactly the frozen helper result. Task normative geometry is not independent collision physics.',
    'Push archive-to-derivation source links are a separate control supplement after this receipt. Neither step authenticates physical construction or terminal meaning.',
    'All helper unchecked fields and remaining boundaries are retained verbatim in their result files.',
    'No semantic verdict, visual review, model capacity/completion verdict, necessary-fact vote, or confirmation admission is produced.',
    'Prior long-packet model executions remain incomplete; this CPU result does not complete them.',
    'Full derivation wrappers stay control-only and are never added to facts packets.',
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def ref(path):
    path = Path(path).resolve()
    with path.open('rb') as handle:
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    return {'path': str(path), 'sha256': digest, 'bytes': path.stat().st_size}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
        handle.write('\n')


def main():
    # Exclusive output preserves incomplete evidence and prevents implicit retries.
    OUT.mkdir(parents=True, exist_ok=False)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.dont_write_bytecode = True
    watched, rows = {}, []
    receipt = {'schema_version': 'p3new.n5_r3_numeric_arithmetic.v1',
        'status': 'NOT_RELEASED', 'execution_status': 'IN_PROGRESS',
        'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'script': ref(Path(__file__)), 'source_root': str(S), 'capture_root': str(CAP),
        'model_calls': 0, 'gpu_calls': 0, 'facts_admitted': False,
        'semantic_review_performed': False, 'observer_or_judge_answers_read': False,
        'task_definition_or_packet_mutations': 0, 'scope_limits': LIMITS,
        'arithmetic': rows, 'input_refs': watched, 'loaded_source_modules': {}}
    watched[receipt['script']['path']] = receipt['script']

    def read(path, expected=None):
        path = Path(path).resolve()
        data = path.read_bytes()
        reference = {'path': str(path), 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
        if expected is not None:
            require(reference['sha256'] == expected, 'Input hash differs: ' + str(path))
        if str(path) in watched:
            require(watched[str(path)] == reference, 'Input changed during execution: ' + str(path))
        watched[str(path)] = reference
        return json.loads(data)

    def frozen_ref(path, manifest):
        reference = ref(path)
        relative = Path(reference['path']).relative_to(S).as_posix()
        expected = manifest[relative]
        require(reference['sha256'] == expected['sha256'] and reference['bytes'] == expected['size_bytes'],
                'Loaded frozen source differs: ' + relative)
        watched[reference['path']] = reference
        return reference

    try:
        source = read(SOURCE_MANIFEST, SOURCE_SHA)
        card = read(CARD, CARD_SHA)
        require(card['status'] == 'FROZEN' and card['output_root'] == str(CAP), 'Wrong R5 capture card/root')
        require(card['source']['remote_root'] == str(S) and card['source']['manifest']['sha256'] == SOURCE_SHA,
                'Capture/source freeze differs')
        protocol = frozen_ref(PROTOCOL, source)
        require(protocol['sha256'] == card['review_protocol']['sha256'], 'Protocol/card binding differs')
        raw_receipt = read(RAW / 'manifest.json')
        packing = read(PACK / 'packing_receipt.json')
        require(raw_receipt['purpose'] == packing['purpose'] == 'confirmation'
                and raw_receipt['material_profile'] == 'related_v1' and packing['material_profile'] == 'related_v2'
                and raw_receipt['stage'] == 'facts' and packing['actual_calls'] == 0,
                'Expected untouched related_v1 raw / related_v2 confirmation facts pack')
        hashes = packing['source_sha256']
        require(hashes['raw_manifest'] == watched[str(RAW / 'manifest.json')]['sha256']
                and hashes['capture_card'] == CARD_SHA and hashes['protocol'] == protocol['sha256']
                and packing['source'] == {'raw_export': str(RAW), 'protocol': str(PROTOCOL), 'capture_card': str(CARD)},
                'Packing input binding differs')
        require(raw_receipt['frozen_inputs']['capture_card']['sha256'] == CARD_SHA
                and raw_receipt['frozen_inputs']['review_protocol']['sha256'] == protocol['sha256'],
                'Raw export/card binding differs')
        private_ref = raw_receipt['private_source_manifest']
        require(private_ref['file'] == '_private/sources.json'
                and hashes['private_source_manifest'] == private_ref['sha256'], 'Private alias source unbound')
        private = read(RAW / private_ref['file'], private_ref['sha256'])
        require(packing['task_definitions']['source_manifest']['sha256'] == private_ref['sha256'],
                'Pack task definitions bind another private source')
        for relative, digest in raw_receipt['source_hashes'].items():
            require(source[relative]['sha256'] == digest, 'Export source differs from freeze: ' + relative)
        for key, filename in (('packer', 'prepare_confirmation_materials.py'), ('runner', 'run_confirmation_review.py')):
            require(hashes[key] == source['PLAN/feedback_sys/P3NEW/' + filename]['sha256'],
                    'Pack implementation differs from freeze: ' + key)

        sys.path[:0] = [str(HELPERS), str(S / 'src'), str(S)]
        modules = {}
        for name in ('verify_stair_confirmation_arithmetic', 'verify_push_confirmation_arithmetic'):
            frozen_ref(HELPERS / (name + '.py'), source)
            modules[name] = importlib.import_module(name)
            require(Path(modules[name].__file__).resolve() == HELPERS / (name + '.py'), 'Wrong helper import origin')
            receipt['loaded_source_modules'][name] = frozen_ref(Path(modules[name].__file__), source)
        stair, push = modules.values()
        import stair_operand_source_adapter as source_adapter
        import terminal_projection_check as projection_check
        for control_module in (source_adapter, projection_check):
            control_ref = ref(control_module.__file__)
            watched[control_ref['path']] = control_ref
            receipt.setdefault('control_modules', {})[control_module.__name__] = control_ref
        projection_contract = projection_check.load_contract(S, SOURCE_MANIFEST)
        frozen_ref(HELPERS / 'export_confirmation_raw.py', source)
        raw_catalogue = {r['file']: r for r in raw_receipt['readable_materials']}
        receipt['runtime'] = {'python': sys.version, 'executable': sys.executable,
            'numpy_version': stair.np.__version__, 'numpy_file': str(Path(stair.np.__file__).resolve()),
            'CUDA_VISIBLE_DEVICES': os.environ['CUDA_VISIBLE_DEVICES']}

        fulls, full_refs, numeric_shas = {}, {}, {}
        for task in ('stair', 'push'):
            numeric_shas[task] = []
            for path in NUMERIC[task]:
                numeric = read(path)
                numeric_shas[task].append(watched[str(path)]['sha256'])
                require(numeric['identity']['evaluation_binding'] == card['tasks'][task]['current_evaluation_binding'],
                        'Numeric/card binding differs: ' + str(path))
            full_path = CAP / task / 'evaluation/public_dev/behavior_full.json'
            phase = read(CAP / task / 'public_dev_result.json')
            require(phase['evaluation_binding_hash'] == card['tasks'][task]['current_evaluation_binding']['evaluation_binding_hash'],
                    'Phase is not this R5 cohort: ' + task)
            declared = phase['task_evidence']['behavior_full']
            fulls[task] = read(full_path, declared['sha256'])
            full_refs[task] = watched[str(full_path)]
            require(full_refs[task]['bytes'] == declared['size_bytes'] and 'observer_evidence' not in fulls[task],
                    'Use original numeric full, without observer attachment')
        require(fulls['stair']['domain']['archive_hash'] == numeric_shas['stair'][0], 'Stair full/archive differs')
        require({r['manifest_sha256'] for r in fulls['push']['sources']} == set(numeric_shas['push']),
                'Push full/archive set differs')
        require({(r['manifest_path'], r['manifest_sha256']) for r in private['sources']} ==
                {(str(path), digest) for task in NUMERIC for path, digest in zip(NUMERIC[task], numeric_shas[task], strict=True)},
                'Private sources do not match the four R5 numeric originals')
        aliases = {(r['task'], r['episode_id']): r['original_id'] for r in private['anonymous_mapping']}
        require(len(aliases) == len(private['anonymous_mapping']), 'Duplicate anonymous source alias')
        indexed = {'stair': {r['episode_id']: (i, r) for i, r in enumerate(fulls['stair']['episodes'])},
                   'push': {f"r{r['source']+1}.e{r['env_id']}": (i, r) for i, r in enumerate(fulls['push']['transcripts'])}}
        units = [u for u in packing['packets'] if u['axis'] == 'episode']
        require(Counter(u['task'] for u in units) == {'stair': 12, 'push': 12}
                and len({u['packet_id'] for u in units}) == 24
                and len({(u['task'], u['scope']) for u in units}) == 24, 'Expected fixed 12/task distinct episode packets')
        for unit in units:
            task, packet, scope = unit['task'], unit['packet_id'], unit['scope']
            manifest_path = PACK / unit['facts_manifest']
            facts = read(manifest_path, unit['facts_manifest_sha256'])
            require(facts['stage'] == 'facts' and facts['packet_id'] == packet, 'Wrong facts packet')
            raw_ref, = [r for r in facts['materials'] if r['role'] == 'raw_numeric']
            raw_path = (manifest_path.parent / raw_ref['file']).resolve()
            raw = read(raw_path, raw_ref['sha256'])
            require(raw['episode']['episode_id'] == scope and raw['episode']['task'] == task, 'Raw episode/packet differs')
            require(facts['reference_contract'] == 'p3new.reference_blocks.v2'
                    and facts['material_profile'] == raw['profile'] == 'related_v2'
                    and raw['schema_version'] == 'p3new.related_episode.v2', 'Wrong R5 facts representation')
            header_name = f'episodes/{scope}/episode.json'
            header_path = RAW / header_name
            header = read(header_path, raw_catalogue[header_name]['sha256'])
            model_header_ref, = [r for r in raw['source_materials'] if r['file'] == header_name]
            require(model_header_ref['sha256'] == raw_catalogue[header_name]['sha256'], 'Actual raw header reference differs')
            projection = projection_check.verify_episode_projection(raw['episode'], header['episode'],
                task=task, contract=projection_contract)
            projection_path = OUT / 'terminal_projection' / (packet + '.json')
            save(projection_path, projection)
            require(projection['status'] == 'PASS_EXACT_RETAINED_FIELDS', 'Actual terminal projection differs: ' + packet)
            logical_id = aliases[task, scope]
            index, episode = indexed[task][logical_id]
            shared = ({'domain': fulls[task]['domain'], 'semantics': fulls[task]['semantics'],
                       'contract': fulls[task]['semantic']['contract']} if task == 'stair' else
                      {k: fulls[task][k] for k in ('binding', 'comparison_domain', 'definitions', 'process_contract', 'source_bindings')})
            derivation = {'source': {'full': full_refs[task]},
                'source_pointer': f"/{'episodes' if task == 'stair' else 'transcripts'}/{index}",
                'episode_id': scope, 'original_logical_id': logical_id, 'shared': shared,
                'complete_episode_derivation': episode}
            derivation_path = OUT / 'inputs' / (packet + '.json')
            save(derivation_path, derivation)
            result = (stair.verify_stair_episode(raw, derivation) if task == 'stair' else
                      push.verify_push_confirmation_arithmetic(raw, derivation))
            result_path = OUT / 'arithmetic' / (packet + '.json')
            save(result_path, result)
            row = {'packet_id': packet, 'task': task, 'scope': scope, 'logical_id': logical_id,
                'facts_manifest': watched[str(manifest_path)], 'raw_numeric': watched[str(raw_path)],
                'full_derivation': ref(derivation_path), 'terminal_projection': ref(projection_path), 'result': ref(result_path), 'status': result['status'],
                'checked_count': result['checked_n'] if task == 'stair' else len(result['checked_field_paths']),
                'failed_count': result['failed_n'] if task == 'stair' else len(result['differences'])}
            rows.append(row)
            if task == 'stair':
                link = source_adapter.verify_stair_operand_sources(raw,
                    episode_header_path=RAW / 'episodes' / scope / 'episode.json',
                    raw_export_manifest_path=RAW / 'manifest.json', packing_receipt_path=PACK / 'packing_receipt.json')
                link_path = OUT / 'source_links' / (packet + '.json')
                save(link_path, link)
                row['source_link'] = {'status': link['status'], 'result': ref(link_path),
                    'failed_count': sum(c['status'] == 'fail' for c in link['checks'])}
            else:
                row['source_link'] = {'status': 'NOT_AVAILABLE_IN_FROZEN_HELPER',
                    'limit': 'This receipt contains only supplied-byte/metadata arithmetic; run the separate R5 Push archive source-link control and retain its own receipt.'}

        receipt['loaded_source_modules'] = {name: frozen_ref(Path(module.__file__).resolve(), source)
            for name, module in tuple(sys.modules.items())
            if (name in modules or name.startswith('true_rl_agent')) and getattr(module, '__file__', None)}
        require(all(ref(path) == original for path, original in watched.items()), 'An input/source changed during execution')
        failures = sum(r['failed_count'] + r['source_link'].get('failed_count', 0) for r in rows)
        nonpassing = [r['packet_id'] for r in rows if not r['status'].startswith('PASS_')
                      or (r['task'] == 'stair' and not r['source_link']['status'].startswith('PASS_'))]
        receipt.update(execution_status='COMPLETED', checked_episode_count=len(rows), failed_comparisons=failures,
            nonpassing_packets=nonpassing, check_status='FAIL' if failures or nonpassing else 'CHECKED_WITH_EXPLICIT_LIMITS',
            inputs_unchanged=True, whole_frozen_tree_rescanned=False,
            completed_at_utc=datetime.now(timezone.utc).isoformat())
        save(OUT / 'arithmetic_receipt.json', receipt)
        print(json.dumps({k: receipt[k] for k in ('status', 'execution_status', 'check_status', 'checked_episode_count', 'failed_comparisons')}))
        return 1 if failures or nonpassing else 0
    except Exception as error:
        receipt.update(execution_status='ERROR', error_type=type(error).__name__, error=str(error),
                       stopped_at_utc=datetime.now(timezone.utc).isoformat())
        with (OUT / 'exception.traceback.txt').open('x', encoding='utf-8') as handle:
            traceback.print_exc(file=handle)
        save(OUT / 'arithmetic_receipt.json', receipt)
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--card-sha256', required=True)
    args = parser.parse_args()
    CARD_SHA = args.card_sha256
    require(len(CARD_SHA) == 64 and all(c in '0123456789abcdef' for c in CARD_SHA), 'Invalid actual card SHA')
    raise SystemExit(main())
