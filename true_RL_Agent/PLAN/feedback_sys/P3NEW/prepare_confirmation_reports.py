"""Offline fixed N5: validate all 80 original facts, then own seals and report inputs.

Default is an audit only. --prepare-reports creates a NEW directory after the
whole engineering barrier passes; it neither certifies facts true nor releases
requests. No transport, retries, scores or automatic semantic acceptance.
See REPORT_PREPARATION.md for the explicit input table and remaining human gate.
"""
import argparse
import json
from pathlib import Path
import re
import shutil

import prepare_confirmation_materials as pack
from prepare_facts_offline import checked, output_cap, roster
import run_confirmation_review as runner


def same(reference, path):
    runner.require(reference['sha256'] == runner.ref(path)['sha256'], 'Bound original SHA differs: ' + str(path))


def audit_row(row, binding):
    paths = {key: checked(row[key]) for key in
             ('facts_manifest', 'prepared', 'request', 'reference_catalog', 'answer', 'driver_receipt', 'structure_audit')}
    card, manifest, request, answer, structure = (runner.read(paths[key]) for key in
                                                ('prepared', 'facts_manifest', 'request', 'answer', 'structure_audit'))
    packet, judge = row['packet_id'], row['judge_id']
    runner.require((card['packet_id'], card['judge_id'], card['stage']) == (packet, judge, 'facts')
                   and manifest['packet_id'] == packet and manifest['stage'] == 'facts', 'Facts identity differs')
    runner.require(card.get('review_assignment') == manifest.get('review_assignment') == binding,
                   'Facts assignment differs')
    runner.require(card.get('own_facts_seal') is None and card.get('capacity_facts_fixture') is None,
                   'Answers or synthetic facts leaked into facts input')
    same(card['manifest'], paths['facts_manifest'])
    runner.require(card['request_sha256'] == runner.ref(paths['request'])['sha256']
                   and card['request_bytes'] == paths['request'].stat().st_size, 'Original request differs')
    # Never reinterpret an old answer under a newly edited helper or protocol.
    same(card['coverage_helper'], Path(runner.coverage.__file__))
    same(card['runner'], Path(runner.__file__))
    rebuilt, _, rebuilt_card = runner.build(paths['facts_manifest'], judge)
    runner.require(rebuilt == paths['request'].read_bytes() and rebuilt_card == card,
                   'Prepared request no longer reproduces from its original sources')
    catalog = runner.read(paths['reference_catalog'])
    runner.require(catalog == runner.coverage.request_reference_catalog(request)
                   and answer.get('reference_catalog_sha256') == runner.sha(runner.coverage.canonical(catalog)),
                   'Original answer/catalog/request binding differs')
    runner.require(runner.envelope(answer, judge, packet, 'facts') and answer.get('complete') is True,
                   'Original facts incomplete')
    runner.require(structure.get('structural_contract_complete') is True and not structure.get('errors')
                   and (structure['packet_id'], structure['judge_id'], structure['stage']) == (packet, judge, 'facts'),
                   'Original structure audit incomplete')
    for key in ('prepared', 'request', 'answer'):
        same(structure['inputs'][key], paths[key])
    same(structure['inputs']['coverage_helper'], Path(runner.coverage.__file__))
    protocols = [m for m in manifest['materials'] if m['role'] == 'review_protocol']
    protocol_data = runner.coverage.request_materials(request)[0][protocols[0]['id']]['data'] if len(protocols) == 1 else {}
    runner.require(len(protocols) == 1 and structure['inputs']['protocol']['sha256'] == protocol_data.get('source_sha256'),
                   'Structure audit protocol differs')
    mechanical = runner.coverage.audit_answer(answer, request, stage='facts')
    runner.require(mechanical['mechanical_valid'] and mechanical['review_complete'], 'Original facts references/coverage incomplete')
    receipt = runner.read(paths['driver_receipt'])
    if judge == 'deepseek':
        runner.require(receipt['status'] == 'received' and receipt.get('complete') is True
                       and receipt.get('finish_reason') == 'stop' and not receipt.get('incomplete_reasons')
                       and (receipt['packet_id'], receipt['judge_id'], receipt['stage']) == (packet, judge, 'facts')
                       and receipt['request_sha256'] == card['request_sha256'], 'DeepSeek transport incomplete')
        same(receipt['content'], paths['answer'])
    else:
        runner.require(receipt['status'] == 'COMPLETED_TRANSPORT_VERIFIED_NOT_RELEASED'
                       and receipt.get('exit_code') == 0, 'SOL transport incomplete')
        same(receipt['raw_answer'], paths['answer'])
        registration_path, transport_path = checked(row['registration']), checked(row['transport_receipt'])
        same(receipt['registration'], registration_path)
        registration, transport = runner.read(registration_path), runner.read(transport_path)
        same(registration['transport'], transport_path)
        runner.require(registration['status'] == 'REGISTERED_ONE_CALL'
                       and transport['status'] == 'PREPARED_TRANSPORT_ONLY_NOT_RELEASED'
                       and not transport['development_only']
                       and (transport['packet_id'], transport['judge_id'], transport['stage']) == (packet, judge, 'facts'),
                       'SOL registration/original identity differs')
        same(transport['source_prepare'], paths['prepared'])
        same(transport['source_request'], paths['request'])
        same(transport['source_manifest'], paths['facts_manifest'])
    return paths


def audit_all(index):
    bindings = roster(index)
    expected = {(p, r['judge_id']) for p, binding in bindings.items() for r in binding['reviewers'].values()}
    rows = index['facts']
    runner.require(len(rows) == 80 and {(r['packet_id'], r['judge_id']) for r in rows} == expected,
                   'All 80 distinct assigned facts required before any seal/report')
    audited = {(r['packet_id'], r['judge_id']): audit_row(r, bindings[r['packet_id']]) for r in rows}
    # Independence keeps the two answers separate; the evidence must be identical.
    for packet, binding in bindings.items():
        pair = [audited[packet, r['judge_id']] for r in binding['reviewers'].values()]
        catalogs = [runner.read(p['reference_catalog']) for p in pair]
        materials = [runner.coverage.request_materials(runner.read(p['request']))[0] for p in pair]
        runner.require(catalogs[0] == catalogs[1] and materials[0] == materials[1], 'Paired raw materials/catalogs differ')
        unit = next(u for u in index['units'] if u['packet_id'] == packet)
        base = checked(unit['facts_manifest'])
        original, _, _ = runner.build(base, binding['reviewers']['A']['judge_id'])
        runner.require(runner.coverage.request_materials(json.loads(original))[0] == materials[0],
                       'Adopted facts differ from the registered original unit materials')
    return audited


def material(path, role, material_id, **extra):
    return {**pack.material(path, str(Path(path).resolve()), role, material_id), **extra}


def verify_source_index(index):
    """Bind task/axis/alias and numeric archive to the existing packing receipts."""
    numeric, media = (runner.read(checked(index[k])) for k in ('numeric_packing', 'media_packing'))
    runner.require(numeric['purpose'] == 'confirmation' and len(numeric['packets']) == 36
                   and len(media['units']) == 4
                   and numeric['source_sha256']['capture_card'] == media['capture_card']['sha256'],
                   'Numeric/media packing do not belong to the same fixed capture')
    for task in ('stair', 'push'):
        runner.require(index['tasks'][task]['raw_manifest']['sha256'] == numeric['source_sha256']['raw_manifest']
                       and index['tasks'][task]['raw_sources']['sha256'] == numeric['source_sha256']['private_source_manifest'],
                       'Report numeric source differs from original facts packing')
    for unit in index['units']:
        if unit['axis'] == 'media':
            matches = [r for r in media['units'] if r['unit_id'] == unit['packet_id']]
            runner.require(len(matches) == 1, 'Missing original media unit')
            row = matches[0]
            runner.require((unit['task'], unit['alias'], unit['original_id']) ==
                           (row['task'], row['alias'], f"r{row['source_run']}.e{row['source_env']}"), 'Media task/alias routing differs')
            expected_sha = row['facts_manifest']['sha256']
        else:
            matches = [r for r in numeric['packets'] if r['packet_id'] == unit['packet_id']]
            runner.require(len(matches) == 1, 'Missing original numeric unit')
            row = matches[0]
            runner.require(all(row[k] == unit[k] for k in ('task', 'axis', 'scope')), 'Numeric task/axis/scope routing differs')
            expected_sha = row['facts_manifest_sha256']
        runner.require(unit['facts_manifest']['sha256'] == expected_sha, 'Registered unit manifest differs from packing')


def task_materials(task, inputs, output):
    """One complete shared task source, reused by all slots; no focus filtering."""
    full_path, phase_path, request_path, raw_manifest, private_path = (
        checked(inputs[k]) for k in ('full', 'phase', 'request', 'raw_manifest', 'raw_sources'))
    raw_export = raw_manifest.parent
    runner.require(private_path == raw_export / '_private/sources.json', 'Raw source index location differs')
    full, phase, request, private = (runner.read(p) for p in (full_path, phase_path, request_path, private_path))
    runner.require(inputs['request_status'] == 'offline_serialized_not_sent', 'Explicit actual offline delivery identity required')
    same(phase['task_evidence']['behavior_full'], full_path)
    source_shas = {s['manifest_sha256'] for s in private['sources'] if s['source_id'].startswith(task)}
    runner.require(source_shas == ({full['domain']['archive_hash']} if task == 'stair'
                                  else {s['manifest_sha256'] for s in full['sources']}), 'Numeric/full sources differ')
    public = phase['task_evidence']['behavior']
    pointer = '/public_feedback_summary/behavior/profile' if task == 'stair' else '/public_feedback_summary/public_dev/behavior'
    located = json.loads(request['messages'][1]['content'])
    for part in pointer.strip('/').split('/'):
        located = located[part]
    runner.require(located == public and runner.encoded(request) == request_path.read_bytes(), 'Actual public/request differs')
    aliases = {r['episode_id']: r['original_id'] for r in private['anonymous_mapping'] if r['task'] == task}
    if task == 'stair':
        shared = {'domain': full['domain'], 'semantics': full['semantics'], 'contract': full['semantic']['contract']}
        rows = {r['episode_id']: (i, r) for i, r in enumerate(full['episodes'])}
    else:
        shared = {k: full[k] for k in ('binding', 'comparison_domain', 'definitions', 'process_contract', 'source_bindings')}
        rows = {f"r{r['source']+1}.e{r['env_id']}": (i, r) for i, r in enumerate(full['transcripts'])}
    output.mkdir(parents=True)
    provenance = {k: inputs[k] for k in ('full', 'phase', 'request', 'raw_manifest', 'raw_sources', 'request_status')}
    logical_ids = sorted(set(re.findall(r'\br\d+\.e\d+\b', json.dumps(public['report']))))
    documents = {
        'pool': pack.report_pool_operands(raw_export, task, aliases),
        'examples': pack.report_examples(raw_export, task, logical_ids, full, aliases),
        'shared': {'source': provenance, 'shared': shared,
                   'slice_scope': 'Shared definitions; complete episode derivations are assessed in episode units. All public/request clauses are assessed here against whole-task operands and every referenced example, across all focus scopes.'},
        'public': public,
        'locator': {'request_material_id': 'report-final-request', 'request_sha256': runner.ref(request_path)['sha256'],
                    'decode_json_string_at': '/messages/1/content', 'then_json_pointer': pointer,
                    'decoded_object_canonical_sha256': runner.sha(runner.encoded(public)),
                    'scored_scope': 'All clauses in the complete default report and corresponding final request.'}}
    refs = {}
    for name, data in documents.items():
        path = output / (name + '.json')
        runner.save(path, data, exclusive=True)
        refs[name] = path
    return dict(refs=refs, full=full, rows=rows, shared=shared, provenance=provenance, aliases=aliases,
                request=request_path, public=public, logical_ids=logical_ids)


def select_materials(unit, data, output, original_materials=None):
    """Append axis materials; prepare_report_manifest retains EVERY original raw item."""
    output.mkdir(parents=True)
    axis = unit['axis']
    extra = []
    if axis == 'episode':
        original = data['aliases'][unit['scope']]
        index, row = data['rows'][original]
        path = output / 'episode.json'
        runner.save(path, {'source': data['provenance'], 'source_pointer': f"/{'episodes' if unit['task'] == 'stair' else 'transcripts'}/{index}",
            'episode_id': unit['scope'], 'original_logical_id': original, 'shared': data['shared'],
            'complete_episode_derivation': pack.review_episode_derivation(row, task=unit['task'], material_profile='related_v2')}, exclusive=True)
        extra.append(material(path, 'full_derivation', 'report-complete-episode'))
    elif axis == 'focus':
        for key, role in (('pool', 'raw_numeric'), ('examples', 'raw_numeric'), ('shared', 'full_derivation'), ('locator', 'automatic_report')):
            extra.append(material(data['refs'][key], role, 'report-' + key))
        extra.append(material(data['request'], 'bounded_request', 'report-final-request', label='Complete actual offline-serialized author request; never sent'))
    elif axis == 'media':
        raw_path, audit_path = checked(unit['observer_output']), checked(unit['observer_audit'])
        raw, audit = runner.read(raw_path), runner.read(audit_path)
        runner.require(audit['raw_output'] == raw and 'semantic_review' not in audit,
                       'Use original observer answer and original mechanical audit, not root verdicts')
        episodes = raw['episodes']
        runner.require(len(episodes) == 1 and episodes[0]['alias'] == unit['alias'], 'Media alias differs')
        runner.require(original_materials is not None, 'Original media facts required')
        image_shas = {m['sha256'] for m in original_materials.values() if m['role'] == 'raw_image'}
        metadata = [m['data'] for m in original_materials.values() if m['role'] == 'raw_media_metadata']
        runner.require(image_shas and any(m.get('alias') == unit['alias'] and
                       {i['sha256'] for i in m['images']} == image_shas for m in metadata), 'Original media image/alias binding differs')
        runner.require(all(o['alias'] == unit['alias'] and o['media_identity']['episode_id'] == unit['original_id']
                           and all(i['image_sha256'] in image_shas for i in o['image_evidence'])
                           for o in audit['observations']), 'Observer audit references another media unit')
        roots, observation_map, hidden_map = [], [], []
        for i, observation in enumerate(episodes[0]['observations']):
            pointer = f'/data/episodes/0/observations/{i}'
            runner.require(all(isinstance(observation.get(k), str) and observation[k] for k in ('text', 'uncertainty')),
                           'Original observation text/uncertainty missing')
            leaves = [{'json_pointer': pointer + '/' + k} for k in ('text', 'uncertainty')]
            roots.extend(leaves)
            observation_map.append({'original_observation': pointer, 'text_roots': leaves,
                                    'preserved_context_fields': list(observation)})
        for i, hidden in enumerate(episodes[0]['not_visible']):
            pointer = f'/data/episodes/0/not_visible/{i}'
            if isinstance(hidden, str):
                leaves = [{'json_pointer': pointer}]
            else:
                leaves = [{'json_pointer': pointer + '/' + k} for k in ('reason', 'statement')
                          if isinstance(hidden, dict) and isinstance(hidden.get(k), str)]
            runner.require(leaves, 'Original not-visible statement missing')
            roots.extend(leaves)
            hidden_map.append({'original_not_visible': pointer, 'text_roots': leaves})
        runner.require(len(observation_map) == len(episodes[0]['observations'])
                       and len(hidden_map) == len(episodes[0]['not_visible']), 'Original observer clause omitted')
        runner.save(output / 'observer_inventory_mapping.json', {'original_answer': runner.ref(raw_path),
            'observations': observation_map, 'not_visible': hidden_map,
            'scope': 'All original statements retained. Body, relation, modality, time and image references remain constraints of each whole observation; absence from a separate text denominator does not exempt them from support review.'}, exclusive=True)
        extra.append(material(raw_path, 'observer_output', 'report-original-observer', review_text_roots=roots,
            label='Original complete observer output. Each text/uncertainty is read with the same observation body, relation, modality, time and evidence references; those fields constrain whole-statement support, not separate extra text counts.'))
        extra.append(material(audit_path, 'arithmetic_audit', 'report-original-numeric-audit', label='Mechanical checks, not semantic truth'))
        # Full actual public remains byte-for-byte represented. Only this alias's
        # projection plus recorded admission counts are scored with these images.
        visual = data['public']['report'].get('visual', [])
        sentences = data['full']['semantic']['sentences'] if unit['task'] == 'stair' else data['full']['sentences']
        own_ids = {s['id'] for s in sentences if s.get('media_identity', {}).get('episode_id') == unit['original_id']}
        public_roots = [{'json_pointer': f'/data/report/visual/{i}/text'} for i, item in enumerate(visual)
                        if item['id'] in own_ids or item['id'] == 'visual.coverage']
        extra.append(material(data['refs']['public'], 'automatic_report', 'report-actual-public',
            review_text_roots=public_roots,
            label='Complete actual public. visual.coverage records admission decisions; counts are not ground truth of visual support. Other-alias clauses lack their images here and are outside this unit.'))
    else:
        raise ValueError('Unknown fixed axis')
    for i, reference in enumerate(unit.get('arithmetic_audits', [])):
        extra.append(material(checked(reference), 'arithmetic_audit', f'report-arithmetic-{i}', label='Existing engineering check; no semantic truth vote'))
    return extra


def prepare(index, output):
    audited = audit_all(index)  # No output directory or partial seals before the full barrier.
    verify_source_index(index)
    output = Path(output).resolve()
    runner.require(not output.exists(), 'Fresh output required; no resume/overwrite')
    output.mkdir(parents=True)
    try:
        # Seal all 80 before ANY report is prepared. Original bytes stay in place.
        seals = {}
        for (packet, judge), paths in audited.items():
            manifest = runner.read(paths['facts_manifest'])
            path = output / 'seals' / packet / (judge + '.json')
            path.parent.mkdir(parents=True, exist_ok=True)
            runner.save(path, {'schema_version': 'p3new.confirmation_facts_seal.v1', 'status': 'SEALED',
                'judge_id': judge, 'packet_id': packet, 'facts_manifest_sha256': runner.ref(paths['facts_manifest'])['sha256'],
                'answer': runner.ref(paths['answer']), 'reference_contract': manifest['reference_contract'],
                'reference_catalog': runner.ref(paths['reference_catalog']),
                'scope': 'Original facts preservation only; necessity, truth and completeness of semantic reading are not certified'}, exclusive=True)
            seals[packet, judge] = path
        tasks = {task: task_materials(task, index['tasks'][task], output / 'shared' / task) for task in ('stair', 'push')}
        result = []
        for unit in index['units']:
            packet = unit['packet_id']
            first = next(paths for key, paths in audited.items() if key[0] == packet)
            original_materials = runner.coverage.request_materials(runner.read(first['request']))[0]
            materials = select_materials(unit, tasks[unit['task']], output / 'materials' / packet, original_materials)
            for judge in [key[1] for key in audited if key[0] == packet]:
                destination = output / 'reports' / packet / judge
                destination.mkdir(parents=True)
                manifest_path = destination / 'report.manifest.json'
                pack.prepare_report_manifest(audited[packet, judge]['facts_manifest'], materials, manifest_path)
                manifest = runner.read(manifest_path)
                manifest['max_output_tokens'] = output_cap(judge)
                runner.save(manifest_path, manifest)
                card = runner.prepare(manifest_path, judge, destination / 'prepared', own_facts_seal=seals[packet, judge])
                result.append({'packet_id': packet, 'judge_id': judge, 'seal': runner.ref(seals[packet, judge]),
                               'prepared': runner.ref(destination / 'prepared/prepare.json'),
                               'request_bytes': card['request_bytes'], 'max_output_tokens': card['max_output_tokens']})
        receipt = {'status': 'PREPARED_NOT_RELEASED', 'assignment': index['assignment'], 'facts_inputs': index['facts'],
                   'source_index': {k: index[k] for k in ('numeric_packing', 'media_packing', 'tasks', 'units')},
                   'reports': result, 'api_calls': 0, 'semantic_acceptance': None, 'capacity_release': False,
                   'media_inventory_bindings': [runner.ref(output / 'materials' / unit['packet_id'] / 'observer_inventory_mapping.json')
                                                for unit in index['units'] if unit['axis'] == 'media'],
                   'pending': 'Actual own-facts request/carrier token and image capacity, operator semantic-scope review and explicit release remain required.'}
        runner.save(output / 'preparation.json', receipt, exclusive=True)
        return receipt
    except BaseException:
        runner.require(output.is_dir() and not output.is_symlink(), 'Unsafe cleanup target')
        shutil.rmtree(output)  # Only the exclusively created output; never original files.
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare-reports', action='store_true')
    args = parser.parse_args()
    index = runner.read(args.index)
    if args.prepare_reports:
        prepare(index, args.output)
    else:
        audited = audit_all(index)
        runner.save(args.output, {'status': 'ENGINEERING_80_COMPLETE_NOT_SEMANTIC_ACCEPTANCE',
                    'facts_inputs': index['facts'], 'rows': len(audited), 'seals': 0, 'reports': 0,
                    'api_calls': 0, 'semantic_acceptance': None}, exclusive=True)
