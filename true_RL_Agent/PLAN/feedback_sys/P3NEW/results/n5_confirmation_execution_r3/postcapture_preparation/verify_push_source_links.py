"""Read-only R3 Push provenance supplement; stdout receipt, no source writes.

Imports no exporter, packer, production reducer, selector, or review helper.
This authenticates recorded bytes and identity, not their physical semantics.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from terminal_projection_check import load_contract, verify_episode_projection
from postcapture_cpu import check_freeze

B = Path('/data/yuyangliu/AGENT/feedback_p3new_20260929')
C = B / 'n5_confirmation_20260930_r3'
E = B / 'n5_execution_20260930_r3'
watched, differences = {}, []


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_bytes(path, expected=None):
    path = Path(path).resolve()
    data = path.read_bytes()
    digest = sha(data)
    if expected is not None and digest != expected:
        raise ValueError(f'input SHA mismatch: {path}')
    watched[str(path)] = {'path': str(path), 'sha256': digest, 'bytes': len(data)}
    return data


def load(path, expected=None):
    return json.loads(read_bytes(path, expected))


def check(label, actual, expected):
    same = actual == expected
    if not same:
        differences.append({'check': label, 'actual': actual, 'expected': expected})
    return same


def signature(array):
    return {'dtype': array.dtype.str, 'shape': list(array.shape),
            'c_order_bytes_sha256': sha(array.tobytes(order='C'))}


def decode(spec, n):
    shape = (n, *spec['component_shape'])
    dtype = np.dtype(spec['dtype'])
    if 'by_c_order_component' in spec:
        columns = [np.asarray(s.split(','), dtype=dtype) for s in spec['by_c_order_component']]
        return np.stack(columns, axis=1).reshape(shape)
    array = np.empty(shape, dtype=dtype)
    cursor = 1
    for start, end, value in spec['inclusive_runs']:
        if start != cursor or not start <= end <= n:
            raise ValueError('non-contiguous related raw RLE')
        array[start-1:end] = value
        cursor = end+1
    if cursor != n+1:
        raise ValueError('related raw RLE does not reach T')
    return array


def main():
    source_root = Path('/data/yuyangliu/AGENT/feedback_p3new_20260929/n5_confirmation_source_r3_final')
    contract = load_contract(source_root, E/'runtime_source_manifest.json')
    read_bytes(Path(__file__))
    read_bytes(Path(__file__).with_name('terminal_projection_check.py'))
    read_bytes(Path(contract['policy_source']['path']), contract['policy_source']['sha256'])
    export_manifest = load(C/'numeric_raw/manifest.json')
    private_ref = export_manifest['private_source_manifest']
    private = load(C/'numeric_raw'/private_ref['file'], private_ref['sha256'])
    packing = load(C/'numeric_materials/packing_receipt.json')
    check('packing/private SHA', packing['raw_private_source_binding']['sha256'], private_ref['sha256'])
    check('R3 raw profile', export_manifest['material_profile'], 'related_v1')
    check('R3 packed profile', packing['material_profile'], 'related_v2')
    card = load(E/'capture_card.json')
    arithmetic = load(E/'control/numeric_arithmetic/arithmetic_receipt.json')
    arithmetic_rows = {r['packet_id']: r for r in arithmetic['arithmetic']}
    readable = {r['file']: r for r in export_manifest['readable_materials']}
    aliases = {r['episode_id']: r['original_id'] for r in private['anonymous_mapping'] if r['task']=='push'}
    selected = {r['id']: r for r in private['selection_original_ids'] if r['task']=='push'}
    export_rows = {r['episode_id']: r for r in private['json_npz_roundtrip_checks']}
    full_path = C/'push/evaluation/public_dev/behavior_full.json'
    full = load(full_path)
    runs = {}
    for source in private['sources']:
        if not source['source_id'].startswith('push_'):
            continue
        path = Path(source['manifest_path'])
        manifest = load(path, source['manifest_sha256'])
        identity = manifest['identity']
        run = source['run']
        check(f'run {run}/identity', source['identity'], identity)
        check(f'run {run}/run_index', identity['run_index'], run)
        check(f'run {run}/task_id', identity['task_id'], 'unitree_push_recovery')
        check(f'run {run}/capture_binding', identity['evaluation_binding'], card['tasks']['push']['current_evaluation_binding'])
        check(f'run {run}/checkpoint', identity['checkpoint_hash'], 'sha256:'+card['tasks']['push']['checkpoint']['sha256'])
        values, chunks, offset = {}, [], 0
        for chunk in manifest['chunks']:
            npz_path = path.parent/chunk['file']
            read_bytes(npz_path, chunk['sha256'])
            with np.load(npz_path, allow_pickle=False) as archive:
                count = len(archive['valid'])
                for name in archive.files:
                    a = archive[name]
                    check(f'{run}/{chunk["file"]}/{name}/schema',
                          {'dtype': a.dtype.str, 'shape': list(a.shape)}, chunk['arrays'][name])
                    values.setdefault(name, []).append(a.copy())
            chunks.append({'path': str(npz_path), 'sha256': chunk['sha256'],
                           'archive_rows_0based_inclusive': [offset, offset+count-1]})
            offset += count
        arrays = {k: np.concatenate(v) for k,v in values.items()}
        runs[run] = {'source': source, 'manifest': manifest, 'arrays': arrays, 'chunks': chunks}

    packets = [r for r in packing['packets'] if r['task']=='push' and r['axis']=='episode']
    check('fixed Push packet count', len(packets), 12)
    result_rows = []
    for packet in packets:
        before = len(differences)
        pid, alias = packet['packet_id'], packet['scope']
        original = aliases[alias]
        selection = selected[original]
        run, env = selection['run'], selection['env']
        data = runs[run]
        manifest, arrays = data['manifest'], data['arrays']
        meta, identity = manifest['metadata'], manifest['identity']
        check(f'{pid}/logical_id', original, f'r{run}.e{env}')
        check(f'{pid}/selection source', selection['source_id'], data['source']['source_id'])
        terminal = meta['runtime_episode_records'][env]
        check(f'{pid}/terminal env', terminal['env_id'], env)
        positions = np.flatnonzero(arrays['valid'][:,env])
        raw = {k: a[positions,env].copy() for k,a in arrays.items()}
        T = len(positions)
        check(f'{pid}/terminal length', T, terminal['episode_steps'])
        check(f'{pid}/clock 1..T', raw['control_step'].tolist(), list(range(1,T+1)))
        check(f'{pid}/episode index', np.unique(raw['episode_index']).tolist(), [terminal['episode_index']])
        check(f'{pid}/export row indices', positions.tolist(), export_rows[alias]['archive_row_indices'])
        header_rel = f'episodes/{alias}/episode.json'
        header_path = C/'numeric_raw'/header_rel
        header = load(header_path, readable[header_rel]['sha256'])
        check(f'{pid}/raw fields', sorted(header['raw_fields']), sorted(raw))
        check(f'{pid}/header task', header['episode']['task'], 'push')
        check(f'{pid}/header alias', header['episode']['episode_id'], alias)
        check(f'{pid}/header run', header['episode']['run_group'], f'run_{run:02d}')
        check(f'{pid}/terminal copy', header['episode']['original_terminal_record'],
              {k:v for k,v in terminal.items() if k not in ('env_id','episode_index')})
        exported_npz = header_path.parent/header['original_values_npz']['file']
        read_bytes(exported_npz, header['original_values_npz']['sha256'])
        parts = [load(header_path.parent/p['file'], p['sha256']) for p in header['parts']]
        for p in parts:
            check(f'{pid}/part alias', p['episode_id'], alias)
        fact_path = C/'numeric_materials'/packet['facts_manifest']
        fact = load(fact_path, packet['facts_manifest_sha256'])
        raw_ref, = [r for r in fact['materials'] if r['role']=='raw_numeric']
        related_path = (fact_path.parent/raw_ref['file']).resolve()
        related = load(related_path, raw_ref['sha256'])
        check(f'{pid}/fact identity', [fact['packet_id'],fact['stage']], [pid,'facts'])
        projection = verify_episode_projection(related['episode'], header['episode'], task='push', contract=contract)
        check(f'{pid}/related exact retained header fields', projection['status'], 'PASS_EXACT_RETAINED_FIELDS')
        check(f'{pid}/related profile', related['profile'], 'related_v2')
        check(f'{pid}/facts reference contract', fact['reference_contract'], 'p3new.reference_blocks.v2')
        check(f'{pid}/related definitions', related['definitions'], header['definitions'])
        for key in ('schema_version','profile','capture_point','time','pose_fields','condition_fields','planar_motion_fields',
                    'direction_status','dwell_scope','coordinates','parameters','contact_status','episode_coverage'):
            check(f'{pid}/original definition/{key}', related['definitions'][key], meta[key])
        decoded = {k:decode(v,T) for k,v in {**related['raw_columns'],**related['raw_discrete']}.items()}
        clock = related['clock']
        decoded['control_step'] = np.arange(clock['first_step'],clock['last_step']+1,clock['stride'],dtype=np.int64)
        check(f'{pid}/related fields', sorted(decoded), sorted(raw))
        fields = []
        with np.load(exported_npz, allow_pickle=False) as archive:
            for name, original_values in sorted(raw.items()):
                expected = signature(original_values)
                stages = {'source': expected,
                          'export_npz': signature(archive['raw__'+name]),
                          'all_json_parts': signature(np.concatenate([np.asarray(p['raw'][name],dtype=original_values.dtype) for p in parts])),
                          'related_decoded': signature(decoded[name])}
                for stage, observed in stages.items():
                    check(f'{pid}/{name}/{stage}', observed, expected)
                check(f'{pid}/{name}/header hash', header['raw_fields'][name], expected)
                check(f'{pid}/{name}/related hash', related['source_fields'][name], expected)
                fields.append({'field':name,**expected,'identical_stages':list(stages)})
        ar = arithmetic_rows[pid]
        derivation_path = Path(ar['full_derivation']['path'])
        derivation = load(derivation_path, ar['full_derivation']['sha256'])
        full_ref = derivation['source']['full']
        check(f'{pid}/full path', str(Path(full_ref['path']).resolve()), str(full_path.resolve()))
        check(f'{pid}/full SHA', full_ref['sha256'], watched[str(full_path.resolve())]['sha256'])
        node = full
        for part in derivation['source_pointer'].strip('/').split('/'):
            node = node[int(part)] if isinstance(node,list) else node[part]
        check(f'{pid}/full pointer exact object', derivation['complete_episode_derivation'], node)
        check(f'{pid}/derivation alias', derivation['episode_id'], alias)
        check(f'{pid}/derivation original', derivation['original_logical_id'], original)
        check(f'{pid}/env_id', node['env_id'], env)
        check(f'{pid}/episode_index', node['episode_index'], terminal['episode_index'])
        binding, = [r for r in full['source_bindings'] if r['source_index']==node['source']]
        check(f'{pid}/binding actual run', binding['actual_run_index'], run)
        check(f'{pid}/binding logical run', binding['logical_run'], run)
        check(f'{pid}/binding manifest', binding['manifest_sha256'], data['source']['manifest_sha256'])
        check(f'{pid}/shared source bindings', derivation['shared']['source_bindings'], full['source_bindings'])
        for key in ('checkpoint_hash','evaluation_plan_hash','program_source_checksum','resolved_cell_manifest_hash','resolved_protocol_hash','runtime_dynamics_hash'):
            check(f'{pid}/full binding/{key}', full['binding'][key], identity[key])
        segments = []
        for chunk in data['chunks']:
            low,high = chunk['archive_rows_0based_inclusive']
            selected_rows = positions[(positions>=low)&(positions<=high)]
            if selected_rows.size:
                segments.append({**chunk,'selected_chunk_rows_0based':(selected_rows-low).tolist(),
                                 'selected_archive_rows_0based':selected_rows.tolist()})
        result_rows.append({'packet_id':pid,'task':'push','anonymous_id':alias,'original_logical_id':original,
            'run_index':run,'env_id_0based':env,'steps':T,'original_manifest':watched[str(Path(data['source']['manifest_path']).resolve())],
            'original_npz_segments':segments,'export_npz':watched[str(exported_npz.resolve())],
            'header':watched[str(header_path.resolve())],'facts_manifest':watched[str(fact_path.resolve())],
            'related_raw':watched[str(related_path.resolve())],'derivation':watched[str(derivation_path.resolve())],
            'full_source_pointer':derivation['source_pointer'],'full_source_index':node['source'],
            'field_byte_checks':fields,'terminal_projection':projection,'difference_count':len(differences)-before,
            'status':'MATCH' if len(differences)==before else 'MISMATCH'})
    unchanged = all(sha(Path(p).read_bytes())==r['sha256'] for p,r in watched.items())
    check('all inputs unchanged',unchanged,True)
    result = {'schema_version':'p3new.push_source_link_supplement.v1','at_utc':datetime.now(timezone.utc).isoformat(),
        'status':'PASS_RECORDED_BYTES_AND_IDENTITY' if not differences else 'FAIL',
        'read_only_source_audit':True,'source_writes':0,'model_calls':0,'simulation_calls':0,
        'production_imports':[],'exporter_or_packer_imports':[],
        'checked_packets':len(result_rows),'checked_raw_field_chains':sum(len(r['field_byte_checks']) for r in result_rows),
        'differences':differences,'inputs_unchanged':unchanged,'input_refs':list(watched.values()),'packets':result_rows,
        'limits':['Recorded NPZ bytes and task/run/env identity only; not independent physical pose construction.',
                  'No quaternion/full angular velocity/reset-origin reconstruction or action norm recomputation.',
                  'No independent physical terminal classification, semantic vote, necessary-fact score or confirmation admission.',
                  'Supplement is separate from frozen original arithmetic results; those retain their original unchecked fields.']}
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0 if not differences else 1


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--card-sha256', required=True)
    args = parser.parse_args()
    check_freeze(args.card_sha256)
    sys.exit(main())
