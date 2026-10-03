"""Independent original-NPZ/actual-historical-triangle oracle; no production imports."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from tests.unit.terrain_mesh_oracle import mesh_height, check_record, check_first_extrema, close


def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def identifier(kind, *parts):
    return kind+':'+hashlib.sha256(json.dumps(parts, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def distribution(values):
    values = sorted(values)
    def q(p):
        if not values: return None
        pos = (len(values)-1)*p; lo = int(pos)
        return values[lo]+(values[min(lo+1, len(values)-1)]-values[lo])*(pos-lo)
    return {'count': len(values), 'quantile_method': 'linear', 'minimum': min(values) if values else None,
            'maximum': max(values) if values else None, 'mean': sum(values)/len(values) if values else None,
            'q25': q(.25), 'median': q(.5), 'q75': q(.75)}


def raw_episodes(source):
    path = ROOT/source['manifest_relative_path']
    assert sha(path) == source['manifest_sha256']
    m = read(path)
    assert m['identity'] == source['identity'] and m['identity']['splits'] == ['public_dev']
    locks = {c['file']: c['sha256'] for c in source['chunks']}
    for run, seed in enumerate(m['metadata']['seeds'], 1):
        assert seed['split'] == 'public_dev' and seed['run_index'] == run
        chunks = [c for c in m['chunks'] if c['metadata']['run_index'] == run]
        parts = {k: [] for k in ('base_pos_w', 'support_pos_w', 'valid', 'episode_length')}
        origins = types = None
        for c in chunks:
            assert sha(path.parent/c['file']) == c['sha256'] == locks[c['file']]
            with np.load(path.parent/c['file'], allow_pickle=False) as z:
                for key in parts: parts[key].append(z[key].copy())
                if origins is None: origins, types = z['terrain_env_origins_w'].copy(), z['terrain_types'].copy()
        arrays = {k: np.concatenate(v) for k, v in parts.items()}
        for env, n in enumerate(seed['episode_steps']):
            assert arrays['valid'][:, env].tolist() == [True]*n+[False]*(len(arrays['valid'])-n)
            assert arrays['episode_length'][:n, env].tolist() == list(range(1, n+1))
            positions = np.concatenate([arrays['base_pos_w'][:n, env, None, :], arrays['support_pos_w'][:n, env]], axis=1)
            yield {'manifest': m, 'run': run, 'env': env, 'n': n, 'positions': positions.astype(np.float64),
                   'origin': origins[env].astype(np.float64), 'type': int(types[env]),
                   'dt': seed['step_dt'], 'chunks': chunks}


def audit(source, directory, historical, audit_output=None):
    started = time.perf_counter()
    build, report = read(directory/'manifest.json'), read(directory/'report.json')
    for name, info in build['artifacts'].items():
        assert sha(directory/name) == info['sha256'] and (directory/name).stat().st_size == info['bytes']
    assert report['binding'] == build['binding'] and report['binding']['manifest_sha256'] == source['manifest_sha256']
    assert report['binding']['identity'] == source['identity']
    assert report['status'] == 'offline_not_production_delivery'
    assert report['binding']['runtime_surface'] == 'unknown_not_recorded'
    assert report['catalog_version'] == 'trajectory.p0.measurements.v5'
    assert report['base_catalog_version'] == 'trajectory.p0.measurements.v2'
    config = {'boundary_guard_m': 1e-4, 'surface': 'reconstructed_nominal_exterior_top',
              'boundary_rule': 'closed_float_constructed_coordinate_intervals', 'runtime_surface': 'unknown_not_recorded'}
    assert report['measurement_config'] == config
    assert report['status_codes'] == ['known', 'nonfinite_position', 'missing_origin', 'unqualified_geometry', 'outside_patch', 'boundary_band']
    cells = {c['cell_id']: c for c in historical['cells']}
    episodes = list(raw_episodes(source))
    # Batch each actual cell's XY queries; oracle solves its actual top triangles.
    grouped = defaultdict(list)
    for ep in episodes:
        cell = cells[historical['env_cell_ids'][ep['env']]]
        assert ep['type'] == cell['terrain_type']
        np.testing.assert_allclose(ep['origin'], cell['expected_origin_w'], atol=1e-4, rtol=0)
        grouped[cell['cell_id']].append(ep)
    for cid, eps in grouped.items():
        points = np.concatenate([(ep['positions']-ep['origin']).reshape(-1, 3) for ep in eps])
        h, relative, status = mesh_height(points, [0, 0, 0], cells[cid])
        offset = 0
        for ep in eps:
            size = ep['n']*5; sl = slice(offset, offset+size)
            ep['expected'] = (h[sl].reshape(-1, 5)+ep['origin'][2], relative[sl].reshape(-1, 5), status[sl].reshape(-1, 5))
            offset += size
    totals, examples, records_checked, offset, maximum_error = {}, {}, 0, 0, 0.
    extreme_index_differences = []
    with np.load(directory/'traces.npz', allow_pickle=False) as stored:
        trace = {key: stored[key] for key in stored.files}
    with (directory/'terrain.jsonl').open(encoding='utf-8') as stream:
        for ep in episodes:
            row = json.loads(next(stream)); m, run, env, n, dt = (ep[k] for k in ('manifest', 'run', 'env', 'n', 'dt'))
            cell = cells[historical['env_cell_ids'][env]]
            names = ['robot_0.root', *['robot_0.support.'+x for x in m['metadata']['runtime_projection']['resolved_roles']['support_effector']]]
            eid = identifier('episode', source['manifest_sha256'], m['identity']['job_id'], run, env)
            assert (row['episode_id'], row['run_index'], row['env_index'], row['dt_s']) == (eid, run, env, dt)
            assert row['runner_seed'] == m['metadata']['seeds'][run-1]['runner_seed']
            assert row['trace_rows_half_open'] == [offset, offset+n] and len(row['records']) == 5
            h, relative, status = ep['expected']
            for name, expected in zip(('height_world_m', 'relative_z_m', 'status'), ep['expected'], strict=True):
                got = trace[name][offset:offset+n]
                if name == 'status': np.testing.assert_array_equal(got, expected)
                else:
                    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-10, equal_nan=True)
                    maximum_error = max(maximum_error, float(np.nanmax(np.abs(got-expected))))
            offset += n
            for col, (entity, record) in enumerate(zip(names, row['records'], strict=True)):
                channel = 'base_pos_w' if col == 0 else 'support_pos_w'
                assert record['episode_id'] == eid and record['measurement_config'] == config
                assert record['record_id'] == identifier('nominal-terrain', {k: v for k, v in record.items() if k != 'record_id'})
                wanted = check_record(record, ep['positions'][:, col], h[:, col], relative[:, col], status[:, col], dt, entity)
                check_first_extrema(record, trace['height_world_m'][offset-n:offset, col],
                                    trace['relative_z_m'][offset-n:offset, col], np.arange(1, n+1))
                refs = [{'file': c['file'], 'sha256': c['sha256'], 'env_index': env,
                         'rows_half_open': [0, min(c['metadata']['steps'], n-c['metadata']['step_start']+1)]}
                        for c in ep['chunks'] if c['metadata']['step_start'] <= n]
                expected_ref = {'manifest_sha256': source['manifest_sha256'], 'run_index': run, 'channels': [channel],
                                'steps_half_open': [1, n+1], 'chunks': refs}
                assert record['evidence_ref'] == expected_ref
                first = ep['chunks'][0]
                static_refs = {key: {'file': first['file'], 'sha256': first['sha256'], 'array': key, 'array_index': [env]}
                               for key in ('terrain_env_origins_w', 'terrain_types')}
                assert record['value']['geometry_binding'] == {
                    'cell_id': cell['cell_id'], 'terrain_type': ep['type'], 'geometry_digest': cell['geometry_digest'],
                    'geometry': cell['geometry'], 'regime': cell['regime'], 'origin_world_m': ep['origin'].tolist(), 'static_refs': static_refs}
                known_indices = np.flatnonzero(status[:, col] == 0)
                indices = {key: int(known_indices[op(values[known_indices, col])]) if len(known_indices) else None
                           for key, values, op in [('height_min', h, np.argmin), ('height_max', h, np.argmax),
                                                  ('relative_min', relative, np.argmin), ('relative_max', relative, np.argmax)]}
                for key, index in indices.items():
                    witness = record['value']['witnesses'][key]
                    if index is not None and witness['step'] != index+1:
                        values = h if key.startswith('height') else relative
                        delta = float(values[witness['step']-1, col]-values[index, col])
                        assert abs(delta) <= 1e-10
                        extreme_index_differences.append({'run': run, 'env': env, 'entity': entity, 'extremum': key,
                            'independent_mesh_first_step': index+1, 'verified_trace_first_step': witness['step'],
                            'independent_height_difference_m': delta})
                def check_witness(w, index):
                    if index is None: assert w is None; return
                    step = index+1
                    c = next(c for c in ep['chunks'] if c['metadata']['step_start'] <= step < c['metadata']['step_start']+c['metadata']['steps'])
                    raw_row = step-c['metadata']['step_start']
                    expected = {'step': step, 'position_world_m': ep['positions'][index, col].tolist(),
                        'height_world_m': float(h[index, col]) if status[index, col] == 0 else None,
                        'relative_z_m': float(relative[index, col]) if status[index, col] == 0 else None,
                        'status': report['status_codes'][int(status[index, col])], 'column': None if col == 0 else col-1,
                        'source_ref': {**expected_ref, 'steps_half_open': [step, step+1], 'chunks': [{
                            'file': c['file'], 'sha256': c['sha256'], 'env_index': env, 'rows_half_open': [raw_row, raw_row+1]}]}}
                    close(w, expected)
                for key, index in indices.items(): check_witness(record['value']['witnesses'][key], index)
                for code, name in enumerate(report['status_codes']):
                    selected = np.flatnonzero(status[:, col] == code)
                    if len(selected): check_witness(record['value']['first_by_status'][name], int(selected[0]))
                    else: assert name not in record['value']['first_by_status']
                totals.setdefault(cell['regime']+'/'+entity, []).append(wanted)
                # Predeclared categories; select lexicographic first across all five entities.
                local_x = ep['positions'][:, col, 0]-ep['origin'][0]+cell['mesh_origin'][0]
                start = cell['geometry']['start_platform_length_m']
                end = start+sum(cell['geometry']['tread_depths_m'])
                masks = {'start': (status[:, col] == 0) & (local_x < start),
                         'tread': (status[:, col] == 0) & (local_x > start) & (local_x < end),
                         'top': (status[:, col] == 0) & (local_x > end), 'unknown': status[:, col] != 0,
                         'negative_difference': (status[:, col] == 0) & (relative[:, col] < 0)}
                for category, mask in masks.items():
                    ix = np.flatnonzero(mask)
                    if not len(ix): continue
                    index = int(ix[0]); key = [run, env, index+1, col]; label = cell['regime']+'/'+category
                    if label not in examples or key < examples[label]['key']:
                        c = next(c for c in ep['chunks'] if c['metadata']['step_start'] <= index+1 < c['metadata']['step_start']+c['metadata']['steps'])
                        examples[label] = {'key': key, 'entity': entity, 'cell_id': cell['cell_id'],
                            'geometry_digest': cell['geometry_digest'], 'terrain_type': ep['type'],
                            'origin_world_m': ep['origin'].tolist(), 'all_five_raw_positions_world_m': ep['positions'][index].tolist(),
                            'nominal_height_world_m': float(h[index, col]) if status[index, col] == 0 else None,
                            'relative_z_m': float(relative[index, col]) if status[index, col] == 0 else None,
                            'status': report['status_codes'][int(status[index, col])], 'manifest_sha256': source['manifest_sha256'],
                            'chunk': c['file'], 'chunk_sha256': c['sha256'], 'row': index+1-c['metadata']['step_start']}
                records_checked += 1
        assert next(stream, None) is None
        assert all(trace[k].shape == (offset, 5) for k in trace)
    assert report['summary']['episodes'] == build['episode_count'] == len(episodes)
    expected_claims = []
    for key, values in totals.items():
        group = report['summary']['groups'][key]
        regime, entity = key.split('/', 1)
        counts = {k: sum(v['counts'][k] for v in values) for k in values[0]['counts']}
        seconds = {k: sum(v['sample_cell_seconds'][k] for v in values) for k in counts}
        expected_group = {'entity_id': entity, 'regime': regime, 'episodes': len(values),
            'known_episodes': sum(v['counts']['known'] > 0 for v in values), 'counts': counts, 'sample_cell_seconds': seconds,
            'episode_equal': {k: distribution([v['statistics']['relative_z_m'][k] for v in values
                                              if v['statistics']['relative_z_m'] is not None]) for k in ('min', 'max', 'mean')},
            'pooled_known_fraction': counts['known']/counts['total'],
            'pooled_known_time_fraction': seconds['known']/seconds['total']}
        close(group, expected_group)
        med = expected_group['episode_equal']['mean']['median']
        text = (f"{regime} / {entity}: nominal terrain height qualified at {counts['known']}/{counts['total']} recorded reference-point samples; "
                + (f"episode-equal median mean vertical difference {med:.6g}m. " if med is not None else 'Vertical difference unknown. ')
                + f"Unknown samples {counts['unknown']}. Reconstructed nominal surface only; runtime surface unknown. "
                  'Not sole clearance, contact distance, or a cause claim.')
        expected_claims.append({'id': identifier('nominal-terrain-claim', report['binding'], config, key, group), 'text': text})
    assert report['optional_statements'] == expected_claims
    base = read(ROOT/'outputs/p2a'/source['source_id']/'public.json')
    preview, receipt = read(directory/'preview.json'), read(directory/'preview_receipt.json')
    assert preview.get('base_report', preview) == base
    encoded = json.dumps(preview, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()
    assert len(encoded) == receipt['combined_bytes'] <= 8192
    included = receipt['included_ids']
    assert preview.get('optional_nominal_terrain_statements', []) == [c for c in expected_claims if c['id'] in included]
    assert set(included) | {x['id'] for x in receipt['omitted']} == {c['id'] for c in expected_claims}
    result = {'status': 'PASS', 'source': source['source_id'], 'episodes': len(episodes), 'records_checked': records_checked,
              'valid_rows': offset, 'reference_point_samples': offset*5, 'maximum_triangle_formula_error_m': maximum_error,
              'independent_extreme_first_index_differences': extreme_index_differences,
              'summary': report['summary'], 'examples': examples, 'all_statements': expected_claims, 'preview_receipt': receipt,
              'absent_categories': sorted({c['regime']+'/'+k for c in historical['cells']
                  for k in ('start', 'tread', 'top', 'unknown', 'negative_difference')}-examples.keys()),
              'wall_seconds': time.perf_counter()-started}
    with (audit_output or directory/'audit.json').open('x', encoding='utf-8', newline='\n') as f: json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k not in ('examples', 'all_statements', 'summary', 'preview_receipt')}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--source', required=True); p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--audit-output', type=Path)
    a = p.parse_args()
    sources = read(Path(__file__).parents[1]/'P2a/source_lock.json')['sources']
    history = read(ROOT/'outputs/p2c3/historical_02/stdout.json')
    assert sha(ROOT/'outputs/p2c3/historical_02/stdout.json') == '84aecf9e412be7578ee567ca65cc1c6ff48351f8f45c1fb789a1a61913e8a48f'
    audit(next(s for s in sources if s['source_id'] == a.source), a.directory,
          next(s for s in history['sources'] if s['source_id'] == a.source), a.audit_output)
