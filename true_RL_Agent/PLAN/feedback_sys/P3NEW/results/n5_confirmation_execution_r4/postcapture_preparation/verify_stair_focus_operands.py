"""Read-only CPU audit of supplied R2 Stair focus operands from original NPZ.

Uses frozen independent P3R reader and task normative surface query only.
No production event reducer/exporter/packer, model, GPU, or simulator calls.
Run remotely via Python -B - from SSH stdin; JSON receipt is stdout only.
"""
from collections import Counter
from dataclasses import fields
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import traceback

import numpy as np

B = Path('/data/yuyangliu/AGENT/feedback_p3new_20260929')
C = B / 'n5_confirmation_20261001_r4'
S = B / 'n5_confirmation_source_r4_final'
REFS, CHECKS, UNCHECKED = {}, [], []
FOCUS_RESULTS = []
FLOAT_ABS_TOL = 1e-10


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def watch(path):
    path = Path(path).resolve()
    if str(path) not in REFS:
        REFS[str(path)] = {'path': str(path), 'before_sha256': sha(path), 'bytes': path.stat().st_size}
    return REFS[str(path)]


def load(path):
    watch(path)
    return json.loads(Path(path).read_bytes())


def normalize(value):
    return json.loads(json.dumps(value, allow_nan=False))


def compare(path, actual, expected, *, basis='independent_original_npz_arithmetic'):
    actual, expected = normalize(actual), normalize(expected)
    different, leaf_count = [], 0
    def visit(a, e, p):
        nonlocal leaf_count
        if isinstance(e, dict):
            if not isinstance(a, dict) or set(a) != set(e):
                different.append({'path': p, 'actual': a, 'expected': e}); return
            for key in e:
                visit(a[key], e[key], p + '/' + str(key))
        elif isinstance(e, list):
            if not isinstance(a, list) or len(a) != len(e):
                different.append({'path': p, 'actual': a, 'expected': e}); return
            if not e:
                leaf_count += 1
            for i, (av, ev) in enumerate(zip(a, e)):
                visit(av, ev, p + '/' + str(i))
        else:
            leaf_count += 1
            equal = ((type(a) in (float, int) and math.isfinite(a) and abs(a-e) <= FLOAT_ABS_TOL)
                     if type(e) is float else type(a) is type(e) and a == e)
            if not equal:
                different.append({'path': p, 'actual': a, 'expected': e})
    visit(actual, expected, path)
    CHECKS.append({'path': path, 'basis': basis, 'status': 'MISMATCH' if different else 'MATCH',
                   'scalar_or_empty_checks': leaf_count, 'mismatch_count': len(different),
                   'mismatch_examples': different[:12]})


def stats(intervals, required, terminal, dt):
    lengths = [b-a+1 for a, b in intervals]
    qualified = [(a, b) for a, b in intervals if b-a+1 >= required]
    first_window = max(1, terminal-math.ceil(1/dt)+1)
    return [sum(lengths), len(intervals), max(lengths, default=0),
            min((a for a, b in intervals), default=None),
            min((a+required-1 for a, b in qualified), default=None), len(qualified),
            any(a < terminal and b >= first_window for a, b in intervals)]


def run():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CUDA_VISIBLE_DEVICES must be explicitly empty')
    raw_manifest = load(C / 'numeric_raw/manifest.json')
    private = load(C / 'numeric_raw/_private/sources.json')
    packing = load(C / 'numeric_materials/packing_receipt.json')
    full_path = C / 'stair/evaluation/public_dev/behavior_full.json'
    full = load(full_path)
    reader_path = S / 'PLAN/feedback_sys/P3R/confirmation_raw_material.py'
    watch(reader_path)
    compare('source/independent_reader_sha256', sha(reader_path), raw_manifest['source_hashes']['PLAN/feedback_sys/P3R/confirmation_raw_material.py'], basis='frozen_source_binding')
    spec = importlib.util.spec_from_file_location('frozen_independent_stair_reader', reader_path)
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)

    sources = [r for r in private['sources'] if r['source_id'] == 'stair_source']
    manifest_paths = sorted(set(r['manifest_path'] for r in sources))
    if len(manifest_paths) != 1:
        raise ValueError('expected one bound Stair archive')
    manifest_path = Path(manifest_paths[0])
    manifest = load(manifest_path)
    meta = manifest['metadata']
    for source in sources:
        compare(f'source/run_{source["run"]}/manifest_sha256', sha(manifest_path), source['manifest_sha256'], basis='archive_identity_binding')
    for chunk in manifest['chunks']:
        path = manifest_path.parent / chunk['file']
        compare('source/chunk/' + chunk['file'], watch(path)['before_sha256'], chunk['sha256'], basis='archive_bytes_binding')

    # Only normative geometry primitive; never production behavior aggregation.
    sys.path.insert(0, str(S / 'src'))
    from true_rl_agent.tasks.stair_climbing.unified_evaluation import UnifiedStairCellSpec
    from true_rl_agent.tasks.stair_climbing.terrain_evidence import stair_surface_query_for_ordered_cell
    for name in ('terrain_evidence', 'rough_surface', 'ordered_evaluation', 'unified_evaluation'):
        path = Path(sys.modules['true_rl_agent.tasks.stair_climbing.' + name].__file__).resolve()
        relative = path.relative_to(S).as_posix()
        compare('source/normative_geometry/' + name, watch(path)['before_sha256'], raw_manifest['source_hashes'][relative], basis='frozen_normative_definition_binding')
    for module in list(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename and Path(filename).resolve().is_relative_to(S):
            watch(filename)

    rows, curves, _ = reader.stair_rows(manifest_path, full_path)
    by_id = {row['id']: row for row in rows}
    aliases = {row['episode_id']: row['original_id'] for row in private['anonymous_mapping'] if row['task'] == 'stair'}
    if len(by_id) != len(rows) or len(aliases) != len(rows):
        raise ValueError('non-unique or incomplete Stair raw identity mapping')
    packets = [p for p in packing['packets'] if p['axis'] == 'focus' and p['task'] == 'stair']
    records, episode_jobs = {}, {}
    definitions_by_pid = {}
    for packet in packets:
        pid = packet['packet_id']
        path = C / 'numeric_materials/materials/related' / (pid + '.json')
        focus = load(path)
        records[pid] = focus
        definitions_by_pid[pid] = focus['definitions']['measurement_definitions']
        raw_focus_path = C / 'numeric_raw/focus' / ('stair_' + packet['scope'] + '.json')
        watch(raw_focus_path)
        reference = [r for r in raw_manifest['readable_materials'] if r['file'] == 'focus/' + raw_focus_path.name]
        if len(reference) != 1:
            raise ValueError('missing unique focus raw source reference')
        compare(pid + '/raw_focus_source_sha256', sha(raw_focus_path), reference[0]['sha256'], basis='frozen_source_binding')
        if len(focus['scope_rows']) != len(focus['operands_in_same_row_order']):
            raise ValueError('focus row/operand length mismatch')
        expected_ids = sorted(r['id'] for r in rows if r['scope'] == packet['scope'])
        actual_ids = [aliases[dict(zip(focus['scope_row_columns'], scope))['episode_id']] for scope in focus['scope_rows']]
        compare(pid + '/scope_identity_set', sorted(actual_ids), expected_ids, basis='archive_identity_binding')
        compare(pid + '/scope_denominator', focus['scope_denominator'], len(expected_ids), basis='archive_identity_binding')
        definition = definitions_by_pid[pid]
        compare(pid + '/protocol_definition', definition['protocol'], meta['semantics'], basis='frozen_definition_binding')
        compare(pid + '/body_order', definition['bodies']['support_effector'], meta['runtime_projection']['resolved_roles']['support_effector'], basis='frozen_definition_binding')
        query = definition['arithmetic_queries']
        rules = full['semantics']['thresholds']
        compare(pid + '/query_definition', [query['x_window_s'], query['x_window_gain_m'], query['unloaded_speed_mps'], query['unloaded_duration_s']],
                [rules['low_progress_window_s'], rules['low_progress_gain_m'], rules['unloaded_low_speed_mps'], rules['unloaded_low_motion_s']], basis='frozen_definition_binding')
        for index, logical in enumerate(actual_ids):
            episode_jobs.setdefault(logical, []).append((pid, index))
        FOCUS_RESULTS.append({'packet_id': pid, 'scope': packet['scope'], 'rows': len(actual_ids)})

    keys = ['valid', 'episode_length', 'base_pos_w', 'support_pos_w', 'support_lin_vel_w',
            'support_contact_forces_w_history', 'forbidden_contact_forces_w_history',
            'done_mask', 'wrapper_done_mask', 'termination_0', 'termination_1']
    surfaces = {}
    for seed in meta['seeds']:
        run_index, dt = seed['run_index'], seed['step_dt']
        array, static = reader.chunks(manifest_path, manifest, keys, run_index)
        for env, terminal in enumerate(seed['outcomes']):
            logical = f'r{run_index}.e{env}'
            if logical not in episode_jobs:
                continue
            row, curve = by_id[logical], curves[logical]
            valid = array['valid'][:, env]
            steps = array['episode_length'][valid, env]
            T = len(steps)
            if not np.array_equal(steps, np.arange(1, T+1)):
                raise ValueError('non-contiguous valid first episode')
            raw = {key: value[valid, env] for key, value in array.items()}
            origin = static['terrain_env_origins_w'][env].astype(float)
            position = raw['base_pos_w'].astype(float)-origin
            velocity = raw['support_lin_vel_w'].astype(float)
            planar = np.sqrt(np.sum(velocity[..., :2]**2, axis=-1))
            speed3d = np.sqrt(np.sum(velocity**2, axis=-1))
            vz = velocity[..., 2]
            history = np.sqrt(np.sum(raw['support_contact_forces_w_history'].astype(float)**2, axis=-1)).max(axis=1)
            forbidden_history = np.sqrt(np.sum(raw['forbidden_contact_forces_w_history'].astype(float)**2, axis=-1)).max(axis=1)
            for pid, index in episode_jobs[logical]:
                focus = records[pid]
                defs = definitions_by_pid[pid]
                signals, query = defs['protocol']['episode_outcomes']['signals'], defs['arithmetic_queries']
                base = f'{pid}/operands_in_same_row_order/{index}'
                scope = dict(zip(focus['scope_row_columns'], focus['scope_rows'][index]))
                compare(base + '/source_row_binding', [scope['run_group'], scope['cell'], scope['steps'], scope['dt_s']],
                        [f'run_{run_index:02d}', terminal['bin_id'], T, dt], basis='archive_identity_binding')
                geometry = focus['definitions']['cells'][terminal['bin_id']]
                cell_key = json.dumps(geometry, sort_keys=True)
                if cell_key not in surfaces:
                    params = {field.name: geometry[field.name] for field in fields(UnifiedStairCellSpec)}
                    for key in ('riser_heights_m', 'tread_depths_m', 'size_m'):
                        params[key] = tuple(params[key])
                    cell = UnifiedStairCellSpec(**params)
                    compare(pid + '/geometry_digest/' + terminal['bin_id'], cell.geometry_digest, geometry['geometry_digest'], basis='frozen_normative_definition_consistency')
                    surfaces[cell_key] = stair_surface_query_for_ordered_cell(cell)
                surface = surfaces[cell_key]
                relative = raw['support_pos_w'].astype(float)-origin
                heights = np.asarray([[surface.sample(progress_m=float(p[0]), lateral_m=float(p[1])).height_m for p in points]
                                      for points in relative], dtype=float)
                clearance = relative[..., 2]-heights
                loaded = curve['force'] > signals['slip']['contact_force_threshold']
                roll, pitch, up = curve['roll'], curve['pitch'], curve['up_z']
                low_progress = np.zeros(T, bool)
                width = math.ceil(query['x_window_s']/dt)
                for i in range(width, T):
                    low_progress[i] = position[i-width:i+1, 0].max()-position[i-width, 0] < query['x_window_gain_m']
                tilt_threshold = signals['base_orientation_instability']['threshold']
                masks = {
                    'root_x_window_max_minus_start_lt_query_m': low_progress[:, None],
                    'abs_roll_or_pitch_gt_protocol_rad': (np.maximum(abs(roll), abs(pitch)) > tilt_threshold)[:, None],
                    'root_roll_gt_protocol_rad': (roll > tilt_threshold)[:, None],
                    'root_roll_lt_negative_protocol_rad': (roll < -tilt_threshold)[:, None],
                    'root_pitch_gt_protocol_rad': (pitch > tilt_threshold)[:, None],
                    'root_pitch_lt_negative_protocol_rad': (pitch < -tilt_threshold)[:, None],
                    'root_up_axis_world_z_lt_zero': (up < 0)[:, None],
                    'support_force_gt_protocol_n': loaded,
                    'all_support_force_le_protocol_n': (~loaded.any(axis=1))[:, None],
                    'loaded_planar_speed_gt_protocol_mps': loaded & (planar > signals['slip']['speed_threshold']),
                    'unloaded_speed3d_le_query_mps': ~loaded & (speed3d <= query['unloaded_speed_mps']),
                    'unloaded_vz_gt_query_mps': ~loaded & (vz > query['signed_vz_mps']),
                    'unloaded_vz_lt_negative_query_mps': ~loaded & (vz < -query['signed_vz_mps']),
                    'unloaded_clearance_lt_protocol_m': ~loaded & (clearance < signals['insufficient_clearance']['threshold']),
                    'support_buffer_force_gt_protocol_n': history > signals['slip']['contact_force_threshold'],
                    'forbidden_buffer_force_gt_protocol_n': forbidden_history > signals['forbidden_contact']['force_threshold_n'],
                }
                requirements = {name: 1 for name in masks}
                requirements.update(abs_roll_or_pitch_gt_protocol_rad=signals['base_orientation_instability']['consecutive_control_steps'],
                    loaded_planar_speed_gt_protocol_mps=signals['slip']['consecutive_control_steps'],
                    unloaded_clearance_lt_protocol_m=signals['insufficient_clearance']['consecutive_control_steps'],
                    unloaded_speed3d_le_query_mps=math.ceil(query['unloaded_duration_s']/dt),
                    unloaded_vz_gt_query_mps=3, unloaded_vz_lt_negative_query_mps=3)
                compare(base + '/predicate_order', focus['predicate_order'], [[name, value.shape[1], requirements[name]] for name, value in masks.items()])
                operands = focus['operands_in_same_row_order'][index]
                cursor = 0
                for name, mask in masks.items():
                    for component in range(mask.shape[1]):
                        expected = stats(reader.spans(mask[:, component], steps), requirements[name], T, dt)
                        basis = 'normative_geometry_consistency_not_collision_physics' if name == 'unloaded_clearance_lt_protocol_m' else 'independent_original_npz_arithmetic'
                        compare(base + '/0/' + str(cursor), operands[0][cursor], expected, basis=basis)
                        cursor += 1
                compare(base + '/predicate_component_count', len(operands[0]), cursor)
                or_names = ['loaded_planar_speed_gt_protocol_mps', 'unloaded_clearance_lt_protocol_m']
                compare(base + '/any_body_or_order', focus['any_body_or_order'], or_names)
                for j, name in enumerate(or_names):
                    expected = stats(reader.spans(masks[name].any(axis=1), steps), requirements[name], T, dt)
                    compare(base + '/1/' + str(j), operands[1][j], expected,
                            basis='normative_geometry_consistency_not_collision_physics' if j else 'independent_original_npz_arithmetic')
                root_extrema = []
                for values in (position[:, 0], roll, pitch, up):
                    ex = reader.extrema(values, steps)
                    root_extrema.append([ex['min'], ex['min_step'], ex['max'], ex['max_step']])
                compare(base + '/2', operands[2], root_extrema)
                for j in range(len(defs['bodies']['support_effector'])):
                    low, fast = int(np.argmin(clearance[:, j])), int(np.argmax(planar[:, j]))
                    minimum, maximum = int(np.argmin(vz[:, j])), int(np.argmax(vz[:, j]))
                    # Equal absolute magnitude chooses positive max, then its earliest occurrence.
                    signed = minimum if abs(vz[minimum, j]) > abs(vz[maximum, j]) else maximum
                    compare(base + f'/3/{j}/0', operands[3][j][0], [float(clearance[low, j]), int(steps[low])], basis='normative_geometry_consistency_not_collision_physics')
                    compare(base + f'/3/{j}/1', operands[3][j][1], [float(planar[fast, j]), int(steps[fast])])
                    compare(base + f'/3/{j}/2', operands[3][j][2], [float(vz[signed, j]), int(steps[signed])])
                x = raw['base_pos_w'][:, 0].astype(float)
                drops = np.maximum.accumulate(x)-x
                trough = int(np.argmax(drops))
                peak = int(np.argmax(x[:trough+1])) if drops[trough] > 0 else 0
                witness = {'start_step': int(steps[peak]), 'end_step': int(steps[trough]),
                           'start_x_w_m': float(x[peak]), 'end_x_w_m': float(x[trough]), 'drawdown_m': float(drops[trough])}
                compare(base + '/4', operands[4], witness)
                compare(base + '/5', operands[5], [position[0].tolist(), position[-1].tolist()])
                termination = {'done_mask': reader.spans(raw['done_mask'], steps),
                               'wrapper_done_mask': reader.spans(raw['wrapper_done_mask'], steps)}
                for i, name in enumerate(defs['termination_term_order']):
                    termination[name] = reader.spans(raw['termination_' + str(i)], steps)
                compare(base + '/6', operands[6], termination, basis='recorded_termination_flags_not_physical_reclassification')
                # This run has every needed field: absence list is checked, no general missing-data inference.
                compare(base + '/7', operands[7], [], basis='all_required_source_fields_present')
        del array

    forbidden_modules = [name for name in sys.modules if any(token in name for token in
        ('behavior_evidence', 'behavior_report', 'export_confirmation_raw', 'prepare_confirmation_materials'))]
    compare('execution/production_reducer_exporter_packer_imports', forbidden_modules, [], basis='execution_boundary')
    compare('execution/gpu_library_imports', [name for name in ('torch', 'isaaclab', 'omni') if name in sys.modules], [], basis='execution_boundary')
    UNCHECKED.extend([
        {'scope': 'normative_surface', 'reason': 'Frozen task surface primitive and supplied frozen cell definitions reused. This verifies normative geometry arithmetic, not an independent collision mesh or physical contact surface implementation.'},
        {'scope': 'definitions_and_assets', 'reason': 'Protocol/body order and source hashes bound; semantic correctness of protocol, query selection, body binding against simulator assets, and cell geometry against physical meshes are not established.'},
        {'scope': 'original_terminal_meaning', 'reason': 'Recorded termination flags are checked as arrays; physical success/failure classification and priority are not independently reclassified.'},
        {'scope': 'semantic_visual_authoring', 'reason': 'No original model answers, observer outputs, necessary-fact judgments, visuals, report prose, or author delivery read or scored.'},
        {'scope': 'outside_six_supplied_focus_scopes', 'reason': '648 supplied focus rows audited; this is not a claim that all 864 raw-pool episodes have supplied packed focus rows.'},
    ])


error = None
try:
    run()
except Exception as exc:
    error = {'type': type(exc).__name__, 'detail': str(exc), 'traceback': traceback.format_exc()}
    UNCHECKED.append({'scope': 'remaining_requested_audit', 'reason': 'Execution stopped before completion; see input_error.'})
for path, ref in REFS.items():
    try:
        ref['after_sha256'] = sha(path)
        ref['unchanged'] = ref['before_sha256'] == ref['after_sha256']
    except Exception as exc:
        ref['unchanged'] = False
        ref['after_error'] = str(exc)
unchanged = all(ref['unchanged'] for ref in REFS.values())
failed = [check for check in CHECKS if check['status'] == 'MISMATCH']
status = 'INCOMPLETE' if error else 'MISMATCH_SUPPLIED_OPERANDS' if failed or not unchanged else 'PASS_SUPPLIED_PACKED_OPERANDS_WITH_EXPLICIT_LIMITS'
print(json.dumps({'schema_version': 'p3new.stair_focus_operand_audit.v1', 'at_utc': datetime.now(timezone.utc).isoformat(),
    'status': status, 'input_error': error, 'focus_units': FOCUS_RESULTS, 'checks': CHECKS, 'failed_checks': failed,
    'check_count': len(CHECKS), 'scalar_or_empty_check_count': sum(c['scalar_or_empty_checks'] for c in CHECKS),
    'mismatch_count': sum(c['mismatch_count'] for c in CHECKS), 'basis_check_counts': dict(Counter(c['basis'] for c in CHECKS)),
    'input_refs': list(REFS.values()), 'inputs_unchanged': unchanged, 'unchecked': UNCHECKED,
    'model_calls': 0, 'simulation_calls': 0, 'remote_source_writes': 0,
    'execution_environment': {key: os.environ.get(key) for key in ('CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS')},
    'comparison_policy': {'integer_boolean_null': 'exact type and value', 'float_absolute_tolerance': FLOAT_ABS_TOL,
                          'interval_endpoints': 'inclusive control steps; confirm at start+required-1',
                          'final_window': 'overlap with final ceil(1/dt) samples and run starts strictly before T',
                          'extrema_ties': 'earliest occurrence; signed-vz abs tie chooses max then earliest'}}, ensure_ascii=False, indent=2))
sys.exit(0 if status.startswith('PASS_') else 1)
