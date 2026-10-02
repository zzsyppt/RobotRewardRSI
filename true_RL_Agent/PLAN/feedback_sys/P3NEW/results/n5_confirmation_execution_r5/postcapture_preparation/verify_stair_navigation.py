"""Independent Stair navigation counts from supplied original inclusive intervals.

Supplement to the existing NPZ/source/operand audits. No producer, packing,
report reducer, model or GPU is imported; this does not certify semantic facts.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


def require(ok, message):
    if not ok:
        raise ValueError(message)


def ref(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'size_bytes': path.stat().st_size}


def read(path): return json.loads(Path(path).read_bytes())


def audit_navigation(raw, related):
    """Enumerate expected components/groups from source, never reported rows."""
    require(raw['task'] == related['task'] == 'stair' and raw['focus'] == related['focus'], 'Focus identity differs')
    definitions = raw['task_definitions']['measurement_definitions']
    require(related['definitions']['measurement_definitions'] == definitions, 'Original definitions differ')
    bodies = definitions['bodies']
    signals = definitions['protocol']['episode_outcomes']['signals']
    query = definitions['arithmetic_queries']
    tilt, slip, clearance = (signals[k] for k in ('base_orientation_instability', 'slip', 'insufficient_clearance'))
    # Fixed exporter query register. Threshold/body authority is the original
    # task definitions, not navigation labels, summary counts or producer code.
    registered = [
        ('root_x_window_max_minus_start_lt_query_m', 'base', f"inclusive {query['x_window_s']}s root-x window max minus start < {query['x_window_gain_m']} m; undefined until full window"),
        ('abs_roll_or_pitch_gt_protocol_rad', 'base', f"max(abs(roll),abs(pitch)) > {tilt['threshold']} rad"),
        ('root_roll_gt_protocol_rad', 'base', f"signed roll > {tilt['threshold']} rad"),
        ('root_roll_lt_negative_protocol_rad', 'base', f"signed roll < {-tilt['threshold']} rad"),
        ('root_pitch_gt_protocol_rad', 'base', f"signed pitch > {tilt['threshold']} rad"),
        ('root_pitch_lt_negative_protocol_rad', 'base', f"signed pitch < {-tilt['threshold']} rad"),
        ('root_up_axis_world_z_lt_zero', 'base', 'root up-axis world-z < 0 (unitless)'),
        ('support_force_gt_protocol_n', 'support_effector', f"instant force norm > {slip['contact_force_threshold']} N"),
        ('all_support_force_le_protocol_n', 'all_support', f"all support instant force norms <= {slip['contact_force_threshold']} N"),
        ('loaded_planar_speed_gt_protocol_mps', 'support_effector', f"same-body instant force norm > {slip['contact_force_threshold']} N AND planar speed > {slip['speed_threshold']} m/s"),
        ('unloaded_speed3d_le_query_mps', 'support_effector', f"same-body instant force norm <= {slip['contact_force_threshold']} N AND speed3d <= {query['unloaded_speed_mps']} m/s"),
        ('unloaded_vz_gt_query_mps', 'support_effector', f"same-body instant force norm <= {slip['contact_force_threshold']} N AND signed world-z velocity > {query['signed_vz_mps']} m/s"),
        ('unloaded_vz_lt_negative_query_mps', 'support_effector', f"same-body instant force norm <= {slip['contact_force_threshold']} N AND signed world-z velocity < {-query['signed_vz_mps']} m/s"),
        ('unloaded_clearance_lt_protocol_m', 'support_effector', f"same-body instant force norm <= {slip['contact_force_threshold']} N AND reference-point clearance < {clearance['threshold']} m"),
        ('support_buffer_force_gt_protocol_n', 'support_effector', f"history-buffer maximum force norm > {slip['contact_force_threshold']} N"),
        ('forbidden_buffer_force_gt_protocol_n', 'forbidden_contact', f"history-buffer maximum force norm > {signals['forbidden_contact']['force_threshold_n']} N"),
    ]
    members = raw['all_scope_rows']
    require(len(members) == raw['scope_denominator'] == related['scope_denominator'] and bool(members), 'Scope denominator differs')
    require(len({r['episode_id'] for r in members}) == len(members), 'Duplicate original member')
    columns = ['episode_id', 'run_group', 'cell', 'steps', 'dt_s', 'original_success', 'original_outcome', 'missing_fields']
    require(related['scope_row_columns'] == columns and related['scope_rows'] == [[r[k] for k in columns] for r in members], 'Original membership/T/outcome/missing fields differ')
    layout = None
    for member in members:
        spans = member['direct_measurements']['threshold_intervals_by_component']
        require(set(spans) <= {r[0] for r in registered}, 'Unknown original predicate')
        dwell = {name: 1 for name in spans}
        dwell.update(abs_roll_or_pitch_gt_protocol_rad=tilt['consecutive_control_steps'],
            loaded_planar_speed_gt_protocol_mps=slip['consecutive_control_steps'],
            unloaded_clearance_lt_protocol_m=clearance['consecutive_control_steps'],
            unloaded_speed3d_le_query_mps=math.ceil(query['unloaded_duration_s'] / member['dt_s']),
            unloaded_vz_gt_query_mps=3, unloaded_vz_lt_negative_query_mps=3)
        current = [[name, len(values), dwell[name]] for name, values in spans.items()]
        if layout is None: layout = current
        require(current == layout, 'Original predicate layout/dwell varies across members')
    require(related['predicate_order'] == layout, 'Predicate order/width/dwell differs from original intervals')
    offsets, cursor = {}, 0
    for name, width, required in layout:
        offsets[name] = (cursor, width, required); cursor += width
    components = []
    for name, role, expression in registered:
        names = bodies['support_effector'] if role == 'all_support' else bodies[role]
        identities = [names] if role == 'all_support' else [[name] for name in names]
        require(role != 'base' or len(identities) == 1, 'Original base identity is not singular')
        supplied = offsets.get(name)
        require(supplied is None or supplied[1] == len(identities), 'Original width/body identity differs')
        for component, identity in enumerate(identities):
            components.append([name, component, role, identity, None if supplied is None else supplied[2],
                               None if supplied is None else supplied[0] + component, expression])
    nav = related['arithmetic_navigation']
    require(nav['predicate_component_columns'] == ['predicate', 'component_index', 'body_role', 'body_names',
            'required_consecutive_steps', 'original_flat_stats_index', 'strict_arithmetic_query'], 'Component columns differ')
    require(nav['predicate_components'] == components, 'Complete registered component/body/dwell/offset/query table differs')
    expected_groups = {('all_scope', None, None, None, None): members}
    for member in members:
        for key in [('all_scope_by_original_outcome', None, None, member['original_success'], member['original_outcome']),
                    ('run_cell_by_original_outcome', member['run_group'], member['cell'], member['original_success'], member['original_outcome'])]:
            expected_groups.setdefault(key, []).append(member)
    require(nav['group_columns'] == ['scope', 'run_group', 'cell', 'original_success', 'original_outcome',
            'member_count', 'counts_in_predicate_component_order'], 'Group columns differ')
    require(nav['count_columns'] == ['summary_supplied_members', 'summary_not_supplied_members',
            'confirmed_before_T_members', 'confirmed_at_T_members', 'no_confirmation_in_supplied_summary_members'], 'Count columns differ')
    # Enumerate the expected group set first: missing, duplicate, invented or
    # relabelled groups cannot disappear merely because the output omitted them.
    actual = nav['groups']
    require(len(actual) == len(expected_groups) and all(len(row) == 7 for row in actual), 'Complete group count/shape differs')
    require([tuple(row[:5]) for row in actual] == list(expected_groups), 'Complete group identities/order differ')
    comparisons, member_component_evaluations = 0, 0
    partition_totals = [0, 0, 0, 0, 0]
    for row, (key, group) in zip(actual, expected_groups.items()):
        require(row[5] == len(group) and len(row[6]) == len(components), 'Group denominator/component count differs: ' + str(key))
        for column, (predicate, component, _, _, required, _, _) in enumerate(components):
            expected = [0, 0, 0, 0, 0]
            for member in group:
                values = member['direct_measurements']['threshold_intervals_by_component'].get(predicate)
                intervals = None if values is None else values[component]
                member_component_evaluations += 1
                if intervals is None:
                    expected[1] += 1
                    continue
                require(required is not None and required > 0, 'Supplied interval lacks positive dwell')
                expected[0] += 1
                confirmation = None
                for start, end in intervals:
                    require(1 <= start <= end <= member['steps'], 'Interval outside original T')
                    if end - start + 1 >= required:
                        candidate = start + required - 1
                        if confirmation is None or candidate < confirmation: confirmation = candidate
                if confirmation is None: expected[4] += 1
                elif confirmation < member['steps']: expected[2] += 1
                elif confirmation == member['steps']: expected[3] += 1
                else: raise ValueError('Confirmation after original terminal')
            require(row[6][column] == expected, 'Navigation count differs at ' + str(key) + '/' + predicate + '/' + str(component))
            require(sum(expected[2:]) == expected[0] and expected[0] + expected[1] == len(group), 'Independent partition invariant failed')
            comparisons += 5
            if key[0] == 'all_scope': partition_totals = [a + b for a, b in zip(partition_totals, expected)]
    return {'focus': raw['focus'], 'members': len(members), 'registered_predicates': len(registered),
            'components': len(components), 'complete_expected_groups': len(expected_groups),
            'count_values_compared': comparisons, 'member_component_interval_evaluations': member_component_evaluations,
            'all_scope_partition_totals': dict(zip(nav['count_columns'], partition_totals)),
            'all_expected_groups_components_counts_equal': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--card-sha256', required=True)
    args = parser.parse_args()
    # Binding-only existing control; its imports do not derive diagnostic values.
    from postcapture_cpu import check_freeze, C, CAP
    check_freeze(args.card_sha256)
    prior = read(C / 'postcapture/audit/receipt.json')
    require(prior['status'] == 'COMPLETED_CPU_ONLY_NOT_RELEASED' and prior['capture_card']['sha256'] == args.card_sha256,
            'Run the existing source/NPZ/operand audit first')
    output = C / 'control/stair_navigation'
    require(not output.exists(), 'Preserve previous output; no automatic retry')
    output.mkdir(parents=True)
    watched = [ref(C / 'postcapture/audit/receipt.json')]
    receipt = {'status': 'IN_PROGRESS', 'started_utc': datetime.now(timezone.utc).isoformat(), 'script': ref(__file__),
               'scope': 'R5 independent navigation arithmetic only', 'capture_card_sha256': args.card_sha256,
               'rows': [], 'inputs': watched, 'model_api_gpu_calls': 0, 'semantic_pass_claimed': False}
    def load(path):
        watched.append(ref(path)); return read(path)
    try:
        raw_root, packed = CAP / 'numeric_raw', CAP / 'numeric_materials'
        raw_manifest, packing = load(raw_root / 'manifest.json'), load(packed / 'packing_receipt.json')
        originals = {r['file']: r for r in raw_manifest['readable_materials'] if r['file'].startswith('focus/stair_')}
        packets = [r for r in packing['packets'] if r['task'] == 'stair' and r['axis'] == 'focus']
        require(len(packets) == len(originals) == 6 and len({p['packet_id'] for p in packets}) == 6,
                'Require all six unique Stair focus packets')
        require({'focus/stair_' + p['scope'] + '.json' for p in packets} == set(originals), 'Complete raw focus set differs')
        for packet in packets:
            source_name = 'focus/stair_' + packet['scope'] + '.json'
            raw_path, related_path = raw_root / source_name, packed / 'materials/related' / (packet['packet_id'] + '.json')
            require(ref(raw_path)['sha256'] == originals[source_name]['sha256'], 'Raw focus SHA differs')
            manifest_path = packed / packet['facts_manifest']
            facts = load(manifest_path)
            related_ref, = [r for r in facts['materials'] if r['id'] == 'related-raw']
            require((manifest_path.parent / related_ref['file']).resolve() == related_path.resolve()
                    and ref(related_path)['sha256'] == related_ref['sha256'], 'Actual packed related material binding differs')
            result = audit_navigation(load(raw_path), load(related_path))
            receipt['rows'].append({'packet_id': packet['packet_id'], **result})
        for original in watched: require(ref(original['path']) == original, 'Input changed during audit')
        receipt['status'] = 'PASS_INDEPENDENT_NAVIGATION_ARITHMETIC_NOT_SEMANTIC'
    except Exception as error:
        receipt.update(status='ERROR', error_type=type(error).__name__, error=str(error))
    receipt['finished_utc'] = datetime.now(timezone.utc).isoformat()
    with (output / 'receipt.json').open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2); stream.write('\n')
    print(json.dumps({'status': receipt['status'], 'receipt': ref(output / 'receipt.json')}))
    return 0 if receipt['status'].startswith('PASS_') else 1


if __name__ == '__main__': raise SystemExit(main())
