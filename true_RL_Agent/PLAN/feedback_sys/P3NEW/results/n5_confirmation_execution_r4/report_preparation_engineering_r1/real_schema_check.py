"""One CPU integration against disclosed R4 originals; select materials only."""
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'current'))
import prepare_confirmation_reports as control
runner = control.runner
CAP = Path('/data/yuyangliu/AGENT/feedback_p3new_20260929/n5_confirmation_20261001_r4')
OUT = HERE / 'development_materials'


def read(path): return json.loads(path.read_bytes())


def originals(manifest):
    result = {}
    for item in read(manifest)['materials']:
        path, raw = runner.referenced(manifest.parent, item)
        result[item['id']] = {**item, 'data': json.loads(raw) if item['mime_type'] == 'application/json' else None}
    return result


def main():
    assert not OUT.exists(), 'Preserve previous development integration; no overwrite'
    OUT.mkdir()
    refs = {}
    numeric_path = CAP / 'numeric_materials/packing_receipt.json'
    media_path = CAP / 'media_materials/confirmation_media_receipt.json'
    numeric, media = read(numeric_path), read(media_path)
    raw_root = Path(numeric['source']['raw_export'])
    tasks = {}
    for task in ('stair', 'push'):
        folder = CAP / 'delivery_offline_r1' / task
        task_refs = {key: runner.ref(path) for key, path in {
            'full': folder / 'behavior_full.json', 'phase': folder / 'public_dev_result.json',
            'request': folder / 'request.http_body.json', 'raw_manifest': raw_root / 'manifest.json',
            'raw_sources': raw_root / '_private/sources.json'}.items()}
        task_refs['request_status'] = 'offline_serialized_not_sent'
        refs[task] = task_refs
        tasks[task] = control.task_materials(task, task_refs, OUT / 'shared' / task)
    units = [{k: row[k] for k in ('packet_id', 'task', 'axis', 'scope')} |
             {'facts_manifest': runner.ref(CAP / 'numeric_materials' / row['facts_manifest'])} for row in numeric['packets']]
    for row in media['units']:
        origin = CAP / 'observers' / row['task'] / row['alias'] / 'sol_response'
        units.append({'packet_id': row['unit_id'], 'task': row['task'], 'axis': 'media', 'scope': row['alias'],
            'alias': row['alias'], 'original_id': f"r{row['source_run']}.e{row['source_env']}",
            'observer_output': runner.ref(origin / 'raw_answer.txt'),
            'observer_audit': runner.ref(origin / 'mechanical_audit.json'), 'facts_manifest': row['facts_manifest']})
    index = {'numeric_packing': runner.ref(numeric_path), 'media_packing': runner.ref(media_path), 'tasks': refs, 'units': units}
    control.verify_source_index(index)
    rows = []
    for unit in units:
        inputs = originals(Path(unit['facts_manifest']['file'])) if unit['axis'] == 'media' else None
        materials = control.select_materials(unit, tasks[unit['task']], OUT / unit['packet_id'], inputs)
        envelopes = {m['id']: {**m, 'data': read(Path(m['file']))} for m in materials}
        inventory = runner.coverage.build_inventory(envelopes)
        row = {'packet_id': unit['packet_id'], 'task': unit['task'], 'axis': unit['axis'], 'materials': materials,
               'inventory_items': len(inventory['items'])}
        if unit['axis'] == 'media':
            mapping = read(OUT / unit['packet_id'] / 'observer_inventory_mapping.json')
            row.update(original_observations=len(mapping['observations']), not_visible=len(mapping['not_visible']),
                       mapping=runner.ref(OUT / unit['packet_id'] / 'observer_inventory_mapping.json'))
        rows.append(row)
    for task in refs.values():
        for reference in task.values():
            if isinstance(reference, dict): control.checked(reference)
    private = read(raw_root / '_private/sources.json')
    receipt = {'status': 'DEVELOPMENT_REAL_SCHEMA_PASS_NOT_N5', 'finished_utc': datetime.now(timezone.utc).isoformat(),
        'inputs': index, 'rows': rows, 'unit_count': len(rows),
        'source_counts': {task: {'rows': sum(s['source_id'].startswith(task) for s in private['sources']),
                               'unique_manifest_shas': sorted({s['manifest_sha256'] for s in private['sources'] if s['source_id'].startswith(task)})}
                          for task in ('stair', 'push')},
        'current_control_sources': [runner.ref(HERE / 'current' / name) for name in (
            'prepare_confirmation_reports.py', 'prepare_facts_offline.py', 'prepare_confirmation_materials.py',
            'run_confirmation_review.py', 'confirmation_review_coverage.py')],
        'scope': 'Real disclosed R4 schema/source/axis integration only; no semantic acceptance or future cohort qualification',
        'models': 0, 'api_calls': 0, 'gpu_calls': 0, 'facts_seals': 0, 'report_manifests': 0, 'report_requests': 0}
    runner.save(OUT / 'verification.json', receipt, exclusive=True)
    print(json.dumps({'status': receipt['status'], 'units': len(rows), 'receipt': runner.ref(OUT / 'verification.json'),
                      'source_counts': receipt['source_counts'], 'media': [{k: r[k] for k in ('packet_id','original_observations','not_visible','inventory_items')} for r in rows if r['axis']=='media']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    OUT = args.output.resolve()
    main()
