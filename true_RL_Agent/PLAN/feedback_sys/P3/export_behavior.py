"""Recompute P3 views from immutable P2/P3 public-dev runs; never edit receipts."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    data = (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    Path(path).write_bytes(data)
    return {'file': Path(path).name, 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--only', action='append', default=[])
    args = parser.parse_args()
    sys.path.insert(0, str(args.project_root / 'src'))
    from true_rl_agent.evaluation.binding import canonical_json_hash
    from true_rl_agent.evaluation.public_projection import assert_public_projection_safe
    from true_rl_agent.runner.source_snapshot import project_source_snapshot
    from true_rl_agent.composition.cell_manifest import load_resolved_cell_manifest
    from true_rl_agent.tasks.stair_climbing.behavior import build_stair_behavior, project_stair_behavior, compare_stair_behavior
    from true_rl_agent.tasks.push_recovery.behavior import build_push_behavior, project_push_behavior, push_behavior_context

    args.output.mkdir(parents=True, exist_ok=True)
    reports, views = {}, {}
    for name, item in read(args.inventory).items():
        if args.only and name not in args.only:
            continue
        target = args.output / name
        target.mkdir(exist_ok=False)
        started = time.time()
        root = Path(item['run_root'])
        phase_path = root / 'public_dev_result.json'
        phase = read(phase_path)
        if phase['phase'] != 'public_dev':
            raise ValueError('not public dev')
        if canonical_json_hash(phase['task_evidence']) != phase['task_evidence_hash']:
            raise ValueError('phase evidence content hash mismatch')
        if phase['checkpoint_hash'] != item['expected_checkpoint_hash']:
            raise ValueError('checkpoint differs from the frozen external identity: ' + name)
        source_hash = canonical_json_hash(project_source_snapshot(Path(item['capture_source_root'])))
        if source_hash != item['expected_capture_source_hash']:
            raise ValueError('capture source differs from the frozen source inventory: ' + name)
        if item['task'] == 'stair':
            reference = phase['task_evidence']['numeric_evidence']
            full = build_stair_behavior(reference, expected_identity={
                key: phase[key] for key in ('checkpoint_hash', 'evaluation_plan_hash', 'program_source_checksum')})
            view = project_stair_behavior(full)
            rows = full['episodes']
            counts = dict(Counter(row['primary_reason'] for row in rows))
        else:
            manifest = load_resolved_cell_manifest((root / 'composition/resolved_cell_manifest.json').resolve())
            context = push_behavior_context(manifest, evaluation_plan_hash=phase['evaluation_plan_hash'],
                checkpoint_hash=phase['checkpoint_hash'], program_source_checksum=phase['program_source_checksum'],
                runtime_dynamics_hash=phase['environment_source_preflight']['runtime_dynamics_hash'],
                execution_source_hash=source_hash)
            full = build_push_behavior(phase['task_evidence']['seed_summaries'], context=context)
            view = project_push_behavior(full)
            rows = full['transcripts']
            counts = dict(Counter(row['terminal_label'] for row in rows))
        if len(rows) != item['expected_episodes']:
            raise ValueError('episode count differs from frozen inventory: ' + name)
        for label, expected in item.get('expected_outcomes', {}).items():
            if counts.get(label, 0) != expected:
                raise ValueError('outcome differs from independent P2 acceptance: ' + name + ':' + label)
        assert_public_projection_safe(view)
        if view != (project_stair_behavior(view) if item['task'] == 'stair' else project_push_behavior(view)):
            raise ValueError('public projection is not idempotent')
        reports[name] = {'status': 'PASS', 'task': item['task'], 'episodes': len(rows), 'outcomes': counts,
            'phase_file_sha256': hashlib.sha256(phase_path.read_bytes()).hexdigest(),
            'phase_result_hash': canonical_json_hash(phase), 'checkpoint_hash': phase['checkpoint_hash'],
            'capture_execution_source_hash': source_hash,
            'capture_source_binding': 'independently frozen capture source inventory; not inferred from a receipt path',
            'derived_at_unix': started,
            'full': write(target / 'full.json', full), 'public': write(target / 'public.json', view),
            'elapsed_seconds': time.time()-started}
        write(target / 'provenance.json', reports[name])
        views[name] = view
        print(json.dumps({'dataset': name, **reports[name]}), flush=True)
    if {'stair_parent_v1', 'stair_gen8_v1'} <= views.keys():
        pair = compare_stair_behavior(views['stair_parent_v1'], views['stair_gen8_v1'],
                                     expected_parent_checkpoint_hash=reports['stair_parent_v1']['checkpoint_hash'])
        if pair['status'] != 'available':
            raise ValueError('known parent/candidate common domain did not compare')
        write(args.output / 'stair_parent_gen8_diff.json', pair)
    write(args.output / 'inventory.json', reports)


if __name__ == '__main__':
    main()
