"""CPU audit of r2 using the existing exact numeric comparison and extraction."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image

BASE = Path('/data/yuyangliu/AGENT/feedback_p3new_20260929')
CONTROL = BASE / 'body_grounding_dense_r2'
CAPTURE = BASE / 'push_body_grounding_dev_r2'
SOURCE = BASE / 'source_body_grounding_r1'
PYTHON = '/data/yuyangliu/envs/isaaclab6_validation/bin/python'


def main():
    actual = json.loads((CAPTURE / 'result.json').read_text())
    print(json.dumps({k: actual.get(k) for k in ('status', 'error', 'gpu_capacity_seconds')}), flush=True)
    if actual['status'] != 'PASS':
        return 10 if actual['status'] == 'RUNNING' else 11
    output = CONTROL / 'verification.json'
    if output.exists():
        raise FileExistsError(output)
    card = json.loads((CONTROL / 'capture_card.json').read_text())
    sys.path.insert(0, str(SOURCE / 'src'))
    spec = importlib.util.spec_from_file_location('existing_exact_audit', SOURCE / 'PLAN/feedback_sys/P3NEW/verify_multi_env_capture.py')
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    from true_rl_agent.tasks.push_recovery.visual_observation import align_push_snapshot_samples
    from true_rl_agent.runner.native_video import _camera_params_data

    result = {'schema_version': 'p3new.body_grounding_dense_audit.v1', 'status': 'running',
              'started_at_utc': datetime.now(timezone.utc).isoformat(), 'input_sha256': {}, 'runs': [],
              'scope': 'old_development_interval1_engineering_not_N5', 'provider_requests': 0,
              'script_sha256': audit.sha(__file__), 'camera_image_landing': 'inherits_r1_development_review; no new independent pixel error bound'}
    try:
        evidence = result['input_sha256']
        source_manifest = audit.read(Path(card['source_manifest_path']), evidence)
        audit.require(audit.sha(Path(card['source_manifest_path'])) == card['source_manifest_sha256'], 'source manifest differs')
        audit.require(len(source_manifest['files']) == 401 and all(audit.sha(SOURCE / p) == h for p, h in source_manifest['files'].items()), '401 frozen source files differ')
        left, _ = audit.read_runs(Path(card['baseline_root']), 'push', evidence)
        right, recordings = audit.read_runs(CAPTURE, 'push', evidence)
        audit.require(len(recordings['recordings']) == 3, 'expected one env for three seeds')
        images, sample_count, max_eye_error = set(), 0, 0.
        for run in (1, 2, 3):
            a, b = left[run], right[run]
            audit.require(a['seed'] == b['seed'] and all(a['manifest']['identity'].get(k) == b['manifest']['identity'].get(k) for k in audit.IDENTITY), 'numeric/seed identity changed')
            la, ls, _ = audit.load_arrays(a, run, 'push', evidence)
            ra, rs, _ = audit.load_arrays(b, run, 'push', evidence)
            row = {'run_index': run, 'numeric_comparison': audit.compare_run(a, b, la, ra, ls, rs)}
            result['runs'].append(row)
            audit.require(len(b['records']) == 1 and b['records'][0]['env_index'] == 19, 'wrong selected env')
            record = b['records'][0]
            snapshot_path = CAPTURE / 'video' / record['file']
            snapshot = audit.read(snapshot_path, evidence)
            t = b['outcomes'][19]['episode_steps']
            audit.require(snapshot == record['manifest'] and audit.sha(snapshot_path) == record['sha256'], 'snapshot binding differs')
            audit.require(record['episode_outcome'] == b['outcomes'][19] and snapshot['episode_control_step_count'] == t, 'terminal binding differs')
            audit.require(snapshot['schema_version'] == 'native_rgb_snapshots.v4' and snapshot['selected_control_steps'] == [] and snapshot['snapshot_interval_steps'] == 1 and snapshot['sampling_mode'] == 'interval_from_step_1', 'not registered interval1 v4')
            audit.require([s['control_step'] for s in snapshot['samples']] == list(range(1, t + 1)) and record['includes_terminal_frame'] is True, 'dense own-episode coverage incomplete')
            valid = ra['valid'][:, 19]
            row['numeric_overlap'] = align_push_snapshot_samples(snapshot, control_step=ra['control_step'][valid, 19], pose=ra['pose'][valid, 19], planar_motion=ra['planar_motion'][valid, 19])
            row.update(T=t, sample_count=t, snapshot_sha256=record['sha256'], images=[])
            for sample in snapshot['samples']:
                state = sample['state_before']
                audit.require(state == sample['state_after'] and state['articulation_body_names'] == snapshot['articulation_body_names'], 'same-state/name binding differs')
                audit.require([v['view_id'] for v in sample['views']] == ['v0', 'v1'], 'missing/duplicate view')
                audit.require(sample['views'][0]['camera_params']['cameraViewTransform'] != sample['views'][1]['camera_params']['cameraViewTransform'], 'view matrices are identical')
                sample_count += 1
                for view in sample['views']:
                    params = _camera_params_data(view['camera_params'], (1920, 1080))
                    eye = np.linalg.inv(np.asarray(params['cameraViewTransform']))[3, :3] * params['metersPerSceneUnit']
                    max_eye_error = max(max_eye_error, float(np.max(np.abs(eye - view['camera_eye_w']))))
                    p = snapshot_path.parent / view['file']
                    audit.require(p not in images and audit.sha(p) == view['sha256'], 'PNG hash/identity differs')
                    with Image.open(p) as image:
                        audit.require(image.format == 'PNG' and image.mode == 'RGB' and image.size == (1920, 1080), 'wrong native RGB dimensions')
                        image.verify()
                    images.add(p)
                    row['images'].append({'file': p.name, 'sha256': view['sha256'], 'step': sample['control_step'], 'view': view['view_id']})
        audit.require(images == set((CAPTURE / 'video').glob('*.png')), 'missing/orphan PNG')
        audit.require(max_eye_error < 1e-5, 'actual camera origin differs from recorded tracking position')
        result.update(image_count=len(images), sample_count=sample_count, max_abs_camera_origin_error_m=max_eye_error,
            all_valid_numeric_fields_and_terminals_identical=all(r['numeric_comparison']['identical'] for r in result['runs']),
            all_media_numeric_overlaps_aligned=all(r['numeric_overlap']['status'] == 'overlap_aligned' for r in result['runs']))
        result['status'] = 'pass' if result['all_valid_numeric_fields_and_terminals_identical'] and result['all_media_numeric_overlaps_aligned'] else 'difference_observed'
    except Exception as exc:
        result.update(status='incomplete_or_failed', error_type=type(exc).__name__, error=str(exc))
    result['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({k: result.get(k) for k in ('status', 'image_count', 'sample_count', 'max_abs_camera_origin_error_m', 'error')}), flush=True)
    if result['status'] != 'pass':
        return 1
    export = BASE / 'push_body_grounding_dev_r2_export'
    subprocess.run([PYTHON, str(SOURCE / 'PLAN/feedback_sys/P3NEW/extract_dual_view_evidence.py'),
        '--summary', str(CAPTURE / 'evaluation/public_dev/summary.json'), '--recordings', str(CAPTURE / 'video/recordings.json'),
        '--output', str(export), '--resolved-cell-manifest', str(CAPTURE / 'composition/resolved_cell_manifest.json')],
        env=dict(os.environ, PYTHONPATH=str(SOURCE / 'src')), check=True)
    print(json.dumps({'complete_export': str(export), 'extraction_sha256': audit.sha(export / 'extraction.json')}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
