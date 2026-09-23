"""Apply the predeclared G diagnostic video selection; no physical execution."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
P6 = ROOT / 'PLAN/2new_world/P6'
DEST = P6 / 'force_strength_01'


def read(path):
    return json.loads(path.read_text())


def cases(directory):
    found = []
    for index in (1, 2, 3):
        meta = read(directory / f'seed_{index:02d}.json')
        with np.load(directory / f'seed_{index:02d}.npz') as a:
            for row in meta['episodes']:
                env = row['env_id']; mask = a['valid'][:, env]
                x, y, z, w = a['quaternion_xyzw'][mask, env].T
                roll = np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y))
                pitch = np.arcsin(np.clip(2*(w*y-z*x), -1, 1))
                tilt = np.maximum(np.abs(roll), np.abs(pitch))
                delay = max((s-e['end_control_step'] for s, e in zip(row['event_recovery_control_steps'], row['force_events']) if s is not None), default=-1)
                found.append({'seed_index': index, 'seed': meta['seed'], 'env_index': env,
                    'max_abs_tilt_rad': float(tilt.max()), 'worst_tilt_step': int(a['steps'][mask,env][tilt.argmax()]),
                    'recovery_delay_steps': delay, 'expected_episode': row,
                    'source_raw_sha256': meta['raw_sha256'], 'source_dir': str(directory)})
    return found


def ident(c):
    return c['seed_index'], c['env_index']


def select(rows, reason):
    successes = [r for r in rows if r['expected_episode']['success']]
    if reason == 'slowest_success':
        return min(successes, key=lambda r: (-r['recovery_delay_steps'], *ident(r))) if successes else None
    if reason == 'largest_tilt_success':
        return min(successes, key=lambda r: (-r['max_abs_tilt_rad'], *ident(r))) if successes else None
    physical = [r for r in rows if r['expected_episode']['failure_label'] in ('fall_forward','fall_backward','fall_side','height_collapse')]
    failures = physical or [r for r in rows if not r['expected_episode']['success']]
    return min(failures, key=lambda r: (r['expected_episode']['episode_steps'], *ident(r))) if failures else None


selected = []
for robot, scale, source, reasons in (
    ('go2',1.,P6/'force_01/go2',('slowest_success',)),
    ('go2',1.5,DEST/'scale_1p5',('slowest_success','largest_tilt_success','earliest_failure')),
    ('go2',2.,DEST/'scale_2p0',('slowest_success','largest_tilt_success','earliest_failure')),
    ('g1',1.,P6/'force_01/g1',('earliest_failure',)),
):
    rows = cases(source); used = set()
    for reason in reasons:
        item = select([r for r in rows if ident(r) not in used], reason)
        if item is None:
            continue
        used.add(ident(item))
        selected.append({**item, 'robot': robot, 'force_scale': scale, 'selection_reason': reason,
            'case_id': f'{robot}_{str(scale).replace(".","p")}_s{item["seed_index"]}_e{item["env_index"]}'})
assert len(selected) <= 8
record = {'selected_at_unix':time.time(), 'maximum_cases':8, 'replay_not_new_statistical_samples':True,
    'plan_sha256':hashlib.sha256((P6/'FORCE_STRENGTH_PLAN_ZH.md').read_bytes()).hexdigest(),
    'selector_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'cases':selected}
path = DEST/'records/video_selection.json'
with path.open('x') as f:
    json.dump(record,f,indent=2)
for r in selected:
    print(r['case_id'],r['selection_reason'],r['expected_episode']['failure_label'],r['recovery_delay_steps'],r['max_abs_tilt_rad'])
