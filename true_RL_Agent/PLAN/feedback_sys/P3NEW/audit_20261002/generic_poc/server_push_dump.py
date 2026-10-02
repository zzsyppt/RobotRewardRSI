# Read-only audit probe: dump per-step columns of the sealed R4 Push archive as NPZ on stdout.
import sys, json, io
SRC = '/data/yuyangliu/AGENT/feedback_p3new_20260929/n5_confirmation_source_r4_final/src'
sys.path.insert(0, SRC)
import numpy as np
from true_rl_agent.runner.numeric_evidence import load_numeric_evidence_manifest, iter_numeric_evidence
refs = json.load(open('/data/yuyangliu/AGENT/feedback_p3new_20260929/n5_confirmation_20261001_r4/push/numeric_references.json'))
rows, eps = [], []
for ri, ref in enumerate(refs, 1):
    man = load_numeric_evidence_manifest(ref); md = man['metadata']
    recs = md['runtime_episode_records']
    chunks = [a for _, a in iter_numeric_evidence(ref)]
    A = {k: np.concatenate([c[k] for c in chunks], axis=0) for k in ('pose', 'planar_motion', 'valid', 'event_steps', 'conditions')}
    for e in range(A['pose'].shape[1]):
        v = A['valid'][:, e].astype(bool)
        block = np.concatenate([A['pose'][v, e], A['planar_motion'][v, e], A['conditions'][v, e].astype(np.float32)], axis=1)
        ev = A['event_steps'][v, e]
        rec = recs[e]
        eps.append({"run": ri, "env": e, "stratum": rec['stratum'], "reason": rec['failure_label'], "steps": int(block.shape[0]),
                    "dt": float(md['time']['step_dt_s']), "anchors": sorted(set(int(x) for x in ev[-1] if x > 0)),
                    "delta_v": [pe['observed_delta_v_world_mps'] for pe in rec['push_events']], "start": len(rows)})
        rows.extend(block.tolist())
COLS = ["height", "roll", "pitch", "planar_speed", "angular_speed", "xy_drift", "vx", "vy", "x", "y",
        "ok_planar_speed", "ok_angular_speed", "ok_roll", "ok_pitch", "ok_height"]
out = io.BytesIO()
np.savez_compressed(out, data=np.asarray(rows, dtype=np.float32), columns=np.array(json.dumps(COLS)), episodes=np.array(json.dumps(eps)))
sys.stdout.buffer.write(out.getvalue())
