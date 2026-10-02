# Read-only audit probe: root-motion characterisation of the sealed R4 Push archive (stdout only).
import sys, json
SRC = '/data/yuyangliu/AGENT/feedback_p3new_20260929/n5_confirmation_source_r4_final/src'
sys.path.insert(0, SRC)
import numpy as np
from true_rl_agent.runner.numeric_evidence import load_numeric_evidence_manifest, iter_numeric_evidence
refs = json.load(open('/data/yuyangliu/AGENT/feedback_p3new_20260929/n5_confirmation_20261001_r4/push/numeric_references.json'))
out = []
for ri, ref in enumerate(refs, 1):
    man = load_numeric_evidence_manifest(ref); md = man['metadata']
    recs = md['runtime_episode_records']
    chunks = [a for _, a in iter_numeric_evidence(ref)]
    keys = chunks[0].keys()
    A = {k: np.concatenate([c[k] for c in chunks], axis=0) for k in ('pose', 'planar_motion', 'valid', 'control_step', 'event_steps', 'conditions')}
    n = A['pose'].shape[1]
    if ri == 1:
        print(json.dumps({"keys": sorted(keys), "pose_shape": list(A['pose'].shape), "rec0": recs[0], "md_keys": sorted(md.keys())}), file=sys.stderr)
    thr = float(md['parameters']['stable_planar_speed_mps']) if 'stable_planar_speed_mps' in md.get('parameters', {}) else 0.215546
    for e in range(n):
        v = A['valid'][:, e].astype(bool)
        pose = A['pose'][v, e]; pm = A['planar_motion'][v, e]; ev = A['event_steps'][v, e]
        T = pose.shape[0]
        h, roll, pitch, spd, ang, drift = pose.T
        anchors = sorted(set(int(x) for x in ev[-1] if x > 0))
        a0 = anchors[0] if anchors else None
        rec = recs[e]
        d = {"run": ri, "env": e, "T": int(T), "label": rec.get('failure_label'),
             "rec_keys": sorted(rec.keys()) if (ri == 1 and e == 0) else None, "anchors": anchors,
             "pre_speed_mean": float(spd[20:a0-1].mean()) if a0 and a0 > 25 else None,
             "pre_frac_above": float((spd[20:a0-1] > thr).mean()) if a0 and a0 > 25 else None}
        if a0 is not None and T > a0 + 5:
            post = spd[a0:]
            d.update(post_frac_above=float((post > thr).mean()), post_max=float(post.max()),
                     upright_all=bool((np.abs(roll[a0:]) <= 0.2).all() and (np.abs(pitch[a0:]) <= 0.2).all() and (h[a0:] >= 0.518).all()))
            last = anchors[-1]; late = spd[min(T, last + 110):]
            if len(late) > 10:
                ab = late > thr
                runs = np.diff(np.concatenate([[0], ab.astype(int), [0]])); starts = np.nonzero(runs == 1)[0]; ends = np.nonzero(runs == -1)[0]
                d.update(late_steps=int(len(late)), late_frac_above=float(ab.mean()), late_max=float(late.max()), late_mean=float(late.mean()),
                         late_bursts=int(len(starts)), late_burst_len_median=float(np.median(ends - starts)) if len(starts) else 0.0,
                         late_height_mean=float(h[min(T, last + 110):].mean()), late_ang_mean=float(ang[min(T, last + 110):].mean()))
        out.append(d)
print(json.dumps(out))
