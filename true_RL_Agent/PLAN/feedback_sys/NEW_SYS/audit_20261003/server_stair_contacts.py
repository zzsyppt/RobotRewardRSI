# Read-only audit probe: per-step production contact flags (force norm > task threshold) and 3-D foot forces of the
# sealed R4 Stair archive as NPZ on stdout. Uses the frozen source decoder; writes nothing on the server.
import sys, json, math, io
SRC = '/data/yuyangliu/AGENT/feedback_p3new_20260929/n5_confirmation_source_r4_final/src'
sys.path.insert(0, SRC)
import numpy as np
from true_rl_agent.runner.numeric_evidence import load_numeric_evidence_manifest
from true_rl_agent.tasks.stair_climbing.numeric_capture import recompute_stair_numeric_evidence
REF = {"manifest_path": "/data/yuyangliu/AGENT/feedback_p3new_20260929/n5_confirmation_20261001_r4/stair/numeric_evidence/manifest.json",
       "manifest_sha256": "8c5f2d68db3df151c850beb6e2147ced671fb00acc802e23427f188c0d7e9d83",
       "schema_version": "true_rl_agent.numeric_evidence.ref.v1"}
buf, rows, eps = {}, [], []
def observe(run, t, raw, outcome):
    i = t.env_index
    b = buf.setdefault((run, i), [])
    force = raw['support_contact_forces_w'][i].tolist()
    row = [float(c) for c in t.support_contacts] + [math.sqrt(sum(v * v for v in f)) for f in force] + [f[2] for f in force]
    b.append(row)
    if outcome is None:
        return
    eps.append({"run": run, "env": i, "reason": outcome.primary_reason, "steps": len(b), "start": len(rows)})
    rows.extend(b)
    del buf[(run, i)]
res = recompute_stair_numeric_evidence(REF, observe_transition=observe)
assert not buf and len(eps) == res['episode_count']
out = io.BytesIO()
np.savez_compressed(out, data=np.asarray(rows, dtype=np.float32), episodes=np.array(json.dumps(eps)))
sys.stdout.buffer.write(out.getvalue())
