# Read-only audit probe: clearance of sample points on the base (trunk) collision box above the analytic stair surface,
# every step, sealed R4 Stair archive. NPZ on stdout; writes nothing on the server.
import sys, json, math, io
SRC = '/data/yuyangliu/AGENT/feedback_p3new_20260929/n5_confirmation_source_r4_final/src'
sys.path.insert(0, SRC)
import numpy as np
from true_rl_agent.runner.numeric_evidence import load_numeric_evidence_manifest
from true_rl_agent.tasks.stair_climbing.numeric_capture import recompute_stair_numeric_evidence
from true_rl_agent.tasks.stair_climbing.unified_evaluation import StairUnifiedEvaluationSuite
from true_rl_agent.tasks.stair_climbing.unified_training import StairUnifiedTrainingProfile
from true_rl_agent.tasks.stair_climbing.terrain_evidence import stair_surface_query_for_ordered_cell
REF = {"manifest_path": "/data/yuyangliu/AGENT/feedback_p3new_20260929/n5_confirmation_20261001_r4/stair/numeric_evidence/manifest.json",
       "manifest_sha256": "8c5f2d68db3df151c850beb6e2147ced671fb00acc802e23427f188c0d7e9d83",
       "schema_version": "true_rl_agent.numeric_evidence.ref.v1"}
archive = load_numeric_evidence_manifest(REF); md = archive['metadata']
training = StairUnifiedTrainingProfile.from_mapping(md['training_profile'])
suite = StairUnifiedEvaluationSuite.from_mapping(md['evaluation_suite'], training_profile=training)
cells = suite.private_cell_assignment('public_dev')
queries = [stair_surface_query_for_ordered_cell(c.spec) for c in cells]
# Sample points in the base frame (x forward, y left, z up). Box half extents from the Go2 trunk collision box
# (0.3762 x 0.0935 x 0.114 m); head point from the project's body-frame note (Head_upper at x=+0.285).
HX, HY, HZ = 0.1881, 0.04675, 0.057
PTS = [("bottom_front_left", HX, HY, -HZ), ("bottom_front_right", HX, -HY, -HZ), ("bottom_rear_left", -HX, HY, -HZ),
       ("bottom_rear_right", -HX, -HY, -HZ), ("bottom_centre", 0.0, 0.0, -HZ), ("bottom_front_mid", HX, 0.0, -HZ),
       ("bottom_rear_mid", -HX, 0.0, -HZ), ("head_low", 0.30, 0.0, -HZ), ("head_mid", 0.30, 0.0, 0.0),
       ("top_front_mid", HX, 0.0, HZ), ("top_rear_mid", -HX, 0.0, HZ), ("top_centre", 0.0, 0.0, HZ)]
buf, rows, eps = {}, [], []
def observe(run, t, raw, outcome):
    i = t.env_index; key = (run, i); b = buf.setdefault(key, [])
    w, x, y, z = raw['base_quat_w'][i].tolist()
    R = ((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)), (2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)), (2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)))
    q = queries[i]; row = []
    for _, px, py, pz in PTS:
        dx = R[0][0]*px + R[0][1]*py + R[0][2]*pz; dy = R[1][0]*px + R[1][1]*py + R[1][2]*pz; dz = R[2][0]*px + R[2][1]*py + R[2][2]*pz
        s = q.sample(progress_m=t.progress_m + dx, lateral_m=t.lateral_m + dy)
        row.append(t.base_height_m + dz - s.height_m)
    b.append(row)
    if outcome is None:
        return
    eps.append({"run": run, "env": i, "reason": outcome.primary_reason, "steps": len(b), "start": len(rows)})
    rows.extend(b); del buf[key]
res = recompute_stair_numeric_evidence(REF, observe_transition=observe)
assert not buf and len(eps) == res['episode_count']
out = io.BytesIO()
np.savez_compressed(out, data=np.asarray(rows, dtype=np.float32), points=np.array(json.dumps([p[0] for p in PTS])), episodes=np.array(json.dumps(eps)))
sys.stdout.buffer.write(out.getvalue())
