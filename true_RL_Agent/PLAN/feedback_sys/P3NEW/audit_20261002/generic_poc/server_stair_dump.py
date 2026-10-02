# Read-only audit probe: dump per-step generic state columns of the sealed R4 Stair archive as NPZ on stdout.
# Uses the frozen source decoder; writes nothing on the server.
import sys, json, math, io
SRC = '/data/yuyangliu/AGENT/feedback_p3new_20260929/n5_confirmation_source_r4_final/src'
sys.path.insert(0, SRC)
import numpy as np
from true_rl_agent.runner.numeric_evidence import load_numeric_evidence_manifest
from true_rl_agent.tasks.stair_climbing.numeric_capture import recompute_stair_numeric_evidence
from true_rl_agent.tasks.stair_climbing.unified_evaluation import StairUnifiedEvaluationSuite
from true_rl_agent.tasks.stair_climbing.unified_training import StairUnifiedTrainingProfile
REF = {"manifest_path": "/data/yuyangliu/AGENT/feedback_p3new_20260929/n5_confirmation_20261001_r4/stair/numeric_evidence/manifest.json",
       "manifest_sha256": "8c5f2d68db3df151c850beb6e2147ced671fb00acc802e23427f188c0d7e9d83",
       "schema_version": "true_rl_agent.numeric_evidence.ref.v1"}
archive = load_numeric_evidence_manifest(REF); md = archive['metadata']
training = StairUnifiedTrainingProfile.from_mapping(md['training_profile'])
suite = StairUnifiedEvaluationSuite.from_mapping(md['evaluation_suite'], training_profile=training)
cells = suite.private_cell_assignment('public_dev')
COLS = (["progress", "lateral", "base_height", "base_clearance", "qw", "qx", "qy", "qz", "forbidden_force"]
        + [f"{k}_{i}" for k in ("foot_dx", "foot_dy", "foot_z", "foot_clr", "foot_speed", "foot_fz", "foot_level") for i in range(4)]
        + [f"joint_{i}" for i in range(12)])
buf, rows, eps = {}, [], []
def observe(run, t, raw, outcome):
    i = t.env_index
    key = (run, i)
    b = buf.setdefault(key, [])
    origin = raw['terrain_env_origins_w'][i].tolist()
    q = raw['base_quat_w'][i].tolist()
    feet = raw['support_pos_w'][i].tolist(); force = raw['support_contact_forces_w'][i].tolist()
    fb = raw['forbidden_contact_forces_w'][i].tolist()
    fbn = max(math.sqrt(sum(v*v for v in vec)) for vec in fb) if fb else 0.0
    row = [t.progress_m, t.lateral_m, t.base_height_m, t.base_clearance_m, *q, fbn]
    row += [t.support_progress_m[k] - t.progress_m for k in range(4)]
    row += [t.support_lateral_m[k] - t.lateral_m for k in range(4)]
    row += [feet[k][2] - origin[2] for k in range(4)]
    row += list(t.support_surface_clearances_m)
    row += list(t.support_planar_speeds_mps)
    row += [force[k][2] for k in range(4)]
    row += [float(v) for v in t.support_surface_levels]
    row += raw['joint_pos'][i].tolist()
    b.append(row)
    if outcome is None:
        return
    spec = cells[i].spec
    eps.append({"run": run, "env": i, "cell": spec.cell_id, "stratum": spec.stratum_id, "reason": outcome.primary_reason,
                "steps": len(b), "dt": spec.control_dt_s, "goal_m": spec.anchor_progress_goal_m,
                "n_levels": len(spec.riser_heights_m), "start": len(rows)})
    rows.extend(b)
    del buf[key]
res = recompute_stair_numeric_evidence(REF, observe_transition=observe)
assert not buf and len(eps) == res['episode_count']
out = io.BytesIO()
np.savez_compressed(out, data=np.asarray(rows, dtype=np.float32), columns=np.array(json.dumps(COLS)), episodes=np.array(json.dumps(eps)))
sys.stdout.buffer.write(out.getvalue())
