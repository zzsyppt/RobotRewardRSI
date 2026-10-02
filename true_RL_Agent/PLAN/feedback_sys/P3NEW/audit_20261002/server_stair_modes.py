# Read-only audit probe: geometry-anchored per-episode facts from the sealed R4 Stair archive.
# Uses the frozen source's own decoder; writes nothing on the server (results go to stdout).
import sys, json, math, collections, statistics as st
SRC = '/data/yuyangliu/AGENT/feedback_p3new_20260929/n5_confirmation_source_r4_final/src'
sys.path.insert(0, SRC)
import torch
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
buf = {}
episodes = []
def observe(run, t, raw, outcome):
    key = (run, t.env_index)
    b = buf.setdefault(key, [])
    w, x, y, z = (float(v) for v in raw['base_quat_w'][t.env_index].tolist())
    bx = (1-2*(y*y+z*z), 2*(x*y+w*z))
    bz = (2*(x*z+w*y), 2*(y*z-w*x), 1-2*(x*x+y*y))
    h = math.atan2(bx[1], bx[0]); c, s = math.cos(-h), math.sin(-h)
    lean_f, lean_l = c*bz[0]-s*bz[1], s*bz[0]+c*bz[1]
    roll = math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y)); pitch = math.asin(max(-1., min(1., 2*(w*y-z*x))))
    b.append((t.progress_m, t.base_clearance_m, t.support_surface_levels, t.support_contacts,
              roll, pitch, bz[2], lean_f, lean_l, t.lateral_m, t.base_height_m))
    if outcome is None:
        return
    cell = cells[t.env_index]; spec = cell.spec; dt = spec.control_dt_s
    T = len(b); xs = [r[0] for r in b]
    runmax = xs[0]; last_gain = 0
    for i, v in enumerate(xs):
        if v > runmax + 1e-6: runmax, last_gain = v, i
    n_levels = len(spec.riser_heights_m)
    k = int(round(0.5/dt))
    tail = b[max(0, T-k):]
    lv_end = b[-1][2]
    upz = [r[6] for r in b]
    first_inv = next((i for i, v in enumerate(upz) if v < 0), None)
    first_60 = next((i for i, v in enumerate(upz) if v < 0.5), None)
    direction = None
    if first_60 is not None:
        f, l = b[first_60][7], b[first_60][8]
        direction = ('right' if l < 0 else 'left') if abs(l) >= abs(f) else ('backward' if f < 0 else 'forward')
    on_stairs = [r[1] for r in b if max(r[2]) >= 1]
    onr = [r for r in b if max(r[2]) >= 1 and min(r[2]) < n_levels]
    spread2 = sum((max(r[2][0], r[2][1]) - min(r[2][2], r[2][3])) >= 2 for r in onr)
    low08 = sum(r[1] < 0.08 for r in onr); low05 = sum(r[1] < 0.05 for r in onr)
    pitch_on = sorted(r[5] for r in onr)
    rec = {"run": run, "env": t.env_index, "cell": spec.cell_id, "stratum": spec.stratum_id if hasattr(spec, 'stratum_id') else None,
           "reason": outcome.primary_reason, "steps": T, "dt": dt, "n_levels": n_levels,
           "stall_s": (T-1-last_gain)*dt, "x_end": xs[-1], "x_max": runmax,
           "adv_last_0p5s": xs[-1]-xs[max(0, T-1-k)],
           "front_level_end": max(lv_end[0], lv_end[1]), "hind_level_end": min(lv_end[2], lv_end[3]),
           "front_level_max": max(max(r[2][0], r[2][1]) for r in b),
           "loaded_end": sum(b[-1][3]), "loaded_last5_mean": sum(sum(r[3]) for r in b[-5:])/len(b[-5:]),
           "base_clr_end": b[-1][1], "base_clr_min_tail": min(r[1] for r in tail),
           "base_clr_median_on_stairs": st.median(on_stairs) if on_stairs else None,
           "base_clr_min_on_stairs": min(on_stairs) if on_stairs else None,
           "pitch_end": b[-1][5], "pitch_min_tail": min(r[5] for r in tail), "roll_end": b[-1][4],
           "inverted": first_inv is not None, "tilt60_dir": direction,
           "first_inv_before_end_s": None if first_inv is None else (T-1-first_inv)*dt,
           "lateral_end": b[-1][9], "on_stairs_steps": len(onr), "spread2_steps": spread2,
           "clr_lt_0p08_steps": low08, "clr_lt_0p05_steps": low05,
           "pitch_median_on_stairs": (pitch_on[len(pitch_on)//2] if pitch_on else None),
           "pitch_min_on_stairs": (pitch_on[0] if pitch_on else None),
           "spread_end": max(lv_end[0], lv_end[1]) - min(lv_end[2], lv_end[3])}
    episodes.append(rec)
    del buf[key]
res = recompute_stair_numeric_evidence(REF, observe_transition=observe)
assert not buf and len(episodes) == res['episode_count'], (len(buf), len(episodes), res['episode_count'])
print(json.dumps({"episode_count": len(episodes), "episodes": episodes}))
