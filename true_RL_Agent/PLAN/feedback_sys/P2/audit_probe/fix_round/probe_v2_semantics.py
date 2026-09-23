"""(1) Are history slots previous control steps or intra-step samples? (2) velocity-protocol direction at +1/+2 steps."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "/data/yuyangliu/p2_audit/src_fix/src")
from true_rl_agent.runner.numeric_evidence import iter_numeric_evidence, load_numeric_evidence_manifest
R = Path("/data/yuyangliu/AGENT/feedback_p2_audit_20260924")
out = {}
name = "stair_parent_on"
ref = json.load(open(R/name/"public_dev_result.json"))["task_evidence"]["numeric_evidence"]
man = load_numeric_evidence_manifest(ref)
seeds = man["metadata"]["seeds"]
T = {(si+1, e): (int(o["episode_steps"]), o["primary_reason"]) for si, s in enumerate(seeds) for e, o in enumerate(s["outcomes"])}
# collect per (run, env) the sequence of forbidden inst norms and history slot norms for all valid steps (only forbidden episodes, to limit memory)
seq = {}
for meta, arrays in iter_numeric_evidence(ref):
    run, start = meta["run_index"], meta["step_start"]
    fi = np.linalg.norm(arrays["forbidden_contact_forces_w"], axis=-1).max(axis=-1)          # [steps, env]
    fh = np.linalg.norm(arrays["forbidden_contact_forces_w_history"], axis=-1).max(axis=-1)  # [steps, env, H]
    si_ = np.linalg.norm(arrays["support_contact_forces_w"], axis=-1)                         # [steps, env, B]
    sh = np.linalg.norm(arrays["support_contact_forces_w_history"], axis=-1)                  # [steps, env, H, B]
    valid = arrays["valid"]
    for e in range(valid.shape[1]):
        if T[(run, e)][1] != "forbidden_contact": continue
        for s in range(meta["steps"]):
            if valid[s, e]:
                seq.setdefault((run, e), []).append((start + s, float(fi[s, e]), fh[s, e].tolist(), si_[s, e].tolist(), sh[s, e].tolist()))
# Test the "slot k == instantaneous of control step t-k" hypothesis on support bodies (frequent nonzero) across all forbidden episodes
agree1 = total1 = agree2 = total2 = 0
examples = []
for key, rows in seq.items():
    rows.sort()
    for i in range(2, len(rows)):
        t, fi_t, fh_t, si_t, sh_t = rows[i]
        _, _, _, si_t1, _ = rows[i-1]; _, _, _, si_t2, _ = rows[i-2]
        for b in range(len(si_t)):
            total1 += 1; agree1 += int(abs(sh_t[1][b] - si_t1[b]) <= 1e-3 * max(1.0, si_t1[b]))
            total2 += 1; agree2 += int(abs(sh_t[2][b] - si_t2[b]) <= 1e-3 * max(1.0, si_t2[b]))
    t, fi_t, fh_t, si_t, sh_t = rows[-1]
    if fi_t == 0 and max(fh_t) > 1.0 and len(examples) < 6:
        prev = [(r[0], round(r[1], 1), [round(x, 1) for x in r[2]]) for r in rows[-4:]]
        examples.append({"run_env": list(key), "terminal": t, "last4_steps_(step, inst_forbidden_N, hist_slots_N)": prev})
out["slot_k_equals_inst_of_step_t_minus_k"] = {"slot1_agree_frac": agree1 / max(1, total1), "slot2_agree_frac": agree2 / max(1, total2), "bodyframes": total1}
out["examples_inst_zero_hist_nonzero_terminal"] = examples
print(json.dumps(out, indent=1))

# (2) velocity protocol: direction of root_lin_vel_w.xy vs requested delta-v at injection step + k
name = "push_go2_v2_on"
res = {}
for ref in json.load(open(R/name/"numeric_references.json")):
    man = load_numeric_evidence_manifest(ref); recs = man["metadata"]["runtime_episode_records"]
    frames = {}
    for meta, arrays in iter_numeric_evidence(ref):
        v = arrays["valid"]; pm = arrays["planar_motion"]
        for row in range(v.shape[0]):
            for e in range(v.shape[1]):
                if v[row, e]: frames[(e, int(arrays["control_step"][row, e]))] = pm[row, e]
    for e, rec in enumerate(recs):
        for ev in rec.get("push_events") or []:
            d = np.array(ev["requested_delta_v_world_mps"], dtype=np.float64); step = int(ev["control_step"])
            before = np.array(ev["velocity_before_world_6d"][:2]); after = np.array(ev["velocity_after_next_physics_step_world_6d"][:2])
            for k in (0, 1, 2, 3):
                f = frames.get((e, step + k))
                if f is None or np.linalg.norm(d) == 0: continue
                vel = f[:2].astype(np.float64)
                if np.linalg.norm(vel) > 1e-6:
                    res.setdefault(f"cos_at_injection_plus_{k}", []).append(float(vel @ d / (np.linalg.norm(vel) * np.linalg.norm(d))))
            f0 = frames.get((e, step))
            if f0 is not None:
                res.setdefault("frame_at_injection_step_matches_velocity_before", []).append(float(np.linalg.norm(f0[:2] - before)))
                res.setdefault("frame_at_injection_step_matches_velocity_after_next_physics", []).append(float(np.linalg.norm(f0[:2] - after)))
            f1 = frames.get((e, step + 1))
            if f1 is not None:
                res.setdefault("frame_at_injection_plus1_matches_velocity_after_next_physics", []).append(float(np.linalg.norm(f1[:2] - after)))
summary = {k: {"n": len(v), "median": float(np.median(v)), "frac_gt_0.9": float((np.array(v) > 0.9).mean())} if k.startswith("cos") else {"n": len(v), "median_abs_diff": float(np.median(v)), "p90": float(np.quantile(v, 0.9))} for k, v in res.items()}
print("VELOCITY_DIRECTION", json.dumps(summary, indent=1))
Path("/data/yuyangliu/p2_audit/probe_v2_semantics.json").write_text(json.dumps({**out, "velocity_direction": summary}, indent=1))
