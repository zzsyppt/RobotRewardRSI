"""Independent checks of the new v2 operands: Stair history predicate vs manager term; Push motion direction physics."""
import json, math, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "/data/yuyangliu/p2_audit/src_fix/src")
from true_rl_agent.runner.numeric_evidence import iter_numeric_evidence, load_numeric_evidence_manifest
R = Path("/data/yuyangliu/AGENT/feedback_p2_audit_20260924")
out = {}
# ---- Stair: history-max predicate == base_contact manager mask on every valid frame (own implementation)
for name in ("stair_parent_on", "stair_g1_on"):
    ref = json.load(open(R/name/"public_dev_result.json"))["task_evidence"]["numeric_evidence"]
    man = load_numeric_evidence_manifest(ref)
    thr = float(man["metadata"]["semantics"]["episode_outcomes"]["signals"]["forbidden_contact"]["force_threshold_n"])
    seeds = man["metadata"]["seeds"]
    T = {(si+1, e): (int(o["episode_steps"]), o["primary_reason"]) for si, s in enumerate(seeds) for e, o in enumerate(s["outcomes"])}
    st = {"valid_frames": 0, "predicate_mismatch": 0, "manager_true_frames": 0, "terminal_forbidden": 0,
          "terminal_inst_zero_hist_nonzero": 0, "terminal_hist_zero_manager_true": 0, "hist_newest_equals_instantaneous_frames": 0,
          "support_hist_vs_inst_contact_mismatch_bodyframes": 0, "support_bodyframes": 0, "padding_history_nonzero": 0, "threshold": thr}
    for meta, arrays in iter_numeric_evidence(ref):
        bc = meta["termination_term_order"].index("base_contact")
        valid = arrays["valid"]                                   # [steps, env]
        fh = arrays["forbidden_contact_forces_w_history"]         # [steps, env, H, B, 3]
        fi = arrays["forbidden_contact_forces_w"]                 # [steps, env, B, 3]
        sh = arrays["support_contact_forces_w_history"]; si_ = arrays["support_contact_forces_w"]
        pred = (np.linalg.norm(fh, axis=-1).max(axis=2) > thr).any(axis=-1)   # [steps, env]
        mgr = arrays[f"termination_{bc}"]
        st["valid_frames"] += int(valid.sum())
        st["predicate_mismatch"] += int((pred[valid] != mgr[valid]).sum())
        st["manager_true_frames"] += int(mgr[valid].sum())
        st["hist_newest_equals_instantaneous_frames"] += int(np.all(np.isclose(fh[:, :, 0], fi), axis=(-1, -2))[valid].sum())
        sc_h = np.linalg.norm(sh, axis=-1).max(axis=2) > thr; sc_i = np.linalg.norm(si_, axis=-1) > thr
        st["support_bodyframes"] += int(valid.sum()) * sh.shape[3]
        st["support_hist_vs_inst_contact_mismatch_bodyframes"] += int((sc_h != sc_i)[valid].sum())
        st["padding_history_nonzero"] += int((np.abs(fh[~valid]).sum(axis=(-1,-2,-3)) > 0).sum())
        run, start = meta["run_index"], meta["step_start"]
        for e in range(valid.shape[1]):
            t, reason = T[(run, e)]
            if reason != "forbidden_contact" or not (start <= t < start + meta["steps"]): continue
            s = t - start
            st["terminal_forbidden"] += 1
            inst = float(np.linalg.norm(fi[s, e], axis=-1).max()); hist = float(np.linalg.norm(fh[s, e], axis=-1).max())
            st["terminal_inst_zero_hist_nonzero"] += int(inst == 0 and hist > 0)
            st["terminal_hist_zero_manager_true"] += int(hist <= thr and bool(mgr[s, e]))
    out[name] = st; print(name, json.dumps(st), flush=True)
# ---- Push: direction physics
def push_probe(name, kind):
    refs = json.load(open(R/name/"numeric_references.json"))
    st = {"kind": kind, "seeds": 0, "norm_speed_max_rel_err": 0.0, "norm_drift_max_rel_err": 0.0, "events": 0,
          "cos_vel_vs_push_at_injection_median": None, "cos_list": [], "cos_drift_vs_push_1s_later_median": None, "drift_cos": []}
    for ref in refs:
        man = load_numeric_evidence_manifest(ref); md = man["metadata"]; st["seeds"] += 1
        recs = md["runtime_episode_records"]
        frames = {}
        for meta, arrays in iter_numeric_evidence(ref):
            v = arrays["valid"]; pm = arrays["planar_motion"]; pose = arrays["pose"]
            sp = np.linalg.norm(pm[..., :2].astype(np.float64), axis=-1); dr = np.linalg.norm(pm[..., 2:].astype(np.float64), axis=-1)
            with np.errstate(divide="ignore", invalid="ignore"):
                st["norm_speed_max_rel_err"] = max(st["norm_speed_max_rel_err"], float(np.nanmax(np.where(v, np.abs(sp - pose[..., 3]) / np.maximum(pose[..., 3], 1e-6), 0))))
                st["norm_drift_max_rel_err"] = max(st["norm_drift_max_rel_err"], float(np.nanmax(np.where(v, np.abs(dr - pose[..., 5]) / np.maximum(pose[..., 5], 1e-6), 0))))
            for row in range(v.shape[0]):
                for e in range(v.shape[1]):
                    if v[row, e]: frames[(e, int(arrays["control_step"][row, e]))] = pm[row, e]
        for e, rec in enumerate(recs):
            events = rec.get("push_events") or rec.get("force_events") or []
            for ev in events:
                if kind == "velocity":
                    d = np.array(ev["requested_delta_v_world_mps"], dtype=np.float64); step = int(ev["control_step"]); anchor = step
                else:
                    d = np.array(ev["force_world_n"][:2], dtype=np.float64); step = int(ev["end_control_step"]); anchor = step
                if np.linalg.norm(d) == 0 or (e, step) not in frames: continue
                vel = frames[(e, step)][:2].astype(np.float64)
                if np.linalg.norm(vel) > 0:
                    st["cos_list"].append(float(vel @ d / (np.linalg.norm(vel) * np.linalg.norm(d)))); st["events"] += 1
                later = frames.get((e, anchor + 50))
                if later is not None and np.linalg.norm(later[2:]) > 0.01:
                    st["drift_cos"].append(float(later[2:].astype(np.float64) @ d / (np.linalg.norm(later[2:]) * np.linalg.norm(d))))
    if st["cos_list"]:
        c = np.array(st["cos_list"]); st["cos_vel_vs_push_at_injection_median"] = float(np.median(c)); st["cos_vel_frac_gt_0.9"] = float((c > 0.9).mean())
    if st["drift_cos"]:
        c = np.array(st["drift_cos"]); st["cos_drift_vs_push_1s_later_median"] = float(np.median(c)); st["drift_cos_frac_gt_0"] = float((c > 0).mean())
    st.pop("cos_list"); st.pop("drift_cos"); return st
for name, kind in (("push_go2_v2_on", "velocity"), ("push_g1_v3_on", "force")):
    out[name] = push_probe(name, kind); print(name, json.dumps(out[name]), flush=True)
Path("/data/yuyangliu/p2_audit/probe_v2_fields.json").write_text(json.dumps(out, indent=1))
