"""Independent re-derivation of the Push P3 vocabulary from raw archive arrays.

Reimplements windows, condition spans, dwell/latch/interruptions, post-recovery
instability and the direction sample with numpy; compares with full.json for
every recovery unit. Also measures how informative the chosen direction sample
(last window step) is versus alternatives, using the same arrays.
"""
from __future__ import annotations
import json, sys, time
from collections import defaultdict
import numpy as np

SRC, DEV_RESULT, FULL_JSON, OUT = sys.argv[1:5]
sys.path.insert(0, SRC)
from true_rl_agent.runner.numeric_evidence import load_numeric_evidence_manifest, iter_numeric_evidence

t0 = time.time()
dev = json.load(open(DEV_RESULT))
refs = [s["numeric_evidence_ref"] for s in dev["task_evidence"]["seed_summaries"]]
full = json.load(open(FULL_JSON))
transcripts = {(t["source"], t["env_id"]): t for t in full["transcripts"]}
manifests = sorted(((load_numeric_evidence_manifest(r), r) for r in refs), key=lambda p: p[0]["identity"]["run_index"])
CONDITIONS = manifests[0][0]["metadata"]["condition_fields"]
stats = defaultdict(int); examples = []; alt = defaultdict(list)

def spans(steps, mask):
    if not mask.any():
        return []
    sel = steps[mask]
    parts = np.split(sel, np.flatnonzero(np.diff(sel) != 1) + 1)
    return [{"start_step": int(p[0]), "end_step_inclusive": int(p[-1]), "observed_steps": int(len(p))} for p in parts]

for source, (manifest, ref) in enumerate(manifests):
    md = manifest["metadata"]
    W, D = md["time"]["recovery_window_steps"], md["time"]["required_dwell_steps"]
    dt = md["time"]["step_dt_s"]
    force = "force" in md
    chunks = [a for _, a in iter_numeric_evidence(ref)]
    cat = lambda k: np.concatenate([c[k] for c in chunks], axis=0)
    pose, cond, motion, cstep, valid = cat("pose"), cat("conditions"), cat("planar_motion").astype(np.float64), cat("control_step"), cat("valid")
    records = md["runtime_episode_records"]
    for env in range(valid.shape[1]):
        v = valid[:, env]
        steps, c, m = cstep[v, env], cond[v, env], motion[v, env]
        stable = c.all(axis=1)
        rec = records[env]
        theirs = transcripts[(source, env)]
        stats["episodes"] += 1
        latched = []
        for k, unit in enumerate(theirs["recoveries"]):
            stats["units"] += 1
            if force:
                events = rec["force_events"]
                anchor = unit["anchor_control_step"]  # force anchor = last pulse end (reducer-owned); verified below only structurally
            else:
                ev = next((e for e in rec["push_events"] if e["event_index"] == k), None)
                if ev is None:
                    stats["units_without_executed_event"] += 1
                    if unit["anchor_control_step"] is not None:
                        stats["anchor_without_event_record"] += 1
                    continue
                anchor = ev["control_step"] if ev.get("injection_verified") else None
                if unit["anchor_control_step"] != ev["control_step"]:
                    stats["anchor_step_mismatch"] += 1
            if anchor is None or not unit["anchor_verified"]:
                stats["units_without_verified_anchor"] += 1
                continue
            window = (steps > anchor) & (steps <= anchor + W)
            observed = int(window.sum())
            # spans and failed-step counts per condition
            my_spans = {name: spans(steps, window & ~c[:, i]) for i, name in enumerate(CONDITIONS)}
            their_spans = {name: [{kk: s[kk] for kk in ("start_step", "end_step_inclusive", "observed_steps")} for s in unit["condition_violation_spans"][name]] for name in CONDITIONS}
            if my_spans != their_spans:
                stats["span_mismatch_units"] += 1
                if len(examples) < 20: examples.append({"source": source, "env": env, "unit": k, "mine": my_spans, "theirs": their_spans})
            my_failed = {name: int((window & ~c[:, i]).sum()) for i, name in enumerate(CONDITIONS)}
            if my_failed != unit["failed_condition_steps"]:
                stats["failed_steps_mismatch_units"] += 1
            if observed != unit["observed_window_steps"]:
                stats["observed_window_mismatch"] += 1
            if int((window & stable).sum()) != unit["stable_window_steps"]:
                stats["stable_window_mismatch"] += 1
            # dwell / latch / interruptions from scratch
            streak, max_dwell, latch, interruptions = 0, 0, None, []
            for idx in np.flatnonzero(window):
                if stable[idx]:
                    streak += 1; max_dwell = max(max_dwell, streak)
                    if streak >= D and latch is None:
                        latch = int(steps[idx]); break
                else:
                    if streak > 0:
                        interruptions.append({"control_step": int(steps[idx]), "after_anchor_steps": int(steps[idx] - anchor),
                                              "preceding_dwell_steps": streak,
                                              "failed_conditions": [n for n, ok in zip(CONDITIONS, c[idx]) if not ok]})
                    streak = 0
            if latch != unit["recovery_control_step"]:
                stats["latch_mismatch"] += 1
                if len(examples) < 20: examples.append({"source": source, "env": env, "unit": k, "latch_mine": latch, "latch_theirs": unit["recovery_control_step"]})
            their_int = [{kk: e[kk] for kk in ("control_step", "after_anchor_steps", "preceding_dwell_steps", "failed_conditions")} for e in unit["dwell_interruptions"]]
            if interruptions != their_int:
                stats["interruption_mismatch"] += 1
                if len(examples) < 20: examples.append({"source": source, "env": env, "unit": k, "int_mine": interruptions, "int_theirs": their_int})
            if max_dwell != unit["max_dwell_steps"]:
                stats["max_dwell_mismatch"] += 1
            if latch is not None: latched.append(latch)
            # direction sample: last window step (as implemented) + alternatives
            d = unit["direction"]
            if d.get("status") == "available" and not force:
                vec = np.array(ev["observed_delta_v_world_mps"][:2]); along = vec / np.linalg.norm(vec); lat = np.array([-along[1], along[0]])
                idxs = np.flatnonzero(window)
                last = m[idxs[-1]]
                mine_dir = {"control_step": int(steps[idxs[-1]]), "velocity_parallel_mps": float(last[:2] @ along), "velocity_lateral_mps": float(last[:2] @ lat),
                            "reset_displacement_parallel_m": float(last[2:] @ along), "reset_displacement_lateral_m": float(last[2:] @ lat)}
                if mine_dir["control_step"] != d["control_step"] or any(abs(mine_dir[kk] - d[kk]) > 1e-9 for kk in mine_dir if kk != "control_step"):
                    stats["direction_mismatch"] += 1
                    if len(examples) < 20: examples.append({"source": source, "env": env, "unit": k, "dir_mine": mine_dir, "dir_theirs": d})
                vpar = m[idxs, :2] @ along
                first = m[idxs[0]]
                alt["v_par_last"].append(float(vpar[-1])); alt["v_par_first"].append(float(vpar[0])); alt["v_par_peak"].append(float(vpar.max()))
                alt["v_par_last_abs"].append(abs(float(vpar[-1])))
                alt["disp_par_over_window"].append(float((last[2:] - first[2:]) @ along))
                alt["disp_par_reset_last"].append(float(last[2:] @ along))
                alt["push_mag"].append(float(np.linalg.norm(vec)))
                alt["outcome"].append(theirs["terminal_label"])
                alt["latched"].append(latch is not None)
        # post-recovery instability
        all_latched = bool(latched) and len(latched) == len(theirs["recoveries"]) and all(u["recovery_control_step"] is not None for u in theirs["recoveries"])
        if all_latched:
            later = (steps > max(latched)) & ~stable
            n_later = int(later.sum())
            first_later = int(np.flatnonzero(later)[0]) if later.any() else None
            if n_later != theirs["unstable_steps_after_all_recoveries"]:
                stats["post_recovery_count_mismatch"] += 1
            mine_first = None if first_later is None else {"control_step": int(steps[first_later]), "failed_conditions": [n for n, ok in zip(CONDITIONS, c[first_later]) if not ok]}
            th = theirs["first_instability_after_all_recoveries"]
            th = None if th is None else {"control_step": th["control_step"], "failed_conditions": th["failed_conditions"]}
            if mine_first != th:
                stats["post_recovery_first_mismatch"] += 1
        elif theirs["unstable_steps_after_all_recoveries"] != 0 or theirs["all_required_recoveries_latched"]:
            stats["post_recovery_flag_mismatch"] += 1
        final_failed = [n for n, ok in zip(CONDITIONS, c[-1]) if not ok]
        if final_failed != theirs["final_failed_conditions"]:
            stats["final_failed_mismatch"] += 1
        # sanity of frame convention: the event-step frame is pre-injection (P2 fact)
        if not force:
            for ev in rec["push_events"]:
                i_ev = np.flatnonzero(steps == ev["control_step"])
                if len(i_ev) and ev.get("injection_verified"):
                    vb = np.array(ev["velocity_before_world_6d"][:2]); va = np.array(ev["velocity_after_next_physics_step_world_6d"][:2])
                    at = m[i_ev[0], :2]; nxt = m[i_ev[0] + 1, :2] if i_ev[0] + 1 < len(steps) else None
                    stats["event_frame_equals_velocity_before"] += int(np.allclose(at, vb, atol=1e-5))
                    if nxt is not None:
                        stats["event_plus1_cos_gt_0.9"] += int(float(nxt @ (va / (np.linalg.norm(va) + 1e-12))) / (np.linalg.norm(nxt) + 1e-12) > 0.9)
                    stats["events_frame_checked"] += 1

# informativeness summary of the direction sample
if alt["v_par_last"]:
    A = {k: np.array(v) for k, v in alt.items() if k not in ("outcome", "latched")}
    outcome = np.array(alt["outcome"]); latched = np.array(alt["latched"])
    summary = {}
    for key in ("v_par_last", "v_par_last_abs", "v_par_first", "v_par_peak", "disp_par_over_window", "disp_par_reset_last"):
        summary[key] = {"median": float(np.median(A[key])), "p10": float(np.percentile(A[key], 10)), "p90": float(np.percentile(A[key], 90)),
                        "median_success": float(np.median(A[key][outcome == "success_recovered"])) if (outcome == "success_recovered").any() else None,
                        "median_not_recovered": float(np.median(A[key][outcome == "not_recovered"])) if (outcome == "not_recovered").any() else None}
    summary["push_mag_median"] = float(np.median(A["push_mag"]))
    summary["fraction_last_velocity_below_recovery_threshold_0.2155"] = float((A["v_par_last_abs"] < 0.21554618994545).mean())
    summary["fraction_peak_parallel_gt_half_push"] = float((A["v_par_peak"] > 0.5 * A["push_mag"]).mean())
    summary["units_measured"] = int(len(A["v_par_last"]))
else:
    summary = {}
out = {"stats": dict(stats), "examples": examples, "direction_sample_informativeness": summary, "elapsed_s": round(time.time() - t0, 1)}
json.dump(out, open(OUT, "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "examples"}, indent=1))
print("examples:", json.dumps(examples[:5])[:3000])
