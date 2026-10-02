"""Independent re-derivation of the Stair P3 vocabulary from raw archive arrays.

Uses only: the P2 archive reader (chunk iteration), the ordered-cell surface
query for clearance, and the cell assignment. Everything else (Euler angles,
predicates, streaks, events, anchor, windows, ranges) is reimplemented here
with numpy and compared against the user's full.json for all episodes.
"""
from __future__ import annotations
import json, math, sys, time
from collections import defaultdict
import numpy as np

SRC = sys.argv[1]
DEV_RESULT = sys.argv[2]
FULL_JSON = sys.argv[3]
OUT = sys.argv[4]
sys.path.insert(0, SRC)
from true_rl_agent.runner.numeric_evidence import load_numeric_evidence_manifest, iter_numeric_evidence
from true_rl_agent.composition.cell_manifest import ResolvedCellManifestV1
from true_rl_agent.tasks.stair_climbing.unified_evaluation import StairUnifiedEvaluationSuite
from true_rl_agent.tasks.stair_climbing.unified_training import StairUnifiedTrainingProfile
from true_rl_agent.tasks.stair_climbing.terrain_evidence import stair_surface_query_for_ordered_cell

t0 = time.time()
ref = json.load(open(DEV_RESULT))["task_evidence"]["numeric_evidence"]
manifest = load_numeric_evidence_manifest(ref)
md = manifest["metadata"]
signals = md["semantics"]["episode_outcomes"]["signals"]
TILT, TILT_N = signals["base_orientation_instability"]["threshold"], signals["base_orientation_instability"]["consecutive_control_steps"]
SLIP_V, SLIP_N, CONTACT_N = signals["slip"]["speed_threshold"], signals["slip"]["consecutive_control_steps"], signals["slip"]["contact_force_threshold"]
CLR, CLR_N = signals["insufficient_clearance"]["threshold"], signals["insufficient_clearance"]["consecutive_control_steps"]
FORB_N = signals["forbidden_contact"]["force_threshold_n"]
training = StairUnifiedTrainingProfile.from_mapping(md["training_profile"])
suite = StairUnifiedEvaluationSuite.from_mapping(md["evaluation_suite"], training_profile=training)
cells = suite.private_cell_assignment("public_dev")
names = list(md["runtime_projection"]["resolved_roles"]["support_effector"])
queries = [stair_surface_query_for_ordered_cell(c.spec) for c in cells]

# ---- gather per-run arrays -------------------------------------------------
runs = defaultdict(list)
for header, arrays in iter_numeric_evidence(ref):
    runs[header["run_index"]].append((header["step_start"], header["termination_term_order"], arrays))
full = json.load(open(FULL_JSON))
by_id = {e["episode_id"]: e for e in full["episodes"]}
print("episodes in full.json:", len(by_id), "runs:", sorted(runs), "elapsed", round(time.time() - t0, 1), flush=True)

REQUIRED = {"tilt": TILT_N, "slip_any": SLIP_N, "low_clearance_any": CLR_N, "low_progress": 1,
            "slip_body": SLIP_N, "low_clearance_body": CLR_N, "unloaded_rising": 3, "unloaded_lowering": 3}
WINDOW_STEPS_1S = None  # set per cell dt

def runs_of(mask):
    """Maximal runs of True: list of (start_idx, end_idx) inclusive, 0-based."""
    if not mask.any():
        return []
    padded = np.concatenate(([False], mask, [False]))
    d = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(d == 1); ends = np.flatnonzero(d == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))

def events_for(mask, code, body, required):
    out = []
    for s, e in runs_of(mask):
        length = e - s + 1
        out.append({"code": code, "body": body, "start": s + 1, "end": e + 1,
                    "confirmed": (s + 1) + required - 1 if length >= required else None})
    return out

def first_confirmed(mask, required):
    """First 1-based step at which the running streak reaches `required`."""
    streak = 0
    for i, v in enumerate(mask):
        streak = streak + 1 if v else 0
        if streak >= required:
            return i + 1
    return None

stats = defaultdict(int)
mismatch_examples = []
codes_checked = defaultdict(int)
per_episode_diff = 0
total_events = 0
for run_index in sorted(runs):
    chunks = sorted(runs[run_index], key=lambda c: c[0])
    term_order = chunks[0][1]
    cat = lambda key: np.concatenate([c[2][key] for c in chunks], axis=0)
    base_pos, quat = cat("base_pos_w").astype(np.float64), cat("base_quat_w").astype(np.float64)
    spos, svel, sforce = cat("support_pos_w").astype(np.float64), cat("support_lin_vel_w").astype(np.float64), cat("support_contact_forces_w").astype(np.float64)
    shist, fhist = cat("support_contact_forces_w_history").astype(np.float64), cat("forbidden_contact_forces_w_history").astype(np.float64)
    ep_len, valid = cat("episode_length"), cat("valid")
    term = {name: cat(f"termination_{i}") for i, name in enumerate(term_order)}
    origins = chunks[0][2]["terrain_env_origins_w"].astype(np.float64)
    n_env = valid.shape[1]
    for env in range(n_env):
        v = np.flatnonzero(valid[:, env])
        if not len(v):
            continue
        cell = cells[env]
        dt = cell.spec.control_dt_s
        steps = ep_len[v, env]
        assert np.array_equal(steps, np.arange(1, len(v) + 1)), ("episode steps not 1..N", run_index, env, steps[:5])
        T = len(v)
        q = quat[v, env]; w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
        origin = origins[env]
        progress = base_pos[v, env, 0] - origin[0]
        pos = spos[v, env]; vel = svel[v, env]; force = sforce[v, env]
        fnorm = np.linalg.norm(force, axis=-1)
        contact = fnorm > CONTACT_N
        speed = np.hypot(vel[:, :, 0], vel[:, :, 1])
        vz = vel[:, :, 2]
        clearance = np.empty_like(speed)
        for j in range(len(names)):
            for i in range(T):
                surf = queries[env].sample(progress_m=pos[i, j, 0] - origin[0], lateral_m=pos[i, j, 1] - origin[1])
                clearance[i, j] = pos[i, j, 2] - origin[2] - surf.height_m
        hist_norm = np.linalg.norm(shist[v, env], axis=-1)  # [T, 3, body]
        contact_hist = hist_norm.max(axis=1) > CONTACT_N
        forb_hist = (np.linalg.norm(fhist[v, env], axis=-1) > FORB_N).any(axis=(1, 2))
        forb_mgr = term["base_contact"][v, env]
        slip = contact & (speed > SLIP_V)
        low = (~contact) & (clearance < CLR)
        tilt = np.maximum(np.abs(roll), np.abs(pitch)) > TILT
        win = int(math.ceil(0.5 / dt))
        low_progress = np.zeros(T, dtype=bool)
        for i in range(win, T):
            seg = progress[i - win:i + 1]
            low_progress[i] = (seg.max() - seg[0]) < 0.02
        masks = {("tilt", "root"): tilt, ("slip_any", "any_support"): slip.any(axis=1),
                 ("low_clearance_any", "any_support"): low.any(axis=1), ("low_progress", "root"): low_progress,
                 ("forbidden_manager", "forbidden_role"): forb_mgr, ("no_instant_support", "all_support"): ~contact.any(axis=1),
                 ("forbidden_history", "forbidden_role"): forb_hist}
        for j, body in enumerate(names):
            masks[("contact_instant", body)] = contact[:, j]
            masks[("slip_body", body)] = slip[:, j]
            masks[("low_clearance_body", body)] = low[:, j]
            masks[("unloaded_rising", body)] = (~contact[:, j]) & (vz[:, j] > 0.05)
            masks[("unloaded_lowering", body)] = (~contact[:, j]) & (vz[:, j] < -0.05)
            masks[("contact_history", body)] = contact_hist[:, j]
        events = []
        for (code, body), mask in masks.items():
            events.extend(events_for(mask, code, body, REQUIRED.get(code, 1)))
        events.sort(key=lambda e: (e["start"], e["body"], e["code"]))
        terminal = T
        window_first = max(1, terminal - int(math.ceil(1.0 / dt)) + 1)
        before, window_present = {}, {}
        for code, body in (("tilt", "root"), ("slip_any", "any_support"), ("low_clearance_any", "any_support"), ("low_progress", "root")):
            fc = first_confirmed(masks[(code, body)], REQUIRED[code])
            before[code] = fc is not None and fc < terminal
            window_present[code] = any(e["code"] == code and e["start"] < terminal and e["end"] >= window_first for e in events)
        crossed = np.flatnonzero(progress >= cell.start_platform_length_m / 2)
        if not len(crossed):
            anchor = {"status": "not_reached", "step": None}
        elif crossed[0] == 0:
            anchor = {"status": "already_beyond_first_sample", "step": None}
        else:
            anchor = {"status": "crossed", "step": int(crossed[0] + 1)}
        r6 = lambda a: round(float(a), 6)
        ranges = {"roll_rad": [r6(roll.min()), r6(roll.max())], "pitch_rad": [r6(pitch.min()), r6(pitch.max())],
                  "progress_m": [r6(progress.min()), r6(progress.max())]}
        extrema = [[body, r6(clearance[:, j].min()), r6(speed[:, j].max()), r6(np.abs(vz[:, j]).max())] for j, body in enumerate(names)]
        mine = {"steps": T, "events": events, "before_terminal": before, "window_present": window_present,
                "anchor": anchor, "ranges": ranges, "body_extrema": extrema, "cell": cell.cell_id, "stratum": cell.spec.stratum_id,
                "coverage": {"first_step": 1, "last_step": T, "terminal_included": True, "window_first_step": window_first}}
        theirs = by_id[f"r{run_index}.e{env}"]
        total_events += len(events)
        diff_here = False
        for key in mine:
            if key == "events":
                a = [json.dumps(e, sort_keys=True) for e in mine["events"]]
                b = [json.dumps(e, sort_keys=True) for e in theirs["events"]]
                if a != b:
                    diff_here = True
                    sa, sb = set(a), set(b)
                    for item in sorted(sa ^ sb)[:3]:
                        mismatch_examples.append({"episode": theirs["episode_id"], "only_in": "mine" if item in sa else "theirs", "event": item})
                    stats["events_mismatch_episodes"] += 1
                    stats["events_only_mine"] += len(sa - sb); stats["events_only_theirs"] += len(sb - sa)
                else:
                    for e in mine["events"]:
                        codes_checked[e["code"]] += 1
            elif key in ("ranges", "body_extrema"):
                flat_a = np.array([x for row in (mine[key].values() if key == "ranges" else [r[1:] for r in mine[key]]) for x in row], dtype=float)
                flat_b = np.array([x for row in (theirs[key].values() if key == "ranges" else [r[1:] for r in theirs[key]]) for x in row], dtype=float)
                if flat_a.shape != flat_b.shape or np.abs(flat_a - flat_b).max() > 2e-6:
                    diff_here = True; stats[f"{key}_mismatch"] += 1
                    if len(mismatch_examples) < 40:
                        mismatch_examples.append({"episode": theirs["episode_id"], "key": key, "mine": mine[key], "theirs": theirs[key]})
            elif mine[key] != theirs[key]:
                diff_here = True; stats[f"{key}_mismatch"] += 1
                if len(mismatch_examples) < 40:
                    mismatch_examples.append({"episode": theirs["episode_id"], "key": key, "mine": mine[key], "theirs": theirs[key]})
        # terminal cross-checks: forbidden episodes end with the manager term active
        if theirs["primary_reason"] == "forbidden_contact" and not forb_mgr[-1]:
            stats["forbidden_without_manager_at_terminal"] += 1
        if forb_mgr[-1] and not forb_hist[-1]:
            stats["manager_terminal_without_history_predicate"] += 1
        if forb_mgr[-1] and theirs["primary_reason"] != "forbidden_contact":
            stats["manager_terminal_but_other_reason"] += 1
        per_episode_diff += diff_here
        stats["episodes_checked"] += 1
    print(f"run {run_index} done, elapsed {round(time.time() - t0, 1)} s, episodes {stats['episodes_checked']}, differing {per_episode_diff}", flush=True)

result = {"episodes_checked": stats["episodes_checked"], "episodes_with_any_difference": per_episode_diff,
          "total_events_mine": total_events, "events_checked_by_code": dict(codes_checked), "stats": dict(stats),
          "mismatch_examples": mismatch_examples[:40], "elapsed_s": round(time.time() - t0, 1),
          "thresholds": {"tilt": [TILT, TILT_N], "slip": [SLIP_V, SLIP_N, CONTACT_N], "clearance": [CLR, CLR_N], "forbidden_N": FORB_N}}
json.dump(result, open(OUT, "w"), indent=1)
print(json.dumps({k: v for k, v in result.items() if k != "mismatch_examples"}, indent=1))
