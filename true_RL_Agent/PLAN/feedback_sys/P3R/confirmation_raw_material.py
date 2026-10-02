"""Independent confirmation material from raw archives; no production imports.

Selects preregistered episodes, exposes complete measured curves, and computes
direct quantities for reviewers. These quantities are not semantic PASS labels.
Never reads a behavior report or an author answer. Development dry-runs must use
--purpose development; only an independently frozen new cohort is confirmation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np

STAIR_SCOPES = ["c1_conditional_frontier", "c2_mechanism_ood", "height_ood",
                "tread_ood", "joint_ood", "phase_composition_ood"]
PUSH_SCOPES = ["low", "medium", "high", "repeated", "medium", "repeated"]
CONDITIONS = ["planar_speed", "angular_speed", "roll", "pitch", "height"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def spans(mask, steps, minimum=1):
    mask = np.asarray(mask, dtype=bool)
    starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
    ends = np.flatnonzero(mask & ~np.r_[mask[1:], False])
    return [[int(steps[a]), int(steps[b])] for a, b in zip(starts, ends) if b - a + 1 >= minimum]


def extrema(values, steps):
    lo, hi = int(np.argmin(values)), int(np.argmax(values))
    return {"min": float(values[lo]), "min_step": int(steps[lo]),
            "max": float(values[hi]), "max_step": int(steps[hi])}


def sample_indexes(n):
    return sorted({int(round((n - 1) * f)) for f in (0, .25, .5, .75, 1)})


def chunks(path, manifest, keys, run=None):
    result = defaultdict(list)
    static = {}
    for c in manifest["chunks"]:
        if run is not None and c["metadata"]["run_index"] != run:
            continue
        f = path.parent / c["file"]
        if sha(f) != c["sha256"]:
            raise ValueError(f"raw chunk checksum: {f}")
        with np.load(f, allow_pickle=False) as a:
            for key in keys:
                result[key].append(a[key].copy())
            for key in ("terrain_env_origins_w", "terrain_types"):
                if key in a:
                    static[key] = a[key].copy()
    return {k: np.concatenate(v) for k, v in result.items()}, static


def stair_rows(path, rules_path):
    m = read(path)
    if not m["complete"]:
        raise ValueError("incomplete archive")
    bodies = m["metadata"]["runtime_projection"]["resolved_roles"]["support_effector"]
    sig = m["metadata"]["semantics"]["episode_outcomes"]["signals"]
    contact_n = sig["slip"]["contact_force_threshold"]
    rules = read(rules_path)["semantics"]["thresholds"]
    rows, curves = [], {}
    keys = ["valid", "episode_length", "base_pos_w", "base_quat_w",
            "support_lin_vel_w", "support_contact_forces_w", "termination_0", "termination_1"]
    for seed in m["metadata"]["seeds"]:
        run = seed["run_index"]
        a, static = chunks(path, m, keys, run)
        for env, terminal in enumerate(seed["outcomes"]):
            valid = a["valid"][:, env]
            steps = a["episode_length"][valid, env]
            if not np.array_equal(steps, np.arange(1, len(steps) + 1)):
                raise ValueError("non-contiguous measured episode")
            pos = a["base_pos_w"][valid, env].astype(float) - static["terrain_env_origins_w"][env]
            w, x, y, z = a["base_quat_w"][valid, env].astype(float).T
            roll = np.arctan2(2 * (w*x + y*z), 1 - 2 * (x*x + y*y))
            pitch = np.arcsin(np.clip(2 * (w*y - z*x), -1, 1))
            up_z = 1 - 2 * (x*x + y*y)
            force = np.linalg.norm(a["support_contact_forces_w"][valid, env].astype(float), axis=-1)
            speed = np.linalg.norm(a["support_lin_vel_w"][valid, env].astype(float), axis=-1)
            drawdown = np.maximum.accumulate(pos[:, 0]) - pos[:, 0]
            trough = int(np.argmax(drawdown))
            peak = int(np.argmax(pos[:trough + 1, 0]))
            tilt = np.maximum(abs(roll), abs(pitch)) > sig["base_orientation_instability"]["threshold"]
            low_window = math.ceil(rules["low_progress_window_s"] / seed["step_dt"])
            low_progress = np.zeros(len(steps), dtype=bool)
            for i in range(low_window, len(steps)):
                low_progress[i] = max(pos[i-low_window:i+1, 0]) - pos[i-low_window, 0] < rules["low_progress_gain_m"]
            slow_min = math.ceil(rules["unloaded_low_motion_s"] / seed["step_dt"])
            slow = (force <= contact_n) & (speed <= rules["unloaded_low_speed_mps"])
            ident = f"r{run}.e{env}"
            row = {"id": ident, "run": run, "env": env, "task": "stair",
                   "scope": terminal["bin_id"].split(".", 1)[1].rsplit("_level_", 1)[0],
                   "cell": terminal["bin_id"], "success": terminal["success"],
                   "outcome": terminal["primary_reason"], "steps": len(steps), "dt": seed["step_dt"],
                   "measured_net_x_m": float(pos[-1, 0] - pos[0, 0]),
                   "largest_x_retreat": {"m": float(drawdown[trough]), "from_step": int(steps[peak]), "to_step": int(steps[trough])},
                   "roll_rad": extrema(roll, steps), "pitch_rad": extrema(pitch, steps), "up_axis_z": extrema(up_z, steps),
                   "tilt_exceedance_intervals": spans(tilt, steps),
                   "sustained_tilt_intervals": spans(tilt, steps, sig["base_orientation_instability"]["consecutive_control_steps"]),
                   "low_progress_intervals": spans(low_progress, steps),
                   "same_support_unloaded_low_motion": {
                       name: [{"start_step": start, "end_step": end,
                               "confirmed_step": start + slow_min - 1,
                               "max_3d_point_speed_mps": float(speed[start-1:end, i].max()),
                               "root_x_gain_m": float(pos[end-1, 0] - pos[start-1, 0]),
                               "overlap_low_progress_steps": int(low_progress[start-1:end].sum())}
                              for start, end in spans(slow[:, i], steps, slow_min)]
                       for i, name in enumerate(bodies)},
                   "instant_all_support_unloaded_intervals": spans((force <= contact_n).all(axis=1), steps),
                   "support_unloaded_intervals": {name: spans(force[:, i] <= contact_n, steps) for i, name in enumerate(bodies)},
                   "quarter_samples": [{"step": int(steps[i]), "root_xyz_env_m": pos[i].tolist(),
                                        "roll_pitch_upz": [float(roll[i]), float(pitch[i]), float(up_z[i])]} for i in sample_indexes(len(steps))]}
            # Plot the actual one-second x change; this is not a gait/freeze label.
            lag = max(1, int(round(1 / seed["step_dt"])))
            dx = np.full(len(steps), np.nan)
            dx[lag:] = pos[lag:, 0] - pos[:-lag, 0]
            curves[ident] = {"steps": steps, "position": pos, "roll": roll, "pitch": pitch,
                            "up_z": up_z, "force": force, "speed": speed, "dx_1s": dx}
            rows.append(row)
    return rows, curves, {"identity": m["identity"], "support_names": bodies, "numeric_rule_definition_sha256": sha(rules_path),
                          "numeric_rule_thresholds": rules,
                          "semantics": m["metadata"]["semantics"], "coordinates": m["metadata"]["coordinates"]}


def push_rows(paths):
    rows, curves, sources = [], {}, []
    keys = ["valid", "control_step", "pose", "planar_motion", "conditions"]
    for path in paths:
        m = read(path)
        if not m["complete"]:
            raise ValueError("incomplete archive")
        meta = m["metadata"]
        if meta["parameters"]["mechanism"] != "world_planar_velocity_increment":
            raise ValueError("this confirmation material is restricted to the frozen velocity profile")
        a, _ = chunks(path, m, keys)
        p = meta["parameters"]
        run = int(m["identity"]["run_index"])
        sources.append({"identity": m["identity"], "parameters": p, "time": meta["time"], "coordinates": meta["coordinates"]})
        for env, terminal in enumerate(meta["runtime_episode_records"]):
            valid = a["valid"][:, env]
            steps, pose, motion = [a[k][valid, env] for k in ("control_step", "pose", "planar_motion")]
            if not np.array_equal(steps, np.arange(1, len(steps) + 1)):
                raise ValueError("non-contiguous measured episode")
            bound = lambda k: np.asarray(p[k], dtype=pose.dtype)
            good = np.column_stack([pose[:, 3] <= bound("recovery_velocity_mps"),
                                    pose[:, 4] <= bound("recovery_angular_velocity_radps"),
                                    abs(pose[:, 1]) <= bound("recovery_tilt_rad"),
                                    abs(pose[:, 2]) <= bound("recovery_tilt_rad"),
                                    pose[:, 0] >= bound("recovery_height_m")])
            if not np.array_equal(good, a["conditions"][valid, env]):
                raise ValueError("recorded condition differs from raw pose and protocol")
            events = {e["event_index"]: e for e in terminal["push_events"]}
            event_rows = []
            for event in range(terminal["planned_push_count"]):
                e = events.get(event)
                if e is None:
                    event_rows.append({"event": event, "status": "not_executed", "planned_step": meta["time"]["event_scheduled_steps"][event]})
                    continue
                anchor = e["control_step"]
                inds = np.flatnonzero((steps > anchor) & (steps <= anchor + meta["time"]["recovery_window_steps"]))
                stable = good[inds].all(axis=1)
                runs = spans(stable, steps[inds]) if len(inds) else []
                qualifying = [s for s in runs if s[1] - s[0] + 1 >= meta["time"]["required_dwell_steps"]]
                latch = qualifying[0][0] + meta["time"]["required_dwell_steps"] - 1 if qualifying and e["injection_verified"] else None
                if latch != e["recovery_control_step"]:
                    raise ValueError("raw dwell differs from original runtime receipt")
                next_event_steps = [x["control_step"] for k, x in events.items() if k > event]
                next_injection = min(next_event_steps) if next_event_steps else None
                whole_after = (steps > latch) & ~good.all(axis=1) if latch is not None else np.zeros(len(steps), dtype=bool)
                own_after = whole_after & (steps <= next_injection) if next_injection is not None else whole_after
                q = {"event": event, "status": "executed", "anchor_step": anchor, "verified": e["injection_verified"],
                     "observed_delta_v_xy": e["observed_delta_v_world_mps"], "first_latch_step": latch,
                     "stable_intervals": runs,
                     "violation_intervals": {name: spans(~good[inds, i], steps[inds]) for i, name in enumerate(CONDITIONS)},
                     "next_actual_injection_step": next_injection,
                     "per_event_after_latch_through_pre_injection_anchor": spans(own_after, steps),
                     "whole_episode_after_latch_including_later_events": spans(whole_after, steps),
                     "window_complete": bool(len(inds) and steps[inds[-1]] == anchor + meta["time"]["recovery_window_steps"])}
                if len(inds):
                    first, last = int(inds[0]), int(inds[-1])
                    peak = int(inds[np.argmax(pose[inds, 3])])
                    pre = int(np.flatnonzero(steps == anchor)[0])
                    q.update(first_step=int(steps[first]), peak_step=int(steps[peak]), last_step=int(steps[last]),
                             first_peak_last_speed_mps=[float(pose[i, 3]) for i in (first, peak, last)],
                             anchor_to_window_end_xy_m=(motion[last, 2:].astype(float)-motion[pre, 2:]).tolist(),
                             window_pose_extrema={name: extrema(pose[inds, i], steps[inds]) for i, name in enumerate(meta["pose_fields"])})
                event_rows.append(q)
            ident = f"r{run}.e{env}"
            rows.append({"id": ident, "run": run, "env": env, "task": "push", "scope": terminal["stratum"],
                         "cell": terminal["stratum"], "success": terminal["success"], "outcome": terminal["failure_label"],
                         "steps": len(steps), "dt": meta["time"]["step_dt_s"], "events": event_rows,
                         "terminal_failed_conditions": [name for i, name in enumerate(CONDITIONS) if not good[-1, i]],
                         "whole_episode_pose_extrema": {name: extrema(pose[:, i], steps) for i, name in enumerate(meta["pose_fields"])},
                         "quarter_samples": [{"step": int(steps[i]), "pose": pose[i].tolist(), "motion": motion[i].tolist()} for i in sample_indexes(len(steps))]})
            curves[ident] = {"steps": steps, "pose": pose, "motion": motion, "good": good}
    return rows, curves, {"sources": sources, "pose_fields": meta["pose_fields"], "condition_fields": CONDITIONS,
                         "post_latch_scopes": "per-event interval includes the next actual injection anchor frame (observed before the write), excludes its first post-injection frame and later steps; whole-episode interval explicitly includes later injections, not spontaneous relapse evidence"}


def rank(row):
    key = f"p3r-confirm-v1|{row['task']}|{row['run']}|{row['env']}"
    return hashlib.sha256(key.encode()).hexdigest(), row["id"]


def select(rows, task):
    picked, used = [], set()
    for scope in STAIR_SCOPES if task == "stair" else PUSH_SCOPES:
        group = [r for r in rows if r["scope"] == scope and r["id"] not in used]
        failures = [r for r in group if not r["success"]]
        counts = Counter(r["outcome"] for r in failures)
        reason = sorted(counts, key=lambda k: (-counts[k], k))[0] if counts else None
        pool = [r for r in failures if r["outcome"] == reason] if failures else group
        first = min(pool, key=rank)
        used.add(first["id"])
        successes = [r for r in group if r["success"] and r["id"] not in used]
        if not successes:
            raise ValueError(f"no unused success counterexample for {scope}")
        match = lambda r: (0 if r["cell"] == first["cell"] and r["run"] == first["run"] else 1 if r["cell"] == first["cell"] else 2, rank(r))
        second = min(successes, key=match)
        used.add(second["id"])
        picked.extend([{**first, "selection_role": "primary_outcome_example"},
                       {**second, "selection_role": "success_counterexample", "matched_to": first["id"], "match_level": match(second)[0]}])
    if len(used) != 12:
        raise ValueError("confirmation requires 12 distinct episodes per task")
    return picked


def plots(output, task, rows, curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for row in rows:
        a = curves[row["id"]]
        t = a["steps"] * row["dt"]
        fig, axes = plt.subplots(5, 1, figsize=(13, 12), sharex=True)
        if task == "stair":
            for i, label in enumerate("xyz"):
                axes[0].plot(t, a["position"][:, i], label=label)
            axes[0].set_ylabel("root relative to env (m)")
            for name in ("roll", "pitch", "up_z"):
                axes[1].plot(t, a[name], label=name)
            axes[1].set_ylabel("rad / unitless up_z")
            axes[2].plot(t, a["dx_1s"], label="measured x change over 1s")
            for i in range(a["force"].shape[1]):
                axes[3].plot(t, a["force"][:, i], label=f"bound foot {i}")
                axes[4].plot(t, a["speed"][:, i], label=f"bound foot {i}")
            axes[3].set_ylabel("instant net force (N)")
            axes[4].set_ylabel("body point 3D speed (m/s)")
        else:
            for i, label in [(3, "planar speed"), (4, "angular speed")]:
                axes[0].plot(t, a["pose"][:, i], label=label)
            for i, label in [(1, "roll"), (2, "pitch")]:
                axes[1].plot(t, a["pose"][:, i], label=label)
            axes[2].plot(t, a["pose"][:, 0], label="height above env (m)")
            for i, label in [(2, "reset relative x"), (3, "reset relative y")]:
                axes[3].plot(t, a["motion"][:, i], label=label)
            for i, name in enumerate(CONDITIONS):
                axes[4].plot(t, (~a["good"][:, i]).astype(float) + 1.3*i, label=name)
            for event in row["events"]:
                if event["status"] == "executed":
                    for ax in axes:
                        ax.axvline(event["anchor_step"] * row["dt"], color="gray", linestyle="--", linewidth=.8)
        for ax in axes:
            ax.grid(alpha=.2)
            ax.legend(loc="upper left", fontsize=8, ncol=3)
        axes[-1].set_xlabel("control time (s)")
        fig.suptitle(f"{task} {row['id']} / {row['scope']} / original outcome {row['outcome']}")
        fig.tight_layout()
        fig.savefig(output / f"{row['id']}.png", dpi=120)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["stair", "push"], required=True)
    ap.add_argument("--manifest", type=Path, nargs="+", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--purpose", choices=["development", "confirmation"], required=True)
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--stair-rules", type=Path)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if args.task == "stair" and len(args.manifest) != 1:
        raise ValueError("one Stair archive expected")
    if args.task == "stair" and args.stair_rules is None:
        raise ValueError("explicit frozen numeric definitions required for Stair")
    rows, curves, definitions = stair_rows(args.manifest[0], args.stair_rules) if args.task == "stair" else push_rows(args.manifest)
    chosen = select(rows, args.task)
    write(args.output / "raw_pool.json", {"purpose": args.purpose, "truth_status": "DIRECT_NUMERIC_MATERIAL_NOT_SEMANTIC_VERDICT",
        "reader_sha256": sha(__file__), "sources": [{"path": str(p), "sha256": sha(p)} for p in args.manifest],
        "definitions": definitions, "episodes": rows})
    write(args.output / "selected_episodes.json", {"selection": "CONFIRMATION_PROTOCOL_ZH.md", "purpose": args.purpose, "episodes": chosen})
    for row in chosen:
        np.savez_compressed(args.output / f"{row['id']}.npz", **curves[row["id"]])
    if args.plots:
        plots(args.output, args.task, chosen, curves)
    print(json.dumps({"purpose": args.purpose, "task": args.task, "raw_episodes": len(rows), "selected": [r["id"] for r in chosen], "semantic_gate": "NOT_GRADED"}))


if __name__ == "__main__":
    main()
