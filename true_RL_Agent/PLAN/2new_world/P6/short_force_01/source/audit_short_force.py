"""Read-only independent NumPy audit: fixed Go2 2x force, long versus short gap.

No product runtime or diagnostic scorer imports. JSON is printed to stdout;
the input evidence is never written. --old-only validates the existing arm and
also re-scores its raw trajectory using the last-force sequence rule.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
P6 = ROOT / "PLAN/2new_world/P6"
STRATA = ("low", "medium", "high", "repeated")
SEEDS = [10042, 10043, 10044]
CHECKPOINT = "sha256:1495390b192b913432b8472cb150e2db07fdc2eb811eeabe4ab87c577b24e0ff"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def verify(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value, omitted):
    value = {key: item for key, item in value.items() if key != omitted}
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def rotation(q):
    x, y, z, w = np.moveaxis(np.asarray(q, dtype=float), -1, 0)
    return np.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                     2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                     2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), axis=-1).reshape(q.shape[:-1]+(3, 3))


def first_dwell(steps, stable, end, window=81, dwell=16):
    selected = (steps > end) & (steps <= end + window)
    values, indices = stable[selected], steps[selected]
    if len(values) < dwell:
        return None
    hits = np.flatnonzero(np.convolve(values.astype(int), np.ones(dwell, dtype=int), "valid") == dwell)
    return int(indices[hits[0]+dwell-1]) if len(hits) else None


def summarize(rows):
    return {"episodes": len(rows), "successes": sum(row["success"] for row in rows),
        "seeds": [{"seed": seed, "episodes": 64, "successes": sum(row["success"] for row in rows if row["seed"] == seed)} for seed in SEEDS],
        "failure_counts": dict(Counter(row["failure_label"] for row in rows)),
        "strata": {s: {"episodes": 48, "successes": sum(row["success"] for row in rows if row["stratum"] == s)} for s in STRATA}}


def audit(directory, short, baseline):
    d, p, identity = (read(directory / name) for name in ("definition.json", "parameters.json", "source_identity.json"))
    starts, pulse, window, dwell = [51, 73 if short else 172], 11, 81, 16
    verify(canonical(d, "definition_hash") == d["definition_hash"], "definition hash")
    verify(canonical(p, "resolved_protocol_hash") == p["resolved_protocol_hash"], "parameters hash")
    verify(identity["definition_hash"] == p["protocol_hash"] == d["definition_hash"], "diagnostic identity")
    verify(identity["checkpoint_sha256"] == CHECKPOINT, "fixed Go2 D checkpoint")
    required = {"robot": "go2", "body_name": "base", "positions": None, "is_global": True,
        "direct_torque_nm": [0, 0, 0], "base_seed": 42, "derived_seeds": SEEDS,
        "envs_per_seed": 64, "seed_count": 3, "force_scale": 2.0,
        "control_dt_s": .02, "physics_dt_s": .005, "pulse_control_steps": pulse,
        "pulse_duration_s": .22, "start_control_steps": starts, "window_steps": window,
        "dwell_steps": dwell, "train_jobs": 0, "holdout_jobs": 0, "provider_calls": 0}
    required.update(schema_version="true_rl_agent.p6.external_force_diagnostic.v3" if short else "true_rl_agent.p6.external_force_diagnostic.v2",
        recovery_scope="all_planned_pulses_then_one_fresh_window_and_final_stable" if short else "every_completed_pulse_independent_window_then_final_stable")
    if short:
        required.update(repeat_gap_rule="ceil(time_scale_s / control_dt_s) * control_dt_s",
            repeat_gap_control_steps=11, repeat_gap_s=.22)
    for key, value in required.items():
        verify(d[key] == value, "frozen setting: " + key)
    for key in ("mechanism", "mass_rule", "force_rule", "source_resolved_protocol_hash", "first_recovery_sample"):
        verify(d[key] == baseline["definition"][key], "unchanged force/source setting: " + key)
    for key in ("source_manifest_hash", "source_training_binding_hash", "source_train_summary_sha256", "source_environment_hash", "source_dev_file_sha256"):
        verify(identity[key] == baseline["identity"][key], "old training/checkpoint source unchanged: " + key)
    for key, value in baseline["parameters"].items():
        if key not in ("push_times_s", "protocol_hash", "resolved_protocol_hash", "schema_version"):
            verify(p[key] == value, "unchanged physical/recovery parameter: " + key)
    verify([math.ceil(t/.02) for t in p["push_times_s"]] == starts, "parameter start quantization")
    verify((math.ceil(p["time_scale_s"]/.02), math.ceil(p["recovery_window_s"]/.02), math.ceil(p["stable_dwell_s"]/.02)) == (pulse, window, dwell), "physical time quantization")
    horizon = math.ceil(p["episode_length_s"]/.02)
    rows, legacy_rows, raw_hashes, snapshots, worst, tails = [], [], {}, {}, {}, []
    counts, maximum = Counter(), Counter()
    ranges = {key: [] for key in ("actual_force_n", "actual_impulse_ns", "mass_kg")}
    before_second = []
    for index, seed in enumerate(SEEDS, 1):
        meta, raw_path = read(directory / f"seed_{index:02d}.json"), directory / f"seed_{index:02d}.npz"
        raw_hashes[raw_path.name] = sha(raw_path)
        verify(raw_hashes[raw_path.name] == meta["raw_sha256"], f"{seed}: raw SHA")
        verify(meta["seed"] == seed and meta["definition_hash"] == d["definition_hash"], f"{seed}: identity")
        verify(meta["capture_point"] == "control_state_after_physics_before_termination_and_reset", f"{seed}: capture point")
        verify(meta["body_names"][meta["force_body_index"]] == "base", f"{seed}: force link")
        with np.load(raw_path, allow_pickle=False) as loaded:
            a = {key: loaded[key] for key in loaded.files}
        verify(all(np.isfinite(value).all() for value in a.values()), f"{seed}: finite raw")
        valid, before, physical_valid = a["valid"], a["physics_control_step_before"], a["physics_valid"]
        verify(valid.shape[1] == 64 and len(before) == len(valid)*4, f"{seed}: physical sample count")
        verify(np.array_equal(physical_valid, np.repeat(valid, 4, axis=0)), f"{seed}: live masks")
        verify(np.array_equal((before+1)[physical_valid], np.repeat(a["steps"], 4, axis=0)[physical_valid]), f"{seed}: substep clock")
        for key, mask in (("quaternion_xyzw", valid), ("physics_body_quaternion_xyzw", physical_valid)):
            error = float(np.abs(np.linalg.norm(a[key][mask].astype(float), axis=-1)-1).max())
            maximum[key+"_norm_error"] = max(maximum[key+"_norm_error"], error)
            verify(error < 5e-4, f"{seed}: unit quaternion {key}")
        mass = a["body_masses_kg"].astype(float).sum(axis=1)
        verify((mass > 0).all(), f"{seed}: actual total masses")
        ranges["mass_kg"].extend(mass.tolist())
        angle = random.Random(seed).random()*2*math.pi + np.arange(64)%16*(2*math.pi/16)
        angle = np.stack((angle, angle+math.pi/2), axis=1)
        direction = np.stack((np.cos(angle), np.sin(angle), np.zeros_like(angle)), axis=-1)
        magnitude = np.array([p["velocity_magnitudes_mps"]["medium" if s == "repeated" else s] for s in STRATA for _ in range(16)])
        expected = direction*(mass*magnitude*2/.22)[:, None, None]
        verify(np.allclose(a["expected_force_w"], expected, atol=2e-3, rtol=3e-6), f"{seed}: independent expected force")
        scheduled = np.full(before.shape, -1, dtype=int)
        requested = np.zeros_like(a["physics_requested_force_w"], dtype=float)
        for event, start in enumerate(starts):
            mask = physical_valid & ((np.arange(64) >= 48) if event else True) & (before >= start) & (before < start+pulse)
            scheduled[mask] = event
            requested[mask] = np.broadcast_to(expected[:, event], requested.shape)[mask]
        verify(np.array_equal(scheduled, a["physics_event_index"]), f"{seed}: exact planned physical pulse timing")
        delivered = np.einsum("...ij,...j->...i", rotation(a["physics_body_quaternion_xyzw"]), a["physics_delivered_force_b"].astype(float))
        for key, other, limit in (("delivered_requested_error_n", a["physics_requested_force_w"], 1e-3), ("independent_force_error_n", requested, 2e-3)):
            error = float(np.linalg.norm(delivered-other, axis=-1).max())
            maximum[key] = max(maximum[key], error)
            verify(error < limit, f"{seed}: {key}, including early-failure samples")
        verify(np.allclose(a["physics_requested_force_w"], requested, atol=2e-3, rtol=3e-6), f"{seed}: requested force")
        zero = float(np.abs(delivered[scheduled < 0]).max())
        maximum["outside_pulse_force_n"] = max(maximum["outside_pulse_force_n"], zero)
        verify(zero < 1e-5, f"{seed}: force clears in gap, after pulse, and closed episodes")
        verify(np.abs(a["physics_delivered_torque_b"]).max() < 1e-5 and np.abs(a["physics_other_body_force_max"]).max() < 1e-5, f"{seed}: no extra direct torque or other-link force")
        verify(len(meta["episodes"]) == 64, f"{seed}: all 64 episodes retained")
        for env, saved in enumerate(meta["episodes"]):
            context = f"{seed}/env{env}"
            mask = valid[:, env]; steps = a["steps"][mask, env]; n = len(steps)
            verify(n > 0 and np.array_equal(mask, np.arange(len(mask)) < n), context+": first-episode prefix")
            verify(np.array_equal(steps, np.arange(1, n+1)), context+": continuous first-episode steps")
            r = rotation(a["quaternion_xyzw"][mask, env])
            roll, pitch = np.arctan2(r[:, 2, 1], r[:, 2, 2]), np.arcsin(np.clip(-r[:, 2, 0], -1, 1))
            height = a["position"][mask, env, 2].astype(float)-meta["origins"][env][2]
            drift = np.linalg.norm(a["position"][mask, env, :2].astype(float)-meta["initial_xy"][env], axis=1)
            speed = np.linalg.norm(a["linear"][mask, env, :2].astype(float), axis=1)
            omega = np.linalg.norm(a["angular"][mask, env].astype(float), axis=1)
            masks = (pitch > p["fall_tilt_rad"], pitch < -p["fall_tilt_rad"], abs(roll) > p["fall_tilt_rad"], height < p["fall_height_m"], drift > p["max_displacement_m"])
            labels = ("fall_forward", "fall_backward", "fall_side", "height_collapse", "out_of_bounds")
            failed = np.logical_or.reduce(masks)
            failure = next((label for test, label in zip(masks, labels) if test[-1]), None)
            verify(not failed[:-1].any() and (failed[-1] or n == horizon), context+": independently reconstructed failure/timeout closure")
            stable = (abs(roll) <= p["recovery_tilt_rad"]) & (abs(pitch) <= p["recovery_tilt_rad"]) & (height >= p["recovery_height_m"]) & (speed <= p["recovery_velocity_mps"]) & (omega <= p["recovery_angular_velocity_radps"])
            planned, events, old_recoveries = (2 if env >= 48 else 1), [], []
            counts["planned_pulses"] += planned
            for event in range(planned):
                active = scheduled[:, env] == event; count = int(active.sum())
                if not count:
                    continue
                end = int(before[active, env][-1])+1
                verify(np.array_equal(before[active, env], np.repeat(np.arange(starts[event], end), 4)), context+": every physical substep")
                complete = count == 44
                verify(count <= 44 and (complete or failure is not None), context+": incomplete pulse only after physical failure")
                impulse = delivered[active, env].sum(axis=0)*.005
                error = float(np.linalg.norm(impulse-expected[env, event]*count*.005))
                maximum["impulse_error_ns"] = max(maximum["impulse_error_ns"], error)
                verify(error < 5e-4, context+": actual external impulse")
                ranges["actual_force_n"].extend(np.linalg.norm(delivered[active, env], axis=-1).tolist())
                ranges["actual_impulse_ns"].append(float(np.linalg.norm(impulse)))
                old_recoveries.append(first_dwell(steps, stable, end) if complete else None)
                events.append((event, count, end, complete, impulse))
                counts["observed_pulses"] += 1; counts["complete_pulses"] += int(complete); counts["partial_pulses"] += int(not complete)
            verify(len(events) == planned or failure is not None, context+": unobserved future pulse requires early failure")
            counts["unobserved_pulses_after_failure"] += planned-len(events)
            all_complete = len(events) == planned and all(event[3] for event in events)
            recovery = first_dwell(steps, stable, starts[planned-1]+pulse) if all_complete else None
            success = failure is None and recovery is not None and bool(stable[-1])
            label = failure or ("success_recovered" if success else "not_recovered")
            own = {"seed": seed, "env_id": env, "stratum": STRATA[env//16], "success": success, "failure_label": label,
                "sequence_recovered": recovery is not None, "sequence_recovery_control_step": recovery}
            legacy_success = failure is None and all_complete and all(v is not None for v in old_recoveries) and bool(stable[-1])
            legacy = {**own, "success": legacy_success, "failure_label": failure or ("success_recovered" if legacy_success else "not_recovered")}
            compare = own if short else legacy
            verify(saved["env_id"] == env and saved["stratum"] == own["stratum"] and saved["episode_steps"] == n and saved["planned_push_count"] == planned, context+": saved episode identity")
            verify(saved["success"] == compare["success"] and saved["failure_label"] == compare["failure_label"], context+": reconstructed classifier")
            if short:
                verify(saved["sequence_recovery_control_step"] == recovery and saved["sequence_recovered"] == (recovery is not None), context+": independently reconstructed sequence recovery")
                verify("event_recovery_control_steps" not in saved and "recovered_event_count" not in saved, context+": sequence not independent-event evidence")
            else:
                verify(saved["event_recovery_control_steps"] == old_recoveries, context+": original long-gap per-event recovery")
            verify(saved["invalid_injection_count"] == 0 and len(saved["force_events"]) == len(events), context+": no bad injection excused by failure")
            for recorded, (event, count, end, complete, impulse) in zip(saved["force_events"], events):
                verify(recorded["event_index"] == event and recorded["start_control_step"] == starts[event] and recorded["end_control_step"] == end and recorded["actual_substeps"] == count and recorded["expected_substeps"] == 44 and recorded["completed"] == complete and recorded["injection_verified"], context+": saved actual pulse metadata")
                verify(np.allclose(recorded["force_world_n"], expected[env, event], atol=2e-3, rtol=3e-6) and np.allclose(recorded["delivered_impulse_world_ns"], impulse, atol=5e-4, rtol=3e-6), context+": saved force/impulse")
            if planned == 2:
                preceding = np.flatnonzero(steps == starts[1])
                before_second.append({"seed": seed, "env_id": env, "reached_pre_second_sample": bool(len(preceding)),
                    "stable_immediately_before_second": bool(stable[preceding[0]]) if len(preceding) else None,
                    "first_force_dwell_completed_before_second": first_dwell(steps, stable, starts[0]+pulse, min(window, starts[1]-starts[0]-pulse)) is not None})
            if success:
                unstable = np.flatnonzero(~stable); tail = len(stable) if not len(unstable) else len(stable)-1-int(unstable[-1])
                tails.append({"steps": tail, "duration_s": tail*.02, "seed": seed, "env_id": env})
                for name, values in {"max_abs_roll_rad": abs(roll), "max_abs_pitch_rad": abs(pitch), "min_height_m": height, "max_planar_speed_mps": speed, "max_angular_speed_radps": omega, "max_xy_drift_m": drift}.items():
                    at = int(np.argmin(values) if name == "min_height_m" else np.argmax(values)); value = float(values[at]); old = worst.get(name)
                    if old is None or (value < old["value"] if name == "min_height_m" else value > old["value"]):
                        worst[name] = {"value": value, "seed": seed, "env_id": env, "control_step": int(steps[at])}
            rows.append(own); legacy_rows.append(legacy)
        snapshots[seed] = ({key: a[key] for key in ("valid", "steps", "position", "quaternion_xyzw", "linear", "angular", "body_masses_kg")}, meta)
    target = read(directory / "summary.json")
    for key, value in summarize(rows if short else legacy_rows).items():
        verify(target[key] == value, "saved summary: " + key)
    verify(target["invalid_injections"] == 0, "summary has no invalid injection")
    reached = [row for row in before_second if row["reached_pre_second_sample"]]
    report = {"evidence_directory": str(directory.resolve()), "scoring": "all_planned_verified_complete_then_last_force_fresh_window_and_final_stable",
        "timing": {"scheduled_starts": starts, "repeated_last_force_end": starts[1]+pulse,
            "repeated_recovery_samples_inclusive": [starts[1]+pulse+1, starts[1]+pulse+window],
            "episode_horizon_steps": horizon, "repeated_post_force_observation_steps": horizon-starts[1]-pulse},
        "sequence_summary": summarize(rows), "original_saved_summary": summarize(rows if short else legacy_rows),
        "repeated_seed_results": [{"seed": seed, "episodes": 16,
            "successes": sum(row["success"] for row in rows if row["seed"] == seed and row["stratum"] == "repeated")} for seed in SEEDS],
        "window_recovered_episodes": {s: sum(row["sequence_recovered"] for row in rows if row["stratum"] == s) for s in STRATA},
        "old_rule_to_sequence_flips": [row for row, old in zip(rows, legacy_rows) if row["success"] != old["success"]] if not short else [],
        "pulse_counts": dict(counts), "maximum_errors": dict(maximum), "raw_sha256": raw_hashes,
        "ranges": {key: {"min": min(values), "max": max(values)} if values else None for key, values in ranges.items()},
        "successful_worst_samples": worst, "successful_terminal_stable_tail_min": min(tails, key=lambda row: row["steps"]) if tails else None,
        "before_second": {"planned_episodes": 48, "reached_pre_second_sample": len(reached),
            "stable_immediately_before_second": sum(row["stable_immediately_before_second"] for row in reached),
            "first_force_dwell_completed_before_second": sum(row["first_force_dwell_completed_before_second"] for row in before_second),
            "denominator_note": "Stable count uses reached episodes; dwell count uses all 48 planned repeated episodes. Early failures remain in outcome denominator."}}
    return report, rows, snapshots


def compare_trajectories(old, new):
    result = []
    for seed in SEEDS:
        a, ma = old[seed]; b, mb = new[seed]
        row = {"seed": seed, "initial_xy_equal": ma["initial_xy"] == mb["initial_xy"], "origins_equal": ma["origins"] == mb["origins"], "body_masses_equal": np.array_equal(a["body_masses_kg"], b["body_masses_kg"])}
        for name, env_ids, cutoff in (("non_repeated", range(48), None), ("before_first_force", range(64), 51)):
            exact, differences = 0, Counter()
            for env in env_ids:
                am, bm = a["valid"][:, env], b["valid"][:, env]
                if cutoff is not None:
                    am = am & (a["steps"][:, env] <= cutoff); bm = bm & (b["steps"][:, env] <= cutoff)
                same = np.array_equal(a["steps"][am, env], b["steps"][bm, env])
                for key in ("position", "quaternion_xyzw", "linear", "angular"):
                    av, bv = a[key][am, env], b[key][bm, env]; common = min(len(av), len(bv))
                    same = same and np.array_equal(av, bv)
                    if common:
                        differences[key] = max(differences[key], float(np.abs(av[:common].astype(float)-bv[:common].astype(float)).max()))
                exact += int(same)
            row[name] = {"episodes": len(env_ids), "bitwise_equal_raw_episodes": exact, "max_abs_difference_on_common_steps": dict(differences)}
        result.append(row)
    return {"seeds": result, "interpretation": "Descriptive only: same seeds do not erase prior reset/RNG/parallel-physics history. No bitwise-equality admission gate."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, default=P6 / "force_strength_01/scale_2p0")
    parser.add_argument("--short", type=Path, default=P6 / "short_force_01/short")
    parser.add_argument("--old-only", action="store_true")
    args = parser.parse_args(); reports = []
    try:
        baseline = {"parameters": read(args.old / "parameters.json"), "identity": read(args.old / "source_identity.json"), "definition": read(args.old / "definition.json")}
        report, old_rows, old_raw = audit(args.old, False, baseline); reports.append(report)
        result = {"status": "PASS", "arms": reports, "audit_script_sha256": sha(Path(__file__)),
            "limits": ["No training, holdout, new checkpoint identity, or qualification claim.",
                "Wrench API arguments verify applied external impulse, not net robot momentum change.",
                "Root posture is sampled each control step; no visual or substep posture proof.",
                "Recorded first-episode masks are checked against independent raw failure/timeout closure."]}
        if not args.old_only:
            report, rows, new_raw = audit(args.short, True, baseline); reports.append(report)
            result["paired_sequence_outcomes"] = dict(Counter(f"{old['failure_label']} -> {new['failure_label']}" for old, new in zip(old_rows, rows, strict=True)))
            result["paired_repeated_sequence_outcomes"] = dict(Counter(f"{old['failure_label']} -> {new['failure_label']}" for old, new in zip(old_rows, rows, strict=True) if old["stratum"] == "repeated"))
            result["paired_flips"] = [{"seed": old["seed"], "env_id": old["env_id"], "stratum": old["stratum"], "old": old["failure_label"], "short": new["failure_label"]} for old, new in zip(old_rows, rows, strict=True) if old["failure_label"] != new["failure_label"]]
            result["trajectory_comparison"] = compare_trajectories(old_raw, new_raw)
        print(json.dumps(result, ensure_ascii=False, indent=2)); return 0
    except (ValueError, KeyError, FileNotFoundError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc), "completed_arms": reports}, ensure_ascii=False, indent=2)); return 1


if __name__ == "__main__":
    raise SystemExit(main())
