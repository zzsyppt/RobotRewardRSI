"""Read-only NumPy audit of the fixed Go2 1x/1.5x/2x force diagnostic."""
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
CHECKPOINT = "sha256:1495390b192b913432b8472cb150e2db07fdc2eb811eeabe4ab87c577b24e0ff"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def verify(condition, context):
    if not condition:
        raise ValueError(context)


def canonical(value, omitted):
    payload = {key: item for key, item in value.items() if key != omitted}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def rotation(q):
    x, y, z, w = np.moveaxis(np.asarray(q, dtype=float), -1, 0)
    return np.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                     2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                     2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), axis=-1).reshape(q.shape[:-1]+(3, 3))


def first_stable_block(steps, stable, end, window, dwell):
    selected = (steps > end) & (steps <= end + window)
    values, indices = stable[selected], steps[selected]
    if len(values) < dwell:
        return None
    blocks = np.convolve(values.astype(int), np.ones(dwell, dtype=int), "valid")
    hits = np.flatnonzero(blocks == dwell)
    return int(indices[hits[0]+dwell-1]) if len(hits) else None


def audit(directory, scale, baseline):
    definition, p, identity, saved_summary = (read(directory / name) for name in
        ("definition.json", "parameters.json", "source_identity.json", "summary.json"))
    verify(definition.get("force_scale", 1.0) == scale, "declared force scale")
    verify(canonical(definition, "definition_hash") == definition["definition_hash"], "definition hash")
    verify(canonical(p, "resolved_protocol_hash") == p["resolved_protocol_hash"], "parameters hash")
    verify(p["protocol_hash"] == identity["definition_hash"] == definition["definition_hash"], "diagnostic identity")
    verify(identity["checkpoint_sha256"] == CHECKPOINT, "fixed D checkpoint")
    verify(definition["derived_seeds"] == [10042, 10043, 10044], "fixed public-dev seeds")
    verify(all(definition[key] == 0 for key in ("train_jobs", "holdout_jobs", "provider_calls")), "eval-only declaration")
    base_p, base_d = baseline["parameters"], baseline["definition"]
    for key, value in base_p.items():
        if key not in ("protocol_hash", "resolved_protocol_hash", "schema_version"):
            verify(p[key] == value, "unchanged physical/recovery parameter: " + key)
    for key in ("source_manifest_hash", "source_training_binding_hash", "source_train_summary_sha256", "source_environment_hash"):
        verify(identity[key] == baseline["identity"][key], "unchanged source: " + key)
    for key in ("body_name", "positions", "is_global", "direct_torque_nm", "base_seed", "envs_per_seed", "seed_count",
                "control_dt_s", "physics_dt_s", "pulse_control_steps", "pulse_duration_s", "start_control_steps",
                "window_steps", "dwell_steps", "first_recovery_sample", "recovery_scope"):
        verify(definition[key] == base_d[key], "unchanged diagnostic setting: " + key)
    dt, phy_dt = definition["control_dt_s"], definition["physics_dt_s"]
    decimation = round(dt / phy_dt)
    pulse, starts = math.ceil(p["time_scale_s"] / dt), [math.ceil(t / dt) for t in p["push_times_s"]]
    window, dwell = math.ceil(p["recovery_window_s"] / dt), math.ceil(p["stable_dwell_s"] / dt)
    horizon = math.ceil(p["episode_length_s"] / dt)
    verify((pulse, starts, window, dwell) == (definition["pulse_control_steps"], definition["start_control_steps"],
           definition["window_steps"], definition["dwell_steps"]), "quantized schedule")
    verify(abs(decimation * phy_dt - dt) < 1e-12 and abs(pulse * dt - definition["pulse_duration_s"]) < 1e-12, "duration")
    rows, seeds, worst, raw_hashes = [], [], {}, {}
    counts, maximum, ranges = Counter(), Counter(), {key: [] for key in ("force_n", "impulse_ns", "mass_kg")}
    for index, seed in enumerate(definition["derived_seeds"], 1):
        meta = read(directory / f"seed_{index:02d}.json")
        raw_path = directory / f"seed_{index:02d}.npz"
        raw_hashes[raw_path.name] = sha(raw_path)
        verify(raw_hashes[raw_path.name] == meta["raw_sha256"], f"{seed}: raw hash")
        verify(meta["seed"] == seed and meta["definition_hash"] == definition["definition_hash"], f"{seed}: identity")
        verify(meta["body_names"][meta["force_body_index"]] == definition["body_name"], f"{seed}: force body")
        with np.load(raw_path, allow_pickle=False) as loaded:
            a = {key: loaded[key] for key in loaded.files}
        verify(all(np.isfinite(value).all() for value in a.values()), f"{seed}: finite raw data")
        valid, before, physical_valid = a["valid"], a["physics_control_step_before"], a["physics_valid"]
        verify(valid.shape[1] == 64 and len(before) == len(valid) * decimation, f"{seed}: physical substeps")
        verify(np.array_equal(physical_valid, np.repeat(valid, decimation, axis=0)), f"{seed}: live masks")
        verify(np.array_equal((before+1)[physical_valid], np.repeat(a["steps"], decimation, axis=0)[physical_valid]), f"{seed}: substep clock")
        for key, mask in (("quaternion_xyzw", valid), ("physics_body_quaternion_xyzw", physical_valid)):
            error = float(np.abs(np.linalg.norm(a[key][mask].astype(float), axis=-1)-1).max())
            maximum[key + "_norm_error"] = max(maximum[key + "_norm_error"], error)
            verify(error < 5e-4, f"{seed}: {key} unit quaternion")
        mass = a["body_masses_kg"].astype(float).sum(axis=1)
        verify((mass > 0).all(), f"{seed}: positive total masses")
        ranges["mass_kg"].extend(mass.tolist())
        phase = random.Random(seed).random() * 2 * math.pi
        angles = phase + (np.arange(64) % 16) * (2 * math.pi / 16)
        angles = np.stack((angles, angles + math.pi/2), axis=1)
        direction = np.stack((np.cos(angles), np.sin(angles), np.zeros_like(angles)), axis=-1)
        magnitude = np.array([p["velocity_magnitudes_mps"]["medium" if s == "repeated" else s] for s in STRATA for _ in range(16)])
        expected = direction * (mass * magnitude * scale / (pulse*dt))[:, None, None]
        verify(np.allclose(a["expected_force_w"], expected, atol=2e-3, rtol=3e-6), f"{seed}: force scale, mass and direction")
        scheduled = np.full(before.shape, -1, dtype=int)
        requested = np.zeros_like(a["physics_requested_force_w"], dtype=float)
        for event, start in enumerate(starts):
            mask = physical_valid & ((np.arange(64) >= 48) if event else True) & (before >= start) & (before < start+pulse)
            scheduled[mask] = event
            requested[mask] = np.broadcast_to(expected[:, event], requested.shape)[mask]
        verify(np.array_equal(scheduled, a["physics_event_index"]), f"{seed}: scheduled pulses")
        delivered = np.einsum("...ij,...j->...i", rotation(a["physics_body_quaternion_xyzw"]), a["physics_delivered_force_b"].astype(float))
        transform_error = float(np.linalg.norm(delivered-a["physics_requested_force_w"].astype(float), axis=-1).max())
        verify(transform_error < 1e-3, f"{seed}: original delivered/requested transform tolerance")
        maximum["delivered_requested_error_n"] = max(maximum["delivered_requested_error_n"], transform_error)
        force_error = float(np.linalg.norm(delivered-requested, axis=-1).max())
        maximum["force_error_n"] = max(maximum["force_error_n"], force_error)
        verify(force_error < 2e-3, f"{seed}: delivered force versus independent expectation")
        verify(np.allclose(a["physics_requested_force_w"], requested, atol=2e-3, rtol=3e-6), f"{seed}: requested force")
        maximum["outside_pulse_force_n"] = max(maximum["outside_pulse_force_n"], float(np.abs(delivered[scheduled < 0]).max()))
        verify(maximum["outside_pulse_force_n"] < 1e-5, f"{seed}: force clears outside pulse")
        verify(np.abs(a["physics_delivered_torque_b"]).max() < 1e-5 and np.abs(a["physics_other_body_force_max"]).max() < 1e-5, f"{seed}: no extra torque/body force")
        verify(len(meta["episodes"]) == 64, f"{seed}: denominator")
        seed_rows = []
        for env, saved in enumerate(meta["episodes"]):
            context = f"seed={seed} env={env}"
            mask = valid[:, env]; steps = a["steps"][mask, env]; n = len(steps)
            verify(n > 0 and np.array_equal(mask, np.arange(len(mask)) < n), context + ": first-episode prefix")
            verify(np.array_equal(steps, np.arange(1, n+1)), context + ": contiguous steps")
            r = rotation(a["quaternion_xyzw"][mask, env])
            roll, pitch = np.arctan2(r[:, 2, 1], r[:, 2, 2]), np.arcsin(np.clip(-r[:, 2, 0], -1, 1))
            height = a["position"][mask, env, 2].astype(float) - meta["origins"][env][2]
            drift = np.linalg.norm(a["position"][mask, env, :2].astype(float)-meta["initial_xy"][env], axis=1)
            speed = np.linalg.norm(a["linear"][mask, env, :2].astype(float), axis=1)
            omega = np.linalg.norm(a["angular"][mask, env].astype(float), axis=1)
            failures = (pitch > p["fall_tilt_rad"], pitch < -p["fall_tilt_rad"], abs(roll) > p["fall_tilt_rad"],
                        height < p["fall_height_m"], drift > p["max_displacement_m"])
            labels = ("fall_forward", "fall_backward", "fall_side", "height_collapse", "out_of_bounds")
            failed = np.logical_or.reduce(failures)
            failure = next((label for test, label in zip(failures, labels) if test.any()), None)
            verify(not failed[:-1].any() and (failed[-1] or n == horizon), context + ": first failure/timeout closure")
            stable = (speed <= p["recovery_velocity_mps"]) & (omega <= p["recovery_angular_velocity_radps"]) & (abs(roll) <= p["recovery_tilt_rad"]) & (abs(pitch) <= p["recovery_tilt_rad"]) & (height >= p["recovery_height_m"])
            planned, events, recoveries = (2 if env >= 48 else 1), [], []
            for event in range(planned):
                active = scheduled[:, env] == event; count = int(active.sum())
                if not count:
                    continue
                end = int(before[active, env][-1])+1
                verify(np.array_equal(before[active, env], np.repeat(np.arange(starts[event], end), decimation)), context + ": exact pulse substeps")
                complete = count == pulse*decimation
                verify(count <= pulse*decimation and (complete or failure is not None), context + ": partial pulse requires failure")
                impulse = delivered[active, env].sum(axis=0)*phy_dt
                error = float(np.linalg.norm(impulse-expected[env, event]*count*phy_dt))
                maximum["impulse_error_ns"] = max(maximum["impulse_error_ns"], error)
                verify(error < 5e-4, context + ": integrated impulse")
                ranges["force_n"].append(float(np.linalg.norm(expected[env, event])))
                ranges["impulse_ns"].append(float(np.linalg.norm(impulse)))
                rec = first_stable_block(steps, stable, end, window, dwell) if complete else None
                recoveries.append(rec); events.append((count, end, complete, impulse))
                counts["observed_pulses"] += 1; counts["completed_pulses"] += int(complete)
                counts["partial_pulses"] += int(not complete); counts["recovered_pulses"] += int(rec is not None)
            verify(len(events) == planned or failure is not None, context + ": missing pulse requires failure")
            counts["missing_pulses_after_failure"] += planned-len(events)
            success = failure is None and len(events) == planned and all(v is not None for v in recoveries) and bool(stable[-1])
            label = failure or ("success_recovered" if success else "not_recovered")
            verify(saved["env_id"] == env and saved["stratum"] == STRATA[env//16] and saved["episode_steps"] == n, context + ": saved episode identity")
            verify(saved["success"] == success and saved["failure_label"] == label and saved["event_recovery_control_steps"] == recoveries, context + ": independently reconstructed outcome")
            verify(saved["invalid_injection_count"] == 0 and len(saved["force_events"]) == len(events), context + ": saved event count")
            for old, (count, end, complete, impulse) in zip(saved["force_events"], events):
                verify(old["actual_substeps"] == count and old["end_control_step"] == end and old["completed"] == complete and old["injection_verified"], context + ": saved pulse metadata")
                verify(np.allclose(old["delivered_impulse_world_ns"], impulse, atol=5e-4, rtol=3e-6), context + ": saved impulse")
            own = {"seed": seed, "env_id": env, "stratum": STRATA[env//16], "success": success, "failure_label": label}
            if success:
                metrics = {"max_abs_roll_rad": abs(roll), "max_abs_pitch_rad": abs(pitch), "min_height_m": height,
                           "max_planar_speed_mps": speed, "max_angular_speed_radps": omega, "max_xy_drift_m": drift}
                for name, values in metrics.items():
                    at = int(np.argmin(values) if name == "min_height_m" else np.argmax(values)); value = float(values[at])
                    previous = worst.get(name)
                    if previous is None or (value < previous["value"] if name == "min_height_m" else value > previous["value"]):
                        worst[name] = {"value": value, "seed": seed, "env_id": env, "control_step": int(steps[at])}
            seed_rows.append(own)
        seeds.append({"seed": seed, "episodes": 64, "successes": sum(row["success"] for row in seed_rows)})
        rows.extend(seed_rows)
    summary = {"episodes": len(rows), "successes": sum(row["success"] for row in rows), "seeds": seeds,
               "failure_counts": dict(Counter(row["failure_label"] for row in rows)),
               "strata": {s: {"episodes": 48, "successes": sum(row["success"] for row in rows if row["stratum"] == s)} for s in STRATA}}
    for key, value in summary.items():
        verify(saved_summary[key] == value, "saved summary: " + key)
    report = {"force_scale": scale, "evidence_directory": str(directory.resolve()), **summary,
              "pulse_counts": dict(counts), "maximum_errors": dict(maximum), "successful_episode_worst_samples": worst,
              "ranges": {key: {"min": min(values), "max": max(values)} if values else None for key, values in ranges.items()},
              "raw_sha256": raw_hashes}
    return report, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=P6 / "force_01/go2")
    parser.add_argument("--evidence-root", type=Path, default=P6 / "force_strength_01")
    parser.add_argument("--baseline-only", action="store_true")
    args = parser.parse_args()
    baseline = {name: read(args.baseline / filename) for name, filename in
                (("parameters", "parameters.json"), ("definition", "definition.json"), ("identity", "source_identity.json"))}
    results = []
    try:
        first, base_rows = audit(args.baseline, 1.0, baseline)
        results.append(first)
        if not args.baseline_only:
            for scale, name in ((1.5, "scale_1p5"), (2.0, "scale_2p0")):
                result, rows = audit(args.evidence_root / name, scale, baseline)
                result["paired_with_1x"] = dict(Counter(f"{old['failure_label']} -> {new['failure_label']}"
                    for old, new in zip(base_rows, rows, strict=True)))
                results.append(result)
        report = {"status": "PASS", "method": "Independent NumPy matrices and continuous-window convolution; no runtime/scorer imports.",
                  "arms": results, "audit_script_sha256": sha(Path(__file__)),
                  "limits": ["Saved PhysX force-call inputs establish delivered external impulse, not net momentum change.",
                             "Root states are control-step samples; this is not a visual or continuous-substep posture proof.",
                             "First-episode masks use the recorder's state.closed; first failure/timeout is reconstructed from raw samples.",
                             "Fixed historical checkpoint and public seeds; no qualification or new-training claim."]}
        print(json.dumps(report, ensure_ascii=False, indent=2)); return 0
    except (ValueError, KeyError, FileNotFoundError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc), "completed_arms": results}, ensure_ascii=False, indent=2)); return 1


if __name__ == "__main__":
    raise SystemExit(main())
