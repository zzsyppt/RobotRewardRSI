"""Read-only arithmetic audit of saved force-evaluation summaries for this run.

This checks consistency of exported records, not an independent reconstruction
of the simulator trajectory or the wrench actually received by PhysX.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import struct


def _float32(value: float) -> float:
    """Match Torch's scalar cast when comparing float32 runtime tensors."""
    return struct.unpack("f", struct.pack("f", value))[0]


def audit(session: Path) -> dict:
    reports = []
    for path in sorted(session.rglob("summary.json")):
        raw = path.read_bytes()
        data = json.loads(raw)
        seeds = data.get("task_evidence", {}).get("seed_summaries", [])
        if not seeds:
            continue
        failures, counts = [], Counter()
        maxima = {"force_error_n": 0.0, "impulse_residual_ns": 0.0}

        def check(condition, detail):
            counts["checks"] += 1
            if not condition:
                failures.append(detail)

        check(data.get("candidate_code_loaded_in_eval") is False, "candidate code loaded in evaluation")
        for seed_index, seed in enumerate(seeds):
            p = seed["configuration"]["push_resolved_evaluation_protocol"]
            dt = seed["environment"]["step_dt"]
            pulse = math.ceil(p["pulse_duration_s"] / dt)
            gap = math.ceil(p["repeated_gap_s"] / dt)
            first = math.ceil(p["push_times_s"][0] / dt)
            starts = (first, first + pulse + gap)
            check(p["max_displacement_m"] is None, "displacement failure must be disabled")
            check(p["repeated_high_fraction"] == 0.8 and p["repeated_gap_s"] == 0.22,
                  "frozen repeated-force semantics changed")
            labels, strata = Counter(), {}
            for row in seed["per_episode"]:
                prefix = f"seed[{seed_index}].env[{row['env_id']}]"
                events, planned = row["force_events"], row["planned_push_count"]
                fall = next((label for condition, label in (
                    (row["max_pitch_rad"] > p["fall_tilt_rad"], "fall_forward"),
                    (row["min_pitch_rad"] < -p["fall_tilt_rad"], "fall_backward"),
                    (row["max_abs_roll_rad"] > p["fall_tilt_rad"], "fall_side"),
                    (row["min_base_height_m"] < p["fall_height_m"], "height_collapse"),
                ) if condition), None)
                expected = fall or ("invalid_push_injection" if row["invalid_injection_count"] else
                    "no_push_control" if row["control_only"] else
                    "success_recovered" if row["sequence_recovered"] and row["final_recovery_condition"] else "not_recovered")
                check(row["failure_label"] == expected, prefix + ": classification")
                check(row["success"] == (expected == "success_recovered"), prefix + ": success flag")
                check(row["invalid_injection_count"] == 0, prefix + ": invalid injection")
                check(planned == (2 if row["stratum"] == "repeated" else 1), prefix + ": cohort retained")
                check([event["event_index"] for event in events] ==
                      [index for index in range(planned) if row["episode_steps"] > starts[index]],
                      prefix + ": events present before episode closure")
                if len(events) < planned or any(not event["completed"] for event in events):
                    counts["episodes_with_incomplete_pulses"] += 1
                    check(fall is not None, prefix + ": incomplete pulse without a recorded fall")
                for event in events:
                    counts["force_events"] += 1
                    index = event["event_index"]
                    check(event["start_control_step"] == starts[index], prefix + ": event schedule")
                    expected_end = min(starts[index] + pulse, row["episode_steps"])
                    check(event["end_control_step"] == expected_end, prefix + ": event end before episode closure")
                    check(event["expected_substeps"] % pulse == 0 and
                          event["actual_substeps"] * pulse ==
                          (expected_end - starts[index]) * event["expected_substeps"],
                          prefix + ": substep count matches delivered control steps")
                    completed = (event["actual_substeps"] == event["expected_substeps"]
                                 and event["start_control_step"] == starts[index]
                                 and event["end_control_step"] == starts[index] + pulse)
                    check(event["completed"] == completed, prefix + ": completed flag")
                    check(event["injection_verified"] and event["max_force_error_n"] <= _float32(p["force_tolerance_n"]),
                          prefix + ": recorded force verification")
                    maxima["force_error_n"] = max(maxima["force_error_n"], event["max_force_error_n"])
                    magnitude = (p["impulse_velocity_mps"]["high"] * p["repeated_high_fraction"]
                                 if row["stratum"] == "repeated" else p["impulse_velocity_mps"][row["stratum"]])
                    expected_force = row["actual_total_mass_kg"] * magnitude / (pulse * dt)
                    check(math.isclose(math.dist(event["force_world_n"], (0, 0, 0)), expected_force, rel_tol=2e-6),
                          prefix + ": mass-scaled force amplitude")
                    duration = pulse * dt * event["actual_substeps"] / event["expected_substeps"]
                    maxima["impulse_residual_ns"] = max(maxima["impulse_residual_ns"],
                        max(abs(impulse - force * duration) for impulse, force in
                            zip(event["delivered_impulse_world_ns"], event["force_world_n"])))
                    # Exported accumulation is float32; allow its rounding in
                    # addition to the per-substep force tolerance integrated in time.
                    check(all(abs(impulse - force * duration) <= p["force_tolerance_n"] * duration + 1e-4
                              for impulse, force in zip(event["delivered_impulse_world_ns"], event["force_world_n"])),
                          prefix + ": delivered impulse")
                check(row["sequence_recovered"] == (row["sequence_recovery_control_step"] is not None),
                      prefix + ": recovery flag")
                if row["sequence_recovered"]:
                    last_end = starts[planned - 1] + pulse
                    recovered_at = row["sequence_recovery_control_step"]
                    check(len(events) == planned and all(event["completed"] for event in events), prefix + ": full sequence")
                    check(last_end + math.ceil(p["stable_dwell_s"] / dt) <= recovered_at <=
                          min(last_end + math.ceil(p["recovery_window_s"] / dt), row["episode_steps"]),
                          prefix + ": fresh recovery window")
                if row["success"]:
                    check(row["final_planar_speed_mps"] <= _float32(p["recovery_velocity_mps"]), prefix + ": final speed")
                    counts["successful_episodes"] += 1
                labels[row["failure_label"]] += 1
                stratum = strata.setdefault(row["stratum"], Counter())
                stratum["episodes"] += 1
                stratum["successes"] += row["success"]
                counts["episodes"] += 1
            aggregate = seed["aggregate"]
            check(dict(labels) == aggregate["failure_breakdown"]["counts"], f"seed[{seed_index}]: failure counts")
            check(labels["success_recovered"] == aggregate["successes"], f"seed[{seed_index}]: successes")
            for name, values in strata.items():
                check(all(aggregate["stratum_results"][name][key] == value for key, value in values.items()),
                      f"seed[{seed_index}]: stratum {name}")
        reports.append({"path": path.relative_to(session).as_posix(), "sha256": hashlib.sha256(raw).hexdigest(),
                        "counts": dict(counts), "observed_maxima": maxima, "failures": failures})
    return {"status": "PASS" if reports and not any(row["failures"] for row in reports) else "FAIL",
            "scope": "Saved episode-summary arithmetic consistency only; PASS is not independent physical or visual validation.",
            "limitations": [
                "Pose extrema and wrench verification are runtime exports, not independently read sensor/PhysX data.",
                "No raw-frame reconstruction of continuous stable dwell, or final angular velocity, roll/pitch and height.",
                "Pulse timing/count checks use the recorded control dt and expected substep count; physics dt is not independently reconstructed.",
                "Fall labels use exported extrema and the same strict threshold ordering as episode_records; this is not a new outcome definition.",
            ],
            "numerical_tolerances": {"force_amplitude_relative": 2e-6,
                "impulse_absolute_rounding_ns": 1e-4,
                "impulse_total_rule": "force_tolerance_n * delivered_duration_s + float32 accumulation allowance"},
            "summary_count": len(reports), "summaries": reports}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("session", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = audit(args.session)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(result["status"], result["summary_count"])
    raise SystemExit(result["status"] != "PASS")
