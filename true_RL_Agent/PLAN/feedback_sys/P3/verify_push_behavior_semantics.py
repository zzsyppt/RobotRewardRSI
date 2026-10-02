"""Small independent CPU arithmetic checks of preregistered P2 Push examples.

Reads hash-verified numeric chunks, not the production behavior/recovery reducer.
These purposive examples validate semantics, not independent statistical cases.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

CASES = (
    ("push_g1_force_v1", 1, 1, "failure_with_dwell_interruptions"),
    ("push_g1_force_v1", 1, 2, "success_with_dwell_interruptions"),
    ("push_go2_velocity_v1", 1, 46, "zero_stable_window_without_dwell_interruption"),
    ("push_g1_velocity_v1", 1, 1, "all_latched_but_final_unstable"),
    ("push_g1_force_v2", 1, 32, "terminal_before_sequence_anchor"),
    ("push_g1_force_v2", 1, 1, "signed_direction_with_completed_force_sequence"),
    ("push_go2_velocity_v2", 1, 0, "signed_direction_with_velocity_event"),
)
NAMES = ("planar_speed", "angular_speed", "roll", "pitch", "height")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check_case(name, run, env, category, *, inventory, exports):
    from true_rl_agent.runner.numeric_evidence import iter_numeric_evidence, load_numeric_evidence_manifest
    phase_path = Path(inventory[name]["run_root"]) / "public_dev_result.json"
    phase = read(phase_path)
    reference = next(summary["numeric_evidence_ref"] for summary in phase["task_evidence"]["seed_summaries"]
                     if summary["run"]["run_index"] == run)
    manifest = load_numeric_evidence_manifest(reference)
    meta, identity = manifest["metadata"], manifest["identity"]
    full_path = exports / name / "full.json"
    full = read(full_path)
    assert full["binding"]["checkpoint_hash"] == identity["checkpoint_hash"] == inventory[name]["expected_checkpoint_hash"]
    source = next(index for index, source in enumerate(full["sources"]) if source["manifest_sha256"] == reference["manifest_sha256"])
    row = next(row for row in full["transcripts"] if row["source"] == source and row["env_id"] == env)
    chunks = [arrays for _, arrays in iter_numeric_evidence(reference)]
    values = {key: np.concatenate([a[key][:, env] for a in chunks]) for key in chunks[0]}
    valid = values["valid"]
    steps, pose = values["control_step"][valid], values["pose"][valid]
    parameters, timing = meta["parameters"], meta["time"]
    threshold = lambda key: np.asarray(parameters[key], dtype=pose.dtype)
    predicates = np.column_stack((pose[:, 3] <= threshold("recovery_velocity_mps"),
        pose[:, 4] <= threshold("recovery_angular_velocity_radps"),
        abs(pose[:, 1]) <= threshold("recovery_tilt_rad"), abs(pose[:, 2]) <= threshold("recovery_tilt_rad"),
        pose[:, 0] >= threshold("recovery_height_m")))
    assert np.array_equal(predicates, values["conditions"][valid])
    assert row["valid_steps"] == len(steps) and np.array_equal(steps, np.arange(1, len(steps) + 1))
    original = meta["runtime_episode_records"][env]
    assert row["terminal_label"] == original["failure_label"]
    force = "force" in meta
    recovered, units = [], []
    for unit in row["recoveries"]:
        index = len(original["force_events"]) - 1 if force else int(unit["identity"].split("_")[1])
        if force:
            events = original["force_events"]
            completed = len(events) == original["planned_push_count"] and all(e["completed"] for e in events)
            anchor = events[-1]["end_control_step"] if completed else None
            verified = completed and all(e["injection_verified"] for e in events) and not bool(values["unplanned_wrench"][valid][-1])
        else:
            event = next((event for event in original["push_events"] if event["event_index"] == index), None)
            anchor = event["control_step"] if event else None
            verified = event is not None and event["injection_verified"]
        assert unit["anchor_control_step"] == anchor
        mask = (steps > anchor) & (steps <= anchor + timing["recovery_window_steps"]) if anchor is not None and (not force or verified) else np.zeros(len(steps), dtype=bool)
        local_steps, stable = steps[mask], predicates[mask].all(axis=1)
        dwell, maximum, latch, interruptions = 0, 0, None, []
        for step, good, conditions in zip(local_steps, stable, predicates[mask]):
            if latch is not None:
                continue
            if not good and dwell:
                interruptions.append((int(step), dwell, [key for key, passed in zip(NAMES, conditions) if not passed]))
            dwell = dwell + 1 if good else 0
            maximum = max(maximum, dwell)
            if dwell >= timing["required_dwell_steps"] and verified:
                latch = int(step)
        assert unit["observed_window_steps"] == len(local_steps)
        assert unit["stable_window_steps"] == int(stable.sum()) and unit["max_dwell_steps"] == maximum
        assert unit["recovery_control_step"] == latch
        assert [(e["control_step"], e["preceding_dwell_steps"], e["failed_conditions"]) for e in unit["dwell_interruptions"]] == interruptions
        for event in unit["dwell_interruptions"]:
            assert event["after_anchor_steps"] == event["control_step"] - anchor
            assert np.isclose(event["after_anchor_s"], (event["control_step"] - anchor) * timing["step_dt_s"], rtol=0, atol=1e-12)
        for column, key in enumerate(NAMES):
            observed = local_steps[~predicates[mask, column]].tolist()
            reconstructed = [step for span in unit["condition_violation_spans"][key]
                             for step in range(span["start_step"], span["end_step_inclusive"] + 1)]
            assert observed == reconstructed and len(observed) == unit["failed_condition_steps"][key]
            for span in unit["condition_violation_spans"][key]:
                assert span["observed_steps"] == span["end_step_inclusive"] - span["start_step"] + 1
                for time_key, step_key in (("start_after_anchor_s", "start_step"), ("end_after_anchor_s", "end_step_inclusive")):
                    assert np.isclose(span[time_key], (span[step_key] - anchor) * timing["step_dt_s"], rtol=0, atol=1e-12)
        direction = unit["direction"]
        if "planar_motion" not in values:
            assert direction == {"status": "not_collected_in_v1"}
        elif direction["status"] == "available":
            event = original["force_events" if force else "push_events"][index]
            vector = np.asarray(event["delivered_impulse_world_ns" if force else "observed_delta_v_world_mps"][:2])
            along = vector / np.linalg.norm(vector)
            sideways = np.array([-along[1], along[0]])
            motion = values["planar_motion"][valid][mask][-1]
            assert np.array_equal(direction["velocity_world_xy_mps"], motion[:2])
            assert np.array_equal(direction["reset_displacement_world_xy_m"], motion[2:])
            for key, expected in (("velocity_parallel_mps", motion[:2] @ along), ("velocity_lateral_mps", motion[:2] @ sideways),
                                  ("reset_displacement_parallel_m", motion[2:] @ along), ("reset_displacement_lateral_m", motion[2:] @ sideways)):
                assert np.isclose(direction[key], expected, rtol=0, atol=1e-12)
        recovered.append(latch)
        units.append({"identity": unit["identity"], "anchor": anchor, "observed_steps": len(local_steps),
            "stable_steps": int(stable.sum()), "max_dwell": maximum, "latch": latch, "interruptions": interruptions,
            "coverage": unit["coverage"], "direction_status": direction["status"]})
    all_latched = bool(recovered) and all(step is not None for step in recovered)
    after = (steps > max(recovered)) & ~predicates.all(axis=1) if all_latched else np.zeros(len(steps), dtype=bool)
    assert row["all_required_recoveries_latched"] == all_latched
    assert row["unstable_steps_after_all_recoveries"] == int(after.sum())
    final_failed = [key for key, passed in zip(NAMES, predicates[-1]) if not passed]
    assert row["final_failed_conditions"] == final_failed
    if category in {"failure_with_dwell_interruptions", "success_with_dwell_interruptions"}:
        assert any(unit["interruptions"] for unit in units)
        assert row["terminal_label"] == ("success_recovered" if category.startswith("success") else "not_recovered")
    if category == "zero_stable_window_without_dwell_interruption":
        assert row["terminal_label"] == "not_recovered" and any(unit["stable_steps"] == 0 and not unit["interruptions"] for unit in units)
    if category == "all_latched_but_final_unstable":
        assert all_latched and final_failed and row["terminal_label"] == "not_recovered"
    if category == "terminal_before_sequence_anchor":
        assert units[0]["anchor"] is None and units[0]["observed_steps"] == 0
    if category.startswith("signed_direction"):
        assert any(unit["direction_status"] == "available" for unit in units)
    return {"dataset": name, "run_ordinal": run, "env_id": env, "category": category, "status": "PASS",
        "manifest_sha256": reference["manifest_sha256"], "full_sha256": hashlib.sha256(full_path.read_bytes()).hexdigest(),
        "label": row["terminal_label"], "valid_steps": len(steps), "final_failed_conditions": final_failed,
        "unstable_steps_after_all_recoveries": int(after.sum()), "units": units}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--exports", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.project_root / "src"))
    results = [check_case(*case, inventory=read(args.inventory), exports=args.exports) for case in CASES]
    report = {"status": "PASS", "scope": "seven fixed examples; independent arithmetic, not new physics or population inference", "results": results}
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "cases": len(results)}))


if __name__ == "__main__":
    main()
