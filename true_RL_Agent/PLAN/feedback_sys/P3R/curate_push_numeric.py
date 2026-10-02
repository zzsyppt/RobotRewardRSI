"""R0-b fixture preparation; independent NumPy arithmetic, never a runtime rule.

Run --extract on the archive host (stdout only); --assemble locally checks the
frozen full evidence and writes the small curated development fixture.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np

CONDITIONS = ["planar_speed", "angular_speed", "roll", "pitch", "height"]
REPRESENTATIVES = {"medium": [(0, 19), (0, 23)], "repeated": [(0, 55), (0, 57)]}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sample(steps, pose, motion, i):
    return {"control_step": int(steps[i]), "planar_speed_mps": float(pose[i, 3]),
        "angular_speed_radps": float(pose[i, 4]), "height_m": float(pose[i, 0]),
        "roll_rad": float(pose[i, 1]), "pitch_rad": float(pose[i, 2]),
        "velocity_world_xy_mps": motion[i, :2].tolist(),
        "reset_relative_position_world_xy_m": motion[i, 2:].tolist()}


def extract(phase_path):
    phase = read(phase_path)
    rows, sources = [], []
    for summary in phase["task_evidence"]["seed_summaries"]:
        ref = summary["numeric_evidence_ref"]
        path = Path(ref["manifest_path"])
        assert digest(path) == ref["manifest_sha256"]
        manifest = read(path)
        assert manifest["complete"]
        meta = manifest["metadata"]
        assert meta["schema_version"].endswith("v2") and "force" not in meta
        chunks = []
        for chunk in manifest["chunks"]:
            binary = (path.parent / chunk["file"]).read_bytes()
            assert hashlib.sha256(binary).hexdigest() == chunk["sha256"]
            with np.load(io.BytesIO(binary), allow_pickle=False) as arrays:
                chunks.append({key: arrays[key] for key in ("valid", "control_step", "pose", "conditions", "planar_motion")})
        arrays = {key: np.concatenate([chunk[key] for chunk in chunks]) for key in chunks[0]}
        source = int(manifest["identity"]["run_index"]) - 1
        sources.append({"source": source, "reference": ref, "identity": manifest["identity"],
            "coordinates": meta["coordinates"], "time": meta["time"],
            "parameters": {k: meta["parameters"][k] for k in (
                "recovery_velocity_mps", "recovery_angular_velocity_radps", "recovery_tilt_rad", "recovery_height_m")}})
        for env, original in enumerate(meta["runtime_episode_records"]):
            valid = arrays["valid"][:, env]
            steps, pose, motion, recorded_conditions = [arrays[key][valid, env] for key in (
                "control_step", "pose", "planar_motion", "conditions")]
            p = meta["parameters"]
            threshold = lambda key: np.asarray(p[key], dtype=pose.dtype)
            conditions = np.column_stack((pose[:, 3] <= threshold("recovery_velocity_mps"),
                pose[:, 4] <= threshold("recovery_angular_velocity_radps"),
                abs(pose[:, 1]) <= threshold("recovery_tilt_rad"),
                abs(pose[:, 2]) <= threshold("recovery_tilt_rad"),
                pose[:, 0] >= threshold("recovery_height_m")))
            assert np.array_equal(conditions, recorded_conditions)
            assert np.array_equal(steps, np.arange(1, len(steps) + 1))
            np.testing.assert_allclose(np.linalg.norm(motion[:, :2], axis=1), pose[:, 3], rtol=1e-6, atol=1e-6)
            np.testing.assert_allclose(np.linalg.norm(motion[:, 2:], axis=1), pose[:, 5], rtol=1e-6, atol=1e-6)
            row = {"source": source, "env_id": env, "stratum": original["stratum"],
                "terminal_label": original["failure_label"], "valid_steps": len(steps),
                "final_failed_conditions": [k for k, passed in zip(CONDITIONS, conditions[-1]) if not passed],
                "first": sample(steps, pose, motion, 0), "terminal": sample(steps, pose, motion, -1),
                "whole_episode_abs_roll_pitch_max_rad": [float(abs(pose[:, i]).max()) for i in (1, 2)],
                "recoveries": []}
            events = {e["event_index"]: e for e in original["push_events"]}
            for index in range(original["planned_push_count"]):
                event = events.get(index)
                unit = {"identity": f"event_{index}", "event_index": index,
                    "anchor_control_step": event["control_step"] if event else None,
                    "anchor_verified": bool(event and event["injection_verified"])}
                if event is None:
                    unit.update(observed_window_steps=0, stable_window_steps=0, max_dwell_steps=0,
                        recovery_control_step=None, failed_condition_steps=dict.fromkeys(CONDITIONS, 0),
                        dwell_interruptions=[], status="event_not_executed_before_terminal")
                    row["recoveries"].append(unit)
                    continue
                anchor = event["control_step"]
                indexes = np.flatnonzero((steps > anchor) & (steps <= anchor + meta["time"]["recovery_window_steps"]))
                local = conditions[indexes]
                stable = local.all(axis=1)
                dwell = maximum = 0
                latch = None
                interruptions = []
                for i, good in zip(indexes, stable):
                    if latch is not None:
                        continue
                    if not good and dwell:
                        interruptions.append({"control_step": int(steps[i]), "preceding_dwell_steps": dwell,
                            "failed_conditions": [k for k, passed in zip(CONDITIONS, conditions[i]) if not passed]})
                    dwell = dwell + 1 if good else 0
                    maximum = max(maximum, dwell)
                    if dwell >= meta["time"]["required_dwell_steps"] and unit["anchor_verified"]:
                        latch = int(steps[i])
                assert latch == event["recovery_control_step"]
                unit.update(observed_window_steps=len(indexes), stable_window_steps=int(stable.sum()),
                    max_dwell_steps=maximum, recovery_control_step=latch,
                    failed_condition_steps={k: int((~local[:, c]).sum()) for c, k in enumerate(CONDITIONS)},
                    dwell_interruptions=interruptions, status="observed")
                if len(indexes):
                    first, last = indexes[0], indexes[-1]
                    before = int(np.flatnonzero(steps == anchor)[0])
                    peak = indexes[int(np.argmax(pose[indexes, 3]))]
                    np.testing.assert_allclose(motion[first, :2], event["velocity_after_next_physics_step_world_6d"][:2], rtol=0, atol=1e-6)
                    np.testing.assert_allclose(motion[before, :2], event["velocity_before_world_6d"][:2], rtol=0, atol=1e-6)
                    dv = np.asarray(event["observed_delta_v_world_mps"][:2], dtype=float)
                    along = dv / np.linalg.norm(dv)
                    net = motion[last, 2:].astype(float) - motion[before, 2:].astype(float)
                    unit["process"] = {
                        "window_first": sample(steps, pose, motion, first),
                        "window_speed_peak": sample(steps, pose, motion, peak),
                        "window_last": sample(steps, pose, motion, last),
                        "pre_injection_anchor": sample(steps, pose, motion, before),
                        "actual_injected_delta_velocity_world_xy_mps": dv.tolist(),
                        "window_net_displacement_world_xy_m": net.tolist(),
                        "window_net_displacement_parallel_to_this_push_m": float(net @ along),
                        "window_velocity_parallel_first_peak_last_mps": [float(motion[i, :2] @ along) for i in (first, peak, last)],
                        "window_abs_roll_pitch_max_rad": [float(abs(pose[indexes, i]).max()) for i in (1, 2)],
                        "observed_window_duration_s": len(indexes) * meta["time"]["step_dt_s"],
                        "window_reaches_planned_end": int(steps[last]) == anchor + meta["time"]["recovery_window_steps"],
                    }
                row["recoveries"].append(unit)
            latches = [u["recovery_control_step"] for u in row["recoveries"]]
            row["all_required_recoveries_latched"] = bool(latches) and all(v is not None for v in latches)
            later = (steps > max(latches)) & ~conditions.all(axis=1) if row["all_required_recoveries_latched"] else np.zeros(len(steps), bool)
            row["unstable_steps_after_all_recoveries"] = int(later.sum())
            row["first_instability_after_all_recoveries"] = int(steps[later][0]) if later.any() else None
            rows.append(row)
    return {"phase_file_sha256": digest(Path(phase_path)), "sources": sources, "episodes": rows}


def summarize(rows):
    units = [u for r in rows for u in r["recoveries"]]
    return {"episodes": len(rows), "terminal_counts": dict(Counter(r["terminal_label"] for r in rows)),
        "planned_recovery_windows": len(units),
        "executed_events": sum(u["anchor_control_step"] is not None for u in units),
        "observed_window_steps": sum(u["observed_window_steps"] for u in units),
        "stable_window_steps": sum(u["stable_window_steps"] for u in units),
        "stable_observed_window_step_fraction": (
            sum(u["stable_window_steps"] for u in units) / sum(u["observed_window_steps"] for u in units)
            if sum(u["observed_window_steps"] for u in units) else None
        ),
        "dwell_interruptions": sum(len(u["dwell_interruptions"]) for u in units),
        "interrupted_episodes": sum(any(u["dwell_interruptions"] for u in r["recoveries"]) for r in rows),
        "failed_condition_steps": {k: sum(u["failed_condition_steps"][k] for u in units) for k in CONDITIONS},
        "all_recovered_then_final_unstable": sum(r["all_required_recoveries_latched"] and bool(r["final_failed_conditions"]) for r in rows),
        "any_instability_after_all_recoveries": sum(r["unstable_steps_after_all_recoveries"] > 0 for r in rows),
        "episodes_with_any_window_roll_pitch_violation": sum(any(u["failed_condition_steps"]["roll"] or u["failed_condition_steps"]["pitch"] for u in r["recoveries"]) for r in rows)}


def prompt_behavior(curated, case_id):
    case = curated["cases"][case_id]
    facts = case["facts"]
    terminal_counts = facts["whole_stratum"]["terminal_counts"]
    contrast_keys = ("episodes", "planned_recovery_windows", "observed_window_steps", "stable_window_steps",
        "stable_observed_window_step_fraction", "interrupted_episodes", "dwell_interruptions",
        "all_recovered_then_final_unstable", "any_instability_after_all_recoveries",
        "episodes_with_any_window_roll_pitch_violation")
    representatives = []
    for row in facts["representatives"]:
        episode = {"episode": f"source{row['source']}.env{row['env_id']}",
            "outcome": row["terminal_label"], "terminal_step": row["valid_steps"],
            "final_failed_conditions": row["final_failed_conditions"],
            "first_unstable_step_after_all_latches": row["first_instability_after_all_recoveries"],
            "terminal": {k: v for k, v in row["terminal"].items() if k not in ("velocity_world_xy_mps", "control_step")},
            "whole_episode_abs_roll_pitch_max_rad": row["whole_episode_abs_roll_pitch_max_rad"], "events": []}
        for unit in row["recoveries"]:
            event = {k: unit[k] for k in ("event_index", "anchor_control_step", "status", "observed_window_steps",
                "stable_window_steps", "max_dwell_steps", "recovery_control_step", "failed_condition_steps")}
            if "process" in unit:
                p = unit["process"]
                event.update({"speed_first_peak_last_mps_at_step": [[p[key]["control_step"], p[key]["planar_speed_mps"]]
                    for key in ("window_first", "window_speed_peak", "window_last")],
                    "pre_push_speed_mps": p["pre_injection_anchor"]["planar_speed_mps"],
                    "injected_delta_velocity_world_xy_mps": p["actual_injected_delta_velocity_world_xy_mps"],
                    "window_net_displacement_world_xy_m": p["window_net_displacement_world_xy_m"],
                    "window_net_displacement_parallel_m": p["window_net_displacement_parallel_to_this_push_m"],
                    "reset_position_xy_at_window_start_and_end_m": [p["pre_injection_anchor"]["reset_relative_position_world_xy_m"], p["window_last"]["reset_relative_position_world_xy_m"]],
                    "window_abs_roll_pitch_max_rad": p["window_abs_roll_pitch_max_rad"],
                    "window_reaches_planned_end": p["window_reaches_planned_end"]})
            episode["events"].append(event)
        representatives.append(episode)
    evidence = {"schema_version": "p3r.curated_numeric_behavior.v1", "source_kind": "measured_and_independently_recomputed",
        "scope": {"case": case_id, "stratum": case["stratum"], "runs": 3, "episodes": 48,
            "checkpoint_hash": curated["source"]["binding"]["checkpoint_hash"], "split": "public_dev"},
        "measurement_contract": {"step_dt_s": 0.02, "recovery_window_steps": 110,
            "required_consecutive_stable_steps": 21, "stable_limits": curated["source"]["numeric"][0]["parameters"],
            "window": "steps anchor+1 through anchor+110, shortened at termination; terminal judgment also tests final stability",
            "units_and_axes": "speed m/s, angular speed rad/s, height/displacement m, roll/pitch rad; all planar vectors are world xy",
            "net_displacement": "window end minus this event's pre-injection anchor position; reset-relative displacement is separate and includes earlier motion",
            "dwell": "counts and interruptions up to first successful latch; violated-condition totals cover whole observed window; conditions overlap"},
        "cohort_outcomes": {label: {"count": count, "denominator": 48, "fraction": count / 48} for label, count in terminal_counts.items()},
        "not_recovered_vs_success": {label: {k: facts["outcome_groups"][label][k] for k in contrast_keys}
            for label in ("not_recovered", "success_recovered")},
        "representatives": representatives,
        "interpretation_limits": ["A recovery latch and stable posture at termination are different requirements. Brief later instability also occurs in successful episodes.",
            "Examples share cohort/stratum but not identical initial pose or push direction. They are descriptive counterexamples, not causal pairs.",
            "No feet/contact/yaw data: step count or support strategy unknown. Root low speed does not establish that all limbs are motionless.",
            "Measurements establish behavior, not why learning produced it or whether any reward change will improve it."]}
    if case_id == "push-04":
        evidence["interpretation_limits"].append("source0.env55 terminates before its second scheduled push; do not attribute that fall to a second push.")
    return evidence


def assemble(raw_path, full_path, output):
    raw, full = read(raw_path), read(full_path)
    assert len(raw["episodes"]) == len(full["transcripts"]) == 192
    for source in raw["sources"]:
        assert source["reference"]["manifest_sha256"] == full["sources"][source["source"]]["manifest_sha256"]
        assert source["identity"]["split"] == "public_dev"
        for key in ("checkpoint_hash", "evaluation_plan_hash", "program_source_checksum", "resolved_cell_manifest_hash", "resolved_protocol_hash", "runtime_dynamics_hash"):
            assert source["identity"][key] == full["binding"][key], key
    old = {(r["source"], r["env_id"]): r for r in full["transcripts"]}
    for r in raw["episodes"]:
        frozen = old[r["source"], r["env_id"]]
        for key in ("terminal_label", "valid_steps", "final_failed_conditions", "all_required_recoveries_latched", "unstable_steps_after_all_recoveries"):
            assert r[key] == frozen[key], (r["source"], r["env_id"], key)
        for new, before in zip(r["recoveries"], frozen["recoveries"]):
            for key in ("identity", "anchor_control_step", "observed_window_steps", "stable_window_steps", "max_dwell_steps", "recovery_control_step", "failed_condition_steps"):
                assert new[key] == before[key], (r["source"], r["env_id"], key)
            assert len(new["dwell_interruptions"]) == len(before["dwell_interruptions"])
    result = {"schema_version": "p3r.r0b.curated_push.v1", "status": "development_facts_not_value_acceptance",
        "scope": "public_dev; one qualified policy; three frozen runs; 192 closed first episodes; no training-effect claim",
        "source": {"full_path": str(full_path), "full_sha256": digest(Path(full_path)),
            "phase_file_sha256": raw["phase_file_sha256"], "binding": full["binding"],
            "numeric": raw["sources"], "independent_check": "192/192 raw condition/dwell/terminal rows agree with original full evidence"},
        "definitions": {"dt_s": 0.02, "dwell_steps": 21, "window_steps": 110,
            "window": "anchor < step <= anchor+110; anchor69/234 is pre-injection state; first post-step70/235",
            "planar_motion": "signed world root linear velocity xy [m/s]; signed root position xy minus episode reset position [m]",
            "window_net_displacement": "last observed window position minus position at this event's anchor; excludes pre-event drift/prior-push displacement; not necessarily displacement caused only by the push",
            "peak": "maximum planar-speed sample inside observed recovery window, earliest tie; not injected delta-v",
            "directions": "parallel basis is measured injected delta-v for this event; repeated events have separate directions",
            "conditions": "overlap; failed-condition counts do not sum to window steps; dwell stops counting at first latch, condition counts cover whole window"},
        "cases": {},
        "uncertainty": ["No feet/contact/yaw arrays: cannot infer step counts, support strategy or collision force.",
            "Coarse whole-population statistics and representatives have different denominators; selected examples do not establish prevalence.",
            "Not_recovered includes missed dwell and final instability after successful dwell; endpoint upright is not whole-trajectory upright.",
            "Observed instability does not prove a reward term caused it or that an untrained change will work."]}
    for case_id, stratum in (("push-02", "medium"), ("push-04", "repeated")):
        rows = [r for r in raw["episodes"] if r["stratum"] == stratum]
        summary = summarize(rows)
        frozen = full["profile"]["by_stratum"][stratum]
        for key in ("episodes", "terminal_counts", "observed_window_steps", "stable_window_steps", "dwell_interruptions", "failed_condition_steps"):
            assert summary[key] == frozen[key], key
        result["cases"][case_id] = {"stratum": stratum,
            "facts": {"whole_stratum": summary,
                "outcome_groups": {label: summarize([r for r in rows if r["terminal_label"] == label]) for label in sorted(summary["terminal_counts"])},
                "representatives": [next(r for r in rows if (r["source"], r["env_id"]) == key) for key in REPRESENTATIVES[stratum]]},
            "selection": "development-only failure and success from source0 and same stratum; not matched directions or a causal pair",
            "visual": {"status": "pending_separate_media_audit"}}
    video_dir = Path(full_path).parent.parent / "review_video_20260924"
    video_path = video_dir / "push55_recordings.json"
    video = read(video_path)
    record = video["recordings"][0]
    representative = result["cases"]["push-04"]["facts"]["representatives"][0]
    assert video["checkpoint"]["checkpoint_hash"] == full["binding"]["checkpoint_hash"]
    assert video["resolved_cell_manifest_hash"] == full["binding"]["resolved_cell_manifest_hash"]
    assert record["resolved_protocol_hash"] == full["binding"]["resolved_protocol_hash"]
    assert record["env_index"] == representative["env_id"] == 55
    assert record["run_index"] == representative["source"] + 1 == 1
    assert record["episode_outcome"]["episode_steps"] == representative["valid_steps"]
    assert record["episode_outcome"]["failure_label"] == representative["terminal_label"]
    assert abs(record["episode_outcome"]["final_planar_speed_mps"] - representative["terminal"]["planar_speed_mps"]) < 1e-6
    result["cases"]["push-04"]["visual"] = {
        "status": "existing_replay_montage_inspected_developer_observation",
        "episode": {"source": 0, "env_id": 55},
        "sidecar_path": str(video_path), "sidecar_sha256": digest(video_path),
        "montage_path": str(video_dir / "gp55_r1.png"), "montage_sha256": digest(video_dir / "gp55_r1.png"),
        "video_sha256_from_sidecar": record["sha256"], "video_frame_count": record["frame_count"],
        "viewed_montage_control_step_range": [68, 110],
        "observation": "During the sampled post-push sequence the leg configuration and relative foot positions change; late samples show substantial knee flexion and the body lowering. This is visible movement, not evidence of a wholly motionless body.",
        "uncertainty": "Replay is a new run. Checkpoint, cell, protocol, outcome, duration and endpoint speed match the archive, but full per-step visual/numeric equivalence was not verified. Single oblique camera, selected montage samples, no contact forces; do not infer exact step count, support timing or reward causation. The terminal step111 itself is not in the 110-frame video. No second push occurs before this termination.",
        "success_representative_media": "not_available_for_env57",
    }
    result["cases"]["push-02"]["visual"] = {
        "status": "missing_medium_video_at_preparation",
        "checked_scope": "/data/yuyangliu/AGENT/feedback_p3_20260924 recursive recordings.json inspection; only push12(low) and push55(repeated)",
        "excluded_substitute": "push12 low is not medium evidence",
        "suggested_replay": {"launcher": "P3/review_video_20260924/launch_push_video.py",
            "env_index": 19, "success_counterexample_env_index": 23,
            "capture_source_root": "/data/yuyangliu/AGENT/feedback_p3_20260924/source_r3",
            "eval_base_seed": 1000000000, "first_public_run_seed": 1000010000,
            "num_envs": 64, "public_seed_count": 3,
            "checkpoint_hash": full["binding"]["checkpoint_hash"],
            "expected_first_run_outcome": "not_recovered", "expected_first_run_steps": 412,
            "expected_first_run_recovery_latch_step": 109,
            "instruction": "Reuse original qualified checkpoint/cell/source/seed cohort; only video env and fresh output directory change. Resource owner must acquire GPU and account replay cost separately; this preparation did not launch a replay."},
    }
    Path(output).write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    prompt_dir = Path(output).parent / "r0b_evidence"
    prompt_dir.mkdir(exist_ok=True)
    for case_id in result["cases"]:
        behavior = prompt_behavior(result, case_id)
        path = prompt_dir / f"{case_id}_Tc-num.json"
        path.write_text(json.dumps(behavior, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if case_id == "push-04":
            visual = result["cases"][case_id]["visual"]
            augmented = {**behavior, "visual_observations": [{"source_kind": "developer_replay_visual_observation",
                "episode": "source0.env55", "sampled_step_range": [68, 110],
                "visible_behavior": visual["observation"], "limitations": visual["uncertainty"]}],
                "visual_input_mode": "textual observations from a checked replay montage; no images sent to this author"}
            (prompt_dir / f"{case_id}_Tc-vis.json").write_text(json.dumps(augmented, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "verified_episodes": 192, "bytes": Path(output).stat().st_size}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract")
    parser.add_argument("--assemble", nargs=3, metavar=("RAW", "FULL", "OUTPUT"))
    args = parser.parse_args()
    if args.extract:
        print(json.dumps(extract(args.extract), ensure_ascii=False, allow_nan=False))
    else:
        assemble(*args.assemble)
