"""Push-owned root descriptors. Task thresholds are inputs, never fitted here."""
from __future__ import annotations

import math
from copy import deepcopy
import numpy as np

from true_rl_agent.evaluation.trajectory_translation.contracts import NumericEpisode, PredicateTrack, stable_id
from true_rl_agent.evaluation.trajectory_translation.temporal import interval_rows
from .numeric_capture import POSE_FIELDS, PLANAR_MOTION_FIELDS, CONDITIONS

CATALOG_VERSION = "push.root_measurements.v1"
ROOT_METRICS = {
    key: (POSE_FIELDS.index(key), label, unit) for key, label, unit in (
        ("height_m", "根部高度", "m"), ("roll_rad", "根部横滚主值", "rad"),
        ("pitch_rad", "根部俯仰主值", "rad"), ("planar_speed_mps", "根部平面速度模", "m/s"),
        ("angular_speed_radps", "根部角速度模", "rad/s"), ("xy_drift_m", "根部相对重置点距离模", "m"))
}
DISTURBANCES = {
    "world_planar_external_force": ("force_world_n", "N"),
    "world_planar_velocity_increment": ("observed_delta_v_world_mps", "m/s"),
}
CONFIG = {"catalog_version": CATALOG_VERSION, "pre_window_s": 0.5,
          "threshold_source": "archive.parameters_native_pose_dtype",
          "sample_clock": "after_physics_before_reset_and_interval_write",
          "integral": "finite_control_sample_cells_sum_speed_times_dt_not_exact_path",
          "recovery": "first_verified_dwell_ge_required_latched",
          "temporal": "readout_runs_with_unknown_gaps_and_censoring"}


def stability(episode: NumericEpisode) -> tuple[np.ndarray, np.ndarray]:
    pose, p = episode.arrays["pose"], episode.task["parameters"]
    columns = (3, 4, 1, 2, 0)
    names = ("recovery_velocity_mps", "recovery_angular_velocity_radps",
             "recovery_tilt_rad", "recovery_tilt_rad", "recovery_height_m")
    tracks = []
    for column, name in zip(columns, names):
        value = np.abs(pose[:, column]) if column in (1, 2) else pose[:, column]
        threshold = np.asarray(p[name], dtype=pose.dtype)
        passed = value >= threshold if column == 0 else value <= threshold
        tracks.append(np.where(np.isfinite(value), passed.astype(np.int8), -1))
    conditions = np.stack(tracks, axis=1)
    stable = np.where(np.any(conditions == 0, axis=1), 0,
                      np.where(np.all(conditions == 1, axis=1), 1, -1)).astype(np.int8)
    return conditions, stable


def _scalar(values: np.ndarray, total: int) -> dict:
    finite = np.isfinite(values)
    selected = np.asarray(values[finite], dtype=np.float64)
    n = len(selected)
    return {"valid_samples": n, "unknown_samples": total - n,
            "sum": float(selected.sum()) if n else None,
            "mean": float(selected.mean()) if n else None,
            "minimum": float(selected.min()) if n else None,
            "maximum": float(selected.max()) if n else None}


def _reference(episode: NumericEpisode, mask: np.ndarray, channels: tuple[str, ...]) -> dict | None:
    rows = np.flatnonzero(mask)
    return episode.reference(channels, int(rows[0]), int(rows[-1]) + 1) if len(rows) else None


def _window(episode: NumericEpisode, stable: np.ndarray, start: int, end: int) -> dict:
    """Inclusive recorder-step window; no invented sample for an empty window."""
    mask = (episode.steps >= start) & (episode.steps <= end)
    count, expected = int(mask.sum()), max(0, end - start + 1)
    pose = episode.arrays["pose"][mask]
    return {"steps_inclusive": [start, end], "expected_samples": expected, "observed_samples": count,
            "unobserved_samples": expected - count,
            "stable_samples": int((stable[mask] == 1).sum()),
            "known_stability_samples": int((stable[mask] >= 0).sum()),
            "unknown_stability_samples": expected - int((stable[mask] >= 0).sum()),
            "pose": {key: {**_scalar(pose[:, spec[0]], expected), "label": spec[1], "unit": spec[2]} for key, spec in ROOT_METRICS.items()},
            "evidence_ref": _reference(episode, mask, ("pose", "control_step"))}


def _recoveries(episode: NumericEpisode, stable: np.ndarray, conditions: np.ndarray) -> list[dict]:
    task, a = episode.task, episode.arrays
    timing, force = task["time"], task["force"]
    planned = task["planned_push_count"]
    if planned == 0:
        return []
    results = []
    for slot in range(1 if force else planned):
        if force:
            scheduled = np.asarray(timing["event_scheduled_steps"])
            anchor = int(scheduled[planned - 1] + force["pulse_steps"])
            complete = ((a["actual_substeps"] == force["expected_substeps"])
                        & (a["event_steps"] == scheduled)
                        & (a["event_ends"] == scheduled + force["pulse_steps"]))
            eligible = (complete & a["event_verified"])[:, :planned].all(axis=1)
            verified = ~a["unplanned_wrench"] & a["event_verified"][:, :planned].all(axis=1)
            anchor_seen = bool(np.any(complete[:, :planned].all(axis=1) & (episode.steps >= anchor)))
        else:
            anchor = int(a["event_steps"][-1, slot])
            eligible = a["event_steps"][:, slot] >= 0
            verified = a["event_verified"][:, slot]
            anchor_seen = anchor >= 0
        in_window = eligible & (episode.steps > anchor) & (episode.steps <= anchor + timing["recovery_window_steps"])
        dwell = maximum = 0
        recovery = None
        interruptions = []
        previous_step = None
        for row in np.flatnonzero(in_window):
            step = int(episode.steps[row])
            if recovery is None:
                if previous_step is not None and step != previous_step + 1:
                    dwell = 0
                if stable[row] == 1:
                    dwell += 1
                else:
                    if dwell:
                        interruptions.append({"step": step, "preceding_dwell_steps": dwell,
                                              "state": "unknown" if stable[row] == -1 else "false"})
                    dwell = 0
                maximum = max(maximum, dwell)
                if dwell >= timing["required_dwell_steps"] and verified[row]:
                    recovery = step
            previous_step = step
        expected = timing["recovery_window_steps"]
        observed = int(in_window.sum())
        unknown = int((stable[in_window] == -1).sum())
        if recovery is not None:
            status = "recovered"
        elif not anchor_seen:
            status = "anchor_not_observed"
        elif not bool(verified[-1]):
            status = "invalid_injection"
        elif observed == expected and not unknown:
            status = "full_window_without_dwell"
        elif (unknown or (observed and not np.all(np.diff(episode.steps[in_window]) == 1))
              or (episode.steps[-1] >= anchor + expected and observed < expected)):
            status = "unknown_samples"
        elif task["closed"] and episode.steps[-1] < anchor + expected:
            status = "terminal_before_window_end"
        else:
            status = "rollout_truncated"
        channels = ("pose", "event_steps", "event_verified", "control_step")
        if force:
            channels += ("event_ends", "actual_substeps", "unplanned_wrench")
        results.append({"unit": "sequence" if force else "event", "slot": slot,
            "anchor_step": anchor if anchor_seen else None, "scheduled_anchor_step":
                anchor if force else timing["event_scheduled_steps"][slot],
            "status": status, "recovery_step": recovery,
            "latency_s": (recovery - anchor) * episode.dt_s if recovery is not None else None,
            "required_dwell_steps": timing["required_dwell_steps"], "max_dwell_before_latch_steps": maximum,
            "observed_window_samples": observed, "expected_window_samples": expected,
            "unobserved_window_samples": expected - observed, "unknown_stability_samples": unknown,
            "stable_window_samples": int((stable[in_window] == 1).sum()),
            "failed_condition_samples": {name: int((conditions[in_window, i] == 0).sum()) for i, name in enumerate(CONDITIONS)},
            "interruptions": interruptions, "right_censored": recovery is None and status != "full_window_without_dwell",
            "evidence_ref": episode.reference(channels)})
    return results


def _events(episode: NumericEpisode, stable: np.ndarray) -> list[dict]:
    task = episode.task
    force = task["force"]
    events = []
    pre = math.ceil(CONFIG["pre_window_s"] / episode.dt_s)
    for slot in range(task["planned_push_count"]):
        observed = next((e for e in task["events"] if e["event_index"] == slot), None)
        pointer = f"/metadata/runtime_episode_records/{episode.env_index}/{task['event_metadata_key']}"
        if observed is None:
            events.append({"slot": slot, "status": "not_observed", "metadata_pointer": pointer,
                           "windows": None, "root_along_disturbance": None})
            continue
        start = observed["start_control_step" if force else "control_step"]
        end = observed["end_control_step"] if force else start
        next_start = task["time"]["event_scheduled_steps"][slot + 1] if slot + 1 < task["planned_push_count"] else None
        post_end = end + task["time"]["recovery_window_steps"]
        if next_start is not None:
            post_end = min(post_end, next_start)
        vector_key, vector_unit = DISTURBANCES[task["parameters"]["mechanism"]]
        vector = np.asarray(observed[vector_key], dtype=float)
        windows = {"pre": _window(episode, stable, max(1, start - pre + 1), start),
                   "during": _window(episode, stable, start + 1, end) if force else None,
                   "post": _window(episode, stable, end + 1, post_end)}
        direction = {"status": "not_recorded_root_signed_xy", "mean_velocity_mps": None,
                     "valid_samples": 0, "unknown_samples": windows["post"]["expected_samples"]}
        if "planar_motion" in episode.arrays:
            norm = float(np.linalg.norm(vector[:2]))
            if norm > 0 and observed["injection_verified"]:
                mask = (episode.steps > end) & (episode.steps <= post_end)
                velocities = episode.arrays["planar_motion"][mask, :2].astype(float)
                projections = velocities @ (vector[:2] / norm)
                metric = _scalar(projections, windows["post"]["expected_samples"])
                direction = {"status": "world_xy_projection", "mean_velocity_mps": metric["mean"],
                    "valid_samples": metric["valid_samples"], "unknown_samples": metric["unknown_samples"],
                    "unit_vector_world_xy": (vector[:2] / norm).tolist(),
                    "evidence_ref": _reference(episode, mask, ("planar_motion", "control_step"))}
            else:
                direction["status"] = "unverified_or_zero_disturbance"
        events.append({"slot": slot, "status": "recorded", "start_step": start, "end_step": end,
            "injection_verified": observed["injection_verified"], "vector_key": vector_key,
            "vector_world": vector.tolist(), "vector_unit": vector_unit,
            "delivered_impulse_world_ns": list(observed["delivered_impulse_world_ns"]) if force else None,
            "metadata_pointer": pointer + f"/{task['events'].index(observed)}", "windows": windows,
            "root_along_disturbance": direction})
    return events


def measure_push_episode(episode: NumericEpisode) -> dict:
    """Full observed episode, including unsuccessful and censored episodes."""
    a, task = episode.arrays, episode.task
    n = len(episode.steps)
    expected = int(task["expected_last_step"])
    if episode.steps[-1] > expected:
        raise ValueError("Push sample extends past declared episode")
    conditions, stable = stability(episode)
    track = PredicateTrack(episode.episode_id, "root_stability", ("root",), ("pose",), stable, CONFIG)
    pose = {key: {**_scalar(a["pose"][:, spec[0]], expected), "label": spec[1], "unit": spec[2]} for key, spec in ROOT_METRICS.items()}
    final_available = bool(task["closed"] and episode.steps[-1] == expected)
    final_values = {key: float(a["pose"][-1, i]) if final_available and np.isfinite(a["pose"][-1, i]) else None
                    for i, key in enumerate(POSE_FIELDS)}
    final_stable = bool(stable[-1]) if final_available and stable[-1] >= 0 else None
    recoveries = _recoveries(episode, stable, conditions)
    any_recovered = any(r["recovery_step"] is not None for r in recoveries)
    recovery_known_absent = bool(recoveries) and all(r["status"] == "full_window_without_dwell" for r in recoveries)
    signed = None
    if "planar_motion" in a:
        signed = {key: {**_scalar(a["planar_motion"][:, i], expected),
                       "final": float(a["planar_motion"][-1, i]) if final_available and np.isfinite(a["planar_motion"][-1, i]) else None}
                  for i, key in enumerate(PLANAR_MOTION_FIELDS)}
    value = {"episode_id": episode.episode_id, "run_index": episode.run_index, "env_index": episode.env_index,
        "source": deepcopy(episode.source), "stratum": task["stratum"], "control_only": task["control_only"],
        "planned_push_count": task["planned_push_count"], "task_outcome": task["terminal_label"] if final_available else None,
        "outcome_class": ("success" if task["terminal_label"] == "success_recovered" else "failure")
            if final_available and not task["control_only"] and task["terminal_label"] is not None else "unknown",
        "task_outcome_ref": {"manifest_sha256": episode.source["manifest_sha256"],
            "pointer": f"/metadata/runtime_episode_records/{episode.env_index}/failure_label"},
        "coverage": {"observed_samples": n, "expected_samples": expected, "unobserved_samples": expected - n,
                     "dt_s": episode.dt_s, "first_step": int(episode.steps[0]), "last_step": int(episode.steps[-1]),
                     "sample_cell_duration_proxy_s": n * episode.dt_s,
                     "observed_span_s": float(episode.times[-1] - episode.times[0]), "terminal_observed": final_available},
        "pose": pose, "signed_xy": signed,
        "speed_integral_proxy_m": pose["planar_speed_mps"]["sum"] * episode.dt_s
            if pose["planar_speed_mps"]["sum"] is not None else None,
        "stability": {"true_samples": int((stable == 1).sum()), "false_samples": int((stable == 0).sum()),
                      "known_samples": int((stable >= 0).sum()), "unknown_samples": expected - int((stable >= 0).sum()),
                      "intervals": interval_rows(episode, track)},
        "final": {"pose": final_values, "stable": final_stable,
                  "failure_flag": bool(a["failure"][-1]) if final_available else None,
                  "timeout_flag": bool(a["timeout"][-1]) if final_available else None,
                  "evidence_ref": episode.reference(("pose", "failure", "timeout"), n - 1) if final_available else None},
        "recoveries": recoveries, "events": _events(episode, stable),
        "any_recovered": True if any_recovered else False if recovery_known_absent else None,
        "final_unstable_after_recovery": (not final_stable) if any_recovered and final_stable is not None else None,
        "parameters": deepcopy(task["parameters"]), "timing": deepcopy(task["time"])}
    ref = episode.reference(tuple(a))
    return {"record_id": stable_id("push-evidence", episode.episode_id, CONFIG, value),
            "record_type": "root_episode_measurements", "catalog_version": CATALOG_VERSION,
            "measurement_config": dict(CONFIG), "evidence_ref": ref, "value": value}
