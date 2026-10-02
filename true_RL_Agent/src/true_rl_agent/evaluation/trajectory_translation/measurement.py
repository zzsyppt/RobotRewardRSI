"""Deterministic numeric descriptors with explicit per-channel validity.

All values use float64 operands. This module has no archive selection, task
classification, natural-language authoring, or reference-oracle dependency.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .contracts import CATALOG_VERSION, NumericEpisode, PredicateTrack, predicate
from .temporal import compose_track, contiguous_slices, interval_rows, pair_onsets, relate_states

NUMERIC_CONFIG = {"version": CATALOG_VERSION, "quaternion_norm_tolerance": 1e-4,
                  "low_horizontal_speed_mps": 0.05, "progress_window_s": 0.5,
                  "progress_increment_m": 0.02, "share_zero_tolerance": 1e-12,
                  "low_speed_and_progress_threshold_source": "development_calibration",
                  "up_z_negative_threshold": 0.0, "up_z_threshold_source": "robot_scale_rule",
                  "combination_encoding": "role_index_lsb.v1"}


def _channel(episode: NumericEpisode, name: str, tail: tuple[int, ...]) -> np.ndarray:
    if name not in episode.arrays:
        # Missing is represented internally as NaN, never as a physical zero.
        return np.full((len(episode.steps), *tail), np.nan)
    values = np.asarray(episode.arrays[name], dtype=np.float64)
    if values.shape != (len(episode.steps), *tail):
        raise ValueError(f"{name}: entity/component dimensions differ from binding")
    return values


def _range(values: np.ndarray, steps: np.ndarray) -> dict[str, Any]:
    valid = np.isfinite(values)
    indices = np.flatnonzero(valid)
    if not len(indices):
        return {"valid_count": 0, "unknown_count": len(steps), "minimum": None, "maximum": None,
                "first": None, "last": None, "range": None, "minimum_step": None, "maximum_step": None,
                "first_valid_step": None, "last_valid_step": None}
    selected = values[indices]
    lo, hi = int(indices[np.argmin(selected)]), int(indices[np.argmax(selected)])
    return {"valid_count": len(indices), "unknown_count": len(steps) - len(indices),
            "minimum": float(values[lo]), "maximum": float(values[hi]),
            "first": float(selected[0]), "last": float(selected[-1]),
            "range": float(values[hi] - values[lo]),
            "first_valid_step": int(steps[indices[0]]), "last_valid_step": int(steps[indices[-1]]),
            "minimum_step": int(steps[lo]), "maximum_step": int(steps[hi])}


def _fraction(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if abs(denominator) > NUMERIC_CONFIG["share_zero_tolerance"] else None


def _track_summary(episode: NumericEpisode, track: PredicateTrack) -> dict[str, Any]:
    rows = interval_rows(episode, track)
    on = int(np.count_nonzero(track.values == 1))
    off = int(np.count_nonzero(track.values == 0))
    return {"true_count": on, "false_count": off, "unknown_count": len(track.values) - on - off,
            "valid_count": on + off, "fraction": _fraction(on, on + off),
            "numerator_cell_s": on * episode.dt_s, "denominator_cell_s": (on + off) * episode.dt_s,
            "weighting": "sample_cell_duration_proxy", "intervals": rows,
            "observed_onset_count": int(np.count_nonzero((track.values[:-1] == 0) & (track.values[1:] == 1) & episode.adjacent)),
            "observed_offset_count": int(np.count_nonzero((track.values[:-1] == 1) & (track.values[1:] == 0) & episode.adjacent))}


def _drawdown(episode: NumericEpisode, progress: np.ndarray) -> dict[str, Any]:
    best = None
    for start, stop in contiguous_slices(episode.steps, np.isfinite(progress)):
        peak = start
        for trough in range(start, stop):
            if progress[trough] > progress[peak]:
                peak = trough
            amount = float(progress[peak] - progress[trough])
            if best is None or amount > best[0]:
                best = (amount, peak, trough, stop)
    if best is None:
        return {"max_drawdown_m": None, "peak_step": None, "trough_step": None, "return_step": None}
    amount, peak, trough, stop = best
    recovered = next((i for i in range(trough + 1, stop) if progress[i] >= progress[peak]), None)
    return {"max_drawdown_m": amount, "peak_step": int(episode.steps[peak]),
            "trough_step": int(episode.steps[trough]),
            "return_step": int(episode.steps[recovered]) if recovered is not None else None,
            "return_observation_right_censored": recovered is None,
            "search_end_step": int(episode.steps[stop - 1]), "cross_gap_search": False}


def low_progress_track(episode: NumericEpisode, progress: np.ndarray) -> PredicateTrack:
    width = math.ceil(NUMERIC_CONFIG["progress_window_s"] / episode.dt_s)
    values = np.full(len(progress), -1, dtype=np.int8)
    for start, stop in contiguous_slices(episode.steps, np.isfinite(progress)):
        for end in range(start + width, stop):
            values[end] = (np.max(progress[end - width:end + 1]) - progress[end - width]
                           < NUMERIC_CONFIG["progress_increment_m"])
    return PredicateTrack(episode.episode_id, "low_progress", episode.entities["root"], ("base_pos_w",), values,
        {"formula": "max(progress[k-w:k+1])-progress[k-w] < increment_m",
         "threshold_source": "development_calibration", "increment_m": NUMERIC_CONFIG["progress_increment_m"],
         "lookback_samples": width, "requested_window_s": NUMERIC_CONFIG["progress_window_s"],
         "progress_direction_world": episode.task["progress_direction_world"],
         "actual_window_span_s": width * episode.dt_s,
         "confirmation_semantics": "window_end_not_physical_onset"})


def measure_episode(episode: NumericEpisode) -> list[dict[str, Any]]:
    """Process every supplied row, retaining numeric summaries and raw short events."""
    n, steps = len(episode.steps), episode.steps
    support = episode.entities["support"]
    bodies = episode.entities["forbidden"]
    joints = episode.entities["joints"]
    root = episode.entities["root"]
    threshold = float(episode.task["contact_threshold_n"])
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("contact threshold must be finite and nonnegative")
    config = {**NUMERIC_CONFIG, "contact_threshold_n": threshold,
              "contact_threshold_source": "task_spec"}
    records: list[dict[str, Any]] = []
    tracks: dict[str, PredicateTrack] = {}
    track_records: dict[str, list[dict[str, Any]]] = {}

    def emit(key, entities, channels, value, unit, frame, definition):
        absent = [name for name in channels if name not in episode.arrays]
        records.append(episode.record(key, "measurement", entities, channels,
            {**value, "missing_channels": absent, "processing_completeness": "processed",
             "validity_policy": "per_quantity_required_components; see valid/unknown counts"}, unit=unit, frame=frame,
            config={**config, "definition": definition}))

    def add_track(key, entities, channels, values, definition):
        track = PredicateTrack(episode.episode_id, key, entities, channels, values,
                               {**config, "definition": definition})
        tracks[key] = track
        track_records[key] = compose_track(episode, track)
        records.extend(track_records[key])
        return _track_summary(episode, track)

    p = _channel(episode, "base_pos_w", (3,))
    position_valid = np.isfinite(p).all(axis=1)
    direction = np.asarray(episode.task["progress_direction_world"], dtype=float)
    origin = np.asarray(episode.task["origin_world"], dtype=float)
    if direction.shape != (3,) or origin.shape != (3,) or not np.isfinite([direction, origin]).all() or not np.isclose(np.linalg.norm(direction), 1):
        raise ValueError("task progress requires a finite origin and unit direction")
    progress = (p - origin) @ direction
    progress[~position_valid] = np.nan
    edges = episode.adjacent & position_valid[:-1] & position_valid[1:]
    delta = np.diff(p, axis=0)[edges]
    dp = np.diff(progress)[edges]
    net = (p[-1] - p[0]).tolist() if position_valid[0] and position_valid[-1] else None
    emit("root_progress", root, ("base_pos_w",), {
        "valid_count": int(position_valid.sum()), "unknown_count": int((~position_valid).sum()),
        "progress": _range(progress, steps), "net_world_xyz_m": net,
        "net_progress_m": float(progress[-1] - progress[0]) if net is not None else None,
        "sampled_path_xy_m": float(np.linalg.norm(delta[:, :2], axis=1).sum()) if len(delta) else None,
        "sampled_path_xyz_m": float(np.linalg.norm(delta, axis=1).sum()) if len(delta) else None,
        "positive_progress_variation_m": float(np.maximum(dp, 0).sum()) if len(dp) else None,
        "negative_progress_variation_m": float(np.maximum(-dp, 0).sum()) if len(dp) else None,
        "valid_edge_count": int(edges.sum()), "path_complete": bool(edges.all()) and n > 1,
        "interval_average_speed_mps": {"minimum": float(np.min(np.linalg.norm(delta, axis=1) / episode.dt_s)) if len(delta) else None,
                                       "maximum": float(np.max(np.linalg.norm(delta, axis=1) / episode.dt_s)) if len(delta) else None},
        "drawdown": _drawdown(episode, progress), "origin_world": origin.tolist(),
        "progress_direction_world": direction.tolist(), "static_refs": episode.task.get("static_refs", {})},
        "m", "world_and_task_projection", "M01 endpoint displacement; adjacent-valid sample polylines; no continuous path claim")
    low = low_progress_track(episode, progress)
    tracks[low.key] = low
    track_records[low.key] = compose_track(episode, low)
    records.extend(track_records[low.key])
    emit("low_progress", root, ("base_pos_w",), _track_summary(episode, low), "1", "task_projection", low.config)

    raw_q = _channel(episode, "base_quat_w", (4,))
    qnorm = np.linalg.norm(raw_q, axis=1)
    qvalid = np.isfinite(raw_q).all(axis=1) & (np.abs(qnorm - 1) <= config["quaternion_norm_tolerance"])
    q = np.full_like(raw_q, np.nan)
    np.divide(raw_q, qnorm[:, None], out=q, where=qvalid[:, None])
    w, x, y, z = q.T
    up = 1 - 2 * (x * x + y * y)
    raw_up = 1 - 2 * (raw_q[:, 1] ** 2 + raw_q[:, 2] ** 2)
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    negative = add_track("body_up_negative", root, ("base_quat_w",), predicate(up < 0, qvalid),
                         "M02 normalized wxyz up_z < 0; geometric axis relation, not failure cause")
    emit("body_up", root, ("base_quat_w",), {
        "up_z": _range(up, steps), "roll_rad": _range(roll, steps), "pitch_rad": _range(pitch, steps),
        "negative": negative, "max_quaternion_norm_error": float(np.max(np.abs(qnorm[np.isfinite(qnorm)] - 1))) if np.isfinite(qnorm).any() else None,
        "raw_body_up_diagnostic": {**_range(raw_up, steps),
            "first_negative_step": int(steps[np.flatnonzero(np.isfinite(raw_up) & (raw_up < 0))[0]]) if np.any(np.isfinite(raw_up) & (raw_up < 0)) else None},
        "raw_vs_normalized_up_z_max_abs_delta": float(np.max(np.abs(raw_up[qvalid] - up[qvalid]))) if qvalid.any() else None},
        "1; roll/pitch rad", "body_to_world_wxyz", "M02 normalize only after raw norm validation; raw diagnostic separate")

    force = _channel(episode, "support_contact_forces_w", (len(support), 3))
    fvalid = np.isfinite(force).all(axis=2)
    fnorm = np.linalg.norm(force, axis=2)
    markers = predicate(fnorm > threshold, fvalid)
    velocity = _channel(episode, "support_lin_vel_w", (len(support), 3))
    vxy = np.linalg.norm(velocity[:, :, :2], axis=2)
    vxyz = np.linalg.norm(velocity, axis=2)
    for i, entity in enumerate(support):
        marker_key = f"{entity}.marker"
        stats = add_track(marker_key, (entity,), ("support_contact_forces_w",), markers[:, i],
                          "M03 norm(current recorded net normal contact force)>threshold; sampled marker only")
        emit(marker_key, (entity,), ("support_contact_forces_w",), stats, "1", "world", "M03 sample-cell weighted occupancy")
        # Only xy components are dependencies of the M04 speed predicate.
        valid = fvalid[:, i] & np.isfinite(velocity[:, i, :2]).all(axis=1)
        vals = predicate((markers[:, i] == 0) & (vxy[:, i] < config["low_horizontal_speed_mps"]), valid)
        key = f"{entity}.low_speed_off"
        summary = add_track(key, (entity,), ("support_contact_forces_w", "support_lin_vel_w"), vals,
                            "M04 current marker off AND recorded world XY speed strictly below threshold")
        true_runs = [row for row in summary["intervals"] if row["state"] == "true"]
        longest = max(true_runs, key=lambda row: row["sample_count"], default=None)
        diagnostics = None
        if longest:
            a, b = longest["index_half_open"]
            qmask = qvalid[a:b]
            diagnostics = {"interval": longest, "max_speed_xy_mps": float(np.max(vxy[a:b, i])),
                "median_speed_xy_mps": float(np.median(vxy[a:b, i])),
                "max_speed_xyz_mps": float(np.max(vxyz[a:b, i])) if np.isfinite(vxyz[a:b, i]).all() else None,
                "median_speed_xyz_mps": float(np.median(vxyz[a:b, i])) if np.isfinite(vxyz[a:b, i]).all() else None,
                "max_force_norm_n": float(np.max(fnorm[a:b, i])),
                "minimum_up_z": float(np.min(up[a:b][qmask])) if qmask.any() else None,
                "maximum_up_z": float(np.max(up[a:b][qmask])) if qmask.any() else None,
                "orientation_valid_count": int(qmask.sum())}
        emit(key, (entity,), ("support_contact_forces_w", "support_lin_vel_w", "base_quat_w"),
             {**summary, "longest": diagnostics}, "1; diagnostics m/s,N", "world", "M04 XY predicate; XYZ diagnostics only")

    joint_valid = fvalid.all(axis=1)
    codes = sum((markers[:, i] == 1).astype(np.int64) * (1 << i) for i in range(len(support)))
    codes[~joint_valid] = -1
    counts = [int(np.count_nonzero(codes == code)) for code in range(1 << len(support))]
    changes = [{"step": int(steps[k]), "before": int(codes[k - 1]), "after": int(codes[k]),
                "readout_range_s": episode.times[k - 1:k + 1].tolist()}
               for k in range(1, n) if episode.adjacent[k - 1] and codes[k - 1] >= 0 and codes[k] >= 0 and codes[k] != codes[k - 1]]
    equalities = []
    for i, left in enumerate(support):
        for j in range(i + 1, len(support)):
            right = support[j]
            both = fvalid[:, i] & fvalid[:, j]
            equalities.append({"entities": [left, right], "equal_count": int(np.count_nonzero(both & (markers[:, i] == markers[:, j]))),
                               "valid_count": int(both.sum()), "includes_both_off": True})
            records.extend(pair_onsets(episode, tracks[f"{left}.marker"], tracks[f"{right}.marker"]))
    emit("contact_combinations", support, ("support_contact_forces_w",),
         {"encoding_version": config["combination_encoding"], "bit_order": list(support),
          "bit_weights": [1 << i for i in range(len(support))], "counts": counts,
          "valid_count": int(joint_valid.sum()), "unknown_count": int((~joint_valid).sum()),
          "transitions": changes, "pair_state_equality": equalities,
          "pairing_direction": "each lower role index is reference; all original events retained",
          "all_off_does_not_establish_flight": True}, "sample_count", "role_order", "M05 role_index_lsb.v1; M06 no smoothing")

    # Rectangular cell weights and adjacent-sample trapezoids are distinct.
    joint_z_valid = np.isfinite(force[:, :, 2]).all(axis=1)
    norm_sums = fnorm[joint_valid].sum(axis=0)
    z_sums = force[joint_z_valid, :, 2].sum(axis=0)
    norm_den, z_den = float(norm_sums.sum()), float(z_sums.sum())
    integrals, force_stats = [], []
    for i, entity in enumerate(support):
        edges_i = episode.adjacent & fvalid[:-1, i] & fvalid[1:, i]
        trap = ((force[:-1, i] + force[1:, i]) / 2)[edges_i].sum(axis=0) * episode.dt_s
        trap_norm = ((fnorm[:-1, i] + fnorm[1:, i]) / 2)[edges_i].sum() * episode.dt_s
        integrals.append({"entity": entity, "valid_edge_count": int(edges_i.sum()),
                          "covered_span_s": int(edges_i.sum()) * episode.dt_s,
                          "vector_ns": trap.tolist() if edges_i.any() else None,
                          "norm_ns": float(trap_norm) if edges_i.any() else None})
        force_stats.append({"entity": entity, "norm_n": _range(fnorm[:, i], steps),
                            "signed_fz_n": _range(force[:, i, 2], steps)})
    emit("force_summary", support, ("support_contact_forces_w",), {
        "joint_valid_count": int(joint_valid.sum()), "unknown_joint_count": int((~joint_valid).sum()),
        "signed_fz_joint_valid_count": int(joint_z_valid.sum()), "signed_fz_unknown_joint_count": int((~joint_z_valid).sum()),
        "sample_sum_norm_n": norm_sums.tolist(), "sample_sum_signed_fz_n": z_sums.tolist(),
        "rectangle_norm_ns": (norm_sums * episode.dt_s).tolist(),
        "rectangle_signed_fz_ns": (z_sums * episode.dt_s).tolist(),
        "norm_share": [_fraction(float(v), norm_den) for v in norm_sums],
        "signed_fz_share": [_fraction(float(v), z_den) for v in z_sums],
        "share_weighting": "joint_valid_sample_cell_duration_proxy", "per_entity": force_stats,
        "trapezoid_sample_integral_proxy": integrals, "physical_impulse": "unavailable",
        "momentum_balance": "unavailable_missing_mass_momentum_tangential_and_external_forces"},
        "N; integral proxy N*s; share 1", "world", "M07 net normal contact force only; rectangles are not trapezoids or physical impulses")

    support_p = _channel(episode, "support_pos_w", (len(support), 3))
    offset = support_p - p[:, None, :]
    # Inverse unit-quaternion rotation: v - 2*w*(q_xyz x v) + 2*q_xyz x (q_xyz x v).
    qvec = q[:, None, 1:]
    cross = np.cross(qvec, offset)
    relative = offset - 2 * q[:, None, :1] * cross + 2 * np.cross(qvec, cross)
    for i, entity in enumerate(support):
        valid = qvalid & position_valid & np.isfinite(support_p[:, i]).all(axis=1)
        relative[~valid, i] = np.nan
        edge = episode.adjacent & valid[:-1] & valid[1:]
        relative_delta = np.diff(relative[:, i], axis=0)[edge]
        emit(f"{entity}.relative_position", (entity, *root), ("support_pos_w", "base_pos_w", "base_quat_w", "support_lin_vel_w"), {
            "body_xyz_m": [_range(relative[:, i, axis], steps) for axis in range(3)],
            "first_body_xyz_m": relative[0, i].tolist() if valid[0] else None,
            "last_body_xyz_m": relative[-1, i].tolist() if valid[-1] else None,
            "mean_body_xyz_m": np.mean(relative[valid, i], axis=0).tolist() if valid.any() else None,
            "relative_sample_path_m": float(np.linalg.norm(relative_delta, axis=1).sum()) if edge.any() else None,
            "valid_edge_count": int(edge.sum()), "world_xy_speed_mps": _range(vxy[:, i], steps),
            "world_xyz_speed_mps": _range(vxyz[:, i], steps),
            "default_pose_reference": "unavailable", "position_velocity_reference_equivalence": "unverified"},
            "m; recorded speed m/s", "body_and_world_separate", "M08 full normalized rotation of recorded link reference; COM velocity is not its exact derivative")

    joint_p = _channel(episode, "joint_pos", (len(joints),))
    for i, entity in enumerate(joints):
        emit(f"{entity}.pose", (entity,), ("joint_pos",),
             {**_range(joint_p[:, i], steps), "limit_proximity": "unavailable_unbound_effective_or_nominal_limits"},
             "rad", "joint_coordinate", "M09 absolute recorded joint position; no default abnormality threshold")

    # Buffer-any is deliberately a separate readout quantity, with no slot clock.
    for prefix, entities, current in (("support", support, force), ("forbidden", bodies,
            _channel(episode, "forbidden_contact_forces_w", (len(bodies), 3)))):
        channel = f"{prefix}_contact_forces_w"
        history_name = channel + "_history"
        history_length = int(episode.source["contact_force_history"]["history_length"])
        history = _channel(episode, history_name, (history_length, len(entities), 3))
        hnorm = np.linalg.norm(history, axis=3)
        hvalid = np.isfinite(history).all(axis=(1, 3))
        cvalid = np.isfinite(current).all(axis=2)
        cnorm = np.linalg.norm(current, axis=2)
        hmarkers = predicate((hnorm > threshold).any(axis=1), hvalid)
        cmarkers = predicate(cnorm > threshold, cvalid)
        for i, entity in enumerate(entities):
            if prefix == "forbidden":
                stats = add_track(f"{entity}.current_marker", (entity,), (channel,), cmarkers[:, i],
                                  "M07 recorded non-support body current-force marker")
                emit(f"{entity}.current_marker", (entity,), (channel,),
                     {**stats, "force_norm_n": _range(cnorm[:, i], steps)}, "1; force N", "world", "M07 measured body only; contact object unknown")
            hstats = add_track(f"{entity}.history_any_marker", (entity,), (history_name,), hmarkers[:, i],
                               "ANY stored history slot exceeds threshold; not a physical substep series")
            both = hvalid[:, i] & cvalid[:, i]
            emit(f"{entity}.history_diagnostic", (entity,), (channel, history_name), {
                **hstats, "history_any_on_current_off_count": int(np.count_nonzero(both & (hmarkers[:, i] == 1) & (cmarkers[:, i] == 0))),
                "current_on_history_all_off_count": int(np.count_nonzero(both & (cmarkers[:, i] == 1) & (hmarkers[:, i] == 0))),
                "joint_valid_count": int(both.sum()), "slot_timestamps": "not_recorded", "slot_validity": "not_recorded",
                "full_physics_coverage": False, "integrable_as_substeps": False},
                "sample_count", "sensor_buffer_read_at_control_endpoint", "History diagnostics do not replace current markers")

    terms = tuple(episode.task["termination_term_order"])
    flags = ("done_mask", "wrapper_done_mask", *(f"termination_{i}" for i in range(len(terms))))
    terminal_flags = {}
    for name in flags:
        arr = _channel(episode, name, ())
        valid_flag = np.isfinite(arr) & np.isin(arr, [0, 1])
        vals = predicate(arr == 1, valid_flag)
        add_track(name, root, (name,), vals, "M10 archived manager/wrapper signal; not translator outcome")
        terminal_flags[name] = bool(arr[-1]) if valid_flag[-1] else None
    tail = {name: [bool(v) if np.isfinite(v) and v in (0, 1) else None
                   for v in episode.arrays[name][-min(5, n):]] for name in flags if name in episode.arrays}
    forbidden_force = _channel(episode, "forbidden_contact_forces_w", (len(bodies), 3))
    forbidden_history = _channel(episode, "forbidden_contact_forces_w_history", (history_length, len(bodies), 3))
    def finite_tail(values):
        array = values[-min(5, n):]
        return np.where(np.isfinite(array), array.astype(object), None).tolist()
    emit("task_endpoint", (*root, *bodies, *support), (*flags, "base_pos_w", "support_contact_forces_w", "forbidden_contact_forces_w", "forbidden_contact_forces_w_history"), {"outcome": episode.task["outcome"],
        "authority": "archived_task_evaluator", "terminal_step": int(steps[-1]),
        "terminal_time_s": float(episode.times[-1]), "terminal_flags": terminal_flags,
        "goal_margin_m": float(episode.task["goal_progress_m"] - progress[-1]) if np.isfinite(progress[-1]) else None,
        "deadline_margin_s": float(episode.task["deadline_s"] - episode.times[-1]),
        "goal_progress_m": episode.task["goal_progress_m"], "deadline_s": episode.task["deadline_s"],
        "static_refs": episode.task.get("static_refs", {}),
        "tail_steps": steps[-min(5, n):].tolist(), "tail_flags": tail,
        "tail_forbidden_norms_n": finite_tail(np.linalg.norm(forbidden_force, axis=2)),
        "tail_support_norms_n": finite_tail(fnorm),
        "tail_forbidden_history_norms_n": finite_tail(np.linalg.norm(forbidden_history, axis=3)),
        "geometry_clearance": "deferred_pending_independent_geometry_validation"},
        "m; time s; force N; flags bool", "task", "M10 immutable archived outcome and valid terminal; no outcome reassignment")

    # Small deterministic combinations keep their operands, rather than assigning
    # a failure name. All runs/events above remain present regardless of selection.
    inverted = [r for r in track_records["body_up_negative"] if r["record_type"] == "state_interval" and r["value"]["state"] == "true"]
    slow_progress = [r for r in track_records["low_progress"] if r["record_type"] == "state_interval" and r["value"]["state"] == "true"]
    for entity in support:
        quiet = [r for r in track_records[f"{entity}.low_speed_off"] if r["record_type"] == "state_interval" and r["value"]["state"] == "true"]
        for state in quiet:
            for inv in inverted:
                records.append(relate_states(episode, inv, state, "onset_before"))
                records.append(relate_states(episode, inv, state, "overlap"))
            for low_state in slow_progress:
                if max(state["value"]["first_observed_s"], low_state["value"]["first_observed_s"]) <= min(state["value"]["last_observed_s"], low_state["value"]["last_observed_s"]):
                    records.append(relate_states(episode, state, low_state, "overlap"))
    return records


def aggregate_fractions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Explicit arithmetic units; no policy selection or reference comparison.

    Each row supplies numerator_cell_s/denominator_cell_s for one episode.
    Missing denominators remain unknown episodes, never zero-valued observations.
    """
    known = [r for r in rows if r["denominator_cell_s"] > 0]
    numerator = sum(r["numerator_cell_s"] for r in known)
    denominator = sum(r["denominator_cell_s"] for r in known)
    return {"total_episode_count": len(rows), "valid_episode_count": len(known),
            "unknown_episode_count": len(rows) - len(known),
            "episode_equal_mean": sum(r["numerator_cell_s"] / r["denominator_cell_s"] for r in known) / len(known) if known else None,
            "sample_cell_duration_weighted_fraction": _fraction(numerator, denominator),
            "numerator_cell_s": numerator, "denominator_cell_s": denominator,
            "units": {"episode_equal_mean": "episode_equal_fraction", "sample_cell_duration_weighted_fraction": "sample_cell_time_fraction"}}
