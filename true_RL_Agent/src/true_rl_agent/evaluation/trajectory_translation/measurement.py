"""Deterministic numeric descriptors with explicit per-channel validity.

All values use float64 operands. This module has no archive selection, task
classification, natural-language authoring, or reference-oracle dependency.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .contracts import CATALOG_VERSION, NumericEpisode, PredicateTrack, predicate
from .temporal import contiguous_slices, interval_rows

NUMERIC_CONFIG = {"version": CATALOG_VERSION, "quaternion_norm_tolerance": 1e-4,
                  "terminal_window_s": .5,
                  "motion_proxy": "adjacent_recorded_position.v2",
                  "low_horizontal_speed_mps": 0.05, "progress_window_s": 0.5,
                  "progress_increment_m": 0.02, "share_zero_tolerance": 1e-12,
                  "low_speed_and_progress_threshold_source": "development_calibration",
                  "up_z_negative_threshold": 0.0, "up_z_threshold_source": "robot_scale_rule",
                  "combination_encoding": "role_index_lsb.v1"}

# Recorded quaternions are float32. This only resolves numerical degeneracy.
DIRECTION_EPS = float(8 * np.finfo(np.float32).eps)


def terminal_window_steps(dt_s):
    return math.floor(NUMERIC_CONFIG["terminal_window_s"] / dt_s + 1e-9)


def terminal_window_mask(episode):
    return episode.steps >= episode.steps[-1] - terminal_window_steps(episode.dt_s)


def axis_azimuth(axis, forward):
    lateral = np.cross([0., 0., 1.], forward)
    x, y = axis @ forward, axis @ lateral
    degenerate = np.isfinite(x) & np.isfinite(y) & (np.hypot(x, y) <= DIRECTION_EPS)
    angle = np.where(degenerate, np.nan, np.degrees(np.arctan2(y, x)))
    return angle, degenerate


def _low_progress_facet(steps, low, up, moving):
    selected = low.values == 1
    counts = {}
    for name, posture in (("up_negative", up < 0), ("up_nonnegative", up >= 0),
                          ("posture_unknown", ~np.isfinite(up))):
        mask = selected & posture
        counts[name] = {"active": int(np.count_nonzero(mask & (moving == 1))),
                       "quiet": int(np.count_nonzero(mask & (moving == 0))),
                       "unknown": int(np.count_nonzero(mask & ~np.isfinite(moving)))}
    positive = np.flatnonzero(selected)
    na = int(np.count_nonzero(steps < 1 + low.config["lookback_samples"]))
    return {"label": "低进度确认点的姿态×足相对活动", "unit": "记录点",
        "counts": counts, "positive": len(positive), "negative": int(np.count_nonzero(low.values == 0)),
        "not_applicable": na, "unknown": int(np.count_nonzero(low.values < 0))-na,
        "first_confirmed_step": int(steps[positive[0]]) if len(positive) else None}


def _channel(episode: NumericEpisode, name: str, tail: tuple[int, ...]) -> np.ndarray:
    if name not in episode.arrays:
        # Missing is represented internally as NaN, never as a physical zero.
        return np.full((len(episode.steps), *tail), np.nan)
    values = np.asarray(episode.arrays[name], dtype=np.float64)
    if values.shape != (len(episode.steps), *tail):
        raise ValueError(f"{name}: entity/component dimensions differ from binding")
    return values


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
         "not_applicable_count": int(np.count_nonzero(episode.steps < 1 + width)),
         "confirmation_semantics": "window_end_not_physical_onset"})


def position_motion(episode: NumericEpisode) -> dict[str, np.ndarray]:
    """Recorded-point derivatives at the edge end; never bridge missing rows."""
    n, feet = len(episode.steps), len(episode.entities["support"])
    root = _channel(episode, "base_pos_w", (3,))
    world = _channel(episode, "support_pos_w", (feet, 3))
    raw = _channel(episode, "base_quat_w", (4,))
    norm = np.linalg.norm(raw, axis=1)
    valid = np.isfinite(raw).all(axis=1) & (np.abs(norm - 1) <= NUMERIC_CONFIG["quaternion_norm_tolerance"])
    q = np.full_like(raw, np.nan)
    np.divide(raw, norm[:, None], out=q, where=valid[:, None])
    offset = world - root[:, None]
    cross = np.cross(q[:, None, 1:], offset)
    relative = offset - 2*q[:, None, :1]*cross + 2*np.cross(q[:, None, 1:], cross)

    def derivative(points):
        result = np.full_like(points, np.nan)
        edges = episode.adjacent.reshape((-1,) + (1,)*(points.ndim-1))
        result[1:] = np.where(edges, np.diff(points, axis=0)/episode.dt_s, np.nan)
        return result

    return {"root": root, "world": world, "relative": relative, "quaternion": q,
            "root_velocity": derivative(root), "world_velocity": derivative(world),
            "relative_velocity": derivative(relative)}


def force_readouts(episode, prefix, entities, threshold):
    """Current and stored-buffer readouts share one threshold, not one clock."""
    channel = prefix + "_contact_forces_w"
    current = _channel(episode, channel, (len(entities), 3))
    cnorm = np.linalg.norm(current, axis=2)
    cmarker = predicate(cnorm > threshold, np.isfinite(current).all(axis=2))
    history = episode.arrays.get(channel + "_history")
    hnorm = None if history is None else np.linalg.norm(np.asarray(history, dtype=float), axis=3)
    hmarker = (np.full_like(cmarker, -1) if history is None else
               predicate((hnorm > threshold).any(axis=1), np.isfinite(history).all(axis=(1, 3))))
    return cnorm, cmarker, hnorm, hmarker


def measure_behavior(episode: NumericEpisode) -> dict:
    """Stair-shaped behavior operands sharing the point/clock definitions."""
    motion = position_motion(episode)
    p, q = motion["root"], motion["quaternion"]
    steps, dt = episode.steps, episode.dt_s
    expected_end = int(episode.task["outcome"].get("episode_steps", steps[-1]))
    forward = np.asarray(episode.task["progress_direction_world"])
    lateral = np.cross([0., 0., 1.], forward)
    progress = (p - episode.task["origin_world"]) @ forward
    low = low_progress_track(episode, progress)
    up = 1 - 2*(q[:, 1]**2 + q[:, 2]**2)
    w, x, y, z = q.T
    up_axis = np.column_stack((2*(x*z+w*y), 2*(y*z-w*x), up))
    forward_axis = np.column_stack((1-2*(y*y+z*z), 2*(x*y+w*z), 2*(x*z-w*y)))
    heading, heading_na = axis_azimuth(forward_axis, forward)
    up_direction, up_direction_na = axis_azimuth(up_axis, forward)
    tilt = np.degrees(np.arccos(np.clip(up, -1, 1)))
    rotation = np.full(len(steps), np.nan)
    rotation[1:] = np.where(episode.adjacent,
        np.degrees(2*np.arccos(np.clip(np.abs(np.sum(q[:-1]*q[1:], axis=1)), 0, 1))), np.nan)
    forward_z = 2*(q[:, 1]*q[:, 3] - q[:, 0]*q[:, 2])
    elevation = np.degrees(np.arcsin(np.clip(forward_z, -1, 1)))
    force = _channel(episode, "support_contact_forces_w", (len(episode.entities["support"]), 3))
    marker = np.where(np.isfinite(force).all(axis=2),
                      (np.linalg.norm(force, axis=2) > episode.task["contact_threshold_n"]).astype(float), np.nan)
    relative_speed = np.linalg.norm(motion["relative_velocity"], axis=2)
    root_speed = np.linalg.norm(motion["root_velocity"][:, :2], axis=1)
    moving = np.where(np.any(relative_speed >= NUMERIC_CONFIG["low_horizontal_speed_mps"], axis=1), 1.,
                     np.where(np.isfinite(relative_speed).all(axis=1), 0., np.nan))
    low_values = np.where(low.values >= 0, low.values.astype(float), np.nan)
    activity = np.where(low.values == 0, 0., np.where(low.values == 1, moving, np.nan))
    facts = {}

    def scalar(target, key, label, unit, values, mask, mode="mean", applicable=None):
        selected = np.asarray(values)[mask]
        valid = np.isfinite(selected)
        na = 0 if applicable is None else int(np.count_nonzero(mask & ~applicable))
        known = selected[valid]
        value = None
        if len(known):
            value = float({"mean": np.mean, "max": np.max, "min": np.min, "sum": np.sum}[mode](known))
        target[key] = {"label": label, "unit": unit, "value": value,
                       "valid": int(valid.sum()), "unknown": int((~valid).sum())-na,
                       "not_applicable": na}

    width = terminal_window_steps(dt)
    complete_history = (steps[-1]-1)*dt + 1e-9 >= NUMERIC_CONFIG["terminal_window_s"]
    for window, mask in (("all", np.ones(len(steps), dtype=bool)),
                         ("terminal", terminal_window_mask(episode))):
        f = facts[window] = {}
        rows = np.flatnonzero(mask)
        edge_mask = mask if window == "all" else mask & (steps > steps[rows[0]])
        scalar(f, "progress", "终点任务位置", "m", progress, steps == steps[-1])
        scalar(f, "speed", "根部位置差分水平速度", "m/s", root_speed, edge_mask, applicable=steps>1)
        scalar(f, "forward_speed", "正向差分速度", "m/s", motion["root_velocity"] @ np.asarray(episode.task["progress_direction_world"]), edge_mask, applicable=steps>1)
        scalar(f, "lateral_speed", "横向差分速度", "m/s", motion["root_velocity"] @ lateral, edge_mask, applicable=steps>1)
        scalar(f, "time", "记录结束时刻", "s", steps*dt, steps==steps[-1])
        scalar(f, "tilt", "身体上轴最大倾角", "°", tilt, mask, "max")
        scalar(f, "tilt_end", "末端倾角", "°", tilt, steps == steps[-1])
        scalar(f, "heading_end", "身体前轴投影朝向", "°", heading, steps == steps[-1], applicable=~heading_na)
        scalar(f, "up_direction_end", "上轴投影方位", "°", up_direction, steps == steps[-1], applicable=~up_direction_na)
        scalar(f, "forward_elevation", "身体前轴最大仰角", "°", elevation, mask, "max")
        scalar(f, "forward_elevation_min", "身体前轴最小仰角", "°", elevation, mask, "min")
        scalar(f, "height", "根部相对场景原点z", "m", p[:,2]-episode.task["origin_world"][2], mask)
        scalar(f, "rotation", "相邻姿态短弧转角总和", "°", rotation, edge_mask, "sum", applicable=steps>1)
        scalar(f, "inverted", "上轴向下记录占比", "%", np.where(np.isfinite(up), 100*(up < 0), np.nan), mask)
        expected = expected_end if window == "all" else min(expected_end, width+1)
        f["inverted"]["presence"] = (True if np.any(up[mask] < 0) else
            False if np.isfinite(up[mask]).all() and len(rows) == expected and steps[-1] == expected_end else None)
        applicable = steps >= 1 + low.config["lookback_samples"]
        scalar(f, "low_progress", "低正向进度确认占比", "%", 100*low_values, mask, applicable=applicable)
        scalar(f, "active", "低进度且足相对身体变化占比", "%", 100*activity, mask, applicable=applicable)
        change = np.full(len(steps), np.nan)
        change[1:] = np.where(episode.adjacent, np.diff(progress), np.nan)
        scalar(f, "backward", "反向位移累计", "m", np.maximum(-change, 0), edge_mask, "sum", applicable=steps>1)
        # Endpoint displacement requires both requested ends, not merely two finite inner rows.
        delta = progress[rows[-1]] - progress[rows[0]]
        f["net"] = {"label": "净正向位移", "unit": "m", "value": float(delta) if np.isfinite(delta) else None,
                    "valid": int(np.isfinite(delta)), "unknown": int(not np.isfinite(delta)), "not_applicable": 0}
        delta = (p[rows[-1]] - p[rows[0]]) @ lateral
        f["net_lateral"] = {"label": "净横向位移", "unit": "m", "value": float(delta) if np.isfinite(delta) else None,
                            "valid": int(np.isfinite(delta)), "unknown": int(not np.isfinite(delta)), "not_applicable": 0}
        f["duration"] = {"label": "观测跨度", "unit": "s", "value": float((steps[rows[-1]]-steps[rows[0]])*dt),
                         "valid": len(rows), "unknown": int(steps[rows[-1]]-steps[rows[0]]+1-len(rows)), "not_applicable": 0}
        for i, name in enumerate(episode.entities["support"]):
            scalar(f, name+".contact", name+"接触标记占比", "%", 100*marker[:, i], mask)
            scalar(f, name+".world", name+"世界位置差分速度", "m/s", np.linalg.norm(motion["world_velocity"][:, i], axis=1), edge_mask, applicable=steps>1)
            scalar(f, name+".relative", name+"身体相对位置差分速度", "m/s", relative_speed[:, i], edge_mask, applicable=steps>1)
            for axis, xyz in enumerate("xyz"):
                scalar(f, name+"."+xyz, name+"身体相对"+xyz+"均值", "m", motion["relative"][:, i, axis], mask)
        codes = np.sum(np.where(np.isfinite(marker), marker, 0).astype(int) * (1 << np.arange(marker.shape[1])), axis=1)
        joint_valid = np.isfinite(marker).all(axis=1)
        for code in range(1 << len(episode.entities["support"])):
            scalar(f, "contact."+str(code), "+".join(e for i,e in enumerate(episode.entities["support"]) if code & (1<<i)) or "全部未触发",
                   "%", np.where(joint_valid, 100*(codes == code), np.nan), mask)
            if window == "terminal":
                contact = f["contact."+str(code)]
                if not complete_history:
                    contact.update(value=None, valid=0, unknown=0, not_applicable=len(rows))
                else:
                    contact["unknown"] += width+1-len(rows)
    events = []
    drawdown = _drawdown(episode, progress)
    if (episode.task["outcome"]["primary_reason"] == "success" and
            (drawdown["max_drawdown_m"] or 0) >= NUMERIC_CONFIG["progress_increment_m"]
            and drawdown["return_step"] is not None):
        indices = [int(np.searchsorted(steps, drawdown[k])) for k in ("peak_step", "trough_step", "return_step")]
        events.append({"kind": "success_drawdown_return", **drawdown,
            "progress_m": progress[indices].tolist(),
            "source": episode.reference(("base_pos_w",), indices[0], indices[-1]+1)})
    threshold = episode.task["contact_threshold_n"]
    bodies = episode.entities["forbidden"]
    cnorm, current, hnorm, history = force_readouts(episode, "forbidden", bodies, threshold)
    for i, entity in enumerate(bodies):
        if history[-1, i] == 1 and current[-1, i] == 0:
            positive = np.flatnonzero(history[:, i] == 1)
            flags = {name: bool(episode.arrays["termination_"+str(j)][-1])
                     for j, name in enumerate(episode.task["termination_term_order"])
                     if "termination_"+str(j) in episode.arrays and
                     episode.arrays["termination_"+str(j)][-1] in (0, 1)}
            events.append({"kind": "terminal_history_difference", "entity": entity,
                "step": int(steps[-1]), "current_positive": int((current[:, i] == 1).sum()),
                "current_unknown": int((current[:, i] < 0).sum()) + int(steps[-1])-len(steps),
                "history_positive": len(positive), "history_unknown": int((history[:, i] < 0).sum()) + int(steps[-1])-len(steps),
                "current_terminal_n": float(cnorm[-1, i]), "history_terminal_n": hnorm[-1, :, i].tolist(),
                "terminal_flags": flags,
                "source": episode.reference(("forbidden_contact_forces_w", "forbidden_contact_forces_w_history",
                    *("termination_"+str(j) for j in range(len(episode.task["termination_term_order"])))) )})
    first_down = np.flatnonzero(up < 0)
    first_observation = None
    if len(first_down):
        first = int(first_down[0])
        def observed(index):
            return {"step": int(steps[index]), "time_s": float(steps[index]*dt),
                    "tilt_deg": float(tilt[index]),
                    "up_direction_deg": float(up_direction[index]) if np.isfinite(up_direction[index]) else None}
        prior = (observed(first-1) if first and episode.adjacent[first-1]
                 and np.isfinite(up[first-1]) and up[first-1] >= 0 else None)
        first_observation = {**observed(first), "previous": prior,
            "source": episode.reference(("base_quat_w",), first-1 if prior else first, first+1)}
    presence = facts["all"]["inverted"]["presence"]
    facts["all"]["first_down_time"] = {"label":"首次观察上轴向下时刻", "unit":"s",
        "value": first_observation["time_s"] if first_observation else None,
        "valid": int(first_observation is not None), "unknown": int(presence is None),
        "not_applicable": int(presence is False)}
    if len(first_down):
        first = int(first_down[0])
        speed = np.linalg.norm(motion["world_velocity"][:, :, :2], axis=2)
        for i, entity in enumerate(episode.entities["support"]):
            valid = np.isfinite(speed[:, i]) & np.isfinite(force[:, i]).all(axis=1)
            off = predicate((np.linalg.norm(force[:, i], axis=1) <= threshold) &
                            (speed[:, i] < NUMERIC_CONFIG["low_horizontal_speed_mps"]), valid)
            track = PredicateTrack(episode.episode_id, "low_speed_off", (entity,), (), off, {})
            longest = max((r for r in interval_rows(episode, track) if r["state"] == "true"
                           and r["steps_half_open"][0] > steps[first]
                           and (r["sample_count"]-1)*dt >= NUMERIC_CONFIG["progress_window_s"]),
                          key=lambda r:r["sample_count"], default=None)
            if longest:
                a, b = longest["index_half_open"]
                events.append({"kind": "down_before_low_off", "entity": entity,
                    "first_observed_down_step": int(steps[first]), "start_step": int(steps[a]),
                    "end_step": int(steps[b-1]), "samples": b-a, "span_s": float((steps[b-1]-steps[a])*dt),
                    "max_world_xy_speed_mps": float(speed[a:b, i].max()),
                    "source": episode.reference(("base_quat_w", "support_pos_w", "support_contact_forces_w"),
                                                min(first, a-1), b)})
    # Principled posture extrema retain principal-angle meaning; no Euler unwrap.
    return {"episode_id": episode.episode_id, "run_index": episode.run_index, "env_index": episode.env_index,
            "outcome": episode.task["outcome"], "stratum": episode.task["stratum_id"], "cell": episode.task["cell_id"],
            "support": list(episode.entities["support"]), "contact_threshold_n": episode.task["contact_threshold_n"],
            "facts": facts, "events": events, "first_down_observation": first_observation,
            "low_progress_facet": _low_progress_facet(steps, low, up, moving),
            "terminal_motion_complete": bool(steps[-1] == expected_end and width > 0 and complete_history
                and facts["terminal"]["forward_speed"]["valid"] == width),
            "source": episode.reference(("base_pos_w", "base_quat_w", "support_pos_w", "support_contact_forces_w")),
            "window": {"terminal_s": NUMERIC_CONFIG["terminal_window_s"], "edge_assignment": "both_adjacent_endpoints_inside_window",
                       "low_progress_support_samples": low.config["lookback_samples"], "confirmation_not_onset": True}}
