"""Export first-stage N5 numeric material, without reading process reports.

Only raw numeric manifests/chunks are inputs. P3R's fixed outcome-based selection
is reused unchanged. Full readable JSON is for BOTH reviewers; NPZ is an audit
copy, not a substitute for delivering the JSON. No model calls or scoring.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
P3R = HERE.parent / "P3R" / "confirmation_raw_material.py"
spec = importlib.util.spec_from_file_location("p3r_raw_selection", P3R)
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)
sys.path.insert(0, str(ROOT / "src"))

STAIR_FIELDS = ("valid", "episode_length", "base_pos_w", "base_quat_w", "joint_pos",
                "support_pos_w", "support_lin_vel_w", "support_contact_forces_w",
                "forbidden_contact_forces_w", "support_contact_forces_w_history",
                "forbidden_contact_forces_w_history", "done_mask", "wrapper_done_mask")
PUSH_FIELDS = ("valid", "control_step", "episode_index", "pose", "planar_motion",
               "conditions", "failure", "timeout", "event_steps", "event_verified",
               "observed_dwell", "observed_recovery_step")
STATIC_FIELDS = ("terrain_env_origins_w", "terrain_types")
QUESTIONS = ["完整样本分母与原始结局", "主导失败中常见与不常见的测量过程",
             "相容成功反例是否也出现同一过程及匹配范围", "有原始定位依据的主要过程关系或关键量",
             "限制判断的缺观测和采样范围"]
# Descriptive arithmetic queries, not semantic labels or necessary-fact answers.
QUERY = {"x_window_s": 0.5, "x_window_gain_m": 0.02,
         "unloaded_speed_mps": 0.05, "unloaded_duration_s": 0.5, "signed_vz_mps": 0.05}


# Facts delivery has a narrower terminal view than the immutable raw export.
# Reviewed against StairEpisodeOutcome.to_public_dict and Push runtime
# episode_records/classify_recovery_episode, including the R2 velocity records.
# Native terminal outcomes and event latch state are provenance, not independent
# process diagnoses. In particular, Stair diagnostic_flags is not a measurement.
STAIR_FACT_TERMINAL_FIELDS = (
    "episode_steps", "primary_reason", "success", "max_progress_m", "progress_fraction",
)
PUSH_FACT_TERMINAL_FIELDS = (
    "episode_steps", "stratum", "control_only", "planned_push_count", "push_events",
    "min_base_height_m", "max_pitch_rad", "min_pitch_rad", "max_abs_roll_rad",
    "max_abs_pitch_rad", "max_xy_drift_m", "final_planar_speed_mps",
    "final_recovery_condition", "success", "control_success", "control_failure_label",
    "push_event_count", "recovered_event_count", "invalid_injection_count", "failure_label",
    "mean_action_l2", "mean_action_rate_l2",
)
PUSH_FACT_EVENT_FIELDS = (
    "event_index", "control_step", "time_s", "requested_delta_v_world_mps",
    "observed_delta_v_world_mps", "velocity_before_world_6d", "velocity_after_write_world_6d",
    "velocity_after_next_physics_step_world_6d", "injection_verified", "recovered",
    "recovery_control_step", "recovery_latency_s", "recovery_censored",
)


def facts_episode_projection(task: str, row: Mapping) -> dict:
    """Copy one export episode row/header with only reviewed terminal fields.

    This delivery-only helper never changes the raw exporter or its archives.
    Preserve values, missing keys, event order, and native outcome/latch semantics;
    exclude unreviewed terminal/event fields rather than filtering key substrings.
    Non-terminal row fields (including direct_measurements) are copied unchanged;
    their facts presentation is the caller's responsibility. Push here means the
    preregistered velocity-increment profile supported by this exporter.
    """
    if task not in ("stair", "push"):
        raise ValueError(f"unsupported facts episode task: {task!r}")
    if not isinstance(row, Mapping):
        raise TypeError("facts episode row must be a mapping")
    if row.get("task", task) != task:
        raise ValueError("facts episode task differs from row task")
    projected = deepcopy(dict(row))
    if "original_terminal_record" not in row:
        return projected
    terminal = row["original_terminal_record"]
    if not isinstance(terminal, Mapping):
        raise TypeError("facts original_terminal_record must be a mapping")
    fields = STAIR_FACT_TERMINAL_FIELDS if task == "stair" else PUSH_FACT_TERMINAL_FIELDS
    kept = {key: deepcopy(value) for key, value in terminal.items() if key in fields}
    if task == "push" and "push_events" in terminal:
        events = terminal["push_events"]
        if not isinstance(events, (list, tuple)) or any(not isinstance(event, Mapping) for event in events):
            raise TypeError("facts push_events must be a sequence of mappings")
        kept["push_events"] = [
            {key: deepcopy(value) for key, value in event.items() if key in PUSH_FACT_EVENT_FIELDS}
            for event in events
        ]
    projected["original_terminal_record"] = kept
    return projected


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def plain(value):
    if isinstance(value, Mapping):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plain(value), ensure_ascii=False, separators=(",", ":"),
                               allow_nan=False) + "\n", encoding="utf-8")
    return {"file": path.name, "sha256": sha(path), "bytes": path.stat().st_size}


def read_manifest(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != "true_rl_agent.numeric_evidence.v1" or not value.get("complete"):
        raise ValueError(f"not a complete raw numeric manifest: {path}")
    return value


def load_run(path, manifest, task, run=None):
    """Verify original bytes/schema; never concatenate a partly missing column."""
    chunks = [c for c in manifest["chunks"] if run is None or c["metadata"]["run_index"] == run]
    wanted = set(STAIR_FIELDS if task == "stair" else PUSH_FIELDS)
    terms = chunks[0]["metadata"].get("termination_term_order", []) if chunks else []
    if task == "stair":
        wanted.update(f"termination_{i}" for i in range(len(terms)))
    values, static, sources = defaultdict(list), {}, []
    clock = "episode_length" if task == "stair" else "control_step"
    offset = 0
    for chunk in chunks:
        file = path.parent / chunk["file"]
        if sha(file) != chunk["sha256"]:
            raise ValueError(f"raw chunk hash mismatch: {file}")
        if task == "stair" and chunk["metadata"].get("termination_term_order") != terms:
            raise ValueError("termination column order changed within run")
        with np.load(file, allow_pickle=False) as archive:
            if "valid" not in archive or clock not in archive:
                raise ValueError("raw validity/clock missing")
            count = len(archive["valid"])
            for name in wanted | set(STATIC_FIELDS):
                if name not in archive:
                    continue
                a = archive[name]
                declared = chunk["arrays"].get(name)
                if declared != {"shape": list(a.shape), "dtype": a.dtype.str}:
                    raise ValueError(f"raw field schema mismatch: {file.name}/{name}")
                if name in STATIC_FIELDS:
                    if name in static and not np.array_equal(static[name], a):
                        raise ValueError(f"static field changed within run: {name}")
                    static[name] = a.copy()
                else:
                    if a.shape[:2] != archive["valid"].shape:
                        raise ValueError(f"raw field clock/env axes differ: {name}")
                    if values[name] and (a.dtype != values[name][0].dtype or a.shape[1:] != values[name][0].shape[1:]):
                        raise ValueError(f"raw field dtype/body axes changed within run: {name}")
                    values[name].append(a.copy())
            sources.append({"file": chunk["file"], "sha256": chunk["sha256"],
                            "archive_rows": [offset, offset + count - 1], "arrays": chunk["arrays"]})
            offset += count
    missing = sorted(name for name in wanted if len(values[name]) != len(chunks) or not chunks)
    arrays = {name: np.concatenate(parts) for name, parts in values.items() if name not in missing}
    if task == "stair":
        missing += [name for name in STATIC_FIELDS if name not in static]
    return arrays, static, terms, missing, sources


def episode_arrays(arrays, env, clock, expected_steps):
    valid = arrays["valid"][:, env]
    raw = {name: a[valid, env].copy() for name, a in arrays.items()}
    steps = raw[clock]
    if not np.array_equal(steps, np.arange(1, expected_steps + 1)):
        raise ValueError("first-episode valid clock differs from original terminal length")
    if any(a.dtype.kind == "f" and not np.isfinite(a).all() for a in raw.values()):
        raise ValueError("non-finite value in valid original measurements")
    return raw, np.flatnonzero(valid)


def ranges(curves, steps):
    result = {}
    for name, a in curves.items():
        flat = a.reshape(len(steps), -1)
        columns = []
        for j in range(flat.shape[1]):
            keep = np.isfinite(flat[:, j])
            columns.append(legacy.extrema(flat[keep, j], steps[keep]) if keep.any() else None)
        result[name] = {"component_shape": list(a.shape[1:]), "c_order_components": columns}
    return result


def stair_measurements(raw, origin, surface, signals, dt, terms):
    """Direct geometry/arithmetic only; no production behavior/reducer import."""
    steps = raw["episode_length"]
    curves, predicates, drawdown_witness = {}, {}, None
    if "base_pos_w" in raw:
        world_x = raw["base_pos_w"][:, 0].astype(float)
        end = int(np.argmax(np.maximum.accumulate(world_x) - world_x))
        start = int(np.argmax(world_x[:end+1]))
        drawdown_witness = {"start_step": int(steps[start]), "end_step": int(steps[end]),
                            "start_x_w_m": float(world_x[start]), "end_x_w_m": float(world_x[end]),
                            "drawdown_m": float(world_x[start] - world_x[end])}
    if origin is not None and "base_pos_w" in raw:
        pos = raw["base_pos_w"].astype(float) - origin
        curves["root_xyz_terrain_m"] = pos
        curves["root_x_drawdown_m"] = np.maximum.accumulate(pos[:, 0]) - pos[:, 0]
        width = math.ceil(QUERY["x_window_s"] / dt)
        gain = np.full(len(steps), np.nan)
        for i in range(width, len(steps)):
            gain[i] = pos[i-width:i+1, 0].max() - pos[i-width, 0]
        curves["root_x_window_max_minus_start_m"] = gain
        predicates["root_x_window_max_minus_start_lt_query_m"] = gain < QUERY["x_window_gain_m"]
    if "base_quat_w" in raw:
        w, x, y, z = raw["base_quat_w"].astype(float).T
        curves["root_roll_pitch_rad"] = np.column_stack((np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y)),
                                                        np.arcsin(np.clip(2*(w*y-z*x), -1, 1))))
        curves["root_up_axis_world_z"] = 1-2*(x*x+y*y)
        predicates["abs_roll_or_pitch_gt_protocol_rad"] = (np.abs(curves["root_roll_pitch_rad"]).max(axis=1)
                                                          > signals["base_orientation_instability"]["threshold"])
        threshold = signals["base_orientation_instability"]["threshold"]
        for index, axis in enumerate(("roll", "pitch")):
            predicates[f"root_{axis}_gt_protocol_rad"] = curves["root_roll_pitch_rad"][:, index] > threshold
            predicates[f"root_{axis}_lt_negative_protocol_rad"] = curves["root_roll_pitch_rad"][:, index] < -threshold
        predicates["root_up_axis_world_z_lt_zero"] = curves["root_up_axis_world_z"] < 0
    if "support_lin_vel_w" in raw:
        vel = raw["support_lin_vel_w"].astype(float)
        curves.update(support_planar_speed_mps=np.linalg.norm(vel[..., :2], axis=-1),
                      support_speed3d_mps=np.linalg.norm(vel, axis=-1), support_signed_vz_mps=vel[..., 2])
    if origin is not None and "support_pos_w" in raw:
        pos = raw["support_pos_w"].astype(float)-origin
        heights = np.asarray([[surface.sample(progress_m=float(p[0]), lateral_m=float(p[1])).height_m
                               for p in row] for row in pos])
        curves.update(support_xyz_terrain_m=pos, support_surface_height_terrain_m=heights,
                      support_reference_clearance_m=pos[..., 2]-heights)
    for role in ("support", "forbidden"):
        force = f"{role}_contact_forces_w"
        if force in raw:
            curves[f"{role}_force_norm_n"] = np.linalg.norm(raw[force].astype(float), axis=-1)
        history = force + "_history"
        if history in raw:
            curves[f"{role}_buffer_max_force_norm_n"] = np.linalg.norm(raw[history].astype(float), axis=-1).max(axis=1)
    if "support_force_norm_n" in curves:
        loaded = curves["support_force_norm_n"] > signals["slip"]["contact_force_threshold"]
        predicates["support_force_gt_protocol_n"] = loaded
        predicates["all_support_force_le_protocol_n"] = ~loaded.any(axis=1)
        if "support_planar_speed_mps" in curves:
            predicates["loaded_planar_speed_gt_protocol_mps"] = loaded & (curves["support_planar_speed_mps"] > signals["slip"]["speed_threshold"])
            predicates["unloaded_speed3d_le_query_mps"] = ~loaded & (curves["support_speed3d_mps"] <= QUERY["unloaded_speed_mps"])
            predicates["unloaded_vz_gt_query_mps"] = ~loaded & (curves["support_signed_vz_mps"] > QUERY["signed_vz_mps"])
            predicates["unloaded_vz_lt_negative_query_mps"] = ~loaded & (curves["support_signed_vz_mps"] < -QUERY["signed_vz_mps"])
        if "support_reference_clearance_m" in curves:
            predicates["unloaded_clearance_lt_protocol_m"] = ~loaded & (curves["support_reference_clearance_m"] < signals["insufficient_clearance"]["threshold"])
    for role in ("support", "forbidden"):
        key = f"{role}_buffer_max_force_norm_n"
        if key in curves:
            threshold = signals["slip"]["contact_force_threshold"] if role == "support" else signals["forbidden_contact"]["force_threshold_n"]
            predicates[f"{role}_buffer_force_gt_protocol_n"] = curves[key] > threshold
    intervals = {name: [legacy.spans(column, steps) for column in mask.reshape(len(steps), -1).T]
                 for name, mask in predicates.items()}
    marks = {name: legacy.spans(raw[name], steps) for name in ("done_mask", "wrapper_done_mask") if name in raw}
    marks.update({term: legacy.spans(raw[f"termination_{i}"], steps) for i, term in enumerate(terms)
                  if f"termination_{i}" in raw})
    return curves, {"curve_extrema": ranges(curves, steps), "threshold_intervals_by_component": intervals,
                    "raw_termination_intervals": marks,
                    "root_x_max_drawdown_witness": drawdown_witness,
                    "focus_operand_missing_fields": [name for name in ("base_pos_w", "base_quat_w") if name not in raw],
                    "root_first_last_xyz_terrain_m": ([curves["root_xyz_terrain_m"][i].tolist() for i in (0, -1)]
                                                      if "root_xyz_terrain_m" in curves else None)}


def push_focus_operands(raw, terminal, meta, *, include_window_peak=False):
    """Lossless raw condition runs and fixed event landmarks, without new labels."""
    steps = raw["control_step"]
    condition_runs = None
    if "conditions" in raw:
        conditions = raw["conditions"]
        edges = np.r_[0, np.flatnonzero(np.any(conditions[1:] != conditions[:-1], axis=1)) + 1, len(steps)]
        condition_runs = [[int(steps[start]), int(steps[end-1]), conditions[start].tolist()]
                          for start, end in zip(edges[:-1], edges[1:])]
        restored = np.concatenate([np.repeat(np.asarray(value, dtype=conditions.dtype)[None, :], end-start+1, axis=0)
                                   for start, end, value in condition_runs])
        if not np.array_equal(restored, conditions):
            raise ValueError("full-episode raw conditions RLE is not lossless")
    wanted, events = {int(terminal["episode_steps"])}, []
    for event in terminal["push_events"]:
        anchor = event["control_step"]
        window_steps = steps[(steps > anchor) & (steps <= anchor + meta["time"]["recovery_window_steps"])]
        end = int(window_steps[-1]) if len(window_steps) else None
        wanted.update((anchor, anchor+1))
        if end is not None:
            wanted.add(end)
        landmark = {"event_index": event["event_index"], "anchor_step": anchor,
                    "first_response_step": anchor+1, "observed_window_end_step": end}
        if include_window_peak:
            positions = np.flatnonzero((steps > anchor) & (steps <= anchor + meta["time"]["recovery_window_steps"]))
            peak = (int(steps[positions[np.argmax(raw["pose"][positions, 3])]])
                    if len(positions) and "pose" in raw else None)
            landmark["window_first_speed_maximum_step"] = peak
            if peak is not None:
                wanted.add(peak)
        events.append(landmark)
    positions = {int(step): index for index, step in enumerate(steps)}
    samples = [{"control_step": step,
                **{name: raw[name][positions[step]].tolist() if name in raw else None
                   for name in ("pose", "planar_motion")}} for step in sorted(wanted) if step in positions]
    return {"first_step": int(steps[0]), "last_step": int(steps[-1]), "row_count": len(steps),
            "conditions_rle": condition_runs, "event_landmarks": events, "terminal_step": int(terminal["episode_steps"]),
            "landmark_samples": samples, "requested_steps_not_observed": sorted(wanted - set(positions)),
            "missing_fields": [name for name in ("conditions", "pose", "planar_motion") if name not in raw]}


def push_measurements(raw, terminal, meta, *, include_window_peak=False):
    steps, p = raw["control_step"], meta["parameters"]
    curves = {name: raw[name].astype(float) for name in ("pose", "planar_motion") if name in raw}
    events, checks = [], []
    focus_operands = push_focus_operands(raw, terminal, meta, include_window_peak=include_window_peak)
    if "pose" not in raw or "conditions" not in raw:
        return curves, {"events": [], "missing_operands": True, "focus_raw_operands": focus_operands}
    pose = raw["pose"]
    bound = lambda name: np.asarray(p[name], dtype=pose.dtype)
    good = np.column_stack((pose[:, 3] <= bound("recovery_velocity_mps"), pose[:, 4] <= bound("recovery_angular_velocity_radps"),
                            abs(pose[:, 1]) <= bound("recovery_tilt_rad"), abs(pose[:, 2]) <= bound("recovery_tilt_rad"),
                            pose[:, 0] >= bound("recovery_height_m")))
    if not np.array_equal(good, raw["conditions"]):
        raise ValueError("raw Push conditions disagree with native-dtype pose comparison")
    for event in terminal["push_events"]:
        anchor = event["control_step"]
        window = (steps > anchor) & (steps <= anchor+meta["time"]["recovery_window_steps"])
        ws = steps[window]
        spans = legacy.spans(good[window].all(axis=1), ws) if len(ws) else []
        dwell = meta["time"]["required_dwell_steps"]
        latch = next((start+dwell-1 for start, end in spans if end-start+1 >= dwell), None) if event["injection_verified"] else None
        checks.append(latch == event["recovery_control_step"])
        events.append({"event_index": event["event_index"], "anchor_step": anchor,
                       "window_observed_steps": ws.tolist(), "stable_intervals": spans,
                       "condition_false_intervals": {name: legacy.spans(~good[window, i], ws) if len(ws) else []
                                                     for i, name in enumerate(meta["condition_fields"])},
                       "first_dwell_end_step_by_arithmetic": latch,
                       "pre_latch_stable_intervals": [[a, min(b, latch) if latch is not None else b] for a, b in spans if latch is None or a <= latch],
                       "window_extrema": ranges({k: v[window] for k, v in curves.items()}, ws) if len(ws) else {}})
    return curves, {"curve_extrema": ranges(curves, steps), "events": events,
                    "arithmetic_latch_equals_original_record": checks, "focus_raw_operands": focus_operands}


def field_info(arrays):
    return {name: {"dtype": a.dtype.str, "shape": list(a.shape),
                   "c_order_bytes_sha256": hashlib.sha256(a.tobytes()).hexdigest()} for name, a in arrays.items()}


def export_episode(output, row, raw, static, curves, definitions, source_rows, part_steps):
    dest = output / "episodes" / row["episode_id"]
    dest.mkdir(parents=True)
    npz = dest / "original_values.npz"
    np.savez_compressed(npz, **{f"raw__{k}": v for k, v in raw.items()}, **{f"static__{k}": v for k, v in static.items()})
    clock = "episode_length" if row["task"] == "stair" else "control_step"
    parts = []
    for start in range(0, len(raw[clock]), part_steps):
        stop = start+part_steps
        part = {"episode_id": row["episode_id"], "steps": raw[clock][start:stop],
                "raw": {k: a[start:stop] for k, a in raw.items()},
                "direct_curves": {k: a[start:stop] for k, a in curves.items()}}
        parts.append(write(dest / f"values_{len(parts):03d}.json", part))
    metadata = {"schema_version": "p3new.raw_episode.v1", "episode": row,
                "definitions": definitions, "raw_fields": field_info(raw), "static_values": static,
                "static_fields": field_info(static), "direct_curve_fields": field_info(curves),
                "direct_curve_null": "undefined before enough preceding samples; never an original zero",
                "parts": parts, "parts_required": "ALL parts in order, same bytes/content for both reviewers; no downsampling",
                "original_values_npz": {"file": npz.name, "sha256": sha(npz), "bytes": npz.stat().st_size}}
    write(dest / "episode.json", metadata)
    # Verify the delivered readable numeric values against their original dtype.
    decoded = [json.loads((dest / p["file"]).read_text(encoding="utf-8")) for p in parts]
    with np.load(npz, allow_pickle=False) as archive:
        for name, original in raw.items():
            restored = np.concatenate([np.asarray(part["raw"][name], dtype=original.dtype) for part in decoded])
            if not np.array_equal(restored, original) or not np.array_equal(archive[f"raw__{name}"], original):
                raise ValueError("readable JSON/NPZ differs from original selected values")
    return {"episode_id": row["episode_id"], "parts": len(parts), "original_values_checked": sum(a.size for a in raw.values()),
            "archive_row_indices": source_rows.tolist()}


def source_rows(task, manifest, source_id):
    result = []
    seeds = manifest["metadata"]["seeds"] if task == "stair" else [
        {"run_index": manifest["identity"]["run_index"], "outcomes": manifest["metadata"]["runtime_episode_records"]}]
    for seed in seeds:
        for env, terminal in enumerate(seed["outcomes"]):
            scope = terminal["bin_id"].split(".", 1)[1].rsplit("_level_", 1)[0] if task == "stair" else terminal["stratum"]
            result.append({"id": f"r{seed['run_index']}.e{env}", "task": task, "run": seed["run_index"], "env": env,
                           "scope": scope, "cell": terminal["bin_id"] if task == "stair" else scope,
                           "success": terminal["success"], "outcome": terminal["primary_reason"] if task == "stair" else terminal["failure_label"],
                           "source_id": source_id, "steps": terminal["episode_steps"], "terminal": terminal})
    return result


def stair_definitions(meta, cell, terms):
    return {"coordinates": meta["coordinates"], "clock": "one-based pre-reset control-step end; all valid samples through original T",
            "bodies": meta["runtime_projection"]["resolved_roles"], "termination_term_order": terms,
            "joint_names": meta["runtime_projection"]["joint_names"], "joint_pos_unit": "rad",
            "field_axes": {"base_pos_w": "step,xyz", "base_quat_w": "step,wxyz", "joint_pos": "step,joint_names",
                           "support_vectors": "step,support_effector,xyz", "forbidden_vectors": "step,forbidden_contact,xyz",
                           "history_vectors": "step,history_slot,role_body,xyz", "termination_and_validity": "step"},
            "curve_units": "*_m: metre; *_mps: metre/second; *_n: newton; *_rad: radian; up_axis_world_z: unitless",
            "protocol": meta["semantics"], "contact_history": meta.get("contact_force_history", {"status": "not_recorded"}),
            "geometry": plain(vars(cell.spec)), "terrain_type_index": cell.terrain_type_index,
            "surface_calculation": "task-owned analytic C1/C2 geometry query at the body's world position minus terrain origin; reference point clearance, not toe skin or contact-pair truth",
            "arithmetic_queries": QUERY, "arithmetic_query_authority": "descriptive queries only; not necessary facts, behavior labels, or PASS",
            "window_definition": "root_x_window_max_minus_start uses inclusive samples [step-ceil(x_window_s/dt),step]; null until full width",
            "drawdown_witness_definition": "maximum earlier world-x minus later world-x; earliest maximizing end, then earliest preceding maximum; raw world-x endpoints; terrain origin cancels. Zero gives the first step twice; the interval is not a claim of monotonic retreat",
            "signed_attitude_intervals": "root_roll/pitch strictly > protocol base_orientation_instability threshold or strictly < its negative; root_up_axis_world_z strictly < 0. These are arithmetic intervals, not behavioral or causal labels",
            "interval_definition": "inclusive control steps; one list per C-order component; threshold intervals are not automatically sustained/confirmed; retain protocol durations",
            "buffer_axes": ["control_step", "history_slot", "bound_body", "xyz"],
            "body_vector_axes": ["control_step", "bound_body", "xyz"],
            "sample_limit": "control-step measurements, not continuous motion; history slots lack observed timestamps/validity and are not a full physics trajectory"}


def push_definitions(meta, *, include_window_peak=False):
    keep = ("schema_version", "profile", "capture_point", "time", "pose_fields", "condition_fields", "planar_motion_fields",
            "direction_status", "dwell_scope", "coordinates", "parameters", "contact_status", "episode_coverage")
    return {**{k: meta[k] for k in keep if k in meta},
            "field_axes": {"pose": "step,pose_fields", "planar_motion": "step,planar_motion_fields", "conditions": "step,condition_fields",
                           "event_steps/event_verified/observed_dwell/observed_recovery_step": "step,event_index",
                           "control_step/episode_index/valid/failure/timeout": "step"},
            "integer_units": "control/event/recovery steps and observed dwell are control-step indices/counts; -1 event/recovery values are unobserved sentinels, not time",
            "dwell_detail": "eligible steps: 0 < step-anchor <= window; first dwell is latched once. Max dwell/interruption before first latch differs from all-window stable/false spans. Per-event post-latch stops at the next actual injection anchor (pre-write sample); whole-episode post-latch may include later injections.",
            "velocity_event_clock": "anchor sample precedes velocity write; first response is anchor+1 if recorded. Delta-v and before/after-write vectors are original event receipts, not continuous force sensors.",
            "focus_raw_operands": {"conditions_rle": "lossless [inclusive_start_step,inclusive_end_step,original_conditions_in_condition_fields_order] for every valid first-episode row, including outside recovery windows; not new rule labels",
                                   "landmark_samples": "original pose/planar_motion values at each actual event anchor, anchor+1, last observed step with 0<step-anchor<=window, and original terminal T; coincident steps stored once; missing requested steps listed, never zero-filled" + ("; also the earliest step attaining maximum pose.planar_speed in each observed recovery window, with ALL pose and planar_motion components from that same row" if include_window_peak else ""),
                                   "scope": "full conditions are supplied; pose/planar_motion at other unselected-episode steps remain recorded in NPZ but not_supplied here, not not_recorded",
                                   "missing_fields": "absent from supplied raw arrays; conditions_rle or landmark value is null, never a false/zero substitute"},
            "sample_limit": "all recorded first-episode control steps; no unrecorded foot/contact/full-body quantities supplied"}


def verify_confirmation_source(task, manifest, spec, *, run_index=None):
    """Bind the original numeric capture to the frozen task/cohort, before selection."""
    identity, meta = manifest["identity"], manifest["metadata"]
    expected_task = {"stair": "unitree_stair_climbing", "push": "unitree_push_recovery"}[task]
    if identity["task_id"] != expected_task or identity["checkpoint_hash"] != "sha256:" + spec["checkpoint"]["sha256"]:
        raise ValueError("confirmation numeric task/checkpoint differs")
    # Push raw manifests do not store runner_seed. The complete preflight binding
    # includes seed_cohort_hash; do not pretend to have verified an absent field.
    if identity["evaluation_binding"] != spec["current_evaluation_binding"]:
        raise ValueError("confirmation numeric evaluation/cohort binding differs")
    envs, runs = spec["numeric_eval_envs"], spec["numeric_public_dev_seed_runs"]
    if runs != 3 or not manifest["chunks"] or any(c["arrays"]["valid"]["shape"][1] != envs for c in manifest["chunks"]):
        raise ValueError("confirmation numeric run/env coverage differs")
    if task == "stair":
        seeds = meta["seeds"]
        if (identity["splits"] != ["public_dev"] or identity["seed_namespace"] != spec["cohort"]["namespace"]
                or [s["run_index"] for s in seeds] != list(range(1, runs + 1))
                or any(s["split"] != "public_dev" or len(s["outcomes"]) != envs for s in seeds)
                or {c["metadata"]["run_index"] for c in manifest["chunks"]} != set(range(1, runs + 1))):
            raise ValueError("confirmation Stair cohort/run/outcome coverage differs")
    else:
        if (identity["split"] != "public_dev" or identity["run_index"] != run_index
                or run_index not in range(1, runs + 1) or meta["num_envs"] != envs
                or [r["env_id"] for r in meta["runtime_episode_records"]] != list(range(envs))):
            raise ValueError("confirmation Push split/run/env coverage differs")


def verify_confirmation_sources(manifests, card):
    if len(manifests) != 4:
        raise ValueError("confirmation requires one Stair and three Push manifests")
    verify_confirmation_source("stair", manifests[0], card["tasks"]["stair"])
    for run, manifest in enumerate(manifests[1:], 1):
        verify_confirmation_source("push", manifest, card["tasks"]["push"], run_index=run)
        for field in ("subject_manifest_hash", "evaluation_plan_hash", "resolved_cell_manifest_hash",
                      "training_binding_hash", "program_source_checksum", "runtime_dynamics_hash",
                      "resolved_protocol_hash", "backend_execution_environment_verification"):
            if manifest["identity"][field] != manifests[1]["identity"][field]:
                raise ValueError("confirmation Push sources differ: " + field)


def export(args):
    frozen_inputs = None
    if args.purpose == "confirmation":
        card_path = getattr(args, "capture_card", None)
        protocol_path = getattr(args, "review_protocol", None)
        if card_path is None or protocol_path is None:
            raise ValueError("confirmation requires its frozen capture card and review protocol")
        card = json.loads(card_path.read_bytes())
        if card.get("status") != "FROZEN":
            raise ValueError("confirmation capture card is not frozen")
        protocol_hash = sha(protocol_path)
        if card.get("review_protocol", {}).get("sha256") != protocol_hash:
            raise ValueError("review protocol differs from the capture freeze")
        frozen_inputs = {"capture_card": {"sha256": sha(card_path)},
                         "review_protocol": {"sha256": protocol_hash}}
    source_files = [Path(__file__), P3R] + [ROOT / "src/true_rl_agent/tasks/stair_climbing" / name for name in
        ("terrain_evidence.py", "rough_surface.py", "ordered_evaluation.py", "unified_evaluation.py", "unified_training.py")]
    source_hashes = {p.relative_to(ROOT).as_posix(): sha(p) for p in source_files}
    paths = [args.stair, *args.push]
    manifest_hashes = [sha(path) for path in paths]
    manifests = [read_manifest(path) for path in paths]
    if frozen_inputs is not None:
        verify_confirmation_sources(manifests, card)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    rows = source_rows("stair", manifests[0], "stair_source")
    for i, manifest in enumerate(manifests[1:], 1):
        if manifest["metadata"]["parameters"]["mechanism"] != "world_planar_velocity_increment":
            raise ValueError("exporter supports the preregistered Push velocity profile only")
        rows += source_rows("push", manifest, f"push_source_{i:02d}")
    groups = {task: [r for r in rows if r["task"] == task] for task in ("stair", "push")}
    selected = [r for task in groups for r in legacy.select(groups[task], task)]
    keys = {(r["task"], r["id"]) for r in selected}
    if len(keys) != 24 or any(len({r["id"] for r in group}) != len(group) for group in groups.values()):
        raise ValueError("duplicate identity or incomplete 24-episode selection")
    aliases = {(r["task"], r["id"]): f"{r['task']}-{i:04d}" for task, group in groups.items()
               for i, r in enumerate(sorted(group, key=lambda r: (r["run"], r["env"])), 1)}
    public_rows, receipts, sources, task_definitions = [], [], [], {}
    from true_rl_agent.tasks.stair_climbing.unified_evaluation import StairUnifiedEvaluationSuite
    from true_rl_agent.tasks.stair_climbing.unified_training import StairUnifiedTrainingProfile
    from true_rl_agent.tasks.stair_climbing.terrain_evidence import stair_surface_query_for_ordered_cell
    meta = manifests[0]["metadata"]
    cells = StairUnifiedEvaluationSuite.from_mapping(meta["evaluation_suite"],
            training_profile=StairUnifiedTrainingProfile.from_mapping(meta["training_profile"])).private_cell_assignment("public_dev")
    for index, (path, manifest) in enumerate(zip(paths, manifests, strict=True)):
        task = "stair" if index == 0 else "push"
        meta = manifest["metadata"]
        runs = [seed["run_index"] for seed in meta["seeds"]] if task == "stair" else [manifest["identity"]["run_index"]]
        source_id = "stair_source" if task == "stair" else f"push_source_{index:02d}"
        for run in runs:
            arrays, static, terms, missing, chunks = load_run(path, manifest, task, run if task == "stair" else None)
            sources.append({"source_id": source_id, "run": run, "manifest_path": str(path.resolve()), "manifest_sha256": sha(path),
                            "identity": manifest["identity"], "chunks": chunks, "missing_fields": missing})
            for record in (r for r in rows if r["task"] == task and r["run"] == run and r["source_id"] == source_id):
                env = record["env"]
                raw, positions = episode_arrays(arrays, env, "episode_length" if task == "stair" else "control_step", record["steps"])
                per_static = {k: a[env].copy() for k, a in static.items()}
                if task == "stair":
                    cell = cells[env]
                    if cell.cell_id != record["cell"] or ("terrain_types" in per_static and int(per_static["terrain_types"]) != cell.terrain_type_index):
                        raise ValueError("Stair raw env/cell/terrain binding mismatch")
                    origin = per_static.get("terrain_env_origins_w")
                    if origin is not None and not np.allclose(origin, cell.expected_origin_m, rtol=0, atol=1e-5):
                        raise ValueError("Stair raw origin differs from bound evaluation cell")
                    dt = next(s["step_dt"] for s in meta["seeds"] if s["run_index"] == run)
                    curves, summary = stair_measurements(raw, origin, stair_surface_query_for_ordered_cell(cell.spec), meta["semantics"]["episode_outcomes"]["signals"], dt, terms)
                    definitions = stair_definitions(meta, cell, terms)
                    shared = {k: v for k, v in definitions.items() if k not in ("geometry", "terrain_type_index")}
                    task_definitions.setdefault("stair", {"measurement_definitions": shared, "cells": {}})["cells"][cell.cell_id] = definitions["geometry"]
                    terminal = {k: v for k, v in record["terminal"].items() if k not in ("bin_id", "evaluation_cell_id")}
                else:
                    dt = meta["time"]["step_dt_s"]
                    include_peak = args.profile == "related_v1"
                    curves, summary = push_measurements(raw, record["terminal"], meta, include_window_peak=include_peak)
                    definitions = push_definitions(meta, include_window_peak=include_peak)
                    task_definitions.setdefault("push", {})[source_id] = definitions
                    terminal = {k: v for k, v in record["terminal"].items() if k not in ("env_id", "episode_index")}
                row = {"episode_id": aliases[task, record["id"]], "task": task, "run_group": f"run_{run:02d}",
                       "scope": record["scope"], "cell": record["cell"], "steps": record["steps"], "dt_s": dt,
                       "original_success": record["success"], "original_outcome": record["outcome"], "original_terminal_record": terminal,
                       "missing_fields": missing, "direct_measurements": summary}
                public_rows.append(row)
                if (task, record["id"]) in keys:
                    receipts.append(export_episode(output, row, raw, per_static, curves, definitions, positions, args.part_steps))
            print(f"exported {task} run {run}; missing={missing}", flush=True)
    selection = [{"episode_id": aliases[r["task"], r["id"]], "selection_role": r["selection_role"],
                  "matched_to": aliases[r["task"], r["matched_to"]] if "matched_to" in r else None,
                  "match_level": r.get("match_level")} for r in selected]
    write(output / "selected_episodes.json", selection)
    denominator = [{k: r[k] for k in ("episode_id", "task", "run_group", "scope", "cell", "original_success", "original_outcome")}
                   for r in public_rows]
    write(output / "pool_members.json", denominator)
    # Also preserve arithmetic for pool members outside the twelve focus scopes.
    # Fixed-size JSON chunks contain their values, not links to inaccessible NPZ.
    for task in groups:
        members = [r for r in public_rows if r["task"] == task]
        write(output / "definitions" / f"{task}.json", task_definitions[task])
        for start in range(0, len(members), 32):
            write(output / "pool" / f"{task}_{start//32:03d}.json",
                  {"task": task, "pool_rows": members[start:start+32], "complete_task_denominator": len(members)})
    focus_files = []
    for task in groups:
        scopes = legacy.STAIR_SCOPES if task == "stair" else ["low", "medium", "high", "repeated", "all", "logical_run_01"]
        task_rows = [r for r in public_rows if r["task"] == task]
        for scope in scopes:
            members = [r for r in task_rows if scope == "all" or (scope == "logical_run_01" and r["run_group"] == "run_01") or r["scope"] == scope]
            if not members:
                raise ValueError(f"empty preregistered focus: {task}/{scope}")
            value = {"schema_version": "p3new.raw_focus.v1", "task": task, "focus": scope, "questions": QUESTIONS,
                     "task_definitions": ({**task_definitions[task], "cells": {k: v for k, v in task_definitions[task]["cells"].items()
                                               if any(r["cell"] == k for r in members)}} if task == "stair" else task_definitions[task]),
                     "complete_task_pool_members": [r for r in denominator if r["task"] == task],
                     "scope_denominator": len(members), "scope_outcomes": dict(Counter(r["original_outcome"] for r in members)),
                     "scope_successes": sum(r["original_success"] for r in members), "all_scope_rows": members,
                     "coverage": "every episode in this focus, not only the 12 selected examples; rows are direct summaries, not full trajectories for unselected episodes",
                     "interpretation": "original outcomes and arithmetic evidence only; reviewer independently establishes necessary facts and unknown combinations"}
            file = output / "focus" / f"{task}_{scope}.json"
            write(file, value)
            focus_files.append(str(file.relative_to(output)))
    write(output / "_private" / "sources.json", {"sources": sources, "selection_original_ids": selected,
          "anonymous_mapping": [{"task": t, "original_id": ident, "episode_id": alias} for (t, ident), alias in aliases.items()],
          "json_npz_roundtrip_checks": receipts})
    if source_hashes != {p.relative_to(ROOT).as_posix(): sha(p) for p in source_files}:
        raise ValueError("export/selection/geometry source changed during export; preserve output and rerun")
    materials = []
    for path in sorted(output.rglob("*.json")):
        if "_private" not in path.relative_to(output).parts:
            materials.append({"file": str(path.relative_to(output)).replace("\\", "/"), "sha256": sha(path), "bytes": path.stat().st_size})
    result = {"schema_version": "p3new.raw_export.v1", "purpose": args.purpose, "stage": "facts",
              "status": "EXPORTED_WITH_MISSING_FIELDS" if any(s["missing_fields"] for s in sources) else "EXPORTED_NUMERIC_NOT_REVIEWED",
              "not_semantic_pass": True, "automatic_report_inputs": [], "production_behavior_imports": [],
              "source_hashes": source_hashes, "selection": "unchanged P3R.select/rank; outcome only; once 12/task; no adaptive reselection",
              "private_source_manifest": {"file": "_private/sources.json", "sha256": sha(output / "_private/sources.json")},
              "pool_counts": {t: len(v) for t, v in groups.items()}, "selected_count": len(selection), "focus_count": len(focus_files),
              "part_steps": args.part_steps, "material_profile": args.profile, "readable_materials": materials,
              "selected_raw_values_roundtrip_checked": sum(r["original_values_checked"] for r in receipts),
              "missing_fields_by_source_run": [{k: s[k] for k in ("source_id", "run", "missing_fields")} for s in sources],
              "raw_npz": "original archives retained in place with chunk hashes; selected original-value slices kept beside JSON; never a model-input substitute",
              "media": "not included by this numeric exporter; frozen same-run original windows must be supplied separately",
              "delivery": "both reviewers must receive every required readable JSON part and task definitions; this is not a provider/release manifest"}
    if frozen_inputs is not None:
        if frozen_inputs["capture_card"]["sha256"] != sha(args.capture_card) or frozen_inputs["review_protocol"]["sha256"] != sha(args.review_protocol):
            raise ValueError("confirmation freeze changed during raw export")
        if manifest_hashes != [sha(path) for path in paths]:
            raise ValueError("confirmation numeric manifests changed during raw export")
        result["frozen_inputs"] = frozen_inputs
    if any(name.endswith((".behavior", ".process_report")) for name in sys.modules):
        raise ValueError("automatic behavior/report module imported into first-stage export")
    write(output / "manifest.json", result)
    print(json.dumps({k: result[k] for k in ("status", "pool_counts", "selected_count", "focus_count", "selected_raw_values_roundtrip_checked")}))


def self_test():
    import tempfile
    import unittest

    class RawTests(unittest.TestCase):
        def test_valid_episode_not_wrong_env_or_padding(self):
            a = {"valid": np.array([[1, 1], [1, 0]], bool), "control_step": np.array([[1, 1], [2, 0]]),
                 "pose": np.array([[[0.1], [0.9]], [[0.2], [99.]]], dtype=np.float32)}
            raw, indices = episode_arrays(a, 1, "control_step", 1)
            self.assertEqual(indices.tolist(), [0])
            self.assertEqual(raw["pose"].tolist(), a["pose"][:1, 1].tolist())
            with self.assertRaises(ValueError):
                episode_arrays(a, 1, "control_step", 2)

        def test_missing_and_corrupt_original_chunk(self):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "manifest.json"
                a = {"valid": np.ones((2, 1), bool), "control_step": np.array([[1], [2]]),
                     "pose": np.zeros((2, 1, 6), np.float32)}
                np.savez(path.parent / "one.npz", **a)
                chunk = {"file": "one.npz", "sha256": sha(path.parent / "one.npz"), "metadata": {},
                         "arrays": {k: {"shape": list(v.shape), "dtype": v.dtype.str} for k, v in a.items()}}
                arrays, _, _, missing, _ = load_run(path, {"chunks": [chunk]}, "push")
                self.assertIn("planar_motion", missing)
                self.assertNotIn("planar_motion", arrays)
                chunk["sha256"] = "0"*64
                with self.assertRaises(ValueError):
                    load_run(path, {"chunks": [chunk]}, "push")

        def test_json_original_float32_roundtrip(self):
            with tempfile.TemporaryDirectory() as directory:
                raw = {"control_step": np.array([1, 2]), "pose": np.array([[.1, -.2], [.3, 0]], np.float32)}
                result = export_episode(Path(directory), {"episode_id": "example", "task": "push"}, raw, {}, {}, {}, np.array([0, 1]), 1)
                self.assertEqual(result["original_values_checked"], 6)
                self.assertEqual(result["parts"], 2)

        def test_window_peak_is_first_tie_and_keeps_same_row(self):
            raw = {"control_step": np.arange(1, 6), "pose": np.zeros((5, 6), np.float32),
                   "planar_motion": np.arange(20, dtype=np.float32).reshape(5, 4)}
            raw["pose"][:, 3] = [99, 2, 5, 5, 1]
            terminal = {"episode_steps": 5, "push_events": [{"event_index": 0, "control_step": 1}]}
            meta = {"time": {"recovery_window_steps": 4}}
            old = push_focus_operands(raw, terminal, meta)
            self.assertNotIn("window_first_speed_maximum_step", old["event_landmarks"][0])
            result = push_focus_operands(raw, terminal, meta, include_window_peak=True)
            self.assertEqual(result["event_landmarks"][0]["window_first_speed_maximum_step"], 3)
            sample = next(v for v in result["landmark_samples"] if v["control_step"] == 3)
            self.assertEqual(sample["planar_motion"], raw["planar_motion"][2].tolist())
            raw.pop("pose")
            missing = push_focus_operands(raw, terminal, meta, include_window_peak=True)
            self.assertIsNone(missing["event_landmarks"][0]["window_first_speed_maximum_step"])

        def test_surface_signed_velocity_and_sensor_buffer_are_distinct(self):
            from types import SimpleNamespace
            raw = {"episode_length": np.array([1, 2]), "base_pos_w": np.array([[0., 0., 2.], [1., 0., 2.]]),
                   "support_pos_w": np.array([[[0., 0., 2.]], [[1., 0., 2.]]]),
                   "support_lin_vel_w": np.array([[[3., 4., -12.]], [[0., 0., .1]]]),
                   "support_contact_forces_w": np.zeros((2, 1, 3)),
                   "support_contact_forces_w_history": np.array([[[[0., 0., 0.]], [[0., 0., 2.]]]]*2)}
            signals = {"slip": {"contact_force_threshold": 1., "speed_threshold": .5},
                       "insufficient_clearance": {"threshold": .018}, "forbidden_contact": {"force_threshold_n": 1.}}
            surface = SimpleNamespace(sample=lambda **kwargs: SimpleNamespace(height_m=.75))
            curves, summary = stair_measurements(raw, np.array([0., 0., .5]), surface, signals, .02, [])
            self.assertEqual(curves["support_reference_clearance_m"][0, 0], .75)
            self.assertEqual(curves["support_planar_speed_mps"][0, 0], 5.)
            self.assertEqual(curves["support_speed3d_mps"][0, 0], 13.)
            self.assertEqual(curves["support_signed_vz_mps"][0, 0], -12.)
            intervals = summary["threshold_intervals_by_component"]
            self.assertEqual(intervals["support_force_gt_protocol_n"], [[]])
            self.assertEqual(intervals["support_buffer_force_gt_protocol_n"], [[[1, 2]]])

        def test_drawdown_endpoints_signed_tilt_and_up_axis(self):
            angles = np.array([0., 1., -1., 2., -2.])
            raw = {"episode_length": np.arange(1, 6),
                   "base_pos_w": np.column_stack(([1., 4., 3., 2., 4.], np.zeros((5, 2)))),
                   "base_quat_w": np.column_stack((np.cos(angles/2), np.sin(angles/2), np.zeros((5, 2))))}
            _, summary = stair_measurements(raw, None, None, {"base_orientation_instability": {"threshold": .5}}, .02, [])
            self.assertEqual(summary["root_x_max_drawdown_witness"],
                             {"start_step": 2, "end_step": 4, "start_x_w_m": 4., "end_x_w_m": 2., "drawdown_m": 2.})
            intervals = summary["threshold_intervals_by_component"]
            self.assertEqual(intervals["root_roll_gt_protocol_rad"], [[[2, 2], [4, 4]]])
            self.assertEqual(intervals["root_roll_lt_negative_protocol_rad"], [[[3, 3], [5, 5]]])
            self.assertEqual(intervals["root_pitch_gt_protocol_rad"], [[]])
            self.assertEqual(intervals["root_up_axis_world_z_lt_zero"], [[[4, 5]]])
            _, missing = stair_measurements({"episode_length": np.array([1])}, None, None, {}, .02, [])
            self.assertIsNone(missing["root_x_max_drawdown_witness"])
            self.assertEqual(missing["focus_operand_missing_fields"], ["base_pos_w", "base_quat_w"])

        def test_push_conditions_and_coincident_truncated_landmarks(self):
            raw = {"control_step": np.arange(1, 5), "conditions": np.array([[1]*5, [1]*5, [0, 1, 1, 1, 1], [1]*5], bool),
                   "pose": np.arange(24, dtype=np.float32).reshape(4, 6),
                   "planar_motion": np.arange(16, dtype=np.float32).reshape(4, 4)}
            terminal = {"episode_steps": 4, "push_events": [{"event_index": 0, "control_step": 3}, {"event_index": 1, "control_step": 4}]}
            result = push_focus_operands(raw, terminal, {"time": {"recovery_window_steps": 3}})
            self.assertEqual(result["conditions_rle"], [[1, 2, [True]*5], [3, 3, [False, True, True, True, True]], [4, 4, [True]*5]])
            self.assertEqual([r["control_step"] for r in result["landmark_samples"]], [3, 4])
            self.assertEqual(result["landmark_samples"][0]["planar_motion"], raw["planar_motion"][2].tolist())
            self.assertIsNone(result["event_landmarks"][1]["observed_window_end_step"])
            self.assertEqual(result["requested_steps_not_observed"], [5])
            raw.pop("planar_motion")
            missing = push_focus_operands(raw, terminal, {"time": {"recovery_window_steps": 3}})
            self.assertEqual(missing["missing_fields"], ["planar_motion"])
            self.assertIsNone(missing["landmark_samples"][0]["planar_motion"])

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RawTests))
    return 0 if result.wasSuccessful() else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stair", type=Path, help="raw Stair manifest (all three runs)")
    parser.add_argument("--push", type=Path, nargs=3, help="three raw Push velocity manifests in frozen order")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--purpose", choices=("development", "confirmation"))
    parser.add_argument("--capture-card", type=Path, help="FROZEN capture card, required for confirmation")
    parser.add_argument("--review-protocol", type=Path, help="protocol bound by capture card, required for confirmation")
    parser.add_argument("--part-steps", type=int, default=64, help="fixed consecutive rows per readable part; freeze before use")
    parser.add_argument("--profile", choices=("original", "related_v1"), default="original",
                        help="related_v1 additionally supplies complete earliest-window-speed-peak raw samples; selection/numerics unchanged")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if not all((args.stair, args.push, args.output, args.purpose)) or args.part_steps <= 0:
        parser.error("--stair, three --push manifests, --output, --purpose and positive --part-steps required")
    export(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
