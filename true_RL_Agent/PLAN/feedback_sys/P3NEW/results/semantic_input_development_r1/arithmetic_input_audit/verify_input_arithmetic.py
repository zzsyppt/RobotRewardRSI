"""Read-only independent arithmetic audit of the delivered related_v2 views.

No exporter, packer, production behavior/reducer or previous arithmetic helper is
imported. The only reused task primitive is frozen normative stair geometry.
"""
from __future__ import annotations
import argparse
from dataclasses import fields
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import traceback

import numpy as np

WATCH = {}
CHECKS = []
ABS_TOL = 1e-10


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def watch(path):
    path = Path(path).resolve()
    WATCH.setdefault(str(path), digest(path))
    return path


def read(path):
    return json.loads(watch(path).read_text(encoding="utf-8"))


def compare(path, actual, expected, basis="independent_raw_arithmetic", exact=False):
    differences = []
    count = 0
    def visit(a, e, loc):
        nonlocal count
        if isinstance(e, np.ndarray):
            e = e.tolist()
        if isinstance(a, np.ndarray):
            a = a.tolist()
        if isinstance(e, np.generic):
            e = e.item()
        if isinstance(a, np.generic):
            a = a.item()
        if isinstance(e, dict):
            count += 1
            if not isinstance(a, dict) or set(a) != set(e):
                differences.append({"path": loc, "actual_keys": sorted(a) if isinstance(a, dict) else str(type(a)), "expected_keys": sorted(e)})
                return
            for key in e:
                visit(a[key], e[key], loc + "/" + str(key))
        elif isinstance(e, (list, tuple)):
            count += 1
            if not isinstance(a, (list, tuple)) or len(a) != len(e):
                differences.append({"path": loc, "actual_length": len(a) if isinstance(a, (list, tuple)) else None, "expected_length": len(e)})
                return
            for i, value in enumerate(e):
                visit(a[i], value, loc + "/" + str(i))
        else:
            count += 1
            if isinstance(e, bool) or e is None or isinstance(e, str):
                good = type(a) is type(e) and a == e
            elif isinstance(e, (int, float)):
                good = isinstance(a, (int, float)) and not isinstance(a, bool)
                if good:
                    good = (a == e) if exact or isinstance(e, int) else math.isclose(float(a), float(e), rel_tol=0, abs_tol=ABS_TOL)
            else:
                good = a == e
            if not good:
                differences.append({"path": loc, "actual": a, "expected": e})
    visit(actual, expected, path)
    result = {"path": path, "basis": basis, "objects_and_scalars_checked": count,
              "equal": not differences, "difference_count": len(differences), "differences": differences[:40]}
    CHECKS.append(result)
    return result


def spans(mask, steps):
    result, start, previous = [], None, None
    for active, step in zip(np.asarray(mask, dtype=bool), steps, strict=True):
        step = int(step)
        if active and start is None:
            start = step
        if start is not None and not active:
            result.append([start, previous])
            start = None
        previous = step
    if start is not None:
        result.append([start, previous])
    return result


def extrema(curves, steps):
    result = {}
    for name, a in curves.items():
        a = np.asarray(a)
        values = []
        for column in a.reshape(len(steps), -1).T:
            rows = [(float(v), int(t)) for v, t in zip(column, steps, strict=True) if math.isfinite(float(v))]
            if not rows:
                values.append(None)
                continue
            minimum = min(rows, key=lambda p: p[0])
            maximum = max(rows, key=lambda p: p[0])
            values.append({"min": minimum[0], "min_step": minimum[1], "max": maximum[0], "max_step": maximum[1]})
        result[name] = {"component_shape": list(a.shape[1:]), "c_order_components": values}
    return result


def decode(column, length):
    dtype = np.dtype(column["dtype"])
    shape = column["component_shape"]
    if "by_c_order_component" in column:
        values = [[float(x) if x != "null" else np.nan for x in s.split(",")] for s in column["by_c_order_component"]]
    else:
        values = []
        for intervals in column["inclusive_constant_intervals_by_component"]:
            row = []
            for start, end, value in intervals:
                if start != len(row) + 1 or end < start:
                    raise ValueError("noncontiguous supplied column intervals")
                row.extend([value] * (end-start+1))
            values.append(row)
    if any(len(v) != length for v in values):
        raise ValueError("supplied column length mismatch")
    return np.asarray(values, dtype=dtype).T.reshape((length, *shape))


def stair_expected(raw, static, definitions, dt, geometry_classes):
    Spec, query_surface = geometry_classes
    steps = raw["episode_length"]
    origin = static.get("terrain_env_origins_w")
    signals = definitions["protocol"]["episode_outcomes"]["signals"]
    query = definitions["arithmetic_queries"]
    curves, masks = {}, {}
    witness = None
    if "base_pos_w" in raw:
        x = raw["base_pos_w"][:, 0].astype(float)
        peak, peak_step, best, witness_steps = float(x[0]), int(steps[0]), 0.0, [int(steps[0]), int(steps[0])]
        for value, step in zip(x, steps, strict=True):
            if value > peak:
                peak, peak_step = float(value), int(step)
            if peak - value > best:
                best, witness_steps = peak - float(value), [peak_step, int(step)]
        a, b = witness_steps
        witness = {"start_step": a, "end_step": b, "start_x_w_m": float(x[a-1]), "end_x_w_m": float(x[b-1]), "drawdown_m": best}
        if origin is not None:
            position = raw["base_pos_w"].astype(float)-origin
            curves["root_xyz_terrain_m"] = position
            running = -math.inf
            drawdown = []
            for value in position[:, 0]:
                running = max(running, float(value))
                drawdown.append(running-float(value))
            curves["root_x_drawdown_m"] = np.asarray(drawdown)
            width = math.ceil(query["x_window_s"]/dt)
            gain = np.full(len(steps), np.nan)
            for i in range(width, len(steps)):
                gain[i] = max(position[i-width:i+1, 0])-position[i-width, 0]
            curves["root_x_window_max_minus_start_m"] = gain
            masks["root_x_window_max_minus_start_lt_query_m"] = gain < query["x_window_gain_m"]
    if "base_quat_w" in raw:
        w, x, y, z = raw["base_quat_w"].astype(float).T
        roll = np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y))
        pitch = np.arcsin(np.clip(2*(w*y-z*x), -1, 1))
        curves["root_roll_pitch_rad"] = np.column_stack((roll, pitch))
        curves["root_up_axis_world_z"] = 1-2*(x*x+y*y)
        threshold = signals["base_orientation_instability"]["threshold"]
        masks["abs_roll_or_pitch_gt_protocol_rad"] = np.maximum(abs(roll), abs(pitch)) > threshold
        for name, values in (("roll", roll), ("pitch", pitch)):
            masks[f"root_{name}_gt_protocol_rad"] = values > threshold
            masks[f"root_{name}_lt_negative_protocol_rad"] = values < -threshold
        masks["root_up_axis_world_z_lt_zero"] = curves["root_up_axis_world_z"] < 0
    if "support_lin_vel_w" in raw:
        v = raw["support_lin_vel_w"].astype(float)
        curves["support_planar_speed_mps"] = np.sqrt(v[..., 0]**2+v[..., 1]**2)
        curves["support_speed3d_mps"] = np.sqrt(np.sum(v*v, axis=-1))
        curves["support_signed_vz_mps"] = v[..., 2]
    if origin is not None and "support_pos_w" in raw:
        geometry = definitions["geometry"]
        params = {f.name: geometry[f.name] for f in fields(Spec)}
        for name in ("riser_heights_m", "tread_depths_m", "size_m"):
            params[name] = tuple(params[name])
        spec = Spec(**params)
        if spec.geometry_digest != geometry["geometry_digest"]:
            raise ValueError("geometry digest mismatch")
        surface = query_surface(spec)
        position = raw["support_pos_w"].astype(float)-origin
        height = np.asarray([[surface.sample(progress_m=float(p[0]), lateral_m=float(p[1])).height_m for p in row] for row in position])
        curves["support_xyz_terrain_m"] = position
        curves["support_surface_height_terrain_m"] = height
        curves["support_reference_clearance_m"] = position[..., 2]-height
    for role in ("support", "forbidden"):
        name = role + "_contact_forces_w"
        if name in raw:
            f = raw[name].astype(float)
            curves[role + "_force_norm_n"] = np.sqrt(np.sum(f*f, axis=-1))
        if name + "_history" in raw:
            f = raw[name + "_history"].astype(float)
            curves[role + "_buffer_max_force_norm_n"] = np.sqrt(np.sum(f*f, axis=-1)).max(axis=1)
    if "support_force_norm_n" in curves:
        loaded = curves["support_force_norm_n"] > signals["slip"]["contact_force_threshold"]
        masks["support_force_gt_protocol_n"] = loaded
        masks["all_support_force_le_protocol_n"] = ~loaded.any(axis=1)
        if "support_planar_speed_mps" in curves:
            masks["loaded_planar_speed_gt_protocol_mps"] = loaded & (curves["support_planar_speed_mps"] > signals["slip"]["speed_threshold"])
            masks["unloaded_speed3d_le_query_mps"] = ~loaded & (curves["support_speed3d_mps"] <= query["unloaded_speed_mps"])
            masks["unloaded_vz_gt_query_mps"] = ~loaded & (curves["support_signed_vz_mps"] > query["signed_vz_mps"])
            masks["unloaded_vz_lt_negative_query_mps"] = ~loaded & (curves["support_signed_vz_mps"] < -query["signed_vz_mps"])
        if "support_reference_clearance_m" in curves:
            masks["unloaded_clearance_lt_protocol_m"] = ~loaded & (curves["support_reference_clearance_m"] < signals["insufficient_clearance"]["threshold"])
    for role in ("support", "forbidden"):
        key = role + "_buffer_max_force_norm_n"
        if key in curves:
            threshold = signals["slip"]["contact_force_threshold"] if role == "support" else signals["forbidden_contact"]["force_threshold_n"]
            masks[role + "_buffer_force_gt_protocol_n"] = curves[key] > threshold
    intervals = {name: [spans(col, steps) for col in mask.reshape(len(steps), -1).T] for name, mask in masks.items()}
    termination = {name: spans(raw[name], steps) for name in ("done_mask", "wrapper_done_mask") if name in raw}
    for i, name in enumerate(definitions["termination_term_order"]):
        if f"termination_{i}" in raw:
            termination[name] = spans(raw[f"termination_{i}"], steps)
    return curves, {"curve_extrema": extrema(curves, steps), "threshold_intervals_by_component": intervals,
        "raw_termination_intervals": termination, "root_x_max_drawdown_witness": witness,
        "focus_operand_missing_fields": [k for k in ("base_pos_w", "base_quat_w") if k not in raw],
        "root_first_last_xyz_terrain_m": curves["root_xyz_terrain_m"][[0, -1]].tolist() if "root_xyz_terrain_m" in curves else None}


def push_expected(raw, terminal, definitions):
    steps = raw["control_step"]
    curves = {name: raw[name].astype(float) for name in ("pose", "planar_motion") if name in raw}
    if "pose" not in raw or "conditions" not in raw:
        return curves, {"events": [], "missing_operands": True}
    p, pose = definitions["parameters"], raw["pose"]
    bound = lambda name: np.asarray(p[name], dtype=pose.dtype)
    good = np.column_stack((pose[:, 3] <= bound("recovery_velocity_mps"), pose[:, 4] <= bound("recovery_angular_velocity_radps"),
                          abs(pose[:, 1]) <= bound("recovery_tilt_rad"), abs(pose[:, 2]) <= bound("recovery_tilt_rad"),
                          pose[:, 0] >= bound("recovery_height_m")))
    if not np.array_equal(good, raw["conditions"]):
        raise ValueError("source conditions do not match independently checked native-dtype thresholds")
    result = []
    width, dwell = definitions["time"]["recovery_window_steps"], definitions["time"]["required_dwell_steps"]
    for event in terminal["push_events"]:
        anchor = event["control_step"]
        indices = np.asarray([i for i, step in enumerate(steps) if 0 < int(step)-anchor <= width], dtype=int)
        ws = steps[indices]
        both = [all(bool(v) for v in good[i]) for i in indices]
        intervals = spans(both, ws)
        count, latch = 0, None
        for active, step in zip(both, ws, strict=True):
            count = count+1 if active else 0
            if event["injection_verified"] and count >= dwell:
                latch = int(step)
                break
        result.append({"event_index": event["event_index"], "anchor_step": anchor,
            "window_observed_steps": ws.tolist(), "stable_intervals": intervals,
            "condition_false_intervals": {name: spans(~good[indices, j], ws) for j, name in enumerate(definitions["condition_fields"])},
            "first_dwell_end_step_by_arithmetic": latch,
            "pre_latch_stable_intervals": [[a, b if latch is None else min(b, latch)] for a, b in intervals if latch is None or a <= latch],
            "window_extrema": extrema({k: v[indices] for k, v in curves.items()}, ws) if len(ws) else {}})
    return curves, {"curve_extrema": extrema(curves, steps), "events": result}


def audit_packet(label, packet_path, source_folder, geometry_classes):
    start = len(CHECKS)
    supplied = read(packet_path)
    header = read(source_folder / "episode.json")
    task, identity = header["episode"]["task"], header["episode"]["episode_id"]
    npz_path = watch(source_folder / header["original_values_npz"]["file"])
    compare(label+"/source_npz_sha256", digest(npz_path), header["original_values_npz"]["sha256"], "source_binding", True)
    compare(label+"/delivered_npz_identity", supplied["original_values_npz"], header["original_values_npz"], "source_binding", True)
    compare(label+"/episode_identity", supplied["episode"]["episode_id"], identity, "source_binding", True)
    compare(label+"/definitions", supplied["definitions"], header["definitions"], "source_binding", True)
    compare(label+"/clock", supplied["clock"], {"source_field": "episode_length" if task == "stair" else "control_step", "first_step": 1, "last_step": header["episode"]["steps"], "stride": 1}, "source_binding", True)
    with np.load(npz_path, allow_pickle=False) as archive:
        raw = {k[5:]: archive[k].copy() for k in archive.files if k.startswith("raw__")}
        static = {k[8:]: archive[k].copy() for k in archive.files if k.startswith("static__")}
    T = header["episode"]["steps"]
    for name, a in raw.items():
        compare(label+"/raw_source_hash/"+name, hashlib.sha256(a.tobytes()).hexdigest(), header["raw_fields"][name]["c_order_bytes_sha256"], "source_binding", True)
    for name, column in supplied["raw_columns"].items():
        a = decode(column, T)
        compare(label+"/raw_columns/"+name+"/bytes", [a.dtype.str, list(a.shape), hashlib.sha256(a.tobytes()).hexdigest()],
                [raw[name].dtype.str, list(raw[name].shape), hashlib.sha256(raw[name].tobytes()).hexdigest()], "full_time_serialization", True)
    if task == "stair":
        curves, expected = stair_expected(raw, static, header["definitions"], header["episode"]["dt_s"], geometry_classes)
    else:
        curves, expected = push_expected(raw, header["episode"]["original_terminal_record"], header["definitions"])
    compare(label+"/direct_arithmetic/top_level_keys", sorted(supplied["direct_arithmetic"]), sorted(expected), exact=True)
    for key, value in expected.items():
        compare(label+"/direct_arithmetic/"+key, supplied["direct_arithmetic"].get(key), value)
    for name, column in supplied["direct_columns"].items():
        compare(label+"/direct_columns/"+name, decode(column, T), curves[name])
    if task == "stair":
        original = {name: [] for name in ("support_planar_speed_mps", "support_reference_clearance_m")}
        for part in header["parts"]:
            part_path = source_folder / part["file"]
            compare(label+"/original_part_hash/"+part["file"], digest(watch(part_path)), part["sha256"], "source_binding", True)
            data = read(part_path)
            for name in original:
                original[name].extend(data["direct_curves"][name])
        for name, values in original.items():
            compare(label+"/new_direct_column_preserves_source/"+name, decode(supplied["direct_columns"][name], T), values, "original_direct_column_serialization", True)
    checks = CHECKS[start:]
    return {"packet": label, "task": task, "episode_id": identity, "source": str(source_folder), "steps": T,
            "check_groups": len(checks), "objects_and_scalars_checked": sum(x["objects_and_scalars_checked"] for x in checks),
            "difference_count": sum(x["difference_count"] for x in checks), "status": "PASS_WITH_EXPLICIT_LIMITS" if all(x["equal"] for x in checks) else "FAIL"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--original-capture", type=Path, required=True)
    parser.add_argument("--frozen-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CUDA_VISIBLE_DEVICES must be explicitly empty")
    args.output.mkdir(parents=True, exist_ok=False)
    result = {"status": "INCOMPLETE", "semantic_score": None, "model_calls": 0, "gpu_calls": 0,
        "script": {"path": str(Path(__file__).resolve()), "sha256": digest(__file__)},
        "args": {k: str(v) for k, v in vars(args).items()}, "absolute_float_tolerance": ABS_TOL,
        "limits": ["Checks delivered episode/media arithmetic and columns, not production full report correctness or model semantic coverage.",
                   "Frozen normative stair surface primitive is reused; this is normative geometry arithmetic, not independent collision geometry or contact physics.",
                   "Original event receipts and task definitions are bound as sources; their physical truth is not established here.",
                   "Focus packed tables are not re-audited by this script; prior focus audit has its own scope."]}
    packets = []
    try:
        watch(__file__)
        manifest = read(args.original_capture / "numeric_raw/manifest.json")
        sys.path.insert(0, str(args.frozen_source / "src"))
        from true_rl_agent.tasks.stair_climbing.unified_evaluation import UnifiedStairCellSpec
        from true_rl_agent.tasks.stair_climbing.terrain_evidence import stair_surface_query_for_ordered_cell
        for module_name in ("terrain_evidence", "rough_surface", "ordered_evaluation", "unified_evaluation"):
            path = watch(sys.modules["true_rl_agent.tasks.stair_climbing."+module_name].__file__)
            relative = path.relative_to(args.frozen_source).as_posix()
            compare("source/normative_geometry/"+module_name, digest(path), manifest["source_hashes"][relative], "frozen_source_binding", True)
        geometry_classes = (UnifiedStairCellSpec, stair_surface_query_for_ordered_cell)
        paths = sorted((args.attempt / "numeric_materials/materials/related").glob("episode-*.json"))
        compare("scope/numeric_episode_count", len(paths), 24, "audit_scope", True)
        for path in paths:
            data = read(path)
            folder = args.original_capture / "numeric_raw/episodes" / data["episode"]["episode_id"]
            packets.append(audit_packet(path.stem, path, folder, geometry_classes))
            print(json.dumps(packets[-1]), flush=True)
        media_paths = sorted((args.attempt / "media_materials").glob("media-*/related_numeric.json"))
        compare("scope/media_episode_count", len(media_paths), 4, "audit_scope", True)
        for path in media_paths:
            original_root = args.original_capture / "media_materials_control_corrected" / path.parent.name / "raw/episodes"
            folders = [p for p in original_root.iterdir() if p.is_dir() and (p / "episode.json").is_file()]
            if len(folders) != 1:
                raise ValueError("media source must resolve to one existing episode")
            packets.append(audit_packet(path.parent.name, path, folders[0], geometry_classes))
            print(json.dumps(packets[-1]), flush=True)
        for module in list(sys.modules.values()):
            filename = getattr(module, "__file__", None)
            if filename and Path(filename).resolve().is_relative_to(args.frozen_source):
                watch(filename)
        for path, before in list(WATCH.items()):
            compare("unchanged/"+path, digest(path), before, "read_only_input_hash", True)
        result["status"] = "PASS_WITH_EXPLICIT_LIMITS" if all(x["equal"] for x in CHECKS) else "FAIL"
    except Exception as exc:
        result["exception"] = {"type": type(exc).__name__, "message": str(exc)}
        (args.output / "exception.traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
    result.update(packets=packets, check_groups=len(CHECKS), objects_and_scalars_checked=sum(x["objects_and_scalars_checked"] for x in CHECKS),
                  difference_count=sum(x["difference_count"] for x in CHECKS), checked_inputs=[{"path": p, "before_sha256": h, "after_sha256": digest(p)} for p, h in WATCH.items()])
    (args.output / "checks.json").write_text(json.dumps(CHECKS, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    (args.output / "receipt.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("status", "check_groups", "objects_and_scalars_checked", "difference_count")}), flush=True)
    return 0 if result["status"] == "PASS_WITH_EXPLICIT_LIMITS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
