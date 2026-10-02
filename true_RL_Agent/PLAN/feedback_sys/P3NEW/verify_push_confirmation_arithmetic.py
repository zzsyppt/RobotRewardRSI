"""Independent arithmetic check of a supplied Push related_v1 episode.

No production reducers, selectors, templates, providers, or simulation are used.
The supported contract is the recorded v2 velocity-increment episode, with all
first-episode rows supplied. This is arithmetic/receipt consistency, not semantic
review, archive authentication, cross-episode identity validation, or N5 admission.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def verify_push_confirmation_arithmetic(raw_numeric: dict, full_derivation: dict) -> dict:
    """Rebuild complete recovery sets from raw operands; return exact path coverage."""
    raw, full = raw_numeric, full_derivation
    checked, differences, unchecked = [], [], []
    result = {
        "schema_version": "p3new.push_confirmation_arithmetic.v1",
        "purpose": "independent_arithmetic_only_not_confirmation_admission",
        "status": "INCOMPLETE", "episode_id": raw.get("episode", {}).get("episode_id"),
        "checked_field_paths": checked, "differences": differences,
        "unchecked_fields": unchecked,
        "scope": "Exact supplied row/set/order and boolean/null checks; float tolerance 1e-10 absolute. Raw dtype byte hashes checked against supplied metadata only. No independent archive authentication or semantic/visual review.",
    }

    def compare(path, observed, expected):
        checked.append(path)
        if isinstance(expected, dict):
            if not isinstance(observed, dict):
                differences.append({"path": path, "expected": expected, "actual": observed})
                return
            if set(observed) != set(expected):
                differences.append({"path": path, "expected_keys": sorted(expected),
                                    "actual_keys": sorted(observed)})
            for key, value in expected.items():
                if key in observed:
                    compare(f"{path}/{key}", observed[key], value)
                else:
                    differences.append({"path": f"{path}/{key}", "error": "missing_field"})
        elif isinstance(expected, list):
            if not isinstance(observed, list):
                differences.append({"path": path, "expected": expected, "actual": observed})
                return
            if len(observed) != len(expected):
                differences.append({"path": path, "expected_length": len(expected),
                                    "actual_length": len(observed)})
            for i, value in enumerate(expected):
                if i < len(observed):
                    compare(f"{path}/{i}", observed[i], value)
                else:
                    differences.append({"path": f"{path}/{i}", "error": "missing_item"})
        else:
            # bool is a subclass of int: it must never pass a numeric comparison.
            if isinstance(expected, float):
                same = (type(observed) in (float, int) and math.isfinite(observed)
                        and abs(observed - expected) <= 1e-10)
            else:
                same = type(observed) is type(expected) and observed == expected
            if not same:
                differences.append({"path": path, "expected": expected, "actual": observed})

    def array_check(path, observed, expected):
        checked.append(path)
        if observed.shape != expected.shape or not np.array_equal(observed, expected):
            item = {"path": path, "expected_shape": list(expected.shape),
                    "actual_shape": list(observed.shape)}
            if observed.shape == expected.shape:
                item["mismatch_indices"] = np.argwhere(observed != expected).tolist()
            differences.append(item)

    try:
        if raw["profile"] not in {"related_v1", "related_v2"} or raw["episode"]["task"] != "push":
            raise ValueError("requires a complete Push related episode")
        definitions = raw["definitions"]
        parameters, time = definitions["parameters"], definitions["time"]
        if parameters["mechanism"] != "world_planar_velocity_increment":
            raise ValueError("force/pulse protocols are unsupported; their operands/rules differ")
        fields = definitions["condition_fields"]
        poses = definitions["pose_fields"]
        if fields != ["planar_speed", "angular_speed", "roll", "pitch", "height"]:
            raise ValueError("unsupported condition field order")
        if poses != ["height_m", "roll_rad", "pitch_rad", "planar_speed_mps", "angular_speed_radps", "xy_drift_m"]:
            raise ValueError("unsupported pose field order")
        if definitions["planar_motion_fields"] != ["root_lin_vel_w_x_mps", "root_lin_vel_w_y_mps", "reset_relative_root_pos_w_x_m", "reset_relative_root_pos_w_y_m"]:
            raise ValueError("unsupported planar-motion field order")
        first, last = raw["clock"]["first_step"], raw["clock"]["last_step"]
        if first != 1 or raw["clock"]["stride"] != 1:
            raise ValueError("requires consecutive complete first-episode rows beginning at step 1")
        steps = np.arange(first, last + 1, dtype=np.int64)
        n, dt = len(steps), time["step_dt_s"]
        arrays = {"control_step": steps}
        for name, spec in {**raw["raw_columns"], **raw["raw_discrete"]}.items():
            shape = (n, *spec["component_shape"])
            dtype = np.dtype(spec["dtype"])
            if "by_c_order_component" in spec:
                columns = [np.array(column.split(","), dtype=dtype)
                           for column in spec["by_c_order_component"]]
                arr = np.stack(columns, axis=1).reshape(shape)
            elif "inclusive_runs" in spec:
                arr, through = np.empty(shape, dtype=dtype), first - 1
                for a, b, value in spec["inclusive_runs"]:
                    if a != through + 1 or b < a or b > last:
                        raise ValueError(f"{name}: RLE is not an exact ordered partition")
                    arr[a-first:b-first+1] = value
                    through = b
                if through != last:
                    raise ValueError(f"{name}: RLE omits final rows")
            else:
                raise ValueError(f"{name}: unsupported raw encoding")
            arrays[name] = arr
        required = {"control_step", "pose", "planar_motion", "conditions", "valid",
                    "episode_index", "failure", "timeout", "event_steps", "event_verified",
                    "observed_dwell", "observed_recovery_step"}
        missing = required - arrays.keys()
        if missing:
            raise ValueError(f"missing raw operands: {sorted(missing)}")
        for name in sorted(required):
            arr, spec = arrays[name], raw["source_fields"][name]
            compare(f"/raw_numeric/source_fields/{name}/shape", list(arr.shape), spec["shape"])
            compare(f"/raw_numeric/source_fields/{name}/dtype", arr.dtype.str, spec["dtype"])
            compare(f"/raw_numeric/source_fields/{name}/c_order_bytes_sha256",
                    hashlib.sha256(arr.tobytes(order="C")).hexdigest(), spec["c_order_bytes_sha256"])
        if not arrays["valid"].all() or len(np.unique(arrays["episode_index"])) != 1:
            raise ValueError("requires only valid rows from one complete episode")
        if not np.isfinite(arrays["pose"]).all() or not np.isfinite(arrays["planar_motion"]).all():
            raise ValueError("nonfinite pose/planar motion operands")
        pose, motion = arrays["pose"].astype(float), arrays["planar_motion"].astype(float)
        receipt = raw["episode"]["original_terminal_record"]
        planned = receipt["planned_push_count"]
        window, dwell = time["recovery_window_steps"], time["required_dwell_steps"]
        if type(planned) is not int or planned < 1 or planned > len(time["event_scheduled_steps"]):
            raise ValueError("unsupported/missing planned event count")
        for name in ("event_steps", "event_verified", "observed_dwell", "observed_recovery_step"):
            if arrays[name].shape != (n, len(time["event_scheduled_steps"])):
                raise ValueError(f"{name}: event axis does not match the raw schedule")
    except (KeyError, ValueError, TypeError) as exc:
        result["input_error"] = str(exc)
        unchecked.append({"path": "/complete_episode_derivation", "reason": str(exc)})
        return result

    compare("/raw_numeric/episode/steps", raw["episode"]["steps"], n)
    compare("/raw_numeric/episode/dt_s", raw["episode"]["dt_s"], dt)
    compare("/raw_numeric/episode/original_terminal_record/episode_steps", receipt["episode_steps"], last)
    # Comparisons are performed in the original float32 dtype, as specified by
    # related_v1. Casting threshold operands to f32 preserves the sampled rule.
    f32 = arrays["pose"]
    cond = np.stack((f32[:, 3] <= parameters["recovery_velocity_mps"],
                     f32[:, 4] <= parameters["recovery_angular_velocity_radps"],
                     np.abs(f32[:, 1]) <= parameters["recovery_tilt_rad"],
                     np.abs(f32[:, 2]) <= parameters["recovery_tilt_rad"],
                     f32[:, 0] >= parameters["recovery_height_m"]), axis=1)
    array_check("/raw_numeric/raw_discrete/conditions", arrays["conditions"], cond)
    stable = cond.all(axis=1)

    def spans(mask, anchor):
        ranges = []
        for step in steps[mask].tolist():
            if ranges and step == ranges[-1][1] + 1:
                ranges[-1][1] = step
            else:
                ranges.append([step, step])
        return [{"start_step": a, "end_step_inclusive": b, "observed_steps": b-a+1,
                 "start_after_anchor_s": (a-anchor)*dt, "end_after_anchor_s": (b-anchor)*dt}
                for a, b in ranges]

    def sample(step):
        i = step - first
        return {"control_step": step, **dict(zip(poses, pose[i].tolist())),
                "velocity_world_xy_mps": motion[i, :2].tolist(),
                "reset_relative_position_world_xy_m": motion[i, 2:].tolist()}

    receipts = receipt["push_events"]
    event_ids = [event["event_index"] for event in receipts]
    actual_ids = [i for i in range(planned) if (arrays["event_steps"][:, i] >= 0).any()]
    compare("/raw_numeric/episode/original_terminal_record/push_events/event_index_order", event_ids, actual_ids)
    if event_ids != actual_ids:
        result["status"] = "FAIL"
        result["input_error"] = "raw event receipt identities/order disagree with event arrays"
        return result
    events = {event["event_index"]: event for event in receipts}
    anchors, verified = {}, {}
    for i in range(planned):
        observed = sorted(set(arrays["event_steps"][:, i].tolist()) - {-1})
        if len(observed) > 1:
            result["input_error"] = f"event {i} has multiple raw anchor values"
            return result
        anchors[i] = observed[0] if observed else None
        if i not in events:
            verified[i] = False
            continue
        event = events[i]
        prefix = f"/raw_numeric/episode/original_terminal_record/push_events/{event_ids.index(i)}"
        before = np.asarray(event["velocity_before_world_6d"], dtype=np.float32)
        after = np.asarray(event["velocity_after_write_world_6d"], dtype=np.float32)
        delta = (after[:2] - before[:2]).astype(float)
        compare(prefix + "/observed_delta_v_world_mps", event["observed_delta_v_world_mps"], delta.tolist())
        target = np.asarray(event["requested_delta_v_world_mps"], dtype=np.float32).astype(float)
        verified[i] = bool(np.max(np.abs(delta-target)) <= parameters["injection_tolerance_mps"])
        compare(prefix + "/injection_verified", event["injection_verified"], verified[i])
        compare(prefix + "/control_step", event["control_step"], anchors[i])
        compare(prefix + "/time_s", event["time_s"], anchors[i]*dt)

    recoveries = []
    expected_dwell = np.zeros_like(arrays["observed_dwell"])
    dwell_check_mask = np.zeros_like(arrays["observed_dwell"], dtype=bool)
    expected_recovery = np.full_like(arrays["observed_recovery_step"], -1)
    expected_events = np.full_like(arrays["event_steps"], -1)
    expected_verified = np.zeros_like(arrays["event_verified"])
    for i in range(planned):
        anchor = anchors[i]
        eligible = np.zeros(n, bool) if anchor is None or not verified[i] else ((steps > anchor) & (steps <= anchor+window))
        indices = np.flatnonzero(eligible)
        latch, maximum, current, interruptions = None, 0, 0, []
        for ix in range(n):
            step = int(steps[ix])
            if anchor is not None and step > anchor:
                expected_events[ix, i] = anchor
                expected_verified[ix, i] = verified[i]
            if eligible[ix] and latch is None:
                if stable[ix]:
                    current += 1
                    maximum = max(maximum, current)
                    if current >= dwell:
                        latch = step
                else:
                    if current:
                        interruptions.append({"control_step": step, "after_anchor_steps": step-anchor,
                            "after_anchor_s": (step-anchor)*dt, "preceding_dwell_steps": current,
                            "failed_conditions": [name for j, name in enumerate(fields) if not cond[ix, j]]})
                    current = 0
            elif latch is None:
                current = 0
            expected_dwell[ix, i] = current
            dwell_check_mask[ix, i] = bool(eligible[ix] or latch is not None)
            expected_recovery[ix, i] = latch if latch is not None else -1
        coverage = ("anchor_not_observed" if anchor is None else "anchor_not_verified" if not verified[i]
                    else "recovered" if latch is not None else "terminal_before_window_end"
                    if last < anchor+window else "full_window_without_dwell")
        # Recovery time is right-censored whenever the first latch was not
        # observed. Window completeness is a separate predicate above.
        censored = latch is None
        unit = {"identity": f"event_{i}", "scheduled_anchor_control_step": time["event_scheduled_steps"][i],
                "anchor_control_step": anchor, "anchor_verified": verified[i], "coverage": coverage,
                "observed_window_steps": int(eligible.sum()), "observed_window_duration_s": int(eligible.sum())*dt,
                "stable_window_steps": int((eligible & stable).sum()), "max_dwell_steps": maximum,
                "recovery_control_step": latch, "recovery_censored": censored,
                "failed_condition_steps": {name: int((eligible & ~cond[:, j]).sum()) for j, name in enumerate(fields)},
                "condition_violation_spans": {name: spans(eligible & ~cond[:, j], anchor) for j, name in enumerate(fields)},
                "dwell_interruptions": interruptions}
        if not len(indices):
            unit["process"] = {"status": "no_verified_observed_window", "reason": coverage}
        else:
            a, b = int(steps[indices[0]]), int(steps[indices[-1]])
            peak = int(steps[indices[np.argmax(pose[indices, 3])]])
            reference = np.asarray(events[i]["observed_delta_v_world_mps"], dtype=float)
            direction = reference/np.linalg.norm(reference)
            displacement = motion[b-first, 2:] - motion[anchor-first, 2:]
            next_anchors = [value for value in anchors.values() if value is not None and value > anchor]
            stop = min(next_anchors) if next_anchors else last
            post = np.zeros(n, bool) if latch is None else ((steps > latch) & (steps <= stop) & ~stable)
            unit["process"] = {
                "status": "available", "reference_event_index": i, "recovery_anchor_kind": "velocity_injection",
                "injection_reference_units": "m/s", "injection_reference_world_xy": reference.tolist(),
                "injection_start_sample": sample(anchor), "recovery_anchor_sample": sample(anchor),
                "first_observed_response": sample(anchor+1), "window_first": sample(a),
                "window_last": sample(b), "window_speed_peak": sample(peak), "peak_after_anchor_s": (peak-anchor)*dt,
                "window_abs_roll_pitch_max_rad": np.abs(pose[indices, 1:3]).max(axis=0).tolist(),
                "window_min_height_m": float(pose[indices, 0].min()),
                "window_reaches_planned_end": b == anchor+window,
                "stable_spans": spans(eligible & stable, anchor), "displacement_status": "available",
                "window_net_displacement_world_xy_m": displacement.tolist(),
                "window_net_displacement_m": float(np.linalg.norm(displacement)),
                "window_net_displacement_parallel_m": float(displacement @ direction),
                "window_velocity_parallel_first_peak_last_mps": [float(motion[s-first, :2] @ direction) for s in (a, peak, b)],
                "post_latch": {"status": "not_latched" if latch is None else "observed",
                    "observed_through_step": None if latch is None else stop,
                    "unstable_steps": int(post.sum()), "unstable_spans": spans(post, anchor),
                    "until": "next_executed_injection" if next_anchors else "last_observation"}}
        recoveries.append(unit)
        if i in events:
            prefix = f"/raw_numeric/episode/original_terminal_record/push_events/{event_ids.index(i)}"
            values = {"recovered": latch is not None, "recovery_control_step": latch,
                      "recovery_censored": censored}
            if latch is not None:
                values["recovery_latency_s"] = (latch-anchor)*dt
            else:
                unchecked.append({"path": prefix+"/recovery_latency_s",
                    "reason": "No recovery was observed; metadata does not define the numeric latency code for an unlatched/censored receipt. It is not an observed recovery time."})
            for key, value in values.items():
                compare(prefix+"/"+key, events[i][key], value)
    for name, expected in (("event_steps", expected_events), ("event_verified", expected_verified),
                           ("observed_recovery_step", expected_recovery)):
        array_check(f"/raw_numeric/raw_discrete/{name}", arrays[name], expected)
    array_check("/raw_numeric/raw_discrete/observed_dwell[eligible_or_latched]",
                arrays["observed_dwell"][dwell_check_mask], expected_dwell[dwell_check_mask])
    result["raw_dwell_checked_steps_by_event"] = {
        str(i): steps[dwell_check_mask[:, i]].tolist() for i in range(planned)}
    unchecked.append({"path": "/raw_numeric/raw_discrete/observed_dwell[outside_eligible_and_unlatched]",
        "reason": "Metadata defines eligible steps and first latch, but not the raw counter storage outside an unsuccessful window. Complete eligible dwell, maximum, interruptions and latch are independently recomputed."})

    latches = [u["recovery_control_step"] for u in recoveries if u["recovery_control_step"] is not None]
    all_latched = len(latches) == planned
    after = (steps > max(latches)) & ~stable if all_latched else np.zeros(n, bool)
    first_bad = int(steps[np.flatnonzero(after)[0]]) if after.any() else None
    stratum = receipt["stratum"]
    expected = {
        "episode_index": int(arrays["episode_index"][0]), "stratum": stratum,
        "coverage": definitions["episode_coverage"], "valid_steps": n, "final_control_step": last,
        "final_pose": dict(zip(poses, pose[-1].tolist())), "final_stable": bool(stable[-1]),
        "final_failed_conditions": [name for j, name in enumerate(fields) if not cond[-1, j]],
        "recoveries": recoveries, "all_required_recoveries_latched": all_latched,
        "unstable_steps_after_all_recoveries": int(after.sum()),
        "first_instability_after_all_recoveries": None if first_bad is None else {
            "control_step": first_bad, "after_last_recovery_s": (first_bad-max(latches))*dt,
            "failed_conditions": [name for j, name in enumerate(fields) if not cond[first_bad-first, j]]},
        "final_unstable_after_any_recovery": bool(latches and not stable[-1]),
        "terminal_label": receipt["failure_label"],
        "event_conditions": {"planned_events": planned, "executed_events": len(actual_ids),
            "completed_events": len(actual_ids), "verified_events": sum(verified.values()), "stratum": stratum,
            "nominal_velocity_or_impulse_per_mass_mps": parameters["velocity_magnitudes_mps"]["medium" if stratum == "repeated" else stratum]},
    }
    actual = full["complete_episode_derivation"]
    compare("/full_derivation/complete_episode_derivation",
            {k: v for k, v in actual.items() if k not in ("source", "env_id")}, expected)
    compare("/full_derivation/episode_id", full["episode_id"], raw["episode"]["episode_id"])
    ranges = {"min_base_height_m": float(pose[:, 0].min()), "max_pitch_rad": float(pose[:, 2].max()),
              "min_pitch_rad": float(pose[:, 2].min()), "max_abs_roll_rad": float(np.abs(pose[:, 1]).max()),
              "max_abs_pitch_rad": float(np.abs(pose[:, 2]).max()), "max_xy_drift_m": float(pose[:, 5].max()),
              "final_planar_speed_mps": float(pose[-1, 3]), "final_recovery_condition": bool(stable[-1]),
              "push_event_count": len(actual_ids), "recovered_event_count": len(latches),
              "invalid_injection_count": len(actual_ids)-sum(verified.values())}
    for key, value in ranges.items():
        compare("/raw_numeric/episode/original_terminal_record/"+key, receipt[key], value)
    result["recomputed_whole_episode_ranges_and_counts"] = ranges
    result["checked_recovery_units"] = len(recoveries)
    for path, reason in (
        ("/full_derivation/complete_episode_derivation/source", "Original source ordinal is absent from the supplied raw episode; do not infer from a display alias."),
        ("/full_derivation/complete_episode_derivation/env_id", "Original environment identity is absent from the supplied raw episode."),
        ("/full_derivation/source", "Source paths and digests are supplied claims; original archives were not opened/authenticated."),
        ("/full_derivation/source_pointer", "Original archive transcript position is absent from raw operands."),
        ("/full_derivation/original_logical_id", "Requires original source/env binding, absent from raw operands."),
        ("/full_derivation/shared", "Cohort bindings/comparisons, sampling policy and prose definitions require separate identity/semantic review."),
        ("/full_derivation/complete_episode_derivation/terminal_label", "Checked only against original terminal receipt, not independently established label meaning/priority."),
        ("/raw_numeric/episode/original_terminal_record/mean_action_l2", "Action vectors absent from related_v1."),
        ("/raw_numeric/episode/original_terminal_record/mean_action_rate_l2", "Consecutive action vectors absent from related_v1."),
        ("/raw_numeric/raw_columns/pose", "Threshold predicates/ranges checked; quaternion, full root angular velocity and absolute/reset root position absent, so pose channel physical construction is not independently recomputed."),
    ):
        unchecked.append({"path": path, "reason": reason})
    result["status"] = "FAIL" if differences else "PASS_WITH_EXPLICIT_LIMITS"
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True, help="Existing episode report request.http_body.json; never sent")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    body = json.loads(args.request.read_text(encoding="utf-8"))
    materials = []
    for message in body["messages"]:
        if isinstance(message.get("content"), list):
            for content in message["content"]:
                if content.get("type") == "text":
                    value = json.loads(content["text"])
                    if isinstance(value, dict) and "role" in value and "data" in value:
                        materials.append(value)
    raw = [m["data"] for m in materials if m["role"] == "raw_numeric"]
    full = [m["data"] for m in materials if m["role"] == "full_derivation"]
    if len(raw) != 1 or len(full) != 1:
        parser.error("requires exactly one raw_numeric and one full_derivation material")
    report = verify_push_confirmation_arithmetic(raw[0], full[0])
    report["input"] = {"request": str(args.request.resolve()), "request_bytes": args.request.stat().st_size,
                       "request_sha256": hashlib.sha256(args.request.read_bytes()).hexdigest()}
    report["verifier_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k: report.get(k) for k in ("status", "episode_id", "checked_recovery_units", "input_error")}))
    raise SystemExit(0 if report["status"] == "PASS_WITH_EXPLICIT_LIMITS" else 1)
