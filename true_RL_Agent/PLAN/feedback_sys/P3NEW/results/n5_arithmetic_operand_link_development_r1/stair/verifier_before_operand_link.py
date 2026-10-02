"""Independent, offline arithmetic checks for one related_v1 Stair episode.

This acceptance helper imports no production behavior/template/selection code.
It reconstructs ALL event streams, including empty streams and unconfirmed runs.
Force norms supplied by related_v1 are operands, not independently verified L0.
Definitions, provenance and semantic prose are outside this arithmetic verdict.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

EVENT_COLUMNS = ["body", "code", "start", "confirmed", "end"]
PROFILE_CODES = ("tilt", "slip_any", "low_clearance_any", "low_progress")
ALL_CODES = (*PROFILE_CODES, "forbidden_manager", "no_instant_support", "contact_instant",
             "slip_body", "low_clearance_body", "unloaded_low_motion", "unloaded_rising",
             "unloaded_lowering", "contact_history", "forbidden_history")


def encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def decode_column(value, n):
    """Expand declared-dtype columns/runs with no gaps, overlap or zero filling."""
    dtype, shape = np.dtype(value["dtype"]), tuple(value["component_shape"])
    width = math.prod(shape)
    if "by_c_order_component" in value:
        cols = [[float(x) if x != "null" else np.nan for x in col.split(",")]
                for col in value["by_c_order_component"]]
        if len(cols) != width or any(len(col) != n for col in cols):
            raise ValueError("column dimensions do not cover the full clock")
        out = np.asarray(cols, dtype=dtype).T.reshape((n, *shape))
    else:
        out = np.empty((n, *shape), dtype=dtype)
        if "inclusive_runs" in value:
            groups, target = [value["inclusive_runs"]], [out]
        else:
            groups = value["inclusive_constant_intervals_by_component"]
            if len(groups) != width:
                raise ValueError("constant interval component count differs")
            target = list(out.reshape(n, width).T)
        for rows, array in zip(groups, target, strict=True):
            cursor = 1
            for start, end, constant in rows:
                if type(start) is not int or type(end) is not int or start != cursor or not start <= end <= n:
                    raise ValueError("intervals must cover every step exactly once")
                array[start-1:end] = constant
                cursor = end + 1
            if cursor != n + 1:
                raise ValueError("intervals end before the final sample")
    if not np.isfinite(out).all():
        raise ValueError("non-finite required arithmetic operand")
    return out.astype(np.float64) if dtype.kind == "f" else out


def decode_events(events):
    if isinstance(events, list):
        return deepcopy(events)
    if events.get("schema_version") != "p3new.stair_review_events_table.v1" or events.get("columns") != EVENT_COLUMNS:
        raise ValueError("unsupported event table")
    rows = [dict(zip(EVENT_COLUMNS, row, strict=True)) for row in events["rows"]]
    for index, measured in events["measured_by_original_event_index"].items():
        if str(int(index)) != index or not 0 <= int(index) < len(rows):
            raise ValueError("measured event index is outside the original table")
        rows[int(index)]["measured"] = deepcopy(measured)
    return rows


def mismatch_examples(actual, expected, path, limit=12):
    """Exact field comparison after the documented six-decimal presentation."""
    examples, count = [], 0
    def visit(a, e, p):
        nonlocal count
        if isinstance(a, dict) and isinstance(e, dict):
            for key in sorted(a.keys() | e.keys()):
                visit(a.get(key, {"missing": True}), e.get(key, {"missing": True}), p + "/" + key)
        elif isinstance(a, list) and isinstance(e, list) and len(a) == len(e):
            for i, (av, ev) in enumerate(zip(a, e)):
                visit(av, ev, p + "/" + str(i))
        elif encoded(a) != encoded(e):
            count += 1
            if len(examples) < limit:
                examples.append({"locator": p, "actual": a if not isinstance(a, list) else {"length": len(a)},
                                 "expected": e if not isinstance(e, list) else {"length": len(e)}})
    visit(actual, expected, path)
    return count, examples


def verify_stair_episode(raw_numeric, full_derivation):
    """Return checked/unchecked locators and recomputed fields; never score prose.

Arguments are the two material ``data`` objects, not request/response wrappers.
Missing operands mark dependent checks unchecked; malformed encodings raise.
"""
    raw, full = raw_numeric, full_derivation
    if raw.get("schema_version") != "p3new.related_episode.v1" or raw["episode"]["task"] != "stair":
        raise ValueError("requires one related_v1 Stair episode")
    if full["shared"]["domain"]["rule_version"] != "stair.observed_motion.v2":
        raise ValueError("only the frozen observed_motion.v2 arithmetic is implemented")
    e, definitions = full["complete_episode_derivation"], raw["definitions"]
    clock = raw["clock"]
    if clock["first_step"] != 1 or clock["stride"] != 1 or clock["last_step"] < 1:
        raise ValueError("requires complete consecutive first-episode clock")
    n, dt = clock["last_step"], float(raw["episode"]["dt_s"])
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("invalid control dt")
    arrays, paths = {}, {}
    for section in ("raw_columns", "raw_discrete", "direct_columns"):
        for name, value in raw.get(section, {}).items():
            arrays[name] = decode_column(value, n)
            paths[name] = f"raw_numeric/data/{section}/{name}"
    signals = definitions["protocol"]["episode_outcomes"]["signals"]
    query = definitions["arithmetic_queries"]
    names = definitions["bodies"]["support_effector"]
    base_path = "full_derivation/data/complete_episode_derivation"
    source_path = full["source_pointer"]
    checks, unchecked, expected = [], [], {}

    def check(field, value, basis, operands):
        actual = e.get(field, {"missing": True})
        count, examples = mismatch_examples(actual, value, base_path + "/" + field)
        checks.append({"field": field, "locator": base_path + "/" + field,
                       "original_locator": source_path + "/" + field, "basis": basis,
                       "operands": operands, "status": "fail" if count else "pass",
                       "mismatch_count": count, "mismatch_examples": examples})
        expected[field] = value

    def need(fields, target):
        missing = [name for name in fields if name not in arrays]
        if missing:
            unchecked.append({"locator": target, "reason": "missing operands", "missing": missing})
        return not missing

    def rounded(value):
        return round(float(value), 6)

    qfields = ["base_pos_w", "base_quat_w"]
    root = None
    origin = np.asarray(raw["static_values"]["terrain_env_origins_w"], dtype=float)
    if need(qfields, base_path + "/{ranges,root_motion,anchor}"):
        if definitions["coordinates"]["base_quat_w"] != "wxyz":
            raise ValueError("unexpected quaternion order")
        w, x, y, z = arrays["base_quat_w"].T
        roll = np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y))
        pitch = np.arcsin(np.clip(2*(w*y-z*x), -1, 1))
        progress, up = arrays["base_pos_w"][:, 0] - origin[0], 1-2*(x*x+y*y)
        root = {"x_m": progress, "roll_rad": roll, "pitch_rad": pitch, "up_z": up}
        bounds = {key: [rounded(a.min()), int(a.argmin())+1, rounded(a.max()), int(a.argmax())+1]
                  for key, a in root.items()}
        drops = np.maximum.accumulate(progress) - progress
        trough = int(drops.argmax())
        peak = int(progress[:trough+1].argmax()) if drops[trough] > 0 else 0
        negative = np.flatnonzero(up < 0)
        motion = {"first": [rounded(a[0]) for a in root.values()], "last": [rounded(a[-1]) for a in root.values()],
                  "extrema": bounds, "max_x_retreat": [peak+1, trough+1, rounded(drops[trough])],
                  "first_up_z_negative_step": int(negative[0])+1 if len(negative) else None}
        check("root_motion", motion, "independent_raw_arithmetic", [paths[k] for k in qfields])
        check("ranges", {key: [bounds[src][0], bounds[src][2]] for key, src in
                         (("progress_m", "x_m"), ("roll_rad", "roll_rad"), ("pitch_rad", "pitch_rad"))},
              "independent_raw_arithmetic", [paths[k] for k in qfields])
        crossed = np.flatnonzero(progress >= definitions["geometry"]["start_platform_length_m"] / 2)
        anchor = ({"status": "not_reached", "step": None} if not len(crossed) else
                  {"status": "already_beyond_first_sample", "step": None} if crossed[0] == 0 else
                  {"status": "crossed", "step": int(crossed[0])+1})
        check("anchor", anchor, "independent_raw_arithmetic", [paths["base_pos_w"], "raw_numeric/data/definitions/geometry/start_platform_length_m"])

    # Surface and force reductions are supplied operands, not a second geometry/sensor implementation.
    clearance, speeds, speed3d, vz = None, None, None, None
    if need(["support_pos_w", "support_surface_height_terrain_m"], base_path + "/body_extrema/*/1"):
        clearance = arrays["support_pos_w"][..., 2] - origin[2] - arrays["support_surface_height_terrain_m"]
    if need(["support_lin_vel_w"], base_path + "/body_extrema/*/{2,3}"):
        velocity = arrays["support_lin_vel_w"]
        speeds, speed3d, vz = np.linalg.norm(velocity[..., :2], axis=-1), np.linalg.norm(velocity, axis=-1), velocity[..., 2]
    if clearance is not None and speeds is not None:
        check("body_extrema", [[name, rounded(clearance[:, j].min()), rounded(speeds[:, j].max()), rounded(abs(vz[:, j]).max())]
                               for j, name in enumerate(names)],
              "independent_velocity_arithmetic_and_clearance_conditional_on_supplied_surface",
              [paths["support_pos_w"], paths["support_lin_vel_w"], paths["support_surface_height_terrain_m"]])

    masks, missing_codes = {}, set()
    def add(code, body, mask, required=1):
        masks[(code, body)] = (np.asarray(mask, bool), int(required))
    contact = arrays.get("support_force_norm_n")
    loaded = contact > signals["slip"]["contact_force_threshold"] if contact is not None else None
    if root is not None:
        add("tilt", "root", np.maximum(abs(roll), abs(pitch)) > signals["base_orientation_instability"]["threshold"],
            signals["base_orientation_instability"]["consecutive_control_steps"])
        span = math.ceil(query["x_window_s"] / dt)
        low_progress = np.zeros(n, bool)
        for i in range(span, n):
            low_progress[i] = progress[i-span:i+1].max() - progress[i-span] < query["x_window_gain_m"]
        add("low_progress", "root", low_progress)
    else:
        missing_codes.update(("tilt", "low_progress"))
    if loaded is not None:
        add("no_instant_support", "all_support", ~loaded.any(axis=1))
        for j, body in enumerate(names):
            add("contact_instant", body, loaded[:, j])
        for code, values, required in (
            ("slip", None if speeds is None else loaded & (speeds > signals["slip"]["speed_threshold"]), signals["slip"]["consecutive_control_steps"]),
            ("low_clearance", None if clearance is None else ~loaded & (clearance < signals["insufficient_clearance"]["threshold"]), signals["insufficient_clearance"]["consecutive_control_steps"]),
        ):
            if values is None:
                missing_codes.update((code+"_any", code+"_body"))
            else:
                add(code+"_any", "any_support", values.any(axis=1), required)
                for j, body in enumerate(names): add(code+"_body", body, values[:, j], required)
        if speed3d is not None and root is not None:
            for j, body in enumerate(names):
                add("unloaded_low_motion", body, ~loaded[:, j] & (speed3d[:, j] <= query["unloaded_speed_mps"]), math.ceil(query["unloaded_duration_s"] / dt))
                add("unloaded_rising", body, ~loaded[:, j] & (vz[:, j] > query["signed_vz_mps"]), 3)
                add("unloaded_lowering", body, ~loaded[:, j] & (vz[:, j] < -query["signed_vz_mps"]), 3)
        else:
            missing_codes.update(("unloaded_low_motion", "unloaded_rising", "unloaded_lowering"))
    else:
        missing_codes.update(("contact_instant", "no_instant_support", "slip_any", "slip_body", "low_clearance_any", "low_clearance_body",
                              "unloaded_low_motion", "unloaded_rising", "unloaded_lowering"))
    history = arrays.get("support_buffer_max_force_norm_n")
    if history is not None:
        for j, body in enumerate(names): add("contact_history", body, history[:, j] > signals["slip"]["contact_force_threshold"])
    else: missing_codes.add("contact_history")
    forbidden_history = arrays.get("forbidden_buffer_max_force_norm_n")
    if forbidden_history is not None:
        add("forbidden_history", "forbidden_role", (forbidden_history > signals["forbidden_contact"]["force_threshold_n"]).any(axis=1))
    else: missing_codes.add("forbidden_history")
    terms = definitions["termination_term_order"]
    manager_field = "termination_" + str(terms.index(signals["forbidden_contact"]["manager_term"]))
    if manager_field in arrays: add("forbidden_manager", "forbidden_role", arrays[manager_field])
    else: missing_codes.add("forbidden_manager")

    events = []
    for (code, body), (mask, required) in masks.items():
        edges = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(np.int8)))
        for start, stop in edges.reshape(-1, 2):
            event = {"body": body, "code": code, "start": int(start)+1, "end": int(stop),
                     "confirmed": int(start)+required if stop-start >= required else None}
            if code == "unloaded_low_motion":
                event["measured"] = {"max_speed_mps": rounded(speed3d[start:stop, names.index(body)].max()),
                                     "root_x_gain_m": rounded(progress[stop-1]-progress[start])}
            events.append(event)
    events.sort(key=lambda row: (row["start"], row["body"], row["code"]))
    actual_events = decode_events(e["events"])
    actual_subset = [row for row in actual_events if row.get("code") not in missing_codes]
    count, examples = mismatch_examples(actual_subset, events, source_path + "/events")
    expected_counter, actual_counter = Counter(map(digest, events)), Counter(map(digest, actual_subset))
    event_check = {"field": "events", "locator": base_path + "/events", "original_locator": source_path + "/events",
                   "fields": [*EVENT_COLUMNS, "measured/max_speed_mps", "measured/root_x_gain_m"],
                   "table_locator_mapping": "original /events/i/{field} = table /events/rows/i/column; measured at /events/measured_by_original_event_index/i",
                   "basis": "independent_full_stream_reduction_conditional_on_supplied_force_norms_and_surface",
                   "operands": [paths[k] for k in ("base_pos_w", "base_quat_w", "support_lin_vel_w", "support_pos_w", "support_surface_height_terrain_m", "support_force_norm_n", "support_buffer_max_force_norm_n", "forbidden_buffer_max_force_norm_n", manager_field) if k in paths],
                   "checked_codes_including_empty": sorted(set(ALL_CODES)-missing_codes), "predicate_stream_count": len(masks),
                   "expected_event_count": len(events), "reported_checked_event_count": len(actual_subset),
                   "missing_event_count": sum((expected_counter-actual_counter).values()),
                   "extra_or_changed_event_count": sum((actual_counter-expected_counter).values()),
                   "order_and_all_fields_equal": count == 0, "status": "fail" if count else "pass", "mismatch_count": count,
                   "mismatch_examples": examples, "expected_events_sha256": digest(events)}
    checks.append(event_check)
    expected["events"] = events
    if missing_codes:
        unchecked.append({"locator": base_path + "/events", "codes": sorted(missing_codes), "reason": "dependent raw/direct operand missing"})
    window_first = max(1, n-math.ceil(1.0/dt)+1)
    for field, codes in (("before_terminal", (*PROFILE_CODES, "unloaded_low_motion")), ("window_present", PROFILE_CODES)):
        if any(code in missing_codes for code in codes):
            unchecked.append({"locator": base_path + "/" + field, "reason": "some complete predicate streams unavailable"})
            continue
        values = {code: any(row["code"] == code and (row["confirmed"] is not None and row["confirmed"] < n
                          if field == "before_terminal" else row["start"] < n and row["end"] >= window_first) for row in events) for code in codes}
        check(field, values, "complete_stream_temporal_arithmetic_conditional_on_supplied_force_norms_and_surface", ["recomputed complete events", "raw_numeric/data/clock"])
    check("steps", n, "declared_clock_and_sample_count_consistency", ["raw_numeric/data/clock"])
    check("dt_s", dt, "copied_metadata_consistency_only", ["raw_numeric/data/episode/dt_s"])
    check("coverage", {"first_step": 1, "last_step": n, "terminal_included": True, "window_first_step": window_first},
          "declared_full_episode_clock; observed_motion.v2 final 1.0s window", ["raw_numeric/data/clock", "raw_numeric/data/episode/original_terminal_record/episode_steps"])
    check("primary_reason", raw["episode"]["original_terminal_record"]["primary_reason"], "copied_original_outcome_only_not_reclassification",
          ["raw_numeric/data/episode/original_terminal_record/primary_reason"])
    terminal_consistent = n == raw["episode"]["steps"] == raw["episode"]["original_terminal_record"]["episode_steps"]
    valid_consistent = bool(arrays["valid"].all()) if "valid" in arrays else None
    if valid_consistent is None:
        unchecked.append({"locator": "raw_numeric/data/raw_discrete/valid", "reason": "validity operand missing"})
    checks.append({"field": "input_clock_and_terminal", "locator": "raw_numeric/data/{clock,episode,raw_discrete/valid}",
                   "basis": "material_internal_consistency_not_physical_termination", "status": "pass" if terminal_consistent and valid_consistent is not False else "fail",
                   "declared_step_counts_equal": terminal_consistent, "all_samples_valid": valid_consistent})
    unchecked.extend([
        {"locator": "raw_numeric/data/direct_columns/support_surface_height_terrain_m",
         "reason": "geometry surface calculation is not independently recomputed; supplied direct operand only"},
        {"locator": "raw_numeric/data/direct_columns/{support_force_norm_n,support_buffer_max_force_norm_n,forbidden_buffer_max_force_norm_n}",
         "reason": "raw force vectors/history slots absent from related_v1; their norm/max reductions are not independently verified"},
        {"locator": base_path + "/primary_reason", "reason": "original outcome copy checked only; no physical success/termination reclassification, missing upward forces and external latch evidence"},
        {"locator": "full_derivation/data/{source,source_pointer,episode_id,original_logical_id,shared}", "reason": "identity/hash authority, definition prose, ABI claims and cohort/selection contract are outside this arithmetic verifier"},
        {"locator": base_path + "/{episode_id,cell,stratum,run,slot}", "reason": "source identity/binding not judged by arithmetic"},
    ])
    failed = sum(c["status"] == "fail" for c in checks)
    return {"schema_version": "p3new.stair_confirmation_arithmetic.v1", "status": "FAIL" if failed else "PASS_CHECKED_FIELDS_WITH_UNCHECKED",
            "scope": "old-development arithmetic; not semantic acceptance, N5, provenance verification or complete independent L0 coverage",
            "production_oracle_calls": 0, "complete_independent_l0_coverage": False,
            "episode_id": raw["episode"]["episode_id"], "steps": n, "checks": checks, "unchecked": unchecked,
            "checked_n": len(checks), "failed_n": failed, "unchecked_n": len(unchecked),
            "rounding": "floating report outputs rounded to 6 decimals; event steps, bodies, codes, booleans and nulls compare exactly",
            "fixed_rule_details": {"window_s": 1.0, "unloaded_vertical_motion_required_steps": 3, "event_order": "start,body,code", "extrema_ties": "earliest"},
            "recomputed_fields": expected}


def materials_from_http(path):
    request = json.loads(Path(path).read_bytes())
    found = {}
    for message in request["messages"]:
        if not isinstance(message.get("content"), list): continue
        for block in message["content"]:
            try: value = json.loads(block.get("text", ""))
            except (ValueError, TypeError): continue
            if isinstance(value, dict) and value.get("role") in ("raw_numeric", "full_derivation"):
                if value["role"] in found: raise ValueError("expected exactly one episode material per role")
                found[value["role"]] = value["data"]
    return found["raw_numeric"], found["full_derivation"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mutation-check", action="store_true", help="three old-development omission/confirmation/body perturbations")
    args = parser.parse_args()
    if args.output.exists(): raise ValueError("output directory must be fresh; frozen results are never overwritten")
    before = hashlib.sha256(args.request.read_bytes()).hexdigest()
    raw, full = materials_from_http(args.request)
    result = verify_stair_episode(raw, full)
    recomputed = result.pop("recomputed_fields")
    mutations = []
    if args.mutation_check:
        events = decode_events(full["complete_episode_derivation"]["events"])
        index = next(i for i, row in enumerate(events) if row["confirmed"] is None)
        for name in ("omit_one_event", "unconfirmed_to_confirmed", "wrong_named_body"):
            changed = deepcopy(full)
            changed["complete_episode_derivation"]["events"] = deepcopy(events)
            rows = changed["complete_episode_derivation"]["events"]
            if name == "omit_one_event": rows.pop(index)
            elif name == "unconfirmed_to_confirmed": rows[index]["confirmed"] = rows[index]["end"]
            else: rows[index]["body"] = "FL_calf"
            checked = verify_stair_episode(raw, changed)
            event_result = next(c for c in checked["checks"] if c["field"] == "events")
            mutations.append({"mutation": name, "original_event_index": index, "caught": event_result["status"] == "fail",
                              "event_check": event_result})
    if hashlib.sha256(args.request.read_bytes()).hexdigest() != before: raise AssertionError("frozen request changed")
    result.update(request_path=str(args.request.resolve()), request_sha256=before,
                  verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  raw_numeric_material_sha256=digest(raw), full_derivation_material_sha256=digest(full),
                  invocation=["python", "-B", str(Path(__file__).resolve()), "--request", str(args.request.resolve()),
                              "--output", str(args.output.resolve()), *(["--mutation-check"] if args.mutation_check else [])],
                  new_api_calls=0, gpu_jobs=0, ssh_calls=0, mutations=mutations)
    args.output.mkdir(parents=True)
    (args.output / "recomputed_fields.json").write_bytes(encoded(recomputed))
    result["recomputed_fields_sha256"] = hashlib.sha256((args.output / "recomputed_fields.json").read_bytes()).hexdigest()
    (args.output / "verification.json").write_bytes(json.dumps(result, ensure_ascii=False, indent=2).encode() + b"\n")
    print(json.dumps({k: result[k] for k in ("status", "checked_n", "failed_n", "unchecked_n", "request_sha256", "verifier_sha256")}))
    print(json.dumps({"mutations_caught": [m["mutation"] for m in mutations if m["caught"]]}))
    raise SystemExit(1 if result["failed_n"] or any(not m["caught"] for m in mutations) else 0)


if __name__ == "__main__":
    main()
