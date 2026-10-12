"""Pack development or explicitly frozen confirmation evidence offline.

No API, release, semantic scoring, automatic report import, or new cohort access.
Report manifests remain non-executable drafts until reports and real facts seals exist.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import shutil

import numpy as np

import run_confirmation_review as runner


HERE = Path(__file__).resolve().parent
QUESTIONS = [
    ("Q1", "完整样本分母与原始结局"),
    ("Q2", "失败中常见与不常见的测量过程"),
    ("Q3", "相容成功反例是否也出现同一过程及匹配范围"),
    ("Q4", "有原始定位依据的主要过程关系或关键量"),
    ("Q5", "限制判断的缺观测和采样范围"),
]


def read(path):
    return json.loads(Path(path).read_bytes())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2).encode("utf-8") + b"\n")


def material(path, relative_file, role, material_id):
    return {"id": material_id, "role": role, "file": relative_file,
            "sha256": digest(path), "mime_type": "application/json"}


def decimal_columns(array, *, constant_intervals=False):
    """Readable full-time columns; finite values round-trip at the declared dtype."""
    array = np.asarray(array)
    flat = array.reshape(len(array), -1).T
    columns = [[("null" if not np.isfinite(v) else min(np.format_float_positional(v, unique=True, trim="-"),
                  np.format_float_scientific(v, unique=True, trim="-"), key=len)
                 if array.dtype.itemsize == 4 else repr(float(v))) for v in column] for column in flat]
    restored = np.asarray([[np.nan if v == "null" else float(v) for v in column]
                           for column in columns], dtype=array.dtype).T.reshape(array.shape)
    if not np.array_equal(restored, array, equal_nan=True):
        raise ValueError("decimal column changed a source value")
    if np.isfinite(array).all() and restored.tobytes() != array.tobytes():
        raise ValueError("decimal column changed original dtype bytes")
    result = {"dtype": array.dtype.str, "component_shape": list(array.shape[1:]),
              "by_c_order_component": [",".join(column) for column in columns]}
    if constant_intervals and np.isfinite(array).all():
        intervals = []
        for column in flat:
            # Compare source bytes so -0.0 and +0.0 are never merged.
            values = column.copy().view(np.dtype((np.void, array.dtype.itemsize)))
            edges = np.r_[0, np.flatnonzero(values[1:] != values[:-1])+1, len(column)]
            intervals.append([[int(a+1), int(b), float(column[a])] for a, b in zip(edges[:-1], edges[1:])])
        compact = {"dtype": array.dtype.str, "component_shape": list(array.shape[1:]),
                   "inclusive_constant_intervals_by_component": intervals}
        rebuilt = np.asarray([[v for a, b, v in rows for _ in range(b-a+1)] for rows in intervals],
                             dtype=array.dtype).T.reshape(array.shape)
        if rebuilt.tobytes() != array.tobytes():
            raise ValueError("constant intervals changed original dtype bytes")
        if len(json.dumps(compact)) < len(json.dumps(result)):
            result = compact
    return result


def discrete_runs(array, steps):
    array = np.asarray(array)
    flat = array.reshape(len(array), -1)
    edges = np.r_[0, np.flatnonzero(np.any(flat[1:] != flat[:-1], axis=1))+1, len(array)]
    rows = [[int(steps[a]), int(steps[b-1]), array[a].tolist()] for a, b in zip(edges[:-1], edges[1:])]
    restored = np.concatenate([np.repeat(np.asarray(v, dtype=array.dtype)[None], b-a+1, axis=0) for a, b, v in rows])
    if restored.tobytes() != array.tobytes():
        raise ValueError("discrete runs changed original values")
    return {"dtype": array.dtype.str, "component_shape": list(array.shape[1:]), "inclusive_runs": rows}


def facts_arithmetic(task, measurements):
    """Existing raw-only arithmetic, never comparison with the report under review."""
    names = (("curve_extrema", "threshold_intervals_by_component", "raw_termination_intervals",
              "root_x_max_drawdown_witness", "root_first_last_xyz_terrain_m", "focus_operand_missing_fields")
             if task == "stair" else
             ("curve_extrema", "events", "missing_operands", "focus_raw_operands"))
    return {k: deepcopy(measurements[k]) for k in names if k in measurements}


def facts_episode_projection(task, row):
    import export_confirmation_raw as exporter
    result = exporter.facts_episode_projection(task, row)
    if "direct_measurements" in result:
        result["direct_measurements"] = facts_arithmetic(task, result["direct_measurements"])
    return result


def facts_original_material(task, value):
    """Project known episode/focus containers; retain original files as archives."""
    result = deepcopy(value)
    if "episode" in result:
        result["episode"] = facts_episode_projection(task, result["episode"])
    if "all_scope_rows" in result:
        result["all_scope_rows"] = [facts_episode_projection(task, row) for row in result["all_scope_rows"]]
    return result


def reconstructible_stair_fields(raw_names, direct_names, static_names, available_fields):
    # Task-owned numerical definitions; no behavior labels or necessary-fact answers.
    recipes = {
        "root_xyz_terrain_m": ("base_pos_w - terrain_env_origins_w (world axes)",
            ["raw_columns/base_pos_w", "static_values/terrain_env_origins_w"]),
        "support_xyz_terrain_m": ("support_pos_w - terrain_env_origins_w (world axes, bound reference points)",
            ["raw_columns/support_pos_w", "static_values/terrain_env_origins_w"]),
        "support_signed_vz_mps": ("support_lin_vel_w[...,2] (world-z signed velocity)",
            ["raw_columns/support_lin_vel_w"]),
        "support_speed3d_mps": ("sqrt(vx*vx + vy*vy + vz*vz), v=support_lin_vel_w",
            ["raw_columns/support_lin_vel_w"]),
        "root_x_drawdown_m": ("max(x[1:t])-x[t], x=base_pos_w[:,0]; not monotonic retreat",
            ["raw_columns/base_pos_w"]),
        "root_x_window_max_minus_start_m": (
            "w=ceil(definitions.arithmetic_queries.x_window_s/episode.dt_s); max(x[t-w:t] inclusive)-x[t-w]; null for t<=w",
            ["raw_columns/base_pos_w"]),
        "support_planar_speed_mps": ("sqrt(vx*vx + vy*vy), v=support_lin_vel_w (world-xy)",
            ["raw_columns/support_lin_vel_w"]),
        "support_reference_clearance_m": (
            "support_pos_w[...,2]-terrain_env_origins_w[2]-support_surface_height_terrain_m; reference point, not skin/contact truth",
            ["raw_columns/support_pos_w", "static_values/terrain_env_origins_w", "direct_columns/support_surface_height_terrain_m"]),
    }
    supplied = {"raw_columns": set(raw_names), "direct_columns": set(direct_names), "static_values": set(static_names)}
    return {name: {"status": "reconstructible_from_supplied", "formula": formula,
                   "operand_pointers": ["/data/"+p for p in operands],
                   "scope": "Complete clock coverage; C-order body components and native source dtype as declared. Control-step samples only."}
            for name, (formula, operands) in recipes.items()
            if name in available_fields and name not in direct_names
            and all(p.split("/")[1] in supplied[p.split("/")[0]] for p in operands)}


def related_episode(header, parts, npz_path):
    task = header["episode"]["task"]
    with np.load(npz_path, allow_pickle=False) as archive:
        raw = {k[5:]: archive[k].copy() for k in archive.files if k.startswith("raw__")}
    clock = "episode_length" if task == "stair" else "control_step"
    steps = raw[clock]
    if not np.array_equal(steps, np.arange(1, len(steps)+1)):
        raise ValueError("related_v2 requires the existing complete first-episode clock")
    wanted = (("base_pos_w", "base_quat_w", "support_pos_w", "support_lin_vel_w") if task == "stair"
              else ("pose", "planar_motion"))
    curve_names = (("root_roll_pitch_rad", "root_up_axis_world_z", "support_surface_height_terrain_m",
                    "support_planar_speed_mps", "support_reference_clearance_m",
                    "support_force_norm_n", "support_buffer_max_force_norm_n",
                    "forbidden_force_norm_n", "forbidden_buffer_max_force_norm_n") if task == "stair" else ())
    discrete = [k for k, a in raw.items() if a.dtype.kind in "biu" and k != clock]
    curves = {k: np.concatenate([np.asarray(p["direct_curves"][k], dtype=np.float64) for p in parts])
              for k in curve_names if k in header["direct_curve_fields"]}
    for name, a in raw.items():
        if hashlib.sha256(a.tobytes()).hexdigest() != header["raw_fields"][name]["c_order_bytes_sha256"]:
            raise ValueError("original episode slice field hash changed")
    episode = facts_episode_projection(task, header["episode"])
    arithmetic = episode.pop("direct_measurements", {})
    arithmetic.pop("focus_raw_operands", None)  # Complete raw columns already carry these Push samples.
    reconstructible = (reconstructible_stair_fields(set(wanted) & set(raw), set(curves),
                        header["static_values"], header["direct_curve_fields"]) if task == "stair" else {})
    if task == "push":
        reconstructible = {k: {"status": "reconstructible_from_supplied", "formula": "identity of supplied native-dtype raw column",
            "operand_pointers": ["/data/raw_columns/"+k]} for k in ("pose", "planar_motion")
            if k in raw and k in header["direct_curve_fields"]}
    return {"schema_version": "p3new.related_episode.v2", "profile": "related_v2",
        "episode": episode,
        "definitions": header["definitions"], "static_values": header["static_values"],
        "clock": {"source_field": clock, "first_step": int(steps[0]), "last_step": int(steps[-1]), "stride": 1},
        "decimal_encoding": "Each string is a comma-separated full-time numeric column, one value per consecutive control step; component order is C order. Cast raw values to declared dtype before arithmetic. Float32 uses the shortest round-tripping decimal, verified back to identical source bytes. Derived float64 finite values round-trip exactly; null means undefined, never zero. For surface height and forbidden force norms only, a shorter inclusive [first_step,last_step,constant_value] interval list may replace a column; these intervals cover every step and expand to identical dtype bytes. No time sampling.",
        "raw_columns": {k: decimal_columns(raw[k]) for k in wanted if k in raw},
        "raw_discrete": {k: discrete_runs(raw[k], steps) for k in discrete},
        "direct_columns": {k: decimal_columns(a, constant_intervals=k in {
            "support_surface_height_terrain_m", "forbidden_force_norm_n", "forbidden_buffer_max_force_norm_n"})
            for k, a in curves.items()},
        "direct_arithmetic": arithmetic,
        "arithmetic_scope": "Raw-only fixed arithmetic queries supplied for every episode, including empty intervals, before any report. Intervals are inclusive control-step sets, not necessary-fact answers, physical behavior labels or causal claims. Signed attitude, co-occurrence and repeated intervals need independent interpretation; no prior diagnosis or comparison PASS is supplied. Push stable_intervals means only the recorded five-condition AND inside its declared window. First dwell does not certify later stability.",
        "source_fields": header["raw_fields"], "source_direct_fields": header["direct_curve_fields"],
        "source_json_parts": header["parts"], "original_values_npz": header["original_values_npz"],
        "not_supplied_raw_fields": sorted(set(raw)-set(wanted)-set(discrete)-{clock}),
        "requested_but_not_recorded_fields": sorted(set(wanted)-set(raw)),
        "reconstructible_direct_fields": reconstructible,
        "not_transmitted_direct_fields": sorted(set(header["direct_curve_fields"])-set(curves)),
        "not_supplied_direct_fields": sorted(set(header["direct_curve_fields"])-set(curves)-set(reconstructible)),
        "availability_meaning": "raw_columns/raw_discrete/direct_columns contain supplied values; reconstructible_direct_fields are fully answerable from this packet, not missing observations. not_transmitted names only a serialization omission. not_supplied means recorded but neither supplied nor reconstructed here. requested_but_not_recorded means absent from the source archive. not_analyzed is reviewer work remaining, never a measurement status.",
        "scope_limit": "All recorded root and foot position/velocity steps remain. Joint strategy and force-vector components are not supplied; full raw arrays remain archived. Stair history-slot axis order remains in definitions/contact_history when recorded, but per-slot force values are not supplied in this view; observed per-slot timestamps and validity are not recorded. Instant force norm and history-buffer maximum norm stay distinct. Stair reference-point surface clearance is not toe skin/contact-pair truth; Push has no recorded foot/contact data. Repeated raw/direct pose and geometry-derived curves are omitted, not missing measurements."}


def interval_stats(spans, required=1, terminal=None, window_first=None):
    return [sum(b-a+1 for a, b in spans), len(spans), max((b-a+1 for a, b in spans), default=0),
            min((a for a, b in spans), default=None),
            min((a+required-1 for a, b in spans if b-a+1 >= required), default=None),
            sum(b-a+1 >= required for a, b in spans),
            None if terminal is None else any(a < terminal and b >= window_first for a, b in spans)]


def mask_intervals(mask):
    edges = np.flatnonzero(np.diff(np.r_[False, np.asarray(mask, bool), False].astype(int)))
    return [[int(a+1), int(b)] for a, b in edges.reshape(-1, 2)]


def stair_focus_arithmetic_navigation(out):
    """Name every registered component and count supplied summaries, never facts."""
    definitions = out["definitions"]["measurement_definitions"]
    bodies = definitions["bodies"]
    signals = definitions["protocol"]["episode_outcomes"]["signals"]
    query = definitions["arithmetic_queries"]
    tilt = signals["base_orientation_instability"]
    slip = signals["slip"]
    clearance = signals["insufficient_clearance"]
    # These are exactly the exporter's registered arithmetic queries. Bodies and
    # numeric thresholds come from the supplied task definitions, not reports.
    registrations = [
        ("root_x_window_max_minus_start_lt_query_m", "base", f"inclusive {query['x_window_s']}s root-x window max minus start < {query['x_window_gain_m']} m; undefined until full window"),
        ("abs_roll_or_pitch_gt_protocol_rad", "base", f"max(abs(roll),abs(pitch)) > {tilt['threshold']} rad"),
        ("root_roll_gt_protocol_rad", "base", f"signed roll > {tilt['threshold']} rad"),
        ("root_roll_lt_negative_protocol_rad", "base", f"signed roll < {-tilt['threshold']} rad"),
        ("root_pitch_gt_protocol_rad", "base", f"signed pitch > {tilt['threshold']} rad"),
        ("root_pitch_lt_negative_protocol_rad", "base", f"signed pitch < {-tilt['threshold']} rad"),
        ("root_up_axis_world_z_lt_zero", "base", "root up-axis world-z < 0 (unitless)"),
        ("support_force_gt_protocol_n", "support_effector", f"instant force norm > {slip['contact_force_threshold']} N"),
        ("all_support_force_le_protocol_n", "all_support", f"all support instant force norms <= {slip['contact_force_threshold']} N"),
        ("loaded_planar_speed_gt_protocol_mps", "support_effector", f"same-body instant force norm > {slip['contact_force_threshold']} N AND planar speed > {slip['speed_threshold']} m/s"),
        ("unloaded_speed3d_le_query_mps", "support_effector", f"same-body instant force norm <= {slip['contact_force_threshold']} N AND speed3d <= {query['unloaded_speed_mps']} m/s"),
        ("unloaded_vz_gt_query_mps", "support_effector", f"same-body instant force norm <= {slip['contact_force_threshold']} N AND signed world-z velocity > {query['signed_vz_mps']} m/s"),
        ("unloaded_vz_lt_negative_query_mps", "support_effector", f"same-body instant force norm <= {slip['contact_force_threshold']} N AND signed world-z velocity < {-query['signed_vz_mps']} m/s"),
        ("unloaded_clearance_lt_protocol_m", "support_effector", f"same-body instant force norm <= {slip['contact_force_threshold']} N AND reference-point clearance < {clearance['threshold']} m"),
        ("support_buffer_force_gt_protocol_n", "support_effector", f"history-buffer maximum force norm > {slip['contact_force_threshold']} N"),
        ("forbidden_buffer_force_gt_protocol_n", "forbidden_contact", f"history-buffer maximum force norm > {signals['forbidden_contact']['force_threshold_n']} N"),
    ]
    supplied, offset = {}, 0
    for name, width, dwell in out["predicate_order"]:
        supplied[name] = (offset, width, dwell)
        offset += width
    unknown = set(supplied) - {name for name, _, _ in registrations}
    if unknown:
        raise ValueError(f"unregistered Stair focus predicates: {sorted(unknown)}")
    catalog = []
    for name, role, expression in registrations:
        identity = bodies["support_effector"] if role == "all_support" else bodies[role]
        components = [identity] if role == "all_support" else [[body] for body in identity]
        if role == "base" and len(components) != 1:
            raise ValueError("Stair root query requires the supplied single base identity")
        info = supplied.get(name)
        if info is not None and info[1] != len(components):
            raise ValueError("Stair predicate component count differs from supplied bodies")
        for component, names in enumerate(components):
            catalog.append([name, component, role, names, None if info is None else info[2],
                            None if info is None else info[0]+component, expression])
    row_fields = {name: i for i, name in enumerate(out["scope_row_columns"])}
    grouped = {("all_scope", None, None, None, None): list(range(len(out["scope_rows"])))}
    for index, row in enumerate(out["scope_rows"]):
        success, outcome = (row[row_fields[k]] for k in ("original_success", "original_outcome"))
        for key in (("all_scope_by_original_outcome", None, None, success, outcome),
                    ("run_cell_by_original_outcome", row[row_fields["run_group"]], row[row_fields["cell"]], success, outcome)):
            grouped.setdefault(key, []).append(index)
    groups = []
    for key, members in grouped.items():
        counts = []
        for _, _, _, _, _, stats_index, _ in catalog:
            available = before = at_terminal = never = 0
            for index in members:
                stats = out["operands_in_same_row_order"][index][0]
                value = None if stats_index is None or stats is None else stats[stats_index]
                if value is None:
                    continue
                if len(value) != 7:
                    raise ValueError("Stair predicate summary must retain all seven fields")
                available += 1
                confirmation = value[4]
                terminal = out["scope_rows"][index][row_fields["steps"]]
                if confirmation is None:
                    never += 1
                elif confirmation < terminal:
                    before += 1
                elif confirmation == terminal:
                    at_terminal += 1
                else:
                    raise ValueError("Stair confirmation lies beyond original terminal")
            counts.append([available, len(members)-available, before, at_terminal, never])
        groups.append([*key, len(members), counts])
    return {
        "purpose": "Fixed arithmetic navigation only; not complete processes, necessary facts, behavior labels or causal claims. Same predicate does not mean the same complete process. Mixed run/cell groups are unmatched, not compatible counterexample matches.",
        "availability": "summary_supplied means this predicate/component's original seven-field interval summary is supplied; summary_not_supplied means it is not supplied. Empty intervals have no confirmation in the supplied summary, not missing observations. This does not certify per-step validity or full physics coverage. See /definitions/measurement_definitions, /scope_rows missing_fields, and /operands_in_same_row_order/*/7; definitions retain early undefined root-x windows and history timestamp/validity limits.",
        "confirmation_query": "first_required_confirmation_step < original T; ==T is separate; null means no confirmation in the supplied summary. Counts are episode-members, not intervals or true-step counts. Same-body components remain separate; any_body_or_stats is not substituted.",
        "predicate_component_columns": ["predicate", "component_index", "body_role", "body_names", "required_consecutive_steps", "original_flat_stats_index", "strict_arithmetic_query"],
        "predicate_components": catalog,
        "group_columns": ["scope", "run_group", "cell", "original_success", "original_outcome", "member_count", "counts_in_predicate_component_order"],
        "count_columns": ["summary_supplied_members", "summary_not_supplied_members", "confirmed_before_T_members", "confirmed_at_T_members", "no_confirmation_in_supplied_summary_members"],
        "groups": groups,
        "original_operands": "/operands_in_same_row_order/*/0; all seven original fields remain unchanged; /scope_rows retains episode identity, T and dt_s",
    }


def related_focus(focus):
    """Fixed full-member arithmetic distribution table, not unseen trajectories."""
    task = focus["task"]
    focus = facts_original_material(task, focus)
    columns = ["episode_id", "run_group", "cell", "steps", "dt_s", "original_success", "original_outcome", "missing_fields"]
    member_columns = ["episode_id", "run_group", "scope", "cell", "original_success", "original_outcome"]
    membership = {}
    for row in focus["complete_task_pool_members"]:
        key = tuple(row[k] for k in member_columns[1:])
        membership.setdefault(key, []).append(row["episode_id"])
    out = {"schema_version": "p3new.related_focus.v1", "profile": "related_v1", "task": task,
        "focus": focus["focus"], "questions": focus["questions"], "definitions": focus["task_definitions"],
        "complete_pool_group_columns": member_columns[1:] + ["episode_ids"],
        "complete_pool_groups": [[*key, ids] for key, ids in membership.items()],
        "scope_denominator": focus["scope_denominator"], "scope_row_columns": columns,
        "scope_rows": [[r[k] for k in columns] for r in focus["all_scope_rows"]], "operands_in_same_row_order": [],
        "interval_arithmetic_columns": ["true_steps", "run_count", "longest_consecutive_steps", "first_start_step",
            "first_required_confirmation_step", "runs_at_least_required", "present_in_final_1s_before_T"],
        "scope_limit": "Every member of this scope is included regardless of report positivity. Direct interval/extrema arithmetic supports denominators, matching and fixed process distributions; it is not semantic necessary-fact truth. Complete unselected-episode trajectories, unknown foot sequences, per-interval cross-body co-occurrence and arbitrary causal mechanisms are not_supplied here. Complete related trajectories are supplied separately for ALL 24 selected episodes. Unprovided recorded operands are not the same as not_recorded."}
    for row in focus["all_scope_rows"]:
        m, T, dt = row["direct_measurements"], row["steps"], row["dt_s"]
        if task == "stair":
            spans = m["threshold_intervals_by_component"]
            signals = focus["task_definitions"]["measurement_definitions"]["protocol"]["episode_outcomes"]["signals"]
            required = {k: 1 for k in spans}
            required.update(abs_roll_or_pitch_gt_protocol_rad=signals["base_orientation_instability"]["consecutive_control_steps"],
                loaded_planar_speed_gt_protocol_mps=signals["slip"]["consecutive_control_steps"],
                unloaded_clearance_lt_protocol_m=signals["insufficient_clearance"]["consecutive_control_steps"],
                unloaded_speed3d_le_query_mps=math.ceil(.5/dt), unloaded_vz_gt_query_mps=3, unloaded_vz_lt_negative_query_mps=3)
            out.setdefault("predicate_order", [[k, len(v), required[k]] for k, v in spans.items()])
            if out["predicate_order"] != [[k, len(v), required[k]] for k, v in spans.items()]:
                raise ValueError("focus predicate availability varies; preserve original profile for this scope")
            args = {"terminal": T, "window_first": max(1, T-math.ceil(1/dt)+1)}
            stats = [interval_stats(v, required[k], **args) for k, values in spans.items() for v in values]
            or_names = ["loaded_planar_speed_gt_protocol_mps", "unloaded_clearance_lt_protocol_m"]
            ors = []
            for name in or_names:
                mask = np.zeros(T, bool)
                for values in spans.get(name, []):
                    for a, b in values:
                        mask[a-1:b] = True
                ors.append(interval_stats(mask_intervals(mask), required[name], **args) if name in spans else None)
            out["any_body_or_order"] = or_names
            root = []
            for name, indices in (("root_xyz_terrain_m", [0]), ("root_roll_pitch_rad", [0, 1]), ("root_up_axis_world_z", [0])):
                components = m["curve_extrema"].get(name, {}).get("c_order_components", [])
                root += [[v["min"], v["min_step"], v["max"], v["max_step"]] if (v := components[j] if j < len(components) else None) else None for j in indices]
            feet = []
            for j in range(len(focus["task_definitions"]["measurement_definitions"]["bodies"]["support_effector"])):
                values = []
                for name, bound in (("support_reference_clearance_m", "min"), ("support_planar_speed_mps", "max"), ("support_signed_vz_mps", "absmax")):
                    components = m["curve_extrema"].get(name, {}).get("c_order_components", [])
                    if j >= len(components):
                        values.append(None)
                        continue
                    v = components[j]
                    if bound == "absmax":
                        bound = "min" if abs(v["min"]) > abs(v["max"]) else "max"
                    values.append([v[bound], v[bound+"_step"]])
                feet.append(values)
            out["stair_operand_columns"] = ["predicate_component_stats", "any_body_or_stats", "root_x_roll_pitch_upz_extrema_min_step_max_step",
                "foot_min_clearance_max_planar_speed_signed_vz_of_largest_magnitude_with_steps", "maximum_drawdown_witness",
                "root_first_last_xyz", "original_termination_intervals", "missing_fields"]
            out["stair_arithmetic_note"] = "low-progress window predicate is already computed over the full inclusive 0.5s window and requires one true step, not a second dwell. Same-foot unloaded dwell remains per body; never substitute any-foot alternating OR. Before-terminal confirmation requires step<T. OR slip/clearance masks are formed before streak counting."
            out["operands_in_same_row_order"].append([stats, ors, root, feet, m["root_x_max_drawdown_witness"],
                m["root_first_last_xyz_terrain_m"], m["raw_termination_intervals"], m["focus_operand_missing_fields"]])
        else:
            raw, terminal = m["focus_raw_operands"], row["original_terminal_record"]
            if raw["conditions_rle"] is None:
                out["operands_in_same_row_order"].append({"not_recorded": "conditions", "original_events": terminal["push_events"]})
                continue
            conditions = np.concatenate([np.repeat(np.asarray(v, bool)[None], b-a+1, axis=0) for a, b, v in raw["conditions_rle"]])
            if len(conditions) != T or raw["first_step"] != 1:
                raise ValueError("group conditions do not cover the complete first episode")
            definitions = next(iter(focus["task_definitions"].values()))
            dwell = definitions["time"]["required_dwell_steps"]
            points = {v["control_step"]: v for v in raw["landmark_samples"]}
            landmarks = {v["event_index"]: v for v in raw["event_landmarks"]}
            originals = {v["event_index"]: v for v in terminal["push_events"]}
            def sample(step):
                v = points.get(step)
                return None if v is None else [step, *[None if v[k] is None else ",".join(decimal_columns(np.asarray(v[k], np.float32)[None])["by_c_order_component"]) for k in ("pose", "planar_motion")]]
            events = []
            for event in m["events"]:
                idx, anchor = event["event_index"], event["anchor_step"]
                original, landmark = originals[idx], landmarks[idx]
                if "window_first_speed_maximum_step" not in landmark:
                    raise ValueError("related_v1 needs exporter --profile related_v1 earliest-peak samples")
                ws, latch = event["window_observed_steps"], event["first_dwell_end_step_by_arithmetic"]
                stop = min([e["control_step"] for e in terminal["push_events"] if e["control_step"] > anchor], default=T)
                after = (np.arange(1, T+1) > (latch or T)) & (np.arange(1, T+1) <= stop) & ~conditions.all(axis=1)
                interruptions = [[b+1, b-a+1, (~conditions[b]).tolist()] for a, b in event["pre_latch_stable_intervals"]
                                 if ws and b < ws[-1] and (latch is None or b < latch)]
                events.append([idx, anchor, original["requested_delta_v_world_mps"], original["observed_delta_v_world_mps"],
                    original["injection_verified"], original["recovery_control_step"], original["recovery_censored"],
                    [ws[0], ws[-1], len(ws)] if ws else [None, None, 0],
                    [interval_stats(event["condition_false_intervals"][k])[:4] for k in definitions["condition_fields"]],
                    interval_stats(event["stable_intervals"], dwell)[:6], latch,
                    interval_stats(event["pre_latch_stable_intervals"], dwell)[:6], interruptions,
                    [sample(s) for s in (anchor, anchor+1, landmark["window_first_speed_maximum_step"], ws[-1] if ws else None)],
                    interval_stats(mask_intervals(after))[:4] if latch is not None else None])
            latches = [event["first_dwell_end_step_by_arithmetic"] for event in m["events"]]
            all_latched = len(latches) == terminal["planned_push_count"] and bool(latches) and all(x is not None for x in latches)
            later = (np.arange(1, T+1) > max(latches)) & ~conditions.all(axis=1) if all_latched else None
            out["push_event_columns"] = ["event_index", "anchor", "requested_delta_v_xy", "observed_delta_v_xy", "original_verified",
                "original_latch", "original_censored", "observed_window_first_last_count", "condition_false_count_runs_maxstreak_first",
                "all_window_stable_stats", "arithmetic_first_dwell", "pre_latch_stable_stats", "interruptions_step_prior_dwell_failed_condition_mask",
                "anchor_first_response_first_speed_peak_window_last_full_pose_motion", "post_latch_to_next_injection_unstable_count_runs_maxstreak_first"]
            out["push_operand_columns"] = ["planned_events", "actual_events", "whole_condition_false_stats", "terminal_pose_motion",
                "actual_event_rows", "post_all_required_latches_unstable_stats", "terminal_conditions", "missing_fields", "requested_steps_not_observed"]
            out["push_sample_encoding"] = "[control_step,pose_csv,planar_motion_csv]; comma-separated components follow the original field order, each shortest decimal round-trips to source float32. null is unavailable, not zero. Original outcome/events retained; arithmetic is not a replacement outcome."
            out["operands_in_same_row_order"].append([terminal["planned_push_count"], terminal["push_event_count"],
                [interval_stats(mask_intervals(~conditions[:, i]))[:4] for i in range(5)], sample(T), events,
                None if later is None else interval_stats(mask_intervals(later))[:4], conditions[-1].tolist(),
                raw["missing_fields"], raw["requested_steps_not_observed"]])
    if len(out["scope_rows"]) != focus["scope_denominator"] or len(out["operands_in_same_row_order"]) != focus["scope_denominator"]:
        raise ValueError("compact group lost a denominator member")
    if task == "stair":
        out["arithmetic_navigation"] = stair_focus_arithmetic_navigation(out)
    return out


def _prepare_task_definition_inputs(raw_export, selected, focuses, paths, push_resolved_cell_manifest, *, required):
    """Verify explicit contracts and route them before creating any output."""
    if not paths:
        if required:
            raise ValueError("Confirmation requires explicit --task-definition files for every task/numeric source")
        if push_resolved_cell_manifest is not None:
            raise ValueError("--push-resolved-cell-manifest requires --task-definition")
        return {}, None
    import prepare_review_task_contract as contracts

    source_path = raw_export / "_private/sources.json"
    watched = {str(source_path.resolve()): digest(source_path), str(Path(contracts.__file__).resolve()): digest(Path(contracts.__file__))}
    expected, source_runs = {}, {}
    task_ids = {"unitree_stair_climbing": "stair", "unitree_push_recovery": "push"}
    for source in read(source_path)["sources"]:
        numeric_path = Path(source["manifest_path"]).resolve()
        numeric_sha = digest(numeric_path)
        numeric = read(numeric_path)
        if numeric_sha != source["manifest_sha256"] or numeric["identity"] != source["identity"]:
            raise ValueError("Task definition numeric source identity/hash differs: " + str(numeric_path))
        task = task_ids[numeric["identity"]["task_id"]]
        runs = ([seed["run_index"] for seed in numeric["metadata"]["seeds"]] if task == "stair"
                else [numeric["identity"]["run_index"]])
        run = source["run"]
        if type(run) is not int or run not in runs:
            raise ValueError("Task definition source run differs from numeric manifest")
        key, run_key = (task, numeric_sha), (task, f"run_{run:02d}")
        if run_key in source_runs or (key in expected and expected[key] != numeric_path):
            raise ValueError("Ambiguous task/run/numeric source for task definitions")
        expected[key], source_runs[run_key] = numeric_path, key
        watched[str(numeric_path)] = numeric_sha
    supplied = {}
    for path in paths:
        path = Path(path).resolve()
        contract = read(path)
        key = (contract["task"], contract["numeric_manifest"]["sha256"])
        if key in supplied:
            raise ValueError("Duplicate task definition for task/numeric SHA: " + repr(key))
        supplied[key] = (path, contract)
        watched[str(path)] = digest(path)
    if not expected or set(supplied) != set(expected):
        raise ValueError(f"Task definitions do not cover numeric sources: missing={sorted(set(expected)-set(supplied))}, unexpected={sorted(set(supplied)-set(expected))}")
    cell_path = Path(push_resolved_cell_manifest).resolve() if push_resolved_cell_manifest is not None else None
    if any(task == "push" for task, _ in expected) and cell_path is None:
        raise ValueError("Push task definitions require explicit --push-resolved-cell-manifest")
    if cell_path is not None:
        watched[str(cell_path)] = digest(cell_path)
    references = {}
    for key, numeric_path in expected.items():
        path, contract = supplied[key]
        contracts.verify(contract, numeric_path, source_path, cell_path if key[0] == "push" else None)
        references[key] = material(path, str(path), "task_definition", f"task-definition-{key[0]}-{key[1][:12]}")
    by_run = {key: references[source_key] for key, source_key in source_runs.items()}
    routes = {}
    for index, selected_row in enumerate(selected, 1):
        header = read(raw_export / "episodes" / selected_row["episode_id"] / "episode.json")
        episode = header["episode"]
        key = (episode["task"], episode["run_group"])
        if key not in by_run:
            raise ValueError("Episode has no verified task/run definition: " + repr(key))
        routes[f"episode-{index:02d}"] = [by_run[key]]
    for index, name in enumerate(focuses, 1):
        task = read(raw_export / name)["task"]
        # Every focus also includes the complete task pool, even a one-run focus.
        refs = {ref["id"]: ref for (owner, _), ref in sorted(by_run.items()) if owner == task}
        if not refs:
            raise ValueError("Focus has no verified task definitions: " + task)
        routes[f"focus-{index:02d}"] = list(refs.values())
    receipt = {"source_manifest": {"path": str(source_path.resolve()), "sha256": watched[str(source_path.resolve())]},
               "contracts": list(references.values()),
               "task_run_bindings": [{"task": task, "run_group": run, "numeric_manifest_sha256": source_runs[task, run][1],
                                       "task_definition_id": ref["id"]} for (task, run), ref in sorted(by_run.items())],
               "verified_input_sha256": watched,
               "routing": "episode: own task/run; focus: all task runs, identical numeric manifests deduplicated"}
    return routes, receipt


def pack(raw_export, output, protocol, capture_card, context_tokens=None, profile="original", max_output_tokens=65536,
         *, task_definitions=None, push_resolved_cell_manifest=None, review_assignment=None):
    raw_export, output = raw_export.resolve(), output.resolve()
    source_manifest = read(raw_export / "manifest.json")
    confirmation = source_manifest["purpose"] == "confirmation"
    if source_manifest["purpose"] not in {"development", "confirmation"} or source_manifest["stage"] != "facts":
        raise ValueError("This preparation command requires development or confirmation facts")
    if confirmation:
        frozen = source_manifest.get("frozen_inputs", {})
        if (read(capture_card).get("status") != "FROZEN"
                or frozen.get("capture_card", {}).get("sha256") != digest(capture_card)
                or frozen.get("review_protocol", {}).get("sha256") != digest(protocol)):
            raise ValueError("Confirmation requires the exact frozen capture card and review protocol")
    purpose = "confirmation" if confirmation else "old_development_numeric_packing_offline_only"
    if output.exists():
        raise ValueError("Output must be a new directory; preserve previous evidence")
    if profile not in {"original", "related_v2"}:
        raise ValueError("New facts packing uses original or related_v2; archived related_v1 remains unchanged")
    if profile == "related_v2" and source_manifest.get("material_profile") != "related_v1":
        raise ValueError("related_v2 packing requires raw export related_v1, which includes earliest-peak samples")
    source_hashes = {"raw_manifest": digest(raw_export / "manifest.json"),
                     "protocol": digest(protocol), "capture_card": digest(capture_card),
                     "packer": digest(Path(__file__)), "runner": digest(Path(runner.__file__)),
                     "facts_projector": digest(HERE / "export_confirmation_raw.py")}
    private_binding = source_manifest.get("private_source_manifest")
    if private_binding is None:
        if confirmation:
            raise ValueError("Confirmation raw export lacks private_source_manifest binding; export a new bound directory")
        private_binding_receipt = {"status": "UNBOUND_LEGACY_DEVELOPMENT"}
    else:
        if (private_binding.get("file") != "_private/sources.json"
                or private_binding.get("sha256") != digest(raw_export / "_private/sources.json")):
            raise ValueError("Raw export private_source_manifest binding differs")
        source_hashes["private_source_manifest"] = private_binding["sha256"]
        private_binding_receipt = {"status": "VERIFIED", **private_binding}
    catalogue = {item["file"]: item for item in source_manifest["readable_materials"]}
    selected = read(raw_export / "selected_episodes.json")
    focuses = sorted(name for name in catalogue if name.startswith("focus/"))
    if len(selected) != 24 or len(focuses) != 12:
        raise ValueError("Expected the registered 24 episode / 12 focus selection")
    definition_routes, definition_receipt = _prepare_task_definition_inputs(
        raw_export, selected, focuses, task_definitions, push_resolved_cell_manifest, required=confirmation)
    output.mkdir(parents=True)
    copied = {}

    def copy_raw(name, expected=None):
        if name in copied:
            return copied[name]
        source = raw_export / name
        expected = expected or catalogue[name]
        if digest(source) != expected["sha256"] or source.stat().st_size != expected["bytes"]:
            raise ValueError("Raw export drift: " + name)
        destination = output / "materials" / "raw" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        copied[name] = {"sha256": digest(destination), "bytes": destination.stat().st_size}
        return copied[name]

    # Judge clauses have one source; launch and cross-packet duties stay with the operator.
    protocol_path = output / "materials" / "review_protocol.json"
    save(protocol_path, {"status": "FROZEN_FOR_CONFIRMATION_NOT_RELEASED" if confirmation else "DRAFT_NOT_FROZEN_NOT_RELEASED",
        **runner.packet_protocol(protocol), "this_packet_modality": "numeric_only",
        "numeric_only_scope": ["No visual evidence is supplied in this packet. major_visible_forms must be empty.",
            "Media absent from this packet means not_supplied, not proof that it was never recorded.",
            "Establish necessary facts independently; do not score an unseen report or treat arithmetic queries as truth labels.",
            "Material IDs, existing neutral episode aliases and JSON pointers identify evidence; never open unlisted references."]})
    units = []
    for index, item in enumerate(selected, 1):
        alias = item["episode_id"]
        header_name = f"episodes/{alias}/episode.json"
        header = read(raw_export / header_name)
        names = [header_name]
        for part in header["parts"]:
            name = f"episodes/{alias}/{part['file']}"
            if catalogue[name]["sha256"] != part["sha256"]:
                raise ValueError("Episode part catalogue mismatch")
            names.append(name)
        original = header["original_values_npz"]
        copy_raw(f"episodes/{alias}/{original['file']}", original)
        units.append((f"episode-{index:02d}", "episode", header["episode"]["task"], alias, names))
    for index, name in enumerate(focuses, 1):
        focus = read(raw_export / name)
        if len(focus["all_scope_rows"]) != focus["scope_denominator"]:
            raise ValueError("Focus scope denominator mismatch")
        units.append((f"focus-{index:02d}", "focus", focus["task"], focus["focus"], [name]))

    rows = []
    for packet, axis, task, scope, names in units:
        assignment = runner.assignment_fields(review_assignment, packet)
        packet_dir = output / "packets" / packet
        question_path = packet_dir / "questions.json"
        save(question_path, {"status": "FROZEN_RULES_NOT_RELEASED" if confirmation else "DRAFT_NOT_FROZEN", "unit_id": packet,
            "axis": axis, "task": task, "evidence_scope": scope, "material_profile": profile,
            "five_question_bundles": [{"id": key, "question": question} for key, question in QUESTIONS],
            "scope_instruction": (
                "本单位仅一个回合：完整轨迹的主要阶段、重复/重叠过程、关键量与时段均须审读。"
                "Q1只核本回合原始结局与实际覆盖。Q2/Q3中的组级频率和跨回合反例在focus单位回答，"
                "本包未供同组全体，不能从单回合推出组级结论，也不要求另造组级必要事实。"
                if axis == "episode" else
                "本单位包含全部scope成员及全任务成员分母，按同域结局比较所有成员，"
                "不以选中回合或自动报告阳性替代全组。过程阈值/直接摘要是原始算术查询，"
                "自行确定哪些事实必要；未提供的完整时序操作数明确记not_supplied。"),
            "facts_output": "Use the protocol facts findings and evidence locators; no report score or reward-change task.",
            "report_output": "Only after facts seals exist, assess all supplied report claims and own necessary facts by axis; preserve polarity, scope, provenance and unknowns."})
        materials = [material(protocol_path, "../../materials/review_protocol.json", "review_protocol", "protocol"),
                      material(question_path, "questions.json", "focus_questions", "questions")]
        for number, name in enumerate(names):
            copy_raw(name)
            if profile == "original":
                projected = facts_original_material(task, read(raw_export / name))
                projected["original_archive_material"] = {"file": name, **copied[name]}
                path = output / "materials/facts" / name
                save(path, projected)
                materials.append(material(path, "../../materials/facts/" + name,
                                          "raw_numeric", f"raw-{number:03d}"))
        supplied_bytes = sum((output / "materials/facts" / name).stat().st_size for name in names) if profile == "original" else 0
        supplied_names = ["materials/facts/"+name for name in names]
        if profile == "related_v2":
            if axis == "episode":
                header = read(raw_export / names[0])
                compact = related_episode(header, [read(raw_export / name) for name in names[1:]],
                    raw_export / f"episodes/{scope}/{header['original_values_npz']['file']}")
            else:
                compact = related_focus(read(raw_export / names[0]))
            compact["source_materials"] = [{"file": name, **copied[name]} for name in names]
            compact_path = output / "materials/related" / f"{packet}.json"
            save(compact_path, compact)
            materials.append(material(compact_path, f"../../materials/related/{packet}.json", "raw_numeric", "related-raw"))
            supplied_bytes = compact_path.stat().st_size
            supplied_names = [f"materials/related/{packet}.json"]
        materials.extend(definition_routes.get(packet, []))
        facts_path = packet_dir / "facts.manifest.json"
        save(facts_path, {"schema_version": "p3new.confirmation_review_materials.v1",
            "packet_id": packet, "stage": "facts", "purpose": purpose,
            "reference_contract": runner.coverage.BLOCK_CONTRACT,
            "material_profile": profile,
            "max_output_tokens": max_output_tokens,
            **assignment,
            "materials": materials})
        # Deliberately NOT the executable runner schema: it must not accidentally
        # pass as a report assessment with only raw evidence and no report text.
        save(packet_dir / "report.manifest.draft.json", {
            "schema_version": "p3new.confirmation_review_materials.draft.v1",
            "target_schema_version": "p3new.confirmation_review_materials.v1",
            "reference_contract": runner.coverage.BLOCK_CONTRACT,
            "status": "NOT_READY_REPORT_INPUTS_AND_REAL_FACTS_SEALS_MISSING",
             "packet_id": packet, "stage": "report", "purpose": purpose,
             "max_output_tokens": max_output_tokens,
             **assignment,
            "facts_manifest_sha256": digest(facts_path), "materials": materials,
            "required_report_materials": (
                ["Complete selected-episode full derivation slice with required shared definitions and original provenance"]
                if axis == "episode" else
                ["Whole task bounded report", "Relevant group full claims with their full raw denominator",
                 "Corresponding bounded_request; label offline production serialization unless actually sent"]),
            "required_separately": ["Each judge's genuine complete facts answer and matching seal",
                "Root confirmation that both judges completed/sealed ALL required facts units before any report reveal"],
            "actual_report_serialization": None})
        variants = {}
        reference_parts = None
        for judge in runner.review_judges({"packet_id": packet, **assignment}):
            prepared = output / "prepared" / packet / judge
            card = runner.prepare(facts_path, judge, prepared)
            body = (prepared / "request.http_body.json").read_bytes()
            payload = json.loads(body)
            parts = payload["messages"][1]["content"]
            if reference_parts is not None and parts != reference_parts:
                raise ValueError("Reviewer evidence bytes/order differ")
            reference_parts = parts
            input_bytes = len(payload["messages"][0]["content"].encode("utf-8")) + sum(
                len(block["text"].encode("utf-8")) for block in parts)
            # Conservative engineering estimate, not a provider tokenizer count.
            input_estimate = input_bytes + 1024
            required = input_estimate + payload["max_tokens"] + 8192
            variants[judge] = {"serialization": "offline production serialization; not sent",
                "http_body_bytes": len(body), "http_body_sha256": card["request_sha256"],
                "model_visible_text_utf8_bytes": input_bytes,
                "input_token_estimate": input_estimate, "output_token_reserve": payload["max_tokens"],
                "additional_context_margin": 8192, "estimated_required_context_tokens": required,
                "context_limit_tokens_supplied": context_tokens,
                "fits_supplied_limit_under_estimate": None if context_tokens is None else required <= context_tokens,
                "http_body_limit_pass": len(body) <= runner.MAX_BODY, "actual_calls": 0,
                "carrier": "DeepSeek HTTP body, offline" if judge == "deepseek" else "fresh-agent material carrier; not a GPT API request"}
        rows.append({"packet_id": packet, "axis": axis, "task": task, "scope": scope,
            "facts_manifest": f"packets/{packet}/facts.manifest.json", "facts_manifest_sha256": digest(facts_path),
            "required_raw_json_files": names, "raw_readable_utf8_bytes": sum(copied[n]["bytes"] for n in names),
            "supplied_json_files": supplied_names, "supplied_readable_utf8_bytes": supplied_bytes,
            "all_required_parts_in_same_context": True, "reviewer_materials_identical": True, "variants": variants})
        if definition_receipt is not None:
            rows[-1]["task_definition_ids"] = [ref["id"] for ref in definition_routes[packet]]
    if digest(protocol) != source_hashes["protocol"] or digest(capture_card) != source_hashes["capture_card"]:
        raise ValueError("Draft protocol/card changed during packing; retain output as incomplete")
    if private_binding is not None and digest(raw_export / "_private/sources.json") != private_binding["sha256"]:
        raise ValueError("Raw export private source changed during packing; retain output as incomplete")
    if definition_receipt is not None:
        for path, expected_sha in definition_receipt["verified_input_sha256"].items():
            if digest(Path(path)) != expected_sha:
                raise ValueError("Task definition input changed during packing: " + path)
    receipt = {"status": "FROZEN_INPUTS_FACTS_PREPARED_REPORT_NOT_READY_NOT_RELEASED" if confirmation else "OFFLINE_FACTS_PACKED_REPORT_NOT_READY_NOT_FROZEN",
        "purpose": "confirmation" if confirmation else "development",
        "source": {"raw_export": str(raw_export), "protocol": str(protocol), "capture_card": str(capture_card)},
        "source_sha256": source_hashes, "material_profile": profile, "episode_count": 24, "focus_count": 12,
        "raw_private_source_binding": private_binding_receipt,
        "raw_files_copied": copied, "packets": rows,
        "context_estimate_method": "1 token per model-visible UTF-8 byte + 1024 chat overhead; configured output +8192 margin. Conservative sizing assumption, not a verified tokenizer count or model capacity claim.",
        "context_readiness": "UNVERIFIED: provider context limits/tokenizer and fresh-agent context budget must be frozen before calling",
        "media": "not_supplied; numeric-only packing, no image resizing or visual qualification",
        "reports": "draft manifests only; no automatic report was read, no fabricated facts seal, no report request created",
        "raw_npz": ("selected original slices copied for provenance audit only; every original readable JSON part is actually in each judge's request" if profile == "original" else
                    "original NPZ/JSON kept as hashed archive copies; fixed whole-time related columns and full-member distribution operands are actually supplied identically to both judges; see each material's not_supplied fields and scope_limit"),
        "actual_calls": 0, "actual_usage": None, "actual_cost_usd": None, "semantic_review_performed": False}
    if definition_receipt is not None:
        receipt["task_definitions"] = definition_receipt
    save(output / "packing_receipt.json", receipt)
    return receipt


def report_pool_operands(raw_export, task, aliases):
    """Whole-pool direct operands for the fixed default report, after facts sealing."""
    files = sorted((raw_export / "pool").glob(f"{task}_*.json"))
    rows = [r for path in files for r in read(path)["pool_rows"]]
    definitions = read(raw_export / "definitions" / f"{task}.json")
    columns = ["episode_id", "run_group", "scope", "cell", "steps", "dt_s", "original_success", "original_outcome"]
    result = {"stage": "report_verification_only", "task": task, "definitions": definitions,
        "source_materials": [{"file": str(p.relative_to(raw_export)), "sha256": digest(p)} for p in files],
        "member_columns": columns + ["original_logical_id"],
        "members": [[r[k] for k in columns]+[aliases[r["episode_id"]]] for r in rows], "operands": [],
        "scope": "Whole original pool, never limited to this focus. Fixed default-report arithmetic operands, not new necessary-fact answers. Earlier sealed facts are unchanged.",
        "hash_order_definition": "sha256(p3new-20260929|task|original_logical_id), then original_logical_id; arithmetic order only",
        "hash_order_original_ids": sorted([aliases[r["episode_id"]] for r in rows],
            key=lambda x: (hashlib.sha256(f"p3new-20260929|{task}|{x}".encode()).hexdigest(), x))}
    for row in rows:
        m, T, dt = row["direct_measurements"], row["steps"], row["dt_s"]
        if task == "stair":
            predicates = m["threshold_intervals_by_component"]
            required = definitions["measurement_definitions"]["protocol"]["episode_outcomes"]["signals"]["base_orientation_instability"]["consecutive_control_steps"]
            names = ["abs_roll_or_pitch_gt_protocol_rad", "root_x_window_max_minus_start_lt_query_m",
                     "unloaded_speed3d_le_query_mps", "root_up_axis_world_z_lt_zero"]
            result["operand_columns"] = ["earliest_required_run_step_per_fixed_predicate", "missing_fields"]
            result["predicates_in_order"] = names
            firsts = []
            for name, width in zip(names, [required, 1, math.ceil(.5/dt), 1]):
                starts = [a+width-1 for spans in predicates.get(name, []) for a, b in spans if b-a+1 >= width]
                firsts.append(min(starts, default=None))
            result["operands"].append([firsts, row["missing_fields"]])
            result["arithmetic_note"] = "First confirmation is supplied even when equal to T; group pre-terminal predicates require confirmation<T. Same-body unloaded runs are evaluated separately before taking the earliest; time bins use (T-confirmation)*dt."
        else:
            raw, terminal = m["focus_raw_operands"], row["original_terminal_record"]
            conditions = np.concatenate([np.repeat(np.asarray(v, bool)[None], b-a+1, axis=0) for a, b, v in raw["conditions_rle"]]) if raw["conditions_rle"] is not None else None
            points = {v["control_step"]: v for v in raw["landmark_samples"]}
            landmarks = {v["event_index"]: v for v in raw["event_landmarks"]}
            originals = {v["event_index"]: v for v in terminal["push_events"]}
            events = []
            for event in m["events"]:
                index, anchor, latch = event["event_index"], event["anchor_step"], event["first_dwell_end_step_by_arithmetic"]
                ws = event["window_observed_steps"]
                interruption_rows = [[b+1, b-a+1, (~conditions[b]).tolist()] for a, b in event["pre_latch_stable_intervals"]
                    if conditions is not None and ws and b < ws[-1] and (latch is None or b < latch)]
                first, peak, last = (points.get(s) for s in (anchor+1, landmarks[index]["window_first_speed_maximum_step"], ws[-1] if ws else None))
                start = points.get(anchor)
                xy = ([last["planar_motion"][i]-start["planar_motion"][i] for i in (2, 3)]
                      if start and last and start["planar_motion"] and last["planar_motion"] else None)
                events.append([index, anchor, originals[index]["injection_verified"], originals[index]["recovery_control_step"],
                    originals[index]["recovery_censored"], len(ws),
                    [[sum(b-a+1 for a, b in spans), spans[0][0] if spans else None] for spans in event["condition_false_intervals"].values()],
                    sum(b-a+1 for a, b in event["stable_intervals"]),
                    max((b-a+1 for a, b in event["pre_latch_stable_intervals"]), default=0), interruption_rows,
                    None if first is None else first["pose"][3], None if peak is None else [peak["control_step"], peak["pose"][3]], xy])
            latches = [event["first_dwell_end_step_by_arithmetic"] for event in m["events"]]
            all_latched = bool(latches) and len(latches) == terminal["planned_push_count"] and all(s is not None for s in latches)
            later = (np.arange(1, T+1) > max(latches)) & ~conditions.all(axis=1) if all_latched and conditions is not None else None
            result["operand_columns"] = ["planned_events", "executed_events", "events", "terminal_conditions", "unstable_steps_after_all_required_latches", "missing_fields"]
            result["event_columns"] = ["event_index", "anchor", "original_verified", "original_latch", "original_censored", "observed_window_steps",
                "per_condition_false_count_first_step", "all_window_stable_steps", "maximum_pre_latch_dwell", "interruptions_step_prior_dwell_false_mask",
                "window_first_speed", "earliest_peak_step_speed", "window_last_minus_anchor_dx_dy"]
            result["condition_order"] = next(iter(definitions.values()))["condition_fields"]
            result["operands"].append([terminal["planned_push_count"], terminal["push_event_count"], events,
                None if conditions is None else conditions[-1].tolist(), None if later is None else int(later.sum()), raw["missing_fields"]])
    # Factor only repeated membership metadata; every row and arithmetic operand stays.
    columns, members = result["member_columns"], result["members"]
    common = ["run_group", "scope", "cell", "dt_s"]
    kept = [name for name in columns if name not in common]
    groups = {}
    for index, row in enumerate(members):
        groups.setdefault(tuple(row[columns.index(name)] for name in common), []).append(index)
    result["member_columns"] = kept
    result["members"] = [[row[columns.index(name)] for name in kept] for row in members]
    result["member_group_columns"] = common + ["member_row_indices_zero_based"]
    result["member_groups"] = [[*key, indexes] for key, indexes in groups.items()]
    rebuilt = [None] * len(members)
    for *key, indexes in result["member_groups"]:
        for index in indexes:
            values = dict(zip(common, key)) | dict(zip(kept, result["members"][index]))
            rebuilt[index] = [values[name] for name in columns]
    if rebuilt != members:
        raise ValueError("grouping repeated membership metadata changed a member")
    result["membership_encoding"] = "members and operands retain the same row order; member_groups supply only common run/scope/cell/dt values at the listed zero-based row indices. Every original member is reconstructible."
    return result


def report_examples(raw_export, task, logical_ids, full, aliases):
    """Read actual referenced episodes; references may be outside the 24 facts cases."""
    import export_confirmation_raw as exporter
    private = read(raw_export / "_private/sources.json")
    needed = {int(x.split(".")[0][1:]): [] for x in logical_ids}
    for identity in logical_ids:
        needed[int(identity.split(".")[0][1:])].append(identity)
    examples = []
    for run, identities in needed.items():
        source = next(s for s in private["sources"] if s["run"] == run and s["source_id"].startswith(task))
        path = Path(source["manifest_path"])
        if digest(path) != source["manifest_sha256"]:
            raise ValueError("report-verification raw source manifest drift")
        manifest = exporter.read_manifest(path)
        arrays, static, terms, missing, chunks = exporter.load_run(path, manifest, task, run if task == "stair" else None)
        for identity in identities:
            env = int(identity.split(".e")[1])
            if task == "stair":
                terminal = next(s for s in manifest["metadata"]["seeds"] if s["run_index"] == run)["outcomes"][env]
                raw, _ = exporter.episode_arrays(arrays, env, "episode_length", terminal["episode_steps"])
                # Every default report timeline references source event records.
                records = {r["episode_id"]: r for r in full["episodes"]}
                refs = [ref for sentence in full["semantic"]["sentences"] for ref in sentence.get("evidence_refs", [])
                        if ref.startswith(f"episodes/{identity}/events/")]
                codes = {records[identity]["events"][int(ref.rsplit("/", 1)[1])]["code"] for ref in refs}
                fields = ["base_pos_w", "base_quat_w"]
                if codes & {"unloaded_low_motion", "unloaded_rising", "unloaded_lowering", "slip_body", "slip_any"}:
                    fields += ["support_lin_vel_w", "support_contact_forces_w"]
                if codes & {"low_clearance_body", "low_clearance_any"}:
                    fields += ["support_pos_w", "support_contact_forces_w"]
                fields = list(dict.fromkeys(fields))
                definitions = read(raw_export / "definitions/stair.json")
            else:
                terminal = manifest["metadata"]["runtime_episode_records"][env]
                raw, _ = exporter.episode_arrays(arrays, env, "control_step", terminal["episode_steps"])
                fields = ["pose", "planar_motion"]
                definitions = exporter.push_definitions(manifest["metadata"], include_window_peak=True)
            steps = raw["episode_length" if task == "stair" else "control_step"]
            discrete = [k for k, v in raw.items() if v.dtype.kind in "biu" and k not in ("episode_length", "control_step")]
            examples.append({"original_logical_id": identity, "episode_id": next(a for a, original in aliases.items() if original == identity),
                "whole_episode_first_last": [int(steps[0]), int(steps[-1])],
                "definitions_material": "verification_pool.json#/definitions" + ("" if task == "stair" else f"/{source['source_id']}"),
                "original_outcome": terminal, "raw_fields": exporter.field_info(raw),
                "raw_columns": {k: decimal_columns(raw[k]) for k in fields if k in raw},
                "raw_discrete": {k: discrete_runs(raw[k], steps) for k in discrete},
                "static_values": {k: v[env].tolist() for k, v in static.items()},
                "source_manifest_sha256": source["manifest_sha256"], "source_chunks": chunks,
                "not_supplied_raw_fields": sorted(set(raw)-set(fields)-set(discrete)-{"episode_length", "control_step"}),
                "missing_fields": missing})
    return {"stage": "report_verification_only", "selection": "ALL episode identities referenced anywhere in the complete default report; no positivity or focus filter",
        "operand_rule": "Full-time root fields always; Stair source-event references additionally require their foot velocity/force or foot position/force operands; Push complete root trajectory. Full-time operands also verify total interval counts, extrema and whole-episode claims. Original numeric archives remain authoritative.",
        "decimal_encoding": "Same shortest round-tripping full-column CSV encoding as related facts; no time truncation",
        "examples": examples}


def review_episode_derivation(row, *, task, material_profile):
    """A readable lossless event table for related Stair reviews."""
    result = deepcopy(row)
    if task != "stair" or material_profile not in {"related_v1", "related_v2"}:
        return result
    events = row.get("events")
    columns = ["body", "code", "start", "confirmed", "end"]
    if not isinstance(events, list) or any(
            not isinstance(event, dict) or not set(columns) <= set(event)
            or set(event) - set(columns) - {"measured"} for event in events):
        raise ValueError("Stair review event table requires the complete known event fields")
    result["events"] = {
        "schema_version": "p3new.stair_review_events_table.v1",
        "columns": columns,
        "rows": [[event[name] for name in columns] for event in events],
        "measured_by_original_event_index": {str(index): deepcopy(event["measured"])
                                               for index, event in enumerate(events) if "measured" in event},
        "original_locator": "Row i is original /events/i (zero-based), in unchanged order. "
                            "Map columns to row values and add measured when index i appears in "
                            "measured_by_original_event_index. All event fields and null values are preserved; "
                            "prepend this material's source_pointer for the original full-file location.",
    }
    if expand_review_episode_derivation(result) != row:
        raise ValueError("Stair review event table changed the complete episode derivation")
    return result


def expand_review_episode_derivation(row):
    """Reconstruct the original full episode object for offline integrity checks."""
    result = deepcopy(row)
    table = result.get("events")
    if not isinstance(table, dict) or table.get("schema_version") != "p3new.stair_review_events_table.v1":
        return result
    columns = ["body", "code", "start", "confirmed", "end"]
    if (table.get("columns") != columns or not isinstance(table.get("rows"), list)
            or any(not isinstance(values, list) or len(values) != len(columns) for values in table["rows"])
            or not isinstance(table.get("measured_by_original_event_index"), dict)):
        raise ValueError("invalid Stair review event table")
    events = [dict(zip(columns, values)) for values in table["rows"]]
    for index, value in table["measured_by_original_event_index"].items():
        if not isinstance(index, str) or not index.isdecimal() or str(int(index)) != index or int(index) >= len(events):
            raise ValueError("invalid original event index")
        events[int(index)]["measured"] = value
    result["events"] = events
    return result


def pack_capacity_reports(raw_export, output, delivery_root, *, padding_words=32768, review_assignment=None):
    """Old development only, using actual report slices and explicitly fake capacity text."""
    receipt = read(output / "packing_receipt.json")
    if receipt["purpose"] != "development":
        raise ValueError("Synthetic report capacity fixtures are development-only")
    private = read(raw_export / "_private/sources.json")
    aliases = {task: {r["episode_id"]: r["original_id"] for r in private["anonymous_mapping"] if r["task"] == task} for task in ("stair", "push")}
    task_data = {}
    for task in ("stair", "push"):
        full_path, phase_path = delivery_root / task / "behavior_full.json", delivery_root / task / "public_dev_result.json"
        request_path = delivery_root / task / "phase_report/draft/request.http_body.json"
        full, phase = read(full_path), read(phase_path)
        if phase["task_evidence"]["behavior_full"]["sha256"] != digest(full_path):
            raise ValueError("phase/full source reference mismatch")
        source_shas = {s["manifest_sha256"] for s in private["sources"] if s["source_id"].startswith(task)}
        if source_shas != ({full["domain"]["archive_hash"]} if task == "stair" else {s["manifest_sha256"] for s in full["sources"]}):
            raise ValueError("full derivation and raw export come from different numeric archives")
        if runner.encoded(read(request_path)) != request_path.read_bytes():
            raise ValueError("historical request is not exact production JSON serialization")
        dest = output / "materials/report" / task
        dest.mkdir(parents=True)
        public = phase["task_evidence"]["behavior"]
        save(dest / "bounded_report.json", public)
        shutil.copyfile(request_path, dest / "request.http_body.json")
        logical_ids = sorted(set(re.findall(r"\br\d+\.e\d+\b", json.dumps(public["report"]))))
        save(dest / "verification_pool.json", report_pool_operands(raw_export, task, aliases[task]))
        save(dest / "verification_examples.json", report_examples(raw_export, task, logical_ids, full, aliases[task]))
        if task == "stair":
            semantic = full["semantic"]
            shared = {"domain": full["domain"], "semantics": full["semantics"], "contract": semantic["contract"]}
            rows = {r["episode_id"]: (i, r) for i, r in enumerate(full["episodes"])}
            report_pointer = "/public_feedback_summary/behavior/profile"
        else:
            shared = {k: full[k] for k in ("binding", "comparison_domain", "definitions", "process_contract", "source_bindings")}
            rows = {f"r{r['source']+1}.e{r['env_id']}": (i, r) for i, r in enumerate(full["transcripts"])}
            report_pointer = "/public_feedback_summary/public_dev/behavior"
        provenance = {"full_file": str(full_path.resolve()), "full_sha256": digest(full_path),
            "phase_sha256": digest(phase_path), "request_sha256": digest(request_path),
            "request_status": "offline production serialization from delivery_offline_r1, never actually sent",
            "raw_manifest_sha256": digest(raw_export / "manifest.json")}
        save(dest / "group_full.json", {"source": provenance, "shared": shared,
            "slice_scope": "Shared definitions only. Complete per-episode derivation is assessed in the 24 episode units. ALL clauses in the complete default report and author request are assessed here against whole-pool raw operands and referenced-episode trajectories, regardless of focus scope. Non-referenced aggregate derivation rows are not additional scored assertions. Alias bindings are in verification_pool members; no full claim or report clause is filtered by positivity."})
        located = json.loads(read(request_path)["messages"][1]["content"])
        for name in report_pointer.strip("/").split("/"):
            located = located[name]
        if located != public:
            raise ValueError("complete default report differs from its request view")
        save(dest / "bounded_report_locator.json", {
            "complete_report": "The entire default report is already supplied, verbatim, inside request.http_body.json. Read every clause there; this locator removes a duplicate copy only.",
            "request_material_id": "request.http_body.json", "request_sha256": digest(request_path),
            "decode_json_string_at": "/messages/1/content", "then_json_pointer": report_pointer,
            "decoded_object_canonical_sha256": hashlib.sha256(runner.encoded(public)).hexdigest(),
            "archive_file": "bounded_report.json", "archive_sha256": digest(dest / "bounded_report.json"),
            "source_phase_pointer": "/task_evidence/behavior", "source_phase_sha256": digest(phase_path),
            "scored_scope": "All default report clauses and their corresponding final-request locations, across all scopes; one text instance may have both locators."})
        task_data[task] = (rows, shared, provenance)
    prepared_reports = []
    for unit in receipt["packets"]:
        packet, task = unit["packet_id"], unit["task"]
        packet_dir = output / "packets" / packet
        facts_path = packet_dir / "facts.manifest.json"
        facts = read(facts_path)
        if unit["axis"] == "episode":
            rows, shared, provenance = task_data[task]
            original = aliases[task][unit["scope"]]
            index, row = rows[original]
            slice_path = packet_dir / "full_episode.json"
            save(slice_path, {"source": provenance, "source_pointer": f"/{'episodes' if task == 'stair' else 'transcripts'}/{index}",
                "episode_id": unit["scope"], "original_logical_id": original, "shared": shared,
                "complete_episode_derivation": review_episode_derivation(
                    row, task=task, material_profile=receipt.get("material_profile"))})
            materials = list(facts["materials"]) + [material(slice_path, slice_path.name, "full_derivation", "complete-episode-full")]
        else:
            materials = [m for m in facts["materials"] if m["role"] in ("review_protocol", "focus_questions")]
            for name, role in (("verification_pool.json", "raw_numeric"), ("verification_examples.json", "raw_numeric"),
                               ("group_full.json", "full_derivation"), ("bounded_report_locator.json", "automatic_report"),
                               ("request.http_body.json", "bounded_request")):
                path = output / "materials/report" / task / name
                m = material(path, f"../../materials/report/{task}/{name}", role, name)
                if role == "bounded_request":
                    m["label"] = "Complete offline production-serialized author request; not actually sent"
                materials.append(m)
        report_path = packet_dir / "report.capacity.manifest.json"
        save(report_path, {"schema_version": "p3new.confirmation_review_materials.v1", "packet_id": packet,
            "stage": "report", "purpose": "old_development_numeric_packing_offline_only", "max_output_tokens": 32768,
            **({"reference_contract": facts["reference_contract"]} if "reference_contract" in facts else {}),
            **({"review_assignment": facts["review_assignment"]} if "review_assignment" in facts else {}),
            "facts_manifest_sha256": digest(facts_path), "materials": materials,
            "capacity_only": "Synthetic own-facts filler below is not a judge answer or seal. No evaluation is performed."})
        if review_assignment is not None and runner.assignment_fields(review_assignment, packet) != {
                "review_assignment": facts.get("review_assignment")}:
            raise ValueError("Capacity review assignment differs from facts")
        for judge in runner.review_judges(facts):
            fixture = packet_dir / f"{judge}.synthetic_capacity_facts.json"
            save(fixture, {"synthetic_only": True, "judge_id": judge, "packet_id": packet, "stage": "facts", "complete": False,
                "findings": {"not_a_fact_or_score": "CAPACITY_ONLY", "padding": "x " * padding_words}})
            destination = output / "prepared_report_capacity" / packet / judge
            card = runner.prepare(report_path, judge, destination, capacity_facts_fixture=fixture)
            prepared_reports.append({"packet_id": packet, "task": task, "axis": unit["axis"], "judge": judge,
                "status": card["status"], "http_body_bytes": card["request_bytes"], "request_sha256": card["request_sha256"]})
    save(output / "report_capacity_receipt.json", {"status": "SYNTHETIC_OWN_FACTS_CAPACITY_ONLY_NOT_FROZEN", "actual_calls": 0,
        "source_packing_sha256": digest(output / "packing_receipt.json"), "padding_words": padding_words,
        "report_output_reservation": 32768, "packets": prepared_reports,
        "facts_unchanged": "All 36 facts packets prepared before any report was loaded; necessary-fact answers are absent, never manufactured.",
        "stage_difference": "Episode report retains full related curves. Focus report uses its own complete facts plus whole-task report-verification raw operands/examples, never a scope-based support-rate filter."})


def prepare_report_manifest(facts_manifest, report_materials, output):
    """Append explicit report/ABI/arithmetic inputs; sealing/release stays in runner.

    All original facts materials remain. Additional references must be absolute
    (or relative to output.parent); no automatic production report is treated as
    an independent task-definition source.
    """
    facts_manifest, output = Path(facts_manifest).resolve(), Path(output).resolve()
    facts = read(facts_manifest)
    if facts.get("stage") != "facts" or output.exists():
        raise ValueError("Requires a facts manifest and new report-manifest path")
    runner.assigned_reviewers(facts)
    materials = deepcopy(facts["materials"])
    for item in materials:
        path, _ = runner.referenced(facts_manifest.parent, item)
        item["file"] = str(path)
    for item in report_materials:
        if item["role"] not in runner.REPORT_ROLES | {"task_definition", "raw_numeric", "raw_media_metadata", "raw_image"}:
            raise ValueError("Unexpected additional report-stage role")
        runner.referenced(output.parent, item)
        materials.append(deepcopy(item))
    if len({m["id"] for m in materials}) != len(materials) or not any(m["role"] in runner.coverage.REPORT_TEXT_ROLES for m in materials):
        raise ValueError("Report materials need unique IDs and an actual evaluated report")
    manifest = {"schema_version": "p3new.confirmation_review_materials.v1", "packet_id": facts["packet_id"],
        "stage": "report", "purpose": facts["purpose"], "max_output_tokens": facts.get("max_output_tokens", 65536),
        **({"reference_contract": facts["reference_contract"]} if "reference_contract" in facts else {}),
        **({"review_assignment": facts["review_assignment"]} if "review_assignment" in facts else {}),
        "facts_manifest_sha256": digest(facts_manifest), "materials": materials}
    runner.save(output, manifest, exclusive=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--raw-export", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=HERE / "N5_FACT_REVIEW_PROTOCOL_ZH.md")
    parser.add_argument("--capture-card", type=Path, default=HERE / "N5_CAPTURE_DRAFT.json")
    parser.add_argument("--context-tokens", type=int, help="Optional externally verified capacity; not frozen or inferred here")
    parser.add_argument("--profile", choices=("original", "related_v2"), default="related_v2",
                        help="related_v2: full-time columns, explicit availability and raw-only arithmetic; original archives preserved")
    parser.add_argument("--max-output-tokens", type=int, choices=(32768, 65536), default=65536)
    parser.add_argument("--task-definition", type=Path, action="append", default=None,
                        help="Previously generated task contract; repeat for every distinct task/numeric manifest. Required for confirmation.")
    parser.add_argument("--push-resolved-cell-manifest", type=Path,
                        help="Original Push cell manifest bound to the supplied contracts; required when supplying Push contracts")
    parser.add_argument("--report-capacity-root", type=Path,
                         help="optional old delivery_offline_r1 root; append synthetic-only report capacity requests after all facts packets")
    parser.add_argument("--review-assignment", type=Path,
                        help="Frozen packet A/B identities, actual models and carriers; omitted only for the legacy pair")
    args = parser.parse_args()
    receipt = pack(args.raw_export, args.output, args.protocol, args.capture_card, args.context_tokens, args.profile, args.max_output_tokens,
                   task_definitions=args.task_definition, push_resolved_cell_manifest=args.push_resolved_cell_manifest,
                   review_assignment=args.review_assignment)
    if args.report_capacity_root:
        if args.profile != "related_v2" or args.max_output_tokens != 32768:
            parser.error("report capacity fixture requires related_v2 and --max-output-tokens 32768")
        pack_capacity_reports(args.raw_export.resolve(), args.output.resolve(), args.report_capacity_root.resolve(),
                              review_assignment=args.review_assignment)
    print(json.dumps({key: receipt[key] for key in ("status", "episode_count", "focus_count", "actual_calls")}))


if __name__ == "__main__":
    main()
