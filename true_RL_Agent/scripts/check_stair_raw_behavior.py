"""Independent NumPy checks of Stair author facts from any compatible raw manifest.

No translator imports. This is a bounded oracle for motion, posture, current
contact, low-progress activity, condition denominators and runtime joint limits.
Cycle qualification, nominal terrain geometry and optional witnesses remain
outside its numerical coverage. It does not assign physical failure causes.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

EPS = 8 * np.finfo(np.float32).eps
WINDOW = .5
NUMBER = r"(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)"
LABELS = {"success": "成功", "timeout": "超时", "time_out": "超时",
          "base_orientation_instability": "姿态失稳标签", "forbidden_contact": "禁用部位接触标签",
          "off_course": "偏离路线标签", "slip": "滑移标签", "insufficient_clearance": "间隙不足标签"}


def scalar(values, *, na=0, missing=0, operation="mean"):
    values = np.asarray(values, dtype=float).reshape(-1)
    finite = values[np.isfinite(values)]
    return {"value": float(getattr(np, operation)(finite)) if len(finite) else None,
            "valid": len(finite), "unknown": len(values)-len(finite)-na+missing, "not_applicable": na}


def rotation(q):
    q = np.asarray(q, dtype=float)
    norm = np.linalg.norm(q, axis=1)
    good = np.isfinite(q).all(axis=1) & (abs(norm-1) <= 1e-4)
    unit = np.full_like(q, np.nan)
    unit[good] = q[good] / norm[good, None]
    w, x, y, z = unit.T
    # Columns are the body's X, Y, Z axes expressed in world coordinates.
    return np.stack((np.column_stack((1-2*(y*y+z*z), 2*(x*y+w*z), 2*(x*z-w*y))),
                     np.column_stack((2*(x*y-w*z), 1-2*(x*x+z*z), 2*(y*z+w*x))),
                     np.column_stack((2*(x*z+w*y), 2*(y*z-w*x), 1-2*(x*x+y*y)))), axis=2)


def direction(axis):
    if not np.isfinite(axis).all():
        return scalar([np.nan])
    if np.linalg.norm(axis[:2]) <= EPS:
        return scalar([np.nan], na=1)
    return scalar([np.degrees(np.arctan2(axis[1], axis[0]))])


def circular(values):
    if not values:
        return {"circular_mean": None, "resultant": None}
    vector = np.exp(1j*np.radians(values)).mean()
    return {"circular_mean": float(np.degrees(np.angle(vector))) if abs(vector) > EPS else None,
            "resultant": float(abs(vector))}


def episode_facts(a, steps, dt, expected_end, origin, feet, threshold):
    if not len(steps) or dt <= 0 or not np.isfinite(dt) or steps[0] < 1 or np.any(np.diff(steps) <= 0):
        raise ValueError("episode requires finite positive clock and ordered observed steps")
    p = a["base_pos_w"].astype(float)
    rot = rotation(a["base_quat_w"])
    relative = np.einsum("tij,tfi->tfj", rot, a["support_pos_w"].astype(float)-p[:, None])
    def velocity(points):
        out = np.full(points.shape, np.nan)
        adjacent = np.diff(steps) == 1
        out[1:][adjacent] = np.diff(points, axis=0)[adjacent]/dt
        return out
    v, world_v, rel_v = velocity(p), velocity(a["support_pos_w"].astype(float)), velocity(relative)
    up = rot[:, 2, 2]
    down = np.flatnonzero(up < 0)
    presence = True if len(down) else False if (np.isfinite(up).all() and
        np.array_equal(steps, np.arange(1, expected_end+1))) else None
    width = math.floor(WINDOW/dt+1e-9)
    terminal = steps >= steps[-1]-width
    edges = terminal & (steps > steps[terminal][0])
    complete = bool(steps[-1] == expected_end and (steps[-1]-1)*dt+1e-9 >= WINDOW
                    and width > 0 and np.isfinite(v[edges, 0]).sum() == width)
    lookback = math.ceil(WINDOW/dt)
    low = np.full(len(steps), np.nan)
    if len(steps) > lookback:
        positions = np.lib.stride_tricks.sliding_window_view(p[:, 0], lookback+1)
        times = np.lib.stride_tricks.sliding_window_view(steps, lookback+1)
        known = np.isfinite(positions).all(axis=1) & (times[:, -1]-times[:, 0] == lookback)
        low[lookback:][known] = (positions[known].max(axis=1)-positions[known, 0] < .02)
    rel_speed = np.linalg.norm(rel_v, axis=2)
    activity = np.where((rel_speed >= .05).any(axis=1), 1.,
                        np.where(np.isfinite(rel_speed).all(axis=1), 0., np.nan))
    positive = low == 1
    facet = {"positive": int(positive.sum()), "unknown": int((~np.isfinite(low)).sum()-(steps <= lookback).sum()),
             "not_applicable": int((steps <= lookback).sum()), "counts": {}}
    for key, selected in (("up_negative", up < 0), ("up_nonnegative", up >= 0), ("posture_unknown", ~np.isfinite(up))):
        selected = positive & selected
        facet["counts"][key] = {"active": int((selected & (activity == 1)).sum()),
            "quiet": int((selected & (activity == 0)).sum()), "unknown": int((selected & ~np.isfinite(activity)).sum())}
    forces = a["support_contact_forces_w"].astype(float)
    contact = np.where(np.isfinite(forces).all(axis=2), np.linalg.norm(forces, axis=2) > threshold, np.nan)
    valid_contact = np.isfinite(contact).all(axis=1)
    codes = np.nan_to_num(contact).astype(int) @ (1 << np.arange(len(feet)))
    whole = {"net": scalar([p[-1, 0]-p[0, 0]]), "net_lateral": scalar([p[-1, 1]-p[0, 1]]),
             "time": scalar([steps[-1]*dt]), "heading_end": direction(rot[-1, :, 0]),
             "speed": scalar(np.linalg.norm(v[:, :2], axis=1), na=int(steps[0] == 1)),
             "backward": scalar(np.maximum(-v[:, 0]*dt, 0), na=int(steps[0] == 1), operation="sum"),
             "active": scalar(100*np.where(low == 0, 0, np.where(positive, activity, np.nan)), na=int((steps <= lookback).sum())),
             "first_down_time": scalar([steps[down[0]]*dt] if len(down) else [np.nan], na=int(presence is False))}
    end = {"forward_speed": scalar(v[edges, 0]), "lateral_speed": scalar(v[edges, 1]),
           "height": scalar(p[terminal, 2]-origin[2]), "heading_end": direction(rot[-1, :, 0]),
           "tilt_end": scalar([np.degrees(np.arccos(np.clip(up[-1], -1, 1)))]),
           "up_direction_end": direction(rot[-1, :, 2])}
    for code in range(1 << len(feet)):
        samples = np.where(valid_contact, 100*(codes == code), np.nan)
        whole["contact."+str(code)] = scalar(samples)
        end["contact."+str(code)] = (scalar(np.full(terminal.sum(), np.nan), na=int(terminal.sum()))
            if (steps[-1]-1)*dt+1e-9 < WINDOW else scalar(samples[terminal], missing=width+1-int(terminal.sum())))
    for i, foot in enumerate(feet):
        prefix = "robot_0.support."+foot+"."
        whole[prefix+"contact"] = scalar(100*contact[:, i])
        for key, values in (("world", world_v), ("relative", rel_v)):
            whole[prefix+key] = scalar(np.linalg.norm(values[:, i], axis=1), na=int(steps[0] == 1))
        for axis, key in enumerate("xyz"):
            whole[prefix+key] = scalar(relative[:, i, axis])
        end[prefix+"relative"] = scalar(rel_speed[edges, i])
    speed = end["forward_speed"]["value"] if complete else None
    joint = "observed_down" if presence is True else "unknown" if presence is None or speed is None else "no_down_forward" if speed > .2 else "remaining"
    motion = "unknown" if speed is None else "near_zero" if abs(speed) < .01 else "forward" if speed > .2 else "backward" if speed < -.2 else "remaining"
    return {"facts": {"all": whole, "terminal": end}, "presence": presence, "low_progress_facet": facet,
            "joint_process": joint, "terminal_motion": motion, "complete_terminal_speed": speed,
            "missing_steps": expected_end-len(steps),
            "first_down_step": int(steps[down[0]]) if len(down) else None}


def aggregate(rows):
    result = {"episodes": len(rows), "facts": {}, "missing_steps": sum(r["missing_steps"] for r in rows)}
    for window in ("all", "terminal"):
        result["facts"][window] = facts = {}
        for key in rows[0]["facts"][window]:
            records = [r["facts"][window][key] for r in rows]
            values = [r["value"] for r in records if r["value"] is not None]
            operation = "mean" if key.startswith("contact.") else "median"
            distribution = circular(values) if key in ("heading_end", "up_direction_end") else {operation: scalar(values, operation=operation)["value"]}
            facts[key] = {"episodes": len(rows), "valid_episodes": len(values), "distribution": distribution,
                          **{k: sum(r[k] for r in records) for k in ("valid", "unknown", "not_applicable")}}
    result["facts"]["all"]["inverted"] = {"episodes": len(rows), "presence": {
        "observed_positive": sum(r["presence"] is True for r in rows), "complete_negative": sum(r["presence"] is False for r in rows),
        "unknown": sum(r["presence"] is None for r in rows)}}
    facets = [r["low_progress_facet"] for r in rows]
    result["low_progress_facet"] = {k: sum(f[k] for f in facets) for k in ("positive", "unknown", "not_applicable")}
    result["low_progress_facet"]["counts"] = {k: {s: sum(f["counts"][k][s] for f in facets) for s in ("active", "quiet", "unknown")}
        for k in ("up_negative", "up_nonnegative", "posture_unknown")}
    for partition in ("joint_process", "terminal_motion"):
        result[partition] = {}
        for part in sorted({r[partition] for r in rows}):
            selected = [r for r in rows if r[partition] == part]
            speeds = [r["complete_terminal_speed"] for r in selected if r["complete_terminal_speed"] is not None]
            result[partition][part] = {"episodes": len(selected),
                "forward_speed_mps": {"median": scalar(speeds, operation="median")["value"], "count": len(speeds)},
                "end_time_s": {"median": scalar([r["facts"]["all"]["time"]["value"] for r in selected], operation="median")["value"]}}
    return result


def described_partition(group):
    down = group["joint_process"].get("observed_down", {}).get("episodes", 0)
    if down >= 2 and group["episodes"]-down >= 2:
        return "joint_process"
    motion = group["terminal_motion"]
    return "terminal_motion" if "near_zero" in motion and ("forward" in motion or "backward" in motion) else None


def read_raw(path):
    manifest = json.loads(path.read_bytes())
    md = manifest["metadata"]
    if (manifest["identity"]["splits"] != ["public_dev"] or md["coordinates"]["base_quat_w"] != "wxyz"
            or md["coordinates"]["positions"] != "world_m"):
        raise ValueError("only public-dev Stair world-position / wxyz archives are supported")
    feet = md["runtime_projection"]["resolved_roles"]["support_effector"]
    threshold = float(md["semantics"]["episode_outcomes"]["signals"]["slip"]["contact_force_threshold"])
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("contact threshold must be finite and nonnegative")
    limits = md.get("runtime_joint_position_limits")
    if limits and limits["joint_names"] != md["measurement_contract"]["joint_position"]["joint_names"]:
        raise ValueError("runtime joint column mapping differs from measurement contract")
    keys = ("base_pos_w", "base_quat_w", "support_pos_w", "support_contact_forces_w", "episode_length", "valid")
    keys += ("joint_pos",) if limits else ()
    rows, runtime = [], {}
    for seed in md["seeds"]:
        chunks, origin = [], None
        for chunk in sorted((c for c in manifest["chunks"] if c["metadata"]["run_index"] == seed["run_index"]), key=lambda c: c["metadata"]["step_start"]):
            source = path.parent/chunk["file"]
            if hashlib.sha256(source.read_bytes()).hexdigest() != chunk["sha256"]:
                raise ValueError("chunk checksum mismatch: "+chunk["file"])
            with np.load(source, allow_pickle=False) as z:
                chunks.append({k: z[k] for k in keys})
                if "terrain_env_origins_w" in z:
                    origin = z["terrain_env_origins_w"]
        arrays = {k: np.concatenate([c[k] for c in chunks]) for k in keys}
        for env, outcome in enumerate(seed["outcomes"]):
            selected = arrays["valid"][:, env].astype(bool)
            a = {k: v[selected, env] for k, v in arrays.items()}
            row = episode_facts(a, a["episode_length"], float(seed["step_dt"]), int(seed["episode_steps"][env]), origin[env], feet, threshold)
            row.update(outcome=outcome["primary_reason"], cell=outcome["evaluation_cell_id"], run=seed["run_index"], env=env)
            rows.append(row)
            if limits:
                raw = a["joint_pos"]
                q, finite = raw.astype(float), np.isfinite(raw)
                for group in ("all", row["outcome"]):
                    target = runtime.setdefault(group, {"episodes": 0, "facts": {}})
                    target["episodes"] += 1
                    for kind, source in (("hard", "joint_pos_limits"), ("soft", "soft_joint_pos_limits")):
                        bounds = np.asarray(limits["values"][source][env], dtype=float)
                        if not np.isfinite(bounds).all() or np.any(bounds[:, 0] >= bounds[:, 1]):
                            raise ValueError("runtime bounds must be finite ordered lower/upper pairs")
                        tolerance = abs(bounds.astype(raw.dtype).astype(float)-bounds)
                        lo, hi = bounds.T
                        records = target["facts"].setdefault(kind, {})
                        for j, name in enumerate(limits["joint_names"]):
                            r = records.setdefault(name, dict(valid=0, unknown=0, lower_outside=0, upper_outside=0, minimum_margin=None))
                            values = q[finite[:, j], j]
                            r["valid"] += len(values); r["unknown"] += len(q)-len(values)
                            r["lower_outside"] += int((values < lo[j]-tolerance[j, 0]).sum())
                            r["upper_outside"] += int((values > hi[j]+tolerance[j, 1]).sum())
                            if len(values):
                                margin = float(np.minimum(values-lo[j], hi[j]-values).min())
                                r["minimum_margin"] = margin if r["minimum_margin"] is None else min(margin, r["minimum_margin"])
    groups = {"all": aggregate(rows)}
    for reason in sorted({r["outcome"] for r in rows}):
        groups["outcome:"+reason] = aggregate([r for r in rows if r["outcome"] == reason])
    cells = {}
    for cell in sorted({r["cell"] for r in rows}):
        members = [r for r in rows if r["cell"] == cell]
        cells[cell] = {"episodes": len(members), "outcomes": {}}
        for reason in sorted({r["outcome"] for r in members}):
            subset = [r for r in members if r["outcome"] == reason]
            cells[cell]["outcomes"][reason] = {"episodes": len(subset), **{k: dict(Counter(r[k] for r in subset)) for k in ("joint_process", "terminal_motion")}}
    for group in runtime.values():
        for records in group["facts"].values():
            for r in records.values():
                r["outside_percent"] = {s: 100*r[s+"_outside"]/r["valid"] if r["valid"] else None for s in ("lower", "upper")}
    return {"manifest": str(path.resolve()), "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "scope": "bounded_independent_raw_checks; inspected_archives_are_development_data",
            "not_numerically_checked": ["cycle_qualification", "nominal_terrain_geometry", "optional_witness_selection"],
            "feet": feet, "contact_threshold_n": threshold, "groups": groups, "cells": cells, "runtime_joint": runtime,
            "examples": [{k: r[k] for k in ("run", "env", "outcome", "first_down_step", "missing_steps")} for r in rows if r["first_down_step"] is not None][:3]}


def compare_subset(expected, actual, location, errors):
    if isinstance(expected, dict):
        for key, value in expected.items():
            if not isinstance(actual, dict) or key not in actual:
                errors.append(location+"/"+key+": missing")
            else:
                compare_subset(value, actual[key], location+"/"+key, errors)
    elif isinstance(expected, float) and isinstance(actual, (int, float)):
        if not math.isfinite(actual) or not math.isclose(expected, actual, rel_tol=1e-9, abs_tol=1e-8):
            errors.append(f"{location}: expected {expected}, got {actual}")
    elif expected != actual:
        errors.append(f"{location}: expected {expected}, got {actual}")


def check_summary(raw, saved):
    errors = []
    summary = saved["summary"]
    compare_subset(raw["archive_sha256"], summary["archive_sha256"], "archive", errors)
    for name, group in raw["groups"].items():
        # Current projection keeps different quantities for the cohort and outcomes.
        all_keys = ({"net", "net_lateral", "heading_end", "speed", "backward", "active", "inverted"} |
                    {k for k in group["facts"]["all"] if "." in k}) if name == "all" else {"time", "net", "net_lateral", "inverted", "first_down_time"}
        end_keys = {k for k in group["facts"]["terminal"] if k.endswith(".relative")} if name == "all" else set(group["facts"]["terminal"])-{k for k in group["facts"]["terminal"] if k.endswith(".relative")}
        expected = {"episodes": group["episodes"], "facts": {w: {k: group["facts"][w][k] for k in keys} for w, keys in (("all", all_keys), ("terminal", end_keys))}}
        if name != "outcome:success":
            expected["low_progress_facet"] = {k: v for k, v in group["low_progress_facet"].items()
                if k in (("unknown", "not_applicable") if name == "all" else ("unknown", "positive", "counts"))}
        if name != "all":
            chosen = described_partition(group)
            for partition in ("joint_process", "terminal_motion"):
                expected[partition] = {k: {field: value for field, value in v.items() if field == "episodes"
                    or partition == chosen and (field == "forward_speed_mps" or partition == "joint_process")}
                    for k, v in group[partition].items()}
        compare_subset(expected, summary["behavior"]["groups"].get(name), name, errors)
    compare_subset(raw["runtime_joint"], (summary.get("runtime_joint") or {}).get("groups", {}), "runtime_joint", errors)
    for cell, data in raw["cells"].items():
        compare_subset({"episodes": data["episodes"]}, summary["behavior"]["groups"].get("condition:"+cell), cell, errors)
        for reason, counts in data["outcomes"].items():
            if reason != "success":
                expected = {"episodes": counts["episodes"], **{p: {k: {"episodes": v} for k, v in counts[p].items()} for p in ("joint_process", "terminal_motion")}}
                compare_subset(expected, summary["behavior"]["groups"].get("condition:"+cell+"/"+reason), cell+"/"+reason, errors)
    return errors


def check_author(raw, payload, *, secondary_source=None):
    behavior = payload["public_feedback_summary"]["behavior"]
    report = behavior["profile"]["report"]
    rows = {r["id"]: r["text"] for section in report.values() if isinstance(section, list) for r in section if isinstance(r, dict) and "id" in r}
    errors = []
    secondary_text = None
    if secondary_source is not None:
        generation, lane = secondary_source
        matches = [r for r in behavior.get("secondary", []) if (r.get("generation_index"), r.get("lane_id")) == (generation, lane)]
        if len(matches) != 1 or not isinstance(matches[0].get("process"), str):
            return [f"secondary {generation}/{lane}: missing unique absolute failure process"]
        secondary_text = matches[0]["process"]
    def required(key):
        if "claim:"+key not in rows:
            errors.append("claim:"+key+": missing required author fact")
        return rows.get("claim:"+key, "")
    def contains(text, value, where):
        if value not in text:
            errors.append(where+": missing "+value)
    def number(text, anchor, value, unit, where, tolerance=.00051):
        match = re.search(re.escape(anchor)+NUMBER+re.escape(unit), text)
        if value is None:
            if match:
                errors.append(where+": numeric claim for unknown/not-applicable "+anchor)
            contains(text, anchor+"未知"+unit, where)
            return
        if not match or abs(float(match[1])-value) > tolerance:
            errors.append(f"{where}: {anchor} expected {value}{unit}")
    def fact(text, anchor, f, unit, where, statistic="median", denominator=True):
        number(text, anchor, f["distribution"][statistic], unit, where, .051 if unit in ("°", "%") else .00051)
        if denominator and (f["valid_episodes"] != f["episodes"] or f["unknown"]):
            contains(text, f"（有效{f['valid_episodes']}/{f['episodes']}回合，缺失{f['unknown']}读数"+
                     (f"、不适用{f['not_applicable']}读数" if f["not_applicable"] else "")+"）", where)
    def posture(text, facts, where):
        p = facts["inverted"]["presence"]
        contains(text, f"上轴向下{p['observed_positive']}/{facts['inverted']['episodes']}回合", where)
        if p["unknown"]:
            contains(text, f"完整记录未见{p['complete_negative']}回合", where)
            contains(text, f"{p['unknown']}回合因缺测不能判全程阴性", where)
    def heading(text, anchor, f, where):
        if f["distribution"]["circular_mean"] is None:
            contains(text, anchor+("无均向" if f["valid_episodes"] else "不适用" if f["not_applicable"] and not f["unknown"] else "未知"), where)
        else:
            number(text, anchor, f["distribution"]["circular_mean"], "°", where, .051)
        pattern = re.escape(anchor)+r"(?:-?[\d.]+°|无均向|不适用|未知)\(R="
        match = re.search(pattern+NUMBER+r"\)", text)
        value = f["distribution"]["resultant"]
        if value is not None and (not match or abs(float(match[1])-value) > .00051):
            errors.append(where+": missing/wrong circular resultant "+anchor)
        if f["valid_episodes"] != f["episodes"]:
            contains(text, f"（有效{f['valid_episodes']}/{f['episodes']}，不适用{f['not_applicable']}、未知{f['unknown']}）", where)
    def contacts(text, facts, count, where):
        ordered = sorted((k for k in facts if k.startswith("contact.") and (facts[k]["distribution"]["mean"] or 0) > 0), key=lambda k: (-facts[k]["distribution"]["mean"], k))[:count]
        for key in ordered:
            code = int(key.split(".")[1])
            label = "+".join(f.removesuffix("_foot") for i, f in enumerate(raw["feet"]) if code & (1 << i)) or "全部未触发"
            fact(text, label, facts[key], "%", where, "mean")
    all_group = raw["groups"]["all"]
    if secondary_text is None:
        scope = required("scope")
        contains(scope, f"本队列{all_group['episodes']}回合", "scope")
        overall = required("motion")
        for key, anchor, unit in (("net", "净正向位移", "m"), ("net_lateral", "净横向位移", "m"), ("speed", "根部位置差分水平速度", "m/s"), ("backward", "反向位移累计", "m"), ("active", "低进度且足相对身体变化占比", "%")):
            fact(overall, anchor, all_group["facts"]["all"][key], unit, "motion")
        heading(overall, "身体前轴投影朝向", all_group["facts"]["all"]["heading_end"], "motion")
        posture(overall, all_group["facts"]["all"], "motion")
    for name, group in raw["groups"].items():
        if name == "all" or secondary_text is not None and name == "outcome:success":
            continue
        reason = name.split(":", 1)[1]; label = LABELS.get(reason, reason)
        if secondary_text is None:
            contains(scope, label+str(group["episodes"]), "scope")
            text = required("outcome."+reason)
        else:
            anchor = label+("组" if group["episodes"] > 1 else "")+str(group["episodes"])+"回合"
            text = anchor+secondary_text.split(anchor, 1)[-1] if anchor in secondary_text else ""
            for other in raw["groups"]:
                if other not in ("all", name):
                    other_reason = other.split(":", 1)[1]
                    other_label = LABELS.get(other_reason, other_reason)
                    text = text.split(other_label+"组", 1)[0].split(other_label+"1回合", 1)[0]
        contains(text, label+("组" if group["episodes"] > 1 else "")+str(group["episodes"])+"回合", name)
        if group["episodes"] == 1:
            contains(text, "单例，不作群体推广", name)
        for window, key, anchor, unit in (("all", "time", "结束", "s"), ("all", "net", "净前移", "m"), ("all", "net_lateral", "横移", "m"), ("terminal", "forward_speed", "末窗前速", "m/s"), ("terminal", "lateral_speed", "横速", "m/s"), ("terminal", "height", "根部z", "m"), ("terminal", "tilt_end", "倾角", "°")):
            if secondary_text is None or key not in {"lateral_speed", "height"}:
                fact(text, anchor, group["facts"][window][key], unit, name)
        posture(text, group["facts"]["all"], name)
        if secondary_text is None:
            heading(text, "前轴朝向", group["facts"]["terminal"]["heading_end"], name)
            heading(text, "方位", group["facts"]["terminal"]["up_direction_end"], name)
            first = group["facts"]["all"]["first_down_time"]
            if first["valid_episodes"] >= 2:
                fact(text, "首次观察向下时刻中位", first, "s", name, denominator=False)
                contains(text, f"（{first['valid_episodes']}阳性回合）", name)
            contacts(text, group["facts"]["terminal"], 2, name)
            low = group["low_progress_facet"]
            if reason != "success" and low["positive"]:
                contains(text, f"低进度{low['positive']}点", name)
                for key, label in (("up_negative", "上轴向下"), ("up_nonnegative", "上轴z≥0"), ("posture_unknown", "姿态未知")):
                    c = low["counts"][key]
                    if sum(c.values()):
                        contains(text, label+f"活动/低活动{c['active']}/{c['quiet']}点", name)
        partition = described_partition(group)
        labels = ({"observed_down": "已见上轴向下", "no_down_forward": "完整未见向下且末前速>0.2m/s",
                   "remaining": "其余已知", "unknown": "姿态或末窗未定"} if partition == "joint_process" else
                  {"near_zero": "|前向速度|<0.01m/s的", "forward": "前向速度>0.2m/s的",
                   "backward": "前向速度<-0.2m/s的", "remaining": "其余已知的", "unknown": "末窗未定的"})
        if partition:
            for part, value in group[partition].items():
                anchor = labels[part]+str(value["episodes"])+"回合（"
                contains(text, anchor, name)
                clause = text.split(anchor, 1)[-1].split("）", 1)[0]
                number(clause, "末速" if partition == "joint_process" else "速度中位", value["forward_speed_mps"]["median"], "m/s", name)
                if value["forward_speed_mps"]["count"] != value["episodes"]:
                    contains(clause, f"末速有效{value['forward_speed_mps']['count']}/{value['episodes']}回合", name)
                if partition == "joint_process" and secondary_text is None:
                    number(clause, "结束", value["end_time_s"]["median"], "s", name)
        if reason != "success" and secondary_text is None:
            candidates = [(cell, data, data["outcomes"][reason]) for cell, data in raw["cells"].items()
                if data["outcomes"].get(reason, {}).get("episodes", 0) >= 2]
            excess = lambda row: row[2]["episodes"]-row[1]["episodes"]*group["episodes"]/all_group["episodes"]
            chosen = {c: [] for c, data, counts in sorted(candidates, key=lambda r: (-excess(r), -r[2]["episodes"], r[0]))[:2] if excess((c, data, counts)) > 1e-9}
            outcome_cells = set(chosen)
            if partition:
                for part, counts in group[partition].items():
                    represented = [r for r in candidates if r[2][partition].get(part, 0) >= 2]
                    if counts["episodes"] >= 2 and represented:
                        cell, _, _ = min(represented, key=lambda r: (-r[2][partition][part], r[0]))
                        chosen.setdefault(cell, []).append(part)
            if chosen:
                if not partition and "claim:conditions."+reason not in rows:
                    continue  # Outcome-only cell detail is optional under the current budget.
                condition_text = required("conditions."+reason)
                cell_labels = {"observed_down": "向下", "no_down_forward": "完整未见向下且末前进", "remaining": "其余", "unknown": "未知",
                               "near_zero": "末前速近零", "forward": "末前进", "backward": "末倒退"}
                for cell, parts in chosen.items():
                    display = cell.removeprefix("public_dev.")
                    segment = condition_text.split(display+" ", 1)[-1].split("；", 1)[0]
                    contains(condition_text, display+" ", reason)
                    data = raw["cells"][cell]; counts = data["outcomes"][reason]
                    if cell in outcome_cells and sum(counts[partition][p] for p in parts) != counts["episodes"]:
                        contains(segment, f"总计{counts['episodes']}/{data['episodes']}"+("（其中" if parts else ""), cell)
                    for part in parts:
                        contains(segment, cell_labels[part]+f"{counts[partition][part]}/{data['episodes']}", cell)
    if secondary_text is not None:
        if not any(k not in ("all", "outcome:success") for k in raw["groups"]):
            contains(secondary_text, "本队列无失败标签回合", "secondary")
        return errors
    for i, foot in enumerate(raw["feet"]):
        text = required("foot."+str(i)); prefix = "robot_0.support."+foot+"."
        contains(text, foot.removesuffix("_foot")+"：", "foot."+str(i))
        for key, label, unit in (("contact", "接触占比", "%"), ("world", "世界点速", "m/s"), ("relative", "相对点速", "m/s")):
            fact(text, label, all_group["facts"]["all"][prefix+key], unit, foot)
        fact(text, "末窗相对点速", all_group["facts"]["terminal"][prefix+"relative"], "m/s", foot, denominator=False)
        column = "(?:"+NUMBER+"|未知)"
        match = re.search(r"相对\(x,y,z\)均值="+column+"/"+column+"/"+column+"m", text)
        if not match:
            errors.append(foot+": missing/wrong body-relative XYZ columns")
        else:
            for j, axis in enumerate("xyz"):
                expected = all_group["facts"]["all"][prefix+axis]["distribution"]["median"]
                actual = None if match[j+1] is None else float(match[j+1])
                if (actual is not None if expected is None else actual is None or abs(actual-expected) > .00051):
                    errors.append(foot+": wrong body-relative "+axis+" column")
    if any((v["distribution"]["mean"] or 0) > 0 for k, v in all_group["facts"]["all"].items() if k.startswith("contact.")):
        contacts(required("contacts"), all_group["facts"]["all"], 3, "contacts")
    if raw["runtime_joint"]:
        records = raw["runtime_joint"]["all"]["facts"]
        outside = lambda v: v["lower_outside"]+v["upper_outside"]
        selected = {min(r, key=lambda k: (-outside(r[k]), k)) for r in records.values() if any(outside(v) for v in r.values())}
        for joint in sorted(selected):
            text = required("runtime_joint."+joint)
            contains(text, joint.removesuffix("_joint")+f"界外下/上：有效{records['hard'][joint]['valid']}点", joint)
            def percentages(clause, facts, where):
                for kind in ("hard", "soft"):
                    expected = facts[kind][joint]["outside_percent"]
                    column = "(?:"+NUMBER+"|未知)"
                    match = re.search(kind+" "+column+"/"+column+"%", clause)
                    # Display precision only: tiny percentages use one significant digit.
                    tolerance = lambda v: .50001*10**math.floor(math.log10(v)) if 0 < v < .05 else .051
                    if not match:
                        errors.append(where+": wrong/missing "+kind+" lower/upper percentages")
                        continue
                    for i, side in enumerate(("lower", "upper")):
                        value = expected[side]
                        actual = None if match[i+1] is None else float(match[i+1])
                        if (actual is not None if value is None else actual is None or abs(actual-value) > tolerance(value)):
                            errors.append(where+": wrong/missing "+kind+" lower/upper percentages")
                            break
            percentages(text, records, joint)
            matches = re.search(r"最小余量\(hard/soft\)"+NUMBER+"/"+NUMBER+"rad", text)
            if not matches or any(not math.isclose(float(matches[i+1]), records[k][joint]["minimum_margin"], rel_tol=.0006, abs_tol=1e-8) for i, k in enumerate(("hard", "soft"))):
                errors.append(joint+": wrong/missing signed minimum margins")
            unknown = records["hard"][joint]["unknown"]
            missing = re.match(r"。缺测(\d+)点。", text[matches.end():]) if matches else None
            if (unknown and missing is None) or (missing is not None and int(missing[1]) != unknown):
                errors.append(joint+": wrong/missing unknown q-point count")
            contrasts = ["success"] if raw["runtime_joint"].get("success", {}).get("episodes", 0) >= 2 else []
            def difference(key):
                f = raw["runtime_joint"][key]["facts"]
                return max(abs(outside(f[k][joint])-outside(records[k][joint])*f[k][joint]["valid"]/max(1, records[k][joint]["valid"])) for k in ("hard", "soft"))
            candidates = [k for k, v in raw["runtime_joint"].items() if k not in ("all", "success") and v["episodes"] >= 2]
            contrasts += sorted(candidates, key=lambda k: (-difference(k), k))[:2-len(contrasts)]
            for reason in contrasts:
                g = raw["runtime_joint"][reason]
                anchor = LABELS.get(reason, reason)+f"{g['episodes']}回合/{g['facts']['hard'][joint]['valid']}点："
                contains(text, anchor, joint)
                percentages(text.split(anchor, 1)[-1].split("；", 1)[0], g["facts"], joint+"/"+reason)
        if not selected:
            text = required("runtime_joint.none")
            contains(text, "启动hard/soft上下界外均0点", "runtime_joint.none")
            for kind, facts in records.items():
                contains(text, f"{kind}有效{sum(v['valid'] for v in facts.values())}、未知{sum(v['unknown'] for v in facts.values())}关节点", "runtime_joint.none")
    definitions = required("definitions")
    number(definitions, "力模>", raw["contact_threshold_n"], "N", "definitions")
    gaps = required("gaps")
    low = all_group["low_progress_facet"]
    contains(gaps, f"初始不适用{low['not_applicable']}、内部未知{low['unknown']}点", "gaps")
    if "outcome:success" not in raw["groups"]:
        contains("\n".join(rows.values()), "本队列无成功回合", "success contrast")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--summary", type=Path, help="Saved {domain, summary}; checks quantities and denominators")
    parser.add_argument("--user-payload", type=Path, help="Saved actual serialized author user payload")
    parser.add_argument("--secondary-source", nargs=2, metavar=("GENERATION", "LANE"),
                        help="Check this recent lane against its own raw manifest")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = read_raw(args.manifest)  # Establish expectations before opening generated text.
    checks = {}
    if args.summary:
        checks["summary"] = check_summary(raw, json.loads(args.summary.read_bytes()))
    if args.user_payload:
        checks["author"] = check_author(raw, json.loads(args.user_payload.read_bytes()),
            secondary_source=(int(args.secondary_source[0]), args.secondary_source[1]) if args.secondary_source else None)
    raw["checks"] = checks
    raw["status"] = "FAIL" if any(checks.values()) else "PASS_WITH_STATED_COVERAGE" if checks else "RAW_EXPECTATIONS_ONLY"
    args.output.write_text(json.dumps(raw, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"status": raw["status"], "episodes": raw["groups"]["all"]["episodes"], "errors": checks}, ensure_ascii=False))
    raise SystemExit(int(any(checks.values())))


if __name__ == "__main__":
    main()
