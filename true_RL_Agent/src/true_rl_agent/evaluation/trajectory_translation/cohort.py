"""Episode-first behavior summaries and the shared Push distribution helper."""
from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
from .measurement import DIRECTION_EPS


def distribution(values: list[float]) -> dict[str, Any]:
    values = [float(value) for value in values]
    return {"minimum": min(values) if values else None,
            "q25": float(np.quantile(values, .25)) if values else None,
            "median": float(np.median(values)) if values else None,
            "q75": float(np.quantile(values, .75)) if values else None,
            "maximum": max(values) if values else None,
            "mean": float(np.mean(values)) if values else None,
            "quantile_method": "linear", "count": len(values)}


def process_partition(group):
    """The process split shared by aggregation and its prose consumer."""
    parts = group.get("joint_process", {})
    if parts.get("observed_down", {}).get("episodes", 0) >= 2 and sum(v["episodes"] for k,v in parts.items() if k != "observed_down") >= 2:
        return "joint_process"
    motion = group.get("terminal_motion", {})
    if "near_zero" in motion and ("forward" in motion or "backward" in motion):
        return "terminal_motion"
    return None


def circular_distribution(values):
    if not values:
        return {"circular_mean":None, "resultant":None}
    vector = np.mean(np.exp(1j*np.radians(values)))
    return {"circular_mean":float(np.degrees(np.angle(vector))) if abs(vector) > DIRECTION_EPS else None,
            "resultant":float(abs(vector))}


def _aggregate_facts(members, name, window):
    keys = {k for e in members for k in e["facts"][window]}
    if name == "all":
        keys = ({"net","net_lateral","heading_end","speed","backward","active","inverted","duration"} | {k for k in keys if "." in k}
                if window == "all" else {k for k in keys if k.endswith((".relative",".terrain"))})
    else:
        keys = ({"time","net","net_lateral","inverted","first_down_time","tilt","rotation","forward_elevation_min","forward_elevation"}
                if window == "all" else {"progress","forward_speed","lateral_speed","height","tilt_end","heading_end","up_direction_end","low_progress"} |
                {k for k in keys if k.endswith(".terrain") or k.startswith("contact.")})
    facts = {}
    for key in sorted(keys):
        entries = [e["facts"][window][key] for e in members if key in e["facts"][window]]
        fact = {"label": entries[0]["label"], "unit": entries[0]["unit"]}
        if key == "duration":
            fact["unknown"] = sum(v["unknown"] for v in entries)
        elif key == "inverted":
            fact.update(episodes=len(members), presence={
                "observed_positive": sum(v["presence"] is True for v in entries),
                "complete_negative": sum(v["presence"] is False for v in entries),
                "unknown": sum(v["presence"] is None for v in entries)})
        else:
            values = [v["value"] for v in entries if v["value"] is not None]
            statistic = "mean" if key.startswith("contact.") else "median"
            value = float(np.mean(values) if statistic == "mean" else np.median(values)) if values else None
            stats = circular_distribution(values) if key in {"heading_end","up_direction_end"} else {statistic:value}
            fact.update(distribution=stats, episodes=len(members), valid_episodes=len(values),
                        **{k:sum(v[k] for v in entries) for k in ("valid","unknown","not_applicable")})
        facts[key] = fact
    return facts


def _aggregate_process(members, motion_bands, *, describe):
    partitions = {"joint_process": {}, "terminal_motion": {}}
    for e in members:
        speed = e["facts"]["terminal"]["forward_speed"]["value"] if e["terminal_motion_complete"] else None
        presence = e["facts"]["all"]["inverted"]["presence"]
        joint = ("observed_down" if presence is True else "unknown" if presence is None or speed is None
                 else "no_down_forward" if speed > motion_bands["forward_mps"] else "remaining")
        motion = ("unknown" if speed is None else "near_zero" if abs(speed) < motion_bands["near_zero_mps"]
                  else "forward" if speed > motion_bands["forward_mps"]
                  else "backward" if speed < -motion_bands["forward_mps"] else "remaining")
        for key, part in (("joint_process", joint), ("terminal_motion", motion)):
            partitions[key].setdefault(part, []).append((e, speed))
    result = {key:{part:{"episodes":len(rows)} for part,rows in parts.items()} for key,parts in partitions.items()}
    selected = process_partition(result) if describe else None
    if selected:
        for part, rows in partitions[selected].items():
            speeds = [v for _,v in rows if v is not None]
            result[selected][part]["forward_speed_mps"] = {
                "median":float(np.median(speeds)) if speeds else None, "count":len(speeds)}
            if selected == "joint_process":
                result[selected][part]["end_time_s"] = {
                    "median":float(np.median([e["facts"]["all"]["time"]["value"] for e,_ in rows]))}
    return result


def aggregate_behavior(episodes: list[dict], *, motion_bands: dict | None = None) -> dict:
    """Aggregate only facts used by Stair prose, selection and representative witnesses."""
    groups = {"all": episodes}
    for episode in episodes:
        outcome = episode["outcome"]["primary_reason"]
        groups.setdefault("outcome:"+outcome, []).append(episode)
        groups.setdefault("condition:"+episode["cell"], []).append(episode)
        if outcome != "success":
            groups.setdefault("condition:"+episode["cell"]+"/"+outcome, []).append(episode)
    result = {}
    for name, members in groups.items():
        outcome = name.startswith("outcome:")
        group = {"episodes":len(members)}
        if name == "all" or outcome:
            group["facts"] = {window:_aggregate_facts(members,name,window) for window in ("all","terminal")}
        if outcome:
            observed = [e for e in members if e["first_down_observation"] is not None]
            if len(observed) >= 2:
                middle = float(np.median([e["first_down_observation"]["time_s"] for e in observed]))
                e = min(observed,key=lambda e:(abs(e["first_down_observation"]["time_s"]-middle),e["run_index"],e["env_index"]))
                group["first_down_observation"] = {**e["first_down_observation"], "run_index":e["run_index"], "env_index":e["env_index"]}
        if name == "all":
            group["outcomes"] = dict(Counter(e["outcome"]["primary_reason"] for e in members))
        if name == "all" or outcome and name != "outcome:success":
            low = [e["low_progress_facet"] for e in members]
            facet = {"label":low[0]["label"], "unit":low[0]["unit"], "unknown":sum(e["unknown"] for e in low)}
            if name == "all":
                facet["not_applicable"] = sum(e["not_applicable"] for e in low)
            else:
                facet.update(episodes=sum(e["positive"] > 0 for e in low), positive=sum(e["positive"] for e in low),
                    counts={p:{a:sum(e["counts"][p][a] for e in low) for a in low[0]["counts"][p]}
                            for p in low[0]["counts"]})
            group["low_progress_facet"] = facet
        if motion_bands is not None and (outcome or name.startswith("condition:") and "/" in name):
            group.update(_aggregate_process(members,motion_bands,describe=outcome))
        result[name] = group
    candidates = [{**event, "run_index": e["run_index"], "env_index": e["env_index"],
                   "outcome": e["outcome"]["primary_reason"]}
                  for e in episodes for event in e["events"]]
    witnesses = []
    for kind in sorted({e["kind"] for e in candidates}):
        subset = [e for e in candidates if e["kind"] == kind]
        def rank(e):
            if kind == "success_drawdown_return":
                return (-e["max_drawdown_m"], e["run_index"], e["env_index"])
            if kind == "terminal_history_difference":
                return (e["current_positive"] > 0, e["current_unknown"]+e["history_unknown"],
                        e["history_positive"], e["run_index"], e["env_index"], e["entity"])
            return (-e["span_s"], e["run_index"], e["env_index"], e["entity"])
        selected = min(subset, key=rank)
        selected = {**selected, "qualifying_episodes": len({(e["run_index"], e["env_index"]) for e in subset})}
        if kind == "terminal_history_difference":
            keys = ("current_positive", "current_unknown", "history_positive", "history_unknown")
            selected = {**selected, "matching_episodes": len({(e["run_index"], e["env_index"])
                for e in subset if all(e[k] == selected[k] for k in keys)})}
        witnesses.append(selected)
    return {"groups": result, "support": episodes[0]["support"],
            "contact_threshold_n": episodes[0]["contact_threshold_n"], "motion_bands": motion_bands,
            "witnesses": witnesses}
