"""Source-bound, episode-first arithmetic for offline behavior cards.

This module owns no task outcome rules, cross-source pooling, or fitted labels.
Each descriptor carries its own observable denominator. Missing processing and
processed-but-unknown are different states.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np




def distribution(values: list[float]) -> dict[str, Any]:
    values = [float(value) for value in values]
    return {"minimum": min(values) if values else None,
            "q25": float(np.quantile(values, .25)) if values else None,
            "median": float(np.median(values)) if values else None,
            "q75": float(np.quantile(values, .75)) if values else None,
            "maximum": max(values) if values else None,
            "mean": float(np.mean(values)) if values else None,
            "quantile_method": "linear", "count": len(values)}


def aggregate_behavior(episodes: list[dict], *, motion_bands: dict | None = None) -> dict:
    """Whole-cohort and observed outcome/condition facets; no archive-ID ranking."""
    groups = {"all": episodes}
    for episode in episodes:
        outcome = episode["outcome"]["primary_reason"]
        groups.setdefault("outcome:"+outcome, []).append(episode)
        groups.setdefault("condition:"+episode["stratum"], []).append(episode)
        groups.setdefault("condition:"+episode["stratum"]+"/"+outcome, []).append(episode)
    result = {}
    for name, members in groups.items():
        windows = {}
        for window in ("all", "terminal"):
            keys = sorted({k for e in members for k in e["facts"][window]})
            facts = {}
            for key in keys:
                entries = [e["facts"][window][key] for e in members if key in e["facts"][window]]
                values = [v["value"] for v in entries if v["value"] is not None]
                facts[key] = {"label": entries[0]["label"], "unit": entries[0]["unit"], "distribution": distribution(values),
                    "positive_episodes": sum(v>0 for v in values),
                    "episodes": len(members), "valid_episodes": len(values),
                    **{k: sum(v[k] for v in entries) for k in ("valid", "unknown", "not_applicable")}}
                if "presence" in entries[0]:
                    facts[key]["presence"] = {"observed_positive": sum(v["presence"] is True for v in entries),
                        "complete_negative": sum(v["presence"] is False for v in entries),
                        "unknown": sum(v["presence"] is None for v in entries)}
            windows[window] = facts
        result[name] = {"episodes": len(members), "facts": windows,
                        "outcomes": dict(Counter(e["outcome"]["primary_reason"] for e in members)),
                        "members": [{k:e[k] for k in ("run_index", "env_index", "episode_id")} for e in members]}
        if motion_bands is not None:
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
            for key, parts in partitions.items():
                result[name][key] = {part:{"episodes":len(rows),
                    "forward_speed_mps": distribution([v for _,v in rows if v is not None]),
                    "end_time_s": distribution([e["facts"]["all"]["time"]["value"] for e,_ in rows]),
                    "conditions": dict(Counter(e["stratum"] for e,_ in rows))} for part,rows in parts.items()}
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
        if kind == "terminal_history_difference":
            keys = ("current_positive", "current_unknown", "history_positive", "history_unknown")
            selected = {**selected, "matching_episodes": len({(e["run_index"], e["env_index"])
                for e in subset if all(e[k] == selected[k] for k in keys)})}
        witnesses.append(selected)
    return {"groups": result, "support": episodes[0]["support"],
            "contact_threshold_n": episodes[0]["contact_threshold_n"], "motion_bands": motion_bands,
            "witnesses": witnesses}
