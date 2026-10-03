"""Source-bound, episode-first arithmetic for offline behavior cards.

This module owns no task outcome rules, cross-source pooling, or fitted labels.
Each descriptor carries its own observable denominator. Missing processing and
processed-but-unknown are different states.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from .contracts import NumericEpisode, PredicateTrack, stable_id
from .measurement import NUMERIC_CONFIG, low_progress_track
from .temporal import interval_rows

COHORT_VERSION = "trajectory.cohort.v1"
ACTIVITY_VERSION = "trajectory.low_progress_activity.v1"
POOLED_ZERO_TOLERANCE = 1e-12  # operand unit; force pools are N*s, occupancy pools s


def distribution(values: list[float]) -> dict[str, Any]:
    values = [float(value) for value in values]
    return {"minimum": min(values) if values else None,
            "q25": float(np.quantile(values, .25)) if values else None,
            "median": float(np.median(values)) if values else None,
            "q75": float(np.quantile(values, .75)) if values else None,
            "maximum": max(values) if values else None,
            "mean": float(np.mean(values)) if values else None,
            "quantile_method": "linear", "count": len(values)}


def run_summary(intervals: list[dict], state: str) -> dict[str, Any]:
    runs = [r for r in intervals if r["state"] == state]
    complete = [r for r in runs if r["complete_run"]]
    return {"count": len(runs), "complete_count": len(complete),
            "left_censored_count": sum(r["left_censored"] for r in runs),
            "right_censored_count": sum(r["right_censored"] for r in runs),
            "censored_count": sum(not r["complete_run"] for r in runs),
            "complete_observed_spans_s": [r["observed_span_s"] for r in complete],
            "longest": max(runs, key=lambda r: r["sample_count"], default=None),
            "shortest_samples": min((r["sample_count"] for r in runs), default=None)}


def observed_presence(true_count: int, unknown_count: int) -> bool | None:
    """A witness proves observed presence; an unknown region cannot prove absence."""
    return True if true_count else (None if unknown_count else False)


def compose_activity(episode: NumericEpisode) -> dict[str, Any]:
    """Compose frozen M01 and M04 thresholds, without inferring steps or intent.

    any-moving can be witnessed even if another foot is unknown; all-low needs
    every foot. Initial incomplete progress windows remain unknown. The whole
    source interval is referenced, including every progress-window lookback.
    """
    n = len(episode.steps)
    support = episode.entities["support"]
    p = np.asarray(episode.arrays.get("base_pos_w", np.full((n, 3), np.nan)), dtype=float)
    progress = (p - episode.task["origin_world"]) @ np.asarray(episode.task["progress_direction_world"])
    progress[~np.isfinite(p).all(axis=1)] = np.nan
    low = low_progress_track(episode, progress)
    velocity = episode.arrays.get("support_lin_vel_w")
    xy = np.full((n, len(support)), np.nan) if velocity is None else np.asarray(velocity, dtype=float)[:, :, :2]
    speed = xy if velocity is None else np.linalg.norm(xy, axis=2)
    finite = np.isfinite(speed)
    moving = np.any(finite & (speed >= NUMERIC_CONFIG["low_horizontal_speed_mps"]), axis=1)
    all_known = finite.all(axis=1)
    active = np.where(moving, 1, np.where(all_known, 0, -1))
    quiet = np.where(moving, 0, np.where(all_known, 1, -1))
    values = {}
    for key, component in (("low_progress_any_foot_moving", active), ("low_progress_all_feet_low_speed", quiet)):
        combined = np.where((low.values == 0) | (component == 0), 0,
                            np.where((low.values == 1) & (component == 1), 1, -1))
        track = PredicateTrack(episode.episode_id, key, support, ("base_pos_w", "support_lin_vel_w"),
                               combined, {"lookback_samples": low.config["lookback_samples"]})
        intervals = interval_rows(episode, track)
        longest = max((r for r in intervals if r["state"] == "true"), key=lambda r: r["sample_count"], default=None)
        diagnostics = None
        if longest is not None:
            a, b = longest["index_half_open"]
            known = finite[a:b]
            high = known & (speed[a:b] >= NUMERIC_CONFIG["low_horizontal_speed_mps"])
            low_known = known & ~high
            diagnostics = {"interval": longest, "entity_ids": list(support),
                           "xy_ge_threshold_counts": high.sum(axis=0).tolist(),
                           "valid_counts": known.sum(axis=0).tolist(),
                           "same_entity_ge_threshold_every_record": [True if high[:, i].all() else
                               (False if low_known[:, i].any() else None) for i in range(len(support))]}
        values[key] = {"true_count": int((combined == 1).sum()),
                       "false_count": int((combined == 0).sum()),
                       "unknown_count": int((combined == -1).sum()),
                       "intervals": intervals, "longest_true_diagnostics": diagnostics}
    return episode.record("low_progress_activity", "temporal_relation", (*episode.entities["root"], *support),
        ("base_pos_w", "support_lin_vel_w"), values, unit="sample_count", frame="recorder_readout",
        config={"operator_version": ACTIVITY_VERSION, "numeric_config": NUMERIC_CONFIG,
                "meaning": "same recorder rows; any XY speed >=0.05 versus all XY speeds <0.05 during low-progress confirmations",
                "physical_synchrony": "not_established", "step_or_gait_inference": False})


def cohort_binding(header: dict, catalog_version: str) -> dict:
    source = header["source"]
    return {"manifest_sha256": source["manifest_sha256"], "trial_id": source["trial_id"],
            "checkpoint_hash": source["checkpoint_hash"], "task_id": source["task_id"],
            "evaluation_binding": source["evaluation_binding"],
            "seed_namespace": source["seed_namespace"],
            "robot_projection_hash": source["runtime_projection_hash"],
            "semantics_hash": source["semantics_hash"], "catalog_version": catalog_version,
            "acceptance_split": source["acceptance_split"], "profile": source["profile"],
            "entities": header["entities"]}


def summarize_episode(header: dict, measurements: dict[str, dict], references: dict[str, dict],
                      activity: dict | None = None) -> dict:
    """Compact sufficient statistics, with links to complete P1a measurements."""
    n, dt = header["coverage"]["valid_rows"], header["coverage"]["dt_s"]
    missing_steps = header["coverage"]["last_step"] - header["coverage"]["first_step"] + 1 - n
    metrics: dict[str, dict] = {}
    highlights: dict[str, Any] = {}

    def scalar(key, value, ref_key, valid_samples, *, total_samples=n, unit="1", count_unit="record"):
        metrics[key] = {"kind": "scalar", "value": value, "unit": unit,
                        "valid_samples": valid_samples, "unknown_samples": total_samples - valid_samples,
                        "total_samples": total_samples, "source_keys": [ref_key],
                        "count_unit": count_unit, "unobserved_control_steps": missing_steps}

    def ratio(key, numerator, denominator, valid_samples, ref_key, *, true_count=None,
              unknown=None, weight="sample_cell_duration_proxy", unit="1"):
        value = numerator / denominator if abs(denominator) > 1e-12 else None
        metrics[key] = {"kind": "ratio", "value": value, "numerator": numerator,
                        "denominator": denominator, "weight": weight, "unit": unit,
                        "valid_samples": valid_samples, "unknown_samples": n - valid_samples,
                        "total_samples": n, "source_keys": [ref_key],
                        "count_unit": "record", "unobserved_control_steps": missing_steps}
        if true_count is not None:
            metrics[key]["true_samples"] = true_count
            metrics[key]["presence"] = observed_presence(true_count, (n - valid_samples if unknown is None else unknown) + missing_steps)

    def track(key, value, ref_key, *, state="true"):
        positive = value["true_count"] if state == "true" else value["false_count"]
        valid = value["true_count"] + value["false_count"]
        ratio(key, positive * dt, valid * dt, valid, ref_key, true_count=positive)
        metrics[key]["runs"] = run_summary(value["intervals"], state)
        metrics[key]["onset_count"] = value.get("observed_onset_count")
        metrics[key]["offset_count"] = value.get("observed_offset_count")

    for key, record in measurements.items():
        v = record["value"]
        if key == "root_progress":
            scalar("net_progress_m", v["net_progress_m"], key, v["valid_count"], unit="m")
            scalar("complete_path_xy_m", v["sampled_path_xy_m"] if v["path_complete"] else None,
                   key, v["valid_edge_count"], total_samples=max(n - 1, 0), unit="m", count_unit="adjacent_record_edge")
            scalar("max_drawdown_m", v["drawdown"]["max_drawdown_m"], key, v["valid_count"], unit="m")
            highlights["drawdown"] = v["drawdown"]
            highlights["path_complete"] = v["path_complete"]
        elif key == "low_progress":
            track("low_progress", v, key)
        elif key == "body_up":
            track("body_up_negative", v["negative"], key)
            scalar("minimum_up_z", v["up_z"]["minimum"], key, v["up_z"]["valid_count"])
            highlights["first_negative"] = next((r["steps_half_open"][0] for r in v["negative"]["intervals"] if r["state"] == "true"), None)
        elif key.endswith(".marker") or key.endswith(".current_marker"):
            track(key, v, key)
            metrics[key]["off_runs"] = run_summary(v["intervals"], "false")
        elif key.endswith(".low_speed_off"):
            track(key, v, key)
            highlights[key] = v["longest"]
        elif key.endswith(".history_diagnostic"):
            track(key, v, key)
            ratio(key + ".history_on_current_off", v["history_any_on_current_off_count"] * dt,
                  v["joint_valid_count"] * dt, v["joint_valid_count"], key,
                  true_count=v["history_any_on_current_off_count"])
        elif key == "contact_combinations":
            for code, count in enumerate(v["counts"]):
                ratio(f"combination.{code}", count * dt, v["valid_count"] * dt, v["valid_count"], key, true_count=count)
        elif key == "force_summary":
            for field, valid_key, sums_key, rect_key in (
                ("norm_share", "joint_valid_count", "sample_sum_norm_n", "rectangle_norm_ns"),
                ("signed_fz_share", "signed_fz_joint_valid_count", "sample_sum_signed_fz_n", "rectangle_signed_fz_ns")):
                for i, entity in enumerate(header["entities"]["support"]):
                    ratio(f"{entity}.{field}", v[rect_key][i], sum(v[rect_key]), v[valid_key], key,
                          weight="ratio_of_sample_cell_force_sums", unit="force_share")
                    # P1a eligibility uses the unweighted sum in N, not N*s.
                    # Changing dt must not alter an episode's share eligibility.
                    metrics[f"{entity}.{field}"]["value"] = v[field][i]
                    metrics[f"{entity}.{field}"]["episode_share_eligibility"] = "P1a abs(sum_sample_force_N)>1e-12"
                    metrics[f"{entity}.{field}"]["sample_force_sum_n"] = v[sums_key][i]
        elif key.endswith(".relative_position"):
            entity = record["entity_ids"][0]
            for axis, name in enumerate("xyz"):
                r = v["body_xyz_m"][axis]
                scalar(f"{entity}.relative_{name}_range_m", r["range"], key, r["valid_count"], unit="m")
        elif key.endswith(".pose"):
            scalar(key + ".range_rad", v["range"], key, v["valid_count"], unit="rad")
        elif key == "task_endpoint":
            scalar("terminal_time_s", v["terminal_time_s"], key, n, unit="s")
            highlights["terminal_flags"] = v["terminal_flags"]
    if activity is not None:
        for key, value in activity["value"].items():
            track(key, value, "low_progress_activity")
    # Long states use the already-frozen M04 display threshold (0.5 cell seconds).
    for entity in header["entities"]["support"]:
        key = f"{entity}.low_speed_off"
        if key in metrics:
            m = metrics[key]
            longest = m["runs"]["longest"]
            positive = longest is not None and longest["sample_cell_duration_proxy_s"] >= .5
            metrics[key + ".long_observed"] = {"kind": "presence", "value": observed_presence(int(positive), m["unknown_samples"] + missing_steps),
                "source_keys": [key], "unit": "episode", "valid_samples": m["valid_samples"],
                "unknown_samples": m["unknown_samples"], "total_samples": n,
                "count_unit": "record", "unobserved_control_steps": missing_steps}
    return {"record_type": "episode_summary", "record_id": stable_id("summary", header["episode_id"], COHORT_VERSION, ACTIVITY_VERSION),
            "episode_id": header["episode_id"], "display_name": header["display_name"],
            "run_index": header["run_index"], "env_index": header["env_index"],
            "source_manifest_sha256": header["source"]["manifest_sha256"],
            "task": {k: header["task"][k] for k in ("outcome", "cell_id", "stratum_id", "goal_progress_m", "deadline_s")},
            "coverage": header["coverage"], "metrics": metrics, "highlights": highlights,
            "references": references, "activity": activity}


def aggregate_metric(members: list[dict], key: str) -> dict:
    supplied = [m["metrics"][key] for m in members if key in m["metrics"]]
    known = [m for m in supplied if m["value"] is not None]
    out = {"total_episode_count": len(members), "processed_episode_count": len(supplied),
           "valid_episode_count": len(known), "unknown_episode_count": len(supplied) - len(known),
           "unprocessed_episode_count": len(members) - len(supplied),
           "valid_samples": sum(m["valid_samples"] for m in supplied),
           "unknown_samples": sum(m["unknown_samples"] for m in supplied),
           "total_processed_samples": sum(m["total_samples"] for m in supplied),
           "unobserved_control_steps": sum(m["unobserved_control_steps"] for m in supplied),
           "count_unit": supplied[0]["count_unit"] if supplied else None,
           "unit": supplied[0]["unit"] if supplied else None,
           "episode_equal": distribution([m["value"] for m in known]),
           "time_scope": "whole_available_episode_records; unequal termination lengths; no post-outcome inference"}
    if not supplied:
        return out
    if supplied[0]["kind"] == "ratio":
        # A zero-force episode remains an unknown per-episode share; its zero
        # operands are nevertheless part of the pooled physical-channel sum.
        num = sum(m["numerator"] for m in supplied)
        den = sum(m["denominator"] for m in supplied)
        out["pooled"] = {"numerator": num, "denominator": den,
                         "value": num / den if abs(den) > POOLED_ZERO_TOLERANCE else None,
                         "weight": supplied[0]["weight"], "zero_tolerance": POOLED_ZERO_TOLERANCE,
                         "operand_unit": "N*s" if supplied[0]["unit"] == "force_share" else "sample_cell_s"}
        if "sample_force_sum_n" in supplied[0]:
            out["sample_force_sum_n"] = sum(m["sample_force_sum_n"] for m in supplied)
    if "presence" in supplied[0] or supplied[0]["kind"] == "presence":
        states = [m.get("presence", m["value"]) for m in supplied]
        positive, negative = sum(v is True for v in states), sum(v is False for v in states)
        out["observed_presence"] = {"numerator": positive, "valid_denominator": positive + negative,
            "unknown_episode_count": sum(v is None for v in states),
            "unprocessed_episode_count": len(members) - len(supplied), "total_episode_count": len(members),
            "meaning": "witness in available records; absence only with complete predicate coverage; not future absence"}
    for run_key in ("runs", "off_runs"):
        if run_key in supplied[0]:
            runs = [m[run_key] for m in supplied]
            out[run_key] = {k: sum(r[k] for r in runs) for k in
                           ("count", "complete_count", "left_censored_count", "right_censored_count", "censored_count")}
            out[run_key]["complete_observed_span_s"] = distribution([v for r in runs for v in r["complete_observed_spans_s"]])
            out[run_key]["unit"] = "run_equal; sampled span, not physical duration"
    return out


def aggregate_cohort(summaries: list[dict], binding: dict) -> dict:
    if not summaries:
        raise ValueError("empty cohort")
    if len({e["episode_id"] for e in summaries}) != len(summaries):
        raise ValueError("duplicate cohort episode")
    if {s["source_manifest_sha256"] for s in summaries} != {binding["manifest_sha256"]}:
        raise ValueError("cross-source cohort pooling is prohibited")
    groups = {"all": summaries,
              "success": [e for e in summaries if e["task"]["outcome"]["success"]],
              "failure": [e for e in summaries if not e["task"]["outcome"]["success"]]}
    for field in ("stratum_id", "cell_id"):
        for value in sorted({e["task"][field] for e in summaries}):
            subset = [e for e in summaries if e["task"][field] == value]
            prefix = f"{field}:{value}"
            groups[prefix] = subset
            for result in ("success", "failure"):
                groups[f"{prefix}/{result}"] = [e for e in subset if e["task"]["outcome"]["success"] == (result == "success")]
    for reason in sorted({e["task"]["outcome"]["primary_reason"] for e in summaries}):
        groups[f"outcome:{reason}"] = [e for e in summaries if e["task"]["outcome"]["primary_reason"] == reason]
    keys = sorted({key for e in summaries for key in e["metrics"]})
    facts = {}
    for name, members in groups.items():
        facts[name] = {"record_type": "cohort_summary", "record_id": stable_id("cohort", COHORT_VERSION, binding, name,
                        sorted(e["episode_id"] for e in members)), "group": name,
                      "member_ids": [e["record_id"] for e in members], "episode_count": len(members),
                      "recorded_rows": sum(e["coverage"]["valid_rows"] for e in members),
                      "outcome_counts": dict(sorted(Counter(e["task"]["outcome"]["primary_reason"] for e in members).items())),
                      "metrics": {key: aggregate_metric(members, key) for key in keys},
                      "weighting": "episode_equal; pooled channel/sample-cell operands separately; no task-condition weights"}
    return {"binding": binding, "groups": facts, "condition_weighting": "observed_cohort_composition_only",
            "parent_comparison": {"status": "unavailable", "reason": "no explicitly bound parent evidence supplied; legacy examples are not a parent"}}


def select_examples(summaries: list[dict], required: set[str]) -> list[dict]:
    """Fixed E1 cases plus one median-progress example per observed outcome.

    These are source-wide illustrations, not same-condition comparisons.
    A largest-retreat success witness is named an extreme rather than typical.
    """
    selected = {e["episode_id"]: {"episode_id": e["episode_id"], "reasons": ["frozen_E1_selection"]}
                for e in summaries if e["episode_id"] in required}
    if not required.issubset({e["episode_id"] for e in summaries}):
        raise ValueError("required example absent")

    def add(e, reason):
        selected.setdefault(e["episode_id"], {"episode_id": e["episode_id"], "reasons": []})["reasons"].append(reason)

    for reason in sorted({e["task"]["outcome"]["primary_reason"] for e in summaries}):
        pool = [e for e in summaries if e["task"]["outcome"]["primary_reason"] == reason
                and e["metrics"].get("net_progress_m", {}).get("value") is not None]
        if pool:
            center = float(np.median([e["metrics"]["net_progress_m"]["value"] for e in pool]))
            chosen = min(pool, key=lambda e: (abs(e["metrics"]["net_progress_m"]["value"] - center), e["episode_id"]))
            add(chosen, "outcome_median_net_progress_example:" + reason)
    pool = [e for e in summaries if e["task"]["outcome"]["success"]
            and e["highlights"].get("drawdown", {}).get("return_step") is not None
            and (e["highlights"]["drawdown"]["max_drawdown_m"] or 0) > 0]
    if pool:
        add(min(pool, key=lambda e: (-e["highlights"]["drawdown"]["max_drawdown_m"], e["episode_id"])),
            "largest_observed_successful_retreat_with_return_not_typical")
    # Preserve successful witnesses of each reported long-off mode, preferably
    # with a same-cell failure witness. This does not alter any frozen E1 choice.
    keys = sorted({k for e in summaries for k in e["metrics"] if k.endswith(".long_observed")})
    for key in keys:
        successes = [e for e in summaries if e["task"]["outcome"]["success"] and e["metrics"].get(key, {}).get("value") is True]
        failures = [e for e in summaries if not e["task"]["outcome"]["success"] and e["metrics"].get(key, {}).get("value") is True]
        if successes:
            paired_cells = {e["task"]["cell_id"] for e in failures}
            chosen = min(successes, key=lambda e: (e["task"]["cell_id"] not in paired_cells, e["episode_id"]))
            add(chosen, "successful_witness_of_mode:" + key)
            matches = [e for e in failures if e["task"]["cell_id"] == chosen["task"]["cell_id"]]
            if matches:
                add(min(matches, key=lambda e: e["episode_id"]), "same_cell_failure_witness_of_mode:" + key)
    return sorted(selected.values(), key=lambda e: e["episode_id"])
