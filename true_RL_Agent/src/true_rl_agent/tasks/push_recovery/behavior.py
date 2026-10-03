"""Deterministic Push descriptions over sealed public-dev numerical evidence.

This is a view, not a second success criterion. The P2 reducer owns recovery and
terminal reconstruction; this module describes its windows and the stored poses.
No archive is opened by prompt projection or history replay.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
import json
from typing import Any

import numpy as np

from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.runner.numeric_evidence import iter_numeric_evidence, load_numeric_evidence_manifest
from true_rl_agent.tasks.push_recovery.numeric_capture import (
    CONDITIONS, POSE_FIELDS, recompute_push_numeric_evidence,
)
from true_rl_agent.tasks.push_recovery.reward_inputs import PushRecoveryV2RewardInputs

PUSH_BEHAVIOR_V1_SCHEMA = "true_rl_agent.push_recovery.behavior.v1"
PUSH_BEHAVIOR_SCHEMA = "true_rl_agent.push_recovery.behavior.v2"
MAX_PUSH_BEHAVIOR_BYTES = 12 * 1024


def push_behavior_context(manifest: Any, *, evaluation_plan_hash: str,
                          checkpoint_hash: str, program_source_checksum: str,
                          runtime_dynamics_hash: str, execution_source_hash: str | None = None) -> dict[str, Any]:
    """Use the evaluator's already validated cell, including its actual input ABI."""
    fields = list(PushRecoveryV2RewardInputs.field_names())
    input_hash = canonical_json_hash({"schema_version": "true_rl_agent.push_reward_inputs.v2", "fields": fields})
    if manifest.contract_hashes.get("reward_input_schema_hash") != input_hash:
        raise ValueError("Push behavior requires the actual physical reward input ABI")
    projection = manifest.robot_task_binding["task_projection"]
    parameters = projection.get("push_resolved_evaluation_protocol", projection["push_resolved_protocol"])
    return {
        "evaluation_plan_hash": evaluation_plan_hash, "checkpoint_hash": checkpoint_hash,
        "program_source_checksum": program_source_checksum, "runtime_dynamics_hash": runtime_dynamics_hash,
        "resolved_cell_manifest_hash": manifest.manifest_hash,
        "cell_variant_id": manifest.cell_variant_id,
        "robot_profile_id": manifest.robot_profile["robot_profile_id"],
        "resolved_protocol_hash": parameters["resolved_protocol_hash"],
        "reward_input_schema_hash": input_hash, "reward_input_fields": fields,
        "execution_source_hash": execution_source_hash,
    }


def _actionability(context: Mapping[str, Any]) -> dict[str, Any]:
    fields = set(context["reward_input_fields"])
    required = {"base_height_relative", "base_lin_vel_scaled", "base_ang_vel_scaled", "roll", "pitch"}
    if not required <= fields:
        raise ValueError("Push behavior actionability inputs are absent")
    return {
        "input_schema_hash": context["reward_input_schema_hash"],
        "instantaneous_stability": {"status": "expressible", "inputs": sorted(required),
            "semantics": "scaled world velocities; height relative to robot scale; signed roll/pitch; reward-rate shaping is a hypothesis"},
        "root_direction": {"status": "proxy_only", "inputs": ["base_lin_vel_scaled", "push_vec_public", "push_active"],
            "semantics": "world velocity and current executed training velocity push; force is evaluation-only"},
        "dwell_and_post_recovery": {"status": "diagnostic_only",
            "semantics": "window age, recovery latch and future sequence completion are not reward inputs; progress_fraction is training progress"},
        "reset_relative_displacement": {"status": "requires_new_observation",
            "semantics": "base_pos_xy uses environment origin, not episode reset origin"},
        "footstep_or_support_change": {"status": "unsupported", "reason": "this profile has no feet/contact/yaw evidence"},
        "protocol_or_optimizer_change": {"status": "outside_current_authority"},
    }


def _spans(steps: np.ndarray, mask: np.ndarray, anchor: int, dt: float) -> list[dict[str, Any]]:
    selected = steps[mask]
    if not len(selected):
        return []
    parts = np.split(selected, np.flatnonzero(np.diff(selected) != 1) + 1)
    return [{"start_step": int(part[0]), "end_step_inclusive": int(part[-1]),
             "start_after_anchor_s": float((part[0] - anchor) * dt),
             "end_after_anchor_s": float((part[-1] - anchor) * dt), "observed_steps": len(part)} for part in parts]


def _sample(steps: np.ndarray, pose: np.ndarray, motion: np.ndarray | None, index: int) -> dict[str, Any]:
    result = {"control_step": int(steps[index]), **dict(zip(POSE_FIELDS, pose[index].tolist()))}
    if motion is not None:
        result.update(velocity_world_xy_mps=motion[index, :2].tolist(),
                      reset_relative_position_world_xy_m=motion[index, 2:].tolist())
    return result


def _process(raw: Mapping[str, Any], recovery: Mapping[str, Any], *, force: bool, dt: float,
             width: int, steps: np.ndarray, pose: np.ndarray, motion: np.ndarray | None,
             conditions: np.ndarray, window: np.ndarray) -> dict[str, Any]:
    """Measured sequence, never inferred contact or a replacement success test."""
    if not recovery["anchor_verified"] or not window.any():
        return {"status": "no_verified_observed_window", "reason": recovery["coverage"]}
    indices = np.flatnonzero(window)
    first, last = int(indices[0]), int(indices[-1])
    peak = int(indices[np.argmax(pose[indices, 3])])  # earliest tie, native operands
    anchor = recovery["anchor_control_step"]
    anchor_indices = np.flatnonzero(steps == anchor)
    event_index = raw["planned_push_count"] - 1 if force else int(recovery["identity"].split("_")[1])
    event = next(item for item in raw["force_events" if force else "push_events"] if item["event_index"] == event_index)
    injection_step = event["start_control_step" if force else "control_step"]
    injection_indices = np.flatnonzero(steps == injection_step)
    response_indices = np.flatnonzero(steps > injection_step)
    result = {"status": "available", "reference_event_index": event_index,
        "recovery_anchor_kind": "last_pulse_end" if force else "velocity_injection",
        "recovery_anchor_sample": _sample(steps, pose, motion, int(anchor_indices[0])) if len(anchor_indices) else None,
        "injection_start_sample": _sample(steps, pose, motion, int(injection_indices[0])) if len(injection_indices) else None,
        "first_observed_response": _sample(steps, pose, motion, int(response_indices[0])) if len(response_indices) else None,
        "window_first": _sample(steps, pose, motion, first),
        "window_speed_peak": _sample(steps, pose, motion, peak),
        "window_last": _sample(steps, pose, motion, last),
        "peak_after_anchor_s": float((steps[peak] - anchor) * dt),
        "window_reaches_planned_end": bool(steps[last] == anchor + width),
        "window_abs_roll_pitch_max_rad": np.abs(pose[indices, 1:3]).max(axis=0).tolist(),
        "window_min_height_m": float(pose[indices, 0].min()),
        "stable_spans": _spans(steps, window & conditions.all(axis=1), anchor, dt),
        "displacement_status": "not_collected_in_v1" if motion is None else "anchor_sample_missing",
        "injection_reference_world_xy": list(event["delivered_impulse_world_ns" if force else "observed_delta_v_world_mps"][:2]),
        "injection_reference_units": "N s" if force else "m/s"}
    if motion is not None and len(anchor_indices):
        # The same reset origin cancels. Never substitute the endpoint's
        # cumulative reset displacement, especially for a second push.
        net = motion[last, 2:].astype(float) - motion[int(anchor_indices[0]), 2:].astype(float)
        result.update(displacement_status="available", window_net_displacement_world_xy_m=net.tolist(),
                      window_net_displacement_m=float(np.linalg.norm(net)))
        vector = np.asarray(result["injection_reference_world_xy"], dtype=float)
        norm = float(np.linalg.norm(vector))
        if norm > 0 and np.isfinite(norm):
            along = vector / norm
            result["window_net_displacement_parallel_m"] = float(net @ along)
            result["window_velocity_parallel_first_peak_last_mps"] = (motion[[first, peak, last], :2] @ along).tolist()
    latch = recovery["recovery_control_step"]
    # Stop the per-event post-latch observation before the next *executed*
    # injection. The all-required episode-level view below still spans to end.
    next_starts = [item["start_control_step" if force else "control_step"]
                   for item in raw["force_events" if force else "push_events"] if item["event_index"] > event_index]
    stop = min(next_starts) if next_starts else int(steps[-1])
    later = ((steps > latch) & (steps <= stop) & ~conditions.all(axis=1)
             if latch is not None else np.zeros(len(steps), dtype=bool))
    result["post_latch"] = {"status": "observed" if latch is not None else "not_latched",
        "observed_through_step": stop if latch is not None else None,
        "until": "next_executed_injection" if next_starts else "last_observation",
        "unstable_steps": int(later.sum()), "unstable_spans": _spans(steps, later, anchor, dt)}
    return result


def _event_conditions(raw: Mapping[str, Any], meta: Mapping[str, Any], stratum: str) -> dict[str, Any]:
    force = "force" in meta
    events = raw["force_events" if force else "push_events"]
    magnitudes = meta["parameters"]["impulse_velocity_mps" if force else "velocity_magnitudes_mps"]
    intensity = (0.0 if stratum == "control" else magnitudes["high"] * meta["parameters"]["repeated_high_fraction"]
                 if force and stratum == "repeated" else magnitudes["medium" if stratum == "repeated" else stratum])
    return {"stratum": stratum, "nominal_velocity_or_impulse_per_mass_mps": intensity,
            "planned_events": raw["planned_push_count"], "executed_events": len(events),
            "verified_events": sum(item["injection_verified"] for item in events),
            "completed_events": sum(item["completed"] for item in events) if force else len(events)}


def _transcripts(reference: Mapping[str, Any], manifest: Mapping[str, Any], result: Mapping[str, Any], source: int) -> list[dict[str, Any]]:
    # P2 has already verified validity/closure and the native-dtype conditions.
    # These arrays add descriptions only; they never reclassify the episode.
    needed = {"valid", "control_step", "conditions", "pose", "planar_motion"}
    chunks = [{key: value for key, value in arrays.items() if key in needed} for _, arrays in iter_numeric_evidence(reference)]
    arrays = {key: np.concatenate([chunk[key] for chunk in chunks]) for key in
              ("valid", "control_step", "conditions", "pose", *( ("planar_motion",) if "planar_motion" in chunks[0] else ()))}
    meta = manifest["metadata"]
    dt, width = meta["time"]["step_dt_s"], meta["time"]["recovery_window_steps"]
    force = "force" in meta
    episodes = []
    for record in result["per_episode"]:
        env = record["env_id"]
        valid = arrays["valid"][:, env]
        steps, conditions, pose = (arrays[key][valid, env] for key in ("control_step", "conditions", "pose"))
        motion = arrays["planar_motion"][valid, env] if "planar_motion" in arrays else None
        raw = meta["runtime_episode_records"][env]
        recoveries = []
        for item in record["recoveries"]:
            # The force reducer keeps one storage slot even in no-push control
            # episodes. That slot is not a required or censored recovery unit.
            if item["coverage"] == "no_push_control":
                continue
            anchor = item["anchor_control_step"]
            window = ((steps > anchor) & (steps <= anchor + width)
                      if anchor is not None and (not force or item["anchor_verified"]) else np.zeros(len(steps), dtype=bool))
            spans = {name: _spans(steps, window & ~conditions[:, i], anchor or 0, dt) for i, name in enumerate(CONDITIONS)}
            # Keep non-exclusive conditions; their counts need not sum to N.
            if any(sum(span["observed_steps"] for span in spans[name]) != item["failed_condition_steps"][name]
                   for name in CONDITIONS) and item["anchor_verified"]:
                raise ValueError("Push behavior window differs from the recovery reducer")
            recoveries.append({**deepcopy(item), "condition_violation_spans": spans,
                "process": _process(raw, item, force=force, dt=dt, width=width, steps=steps,
                    pose=pose, motion=motion, conditions=conditions, window=window)})
        latched = [item["recovery_control_step"] for item in recoveries]
        all_latched = bool(latched) and all(value is not None for value in latched)
        later = (steps > max(latched)) & ~conditions.all(axis=1) if all_latched else np.zeros(len(steps), dtype=bool)
        first = int(np.flatnonzero(later)[0]) if later.any() else None
        episodes.append({"source": source, "env_id": env, "episode_index": record["episode_index"],
            "stratum": record["stratum"], "terminal_label": record["terminal_label"], "coverage": record["coverage"],
            "event_conditions": _event_conditions(raw, meta, record["stratum"]),
            "valid_steps": record["valid_steps"], "all_required_recoveries_latched": all_latched,
            "final_control_step": int(steps[-1]), "final_stable": bool(conditions[-1].all()),
            "final_unstable_after_any_recovery": record["final_unstable_after_recovery"],
            "final_failed_conditions": [name for name, passed in zip(CONDITIONS, conditions[-1]) if not passed],
            "final_pose": dict(zip(POSE_FIELDS, pose[-1].tolist())),
            "unstable_steps_after_all_recoveries": int(later.sum()),
            "first_instability_after_all_recoveries": None if first is None else {
                "control_step": int(steps[first]), "after_last_recovery_s": float((steps[first] - max(latched)) * dt),
                "failed_conditions": [name for name, passed in zip(CONDITIONS, conditions[first]) if not passed]},
            "recoveries": recoveries})
    return episodes


def _group(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    units = [unit for row in records for unit in row["recoveries"]]
    interruptions = [event for unit in units for event in unit["dwell_interruptions"]]
    processes = [unit["process"] for unit in units if unit.get("process", {}).get("status") == "available"]
    result = {"episodes": len(records), "recovery_units": len(units),
        "terminal_counts": dict(sorted(Counter(row["terminal_label"] or "censored" for row in records).items())),
        "window_coverage": dict(sorted(Counter(unit["coverage"] for unit in units).items())),
        "observed_window_steps": sum(unit["observed_window_steps"] for unit in units),
        "stable_window_steps": sum(unit["stable_window_steps"] for unit in units),
        "zero_stable_units": sum(unit["stable_window_steps"] == 0 for unit in units),
        "interrupted_episodes": sum(any(unit["dwell_interruptions"] for unit in row["recoveries"]) for row in records),
        "dwell_interruptions": len(interruptions),
        "interruption_condition_counts": {name: sum(name in event["failed_conditions"] for event in interruptions) for name in CONDITIONS},
        "failed_condition_steps": {name: sum(unit["failed_condition_steps"][name] for unit in units) for name in CONDITIONS},
        "all_recovered_then_final_unstable": sum(row["all_required_recoveries_latched"] and bool(row["final_failed_conditions"]) for row in records),
        "any_instability_after_all_recoveries": sum(row["unstable_steps_after_all_recoveries"] > 0 for row in records),
        "planned_events": sum(row["event_conditions"]["planned_events"] for row in records),
        "executed_events": sum(row["event_conditions"]["executed_events"] for row in records),
        "unobserved_anchor_units": sum(unit["anchor_control_step"] is None for unit in units),
        "zero_stable_observed_units": sum(unit["observed_window_steps"] > 0 and unit["stable_window_steps"] == 0 for unit in units),
        "incomplete_window_units": sum(not process["window_reaches_planned_end"] for process in processes),
        "episodes_with_window_tilt_violation": sum(any(unit["failed_condition_steps"]["roll"] or unit["failed_condition_steps"]["pitch"] for unit in row["recoveries"]) for row in records)}
    result["stable_window_fraction"] = result["stable_window_steps"] / result["observed_window_steps"] if result["observed_window_steps"] else None
    measurements = {"first_speed_mps": [p["window_first"]["planar_speed_mps"] for p in processes],
        "peak_speed_mps": [p["window_speed_peak"]["planar_speed_mps"] for p in processes],
        "peak_after_anchor_s": [p["peak_after_anchor_s"] for p in processes],
        "window_net_displacement_m": [p["window_net_displacement_m"] for p in processes if p["displacement_status"] == "available"]}
    result["window_measurements"] = {name: {"observed_units": len(values), "median": float(np.median(values)) if values else None}
                                     for name, values in measurements.items()}
    return result


def _matched_profiles(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """All actual-event strata, with success compared only inside its condition."""
    keys = sorted({json.dumps(row["event_conditions"], sort_keys=True) for row in records})
    profiles = []
    for key in keys:
        rows = [row for row in records if json.dumps(row["event_conditions"], sort_keys=True) == key]
        failures = Counter(row["terminal_label"] or "censored" for row in rows
                           if row["terminal_label"] not in {"success_recovered", "no_push_control"})
        for label, count in sorted(failures.items(), key=lambda item: (-item[1], item[0])):
            profiles.append({"conditions": json.loads(key), "failure_label": label, "failure_episodes": count,
                "condition_episodes": len(rows),
                "failure": _group([row for row in rows if (row["terminal_label"] or "censored") == label]),
                "success": _group([row for row in rows if row["terminal_label"] == "success_recovered"])})
    return sorted(profiles, key=lambda row: (-row["failure_episodes"], -row["conditions"]["executed_events"],
                                           row["conditions"]["stratum"], row["failure_label"]))


def build_push_behavior(seed_summaries: Sequence[Mapping[str, Any]], *, context: Mapping[str, Any]) -> dict[str, Any]:
    """Build full derived evidence; caller may export it, phase seals bounded view."""
    binding = {key: value for key, value in context.items() if key != "reward_input_fields"}
    base = {"schema_version": PUSH_BEHAVIOR_SCHEMA, "rule_version": "push_process_v2", "binding": binding,
        "actionability": _actionability(context), "parent_diff": {"status": "missing", "reason": "no_bound_generation_parent"}}
    references = [item.get("numeric_evidence_ref") for item in seed_summaries]
    if not references or any(reference is None for reference in references):
        return {**base, "status": "missing", "reason": "numeric_archive_not_available_for_every_public_dev_run",
                "coverage": {"expected_runs": len(references), "recorded_runs": sum(ref is not None for ref in references)}}
    sources, episodes, domain, definitions = [], [], None, None
    manifests = sorted(((load_numeric_evidence_manifest(ref), ref) for ref in references), key=lambda pair: pair[0]["identity"]["run_index"])
    if len({manifest["identity"]["run_index"] for manifest, _ in manifests}) != len(manifests):
        raise ValueError("Push behavior duplicate run identity")
    for source, (manifest, reference) in enumerate(manifests):
        identity, meta = manifest["identity"], manifest["metadata"]
        if identity.get("split") != "public_dev" or any(identity.get(key) != context[key] for key in
            ("evaluation_plan_hash", "checkpoint_hash", "program_source_checksum", "runtime_dynamics_hash", "resolved_cell_manifest_hash", "resolved_protocol_hash")):
            raise ValueError("Push behavior source binding mismatch")
        verification = identity.get("backend_execution_environment_verification", {})
        execution_hash = verification.get("actual_manifest_hash") if verification.get("status") == "pass" else None
        current_domain = {key: context[key] for key in ("cell_variant_id", "robot_profile_id", "resolved_cell_manifest_hash", "resolved_protocol_hash", "runtime_dynamics_hash", "reward_input_schema_hash", "execution_source_hash")}
        current_domain.update(evaluation_binding_hash=identity["evaluation_binding"]["evaluation_binding_hash"],
            execution_environment_hash=execution_hash, runs=len(manifests), environments_per_run=meta["num_envs"],
            step_dt_s=meta["time"]["step_dt_s"],
            strata={str(key): value for key, value in sorted(Counter(meta["strata"]).items())},
            planned_push_counts={str(key): value for key, value in sorted(Counter(meta["planned_push_counts"]).items())})
        if domain is not None and current_domain != domain:
            raise ValueError("Push behavior mixed comparison domains")
        domain = current_domain
        result = recompute_push_numeric_evidence(reference)
        if not result["matches_runtime"]:
            raise ValueError("Push behavior numerical reconstruction failed")
        sources.append({"manifest_sha256": reference["manifest_sha256"], "direction_status": result["direction_status"],
                        "episodes": result["episodes"], "episode_coverage": result["episode_coverage"]})
        episodes.extend(_transcripts(reference, manifest, result, source))
        p = meta["parameters"]
        definitions = {"role": "root", "clock": "pre_reset_control_step", "step_dt_s": meta["time"]["step_dt_s"],
            "anchor": "complete_sequence_last_pulse_end" if "force" in meta else "each_executed_velocity_event",
            "window_rule": "0 < step - anchor <= window_steps", "window_steps": meta["time"]["recovery_window_steps"],
            "dwell_scope": "max dwell and interruptions stop at first verified latch; stable/failed-condition step counts cover the entire observed recovery window",
            "required_dwell_steps": meta["time"]["required_dwell_steps"],
            "stable_bounds": {"planar_speed_max_mps": p["recovery_velocity_mps"], "angular_speed_max_radps": p["recovery_angular_velocity_radps"],
                "absolute_roll_pitch_max_rad": p["recovery_tilt_rad"], "height_min_m": p["recovery_height_m"]},
            "vocabulary": ["condition_violation_span", "dwell_interrupted", "recovery_latched", "instability_after_all_recoveries", "root_planar_motion"],
            "limits": "condition counts overlap; recovered windows may later be unstable and still succeed; no contact/footstep evidence; descriptive, not causal"}
        definitions.update(
            displacement="window end minus recovery anchor position in world xy; velocity: this injection; force: last pulse end, excludes motion during the pulse; never cumulative reset drift",
            process="first response is first recorded step after injection start; force window starts after last completed pulse; peak is earliest observed planar-speed maximum in the recovery window",
            comparison="same stratum, nominal intensity, planned/executed/verified/completed event counts; not identical directions or causal pairs",
            statistics="window medians cover verified observed windows only, with per-metric observed_units; no-window units are not zero-valued measurements")
    profile = {"all": _group(episodes), "by_stratum": {stratum: _group([row for row in episodes if row["stratum"] == stratum])
                for stratum in sorted({row["stratum"] for row in episodes})},
        "success_failure_comparison": {label: _group([row for row in episodes if row["terminal_label"] == label])
            for label in ("not_recovered", "success_recovered")}}
    # Within-stratum counterexamples share the same measured population, not a
    # causal pairing. Keep the complete terminal denominator above as well.
    for stratum, group in profile["by_stratum"].items():
        group["success_failure_comparison"] = {}
        for label in ("not_recovered", "success_recovered"):
            contrast = _group([row for row in episodes if row["stratum"] == stratum and row["terminal_label"] == label])
            group["success_failure_comparison"][label] = contrast
    profile["first_run"] = {"source": 0, **_group([row for row in episodes if row["source"] == 0])}
    profile["matched_failure_profiles"] = _matched_profiles(episodes)
    return {**base, "status": "available", "comparison_domain": domain, "sources": sources,
        "definitions": definitions, "coverage": {"runs": len(sources), "episodes": len(episodes),
            "complete_episodes": sum(row["coverage"] == "complete" for row in episodes)},
        "profile": profile, "transcripts": episodes}


def _byte_size(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"))


def _representatives(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    ordered = sorted(rows, key=lambda row: (row["source"], row["env_id"], row["episode_index"]))
    # Prioritize one failure and one successful counterexample. Their numeric
    # group counts cover all other episodes; examples never replace denominators.
    failed = next((row for row in ordered if row["terminal_label"] == "not_recovered"), None)
    if failed is None:
        failed = next((row for row in ordered if row["terminal_label"] not in {"success_recovered", "no_push_control"}), None)
    successes = [row for row in ordered if row["terminal_label"] == "success_recovered"]
    if failed is not None:
        successes.sort(key=lambda row: (row["stratum"] != failed["stratum"], row["source"] != failed["source"], row["source"], row["env_id"]))
    success = successes[0] if successes else None
    return [row for row in (failed, success) if row is not None]


_GROUP_COLUMNS = ("episodes", "recovery_units", "terminal_counts", "window_coverage", "observed_window_steps",
                  "stable_window_steps", "zero_stable_units", "interrupted_episodes", "dwell_interruptions",
                  "interruption_condition_counts", "failed_condition_steps", "all_recovered_then_final_unstable",
                  "any_instability_after_all_recoveries")


def _public_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Lossless table: write repeated field names and condition names once."""
    if "columns" in profile:
        return deepcopy(dict(profile))
    rows = []
    def append(stratum: str, outcome: str, group: Mapping[str, Any]) -> None:
        values = [[group[key][name] for name in CONDITIONS] if key in {"interruption_condition_counts", "failed_condition_steps"}
                  else deepcopy(group[key]) for key in _GROUP_COLUMNS]
        rows.append([stratum, outcome, *values])
    append("all", "all", profile["all"])
    for outcome, group in profile["success_failure_comparison"].items():
        append("all", outcome, group)
    for stratum, group in profile["by_stratum"].items():
        append(stratum, "all", group)
        for outcome, contrast in group["success_failure_comparison"].items():
            append(stratum, outcome, contrast)
    return {"columns": ["stratum", "outcome", *_GROUP_COLUMNS], "condition_order": list(CONDITIONS), "rows": rows,
            "scope": "all rows include every terminal label; outcome rows are subsets, not additional episodes; condition arrays are nonexclusive counts in condition_order"}


def _all_group(profile: Mapping[str, Any]) -> Mapping[str, Any]:
    if "all" in profile:
        return profile["all"]
    return next(dict(zip(profile["columns"], row)) for row in profile["rows"] if row[:2] == ["all", "all"])


def _compact_transcript(row: Mapping[str, Any], index: Mapping[str, Any]) -> dict[str, Any]:
    compact = deepcopy(dict(row))
    units = compact.pop("recoveries")
    # A single unit retains the actual event/sequence identity. Unrecovered
    # windows take precedence, otherwise use the first interrupted window.
    chosen = next((unit for unit in units if unit["recovery_control_step"] is None), None)
    if chosen is None:
        chosen = next((unit for unit in units if unit["dwell_interruptions"]), units[0] if units else None)
    compact["recovery_units_total"] = len(units)
    compact["omitted_recovery_units"] = len(units) - (chosen is not None)
    compact["recoveries"] = []
    compact["relationship"] = ("index_example" if row is index else
        "same_stratum_same_run" if (row["stratum"], row["source"]) == (index["stratum"], index["source"]) else
        "same_stratum_other_run" if row["stratum"] == index["stratum"] else "unmatched_stratum_context")
    if chosen is not None:
        events, spans = chosen.pop("dwell_interruptions"), chosen.pop("condition_violation_spans")
        chosen["dwell_interruptions"] = events[:2]
        chosen["omitted_interruptions"] = max(0, len(events) - 2)
        chosen["condition_violation_spans"] = {name: values[:1] for name, values in spans.items() if values}
        chosen["omitted_condition_spans"] = sum(max(0, len(values) - 1) for values in spans.values())
        compact["recoveries"].append(chosen)
    return compact


def _project_legacy_behavior(behavior: Mapping[str, Any], *, max_bytes: int) -> dict[str, Any]:
    """Bound only a derived view; disclose omitted spans/episodes and keep counts."""
    view = deepcopy(dict(behavior))
    if "profile" in view:
        view["profile"] = _public_profile(view["profile"])
    if "comparison_domain" in view:
        view["binding"] = {key: value for key, value in view["binding"].items() if key not in view["comparison_domain"]}
    if "transcripts" in view:
        rows = view.pop("transcripts")
        candidates = _representatives(rows)
        view["representative_transcripts"] = [_compact_transcript(row, candidates[0]) for row in candidates]
        view["selection"] = {"rule": "first_not_recovered_else_other_failure; success_prefers_same_stratum_and_run; source_env_tiebreak; one_unrecovered_else_interrupted_else_first_unit",
            "total_episodes": len(rows), "selected_episodes": 0, "omitted_episodes": len(rows),
            "failure_example": "present" if any(row["terminal_label"] not in {"success_recovered", "no_push_control"} for row in candidates) else "missing_group",
            "success_example": "present" if any(row["terminal_label"] == "success_recovered" for row in candidates) else "missing_group"}
    if "selection" in view:
        count = len(view["representative_transcripts"])
        view["selection"].update(selected_episodes=count, omitted_episodes=view["selection"]["total_episodes"] - count)
    # A parent diff can use the remaining headroom. Drop optional terminal
    # operands in fixed representative order before sacrificing any timeline;
    # the final failed predicates, windows, counts and both examples stay intact.
    if _byte_size(view) > max_bytes:
        for transcript in view.get("representative_transcripts", []):
            if "final_pose" not in transcript:
                continue
            transcript.pop("final_pose")
            transcript["final_pose_omitted"] = "public_byte_budget"
            if _byte_size(view) <= max_bytes:
                break
    # If the core still cannot fit, fail instead of trading away an example.
    if _byte_size(view) > max_bytes:
        raise ValueError("Push behavior core exceeds its declared byte budget")
    return view


def _rounded(value: Any) -> Any:
    # Display precision only; the full artifact and all identities remain exact.
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {key: _rounded(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_rounded(item) for item in value]
    return value


def _group_digest(group: Mapping[str, Any], *, outcomes: bool) -> dict[str, Any]:
    result = {"episodes": group["episodes"],
        "stable_steps": {"count": group["stable_window_steps"], "observed": group["observed_window_steps"], "fraction": group["stable_window_fraction"]},
        "dwell_breaks": group["dwell_interruptions"], "interrupted_episodes": group["interrupted_episodes"],
        "latched_final_unstable": group["all_recovered_then_final_unstable"],
        "later_instability": group["any_instability_after_all_recoveries"],
        "window_tilt_episodes": group["episodes_with_window_tilt_violation"]}
    if outcomes:
        result["terminal_counts"] = deepcopy(group["terminal_counts"])
        result["events"] = {"planned": group["planned_events"], "executed": group["executed_events"]}
        result["windows"] = {"required": group["recovery_units"], "no_anchor": group["unobserved_anchor_units"],
            "short": group["incomplete_window_units"], "observed_zero_stable": group["zero_stable_observed_units"]}
    else:
        measured = group["window_measurements"]
        result["window_medians"] = {"windows": measured["first_speed_mps"]["observed_units"],
            "net_windows": measured["window_net_displacement_m"]["observed_units"],
            "first_mps": measured["first_speed_mps"]["median"], "peak_mps": measured["peak_speed_mps"]["median"],
            "peak_delay_s": measured["peak_after_anchor_s"]["median"], "net_m": measured["window_net_displacement_m"]["median"]}
    return result


def _profile_digest(profile: Mapping[str, Any]) -> dict[str, Any]:
    # Keep the cohort counters used by parent diff verbatim; no display rounding
    # changes an integer or denominator. Per-stratum rows are labeled objects.
    diff_fields = ("episodes", "recovery_units", "terminal_counts", "dwell_interruptions", "zero_stable_units",
                  "all_recovered_then_final_unstable", "any_instability_after_all_recoveries")
    result = {"all": {key: deepcopy(profile["all"][key]) for key in diff_fields},
              "cohort": {key: value for key, value in _group_digest(profile["all"], outcomes=False).items()
                         if key in {"stable_steps", "window_medians"}},
              "by_stratum": {name: {key: value for key, value in _group_digest(group, outcomes=True).items()
                                    if key in {"episodes", "terminal_counts", "events", "windows"}}
                              for name, group in profile["by_stratum"].items()},
              "first_run": {"source": profile["first_run"]["source"], **_group_digest(profile["first_run"], outcomes=True)},
              "matched_failure_profiles": [],
              "scope": "scopes overlap; contrasts are subsets; all outcomes retained; legacy zero_stable_units includes unobserved anchors"}
    selected = set()
    for row in profile["matched_failure_profiles"]:
        stratum = row["conditions"]["stratum"]
        if stratum in selected:
            continue
        selected.add(stratum)
        result["matched_failure_profiles"].append({
            "stratum": stratum,
            "conditions": {"intensity_mps": row["conditions"]["nominal_velocity_or_impulse_per_mass_mps"],
                **{name: row["conditions"][name + "_events"] for name in ("planned", "executed", "verified", "completed")}},
            "failure_label": row["failure_label"],
            "condition_episodes": row["condition_episodes"],
            "failure": _contrast_line(row["failure"]),
            "success": _contrast_line(row["success"]),
            "success_status": "matched" if row["success"]["episodes"] else "missing_matching_events"})
    return _rounded(result)


def _contrast_line(group: Mapping[str, Any]) -> str:
    """One labeled row, not an array requiring several column-index lookups."""
    digest = _rounded(_group_digest(group, outcomes=False))
    stable, medians = digest["stable_steps"], digest["window_medians"]
    return (f"episodes={digest['episodes']}; stable={stable['count']}/{stable['observed']}={stable['fraction']}; "
        f"dwell breaks={digest['dwell_breaks']}({digest['interrupted_episodes']} episodes); "
        f"latched-final-unstable={digest['latched_final_unstable']}, later-unstable={digest['later_instability']}, tilt-window={digest['window_tilt_episodes']} episodes; "
        f"medians(n={medians['windows']},net n={medians['net_windows']}): "
        f"first/peak={medians['first_mps']}/{medians['peak_mps']}m/s, peak+{medians['peak_delay_s']}s, net={medians['net_m']}m")


def _process_representatives(rows: Sequence[Mapping[str, Any]], profiles: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    if not rows:
        return []
    def representative(group: Sequence[Mapping[str, Any]], preferred_source: int | None = None) -> Mapping[str, Any]:
        def stable_fraction(row: Mapping[str, Any]) -> float:
            observed = sum(unit["observed_window_steps"] for unit in row["recoveries"])
            return sum(unit["stable_window_steps"] for unit in row["recoveries"]) / observed if observed else -1.
        middle = float(np.median([stable_fraction(row) for row in group]))
        return min(group, key=lambda row: (abs(stable_fraction(row) - middle),
            row["source"] != preferred_source, row["source"], row["env_id"], row["episode_index"]))
    if not profiles:
        return [representative(rows)]
    focus = profiles[0]
    matched = [row for row in rows if row["event_conditions"] == focus["conditions"]]
    failure = representative([row for row in matched if (row["terminal_label"] or "censored") == focus["failure_label"]])
    successes = [row for row in matched if row["terminal_label"] == "success_recovered"]
    return [failure, representative(successes, failure["source"])] if successes else [failure]


def _process_transcript(row: Mapping[str, Any], *, index: bool) -> dict[str, Any]:
    result = {key: deepcopy(row[key]) for key in ("source", "env_id", "episode_index", "stratum", "terminal_label",
        "coverage", "final_control_step", "final_stable", "final_failed_conditions", "final_pose",
        "all_required_recoveries_latched", "unstable_steps_after_all_recoveries")}
    result["relationship"] = "dominant_failure_example" if index and row["terminal_label"] not in {"success_recovered", "no_push_control"} else "matched_success_context" if not index else "no_failure_in_cohort"
    result["all_latched"] = result.pop("all_required_recoveries_latched")
    result["late_unstable_steps"] = result.pop("unstable_steps_after_all_recoveries")
    result["recoveries"] = []
    def number(value: Any) -> str:
        return json.dumps(_rounded(value), separators=(",", ":"))
    units = row["recoveries"]
    observed = [unit for unit in units if unit["observed_window_steps"]]
    detail = next((unit for unit in observed if unit["recovery_control_step"] is None), observed[-1] if observed else None)
    for unit in row["recoveries"]:
        if unit is not detail:
            post = unit["process"].get("post_latch", {})
            result["recoveries"].append({"identity": unit["identity"], "coverage": unit["coverage"],
                "summary": f"anchor={unit['anchor_control_step']}, scheduled={unit['scheduled_anchor_control_step']}, verified={unit['anchor_verified']}; "
                    f"stable={unit['stable_window_steps']}/{unit['observed_window_steps']} observed steps, max dwell={unit['max_dwell_steps']}, "
                    f"latch={unit['recovery_control_step']}, censored={unit['recovery_censored']}; "
                    f"post-latch unstable steps={post.get('unstable_steps', 'unobserved')}",
                "process_detail": "omitted; unit counted"})
            continue
        compact = {key: deepcopy(unit[key]) for key in ("identity", "anchor_control_step",
            "anchor_verified", "coverage", "recovery_censored", "observed_window_steps", "stable_window_steps",
            "max_dwell_steps", "recovery_control_step")}
        for old, new in (("anchor_control_step", "anchor"), ("anchor_verified", "verified"),
                         ("recovery_censored", "censored"), ("observed_window_steps", "window_steps"),
                         ("stable_window_steps", "stable_steps"), ("max_dwell_steps", "max_dwell"),
                         ("recovery_control_step", "latch")):
            compact[new] = compact.pop(old)
        compact["condition_evolution"] = {}
        for name, spans in unit["condition_violation_spans"].items():
            if spans:
                longest = max(spans, key=lambda item: item["observed_steps"])
                compact["condition_evolution"][name] = (
                    f"{unit['failed_condition_steps'][name]} violated steps/{len(spans)} spans; "
                    f"first {spans[0]['start_step']}..{spans[0]['end_step_inclusive']}; "
                    f"longest {longest['start_step']}..{longest['end_step_inclusive']}")
        interruptions = unit["dwell_interruptions"]
        if interruptions:
            first = interruptions[0]
            compact["dwell_breaks"] = f"{len(interruptions)} total; first step {first['control_step']} after {first['preceding_dwell_steps']} stable steps: {','.join(first['failed_conditions'])}"
        process = unit["process"]
        if process["status"] != "available":
            compact["process"] = deepcopy(process)
        else:
            anchor = process["recovery_anchor_sample"]
            response = process["first_observed_response"]
            samples = "; ".join(f"{label} {process[key]['control_step']}:{number(process[key]['planar_speed_mps'])}"
                for label, key in (("first", "window_first"), ("peak", "window_speed_peak"), ("last", "window_last")))
            compact["speed_mps"] = f"anchor {number(anchor['planar_speed_mps']) if anchor else 'missing'}; window step:value {samples}; peak +{number(process['peak_after_anchor_s'])}s"
            if process["recovery_anchor_kind"] == "last_pulse_end":
                compact["first_response"] = None if response is None else {"step": response["control_step"], "speed_mps": response["planar_speed_mps"]}
            compact["window_complete"] = process["window_reaches_planned_end"]
            compact["window_pose"] = f"max |roll,pitch|={number(process['window_abs_roll_pitch_max_rad'])} rad; min height={number(process['window_min_height_m'])} m"
            compact["injection"] = f"step {process['injection_start_sample']['control_step'] if process['injection_start_sample'] else 'missing'}; xy={number(process['injection_reference_world_xy'])} {process['injection_reference_units']}"
            compact["window_net_xy_m"] = process.get("window_net_displacement_world_xy_m", process["displacement_status"])
            if "window_net_displacement_parallel_m" in process:
                compact["window_net_parallel_m"] = process["window_net_displacement_parallel_m"]
            post = process["post_latch"]
            spans = post["unstable_spans"]
            compact["post_latch"] = f"{post['status']}; {post['unstable_steps']} unstable steps, first {spans[0]['start_step'] if spans else 'none'}; observed to {post['observed_through_step']} ({post['until']})"
        result["recoveries"].append(compact)
    return _rounded(result)


def project_push_behavior(behavior: Mapping[str, Any], *, max_bytes: int = MAX_PUSH_BEHAVIOR_BYTES) -> dict[str, Any]:
    """No archive IO: historical v1 stays v1; v2 prioritizes all strata + process."""
    if behavior.get("schema_version") == "true_rl_agent.push_recovery.behavior.public.v4":
        from .trajectory_report import validate_push_trajectory_report
        return validate_push_trajectory_report(behavior, max_bytes=max_bytes)
    if behavior.get("schema_version") == "true_rl_agent.push_recovery.behavior.full.v4":
        raise ValueError("Push trajectory full replay requires explicit CPU source validation; deliver the sealed public report")
    if behavior.get("schema_version") in {"true_rl_agent.push_recovery.behavior.v3", "true_rl_agent.push_recovery.behavior.public.v3"}:
        from true_rl_agent.tasks.push_recovery.process_report import project_push_process_behavior
        return project_push_process_behavior(behavior, max_bytes=max_bytes)
    if behavior.get("schema_version") == PUSH_BEHAVIOR_V1_SCHEMA:
        return _project_legacy_behavior(behavior, max_bytes=max_bytes)
    if behavior.get("schema_version") != PUSH_BEHAVIOR_SCHEMA:
        raise ValueError("unknown Push behavior schema")
    view = deepcopy(dict(behavior))
    if "comparison_domain" in view:
        view["binding"] = {key: value for key, value in view["binding"].items() if key not in view["comparison_domain"]}
    if "transcripts" in view:
        rows = view.pop("transcripts")
        profiles = view["profile"]["matched_failure_profiles"]
        candidates = _process_representatives(rows, profiles)
        view["profile"] = _profile_digest(view["profile"])
        view["representative_transcripts"] = [_process_transcript(row, index=i == 0) for i, row in enumerate(candidates)]
        view["selection"] = {"rule": "largest failure group; median stability example; matched success; event-count/source/env ties",
            "total_episodes": len(rows), "selected_episodes": len(candidates), "omitted_episodes": len(rows) - len(candidates),
            "success_example": "matched" if len(candidates) == 2 else "no_matched_success_or_no_failure",
            "display_precision_decimals": 6, "scope": "examples are not population statistics"}
        # Compact wording shares the full report's definitions, never its paths.
        definition = view["definitions"]
        view["definitions"] = {key: definition[key] for key in ("step_dt_s", "anchor", "window_steps", "required_dwell_steps", "stable_bounds")}
        view["definitions"].update(
            clock="pre-reset episode steps; window=anchor+1..anchor+width inclusive; velocity window first=first response",
            counting="stable=all five bounds; failures overlap; dwell/breaks stop at first latch; counts cover whole window; success may later destabilize",
            displacement="world xy end-anchor; force anchor=last pulse end (excludes pulse motion); not reset drift",
            matching="same stratum/intensity/event counts; not direction/causality",
            medians="n=verified observed windows; net n=with motion; missing is not zero",
            example_detail="one unrecovered observed else last window detailed; all units counted; spans inclusive; no feet/contact")
        view["actionability"] = {
            "instantaneous_stability": {"status": "expressible", "inputs": view["actionability"]["instantaneous_stability"]["inputs"], "semantics": "scaled world velocities, relative height, signed tilt"},
            "root_direction": {"status": "proxy_only", "inputs": ["base_lin_vel_scaled", "push_vec_public", "push_active"], "semantics": "training velocity push; force eval-only"},
            "dwell_and_post_recovery": {"status": "diagnostic_only", "semantics": "no latch/window-age input; progress_fraction=training progress"},
            "reset_relative_displacement": {"status": "requires_new_observation", "semantics": "base_pos_xy is environment-relative, not reset/event-relative"},
            "footstep_or_support_change": {"status": "unsupported"},
            "protocol_or_optimizer_change": {"status": "outside_current_authority"}}
    if _byte_size(view) > max_bytes:
        for row in view.get("representative_transcripts", []):
            if "final_pose" in row:
                row.pop("final_pose")
                row["final_pose_omitted"] = "public_byte_budget"
            if _byte_size(view) <= max_bytes:
                break
    if _byte_size(view) > max_bytes:
        raise ValueError(f"Push behavior core exceeds its declared byte budget: {_byte_size(view)} > {max_bytes}")
    return view


def compare_push_behavior(candidate: Mapping[str, Any] | None, parent: Mapping[str, Any] | None,
                          *, expected_parent_checkpoint_hash: str | None) -> dict[str, Any]:
    if any(item and item.get("schema_version") in {"true_rl_agent.push_recovery.behavior.v3", "true_rl_agent.push_recovery.behavior.public.v3"} for item in (candidate, parent)):
        from true_rl_agent.tasks.push_recovery.process_report import compare_push_process_behavior
        return compare_push_process_behavior(candidate, parent, expected_parent_checkpoint_hash=expected_parent_checkpoint_hash)
    if not candidate or not parent or candidate.get("status") != "available" or parent.get("status") != "available":
        return {"status": "missing", "reason": "candidate_or_bound_parent_behavior_missing"}
    if (candidate.get("schema_version"), candidate.get("rule_version")) != (parent.get("schema_version"), parent.get("rule_version")):
        return {"status": "not_comparable", "reason": "behavior_semantics_version_mismatch"}
    if not expected_parent_checkpoint_hash or parent["binding"].get("checkpoint_hash") != expected_parent_checkpoint_hash:
        return {"status": "not_comparable", "reason": "actual_generation_parent_checkpoint_mismatch"}
    if not candidate["comparison_domain"].get("execution_source_hash") or not parent["comparison_domain"].get("execution_source_hash"):
        return {"status": "not_comparable", "reason": "execution_source_unavailable"}
    if (candidate.get("comparison_domain") != parent.get("comparison_domain") or
            not candidate["comparison_domain"].get("execution_environment_hash")):
        return {"status": "not_comparable", "reason": "cell_protocol_cohort_execution_or_input_domain_mismatch"}
    if any(item["coverage"]["complete_episodes"] != item["coverage"]["episodes"] for item in (candidate, parent)):
        return {"status": "not_comparable", "reason": "incomplete_episode_coverage"}
    count_fields = ("episodes", "recovery_units", "dwell_interruptions", "zero_stable_units", "all_recovered_then_final_unstable", "any_instability_after_all_recoveries")
    candidate_group, parent_group = _all_group(candidate["profile"]), _all_group(parent["profile"])
    delta = {key: candidate_group[key] - parent_group[key] for key in count_fields}
    labels = set(candidate_group["terminal_counts"]) | set(parent_group["terminal_counts"])
    return {"status": "available", "parent_checkpoint_hash": expected_parent_checkpoint_hash,
        "candidate_checkpoint_hash": candidate["binding"]["checkpoint_hash"], "scope": "cohort_counts_not_causal_or_episode_pairing",
        "count_delta": delta, "terminal_count_delta": {name: candidate_group["terminal_counts"].get(name, 0) - parent_group["terminal_counts"].get(name, 0) for name in sorted(labels)},
        "direction_comparison": "available" if all(source["direction_status"] == "recorded_world_xy" for item in (candidate, parent) for source in item["sources"]) else "missing_direction_in_at_least_one_source"}
