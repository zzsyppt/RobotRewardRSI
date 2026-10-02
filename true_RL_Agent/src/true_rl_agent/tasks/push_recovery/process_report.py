"""Task-owned Push process sentences, rule residuals and numerical coverage.

The v2 reducer remains the numerical authority.  This module adds a versioned
description of those observations, never a new success or acceptance test.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import statistics
from typing import Any

from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.tasks.push_recovery.numeric_capture import CONDITIONS

PUSH_PROCESS_SCHEMA = "true_rl_agent.push_recovery.behavior.v3"
PUSH_PROCESS_PUBLIC_SCHEMA = "true_rl_agent.push_recovery.behavior.public.v3"
PUSH_PROCESS_RULE = "push_process_v3"
PUSH_PROCESS_MAX_BYTES = 12 * 1024
RESIDUAL_RULE = {
    "version": "push_rule_residual.v1",
    "domain": ["stratum", "executed_events", "recovery_ordinal"],
    "signature": "violated conditions and first violation seconds after verified anchor",
    "first_violation_bins_seconds": [0.5, 2.0],
    "minimum_success_windows": 3,
    "negative_success_fraction": 0.5,
    "episode_aggregation": "any positive; else any missing or no actual window => missing; else negative",
}
COVERAGE_RULE = {
    "version": "push_coverage.v1", "hash_seed": "p3new-20260929",
    "anchors_per_episode": 3, "half_window_steps": 25, "residual_episode_cap": 4,
    "baseline": "one per stratum and success/failure, before residual supplementation",
    "scope": "numerical window plan; media selection and its four-episode cap are separate",
}
_SUCCESS = {"success_recovered", "no_push_control"}
_NAMES = {"planar_speed": "planar speed", "angular_speed": "angular speed",
          "roll": "absolute roll", "pitch": "absolute pitch", "height": "height"}
_COUNT_FIELDS = ("episodes", "recovery_units", "dwell_interruptions", "zero_stable_units",
                 "all_recovered_then_final_unstable", "any_instability_after_all_recoveries")


def _episode_id(row: Mapping[str, Any]) -> str:
    # source is the zero-based ordinal after sorting actual run identities.
    # Keep this stable logical alias; the full transcript retains episode_index.
    return f"r{row['source'] + 1}.e{row['env_id']}"


def _failure(row: Mapping[str, Any]) -> bool:
    return row.get("terminal_label") not in _SUCCESS


def _domain(row: Mapping[str, Any], unit: Mapping[str, Any] | None = None) -> tuple[Any, ...]:
    value = (row["stratum"], row["event_conditions"]["executed_events"])
    ordinal = (0 if unit["identity"] == "sequence" else int(unit["identity"].split("_")[1])) if unit is not None else None
    return value + (ordinal,) if unit is not None else value


def _domain_label(key: tuple[Any, ...]) -> str:
    return f"{key[0]}, {key[1]} executed event(s)" + (f", {key[2]}" if len(key) > 2 else "")


def _signature(unit: Mapping[str, Any], dt: float) -> tuple[tuple[str, int], ...] | None:
    if not unit.get("anchor_verified") or not unit.get("observed_window_steps"):
        return None
    signature = []
    for name in CONDITIONS:
        spans = unit["condition_violation_spans"][name]
        if spans:
            elapsed = (spans[0]["start_step"] - unit["anchor_control_step"]) * dt
            signature.append((name, 0 if elapsed <= 0.5 else 1 if elapsed <= 2.0 else 2))
    return tuple(sorted(signature))


def build_push_residual(full: Mapping[str, Any]) -> dict[str, Any]:
    """Compare measured window signatures, then aggregate one state per failure."""
    rows = full.get("transcripts", [])
    dt = full.get("definitions", {}).get("step_dt_s", 0.0)
    controls: dict[tuple[Any, ...], list[tuple[tuple[str, int], ...]]] = defaultdict(list)
    for row in rows:
        if row["terminal_label"] != "success_recovered":
            continue
        for unit in row["recoveries"]:
            signature = _signature(unit, dt)
            if signature is not None:
                controls[_domain(row, unit)].append(signature)
    episodes, grouped = [], defaultdict(list)
    for row in rows:
        if not _failure(row):
            continue
        units = []
        for unit in row["recoveries"]:
            # Scheduled but unexecuted events remain in the process transcript,
            # not the actual-window comparison domain. No actual windows still
            # yields an episode-level missing state below.
            if unit.get("anchor_control_step") is None:
                continue
            signature = _signature(unit, dt)
            pool = controls[_domain(row, unit)]
            matched = sum(value == signature for value in pool) if signature is not None else 0
            state = ("missing" if signature is None or len(pool) < 3 else
                     "negative" if matched / len(pool) >= 0.5 else "positive")
            units.append({"identity": unit["identity"], "state": state,
                "domain": list(_domain(row, unit)),
                "signature": None if signature is None else [list(item) for item in signature],
                "matching_success_windows": matched, "success_windows": len(pool),
                "reason": "no_verified_observed_window" if signature is None else
                    "insufficient_success_windows" if len(pool) < 3 else "signature_prevalence",
                "evidence_ref": f"episode:{_episode_id(row)}/window:{unit['identity']}"})
        states = {unit["state"] for unit in units}
        state = "positive" if "positive" in states else "missing" if not units or "missing" in states else "negative"
        item = {"episode_id": _episode_id(row), "state": state, "domain": list(_domain(row)), "windows": units}
        episodes.append(item)
        grouped[_domain(row)].append(item)
    # Successful-only domains also get an explicit zero-failure denominator.
    domains = []
    for key in sorted({_domain(row) for row in rows}):
        failures = grouped[key]
        counts = Counter(row["state"] for row in failures)
        domains.append({"domain": list(key), "failures": len(failures),
            **{state: counts[state] for state in ("positive", "negative", "missing")},
            "episodes": sum(_domain(row) == key for row in rows)})
    return {"rule": deepcopy(RESIDUAL_RULE), "interpretation": "rule discrimination, not a physical cause",
            "episodes": episodes, "domains": domains}


def _hash_order(episode_id: str) -> tuple[str, str]:
    return hashlib.sha256(f"{COVERAGE_RULE['hash_seed']}|push|{episode_id}".encode()).hexdigest(), episode_id


def _union_steps(windows: Sequence[Mapping[str, Any]]) -> int:
    total, end = 0, -1
    for window in sorted(windows, key=lambda item: (item["start_step"], item["end_step"])):
        start, stop = window["start_step"], window["end_step"]
        total += max(0, stop - max(start, end + 1) + 1)
        end = max(end, stop)
    return total


def sample_push_coverage(full: Mapping[str, Any], residual: Mapping[str, Any]) -> dict[str, Any]:
    """Plan numerical windows independently of known predicates; do not call vision."""
    rows = {_episode_id(row): row for row in full.get("transcripts", [])}
    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for identity, row in rows.items():
        strata[(row["stratum"], "failure" if _failure(row) else "success")].append(identity)
    baseline = {min(ids, key=_hash_order) for ids in strata.values()}
    residual_ids = {item["episode_id"] for item in residual["episodes"] if item["state"] in {"negative", "missing"}}
    # The residual cap counts selected residual episodes, including overlap with
    # the baseline; selecting baseline never consumes an additional residual slot.
    residual_selected = set(sorted(residual_ids, key=_hash_order)[:4])
    selected = baseline | residual_selected
    windows, coverage = [], []
    for identity in sorted(rows, key=_hash_order):
        row = rows[identity]
        first, last = row["final_control_step"] - row["valid_steps"] + 1, row["final_control_step"]
        episode_windows = []
        if identity in selected:
            sources = (["rule_independent_baseline"] if identity in baseline else []) + (
                ["residual_or_missing_supplement"] if identity in residual_selected else [])
            if not _failure(row):
                sources.append("success_control")
            anchors = sorted({first, (first + last) // 2, last})
            for anchor in anchors:
                episode_windows.append({"episode_id": identity, "start_step": max(first, anchor - 25),
                    "end_step": min(last, anchor + 25), "anchor_step": anchor, "sources": sources,
                    "evidence_ref": f"episode:{identity}/steps:{max(first, anchor - 25)}-{min(last, anchor + 25)}"})
            windows.extend(episode_windows)
        union = _union_steps(episode_windows)
        coverage.append({"episode_id": identity, "sampled": identity in selected,
            "covered_steps": union, "total_steps": row["valid_steps"],
            "time_fraction": union / row["valid_steps"] if row["valid_steps"] else None})
    selected_fractions = [item["time_fraction"] for item in coverage if item["sampled"]]
    return {"rule": deepcopy(COVERAGE_RULE), "windows": windows, "per_episode": coverage,
        "by_stratum_outcome": [{"stratum": key[0], "outcome": key[1], "sampled": len(set(ids) & selected),
             "episodes": len(ids)} for key, ids in sorted(strata.items())],
        "selected_episode_ids": sorted(selected, key=_hash_order),
        "baseline_episode_ids": sorted(baseline, key=_hash_order),
        "residual_selected_episode_ids": sorted(residual_selected, key=_hash_order),
        "residual_unsampled_episode_ids": sorted(residual_ids - selected, key=_hash_order),
        "unselected_episode_ids": sorted(set(rows) - selected, key=_hash_order),
        "residual_requested": len(residual_ids), "residual_cap_selected": len(residual_selected),
        "residual_actual_sampled": len(residual_ids & selected),
        "sampled_time_fraction_median": statistics.median(selected_fractions) if selected_fractions else None,
        "sampled_time_fraction_minimum": min(selected_fractions) if selected_fractions else None,
        "omitted_media": "not_selected_or_transmitted_by_numerical_sampler"}


def _pattern(row: Mapping[str, Any]) -> tuple[str, ...]:
    """Frozen combination: conditions confirmed strictly before the last step."""
    end = row["final_control_step"]
    return tuple(name for name in CONDITIONS if any(
        span["start_step"] < end for unit in row["recoveries"] for span in unit["condition_violation_spans"][name]))


def select_push_process_representatives(full: Mapping[str, Any]) -> dict[str, Any]:
    rows = full.get("transcripts", [])
    if not rows:
        return {"status": "missing", "reason": "no_episode_transcripts", "selected_episode_ids": []}
    profiles = full["profile"]["matched_failure_profiles"]
    if profiles:
        focus = profiles[0]
        matched = [row for row in rows if row["event_conditions"] == focus["conditions"]]
        group = [row for row in matched if (row["terminal_label"] or "censored") == focus["failure_label"]]
    else:
        focus, matched, group = None, rows, rows
    counts = Counter(_pattern(row) for row in group)
    pattern = min(counts, key=lambda value: (-counts[value], value))
    combinations = [row for row in group if _pattern(row) == pattern]
    middle = statistics.median(row["valid_steps"] for row in combinations)
    example = min(combinations, key=lambda row: (abs(row["valid_steps"] - middle), row["source"], row["env_id"], row["episode_index"]))
    controls = [row for row in matched if row["terminal_label"] == "success_recovered"]
    success = min(controls, key=lambda row: (row["source"] != example["source"], abs(row["valid_steps"] - middle),
        row["source"], row["env_id"], row["episode_index"])) if controls else None
    return {"status": "available", "rule": "largest compatible failure group; modal pre-terminal condition set; median duration; source/env/episode tie",
        "pattern": list(pattern), "pattern_n": counts[pattern], "group_n": len(group),
        "tied": sum(count == counts[pattern] for count in counts.values()) > 1,
        "majority": counts[pattern] * 2 > len(group),
        "failure_episode_id": _episode_id(example) if focus else None,
        "context_episode_id": _episode_id(example),
        "success_episode_id": _episode_id(success) if success else None,
        "success_matching": "no_failure_in_cohort_success_context" if success and not focus else
            "same_event_conditions_same_run" if success and success["source"] == example["source"] else
            "same_event_conditions_other_run" if success else "missing",
        "selected_episode_ids": [_episode_id(example), *([_episode_id(success)] if success else [])]}


def _predicate_present(row: Mapping[str, Any], predicate: str) -> bool:
    units = row["recoveries"]
    if predicate in CONDITIONS:
        return any(unit["failed_condition_steps"][predicate] > 0 for unit in units)
    if predicate == "dwell_interrupted":
        return any(unit["dwell_interruptions"] for unit in units)
    if predicate == "recovery_latched":
        return any(unit["recovery_control_step"] is not None for unit in units)
    if predicate == "window_censored":
        return any(unit["recovery_censored"] for unit in units)
    if predicate == "unexecuted_event":
        return row["event_conditions"]["executed_events"] < row["event_conditions"]["planned_events"]
    process = [unit["process"] for unit in units if unit["process"].get("status") == "available"]
    if predicate == "condition_violation_span":
        return any(any(unit["condition_violation_spans"].values()) for unit in units)
    if predicate == "all_conditions_stable_span":
        return any(item["stable_spans"] for item in process)
    if predicate == "post_latch_instability":
        return any(item["post_latch"]["unstable_steps"] for item in process)
    if predicate == "window_net_displacement":
        return any(item["displacement_status"] == "available" for item in process)
    return bool(process)


def build_push_lexicon(full: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Cards bind concrete definitions and observed positive/negative controls."""
    definitions = full.get("definitions", {})
    rows = full.get("transcripts", [])
    bounds = definitions.get("stable_bounds", {})
    measurements = {
        "planar_speed": ("pose.planar_speed_mps", "<=", bounds.get("planar_speed_max_mps"), "m/s"),
        "angular_speed": ("pose.angular_speed_radps", "<=", bounds.get("angular_speed_max_radps"), "rad/s"),
        "roll": ("abs(pose.roll_rad)", "<=", bounds.get("absolute_roll_pitch_max_rad"), "rad"),
        "pitch": ("abs(pose.pitch_rad)", "<=", bounds.get("absolute_roll_pitch_max_rad"), "rad"),
        "height": ("pose.height_m", ">=", bounds.get("height_min_m"), "m"),
    }
    specs = {name: {"operands": operand, "stable_comparison": comparison, "threshold": threshold,
        "units": units, "duration_steps": 1, "predicate": "violation of this stable bound"}
        for name, (operand, comparison, threshold, units) in measurements.items()}
    span_units = {"start_step": "control step", "end_step_inclusive": "control step",
        "start_after_anchor_s": "s", "end_after_anchor_s": "s", "observed_steps": "control steps"}
    sample_units = {"control_step": "control step", "height_m": "m", "roll_rad": "rad",
        "pitch_rad": "rad", "planar_speed_mps": "m/s", "angular_speed_radps": "rad/s",
        "xy_drift_m": "m", "velocity_world_xy_mps": "m/s",
        "reset_relative_position_world_xy_m": "m"}
    specs.update({
        "dwell_interrupted": {"operands": "conditions AND, preceding_dwell_steps", "units": {"count": "interruptions", "control_step": "control step", "after_anchor_steps": "control steps", "after_anchor_s": "s", "preceding_dwell_steps": "control steps", "failed_conditions": "condition names"}, "definition": "unstable step after positive dwell, before first latch"},
        "recovery_latched": {"operands": "conditions AND, event_verified, required_dwell_steps", "threshold": definitions.get("required_dwell_steps"), "units": "control steps", "definition": "first verified contiguous stable dwell in eligible window"},
        "condition_violation_span": {"operands": "conditions, control_step", "units": span_units, "definition": "maximal contiguous observed steps violating one bound; intervals may overlap"},
        "all_conditions_stable_span": {"operands": "conditions, control_step", "units": span_units, "definition": "maximal contiguous observed steps satisfying all five bounds"},
        "post_latch_instability": {"operands": "conditions, recovery_control_step, executed event starts", "units": {"unstable_steps": "control steps", "observed_through_step": "control step", "unstable_spans": span_units, "status": "category"}, "definition": "unstable observed steps after latch through next injection sample or final observation, inclusive"},
        "first_response": {"operands": "pose, event injection start", "units": sample_units, "definition": "first recorded step strictly after current injection start, distinct from force recovery-window first"},
        "window_speed_peak": {"operands": "pose.planar_speed_mps, verified window mask", "units": sample_units, "definition": "earliest tie for maximum speed in observed recovery window"},
        "window_last": {"operands": "pose, verified window mask", "units": sample_units, "definition": "last observed recovery-window sample, may be before planned window end"},
        "window_net_displacement": {"operands": "planar_motion.xy_position at anchor and window last", "definition": "world xy endpoint minus current anchor position; force excludes pulse motion", "units": "m"},
        "window_censored": {"operands": "recovery_control_step, coverage", "units": {"recovery_censored": "boolean", "recovery_control_step": "control step or missing", "coverage": "category"}, "definition": "no observed verified latch; full-window failure, truncation and missing anchor remain distinct"},
        "unexecuted_event": {"operands": "planned_events, executed_events", "units": {"planned_events": "events", "executed_events": "events", "predicate": "boolean"}, "definition": "planned count exceeds actual executed count; not zero stable time in an observed window"},
    })
    targets = {"planar_speed": "root translation exceeds its stable bound", "angular_speed": "root rotation exceeds its stable bound",
        "roll": "root lateral tilt exceeds its stable bound", "pitch": "root fore-aft tilt exceeds its stable bound",
        "height": "root height falls below its stable bound", "dwell_interrupted": "a partial stable dwell is interrupted",
        "recovery_latched": "a verified stable dwell was completed", "condition_violation_span": "recurring or sustained bound violations",
        "all_conditions_stable_span": "brief or sustained all-condition stability", "post_latch_instability": "instability recurs after a completed dwell",
        "first_response": "separate injection response from later recovery-window response", "window_speed_peak": "locate the largest observed recovery-window speed",
        "window_last": "distinguish window endpoint from whole-episode endpoint", "window_net_displacement": "locate event-relative root translation",
        "window_censored": "distinguish unlatched windows and observation gaps", "unexecuted_event": "distinguish scheduled events from actual injections"}
    exclusions = {"dwell_interrupted": "interruption count is not a per-episode dwell threshold",
        "recovery_latched": "a latch does not imply later stability or episode success",
        "condition_violation_span": "overlapping conditions are not independent failed episodes",
        "all_conditions_stable_span": "a stable interval need not satisfy the required dwell",
        "post_latch_instability": "later instability is not an explanation of its cause",
        "first_response": "force response after pulse start is not the recovery window after pulse end",
        "window_speed_peak": "window maximum is not a whole-episode maximum",
        "window_last": "last observation need not be the planned window end",
        "window_net_displacement": "net displacement is not path length, bodily immobility or reset drift",
        "window_censored": "missing observation is not a measured zero",
        "unexecuted_event": "an unexecuted event is not a measured unstable response"}
    cards = []
    domains = sorted({(row["source"], *_domain(row)) for row in rows})
    for name, measured in specs.items():
        statistics_rows = []
        for key in domains:
            matched = [row for row in rows if (row["source"], *_domain(row)) == key]
            fail = [row for row in matched if _failure(row)]
            success = [row for row in matched if row["terminal_label"] == "success_recovered"]
            statistics_rows.append({"domain": list(key),
                "failure_n": sum(_predicate_present(row, name) for row in fail), "failure_N": len(fail),
                "success_n": sum(_predicate_present(row, name) for row in success), "success_N": len(success)})
        distinguishing = any(item["failure_N"] and item["success_N"] and
            item["failure_n"] / item["failure_N"] > item["success_n"] / item["success_N"] for item in statistics_rows)
        counter = next((row for row in rows if row["terminal_label"] == "success_recovered" and _predicate_present(row, name)), None)
        kind = "predicate_also_observed_in_success"
        if counter is None:
            counter = next((row for row in rows if _failure(row) and not _predicate_present(row, name)), None)
            kind = "failure_without_this_predicate"
        cards.append({"id": name, "version": PUSH_PROCESS_RULE, "measurable_definition": measured,
            "body_role": "root", "coordinates": "world velocities/xy; height relative to environment origin; signed Euler roll/pitch",
            "window_semantics": {"rule": definitions.get("window_rule"), "anchor": definitions.get("anchor"),
                "dwell": definitions.get("dwell_scope"), "terminal_before": "confirmation step strictly less than final control step"},
            "applicable_profile": full.get("binding", {}).get("robot_profile_id"),
            "missing_field_fallback": "missing, never zero; numeric.v1 cannot supply event-relative displacement",
            "target_failure_shape": targets[name],
            "not_equivalent_to": ["task failure", "physical cause", "foot contact, stepping or ankle/hip strategy",
                exclusions.get(name, "a bound violation does not establish inability to recover")],
            "discrimination": statistics_rows,
            "difficult_counterexample": {"status": "observed" if counter else "missing_in_this_pool", "kind": kind,
                "episode_id": _episode_id(counter) if counter else None,
                "evidence_ref": f"episode:{_episode_id(counter)}" if counter else None},
            "template": f"Root {name} was observed in episode {{episode}}, steps {{start}}-{{end}}; scope {{n}}/{{N}}.",
            "status": "formal" if distinguishing else "formal_descriptive"})
    return cards


def _sentence(identity: str, kind: str, section: str, text: str, *, predicate: str,
              label: str, numerator: int = 0, denominator: int = 0,
              window: Mapping[str, Any] | None = None, refs: Sequence[str] = (), **extra: Any) -> dict[str, Any]:
    row = {"id": identity, "kind": kind, "section": section, "text": text, "subject": "root",
        "predicate": predicate, "rule_version": PUSH_PROCESS_RULE,
        "scope": {"label": label, "numerator": numerator, "denominator": denominator},
        "evidence_refs": list(refs) or ["derived:push-process-v3"], "required": True, "priority": 0}
    if window is not None:
        row["window"] = dict(window)
    row.update(extra)
    return row


def _post_latch_description(unit: Mapping[str, Any]) -> str:
    post = unit["process"].get("post_latch", {})
    latch = unit["recovery_control_step"]
    if latch is None or post.get("status") == "not_latched":
        return "No latch observed; post-latch interval unavailable."
    through = post.get("observed_through_step")
    if through is None or through <= latch:
        return "No post-latch observation."
    spans = post["unstable_spans"]
    limit = f"through {through} ({post['until']}, inclusive)"
    if not spans:
        return f"Post-latch instability: 0 observed steps {limit}."
    first, last = spans[0], spans[-1]
    cutoff = "; last interval reaches observation cutoff" if last["end_step_inclusive"] == through else ""
    return (f"Post-latch instability: {post['unstable_steps']} observed steps in {len(spans)} interval(s), "
            f"first {first['start_step']}-{first['end_step_inclusive']}, "
            f"last {last['start_step']}-{last['end_step_inclusive']}; {limit}{cutoff}.")


def _recovery_outline(row: Mapping[str, Any], unit: Mapping[str, Any]) -> str:
    """Keep each recorded event/sequence, including unexecuted planned slots."""
    identity = unit["identity"]
    anchor = unit["anchor_control_step"]
    if not unit["anchor_verified"] or not unit["observed_window_steps"]:
        executed = row["event_conditions"]["executed_events"]
        unexecuted = executed == 0 or (identity.startswith("event_")
            and int(identity.removeprefix("event_")) >= executed)
        state = "not executed" if unexecuted else "execution recorded; recovery window unavailable"
        return (f"{identity}: {state} ({unit['coverage']}); scheduled anchor {unit['scheduled_anchor_control_step']}, "
                f"actual anchor {anchor}, verified {unit['anchor_verified']}; "
                f"latch {unit['recovery_control_step'] if unit['recovery_control_step'] is not None else 'not observed'}. "
                + _post_latch_description(unit))
    process = unit["process"]
    start = process["window_first"]["control_step"]
    end = process["window_last"]["control_step"]
    return (f"{identity}: anchor {anchor}, observed window {start}-{end}; "
            f"all-five stable {unit['stable_window_steps']}/{unit['observed_window_steps']} steps, "
            f"maximum pre-latch dwell {unit['max_dwell_steps']} steps, "
            f"latch {unit['recovery_control_step'] if unit['recovery_control_step'] is not None else 'not observed'}. "
            + _post_latch_description(unit))


def _window_description(row: Mapping[str, Any], unit: Mapping[str, Any], *, include_outline: bool = True) -> str:
    if not unit["anchor_verified"] or not unit["observed_window_steps"]:
        return _recovery_outline(row, unit)
    process = unit["process"]
    parts = [_recovery_outline(row, unit) if include_outline else f"Detail {unit['identity']}:"]
    for name in CONDITIONS:
        spans = unit["condition_violation_spans"][name]
        if spans:
            longest = max(spans, key=lambda item: item["observed_steps"])
            parts.append(f"{_NAMES[name]} violated in {len(spans)} interval(s), first {spans[0]['start_step']}-{spans[0]['end_step_inclusive']}, longest {longest['start_step']}-{longest['end_step_inclusive']}.")
    stable = process["stable_spans"]
    if stable:
        longest = max(stable, key=lambda item: item["observed_steps"])
        parts.append(f"All-five stability had {len(stable)} interval(s), longest {longest['start_step']}-{longest['end_step_inclusive']}.")
    response = process["first_observed_response"]
    peak = process["window_speed_peak"]
    parts.append(f"First response step {response['control_step'] if response else 'missing'}; window speed first/peak/last {process['window_first']['planar_speed_mps']:.3g}/{peak['planar_speed_mps']:.3g}/{process['window_last']['planar_speed_mps']:.3g} m/s, peak step {peak['control_step']}.")
    if process["displacement_status"] == "available":
        parts.append(f"Window net xy displacement magnitude {process['window_net_displacement_m']:.3g} m from this anchor, not reset drift.")
    else:
        parts.append(f"Window displacement unavailable ({process['displacement_status']}).")
    return " ".join(parts)


def _parent_sentence(diff: Mapping[str, Any], total: int) -> dict[str, Any]:
    if diff.get("status") != "available":
        return _sentence("push.parent.missing", "S5", "parent_diff",
            f"Root parent comparison for {total} candidate episode(s) is unavailable ({diff.get('reason', 'not_recorded')}); no change or cause is inferred.",
            predicate="parent_comparison", label="candidate cohort", denominator=total, gap_type="parent_comparison_missing")
    left, right = diff["left"], diff["right"]
    text = (f"Root cohort outcome under {PUSH_PROCESS_RULE}: parent success {left['successes']}/{left['episodes']} -> candidate {right['successes']}/{right['episodes']}; "
        f"all-recovered then final-unstable {left['final_unstable']}/{left['episodes']} -> {right['final_unstable']}/{right['episodes']}. "
        f"Whole-cohort domain identity matched {diff['paired']}/{diff['paired'] + diff['missing']}, missing {diff['missing']}; no subgroup or episode pairing is asserted.")
    return _sentence("push.parent.comparison", "S4", "parent_diff", text, predicate="cohort_change",
        label="parent and candidate cohorts", numerator=right["successes"], denominator=right["episodes"],
        comparison={"left": left, "right": right, "paired": diff["paired"], "missing": diff["missing"],
                    "coverage_unit": "whole_cohort_domain", "rule_version": PUSH_PROCESS_RULE},
        refs=["parent:" + diff["parent_checkpoint_hash"], "candidate:" + diff["candidate_checkpoint_hash"]])


def build_push_process_sentences(full: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows, sentences = full.get("transcripts", []), []
    total = len(rows)
    if full.get("status") != "available":
        for section in ("results", "process", "counterexample", "visual", "gaps", "parent_diff"):
            sentences.append(_sentence(f"push.{section}.missing", "S5", section,
                f"Root {section} evidence is unavailable for the expected cohort ({full.get('reason', 'not_recorded')}); episode count is unobserved, not zero.",
                predicate="missing_evidence", label="expected cohort; episode count unobserved", gap_type="numeric_evidence_missing"))
        return sentences
    all_group = full["profile"]["all"]
    outcomes = ", ".join(f"{label} {count}/{total}" for label, count in sorted(all_group["terminal_counts"].items()))
    sentences.append(_sentence("push.results.all", "S0", "results",
        f"Root outcomes over whole episodes ({total}/{total} cohort episodes): {outcomes}. Results retain every terminal label; score is not a success count.",
        predicate="terminal_counts", label="whole cohort", numerator=total, denominator=total,
        window={"kind": "whole_episode"}, refs=["profile:all"]))
    definition = full["definitions"]
    bounds = definition["stable_bounds"]
    sentences.append(_sentence("push.results.rule", "S0", "results",
        f"Root stability in recovery windows for all {total} episodes requires all five bounds together (displayed thresholds): planar speed <= {bounds['planar_speed_max_mps']:.6g} m/s, angular speed <= {bounds['angular_speed_max_radps']:.6g} rad/s, "
        f"absolute roll and pitch each <= {bounds['absolute_roll_pitch_max_rad']:.6g} rad, height >= {bounds['height_min_m']:.6g} m. Verified dwell requires {definition['required_dwell_steps']} consecutive steps; each recovery window spans anchor+1 through anchor+{definition['window_steps']} inclusive.",
        predicate="stable_bounds", label="whole cohort", numerator=total, denominator=total,
        window={"kind": "whole_window"}, refs=["definitions:stability"]))
    for stratum, group in sorted(full["profile"]["by_stratum"].items()):
        n = group["episodes"]
        counts = ", ".join(f"{label} {count}/{n}" for label, count in sorted(group["terminal_counts"].items()))
        sentences.append(_sentence(f"push.results.{stratum}", "S0", "results",
            f"Root {stratum} group over whole episodes ({n}/{total} cohort): {counts}; executed/planned events {group['executed_events']}/{group['planned_events']}.",
            predicate="terminal_counts", label=stratum, numerator=n, denominator=total,
            window={"kind": "group_episodes"}, refs=[f"profile:stratum:{stratum}"]))
    selected = full["representative_selection"]
    lookup = {_episode_id(row): row for row in rows}
    example = lookup[selected["context_episode_id"]]
    units = example["recoveries"]
    observed = [unit for unit in units if unit["anchor_verified"] and unit["observed_window_steps"]]
    detail = next((unit for unit in observed if unit["recovery_control_step"] is None), observed[-1] if observed else None)
    pattern_text = "non-majority combination" if not selected["majority"] else "most frequent combination"
    tie_text = "; tied combinations resolved lexicographically" if selected["tied"] else ""
    outline = " ".join(_recovery_outline(example, unit) for unit in units) or "No recorded recovery units."
    description = _window_description(example, detail, include_outline=False) if detail else "No verified recovery window was observed."
    text = (f"Root episode {_episode_id(example)} has the condition combination found in {selected['pattern_n']}/{selected['group_n']} compatible episodes ({pattern_text}{tie_text}); "
        f"conditions confirmed strictly before final step: {', '.join(selected['pattern']) or 'none'}. "
        f"This episode's sequence only: {outline} {description} "
        f"Final observed step {example['final_control_step']}, outcome {example['terminal_label'] or 'censored'}, failed bounds {', '.join(example['final_failed_conditions']) or 'none'}. "
        f"Intervals can overlap; a state at the final step is not evidence that it preceded termination.")
    sentences.append(_sentence("push.process.example", "S1", "process", text, predicate="recovery_evolution",
        label=_episode_id(example), numerator=selected["pattern_n"], denominator=selected["group_n"],
        window={"start": example["final_control_step"] - example["valid_steps"] + 1,
                "end": example["final_control_step"], "terminal": example["final_control_step"] if example["coverage"] == "complete" else None},
        refs=[f"episode:{_episode_id(example)}"]))
    chosen_strata = set()
    for profile in full["profile"]["matched_failure_profiles"]:
        stratum = profile["conditions"]["stratum"]
        if stratum in chosen_strata:
            continue
        chosen_strata.add(stratum)
        matching = [row for row in rows if row["event_conditions"] == profile["conditions"]]
        failures = [row for row in matching if (row["terminal_label"] or "censored") == profile["failure_label"]]
        successes = [row for row in matching if row["terminal_label"] == "success_recovered"]
        failure_any = sum(_predicate_present(row, "condition_violation_span") for row in failures)
        success_any = sum(_predicate_present(row, "condition_violation_span") for row in successes)
        descriptive = not successes or failure_any / len(failures) <= success_any / len(successes)
        phrases = [f"{_NAMES[name]} {sum(_predicate_present(row, name) for row in failures)}/{len(failures)} vs {sum(_predicate_present(row, name) for row in successes)}/{len(successes)}" for name in CONDITIONS]
        sentences.append(_sentence(f"push.process.group.{stratum}", "S2", "process",
            f"Root {stratum}, {profile['conditions']['executed_events']} executed event(s), whole observed recovery windows: {profile['failure_label']} versus compatible successful episodes; "
            f"any violation {failure_any}/{len(failures)} versus {success_any}/{len(successes)} ({'not discriminating' if descriptive else 'different occurrence rates, not a causal attribution'}); "
            f"Per-condition occurrences: {'; '.join(phrases)}.",
            predicate="condition_violation_span", label=stratum, numerator=len(failures), denominator=len(matching),
            window={"kind": "group_episodes"}, descriptive=descriptive,
            comparison={"status": "available" if successes else "missing", "matching": "same_event_conditions",
                "discriminating": not descriptive,
                "left": {"label": profile["failure_label"], "numerator": failure_any, "denominator": len(failures)},
                "right": {"label": "success_recovered", "numerator": success_any, "denominator": len(successes)}},
            refs=[f"profile:matched:{stratum}:{profile['failure_label']}"]))
    if not chosen_strata:
        sentences.append(_sentence("push.process.no_failure_comparison", "S5", "process",
            f"Root cohort has 0/{total} failed episodes over whole observed episodes; a failure-versus-success process contrast is unavailable, and successful observations are not a failure explanation.",
            predicate="failure_comparison", label="whole cohort", denominator=total, gap_type="no_failure_group"))
    unrecovered = [row for row in rows if row["terminal_label"] == "not_recovered"]
    recovered = [row for row in rows if row["terminal_label"] == "success_recovered"]
    pathways = []
    if unrecovered:
        latched_unstable = sum(row["all_required_recoveries_latched"] and not row["final_stable"] for row in unrecovered)
        stable_unlatched = sum(row["final_stable"] and not row["all_required_recoveries_latched"] for row in unrecovered)
        pathways.append(f"not_recovered: {latched_unstable}/{len(unrecovered)} all required recoveries latched, then final-unstable; "
            f"{stable_unlatched}/{len(unrecovered)} final-stable with at least one required latch absent.")
    if recovered:
        later_unstable = sum(row["unstable_steps_after_all_recoveries"] > 0 for row in recovered)
        pathways.append(f"success_recovered: {later_unstable}/{len(recovered)} unstable after all latches, then final-stable.")
    if pathways:
        sentences.append(_sentence("push.process.outcome_pathways", "S0", "process",
            "Root recovery histories. " + " ".join(pathways),
            predicate="recovery_and_final_state", label="recorded outcome groups", numerator=len(unrecovered) + len(recovered),
            denominator=total, window={"kind": "cohort_episodes"}))
    success = lookup.get(selected.get("success_episode_id"))
    if success:
        unit_text = " ".join(f"{unit['identity']} anchor {unit['anchor_control_step']}, latch {unit['recovery_control_step']}, stable {unit['stable_window_steps']}/{unit['observed_window_steps']} steps" for unit in success["recoveries"])
        sentences.append(_sentence("push.counterexample.episode", "S1", "counterexample",
            f"Root compatible success episode {_episode_id(success)} over steps 1-{success['final_control_step']} ({selected['success_matching']}): {unit_text}. "
            f"After all recoveries, {success['unstable_steps_after_all_recoveries']} unstable observed steps; final failed bounds {', '.join(success['final_failed_conditions']) or 'none'}. This successful example does not imply every recovery-window step was stable.",
            predicate="recovery_evolution", label=_episode_id(success), numerator=1, denominator=len(rows),
            window={"start": 1, "end": success["final_control_step"]}, refs=[f"episode:{_episode_id(success)}"]))
    else:
        sentences.append(_sentence("push.counterexample.missing", "S5", "counterexample",
            f"Root compatible success example is missing for the selected group of {selected['group_n']} episodes; another event-count group is not substituted.",
            predicate="success_comparison", label="selected compatible group", denominator=selected["group_n"], gap_type="matching_success_missing"))
    sentences.append(_sentence("push.visual.missing", "S5", "visual",
        f"Root visual supplementation: 0/{total} episodes have an attached checked observer statement in this numerical report; visible foot placement and body shape remain unassessed.",
        predicate="visual_coverage", label="whole cohort", denominator=total, gap_type="observer_not_attached"))
    sampling = full["coverage_sampling"]
    sampled = set(sampling["selected_episode_ids"])
    residual = full["residual"]
    for index, domain in enumerate(residual["domains"]):
        key = tuple(domain["domain"])
        uncertain = [item["episode_id"] for item in residual["episodes"] if tuple(item["domain"]) == key and item["state"] in {"negative", "missing"}]
        hit = len(set(uncertain) & sampled)
        sentences.append(_sentence(f"push.gaps.domain.{index}", "S5", "gaps",
            f"Root {_domain_label(key)}: among {domain['failures']} failed episodes, {domain['negative']} rule-negative and {domain['missing']} comparison-missing; "
            f"numerical windows sample {hit}/{len(uncertain)} residual/missing episodes, {len(uncertain)-hit} unexamined.",
            predicate="rule_residual", label=_domain_label(key), numerator=domain["negative"], denominator=domain["failures"],
            gap_type="rule_negative_or_missing_control", refs=[f"residual:domain:{index}"]))
    coverage_text = "; ".join(f"{item['stratum']} {item['outcome']} {item['sampled']}/{item['episodes']}" for item in sampling["by_stratum_outcome"])
    fractions = (sampling["sampled_time_fraction_median"], sampling["sampled_time_fraction_minimum"])
    rule = residual["rule"]
    early, late = rule["first_violation_bins_seconds"]
    sentences.append(_sentence("push.gaps.coverage", "S5", "gaps",
        f"Root numerical sampling covers {len(sampled)}/{total} episodes: {coverage_text}; sampled-episode time coverage median/minimum {fractions[0]:.3f}/{fractions[1]:.3f}. "
        f"Three dispersed anchors expand by 25 steps each side; overlaps counted once. Numeric channels lack foot/contact/yaw. Visual pose is separate from measured contact or control strategy. "
        "Conditions overlap; matching preserves event counts and nominal intensity, not direction or causality. "
        f"Residual signature: violated bounds and first-onset bins (<= {early:g}, ({early:g},{late:g}], > {late:g} s after anchor). Compare same stratum, executed-event count and recovery ordinal. "
        f"Window: comparison-missing if unverified/unobserved or < {rule['minimum_success_windows']} successful controls; otherwise rule-negative if >= {rule['negative_success_fraction']:.0%} share its exact signature, positive otherwise. "
        "Episode: any positive wins; else any missing/no actual window is missing, else negative. "
        "Rule-negative is a rule label, not no behavior; missing need not mean lost raw data. "
        f"Example: {example['event_conditions']['executed_events']}/{example['event_conditions']['planned_events']} executed/planned; all {len(units)} recovery units outlined, "
        f"{int(detail is not None)} window detailed; others archived.",
        predicate="observation_coverage", label="whole cohort", numerator=len(sampled), denominator=total,
        gap_type="sampling_and_unmeasured_body_roles", refs=["coverage:all", "derived:push-process-v3", f"episode:{_episode_id(example)}"]))
    sentences.append(_parent_sentence(full.get("parent_diff", {}), total))
    return sentences


def build_push_process_behavior(seed_summaries: Sequence[Mapping[str, Any]] | None = None, *,
                                context: Mapping[str, Any] | None = None,
                                legacy_full: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """New derivation, explicitly separate from persisted historical v2 objects."""
    from true_rl_agent.tasks.push_recovery.behavior import build_push_behavior
    if legacy_full is None:
        if context is None or seed_summaries is None:
            raise ValueError("Push process build requires archives and context")
        legacy_full = build_push_behavior(seed_summaries, context=context)
    if legacy_full.get("schema_version") != "true_rl_agent.push_recovery.behavior.v2":
        raise ValueError("Push v3 requires a full v2 numerical derivation")
    if legacy_full.get("status") == "available" and "transcripts" not in legacy_full:
        raise ValueError("Push v3 cannot reconstruct full processes from a bounded view")
    full = deepcopy(dict(legacy_full))
    full.update(schema_version=PUSH_PROCESS_SCHEMA, rule_version=PUSH_PROCESS_RULE,
        numerical_rule_version="push_process_v2", source_derivation_hash=canonical_json_hash(legacy_full))
    actual_runs = {}
    if seed_summaries is not None:
        from true_rl_agent.runner.numeric_evidence import load_numeric_evidence_manifest
        for summary in seed_summaries:
            reference = summary.get("numeric_evidence_ref")
            if reference is not None:
                manifest = load_numeric_evidence_manifest(reference)
                actual_runs[reference["manifest_sha256"]] = manifest["identity"]["run_index"]
    full["source_bindings"] = [{"source_index": index, "logical_run": index + 1,
        "manifest_sha256": source["manifest_sha256"],
        "actual_run_index": actual_runs.get(source["manifest_sha256"]),
        "actual_run_status": "verified_archive_identity" if source["manifest_sha256"] in actual_runs else "unknown_in_legacy_full"}
        for index, source in enumerate(full.get("sources", []))]
    full["process_contract"] = {"schema_version": "true_rl_agent.push_process_contract.v1",
        "residual": deepcopy(RESIDUAL_RULE), "coverage": deepcopy(COVERAGE_RULE),
        "representative_pattern": list(CONDITIONS), "confirmation": "violation span start strictly before final step",
        "episode_alias": "r{logical_run}.e{env}; logical_run is the one-based run-index-sorted source ordinal, not actual run identity",
        "report_max_bytes": PUSH_PROCESS_MAX_BYTES, "source_level": "L0/L1 deterministic derived L2"}
    full["lexicon"] = build_push_lexicon(full)
    full["residual"] = build_push_residual(full)
    full["coverage_sampling"] = sample_push_coverage(full, full["residual"])
    full["representative_selection"] = select_push_process_representatives(full)
    full["sentences"] = build_push_process_sentences(full)
    full["evidence_catalog"] = _evidence_catalog(full)
    full["actionability"]["footstep_or_support_change"] = {
        "status": "requires_new_observation", "reason": "no foot/contact/yaw evidence in this profile"}
    return full


def _evidence_catalog(full: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve report/window ids within the sealed full artifact, without paths."""
    catalog: dict[str, Any] = {"derived:push-process-v3": {"pointer": ""},
        "profile:all": {"pointer": "/profile/all"}, "coverage:all": {"pointer": "/coverage_sampling"},
        "definitions:stability": {"pointer": "/definitions"}}
    for index, row in enumerate(full.get("transcripts", [])):
        reference = f"episode:{_episode_id(row)}"
        binding = full["source_bindings"][row["source"]]
        catalog[reference] = {"pointer": f"/transcripts/{index}", "source_binding": binding,
            "episode_index": row["episode_index"], "env_id": row["env_id"]}
        for unit_index, unit in enumerate(row["recoveries"]):
            catalog[f"{reference}/window:{unit['identity']}"] = {
                "pointer": f"/transcripts/{index}/recoveries/{unit_index}", "source_binding": binding}
    for stratum in full.get("profile", {}).get("by_stratum", {}):
        catalog[f"profile:stratum:{stratum}"] = {"pointer": f"/profile/by_stratum/{stratum}"}
    for index, profile in enumerate(full.get("profile", {}).get("matched_failure_profiles", [])):
        reference = f"profile:matched:{profile['conditions']['stratum']}:{profile['failure_label']}"
        # The report chooses the first (largest) matched failure profile in each
        # stratum; preserve that same stable first entry if labels repeat.
        catalog.setdefault(reference, {"pointer": f"/profile/matched_failure_profiles/{index}"})
    for index in range(len(full["residual"]["domains"])):
        catalog[f"residual:domain:{index}"] = {"pointer": f"/residual/domains/{index}"}
    for index, window in enumerate(full["coverage_sampling"]["windows"]):
        catalog[window["evidence_ref"]] = {"pointer": f"/coverage_sampling/windows/{index}",
            "episode_ref": "episode:" + window["episode_id"]}
    return catalog


def _basis(behavior: Mapping[str, Any]) -> Mapping[str, Any]:
    return behavior["comparison_basis"] if "comparison_basis" in behavior else {
        "counts": {key: behavior["profile"]["all"][key] for key in _COUNT_FIELDS},
        "terminal_counts": behavior["profile"]["all"]["terminal_counts"]}


def compare_push_process_behavior(candidate: Mapping[str, Any] | None, parent: Mapping[str, Any] | None,
                                  *, expected_parent_checkpoint_hash: str | None) -> dict[str, Any]:
    if not candidate or not parent or candidate.get("status") != "available" or parent.get("status") != "available":
        return {"status": "missing", "reason": "candidate_or_bound_parent_behavior_missing"}
    schemas = {PUSH_PROCESS_SCHEMA, PUSH_PROCESS_PUBLIC_SCHEMA}
    if (candidate.get("schema_version") not in schemas or parent.get("schema_version") not in schemas or
            candidate.get("rule_version") != parent.get("rule_version")):
        return {"status": "not_comparable", "reason": "behavior_semantics_version_mismatch"}
    if not expected_parent_checkpoint_hash or parent["binding"].get("checkpoint_hash") != expected_parent_checkpoint_hash:
        return {"status": "not_comparable", "reason": "actual_generation_parent_checkpoint_mismatch"}
    if not candidate["comparison_domain"].get("execution_source_hash") or not parent["comparison_domain"].get("execution_source_hash"):
        return {"status": "not_comparable", "reason": "execution_source_unavailable"}
    if candidate["comparison_domain"] != parent["comparison_domain"] or not candidate["comparison_domain"].get("execution_environment_hash"):
        return {"status": "not_comparable", "reason": "cell_protocol_cohort_execution_or_input_domain_mismatch"}
    if any(item["coverage"]["complete_episodes"] != item["coverage"]["episodes"] for item in (candidate, parent)):
        return {"status": "not_comparable", "reason": "incomplete_episode_coverage"}
    def counts(value: Mapping[str, Any]) -> dict[str, Any]:
        basis = _basis(value)
        return {"label": "whole cohort", "numerator": basis["terminal_counts"].get("success_recovered", 0),
            "denominator": basis["counts"]["episodes"], "episodes": basis["counts"]["episodes"], "successes": basis["terminal_counts"].get("success_recovered", 0),
            "final_unstable": basis["counts"]["all_recovered_then_final_unstable"], "rule_version": PUSH_PROCESS_RULE}
    return {"status": "available", "parent_checkpoint_hash": expected_parent_checkpoint_hash,
        "candidate_checkpoint_hash": candidate["binding"]["checkpoint_hash"],
        "left": counts(parent), "right": counts(candidate), "paired": 1, "missing": 0,
        "scope": "cohort_domain_not_episode_pairing", "rule_version": PUSH_PROCESS_RULE}


def project_push_process_behavior(behavior: Mapping[str, Any], *, max_bytes: int = PUSH_PROCESS_MAX_BYTES) -> dict[str, Any]:
    from true_rl_agent.evaluation.process_report import fit_process_report_capacity, render_process_report, validate_sentence
    from true_rl_agent.evaluation.visual_observer import update_visual_projection_count
    if behavior.get("schema_version") == PUSH_PROCESS_PUBLIC_SCHEMA:
        view = deepcopy(dict(behavior))
        # Parent comparison is attached by the trusted generation projection
        # after the phase is sealed. Re-render just that deterministic section.
        parent = _parent_sentence(view.get("parent_diff", {}), view.get("coverage", {}).get("episodes", 0))
        validate_sentence(parent)
        view["report"]["parent_diff"] = [{"id": parent["id"], "text": parent["text"]}]
        return fit_process_report_capacity(view, max_bytes=max_bytes,
            update_projection=update_visual_projection_count)
    if behavior.get("schema_version") != PUSH_PROCESS_SCHEMA:
        raise ValueError("unknown Push process report schema")
    fields = ("status", "reason", "rule_version", "binding", "comparison_domain", "coverage", "sources", "parent_diff")
    envelope = {key: deepcopy(behavior[key]) for key in fields if key in behavior}
    if "comparison_domain" in envelope:
        envelope["binding"] = {key: value for key, value in envelope["binding"].items() if key not in envelope["comparison_domain"]}
    envelope.update(schema_version=PUSH_PROCESS_PUBLIC_SCHEMA,
        full_derivation_sha256=canonical_json_hash(behavior),
        process_contract_hash=canonical_json_hash(behavior["process_contract"]))
    if behavior.get("status") == "available":
        envelope["comparison_basis"] = deepcopy(_basis(behavior))
    sentences = [deepcopy(row) for row in behavior["sentences"] if row["section"] != "parent_diff"]
    sentences.append(_parent_sentence(behavior.get("parent_diff", {}), behavior.get("coverage", {}).get("episodes", 0)))
    return render_process_report(envelope=envelope, sentences=sentences,
        actionability=deepcopy(behavior["actionability"]), max_bytes=max_bytes,
        update_projection=update_visual_projection_count)


__all__ = ["build_push_process_behavior", "project_push_process_behavior", "compare_push_process_behavior",
           "build_push_residual", "sample_push_coverage", "select_push_process_representatives", "build_push_lexicon"]
