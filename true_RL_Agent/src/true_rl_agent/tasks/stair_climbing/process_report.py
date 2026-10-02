"""Versioned Stair process language, residuals and numeric coverage selection.

This layer consumes complete measured episodes. It neither edits outcome rules
nor infers contact surfaces, control strategies or causes. Old behavior views
remain valid historical artifacts and are never silently upgraded.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from statistics import median
from typing import Any, Mapping

from true_rl_agent.evaluation.binding import canonical_json_hash

FULL_SCHEMA = "true_rl_agent.stair.behavior.full.v3"
PUBLIC_SCHEMA = "true_rl_agent.stair.behavior.public.v4"
DERIVATION_VERSION = "stair.process_report.v2"
LEXICON_VERSION = "stair.process_terms.v1"
MAX_PUBLIC_BYTES = 8192
PATTERN_TERMS = ("tilt", "low_progress", "unloaded_low_motion", "up_z_negative")
SAMPLING_SEED = "p3new-20260929"
CONTRACT = {
    "version": DERIVATION_VERSION, "lexicon_version": LEXICON_VERSION,
    "pattern_terms": list(PATTERN_TERMS), "confirmation": "strictly_before_terminal",
    "relative_terminal_bins_s": [.5, 2.0], "minimum_success_controls": 3,
    "signature_success_fraction": .5, "missing_controls_precede_empty_signature": True,
    "sampling_seed": SAMPLING_SEED, "anchors_per_episode": 3,
    "anchor_half_width_steps": 25, "supplemental_episode_cap": 4,
    "selection_scope": "numeric windows; no claim that images were observed",
    "progress_outline": "at most two largest root-x excursions, retaining any terminal-unreturned excursion; chronological display",
}
_LABELS = {"tilt": "root tilt", "low_progress": "low root forward progress",
    "unloaded_low_motion": "same-foot unloaded low motion", "up_z_negative": "root up_z<0",
    "loaded_point_motion": "loaded support-point motion", "low_clearance": "low support-point clearance",
    "max_x_retreat": "root x retreat", "forbidden_contact": "task forbidden-contact termination"}
_EVENTS = {"loaded_point_motion": "slip_any", "low_clearance": "low_clearance_any"}
_SECTIONS = ("results", "process", "counterexample", "visual", "gaps", "parent_diff", "actionability")


def _size(value):
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8"))


def _cell_label(cell):
    # A readable cell name, not an indexed code requiring a lookup table.
    return cell.removeprefix("public_dev.").replace("_level_", " L")


def _hash_order(episode):
    return hashlib.sha256(f"{SAMPLING_SEED}|stair|{episode['episode_id']}".encode()).hexdigest()


def _first_confirmation(episode, term):
    if term == "up_z_negative":
        step = episode.get("root_motion", {}).get("first_up_z_negative_step")
        return step if step is not None and step < episode["steps"] else None
    code = _EVENTS.get(term, term)
    values = [event["confirmed"] for event in episode["events"] if event["code"] == code
              and event.get("confirmed") is not None and event["confirmed"] < episode["steps"]]
    return min(values, default=None)


def stair_process_signature(episode):
    """Frozen four-term signature; terminal-only events never enter it."""
    result = []
    for term in PATTERN_TERMS:
        step = _first_confirmation(episode, term)
        if step is not None:
            seconds = (episode["steps"] - step) * episode["dt_s"]
            time_bin = "le_0.5s" if seconds <= .5 + 1e-12 else "le_2s" if seconds <= 2 + 1e-12 else "gt_2s"
            result.append((term, time_bin))
    return tuple(result)


def _groups(episodes):
    successes, failures = defaultdict(list), defaultdict(list)
    for episode in episodes:
        key = (episode["cell"], episode["run"])
        if episode["primary_reason"] == "success":
            successes[key].append(episode)
        else:
            failures[(episode["primary_reason"], *key)].append(episode)
    return failures, successes


def build_stair_residuals(full):
    """Every failure has exactly one state, including insufficient controls."""
    failures, successes = _groups(full["episodes"])
    domains, states = [], []
    for (reason, cell, run), failed in sorted(failures.items()):
        controls = successes.get((cell, run), [])
        control_signatures = Counter(stair_process_signature(e) for e in controls)
        counts = Counter()
        members = []
        for episode in sorted(failed, key=lambda e: e["episode_id"]):
            signature = stair_process_signature(episode)
            matching = control_signatures[signature]
            if len(controls) < CONTRACT["minimum_success_controls"]:
                status = "comparison_missing"
            elif not signature or matching / len(controls) >= CONTRACT["signature_success_fraction"]:
                status = "rule_negative"
            else:
                status = "rule_positive"
            counts[status] += 1
            row = {"episode_id": episode["episode_id"], "cell": cell, "run": run,
                   "reason": reason, "status": status, "signature": [list(x) for x in signature],
                   "matching_success_n": matching, "success_n": len(controls)}
            states.append(row)
            members.append(episode["episode_id"])
        domains.append({"reason": reason, "cell": cell, "run": run,
            "stratum": failed[0]["stratum"], "failure_n": len(failed), "success_n": len(controls),
            **{name: counts[name] for name in ("rule_positive", "rule_negative", "comparison_missing")},
            "episode_ids": members})
    total = Counter(row["status"] for row in states)
    return {"lexicon_version": LEXICON_VERSION, "rule_version": full["domain"]["rule_version"],
            "domain_definition": ["failure_reason", "cell", "run"], "domains": domains,
            "episodes": states, "failure_n": len(states),
            **{name: total[name] for name in ("rule_positive", "rule_negative", "comparison_missing")}}


def _union(windows):
    merged = []
    for start, end in sorted(windows):
        if merged and start <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def select_stair_observation_windows(full, residuals):
    """Numeric sampling catalog; media selection/visibility is a separate fact."""
    episodes = {e["episode_id"]: e for e in full["episodes"]}
    groups = defaultdict(list)
    for episode in episodes.values():
        groups[(episode["stratum"], "success" if episode["primary_reason"] == "success" else "failure")].append(episode)
    sources = defaultdict(set)
    for _, group in sorted(groups.items()):
        selected = min(group, key=_hash_order)
        sources[selected["episode_id"]].add("rule_independent_baseline")
        if selected["primary_reason"] == "success":
            sources[selected["episode_id"]].add("success_control")
    eligible = [episodes[row["episode_id"]] for row in residuals["episodes"]
                if row["status"] in {"rule_negative", "comparison_missing"}]
    supplemental = sorted(eligible, key=_hash_order)[:CONTRACT["supplemental_episode_cap"]]
    for episode in supplemental:
        sources[episode["episode_id"]].add("residual_or_missing_supplement")
    windows, coverage = [], []
    for episode_id in sorted(sources):
        episode = episodes[episode_id]
        last = episode["steps"]
        m, w = CONTRACT["anchors_per_episode"], CONTRACT["anchor_half_width_steps"]
        anchors = sorted({1 + (last - 1) * i // (m - 1) for i in range(m)})
        spans = _union([(max(1, step - w), min(last, step + w)) for step in anchors])
        covered = sum(end - start + 1 for start, end in spans)
        for start, end in spans:
            windows.append({"episode_id": episode_id, "cell": episode["cell"], "run": episode["run"],
                "start": start, "end": end, "sources": sorted(sources[episode_id]),
                "checkpoint_hash": full["domain"]["checkpoint_hash"],
                "archive_hash": full["domain"]["archive_hash"], "media_observation_status": "not_observed"})
        coverage.append({"episode_id": episode_id, "covered_steps": covered, "total_steps": last,
                         "fraction": covered / last, "union": spans})
    group_coverage = [{"stratum": key[0], "outcome": key[1], "selected_n": sum(e["episode_id"] in sources for e in group),
                       "episode_n": len(group)} for key, group in sorted(groups.items())]
    ratios = [row["fraction"] for row in coverage]
    return {"contract": deepcopy(CONTRACT), "windows": windows, "episode_coverage": group_coverage,
        "time_coverage": coverage, "time_coverage_median": median(ratios) if ratios else None,
        "time_coverage_minimum": min(ratios) if ratios else None,
        "selected_episode_ids": sorted(sources), "selected_n": len(sources), "episode_n": len(episodes),
        "residual_eligible_n": len(eligible), "supplemental_selected_n": len(supplemental),
        "supplemental_episode_ids": [e["episode_id"] for e in supplemental],
        "residual_selected_n": sum(e["episode_id"] in sources for e in eligible),
        "residual_unselected_episode_ids": sorted(e["episode_id"] for e in eligible if e["episode_id"] not in sources),
        "unselected_episode_ids": sorted(set(episodes) - sources.keys()),
        "visual_observed_episode_n": 0, "scope": "numeric sampling only; not visual coverage"}


def _term_present(episode, term):
    if term == "max_x_retreat":
        return episode["root_motion"]["max_x_retreat"][2] > 0
    if term == "forbidden_contact":
        return episode["primary_reason"] == "forbidden_contact"
    return _first_confirmation(episode, term) is not None


def build_stair_term_cards(development_full):
    """Measured development statistics are inputs, never embedded experiment facts."""
    thresholds = development_full["semantics"]["thresholds"]
    failures, successes = _groups(development_full["episodes"])
    definitions = {
        "tilt": {"operands": ["base_quat_w.wxyz"], "threshold": thresholds.get("tilt_rad"),
                 "consecutive_steps": thresholds.get("tilt_steps"), "unit": "rad", "coordinate": "signed Euler roll/pitch from world quaternion",
                 "definition": "max(abs(roll),abs(pitch)) exceeds threshold; signed extrema retained; Euler values wrap"},
        "low_progress": {"operands": ["base_pos_w.x", "terrain_env_origins_w.x"], "threshold": .02,
            "window_s": .5, "unit": "m", "coordinate": "terrain-relative world x",
            "definition": "maximum x in complete 0.5s window minus initial x is below 0.02m"},
        "unloaded_low_motion": {"operands": ["support_contact_forces_w", "support_lin_vel_w"],
            "force_threshold_N": thresholds.get("contact_N"), "speed_threshold_mps": .05,
            "duration_s": .5, "unit": "m/s", "coordinate": "same bound support reference point, world 3D speed",
            "definition": "same support force norm <= contact threshold AND 3D speed <= 0.05m/s for ceil(0.5/dt) samples"},
        "up_z_negative": {"operands": ["base_quat_w.qx", "base_quat_w.qy"], "threshold": 0,
            "consecutive_steps": 1, "unit": "dimensionless", "coordinate": "body z axis component along world z",
            "definition": "1-2*(qx*qx+qy*qy) < 0"},
        "loaded_point_motion": {"operands": ["support_contact_forces_w", "support_lin_vel_w"],
            "force_threshold_N": thresholds.get("contact_N"), "speed_threshold_mps": thresholds.get("slip_mps"),
            "consecutive_steps": thresholds.get("slip_steps"), "unit": "m/s", "coordinate": "world planar reference-point velocity",
            "definition": "any instant loaded point above speed threshold; global streak can alternate bodies"},
        "low_clearance": {"operands": ["support_pos_w", "support_contact_forces_w", "resolved_cell_manifest"],
            "threshold_m": thresholds.get("clearance_m"), "consecutive_steps": thresholds.get("clearance_steps"),
            "unit": "m", "coordinate": "bound support point relative to task C1/C2 surface",
            "definition": "any instant unloaded point below cell-specific clearance threshold"},
        "max_x_retreat": {"operands": ["base_pos_w.x", "terrain_env_origins_w.x"], "threshold": 0,
            "unit": "m", "coordinate": "terrain-relative world x", "duration": "whole episode including terminal",
            "definition": "maximum prior running x peak minus a later x, earliest ties"},
        "forbidden_contact": {"operands": ["termination_masks", "forbidden_contact_forces_w_history"],
            "threshold_N": thresholds.get("forbidden_N"), "unit": "N", "coordinate": "bound forbidden-role bodies, world force",
            "definition": "recorded task forbidden-contact primary termination; manager remains authority"},
    }
    negatives = {
        "tilt": ["not accumulated rotation", "not failure cause"],
        "low_progress": ["not whole-body stillness", "not a collision"],
        "unloaded_low_motion": ["not physical separation", "not a frozen controller"],
        "up_z_negative": ["not back touching ground", "not a termination cause"],
        "loaded_point_motion": ["not contact-surface slip", "point is not toe or sole centre"],
        "low_clearance": ["not toe collision", "base layer is not contact surface"],
        "max_x_retreat": ["not a typical trajectory", "not net episode displacement"],
        "forbidden_contact": ["not an identified contact surface", "not a causal explanation"],
    }
    cards = []
    for term, definition in definitions.items():
        stats = []
        for (reason, cell, run), failed in sorted(failures.items()):
            control = successes.get((cell, run), [])
            stats.append({"reason": reason, "cell": cell, "run": run,
                          "failed_n": len(failed), "failed_present": sum(_term_present(e, term) for e in failed),
                          "success_n": len(control), "success_present": sum(_term_present(e, term) for e in control)})
        positive_controls = [e for e in development_full["episodes"] if e["primary_reason"] == "success" and _term_present(e, term)]
        counterexample = min(positive_controls, key=lambda e: (e["run"], e["slot"]), default=None)
        descriptive = term in {"loaded_point_motion", "max_x_retreat", "forbidden_contact"}
        cards.append({"id": term, "version": LEXICON_VERSION, "measurement_rule_version": development_full["domain"]["rule_version"],
            "definition": definition, "body_role": "support" if term in {"unloaded_low_motion", "loaded_point_motion", "low_clearance"} else "forbidden_role" if term == "forbidden_contact" else "root",
            "time_semantics": "presence confirmed strictly before terminal except explicitly terminal/whole-episode measurements; end at terminal is cutoff",
            "applicable_profiles": ["stair.unified_first_episode.v1", "stair.unified_first_episode.v2"],
            "missing_field_degradation": "missing required operands: unavailable, never infer zero; v1 history is unavailable; no surface identity",
            "failure_form": _LABELS[term], "does_not_mean": negatives[term], "development_discrimination": stats,
            "development_archive_hash": development_full["domain"]["archive_hash"],
            "hard_counterexample": ({"kind": "observed_success", "episode_id": counterexample["episode_id"],
                                     "cell": counterexample["cell"], "run": counterexample["run"]} if counterexample else
                                    {"kind": "specified_boundary_case", "description": {
                                        "up_z_negative": "first negative up_z occurs exactly at terminal: whole-episode measurement present, before-terminal signature absent",
                                        "forbidden_contact": "forbidden predicate and success same frame: recorded primary reason, not predicate alone, defines termination label",
                                        "unloaded_low_motion": "two different feet each slow for half the duration cannot form one same-foot confirmation",
                                    }.get(term, "condition first confirms at terminal: must not count as before-terminal")}),
            "template": f"{{episode}}: {_LABELS[term]} during steps {{start}}-{{end}}, confirmed {{confirmed}}; terminal {{terminal}}.",
            "status": "formal_descriptive" if descriptive else "formal"})
    return cards


def record_stair_candidate_term(ledger, *, episode_id, start, end, observer_version,
                                original_name, normalized_name=None, merge_record=None):
    """Append an observer proposal without changing any detector or term card."""
    if (not isinstance(ledger, Mapping) or set(ledger) != {"schema_version", "entries", "automatic_promotion"}
            or ledger["schema_version"] != "stair.candidate_terms.v1" or ledger["automatic_promotion"] is not False):
        raise ValueError("Stair candidate term ledger mismatch")
    if (not all(isinstance(value, str) and value.strip() for value in (episode_id, observer_version, original_name))
            or type(start) is not int or type(end) is not int or not 1 <= start <= end):
        raise ValueError("Stair candidate term source mismatch")
    if normalized_name is not None and (not isinstance(normalized_name, str) or not normalized_name.strip()
                                         or not isinstance(merge_record, Mapping) or not merge_record):
        raise ValueError("Stair candidate normalization requires a preserved developer merge record")
    result = deepcopy(dict(ledger))
    name = normalized_name or original_name
    existing = next((row for row in result["entries"] if row["normalized_name"] == name), None)
    if existing is None:
        existing = {"status": "candidate", "normalized_name": name, "sources": [], "occurrence_count": 0,
                    "merge_records": [], "promotion": "developer versioning only after P7 qualification"}
        result["entries"].append(existing)
    source = {"episode_id": episode_id, "start": start, "end": end, "observer_version": observer_version,
              "original_name": original_name}
    if source not in existing["sources"]:
        existing["sources"].append(source)
        existing["occurrence_count"] += 1
    if merge_record is not None and dict(merge_record) not in existing["merge_records"]:
        existing["merge_records"].append(deepcopy(dict(merge_record)))
    return result


def _representative(group):
    patterns = defaultdict(list)
    for episode in group:
        patterns[tuple(_first_confirmation(episode, term) is not None for term in PATTERN_TERMS)].append(episode)
    count = max(map(len, patterns.values()))
    tied = sorted(key for key, values in patterns.items() if len(values) == count)
    key = tied[0]
    ordered = sorted(patterns[key], key=lambda e: (e["steps"], e["run"], e["slot"]))
    # Choose the lower observed median duration, then smallest identifier at it.
    duration = ordered[(len(ordered) - 1) // 2]["steps"]
    chosen = min((e for e in ordered if e["steps"] == duration), key=lambda e: (e["run"], e["slot"]))
    return chosen, {"terms": dict(zip(PATTERN_TERMS, key)), "pattern_n": count, "group_n": len(group),
                    "tie": len(tied) > 1, "majority": count > len(group) / 2}


def _sentence(identifier, kind, section, text, *, subject="root", predicate="process", scope=None,
              window=None, refs=None, **extra):
    scope = deepcopy(scope or {"label": "public-dev cohort"})
    if "denominator" in scope and "numerator" not in scope:
        scope["numerator"] = scope["denominator"]
    row = {"id": identifier, "kind": kind, "section": section, "text": text, "subject": subject,
           "predicate": predicate, "rule_version": LEXICON_VERSION, "scope": scope,
           "window": window or {"kind": "whole_episode"}, "evidence_refs": refs or ["semantic"],
           "required": True, "priority": 0}
    row.update(extra)
    return row


def _timeline(episode, selection):
    spans, refs = [], []
    # All repetitions remain in the full transcript. The short view identifies
    # one interval per term and explicitly reports earlier/repeated intervals.
    for term in ("low_progress", "tilt", "unloaded_low_motion"):
        matches = [(i, e) for i, e in enumerate(episode["events"]) if e["code"] == term and e.get("confirmed") is not None]
        if not matches:
            continue
        index, event = max(matches, key=lambda pair: (pair[1]["end"] == episode["steps"], pair[1]["end"]-pair[1]["start"], -pair[1]["start"], pair[1]["body"]))
        body_label = _LABELS[term] if event["body"] == "root" else f"{event['body']} unloaded low motion"
        spans.append((event["start"], event["end"], f"{body_label} {event['start']}-{event['end']} (confirmed {event['confirmed']}; {len(matches)} intervals confirmed)"))
        refs.append(f"episodes/{episode['episode_id']}/events/{index}")
    negative = episode["root_motion"]["first_up_z_negative_step"]
    if negative is not None:
        spans.append((negative, negative, f"root up_z first negative at {negative}"))
        refs.append(f"episodes/{episode['episode_id']}/root_motion/first_up_z_negative_step")
    spans.sort()
    text = "; ".join(row[2] for row in spans) if spans else "no four-term confirmation interval is available; movement outside these rules remains unclassified"
    overlap = any(max(a[0], b[0]) <= min(a[1], b[1]) for i, a in enumerate(spans) for b in spans[i+1:])
    qualifier = "most frequent combination" if selection["majority"] else "non-majority combination"
    return (f"{episode['episode_id']} ({_cell_label(episode['cell'])}, run {episode['run']}): {qualifier} {selection['pattern_n']}/{selection['group_n']} episodes"
            f"{'; tied patterns, lexicographic selection' if selection['tie'] else ''}. Steps 1-{episode['steps']}: {text}. "
            f"{'Intervals overlap; not a single sequence. ' if overlap else ''}Terminal {episode['steps']}: {episode['primary_reason']}.", refs)


def _progress_outline(episode, *, attitude=True):
    """Keep the change and its observed resolution in the same short account."""
    from .behavior import PROGRESS_PROCESS_VERSION
    process = episode.get("progress_process")
    if not process or process.get("version") != PROGRESS_PROCESS_VERSION:
        return None
    rows = process["excursions"]
    ranked = sorted(range(len(rows)), key=lambda i: (-rows[i]["drop_m"], rows[i]["peak_step"]))
    pending = next((i for i, row in enumerate(rows) if row["recovery_step"] is None), None)
    selected = ([pending] if pending is not None else [])
    selected += [i for i in ranked if i not in selected][:2-len(selected)]
    selected.sort(key=lambda i: rows[i]["peak_step"])
    refs = [f"episodes/{episode['episode_id']}/root_motion"]
    clauses = []
    for i in selected:
        row = rows[i]
        end = (f"re-attains prior x at {row['recovery_step']}" if row["recovery_step"] is not None
               else f"not re-attained through terminal {row['observed_end_step']}")
        clause = f"{row['peak_step']}->{row['trough_step']}: retreats {row['drop_m']:.6g} m; {end}"
        if attitude:
            roll = row["attitude_extrema"]["roll_rad"]
            pitch = row["attitude_extrema"]["pitch_rad"]
            clause += (f"; signed roll [{roll[0]:.3f},{roll[2]:.3f}], pitch [{pitch[0]:.3f},{pitch[2]:.3f}] rad "
                       f"over {row['peak_step']}-{row['observed_end_step']}")
        clauses.append(clause)
        refs.append(f"episodes/{episode['episode_id']}/progress_process/excursions/{i}")
    motion = episode["root_motion"]
    details = "; ".join(clauses) if clauses else "no positive excursion below the running x maximum"
    text = (f"{episode['episode_id']} root x {motion['first'][0]:.3f}->{motion['last'][0]:.3f} m over 1-{episode['steps']}. "
            f"Positive x-excursions shown {len(selected)}/{len(rows)}: {details}.")
    return text, refs


def _comparison_summary(full):
    grouped = defaultdict(list)
    for episode in full["episodes"]:
        grouped[episode["stratum"]].append(episode)
    keys = sorted({(e["cell"], e["run"]) for e in full["episodes"]})
    return {"episode_n": len(full["episodes"]), "cell_run_n": len(keys), "cell_run_keys_hash": canonical_json_hash(keys),
        "strata": {key: {"n": len(group), "success": sum(e["primary_reason"] == "success" for e in group),
                          **{term: sum(_first_confirmation(e, term) is not None for e in group) for term in PATTERN_TERMS}}
                   for key, group in sorted(grouped.items())}}


def _sentences(full, residuals, sampling):
    episodes = full["episodes"]
    failures, successes = _groups(episodes)
    summary = _comparison_summary(full)
    rows = []
    total_success = sum(item["success"] for item in summary["strata"].values())
    results = "; ".join(f"{name} {item['success']}/{item['n']}" for name, item in summary["strata"].items())
    progress_scope = (" X-excursions: largest two including any unreturned tail; counts include tiny motion, not attempts. "
        "X re-attainment proves neither balance recovery nor monotonic motion. Euler angles wrap; ranges are not rotation travelled.") if any("progress_process" in e for e in episodes) else ""
    rows.append(_sentence("results", "S2", "results", f"Successful episodes: {total_success}/{len(episodes)} overall; {results}. Counts are episodes, not scores or independent runs. Group process counts require confirmation strictly before terminal. Contrasts match cell/run. Intervals ending at terminal may continue unseen." + progress_scope,
        predicate="task_outcome", scope={"label": "public-dev cohort", "numerator": total_success, "denominator": len(episodes)},
        window={"kind": "group_episodes"}, comparison={"status": "missing", "reason": "cohort outcome, no external control"}, refs=["episodes/*/primary_reason"]))
    reason_counts = Counter(e["primary_reason"] for e in episodes if e["primary_reason"] != "success")
    ordered = sorted(failures, key=lambda key: (-reason_counts[key[0]], key[0], not bool(successes.get(key[1:])), -len(failures[key]), key[1], key[2]))
    protected = []
    for key in ordered:
        if not protected or not any(failures[existing][0]["stratum"] == failures[key][0]["stratum"] for existing in protected):
            protected.append(key)
    if ordered:
        primary = ordered[0]
        chosen, selection = _representative(failures[primary])
        text, refs = _timeline(chosen, selection)
        rows.append(_sentence("process-example", "S1", "process", text,
            scope={"label": chosen["episode_id"], "numerator": selection["pattern_n"], "denominator": selection["group_n"]},
            window={"start": 1, "end": chosen["steps"], "terminal": chosen["steps"]}, refs=refs or [f"episodes/{chosen['episode_id']}"]))
        motion = chosen["root_motion"]
        roll, pitch = motion["extrema"]["roll_rad"], motion["extrema"]["pitch_rad"]
        progress = _progress_outline(chosen)
        if progress is not None:
            text, refs = progress
            rows.append(_sentence("process-root-measurement", "S0", "process", text,
                predicate="root_progress_excursions", scope={"label": chosen["episode_id"]},
                window={"start": 1, "end": chosen["steps"], "terminal": chosen["steps"]}, refs=refs))
        elif max(abs(roll[0]), abs(roll[2]), abs(pitch[0]), abs(pitch[2])) > full["semantics"]["thresholds"]["tilt_rad"]:
            retreat = motion["max_x_retreat"]
            rows.append(_sentence("process-root-measurement", "S0", "process",
                f"{chosen['episode_id']} root, whole steps 1-{chosen['steps']} including terminal: signed roll [{roll[0]:.3f},{roll[2]:.3f}] rad, pitch [{pitch[0]:.3f},{pitch[2]:.3f}] rad; maximum x retreat {retreat[2]:.3f} m from step {retreat[0]} to {retreat[1]}. Euler angles wrap; these are not contact measurements.",
                predicate="signed_root_motion", scope={"label": chosen["episode_id"]},
                window={"start": 1, "end": chosen["steps"], "terminal": chosen["steps"]},
                refs=[f"episodes/{chosen['episode_id']}/root_motion"]))
        for ordinal, key in enumerate(protected):
            group, control = failures[key], successes.get(key[1:], [])
            term = max(PATTERN_TERMS, key=lambda term: (sum(_first_confirmation(e, term) is not None for e in group), -PATTERN_TERMS.index(term)))
            count = sum(_first_confirmation(e, term) is not None for e in group)
            success_count = sum(_first_confirmation(e, term) is not None for e in control)
            contrast = f"matched success {success_count}/{len(control)}" if control else "success comparison missing (0 controls)"
            descriptive = bool(control) and count * len(control) <= success_count * len(group)
            limitation = " Not discriminating by higher failure occurrence." if descriptive else ""
            domain = next(row for row in residuals["domains"] if (row["reason"], row["cell"], row["run"]) == key)
            rows.append(_sentence(f"contrast-{ordinal}", "S2", "process",
                f"{_cell_label(key[1])}, run {key[2]}, {key[0]}: {_LABELS[term]} {count}/{len(group)} failures; {contrast}.{limitation} "
                f"Rule-negative {domain['rule_negative']}/{domain['failure_n']}; comparison-missing {domain['comparison_missing']}/{domain['failure_n']} failures.",
                predicate=term, scope={"label": f"{key[0]}/{key[1]}/run{key[2]}", "numerator": count, "denominator": len(group)},
                window={"kind": "group_episodes"}, descriptive=descriptive,
                comparison={"status": "matched" if control else "missing", "match_level": "same_cell_run",
                    "discriminating": None if not control else not descriptive,
                    "left": {"label": f"{key[0]}/{key[1]}/run{key[2]}", "numerator": count, "denominator": len(group)},
                    "right": {"label": f"success/{key[1]}/run{key[2]}", "numerator": success_count, "denominator": len(control)}},
                refs=[*[f"episodes/{e['episode_id']}" for e in group + control], "semantic/residuals/domains"]))
        controls = successes.get(primary[1:], [])
        if controls:
            example, selection = _representative(controls)
            text, refs = _timeline(example, selection)
            progress = _progress_outline(example, attitude=False)
            if progress is not None:
                text += " " + progress[0]
                refs += progress[1]
            rows.append(_sentence("compatible-success", "S1", "counterexample", "Same-cell/run success. " + text,
                scope={"label": example["episode_id"], "denominator": len(controls), "numerator": selection["pattern_n"]},
                window={"start": 1, "end": example["steps"], "terminal": example["steps"]}, refs=refs or [f"episodes/{example['episode_id']}"]))
        else:
            rows.append(_sentence("counterexample-missing", "S5", "counterexample",
                f"Primary failure domain has {len(failures[primary])} failed episodes and 0 same-cell/run success controls; this section lacks a compatible success example. No other domain is substituted.",
                scope={"label": "primary failure domain", "denominator": len(failures[primary])}, gap_type="comparison_missing", refs=["semantic/residuals/domains"]))
        all_successes = [e for e in episodes if e["primary_reason"] == "success"]
        if all_successes:
            occurrences = "; ".join(
                f"{_LABELS[term]} {sum(_first_confirmation(e, term) is not None for e in all_successes)}/{len(all_successes)}"
                for term in PATTERN_TERMS)
            rows.append(_sentence("unmatched-success-context", "S2", "counterexample",
                f"Mixed-cell/run successes (not matched controls): {occurrences}. Counts may overlap.",
                predicate="successful_process_occurrence",
                scope={"label": "all successful episodes across cells/runs", "numerator": len(all_successes), "denominator": len(all_successes)},
                window={"kind": "group_episodes"},
                comparison={"status": "missing", "reason": "mixed cells/runs; not a matched failure comparison"},
                refs=[f"episodes/{e['episode_id']}" for e in all_successes]))
    else:
        for section in ("process", "counterexample"):
            rows.append(_sentence(section + "-missing", "S5", section,
                f"This cohort has 0 failed episodes out of {len(episodes)}; {section} has no failure group to characterize. This does not establish absence of unmeasured behavior.",
                scope={"label": "public-dev cohort", "denominator": len(episodes)}, gap_type="no_failure_group"))
    rows.append(_sentence("visual-unavailable", "S5", "visual",
        f"Validated visual observations: 0/{len(episodes)} episodes. Contact surface and leg shape remain unobserved.",
        scope={"label": "public-dev cohort", "denominator": len(episodes)}, gap_type="visual_not_observed"))
    rows.append(_sentence("residual-total", "S5", "gaps",
        f"Failure domains {len(residuals['domains'])}: {residuals['failure_n']} failed episodes; {residuals['rule_negative']} rule-negative, {residuals['comparison_missing']} comparison-missing. "
        f"Comparison-missing means fewer than {CONTRACT['minimum_success_controls']} same-cell/run successes, not missing measurements. "
        "Otherwise rule-negative means an empty four-term pattern or the same term/time pattern in "
        f">={CONTRACT['signature_success_fraction']:.0%} of those successes; it does not mean no movement. "
        "Pattern terms are tilt, low progress, same-foot unloaded low motion and up_z<0; first confirmation is binned by time before terminal "
        f"(<= {CONTRACT['relative_terminal_bins_s'][0]:g}s, <= {CONTRACT['relative_terminal_bins_s'][1]:g}s, > {CONTRACT['relative_terminal_bins_s'][1]:g}s). "
        f"Numeric windows cover {sampling['residual_selected_n']}/{sampling['residual_eligible_n']} residual/missing episodes; {len(sampling['residual_unselected_episode_ids'])} unsampled. "
        f"These counts do not identify causes.",
        scope={"label": "all failure domains", "denominator": residuals["failure_n"]}, gap_type="rule_negative_and_comparison_missing", refs=["semantic/residuals", "semantic/sampling"]))
    # Each protected domain's residual counts are beside its matched contrast;
    # do not repeat its identity and denominator in a second set of sentences.
    coverage_by_stratum = {}
    for row in sampling["episode_coverage"]:
        coverage_by_stratum.setdefault(row["stratum"], []).append(f"{row['outcome']} {row['selected_n']}/{row['episode_n']}")
    group_coverage = "; ".join(f"{stratum}: {', '.join(counts)}" for stratum, counts in coverage_by_stratum.items())
    rows.append(_sentence("sampling-coverage", "S5", "gaps",
        f"Numeric samples: {sampling['selected_n']}/{sampling['episode_n']} episodes ({group_coverage}). "
        f"Window-union time coverage: median {sampling['time_coverage_median']:.3f}, minimum {sampling['time_coverage_minimum']:.3f}. "
        f"Supplement: {sampling['supplemental_selected_n']}/{sampling['residual_eligible_n']}. "
        f"Residual domains omitted: {max(0,len(residuals['domains'])-len(protected))}. Numeric sampling is not visual coverage; unseen behavior remains unknown.",
        scope={"label": "numeric sampling", "denominator": len(episodes)}, gap_type="sampling_scope", refs=["semantic/sampling"]))
    rows.append(_sentence("parent-unavailable", "S5", "parent_diff",
        f"Parent difference unavailable for this {len(episodes)}-episode report: no bound comparable parent supplied. Different rules or cohorts cannot be compared as one measured change.",
        scope={"label": "public-dev cohort", "denominator": len(episodes)}, gap_type="bound_parent_missing"))
    return rows


def build_stair_process_behavior(reference, *, expected_identity=None):
    """Production CPU entry point; the original full-v2 measurements survive."""
    from .behavior import build_stair_behavior
    return derive_stair_process_behavior(build_stair_behavior(reference, expected_identity=expected_identity))


def derive_stair_process_behavior(full, *, development_full=None):
    from .behavior import FULL_SCHEMA as MEASURED_FULL_SCHEMA, RULE_VERSION
    if full.get("schema_version") != MEASURED_FULL_SCHEMA or full.get("domain", {}).get("rule_version") != RULE_VERSION:
        raise ValueError("Stair process derivation requires complete measured full.v2")
    if not full.get("episodes") or len({e["episode_id"] for e in full["episodes"]}) != len(full["episodes"]):
        raise ValueError("Stair process episode coverage mismatch")
    residuals = build_stair_residuals(full)
    sampling = select_stair_observation_windows(full, residuals)
    result = deepcopy(dict(full))
    result["schema_version"] = FULL_SCHEMA
    result["semantic"] = {"contract": deepcopy(CONTRACT), "source_schema_version": MEASURED_FULL_SCHEMA,
        "residuals": residuals, "sampling": sampling,
        "term_cards": build_stair_term_cards(development_full or full),
        "term_card_statistics_scope": "supplied development pool" if development_full is not None else "current derivation pool; descriptive statistics, not qualification",
        "sentences": _sentences(full, residuals, sampling), "comparison_summary": _comparison_summary(full),
        "candidate_lexicon": {"schema_version": "stair.candidate_terms.v1", "entries": [], "automatic_promotion": False},
        "actionability": {"available": ["instant root orientation", "instant support force and velocity", "existing forbidden-contact input"],
            "diagnostic_only": ["window/streak confirmation and residual signatures require history absent from reward ABI"],
            "unavailable": ["contact surface identity", "toe/sole position", "new controller state"],
            "authority": "Existing single-mechanism reward authority only; observations do not authorize new inputs, gates, terrain or PPO."}}
    return result


def project_stair_process_report(value):
    if value.get("schema_version") == PUBLIC_SCHEMA:
        return validate_stair_process_report(value)
    if value.get("schema_version") != FULL_SCHEMA:
        raise ValueError("Stair process schema mismatch")
    from true_rl_agent.evaluation.process_report import render_process_report
    from true_rl_agent.evaluation.visual_observer import update_visual_projection_count
    semantic = value["semantic"]
    if semantic["contract"] != CONTRACT:
        raise ValueError("Stair process contract mismatch")
    envelope = {"schema_version": PUBLIC_SCHEMA, "domain": deepcopy(value["domain"]),
        "derivation_version": DERIVATION_VERSION, "lexicon_version": LEXICON_VERSION,
        "comparison_summary": deepcopy(semantic["comparison_summary"])}
    result = render_process_report(envelope=envelope, sentences=semantic["sentences"],
        actionability=semantic["actionability"], max_bytes=MAX_PUBLIC_BYTES,
        update_projection=update_visual_projection_count)
    return validate_stair_process_report(result)


def validate_stair_process_report(value):
    from true_rl_agent.evaluation.process_report import validate_public_report
    from .behavior import _DOMAIN_KEYS, RULE_VERSION
    expected = {"schema_version", "domain", "derivation_version", "lexicon_version", "comparison_summary", "report", "report_projection"}
    if (set(value) != expected or value.get("schema_version") != PUBLIC_SCHEMA
            or value.get("derivation_version") not in {"stair.process_report.v1", DERIVATION_VERSION}
            or value.get("lexicon_version") != LEXICON_VERSION):
        raise ValueError("Stair process public fields mismatch")
    if set(value["domain"]) != _DOMAIN_KEYS or value["domain"]["rule_version"] != RULE_VERSION:
        raise ValueError("Stair process domain mismatch")
    validate_public_report(value, max_bytes=MAX_PUBLIC_BYTES)
    report = value["report"]
    if set(report) != set(_SECTIONS):
        raise ValueError("Stair process report sections mismatch")
    identifiers = []
    for section in _SECTIONS[:-1]:
        if not isinstance(report[section], list) or not report[section]:
            raise ValueError("Stair process report empty section")
        for row in report[section]:
            if set(row) != {"id", "text"} or not all(isinstance(row[key], str) and row[key] for key in row):
                raise ValueError("Stair process public sentence mismatch")
            identifiers.append(row["id"])
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Stair process duplicate sentence")
    summary = value["comparison_summary"]
    if sum(item["n"] for item in summary["strata"].values()) != summary["episode_n"]:
        raise ValueError("Stair process denominator mismatch")
    for item in summary["strata"].values():
        if set(item) != {"n", "success", *PATTERN_TERMS} or any(type(x) is not int or not 0 <= x <= item["n"] for x in item.values()):
            raise ValueError("Stair process count mismatch")
    if _size(value) > MAX_PUBLIC_BYTES:
        raise ValueError("Stair process public byte budget exceeded")
    return deepcopy(dict(value))


def compare_stair_process_reports(parent, candidate, *, expected_parent_checkpoint_hash):
    if not isinstance(parent, Mapping) or not isinstance(candidate, Mapping) or not expected_parent_checkpoint_hash:
        return {"status": "not_available", "reason": "missing_bound_parent_or_behavior"}
    if parent.get("schema_version") not in {FULL_SCHEMA, PUBLIC_SCHEMA} or candidate.get("schema_version") not in {FULL_SCHEMA, PUBLIC_SCHEMA}:
        return {"status": "not_comparable", "reason": "process_report_version_mismatch"}
    versions = [item.get("derivation_version") if item.get("schema_version") == PUBLIC_SCHEMA
                else item.get("semantic", {}).get("contract", {}).get("version") for item in (parent, candidate)]
    if versions != [DERIVATION_VERSION, DERIVATION_VERSION]:
        return {"status": "not_comparable", "reason": "process_report_version_mismatch"}
    left, right = project_stair_process_report(parent), project_stair_process_report(candidate)
    if left["domain"]["checkpoint_hash"] != expected_parent_checkpoint_hash:
        return {"status": "not_comparable", "reason": "parent_checkpoint_mismatch"}
    if not left["domain"].get("source_hash") or not right["domain"].get("source_hash"):
        return {"status": "not_comparable", "reason": "execution_source_unavailable"}
    fields = ("task_protocol_hash", "suite_hash", "projection_hash", "source_hash", "cohort_hash", "episode_keys_hash", "reward_input_schema_hash", "rule_version")
    unequal = [key for key in fields if left["domain"][key] != right["domain"][key]]
    a, b = left["comparison_summary"], right["comparison_summary"]
    if unequal or a["cell_run_keys_hash"] != b["cell_run_keys_hash"]:
        return {"status": "not_comparable", "reason": "comparison_domain_mismatch", "fields": unequal or ["cell_run_keys_hash"]}
    rows = []
    for stratum in sorted(set(a["strata"]) | set(b["strata"])):
        x, y = a["strata"].get(stratum), b["strata"].get(stratum)
        rows.append({"stratum": stratum, "parent": x, "candidate": y,
                     "rate_delta": {key: y[key]/y["n"]-x[key]/x["n"] for key in ("success", *PATTERN_TERMS)} if x and y else None})
    totals = [{key: sum(row[key] for row in side["strata"].values()) for key in ("n", *PATTERN_TERMS)} for side in (a, b)]
    term = max(PATTERN_TERMS, key=lambda key: (abs(totals[1][key]/totals[1]["n"]-totals[0][key]/totals[0]["n"]), -PATTERN_TERMS.index(key)))
    left_count, right_count = totals[0][term], totals[1][term]
    statement = _sentence("parent-process-diff", "S4", "parent_diff",
        f"Parent to candidate, all observed episodes: {_LABELS[term]} confirmed strictly before terminal {left_count}/{totals[0]['n']} -> {right_count}/{totals[1]['n']}. "
        f"Same-cell/run pairs {a['cell_run_n']}/{a['cell_run_n']}, missing 0. Proportion change, not step alignment or causation.",
        predicate=term, scope={"label": "paired public-dev cohorts"},
        comparison={"left": {"label": "parent cohort", "numerator": left_count, "denominator": totals[0]["n"], "rule_version": LEXICON_VERSION},
                    "right": {"label": "candidate cohort", "numerator": right_count, "denominator": totals[1]["n"], "rule_version": LEXICON_VERSION},
                    "paired": a["cell_run_n"], "missing": 0},
        refs=[f"parent:{expected_parent_checkpoint_hash}/comparison_summary", f"candidate:{right['domain']['checkpoint_hash']}/comparison_summary"])
    from true_rl_agent.evaluation.process_report import validate_sentence
    validate_sentence(statement)
    return {"status": "available", "rule_version": LEXICON_VERSION,
        "parent_checkpoint_hash": expected_parent_checkpoint_hash, "candidate_checkpoint_hash": right["domain"]["checkpoint_hash"],
        "paired_cell_run_n": a["cell_run_n"], "missing_cell_run_n": 0, "rows": rows, "statement": statement,
        "scope": "same-domain whole-episode proportions with identical cell/run coverage; not step-aligned and not causal"}


def attach_stair_process_difference(report, difference):
    """The prompt adapter knows the parent binding; the rollout finalizer does not."""
    report = validate_stair_process_report(report)
    statement = difference.get("statement")
    if difference.get("status") == "available" and isinstance(statement, Mapping):
        from true_rl_agent.evaluation.process_report import validate_sentence
        validate_sentence(statement)
        report["report"]["parent_diff"] = [{"id": statement["id"], "text": statement["text"]}]
    elif difference.get("status") == "not_comparable":
        row = report["report"]["parent_diff"][0]
        row["text"] = f"Parent process comparison is unavailable: {difference['reason']}; {report['comparison_summary']['episode_n']} candidate episodes. No cross-domain change is inferred."
    from true_rl_agent.evaluation.process_report import fit_process_report_capacity
    from true_rl_agent.evaluation.visual_observer import update_visual_projection_count
    return validate_stair_process_report(fit_process_report_capacity(
        report, max_bytes=MAX_PUBLIC_BYTES, update_projection=update_visual_projection_count))
