#!/usr/bin/env python3
"""CPU-only P2 acceptance of a persisted Stair numeric capture on/off pair.

Usage: python verify_stair_pair.py --project-root SOURCE --on-dir ON --off-dir OFF
       --output report.json
The output is diagnostic evidence, never an acceptance/promotion receipt.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time
import traceback

MAX_OVERHEAD = 0.25
MAX_EXTRA_RSS_KIB = 512 * 1024
MAX_ARCHIVE_BYTES = 2 * 1024**3
WINDOW_SECONDS = 1.0


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parse_time(path):
    """Read the frozen /usr/bin/time format; absence is not zero usage."""
    path = Path(path)
    if not path.is_file():
        return {"status": "not_available", "path": str(path)}
    match = re.fullmatch(r"\s*peak_rss_kib=(\d+) elapsed_s=([0-9.]+) exit=(\d+)\s*", path.read_text())
    if not match:
        raise ValueError(f"unrecognized time measurement: {path}")
    return {"status": "available", "path": str(path), "sha256": sha256(path),
            "peak_rss_kib": int(match[1]), "elapsed_s": float(match[2]), "exit": int(match[3])}


def diff_paths(left, right, path="", limit=20):
    if left == right:
        return []
    if isinstance(left, dict) and isinstance(right, dict):
        result = []
        for key in sorted(set(left) | set(right)):
            result.extend(diff_paths(left.get(key), right.get(key), f"{path}/{key}", limit))
            if len(result) >= limit:
                break
        return result[:limit]
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        result = []
        for i, (a, b) in enumerate(zip(left, right)):
            result.extend(diff_paths(a, b, f"{path}/{i}", limit))
            if len(result) >= limit:
                break
        return result[:limit]
    return [path or "/"]


class EpisodeEvents:
    """Bounded streaming counters plus one second of Boolean event samples."""

    def __init__(self, cell, signals, support_names, step_dt):
        self.cell, self.signals, self.support_names = cell, signals, support_names
        self.step_dt = step_dt
        self.window_steps = max(1, int(math.ceil(WINDOW_SECONDS / step_dt)))
        self.window = deque(maxlen=self.window_steps + 1)
        self.required = {
            "base_orientation_condition": int(signals["base_orientation_instability"]["consecutive_control_steps"]),
            "slip_condition_any_support": int(signals["slip"]["consecutive_control_steps"]),
            "low_clearance_condition_any_noncontact_support": int(signals["insufficient_clearance"]["consecutive_control_steps"]),
            "forbidden_contact_manager_active": 1,
            "no_support_contact": 1,
            **{f"support_contact_{i}": 1 for i in range(len(support_names))},
        }
        self.stats = {name: {"first_condition_step": None, "first_sustained_step": None,
                            "last_condition_step": None, "condition_steps": 0,
                            "longest_streak_steps": 0, "streak": 0} for name in self.required}
        self.first_riser_progress = cell.start_platform_length_m / 2.0
        self.anchor = {"progress_m": self.first_riser_progress, "status": "not_reached",
                       "first_observed_crossing_step": None, "time_resolution_s": step_dt}
        self.previous_progress = None
        self.last_snapshot = self.previous_snapshot = None

    def observe(self, transition, raw, env_index):
        from true_rl_agent.tasks.stair_climbing.transition_decoder import _roll_pitch_from_wxyz
        t, signals = transition, self.signals
        conditions = {
            "base_orientation_condition": max(t.abs_roll_rad, t.abs_pitch_rad) > float(signals["base_orientation_instability"]["threshold"]),
            "slip_condition_any_support": any(c and v > float(signals["slip"]["speed_threshold"])
                for c, v in zip(t.support_contacts, t.support_planar_speeds_mps)),
            "low_clearance_condition_any_noncontact_support": any(not c and h < float(signals["insufficient_clearance"]["threshold"])
                for c, h in zip(t.support_contacts, t.support_surface_clearances_m)),
            "forbidden_contact_manager_active": t.forbidden_contact,
            "no_support_contact": not any(t.support_contacts),
            **{f"support_contact_{i}": value for i, value in enumerate(t.support_contacts)},
        }
        for name, value in conditions.items():
            s = self.stats[name]
            s["streak"] = s["streak"] + 1 if value else 0
            if value:
                s["first_condition_step"] = s["first_condition_step"] or t.episode_step
                s["last_condition_step"] = t.episode_step
                s["condition_steps"] += 1
                s["longest_streak_steps"] = max(s["longest_streak_steps"], s["streak"])
                if s["streak"] >= self.required[name] and s["first_sustained_step"] is None:
                    s["first_sustained_step"] = t.episode_step
        self.window.append((t.episode_step, conditions))
        if self.anchor["status"] == "not_reached" and t.progress_m >= self.first_riser_progress:
            self.anchor.update(status=("already_beyond_at_first_sample" if self.previous_progress is None
                                       else "observed_forward_crossing"),
                               first_observed_crossing_step=(None if self.previous_progress is None else t.episode_step))
        self.previous_progress = t.progress_m
        roll, pitch = _roll_pitch_from_wxyz(raw["base_quat_w"][env_index].tolist())
        self.previous_snapshot, self.last_snapshot = self.last_snapshot, {
            "control_step": t.episode_step, "root_progress_m": t.progress_m,
            "root_roll_rad": roll, "root_pitch_rad": pitch, "root_base_clearance_m": t.base_clearance_m,
            "support_contacts": list(t.support_contacts),
            "support_planar_speeds_mps": list(t.support_planar_speeds_mps),
            "support_surface_clearances_m": list(t.support_surface_clearances_m),
            "support_force_norms_n": [math.sqrt(sum(float(x)**2 for x in force))
                                      for force in raw["support_contact_forces_w"][env_index].tolist()],
            "forbidden_body_force_norms_n": [math.sqrt(sum(float(x)**2 for x in force))
                for force in raw["forbidden_contact_forces_w"][env_index].tolist()],
            "forbidden_manager_active": t.forbidden_contact,
            "raw_environment_done": t.done, "active_termination_terms": list(t.termination_terms),
        }

    def finish(self, outcome, run_index, runner_seed):
        terminal = outcome.episode_steps
        start = max(1, terminal - self.window_steps + 1)
        window = [(step, mask) for step, mask in self.window if step >= start]
        events, order = {}, defaultdict(list)
        for name, source in self.stats.items():
            s = {k: v for k, v in source.items() if k != "streak"}
            s["required_consecutive_steps"] = self.required[name]
            s["present_strictly_before_terminal"] = s["first_condition_step"] is not None and s["first_condition_step"] < terminal
            s["sustained_strictly_before_terminal"] = s["first_sustained_step"] is not None and s["first_sustained_step"] < terminal
            s["condition_steps_strictly_before_terminal"] = s["condition_steps"] - int(s["last_condition_step"] == terminal)
            s["window_present_strictly_before_terminal"] = any(mask[name] for step, mask in window if step < terminal)
            intervals = []
            for step, mask in window:
                if mask[name]:
                    if intervals and intervals[-1][1] == step - 1:
                        intervals[-1][1] = step
                    else:
                        intervals.append([step, step])
            s["terminal_window_true_intervals_inclusive"] = intervals
            if s["first_sustained_step"] is not None:
                order[s["first_sustained_step"]].append(name)
            events[name] = s
        return {"run_index": run_index, "runner_seed": runner_seed, "env_index": outcome.env_index,
                "cell_id": outcome.evaluation_cell_id, "outcome": outcome.to_public_dict(),
                "cell_horizon_control_steps": self.cell.episode_horizon_control_steps,
                "first_riser_root_anchor": self.anchor,
                "terminal_window": {"first_control_step": start, "last_control_step": terminal,
                                    "control_dt_s": self.step_dt, "includes_terminal": True},
                "events": events,
                "first_sustained_event_order": [{"control_step": step,
                    "steps_before_terminal": terminal-step, "simultaneous_events": sorted(names)}
                    for step, names in sorted(order.items())],
                "last_pre_terminal_snapshot": self.previous_snapshot,
                "terminal_snapshot": self.last_snapshot}


def summarize_groups(rows):
    """Keep all episodes, including failures that never reached the first riser."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row["run_index"], row["cell_id"], row["outcome"]["primary_reason"])].append(row)
    result = []
    for (run, cell, reason), values in sorted(groups.items()):
        n = len(values)
        result.append({"run_index": run, "runner_seed": values[0]["runner_seed"], "cell_id": cell,
            "primary_reason": reason, "denominator": n,
            "first_riser_anchor_status_counts": dict(Counter(r["first_riser_root_anchor"]["status"] for r in values)),
            "event_rates": {name: {key: {"count": sum(r["events"][name][key] for r in values),
                                           "denominator": n,
                                           "rate": sum(r["events"][name][key] for r in values)/n}
                for key in ("present_strictly_before_terminal", "sustained_strictly_before_terminal",
                            "window_present_strictly_before_terminal")} for name in values[0]["events"]}})
    cells = sorted({(r["run_index"], r["cell_id"]) for r in rows})
    comparisons = []
    for run, cell in cells:
        matched = [g for g in result if g["run_index"] == run and g["cell_id"] == cell]
        index = {g["primary_reason"]: g for g in matched}
        comparisons.append({"run_index": run, "cell_id": cell,
                            "all_episode_denominator": sum(g["denominator"] for g in matched),
                            "forbidden_contact_denominator": index.get("forbidden_contact", {}).get("denominator", 0),
                            "success_denominator": index.get("success", {}).get("denominator", 0),
                            "comparison_status": "available" if "forbidden_contact" in index and "success" in index else "one_group_absent",
                            "rates_location": "groups matched on run_index/cell_id/primary_reason"})
    return {"groups": result, "same_cell_seed_comparison_coverage": comparisons,
            "first_question_comparison_status": "available" if any(c["comparison_status"] == "available" for c in comparisons) else "INCONCLUSIVE_no_same_cell_seed_success_failure_pair",
            "episode_length_range_control_steps": [min(r["outcome"]["episode_steps"] for r in rows), max(r["outcome"]["episode_steps"] for r in rows)],
            "all_episode_denominator": len(rows),
            "primary_reason_counts": dict(Counter(r["outcome"]["primary_reason"] for r in rows))}


def reconstruct(reference):
    """One chunk at a time: bytes -> task decoder -> tracker + event counters."""
    import numpy as np
    import torch
    from true_rl_agent.composition.cell_manifest import ResolvedCellManifestV1
    from true_rl_agent.runner.numeric_evidence import iter_numeric_evidence, load_numeric_evidence_manifest
    from true_rl_agent.tasks.stair_climbing.numeric_capture import STAIR_NUMERIC_PROFILE, STAIR_NUMERIC_PROFILE_V1, _DYNAMIC_FIELDS, _STATIC_FIELDS
    from true_rl_agent.tasks.stair_climbing.legacy_transition_reader import STAIR_CONTACT_HISTORY_TENSOR_FIELDS
    from true_rl_agent.tasks.stair_climbing.runtime_projection import stair_runtime_projection_from_manifest
    from true_rl_agent.tasks.stair_climbing.evaluation_semantics import StairEvaluationSemantics
    from true_rl_agent.tasks.stair_climbing.unified_training import StairUnifiedTrainingProfile
    from true_rl_agent.tasks.stair_climbing.unified_evaluation import StairUnifiedEvaluationSuite
    from true_rl_agent.tasks.stair_climbing.transition_decoder import decode_stair_transition_batch
    from true_rl_agent.tasks.stair_climbing.episode_evidence import StairEpisodeTracker
    from true_rl_agent.tasks.stair_climbing.unified_scoring import build_unified_stair_summary
    manifest = load_numeric_evidence_manifest(reference)
    context = manifest["metadata"]
    if context["profile"] not in {STAIR_NUMERIC_PROFILE_V1, STAIR_NUMERIC_PROFILE}:
        raise ValueError("unsupported Stair numeric profile")
    dynamic_fields = (*_DYNAMIC_FIELDS, *(STAIR_CONTACT_HISTORY_TENSOR_FIELDS if context["profile"] == STAIR_NUMERIC_PROFILE else ()))
    resolved = ResolvedCellManifestV1.from_mapping(context["resolved_cell_manifest"])
    projection = stair_runtime_projection_from_manifest(resolved)
    training = StairUnifiedTrainingProfile.from_mapping(context["training_profile"])
    suite = StairUnifiedEvaluationSuite.from_mapping(context["evaluation_suite"], training_profile=training)
    s = context["semantics"]
    semantics = StairEvaluationSemantics.from_task_outcomes(task_protocol_hash=s["task_protocol_hash"], task_id=s["task_id"], episode_outcomes=s["episode_outcomes"])
    if semantics.to_dict() != s or projection.to_summary_dict() != context["runtime_projection"]:
        raise ValueError("archived semantics/runtime projection mismatch")
    cells = suite.private_cell_assignment("public_dev")
    names = list(resolved.to_dict()["resolved_roles"]["support_effector"])
    rows, per_seed, dtype_sets = [], [], defaultdict(set)
    current_run, observed_steps, valid_count, max_chunk_bytes = 0, 0, 0, 0
    active, outcomes, trackers, events, static = [], [], [], [], {}
    def close_seed():
        expected = context["seeds"][current_run-1]
        if any(active) or [o.to_public_dict() for o in outcomes] != expected["outcomes"]:
            raise ValueError("reconstructed first-episode outcomes differ or seed is incomplete")
        if [o.episode_steps for o in outcomes] != expected["episode_steps"] or observed_steps != expected["observed_control_steps"]:
            raise ValueError("archived episode/control-step counts mismatch")
    for metadata, arrays in iter_numeric_evidence(reference):
        max_chunk_bytes = max(max_chunk_bytes, sum(a.nbytes for a in arrays.values()))
        for name, array in arrays.items():
            dtype_sets[name].add(array.dtype.str)
            expected = np.bool_ if name in {"valid", "done_mask", "wrapper_done_mask"} or name.startswith("termination_") else np.int64 if name in {"episode_length", "terrain_types"} else np.float32
            if array.dtype != np.dtype(expected):
                raise ValueError(f"unexpected/quantized dtype {name}: {array.dtype}, expected {np.dtype(expected)}")
        run = metadata["run_index"]
        if run != current_run:
            if current_run:
                close_seed()
            if run != current_run+1 or not metadata["static_present"]:
                raise ValueError("archive seed order/static boundary mismatch")
            current_run, observed_steps = run, 0
            seed = context["seeds"][run-1]
            if seed["run_index"] != run or seed["split"] != "public_dev":
                raise ValueError("archive seed identity mismatch")
            static = {name: torch.from_numpy(arrays[name]) for name in _STATIC_FIELDS}
            trackers = [StairEpisodeTracker(None, semantics, env_index=i, bin_id=c.bin_id, evaluation_cell=c) for i,c in enumerate(cells)]
            events = [EpisodeEvents(c, s["episode_outcomes"]["signals"], names, seed["step_dt"]) for c in cells]
            outcomes, active = [None]*len(cells), [True]*len(cells)
            per_seed.append(outcomes)
        if metadata["step_start"] != observed_steps+1:
            raise ValueError("archive control-step gap")
        terms = tuple(metadata["termination_term_order"])
        for step in range(metadata["steps"]):
            if arrays["valid"][step].tolist() != active:
                raise ValueError("first-episode valid mask differs (terminal/reset boundary)")
            valid_count += sum(active)
            raw = {**metadata["transition_header"], **static,
                   **{name: torch.from_numpy(arrays[name][step]) for name in dynamic_fields},
                   "termination_term_order": terms,
                   "termination_masks": {name: torch.from_numpy(arrays[f"termination_{i}"][step]) for i,name in enumerate(terms)}}
            transitions = decode_stair_transition_batch(raw, wrapper_done_mask=torch.from_numpy(arrays["wrapper_done_mask"][step]),
                protocol=None, semantics=semantics, expected_bin_assignment=tuple(c.bin_id for c in cells),
                expected_cell_assignment=cells, runtime_projection=projection)
            for i,t in enumerate(transitions):
                if active[i]:
                    events[i].observe(t, raw, i)
                    outcome = trackers[i].observe(t)
                    if outcome is not None:
                        outcomes[i], active[i] = outcome, False
                        rows.append(events[i].finish(outcome, run, seed["runner_seed"]))
            observed_steps += 1
    if not current_run or current_run != len(context["seeds"]):
        raise ValueError("missing archived seeds")
    close_seed()
    summary = build_unified_stair_summary(suite=suite, semantics=semantics, split="public_dev", per_seed_outcomes=per_seed)
    if summary != context["summary"] or valid_count != sum(o.episode_steps for seed in per_seed for o in seed):
        raise ValueError("reconstructed summary/valid-step count mismatch")
    rows.sort(key=lambda r: (r["run_index"], r["env_index"]))
    return {"summary": summary, "rows": rows, "manifest": manifest, "resolved": resolved,
            "signals": s["episode_outcomes"]["signals"], "support_names": names,
            "array_dtypes": {k: sorted(v) for k,v in sorted(dtype_sets.items())},
            "valid_first_episode_control_steps": valid_count, "max_chunk_raw_bytes": max_chunk_bytes}


def verify(args):
    from true_rl_agent.evaluation.binding import canonical_json_hash, EvaluationBindingV1
    from true_rl_agent.evaluation.phases import EvaluationPhase, EvaluationPhaseResult
    from true_rl_agent.tasks.stair_climbing.config import StairClimbingTaskConfig
    from true_rl_agent.tasks.stair_climbing.row_catalog import stair_robot_row
    from true_rl_agent.tasks.stair_climbing.phased_evaluation import canonical_stair_phase_result_summary
    import torch
    torch.set_num_threads(1)
    checks = []
    def check(name, value, detail=None):
        checks.append({"name": name, "status": "INCONCLUSIVE" if value is None else "PASS" if value else "FAIL", "detail": detail})
    dirs = {"on": Path(args.on_dir).resolve(), "off": Path(args.off_dir).resolve()}
    reports, results, jobs, measures = {}, {}, {}, {}
    for label, directory in dirs.items():
        reports[label] = r = read_json(directory/"result.json")
        results[label] = p = read_json(directory/"public_dev_result.json")
        jobs[label] = j = read_json(directory/"public_dev_job.json")
        EvaluationPhaseResult.from_private_mapping(p)
        check(label+"_finished_driver", r["status"] == "PASS" and r["mode"] == "run" and r["capture_numeric"] == (label == "on") and r["phases_executed"] == ["public_dev"])
        check(label+"_receipt_byte_binding", r["public_dev_result_sha256"] == "sha256:"+sha256(directory/"public_dev_result.json") and r["job_file_sha256"] == "sha256:"+sha256(directory/"public_dev_job.json"))
        check(label+"_job_and_report_binding", r["public_dev_job_spec_hash"] == canonical_json_hash(j["spec"]) and r["public_dev_result_hash"] == canonical_json_hash(p)
              and r["evaluation_plan_hash"] == p["evaluation_plan_hash"] == j["spec"]["evaluation_plan_hash"]
              and p["phase"] == j["spec"]["phase"] == "public_dev")
        plan = j["spec"]["evaluation_plan"]
        check(label+"_receipt_matches_job_identity", p["job_id"] == j["spec"]["job_id"]
              and all(p[k] == plan[k] for k in ("task_id", "subject", "subject_manifest_hash", "program_source_checksum", "training_binding_hash", "capability_profile_hash", "eval_protocol_hash", "execution_device"))
              and p["checkpoint_hash"] == plan["checkpoint"]["checksum"] == r["checkpoint_sha256"]
              and p["evaluation_plan_hash"] == canonical_json_hash(plan)
              and p["evaluation_binding_hash"] == EvaluationBindingV1.from_mapping(plan["evaluation_binding"]).binding_hash)
        measures[label] = parse_time(Path(str(directory)+".time"))
    on, off = results["on"], results["off"]
    for field in ("task_id", "subject", "subject_manifest_hash", "program_source_checksum", "training_binding_hash", "capability_profile_hash", "checkpoint_hash", "eval_protocol_hash", "evaluation_binding_hash", "environment_source_preflight", "backend_execution_environment_verification"):
        check("pair_same_"+field, on[field] == off[field])
    for field in ("current_manifest_hash", "source_training_manifest_hash", "source_train_summary_sha256", "planned_public_dev_episodes", "public_dev_seed_count", "eval_num_envs"):
        check("pair_same_"+field, reports["on"][field] == reports["off"][field])
    for label,p in results.items():
        backend = p["backend_execution_environment_verification"]
        check(label+"_source_verified", backend.get("status") == "pass" and backend.get("expected_manifest_hash") == backend.get("actual_manifest_hash") and p["environment_source_preflight"].get("status") == "pass")
    check("capture_off_has_no_numeric_reference", "numeric_evidence" not in off["task_evidence"])
    reference = dict(on["task_evidence"]["numeric_evidence"])
    original_locator = reference["manifest_path"]
    if not Path(original_locator).is_file():
        reference["manifest_path"] = str(dirs["on"]/"numeric_evidence/manifest.json")
    recovered = reconstruct(reference)
    manifest, resolved = recovered["manifest"], recovered["resolved"]
    row = stair_robot_row(resolved.summary_identity()["cell_variant_id"])
    config = replace(StairClimbingTaskConfig.defaults(Path(args.project_root)/row.protocol_relative_path), resolved_cell_task_protocol_hash=resolved.resolved_cell_task_protocol_hash)
    for label,p in results.items():
        canonical = canonical_stair_phase_result_summary(config=config, expected_phase=EvaluationPhase.PUBLIC_DEV, result=EvaluationPhaseResult.from_private_mapping(p))
        check(label+"_canonical_task_summary", canonical == p["task_evidence"]["summary"])
    summary_on, summary_off = (p["task_evidence"]["summary"] for p in (on,off))
    check("full_task_summary_exact", canonical_json_hash(summary_on) == canonical_json_hash(summary_off), diff_paths(summary_on, summary_off))
    check("archive_reconstructs_receipt_summary_exact", canonical_json_hash(recovered["summary"]) == canonical_json_hash(summary_on))
    identity = manifest["identity"]
    expected = {key:on[key] for key in ("job_id", "task_id", "subject_manifest_hash", "program_source_checksum", "training_binding_hash", "capability_profile_hash", "checkpoint_hash", "evaluation_plan_hash", "environment_source_preflight", "backend_execution_environment_verification")}
    expected.update(subject_kind=on["subject"]["kind"], subject_id=on["subject"]["id"], splits=["public_dev"], resolved_cell_manifest_hash=resolved.manifest_hash)
    check("archive_identity_matches_phase", all(identity.get(k) == v for k,v in expected.items()))
    check("archive_manifest_matches_current_receipt", resolved.manifest_hash == reports["on"]["current_manifest_hash"])
    binding = EvaluationBindingV1.from_mapping(identity["evaluation_binding"])
    check("archive_eval_binding_matches_phase", binding.binding_hash == on["evaluation_binding_hash"])
    for label,p in results.items():
        e = p["task_evidence"]["execution_identity"]
        check(label+"_seed_identity_matches_archive", e["seed_namespace"] == identity["seed_namespace"] and e["seed_cohort_hash"] == binding.seed_cohort_hash)
        check(label+"_declared_seed_bindings", jobs[label]["spec"]["evaluation_plan"]["evaluation_binding"] == identity["evaluation_binding"])
    expected_seeds = [config.seed_resolver().derive_seed(split="public_dev", run_index=i, namespace=identity["seed_namespace"]) for i in range(1, config.public_seed_count+1)]
    check("runner_seeds_match_frozen_schedule", [s["runner_seed"] for s in manifest["metadata"]["seeds"]] == expected_seeds)
    rows = recovered["rows"]
    check("complete_episode_denominator", len(rows) == reports["on"]["planned_public_dev_episodes"] == summary_on["ordered_summary"]["total_episode_count"])
    archive_root = Path(reference["manifest_path"]).parent
    disk_files = sorted(p for p in archive_root.rglob("*") if p.is_file())
    disk_bytes = sum(p.stat().st_size for p in disk_files)
    expected_files = {"manifest.json", *(c["file"] for c in manifest["chunks"])}
    check("archive_has_only_published_files", {p.relative_to(archive_root).as_posix() for p in disk_files} == expected_files)
    check("archive_disk_budget", disk_bytes <= MAX_ARCHIVE_BYTES, {"bytes":disk_bytes,"limit_bytes":MAX_ARCHIVE_BYTES})
    check("archive_accounting", sum(c["stored_bytes"] for c in manifest["chunks"]) == manifest["stored_chunk_bytes"] and sum(c["raw_bytes"] for c in manifest["chunks"]) == manifest["raw_bytes"])
    check("bounded_chunks", recovered["max_chunk_raw_bytes"] <= 64*1024**2, recovered["max_chunk_raw_bytes"])
    costs = {label:p["execution_cost"]["splits"]["public_dev"] for label,p in results.items()}
    for field in ("closed_first_episode_control_steps", "evaluation_env_count_per_seed", "seed_count", "simulated_env_control_steps", "vector_env_step_calls"):
        check("pair_same_cost_count_"+field, costs["on"][field] == costs["off"][field])
    check("valid_steps_match_receipt_cost", recovered["valid_first_episode_control_steps"] == costs["on"]["closed_first_episode_control_steps"])
    overhead = costs["on"]["wall_time_s"]/costs["off"]["wall_time_s"]-1 if costs["off"]["wall_time_s"] > 0 else None
    check("rollout_overhead_budget", None if overhead is None else overhead <= MAX_OVERHEAD, {"fraction":overhead,"limit":MAX_OVERHEAD})
    extra_rss = None
    if all(m["status"] == "available" for m in measures.values()):
        check("measured_process_exits", all(m["exit"] == 0 for m in measures.values()))
        extra_rss = measures["on"]["peak_rss_kib"]-measures["off"]["peak_rss_kib"]
    check("extra_peak_rss_budget", None if extra_rss is None else extra_rss <= MAX_EXTRA_RSS_KIB,
          {"delta_kib":extra_rss,"limit_kib":MAX_EXTRA_RSS_KIB})
    counts = Counter(r["outcome"]["primary_reason"] for r in rows)
    check("real_success_and_failure_coverage", counts["success"] > 0 and sum(v for k,v in counts.items() if k != "success") > 0, dict(counts))
    short_count = sum(r["outcome"]["episode_steps"] < r["cell_horizon_control_steps"] for r in rows)
    check("episodes_close_before_horizon_coverage", short_count > 0,
          {"definition":"first episode closes before its own cell horizon; no arbitrary short-duration cutoff", "count":short_count})
    statuses = {c["status"] for c in checks}
    return {"schema_version":"feedback_p2.stair_pair_verification.v1",
        "status":"FAIL" if "FAIL" in statuses else "INCONCLUSIVE" if "INCONCLUSIVE" in statuses else "PASS",
        "checks":checks, "source_paths":{k:str(v) for k,v in dirs.items()},
        "checkpoint_hash":on["checkpoint_hash"], "resolved_cell_identity":resolved.summary_identity(),
        "archive":{"bound_locator":original_locator,"read_locator":reference["manifest_path"],"manifest_sha256":reference["manifest_sha256"],
                   "disk_bytes":disk_bytes,"raw_array_bytes":manifest["raw_bytes"],"chunk_count":len(manifest["chunks"]),
                   "array_dtypes":recovered["array_dtypes"],"max_chunk_raw_bytes":recovered["max_chunk_raw_bytes"],
                   "valid_first_episode_control_steps":recovered["valid_first_episode_control_steps"]},
        "resources":{"time_measurements":measures,"rollout_cost":costs,"rollout_overhead_fraction":overhead,"extra_peak_rss_kib":extra_rss},
        "behavior":{"derivation":"persisted arrays -> production decoder/tracker -> independently collected event counters",
                    "signal_contract":recovered["signals"],"support_reference_point_order":recovered["support_names"],
                    "forbidden_contact_body_order":resolved.to_dict()["resolved_roles"]["forbidden_contact"],
                    **summarize_groups(rows),"episodes":rows},
        "boundaries":[
            "报告不是新评分/接受 gate，也不调用训练、simulator、GPU 或模型 API。",
            "slip/低clearance沿用任务 any-support -> global consecutive streak；不宣称同一脚连续或 toe 碰撞。",
            "身体 reference point 不等于脚中心/toe；保存的合力和终止 manager 不能定位接触对象/接触面。",
            "roll/pitch 来自 articulation-root quaternion；有符号姿态不是完整关节动作分类。",
            "事件顺序以控制步为精度；同一步保留并列。first_condition_step=1可能左删失；window区间可能在窗口前已开始。",
            "成功组与禁触组按同 cell、同 seed 分层，完整首回合分母保留；各自终局前一秒仅是终局对齐，不声称同动作阶段或因果干预效果。",
            "未到首立面的失败保留 not_reached；跨立面仅指 root x 坐标跨阈值，不证明支撑切换或成功上一级。",
            "RSS是/usr/bin/time对计时命令及子进程报告的max RSS指标差，不是各进程同时驻留总和或采集器独占内存；并行负载下单次wall-time差不等于稳定性能估计。",
            "本次public-dev评估通过不代表训练采样、其它协议/形态、媒体或完整CORE全部通过。"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("project-root", "on-dir", "off-dir", "output"):
        parser.add_argument("--"+key, type=Path, required=True)
    args = parser.parse_args(argv)
    sys.path.insert(0, str(args.project_root.resolve()/"src"))
    start = time.monotonic()
    try:
        report = verify(args)
    except Exception:
        report = {"schema_version":"feedback_p2.stair_pair_verification.v1", "status":"FAIL", "error":traceback.format_exc()}
    report["verification_wall_time_s"] = time.monotonic()-start
    report["verifier_sha256"] = sha256(__file__)
    report["project_root"] = str(args.project_root.resolve())
    report["decoder_source_sha256"] = {name:sha256(args.project_root/"src/true_rl_agent/tasks/stair_climbing"/name)
        for name in ("transition_decoder.py", "episode_evidence.py", "unified_scoring.py", "numeric_capture.py")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"status":report["status"],"output":str(args.output),"verification_wall_time_s":report["verification_wall_time_s"]}), flush=True)
    return 0 if report["status"] == "PASS" else 2 if report["status"] == "INCONCLUSIVE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
