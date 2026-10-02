#!/usr/bin/env python3
"""Audit a public Stair v2 archive before trajectory-translator development.

This is an offline input audit, not a behavior extractor or an outcome replay.
All chunks are read and checked; only valid first-episode rows are observations.
The stored history clock is retained, never upgraded from current runtime code.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from true_rl_agent.composition.cell_manifest import ResolvedCellManifestV1
from true_rl_agent.runner.numeric_evidence import (
    NUMERIC_EVIDENCE_REF_SCHEMA, NumericEvidenceError,
    iter_numeric_evidence, load_numeric_evidence_manifest,
)
from true_rl_agent.tasks.stair_climbing.evaluation_semantics import StairEvaluationSemantics
from true_rl_agent.tasks.stair_climbing.runtime_projection import stair_runtime_projection_from_manifest
from true_rl_agent.tasks.stair_climbing.unified_evaluation import StairUnifiedEvaluationSuite
from true_rl_agent.tasks.stair_climbing.unified_training import StairUnifiedTrainingProfile


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise NumericEvidenceError(message)


def _positive_int(value: Any, name: str) -> int:
    _require(type(value) is int and value > 0, f"{name} must be a positive integer")
    return value


def _positive_float(value: Any, name: str) -> float:
    _require(type(value) in (int, float) and math.isfinite(value) and value > 0,
             f"{name} must be finite and positive")
    return float(value)


def inspect_stair_trajectory_input(manifest_path: Path, manifest_sha256: str) -> dict[str, Any]:
    """Return reproducible input inventory; raise on corruption or contract drift."""
    reference = {"schema_version": NUMERIC_EVIDENCE_REF_SCHEMA,
                 "manifest_path": str(manifest_path), "manifest_sha256": manifest_sha256}
    manifest = load_numeric_evidence_manifest(reference)
    identity, context = manifest["identity"], manifest["metadata"]
    _require(identity["splits"] == ["public_dev"], "identity splits must be exactly public_dev")
    _require(context["profile"] == "stair.unified_first_episode.v2", "Stair v2 profile required")
    for name in ("job_id", "subject_kind", "subject_id", "checkpoint_hash", "seed_namespace"):
        _require(isinstance(identity[name], str) and bool(identity[name]), f"identity {name} is missing")
    resolved = ResolvedCellManifestV1.from_mapping(context["resolved_cell_manifest"])
    projection = stair_runtime_projection_from_manifest(resolved)
    _require(identity["task_id"] == resolved.task_id == "unitree_stair_climbing", "task identity mismatch")
    _require(identity["resolved_cell_manifest_hash"] == resolved.manifest_hash, "resolved manifest identity mismatch")
    _require(identity["evaluation_binding"]["task_protocol_hash"] == resolved.resolved_cell_task_protocol_hash,
             "resolved task protocol identity mismatch")
    _require(context["runtime_projection"] == projection.to_summary_dict(), "runtime projection/entity order mismatch")
    raw_semantics = context["semantics"]
    semantics = StairEvaluationSemantics.from_task_outcomes(
        task_id=raw_semantics["task_id"], task_protocol_hash=raw_semantics["task_protocol_hash"],
        episode_outcomes=raw_semantics["episode_outcomes"])
    _require(semantics.to_dict() == raw_semantics and raw_semantics["task_id"] == resolved.task_id
             and raw_semantics["task_protocol_hash"] == resolved.base_task_protocol_hash,
             "base task semantics mismatch")
    # This unified profile uses the same frozen mapping as the task decoder.
    manager_to_reason = {
        "time_out": "timeout",
        raw_semantics["episode_outcomes"]["signals"]["forbidden_contact"]["manager_term"]: "forbidden_contact",
    }
    training = StairUnifiedTrainingProfile.from_mapping(context["training_profile"])
    suite = StairUnifiedEvaluationSuite.from_mapping(context["evaluation_suite"], training_profile=training)
    cells = suite.private_cell_assignment("public_dev")
    env_count = len(cells)

    coordinates = {"positions": "world_m", "forces": "world_N", "velocities": "world_m_per_s",
                   "base_quat_w": "wxyz", "support_position": "bound_rigid_body_reference_point"}
    _require(context["coordinates"] == coordinates, "coordinate/unit declaration mismatch")
    _require(context["coverage"] == "every_control_step_of_each_scored_first_episode_including_pre_reset_terminal",
             "pre-reset first-episode coverage declaration mismatch")
    _require(context["invalid_padding"] == "valid=false; zero-filled numeric values are not observations",
             "padding declaration mismatch")
    history = context["contact_force_history"]
    _require(history["status"] == "recorded" and history["history_length"] == projection.contact_history_length,
             "contact history availability/length mismatch")
    for name, expected in {
        "schema_version": "true_rl_agent.stair.contact_force_history.v1",
        "source": "contact_sensor.net_forces_w_history",
        "force_semantics": "world_frame_net_normal_contact_force_N_not_contact_pair_or_surface",
        "history_order": "newest_to_oldest", "history_clock": "contact_sensor_buffer_update_events",
        "per_history_slot_timestamps": "not_recorded; configured_update_period_is_not_observed_sample_spacing",
        "coverage": "existing_sensor_buffer_at_pre_reset_control_step_end; not_full_physics_trajectory",
        "reset_history": "sensor_zeroes_history_on_reset; no_per_slot_validity_flags",
        "array_axes": ["control_step", "environment", "history", "bound_body", "xyz"],
    }.items():
        _require(history[name] == expected, f"contact history {name} mismatch")
    control_dt = _positive_float(history["control_dt_s"], "control dt")
    physics_dt = _positive_float(history["physics_dt_s"], "physics dt")
    sensor_dt = _positive_float(history["configured_sensor_update_period_s"], "sensor period")
    substeps = _positive_int(history["physics_steps_per_control"], "physics steps per control")
    _require(math.isclose(control_dt, substeps * physics_dt, rel_tol=1e-9)
             and math.isclose(sensor_dt, physics_dt, rel_tol=1e-9), "contact history clock mismatch")
    bound_control = resolved.robot_task_binding["task_projection"]["phase0_binding_facts"]["control"]
    _require(math.isclose(control_dt, bound_control["control_dt_s"], rel_tol=1e-9)
             and math.isclose(physics_dt, bound_control["simulation_dt_s"], rel_tol=1e-9)
             and substeps == bound_control["decimation"]
             and all(math.isclose(control_dt, cell.spec.control_dt_s, rel_tol=1e-9) for cell in cells),
             "history/control clock differs from frozen robot binding or evaluation suite")
    header_history = {key: value for key, value in history.items() if key not in {"status", "array_axes"}}

    seeds = context["seeds"]
    _require(isinstance(seeds, list) and bool(seeds), "no recorded seeds")
    lengths, runs, episodes = {}, [], []
    for run_index, seed in enumerate(seeds, 1):
        _require(seed["split"] == "public_dev", "seed split must be public_dev")
        _require(seed["run_index"] == run_index and type(seed["runner_seed"]) is int, "seed identity/order mismatch")
        _require(math.isclose(_positive_float(seed["step_dt"], "seed step dt"), control_dt, rel_tol=1e-9),
                 "seed/control history time mismatch")
        observed = _positive_int(seed["observed_control_steps"], "observed control steps")
        _require(len(seed["episode_steps"]) == len(seed["outcomes"]) == env_count, "seed environment count mismatch")
        values = [_positive_int(value, "episode steps") for value in seed["episode_steps"]]
        _require(max(values) == observed, "observed control steps/episode lengths mismatch")
        lengths[run_index] = np.asarray(values, dtype=np.int64)
        counts = Counter()
        for env_index, (steps, outcome, cell) in enumerate(zip(values, seed["outcomes"], cells, strict=True)):
            _require(outcome["episode_steps"] == steps, "outcome episode steps mismatch")
            _require(outcome["bin_id"] == cell.bin_id and outcome["evaluation_cell_id"] == cell.cell_id,
                     "outcome/public environment assignment mismatch")
            reason = outcome["primary_reason"]
            _require(reason in raw_semantics["episode_outcomes"]["primary_reasons"]
                     and type(outcome["success"]) is bool and outcome["success"] == (reason == "success"),
                     "outcome label mismatch")
            counts[reason] += 1
            episodes.append({"episode_id": f"r{run_index}.e{env_index}", "run_index": run_index,
                             "env_index": env_index, "runner_seed": seed["runner_seed"],
                             "cell_id": cell.cell_id, "stratum_id": cell.stratum_id,
                             "valid_control_steps": steps, "first_sample_time_s": control_dt,
                             "last_sample_time_s": steps * control_dt,
                             "recorded_outcome": outcome,
                             "right_boundary": "first_scored_outcome; subsequent behavior unobserved"})
        runs.append({"run_index": run_index, "runner_seed": seed["runner_seed"], "split": seed["split"],
                     "observed_control_steps": observed, "environment_count": env_count,
                     "valid_rows": sum(values), "padding_rows": observed * env_count - sum(values),
                     "recorded_outcome_counts": dict(sorted(counts.items()))})

    support, forbidden = projection.support_count, projection.forbidden_contact_count
    float_shapes = {"base_pos_w": (3,), "base_quat_w": (4,), "joint_pos": (projection.joint_count,),
                    "support_pos_w": (support, 3), "support_lin_vel_w": (support, 3),
                    "support_contact_forces_w": (support, 3), "forbidden_contact_forces_w": (forbidden, 3),
                    "support_contact_forces_w_history": (history["history_length"], support, 3),
                    "forbidden_contact_forces_w_history": (history["history_length"], forbidden, 3)}
    term_order = None
    run_steps = {index: 0 for index in lengths}
    current_run = 0
    inventory, anchors = [], []
    for chunk, (metadata, arrays) in zip(manifest["chunks"], iter_numeric_evidence(reference), strict=True):
        run = metadata["run_index"]
        _require(run in lengths and run in {current_run, current_run + 1}, "chunk run order mismatch")
        if run != current_run:
            if current_run:
                _require(run_steps[current_run] == seeds[current_run - 1]["observed_control_steps"], "incomplete run")
            current_run = run
        start = metadata["step_start"]
        steps = _positive_int(metadata["steps"], "chunk steps")
        _require(type(start) is int and start == run_steps[run] + 1, "control-step gap or overlap")
        first = run_steps[run] == 0
        _require(type(metadata["static_present"]) is bool and metadata["static_present"] == first,
                 "static data must appear once at run start")
        header = metadata["transition_header"]
        for name, expected in {
            "schema_version": "true_rl_agent.task.stair_climbing.transition.normalized.v1",
            "source_schema_version": "true_rl_agent.task.stair_climbing.transition.vnext.v1",
            "runtime_projection_hash": projection.projection_hash,
            "reward_input_schema_hash_vnext": projection.reward_input_schema_hash_vnext,
            "support_count": support, "forbidden_contact_count": forbidden,
            "contact_force_history": header_history,
        }.items():
            _require(header[name] == expected, f"transition header {name} mismatch")
        terms = metadata["termination_term_order"]
        _require(isinstance(terms, list) and bool(terms) and len(set(terms)) == len(terms)
                 and all(isinstance(term, str) and term for term in terms), "termination term order invalid")
        _require(set(terms) == set(manager_to_reason), "termination manager terms differ from frozen task mapping")
        if term_order is None:
            term_order = terms
        _require(terms == term_order, "termination entity order mismatch")
        shapes = {name: ((steps, env_count, *tail), "f") for name, tail in float_shapes.items()}
        shapes["episode_length"] = ((steps, env_count), "iu")
        shapes.update({name: ((steps, env_count), "b") for name in
                       ("valid", "done_mask", "wrapper_done_mask", *(f"termination_{i}" for i in range(len(terms))))})
        if first:
            shapes.update(terrain_env_origins_w=((env_count, 3), "f"), terrain_types=((env_count,), "iu"))
        _require(set(arrays) == set(shapes), "required channel fields mismatch")
        for name, (shape, kinds) in shapes.items():
            _require(arrays[name].shape == shape and arrays[name].dtype.kind in kinds, f"{name} shape/dtype mismatch")
        valid = arrays["valid"]
        indices = np.arange(start, start + steps)[:, None]
        _require(np.array_equal(valid, indices <= lengths[run][None, :]), "valid first-episode prefix mismatch")
        _require(np.array_equal(arrays["episode_length"][valid], np.broadcast_to(indices, valid.shape)[valid]),
                 "episode length reset or control-step mismatch")
        for name in float_shapes:
            _require(bool(np.isfinite(arrays[name][valid]).all()), f"non-finite observed {name}")
        norms = np.linalg.norm(arrays["base_quat_w"][valid].astype(np.float64), axis=-1)
        # P0's frozen input contract is stricter than the legacy decoder.
        _require(bool((np.abs(norms - 1.0) <= 1e-4).all()), "observed quaternion is not normalized")
        _require(np.array_equal(arrays["done_mask"][valid], arrays["wrapper_done_mask"][valid]), "wrapper done mismatch")
        term_union = np.logical_or.reduce([arrays[f"termination_{i}"][valid] for i in range(len(terms))])
        _require(np.array_equal(term_union, arrays["done_mask"][valid]), "done differs from OR(termination masks)")
        _require(not bool((arrays["done_mask"] & valid & (indices < lengths[run][None, :])).any()),
                 "done precedes first-episode boundary")
        if first:
            origins = arrays["terrain_env_origins_w"]
            _require(bool(np.isfinite(origins).all()) and np.allclose(origins,
                     np.asarray([cell.expected_origin_m for cell in cells]), rtol=0, atol=1e-4),
                     "terrain origin/environment order mismatch")
            _require(np.array_equal(arrays["terrain_types"], [cell.terrain_type_index for cell in cells]),
                     "terrain type/environment order mismatch")
            for env in range(min(env_count, 2)):
                anchors.append({"episode_id": f"r{run}.e{env}", "chunk_file": chunk["file"],
                                "array_index": [0, env], "control_step": 1, "time_s": control_dt,
                                "values": {name: arrays[name][0, env].tolist() for name in float_shapes},
                                "terrain_env_origin_w": origins[env].tolist()})
        raw_bytes = sum(array.nbytes for array in arrays.values())
        _require(raw_bytes == chunk["raw_bytes"], "chunk raw byte count mismatch")
        inventory.append({"file": chunk["file"], "sha256": chunk["sha256"],
                          "stored_bytes": chunk["stored_bytes"], "raw_bytes": raw_bytes,
                          "run_index": run, "step_start": start, "steps": steps,
                          "valid_rows": int(valid.sum()), "padding_rows": int(valid.size - valid.sum()),
                          "arrays": chunk["arrays"]})
        run_steps[run] += steps
    _require(all(run_steps[index] == seed["observed_control_steps"] for index, seed in enumerate(seeds, 1)),
             "incomplete run/seed coverage")
    _require(sum(item["stored_bytes"] for item in inventory) == manifest["stored_chunk_bytes"]
             and sum(item["raw_bytes"] for item in inventory) == manifest["raw_bytes"], "archive byte totals mismatch")

    return {
        "schema_version": "stair.trajectory_input_audit.v1", "status": "pass",
        "scope": "input integrity and declared coverage; no behavior or outcome recomputation",
        "source_reference": reference, "identity": identity, "profile": context["profile"],
        "runtime_projection": projection.to_summary_dict(), "coordinates": coordinates,
        "clock": {"control_dt_s": control_dt, "origin": "episode_reset; first recorded sample is step 1",
                  "sample_time_formula": "episode_length * control_dt_s",
                  "sample_phase": "pre_reset_control_step_end", "contact_force_history": history},
        "coverage": {"declaration": context["coverage"], "padding": context["invalid_padding"],
                     "episode_count": len(episodes), "valid_rows": sum(run["valid_rows"] for run in runs),
                     "padding_rows": sum(run["padding_rows"] for run in runs),
                     "post_outcome_observations": "not_collected", "reset_time_sample": "not_collected"},
        "capabilities": {
            "observed": {"world_position_m": ["base_pos_w", "support_pos_w", "terrain_env_origins_w"],
                         "unit_quaternion_wxyz": ["base_quat_w"], "joint_angle_rad": ["joint_pos"],
                         "world_velocity_m_per_s": ["support_lin_vel_w"],
                         "world_net_normal_force_N": [name for name in float_shapes if "forces" in name],
                         "discrete": ["episode_length", "valid", "done_mask", "wrapper_done_mask", "terrain_types"],
                         "termination_channels_in_order": term_order},
            "derivable": ["root displacement between recorded control samples", "root roll/pitch/yaw from wxyz",
                          "support reference point relative to root/terrain", "recorded force threshold predicates"],
            "missing": ["policy actions", "joint velocities", "actual torques", "reward terms",
                        "per-environment actual mass", "effective joint limits", "commands"],
            "unknown": ["contact-history slot timestamps and validity", "continuous physical-substep contact",
                        "contact pairs and surfaces", "contact friction force", "post-outcome behavior"],
        },
        "runs": runs, "episodes": episodes, "chunks": inventory, "raw_anchors": anchors,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True, help="New JSON audit file; existing files are not overwritten")
    args = parser.parse_args(argv)
    result = inspect_stair_trajectory_input(args.manifest, args.manifest_sha256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "output": str(args.output), **result["coverage"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
