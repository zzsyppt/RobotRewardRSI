"""Bind public Stair v2 archives to source-traceable numeric episodes.

The complete archive is checked before the first episode is exposed. Task
outcomes remain recorded authority; this adapter never runs an outcome tracker.
Numeric channel NaNs and raw quaternion norms are retained for the measurement
layer, whereas invalid source identity, clocks and entity bindings are rejected.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Any, Iterator

import numpy as np

from true_rl_agent.composition.cell_manifest import ResolvedCellManifestV1
from true_rl_agent.evaluation.trajectory_translation.contracts import NumericEpisode
from true_rl_agent.runner.numeric_evidence import (
    NumericEvidenceError, iter_numeric_evidence, load_numeric_evidence_manifest,
)
from true_rl_agent.tasks.stair_climbing.evaluation_semantics import StairEvaluationSemantics
from true_rl_agent.tasks.stair_climbing.runtime_projection import (
    StairRuntimeProjection, stair_runtime_projection_from_manifest,
)
from true_rl_agent.tasks.stair_climbing.unified_evaluation import StairUnifiedEvaluationSuite
from true_rl_agent.tasks.stair_climbing.unified_training import StairUnifiedTrainingProfile


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise NumericEvidenceError(message)


def _fields(value: Any, required: set[str], name: str) -> None:
    _require(isinstance(value, dict), f"{name} must be an object")
    _require(required <= value.keys(), f"{name} missing required fields: {sorted(required - value.keys())}")


def _positive_int(value: Any, name: str) -> int:
    _require(type(value) is int and value > 0, f"{name} must be a positive integer")
    return value


def _positive_float(value: Any, name: str) -> float:
    _require(type(value) in (int, float) and math.isfinite(value) and value > 0,
             f"{name} must be finite and positive")
    return float(value)


@dataclass
class _Archive:
    manifest: dict[str, Any]
    projection: StairRuntimeProjection
    cells: tuple
    lengths: dict[int, np.ndarray]
    float_shapes: dict[str, tuple[int, ...]]
    header_history: dict[str, Any]
    manager_terms: set[str]
    control_dt: float


def _load_contract(reference: dict) -> _Archive:
    manifest = load_numeric_evidence_manifest(reference)
    _fields(manifest, {"identity", "metadata", "stored_chunk_bytes", "raw_bytes"}, "manifest")
    identity, context = manifest["identity"], manifest["metadata"]
    _fields(identity, {"job_id", "subject_kind", "subject_id", "checkpoint_hash", "seed_namespace",
                       "splits", "task_id", "resolved_cell_manifest_hash", "evaluation_binding"}, "identity")
    _fields(context, {"profile", "resolved_cell_manifest", "runtime_projection", "semantics",
                      "training_profile", "evaluation_suite", "coordinates", "coverage", "invalid_padding",
                      "contact_force_history", "seeds"}, "Stair v2 metadata")
    _require(identity["splits"] == ["public_dev"], "identity splits must be exactly public_dev")
    _require(context["profile"] == "stair.unified_first_episode.v2", "Stair v2 profile required")
    for name in ("job_id", "subject_kind", "subject_id", "checkpoint_hash", "seed_namespace"):
        _require(isinstance(identity[name], str) and bool(identity[name]), f"identity {name} is missing")
    resolved = ResolvedCellManifestV1.from_mapping(context["resolved_cell_manifest"])
    projection = stair_runtime_projection_from_manifest(resolved)
    _require(identity["task_id"] == resolved.task_id == "unitree_stair_climbing", "task identity mismatch")
    _require(identity["resolved_cell_manifest_hash"] == resolved.manifest_hash, "resolved manifest identity mismatch")
    _fields(identity["evaluation_binding"], {"task_protocol_hash"}, "evaluation binding")
    _require(identity["evaluation_binding"]["task_protocol_hash"] == resolved.resolved_cell_task_protocol_hash,
             "resolved task protocol identity mismatch")
    _require(context["runtime_projection"] == projection.to_summary_dict(), "runtime projection/entity order mismatch")
    raw_semantics = context["semantics"]
    _fields(raw_semantics, {"task_id", "task_protocol_hash", "episode_outcomes"}, "semantics")
    semantics = StairEvaluationSemantics.from_task_outcomes(
        task_id=raw_semantics["task_id"], task_protocol_hash=raw_semantics["task_protocol_hash"],
        episode_outcomes=raw_semantics["episode_outcomes"])
    _require(semantics.to_dict() == raw_semantics and raw_semantics["task_id"] == resolved.task_id
             and raw_semantics["task_protocol_hash"] == resolved.base_task_protocol_hash,
             "base task semantics mismatch")
    signals = raw_semantics["episode_outcomes"]["signals"]
    manager_terms = {"time_out", signals["forbidden_contact"]["manager_term"]}
    _positive_float(signals["slip"]["contact_force_threshold"], "contact force threshold")
    training = StairUnifiedTrainingProfile.from_mapping(context["training_profile"])
    suite = StairUnifiedEvaluationSuite.from_mapping(context["evaluation_suite"], training_profile=training)
    cells = suite.private_cell_assignment("public_dev")
    coordinates = {"positions": "world_m", "forces": "world_N", "velocities": "world_m_per_s",
                   "base_quat_w": "wxyz", "support_position": "bound_rigid_body_reference_point"}
    _require(context["coordinates"] == coordinates, "coordinate/unit declaration mismatch")
    _require(context["coverage"] == "every_control_step_of_each_scored_first_episode_including_pre_reset_terminal",
             "pre-reset first-episode coverage declaration mismatch")
    _require(context["invalid_padding"] == "valid=false; zero-filled numeric values are not observations",
             "padding declaration mismatch")
    history = context["contact_force_history"]
    expected_history = {
        "schema_version": "true_rl_agent.stair.contact_force_history.v1", "status": "recorded",
        "history_length": projection.contact_history_length,
        "source": "contact_sensor.net_forces_w_history",
        "force_semantics": "world_frame_net_normal_contact_force_N_not_contact_pair_or_surface",
        "history_order": "newest_to_oldest", "history_clock": "contact_sensor_buffer_update_events",
        "per_history_slot_timestamps": "not_recorded; configured_update_period_is_not_observed_sample_spacing",
        "coverage": "existing_sensor_buffer_at_pre_reset_control_step_end; not_full_physics_trajectory",
        "reset_history": "sensor_zeroes_history_on_reset; no_per_slot_validity_flags",
        "array_axes": ["control_step", "environment", "history", "bound_body", "xyz"],
    }
    _fields(history, set(expected_history) | {"control_dt_s", "physics_dt_s", "configured_sensor_update_period_s",
                                             "physics_steps_per_control", "lazy_sensor_update"}, "contact history")
    for name, expected in expected_history.items():
        _require(history[name] == expected, f"contact history {name} mismatch")
    _require(type(history["lazy_sensor_update"]) is bool, "contact history lazy sensor update must be boolean")
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
    seeds = context["seeds"]
    _require(isinstance(seeds, list) and bool(seeds), "no recorded seeds")
    lengths = {}
    for run, seed in enumerate(seeds, 1):
        _fields(seed, {"split", "run_index", "runner_seed", "step_dt", "observed_control_steps", "episode_steps", "outcomes"},
                "seed")
        _require(seed["split"] == "public_dev", "seed split must be public_dev")
        _require(type(seed["run_index"]) is int and seed["run_index"] == run
                 and type(seed["runner_seed"]) is int, "seed identity/order mismatch")
        _require(math.isclose(_positive_float(seed["step_dt"], "seed step dt"), control_dt, rel_tol=1e-9),
                 "seed/control history time mismatch")
        observed = _positive_int(seed["observed_control_steps"], "observed control steps")
        _require(len(seed["episode_steps"]) == len(seed["outcomes"]) == len(cells), "seed environment count mismatch")
        values = [_positive_int(value, "episode steps") for value in seed["episode_steps"]]
        _require(max(values) == observed, "observed control steps/episode lengths mismatch")
        lengths[run] = np.asarray(values, dtype=np.int64)
        for steps, outcome, cell in zip(values, seed["outcomes"], cells, strict=True):
            _fields(outcome, {"episode_steps", "bin_id", "evaluation_cell_id", "primary_reason", "success"}, "outcome")
            _require(type(outcome["episode_steps"]) is int and outcome["episode_steps"] == steps,
                     "outcome episode steps mismatch")
            _require(outcome["bin_id"] == cell.bin_id and outcome["evaluation_cell_id"] == cell.cell_id,
                     "outcome/public environment assignment mismatch")
            reason = outcome["primary_reason"]
            _require(reason in raw_semantics["episode_outcomes"]["primary_reasons"]
                     and type(outcome["success"]) is bool and outcome["success"] == (reason == "success"),
                     "outcome label mismatch")
    support, forbidden = projection.support_count, projection.forbidden_contact_count
    float_shapes = {"base_pos_w": (3,), "base_quat_w": (4,), "joint_pos": (projection.joint_count,),
                    "support_pos_w": (support, 3), "support_lin_vel_w": (support, 3),
                    "support_contact_forces_w": (support, 3), "forbidden_contact_forces_w": (forbidden, 3),
                    "support_contact_forces_w_history": (history["history_length"], support, 3),
                    "forbidden_contact_forces_w_history": (history["history_length"], forbidden, 3)}
    return _Archive(manifest, projection, cells, lengths, float_shapes,
                    {key: value for key, value in history.items() if key not in {"status", "array_axes"}},
                    manager_terms, control_dt)


def _validate_chunks(reference: dict, archive: _Archive) -> tuple[str, ...]:
    """Read every byte-bound chunk, including seeds later than the first yield."""
    manifest, projection, lengths = archive.manifest, archive.projection, archive.lengths
    seeds = manifest["metadata"]["seeds"]
    env_count = len(archive.cells)
    term_order = None
    dynamic_dtypes = None
    run_steps = dict.fromkeys(lengths, 0)
    current_run, raw_total, stored_total = 0, 0, 0
    for chunk, (metadata, arrays) in zip(manifest["chunks"], iter_numeric_evidence(reference), strict=True):
        _fields(metadata, {"run_index", "step_start", "steps", "static_present", "transition_header",
                           "termination_term_order"}, "chunk metadata")
        run = metadata["run_index"]
        _require(type(run) is int and run in lengths and run in {current_run, current_run + 1}, "chunk run order mismatch")
        if run != current_run:
            if current_run:
                _require(run_steps[current_run] == seeds[current_run - 1]["observed_control_steps"], "incomplete run")
            current_run = run
        start, steps = metadata["step_start"], _positive_int(metadata["steps"], "chunk steps")
        _require(type(start) is int and start == run_steps[run] + 1, "control-step gap or overlap")
        first = run_steps[run] == 0
        _require(type(metadata["static_present"]) is bool and metadata["static_present"] == first,
                 "static data must appear once at run start")
        expected_header = {
            "schema_version": "true_rl_agent.task.stair_climbing.transition.normalized.v1",
            "source_schema_version": "true_rl_agent.task.stair_climbing.transition.vnext.v1",
            "runtime_projection_hash": projection.projection_hash,
            "reward_input_schema_hash_vnext": projection.reward_input_schema_hash_vnext,
            "support_count": projection.support_count, "forbidden_contact_count": projection.forbidden_contact_count,
            "contact_force_history": archive.header_history,
        }
        _fields(metadata["transition_header"], set(expected_header), "transition header")
        for name, expected in expected_header.items():
            _require(metadata["transition_header"][name] == expected, f"transition header {name} mismatch")
        terms = metadata["termination_term_order"]
        _require(isinstance(terms, list) and bool(terms) and all(isinstance(term, str) and term for term in terms)
                 and len(set(terms)) == len(terms), "termination term order invalid")
        _require(set(terms) == archive.manager_terms, "termination manager terms differ from frozen task mapping")
        if term_order is None:
            term_order = tuple(terms)
        _require(tuple(terms) == term_order, "termination entity order mismatch")
        shapes = {name: ((steps, env_count, *tail), "f") for name, tail in archive.float_shapes.items()}
        shapes["episode_length"] = ((steps, env_count), "iu")
        shapes.update({name: ((steps, env_count), "b") for name in
                       ("valid", "done_mask", "wrapper_done_mask", *(f"termination_{i}" for i in range(len(terms))))})
        if first:
            shapes.update(terrain_env_origins_w=((env_count, 3), "f"), terrain_types=((env_count,), "iu"))
        _require(set(arrays) == set(shapes), "required channel fields mismatch")
        for name, (shape, kinds) in shapes.items():
            _require(arrays[name].shape == shape and arrays[name].dtype.kind in kinds, f"{name} shape/dtype mismatch")
        observed_dtypes = {name: value.dtype for name, value in arrays.items()
                           if name not in {"terrain_env_origins_w", "terrain_types"}}
        if dynamic_dtypes is None:
            dynamic_dtypes = observed_dtypes
        _require(observed_dtypes == dynamic_dtypes, "dynamic channel dtype changed between chunks")
        valid, indices = arrays["valid"], np.arange(start, start + steps)[:, None]
        _require(np.array_equal(valid, indices <= lengths[run][None, :]), "valid first-episode prefix mismatch")
        _require(np.array_equal(arrays["episode_length"][valid], np.broadcast_to(indices, valid.shape)[valid]),
                 "episode length reset or control-step mismatch")
        _require(np.array_equal(arrays["done_mask"][valid], arrays["wrapper_done_mask"][valid]), "wrapper done mismatch")
        term_union = np.logical_or.reduce([arrays[f"termination_{i}"][valid] for i in range(len(terms))])
        _require(np.array_equal(term_union, arrays["done_mask"][valid]), "done differs from OR(termination masks)")
        _require(not bool((arrays["done_mask"] & valid & (indices < lengths[run][None, :])).any()),
                 "done precedes first-episode boundary")
        if first:
            origins = arrays["terrain_env_origins_w"]
            _require(bool(np.isfinite(origins).all()) and np.allclose(origins,
                     np.asarray([cell.expected_origin_m for cell in archive.cells]), rtol=0, atol=1e-4),
                     "terrain origin/environment order mismatch")
            _require(np.array_equal(arrays["terrain_types"], [cell.terrain_type_index for cell in archive.cells]),
                     "terrain type/environment order mismatch")
        raw_bytes = sum(array.nbytes for array in arrays.values())
        _require(raw_bytes == chunk["raw_bytes"], "chunk raw byte count mismatch")
        raw_total += raw_bytes
        stored_total += chunk["stored_bytes"]
        run_steps[run] += steps
    _require(all(run_steps[run] == seed["observed_control_steps"] for run, seed in enumerate(seeds, 1)),
             "incomplete run/seed coverage")
    _require(stored_total == manifest["stored_chunk_bytes"] and raw_total == manifest["raw_bytes"],
             "archive byte totals mismatch")
    return term_order


def iter_stair_episodes(reference: dict, *, source_id: str) -> Iterator[NumericEpisode]:
    """Yield all public first episodes, in run/environment order, after source audit.

    The second read assembles only valid rows for one run at a time. No P0
    selection, oracle, decoder replay, or behavior aggregation participates.
    """
    _require(isinstance(source_id, str) and bool(source_id), "source_id must be nonempty")
    archive = _load_contract(reference)
    term_order = _validate_chunks(reference, archive)
    manifest, projection = archive.manifest, archive.projection
    identity, context = manifest["identity"], manifest["metadata"]
    source_chunks = tuple(deepcopy(manifest["chunks"]))
    common_source = {
        "source_id": source_id, "manifest_sha256": reference["manifest_sha256"],
        "manifest_path": reference["manifest_path"], "trial_id": identity["job_id"],
        "subject_kind": identity["subject_kind"], "subject_id": identity["subject_id"],
        "checkpoint_hash": identity["checkpoint_hash"], "task_id": identity["task_id"],
        "acceptance_split": "public_dev", "seed_namespace": identity["seed_namespace"],
        "profile": context["profile"], "resolved_cell_manifest_hash": identity["resolved_cell_manifest_hash"],
        "runtime_projection": context["runtime_projection"],
        "runtime_projection_hash": projection.projection_hash,
        "evaluation_binding": identity["evaluation_binding"], "semantics_hash": context["semantics"]["protocol_hash"],
        "coordinates": context["coordinates"], "contact_force_history": context["contact_force_history"],
        "public_dev_provenance": {"identity_splits": identity["splits"],
                                  "seed_splits": [seed["split"] for seed in context["seeds"]],
                                  "validation": "all_declared_chunks_and_public_seed_assignments"},
        "coverage": {"declaration": context["coverage"], "padding": context["invalid_padding"],
                     "sample_phase": "pre_reset_control_step_end", "reset_time_sample": "not_collected",
                     "post_outcome_observations": "not_collected"},
        "support_reference_points": {
            "position": "bound_rigid_body_reference_point",
            "velocity": "historical_link_or_COM_reference_mapping_not_recorded",
            "position_velocity_derivative_equivalence": "unavailable"},
    }
    entities = {"root": ("robot_0.root",),
                "support": tuple(f"robot_0.support.{name}" for name in projection.support_body_names),
                "forbidden": tuple(f"robot_0.body.{name}" for name in projection.forbidden_contact_body_names),
                "joints": projection.joint_names}
    current_run, assembled = 0, []
    for chunk_index, (metadata, values) in enumerate(iter_numeric_evidence(reference)):
        run = metadata["run_index"]
        if run != current_run:
            yield from assembled
            current_run, assembled = run, []
            seed = context["seeds"][run - 1]
            dynamic_names = sorted(set(values) - {"terrain_env_origins_w", "terrain_types"})
            for env, (length, cell) in enumerate(zip(archive.lengths[run], archive.cells, strict=True)):
                arrays = {name: np.empty((length, *values[name].shape[2:]), dtype=values[name].dtype)
                          for name in dynamic_names}
                source = {**deepcopy(common_source), "runner_seed": seed["runner_seed"]}
                chunk = source_chunks[chunk_index]
                task = {"outcome": deepcopy(seed["outcomes"][env]),
                        "outcome_authority": {"kind": "recorded_task_outcome", "semantics_hash": source["semantics_hash"],
                                              "metadata_path": f"seeds[{run - 1}].outcomes[{env}]"},
                        "origin_world": values["terrain_env_origins_w"][env].tolist(),
                        "progress_direction_world": [1.0, 0.0, 0.0], "goal_progress_m": cell.progress_goal_m,
                        "deadline_s": cell.episode_horizon_s, "stratum_id": cell.stratum_id, "cell_id": cell.cell_id,
                        "termination_term_order": list(term_order),
                        "contact_threshold_n": context["semantics"]["episode_outcomes"]["signals"]["slip"]["contact_force_threshold"],
                        "static_refs": {"terrain_env_origins_w": {"file": chunk["file"], "sha256": chunk["sha256"],
                                                                  "array": "terrain_env_origins_w", "array_index": [env]}}}
                # Populate after construction: positive monotone steps and valid
                # locators are known from the complete first pass.
                assembled.append(NumericEpisode(source, run, env, np.arange(1, length + 1), seed["step_dt"],
                    arrays, dict(entities), source_chunks, np.zeros(length, dtype=np.int64),
                    np.zeros(length, dtype=np.int64), task))
        start = metadata["step_start"] - 1
        for episode in assembled:
            stop = min(start + metadata["steps"], len(episode.steps))
            if start >= stop:
                continue
            for name, array in episode.arrays.items():
                array[start:stop] = values[name][:stop - start, episode.env_index]
            episode.chunk_indices[start:stop] = chunk_index
            episode.chunk_rows[start:stop] = np.arange(stop - start)
            episode.steps[start:stop] = values["episode_length"][:stop - start, episode.env_index]
    yield from assembled
