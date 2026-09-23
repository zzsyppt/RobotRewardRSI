"""Push first-episode facts and an offline reducer, independent of runtime.observe.

The six pose values are the native-dtype operands of the existing task tests;
v2 also preserves their signed world-XY motion operands from the same observation.
Force delivery remains the existing physics-hook
measurement; this profile does not claim to reconstruct unrecorded substep forces.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import numpy as np

from true_rl_agent.runner.numeric_evidence import (
    NumericEvidenceError, NumericEvidenceWriter, iter_numeric_evidence,
    load_numeric_evidence_manifest,
)
from true_rl_agent.tasks.push_recovery.failure_classifier import (
    classify_force_recovery_episode, classify_recovery_episode,
)

PUSH_NUMERIC_SCHEMA = "true_rl_agent.push_recovery.numeric.v2"
PUSH_NUMERIC_V1_SCHEMA = "true_rl_agent.push_recovery.numeric.v1"
POSE_FIELDS = ("height_m", "roll_rad", "pitch_rad", "planar_speed_mps", "angular_speed_radps", "xy_drift_m")
PLANAR_MOTION_FIELDS = ("root_lin_vel_w_x_mps", "root_lin_vel_w_y_mps", "reset_relative_root_pos_w_x_m", "reset_relative_root_pos_w_y_m")
CONDITIONS = ("planar_speed", "angular_speed", "roll", "pitch", "height")


@dataclass(frozen=True)
class PushNumericCaptureOptions:
    root: Path
    identity: Mapping[str, Any]
    max_bytes: int = 2 * 1024**3
    chunk_steps: int = 128
    max_buffer_bytes: int = 64 * 1024**2


def push_numeric_capture_options_from_args(args: Any) -> PushNumericCaptureOptions | None:
    """Build capture identity only from the evaluator's already validated inputs."""
    root = getattr(args, "numeric_evidence_dir", None)
    if root is None:
        return None
    manifest = args.resolved_cell_manifest
    if args.evaluation_phase != "public_dev" or manifest is None or manifest.cell_variant_id not in {
        f"push.{robot}.recovery_{version}" for robot in ("g1_37dof", "go2_12dof") for version in ("v2", "v3")
    }:
        raise ValueError("Push numeric capture requires a public_dev phase of a physical G1/Go2 recovery cell")
    if args.numeric_evidence_max_bytes <= 0:
        raise ValueError("Push numeric evidence byte budget must be positive")
    identity = {**args.evaluation_job_identity.to_public_dict(),
        "evaluation_binding": args.evaluation_binding.to_public_dict(), "evaluation_plan_hash": args.evaluation_plan_hash,
        "checkpoint_hash": args.checkpoint_sha256, "resolved_cell_manifest_hash": manifest.manifest_hash,
        "runtime_dynamics_hash": args.runtime_dynamics_hash, "program_source_checksum": args.program_source_checksum,
        "training_binding_hash": args.training_binding_hash,
        "backend_execution_environment_verification": args.backend_execution_environment_verification}
    return PushNumericCaptureOptions(Path(root), identity, max_bytes=args.numeric_evidence_max_bytes)


def validate_push_numeric_reference(reference: Mapping[str, Any], summary: Mapping[str, Any], *,
    expected_identity: Mapping[str, Any]) -> None:
    """Bind a new archive to its seed/phase before sealing task_evidence."""
    manifest = load_numeric_evidence_manifest(reference)
    identity, metadata = manifest["identity"], manifest["metadata"]
    schema = metadata.get("schema_version")
    if schema not in {PUSH_NUMERIC_V1_SCHEMA, PUSH_NUMERIC_SCHEMA}:
        raise NumericEvidenceError("not a supported Push numeric archive")
    expected = {**expected_identity, "task_numeric_schema": schema,
        "split": summary["run"]["split"], "run_index": summary["run"]["run_index"],
        "resolved_protocol_hash": summary["evidence"]["resolved_protocol_hash"], "mechanism": summary["task"]["mechanism"]}
    if expected["split"] != "public_dev" or any(identity.get(key) != value for key, value in expected.items()):
        raise NumericEvidenceError("Push numeric reference belongs to another phase, policy, or seed")
    if (metadata.get("data_complete") is not True
        or metadata.get("runtime_episode_records") != summary["per_episode"]
        or metadata.get("num_envs") != summary["environment"]["num_envs"]
        or metadata.get("episode_coverage") != ("complete" if summary["evidence"]["all_first_episodes_closed"] else "truncated")):
        raise NumericEvidenceError("Push numeric reference outcomes or coverage differ from the seed summary")


class PushNumericCapture:
    """Bounded device staging, synchronously flushed before manifest publication."""

    def __init__(self, options: PushNumericCaptureOptions, state: Any, *, split_name: str, run_index: int) -> None:
        if not state.evaluation:
            raise ValueError("Push numeric first-episode capture requires evaluation")
        if options.chunk_steps <= 0 or options.max_buffer_bytes <= 0:
            raise ValueError("Push numeric capture budgets must be positive")
        self.options, self.state = options, state
        self.force = hasattr(state, "sequence_dwell")
        self.buffer: dict[str, Any] = {}
        self.used = self.frames = self.buffer_bytes = 0
        self.capture_seconds = 0.0
        identity = dict(options.identity)
        owned = {"task_numeric_schema": PUSH_NUMERIC_SCHEMA, "split": split_name, "run_index": run_index,
            "resolved_protocol_hash": state.parameters["resolved_protocol_hash"],
            "mechanism": state.parameters["mechanism"]}
        if any(key in identity and identity[key] != value for key, value in owned.items()):
            raise ValueError("Push numeric identity disagrees with the running task")
        identity.update(owned)
        self.writer = NumericEvidenceWriter(Path(options.root) / f"{split_name}-{run_index:02d}", identity, options.max_bytes)
        self.definition = {
            "schema_version": PUSH_NUMERIC_SCHEMA, "profile": "first_episode_stability",
            "capture_point": "after_physics_after_task_observe_before_termination_and_reset",
            "time": {"step_dt_s": state.dt, "physics_dt_s": float(state.env.physics_dt) if hasattr(state.env, "physics_dt") else None,
                "control_step": "episode_length_buf; episode-relative; time_s=control_step*step_dt_s",
                "first_sample_step": 1, "recovery_window_steps": state.window, "required_dwell_steps": state.required_dwell,
                "event_scheduled_steps": list(state.event_times), "max_episode_steps": int(state.env.max_episode_length)},
            "pose_fields": list(POSE_FIELDS), "condition_fields": list(CONDITIONS),
            "planar_motion_fields": list(PLANAR_MOTION_FIELDS), "direction_status": "recorded_world_xy",
            "dwell_scope": "eligible post-anchor steps (0 < elapsed <= window), latched at first recovery; force anchor is last pulse end",
            "coordinates": {"height": "root_pos_w.z - env_origin_w.z", "xy_drift": "norm(root_pos_w.xy - reset_root_pos_w.xy)",
                "roll_pitch": "signed Euler roll/pitch from root quaternion converted to wxyz",
                "planar_speed": "norm(root_lin_vel_w.xy)", "angular_speed": "norm(root_ang_vel_w.xyz)",
                "planar_motion": "root_lin_vel_w.xy [m/s], root_pos_w.xy - reset_root_pos_w.xy [m]; signed world axes; same observation as pose norms"},
            "parameters": dict(state.parameters), "num_envs": state.n,
            "control_only": state.control_only, "planned_push_counts": state.planned.detach().cpu().tolist(),
            "strata": state.stratum.detach().cpu().tolist(), "episode_indices": state.episode_index.detach().cpu().tolist(),
            "media_status": "not_collected_by_numeric_profile", "contact_status": "not_collected",
        }
        if self.force:
            self.definition["force"] = {"body_name": state.body_name, "body_index": state.body_index,
                "pulse_steps": state.pulse_steps, "expected_substeps": state.expected_substeps,
                "decimation": state.decimation, "total_mass_kg": state.total_mass.detach().cpu().tolist(),
                "delivery_scope": "existing physics-hook counters, verification and event impulse; no raw substep archive"}

    def record(self, state: Any, failure: Any, active: Any) -> None:
        import torch
        started = perf_counter()
        values = {"pose": state._numeric_pose, "conditions": state._numeric_conditions,
            "planar_motion": state._numeric_planar_motion,
            "control_step": state.steps, "episode_index": state.episode_index, "valid": active,
            "failure": failure, "timeout": state.steps >= state.env.max_episode_length,
            "event_steps": state.event_steps, "event_verified": state.verified}
        if self.force:
            values.update(event_ends=state.event_ends, actual_substeps=state.actual_substeps,
                unplanned_wrench=state.unplanned_wrench, observed_dwell=state.sequence_dwell[:, None],
                observed_recovery_step=state.sequence_recovery_step[:, None])
        else:
            values.update(observed_dwell=state.dwell, observed_recovery_step=state.recovered_step)
        if not self.buffer:
            self.buffer_bytes = sum(value.numel() * value.element_size() for value in values.values()) * self.options.chunk_steps
            # Account for device staging plus its host copy; codec/RSS is measured separately.
            if 2 * self.buffer_bytes > self.options.max_buffer_bytes:
                raise NumericEvidenceError("Push numeric staging exceeds buffer budget")
            self.buffer = {name: torch.empty((self.options.chunk_steps, *value.shape), dtype=value.dtype, device=value.device)
                for name, value in values.items()}
        for name, value in values.items():
            self.buffer[name][self.used].copy_(value)
        self.used += 1
        self.frames += 1
        if self.used == self.options.chunk_steps:
            self._flush()
        self.capture_seconds += perf_counter() - started

    def _flush(self) -> None:
        if not self.used:
            return
        arrays = {name: value[:self.used].detach().cpu().numpy() for name, value in self.buffer.items()}
        self.writer.write_chunk(arrays, {"frame_start": self.frames - self.used, "frame_count": self.used})
        self.used = 0

    def complete(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        started = perf_counter()
        self._flush()
        self.capture_seconds += perf_counter() - started
        closed = self.state.closed.detach().cpu().tolist()
        return self.writer.complete({**self.definition, "data_complete": True,
            "episode_coverage": "complete" if all(closed) else "truncated", "closed": closed,
            "frames": self.frames, "staging_raw_bytes": self.buffer_bytes,
            "capture_seconds": self.capture_seconds, "runtime_episode_records": records})


def _conditions(pose: np.ndarray, parameters: Mapping[str, Any]) -> np.ndarray:
    # Torch scalar comparison rounds the scalar to the tensor dtype.
    def threshold(name: str) -> np.ndarray:
        return np.asarray(parameters[name], dtype=pose.dtype)
    return np.stack((pose[:, 3] <= threshold("recovery_velocity_mps"),
        pose[:, 4] <= threshold("recovery_angular_velocity_radps"),
        np.abs(pose[:, 1]) <= threshold("recovery_tilt_rad"), np.abs(pose[:, 2]) <= threshold("recovery_tilt_rad"),
        pose[:, 0] >= threshold("recovery_height_m")), axis=1)


def recompute_push_numeric_evidence(reference: Mapping[str, Any]) -> dict[str, Any]:
    """Stream persisted facts; independently reconstruct runtime observations.

    Partial episodes have a snapshot comparison but no reconstructed terminal label.
    Event injection verification is an input measurement, not independently measured
    again by this reducer. Terminal labels use the task's shared classifiers over
    reconstructed facts. Invalid/missing coverage fails closed.
    """
    manifest = load_numeric_evidence_manifest(reference)
    d = manifest["metadata"]
    schema = d.get("schema_version")
    if schema not in {PUSH_NUMERIC_V1_SCHEMA, PUSH_NUMERIC_SCHEMA} or manifest["identity"].get("task_numeric_schema") != schema:
        raise NumericEvidenceError("not a Push numeric archive")
    has_direction = schema == PUSH_NUMERIC_SCHEMA
    if (d.get("pose_fields") != list(POSE_FIELDS) or d.get("condition_fields") != list(CONDITIONS)
        or (has_direction and (d.get("planar_motion_fields") != list(PLANAR_MOTION_FIELDS)
                              or d.get("direction_status") != "recorded_world_xy"))):
        raise NumericEvidenceError("Push numeric field definitions mismatch")
    if d.get("data_complete") is not True or d["parameters"]["resolved_protocol_hash"] != manifest["identity"]["resolved_protocol_hash"]:
        raise NumericEvidenceError("Push numeric definition or identity mismatch")
    p, timing = d["parameters"], d["time"]
    n, force = d["num_envs"], "force" in d
    expected_events = [math.ceil(value / timing["step_dt_s"]) for value in p["push_times_s"]]
    if force:
        pulse = math.ceil(p["pulse_duration_s"] / timing["step_dt_s"])
        expected_events = [expected_events[0], expected_events[0] + pulse + math.ceil(p["repeated_gap_s"] / timing["step_dt_s"])]
        if d["force"]["pulse_steps"] != pulse or d["force"]["expected_substeps"] != pulse * d["force"]["decimation"]:
            raise NumericEvidenceError("Push numeric force quantization mismatch")
    if (timing["recovery_window_steps"] != math.ceil(p["recovery_window_s"] / timing["step_dt_s"])
        or timing["required_dwell_steps"] != math.ceil(p["stable_dwell_s"] / timing["step_dt_s"])
        or timing["event_scheduled_steps"] != expected_events):
        raise NumericEvidenceError("Push numeric time quantization mismatch")
    slots = 1 if force else 2
    planned = np.asarray(d["planned_push_counts"], dtype=np.int64)
    dwell = np.zeros((n, slots), dtype=np.int64)
    max_dwell = np.zeros_like(dwell)
    recovered = np.full((n, slots), -1, dtype=np.int64)
    seen, last_step = np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.int64)
    terminal = np.zeros(n, dtype=bool)
    minimum, maximum = np.full((n, 6), np.inf), np.full((n, 6), -np.inf)
    max_roll = np.zeros(n)
    final_pose, final_stable = np.zeros((n, 6)), np.zeros(n, dtype=bool)
    window_seen, stable_seen = np.zeros_like(dwell), np.zeros_like(dwell)
    failed_condition_steps = np.zeros((n, slots, 5), dtype=np.int64)
    interruptions: list[list[list[dict[str, Any]]]] = [[[] for _ in range(slots)] for _ in range(n)]
    last_event = np.full((n, 2), -1, dtype=np.int64)
    last_verified = np.zeros((n, 2), dtype=bool)
    last_complete = np.zeros((n, 2), dtype=bool)
    unplanned = np.zeros(n, dtype=bool)
    frames = 0
    checks = {"conditions": True, "termination": True, "dwell": True, "recovery": True}
    if has_direction:
        checks["planar_motion_norms"] = True
    for chunk, arrays in iter_numeric_evidence(reference):
        count = chunk.get("frame_count")
        if chunk.get("frame_start") != frames or not isinstance(count, int) or count <= 0:
            raise NumericEvidenceError("Push numeric frame order or count mismatch")
        required = {"pose": (n, 6), "conditions": (n, 5), "control_step": (n,), "episode_index": (n,),
            "valid": (n,), "failure": (n,), "timeout": (n,), "event_steps": (n, 2), "event_verified": (n, 2),
            "observed_dwell": (n, slots), "observed_recovery_step": (n, slots)}
        if has_direction:
            required["planar_motion"] = (n, 4)
        if force:
            required.update(event_ends=(n, 2), actual_substeps=(n, 2), unplanned_wrench=(n,))
        if set(arrays) != set(required) or any(arrays[key].shape != (count, *shape) for key, shape in required.items()):
            raise NumericEvidenceError("Push numeric fields or dimensions mismatch")
        for row in range(count):
            a = {key: value[row] for key, value in arrays.items()}
            valid, step, pose = a["valid"], a["control_step"], a["pose"]
            if valid.dtype.kind != "b" or not np.isfinite(pose[valid]).all():
                raise NumericEvidenceError("Push numeric valid samples must be finite")
            if has_direction:
                motion = a["planar_motion"]
                if motion.dtype != pose.dtype or motion.dtype.kind != "f" or not np.isfinite(motion[valid]).all():
                    raise NumericEvidenceError("Push numeric planar motion must retain finite pose-dtype operands")
                # NumPy and the device's Torch norm may round their reductions
                # differently. Check source consistency without replacing the
                # exact archived operands of the production stability tests.
                tolerance = np.finfo(pose.dtype)
                norms = np.stack((np.linalg.norm(motion[valid, :2], axis=1),
                                  np.linalg.norm(motion[valid, 2:], axis=1)), axis=1)
                checks["planar_motion_norms"] &= bool(np.allclose(
                    norms, pose[valid][:, (3, 5)], rtol=4 * tolerance.eps, atol=tolerance.tiny))
            if np.any(valid & terminal) or np.any(valid & (step != last_step + 1)) or np.any(valid & (a["episode_index"] != d["episode_indices"])):
                raise NumericEvidenceError("Push numeric episode gap, reset, or post-terminal sample")
            # A first episode cannot disappear and later resume; closed padding is excluded.
            if np.any(~valid & ~terminal):
                raise NumericEvidenceError("Push numeric missing first-episode sample")
            cond = _conditions(pose, p)
            stable = cond.all(axis=1)
            threshold = lambda name: np.asarray(p[name], dtype=pose.dtype)
            failure = (pose[:, 0] < threshold("fall_height_m")) | (np.abs(pose[:, 1]) > threshold("fall_tilt_rad")) | (np.abs(pose[:, 2]) > threshold("fall_tilt_rad"))
            if p["max_displacement_m"] is not None:
                failure |= pose[:, 5] > threshold("max_displacement_m")
            timeout = step >= timing["max_episode_steps"]
            checks["conditions"] &= bool(np.array_equal(cond[valid], a["conditions"][valid]))
            checks["termination"] &= bool(np.array_equal(failure[valid], a["failure"][valid]) and np.array_equal(timeout[valid], a["timeout"][valid]))
            minimum[valid] = np.minimum(minimum[valid], pose[valid])
            maximum[valid] = np.maximum(maximum[valid], pose[valid])
            max_roll[valid] = np.maximum(max_roll[valid], np.abs(pose[valid, 1]))
            final_pose[valid], final_stable[valid] = pose[valid], stable[valid]
            last_event[valid], last_verified[valid] = a["event_steps"][valid], a["event_verified"][valid]
            if force:
                scheduled = np.asarray(timing["event_scheduled_steps"])
                complete = (a["actual_substeps"] == d["force"]["expected_substeps"]) & a["event_verified"]
                complete &= (a["event_steps"] == scheduled) & (a["event_ends"] == scheduled + d["force"]["pulse_steps"])
                # 'completed' in the original record is independent of verification.
                last_complete[valid] = ((a["actual_substeps"] == d["force"]["expected_substeps"]) &
                    (a["event_steps"] == scheduled) & (a["event_ends"] == scheduled + d["force"]["pulse_steps"]))[valid]
                anchors = (np.where(planned == 2, scheduled[1], scheduled[0]) + d["force"]["pulse_steps"])[:, None]
                eligible = (complete[:, 0] & ((planned == 1) | complete[:, 1]) & (planned > 0))[:, None]
                verification = ~a["unplanned_wrench"][:, None]
                unplanned[valid] = a["unplanned_wrench"][valid]
            else:
                anchors, eligible, verification = a["event_steps"], a["event_steps"] >= 0, a["event_verified"]
            elapsed = step[:, None] - anchors
            in_window = valid[:, None] & eligible & (elapsed > 0) & (elapsed <= timing["recovery_window_steps"])
            observing = in_window & (recovered < 0)
            for env_id, slot in np.argwhere(observing & ~stable[:, None] & (dwell > 0)):
                interruptions[env_id][slot].append({"control_step": int(step[env_id]), "after_anchor_steps": int(elapsed[env_id, slot]),
                    "after_anchor_s": float(elapsed[env_id, slot] * timing["step_dt_s"]), "preceding_dwell_steps": int(dwell[env_id, slot]),
                    "failed_conditions": [name for name, passed in zip(CONDITIONS, cond[env_id]) if not passed]})
            window_seen += in_window
            stable_seen += in_window & stable[:, None]
            failed_condition_steps += in_window[:, :, None] & ~cond[:, None, :]
            dwell[observing] = np.where(np.broadcast_to(stable[:, None], dwell.shape)[observing], dwell[observing] + 1, 0)
            max_dwell = np.maximum(max_dwell, dwell)
            reached = observing & (dwell >= timing["required_dwell_steps"]) & verification
            recovered[reached] = np.broadcast_to(step[:, None], recovered.shape)[reached]
            checks["dwell"] &= bool(np.array_equal(dwell[valid], a["observed_dwell"][valid]))
            checks["recovery"] &= bool(np.array_equal(recovered[valid], a["observed_recovery_step"][valid]))
            terminal |= valid & (failure | timeout)
            seen += valid
            last_step[valid] = step[valid]
        frames += count
    if frames != d["frames"] or not np.all(seen > 0) or terminal.tolist() != d["closed"]:
        raise NumericEvidenceError("Push numeric coverage disagrees with completion metadata")
    if d["episode_coverage"] != ("complete" if terminal.all() else "truncated"):
        raise NumericEvidenceError("Push numeric episode completion mismatch")
    records = []
    expected_records = d["runtime_episode_records"]
    if len(expected_records) != n:
        raise NumericEvidenceError("Push numeric outcome count mismatch")
    for env_id in range(n):
        actual_mask = last_event[env_id] >= 0
        metrics = {"episode_steps": int(last_step[env_id]), "min_base_height_m": float(minimum[env_id, 0]),
            "max_pitch_rad": float(maximum[env_id, 2]), "min_pitch_rad": float(minimum[env_id, 2]),
            "max_abs_roll_rad": float(max_roll[env_id]), "max_abs_pitch_rad": float(max(abs(minimum[env_id, 2]), abs(maximum[env_id, 2]))),
            "max_xy_drift_m": float(maximum[env_id, 5]), "final_planar_speed_mps": float(final_pose[env_id, 3]),
            "final_recovery_condition": bool(final_stable[env_id])}
        classifier_inputs = {**metrics, "control_only": d["control_only"], "planned_push_count": int(planned[env_id])}
        if force:
            metrics.update(sequence_recovered=bool(recovered[env_id, 0] >= 0),
                sequence_recovery_control_step=int(recovered[env_id, 0]) if recovered[env_id, 0] >= 0 else None)
            classification = classify_force_recovery_episode({**classifier_inputs,
                "sequence_recovered": metrics["sequence_recovered"], "unplanned_wrench": bool(unplanned[env_id]),
                "force_events": [{"injection_verified": bool(last_verified[env_id, slot]),
                                  "completed": bool(last_complete[env_id, slot])}
                                 for slot in np.flatnonzero(actual_mask)]}, p)
        else:
            classification = classify_recovery_episode({**classifier_inputs,
                "push_events": [{"injection_verified": bool(last_verified[env_id, slot]),
                                 "recovered": bool(recovered[env_id, slot] >= 0)}
                                for slot in np.flatnonzero(actual_mask)]}, p)
        label = classification.pop("label")
        metrics.update(classification, failure_label=label)
        original = expected_records[env_id]
        differences = [key for key, value in metrics.items() if original.get(key) != value]
        if original["env_id"] != env_id or original["episode_index"] != d["episode_indices"][env_id]:
            raise NumericEvidenceError("Push numeric outcome identity mismatch")
        recoveries = []
        for slot in range(slots if force else int(planned[env_id])):
            anchor = (timing["event_scheduled_steps"][1 if planned[env_id] == 2 else 0] + d["force"]["pulse_steps"]
                if force else int(last_event[env_id, slot]))
            recovery = int(recovered[env_id, slot])
            observed = int(window_seen[env_id, slot])
            anchor_observed = (bool(planned[env_id] > 0 and last_complete[env_id, :planned[env_id]].all()) if force else anchor >= 0)
            anchor_verified = (bool(last_verified[env_id, :planned[env_id]].all() and not unplanned[env_id]) if force else bool(last_verified[env_id, slot]))
            status = ("no_push_control" if d["control_only"] else "recovered" if recovery >= 0 else
                "anchor_not_observed" if not anchor_observed else "invalid_injection" if not anchor_verified else
                "full_window_without_dwell" if last_step[env_id] >= anchor + timing["recovery_window_steps"] else
                "terminal_before_window_end" if terminal[env_id] else "rollout_truncated")
            recoveries.append({"identity": "sequence" if force else f"event_{slot}",
                "anchor_control_step": anchor if anchor_observed else None,
                "scheduled_anchor_control_step": (anchor if force else timing["event_scheduled_steps"][slot]),
                "anchor_verified": bool(anchor_observed and anchor_verified), "recovery_censored": recovery < 0,
                "recovery_control_step": recovery if recovery >= 0 else None, "coverage": status,
                "observed_window_steps": observed, "stable_window_steps": int(stable_seen[env_id, slot]),
                "max_dwell_steps": int(max_dwell[env_id, slot]),
                "observed_window_duration_s": observed * timing["step_dt_s"],
                "failed_condition_steps": dict(zip(CONDITIONS, failed_condition_steps[env_id, slot].tolist())),
                "dwell_interruptions": interruptions[env_id][slot]})
            if not force and anchor >= 0:
                event = next((event for event in original["push_events"] if event["event_index"] == slot), None)
                if event is None or event["recovery_control_step"] != (recovery if recovery >= 0 else None):
                    differences.append(f"event_{slot}.recovery_control_step")
        records.append({"env_id": env_id, "episode_index": d["episode_indices"][env_id],
            "stratum": ("control", "low", "medium", "high", "repeated")[d["strata"][env_id]],
            "planned_push_count": int(planned[env_id]), "control_only": d["control_only"],
            "coverage": "complete" if terminal[env_id] else "truncated", "valid_steps": int(seen[env_id]),
            "terminal_label": label if terminal[env_id] else None, "snapshot_metrics": metrics,
            "final_unstable_after_recovery": bool(np.any(recovered[env_id] >= 0) and not final_stable[env_id]),
            "recoveries": recoveries, "runtime_differences": differences})
    return {"schema_version": "true_rl_agent.push_recovery.numeric_recompute.v1", "identity": manifest["identity"],
        "direction_status": "recorded_world_xy" if has_direction else "not_collected_in_v1",
        "checks": checks, "matches_runtime": all(checks.values()) and not any(record["runtime_differences"] for record in records),
        "episode_coverage": d["episode_coverage"], "episodes": n, "valid_control_steps": int(seen.sum()), "per_episode": records}
