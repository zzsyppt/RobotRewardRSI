"""Qualified Push archives to source-bound root episodes; no simulator imports."""
from __future__ import annotations

from copy import deepcopy
from typing import Iterator

import numpy as np

from true_rl_agent.evaluation.trajectory_translation.contracts import NumericEpisode
from true_rl_agent.runner.numeric_evidence import (
    NumericEvidenceError, iter_numeric_evidence, load_numeric_evidence_manifest,
)
from .numeric_capture import (
    PUSH_NUMERIC_SCHEMA, PUSH_NUMERIC_V1_SCHEMA, recompute_push_numeric_evidence,
)

ADAPTER_VERSION = "push.trajectory_adapter.v1"
ARCHIVE_CAPABILITIES = {
    PUSH_NUMERIC_V1_SCHEMA: {"root_signed_xy": "not_recorded"},
    PUSH_NUMERIC_SCHEMA: {"root_signed_xy": "recorded_world_xy"},
}
MECHANISMS = {
    "world_planar_external_force": "true_rl_agent.task.push_recovery.resolved_force_evaluation.v1",
    "world_planar_velocity_increment": "true_rl_agent.task.push_recovery.resolved_protocol.v2",
}
COORDINATES = {
    "height": "root_pos_w.z - env_origin_w.z",
    "xy_drift": "norm(root_pos_w.xy - reset_root_pos_w.xy)",
    "roll_pitch": "signed Euler roll/pitch from root quaternion converted to wxyz",
    "planar_speed": "norm(root_lin_vel_w.xy)",
    "angular_speed": "norm(root_ang_vel_w.xyz)",
}


def iter_push_episodes(reference: dict, *, expected_identity: dict) -> Iterator[NumericEpisode]:
    """Validate archive/task operands, then bind each episode's event metadata.

    The existing task reducer checks native-dtype conditions, terminal flags,
    first-episode coverage, event/dwell operands and recorded task outcomes.
    Its output is admission evidence, not the new descriptor implementation.
    """
    manifest = load_numeric_evidence_manifest(reference)
    identity, metadata = manifest["identity"], manifest["metadata"]
    schema = metadata["schema_version"]
    mechanism = identity.get("mechanism")
    if (identity != expected_identity or identity.get("split") != "public_dev"
            or identity.get("task_id") != "unitree_push_recovery"
            or identity.get("subject_kind") not in {"reference", "candidate"}
            or schema not in ARCHIVE_CAPABILITIES or mechanism not in MECHANISMS
            or metadata["parameters"]["mechanism"] != mechanism
            or metadata["parameters"]["schema_version"] != MECHANISMS[mechanism]
            or ("force" in metadata) != (mechanism == "world_planar_external_force")):
        raise NumericEvidenceError("Push public source identity, capability or mechanism mismatch")
    coordinates = dict(COORDINATES)
    if schema == PUSH_NUMERIC_SCHEMA:
        coordinates["planar_motion"] = (
            "root_lin_vel_w.xy [m/s], root_pos_w.xy - reset_root_pos_w.xy [m]; "
            "signed world axes; same observation as pose norms")
    if (metadata["coordinates"] != coordinates
            or metadata["capture_point"] != "after_physics_after_task_observe_before_termination_and_reset"
            or metadata["time"]["first_sample_step"] != 1
            or metadata["time"]["control_step"] != "episode_length_buf; episode-relative; time_s=control_step*step_dt_s"):
        raise NumericEvidenceError("Push coordinates or recorder clock mismatch")
    replay = recompute_push_numeric_evidence(reference)
    if not replay["matches_runtime"]:
        raise NumericEvidenceError("Push task operands disagree with recorded conditions/outcomes")
    chunks = list(iter_numeric_evidence(reference))
    arrays = {name: np.concatenate([chunk[1][name] for chunk in chunks]) for name in chunks[0][1]}
    chunk_ids = np.concatenate([np.full(c[0]["frame_count"], i, dtype=np.int64) for i, c in enumerate(chunks)])
    chunk_rows = np.concatenate([np.arange(c[0]["frame_count"]) for c in chunks])
    group_identity = {k: v for k, v in identity.items() if k != "run_index"}
    source = {
        "manifest_sha256": reference["manifest_sha256"], "trial_id": identity["job_id"],
        "checkpoint_hash": identity["checkpoint_hash"], "acceptance_split": "public_dev",
        "identity": deepcopy(identity), "group_identity": group_identity,
        "capabilities": {**ARCHIVE_CAPABILITIES[schema], "foot_contact": "not_recorded",
                         "actions_torques_step_rewards": "not_recorded"},
        "adapter_version": ADAPTER_VERSION,
    }
    for env in range(metadata["num_envs"]):
        valid = arrays["valid"][:, env]
        episode_arrays = {name: value[valid, env].copy() for name, value in arrays.items() if name != "valid"}
        task_record = metadata["runtime_episode_records"][env]
        event_key = "force_events" if "force" in metadata else "push_events"
        events = task_record[event_key]
        expected_slots = np.flatnonzero(episode_arrays["event_steps"][-1] >= 0).tolist()
        if ([event["event_index"] for event in events] != expected_slots
                or task_record["stratum"] != replay["per_episode"][env]["stratum"]
                or task_record["planned_push_count"] != metadata["planned_push_counts"][env]
                or task_record["control_only"] != metadata["control_only"]):
            raise NumericEvidenceError("Push event metadata/condition assignment mismatch")
        for event in events:
            slot = event["event_index"]
            step_key = "start_control_step" if "force" in metadata else "control_step"
            if (event[step_key] != int(episode_arrays["event_steps"][-1, slot])
                    or event["injection_verified"] != bool(episode_arrays["event_verified"][-1, slot])):
                raise NumericEvidenceError("Push event metadata and array timing/verification mismatch")
            vector_key = "force_world_n" if "force" in metadata else "observed_delta_v_world_mps"
            vector = np.asarray(event[vector_key])
            if vector.shape != ((3,) if "force" in metadata else (2,)) or not np.isfinite(vector).all():
                raise NumericEvidenceError("Push disturbance must be finite force XYZ or velocity XY")
            if "force" in metadata and event["end_control_step"] != int(episode_arrays["event_ends"][-1, slot]):
                raise NumericEvidenceError("Push pulse end metadata mismatch")
        task = {"parameters": deepcopy(metadata["parameters"]), "time": deepcopy(metadata["time"]),
                "force": deepcopy(metadata.get("force")), "planned_push_count": metadata["planned_push_counts"][env],
                "stratum": task_record["stratum"], "control_only": metadata["control_only"],
                "events": deepcopy(events), "event_metadata_key": event_key,
                "closed": metadata["closed"][env], "expected_last_step": task_record["episode_steps"],
                "terminal_label": replay["per_episode"][env]["terminal_label"],
                "task_replay": deepcopy(replay["per_episode"][env])}
        yield NumericEpisode(deepcopy(source), identity["run_index"], env,
            episode_arrays["control_step"], metadata["time"]["step_dt_s"], episode_arrays,
            {"root": ("root",)}, tuple(manifest["chunks"]), chunk_ids[valid], chunk_rows[valid], task)
