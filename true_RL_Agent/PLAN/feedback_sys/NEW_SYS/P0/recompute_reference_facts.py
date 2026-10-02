"""Independent P0 facts: stdlib + NumPy, never import a production decoder.

Run from any directory with the project's .venv Python. Defaults find the project
from this file. The frozen selection is development material, not an E1 result.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np


FEET = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
DT = 0.02
THRESHOLD_N = 1.0
TOLERANCE = {"absolute": 1e-6, "relative": 1e-6,
             "integer_counts_and_sample_indices": "exact"}
P0 = Path("PLAN/feedback_sys/NEW_SYS/P0")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def native(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(v) for v in value]
    return value


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(native(data), ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def relative(path, root):
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def deterministic_npz(path, arrays):
    """NPZ with fixed ZIP metadata so repeated evidence hashes are identical."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for key, value in sorted(arrays.items()):
            stream = io.BytesIO()
            np.lib.format.write_array(stream, np.asarray(value), allow_pickle=False)
            info = zipfile.ZipInfo(key + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, stream.getvalue())


def intervals(predicate):
    """Return observed sample runs, with no invented continuous-time duration."""
    predicate = np.asarray(predicate, dtype=bool)
    edges = np.diff(np.r_[False, predicate, False].astype(np.int8))
    result = []
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        left, right = start == 0, stop == len(predicate)
        result.append({
            "first_step": int(start + 1), "last_step": int(stop),
            "sample_count": int(stop - start),
            "marker_cell_duration_s": float((stop - start) * DT),
            "observed_endpoint_span_s": float((stop - start - 1) * DT),
            "left_censored": bool(left), "right_censored": bool(right),
            "onset_boundary_s": None if left else [float(start * DT), float((start + 1) * DT)],
            "end_boundary_s": None if right else [float(stop * DT), float((stop + 1) * DT)],
            "boundary_interval_semantics": "(previous sample, first sample] / (last sample, next sample]",
            "unknown_boundary": bool(left or right),
            "physical_continuous_duration_s": None,
        })
    return result


def longest(runs):
    return max(runs, key=lambda item: item["sample_count"]) if runs else None


def inverse_rotate_wxyz(q, vector):
    """q conjugate times vector times q; caller supplies normalized q."""
    xyz = -q[:, 1:4]
    w = q[:, :1]
    if vector.ndim == 3:
        xyz, w = xyz[:, None, :], w[:, None, :]
    return vector + 2 * np.cross(xyz, np.cross(xyz, vector) + w * vector)


def load_source(root, lock):
    manifest_path = root / lock["manifest_relative_path"]
    require(digest(manifest_path) == lock["manifest_sha256"], "Manifest hash mismatch: " + str(manifest_path))
    require(manifest_path.stat().st_size == lock["manifest_bytes"], "Manifest byte count mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest["complete"] is True, "Incomplete source")
    require(manifest["identity"] == lock["identity"], "Source identity mismatch")
    require(manifest["identity"].get("splits") == ["public_dev"], "Independent P0 facts reject non-public_dev identity")
    metadata = manifest["metadata"]
    require(all(seed.get("split") == "public_dev" for seed in metadata["seeds"]),
            "Independent P0 facts reject non-public_dev seeds")
    require(metadata["coordinates"]["base_quat_w"] == "wxyz", "Quaternion order mismatch")
    require(metadata["runtime_projection"]["resolved_roles"]["support_effector"] == FEET,
            "Support body order mismatch")
    require(metadata["coverage"] == "every_control_step_of_each_scored_first_episode_including_pre_reset_terminal",
            "Unsupported coverage")
    locked_chunks = {c["file"]: c for c in lock["chunks"]}
    require(set(locked_chunks) == {c["file"] for c in manifest["chunks"]}, "Chunk set mismatch")
    runs = {}
    receipts = []
    for number, chunk in enumerate(manifest["chunks"]):
        path = manifest_path.parent / chunk["file"]
        actual_hash = digest(path)
        require(actual_hash == chunk["sha256"] == locked_chunks[chunk["file"]]["sha256"],
                "Chunk hash mismatch: " + str(path))
        require(path.stat().st_size == chunk["stored_bytes"] == locked_chunks[chunk["file"]]["stored_bytes"],
                "Chunk byte count mismatch")
        cm = chunk["metadata"]
        run = runs.setdefault(cm["run_index"], {"parts": {}, "static": {}, "chunk_ids": []})
        run["chunk_ids"].append(number)
        with np.load(path, allow_pickle=False) as archive:
            require(set(archive.files) == set(chunk["arrays"]), "Array inventory mismatch")
            valid = archive["valid"]
            require(valid.dtype == np.bool_, "Invalid valid dtype")
            for key in archive.files:
                value = archive[key]
                desc = chunk["arrays"][key]
                require(list(value.shape) == desc["shape"] and value.dtype.str == desc["dtype"],
                        "Array shape/dtype mismatch: " + key)
                if key in {"terrain_env_origins_w", "terrain_types"}:
                    if key in run["static"]:
                        require(np.array_equal(value, run["static"][key]), "Static array changed")
                    run["static"][key] = value.copy()
                else:
                    require(value.shape[:2] == valid.shape, "Unexpected temporal axes: " + key)
                    if np.issubdtype(value.dtype, np.floating):
                        require(np.isfinite(value[valid]).all(), "Nonfinite valid data: " + key)
                    if key == "base_quat_w":
                        require(np.all(value[~valid] == np.array([1, 0, 0, 0])), "Invalid quaternion padding")
                    elif key == "episode_length":
                        require(np.all(value[~valid] == 1), "Invalid episode-length padding")
                    else:
                        require(np.all(value[~valid] == 0), "Nonzero invalid padding: " + key)
                    run["parts"].setdefault(key, []).append(value.copy())
            n = valid.shape[0]
            require(n == cm["steps"], "Chunk step count mismatch")
            run["parts"].setdefault("step_index", []).append(np.arange(cm["step_start"], cm["step_start"] + n))
            run["parts"].setdefault("source_chunk_index", []).append(np.full(n, number, dtype=np.int32))
            run["parts"].setdefault("source_chunk_row", []).append(np.arange(n, dtype=np.int32))
        receipts.append({"chunk_index": number, "file": chunk["file"], "sha256": actual_hash,
                         "run_index": cm["run_index"], "first_step": cm["step_start"],
                         "last_step": cm["step_start"] + cm["steps"] - 1})
    seeds = {s["run_index"]: s for s in metadata["seeds"]}
    for index, run in runs.items():
        run["arrays"] = {k: np.concatenate(v, axis=0) for k, v in run.pop("parts").items()}
        a, seed = run["arrays"], seeds[index]
        require(seed["step_dt"] == DT, "Unexpected sample clock")
        require(np.array_equal(a["step_index"], np.arange(1, len(a["step_index"]) + 1)), "Noncontiguous step index")
        expected_lengths = np.asarray(seed["episode_steps"])
        expected_valid = a["step_index"][:, None] <= expected_lengths[None, :]
        require(np.array_equal(a["valid"], expected_valid), "Valid mask not first-episode prefix")
        require(np.array_equal(a["episode_length"][a["valid"]],
                               np.broadcast_to(a["step_index"][:, None], a["valid"].shape)[a["valid"]]),
                "Episode length disagrees with recorded step")
        require(len(seed["outcomes"]) == a["valid"].shape[1], "Outcome count mismatch")
        require([o["episode_steps"] for o in seed["outcomes"]] == seed["episode_steps"], "Outcome length mismatch")
    return {"source_id": lock["source_id"], "lock": lock, "manifest": manifest,
            "runs": runs, "seeds": seeds, "receipts": receipts}


def episode_facts(root, source, selected, samples_dir):
    run = source["runs"][selected["run_index"]]
    env = selected["env_index"]
    data = run["arrays"]
    n = int(data["valid"][:, env].sum())
    a = {k: (v[:n].copy() if k in {"step_index", "source_chunk_index", "source_chunk_row"}
             else v[:n, env].copy()) for k, v in data.items()}
    a.update({k: v[env].copy() for k, v in run["static"].items()})
    episode_id = selected["episode_id"]
    require(episode_id == f'{source["source_id"]}.r{selected["run_index"]}.e{env}', "Namespaced ID mismatch")
    outcome = source["seeds"][selected["run_index"]]["outcomes"][env]
    require(all(outcome[k] == v for k, v in selected["archived_outcome"].items()), "Frozen outcome mismatch")
    sample_path = samples_dir / (episode_id + ".npz")
    deterministic_npz(sample_path, a)
    sample_ref = {"path": relative(sample_path, root), "sha256": digest(sample_path),
                  "row_mapping": "row = step - 1; source_chunk_index indexes source.chunks; source_chunk_row is zero-based",
                  "env_index": env}
    raw = {"source_id": source["source_id"], "manifest_sha256": source["lock"]["manifest_sha256"],
           "run_index": selected["run_index"], "env_index": env,
           "sample_artifact": sample_ref, "valid_steps_inclusive": [1, n],
           "chunk_indices": sorted(set(a["source_chunk_index"].tolist()))}
    facts = []

    def fact(name, category, channels, formula, value, meaning, scope=None, raw_values=None):
        operands = {"channels": channels, "steps_inclusive": scope or [1, n],
                    "raw_locator": "episode.raw_reference; resolve every row through source_chunk_index/source_chunk_row"}
        if raw_values is not None:
            operands["raw_values"] = raw_values
        facts.append({"fact_id": episode_id + "." + name, "category": category,
                      "operands": operands, "formula": formula, "value": native(value),
                      "tolerance": TOLERANCE, "allowed_meaning": meaning})

    fact("source_entity_clock", "source_entity_time_and_scope", ["valid", "episode_length", "step_index"],
         "Use source namespace; t[k]=k*0.02, k=1..N; retain exactly valid=true rows.",
         {"source_id": source["source_id"], "episode_id": episode_id, "support_order": FEET,
          "quat_order": "wxyz", "dt_s": DT, "first_time_s": DT, "last_time_s": n * DT,
          "sample_phase": "post_step_pre_reset", "valid_count": n, "invalid_padding_excluded": True},
         "Only this namespaced first episode; no t=0 observation; current samples do not cover every physics substep.")
    terminal = {k: bool(a[k][-1]) for k in ["done_mask", "wrapper_done_mask", "termination_0", "termination_1"]}
    fact("archived_result_and_endpoint", "task_result_and_recorded_endpoint",
         ["valid", "episode_length", "done_mask", "wrapper_done_mask", "termination_0", "termination_1"],
         "Outcome is copied from hashed manifest.metadata.seeds[run].outcomes[env]; endpoint is last valid NPZ row.",
         {"archived_outcome": outcome, "terminal_step": n, "terminal_time_s": n * DT,
          "terminal_npz_flags": terminal},
         "Archived evaluator result and recorded endpoint; a label is not an independently observed cause.",
         raw_values={"terminal_episode_length": int(a["episode_length"][-1])})
    p, q = a["base_pos_w"].astype(np.float64), a["base_quat_w"].astype(np.float64)
    require(np.max(np.abs(np.linalg.norm(q, axis=1) - 1)) <= 1e-4, "Quaternion norm exceeds declared tolerance")
    x, delta = p[:, 0], np.diff(p, axis=0)
    drawdown = np.maximum.accumulate(x) - x
    trough = int(np.argmax(drawdown))
    peak = int(np.argmax(x[:trough + 1]))
    returns = np.flatnonzero(x[trough + 1:] >= x[peak])
    return_index = int(trough + 1 + returns[0]) if len(returns) else None
    fact("root_progress", "root_progress_and_attitude", ["base_pos_w"],
         "net=p[N]-p[1]; path=sum(norm(diff(p))); forward/backward=sum(max(+/-diff(x),0)); max_drawdown=max(cummax(x)-x)",
         {"net_world_xyz_m": p[-1] - p[0], "sampled_path_xyz_m": np.linalg.norm(delta, axis=1).sum(),
          "sampled_path_xy_m": np.linalg.norm(delta[:, :2], axis=1).sum(),
          "positive_x_variation_m": np.maximum(delta[:, 0], 0).sum(),
          "negative_x_variation_m": np.maximum(-delta[:, 0], 0).sum(),
          "max_x_drawdown_m": drawdown[trough], "drawdown_peak_step": peak + 1,
          "drawdown_trough_step": trough + 1, "first_recovered_peak_step": None if return_index is None else return_index + 1},
         "Sampled world +x displacement/path and retreat/return; path is a sample polyline, not full continuous motion.",
         raw_values={"p_first": p[0], "p_last": p[-1], "peak_xyz": p[peak], "trough_xyz": p[trough],
                     "return_xyz": None if return_index is None else p[return_index]})
    raw_q = q.copy()
    raw_up = 1 - 2 * (raw_q[:, 1] ** 2 + raw_q[:, 2] ** 2)
    q = raw_q / np.linalg.norm(raw_q, axis=1)[:, None]
    up = 1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2)
    raw_inverted = np.flatnonzero(raw_up < 0)
    inverted = np.flatnonzero(up < 0)
    imin = int(np.argmin(up))
    fact("body_up", "root_progress_and_attitude", ["base_quat_w"],
         "Validate abs(norm(raw_q)-1)<=1e-4; q=raw_q/norm(raw_q); body-up world-z=1-2*(qx*qx+qy*qy). All primary values/runs use normalized q; raw values are diagnostic only.",
         {"first_up_z": up[0], "last_up_z": up[-1], "minimum_up_z": up[imin], "minimum_step": imin + 1,
          "first_negative_step": int(inverted[0] + 1) if len(inverted) else None,
          "negative_sample_count": len(inverted), "negative_runs": intervals(up < 0),
           "max_quaternion_norm_error": np.max(np.abs(np.linalg.norm(raw_q, axis=1) - 1)),
           "raw_vs_normalized_up_z_max_abs_delta": np.max(np.abs(raw_up - up)),
           "raw_body_up_diagnostic": {"first_up_z": raw_up[0], "last_up_z": raw_up[-1],
                                      "minimum_up_z": raw_up.min(),
                                      "first_negative_step": int(raw_inverted[0] + 1) if len(raw_inverted) else None}},
         "Orientation of the recorded root. up_z<0 means its local +z points below world horizontal, not failure causation.",
         raw_values={"q_first_wxyz": raw_q[0], "q_last_wxyz": raw_q[-1], "q_at_minimum_wxyz": raw_q[imin],
                     "q_at_first_negative_wxyz": raw_q[inverted[0]] if len(inverted) else None})
    force = a["support_contact_forces_w"].astype(np.float64)
    norms = np.linalg.norm(force, axis=-1)
    on = norms > THRESHOLD_N
    speed3 = np.linalg.norm(a["support_lin_vel_w"].astype(np.float64), axis=-1)
    speedxy = np.linalg.norm(a["support_lin_vel_w"].astype(np.float64)[..., :2], axis=-1)
    for foot, entity in enumerate(FEET):
        ons, offs = intervals(on[:, foot]), intervals(~on[:, foot])
        transitions = np.diff(on[:, foot].astype(np.int8))
        fact(entity + ".markers", "per_foot_marker_counts_and_censored_runs", ["support_contact_forces_w"],
             f"body_index={foot}; marker=norm(force_xyz)>1N; occupancy=sum(marker)/N; transitions compare adjacent samples only",
             {"body_index": foot, "on_count": int(on[:, foot].sum()), "off_count": int((~on[:, foot]).sum()),
              "occupancy": on[:, foot].mean(), "observed_onset_steps": (np.flatnonzero(transitions == 1) + 2).tolist(),
              "observed_offset_steps": (np.flatnonzero(transitions == -1) + 2).tolist(),
              "on_runs": ons, "off_runs": offs, "longest_off_run": longest(offs),
              "complete_off_run_count": sum(not r["left_censored"] and not r["right_censored"] for r in offs)},
             "Current-force threshold marker counts and sample runs for this foot; no inference of physical flight, load share, or uninterrupted absence between samples.",
             raw_values={"first_force_xyz_n": force[0, foot], "last_force_xyz_n": force[-1, foot]})
        low_off = (~on[:, foot]) & (speedxy[:, foot] < 0.05)
        low_runs = intervals(low_off)
        lr = longest(low_runs)
        details = None
        if lr:
            start, stop = lr["first_step"] - 1, lr["last_step"]
            details = {"longest_run": lr, "max_speed_xyz_mps": speed3[start:stop, foot].max(),
                       "median_speed_xyz_mps": np.median(speed3[start:stop, foot]),
                        "max_speed_xy_mps": speedxy[start:stop, foot].max(),
                        "median_speed_xy_mps": np.median(speedxy[start:stop, foot]),
                       "max_force_norm_n": norms[start:stop, foot].max(),
                       "minimum_up_z": up[start:stop].min(), "maximum_up_z": up[start:stop].max()}
        fact(entity + ".low_speed_off", "per_foot_marker_counts_and_censored_runs",
             ["support_contact_forces_w", "support_lin_vel_w", "base_quat_w"],
             f"body_index={foot}; norm(force)<=1N AND norm(recorded world xy velocity)<0.05m/s; report longest sampled run; xyz speed is diagnostic only",
             {"speed_threshold_mps": 0.05, "threshold_source": "P0 development descriptive threshold, not task success/failure rule",
              "longest": details},
             "The recorded support_lin_vel_w channel has low horizontal speed while its marker is off. Historical link-position/COM-velocity asset mapping is not fully verified; this is not an exact derivative of support_pos_w, sole motion, or proof of no physical contact.")
    sum_norm, sum_fz = norms.sum(axis=0), force[:, :, 2].sum(axis=0)
    fact("force_semantics", "force_semantics_and_missing_channels", ["support_contact_forces_w"],
         "norm_share[i]=sum_k(norm(F[k,i]))/sum_ki(norm(F[k,i])); signed_fz_share[i]=sum_k(Fz[k,i])/sum_ki(Fz[k,i]); means divide by N",
         {"sample_count": n, "sample_sum_norm_n": sum_norm, "sample_sum_signed_fz_n": sum_fz,
          "mean_norm_n": sum_norm / n, "mean_signed_fz_n": sum_fz / n,
          "norm_share": sum_norm / sum_norm.sum() if sum_norm.sum() else None,
          "signed_fz_share": sum_fz / sum_fz.sum() if sum_fz.sum() else None,
          "norm_denominator_n": sum_norm.sum(), "signed_fz_denominator_n": sum_fz.sum()},
         "Shares of sampled recorded net normal-contact force norms or signed world-z components, separately. Neither is total force/load/body-weight support; tangential forces are absent.")
    bits = on.astype(np.int64) @ (1 << np.arange(4))
    fact("contact_combinations", "per_foot_marker_counts_and_censored_runs", ["support_contact_forces_w"],
         "code=sum(int(marker[i])<<i), entities=[FL,FR,RL,RR], weights=[1,2,4,8]; counts over valid current-force samples",
         {"encoding_version": "role_index_lsb.v1", "bit_order": FEET, "bit_weights": [1, 2, 4, 8],
          "counts_0_to_15": np.bincount(bits, minlength=16),
          "all_off_count": int(np.sum(bits == 0)), "all_off_fraction": np.mean(bits == 0),
          "FL_FR_RR_only_count": int(np.sum(bits == 11)), "FL_FR_RR_only_fraction": np.mean(bits == 11)},
         "Simultaneous threshold-marker combinations; all four off does not prove whole robot airborne.")
    offsets_world = a["support_pos_w"].astype(np.float64) - p[:, None, :]
    offsets_body = inverse_rotate_wxyz(q, offsets_world)
    fact("body_frame_anchors", "source_entity_time_and_scope", ["base_pos_w", "base_quat_w", "support_pos_w"],
         "offset_body=R(normalized_wxyz).T@(support_pos_w-base_pos_w); validate raw norm error<=1e-4, then normalize; x forward, y left, z up",
         {"first_body_xyz_m": offsets_body[0], "last_body_xyz_m": offsets_body[-1],
          "mean_body_xyz_m": offsets_body.mean(axis=0),
          "first_left_y_positive": bool(np.all(offsets_body[0, [0, 2], 1] > 0)),
          "first_right_y_negative": bool(np.all(offsets_body[0, [1, 3], 1] < 0)),
          "first_front_x_positive": bool(np.all(offsets_body[0, [0, 1], 0] > 0)),
          "first_rear_x_negative": bool(np.all(offsets_body[0, [2, 3], 0] < 0))},
         "Recorded body_pos_w position anchors check robot-relative left/right/front/rear; not image left/right or sole geometry. Historical link-position/COM-velocity mapping is not fully verified; no derivative equivalence is claimed.",
         raw_values={"first_root_xyz": p[0], "first_quat_wxyz": raw_q[0], "first_support_xyz": a["support_pos_w"][0]})
    base_force = a["forbidden_contact_forces_w"].astype(np.float64)
    base_norm = np.linalg.norm(base_force, axis=-1)
    base_on = np.any(base_norm > THRESHOLD_N, axis=1)
    base_runs = intervals(base_on)
    base_history = a["forbidden_contact_forces_w_history"].astype(np.float64)
    base_history_norm = np.linalg.norm(base_history, axis=-1)
    base_history_on = np.any(base_history_norm > THRESHOLD_N, axis=(1, 2))
    tail_start = max(0, n - 5)
    fact("short_terminal_events", "raw_short_transitions_preserved",
         ["forbidden_contact_forces_w", "forbidden_contact_forces_w_history", "support_contact_forces_w", "done_mask", "wrapper_done_mask", "termination_0", "termination_1"],
         "Keep every raw current-forbidden marker run (any body norm>1N), separately any-history-slot marker run, and final five frames; no duration cutoff or smoothing",
         {"forbidden_body_order": source["manifest"]["metadata"]["runtime_projection"]["resolved_roles"]["forbidden_contact"],
          "current_forbidden_on_count": int(base_on.sum()), "current_forbidden_on_runs": base_runs,
          "history_any_forbidden_on_count": int(base_history_on.sum()),
          "history_any_forbidden_on_runs": intervals(base_history_on),
          "tail_steps": a["step_index"][tail_start:], "tail_support_norms_n": norms[tail_start:],
          "tail_forbidden_norms_n": base_norm[tail_start:],
          "tail_forbidden_history_norms_n": base_history_norm[tail_start:],
          "tail_flags": {k: a[k][tail_start:] for k in terminal}},
         "Preserve even a one-sample terminal contact marker. Recorded terminal labels may have distinct detector history and are not overwritten by current-force markers.",
         raw_values={"tail_forbidden_force_xyz_n": base_force[tail_start:],
                     "tail_forbidden_history_xyz_n": base_history[tail_start:]})
    if episode_id == "r4.r1.e129":
        require(terminal["termination_1"] and base_history_on[-1], "Required short terminal history event not found")
    history = a["support_contact_forces_w_history"].astype(np.float64)
    history_on = np.any(np.linalg.norm(history, axis=-1) > THRESHOLD_N, axis=1)
    fact("history_and_missing_channels", "force_semantics_and_missing_channels",
         ["support_contact_forces_w", "support_contact_forces_w_history"],
         "Compare current marker with ANY history-slot marker as two different sampled quantities; inspect actual NPZ channel inventory.",
         {"history_any_on_counts": history_on.sum(axis=0),
          "history_any_on_current_off_counts": np.sum(history_on & ~on, axis=0),
          "current_on_history_all_off_counts": np.sum(on & ~history_on, axis=0),
          "history_length": history.shape[1], "history_full_physics_coverage": False,
          "history_slot_timestamps": "not recorded", "history_slot_validity": "not recorded",
          "missing_channels": ["actions", "joint_velocities", "applied_torques", "per_step_reward_terms",
                               "tangential_contact_forces", "all_nonfoot_body_contact_forces",
                               "per_environment_actual_mass", "effective_joint_limits", "parent_archive"],
          "existing_temporal_channels": sorted(k for k in a if k not in {"terrain_env_origins_w", "terrain_types"})},
         "Three history slots are existing sensor-buffer snapshots, with unrecorded slot times and reset validity; they cannot prove full four-substep coverage. Missing inputs block exact reward/total-support claims.")
    if episode_id == "legacy_dev.r1.e54":
        front = []
        for foot in [0, 1]:
            lr = longest(intervals((~on[:, foot]) & (speedxy[:, foot] < 0.05)))
            require(lr is not None, "Legacy inverted counterexample: no front low-speed/off run")
            require(len(inverted) > 0 and inverted[0] + 1 < lr["first_step"], "Inversion did not precede low-speed/off")
            require(lr["observed_endpoint_span_s"] > 5, "Front low-speed/off span not above 5 seconds")
            start, stop = lr["first_step"] - 1, lr["last_step"]
            require(np.all(up[start:stop] < 0), "Legacy low-speed/off run not inverted throughout sampled rows")
            front.append({"entity": FEET[foot], "run": lr})
        fact("required_inverted_counterexample", "root_progress_and_attitude",
             ["base_quat_w", "support_contact_forces_w", "support_lin_vel_w"],
             "first(normalized_up_z<0) precedes each longest front-foot (force_norm<=1N AND recorded_speed_xy<0.05m/s) run; sampled endpoint span>5s",
             {"first_negative_step": int(inverted[0] + 1), "first_negative_time_s": (inverted[0] + 1) * DT,
              "front_low_speed_off_runs": front, "check": "confirmed_by_raw_operands"},
             "Inversion precedes prolonged low-speed front-foot marker-off; upright foot suspension is contradicted, and right-censoring prohibits complete-duration language.")
    if episode_id == "legacy_dev.r1.e60":
        require(outcome["success"] and drawdown[trough] > 0.1 and return_index is not None, "Legacy success/retreat/return counterexample not confirmed")
        fact("required_return_counterexample", "root_progress_and_attitude", ["base_pos_w"],
             "Require archived success, maximum sampled x drawdown>0.1m, and a subsequent sample recovering its previous peak",
             {"peak_step": peak + 1, "trough_step": trough + 1, "return_step": return_index + 1,
              "drawdown_m": drawdown[trough], "success": True, "check": "confirmed_by_raw_operands"},
             "A successful recorded episode contains retreat and return; final success is not a monotone-progress claim.")
    return {"episode_id": episode_id, "raw_reference": raw,
            "selection_reason": selected["selection_reason"],
            "required_fact_categories": selected["required_fact_categories"], "facts": facts}


def cohort_facts(source):
    """All R4 episodes, no legacy pooling; pooled valid-sample denominators explicit."""
    groups = {name: {"episodes": 0, "valid_samples": 0, "on_counts": np.zeros(4, np.int64),
                     "sample_sum_norm_n": np.zeros(4), "sample_sum_signed_fz_n": np.zeros(4),
                     "combination_counts": np.zeros(16, np.int64)} for name in ("all", "success", "failure")}
    counts = Counter()
    for index, run in source["runs"].items():
        a = run["arrays"]
        force = a["support_contact_forces_w"].astype(np.float64)
        norms = np.linalg.norm(force, axis=-1)
        on = norms > THRESHOLD_N
        bits = on.astype(np.int64) @ (1 << np.arange(4))
        outcomes = source["seeds"][index]["outcomes"]
        counts.update(o["primary_reason"] for o in outcomes)
        success = np.array([o["success"] for o in outcomes])
        for name, membership in [("all", np.ones(len(success), bool)), ("success", success), ("failure", ~success)]:
            keep = a["valid"] & membership[None, :]
            g = groups[name]
            g["episodes"] += int(membership.sum())
            g["valid_samples"] += int(keep.sum())
            g["on_counts"] += on[keep].sum(axis=0)
            g["sample_sum_norm_n"] += norms[keep].sum(axis=0)
            g["sample_sum_signed_fz_n"] += force[keep, :, 2].sum(axis=0)
            g["combination_counts"] += np.bincount(bits[keep], minlength=16)
    for g in groups.values():
        g["marker_occupancy"] = g["on_counts"] / g["valid_samples"]
        g["norm_share"] = g["sample_sum_norm_n"] / g["sample_sum_norm_n"].sum()
        g["signed_fz_share"] = g["sample_sum_signed_fz_n"] / g["sample_sum_signed_fz_n"].sum()
        g["mean_norm_n"] = g["sample_sum_norm_n"] / g["valid_samples"]
        g["mean_signed_fz_n"] = g["sample_sum_signed_fz_n"] / g["valid_samples"]
    for reason in source["manifest"]["metadata"]["semantics"]["episode_outcomes"]["primary_reasons"]:
        counts.setdefault(reason, 0)
    return {"fact_id": "r4.all_episodes_basic_counts_and_forces", "source_id": "r4",
            "operands": {"manifest_sha256": source["lock"]["manifest_sha256"],
                         "chunks": "all source.r4.chunks verified in source_inventory",
                         "channels": ["valid", "support_contact_forces_w"],
                         "steps": "each episode valid=true prefix; no stage restriction",
                         "outcomes": "hashed manifest.metadata.seeds[*].outcomes; evaluator authority"},
            "formula": "pool valid samples by archived success; sum norm(F) and signed Fz separately; share=sum_i/sum_all_feet; count each manifest outcome once",
            "value": {"primary_outcome_counts": dict(sorted(counts.items())), "support_order": FEET, "groups": groups},
            "tolerance": TOLERANCE,
            "allowed_meaning": "R4 development cohort only, whole recorded episodes, sample-weighted current-marker and net-normal-force statistics. Not step-stage, per-episode-equal means, physical impulse truth, or E1 success."}


def summarize_for_review(full, ledger_path, root):
    """Tracked necessities, shared formulas, and explicit links to unabridged facts."""
    compact = {k: v for k, v in full.items() if k not in {"episodes", "source_inventory"}}
    compact["unabridged_ledger"] = {"path": relative(ledger_path, root), "sha256": digest(ledger_path)}
    compact["fact_resolution"] = (
        "Each episode fact inherits source/run/env/chunk/step/raw-NPZ operands from raw_reference, "
        "and formula/channels/category/tolerance/allowed_meaning from fact_templates[template]. "
        "id = episode_id + '.' + template. Unabridged ledger contains all raw transitions, "
        "sample operands and facts; summaries never replace it. Values omitted here are not claims."
    )
    compact["summary_notation"] = {
        "numeric_rounding": "Summary floats rounded to 6 decimal places (absolute error <=5e-7), except nonzero magnitudes <1e-5 use 12 places; full ledger and NPZ retain unrounded values.",
        "run_fields": "steps=[first,last] inclusive; n=sample count; cell_s=n*dt; span_s=(n-1)*dt; onset_s/end_s are sample brackets or null unknown boundary; physical continuous duration is always unknown.",
        "source_entities": FEET, "sample_phase": "post_step_pre_reset", "quaternion_order": "wxyz",
    }
    compact["source_inventory"] = []
    for s in full["source_inventory"]:
        compact["source_inventory"].append({
            "source_id": s["source_id"], "manifest_path": s["manifest_path"],
            "manifest_sha256": s["manifest_sha256"], "hash_and_validity_checks": s["hash_and_validity_checks"],
            "verified_chunk_count": len(s["chunks"]),
            "chunks_and_history_contract": "unabridged_ledger.source_inventory for this source_id",
        })
    templates, episodes = {}, []
    for ep in full["episodes"]:
        summarized = {k: v for k, v in ep.items() if k not in {"facts", "selection_reason", "required_fact_categories"}}
        summarized["facts"] = []
        for f in ep["facts"]:
            name = f["fact_id"][len(ep["episode_id"]) + 1:]
            val = f["value"]
            if name.endswith(".low_speed_off"):
                detail = val["longest"]
                if detail is None or detail["longest_run"]["observed_endpoint_span_s"] < 1:
                    continue
                val = {"speed_threshold_mps": val["speed_threshold_mps"], "longest": detail}
            elif name.endswith(".markers"):
                val = {k: v for k, v in val.items() if k not in {"on_runs", "off_runs", "observed_onset_steps", "observed_offset_steps"}}
                val["observed_onset_count"] = len(f["value"]["observed_onset_steps"])
                val["observed_offset_count"] = len(f["value"]["observed_offset_steps"])
                val["shortest_observed_off_run_samples"] = min((r["sample_count"] for r in f["value"]["off_runs"]), default=None)
            elif name == "body_up":
                val = {k: v for k, v in val.items() if k != "negative_runs"}
            elif name == "force_semantics":
                val = {k: v for k, v in val.items() if k not in {"mean_norm_n", "mean_signed_fz_n", "norm_denominator_n", "signed_fz_denominator_n"}}
            elif name == "body_frame_anchors":
                val = {k: v for k, v in val.items() if k not in {"last_body_xyz_m", "mean_body_xyz_m"}}
            elif name == "short_terminal_events":
                val = {k: (v[-2:] if k.startswith("tail_") and k != "tail_flags" else v)
                       for k, v in val.items() if k != "tail_support_norms_n"}
                val["tail_flags"] = {k: v[-2:] for k, v in f["value"]["tail_flags"].items()}
            elif name == "history_and_missing_channels":
                val = {k: v for k, v in val.items() if k not in {"missing_channels", "existing_temporal_channels"}}
                compact["missing_inputs"] = f["value"]["missing_channels"]
            elif name == "source_entity_clock":
                val = {"valid_count": val["valid_count"], "first_time_s": val["first_time_s"], "last_time_s": val["last_time_s"]}
            elif name == "archived_result_and_endpoint":
                val = {**val, "archived_outcome": {k: v for k, v in val["archived_outcome"].items()
                                                  if k in {"bin_id", "primary_reason", "success", "episode_steps"}}}
            if name not in templates:
                templates[name] = {k: f[k] for k in ("category", "formula", "tolerance", "allowed_meaning")}
                templates[name]["raw_channels"] = f["operands"]["channels"]
            summarized["facts"].append({"template": name, "value": val})
        episodes.append(summarized)
    compact["fact_templates"] = templates
    compact["episodes"] = episodes
    return compact_values(native(compact))


def compact_values(value):
    if isinstance(value, float):
        return round(value, 12 if 0 < abs(value) < 1e-5 else 6)
    if isinstance(value, list):
        return [compact_values(v) for v in value]
    if isinstance(value, dict):
        if {"first_step", "last_step", "sample_count", "physical_continuous_duration_s"} <= set(value):
            value = {"steps": [value["first_step"], value["last_step"]], "n": value["sample_count"],
                     "cell_s": value["marker_cell_duration_s"], "span_s": value["observed_endpoint_span_s"],
                     "left_censored": value["left_censored"], "right_censored": value["right_censored"],
                     "onset_s": value["onset_boundary_s"], "end_s": value["end_boundary_s"]}
        return {k: compact_values(v) for k, v in value.items()}
    return value


def write_review_json(path, data):
    """Pretty top-level structure, one complete episode/fact template per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["{"]
    for index, (key, value) in enumerate(data.items()):
        suffix = "," if index < len(data) - 1 else ""
        if key in {"episodes", "source_inventory", "cohort_facts"}:
            lines.append("  " + json.dumps(key) + ": [")
            lines.extend("    " + json.dumps(native(item), ensure_ascii=False, separators=(",", ":"), allow_nan=False)
                         + ("," if i < len(value) - 1 else "") for i, item in enumerate(value))
            lines.append("  ]" + suffix)
        else:
            encoded = json.dumps(native(value), ensure_ascii=False, indent=2, allow_nan=False)
            lines.append("  " + json.dumps(key) + ": " + encoded.replace("\n", "\n  ") + suffix)
    lines.append("}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument("--output", type=Path, help="JSON path, default PROJECT/PLAN/feedback_sys/NEW_SYS/P0/necessary_facts.json")
    parser.add_argument("--samples-dir", type=Path, help="Raw selected episode NPZ directory, default PROJECT/outputs/trajectory_p0/reference")
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = (args.output or root / P0 / "necessary_facts.json").resolve()
    samples_dir = (args.samples_dir or root / "outputs/trajectory_p0/reference").resolve()
    lock_path, selection_path = root / P0 / "source_lock.json", root / P0 / "e1_selection.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    require(selection["frozen_before_translator_output"] is True, "Selection was not frozen")
    require(len(selection["episodes"]) == 18, "Expected frozen 18-episode selection")
    source_inventory, episodes, cohort = [], [], None
    for source_lock in lock["sources"]:
        require(selection["source_manifest_hashes"][source_lock["source_id"]] == source_lock["manifest_sha256"], "Selection source mismatch")
        source = load_source(root, source_lock)
        metadata = source["manifest"]["metadata"]
        source_inventory.append({"source_id": source["source_id"],
                                 "manifest_path": source_lock["manifest_relative_path"],
                                 "manifest_sha256": source_lock["manifest_sha256"],
                                 "identity": source["manifest"]["identity"],
                                 "coordinates": metadata["coordinates"],
                                 "coverage": metadata["coverage"],
                                 "history_contract": metadata["contact_force_history"],
                                 "chunks": source["receipts"], "hash_and_validity_checks": "passed"})
        for selected in selection["episodes"]:
            if selected["source_id"] == source["source_id"]:
                ep = episode_facts(root, source, selected, samples_dir)
                require(set(f["category"] for f in ep["facts"]) >= set(selected["required_fact_categories"]), "Missing necessary fact category")
                episodes.append(ep)
        if source["source_id"] == "r4":
            cohort = cohort_facts(source)
        del source
    by_id = {ep["episode_id"]: ep for ep in episodes}
    episodes = [by_id[s["episode_id"]] for s in selection["episodes"]]
    result = {"schema_version": "trajectory.p0.necessary_facts.v1",
              "measurement_semantics_version": "trajectory.p0.measurements.v2",
              "status": "independent_pretranslator_development_reference_only_not_E1_pass",
              "baseline_commit": lock["baseline_commit"],
              "independence": {"imports": "Python standard library and numpy only",
                               "raw_inputs": "locked original NPZ plus hashed metadata; no production decoder, behavior, inspector, or translator output",
                               "frozen_before_new_translator_output": True,
                               "past_probe_numbers_used_as_truth": False,
                               "legacy_and_r4_pooled": False},
              "reproduction": {"script": relative(Path(__file__), root), "script_sha256": digest(__file__),
                               "command": ".venv/Scripts/python.exe PLAN/feedback_sys/NEW_SYS/P0/recompute_reference_facts.py --project-root .",
                               "numpy_version": np.__version__, "source_lock_sha256": digest(lock_path),
                               "selection_sha256": digest(selection_path)},
              "conventions": {"step_numbering": "one-based observed post_step/pre_reset; t=k*0.02",
                              "force_readout_clock": "t stamps the control-step recorder readout, not an observed physical force-update time; current-sensor age and history-slot times are unrecorded",
                              "invalid_padding": "Ignored through valid=false; actual padding includes identity quaternion and episode_length=1 in addition to zero-filled other arrays",
                              "sample_span_vs_cell_time": "N consecutive marker samples have endpoint span (N-1)*dt and bookkeeping cell-duration N*dt; neither proves physical continuous duration",
                              "interval_boundaries": "Adjacent recorder-sample transitions only; force-marker brackets locate buffer-readout sequence boundaries, not physical 20ms contact-event bounds. Sensor update time/latency is unknown; physical onset/end cannot be localized from these brackets, and censored sequence boundaries remain unknown.",
                              "float_precision": "raw float32 operands promoted to float64; no rounding before computations",
                              "quaternion_policy": "Validate raw norm error<=1e-4, then normalize for all primary up-z, inversion runs and body-frame rotations; raw up-z fields are diagnostics only.",
                              "support_position_velocity": "Recorded body_pos_w link-position and body_lin_vel_w COM-velocity channels; historical asset reference-point/COM-offset mapping is not fully verified. Do not equate recorded velocity with the exact derivative of recorded position.",
                              "low_speed_off": "M04 uses norm(recorded support_lin_vel_w.xy)<0.05 m/s with current force norm<=1 N; xyz speed is diagnostic only.",
                              "contact_combination_encoding": "role_index_lsb.v1: code=sum(int(marker[i])<<i); entities=[FL,FR,RL,RR], weights=[1,2,4,8]; no production ABI change.",
                              "force_semantics": "recorded world net normal-contact force; tangential forces not recorded",
                              "scope": "development regression/descriptive audit; source identity and entity order are required in every lookup"},
              "source_inventory": source_inventory, "selected_episode_count": len(episodes),
              "episodes": episodes, "cohort_facts": [cohort],
              "not_claimed": ["E1 passed", "translator correctness", "cross-policy generalization",
                              "complete physical contact coverage", "causal failure explanation"]}
    ledger_path = samples_dir / "full_reference_facts.json"
    write_json(ledger_path, result)
    compact = summarize_for_review(result, ledger_path, root)
    write_review_json(output, compact)
    summary = {"output": relative(output, root), "sha256": digest(output),
               "selected_episodes": len(episodes), "fact_count": sum(len(e["facts"]) for e in episodes) + 1,
               "verified_source_count": len(source_inventory), "verified_chunk_count": sum(len(s["chunks"]) for s in source_inventory),
               "status": result["status"]}
    write_json(samples_dir / "recompute_receipt.json", summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError) as error:
        print("REFERENCE RECOMPUTATION FAILED: " + str(error), file=sys.stderr)
        raise SystemExit(1) from error
