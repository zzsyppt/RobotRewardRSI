"""Push root measurements for explicitly claimed displayed frames."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from math import ceil, degrees, isfinite
from numbers import Integral, Real
from typing import Any

import numpy as np

from true_rl_agent.evaluation.visual_observer import (
    IDENTITY_FIELDS,
    MediaWindow,
    visual_process_sentences,
)
from true_rl_agent.tasks.push_recovery.failure_classifier import RECOVERY_FAILURE_LABELS


PUSH_S0_DEFINITIONS = (
    {"name": "root_height", "unit": "m", "body": "root",
     "reference_frame": "world z relative to environment origin",
     "source": "pose[:,0]: root_pos_w.z - env_origin_w.z"},
    {"name": "root_reset_x", "unit": "m", "body": "root",
     "reference_frame": "world x relative to episode reset root",
     "source": "planar_motion[:,2]: root_pos_w.x - reset_root_pos_w.x"},
    {"name": "root_reset_y", "unit": "m", "body": "root",
     "reference_frame": "world y relative to episode reset root",
     "source": "planar_motion[:,3]: root_pos_w.y - reset_root_pos_w.y"},
    {"name": "root_roll_rad", "unit": "rad", "body": "root",
     "reference_frame": "signed Euler roll of the articulation root orientation in world",
     "source": "pose[:,1]: roll from root quaternion converted to wxyz; root rotation, not an independent torso link or knee angle"},
    {"name": "root_pitch_rad", "unit": "rad", "body": "root",
     "reference_frame": "signed Euler pitch of the articulation root orientation in world",
     "source": "pose[:,2]: pitch from root quaternion converted to wxyz; root rotation, not an independent torso link or knee angle"},
)


PUSH_MEDIA_SAMPLING_RULE = "push.media_steps.v1"


def _push_media_integer(value: Any, minimum: int = 0) -> int | None:
    if isinstance(value, Integral) and not isinstance(value, bool) and value >= minimum:
        return int(value)
    return None


def select_push_media_steps(
    transcript: Mapping[str, Any], *, runtime_episode_record: Mapping[str, Any],
    available_steps: Sequence[int], step_dt_s: float,
) -> dict[str, Any]:
    """Choose at most six real frames; caller intersects valid numeric and both views.

    Preserve the episode's observed first/middle/last frames, then describe the
    last verified executed injection with available recovery evidence. Scheduled
    events never supply occurrence evidence. Neither a local latch nor a window
    endpoint asserts that the whole episode recovered.
    """
    if (not isinstance(step_dt_s, Real) or isinstance(step_dt_s, bool)
            or not isfinite(step_dt_s) or step_dt_s <= 0
            or not isfinite(0.5 / step_dt_s)):
        raise ValueError("Push media sampling requires a positive finite bound step_dt_s")
    dt = float(step_dt_s)
    offset = ceil(0.5 / dt)
    steps = tuple(_push_media_integer(step, 1) for step in available_steps)
    if not steps or any(step is None for step in steps) or tuple(sorted(set(steps))) != steps:
        raise ValueError("Push media available_steps must be sorted distinct positive integers")
    final = _push_media_integer(transcript.get("final_control_step"), 1)
    valid = _push_media_integer(transcript.get("valid_steps"), 1)
    if final is None or valid is None or valid > final:
        raise ValueError("Push media transcript requires its real observed episode bounds")
    first = final - valid + 1
    if steps[0] != first or steps[-1] != final:
        raise ValueError("Push media available_steps must include the real first and final valid steps")
    for key in ("env_id", "episode_index"):
        if key in transcript and key in runtime_episode_record and transcript[key] != runtime_episode_record[key]:
            raise ValueError("Push media transcript and runtime episode identities differ")
    midpoint = min(steps, key=lambda step: (abs(2 * step - first - final), step))
    baseline = sorted({first, midpoint, final})
    available = set(steps)
    process: dict[str, Any] = {
        "status": "no_verified_observed_recovery", "source": None,
        "recovery_identity": None, "reference_event_index": None,
        "injection_start_step": None, "before_target_step": None,
        "before": None, "middle": None, "after": None,
        "before_source": None, "middle_source": None, "after_source": None,
        "meanings": {
            "before": "observed context strictly before actual injection, targeting at least 0.5 seconds earlier",
            "middle": "observed recovery-window speed peak, otherwise first observed response after injection",
            "after": "local recovery latch at or after middle, otherwise last sampled recovery-window state; not whole-episode recovery",
        },
    }
    event_keys = [key for key in ("push_events", "force_events") if key in runtime_episode_record]
    if len(event_keys) > 1:
        raise ValueError("Push media runtime episode has ambiguous injection mechanisms")
    candidates = []
    if event_keys:
        event_key = event_keys[0]
        force = event_key == "force_events"
        events = runtime_episode_record[event_key]
        for unit in transcript.get("recoveries", ()):
            observed = unit.get("process", {})
            anchor = _push_media_integer(unit.get("anchor_control_step"))
            index = _push_media_integer(observed.get("reference_event_index"))
            if (unit.get("anchor_verified") is not True or observed.get("status") != "available"
                    or anchor is None or index is None):
                continue
            matches = [event for event in events
                       if _push_media_integer(event.get("event_index")) == index]
            if len(matches) != 1:
                continue
            event = matches[0]
            start = _push_media_integer(event.get("start_control_step" if force else "control_step"))
            expected_anchor = _push_media_integer(event.get("end_control_step")) if force else start
            if (event.get("injection_verified") is not True or (force and event.get("completed") is not True)
                    or start is None or start > final or expected_anchor != anchor
                    or not start <= anchor <= final):
                continue
            candidates.append((start, index, unit, observed, event_key))
    if candidates:
        start, index, unit, observed, event_key = max(candidates, key=lambda item: (item[0], item[1]))
        target = start - offset
        prior = [step for step in steps if step < start]
        before_target = [step for step in prior if step <= target]
        before = max(before_target) if before_target else min(prior) if prior else None
        middle, middle_source = None, None
        for name in ("window_speed_peak", "first_observed_response"):
            sample = observed.get(name)
            step = _push_media_integer(sample.get("control_step")) if isinstance(sample, Mapping) else None
            if step in available and step > start:
                middle, middle_source = step, name
                break
        last_sample = observed.get("window_last")
        last = _push_media_integer(last_sample.get("control_step")) if isinstance(last_sample, Mapping) else None
        after, after_source = None, None
        for step, name in ((_push_media_integer(unit.get("recovery_control_step")), "recovery_control_step"),
                           (last, "window_last")):
            if step in available and step > start and (middle is None or step >= middle):
                after, after_source = step, name
                break
        process.update(
            status="available", source=f"runtime_episode_record.{event_key}",
            recovery_identity=unit.get("identity"), reference_event_index=index,
            injection_start_step=start, before_target_step=target,
            before=before, middle=middle, after=after,
            before_source=("latest_at_or_before_context_target" if before_target else
                           "earliest_available_before_injection" if prior else None),
            middle_source=middle_source, after_source=after_source,
        )
    selected = set(baseline)
    selected.update(process[name] for name in ("before", "middle", "after") if process[name] is not None)
    fill = []
    while len(selected) < min(6, len(steps)):
        step = max((step for step in steps if step not in selected),
                   key=lambda step: (min(abs(step - chosen) for chosen in selected), -step))
        selected.add(step)
        fill.append(step)
    return {"sampling_rule": PUSH_MEDIA_SAMPLING_RULE, "selected_steps": sorted(selected),
            "baseline_steps": baseline, "process": process, "fill_steps": fill,
            "step_dt_s": dt, "context_offset_steps": offset}


def push_measurement_claims(recovery_height_m: float) -> tuple[dict[str, str], ...]:
    """Use the resolved protocol threshold, not a report-derived outcome."""
    threshold = float(recovery_height_m)
    if not np.isfinite(threshold):
        raise ValueError("height threshold must be finite")
    return (
        {"id": "root_low_height_sample", "body": "root", "definition":
         f"Minimum root height (root_pos_w.z - env_origin_w.z) over cited displayed control steps is < {repr(threshold)} m. Numeric height only, not contact."},
        {"id": "root_small_xy_endpoint_delta", "body": "root", "definition":
         "Euclidean world-xy distance between reset-relative planar_motion xy positions at last and first cited displayed control steps is < 0.02 m. The fixed reset reference cancels; endpoint displacement only, not stillness or contact."},
    )


def _push_samples(identity, control_step, pose, planar_motion):
    bound = dict(identity)
    if not all(isinstance(bound.get(key), str) and bound[key] for key in IDENTITY_FIELDS):
        raise ValueError("numeric identity is incomplete")
    if control_step is None:
        return bound, None
    steps = tuple(int(s) if isinstance(s, Integral) and not isinstance(s, bool) else s for s in control_step)
    poses = None if pose is None else np.array(pose, dtype=float, copy=True)
    motion = None if planar_motion is None else np.array(planar_motion, dtype=float, copy=True)
    if (not steps or any(type(s) is not int or s < 0 for s in steps)
            or tuple(sorted(set(steps))) != steps
            or (poses is not None and (poses.ndim != 2 or poses.shape[0] != len(steps)
                                       or poses.shape[1] < 1 or not np.isfinite(poses).all()))
            or (motion is not None and (motion.ndim != 2 or motion.shape[0] != len(steps)
                                        or motion.shape[1] < 4 or not np.isfinite(motion).all()))):
        raise ValueError("push numeric valid-step arrays do not correspond")
    return bound, {step: (None if poses is None else poses[i, 0],
                          None if motion is None else motion[i, 2:4],
                          None if poses is None or poses.shape[1] < 2 else poses[i, 1],
                          None if poses is None or poses.shape[1] < 3 else poses[i, 2])
                   for i, step in enumerate(steps)}


def _cited_steps(observation, window):
    refs = observation.get("evidence_refs")
    if (not isinstance(refs, list) or not refs
            or any(not isinstance(ref, Mapping) or set(ref) != {"step", "view_id"}
                   or type(ref["step"]) is not int or not isinstance(ref["view_id"], str)
                   for ref in refs)):
        return None
    pairs = [(ref["step"], ref["view_id"]) for ref in refs]
    available = [(step, image.view_id) for image in window.images for step in image.displayed_steps]
    if len(set(pairs)) != len(pairs) or any(available.count(pair) != 1 for pair in pairs):
        return None
    return sorted({step for step, view in pairs})


def push_s0_for_window(window: MediaWindow, *, identity: Mapping[str, Any],
                       control_step: Sequence[int] | None, pose: Any | None,
                       planar_motion: Any | None) -> tuple[dict[str, Any], ...]:
    """Displayed root scalars from one verified episode, without default values.

    pose[:,0] already subtracts environment-origin z. planar_motion[:,2:4]
    already subtracts this episode's reset root xy; neither is absolute position.
    Existing pose roll/pitch columns describe the articulation root, not knees or
    an independently moving torso link. Older narrow arrays leave them absent.
    """
    bound, samples = _push_samples(identity, control_step, pose, planar_motion)
    if any(window.identity.get(key) != bound[key]
           or window.identity.get("numeric_" + key) != bound[key] for key in IDENTITY_FIELDS):
        raise ValueError("push media and numeric identities differ")
    result = []
    for step in sorted({s for image in window.images for s in image.displayed_steps}):
        if samples is None or step not in samples:
            continue
        height, xy, roll, pitch = samples[step]
        if height is not None:
            result.append({**PUSH_S0_DEFINITIONS[0], "step": step, "value": float(height)})
        if xy is not None:
            for definition, value in zip(PUSH_S0_DEFINITIONS[1:3], xy):
                result.append({**definition, "step": step, "value": float(value)})
        for definition, value in zip(PUSH_S0_DEFINITIONS[3:], (roll, pitch)):
            if value is not None:
                result.append({**definition, "step": step, "value": float(value)})
    return tuple(result)


def push_pose_facts(window: MediaWindow) -> tuple[dict[str, Any], ...]:
    """Render at most three dimensions of this task's already constructed S0.

    These are sampled root measurements, not visual observations or whole-body
    pose labels. The scalar references retain exact inputs behind rounded text.
    """
    definitions = {row["name"]: row for row in PUSH_S0_DEFINITIONS
                   if row["name"] in {"root_roll_rad", "root_pitch_rad", "root_height"}}
    displayed = {step for image in window.images for step in image.displayed_steps}
    values: dict[str, dict[int, float]] = {name: {} for name in definitions}
    for scalar in window.s0:
        name = scalar.get("name")
        if name not in definitions:
            continue
        step, value = scalar.get("step"), scalar.get("value")
        if (any(scalar.get(key) != expected for key, expected in definitions[name].items())
                or type(step) is not int or step not in displayed or step in values[name]
                or isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value)
                or (name != "root_height" and not isfinite(degrees(value)))):
            raise ValueError("Push pose facts require distinct, finite task S0 at displayed steps")
        values[name][step] = float(value)
    facts = []
    for name, identifier, label, unit, precision in (
        ("root_roll_rad", "push.root_roll", "Articulation root signed Euler roll in world", "deg", 1),
        ("root_pitch_rad", "push.root_pitch", "Articulation root signed Euler pitch in world", "deg", 1),
        ("root_height", "push.root_height", "Root world z relative to environment origin", "m", 3),
    ):
        if not values[name]:
            continue
        samples = sorted(values[name].items())
        rendered = "; ".join(f"{step}: {(degrees(value) if unit == 'deg' else value):.{precision}f}"
                             for step, value in samples)
        boundary = "Root rotation, not an independent torso link or knee angle." if unit == "deg" else "Not foot clearance."
        facts.append({"id": identifier, "body": "root",
                      "text": f"{label} at displayed steps (approx {unit}): {rendered}. {boundary}",
                      "scalar_refs": [{"name": name, "step": step} for step, _ in samples]})
    return tuple(facts)


def push_numeric_check(*, identity: Mapping[str, Any], control_step: Sequence[int] | None,
                       pose: Any | None, planar_motion: Any | None,
                       recovery_height_m: float):
    """Bind valid one-episode arrays and check only explicitly cited step/view pairs.

    pose[:,0] is origin-relative root z; planar_motion[:,2:4] is reset-relative
    world-xy position. The measured predicate is compared with its explicit
    asserted polarity, not with free-form visual text. Media relation validation
    belongs to the common audit.
    """
    claims = push_measurement_claims(recovery_height_m)
    bound, samples = _push_samples(identity, control_step, pose, planar_motion)

    def check(observation: Mapping[str, Any], window: MediaWindow) -> dict[str, Any] | None:
        claim = observation.get("numeric_claim")
        if (not isinstance(claim, Mapping) or set(claim) != {"id", "asserted"}
                or not isinstance(claim["id"], str) or type(claim["asserted"]) is not bool):
            return None
        claim_id, asserted = claim["id"], claim["asserted"]
        expected = next((row for row in claims if row["id"] == claim_id), None)
        if expected is None or observation.get("body") != "root" or expected not in window.measurement_claims:
            return None
        media = window.identity
        if any(media.get(key) is None for key in IDENTITY_FIELDS):
            return None
        if any(media[key] != bound[key] for key in IDENTITY_FIELDS):
            raise ValueError("push media and numeric identities differ")
        shown = _cited_steps(observation, window)
        if samples is None or not shown or any(step not in samples for step in shown):
            return None
        if claim_id == "root_low_height_sample":
            if any(samples[s][0] is None for s in shown):
                return None
            value, threshold = min(float(samples[s][0]) for s in shown), float(recovery_height_m)
        else:
            if len(shown) < 2 or samples[shown[0]][1] is None or samples[shown[-1]][1] is None:
                return None
            value = float(np.linalg.norm(samples[shown[-1]][1] - samples[shown[0]][1]))
            threshold = 0.02
        observed = value < threshold
        return {"status": "consistent" if observed == asserted else "conflict",
                "numeric_evidence": {"claim": claim_id, "asserted": asserted, "observed": observed,
                                     "cited_steps": shown,
                                     "evidence_refs": [dict(ref) for ref in observation["evidence_refs"]],
                                     "value": value, "threshold": threshold}}

    return check


def attach_push_visual_observations(full: Mapping[str, Any], audited: Mapping[str, Any],
                                    *, observer_version: str) -> dict[str, Any]:
    """Attach media only when its physical run has a verified logical alias."""
    result = deepcopy(dict(full))
    result["observer_evidence"] = deepcopy(dict(audited))
    binding = result["binding"]
    sources = result["source_bindings"]
    transcript_keys = {(row["source"], row["env_id"]) for row in result["transcripts"]}
    public_audit = deepcopy(dict(audited))
    approved = {}
    for record in public_audit["observations"]:
        identity = record.get("media_identity") or {}
        source = next((row for row in sources
                       if row["manifest_sha256"] == identity.get("source_id")), None)
        valid = (source is not None and source.get("actual_run_index") is not None
                 and source.get("actual_run_status") == "verified_archive_identity"
                 and all(identity.get(key) and identity.get(key) == identity.get("numeric_" + key)
                         for key in IDENTITY_FIELDS)
                 and identity["checkpoint_id"] == binding["checkpoint_hash"]
                 and identity["protocol_id"] == binding["resolved_protocol_hash"]
                 and identity["run_id"] == str(source["actual_run_index"]))
        env = None
        if valid:
            prefix = f"r{source['actual_run_index']}.e"
            suffix = identity["episode_id"].removeprefix(prefix)
            if identity["episode_id"].startswith(prefix) and suffix.isdecimal():
                env = int(suffix)
            valid = env is not None and (source["source_index"], env) in transcript_keys
        if valid:
            alias, label = record["alias"], f"r{source['logical_run']}.e{env}"
            if alias in approved and approved[alias] != label:
                record.setdefault("violations", []).append("source_binding")
            else:
                approved[alias] = label
        else:
            record.setdefault("violations", []).append("source_binding")
    rejected = {record.get("alias") for record in public_audit["observations"]
                if "source_binding" in record.get("violations", [])}
    for record in public_audit["observations"]:
        if record.get("alias") in rejected:
            record.setdefault("violations", []).append("source_binding")
    for alias in rejected:
        approved.pop(alias, None)
    result["observer_source_binding"] = {
        "rejected_observation_indices": [i for i, record in enumerate(public_audit["observations"])
                                         if "source_binding" in record.get("violations", [])],
        "approved_aliases": dict(approved),
    }
    exemplar = result.get("representative_selection", {}).get("context_episode_id")
    logical_runs = {source["source_index"]: source["logical_run"] for source in sources}
    failed = {f"r{logical_runs[row['source']]}.e{row['env_id']}" for row in result["transcripts"]
              if row["source"] in logical_runs and row.get("terminal_label") in RECOVERY_FAILURE_LABELS}
    priority = {alias: 0 if label == exemplar else 1 if label in failed else 2
                for alias, label in approved.items()}
    rows = visual_process_sentences(public_audit, episode_labels=approved,
                                    cohort_episode_n=len(result["transcripts"]),
                                    observer_version=observer_version,
                                    evidence_ref="observer_evidence", episode_priority=priority)
    result["sentences"] = [row for row in result["sentences"]
                           if row.get("section") != "visual" and row.get("id") != "push.visual.missing"] + rows
    result["evidence_catalog"]["observer_evidence"] = {"pointer": "/observer_evidence"}
    for index in range(len(audited["observations"])):
        result["evidence_catalog"][f"observer_evidence/observations/{index}"] = {
            "pointer": f"/observer_evidence/observations/{index}"}
    return result


def align_push_snapshot_samples(
    manifest: Mapping[str, Any], *, control_step: Sequence[int],
    pose: Any | None = None, planar_motion: Any | None = None,
) -> dict[str, Any]:
    """Compare only Push's overlapping numeric operands, not full-body truth.

    Caller supplies verified valid rows from one episode, already bound to the
    manifest/env/seed. JSON direct copies are compared in the NPZ dtype exactly;
    float32 quaternion/norm derivations use atol=1e-5, rtol=0. Missing reset/origin
    references are never replaced by the first sampled position.
    """
    steps = np.asarray(control_step)
    if steps.ndim != 1 or (len(steps) and steps.dtype.kind not in "iu") or len(set(steps.tolist())) != len(steps):
        raise ValueError("push alignment needs distinct integer control steps")
    rows = {int(step): i for i, step in enumerate(steps)}
    pose_array = None if pose is None else np.asarray(pose)
    motion_array = None if planar_motion is None else np.asarray(planar_motion)
    samples = []
    for sample in manifest.get("samples", []):
        step = sample.get("control_step")
        index = rows.get(step) if type(step) is int else None
        record = {"control_step": step, "numeric_row": index, "fields": []}
        samples.append(record)
        fields = record["fields"]
        if index is None:
            fields.append({"field": "control_step", "status": "missing", "reason": "no valid numeric row"})
            continue
        state = sample.get("state_before", {})
        specifications = (
            ("roll", "root_quat_w", pose_array, 1, 4, True),
            ("pitch", "root_quat_w", pose_array, 2, 4, True),
            ("planar_velocity_xy", "root_lin_vel_w", motion_array, slice(0, 2), 3, False),
            ("planar_speed", "root_lin_vel_w", pose_array, 3, 3, True),
            ("angular_speed", "root_ang_vel_w", pose_array, 4, 3, True),
        )
        for field, source, array, column, width, derived in specifications:
            result = {"field": field, "snapshot_field": source,
                      "atol": 1e-5 if derived else 0.0, "rtol": 0.0}
            fields.append(result)
            if array is None or source not in state:
                result.update(status="missing", reason="numeric or snapshot field absent")
                continue
            needed = 2 if isinstance(column, slice) else column + 1
            if (array.ndim != 2 or array.shape[0] != len(steps) or array.shape[1] < needed
                    or array.dtype.kind != "f"):
                result.update(status="mismatch", reason="numeric layout does not match field/control_step")
                continue
            raw = np.asarray(state[source], dtype=np.float32)
            expected = array[index, column]
            if raw.shape != (width,) or not np.isfinite(raw).all() or not np.isfinite(expected).all():
                result.update(status="mismatch", reason="invalid shape or nonfinite value")
                continue
            if source == "root_quat_w":
                if "xyzw" not in manifest.get("state_conventions", {}).get(source, ""):
                    result.update(status="unsupported", reason="snapshot quaternion convention is not declared xyzw")
                    continue
                x, y, z, w = raw
                observed = (np.arctan2(2 * (w*x + y*z), 1 - 2 * (x*x + y*y)) if field == "roll"
                            else np.arcsin(np.clip(2 * (w*y - z*x), -1.0, 1.0)))
            elif field == "planar_velocity_xy":
                observed = raw[:2].astype(array.dtype)
            else:
                observed = np.linalg.norm(raw[:2] if field == "planar_speed" else raw)
            error = float(np.max(np.abs(np.asarray(observed, dtype=np.float64) - np.asarray(expected, dtype=np.float64))))
            equal = (error <= 1e-5 if derived else np.array_equal(observed, expected))
            result.update(status="aligned" if equal else "mismatch", max_abs_error=error,
                          comparison_dtype="float32 derivation" if derived else str(array.dtype),
                          comparison="absolute_error" if derived else "dtype_exact")
    statuses = [field["status"] for sample in samples for field in sample["fields"]]
    status = ("mismatch" if "mismatch" in statuses else
              "overlap_aligned" if statuses and all(s == "aligned" for s in statuses) else "incomplete")
    return {"scope": "push_roll_pitch_and_velocity_overlap_only", "status": status, "samples": samples,
            "not_checked": [
                {"field": "height", "reason": "same-run environment-origin z not supplied"},
                {"field": "reset_relative_xy_and_drift", "reason": "same-episode reset initial_xy not supplied"},
                {"field": "full_root_pose_and_joints", "reason": "not recorded as full state in Push numeric arrays"},
                {"field": "PNG pixels", "reason": "outside numeric overlap check"}],
            "binding_scope": "caller verified same manifest/env/seed and one episode valid rows"}
