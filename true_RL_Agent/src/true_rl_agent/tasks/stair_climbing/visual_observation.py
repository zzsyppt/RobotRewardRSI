"""Stair root measurements for explicitly claimed displayed frames."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import math
from numbers import Integral, Real
from typing import Any

import numpy as np

from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.evaluation.body_projection import project_snapshot_body_links
from true_rl_agent.evaluation.visual_observer import (
    IDENTITY_FIELDS,
    MediaWindow,
    visual_process_sentences,
)
from true_rl_agent.robots.base import RobotProfile

STAIR_S0_DEFINITIONS = (
    {"name": "root_up_world_z", "unit": "1", "body": "root",
     "reference_frame": "world z component of root up-axis",
     "source": "base_quat_w (unit wxyz): 1 - 2*(qx^2 + qy^2)"},
    {"name": "root_terrain_x", "unit": "m", "body": "root",
     "reference_frame": "world x relative to terrain origin",
     "source": "base_pos_w.x - terrain_env_origins_w.x"},
    {"name": "root_terrain_z", "unit": "m", "body": "root",
     "reference_frame": "world z relative to terrain origin; not support clearance",
     "source": "base_pos_w.z - terrain_env_origins_w.z"},
)

STAIR_MEASUREMENT_CLAIMS = (
    {"id": "root_up_z_negative", "body": "root", "definition":
     "At cited displayed control steps, root up-axis world-z = 1 - 2(qx^2 + qy^2) from unit wxyz quaternion; minimum over cited steps is < 0. Orientation only, not contact."},
    {"id": "root_small_x_endpoint_delta", "body": "root", "definition":
     "Absolute difference between root terrain-relative x at last and first cited displayed control steps is < 0.02 m; the fixed terrain origin cancels in this difference. Endpoint displacement only, not stillness or contact."},
)


def bind_stair_root_frame(frame_spec: Mapping[str, Any] | None, *,
                          runtime_projection: Mapping[str, Any],
                          robot_profile: Mapping[str, Any]) -> dict[str, Any] | None:
    """Bind an asset-verified direction specification without changing a profile.

    The caller supplies a reviewed specification and the run's original profile.
    Absence remains absence; neither robot names nor default axes imply forward.
    """
    if frame_spec is None:
        return None
    profile = RobotProfile.from_mapping(robot_profile)
    if (frame_spec.get("schema_version") != "true_rl_agent.robot_body_frame.v1"
            or frame_spec.get("robot_profile_id") != profile.robot_profile_id
            or frame_spec.get("robot_profile_hash") != profile.profile_hash
            or runtime_projection.get("robot_profile_id") != profile.robot_profile_id
            or runtime_projection.get("robot_profile_hash") != profile.profile_hash):
        raise ValueError("root frame specification differs from the run's robot profile")
    body = frame_spec.get("body_name")
    base_role = profile.semantic_roles.get("base")
    if (not isinstance(body, str) or base_role is None or base_role.targets != (body,)
            or runtime_projection.get("resolved_roles", {}).get("base") != [body]
            or frame_spec.get("kinematic_base_source") != "articulation_root"
            or runtime_projection.get("kinematic_base_source") != "articulation_root"):
        raise ValueError("root frame specification requires the bound articulation root")
    closure = profile.runtime_facts["asset_closure"]
    root_asset = frame_spec.get("root_asset", {})
    if (root_asset.get("identity") != closure["root_identity"]
            or not any(entry["identity"] == root_asset.get("identity")
                       and entry["sha256"] == root_asset.get("sha256") for entry in closure["entries"])):
        raise ValueError("root frame specification differs from the bound asset hash")
    forward, up = (np.asarray(frame_spec.get(name), dtype=float)
                   for name in ("forward_axis_body", "up_axis_body"))
    if (forward.shape != (3,) or up.shape != (3,)
            or not np.isfinite(forward).all() or not np.isfinite(up).all()
            or not np.isclose(np.linalg.norm(forward), 1, atol=1e-9, rtol=0)
            or not np.isclose(np.linalg.norm(up), 1, atol=1e-9, rtol=0)
            or not np.isclose(np.dot(forward, up), 0, atol=1e-9, rtol=0)):
        raise ValueError("root frame directions must be orthogonal unit vectors")
    # Existing Stair root_up_z predicates explicitly use the root's +Z axis.
    if not np.array_equal(up, [0, 0, 1]):
        raise ValueError("Stair root-up definition requires asset-verified +Z up")
    image_bodies = frame_spec.get("image_body_names", [])
    body_names = profile.runtime_facts["body_names"]
    if (not isinstance(image_bodies, list) or len(image_bodies) != len(set(image_bodies))
            or any(name not in body_names for name in image_bodies)):
        raise ValueError("Stair image bodies must be distinct names from the bound asset")
    return {"robot_profile_id": profile.robot_profile_id, "robot_profile_hash": profile.profile_hash,
            "body_name": body, "forward_axis_body": forward.tolist(), "up_axis_body": up.tolist(),
            "body_names": list(body_names), "image_body_names": list(image_bodies),
            "root_asset": dict(root_asset), "frame_spec_hash": canonical_json_hash(frame_spec)}


def stair_body_image_points(window: MediaWindow, snapshot: Mapping[str, Any], *,
                            root_frame: Mapping[str, Any] | None) -> tuple[dict[str, Any], ...]:
    """Locate asset-selected link origins, without asserting visible anatomy.

    The caller verifies snapshot/image hashes and numeric alignment. The four
    Go2 calf origins distinguish limb identities even when feet cross; their
    projections may still be hidden. Older captures/specifications stay empty.
    """
    if (root_frame is None or not root_frame.get("image_body_names")
            or snapshot.get("schema_version") == "native_rgb_snapshots.v3"):
        return ()
    if (window.identity.get("robot_id") != root_frame["robot_profile_id"]
            or snapshot.get("env_index") != window.identity.get("env_index")
            or snapshot.get("seed") != window.identity.get("runner_seed")
            or set(snapshot.get("articulation_body_names", ())) != set(root_frame["body_names"])):
        raise ValueError("Stair body projection snapshot differs from the bound window")
    pairs = sorted({(step, image.view_id) for image in window.images for step in image.displayed_steps})
    return tuple(row for step, view in pairs for row in project_snapshot_body_links(
        snapshot, control_step=step, view_id=view, body_names=root_frame["image_body_names"]))


def stair_s0_definitions(root_frame: Mapping[str, Any] | None = None) -> tuple[dict[str, str], ...]:
    """Task-owned frame meaning fits the existing public S0 definition contract."""
    if root_frame is None:
        return STAIR_S0_DEFINITIONS
    return (*STAIR_S0_DEFINITIONS, {
        "name": "root_forward_world_z", "unit": "1", "body": "root",
        "reference_frame": "signed world-z component of forward axis; positive points upward, negative downward",
        "source": (f"Asset-verified root={root_frame['body_name']}; body forward={root_frame['forward_axis_body']}, "
                   f"up={root_frame['up_axis_body']}. From base_quat_w (unit wxyz): (R(q) @ forward)_z; "
                   "orientation, not displacement or contact."),
    })


def select_stair_media_steps(episode: Mapping[str, Any], *, available_steps: Sequence[int]) -> dict[str, Any]:
    """Six actual media steps for one already selected, source-bound episode.

    The caller supplies the intersection of valid numeric steps and actual
    two-view samples. Three whole-episode anchors protect rule-independent
    coverage. Process priority is framing only, never severity or causality.
    This neither selects episodes nor changes the numeric sampling catalog.
    """
    is_step = lambda value: isinstance(value, Integral) and not isinstance(value, bool)
    supplied = tuple(available_steps)
    first, last = episode["coverage"]["first_step"], episode["steps"]
    if (not supplied or not all(is_step(s) for s in supplied)
            or tuple(sorted(set(supplied))) != supplied
            or not is_step(first) or not is_step(last) or first < 1 or last < first
            or supplied[0] != first or supplied[-1] != last):
        raise ValueError("Stair media steps must be ordered real samples including episode first and T")
    steps = tuple(int(s) for s in supplied)
    dt = episode.get("dt_s")
    if not isinstance(dt, Real) or isinstance(dt, bool) or not math.isfinite(dt) or dt <= 0:
        raise ValueError("Stair media selection requires the actual positive step dt")
    offset = max(1, math.ceil(0.5 / float(dt)))
    baseline = sorted({steps[0], min(steps, key=lambda s: (abs(2*s-first-last), s)), steps[-1]})

    def before(start):
        earlier = [s for s in steps if s < start]
        target = [s for s in earlier if s <= start-offset]
        return max(target) if target else min(earlier) if earlier else None

    def after(end):
        later = [s for s in steps if s > end]
        target = [s for s in later if s >= end+offset]
        return min(target) if target else max(later) if later else None

    process = {"source": None, "before": None, "middle": None, "after": None}
    priority = {name: rank for rank, name in enumerate(
        ("up_z_negative", "tilt", "unloaded_low_motion", "low_progress"))}
    events = [(index, event) for index, event in enumerate(episode.get("events", []))
              if event.get("code") in priority
              and all(is_step(event.get(k)) for k in ("start", "confirmed", "end"))
              and first <= event["start"] <= event["confirmed"] <= event["end"] <= last
              and event["confirmed"] < last]
    start = middle = end = None
    if events:
        index, event = min(events, key=lambda pair: (priority[pair[1]["code"]], pair[1]["start"],
                          pair[1]["confirmed"], pair[1]["body"]))
        start, middle, end = event["start"], event["confirmed"], event["end"]
        process.update(source=f"events/{index}", code=event["code"], body=event["body"],
                       selection_meaning="framing priority, not severity or cause")
    else:
        motion = episode.get("root_motion", {})
        minimum = motion.get("extrema", {}).get("up_z")
        retreat = motion.get("max_x_retreat")
        anchor = episode.get("anchor", {})
        if (minimum is not None and len(minimum) == 4 and minimum[0] < minimum[2]
                and is_step(minimum[1]) and first <= minimum[1] <= last):
            start = middle = end = minimum[1]
            process.update(source="root_motion/extrema/up_z", selection_meaning="observed minimum, not a failure predicate")
        elif (retreat is not None and len(retreat) == 3 and retreat[2] > 0
              and is_step(retreat[0]) and is_step(retreat[1])
              and first <= retreat[0] < retreat[1] <= last):
            start, middle, end = retreat[0], retreat[1], retreat[1]
            process.update(source="root_motion/max_x_retreat", selection_meaning="prior root-x peak to trough, not cause")
        elif anchor.get("status") == "crossed" and is_step(anchor.get("step")) and first <= anchor["step"] <= last:
            start = middle = end = anchor["step"]
            process.update(source="anchor/step", selection_meaning="root position threshold crossing, not foot contact")
    if start is not None:
        process.update(start_step=int(start), middle_step=int(middle), end_step=int(end),
                       before=before(start), middle=int(middle) if middle in steps else None, after=after(end),
                       after_meaning="one actual sample strictly after the interval/anchor; missing if none exists")
    selected = set(baseline) | {process[k] for k in ("before", "middle", "after") if process[k] is not None}
    fill = []
    while len(selected) < min(6, len(steps)):
        step = max((s for s in steps if s not in selected), key=lambda s: (min(abs(s-t) for t in selected), -s))
        selected.add(step)
        fill.append(step)
    return {"sampling_rule": "true_rl_agent.stair.media_steps.v1", "selected_steps": sorted(selected),
            "baseline_steps": baseline, "process": process, "fill_steps": fill,
            "step_dt_s": float(dt), "context_offset_steps": offset}


def _stair_samples(identity, control_step, base_pos_w, base_quat_w):
    bound = dict(identity)
    if not all(isinstance(bound.get(key), str) and bound[key] for key in IDENTITY_FIELDS):
        raise ValueError("numeric identity is incomplete")
    if control_step is None:
        return bound, None
    steps = tuple(int(s) if isinstance(s, Integral) and not isinstance(s, bool) else s for s in control_step)
    pos = None if base_pos_w is None else np.array(base_pos_w, dtype=float, copy=True)
    quat = None if base_quat_w is None else np.array(base_quat_w, dtype=float, copy=True)
    if (not steps or any(type(s) is not int or s < 0 for s in steps)
            or tuple(sorted(set(steps))) != steps
            or (pos is not None and (pos.shape != (len(steps), 3) or not np.isfinite(pos).all()))
            or (quat is not None and (quat.shape != (len(steps), 4) or not np.isfinite(quat).all()
                                     or not np.allclose(np.linalg.norm(quat, axis=1), 1, atol=1e-3)))):
        raise ValueError("stair numeric valid-step arrays do not correspond")
    return bound, {step: (None if pos is None else pos[i], None if quat is None else quat[i])
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


def stair_s0_for_window(window: MediaWindow, *, identity: Mapping[str, Any],
                        control_step: Sequence[int] | None, base_pos_w: Any | None,
                        base_quat_w: Any | None, terrain_origin_w: Any | None = None,
                        root_frame: Mapping[str, Any] | None = None
                        ) -> tuple[dict[str, Any], ...]:
    """Display scalars from one verified episode; missing fields stay absent.

    The caller supplies valid-step arrays and the static terrain origin from the
    same verified archive. S0 contains no identities, rules or outcome labels.
    """
    bound, samples = _stair_samples(identity, control_step, base_pos_w, base_quat_w)
    if any(window.identity.get(key) != bound[key]
           or window.identity.get("numeric_" + key) != bound[key] for key in IDENTITY_FIELDS):
        raise ValueError("stair media and numeric identities differ")
    if root_frame is not None and root_frame["robot_profile_id"] != bound["robot_id"]:
        raise ValueError("stair root frame belongs to another robot")
    definitions = stair_s0_definitions(root_frame)
    origin = None if terrain_origin_w is None else np.asarray(terrain_origin_w, dtype=float)
    if origin is not None and (origin.shape != (3,) or not np.isfinite(origin).all()):
        raise ValueError("stair terrain origin must be one finite world xyz vector")
    result = []
    for step in sorted({s for image in window.images for s in image.displayed_steps}):
        if samples is None or step not in samples:
            continue
        pos, quat = samples[step]
        if quat is not None:
            result.append({**STAIR_S0_DEFINITIONS[0], "step": step,
                           "value": float(1 - 2 * (quat[1] ** 2 + quat[2] ** 2))})
            if root_frame is not None:
                w, x, y, z = quat
                rotation_world_z = np.array([2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y)])
                result.append({**definitions[-1], "step": step,
                               "value": float(rotation_world_z @ root_frame["forward_axis_body"])})
        if pos is not None and origin is not None:
            for definition, axis in zip(STAIR_S0_DEFINITIONS[1:], (0, 2)):
                result.append({**definition, "step": step, "value": float(pos[axis] - origin[axis])})
    return tuple(result)


def stair_pose_facts(window: MediaWindow) -> tuple[dict[str, Any], ...]:
    """Render at most three measured root dimensions from task-constructed S0.

    Forward elevation exists only when the caller supplied the asset-bound
    forward S0 field. No limb orientation or continuous motion is inferred.
    """
    names = {"root_forward_world_z", "root_up_world_z", "root_terrain_z"}
    definitions = {row["name"]: row for row in window.s0_definitions if row["name"] in names}
    displayed = {step for image in window.images for step in image.displayed_steps}
    values: dict[str, dict[int, float]] = {name: {} for name in names}
    for scalar in window.s0:
        name = scalar.get("name")
        if name not in names:
            continue
        step, value = scalar.get("step"), scalar.get("value")
        definition = definitions.get(name)
        expected_unit = "m" if name == "root_terrain_z" else "1"
        if (definition is None or any(scalar.get(key) != expected for key, expected in definition.items())
                or scalar.get("body") != "root" or scalar.get("unit") != expected_unit
                or type(step) is not int or step not in displayed or step in values[name]
                or isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)):
            raise ValueError("Stair pose facts require distinct, finite task S0 at displayed steps")
        value = float(value)
        if expected_unit == "1":
            # Float32 unit-quaternion arithmetic can drift just outside [-1, 1].
            # Only roundoff is clamped; larger errors are not normalized away.
            if abs(value) > 1.0 + 1e-5:
                raise ValueError("Stair root axis component exceeds unit-vector roundoff tolerance")
            value = max(-1.0, min(1.0, value))
        values[name][step] = value
    facts = []
    for name, identifier, label, transform, unit, precision, boundary in (
        ("root_forward_world_z", "stair.root_forward_elevation",
         "Asset-bound root forward-axis elevation above world horizontal",
         lambda value: math.degrees(math.asin(value)), "deg", 1, "Positive is upward; negative downward."),
        ("root_up_world_z", "stair.root_up_world_z_angle", "Root up-axis angle from world +Z",
         lambda value: math.degrees(math.acos(value)), "deg", 1, "Root orientation only."),
        ("root_terrain_z", "stair.root_terrain_height", "Root world z relative to terrain origin",
         float, "m", 3, "Not foot clearance."),
    ):
        if not values[name]:
            continue
        samples = sorted(values[name].items())
        rendered = "; ".join(f"{step}: {transform(value):.{precision}f}" for step, value in samples)
        facts.append({"id": identifier, "body": "root",
                      "text": f"{label} at displayed steps (approx {unit}): {rendered}. {boundary}",
                      "scalar_refs": [{"name": name, "step": step} for step, _ in samples]})
    return tuple(facts)


def stair_numeric_check(*, identity: Mapping[str, Any], control_step: Sequence[int] | None,
                        base_pos_w: Any | None, base_quat_w: Any | None):
    """Bind valid rows from one verified episode; check only cited step/view pairs.

    Media relation/continuity validation is owned by the common observer audit.
    The task compares the measured predicate with its explicit asserted polarity,
    without treating uncited frames or the free-form visual text as evidence.
    """
    bound, samples = _stair_samples(identity, control_step, base_pos_w, base_quat_w)

    def check(observation: Mapping[str, Any], window: MediaWindow) -> dict[str, Any] | None:
        claim = observation.get("numeric_claim")
        if (not isinstance(claim, Mapping) or set(claim) != {"id", "asserted"}
                or not isinstance(claim["id"], str) or type(claim["asserted"]) is not bool):
            return None
        claim_id, asserted = claim["id"], claim["asserted"]
        expected = next((row for row in STAIR_MEASUREMENT_CLAIMS if row["id"] == claim_id), None)
        if expected is None or observation.get("body") != "root" or expected not in window.measurement_claims:
            return None
        media = window.identity
        if any(media.get(key) is None for key in IDENTITY_FIELDS):
            return None
        if any(media[key] != bound[key] for key in IDENTITY_FIELDS):
            raise ValueError("stair media and numeric identities differ")
        shown = _cited_steps(observation, window)
        if samples is None or not shown or any(step not in samples for step in shown):
            return None
        if claim_id == "root_up_z_negative":
            if any(samples[s][1] is None for s in shown):
                return None
            value = min(float(1 - 2 * (samples[s][1][1] ** 2 + samples[s][1][2] ** 2)) for s in shown)
            threshold = 0.0
        else:
            if len(shown) < 2 or samples[shown[0]][0] is None or samples[shown[-1]][0] is None:
                return None
            value = float(abs(samples[shown[-1]][0][0] - samples[shown[0]][0][0]))
            threshold = 0.02
        observed = value < threshold
        return {"status": "consistent" if observed == asserted else "conflict",
                "numeric_evidence": {"claim": claim_id, "asserted": asserted, "observed": observed,
                                     "cited_steps": shown,
                                     "evidence_refs": [dict(ref) for ref in observation["evidence_refs"]],
                                     "value": value, "threshold": threshold}}

    return check


def attach_stair_visual_observations(full: Mapping[str, Any], audited: Mapping[str, Any],
                                     *, observer_version: str) -> dict[str, Any]:
    """Attach audited media to a complete Stair process derivation."""
    result = deepcopy(dict(full))
    result["observer_evidence"] = deepcopy(dict(audited))
    domain = result["domain"]
    episodes = {row["episode_id"]: row for row in result["episodes"]}
    public_audit = deepcopy(dict(audited))
    approved = {}
    for record in public_audit["observations"]:
        identity = record.get("media_identity") or {}
        episode = episodes.get(identity.get("episode_id"))
        valid = (episode is not None
                 and all(identity.get(key) and identity.get(key) == identity.get("numeric_" + key)
                         for key in IDENTITY_FIELDS)
                 and identity["source_id"] == domain["archive_hash"]
                 and identity["checkpoint_id"] == domain["checkpoint_hash"]
                 and identity["protocol_id"] == domain["task_protocol_hash"]
                 and identity["run_id"] == str(episode["run"])
                 and identity["episode_id"] == f"r{episode['run']}.e{episode['slot']}")
        if valid:
            alias = record["alias"]
            if alias in approved and approved[alias] != identity["episode_id"]:
                record.setdefault("violations", []).append("source_binding")
            else:
                approved[alias] = identity["episode_id"]
        else:
            record.setdefault("violations", []).append("source_binding")
    # An alias is usable only if every record for it has the approved binding.
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
    semantic = result["semantic"]
    exemplar = next((row.get("scope", {}).get("label") for row in semantic["sentences"]
                     if row.get("id") == "process-example"), None)
    failed = {label for label, episode in episodes.items()
              if isinstance(reason := episode.get("primary_reason"), str)
              and reason.strip() and reason != "success"}
    priority = {alias: 0 if label == exemplar else 1 if label in failed else 2
                for alias, label in approved.items()}
    rows = visual_process_sentences(public_audit, episode_labels=approved,
                                    cohort_episode_n=len(episodes), observer_version=observer_version,
                                    evidence_ref="observer_evidence", episode_priority=priority)
    semantic["sentences"] = [row for row in semantic["sentences"]
                             if row.get("section") != "visual" and row.get("id") != "visual-unavailable"] + rows
    semantic["sampling"]["visual_observed_episode_n"] = (
        0 if public_audit.get("violations") else len(set(approved.values())))
    return result


def _stair_snapshot_joint_indices(manifest, *, numeric_joint_names, joint_ids,
                                  native_count, numeric_count):
    indices = None if joint_ids is None else np.asarray(joint_ids)
    if indices is not None and (
            indices.ndim != 1 or len(indices) != numeric_count or not len(indices)
            or indices.dtype.kind not in "iu" or len(set(indices.tolist())) != len(indices)
            or np.any(indices < 0) or np.any(indices >= native_count)):
        return None, {"status": "mismatch", "reason": "resolved joint_ids do not index snapshot articulation"}
    native_names = manifest.get("articulation_joint_names")
    if numeric_joint_names is None or native_names is None:
        if indices is None:
            return None, {"status": "missing", "reason": "resolved joint_ids or both recorded joint-name sequences required"}
        return indices, {"joint_binding_source": "explicit_resolved_joint_ids"}
    for names, count in ((numeric_joint_names, numeric_count), (native_names, native_count)):
        if (not isinstance(names, Sequence) or isinstance(names, (str, bytes))
                or len(names) != count or not names
                or any(not isinstance(name, str) or not name for name in names)
                or len(set(names)) != len(names)):
            return None, {"status": "mismatch", "reason": "joint names must be unique and match their array column count"}
    lookup = {name: index for index, name in enumerate(native_names)}
    if any(name not in lookup for name in numeric_joint_names):
        return None, {"status": "missing" if indices is None else "mismatch",
                      "reason": "numeric joint name absent from recorded articulation names"}
    named_indices = np.asarray([lookup[name] for name in numeric_joint_names], dtype=np.int64)
    if indices is not None and not np.array_equal(indices, named_indices):
        return None, {"status": "mismatch", "reason": "explicit joint_ids disagree with recorded joint names"}
    return named_indices, {"joint_binding_source": (
        "recorded_joint_names" if indices is None else "explicit_joint_ids_checked_against_recorded_names")}


def align_stair_snapshot_samples(
    manifest: Mapping[str, Any], *, control_step: Sequence[int],
    base_pos_w: Any | None = None, base_quat_w: Any | None = None,
    joint_pos: Any | None = None, joint_ids: Sequence[int] | None = None,
    numeric_joint_names: Sequence[str] | None = None,
    kinematic_base_source: str | None = None, episode_length: Any | None = None,
) -> dict[str, Any]:
    """Compare final native samples to caller-bound, valid one-episode NPZ rows.

    The caller owns manifest/env/seed binding and valid-row selection. Root pose
    requires an articulation-root binding. Numeric joint order comes from actual
    resolved joint_ids or numeric runtime_projection.joint_names matched to the
    snapshot's recorded articulation names. If both exist they must agree.
    Direct copies must equal after casting JSON back to the NPZ dtype. This
    function does not read files or validate PNG contents.
    """
    steps = np.asarray(control_step)
    if steps.ndim != 1 or (len(steps) and steps.dtype.kind not in "iu") or len(set(steps.tolist())) != len(steps):
        raise ValueError("stair alignment needs distinct integer control steps")
    rows = {int(step): i for i, step in enumerate(steps)}
    arrays = {name: None if value is None else np.asarray(value) for name, value in (
        ("base_pos_w", base_pos_w), ("base_quat_w", base_quat_w),
        ("joint_pos", joint_pos), ("episode_length", episode_length))}
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
        for field, source in (("episode_length", None), ("base_pos_w", "root_pos_w"),
                              ("base_quat_w", "root_quat_w"), ("joint_pos", "joint_pos")):
            result = {"field": field, "snapshot_field": source or "control_step", "atol": 0.0, "rtol": 0.0}
            fields.append(result)
            array = arrays[field]
            if array is None or (source is not None and source not in state):
                result.update(status="missing", reason="numeric or snapshot field absent")
                continue
            if array.ndim < 1 or array.shape[0] != len(steps):
                result.update(status="mismatch", reason="numeric rows do not match control_step")
                continue
            if field in ("base_pos_w", "base_quat_w") and kinematic_base_source != "articulation_root":
                result.update(status="missing" if kinematic_base_source is None else "unsupported",
                              reason="root pose requires actual articulation_root binding")
                continue
            observed = np.asarray(step if source is None else state[source])
            expected = array[index]
            if field == "base_quat_w":
                if "xyzw" not in manifest.get("state_conventions", {}).get("root_quat_w", ""):
                    result.update(status="unsupported", reason="snapshot quaternion convention is not declared xyzw")
                    continue
                if observed.shape == (4,):
                    observed = observed[[3, 0, 1, 2]]
            if field == "joint_pos":
                if observed.ndim != 1 or np.ndim(expected) != 1:
                    result.update(status="mismatch", reason="joint positions must be one vector per sample")
                    continue
                indices, binding = _stair_snapshot_joint_indices(
                    manifest, numeric_joint_names=numeric_joint_names, joint_ids=joint_ids,
                    native_count=len(observed), numeric_count=len(expected))
                result.update(binding)
                if indices is None:
                    continue
                result["joint_ids"] = indices.tolist()
                observed = observed[indices]
            width = {"base_pos_w": (3,), "base_quat_w": (4,), "episode_length": ()}.get(field)
            if (observed.shape != np.shape(expected) or (width is not None and observed.shape != width)
                    or array.dtype.kind not in ("iu" if field == "episode_length" else "f")
                    or not np.isfinite(observed).all() or not np.isfinite(expected).all()):
                result.update(status="mismatch", reason="invalid shape, dtype or nonfinite value")
                continue
            observed = observed.astype(array.dtype)
            error = float(np.max(np.abs(observed.astype(np.float64) - np.asarray(expected, dtype=np.float64))))
            result.update(status="aligned" if np.array_equal(observed, expected) else "mismatch",
                          max_abs_error=error, comparison_dtype=str(array.dtype), comparison="dtype_exact")
    statuses = [field["status"] for sample in samples for field in sample["fields"]]
    status = ("mismatch" if "mismatch" in statuses else
              "overlap_aligned" if statuses and all(s == "aligned" for s in statuses) else "incomplete")
    return {"scope": "stair_root_pose_and_bound_joints_only", "status": status,
            "samples": samples, "not_checked": ["root velocities", "joint_vel", "contact", "PNG pixels"],
            "binding_scope": "caller verified same manifest/env/seed and one episode valid rows"}
