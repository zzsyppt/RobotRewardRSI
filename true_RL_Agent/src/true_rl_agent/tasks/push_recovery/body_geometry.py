"""Asset-bound robot link geometry for Push images; no anatomical pose labels."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math
from numbers import Real
from typing import Any

import numpy as np

from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.evaluation.body_projection import project_snapshot_body_links
from true_rl_agent.evaluation.visual_observer import MediaWindow
from true_rl_agent.robots.base import RobotProfile


def bind_push_body_anchors(spec: Mapping[str, Any] | None, *,
                           runtime_projection: Mapping[str, Any],
                           robot_profile: Mapping[str, Any]) -> dict[str, Any] | None:
    """Bind a reviewed asset specification to the original runtime profile.

    ``runtime_projection`` must come from the run's verified profile binding,
    not a profile hash inferred from its robot name. No profile is modified.
    This small implementation supports only asset-proven child-link origins.
    """
    if spec is None:
        return None
    profile = RobotProfile.from_mapping(robot_profile)
    if (spec.get("schema_version") != "true_rl_agent.robot_body_anchors.v1"
            or any(source.get("robot_profile_id") != profile.robot_profile_id
                   or source.get("robot_profile_hash") != profile.profile_hash
                   for source in (spec, runtime_projection))):
        raise ValueError("body anchors differ from the verified runtime robot profile")
    closure, asset = profile.runtime_facts["asset_closure"], spec.get("root_asset", {})
    if (asset.get("identity") != closure["root_identity"]
            or not any(entry["identity"] == asset.get("identity") and entry["sha256"] == asset.get("sha256")
                       for entry in closure["entries"])):
        raise ValueError("body anchors differ from the bound asset hash")
    names = runtime_projection.get("body_names", [])
    if (not isinstance(names, (list, tuple)) or len(names) != len(set(names))
            or set(names) != set(profile.runtime_facts["body_names"])):
        raise ValueError("body anchor runtime names differ from the bound profile")
    legs, palms = spec.get("legs", {}), spec.get("palm_origins", {})
    if set(legs) != {"left", "right"} or set(palms) != {"left", "right"}:
        raise ValueError("body anchor specification needs both separate sides")
    for side in ("left", "right"):
        if set(legs[side]) != {"H", "K", "A"}:
            raise ValueError("leg geometry requires H, K and A")
        anchors = [*legs[side].values(), palms[side]]
        if len({row["body_name"] for row in anchors}) != len(anchors):
            raise ValueError("body anchor points must identify distinct links")
        for row in anchors:
            if row.get("body_name") not in names or row.get("child_local_position_m") != [0, 0, 0]:
                raise ValueError("body anchors require known links and asset-proven zero child offsets")
        if any(row.get("joint_name") not in profile.runtime_facts["joint_names"] for row in legs[side].values()):
            raise ValueError("leg anchor joints differ from the bound asset profile")
    return {"robot_profile_id": profile.robot_profile_id, "robot_profile_hash": profile.profile_hash,
            "root_asset": dict(asset), "body_names": list(names), "legs": deepcopy(legs),
            "palm_origins": deepcopy(palms), "anchor_spec_hash": canonical_json_hash(spec)}


def push_body_geometry_definitions(binding: Mapping[str, Any] | None) -> tuple[dict[str, str], ...]:
    if binding is None:
        return ()
    return tuple({"name": f"{side}_hka_geometry_deg", "unit": "deg", "body": binding["legs"][side]["K"]["body_name"],
        "reference_frame": "interior angle at K between world-position vectors H-K and A-K",
        "source": (f"Asset-bound link origins H={binding['legs'][side]['H']['body_name']}, "
                   f"K={binding['legs'][side]['K']['body_name']}, A={binding['legs'][side]['A']['body_name']}; "
                   "body_link_pos_w metres; degrees(acos(dot(H-K,A-K)/(norm(H-K)*norm(A-K)))). "
                   "Robot-link geometry, not an anatomical knee angle or whole-body crouch.")}
        for side in ("left", "right"))


def _snapshot_body_names(window: MediaWindow, snapshot: Mapping[str, Any], binding: Mapping[str, Any]) -> list[str]:
    if (snapshot.get("schema_version") != "native_rgb_snapshots.v4"
            or window.identity.get("robot_id") != binding["robot_profile_id"]
            or snapshot.get("env_index") != window.identity.get("env_index")
            or snapshot.get("seed") != window.identity.get("runner_seed")):
        raise ValueError("body geometry snapshot differs from the bound window")
    convention = snapshot.get("state_conventions", {}).get("body_link_pos_w", "")
    if not all(term in convention for term in ("world xyz", "metres", "articulation_body_names")):
        raise ValueError("body geometry requires declared world-metre link positions in named order")
    names = snapshot.get("articulation_body_names", [])
    if len(names) != len(set(names)) or set(names) != set(binding["body_names"]):
        raise ValueError("snapshot body names differ from the bound articulation")
    return names


def push_body_geometry_s0(window: MediaWindow, snapshot: Mapping[str, Any], *,
                          binding: Mapping[str, Any] | None) -> tuple[dict[str, Any], ...]:
    """Measure displayed native samples only; an old missing field stays absent.

    The caller hash-verifies this snapshot and its images before supplying the
    window. Here the episode, names, state equality and geometry are checked.
    """
    if binding is None:
        return ()
    samples = snapshot.get("samples", [])
    if not any("body_link_pos_w" in row.get("state_before", {}) for row in samples):
        return ()
    names = _snapshot_body_names(window, snapshot, binding)
    by_step = {row["control_step"]: row for row in samples}
    steps = sorted({step for image in window.images for step in image.displayed_steps})
    if len(by_step) != len(samples) or not set(steps) <= set(by_step):
        raise ValueError("body geometry needs one native sample per displayed control step")
    definitions = push_body_geometry_definitions(binding)
    result = []
    for step in steps:
        sample = by_step[step]
        if "body_link_pos_w" not in sample.get("state_before", {}):
            continue
        if any(sample.get(key, {}).get("articulation_body_names") != names
               for key in ("state_before", "state_after")):
            raise ValueError("body geometry state name order differs from the snapshot")
        positions = np.asarray(sample["state_before"]["body_link_pos_w"], dtype=float)
        after = np.asarray(sample.get("state_after", {}).get("body_link_pos_w"), dtype=float)
        if (positions.shape != (len(names), 3) or not np.isfinite(positions).all()
                or not np.array_equal(positions, after)):
            raise ValueError("body geometry link positions are invalid or changed during rendering")
        for side, definition in zip(("left", "right"), definitions):
            h, k, a = (positions[names.index(binding["legs"][side][point]["body_name"])] for point in ("H", "K", "A"))
            u, v = h-k, a-k
            denominator = float(np.linalg.norm(u) * np.linalg.norm(v))
            if not math.isfinite(denominator) or denominator <= 0:
                raise ValueError("body geometry angle is undefined for zero-length H-K or A-K")
            angle = math.degrees(math.acos(float(np.clip(np.dot(u, v) / denominator, -1, 1))))
            result.append({**definition, "step": step, "value": angle})
    return tuple(result)


def push_body_image_points(window: MediaWindow, snapshot: Mapping[str, Any], *,
                           binding: Mapping[str, Any] | None) -> tuple[dict[str, Any], ...]:
    """Locate six asset-bound link origins; projected pixels do not prove visibility.

    As for S0, the caller verifies the source snapshot and image hashes. Palm
    link origins locate wrist-side attachments, not palm centres. No root or
    anatomical position is inferred from these selected bodies.
    """
    if binding is None or snapshot.get("schema_version") == "native_rgb_snapshots.v3":
        return ()
    _snapshot_body_names(window, snapshot, binding)
    names = [name for side in ("left", "right") for name in (
        binding["legs"][side]["H"]["body_name"], binding["legs"][side]["K"]["body_name"],
        binding["palm_origins"][side]["body_name"])]
    pairs = sorted({(step, image.view_id) for image in window.images for step in image.displayed_steps})
    return tuple(row for step, view in pairs for row in project_snapshot_body_links(
        snapshot, control_step=step, view_id=view, body_names=names))


def push_body_geometry_facts(window: MediaWindow, *, binding: Mapping[str, Any] | None) -> tuple[dict[str, Any], ...]:
    """One independent sampled H-K-A geometric-angle fact for each side."""
    definitions = push_body_geometry_definitions(binding)
    displayed = {step for image in window.images for step in image.displayed_steps}
    facts = []
    for side, definition in zip(("left", "right"), definitions):
        values = {}
        for row in window.s0:
            if row.get("name") != definition["name"]:
                continue
            step, value = row.get("step"), row.get("value")
            if (any(row.get(key) != expected for key, expected in definition.items())
                    or type(step) is not int or step not in displayed or step in values
                    or isinstance(value, bool) or not isinstance(value, Real)
                    or not math.isfinite(value) or not 0 <= value <= 180):
                raise ValueError("body geometry facts require valid distinct displayed-step scalars")
            values[step] = float(value)
        if values:
            text = "; ".join(f"{step}: {value:.1f}" for step, value in sorted(values.items()))
            facts.append({"id": f"push.{side}_hka_geometry", "body": definition["body"],
                "text": (f"{side.capitalize()} robot-link H-K-A interior angle at displayed steps (approx deg): {text}. "
                         "180 means collinear H-K-A, not anatomical knee extension or whole-body crouch."),
                "scalar_refs": [{"name": definition["name"], "step": step} for step in sorted(values)]})
    return tuple(facts)
