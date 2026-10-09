"""Named link origins projected through the captured RGB camera.

These are geometric image locations, not visibility or occlusion judgments.
Robot/task owners choose which body names have useful meanings.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def project_snapshot_body_links(
    snapshot: Mapping[str, Any], *, control_step: int, view_id: str,
    body_names: Sequence[str],
) -> list[dict[str, Any]]:
    """Return caller-selected link origins in one captured image's pixel frame.

    ``in_frame`` means only that the projected point is inside the 2D image.
    It does not establish visibility, absence of occlusion, or semantic identity
    of any surrounding pixels. Pixel origin is top-left, +x right and +y down.
    Old v3 captures have no calibration/link poses and return no evidence.
    Malformed v4 data raises ValueError rather than using camera look-at values.
    """
    if snapshot.get("schema_version") == "native_rgb_snapshots.v3":
        return []
    if snapshot.get("schema_version") != "native_rgb_snapshots.v4":
        raise ValueError("body projection requires a native_rgb_snapshots.v4 snapshot")
    samples = [row for row in snapshot.get("samples", ()) if row.get("control_step") == control_step]
    if len(samples) != 1:
        raise ValueError("body projection requires one matching snapshot control step")
    sample = samples[0]
    views = [row for row in sample.get("views", ()) if row.get("view_id") == view_id]
    if len(views) != 1:
        raise ValueError("body projection requires one matching snapshot view")
    view = views[0]
    names = snapshot.get("articulation_body_names", [])
    if not names or len(set(names)) != len(names):
        raise ValueError("body projection requires distinct articulation body names")
    convention = snapshot.get("state_conventions", {}).get("body_link_pos_w", "")
    if not all(word in convention for word in ("world xyz", "metres", "articulation_body_names")):
        raise ValueError("body projection requires world-metre named link positions")
    before, after = sample.get("state_before", {}), sample.get("state_after", {})
    if before.get("articulation_body_names") != names or after.get("articulation_body_names") != names:
        raise ValueError("body projection names do not match the sampled link order")
    points_m = _finite_array(before.get("body_link_pos_w"), (len(names), 3), "body_link_pos_w")
    after_points = _finite_array(after.get("body_link_pos_w"), (len(names), 3), "state_after.body_link_pos_w")
    if (not np.array_equal(points_m, after_points) or not before.get("sha256")
            or before["sha256"] != after.get("sha256")):
        raise ValueError("body projection requires unchanged before/after snapshot state")
    requested = list(body_names)
    if len(set(requested)) != len(requested) or any(name not in names for name in requested):
        raise ValueError("body projection requested names must be distinct sampled body names")

    params = view.get("camera_params", {})
    if params.get("cameraModel") != "pinhole":
        raise ValueError("body projection requires captured pinhole camera_params")
    resolution = _finite_array(params.get("renderProductResolution"), (2,), "renderProductResolution")
    if (not np.equal(resolution, np.floor(resolution)).all() or np.any(resolution <= 0)
            or list(resolution) != snapshot.get("resolution")):
        raise ValueError("body projection camera resolution does not match the snapshot")
    metres_per_unit = _finite_array(params.get("metersPerSceneUnit"), (), "metersPerSceneUnit").item()
    if metres_per_unit <= 0:
        raise ValueError("body projection requires positive metersPerSceneUnit")
    view_matrix = _finite_array(params.get("cameraViewTransform"), (4, 4), "cameraViewTransform")
    projection = _finite_array(params.get("cameraProjection"), (4, 4), "cameraProjection")

    rows = []
    for name in requested:
        # Replicator writers use row vectors for the original matrix order:
        # world -> camera -> clip; NDC y is flipped for top-left image pixels.
        world = np.append(points_m[names.index(name)] / metres_per_unit, 1.0)
        clip = world @ view_matrix @ projection
        if not np.isfinite(clip).all():
            raise ValueError("body projection produced nonfinite clip coordinates")
        xy = None
        if clip[3] < 0:
            status = "behind_camera"
        elif clip[3] == 0:
            status = "on_camera_plane"
        else:
            ndc = clip[:2] / clip[3]
            xy = [float((ndc[0] + 1) * resolution[0] / 2),
                  float((1 - ndc[1]) * resolution[1] / 2)]
            status = "in_frame" if all(0 <= xy[i] < resolution[i] for i in (0, 1)) else "outside_frame"
        rows.append({"body_name": name, "control_step": control_step, "view_id": view_id,
                     "image_xy_px": xy, "projection_status": status})
    return rows


def _finite_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"body projection requires finite {name} with shape {shape}")
    return array
