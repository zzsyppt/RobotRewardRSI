"""Task-owned Stair geometry used by evaluation evidence projection."""

from __future__ import annotations

import math
from typing import Any, Mapping


def expected_stair_surface_height_m(
    *,
    progress_m: float,
    step_height_m: float,
    protocol: Mapping[str, Any],
    tread_depth_m: float | None = None,
) -> float:
    """Return the frozen tread surface under a local +x progress coordinate."""

    if not math.isfinite(progress_m) or not math.isfinite(step_height_m):
        raise ValueError("stair surface inputs must be finite")
    if step_height_m <= 0.0:
        raise ValueError("stair step height must be positive")
    level = stair_surface_level(
        progress_m=progress_m,
        protocol=protocol,
        tread_depth_m=tread_depth_m,
    )
    return float(level) * step_height_m


def stair_surface_level(
    *,
    progress_m: float,
    protocol: Mapping[str, Any],
    tread_depth_m: float | None = None,
) -> int:
    """Return the physical tread level at one local progress coordinate."""

    if not math.isfinite(progress_m):
        raise ValueError("stair progress must be finite")
    terrain = protocol.get("terrain")
    if not isinstance(terrain, Mapping):
        raise ValueError("stair terrain protocol is missing")
    start_platform = float(terrain["start_platform_length_m"])
    tread_depth = (
        float(terrain["tread_depth_m"])
        if tread_depth_m is None
        else float(tread_depth_m)
    )
    if not math.isfinite(tread_depth) or tread_depth <= 0.0:
        raise ValueError("stair tread depth must be positive and finite")
    step_count = int(terrain["step_count"])
    local_origin_x = start_platform / 2.0
    local_x = progress_m + local_origin_x
    level = math.ceil((local_x - start_platform) / tread_depth)
    level = min(step_count, max(0, level))
    return int(level)


def supporting_stair_surface_height_m(
    *,
    foot_progress_m: tuple[float, float],
    foot_contacts: tuple[bool, bool],
    foot_surface_clearance_m: tuple[float, float],
    step_height_m: float,
    maximum_projected_surface_penetration_m: float,
    protocol: Mapping[str, Any],
    tread_depth_m: float | None = None,
) -> float:
    """Return the lowest vertically feasible tread supporting the feet.

    During a riser transition the base may project over the next tread while a
    foot is still below its surface. Combining contact, horizontal projection,
    and foot height avoids treating a riser collision as completed support
    transfer.
    """

    if (
        not isinstance(foot_progress_m, tuple)
        or len(foot_progress_m) != 2
        or not isinstance(foot_contacts, tuple)
        or len(foot_contacts) != 2
        or any(not isinstance(value, bool) for value in foot_contacts)
        or not isinstance(foot_surface_clearance_m, tuple)
        or len(foot_surface_clearance_m) != 2
        or any(not math.isfinite(value) for value in foot_surface_clearance_m)
    ):
        raise ValueError("stair support evidence must contain two feet")
    if not math.isfinite(step_height_m) or step_height_m <= 0.0:
        raise ValueError("stair step height must be positive and finite")
    if (
        isinstance(maximum_projected_surface_penetration_m, bool)
        or not math.isfinite(maximum_projected_surface_penetration_m)
        or maximum_projected_surface_penetration_m < 0.0
    ):
        raise ValueError(
            "stair support surface penetration must be finite and non-negative"
        )
    levels: list[int] = []
    for progress_m, projected_clearance_m in zip(
        foot_progress_m,
        foot_surface_clearance_m,
        strict=True,
    ):
        level = stair_surface_level(
            progress_m=progress_m,
            protocol=protocol,
            tread_depth_m=tread_depth_m,
        )
        clearance_m = projected_clearance_m
        while (
            level > 0
            and clearance_m < -maximum_projected_surface_penetration_m
        ):
            level -= 1
            clearance_m += step_height_m
        levels.append(level)
    contacting_levels = tuple(
        level for level, contact in zip(levels, foot_contacts, strict=True) if contact
    )
    support_level = min(contacting_levels or levels)
    return float(support_level) * step_height_m


__all__ = [
    "expected_stair_surface_height_m",
    "stair_surface_level",
    "supporting_stair_surface_height_m",
]
