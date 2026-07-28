"""Task-owned Stair geometry used by evaluation evidence projection."""

from __future__ import annotations

import math
from typing import Any, Mapping


def expected_stair_surface_height_m(
    *,
    progress_m: float,
    step_height_m: float,
    protocol: Mapping[str, Any],
) -> float:
    """Return the frozen tread surface under a local +x progress coordinate."""

    if not math.isfinite(progress_m) or not math.isfinite(step_height_m):
        raise ValueError("stair surface inputs must be finite")
    if step_height_m <= 0.0:
        raise ValueError("stair step height must be positive")
    terrain = protocol.get("terrain")
    if not isinstance(terrain, Mapping):
        raise ValueError("stair terrain protocol is missing")
    start_platform = float(terrain["start_platform_length_m"])
    tread_depth = float(terrain["tread_depth_m"])
    step_count = int(terrain["step_count"])
    local_origin_x = start_platform / 2.0
    local_x = progress_m + local_origin_x
    level = math.ceil((local_x - start_platform) / tread_depth)
    level = min(step_count, max(0, level))
    return float(level) * step_height_m


__all__ = ["expected_stair_surface_height_m"]
