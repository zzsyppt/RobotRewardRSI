"""Task-owned Stair surface queries used by evaluation evidence projection."""

from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

from true_rl_agent.tasks.stair_climbing.ordered_evaluation import (
    OrderedStairCellSpec,
)
from true_rl_agent.tasks.stair_climbing.rough_surface import (
    STAIR_C2_SURFACE_KIND,
    STAIR_FLAT_SURFACE_KIND,
    normalize_stair_surface_profile,
    rough_surface_offset_m,
)


@dataclass(frozen=True)
class StairSurfaceSample:
    level: int
    height_m: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.level, bool)
            or not isinstance(self.level, int)
            or self.level < 0
        ):
            raise ValueError("Stair surface level must be non-negative")
        if not math.isfinite(self.height_m):
            raise ValueError("Stair surface height must be finite")


@runtime_checkable
class StairSurfaceQuery(Protocol):
    """Task-owned seam implemented by each concrete Stair surface."""

    def sample(self, *, progress_m: float, lateral_m: float) -> StairSurfaceSample: ...

    def height_for_level_m(
        self,
        *,
        level: int,
        progress_m: float,
        lateral_m: float,
    ) -> float: ...


@dataclass(frozen=True)
class FlatStairSurfaceQuery:
    """Exact query for the current flat, equal-height Stair geometry."""

    protocol: Mapping[str, Any]
    step_height_m: float
    tread_depth_m: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.protocol, Mapping):
            raise TypeError("Stair surface protocol must be an object")
        if not math.isfinite(self.step_height_m) or self.step_height_m <= 0.0:
            raise ValueError("Stair step height must be positive and finite")
        if self.tread_depth_m is not None and (
            not math.isfinite(self.tread_depth_m) or self.tread_depth_m <= 0.0
        ):
            raise ValueError("Stair tread depth must be positive and finite")

    def sample(self, *, progress_m: float, lateral_m: float) -> StairSurfaceSample:
        _finite_coordinate(progress_m, "progress")
        _finite_coordinate(lateral_m, "lateral")
        level = stair_surface_level(
            progress_m=progress_m,
            protocol=self.protocol,
            tread_depth_m=self.tread_depth_m,
        )
        return StairSurfaceSample(
            level=level,
            height_m=self.height_for_level_m(
                level=level,
                progress_m=progress_m,
                lateral_m=lateral_m,
            ),
        )

    def height_for_level_m(
        self,
        *,
        level: int,
        progress_m: float,
        lateral_m: float,
    ) -> float:
        _finite_coordinate(progress_m, "progress")
        _finite_coordinate(lateral_m, "lateral")
        terrain = self.protocol.get("terrain")
        if not isinstance(terrain, Mapping):
            raise ValueError("Stair terrain protocol is missing")
        step_count = int(terrain["step_count"])
        if (
            isinstance(level, bool)
            or not isinstance(level, int)
            or level < 0
            or level > step_count
        ):
            raise ValueError("Stair requested surface level is outside geometry")
        return float(level) * self.step_height_m


@dataclass(frozen=True)
class IrregularFlatStairSurfaceQuery:
    """Exact analytic query for one explicit C1 unequal-riser course."""

    cell: OrderedStairCellSpec

    def __post_init__(self) -> None:
        if not isinstance(self.cell, OrderedStairCellSpec):
            raise TypeError("irregular Stair surface query requires an ordered cell")

    def sample(self, *, progress_m: float, lateral_m: float) -> StairSurfaceSample:
        _finite_coordinate(progress_m, "progress")
        _finite_coordinate(lateral_m, "lateral")
        local_x = progress_m + self.cell.start_platform_length_m / 2.0
        level = min(
            self.cell.step_count,
            bisect_left(self.cell.cumulative_tread_boundaries_m, local_x),
        )
        return StairSurfaceSample(
            level=level,
            height_m=self.height_for_level_m(
                level=level,
                progress_m=progress_m,
                lateral_m=lateral_m,
            ),
        )

    def height_for_level_m(
        self,
        *,
        level: int,
        progress_m: float,
        lateral_m: float,
    ) -> float:
        _finite_coordinate(progress_m, "progress")
        _finite_coordinate(lateral_m, "lateral")
        if (
            isinstance(level, bool)
            or not isinstance(level, int)
            or level < 0
            or level > self.cell.step_count
        ):
            raise ValueError("irregular Stair requested surface level is outside geometry")
        return self.cell.cumulative_riser_heights_m[level]


@dataclass(frozen=True)
class RoughStairSurfaceQuery:
    """Exact query for the same piecewise-linear C2 triangles used by Isaac."""

    cell: OrderedStairCellSpec
    _surface_profile: Mapping[str, Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.cell, OrderedStairCellSpec):
            raise TypeError("rough Stair surface query requires an ordered cell")
        profile = normalize_stair_surface_profile(self.cell.surface_profile)
        if profile["kind"] != STAIR_C2_SURFACE_KIND:
            raise ValueError("rough Stair surface query requires C2 geometry")
        object.__setattr__(self, "_surface_profile", profile)

    def sample(self, *, progress_m: float, lateral_m: float) -> StairSurfaceSample:
        _finite_coordinate(progress_m, "progress")
        _finite_coordinate(lateral_m, "lateral")
        local_x = progress_m + self.cell.start_platform_length_m / 2.0
        level = min(
            self.cell.step_count,
            bisect_left(self.cell.cumulative_tread_boundaries_m, local_x),
        )
        return StairSurfaceSample(
            level=level,
            height_m=self.height_for_level_m(
                level=level,
                progress_m=progress_m,
                lateral_m=lateral_m,
            ),
        )

    def height_for_level_m(
        self,
        *,
        level: int,
        progress_m: float,
        lateral_m: float,
    ) -> float:
        _finite_coordinate(progress_m, "progress")
        _finite_coordinate(lateral_m, "lateral")
        if (
            isinstance(level, bool)
            or not isinstance(level, int)
            or level < 0
            or level > self.cell.step_count
        ):
            raise ValueError("rough Stair requested surface level is outside geometry")
        nominal = self.cell.cumulative_riser_heights_m[level]
        if level == 0:
            return nominal
        local_x = progress_m + self.cell.start_platform_length_m / 2.0
        boundaries = self.cell.cumulative_tread_boundaries_m
        physical_level = min(self.cell.step_count, bisect_left(boundaries, local_x))
        if physical_level != level or not (
            boundaries[level - 1] <= local_x <= boundaries[level]
        ):
            return nominal
        tread_depth = self.cell.tread_depths_m[level - 1]
        u = min(1.0, max(0.0, (local_x - boundaries[level - 1]) / tread_depth))
        local_y = lateral_m + self.cell.size_m[1] / 2.0
        v = min(1.0, max(0.0, local_y / self.cell.size_m[1]))
        return nominal + rough_surface_offset_m(
            self._surface_profile,
            tread_index=level - 1,
            longitudinal_fraction=u,
            lateral_fraction=v,
        )


def stair_surface_query_for_ordered_cell(
    cell: OrderedStairCellSpec,
) -> StairSurfaceQuery:
    """Resolve the task-owned physical surface without a runner/shared-layer branch."""

    if not isinstance(cell, OrderedStairCellSpec):
        raise TypeError("Stair ordered surface query requires an ordered cell")
    profile = normalize_stair_surface_profile(cell.surface_profile)
    if profile["kind"] == STAIR_FLAT_SURFACE_KIND:
        return IrregularFlatStairSurfaceQuery(cell)
    if profile["kind"] == STAIR_C2_SURFACE_KIND:
        return RoughStairSurfaceQuery(cell)
    raise ValueError("Stair ordered surface kind is unsupported")


def expected_stair_surface_height_m(
    *,
    progress_m: float,
    step_height_m: float,
    protocol: Mapping[str, Any],
    tread_depth_m: float | None = None,
    lateral_m: float = 0.0,
) -> float:
    """Compatibility helper backed by the real task-owned query interface."""

    return (
        FlatStairSurfaceQuery(
            protocol=protocol,
            step_height_m=step_height_m,
            tread_depth_m=tread_depth_m,
        )
        .sample(progress_m=progress_m, lateral_m=lateral_m)
        .height_m
    )


def stair_surface_level(
    *,
    progress_m: float,
    protocol: Mapping[str, Any],
    tread_depth_m: float | None = None,
) -> int:
    """Return the physical tread level at one local progress coordinate."""

    _finite_coordinate(progress_m, "progress")
    terrain = protocol.get("terrain")
    if not isinstance(terrain, Mapping):
        raise ValueError("Stair terrain protocol is missing")
    start_platform = float(terrain["start_platform_length_m"])
    tread_depth = (
        float(terrain["tread_depth_m"])
        if tread_depth_m is None
        else float(tread_depth_m)
    )
    if not math.isfinite(tread_depth) or tread_depth <= 0.0:
        raise ValueError("Stair tread depth must be positive and finite")
    step_count = int(terrain["step_count"])
    local_origin_x = start_platform / 2.0
    local_x = progress_m + local_origin_x
    level = math.ceil((local_x - start_platform) / tread_depth)
    return int(min(step_count, max(0, level)))


def supporting_stair_surface_height_m(
    *,
    support_progress_m: tuple[float, ...],
    support_lateral_m: tuple[float, ...],
    support_contacts: tuple[bool, ...],
    support_surface_clearance_m: tuple[float, ...],
    maximum_projected_surface_penetration_m: float,
    surface_query: StairSurfaceQuery,
) -> float:
    """Return the lowest vertically feasible surface supporting all effectors."""

    support_count = _support_cardinality(
        support_progress_m,
        support_lateral_m,
        support_contacts,
        support_surface_clearance_m,
    )
    if not isinstance(surface_query, StairSurfaceQuery):
        raise TypeError("Stair surface query does not implement the query contract")
    if (
        isinstance(maximum_projected_surface_penetration_m, bool)
        or not math.isfinite(maximum_projected_surface_penetration_m)
        or maximum_projected_surface_penetration_m < 0.0
    ):
        raise ValueError(
            "Stair support surface penetration must be finite and non-negative"
        )

    feasible_heights: list[float] = []
    for index in range(support_count):
        progress_m = support_progress_m[index]
        lateral_m = support_lateral_m[index]
        sample = surface_query.sample(
            progress_m=progress_m,
            lateral_m=lateral_m,
        )
        level = sample.level
        height_m = sample.height_m
        clearance_m = support_surface_clearance_m[index]
        while level > 0 and clearance_m < -maximum_projected_surface_penetration_m:
            lower_height_m = surface_query.height_for_level_m(
                level=level - 1,
                progress_m=progress_m,
                lateral_m=lateral_m,
            )
            clearance_m += height_m - lower_height_m
            level -= 1
            height_m = lower_height_m
        feasible_heights.append(height_m)
    contacting_heights = tuple(
        height
        for height, contact in zip(
            feasible_heights,
            support_contacts,
            strict=True,
        )
        if contact
    )
    return min(contacting_heights or tuple(feasible_heights))


def _support_cardinality(
    progress: Any,
    lateral: Any,
    contacts: Any,
    clearances: Any,
) -> int:
    values = (progress, lateral, contacts, clearances)
    if any(not isinstance(item, tuple) or not item for item in values):
        raise ValueError("Stair support evidence must contain non-empty tuples")
    lengths = {len(item) for item in values}
    if len(lengths) != 1:
        raise ValueError("Stair support evidence cardinality mismatch")
    if any(not isinstance(item, bool) for item in contacts):
        raise ValueError("Stair support contacts must be boolean")
    for field_name, sequence in (
        ("progress", progress),
        ("lateral", lateral),
        ("clearance", clearances),
    ):
        for item in sequence:
            _finite_coordinate(item, field_name)
    return len(progress)


def _finite_coordinate(value: Any, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
    ):
        raise ValueError(f"Stair surface {field_name} must be finite")
    return float(value)


__all__ = [
    "FlatStairSurfaceQuery",
    "IrregularFlatStairSurfaceQuery",
    "RoughStairSurfaceQuery",
    "StairSurfaceQuery",
    "StairSurfaceSample",
    "expected_stair_surface_height_m",
    "stair_surface_level",
    "stair_surface_query_for_ordered_cell",
    "supporting_stair_surface_height_m",
]
