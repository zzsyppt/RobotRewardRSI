"""Task-owned bounded C2 tread surface shared by mesh and evidence projection."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

STAIR_FLAT_SURFACE_KIND = "flat"
STAIR_C2_SURFACE_KIND = "bounded_piecewise_linear_v1"

_ROUGH_FIELDS = {
    "kind",
    "amplitude_m",
    "longitudinal_cycles",
    "lateral_cycles",
    "phase_index",
    "phase_count",
    "step_phase_stride",
    "longitudinal_vertices",
    "lateral_vertices",
}


def normalize_stair_surface_profile(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and copy one physical surface descriptor into canonical JSON data."""

    if not isinstance(value, Mapping):
        raise TypeError("Stair surface profile must be an object")
    if value.get("kind") == STAIR_FLAT_SURFACE_KIND:
        if set(value) != {"kind"}:
            raise ValueError("flat Stair surface profile fields mismatch")
        return {"kind": STAIR_FLAT_SURFACE_KIND}
    if value.get("kind") != STAIR_C2_SURFACE_KIND or set(value) != _ROUGH_FIELDS:
        raise ValueError("C2 Stair surface profile fields mismatch")

    amplitude = _finite_number(value["amplitude_m"], "amplitude")
    # The surface function defines a shape, not a robot-independent difficulty.
    # The owning cell bounds amplitude against its actual stair dimensions.
    if amplitude <= 0.0:
        raise ValueError("C2 Stair amplitude must be positive")
    longitudinal_cycles = _bounded_int(
        value["longitudinal_cycles"], "longitudinal cycles", minimum=1, maximum=128
    )
    lateral_cycles = _bounded_int(
        value["lateral_cycles"], "lateral cycles", minimum=1, maximum=128
    )
    phase_count = _bounded_int(
        value["phase_count"], "phase count", minimum=2, maximum=32
    )
    phase_index = _bounded_int(
        value["phase_index"],
        "phase index",
        minimum=0,
        maximum=phase_count - 1,
    )
    step_phase_stride = _bounded_int(
        value["step_phase_stride"],
        "step phase stride",
        minimum=0,
        maximum=phase_count - 1,
    )
    longitudinal_vertices = _bounded_int(
        value["longitudinal_vertices"],
        "longitudinal vertices",
        minimum=3,
        maximum=513,
    )
    lateral_vertices = _bounded_int(
        value["lateral_vertices"],
        "lateral vertices",
        minimum=3,
        maximum=513,
    )
    if longitudinal_vertices < 2 * longitudinal_cycles + 1:
        raise ValueError("C2 Stair longitudinal grid undersamples its carrier")
    if lateral_vertices < 2 * lateral_cycles + 1:
        raise ValueError("C2 Stair lateral grid undersamples its carrier")
    return {
        "kind": STAIR_C2_SURFACE_KIND,
        "amplitude_m": amplitude,
        "longitudinal_cycles": longitudinal_cycles,
        "lateral_cycles": lateral_cycles,
        "phase_index": phase_index,
        "phase_count": phase_count,
        "step_phase_stride": step_phase_stride,
        "longitudinal_vertices": longitudinal_vertices,
        "lateral_vertices": lateral_vertices,
    }


def rough_surface_vertex_offset_m(
    profile: Mapping[str, Any],
    *,
    tread_index: int,
    longitudinal_index: int,
    lateral_index: int,
) -> float:
    """Return one frozen C2 mesh-vertex offset from the nominal tread plane."""

    canonical = normalize_stair_surface_profile(profile)
    if canonical["kind"] != STAIR_C2_SURFACE_KIND:
        raise ValueError("rough Stair vertex requires a C2 surface profile")
    return _rough_surface_vertex_offset_canonical(
        canonical,
        tread_index=tread_index,
        longitudinal_index=longitudinal_index,
        lateral_index=lateral_index,
    )


def _rough_surface_vertex_offset_canonical(
    canonical: Mapping[str, Any],
    *,
    tread_index: int,
    longitudinal_index: int,
    lateral_index: int,
) -> float:
    tread = _bounded_int(tread_index, "tread index", minimum=0, maximum=10_000)
    nx = int(canonical["longitudinal_vertices"])
    ny = int(canonical["lateral_vertices"])
    ix = _bounded_int(
        longitudinal_index,
        "longitudinal index",
        minimum=0,
        maximum=nx - 1,
    )
    iy = _bounded_int(
        lateral_index,
        "lateral index",
        minimum=0,
        maximum=ny - 1,
    )
    u = ix / (nx - 1)
    v = iy / (ny - 1)
    envelope = math.sin(math.pi * u) * math.sin(math.pi * v)
    phase = (
        int(canonical["phase_index"])
        + tread * int(canonical["step_phase_stride"])
    ) / int(canonical["phase_count"])
    carrier = math.sin(
        2.0
        * math.pi
        * (
            int(canonical["longitudinal_cycles"]) * u
            + int(canonical["lateral_cycles"]) * v
            + phase
        )
    )
    value = float(canonical["amplitude_m"]) * envelope * carrier
    if abs(value) > float(canonical["amplitude_m"]) + 1.0e-12:
        raise AssertionError("C2 Stair surface exceeded its analytic amplitude")
    return float(value)


def rough_surface_offset_m(
    profile: Mapping[str, Any],
    *,
    tread_index: int,
    longitudinal_fraction: float,
    lateral_fraction: float,
) -> float:
    """Evaluate the exact piecewise-linear surface used by the triangle mesh."""

    canonical = normalize_stair_surface_profile(profile)
    if canonical["kind"] != STAIR_C2_SURFACE_KIND:
        return 0.0
    u = _unit_coordinate(longitudinal_fraction, "longitudinal fraction")
    v = _unit_coordinate(lateral_fraction, "lateral fraction")
    nx = int(canonical["longitudinal_vertices"])
    ny = int(canonical["lateral_vertices"])
    x_scaled = u * (nx - 1)
    y_scaled = v * (ny - 1)
    ix = min(nx - 2, int(math.floor(x_scaled)))
    iy = min(ny - 2, int(math.floor(y_scaled)))
    fx = x_scaled - ix
    fy = y_scaled - iy
    if u == 1.0:
        fx = 1.0
    if v == 1.0:
        fy = 1.0

    z00 = _rough_surface_vertex_offset_canonical(
        canonical,
        tread_index=tread_index,
        longitudinal_index=ix,
        lateral_index=iy,
    )
    z10 = _rough_surface_vertex_offset_canonical(
        canonical,
        tread_index=tread_index,
        longitudinal_index=ix + 1,
        lateral_index=iy,
    )
    z01 = _rough_surface_vertex_offset_canonical(
        canonical,
        tread_index=tread_index,
        longitudinal_index=ix,
        lateral_index=iy + 1,
    )
    z11 = _rough_surface_vertex_offset_canonical(
        canonical,
        tread_index=tread_index,
        longitudinal_index=ix + 1,
        lateral_index=iy + 1,
    )
    if fy <= fx:
        return float((1.0 - fx) * z00 + (fx - fy) * z10 + fy * z11)
    return float((1.0 - fy) * z00 + fx * z11 + (fy - fx) * z01)


def _finite_number(value: Any, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"C2 Stair {field_name} must be finite")
    return float(value)


def _bounded_int(value: Any, field_name: str, *, minimum: int, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not minimum <= value <= maximum
    ):
        raise ValueError(f"C2 Stair {field_name} is outside its bounded range")
    return int(value)


def _unit_coordinate(value: Any, field_name: str) -> float:
    result = _finite_number(value, field_name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"C2 Stair {field_name} must be in [0, 1]")
    return result


__all__ = [
    "STAIR_C2_SURFACE_KIND",
    "STAIR_FLAT_SURFACE_KIND",
    "normalize_stair_surface_profile",
    "rough_surface_offset_m",
    "rough_surface_vertex_offset_m",
]
