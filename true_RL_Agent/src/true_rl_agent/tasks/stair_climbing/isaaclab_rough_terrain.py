"""Production C2 Stair mesh with bounded, queryable tread roughness."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import trimesh
from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from isaaclab.utils import configclass

from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.tasks.stair_climbing.ordered_evaluation import (
    ordered_stair_geometry_payload,
)
from true_rl_agent.tasks.stair_climbing.rough_surface import (
    STAIR_C2_SURFACE_KIND,
    normalize_stair_surface_profile,
    rough_surface_vertex_offset_m,
)


def rough_one_way_stairs_terrain(
    difficulty: float,
    cfg: RoughOneWayStairsTerrainCfg,
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate one digest-bound C2 course; difficulty cannot mutate geometry."""

    del difficulty
    profile = _validate_rough_terrain_cfg(cfg)
    return build_rough_stair_meshes(
        size=tuple(float(value) for value in cfg.size),
        floor_depth=float(cfg.floor_depth),
        start_platform_length=float(cfg.start_platform_length),
        riser_heights_m=tuple(float(value) for value in cfg.riser_heights_m),
        tread_depths_m=tuple(float(value) for value in cfg.tread_depths_m),
        top_platform_length=float(cfg.top_platform_length),
        surface_profile=profile,
    )


def build_rough_stair_meshes(
    *,
    size: tuple[float, float],
    floor_depth: float,
    start_platform_length: float,
    riser_heights_m: tuple[float, ...],
    tread_depths_m: tuple[float, ...],
    top_platform_length: float,
    surface_profile: Mapping[str, Any],
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Build closed C2 solids whose top triangles define the evidence surface."""

    size_x, size_y = size
    profile = normalize_stair_surface_profile(surface_profile)
    if profile["kind"] != STAIR_C2_SURFACE_KIND:
        raise ValueError("C2 Stair mesh requires a rough surface profile")
    if (
        size_x <= 0.0
        or size_y <= 0.0
        or floor_depth <= 0.0
        or start_platform_length <= 0.0
        or top_platform_length <= 0.0
        or not riser_heights_m
        or len(riser_heights_m) != len(tread_depths_m)
    ):
        raise ValueError("C2 Stair mesh geometry is invalid")
    meshes: list[trimesh.Trimesh] = [
        _box(
            extents=(size_x, size_y, floor_depth),
            center=(size_x / 2.0, size_y / 2.0, -floor_depth / 2.0),
        )
    ]
    segment_start = start_platform_length
    cumulative_height = 0.0
    for tread_index, (tread_depth, riser_height) in enumerate(
        zip(tread_depths_m, riser_heights_m, strict=True)
    ):
        if tread_depth <= 0.0 or riser_height <= 0.0:
            raise ValueError("C2 Stair step dimensions must be positive")
        cumulative_height += riser_height
        meshes.append(
            _rough_tread_solid(
                x_start=segment_start,
                tread_depth=float(tread_depth),
                size_y=float(size_y),
                bottom_z=-float(floor_depth),
                nominal_top_z=float(cumulative_height),
                tread_index=tread_index,
                surface_profile=profile,
            )
        )
        segment_start += tread_depth
    if not np.isclose(segment_start + top_platform_length, size_x, atol=1.0e-9):
        raise ValueError("C2 Stair mesh segments do not fill terrain length")
    meshes.append(
        _box(
            extents=(top_platform_length, size_y, floor_depth + cumulative_height),
            center=(
                segment_start + top_platform_length / 2.0,
                size_y / 2.0,
                (cumulative_height - floor_depth) / 2.0,
            ),
        )
    )
    return meshes, np.array(
        [start_platform_length / 2.0, size_y / 2.0, 0.0], dtype=np.float64
    )


@configclass
class RoughOneWayStairsTerrainCfg(SubTerrainBaseCfg):
    """Configuration for one digest-bound C2 evaluation course."""

    function = rough_one_way_stairs_terrain

    cell_id: str = ""
    geometry_digest: str = ""
    floor_depth: float = 0.0
    start_platform_length: float = 0.0
    riser_heights_m: tuple[float, ...] = ()
    tread_depths_m: tuple[float, ...] = ()
    top_platform_length: float = 0.0
    top_platform_goal_offset_m: float = 0.0
    surface_profile: dict[str, Any] | None = None


def _validate_rough_terrain_cfg(
    cfg: RoughOneWayStairsTerrainCfg,
) -> dict[str, Any]:
    if len(cfg.size) != 2 or min(cfg.size) <= 0.0:
        raise ValueError("C2 Stair terrain size must be positive")
    if not cfg.cell_id or not cfg.geometry_digest:
        raise ValueError("C2 Stair terrain identity is missing")
    if cfg.floor_depth <= 0.0 or cfg.start_platform_length <= 0.0:
        raise ValueError("C2 Stair floor/start geometry is invalid")
    if (
        not cfg.riser_heights_m
        or len(cfg.riser_heights_m) != len(cfg.tread_depths_m)
        or any(value <= 0.0 for value in cfg.riser_heights_m)
        or any(value <= 0.0 for value in cfg.tread_depths_m)
    ):
        raise ValueError("C2 Stair step sequences are invalid")
    if cfg.top_platform_length <= cfg.top_platform_goal_offset_m:
        raise ValueError("C2 Stair top platform is too short")
    occupied = (
        cfg.start_platform_length + sum(cfg.tread_depths_m) + cfg.top_platform_length
    )
    if not np.isclose(occupied, cfg.size[0], atol=1.0e-9):
        raise ValueError("C2 Stair segments must fill the terrain x dimension")
    profile = normalize_stair_surface_profile(cfg.surface_profile)
    if profile["kind"] != STAIR_C2_SURFACE_KIND:
        raise ValueError("C2 Stair terrain requires roughness")
    expected_digest = canonical_json_hash(
        ordered_stair_geometry_payload(
            size_m=tuple(float(value) for value in cfg.size),
            floor_depth_m=float(cfg.floor_depth),
            start_platform_length_m=float(cfg.start_platform_length),
            riser_heights_m=tuple(float(value) for value in cfg.riser_heights_m),
            tread_depths_m=tuple(float(value) for value in cfg.tread_depths_m),
            top_platform_goal_offset_m=float(cfg.top_platform_goal_offset_m),
            surface_profile=profile,
        )
    )
    if expected_digest != cfg.geometry_digest:
        raise ValueError("C2 Stair runtime geometry digest mismatch")
    return profile


def _rough_tread_solid(
    *,
    x_start: float,
    tread_depth: float,
    size_y: float,
    bottom_z: float,
    nominal_top_z: float,
    tread_index: int,
    surface_profile: Mapping[str, Any],
) -> trimesh.Trimesh:
    nx = int(surface_profile["longitudinal_vertices"])
    ny = int(surface_profile["lateral_vertices"])
    vertices: list[tuple[float, float, float]] = []
    for ix in range(nx):
        u = ix / (nx - 1)
        for iy in range(ny):
            v = iy / (ny - 1)
            vertices.append(
                (
                    x_start + u * tread_depth,
                    v * size_y,
                    nominal_top_z
                    + rough_surface_vertex_offset_m(
                        surface_profile,
                        tread_index=tread_index,
                        longitudinal_index=ix,
                        lateral_index=iy,
                    ),
                )
            )
    vertices.extend(
        (
            x_start + ix / (nx - 1) * tread_depth,
            iy / (ny - 1) * size_y,
            bottom_z,
        )
        for ix in range(nx)
        for iy in range(ny)
    )

    def top(ix: int, iy: int) -> int:
        return ix * ny + iy

    def bottom(ix: int, iy: int) -> int:
        return nx * ny + ix * ny + iy

    faces: list[tuple[int, int, int]] = []
    for ix in range(nx - 1):
        for iy in range(ny - 1):
            v00 = top(ix, iy)
            v10 = top(ix + 1, iy)
            v01 = top(ix, iy + 1)
            v11 = top(ix + 1, iy + 1)
            faces.extend(((v00, v10, v11), (v00, v11, v01)))
            b00 = bottom(ix, iy)
            b10 = bottom(ix + 1, iy)
            b01 = bottom(ix, iy + 1)
            b11 = bottom(ix + 1, iy + 1)
            faces.extend(((b00, b11, b10), (b00, b01, b11)))

    for ix in range(nx - 1):
        left = top(ix, 0)
        right = top(ix + 1, 0)
        bottom_left = bottom(ix, 0)
        bottom_right = bottom(ix + 1, 0)
        faces.extend(
            ((bottom_left, right, left), (bottom_left, bottom_right, right))
        )
        left = top(ix, ny - 1)
        right = top(ix + 1, ny - 1)
        bottom_left = bottom(ix, ny - 1)
        bottom_right = bottom(ix + 1, ny - 1)
        faces.extend(
            ((bottom_left, left, right), (bottom_left, right, bottom_right))
        )
    for iy in range(ny - 1):
        low = top(0, iy)
        high = top(0, iy + 1)
        bottom_low = bottom(0, iy)
        bottom_high = bottom(0, iy + 1)
        faces.extend(((bottom_low, low, high), (bottom_low, high, bottom_high)))
        low = top(nx - 1, iy)
        high = top(nx - 1, iy + 1)
        bottom_low = bottom(nx - 1, iy)
        bottom_high = bottom(nx - 1, iy + 1)
        faces.extend(((bottom_low, high, low), (bottom_low, bottom_high, high)))
    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    if not mesh.is_watertight:
        raise ValueError("C2 Stair tread mesh is not watertight")
    return mesh


def _box(
    *,
    extents: tuple[float, float, float],
    center: tuple[float, float, float],
) -> trimesh.Trimesh:
    transform = np.eye(4)
    transform[:3, -1] = np.asarray(center, dtype=np.float64)
    return trimesh.creation.box(extents=extents, transform=transform)


__all__ = [
    "RoughOneWayStairsTerrainCfg",
    "build_rough_stair_meshes",
    "rough_one_way_stairs_terrain",
]
