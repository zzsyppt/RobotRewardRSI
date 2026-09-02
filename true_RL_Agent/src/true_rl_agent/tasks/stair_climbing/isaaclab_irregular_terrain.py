"""Shared Isaac Lab geometry primitives for flat-tread C1 Stair samples."""

from __future__ import annotations

import numpy as np
import trimesh
from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from isaaclab.utils import configclass

from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.tasks.stair_climbing.ordered_evaluation import (
    ordered_stair_geometry_payload,
)


def irregular_one_way_stairs_terrain(
    difficulty: float,
    cfg: IrregularOneWayStairsTerrainCfg,
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate one explicit flat-tread C1 course without scalar approximation."""

    del difficulty
    _validate_irregular_terrain_cfg(cfg)
    return build_irregular_stair_meshes(
        size=tuple(cfg.size),
        floor_depth=float(cfg.floor_depth),
        start_platform_length=float(cfg.start_platform_length),
        riser_heights_m=tuple(cfg.riser_heights_m),
        tread_depths_m=tuple(cfg.tread_depths_m),
        top_platform_length=float(cfg.top_platform_length),
    )


def build_irregular_stair_meshes(
    *,
    size: tuple[float, float],
    floor_depth: float,
    start_platform_length: float,
    riser_heights_m: tuple[float, ...],
    tread_depths_m: tuple[float, ...],
    top_platform_length: float,
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Build flat-tread geometry shared by C1 training and evaluation."""

    size_x, size_y = size
    if (
        size_x <= 0.0
        or size_y <= 0.0
        or floor_depth <= 0.0
        or start_platform_length <= 0.0
        or top_platform_length <= 0.0
        or not riser_heights_m
        or len(riser_heights_m) != len(tread_depths_m)
    ):
        raise ValueError("C1 Stair mesh geometry is invalid")
    meshes: list[trimesh.Trimesh] = [
        _box(
            extents=(size_x, size_y, floor_depth),
            center=(size_x / 2.0, size_y / 2.0, -floor_depth / 2.0),
        )
    ]
    segment_start = start_platform_length
    cumulative_height = 0.0
    for tread_depth, riser_height in zip(tread_depths_m, riser_heights_m, strict=True):
        if tread_depth <= 0.0 or riser_height <= 0.0:
            raise ValueError("C1 Stair mesh step dimensions must be positive")
        cumulative_height += riser_height
        meshes.append(
            _box(
                extents=(tread_depth, size_y, floor_depth + cumulative_height),
                center=(
                    segment_start + tread_depth / 2.0,
                    size_y / 2.0,
                    (cumulative_height - floor_depth) / 2.0,
                ),
            )
        )
        segment_start += tread_depth
    if not np.isclose(segment_start + top_platform_length, size_x):
        raise ValueError("C1 Stair mesh segments do not fill terrain length")
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
class IrregularOneWayStairsTerrainCfg(SubTerrainBaseCfg):
    """Configuration for one digest-bound C1 evaluation course."""

    function = irregular_one_way_stairs_terrain

    cell_id: str = ""
    geometry_digest: str = ""
    floor_depth: float = 0.0
    start_platform_length: float = 0.0
    riser_heights_m: tuple[float, ...] = ()
    tread_depths_m: tuple[float, ...] = ()
    top_platform_length: float = 0.0
    top_platform_goal_offset_m: float = 0.0


def _validate_irregular_terrain_cfg(cfg: IrregularOneWayStairsTerrainCfg) -> None:
    if len(cfg.size) != 2 or min(cfg.size) <= 0.0:
        raise ValueError("irregular Stair terrain size must be positive")
    if not cfg.cell_id or not cfg.geometry_digest:
        raise ValueError("irregular Stair terrain identity is missing")
    if cfg.floor_depth <= 0.0 or cfg.start_platform_length <= 0.0:
        raise ValueError("irregular Stair floor/start geometry is invalid")
    if (
        not cfg.riser_heights_m
        or len(cfg.riser_heights_m) != len(cfg.tread_depths_m)
        or any(value <= 0.0 for value in cfg.riser_heights_m)
        or any(value <= 0.0 for value in cfg.tread_depths_m)
    ):
        raise ValueError("irregular Stair step sequences are invalid")
    if cfg.top_platform_length <= cfg.top_platform_goal_offset_m:
        raise ValueError("irregular Stair top platform is too short")
    occupied = (
        cfg.start_platform_length + sum(cfg.tread_depths_m) + cfg.top_platform_length
    )
    if not np.isclose(occupied, cfg.size[0], atol=1.0e-9):
        raise ValueError("irregular Stair segments must fill the terrain x dimension")
    expected_digest = canonical_json_hash(
        ordered_stair_geometry_payload(
            size_m=tuple(float(value) for value in cfg.size),
            floor_depth_m=float(cfg.floor_depth),
            start_platform_length_m=float(cfg.start_platform_length),
            riser_heights_m=tuple(float(value) for value in cfg.riser_heights_m),
            tread_depths_m=tuple(float(value) for value in cfg.tread_depths_m),
            top_platform_goal_offset_m=float(cfg.top_platform_goal_offset_m),
            surface_profile={"kind": "flat"},
        )
    )
    if expected_digest != cfg.geometry_digest:
        raise ValueError("irregular Stair runtime geometry digest mismatch")


def _box(
    *,
    extents: tuple[float, float, float],
    center: tuple[float, float, float],
) -> trimesh.Trimesh:
    transform = np.eye(4)
    transform[:3, -1] = np.asarray(center, dtype=np.float64)
    return trimesh.creation.box(extents=extents, transform=transform)


__all__ = [
    "IrregularOneWayStairsTerrainCfg",
    "build_irregular_stair_meshes",
    "irregular_one_way_stairs_terrain",
]
