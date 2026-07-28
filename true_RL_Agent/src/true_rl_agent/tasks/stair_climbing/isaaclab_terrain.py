"""Task-owned one-way stair terrain for Isaac Lab.

This module is imported only by Isaac Lab entrypoints after AppLauncher starts.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import trimesh
from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.utils import configclass


def one_way_stairs_terrain(
    difficulty: float,
    cfg: OneWayStairsTerrainCfg,
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate stairs ascending along local +x with origin on the bottom platform."""

    _validate_terrain_cfg(cfg)
    difficulty_value = min(1.0, max(0.0, float(difficulty)))
    lower, upper = cfg.step_height_range
    step_height = lower + difficulty_value * (upper - lower)
    size_x, size_y = cfg.size
    meshes: list[trimesh.Trimesh] = []

    meshes.append(
        _box(
            extents=(size_x, size_y, cfg.floor_depth),
            center=(size_x / 2.0, size_y / 2.0, -cfg.floor_depth / 2.0),
        )
    )

    for level in range(1, cfg.step_count + 1):
        segment_start = cfg.start_platform_length + (level - 1) * cfg.tread_depth
        height = level * step_height
        meshes.append(
            _box(
                extents=(
                    cfg.tread_depth,
                    size_y,
                    cfg.floor_depth + height,
                ),
                center=(
                    segment_start + cfg.tread_depth / 2.0,
                    size_y / 2.0,
                    (height - cfg.floor_depth) / 2.0,
                ),
            )
        )

    top_start = cfg.start_platform_length + cfg.step_count * cfg.tread_depth
    top_height = cfg.step_count * step_height
    meshes.append(
        _box(
            extents=(
                cfg.top_platform_length,
                size_y,
                cfg.floor_depth + top_height,
            ),
            center=(
                top_start + cfg.top_platform_length / 2.0,
                size_y / 2.0,
                (top_height - cfg.floor_depth) / 2.0,
            ),
        )
    )

    origin = np.array(
        [cfg.start_platform_length / 2.0, size_y / 2.0, 0.0],
        dtype=np.float64,
    )
    return meshes, origin


@configclass
class OneWayStairsTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a bottom-origin, single-direction staircase."""

    function = one_way_stairs_terrain

    step_height_range: tuple[float, float] = (0.0, 0.0)
    floor_depth: float = 0.0
    start_platform_length: float = 0.0
    step_count: int = 0
    tread_depth: float = 0.0
    top_platform_length: float = 0.0


def build_eval_terrain_generator(
    protocol: Mapping[str, Any],
    *,
    seed: int,
) -> TerrainGeneratorCfg:
    """Build the deterministic one-column-per-bin evaluation terrain grid."""

    terrain = _terrain_payload(protocol)
    bins = terrain["evaluation_bins"]
    proportion = 1.0 / len(bins)
    sub_terrains = {
        item["bin_id"]: _sub_terrain_cfg(
            terrain,
            step_height_range=(float(item["height_m"]), float(item["height_m"])),
            proportion=proportion,
        )
        for item in bins
    }
    return TerrainGeneratorCfg(
        seed=int(seed),
        curriculum=True,
        size=tuple(float(value) for value in terrain["size_m"]),
        border_width=0.0,
        num_rows=1,
        num_cols=len(bins),
        color_scheme="height",
        sub_terrains=sub_terrains,
        difficulty_range=(0.0, 1.0),
        use_cache=False,
    )


def build_train_terrain_generator(
    protocol: Mapping[str, Any],
    *,
    seed: int,
) -> TerrainGeneratorCfg:
    """Build the seeded training terrain grid from the frozen height range."""

    terrain = _terrain_payload(protocol)
    distribution = terrain["training_distribution"]
    sub_terrain = _sub_terrain_cfg(
        terrain,
        step_height_range=tuple(
            float(value) for value in distribution["step_height_range_m"]
        ),
        proportion=1.0,
    )
    return TerrainGeneratorCfg(
        seed=int(seed),
        curriculum=False,
        size=tuple(float(value) for value in terrain["size_m"]),
        border_width=0.0,
        num_rows=int(distribution["terrain_rows"]),
        num_cols=int(distribution["terrain_cols"]),
        color_scheme="height",
        sub_terrains={"one_way_stairs": sub_terrain},
        difficulty_range=(0.0, 1.0),
        use_cache=False,
    )


def build_geometry_report(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Inspect generated vertices for all frozen evaluation heights."""

    terrain = _terrain_payload(protocol)
    reports: list[dict[str, Any]] = []
    for item in terrain["evaluation_bins"]:
        height = float(item["height_m"])
        cfg = _sub_terrain_cfg(
            terrain,
            step_height_range=(height, height),
            proportion=1.0,
        )
        cfg.size = tuple(float(value) for value in terrain["size_m"])
        meshes, origin = one_way_stairs_terrain(0.5, cfg)
        vertices = np.concatenate([mesh.vertices for mesh in meshes], axis=0)
        observed_z = sorted(
            {
                round(float(value), 8)
                for value in vertices[:, 2]
                if float(value) >= -1e-8
            }
        )
        expected_tops = [
            round(level * height, 8)
            for level in range(int(terrain["step_count"]) + 1)
        ]
        missing_tops = [
            value
            for value in expected_tops
            if not any(abs(value - observed) <= 1e-7 for observed in observed_z)
        ]
        max_height = float(vertices[:, 2].max())
        reports.append(
            {
                "bin_id": str(item["bin_id"]),
                "height_m": height,
                "mesh_count": len(meshes),
                "origin_m": [float(value) for value in origin],
                "expected_top_levels_m": expected_tops,
                "missing_top_levels_m": missing_tops,
                "max_height_m": max_height,
                "is_non_flat": max_height > 0.0,
            }
        )
    return reports


def _terrain_payload(protocol: Mapping[str, Any]) -> Mapping[str, Any]:
    terrain = protocol.get("terrain")
    if not isinstance(terrain, Mapping):
        raise ValueError("stair protocol terrain must be an object")
    return terrain


def _sub_terrain_cfg(
    terrain: Mapping[str, Any],
    *,
    step_height_range: tuple[float, float],
    proportion: float,
) -> OneWayStairsTerrainCfg:
    return OneWayStairsTerrainCfg(
        function=one_way_stairs_terrain,
        proportion=float(proportion),
        step_height_range=step_height_range,
        floor_depth=float(terrain["floor_depth_m"]),
        start_platform_length=float(terrain["start_platform_length_m"]),
        step_count=int(terrain["step_count"]),
        tread_depth=float(terrain["tread_depth_m"]),
        top_platform_length=float(terrain["top_platform_length_m"]),
    )


def _validate_terrain_cfg(cfg: OneWayStairsTerrainCfg) -> None:
    if len(cfg.size) != 2 or min(cfg.size) <= 0.0:
        raise ValueError("one-way stair terrain size must be positive")
    if cfg.floor_depth <= 0.0:
        raise ValueError("one-way stair floor_depth must be positive")
    if cfg.start_platform_length <= 0.0:
        raise ValueError("one-way stair start platform must be positive")
    if cfg.step_count <= 0:
        raise ValueError("one-way stair step_count must be positive")
    if cfg.tread_depth <= 0.0 or cfg.top_platform_length <= 0.0:
        raise ValueError("one-way stair tread and top platform must be positive")
    if (
        cfg.step_height_range[0] <= 0.0
        or cfg.step_height_range[0] > cfg.step_height_range[1]
    ):
        raise ValueError("one-way stair height range is invalid")
    occupied = (
        cfg.start_platform_length
        + cfg.step_count * cfg.tread_depth
        + cfg.top_platform_length
    )
    if not np.isclose(occupied, cfg.size[0], atol=1e-9):
        raise ValueError("one-way stair segments must fill the terrain x dimension")


def _box(
    *,
    extents: tuple[float, float, float],
    center: tuple[float, float, float],
) -> trimesh.Trimesh:
    transform = trimesh.transformations.translation_matrix(center)
    return trimesh.creation.box(extents=extents, transform=transform)


__all__ = [
    "OneWayStairsTerrainCfg",
    "build_eval_terrain_generator",
    "build_geometry_report",
    "build_train_terrain_generator",
    "one_way_stairs_terrain",
]
