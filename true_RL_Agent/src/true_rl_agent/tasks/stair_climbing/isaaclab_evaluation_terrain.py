"""Production-only expanded Stair evaluation terrain for Isaac Lab."""

from __future__ import annotations

from typing import Any, Mapping

from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

from true_rl_agent.tasks.stair_climbing.evaluation_profile import (
    StairEvaluationProfile,
)
from true_rl_agent.tasks.stair_climbing.ordered_evaluation import (
    OrderedStairEvaluationCell,
)
from true_rl_agent.tasks.stair_climbing.isaaclab_irregular_terrain import (
    IrregularOneWayStairsTerrainCfg,
    irregular_one_way_stairs_terrain,
)
from true_rl_agent.tasks.stair_climbing.isaaclab_rough_terrain import (
    RoughOneWayStairsTerrainCfg,
    rough_one_way_stairs_terrain,
)
from true_rl_agent.tasks.stair_climbing.isaaclab_terrain import (
    OneWayStairsTerrainCfg,
    one_way_stairs_terrain,
)
from true_rl_agent.tasks.stair_climbing.unified_training import (
    STAIR_TERRAIN_REGIME_C2,
)


def build_profile_eval_terrain_generator(
    protocol: Mapping[str, Any],
    *,
    profile: StairEvaluationProfile,
    seed: int,
) -> TerrainGeneratorCfg:
    """Build one deterministic terrain column per frozen evaluation cell."""

    if not isinstance(profile, StairEvaluationProfile):
        raise TypeError("Stair eval terrain requires StairEvaluationProfile")
    terrain = protocol.get("terrain")
    if not isinstance(terrain, Mapping):
        raise ValueError("Stair protocol terrain must be an object")
    proportion = 1.0 / len(profile.cells)
    sub_terrains = {
        cell.cell_id: OneWayStairsTerrainCfg(
            function=one_way_stairs_terrain,
            proportion=proportion,
            step_height_range=(cell.height_m, cell.height_m),
            floor_depth=float(terrain["floor_depth_m"]),
            start_platform_length=float(terrain["start_platform_length_m"]),
            step_count=int(terrain["step_count"]),
            tread_depth=cell.tread_depth_m,
            top_platform_length=cell.top_platform_length_m,
        )
        for cell in profile.cells
    }
    return TerrainGeneratorCfg(
        seed=int(seed),
        curriculum=True,
        size=tuple(float(value) for value in terrain["size_m"]),
        border_width=0.0,
        num_rows=1,
        num_cols=len(profile.cells),
        # Uniform material: target trimesh lacks Lab's height-color "jet" map.
        color_scheme="none",
        sub_terrains=sub_terrains,
        difficulty_range=(0.0, 1.0),
        use_cache=False,
    )


def build_ordered_eval_terrain_generator(
    *,
    cells: tuple[OrderedStairEvaluationCell, ...],
    seed: int,
) -> TerrainGeneratorCfg:
    """Build one deterministic terrain column per explicit ordered cell."""

    if (
        not isinstance(cells, tuple)
        or not cells
        or any(not isinstance(cell, OrderedStairEvaluationCell) for cell in cells)
    ):
        raise TypeError("Stair ordered eval terrain requires runtime cells")
    if tuple(cell.terrain_type_index for cell in cells) != tuple(range(len(cells))):
        raise ValueError("Stair ordered eval terrain indexes are not contiguous")
    size = cells[0].size_m
    if any(cell.size_m != size for cell in cells):
        raise ValueError("Stair ordered eval cells must share terrain size")
    proportion = 1.0 / len(cells)
    sub_terrains = {
        cell.cell_id: _ordered_sub_terrain_cfg(cell, proportion=proportion)
        for cell in cells
    }
    return TerrainGeneratorCfg(
        seed=int(seed),
        curriculum=True,
        size=size,
        border_width=0.0,
        num_rows=1,
        num_cols=len(cells),
        # Uniform material: target trimesh lacks Lab's height-color "jet" map.
        color_scheme="none",
        sub_terrains=sub_terrains,
        difficulty_range=(0.0, 1.0),
        use_cache=False,
    )


def _ordered_sub_terrain_cfg(
    cell: OrderedStairEvaluationCell,
    *,
    proportion: float,
) -> IrregularOneWayStairsTerrainCfg | RoughOneWayStairsTerrainCfg:
    common = {
        "proportion": proportion,
        "cell_id": cell.cell_id,
        "geometry_digest": cell.geometry_digest,
        "floor_depth": cell.floor_depth_m,
        "start_platform_length": cell.start_platform_length_m,
        "riser_heights_m": cell.riser_heights_m,
        "tread_depths_m": cell.tread_depths_m,
        "top_platform_length": cell.top_platform_length_m,
        "top_platform_goal_offset_m": cell.top_platform_goal_offset_m,
    }
    if cell.terrain_regime_id == STAIR_TERRAIN_REGIME_C2:
        return RoughOneWayStairsTerrainCfg(
            function=rough_one_way_stairs_terrain,
            surface_profile=dict(cell.surface_profile),
            **common,
        )
    return IrregularOneWayStairsTerrainCfg(
        function=irregular_one_way_stairs_terrain,
        **common,
    )


__all__ = [
    "build_ordered_eval_terrain_generator",
    "build_profile_eval_terrain_generator",
]
