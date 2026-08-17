"""Production-only expanded Stair evaluation terrain for Isaac Lab."""

from __future__ import annotations

from typing import Any, Mapping

from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

from true_rl_agent.tasks.stair_climbing.evaluation_profile import (
    StairEvaluationProfile,
)
from true_rl_agent.tasks.stair_climbing.isaaclab_terrain import (
    OneWayStairsTerrainCfg,
    one_way_stairs_terrain,
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
        color_scheme="height",
        sub_terrains=sub_terrains,
        difficulty_range=(0.0, 1.0),
        use_cache=False,
    )


__all__ = ["build_profile_eval_terrain_generator"]
