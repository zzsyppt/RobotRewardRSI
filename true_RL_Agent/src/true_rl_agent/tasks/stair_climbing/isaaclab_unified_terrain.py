"""Isaac Lab terrain and curriculum runtime for unified Stair training."""

from __future__ import annotations

from typing import Any

import numpy as np
import trimesh
from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.utils import configclass

from true_rl_agent.tasks.stair_climbing.isaaclab_irregular_terrain import (
    build_irregular_stair_meshes,
)
from true_rl_agent.tasks.stair_climbing.isaaclab_terrain import (
    OneWayStairsTerrainCfg,
    one_way_stairs_terrain,
)
from true_rl_agent.tasks.stair_climbing.unified_training import (
    STAIR_TERRAIN_REGIME_C0,
    StairUnifiedTrainingProfile,
    goal_progress_m_by_terrain_type as goal_progress_m_by_terrain_type,
    unified_assignment_summary,
    unified_initial_terrain_assignment,
)


def unified_irregular_training_stairs_terrain(
    difficulty: float,
    cfg: UnifiedIrregularTrainingStairsTerrainCfg,
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate one C1 column with riser difficulty independent of tread."""

    difficulty_value = min(1.0, max(0.0, float(difficulty)))
    riser_base = cfg.riser_base_range[0] + difficulty_value * (
        cfg.riser_base_range[1] - cfg.riser_base_range[0]
    )
    values = _rotate(tuple(cfg.pattern_multipliers), cfg.phase_steps)
    tread_values = _rotate(tuple(reversed(cfg.pattern_multipliers)), cfg.phase_steps + 1)
    risers = tuple(
        round(riser_base + value * cfg.riser_offset, 6) for value in values
    )
    treads = tuple(
        round(cfg.tread_base + value * cfg.tread_offset, 6)
        for value in tread_values
    )
    top_platform_length = cfg.size[0] - cfg.start_platform_length - sum(treads)
    return build_irregular_stair_meshes(
        size=tuple(cfg.size),
        floor_depth=float(cfg.floor_depth),
        start_platform_length=float(cfg.start_platform_length),
        riser_heights_m=risers,
        tread_depths_m=treads,
        top_platform_length=float(top_platform_length),
    )


@configclass
class UnifiedIrregularTrainingStairsTerrainCfg(SubTerrainBaseCfg):
    """One pattern/tread column in the unified C1 training support."""

    function = unified_irregular_training_stairs_terrain

    pattern_id: str = ""
    phase_steps: int = 0
    floor_depth: float = 0.0
    start_platform_length: float = 0.0
    riser_base_range: tuple[float, float] = (0.0, 0.0)
    tread_base: float = 0.0
    riser_offset: float = 0.0
    tread_offset: float = 0.0
    pattern_multipliers: tuple[float, ...] = ()


def build_unified_train_terrain_generator(
    profile: StairUnifiedTrainingProfile,
    *,
    seed: int,
) -> TerrainGeneratorCfg:
    """Build the frozen 10x10 C0/C1 terrain grid."""

    if not isinstance(profile, StairUnifiedTrainingProfile):
        raise TypeError("unified Stair terrain requires its task-owned profile")
    terrain = profile.terrain
    grid = profile.grid
    size = tuple(float(value) for value in terrain["size_m"])
    c0 = terrain["c0"]
    c1 = terrain["c1"]
    columns = profile.column_descriptors()
    proportion = 1.0 / len(columns)
    sub_terrains: dict[str, SubTerrainBaseCfg] = {}
    for column in columns:
        column_id = str(column["column_id"])
        if column["terrain_regime_id"] == STAIR_TERRAIN_REGIME_C0:
            sub_terrains[column_id] = OneWayStairsTerrainCfg(
                function=one_way_stairs_terrain,
                proportion=proportion,
                step_height_range=tuple(
                    float(value) for value in c0["riser_range_m"]
                ),
                floor_depth=float(terrain["floor_depth_m"]),
                start_platform_length=float(terrain["start_platform_length_m"]),
                step_count=int(terrain["step_count"]),
                tread_depth=float(c0["tread_depth_m"]),
                top_platform_length=(
                    size[0]
                    - float(terrain["start_platform_length_m"])
                    - int(terrain["step_count"]) * float(c0["tread_depth_m"])
                ),
            )
            continue
        pattern_id = str(column["pattern_id"])
        pattern = profile.pattern(pattern_id)
        sub_terrains[column_id] = UnifiedIrregularTrainingStairsTerrainCfg(
            function=unified_irregular_training_stairs_terrain,
            proportion=proportion,
            pattern_id=pattern_id,
            phase_steps=int(column["phase_steps"]),
            floor_depth=float(terrain["floor_depth_m"]),
            start_platform_length=float(terrain["start_platform_length_m"]),
            riser_base_range=tuple(
                float(value) for value in c1["riser_base_range_m"]
            ),
            tread_base=float(column["tread_base_m"]),
            riser_offset=float(c1["riser_offset_m"]),
            tread_offset=float(c1["tread_offset_m"]),
            pattern_multipliers=tuple(
                float(value) for value in pattern["multipliers"]
            ),
        )
    if len(sub_terrains) != 10:
        raise RuntimeError("unified Stair terrain column count drifted")
    return TerrainGeneratorCfg(
        seed=int(seed),
        curriculum=True,
        size=size,
        border_width=0.0,
        num_rows=int(grid["terrain_rows"]),
        num_cols=int(grid["terrain_cols"]),
        # Uniform material: target trimesh lacks Lab's height-color "jet" map.
        color_scheme="none",
        sub_terrains=sub_terrains,
        difficulty_range=tuple(float(value) for value in grid["difficulty_range"]),
        use_cache=False,
    )


def apply_unified_initial_terrain_assignment(
    env: Any,
    *,
    seed: int,
    profile: StairUnifiedTrainingProfile,
) -> dict[str, Any]:
    """Write only Isaac Lab's existing terrain type/level/origin buffers."""

    import torch

    terrain = env.scene.terrain
    terrain_types = terrain.terrain_types
    terrain_levels = terrain.terrain_levels
    types, levels = unified_initial_terrain_assignment(
        num_envs=int(terrain_types.numel()), seed=seed, profile=profile
    )
    type_tensor = torch.tensor(types, dtype=terrain_types.dtype, device=terrain_types.device)
    level_tensor = torch.tensor(
        levels, dtype=terrain_levels.dtype, device=terrain_levels.device
    )
    terrain_types.copy_(type_tensor)
    terrain_levels.copy_(level_tensor)
    terrain.env_origins.copy_(terrain.terrain_origins[level_tensor, type_tensor])
    return unified_assignment_summary(types=types, levels=levels, profile=profile)


def stair_goal_progress_levels(
    env: Any,
    env_ids: Any,
    *,
    goal_progress_m_by_terrain_type: tuple[float, ...],
    move_up_threshold: float,
    move_down_threshold: float,
    asset_cfg: Any,
) -> Any:
    """Move only the terrain level using actual start-to-goal progress."""

    import torch

    terrain = env.scene.terrain
    types_before = terrain.terrain_types[env_ids].clone()
    goals = torch.tensor(
        goal_progress_m_by_terrain_type,
        dtype=terrain.env_origins.dtype,
        device=terrain.env_origins.device,
    )
    if goals.numel() != int(terrain.terrain_origins.shape[1]):
        raise RuntimeError("unified Stair goal map does not cover terrain types")
    root_pos = env.scene[asset_cfg.name].data.root_pos_w[env_ids]
    forward_progress = root_pos[:, 0] - terrain.env_origins[env_ids, 0]
    goal_progress = torch.clamp(
        forward_progress / goals[types_before], min=0.0, max=1.0
    )
    completed_control_steps = env.episode_length_buf[env_ids] > 0
    move_up = completed_control_steps & (
        goal_progress >= float(move_up_threshold)
    )
    move_down = completed_control_steps & (
        goal_progress < float(move_down_threshold)
    )
    terrain.update_env_origins(env_ids, move_up, move_down)
    if not torch.equal(types_before, terrain.terrain_types[env_ids]):
        raise RuntimeError("unified Stair curriculum changed terrain type")
    return torch.mean(terrain.terrain_levels.float())


def _rotate(values: tuple[float, ...], count: int) -> tuple[float, ...]:
    offset = int(count) % len(values)
    return values[offset:] + values[:offset]


__all__ = [
    "UnifiedIrregularTrainingStairsTerrainCfg",
    "apply_unified_initial_terrain_assignment",
    "build_unified_train_terrain_generator",
    "stair_goal_progress_levels",
    "unified_irregular_training_stairs_terrain",
]
