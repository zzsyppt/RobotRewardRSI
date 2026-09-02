"""Shared ordered-cell primitives for Stair evaluation suites.

The active suite schema and every ID/OOD claim live in ``unified_evaluation``.
This module only provides immutable geometry and deterministic env assignment;
it does not define a second suite or a terrain-specific lifecycle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.evaluation.protocol import EvaluationProtocol
from true_rl_agent.tasks.stair_climbing.rough_surface import (
    normalize_stair_surface_profile,
)

STAIR_ORDERED_CELL_SET_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.ordered_cell_set.v1"
)
STAIR_ORDERED_COHORT_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.ordered_evaluation_cohort.v1"
)
STAIR_ORDERED_GEOMETRY_SCHEMA_VERSION = "true_rl_agent.task.stair_climbing.geometry.v3"
STAIR_ORDERED_FORMAL_COHORTS = (
    "public_dev",
    "public_accept",
    "sealed_validation",
)
STAIR_ORDERED_RUNNABLE_COHORTS = ("calibration", *STAIR_ORDERED_FORMAL_COHORTS)

_FACTOR_NAMES = (
    "phase",
    "riser_band",
    "tread_band",
    "sequence_pattern",
    "roughness_band",
    "surface_pattern",
)


def ordered_stair_geometry_payload(
    *,
    size_m: tuple[float, float],
    floor_depth_m: float,
    start_platform_length_m: float,
    riser_heights_m: tuple[float, ...],
    tread_depths_m: tuple[float, ...],
    top_platform_goal_offset_m: float,
    surface_profile: Mapping[str, Any],
) -> dict[str, Any]:
    """Return geometry-only identity, independent of cell/cohort labels."""

    total_run = round(sum(tread_depths_m), 10)
    return {
        "schema_version": STAIR_ORDERED_GEOMETRY_SCHEMA_VERSION,
        "size_m": list(size_m),
        "floor_depth_m": floor_depth_m,
        "start_platform_length_m": start_platform_length_m,
        "riser_heights_m": list(riser_heights_m),
        "tread_depths_m": list(tread_depths_m),
        "top_platform_length_m": round(
            size_m[0] - start_platform_length_m - total_run,
            10,
        ),
        "progress_goal_m": round(
            start_platform_length_m / 2.0 + total_run + top_platform_goal_offset_m,
            10,
        ),
        "surface_profile": normalize_stair_surface_profile(surface_profile),
    }


@dataclass(frozen=True)
class OrderedStairCellSpec:
    """One digest-bound Stair course before runtime placement."""

    cell_id: str
    cohort: str
    factor_levels: Mapping[str, str]
    riser_heights_m: tuple[float, ...]
    tread_depths_m: tuple[float, ...]
    size_m: tuple[float, float]
    floor_depth_m: float
    start_platform_length_m: float
    top_platform_goal_offset_m: float
    control_dt_s: float
    anchor_progress_goal_m: float
    anchor_horizon_control_steps: int
    surface_profile: Mapping[str, Any]
    geometry_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.cell_id, str) or not self.cell_id:
            raise ValueError("ordered Stair cell_id must be non-empty")
        if self.cohort not in STAIR_ORDERED_RUNNABLE_COHORTS:
            raise ValueError("ordered Stair cell cohort is invalid")
        if set(self.factor_levels) != set(_FACTOR_NAMES):
            raise ValueError("ordered Stair cell factors mismatch")
        normalize_stair_surface_profile(self.surface_profile)
        if (
            not self.riser_heights_m
            or len(self.riser_heights_m) != len(self.tread_depths_m)
            or any(
                value <= 0.0 or not math.isfinite(value)
                for value in self.riser_heights_m
            )
            or any(
                value <= 0.0 or not math.isfinite(value)
                for value in self.tread_depths_m
            )
        ):
            raise ValueError("ordered Stair cell geometry sequences are invalid")
        is_regular = (
            getattr(self, "terrain_regime_id", "c1_irregular_flat") == "c0_regular"
        )
        if not is_regular and any(
            math.isclose(left, right, rel_tol=0.0, abs_tol=1.0e-12)
            for left, right in zip(self.riser_heights_m, self.riser_heights_m[1:])
        ):
            raise ValueError("ordered Stair adjacent risers must be unequal")
        if not is_regular and any(
            math.isclose(left, right, rel_tol=0.0, abs_tol=1.0e-12)
            for left, right in zip(self.tread_depths_m, self.tread_depths_m[1:])
        ):
            raise ValueError("ordered Stair adjacent treads must be unequal")
        if self.top_platform_length_m <= self.top_platform_goal_offset_m:
            raise ValueError("ordered Stair cell leaves no top-platform goal region")
        if self.progress_goal_m <= 0.0 or self.episode_horizon_control_steps <= 0:
            raise ValueError("ordered Stair cell progress or horizon is invalid")
        if self.geometry_digest != canonical_json_hash(self.geometry_payload()):
            raise ValueError("ordered Stair cell geometry digest mismatch")

    @property
    def step_count(self) -> int:
        return len(self.riser_heights_m)

    @property
    def total_height_m(self) -> float:
        return round(sum(self.riser_heights_m), 10)

    @property
    def total_run_m(self) -> float:
        return round(sum(self.tread_depths_m), 10)

    @property
    def top_platform_length_m(self) -> float:
        return round(
            self.size_m[0] - self.start_platform_length_m - self.total_run_m,
            10,
        )

    @property
    def progress_goal_m(self) -> float:
        return round(
            self.start_platform_length_m / 2.0
            + self.total_run_m
            + self.top_platform_goal_offset_m,
            10,
        )

    @property
    def episode_horizon_control_steps(self) -> int:
        return int(
            math.floor(
                self.anchor_horizon_control_steps
                * self.progress_goal_m
                / self.anchor_progress_goal_m
                + 0.5
            )
        )

    @property
    def episode_horizon_s(self) -> float:
        return round(self.episode_horizon_control_steps * self.control_dt_s, 10)

    @property
    def cumulative_riser_heights_m(self) -> tuple[float, ...]:
        total = 0.0
        values = [0.0]
        for value in self.riser_heights_m:
            total = round(total + value, 10)
            values.append(total)
        return tuple(values)

    @property
    def cumulative_tread_boundaries_m(self) -> tuple[float, ...]:
        total = self.start_platform_length_m
        values = [round(total, 10)]
        for value in self.tread_depths_m:
            total = round(total + value, 10)
            values.append(total)
        return tuple(values)

    def geometry_payload(self) -> dict[str, Any]:
        return ordered_stair_geometry_payload(
            size_m=self.size_m,
            floor_depth_m=self.floor_depth_m,
            start_platform_length_m=self.start_platform_length_m,
            riser_heights_m=self.riser_heights_m,
            tread_depths_m=self.tread_depths_m,
            top_platform_goal_offset_m=self.top_platform_goal_offset_m,
            surface_profile=self.surface_profile,
        )

    def identity_payload(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "cohort": self.cohort,
            "factor_levels": dict(self.factor_levels),
            "geometry": self.geometry_payload(),
            "geometry_digest": self.geometry_digest,
            "episode_horizon_control_steps": self.episode_horizon_control_steps,
            "episode_horizon_s": self.episode_horizon_s,
        }


@dataclass(frozen=True)
class OrderedStairEvaluationCell:
    """One ordered course placed at a deterministic terrain column."""

    spec: OrderedStairCellSpec
    terrain_type_index: int
    expected_origin_m: tuple[float, float, float]

    @property
    def cell_id(self) -> str:
        return self.spec.cell_id

    @property
    def bin_id(self) -> str:
        return self.spec.cell_id

    def __getattr__(self, name: str) -> Any:
        return getattr(self.spec, name)


@dataclass(frozen=True)
class StairOrderedEvaluationSuite:
    """Common runtime view implemented by the active unified suite."""

    _payload: Mapping[str, Any] = field(repr=False)
    _cells: tuple[OrderedStairCellSpec, ...] = field(repr=False)
    evaluation_protocol: EvaluationProtocol

    @property
    def suite_id(self) -> str:
        return str(self._payload["suite_id"])

    @property
    def suite_hash(self) -> str:
        return str(self._payload["suite_hash"])

    @property
    def cell_set_hash(self) -> str:
        return str(self._payload["cell_set_hash"])

    @property
    def episodes_per_seed_per_cell(self) -> int:
        return int(self._payload["episodes_per_seed_per_cell"])

    @property
    def calibration_seed_count(self) -> int:
        return int(self._payload["calibration_seed_count"])

    def seed_count(self, cohort: str) -> int:
        if cohort == "calibration":
            return self.calibration_seed_count
        if cohort not in STAIR_ORDERED_FORMAL_COHORTS:
            raise ValueError("Stair ordered cohort has no evaluation seed count")
        return int(self._payload["seed_counts"][cohort])

    @property
    def cells(self) -> tuple[OrderedStairCellSpec, ...]:
        return self._cells

    @property
    def max_episode_horizon_control_steps(self) -> int:
        return max(cell.episode_horizon_control_steps for cell in self._cells)

    @property
    def max_episode_horizon_s(self) -> float:
        return max(cell.episode_horizon_s for cell in self._cells)

    def cells_for_cohort(self, cohort: str) -> tuple[OrderedStairCellSpec, ...]:
        if cohort not in STAIR_ORDERED_RUNNABLE_COHORTS:
            raise ValueError("unknown Stair ordered cohort")
        return tuple(cell for cell in self._cells if cell.cohort == cohort)

    def runtime_cells(self, cohort: str) -> tuple[OrderedStairEvaluationCell, ...]:
        cells = self.cells_for_cohort(cohort)
        if not cells:
            raise ValueError("Stair ordered cohort is empty")
        size_y = cells[0].size_m[1]
        return tuple(
            OrderedStairEvaluationCell(
                spec=cell,
                terrain_type_index=index,
                expected_origin_m=(
                    -cell.size_m[0] / 2.0 + cell.start_platform_length_m / 2.0,
                    (index - (len(cells) - 1) / 2.0) * size_y,
                    0.0,
                ),
            )
            for index, cell in enumerate(cells)
        )

    def private_cell_assignment(
        self, cohort: str
    ) -> tuple[OrderedStairEvaluationCell, ...]:
        return tuple(
            cell
            for cell in self.runtime_cells(cohort)
            for _ in range(self.episodes_per_seed_per_cell)
        )

    def cohort_hash(self, cohort: str) -> str:
        return str(self._payload["cohort_hashes"][cohort])

    def cell(self, cell_id: str) -> OrderedStairCellSpec:
        for cell in self._cells:
            if cell.cell_id == cell_id:
                return cell
        raise KeyError(f"unknown ordered Stair cell: {cell_id}")

    def to_dict(self) -> dict[str, Any]:
        return _thaw_json(self._payload)


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


__all__ = [
    "OrderedStairCellSpec",
    "OrderedStairEvaluationCell",
    "STAIR_ORDERED_CELL_SET_SCHEMA_VERSION",
    "STAIR_ORDERED_COHORT_SCHEMA_VERSION",
    "STAIR_ORDERED_FORMAL_COHORTS",
    "STAIR_ORDERED_GEOMETRY_SCHEMA_VERSION",
    "STAIR_ORDERED_RUNNABLE_COHORTS",
    "StairOrderedEvaluationSuite",
    "ordered_stair_geometry_payload",
]
