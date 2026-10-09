"""Frozen task-owned geometry and scoring profile for Stair evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.evaluation.protocol import EvaluationProtocol
from true_rl_agent.json_contract import load_strict_json_object
from true_rl_agent.tasks.stair_climbing.phase0_contract import StairProtocolV1

STAIR_EVALUATION_PROFILE_RELATIVE_PATH = (
    "configs/tasks/stair_evaluation_profile_v5.json"
)
STAIR_EVALUATION_PROFILE_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.evaluation_profile.v5"
)
STAIR_EPISODE_HORIZON_POLICY = (
    "scale_base_horizon_by_progress_goal_ratio_to_anchor_nearest_control_step"
)


@dataclass(frozen=True)
class StairEvaluationCell:
    """One fixed height/tread course in the evaluation terrain grid."""

    cell_id: str
    bin_id: str
    height_m: float
    tread_depth_m: float
    terrain_type_index: int
    expected_origin_m: tuple[float, float, float]
    progress_goal_m: float
    top_platform_length_m: float
    episode_horizon_control_steps: int
    episode_horizon_s: float


@dataclass(frozen=True)
class StairEvaluationProfile:
    """Validated immutable extension of the historical fixed-tread protocol."""

    _payload: Mapping[str, Any] = field(repr=False)
    _cells: tuple[StairEvaluationCell, ...] = field(repr=False)
    evaluation_protocol: EvaluationProtocol

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        base_protocol: StairProtocolV1,
    ) -> "StairEvaluationProfile":
        payload = load_strict_json_object(
            Path(path),
            context="Stair evaluation profile",
        )
        return cls.from_mapping(payload, base_protocol=base_protocol)

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        base_protocol: StairProtocolV1,
    ) -> "StairEvaluationProfile":
        if not isinstance(base_protocol, StairProtocolV1):
            raise TypeError("Stair evaluation profile requires StairProtocolV1")
        copied = _copy_json(payload)
        _validate_profile(copied, base_protocol=base_protocol)
        base_eval = base_protocol.evaluation_protocol
        evaluation_protocol = EvaluationProtocol(
            eval_protocol_version=str(copied["evaluation_protocol_version"]),
            public_seed_count=base_eval.public_seed_count,
            public_accept_seed_count=base_eval.public_accept_seed_count,
            sealed_validation_seed_count=base_eval.sealed_validation_seed_count,
            eval_num_envs=int(copied["eval_num_envs"]),
        )
        cells = _build_cells(copied, base_protocol=base_protocol)
        return cls(
            _payload=_freeze_json(copied),
            _cells=cells,
            evaluation_protocol=evaluation_protocol,
        )

    @property
    def profile_hash(self) -> str:
        return str(self._payload["profile_hash"])

    @property
    def anchor_tread_depth_m(self) -> float:
        return float(self._payload["anchor_tread_depth_m"])

    @property
    def tread_depths_m(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self._payload["tread_depths_m"])

    @property
    def episodes_per_seed_per_cell(self) -> int:
        return int(self._payload["episodes_per_seed_per_cell"])

    @property
    def episode_horizon_policy(self) -> str:
        return str(self._payload["episode_horizon_policy"])

    @property
    def cells(self) -> tuple[StairEvaluationCell, ...]:
        return self._cells

    @property
    def anchor_cells(self) -> tuple[StairEvaluationCell, ...]:
        return tuple(
            cell
            for cell in self._cells
            if math.isclose(
                cell.tread_depth_m,
                self.anchor_tread_depth_m,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        )

    @property
    def max_episode_horizon_control_steps(self) -> int:
        return max(cell.episode_horizon_control_steps for cell in self._cells)

    @property
    def max_episode_horizon_s(self) -> float:
        return max(cell.episode_horizon_s for cell in self._cells)

    @property
    def episode_horizons_by_tread_m(self) -> dict[str, float]:
        return {
            f"{cell.tread_depth_m:.2f}": cell.episode_horizon_s
            for cell in self._cells[:: len(self.anchor_cells)]
        }

    def cell(self, cell_id: str) -> StairEvaluationCell:
        for cell in self._cells:
            if cell.cell_id == cell_id:
                return cell
        raise KeyError(f"unknown Stair evaluation cell: {cell_id}")

    def private_cell_assignment(self) -> tuple[StairEvaluationCell, ...]:
        return tuple(
            cell for cell in self._cells for _ in range(self.episodes_per_seed_per_cell)
        )

    def to_dict(self) -> dict[str, Any]:
        return _thaw_json(self._payload)


def _validate_profile(
    payload: Mapping[str, Any],
    *,
    base_protocol: StairProtocolV1,
) -> None:
    expected_fields = {
        "schema_version",
        "profile_id",
        "base_protocol_hash",
        "evaluation_protocol_version",
        "anchor_tread_depth_m",
        "tread_depths_m",
        "episodes_per_seed_per_cell",
        "eval_num_envs",
        "top_platform_goal_offset_m",
        "episode_horizon_policy",
        "cell_aggregation",
        "terrain_layout",
        "profile_hash",
    }
    if set(payload) != expected_fields:
        raise ValueError("Stair evaluation profile fields mismatch")
    if payload.get("schema_version") != STAIR_EVALUATION_PROFILE_SCHEMA_VERSION:
        raise ValueError("Stair evaluation profile schema mismatch")
    for key in ("profile_id", "evaluation_protocol_version"):
        if not isinstance(payload.get(key), str) or not payload[key]:
            raise ValueError(f"Stair evaluation profile {key} must be non-empty")
    if payload.get("base_protocol_hash") != base_protocol.protocol_hash:
        raise ValueError("Stair evaluation profile base protocol mismatch")
    if payload.get("cell_aggregation") != "equal_weight_mean_over_height_tread_cells":
        raise ValueError("Stair evaluation profile aggregation mismatch")
    if payload.get("terrain_layout") != "anchor_first_then_ascending_tread":
        raise ValueError("Stair evaluation profile terrain layout mismatch")
    if payload.get("episode_horizon_policy") != STAIR_EPISODE_HORIZON_POLICY:
        raise ValueError("Stair evaluation profile episode horizon policy mismatch")
    terrain = base_protocol.to_dict()["terrain"]
    anchor = _positive_number(
        payload.get("anchor_tread_depth_m"),
        "anchor_tread_depth_m",
    )
    if not math.isclose(
        anchor,
        float(terrain["tread_depth_m"]),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("Stair evaluation anchor tread differs from base protocol")
    raw_treads = payload.get("tread_depths_m")
    if not isinstance(raw_treads, list) or not raw_treads:
        raise ValueError("Stair evaluation tread depths must be a non-empty list")
    treads = tuple(_positive_number(value, "tread_depths_m") for value in raw_treads)
    if tuple(sorted(treads)) != treads or len(treads) != len(set(treads)):
        raise ValueError("Stair evaluation tread depths must be sorted and unique")
    if anchor not in treads:
        raise ValueError("Stair evaluation tread depths must include the anchor")

    episodes = _positive_int(
        payload.get("episodes_per_seed_per_cell"),
        "episodes_per_seed_per_cell",
    )
    base_episodes = int(
        base_protocol.to_dict()["evaluation"]["env_to_bin_mapping"]["envs_per_bin"]
    )
    if episodes != base_episodes:
        raise ValueError("Stair evaluation cell sample count differs from the anchor")
    expected_envs = len(base_protocol.eval_bins) * len(treads) * episodes
    if _positive_int(payload.get("eval_num_envs"), "eval_num_envs") != expected_envs:
        raise ValueError("Stair evaluation env count does not cover every cell")

    top_platform_goal_offset = _positive_number(
        payload.get("top_platform_goal_offset_m"),
        "top_platform_goal_offset_m",
    )
    size_x = float(terrain["size_m"][0])
    start = float(terrain["start_platform_length_m"])
    step_count = int(terrain["step_count"])
    _base_horizon_steps(base_protocol)
    for tread in treads:
        top_length = size_x - start - step_count * tread
        if top_length <= top_platform_goal_offset:
            raise ValueError(
                "Stair evaluation tread leaves no valid top-platform goal region"
            )

    declared_hash = payload.get("profile_hash")
    if not isinstance(declared_hash, str) or declared_hash != canonical_json_hash(
        {key: value for key, value in payload.items() if key != "profile_hash"}
    ):
        raise ValueError("Stair evaluation profile hash mismatch")


def _build_cells(
    payload: Mapping[str, Any],
    *,
    base_protocol: StairProtocolV1,
) -> tuple[StairEvaluationCell, ...]:
    terrain = base_protocol.to_dict()["terrain"]
    anchor = float(payload["anchor_tread_depth_m"])
    treads = tuple(float(value) for value in payload["tread_depths_m"])
    ordered_treads = (anchor,) + tuple(value for value in treads if value != anchor)
    size_x, size_y = (float(value) for value in terrain["size_m"])
    start = float(terrain["start_platform_length_m"])
    step_count = int(terrain["step_count"])
    top_platform_goal_offset = float(payload["top_platform_goal_offset_m"])
    control_dt_s = float(
        base_protocol.to_dict()["policy_compatibility"]["control_dt_s"]
    )
    base_horizon_steps = _base_horizon_steps(base_protocol)
    anchor_goal_m = start / 2.0 + step_count * anchor + top_platform_goal_offset
    cell_count = len(base_protocol.eval_bins) * len(ordered_treads)
    cells: list[StairEvaluationCell] = []
    for tread in ordered_treads:
        for raw_bin in base_protocol.eval_bins:
            index = len(cells)
            bin_id = str(raw_bin["bin_id"])
            height = float(raw_bin["height_m"])
            tread_mm = int(round(tread * 1000.0))
            progress_goal_m = (
                start / 2.0 + step_count * tread + top_platform_goal_offset
            )
            horizon_steps = math.floor(
                base_horizon_steps * progress_goal_m / anchor_goal_m + 0.5
            )
            cells.append(
                StairEvaluationCell(
                    cell_id=f"{bin_id}__tread_{tread_mm:03d}",
                    bin_id=bin_id,
                    height_m=height,
                    tread_depth_m=tread,
                    terrain_type_index=index,
                    expected_origin_m=(
                        -size_x / 2.0 + start / 2.0,
                        (index - (cell_count - 1) / 2.0) * size_y,
                        0.0,
                    ),
                    progress_goal_m=progress_goal_m,
                    top_platform_length_m=size_x - start - step_count * tread,
                    episode_horizon_control_steps=horizon_steps,
                    episode_horizon_s=round(horizon_steps * control_dt_s, 10),
                )
            )
    if len({cell.cell_id for cell in cells}) != len(cells):
        raise ValueError("Stair evaluation cell identity collision")
    return tuple(cells)


def _base_horizon_steps(base_protocol: StairProtocolV1) -> int:
    payload = base_protocol.to_dict()
    horizon_s = float(payload["evaluation"]["episode_length_s"])
    control_dt_s = float(payload["policy_compatibility"]["control_dt_s"])
    raw_steps = horizon_s / control_dt_s
    steps = int(round(raw_steps))
    if steps <= 0 or not math.isclose(raw_steps, steps, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError("Stair base evaluation horizon is not an integral control-step count")
    return steps


def _positive_number(value: Any, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"Stair evaluation {field_name} must be positive")
    return float(value)


def _nonnegative_number(value: Any, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"Stair evaluation {field_name} must be non-negative")
    return float(value)


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"Stair evaluation {field_name} must be a positive integer")
    return value


def _copy_json(value: Mapping[str, Any]) -> dict[str, Any]:
    return _thaw_json(_freeze_json(dict(value)))


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list | tuple):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_thaw_json(item) for item in value]
    return value


__all__ = [
    "STAIR_EVALUATION_PROFILE_RELATIVE_PATH",
    "STAIR_EVALUATION_PROFILE_SCHEMA_VERSION",
    "StairEvaluationCell",
    "StairEvaluationProfile",
]
