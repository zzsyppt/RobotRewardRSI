"""Runner-owned C0/C1 training distribution for the unified Stair cell."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.json_contract import load_strict_json_object

STAIR_UNIFIED_TRAINING_PROFILE_RELATIVE_PATH = (
    "configs/tasks/stair_unified_training_profile_v1.json"
)
STAIR_UNIFIED_TRAINING_PROFILE_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.unified_training_profile.v1"
)
STAIR_TERRAIN_REGIME_C0 = "c0_regular"
STAIR_TERRAIN_REGIME_C1 = "c1_irregular_flat"
STAIR_TERRAIN_REGIME_C2 = "c2_rough"

_PROFILE_FIELDS = {
    "schema_version",
    "profile_id",
    "task_id",
    "allocation",
    "grid",
    "terrain",
    "pattern_families",
    "curriculum",
    "training_exposure",
    "profile_hash",
}


@dataclass(frozen=True)
class StairUnifiedTrainingProfile:
    """Validated task-owned distribution; no robot or policy facts live here."""

    _payload: Mapping[str, Any] = field(repr=False)

    @classmethod
    def load(cls, path: str | Path) -> "StairUnifiedTrainingProfile":
        payload = load_strict_json_object(
            Path(path), context="Stair unified training profile"
        )
        return cls.from_mapping(payload)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "StairUnifiedTrainingProfile":
        copied = _json_copy(payload)
        _validate_profile(copied)
        expected_hash = canonical_json_hash(
            {key: value for key, value in copied.items() if key != "profile_hash"}
        )
        if copied["profile_hash"] != expected_hash:
            raise ValueError("Stair unified training profile hash mismatch")
        return cls(_payload=_freeze_json(copied))

    @property
    def profile_id(self) -> str:
        return str(self._payload["profile_id"])

    @property
    def profile_hash(self) -> str:
        return str(self._payload["profile_hash"])

    @property
    def allocation(self) -> Mapping[str, Any]:
        return _mapping(self._payload["allocation"], "allocation")

    @property
    def grid(self) -> Mapping[str, Any]:
        return _mapping(self._payload["grid"], "grid")

    @property
    def terrain(self) -> Mapping[str, Any]:
        return _mapping(self._payload["terrain"], "terrain")

    @property
    def curriculum(self) -> Mapping[str, Any]:
        return _mapping(self._payload["curriculum"], "curriculum")

    @property
    def pattern_ids(self) -> tuple[str, ...]:
        return tuple(self._payload["pattern_families"])

    @property
    def tread_bases_m(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self.terrain["c1"]["tread_bases_m"])

    def pattern(self, pattern_id: str) -> Mapping[str, Any]:
        patterns = _mapping(self._payload["pattern_families"], "pattern_families")
        try:
            return _mapping(patterns[pattern_id], f"pattern {pattern_id}")
        except KeyError as exc:
            raise KeyError(f"unknown Stair pattern family: {pattern_id}") from exc

    def c1_sequence(
        self,
        *,
        riser_base_m: float,
        tread_base_m: float,
        pattern_id: str,
        phase_steps: int,
    ) -> dict[str, Any]:
        c1 = _mapping(self.terrain["c1"], "terrain.c1")
        values = tuple(
            float(value) for value in self.pattern(pattern_id)["multipliers"]
        )
        rotated, tread_pattern = _phase_signature(values, phase_steps)
        return {
            "terrain_regime_id": STAIR_TERRAIN_REGIME_C1,
            "pattern_id": pattern_id,
            "phase_steps": int(phase_steps),
            "riser_heights_m": [
                round(
                    float(riser_base_m) + value * float(c1["riser_offset_m"]),
                    6,
                )
                for value in rotated
            ],
            "tread_depths_m": [
                round(
                    float(tread_base_m) + value * float(c1["tread_offset_m"]),
                    6,
                )
                for value in tread_pattern
            ],
        }

    def phase_changes_training_sequence(
        self,
        *,
        pattern_id: str,
        phase_steps: int,
    ) -> bool:
        pattern = self.pattern(pattern_id)
        values = tuple(float(value) for value in pattern["multipliers"])
        training_signature = _phase_signature(
            values,
            int(pattern["training_phase_steps"]),
        )
        return _phase_signature(values, phase_steps) != training_signature

    def is_held_out_phase(self, *, pattern_id: str, phase_steps: int) -> bool:
        pattern = self.pattern(pattern_id)
        return int(phase_steps) in {
            int(value) for value in pattern["held_out_phase_steps"]
        } and self.phase_changes_training_sequence(
            pattern_id=pattern_id,
            phase_steps=phase_steps,
        )

    def column_descriptors(self) -> tuple[dict[str, Any], ...]:
        columns: list[dict[str, Any]] = [
            {
                "column_index": 0,
                "terrain_regime_id": STAIR_TERRAIN_REGIME_C0,
                "column_id": "c0_regular",
            }
        ]
        for pattern_id in self.pattern_ids:
            phase_steps = int(self.pattern(pattern_id)["training_phase_steps"])
            for tread_base_m in self.tread_bases_m:
                columns.append(
                    {
                        "column_index": len(columns),
                        "terrain_regime_id": STAIR_TERRAIN_REGIME_C1,
                        "column_id": (f"c1_{pattern_id}_tread_{tread_base_m:.2f}"),
                        "pattern_id": pattern_id,
                        "phase_steps": phase_steps,
                        "tread_base_m": tread_base_m,
                    }
                )
        return tuple(columns)

    def task_projection(self) -> dict[str, Any]:
        return {
            key: _thaw_json(value)
            for key, value in self._payload.items()
            if key not in {"profile_id", "profile_hash"}
        }

    def to_dict(self) -> dict[str, Any]:
        return _thaw_json(self._payload)


def unified_initial_terrain_assignment(
    *,
    num_envs: int,
    seed: int,
    profile: StairUnifiedTrainingProfile,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return deterministic 50/50 terrain types and regime-local levels."""

    if not isinstance(profile, StairUnifiedTrainingProfile):
        raise TypeError("unified Stair assignment requires its training profile")
    if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0:
        raise ValueError("unified Stair num_envs must be positive")
    if num_envs % 2:
        raise ValueError("unified Stair 50/50 assignment requires an even env count")
    indices = list(range(num_envs))
    random.Random(int(seed)).shuffle(indices)
    c0_count = num_envs // 2
    terrain_types = [0] * num_envs
    terrain_levels = [0] * num_envs
    c0_max = int(profile.curriculum["initial_max_level"][STAIR_TERRAIN_REGIME_C0])
    c1_max = int(profile.curriculum["initial_max_level"][STAIR_TERRAIN_REGIME_C1])
    for rank, env_index in enumerate(indices[:c0_count]):
        terrain_types[env_index] = 0
        terrain_levels[env_index] = rank % (c0_max + 1)
    c1_column_count = len(profile.column_descriptors()) - 1
    for rank, env_index in enumerate(indices[c0_count:]):
        terrain_types[env_index] = 1 + rank % c1_column_count
        terrain_levels[env_index] = rank % (c1_max + 1)
    return tuple(terrain_types), tuple(terrain_levels)


def unified_assignment_summary(
    *,
    types: tuple[int, ...],
    levels: tuple[int, ...],
    profile: StairUnifiedTrainingProfile,
) -> dict[str, Any]:
    if len(types) != len(levels) or not types:
        raise ValueError("unified Stair assignment vectors are invalid")
    c0_indices = tuple(index for index, value in enumerate(types) if value == 0)
    c1_indices = tuple(index for index, value in enumerate(types) if value != 0)
    return {
        "schema_version": (
            "true_rl_agent.task.stair_climbing.unified_initial_assignment.v1"
        ),
        "training_profile_hash": profile.profile_hash,
        "num_envs": len(types),
        "regime_counts": {
            STAIR_TERRAIN_REGIME_C0: len(c0_indices),
            STAIR_TERRAIN_REGIME_C1: len(c1_indices),
        },
        "terrain_type_counts": {
            str(value): types.count(value) for value in sorted(set(types))
        },
        "maximum_initial_level": {
            STAIR_TERRAIN_REGIME_C0: max(levels[index] for index in c0_indices),
            STAIR_TERRAIN_REGIME_C1: max(levels[index] for index in c1_indices),
        },
    }


def goal_progress_m_by_terrain_type(
    profile: StairUnifiedTrainingProfile,
) -> tuple[float, ...]:
    terrain = profile.terrain
    start_half = float(terrain["start_platform_length_m"]) / 2.0
    goal_offset = float(terrain["top_platform_goal_offset_m"])
    c0_run = int(terrain["step_count"]) * float(terrain["c0"]["tread_depth_m"])
    values = [round(start_half + c0_run + goal_offset, 6)]
    c1_riser_mid = sum(terrain["c1"]["riser_base_range_m"]) / 2.0
    for column in profile.column_descriptors()[1:]:
        sequence = profile.c1_sequence(
            riser_base_m=c1_riser_mid,
            tread_base_m=float(column["tread_base_m"]),
            pattern_id=str(column["pattern_id"]),
            phase_steps=int(column["phase_steps"]),
        )
        values.append(
            round(start_half + sum(sequence["tread_depths_m"]) + goal_offset, 6)
        )
    return tuple(values)


def _validate_profile(payload: Mapping[str, Any]) -> None:
    if set(payload) != _PROFILE_FIELDS:
        raise ValueError("Stair unified training profile fields mismatch")
    if payload.get("schema_version") != STAIR_UNIFIED_TRAINING_PROFILE_SCHEMA_VERSION:
        raise ValueError("Stair unified training profile schema mismatch")
    profile_id = payload.get("profile_id")
    if (
        not isinstance(profile_id, str)
        or not profile_id.startswith("stair_unified_")
        or not profile_id.endswith("_v1")
    ):
        raise ValueError("Stair unified training profile_id is invalid")
    if payload.get("task_id") != "unitree_stair_climbing":
        raise ValueError("Stair unified training task_id mismatch")

    allocation = _mapping(payload.get("allocation"), "allocation")
    if set(allocation) != {"c0_regular_fraction", "c1_irregular_flat_fraction"}:
        raise ValueError("Stair unified allocation fields mismatch")
    c0_fraction = _finite(allocation["c0_regular_fraction"], "C0 allocation")
    c1_fraction = _finite(allocation["c1_irregular_flat_fraction"], "C1 allocation")
    if not math.isclose(c0_fraction, 0.5) or not math.isclose(c1_fraction, 0.5):
        raise ValueError("Stair unified allocation must be 50/50")

    grid = _mapping(payload.get("grid"), "grid")
    if set(grid) != {
        "terrain_rows",
        "terrain_cols",
        "c0_column_count",
        "c1_column_count",
        "difficulty_range",
    }:
        raise ValueError("Stair unified grid fields mismatch")
    if (
        grid["terrain_rows"] != 10
        or grid["terrain_cols"] != 10
        or grid["c0_column_count"] != 1
        or grid["c1_column_count"] != 9
        or grid["difficulty_range"] != [0.0, 1.0]
    ):
        raise ValueError("Stair unified grid must be the frozen 10x10 layout")

    terrain = _mapping(payload.get("terrain"), "terrain")
    if set(terrain) != {
        "size_m",
        "floor_depth_m",
        "start_platform_length_m",
        "step_count",
        "top_platform_goal_offset_m",
        "c0",
        "c1",
    }:
        raise ValueError("Stair unified terrain fields mismatch")
    if terrain["size_m"] != [8.0, 4.0] or terrain["step_count"] != 8:
        raise ValueError("Stair unified terrain dimensions drifted")
    for name in (
        "floor_depth_m",
        "start_platform_length_m",
        "top_platform_goal_offset_m",
    ):
        if _finite(terrain[name], name) <= 0.0:
            raise ValueError(f"Stair unified {name} must be positive")
    c0 = _mapping(terrain["c0"], "terrain.c0")
    c1 = _mapping(terrain["c1"], "terrain.c1")
    if set(c0) != {"riser_range_m", "tread_depth_m"}:
        raise ValueError("Stair C0 training support fields mismatch")
    c0_range = c0["riser_range_m"]
    if (
        not isinstance(c0_range, list)
        or len(c0_range) != 2
        or any(_finite(value, "C0 riser range") <= 0.0 for value in c0_range)
        or float(c0_range[0]) >= float(c0_range[1])
        or _finite(c0["tread_depth_m"], "C0 tread depth") <= 0.0
    ):
        raise ValueError("Stair C0 training support is invalid")
    if set(c1) != {
        "riser_base_range_m",
        "riser_offset_m",
        "tread_bases_m",
        "tread_offset_m",
    }:
        raise ValueError("Stair C1 training support fields mismatch")
    c1_range = c1["riser_base_range_m"]
    tread_bases = c1["tread_bases_m"]
    riser_offset = _finite(c1["riser_offset_m"], "C1 riser offset")
    tread_offset = _finite(c1["tread_offset_m"], "C1 tread offset")
    if (
        not isinstance(c1_range, list)
        or len(c1_range) != 2
        or any(_finite(value, "C1 riser range") <= 0.0 for value in c1_range)
        or float(c1_range[0]) >= float(c1_range[1])
        or not isinstance(tread_bases, list)
        or len(tread_bases) != 3
        or len({float(value) for value in tread_bases}) != 3
        or any(_finite(value, "C1 tread base") <= 0.0 for value in tread_bases)
        or riser_offset <= 0.0
        or riser_offset >= float(c1_range[0])
        or tread_offset <= 0.0
        or tread_offset >= min(float(value) for value in tread_bases)
        or not any(
            math.isclose(
                float(c0["tread_depth_m"]),
                float(value),
                rel_tol=0.0,
                abs_tol=1.0e-9,
            )
            for value in tread_bases
        )
    ):
        raise ValueError("Stair C1 training support is invalid")

    patterns = _mapping(payload.get("pattern_families"), "pattern_families")
    if tuple(patterns) != ("alternating", "front_loaded", "back_loaded"):
        raise ValueError("Stair pattern family order drifted")
    for pattern_id, raw in patterns.items():
        pattern = _mapping(raw, f"pattern {pattern_id}")
        if set(pattern) != {
            "multipliers",
            "training_phase_steps",
            "held_out_phase_steps",
        }:
            raise ValueError("Stair pattern family fields mismatch")
        multipliers = pattern["multipliers"]
        if (
            not isinstance(multipliers, list)
            or len(multipliers) != 8
            or len(set(float(value) for value in multipliers)) < 2
            or any(
                abs(_finite(value, "pattern multiplier")) > 1.0 for value in multipliers
            )
        ):
            raise ValueError("Stair pattern family multipliers are invalid")
        training_phase = pattern["training_phase_steps"]
        held_out_phases = pattern["held_out_phase_steps"]
        if (
            isinstance(training_phase, bool)
            or not isinstance(training_phase, int)
            or not 0 <= training_phase < len(multipliers)
            or not isinstance(held_out_phases, list)
            or not held_out_phases
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value < len(multipliers)
                for value in held_out_phases
            )
            or len(held_out_phases) != len(set(held_out_phases))
        ):
            raise ValueError("Stair pattern phase split is invalid")
        training_signature = _phase_signature(
            tuple(float(value) for value in multipliers),
            training_phase,
        )
        held_out_signatures = {
            _phase_signature(tuple(float(value) for value in multipliers), phase)
            for phase in held_out_phases
        }
        if training_signature in held_out_signatures or len(held_out_signatures) != len(
            held_out_phases
        ):
            raise ValueError(
                "Stair held-out phase must produce a distinct geometry sequence"
            )

    curriculum = _mapping(payload.get("curriculum"), "curriculum")
    if curriculum != {
        "progress_source": "goal_progress_fraction_from_actual_geometry",
        "move_up_threshold": 0.8,
        "move_down_threshold": 0.4,
        "terrain_type_is_fixed": True,
        "initial_max_level": {"c0_regular": 5, "c1_irregular_flat": 1},
    }:
        raise ValueError("Stair unified curriculum drifted")
    exposure = _mapping(payload.get("training_exposure"), "training_exposure")
    if exposure != {
        "allowed_regimes": ["c0_regular", "c1_irregular_flat"],
        "forbidden_regimes": ["c2_rough"],
        "evaluation_only_cohorts": ["sealed_validation"],
    }:
        raise ValueError("Stair unified training exposure drifted")


def _rotate(values: tuple[float, ...], count: int) -> tuple[float, ...]:
    offset = int(count) % len(values)
    return values[offset:] + values[:offset]


def _phase_signature(
    values: tuple[float, ...],
    phase_steps: int,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    return (
        _rotate(values, phase_steps),
        _rotate(tuple(reversed(values)), phase_steps + 1),
    )


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Stair unified {field_name} must be an object")
    return value


def _finite(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"Stair unified {field_name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Stair unified {field_name} must be finite")
    return result


def _json_copy(value: Mapping[str, Any]) -> dict[str, Any]:
    import json

    return json.loads(json.dumps(dict(value), allow_nan=False))


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


__all__ = [
    "STAIR_TERRAIN_REGIME_C0",
    "STAIR_TERRAIN_REGIME_C1",
    "STAIR_TERRAIN_REGIME_C2",
    "STAIR_UNIFIED_TRAINING_PROFILE_RELATIVE_PATH",
    "STAIR_UNIFIED_TRAINING_PROFILE_SCHEMA_VERSION",
    "StairUnifiedTrainingProfile",
    "goal_progress_m_by_terrain_type",
    "unified_assignment_summary",
    "unified_initial_terrain_assignment",
]
