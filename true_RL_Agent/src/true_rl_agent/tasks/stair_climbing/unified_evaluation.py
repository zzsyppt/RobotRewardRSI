"""Stratified ID/OOD evaluation suite for the unified Stair task."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.evaluation.protocol import EvaluationProtocol
from true_rl_agent.json_contract import load_strict_json_object
from true_rl_agent.tasks.stair_climbing.ordered_evaluation import (
    STAIR_ORDERED_CELL_SET_SCHEMA_VERSION,
    STAIR_ORDERED_COHORT_SCHEMA_VERSION,
    OrderedStairCellSpec,
    StairOrderedEvaluationSuite,
    ordered_stair_geometry_payload,
)
from true_rl_agent.tasks.stair_climbing.rough_surface import (
    STAIR_C2_SURFACE_KIND,
    STAIR_FLAT_SURFACE_KIND,
    normalize_stair_surface_profile,
)
from true_rl_agent.tasks.stair_climbing.unified_training import (
    STAIR_TERRAIN_REGIME_C0,
    STAIR_TERRAIN_REGIME_C1,
    STAIR_TERRAIN_REGIME_C2,
    StairUnifiedTrainingProfile,
)

STAIR_UNIFIED_EVALUATION_SUITE_RELATIVE_PATH = (
    "configs/tasks/stair_unified_evaluation_suite_v1.json"
)
STAIR_UNIFIED_EVALUATION_SUITE_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.unified_evaluation_suite.v1"
)
STAIR_UNIFIED_FORMAL_COHORTS = (
    "public_dev",
    "public_accept",
    "sealed_validation",
)
STAIR_UNIFIED_RUNNABLE_COHORTS = ("calibration", *STAIR_UNIFIED_FORMAL_COHORTS)

_SUITE_FIELDS = {
    "schema_version",
    "suite_id",
    "evaluation_protocol_version",
    "training_profile_hash",
    "seed_counts",
    "calibration_seed_count",
    "episodes_per_seed_per_cell",
    "formal_cohorts",
    "geometry_policy",
    "score_contract",
    "qualification_rule",
    "cell_templates",
    "cohort_geometry_overrides",
    "calibration_templates",
    "cell_set_hash",
    "cohort_hashes",
    "suite_hash",
}
_STRATA = (
    "c0_id",
    "c1_id",
    "height_ood",
    "tread_ood",
    "joint_ood",
    "phase_composition_ood",
    "c1_conditional_frontier",
    "c2_mechanism_ood",
)


@dataclass(frozen=True)
class UnifiedStairCellSpec(OrderedStairCellSpec):
    terrain_regime_id: str = STAIR_TERRAIN_REGIME_C1
    stratum_id: str = "c1_id"
    claim_kind: str = "in_distribution"
    template_id: str = ""

    def __post_init__(self) -> None:
        if self.terrain_regime_id not in {
            STAIR_TERRAIN_REGIME_C0,
            STAIR_TERRAIN_REGIME_C1,
            STAIR_TERRAIN_REGIME_C2,
        }:
            raise ValueError("unified Stair terrain regime is invalid")
        if self.stratum_id not in _STRATA:
            raise ValueError("unified Stair stratum is invalid")
        if not self.claim_kind or not self.template_id:
            raise ValueError("unified Stair claim/template identity is missing")
        super().__post_init__()

    def identity_payload(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "template_id": self.template_id,
            "cohort": self.cohort,
            "terrain_regime_id": self.terrain_regime_id,
            "stratum_id": self.stratum_id,
            "claim_kind": self.claim_kind,
            "factor_levels": dict(self.factor_levels),
            "geometry": self.geometry_payload(),
            "geometry_digest": self.geometry_digest,
            "episode_horizon_control_steps": self.episode_horizon_control_steps,
            "episode_horizon_s": self.episode_horizon_s,
        }


@dataclass(frozen=True)
class StairUnifiedEvaluationSuite(StairOrderedEvaluationSuite):
    """Compact declarations expanded to equal-cost split-local cell sets."""

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        training_profile: StairUnifiedTrainingProfile,
    ) -> "StairUnifiedEvaluationSuite":
        return cls.from_mapping(
            load_strict_json_object(
                Path(path), context="Stair unified evaluation suite"
            ),
            training_profile=training_profile,
        )

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        training_profile: StairUnifiedTrainingProfile,
    ) -> "StairUnifiedEvaluationSuite":
        copied = _json_copy(payload)
        _validate_suite(copied, training_profile=training_profile)
        cells = _expand_cells(copied, training_profile=training_profile)
        _validate_formal_geometry_isolation(cells)
        declarations = compute_unified_suite_declarations(copied, cells=cells)
        for name, expected in declarations.items():
            if copied.get(name) != expected:
                raise ValueError(f"Stair unified suite {name} mismatch")
        formal_count = len(copied["cell_templates"])
        episodes = int(copied["episodes_per_seed_per_cell"])
        protocol = EvaluationProtocol(
            eval_protocol_version=str(copied["evaluation_protocol_version"]),
            public_seed_count=int(copied["seed_counts"]["public_dev"]),
            public_accept_seed_count=int(copied["seed_counts"]["public_accept"]),
            sealed_validation_seed_count=int(
                copied["seed_counts"]["sealed_validation"]
            ),
            eval_num_envs=formal_count * episodes,
        )
        return cls(
            _payload=_freeze_json(copied),
            _cells=cells,
            evaluation_protocol=protocol,
        )

    @property
    def score_contract(self) -> Mapping[str, Any]:
        return MappingProxyType(dict(self._payload["score_contract"]))

    @property
    def qualification_rule(self) -> Mapping[str, Any]:
        return MappingProxyType(dict(self._payload["qualification_rule"]))

    def task_projection(self) -> dict[str, Any]:
        return {
            key: _thaw_json(value)
            for key, value in self._payload.items()
            if key
            not in {
                "suite_id",
                "training_profile_hash",
                "cell_set_hash",
                "cohort_hashes",
                "suite_hash",
            }
        }


def compute_unified_suite_declarations(
    payload: Mapping[str, Any],
    *,
    cells: tuple[UnifiedStairCellSpec, ...],
) -> dict[str, Any]:
    cell_set_hash = canonical_json_hash(
        {
            "schema_version": STAIR_ORDERED_CELL_SET_SCHEMA_VERSION,
            "cells": [cell.identity_payload() for cell in cells],
        }
    )
    cohort_hashes = {
        cohort: canonical_json_hash(
            {
                "schema_version": STAIR_ORDERED_COHORT_SCHEMA_VERSION,
                "suite_id": payload["suite_id"],
                "cohort": cohort,
                "episodes_per_seed_per_cell": payload["episodes_per_seed_per_cell"],
                "seed_count": (
                    payload["calibration_seed_count"]
                    if cohort == "calibration"
                    else payload["seed_counts"][cohort]
                ),
                "cells": [
                    {
                        "cell_id": cell.cell_id,
                        "geometry_digest": cell.geometry_digest,
                        "stratum_id": cell.stratum_id,
                    }
                    for cell in cells
                    if cell.cohort == cohort
                ],
            }
        )
        for cohort in STAIR_UNIFIED_RUNNABLE_COHORTS
    }
    unhashed = {
        key: value
        for key, value in payload.items()
        if key not in {"cell_set_hash", "cohort_hashes", "suite_hash"}
    }
    suite_hash = canonical_json_hash(
        {**unhashed, "cell_set_hash": cell_set_hash, "cohort_hashes": cohort_hashes}
    )
    return {
        "cell_set_hash": cell_set_hash,
        "cohort_hashes": cohort_hashes,
        "suite_hash": suite_hash,
    }


def _validate_suite(
    payload: Mapping[str, Any],
    *,
    training_profile: StairUnifiedTrainingProfile,
) -> None:
    if set(payload) != _SUITE_FIELDS:
        raise ValueError("Stair unified evaluation suite fields mismatch")
    if payload.get("schema_version") != STAIR_UNIFIED_EVALUATION_SUITE_SCHEMA_VERSION:
        raise ValueError("Stair unified evaluation suite schema mismatch")
    suite_id = payload.get("suite_id")
    if (
        not isinstance(suite_id, str)
        or not suite_id.startswith("stair_unified_")
        or not suite_id.endswith("_v1")
    ):
        raise ValueError("Stair unified evaluation suite_id is invalid")
    if payload.get("training_profile_hash") != training_profile.profile_hash:
        raise ValueError("Stair unified suite/training profile mismatch")
    if payload.get("formal_cohorts") != list(STAIR_UNIFIED_FORMAL_COHORTS):
        raise ValueError("Stair unified formal cohorts drifted")
    seed_counts = payload.get("seed_counts")
    if (
        not isinstance(seed_counts, Mapping)
        or set(seed_counts) != set(STAIR_UNIFIED_FORMAL_COHORTS)
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in seed_counts.values()
        )
    ):
        raise ValueError("Stair unified seed counts are invalid")
    if payload.get("calibration_seed_count") != 3:
        raise ValueError("Stair unified calibration seed count drifted")
    if payload.get("episodes_per_seed_per_cell") != 3:
        raise ValueError("Stair unified episodes per cell drifted")
    geometry_policy = payload.get("geometry_policy")
    if (
        not isinstance(geometry_policy, Mapping)
        or set(geometry_policy) != {"control_dt_s", "anchor_horizon_control_steps"}
        or _positive(geometry_policy.get("control_dt_s"), "control dt") <= 0.0
        or isinstance(geometry_policy.get("anchor_horizon_control_steps"), bool)
        or not isinstance(geometry_policy.get("anchor_horizon_control_steps"), int)
        or int(geometry_policy["anchor_horizon_control_steps"]) <= 0
    ):
        raise ValueError("Stair unified evaluation geometry policy drifted")
    qualification = payload.get("qualification_rule")
    if not isinstance(qualification, Mapping) or qualification != {
        "minimum_cells_per_stratum_per_cohort": 3,
        "minimum_reference_score": 0.1,
        "family_mask_mode": "exclude_reference_floor_only",
        "requires_at_least_one_qualified_ood_family": True,
    }:
        raise ValueError("Stair unified qualification rule drifted")
    minimum_cells = int(qualification["minimum_cells_per_stratum_per_cohort"])
    templates = payload.get("cell_templates")
    if (
        not isinstance(templates, list)
        or len(templates) != len(_STRATA) * minimum_cells
    ):
        raise ValueError("Stair unified suite template count mismatches its strata")
    ids = [item.get("template_id") for item in templates if isinstance(item, Mapping)]
    if len(ids) != len(templates) or len(ids) != len(set(ids)):
        raise ValueError("Stair unified template ids are invalid")
    counts = {
        stratum: sum(item.get("stratum_id") == stratum for item in templates)
        for stratum in _STRATA
    }
    if any(count != minimum_cells for count in counts.values()):
        raise ValueError("Stair unified strata do not meet the declared template count")
    for template in templates:
        _validate_template_claim(template, training_profile=training_profile)
    overrides = payload.get("cohort_geometry_overrides")
    if not isinstance(overrides, Mapping) or set(overrides) != {
        "public_accept",
        "sealed_validation",
    }:
        raise ValueError("Stair unified cohort geometry overrides mismatch")
    templates_by_id = {str(item["template_id"]): item for item in templates}
    for cohort, raw_overrides in overrides.items():
        if not isinstance(raw_overrides, Mapping) or set(raw_overrides) != set(ids):
            raise ValueError(
                f"Stair unified {cohort} geometry overrides are incomplete"
            )
        for template_id, geometry in raw_overrides.items():
            _validate_template_claim(
                templates_by_id[str(template_id)],
                geometry=geometry,
                training_profile=training_profile,
            )
    calibration = payload.get("calibration_templates")
    if (
        not isinstance(calibration, list)
        or len(calibration) != 5
        or len(calibration) != len(set(calibration))
        or not set(calibration) <= set(ids)
    ):
        raise ValueError("Stair unified calibration templates are invalid")
    score = payload.get("score_contract")
    if not isinstance(score, Mapping) or score.get("named_guards") != [
        "j_ood",
        "c0_id",
        "c1_id",
    ]:
        raise ValueError("Stair unified score contract drifted")
    if not isinstance(payload.get("cohort_hashes"), Mapping) or set(
        payload["cohort_hashes"]
    ) != set(STAIR_UNIFIED_RUNNABLE_COHORTS):
        raise ValueError("Stair unified cohort hash fields mismatch")


def _validate_template_claim(
    template: Mapping[str, Any],
    *,
    training_profile: StairUnifiedTrainingProfile,
    geometry: Any | None = None,
) -> None:
    """Bind every ID/OOD label to the declared training support."""

    required = {
        "template_id",
        "stratum_id",
        "terrain_regime_id",
        "claim_kind",
        "geometry",
    }
    if set(template) != required or not isinstance(template.get("geometry"), Mapping):
        raise ValueError("Stair unified template fields mismatch")
    stratum = str(template["stratum_id"])
    regime = str(template["terrain_regime_id"])
    claim = str(template["claim_kind"])
    geometry = template["geometry"] if geometry is None else geometry
    if not isinstance(geometry, Mapping):
        raise ValueError("Stair unified cohort geometry must be an object")
    expected_claims = {
        "c0_id": (STAIR_TERRAIN_REGIME_C0, "in_distribution", "regular"),
        "c1_id": (STAIR_TERRAIN_REGIME_C1, "in_distribution", "c1"),
        "height_ood": (
            STAIR_TERRAIN_REGIME_C0,
            "scalar_extrapolation",
            "regular",
        ),
        "tread_ood": (
            STAIR_TERRAIN_REGIME_C0,
            "scalar_extrapolation",
            "regular",
        ),
        "joint_ood": (
            STAIR_TERRAIN_REGIME_C0,
            "joint_scalar_extrapolation",
            "regular",
        ),
        "phase_composition_ood": (
            STAIR_TERRAIN_REGIME_C1,
            "known_family_held_out_phase",
            "c1",
        ),
        "c1_conditional_frontier": (
            STAIR_TERRAIN_REGIME_C1,
            "conditional_on_c1_support",
            "c1",
        ),
        "c2_mechanism_ood": (
            STAIR_TERRAIN_REGIME_C2,
            "surface_mechanism_shift",
            "c2",
        ),
    }
    if (regime, claim, geometry.get("kind")) != expected_claims[stratum]:
        raise ValueError("Stair unified template claim identity mismatch")

    terrain = training_profile.terrain
    c0 = terrain["c0"]
    c1 = terrain["c1"]
    c0_low, c0_high = (float(value) for value in c0["riser_range_m"])
    c0_tread = float(c0["tread_depth_m"])
    if geometry.get("kind") == "regular":
        riser = _positive(geometry.get("riser_height_m"), "riser height")
        tread = _positive(geometry.get("tread_depth_m"), "tread depth")
        height_in = c0_low <= riser <= c0_high
        tread_in = math.isclose(tread, c0_tread, rel_tol=0.0, abs_tol=1.0e-9)
        expected_support = {
            "c0_id": (True, True),
            "height_ood": (False, True),
            "tread_ood": (True, False),
            "joint_ood": (False, False),
        }[stratum]
        if (height_in, tread_in) != expected_support:
            raise ValueError(
                "Stair unified scalar ID/OOD claim mismatches training support"
            )
        return

    riser_base = _positive(geometry.get("riser_base_m"), "riser base")
    tread_base = _positive(geometry.get("tread_base_m"), "tread base")
    pattern_id = str(geometry.get("pattern_id"))
    phase_steps = geometry.get("phase_steps")
    if isinstance(phase_steps, bool) or not isinstance(phase_steps, int):
        raise ValueError("Stair unified phase_steps must be an integer")
    if pattern_id not in training_profile.pattern_ids:
        raise ValueError("Stair unified C1 pattern is outside the declared family")
    pattern = training_profile.pattern(pattern_id)
    riser_low, riser_high = (float(value) for value in c1["riser_base_range_m"])
    riser_in = riser_low <= riser_base <= riser_high
    tread_in = any(
        math.isclose(tread_base, value, rel_tol=0.0, abs_tol=1.0e-9)
        for value in training_profile.tread_bases_m
    )
    training_phase = int(pattern["training_phase_steps"])
    if stratum == "c1_id" and not (
        riser_in and tread_in and phase_steps == training_phase
    ):
        raise ValueError("Stair unified C1-ID claim mismatches training support")
    if stratum == "phase_composition_ood" and not (
        riser_in
        and tread_in
        and training_profile.is_held_out_phase(
            pattern_id=pattern_id,
            phase_steps=phase_steps,
        )
    ):
        raise ValueError(
            "Stair unified held-out-phase claim mismatches training support"
        )
    if stratum == "c1_conditional_frontier" and not (
        phase_steps == training_phase and not (riser_in and tread_in)
    ):
        raise ValueError(
            "Stair unified conditional-frontier claim is not outside support"
        )
    if stratum == "c2_mechanism_ood":
        if not (riser_in and tread_in and phase_steps == training_phase):
            raise ValueError(
                "Stair unified C2 claim must isolate the unseen surface mechanism"
            )
        surface = normalize_stair_surface_profile(geometry.get("surface_profile"))
        if surface["kind"] != STAIR_C2_SURFACE_KIND:
            raise ValueError("Stair unified C2 claim requires a rough surface")


def _expand_cells(
    payload: Mapping[str, Any],
    *,
    training_profile: StairUnifiedTrainingProfile,
) -> tuple[UnifiedStairCellSpec, ...]:
    geometry_policy = payload["geometry_policy"]
    terrain = training_profile.terrain
    template_by_id = {
        str(item["template_id"]): item for item in payload["cell_templates"]
    }
    overrides = payload["cohort_geometry_overrides"]
    rows: list[tuple[str, Mapping[str, Any], Mapping[str, Any]]] = []
    for cohort in STAIR_UNIFIED_FORMAL_COHORTS:
        rows.extend(
            (
                cohort,
                item,
                (
                    item["geometry"]
                    if cohort == "public_dev"
                    else overrides[cohort][str(item["template_id"])]
                ),
            )
            for item in payload["cell_templates"]
        )
    rows.extend(
        (
            "calibration",
            template_by_id[template_id],
            template_by_id[template_id]["geometry"],
        )
        for template_id in payload["calibration_templates"]
    )
    cells: list[UnifiedStairCellSpec] = []
    anchor_progress_goal_m = round(
        float(terrain["start_platform_length_m"]) / 2.0
        + int(terrain["step_count"]) * float(terrain["c0"]["tread_depth_m"])
        + float(terrain["top_platform_goal_offset_m"]),
        10,
    )
    for cohort, template, geometry in rows:
        template_id = str(template["template_id"])
        cell_id = f"{cohort}.{template_id}"
        risers, treads, factors, surface_profile = _resolve_geometry(
            geometry,
            training_profile=training_profile,
            step_count=int(terrain["step_count"]),
        )
        common = {
            "cell_id": cell_id,
            "cohort": cohort,
            "factor_levels": MappingProxyType(factors),
            "riser_heights_m": risers,
            "tread_depths_m": treads,
            "size_m": tuple(float(value) for value in terrain["size_m"]),
            "floor_depth_m": float(terrain["floor_depth_m"]),
            "start_platform_length_m": float(terrain["start_platform_length_m"]),
            "top_platform_goal_offset_m": float(terrain["top_platform_goal_offset_m"]),
            "control_dt_s": float(geometry_policy["control_dt_s"]),
            "anchor_progress_goal_m": anchor_progress_goal_m,
            "anchor_horizon_control_steps": int(
                geometry_policy["anchor_horizon_control_steps"]
            ),
            "surface_profile": MappingProxyType(surface_profile),
            "terrain_regime_id": str(template["terrain_regime_id"]),
            "stratum_id": str(template["stratum_id"]),
            "claim_kind": str(template["claim_kind"]),
            "template_id": template_id,
        }
        cells.append(
            UnifiedStairCellSpec(
                **common,
                geometry_digest=canonical_json_hash(
                    ordered_stair_geometry_payload(
                        size_m=common["size_m"],
                        floor_depth_m=common["floor_depth_m"],
                        start_platform_length_m=common["start_platform_length_m"],
                        riser_heights_m=risers,
                        tread_depths_m=treads,
                        top_platform_goal_offset_m=common["top_platform_goal_offset_m"],
                        surface_profile=surface_profile,
                    )
                ),
            )
        )
    return tuple(cells)


def _validate_formal_geometry_isolation(
    cells: tuple[UnifiedStairCellSpec, ...],
) -> None:
    digests = {
        cohort: {cell.geometry_digest for cell in cells if cell.cohort == cohort}
        for cohort in STAIR_UNIFIED_FORMAL_COHORTS
    }
    for cohort, cohort_digests in digests.items():
        cell_count = sum(cell.cohort == cohort for cell in cells)
        if len(cohort_digests) != cell_count:
            raise ValueError(f"Stair unified {cohort} contains duplicate geometries")
    for index, left in enumerate(STAIR_UNIFIED_FORMAL_COHORTS):
        for right in STAIR_UNIFIED_FORMAL_COHORTS[index + 1 :]:
            if digests[left] & digests[right]:
                raise ValueError(
                    f"Stair unified formal cohort geometries overlap: {left}/{right}"
                )


def _resolve_geometry(
    geometry: Mapping[str, Any],
    *,
    training_profile: StairUnifiedTrainingProfile,
    step_count: int,
) -> tuple[
    tuple[float, ...],
    tuple[float, ...],
    dict[str, str],
    dict[str, Any],
]:
    kind = geometry.get("kind")
    if kind == "regular":
        if set(geometry) != {"kind", "riser_height_m", "tread_depth_m"}:
            raise ValueError("Stair regular template geometry fields mismatch")
        riser = _positive(geometry["riser_height_m"], "riser height")
        tread = _positive(geometry["tread_depth_m"], "tread depth")
        return (
            (riser,) * step_count,
            (tread,) * step_count,
            {
                "phase": "p0",
                "riser_band": f"{riser:.3f}",
                "tread_band": f"{tread:.3f}",
                "sequence_pattern": "regular",
                "roughness_band": "0.000",
                "surface_pattern": STAIR_FLAT_SURFACE_KIND,
            },
            {"kind": STAIR_FLAT_SURFACE_KIND},
        )
    c1_fields = {
        "kind",
        "riser_base_m",
        "tread_base_m",
        "pattern_id",
        "phase_steps",
    }
    c2_fields = {*c1_fields, "surface_profile"}
    if kind == "c1" and set(geometry) == c1_fields:
        surface_profile = {"kind": STAIR_FLAT_SURFACE_KIND}
    elif kind == "c2" and set(geometry) == c2_fields:
        surface_profile = normalize_stair_surface_profile(
            geometry["surface_profile"]
        )
        if surface_profile["kind"] != STAIR_C2_SURFACE_KIND:
            raise ValueError("Stair C2 template requires a rough surface")
    else:
        raise ValueError("Stair C1/C2 template geometry fields mismatch")
    pattern_id = str(geometry["pattern_id"])
    phase_steps = int(geometry["phase_steps"])
    sequence = training_profile.c1_sequence(
        riser_base_m=_positive(geometry["riser_base_m"], "riser base"),
        tread_base_m=_positive(geometry["tread_base_m"], "tread base"),
        pattern_id=pattern_id,
        phase_steps=phase_steps,
    )
    return (
        tuple(sequence["riser_heights_m"]),
        tuple(sequence["tread_depths_m"]),
        {
            "phase": f"p{phase_steps}",
            "riser_band": f"{float(geometry['riser_base_m']):.3f}",
            "tread_band": f"{float(geometry['tread_base_m']):.3f}",
            "sequence_pattern": pattern_id,
            "roughness_band": f"{float(surface_profile.get('amplitude_m', 0.0)):.3f}",
            "surface_pattern": (
                STAIR_FLAT_SURFACE_KIND
                if surface_profile["kind"] == STAIR_FLAT_SURFACE_KIND
                else (
                    f"l{surface_profile['longitudinal_cycles']}"
                    f"_y{surface_profile['lateral_cycles']}"
                )
            ),
        },
        surface_profile,
    )


def _positive(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"Stair unified {field_name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"Stair unified {field_name} must be positive")
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
    "STAIR_UNIFIED_EVALUATION_SUITE_RELATIVE_PATH",
    "STAIR_UNIFIED_EVALUATION_SUITE_SCHEMA_VERSION",
    "STAIR_UNIFIED_FORMAL_COHORTS",
    "STAIR_UNIFIED_RUNNABLE_COHORTS",
    "StairUnifiedEvaluationSuite",
    "UnifiedStairCellSpec",
    "compute_unified_suite_declarations",
]
