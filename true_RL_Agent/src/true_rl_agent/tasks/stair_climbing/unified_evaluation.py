"""Stratified ID/OOD evaluation suite for the unified Stair task."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from true_rl_agent.evaluation.binding import canonical_json_hash, require_sha256_digest
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
STAIR_CALIBRATED_EVALUATION_SUITE_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.unified_evaluation_suite.v2"
)
STAIR_CELL_SCORE_FORMULA = "0.5*success_rate+0.5*mean_goal_progress_fraction"
# Frozen summaries used this spelling; preserve their bytes on historical replay.
LEGACY_STAIR_CELL_SCORE_FORMULA = "0.5*success_rate+0.5*goal_progress_fraction"
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
STAIR_STRATUM_CLAIMS = {
    "c0_id": (STAIR_TERRAIN_REGIME_C0, "in_distribution", "regular"),
    "c1_id": (STAIR_TERRAIN_REGIME_C1, "in_distribution", "c1"),
    "height_ood": (STAIR_TERRAIN_REGIME_C0, "scalar_extrapolation", "regular"),
    "tread_ood": (STAIR_TERRAIN_REGIME_C0, "scalar_extrapolation", "regular"),
    "joint_ood": (STAIR_TERRAIN_REGIME_C0, "joint_scalar_extrapolation", "regular"),
    "phase_composition_ood": (STAIR_TERRAIN_REGIME_C1, "known_family_held_out_phase", "c1"),
    "c1_conditional_frontier": (STAIR_TERRAIN_REGIME_C1, "conditional_on_c1_support", "c1"),
    "c2_mechanism_ood": (STAIR_TERRAIN_REGIME_C2, "surface_mechanism_shift", "c2"),
}
_STRATA = tuple(STAIR_STRATUM_CLAIMS)

STAIR_CALIBRATED_SCORE_CONTRACT = {
    "episode_to_cell": "equal_weight_mean_over_episodes",
    "cell_score": STAIR_CELL_SCORE_FORMULA,
    "cell_to_stratum": "equal_weight_mean_over_cells",
    "stratum_to_seed": "preserve_paired_seed_vector",
    "j_id": "0.5*S_C0-ID+0.5*S_C1-ID",
    "j_ood": "equal_weight_mean_over_qualified_nonempty_families",
    "j_primary": "J_OOD",
    "families": {"F_scalar": ["height_ood", "tread_ood", "joint_ood"],
                 "F_c1": ["c1_conditional_frontier"],
                 "F_c2": ["c2_mechanism_ood"]},
    "named_guards": ["j_ood", "c0_id", "c1_id", "joint_ood", "phase_control"],
}


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
        cells = resolve_stair_suite_cells(copied, training_profile=training_profile)
        _validate_formal_geometry_isolation(cells)
        if "final_holdout_policy" in copied:
            for cohort in STAIR_UNIFIED_FORMAL_COHORTS:
                cohort_cells = [cell for cell in cells if cell.cohort == cohort]
                flat_bases = {(cell.riser_heights_m, cell.tread_depths_m) for cell in cohort_cells
                              if cell.surface_profile["kind"] == STAIR_FLAT_SURFACE_KIND}
                if any((cell.riser_heights_m, cell.tread_depths_m) not in flat_bases
                       for cell in cohort_cells if cell.stratum_id == "c2_mechanism_ood"):
                    raise ValueError("Stair formal C2 is missing its matching flat control")
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

    @classmethod
    def freeze(
        cls, payload: Mapping[str, Any], *, training_profile: StairUnifiedTrainingProfile
    ) -> "StairUnifiedEvaluationSuite":
        """Freeze task-authored declarations through the same loader as execution."""
        copied = _json_copy(payload)
        copied.update(compute_unified_suite_declarations(
            copied, cells=resolve_stair_suite_cells(copied, training_profile=training_profile)
        ))
        return cls.from_mapping(copied, training_profile=training_profile)

    @property
    def score_contract(self) -> Mapping[str, Any]:
        return MappingProxyType(dict(self._payload["score_contract"]))

    @property
    def is_calibrated(self) -> bool:
        """The calibrated suite schema owns scale, scoring and qualification."""
        return self._payload["schema_version"] == STAIR_CALIBRATED_EVALUATION_SUITE_SCHEMA_VERSION

    @property
    def records_episode_facts(self) -> bool:
        return self.is_calibrated

    @property
    def qualification_rule(self) -> Mapping[str, Any]:
        return MappingProxyType(dict(self._payload["qualification_rule"]))

    @property
    def source_suite_hash(self) -> str:
        """The public suite that qualified the reference for this final instance."""
        instance = self._payload.get("study_instance")
        return str(instance["source_suite_hash"]) if instance is not None else self.suite_hash

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
                **({"study_instance": payload["study_instance"]}
                   if cohort == "sealed_validation" and "study_instance" in payload else {}),
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
    calibrated = payload.get("schema_version") == STAIR_CALIBRATED_EVALUATION_SUITE_SCHEMA_VERSION
    expected_fields = _SUITE_FIELDS | {"scale_context"} if calibrated else _SUITE_FIELDS
    if calibrated and "final_holdout_policy" in payload:
        from true_rl_agent.tasks.stair_climbing.study_holdout import FINAL_HOLDOUT_POLICY, LEGACY_FINAL_HOLDOUT_POLICY

        expected_fields |= {"final_holdout_policy"}
        if payload["final_holdout_policy"] not in (FINAL_HOLDOUT_POLICY, LEGACY_FINAL_HOLDOUT_POLICY):
            raise ValueError("Stair study holdout generation rule drifted")
        if "study_instance" in payload:
            expected_fields |= {"study_instance"}
            instance = payload["study_instance"]
            if not isinstance(instance, Mapping) or set(instance) != {"source_suite_hash", "instance_key_hash"}:
                raise ValueError("Stair study instance identity fields mismatch")
            for key, value in instance.items():
                require_sha256_digest(value, "Stair study " + key)
    if set(payload) != expected_fields:
        raise ValueError("Stair unified evaluation suite fields mismatch")
    if payload.get("schema_version") not in {
        STAIR_UNIFIED_EVALUATION_SUITE_SCHEMA_VERSION,
        STAIR_CALIBRATED_EVALUATION_SUITE_SCHEMA_VERSION,
    }:
        raise ValueError("Stair unified evaluation suite schema mismatch")
    suite_id = payload.get("suite_id")
    if (
        not isinstance(suite_id, str)
        or not suite_id.startswith("stair_unified_")
        or not suite_id.endswith("_v2" if calibrated else "_v1")
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
    episodes = payload.get("episodes_per_seed_per_cell")
    if (isinstance(episodes, bool) or not isinstance(episodes, int)
            or (episodes < 12 if calibrated else episodes != 3)):
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
    expected_qualification = {
        "minimum_cells_per_stratum_per_cohort": 3,
        "minimum_reference_score": 0.1,
        "family_mask_mode": "exclude_reference_floor_only",
        "requires_at_least_one_qualified_ood_family": True,
    }
    if calibrated:
        expected_qualification.update(
            family_mask_mode="require_all_objective_strata_with_success_transition",
            minimum_learnable_success_rate=0.2,
            maximum_learnable_success_rate=0.8,
        )
        scale = payload["scale_context"]
        if not isinstance(scale, Mapping) or set(scale) != {
            "robot_profile_hash", "length_source", "support_length_m", "gravity_mps2"
        } or scale["length_source"] != "nominal_root_to_support_body_center_z_m":
            raise ValueError("Stair calibrated scale context mismatch")
        _positive(scale["support_length_m"], "support length")
        _positive(scale["gravity_mps2"], "gravity")
        require_sha256_digest(scale["robot_profile_hash"], "Stair scale robot profile hash")
    if not isinstance(qualification, Mapping) or qualification != expected_qualification:
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
    if calibrated:
        if not isinstance(calibration, list) or len(calibration) != 6 * len(_STRATA):
            raise ValueError("Stair calibrated pool must contain six levels per stratum")
        calibration_ids = [item["template_id"] for item in calibration]
        if len(calibration_ids) != len(set(calibration_ids)):
            raise ValueError("Stair calibration template IDs are duplicated")
        for item in calibration:
            _validate_template_claim(item, training_profile=training_profile)
        if any(sum(item["stratum_id"] == name for item in calibration) != 6 for name in _STRATA):
            raise ValueError("Stair calibration stratum coverage mismatch")
    elif (
        not isinstance(calibration, list)
        or len(calibration) != 5
        or len(calibration) != len(set(calibration))
        or not set(calibration) <= set(ids)
    ):
        raise ValueError("Stair unified calibration templates are invalid")
    score = payload.get("score_contract")
    expected_guards = [
        "j_ood",
        "c0_id",
        "c1_id",
    ] + (["joint_ood", "phase_control"] if calibrated else [])
    if not isinstance(score, Mapping) or score.get("named_guards") != expected_guards:
        raise ValueError("Stair unified score contract drifted")
    if calibrated and score != STAIR_CALIBRATED_SCORE_CONTRACT:
        raise ValueError("Stair calibrated score contract differs from its arithmetic")
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
    if (regime, claim, geometry.get("kind")) != STAIR_STRATUM_CLAIMS[stratum]:
        raise ValueError("Stair unified template claim identity mismatch")

    if stratum == "c2_mechanism_ood" and "base_geometry" in geometry:
        if set(geometry) != {"kind", "base_geometry", "surface_profile"}:
            raise ValueError("Stair C2 composed geometry fields mismatch")
        base = geometry["base_geometry"]
        if not isinstance(base, Mapping) or base.get("kind") not in {"regular", "c1"}:
            raise ValueError("Stair C2 base must be a known training regime")
        regular = base["kind"] == "regular"
        _validate_template_claim(
            {"template_id": template["template_id"],
             "stratum_id": "c0_id" if regular else "c1_id",
             "terrain_regime_id": STAIR_TERRAIN_REGIME_C0 if regular else STAIR_TERRAIN_REGIME_C1,
             "claim_kind": "in_distribution", "geometry": base},
            training_profile=training_profile,
        )
        if normalize_stair_surface_profile(geometry["surface_profile"])["kind"] != STAIR_C2_SURFACE_KIND:
            raise ValueError("Stair C2 requires a rough surface")
        return

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
    offsets_in = all(
        math.isclose(_positive(geometry.get(key, c1[key]), key), float(c1[key]), abs_tol=1e-9)
        for key in ("riser_offset_m", "tread_offset_m")
    )
    if stratum == "c1_id" and not (
        riser_in and tread_in and offsets_in and phase_steps == training_phase
    ):
        raise ValueError("Stair unified C1-ID claim mismatches training support")
    if stratum == "phase_composition_ood" and not (
        riser_in
        and tread_in
        and offsets_in
        and training_profile.is_held_out_phase(
            pattern_id=pattern_id,
            phase_steps=phase_steps,
        )
    ):
        raise ValueError(
            "Stair unified held-out-phase claim mismatches training support"
        )
    if stratum == "c1_conditional_frontier" and not (
        phase_steps == training_phase and not (riser_in and tread_in and offsets_in)
    ):
        raise ValueError(
            "Stair unified conditional-frontier claim is not outside support"
        )
    if stratum == "c2_mechanism_ood":
        if not (riser_in and tread_in and offsets_in and phase_steps == training_phase):
            raise ValueError(
                "Stair unified C2 claim must isolate the unseen surface mechanism"
            )
        surface = normalize_stair_surface_profile(geometry.get("surface_profile"))
        if surface["kind"] != STAIR_C2_SURFACE_KIND:
            raise ValueError("Stair unified C2 claim requires a rough surface")


def resolve_stair_suite_cells(
    payload: Mapping[str, Any],
    *,
    training_profile: StairUnifiedTrainingProfile,
) -> tuple[UnifiedStairCellSpec, ...]:
    """Resolve declared geometry identically for freezing and holdout exclusion."""
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
    calibration = payload["calibration_templates"]
    calibration_items = (
        calibration if payload["schema_version"] == STAIR_CALIBRATED_EVALUATION_SUITE_SCHEMA_VERSION
        else [template_by_id[template_id] for template_id in calibration]
    )
    rows.extend(
        (
            "calibration",
            template,
            template["geometry"],
        )
        for template in calibration_items
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


def _surface_control_geometry_digests(cells) -> set[str]:
    """Exact ID controls paired to a rough surface within the same cohort."""
    bases = {
        (cell.cohort, cell.riser_heights_m, cell.tread_depths_m)
        for cell in cells if cell.stratum_id == "c2_mechanism_ood"
    }
    return {
        cell.geometry_digest for cell in cells
        if cell.stratum_id in {"c0_id", "c1_id"}
        and cell.surface_profile["kind"] == STAIR_FLAT_SURFACE_KIND
        and (cell.cohort, cell.riser_heights_m, cell.tread_depths_m) in bases
    }


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
            shared_controls = _surface_control_geometry_digests(
                tuple(cell for cell in cells if cell.cohort == left)
            ) & _surface_control_geometry_digests(
                tuple(cell for cell in cells if cell.cohort == right)
            )
            if (digests[left] & digests[right]) - shared_controls:
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
    if kind == "c2" and "base_geometry" in geometry:
        risers, treads, factors, _ = _resolve_geometry(
            geometry["base_geometry"], training_profile=training_profile, step_count=step_count,
        )
        surface = normalize_stair_surface_profile(geometry["surface_profile"])
        factors.update(roughness_band=f"{surface['amplitude_m']:.3f}",
                       surface_pattern=f"l{surface['longitudinal_cycles']}_y{surface['lateral_cycles']}")
        return risers, treads, factors, surface
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
    offset_fields = {"riser_offset_m", "tread_offset_m"}
    if offset_fields & set(geometry):
        c1_fields |= offset_fields
        c2_fields |= offset_fields
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
        riser_offset_m=geometry.get("riser_offset_m"),
        tread_offset_m=geometry.get("tread_offset_m"),
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


def stair_geometry_variant(
    geometry: Mapping[str, Any], *, stratum_id: str,
    training_profile: StairUnifiedTrainingProfile, support_length_m: float,
    height_fraction: float, tread_fraction: float, surface_phase_shift: int = 0,
    preserve_surface_ratio: bool = True,
) -> dict[str, Any]:
    """Real geometric variation within the declared training-support relation.

    ID dimensions stay in training support; extrapolated dimensions reflect at
    its boundary. C2 changes the same flat base as its paired ID control and
    preserves amplitude / minimum realized riser, including C1 offsets.
    The false setting is only for replaying the frozen v1 final generator.
    """
    result = _json_copy(geometry)
    if "base_geometry" in result:
        base = result["base_geometry"]
        result["base_geometry"] = stair_geometry_variant(
            base, stratum_id="c0_id" if base["kind"] == "regular" else "c1_id",
            training_profile=training_profile, support_length_m=support_length_m,
            height_fraction=height_fraction, tread_fraction=0.0,
        )
        surface = result["surface_profile"]
        if preserve_surface_ratio:
            old_risers = _resolve_geometry(base, training_profile=training_profile,
                step_count=int(training_profile.terrain["step_count"]))[0]
            new_risers = _resolve_geometry(result["base_geometry"], training_profile=training_profile,
                step_count=int(training_profile.terrain["step_count"]))[0]
            surface["amplitude_m"] *= min(new_risers) / min(old_risers)
        surface["phase_index"] = (surface["phase_index"] + surface_phase_shift) % surface["phase_count"]
        return result
    regular = result["kind"] == "regular"
    terrain = training_profile.terrain["c0" if regular else "c1"]
    low, high = map(float, terrain["riser_range_m" if regular else "riser_base_range_m"])
    key = "riser_height_m" if regular else "riser_base_m"
    value = float(result[key]) + height_fraction * support_length_m
    extrapolated = stratum_id in {"height_ood", "joint_ood", "c1_conditional_frontier"}
    if extrapolated:
        value = high + max(0.001 * (high - low), abs(value - high))
    else:
        width = high - low
        folded = (value - low) % (2 * width)
        value = low + (folded if folded <= width else 2 * width - folded)
    result[key] = value
    if stratum_id in {"tread_ood", "joint_ood"}:
        d0 = float(terrain["tread_depth_m"])
        gap = abs(d0 - result["tread_depth_m"] - tread_fraction * support_length_m)
        result["tread_depth_m"] = d0 - min(0.95 * d0, max(0.001 * d0, gap))
    elif stratum_id == "c1_conditional_frontier":
        result["tread_base_m"] += tread_fraction * support_length_m
    return result


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
