"""Bounded Stair calibration from robot facts, without robot-name branches."""

from __future__ import annotations

import copy
import math
from typing import Any, Mapping

from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.robots.base import RobotProfile
from true_rl_agent.tasks.stair_climbing.unified_evaluation import (
    STAIR_CALIBRATED_EVALUATION_SUITE_SCHEMA_VERSION,
    STAIR_CALIBRATED_SCORE_CONTRACT,
    STAIR_STRATUM_CLAIMS,
    STAIR_UNIFIED_FORMAL_COHORTS,
    StairUnifiedEvaluationSuite,
    _surface_control_geometry_digests,
    stair_geometry_variant,
)
from true_rl_agent.tasks.stair_climbing.unified_training import StairUnifiedTrainingProfile
from true_rl_agent.tasks.stair_climbing.unified_scoring import learnable_cell_ids


def stair_scale_context(profile: RobotProfile) -> dict[str, Any]:
    source = "nominal_root_to_support_body_center_z_m"
    length = float(profile.runtime_facts["morphology"][source])
    if not math.isfinite(length) or length <= 0:
        raise ValueError("Stair calibration requires a positive nominal support height")
    return {"robot_profile_hash": profile.profile_hash, "length_source": source,
            "support_length_m": length, "gravity_mps2": 9.81}


def stair_prototype_templates() -> tuple[dict[str, str], ...]:
    """The eight task claims needed by a new robot; no legacy geometry scaffold."""
    return tuple({"stratum_id": name, "terrain_regime_id": regime, "claim_kind": claim}
                 for name, (regime, claim, _) in STAIR_STRATUM_CLAIMS.items())


def build_stair_calibration_suite(
    *, training_profile: StairUnifiedTrainingProfile, robot_profile: RobotProfile,
    geometry_policy: Mapping[str, Any], episodes_per_seed_per_cell: int = 12,
) -> StairUnifiedEvaluationSuite:
    """Generate a robot's public pool directly from its frozen training support."""
    scale = stair_scale_context(robot_profile)
    length = scale["support_length_m"]
    payload = {
        "schema_version": STAIR_CALIBRATED_EVALUATION_SUITE_SCHEMA_VERSION,
        "suite_id": "stair_unified_" + robot_profile.robot_profile_id + "_calibration_v2",
        "evaluation_protocol_version": "stair_calibrated_terrain_eval_v2",
        "training_profile_hash": training_profile.profile_hash,
        "scale_context": scale, "geometry_policy": dict(geometry_policy),
        "seed_counts": dict(zip(STAIR_UNIFIED_FORMAL_COHORTS, (3, 5, 7))),
        "formal_cohorts": list(STAIR_UNIFIED_FORMAL_COHORTS),
        "calibration_seed_count": 3, "episodes_per_seed_per_cell": episodes_per_seed_per_cell,
        "score_contract": copy.deepcopy(STAIR_CALIBRATED_SCORE_CONTRACT),
        "qualification_rule": {
            "minimum_cells_per_stratum_per_cohort": 3, "minimum_reference_score": 0.1,
            "family_mask_mode": "require_all_objective_strata_with_success_transition",
            "requires_at_least_one_qualified_ood_family": True,
            "minimum_learnable_success_rate": 0.2, "maximum_learnable_success_rate": 0.8,
        },
    }
    terrain = training_profile.terrain
    c0_lo, c0_hi = terrain["c0"]["riser_range_m"]
    c1_lo, c1_hi = terrain["c1"]["riser_base_range_m"]
    tread = float(terrain["c0"]["tread_depth_m"])
    pool = []
    for prototype in stair_prototype_templates():
        stratum = prototype["stratum_id"]
        for level, fraction in enumerate((0.15, 0.31, 0.47, 0.63, 0.79, 0.95)):
            height_delta = length * (0.005, 0.01, 0.02, 0.035, 0.05, 0.07)[level]
            # Nominal support height only sets the initial search scale; it is
            # not foot size or reachable step length. The anchor locates the
            # transition separately for each morphology. The training-relative
            # cap keeps even short existing treads positive.
            tread_delta = min(0.5 * length, 0.75 * tread) * (0.05, 0.20, 0.40, 0.60, 0.80, 1.0)[level]
            template = copy.deepcopy(prototype)
            template["template_id"] = f"{stratum}_level_{level}"
            if prototype["terrain_regime_id"] == "c0_regular":
                geometry = {"kind": "regular", "riser_height_m": c0_lo + fraction * (c0_hi - c0_lo),
                            "tread_depth_m": tread}
                if stratum in {"height_ood", "joint_ood"}:
                    geometry["riser_height_m"] = c0_hi + height_delta
                if stratum in {"tread_ood", "joint_ood"}:
                    geometry["tread_depth_m"] = tread - tread_delta
                if stratum == "tread_ood":
                    geometry["riser_height_m"] = c0_lo + 0.8 * (c0_hi - c0_lo)
            else:
                pattern_id = training_profile.pattern_ids[level % len(training_profile.pattern_ids)]
                pattern = training_profile.pattern(pattern_id)
                geometry = {
                    "kind": "c1", "riser_base_m": c1_lo + fraction * (c1_hi - c1_lo),
                    "tread_base_m": training_profile.tread_bases_m[level % len(training_profile.tread_bases_m)],
                    "pattern_id": pattern_id, "phase_steps": int(pattern["training_phase_steps"]),
                }
                if stratum == "phase_composition_ood":
                    geometry["phase_steps"] = int(pattern["held_out_phase_steps"][0])
                    geometry["riser_base_m"] = c1_lo + (0.75 + 0.04 * level) * (c1_hi - c1_lo)
                if stratum == "c1_conditional_frontier":
                    geometry["riser_base_m"] = c1_hi + length * (0.01, 0.06, 0.12, 0.18, 0.24, 0.30)[level]
                    geometry["riser_offset_m"] = float(terrain["c1"]["riser_offset_m"]) + length * (0, 0.01, 0.02, 0.04, 0.06, 0.08)[level]
                    geometry["tread_offset_m"] = float(terrain["c1"]["tread_offset_m"]) + length * (0, 0.01, 0.02, 0.03, 0.04, 0.05)[level]
                if stratum == "c2_mechanism_ood":
                    # Provisional C0 control is measured in the same coarse pass;
                    # surface-base calibration may replace it before refinement.
                    base = {"kind": "regular", "riser_height_m": c0_lo + 0.95 * (c0_hi - c0_lo),
                            "tread_depth_m": tread}
                    cycles_x = max(1, round(tread / (0.35 * length)))
                    cycles_y = max(1, round(float(terrain["size_m"][1]) / (0.70 * length)))
                    geometry = {"kind": "c2", "base_geometry": base, "surface_profile": {
                        "kind": "bounded_piecewise_linear_v1",
                        "amplitude_m": base["riser_height_m"] * (0.04, 0.10, 0.18, 0.26, 0.34, 0.42)[level],
                        "longitudinal_cycles": cycles_x, "lateral_cycles": cycles_y,
                        "phase_index": 1, "phase_count": 8, "step_phase_stride": 1,
                        "longitudinal_vertices": 4 * cycles_x + 1, "lateral_vertices": 4 * cycles_y + 1,
                    }}
            template["geometry"] = geometry
            pool.append(template)
    payload["calibration_templates"] = pool
    chosen = [item for item in pool if item["template_id"].endswith(("_0", "_2", "_5"))]
    _set_formal_instances(payload, chosen, training_profile=training_profile)
    return StairUnifiedEvaluationSuite.freeze(payload, training_profile=training_profile)


def select_stair_calibration_levels(
    *, suite: StairUnifiedEvaluationSuite, summary: Mapping[str, Any]
) -> dict[str, Any]:
    """Select transition cells for objectives and coverage cells for controls."""
    if summary["split"] != "calibration" or summary["ordered_suite_hash"] != suite.suite_hash:
        raise ValueError("Stair calibration selection requires its frozen public pool")
    selected, learnable, measurements = {}, {}, {}
    success = summary["ordered_summary"]["aggregate_success_rate_by_cell"]
    for stratum, cell_ids in summary["stratum_cell_ids"].items():
        qualified = learnable_cell_ids(suite=suite, summary=summary, cell_ids=cell_ids)
        learnable[stratum] = qualified
        rank = lambda cell_id: (abs(success[cell_id] - 0.5), cell_ids.index(cell_id))
        if stratum in {"c1_id", "phase_composition_ood", "c1_conditional_frontier"}:
            by_pattern = {}
            for cell_id in cell_ids:
                by_pattern.setdefault(suite.cell(cell_id).factor_levels["sequence_pattern"], []).append(cell_id)
            if stratum == "c1_conditional_frontier":
                anchors = [min(group, key=rank) for group in by_pattern.values()]
            else:
                anchors = [max((c for c in group if success[c] >= 0.9),
                               key=lambda c: sum(suite.cell(c).riser_heights_m), default=group[0])
                           for group in by_pattern.values()]
        elif stratum == "c0_id":
            eligible = [c for c in cell_ids if success[c] >= 0.9]
            anchors = (eligible if len(eligible) >= 3 else cell_ids)[-3:]
        else:
            anchors = sorted(cell_ids, key=rank)[:3]
        anchors = sorted(anchors, key=cell_ids.index)
        selected[stratum] = [suite.cell(cell_id).template_id for cell_id in anchors]
        measurements[stratum] = {
            "success_rate_by_cell": {cell_id: success[cell_id] for cell_id in cell_ids},
            "learnable_cell_ids": qualified,
            "selected_template_ids": selected[stratum],
        }
    # C2 may choose an ID level outside the usual three controls. Include that
    # exact template, retaining C1 pattern coverage if the base is irregular.
    templates = {item["template_id"]: item for item in suite.to_dict()["calibration_templates"]}
    base = templates[selected["c2_mechanism_ood"][0]]["geometry"]["base_geometry"]
    controls = [item for item in templates.values() if item["stratum_id"] in {"c0_id", "c1_id"}
                and item["geometry"] == base]
    if len(controls) != 1:
        raise ValueError("Stair C2 needs one measured matching flat control")
    control = controls[0]
    chosen = selected[control["stratum_id"]]
    if control["template_id"] not in chosen:
        victim = next((i for i, name in enumerate(chosen)
                       if templates[name]["geometry"].get("pattern_id") == base.get("pattern_id")), 0)
        chosen[victim] = control["template_id"]
        measurements[control["stratum_id"]]["selected_template_ids"] = chosen
    requirements = dict(suite.score_contract["families"])
    statuses = {name: all(learnable[stratum] for stratum in strata) for name, strata in requirements.items()}
    return {"status": "GO" if all(statuses.values()) else "NO_GO",
            "ordered_suite_hash": suite.suite_hash, "requirements": statuses,
            "measurements": measurements, "selected_template_ids": selected}


def stair_cell_scale_report(suite: StairUnifiedEvaluationSuite, split: str) -> dict[str, Any]:
    scale = suite.to_dict()["scale_context"]
    length, gravity = scale["support_length_m"], scale["gravity_mps2"]
    return {
        "scale_context": scale,
        "cells": {cell.cell_id: {
            "riser_heights_m": list(cell.riser_heights_m),
            "riser_heights_over_support_length": [h / length for h in cell.riser_heights_m],
            "tread_depths_m": list(cell.tread_depths_m),
            "tread_depths_over_support_length": [d / length for d in cell.tread_depths_m],
            "max_adjacent_riser_change_over_support_length": max(
                abs(a - b) / length for a, b in zip(cell.riser_heights_m, cell.riser_heights_m[1:])
            ),
            "max_adjacent_tread_change_over_support_length": max(
                abs(a - b) / length for a, b in zip(cell.tread_depths_m, cell.tread_depths_m[1:])
            ),
            "roughness_amplitude_m": float(cell.surface_profile.get("amplitude_m", 0)),
            "roughness_amplitude_over_support_length": float(cell.surface_profile.get("amplitude_m", 0)) / length,
            "roughness_wavelengths_over_support_length": (
                {"longitudinal_by_tread": [d / cell.surface_profile["longitudinal_cycles"] / length
                                             for d in cell.tread_depths_m],
                 "lateral": cell.size_m[1] / cell.surface_profile["lateral_cycles"] / length}
                if cell.surface_profile.get("amplitude_m") else None
            ),
            "horizon_s": cell.episode_horizon_s,
            "horizon_over_gravity_time": cell.episode_horizon_s * math.sqrt(gravity / length),
        } for cell in suite.cells_for_cohort(split)},
    }


def refine_stair_calibration_suite(
    *, suite: StairUnifiedEvaluationSuite, summary: Mapping[str, Any],
    training_profile: StairUnifiedTrainingProfile,
) -> tuple[StairUnifiedEvaluationSuite, dict[str, Any]]:
    """One public refinement: six nearby levels, with C1 patterns kept separate."""
    bound, surface_decision = calibrate_stair_surface_base(
        suite=suite, summary=summary, training_profile=training_profile)
    payload = bound.to_dict()
    templates = {item["template_id"]: item for item in payload["calibration_templates"]}
    source_templates = {item["template_id"]: item for item in suite.to_dict()["calibration_templates"]}
    rates = summary["ordered_summary"]["aggregate_success_rate_by_cell"]
    changes = {}

    def interpolate(left, right, fraction, prototype):
        result = copy.deepcopy(prototype)
        if "surface_profile" in result:
            target, a, b = result["surface_profile"], left["surface_profile"], right["surface_profile"]
            fields = ("amplitude_m",)
        else:
            target, a, b = result, left, right
            fields = ("riser_height_m", "tread_depth_m") if result["kind"] == "regular" else (
                "riser_base_m", "riser_offset_m", "tread_offset_m")
        for key in fields:
            target[key] = float(a[key]) + fraction * (float(b[key]) - float(a[key]))
        return result

    for stratum in sorted({name for members in suite.score_contract["families"].values() for name in members}):
        ids = summary["stratum_cell_ids"][stratum]
        items = [templates[suite.cell(cell_id).template_id] for cell_id in ids]
        if stratum == "c2_mechanism_ood" and any(
            item["geometry"]["base_geometry"] != source_templates[item["template_id"]]["geometry"]["base_geometry"]
            for item in items
        ):
            changes[stratum] = "new measured flat base; retain full surface ladder"
            continue
        if stratum == "c1_conditional_frontier":
            groups = {}
            for i, item in enumerate(items):
                groups.setdefault(item["geometry"]["pattern_id"], []).append(i)
            original = copy.deepcopy(items)
            for indices in groups.values():
                first, last = indices[0], indices[-1]
                left, right = original[first]["geometry"], original[last]["geometry"]
                if rates[ids[last]] > 0.8:
                    left, right = right, original[-1]["geometry"]
                elif rates[ids[first]] < 0.2:
                    left, right = original[0]["geometry"], left
                for i, fraction in zip(indices, (0.35, 0.75), strict=True):
                    items[i]["geometry"] = interpolate(left, right, fraction, original[i]["geometry"])
            changes[stratum] = "two refinements per training pattern"
        else:
            middle = min(range(len(ids)), key=lambda i: (abs(rates[ids[i]] - 0.5), i))
            lo, hi = max(0, middle - 1), min(len(ids) - 1, middle + 1)
            left, right = copy.deepcopy(items[lo]["geometry"]), copy.deepcopy(items[hi]["geometry"])
            for i, item in enumerate(items):
                item["geometry"] = interpolate(left, right, i / 5, item["geometry"])
            changes[stratum] = {"lower_source_cell": ids[lo], "upper_source_cell": ids[hi]}
    payload["suite_id"] = payload["suite_id"].replace("_surface_calibration_v2", "_refined_calibration_v2")
    result = StairUnifiedEvaluationSuite.freeze(payload, training_profile=training_profile)
    return result, {"source_suite_hash": suite.suite_hash,
                    "source_summary_hash": canonical_json_hash(summary),
                    "surface_base": surface_decision, "refinements": changes,
                    "result_suite_hash": result.suite_hash}


def freeze_stair_formal_suite(
    *, calibration_suite: StairUnifiedEvaluationSuite, calibration_summary: Mapping[str, Any],
    training_profile: StairUnifiedTrainingProfile,
) -> StairUnifiedEvaluationSuite:
    """Apply the declared selection once, without inspecting accept or sealed results."""
    selection = select_stair_calibration_levels(suite=calibration_suite, summary=calibration_summary)
    if selection["status"] != "GO":
        raise ValueError("Stair calibration has no success transition in every OOD family")
    payload = calibration_suite.to_dict()
    from true_rl_agent.tasks.stair_climbing.study_holdout import FINAL_HOLDOUT_POLICY

    payload["final_holdout_policy"] = dict(FINAL_HOLDOUT_POLICY)
    templates = {item["template_id"]: item for item in payload["calibration_templates"]}
    chosen = [templates[identifier] for identifiers in selection["selected_template_ids"].values()
              for identifier in identifiers]
    payload["suite_id"] = payload["suite_id"].replace("_calibration_v2", "_calibrated_formal_v2")
    _set_formal_instances(payload, chosen, training_profile=training_profile)
    frozen = StairUnifiedEvaluationSuite.freeze(payload, training_profile=training_profile)
    calibration_geometries = {cell.geometry_digest for cell in frozen.cells_for_cohort("calibration")}
    for cohort in frozen.to_dict()["formal_cohorts"]:
        cells = frozen.cells_for_cohort(cohort)
        overlap = calibration_geometries & {cell.geometry_digest for cell in cells}
        if overlap - _surface_control_geometry_digests(cells):
            raise ValueError("Stair formal geometry overlaps the public calibration pool")
    return frozen


def _set_formal_instances(payload, chosen, *, training_profile):
    """Vary C2 surface phase on its measured base; shift other task strata."""
    length = payload["scale_context"]["support_length_m"]
    positions, counts = {}, {}
    for item in chosen:
        stratum = item["stratum_id"]
        positions[canonical_json_hash(item["geometry"])] = counts.get(stratum, 0)
        counts[stratum] = counts.get(stratum, 0) + 1
    surface_bases = {
        canonical_json_hash(item["geometry"]["base_geometry"])
        for item in chosen if item["stratum_id"] == "c2_mechanism_ood"
    }

    def instance(item, cohort_index):
        geometry = item["geometry"]
        base = geometry.get("base_geometry", geometry)
        try:
            position = positions[canonical_json_hash(base)]
        except KeyError as exc:
            raise ValueError(
                f"Stair formal template {item['template_id']} lacks its selected flat control"
            ) from exc
        if canonical_json_hash(base) in surface_bases:
            # A shared base gives every C2 amplitude the same macro difficulty.
            # Its exact flat control is reusable evidence, not new geometry.
            result = copy.deepcopy(geometry)
            if "surface_profile" in result:
                surface = result["surface_profile"]
                surface["phase_index"] = (
                    surface["phase_index"] + 2 * (cohort_index + 1)
                ) % surface["phase_count"]
            return result
        index = (position + cohort_index) % 3
        return stair_geometry_variant(
            geometry, stratum_id=item["stratum_id"], training_profile=training_profile,
            support_length_m=length, height_fraction=(-0.021, 0.007, 0.023)[index],
            tread_fraction=(0.025, -0.025, 0.009)[index],
            surface_phase_shift=2 * cohort_index,
        )
    payload["cell_templates"] = [dict(item, geometry=instance(item, 0)) for item in chosen]
    payload["cohort_geometry_overrides"] = {
        cohort: {item["template_id"]: instance(item, index) for item in chosen}
        for index, cohort in enumerate(STAIR_UNIFIED_FORMAL_COHORTS) if index
    }


def calibrate_stair_surface_base(
    *, suite: StairUnifiedEvaluationSuite, summary: Mapping[str, Any],
    training_profile: StairUnifiedTrainingProfile,
) -> tuple[StairUnifiedEvaluationSuite, dict[str, Any]]:
    """Place unseen roughness on an empirically supported ID staircase.

    Scaling alone cannot encode foot geometry, torque or gait. The same frozen
    anchor chooses the highest ID rise it completes at least 90% of the time;
    C2 then changes only the surface, with a matching flat control retained.
    """
    if summary["split"] != "calibration" or summary["ordered_suite_hash"] != suite.suite_hash:
        raise ValueError("Stair surface-base selection requires its calibration evidence")
    rates = summary["ordered_summary"]["aggregate_success_rate_by_cell"]
    eligible = [cell for cell in suite.cells_for_cohort("calibration")
                if cell.stratum_id in {"c0_id", "c1_id"} and rates[cell.cell_id] >= 0.9]
    if not eligible:
        raise ValueError("Stair anchor cannot support a matched ID surface control")
    base_cell = max(eligible, key=lambda cell: (sum(cell.riser_heights_m) / cell.step_count, cell.cell_id))
    payload = suite.to_dict()
    templates = {item["template_id"]: item for item in payload["calibration_templates"]}
    base = templates[base_cell.template_id]["geometry"]
    length = payload["scale_context"]["support_length_m"]
    cycles_x = max(1, round((sum(base_cell.tread_depths_m) / base_cell.step_count) / (0.35 * length)))
    cycles_y = max(1, round(base_cell.size_m[1] / (0.70 * length)))
    vertices_x, vertices_y = 4 * cycles_x + 1, 4 * cycles_y + 1
    if max(vertices_x, vertices_y) > 513:
        raise ValueError("Stair normalized wavelengths exceed the supported mesh resolution")
    for level, fraction in enumerate((0.04, 0.10, 0.18, 0.26, 0.34, 0.42)):
        template = templates[f"c2_mechanism_ood_level_{level}"]
        surface = copy.deepcopy(template["geometry"]["surface_profile"])
        surface["amplitude_m"] = min(base_cell.riser_heights_m) * fraction
        surface.update(longitudinal_cycles=cycles_x, lateral_cycles=cycles_y,
                       longitudinal_vertices=vertices_x, lateral_vertices=vertices_y)
        template["geometry"] = {"kind": "c2", "base_geometry": copy.deepcopy(base),
                                "surface_profile": surface}
    payload["suite_id"] = payload["suite_id"].replace("_calibration_v2", "_surface_calibration_v2")
    bound = StairUnifiedEvaluationSuite.freeze(payload, training_profile=training_profile)
    return bound, {"source_suite_hash": suite.suite_hash, "source_summary_hash": canonical_json_hash(summary),
                   "flat_control_cell_id": base_cell.cell_id,
                   "flat_control_geometry_digest": base_cell.geometry_digest,
                   "observed_success_rate": rates[base_cell.cell_id],
                   "minimum_base_success_rate": 0.9, "selected_base_geometry": base,
                   "target_wavelength_over_support_length": {"longitudinal": 0.35, "lateral": 0.70},
                   "samples_per_cycle": 4,
                   "result_suite_hash": bound.suite_hash}
