from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from true_rl_agent.evaluation.binding import (  # noqa: E402
    canonical_json_hash,
)
from true_rl_agent.agent_core.session_genesis import (  # noqa: E402
    ReferenceProgramGenesisAnchor,
    SessionGenesisArtifact,
)
from true_rl_agent.evaluation.phases import (  # noqa: E402
    EvaluationPhase,
    EvaluationPhaseResult,
)
from true_rl_agent.evaluation.public_result import PublicEvalResult  # noqa: E402
from true_rl_agent.robots.defaults import (  # noqa: E402
    G1_37DOF_ISAACLAB_GROUPED_ADAPTER_ID,
    G1_37DOF_ISAACLAB_GROUPED_PROFILE_ID,
    GO2_12DOF_ISAACLAB_ADAPTER_ID,
    GO2_12DOF_ISAACLAB_PROFILE_ID,
    build_default_robot_registry,
)
from true_rl_agent.robots.isaaclab_locomotion import (  # noqa: E402
    isaaclab_locomotion_runtime_spec,
)
from true_rl_agent.tasks.stair_climbing.config import (  # noqa: E402
    StairClimbingTaskConfig,
)
from true_rl_agent.tasks.stair_climbing.plugin import (  # noqa: E402
    StairClimbingTaskPlugin,
)
from true_rl_agent.tasks.stair_climbing.phased_evaluation import (  # noqa: E402
    canonical_stair_phase_result_summary,
)
from true_rl_agent.tasks.stair_climbing.reference_qualification import (  # noqa: E402
    qualify_stair_reference_public_result,
)
from true_rl_agent.tasks.stair_climbing.reference_program import (  # noqa: E402
    resolve_official_stair_reference_program,
)
from true_rl_agent.tasks.stair_climbing.robot_binding import (  # noqa: E402
    resolve_stair_cell_manifest,
)
from true_rl_agent.tasks.stair_climbing.row_catalog import (  # noqa: E402
    STAIR_G1_37DOF_UNIFIED_CELL_VARIANT_ID,
    STAIR_G1_37DOF_UNIFIED_V1_CELL_VARIANT_ID,
    STAIR_GO2_12DOF_UNIFIED_CELL_VARIANT_ID,
    STAIR_GO2_12DOF_UNIFIED_V1_CELL_VARIANT_ID,
    active_unified_stair_cell_variant_ids,
    stair_robot_row,
)
from true_rl_agent.tasks.stair_climbing.runtime_projection import (  # noqa: E402
    stair_runtime_projection_from_manifest,
)
from true_rl_agent.tasks.stair_climbing.unified_training import (  # noqa: E402
    STAIR_TERRAIN_REGIME_C0,
    STAIR_TERRAIN_REGIME_C1,
    STAIR_TERRAIN_REGIME_C2,
    unified_assignment_summary,
    unified_initial_terrain_assignment,
)
from true_rl_agent.tasks.stair_climbing.unified_scoring import (  # noqa: E402
    build_reference_objective_family_mask,
)

P5_DIR = Path(__file__).resolve().parent
P0_RUNTIME_RECEIPT = ROOT / "PLAN/Variant_construction/P0/PHASE0_RUNTIME_PROBE_RECEIPT.json"
RUNTIME_RECEIPT = P5_DIR / "PHASE5_GO2_RUNTIME_RECEIPT.json"
ORACLE_STAGE_RECEIPT = P5_DIR / "PHASE5_GO2_ORACLE_STAGE_RECEIPT.json"
SMOKE_STAGE_RECEIPT = P5_DIR / "PHASE5_GO2_SMOKE_STAGE_RECEIPT.json"
CURRICULUM_ACTIVATION_PROBE = (
    P5_DIR / "PHASE5_GO2_CURRICULUM_ACTIVATION_PROBE.json"
)
QUALIFICATION_RECEIPT = (
    P5_DIR / "PHASE5_FIRST_ARRIVAL_REQUALIFICATION.json"
)
QUALIFICATION_LOAD_ONLY_RECEIPT = P5_DIR / "PHASE5_GENESIS_LOAD_ONLY_RECEIPT.json"
QUALIFICATION_GENESIS_IMPORT_RECEIPT = P5_DIR / "PHASE5_GENESIS_IMPORT_RECEIPT.json"
QUALIFICATION_TRAIN_SUMMARY = P5_DIR / "PHASE5_REFERENCE_TRAIN_SUMMARY.json"
QUALIFICATION_PHASE_CONTRACT = P5_DIR / "PHASE5_REFERENCE_PHASE_CONTRACT.json"
QUALIFICATION_PHASE_RECEIPTS = {
    EvaluationPhase.PUBLIC_DEV: P5_DIR / "PHASE5_REFERENCE_PUBLIC_DEV.json",
    EvaluationPhase.PUBLIC_ACCEPT: P5_DIR / "PHASE5_REFERENCE_PUBLIC_ACCEPT.json",
    EvaluationPhase.SEALED_VALIDATION: (
        P5_DIR / "PHASE5_REFERENCE_SEALED_VALIDATION.json"
    ),
}
QUALIFICATION_REFERENCE_ARTIFACT = (
    P5_DIR / "PHASE5_REFERENCE_EVAL_ARTIFACT.json"
)
QUALIFICATION_SESSION_GENESIS = P5_DIR / "PHASE5_SESSION_GENESIS.json"
G1_QUALIFICATION_RECEIPT = P5_DIR / "PHASE5_G1_V2_REQUALIFICATION.json"
G1_QUALIFICATION_LOAD_ONLY_RECEIPT = (
    P5_DIR / "PHASE5_G1_V2_GENESIS_LOAD_ONLY_RECEIPT.json"
)
G1_QUALIFICATION_GENESIS_IMPORT_RECEIPT = (
    P5_DIR / "PHASE5_G1_V2_GENESIS_IMPORT_RECEIPT.json"
)
G1_QUALIFICATION_TRAIN_SUMMARY = (
    P5_DIR / "PHASE5_G1_V2_REFERENCE_TRAIN_SUMMARY.json"
)
G1_QUALIFICATION_PHASE_CONTRACT = (
    P5_DIR / "PHASE5_G1_V2_REFERENCE_PHASE_CONTRACT.json"
)
G1_QUALIFICATION_PHASE_RECEIPTS = {
    EvaluationPhase.PUBLIC_DEV: P5_DIR / "PHASE5_G1_V2_REFERENCE_PUBLIC_DEV.json",
    EvaluationPhase.PUBLIC_ACCEPT: (
        P5_DIR / "PHASE5_G1_V2_REFERENCE_PUBLIC_ACCEPT.json"
    ),
    EvaluationPhase.SEALED_VALIDATION: (
        P5_DIR / "PHASE5_G1_V2_REFERENCE_SEALED_VALIDATION.json"
    ),
}
G1_QUALIFICATION_REFERENCE_ARTIFACT = (
    P5_DIR / "PHASE5_G1_V2_REFERENCE_EVAL_ARTIFACT.json"
)
G1_QUALIFICATION_SESSION_GENESIS = P5_DIR / "PHASE5_G1_V2_SESSION_GENESIS.json"
V1_COUNTERFACTUAL_DIR = P5_DIR / "visualization/go2_derived_c0_id_01"
V1_COUNTERFACTUAL_CAPTURE = V1_COUNTERFACTUAL_DIR / "capture_result.json"
V1_COUNTERFACTUAL_RECEIPT = V1_COUNTERFACTUAL_DIR / "visualization_receipt.json"
GO2_PROTOCOL = ROOT / "configs/tasks/stair_go2_12dof_unified_protocol_v2.json"
GO2_TRAINING_PROTOCOL = ROOT / "configs/tasks/stair_go2_12dof_unified_protocol_v1.json"
G1_PROTOCOL = ROOT / "configs/tasks/stair_unified_protocol_v2.json"
G1_TRAINING_PROTOCOL = ROOT / "configs/tasks/stair_unified_protocol_v1.json"

VERIFY_SCHEMA_VERSION = "true_rl_agent.variant_construction.phase5_verify.v1"
QUALIFICATION_SCHEMA_VERSION = (
    "true_rl_agent.variant_construction.phase5_first_arrival_requalification.v1"
)
_TRIAL_EXECUTION_CONTRACT_SCHEMA_VERSION = (
    "true_rl_agent.trial_execution_contract.v1"
)
_EVALUATION_PHASE_RECEIPT_SCHEMA_VERSION = (
    "true_rl_agent.trial_eval_phase_receipt.v1"
)
_REFERENCE_SESSION_ID = "p5-go2-pseudofall-removed-0a3f08a"
_TRAINING_SESSION_ID = "p5-go2-derived-reference-ef6826e"
_TRAINING_SOURCE_COMMIT = "ef6826eeac13de6fd44f4cb25eaf1b7249f03e55"
_PHASE_EXECUTION_SOURCE_COMMIT = "0a3f08a80969b52dce8be7a1c307a0420abe50c5"
_QUALIFICATION_SOURCE_COMMIT = "0a3f08a80969b52dce8be7a1c307a0420abe50c5"
_REFERENCE_BUDGET_STEPS = 196_608_000
_REFERENCE_TRAIN_SEED = 1_262_835_026
_G1_REFERENCE_SESSION_ID = "p5-g1-pseudofall-removed-0a3f08a"
_G1_TRAINING_SESSION_ID = "phase4-unified-validation-5a30129-20260829"
_G1_REFERENCE_TRAIN_SEED = 1_275_486_543

_EXPECTED_REWARD_WEIGHTS = {
    "track_lin_vel_xy_exp": 3.5,
    "track_ang_vel_z_exp": 0.75,
    "ang_vel_xy_l2": -0.05,
    "dof_torques_l2": -0.0002,
    "dof_acc_l2": -2.5e-7,
    "action_rate_l2": -0.01,
    "feet_air_time": 0.25,
    "flat_orientation_l2": -2.5,
}
_EXPECTED_STRATA = {
    "c0_id",
    "c1_id",
    "height_ood",
    "tread_ood",
    "joint_ood",
    "phase_composition_ood",
    "c1_conditional_frontier",
    "c2_mechanism_ood",
}
_TASK_ROBOT_ADAPTER_PATHS = {
    "legacy_reward_adapter.py",
    "phase0_contract.py",
    "robot_binding.py",
    "row_catalog.py",
}


def _go2_qualification_spec() -> dict[str, Any]:
    return {
        "receipt": QUALIFICATION_RECEIPT,
        "load_only_receipt": QUALIFICATION_LOAD_ONLY_RECEIPT,
        "genesis_import_receipt": QUALIFICATION_GENESIS_IMPORT_RECEIPT,
        "train_summary": QUALIFICATION_TRAIN_SUMMARY,
        "phase_contract": QUALIFICATION_PHASE_CONTRACT,
        "phase_receipts": QUALIFICATION_PHASE_RECEIPTS,
        "reference_artifact": QUALIFICATION_REFERENCE_ARTIFACT,
        "session_genesis": QUALIFICATION_SESSION_GENESIS,
        "reference_session_id": _REFERENCE_SESSION_ID,
        "training_session_id": _TRAINING_SESSION_ID,
        "phase_execution_source_commit": _PHASE_EXECUTION_SOURCE_COMMIT,
        "qualification_source_commit": _QUALIFICATION_SOURCE_COMMIT,
        "reference_subject_id": "go2-stair-official-flat-derived-v1",
        "evaluation_device": "cuda:0",
        "training_device": "cuda:0",
        "training_seed": _REFERENCE_TRAIN_SEED,
        "reward_term_count": len(_EXPECTED_REWARD_WEIGHTS),
        "protocol": GO2_PROTOCOL,
    }


def _g1_qualification_spec() -> dict[str, Any]:
    return {
        "receipt": G1_QUALIFICATION_RECEIPT,
        "load_only_receipt": G1_QUALIFICATION_LOAD_ONLY_RECEIPT,
        "genesis_import_receipt": G1_QUALIFICATION_GENESIS_IMPORT_RECEIPT,
        "train_summary": G1_QUALIFICATION_TRAIN_SUMMARY,
        "phase_contract": G1_QUALIFICATION_PHASE_CONTRACT,
        "phase_receipts": G1_QUALIFICATION_PHASE_RECEIPTS,
        "reference_artifact": G1_QUALIFICATION_REFERENCE_ARTIFACT,
        "session_genesis": G1_QUALIFICATION_SESSION_GENESIS,
        "reference_session_id": _G1_REFERENCE_SESSION_ID,
        "training_session_id": _G1_TRAINING_SESSION_ID,
        "phase_execution_source_commit": _PHASE_EXECUTION_SOURCE_COMMIT,
        "qualification_source_commit": _QUALIFICATION_SOURCE_COMMIT,
        "reference_subject_id": "official-g1-rough-reward-vnext-v1",
        "evaluation_device": "cuda:1",
        "training_device": "cuda:0",
        "training_seed": _G1_REFERENCE_TRAIN_SEED,
        "reward_term_count": 15,
        "protocol": G1_PROTOCOL,
    }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def _without(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != field}


def _file_evidence(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "relative_path": path.relative_to(P5_DIR).as_posix(),
        "sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _verify_hashed_payload(
    payload: Mapping[str, Any],
    *,
    hash_key: str,
    name: str,
) -> None:
    expected = payload.get(hash_key)
    _require(isinstance(expected, str), f"{name} has no {hash_key}")
    _require(
        expected == canonical_json_hash(_without(payload, hash_key)),
        f"{name} {hash_key} mismatch",
    )


def _outside_interval_distance(value: float, *, low: float, high: float) -> float:
    if value < low:
        return low - value
    if value > high:
        return value - high
    return 0.0


def _plain_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_plain_value(item) for item in value]
    return value


def _resolve_config(path: Path) -> tuple[Any, Any, Any, Any]:
    provisional = StairClimbingTaskConfig.defaults(path)
    plugin = StairClimbingTaskPlugin(provisional)
    manifest = resolve_stair_cell_manifest(
        cell_variant_id=provisional.cell_variant_id,
        config=provisional,
        plugin=plugin,
    )
    config = StairClimbingTaskConfig.defaults(
        path,
        resolved_cell_task_protocol_hash=manifest.resolved_cell_task_protocol_hash,
    )
    plugin = StairClimbingTaskPlugin(config, resolved_cell_manifest=manifest)
    projection = stair_runtime_projection_from_manifest(manifest)
    return config, plugin, manifest, projection


def _verify_phase0_robot_facts(profile: Any) -> dict[str, Any]:
    receipt = _load_object(P0_RUNTIME_RECEIPT)
    _verify_hashed_payload(receipt, hash_key="receipt_hash", name="Phase 0 runtime receipt")
    matches = [
        item
        for item in receipt["robot_runtime_facts"]
        if item.get("profile_key") == "p0.go2_upstream_execution_basis"
    ]
    _require(len(matches) == 1, "Phase 0 Go2 runtime fact record is not unique")
    authoritative = matches[0]
    _require(
        profile.runtime_facts == authoritative,
        "Go2 RobotProfile facts differ from the authoritative Phase 0 Isaac probe",
    )
    _require(
        profile.runtime_facts_hash == authoritative["runtime_facts_hash"],
        "Go2 RobotProfile runtime facts hash differs from Phase 0",
    )
    return {
        "runtime_facts_hash": profile.runtime_facts_hash,
        "joint_count": len(authoritative["joint_names"]),
        "body_count": len(authoritative["body_names"]),
        "support_count": len(authoritative["support_body_names"]),
        "asset_file_count": authoritative["asset_closure"]["file_count"],
        "asset_total_bytes": authoritative["asset_closure"]["total_bytes"],
        "status": "pass",
    }


def _verify_cell_and_distribution() -> dict[str, Any]:
    registry = build_default_robot_registry()
    profile, adapter = registry.execution(
        robot_profile_id=GO2_12DOF_ISAACLAB_PROFILE_ID,
        runtime_adapter_id=GO2_12DOF_ISAACLAB_ADAPTER_ID,
    )
    p0 = _verify_phase0_robot_facts(profile)
    runtime = isaaclab_locomotion_runtime_spec(adapter.runtime_adapter_id)
    _require(runtime.runtime_adapter_id == adapter.runtime_adapter_id, "Go2 runtime adapter drift")
    _require("config.go2.rough_env_cfg" in runtime.environment_cfg_entrypoint, "Go2 env source drift")
    _require(runtime.agent_configuration.algorithm["entropy_coef"] == 0.01, "Go2 PPO drift")

    config, plugin, manifest, projection = _resolve_config(GO2_PROTOCOL)
    _require(
        config.cell_variant_id == STAIR_GO2_12DOF_UNIFIED_CELL_VARIANT_ID,
        "Go2 protocol selected the wrong cell",
    )
    row = stair_robot_row(config.cell_variant_id)
    _require(row.protocol_relative_path == GO2_PROTOCOL.relative_to(ROOT).as_posix(), "row protocol drift")
    _require(
        row.load_binding(project_root=ROOT).binding_hash
        == manifest.robot_task_binding["binding_hash"],
        "row binding drift",
    )
    training_config, _training_plugin, training_manifest, _training_projection = (
        _resolve_config(GO2_TRAINING_PROTOCOL)
    )
    _require(
        training_config.cell_variant_id
        == STAIR_GO2_12DOF_UNIFIED_V1_CELL_VARIANT_ID,
        "Go2 predecessor training protocol selected the wrong cell",
    )
    training_row = stair_robot_row(training_config.cell_variant_id)
    _require(
        training_row.protocol_relative_path
        == GO2_TRAINING_PROTOCOL.relative_to(ROOT).as_posix(),
        "Go2 predecessor row protocol drift",
    )
    training_lineage = config.unified_protocol.to_dict().get("training_lineage")
    _require(
        training_lineage
        == {
            "mode": "evaluation_only_successor",
            "source_cell_variant_id": training_config.cell_variant_id,
            "source_base_task_protocol_hash": training_config.base_task_protocol_hash,
            "source_resolved_cell_task_protocol_hash": (
                training_manifest.resolved_cell_task_protocol_hash
            ),
            "source_resolved_cell_manifest_hash": training_manifest.manifest_hash,
            "source_training_profile_hash": (
                training_config.unified_training_profile.profile_hash
            ),
        },
        "Go2 evaluation-only successor lineage drift",
    )
    _require(
        config.unified_training_profile.profile_hash
        == training_config.unified_training_profile.profile_hash
        and config.training_agent_configuration().agent_configuration_hash
        == training_config.training_agent_configuration().agent_configuration_hash
        and config.formal_training_binding().binding_hash
        == training_config.formal_training_binding().binding_hash,
        "Go2 evaluation successor changed training semantics",
    )
    success_contract = config.episode_outcomes()["success"]
    _require(
        success_contract["completion_mode"] == "first_valid_top_arrival_latched"
        and success_contract["post_arrival_failure_policy"]
        == "not_evaluated_outcome_closes_on_first_valid_top_arrival"
        and success_contract["pre_arrival_blocking_latches"]
        == ["forbidden_contact", "off_course"]
        and "low_base_clearance_detected"
        not in config.episode_outcomes()["diagnostic_flags"]
        and "low_base_clearance"
        not in config.episode_outcomes()["signals"]
        and "low_base_clearance"
        not in config.episode_outcomes()["primary_reasons"],
        "Go2 active Stair completion semantics drift",
    )
    _require(
        "minimum_kinematic_base_clearance_above_top_m" not in success_contract
        and "requires_kinematic_base_clearance_above_top" not in success_contract,
        "Go2 active success retained pseudo-fall clearance",
    )
    _require(projection.action_count == 12, "Go2 action dimension drift")
    _require(projection.support_count == 4, "Go2 support cardinality drift")
    _require(projection.forbidden_contact_count == 1, "Go2 forbidden-contact selection drift")
    _require(projection.kinematic_base_source == "articulation_root", "Go2 base source drift")
    _require(projection.support_transition_inputs, "Go2 air-time transition inputs are absent")
    _require(abs(projection.task_vertical_scale - 0.589087) <= 1.0e-12, "Go2 vertical scale drift")
    _require(projection.task_horizontal_scale == 1.0, "Go2 horizontal scale drift")

    suite = config.ordered_evaluation_suite
    profile_payload = config.unified_training_profile.to_dict()
    _require(
        profile_payload["training_exposure"]["allowed_regimes"]
        == [STAIR_TERRAIN_REGIME_C0, STAIR_TERRAIN_REGIME_C1],
        "Go2 training regimes changed",
    )
    _require(
        profile_payload["training_exposure"]["forbidden_regimes"]
        == [STAIR_TERRAIN_REGIME_C2],
        "C2 leaked into Go2 training",
    )
    terrain_types, terrain_levels = unified_initial_terrain_assignment(
        num_envs=config.train_num_envs,
        seed=config.train_seed,
        profile=config.unified_training_profile,
    )
    assignment = unified_assignment_summary(
        types=terrain_types,
        levels=terrain_levels,
        profile=config.unified_training_profile,
    )
    _require(
        assignment["regime_counts"]
        == {STAIR_TERRAIN_REGIME_C0: 2048, STAIR_TERRAIN_REGIME_C1: 2048},
        "Go2 training assignment is not 50/50",
    )
    _require(STAIR_TERRAIN_REGIME_C2 not in assignment["regime_counts"], "C2 appears in training assignment")
    geometry_by_cohort: dict[str, set[str]] = {}
    phase_labels: set[str] = set()
    c2_amplitudes_by_cohort: dict[str, list[float]] = {}
    scalar_distances_by_cohort: dict[str, dict[str, list[float]]] = {}
    height_upper = float(profile_payload["terrain"]["c0"]["riser_range_m"][1])
    c1_terrain = profile_payload["terrain"]["c1"]
    tread_low = min(float(value) for value in c1_terrain["tread_bases_m"]) - float(
        c1_terrain["tread_offset_m"]
    )
    tread_high = max(float(value) for value in c1_terrain["tread_bases_m"]) + float(
        c1_terrain["tread_offset_m"]
    )
    for cohort in ("public_dev", "public_accept", "sealed_validation"):
        cells = suite.cells_for_cohort(cohort)
        _require(len(cells) == 24, f"{cohort} cell count changed")
        counts = Counter(cell.stratum_id for cell in cells)
        _require(set(counts) == _EXPECTED_STRATA, f"{cohort} stratum set changed")
        _require(set(counts.values()) == {3}, f"{cohort} stratum balance changed")
        geometries = {cell.geometry_digest for cell in cells}
        _require(len(geometries) == 24, f"{cohort} contains duplicate geometry")
        _require(
            all(
                cell.floor_depth_m
                == float(config.unified_training_profile.terrain["floor_depth_m"])
                for cell in cells
            ),
            f"{cohort} floor geometry differs from training",
        )
        for cell in cells:
            if cell.stratum_id == "phase_composition_ood":
                phase = cell.factor_levels["phase"]
                phase_labels.add(phase)
                _require(
                    config.unified_training_profile.phase_changes_training_sequence(
                        pattern_id=cell.factor_levels["sequence_pattern"],
                        phase_steps=int(phase[1:]),
                    ),
                    "phase OOD reuses a training sequence",
                )
        c2_amplitudes_by_cohort[cohort] = sorted(
            float(cell.surface_profile["amplitude_m"])
            for cell in cells
            if cell.stratum_id == "c2_mechanism_ood"
        )
        scalar_distances_by_cohort[cohort] = {
            "height_ood": sorted(
                float(cell.riser_heights_m[0]) - height_upper
                for cell in cells
                if cell.stratum_id == "height_ood"
            ),
            "tread_ood": sorted(
                _outside_interval_distance(
                    float(cell.tread_depths_m[0]),
                    low=tread_low,
                    high=tread_high,
                )
                for cell in cells
                if cell.stratum_id == "tread_ood"
            ),
        }
        geometry_by_cohort[cohort] = geometries
    _require(phase_labels == {"p1", "p2", "p4", "p6"}, "phase OOD coverage changed")
    _require(
        set(map(tuple, c2_amplitudes_by_cohort.values()))
        == {(0.002356, 0.003535, 0.004713)},
        "Go2 C2 amplitude ladder changed across cohorts",
    )
    maximum_scalar_ladder_spread = max(
        max(values) - min(values)
        for stratum in ("height_ood", "tread_ood")
        for values in zip(
            *(
                scalar_distances_by_cohort[cohort][stratum]
                for cohort in ("public_dev", "public_accept", "sealed_validation")
            ),
            strict=True,
        )
    )
    maximum_scalar_ladder_spread = round(maximum_scalar_ladder_spread, 9)
    _require(
        maximum_scalar_ladder_spread <= 0.01,
        "formal cohorts no longer use comparable scalar OOD distance ladders",
    )
    _require(
        len(set().union(*geometry_by_cohort.values())) == 72,
        "formal cohorts are not geometry-disjoint",
    )
    formal_episodes = (
        sum(suite.seed_count(cohort) for cohort in geometry_by_cohort)
        * 24
        * suite.episodes_per_seed_per_cell
    )
    _require(formal_episodes == 1080, "formal evaluation cost changed")

    g1_config, _g1_plugin, g1_manifest, g1_projection = _resolve_config(G1_PROTOCOL)
    g1_profile, _g1_adapter = registry.execution(
        robot_profile_id=G1_37DOF_ISAACLAB_GROUPED_PROFILE_ID,
        runtime_adapter_id=G1_37DOF_ISAACLAB_GROUPED_ADAPTER_ID,
    )
    _require(
        g1_config.cell_variant_id == STAIR_G1_37DOF_UNIFIED_CELL_VARIANT_ID,
        "G1 unified cell identity changed",
    )
    g1_anchors = {
        (cell.geometry_digest, cell.stratum_id)
        for cell in g1_config.ordered_evaluation_suite.cells
    }
    go2_anchors = {(cell.geometry_digest, cell.stratum_id) for cell in suite.cells}
    shared_anchor_strata = {stratum for _, stratum in g1_anchors & go2_anchors}
    _require(shared_anchor_strata == {"c0_id"}, "cross-row anchors changed")
    g1_training_config, _g1_training_plugin, g1_training_manifest, _ = (
        _resolve_config(G1_TRAINING_PROTOCOL)
    )
    _require(
        g1_training_config.cell_variant_id
        == STAIR_G1_37DOF_UNIFIED_V1_CELL_VARIANT_ID
        and g1_training_config.base_task_protocol_hash
        == "e65239927cbf98f6cf12b0ddb959c6bd1390bdf7fd1c6a262fbbab7f1f04dcf8",
        "G1 predecessor training identity regressed",
    )
    _require(
        g1_training_config.training_agent_configuration().agent_configuration_hash
        == "8a5d64c47ac034695d2eec23ff60a89ecbf1253694edb8c373a4da86872bb450",
        "G1 unified agent configuration regressed",
    )
    g1_training_lineage = g1_config.unified_protocol.to_dict().get("training_lineage")
    _require(
        g1_training_lineage.get("source_resolved_cell_manifest_hash")
        == g1_training_manifest.manifest_hash
        and g1_training_lineage.get("source_resolved_cell_task_protocol_hash")
        == g1_training_manifest.resolved_cell_task_protocol_hash
        and g1_config.formal_training_binding().binding_hash
        == g1_training_config.formal_training_binding().binding_hash,
        "G1 evaluation-only successor changed training lineage",
    )
    go2_protocol_payload = config.unified_protocol.to_dict()
    g1_protocol_payload = g1_config.unified_protocol.to_dict()
    shared_protocol_sections = (
        "training",
        "training_execution",
        "runtime_controls",
        "candidate_authority",
        "reference_readiness",
        "required_pass_gates",
    )
    _require(
        all(
            go2_protocol_payload[name] == g1_protocol_payload[name]
            for name in shared_protocol_sections
        ),
        "Go2 row changed robot-independent Stair protocol authority",
    )
    go2_outcomes = _plain_value(config.unified_protocol.episode_outcomes)
    g1_outcomes = _plain_value(g1_config.unified_protocol.episode_outcomes)
    vertically_scaled_outcome_paths = (
        (
            "support_surface_projection",
            "maximum_projected_surface_penetration_m",
        ),
        ("signals", "insufficient_clearance", "threshold"),
        ("success", "support_effector_surface_clearance_range_m", 1),
    )
    for path in vertically_scaled_outcome_paths:
        g1_parent: Any = g1_outcomes
        go2_parent: Any = go2_outcomes
        for key in path[:-1]:
            g1_parent = g1_parent[key]
            go2_parent = go2_parent[key]
        leaf = path[-1]
        g1_value = float(g1_parent[leaf])
        go2_value = float(go2_parent[leaf])
        _require(
            abs(go2_value - g1_value * projection.task_vertical_scale) <= 0.0005,
            "Go2 task outcome threshold is not explained by robot vertical scale",
        )
        go2_parent[leaf] = g1_parent[leaf]
    _require(
        go2_outcomes == g1_outcomes,
        "Go2 row changed Stair outcome semantics beyond vertical scaling",
    )
    go2_suite_payload = suite.to_dict()
    g1_suite_payload = g1_config.ordered_evaluation_suite.to_dict()
    shared_suite_sections = (
        "seed_counts",
        "calibration_seed_count",
        "episodes_per_seed_per_cell",
        "formal_cohorts",
        "geometry_policy",
        "score_contract",
        "qualification_rule",
    )
    _require(
        all(
            go2_suite_payload[name] == g1_suite_payload[name]
            for name in shared_suite_sections
        ),
        "Go2 row changed robot-independent Stair evaluation semantics",
    )
    g1_inputs = {field.name for field in g1_projection.reward_input_fields_vnext}
    _require("support_last_air_time" not in g1_inputs, "Go2-only input leaked into G1 ABI")
    _require("support_first_contact" not in g1_inputs, "Go2-only input leaked into G1 ABI")
    _require(
        {
            STAIR_G1_37DOF_UNIFIED_CELL_VARIANT_ID,
            STAIR_GO2_12DOF_UNIFIED_CELL_VARIANT_ID,
        }
        <= set(active_unified_stair_cell_variant_ids()),
        "an established unified Stair row disappeared",
    )

    official = resolve_official_stair_reference_program(
        config,
        project_root=ROOT,
        resolved_cell_manifest=manifest,
    )
    training_official = resolve_official_stair_reference_program(
        training_config,
        project_root=ROOT,
        resolved_cell_manifest=training_manifest,
    )
    g1_official = resolve_official_stair_reference_program(
        g1_config,
        project_root=ROOT,
        resolved_cell_manifest=g1_manifest,
    )
    g1_training_official = resolve_official_stair_reference_program(
        g1_training_config,
        project_root=ROOT,
        resolved_cell_manifest=g1_training_manifest,
    )
    weights = {term.name: term.weight for term in official.reward_program.reward_terms}
    _require(
        weights == _EXPECTED_REWARD_WEIGHTS,
        "Go2 reference reward differs from the frozen derived manager",
    )
    equivalence = config.vnext_reward_authority.runtime_authority_payload()["reference_equivalence"]
    _require(set(equivalence["active_term_names"]) == set(weights), "active reward authority drift")
    _require(
        equivalence["zero_weight_term_names"]
        == ["lin_vel_z_l2", "dof_pos_limits"],
        "zero-weight reward authority drift",
    )
    _require(equivalence["disabled_term_names"] == ["undesired_contacts"], "disabled reward authority drift")

    prompt_sections = plugin.prompt_adapter().build_prompt_capacity_preflight_sections()
    task_section = prompt_sections["task"]
    reward_inputs_contract = prompt_sections["reward_inputs_contract"]
    eval_summary = task_section["fixed_eval_protocol_summary"]
    train_summary = task_section["fixed_train_protocol_summary"]
    public_contract = prompt_sections["public_eval_contract"]
    public_strata = list(
        dict.fromkeys(
            cell.stratum_id for cell in suite.cells_for_cohort("public_dev")
        )
    )
    _require(eval_summary["strata"] == public_strata, "Go2 prompt strata differ from suite")
    _require(eval_summary["cells_per_stratum"] == 3, "Go2 prompt cell count differs from suite")
    _require(train_summary["c2_training_exposure"] == "forbidden", "Go2 prompt misstates C2 exposure")
    _require(
        reward_inputs_contract["schema_version"]
        == "true_rl_agent.reward_inputs_contract.stair.v13",
        "Go2 prompt uses the legacy reward-input contract projection",
    )
    _require(
        str(reward_inputs_contract["input_type"]).endswith(".StairRewardInputsVNext"),
        "Go2 prompt misstates its runtime input type",
    )
    prompt_constraints = reward_inputs_contract["field_semantics_constraints"]
    _require(
        "support_contact_force_history"
        in prompt_constraints["high_risk_axis_semantics"],
        "Go2 prompt lacks morphology-neutral support history semantics",
    )
    _require(
        prompt_constraints["termination_masks"]["equivalent_fields"]
        == {"terminated_non_timeout": "forbidden_contact"},
        "Go2 prompt names a non-existent legacy termination field",
    )
    _require(public_contract["primary_objective"] == suite.score_contract["j_primary"], "Go2 prompt objective drift")
    _require(
        public_contract["named_non_regression_guards"]
        == list(suite.score_contract["named_guards"]),
        "Go2 prompt guards drift",
    )
    prompt_text = json.dumps(prompt_sections, sort_keys=True)
    _require("foot(0 left;1 right)" not in prompt_text, "two-foot semantics leaked into Go2 prompt")
    _require("StairRewardInputsV2" not in prompt_text, "legacy input type leaked into Go2 prompt")
    _require("cohort_geometry_overrides" not in prompt_text, "private geometry overrides leaked into prompt")
    _require(
        not any(
            cell.cell_id in prompt_text
            for cell in suite.cells_for_cohort("sealed_validation")
        ),
        "sealed Go2 geometry leaked into prompt",
    )
    prompt_morphology_names = {
        str(name)
        for field in ("body_names", "joint_names")
        for name in profile.runtime_facts[field]
        if str(name) not in set(profile.semantic_roles)
    }
    _require(
        not {name for name in prompt_morphology_names if name in prompt_text},
        "Go2 concrete morphology leaked into the reward-authoring prompt",
    )
    expected_manifest_contracts = {
        "acceptance_protocol_hash": config.acceptance_protocol.protocol_hash,
        "evaluation_protocol_hash": suite.evaluation_protocol.protocol_hash,
        "candidate_capability_profile_hash": config.candidate_capability_profile.profile_hash,
        "agent_configuration_hash": config.training_agent_configuration().agent_configuration_hash,
        "training_profile_hash": config.unified_training_profile.profile_hash,
        "evaluation_suite_hash": suite.suite_hash,
        "vnext_reward_authority_hash": config.vnext_reward_authority.authority_hash,
    }
    for name, expected in expected_manifest_contracts.items():
        _require(manifest.contract_hashes[name] == expected, f"manifest contract drift: {name}")

    g1_cell = {
        "cell_variant_id": g1_config.cell_variant_id,
        "manifest_hash": g1_manifest.manifest_hash,
        "task_protocol_hash": g1_config.task_protocol_hash,
        "resolved_cell_task_protocol_hash": (
            g1_manifest.resolved_cell_task_protocol_hash
        ),
        "base_task_protocol_hash": g1_config.base_task_protocol_hash,
        "binding_hash": g1_manifest.robot_task_binding["binding_hash"],
        "robot_profile_hash": g1_manifest.robot_profile["profile_hash"],
        "runtime_adapter_hash": g1_manifest.runtime_adapter["adapter_hash"],
        "training_profile_hash": g1_config.unified_training_profile.profile_hash,
        "evaluation_suite_hash": g1_config.ordered_evaluation_suite.suite_hash,
        "acceptance_protocol_hash": g1_config.acceptance_protocol.protocol_hash,
        "evaluation_protocol_hash": (
            g1_config.ordered_evaluation_suite.evaluation_protocol.protocol_hash
        ),
        "candidate_capability_profile_hash": (
            g1_config.candidate_capability_profile.profile_hash
        ),
        "agent_configuration_hash": (
            g1_config.training_agent_configuration().agent_configuration_hash
        ),
        "training_binding_hash": g1_config.formal_training_binding().binding_hash,
        "reward_authority_hash": g1_config.vnext_reward_authority.authority_hash,
        "reference_manifest_hash": g1_official.artifact.manifest.manifest_hash,
        "reference_capability_profile_hash": (
            g1_official.artifact.manifest.capability_profile_hash
        ),
        "reference_source_checksum": (
            g1_official.reward_program.program_source_checksum
        ),
        "formal_episode_count": formal_episodes,
        "training_predecessor": {
            "cell_variant_id": g1_training_config.cell_variant_id,
            "manifest_hash": g1_training_manifest.manifest_hash,
            "task_protocol_hash": g1_training_config.task_protocol_hash,
            "resolved_cell_task_protocol_hash": (
                g1_training_manifest.resolved_cell_task_protocol_hash
            ),
            "base_task_protocol_hash": g1_training_config.base_task_protocol_hash,
            "binding_hash": g1_training_manifest.robot_task_binding["binding_hash"],
            "robot_profile_hash": g1_training_manifest.robot_profile["profile_hash"],
            "runtime_adapter_hash": (
                g1_training_manifest.runtime_adapter["adapter_hash"]
            ),
            "training_profile_hash": (
                g1_training_config.unified_training_profile.profile_hash
            ),
            "evaluation_suite_hash": (
                g1_training_config.ordered_evaluation_suite.suite_hash
            ),
            "agent_configuration_hash": (
                g1_training_config.training_agent_configuration().agent_configuration_hash
            ),
            "training_binding_hash": (
                g1_training_config.formal_training_binding().binding_hash
            ),
            "reward_authority_hash": (
                g1_training_config.vnext_reward_authority.authority_hash
            ),
            "reference_manifest_hash": (
                g1_training_official.artifact.manifest.manifest_hash
            ),
            "reference_capability_profile_hash": (
                g1_training_official.artifact.manifest.capability_profile_hash
            ),
            "reference_source_checksum": (
                g1_training_official.reward_program.program_source_checksum
            ),
        },
        "status": "pass",
    }

    return {
        "cell_variant_id": config.cell_variant_id,
        "manifest_hash": manifest.manifest_hash,
        "task_protocol_hash": config.task_protocol_hash,
        "resolved_cell_task_protocol_hash": manifest.resolved_cell_task_protocol_hash,
        "base_task_protocol_hash": config.base_task_protocol_hash,
        "binding_hash": manifest.robot_task_binding["binding_hash"],
        "robot_profile_hash": manifest.robot_profile["profile_hash"],
        "runtime_adapter_hash": manifest.runtime_adapter["adapter_hash"],
        "training_profile_hash": config.unified_training_profile.profile_hash,
        "evaluation_suite_hash": suite.suite_hash,
        "acceptance_protocol_hash": config.acceptance_protocol.protocol_hash,
        "evaluation_protocol_hash": suite.evaluation_protocol.protocol_hash,
        "candidate_capability_profile_hash": config.candidate_capability_profile.profile_hash,
        "agent_configuration_hash": config.training_agent_configuration().agent_configuration_hash,
        "training_binding_hash": config.formal_training_binding().binding_hash,
        "reward_authority_hash": config.vnext_reward_authority.authority_hash,
        "reference_manifest_hash": official.artifact.manifest.manifest_hash,
        "reference_capability_profile_hash": (
            official.artifact.manifest.capability_profile_hash
        ),
        "reference_source_checksum": official.reward_program.program_source_checksum,
        "formal_episode_count": formal_episodes,
        "maximum_scalar_ladder_spread_m": maximum_scalar_ladder_spread,
        "cross_row_anchor_strata": sorted(shared_anchor_strata),
        "shared_protocol_sections": list(shared_protocol_sections),
        "shared_suite_sections": list(shared_suite_sections),
        "vertically_scaled_outcome_paths": [
            ".".join(str(item) for item in path)
            for path in vertically_scaled_outcome_paths
        ],
        "prompt_strata": public_strata,
        "phase0_robot_facts": p0,
        "training_predecessor": {
            "cell_variant_id": training_config.cell_variant_id,
            "manifest_hash": training_manifest.manifest_hash,
            "task_protocol_hash": training_config.task_protocol_hash,
            "resolved_cell_task_protocol_hash": (
                training_manifest.resolved_cell_task_protocol_hash
            ),
            "base_task_protocol_hash": training_config.base_task_protocol_hash,
            "binding_hash": training_manifest.robot_task_binding["binding_hash"],
            "robot_profile_hash": training_manifest.robot_profile["profile_hash"],
            "runtime_adapter_hash": training_manifest.runtime_adapter["adapter_hash"],
            "training_profile_hash": (
                training_config.unified_training_profile.profile_hash
            ),
            "evaluation_suite_hash": (
                training_config.ordered_evaluation_suite.suite_hash
            ),
            "agent_configuration_hash": (
                training_config.training_agent_configuration().agent_configuration_hash
            ),
            "training_binding_hash": (
                training_config.formal_training_binding().binding_hash
            ),
            "reward_authority_hash": (
                training_config.vnext_reward_authority.authority_hash
            ),
            "reference_manifest_hash": (
                training_official.artifact.manifest.manifest_hash
            ),
            "reference_capability_profile_hash": (
                training_official.artifact.manifest.capability_profile_hash
            ),
            "reference_source_checksum": (
                training_official.reward_program.program_source_checksum
            ),
        },
        "g1_v2": g1_cell,
        "status": "pass",
    }


def _verify_task_ownership() -> dict[str, Any]:
    registry = build_default_robot_registry()
    concrete_names: set[str] = set()
    for profile_id in registry.profile_ids():
        profile = registry.profile(profile_id)
        role_names = set(profile.semantic_roles)
        concrete_names.update(
            str(name)
            for field in ("body_names", "joint_names")
            for name in profile.runtime_facts[field]
            if str(name) not in role_names
        )
    task_root = ROOT / "src/true_rl_agent/tasks/stair_climbing"
    generic_paths = [
        path
        for path in task_root.rglob("*.py")
        if path.name not in _TASK_ROBOT_ADAPTER_PATHS
        and "reference_programs" not in path.relative_to(task_root).parts
    ]
    scanned: list[str] = []
    for path in generic_paths:
        relative_path = path.relative_to(task_root).as_posix()
        source = path.read_text(encoding="utf-8")
        literals = {
            node.value
            for node in ast.walk(ast.parse(source, filename=str(path)))
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        literal_tokens = {
            token
            for literal in literals
            for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", literal)
        }
        hits = concrete_names & literal_tokens
        _require(not hits, f"robot morphology leaked into task runtime: {relative_path}: {sorted(hits)}")
        _require("Unitree/Go2" not in source, f"Go2 USD path leaked into {relative_path}")
        _require("go2.usd" not in source, f"Go2 asset leaked into {relative_path}")
        scanned.append(relative_path)
    for relative_path in ("isaaclab_runtime_env.py", "isaaclab_execution.py"):
        source = (task_root / relative_path).read_text(encoding="utf-8").lower()
        _require("go2" not in source, f"Go2 dispatch leaked into {relative_path}")
    return {
        "generic_task_file_count": len(scanned),
        "robot_profile_count": len(registry.profile_ids()),
        "concrete_name_count": len(concrete_names),
        "status": "pass",
    }


def _verify_curriculum_activation_probe(
    cell: Mapping[str, Any],
) -> dict[str, Any]:
    probe = _load_object(CURRICULUM_ACTIVATION_PROBE)
    _require(
        probe.get("schema_version")
        == (
            "true_rl_agent.variant_construction."
            "phase5_go2_curriculum_activation_probe.v1"
        ),
        "curriculum activation probe schema changed",
    )
    _require(probe.get("status") == "pass", "curriculum activation probe failed")
    _require(
        probe.get("source_commit") == _TRAINING_SOURCE_COMMIT,
        "curriculum activation probe source changed",
    )
    _require(
        probe.get("cell_variant_id") == cell["cell_variant_id"],
        "curriculum activation probe cell changed",
    )
    _require(
        probe.get("manifest_hash") == cell["manifest_hash"],
        "curriculum activation probe manifest changed",
    )
    _require(
        probe.get("training_profile_hash") == cell["training_profile_hash"],
        "curriculum activation probe training profile changed",
    )
    _require(
        probe.get("reference_program_id")
        == "go2-stair-official-flat-derived-v1",
        "curriculum activation probe reference changed",
    )
    _require(
        probe.get("checkpoint_sha256")
        == (
            "sha256:"
            "f2aa77bf0349c10ec2a00071162e37123d0cf80f3447c3989c17a618e1b24ad2"
        ),
        "curriculum activation probe checkpoint changed",
    )
    _require(
        probe.get("num_envs") == 64
        and probe.get("control_steps") == 800
        and probe.get("done_events") == 64,
        "curriculum activation probe workload changed",
    )
    _require(
        probe.get("type_identity_preserved") is True,
        "curriculum activation probe changed terrain type",
    )
    trace = probe.get("curriculum_trace")
    _require(isinstance(trace, Mapping), "curriculum activation trace is absent")
    _require(trace.get("type_changes") == 0, "curriculum changed terrain type")
    _require(
        int(trace.get("up_requests", 0)) > 0
        and int(trace.get("up_requests_c0", 0)) > 0
        and int(trace.get("up_requests_c1", 0)) > 0,
        "capable checkpoint did not activate both curriculum regimes",
    )
    _require(
        float(probe.get("final_level_mean", -1.0))
        > float(probe.get("initial_level_mean", -1.0)),
        "curriculum activation probe did not increase mean level",
    )
    return {
        **_file_evidence(CURRICULUM_ACTIVATION_PROBE),
        "up_requests": int(trace["up_requests"]),
        "up_requests_c0": int(trace["up_requests_c0"]),
        "up_requests_c1": int(trace["up_requests_c1"]),
        "down_requests": int(trace["down_requests"]),
        "status": "pass",
    }


def _verify_runtime_receipts(cell: Mapping[str, Any]) -> dict[str, Any]:
    receipt = _load_object(RUNTIME_RECEIPT)
    oracle = _load_object(ORACLE_STAGE_RECEIPT)
    smoke = _load_object(SMOKE_STAGE_RECEIPT)
    _verify_hashed_payload(receipt, hash_key="receipt_hash", name="Phase 5 runtime receipt")
    _verify_hashed_payload(oracle, hash_key="stage_hash", name="Phase 5 oracle stage")
    _verify_hashed_payload(smoke, hash_key="stage_hash", name="Phase 5 smoke stage")
    _require(receipt.get("status") == "pass", "Phase 5 runtime receipt did not pass")
    _require(oracle.get("status") == "pass" and smoke.get("status") == "pass", "runtime stage failed")
    _require(oracle["stage"] == "oracle" and smoke["stage"] == "smoke", "runtime stage identity changed")
    _require(oracle["cell"] == smoke["cell"] == receipt["cell"], "runtime stages used different cells")
    _require(oracle["environment"] == smoke["environment"] == receipt["environment"], "runtime environments differ")
    _require(
        receipt["stage_receipts"]
        == {
            "oracle_stage_hash": oracle["stage_hash"],
            "smoke_stage_hash": smoke["stage_hash"],
        },
        "runtime stage closure changed",
    )
    for key in (
        "cell_variant_id",
        "manifest_hash",
        "resolved_cell_task_protocol_hash",
        "base_task_protocol_hash",
        "binding_hash",
        "robot_profile_hash",
        "runtime_adapter_hash",
        "training_profile_hash",
        "evaluation_suite_hash",
        "vnext_reward_authority_hash",
        "reference_program_source_checksum",
    ):
        runtime_key = {
            "reward_authority_hash": "vnext_reward_authority_hash",
            "reference_source_checksum": "reference_program_source_checksum",
        }.get(key, key)
        if key in cell:
            _require(receipt["cell"][runtime_key] == cell[key], f"runtime cell identity changed: {key}")

    _require(receipt["product_source_hash_mode"] == "normalized_lf_sha256", "runtime source hash mode changed")
    for relative_path, expected_hash in receipt["product_source_hashes"].items():
        _require(
            isinstance(relative_path, str)
            and bool(relative_path)
            and not Path(relative_path).is_absolute()
            and ".." not in Path(relative_path).parts,
            "runtime source path is invalid",
        )
        _require(
            isinstance(expected_hash, str)
            and len(expected_hash) == 64
            and all(character in "0123456789abcdef" for character in expected_hash),
            f"recorded runtime source hash is invalid: {relative_path}",
        )

    equivalence = receipt["reference_reward_equivalence"]
    _require(equivalence == oracle["reference_reward_equivalence"], "oracle evidence drift")
    _require(equivalence["status"] == "pass", "reward oracle did not pass")
    _require(
        equivalence.get("zero_weight_terms")
        == {"lin_vel_z_l2": 0.0, "dof_pos_limits": 0.0},
        "Go2 task-context zero weights changed",
    )
    _require(
        {
            item["term_name"]: (
                item["upstream_weight"], item["reference_weight"]
            )
            for item in equivalence.get("weight_adaptations", [])
        }
        == {
            "track_lin_vel_xy_exp": (1.5, 3.5),
            "lin_vel_z_l2": (-2.0, 0.0),
        },
        "Go2 task-context reward adaptations changed",
    )
    _require(set(equivalence["maximum_absolute_raw_difference_by_term"]) == set(_EXPECTED_REWARD_WEIGHTS), "reward oracle term set changed")
    _require(max(equivalence["maximum_absolute_raw_difference_by_term"].values()) == 0.0, "raw reward mismatch")
    _require(max(equivalence["maximum_absolute_weighted_difference_by_term"].values()) == 0.0, "weighted reward mismatch")
    _require(equivalence["maximum_absolute_total_rate_difference"] == 0.0, "total reward mismatch")
    _require(equivalence["maximum_absolute_manager_step_difference"] <= 1.0e-6, "manager-step reward mismatch")
    _require(equivalence["control_step_count"] == 40, "reward oracle horizon changed")
    _require(equivalence["sample_count"] == 2880, "reward oracle sample count changed")
    _require(
        equivalence["runtime_schema"]
        == {
            "action_count": 12,
            "body_count": 19,
            "default_root_height_m": 0.4,
            "height_scan_ray_count": 187,
            "joint_count": 12,
            "policy_observation_dimension": 235,
            "support_count": 4,
        },
        "real Go2 runtime schema changed",
    )
    _require(
        {item["terrain_regime_id"] for item in receipt["geometry_probe"]["representatives"]}
        == {"c0_regular", "c1_irregular_flat", "c2_rough"},
        "runtime geometry coverage changed",
    )

    training = receipt["training_smoke"]
    _require(training == smoke["training_smoke"], "training smoke evidence drift")
    _require(training["status"] == "pass" and training["training_executed"], "training smoke did not execute")
    _require(training["training"]["budget_steps"] == 128, "smoke budget changed")
    _require(training["training"]["budget_accounting"]["status"] == "pass", "smoke budget accounting failed")
    _require(
        training["training"]["initial_terrain_assignment"]["regime_counts"]
        == {"c0_regular": 8, "c1_irregular_flat": 8},
        "smoke terrain assignment changed",
    )
    activation = training["runtime_activation_summary"]
    _require(activation["status"] == "pass", "reward runtime activation failed")
    _require(
        activation["declared_term_count"] == len(_EXPECTED_REWARD_WEIGHTS),
        "runtime reward term count changed",
    )
    _require(activation["dispatcher_calls"] > 1, "runtime reward dispatcher did not run")
    _require(activation["nonzero_reward_contribution"] is True, "runtime reward was degenerate")
    _require(receipt["source_preflight"]["summary"]["status"] == "pass", "source preflight failed")
    source_evidence = receipt["source_preflight"]["evidence"]
    for source_file in source_evidence["source_files"]:
        _require(
            isinstance(source_file.get("repository_path"), str)
            and isinstance(source_file.get("sha256"), str)
            and len(source_file["sha256"]) == 64,
            "source-preflight recorded source evidence is invalid",
        )
    robot_sources = source_evidence["robot_runtime_source_evidence"]
    _require(
        robot_sources["runtime_adapter_id"] == GO2_12DOF_ISAACLAB_ADAPTER_ID,
        "source preflight selected the wrong robot runtime",
    )
    _require(
        {
            item["package_relative_path"]
            for item in robot_sources["source_files"]
        }
        == {
            "robots/unitree.py",
            "manager_based/locomotion/velocity/velocity_env_cfg.py",
            "manager_based/locomotion/velocity/config/go2/rough_env_cfg.py",
            "manager_based/locomotion/velocity/config/go2/flat_env_cfg.py",
            "manager_based/locomotion/velocity/config/go2/agents/rsl_rl_ppo_cfg.py",
        },
        "Go2 upstream source closure changed",
    )
    _require(receipt["eligibility"]["candidate_acceptance"] is False, "smoke was misused as acceptance")
    _require(receipt["eligibility"]["reference_qualification"] is False, "smoke was misused as qualification")
    curriculum_activation = _verify_curriculum_activation_probe(cell)
    return {
        "runtime_receipt_hash": receipt["receipt_hash"],
        "oracle_stage_hash": oracle["stage_hash"],
        "smoke_stage_hash": smoke["stage_hash"],
        "reward_term_count": len(_EXPECTED_REWARD_WEIGHTS),
        "oracle_sample_count": equivalence["sample_count"],
        "smoke_budget_steps": training["training"]["budget_steps"],
        "curriculum_activation": curriculum_activation,
        "status": "pass",
    }


def _verify_genesis_load_only(
    *,
    cell: Mapping[str, Any],
    genesis: SessionGenesisArtifact,
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    receipt_path = Path(spec["load_only_receipt"])
    receipt = _load_object(receipt_path)
    _require(
        receipt.get("status") == "pass"
        and receipt.get("mode") == "load-only"
        and receipt.get("session_id") == spec["reference_session_id"]
        and receipt.get("task_id") == "unitree_stair_climbing"
        and receipt.get("cell_variant_id") == cell["cell_variant_id"],
        "Session Genesis load-only identity changed",
    )
    _require(
        receipt.get("provider_call_count") == 0
        and receipt.get("ledger_state") == "absent"
        and receipt.get("ledger_record_count") == 0,
        "Session Genesis load-only crossed the provider or ledger boundary",
    )
    load_only = receipt.get("load_only_evidence")
    _require(isinstance(load_only, Mapping), "Session Genesis load-only evidence is absent")
    _require(
        load_only.get("status") == "pass"
        and load_only.get("replay_status") == "pass"
        and load_only.get("runner_call_count") == 0
        and load_only.get("provider_call_count") == 0
        and load_only.get("genesis_checksum_before")
        == load_only.get("genesis_checksum_after"),
        "Session Genesis load-only replay was not side-effect free",
    )
    _require(
        receipt.get("session_genesis_hash") == genesis.session_genesis_hash,
        "Session Genesis load-only hash changed",
    )
    return {
        **_file_evidence(receipt_path),
        "session_genesis_hash": genesis.session_genesis_hash,
        "runner_call_count": 0,
        "provider_call_count": 0,
        "status": "pass",
    }


def _verify_genesis_import(
    *,
    cell: Mapping[str, Any],
    genesis: SessionGenesisArtifact,
    checkpoint_hash: str,
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    receipt_path = Path(spec["genesis_import_receipt"])
    receipt = _load_object(receipt_path)
    _verify_hashed_payload(
        receipt,
        hash_key="receipt_hash",
        name="Phase 5 successor Genesis import receipt",
    )
    _require(
        receipt.get("status") == "pass"
        and receipt.get("session_id") == spec["reference_session_id"]
        and receipt.get("cell_variant_id") == cell["cell_variant_id"]
        and receipt.get("session_genesis_hash") == genesis.session_genesis_hash,
        "successor Genesis import identity changed",
    )
    _require(
        receipt.get("training_executed") is False
        and receipt.get("evaluation_executed") is False
        and receipt.get("provider_call_count") == 0
        and receipt.get("train_receipt_replay_calls") == 1
        and receipt.get("eval_phase_receipt_replay_calls") == 3
        and receipt.get("unexpected_execution_calls") == 0,
        "successor Genesis import executed new work",
    )
    _require(
        receipt.get("checkpoint_hash") == checkpoint_hash
        and receipt.get("session_genesis_checksum")
        == _file_evidence(Path(spec["session_genesis"]))["sha256"]
        and receipt.get("reference_eval_artifact_checksum")
        == _file_evidence(Path(spec["reference_artifact"]))["sha256"],
        "successor Genesis import byte closure changed",
    )
    return {
        **_file_evidence(receipt_path),
        "training_executed": False,
        "evaluation_executed": False,
        "provider_call_count": 0,
        "status": "pass",
    }


def _verify_v1_counterfactual() -> dict[str, Any]:
    """Verify the same-checkpoint V1 run that isolates completion semantics."""

    capture = _load_object(V1_COUNTERFACTUAL_CAPTURE)
    receipt = _load_object(V1_COUNTERFACTUAL_RECEIPT)
    checkpoint_hash = (
        "sha256:fca6a59c8142d0120c5b534addab03885995dd64931fc3b974b9f94b997fe729"
    )
    _require(
        capture.get("schema_version")
        == "true_rl_agent.task.stair_climbing.checkpoint_capture_result.v6",
        "V1 counterfactual capture schema changed",
    )
    checkpoint_report = capture.get("checkpoint_report")
    source_preflight = capture.get("source_preflight")
    summary = capture.get("rollout_summary")
    _require(
        isinstance(checkpoint_report, Mapping)
        and checkpoint_report.get("status") == "pass"
        and checkpoint_report.get("checkpoint_hash") == checkpoint_hash
        and checkpoint_report.get("expected_checkpoint_hash") == checkpoint_hash,
        "V1 counterfactual checkpoint identity changed",
    )
    _require(
        isinstance(source_preflight, Mapping)
        and source_preflight.get("status") == "pass"
        and source_preflight.get("task_protocol_hash")
        == stair_robot_row(
            STAIR_GO2_12DOF_UNIFIED_V1_CELL_VARIANT_ID
        ).frozen_task_protocol_hash,
        "V1 counterfactual did not use the frozen predecessor semantics",
    )
    _require(
        isinstance(summary, Mapping)
        and summary.get("scope") == "one_unified_public_dev_run_not_acceptance_score"
        and summary.get("split") == "public_dev"
        and summary.get("episode_count") == 72
        and summary.get("episodes_per_cell") == 3
        and summary.get("macro_success_rate") == 0.0
        and summary.get("primary_outcome_counts")
        == {"fall": 22, "forbidden_contact": 50}
        and sum(summary["primary_outcome_counts"].values()) == 72
        and summary.get("outcome_source")
        == "task_owned_formal_episode_state_machine"
        and summary.get("reclassified_by_video_exporter") is False,
        "V1 counterfactual 72-episode evidence changed",
    )
    _require(
        receipt.get("status") == "pass"
        and receipt.get("checkpoint_hash") == checkpoint_hash
        and receipt.get("tracked_outcomes") == capture.get("tracked_outcomes")
        and receipt.get("render", {}).get("physics_recomputed") is False,
        "V1 counterfactual visualization receipt changed",
    )
    tracked = receipt["tracked_outcomes"]
    _require(
        isinstance(tracked, list)
        and len(tracked) == 3
        and all(item.get("progress_fraction") == 1.0 for item in tracked)
        and all(item.get("success") is False for item in tracked)
        and {item.get("primary_reason") for item in tracked}
        == {"fall", "forbidden_contact"},
        "V1 counterfactual tracked arrival/failure evidence changed",
    )
    trajectory_path = V1_COUNTERFACTUAL_DIR / str(capture.get("trajectory_filename"))
    video_path = V1_COUNTERFACTUAL_DIR / str(receipt.get("video_filename"))
    _require(
        "sha256:" + hashlib.sha256(trajectory_path.read_bytes()).hexdigest()
        == receipt.get("trajectory_hash"),
        "V1 counterfactual trajectory bytes changed",
    )
    _require(
        "sha256:" + hashlib.sha256(video_path.read_bytes()).hexdigest()
        == receipt.get("video_hash"),
        "V1 counterfactual video bytes changed",
    )
    return {
        "status": "pass",
        "checkpoint_hash": checkpoint_hash,
        "task_protocol_hash": source_preflight["task_protocol_hash"],
        "episode_count": 72,
        "macro_success_rate": 0.0,
        "primary_outcome_counts": dict(summary["primary_outcome_counts"]),
        "tracked_full_progress_failure_count": 3,
        "capture": _file_evidence(V1_COUNTERFACTUAL_CAPTURE),
        "visualization": _file_evidence(V1_COUNTERFACTUAL_RECEIPT),
    }


def _load_reference_phase_contract(
    cell: Mapping[str, Any],
    spec: Mapping[str, Any],
) -> tuple[str, str, dict[str, Any]]:
    contract_path = Path(spec["phase_contract"])
    envelope = _load_object(contract_path)
    _require(
        set(envelope) == {"schema_version", "contract_hash", "contract"},
        "reference phase contract envelope fields changed",
    )
    _require(
        envelope.get("schema_version") == _TRIAL_EXECUTION_CONTRACT_SCHEMA_VERSION,
        "reference phase contract schema changed",
    )
    contract = envelope.get("contract")
    _require(isinstance(contract, Mapping), "reference phase contract is absent")
    contract_hash = canonical_json_hash(contract)
    _require(envelope.get("contract_hash") == contract_hash, "reference phase contract hash mismatch")
    _require(
        contract.get("round_id") == spec["reference_session_id"],
        "reference phase round changed",
    )
    _require(
        contract.get("purpose") == "session_genesis_reference_phased_evaluation",
        "reference phase purpose changed",
    )
    commitment = contract.get("commitment")
    _require(isinstance(commitment, Mapping), "reference phase commitment is absent")
    _require(
        commitment.get("session_id") == spec["reference_session_id"]
        and commitment.get("reference_manifest_hash")
        == cell["reference_manifest_hash"],
        "reference phase commitment identity changed",
    )
    evaluation_plan_hash = commitment.get("evaluation_plan_hash")
    _require(
        isinstance(evaluation_plan_hash, str)
        and re.fullmatch(r"[0-9a-f]{64}", evaluation_plan_hash) is not None,
        "reference phase evaluation plan hash is invalid",
    )
    return (
        contract_hash,
        evaluation_plan_hash,
        _file_evidence(contract_path),
    )


def _load_reference_phase_evidence(
    *,
    cell: Mapping[str, Any],
    config: StairClimbingTaskConfig,
    contract_hash: str,
    evaluation_plan_hash: str,
    spec: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], str, int]:
    summaries: dict[str, Any] = {}
    records: dict[str, dict[str, Any]] = {}
    checkpoint_hashes: set[str] = set()
    evaluation_plan_hashes: set[str] = set()
    evaluation_binding_hashes: set[str] = set()
    environment_source_manifest_hashes: set[str] = set()
    backend_environment_manifest_hashes: set[str] = set()
    total_episode_count = 0
    for phase, raw_path in spec["phase_receipts"].items():
        path = Path(raw_path)
        receipt = _load_object(path)
        _require(
            set(receipt)
            == {
                "schema_version",
                "contract_hash",
                "job_spec_hash",
                "phase_result",
                "phase_result_hash",
            },
            f"{phase.value} receipt fields changed",
        )
        _require(
            receipt.get("schema_version") == _EVALUATION_PHASE_RECEIPT_SCHEMA_VERSION,
            f"{phase.value} receipt schema changed",
        )
        _require(receipt.get("contract_hash") == contract_hash, f"{phase.value} contract changed")
        result_payload = receipt.get("phase_result")
        _require(
            canonical_json_hash(result_payload) == receipt.get("phase_result_hash"),
            f"{phase.value} result hash mismatch",
        )
        result = EvaluationPhaseResult.from_private_mapping(result_payload)
        _require(result.phase is phase, f"{phase.value} result phase changed")
        _require(result.task_id == "unitree_stair_climbing", f"{phase.value} task changed")
        _require(
            result.subject.kind == "reference"
            and result.subject.subject_id == spec["reference_subject_id"],
            f"{phase.value} subject changed",
        )
        expected_identity = {
            "subject_manifest_hash": cell["reference_manifest_hash"],
            "program_source_checksum": cell["reference_source_checksum"],
            "training_binding_hash": cell["training_binding_hash"],
            "capability_profile_hash": cell["reference_capability_profile_hash"],
            "eval_protocol_hash": cell["evaluation_protocol_hash"],
            "execution_device": spec["evaluation_device"],
        }
        for name, value in expected_identity.items():
            _require(getattr(result, name) == value, f"{phase.value} identity changed: {name}")
        _require(
            result.environment_source_preflight.get("status") == "pass"
            and result.environment_source_preflight.get("task_protocol_hash")
            == cell["resolved_cell_task_protocol_hash"],
            f"{phase.value} source preflight failed",
        )
        _require(
            result.backend_execution_environment_verification.get("status") == "pass",
            f"{phase.value} backend environment verification failed",
        )
        safety = result.checkpoint_safety_report
        _require(
            safety.get("status") == "pass"
            and safety.get("checkpoint_hash") == result.checkpoint_hash
            and safety.get("expected_checkpoint_hash") == result.checkpoint_hash,
            f"{phase.value} checkpoint safety failed",
        )
        summary = canonical_stair_phase_result_summary(
            config=config,
            expected_phase=phase,
            result=result,
        )
        expected_episode_count = (
            len(config.ordered_evaluation_suite.cells_for_cohort(phase.value))
            * config.ordered_evaluation_suite.seed_count(phase.value)
            * config.ordered_evaluation_suite.episodes_per_seed_per_cell
        )
        observed_episode_count = int(summary["ordered_summary"]["total_episode_count"])
        _require(
            observed_episode_count == expected_episode_count,
            f"{phase.value} episode count changed",
        )
        total_episode_count += observed_episode_count
        summaries[phase.value] = summary
        checkpoint_hashes.add(result.checkpoint_hash)
        evaluation_plan_hashes.add(result.evaluation_plan_hash)
        evaluation_binding_hashes.add(result.evaluation_binding_hash)
        environment_source_manifest_hashes.add(
            str(result.environment_source_preflight["source_manifest_hash"])
        )
        backend_environment_manifest_hashes.add(
            str(
                result.backend_execution_environment_verification[
                    "actual_manifest_hash"
                ]
            )
        )
        records[phase.value] = {
            **_file_evidence(path),
            "job_id": result.job_id,
            "job_spec_hash": receipt["job_spec_hash"],
            "phase_result_hash": receipt["phase_result_hash"],
            "task_evidence_hash": result.task_evidence_hash,
            "episode_count": observed_episode_count,
            "environment_source_manifest_hash": (
                result.environment_source_preflight["source_manifest_hash"]
            ),
            "backend_environment_manifest_hash": (
                result.backend_execution_environment_verification[
                    "actual_manifest_hash"
                ]
            ),
        }
    _require(len(checkpoint_hashes) == 1, "reference phases used different checkpoints")
    _require(
        evaluation_plan_hashes == {evaluation_plan_hash},
        "reference phases differ from their committed eval plan",
    )
    _require(len(evaluation_binding_hashes) == 1, "reference phases used different bindings")
    _require(
        len(environment_source_manifest_hashes) == 1,
        "reference phases used different task source manifests",
    )
    _require(
        len(backend_environment_manifest_hashes) == 1,
        "reference phases used different backend manifests",
    )
    _require(total_episode_count == cell["formal_episode_count"], "reference eval cost changed")
    return summaries, records, checkpoint_hashes.pop(), total_episode_count


def _load_reference_training_evidence(
    *,
    cell: Mapping[str, Any],
    checkpoint_hash: str,
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    train_summary_path = Path(spec["train_summary"])
    summary = _load_object(train_summary_path)
    _require(summary.get("status") == "pass", "formal reference training failed")
    _require(summary.get("errors") == [], "formal reference training reported errors")
    _require(
        summary.get("job_id") == f"{spec['training_session_id']}-train",
        "formal reference training job changed",
    )
    checkpoint = summary.get("checkpoint")
    training = summary.get("training")
    _require(isinstance(checkpoint, Mapping), "formal training checkpoint is absent")
    _require(isinstance(training, Mapping), "formal training accounting is absent")
    _require(checkpoint.get("checksum") == checkpoint_hash, "train/eval checkpoint mismatch")
    expected_training = {
        "num_envs": 4096,
        "budget_steps": _REFERENCE_BUDGET_STEPS,
        "num_steps_per_env": 24,
        "max_iterations": 2000,
        "device": spec["training_device"],
    }
    for name, value in expected_training.items():
        _require(training.get(name) == value, f"formal training drift: {name}")
    effective_agent = training.get("effective_agent_execution")
    _require(
        isinstance(effective_agent, Mapping),
        "formal training effective agent execution is absent",
    )
    expected_agent = {
        "agent_configuration_hash": cell["agent_configuration_hash"],
        "seed": spec["training_seed"],
        "num_steps_per_env": 24,
        "max_iterations": 2000,
        "device": spec["training_device"],
        "resume": False,
    }
    for name, value in expected_agent.items():
        _require(
            effective_agent.get(name) == value,
            f"formal effective agent drift: {name}",
        )
    _require(
        training.get("expected_initial_policy_state_hash") is None
        and training.get("initial_policy_state_match") is None,
        "formal reference unexpectedly used a supplied initial policy",
    )
    initial_policy_state_hash = training.get("initial_policy_state_hash")
    _require(
        isinstance(initial_policy_state_hash, str)
        and re.fullmatch(r"[0-9a-f]{64}", initial_policy_state_hash) is not None,
        "formal reference initial policy state hash is invalid",
    )
    _require(
        training.get("budget_accounting")
        == {
            "status": "pass",
            "budget_steps": _REFERENCE_BUDGET_STEPS,
            "num_envs": 4096,
            "num_steps_per_env": 24,
            "max_iterations": 2000,
            "planned_env_steps": _REFERENCE_BUDGET_STEPS,
            "config_derived_total_timesteps": _REFERENCE_BUDGET_STEPS,
        },
        "formal reference budget accounting changed",
    )
    training_distribution = training.get("training_distribution")
    _require(
        isinstance(training_distribution, Mapping)
        and training_distribution.get("training_profile_hash")
        == cell["training_profile_hash"]
        and training_distribution.get("status") == "unified_c0_c1"
        and training_distribution.get("training_exposure", {}).get(
            "forbidden_regimes"
        )
        == ["c2_rough"],
        "formal reference training distribution changed",
    )
    activation = summary.get("runtime_activation_summary")
    _require(isinstance(activation, Mapping), "formal reference activation is absent")
    _require(
        activation.get("status") == "pass"
        and activation.get("declared_term_count") == spec["reward_term_count"]
        and activation.get("nonzero_reward_contribution") is True,
        "formal reference reward program was not active",
    )
    return {
        **_file_evidence(train_summary_path),
        "checkpoint_hash": checkpoint_hash,
        "budget_steps": training["budget_steps"],
        "max_iterations": training["max_iterations"],
        "latest_checkpoint_iteration": training["latest_checkpoint_iteration"],
        "runtime_activation_summary_hash": canonical_json_hash(activation),
    }


def _verify_reference_product_outputs(
    *,
    cell: Mapping[str, Any],
    checkpoint_hash: str,
    phase_records: Mapping[str, Mapping[str, Any]],
    qualification_gate: Mapping[str, Any],
    qualification_evidence: Mapping[str, Any],
    spec: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    plain_qualification_gate = _plain_value(qualification_gate)
    plain_qualification_evidence = _plain_value(qualification_evidence)
    _require(
        Path(spec["reference_artifact"]).is_file()
        and Path(spec["session_genesis"]).is_file(),
        "qualified reference lacks Genesis outputs",
    )
    reference_artifact_path = Path(spec["reference_artifact"])
    session_genesis_path = Path(spec["session_genesis"])
    artifact = _load_object(reference_artifact_path)
    _require(
        artifact.get("session_id") == spec["reference_session_id"]
        and artifact.get("task_id") == "unitree_stair_climbing",
        "reference eval artifact session changed",
    )
    reference_manifest = artifact.get("reference_manifest")
    training_binding = artifact.get("training_binding")
    _require(isinstance(reference_manifest, Mapping), "reference artifact manifest is absent")
    _require(isinstance(training_binding, Mapping), "reference artifact binding is absent")
    _require(
        reference_manifest.get("manifest_hash") == cell["reference_manifest_hash"]
        and reference_manifest.get("source_checksum") == cell["reference_source_checksum"]
        and training_binding.get("binding_hash") == cell["training_binding_hash"],
        "reference eval artifact lineage changed",
    )
    _require(artifact.get("checkpoint_hash") == checkpoint_hash, "reference artifact checkpoint changed")
    _require(
        artifact.get("qualification_gate") == plain_qualification_gate,
        "reference artifact gate changed",
    )
    _require(
        artifact.get("qualification_evidence") == plain_qualification_evidence,
        "reference artifact qualification evidence changed",
    )
    program_source = artifact.get("program_source")
    _require(isinstance(program_source, str), "reference artifact source is absent")
    _require(
        "sha256:" + hashlib.sha256(program_source.encode("utf-8")).hexdigest()
        == cell["reference_source_checksum"],
        "reference artifact source bytes changed",
    )
    assembly = artifact.get("evaluation_assembly")
    _require(isinstance(assembly, Mapping), "phased reference assembly is absent")
    _require(
        assembly.get("assembly_hash")
        == canonical_json_hash(_without(assembly, "assembly_hash")),
        "reference evaluation assembly hash mismatch",
    )
    raw_refs = assembly.get("phase_receipts")
    _require(isinstance(raw_refs, list) and len(raw_refs) == 3, "reference assembly phases changed")
    refs = {item.get("phase"): item for item in raw_refs if isinstance(item, Mapping)}
    _require(set(refs) == set(phase_records), "reference assembly phase set changed")
    for phase, record in phase_records.items():
        ref = refs[phase]
        _require(
            ref.get("checksum") == record["sha256"]
            and ref.get("job_spec_hash") == record["job_spec_hash"]
            and ref.get("phase_result_hash") == record["phase_result_hash"],
            f"reference assembly receipt changed: {phase}",
        )

    genesis_payload = _load_object(session_genesis_path)
    genesis = SessionGenesisArtifact.from_mapping(genesis_payload)
    _require(
        genesis.session_id == spec["reference_session_id"]
        and genesis.task_id == "unitree_stair_climbing"
        and genesis.policy_initialization_kind == "random",
        "reference Session Genesis identity changed",
    )
    anchor = genesis.anchor
    _require(
        isinstance(anchor, ReferenceProgramGenesisAnchor),
        "reference Session Genesis anchor kind changed",
    )
    _require(
        anchor.reference_manifest.manifest_hash == cell["reference_manifest_hash"]
        and anchor.training_binding.binding_hash == cell["training_binding_hash"]
        and anchor.checkpoint_hash == checkpoint_hash
        and _plain_value(anchor.qualification_gate) == plain_qualification_gate,
        "reference Session Genesis lineage changed",
    )
    artifact_file = _file_evidence(reference_artifact_path)
    _require(
        anchor.private_eval_artifact.checksum == artifact_file["sha256"],
        "Session Genesis reference artifact checksum changed",
    )
    load_only_evidence = _verify_genesis_load_only(
        cell=cell,
        genesis=genesis,
        spec=spec,
    )
    genesis_import_evidence = _verify_genesis_import(
        cell=cell,
        genesis=genesis,
        checkpoint_hash=checkpoint_hash,
        spec=spec,
    )
    return {
        reference_artifact_path.name: artifact_file,
        session_genesis_path.name: _file_evidence(session_genesis_path),
        Path(spec["load_only_receipt"]).name: load_only_evidence,
        Path(spec["genesis_import_receipt"]).name: genesis_import_evidence,
    }


def _verify_qualification_receipt(
    cell: Mapping[str, Any],
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    receipt = _load_object(Path(spec["receipt"]))
    _require(receipt.get("schema_version") == QUALIFICATION_SCHEMA_VERSION, "qualification schema changed")
    _verify_hashed_payload(receipt, hash_key="receipt_hash", name="Phase 5 qualification receipt")
    _require(
        receipt.get("status") == "go"
        and receipt.get("cell_loop_ready") is True
        and receipt.get("training_executed") is False
        and receipt.get("training_reused") is True
        and receipt.get("provider_call_count") == 0,
        "first-arrival requalification did not close as a no-training GO",
    )
    _require(
        receipt.get("phase_execution_source_commit")
        == spec["phase_execution_source_commit"]
        and receipt.get("qualification_source_commit")
        == spec["qualification_source_commit"],
        "first-arrival requalification provenance changed",
    )

    training_cell = cell["training_predecessor"]
    compatibility = receipt.get("training_compatibility")
    _require(isinstance(compatibility, Mapping), "training compatibility evidence is absent")
    _verify_hashed_payload(
        compatibility,
        hash_key="compatibility_hash",
        name="evaluation-only successor compatibility",
    )
    _require(
        compatibility.get("status") == "pass"
        and compatibility.get("scope")
        == "training_semantics_unchanged_evaluation_semantics_changed",
        "checkpoint reuse compatibility did not pass",
    )
    source_cell = compatibility.get("source_training_cell")
    target_cell = compatibility.get("target_evaluation_cell")
    _require(
        isinstance(source_cell, Mapping)
        and source_cell.get("cell_variant_id") == training_cell["cell_variant_id"]
        and source_cell.get("resolved_cell_manifest_hash")
        == training_cell["manifest_hash"]
        and source_cell.get("resolved_cell_task_protocol_hash")
        == training_cell["resolved_cell_task_protocol_hash"]
        and source_cell.get("training_binding_hash")
        == training_cell["training_binding_hash"],
        "requalification source training lineage changed",
    )
    _require(
        isinstance(target_cell, Mapping)
        and target_cell.get("cell_variant_id") == cell["cell_variant_id"]
        and target_cell.get("resolved_cell_manifest_hash") == cell["manifest_hash"]
        and target_cell.get("resolved_cell_task_protocol_hash")
        == cell["resolved_cell_task_protocol_hash"]
        and target_cell.get("formal_training_binding_hash")
        == cell["training_binding_hash"],
        "requalification target evaluation lineage changed",
    )

    config, _plugin, _manifest, _projection = _resolve_config(Path(spec["protocol"]))
    contract_hash, evaluation_plan_hash, _contract_evidence = (
        _load_reference_phase_contract(cell, spec)
    )
    summaries, phase_records, checkpoint_hash, total_episodes = (
        _load_reference_phase_evidence(
            cell=cell,
            config=config,
            contract_hash=contract_hash,
            evaluation_plan_hash=evaluation_plan_hash,
            spec=spec,
        )
    )
    training_evidence = _load_reference_training_evidence(
        cell=training_cell,
        checkpoint_hash=checkpoint_hash,
        spec=spec,
    )
    evaluation = receipt.get("evaluation")
    _require(isinstance(evaluation, Mapping), "requalification evaluation evidence is absent")
    expected_evaluation_identity = {
        "cell_variant_id": cell["cell_variant_id"],
        "resolved_cell_manifest_hash": cell["manifest_hash"],
        "task_protocol_hash": cell["resolved_cell_task_protocol_hash"],
        "training_binding_hash": cell["training_binding_hash"],
        "evaluation_plan_hash": evaluation_plan_hash,
        "checkpoint_hash": checkpoint_hash,
    }
    for name, value in expected_evaluation_identity.items():
        _require(evaluation.get(name) == value, f"requalification evaluation drift: {name}")
    receipt_assembly = evaluation.get("assembly")
    _require(isinstance(receipt_assembly, Mapping), "requalification assembly is absent")
    _require(
        receipt_assembly.get("summary_hash")
        == canonical_json_hash(receipt_assembly.get("summary")),
        "requalification summary hash mismatch",
    )
    receipt_phase_refs = {
        item.get("phase"): item
        for item in receipt_assembly.get("phase_receipts", [])
        if isinstance(item, Mapping)
    }
    _require(set(receipt_phase_refs) == set(phase_records), "requalification phase set changed")
    for phase, phase_record in phase_records.items():
        _require(
            receipt_phase_refs[phase].get("job_spec_hash")
            == phase_record["job_spec_hash"]
            and receipt_phase_refs[phase].get("phase_result_hash")
            == phase_record["phase_result_hash"],
            f"requalification phase identity changed: {phase}",
        )

    public_accept = summaries[EvaluationPhase.PUBLIC_ACCEPT.value]
    objective_family_mask = build_reference_objective_family_mask(
        suite=config.ordered_evaluation_suite,
        public_accept_summary=public_accept,
    )
    qualification = qualify_stair_reference_public_result(
        config=config,
        public_result=PublicEvalResult(
            status="pass",
            public_score=float(public_accept["j_primary"]),
            metrics={
                "public_accept": public_accept,
                "objective_family_mask": objective_family_mask,
                "acceptance": {
                    "required_pass_gates": list(
                        config.unified_protocol.required_pass_gates
                    )
                },
            },
        ),
    )
    _require(qualification.passed, "replayed first-arrival qualification is NO-GO")
    _require(
        receipt.get("qualification_gate") == _plain_value(qualification.gate)
        and receipt.get("qualification_evidence")
        == _plain_value(qualification.private_evidence),
        "stored qualification differs from raw phase replay",
    )
    product_outputs = _verify_reference_product_outputs(
        cell=cell,
        checkpoint_hash=checkpoint_hash,
        phase_records=phase_records,
        qualification_gate=qualification.gate,
        qualification_evidence=qualification.private_evidence,
        spec=spec,
    )
    _require(len(product_outputs) == 4, "qualified reference closure is incomplete")
    return {
        "status": "go",
        "receipt_hash": receipt["receipt_hash"],
        "go": True,
        "cell_loop_ready": receipt["cell_loop_ready"],
        "checkpoint_hash": checkpoint_hash,
        "total_episode_count": total_episodes,
        "formal_training": training_evidence,
        "product_outputs": product_outputs,
        "public_accept_j_primary": float(public_accept["j_primary"]),
        "sealed_validation_j_primary": float(
            summaries[EvaluationPhase.SEALED_VALIDATION.value]["j_primary"]
        ),
    }


def verify(
    *,
    foundation_only: bool = False,
    require_go: bool = False,
) -> dict[str, Any]:
    _require(
        not (foundation_only and require_go),
        "foundation-only verification cannot require qualification GO",
    )
    cell = _verify_cell_and_distribution()
    ownership = _verify_task_ownership()
    runtime = _verify_runtime_receipts(cell["training_predecessor"])
    v1_counterfactual = _verify_v1_counterfactual()
    go2_spec = _go2_qualification_spec()
    g1_spec = _g1_qualification_spec()
    qualification = (
        {"status": "not_checked", "reason": "foundation_only"}
        if foundation_only
        else _verify_qualification_receipt(cell, go2_spec)
    )
    g1_v2_qualification = (
        {"status": "not_checked", "reason": "foundation_only"}
        if foundation_only
        else _verify_qualification_receipt(cell["g1_v2"], g1_spec)
    )
    if require_go:
        _require(
            qualification.get("go") is True
            and g1_v2_qualification.get("go") is True,
            "Phase 5 row qualification is NO-GO",
        )
    return {
        "schema_version": VERIFY_SCHEMA_VERSION,
        "status": "pass",
        "cell": cell,
        "ownership": ownership,
        "runtime": runtime,
        "v1_counterfactual": v1_counterfactual,
        "genesis_load_only": (
            {"status": "not_checked", "reason": "foundation_only"}
            if foundation_only
            else qualification["product_outputs"][
                QUALIFICATION_LOAD_ONLY_RECEIPT.name
            ]
        ),
        "qualification": qualification,
        "g1_v2_qualification": g1_v2_qualification,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify Phase 5 Go2 Stair evidence")
    parser.add_argument("--foundation-only", action="store_true")
    parser.add_argument("--require-go", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            verify(
                foundation_only=args.foundation_only,
                require_go=args.require_go,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
