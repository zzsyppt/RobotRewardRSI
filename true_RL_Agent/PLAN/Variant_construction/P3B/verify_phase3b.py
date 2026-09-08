from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.tasks.stair_climbing.config import StairClimbingTaskConfig
from true_rl_agent.tasks.stair_climbing.plugin import StairClimbingTaskPlugin
from true_rl_agent.tasks.stair_climbing.robot_binding import (
    build_stair_cell_catalog,
    resolve_stair_cell_manifest,
)
from true_rl_agent.tasks.stair_climbing.unified_training import (
    STAIR_TERRAIN_REGIME_C2,
)

SNAPSHOT_SCHEMA_VERSION = "true_rl_agent.variant_construction.phase3b_snapshot.v1"
VERIFY_SCHEMA_VERSION = "true_rl_agent.variant_construction.phase3b_verify.v1"
RUNTIME_SCHEMA_VERSION = "true_rl_agent.variant_construction.phase3b_runtime.v1"
TRAINING_SCHEMA_VERSION = (
    "true_rl_agent.variant_construction.phase3r_runtime_smoke.v1"
)

P3B_DIR = ROOT / "PLAN/Variant_construction/P3B"
SNAPSHOT_PATH = P3B_DIR / "PHASE3B_CONTRACT_SNAPSHOT.json"
RUNTIME_RECEIPT_PATH = P3B_DIR / "PHASE3B_RUNTIME_RECEIPT.json"
TRAINING_RECEIPT_PATH = (
    P3B_DIR / "PHASE3B_TRAINING_EXCLUSION_RUNTIME_RECEIPT.json"
)
PROTOCOL_PATH = ROOT / "configs/tasks/stair_unified_protocol_v1.json"

COHORTS = ("public_dev", "public_accept", "sealed_validation")
EXPECTED_STRATA = {
    "c0_id": 3,
    "c1_id": 3,
    "height_ood": 3,
    "tread_ood": 3,
    "joint_ood": 3,
    "phase_composition_ood": 3,
    "c1_conditional_frontier": 3,
    "c2_mechanism_ood": 3,
}
PRODUCT_PATHS = (
    "configs/tasks/stair_unified_protocol_v1.json",
    "configs/tasks/stair_unified_training_profile_v1.json",
    "configs/tasks/stair_unified_evaluation_suite_v1.json",
    "configs/tasks/stair_unified_acceptance_protocol_v6.json",
    "configs/tasks/stair_vnext_reward_authority_v1.json",
    "src/true_rl_agent/tasks/stair_climbing/ordered_evaluation.py",
    "src/true_rl_agent/tasks/stair_climbing/unified_evaluation.py",
    "src/true_rl_agent/tasks/stair_climbing/unified_training.py",
    "src/true_rl_agent/tasks/stair_climbing/rough_surface.py",
    "src/true_rl_agent/tasks/stair_climbing/isaaclab_rough_terrain.py",
    "src/true_rl_agent/tasks/stair_climbing/isaaclab_evaluation_terrain.py",
    "src/true_rl_agent/tasks/stair_climbing/isaaclab_execution.py",
    "src/true_rl_agent/tasks/stair_climbing/isaaclab_runtime_env.py",
    "src/true_rl_agent/tasks/stair_climbing/terrain_evidence.py",
    "src/true_rl_agent/tasks/stair_climbing/transition_decoder.py",
    "src/true_rl_agent/tasks/stair_climbing/unified_protocol.py",
    "src/true_rl_agent/tasks/stair_climbing/unified_scoring.py",
    "src/true_rl_agent/tasks/stair_climbing/prompt_adapter.py",
)
ROUGHNESS_MARKERS = (
    "c2_rough",
    "rough_surface",
    "bounded_piecewise_linear_v1",
    "roughness_band",
    "surface_pattern",
)
def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"expected JSON object: {path}")
    return value


def _without_hash(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != field}


def _normalized_text_hash(path: Path) -> str:
    data = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(data).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _resolve_cell() -> tuple[Any, Any, Any]:
    provisional = StairClimbingTaskConfig.defaults(PROTOCOL_PATH)
    provisional_plugin = StairClimbingTaskPlugin(provisional)
    manifest = resolve_stair_cell_manifest(
        cell_variant_id=provisional.cell_variant_id,
        config=provisional,
        plugin=provisional_plugin,
    )
    config = StairClimbingTaskConfig.defaults(
        PROTOCOL_PATH,
        resolved_cell_task_protocol_hash=manifest.resolved_cell_task_protocol_hash,
    )
    plugin = StairClimbingTaskPlugin(config, resolved_cell_manifest=manifest)
    repeated = resolve_stair_cell_manifest(
        cell_variant_id=config.cell_variant_id,
        config=config,
        plugin=plugin,
    )
    _require(repeated.to_dict() == manifest.to_dict(), "cell manifest is not stable")
    return config, manifest, plugin


def _source_hashes_from_receipt(
    receipt: Mapping[str, Any],
) -> dict[str, str]:
    _require(
        receipt.get("product_source_hash_mode") == "normalized_lf_sha256",
        "runtime receipt source hash mode mismatch",
    )
    hashes = receipt.get("product_source_hashes")
    _require(isinstance(hashes, Mapping) and bool(hashes), "missing source hashes")
    for relative_path, expected in hashes.items():
        _require(
            isinstance(relative_path, str)
            and bool(relative_path)
            and not Path(relative_path).is_absolute()
            and ".." not in Path(relative_path).parts,
            "invalid source path",
        )
        _require(
            isinstance(expected, str)
            and len(expected) == 64
            and all(character in "0123456789abcdef" for character in expected),
            f"invalid recorded source hash: {relative_path}",
        )
    return {str(key): str(value) for key, value in hashes.items()}


def _validate_receipt(path: Path, schema_version: str) -> dict[str, Any]:
    receipt = _load_object(path)
    _require(receipt.get("schema_version") == schema_version, "receipt schema mismatch")
    _require(receipt.get("status") == "pass", "runtime receipt did not pass")
    expected_hash = canonical_json_hash(_without_hash(receipt, "receipt_hash"))
    _require(receipt.get("receipt_hash") == expected_hash, "receipt hash mismatch")
    return receipt


def _runtime_receipt(config: Any, manifest: Any) -> dict[str, Any]:
    receipt = _validate_receipt(RUNTIME_RECEIPT_PATH, RUNTIME_SCHEMA_VERSION)
    sources = _source_hashes_from_receipt(receipt)
    cell = receipt["cell"]
    _require(cell["protocol_state"] == "active", "C2 receipt is not active")
    _require(
        cell["cell_variant_id"] == config.cell_variant_id,
        "runtime receipt belongs to another robot-task cell",
    )
    _require(
        cell["training_profile_hash"]
        == config.unified_training_profile.profile_hash,
        "runtime receipt training distribution changed",
    )

    geometry = receipt["geometry_probe"]
    _require(geometry["status"] == "pass", "C2 geometry probe failed")
    _require(geometry["cell_count"] == 9, "C2 geometry cell count mismatch")
    _require(
        geometry["mesh_query_max_abs_error_m"] <= 1.0e-9,
        "mesh/query agreement is not exact enough",
    )
    _require(
        geometry["observed_max_abs_offset_m"] <= 0.008000001,
        "observed C2 offset exceeds the declared suite",
    )
    _require(
        geometry["hard_max_abs_offset_m"] == 0.015,
        "C2 hard amplitude bound changed",
    )
    _require(
        all(row["all_meshes_watertight"] for row in geometry["cells"]),
        "a C2 mesh is not watertight",
    )
    current_c2_geometry = {
        candidate.geometry_digest
        for cohort in COHORTS
        for candidate in config.ordered_evaluation_suite.cells_for_cohort(cohort)
        if candidate.terrain_regime_id == STAIR_TERRAIN_REGIME_C2
    }
    _require(
        {row["geometry_digest"] for row in geometry["cells"]}
        == current_c2_geometry,
        "runtime C2 geometry differs from the current suite",
    )

    scanner = receipt["height_scanner_and_decoder_probe"]
    visibility = scanner["height_scanner"]
    _require(
        visibility["visible_ray_count"] >= visibility["required_visible_ray_count"],
        "height scanner cannot resolve enough C2 rays",
    )
    _require(
        visibility["maximum_abs_hit_height_delta_m"]
        >= visibility["required_maximum_delta_m"],
        "height scanner C2 signal is too small",
    )
    _require(scanner["decoder"]["reward_finite"] is True, "decoder reward is non-finite")

    failure = receipt["failure_feedback_calibration"]
    _require(failure["status"] == "pass", "failure calibration failed")
    _require(
        failure["surface_relative_exact_parity"] is True,
        "surface-relative failure semantics drifted",
    )
    _require(failure["fixed_world_flip_count"] > 0, "C2 calibration has no sensitivity")
    _require(
        failure["out_of_amplitude_margin_flip_count"] == 0,
        "failure flips escaped the roughness amplitude band",
    )
    _require(
        failure["maximum_clearance_delta_m"] <= failure["surface_amplitude_m"],
        "failure calibration exceeds the declared surface amplitude",
    )
    cost = receipt["cost"]
    _require(cost["formal_episode_count"] == 1080, "formal C2 cost changed")
    _require(
        cost["formal_episode_count"] <= cost["p3r_formal_episode_cap"],
        "formal evaluation exceeds the P3R cap",
    )
    return {
        "receipt_hash": receipt["receipt_hash"],
        "source_closure_hash": canonical_json_hash(sources),
        "source_file_count": len(sources),
        "geometry_cell_count": geometry["cell_count"],
        "mesh_query_max_abs_error_m": geometry["mesh_query_max_abs_error_m"],
        "watertight_cell_count": sum(
            bool(row["all_meshes_watertight"]) for row in geometry["cells"]
        ),
        "visible_ray_count": visibility["visible_ray_count"],
        "maximum_ray_delta_m": visibility["maximum_abs_hit_height_delta_m"],
        "failure_flip_count": failure["fixed_world_flip_count"],
        "out_of_margin_flip_count": failure["out_of_amplitude_margin_flip_count"],
        "formal_episode_count": cost["formal_episode_count"],
        "cuda_device_name": receipt["environment"]["cuda_device_name"],
        "contract_transition": {
            "reason": "phase4_pre_reference_scoring_rule_correction",
            "rollout_geometry_or_training_changed": False,
            "historical_manifest_hash": cell["manifest_hash"],
            "current_manifest_hash": manifest.manifest_hash,
            "historical_base_task_protocol_hash": cell[
                "base_task_protocol_hash"
            ],
            "current_base_task_protocol_hash": config.base_task_protocol_hash,
            "historical_evaluation_suite_hash": cell["evaluation_suite_hash"],
            "current_evaluation_suite_hash": (
                config.ordered_evaluation_suite.suite_hash
            ),
        },
    }


def _training_exclusion_receipt(config: Any) -> dict[str, Any]:
    receipt = _validate_receipt(TRAINING_RECEIPT_PATH, TRAINING_SCHEMA_VERSION)
    sources = _source_hashes_from_receipt(receipt)
    _require(
        receipt["cell"]["protocol_state"] == "draft",
        "training exclusion receipt must precede activation",
    )
    _require(
        receipt["cell"]["cell_variant_id"] == config.cell_variant_id,
        "training exclusion receipt belongs to another robot-task cell",
    )
    _require(
        receipt["cell"]["training_profile_hash"]
        == config.unified_training_profile.profile_hash,
        "training profile changed after the exclusion smoke",
    )
    training = receipt["training_smoke"]["training"]
    assignment = training["initial_terrain_assignment"]
    _require(training["budget_steps"] == 128, "training smoke budget changed")
    _require(
        assignment["regime_counts"]
        == {"c0_regular": 8, "c1_irregular_flat": 8},
        "training smoke distribution is not C0/C1 50:50",
    )
    _require(
        STAIR_TERRAIN_REGIME_C2 not in assignment["regime_counts"],
        "C2 leaked into training",
    )
    _require(
        receipt["cost"]["budget_fraction"] <= 0.05,
        "training exclusion smoke exceeded its budget",
    )
    return {
        "receipt_hash": receipt["receipt_hash"],
        "source_closure_hash": canonical_json_hash(sources),
        "source_file_count": len(sources),
        "state_transition": (
            "draft_runtime_smoke_then_activation_then_phase4_pre_reference_scoring_correction"
        ),
        "budget_steps": training["budget_steps"],
        "budget_fraction": receipt["cost"]["budget_fraction"],
        "regime_counts": dict(assignment["regime_counts"]),
        "c2_training_env_count": int(
            assignment["regime_counts"].get(STAIR_TERRAIN_REGIME_C2, 0)
        ),
    }


def _contract_receipt(config: Any, manifest: Any, plugin: Any) -> dict[str, Any]:
    catalog = build_stair_cell_catalog(config=config, plugin=plugin)
    _require(catalog.ids() == (config.cell_variant_id,), "Stair has more than one cell")
    _require(catalog.require_active(config.cell_variant_id).protocol_state == "active", "cell is not active")

    suite = config.ordered_evaluation_suite
    profile = config.unified_training_profile
    profile_payload = profile.to_dict()
    exposure = profile_payload["training_exposure"]
    _require(
        exposure["allowed_regimes"] == ["c0_regular", "c1_irregular_flat"],
        "training regimes changed",
    )
    _require(
        exposure["forbidden_regimes"] == [STAIR_TERRAIN_REGIME_C2],
        "C2 is not evaluation-only",
    )

    geometry_sets: dict[str, set[str]] = {}
    c2_profiles: dict[str, list[str]] = {}
    cohort_receipts: dict[str, Any] = {}
    for cohort in COHORTS:
        cells = suite.cells_for_cohort(cohort)
        strata = Counter(cell.stratum_id for cell in cells)
        _require(len(cells) == 24, f"{cohort} cell count changed")
        _require(dict(strata) == EXPECTED_STRATA, f"{cohort} strata changed")
        geometry = {cell.geometry_digest for cell in cells}
        _require(len(geometry) == 24, f"{cohort} contains repeated geometry")
        geometry_sets[cohort] = geometry
        rough = [cell for cell in cells if cell.terrain_regime_id == STAIR_TERRAIN_REGIME_C2]
        _require(len(rough) == 3, f"{cohort} C2 count changed")
        amplitudes = sorted(float(cell.surface_profile["amplitude_m"]) for cell in rough)
        _require(amplitudes == [0.004, 0.006, 0.008], f"{cohort} C2 ladder changed")
        signatures = sorted(canonical_json_hash(dict(cell.surface_profile)) for cell in rough)
        c2_profiles[cohort] = signatures
        cohort_receipts[cohort] = {
            "cell_count": len(cells),
            "seed_count": suite.seed_count(cohort),
            "stratum_counts": dict(sorted(strata.items())),
            "geometry_closure_hash": canonical_json_hash(sorted(geometry)),
            "c2_surface_profile_hashes": signatures,
        }
    for index, left in enumerate(COHORTS):
        for right in COHORTS[index + 1 :]:
            _require(
                not geometry_sets[left] & geometry_sets[right],
                f"{left} and {right} share evaluation geometry",
            )
    _require(
        len({tuple(value) for value in c2_profiles.values()}) == 1,
        "C2 difficulty ladder differs between cohorts",
    )

    public_cells = suite.cells_for_cohort("public_dev")
    suite_strata = list(dict.fromkeys(cell.stratum_id for cell in public_cells))
    suite_claims = {
        stratum: next(
            cell.claim_kind
            for cell in public_cells
            if cell.stratum_id == stratum
        )
        for stratum in suite_strata
    }
    sections = plugin.prompt_adapter().build_prompt_capacity_preflight_sections()
    task_section = sections["task"]
    eval_summary = task_section["fixed_eval_protocol_summary"]
    train_summary = task_section["fixed_train_protocol_summary"]
    public_contract = sections["public_eval_contract"]
    _require(
        task_section["schema_version"]
        == "true_rl_agent.task_prompt.stair_climbing.v4",
        "unified Stair task prompt schema is stale",
    )
    _require(
        public_contract["schema_version"]
        == "true_rl_agent.public_eval_contract.stair.unified.v3",
        "unified Stair public-eval prompt schema is stale",
    )
    _require(eval_summary["strata"] == suite_strata, "prompt strata differ from suite")
    _require(
        eval_summary["cells_per_stratum"]
        == len(public_cells) // len(suite_strata),
        "prompt cell count differs from suite",
    )
    _require(
        train_summary["c2_training_exposure"] == "forbidden",
        "prompt training exposure differs from profile",
    )
    _require(
        public_contract["primary_objective"] == suite.score_contract["j_primary"],
        "prompt objective differs from suite",
    )
    _require(
        public_contract["named_non_regression_guards"]
        == list(suite.score_contract["named_guards"]),
        "prompt guards differ from suite",
    )
    _require(
        public_contract["generalization_claims"] == suite_claims,
        "prompt generalization claims differ from suite",
    )
    c2_prompt_claim = public_contract["c2_evidence_scope"]
    _require(
        c2_prompt_claim
        == "current evaluation-only surface/contact mechanism shift; "
        "zero training exposure; height-scan policy salience not claimed",
        "prompt C2 claim is not the calibrated Phase 3B claim",
    )
    _require(
        "future" not in json.dumps(public_contract, sort_keys=True).lower(),
        "prompt still describes active C2 as future work",
    )
    formal_episode_count = sum(
        suite.seed_count(cohort)
        * len(suite.cells_for_cohort(cohort))
        * suite.episodes_per_seed_per_cell
        for cohort in COHORTS
    )
    _require(formal_episode_count == 1080, "formal episode count changed")

    task_projection_text = json.dumps(
        config.unified_protocol.task_projection(), ensure_ascii=False, sort_keys=True
    ).lower()
    for marker in ("torso_link", "pelvis", "ankle_roll_link", ".usd", "37dof"):
        _require(marker not in task_projection_text, f"robot morphology leaked into task: {marker}")

    return {
        "cell_variant_id": config.cell_variant_id,
        "protocol_state": config.protocol_state,
        "protocol_hash": config.unified_protocol.protocol_hash,
        "base_task_protocol_hash": config.base_task_protocol_hash,
        "resolved_cell_task_protocol_hash": manifest.resolved_cell_task_protocol_hash,
        "manifest_hash": manifest.manifest_hash,
        "training_profile_hash": profile.profile_hash,
        "evaluation_suite_hash": suite.suite_hash,
        "acceptance_protocol_hash": config.acceptance_protocol.protocol_hash,
        "single_cell_count": len(catalog.ids()),
        "training_exposure": exposure,
        "episodes_per_seed_per_cell": suite.episodes_per_seed_per_cell,
        "formal_episode_count": formal_episode_count,
        "cohorts": cohort_receipts,
        "prompt_projection": {
            "task_schema_version": task_section["schema_version"],
            "public_eval_schema_version": public_contract["schema_version"],
            "strata": suite_strata,
            "claim_kinds_by_stratum": suite_claims,
            "c2": c2_prompt_claim,
        },
    }


def _boundary_receipt() -> dict[str, Any]:
    stair_root = ROOT / "src/true_rl_agent/tasks/stair_climbing"
    leaks: list[str] = []
    for path in (ROOT / "src/true_rl_agent").rglob("*.py"):
        if path.is_relative_to(stair_root):
            continue
        text = path.read_text(encoding="utf-8").lower()
        for marker in ROUGHNESS_MARKERS:
            if marker in text:
                leaks.append(f"{path.relative_to(ROOT).as_posix()}:{marker}")
    _require(not leaks, f"Stair C2 semantics leaked outside TaskPlugin: {leaks}")
    return {
        "roughness_owner": "src/true_rl_agent/tasks/stair_climbing",
        "shared_layer_marker_hits": leaks,
        "markers": list(ROUGHNESS_MARKERS),
    }


def emit_current() -> dict[str, Any]:
    config, manifest, plugin = _resolve_cell()
    payload = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "contract": _contract_receipt(config, manifest, plugin),
        "runtime": _runtime_receipt(config, manifest),
        "training_exclusion": _training_exclusion_receipt(config),
        "boundaries": _boundary_receipt(),
        "product_source_hash_mode": "normalized_lf_sha256",
        "product_source_hashes": {
            path: _normalized_text_hash(ROOT / path) for path in PRODUCT_PATHS
        },
    }
    payload["snapshot_hash"] = canonical_json_hash(payload)
    return payload


def verify(snapshot_path: Path = SNAPSHOT_PATH) -> dict[str, Any]:
    expected = _load_object(snapshot_path)
    _require(
        expected.get("schema_version") == SNAPSHOT_SCHEMA_VERSION,
        "snapshot schema mismatch",
    )
    _require(
        expected.get("snapshot_hash")
        == canonical_json_hash(_without_hash(expected, "snapshot_hash")),
        "snapshot hash mismatch",
    )
    current = emit_current()
    for section in ("contract", "runtime", "training_exclusion", "boundaries"):
        _require(
            current[section] == expected[section],
            f"current Phase 3B semantic contract differs: {section}",
        )
    _require(
        expected.get("product_source_hash_mode") == "normalized_lf_sha256",
        "snapshot source hash mode mismatch",
    )
    _source_hashes_from_receipt(expected)
    return {
        "schema_version": VERIFY_SCHEMA_VERSION,
        "status": "pass",
        "snapshot_path": str(snapshot_path.resolve()),
        "snapshot_hash": expected["snapshot_hash"],
        "runtime_receipt_hash": expected["runtime"]["receipt_hash"],
        "training_exclusion_receipt_hash": expected["training_exclusion"][
            "receipt_hash"
        ],
        "protocol_state": expected["contract"]["protocol_state"],
        "formal_episode_count": expected["contract"]["formal_episode_count"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify the Phase 3B contract")
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT_PATH)
    parser.add_argument("--emit-current", action="store_true")
    parser.add_argument("--write-snapshot", action="store_true")
    args = parser.parse_args(argv)
    if args.emit_current and args.write_snapshot:
        parser.error("choose only one of --emit-current and --write-snapshot")
    if args.write_snapshot:
        payload = emit_current()
        args.snapshot.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        result = {
            "schema_version": VERIFY_SCHEMA_VERSION,
            "status": "written",
            "snapshot_path": str(args.snapshot.resolve()),
            "snapshot_hash": payload["snapshot_hash"],
        }
    else:
        result = emit_current() if args.emit_current else verify(args.snapshot)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
