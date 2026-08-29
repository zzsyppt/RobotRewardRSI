from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from true_rl_agent.composition.base import (  # noqa: E402
    TaskCompositionContext,
    resolve_task_composition,
)
from true_rl_agent.evaluation.binding import canonical_json_hash  # noqa: E402
from true_rl_agent.json_contract import canonical_json_bytes  # noqa: E402
from true_rl_agent.tasks.stair_climbing.composition import (  # noqa: E402
    StairClimbingCompositionProvider,
)
from true_rl_agent.tasks.stair_climbing.config import (  # noqa: E402
    STAIR_G1_37DOF_IRREGULAR_CELL_VARIANT_ID,
)
from true_rl_agent.tasks.stair_climbing.robot_binding import (  # noqa: E402
    STAIR_G1_37DOF_CLEAN_LEGACY_CELL_VARIANT_ID,
)
from true_rl_agent.tasks.stair_climbing.runtime_projection import (  # noqa: E402
    stair_runtime_projection_from_manifest,
)

SNAPSHOT_PATH = Path(__file__).with_name("P3A_C1_IRREGULAR_SNAPSHOT.json")
CALIBRATION_PATH = Path(__file__).with_name("PHASE3A_CALIBRATION_RECEIPT.json")
REFERENCE_PATH = Path(__file__).with_name(
    "PHASE3A_REFERENCE_QUALIFICATION_RECEIPT.json"
)
RUNTIME_EVAL_PATH = Path(__file__).with_name("PHASE3A_RUNTIME_EVAL_RECEIPT.json")
RUNTIME_TRAIN_PATH = Path(__file__).with_name(
    "PHASE3A_RUNTIME_TRAIN_SMOKE_RECEIPT.json"
)

SNAPSHOT_SCHEMA_VERSION = "true_rl_agent.variant_construction.phase3a_snapshot.v1"
VERIFY_SCHEMA_VERSION = "true_rl_agent.variant_construction.phase3a_verify.v1"

_PRODUCT_PATHS = (
    "configs/robots/unitree_g1_37dof_isaaclab_v2.json",
    "scripts/isaaclab_stair_rsl_rl_eval.py",
    "scripts/isaaclab_stair_rsl_rl_train.py",
    "src/true_rl_agent/composition/base.py",
    "src/true_rl_agent/robots/defaults.py",
    "src/true_rl_agent/tasks/stair_climbing/candidate_admission.py",
    "src/true_rl_agent/tasks/stair_climbing/composition.py",
    "src/true_rl_agent/tasks/stair_climbing/config.py",
    "src/true_rl_agent/tasks/stair_climbing/entrypoint_contract.py",
    "src/true_rl_agent/tasks/stair_climbing/episode_evidence.py",
    "src/true_rl_agent/tasks/stair_climbing/evaluation_runtime.py",
    "src/true_rl_agent/tasks/stair_climbing/irregular_calibration.py",
    "src/true_rl_agent/tasks/stair_climbing/irregular_reference_gate.py",
    "src/true_rl_agent/tasks/stair_climbing/irregular_training.py",
    "src/true_rl_agent/tasks/stair_climbing/isaaclab_execution.py",
    "src/true_rl_agent/tasks/stair_climbing/isaaclab_irregular_terrain.py",
    "src/true_rl_agent/tasks/stair_climbing/isaaclab_reward_runtime.py",
    "src/true_rl_agent/tasks/stair_climbing/isaaclab_runtime_env.py",
    "src/true_rl_agent/tasks/stair_climbing/memory.py",
    "src/true_rl_agent/tasks/stair_climbing/ordered_evaluation.py",
    "src/true_rl_agent/tasks/stair_climbing/ordered_summary.py",
    "src/true_rl_agent/tasks/stair_climbing/phased_adapter.py",
    "src/true_rl_agent/tasks/stair_climbing/phased_evaluation.py",
    "src/true_rl_agent/tasks/stair_climbing/plugin.py",
    "src/true_rl_agent/tasks/stair_climbing/prompt_adapter.py",
    "src/true_rl_agent/tasks/stair_climbing/public_projection.py",
    "src/true_rl_agent/tasks/stair_climbing/reference_program.py",
    "src/true_rl_agent/tasks/stair_climbing/reference_programs/g1_irregular_v1/training_code/components.py",
    "src/true_rl_agent/tasks/stair_climbing/reference_programs/official_g1_rough_v1/training_code/components.py",
    "src/true_rl_agent/tasks/stair_climbing/reference_qualification.py",
    "src/true_rl_agent/tasks/stair_climbing/reward_inputs.py",
    "src/true_rl_agent/tasks/stair_climbing/reward_program.py",
    "src/true_rl_agent/tasks/stair_climbing/reward_source_semantics.py",
    "src/true_rl_agent/tasks/stair_climbing/robot_binding.py",
    "src/true_rl_agent/tasks/stair_climbing/runtime_projection.py",
    "src/true_rl_agent/tasks/stair_climbing/seed_policy.py",
    "src/true_rl_agent/tasks/stair_climbing/terrain_evidence.py",
    "src/true_rl_agent/tasks/stair_climbing/transition_decoder.py",
    "src/true_rl_agent/tasks/stair_climbing/vnext_reward_contract.py",
    "configs/tasks/stair_g1_37dof_irregular_acceptance_protocol_v1.json",
    "configs/tasks/stair_g1_37dof_irregular_calibration_protocol_v1.json",
    "configs/tasks/stair_g1_37dof_irregular_evaluation_suite_v1.json",
    "configs/tasks/stair_g1_37dof_irregular_reference_gate_v1.json",
    "configs/tasks/stair_g1_37dof_irregular_training_profile_v1.json",
    "PLAN/Variant_construction/P3A/capture_phase3a_calibration_receipt.py",
    "PLAN/Variant_construction/P3A/capture_phase3a_reference_qualification.py",
    "PLAN/Variant_construction/P3A/capture_phase3a_runtime_receipt.py",
)


def _normalized_text_hash(path: Path) -> str:
    payload = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"JSON root must be an object: {path}")
    return value


def _without_receipt_hash(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "receipt_hash"}


def _assert_receipt(path: Path) -> dict[str, Any]:
    payload = _load_object(path)
    if payload.get("receipt_hash") != canonical_json_hash(
        _without_receipt_hash(payload)
    ):
        raise AssertionError(f"receipt hash mismatch: {path.name}")
    source_hashes = payload.get("product_source_hashes")
    if not isinstance(source_hashes, Mapping) or not source_hashes:
        raise AssertionError(f"source closure missing: {path.name}")
    for relative_path, expected in source_hashes.items():
        actual = _normalized_text_hash(PROJECT_ROOT / str(relative_path))
        if actual != expected:
            raise AssertionError(
                f"source closure drift: {path.name}:{relative_path}"
            )
    return payload


def _captured_head() -> str | None:
    result = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    return value if result.returncode == 0 and len(value) == 40 else None


def _resolve(cell_variant_id: str) -> Any:
    provider = StairClimbingCompositionProvider()
    values = dict(provider.cli_defaults())
    values["cell_variant_id"] = cell_variant_id
    with tempfile.TemporaryDirectory() as temporary:
        return resolve_task_composition(
            provider=provider,
            values=values,
            context=TaskCompositionContext(
                project_root=PROJECT_ROOT,
                output_dir=Path(temporary),
            ),
        )


def _contract_receipt() -> dict[str, Any]:
    composition = _resolve(STAIR_G1_37DOF_IRREGULAR_CELL_VARIANT_ID)
    manifest = composition.resolved_cell_manifest
    if manifest is None or composition.reference_program_artifact is None:
        raise AssertionError("C1 composition is incomplete")
    config = composition.task_plugin.config
    suite = config.ordered_evaluation_suite
    training = config.irregular_training_profile
    coverage = suite.pairwise_coverage_report()
    if coverage["all_pairs_fully_covered"] is not True:
        raise AssertionError("C1 pairwise coverage is incomplete")
    geometry = {
        cell.cell_id: {
            "geometry_digest": cell.geometry_digest,
            "recomputed_geometry_digest": canonical_json_hash(
                cell.geometry_payload()
            ),
            "progress_goal_m": cell.progress_goal_m,
            "episode_horizon_control_steps": cell.episode_horizon_control_steps,
        }
        for cell in suite.cells
    }
    if any(
        item["geometry_digest"] != item["recomputed_geometry_digest"]
        for item in geometry.values()
    ):
        raise AssertionError("C1 geometry digest mismatch")
    training_endpoints = {
        f"{pattern_id}:{difficulty:.1f}": training.sequence(
            pattern_id, difficulty
        )
        for pattern_id in sorted(training.terrain["sequence_patterns"])
        for difficulty in (0.0, 1.0)
    }
    resolved_hash = manifest.resolved_cell_task_protocol_hash
    evaluation_binding = composition.backend_adapter.config.evaluation_binding
    training_binding = config.formal_training_binding()
    identity_hashes = {
        "manifest": resolved_hash,
        "task_spec": composition.task_plugin.task_spec.resolved_task_protocol_hash(),
        "evaluation_binding": evaluation_binding.task_protocol_hash,
        "training_binding_v2": training_binding.task_protocol_hash,
    }
    if set(identity_hashes.values()) != {resolved_hash}:
        raise AssertionError("C1 task/eval/train comparison domains diverged")
    projection = stair_runtime_projection_from_manifest(manifest)
    reference_manifest = composition.reference_program_artifact.manifest
    clean = _resolve(STAIR_G1_37DOF_CLEAN_LEGACY_CELL_VARIANT_ID)
    clean_manifest = clean.resolved_cell_manifest
    if clean_manifest is None:
        raise AssertionError("legacy Stair composition is incomplete")
    formal_cohorts = ("public_dev", "public_accept", "sealed_validation")
    return {
        "cell_variant_id": manifest.cell_variant_id,
        "manifest": manifest.to_dict(),
        "identity_hashes": identity_hashes,
        "evaluation_binding": evaluation_binding.to_public_dict(),
        "training_binding": training_binding.to_public_dict(),
        "ordered_suite": suite.to_dict(),
        "pairwise_coverage": coverage,
        "geometry": geometry,
        "training_profile": training.to_dict(),
        "training_endpoint_hash": canonical_json_hash(training_endpoints),
        "formal_eval_env_counts": {
            cohort: len(suite.cells_for_cohort(cohort))
            * suite.episodes_per_seed_per_cell
            for cohort in formal_cohorts
        },
        "runtime_projection": {
            "projection_hash": projection.projection_hash,
            "reward_input_schema_hash_vnext": (
                projection.reward_input_schema_hash_vnext
            ),
            "reward_capability_hash_vnext": projection.reward_capability_hash_vnext,
        },
        "reference_program": reference_manifest.to_public_dict(),
        "legacy_anchor": {
            "cell_variant_id": clean_manifest.cell_variant_id,
            "manifest_hash": clean_manifest.manifest_hash,
            "task_protocol_hash": clean.task_plugin.task_spec.resolved_task_protocol_hash(),
            "evaluation_binding_hash": (
                clean.backend_adapter.config.evaluation_binding.binding_hash
            ),
        },
    }


def _product_tree_receipt() -> dict[str, Any]:
    hashes = {
        relative_path: _normalized_text_hash(PROJECT_ROOT / relative_path)
        for relative_path in _PRODUCT_PATHS
    }
    return {
        "hash_mode": "normalized_lf_sha256",
        "file_count": len(hashes),
        "file_hashes": hashes,
        "product_tree_hash": canonical_json_hash(hashes),
    }


def _evidence_receipt() -> dict[str, Any]:
    calibration = _assert_receipt(CALIBRATION_PATH)
    reference = _assert_receipt(REFERENCE_PATH)
    runtime_eval = _assert_receipt(RUNTIME_EVAL_PATH)
    runtime_train = _assert_receipt(RUNTIME_TRAIN_PATH)
    return {
        "calibration": {
            "receipt_hash": calibration["receipt_hash"],
            "status": calibration["status"],
            "resolved_cell_manifest_hash": calibration[
                "resolved_cell_manifest_hash"
            ],
            "terrain_qualification_result_hash": calibration[
                "terrain_qualification"
            ]["result_hash"],
        },
        "reference_qualification": {
            "receipt_hash": reference["receipt_hash"],
            "status": reference["status"],
            "cell_loop_ready": reference["cell_loop_ready"],
            "commitment_hash": reference["commitment_hash"],
            "resolved_cell_manifest_hash": reference["resolved_cell_manifest"][
                "manifest_hash"
            ],
            "reference_manifest_hash": reference["reference_program"][
                "manifest_hash"
            ],
            "checkpoint_checksum": reference["checkpoint"]["checksum"],
            "required_gate_statuses": reference["required_gate_statuses"],
            "sealed_validation_status": reference["sealed_validation_status"],
            "formal_training_executed": reference["formal_training_executed"],
            "execution_cost": reference["execution_cost"],
        },
        "runtime_eval": {
            "receipt_hash": runtime_eval["receipt_hash"],
            "status": runtime_eval["status"],
        },
        "runtime_train_smoke": {
            "receipt_hash": runtime_train["receipt_hash"],
            "status": runtime_train["status"],
        },
    }


def emit_current() -> dict[str, Any]:
    payload = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "provenance": {
            "captured_repository_head": _captured_head(),
            "head_identity_semantics": (
                "provenance only; product bytes and resolved contracts are gates"
            ),
        },
        "verification_contract": _contract_receipt(),
        "product_tree": _product_tree_receipt(),
        "runtime_evidence": _evidence_receipt(),
    }
    payload["receipt_hash"] = canonical_json_hash(payload)
    return payload


def _assert_remote_artifacts(reference: Mapping[str, Any]) -> None:
    checkpoint = reference["checkpoint"]
    checkpoint_path = Path(str(checkpoint["path"]))
    if _file_sha256(checkpoint_path) != checkpoint["checksum"]:
        raise AssertionError("reference checkpoint bytes mismatch")
    for phase_ref in reference["phase_receipt_refs"].values():
        phase_path = Path(str(phase_ref["path"]))
        if _file_sha256(phase_path) != phase_ref["checksum"]:
            raise AssertionError("reference phase receipt bytes mismatch")


def verify(
    snapshot_path: Path = SNAPSHOT_PATH,
    *,
    require_remote_artifacts: bool = False,
    require_go: bool = False,
) -> dict[str, Any]:
    expected = _load_object(snapshot_path)
    if expected.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise AssertionError("Phase 3A snapshot schema mismatch")
    if expected.get("receipt_hash") != canonical_json_hash(
        _without_receipt_hash(expected)
    ):
        raise AssertionError("Phase 3A snapshot receipt hash mismatch")
    current = emit_current()
    for section in ("verification_contract", "product_tree", "runtime_evidence"):
        if current[section] != expected.get(section):
            raise AssertionError(f"Phase 3A current {section} mismatch")
    reference = _assert_receipt(REFERENCE_PATH)
    phase_go = (
        reference.get("status") == "pass"
        and reference.get("cell_loop_ready") is True
        and reference.get("sealed_validation_status") == "pass"
        and reference.get("formal_training_executed") is True
        and all(
            value == "pass"
            for value in reference.get("required_gate_statuses", {}).values()
        )
    )
    if require_go and not phase_go:
        raise AssertionError("Phase 3A reference qualification is not GO")
    if require_remote_artifacts:
        _assert_remote_artifacts(reference)
    contract = current["verification_contract"]
    return {
        "schema_version": VERIFY_SCHEMA_VERSION,
        "status": "pass",
        "cell_variant_id": contract["cell_variant_id"],
        "ordered_cell_count": len(contract["ordered_suite"]["cells"]),
        "pairwise_coverage_status": "pass",
        "phase_status": "go" if phase_go else "no_go",
        "cell_loop_ready": phase_go,
        "snapshot_receipt_hash": expected["receipt_hash"],
        "remote_artifacts_verified": require_remote_artifacts,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--emit", action="store_true")
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT_PATH)
    parser.add_argument("--require-remote-artifacts", action="store_true")
    parser.add_argument("--require-go", action="store_true")
    args = parser.parse_args()
    if args.emit:
        args.snapshot.write_bytes(
            canonical_json_bytes(emit_current(), trailing_newline=True)
        )
        print(args.snapshot)
        return 0
    print(
        json.dumps(
            verify(
                args.snapshot,
                require_remote_artifacts=args.require_remote_artifacts,
                require_go=args.require_go,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
