from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
for root in (PROJECT_ROOT, SRC_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from true_rl_agent.evaluation.binding import canonical_json_hash  # noqa: E402
from true_rl_agent.tasks.stair_climbing.unified_training import (  # noqa: E402
    STAIR_TERRAIN_REGIME_C0,
    STAIR_TERRAIN_REGIME_C1,
)

SNAPSHOT_PATH = Path(__file__).with_name("PHASE3R_CONTRACT_SNAPSHOT.json")
RUNTIME_SMOKE_PATH = Path(__file__).with_name("PHASE3R_RUNTIME_SMOKE_RECEIPT.json")
SNAPSHOT_SCHEMA_VERSION = "true_rl_agent.variant_construction.phase3r_snapshot.v1"
VERIFY_SCHEMA_VERSION = "true_rl_agent.variant_construction.phase3r_verify.v1"
RUNTIME_SMOKE_SCHEMA_VERSION = (
    "true_rl_agent.variant_construction.phase3r_runtime_smoke.v1"
)

def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"JSON root must be an object: {path}")
    return value


def _without_receipt_hash(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "receipt_hash"}


def _runtime_receipt(
    contract: Mapping[str, Any],
    path: Path = RUNTIME_SMOKE_PATH,
) -> dict[str, Any]:
    receipt = _load_object(path)
    if receipt.get("schema_version") != RUNTIME_SMOKE_SCHEMA_VERSION:
        raise AssertionError("Phase 3R runtime smoke schema mismatch")
    if receipt.get("receipt_hash") != canonical_json_hash(
        _without_receipt_hash(receipt)
    ):
        raise AssertionError("Phase 3R runtime smoke receipt hash mismatch")
    if receipt.get("status") != "pass":
        raise AssertionError("Phase 3R runtime smoke did not pass")

    source_hashes = receipt.get("product_source_hashes")
    if (
        receipt.get("product_source_hash_mode") != "normalized_lf_sha256"
        or not isinstance(source_hashes, Mapping)
        or not source_hashes
    ):
        raise AssertionError("Phase 3R runtime source closure is invalid")
    for relative_path, expected_hash in source_hashes.items():
        if (
            not isinstance(relative_path, str)
            or not relative_path
            or Path(relative_path).is_absolute()
            or ".." in Path(relative_path).parts
            or not isinstance(expected_hash, str)
            or len(expected_hash) != 64
            or any(character not in "0123456789abcdef" for character in expected_hash)
        ):
            raise AssertionError("Phase 3R runtime source entry is invalid")

    source_preflight = receipt.get("source_preflight")
    if not isinstance(source_preflight, Mapping):
        raise AssertionError("Phase 3R execution source preflight is missing")
    source_summary = source_preflight.get("summary")
    source_evidence = source_preflight.get("evidence")
    if not isinstance(source_summary, Mapping) or not isinstance(
        source_evidence, Mapping
    ):
        raise AssertionError("Phase 3R execution source preflight is invalid")
    evidence_payload = {
        key: value
        for key, value in source_evidence.items()
        if key != "source_manifest_hash"
    }
    if (
        source_summary.get("status") != "pass"
        or source_summary.get("task_protocol_hash")
        != contract.get("resolved_cell_task_protocol_hash")
        or source_summary.get("source_manifest_hash")
        != source_evidence.get("source_manifest_hash")
        or source_evidence.get("source_manifest_hash")
        != canonical_json_hash(evidence_payload)
    ):
        raise AssertionError("Phase 3R execution source preflight identity drifted")
    source_files = source_evidence.get("source_files")
    if not isinstance(source_files, list) or not source_files:
        raise AssertionError("Phase 3R execution source file evidence is missing")
    for entry in source_files:
        if (
            not isinstance(entry, Mapping)
            or not isinstance(entry.get("repository_path"), str)
            or not isinstance(entry.get("size_bytes"), int)
            or not isinstance(entry.get("sha256"), str)
            or len(entry["sha256"]) != 64
        ):
            raise AssertionError("Phase 3R execution source entry is invalid")

    cell = receipt.get("cell")
    if not isinstance(cell, Mapping):
        raise AssertionError("Phase 3R runtime cell identity is missing")
    identity_pairs = {
        "cell_variant_id": "cell_variant_id",
        "protocol_state": "protocol_state",
        "manifest_hash": "resolved_cell_manifest_hash",
        "base_task_protocol_hash": "base_task_protocol_hash",
        "resolved_cell_task_protocol_hash": "resolved_cell_task_protocol_hash",
        "training_profile_hash": "training_profile_hash",
        "evaluation_suite_hash": "evaluation_suite_hash",
        "vnext_reward_authority_hash": "vnext_reward_authority_hash",
    }
    for runtime_name, contract_name in identity_pairs.items():
        if cell.get(runtime_name) != contract.get(contract_name):
            raise AssertionError(
                f"Phase 3R runtime cell identity drifted: {runtime_name}"
            )

    training_smoke = receipt.get("training_smoke")
    if not isinstance(training_smoke, Mapping):
        raise AssertionError("Phase 3R training smoke evidence is missing")
    training = training_smoke.get("training")
    activation = training_smoke.get("runtime_activation_summary")
    if (
        training_smoke.get("status") != "pass"
        or training_smoke.get("training_executed") is not True
        or not isinstance(training, Mapping)
        or not isinstance(activation, Mapping)
    ):
        raise AssertionError("Phase 3R real PPO smoke did not execute")
    assignment = training.get("initial_terrain_assignment")
    if not isinstance(assignment, Mapping) or assignment.get("regime_counts") != {
        STAIR_TERRAIN_REGIME_C0: 8,
        STAIR_TERRAIN_REGIME_C1: 8,
    }:
        raise AssertionError("Phase 3R runtime C0/C1 assignment drifted")
    if (
        training.get("budget_steps") != 128
        or activation.get("declared_term_count") != 15
        or int(activation.get("dispatcher_calls", 0)) <= 1
        or activation.get("nonzero_reward_contribution") is not True
        or not math.isfinite(float(activation.get("last_total_rate_mean", math.nan)))
    ):
        raise AssertionError("Phase 3R runtime reward/PPO evidence is invalid")

    curriculum = receipt.get("curriculum_probe")
    if (
        not isinstance(curriculum, Mapping)
        or curriculum.get("status") != "pass"
        or curriculum.get("terrain_type_immutable") is not True
        or curriculum.get("levels_before") != [2, 0, 1]
        or curriculum.get("levels_after") != [3, 0, 0]
        or curriculum.get("terrain_types_before")
        != curriculum.get("terrain_types_after")
    ):
        raise AssertionError("Phase 3R runtime curriculum evidence is invalid")

    geometry = receipt.get("evaluation_geometry_probe")
    representatives = (
        geometry.get("representatives") if isinstance(geometry, Mapping) else None
    )
    if (
        not isinstance(geometry, Mapping)
        or geometry.get("status") != "pass"
        or not isinstance(representatives, list)
        or [item.get("cohort") for item in representatives]
        != [
            "public_dev",
            "public_accept",
            "sealed_validation",
        ]
        or len({item.get("geometry_digest") for item in representatives}) != 3
        or any(
            item.get("mesh_count") != 10
            or not math.isclose(
                float(item.get("floor_depth_m", math.nan)),
                float(contract["shared_floor_depth_m"]),
                abs_tol=1.0e-12,
            )
            for item in representatives
        )
    ):
        raise AssertionError("Phase 3R runtime evaluation geometry evidence is invalid")

    eligibility = receipt.get("eligibility")
    if not isinstance(eligibility, Mapping) or eligibility != {
        "reference_qualification": False,
        "current_best": False,
        "candidate_acceptance": False,
        "reason": "phase3r_draft_smoke_only",
    }:
        raise AssertionError("Phase 3R draft smoke eligibility drifted")
    cost = receipt.get("cost")
    if (
        not isinstance(cost, Mapping)
        or cost.get("smoke_budget_steps") != 128
        or cost.get("full_training_budget_steps") != 196_608_000
        or float(cost.get("budget_fraction", math.inf))
        > float(cost.get("maximum_authorized_fraction", -math.inf))
        or float(cost.get("maximum_authorized_fraction", math.nan)) != 0.05
    ):
        raise AssertionError("Phase 3R runtime smoke budget evidence is invalid")

    return {
        "receipt_hash": receipt["receipt_hash"],
        "source_closure_hash": canonical_json_hash(dict(source_hashes)),
        "source_file_count": len(source_hashes),
        "execution_source_manifest_hash": source_summary["source_manifest_hash"],
        "device": receipt["environment"]["device"],
        "cuda_device_name": receipt["environment"]["cuda_device_name"],
        "smoke_budget_steps": cost["smoke_budget_steps"],
        "budget_fraction": cost["budget_fraction"],
        "runtime_assignment": assignment["regime_counts"],
        "dispatcher_calls": activation["dispatcher_calls"],
        "reward_term_calls": activation["reward_term_calls"],
        "curriculum_status": curriculum["status"],
        "evaluation_geometry_status": geometry["status"],
        "eligibility": dict(eligibility),
    }


def verify(snapshot_path: Path = SNAPSHOT_PATH) -> dict[str, Any]:
    expected = _load_object(snapshot_path)
    if expected.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise AssertionError("Phase 3R snapshot schema mismatch")
    if expected.get("receipt_hash") != canonical_json_hash(
        _without_receipt_hash(expected)
    ):
        raise AssertionError("Phase 3R snapshot receipt hash mismatch")
    product_tree = expected.get("product_tree")
    if not isinstance(product_tree, Mapping):
        raise AssertionError("Phase 3R product provenance is missing")
    file_hashes = product_tree.get("file_hashes")
    if (
        product_tree.get("hash_mode") != "normalized_lf_sha256"
        or not isinstance(file_hashes, Mapping)
        or not file_hashes
        or product_tree.get("file_count") != len(file_hashes)
        or product_tree.get("product_tree_hash")
        != canonical_json_hash(dict(file_hashes))
    ):
        raise AssertionError("Phase 3R product provenance closure is invalid")
    for relative_path, digest in file_hashes.items():
        if (
            not isinstance(relative_path, str)
            or not relative_path
            or Path(relative_path).is_absolute()
            or ".." in Path(relative_path).parts
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise AssertionError("Phase 3R product provenance entry is invalid")
    runtime = _runtime_receipt(expected["contract"])
    if runtime != expected.get("runtime_smoke"):
        raise AssertionError("Phase 3R frozen runtime evidence differs from snapshot")
    return {
        "schema_version": VERIFY_SCHEMA_VERSION,
        "status": "pass",
        "snapshot_path": str(snapshot_path.resolve()),
        "receipt_hash": expected["receipt_hash"],
        "product_tree_hash": expected["product_tree"]["product_tree_hash"],
        "runtime_smoke_receipt_hash": expected["runtime_smoke"]["receipt_hash"],
        "protocol_state": expected["contract"]["protocol_state"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify the Phase 3R contract snapshot"
    )
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT_PATH)
    args = parser.parse_args(argv)
    payload = verify(args.snapshot)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
