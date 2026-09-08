from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from true_rl_agent.evaluation.binding import canonical_json_hash  # noqa: E402

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
        if (
            not isinstance(relative_path, str)
            or not relative_path
            or Path(relative_path).is_absolute()
            or ".." in Path(relative_path).parts
            or not isinstance(expected, str)
            or len(expected) != 64
            or any(character not in "0123456789abcdef" for character in expected)
        ):
            raise AssertionError(f"source closure entry invalid: {path.name}")
    return payload


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
    product_tree = expected.get("product_tree")
    if not isinstance(product_tree, Mapping):
        raise AssertionError("Phase 3A product provenance is missing")
    file_hashes = product_tree.get("file_hashes")
    if (
        product_tree.get("hash_mode") != "normalized_lf_sha256"
        or not isinstance(file_hashes, Mapping)
        or not file_hashes
        or product_tree.get("file_count") != len(file_hashes)
        or product_tree.get("product_tree_hash")
        != canonical_json_hash(dict(file_hashes))
    ):
        raise AssertionError("Phase 3A product provenance closure is invalid")
    runtime_evidence = _evidence_receipt()
    if runtime_evidence != expected.get("runtime_evidence"):
        raise AssertionError("Phase 3A frozen runtime evidence differs from snapshot")
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
    contract = expected["verification_contract"]
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
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT_PATH)
    parser.add_argument("--require-remote-artifacts", action="store_true")
    parser.add_argument("--require-go", action="store_true")
    args = parser.parse_args()
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
