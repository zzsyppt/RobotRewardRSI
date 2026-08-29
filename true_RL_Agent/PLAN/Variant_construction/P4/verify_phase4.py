from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
P3B_DIR = ROOT / "PLAN/Variant_construction/P3B"
for path in (SRC, P3B_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from true_rl_agent.evaluation.binding import canonical_json_hash  # noqa: E402

import verify_phase3b  # noqa: E402

SNAPSHOT_PATH = Path(__file__).with_name("PHASE4_EVIDENCE_SNAPSHOT.json")
REWARD_RECEIPT_PATH = Path(__file__).with_name(
    "PHASE4_REWARD_EQUIVALENCE_RECEIPT.json"
)
SNAPSHOT_SCHEMA_VERSION = "true_rl_agent.variant_construction.phase4_evidence.v1"
VERIFY_SCHEMA_VERSION = "true_rl_agent.variant_construction.phase4_verify.v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def _without(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != field}


def _normalized_text_hash(path: Path) -> str:
    data = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(data).hexdigest()


def _raw_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _close(actual: object, expected: object, *, name: str) -> None:
    _require(
        isinstance(actual, int | float) and not isinstance(actual, bool),
        f"{name} is not numeric",
    )
    _require(
        isinstance(expected, int | float) and not isinstance(expected, bool),
        f"frozen {name} is not numeric",
    )
    _require(
        math.isclose(float(actual), float(expected), rel_tol=1.0e-10, abs_tol=1.0e-10),
        f"{name} changed: {actual!r} != {expected!r}",
    )


def _all_true(value: Mapping[str, Any], *, name: str) -> None:
    false_keys = [key for key, item in value.items() if isinstance(item, bool) and not item]
    _require(not false_keys, f"{name} contains failed checks: {sorted(false_keys)}")


def _resolve_artifact_path(root: Path, recorded_path: str) -> Path:
    direct = Path(recorded_path)
    if direct.exists():
        return direct
    parts = direct.parts
    try:
        index = parts.index(root.name)
    except ValueError as exc:
        raise AssertionError(f"artifact path cannot be rebased: {recorded_path}") from exc
    rebased = root.joinpath(*parts[index + 1 :])
    _require(rebased.exists(), f"missing rebased artifact: {rebased}")
    return rebased


def _verify_product_sources(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    _require(
        snapshot.get("product_source_hash_mode") == "normalized_lf_sha256",
        "product hash mode changed",
    )
    hashes = snapshot.get("product_source_hashes")
    _require(isinstance(hashes, Mapping) and bool(hashes), "product source closure is empty")
    for relative_path, expected_hash in hashes.items():
        _require(isinstance(relative_path, str), "product source path is invalid")
        path = ROOT / relative_path
        _require(path.is_file(), f"missing product source: {relative_path}")
        _require(
            _normalized_text_hash(path) == expected_hash,
            f"product source drift: {relative_path}",
        )
    return {"file_count": len(hashes), "status": "pass"}


def _verify_reward_equivalence(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    receipt = _load_object(REWARD_RECEIPT_PATH)
    _require(receipt.get("status") == "pass", "reward equivalence did not pass")
    _require(
        receipt.get("receipt_hash")
        == canonical_json_hash(_without(receipt, "receipt_hash")),
        "reward equivalence receipt hash mismatch",
    )
    expected = snapshot["reward_equivalence"]
    for key in (
        "receipt_hash",
        "term_count",
        "sample_count",
        "weighted_total_max_abs_difference",
    ):
        _require(receipt.get(key) == expected[key], f"reward equivalence {key} changed")
    _require(len(receipt["term_evidence"]) == 15, "reward term evidence is incomplete")
    _require(
        all(item["finite"] for item in receipt["term_evidence"].values()),
        "reward equivalence contains a non-finite term",
    )
    _require(
        all(item["max_abs_raw_difference"] == 0.0 for item in receipt["term_evidence"].values()),
        "a reward term differs from the official implementation",
    )
    return {
        "receipt_hash": receipt["receipt_hash"],
        "term_count": receipt["term_count"],
        "status": "pass",
    }


def _verify_current_cell(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    p3b = verify_phase3b.verify()
    _require(p3b["status"] == "pass", "Phase 3B prerequisite no longer verifies")
    config, manifest, _plugin = verify_phase3b._resolve_cell()
    identities = snapshot["identities"]
    current = {
        "cell_variant_id": config.cell_variant_id,
        "resolved_cell_manifest_hash": manifest.manifest_hash,
        "task_protocol_hash": config.task_protocol_hash,
        "base_task_protocol_hash": config.base_task_protocol_hash,
        "training_binding_hash": config.formal_training_binding().binding_hash,
        "acceptance_protocol_hash": config.unified_protocol.acceptance_protocol.protocol_hash,
        "evaluation_protocol_hash": (
            config.ordered_evaluation_suite.evaluation_protocol.protocol_hash
        ),
        "seed_cohort_hash": config.seed_resolver().namespace_cohort_hash(
            "formal_reference_candidate_eval"
        ),
        "ordered_evaluation_suite_hash": config.ordered_evaluation_suite.suite_hash,
        "capability_profile_hash": config.candidate_capability_profile.profile_hash,
    }
    for key, actual in current.items():
        _require(actual == identities[key], f"current cell identity changed: {key}")
    return {
        "manifest_hash": manifest.manifest_hash,
        "p3b_snapshot_hash": p3b["snapshot_hash"],
        "status": "pass",
    }


def _verify_raw_artifacts(root: Path, expected: Mapping[str, str]) -> None:
    _require(root.is_dir(), f"artifact root does not exist: {root}")
    for relative_path, expected_hash in expected.items():
        path = root / relative_path
        _require(path.is_file(), f"missing session artifact: {path}")
        _require(_raw_hash(path) == expected_hash, f"raw artifact drift: {path}")


def _verify_reference(
    root: Path,
    expected_session: Mapping[str, Any],
    snapshot: Mapping[str, Any],
) -> Path:
    artifact = _load_object(root / "reference_eval_artifact.json")
    genesis = _load_object(root / "session_genesis.json")
    manifest = _load_object(root / "resolved_cell_manifest.json")
    expected = snapshot["reference"]
    identities = snapshot["identities"]

    _require(artifact["session_id"] == expected_session["session_id"], "reference session mismatch")
    _require(artifact["reference_manifest"]["reference_id"] == expected["reference_id"], "reference id changed")
    _require(
        artifact["reference_manifest"]["manifest_hash"] == expected["reference_manifest_hash"],
        "reference manifest changed",
    )
    _require(
        artifact["reference_manifest"]["source_checksum"] == expected["program_source_checksum"],
        "reference source checksum changed",
    )
    source_hash = hashlib.sha256(artifact["program_source"].encode("utf-8")).hexdigest()
    _require(
        f"sha256:{source_hash}" == expected["program_source_checksum"],
        "embedded reference source bytes changed",
    )
    _require(artifact["checkpoint_hash"] == expected["checkpoint_hash"], "reference checkpoint changed")
    _require(
        artifact["training_binding"]["initialization_mode"] == "random",
        "reference was not randomly initialized",
    )
    _require(
        artifact["training_binding"]["initialization_checkpoint_hash"] is None,
        "reference unexpectedly resumed from a checkpoint",
    )
    _require(
        artifact["training_binding"]["budget_steps"] == expected["budget_steps"],
        "reference budget changed",
    )
    _require(
        artifact["training_binding"]["binding_hash"] == identities["training_binding_hash"],
        "reference training binding changed",
    )
    _require(
        genesis["evaluation_binding"]["evaluation_binding_hash"]
        == identities["evaluation_binding_hash"],
        "Genesis evaluation binding changed",
    )
    _require(manifest["manifest_hash"] == identities["resolved_cell_manifest_hash"], "runtime manifest changed")

    evidence = artifact["qualification_evidence"]
    _require(artifact["qualification_gate"]["status"] == "pass", "reference qualification failed")
    _require(evidence["task_gate"]["status"] == "pass", "reference task readiness failed")
    _require(
        evidence["evidence_hash"] == expected["qualification_evidence_hash"],
        "reference qualification evidence changed",
    )
    _require(
        evidence["objective_family_mask_hash"] == expected["objective_family_mask_hash"],
        "reference family mask changed",
    )
    _require(
        evidence["objective_family_mask"]["qualified_ood_families"]
        == expected["qualified_ood_families"],
        "qualified OOD families changed",
    )

    metrics = artifact["public_result"]["metrics"]
    unified = metrics["unified_score"]
    frozen_accept = expected["public_accept"]
    for key in ("j_primary", "j_ood"):
        _close(unified[key]["score_stats"]["mean"], frozen_accept[key], name=f"reference {key}")
        _close(
            unified[key]["score_stats"]["standard_error"],
            frozen_accept[f"{key}_standard_error"],
            name=f"reference {key} standard error",
        )
    _close(
        (unified["c0_id"]["score_stats"]["mean"] + unified["c1_id"]["score_stats"]["mean"]) / 2.0,
        frozen_accept["j_id"],
        name="reference j_id",
    )
    seed_summaries = metrics["public_accept"]["seed_summaries"]
    for family, family_expected in frozen_accept["families"].items():
        values = [row["family_scores"][family] for row in seed_summaries]
        _close(statistics.mean(values), family_expected["score"], name=f"reference {family}")
        _close(
            statistics.stdev(values) / math.sqrt(len(values)),
            family_expected["fixed_seed_vector_standard_error"],
            name=f"reference {family} standard error",
        )

    checkpoint_path = _resolve_artifact_path(root, artifact["checkpoint_artifact"]["path"])
    _require(
        f"sha256:{_raw_hash(checkpoint_path)}" == expected["checkpoint_hash"],
        "reference checkpoint bytes changed",
    )
    return checkpoint_path


def _load_execution_record(root: Path, reference: Mapping[str, Any]) -> dict[str, Any]:
    path = _resolve_artifact_path(root, str(reference["path"]))
    _require(
        f"sha256:{_raw_hash(path)}" == reference["checksum"],
        f"execution receipt checksum mismatch: {path}",
    )
    receipt = _load_object(path)
    _require(
        receipt["record_hash"] == canonical_json_hash(receipt["record"]),
        "execution receipt record hash mismatch",
    )
    return receipt["record"]


def _verify_candidate_authority(record: Mapping[str, Any]) -> None:
    _require(len(record["trials"]) == 2, "generation width changed")
    for trial in record["trials"]:
        _require(
            trial["candidate"]["changed_components"] == ["reward_callable"],
            "candidate changed something other than the reward program",
        )
        _require(trial["verifier"]["status"] == "pass", "candidate verifier failed")


def _selected_trial_execution_record(root: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    formal = record["formal"]
    _require(isinstance(formal, Mapping), "generation has no formal record")
    kind = f"{formal['selected_lane_id']}_trial_execution_receipt"
    artifact = next(item for item in record["private_artifacts"] if item["kind"] == kind)
    path = _resolve_artifact_path(root, artifact["uri"])
    _require(
        f"sha256:{_raw_hash(path)}" == artifact["checksum"],
        "selected trial execution receipt changed",
    )
    receipt = _load_object(path)
    _require(receipt["record_hash"] == formal["round_record_hash"], "formal round hash changed")
    return receipt["record"]


def _verify_prompt_privacy(root: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    paths = sorted(root.rglob("prompt_payload.json"))
    expected_count = expected.get("prompt_payload_file_count", expected.get("file_count"))
    expected_total = expected.get("prompt_payload_total_bytes", expected.get("total_bytes"))
    expected_sizes = expected.get("prompt_payload_sizes_bytes", expected.get("sizes_bytes"))
    _require(len(paths) == expected_count, "provider prompt count changed")
    sizes = [path.stat().st_size for path in paths]
    _require(sum(sizes) == expected_total, "provider prompt byte total changed")
    _require(sorted(sizes) == sorted(expected_sizes), "provider prompt size multiset changed")

    forbidden = (
        "public_accept.",
        "sealed_validation.",
        "c600fe0b23cb898035d764f971d60756d4670e51f378833a2eb824cb5490e826",
        "89537aa93a5bc18f07563cf07e6bd3d68ec4aaf49cfee67eab8afe0a72853f79",
    )
    for path in paths:
        text = path.read_text(encoding="utf-8")
        hits = [marker for marker in forbidden if marker in text]
        _require(not hits, f"private evaluation evidence leaked into {path}: {hits}")
    return {"file_count": len(paths), "total_bytes": sum(sizes), "status": "pass"}


def _verify_cost(report: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    receipt = report["actual_cost_receipt"]
    _require(receipt["cap_checks"]["all_passed"] is True, "session exceeded a cost cap")
    _all_true(receipt["cap_checks"], name="session cost cap checks")
    actual = receipt["actual"]
    for key, expected_value in expected.items():
        if isinstance(expected_value, float):
            _close(actual[key], expected_value, name=f"actual cost {key}")
        else:
            _require(actual[key] == expected_value, f"actual cost changed: {key}")


def _verify_pilot(root: Path, snapshot: Mapping[str, Any]) -> dict[str, Any]:
    expected = snapshot["pilot"]
    _verify_raw_artifacts(root, expected["raw_artifact_hashes"])
    report = _load_object(root / "generation_session_report.json")
    _require(report["source_commit"] == snapshot["source"]["artifact_source_commit"], "pilot source commit changed")
    _require(report["session_id"] == expected["session_id"], "pilot session id changed")
    _require(report["session_mode"] == expected["session_mode"], "pilot mode changed")
    _require(report["generation_outcome"] == "accepted", "pilot generation was not accepted")
    _all_true(report["checks"], name="pilot checks")
    for key, value in expected["terminal_binding"].items():
        _require(report["terminal_binding"][key] == value, f"pilot terminal binding changed: {key}")
    _verify_cost(report, expected["actual_cost"])

    ledger = [json.loads(line) for line in (root / "ledger.jsonl").read_text(encoding="utf-8").splitlines()]
    _require(len(ledger) == 1, "pilot ledger is not one generation")
    record = _load_execution_record(root, report["generation_execution_receipt"])
    _require(record == ledger[0], "pilot execution receipt differs from ledger")
    _verify_candidate_authority(record)
    selected = next(trial for trial in record["trials"] if trial["trial_status"] == "accepted")
    frozen = expected["accepted_candidate"]
    _require(selected["candidate"]["candidate_id"] == frozen["candidate_id"], "pilot accepted candidate changed")
    _require(selected["candidate"]["artifact_hash"] == frozen["artifact_hash"], "pilot candidate artifact changed")
    _require(selected["candidate"]["program_source_checksum"] == frozen["source_checksum"], "pilot source changed")
    _require(selected["train"]["checkpoint_hash"] == frozen["checkpoint_hash"], "pilot checkpoint changed")
    _close(selected["public_dev"]["public_summary"]["j_primary"], frozen["public_dev_j_primary"], name="pilot public-dev score")
    _close(selected["public_accept"]["public_summary"]["j_primary"], frozen["public_accept_j_primary"], name="pilot public-accept score")
    prompt = _verify_prompt_privacy(root, expected["prompt_payload"])
    reference_checkpoint = _verify_reference(root, expected, snapshot)
    return {"prompt": prompt, "reference_checkpoint": reference_checkpoint, "status": "pass"}


def _verify_validation(root: Path, snapshot: Mapping[str, Any]) -> dict[str, Any]:
    expected = snapshot["validation"]
    _verify_raw_artifacts(root, expected["raw_artifact_hashes"])
    report = _load_object(root / "generation_session_report.json")
    _require(report["source_commit"] == snapshot["source"]["artifact_source_commit"], "validation source commit changed")
    _require(report["session_id"] == expected["session_id"], "validation session id changed")
    _require(report["session_mode"] == expected["session_mode"], "validation mode changed")
    _all_true(report["checks"], name="validation checks")
    _require(len(report["generations"]) == 3, "validation did not close exactly three generations")
    for key, value in expected["terminal_binding"].items():
        _require(report["terminal_binding"][key] == value, f"validation terminal binding changed: {key}")
    _verify_cost(report, expected["actual_cost"])

    metrics = report["session_metrics"]
    frozen_metrics = expected["terminal_metrics"]
    _require(metrics["generation_outcome_counts"] == frozen_metrics["generation_outcome_counts"], "generation outcomes changed")
    _require(metrics["paired_ready_generation_count"] == 3, "not all generations are paired-ready")
    _close(metrics["paired_ready_rate"], 1.0, name="paired-ready rate")
    _require(metrics["retry_eval_seed_units_consumed"] == 0, "evaluation retries consumed budget")
    _require(metrics["retry_train_jobs_consumed"] == 0, "training retries consumed budget")
    _require(metrics["parent_bootstrap_workload_delta"]["eval_seed_units"] == 0, "duplicate parent bootstrap returned")
    selector = metrics["selector_reversal"]
    _require(selector["comparison_source"] == "runner_private_public_accept_phase_receipts", "selector audit source changed")
    _require(selector["observed_reversal_count"] == 0, "selector reversal observed")
    _require(selector["blocking_reversal_count"] == 0, "blocking selector reversal observed")
    _require(selector["public_dev_accept_consistency"]["numerator"] == 2, "selector consistency numerator changed")
    _require(selector["public_dev_accept_consistency"]["denominator"] == 2, "selector consistency denominator changed")

    ledger = [json.loads(line) for line in (root / "ledger.jsonl").read_text(encoding="utf-8").splitlines()]
    _require(len(ledger) == 3, "validation ledger does not contain exactly three records")
    expected_records = expected["generation_records"]
    previous_head: str | None = None
    for index, (summary, record, frozen) in enumerate(
        zip(report["generations"], ledger, expected_records, strict=True),
        start=1,
    ):
        _require(record["generation_index"] == index, "generation indices are not contiguous")
        _require(record["generation_outcome"] == frozen["outcome"], "generation outcome changed")
        _require(canonical_json_hash(record) == frozen["record_hash"], "ledger record hash changed")
        _require(summary["generation_record_hash"] == frozen["record_hash"], "report/ledger record hash mismatch")
        _require(summary["generation_closure"]["closure_hash"] == frozen["closure_hash"], "generation closure changed")
        _require(summary["generation_closure"]["resource_evidence_hash"] == frozen["resource_evidence_hash"], "resource closure changed")
        _require(summary["generation_closure"]["resource_evidence"]["final_compute_idle"]["rows"] == [], "GPU process remained at generation close")
        _require(summary["generation_closure"]["resource_evidence"]["attempt_chain"]["open_reservation_count"] == 0, "operation reservation remained open")
        loaded = _load_execution_record(root, summary["generation_execution_receipt"])
        _require(loaded == record, "generation execution receipt differs from ledger")
        _require(summary["generation_execution_receipt"]["checksum"] == frozen["execution_receipt_checksum"], "generation execution checksum changed")
        _verify_candidate_authority(record)
        if previous_head is not None:
            _require(record["parent_current_best_hash"] == previous_head, "current-best lineage is discontinuous")
        previous_head = record["resulting_current_best_hash"]
        formal = record["formal"]
        if frozen["formal_selected_lane"] is None:
            _require(formal is None, "no-selection generation unexpectedly has a formal decision")
        else:
            _require(formal["selected_lane_id"] == frozen["formal_selected_lane"], "formal lane changed")

    accepted_record = ledger[0]
    accepted_trial = next(trial for trial in accepted_record["trials"] if trial["trial_status"] == "accepted")
    frozen_candidate = expected["accepted_candidate"]
    _require(accepted_trial["candidate"]["candidate_id"] == frozen_candidate["candidate_id"], "accepted candidate changed")
    _require(accepted_trial["candidate"]["artifact_hash"] == frozen_candidate["artifact_hash"], "accepted artifact changed")
    _require(accepted_trial["candidate"]["program_source_checksum"] == frozen_candidate["source_checksum"], "accepted source changed")
    _require(accepted_trial["train"]["checkpoint_hash"] == frozen_candidate["checkpoint_hash"], "accepted checkpoint changed")
    decision = _selected_trial_execution_record(root, accepted_record)["decision"]
    _require(decision["accepted"] is True, "generation 1 decision is no longer accepted")
    for key in ("delta", "threshold"):
        _close(decision["comparison"][key], frozen_candidate["decision"][key], name=f"accepted decision {key}")
    _close(
        decision["comparison"]["threshold_components"]["paired_standard_error"],
        frozen_candidate["decision"]["paired_standard_error"],
        name="accepted decision paired standard error",
    )
    reference_rows = _load_object(root / "reference_eval_artifact.json")[
        "public_result"
    ]["metrics"]["public_accept"]["seed_summaries"]
    candidate_rows = accepted_trial["public_accept"]["public_summary"][
        "seed_summaries"
    ]
    for family, frozen_delta in frozen_candidate["decision"][
        "family_paired_deltas"
    ].items():
        deltas = [
            candidate["family_scores"][family]
            - reference["family_scores"][family]
            for reference, candidate in zip(
                reference_rows,
                candidate_rows,
                strict=True,
            )
        ]
        _close(
            statistics.mean(deltas),
            frozen_delta["delta"],
            name=f"accepted {family} paired delta",
        )
        _close(
            statistics.stdev(deltas) / math.sqrt(len(deltas)),
            frozen_delta["paired_standard_error"],
            name=f"accepted {family} paired standard error",
        )

    sealed_artifact = next(
        item
        for item in accepted_record["private_artifacts"]
        if item["kind"] == "lane-b_sealed_validation_phase_receipt"
    )
    sealed_path = _resolve_artifact_path(root, sealed_artifact["uri"])
    _require(
        f"sha256:{_raw_hash(sealed_path)}" == sealed_artifact["checksum"],
        "accepted sealed receipt changed",
    )
    sealed = _load_object(sealed_path)["phase_result"]["task_evidence"]["summary"]
    for key in ("j_primary", "j_id", "j_ood"):
        _close(
            sealed[key],
            frozen_candidate["sealed_validation"][key],
            name=f"accepted sealed {key}",
        )
    for family in ("F_scalar", "F_c1", "F_c2"):
        _close(
            sealed["family_scores"][family],
            frozen_candidate["sealed_validation"][family],
            name=f"accepted sealed {family}",
        )
    sealed_j_primary = [row["j_primary"] for row in sealed["seed_summaries"]]
    _close(
        statistics.stdev(sealed_j_primary) / math.sqrt(len(sealed_j_primary)),
        frozen_candidate["sealed_validation"]["j_primary_standard_error"],
        name="accepted sealed j_primary standard error",
    )

    rejected_record = ledger[1]
    rejected = _selected_trial_execution_record(root, rejected_record)["decision"]
    frozen_rejection = expected["generation_2_rejection"]
    _require(rejected["accepted"] is False, "generation 2 rejection changed")
    _require(rejected["reason"] == "named_guard_score_regression", "generation 2 rejection reason changed")
    _close(rejected["comparison"]["delta"], frozen_rejection["delta"], name="generation 2 delta")
    _close(rejected["comparison"]["threshold"], frozen_rejection["threshold"], name="generation 2 threshold")
    failed_guard = rejected["comparison"]["named_guards"][frozen_rejection["failed_guard"]]
    _require(failed_guard["met"] is False, "generation 2 failed guard now passes")
    _close(failed_guard["delta"], frozen_rejection["failed_guard_delta"], name="generation 2 failed guard delta")

    prompt = _verify_prompt_privacy(root, expected["prompt_and_memory_audit"])
    reference_checkpoint = _verify_reference(root, expected, snapshot)
    return {"prompt": prompt, "reference_checkpoint": reference_checkpoint, "status": "pass"}


def verify(
    snapshot_path: Path = SNAPSHOT_PATH,
    *,
    pilot_root: Path | None = None,
    validation_root: Path | None = None,
) -> dict[str, Any]:
    snapshot = _load_object(snapshot_path)
    _require(snapshot.get("schema_version") == SNAPSHOT_SCHEMA_VERSION, "snapshot schema changed")
    _require(snapshot.get("status") == "pass", "Phase 4 snapshot did not pass")
    _require(
        snapshot.get("snapshot_hash")
        == canonical_json_hash(_without(snapshot, "snapshot_hash")),
        "snapshot hash mismatch",
    )
    product = _verify_product_sources(snapshot)
    reward = _verify_reward_equivalence(snapshot)
    cell = _verify_current_cell(snapshot)

    runtime: dict[str, Any] = {"status": "not_rechecked"}
    pilot_result: dict[str, Any] | None = None
    validation_result: dict[str, Any] | None = None
    if pilot_root is not None:
        pilot_result = _verify_pilot(pilot_root.resolve(), snapshot)
    if validation_root is not None:
        validation_result = _verify_validation(validation_root.resolve(), snapshot)
    if pilot_result is not None or validation_result is not None:
        runtime = {
            "pilot": None if pilot_result is None else {key: value for key, value in pilot_result.items() if key != "reference_checkpoint"},
            "validation": None if validation_result is None else {key: value for key, value in validation_result.items() if key != "reference_checkpoint"},
            "status": "pass",
        }
    if pilot_result is not None and validation_result is not None:
        pilot_checkpoint = pilot_result["reference_checkpoint"]
        validation_checkpoint = validation_result["reference_checkpoint"]
        _require(not os.path.samefile(pilot_checkpoint, validation_checkpoint), "validation reused the pilot reference file")
        _require(_raw_hash(pilot_checkpoint) == _raw_hash(validation_checkpoint), "deterministic reference bytes diverged")

    return {
        "schema_version": VERIFY_SCHEMA_VERSION,
        "status": "pass",
        "snapshot_hash": snapshot["snapshot_hash"],
        "loop_verified": snapshot["phase_gate"]["loop_verified"],
        "improvement_observed": snapshot["phase_gate"]["improvement_observed"],
        "product": product,
        "reward_equivalence": reward,
        "cell": cell,
        "runtime": runtime,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the frozen Phase 4 evidence")
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT_PATH)
    parser.add_argument("--pilot-root", type=Path)
    parser.add_argument("--validation-root", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            verify(
                args.snapshot,
                pilot_root=args.pilot_root,
                validation_root=args.validation_root,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
