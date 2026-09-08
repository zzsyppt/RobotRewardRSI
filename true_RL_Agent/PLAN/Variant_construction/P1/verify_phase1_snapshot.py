from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from true_rl_agent.candidates.artifact import (  # noqa: E402
    CANDIDATE_ARTIFACT_MANIFEST_FILENAME,
    load_sealed_candidate_artifact,
)
from true_rl_agent.candidates.schema import CandidateChangeSet  # noqa: E402
from true_rl_agent.composition.cell_manifest import (  # noqa: E402
    ResolvedCellManifestV1,
    compile_cell_manifest,
)
from true_rl_agent.evaluation.binding import canonical_json_hash  # noqa: E402
from true_rl_agent.json_contract import (  # noqa: E402
    load_strict_json_object,
    strict_json_object_loads,
)
from true_rl_agent.robots.defaults import build_default_robot_registry  # noqa: E402
from true_rl_agent.tasks.push_recovery.config import (  # noqa: E402
    PushRecoveryTaskConfig,
)
from true_rl_agent.tasks.push_recovery.plugin import (  # noqa: E402
    PushRecoveryTaskPlugin,
)
from true_rl_agent.tasks.push_recovery.robot_binding import (  # noqa: E402
    PUSH_G1_29DOF_LEGACY_CELL_VARIANT_ID,
    build_push_cell_catalog,
)
from true_rl_agent.tasks.stair_climbing.config import (  # noqa: E402
    StairClimbingTaskConfig,
)
from true_rl_agent.tasks.stair_climbing.plugin import (  # noqa: E402
    StairClimbingTaskPlugin,
)
from true_rl_agent.tasks.stair_climbing.robot_binding import (  # noqa: E402
    STAIR_G1_37DOF_CLEAN_LEGACY_CELL_VARIANT_ID,
    build_stair_cell_catalog,
)
from true_rl_agent.tasks.stair_climbing.row_catalog import stair_robot_row  # noqa: E402


SNAPSHOT_PATH = Path(__file__).with_name("P1_MATRIX_CONTRACT_SNAPSHOT.json")
EVIDENCE_ROOT = Path(__file__).with_name("PUSH_SHARED_LOOP_DRY_RUN")
SNAPSHOT_SCHEMA_VERSION = (
    "true_rl_agent.variant_construction.phase1_matrix_contract_receipt.v1"
)
VERIFY_SCHEMA_VERSION = (
    "true_rl_agent.variant_construction.phase1_matrix_contract_verify.v1"
)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_equal(actual: Any, expected: Any, name: str) -> None:
    if actual != expected:
        raise AssertionError(f"{name} mismatch: {actual!r} != {expected!r}")


def _without_hash(payload: Mapping[str, Any], field_name: str) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != field_name}


def _all_lists_empty(value: Any) -> bool:
    if isinstance(value, list):
        return not value
    if isinstance(value, Mapping):
        return all(_all_lists_empty(item) for item in value.values())
    return True


def _profile_receipt(registry: Any, profile_id: str) -> dict[str, Any]:
    profile = registry.profile(profile_id)
    adapter_id = next(
        adapter_id
        for adapter_id in registry.runtime_adapter_ids()
        if registry.runtime_adapter(adapter_id).robot_profile_id == profile_id
    )
    adapter = registry.runtime_adapter(adapter_id)
    facts = profile.runtime_facts
    return {
        "asset_closure_hash": facts["asset_closure"]["closure_hash"],
        "body_count": len(facts["body_names"]),
        "joint_count": len(facts["joint_names"]),
        "profile_hash": profile.profile_hash,
        "runtime_adapter_hash": adapter.adapter_hash,
        "runtime_adapter_id": adapter.runtime_adapter_id,
        "runtime_facts_hash": profile.runtime_facts_hash,
    }


def _cell_receipt(manifest: ResolvedCellManifestV1) -> dict[str, Any]:
    return {
        "base_task_protocol_hash": manifest.base_task_protocol_hash,
        "binding_hash": manifest.robot_task_binding["binding_hash"],
        "environment_hash": manifest.environment["environment_hash"],
        "legacy_training_binding_v1_grant_hash": (
            manifest.legacy_training_binding_v1_grant_hash
        ),
        "manifest_hash": manifest.manifest_hash,
        "resolved_cell_task_protocol_hash": (
            manifest.resolved_cell_task_protocol_hash
        ),
        "robot_profile_id": manifest.robot_profile["robot_profile_id"],
        "runtime_adapter_id": manifest.runtime_adapter["runtime_adapter_id"],
        "task_id": manifest.task_id,
    }


def _current_matrix_contract() -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, ResolvedCellManifestV1],
]:
    registry = build_default_robot_registry()
    profiles = {
        profile_id: _profile_receipt(registry, profile_id)
        for profile_id in registry.profile_ids()
    }

    push_config = PushRecoveryTaskConfig()
    push_plugin = PushRecoveryTaskPlugin(push_config)
    push_definition = build_push_cell_catalog(
        config=push_config,
        plugin=push_plugin,
        robot_registry=registry,
    ).resolve(PUSH_G1_29DOF_LEGACY_CELL_VARIANT_ID)
    push_manifest = compile_cell_manifest(
        definition=push_definition,
        robot_registry=registry,
        requirements=push_plugin.robot_requirements(),
        base_task_protocol_hash=push_plugin.task_spec.resolved_task_protocol_hash(),
    ).require_supported()

    stair_row = stair_robot_row(STAIR_G1_37DOF_CLEAN_LEGACY_CELL_VARIANT_ID)
    stair_config = StairClimbingTaskConfig.defaults(
        PROJECT_ROOT / stair_row.protocol_relative_path,
        cell_variant_id=STAIR_G1_37DOF_CLEAN_LEGACY_CELL_VARIANT_ID,
    )
    stair_plugin = StairClimbingTaskPlugin(stair_config)
    stair_definition = build_stair_cell_catalog(
        config=stair_config,
        plugin=stair_plugin,
        robot_registry=registry,
    ).resolve(STAIR_G1_37DOF_CLEAN_LEGACY_CELL_VARIANT_ID)
    stair_manifest = compile_cell_manifest(
        definition=stair_definition,
        robot_registry=registry,
        requirements=stair_plugin.robot_requirements(),
        base_task_protocol_hash=stair_config.task_protocol_hash,
    ).require_supported()

    manifests = {
        push_manifest.cell_variant_id: push_manifest,
        stair_manifest.cell_variant_id: stair_manifest,
    }
    cells = {
        cell_variant_id: _cell_receipt(manifest)
        for cell_variant_id, manifest in sorted(manifests.items())
    }
    return profiles, cells, manifests


def _validate_frozen_product_tree(product_tree: Any) -> None:
    if not isinstance(product_tree, Mapping):
        raise AssertionError("Phase 1 frozen product tree is missing")
    file_hashes = product_tree.get("file_hashes")
    if not isinstance(file_hashes, Mapping) or not file_hashes:
        raise AssertionError("Phase 1 frozen product source closure is missing")
    normalized = {str(path): str(digest) for path, digest in file_hashes.items()}
    if any(
        not path
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        for path, digest in normalized.items()
    ):
        raise AssertionError("Phase 1 frozen product source closure is invalid")
    _assert_equal(
        product_tree.get("file_count"),
        len(normalized),
        "Phase 1 frozen product file count",
    )
    _assert_equal(
        product_tree.get("product_tree_hash"),
        canonical_json_hash(normalized),
        "Phase 1 frozen product tree hash",
    )


def _cell_semantics(cell_variant_id: str, cell: Mapping[str, Any]) -> dict[str, Any]:
    semantics = {
        key: cell[key]
        for key in (
            "cell_variant_id",
            "task_id",
            "robot_profile_id",
            "runtime_adapter_id",
        )
    }
    if cell_variant_id == PUSH_G1_29DOF_LEGACY_CELL_VARIANT_ID:
        semantics.update(
            binding_hash=cell["binding_hash"],
            environment_hash=cell["environment_hash"],
        )
    return semantics


def _manifest_axes(manifest: ResolvedCellManifestV1) -> dict[str, Any]:
    return {
        "cell_variant_id": manifest.cell_variant_id,
        "task_id": manifest.task_id,
        "robot_profile_id": manifest.robot_profile["robot_profile_id"],
        "runtime_adapter_id": manifest.runtime_adapter["runtime_adapter_id"],
        "binding_hash": manifest.robot_task_binding["binding_hash"],
        "environment_hash": manifest.environment["environment_hash"],
    }


def _evidence_entries(evidence_root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(evidence_root).as_posix(),
            "sha256": _file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(item for item in evidence_root.rglob("*") if item.is_file())
    ]


def _verify_git_preserves_evidence_bytes(evidence_root: Path) -> None:
    try:
        git_root = Path(
            subprocess.run(
                ["git", "-C", str(evidence_root), "rev-parse", "--show-toplevel"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        ).resolve()
        relative_paths = [
            path.resolve().relative_to(git_root).as_posix()
            for path in sorted(item for item in evidence_root.rglob("*") if item.is_file())
        ]
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        raise AssertionError("unable to resolve Git attributes for P1 evidence") from exc
    result = subprocess.run(
        ["git", "-C", str(git_root), "check-attr", "text", "eol", "--", *relative_paths],
        check=True,
        capture_output=True,
        text=True,
    )
    expected = {
        f"{relative_path}: {attribute}: unset"
        for relative_path in relative_paths
        for attribute in ("text", "eol")
    }
    observed = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    _assert_equal(observed, expected, "P1 evidence byte-preserving Git attributes")


def _verify_sealed_candidate(evidence_root: Path) -> str:
    manifests = tuple(
        evidence_root.glob(
            f"candidate_artifacts/**/{CANDIDATE_ARTIFACT_MANIFEST_FILENAME}"
        )
    )
    _assert_equal(len(manifests), 1, "sealed candidate artifact count")
    manifest_path = manifests[0]
    manifest = load_strict_json_object(
        manifest_path,
        context="P1 sealed candidate artifact manifest",
        require_canonical=True,
        trailing_newline=True,
    )
    artifact = load_sealed_candidate_artifact(
        manifest_path.parent,
        change_set=CandidateChangeSet(frozenset()),
        expected_candidate_id=manifest["candidate_id"],
        expected_artifact_hash=manifest["artifact_hash"],
        expected_subject_manifest_hash=manifest["subject_manifest_hash"],
        expected_program_source_checksum=manifest["program_source_checksum"],
    )
    return artifact.artifact_hash


def _verify_evidence(
    *,
    snapshot: Mapping[str, Any],
    evidence_root: Path,
    current_manifests: Mapping[str, ResolvedCellManifestV1],
) -> dict[str, Any]:
    declared = snapshot["push_shared_loop_evidence"]
    entries = _evidence_entries(evidence_root)
    _assert_equal(entries, declared["evidence_entries"], "P1 evidence entries")
    _assert_equal(len(entries), declared["evidence_file_count"], "P1 evidence file count")
    _assert_equal(
        sum(item["size_bytes"] for item in entries),
        declared["evidence_total_bytes"],
        "P1 evidence total bytes",
    )
    _assert_equal(
        canonical_json_hash(entries),
        declared["evidence_closure_hash"],
        "P1 evidence closure hash",
    )
    _verify_git_preserves_evidence_bytes(evidence_root)
    artifact_hash = _verify_sealed_candidate(evidence_root)

    audit = load_strict_json_object(
        evidence_root / "audit_summary.json",
        context="P1 Push shared LOOP audit summary",
    )
    genesis = load_strict_json_object(
        evidence_root / "session_genesis.json",
        context="P1 Push shared LOOP session Genesis",
    )
    derived = {
        "accepted_count": audit["accepted_count"],
        "all_checks_passed": all(value is True for value in audit["checks"].values()),
        "all_violation_lists_empty": _all_lists_empty(audit["violations"]),
        "authored_candidate_count": audit["authored_candidate_count"],
        "check_count": len(audit["checks"]),
        "evidence_closure_hash": canonical_json_hash(entries),
        "evidence_entries": entries,
        "evidence_file_count": len(entries),
        "evidence_total_bytes": sum(item["size_bytes"] for item in entries),
        "full_eval_completed_count": audit["full_eval_completed_count"],
        "rounds_completed": audit["rounds_completed"],
        "runtime_snapshot_hash": genesis["runtime_snapshot_hash"],
        "sealed_validation_completed_count": audit[
            "sealed_validation_completed_count"
        ],
        "session_genesis_hash": audit["session_genesis_hash"],
        "session_id": audit["session_id"],
        "trained_candidate_count": audit["trained_candidate_count"],
    }
    _assert_equal(derived, declared, "P1 Push shared LOOP evidence receipt")
    if not derived["all_checks_passed"] or not derived["all_violation_lists_empty"]:
        raise AssertionError("P1 Push shared LOOP audit did not pass")
    _assert_equal(
        genesis["session_genesis_hash"],
        audit["session_genesis_hash"],
        "P1 session Genesis hash",
    )

    frozen_manifest = ResolvedCellManifestV1.from_mapping(
        strict_json_object_loads(
            (evidence_root / "resolved_cell_manifest.json").read_bytes(),
            context="P1 frozen Push resolved cell manifest",
        )
    )
    current_push = current_manifests[PUSH_G1_29DOF_LEGACY_CELL_VARIANT_ID]
    _assert_equal(
        _manifest_axes(frozen_manifest),
        _manifest_axes(current_push),
        "P1 frozen/current Push cell axes",
    )
    return {
        "artifact_hash": artifact_hash,
        "evidence_file_count": len(entries),
        "session_id": audit["session_id"],
    }


def verify(
    *,
    snapshot_path: Path = SNAPSHOT_PATH,
    evidence_root: Path = EVIDENCE_ROOT,
) -> dict[str, Any]:
    snapshot = load_strict_json_object(
        snapshot_path,
        context="Phase 1 matrix contract snapshot",
    )
    _assert_equal(
        snapshot["schema_version"],
        SNAPSHOT_SCHEMA_VERSION,
        "Phase 1 snapshot schema",
    )
    _assert_equal(
        canonical_json_hash(_without_hash(snapshot, "receipt_hash")),
        snapshot["receipt_hash"],
        "Phase 1 receipt hash",
    )
    _validate_frozen_product_tree(snapshot.get("product_tree"))
    profiles, cells, manifests = _current_matrix_contract()
    frozen_profiles = snapshot["robot_profiles"]
    current_legacy_profiles = {
        profile_id: profiles[profile_id] for profile_id in frozen_profiles
    }
    _assert_equal(
        current_legacy_profiles,
        frozen_profiles,
        "Phase 1 legacy robot profiles",
    )
    _assert_equal(
        {
            cell_id: _cell_semantics(
                cell_id,
                {"cell_variant_id": cell_id, **cell},
            )
            for cell_id, cell in cells.items()
        },
        {
            cell_id: _cell_semantics(
                cell_id,
                {"cell_variant_id": cell_id, **cell},
            )
            for cell_id, cell in snapshot["legacy_cells"].items()
        },
        "Phase 1 legacy cell axes",
    )
    evidence = _verify_evidence(
        snapshot=snapshot,
        evidence_root=evidence_root.resolve(),
        current_manifests=manifests,
    )
    return {
        "status": "pass",
        "schema_version": VERIFY_SCHEMA_VERSION,
        "product_file_count": snapshot["product_tree"]["file_count"],
        "robot_profile_count": len(frozen_profiles),
        "current_registry_profile_count": len(profiles),
        "legacy_cell_count": len(cells),
        "source_identity_semantics": "historical_provenance",
        **evidence,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT_PATH)
    parser.add_argument("--evidence-root", type=Path, default=EVIDENCE_ROOT)
    args = parser.parse_args()
    print(
        json.dumps(
            verify(
                snapshot_path=args.snapshot,
                evidence_root=args.evidence_root,
            ),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
