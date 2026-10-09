"""Content-addressed storage for complete Stair Phase D pilot evidence."""

from __future__ import annotations

import os
import shutil
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from true_rl_agent.evaluation.binding import (
    canonical_json_hash,
    require_sha256_checksum,
    require_sha256_digest,
)
from true_rl_agent.json_contract import canonical_json_bytes
from true_rl_agent.runner.checkpoint import file_sha256
from true_rl_agent.tasks.stair_climbing.pilot_artifacts import (
    load_canonical_pilot_json,
)

STAIR_PHASE_D_PILOT_ARTIFACT_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.phase_d_pilot_artifact.v1"
)
_ARTIFACT_KIND = "stair_phase_d_pilot_run"
_ARTIFACT_URI_PREFIX = "artifact://stair-phase-d/sha256/"


def materialize_stair_phase_d_pilot_artifact(
    *,
    run_root: Path,
    artifact_store_root: Path,
) -> dict[str, Any]:
    """Copy a complete run into immutable content-addressed storage."""

    run_root = _existing_directory(run_root, "run_root")
    artifact_store_root = _local_directory_target(
        artifact_store_root,
        "artifact_store_root",
    )
    report = load_canonical_pilot_json(
        run_root / "phase_d_pilot_report.json",
        context="stair Phase D report",
    )
    draft_hash = require_sha256_digest(
        report.get("draft_protocol_hash"),
        "stair Phase D artifact draft protocol hash",
    )
    report_hash = require_sha256_digest(
        report.get("report_hash"),
        "stair Phase D artifact report hash",
    )
    files = _scan_payload(run_root)
    identity = {
        "artifact_kind": _ARTIFACT_KIND,
        "draft_protocol_hash": draft_hash,
        "phase_d_report_hash": report_hash,
        "files": files,
    }
    content_hash = canonical_json_hash(identity)
    artifact_uri = _ARTIFACT_URI_PREFIX + content_hash
    manifest = {
        "schema_version": STAIR_PHASE_D_PILOT_ARTIFACT_SCHEMA_VERSION,
        **identity,
        "artifact_uri": artifact_uri,
        "content_hash": content_hash,
        "file_count": len(files),
        "total_size_bytes": sum(item["size_bytes"] for item in files),
    }
    manifest["manifest_hash"] = canonical_json_hash(manifest)

    artifact_store_root.mkdir(parents=True, exist_ok=True)
    final_root = artifact_store_root / content_hash
    if final_root.exists() or final_root.is_symlink():
        return validate_stair_phase_d_pilot_artifact(
            manifest=manifest,
            artifact_store_root=artifact_store_root,
        )["manifest"]

    temporary_root = (
        artifact_store_root / f".tmp-{content_hash[:12]}-{uuid.uuid4().hex[:8]}"
    )
    payload_root = temporary_root / "payload"
    try:
        payload_root.mkdir(parents=True, exist_ok=False)
        for entry in files:
            source = run_root / entry["relative_path"]
            destination = payload_root / entry["relative_path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            if (
                file_sha256(source) != entry["checksum"]
                or file_sha256(destination) != entry["checksum"]
                or destination.stat().st_size != entry["size_bytes"]
            ):
                raise ValueError("stair Phase D artifact bytes changed during copy")
        (temporary_root / "manifest.json").write_bytes(
            canonical_json_bytes(manifest, trailing_newline=True)
        )
        os.replace(temporary_root, final_root)
    except Exception:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
    return validate_stair_phase_d_pilot_artifact(
        manifest=manifest,
        artifact_store_root=artifact_store_root,
    )["manifest"]


def validate_stair_phase_d_pilot_artifact_manifest(
    *,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    fields = {
        "schema_version",
        "artifact_kind",
        "draft_protocol_hash",
        "phase_d_report_hash",
        "files",
        "artifact_uri",
        "content_hash",
        "file_count",
        "total_size_bytes",
        "manifest_hash",
    }
    if not isinstance(manifest, Mapping) or set(manifest) != fields:
        raise ValueError("stair Phase D pilot artifact manifest fields mismatch")
    normalized = dict(manifest)
    if normalized["schema_version"] != STAIR_PHASE_D_PILOT_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("stair Phase D pilot artifact schema mismatch")
    if normalized["artifact_kind"] != _ARTIFACT_KIND:
        raise ValueError("stair Phase D pilot artifact kind mismatch")
    require_sha256_digest(
        normalized["draft_protocol_hash"],
        "stair Phase D artifact draft protocol hash",
    )
    require_sha256_digest(
        normalized["phase_d_report_hash"],
        "stair Phase D artifact report hash",
    )
    raw_files = normalized["files"]
    if not isinstance(raw_files, list) or not raw_files:
        raise ValueError("stair Phase D pilot artifact file manifest is empty")
    files = []
    previous_path = None
    for item in raw_files:
        entry = _validate_file_entry(item, previous_path=previous_path)
        files.append(entry)
        previous_path = entry["relative_path"]
    identity = {
        "artifact_kind": normalized["artifact_kind"],
        "draft_protocol_hash": normalized["draft_protocol_hash"],
        "phase_d_report_hash": normalized["phase_d_report_hash"],
        "files": files,
    }
    content_hash = canonical_json_hash(identity)
    if normalized["content_hash"] != content_hash:
        raise ValueError("stair Phase D pilot artifact content hash mismatch")
    if normalized["artifact_uri"] != _ARTIFACT_URI_PREFIX + content_hash:
        raise ValueError("stair Phase D pilot artifact URI mismatch")
    if normalized["file_count"] != len(files) or normalized[
        "total_size_bytes"
    ] != sum(item["size_bytes"] for item in files):
        raise ValueError("stair Phase D pilot artifact aggregate mismatch")
    if normalized["manifest_hash"] != canonical_json_hash(
        {key: value for key, value in normalized.items() if key != "manifest_hash"}
    ):
        raise ValueError("stair Phase D pilot artifact manifest hash mismatch")
    normalized["files"] = files
    return normalized


def validate_stair_phase_d_pilot_artifact(
    *,
    manifest: Mapping[str, Any],
    artifact_store_root: Path,
) -> dict[str, Any]:
    normalized = validate_stair_phase_d_pilot_artifact_manifest(
        manifest=manifest
    )
    files = normalized["files"]
    content_hash = normalized["content_hash"]

    artifact_store_root = _existing_directory(
        artifact_store_root,
        "artifact_store_root",
    )
    artifact_root = artifact_store_root / content_hash
    if artifact_root.is_symlink() or not artifact_root.is_dir():
        raise ValueError("stair Phase D pilot artifact directory is missing")
    stored_manifest = load_canonical_pilot_json(
        artifact_root / "manifest.json",
        context="stored stair Phase D pilot artifact manifest",
    )
    if stored_manifest != normalized:
        raise ValueError("stored stair Phase D pilot artifact manifest differs")
    payload_root = _existing_directory(artifact_root / "payload", "artifact payload")
    actual_files = _scan_payload(payload_root)
    if actual_files != files:
        raise ValueError("stair Phase D pilot artifact payload differs from manifest")
    return {
        "manifest": normalized,
        "artifact_root": artifact_root,
        "payload_root": payload_root,
    }


def _scan_payload(root: Path) -> list[dict[str, Any]]:
    entries = []
    for current_root, directory_names, file_names in os.walk(
        root,
        topdown=True,
        followlinks=False,
    ):
        current = Path(current_root)
        for name in directory_names:
            if (current / name).is_symlink():
                raise ValueError("stair Phase D pilot artifact rejects directory symlinks")
        for name in file_names:
            path = current / name
            if path.is_symlink() or not path.is_file():
                raise ValueError("stair Phase D pilot artifact requires regular files")
            relative = path.relative_to(root).as_posix()
            checksum = file_sha256(path)
            if checksum is None:
                raise ValueError("stair Phase D pilot artifact file could not be hashed")
            entries.append(
                {
                    "relative_path": relative,
                    "checksum": checksum,
                    "size_bytes": path.stat().st_size,
                }
            )
    entries.sort(key=lambda item: item["relative_path"])
    if not entries:
        raise ValueError("stair Phase D pilot artifact payload is empty")
    return entries


def _validate_file_entry(
    value: Any,
    *,
    previous_path: str | None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "relative_path",
        "checksum",
        "size_bytes",
    }:
        raise ValueError("stair Phase D pilot artifact file fields mismatch")
    relative_path = value["relative_path"]
    if (
        not isinstance(relative_path, str)
        or not relative_path
        or relative_path != relative_path.strip()
    ):
        raise ValueError("stair Phase D pilot artifact relative path is invalid")
    relative = Path(relative_path)
    if (
        relative.is_absolute()
        or relative_path != relative.as_posix()
        or ".." in relative.parts
    ):
        raise ValueError("stair Phase D pilot artifact path is not canonical relative")
    if previous_path is not None and relative_path <= previous_path:
        raise ValueError("stair Phase D pilot artifact files are not strictly sorted")
    checksum = require_sha256_checksum(
        value["checksum"],
        "stair Phase D pilot artifact file checksum",
    )
    size_bytes = value["size_bytes"]
    if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes < 0:
        raise ValueError("stair Phase D pilot artifact file size is invalid")
    return {
        "relative_path": relative_path,
        "checksum": checksum,
        "size_bytes": size_bytes,
    }


def _existing_directory(value: Path, field_name: str) -> Path:
    if not isinstance(value, Path):
        raise TypeError(f"stair Phase D artifact {field_name} must be a Path")
    if value.is_symlink():
        raise ValueError(f"stair Phase D artifact {field_name} must not be a symlink")
    path = value.resolve(strict=True)
    if not path.is_dir():
        raise ValueError(f"stair Phase D artifact {field_name} must be a directory")
    return path


def _local_directory_target(value: Path, field_name: str) -> Path:
    if not isinstance(value, Path):
        raise TypeError(f"stair Phase D artifact {field_name} must be a Path")
    if value.is_symlink():
        raise ValueError(f"stair Phase D artifact {field_name} must not be a symlink")
    path = value.resolve()
    if path.exists() and not path.is_dir():
        raise ValueError(f"stair Phase D artifact {field_name} must be a directory")
    return path


__all__ = [
    "STAIR_PHASE_D_PILOT_ARTIFACT_SCHEMA_VERSION",
    "materialize_stair_phase_d_pilot_artifact",
    "validate_stair_phase_d_pilot_artifact",
    "validate_stair_phase_d_pilot_artifact_manifest",
]
