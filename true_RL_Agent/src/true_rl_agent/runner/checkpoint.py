"""Runner-owned checkpoint sealing and byte-level safety evidence."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from true_rl_agent.evaluation.binding import (
    require_sha256_checksum,
)
from true_rl_agent.runner.specs import CheckpointRef

CHECKPOINT_SAFETY_REPORT_SCHEMA_VERSION = "true_rl_agent.checkpoint_safety_report.v1"


@dataclass(frozen=True)
class CheckpointSafetyReport:
    """Public-safe byte-level checkpoint safety evidence from an eval process."""

    status: str
    checkpoint_hash: str | None
    expected_checkpoint_hash: str | None
    file_exists: bool
    byte_read_status: str
    weights_only_load_status: str
    checkpoint_hash_match_status: str
    loader_error_class: str | None
    loader_error_summary: str | None
    reason: str

    def __post_init__(self) -> None:
        if self.checkpoint_hash is not None:
            require_sha256_checksum(
                self.checkpoint_hash,
                "checkpoint safety report checkpoint_hash",
            )
        if self.expected_checkpoint_hash is not None:
            require_sha256_checksum(
                self.expected_checkpoint_hash,
                "checkpoint safety report expected_checkpoint_hash",
            )

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CHECKPOINT_SAFETY_REPORT_SCHEMA_VERSION,
            "status": self.status,
            "checkpoint_hash": self.checkpoint_hash,
            "expected_checkpoint_hash": self.expected_checkpoint_hash,
            "file_exists": self.file_exists,
            "byte_read_status": self.byte_read_status,
            "weights_only_load_status": self.weights_only_load_status,
            "checkpoint_hash_match_status": self.checkpoint_hash_match_status,
            "loader_error_class": self.loader_error_class,
            "loader_error_summary": self.loader_error_summary,
            "reason": self.reason,
        }


def seal_checkpoint(checkpoint: CheckpointRef, *, sealed_uri: str) -> CheckpointRef:
    if checkpoint.role not in {"train_output", "eval"}:
        raise ValueError(f"cannot seal checkpoint with role: {checkpoint.role}")
    return CheckpointRef(
        uri=sealed_uri,
        checksum=checkpoint.checksum,
        role="eval",
        sealed=True,
        metadata={**dict(checkpoint.metadata), "sealed_by": "runner"},
    )


def build_checkpoint_safety_report(
    path: str | Path,
    *,
    expected_checkpoint_hash: str | None = None,
    weights_only_loader: Callable[[Path], Any] | None = None,
) -> CheckpointSafetyReport:
    """Return non-raising byte-level checkpoint safety evidence."""

    checkpoint_path = Path(path)
    file_exists = checkpoint_path.is_file()
    checkpoint_hash = file_sha256(checkpoint_path) if file_exists else None
    byte_read_status = "pass" if checkpoint_hash is not None else "fail"
    expected_hash = (
        require_sha256_checksum(
            expected_checkpoint_hash,
            "expected_checkpoint_hash",
        )
        if expected_checkpoint_hash is not None
        else None
    )
    if expected_hash is None:
        hash_match_status = "not_checked"
    elif checkpoint_hash is None:
        hash_match_status = "not_checked"
    elif checkpoint_hash == expected_hash:
        hash_match_status = "pass"
    else:
        hash_match_status = "fail"

    loader = weights_only_loader or _torch_weights_only_load
    weights_only_load_status = "fail"
    loader_error_class: str | None = None
    loader_error_summary: str | None = None
    if file_exists and byte_read_status == "pass":
        try:
            loader(checkpoint_path)
            weights_only_load_status = "pass"
        except Exception as exc:  # noqa: BLE001 - any loader failure means unsafe or unproven.
            loader_error_class = type(exc).__name__
            loader_error_summary = type(exc).__name__
    else:
        loader_error_class = "FileNotFoundError"
        loader_error_summary = "FileNotFoundError"

    reason = "checkpoint_safe_to_load"
    status = "pass"
    if not file_exists:
        status = "fail"
        reason = "checkpoint_missing"
    elif byte_read_status != "pass":
        status = "fail"
        reason = "checkpoint_byte_read_failed"
    elif hash_match_status == "fail":
        status = "fail"
        reason = "checkpoint_hash_mismatch"
    elif weights_only_load_status != "pass":
        status = "fail"
        reason = "checkpoint_weights_only_load_failed"

    return CheckpointSafetyReport(
        status=status,
        checkpoint_hash=checkpoint_hash,
        expected_checkpoint_hash=expected_hash,
        file_exists=file_exists,
        byte_read_status=byte_read_status,
        weights_only_load_status=weights_only_load_status,
        checkpoint_hash_match_status=hash_match_status,
        loader_error_class=loader_error_class,
        loader_error_summary=loader_error_summary,
        reason=reason,
    )


def validate_checkpoint_safety_report(
    report: Any,
    *,
    expected_checkpoint_hash: str | None = None,
) -> dict[str, Any]:
    """Fail closed unless checkpoint byte safety evidence proves a safe load."""

    if not isinstance(report, Mapping):
        raise ValueError("checkpoint_safety_report must be an object")
    required = (
        "schema_version",
        "status",
        "checkpoint_hash",
        "expected_checkpoint_hash",
        "file_exists",
        "byte_read_status",
        "weights_only_load_status",
        "checkpoint_hash_match_status",
        "loader_error_class",
        "loader_error_summary",
        "reason",
    )
    for key in required:
        if key not in report:
            raise ValueError(f"checkpoint_safety_report {key} is required")
    if report["schema_version"] != CHECKPOINT_SAFETY_REPORT_SCHEMA_VERSION:
        raise ValueError("checkpoint_safety_report schema_version mismatch")
    normalized = dict(report)
    if normalized["status"] != "pass":
        raise ValueError(f"checkpoint safety failed: {normalized.get('reason')}")
    if normalized["file_exists"] is not True:
        raise ValueError("checkpoint safety failed: checkpoint_missing")
    if normalized["byte_read_status"] != "pass":
        raise ValueError("checkpoint safety failed: checkpoint_byte_read_failed")
    if normalized["weights_only_load_status"] != "pass":
        raise ValueError("checkpoint safety failed: checkpoint_weights_only_load_failed")
    if normalized["checkpoint_hash_match_status"] == "fail":
        raise ValueError("checkpoint safety failed: checkpoint_hash_mismatch")
    checkpoint_hash = require_sha256_checksum(
        normalized["checkpoint_hash"],
        "checkpoint_safety_report checkpoint_hash",
    )
    expected_hash = normalized["expected_checkpoint_hash"]
    if expected_hash is not None:
        expected_hash = require_sha256_checksum(
            expected_hash,
            "checkpoint_safety_report expected_checkpoint_hash",
        )
        if checkpoint_hash != expected_hash:
            raise ValueError(
                "checkpoint safety report checkpoint hash does not match "
                "its expected hash"
            )
        if normalized["checkpoint_hash_match_status"] != "pass":
            raise ValueError(
                "checkpoint safety report checkpoint_hash_match_status must be "
                "pass when an expected hash is present"
            )
    elif normalized["checkpoint_hash_match_status"] != "not_checked":
        raise ValueError(
            "checkpoint safety report checkpoint_hash_match_status must be "
            "not_checked without an expected hash"
        )
    if expected_checkpoint_hash is not None:
        expected_checkpoint_hash = require_sha256_checksum(
            expected_checkpoint_hash,
            "expected_checkpoint_hash",
        )
        if checkpoint_hash != expected_checkpoint_hash:
            raise ValueError("checkpoint safety report hash does not match eval checkpoint")
        if expected_hash != expected_checkpoint_hash:
            raise ValueError("checkpoint safety report expected hash does not match eval checkpoint")
        if normalized["checkpoint_hash_match_status"] != "pass":
            raise ValueError("checkpoint safety report checkpoint_hash_match_status must be pass")
    for key in ("loader_error_class", "loader_error_summary", "reason"):
        value = normalized.get(key)
        if value is not None and ("://" in str(value) or "/" in str(value) or "\\" in str(value)):
            raise ValueError(f"checkpoint_safety_report {key} must be public-safe")
    return normalized


def file_sha256(path: str | Path) -> str | None:
    """Return a canonical sha256 checksum for a local file, or ``None`` on read failure."""

    path = Path(path)
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return "sha256:" + digest.hexdigest()


def validate_train_checkpoint_artifact(
    checkpoint: CheckpointRef,
    *,
    expected_path: str | Path,
    declared_path: str | Path,
) -> CheckpointRef:
    """Bind a train summary checkpoint to the runner-planned local artifact bytes."""

    expected = Path(os.path.abspath(expected_path))
    declared = Path(os.path.abspath(declared_path))
    if declared != expected:
        raise ValueError(
            "train summary checkpoint path mismatch: "
            f"expected={str(expected)!r} actual={str(declared)!r}"
        )
    if expected.is_symlink():
        raise ValueError(f"train checkpoint output must not be a symlink: {expected}")
    if not expected.is_file():
        raise ValueError(f"train checkpoint output was not created: {expected}")
    actual_checksum = file_sha256(expected)
    if actual_checksum is None:
        raise ValueError(f"train checkpoint output could not be read: {expected}")
    if checkpoint.checksum != actual_checksum:
        raise ValueError(
            "train summary checkpoint checksum does not match output bytes: "
            f"expected={checkpoint.checksum!r} actual={actual_checksum!r}"
        )
    return CheckpointRef(
        uri=checkpoint.uri,
        checksum=checkpoint.checksum,
        role=checkpoint.role,
        sealed=checkpoint.sealed,
        metadata={**dict(checkpoint.metadata), "local_path": str(expected)},
    )


def _torch_weights_only_load(path: Path) -> Any:
    import torch  # type: ignore[import-not-found]  # noqa: PLC0415

    return torch.load(path, map_location="cpu", weights_only=True)
