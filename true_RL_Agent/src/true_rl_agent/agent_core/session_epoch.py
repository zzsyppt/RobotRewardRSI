"""Task-neutral writable ledger epochs anchored to an immutable Genesis."""

from __future__ import annotations

import os
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from true_rl_agent.agent_core.ledger import load_ledger
from true_rl_agent.agent_core.process_lock import (
    ProcessFileLock,
    ProcessFileLockUnavailableError,
    acquire_process_file_lock,
)
from true_rl_agent.agent_core.session_genesis import (
    SESSION_GENESIS_FILENAME,
    VerifiedSessionGenesis,
)
from true_rl_agent.evaluation.binding import (
    canonical_json_hash,
    require_sha256_checksum,
    require_sha256_digest,
)
from true_rl_agent.evaluation.subject import validate_experiment_subject_id
from true_rl_agent.fs_contract import create_once_bytes
from true_rl_agent.json_contract import canonical_json_bytes, strict_json_object_loads
from true_rl_agent.runner.checkpoint import file_sha256

LEDGER_EPOCH_SCHEMA_VERSION = "true_rl_agent.ledger_epoch.v1"
LEDGER_EPOCH_FILENAME = "ledger_epoch.json"


class LiveSessionInProgressError(RuntimeError):
    """Raised when another process owns the writable live-session epoch."""


@dataclass(frozen=True)
class _BoundFile:
    path: str
    checksum: str
    size_bytes: int

    @classmethod
    def capture(cls, path: Path, *, field_name: str) -> _BoundFile:
        if path.is_symlink():
            raise ValueError(f"{field_name} must not be a symlink")
        resolved = path.resolve()
        if not resolved.is_file():
            raise ValueError(f"{field_name} must be a regular file")
        checksum = file_sha256(resolved)
        if checksum is None:
            raise ValueError(f"failed to hash {field_name}")
        return cls(
            path=str(resolved),
            checksum=checksum,
            size_bytes=resolved.stat().st_size,
        )

    @classmethod
    def from_mapping(cls, value: Any, *, field_name: str) -> _BoundFile:
        if not isinstance(value, Mapping) or set(value) != {
            "path",
            "checksum",
            "size_bytes",
        }:
            raise ValueError(f"{field_name} fields mismatch")
        path = value.get("path")
        if not isinstance(path, str) or not path:
            raise ValueError(f"{field_name} path is invalid")
        checksum = require_sha256_checksum(
            value.get("checksum"),
            f"{field_name} checksum",
        )
        size_bytes = value.get("size_bytes")
        if (
            isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes < 0
        ):
            raise ValueError(f"{field_name} size_bytes is invalid")
        return cls(path=path, checksum=checksum, size_bytes=size_bytes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "checksum": self.checksum,
            "size_bytes": self.size_bytes,
        }

    def assert_unchanged(self, *, field_name: str) -> None:
        path = Path(self.path)
        if path.resolve() != path or path.is_symlink() or not path.is_file():
            raise ValueError(f"{field_name} path drift")
        if path.stat().st_size != self.size_bytes:
            raise ValueError(f"{field_name} size mismatch")
        if file_sha256(path) != self.checksum:
            raise ValueError(f"{field_name} checksum mismatch")


@dataclass(frozen=True)
class _PredecessorLedger:
    path: str
    status: str
    checksum: str | None
    size_bytes: int | None

    @classmethod
    def capture(
        cls,
        path: Path,
        *,
        expected_session_genesis_hash: str,
    ) -> _PredecessorLedger:
        if path.is_symlink():
            raise ValueError("predecessor ledger must not be a symlink")
        resolved = path.resolve(strict=False)
        if resolved.exists():
            _assert_predecessor_preserves_genesis_head(
                resolved,
                expected_session_genesis_hash=(
                    expected_session_genesis_hash
                ),
            )
            bound = _BoundFile.capture(
                resolved,
                field_name="predecessor ledger",
            )
            return cls(
                path=bound.path,
                status="bound",
                checksum=bound.checksum,
                size_bytes=bound.size_bytes,
            )
        return cls(
            path=str(resolved),
            status="absent",
            checksum=None,
            size_bytes=None,
        )

    @classmethod
    def from_mapping(cls, value: Any) -> _PredecessorLedger:
        if not isinstance(value, Mapping) or set(value) != {
            "path",
            "status",
            "checksum",
            "size_bytes",
        }:
            raise ValueError("predecessor ledger fields mismatch")
        path = value.get("path")
        status = value.get("status")
        if not isinstance(path, str) or not path:
            raise ValueError("predecessor ledger path is invalid")
        if status == "bound":
            checksum = require_sha256_checksum(
                value.get("checksum"),
                "predecessor ledger checksum",
            )
            size_bytes = value.get("size_bytes")
            if (
                isinstance(size_bytes, bool)
                or not isinstance(size_bytes, int)
                or size_bytes < 0
            ):
                raise ValueError("predecessor ledger size_bytes is invalid")
        elif status == "absent":
            if value.get("checksum") is not None or value.get("size_bytes") is not None:
                raise ValueError("absent predecessor ledger carries byte identity")
            checksum = None
            size_bytes = None
        else:
            raise ValueError("predecessor ledger status mismatch")
        return cls(
            path=path,
            status=status,
            checksum=checksum,
            size_bytes=size_bytes,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "status": self.status,
            "checksum": self.checksum,
            "size_bytes": self.size_bytes,
        }

    def assert_unchanged(self) -> None:
        path = Path(self.path)
        if path.resolve(strict=False) != path or path.is_symlink():
            raise ValueError("predecessor ledger path drift")
        if self.status == "absent":
            if path.exists():
                raise ValueError("predecessor ledger appeared after epoch creation")
            return
        if not path.is_file():
            raise ValueError("predecessor ledger is missing")
        if path.stat().st_size != self.size_bytes:
            raise ValueError("predecessor ledger size mismatch")
        if file_sha256(path) != self.checksum:
            raise ValueError("predecessor ledger checksum mismatch")


@dataclass(frozen=True)
class VerifiedLedgerEpoch:
    """Validated link from one immutable Genesis to a fresh writable ledger."""

    epoch_id: str
    session_id: str
    session_genesis_hash: str
    genesis_artifact: _BoundFile
    predecessor_ledger: _PredecessorLedger
    ledger_path: str
    manifest_hash: str
    manifest_path: Path

    @classmethod
    def from_mapping(
        cls,
        value: Any,
        *,
        manifest_path: Path,
    ) -> VerifiedLedgerEpoch:
        if not isinstance(value, Mapping) or set(value) != {
            "schema_version",
            "epoch_id",
            "session_id",
            "session_genesis_hash",
            "genesis_artifact",
            "predecessor_ledger",
            "ledger_path",
            "manifest_hash",
        }:
            raise ValueError("ledger epoch fields mismatch")
        if value.get("schema_version") != LEDGER_EPOCH_SCHEMA_VERSION:
            raise ValueError("ledger epoch schema_version mismatch")
        epoch_id = validate_experiment_subject_id(value.get("epoch_id"))
        session_id = validate_experiment_subject_id(value.get("session_id"))
        session_genesis_hash = require_sha256_digest(
            value.get("session_genesis_hash"),
            "ledger epoch session_genesis_hash",
        )
        ledger_path = value.get("ledger_path")
        if not isinstance(ledger_path, str) or not ledger_path:
            raise ValueError("ledger epoch ledger_path is invalid")
        manifest_hash = require_sha256_digest(
            value.get("manifest_hash"),
            "ledger epoch manifest_hash",
        )
        base = {key: item for key, item in value.items() if key != "manifest_hash"}
        if canonical_json_hash(base) != manifest_hash:
            raise ValueError("ledger epoch manifest_hash mismatch")
        return cls(
            epoch_id=epoch_id,
            session_id=session_id,
            session_genesis_hash=session_genesis_hash,
            genesis_artifact=_BoundFile.from_mapping(
                value.get("genesis_artifact"),
                field_name="ledger epoch Genesis artifact",
            ),
            predecessor_ledger=_PredecessorLedger.from_mapping(
                value.get("predecessor_ledger")
            ),
            ledger_path=ledger_path,
            manifest_hash=manifest_hash,
            manifest_path=manifest_path.resolve(strict=False),
        )

    def to_dict(self) -> dict[str, Any]:
        base = {
            "schema_version": LEDGER_EPOCH_SCHEMA_VERSION,
            "epoch_id": self.epoch_id,
            "session_id": self.session_id,
            "session_genesis_hash": self.session_genesis_hash,
            "genesis_artifact": self.genesis_artifact.to_dict(),
            "predecessor_ledger": self.predecessor_ledger.to_dict(),
            "ledger_path": self.ledger_path,
        }
        return {**base, "manifest_hash": canonical_json_hash(base)}

    def assert_unchanged(
        self,
        *,
        output_dir: Path,
        genesis: VerifiedSessionGenesis,
    ) -> None:
        expected_manifest = output_dir.resolve() / LEDGER_EPOCH_FILENAME
        if self.manifest_path != expected_manifest:
            raise ValueError("ledger epoch belongs to a different output directory")
        actual = _load_epoch_manifest(expected_manifest)
        if actual.to_dict() != self.to_dict():
            raise ValueError("ledger epoch manifest changed after verification")
        if self.session_genesis_hash != genesis.session_genesis_hash:
            raise ValueError("ledger epoch Genesis hash mismatch")
        if Path(self.genesis_artifact.path) != genesis.artifact_path.resolve():
            raise ValueError("ledger epoch Genesis path mismatch")
        expected_ledger = output_dir.resolve() / "ledger.jsonl"
        if Path(self.ledger_path) != expected_ledger:
            raise ValueError("ledger epoch writable ledger path mismatch")
        if expected_ledger.is_symlink() or (
            expected_ledger.exists() and not expected_ledger.is_file()
        ):
            raise ValueError("ledger epoch writable ledger must be a regular file")
        self.genesis_artifact.assert_unchanged(
            field_name="ledger epoch Genesis artifact"
        )
        self.predecessor_ledger.assert_unchanged()

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "schema_version": LEDGER_EPOCH_SCHEMA_VERSION,
            "epoch_id": self.epoch_id,
            "manifest_path": str(self.manifest_path),
            "manifest_hash": self.manifest_hash,
            "predecessor_ledger_status": self.predecessor_ledger.status,
        }


def prepare_ledger_epoch(
    *,
    output_dir: Path,
    genesis_dir: Path,
    epoch_id: str | None,
    session_id: str,
    genesis: VerifiedSessionGenesis,
) -> VerifiedLedgerEpoch | None:
    """Create or verify a fresh writable ledger root for an existing Genesis."""

    output_root = output_dir.resolve()
    genesis_root = genesis_dir.resolve()
    if output_root == genesis_root:
        if epoch_id is not None:
            raise ValueError(
                "ledger_epoch_id is only valid with a separate session Genesis directory"
            )
        return None
    if epoch_id is None:
        raise ValueError(
            "ledger_epoch_id is required with a separate session Genesis directory"
        )
    epoch_id = validate_experiment_subject_id(epoch_id)
    session_id = validate_experiment_subject_id(session_id)
    if genesis.artifact.session_id != session_id:
        raise ValueError("ledger epoch session_id does not match Genesis")
    genesis.assert_file_unchanged(genesis_root)

    manifest_path = output_root / LEDGER_EPOCH_FILENAME
    expected = _build_epoch(
        epoch_id=epoch_id,
        session_id=session_id,
        genesis=genesis,
        predecessor_ledger_path=genesis_root / "ledger.jsonl",
        ledger_path=output_root / "ledger.jsonl",
        manifest_path=manifest_path,
    )
    if not manifest_path.exists():
        _reject_unbound_epoch_state(output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        try:
            create_once_bytes(
                manifest_path,
                canonical_json_bytes(expected.to_dict(), trailing_newline=True),
            )
        except FileExistsError:
            pass
    actual = _load_epoch_manifest(manifest_path)
    if actual.to_dict() != expected.to_dict():
        raise ValueError("ledger epoch differs from the requested immutable binding")
    actual.assert_unchanged(output_dir=output_root, genesis=genesis)
    return actual


def acquire_live_session_lease(
    *,
    output_dir: Path,
    session_id: str,
    session_genesis_hash: str,
    ledger_epoch_hash: str,
) -> ProcessFileLock:
    payload = canonical_json_bytes(
        {
            "schema_version": "true_rl_agent.live_session_lease.v1",
            "session_id": validate_experiment_subject_id(session_id),
            "session_genesis_hash": require_sha256_digest(
                session_genesis_hash,
                "live session Genesis hash",
            ),
            "ledger_epoch_hash": require_sha256_digest(
                ledger_epoch_hash,
                "live session ledger epoch hash",
            ),
            "process_id": os.getpid(),
            "owner_token": uuid.uuid4().hex,
            "ownership_model": "process_held_os_file_lock",
        },
        trailing_newline=True,
    )
    try:
        return acquire_process_file_lock(
            output_dir.resolve() / ".live_session.lease",
            blocking=False,
            payload=payload,
        )
    except ProcessFileLockUnavailableError as exc:
        raise LiveSessionInProgressError(
            f"live session already in progress: {session_id}"
        ) from exc


def _build_epoch(
    *,
    epoch_id: str,
    session_id: str,
    genesis: VerifiedSessionGenesis,
    predecessor_ledger_path: Path,
    ledger_path: Path,
    manifest_path: Path,
) -> VerifiedLedgerEpoch:
    genesis_artifact = _BoundFile.capture(
        genesis.artifact_path,
        field_name="session Genesis",
    )
    predecessor = _PredecessorLedger.capture(
        predecessor_ledger_path,
        expected_session_genesis_hash=genesis.session_genesis_hash,
    )
    base = {
        "schema_version": LEDGER_EPOCH_SCHEMA_VERSION,
        "epoch_id": epoch_id,
        "session_id": session_id,
        "session_genesis_hash": genesis.session_genesis_hash,
        "genesis_artifact": genesis_artifact.to_dict(),
        "predecessor_ledger": predecessor.to_dict(),
        "ledger_path": str(ledger_path.resolve(strict=False)),
    }
    return VerifiedLedgerEpoch(
        epoch_id=epoch_id,
        session_id=session_id,
        session_genesis_hash=genesis.session_genesis_hash,
        genesis_artifact=genesis_artifact,
        predecessor_ledger=predecessor,
        ledger_path=str(ledger_path.resolve(strict=False)),
        manifest_hash=canonical_json_hash(base),
        manifest_path=manifest_path.resolve(strict=False),
    )


def _load_epoch_manifest(path: Path) -> VerifiedLedgerEpoch:
    if path.is_symlink() or not path.is_file():
        raise ValueError("ledger epoch manifest must be a regular file")
    payload = path.read_bytes()
    parsed = strict_json_object_loads(payload, context="ledger epoch manifest")
    if payload != canonical_json_bytes(parsed, trailing_newline=True):
        raise ValueError("ledger epoch manifest is not canonical JSON")
    return VerifiedLedgerEpoch.from_mapping(parsed, manifest_path=path)


def _assert_predecessor_preserves_genesis_head(
    path: Path,
    *,
    expected_session_genesis_hash: str,
) -> None:
    """Allow a fresh epoch only when the predecessor never advanced Genesis."""

    expected_hash = require_sha256_digest(
        expected_session_genesis_hash,
        "predecessor ledger session Genesis hash",
    )
    for index, record in enumerate(load_ledger(path), start=1):
        if record.get("session_genesis_hash") != expected_hash:
            raise ValueError(
                "predecessor ledger belongs to a different session Genesis: "
                f"record={index}"
            )
        decision = record.get("decision")
        if not isinstance(decision, Mapping) or decision.get("accepted") is not False:
            raise ValueError(
                "predecessor ledger may have advanced the Genesis current-best; "
                "start a new session or implement an explicit head migration"
            )


def _reject_unbound_epoch_state(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise ValueError("ledger epoch output_dir must be a regular directory")
    protected = (
        "ledger.jsonl",
        "rounds",
        "runner",
        "candidates",
        "candidate_artifacts",
        "live_summary.json",
        SESSION_GENESIS_FILENAME,
    )
    if any((output_dir / name).exists() for name in protected):
        raise ValueError("ledger epoch output_dir already contains unbound live state")


__all__ = [
    "LEDGER_EPOCH_FILENAME",
    "LEDGER_EPOCH_SCHEMA_VERSION",
    "LiveSessionInProgressError",
    "VerifiedLedgerEpoch",
    "acquire_live_session_lease",
    "prepare_ledger_epoch",
]
