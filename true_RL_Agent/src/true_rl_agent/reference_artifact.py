"""Runner-owned immutable artifacts for trusted reference programs."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from true_rl_agent.evaluation.binding import (
    canonical_json_hash,
    require_sha256_checksum,
    require_sha256_digest,
)
from true_rl_agent.json_contract import canonical_json_bytes, load_strict_json_object
from true_rl_agent.training_programs import (
    CANONICAL_TRAINING_COMPONENTS_PATH,
    ReferenceProgramManifest,
    VerifiedReferenceProgramArtifact,
    verify_reference_program_artifact_bytes,
)

REFERENCE_ARTIFACT_SCHEMA_VERSION = (
    "true_rl_agent.materialized_reference_program.v1"
)
REFERENCE_ARTIFACT_MANIFEST_FILENAME = (
    ".true_rl_agent_reference_artifact.json"
)


@dataclass(frozen=True, init=False)
class VerifiedReferenceProgramArtifactRef:
    """Capability proving that a local reference artifact matches trusted bytes."""

    path: str
    artifact_hash: str
    program: VerifiedReferenceProgramArtifact

    _TOKEN = object()

    def __init__(
        self,
        *,
        path: str,
        artifact_hash: str,
        program: VerifiedReferenceProgramArtifact,
        _token: object | None = None,
    ) -> None:
        if _token is not self._TOKEN:
            raise TypeError(
                "verified reference artifact refs must come from the artifact loader"
            )
        normalized_path = path.replace("\\", "/") if isinstance(path, str) else ""
        if (
            not isinstance(path, str)
            or not path
            or normalized_path.startswith("//")
        ):
            raise ValueError("reference artifact path must be a non-empty local path")
        resolved = Path(path)
        if not resolved.is_absolute() or "://" in path:
            raise ValueError("reference artifact path must be absolute and local")
        require_sha256_digest(artifact_hash, "reference artifact hash")
        if not isinstance(program, VerifiedReferenceProgramArtifact):
            raise TypeError("reference artifact program capability is invalid")
        object.__setattr__(self, "path", str(resolved))
        object.__setattr__(self, "artifact_hash", artifact_hash)
        object.__setattr__(self, "program", program)

    @classmethod
    def _validated(
        cls,
        *,
        path: Path,
        artifact_hash: str,
        program: VerifiedReferenceProgramArtifact,
    ) -> "VerifiedReferenceProgramArtifactRef":
        return cls(
            path=str(path),
            artifact_hash=artifact_hash,
            program=program,
            _token=cls._TOKEN,
        )

    @property
    def reference_id(self) -> str:
        return self.program.manifest.reference_id

    @property
    def task_id(self) -> str:
        return self.program.manifest.task_id

    @property
    def subject_manifest_hash(self) -> str:
        assert self.program.manifest.manifest_hash is not None
        return self.program.manifest.manifest_hash

    @property
    def program_source_checksum(self) -> str:
        return self.program.manifest.source_checksum

    @property
    def program_source_size_bytes(self) -> int:
        return self.program.manifest.source_size_bytes

    @property
    def verification_hash(self) -> str:
        return self.program.verification_hash

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "reference_id": self.reference_id,
            "task_id": self.task_id,
            "subject_manifest_hash": self.subject_manifest_hash,
            "program_source_checksum": self.program_source_checksum,
            "program_source_size_bytes": self.program_source_size_bytes,
            "artifact_hash": self.artifact_hash,
            "verification_hash": self.verification_hash,
        }


def materialize_reference_program_artifact(
    program: VerifiedReferenceProgramArtifact,
    *,
    artifact_root: str | Path,
) -> VerifiedReferenceProgramArtifactRef:
    """Publish exact trusted source bytes into a content-addressed local directory."""

    if not isinstance(program, VerifiedReferenceProgramArtifact):
        raise TypeError(
            "reference artifact materialization requires a verified program"
        )
    unresolved_root = Path(artifact_root)
    if unresolved_root.is_symlink():
        raise ValueError("reference artifact root must not be a symlink")
    destination_root = unresolved_root.resolve()
    destination_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=".reference-artifact-", dir=destination_root)
    )
    try:
        source_path = temporary / CANONICAL_TRAINING_COMPONENTS_PATH
        source_path.parent.mkdir(parents=True, exist_ok=True)
        _write_bytes_fsync(source_path, program.source_bytes)
        files = _reference_file_evidence(temporary)
        payload = {
            "schema_version": REFERENCE_ARTIFACT_SCHEMA_VERSION,
            "reference_manifest": program.manifest.to_public_dict(),
            "verification_hash": program.verification_hash,
            "files": files,
        }
        artifact_hash = canonical_json_hash(payload)
        manifest = {**payload, "artifact_hash": artifact_hash}
        _write_bytes_fsync(
            temporary / REFERENCE_ARTIFACT_MANIFEST_FILENAME,
            canonical_json_bytes(manifest, trailing_newline=True),
        )

        final_parent = destination_root / program.manifest.reference_id
        final_parent.mkdir(parents=True, exist_ok=True)
        if final_parent.is_symlink():
            raise ValueError("reference artifact subject directory must not be a symlink")
        final_path = final_parent / artifact_hash
        if final_path.exists():
            return load_reference_program_artifact(
                final_path,
                expected_program=program,
                expected_artifact_hash=artifact_hash,
            )
        try:
            os.rename(temporary, final_path)
        except OSError:
            if not final_path.exists():
                raise
        return load_reference_program_artifact(
            final_path,
            expected_program=program,
            expected_artifact_hash=artifact_hash,
        )
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def load_reference_program_artifact(
    artifact_path: str | Path,
    *,
    expected_program: VerifiedReferenceProgramArtifact | None = None,
    expected_artifact_hash: str | None = None,
    expected_reference_id: str | None = None,
    expected_task_id: str | None = None,
    expected_subject_manifest_hash: str | None = None,
    expected_program_source_checksum: str | None = None,
) -> VerifiedReferenceProgramArtifactRef:
    unresolved = Path(artifact_path)
    if unresolved.is_symlink():
        raise ValueError("reference artifact path must not be a symlink")
    root = unresolved.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("reference artifact must be a directory")
    manifest_payload = load_strict_json_object(
        root / REFERENCE_ARTIFACT_MANIFEST_FILENAME,
        context="reference artifact manifest",
        require_canonical=True,
        trailing_newline=True,
    )
    expected_fields = {
        "schema_version",
        "reference_manifest",
        "verification_hash",
        "files",
        "artifact_hash",
    }
    if set(manifest_payload) != expected_fields:
        raise ValueError("reference artifact manifest fields mismatch")
    if manifest_payload.get("schema_version") != REFERENCE_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("reference artifact schema version mismatch")
    artifact_hash = require_sha256_digest(
        manifest_payload.get("artifact_hash"),
        "reference artifact hash",
    )
    payload = {
        key: value
        for key, value in manifest_payload.items()
        if key != "artifact_hash"
    }
    if canonical_json_hash(payload) != artifact_hash:
        raise ValueError("reference artifact hash mismatch")
    if expected_artifact_hash is not None:
        require_sha256_digest(
            expected_artifact_hash,
            "expected reference artifact hash",
        )
        if artifact_hash != expected_artifact_hash:
            raise ValueError("reference artifact does not match expected hash")

    declared_files = _validated_reference_file_evidence(
        manifest_payload.get("files")
    )
    if _reference_file_evidence(root) != declared_files:
        raise ValueError("reference artifact file evidence mismatch")
    source_path = root / CANONICAL_TRAINING_COMPONENTS_PATH
    source_bytes = source_path.read_bytes()
    manifest = ReferenceProgramManifest.from_mapping(
        manifest_payload.get("reference_manifest")
    )
    verification_hash = require_sha256_digest(
        manifest_payload.get("verification_hash"),
        "reference artifact verification hash",
    )
    program = verify_reference_program_artifact_bytes(
        manifest=manifest,
        source_bytes=source_bytes,
        expected_verification_hash=verification_hash,
    )
    _assert_expected_program_identity(
        program,
        expected_program=expected_program,
        expected_reference_id=expected_reference_id,
        expected_task_id=expected_task_id,
        expected_subject_manifest_hash=expected_subject_manifest_hash,
        expected_program_source_checksum=expected_program_source_checksum,
    )
    return VerifiedReferenceProgramArtifactRef._validated(
        path=root,
        artifact_hash=artifact_hash,
        program=program,
    )


def verify_reference_program_artifact_ref(
    artifact: VerifiedReferenceProgramArtifactRef,
) -> VerifiedReferenceProgramArtifactRef:
    if not isinstance(artifact, VerifiedReferenceProgramArtifactRef):
        raise TypeError(
            "reference execution requires a verified artifact reference"
        )
    verified = load_reference_program_artifact(
        artifact.path,
        expected_program=artifact.program,
        expected_artifact_hash=artifact.artifact_hash,
    )
    if verified != artifact:
        raise ValueError("reference artifact changed during verification")
    return artifact


def _assert_expected_program_identity(
    program: VerifiedReferenceProgramArtifact,
    *,
    expected_program: VerifiedReferenceProgramArtifact | None,
    expected_reference_id: str | None,
    expected_task_id: str | None,
    expected_subject_manifest_hash: str | None,
    expected_program_source_checksum: str | None,
) -> None:
    if expected_program is not None and program != expected_program:
        raise ValueError("reference artifact differs from expected program")
    manifest = program.manifest
    expected = {
        "reference_id": expected_reference_id,
        "task_id": expected_task_id,
        "manifest_hash": expected_subject_manifest_hash,
        "source_checksum": expected_program_source_checksum,
    }
    actual = {
        "reference_id": manifest.reference_id,
        "task_id": manifest.task_id,
        "manifest_hash": manifest.manifest_hash,
        "source_checksum": manifest.source_checksum,
    }
    for key, expected_value in expected.items():
        if expected_value is not None and actual[key] != expected_value:
            raise ValueError(f"reference artifact {key} mismatch")


def _reference_file_evidence(root: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if path.is_symlink():
            raise ValueError(f"reference artifact contains symlink: {relative}")
        if not path.is_file():
            continue
        relative_text = relative.as_posix()
        if relative_text == REFERENCE_ARTIFACT_MANIFEST_FILENAME:
            continue
        payload = path.read_bytes()
        result.append(
            {
                "path": relative_text,
                "checksum": "sha256:" + hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        )
    if [item["path"] for item in result] != [CANONICAL_TRAINING_COMPONENTS_PATH]:
        raise ValueError("reference artifact must contain only canonical source bytes")
    return result


def _validated_reference_file_evidence(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != 1:
        raise ValueError("reference artifact file evidence is invalid")
    item = value[0]
    if not isinstance(item, Mapping) or set(item) != {
        "path",
        "checksum",
        "size_bytes",
    }:
        raise ValueError("reference artifact file evidence fields mismatch")
    if item.get("path") != CANONICAL_TRAINING_COMPONENTS_PATH:
        raise ValueError("reference artifact source path is not canonical")
    checksum = require_sha256_checksum(
        item.get("checksum"),
        "reference artifact source checksum",
    )
    size = item.get("size_bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError("reference artifact source size must be positive")
    return [
        {
            "path": CANONICAL_TRAINING_COMPONENTS_PATH,
            "checksum": checksum,
            "size_bytes": size,
        }
    ]


def _write_bytes_fsync(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


__all__ = [
    "REFERENCE_ARTIFACT_MANIFEST_FILENAME",
    "REFERENCE_ARTIFACT_SCHEMA_VERSION",
    "VerifiedReferenceProgramArtifactRef",
    "load_reference_program_artifact",
    "materialize_reference_program_artifact",
    "verify_reference_program_artifact_ref",
]
