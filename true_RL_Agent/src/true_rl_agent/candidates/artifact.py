"""Runner-owned, content-addressed candidate execution artifacts."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from true_rl_agent.candidates.schema import CandidateChangeSet, CandidatePackageRef
from true_rl_agent.evaluation.binding import (
    canonical_json_hash,
    require_sha256_checksum,
    require_sha256_digest,
)
from true_rl_agent.json_contract import (
    canonical_json_bytes,
    load_strict_json_object,
)
from true_rl_agent.training_programs import resolve_candidate_program_identity

CANDIDATE_ARTIFACT_SCHEMA_VERSION = (
    "true_rl_agent.sealed_candidate_artifact.v1"
)
CANDIDATE_ARTIFACT_MANIFEST_FILENAME = (
    ".true_rl_agent_candidate_artifact.json"
)


@dataclass(frozen=True)
class SealedCandidateArtifactRef:
    """Typed reference to runner-owned candidate bytes, not a provider directory."""

    candidate_id: str
    path: str
    change_set: CandidateChangeSet
    subject_manifest_hash: str
    program_source_checksum: str
    program_source_size_bytes: int
    artifact_hash: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id:
            raise ValueError("sealed candidate artifact candidate_id is required")
        if not isinstance(self.path, str) or not self.path or "://" in self.path:
            raise ValueError("sealed candidate artifact path must be a local path")
        if not Path(self.path).is_absolute():
            raise ValueError("sealed candidate artifact path must be absolute")
        if not isinstance(self.change_set, CandidateChangeSet):
            raise TypeError("sealed candidate artifact change_set is invalid")
        require_sha256_digest(
            self.subject_manifest_hash,
            "sealed candidate subject manifest hash",
        )
        require_sha256_checksum(
            self.program_source_checksum,
            "sealed candidate program source checksum",
        )
        if (
            isinstance(self.program_source_size_bytes, bool)
            or not isinstance(self.program_source_size_bytes, int)
            or self.program_source_size_bytes <= 0
        ):
            raise ValueError("sealed candidate source size must be positive")
        require_sha256_digest(
            self.artifact_hash,
            "sealed candidate artifact hash",
        )
        if not isinstance(self.metadata, Mapping):
            raise TypeError("sealed candidate metadata must be an object")

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "changed_components": sorted(self.change_set.changed_components),
            "subject_manifest_hash": self.subject_manifest_hash,
            "program_source_checksum": self.program_source_checksum,
            "program_source_size_bytes": self.program_source_size_bytes,
            "artifact_hash": self.artifact_hash,
        }


@dataclass(frozen=True, init=False)
class VerifiedCandidateArtifactRef:
    """Capability proving that one sealed artifact passed a bound verifier."""

    sealed_artifact: SealedCandidateArtifactRef
    verification_policy: str
    verification_hash: str

    _TOKEN = object()

    def __init__(
        self,
        *,
        sealed_artifact: SealedCandidateArtifactRef,
        verification_policy: str,
        verification_hash: str,
        _token: object | None = None,
    ) -> None:
        if _token is not self._TOKEN:
            raise TypeError(
                "verified candidate artifacts must be created by the verification pipeline"
            )
        if not isinstance(sealed_artifact, SealedCandidateArtifactRef):
            raise TypeError("verified candidate sealed artifact is invalid")
        if not isinstance(verification_policy, str) or not verification_policy:
            raise ValueError("verified candidate verification_policy is required")
        require_sha256_digest(
            verification_hash,
            "verified candidate verification hash",
        )
        object.__setattr__(self, "sealed_artifact", sealed_artifact)
        object.__setattr__(self, "verification_policy", verification_policy)
        object.__setattr__(self, "verification_hash", verification_hash)

    @classmethod
    def _validated(
        cls,
        *,
        sealed_artifact: SealedCandidateArtifactRef,
        verification_policy: str,
        verification_hash: str,
    ) -> "VerifiedCandidateArtifactRef":
        return cls(
            sealed_artifact=sealed_artifact,
            verification_policy=verification_policy,
            verification_hash=verification_hash,
            _token=cls._TOKEN,
        )

    @property
    def candidate_id(self) -> str:
        return self.sealed_artifact.candidate_id

    @property
    def path(self) -> str:
        return self.sealed_artifact.path

    @property
    def change_set(self) -> CandidateChangeSet:
        return self.sealed_artifact.change_set

    @property
    def subject_manifest_hash(self) -> str:
        return self.sealed_artifact.subject_manifest_hash

    @property
    def program_source_checksum(self) -> str:
        return self.sealed_artifact.program_source_checksum

    @property
    def program_source_size_bytes(self) -> int:
        return self.sealed_artifact.program_source_size_bytes

    @property
    def artifact_hash(self) -> str:
        return self.sealed_artifact.artifact_hash

    @property
    def metadata(self) -> Mapping[str, Any]:
        return self.sealed_artifact.metadata

    def to_public_dict(self) -> dict[str, Any]:
        return {
            **self.sealed_artifact.to_public_dict(),
            "verification": {
                "status": "pass",
                "policy": self.verification_policy,
                "verification_hash": self.verification_hash,
            },
        }


def seal_candidate_package(
    candidate: CandidatePackageRef,
    *,
    artifact_root: str | Path,
) -> SealedCandidateArtifactRef:
    """Snapshot provider output into a content-addressed runner directory."""

    if not isinstance(candidate, CandidatePackageRef):
        raise TypeError("candidate sealing requires a CandidatePackageRef")
    if "://" in candidate.path:
        raise ValueError("candidate sealing requires a local provider directory")
    source_path = Path(candidate.path)
    if source_path.is_symlink():
        raise ValueError("candidate provider path must not be a symlink")
    source_root = source_path.resolve(strict=True)
    if not source_root.is_dir():
        raise ValueError("candidate provider path must be a non-symlink directory")

    destination_root = Path(artifact_root).resolve()
    destination_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=".candidate-artifact-", dir=destination_root)
    )
    try:
        _copy_candidate_snapshot(source_root, temporary)
        identity = resolve_candidate_program_identity(
            temporary,
            expected_candidate_id=candidate.candidate_id,
        )
        files = _file_evidence(temporary)
        payload = {
            "schema_version": CANDIDATE_ARTIFACT_SCHEMA_VERSION,
            "candidate_id": candidate.candidate_id,
            "subject_manifest_hash": identity.manifest_hash,
            "program_source_checksum": identity.source_checksum,
            "program_source_size_bytes": identity.source_size_bytes,
            "files": files,
        }
        artifact_hash = canonical_json_hash(payload)
        manifest = {**payload, "artifact_hash": artifact_hash}
        (temporary / CANDIDATE_ARTIFACT_MANIFEST_FILENAME).write_bytes(
            canonical_json_bytes(manifest, trailing_newline=True)
        )

        final_parent = destination_root / candidate.candidate_id
        final_parent.mkdir(parents=True, exist_ok=True)
        final_path = final_parent / artifact_hash
        if final_path.exists():
            existing = load_sealed_candidate_artifact(
                final_path,
                change_set=candidate.change_set,
                metadata=candidate.metadata,
                expected_candidate_id=candidate.candidate_id,
                expected_artifact_hash=artifact_hash,
            )
            return existing
        try:
            os.rename(temporary, final_path)
        except OSError:
            if not final_path.exists():
                raise
        return load_sealed_candidate_artifact(
            final_path,
            change_set=candidate.change_set,
            metadata=candidate.metadata,
            expected_candidate_id=candidate.candidate_id,
            expected_artifact_hash=artifact_hash,
        )
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def load_sealed_candidate_artifact(
    artifact_path: str | Path,
    *,
    change_set: CandidateChangeSet,
    metadata: Mapping[str, Any] | None = None,
    expected_candidate_id: str | None = None,
    expected_artifact_hash: str | None = None,
    expected_subject_manifest_hash: str | None = None,
    expected_program_source_checksum: str | None = None,
) -> SealedCandidateArtifactRef:
    unresolved_root = Path(artifact_path)
    if unresolved_root.is_symlink():
        raise ValueError("sealed candidate artifact path must not be a symlink")
    root = unresolved_root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("sealed candidate artifact must be a non-symlink directory")
    manifest = load_strict_json_object(
        root / CANDIDATE_ARTIFACT_MANIFEST_FILENAME,
        context="sealed candidate artifact manifest",
        require_canonical=True,
        trailing_newline=True,
    )
    expected_fields = {
        "schema_version",
        "candidate_id",
        "subject_manifest_hash",
        "program_source_checksum",
        "program_source_size_bytes",
        "files",
        "artifact_hash",
    }
    if set(manifest) != expected_fields:
        raise ValueError("sealed candidate artifact manifest fields mismatch")
    if manifest.get("schema_version") != CANDIDATE_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("sealed candidate artifact schema version mismatch")
    artifact_hash = require_sha256_digest(
        manifest.get("artifact_hash"),
        "sealed candidate artifact hash",
    )
    payload = {
        key: value for key, value in manifest.items() if key != "artifact_hash"
    }
    if canonical_json_hash(payload) != artifact_hash:
        raise ValueError("sealed candidate artifact hash mismatch")
    if expected_artifact_hash is not None and artifact_hash != expected_artifact_hash:
        raise ValueError("sealed candidate artifact does not match expected hash")

    candidate_id = manifest.get("candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id:
        raise ValueError("sealed candidate artifact candidate_id is invalid")
    if expected_candidate_id is not None and candidate_id != expected_candidate_id:
        raise ValueError("sealed candidate artifact candidate_id mismatch")
    declared_files = _validated_file_evidence(manifest.get("files"))
    if _file_evidence(root) != declared_files:
        raise ValueError("sealed candidate artifact file evidence mismatch")

    identity = resolve_candidate_program_identity(
        root,
        expected_candidate_id=candidate_id,
    )
    declared_identity = (
        manifest.get("subject_manifest_hash"),
        manifest.get("program_source_checksum"),
        manifest.get("program_source_size_bytes"),
    )
    actual_identity = (
        identity.manifest_hash,
        identity.source_checksum,
        identity.source_size_bytes,
    )
    if declared_identity != actual_identity:
        raise ValueError("sealed candidate artifact program identity mismatch")
    if (
        expected_subject_manifest_hash is not None
        and identity.manifest_hash != expected_subject_manifest_hash
    ):
        raise ValueError("sealed candidate subject manifest hash mismatch")
    if (
        expected_program_source_checksum is not None
        and identity.source_checksum != expected_program_source_checksum
    ):
        raise ValueError("sealed candidate program source checksum mismatch")
    return SealedCandidateArtifactRef(
        candidate_id=candidate_id,
        path=str(root),
        change_set=change_set,
        subject_manifest_hash=identity.manifest_hash,
        program_source_checksum=identity.source_checksum,
        program_source_size_bytes=identity.source_size_bytes,
        artifact_hash=artifact_hash,
        metadata=dict(metadata or {}),
    )


def verify_sealed_candidate_artifact(
    artifact: SealedCandidateArtifactRef,
) -> SealedCandidateArtifactRef:
    if not isinstance(artifact, SealedCandidateArtifactRef):
        raise TypeError("candidate execution requires a sealed artifact reference")
    return load_sealed_candidate_artifact(
        artifact.path,
        change_set=artifact.change_set,
        metadata=artifact.metadata,
        expected_candidate_id=artifact.candidate_id,
        expected_artifact_hash=artifact.artifact_hash,
        expected_subject_manifest_hash=artifact.subject_manifest_hash,
        expected_program_source_checksum=artifact.program_source_checksum,
    )


def verify_verified_candidate_artifact(
    artifact: VerifiedCandidateArtifactRef,
) -> VerifiedCandidateArtifactRef:
    if not isinstance(artifact, VerifiedCandidateArtifactRef):
        raise TypeError("candidate execution requires a verified artifact reference")
    if verify_sealed_candidate_artifact(artifact.sealed_artifact) != (
        artifact.sealed_artifact
    ):
        raise ValueError("verified candidate sealed artifact changed during verification")
    return artifact


def _copy_candidate_snapshot(source_root: Path, destination_root: Path) -> None:
    for source in sorted(source_root.rglob("*")):
        relative = source.relative_to(source_root)
        if source.is_symlink():
            raise ValueError(f"candidate artifact does not allow symlinks: {relative}")
        if not source.is_file():
            continue
        if _ignored_runtime_file(relative):
            continue
        if relative.as_posix() == CANDIDATE_ARTIFACT_MANIFEST_FILENAME:
            raise ValueError("provider candidate contains reserved artifact manifest")
        target = destination_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())


def _file_evidence(root: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if path.is_symlink():
            raise ValueError(
                f"sealed candidate artifact contains symlink: {relative}"
            )
        if not path.is_file() or _ignored_runtime_file(relative):
            continue
        relative_text = relative.as_posix()
        if relative_text == CANDIDATE_ARTIFACT_MANIFEST_FILENAME:
            continue
        payload = path.read_bytes()
        result.append(
            {
                "path": relative_text,
                "checksum": "sha256:" + hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        )
    return result


def _validated_file_evidence(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("sealed candidate file evidence must be non-empty")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, Mapping) or set(item) != {
            "path",
            "checksum",
            "size_bytes",
        }:
            raise ValueError("sealed candidate file evidence entry is invalid")
        relative = item.get("path")
        if (
            not isinstance(relative, str)
            or not relative
            or relative in seen
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or Path(relative).as_posix() != relative
        ):
            raise ValueError("sealed candidate file evidence path is invalid")
        seen.add(relative)
        checksum = require_sha256_checksum(
            item.get("checksum"),
            "sealed candidate file checksum",
        )
        size = item.get("size_bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError("sealed candidate file size is invalid")
        normalized.append(
            {"path": relative, "checksum": checksum, "size_bytes": size}
        )
    if normalized != sorted(normalized, key=lambda item: item["path"]):
        raise ValueError("sealed candidate file evidence must be sorted")
    return normalized


def _ignored_runtime_file(relative: Path) -> bool:
    return "__pycache__" in relative.parts or relative.suffix == ".pyc"


__all__ = [
    "CANDIDATE_ARTIFACT_MANIFEST_FILENAME",
    "CANDIDATE_ARTIFACT_SCHEMA_VERSION",
    "SealedCandidateArtifactRef",
    "VerifiedCandidateArtifactRef",
    "load_sealed_candidate_artifact",
    "seal_candidate_package",
    "verify_sealed_candidate_artifact",
    "verify_verified_candidate_artifact",
]
