"""Trusted reference-program source manifests."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from true_rl_agent.candidates.loader import (
    CONTROLLED_ENTRYPOINT,
    load_candidate_manifest,
)
from true_rl_agent.evaluation.binding import (
    canonical_json_hash,
    require_sha256_checksum,
    require_sha256_digest,
)
from true_rl_agent.evaluation.subject import validate_experiment_subject_id

REFERENCE_PROGRAM_MANIFEST_SCHEMA_VERSION = (
    "true_rl_agent.reference_program_manifest.v1"
)
VERIFIED_REFERENCE_PROGRAM_SCHEMA_VERSION = (
    "true_rl_agent.verified_reference_program.v1"
)
CANONICAL_TRAINING_COMPONENTS_PATH = "training_code/components.py"
CANONICAL_TRAINING_COMPONENTS_ENTRYPOINT = (
    CONTROLLED_ENTRYPOINT
)
REFERENCE_PROGRAM_AUTHORSHIP_ORIGINS = frozenset(
    {"official_upstream", "llm_generated", "human_authored"}
)
CANDIDATE_PROGRAM_IDENTITY_SCHEMA_VERSION = (
    "true_rl_agent.candidate_program_identity.v1"
)


@dataclass(frozen=True)
class CandidateProgramIdentity:
    candidate_id: str
    source_checksum: str
    source_size_bytes: int
    manifest_hash: str


def resolve_candidate_program_identity(
    candidate_root: str | Path,
    *,
    expected_candidate_id: str,
) -> CandidateProgramIdentity:
    root = Path(candidate_root).resolve(strict=True)
    manifest_path = root / "candidate_manifest.yaml"
    source_path = root / CANONICAL_TRAINING_COMPONENTS_PATH
    if manifest_path.is_symlink() or source_path.is_symlink():
        raise ValueError("candidate program identity does not allow symlinks")
    manifest = load_candidate_manifest(root)
    if manifest.candidate_id != expected_candidate_id:
        raise ValueError("candidate program identity candidate_id mismatch")
    payload = source_path.read_bytes()
    if not payload:
        raise ValueError("candidate components source must be non-empty")
    source_checksum = "sha256:" + hashlib.sha256(payload).hexdigest()
    manifest_hash = canonical_json_hash(
        {
            "schema_version": CANDIDATE_PROGRAM_IDENTITY_SCHEMA_VERSION,
            "candidate_schema_version": manifest.schema_version,
            "candidate_id": expected_candidate_id,
            "source_path": CANONICAL_TRAINING_COMPONENTS_PATH,
            "entrypoint": CANONICAL_TRAINING_COMPONENTS_ENTRYPOINT,
            "source_checksum": source_checksum,
            "source_size_bytes": len(payload),
        }
    )
    return CandidateProgramIdentity(
        candidate_id=expected_candidate_id,
        source_checksum=source_checksum,
        source_size_bytes=len(payload),
        manifest_hash=manifest_hash,
    )


@dataclass(frozen=True)
class ReferenceProgramManifest:
    reference_id: str
    task_id: str
    source_uri: str
    source_checksum: str
    source_size_bytes: int
    authorship_origin: str
    upstream_project: str
    upstream_revision: str
    upstream_source_path: str
    capability_profile_hash: str
    manifest_hash: str | None = None

    def __post_init__(self) -> None:
        validate_experiment_subject_id(self.reference_id)
        validate_experiment_subject_id(self.task_id)
        for field_name in (
            "source_uri",
            "upstream_project",
            "upstream_revision",
            "upstream_source_path",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"reference program {field_name} must be non-empty")
        require_sha256_checksum(
            self.source_checksum,
            "reference program source checksum",
        )
        if (
            isinstance(self.source_size_bytes, bool)
            or not isinstance(self.source_size_bytes, int)
            or self.source_size_bytes <= 0
        ):
            raise ValueError("reference program source_size_bytes must be positive")
        if self.authorship_origin not in REFERENCE_PROGRAM_AUTHORSHIP_ORIGINS:
            raise ValueError("reference program authorship_origin is invalid")
        require_sha256_digest(
            self.capability_profile_hash,
            "reference program capability profile hash",
        )
        expected = canonical_json_hash(self._hash_payload())
        if self.manifest_hash is not None:
            require_sha256_digest(
                self.manifest_hash,
                "reference program manifest hash",
            )
            if self.manifest_hash != expected:
                raise ValueError("reference program manifest hash mismatch")
        object.__setattr__(self, "manifest_hash", expected)

    @classmethod
    def from_mapping(cls, value: Any) -> "ReferenceProgramManifest":
        if not isinstance(value, Mapping):
            raise ValueError("reference program manifest must be an object")
        expected_fields = {
            "schema_version",
            "reference_id",
            "task_id",
            "source_uri",
            "source_path",
            "entrypoint",
            "source_checksum",
            "source_size_bytes",
            "authorship_origin",
            "upstream_project",
            "upstream_revision",
            "upstream_source_path",
            "capability_profile_hash",
            "manifest_hash",
        }
        if set(value) != expected_fields:
            raise ValueError("reference program manifest fields mismatch")
        if value.get("schema_version") != REFERENCE_PROGRAM_MANIFEST_SCHEMA_VERSION:
            raise ValueError("reference program manifest schema_version mismatch")
        if value.get("source_path") != CANONICAL_TRAINING_COMPONENTS_PATH:
            raise ValueError("reference program source_path is not canonical")
        if value.get("entrypoint") != CANONICAL_TRAINING_COMPONENTS_ENTRYPOINT:
            raise ValueError("reference program entrypoint is not canonical")
        return cls(
            reference_id=value.get("reference_id"),
            task_id=value.get("task_id"),
            source_uri=value.get("source_uri"),
            source_checksum=value.get("source_checksum"),
            source_size_bytes=value.get("source_size_bytes"),
            authorship_origin=value.get("authorship_origin"),
            upstream_project=value.get("upstream_project"),
            upstream_revision=value.get("upstream_revision"),
            upstream_source_path=value.get("upstream_source_path"),
            capability_profile_hash=value.get("capability_profile_hash"),
            manifest_hash=value.get("manifest_hash"),
        )

    def verify_source_file(self, source_path: str | Path) -> None:
        path = Path(source_path).resolve(strict=True)
        if not path.is_file():
            raise ValueError("reference program source is not a file")
        self.verify_source_bytes(path.read_bytes())

    def verify_source_bytes(self, payload: bytes) -> None:
        if not isinstance(payload, bytes) or not payload:
            raise ValueError("reference program source bytes must be non-empty")
        if len(payload) != self.source_size_bytes:
            raise ValueError("reference program source size mismatch")
        checksum = "sha256:" + hashlib.sha256(payload).hexdigest()
        if checksum != self.source_checksum:
            raise ValueError("reference program source checksum mismatch")

    def to_public_dict(self) -> dict[str, Any]:
        assert self.manifest_hash is not None
        return {**self._hash_payload(), "manifest_hash": self.manifest_hash}

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "schema_version": REFERENCE_PROGRAM_MANIFEST_SCHEMA_VERSION,
            "reference_id": self.reference_id,
            "task_id": self.task_id,
            "source_uri": self.source_uri,
            "source_path": CANONICAL_TRAINING_COMPONENTS_PATH,
            "entrypoint": CANONICAL_TRAINING_COMPONENTS_ENTRYPOINT,
            "source_checksum": self.source_checksum,
            "source_size_bytes": self.source_size_bytes,
            "authorship_origin": self.authorship_origin,
            "upstream_project": self.upstream_project,
            "upstream_revision": self.upstream_revision,
            "upstream_source_path": self.upstream_source_path,
            "capability_profile_hash": self.capability_profile_hash,
        }


@dataclass(frozen=True, init=False)
class VerifiedReferenceProgramArtifact:
    """Capability proving that one manifest is backed by exact source bytes."""

    manifest: ReferenceProgramManifest
    source_bytes: bytes
    verification_hash: str

    _TOKEN = object()

    def __init__(
        self,
        *,
        manifest: ReferenceProgramManifest,
        source_bytes: bytes,
        verification_hash: str,
        _token: object | None = None,
    ) -> None:
        if _token is not self._TOKEN:
            raise TypeError(
                "verified reference programs must be created by the trusted resolver"
            )
        if not isinstance(manifest, ReferenceProgramManifest):
            raise TypeError("verified reference program manifest is invalid")
        manifest.verify_source_bytes(source_bytes)
        expected_hash = _reference_program_verification_hash(manifest)
        require_sha256_digest(
            verification_hash,
            "verified reference program verification hash",
        )
        if verification_hash != expected_hash:
            raise ValueError("verified reference program verification hash mismatch")
        object.__setattr__(self, "manifest", manifest)
        object.__setattr__(self, "source_bytes", bytes(source_bytes))
        object.__setattr__(self, "verification_hash", verification_hash)

    @classmethod
    def _validated(
        cls,
        *,
        manifest: ReferenceProgramManifest,
        source_bytes: bytes,
    ) -> "VerifiedReferenceProgramArtifact":
        return cls(
            manifest=manifest,
            source_bytes=source_bytes,
            verification_hash=_reference_program_verification_hash(manifest),
            _token=cls._TOKEN,
        )

    @property
    def source_text(self) -> str:
        return self.source_bytes.decode("utf-8")

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "schema_version": VERIFIED_REFERENCE_PROGRAM_SCHEMA_VERSION,
            "manifest": self.manifest.to_public_dict(),
            "verification": {
                "status": "pass",
                "verification_hash": self.verification_hash,
            },
        }


def resolve_reference_program_artifact(
    source_path: str | Path,
    *,
    reference_id: str,
    task_id: str,
    authorship_origin: str,
    upstream_project: str,
    upstream_revision: str,
    upstream_source_path: str,
    capability_profile_hash: str,
) -> VerifiedReferenceProgramArtifact:
    """Read one real source file and derive its complete trusted identity."""

    unresolved = Path(source_path)
    if unresolved.is_symlink():
        raise ValueError("reference program source must not be a symlink")
    resolved = unresolved.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError("reference program source is not a file")
    payload = resolved.read_bytes()
    if not payload:
        raise ValueError("reference program source bytes must be non-empty")
    try:
        payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("reference program source must be valid UTF-8") from exc
    source_checksum = "sha256:" + hashlib.sha256(payload).hexdigest()
    source_uri = (
        "artifact://reference-program/"
        + source_checksum.removeprefix("sha256:")
        + "/"
        + CANONICAL_TRAINING_COMPONENTS_PATH
    )
    manifest = ReferenceProgramManifest(
        reference_id=reference_id,
        task_id=task_id,
        source_uri=source_uri,
        source_checksum=source_checksum,
        source_size_bytes=len(payload),
        authorship_origin=authorship_origin,
        upstream_project=upstream_project,
        upstream_revision=upstream_revision,
        upstream_source_path=upstream_source_path,
        capability_profile_hash=capability_profile_hash,
    )
    return VerifiedReferenceProgramArtifact._validated(
        manifest=manifest,
        source_bytes=payload,
    )


def _reference_program_verification_hash(
    manifest: ReferenceProgramManifest,
) -> str:
    if manifest.manifest_hash is None:
        raise ValueError("reference program manifest hash is missing")
    return canonical_json_hash(
        {
            "schema_version": VERIFIED_REFERENCE_PROGRAM_SCHEMA_VERSION,
            "manifest_hash": manifest.manifest_hash,
            "source_checksum": manifest.source_checksum,
            "source_size_bytes": manifest.source_size_bytes,
        }
    )


__all__ = [
    "CANDIDATE_PROGRAM_IDENTITY_SCHEMA_VERSION",
    "CANONICAL_TRAINING_COMPONENTS_ENTRYPOINT",
    "CANONICAL_TRAINING_COMPONENTS_PATH",
    "REFERENCE_PROGRAM_AUTHORSHIP_ORIGINS",
    "REFERENCE_PROGRAM_MANIFEST_SCHEMA_VERSION",
    "VERIFIED_REFERENCE_PROGRAM_SCHEMA_VERSION",
    "CandidateProgramIdentity",
    "ReferenceProgramManifest",
    "VerifiedReferenceProgramArtifact",
    "resolve_reference_program_artifact",
    "resolve_candidate_program_identity",
]
