"""Runner-owned, content-addressed candidate execution artifacts."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from true_rl_agent.candidates.package_contract import (
    LEGACY_SEALED_REQUIRED_CANDIDATE_FILES,
    PROVIDER_AUTHORED_CANDIDATE_FILES,
    SEALED_CANDIDATE_ARTIFACT_MANIFEST_FILENAME,
    SEALED_CANDIDATE_ARTIFACT_MANIFEST_MAX_BYTES,
    CandidatePackageContractError,
    CandidatePackagePreflight,
    copy_preflighted_candidate_package,
    open_preflighted_candidate_file,
    preflight_candidate_package,
)
from true_rl_agent.candidates.schema import (
    CandidateChangeSet,
    CandidatePackageRef,
    require_candidate_id,
)
from true_rl_agent.evaluation.binding import (
    canonical_json_hash,
    require_sha256_checksum,
    require_sha256_digest,
)
from true_rl_agent.json_contract import (
    canonical_json_bytes,
    canonical_json_text,
    strict_json_object_loads,
)
from true_rl_agent.training_programs import resolve_candidate_program_identity

CANDIDATE_ARTIFACT_SCHEMA_VERSION = (
    "true_rl_agent.sealed_candidate_artifact.v1"
)
CANDIDATE_ARTIFACT_MANIFEST_FILENAME = (
    SEALED_CANDIDATE_ARTIFACT_MANIFEST_FILENAME
)
CANDIDATE_EXECUTION_AUTHORIZATION_SCHEMA_VERSION = (
    "true_rl_agent.candidate_execution_authorization.v1"
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
        require_candidate_id(
            self.candidate_id,
            "sealed candidate artifact candidate_id",
        )
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
class CandidateExecutionAuthorization:
    """Content-addressed task payload minted by the verification pipeline."""

    admission_policy_id: str
    admission_policy_hash: str
    payload_schema_version: str
    canonical_json: str
    authorization_hash: str

    _TOKEN = object()

    def __init__(
        self,
        *,
        admission_policy_id: str,
        admission_policy_hash: str,
        payload_schema_version: str,
        canonical_json: str,
        authorization_hash: str,
        _token: object | None = None,
    ) -> None:
        if _token is not self._TOKEN:
            raise TypeError(
                "candidate execution authorizations must be created from canonical JSON"
            )
        object.__setattr__(self, "admission_policy_id", admission_policy_id)
        object.__setattr__(self, "admission_policy_hash", admission_policy_hash)
        object.__setattr__(self, "payload_schema_version", payload_schema_version)
        object.__setattr__(self, "canonical_json", canonical_json)
        object.__setattr__(self, "authorization_hash", authorization_hash)

    @classmethod
    def from_payload(
        cls,
        *,
        admission_policy_id: str,
        admission_policy_hash: str,
        payload: Mapping[str, Any],
    ) -> "CandidateExecutionAuthorization":
        if not isinstance(payload, Mapping):
            raise TypeError("candidate execution authorization payload must be an object")
        envelope = {
            "schema_version": CANDIDATE_EXECUTION_AUTHORIZATION_SCHEMA_VERSION,
            "admission_policy_id": admission_policy_id,
            "admission_policy_hash": admission_policy_hash,
            "payload": dict(payload),
        }
        return cls.from_canonical_json(canonical_json_text(envelope))

    @classmethod
    def from_canonical_json(
        cls,
        value: str,
        *,
        expected_authorization_hash: str | None = None,
    ) -> "CandidateExecutionAuthorization":
        if not isinstance(value, str) or not value:
            raise ValueError("candidate execution authorization JSON is required")
        envelope = strict_json_object_loads(
            value,
            context="candidate execution authorization",
        )
        if canonical_json_text(envelope) != value:
            raise ValueError("candidate execution authorization must be canonical JSON")
        if set(envelope) != {
            "schema_version",
            "admission_policy_id",
            "admission_policy_hash",
            "payload",
        }:
            raise ValueError("candidate execution authorization fields mismatch")
        if (
            envelope.get("schema_version")
            != CANDIDATE_EXECUTION_AUTHORIZATION_SCHEMA_VERSION
        ):
            raise ValueError("candidate execution authorization schema mismatch")
        admission_policy_id = envelope.get("admission_policy_id")
        if not isinstance(admission_policy_id, str) or not admission_policy_id:
            raise ValueError("candidate execution authorization policy ID is required")
        admission_policy_hash = require_sha256_digest(
            envelope.get("admission_policy_hash"),
            "candidate execution authorization policy hash",
        )
        payload = envelope.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("candidate execution authorization payload must be an object")
        payload_schema_version = payload.get("schema_version")
        if not isinstance(payload_schema_version, str) or not payload_schema_version:
            raise ValueError(
                "candidate execution authorization payload schema is required"
            )
        authorization_hash = canonical_json_hash(envelope)
        if expected_authorization_hash is not None:
            if require_sha256_digest(
                expected_authorization_hash,
                "candidate execution authorization hash",
            ) != authorization_hash:
                raise ValueError("candidate execution authorization hash mismatch")
        return cls(
            admission_policy_id=admission_policy_id,
            admission_policy_hash=admission_policy_hash,
            payload_schema_version=payload_schema_version,
            canonical_json=value,
            authorization_hash=authorization_hash,
            _token=cls._TOKEN,
        )

    @property
    def payload(self) -> dict[str, Any]:
        envelope = strict_json_object_loads(
            self.canonical_json,
            context="candidate execution authorization",
        )
        payload = envelope["payload"]
        assert isinstance(payload, dict)
        return payload

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CANDIDATE_EXECUTION_AUTHORIZATION_SCHEMA_VERSION,
            "admission_policy_id": self.admission_policy_id,
            "admission_policy_hash": self.admission_policy_hash,
            "payload_schema_version": self.payload_schema_version,
            "authorization_hash": self.authorization_hash,
        }


@dataclass(frozen=True, init=False)
class VerifiedCandidateArtifactRef:
    """Capability proving that one sealed artifact passed a bound verifier."""

    sealed_artifact: SealedCandidateArtifactRef
    verification_policy: str
    verification_policy_hash: str
    verification_hash: str
    execution_authorization: CandidateExecutionAuthorization | None

    _TOKEN = object()

    def __init__(
        self,
        *,
        sealed_artifact: SealedCandidateArtifactRef,
        verification_policy: str,
        verification_policy_hash: str | None = None,
        verification_hash: str,
        execution_authorization: CandidateExecutionAuthorization | None = None,
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
            verification_policy_hash,
            "verified candidate verification policy hash",
        )
        require_sha256_digest(
            verification_hash,
            "verified candidate verification hash",
        )
        if execution_authorization is not None:
            if not isinstance(
                execution_authorization,
                CandidateExecutionAuthorization,
            ):
                raise TypeError("verified candidate execution authorization is invalid")
            if (
                execution_authorization.admission_policy_id
                != verification_policy
                or execution_authorization.admission_policy_hash
                != verification_policy_hash
            ):
                raise ValueError(
                    "candidate execution authorization policy identity mismatch"
                )
        object.__setattr__(self, "sealed_artifact", sealed_artifact)
        object.__setattr__(self, "verification_policy", verification_policy)
        object.__setattr__(
            self,
            "verification_policy_hash",
            verification_policy_hash,
        )
        object.__setattr__(self, "verification_hash", verification_hash)
        object.__setattr__(
            self,
            "execution_authorization",
            execution_authorization,
        )

    @classmethod
    def _validated(
        cls,
        *,
        sealed_artifact: SealedCandidateArtifactRef,
        verification_policy: str,
        verification_policy_hash: str,
        verification_hash: str,
        execution_authorization: CandidateExecutionAuthorization | None = None,
    ) -> "VerifiedCandidateArtifactRef":
        return cls(
            sealed_artifact=sealed_artifact,
            verification_policy=verification_policy,
            verification_policy_hash=verification_policy_hash,
            verification_hash=verification_hash,
            execution_authorization=execution_authorization,
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
        result = {
            **self.sealed_artifact.to_public_dict(),
            "verification": {
                "status": "pass",
                "policy": self.verification_policy,
                "policy_hash": self.verification_policy_hash,
                "verification_hash": self.verification_hash,
            },
        }
        if self.execution_authorization is not None:
            result["execution_authorization"] = (
                self.execution_authorization.to_public_dict()
            )
        return result


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
    source_preflight = preflight_candidate_package(candidate.path)
    source_root = source_preflight.root

    destination_root = Path(artifact_root).resolve()
    destination_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=".candidate-artifact-", dir=destination_root)
    )
    try:
        copied_files = copy_preflighted_candidate_package(
            source_preflight,
            temporary,
        )
        if preflight_candidate_package(source_root) != source_preflight:
            raise CandidatePackageContractError(
                "candidate package changed while it was being sealed",
                issue="package_changed_during_sealing",
            )
        copied_preflight = preflight_candidate_package(temporary)
        files = _file_evidence(copied_preflight)
        if files != copied_files:
            raise ValueError("candidate artifact copy evidence mismatch")
        identity = resolve_candidate_program_identity(
            temporary,
            expected_candidate_id=candidate.candidate_id,
        )
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
    preflight = preflight_candidate_package(
        artifact_path,
        sealed_artifact=True,
        enforce_size_limits=False,
    )
    root = preflight.root
    with open_preflighted_candidate_file(
        preflight,
        CANDIDATE_ARTIFACT_MANIFEST_FILENAME,
    ) as manifest_source:
        manifest_payload = manifest_source.read(
            SEALED_CANDIDATE_ARTIFACT_MANIFEST_MAX_BYTES + 1
        )
    if len(manifest_payload) > SEALED_CANDIDATE_ARTIFACT_MANIFEST_MAX_BYTES:
        raise CandidatePackageContractError(
            "sealed candidate artifact manifest exceeds its byte limit",
            issue=(
                "file_too_large:"
                f"{CANDIDATE_ARTIFACT_MANIFEST_FILENAME}:"
                f"{len(manifest_payload)}>"
                f"{SEALED_CANDIDATE_ARTIFACT_MANIFEST_MAX_BYTES}"
            ),
        )
    manifest = strict_json_object_loads(
        manifest_payload,
        context="sealed candidate artifact manifest",
    )
    if manifest_payload != canonical_json_bytes(manifest, trailing_newline=True):
        raise ValueError("sealed candidate artifact manifest is not canonical JSON")
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

    candidate_id = require_candidate_id(
        manifest.get("candidate_id"),
        "sealed candidate artifact candidate_id",
    )
    if expected_candidate_id is not None and candidate_id != expected_candidate_id:
        raise ValueError("sealed candidate artifact candidate_id mismatch")
    declared_files = _validated_file_evidence(manifest.get("files"))
    if _file_evidence(preflight) != declared_files:
        raise ValueError("sealed candidate artifact file evidence mismatch")

    identity = resolve_candidate_program_identity(
        root,
        expected_candidate_id=candidate_id,
        sealed_artifact=True,
        enforce_package_size_limits=False,
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
    authorization = artifact.execution_authorization
    if authorization is not None:
        verified_authorization = CandidateExecutionAuthorization.from_canonical_json(
            authorization.canonical_json,
            expected_authorization_hash=authorization.authorization_hash,
        )
        if verified_authorization != authorization:
            raise ValueError("candidate execution authorization changed")
        if (
            authorization.admission_policy_id != artifact.verification_policy
            or authorization.admission_policy_hash
            != artifact.verification_policy_hash
        ):
            raise ValueError("candidate execution authorization policy mismatch")
    return artifact


def _file_evidence(
    preflight: CandidatePackagePreflight,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for state in sorted(preflight.files, key=lambda item: item.relative_path):
        digest = hashlib.sha256()
        size_bytes = 0
        with open_preflighted_candidate_file(
            preflight,
            state.relative_path,
        ) as source:
            while chunk := source.read(64 * 1024):
                digest.update(chunk)
                size_bytes += len(chunk)
        result.append(
            {
                "path": state.relative_path,
                "checksum": "sha256:" + digest.hexdigest(),
                "size_bytes": size_bytes,
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
    evidence_paths = {item["path"] for item in normalized}
    allowed_paths = set(PROVIDER_AUTHORED_CANDIDATE_FILES)
    if (
        not LEGACY_SEALED_REQUIRED_CANDIDATE_FILES <= evidence_paths
        or not evidence_paths <= allowed_paths
    ):
        raise ValueError(
            "sealed candidate file evidence does not match a fixed-package subset"
        )
    return normalized


__all__ = [
    "CANDIDATE_ARTIFACT_MANIFEST_FILENAME",
    "CANDIDATE_ARTIFACT_SCHEMA_VERSION",
    "CANDIDATE_EXECUTION_AUTHORIZATION_SCHEMA_VERSION",
    "CandidateExecutionAuthorization",
    "SealedCandidateArtifactRef",
    "VerifiedCandidateArtifactRef",
    "load_sealed_candidate_artifact",
    "seal_candidate_package",
    "verify_sealed_candidate_artifact",
    "verify_verified_candidate_artifact",
]
