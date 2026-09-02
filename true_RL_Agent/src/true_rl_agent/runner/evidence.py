"""Task-neutral public-safe evidence artifact manifests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from true_rl_agent.evaluation.binding import require_sha256_checksum

EVIDENCE_ARTIFACT_SCHEMA_VERSION = "true_rl_agent.evidence.artifact.v1"
EVIDENCE_VISIBILITIES = frozenset({"private_only", "public_projection_only"})
_FORBIDDEN_PUBLIC_ARTIFACT_KEYS = frozenset({"path", "uri", "url", "raw_payload", "data"})
_PUBLIC_ARTIFACT_KEYS = frozenset(
    {
        "schema_version",
        "artifact_id",
        "kind",
        "sha256",
        "byte_size",
        "visibility",
        "producer",
        "mime_type",
        "public_projection_ref",
    }
)
_REQUIRED_PUBLIC_ARTIFACT_KEYS = frozenset(
    {
        "schema_version",
        "artifact_id",
        "kind",
        "sha256",
        "byte_size",
        "visibility",
        "producer",
    }
)


@dataclass(frozen=True)
class EvidenceArtifact:
    """Public-safe manifest entry for private or projected evidence artifacts."""

    artifact_id: str
    kind: str
    sha256: str
    byte_size: int
    visibility: str
    producer: str
    mime_type: str = "application/octet-stream"
    public_projection_ref: str | None = None

    def __post_init__(self) -> None:
        _validate_identifier(self.artifact_id, field_name="artifact_id")
        _validate_identifier(self.kind, field_name="kind")
        _validate_identifier(self.producer, field_name="producer")
        if self.visibility not in EVIDENCE_VISIBILITIES:
            raise ValueError(f"unsupported evidence artifact visibility: {self.visibility}")
        require_sha256_checksum(self.sha256, "evidence artifact sha256")
        _validate_byte_size(self.byte_size)
        if (
            not isinstance(self.mime_type, str)
            or not self.mime_type
            or "://" in self.mime_type
        ):
            raise ValueError("evidence artifact mime_type must be local metadata")
        if self.public_projection_ref is not None:
            _validate_projection_ref(self.public_projection_ref)

    def to_public_dict(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "schema_version": EVIDENCE_ARTIFACT_SCHEMA_VERSION,
            "artifact_id": self.artifact_id,
            "kind": self.kind,
            "sha256": self.sha256,
            "byte_size": self.byte_size,
            "visibility": self.visibility,
            "producer": self.producer,
            "mime_type": self.mime_type,
        }
        if self.public_projection_ref is not None:
            record["public_projection_ref"] = self.public_projection_ref
        return record


def evidence_artifact_from_json(
    *,
    artifact_id: str,
    kind: str,
    payload: Mapping[str, Any],
    producer: str,
    visibility: str,
    public_projection_ref: str | None = None,
) -> EvidenceArtifact:
    """Build a manifest entry for a JSON projection without exposing its payload."""

    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return EvidenceArtifact(
        artifact_id=artifact_id,
        kind=kind,
        sha256="sha256:" + hashlib.sha256(encoded).hexdigest(),
        byte_size=len(encoded),
        visibility=visibility,
        producer=producer,
        mime_type="application/json",
        public_projection_ref=public_projection_ref,
    )


def validate_evidence_artifact_manifest(items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Validate and normalize a public-safe evidence artifact manifest."""

    normalized: list[dict[str, Any]] = []
    for raw in items:
        if not isinstance(raw, Mapping):
            raise ValueError("evidence artifact manifest entries must be objects")
        for key in raw:
            if key in _FORBIDDEN_PUBLIC_ARTIFACT_KEYS:
                raise ValueError(f"forbidden evidence artifact field: {key}")
            if key not in _PUBLIC_ARTIFACT_KEYS:
                raise ValueError(f"unsupported evidence artifact field: {key}")
        for key in _REQUIRED_PUBLIC_ARTIFACT_KEYS:
            if key not in raw:
                raise ValueError(f"evidence artifact {key} is required")
        artifact = EvidenceArtifact(
            artifact_id=raw["artifact_id"],
            kind=raw["kind"],
            sha256=raw["sha256"],
            byte_size=raw["byte_size"],
            visibility=raw["visibility"],
            producer=raw["producer"],
            mime_type=raw.get("mime_type", "application/octet-stream"),
            public_projection_ref=raw.get("public_projection_ref"),
        )
        record = artifact.to_public_dict()
        if raw.get("schema_version") != EVIDENCE_ARTIFACT_SCHEMA_VERSION:
            raise ValueError("evidence artifact schema_version mismatch")
        normalized.append(record)
    return normalized


def validate_evidence_projection_bindings(
    items: Sequence[Mapping[str, Any]],
    *,
    projections: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Validate manifests and bind public projection hashes to canonical payloads."""

    normalized = validate_evidence_artifact_manifest(items)
    for record in normalized:
        if record["visibility"] != "public_projection_only":
            continue
        projection_ref = record.get("public_projection_ref")
        if not isinstance(projection_ref, str):
            raise ValueError(
                "public projection evidence artifact requires public_projection_ref"
            )
        payload = projections.get(projection_ref)
        if payload is None:
            raise ValueError(
                f"evidence artifact public_projection_ref is not available: {projection_ref}"
            )
        canonical = evidence_artifact_from_json(
            artifact_id=record["artifact_id"],
            kind=record["kind"],
            payload=payload,
            producer=record["producer"],
            visibility=record["visibility"],
            public_projection_ref=projection_ref,
        ).to_public_dict()
        for field_name in ("sha256", "byte_size", "mime_type"):
            if record[field_name] != canonical[field_name]:
                raise ValueError(
                    "evidence artifact projection binding mismatch for "
                    f"{record['artifact_id']}.{field_name}"
                )
    return normalized


def _validate_byte_size(value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("evidence artifact byte_size must be a positive integer")


def _validate_identifier(value: Any, *, field_name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"evidence artifact {field_name} must be non-empty")
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    if any(char not in allowed for char in value):
        raise ValueError(f"evidence artifact {field_name} has invalid characters")


def _validate_projection_ref(value: Any) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError("evidence artifact public_projection_ref must be non-empty")
    if "://" in value or "/" in value or "\\" in value:
        raise ValueError("evidence artifact public_projection_ref must not be a path or URL")
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    if any(char not in allowed for char in value):
        raise ValueError("evidence artifact public_projection_ref has invalid characters")
