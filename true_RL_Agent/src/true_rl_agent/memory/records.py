"""Experiment records with explicit public/private projections."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from true_rl_agent.agent_core.context import sanitize_public_context
from true_rl_agent.evaluation.binding import require_sha256_checksum


@dataclass(frozen=True)
class PrivateArtifactRef:
    """Reference to private evidence that must not be prompt input."""

    kind: str
    uri: str
    checksum: str | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.checksum is not None:
            require_sha256_checksum(
                self.checksum,
                "private artifact checksum",
            )

    def to_private_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "kind": self.kind,
            "uri": self.uri,
        }
        if self.checksum is not None:
            data["checksum"] = self.checksum
        if self.reason is not None:
            data["reason"] = self.reason
        return data

    def to_public_redaction(self) -> dict[str, Any]:
        return {
            "kind": "non_prompt_artifact",
            "visibility": "not_prompt_visible",
            "included_in_prompt": False,
        }


@dataclass(frozen=True)
class PromptPayloadRef:
    """Public prompt payload artifact reference."""

    path: str
    visibility: str = "public_prompt_payload"

    def to_public_dict(self) -> dict[str, str]:
        return {
            "path": self.path,
            "visibility": self.visibility,
        }


@dataclass(frozen=True)
class ExperimentRecord:
    """Round record split into public prompt state and private artifact refs."""

    round_id: str
    public_record: Mapping[str, Any]
    private_artifacts: tuple[PrivateArtifactRef, ...] = field(default_factory=tuple)
    prompt_payload_ref: PromptPayloadRef | None = None
    schema_version: str = "true_rl_agent.experiment_record.v1"

    @classmethod
    def from_ledger_record(
        cls,
        ledger_record: Mapping[str, Any],
        *,
        private_artifacts: tuple[PrivateArtifactRef, ...] = (),
        prompt_payload_ref: PromptPayloadRef | None = None,
    ) -> "ExperimentRecord":
        artifacts = _dedupe_private_artifacts(
            (*private_artifacts, *_private_artifacts_from_ledger_record(ledger_record))
        )
        return cls(
            round_id=str(ledger_record.get("round_id", "")),
            public_record=sanitize_public_context(_prompt_public_ledger_record(ledger_record)),
            private_artifacts=artifacts,
            prompt_payload_ref=prompt_payload_ref,
        )

    def ledger_public_record(self) -> dict[str, Any]:
        return sanitize_public_context(dict(self.public_record))

    def to_public_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "schema_version": self.schema_version,
            "round_id": self.round_id,
            "public_record": self.ledger_public_record(),
            "non_prompt_artifact_count": len(self.private_artifacts),
            "artifact_redactions": [artifact.to_public_redaction() for artifact in self.private_artifacts],
        }
        if self.prompt_payload_ref is not None:
            data["prompt_payload"] = self.prompt_payload_ref.to_public_dict()
        return sanitize_public_context(data)

    def to_private_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "round_id": self.round_id,
            "private_artifacts": [artifact.to_private_dict() for artifact in self.private_artifacts],
        }


def _private_artifacts_from_ledger_record(ledger_record: Mapping[str, Any]) -> tuple[PrivateArtifactRef, ...]:
    refs: list[PrivateArtifactRef] = []
    raw_refs = ledger_record.get("private_artifacts")
    if isinstance(raw_refs, list | tuple):
        for item in raw_refs:
            if not isinstance(item, Mapping):
                continue
            kind = item.get("kind")
            uri = item.get("uri")
            if isinstance(kind, str) and isinstance(uri, str):
                refs.append(
                    PrivateArtifactRef(
                        kind=kind,
                        uri=uri,
                        checksum=item.get("checksum") if isinstance(item.get("checksum"), str) else None,
                        reason=item.get("reason") if isinstance(item.get("reason"), str) else None,
                    )
                )
    candidate = ledger_record.get("candidate")
    if isinstance(candidate, Mapping):
        path = candidate.get("path")
        if isinstance(path, str) and path:
            refs.append(
                PrivateArtifactRef(
                    kind="candidate_package",
                    uri=path,
                    reason="candidate_source_resolution",
                )
            )
    return tuple(refs)


def _prompt_public_ledger_record(ledger_record: Mapping[str, Any]) -> dict[str, Any]:
    record = _public_copy(ledger_record)
    record.pop("private_artifacts", None)
    candidate = record.get("candidate")
    if isinstance(candidate, dict):
        candidate.pop("path", None)
        candidate.pop("candidate_path", None)
    return record


def _public_copy(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _public_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_public_copy(item) for item in value]
    if isinstance(value, tuple):
        return [_public_copy(item) for item in value]
    return value


def _dedupe_private_artifacts(refs: tuple[PrivateArtifactRef, ...]) -> tuple[PrivateArtifactRef, ...]:
    deduped: list[PrivateArtifactRef] = []
    seen: set[tuple[str, str]] = set()
    for ref in refs:
        key = (ref.kind, ref.uri)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ref)
    return tuple(deduped)
