"""Immutable task-selected capability profiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from true_rl_agent.evaluation.binding import canonical_json_hash, require_sha256_digest
from true_rl_agent.evaluation.subject import validate_experiment_subject_id

CANDIDATE_CAPABILITY_PROFILE_SCHEMA_VERSION = (
    "true_rl_agent.candidate_capability_profile.v1"
)


@dataclass(frozen=True)
class CandidateCapabilityProfile:
    """Closed selection of capability IDs from the shared capability registry."""

    profile_id: str
    capability_ids: tuple[str, ...]
    profile_hash: str | None = None

    def __post_init__(self) -> None:
        validate_experiment_subject_id(self.profile_id)
        if not isinstance(self.capability_ids, tuple):
            raise TypeError("capability profile capability_ids must be a tuple")
        if not self.capability_ids:
            raise ValueError("capability profile must select at least one capability")
        if any(not isinstance(item, str) or not item for item in self.capability_ids):
            raise ValueError("capability profile IDs must be non-empty strings")
        if len(set(self.capability_ids)) != len(self.capability_ids):
            raise ValueError("capability profile IDs must be unique")
        if self.capability_ids != tuple(sorted(self.capability_ids)):
            raise ValueError("capability profile IDs must be sorted")
        expected = canonical_json_hash(self._hash_payload())
        if self.profile_hash is not None:
            require_sha256_digest(self.profile_hash, "capability profile hash")
            if self.profile_hash != expected:
                raise ValueError("capability profile hash mismatch")
        object.__setattr__(self, "profile_hash", expected)

    @classmethod
    def build(
        cls,
        *,
        profile_id: str,
        capability_ids: Iterable[str],
        known_capability_ids: Iterable[str],
    ) -> "CandidateCapabilityProfile":
        selected = tuple(sorted(capability_ids))
        known = frozenset(known_capability_ids)
        unknown = sorted(set(selected) - known)
        if unknown:
            raise ValueError(
                "capability profile selects unknown capability ID(s): "
                + ", ".join(unknown)
            )
        return cls(profile_id=profile_id, capability_ids=selected)

    @classmethod
    def from_mapping(
        cls,
        value: Any,
        *,
        known_capability_ids: Iterable[str],
    ) -> "CandidateCapabilityProfile":
        if not isinstance(value, Mapping):
            raise ValueError("capability profile must be an object")
        expected_fields = {
            "schema_version",
            "profile_id",
            "capability_ids",
            "profile_hash",
        }
        if set(value) != expected_fields:
            raise ValueError("capability profile fields mismatch")
        if value.get("schema_version") != CANDIDATE_CAPABILITY_PROFILE_SCHEMA_VERSION:
            raise ValueError("capability profile schema_version mismatch")
        raw_ids = value.get("capability_ids")
        if not isinstance(raw_ids, list):
            raise ValueError("capability profile capability_ids must be a list")
        profile = cls(
            profile_id=value.get("profile_id"),
            capability_ids=tuple(raw_ids),
            profile_hash=value.get("profile_hash"),
        )
        unknown = sorted(
            set(profile.capability_ids) - frozenset(known_capability_ids)
        )
        if unknown:
            raise ValueError(
                "capability profile selects unknown capability ID(s): "
                + ", ".join(unknown)
            )
        return profile

    def to_public_dict(self) -> dict[str, Any]:
        assert self.profile_hash is not None
        return {**self._hash_payload(), "profile_hash": self.profile_hash}

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "schema_version": CANDIDATE_CAPABILITY_PROFILE_SCHEMA_VERSION,
            "profile_id": self.profile_id,
            "capability_ids": list(self.capability_ids),
        }


__all__ = [
    "CANDIDATE_CAPABILITY_PROFILE_SCHEMA_VERSION",
    "CandidateCapabilityProfile",
]
