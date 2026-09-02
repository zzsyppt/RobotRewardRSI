"""Immutable readiness derived from candidate execution surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from true_rl_agent.candidates.capability_profile import CandidateCapabilityProfile
from true_rl_agent.evaluation.binding import canonical_json_hash, require_sha256_digest

COMPLETE_REWARD_PROGRAM_CAPABILITY_ID = "complete_reward_program"
READINESS_REQUIRED_CAPABILITY_IDS = frozenset(
    {COMPLETE_REWARD_PROGRAM_CAPABILITY_ID}
)
AUTHORING_CAPABILITY_SURFACES = (
    "admission",
    "materializer_runtime",
    "task_smoke",
)
AUTHORING_CAPABILITY_READINESS_SCHEMA_VERSION = (
    "true_rl_agent.authoring_capability_readiness.v1"
)


@dataclass(frozen=True)
class AuthoringCapabilityReadiness:
    """One frozen intersection of the surfaces that can execute a candidate."""

    profile_id: str
    capability_profile_hash: str
    selected_capability_ids: tuple[str, ...]
    admission_policy_id: str
    admission_policy_hash: str
    surface_capability_ids: Mapping[str, tuple[str, ...]]
    effective_capability_ids: tuple[str, ...]
    readiness_hash: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("profile_id", "admission_policy_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"authoring readiness {field_name} is required")
        require_sha256_digest(
            self.capability_profile_hash,
            "authoring readiness capability profile hash",
        )
        require_sha256_digest(
            self.admission_policy_hash,
            "authoring readiness admission policy hash",
        )
        if not isinstance(self.surface_capability_ids, Mapping):
            raise TypeError("authoring readiness surfaces must be an object")
        if set(self.surface_capability_ids) != set(AUTHORING_CAPABILITY_SURFACES):
            raise ValueError("authoring readiness surfaces are incomplete")
        normalized = {
            name: _normalized_capability_ids(
                self.surface_capability_ids[name],
                field_name=f"authoring readiness {name}",
            )
            for name in AUTHORING_CAPABILITY_SURFACES
        }
        effective = _normalized_capability_ids(
            self.effective_capability_ids,
            field_name="authoring readiness effective capabilities",
        )
        selected = _normalized_capability_ids(
            self.selected_capability_ids,
            field_name="authoring readiness selected capabilities",
        )
        if any(not set(value) <= set(selected) for value in normalized.values()):
            raise ValueError(
                "authoring readiness surface contains an unselected capability"
            )
        expected_effective = tuple(
            sorted(set.intersection(*(set(value) for value in normalized.values())))
        )
        if effective != expected_effective:
            raise ValueError(
                "authoring readiness effective capabilities do not match surfaces"
            )
        object.__setattr__(
            self,
            "surface_capability_ids",
            MappingProxyType(normalized),
        )
        object.__setattr__(self, "selected_capability_ids", selected)
        object.__setattr__(self, "effective_capability_ids", effective)
        expected_hash = canonical_json_hash(self._hash_payload())
        if self.readiness_hash is not None:
            require_sha256_digest(
                self.readiness_hash,
                "authoring capability readiness hash",
            )
            if self.readiness_hash != expected_hash:
                raise ValueError("authoring capability readiness hash mismatch")
        object.__setattr__(self, "readiness_hash", expected_hash)

    @classmethod
    def derive(
        cls,
        *,
        profile: CandidateCapabilityProfile,
        admission_policy_id: str,
        admission_policy_hash: str,
        admission_capability_ids: Iterable[str],
        materializer_runtime_capability_ids: Iterable[str],
        task_smoke_capability_ids: Iterable[str],
    ) -> "AuthoringCapabilityReadiness":
        if not isinstance(profile, CandidateCapabilityProfile):
            raise TypeError("authoring readiness requires a capability profile")
        selected = set(profile.capability_ids)
        surfaces = {
            "admission": tuple(sorted(selected & set(admission_capability_ids))),
            "materializer_runtime": tuple(
                sorted(selected & set(materializer_runtime_capability_ids))
            ),
            "task_smoke": tuple(sorted(selected & set(task_smoke_capability_ids))),
        }
        effective = tuple(
            sorted(set.intersection(*(set(value) for value in surfaces.values())))
        )
        assert profile.profile_hash is not None
        return cls(
            profile_id=profile.profile_id,
            capability_profile_hash=profile.profile_hash,
            selected_capability_ids=profile.capability_ids,
            admission_policy_id=admission_policy_id,
            admission_policy_hash=admission_policy_hash,
            surface_capability_ids=surfaces,
            effective_capability_ids=effective,
        )

    def assert_profile(self, profile: CandidateCapabilityProfile) -> None:
        if not isinstance(profile, CandidateCapabilityProfile):
            raise TypeError("authoring readiness profile is invalid")
        if (
            profile.profile_id != self.profile_id
            or profile.profile_hash != self.capability_profile_hash
        ):
            raise ValueError("authoring readiness capability profile mismatch")

    def assert_matches(
        self,
        *,
        profile: CandidateCapabilityProfile,
        admission_policy_id: str,
        admission_policy_hash: str,
        admission_capability_ids: Iterable[str],
        materializer_runtime_capability_ids: Iterable[str],
        task_smoke_capability_ids: Iterable[str],
    ) -> None:
        expected = type(self).derive(
            profile=profile,
            admission_policy_id=admission_policy_id,
            admission_policy_hash=admission_policy_hash,
            admission_capability_ids=admission_capability_ids,
            materializer_runtime_capability_ids=(
                materializer_runtime_capability_ids
            ),
            task_smoke_capability_ids=task_smoke_capability_ids,
        )
        if self != expected:
            raise ValueError(
                "authoring readiness does not match runtime execution surfaces"
            )

    def require_selected_capabilities_ready(
        self,
        required_capability_ids: Iterable[str] = READINESS_REQUIRED_CAPABILITY_IDS,
    ) -> None:
        selected_required = set(required_capability_ids) & set(
            self.selected_capability_ids
        )
        unavailable = selected_required - set(self.effective_capability_ids)
        if unavailable:
            raise ValueError(
                "authoring capability is not ready on every execution surface: "
                + ", ".join(sorted(unavailable))
            )

    def to_prompt_mapping(self) -> Mapping[str, Mapping[str, Any]]:
        assert self.readiness_hash is not None
        return MappingProxyType(
            {
                capability_id: MappingProxyType(
                    {
                        "status": "ready",
                        "capability_profile_hash": self.capability_profile_hash,
                        "admission_policy_id": self.admission_policy_id,
                        "admission_policy_hash": self.admission_policy_hash,
                        "readiness_hash": self.readiness_hash,
                    }
                )
                for capability_id in self.effective_capability_ids
            }
        )

    def to_public_dict(self) -> dict[str, Any]:
        assert self.readiness_hash is not None
        return {
            **self._hash_payload(),
            "readiness_hash": self.readiness_hash,
        }

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "schema_version": AUTHORING_CAPABILITY_READINESS_SCHEMA_VERSION,
            "profile_id": self.profile_id,
            "capability_profile_hash": self.capability_profile_hash,
            "selected_capability_ids": list(self.selected_capability_ids),
            "admission_policy_id": self.admission_policy_id,
            "admission_policy_hash": self.admission_policy_hash,
            "surface_capability_ids": {
                name: list(self.surface_capability_ids[name])
                for name in AUTHORING_CAPABILITY_SURFACES
            },
            "effective_capability_ids": list(self.effective_capability_ids),
        }


def require_readiness_entry(
    capability_id: str,
    value: Any,
    *,
    expected_profile_hash: str,
) -> dict[str, str]:
    """Validate the prompt projection for one readiness-gated capability."""

    if not isinstance(value, Mapping):
        raise ValueError(f"authoring readiness is missing for {capability_id}")
    expected_fields = {
        "status",
        "capability_profile_hash",
        "admission_policy_id",
        "admission_policy_hash",
        "readiness_hash",
    }
    if set(value) != expected_fields or value.get("status") != "ready":
        raise ValueError(f"authoring readiness is invalid for {capability_id}")
    profile_hash = require_sha256_digest(
        value.get("capability_profile_hash"),
        "authoring readiness capability profile hash",
    )
    if profile_hash != expected_profile_hash:
        raise ValueError("authoring readiness capability profile hash mismatch")
    policy_id = value.get("admission_policy_id")
    if not isinstance(policy_id, str) or not policy_id:
        raise ValueError("authoring readiness admission policy ID is invalid")
    return {
        "status": "ready",
        "capability_profile_hash": profile_hash,
        "admission_policy_id": policy_id,
        "admission_policy_hash": require_sha256_digest(
            value.get("admission_policy_hash"),
            "authoring readiness admission policy hash",
        ),
        "readiness_hash": require_sha256_digest(
            value.get("readiness_hash"),
            "authoring readiness hash",
        ),
    }


def _normalized_capability_ids(
    value: Iterable[str],
    *,
    field_name: str,
) -> tuple[str, ...]:
    if isinstance(value, str):
        raise TypeError(f"{field_name} must be a collection")
    normalized = tuple(value)
    if any(not isinstance(item, str) or not item for item in normalized):
        raise ValueError(f"{field_name} must contain non-empty strings")
    if normalized != tuple(sorted(set(normalized))):
        raise ValueError(f"{field_name} must be sorted and unique")
    return normalized


__all__ = [
    "AUTHORING_CAPABILITY_READINESS_SCHEMA_VERSION",
    "AUTHORING_CAPABILITY_SURFACES",
    "AuthoringCapabilityReadiness",
    "COMPLETE_REWARD_PROGRAM_CAPABILITY_ID",
    "READINESS_REQUIRED_CAPABILITY_IDS",
    "require_readiness_entry",
]
