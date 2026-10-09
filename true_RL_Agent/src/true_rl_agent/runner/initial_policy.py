"""Task-neutral evidence for deterministic initial policy pairing."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from true_rl_agent.evaluation.binding import (
    canonical_json_hash,
    require_sha256_digest,
)

INITIAL_POLICY_STATE_PAIRING_SCHEMA_VERSION = (
    "true_rl_agent.initial_policy_state_pairing.v1"
)


@dataclass(frozen=True)
class InitialPolicyStatePairing:
    """Bind an expected policy state to the bytes initialized by a train job."""

    expected_initial_policy_state_hash: str
    actual_initial_policy_state_hash: str
    match: bool

    def __post_init__(self) -> None:
        require_sha256_digest(
            self.expected_initial_policy_state_hash,
            "expected initial policy state hash",
        )
        require_sha256_digest(
            self.actual_initial_policy_state_hash,
            "actual initial policy state hash",
        )
        if not isinstance(self.match, bool):
            raise TypeError("initial policy state match must be boolean")
        if self.match is not (
            self.expected_initial_policy_state_hash
            == self.actual_initial_policy_state_hash
        ):
            raise ValueError("initial policy state match flag is inconsistent")

    @classmethod
    def from_training_summary(
        cls,
        summary: Mapping[str, Any],
        *,
        expected_initial_policy_state_hash: str,
    ) -> "InitialPolicyStatePairing":
        """Validate the runtime-owned pairing fields in a train summary."""

        expected = require_sha256_digest(
            expected_initial_policy_state_hash,
            "expected initial policy state hash",
        )
        if not isinstance(summary, Mapping):
            raise TypeError("train summary must be an object")
        training = summary.get("training")
        if not isinstance(training, Mapping):
            raise ValueError("train summary is missing initial policy state evidence")
        declared_expected = require_sha256_digest(
            training.get("expected_initial_policy_state_hash"),
            "train summary expected initial policy state hash",
        )
        if declared_expected != expected:
            raise ValueError("train summary expected initial policy state hash mismatch")
        match = training.get("initial_policy_state_match")
        if not isinstance(match, bool):
            raise ValueError("train summary initial policy state match is missing")
        return cls(
            expected_initial_policy_state_hash=declared_expected,
            actual_initial_policy_state_hash=require_sha256_digest(
                training.get("initial_policy_state_hash"),
                "train summary actual initial policy state hash",
            ),
            match=match,
        )

    @classmethod
    def from_mapping(cls, value: Any) -> "InitialPolicyStatePairing":
        if not isinstance(value, Mapping):
            raise TypeError("initial policy state pairing must be an object")
        expected_fields = {
            "schema_version",
            "expected_initial_policy_state_hash",
            "actual_initial_policy_state_hash",
            "match",
        }
        if set(value) != expected_fields:
            raise ValueError("initial policy state pairing fields mismatch")
        if value.get("schema_version") != INITIAL_POLICY_STATE_PAIRING_SCHEMA_VERSION:
            raise ValueError("initial policy state pairing schema mismatch")
        return cls(
            expected_initial_policy_state_hash=value.get(
                "expected_initial_policy_state_hash"
            ),
            actual_initial_policy_state_hash=value.get(
                "actual_initial_policy_state_hash"
            ),
            match=value.get("match"),
        )

    def require_match(self) -> None:
        if not self.match:
            raise ValueError("actual initial policy state does not match expected")

    @property
    def pairing_hash(self) -> str:
        return canonical_json_hash(self.to_public_dict())

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "schema_version": INITIAL_POLICY_STATE_PAIRING_SCHEMA_VERSION,
            "expected_initial_policy_state_hash": (
                self.expected_initial_policy_state_hash
            ),
            "actual_initial_policy_state_hash": self.actual_initial_policy_state_hash,
            "match": self.match,
        }


def verified_initial_policy_state_pairing_hash(value: Any) -> str | None:
    """Return the canonical hash of an optional, matching pairing artifact."""

    if value is None:
        return None
    pairing = InitialPolicyStatePairing.from_mapping(value)
    pairing.require_match()
    return pairing.pairing_hash


__all__ = [
    "INITIAL_POLICY_STATE_PAIRING_SCHEMA_VERSION",
    "InitialPolicyStatePairing",
    "verified_initial_policy_state_pairing_hash",
]
