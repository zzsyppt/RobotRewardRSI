"""Task-owned session start profiles for Stair reward authoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from true_rl_agent.evaluation.binding import (
    canonical_json_hash,
    require_sha256_digest,
)

STAIR_REFERENCE_START_MODE = "reference"
STAIR_REWARD_COLD_START_MODE = "reward_cold_start"
STAIR_DEFAULT_START_MODE = STAIR_REFERENCE_START_MODE
STAIR_START_MODES = frozenset(
    {STAIR_REFERENCE_START_MODE, STAIR_REWARD_COLD_START_MODE}
)

# Runner-owned Phase E evidence for the fixed formal TrainingBinding and seed.
STAIR_FORMAL_INITIAL_POLICY_STATE_HASH = (
    "55154410d8451cece112f6418cf6f18da6a7ed6c76571cd65e08f99e4c2d543a"
)


@dataclass(frozen=True)
class StairSessionStartProfile:
    """Select only the Stair session anchor; task/eval semantics remain in V2."""

    profile_id: str
    genesis_anchor_kind: str
    task_protocol_hash: str
    training_binding_hash: str
    initial_policy_state_hash: str | None = None
    bootstrap_initial_best_score: float | None = None

    def __post_init__(self) -> None:
        if self.profile_id not in STAIR_START_MODES:
            raise ValueError("unsupported Stair session start profile")
        require_sha256_digest(
            self.task_protocol_hash,
            "Stair start profile task protocol hash",
        )
        require_sha256_digest(
            self.training_binding_hash,
            "Stair start profile training binding hash",
        )
        if self.profile_id == STAIR_REFERENCE_START_MODE:
            if self.genesis_anchor_kind != "reference_program":
                raise ValueError("Stair reference start must use reference_program")
            if (
                self.initial_policy_state_hash is not None
                or self.bootstrap_initial_best_score is not None
            ):
                raise ValueError("Stair reference start cannot carry bootstrap state")
            return
        if self.genesis_anchor_kind != "bootstrap":
            raise ValueError("Stair reward cold start must use bootstrap")
        require_sha256_digest(
            self.initial_policy_state_hash,
            "Stair cold-start initial policy state hash",
        )
        if (
            isinstance(self.bootstrap_initial_best_score, bool)
            or not isinstance(self.bootstrap_initial_best_score, int | float)
            or float(self.bootstrap_initial_best_score) != 0.0
        ):
            raise ValueError("Stair cold-start initial best score must be 0.0")

    @property
    def requires_reference_program(self) -> bool:
        return self.genesis_anchor_kind == "reference_program"

    @property
    def profile_hash(self) -> str:
        return canonical_json_hash(self.to_public_dict())

    def to_public_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": "true_rl_agent.stair.session_start_profile.v1",
            "profile_id": self.profile_id,
            "genesis_anchor_kind": self.genesis_anchor_kind,
            "task_protocol_hash": self.task_protocol_hash,
            "training_binding_hash": self.training_binding_hash,
        }
        if self.initial_policy_state_hash is not None:
            payload["initial_policy_state_hash"] = self.initial_policy_state_hash
            payload["bootstrap_initial_best_score"] = float(
                self.bootstrap_initial_best_score
            )
        return payload


def resolve_stair_session_start_profile(
    mode: Any,
    *,
    task_protocol_hash: str,
    training_binding_hash: str,
) -> StairSessionStartProfile:
    """Resolve a strict Stair-owned start profile from one public mode value."""

    if not isinstance(mode, str) or mode not in STAIR_START_MODES:
        raise ValueError(
            "Stair start_mode must be one of: "
            + ", ".join(sorted(STAIR_START_MODES))
        )
    if mode == STAIR_REFERENCE_START_MODE:
        return StairSessionStartProfile(
            profile_id=mode,
            genesis_anchor_kind="reference_program",
            task_protocol_hash=task_protocol_hash,
            training_binding_hash=training_binding_hash,
        )
    return StairSessionStartProfile(
        profile_id=mode,
        genesis_anchor_kind="bootstrap",
        task_protocol_hash=task_protocol_hash,
        training_binding_hash=training_binding_hash,
        initial_policy_state_hash=STAIR_FORMAL_INITIAL_POLICY_STATE_HASH,
        bootstrap_initial_best_score=0.0,
    )


__all__ = [
    "STAIR_DEFAULT_START_MODE",
    "STAIR_FORMAL_INITIAL_POLICY_STATE_HASH",
    "STAIR_REFERENCE_START_MODE",
    "STAIR_REWARD_COLD_START_MODE",
    "STAIR_START_MODES",
    "StairSessionStartProfile",
    "resolve_stair_session_start_profile",
]
