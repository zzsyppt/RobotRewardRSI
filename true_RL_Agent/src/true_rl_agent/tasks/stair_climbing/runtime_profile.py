"""Task-owned runtime profile binding for Stair Phase B."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from true_rl_agent.evaluation.binding import canonical_json_hash, require_sha256_digest
from true_rl_agent.tasks.stair_climbing.termination_capture import (
    STAIR_TERMINATION_CAPTURE_SCHEME_VERSION,
)

STAIR_PILOT_TRAIN_PROFILE = "stair_v2_draft_train"
STAIR_PILOT_EVAL_PROFILE = "stair_v2_draft_eval"
STAIR_TRAIN_PROFILE = "stair_v2_train"
STAIR_EVAL_PROFILE = "stair_v2_eval"
STAIR_UNIFIED_TRAIN_PROFILE = "stair_unified_v1_train"
STAIR_UNIFIED_EVAL_PROFILE = "stair_unified_v1_eval"
STAIR_ENV_BUILDER_VERSION = "stair_g1_rough_env_v3"
STAIR_UNIFIED_ENV_BUILDER_VERSION = "stair_unified_c0_c1_env_v1"
STAIR_LEGACY_TRANSITION_RECORDER_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.transition.v1"
)
STAIR_TRANSITION_RECORDER_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.transition.v2"
)
STAIR_EVAL_TRANSITION_EXTRAS_KEY = "stair_eval_pre_reset_transition_v2"
STAIR_TRAIN_GYM_ID = "TrueRL-Unitree-Stair-Climbing-G1-Train-v0"
STAIR_EVAL_GYM_ID = "TrueRL-Unitree-Stair-Climbing-G1-Eval-v0"
STAIR_COMPLETE_REWARD_RUNTIME_STATUS = "active_pre_reset_complete_reward_v1"
STAIR_REWARD_ACTIVATION_PREFLIGHT_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.reward_activation_preflight.v2"
)


@dataclass(frozen=True)
class StairRuntimeProfile:
    """Historical draft/pilot runtime hash retained for artifact replay."""

    mode: Literal["train", "eval"]
    profile_id: str
    base_protocol_hash: str
    draft_protocol_hash: str

    def __post_init__(self) -> None:
        if self.mode not in {"train", "eval"}:
            raise ValueError("stair runtime profile mode is invalid")
        expected_profile = (
            STAIR_PILOT_TRAIN_PROFILE
            if self.mode == "train"
            else STAIR_PILOT_EVAL_PROFILE
        )
        if self.profile_id != expected_profile:
            raise ValueError("stair runtime profile ID mismatch")
        require_sha256_digest(
            self.base_protocol_hash,
            "stair runtime base protocol hash",
        )
        require_sha256_digest(
            self.draft_protocol_hash,
            "stair runtime draft protocol hash",
        )

    @property
    def runtime_dynamics_hash(self) -> str:
        return "sha256:" + canonical_json_hash(self._hash_payload())

    def to_public_dict(self) -> dict[str, Any]:
        return {
            **self._hash_payload(),
            "runtime_dynamics_hash": self.runtime_dynamics_hash,
        }

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "schema_version": "true_rl_agent.task.stair_climbing.runtime_profile.v1",
            "mode": self.mode,
            "profile_id": self.profile_id,
            "base_protocol_hash": self.base_protocol_hash,
            "draft_protocol_hash": self.draft_protocol_hash,
            "env_builder_version": STAIR_ENV_BUILDER_VERSION,
            "transition_recorder_schema_version": (
                STAIR_LEGACY_TRANSITION_RECORDER_SCHEMA_VERSION
                if self.mode == "eval"
                else None
            ),
            "transition_recorder_enabled": self.mode == "eval",
            "termination_capture_scheme_version": (
                STAIR_TERMINATION_CAPTURE_SCHEME_VERSION
            ),
            "complete_reward_runtime": STAIR_COMPLETE_REWARD_RUNTIME_STATUS,
        }


@dataclass(frozen=True)
class StairProductionRuntimeProfile:
    """Opaque backend-facing hash of final task-owned runtime semantics."""

    mode: Literal["train", "eval"]
    profile_id: str
    base_protocol_hash: str
    task_protocol_hash: str

    def __post_init__(self) -> None:
        if self.mode not in {"train", "eval"}:
            raise ValueError("stair production runtime profile mode is invalid")
        expected_profiles = (
            {STAIR_TRAIN_PROFILE, STAIR_UNIFIED_TRAIN_PROFILE}
            if self.mode == "train"
            else {STAIR_EVAL_PROFILE, STAIR_UNIFIED_EVAL_PROFILE}
        )
        if self.profile_id not in expected_profiles:
            raise ValueError("stair production runtime profile ID mismatch")
        require_sha256_digest(
            self.base_protocol_hash,
            "stair production runtime base protocol hash",
        )
        require_sha256_digest(
            self.task_protocol_hash,
            "stair production runtime task protocol hash",
        )

    @property
    def runtime_dynamics_hash(self) -> str:
        return "sha256:" + canonical_json_hash(self._hash_payload())

    def to_public_dict(self) -> dict[str, Any]:
        return {
            **self._hash_payload(),
            "runtime_dynamics_hash": self.runtime_dynamics_hash,
        }

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "schema_version": "true_rl_agent.task.stair_climbing.runtime_profile.v2",
            "mode": self.mode,
            "profile_id": self.profile_id,
            "base_protocol_hash": self.base_protocol_hash,
            "task_protocol_hash": self.task_protocol_hash,
            "env_builder_version": (
                STAIR_UNIFIED_ENV_BUILDER_VERSION
                if self.profile_id
                in {STAIR_UNIFIED_TRAIN_PROFILE, STAIR_UNIFIED_EVAL_PROFILE}
                else STAIR_ENV_BUILDER_VERSION
            ),
            "transition_recorder_schema_version": (
                STAIR_TRANSITION_RECORDER_SCHEMA_VERSION
                if self.mode == "eval"
                else None
            ),
            "transition_recorder_enabled": self.mode == "eval",
            "termination_capture_scheme_version": (
                STAIR_TERMINATION_CAPTURE_SCHEME_VERSION
            ),
            "complete_reward_runtime": STAIR_COMPLETE_REWARD_RUNTIME_STATUS,
        }


__all__ = [
    "STAIR_COMPLETE_REWARD_RUNTIME_STATUS",
    "STAIR_ENV_BUILDER_VERSION",
    "STAIR_EVAL_GYM_ID",
    "STAIR_EVAL_PROFILE",
    "STAIR_PILOT_EVAL_PROFILE",
    "STAIR_PILOT_TRAIN_PROFILE",
    "STAIR_EVAL_TRANSITION_EXTRAS_KEY",
    "STAIR_LEGACY_TRANSITION_RECORDER_SCHEMA_VERSION",
    "STAIR_REWARD_ACTIVATION_PREFLIGHT_SCHEMA_VERSION",
    "STAIR_TRAIN_GYM_ID",
    "STAIR_TRAIN_PROFILE",
    "STAIR_UNIFIED_TRAIN_PROFILE",
    "STAIR_UNIFIED_EVAL_PROFILE",
    "STAIR_UNIFIED_ENV_BUILDER_VERSION",
    "STAIR_TRANSITION_RECORDER_SCHEMA_VERSION",
    "StairProductionRuntimeProfile",
    "StairRuntimeProfile",
]
