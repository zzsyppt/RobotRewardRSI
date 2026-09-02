"""Read frozen Stair transition V2 bytes into the morphology-neutral payload."""

from __future__ import annotations

from typing import Any, Mapping

from true_rl_agent.tasks.stair_climbing.runtime_profile import (
    STAIR_TRANSITION_RECORDER_SCHEMA_VERSION,
)
from true_rl_agent.tasks.stair_climbing.runtime_projection import (
    STAIR_TRANSITION_VNEXT_SCHEMA_VERSION,
)

STAIR_NORMALIZED_TRANSITION_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.transition.normalized.v1"
)
LEGACY_G1_SUPPORT_COUNT = 2
LEGACY_G1_FORBIDDEN_CONTACT_COUNT = 1

_NORMALIZED_COMMON_FIELDS = {
    "base_pos_w",
    "base_quat_w",
    "joint_pos",
    "episode_length",
    "terrain_env_origins_w",
    "terrain_types",
    "termination_term_order",
    "termination_masks",
    "done_mask",
}
_LEGACY_COMMON_FIELDS = {
    "root_pos_w",
    "root_quat_w",
    *(_NORMALIZED_COMMON_FIELDS - {"base_pos_w", "base_quat_w"}),
}
_LEGACY_FIELDS = {
    "schema_version",
    *_LEGACY_COMMON_FIELDS,
    "foot_pos_w",
    "foot_lin_vel_w",
    "foot_contact_forces_w",
    "base_contact_forces_w",
}
_VNEXT_FIELDS = {
    "schema_version",
    *_NORMALIZED_COMMON_FIELDS,
    "support_pos_w",
    "support_lin_vel_w",
    "support_contact_forces_w",
    "forbidden_contact_forces_w",
    "runtime_projection_hash",
    "reward_input_schema_hash_vnext",
    "support_count",
    "forbidden_contact_count",
}
_NORMALIZED_FIELDS = {
    "schema_version",
    "source_schema_version",
    *_NORMALIZED_COMMON_FIELDS,
    "support_pos_w",
    "support_lin_vel_w",
    "support_contact_forces_w",
    "forbidden_contact_forces_w",
    "runtime_projection_hash",
    "reward_input_schema_hash_vnext",
    "support_count",
    "forbidden_contact_count",
}


def normalize_stair_transition_payload(raw: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError("Stair transition recorder payload must be an object")
    schema_version = raw.get("schema_version")
    if schema_version == STAIR_NORMALIZED_TRANSITION_SCHEMA_VERSION:
        if set(raw) != _NORMALIZED_FIELDS:
            raise ValueError("Stair normalized transition fields mismatch")
        return dict(raw)
    if schema_version == STAIR_TRANSITION_RECORDER_SCHEMA_VERSION:
        if set(raw) != _LEGACY_FIELDS:
            raise ValueError("Stair legacy transition recorder fields mismatch")
        return {
            **{
                name: raw[name]
                for name in _LEGACY_COMMON_FIELDS
                if name not in {"root_pos_w", "root_quat_w"}
            },
            "base_pos_w": raw["root_pos_w"],
            "base_quat_w": raw["root_quat_w"],
            "schema_version": STAIR_NORMALIZED_TRANSITION_SCHEMA_VERSION,
            "source_schema_version": schema_version,
            "support_pos_w": raw["foot_pos_w"],
            "support_lin_vel_w": raw["foot_lin_vel_w"],
            "support_contact_forces_w": raw["foot_contact_forces_w"],
            "forbidden_contact_forces_w": raw["base_contact_forces_w"],
            "runtime_projection_hash": None,
            "reward_input_schema_hash_vnext": None,
            "support_count": LEGACY_G1_SUPPORT_COUNT,
            "forbidden_contact_count": LEGACY_G1_FORBIDDEN_CONTACT_COUNT,
        }
    if schema_version == STAIR_TRANSITION_VNEXT_SCHEMA_VERSION:
        if set(raw) != _VNEXT_FIELDS:
            raise ValueError("Stair VNext transition recorder fields mismatch")
        return {
            **{name: raw[name] for name in _NORMALIZED_COMMON_FIELDS},
            "schema_version": STAIR_NORMALIZED_TRANSITION_SCHEMA_VERSION,
            "source_schema_version": schema_version,
            "support_pos_w": raw["support_pos_w"],
            "support_lin_vel_w": raw["support_lin_vel_w"],
            "support_contact_forces_w": raw["support_contact_forces_w"],
            "forbidden_contact_forces_w": raw["forbidden_contact_forces_w"],
            "runtime_projection_hash": raw["runtime_projection_hash"],
            "reward_input_schema_hash_vnext": raw["reward_input_schema_hash_vnext"],
            "support_count": raw["support_count"],
            "forbidden_contact_count": raw["forbidden_contact_count"],
        }
    raise ValueError("Stair transition recorder schema mismatch")


__all__ = [
    "LEGACY_G1_SUPPORT_COUNT",
    "LEGACY_G1_FORBIDDEN_CONTACT_COUNT",
    "STAIR_NORMALIZED_TRANSITION_SCHEMA_VERSION",
    "normalize_stair_transition_payload",
]
