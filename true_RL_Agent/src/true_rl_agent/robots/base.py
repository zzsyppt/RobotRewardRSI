"""Task-neutral robot execution identity and semantic capability contracts."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping

from true_rl_agent.evaluation.binding import (
    canonical_json_hash,
    require_sha256_digest,
)

ROBOT_PROFILE_SCHEMA_VERSION = "true_rl_agent.robot_profile.v1"
ROBOT_RUNTIME_ADAPTER_SCHEMA_VERSION = "true_rl_agent.robot_runtime_adapter.v1"
ROBOT_REQUIREMENTS_SCHEMA_VERSION = "true_rl_agent.robot_requirements.v1"

BASE_ROLE = "base"
SUPPORT_EFFECTOR_ROLE = "support_effector"
FORBIDDEN_CONTACT_ROLE = "forbidden_contact"
JOINT_GROUP_ROLE_PREFIX = "joint_group."

_MUTUALLY_EXCLUSIVE_ROLE_PAIRS = frozenset(
    {
        frozenset({SUPPORT_EFFECTOR_ROLE, FORBIDDEN_CONTACT_ROLE}),
    }
)

_VERSIONED_ID = re.compile(r"[a-z0-9][a-z0-9_.-]*_v[1-9][0-9]*\Z")
_CAPABILITY_ID = re.compile(r"[a-z][a-z0-9_.-]*\Z")


def require_versioned_id(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or _VERSIONED_ID.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase versioned ID")
    return value


def require_capability_id(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or _CAPABILITY_ID.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a canonical capability ID")
    return value


def require_semantic_role_id(value: Any, field_name: str) -> str:
    if value in {BASE_ROLE, SUPPORT_EFFECTOR_ROLE, FORBIDDEN_CONTACT_ROLE}:
        return str(value)
    if (
        isinstance(value, str)
        and value.startswith(JOINT_GROUP_ROLE_PREFIX)
        and _CAPABILITY_ID.fullmatch(value) is not None
    ):
        return value
    raise ValueError(f"{field_name} is not in the shared semantic-role vocabulary")


def semantic_roles_are_mutually_exclusive(
    first_role_id: str,
    second_role_id: str,
) -> bool:
    """Return whether one concrete target cannot validly serve both roles."""

    require_semantic_role_id(first_role_id, "first semantic role")
    require_semantic_role_id(second_role_id, "second semantic role")
    return (
        frozenset({first_role_id, second_role_id})
        in _MUTUALLY_EXCLUSIVE_ROLE_PAIRS
    )


@dataclass(frozen=True)
class SemanticRole:
    """A task-neutral role mapped to concrete body or joint names."""

    role_id: str
    target_kind: str
    targets: tuple[str, ...]

    def __post_init__(self) -> None:
        require_semantic_role_id(self.role_id, "semantic role_id")
        if self.target_kind not in {"body", "joint"}:
            raise ValueError("semantic role target_kind must be body or joint")
        if not isinstance(self.targets, tuple) or not self.targets:
            raise ValueError("semantic role targets must be a non-empty tuple")
        if any(not isinstance(item, str) or not item for item in self.targets):
            raise ValueError("semantic role targets must be non-empty strings")
        if len(set(self.targets)) != len(self.targets):
            raise ValueError(
                f"semantic role contains duplicate targets: {self.role_id}"
            )

    @classmethod
    def from_mapping(cls, role_id: str, value: Any) -> "SemanticRole":
        if not isinstance(value, Mapping) or set(value) != {"target_kind", "targets"}:
            raise ValueError(f"semantic role fields mismatch: {role_id}")
        raw_targets = value.get("targets")
        if not isinstance(raw_targets, list):
            raise ValueError(f"semantic role targets must be a list: {role_id}")
        return cls(
            role_id=role_id,
            target_kind=value.get("target_kind"),
            targets=tuple(raw_targets),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_kind": self.target_kind,
            "targets": list(self.targets),
        }


@dataclass(frozen=True)
class SemanticRoleRequirement:
    """Cardinality and target-kind requirement declared by a TaskPlugin."""

    role_id: str
    target_kind: str
    minimum_count: int
    maximum_count: int | None = None
    require_all_profile_targets: bool = False

    def __post_init__(self) -> None:
        require_semantic_role_id(self.role_id, "semantic role requirement role_id")
        if self.target_kind not in {"body", "joint"}:
            raise ValueError("semantic role requirement target_kind is invalid")
        if (
            isinstance(self.minimum_count, bool)
            or not isinstance(self.minimum_count, int)
            or self.minimum_count < 0
        ):
            raise ValueError("semantic role requirement minimum_count is invalid")
        if self.maximum_count is not None:
            if (
                isinstance(self.maximum_count, bool)
                or not isinstance(self.maximum_count, int)
                or self.maximum_count < self.minimum_count
            ):
                raise ValueError("semantic role requirement maximum_count is invalid")
        if not isinstance(self.require_all_profile_targets, bool):
            raise TypeError(
                "semantic role requirement require_all_profile_targets must be boolean"
            )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "role_id": self.role_id,
            "target_kind": self.target_kind,
            "minimum_count": self.minimum_count,
            "maximum_count": self.maximum_count,
        }
        if self.require_all_profile_targets:
            payload["require_all_profile_targets"] = True
        return payload


@dataclass(frozen=True)
class RobotRequirements:
    """Small shared vocabulary consumed by the matrix compiler."""

    roles: tuple[SemanticRoleRequirement, ...] = ()
    capabilities: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if not isinstance(self.roles, tuple) or any(
            not isinstance(item, SemanticRoleRequirement) for item in self.roles
        ):
            raise TypeError("robot requirements roles must be role requirements")
        role_ids = tuple(item.role_id for item in self.roles)
        if len(set(role_ids)) != len(role_ids):
            raise ValueError("robot requirements contain duplicate roles")
        if not isinstance(self.capabilities, frozenset):
            raise TypeError("robot requirements capabilities must be a frozenset")
        for item in self.capabilities:
            require_capability_id(item, "robot requirement capability")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ROBOT_REQUIREMENTS_SCHEMA_VERSION,
            "roles": [item.to_dict() for item in self.roles],
            "capabilities": sorted(self.capabilities),
        }

    @property
    def requirements_hash(self) -> str:
        return canonical_json_hash(self.to_dict())


@dataclass(frozen=True)
class RobotProfile:
    """Exact execution embodiment; policy and task semantics are excluded."""

    robot_profile_id: str
    robot_family_id: str
    display_name: str
    runtime_facts: Mapping[str, Any]
    semantic_roles: Mapping[str, SemanticRole]
    provided_capabilities: frozenset[str]
    profile_hash: str | None = None

    def __post_init__(self) -> None:
        require_versioned_id(self.robot_profile_id, "robot_profile_id")
        if not isinstance(self.robot_family_id, str) or not self.robot_family_id:
            raise ValueError("robot_family_id must be non-empty")
        if not isinstance(self.display_name, str) or not self.display_name:
            raise ValueError("robot profile display_name must be non-empty")
        if not isinstance(self.runtime_facts, Mapping):
            raise TypeError("robot profile runtime_facts must be an object")
        runtime_facts = dict(self.runtime_facts)
        _validate_runtime_facts(runtime_facts, robot_family_id=self.robot_family_id)
        if not isinstance(self.semantic_roles, Mapping):
            raise TypeError("robot profile semantic_roles must be an object")
        roles = dict(self.semantic_roles)
        if any(
            key != role.role_id or not isinstance(role, SemanticRole)
            for key, role in roles.items()
        ):
            raise ValueError("robot profile semantic role mapping is invalid")
        _validate_role_targets(roles, runtime_facts=runtime_facts)
        if not isinstance(self.provided_capabilities, frozenset):
            raise TypeError("robot profile provided_capabilities must be a frozenset")
        for capability_id in self.provided_capabilities:
            require_capability_id(capability_id, "robot profile capability")
        expected = canonical_json_hash(self._hash_payload())
        if self.profile_hash is not None:
            require_sha256_digest(self.profile_hash, "robot profile hash")
            if self.profile_hash != expected:
                raise ValueError("robot profile hash mismatch")
        object.__setattr__(self, "runtime_facts", runtime_facts)
        object.__setattr__(self, "semantic_roles", roles)
        object.__setattr__(self, "profile_hash", expected)

    @classmethod
    def from_mapping(cls, value: Any) -> "RobotProfile":
        if not isinstance(value, Mapping):
            raise ValueError("robot profile must be an object")
        expected_fields = {
            "schema_version",
            "robot_profile_id",
            "robot_family_id",
            "display_name",
            "runtime_facts",
            "semantic_roles",
            "provided_capabilities",
        }
        if set(value) != expected_fields:
            raise ValueError("robot profile fields mismatch")
        if value.get("schema_version") != ROBOT_PROFILE_SCHEMA_VERSION:
            raise ValueError("robot profile schema_version mismatch")
        raw_roles = value.get("semantic_roles")
        if not isinstance(raw_roles, Mapping):
            raise ValueError("robot profile semantic_roles must be an object")
        raw_capabilities = value.get("provided_capabilities")
        if not isinstance(raw_capabilities, list):
            raise ValueError("robot profile provided_capabilities must be a list")
        if len(set(raw_capabilities)) != len(raw_capabilities):
            raise ValueError("robot profile provided_capabilities contain duplicates")
        return cls(
            robot_profile_id=value.get("robot_profile_id"),
            robot_family_id=value.get("robot_family_id"),
            display_name=value.get("display_name"),
            runtime_facts=value.get("runtime_facts"),
            semantic_roles={
                str(role_id): SemanticRole.from_mapping(str(role_id), role)
                for role_id, role in raw_roles.items()
            },
            provided_capabilities=frozenset(raw_capabilities),
        )

    @property
    def runtime_facts_hash(self) -> str:
        return str(self.runtime_facts["runtime_facts_hash"])

    def to_dict(self) -> dict[str, Any]:
        assert self.profile_hash is not None
        return {**self._hash_payload(), "profile_hash": self.profile_hash}

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "schema_version": ROBOT_PROFILE_SCHEMA_VERSION,
            "robot_profile_id": self.robot_profile_id,
            "robot_family_id": self.robot_family_id,
            "display_name": self.display_name,
            "runtime_facts": dict(self.runtime_facts),
            "semantic_roles": {
                key: self.semantic_roles[key].to_dict()
                for key in sorted(self.semantic_roles)
            },
            "provided_capabilities": sorted(self.provided_capabilities),
        }


@dataclass(frozen=True)
class RobotRuntimeAdapter:
    """Backend descriptor kept separate from task-owned environment execution."""

    runtime_adapter_id: str
    robot_profile_id: str
    backend_id: str
    runtime_provider: str
    upstream_revision: str
    articulation_source: str
    provided_capabilities: frozenset[str] = field(default_factory=frozenset)
    adapter_hash: str | None = None

    def __post_init__(self) -> None:
        require_versioned_id(self.runtime_adapter_id, "runtime_adapter_id")
        require_versioned_id(self.robot_profile_id, "runtime adapter robot_profile_id")
        for field_name in (
            "backend_id",
            "runtime_provider",
            "upstream_revision",
            "articulation_source",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"runtime adapter {field_name} must be non-empty")
        if not isinstance(self.provided_capabilities, frozenset):
            raise TypeError("runtime adapter provided_capabilities must be a frozenset")
        for capability_id in self.provided_capabilities:
            require_capability_id(capability_id, "runtime adapter capability")
        expected = canonical_json_hash(self._hash_payload())
        if self.adapter_hash is not None:
            require_sha256_digest(self.adapter_hash, "runtime adapter hash")
            if self.adapter_hash != expected:
                raise ValueError("runtime adapter hash mismatch")
        object.__setattr__(self, "adapter_hash", expected)

    def to_dict(self) -> dict[str, Any]:
        assert self.adapter_hash is not None
        return {**self._hash_payload(), "adapter_hash": self.adapter_hash}

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "schema_version": ROBOT_RUNTIME_ADAPTER_SCHEMA_VERSION,
            "runtime_adapter_id": self.runtime_adapter_id,
            "robot_profile_id": self.robot_profile_id,
            "backend_id": self.backend_id,
            "runtime_provider": self.runtime_provider,
            "upstream_revision": self.upstream_revision,
            "articulation_source": self.articulation_source,
            "provided_capabilities": sorted(self.provided_capabilities),
        }


def _validate_runtime_facts(
    runtime_facts: Mapping[str, Any],
    *,
    robot_family_id: str,
) -> None:
    required_fields = {
        "actuators",
        "asset_closure",
        "body_names",
        "default_state",
        "joint_names",
        "morphology",
        "profile_key",
        "robot_family_id",
        "runtime_facts_hash",
        "runtime_provider",
        "schema_version",
        "support_body_names",
        "upstream_revision",
    }
    if set(runtime_facts) != required_fields:
        raise ValueError("robot profile runtime_facts fields mismatch")
    if runtime_facts.get("robot_family_id") != robot_family_id:
        raise ValueError("robot profile family differs from runtime facts")
    declared_hash = require_sha256_digest(
        runtime_facts.get("runtime_facts_hash"),
        "runtime facts hash",
    )
    payload = {
        key: item for key, item in runtime_facts.items() if key != "runtime_facts_hash"
    }
    if canonical_json_hash(payload) != declared_hash:
        raise ValueError("robot profile runtime facts hash mismatch")
    for field_name in ("joint_names", "body_names", "support_body_names"):
        values = runtime_facts.get(field_name)
        if (
            not isinstance(values, list)
            or any(not isinstance(item, str) or not item for item in values)
            or len(set(values)) != len(values)
        ):
            raise ValueError(f"runtime facts {field_name} must be unique names")
    if not set(runtime_facts["support_body_names"]).issubset(
        runtime_facts["body_names"]
    ):
        raise ValueError("runtime facts support bodies are not robot bodies")


def _validate_role_targets(
    roles: Mapping[str, SemanticRole],
    *,
    runtime_facts: Mapping[str, Any],
) -> None:
    body_names = set(runtime_facts["body_names"])
    joint_names = set(runtime_facts["joint_names"])
    for role in roles.values():
        allowed = body_names if role.target_kind == "body" else joint_names
        unknown = sorted(set(role.targets) - allowed)
        if unknown:
            raise ValueError(
                f"semantic role {role.role_id} contains unknown targets: {unknown}"
            )


__all__ = [
    "BASE_ROLE",
    "FORBIDDEN_CONTACT_ROLE",
    "JOINT_GROUP_ROLE_PREFIX",
    "ROBOT_PROFILE_SCHEMA_VERSION",
    "ROBOT_REQUIREMENTS_SCHEMA_VERSION",
    "ROBOT_RUNTIME_ADAPTER_SCHEMA_VERSION",
    "SUPPORT_EFFECTOR_ROLE",
    "RobotProfile",
    "RobotRequirements",
    "RobotRuntimeAdapter",
    "SemanticRole",
    "SemanticRoleRequirement",
    "require_capability_id",
    "require_semantic_role_id",
    "require_versioned_id",
    "semantic_roles_are_mutually_exclusive",
]
