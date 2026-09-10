"""Robot x task composition contracts and deterministic resolved manifests."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from true_rl_agent.evaluation.binding import (
    canonical_json_hash,
    require_sha256_digest,
)
from true_rl_agent.json_contract import (
    canonical_json_bytes,
    load_strict_json_object,
)
from true_rl_agent.fs_contract import create_once_bytes
from true_rl_agent.robots.base import (
    RobotProfile,
    RobotRequirements,
    require_capability_id,
    require_semantic_role_id,
    require_versioned_id,
    semantic_roles_are_mutually_exclusive,
)
from true_rl_agent.robots.registry import RobotRegistry

ROBOT_TASK_BINDING_SCHEMA_VERSION = "true_rl_agent.robot_task_binding.v1"
CELL_ENVIRONMENT_SCHEMA_VERSION = "true_rl_agent.cell_environment.v1"
RESOLVED_CELL_MANIFEST_SCHEMA_VERSION = "true_rl_agent.resolved_cell_manifest.v1"
RESOLVED_CELL_PROTOCOL_SCHEMA_VERSION = "true_rl_agent.resolved_cell_task_protocol.v1"
RESOLVED_CELL_MANIFEST_FILENAME = "resolved_cell_manifest.json"
LEGACY_TRAINING_BINDING_V1_GRANT_SCHEMA_VERSION = (
    "true_rl_agent.legacy_training_binding_v1_grant.v1"
)
CELL_PROTOCOL_STATES = frozenset({"draft", "active", "archived"})
CELL_PROTOCOL_STATE_SCHEMA_VERSION = "true_rl_agent.cell_protocol_state.v1"


@dataclass(frozen=True)
class RobotTaskBinding:
    """Task-owned projection from shared roles to one exact execution profile."""

    binding_id: str
    cell_variant_id: str
    task_id: str
    robot_profile_id: str
    runtime_adapter_id: str
    role_bindings: Mapping[str, tuple[str, ...]]
    provided_capabilities: frozenset[str]
    task_projection: Mapping[str, Any]
    binding_hash: str | None = None

    def __post_init__(self) -> None:
        require_versioned_id(self.binding_id, "robot task binding_id")
        require_versioned_id(self.cell_variant_id, "cell_variant_id")
        if not isinstance(self.task_id, str) or not self.task_id:
            raise ValueError("robot task binding task_id must be non-empty")
        require_versioned_id(self.robot_profile_id, "robot task profile_id")
        require_versioned_id(self.runtime_adapter_id, "robot task adapter_id")
        if not isinstance(self.role_bindings, Mapping):
            raise TypeError("robot task role_bindings must be an object")
        normalized_roles: dict[str, tuple[str, ...]] = {}
        for role_id, targets in self.role_bindings.items():
            require_semantic_role_id(role_id, "robot task role binding")
            if not isinstance(targets, tuple) or not targets:
                raise ValueError(
                    f"robot task role binding must be a non-empty tuple: {role_id}"
                )
            if any(not isinstance(item, str) or not item for item in targets):
                raise ValueError(f"robot task role target is invalid: {role_id}")
            if len(set(targets)) != len(targets):
                raise ValueError(f"robot task role target is duplicated: {role_id}")
            normalized_roles[role_id] = targets
        if not isinstance(self.provided_capabilities, frozenset):
            raise TypeError("binding provided_capabilities must be a frozenset")
        for capability_id in self.provided_capabilities:
            require_capability_id(capability_id, "binding capability")
        if not isinstance(self.task_projection, Mapping):
            raise TypeError("robot task projection must be an object")
        object.__setattr__(
            self,
            "role_bindings",
            MappingProxyType(normalized_roles),
        )
        object.__setattr__(
            self,
            "task_projection",
            MappingProxyType(dict(self.task_projection)),
        )
        expected = canonical_json_hash(self._hash_payload())
        if self.binding_hash is not None:
            require_sha256_digest(self.binding_hash, "robot task binding hash")
            if self.binding_hash != expected:
                raise ValueError("robot task binding hash mismatch")
        object.__setattr__(self, "binding_hash", expected)

    @classmethod
    def from_mapping(cls, value: Any) -> "RobotTaskBinding":
        if not isinstance(value, Mapping):
            raise ValueError("robot task binding must be an object")
        expected_fields = {
            "schema_version",
            "binding_id",
            "cell_variant_id",
            "task_id",
            "robot_profile_id",
            "runtime_adapter_id",
            "role_bindings",
            "provided_capabilities",
            "task_projection",
            "binding_hash",
        }
        if set(value) != expected_fields:
            raise ValueError("robot task binding fields mismatch")
        if value.get("schema_version") != ROBOT_TASK_BINDING_SCHEMA_VERSION:
            raise ValueError("robot task binding schema_version mismatch")
        raw_roles = value.get("role_bindings")
        if not isinstance(raw_roles, Mapping):
            raise ValueError("robot task binding role_bindings must be an object")
        raw_capabilities = value.get("provided_capabilities")
        if not isinstance(raw_capabilities, list):
            raise ValueError("robot task binding provided_capabilities must be a list")
        if len(set(raw_capabilities)) != len(raw_capabilities):
            raise ValueError(
                "robot task binding provided_capabilities contain duplicates"
            )
        return cls(
            binding_id=value.get("binding_id"),
            cell_variant_id=value.get("cell_variant_id"),
            task_id=value.get("task_id"),
            robot_profile_id=value.get("robot_profile_id"),
            runtime_adapter_id=value.get("runtime_adapter_id"),
            role_bindings={
                str(role_id): tuple(targets) if isinstance(targets, list) else ()
                for role_id, targets in raw_roles.items()
            },
            provided_capabilities=frozenset(raw_capabilities),
            task_projection=value.get("task_projection"),
            binding_hash=value.get("binding_hash"),
        )

    def to_dict(self) -> dict[str, Any]:
        assert self.binding_hash is not None
        return {**self._hash_payload(), "binding_hash": self.binding_hash}

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "schema_version": ROBOT_TASK_BINDING_SCHEMA_VERSION,
            "binding_id": self.binding_id,
            "cell_variant_id": self.cell_variant_id,
            "task_id": self.task_id,
            "robot_profile_id": self.robot_profile_id,
            "runtime_adapter_id": self.runtime_adapter_id,
            "role_bindings": {
                key: list(self.role_bindings[key]) for key in sorted(self.role_bindings)
            },
            "provided_capabilities": sorted(self.provided_capabilities),
            "task_projection": dict(self.task_projection),
        }


@dataclass(frozen=True)
class CellEnvironmentDescriptor:
    """Opaque, task-owned environment identity without train/eval authority."""

    environment_id: str
    cell_variant_id: str
    task_id: str
    payload: Mapping[str, Any]
    environment_hash: str | None = None

    def __post_init__(self) -> None:
        require_versioned_id(self.environment_id, "environment_id")
        require_versioned_id(self.cell_variant_id, "environment cell_variant_id")
        if not isinstance(self.task_id, str) or not self.task_id:
            raise ValueError("cell environment task_id must be non-empty")
        if not isinstance(self.payload, Mapping):
            raise TypeError("cell environment payload must be an object")
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))
        expected = canonical_json_hash(self._hash_payload())
        if self.environment_hash is not None:
            require_sha256_digest(self.environment_hash, "cell environment hash")
            if self.environment_hash != expected:
                raise ValueError("cell environment hash mismatch")
        object.__setattr__(self, "environment_hash", expected)

    def to_dict(self) -> dict[str, Any]:
        assert self.environment_hash is not None
        return {**self._hash_payload(), "environment_hash": self.environment_hash}

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "schema_version": CELL_ENVIRONMENT_SCHEMA_VERSION,
            "environment_id": self.environment_id,
            "cell_variant_id": self.cell_variant_id,
            "task_id": self.task_id,
            "payload": dict(self.payload),
        }


@dataclass(frozen=True)
class CellVariantDefinition:
    cell_variant_id: str
    robot_task_cell_id: str
    task_id: str
    robot_profile_id: str
    runtime_adapter_id: str
    robot_task_binding: RobotTaskBinding
    environment: CellEnvironmentDescriptor
    contract_hashes: Mapping[str, str]
    legacy_training_binding_v1_grant_hash: str | None = None
    protocol_state: str = "active"

    def __post_init__(self) -> None:
        require_versioned_id(self.cell_variant_id, "cell_variant_id")
        require_versioned_id(self.robot_task_cell_id, "robot_task_cell_id")
        if not isinstance(self.task_id, str) or not self.task_id:
            raise ValueError("cell task_id must be non-empty")
        require_versioned_id(self.robot_profile_id, "cell robot_profile_id")
        require_versioned_id(self.runtime_adapter_id, "cell runtime_adapter_id")
        if not isinstance(self.robot_task_binding, RobotTaskBinding):
            raise TypeError("cell robot_task_binding is invalid")
        if not isinstance(self.environment, CellEnvironmentDescriptor):
            raise TypeError("cell environment is invalid")
        expected_binding_identity = (
            self.cell_variant_id,
            self.task_id,
        )
        binding_identity = (
            self.robot_task_binding.cell_variant_id,
            self.robot_task_binding.task_id,
        )
        if binding_identity != expected_binding_identity:
            raise ValueError("cell robot task binding identity mismatch")
        if (
            self.environment.cell_variant_id,
            self.environment.task_id,
        ) != (self.cell_variant_id, self.task_id):
            raise ValueError("cell environment identity mismatch")
        if not isinstance(self.contract_hashes, Mapping) or not self.contract_hashes:
            raise ValueError("cell contract_hashes must be a non-empty object")
        normalized_hashes: dict[str, str] = {}
        for name, value in self.contract_hashes.items():
            require_capability_id(name, "cell contract name")
            normalized_hashes[name] = require_sha256_digest(
                value,
                f"cell contract {name}",
            )
        if self.legacy_training_binding_v1_grant_hash is not None:
            require_sha256_digest(
                self.legacy_training_binding_v1_grant_hash,
                "legacy TrainingBindingV1 grant hash",
            )
        if self.protocol_state not in CELL_PROTOCOL_STATES:
            raise ValueError("cell protocol_state is invalid")
        object.__setattr__(
            self,
            "contract_hashes",
            MappingProxyType(normalized_hashes),
        )

    @property
    def protocol_state_hash(self) -> str:
        return canonical_json_hash(
            {
                "schema_version": CELL_PROTOCOL_STATE_SCHEMA_VERSION,
                "cell_variant_id": self.cell_variant_id,
                "protocol_state": self.protocol_state,
            }
        )


def grant_legacy_training_binding_v1(
    definition: CellVariantDefinition,
    *,
    robot_registry: RobotRegistry,
    requirements: RobotRequirements,
    base_task_protocol_hash: str,
) -> CellVariantDefinition:
    """Bind the V1 compatibility exception to one exact trusted cell contract."""

    profile, adapter = robot_registry.execution(
        robot_profile_id=definition.robot_profile_id,
        runtime_adapter_id=definition.runtime_adapter_id,
    )
    grant_hash = _legacy_training_binding_v1_grant_hash(
        definition=definition,
        profile=profile,
        runtime_adapter_hash=str(adapter.adapter_hash),
        requirements=requirements,
        base_task_protocol_hash=base_task_protocol_hash,
    )
    return replace(
        definition,
        legacy_training_binding_v1_grant_hash=grant_hash,
    )


def _legacy_training_binding_v1_grant_hash(
    *,
    definition: CellVariantDefinition,
    profile: RobotProfile,
    runtime_adapter_hash: str,
    requirements: RobotRequirements,
    base_task_protocol_hash: str,
) -> str:
    assert profile.profile_hash is not None
    assert definition.robot_task_binding.binding_hash is not None
    assert definition.environment.environment_hash is not None
    return _legacy_training_binding_v1_grant_hash_from_values(
        cell_variant_id=definition.cell_variant_id,
        robot_task_cell_id=definition.robot_task_cell_id,
        task_id=definition.task_id,
        base_task_protocol_hash=base_task_protocol_hash,
        robot_profile_hash=profile.profile_hash,
        runtime_adapter_hash=runtime_adapter_hash,
        robot_task_binding_hash=definition.robot_task_binding.binding_hash,
        environment_hash=definition.environment.environment_hash,
        task_requirements_hash=requirements.requirements_hash,
        contract_hashes=definition.contract_hashes,
    )


def _legacy_training_binding_v1_grant_hash_from_values(
    *,
    cell_variant_id: str,
    robot_task_cell_id: str,
    task_id: str,
    base_task_protocol_hash: str,
    robot_profile_hash: str,
    runtime_adapter_hash: str,
    robot_task_binding_hash: str,
    environment_hash: str,
    task_requirements_hash: str,
    contract_hashes: Mapping[str, str],
) -> str:
    for field_name, value in (
        ("base task protocol hash", base_task_protocol_hash),
        ("robot profile hash", robot_profile_hash),
        ("runtime adapter hash", runtime_adapter_hash),
        ("robot task binding hash", robot_task_binding_hash),
        ("environment hash", environment_hash),
        ("task requirements hash", task_requirements_hash),
    ):
        require_sha256_digest(value, field_name)
    return canonical_json_hash(
        {
            "schema_version": LEGACY_TRAINING_BINDING_V1_GRANT_SCHEMA_VERSION,
            "cell_variant_id": cell_variant_id,
            "robot_task_cell_id": robot_task_cell_id,
            "task_id": task_id,
            "base_task_protocol_hash": base_task_protocol_hash,
            "robot_profile_hash": robot_profile_hash,
            "runtime_adapter_hash": runtime_adapter_hash,
            "robot_task_binding_hash": robot_task_binding_hash,
            "environment_hash": environment_hash,
            "task_requirements_hash": task_requirements_hash,
            "contract_hashes": dict(contract_hashes),
        }
    )


class CellVariantCatalog:
    """Closed catalog selected by one task provider."""

    def __init__(self) -> None:
        self._items: dict[str, CellVariantDefinition] = {}

    def register(self, definition: CellVariantDefinition) -> None:
        if not isinstance(definition, CellVariantDefinition):
            raise TypeError("cell catalog entry must be CellVariantDefinition")
        if definition.cell_variant_id in self._items:
            raise ValueError(
                f"cell variant already registered: {definition.cell_variant_id}"
            )
        self._items[definition.cell_variant_id] = definition

    def resolve(self, cell_variant_id: str) -> CellVariantDefinition:
        try:
            return self._items[cell_variant_id]
        except KeyError as exc:
            raise KeyError(f"unknown cell variant: {cell_variant_id}") from exc

    def ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._items))

    def require_active(self, cell_variant_id: str) -> CellVariantDefinition:
        definition = self.resolve(cell_variant_id)
        if definition.protocol_state != "active":
            raise ValueError(
                "cell protocol is not active: "
                f"{cell_variant_id} ({definition.protocol_state})"
            )
        return definition


@dataclass(frozen=True)
class ResolvedCellManifestV1:
    cell_variant_id: str
    robot_task_cell_id: str
    task_id: str
    base_task_protocol_hash: str
    robot_profile: Mapping[str, Any]
    runtime_adapter: Mapping[str, Any]
    robot_task_binding: Mapping[str, Any]
    environment: Mapping[str, Any]
    task_requirements: Mapping[str, Any]
    resolved_roles: Mapping[str, tuple[str, ...]]
    provided_capabilities: tuple[str, ...]
    contract_hashes: Mapping[str, str]
    legacy_training_binding_v1_grant_hash: str | None
    manifest_hash: str | None = None
    resolved_cell_task_protocol_hash: str | None = None

    def __post_init__(self) -> None:
        require_versioned_id(self.cell_variant_id, "manifest cell_variant_id")
        require_versioned_id(self.robot_task_cell_id, "manifest robot_task_cell_id")
        if not isinstance(self.task_id, str) or not self.task_id:
            raise ValueError("manifest task_id must be non-empty")
        require_sha256_digest(
            self.base_task_protocol_hash,
            "manifest base task protocol hash",
        )
        for name in (
            "robot_profile",
            "runtime_adapter",
            "robot_task_binding",
            "environment",
            "task_requirements",
            "resolved_roles",
            "contract_hashes",
        ):
            if not isinstance(getattr(self, name), Mapping):
                raise TypeError(f"manifest {name} must be an object")
        normalized_roles: dict[str, tuple[str, ...]] = {}
        for role_id, targets in self.resolved_roles.items():
            require_semantic_role_id(role_id, "manifest resolved role")
            if not isinstance(targets, tuple) or not targets:
                raise ValueError("manifest resolved roles are invalid")
            normalized_roles[role_id] = targets
        if not isinstance(self.provided_capabilities, tuple):
            raise TypeError("manifest capabilities must be a tuple")
        if tuple(sorted(set(self.provided_capabilities))) != self.provided_capabilities:
            raise ValueError("manifest capabilities must be sorted and unique")
        for capability_id in self.provided_capabilities:
            require_capability_id(capability_id, "manifest capability")
        if self.legacy_training_binding_v1_grant_hash is not None:
            require_sha256_digest(
                self.legacy_training_binding_v1_grant_hash,
                "manifest legacy TrainingBindingV1 grant hash",
            )
            expected_grant_hash = (
                _legacy_training_binding_v1_grant_hash_from_values(
                    cell_variant_id=self.cell_variant_id,
                    robot_task_cell_id=self.robot_task_cell_id,
                    task_id=self.task_id,
                    base_task_protocol_hash=self.base_task_protocol_hash,
                    robot_profile_hash=str(self.robot_profile.get("profile_hash")),
                    runtime_adapter_hash=str(
                        self.runtime_adapter.get("adapter_hash")
                    ),
                    robot_task_binding_hash=str(
                        self.robot_task_binding.get("binding_hash")
                    ),
                    environment_hash=str(self.environment.get("environment_hash")),
                    task_requirements_hash=canonical_json_hash(
                        dict(self.task_requirements)
                    ),
                    contract_hashes=self.contract_hashes,
                )
            )
            if self.legacy_training_binding_v1_grant_hash != expected_grant_hash:
                raise ValueError(
                    "manifest legacy TrainingBindingV1 grant hash mismatch"
                )
        object.__setattr__(self, "resolved_roles", MappingProxyType(normalized_roles))
        object.__setattr__(
            self, "robot_profile", MappingProxyType(dict(self.robot_profile))
        )
        object.__setattr__(
            self, "runtime_adapter", MappingProxyType(dict(self.runtime_adapter))
        )
        object.__setattr__(
            self, "robot_task_binding", MappingProxyType(dict(self.robot_task_binding))
        )
        object.__setattr__(
            self, "environment", MappingProxyType(dict(self.environment))
        )
        object.__setattr__(
            self, "task_requirements", MappingProxyType(dict(self.task_requirements))
        )
        object.__setattr__(
            self, "contract_hashes", MappingProxyType(dict(self.contract_hashes))
        )
        expected_manifest_hash = canonical_json_hash(self._hash_payload())
        if self.manifest_hash is not None:
            require_sha256_digest(self.manifest_hash, "resolved cell manifest hash")
            if self.manifest_hash != expected_manifest_hash:
                raise ValueError("resolved cell manifest hash mismatch")
        expected_protocol_hash = canonical_json_hash(
            {
                "schema_version": RESOLVED_CELL_PROTOCOL_SCHEMA_VERSION,
                "base_task_protocol_hash": self.base_task_protocol_hash,
                "resolved_cell_manifest_hash": expected_manifest_hash,
            }
        )
        if self.resolved_cell_task_protocol_hash is not None:
            require_sha256_digest(
                self.resolved_cell_task_protocol_hash,
                "resolved cell task protocol hash",
            )
            if self.resolved_cell_task_protocol_hash != expected_protocol_hash:
                raise ValueError("resolved cell task protocol hash mismatch")
        object.__setattr__(self, "manifest_hash", expected_manifest_hash)
        object.__setattr__(
            self,
            "resolved_cell_task_protocol_hash",
            expected_protocol_hash,
        )

    @classmethod
    def from_mapping(cls, value: Any) -> "ResolvedCellManifestV1":
        if not isinstance(value, Mapping):
            raise ValueError("resolved cell manifest must be an object")
        expected_fields = {
            "schema_version",
            "cell_variant_id",
            "robot_task_cell_id",
            "task_id",
            "base_task_protocol_hash",
            "robot_profile",
            "runtime_adapter",
            "robot_task_binding",
            "environment",
            "task_requirements",
            "resolved_roles",
            "provided_capabilities",
            "contract_hashes",
            "legacy_training_binding_v1_grant_hash",
            "manifest_hash",
            "resolved_cell_task_protocol_hash",
        }
        if set(value) != expected_fields:
            raise ValueError("resolved cell manifest fields mismatch")
        if value.get("schema_version") != RESOLVED_CELL_MANIFEST_SCHEMA_VERSION:
            raise ValueError("resolved cell manifest schema_version mismatch")
        raw_roles = value.get("resolved_roles")
        if not isinstance(raw_roles, Mapping):
            raise ValueError("resolved cell manifest roles must be an object")
        capabilities = value.get("provided_capabilities")
        if not isinstance(capabilities, list):
            raise ValueError("resolved cell manifest capabilities must be a list")
        return cls(
            cell_variant_id=value.get("cell_variant_id"),
            robot_task_cell_id=value.get("robot_task_cell_id"),
            task_id=value.get("task_id"),
            base_task_protocol_hash=value.get("base_task_protocol_hash"),
            robot_profile=value.get("robot_profile"),
            runtime_adapter=value.get("runtime_adapter"),
            robot_task_binding=value.get("robot_task_binding"),
            environment=value.get("environment"),
            task_requirements=value.get("task_requirements"),
            resolved_roles={
                str(role_id): tuple(targets) if isinstance(targets, list) else ()
                for role_id, targets in raw_roles.items()
            },
            provided_capabilities=tuple(capabilities),
            contract_hashes=value.get("contract_hashes"),
            legacy_training_binding_v1_grant_hash=value.get(
                "legacy_training_binding_v1_grant_hash"
            ),
            manifest_hash=value.get("manifest_hash"),
            resolved_cell_task_protocol_hash=value.get(
                "resolved_cell_task_protocol_hash"
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        assert self.manifest_hash is not None
        assert self.resolved_cell_task_protocol_hash is not None
        return {
            **self._hash_payload(),
            "manifest_hash": self.manifest_hash,
            "resolved_cell_task_protocol_hash": (self.resolved_cell_task_protocol_hash),
        }

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "schema_version": RESOLVED_CELL_MANIFEST_SCHEMA_VERSION,
            "cell_variant_id": self.cell_variant_id,
            "robot_task_cell_id": self.robot_task_cell_id,
            "task_id": self.task_id,
            "base_task_protocol_hash": self.base_task_protocol_hash,
            "robot_profile": dict(self.robot_profile),
            "runtime_adapter": dict(self.runtime_adapter),
            "robot_task_binding": dict(self.robot_task_binding),
            "environment": dict(self.environment),
            "task_requirements": dict(self.task_requirements),
            "resolved_roles": {
                key: list(self.resolved_roles[key])
                for key in sorted(self.resolved_roles)
            },
            "provided_capabilities": list(self.provided_capabilities),
            "contract_hashes": dict(self.contract_hashes),
            "legacy_training_binding_v1_grant_hash": (
                self.legacy_training_binding_v1_grant_hash
            ),
        }

    @property
    def uses_legacy_training_binding_v1(self) -> bool:
        return self.legacy_training_binding_v1_grant_hash is not None

    def summary_identity(self) -> dict[str, str]:
        return {
            "cell_variant_id": self.cell_variant_id,
            "robot_task_cell_id": self.robot_task_cell_id,
            "robot_profile_id": str(self.robot_profile["robot_profile_id"]),
            "runtime_adapter_id": str(self.runtime_adapter["runtime_adapter_id"]),
            "robot_task_binding_id": str(self.robot_task_binding["binding_id"]),
            "environment_id": str(self.environment["environment_id"]),
            "resolved_cell_task_protocol_hash": str(
                self.resolved_cell_task_protocol_hash
            ),
        }


@dataclass(frozen=True)
class CellResolution:
    status: str
    reasons: tuple[str, ...] = ()
    manifest: ResolvedCellManifestV1 | None = None

    def __post_init__(self) -> None:
        if self.status == "supported":
            if self.manifest is None or self.reasons:
                raise ValueError(
                    "supported cell resolution must contain only a manifest"
                )
        elif self.status == "unsupported":
            if self.manifest is not None or not self.reasons:
                raise ValueError("unsupported cell resolution must contain reasons")
        else:
            raise ValueError("cell resolution status is invalid")

    @property
    def supported(self) -> bool:
        return self.status == "supported"

    def require_supported(self) -> ResolvedCellManifestV1:
        if not self.supported or self.manifest is None:
            raise ValueError("unsupported robot-task cell: " + "; ".join(self.reasons))
        return self.manifest


def compile_cell_manifest(
    *,
    definition: CellVariantDefinition,
    robot_registry: RobotRegistry,
    requirements: RobotRequirements,
    base_task_protocol_hash: str,
) -> CellResolution:
    """Resolve exact identities and capabilities without importing a simulator."""

    require_sha256_digest(base_task_protocol_hash, "base task protocol hash")
    reasons: list[str] = []
    try:
        profile, adapter = robot_registry.execution(
            robot_profile_id=definition.robot_profile_id,
            runtime_adapter_id=definition.runtime_adapter_id,
        )
    except (KeyError, ValueError) as exc:
        reasons.append(f"execution_identity:{exc}")
        return CellResolution("unsupported", tuple(sorted(reasons)))
    binding = definition.robot_task_binding
    if binding.robot_profile_id != profile.robot_profile_id:
        reasons.append("profile_binding_mismatch")
    if binding.runtime_adapter_id != adapter.runtime_adapter_id:
        reasons.append("adapter_binding_mismatch")
    resolved_roles: dict[str, tuple[str, ...]] = {}
    selected_roles_by_target: dict[tuple[str, str], set[str]] = {}
    for requirement in requirements.roles:
        profile_role = profile.semantic_roles.get(requirement.role_id)
        selected = binding.role_bindings.get(requirement.role_id)
        if profile_role is None:
            reasons.append(f"missing_profile_role:{requirement.role_id}")
            continue
        if selected is None:
            reasons.append(f"missing_binding_role:{requirement.role_id}")
            continue
        if profile_role.target_kind != requirement.target_kind:
            reasons.append(f"role_kind_mismatch:{requirement.role_id}")
            continue
        unknown = sorted(set(selected) - set(profile_role.targets))
        if unknown:
            reasons.append(f"role_target_outside_profile:{requirement.role_id}")
            continue
        count = len(selected)
        if count < requirement.minimum_count or (
            requirement.maximum_count is not None and count > requirement.maximum_count
        ):
            reasons.append(f"role_cardinality:{requirement.role_id}")
            continue
        if requirement.require_all_profile_targets and tuple(selected) != tuple(
            profile_role.targets
        ):
            reasons.append(f"role_selection_incomplete:{requirement.role_id}")
            continue
        for target in selected:
            target_key = (requirement.target_kind, target)
            previous_roles = selected_roles_by_target.setdefault(target_key, set())
            for previous_role in previous_roles:
                if semantic_roles_are_mutually_exclusive(
                    previous_role,
                    requirement.role_id,
                ):
                    first_role, second_role = sorted(
                        (previous_role, requirement.role_id)
                    )
                    reasons.append(
                        "conflicting_role_target:"
                        f"{first_role}:{second_role}:{target}"
                    )
            previous_roles.add(requirement.role_id)
        resolved_roles[requirement.role_id] = tuple(selected)
    unknown_binding_roles = sorted(
        set(binding.role_bindings) - {item.role_id for item in requirements.roles}
    )
    if unknown_binding_roles:
        reasons.extend(
            f"unrequested_binding_role:{item}" for item in unknown_binding_roles
        )
    provided_capabilities = frozenset().union(
        profile.provided_capabilities,
        adapter.provided_capabilities,
        binding.provided_capabilities,
    )
    missing_capabilities = sorted(requirements.capabilities - provided_capabilities)
    reasons.extend(f"missing_capability:{item}" for item in missing_capabilities)
    grant_hash = definition.legacy_training_binding_v1_grant_hash
    if grant_hash is not None:
        expected_grant_hash = _legacy_training_binding_v1_grant_hash(
            definition=definition,
            profile=profile,
            runtime_adapter_hash=str(adapter.adapter_hash),
            requirements=requirements,
            base_task_protocol_hash=base_task_protocol_hash,
        )
        if grant_hash != expected_grant_hash:
            reasons.append("legacy_training_binding_v1_grant_mismatch")
    if reasons:
        return CellResolution("unsupported", tuple(sorted(set(reasons))))
    manifest = ResolvedCellManifestV1(
        cell_variant_id=definition.cell_variant_id,
        robot_task_cell_id=definition.robot_task_cell_id,
        task_id=definition.task_id,
        base_task_protocol_hash=base_task_protocol_hash,
        robot_profile=_profile_manifest_descriptor(profile),
        runtime_adapter=adapter.to_dict(),
        robot_task_binding=binding.to_dict(),
        environment=definition.environment.to_dict(),
        task_requirements=requirements.to_dict(),
        resolved_roles=resolved_roles,
        provided_capabilities=tuple(sorted(provided_capabilities)),
        contract_hashes=definition.contract_hashes,
        legacy_training_binding_v1_grant_hash=grant_hash,
    )
    return CellResolution("supported", manifest=manifest)


def materialize_resolved_cell_manifest(
    manifest: ResolvedCellManifestV1,
    *,
    output_dir: Path,
    require_existing: bool = False,
) -> Path:
    """Publish a cell identity once, or verify its existing immutable bytes."""
    path = (output_dir / RESOLVED_CELL_MANIFEST_FILENAME).resolve()
    encoded = canonical_json_bytes(manifest.to_dict(), trailing_newline=True)
    try:
        current = path.read_bytes()
    except FileNotFoundError:
        if require_existing:
            raise
        current = None
    if current is None:
        output_dir.mkdir(parents=True, exist_ok=True)
        create_once_bytes(path, encoded)
    elif current != encoded:
        raise ValueError("existing resolved cell manifest differs from the requested cell")
    return path


def load_resolved_cell_manifest(
    path: str | Path,
    *,
    expected_manifest_hash: str | None = None,
    expected_task_id: str | None = None,
    expected_base_task_protocol_hash: str | None = None,
) -> ResolvedCellManifestV1:
    resolved_path = Path(path)
    if not resolved_path.is_absolute():
        raise ValueError("resolved cell manifest path must be absolute")
    payload = load_strict_json_object(
        resolved_path,
        context="resolved cell manifest",
        require_canonical=True,
        trailing_newline=True,
    )
    manifest = ResolvedCellManifestV1.from_mapping(payload)
    if expected_manifest_hash is not None and (
        manifest.manifest_hash != expected_manifest_hash
    ):
        raise ValueError("resolved cell manifest argv hash mismatch")
    if expected_task_id is not None and manifest.task_id != expected_task_id:
        raise ValueError("resolved cell manifest task_id mismatch")
    if expected_base_task_protocol_hash is not None and (
        manifest.base_task_protocol_hash != expected_base_task_protocol_hash
    ):
        raise ValueError("resolved cell manifest base task protocol mismatch")
    return manifest


def add_resolved_cell_manifest_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--resolved-cell-manifest-path", default=None)
    parser.add_argument("--resolved-cell-manifest-hash", default=None)


def load_optional_resolved_cell_manifest_from_args(
    args: argparse.Namespace,
    *,
    expected_task_id: str,
    expected_base_task_protocol_hash: str,
) -> ResolvedCellManifestV1 | None:
    path = getattr(args, "resolved_cell_manifest_path", None)
    manifest_hash = getattr(args, "resolved_cell_manifest_hash", None)
    if path is None and manifest_hash is None:
        return None
    if not isinstance(path, str) or not path or not isinstance(manifest_hash, str):
        raise ValueError("resolved cell manifest path/hash must be provided together")
    require_sha256_digest(manifest_hash, "resolved cell manifest argv hash")
    return load_resolved_cell_manifest(
        path,
        expected_manifest_hash=manifest_hash,
        expected_task_id=expected_task_id,
        expected_base_task_protocol_hash=expected_base_task_protocol_hash,
    )


def resolved_cell_summary_fields(
    manifest: ResolvedCellManifestV1 | None,
) -> dict[str, Any]:
    if manifest is None:
        return {}
    return {
        "resolved_cell_manifest_hash": manifest.manifest_hash,
        "resolved_cell_identity": manifest.summary_identity(),
    }


def _profile_manifest_descriptor(profile: RobotProfile) -> dict[str, Any]:
    runtime_facts = profile.runtime_facts
    return {
        "robot_profile_id": profile.robot_profile_id,
        "robot_family_id": profile.robot_family_id,
        "profile_hash": profile.profile_hash,
        "runtime_facts_hash": profile.runtime_facts_hash,
        "asset_closure": dict(runtime_facts["asset_closure"]),
        "joint_count": len(runtime_facts["joint_names"]),
        "body_count": len(runtime_facts["body_names"]),
        "semantic_roles": {
            key: profile.semantic_roles[key].to_dict()
            for key in sorted(profile.semantic_roles)
        },
    }


__all__ = [
    "CELL_ENVIRONMENT_SCHEMA_VERSION",
    "CELL_PROTOCOL_STATES",
    "CELL_PROTOCOL_STATE_SCHEMA_VERSION",
    "LEGACY_TRAINING_BINDING_V1_GRANT_SCHEMA_VERSION",
    "RESOLVED_CELL_MANIFEST_FILENAME",
    "RESOLVED_CELL_MANIFEST_SCHEMA_VERSION",
    "ROBOT_TASK_BINDING_SCHEMA_VERSION",
    "CellEnvironmentDescriptor",
    "CellResolution",
    "CellVariantCatalog",
    "CellVariantDefinition",
    "ResolvedCellManifestV1",
    "RobotTaskBinding",
    "add_resolved_cell_manifest_arguments",
    "compile_cell_manifest",
    "grant_legacy_training_binding_v1",
    "load_optional_resolved_cell_manifest_from_args",
    "load_resolved_cell_manifest",
    "materialize_resolved_cell_manifest",
    "resolved_cell_summary_fields",
]
