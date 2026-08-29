"""Task-owned projection from one resolved robot cell to Stair logical inputs."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Mapping

from true_rl_agent.composition.cell_manifest import (
    ResolvedCellManifestV1,
    RobotTaskBinding,
)
from true_rl_agent.evaluation.binding import canonical_json_hash, require_sha256_digest
from true_rl_agent.robots.base import (
    BASE_ROLE,
    FORBIDDEN_CONTACT_ROLE,
    JOINT_GROUP_ROLE_PREFIX,
    SUPPORT_EFFECTOR_ROLE,
    RobotProfile,
    RobotRuntimeAdapter,
)
from true_rl_agent.tasks.stair_climbing.config import STAIR_TASK_ID

STAIR_RUNTIME_PROJECTION_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.runtime_projection.vnext.v1"
)
STAIR_REWARD_INPUTS_VNEXT_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.reward_inputs.vnext.v1"
)
STAIR_REWARD_CAPABILITY_VNEXT_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.reward_capability.vnext.v1"
)
STAIR_TRANSITION_VNEXT_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.transition.vnext.v1"
)
STAIR_RUNTIME_BINDING_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.runtime_binding.v1"
)
STAIR_KINEMATIC_BASE_ARTICULATION_ROOT = "articulation_root"
STAIR_KINEMATIC_BASE_BODY_ROLE = "body_role"
_ALL_JOINTS_ROLE = f"{JOINT_GROUP_ROLE_PREFIX}all"
_JOINT_GROUP_SOURCE_SPECS = {
    "joint_pos": ("pos", "rad", ()),
    "joint_pos_error": ("pos_error", "rad", ()),
    "joint_acc": ("acc", "rad/s^2", ()),
    "joint_torque": ("torque", "N*m", ()),
    "soft_joint_pos_limits": ("soft_pos_limits", "rad", (2,)),
}
_JOINT_GROUP_ID_PATTERN = re.compile(r"[a-z][a-z0-9_]*")


@dataclass(frozen=True)
class StairRewardInputFieldVNext:
    name: str
    dtype: str
    trailing_shape: tuple[int | str, ...]
    unit: str
    description: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Stair VNext reward field name must be non-empty")
        if self.dtype not in {"float", "bool"}:
            raise ValueError("Stair VNext reward field dtype is invalid")
        if not isinstance(self.trailing_shape, tuple) or any(
            (isinstance(item, bool))
            or not isinstance(item, int | str)
            or (isinstance(item, int) and item <= 0)
            or (isinstance(item, str) and not item)
            for item in self.trailing_shape
        ):
            raise ValueError("Stair VNext reward field shape is invalid")
        if not isinstance(self.unit, str) or not self.unit:
            raise ValueError("Stair VNext reward field unit must be non-empty")
        if not isinstance(self.description, str) or not self.description:
            raise ValueError("Stair VNext reward field description must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "trailing_shape": list(self.trailing_shape),
            "unit": self.unit,
            "description": self.description,
        }


@dataclass(frozen=True)
class StairRuntimeProjection:
    """Resolved Stair view; robot names remain data rather than task code."""

    cell_variant_id: str
    robot_profile_id: str
    runtime_adapter_id: str
    robot_profile_hash: str
    runtime_adapter_hash: str
    robot_task_binding_hash: str
    base_body_names: tuple[str, ...]
    support_body_names: tuple[str, ...]
    forbidden_contact_body_names: tuple[str, ...]
    joint_names: tuple[str, ...]
    action_joint_names: tuple[str, ...]
    contact_sensor_name: str
    height_sensor_name: str
    contact_history_length: int
    policy_io_schema_hash: str
    sensor_schema_hash: str
    binding_facts_hash: str
    kinematic_base_source: str
    joint_groups: tuple[tuple[str, tuple[str, ...]], ...] = ()
    joint_group_inputs: tuple[tuple[str, str], ...] = ()
    height_scan_ray_count: int = 187
    legacy_reward_adapter_id: str | None = None
    resolved_cell_manifest_hash: str | None = None
    support_transition_inputs: bool = False
    task_horizontal_scale: float = 1.0
    task_vertical_scale: float = 1.0

    def __post_init__(self) -> None:
        for field_name in (
            "cell_variant_id",
            "robot_profile_id",
            "runtime_adapter_id",
            "contact_sensor_name",
            "height_sensor_name",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"Stair runtime projection {field_name} is invalid")
        for field_name in (
            "robot_profile_hash",
            "runtime_adapter_hash",
            "robot_task_binding_hash",
            "policy_io_schema_hash",
            "sensor_schema_hash",
            "binding_facts_hash",
        ):
            require_sha256_digest(
                getattr(self, field_name),
                f"Stair runtime projection {field_name}",
            )
        if self.resolved_cell_manifest_hash is not None:
            require_sha256_digest(
                self.resolved_cell_manifest_hash,
                "Stair runtime projection manifest hash",
            )
        for field_name in (
            "base_body_names",
            "support_body_names",
            "forbidden_contact_body_names",
            "joint_names",
            "action_joint_names",
        ):
            values = getattr(self, field_name)
            if (
                not isinstance(values, tuple)
                or not values
                or len(values) != len(set(values))
                or any(not isinstance(item, str) or not item for item in values)
            ):
                raise ValueError(f"Stair runtime projection {field_name} is invalid")
        if len(self.base_body_names) != 1:
            raise ValueError(
                "Stair runtime projection requires exactly one logical base"
            )
        if self.kinematic_base_source not in {
            STAIR_KINEMATIC_BASE_ARTICULATION_ROOT,
            STAIR_KINEMATIC_BASE_BODY_ROLE,
        }:
            raise ValueError("Stair kinematic base source is invalid")
        if any(name not in self.joint_names for name in self.action_joint_names):
            raise ValueError(
                "Stair action joints must belong to the resolved joint schema"
            )
        if not isinstance(self.joint_groups, tuple):
            raise ValueError("Stair joint groups must be an immutable tuple")
        group_ids: list[str] = []
        for item in self.joint_groups:
            if (
                not isinstance(item, tuple)
                or len(item) != 2
                or not isinstance(item[0], str)
                or not item[0].startswith(JOINT_GROUP_ROLE_PREFIX)
                or item[0] == _ALL_JOINTS_ROLE
                or not isinstance(item[1], tuple)
                or not item[1]
                or len(item[1]) != len(set(item[1]))
                or any(name not in self.joint_names for name in item[1])
            ):
                raise ValueError("Stair joint group projection is invalid")
            group_ids.append(item[0])
        if group_ids != sorted(group_ids) or len(group_ids) != len(set(group_ids)):
            raise ValueError("Stair joint group identities are invalid")
        if not isinstance(self.joint_group_inputs, tuple):
            raise ValueError("Stair joint group inputs must be an immutable tuple")
        seen_group_inputs: set[tuple[str, str]] = set()
        for item in self.joint_group_inputs:
            if (
                not isinstance(item, tuple)
                or len(item) != 2
                or item[0] not in group_ids
                or item[1] not in _JOINT_GROUP_SOURCE_SPECS
                or item in seen_group_inputs
            ):
                raise ValueError("Stair joint group input binding is invalid")
            group_name = item[0][len(JOINT_GROUP_ROLE_PREFIX) :]
            if _JOINT_GROUP_ID_PATTERN.fullmatch(group_name) is None:
                raise ValueError("Stair joint group input identity is invalid")
            seen_group_inputs.add(item)
        if (
            isinstance(self.contact_history_length, bool)
            or not isinstance(self.contact_history_length, int)
            or self.contact_history_length <= 0
        ):
            raise ValueError("Stair contact history length must be positive")
        if (
            isinstance(self.height_scan_ray_count, bool)
            or not isinstance(self.height_scan_ray_count, int)
            or self.height_scan_ray_count <= 0
        ):
            raise ValueError("Stair height scan ray count must be positive")
        if self.legacy_reward_adapter_id is not None and (
            not isinstance(self.legacy_reward_adapter_id, str)
            or not self.legacy_reward_adapter_id
        ):
            raise ValueError("Stair legacy reward adapter ID is invalid")
        if not isinstance(self.support_transition_inputs, bool):
            raise TypeError("Stair support transition input flag must be boolean")
        for name in ("task_horizontal_scale", "task_vertical_scale"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(f"Stair runtime projection {name} is invalid")

    @property
    def support_count(self) -> int:
        return len(self.support_body_names)

    @property
    def joint_count(self) -> int:
        return len(self.joint_names)

    @property
    def forbidden_contact_count(self) -> int:
        return len(self.forbidden_contact_body_names)

    @property
    def action_count(self) -> int:
        return len(self.action_joint_names)

    @property
    def joint_group_indices(self) -> dict[str, tuple[int, ...]]:
        index_by_name = {name: index for index, name in enumerate(self.joint_names)}
        return {
            role_id: tuple(index_by_name[name] for name in names)
            for role_id, names in self.joint_groups
        }

    @property
    def joint_group_input_specs(
        self,
    ) -> tuple[tuple[str, str, str, str, tuple[int, ...]], ...]:
        result = []
        for role_id, source_name in self.joint_group_inputs:
            group_name = role_id[len(JOINT_GROUP_ROLE_PREFIX) :]
            suffix, unit, trailing_shape = _JOINT_GROUP_SOURCE_SPECS[source_name]
            result.append(
                (
                    f"{group_name}_group_{suffix}",
                    role_id,
                    source_name,
                    unit,
                    trailing_shape,
                )
            )
        return tuple(result)

    @property
    def reward_input_fields_vnext(self) -> tuple[StairRewardInputFieldVNext, ...]:
        support_count = self.support_count
        joint_count = self.joint_count
        action_count = self.action_count
        forbidden_count = self.forbidden_contact_count
        history = self.contact_history_length
        common_fields = (
            _field("base_height", "float", (), "m"),
            _field("roll", "float", (), "rad"),
            _field("pitch", "float", (), "rad"),
            _field("base_pos_xy", "float", (2,), "m"),
            _field("base_lin_vel", "float", (3,), "m/s"),
            _field("base_ang_vel", "float", (3,), "rad/s"),
            _field("projected_gravity", "float", (3,), "unit_vector"),
            _field("commanded_base_vel", "float", (3,), "mixed_mps_radps"),
            _field("joint_pos", "float", (joint_count,), "rad"),
            _field("joint_vel", "float", (joint_count,), "rad/s"),
            _field("joint_pos_error", "float", (joint_count,), "rad"),
            _field("last_action", "float", (action_count,), "policy_action"),
            _field("prev_action", "float", (action_count,), "policy_action"),
            _field("action_delta", "float", (action_count,), "policy_action"),
            _field("progress_fraction", "float", (), "fraction"),
            _field("joint_acc", "float", (joint_count,), "rad/s^2"),
            _field("joint_torque", "float", (joint_count,), "N*m"),
            _field("soft_joint_pos_limits", "float", (joint_count, 2), "rad"),
        )
        groups = dict(self.joint_groups)
        group_fields = tuple(
            _field(
                field_name,
                "float",
                (len(groups[role_id]), *trailing_shape),
                unit,
            )
            for field_name, role_id, _, unit, trailing_shape in (
                self.joint_group_input_specs
            )
        )
        support_transition_fields = (
            (
                _field("support_last_air_time", "float", (support_count,), "s"),
                _field("support_first_contact", "bool", (support_count,), "boolean"),
            )
            if self.support_transition_inputs
            else ()
        )
        return (
            *common_fields,
            *group_fields,
            _field("base_lin_vel_yaw", "float", (3,), "m/s"),
            _field("base_ang_vel_world", "float", (3,), "rad/s"),
            _field("base_quat_world", "float", (4,), "unit_quaternion_wxyz"),
            _field("support_air_time", "float", (support_count,), "s"),
            _field("support_contact_time", "float", (support_count,), "s"),
            *support_transition_fields,
            _field(
                "support_contact_force_history",
                "float",
                (history, support_count, 3),
                "N",
            ),
            _field("support_contact_force", "float", (support_count, 3), "N"),
            _field("support_contact", "bool", (support_count,), "boolean"),
            _field("support_lin_vel", "float", (support_count, 3), "m/s"),
            _field(
                "support_pos_relative_to_terrain_origin",
                "float",
                (support_count, 3),
                "m",
            ),
            _field("base_pos_relative_to_terrain_origin", "float", (3,), "m"),
            _field(
                "forbidden_contact_force",
                "float",
                (forbidden_count, 3),
                "N",
            ),
            _field("height_scan", "float", (self.height_scan_ray_count,), "m"),
            _field("forward_progress_m", "float", (), "m"),
            _field("remaining_goal_distance_m", "float", (), "m"),
            _field("goal_progress_fraction", "float", (), "fraction"),
            _field("terminated_non_timeout", "bool", (), "boolean"),
            _field("time_out", "bool", (), "boolean"),
            _field("forbidden_contact", "bool", (), "boolean"),
        )

    @property
    def reward_input_schema_hash_vnext(self) -> str:
        return canonical_json_hash(self.reward_input_schema_vnext())

    @property
    def reward_capability_hash_vnext(self) -> str:
        return canonical_json_hash(
            {
                "schema_version": STAIR_REWARD_CAPABILITY_VNEXT_SCHEMA_VERSION,
                "capability_id": "complete_reward_program_vnext",
                "input_schema_hash": self.reward_input_schema_hash_vnext,
                "authority": "task_owned_stair_reward_only",
            }
        )

    @property
    def projection_hash(self) -> str:
        return canonical_json_hash(self._hash_payload())

    def reward_input_schema_vnext(self) -> dict[str, Any]:
        return {
            "schema_version": STAIR_REWARD_INPUTS_VNEXT_SCHEMA_VERSION,
            "fields": [item.to_dict() for item in self.reward_input_fields_vnext],
            "source_transforms": {
                "height_scan": {
                    "clip_max": 1.0,
                    "clip_min": -1.0,
                    "corruption": "disabled",
                    "offset_m": 0.5,
                    "ray_miss_policy": "saturate_infinity_to_clip_preserve_nan",
                    "source_formula": (
                        "sensor_pos_w_z_minus_ray_hit_w_z_minus_offset"
                    ),
                }
            },
        }

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            **self._hash_payload(),
            "resolved_cell_manifest_hash": self.resolved_cell_manifest_hash,
            "projection_hash": self.projection_hash,
            "reward_input_schema_hash_vnext": self.reward_input_schema_hash_vnext,
            "reward_capability_hash_vnext": self.reward_capability_hash_vnext,
        }

    def _hash_payload(self) -> dict[str, Any]:
        payload = {
            "schema_version": STAIR_RUNTIME_PROJECTION_SCHEMA_VERSION,
            "cell_variant_id": self.cell_variant_id,
            "robot_profile_id": self.robot_profile_id,
            "runtime_adapter_id": self.runtime_adapter_id,
            "robot_profile_hash": self.robot_profile_hash,
            "runtime_adapter_hash": self.runtime_adapter_hash,
            "robot_task_binding_hash": self.robot_task_binding_hash,
            "resolved_roles": {
                "base": list(self.base_body_names),
                "support_effector": list(self.support_body_names),
                "forbidden_contact": list(self.forbidden_contact_body_names),
            },
            "joint_names": list(self.joint_names),
            "joint_groups": {
                role_id: list(names) for role_id, names in self.joint_groups
            },
            "action_joint_names": list(self.action_joint_names),
            "contact_sensor_name": self.contact_sensor_name,
            "height_sensor_name": self.height_sensor_name,
            "contact_history_length": self.contact_history_length,
            "height_scan_ray_count": self.height_scan_ray_count,
            "policy_io_schema_hash": self.policy_io_schema_hash,
            "sensor_schema_hash": self.sensor_schema_hash,
            "binding_facts_hash": self.binding_facts_hash,
            "kinematic_base_source": self.kinematic_base_source,
            "legacy_reward_adapter_id": self.legacy_reward_adapter_id,
        }
        if self.joint_group_inputs:
            payload["joint_group_inputs"] = [
                {"role_id": role_id, "source": source}
                for role_id, source in self.joint_group_inputs
            ]
        if self.support_transition_inputs:
            payload["support_transition_inputs"] = True
        if not math.isclose(
            float(self.task_horizontal_scale), 1.0, rel_tol=0.0, abs_tol=0.0
        ) or not math.isclose(
            float(self.task_vertical_scale), 1.0, rel_tol=0.0, abs_tol=0.0
        ):
            payload["task_scales"] = {
                "horizontal": float(self.task_horizontal_scale),
                "vertical": float(self.task_vertical_scale),
            }
        return payload


@dataclass(frozen=True)
class ResolvedStairRuntimeProjection:
    projection: StairRuntimeProjection
    joint_ids: tuple[int, ...]
    base_body_ids: tuple[int, ...]
    support_body_ids: tuple[int, ...]
    contact_support_body_ids: tuple[int, ...]
    contact_forbidden_body_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        expected_lengths = (
            (self.joint_ids, self.projection.joint_count),
            (self.base_body_ids, len(self.projection.base_body_names)),
            (self.support_body_ids, self.projection.support_count),
            (self.contact_support_body_ids, self.projection.support_count),
            (
                self.contact_forbidden_body_ids,
                self.projection.forbidden_contact_count,
            ),
        )
        for values, expected in expected_lengths:
            if (
                not isinstance(values, tuple)
                or len(values) != expected
                or len(values) != len(set(values))
                or any(
                    isinstance(item, bool) or not isinstance(item, int) or item < 0
                    for item in values
                )
            ):
                raise ValueError("resolved Stair runtime projection IDs are invalid")

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            **self.projection.to_summary_dict(),
            "resolved_runtime": {
                "joint_ids": list(self.joint_ids),
                "base_body_ids": list(self.base_body_ids),
                "support_body_ids": list(self.support_body_ids),
                "contact_support_body_ids": list(self.contact_support_body_ids),
                "contact_forbidden_body_ids": list(self.contact_forbidden_body_ids),
            },
        }


def stair_runtime_projection_from_manifest(
    manifest: ResolvedCellManifestV1,
) -> StairRuntimeProjection:
    if not isinstance(manifest, ResolvedCellManifestV1):
        raise TypeError("Stair runtime projection requires a resolved cell manifest")
    if manifest.task_id != STAIR_TASK_ID:
        raise ValueError("Stair runtime projection manifest task mismatch")
    profile = manifest.robot_profile
    adapter = manifest.runtime_adapter
    binding = manifest.robot_task_binding
    projection = _build_projection(
        cell_variant_id=manifest.cell_variant_id,
        robot_profile_id=profile.get("robot_profile_id"),
        runtime_adapter_id=adapter.get("runtime_adapter_id"),
        robot_profile_hash=profile.get("profile_hash"),
        runtime_adapter_hash=adapter.get("adapter_hash"),
        robot_task_binding_hash=binding.get("binding_hash"),
        profile_roles=profile.get("semantic_roles"),
        binding_roles=manifest.resolved_roles,
        task_projection=binding.get("task_projection"),
        resolved_cell_manifest_hash=manifest.manifest_hash,
    )
    expected_schema = manifest.contract_hashes.get("reward_input_vnext_schema_hash")
    expected_capability = manifest.contract_hashes.get("reward_capability_vnext_hash")
    if expected_schema != projection.reward_input_schema_hash_vnext:
        raise ValueError("Stair manifest VNext reward input schema mismatch")
    if expected_capability != projection.reward_capability_hash_vnext:
        raise ValueError("Stair manifest VNext reward capability mismatch")
    return projection


def stair_runtime_projection_from_execution(
    *,
    profile: RobotProfile,
    adapter: RobotRuntimeAdapter,
    binding: RobotTaskBinding,
) -> StairRuntimeProjection:
    if not isinstance(profile, RobotProfile):
        raise TypeError("Stair projection profile is invalid")
    if not isinstance(adapter, RobotRuntimeAdapter):
        raise TypeError("Stair projection adapter is invalid")
    if not isinstance(binding, RobotTaskBinding):
        raise TypeError("Stair projection binding is invalid")
    if (
        profile.robot_profile_id != adapter.robot_profile_id
        or profile.robot_profile_id != binding.robot_profile_id
        or adapter.runtime_adapter_id != binding.runtime_adapter_id
        or binding.task_id != STAIR_TASK_ID
    ):
        raise ValueError("Stair projection execution identity mismatch")
    profile_roles = {
        role_id: role.to_dict() for role_id, role in profile.semantic_roles.items()
    }
    return _build_projection(
        cell_variant_id=binding.cell_variant_id,
        robot_profile_id=profile.robot_profile_id,
        runtime_adapter_id=adapter.runtime_adapter_id,
        robot_profile_hash=profile.profile_hash,
        runtime_adapter_hash=adapter.adapter_hash,
        robot_task_binding_hash=binding.binding_hash,
        profile_roles=profile_roles,
        binding_roles=binding.role_bindings,
        task_projection=binding.task_projection,
        resolved_cell_manifest_hash=None,
    )


def resolve_stair_runtime_projection(
    projection: StairRuntimeProjection,
    *,
    robot: Any,
    contact_sensor: Any,
) -> ResolvedStairRuntimeProjection:
    if not isinstance(projection, StairRuntimeProjection):
        raise TypeError("Stair runtime projection is invalid")
    return ResolvedStairRuntimeProjection(
        projection=projection,
        joint_ids=_resolve_ids(
            robot,
            method_name="find_joints",
            names=projection.joint_names,
            context="policy joints",
        ),
        base_body_ids=_resolve_ids(
            robot,
            method_name="find_bodies",
            names=projection.base_body_names,
            context="logical base",
        ),
        support_body_ids=_resolve_ids(
            robot,
            method_name="find_bodies",
            names=projection.support_body_names,
            context="support effectors",
        ),
        contact_support_body_ids=_resolve_ids(
            contact_sensor,
            method_name="find_bodies",
            names=projection.support_body_names,
            context="contact support effectors",
        ),
        contact_forbidden_body_ids=_resolve_ids(
            contact_sensor,
            method_name="find_bodies",
            names=projection.forbidden_contact_body_names,
            context="forbidden contacts",
        ),
    )


def _build_projection(
    *,
    cell_variant_id: Any,
    robot_profile_id: Any,
    runtime_adapter_id: Any,
    robot_profile_hash: Any,
    runtime_adapter_hash: Any,
    robot_task_binding_hash: Any,
    profile_roles: Any,
    binding_roles: Mapping[str, tuple[str, ...]] | Any,
    task_projection: Any,
    resolved_cell_manifest_hash: str | None,
) -> StairRuntimeProjection:
    if not isinstance(profile_roles, Mapping):
        raise ValueError("Stair profile semantic roles are missing")
    if not isinstance(binding_roles, Mapping):
        raise ValueError("Stair binding roles are missing")
    resolved_roles: dict[str, tuple[str, ...]] = {}
    for role_id in (BASE_ROLE, SUPPORT_EFFECTOR_ROLE, FORBIDDEN_CONTACT_ROLE):
        raw = binding_roles.get(role_id)
        if isinstance(raw, list):
            raw = tuple(raw)
        if not isinstance(raw, tuple) or not raw:
            raise ValueError(f"Stair binding role is missing: {role_id}")
        profile_role = profile_roles.get(role_id)
        if not isinstance(profile_role, Mapping):
            raise ValueError(f"Stair profile role is missing: {role_id}")
        if profile_role.get("target_kind") != "body":
            raise ValueError(f"Stair profile role kind is invalid: {role_id}")
        targets = profile_role.get("targets")
        if not isinstance(targets, list) or any(item not in targets for item in raw):
            raise ValueError(f"Stair binding role differs from profile: {role_id}")
        resolved_roles[role_id] = tuple(raw)
    profile_support = profile_roles[SUPPORT_EFFECTOR_ROLE].get("targets")
    if tuple(profile_support) != resolved_roles[SUPPORT_EFFECTOR_ROLE]:
        raise ValueError("Stair binding must select every profile support effector")
    all_joints = profile_roles.get(_ALL_JOINTS_ROLE)
    if not isinstance(all_joints, Mapping) or all_joints.get("target_kind") != "joint":
        raise ValueError("Stair profile all-joints role is missing")
    joint_names_raw = all_joints.get("targets")
    if (
        not isinstance(joint_names_raw, list)
        or not joint_names_raw
        or len(joint_names_raw) != len(set(joint_names_raw))
        or any(not isinstance(item, str) or not item for item in joint_names_raw)
    ):
        raise ValueError("Stair profile all-joints targets are invalid")
    joint_groups: list[tuple[str, tuple[str, ...]]] = []
    for role_id in sorted(profile_roles):
        if not role_id.startswith(JOINT_GROUP_ROLE_PREFIX) or role_id == _ALL_JOINTS_ROLE:
            continue
        role = profile_roles[role_id]
        if not isinstance(role, Mapping) or role.get("target_kind") != "joint":
            raise ValueError(f"Stair profile joint group is invalid: {role_id}")
        targets = role.get("targets")
        if (
            not isinstance(targets, list)
            or not targets
            or len(targets) != len(set(targets))
            or any(name not in joint_names_raw for name in targets)
        ):
            raise ValueError(f"Stair profile joint group targets are invalid: {role_id}")
        joint_groups.append((role_id, tuple(targets)))
    if not isinstance(task_projection, Mapping):
        raise ValueError("Stair task projection is missing")
    runtime_binding = task_projection.get("stair_runtime_projection")
    required_runtime_fields = {
        "kinematic_base",
        "legacy_reward_adapter_id",
        "schema_version",
    }
    allowed_runtime_fields = required_runtime_fields | {
        "reward_joint_group_inputs",
        "support_transition_inputs",
        "task_scales",
    }
    if (
        not isinstance(runtime_binding, Mapping)
        or not required_runtime_fields.issubset(runtime_binding)
        or not set(runtime_binding).issubset(allowed_runtime_fields)
    ):
        raise ValueError("Stair runtime binding is missing or malformed")
    if runtime_binding.get("schema_version") != STAIR_RUNTIME_BINDING_SCHEMA_VERSION:
        raise ValueError("Stair runtime binding schema mismatch")
    kinematic_base = runtime_binding.get("kinematic_base")
    if not isinstance(kinematic_base, Mapping) or set(kinematic_base) != {"source"}:
        raise ValueError("Stair kinematic base binding is malformed")
    kinematic_base_source = kinematic_base.get("source")
    legacy_reward_adapter_id = runtime_binding.get("legacy_reward_adapter_id")
    if legacy_reward_adapter_id is not None and (
        not isinstance(legacy_reward_adapter_id, str)
        or not legacy_reward_adapter_id
    ):
        raise ValueError("Stair legacy reward adapter binding is invalid")
    raw_group_inputs = runtime_binding.get("reward_joint_group_inputs", [])
    if not isinstance(raw_group_inputs, list):
        raise ValueError("Stair reward joint group inputs are malformed")
    joint_group_inputs: list[tuple[str, str]] = []
    for item in raw_group_inputs:
        if not isinstance(item, Mapping) or set(item) != {"role_id", "source"}:
            raise ValueError("Stair reward joint group input is malformed")
        role_id = item.get("role_id")
        source = item.get("source")
        if not isinstance(role_id, str) or not isinstance(source, str):
            raise ValueError("Stair reward joint group input values are invalid")
        joint_group_inputs.append((role_id, source))
    support_transition_inputs = runtime_binding.get(
        "support_transition_inputs", False
    )
    if not isinstance(support_transition_inputs, bool):
        raise ValueError("Stair support transition input binding is invalid")
    task_scales = runtime_binding.get(
        "task_scales", {"horizontal": 1.0, "vertical": 1.0}
    )
    if not isinstance(task_scales, Mapping) or set(task_scales) != {
        "horizontal",
        "vertical",
    }:
        raise ValueError("Stair task scale binding is invalid")
    facts = task_projection.get("phase0_binding_facts")
    if not isinstance(facts, Mapping):
        raise ValueError("Stair binding facts are missing")
    actions = facts.get("actions")
    if not isinstance(actions, list) or len(actions) != 1:
        raise ValueError("Stair action projection must contain one action")
    action = actions[0]
    if (
        not isinstance(action, Mapping)
        or action.get("asset_name") != "robot"
        or not str(action.get("class", "")).endswith("JointPositionActionCfg")
    ):
        raise ValueError("Stair action projection is unsupported")
    action_joint_selector = action.get("joint_names")
    if action_joint_selector == [".*"]:
        action_joint_names = tuple(joint_names_raw)
    elif (
        isinstance(action_joint_selector, list)
        and action_joint_selector
        and len(action_joint_selector) == len(set(action_joint_selector))
        and all(
            isinstance(name, str) and name in joint_names_raw
            for name in action_joint_selector
        )
    ):
        action_joint_names = tuple(action_joint_selector)
    else:
        raise ValueError("Stair action joint projection is invalid")
    sensors = facts.get("sensors")
    if not isinstance(sensors, Mapping):
        raise ValueError("Stair sensor projection is missing")
    contact_names = tuple(
        name
        for name, value in sensors.items()
        if isinstance(name, str)
        and isinstance(value, Mapping)
        and str(value.get("class", "")).endswith("ContactSensorCfg")
    )
    height_names = tuple(
        name
        for name, value in sensors.items()
        if isinstance(name, str)
        and isinstance(value, Mapping)
        and str(value.get("class", "")).endswith("RayCasterCfg")
    )
    if len(contact_names) != 1 or len(height_names) != 1:
        raise ValueError("Stair binding must resolve one contact and one height sensor")
    contact_sensor = sensors[contact_names[0]]
    history_length = contact_sensor.get("history_length")
    height_sensor = sensors[height_names[0]]
    pattern = height_sensor.get("pattern")
    if not isinstance(pattern, Mapping):
        raise ValueError("Stair height sensor pattern is missing")
    pattern_size = pattern.get("size_m")
    resolution = pattern.get("resolution_m")
    if (
        not isinstance(pattern_size, list)
        or len(pattern_size) != 2
        or isinstance(resolution, bool)
        or not isinstance(resolution, int | float)
        or float(resolution) <= 0.0
    ):
        raise ValueError("Stair height sensor grid shape is invalid")
    height_scan_ray_count = 1
    for extent in pattern_size:
        count_float = float(extent) / float(resolution)
        count = int(round(count_float))
        if not math.isclose(count_float, count, rel_tol=0.0, abs_tol=1.0e-9):
            raise ValueError("Stair height sensor grid is not integral")
        height_scan_ray_count *= count + 1
    return StairRuntimeProjection(
        cell_variant_id=cell_variant_id,
        robot_profile_id=robot_profile_id,
        runtime_adapter_id=runtime_adapter_id,
        robot_profile_hash=robot_profile_hash,
        runtime_adapter_hash=runtime_adapter_hash,
        robot_task_binding_hash=robot_task_binding_hash,
        base_body_names=resolved_roles[BASE_ROLE],
        support_body_names=resolved_roles[SUPPORT_EFFECTOR_ROLE],
        forbidden_contact_body_names=resolved_roles[FORBIDDEN_CONTACT_ROLE],
        joint_names=tuple(joint_names_raw),
        action_joint_names=action_joint_names,
        contact_sensor_name=contact_names[0],
        height_sensor_name=height_names[0],
        contact_history_length=history_length,
        policy_io_schema_hash=facts.get("policy_io_schema_hash"),
        sensor_schema_hash=facts.get("sensor_schema_hash"),
        binding_facts_hash=facts.get("binding_facts_hash"),
        kinematic_base_source=kinematic_base_source,
        joint_groups=tuple(joint_groups),
        joint_group_inputs=tuple(joint_group_inputs),
        height_scan_ray_count=height_scan_ray_count,
        legacy_reward_adapter_id=legacy_reward_adapter_id,
        resolved_cell_manifest_hash=resolved_cell_manifest_hash,
        support_transition_inputs=support_transition_inputs,
        task_horizontal_scale=task_scales["horizontal"],
        task_vertical_scale=task_scales["vertical"],
    )


def _field(
    name: str,
    dtype: str,
    trailing_shape: tuple[int | str, ...],
    unit: str,
) -> StairRewardInputFieldVNext:
    return StairRewardInputFieldVNext(
        name,
        dtype,
        trailing_shape,
        unit,
        f"Task-owned morphology-neutral runtime value for {name}.",
    )


def _resolve_ids(
    asset: Any,
    *,
    method_name: str,
    names: tuple[str, ...],
    context: str,
) -> tuple[int, ...]:
    method = getattr(asset, method_name, None)
    if not callable(method):
        raise RuntimeError(f"Stair {context} asset resolver is unavailable")
    ids, resolved_names = method(list(names), preserve_order=True)
    if tuple(resolved_names) != names or len(ids) != len(names):
        raise RuntimeError(f"Stair {context} schema drifted")
    return tuple(int(item) for item in ids)


__all__ = [
    "STAIR_KINEMATIC_BASE_ARTICULATION_ROOT",
    "STAIR_KINEMATIC_BASE_BODY_ROLE",
    "STAIR_REWARD_CAPABILITY_VNEXT_SCHEMA_VERSION",
    "STAIR_REWARD_INPUTS_VNEXT_SCHEMA_VERSION",
    "STAIR_RUNTIME_PROJECTION_SCHEMA_VERSION",
    "STAIR_RUNTIME_BINDING_SCHEMA_VERSION",
    "STAIR_TRANSITION_VNEXT_SCHEMA_VERSION",
    "ResolvedStairRuntimeProjection",
    "StairRewardInputFieldVNext",
    "StairRuntimeProjection",
    "resolve_stair_runtime_projection",
    "stair_runtime_projection_from_execution",
    "stair_runtime_projection_from_manifest",
]
