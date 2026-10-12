"""Task-owned deterministic complete-reward smoke fixture for Stair."""

from __future__ import annotations

import math
from dataclasses import dataclass, field as dataclass_field
from typing import Any

from true_rl_agent.candidates.capability_readiness import (
    COMPLETE_REWARD_PROGRAM_CAPABILITY_ID,
)
from true_rl_agent.composition.cell_manifest import ResolvedCellManifestV1
from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.tasks.component_smoke import ComponentSmokeFixture
from true_rl_agent.tasks.stair_climbing.config import (
    STAIR_TASK_ID,
    StairClimbingTaskConfig,
)
from true_rl_agent.tasks.stair_climbing.protocol_v2 import (
    StairCompleteRewardRuntimeContract,
)
from true_rl_agent.tasks.stair_climbing.reward_inputs import StairRewardInputsV2
from true_rl_agent.tasks.stair_climbing.runtime_projection import (
    stair_runtime_projection_from_manifest,
)
from true_rl_agent.tasks.stair_climbing.vnext_reward_contract import (
    build_stair_vnext_reward_runtime_contract,
)

STAIR_SMOKE_FIXTURE_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.complete_reward_smoke_fixture.v3"
)
STAIR_SMOKE_NUM_ENVS = 5


@dataclass(frozen=True)
class StairSmokeRewardInputs:
    """Pickle-safe advisory view; trusted admission uses StairRewardInputsV2."""

    values: dict[str, Any]

    def __getattr__(self, name: str) -> Any:
        try:
            values = object.__getattribute__(self, "values")
        except AttributeError:
            raise AttributeError(name) from None
        try:
            return values[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


@dataclass(frozen=True)
class _UnavailableTorchTensor:
    """Shape-only local placeholder when the Isaac/Torch runtime is absent."""

    shape: tuple[int, ...]


@dataclass(frozen=True)
class StairClimbingComponentSmokeAdapter:
    """Build the same opaque tensor ABI used by the pre-reset dispatcher."""

    config: StairClimbingTaskConfig
    resolved_cell_manifest: ResolvedCellManifestV1 | None = dataclass_field(
        default=None,
        repr=False,
        compare=False,
    )
    task_id: str = STAIR_TASK_ID

    def __post_init__(self) -> None:
        if not isinstance(self.config, StairClimbingTaskConfig):
            raise TypeError("stair smoke adapter requires StairClimbingTaskConfig")
        manifest = self.resolved_cell_manifest
        if self.config.is_unified_protocol:
            if not isinstance(manifest, ResolvedCellManifestV1):
                raise ValueError("unified Stair smoke requires the resolved manifest")
            if (
                manifest.cell_variant_id != self.config.cell_variant_id
                or manifest.resolved_cell_task_protocol_hash
                != self.config.task_protocol_hash
            ):
                raise ValueError("Stair smoke manifest/config identity mismatch")

    @property
    def candidate_capability_ids(self) -> frozenset[str]:
        return frozenset({COMPLETE_REWARD_PROGRAM_CAPABILITY_ID})

    @property
    def contract(self) -> StairCompleteRewardRuntimeContract:
        if self.config.is_unified_protocol:
            assert self.resolved_cell_manifest is not None
            return build_stair_vnext_reward_runtime_contract(
                projection=stair_runtime_projection_from_manifest(
                    self.resolved_cell_manifest
                ),
                authority=self.config.vnext_reward_authority,
            )
        assert self.config.protocol_v2 is not None
        return self.config.protocol_v2.complete_reward_runtime

    @property
    def nominal_morphology(self) -> dict[str, Any]:
        if self.config.is_unified_protocol:
            scale = self.config.ordered_evaluation_suite.to_dict().get("scale_context")
            if scale is not None:
                return {**scale, "usage": "nominal body scale, not terrain-relative base clearance or a success threshold"}
        # Historical protocols predate the calibrated suite; use their existing binding.
        from true_rl_agent.tasks.stair_climbing.calibration import stair_scale_context
        profile, _ = self._profile_and_projection()
        return stair_scale_context(profile)

    def _profile_and_projection(self):
        from true_rl_agent.robots.defaults import build_default_robot_registry
        from true_rl_agent.tasks.stair_climbing.legacy_runtime_projection import build_legacy_stair_runtime_projection
        projection = (stair_runtime_projection_from_manifest(self.resolved_cell_manifest)
                      if self.resolved_cell_manifest is not None else build_legacy_stair_runtime_projection())
        profile = build_default_robot_registry().profile(projection.robot_profile_id)
        if profile.profile_hash != projection.robot_profile_hash:
            raise ValueError("Stair nominal morphology differs from the bound profile")
        return profile, projection

    @property
    def fixture_hash(self) -> str:
        return canonical_json_hash(
            {
                "schema_version": STAIR_SMOKE_FIXTURE_SCHEMA_VERSION,
                "input_schema_hash": self.contract.input_schema_hash,
                "num_envs": STAIR_SMOKE_NUM_ENVS,
                "value_recipe": "nominal_support_pose_landing_air_tilt_inverted_v4",
                "nominal_morphology": self.nominal_morphology,
            }
        )

    def _tensor_values(self) -> dict[str, Any]:
        import torch

        values = {}
        for field in self.contract.input_fields:
            shape = (STAIR_SMOKE_NUM_ENVS, *field.trailing_shape)
            row_shape = (STAIR_SMOKE_NUM_ENVS,) + (1,) * len(
                field.trailing_shape
            )
            row_ids = torch.arange(
                STAIR_SMOKE_NUM_ENVS,
                dtype=torch.float32,
                device="cpu",
            ).reshape(row_shape)
            if field.dtype == "bool":
                values[field.name] = (
                    (row_ids.to(dtype=torch.int64) % 3) == 1
                ).expand(shape).clone()
                continue
            trailing_size = math.prod(field.trailing_shape)
            trailing_offsets = (
                torch.arange(
                    trailing_size,
                    dtype=torch.float32,
                    device="cpu",
                ).reshape(field.trailing_shape)
                * 0.001
                if field.trailing_shape
                else 0.0
            )
            values[field.name] = (
                (row_ids + 1.0) * 0.1 + trailing_offsets
            ).expand(shape).clone()
        # Five independent transitions. Other fields retain distinct finite row
        # markers for shape/isolation checks; this is a bounded probe, not a rollout.
        length = self.nominal_morphology["support_length_m"]
        values["base_height"][:] = length * torch.tensor([1.0, 1.025, 1.15, 0.9, 0.4])
        values["roll"][:] = torch.tensor([0.0, 0.0, 0.0, 0.5, math.pi])
        values["pitch"][:] = 0.0
        profile, projection = self._profile_and_projection()
        morphology = profile.runtime_facts["morphology"]
        positions = morphology["default_body_pos_relative_to_root_m"]
        relative = torch.tensor([positions[name] for name in projection.support_body_names])
        relative -= torch.tensor(positions[projection.base_body_names[0]])
        roll = values["roll"]
        support_offset = relative[None, :, :].expand(STAIR_SMOKE_NUM_ENVS, -1, -1).clone()
        support_offset[:, :, 1] = roll.cos()[:, None] * relative[:, 1] - roll.sin()[:, None] * relative[:, 2]
        support_offset[:, :, 2] = roll.sin()[:, None] * relative[:, 1] + roll.cos()[:, None] * relative[:, 2]
        base_pos = values["base_pos_relative_to_terrain_origin"]
        base_pos[:, 2] = values["base_height"]
        position_prefix = "support" if "support_pos_relative_to_terrain_origin" in values else "foot"
        values[f"{position_prefix}_pos_relative_to_terrain_origin"][:] = base_pos[:, None, :] + support_offset
        if "base_quat_world" in values:
            zero = torch.zeros_like(roll)
            values["base_quat_world"][:] = torch.stack(((roll / 2).cos(), (roll / 2).sin(), zero, zero), dim=-1)
        if "projected_gravity" in values:
            roll = values["roll"]
            values["projected_gravity"][:] = torch.stack((torch.zeros_like(roll), -roll.sin(), -roll.cos()), dim=-1)
        for name in ("base_contact", "forbidden_contact", "terminated_non_timeout"):
            if name in values:
                values[name][:] = torch.tensor([False, False, False, False, True])
        prefix = "support" if "support_contact_force_history" in values else "foot"
        history = values[f"{prefix}_contact_force_history"]
        history.zero_()
        history[0, :, :, 2] = 30.0  # sustained support
        history[1, 0, :, 2] = 30.0  # landing after airborne history
        history[3, :, :, 2] = 30.0  # tilted support
        if f"{prefix}_contact_force" in values:
            values[f"{prefix}_contact_force"][:] = history[:, 0]
        if f"{prefix}_contact" in values:
            values[f"{prefix}_contact"][:] = history.norm(dim=-1).amax(dim=1) > 1.0
        for name, row_values in {
            "support_contact_time": [0.4, 0.005, 0.0, 0.2, 0.0],
            "support_air_time": [0.0, 0.0, 0.3, 0.0, 0.2],
            "support_last_air_time": [0.2, 0.6, 0.2, 0.1, 0.1],
            "support_first_contact": [False, True, False, False, False],
        }.items():
            if name in values:
                values[name][:] = torch.tensor(row_values, dtype=values[name].dtype)[:, None]
        return values

    def build_reward_inputs(self) -> StairRewardInputsV2:
        return StairRewardInputsV2._validated(
            contract=self.contract,
            values=self._tensor_values(),
        )

    def _advisory_values(self) -> dict[str, Any]:
        try:
            return self._tensor_values()
        except ModuleNotFoundError as exc:
            if exc.name != "torch":
                raise
            return {
                field.name: _UnavailableTorchTensor(
                    (STAIR_SMOKE_NUM_ENVS, *field.trailing_shape)
                )
                for field in self.contract.input_fields
            }

    def build_component_smoke_fixture(self) -> ComponentSmokeFixture:
        return ComponentSmokeFixture(
            task_id=self.task_id,
            reward_inputs=StairSmokeRewardInputs(self._advisory_values()),
            expected_reward_output_shape=(STAIR_SMOKE_NUM_ENVS,),
            public_summary={
                "schema_version": STAIR_SMOKE_FIXTURE_SCHEMA_VERSION,
                "fixture_owner": "task_owned_stair_climbing",
                "input_schema_hash": self.contract.input_schema_hash,
                "fixture_hash": self.fixture_hash,
                "num_envs": STAIR_SMOKE_NUM_ENVS,
                "purpose": "complete_reward_admission_preflight",
                "sample_cases": ["nominal_upright_support", "landing", "airborne", "tilted_support", "inverted"],
                "coverage": "five synthetic transitions; no proof of universal zero or causal benefit",
                "nominal_morphology": self.nominal_morphology,
            },
        )


__all__ = [
    "STAIR_SMOKE_FIXTURE_SCHEMA_VERSION",
    "STAIR_SMOKE_NUM_ENVS",
    "StairClimbingComponentSmokeAdapter",
    "StairSmokeRewardInputs",
]
