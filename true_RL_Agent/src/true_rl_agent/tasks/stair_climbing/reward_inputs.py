"""Task-owned opaque input object for complete Stair reward programs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from true_rl_agent.tasks.stair_climbing.protocol_v2 import (
    StairCompleteRewardRuntimeContract,
)
from true_rl_agent.tasks.stair_climbing.reward_source_semantics import (
    stair_reward_source_semantics_policy,
)
from true_rl_agent.tasks.stair_climbing.runtime_projection import (
    StairRuntimeProjection,
)


class StairRewardInputsVNext:
    """Morphology-neutral, builder-owned Stair reward input snapshot."""

    __slots__ = ("__field_names", "__projection", "__values")
    __TOKEN = object()

    def __init__(
        self,
        *,
        projection: StairRuntimeProjection,
        values: tuple[Any, ...],
        token: object | None = None,
    ) -> None:
        if token is not self.__TOKEN:
            raise TypeError(
                "Stair VNext reward inputs must be created by the task-owned builder"
            )
        if not isinstance(projection, StairRuntimeProjection):
            raise TypeError("Stair VNext reward input projection is invalid")
        field_names = tuple(item.name for item in projection.reward_input_fields_vnext)
        if not isinstance(values, tuple) or len(values) != len(field_names):
            raise ValueError("Stair VNext reward input value count mismatch")
        object.__setattr__(self, "_StairRewardInputsVNext__projection", projection)
        object.__setattr__(self, "_StairRewardInputsVNext__field_names", field_names)
        object.__setattr__(self, "_StairRewardInputsVNext__values", values)

    @classmethod
    def _validated(
        cls,
        *,
        projection: StairRuntimeProjection,
        values: Mapping[str, Any],
    ) -> "StairRewardInputsVNext":
        if not isinstance(values, Mapping):
            raise TypeError("Stair VNext reward input values must be an object")
        field_names = tuple(item.name for item in projection.reward_input_fields_vnext)
        if set(values) != set(field_names):
            missing = sorted(set(field_names) - set(values))
            extra = sorted(set(values) - set(field_names))
            raise ValueError(
                "Stair VNext reward input fields mismatch: "
                f"missing={missing!r} extra={extra!r}"
            )
        ordered = tuple(values[name] for name in field_names)
        return cls(projection=projection, values=ordered, token=cls.__TOKEN)

    def __getattr__(self, name: str) -> Any:
        field_names = object.__getattribute__(
            self,
            "_StairRewardInputsVNext__field_names",
        )
        try:
            index = field_names.index(name)
        except ValueError as exc:
            raise AttributeError(name) from exc
        values = object.__getattribute__(
            self,
            "_StairRewardInputsVNext__values",
        )
        return values[index]

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("Stair VNext reward inputs are immutable")

    @property
    def schema_hash(self) -> str:
        projection = object.__getattribute__(
            self,
            "_StairRewardInputsVNext__projection",
        )
        return projection.reward_input_schema_hash_vnext

    @property
    def projection_hash(self) -> str:
        projection = object.__getattribute__(
            self,
            "_StairRewardInputsVNext__projection",
        )
        return projection.projection_hash

    @property
    def field_names(self) -> tuple[str, ...]:
        return object.__getattribute__(
            self,
            "_StairRewardInputsVNext__field_names",
        )


class StairRewardInputsV2:
    """Read-only attribute view over one validated detached tensor snapshot."""

    __slots__ = ("__contract", "__values")
    __TOKEN = object()

    def __init__(
        self,
        *,
        contract: StairCompleteRewardRuntimeContract,
        values: tuple[Any, ...],
        token: object | None = None,
    ) -> None:
        if token is not self.__TOKEN:
            raise TypeError(
                "Stair reward inputs must be created by the task-owned builder"
            )
        if not isinstance(contract, StairCompleteRewardRuntimeContract):
            raise TypeError("Stair reward input contract is invalid")
        if not isinstance(values, tuple) or len(values) != len(contract.input_fields):
            raise ValueError("Stair reward input value count mismatch")
        object.__setattr__(self, "_StairRewardInputsV2__contract", contract)
        object.__setattr__(self, "_StairRewardInputsV2__values", values)

    @classmethod
    def _validated(
        cls,
        *,
        contract: StairCompleteRewardRuntimeContract,
        values: Mapping[str, Any],
    ) -> "StairRewardInputsV2":
        if not isinstance(values, Mapping):
            raise TypeError("Stair reward input values must be an object")
        if set(values) != set(contract.field_names):
            missing = sorted(set(contract.field_names) - set(values))
            extra = sorted(set(values) - set(contract.field_names))
            raise ValueError(
                "Stair reward input fields mismatch: "
                f"missing={missing!r} extra={extra!r}"
            )
        ordered = tuple(values[name] for name in contract.field_names)
        return cls(contract=contract, values=ordered, token=cls.__TOKEN)

    def __getattr__(self, name: str) -> Any:
        contract = object.__getattribute__(
            self,
            "_StairRewardInputsV2__contract",
        )
        try:
            index = contract.field_names.index(name)
        except ValueError as exc:
            raise AttributeError(name) from exc
        values = object.__getattribute__(self, "_StairRewardInputsV2__values")
        return values[index]

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("Stair reward inputs are immutable")

    @property
    def schema_hash(self) -> str:
        contract = object.__getattribute__(
            self,
            "_StairRewardInputsV2__contract",
        )
        return contract.input_schema_hash

    @property
    def field_names(self) -> tuple[str, ...]:
        contract = object.__getattribute__(
            self,
            "_StairRewardInputsV2__contract",
        )
        return contract.field_names


def stair_reward_input_values(inputs: StairRewardInputsV2) -> tuple[Any, ...]:
    """Return ordered values to trusted task runtime code only."""

    if type(inputs) is not StairRewardInputsV2:
        raise TypeError("Stair reward inputs have an invalid runtime type")
    return object.__getattribute__(
        inputs,
        "_StairRewardInputsV2__values",
    )


def stair_reward_input_vnext_values(
    inputs: StairRewardInputsVNext,
) -> tuple[Any, ...]:
    if type(inputs) is not StairRewardInputsVNext:
        raise TypeError("Stair VNext reward inputs have an invalid runtime type")
    return object.__getattribute__(
        inputs,
        "_StairRewardInputsVNext__values",
    )


def stair_reward_input_prompt_contract(
    contract: StairCompleteRewardRuntimeContract,
) -> dict[str, Any]:
    """Project the canonical task ABI without execution or private runtime data."""

    if not isinstance(contract, StairCompleteRewardRuntimeContract):
        raise TypeError("Stair reward input prompt contract is invalid")
    public_contract = contract.to_public_dict()
    source_transforms = public_contract["input_schema"]["source_transforms"]
    source_semantics = stair_reward_source_semantics_policy(contract)
    return {
        "schema_version": "true_rl_agent.reward_inputs_contract.stair.v12",
        "status": "available",
        "input_schema_hash": contract.input_schema_hash,
        "input_type": (
            "true_rl_agent.tasks.stair_climbing.reward_inputs.StairRewardInputsV2"
        ),
        "field_semantics_constraints": {
            "high_risk_axis_semantics": {
                "environment": (
                    "dim0=independent environments at one control step; output[i] only "
                    "from input[i]; no diff/roll/cumsum/sort/index/reduction across "
                    "dim0; declared history fields carry time"
                ),
                "foot_contact_force_history": (
                    "axes=[env,history(0 newest; larger older),foot(0 left;1 right),"
                    "xyz]; contact=norm xyz,max all 3 history,then threshold; "
                    "air-to-contact=index0 contact and older air"
                ),
                "spatial_vectors": (
                    "components=[x,y,z]; world +x stair-forward,+y left,+z up; "
                    "body/yaw +x robot-forward,+y left,+z up"
                ),
            },
            "remaining_goal_distance_m": {
                "source_equivalence": "goal_x-current_x",
                "sign": "positive before, zero at, negative beyond goal",
            },
            "base_height": {
                "source_equivalence": ("base_pos_relative_to_terrain_origin[:, 2]"),
                "reference_frame": ("terrain origin at the bottom start platform"),
                "local_clearance": False,
                "candidate_usage": (
                    "Includes staircase elevation, not clearance above the supporting "
                    "step; standing-height thresholds do not prove posture or support. "
                    "Use roll/pitch/projected_gravity, goal_progress_fraction, and "
                    "contact fields."
                ),
            },
            "projected_gravity": {
                "source_equivalence": "robot_data.projected_gravity_b[:, :3]",
                "reference_frame": "FLU base frame (+x forward,+y left,+z up)",
                "definition": (
                    "right-handed R_WB=Rz(yaw)@Ry(pitch)@Rx(roll); g_b=R_WB.T@[0,0,-1]"
                ),
                "components_xyz": [
                    "sin(pitch)",
                    "-sin(roll)*cos(pitch)",
                    "-cos(roll)*cos(pitch)",
                ],
                "small_angle_xyz": ["pitch", "-roll", "-1"],
                "candidate_usage": "Use -z for upright reward; +z favors inversion.",
            },
            "height_scan": {
                "source_transform_ref": "source_transforms.height_scan",
                "sensor_semantics": (
                    "downward terrain scan; positive=lower/farther, "
                    "negative=higher/closer; no overhead or torso obstacle sensing"
                ),
                "flattened_index_to_ray_mapping": "not_exposed",
                "candidate_usage": (
                    "Use order-invariant reductions only; ray directions are not exposed."
                ),
            },
            "raw_physical_magnitudes": source_semantics[
                "raw_physical_magnitude_contract"
            ],
            "base_contact_force": {
                "semantics": "current torso contact-force vector in Newtons",
                "binary_alternative": "base_contact",
                "candidate_usage": (
                    "Use base_contact for binary torso contact; dense force terms follow "
                    "raw_physical_magnitudes."
                ),
            },
            "termination_masks": {
                "fixed_active_terms": ["time_out", "base_contact"],
                "equivalent_fields": {
                    "terminated_non_timeout": "base_contact",
                },
                "candidate_usage": (
                    "Frozen Stair: base_contact and terminated_non_timeout are identical "
                    "masks; do not add both."
                ),
            },
            "progress_fraction": {
                "semantics": "elapsed fraction of the current episode horizon",
                "spatial_alternative": "goal_progress_fraction",
                "candidate_usage": (
                    "For episode-time scheduling only; it does not mean training "
                    "progress, position, proximity to the top, or completion. Spatial "
                    "progress is goal_progress_fraction."
                ),
            },
            "foot_contact_force": {
                "semantics": (
                    "support/contact force in Newtons; nonzero is required for support"
                ),
                "candidate_usage": (
                    "Contact is not inherently bad; normalize force-derived terms."
                ),
            },
        },
        "fields": [item.to_public_dict() for item in contract.input_fields],
        "source_transforms": source_transforms,
        "opaque_axis_contract": source_semantics["opaque_axis_contract"],
        "isolation": contract.to_public_dict()["dispatcher"]["input_isolation"],
        "forbidden_context": [
            "runner_execution_objects",
            "private_evaluation_assignment",
            "acceptance_and_artifact_state",
        ],
    }


__all__ = [
    "StairRewardInputsV2",
    "StairRewardInputsVNext",
    "stair_reward_input_prompt_contract",
    "stair_reward_input_values",
    "stair_reward_input_vnext_values",
]
