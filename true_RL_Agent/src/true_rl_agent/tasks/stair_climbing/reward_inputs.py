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
        "schema_version": "true_rl_agent.reward_inputs_contract.stair.v11",
        "status": "available",
        "input_schema_hash": contract.input_schema_hash,
        "input_type": (
            "true_rl_agent.tasks.stair_climbing.reward_inputs."
            "StairRewardInputsV2"
        ),
        "fields": [item.to_public_dict() for item in contract.input_fields],
        "source_transforms": source_transforms,
        "opaque_axis_contract": source_semantics["opaque_axis_contract"],
        "field_semantics_constraints": {
            "base_height": {
                "source_equivalence": (
                    "base_pos_relative_to_terrain_origin[:, 2]"
                ),
                "reference_frame": (
                    "per-environment terrain origin at the bottom start platform"
                ),
                "local_clearance": False,
                "candidate_usage": (
                    "This value includes the staircase elevation accumulated while "
                    "climbing; it is not base clearance above the current supporting "
                    "step. Do not compare it with nominal standing-height thresholds "
                    "to infer posture or support. Use roll/pitch or projected_gravity "
                    "for posture, goal_progress_fraction for proximity to the frozen "
                    "goal, and the declared contact fields for support."
                ),
            },
            "projected_gravity": {
                "source_equivalence": "robot_data.projected_gravity_b[:, :3]",
                "reference_frame": "FLU base frame (+x forward,+y left,+z up)",
                "definition": (
                    "right-handed R_WB=Rz(yaw)@Ry(pitch)@Rx(roll); "
                    "g_b=R_WB.T@[0,0,-1]"
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
                    "Use order-invariant reductions only; fixed ray directions are not exposed."
                ),
            },
            "raw_physical_magnitudes": source_semantics[
                "raw_physical_magnitude_contract"
            ],
            "base_contact_force": {
                "semantics": "raw current torso contact-force vector in Newtons",
                "binary_alternative": "base_contact",
                "candidate_usage": (
                    "Use base_contact for an exact binary torso-contact event. Dense "
                    "force-derived terms are governed by raw_physical_magnitudes."
                ),
            },
            "termination_masks": {
                "fixed_active_terms": ["time_out", "base_contact"],
                "equivalent_fields": {
                    "terminated_non_timeout": "base_contact",
                },
                "candidate_usage": (
                    "For the frozen Stair training task, base_contact is the only "
                    "non-timeout termination term, so base_contact and "
                    "terminated_non_timeout are identical masks. Do not add separate "
                    "reward mechanisms for both fields."
                ),
            },
            "progress_fraction": {
                "semantics": "elapsed fraction of the current episode horizon",
                "spatial_alternative": "goal_progress_fraction",
                "candidate_usage": (
                    "Use progress_fraction only for episode-time scheduling. It does not "
                    "measure training progress, forward position, proximity to the top, "
                    "or goal completion; use goal_progress_fraction for clamped spatial "
                    "progress from the frozen start to the frozen goal."
                ),
            },
            "foot_contact_force": {
                "semantics": (
                    "raw support/contact force in Newtons; nonzero force is required for "
                    "valid foot support"
                ),
                "candidate_usage": (
                    "Do not penalize raw foot-force magnitude as if all contact were bad; "
                    "preserve the support semantics and normalize any force-derived term."
                ),
            },
        },
        "isolation": contract.to_public_dict()["dispatcher"]["input_isolation"],
        "forbidden_context": [
            "runner_execution_objects",
            "private_evaluation_assignment",
            "acceptance_and_artifact_state",
        ],
    }


__all__ = [
    "StairRewardInputsV2",
    "stair_reward_input_prompt_contract",
    "stair_reward_input_values",
]
