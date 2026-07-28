"""Task-owned opaque input object for complete Stair reward programs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from true_rl_agent.tasks.stair_climbing.protocol_v2 import (
    StairCompleteRewardRuntimeContract,
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
    return {
        "schema_version": "true_rl_agent.reward_inputs_contract.stair.v2",
        "status": "available",
        "input_schema_hash": contract.input_schema_hash,
        "input_type": (
            "true_rl_agent.tasks.stair_climbing.reward_inputs."
            "StairRewardInputsV2"
        ),
        "fields": [item.to_public_dict() for item in contract.input_fields],
        "source_transforms": contract.to_public_dict()["input_schema"][
            "source_transforms"
        ],
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
