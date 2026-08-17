"""Task-owned deterministic complete-reward smoke fixture for Stair."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from true_rl_agent.candidates.capability_readiness import (
    COMPLETE_REWARD_PROGRAM_CAPABILITY_ID,
)
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

STAIR_SMOKE_FIXTURE_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.complete_reward_smoke_fixture.v1"
)
STAIR_SMOKE_NUM_ENVS = 2


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
    task_id: str = STAIR_TASK_ID

    def __post_init__(self) -> None:
        if not isinstance(self.config, StairClimbingTaskConfig):
            raise TypeError("stair smoke adapter requires StairClimbingTaskConfig")

    @property
    def candidate_capability_ids(self) -> frozenset[str]:
        return frozenset({COMPLETE_REWARD_PROGRAM_CAPABILITY_ID})

    @property
    def contract(self) -> StairCompleteRewardRuntimeContract:
        return self.config.protocol_v2.complete_reward_runtime

    @property
    def fixture_hash(self) -> str:
        return canonical_json_hash(
            {
                "schema_version": STAIR_SMOKE_FIXTURE_SCHEMA_VERSION,
                "input_schema_hash": self.contract.input_schema_hash,
                "num_envs": STAIR_SMOKE_NUM_ENVS,
                "value_recipe": "typed_cpu_zeros_v1",
            }
        )

    def _tensor_values(self) -> dict[str, Any]:
        import torch

        values = {}
        for field in self.contract.input_fields:
            shape = (STAIR_SMOKE_NUM_ENVS, *field.trailing_shape)
            dtype = torch.bool if field.dtype == "bool" else torch.float32
            values[field.name] = torch.zeros(shape, dtype=dtype, device="cpu")
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
            },
        )


__all__ = [
    "STAIR_SMOKE_FIXTURE_SCHEMA_VERSION",
    "STAIR_SMOKE_NUM_ENVS",
    "StairClimbingComponentSmokeAdapter",
    "StairSmokeRewardInputs",
]
