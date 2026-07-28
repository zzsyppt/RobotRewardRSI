"""Phase-gated component smoke adapter for Stair."""

from __future__ import annotations

from dataclasses import dataclass

from true_rl_agent.tasks.stair_climbing.config import (
    STAIR_TASK_ID,
    StairClimbingTaskConfig,
)


@dataclass(frozen=True)
class StairClimbingComponentSmokeAdapter:
    """Do not fabricate complete-reward inputs before Phase C activates them."""

    config: StairClimbingTaskConfig
    task_id: str = STAIR_TASK_ID

    def __post_init__(self) -> None:
        if not isinstance(self.config, StairClimbingTaskConfig):
            raise TypeError("stair smoke adapter requires StairClimbingTaskConfig")

    def build_component_smoke_fixture(self) -> None:
        return None


__all__ = ["StairClimbingComponentSmokeAdapter"]
