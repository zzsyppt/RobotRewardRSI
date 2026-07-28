"""Push-recovery task-owned RewardInputs extension."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from true_rl_agent.candidates.components import RewardInputs


@dataclass(frozen=True)
class PushRecoveryRewardInputs(RewardInputs):
    """Reward inputs available only for the push-recovery task."""

    push_active: Any
    push_vec_public: Any
