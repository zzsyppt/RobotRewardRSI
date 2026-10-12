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


@dataclass(frozen=True)
class PushRecoveryV2RewardInputs(PushRecoveryRewardInputs):
    """Versioned task-only normalization; historical input identities stay intact."""

    nominal_base_height: Any
    base_height_relative: Any
    base_lin_vel_scaled: Any
    base_ang_vel_scaled: Any
    terminated_non_timeout: Any
