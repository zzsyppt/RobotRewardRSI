"""Push-recovery ownership of candidate component classifications."""

from __future__ import annotations

from true_rl_agent.candidates.schedule_contract import REWARD_WEIGHT_SCALE_KEY
from true_rl_agent.candidates.schema import CandidateSurfacePolicy

PUSH_RECOVERY_CANDIDATE_SURFACE_POLICY = CandidateSurfacePolicy(
    parameter_only_components=frozenset(
        {
            "reward_weight",
            REWARD_WEIGHT_SCALE_KEY,
            "terrain_range",
            "height_range",
            "ppo_hyperparameter",
            "eval_threshold",
            "budget",
            "seed",
            "schedule",
            "single_stage_numeric_schedule",
        }
    ),
    diagnostic_components=frozenset({"diagnostic"}),
    engineering_components=frozenset(
        {"failure_classifier", "logging", "public_projection"}
    ),
    code_level_components=frozenset(
        {"reward_callable", "control_logic", "recovery_logic"}
    ),
)


__all__ = ["PUSH_RECOVERY_CANDIDATE_SURFACE_POLICY"]
