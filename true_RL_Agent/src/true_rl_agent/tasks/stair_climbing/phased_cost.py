"""Task-owned cost envelope for the fixed two-trial Stair evaluation topology."""

from __future__ import annotations

from typing import Any

from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.tasks.stair_climbing.config import (
    STAIR_TASK_ID,
    StairClimbingTaskConfig,
)


STAIR_PHASED_EVAL_COST_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.phased_eval_cost_contract.v2"
)


def stair_phased_evaluation_cost_contract(
    config: StairClimbingTaskConfig,
) -> dict[str, Any]:
    """Return the scientific-path ceiling; execution retries remain separate."""

    if not isinstance(config, StairClimbingTaskConfig):
        raise TypeError("Stair phased eval cost requires StairClimbingTaskConfig")
    episodes_per_seed = len(config.evaluation_profile.private_cell_assignment())
    per_generation_phases = {
        "public_dev": {
            "max_jobs": 2,
            "seed_count_per_job": config.public_seed_count,
        },
        "public_accept": {
            "max_jobs": 2,
            "seed_count_per_job": config.public_accept_seed_count,
        },
        "sealed_validation": {
            "max_jobs": 1,
            "seed_count_per_job": config.sealed_validation_seed_count,
        },
    }
    per_generation_seed_units = sum(
        item["max_jobs"] * item["seed_count_per_job"]
        for item in per_generation_phases.values()
    )
    session_bootstrap = {
        "parent_public_dev": {
            "max_jobs": 1,
            "seed_count_per_job": config.public_seed_count,
            "purpose": "create the frozen-parent guard reference for generation one",
        }
    }
    session_bootstrap_seed_units = config.public_seed_count
    payload = {
        "schema_version": STAIR_PHASED_EVAL_COST_SCHEMA_VERSION,
        "task_id": STAIR_TASK_ID,
        "evaluation_protocol_hash": config.evaluation_protocol().protocol_hash,
        "evaluation_profile_hash": config.evaluation_profile.profile_hash,
        "topology": "parent_bootstrap_then_two_public_siblings_winner_only_sealed_v2",
        "session_bootstrap": session_bootstrap,
        "per_generation_phases": per_generation_phases,
        "max_seed_units_per_generation": per_generation_seed_units,
        "max_session_bootstrap_seed_units": session_bootstrap_seed_units,
        "episodes_per_seed_unit": episodes_per_seed,
        "max_episode_executions_per_generation": (
            per_generation_seed_units * episodes_per_seed
        ),
        "max_session_bootstrap_episode_executions": (
            session_bootstrap_seed_units * episodes_per_seed
        ),
        "scientific_trial_count": 2,
        "sealed_query_count": 1,
        "execution_retry_reserve_included": False,
    }
    return {**payload, "contract_hash": canonical_json_hash(payload)}


__all__ = [
    "STAIR_PHASED_EVAL_COST_SCHEMA_VERSION",
    "stair_phased_evaluation_cost_contract",
]
