"""Prompt-visible Stair semantics derived only from validated public records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from true_rl_agent.agent_core.context import (
    sanitize_public_records_with_policy,
    sanitize_public_value_with_policy,
)
from true_rl_agent.tasks.prompt_adapter import sanitize_task_prompt_sections
from true_rl_agent.tasks.stair_climbing.config import (
    STAIR_TASK_ID,
    StairClimbingTaskConfig,
)

_SCRUBBED_KEY_PARTS = (
    "raw_seed",
    "seed_cohort_nonce",
    "env_to_bin",
    "terrain_env_origins",
    "sealed_validation_evidence",
    "checkpoint_local_path",
    "checkpoint_output_path",
)


@dataclass(frozen=True)
class StairClimbingPromptAdapter:
    config: StairClimbingTaskConfig
    task_id: str = STAIR_TASK_ID

    def __post_init__(self) -> None:
        if not isinstance(self.config, StairClimbingTaskConfig):
            raise TypeError("stair prompt adapter requires StairClimbingTaskConfig")

    def build_prompt_sections(
        self,
        *,
        public_records: Sequence[Mapping[str, Any]],
        public_context: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        policy = self.prompt_public_record_scrub_policy()
        safe_records = sanitize_public_records_with_policy(
            public_records,
            blocked_key_parts=policy["blocked_key_parts"],
            blocked_text_parts=policy["blocked_text_parts"],
        )
        safe_context = sanitize_public_value_with_policy(
            public_context,
            blocked_key_parts=policy["blocked_key_parts"],
            blocked_text_parts=policy["blocked_text_parts"],
        )
        if not isinstance(safe_context, Mapping):
            safe_context = {}
        sections = {
            "task": self._task_section(),
            "reward_inputs_contract": {
                "schema_version": "true_rl_agent.reward_inputs_contract.stair.v1",
                "status": "not_available",
                "reason": "pre_reset_complete_reward_runtime_not_activated",
            },
            "public_eval_contract": self._public_eval_contract(),
            "public_feedback_summary": _latest_public_feedback(
                safe_records,
                safe_context,
            ),
            "runtime_feedback_summary": {
                "schema_version": "true_rl_agent.runtime_feedback_summary.stair.v1",
                "status": "not_available",
                "reason": "complete_reward_runtime_not_activated",
            },
            "known_good_examples": [],
            "selection_guidance": [
                "Use only validated public failure, diagnostic, and progress summaries.",
                "Do not author a Stair reward program until the reward input contract is available.",
                "Do not change training budget, PPO, seeds, terrain bins, termination rules, or acceptance rules.",
            ],
        }
        return sanitize_task_prompt_sections(sections)

    def prompt_public_record_scrub_policy(self) -> Mapping[str, Sequence[str]]:
        return {
            "blocked_key_parts": _SCRUBBED_KEY_PARTS,
            "blocked_text_parts": _SCRUBBED_KEY_PARTS,
        }

    def _task_section(self) -> dict[str, Any]:
        base = self.config.protocol_v1.to_dict()
        evaluation = self.config.evaluation_protocol().to_public_dict()
        return {
            "schema_version": "true_rl_agent.task_prompt.stair_climbing.v1",
            "task_id": self.task_id,
            "name": "Unitree G1 Stair Climbing",
            "objective": (
                "Train from the fixed random initialization protocol to climb the "
                "one-way staircase and hold the top platform without falling or base contact."
            ),
            "candidate_authoring_status": "not_activated_until_complete_reward_runtime",
            "fixed_train_protocol_summary": {
                "budget_steps": self.config.budget_steps,
                "train_num_envs": self.config.train_num_envs,
                "train_max_iterations": self.config.train_max_iterations,
                "train_steps_per_iteration": self.config.train_steps_per_iteration,
                "policy_initialization": "fixed_random_session_seed",
                "candidate_can_change": False,
            },
            "fixed_eval_protocol_summary": {
                **evaluation,
                "bin_ids": [str(item["bin_id"]) for item in self.config.protocol_v1.eval_bins],
                "episodes_per_seed_per_bin": int(
                    base["evaluation"]["env_to_bin_mapping"]["envs_per_bin"]
                ),
                "statistics_unit": "one_equal_weight_seven_bin_macro_score_per_seed",
                "seed_values_exposed": False,
            },
            "primary_outcomes": list(base["episode_outcomes"]["primary_reasons"]),
            "diagnostic_flags": list(base["episode_outcomes"]["diagnostic_flags"]),
            "candidate_must_not_change": [
                "checkpoint_policy"
                if item == "checkpoint_selector"
                else str(item)
                for item in base["candidate_boundary"]["forbidden_components"]
            ],
        }

    def _public_eval_contract(self) -> dict[str, Any]:
        return {
            "schema_version": "true_rl_agent.public_eval_contract.stair.v1",
            "score_source": "public_accept.score_stats.mean",
            "aggregation": "per_seed_seven_bin_macro_then_cross_seed_statistics",
            "public_feedback_source": "public_dev_task_recorder_projection",
            "primary_failure_is_mutually_exclusive": True,
            "diagnostic_flags_may_overlap": True,
            "classification_warnings_block_pass": True,
        }


def _latest_public_feedback(
    public_records: Sequence[Mapping[str, Any]],
    public_context: Mapping[str, Any],
) -> dict[str, Any]:
    for record in reversed(public_records):
        result = _mapping(record.get("public_result"))
        metrics = _mapping(result.get("metrics"))
        feedback = metrics.get("prompt_feedback")
        if isinstance(feedback, Mapping):
            return dict(feedback)
    task_summary = public_context.get("task_public_summary")
    if isinstance(task_summary, Mapping) and task_summary:
        return {
            "schema_version": "true_rl_agent.task.stair_climbing.prompt_feedback.v1",
            "status": "memory_only",
            "task_public_summary": dict(task_summary),
        }
    return {
        "schema_version": "true_rl_agent.task.stair_climbing.prompt_feedback.v1",
        "status": "not_available",
        "reason": "no_validated_public_stair_evaluation",
    }


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


__all__ = ["StairClimbingPromptAdapter"]
