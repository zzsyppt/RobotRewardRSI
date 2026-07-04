"""Task-owned prompt semantics interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from true_rl_agent.prompting.task_sections import (
    DISALLOWED_TASK_PROMPT_KEY_PARTS,
    MAX_TASK_PROMPT_SECTIONS_BYTES,
    TASK_PROMPT_SECTION_KEYS,
    sanitize_task_prompt_sections,
)
from true_rl_agent.tasks.base import TaskSpec


class TaskPromptAdapter(Protocol):
    """Task-owned source for prompt sections that shared code should not interpret."""

    task_id: str

    def build_prompt_sections(
        self,
        *,
        public_records: Sequence[Mapping[str, Any]],
        public_context: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Return public-safe task prompt sections from public records only."""
        ...


@dataclass(frozen=True)
class NeutralTaskPromptAdapter:
    """Fallback prompt adapter for tasks that have not provided task semantics yet."""

    task_spec: TaskSpec

    @property
    def task_id(self) -> str:
        return self.task_spec.task_id

    def build_prompt_sections(
        self,
        *,
        public_records: Sequence[Mapping[str, Any]],
        public_context: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        _ = public_records
        _ = public_context
        sections = {
            "task": {
                "schema_version": "true_rl_agent.task_prompt.neutral.v1",
                "task_id": self.task_spec.task_id,
                "name": self.task_spec.name,
                "supported_policy_initializations": sorted(self.task_spec.supported_policy_initializations),
                "prompt_adapter": "neutral_fallback",
                "fixed_train_protocol_summary": {"status": "not_provided_by_task_prompt_adapter"},
                "fixed_eval_protocol_summary": {"status": "not_provided_by_task_prompt_adapter"},
                "candidate_must_not_change": [
                    "runner",
                    "eval protocol",
                    "training budget",
                    "seed policy",
                    "checkpoints",
                    "PPO",
                    "acceptance rules",
                ],
            },
            "reward_inputs_contract": {
                "schema_version": "true_rl_agent.reward_inputs_contract.neutral.v1",
                "status": "not_available",
                "reason": "task_prompt_adapter_not_provided",
            },
            "public_eval_contract": {
                "schema_version": "true_rl_agent.public_eval_contract.neutral.v1",
                "status": "not_available",
                "reason": "task_prompt_adapter_not_provided",
            },
            "public_feedback_summary": {
                "schema_version": "true_rl_agent.public_feedback_summary.neutral.v1",
                "status": "not_available",
                "reason": "no_task_specific_public_feedback_adapter",
            },
            "runtime_feedback_summary": {
                "schema_version": "true_rl_agent.runtime_feedback_summary.neutral.v1",
                "status": "not_available",
                "reason": "no_task_specific_runtime_feedback_adapter",
            },
            "known_good_examples": [],
            "selection_guidance": [
                "Use only prompt-visible public inputs.",
                "Return a candidate package compatible with the shared candidate component contract.",
                "Do not modify runner, eval protocol, training budget, seed policy, checkpoints, PPO, or acceptance rules.",
            ],
        }
        return sanitize_task_prompt_sections(sections)
