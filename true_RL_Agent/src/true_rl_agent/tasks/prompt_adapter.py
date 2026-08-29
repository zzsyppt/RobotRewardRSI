"""Task-owned prompt semantics interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from true_rl_agent.prompting.task_sections import (
    MAX_TASK_PROMPT_SECTIONS_BYTES as MAX_TASK_PROMPT_SECTIONS_BYTES,
)
from true_rl_agent.prompting.task_sections import sanitize_task_prompt_sections
from true_rl_agent.tasks.base import TaskSpec


@dataclass(frozen=True)
class PromptPublicHistoryProjection:
    """Task-owned provider view over already validated public history.

    Acceptance replay remains authoritative.  This projection only chooses the
    score semantics and record view exposed to the next provider request.
    """

    records: tuple[Mapping[str, Any], ...]
    public_score_source: str | None = None
    initial_anchor_score: float | None = None


class TaskPromptAdapter(Protocol):
    """Task-owned source for prompt sections that shared code should not interpret."""

    task_id: str

    def build_prompt_sections(
        self,
        *,
        public_records: Sequence[Mapping[str, Any]],
        public_context: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Return sections from shared, validated public records only."""
        ...

    def build_prompt_capacity_preflight_sections(self) -> Mapping[str, Any]:
        """Return the task-owned maximum minimum-core shape for startup checks."""
        ...

    def prompt_public_record_scrub_policy(self) -> Mapping[str, Sequence[str]]:
        """Return task-owned extra public-record scrub terms before prompt construction."""
        ...

    def project_prompt_public_history(
        self,
        *,
        public_records: Sequence[Mapping[str, Any]],
        initial_anchor_public_evidence: Mapping[str, Any] | None,
    ) -> PromptPublicHistoryProjection:
        """Project validated history into task-owned provider score semantics."""
        ...

    def nonaccepted_authoring_tiebreak_key(
        self,
        public_record: Mapping[str, Any],
    ) -> tuple[float, ...]:
        """Return public task metrics used only when nonaccepted scores tie."""
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
        public_feedback_summary: dict[str, Any] = {
            "schema_version": "true_rl_agent.public_feedback_summary.neutral.v1",
            "status": "not_available",
            "reason": "no_task_specific_public_feedback_adapter",
        }
        task_public_summary = public_context.get("task_public_summary")
        if isinstance(task_public_summary, Mapping):
            public_feedback_summary["task_public_summary"] = dict(task_public_summary)
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
            "public_feedback_summary": public_feedback_summary,
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

    def build_prompt_capacity_preflight_sections(self) -> Mapping[str, Any]:
        return self.build_prompt_sections(public_records=(), public_context={})

    def prompt_public_record_scrub_policy(self) -> Mapping[str, Sequence[str]]:
        return {"blocked_key_parts": (), "blocked_text_parts": ()}

    def project_prompt_public_history(
        self,
        *,
        public_records: Sequence[Mapping[str, Any]],
        initial_anchor_public_evidence: Mapping[str, Any] | None,
    ) -> PromptPublicHistoryProjection:
        _ = initial_anchor_public_evidence
        return PromptPublicHistoryProjection(
            records=tuple(dict(record) for record in public_records),
        )

    def nonaccepted_authoring_tiebreak_key(
        self,
        public_record: Mapping[str, Any],
    ) -> tuple[float, ...]:
        _ = public_record
        return ()
