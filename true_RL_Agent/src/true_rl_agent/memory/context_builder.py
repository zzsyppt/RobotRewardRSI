"""Public context and prompt payload construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from true_rl_agent.agent_core.context import (
    build_public_context,
    sanitize_prompt_public_context,
)
from true_rl_agent.candidates.prompt_projection import (
    assert_public_only_prompt_payload,
    build_prompt_payload,
    serialize_prompt_payload,
)
from true_rl_agent.memory.records import ExperimentRecord, PromptPayloadRef
from true_rl_agent.memory.task_adapter import TaskMemoryAdapter
from true_rl_agent.prompting.source_catalog import build_prompt_source_context
from true_rl_agent.tasks.prompt_adapter import TaskPromptAdapter, sanitize_task_prompt_sections


class PublicContextBuilder:
    """Build next-round prompt context from public experiment records only."""

    def __init__(
        self,
        task_memory_adapter: TaskMemoryAdapter | None = None,
        task_prompt_adapter: TaskPromptAdapter | None = None,
    ) -> None:
        self.task_memory_adapter = task_memory_adapter
        self.task_prompt_adapter = task_prompt_adapter

    def build(
        self,
        records: Sequence[ExperimentRecord | Mapping[str, Any]],
        *,
        latest_failed_candidate: Any = None,
    ) -> dict[str, Any]:
        experiment_records = [_experiment_record(record) for record in records]
        public_records = [record.ledger_public_record() for record in experiment_records]
        context = build_public_context(public_records)
        if self.task_memory_adapter is not None:
            task_summary = self.task_memory_adapter.build_public_summary(public_records)
            if task_summary:
                context["task_public_summary"] = sanitize_prompt_public_context(dict(task_summary))
        context = sanitize_prompt_public_context(context)
        task_prompt_sections: dict[str, Any] = {}
        if self.task_prompt_adapter is not None:
            task_prompt_sections = self.task_prompt_adapter.build_prompt_sections(
                public_records=public_records,
                public_context=context,
            )
            if task_prompt_sections:
                task_prompt_sections = sanitize_task_prompt_sections(task_prompt_sections)
                context["task_prompt_sections"] = task_prompt_sections
        context.update(
            build_prompt_source_context(
                experiment_records,
                known_good_examples=task_prompt_sections.get("known_good_examples"),
                latest_failed_candidate=latest_failed_candidate,
            )
        )
        return sanitize_prompt_public_context(context)

    def build_prompt_payload(
        self,
        records: Sequence[ExperimentRecord | Mapping[str, Any]],
        *,
        latest_failed_candidate: Any = None,
    ) -> dict[str, Any]:
        payload = build_prompt_payload(
            public_context=self.build(records, latest_failed_candidate=latest_failed_candidate)
        )
        assert_public_only_prompt_payload(payload)
        return payload

    def save_prompt_payload(self, path: Path, payload: Mapping[str, Any]) -> PromptPayloadRef:
        sanitized_payload = sanitize_prompt_public_context(dict(payload))
        assert_public_only_prompt_payload(sanitized_payload)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(serialize_prompt_payload(sanitized_payload) + "\n", encoding="utf-8")
        return PromptPayloadRef(path=str(path))


def _public_record(record: ExperimentRecord | Mapping[str, Any]) -> dict[str, Any]:
    return _experiment_record(record).ledger_public_record()


def _experiment_record(record: ExperimentRecord | Mapping[str, Any]) -> ExperimentRecord:
    if isinstance(record, ExperimentRecord):
        return record
    return ExperimentRecord.from_ledger_record(record)
