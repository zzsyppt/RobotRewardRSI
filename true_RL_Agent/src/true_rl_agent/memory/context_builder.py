"""Public context and prompt payload construction."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from true_rl_agent.agent_core.context import (
    build_public_context_from_projected,
    sanitize_prompt_public_context,
    sanitize_public_records_with_policy,
)
from true_rl_agent.candidates.capability_profile import CandidateCapabilityProfile
from true_rl_agent.evaluation.acceptance_protocol import AcceptanceProtocol
from true_rl_agent.evaluation.acceptance_replay import (
    ProjectedPublicHistory,
    project_public_history,
)
from true_rl_agent.evaluation.current_best import CurrentBestEvalStats
from true_rl_agent.memory.records import ExperimentRecord, PromptPayloadRef
from true_rl_agent.memory.task_adapter import TaskMemoryAdapter
from true_rl_agent.prompting import (
    assert_public_only_prompt_payload,
    build_prompt_payload,
    serialize_prompt_payload,
)
from true_rl_agent.prompting.source_catalog import (
    build_prompt_source_context_from_projected,
)
from true_rl_agent.tasks.prompt_adapter import (
    TaskPromptAdapter,
    sanitize_task_prompt_sections,
)


@dataclass(frozen=True)
class PublicPromptBuild:
    """One public-record projection split into internal and provider views."""

    round_public_context: Mapping[str, Any]
    provider_payload: dict[str, Any]


class PublicContextBuilder:
    """Build next-round prompt context from public experiment records only."""

    def __init__(
        self,
        task_memory_adapter: TaskMemoryAdapter | None = None,
        task_prompt_adapter: TaskPromptAdapter | None = None,
        acceptance_protocol: AcceptanceProtocol | None = None,
        initial_current_best: CurrentBestEvalStats | None = None,
        session_genesis_hash: str | None = None,
        candidate_capability_profile: CandidateCapabilityProfile | None = None,
        initial_reference_program_source: str | None = None,
    ) -> None:
        self.task_memory_adapter = task_memory_adapter
        self.task_prompt_adapter = task_prompt_adapter
        self.acceptance_protocol = acceptance_protocol
        self.initial_current_best = initial_current_best
        self.session_genesis_hash = session_genesis_hash
        self.candidate_capability_profile = candidate_capability_profile
        self.initial_reference_program_source = initial_reference_program_source

    def build(
        self,
        records: Sequence[ExperimentRecord | Mapping[str, Any]],
        *,
        latest_failed_candidate: Any = None,
    ) -> dict[str, Any]:
        # Compatibility-only argument: failed-candidate source projection belongs
        # to the provider view built by build_prompt().
        _ = latest_failed_candidate
        _, history = self._prepare_records(records)
        return self._build_round_public_context(history)

    def build_prompt(
        self,
        records: Sequence[ExperimentRecord | Mapping[str, Any]],
        *,
        latest_failed_candidate: Any = None,
        candidate_repair_feedback: Mapping[str, Any] | None = None,
        source_payloads: Sequence[Mapping[str, Any]] | None = None,
        frame_kind: str | None = None,
        runner_capability_readiness: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> PublicPromptBuild:
        experiment_records, history = self._prepare_records(records)
        public_records = history.records
        round_public_context = self._build_round_public_context(history)
        provider_context = dict(round_public_context)

        task_prompt_sections = self._build_task_prompt_sections(
            public_records=public_records,
            round_public_context=round_public_context,
        )
        if task_prompt_sections:
            provider_context["task_prompt_sections"] = task_prompt_sections
        provider_context.update(
            build_prompt_source_context_from_projected(
                experiment_records,
                history,
                known_good_examples=task_prompt_sections.get("known_good_examples"),
                latest_failed_candidate=latest_failed_candidate,
                initial_reference_program_source=(
                    self.initial_reference_program_source
                ),
            )
        )
        if candidate_repair_feedback is not None:
            provider_context["candidate_repair_feedback"] = sanitize_prompt_public_context(
                dict(candidate_repair_feedback)
            )
        if source_payloads is not None:
            provider_context["source_payloads"] = [dict(item) for item in source_payloads]

        provider_payload = build_prompt_payload(
            public_context=sanitize_prompt_public_context(provider_context),
            frame_kind=frame_kind,
            runner_capability_readiness=runner_capability_readiness,
            candidate_capability_profile=self.candidate_capability_profile,
        )
        assert_public_only_prompt_payload(provider_payload)
        return PublicPromptBuild(
            round_public_context=round_public_context,
            provider_payload=provider_payload,
        )

    def _prepare_records(
        self,
        records: Sequence[ExperimentRecord | Mapping[str, Any]],
    ) -> tuple[list[ExperimentRecord], ProjectedPublicHistory]:
        experiment_records = [_experiment_record(record) for record in records]
        raw_public_records = [record.ledger_public_record() for record in experiment_records]
        history = project_public_history(
            raw_public_records,
            initial_current_best=self.initial_current_best,
            expected_session_genesis_hash=self.session_genesis_hash,
            acceptance_protocol=self.acceptance_protocol,
        )
        record_scrub_policy = _prompt_public_record_scrub_policy(self.task_prompt_adapter)
        public_records = sanitize_public_records_with_policy(
            history.records,
            blocked_key_parts=record_scrub_policy["blocked_key_parts"],
            blocked_text_parts=record_scrub_policy["blocked_text_parts"],
        )
        sanitized_history = ProjectedPublicHistory(
            records=tuple(public_records),
            initial_current_best=history.initial_current_best,
            session_genesis_hash=history.session_genesis_hash,
            acceptance_protocol_hash=history.acceptance_protocol_hash,
        )
        projected_experiment_records = [
            replace(record, public_record=public_record)
            for record, public_record in zip(
                experiment_records, public_records, strict=True
            )
        ]
        return projected_experiment_records, sanitized_history

    def _build_round_public_context(
        self,
        history: ProjectedPublicHistory,
    ) -> dict[str, Any]:
        context = build_public_context_from_projected(history)
        if self.task_memory_adapter is not None:
            task_summary = self.task_memory_adapter.build_public_summary(
                history.records,
            )
            if task_summary:
                context["task_public_summary"] = sanitize_prompt_public_context(dict(task_summary))
        return sanitize_prompt_public_context(context)

    def _build_task_prompt_sections(
        self,
        *,
        public_records: Sequence[Mapping[str, Any]],
        round_public_context: Mapping[str, Any],
    ) -> dict[str, Any]:
        if self.task_prompt_adapter is not None:
            task_prompt_sections = self.task_prompt_adapter.build_prompt_sections(
                public_records=public_records,
                public_context=sanitize_prompt_public_context(dict(round_public_context)),
            )
            if task_prompt_sections:
                return sanitize_task_prompt_sections(task_prompt_sections)
        return {}

    def build_prompt_payload(
        self,
        records: Sequence[ExperimentRecord | Mapping[str, Any]],
        *,
        latest_failed_candidate: Any = None,
        candidate_repair_feedback: Mapping[str, Any] | None = None,
        source_payloads: Sequence[Mapping[str, Any]] | None = None,
        frame_kind: str | None = None,
        runner_capability_readiness: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        return self.build_prompt(
            records,
            latest_failed_candidate=latest_failed_candidate,
            candidate_repair_feedback=candidate_repair_feedback,
            source_payloads=source_payloads,
            frame_kind=frame_kind,
            runner_capability_readiness=runner_capability_readiness,
        ).provider_payload

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


def _prompt_public_record_scrub_policy(task_prompt_adapter: TaskPromptAdapter | None) -> dict[str, tuple[str, ...]]:
    if task_prompt_adapter is None:
        return {"blocked_key_parts": (), "blocked_text_parts": ()}
    policy_fn = getattr(task_prompt_adapter, "prompt_public_record_scrub_policy", None)
    if not callable(policy_fn):
        return {"blocked_key_parts": (), "blocked_text_parts": ()}
    policy = policy_fn()
    if not isinstance(policy, Mapping):
        return {"blocked_key_parts": (), "blocked_text_parts": ()}
    return {
        "blocked_key_parts": _string_tuple(policy.get("blocked_key_parts")),
        "blocked_text_parts": _string_tuple(policy.get("blocked_text_parts")),
    }


def _string_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return ()
    return tuple(str(item) for item in value if isinstance(item, str) and item)
