"""Public context and prompt payload construction."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from true_rl_agent.agent_core.context import (
    build_public_context_from_projected,
    sanitize_prompt_public_context,
    sanitize_public_records_with_policy,
)
from true_rl_agent.agent_core.ledger import (
    NON_PROVIDER_AUTHORING_MODE,
    require_ledger_authoring_mode,
)
from true_rl_agent.candidates.capability_profile import CandidateCapabilityProfile
from true_rl_agent.evaluation.acceptance_protocol import AcceptanceProtocol
from true_rl_agent.evaluation.acceptance_replay import (
    ProjectedPublicHistory,
    project_public_history,
)
from true_rl_agent.evaluation.binding import (
    canonical_json_hash,
    require_sha256_checksum,
    require_sha256_digest,
)
from true_rl_agent.evaluation.current_best import CurrentBestEvalStats
from true_rl_agent.evaluation.public_result import PublicEvalResult
from true_rl_agent.memory.records import ExperimentRecord, PromptPayloadRef
from true_rl_agent.memory.task_adapter import TaskMemoryAdapter
from true_rl_agent.prompt_artifacts import publish_prompt_payload
from true_rl_agent.prompting import (
    assert_public_only_prompt_payload,
    build_prompt_payload,
)
from true_rl_agent.prompting.source_catalog import (
    build_prompt_source_context_from_projected,
)
from true_rl_agent.prompting.task_sections import (
    validate_provider_task_prompt_receipt,
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
    parent_program_source_checksum: str | None


def preflight_task_prompt_capacity(
    *,
    task_prompt_adapter: TaskPromptAdapter,
    candidate_capability_profile: CandidateCapabilityProfile | None,
) -> dict[str, Any]:
    """Validate a task-owned worst-case shape before a live invocation exists."""

    preflight_builder = getattr(
        task_prompt_adapter,
        "build_prompt_capacity_preflight_sections",
        None,
    )
    if callable(preflight_builder):
        first_raw = preflight_builder()
        second_raw = preflight_builder()
    else:
        first_raw = task_prompt_adapter.build_prompt_sections(
            public_records=(),
            public_context={},
        )
        second_raw = task_prompt_adapter.build_prompt_sections(
            public_records=(),
            public_context={},
        )
    first = sanitize_task_prompt_sections(first_raw)
    second = sanitize_task_prompt_sections(second_raw)
    if first != second:
        raise ValueError("task prompt capacity preflight is not deterministic")

    provider_payload = build_prompt_payload(
        public_context={"task_prompt_sections": first},
        frame_kind="candidate_draft",
        candidate_capability_profile=candidate_capability_profile,
    )
    frame = provider_payload.get("authoring_context_frame")
    if not isinstance(frame, Mapping):
        raise ValueError("prompt capacity preflight has no authoring context frame")
    budget = frame.get("budget_metadata")
    if not isinstance(budget, Mapping):
        raise ValueError("prompt capacity preflight has no budget metadata")
    receipt = budget.get("provider_task_sections")
    return validate_provider_task_prompt_receipt(
        receipt,
        prompt_payload=provider_payload,
    )


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
        initial_anchor_public_evidence: Mapping[str, Any] | None = None,
        expected_initial_policy_state_hash: str | None = None,
        ledger_authoring_mode: str = NON_PROVIDER_AUTHORING_MODE,
        expected_ledger_epoch_hash: str | None = None,
    ) -> None:
        self.task_memory_adapter = task_memory_adapter
        self.task_prompt_adapter = task_prompt_adapter
        self.acceptance_protocol = acceptance_protocol
        self.initial_current_best = initial_current_best
        self.session_genesis_hash = session_genesis_hash
        self.candidate_capability_profile = candidate_capability_profile
        self.initial_reference_program_source = initial_reference_program_source
        self.initial_anchor_public_evidence = _validate_initial_anchor_public_evidence(
            initial_anchor_public_evidence,
            initial_current_best=initial_current_best,
            acceptance_protocol=acceptance_protocol,
        )
        self.expected_initial_policy_state_hash = (
            expected_initial_policy_state_hash
        )
        self.ledger_authoring_mode = require_ledger_authoring_mode(
            ledger_authoring_mode
        )
        self.expected_ledger_epoch_hash = expected_ledger_epoch_hash

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
        authoring_capability_readiness: Mapping[str, Mapping[str, Any]] | None = None,
        candidate_verifier_prompt_rules: Mapping[str, Any] | None = None,
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
                acceptance_protocol=self.acceptance_protocol,
                known_good_examples=task_prompt_sections.get("known_good_examples"),
                latest_failed_candidate=latest_failed_candidate,
                initial_reference_program_source=(
                    self.initial_reference_program_source
                ),
                nonaccepted_authoring_tiebreak_key=(
                    _nonaccepted_authoring_tiebreak_key(self.task_prompt_adapter)
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
            authoring_capability_readiness=authoring_capability_readiness,
            candidate_capability_profile=self.candidate_capability_profile,
            candidate_verifier_prompt_rules=candidate_verifier_prompt_rules,
        )
        assert_public_only_prompt_payload(provider_payload)
        lineage = provider_payload.get("candidate_lineage")
        if not isinstance(lineage, Mapping):
            raise ValueError("provider prompt has no candidate lineage")
        parent_program_source_checksum = lineage.get(
            "parent_program_source_checksum"
        )
        if parent_program_source_checksum is not None:
            parent_program_source_checksum = require_sha256_checksum(
                parent_program_source_checksum,
                "prompt parent program source checksum",
            )
        return PublicPromptBuild(
            round_public_context=round_public_context,
            provider_payload=provider_payload,
            parent_program_source_checksum=parent_program_source_checksum,
        )

    def _prepare_records(
        self,
        records: Sequence[ExperimentRecord | Mapping[str, Any]],
    ) -> tuple[list[ExperimentRecord], ProjectedPublicHistory]:
        experiment_records = [_experiment_record(record) for record in records]
        replay_records = [record.replay_record() for record in experiment_records]
        history = project_public_history(
            replay_records,
            initial_current_best=self.initial_current_best,
            expected_session_genesis_hash=self.session_genesis_hash,
            acceptance_protocol=self.acceptance_protocol,
            expected_initial_policy_state_hash=(
                self.expected_initial_policy_state_hash
            ),
            expected_authoring_mode=self.ledger_authoring_mode,
            expected_ledger_epoch_hash=self.expected_ledger_epoch_hash,
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
            current_best=history.current_best,
            public_score_source=history.public_score_source,
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
            adapter_context = sanitize_prompt_public_context(
                dict(round_public_context)
            )
            if self.initial_anchor_public_evidence is not None:
                adapter_context["initial_anchor_public_evidence"] = dict(
                    self.initial_anchor_public_evidence
                )
            task_prompt_sections = self.task_prompt_adapter.build_prompt_sections(
                public_records=public_records,
                public_context=adapter_context,
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
        authoring_capability_readiness: Mapping[str, Mapping[str, Any]] | None = None,
        candidate_verifier_prompt_rules: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.build_prompt(
            records,
            latest_failed_candidate=latest_failed_candidate,
            candidate_repair_feedback=candidate_repair_feedback,
            source_payloads=source_payloads,
            frame_kind=frame_kind,
            authoring_capability_readiness=authoring_capability_readiness,
            candidate_verifier_prompt_rules=candidate_verifier_prompt_rules,
        ).provider_payload

    def save_prompt_payload(self, path: Path, payload: Mapping[str, Any]) -> PromptPayloadRef:
        sanitized_payload = sanitize_prompt_public_context(dict(payload))
        assert_public_only_prompt_payload(sanitized_payload)
        return publish_prompt_payload(path, sanitized_payload)


def _public_record(record: ExperimentRecord | Mapping[str, Any]) -> dict[str, Any]:
    return _experiment_record(record).ledger_public_record()


def _experiment_record(record: ExperimentRecord | Mapping[str, Any]) -> ExperimentRecord:
    if isinstance(record, ExperimentRecord):
        return record
    return ExperimentRecord.from_ledger_record(record)


def _validate_initial_anchor_public_evidence(
    value: Mapping[str, Any] | None,
    *,
    initial_current_best: CurrentBestEvalStats | None,
    acceptance_protocol: AcceptanceProtocol | None,
) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("initial anchor public evidence must be an object")
    if set(value) != {"anchor_kind", "public_result", "public_result_hash"}:
        raise ValueError("initial anchor public evidence fields mismatch")
    if initial_current_best is None or acceptance_protocol is None:
        raise ValueError(
            "initial anchor public evidence requires current-best and acceptance protocol"
        )

    anchor_kind = value.get("anchor_kind")
    public_result_value = value.get("public_result")
    if not isinstance(public_result_value, Mapping):
        raise ValueError("initial anchor public result must be an object")
    if set(public_result_value) != {"status", "public_score", "metrics"}:
        raise ValueError("initial anchor public result fields mismatch")
    public_result = PublicEvalResult(
        status=public_result_value.get("status"),
        public_score=public_result_value.get("public_score"),
        metrics=public_result_value.get("metrics"),
    )
    public_result_dict = public_result.to_public_dict()
    public_result_hash = require_sha256_digest(
        value.get("public_result_hash"),
        "initial anchor public result hash",
    )
    if canonical_json_hash(public_result_dict) != public_result_hash:
        raise ValueError("initial anchor public result hash mismatch")

    binding = initial_current_best.evaluation_binding()
    if anchor_kind == "baseline":
        if initial_current_best.anchor_kind != "baseline":
            raise ValueError("initial baseline evidence does not match current-best")
        restored = CurrentBestEvalStats.from_baseline_public_result_mapping(
            baseline_id=initial_current_best.anchor_id,
            baseline_manifest_hash=initial_current_best.subject_manifest_hash,
            checkpoint_hash=initial_current_best.checkpoint_hash,
            public_result=public_result_dict,
            acceptance_protocol=acceptance_protocol,
            expected_evaluation_binding=binding,
            capability_profile_hash=initial_current_best.capability_profile_hash,
        )
    elif anchor_kind == "reference_program":
        if initial_current_best.anchor_kind != "reference":
            raise ValueError("initial reference evidence does not match current-best")
        restored = CurrentBestEvalStats.from_reference_public_result_mapping(
            reference_id=initial_current_best.anchor_id,
            reference_manifest_hash=initial_current_best.subject_manifest_hash,
            program_source_checksum=initial_current_best.program_source_checksum,
            training_binding_hash=initial_current_best.training_binding_hash,
            capability_profile_hash=initial_current_best.capability_profile_hash,
            checkpoint_hash=initial_current_best.checkpoint_hash,
            public_result=public_result_dict,
            acceptance_protocol=acceptance_protocol,
            expected_evaluation_binding=binding,
        )
    else:
        raise ValueError("initial anchor public evidence kind is invalid")
    if restored.to_public_dict() != initial_current_best.to_public_dict():
        raise ValueError("initial anchor public evidence does not restore current-best")
    return {
        "anchor_kind": anchor_kind,
        "public_result": public_result_dict,
        "public_result_hash": public_result_hash,
    }


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


def _nonaccepted_authoring_tiebreak_key(
    task_prompt_adapter: TaskPromptAdapter | None,
) -> Callable[[Mapping[str, Any]], Sequence[float]] | None:
    if task_prompt_adapter is None:
        return None
    ranker = getattr(
        task_prompt_adapter,
        "nonaccepted_authoring_tiebreak_key",
        None,
    )
    if ranker is None:
        return None
    if not callable(ranker):
        raise TypeError(
            "task nonaccepted_authoring_tiebreak_key must be callable"
        )
    return ranker


def _string_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return ()
    return tuple(str(item) for item in value if isinstance(item, str) and item)
