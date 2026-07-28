"""Static candidate authoring attempts before runner execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from true_rl_agent.agent_core.authoring_response import (
    CandidateDraftResponse,
    SourceRequestResponse,
    next_authoring_response,
)
from true_rl_agent.agent_core.context import sanitize_prompt_public_context
from true_rl_agent.agent_core.provider import CandidateProvider
from true_rl_agent.agent_core.provider_errors import (
    ProviderCandidateError,
    build_provider_repair_feedback,
)
from true_rl_agent.agent_core.source_request_feedback import (
    build_source_request_round_feedback,
)
from true_rl_agent.agent_core.source_request_limits import MAX_SOURCE_REQUESTS_PER_ROUND
from true_rl_agent.agent_core.source_request_service import resolve_source_request
from true_rl_agent.candidates.authoring_checks import (
    assess_anti_degenerate_candidate,
    build_anti_degenerate_repair_feedback,
    build_component_smoke_repair_feedback,
    build_static_authoring_failure,
    run_component_smoke_check,
)
from true_rl_agent.candidates.schema import CandidatePackageRef
from true_rl_agent.candidates.verifier import CandidateVerifier, VerificationResult
from true_rl_agent.memory import (
    ExperimentRecord,
    PromptPayloadRef,
    PublicContextBuilder,
    PublicPromptBuild,
)
from true_rl_agent.prompting import assert_public_only_prompt_payload
from true_rl_agent.prompting.source_catalog import (
    LATEST_FAILED_SOURCE_ID,
    resolve_candidate_components_source_from_record,
)
from true_rl_agent.tasks.component_smoke import ComponentSmokeAdapter


@dataclass(frozen=True)
class CandidateAuthoringResult:
    """Result of provider-compatible static authoring for one round."""

    stop_reason: str
    prompt_payload_path: Path
    prompt_payload_ref: PromptPayloadRef
    round_public_context: Mapping[str, Any]
    attempt_records: list[dict[str, Any]]
    attempt_prompt_payload_paths: list[str]
    source_request_feedback: Mapping[str, Any] | None = None
    candidate: CandidatePackageRef | None = None
    provider_failure: Mapping[str, Any] | None = None
    static_failure: Mapping[str, Any] | None = None


class CandidateAuthoringLoop:
    """Run bounded provider attempts and static repair before training."""

    def __init__(
        self,
        *,
        context_builder: PublicContextBuilder,
        provider: CandidateProvider,
        verifier: CandidateVerifier,
        round_dir: Path,
        max_attempts: int,
        component_smoke_adapter: ComponentSmokeAdapter | None = None,
        max_source_requests_per_round: int = MAX_SOURCE_REQUESTS_PER_ROUND,
        runner_capability_readiness: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        if max_source_requests_per_round < 0:
            raise ValueError("max_source_requests_per_round must be non-negative")
        self.context_builder = context_builder
        self.provider = provider
        self.verifier = verifier
        self.round_dir = round_dir
        self.max_attempts = max_attempts
        self.component_smoke_adapter = component_smoke_adapter
        self.max_source_requests_per_round = max_source_requests_per_round
        self.runner_capability_readiness = runner_capability_readiness

    def run_round(
        self,
        *,
        round_id: str,
        records: Sequence[ExperimentRecord | Mapping[str, Any]],
    ) -> CandidateAuthoringResult:
        repair_feedback: dict[str, Any] | None = None
        latest_failed_candidate: CandidatePackageRef | None = None
        frame_kind: str | None = None
        source_payloads: list[dict[str, Any]] | None = None
        source_repository_payload: dict[str, Any] | None = None
        source_request_count = 0
        source_request_bytes_returned = 0
        attempt_records: list[dict[str, Any]] = []
        attempt_prompt_payload_paths: list[str] = []
        evaluated_history_sources = _evaluated_candidate_sources(records)

        selected_prompt_path: Path | None = None
        selected_prompt_payload_ref: PromptPayloadRef | None = None
        selected_round_public_context: Mapping[str, Any] | None = None
        selected_candidate: CandidatePackageRef | None = None
        last_provider_failure: Mapping[str, Any] | None = None

        for attempt in range(1, self.max_attempts + 1):
            prompt_build = build_authoring_prompt(
                context_builder=self.context_builder,
                records=records,
                repair_feedback=repair_feedback,
                latest_failed_candidate=latest_failed_candidate,
                frame_kind=frame_kind,
                source_payloads=source_payloads if frame_kind == "source_followup" else None,
                runner_capability_readiness=self.runner_capability_readiness,
            )
            prompt_payload = prompt_build.provider_payload
            prompt_frame_kind = prompt_payload["authoring_context_frame"]["frame_kind"]
            if prompt_frame_kind != "source_followup":
                source_repository_payload = prompt_payload
                source_payloads = None
            frame_kind = None
            prompt_path = self.round_dir / _prompt_payload_filename(attempt)
            prompt_payload_ref = self.context_builder.save_prompt_payload(prompt_path, prompt_payload)
            attempt_prompt_payload_paths.append(str(prompt_path))

            attempt_record = {
                "round_id": round_id,
                "attempt": attempt,
            }
            attempt_records.append(attempt_record)
            selected_prompt_path = prompt_path
            selected_prompt_payload_ref = prompt_payload_ref
            selected_round_public_context = prompt_build.round_public_context

            try:
                provider_response = next_authoring_response(self.provider, prompt_payload=prompt_payload)
            except ProviderCandidateError as exc:
                failure = exc.to_public_dict()
                attempt_record["provider_status"] = "fail"
                attempt_record["provider_failure"] = failure
                last_provider_failure = failure
                if attempt == self.max_attempts:
                    break
                repair_feedback = build_provider_repair_feedback(exc)
                latest_failed_candidate = None
                continue

            if isinstance(provider_response, SourceRequestResponse):
                resolver_payload = source_repository_payload or prompt_payload
                resolution = resolve_source_request(
                    prompt_payload=resolver_payload,
                    response=provider_response,
                    source_request_count=source_request_count,
                    source_request_bytes_returned=source_request_bytes_returned,
                    max_source_requests_per_round=self.max_source_requests_per_round,
                )
                source_request_count += 1
                source_request_bytes_returned += resolution.returned_bytes
                attempt_record["provider_response_mode"] = provider_response.response_mode
                attempt_record["provider_status"] = "source_request"
                attempt_record["source_request_result"] = resolution.attempt_record()
                if attempt == self.max_attempts:
                    error = ProviderCandidateError(
                        "source_request_denied",
                        "provider returned source_request on the final authoring attempt",
                        details=resolution.feedback,
                    )
                    failure = error.to_public_dict()
                    attempt_record["provider_failure"] = failure
                    last_provider_failure = failure
                    break
                repair_feedback = resolution.feedback
                source_payloads = _merge_source_payloads(source_payloads or [], resolution.source_payloads)
                frame_kind = "source_followup"
                continue
            assert isinstance(provider_response, CandidateDraftResponse)
            candidate = provider_response.candidate
            attempt_record["provider_status"] = "pass"
            attempt_record["provider_response_mode"] = provider_response.response_mode
            verification = self.verifier.verify(candidate)
            attempt_record.update(_candidate_attempt_public_record(candidate, verification))
            selected_candidate = candidate
            if verification.status != "pass":
                repair_feedback = build_verifier_repair_feedback(candidate=candidate, verification=verification)
                if attempt == self.max_attempts:
                    return _result(
                        stop_reason="static_failed",
                        prompt_payload_path=selected_prompt_path,
                        prompt_payload_ref=selected_prompt_payload_ref,
                        round_public_context=selected_round_public_context,
                        attempt_records=attempt_records,
                        attempt_prompt_payload_paths=attempt_prompt_payload_paths,
                        candidate=selected_candidate,
                        static_failure=build_static_authoring_failure(
                            candidate=candidate,
                            feedback=repair_feedback,
                        ),
                    )
                latest_failed_candidate = candidate
                continue

            smoke_result = run_component_smoke_check(candidate, self.component_smoke_adapter)
            attempt_record["component_smoke"] = smoke_result
            if smoke_result.get("status") == "fail":
                repair_feedback = build_component_smoke_repair_feedback(
                    candidate=candidate,
                    smoke_result=smoke_result,
                )
                if attempt == self.max_attempts:
                    return _result(
                        stop_reason="static_failed",
                        prompt_payload_path=selected_prompt_path,
                        prompt_payload_ref=selected_prompt_payload_ref,
                        round_public_context=selected_round_public_context,
                        attempt_records=attempt_records,
                        attempt_prompt_payload_paths=attempt_prompt_payload_paths,
                        candidate=selected_candidate,
                        static_failure=build_static_authoring_failure(
                            candidate=candidate,
                            feedback=repair_feedback,
                        ),
                    )
                latest_failed_candidate = candidate
                continue

            anti_degenerate = assess_anti_degenerate_candidate(
                candidate,
                prompt_payload,
                evaluated_history_sources=evaluated_history_sources,
            )
            attempt_record["anti_degenerate_assessment"] = anti_degenerate
            if anti_degenerate.get("blocking") is True:
                repair_feedback = build_anti_degenerate_repair_feedback(
                    candidate=candidate,
                    assessment=anti_degenerate,
                )
                if attempt == self.max_attempts:
                    return _result(
                        stop_reason="static_failed",
                        prompt_payload_path=selected_prompt_path,
                        prompt_payload_ref=selected_prompt_payload_ref,
                        round_public_context=selected_round_public_context,
                        attempt_records=attempt_records,
                        attempt_prompt_payload_paths=attempt_prompt_payload_paths,
                        candidate=selected_candidate,
                        static_failure=build_static_authoring_failure(
                            candidate=candidate,
                            feedback=repair_feedback,
                        ),
                    )
                latest_failed_candidate = candidate
                continue

            return _result(
                stop_reason="candidate_verified",
                prompt_payload_path=selected_prompt_path,
                prompt_payload_ref=selected_prompt_payload_ref,
                round_public_context=selected_round_public_context,
                attempt_records=attempt_records,
                attempt_prompt_payload_paths=attempt_prompt_payload_paths,
                candidate=selected_candidate,
            )

        return _result(
            stop_reason="provider_failed",
            prompt_payload_path=selected_prompt_path,
            prompt_payload_ref=selected_prompt_payload_ref,
            round_public_context=selected_round_public_context,
            attempt_records=attempt_records,
            attempt_prompt_payload_paths=attempt_prompt_payload_paths,
            provider_failure=last_provider_failure,
        )


def build_authoring_prompt(
    *,
    context_builder: PublicContextBuilder,
    records: Sequence[ExperimentRecord | Mapping[str, Any]],
    repair_feedback: dict[str, Any] | None,
    latest_failed_candidate: CandidatePackageRef | None,
    frame_kind: str | None = None,
    source_payloads: Sequence[Mapping[str, Any]] | None = None,
    runner_capability_readiness: Mapping[str, Mapping[str, Any]] | None = None,
) -> PublicPromptBuild:
    clean_repair_feedback = (
        _sanitize_repair_feedback(repair_feedback) if repair_feedback is not None else None
    )
    prompt_build = context_builder.build_prompt(
        records,
        latest_failed_candidate=latest_failed_candidate,
        candidate_repair_feedback=clean_repair_feedback,
        source_payloads=source_payloads,
        frame_kind=frame_kind,
        runner_capability_readiness=runner_capability_readiness,
    )
    assert_public_only_prompt_payload(prompt_build.provider_payload)
    return prompt_build


def build_authoring_prompt_payload(
    *,
    context_builder: PublicContextBuilder,
    records: Sequence[ExperimentRecord | Mapping[str, Any]],
    repair_feedback: dict[str, Any] | None,
    latest_failed_candidate: CandidatePackageRef | None,
    frame_kind: str | None = None,
    source_payloads: Sequence[Mapping[str, Any]] | None = None,
    runner_capability_readiness: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    return dict(
        build_authoring_prompt(
            context_builder=context_builder,
            records=records,
            repair_feedback=repair_feedback,
            latest_failed_candidate=latest_failed_candidate,
            frame_kind=frame_kind,
            source_payloads=source_payloads,
            runner_capability_readiness=runner_capability_readiness,
        ).provider_payload
    )


def _merge_source_payloads(
    existing: Sequence[Mapping[str, Any]],
    new_payloads: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {str(item.get("logical_id")): dict(item) for item in existing}
    for item in new_payloads:
        logical_id = item.get("logical_id")
        if not isinstance(logical_id, str):
            continue
        merged[logical_id] = dict(item)
    payloads = list(merged.values())
    assert_public_only_prompt_payload(payloads)
    return payloads


def build_verifier_repair_feedback(
    *,
    candidate: CandidatePackageRef,
    verification: VerificationResult,
) -> dict[str, Any]:
    feedback = {
        "schema_version": "true_rl_agent.provider_repair_feedback.v1",
        "previous_candidate_id": candidate.candidate_id,
        "previous_candidate_changed_components": sorted(candidate.change_set.changed_components),
        "previous_candidate_source_logical_id": LATEST_FAILED_SOURCE_ID,
        "previous_candidate_source_delivery": "see_source_catalog",
        "failure_category": "verifier_error",
        "verifier_status": verification.status,
        "verifier_reason": verification.reason,
        "verifier_details": dict(verification.details),
        "instructions": [
            "Fix only the verifier issues in the candidate training_code/components.py.",
            "Do not use import torch or other top-level imports beyond the allowed component import.",
            "Use only fields and tensor operations described in the prompt reward_inputs_contract.",
            "Keep runner, eval, seed, budget, checkpoint, curriculum, and acceptance rules unchanged.",
        ],
    }
    return _sanitize_repair_feedback(feedback)


def _candidate_attempt_public_record(
    candidate: CandidatePackageRef,
    verification: VerificationResult,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "candidate_changed_components": sorted(candidate.change_set.changed_components),
        "candidate_source_visibility": "private_non_prompt_artifact",
        "candidate_source_included_in_prompt": False,
        "verifier_status": verification.status,
        "verifier_reason": verification.reason,
    }


def _evaluated_candidate_sources(
    records: Sequence[ExperimentRecord | Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Resolve prompt-bounded sources for candidates that reached public eval."""

    sources: list[dict[str, Any]] = []
    for raw_record in records:
        record = (
            raw_record
            if isinstance(raw_record, ExperimentRecord)
            else ExperimentRecord.from_ledger_record(raw_record)
        )
        public_record = record.ledger_public_record()
        public_result = public_record.get("public_result")
        if not isinstance(public_result, Mapping) or not public_result:
            continue
        source = resolve_candidate_components_source_from_record(record)
        content = source.get("content")
        if source.get("status") != "inline" or not isinstance(content, str):
            continue
        candidate = public_record.get("candidate")
        candidate_id = (
            candidate.get("candidate_id")
            if isinstance(candidate, Mapping)
            else None
        )
        sources.append(
            {
                "round_id": record.round_id,
                "candidate_id": candidate_id,
                "content": content,
                "content_sha256": source.get("content_sha256"),
            }
        )
    return tuple(sources)


def _result(
    *,
    stop_reason: str,
    prompt_payload_path: Path | None,
    prompt_payload_ref: PromptPayloadRef | None,
    round_public_context: Mapping[str, Any] | None,
    attempt_records: list[dict[str, Any]],
    attempt_prompt_payload_paths: list[str],
    candidate: CandidatePackageRef | None = None,
    provider_failure: Mapping[str, Any] | None = None,
    static_failure: Mapping[str, Any] | None = None,
) -> CandidateAuthoringResult:
    assert prompt_payload_path is not None
    assert prompt_payload_ref is not None
    assert round_public_context is not None
    source_request_results = [
        attempt_record["source_request_result"]
        for attempt_record in attempt_records
        if isinstance(attempt_record.get("source_request_result"), Mapping)
    ]
    source_request_feedback = build_source_request_round_feedback(
        source_request_results
    )
    return CandidateAuthoringResult(
        stop_reason=stop_reason,
        prompt_payload_path=prompt_payload_path,
        prompt_payload_ref=prompt_payload_ref,
        round_public_context=round_public_context,
        attempt_records=attempt_records,
        attempt_prompt_payload_paths=attempt_prompt_payload_paths,
        source_request_feedback=source_request_feedback,
        candidate=candidate,
        provider_failure=provider_failure,
        static_failure=static_failure,
    )


def _prompt_payload_filename(attempt: int) -> str:
    if attempt == 1:
        return "prompt_payload.json"
    return f"prompt_payload_attempt_{attempt:03d}.json"


def _sanitize_repair_feedback(feedback: Mapping[str, Any]) -> dict[str, Any]:
    clean = sanitize_prompt_public_context(dict(feedback))
    assert_public_only_prompt_payload(clean)
    return clean
