"""Static candidate authoring attempts before runner execution."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from true_rl_agent.agent_core.authoring_response import (
    CandidateDraftResponse,
    SourceRequestResponse,
    next_authoring_response,
)
from true_rl_agent.agent_core.context import sanitize_prompt_public_context
from true_rl_agent.agent_core.generation_contract import require_generation_lane_intent
from true_rl_agent.agent_core.provider import CandidateProvider
from true_rl_agent.agent_core.provider_errors import (
    ProviderCandidateError,
    build_provider_repair_feedback,
)
from true_rl_agent.agent_core.source_request_limits import (
    MAX_SOURCE_IDS_PER_REQUEST,
    MAX_SOURCE_REQUEST_BYTES_PER_ROUND,
    MAX_SOURCE_REQUESTS_PER_ROUND,
)
from true_rl_agent.agent_core.source_request_service import resolve_source_request
from true_rl_agent.authoring_artifact_contract import require_authoring_attempt_count
from true_rl_agent.candidates.authoring_checks import (
    assess_anti_degenerate_candidate,
    build_anti_degenerate_repair_feedback,
    build_component_smoke_repair_feedback,
    build_static_authoring_failure,
    run_component_smoke_check,
)
from true_rl_agent.candidates.authoring_policy import (
    select_authoring_anchor_source,
    verifier_equivalence_exclusion_set_hash,
)
from true_rl_agent.candidates.schema import CandidateChangeSet, CandidatePackageRef
from true_rl_agent.candidates.sibling_exclusion import (
    SingleComponentDeltaComparisonUnavailable,
    SingleComponentSemanticExclusion,
    candidate_single_component_verifier_equivalence_key,
)
from true_rl_agent.candidates.verifier import CandidateVerifier, VerificationResult
from true_rl_agent.evaluation.binding import require_sha256_checksum
from true_rl_agent.memory import (
    ExperimentRecord,
    PromptPayloadRef,
    PublicContextBuilder,
    PublicPromptBuild,
)
from true_rl_agent.prompt_artifacts import (
    AUTHORING_EVIDENCE_KIND,
    AuthoringEvidenceRef,
    RoundAuthoringJournal,
    authoring_outcome_from_evidence,
    validate_authoring_budget_usage,
    verify_authoring_anchor_prompt,
    verify_authoring_evidence,
    verify_prompt_payload,
)
from true_rl_agent.prompting import assert_public_only_prompt_payload
from true_rl_agent.prompting.source_catalog import (
    LATEST_FAILED_SOURCE_ID,
    PROMPT_SOURCE_SECTION_KEYS,
    resolve_candidate_components_source_from_record,
    validate_prompt_source_sections,
)
from true_rl_agent.tasks.component_smoke import ComponentSmokeAdapter
from true_rl_agent.training_programs import resolve_candidate_program_identity


@dataclass(frozen=True)
class CandidateAuthoringResult:
    """Result of provider-compatible static authoring for one round."""

    stop_reason: str
    prompt_payload_ref: PromptPayloadRef
    round_public_context: Mapping[str, Any]
    attempt_records: list[dict[str, Any]]
    attempt_prompt_payload_refs: tuple[PromptPayloadRef, ...]
    authoring_evidence_ref: AuthoringEvidenceRef
    parent_program_source_checksum: str | None
    source_request_feedback: Mapping[str, Any] | None = None
    candidate: CandidatePackageRef | None = None
    provider_failure: Mapping[str, Any] | None = None
    static_failure: Mapping[str, Any] | None = None

    @property
    def prompt_payload_path(self) -> Path:
        return Path(self.prompt_payload_ref.path)

    @property
    def attempt_prompt_payload_paths(self) -> list[str]:
        return [ref.path for ref in self.attempt_prompt_payload_refs]


class AuthoringBudgetExhaustedError(RuntimeError):
    """Raised when durable attempt reservations consume a round budget."""


@dataclass(frozen=True)
class _AuthoringContinuation:
    round_id: str
    disposition: str
    repair_feedback: Mapping[str, Any] | None
    latest_failed_candidate: CandidatePackageRef | None
    frame_kind: str | None
    source_payloads: tuple[Mapping[str, Any], ...]
    source_repository_prompt_ref: PromptPayloadRef | None
    source_request_count: int
    source_request_bytes_returned: int
    selected_prompt_payload_ref: PromptPayloadRef
    selected_round_public_context: Mapping[str, Any]
    selected_parent_program_source_checksum: str | None
    selected_candidate: CandidatePackageRef | None
    last_provider_failure: Mapping[str, Any] | None
    stop_reason: str | None
    static_failure: Mapping[str, Any] | None

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": "true_rl_agent.authoring_continuation.v1",
            "round_id": self.round_id,
            "disposition": self.disposition,
            "repair_feedback": _optional_dict(self.repair_feedback),
            "latest_failed_candidate": _candidate_recovery_mapping(
                self.latest_failed_candidate
            ),
            "frame_kind": self.frame_kind,
            "source_payloads": [dict(item) for item in self.source_payloads],
            "source_repository_prompt_ref": (
                self.source_repository_prompt_ref.to_public_dict()
                if self.source_repository_prompt_ref is not None
                else None
            ),
            "source_request_count": self.source_request_count,
            "source_request_bytes_returned": self.source_request_bytes_returned,
            "selected_prompt_payload_ref": (
                self.selected_prompt_payload_ref.to_public_dict()
            ),
            "selected_round_public_context": dict(
                self.selected_round_public_context
            ),
            "selected_parent_program_source_checksum": (
                self.selected_parent_program_source_checksum
            ),
            "selected_candidate": _candidate_recovery_mapping(
                self.selected_candidate
            ),
            "last_provider_failure": _optional_dict(
                self.last_provider_failure
            ),
            "stop_reason": self.stop_reason,
            "static_failure": _optional_dict(self.static_failure),
        }


class CandidateAuthoringLoop:
    """Run bounded provider attempts and static repair before training."""

    def __init__(
        self,
        *,
        context_builder: PublicContextBuilder,
        provider: CandidateProvider,
        candidate_admission_policy: CandidateVerifier,
        round_dir: Path,
        max_attempts: int,
        component_smoke_adapter: ComponentSmokeAdapter | None = None,
        max_source_requests_per_round: int = MAX_SOURCE_REQUESTS_PER_ROUND,
        authoring_capability_readiness: Mapping[str, Mapping[str, Any]] | None = None,
        sibling_semantic_exclusion: SingleComponentSemanticExclusion | None = None,
        generation_lane_intent: Mapping[str, Any] | None = None,
        blocked_verifier_equivalence_keys: Sequence[str] = (),
    ) -> None:
        require_authoring_attempt_count(max_attempts)
        if max_source_requests_per_round < 0:
            raise ValueError("max_source_requests_per_round must be non-negative")
        self.context_builder = context_builder
        self.provider = provider
        self.candidate_admission_policy = candidate_admission_policy
        raw_verifier_prompt_rules = candidate_admission_policy.authoring_prompt_rules()
        if not isinstance(raw_verifier_prompt_rules, Mapping):
            raise TypeError("candidate verifier prompt rules must be an object")
        verifier_prompt_rules = dict(raw_verifier_prompt_rules)
        if generation_lane_intent is not None:
            verifier_prompt_rules["generation_lane_intent"] = (
                require_generation_lane_intent(generation_lane_intent)
            )
        if sibling_semantic_exclusion is not None:
            if not isinstance(
                sibling_semantic_exclusion,
                SingleComponentSemanticExclusion,
            ):
                raise TypeError(
                    "sibling_semantic_exclusion must be verifier-owned"
                )
            verifier_prompt_rules["single_component_semantic_exclusion"] = dict(
                sibling_semantic_exclusion.public_projection
            )
        self.candidate_verifier_prompt_rules = sanitize_prompt_public_context(
            verifier_prompt_rules
        )
        assert_public_only_prompt_payload(self.candidate_verifier_prompt_rules)
        self.round_dir = round_dir
        self.max_attempts = max_attempts
        self.component_smoke_adapter = component_smoke_adapter
        self.max_source_requests_per_round = max_source_requests_per_round
        self.authoring_capability_readiness = authoring_capability_readiness
        self.sibling_semantic_exclusion = sibling_semantic_exclusion
        self.generation_lane_intent = (
            require_generation_lane_intent(generation_lane_intent)
            if generation_lane_intent is not None
            else None
        )
        blocked_keys = tuple(blocked_verifier_equivalence_keys)
        blocked_set_hash = verifier_equivalence_exclusion_set_hash(blocked_keys)
        intent_set_hash = (
            self.generation_lane_intent.get(
                "verifier_equivalence_exclusion_set_hash"
            )
            if self.generation_lane_intent is not None
            else None
        )
        if intent_set_hash is not None and intent_set_hash != blocked_set_hash:
            raise ValueError(
                "generation lane intent verifier exclusion set does not match authoring gate"
            )
        if blocked_keys and intent_set_hash is None:
            raise ValueError(
                "historical verifier exclusions require a bound generation lane intent"
            )
        self.blocked_verifier_equivalence_keys = blocked_keys

    def run_round(
        self,
        *,
        round_id: str,
        records: Sequence[ExperimentRecord | Mapping[str, Any]],
    ) -> CandidateAuthoringResult:
        journal = RoundAuthoringJournal(self.round_dir, round_id=round_id)
        with journal.acquire_lease():
            budget_contract = journal.bind_budget(
                max_provider_attempts=self.max_attempts,
                max_source_requests_per_round=(
                    self.max_source_requests_per_round
                ),
                max_source_ids_per_request=MAX_SOURCE_IDS_PER_REQUEST,
                max_source_request_bytes_per_round=(
                    MAX_SOURCE_REQUEST_BYTES_PER_ROUND
                ),
            )
            return self._run_round_locked(
                round_id=round_id,
                records=records,
                journal=journal,
                budget_contract=budget_contract,
            )

    def _run_round_locked(
        self,
        *,
        round_id: str,
        records: Sequence[ExperimentRecord | Mapping[str, Any]],
        journal: RoundAuthoringJournal,
        budget_contract: Mapping[str, Any],
    ) -> CandidateAuthoringResult:
        snapshot = journal.snapshot()
        if snapshot.round_authoring_anchor is None:
            round_authoring_anchor_prompt = None
            round_authoring_parent_program_source_checksum = None
        else:
            (
                round_authoring_anchor_prompt,
                round_authoring_parent_program_source_checksum,
            ) = _round_authoring_anchor_values(
                snapshot.round_authoring_anchor
            )
        restored_source_repository_payload = (
            _source_repository_from_prompt_refs(snapshot.prompt_payload_refs)
        )
        continuation = _restore_authoring_continuation(
            snapshot.latest_continuation_state,
            expected_round_id=round_id,
        )
        attempt_records = [
            dict(item) for item in snapshot.completed_attempt_records
        ]
        attempt_prompt_payload_refs = list(snapshot.prompt_payload_refs)
        budget_usage = validate_authoring_budget_usage(
            attempt_records,
            budget_contract=budget_contract,
            expected_round_id=round_id,
            reserved_attempt_count=snapshot.reserved_attempt_count,
        )
        max_provider_attempts = int(budget_contract["max_provider_attempts"])
        max_source_requests = int(
            budget_contract["max_source_requests_per_round"]
        )
        max_source_ids = int(budget_contract["max_source_ids_per_request"])
        max_source_request_bytes = int(
            budget_contract["max_source_request_bytes_per_round"]
        )
        evaluated_history_sources = _evaluated_candidate_sources(records)

        if continuation is None:
            repair_feedback: dict[str, Any] | None = None
            latest_failed_candidate: CandidatePackageRef | None = None
            frame_kind: str | None = None
            source_payloads: list[dict[str, Any]] | None = None
            source_repository_prompt_ref: PromptPayloadRef | None = None
            source_repository_payload = restored_source_repository_payload
            source_request_count = 0
            source_request_bytes_returned = 0
            selected_prompt_payload_ref: PromptPayloadRef | None = None
            selected_round_public_context: Mapping[str, Any] | None = None
            selected_parent_program_source_checksum: str | None = None
            selected_candidate: CandidatePackageRef | None = None
            last_provider_failure: Mapping[str, Any] | None = None
        else:
            if (
                continuation.source_request_count
                != budget_usage["source_request_count"]
                or continuation.source_request_bytes_returned
                != budget_usage["source_request_bytes_returned"]
            ):
                raise ValueError(
                    "authoring continuation source usage differs from the journal"
                )
            repair_feedback = _optional_dict(continuation.repair_feedback)
            latest_failed_candidate = continuation.latest_failed_candidate
            frame_kind = continuation.frame_kind
            source_payloads = [dict(item) for item in continuation.source_payloads]
            source_repository_prompt_ref = (
                continuation.source_repository_prompt_ref
            )
            source_repository_payload = restored_source_repository_payload
            if (
                source_repository_payload is None
                and source_repository_prompt_ref is not None
            ):
                source_repository_payload = _source_repository_from_prompt_refs(
                    (source_repository_prompt_ref,),
                )
            source_request_count = continuation.source_request_count
            source_request_bytes_returned = (
                continuation.source_request_bytes_returned
            )
            selected_prompt_payload_ref = (
                continuation.selected_prompt_payload_ref
            )
            verify_prompt_payload(selected_prompt_payload_ref)
            selected_round_public_context = (
                continuation.selected_round_public_context
            )
            selected_parent_program_source_checksum = (
                continuation.selected_parent_program_source_checksum
            )
            selected_candidate = continuation.selected_candidate
            last_provider_failure = continuation.last_provider_failure
            if continuation.disposition == "terminal":
                assert continuation.stop_reason is not None
                return _result(
                    stop_reason=continuation.stop_reason,
                    prompt_payload_ref=selected_prompt_payload_ref,
                    round_public_context=selected_round_public_context,
                    attempt_records=attempt_records,
                    attempt_prompt_payload_refs=attempt_prompt_payload_refs,
                    journal=journal,
                    parent_program_source_checksum=(
                        selected_parent_program_source_checksum
                    ),
                    candidate=(
                        selected_candidate
                        if continuation.stop_reason != "provider_failed"
                        else None
                    ),
                    provider_failure=(
                        last_provider_failure
                        if continuation.stop_reason == "provider_failed"
                        else None
                    ),
                    static_failure=continuation.static_failure,
                )

        if snapshot.reserved_attempt_count >= max_provider_attempts:
            raise AuthoringBudgetExhaustedError(
                "authoring provider budget was consumed by durable attempts: "
                f"round={round_id} reserved={snapshot.reserved_attempt_count} "
                f"max={max_provider_attempts}"
            )

        for attempt in range(
            snapshot.reserved_attempt_count + 1,
            max_provider_attempts + 1,
        ):
            prompt_build = build_authoring_prompt(
                context_builder=self.context_builder,
                records=records,
                repair_feedback=repair_feedback,
                latest_failed_candidate=latest_failed_candidate,
                frame_kind=frame_kind,
                source_payloads=(
                    source_payloads if frame_kind == "source_followup" else None
                ),
                authoring_capability_readiness=(
                    self.authoring_capability_readiness
                ),
                candidate_verifier_prompt_rules=(
                    self.candidate_verifier_prompt_rules
                ),
            )
            prompt_payload = prompt_build.provider_payload
            attempt_handle = journal.begin_attempt(prompt_payload)
            if attempt_handle.sequence != attempt:
                raise RuntimeError("authoring journal sequence changed while leased")
            prompt_payload_ref = attempt_handle.prompt_payload_ref
            prompt_payload = verify_prompt_payload(prompt_payload_ref)
            if round_authoring_anchor_prompt is None:
                (
                    round_authoring_anchor_prompt,
                    round_authoring_parent_program_source_checksum,
                ) = _round_authoring_anchor_values(
                    journal.load_round_authoring_anchor()
                )
            prompt_frame_kind = prompt_payload["authoring_context_frame"][
                "frame_kind"
            ]
            source_repository_payload = _merge_source_repository_payload(
                source_repository_payload,
                prompt_payload,
            )
            if prompt_frame_kind != "source_followup":
                source_repository_prompt_ref = prompt_payload_ref
                source_payloads = None
            frame_kind = None
            attempt_prompt_payload_refs.append(prompt_payload_ref)

            if (
                selected_prompt_payload_ref is not None
                and selected_parent_program_source_checksum
                != prompt_build.parent_program_source_checksum
            ):
                raise ValueError("authoring attempts changed parent program source")
            if (
                prompt_build.parent_program_source_checksum
                != round_authoring_parent_program_source_checksum
            ):
                raise ValueError("authoring attempts changed round anchor parent")

            attempt_record = {
                "round_id": round_id,
                "attempt": attempt,
                "attempt_id": attempt_handle.attempt_id,
            }
            attempt_records.append(attempt_record)
            selected_prompt_payload_ref = prompt_payload_ref
            selected_round_public_context = prompt_build.round_public_context
            selected_parent_program_source_checksum = (
                prompt_build.parent_program_source_checksum
            )
            selected_candidate = None

            def complete_attempt(
                *,
                disposition: str,
                stop_reason: str | None = None,
                static_failure: Mapping[str, Any] | None = None,
            ) -> None:
                assert selected_prompt_payload_ref is not None
                assert selected_round_public_context is not None
                state = _AuthoringContinuation(
                    round_id=round_id,
                    disposition=disposition,
                    repair_feedback=repair_feedback,
                    latest_failed_candidate=latest_failed_candidate,
                    frame_kind=frame_kind,
                    source_payloads=tuple(source_payloads or ()),
                    source_repository_prompt_ref=(
                        source_repository_prompt_ref
                    ),
                    source_request_count=source_request_count,
                    source_request_bytes_returned=(
                        source_request_bytes_returned
                    ),
                    selected_prompt_payload_ref=(
                        selected_prompt_payload_ref
                    ),
                    selected_round_public_context=(
                        selected_round_public_context
                    ),
                    selected_parent_program_source_checksum=(
                        selected_parent_program_source_checksum
                    ),
                    selected_candidate=selected_candidate,
                    last_provider_failure=last_provider_failure,
                    stop_reason=stop_reason,
                    static_failure=static_failure,
                )
                journal.complete_attempt(
                    attempt_handle,
                    attempt_record,
                    continuation_state=state.to_mapping(),
                )

            try:
                provider_response = next_authoring_response(
                    self.provider,
                    prompt_payload=prompt_payload,
                )
            except ProviderCandidateError as exc:
                failure = exc.to_public_dict()
                attempt_record["provider_status"] = "fail"
                attempt_record["provider_failure"] = failure
                if exc.provider_usage:
                    attempt_record["provider_usage"] = dict(
                        exc.provider_usage
                    )
                if exc.provider_receipt:
                    attempt_record["provider_response_receipt"] = dict(
                        exc.provider_receipt
                    )
                last_provider_failure = failure
                retryable_transport_failure = (
                    exc.category == "runtime_error"
                    and exc.details.get("retryable") is True
                )
                terminal_transport_failure = (
                    exc.category == "runtime_error"
                    and exc.details.get("retryable") is False
                )
                if attempt == max_provider_attempts or terminal_transport_failure:
                    complete_attempt(
                        disposition="terminal",
                        stop_reason="provider_failed",
                    )
                    return _result(
                        stop_reason="provider_failed",
                        prompt_payload_ref=selected_prompt_payload_ref,
                        round_public_context=selected_round_public_context,
                        attempt_records=attempt_records,
                        attempt_prompt_payload_refs=(
                            attempt_prompt_payload_refs
                        ),
                        journal=journal,
                        parent_program_source_checksum=(
                            selected_parent_program_source_checksum
                        ),
                        provider_failure=last_provider_failure,
                    )
                repair_feedback = (
                    _transport_retry_feedback(exc)
                    if retryable_transport_failure
                    else build_provider_repair_feedback(exc)
                )
                complete_attempt(disposition="continue")
                continue

            if provider_response.provider_usage:
                attempt_record["provider_usage"] = dict(
                    provider_response.provider_usage
                )
            if provider_response.provider_receipt:
                attempt_record["provider_response_receipt"] = dict(
                    provider_response.provider_receipt
                )
            if isinstance(provider_response, SourceRequestResponse):
                resolver_payload = source_repository_payload or prompt_payload
                resolution = resolve_source_request(
                    prompt_payload=resolver_payload,
                    response=provider_response,
                    source_request_count=source_request_count,
                    source_request_bytes_returned=(
                        source_request_bytes_returned
                    ),
                    max_source_requests_per_round=max_source_requests,
                    max_source_ids_per_request=max_source_ids,
                    max_source_request_bytes_per_round=(
                        max_source_request_bytes
                    ),
                )
                source_request_count += 1
                source_request_bytes_returned += resolution.returned_bytes
                attempt_record["provider_response_mode"] = (
                    provider_response.response_mode
                )
                attempt_record["provider_status"] = "source_request"
                attempt_record["source_request_result"] = (
                    resolution.attempt_record()
                )
                if attempt == max_provider_attempts:
                    error = ProviderCandidateError(
                        "source_request_denied",
                        "provider returned source_request on the final authoring attempt",
                        details=resolution.feedback,
                    )
                    failure = error.to_public_dict()
                    attempt_record["provider_failure"] = failure
                    last_provider_failure = failure
                    complete_attempt(
                        disposition="terminal",
                        stop_reason="provider_failed",
                    )
                    return _result(
                        stop_reason="provider_failed",
                        prompt_payload_ref=selected_prompt_payload_ref,
                        round_public_context=selected_round_public_context,
                        attempt_records=attempt_records,
                        attempt_prompt_payload_refs=(
                            attempt_prompt_payload_refs
                        ),
                        journal=journal,
                        parent_program_source_checksum=(
                            selected_parent_program_source_checksum
                        ),
                        provider_failure=last_provider_failure,
                    )
                repair_feedback = resolution.feedback
                source_payloads = _merge_source_payloads(
                    source_payloads or [],
                    resolution.source_payloads,
                )
                frame_kind = "source_followup"
                complete_attempt(disposition="continue")
                continue

            assert isinstance(provider_response, CandidateDraftResponse)
            candidate = provider_response.candidate
            attempt_record["provider_status"] = "pass"
            attempt_record["provider_response_mode"] = (
                provider_response.response_mode
            )
            verification = self.candidate_admission_policy.verify(candidate)
            if verification.status == "pass":
                if round_authoring_anchor_prompt is None:
                    raise RuntimeError("round authoring anchor was not frozen")
                parent_source, _, anchor_role = select_authoring_anchor_source(
                    round_authoring_anchor_prompt
                )
                verification = (
                    self.candidate_admission_policy.verify_authoring_delta(
                        candidate=candidate,
                        verification=verification,
                        parent_source=(
                            parent_source
                            if anchor_role == "current_best"
                            else None
                        ),
                    )
                )
                if not isinstance(verification, VerificationResult):
                    raise TypeError(
                        "candidate authoring-delta verification result is invalid"
                    )
            attempt_record.update(
                _candidate_attempt_public_record(candidate, verification)
            )
            selected_candidate = candidate
            if verification.status != "pass":
                repair_feedback = build_verifier_repair_feedback(
                    candidate=candidate,
                    verification=verification,
                    verifier_prompt_rules=self.candidate_verifier_prompt_rules,
                )
                latest_failed_candidate = candidate
                static_failure = (
                    build_static_authoring_failure(
                        candidate=candidate,
                        feedback=repair_feedback,
                    )
                    if attempt == max_provider_attempts
                    else None
                )
                complete_attempt(
                    disposition=(
                        "terminal" if attempt == max_provider_attempts else "continue"
                    ),
                    stop_reason=(
                        "static_failed" if attempt == max_provider_attempts else None
                    ),
                    static_failure=static_failure,
                )
                if attempt == max_provider_attempts:
                    return _result(
                        stop_reason="static_failed",
                        prompt_payload_ref=selected_prompt_payload_ref,
                        round_public_context=selected_round_public_context,
                        attempt_records=attempt_records,
                        attempt_prompt_payload_refs=(
                            attempt_prompt_payload_refs
                        ),
                        journal=journal,
                        parent_program_source_checksum=(
                            selected_parent_program_source_checksum
                        ),
                        candidate=selected_candidate,
                        static_failure=static_failure,
                    )
                continue

            smoke_result = run_component_smoke_check(
                candidate,
                self.component_smoke_adapter,
            )
            attempt_record["component_smoke"] = smoke_result
            advisory_repair_already_offered = any(
                isinstance(record.get("component_smoke"), Mapping)
                and record["component_smoke"].get("status") == "advisory"
                for record in attempt_records[:-1]
            )
            if (
                smoke_result.get("status") == "advisory"
                and attempt < max_provider_attempts
                and not advisory_repair_already_offered
            ):
                # A synthetic fixture can reveal an obviously inert declared
                # term but cannot prove inactivity over the training state
                # distribution. Offer one normal repair opportunity; if the
                # next attempt remains advisory, runtime evidence retains the
                # authority and the candidate is allowed to train.
                repair_feedback = build_component_smoke_repair_feedback(
                    candidate=candidate,
                    smoke_result=smoke_result,
                )
                latest_failed_candidate = candidate
                complete_attempt(disposition="continue")
                continue
            if smoke_result.get("status") == "fail":
                repair_feedback = build_component_smoke_repair_feedback(
                    candidate=candidate,
                    smoke_result=smoke_result,
                )
                latest_failed_candidate = candidate
                static_failure = (
                    build_static_authoring_failure(
                        candidate=candidate,
                        feedback=repair_feedback,
                    )
                    if attempt == max_provider_attempts
                    else None
                )
                complete_attempt(
                    disposition=(
                        "terminal" if attempt == max_provider_attempts else "continue"
                    ),
                    stop_reason=(
                        "static_failed" if attempt == max_provider_attempts else None
                    ),
                    static_failure=static_failure,
                )
                if attempt == max_provider_attempts:
                    return _result(
                        stop_reason="static_failed",
                        prompt_payload_ref=selected_prompt_payload_ref,
                        round_public_context=selected_round_public_context,
                        attempt_records=attempt_records,
                        attempt_prompt_payload_refs=(
                            attempt_prompt_payload_refs
                        ),
                        journal=journal,
                        parent_program_source_checksum=(
                            selected_parent_program_source_checksum
                        ),
                        candidate=selected_candidate,
                        static_failure=static_failure,
                    )
                continue

            if round_authoring_anchor_prompt is None:
                raise RuntimeError("round authoring anchor was not frozen")
            anti_degenerate = assess_anti_degenerate_candidate(
                candidate,
                round_authoring_anchor_prompt,
                evaluated_history_sources=evaluated_history_sources,
            )
            if (
                round_authoring_parent_program_source_checksum is not None
                and anti_degenerate.get("reason")
                == "authoring_anchor_source_not_inline"
            ):
                raise ValueError(
                    "round authoring prompt has a parent but no verifiable source anchor"
                )
            attempt_record["anti_degenerate_assessment"] = anti_degenerate
            blocking_assessment = (
                anti_degenerate
                if anti_degenerate.get("blocking") is True
                else None
            )
            blocking_feedback = None
            if blocking_assessment is not None:
                blocking_feedback = build_anti_degenerate_repair_feedback(
                    candidate=candidate,
                    assessment=blocking_assessment,
                )
            elif self.sibling_semantic_exclusion is not None:
                try:
                    equivalence = self.sibling_semantic_exclusion.compare_candidate(
                        candidate=candidate,
                        candidate_prompt_payload=round_authoring_anchor_prompt,
                    )
                except SingleComponentDeltaComparisonUnavailable:
                    sibling_assessment = {
                        "schema_version": (
                            "true_rl_agent.sibling_semantic_exclusion_assessment.v1"
                        ),
                        "status": "fail",
                        "blocking": True,
                        "reason": "single_component_delta_comparison_unavailable",
                        "equivalence": None,
                        "exclusion_hash": (
                            self.sibling_semantic_exclusion.exclusion_hash
                        ),
                    }
                else:
                    sibling_assessment = {
                        "schema_version": (
                            "true_rl_agent.sibling_semantic_exclusion_assessment.v1"
                        ),
                        "status": (
                            "fail" if equivalence is not None else "pass"
                        ),
                        "blocking": equivalence is not None,
                        "reason": (
                            "sibling_semantic_delta_not_distinct"
                            if equivalence is not None
                            else "sibling_semantic_delta_distinct"
                        ),
                        "equivalence": equivalence,
                        "exclusion_hash": (
                            self.sibling_semantic_exclusion.exclusion_hash
                        ),
                    }
                attempt_record["sibling_semantic_exclusion_assessment"] = (
                    sibling_assessment
                )
                if sibling_assessment["blocking"] is True:
                    blocking_assessment = sibling_assessment
                    blocking_feedback = _build_sibling_exclusion_repair_feedback(
                        candidate=candidate,
                        assessment=sibling_assessment,
                        exclusion=self.sibling_semantic_exclusion,
                    )

            if (
                blocking_assessment is None
                and self.blocked_verifier_equivalence_keys
            ):
                try:
                    equivalence_key = (
                        candidate_single_component_verifier_equivalence_key(
                            candidate=candidate,
                            prompt_payload=round_authoring_anchor_prompt,
                        )
                    )
                except SingleComponentDeltaComparisonUnavailable:
                    historical_assessment = {
                        "schema_version": (
                            "true_rl_agent.historical_verifier_exclusion_assessment.v1"
                        ),
                        "status": "fail",
                        "blocking": True,
                        "reason": "single_component_delta_comparison_unavailable",
                        "verifier_equivalence_key": None,
                        "exclusion_set_hash": (
                            verifier_equivalence_exclusion_set_hash(
                                self.blocked_verifier_equivalence_keys
                            )
                        ),
                    }
                else:
                    repeated = (
                        equivalence_key in self.blocked_verifier_equivalence_keys
                    )
                    historical_assessment = {
                        "schema_version": (
                            "true_rl_agent.historical_verifier_exclusion_assessment.v1"
                        ),
                        "status": "fail" if repeated else "pass",
                        "blocking": repeated,
                        "reason": (
                            "historical_verifier_equivalent_repeat"
                            if repeated
                            else "historical_verifier_equivalent_distinct"
                        ),
                        "verifier_equivalence_key": equivalence_key,
                        "exclusion_set_hash": (
                            verifier_equivalence_exclusion_set_hash(
                                self.blocked_verifier_equivalence_keys
                            )
                        ),
                    }
                attempt_record[
                    "historical_verifier_exclusion_assessment"
                ] = historical_assessment
                if historical_assessment["blocking"] is True:
                    blocking_assessment = historical_assessment
                    blocking_feedback = (
                        _build_historical_verifier_exclusion_repair_feedback(
                            candidate=candidate,
                            assessment=historical_assessment,
                        )
                    )

            if blocking_assessment is not None:
                assert blocking_feedback is not None
                repair_feedback = blocking_feedback
                latest_failed_candidate = candidate
                static_failure = (
                    build_static_authoring_failure(
                        candidate=candidate,
                        feedback=repair_feedback,
                    )
                    if attempt == max_provider_attempts
                    else None
                )
                complete_attempt(
                    disposition=(
                        "terminal" if attempt == max_provider_attempts else "continue"
                    ),
                    stop_reason=(
                        "static_failed" if attempt == max_provider_attempts else None
                    ),
                    static_failure=static_failure,
                )
                if attempt == max_provider_attempts:
                    return _result(
                        stop_reason="static_failed",
                        prompt_payload_ref=selected_prompt_payload_ref,
                        round_public_context=selected_round_public_context,
                        attempt_records=attempt_records,
                        attempt_prompt_payload_refs=(
                            attempt_prompt_payload_refs
                        ),
                        journal=journal,
                        parent_program_source_checksum=(
                            selected_parent_program_source_checksum
                        ),
                        candidate=selected_candidate,
                        static_failure=static_failure,
                    )
                continue

            complete_attempt(
                disposition="terminal",
                stop_reason="candidate_verified",
            )
            return _result(
                stop_reason="candidate_verified",
                prompt_payload_ref=selected_prompt_payload_ref,
                round_public_context=selected_round_public_context,
                attempt_records=attempt_records,
                attempt_prompt_payload_refs=attempt_prompt_payload_refs,
                journal=journal,
                parent_program_source_checksum=(
                    selected_parent_program_source_checksum
                ),
                candidate=selected_candidate,
            )

        raise RuntimeError("authoring loop exhausted without a terminal result")


def build_authoring_prompt(
    *,
    context_builder: PublicContextBuilder,
    records: Sequence[ExperimentRecord | Mapping[str, Any]],
    repair_feedback: dict[str, Any] | None,
    latest_failed_candidate: CandidatePackageRef | None,
    frame_kind: str | None = None,
    source_payloads: Sequence[Mapping[str, Any]] | None = None,
    authoring_capability_readiness: Mapping[str, Mapping[str, Any]] | None = None,
    candidate_verifier_prompt_rules: Mapping[str, Any] | None = None,
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
        authoring_capability_readiness=authoring_capability_readiness,
        candidate_verifier_prompt_rules=candidate_verifier_prompt_rules,
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
    authoring_capability_readiness: Mapping[str, Mapping[str, Any]] | None = None,
    candidate_verifier_prompt_rules: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return dict(
        build_authoring_prompt(
            context_builder=context_builder,
            records=records,
            repair_feedback=repair_feedback,
            latest_failed_candidate=latest_failed_candidate,
            frame_kind=frame_kind,
            source_payloads=source_payloads,
            authoring_capability_readiness=authoring_capability_readiness,
            candidate_verifier_prompt_rules=candidate_verifier_prompt_rules,
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


def _merge_source_repository_payload(
    existing: Mapping[str, Any] | None,
    prompt_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive the round source repository from provider-visible prompt frames."""

    previous_payloads = (
        existing.get("source_payloads", []) if existing is not None else []
    )
    current_payloads = prompt_payload.get("source_payloads")
    if not isinstance(previous_payloads, list) or not isinstance(
        current_payloads, list
    ):
        raise TypeError("authoring source_payloads must be lists")
    current_catalog = prompt_payload.get("source_catalog")
    if not isinstance(current_catalog, Mapping):
        raise TypeError("authoring source_catalog must be an object")
    raw_catalog_items = current_catalog.get("items")
    if not isinstance(raw_catalog_items, list):
        raise TypeError("authoring source_catalog.items must be a list")
    current_catalog_items = {
        item.get("logical_id"): item
        for item in raw_catalog_items
        if isinstance(item, Mapping)
        and isinstance(item.get("logical_id"), str)
    }
    retained_previous_payloads = [
        item
        for item in previous_payloads
        if _source_payload_matches_current_catalog(
            item,
            current_catalog_items,
        )
    ]
    repository = {
        key: prompt_payload.get(key) for key in PROMPT_SOURCE_SECTION_KEYS
    }
    repository["source_payloads"] = _merge_source_payloads(
        retained_previous_payloads,
        current_payloads,
    )
    return validate_prompt_source_sections(
        repository,
        require_inline_payloads=False,
    )


def _source_payload_matches_current_catalog(
    payload: Any,
    catalog_items: Mapping[str, Mapping[str, Any]],
) -> bool:
    """Keep only the exact current version of a compacted logical source slot."""

    if not isinstance(payload, Mapping):
        return False
    logical_id = payload.get("logical_id")
    if not isinstance(logical_id, str):
        return False
    catalog_item = catalog_items.get(logical_id)
    return bool(
        isinstance(catalog_item, Mapping)
        and catalog_item.get("status") == "inline"
        and catalog_item.get("inline_by_default") is True
        and catalog_item.get("content_sha256")
        == payload.get("content_sha256")
        and catalog_item.get("bytes") == payload.get("bytes")
    )


def _source_repository_from_prompt_refs(
    prompt_refs: Sequence[PromptPayloadRef],
) -> dict[str, Any] | None:
    payloads: list[dict[str, Any]] = []
    for prompt_ref in prompt_refs:
        prompt_payload = verify_prompt_payload(prompt_ref)
        prompt_source_payloads = prompt_payload.get("source_payloads")
        if not isinstance(prompt_source_payloads, list):
            raise TypeError("authoring source_payloads must be lists")
        payloads = _merge_source_payloads(
            payloads,
            prompt_source_payloads,
        )
    if not prompt_refs:
        return None
    # Historical prompt frames remain immutable evidence under the source schema
    # that authored them.  Only carry their already-public payload bytes forward;
    # the next current prompt supplies and validates the authoritative catalog and
    # lineage before source requests can be resolved.
    return {"source_payloads": payloads}


def _prompt_parent_program_source_checksum(
    prompt_payload: Mapping[str, Any],
) -> str | None:
    lineage = prompt_payload.get("candidate_lineage")
    if not isinstance(lineage, Mapping):
        raise TypeError("round authoring prompt has no candidate lineage")
    checksum = lineage.get("parent_program_source_checksum")
    if checksum is None:
        return None
    require_sha256_checksum(
        checksum,
        "round authoring parent program source checksum",
    )
    return str(checksum)


def _round_authoring_anchor_values(
    anchor: Mapping[str, Any],
) -> tuple[dict[str, Any], str | None]:
    prompt_ref = PromptPayloadRef.from_mapping(anchor.get("prompt_payload"))
    prompt_payload = verify_prompt_payload(prompt_ref)
    parent_program_source_checksum = _prompt_parent_program_source_checksum(
        prompt_payload
    )
    if anchor.get("parent_program_source_checksum") != (
        parent_program_source_checksum
    ):
        raise ValueError("round authoring anchor parent projection mismatch")
    return prompt_payload, parent_program_source_checksum


def _restore_authoring_continuation(
    value: Mapping[str, Any] | None,
    *,
    expected_round_id: str,
) -> _AuthoringContinuation | None:
    if value is None:
        return None
    expected_fields = {
        "schema_version",
        "round_id",
        "disposition",
        "repair_feedback",
        "latest_failed_candidate",
        "frame_kind",
        "source_payloads",
        "source_repository_prompt_ref",
        "source_request_count",
        "source_request_bytes_returned",
        "selected_prompt_payload_ref",
        "selected_round_public_context",
        "selected_parent_program_source_checksum",
        "selected_candidate",
        "last_provider_failure",
        "stop_reason",
        "static_failure",
    }
    if set(value) != expected_fields:
        raise ValueError("authoring continuation fields mismatch")
    if (
        value.get("schema_version")
        != "true_rl_agent.authoring_continuation.v1"
        or value.get("round_id") != expected_round_id
    ):
        raise ValueError("authoring continuation identity mismatch")
    disposition = value.get("disposition")
    if disposition not in {"continue", "terminal"}:
        raise ValueError("authoring continuation disposition is invalid")
    frame_kind = value.get("frame_kind")
    if frame_kind is not None and not isinstance(frame_kind, str):
        raise ValueError("authoring continuation frame_kind is invalid")
    source_request_count = _nonnegative_int(
        value.get("source_request_count"),
        "source_request_count",
    )
    source_request_bytes_returned = _nonnegative_int(
        value.get("source_request_bytes_returned"),
        "source_request_bytes_returned",
    )
    raw_payloads = value.get("source_payloads")
    if not isinstance(raw_payloads, list) or any(
        not isinstance(item, Mapping) for item in raw_payloads
    ):
        raise ValueError("authoring continuation source_payloads are invalid")
    source_payloads = tuple(dict(item) for item in raw_payloads)
    repair_feedback = _optional_mapping(
        value.get("repair_feedback"),
        "repair_feedback",
    )
    selected_context = _optional_mapping(
        value.get("selected_round_public_context"),
        "selected_round_public_context",
        required=True,
    )
    last_provider_failure = _optional_mapping(
        value.get("last_provider_failure"),
        "last_provider_failure",
    )
    static_failure = _optional_mapping(
        value.get("static_failure"),
        "static_failure",
    )
    for public_value in (
        repair_feedback,
        list(source_payloads),
        selected_context,
        last_provider_failure,
        static_failure,
    ):
        if public_value is not None:
            assert_public_only_prompt_payload(public_value)
    selected_prompt = PromptPayloadRef.from_mapping(
        value.get("selected_prompt_payload_ref")
    )
    raw_repository_ref = value.get("source_repository_prompt_ref")
    repository_ref = (
        PromptPayloadRef.from_mapping(raw_repository_ref)
        if raw_repository_ref is not None
        else None
    )
    parent_checksum = value.get("selected_parent_program_source_checksum")
    if parent_checksum is not None and not isinstance(parent_checksum, str):
        raise ValueError("authoring continuation parent checksum is invalid")
    stop_reason = value.get("stop_reason")
    if stop_reason is not None and not isinstance(stop_reason, str):
        raise ValueError("authoring continuation stop_reason is invalid")
    if disposition == "terminal" and stop_reason not in {
        "candidate_verified",
        "provider_failed",
        "static_failed",
    }:
        raise ValueError("terminal authoring continuation has no valid stop reason")
    if disposition == "continue" and (
        stop_reason is not None or static_failure is not None
    ):
        raise ValueError("continuing authoring state contains terminal fields")
    if stop_reason != "static_failed" and static_failure is not None:
        raise ValueError("authoring continuation has an unexpected static failure")
    return _AuthoringContinuation(
        round_id=expected_round_id,
        disposition=str(disposition),
        repair_feedback=repair_feedback,
        latest_failed_candidate=_candidate_from_recovery_mapping(
            value.get("latest_failed_candidate")
        ),
        frame_kind=frame_kind,
        source_payloads=source_payloads,
        source_repository_prompt_ref=repository_ref,
        source_request_count=source_request_count,
        source_request_bytes_returned=source_request_bytes_returned,
        selected_prompt_payload_ref=selected_prompt,
        selected_round_public_context=selected_context,
        selected_parent_program_source_checksum=parent_checksum,
        selected_candidate=_candidate_from_recovery_mapping(
            value.get("selected_candidate")
        ),
        last_provider_failure=last_provider_failure,
        stop_reason=stop_reason,
        static_failure=static_failure,
    )


def _candidate_recovery_mapping(
    candidate: CandidatePackageRef | None,
) -> dict[str, Any] | None:
    if candidate is None:
        return None
    return {
        "schema_version": "true_rl_agent.authoring_candidate_recovery.v1",
        "candidate_id": candidate.candidate_id,
        "path": candidate.path,
        "changed_components": sorted(candidate.change_set.changed_components),
        "metadata": dict(candidate.metadata),
    }


def _candidate_from_recovery_mapping(value: Any) -> CandidatePackageRef | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "candidate_id",
        "path",
        "changed_components",
        "metadata",
    }:
        raise ValueError("authoring candidate recovery fields mismatch")
    candidate_id = value.get("candidate_id")
    path = value.get("path")
    changed_components = value.get("changed_components")
    metadata = value.get("metadata")
    if (
        value.get("schema_version")
        != "true_rl_agent.authoring_candidate_recovery.v1"
        or not isinstance(candidate_id, str)
        or not candidate_id
        or not isinstance(path, str)
        or not path
        or not isinstance(changed_components, list)
        or any(not isinstance(item, str) for item in changed_components)
        or not isinstance(metadata, Mapping)
    ):
        raise ValueError("authoring candidate recovery identity is invalid")
    return CandidatePackageRef(
        candidate_id=candidate_id,
        path=path,
        change_set=CandidateChangeSet(frozenset(changed_components)),
        metadata=dict(metadata),
    )


def _optional_mapping(
    value: Any,
    field_name: str,
    *,
    required: bool = False,
) -> dict[str, Any] | None:
    if value is None and not required:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"authoring continuation {field_name} is invalid")
    return dict(value)


def _optional_dict(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    return dict(value) if value is not None else None


def _nonnegative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"authoring continuation {field_name} is invalid")
    return value


def build_verifier_repair_feedback(
    *,
    candidate: CandidatePackageRef,
    verification: VerificationResult,
    verifier_prompt_rules: Mapping[str, Any],
) -> dict[str, Any]:
    issues = _verification_issues(verification)
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
        "verifier_policy_ref": _verifier_policy_ref(verifier_prompt_rules),
        "instructions": _verifier_repair_instructions(
            issues=issues,
            verification_reason=verification.reason,
            verifier_prompt_rules=verifier_prompt_rules,
        ),
    }
    return _sanitize_repair_feedback(feedback)


def _build_sibling_exclusion_repair_feedback(
    *,
    candidate: CandidatePackageRef,
    assessment: Mapping[str, Any],
    exclusion: SingleComponentSemanticExclusion,
) -> dict[str, Any]:
    return _sanitize_repair_feedback(
        {
            "schema_version": "true_rl_agent.provider_repair_feedback.v1",
            "previous_candidate_id": candidate.candidate_id,
            "previous_candidate_changed_components": sorted(
                candidate.change_set.changed_components
            ),
            "previous_candidate_source_logical_id": LATEST_FAILED_SOURCE_ID,
            "previous_candidate_source_delivery": "see_source_catalog",
            "failure_category": "sibling_semantic_exclusion_violation",
            "reason": assessment.get("reason"),
            "equivalence": assessment.get("equivalence"),
            "single_component_semantic_exclusion": dict(
                exclusion.public_projection
            ),
            "instructions": [
                "Replace the proposed single component mechanism with one whose normalized callable, input dependencies, and clip structure are substantively different from the excluded sibling.",
                "Changing only a component/function name or numeric parameter is not a distinct sibling mechanism.",
                "If the prior draft produced a compound or unclassifiable delta, return exactly one attributable component delta.",
            ],
        }
    )


def _build_historical_verifier_exclusion_repair_feedback(
    *,
    candidate: CandidatePackageRef,
    assessment: Mapping[str, Any],
) -> dict[str, Any]:
    return _sanitize_repair_feedback(
        {
            "schema_version": "true_rl_agent.provider_repair_feedback.v1",
            "previous_candidate_id": candidate.candidate_id,
            "previous_candidate_changed_components": sorted(
                candidate.change_set.changed_components
            ),
            "previous_candidate_source_logical_id": LATEST_FAILED_SOURCE_ID,
            "previous_candidate_source_delivery": "see_source_catalog",
            "failure_category": "historical_verifier_equivalent_repeat",
            "reason": assessment.get("reason"),
            "verifier_equivalence_key": assessment.get(
                "verifier_equivalence_key"
            ),
            "exclusion_set_hash": assessment.get("exclusion_set_hash"),
            "instructions": [
                "Replace the proposal with one structurally different single-component mechanism.",
                "Renaming a component/function or changing only numeric parameters does not create new evidence.",
                "Tensor indices, slices, reduction axes, input dependencies, callable structure, and clip structure remain semantic identity and may be changed when scientifically justified.",
            ],
        }
    )


_COMPONENT_NAME_ISSUES = frozenset(
    {
        "invalid_training_components_name",
        "invalid_reward_term_name",
        "invalid_diagnostic_term_name",
        "invalid_schedule_term_name",
    }
)


def _verification_issues(verification: VerificationResult) -> tuple[str, ...]:
    raw_issues = verification.details.get("issues")
    if not isinstance(raw_issues, Sequence) or isinstance(raw_issues, str):
        return ()
    return tuple(
        issue
        for issue in raw_issues
        if isinstance(issue, str) and issue
    )


def _verifier_policy_ref(
    verifier_prompt_rules: Mapping[str, Any],
) -> dict[str, str]:
    policy_id = verifier_prompt_rules.get("verification_policy_id")
    policy_hash = verifier_prompt_rules.get("verification_policy_hash")
    if not isinstance(policy_id, str) or not policy_id:
        raise ValueError("verifier prompt rules have no verification_policy_id")
    if not isinstance(policy_hash, str) or not policy_hash:
        raise ValueError("verifier prompt rules have no verification_policy_hash")
    return {
        "verification_policy_id": policy_id,
        "verification_policy_hash": policy_hash,
    }


def _verifier_repair_instructions(
    *,
    issues: Sequence[str],
    verification_reason: str,
    verifier_prompt_rules: Mapping[str, Any],
) -> list[str]:
    issue_summary = "; ".join(issues) if issues else verification_reason
    instructions = [
        (
            "Fix only the current verifier failure in training_code/components.py "
            f"and preserve the candidate mechanism otherwise: {issue_summary}."
        )
    ]
    if any(issue in _COMPONENT_NAME_ISSUES for issue in issues):
        instructions.append(
            _component_name_repair_instruction(verifier_prompt_rules)
        )
    if any(_is_import_issue(issue) for issue in issues):
        instructions.extend(
            _import_repair_instructions(verifier_prompt_rules)
        )
    return instructions


def _component_name_repair_instruction(
    verifier_prompt_rules: Mapping[str, Any],
) -> str:
    callable_contract = verifier_prompt_rules.get("component_callable_contract")
    if not isinstance(callable_contract, Mapping):
        raise TypeError("verifier prompt rules have no component_callable_contract")
    name_contract = callable_contract.get("component_name_contract")
    if not isinstance(name_contract, Mapping):
        raise TypeError("verifier prompt rules have no component_name_contract")
    max_chars = _positive_contract_int(name_contract.get("max_chars"), "max_chars")
    max_utf8_bytes = _positive_contract_int(
        name_contract.get("max_utf8_bytes"),
        "max_utf8_bytes",
    )
    return (
        "For each invalid component name, use a canonical name of at most "
        f"{max_chars} characters and at most {max_utf8_bytes} UTF-8 bytes; "
        "satisfy the same prompt-visible component_name_contract."
    )


def _positive_contract_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"component_name_contract {field_name} is invalid")
    return value


def _is_import_issue(issue: str) -> bool:
    return issue.startswith("forbidden_import") or issue.endswith(":Import")


def _import_repair_instructions(
    verifier_prompt_rules: Mapping[str, Any],
) -> list[str]:
    allowed_imports = verifier_prompt_rules.get("allowed_imports")
    if not isinstance(allowed_imports, Mapping):
        raise TypeError("verifier prompt rules have no allowed_imports")
    import_statements: list[str] = []
    for module, raw_names in sorted(allowed_imports.items()):
        if not isinstance(module, str) or not module:
            raise ValueError("verifier prompt rules contain an invalid import module")
        if not isinstance(raw_names, Sequence) or isinstance(raw_names, str):
            raise TypeError("verifier prompt rules contain invalid allowed imports")
        names = sorted(
            name
            for name in raw_names
            if isinstance(name, str) and name
        )
        if not names or len(names) != len(raw_names):
            raise ValueError("verifier prompt rules contain invalid allowed imports")
        import_statements.append(f"from {module} import {', '.join(names)}")
    allowed_text = "; ".join(import_statements) if import_statements else "none"
    instructions = [
        (
            "Imports are closed-world; the only allowed import statements are: "
            f"{allowed_text}."
        )
    ]
    if "torch" not in allowed_imports:
        instructions.append(
            "Do not import torch; use operators and methods on the tensors supplied "
            "through reward_inputs_contract."
        )
    return instructions


def _candidate_attempt_public_record(
    candidate: CandidatePackageRef,
    verification: VerificationResult,
) -> dict[str, Any]:
    record = {
        "candidate_id": candidate.candidate_id,
        "candidate_changed_components": sorted(candidate.change_set.changed_components),
        "candidate_source_visibility": "private_non_prompt_artifact",
        "candidate_source_included_in_prompt": False,
        "verifier_status": verification.status,
        "verifier_reason": verification.reason,
    }
    if verification.status == "pass" and "://" not in candidate.path:
        identity = resolve_candidate_program_identity(
            candidate.path,
            expected_candidate_id=candidate.candidate_id,
        )
        record.update(
            {
                "candidate_subject_manifest_hash": identity.manifest_hash,
                "candidate_program_source_checksum": identity.source_checksum,
                "candidate_program_source_size_bytes": identity.source_size_bytes,
            }
        )
    return record


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
        decision = public_record.get("decision")
        candidate_id = (
            candidate.get("candidate_id")
            if isinstance(candidate, Mapping)
            else None
        )
        accepted = (
            decision.get("accepted")
            if isinstance(decision, Mapping)
            and isinstance(decision.get("accepted"), bool)
            else None
        )
        projected = {
            "round_id": record.round_id,
            "candidate_id": candidate_id,
            "accepted": accepted,
            "parent_program_source_checksum": public_record.get(
                "parent_program_source_checksum"
            ),
            "content": content,
            "content_sha256": source.get("content_sha256"),
        }
        historical_anchor = _evaluated_candidate_authoring_anchor(
            record,
            public_record,
        )
        if historical_anchor is not None:
            anchor_content, anchor_sha256, anchor_role = historical_anchor
            projected.update(
                {
                    "anchor_content": anchor_content,
                    "anchor_content_sha256": anchor_sha256,
                    "anchor_role": anchor_role,
                }
            )
        sources.append(projected)
    return tuple(sources)


def _evaluated_candidate_authoring_anchor(
    record: ExperimentRecord,
    public_record: Mapping[str, Any],
) -> tuple[str, str, str] | None:
    """Resolve the immutable prompt source that authored one evaluated candidate."""

    prompt_ref = record.prompt_payload_ref
    if prompt_ref is None:
        raw_prompt_ref = public_record.get("prompt_payload")
        if not isinstance(raw_prompt_ref, Mapping):
            return None
        try:
            prompt_ref = PromptPayloadRef.from_mapping(raw_prompt_ref)
        except (TypeError, ValueError):
            return None
    evidence_items = tuple(
        item
        for item in record.private_artifacts
        if item.kind == AUTHORING_EVIDENCE_KIND
    )
    try:
        if evidence_items:
            if len(evidence_items) != 1:
                return None
            evidence_item = evidence_items[0]
            if (
                evidence_item.checksum is None
                or evidence_item.size_bytes is None
            ):
                return None
            evidence = verify_authoring_evidence(
                AuthoringEvidenceRef(
                    path=evidence_item.uri,
                    checksum=evidence_item.checksum,
                    size_bytes=evidence_item.size_bytes,
                ),
                expected_round_id=record.round_id,
                expected_selected_prompt=prompt_ref,
            )
            prompt_payload = verify_authoring_anchor_prompt(evidence)
        else:
            prompt_payload = verify_prompt_payload(prompt_ref)
        anchor_content, anchor_sha256, anchor_role = select_authoring_anchor_source(
            prompt_payload
        )
    except (OSError, TypeError, ValueError):
        return None
    if not (
        isinstance(anchor_content, str)
        and isinstance(anchor_sha256, str)
        and isinstance(anchor_role, str)
        and anchor_role != "none"
    ):
        return None
    return anchor_content, anchor_sha256, anchor_role


def _result(
    *,
    stop_reason: str,
    prompt_payload_ref: PromptPayloadRef | None,
    round_public_context: Mapping[str, Any] | None,
    attempt_records: list[dict[str, Any]],
    attempt_prompt_payload_refs: list[PromptPayloadRef],
    journal: RoundAuthoringJournal,
    parent_program_source_checksum: str | None,
    candidate: CandidatePackageRef | None = None,
    provider_failure: Mapping[str, Any] | None = None,
    static_failure: Mapping[str, Any] | None = None,
) -> CandidateAuthoringResult:
    assert prompt_payload_ref is not None
    assert round_public_context is not None
    round_id = attempt_records[0].get("round_id")
    if not isinstance(round_id, str) or not round_id:
        raise ValueError("authoring result has no round identity")
    selected_attempt_id = attempt_records[-1].get("attempt_id")
    if not isinstance(selected_attempt_id, str) or not selected_attempt_id:
        raise ValueError("authoring result has no selected attempt identity")
    authoring_evidence_ref = journal.publish_evidence(
        selected_attempt_id=selected_attempt_id,
        stop_reason=stop_reason,
        provider_failure=provider_failure,
        static_authoring_failure=static_failure,
    )
    evidence = verify_authoring_evidence(
        authoring_evidence_ref,
        expected_round_id=round_id,
        expected_selected_prompt=prompt_payload_ref,
    )
    outcome = authoring_outcome_from_evidence(evidence)
    if outcome["stop_reason"] != stop_reason:
        raise ValueError("sealed authoring outcome stop reason mismatch")
    return CandidateAuthoringResult(
        stop_reason=str(outcome["stop_reason"]),
        prompt_payload_ref=prompt_payload_ref,
        round_public_context=round_public_context,
        attempt_records=attempt_records,
        attempt_prompt_payload_refs=tuple(attempt_prompt_payload_refs),
        authoring_evidence_ref=authoring_evidence_ref,
        parent_program_source_checksum=parent_program_source_checksum,
        source_request_feedback=outcome["source_request_feedback"],
        candidate=candidate,
        provider_failure=outcome["provider_failure"],
        static_failure=outcome["static_authoring_failure"],
    )


def _sanitize_repair_feedback(feedback: Mapping[str, Any]) -> dict[str, Any]:
    clean = sanitize_prompt_public_context(dict(feedback))
    assert_public_only_prompt_payload(clean)
    return clean


def _transport_retry_feedback(error: ProviderCandidateError) -> dict[str, Any]:
    """Retry a transient transport call without asking the author to change code."""

    feedback = _sanitize_repair_feedback(
        {
            "schema_version": "true_rl_agent.provider_transport_retry_feedback.v1",
            "status": "retry_required",
            "provider_status": "fail",
            "failure_category": error.category,
            "reason": error.reason,
            "failure": error.to_public_dict(),
            "instructions": [
                "Retry the same candidate authoring request; the prior request failed before a provider response was available.",
                "Do not infer a candidate-code defect from this transport failure.",
            ],
        }
    )
    return feedback
