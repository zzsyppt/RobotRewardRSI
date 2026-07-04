"""Static candidate authoring attempts before runner execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from true_rl_agent.agent_core.provider import CandidateProvider
from true_rl_agent.agent_core.provider_errors import ProviderCandidateError, build_provider_repair_feedback
from true_rl_agent.candidates.prompt_projection import assert_public_only_prompt_payload, build_prompt_payload
from true_rl_agent.candidates.schema import CandidatePackageRef
from true_rl_agent.candidates.verifier import CandidateVerifier, VerificationResult
from true_rl_agent.memory import ExperimentRecord, PromptPayloadRef, PublicContextBuilder
from true_rl_agent.agent_core.context import sanitize_prompt_public_context
from true_rl_agent.prompting.source_catalog import LATEST_FAILED_SOURCE_ID


@dataclass(frozen=True)
class CandidateAuthoringResult:
    """Result of provider-compatible static authoring for one round."""

    stop_reason: str
    prompt_payload_path: Path
    prompt_payload_ref: PromptPayloadRef
    public_context: Mapping[str, Any]
    attempt_records: list[dict[str, Any]]
    attempt_prompt_payload_paths: list[str]
    candidate: CandidatePackageRef | None = None
    provider_failure: Mapping[str, Any] | None = None


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
    ) -> None:
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        self.context_builder = context_builder
        self.provider = provider
        self.verifier = verifier
        self.round_dir = round_dir
        self.max_attempts = max_attempts

    def run_round(
        self,
        *,
        round_id: str,
        records: Sequence[ExperimentRecord],
    ) -> CandidateAuthoringResult:
        repair_feedback: dict[str, Any] | None = None
        latest_failed_candidate: CandidatePackageRef | None = None
        attempt_records: list[dict[str, Any]] = []
        attempt_prompt_payload_paths: list[str] = []

        selected_prompt_path: Path | None = None
        selected_prompt_payload_ref: PromptPayloadRef | None = None
        selected_public_context: Mapping[str, Any] | None = None
        selected_candidate: CandidatePackageRef | None = None
        last_provider_failure: Mapping[str, Any] | None = None

        for attempt in range(1, self.max_attempts + 1):
            prompt_payload = build_authoring_prompt_payload(
                context_builder=self.context_builder,
                records=records,
                repair_feedback=repair_feedback,
                latest_failed_candidate=latest_failed_candidate,
            )
            prompt_path = self.round_dir / _prompt_payload_filename(attempt)
            prompt_payload_ref = self.context_builder.save_prompt_payload(prompt_path, prompt_payload)
            attempt_prompt_payload_paths.append(str(prompt_path))

            attempt_record = {
                "round_id": round_id,
                "attempt": attempt,
                "prompt_payload_path": str(prompt_path),
            }
            attempt_records.append(attempt_record)
            selected_prompt_path = prompt_path
            selected_prompt_payload_ref = prompt_payload_ref
            selected_public_context = prompt_payload["public_context"]

            try:
                candidate = self.provider.next_candidate(prompt_payload=prompt_payload)
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

            attempt_record["provider_status"] = "pass"
            verification = self.verifier.verify(candidate)
            attempt_record.update(_candidate_attempt_public_record(candidate, verification))
            selected_candidate = candidate
            if verification.status != "pass":
                if attempt == self.max_attempts:
                    return _result(
                        stop_reason="static_failed",
                        prompt_payload_path=selected_prompt_path,
                        prompt_payload_ref=selected_prompt_payload_ref,
                        public_context=selected_public_context,
                        attempt_records=attempt_records,
                        attempt_prompt_payload_paths=attempt_prompt_payload_paths,
                        candidate=selected_candidate,
                    )
                latest_failed_candidate = candidate
                repair_feedback = build_verifier_repair_feedback(candidate=candidate, verification=verification)
                continue

            return _result(
                stop_reason="candidate_verified",
                prompt_payload_path=selected_prompt_path,
                prompt_payload_ref=selected_prompt_payload_ref,
                public_context=selected_public_context,
                attempt_records=attempt_records,
                attempt_prompt_payload_paths=attempt_prompt_payload_paths,
                candidate=selected_candidate,
            )

        return _result(
            stop_reason="provider_failed",
            prompt_payload_path=selected_prompt_path,
            prompt_payload_ref=selected_prompt_payload_ref,
            public_context=selected_public_context,
            attempt_records=attempt_records,
            attempt_prompt_payload_paths=attempt_prompt_payload_paths,
            provider_failure=last_provider_failure,
        )


def build_authoring_prompt_payload(
    *,
    context_builder: PublicContextBuilder,
    records: Sequence[ExperimentRecord],
    repair_feedback: dict[str, Any] | None,
    latest_failed_candidate: CandidatePackageRef | None,
) -> dict[str, Any]:
    public_context = context_builder.build(records, latest_failed_candidate=latest_failed_candidate)
    if repair_feedback is not None:
        public_context["candidate_repair_feedback"] = _sanitize_repair_feedback(repair_feedback)
    payload = build_prompt_payload(public_context=public_context)
    assert_public_only_prompt_payload(payload)
    return payload


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


def _result(
    *,
    stop_reason: str,
    prompt_payload_path: Path | None,
    prompt_payload_ref: PromptPayloadRef | None,
    public_context: Mapping[str, Any] | None,
    attempt_records: list[dict[str, Any]],
    attempt_prompt_payload_paths: list[str],
    candidate: CandidatePackageRef | None = None,
    provider_failure: Mapping[str, Any] | None = None,
) -> CandidateAuthoringResult:
    assert prompt_payload_path is not None
    assert prompt_payload_ref is not None
    assert public_context is not None
    return CandidateAuthoringResult(
        stop_reason=stop_reason,
        prompt_payload_path=prompt_payload_path,
        prompt_payload_ref=prompt_payload_ref,
        public_context=public_context,
        attempt_records=attempt_records,
        attempt_prompt_payload_paths=attempt_prompt_payload_paths,
        candidate=candidate,
        provider_failure=provider_failure,
    )


def _prompt_payload_filename(attempt: int) -> str:
    if attempt == 1:
        return "prompt_payload.json"
    return f"prompt_payload_attempt_{attempt:03d}.json"


def _sanitize_repair_feedback(feedback: Mapping[str, Any]) -> dict[str, Any]:
    clean = sanitize_prompt_public_context(dict(feedback))
    assert_public_only_prompt_payload(clean)
    return clean
