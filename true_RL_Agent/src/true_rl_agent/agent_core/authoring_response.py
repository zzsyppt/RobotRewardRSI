"""Provider authoring response envelope and compatibility helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping

from true_rl_agent.agent_core.provider_contract import SOURCE_REQUEST_SERVICE_ENABLED
from true_rl_agent.agent_core.provider_errors import ProviderCandidateError
from true_rl_agent.agent_core.public_safety import (
    assert_public_only_prompt_payload,
    sanitize_prompt_public_value,
)
from true_rl_agent.candidates.schema import CandidatePackageRef
from true_rl_agent.prompting.source_catalog import prompt_source_catalog_logical_ids

_LOGICAL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_./-]{0,199}$")


@dataclass(frozen=True)
class CandidateDraftResponse:
    """A provider response that produced a materialized candidate draft."""

    candidate: CandidatePackageRef

    @property
    def response_mode(self) -> str:
        return "candidate_draft"

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "true_rl_agent.authoring_provider_response.v1",
            "response_mode": self.response_mode,
            "candidate_id": self.candidate.candidate_id,
            "candidate_changed_components": sorted(self.candidate.change_set.changed_components),
            "candidate_source_visibility": "private_non_prompt_artifact",
        }


@dataclass(frozen=True)
class SourceRequestResponse:
    """A provider response that asks for allowlisted source by logical id."""

    logical_ids: tuple[str, ...]
    rationale: str
    source_catalog_logical_ids: tuple[str, ...]
    raw_response_public: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        logical_ids = tuple(str(item) for item in self.logical_ids)
        if not logical_ids:
            raise ValueError("source_request logical_ids must be a non-empty list")
        for logical_id in logical_ids:
            _validate_source_logical_id(logical_id)
        source_catalog_logical_ids = tuple(str(item) for item in self.source_catalog_logical_ids)
        if not source_catalog_logical_ids:
            raise ValueError("source_request requires prompt source_catalog logical ids")
        for logical_id in source_catalog_logical_ids:
            _validate_source_logical_id(logical_id)
        unknown = sorted(frozenset(logical_ids) - frozenset(source_catalog_logical_ids))
        if unknown:
            raise ValueError("source_request logical_id is not declared in prompt source_catalog")
        object.__setattr__(self, "logical_ids", logical_ids)
        object.__setattr__(self, "source_catalog_logical_ids", source_catalog_logical_ids)
        object.__setattr__(self, "rationale", str(sanitize_prompt_public_value(str(self.rationale)))[:500])
        assert_public_only_prompt_payload(self.to_public_dict())

    @property
    def response_mode(self) -> str:
        return "source_request"

    def to_public_dict(self) -> dict[str, Any]:
        public = {
            "schema_version": "true_rl_agent.authoring_provider_response.v1",
            "response_mode": self.response_mode,
            "logical_ids": list(self.logical_ids),
            "rationale": self.rationale,
            "source_request_service_enabled": SOURCE_REQUEST_SERVICE_ENABLED,
        }
        assert_public_only_prompt_payload(public)
        return public


AuthoringProviderResponse = CandidateDraftResponse | SourceRequestResponse


def normalize_authoring_provider_response(value: Any) -> AuthoringProviderResponse:
    if isinstance(value, CandidateDraftResponse | SourceRequestResponse):
        return value
    if isinstance(value, CandidatePackageRef):
        return CandidateDraftResponse(candidate=value)
    raise TypeError(f"unsupported authoring provider response type: {type(value).__name__}")


def next_authoring_response(provider: Any, *, prompt_payload: dict[str, Any]) -> AuthoringProviderResponse:
    method = getattr(provider, "next_authoring_response", None)
    if callable(method):
        try:
            response = normalize_authoring_provider_response(method(prompt_payload=prompt_payload))
            if isinstance(response, SourceRequestResponse):
                validate_source_request_response_against_prompt(response, prompt_payload=prompt_payload)
            return response
        except ProviderCandidateError:
            raise
        except (TypeError, ValueError) as exc:
            raise ProviderCandidateError(
                "schema_error",
                str(exc),
                details={"response_mode": "source_request"},
            ) from exc
    return CandidateDraftResponse(candidate=provider.next_candidate(prompt_payload=prompt_payload))


def validate_source_request_response_against_prompt(
    response: SourceRequestResponse,
    *,
    prompt_payload: Mapping[str, Any],
) -> None:
    prompt_logical_ids = prompt_source_catalog_logical_ids(prompt_payload)
    if frozenset(response.source_catalog_logical_ids) != prompt_logical_ids:
        raise ProviderCandidateError(
            "schema_error",
            "source_request source_catalog does not match prompt source_catalog",
            details={"response_mode": response.response_mode},
        )
    unknown = sorted(frozenset(response.logical_ids) - prompt_logical_ids)
    if unknown:
        raise ProviderCandidateError(
            "schema_error",
            "source_request logical_id is not declared in prompt source_catalog",
            details={"response_mode": response.response_mode, "requested_logical_ids": unknown},
        )


def build_source_request_denied_error(response: SourceRequestResponse) -> ProviderCandidateError:
    return ProviderCandidateError(
        "source_request_denied",
        "provider returned source_request on a candidate-only provider path",
        details={
            "response_mode": response.response_mode,
            "source_request_service_enabled": SOURCE_REQUEST_SERVICE_ENABLED,
            "requested_logical_ids": list(response.logical_ids),
        },
    )


def _validate_source_logical_id(logical_id: str) -> None:
    if (
        not _LOGICAL_ID_RE.fullmatch(logical_id)
        or ".." in logical_id.split("/")
        or logical_id.startswith("/")
        or "\\" in logical_id
        or ":" in logical_id
    ):
        raise ValueError("source_request logical_ids must use prompt source logical ids")
