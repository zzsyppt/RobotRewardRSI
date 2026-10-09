"""Public-safe provider failure records and repair feedback."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from true_rl_agent.agent_core.context import sanitize_prompt_public_context
from true_rl_agent.agent_core.public_safety import (
    redact_authorization_credential_markers,
    redact_secret_key_markers,
    sanitize_untrusted_public_diagnostic,
    scrub_public_paths,
)
from true_rl_agent.prompting import assert_public_only_prompt_payload

PROVIDER_FAILURE_CATEGORIES = frozenset(
    {
        "json_parse_error",
        "response_truncated",
        "reasoning_output_exhausted",
        "schema_error",
        "candidate_materialize_error",
        "verifier_error",
        "runtime_error",
        "source_request_denied",
    }
)
PROVIDER_USAGE_FIELDS = (
    "request_units",
    "response_units",
    "total_units",
    "request_cache_hit_units",
    "request_cache_miss_units",
    "reasoning_units",
)
_PROVIDER_USAGE_SOURCE_FIELDS = {
    "request_units": ("request_units", "input_tokens", "prompt_tokens"),
    "response_units": ("response_units", "output_tokens", "completion_tokens"),
    "total_units": ("total_units", "total_tokens"),
    "request_cache_hit_units": (
        "request_cache_hit_units",
        "input_cache_hit_tokens",
        "prompt_cache_hit_tokens",
    ),
    "request_cache_miss_units": (
        "request_cache_miss_units",
        "input_cache_miss_tokens",
        "prompt_cache_miss_tokens",
    ),
    "reasoning_units": ("reasoning_units", "reasoning_tokens"),
}
_MAX_PROVIDER_USAGE_COUNT = 1_000_000_000_000
_PROVIDER_RECEIPT_SCHEMA_V1 = "true_rl_agent.provider_response_receipt.v1"
_PROVIDER_RECEIPT_SCHEMA_V2 = "true_rl_agent.provider_response_receipt.v2"
_PROVIDER_RECEIPT_SCHEMA_VERSION = "true_rl_agent.provider_response_receipt.v3"
_PROVIDER_RECEIPT_SCHEMA_V4 = "true_rl_agent.provider_response_receipt.v4"
EXTERNAL_RECEIPT_SCHEMA = "true_rl_agent.external_author_receipt.v1"
_PROVIDER_RECEIPT_OPTIONAL_STRINGS = (
    "requested_model",
    "response_model",
    "system_fingerprint",
    "finish_reason",
    "transport_policy_id",
)
_PROVIDER_RECEIPT_PRESENCE_FIELDS = (
    "response_text_present",
    "reasoning_content_present",
)
_PROVIDER_RECEIPT_COUNT_FIELDS = (
    "requested_output_units",
    "response_max_bytes",
    "response_text_utf8_bytes",
    "reasoning_content_utf8_bytes",
)
_PROVIDER_RECEIPT_HASH_FIELDS = (
    "response_text_sha256",
    "reasoning_content_sha256",
)


class ProviderCandidateError(RuntimeError):
    """A provider-side candidate failure that may be repaired by another attempt."""

    def __init__(
        self,
        category: str,
        reason: str,
        *,
        details: Mapping[str, Any] | None = None,
        provider_usage: Mapping[str, Any] | None = None,
        provider_receipt: Mapping[str, Any] | None = None,
    ) -> None:
        if category not in PROVIDER_FAILURE_CATEGORIES:
            raise ValueError(f"unknown provider failure category: {category}")
        self.category = category
        self.reason = _public_string(reason)
        self.details = _sanitize_provider_value(dict(details or {}))
        self.provider_usage = normalize_provider_usage(provider_usage)
        self.provider_receipt = normalize_provider_receipt(provider_receipt)
        super().__init__(f"{self.category}: {self.reason}")

    def to_public_dict(self) -> dict[str, Any]:
        return _sanitize_provider_value(
            {
                "schema_version": "true_rl_agent.provider_failure.v1",
                "category": self.category,
                "reason": self.reason,
                "details": self.details,
            }
        )

    @property
    def requires_service_recovery(self) -> bool:
        """Billing and authorization failures cannot be repaired by a candidate."""
        return provider_failure_requires_service_recovery(
            {"category": self.category, "details": self.details}
        )


def provider_failure_requires_service_recovery(value: Any) -> bool:
    """Use the same service-rejection boundary for live errors and journals."""
    if not isinstance(value, Mapping):
        return False
    details = value.get("details")
    return (
        value.get("category") == "runtime_error"
        and isinstance(details, Mapping)
        and details.get("failure_kind") == "http_error"
        and details.get("retryable") is False
        and details.get("http_status") in {401, 402, 403}
    )


class ProviderServiceUnavailableError(RuntimeError):
    """Stop the invocation while retaining the current authoring continuation."""

    def __init__(self, error: ProviderCandidateError) -> None:
        self.provider_failure = error.to_public_dict()
        super().__init__(f"provider service requires recovery (HTTP {error.details['http_status']})")


def build_provider_repair_feedback(error: ProviderCandidateError) -> dict[str, Any]:
    failure = error.to_public_dict()
    instructions = [
        "Return one valid JSON object matching candidate_draft with candidate_id, declared_changed_components, and components_py.",
        (
            "Put the canonical experiment hypothesis in the non-empty literal "
            "TrainingComponents.rationale inside components_py; do not return a "
            "standalone candidate rationale field."
        ),
        "Fix only the candidate package response or training_code/components.py needed for this failure.",
        "Do not ask to change runner, eval protocol, training budget, seed policy, checkpoints, PPO, or acceptance rules.",
    ]
    if error.category in {"response_truncated", "reasoning_output_exhausted"}:
        instructions[0:0] = [
            "Return a shorter response that closes the complete JSON object within the provider output limit.",
            (
                "Keep components_py complete; shorten comments and repeated explanation "
                "while retaining the embedded TrainingComponents.rationale hypothesis."
            ),
        ]
    feedback = _sanitize_provider_value(
        {
            "schema_version": "true_rl_agent.provider_repair_feedback.v1",
            "status": "repair_required",
            "provider_status": "fail",
            "failure_category": failure["category"],
            "reason": failure["reason"],
            "failure": failure,
            "instructions": instructions,
        }
    )
    assert_public_only_prompt_payload(feedback)
    return feedback


def normalize_provider_usage(value: Any) -> dict[str, int]:
    """Retain bounded usage counters without provider response content."""

    if not isinstance(value, Mapping):
        return {}
    usage: dict[str, int] = {}
    for field_name in PROVIDER_USAGE_FIELDS:
        count = next(
            (
                value[source_name]
                for source_name in _PROVIDER_USAGE_SOURCE_FIELDS[field_name]
                if source_name in value
            ),
            None,
        )
        if field_name == "reasoning_units" and count is None:
            details = value.get("completion_tokens_details")
            if isinstance(details, Mapping):
                count = details.get("reasoning_tokens")
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
            or count > _MAX_PROVIDER_USAGE_COUNT
        ):
            continue
        usage[field_name] = count
    return usage


def normalize_provider_receipt(value: Any) -> dict[str, Any]:
    """Retain bounded response identity and hashes without response content."""

    if isinstance(value, Mapping) and value.get("schema_version") == EXTERNAL_RECEIPT_SCHEMA:
        if value.get("backend") != "external_file" or value.get("usage_status") != "unknown_full_reserve":
            return {}
        model = value.get("requested_model")
        if not isinstance(model, str) or not model:
            return {}
        capture = value.get("response_capture")
        if capture not in {"complete", "byte_limit_prefix"}:
            return {}
        receipt = {"schema_version": EXTERNAL_RECEIPT_SCHEMA, "backend": "external_file",
                   "usage_status": "unknown_full_reserve", "requested_model": _public_string(model),
                   "response_capture": capture}
        for name in ("system_sha256", "user_sha256", "response_text_sha256"):
            item = value.get(name)
            if not isinstance(item, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", item):
                return {}
            receipt[name] = item
        for name in ("response_text_utf8_bytes", "requested_output_units"):
            item = value.get(name)
            if type(item) is not int or not 0 <= item <= _MAX_PROVIDER_USAGE_COUNT:
                return {}
            receipt[name] = item
        return receipt
    if (
        not isinstance(value, Mapping)
        or value.get("schema_version")
        not in {
            _PROVIDER_RECEIPT_SCHEMA_V1,
            _PROVIDER_RECEIPT_SCHEMA_V2,
            _PROVIDER_RECEIPT_SCHEMA_VERSION,
            _PROVIDER_RECEIPT_SCHEMA_V4,
        }
    ):
        return {}
    schema_version = str(value["schema_version"])
    receipt: dict[str, Any] = {
        "schema_version": schema_version,
    }
    for field_name in _PROVIDER_RECEIPT_OPTIONAL_STRINGS:
        if schema_version == _PROVIDER_RECEIPT_SCHEMA_V1 and field_name == "transport_policy_id":
            continue
        item = value.get(field_name)
        if field_name == "transport_policy_id" and schema_version != _PROVIDER_RECEIPT_SCHEMA_V1:
            if isinstance(item, str) and item:
                receipt[field_name] = _public_string(item)
            else:
                return {}
        elif item is None:
            receipt[field_name] = None
        elif isinstance(item, str):
            receipt[field_name] = _public_string(item)
    thinking_mode = value.get("requested_thinking_mode")
    if thinking_mode in {"enabled", "disabled", "provider_default"}:
        receipt["requested_thinking_mode"] = thinking_mode
    if schema_version in {_PROVIDER_RECEIPT_SCHEMA_VERSION, _PROVIDER_RECEIPT_SCHEMA_V4}:
        reasoning_effort = value.get("requested_reasoning_effort")
        if reasoning_effort not in {"high", "max", "provider_default"}:
            return {}
        receipt["requested_reasoning_effort"] = reasoning_effort
    for field_name in _PROVIDER_RECEIPT_PRESENCE_FIELDS:
        item = value.get(field_name)
        if isinstance(item, bool):
            receipt[field_name] = item
    for field_name in _PROVIDER_RECEIPT_COUNT_FIELDS:
        if schema_version == _PROVIDER_RECEIPT_SCHEMA_V1 and field_name == "response_max_bytes":
            continue
        item = value.get(field_name)
        if (
            field_name == "response_max_bytes"
            and schema_version != _PROVIDER_RECEIPT_SCHEMA_V1
            and (
                isinstance(item, bool)
                or not isinstance(item, int)
                or item <= 0
                or item > _MAX_PROVIDER_USAGE_COUNT
            )
        ):
            return {}
        if (
            not isinstance(item, bool)
            and isinstance(item, int)
            and 0 <= item <= _MAX_PROVIDER_USAGE_COUNT
        ):
            receipt[field_name] = item
    for field_name in _PROVIDER_RECEIPT_HASH_FIELDS:
        item = value.get(field_name)
        if item is None:
            receipt[field_name] = None
        elif isinstance(item, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", item):
            receipt[field_name] = item
    if schema_version == _PROVIDER_RECEIPT_SCHEMA_V4:
        from true_rl_agent.agent_core.provider_limits import validate_provider_model_context
        try:
            context = validate_provider_model_context(value.get("model_context"))
        except ValueError:
            return {}
        if "requested_output_units" not in receipt:
            return {}
        input_tokens = value.get("input_tokens")
        if input_tokens is not None and (isinstance(input_tokens, bool) or not isinstance(input_tokens, int) or input_tokens < 0):
            return {}
        expected_status = ("unknown_without_provider_usage" if input_tokens is None
                           else "within_declared_window" if input_tokens + receipt["requested_output_units"] <= context["context_window_tokens"]
                           else "exceeds_declared_window")
        if value.get("joint_capacity_status") != expected_status:
            return {}
        receipt.update(model_context=context, input_tokens=input_tokens, joint_capacity_status=expected_status)
    return receipt


def _sanitize_provider_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        value = sanitize_untrusted_public_diagnostic(value)
        return sanitize_prompt_public_context(
            {str(key): _sanitize_provider_value(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return [_sanitize_provider_value(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_provider_value(item) for item in value]
    if isinstance(value, str):
        return _public_string(value)
    return value


def _public_string(value: str) -> str:
    compact = " ".join(str(value).split())
    compact = redact_secret_key_markers(compact)
    compact = redact_authorization_credential_markers(compact)
    compact = scrub_public_paths(compact)
    sanitized = sanitize_prompt_public_context(compact)
    if isinstance(sanitized, str):
        compact = sanitized
    return compact[:500]
