"""Task-neutral byte contracts for durable authoring artifacts."""

from __future__ import annotations

from typing import Any

from true_rl_agent.agent_core.source_request_limits import (
    MAX_SOURCE_REQUEST_BYTES_PER_ROUND,
)
from true_rl_agent.prompt_serialization import MAX_AUTHORING_PROMPT_PAYLOAD_BYTES

MAX_AUTHORING_PROVIDER_ATTEMPTS = 5
MAX_AUTHORING_BUDGET_ARTIFACT_BYTES = 4 * 1024

# Attempt results persist only a bounded public attempt projection plus private
# continuation state. Raw provider response bytes and candidate source files are
# never embedded in this artifact.
MAX_AUTHORING_ATTEMPT_PUBLIC_RECORD_BYTES = 64 * 1024
MAX_AUTHORING_ATTEMPT_RESULT_FIXED_BYTES = 16 * 1024
MAX_AUTHORING_ATTEMPT_RESULT_BYTES = (
    MAX_AUTHORING_ATTEMPT_RESULT_FIXED_BYTES
    + MAX_AUTHORING_ATTEMPT_PUBLIC_RECORD_BYTES
    + MAX_AUTHORING_PROMPT_PAYLOAD_BYTES
    + MAX_SOURCE_REQUEST_BYTES_PER_ROUND
)

# Evidence contains prompt refs and public attempt projections, not the private
# continuation payload. The second copy allowance covers terminal projections.
MAX_AUTHORING_EVIDENCE_FIXED_BYTES = 64 * 1024
MAX_AUTHORING_EVIDENCE_BYTES_PER_ATTEMPT = (
    2 * MAX_AUTHORING_ATTEMPT_PUBLIC_RECORD_BYTES
)


def require_authoring_attempt_count(value: Any) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
        or value > MAX_AUTHORING_PROVIDER_ATTEMPTS
    ):
        raise ValueError(
            "authoring provider attempts must be between 1 and "
            f"{MAX_AUTHORING_PROVIDER_ATTEMPTS}"
        )
    return value


def authoring_evidence_max_bytes(max_provider_attempts: Any) -> int:
    attempts = require_authoring_attempt_count(max_provider_attempts)
    return (
        MAX_AUTHORING_EVIDENCE_FIXED_BYTES
        + attempts * MAX_AUTHORING_EVIDENCE_BYTES_PER_ATTEMPT
    )


MAX_AUTHORING_EVIDENCE_ABSOLUTE_BYTES = authoring_evidence_max_bytes(
    MAX_AUTHORING_PROVIDER_ATTEMPTS
)


__all__ = [
    "MAX_AUTHORING_ATTEMPT_RESULT_BYTES",
    "MAX_AUTHORING_BUDGET_ARTIFACT_BYTES",
    "MAX_AUTHORING_EVIDENCE_ABSOLUTE_BYTES",
    "MAX_AUTHORING_PROVIDER_ATTEMPTS",
    "authoring_evidence_max_bytes",
    "require_authoring_attempt_count",
]
