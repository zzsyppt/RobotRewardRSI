"""Shared source-request service limits."""

from __future__ import annotations

MAX_SOURCE_REQUESTS_PER_ROUND = 2
MAX_SOURCE_IDS_PER_REQUEST = 2
MAX_SOURCE_REQUEST_BYTES_PER_ROUND = 48 * 1024
MAX_SOURCE_REQUEST_FEEDBACK_ENTRIES = MAX_SOURCE_REQUESTS_PER_ROUND * MAX_SOURCE_IDS_PER_REQUEST
MAX_SOURCE_REQUEST_FEEDBACK_BYTES = 2048

SOURCE_REQUEST_DENIAL_REASONS = frozenset(
    {
        "source_request_service_disabled",
        "source_request_turn_limit_exceeded",
        "source_request_id_limit_exceeded",
        "source_not_in_catalog",
        "source_not_requestable",
        "source_omitted_over_budget",
        "source_payload_not_available",
        "source_request_byte_budget_exceeded",
    }
)

CANDIDATE_COMPONENTS_SOURCE_ID = "api/candidate_components.py"
CURRENT_BEST_SOURCE_ID = "current_best/training_code/components.py"
BEST_OBSERVED_SOURCE_ID = "memory/best_observed/training_code/components.py"
KNOWN_GOOD_SOURCE_ID = "examples/known_good_reward_terms.py"
LATEST_FAILED_SOURCE_ID = "attempt/latest_failed/training_code/components.py"
LATEST_NONACCEPTED_SOURCE_ID = "memory/latest_nonaccepted/training_code/components.py"
PROMPT_SOURCE_LOGICAL_IDS = frozenset(
    {
        CANDIDATE_COMPONENTS_SOURCE_ID,
        CURRENT_BEST_SOURCE_ID,
        BEST_OBSERVED_SOURCE_ID,
        KNOWN_GOOD_SOURCE_ID,
        LATEST_FAILED_SOURCE_ID,
        LATEST_NONACCEPTED_SOURCE_ID,
    }
)
