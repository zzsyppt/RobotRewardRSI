from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from true_rl_agent.agent_core.source_request_feedback import (  # noqa: E402
    SOURCE_REQUEST_ROUND_FEEDBACK_SCHEMA_VERSION,
    build_source_request_round_feedback,
    project_source_request_round_feedback,
    serialized_source_request_feedback_bytes,
)
from true_rl_agent.agent_core.source_request_limits import (  # noqa: E402
    MAX_SOURCE_IDS_PER_REQUEST,
    MAX_SOURCE_REQUEST_BYTES_PER_ROUND,
    MAX_SOURCE_REQUEST_FEEDBACK_BYTES,
    MAX_SOURCE_REQUEST_FEEDBACK_ENTRIES,
    MAX_SOURCE_REQUESTS_PER_ROUND,
)
from true_rl_agent.prompting.serialization import (
    canonical_prompt_json_bytes,  # noqa: E402
)
from true_rl_agent.prompting.source_catalog import (  # noqa: E402
    CURRENT_BEST_SOURCE_ID,
    LATEST_FAILED_SOURCE_ID,
)

API_SOURCE_ID = "api/candidate_components.py"
EXAMPLE_SOURCE_ID = "examples/known_good_reward_terms.py"


def _result(
    *,
    status: str,
    returned: list[str] | None = None,
    denied: list[dict[str, str]] | None = None,
    count: int = 1,
    max_requests: int = MAX_SOURCE_REQUESTS_PER_ROUND,
    cumulative_bytes: int = 0,
) -> dict[str, object]:
    return {
        "status": status,
        "returned_logical_ids": list(returned or []),
        "denied_logical_ids": list(denied or []),
        "returned_bytes": 0,
        "budget": {
            "source_request_count": count,
            "max_source_requests_per_round": max_requests,
            "max_source_ids_per_request": MAX_SOURCE_IDS_PER_REQUEST,
            "cumulative_returned_bytes": cumulative_bytes,
            "max_source_request_bytes_per_round": MAX_SOURCE_REQUEST_BYTES_PER_ROUND,
        },
    }


class SourceRequestRoundFeedbackTests(unittest.TestCase):
    def test_agent_core_context_imports_before_prompting_in_clean_interpreter(self) -> None:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(SRC_ROOT)
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "from true_rl_agent.agent_core.context import build_public_context",
            ],
            cwd=PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_returned_round_feedback_has_fixed_public_shape(self) -> None:
        feedback = build_source_request_round_feedback(
            [
                _result(
                    status="source_payload_returned",
                    returned=[CURRENT_BEST_SOURCE_ID],
                    cumulative_bytes=321,
                )
            ]
        )

        self.assertEqual(
            feedback,
            {
                "schema_version": SOURCE_REQUEST_ROUND_FEEDBACK_SCHEMA_VERSION,
                "status": "source_payload_returned",
                "request_count": 1,
                "returned_logical_ids": [CURRENT_BEST_SOURCE_ID],
                "denied_logical_ids": [],
                "budget": {
                    "source_request_count": 1,
                    "max_source_requests_per_round": MAX_SOURCE_REQUESTS_PER_ROUND,
                    "max_source_ids_per_request": MAX_SOURCE_IDS_PER_REQUEST,
                    "cumulative_returned_bytes": 321,
                    "max_source_request_bytes_per_round": MAX_SOURCE_REQUEST_BYTES_PER_ROUND,
                },
                "omitted_returned_logical_id_count": 0,
                "omitted_denied_logical_id_count": 0,
            },
        )
        self.assertLessEqual(
            serialized_source_request_feedback_bytes(feedback),
            MAX_SOURCE_REQUEST_FEEDBACK_BYTES,
        )

    def test_partial_feedback_prioritizes_recent_denials_and_latest_reason(self) -> None:
        feedback = build_source_request_round_feedback(
            [
                _result(
                    status="source_payload_returned",
                    returned=[CURRENT_BEST_SOURCE_ID, API_SOURCE_ID],
                    count=1,
                    max_requests=6,
                ),
                _result(
                    status="source_payload_returned",
                    returned=[EXAMPLE_SOURCE_ID, LATEST_FAILED_SOURCE_ID],
                    count=2,
                    max_requests=6,
                ),
                _result(
                    status="source_request_denied",
                    denied=[
                        {"logical_id": CURRENT_BEST_SOURCE_ID, "reason": "source_not_requestable"},
                        {"logical_id": API_SOURCE_ID, "reason": "source_request_turn_limit_exceeded"},
                    ],
                    count=3,
                    max_requests=6,
                ),
                _result(
                    status="source_request_denied",
                    denied=[
                        {
                            "logical_id": CURRENT_BEST_SOURCE_ID,
                            "reason": "source_request_byte_budget_exceeded",
                        },
                        {"logical_id": EXAMPLE_SOURCE_ID, "reason": "source_request_id_limit_exceeded"},
                        {"logical_id": LATEST_FAILED_SOURCE_ID, "reason": "source_payload_not_available"},
                    ],
                    count=4,
                    max_requests=6,
                    cumulative_bytes=123,
                ),
            ]
        )

        self.assertEqual(feedback["status"], "source_request_partial")
        self.assertEqual(feedback["request_count"], 4)
        self.assertEqual(feedback["returned_logical_ids"], [])
        self.assertEqual(
            feedback["denied_logical_ids"],
            [
                {"logical_id": API_SOURCE_ID, "reason": "source_request_turn_limit_exceeded"},
                {
                    "logical_id": CURRENT_BEST_SOURCE_ID,
                    "reason": "source_request_byte_budget_exceeded",
                },
                {"logical_id": EXAMPLE_SOURCE_ID, "reason": "source_request_id_limit_exceeded"},
                {"logical_id": LATEST_FAILED_SOURCE_ID, "reason": "source_payload_not_available"},
            ],
        )
        self.assertEqual(feedback["omitted_returned_logical_id_count"], 4)
        self.assertEqual(feedback["omitted_denied_logical_id_count"], 0)
        self.assertEqual(feedback["budget"]["max_source_requests_per_round"], 6)
        self.assertEqual(feedback["budget"]["cumulative_returned_bytes"], 123)
        self.assertLessEqual(
            len(feedback["returned_logical_ids"]) + len(feedback["denied_logical_ids"]),
            MAX_SOURCE_REQUEST_FEEDBACK_ENTRIES,
        )

    def test_repeated_returned_ids_preserve_first_encounter_order(self) -> None:
        feedback = build_source_request_round_feedback(
            [
                _result(
                    status="source_payload_returned",
                    returned=[API_SOURCE_ID, CURRENT_BEST_SOURCE_ID],
                    count=1,
                ),
                _result(
                    status="source_payload_returned",
                    returned=[API_SOURCE_ID],
                    count=2,
                ),
            ]
        )

        self.assertEqual(
            feedback["returned_logical_ids"],
            [API_SOURCE_ID, CURRENT_BEST_SOURCE_ID],
        )

    def test_projector_drops_tainted_fields_and_is_idempotent(self) -> None:
        projected = project_source_request_round_feedback(
            {
                "schema_version": SOURCE_REQUEST_ROUND_FEEDBACK_SCHEMA_VERSION,
                "status": "source_request_partial",
                "request_count": 1,
                "returned_logical_ids": [CURRENT_BEST_SOURCE_ID, "unknown/source.py"],
                "denied_logical_ids": [
                    {"logical_id": API_SOURCE_ID, "reason": "source_not_requestable"},
                    {"logical_id": CURRENT_BEST_SOURCE_ID, "reason": "invented_reason"},
                    {"logical_id": "unknown/source.py", "reason": "source_not_in_catalog"},
                    {
                        "logical_id": LATEST_FAILED_SOURCE_ID,
                        "reason": "source_payload_not_available",
                        "content": "sk-secret D:/private/source.py",
                    },
                ],
                "budget": {
                    "source_request_count": 1,
                    "max_source_requests_per_round": 2,
                    "max_source_ids_per_request": 2,
                    "cumulative_returned_bytes": 0,
                    "max_source_request_bytes_per_round": 49152,
                    "raw_budget_note": "must_not_leak",
                },
                "omitted_returned_logical_id_count": 2,
                "omitted_denied_logical_id_count": 3,
                "source_request": {"rationale": "must_not_leak"},
                "instructions": ["must_not_leak"],
                "path": "D:/private/source.py",
            }
        )

        self.assertEqual(projected["returned_logical_ids"], [CURRENT_BEST_SOURCE_ID])
        self.assertEqual(
            projected["denied_logical_ids"],
            [{"logical_id": API_SOURCE_ID, "reason": "source_not_requestable"}],
        )
        self.assertEqual(projected["omitted_returned_logical_id_count"], 3)
        self.assertEqual(projected["omitted_denied_logical_id_count"], 6)
        self.assertEqual(project_source_request_round_feedback(projected), projected)
        projected_text = json.dumps(projected, sort_keys=True)
        for forbidden in (
            "must_not_leak",
            "invented_reason",
            "unknown/source.py",
            "content",
            "rationale",
            "instructions",
            "D:/private",
        ):
            self.assertNotIn(forbidden, projected_text)

    def test_byte_bound_evicts_oldest_returned_ids_before_denials(self) -> None:
        source_result = _result(
            status="source_request_partial",
            returned=[API_SOURCE_ID, CURRENT_BEST_SOURCE_ID],
            denied=[
                {"logical_id": EXAMPLE_SOURCE_ID, "reason": "source_not_requestable"},
                {"logical_id": LATEST_FAILED_SOURCE_ID, "reason": "source_not_requestable"},
            ],
        )

        complete = build_source_request_round_feedback([source_result])
        byte_limit = canonical_prompt_json_bytes(complete) - 1
        with patch(
            "true_rl_agent.agent_core.source_request_feedback.MAX_SOURCE_REQUEST_FEEDBACK_BYTES",
            byte_limit,
        ):
            feedback = build_source_request_round_feedback([source_result])

        self.assertEqual(feedback["returned_logical_ids"], [CURRENT_BEST_SOURCE_ID])
        self.assertEqual(len(feedback["denied_logical_ids"]), 2)
        self.assertEqual(feedback["omitted_returned_logical_id_count"], 1)
        self.assertEqual(feedback["omitted_denied_logical_id_count"], 0)
        self.assertLessEqual(serialized_source_request_feedback_bytes(feedback), byte_limit)

    def test_projector_uses_provider_canonical_bytes_for_hard_limit(self) -> None:
        feedback = {
            "schema_version": SOURCE_REQUEST_ROUND_FEEDBACK_SCHEMA_VERSION,
            "status": "source_request_denied",
            "request_count": 1,
            "returned_logical_ids": [],
            "denied_logical_ids": [
                {
                    "logical_id": CURRENT_BEST_SOURCE_ID,
                    "reason": "source_not_requestable",
                }
            ],
            "budget": {
                "source_request_count": 1,
                "max_source_requests_per_round": 10 ** 1449,
                "max_source_ids_per_request": 2,
                "cumulative_returned_bytes": 0,
                "max_source_request_bytes_per_round": 49152,
            },
            "omitted_returned_logical_id_count": 0,
            "omitted_denied_logical_id_count": 0,
        }

        projected = project_source_request_round_feedback(feedback)

        self.assertIsNotNone(projected)
        self.assertLessEqual(
            canonical_prompt_json_bytes(projected),
            MAX_SOURCE_REQUEST_FEEDBACK_BYTES,
        )
        self.assertEqual(
            serialized_source_request_feedback_bytes(projected),
            canonical_prompt_json_bytes(projected),
        )

    def test_empty_or_wrong_schema_feedback_fails_closed(self) -> None:
        self.assertIsNone(build_source_request_round_feedback([]))
        self.assertIsNone(project_source_request_round_feedback(None))
        self.assertIsNone(
            project_source_request_round_feedback(
                {"schema_version": "wrong", "status": "source_request_denied"}
            )
        )

    def test_projector_rejects_inconsistent_status_or_request_count(self) -> None:
        denied_feedback = {
            "schema_version": SOURCE_REQUEST_ROUND_FEEDBACK_SCHEMA_VERSION,
            "status": "source_payload_returned",
            "request_count": 1,
            "returned_logical_ids": [],
            "denied_logical_ids": [
                {"logical_id": CURRENT_BEST_SOURCE_ID, "reason": "source_not_requestable"}
            ],
            "budget": {
                "source_request_count": 1,
                "max_source_requests_per_round": 2,
                "max_source_ids_per_request": 2,
                "cumulative_returned_bytes": 0,
                "max_source_request_bytes_per_round": 49152,
            },
            "omitted_returned_logical_id_count": 0,
            "omitted_denied_logical_id_count": 0,
        }

        self.assertIsNone(project_source_request_round_feedback(denied_feedback))
        denied_feedback["status"] = "source_request_denied"
        denied_feedback["budget"]["source_request_count"] = 99
        self.assertIsNone(project_source_request_round_feedback(denied_feedback))

    def test_projector_fails_closed_when_invalid_ids_are_only_opposite_status_evidence(self) -> None:
        malformed = {
            "schema_version": SOURCE_REQUEST_ROUND_FEEDBACK_SCHEMA_VERSION,
            "status": "source_request_denied",
            "request_count": 1,
            "returned_logical_ids": ["unknown/source.py"],
            "denied_logical_ids": [
                {"logical_id": CURRENT_BEST_SOURCE_ID, "reason": "source_not_requestable"}
            ],
            "budget": {
                "source_request_count": 1,
                "max_source_requests_per_round": 2,
                "max_source_ids_per_request": 2,
                "cumulative_returned_bytes": 0,
                "max_source_request_bytes_per_round": 49152,
            },
            "omitted_returned_logical_id_count": 0,
            "omitted_denied_logical_id_count": 0,
        }

        self.assertIsNone(project_source_request_round_feedback(malformed))


if __name__ == "__main__":
    unittest.main()
