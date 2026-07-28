from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tests.fakes.decision_records import (  # noqa: E402
    TEST_ACCEPTANCE_PROTOCOL,
    TEST_SESSION_GENESIS_HASH,
    make_bootstrap_current_best,
    valid_accepted_ledger_record,
)
from true_rl_agent.agent_core.authoring_response import (
    SourceRequestResponse,  # noqa: E402
)
from true_rl_agent.agent_core.source_request_service import (
    resolve_source_request,  # noqa: E402
)
from true_rl_agent.memory import PublicContextBuilder  # noqa: E402
from true_rl_agent.prompting.serialization import (
    canonical_prompt_json_bytes,  # noqa: E402
)
from true_rl_agent.prompting.source_catalog import (  # noqa: E402
    CURRENT_BEST_SOURCE_ID,
    prompt_source_catalog_logical_ids,
)


class SourceRequestServiceTests(unittest.TestCase):
    def test_resolver_rejects_tampered_source_payload_before_returning_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate_dir = Path(tmp) / "candidate-accepted"
            (candidate_dir / "training_code").mkdir(parents=True)
            (candidate_dir / "training_code" / "components.py").write_text(
                "def current_best_reward(inp):\n    return inp.base_height\n",
                encoding="utf-8",
            )
            payload = PublicContextBuilder(
                acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
                initial_current_best=make_bootstrap_current_best(),
                session_genesis_hash=TEST_SESSION_GENESIS_HASH,
            ).build_prompt_payload(
                [
                    valid_accepted_ledger_record(
                        round_id="round-accepted",
                        candidate_id="candidate-accepted",
                        public_score=0.31,
                        candidate_path=str(candidate_dir),
                    )
                ]
            )

        response = SourceRequestResponse(
            logical_ids=(CURRENT_BEST_SOURCE_ID,),
            rationale="Need current best source.",
            source_catalog_logical_ids=tuple(sorted(prompt_source_catalog_logical_ids(payload))),
        )
        clean = resolve_source_request(
            prompt_payload=payload,
            response=response,
            source_request_count=0,
            source_request_bytes_returned=0,
        )
        self.assertEqual(clean.status, "source_payload_returned")
        self.assertEqual(
            clean.attempt_record()["budget"],
            {
                "source_request_count": 1,
                "max_source_requests_per_round": 2,
                "max_source_ids_per_request": 2,
                "cumulative_returned_bytes": clean.returned_bytes,
                "max_source_request_bytes_per_round": 48 * 1024,
            },
        )
        self.assertEqual(
            clean.returned_bytes,
            sum(canonical_prompt_json_bytes(item) for item in clean.source_payloads),
        )
        denied_by_canonical_budget = resolve_source_request(
            prompt_payload=payload,
            response=response,
            source_request_count=0,
            source_request_bytes_returned=0,
            max_source_request_bytes_per_round=clean.returned_bytes - 1,
        )
        self.assertEqual(denied_by_canonical_budget.status, "source_request_denied")
        self.assertEqual(
            denied_by_canonical_budget.feedback["denied_logical_ids"][0]["reason"],
            "source_request_byte_budget_exceeded",
        )

        tampered = copy.deepcopy(payload)
        tampered["source_payloads"][0]["content"] = "def injected_reward(inp):\n    return 999\n"

        with self.assertRaisesRegex(ValueError, "source_payload"):
            resolve_source_request(
                prompt_payload=tampered,
                response=response,
                source_request_count=0,
                source_request_bytes_returned=0,
            )


if __name__ == "__main__":
    unittest.main()
