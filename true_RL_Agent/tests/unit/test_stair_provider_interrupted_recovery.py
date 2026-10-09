from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from true_rl_agent.prompt_artifacts import RoundAuthoringJournal
from true_rl_agent.agent_core.generation_provider_budget import (
    CostGuardedProvider,
)


class StairProviderInterruptedRecoveryTests(unittest.TestCase):
    def _journal_with_prior_and_current_prompt(
        self,
        output_dir: Path,
    ) -> tuple[RoundAuthoringJournal, dict[str, str]]:
        generation_id = "pilot-generation-001"
        round_id = f"{generation_id}-lane-a"
        journal = RoundAuthoringJournal(
            output_dir / "authoring" / generation_id / "lane-a",
            round_id=round_id,
        )
        journal.bind_budget(
            max_provider_attempts=5,
            max_source_requests_per_round=2,
            max_source_ids_per_request=4,
            max_source_request_bytes_per_round=64 * 1024,
        )
        journal.begin_attempt(
            {
                "schema_version": "test.stair.prompt.v1",
                "attempt_kind": "prior-interrupted-request",
                "candidate_lineage": {"parent_program_source_checksum": None},
            }
        )
        current_prompt = {
            "schema_version": "test.stair.prompt.v1",
            "attempt_kind": "fresh-resume-request",
            "candidate_lineage": {"parent_program_source_checksum": None},
        }
        journal.begin_attempt(current_prompt)
        return journal, current_prompt

    def test_p6b_default_still_rejects_prior_request_without_usage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp).resolve()
            journal, current_prompt = self._journal_with_prior_and_current_prompt(
                output_dir
            )
            self.assertEqual(
                [item["status"] for item in journal.snapshot().attempts],
                ["interrupted_after_prompt", "interrupted_after_prompt"],
            )
            delegate = Mock()
            provider = CostGuardedProvider(
                delegate=delegate,
                provider_caps={"attempts_per_lane_per_generation": 5, "input_units_per_attempt_reserve": 300000, "output_units_per_attempt_cap": 32768},
                output_dir=output_dir,
                generation_id="pilot-generation-001",
                lane_id="lane-a",
                max_generations=1,
            )

            with self.assertRaisesRegex(RuntimeError, "authoritative usage"):
                provider.next_authoring_response(prompt_payload=current_prompt)

            delegate.next_authoring_response.assert_not_called()

    def test_p6c_reserves_prior_interrupted_request_and_uses_fresh_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp).resolve()
            journal, current_prompt = self._journal_with_prior_and_current_prompt(
                output_dir
            )
            attempts = journal.snapshot().attempts
            self.assertEqual(
                [item["attempt_id"] for item in attempts],
                ["attempt_000001", "attempt_000002"],
            )
            self.assertNotEqual(
                attempts[0]["prompt_payload"]["checksum"],
                attempts[1]["prompt_payload"]["checksum"],
            )
            expected = object()
            delegate = Mock()
            delegate.next_authoring_response.return_value = expected
            provider = CostGuardedProvider(
                delegate=delegate,
                provider_caps={"attempts_per_lane_per_generation": 5, "input_units_per_attempt_reserve": 300000, "output_units_per_attempt_cap": 32768},
                output_dir=output_dir,
                generation_id="pilot-generation-001",
                lane_id="lane-a",
                max_generations=3,
                allow_conservative_interrupted_recovery=True,
            )

            actual = provider.next_authoring_response(prompt_payload=current_prompt)

            self.assertIs(actual, expected)
            delegate.next_authoring_response.assert_called_once_with(
                prompt_payload=current_prompt
            )


if __name__ == "__main__":
    unittest.main()
