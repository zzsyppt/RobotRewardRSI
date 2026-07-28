from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tests.fakes.decision_records import (
    TEST_ACCEPTANCE_PROTOCOL,
    TEST_EVALUATION_BINDING,
    TEST_SESSION_GENESIS_HASH,
    make_bootstrap_current_best,
    valid_accepted_ledger_record,
)
from true_rl_agent.agent_core.eval_run_audit import build_eval_run_audit
from true_rl_agent.evaluation.acceptance_replay import replay_acceptance_ledger
from true_rl_agent.evaluation.binding import EvaluationBindingV1
from true_rl_agent.evaluation.current_best import CurrentBestEvalStats
from true_rl_agent.evaluation.decision_record_contract import validate_decision_record
from true_rl_agent.memory import PublicContextBuilder


class AcceptanceReplayTests(unittest.TestCase):
    def test_continuation_rejects_genesis_or_binding_drift(self) -> None:
        cases = {}
        wrong_genesis = valid_accepted_ledger_record(
            round_id="round-genesis",
            candidate_id="candidate-genesis",
            public_score=0.50,
        )
        wrong_genesis["session_genesis_hash"] = "a" * 64
        cases["genesis"] = wrong_genesis

        wrong_binding = valid_accepted_ledger_record(
            round_id="round-binding",
            candidate_id="candidate-binding",
            public_score=0.50,
        )
        drifted_binding = EvaluationBindingV1(
            eval_protocol_hash=TEST_EVALUATION_BINDING.eval_protocol_hash,
            acceptance_protocol_hash=(
                TEST_EVALUATION_BINDING.acceptance_protocol_hash
            ),
            task_protocol_hash=TEST_EVALUATION_BINDING.task_protocol_hash,
            seed_cohort_hash="a" * 64,
        )
        wrong_binding["evaluation_binding"] = drifted_binding.to_public_dict()
        cases["binding"] = wrong_binding

        for label, record in cases.items():
            with self.subTest(label=label):
                replay = replay_acceptance_ledger(
                    [record],
                    initial_current_best=make_bootstrap_current_best(),
                    expected_session_genesis_hash=TEST_SESSION_GENESIS_HASH,
                    acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
                )
                with self.assertRaisesRegex(ValueError, "invalid session binding"):
                    replay.raise_for_continuation()

    def test_public_context_requires_trusted_protocol_for_evaluated_history(self) -> None:
        accepted = valid_accepted_ledger_record(
            round_id="round-001",
            candidate_id="candidate-001",
            public_score=0.50,
        )

        with self.assertRaisesRegex(
            ValueError,
            "acceptance_protocol is required",
        ):
            PublicContextBuilder().build([accepted])

        self.assertEqual(PublicContextBuilder().build([])["previous_rounds"], 0)

    def test_forged_previous_best_cannot_self_certify_second_accept(self) -> None:
        first = valid_accepted_ledger_record(
            round_id="round-001",
            candidate_id="candidate-001",
            public_score=0.50,
        )
        second = _delta_accept_record(
            round_id="round-002",
            candidate_id="candidate-002",
            candidate_score=0.51,
            actual_best=first,
            declared_best_score=0.0,
            declared_delta=0.51,
        )

        intrinsic = validate_decision_record(
            second,
            acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
        )
        replay = replay_acceptance_ledger(
            [first, second],
            initial_current_best=make_bootstrap_current_best(),
            expected_session_genesis_hash=TEST_SESSION_GENESIS_HASH,
            acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
        )
        audit = build_eval_run_audit(
            [first, second],
            acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
            initial_current_best=make_bootstrap_current_best(),
            expected_session_genesis_hash=TEST_SESSION_GENESIS_HASH,
        )
        restored = CurrentBestEvalStats.from_ledger_records(
            [first, second],
            initial_current_best=make_bootstrap_current_best(),
            session_genesis_hash=TEST_SESSION_GENESIS_HASH,
            acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
        )

        self.assertTrue(intrinsic.valid_policy_accept, intrinsic.reasons)
        self.assertIn(
            "acceptance_replay_current_best_score_mismatch",
            replay.entries[1].sequence_reasons,
        )
        self.assertIn(
            "acceptance_replay_delta_mismatch",
            replay.entries[1].sequence_reasons,
        )
        self.assertFalse(replay.entries[1].valid_policy_accept)
        self.assertFalse(audit["checks"]["acceptance_replay_consistent"])
        self.assertFalse(audit["checks"]["decision_records_valid"])
        self.assertEqual(audit["counts"]["accepted_count"], 1)
        self.assertEqual(restored.accepted_round_id, "round-001")
        self.assertEqual(restored.score_for_comparison(), 0.50)
        with self.assertRaisesRegex(
            ValueError,
            "round-002.*acceptance_replay_current_best_score_mismatch",
        ):
            replay.raise_for_invalid_declared_accepts()

    def test_continuation_guard_allows_valid_accept_and_invalid_reject(self) -> None:
        accepted = valid_accepted_ledger_record(
            round_id="round-valid",
            candidate_id="candidate-valid",
            public_score=0.50,
        )
        replay_acceptance_ledger(
            [accepted],
            initial_current_best=make_bootstrap_current_best(),
            expected_session_genesis_hash=TEST_SESSION_GENESIS_HASH,
            acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
        ).raise_for_invalid_declared_accepts()

        invalid_reject = copy.deepcopy(accepted)
        invalid_reject["round_id"] = "round-invalid-reject"
        invalid_reject["verifier"] = {"status": "fail"}
        invalid_reject["decision"].update(
            accepted=False,
            outcome="reject",
            reason="candidate_verifier_failed",
        )
        replay_acceptance_ledger(
            [invalid_reject],
            initial_current_best=make_bootstrap_current_best(),
            expected_session_genesis_hash=TEST_SESSION_GENESIS_HASH,
            acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
        ).raise_for_invalid_declared_accepts()

    def test_prompt_history_consumes_fail_closed_replay_projection(self) -> None:
        first = valid_accepted_ledger_record(
            round_id="round-001",
            candidate_id="candidate-001",
            public_score=0.50,
        )
        second = _delta_accept_record(
            round_id="round-002",
            candidate_id="candidate-002",
            candidate_score=0.51,
            actual_best=first,
            declared_best_score=0.0,
            declared_delta=0.51,
        )

        context = PublicContextBuilder(
            acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
            initial_current_best=make_bootstrap_current_best(),
            session_genesis_hash=TEST_SESSION_GENESIS_HASH,
        ).build([first, second])

        self.assertEqual(context["best_public_score"], 0.50)
        self.assertFalse(context["recent_outcomes"][-1]["accepted"])


def _delta_accept_record(
    *,
    round_id: str,
    candidate_id: str,
    candidate_score: float,
    actual_best: dict,
    declared_best_score: float,
    declared_delta: float,
) -> dict:
    record = valid_accepted_ledger_record(
        round_id=round_id,
        candidate_id=candidate_id,
        public_score=candidate_score,
    )
    best_stats = copy.deepcopy(
        actual_best["public_result"]["metrics"]["acceptance"]["score_stats"]
    )
    candidate_stats = copy.deepcopy(
        record["public_result"]["metrics"]["acceptance"]["score_stats"]
    )
    actual_best_stats = CurrentBestEvalStats.from_public_result_mapping(
        round_id=actual_best["round_id"],
        candidate_id=actual_best["candidate"]["candidate_id"],
        public_result=actual_best["public_result"],
        acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
        subject_manifest_hash=actual_best["jobs"]["eval"][
            "subject_manifest_hash"
        ],
        program_source_checksum=actual_best["jobs"]["eval"][
            "program_source_checksum"
        ],
        training_binding_hash=actual_best["jobs"]["eval"][
            "training_binding_hash"
        ],
        capability_profile_hash=actual_best["jobs"]["eval"][
            "capability_profile_hash"
        ],
        checkpoint_hash=actual_best["jobs"]["eval"]["checkpoint"]["checksum"],
        expected_evaluation_binding=TEST_EVALUATION_BINDING,
    )
    record["decision"].update(
        reason="noise_aware_delta",
        comparison={
            "mode": "accepted",
            "score_source": TEST_ACCEPTANCE_PROTOCOL.score_source,
            "acceptance_protocol_hash": TEST_ACCEPTANCE_PROTOCOL.protocol_hash,
            "candidate_score": candidate_score,
            "current_best_score": declared_best_score,
            "delta": declared_delta,
            "threshold": TEST_ACCEPTANCE_PROTOCOL.min_delta,
            "delta_met": declared_delta > TEST_ACCEPTANCE_PROTOCOL.min_delta,
            "absolute_accept_floor": TEST_ACCEPTANCE_PROTOCOL.absolute_accept_floor,
            "candidate_score_stats": candidate_stats,
            "best_score_stats": best_stats,
            "candidate_evaluation_binding_hash": (
                TEST_EVALUATION_BINDING.binding_hash
            ),
            "current_best": actual_best_stats.comparison_identity(),
            "public_score": candidate_score,
        },
    )
    return record


if __name__ == "__main__":
    unittest.main()
