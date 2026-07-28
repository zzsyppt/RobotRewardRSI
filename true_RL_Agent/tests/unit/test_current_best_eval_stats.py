from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tests.fakes.decision_records import (
    TEST_ACCEPTANCE_PROTOCOL,
    TEST_EVALUATION_PROTOCOL,
    TEST_SESSION_GENESIS_HASH,
    make_bootstrap_current_best,
    valid_accepted_ledger_record,
)
from true_rl_agent.evaluation.current_best import CurrentBestEvalStats


class CurrentBestEvalStatsTests(unittest.TestCase):
    def test_restores_latest_accepted_eval_stats_from_ledger_records(self) -> None:
        records = [
            {
                "round_id": "round-001",
                "candidate": {"candidate_id": "rejected-candidate"},
                "decision": {"accepted": False},
                "public_result": {"public_score": 0.90, "metrics": {}},
            },
            valid_accepted_ledger_record(
                round_id="round-002",
                candidate_id="accepted-candidate",
                public_score=0.62,
                initial_best_score=0.10,
            ),
        ]

        stats = _restore(records)

        self.assertEqual(stats.mode, "accepted")
        self.assertEqual(stats.accepted_round_id, "round-002")
        self.assertEqual(stats.accepted_candidate_id, "accepted-candidate")
        self.assertEqual(stats.public_accept_stats["mean"], 0.62)
        self.assertEqual(
            stats.eval_protocol_hash,
            TEST_EVALUATION_PROTOCOL.protocol_hash,
        )
        self.assertEqual(stats.sealed_validation_gate["status"], "pass")
        self.assertEqual(stats.gate_statuses["sealed_validation_gate"], "pass")
        self.assertEqual(stats.score_for_comparison(), 0.62)

    def test_no_accepted_record_returns_bootstrap_stats(self) -> None:
        stats = _restore(
            [{"round_id": "round-001", "decision": {"accepted": False}}]
        )

        self.assertEqual(stats.mode, "bootstrap")
        self.assertEqual(stats.public_score, 0.10)
        self.assertEqual(stats.public_accept_stats, {})
        self.assertIsNone(stats.accepted_round_id)

    def test_ignores_accepted_record_with_missing_required_gate_contract(self) -> None:
        records = [
            {
                "round_id": "round-001",
                "candidate": {"candidate_id": "bad-accepted-candidate"},
                "decision": {
                    "accepted": True,
                    "consumed_gates": ["sealed_validation_gate", "budget_accounting", "motion_quality_gate"],
                    "gate_statuses": {
                        "sealed_validation_gate": "pass",
                        "budget_accounting": "pass",
                        "motion_quality_gate": "pass",
                    },
                },
                "public_result": {
                    "status": "pass",
                    "public_score": 0.99,
                    "metrics": {
                        "acceptance": {
                            "score_stats": {
                                "count": 3,
                                "mean": 0.99,
                                "std": 0.0,
                                "standard_error": 0.0,
                                "eval_protocol_hash": "protocol-a",
                            },
                            "required_pass_gates": [
                                "sealed_validation_gate",
                                "budget_accounting",
                                "motion_quality_gate",
                            ],
                        },
                        "accept_evidence_gate": {"status": "fail", "reason": "insufficient_accept_evidence"},
                        "sealed_validation_gate": _sealed_validation_gate(),
                        "budget_accounting": {"status": "pass"},
                        "motion_quality_gate": {"status": "pass"},
                    },
                },
            }
        ]

        stats = _restore(records)

        self.assertEqual(stats.mode, "bootstrap")
        self.assertEqual(stats.public_score, 0.10)
        self.assertIsNone(stats.accepted_round_id)

    def test_ignores_accepted_record_without_public_accept_split(self) -> None:
        records = [
            {
                "round_id": "round-001",
                "candidate": {"candidate_id": "bad-accepted-candidate"},
                "decision": {
                    "accepted": True,
                    "consumed_gates": [
                        "accept_evidence_gate",
                        "sealed_validation_gate",
                        "budget_accounting",
                    ],
                    "gate_statuses": {
                        "accept_evidence_gate": "pass",
                        "sealed_validation_gate": "pass",
                        "budget_accounting": "pass",
                    },
                },
                "public_result": {
                    "status": "pass",
                    "public_score": 0.99,
                    "metrics": {
                        "acceptance": {
                            "score_stats": {
                                "count": 3,
                                "mean": 0.99,
                                "std": 0.0,
                                "standard_error": 0.0,
                                "eval_protocol_hash": "protocol-a",
                            },
                            "required_pass_gates": [
                                "accept_evidence_gate",
                                "sealed_validation_gate",
                                "budget_accounting",
                            ],
                        },
                        "accept_evidence_gate": {"status": "pass"},
                        "sealed_validation_gate": _sealed_validation_gate(),
                        "budget_accounting": {"status": "pass"},
                    },
                },
            }
        ]

        stats = _restore(records)

        self.assertEqual(stats.mode, "bootstrap")
        self.assertEqual(stats.public_score, 0.10)
        self.assertIsNone(stats.accepted_round_id)

    def test_ignores_accepted_record_with_incomplete_sealed_validation_gate(self) -> None:
        record = _accepted_record()
        record["public_result"]["metrics"]["sealed_validation_gate"] = {"status": "pass"}

        stats = _restore([record])

        self.assertEqual(stats.mode, "bootstrap")
        self.assertEqual(stats.public_score, 0.10)
        self.assertIsNone(stats.accepted_round_id)

    def test_ignores_accepted_record_with_internal_sealed_validation_failure(self) -> None:
        record = _accepted_record()
        record["public_result"]["metrics"]["sealed_validation_gate"]["candidate_isolation_status"] = "fail"

        stats = _restore([record])

        self.assertEqual(stats.mode, "bootstrap")
        self.assertEqual(stats.public_score, 0.10)
        self.assertIsNone(stats.accepted_round_id)

    def test_ignores_accepted_record_with_top_level_candidate_contamination(self) -> None:
        record = _accepted_record()
        record["public_result"]["metrics"]["candidate_code_loaded_in_eval"] = True

        stats = _restore([record])

        self.assertEqual(stats.mode, "bootstrap")
        self.assertEqual(stats.public_score, 0.10)
        self.assertIsNone(stats.accepted_round_id)

    def test_ignores_contradictory_accepted_decision_record(self) -> None:
        record = _accepted_record()
        record["decision"].update(
            accepted=True,
            outcome="reject",
            reason="candidate_verifier_failed",
        )

        stats = _restore([record])

        self.assertEqual(stats.mode, "bootstrap")
        self.assertEqual(stats.public_score, 0.10)
        self.assertIsNone(stats.accepted_round_id)

    def test_ignores_accept_with_failed_verifier_or_missing_train(self) -> None:
        cases = {
            "failed_verifier": lambda record: record.__setitem__(
                "verifier", {"status": "fail"}
            ),
            "missing_train": lambda record: record.__setitem__("train_result", None),
        }
        for label, mutate in cases.items():
            with self.subTest(label=label):
                record = _accepted_record()
                mutate(record)

                stats = _restore([record])

                self.assertEqual(stats.mode, "bootstrap")
                self.assertIsNone(stats.accepted_round_id)

def _restore(records: list[dict[str, object]]) -> CurrentBestEvalStats:
    return CurrentBestEvalStats.from_ledger_records(
        records,
        initial_current_best=make_bootstrap_current_best(
            initial_best_score=0.10,
        ),
        session_genesis_hash=TEST_SESSION_GENESIS_HASH,
        acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
    )


def _accepted_record() -> dict[str, object]:
    return valid_accepted_ledger_record(
        round_id="round-accepted",
        candidate_id="accepted-candidate",
        public_score=0.62,
        initial_best_score=0.10,
    )


def _sealed_validation_gate() -> dict[str, object]:
    return {
        "status": "pass",
        "checkpoint_safety_status": "pass",
        "candidate_isolation_status": "pass",
        "candidate_code_loaded_in_eval": False,
        "candidate_package_loaded_in_eval": False,
        "candidate_path_argument_present": False,
        "eval_protocol_hash_match_status": "pass",
        "checkpoint_hash_match_status": "pass",
    }


if __name__ == "__main__":
    unittest.main()
