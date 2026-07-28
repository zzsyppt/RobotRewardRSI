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
    valid_accepted_ledger_record,
)
from true_rl_agent.agent_core.decision import decide_candidate
from true_rl_agent.candidates.schema import CandidateSurfaceAssessment
from true_rl_agent.evaluation.acceptance_contract import validate_acceptance_evidence
from true_rl_agent.evaluation.acceptance_protocol import (
    PUBLIC_SCORE_BINDING_ACCEPTANCE_MEAN,
    AcceptanceProtocol,
)
from true_rl_agent.evaluation.binding import EvaluationBindingV1
from true_rl_agent.evaluation.current_best import CurrentBestEvalStats
from true_rl_agent.evaluation.decision_record_contract import validate_decision_record
from true_rl_agent.evaluation.protocol import EvaluationProtocol
from true_rl_agent.evaluation.public_result import PublicEvalResult


class AcceptanceContractTests(unittest.TestCase):
    def test_baseline_requires_eval_gates_but_not_candidate_training_budget(self) -> None:
        record = _record()
        metrics = record["public_result"]["metrics"]
        metrics["acceptance"]["required_pass_gates"] = [
            "accept_evidence_gate",
            "sealed_validation_gate",
            "task_behavior_gate",
        ]
        metrics.pop("budget_accounting")
        metrics["task_behavior_gate"] = {"status": "pass"}

        validation = validate_acceptance_evidence(
            metrics,
            acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
            public_score=record["public_result"]["public_score"],
            expected_sealed_seed_count=7,
            expected_checkpoint_hash="sha256:" + "d" * 64,
            evidence_subject_kind="baseline",
        )

        self.assertTrue(validation.valid, validation.reasons)
        self.assertNotIn("budget_accounting", validation.required_gates)
        self.assertEqual(validation.gate_statuses["task_behavior_gate"], "pass")

        metrics.pop("task_behavior_gate")
        missing_task_gate = validate_acceptance_evidence(
            metrics,
            acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
            public_score=record["public_result"]["public_score"],
            expected_sealed_seed_count=7,
            expected_checkpoint_hash="sha256:" + "d" * 64,
            evidence_subject_kind="baseline",
        )
        self.assertFalse(missing_task_gate.valid)
        self.assertIn("task_behavior_gate", missing_task_gate.failed_required_gates)

    def test_eval_metrics_cannot_override_task_owned_bootstrap_floor(self) -> None:
        for label, floor in {
            "missing": "missing",
            "none": None,
            "string": "0.05",
            "negative": -1.0,
            "valid": TEST_ACCEPTANCE_PROTOCOL.absolute_accept_floor,
        }.items():
            with self.subTest(label=label):
                record = valid_accepted_ledger_record(
                    round_id=f"round-{label}",
                    candidate_id=f"candidate-{label}",
                    public_score=0.01,
                )
                acceptance = record["public_result"]["metrics"]["acceptance"]
                if floor == "missing":
                    acceptance.pop("absolute_accept_floor")
                else:
                    acceptance["absolute_accept_floor"] = floor
                decision = decide_candidate(
                    surface=CandidateSurfaceAssessment(
                        "code_level", True, "code_level_candidate"
                    ),
                    public_result=PublicEvalResult(**record["public_result"]),
                    current_best=_bootstrap(
                        TEST_ACCEPTANCE_PROTOCOL,
                        TEST_EVALUATION_BINDING,
                    ),
                    acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
                )

                self.assertFalse(decision.accepted)
                self.assertEqual(
                    decision.reason,
                    (
                        "absolute_accept_floor_not_met"
                        if label == "valid"
                        else "acceptance_protocol_mismatch"
                    ),
                )
    def test_counts_must_be_positive_and_do_not_form_canonical_stats(self) -> None:
        record = _record()
        metrics = record["public_result"]["metrics"]
        metrics["acceptance"]["min_score_count"] = -1
        metrics["acceptance"]["score_stats"]["count"] = -1
        metrics["public_accept"]["score_stats"]["count"] = -1

        validation = validate_acceptance_evidence(metrics)

        self.assertFalse(validation.valid)
        self.assertIn("min_score_count_invalid", validation.reasons)
        self.assertIn("accept_score_count_invalid", validation.reasons)
        self.assertIsNone(validation.min_score_count)
        self.assertIsNone(validation.score_count)
        self.assertEqual(validation.score_stats, {})

    def test_min_count_one_allows_missing_standard_error_everywhere(self) -> None:
        record = _record()
        protocol = AcceptanceProtocol(
            protocol_id="test_acceptance_count_one_v1",
            score_source=TEST_ACCEPTANCE_PROTOCOL.score_source,
            min_score_count=1,
            absolute_accept_floor=0.05,
            min_delta=0.05,
            public_score_binding=PUBLIC_SCORE_BINDING_ACCEPTANCE_MEAN,
        )
        eval_protocol = EvaluationProtocol(
            eval_protocol_version="test_eval_count_one_v1",
            public_seed_count=1,
            public_accept_seed_count=1,
            sealed_validation_seed_count=7,
            eval_num_envs=64,
        )
        binding = EvaluationBindingV1(
            eval_protocol_hash=eval_protocol.protocol_hash,
            acceptance_protocol_hash=protocol.protocol_hash,
            task_protocol_hash=TEST_EVALUATION_BINDING.task_protocol_hash,
            seed_cohort_hash=TEST_EVALUATION_BINDING.seed_cohort_hash,
        )
        initial_current_best = _bootstrap(protocol, binding)
        metrics = record["public_result"]["metrics"]
        metrics["acceptance"]["min_score_count"] = 1
        metrics["acceptance"]["acceptance_protocol_hash"] = protocol.protocol_hash
        record["decision"]["comparison"]["acceptance_protocol_hash"] = protocol.protocol_hash
        record["evaluation_binding"] = binding.to_public_dict()
        metrics["evaluation_binding"] = binding.to_public_dict()
        record["decision"]["comparison"][
            "candidate_evaluation_binding_hash"
        ] = binding.binding_hash
        record["decision"]["comparison"][
            "current_best"
        ] = initial_current_best.comparison_identity()
        for stats in (
            metrics["acceptance"]["score_stats"],
            metrics["public_accept"]["score_stats"],
        ):
            stats["count"] = 1
            stats["eval_protocol_hash"] = eval_protocol.protocol_hash
            stats.pop("standard_error", None)
        sealed_evidence = metrics["sealed_validation"]
        sealed_evidence["eval_protocol_hash"] = eval_protocol.protocol_hash
        sealed_evidence["evaluation_binding_hash"] = binding.binding_hash
        sealed_evidence["score_stats"][
            "eval_protocol_hash"
        ] = eval_protocol.protocol_hash
        record["jobs"]["eval"] = {
            **record["jobs"]["eval"],
            **eval_protocol.to_public_dict(),
            "evaluation_binding": binding.to_public_dict(),
        }
        record["decision"]["comparison"]["candidate_score_stats"] = copy.deepcopy(
            metrics["acceptance"]["score_stats"]
        )

        validation = validate_acceptance_evidence(
            metrics,
            acceptance_protocol=protocol,
        )

        self.assertTrue(validation.valid, validation.reasons)
        self.assertEqual(validation.score_count, 1)
        self.assertEqual(validation.min_score_count, 1)
        self.assertEqual(validation.score_mean, 0.50)
        self.assertIsNone(validation.standard_error)
        self.assertNotIn("standard_error", validation.score_stats)

        ledger_validation = validate_decision_record(
            record,
            acceptance_protocol=protocol,
        )
        self.assertTrue(ledger_validation.valid_policy_accept, ledger_validation.reasons)
        current_best = CurrentBestEvalStats.from_ledger_records(
            [record],
            initial_current_best=initial_current_best,
            session_genesis_hash=TEST_SESSION_GENESIS_HASH,
            acceptance_protocol=protocol,
        )
        self.assertEqual(current_best.mode, "accepted")
        self.assertTrue(current_best.has_full_eval_stats())

    def test_missing_acceptance_score_stats_has_one_canonical_reject_reason(self) -> None:
        record = _record()
        metrics = record["public_result"]["metrics"]
        metrics["acceptance"].pop("score_stats")
        public_result = PublicEvalResult(**record["public_result"])

        evidence = validate_acceptance_evidence(
            metrics,
            acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
        )
        decision = decide_candidate(
            surface=CandidateSurfaceAssessment(
                "code_level", True, "code_level_candidate"
            ),
            public_result=public_result,
            current_best=_bootstrap(
                TEST_ACCEPTANCE_PROTOCOL,
                TEST_EVALUATION_BINDING,
            ),
            acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
        )
        record["decision"] = decision.to_public_dict()
        ledger_validation = validate_decision_record(
            record,
            acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
        )

        self.assertEqual(evidence.reject_reason, "required_score_missing")
        self.assertEqual(evidence.score_stats, {})
        self.assertEqual(decision.reason, evidence.reject_reason)
        self.assertTrue(ledger_validation.valid, ledger_validation.reasons)

    def test_public_score_is_canonical_only_when_bound_to_acceptance_mean(self) -> None:
        record = _record()
        metrics = record["public_result"]["metrics"]

        valid = validate_acceptance_evidence(
            metrics,
            acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
            public_score=0.50,
        )
        mismatched = validate_acceptance_evidence(
            metrics,
            acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
            public_score=0.99,
        )

        self.assertTrue(valid.public_score_binding_valid, valid.reasons)
        self.assertEqual(valid.canonical_public_score, 0.50)
        self.assertFalse(mismatched.public_score_binding_valid)
        self.assertIsNone(mismatched.canonical_public_score)
        self.assertIn("public_score_binding_mismatch", mismatched.reasons)
        self.assertEqual(mismatched.reject_reason, "public_score_binding_mismatch")


def _record() -> dict[str, object]:
    return copy.deepcopy(
        valid_accepted_ledger_record(
            round_id="round-acceptance",
            candidate_id="candidate-acceptance",
            public_score=0.50,
        )
    )


def _bootstrap(
    protocol: AcceptanceProtocol,
    binding: EvaluationBindingV1,
) -> CurrentBestEvalStats:
    return CurrentBestEvalStats.bootstrap(
        initial_best_score=0.0,
        acceptance_protocol=protocol,
        evaluation_binding=binding,
        anchor_id="acceptance-contract-bootstrap",
    )


if __name__ == "__main__":
    unittest.main()
