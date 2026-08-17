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
    TEST_CANDIDATE_FAILURE_CONTRACT,
    TEST_SESSION_GENESIS_HASH,
    make_bootstrap_current_best,
    valid_accepted_ledger_record,
)
from true_rl_agent.agent_core.eval_run_audit import build_eval_run_audit
from true_rl_agent.evaluation.acceptance_protocol import (
    ACCEPTANCE_DELTA_ACCEPT_REASONS,
    ACCEPTANCE_DELTA_REJECT_REASONS,
)
from true_rl_agent.evaluation.current_best import CurrentBestEvalStats
from true_rl_agent.evaluation.decision_record_contract import (
    project_validated_decision as _project_validated_decision,
)
from true_rl_agent.evaluation.decision_record_contract import (
    validate_decision_record as _validate_decision_record,
)


def _accepted_record() -> dict[str, object]:
    return valid_accepted_ledger_record(
        round_id="round-contract",
        candidate_id="candidate-contract",
        public_score=0.50,
    )


def validate_decision_record(record, *, acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL):
    return _validate_decision_record(
        record,
        acceptance_protocol=acceptance_protocol,
    )


def project_validated_decision(
    record,
    *,
    validation=None,
    acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
):
    return _project_validated_decision(
        record,
        validation=validation,
        acceptance_protocol=acceptance_protocol,
    )


class DecisionRecordContractTests(unittest.TestCase):
    def test_valid_policy_accept_satisfies_shared_contract(self) -> None:
        validation = validate_decision_record(_accepted_record())

        self.assertTrue(validation.valid)
        self.assertTrue(validation.valid_policy_accept)
        self.assertEqual(validation.reasons, ())

    def test_evaluated_record_requires_reconstructable_eval_protocol(self) -> None:
        mutations = {
            "missing_sealed_count": (
                lambda record: record["jobs"]["eval"].pop(
                    "sealed_validation_seed_count"
                ),
                "eval_job_eval_protocol_sealed_validation_seed_count_invalid",
            ),
            "nonpositive_public_accept_count": (
                lambda record: record["jobs"]["eval"].__setitem__(
                    "public_accept_seed_count", 0
                ),
                "eval_job_eval_protocol_public_accept_seed_count_invalid",
            ),
            "forged_hash": (
                lambda record: record["jobs"]["eval"].__setitem__(
                    "eval_protocol_hash", "sha256:forged"
                ),
                "eval_job_eval_protocol_hash_mismatch",
            ),
        }
        for label, (mutate, expected_reason) in mutations.items():
            with self.subTest(label=label):
                record = _accepted_record()
                mutate(record)

                validation = validate_decision_record(record)

                self.assertFalse(validation.valid)
                self.assertFalse(validation.valid_policy_accept)
                self.assertIn(expected_reason, validation.reasons)

    def test_evaluated_record_requires_checkpoint_binding(self) -> None:
        record = _accepted_record()
        record["jobs"]["eval"].pop("checkpoint")

        validation = validate_decision_record(record)

        self.assertFalse(validation.valid)
        self.assertFalse(validation.valid_policy_accept)
        self.assertIn("eval_job_checkpoint_hash_missing", validation.reasons)

    def test_accepted_and_outcome_must_be_bidirectionally_consistent(self) -> None:
        cases = (
            (True, "reject", "candidate_verifier_failed"),
            (False, "policy_improvement_accept", "absolute_accept_floor"),
        )
        for accepted, outcome, reason in cases:
            with self.subTest(accepted=accepted, outcome=outcome):
                record = _accepted_record()
                record["decision"].update(
                    accepted=accepted,
                    outcome=outcome,
                    reason=reason,
                )

                validation = validate_decision_record(record)

                self.assertFalse(validation.valid)
                self.assertFalse(validation.valid_policy_accept)
                self.assertIn("decision_acceptance_outcome_mismatch", validation.reasons)

    def test_unknown_reject_reason_fails_closed(self) -> None:
        record = _accepted_record()
        record["decision"].update(
            accepted=False,
            outcome="reject",
            reason="invented_reject_reason",
        )

        validation = validate_decision_record(record)

        self.assertFalse(validation.valid)
        self.assertIn("decision_reason_not_allowed_for_outcome", validation.reasons)

    def test_all_canonical_delta_reasons_are_allowed_for_their_outcome(self) -> None:
        for reason in ACCEPTANCE_DELTA_ACCEPT_REASONS:
            with self.subTest(reason=reason):
                record = _accepted_record()
                record["decision"]["reason"] = reason

                validation = validate_decision_record(record)

                self.assertNotIn(
                    "decision_reason_not_allowed_for_outcome",
                    validation.reasons,
                )
        for reason in ACCEPTANCE_DELTA_REJECT_REASONS:
            with self.subTest(reason=reason):
                record = _accepted_record()
                record["decision"].update(
                    accepted=False,
                    outcome="reject",
                    reason=reason,
                )

                validation = validate_decision_record(record)

                self.assertNotIn(
                    "decision_reason_not_allowed_for_outcome",
                    validation.reasons,
                )

    def test_delta_reason_must_match_the_bound_acceptance_rule(self) -> None:
        record = _accepted_record()
        record["decision"]["reason"] = "bounded_paired_guard_headroom_delta"

        validation = validate_decision_record(record)

        self.assertFalse(validation.valid)
        self.assertIn(
            "bounded_paired_guard_headroom_delta_reason_evidence_mismatch",
            validation.reasons,
        )

    def test_delta_reason_requires_self_consistent_comparison(self) -> None:
        record = _accepted_record()
        record["decision"].update(
            reason="noise_aware_delta",
            comparison={
                **copy.deepcopy(record["decision"]["comparison"]),
                "mode": "accepted",
                "delta": 0.20,
                "threshold": 0.05,
                "delta_met": False,
            },
        )

        validation = validate_decision_record(record)

        self.assertFalse(validation.valid)
        self.assertIn("decision_comparison_delta_met_mismatch", validation.reasons)

    def test_reject_terminal_stage_cannot_coexist_with_later_evidence(self) -> None:
        cases = {
            "task_invariant_with_full_eval": (
                "task_invariant_violation",
                lambda record: None,
            ),
            "runtime_failure_with_pass_eval": (
                "candidate_runtime_failed",
                lambda record: record.__setitem__(
                    "runner_error", {"category": "candidate_runtime_error"}
                ),
            ),
            "verifier_failure_with_full_eval": (
                "candidate_verifier_failed",
                lambda record: record.__setitem__("verifier", {"status": "fail"}),
            ),
            "provider_failure_with_full_eval": (
                "provider_candidate_generation_failed",
                lambda record: record.__setitem__(
                    "provider_failure", {"category": "provider_failure"}
                ),
            ),
        }
        for label, (reason, mutate) in cases.items():
            with self.subTest(label=label):
                record = _accepted_record()
                record["decision"].update(
                    accepted=False,
                    outcome="reject",
                    reason=reason,
                )
                mutate(record)

                validation = validate_decision_record(record)

                self.assertFalse(validation.valid)
                self.assertTrue(
                    any(item.startswith("round_stage_") for item in validation.reasons),
                    validation.reasons,
                )

    def test_current_writer_reject_terminal_stages_satisfy_contract(self) -> None:
        cases = {
            "provider_failure": {
                "provider_failure": {"category": "provider_failure"},
                "decision": _reject_decision("provider_candidate_generation_failed"),
            },
            "static_authoring_failure": {
                "candidate": {"candidate_id": "candidate-static"},
                "static_authoring_failure": {"category": "static_failure"},
                "decision": _reject_decision("candidate_static_authoring_failed"),
            },
            "verifier_failure": {
                "candidate": {"candidate_id": "candidate-verifier"},
                "verifier": {"status": "fail"},
                "decision": _reject_decision("candidate_verifier_failed"),
            },
            "task_invariant_failure": {
                "candidate": {"candidate_id": "candidate-invariant"},
                "verifier": {"status": "pass"},
                "jobs": {"train": {"job_id": "train"}, "eval": {"job_id": "eval"}},
                "decision": _reject_decision("task_invariant_violation"),
            },
            "training_runtime_failure": {
                "candidate": {"candidate_id": "candidate-train-failure"},
                "verifier": {"status": "pass"},
                "jobs": {
                    "train": {
                        "job_id": "train",
                        "candidate_failure_contract_hash": (
                            TEST_CANDIDATE_FAILURE_CONTRACT.contract_hash
                        ),
                    },
                    "eval": {"job_id": "eval"},
                },
                "runner_error": {
                    "type": "CandidateRuntimeError",
                    "message": "candidate failed inside runner",
                    "details": {
                        "status": "fail",
                        "subprocess_returncode": 1,
                        "candidate_failure_contract": (
                            TEST_CANDIDATE_FAILURE_CONTRACT.to_public_dict()
                        ),
                        "candidate_failure": (
                            TEST_CANDIDATE_FAILURE_CONTRACT.evidence(
                                stage="reward_execution",
                                reason_code="runtime_rejected",
                            ).to_public_dict()
                        ),
                    },
                },
                "decision": _reject_decision("candidate_runtime_failed"),
            },
            "post_train_eval_failure": {
                "candidate": {"candidate_id": "candidate-eval-failure"},
                "verifier": {"status": "pass"},
                "jobs": {"train": {"job_id": "train"}, "eval": {"job_id": "eval"}},
                "train_result": {"job_id": "train-complete"},
                "runner_error": {"category": "eval_incomplete"},
                "decision": _reject_decision("eval_incomplete"),
            },
        }
        for label, record in cases.items():
            with self.subTest(label=label):
                validation = validate_decision_record(record)

                self.assertTrue(validation.valid, validation.reasons)
                self.assertTrue(validation.stage_consistent)

    def test_empty_mapping_is_not_stage_evidence(self) -> None:
        record = _accepted_record()
        record["train_result"] = {}
        record["public_result"] = {}

        validation = validate_decision_record(record)

        self.assertFalse(validation.valid)
        self.assertIn("round_stage_empty_evidence:train_result", validation.reasons)
        self.assertIn("round_stage_empty_evidence:public_result", validation.reasons)
        self.assertFalse(validation.round_stage.train_result_present)
        self.assertFalse(validation.round_stage.public_result_present)

    def test_accepted_decision_binds_surface_and_public_score(self) -> None:
        cases = {
            "surface_mismatch": lambda record: record.__setitem__(
                "candidate_surface",
                {
                    "classification": "parameter_only",
                    "eligible_for_policy_accept": False,
                    "reason": "parameter_only_candidate",
                },
            ),
            "surface_missing": lambda record: record.pop("candidate_surface"),
            "public_score_mismatch": lambda record: record["decision"].__setitem__(
                "public_score", 999.0
            ),
        }
        for label, mutate in cases.items():
            with self.subTest(label=label):
                record = _accepted_record()
                mutate(record)

                validation = validate_decision_record(record)

                self.assertFalse(validation.valid_policy_accept)
                self.assertFalse(validation.valid)

    def test_public_score_cannot_self_certify_against_different_acceptance_mean(self) -> None:
        record = _accepted_record()
        record["public_result"]["public_score"] = 0.99
        record["decision"]["public_score"] = 0.99
        record["decision"]["comparison"]["public_score"] = 0.99

        validation = validate_decision_record(
            record,
            acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
        )
        projection = project_validated_decision(
            record,
            validation=validation,
            acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
        )
        audit = build_eval_run_audit(
            [record],
            acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
            initial_current_best=make_bootstrap_current_best(),
            expected_session_genesis_hash=TEST_SESSION_GENESIS_HASH,
        )
        current_best = CurrentBestEvalStats.from_ledger_records(
            [record],
            initial_current_best=make_bootstrap_current_best(),
            session_genesis_hash=TEST_SESSION_GENESIS_HASH,
            acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
        )

        self.assertFalse(validation.valid)
        self.assertFalse(validation.valid_policy_accept)
        self.assertIn("public_score_binding_mismatch", validation.reasons)
        self.assertIsNone(projection["public_score"])
        self.assertFalse(audit["checks"]["decision_records_valid"])
        self.assertEqual(audit["counts"]["accepted_count"], 0)
        self.assertEqual(current_best.mode, "bootstrap")
        self.assertEqual(current_best.public_score, 0.0)

    def test_ledger_public_result_uses_the_shared_field_contract(self) -> None:
        mutations = {
            "bad_status": lambda result: result.__setitem__("status", "banana"),
            "bool_score": lambda result: result.__setitem__("public_score", True),
            "nan_score": lambda result: result.__setitem__(
                "public_score", float("nan")
            ),
            "bad_metrics": lambda result: result.__setitem__("metrics", []),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                record = _accepted_record()
                mutate(record["public_result"])

                validation = validate_decision_record(
                    record,
                    acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
                )
                projection = project_validated_decision(
                    record,
                    validation=validation,
                    acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
                )

                self.assertFalse(validation.valid)
                self.assertFalse(validation.public_result_contract_valid)
                self.assertIn("public_result_contract_invalid", validation.reasons)
                self.assertIsNone(projection["public_score"])

    def test_comparison_binds_candidate_score_and_delta_arithmetic(self) -> None:
        cases = {
            "candidate_score_mismatch": {
                "candidate_score": 0.75,
            },
            "delta_arithmetic_mismatch": {
                "mode": "accepted",
                "candidate_score": 0.50,
                "current_best_score": 0.49,
                "delta": 100.0,
                "threshold": 1.0,
                "delta_met": True,
            },
        }
        for label, comparison_update in cases.items():
            with self.subTest(label=label):
                record = _accepted_record()
                record["decision"]["comparison"].update(comparison_update)
                if label == "delta_arithmetic_mismatch":
                    record["decision"]["reason"] = "noise_aware_delta"

                validation = validate_decision_record(record)

                self.assertFalse(validation.valid_policy_accept)
                self.assertFalse(validation.valid)

    def test_evaluated_policy_reject_still_binds_consumed_gate_evidence(self) -> None:
        record = _accepted_record()
        record["decision"].update(
            accepted=False,
            outcome="reject",
            reason="absolute_accept_floor_not_met",
        )
        record["decision"]["comparison"]["absolute_accept_floor"] = 1.0
        record["decision"]["consumed_gates"] = []

        validation = validate_decision_record(record)

        self.assertFalse(validation.valid)
        self.assertTrue(
            any(
                reason.startswith("decision_required_gate_not_consumed:")
                for reason in validation.reasons
            )
        )

    def test_invalid_decision_public_projection_fails_closed(self) -> None:
        record = _accepted_record()
        record["verifier"] = {"status": "fail"}
        record["decision"].update(
            accepted=True,
            outcome="policy_improvement_accept",
            reason="candidate_verifier_failed",
            public_score=999.0,
        )

        projection = project_validated_decision(record)

        self.assertEqual(projection["validation_status"], "invalid")
        self.assertEqual(projection["outcome"], "reject")
        self.assertFalse(projection["accepted"])
        self.assertEqual(projection["reason"], "decision_record_invalid")
        self.assertIsNone(projection["public_score"])
        self.assertEqual(projection["comparison"], {})
        self.assertTrue(projection["validation_reasons"])


def _reject_decision(reason: str) -> dict[str, object]:
    return {
        "outcome": "reject",
        "accepted": False,
        "reason": reason,
        "surface_classification": "code_level",
        "public_score": None,
        "comparison": {},
        "consumed_gates": [],
        "gate_statuses": {},
    }


if __name__ == "__main__":
    unittest.main()
