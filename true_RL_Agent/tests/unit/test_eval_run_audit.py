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
    TEST_CAPABILITY_PROFILE_HASH,
    TEST_EVALUATION_BINDING,
    TEST_EVALUATION_PROTOCOL,
    TEST_PROGRAM_SOURCE_CHECKSUM,
    TEST_SESSION_GENESIS_HASH,
    TEST_SUBJECT_MANIFEST_HASH,
    TEST_TRAINING_BINDING_HASH,
    bind_test_ledger_record,
    make_bootstrap_current_best,
)
from true_rl_agent.agent_core.eval_run_audit import (
    build_eval_run_audit as _build_eval_run_audit,
)
from true_rl_agent.evaluation.acceptance_contract import validate_acceptance_evidence
from true_rl_agent.evaluation.decision_record_contract import validate_decision_record


def build_eval_run_audit(ledger: list[dict[str, object]]) -> dict[str, object]:
    bound_ledger = []
    for record in ledger:
        bound_record = copy.deepcopy(record)
        bound_record.setdefault("session_genesis_hash", TEST_SESSION_GENESIS_HASH)
        bound_record.setdefault(
            "evaluation_binding",
            TEST_EVALUATION_BINDING.to_public_dict(),
        )
        bound_ledger.append(bound_record)
    return _build_eval_run_audit(
        bound_ledger,
        acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
        initial_current_best=make_bootstrap_current_best(),
        expected_session_genesis_hash=TEST_SESSION_GENESIS_HASH,
    )


class EvalRunAuditTests(unittest.TestCase):
    def test_public_dev_count_without_distribution_stats_is_not_full_eval(self) -> None:
        record = _accepted_record()
        record["public_result"]["metrics"]["public_dev"] = {
            "score_stats": {"count": 1}
        }

        audit = build_eval_run_audit([record])

        self.assertFalse(audit["rounds"][0]["public_dev_complete"])
        self.assertEqual(audit["counts"]["full_eval_completed_count"], 0)
        self.assertFalse(audit["checks"]["trainable_rounds_have_full_eval"])

    def test_public_dev_count_must_match_executed_eval_protocol(self) -> None:
        record = _accepted_record()
        record["jobs"]["eval"]["public_seed_count"] = 3

        audit = build_eval_run_audit([record])

        self.assertFalse(audit["rounds"][0]["public_dev_complete"])
        self.assertEqual(audit["counts"]["full_eval_completed_count"], 0)

    def test_acceptance_stats_hash_must_match_executed_eval_job(self) -> None:
        record = _accepted_record()
        record["jobs"]["eval"]["eval_protocol_hash"] = "sha256:forged-job"

        audit = build_eval_run_audit([record])
        validation = validate_decision_record(
            record,
            acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
        )

        self.assertFalse(audit["checks"]["decision_records_valid"])
        self.assertIn("eval_job_eval_protocol_hash_mismatch", validation.reasons)
        self.assertEqual(audit["counts"]["accepted_count"], 0)

    def test_acceptance_validation_exposes_canonical_protocol_binding(self) -> None:
        metrics = _full_eval_metrics()

        validation = validate_acceptance_evidence(metrics)

        self.assertTrue(validation.protocol_binding_valid)
        self.assertEqual(
            validation.eval_protocol_hash,
            TEST_EVALUATION_PROTOCOL.protocol_hash,
        )

    def test_acceptance_protocol_binding_requires_both_equal_hashes(self) -> None:
        mutations = {
            "missing_acceptance_hash": lambda metrics: metrics["acceptance"][
                "score_stats"
            ].pop("eval_protocol_hash"),
            "missing_public_accept_hash": lambda metrics: metrics["public_accept"][
                "score_stats"
            ].pop("eval_protocol_hash"),
            "conflicting_hash": lambda metrics: metrics["public_accept"][
                "score_stats"
            ].__setitem__("eval_protocol_hash", "sha256:other"),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                metrics = _full_eval_metrics()
                mutate(metrics)

                validation = validate_acceptance_evidence(metrics)

                self.assertFalse(validation.protocol_binding_valid)
                self.assertIsNone(validation.eval_protocol_hash)

    def test_run_audit_uses_canonical_protocol_binding_for_full_eval(self) -> None:
        mutations = {
            "missing_acceptance_hash": lambda metrics: metrics["acceptance"][
                "score_stats"
            ].pop("eval_protocol_hash"),
            "missing_public_accept_hash": lambda metrics: metrics["public_accept"][
                "score_stats"
            ].pop("eval_protocol_hash"),
            "conflicting_hash": lambda metrics: metrics["public_accept"][
                "score_stats"
            ].__setitem__("eval_protocol_hash", "sha256:other"),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                record = _accepted_record()
                mutate(record["public_result"]["metrics"])
                record["decision"].update(
                    accepted=False,
                    outcome="reject",
                    reason="accept_score_stats_untrusted",
                )

                audit = build_eval_run_audit([record])

                self.assertFalse(audit["rounds"][0]["eval_protocol_bound"])
                self.assertEqual(audit["counts"]["full_eval_completed_count"], 0)
                self.assertEqual(audit["counts"]["eval_incomplete_count"], 1)
                self.assertFalse(audit["checks"]["trainable_rounds_have_full_eval"])

    def test_nonpositive_accept_counts_are_not_accepted_or_full_eval(self) -> None:
        record = _accepted_record()
        metrics = record["public_result"]["metrics"]
        metrics["acceptance"]["min_score_count"] = -1
        metrics["acceptance"]["score_stats"]["count"] = -1
        metrics["public_accept"]["score_stats"]["count"] = -1

        audit = build_eval_run_audit([record])

        self.assertEqual(audit["counts"]["accepted_count"], 0)
        self.assertEqual(audit["counts"]["full_eval_completed_count"], 0)
        self.assertFalse(audit["checks"]["decision_records_valid"])

    def test_good_accepted_record_passes_required_phase8_checks(self) -> None:
        audit = build_eval_run_audit([_accepted_record()])

        self.assertTrue(audit["checks"]["trainable_rounds_have_full_eval"])
        self.assertTrue(audit["checks"]["accepted_records_have_required_eval_evidence"])
        self.assertTrue(audit["checks"]["decision_consumed_required_gates"])
        self.assertTrue(audit["checks"]["decision_threshold_consumed"])
        self.assertTrue(audit["checks"]["decision_reasons_consistent_with_eval"])
        self.assertEqual(audit["counts"]["authored_candidate_count"], 1)
        self.assertEqual(audit["counts"]["trainable_candidate_count"], 1)
        self.assertEqual(audit["counts"]["trained_candidate_count"], 1)
        self.assertEqual(audit["counts"]["full_eval_completed_count"], 1)
        self.assertEqual(audit["counts"]["sealed_validation_completed_count"], 1)
        self.assertEqual(audit["counts"]["accepted_count"], 1)
        self.assertEqual(audit["counts"]["eval_incomplete_count"], 0)

    def test_contradictory_accepted_decision_fails_shared_contract(self) -> None:
        record = _accepted_record()
        record["decision"].update(
            accepted=True,
            outcome="reject",
            reason="candidate_verifier_failed",
        )

        audit = build_eval_run_audit([record])

        self.assertFalse(audit["checks"]["decision_records_valid"])
        self.assertFalse(audit["checks"]["decision_reasons_consistent_with_eval"])
        self.assertIn("round-001", audit["violations"]["decision_record_invalid_rounds"])
        self.assertEqual(audit["failure_buckets"], {"decision_contract": 1})
        self.assertEqual(audit["counts"]["declared_accepted_count"], 1)
        self.assertEqual(audit["counts"]["accepted_count"], 0)

    def test_impossible_reject_stages_fail_run_audit(self) -> None:
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

                audit = build_eval_run_audit([record])

                self.assertFalse(audit["checks"]["round_stages_consistent"])
                self.assertFalse(audit["checks"]["decision_records_valid"])
                self.assertTrue(audit["rounds"][0]["round_stage_reasons"])
                self.assertEqual(audit["failure_buckets"], {"decision_contract": 1})
                self.assertEqual(audit["counts"]["trained_candidate_count"], 0)
                self.assertEqual(audit["counts"]["full_eval_completed_count"], 0)

    def test_non_policy_surface_precedes_unconsumed_policy_gate_bucket(self) -> None:
        cases = (
            ("parameter_only", "reject", "parameter_only_candidate"),
            (
                "diagnostic_only",
                "diagnostic_improvement_record",
                "diagnostic_only_candidate",
            ),
            (
                "engineering_only",
                "engineering_improvement_record",
                "engineering_only_candidate",
            ),
        )
        for classification, outcome, reason in cases:
            with self.subTest(classification=classification):
                record = _accepted_record()
                record["candidate_surface"].update(
                    classification=classification,
                    eligible_for_policy_accept=False,
                    reason=reason,
                )
                record["decision"].update(
                    accepted=False,
                    outcome=outcome,
                    reason=reason,
                    surface_classification=classification,
                    comparison={},
                    consumed_gates=[],
                    gate_statuses={},
                )

                audit = build_eval_run_audit([record])

                self.assertTrue(audit["checks"]["decision_records_valid"])
                self.assertEqual(audit["failure_buckets"], {"non_policy_surface": 1})

    def test_accepted_record_missing_sealed_validation_fails_closed(self) -> None:
        record = _accepted_record()
        metrics = record["public_result"]["metrics"]
        metrics.pop("sealed_validation_gate")
        metrics["acceptance"]["required_pass_gates"].remove("sealed_validation_gate")
        record["decision"]["consumed_gates"].remove("sealed_validation_gate")
        record["decision"]["gate_statuses"].pop("sealed_validation_gate")

        audit = build_eval_run_audit([record])

        self.assertFalse(audit["checks"]["accepted_records_have_required_eval_evidence"])
        self.assertIn("round-001", audit["violations"]["accepted_rounds_missing_sealed_validation"])
        self.assertEqual(audit["failure_buckets"], {"sealed_validation": 1})

    def test_accepted_record_with_unconsumed_sealed_validation_fails_closed(self) -> None:
        record = _accepted_record()
        record["decision"]["consumed_gates"].remove("sealed_validation_gate")

        audit = build_eval_run_audit([record])

        self.assertFalse(audit["checks"]["decision_consumed_required_gates"])
        self.assertIn("round-001", audit["violations"]["accepted_rounds_missing_consumed_gates"])
        self.assertEqual(audit["failure_buckets"], {"required_gate": 1})

    def test_accepted_record_with_incomplete_sealed_validation_gate_fails_evidence_check(self) -> None:
        record = _accepted_record()
        metrics = record["public_result"]["metrics"]
        metrics["sealed_validation_gate"] = {"status": "pass"}

        audit = build_eval_run_audit([record])

        self.assertFalse(audit["checks"]["accepted_records_have_required_eval_evidence"])
        self.assertIn("round-001", audit["violations"]["accepted_rounds_missing_required_gates"])
        self.assertEqual(audit["failure_buckets"], {"sealed_validation": 1})

    def test_accepted_record_with_internal_sealed_validation_failure_fails_evidence_check(self) -> None:
        record = _accepted_record()
        metrics = record["public_result"]["metrics"]
        metrics["sealed_validation_gate"]["checkpoint_safety_status"] = "fail"

        audit = build_eval_run_audit([record])

        self.assertFalse(audit["checks"]["accepted_records_have_required_eval_evidence"])
        self.assertIn("round-001", audit["violations"]["accepted_rounds_missing_required_gates"])
        self.assertEqual(audit["failure_buckets"], {"sealed_validation": 1})

    def test_accepted_record_with_top_level_candidate_contamination_fails_evidence_check(self) -> None:
        record = _accepted_record()
        metrics = record["public_result"]["metrics"]
        metrics["candidate_code_loaded_in_eval"] = True

        audit = build_eval_run_audit([record])

        self.assertFalse(audit["checks"]["accepted_records_have_required_eval_evidence"])
        self.assertIn("round-001", audit["violations"]["accepted_rounds_missing_required_gates"])
        self.assertEqual(audit["failure_buckets"], {"sealed_validation": 1})

    def test_failed_sealed_validation_gate_counts_as_completed_eval_not_incomplete(self) -> None:
        record = _accepted_record()
        metrics = record["public_result"]["metrics"]
        metrics["sealed_validation_gate"]["status"] = "fail"
        metrics["sealed_validation_gate"]["candidate_isolation_status"] = "fail"
        record["decision"]["accepted"] = False
        record["decision"]["outcome"] = "reject"
        record["decision"]["reason"] = "required_gate_failed"
        record["decision"]["gate_statuses"]["sealed_validation_gate"] = "fail"

        audit = build_eval_run_audit([record])

        self.assertTrue(audit["checks"]["trainable_rounds_have_full_eval"])
        self.assertEqual(audit["counts"]["full_eval_completed_count"], 1)
        self.assertEqual(audit["counts"]["sealed_validation_completed_count"], 1)
        self.assertEqual(audit["counts"]["eval_incomplete_count"], 0)
        self.assertEqual(audit["failure_buckets"], {"sealed_validation": 1})

    def test_accepted_record_must_consume_all_required_pass_gates(self) -> None:
        record = _accepted_record()
        metrics = record["public_result"]["metrics"]
        metrics["new_hard_gate"] = {"status": "pass"}
        metrics["acceptance"]["required_pass_gates"].append("new_hard_gate")

        audit = build_eval_run_audit([record])

        self.assertFalse(audit["checks"]["decision_consumed_required_gates"])
        self.assertIn("round-001", audit["violations"]["accepted_rounds_missing_consumed_gates"])
        self.assertEqual(audit["failure_buckets"], {"required_gate": 1})

    def test_accepted_record_cannot_omit_failed_accept_evidence_gate_from_contract(self) -> None:
        record = _accepted_record()
        metrics = record["public_result"]["metrics"]
        metrics["accept_evidence_gate"] = {
            "status": "fail",
            "reason": "insufficient_accept_evidence",
            "observed_count": 1,
            "required_count": 3,
        }
        metrics["acceptance"]["required_pass_gates"].remove("accept_evidence_gate")
        record["decision"]["consumed_gates"].remove("accept_evidence_gate")
        record["decision"]["gate_statuses"].pop("accept_evidence_gate")

        audit = build_eval_run_audit([record])

        self.assertFalse(audit["checks"]["accepted_records_have_required_eval_evidence"])
        self.assertIn("round-001", audit["violations"]["accepted_rounds_missing_required_gates"])
        self.assertEqual(audit["failure_buckets"], {"required_gate": 1})

    def test_decision_consumed_gate_must_exist_in_public_metrics(self) -> None:
        record = _accepted_record()
        metrics = record["public_result"]["metrics"]
        metrics["acceptance"]["required_pass_gates"].append("new_hard_gate")
        record["decision"]["consumed_gates"].append("new_hard_gate")
        record["decision"]["gate_statuses"]["new_hard_gate"] = "pass"

        audit = build_eval_run_audit([record])

        self.assertFalse(audit["checks"]["decision_gate_statuses_match_metrics"])
        self.assertFalse(audit["checks"]["decision_reasons_consistent_with_eval"])
        self.assertIn("round-001", audit["violations"]["decision_gate_status_mismatch_rounds"])
        self.assertEqual(audit["failure_buckets"], {"required_gate": 1})

    def test_decision_gate_status_must_match_public_metrics_status(self) -> None:
        record = _accepted_record()
        metrics = record["public_result"]["metrics"]
        metrics["new_hard_gate"] = {"status": "fail"}
        metrics["acceptance"]["required_pass_gates"].append("new_hard_gate")
        record["decision"]["consumed_gates"].append("new_hard_gate")
        record["decision"]["gate_statuses"]["new_hard_gate"] = "pass"

        audit = build_eval_run_audit([record])

        self.assertFalse(audit["checks"]["decision_gate_statuses_match_metrics"])
        self.assertFalse(audit["checks"]["decision_reasons_consistent_with_eval"])
        self.assertIn("round-001", audit["violations"]["decision_gate_status_mismatch_rounds"])
        self.assertEqual(audit["failure_buckets"], {"required_gate": 1})

    def test_accepted_record_with_failed_extra_required_gate_fails_evidence_check(self) -> None:
        record = _accepted_record()
        metrics = record["public_result"]["metrics"]
        metrics["new_hard_gate"] = {"status": "fail", "reason": "task_gate_failed"}
        metrics["acceptance"]["required_pass_gates"].append("new_hard_gate")
        record["decision"]["consumed_gates"].append("new_hard_gate")
        record["decision"]["gate_statuses"]["new_hard_gate"] = "fail"

        audit = build_eval_run_audit([record])

        self.assertFalse(audit["checks"]["accepted_records_have_required_eval_evidence"])
        self.assertIn("round-001", audit["violations"]["accepted_rounds_missing_required_gates"])
        self.assertEqual(audit["failure_buckets"], {"required_gate": 1})

    def test_policy_threshold_requirement_uses_public_metric_gate_statuses(self) -> None:
        record = _accepted_record()
        metrics = record["public_result"]["metrics"]
        metrics["acceptance"]["required_pass_gates"].append("new_hard_gate")
        record["decision"]["accepted"] = False
        record["decision"]["outcome"] = "reject"
        record["decision"]["reason"] = "required_gate_failed"
        record["decision"]["comparison"] = {}
        record["decision"]["consumed_gates"].append("new_hard_gate")
        record["decision"]["gate_statuses"]["new_hard_gate"] = "pass"

        audit = build_eval_run_audit([record])

        self.assertTrue(audit["checks"]["decision_threshold_consumed"])
        self.assertFalse(audit["checks"]["decision_gate_statuses_match_metrics"])
        self.assertIn("round-001", audit["violations"]["decision_gate_status_mismatch_rounds"])

    def test_verifier_failed_candidate_is_not_trainable_or_eval_incomplete(self) -> None:
        record = _accepted_record()
        record["verifier"] = {"status": "fail", "reason": "static_check_failed"}
        record["jobs"] = None
        record["train_result"] = None
        record["public_result"] = None
        record["decision"] = {
            "outcome": "reject",
            "accepted": False,
            "reason": "candidate_verifier_failed",
            "surface_classification": "code_level",
            "public_score": None,
            "comparison": {},
            "consumed_gates": [],
            "gate_statuses": {},
        }

        audit = build_eval_run_audit([record])

        self.assertEqual(audit["counts"]["authored_candidate_count"], 1)
        self.assertEqual(audit["counts"]["trainable_candidate_count"], 0)
        self.assertEqual(audit["counts"]["eval_incomplete_count"], 0)
        self.assertEqual(audit["failure_buckets"], {"authoring": 1})
        self.assertTrue(audit["checks"]["trainable_rounds_have_full_eval"])

    def test_trainable_round_without_full_eval_is_counted_and_attributed(self) -> None:
        record = _accepted_record()
        record["public_result"] = None
        record["runner_error"] = {"category": "eval_incomplete"}
        record["decision"] = {
            "outcome": "reject",
            "accepted": False,
            "reason": "eval_incomplete",
            "surface_classification": "code_level",
            "public_score": None,
            "comparison": {},
            "consumed_gates": [],
            "gate_statuses": {},
        }

        audit = build_eval_run_audit([record])

        self.assertFalse(audit["checks"]["trainable_rounds_have_full_eval"])
        self.assertEqual(audit["counts"]["eval_incomplete_count"], 1)
        self.assertEqual(audit["failure_buckets"], {"public_eval": 1})
        self.assertIn("round-001", audit["violations"]["trainable_rounds_without_full_eval"])

    def test_required_gate_failed_reason_requires_failed_gate_evidence(self) -> None:
        record = _accepted_record()
        record["decision"]["accepted"] = False
        record["decision"]["outcome"] = "reject"
        record["decision"]["reason"] = "required_gate_failed"

        audit = build_eval_run_audit([record])

        self.assertFalse(audit["checks"]["decision_reasons_consistent_with_eval"])
        self.assertIn("round-001", audit["violations"]["decision_reason_inconsistent_rounds"])

    def test_accepted_record_with_failed_public_result_status_fails_audit(self) -> None:
        record = _accepted_record()
        record["public_result"]["status"] = "fail"

        audit = build_eval_run_audit([record])

        self.assertFalse(audit["checks"]["accepted_records_have_required_eval_evidence"])
        self.assertFalse(audit["checks"]["decision_reasons_consistent_with_eval"])
        self.assertIn("round-001", audit["violations"]["accepted_rounds_invalid_preconditions"])
        self.assertEqual(audit["counts"]["full_eval_completed_count"], 0)
        self.assertEqual(audit["counts"]["eval_incomplete_count"], 1)
        self.assertEqual(audit["failure_buckets"], {"public_eval": 1})

    def test_accepted_record_with_failed_verifier_status_fails_audit(self) -> None:
        record = _accepted_record()
        record["verifier"] = {"status": "fail", "reason": "static_check_failed"}

        audit = build_eval_run_audit([record])

        self.assertFalse(audit["checks"]["accepted_records_have_required_eval_evidence"])
        self.assertFalse(audit["checks"]["decision_reasons_consistent_with_eval"])
        self.assertIn("round-001", audit["violations"]["accepted_rounds_invalid_preconditions"])
        self.assertEqual(audit["counts"]["trainable_candidate_count"], 0)
        self.assertEqual(audit["counts"]["eval_incomplete_count"], 0)
        self.assertFalse(audit["checks"]["round_stages_consistent"])
        self.assertEqual(audit["failure_buckets"], {"decision_contract": 1})

    def test_full_eval_completed_requires_protocol_bound_acceptance_evidence(self) -> None:
        record = _accepted_record()
        metrics = record["public_result"]["metrics"]
        metrics["acceptance"]["score_stats"].pop("eval_protocol_hash")
        metrics["public_accept"]["score_stats"].pop("eval_protocol_hash")

        audit = build_eval_run_audit([record])

        self.assertFalse(audit["checks"]["accepted_records_have_required_eval_evidence"])
        self.assertFalse(audit["checks"]["decision_reasons_consistent_with_eval"])
        self.assertEqual(audit["counts"]["full_eval_completed_count"], 0)
        self.assertEqual(audit["counts"]["eval_incomplete_count"], 1)
        self.assertEqual(audit["failure_buckets"], {"public_eval": 1})

    def test_full_eval_completed_requires_public_score_binding(self) -> None:
        record = _accepted_record()
        record["public_result"]["public_score"] = 0.99
        record["decision"]["public_score"] = 0.99
        record["decision"]["comparison"]["public_score"] = 0.99

        audit = build_eval_run_audit([record])

        self.assertEqual(audit["counts"]["full_eval_completed_count"], 0)
        self.assertEqual(audit["counts"]["eval_incomplete_count"], 1)
        self.assertFalse(audit["checks"]["decision_records_valid"])
        self.assertEqual(audit["failure_buckets"], {"decision_contract": 1})


AUDIT_CHECKPOINT_HASH = "sha256:" + "a" * 64


def _accepted_record() -> dict[str, object]:
    initial_current_best = make_bootstrap_current_best()
    record = {
        "round_id": "round-001",
        "session_genesis_hash": TEST_SESSION_GENESIS_HASH,
        "evaluation_binding": TEST_EVALUATION_BINDING.to_public_dict(),
        "candidate": {"candidate_id": "candidate-code", "changed_components": ["reward_callable"]},
        "verifier": {"status": "pass"},
        "candidate_surface": {
            "classification": "code_level",
            "eligible_for_policy_accept": True,
            "reason": "code_level_candidate",
        },
        "jobs": {
            "train": {"job_id": "round-001-train"},
            "eval": {
                "job_id": "round-001-eval",
                **TEST_EVALUATION_PROTOCOL.to_public_dict(),
                "evaluation_binding": TEST_EVALUATION_BINDING.to_public_dict(),
                "checkpoint": {"checksum": AUDIT_CHECKPOINT_HASH},
            },
        },
        "train_result": {"job_id": "round-001-train"},
        "public_result": {
            "status": "pass",
            "public_score": 0.25,
            "metrics": _full_eval_metrics(),
        },
        "decision": {
            "outcome": "policy_improvement_accept",
            "accepted": True,
            "reason": "absolute_accept_floor",
            "surface_classification": "code_level",
            "public_score": 0.25,
            "comparison": {
                "mode": "bootstrap",
                "score_source": TEST_ACCEPTANCE_PROTOCOL.score_source,
                "acceptance_protocol_hash": TEST_ACCEPTANCE_PROTOCOL.protocol_hash,
                "threshold": 0.05,
                "candidate_score": 0.25,
                "current_best_score": 0.0,
                "delta": 0.25,
                "absolute_accept_floor": 0.05,
                "delta_met": True,
                "candidate_score_stats": {
                    "count": 3,
                    "mean": 0.25,
                    "std": 0.0,
                    "standard_error": 0.0,
                    "eval_protocol_hash": TEST_EVALUATION_PROTOCOL.protocol_hash,
                },
                "best_score_stats": {},
                "candidate_evaluation_binding_hash": (
                    TEST_EVALUATION_BINDING.binding_hash
                ),
                "current_best": initial_current_best.comparison_identity(),
                "public_score": 0.25,
            },
            "consumed_gates": [
                "accept_evidence_gate",
                "budget_accounting",
                "motion_quality_gate",
                "behavior_quality_gate",
                "sealed_validation_gate",
            ],
            "gate_statuses": {
                "accept_evidence_gate": "pass",
                "budget_accounting": "pass",
                "motion_quality_gate": "pass",
                "behavior_quality_gate": "pass",
                "sealed_validation_gate": "pass",
            },
        },
    }
    return bind_test_ledger_record(record)


def _full_eval_metrics() -> dict[str, object]:
    score_stats = {
        "count": 3,
        "mean": 0.25,
        "std": 0.0,
        "standard_error": 0.0,
        "values": [0.25, 0.25, 0.25],
        "eval_protocol_hash": TEST_EVALUATION_PROTOCOL.protocol_hash,
    }
    return {
        "evaluation_binding": TEST_EVALUATION_BINDING.to_public_dict(),
        "evaluation_subject": {
            "schema_version": "true_rl_agent.experiment.subject.v2",
            "kind": "candidate",
            "id": "candidate-code",
        },
        "subject_manifest_hash": TEST_SUBJECT_MANIFEST_HASH,
        "program_source_checksum": TEST_PROGRAM_SOURCE_CHECKSUM,
        "training_binding_hash": TEST_TRAINING_BINDING_HASH,
        "capability_profile_hash": TEST_CAPABILITY_PROFILE_HASH,
        "public_dev": {
            "score_stats": {
                "count": 1,
                "mean": 0.25,
                "std": 0.0,
                "standard_error": 0.0,
                "values": [0.25],
            },
            "seed_results": [{"run_index": 1, "status": "pass", "success_rate": 0.25}],
        },
        "public_accept": {
            "score_stats": copy.deepcopy(score_stats),
            "seed_results": [
                {"run_index": 1, "status": "pass", "success_rate": 0.25},
                {"run_index": 2, "status": "pass", "success_rate": 0.25},
                {"run_index": 3, "status": "pass", "success_rate": 0.25},
            ],
        },
        "accept_evidence_gate": {
            "status": "pass",
            "observed_count": 3,
            "required_count": 3,
        },
        "budget_accounting": {"status": "pass"},
        "motion_quality_gate": {"status": "pass"},
        "behavior_quality_gate": {"status": "pass"},
        "sealed_validation": {
            "schema_version": "true_rl_agent.sealed_validation.evidence.v3",
            "status": "pass",
            "reason": "test_holdout_passed",
            "expected_seed_count": 7,
            "observed_seed_count": 7,
            "eval_protocol_hash": TEST_EVALUATION_PROTOCOL.protocol_hash,
            "evaluation_binding_hash": TEST_EVALUATION_BINDING.binding_hash,
            "evaluation_subject": {
                "schema_version": "true_rl_agent.experiment.subject.v2",
                "kind": "candidate",
                "id": "candidate-code",
            },
            "subject_manifest_hash": TEST_SUBJECT_MANIFEST_HASH,
            "program_source_checksum": TEST_PROGRAM_SOURCE_CHECKSUM,
            "training_binding_hash": TEST_TRAINING_BINDING_HASH,
            "capability_profile_hash": TEST_CAPABILITY_PROFILE_HASH,
            "checkpoint_hash": AUDIT_CHECKPOINT_HASH,
            "score_stats": {
                "count": 7,
                "mean": 0.25,
                "std": 0.0,
                "standard_error": 0.0,
                "eval_protocol_hash": TEST_EVALUATION_PROTOCOL.protocol_hash,
            },
        },
        "sealed_validation_gate": {
            "status": "pass",
            "checkpoint_safety_status": "pass",
            "candidate_isolation_status": "pass",
            "candidate_code_loaded_in_eval": False,
            "candidate_package_loaded_in_eval": False,
            "candidate_path_argument_present": False,
            "eval_protocol_hash_match_status": "pass",
            "evaluation_binding_hash_match_status": "pass",
            "checkpoint_hash_match_status": "pass",
            "holdout_rollout_status": "pass",
            "reason": "sealed_validation_complete",
        },
        "acceptance": {
            "score_source": TEST_ACCEPTANCE_PROTOCOL.score_source,
            "public_score_binding": TEST_ACCEPTANCE_PROTOCOL.public_score_binding,
            "score_stats": score_stats,
            "min_score_count": 3,
            "missing_required_score": False,
            "required_pass_gates": [
                "accept_evidence_gate",
                "budget_accounting",
                "motion_quality_gate",
                "behavior_quality_gate",
                "sealed_validation_gate",
            ],
            "absolute_accept_floor": 0.05,
            "min_delta": 0.05,
            "acceptance_protocol_hash": TEST_ACCEPTANCE_PROTOCOL.protocol_hash,
        },
    }


if __name__ == "__main__":
    unittest.main()
