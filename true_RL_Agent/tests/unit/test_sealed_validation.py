from __future__ import annotations

import ast
import copy
import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tests.fakes.decision_records import (
    TEST_ACCEPTANCE_PROTOCOL,
    TEST_SESSION_GENESIS_HASH,
    make_bootstrap_current_best,
    valid_accepted_ledger_record,
)
from true_rl_agent.agent_core.eval_run_audit import build_eval_run_audit
from true_rl_agent.evaluation.decision_record_contract import validate_decision_record
from true_rl_agent.evaluation.sealed_validation import (
    build_sealed_rollout_evidence,
    build_sealed_validation_gate,
    validate_sealed_rollout_evidence,
    validate_sealed_validation_gate_contract,
)
from true_rl_agent.evaluation.subject import ExperimentSubject
from true_rl_agent.memory import PublicContextBuilder
from true_rl_agent.runner.eval_isolation import build_candidate_isolation_gate


class SealedValidationTests(unittest.TestCase):
    def test_real_eval_executes_runner_owned_sealed_rollout_lane(self) -> None:
        tree = ast.parse(
            (PROJECT_ROOT / "scripts" / "isaaclab_rsl_rl_eval.py").read_text(
                encoding="utf-8"
            )
        )
        run_public_eval = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "run_public_eval"
        )
        split_names = []
        for node in ast.walk(run_public_eval):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Name) or node.func.id != "run_rollout":
                continue
            split = next(
                (
                    keyword.value.value
                    for keyword in node.keywords
                    if keyword.arg == "split_name"
                    and isinstance(keyword.value, ast.Constant)
                    and isinstance(keyword.value.value, str)
                ),
                None,
            )
            if split is not None:
                split_names.append(split)

        self.assertCountEqual(
            split_names,
            ["public_dev", "public_accept", "sealed_validation"],
        )

    def test_sealed_gate_is_derived_from_bound_complete_rollout_evidence(self) -> None:
        protocol_hash = "1" * 64
        binding_hash = "a" * 64
        checkpoint_hash = "sha256:" + "c" * 64
        source_checksum = "sha256:" + "d" * 64
        training_binding_hash = "e" * 64
        capability_profile_hash = "f" * 64
        evidence = build_sealed_rollout_evidence(
            score_stats={
                "count": 7,
                "mean": 0.5,
                "std": 0.0,
                "standard_error": 0.0,
                "eval_protocol_hash": protocol_hash,
            },
            expected_seed_count=7,
            eval_protocol_hash=protocol_hash,
            evaluation_binding_hash=binding_hash,
            evaluation_subject=ExperimentSubject.candidate("candidate-001"),
            program_source_checksum=source_checksum,
            training_binding_hash=training_binding_hash,
            capability_profile_hash=capability_profile_hash,
            checkpoint_hash=checkpoint_hash,
            task_validation_status="pass",
            reason="task_holdout_passed",
        )
        gate = build_sealed_validation_gate(
            rollout_evidence=evidence,
            checkpoint_safety_status="pass",
            checkpoint_hash_match_status="pass",
            candidate_isolation_gate=build_candidate_isolation_gate(),
            expected_eval_protocol_hash=protocol_hash,
            expected_evaluation_binding_hash=binding_hash,
        )

        validation = validate_sealed_validation_gate_contract(
            gate,
            rollout_evidence=evidence,
            eval_protocol_hash=protocol_hash,
            evaluation_binding_hash=binding_hash,
            expected_seed_count=7,
            expected_checkpoint_hash=checkpoint_hash,
        )
        self.assertTrue(validation.valid, validation.reasons)
        self.assertEqual(gate["holdout_rollout_status"], "pass")

        forged = copy.deepcopy(evidence)
        forged["observed_seed_count"] = 1
        self.assertFalse(
            validate_sealed_rollout_evidence(
                forged,
                expected_seed_count=7,
                expected_eval_protocol_hash=protocol_hash,
                expected_evaluation_binding_hash=binding_hash,
                expected_evaluation_subject=ExperimentSubject.candidate(
                    "candidate-001"
                ),
                expected_checkpoint_hash=checkpoint_hash,
            ).complete
        )

        legacy_evidence = copy.deepcopy(evidence)
        legacy_evidence["schema_version"] = (
            "true_rl_agent.sealed_validation.evidence.v1"
        )
        legacy_evidence.pop("evaluation_binding_hash")
        legacy_evidence.pop("evaluation_subject")
        legacy_gate = copy.deepcopy(gate)
        legacy_gate.pop("evaluation_binding_hash_match_status")
        self.assertFalse(
            validate_sealed_validation_gate_contract(
                legacy_gate,
                rollout_evidence=legacy_evidence,
                eval_protocol_hash=protocol_hash,
                expected_seed_count=7,
                expected_checkpoint_hash=checkpoint_hash,
            ).valid
        )
        self.assertFalse(
            validate_sealed_validation_gate_contract(
                legacy_gate,
                rollout_evidence=legacy_evidence,
                eval_protocol_hash=protocol_hash,
                evaluation_binding_hash=binding_hash,
                expected_seed_count=7,
                expected_checkpoint_hash=checkpoint_hash,
            ).valid
        )

    def test_sealed_evidence_rejects_noncanonical_actual_and_expected_hashes(
        self,
    ) -> None:
        protocol_hash = "1" * 64
        binding_hash = "2" * 64
        checkpoint_hash = "sha256:" + "3" * 64
        source_checksum = "sha256:" + "4" * 64
        training_binding_hash = "5" * 64
        capability_profile_hash = "6" * 64
        evidence = build_sealed_rollout_evidence(
            score_stats={
                "count": 7,
                "mean": 0.5,
                "std": 0.0,
                "standard_error": 0.0,
                "eval_protocol_hash": protocol_hash,
            },
            expected_seed_count=7,
            eval_protocol_hash=protocol_hash,
            evaluation_binding_hash=binding_hash,
            evaluation_subject=ExperimentSubject.candidate("candidate-001"),
            program_source_checksum=source_checksum,
            training_binding_hash=training_binding_hash,
            capability_profile_hash=capability_profile_hash,
            checkpoint_hash=checkpoint_hash,
            task_validation_status="pass",
            reason="task_holdout_passed",
        )

        for field_name in (
            "eval_protocol_hash",
            "evaluation_binding_hash",
            "checkpoint_hash",
        ):
            with self.subTest(actual_field=field_name):
                forged = copy.deepcopy(evidence)
                forged[field_name] = "not-a-sha256"
                if field_name == "eval_protocol_hash":
                    forged["score_stats"]["eval_protocol_hash"] = "not-a-sha256"
                validation = validate_sealed_rollout_evidence(forged)
                self.assertFalse(validation.complete, validation.reasons)
                self.assertFalse(validation.passed)
                gate = build_sealed_validation_gate(
                    rollout_evidence=forged,
                    checkpoint_safety_status="pass",
                    checkpoint_hash_match_status="pass",
                    candidate_isolation_gate=build_candidate_isolation_gate(),
                    expected_eval_protocol_hash=protocol_hash,
                    expected_evaluation_binding_hash=binding_hash,
                )
                self.assertEqual(gate["status"], "fail")

        expected_hash_cases = {
            "expected_eval_protocol_hash": "not-a-sha256",
            "expected_evaluation_binding_hash": "not-a-sha256",
            "expected_checkpoint_hash": "not-a-sha256",
        }
        expected_args = {
            "expected_seed_count": 7,
            "expected_eval_protocol_hash": protocol_hash,
            "expected_evaluation_binding_hash": binding_hash,
            "expected_evaluation_subject": ExperimentSubject.candidate(
                "candidate-001"
            ),
            "expected_checkpoint_hash": checkpoint_hash,
        }
        for argument_name, malformed_value in expected_hash_cases.items():
            with self.subTest(expected_argument=argument_name):
                arguments = dict(expected_args)
                arguments[argument_name] = malformed_value
                validation = validate_sealed_rollout_evidence(
                    evidence,
                    **arguments,
                )
                self.assertFalse(validation.complete, validation.reasons)
                self.assertFalse(validation.passed)

        for argument_name in (
            "expected_eval_protocol_hash",
            "expected_evaluation_binding_hash",
        ):
            with self.subTest(gate_expected_argument=argument_name):
                arguments = {
                    "expected_eval_protocol_hash": protocol_hash,
                    "expected_evaluation_binding_hash": binding_hash,
                }
                arguments[argument_name] = "not-a-sha256"
                gate = build_sealed_validation_gate(
                    rollout_evidence=evidence,
                    checkpoint_safety_status="pass",
                    checkpoint_hash_match_status="pass",
                    candidate_isolation_gate=build_candidate_isolation_gate(),
                    **arguments,
                )
                self.assertEqual(gate["status"], "fail")

    def test_checkpoint_only_gate_cannot_count_as_full_eval_or_accept(self) -> None:
        record = valid_accepted_ledger_record(
            round_id="round-001",
            candidate_id="candidate-001",
            public_score=0.50,
        )
        record["public_result"]["metrics"].pop("sealed_validation")

        validation = validate_decision_record(
            record,
            acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
        )
        audit = build_eval_run_audit(
            [record],
            acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
            initial_current_best=make_bootstrap_current_best(),
            expected_session_genesis_hash=TEST_SESSION_GENESIS_HASH,
        )

        self.assertFalse(validation.valid_policy_accept)
        self.assertFalse(validation.acceptance_evidence.sealed_validation_complete)
        self.assertEqual(audit["counts"]["full_eval_completed_count"], 0)
        self.assertEqual(audit["counts"]["sealed_validation_completed_count"], 0)
        self.assertFalse(audit["checks"]["trainable_rounds_have_full_eval"])

    def test_prompt_exposes_only_bounded_sealed_gate_summary(self) -> None:
        record = valid_accepted_ledger_record(
            round_id="round-001",
            candidate_id="candidate-001",
            public_score=0.50,
        )
        payload = PublicContextBuilder(
            acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
            initial_current_best=make_bootstrap_current_best(),
            session_genesis_hash=TEST_SESSION_GENESIS_HASH,
        ).build_prompt_payload([record])
        payload_text = json.dumps(payload, sort_keys=True)

        self.assertIn("sealed_validation_gate", payload_text)
        self.assertNotIn("expected_seed_count", payload_text)
        self.assertNotIn("observed_seed_count", payload_text)
        self.assertNotIn("test-checkpoint", payload_text)


if __name__ == "__main__":
    unittest.main()
