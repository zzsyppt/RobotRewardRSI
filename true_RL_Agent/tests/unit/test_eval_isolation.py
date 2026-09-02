from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from true_rl_agent.runner.eval_isolation import (
    build_candidate_isolation_gate,
    reject_candidate_path_arguments,
    validate_sealed_validation_gate,
)


class EvalIsolationTests(unittest.TestCase):
    def test_rejects_candidate_path_arguments(self) -> None:
        for argv in (
            ["--candidate", "artifact://candidate-001"],
            ["--candidate=artifact://candidate-001"],
            ["--candidate-path", "D:/tmp/candidate"],
            ["--candidate-dir", "candidate_pkg"],
        ):
            with self.subTest(argv=argv):
                with self.assertRaisesRegex(ValueError, "candidate path argument is not allowed"):
                    reject_candidate_path_arguments(argv)

    def test_builds_and_validates_candidate_isolation_gate(self) -> None:
        gate = _sealed_gate()

        self.assertEqual(gate["status"], "pass")
        self.assertEqual(gate["candidate_isolation_status"], "pass")
        self.assertIs(validate_sealed_validation_gate(gate, **_evidence_args()), gate)

        failed = _sealed_gate(candidate_code_loaded=True)
        self.assertEqual(failed["status"], "fail")
        self.assertEqual(failed["candidate_isolation_status"], "fail")
        with self.assertRaisesRegex(ValueError, "candidate isolation failed"):
            validate_sealed_validation_gate(failed, **_evidence_args())

    def test_rejects_self_contradictory_candidate_isolation_gate(self) -> None:
        base_gate = _sealed_gate()

        for key in (
            "candidate_code_loaded_in_eval",
            "candidate_package_loaded_in_eval",
            "candidate_path_argument_present",
            "candidate_package_loaded",
        ):
            gate = {**base_gate, key: True}
            gate["status"] = "pass"
            gate["candidate_isolation_status"] = "pass"
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, "candidate isolation failed"):
                    validate_sealed_validation_gate(gate, **_evidence_args())

    def test_rejects_incomplete_candidate_isolation_gate(self) -> None:
        for key in (
            "candidate_code_loaded_in_eval",
            "candidate_package_loaded_in_eval",
            "candidate_path_argument_present",
        ):
            gate = _sealed_gate()
            del gate[key]
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, f"sealed_validation_gate {key} must be boolean"):
                    validate_sealed_validation_gate(gate, **_evidence_args())


def _sealed_gate(*, candidate_code_loaded: bool = False) -> dict[str, object]:
    gate = build_candidate_isolation_gate(candidate_code_loaded=candidate_code_loaded)
    gate.update(
        {
            "checkpoint_safety_status": "pass",
            "checkpoint_hash_match_status": "pass",
            "eval_protocol_hash_match_status": "pass",
            "evaluation_binding_hash_match_status": "pass",
            "holdout_rollout_status": "pass",
            "reason": gate["reason"],
        }
    )
    return gate


def _evidence_args() -> dict[str, object]:
    protocol_hash = "1" * 64
    evaluation_binding_hash = "2" * 64
    return {
        "eval_protocol_hash": protocol_hash,
        "evaluation_binding_hash": evaluation_binding_hash,
        "rollout_evidence": {
            "schema_version": "true_rl_agent.sealed_validation.evidence.v3",
            "status": "pass",
            "reason": "test_holdout_passed",
            "expected_seed_count": 3,
            "observed_seed_count": 3,
            "eval_protocol_hash": protocol_hash,
            "evaluation_binding_hash": evaluation_binding_hash,
            "evaluation_subject": {
                "schema_version": "true_rl_agent.experiment.subject.v2",
                "kind": "candidate",
                "id": "candidate-001",
            },
            "program_source_checksum": "sha256:" + "4" * 64,
            "training_binding_hash": "5" * 64,
            "capability_profile_hash": "6" * 64,
            "checkpoint_hash": "sha256:" + "3" * 64,
            "score_stats": {
                "count": 3,
                "mean": 0.5,
                "std": 0.0,
                "standard_error": 0.0,
                "eval_protocol_hash": protocol_hash,
            },
        },
    }


if __name__ == "__main__":
    unittest.main()
