from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import verify_phase3a as verifier
from true_rl_agent.evaluation.binding import canonical_json_hash


class Phase3ASnapshotVerifierTests(unittest.TestCase):
    @staticmethod
    def _synthetic_current() -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": verifier.SNAPSHOT_SCHEMA_VERSION,
            "provenance": {"captured_repository_head": "1" * 40},
            "verification_contract": {
                "cell_variant_id": "stair.g1_37dof.irregular_v1",
                "ordered_suite": {"suite_id": "current", "cells": [{"id": 1}]},
            },
            "product_tree": {"product_tree_hash": "2" * 64},
            "runtime_evidence": {"reference_qualification": {"status": "fail"}},
        }
        payload["receipt_hash"] = canonical_json_hash(payload)
        return payload

    @staticmethod
    def _failed_reference() -> dict[str, object]:
        return {
            "status": "fail",
            "cell_loop_ready": False,
            "sealed_validation_status": "fail",
            "formal_training_executed": True,
            "required_gate_statuses": {"reference_capability_gate": "fail"},
        }

    def test_old_snapshot_is_rejected_after_reward_and_profile_root_fix(self) -> None:
        with self.assertRaisesRegex(AssertionError, "source closure drift"):
            verifier.verify()

    def test_strict_go_gate_rejects_the_reference_failure(self) -> None:
        snapshot = self._synthetic_current()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "snapshot.json"
            path.write_text(json.dumps(snapshot), encoding="utf-8")
            with (
                mock.patch.object(
                    verifier,
                    "emit_current",
                    return_value=copy.deepcopy(snapshot),
                ),
                mock.patch.object(
                    verifier,
                    "_assert_receipt",
                    return_value=self._failed_reference(),
                ),
                self.assertRaisesRegex(AssertionError, "not GO"),
            ):
                verifier.verify(path, require_go=True)

    def test_current_contract_tamper_is_rejected(self) -> None:
        current = self._synthetic_current()
        snapshot = copy.deepcopy(current)
        snapshot["verification_contract"]["ordered_suite"]["suite_id"] = (
            "tampered"
        )
        snapshot["receipt_hash"] = canonical_json_hash(
            verifier._without_receipt_hash(snapshot)
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "snapshot.json"
            path.write_text(json.dumps(snapshot), encoding="utf-8")
            with (
                mock.patch.object(
                    verifier,
                    "emit_current",
                    return_value=copy.deepcopy(current),
                ),
                self.assertRaisesRegex(AssertionError, "current verification_contract"),
            ):
                verifier.verify(path)

    def test_captured_head_is_provenance_not_a_live_gate(self) -> None:
        current = self._synthetic_current()
        snapshot = copy.deepcopy(current)
        snapshot["provenance"]["captured_repository_head"] = "0" * 40
        snapshot["receipt_hash"] = canonical_json_hash(
            verifier._without_receipt_hash(snapshot)
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "snapshot.json"
            path.write_text(json.dumps(snapshot), encoding="utf-8")
            with (
                mock.patch.object(
                    verifier,
                    "emit_current",
                    return_value=copy.deepcopy(current),
                ),
                mock.patch.object(
                    verifier,
                    "_assert_receipt",
                    return_value=self._failed_reference(),
                ),
            ):
                self.assertEqual(verifier.verify(path)["status"], "pass")

    def test_normalized_source_hash_ignores_line_endings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sample.py"
            path.write_bytes(b"first\nsecond\n")
            expected = verifier._normalized_text_hash(path)
            path.write_bytes(b"first\r\nsecond\r\n")
            self.assertEqual(verifier._normalized_text_hash(path), expected)


if __name__ == "__main__":
    unittest.main()
