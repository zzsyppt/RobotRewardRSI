from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import verify_phase3r as verifier
from true_rl_agent.evaluation.binding import canonical_json_hash


class Phase3RVerifierTests(unittest.TestCase):
    @staticmethod
    def _synthetic_current() -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": verifier.SNAPSHOT_SCHEMA_VERSION,
            "contract": {
                "cell_variant_id": "stair.g1_37dof.unified_v1",
                "protocol_state": "draft",
            },
            "boundaries": {"retired_is_irregular_cell_hits": []},
            "product_tree": {"product_tree_hash": "a" * 64},
            "runtime_smoke": {"receipt_hash": "b" * 64},
        }
        payload["receipt_hash"] = canonical_json_hash(payload)
        return payload

    def test_snapshot_tamper_is_rejected(self) -> None:
        current = self._synthetic_current()
        tampered = copy.deepcopy(current)
        tampered["contract"]["protocol_state"] = "active"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "snapshot.json"
            path.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaisesRegex(AssertionError, "receipt hash mismatch"):
                verifier.verify(path)

    def test_current_contract_drift_is_rejected(self) -> None:
        current = self._synthetic_current()
        expected = copy.deepcopy(current)
        expected["contract"]["protocol_state"] = "active"
        expected["receipt_hash"] = canonical_json_hash(
            verifier._without_receipt_hash(expected)
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "snapshot.json"
            path.write_text(json.dumps(expected), encoding="utf-8")
            with (
                mock.patch.object(
                    verifier,
                    "emit_current",
                    return_value=copy.deepcopy(current),
                ),
                self.assertRaisesRegex(AssertionError, "current contract"),
            ):
                verifier.verify(path)

    def test_normalized_source_hash_ignores_line_endings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sample.py"
            path.write_bytes(b"first\nsecond\n")
            expected = verifier._normalized_text_hash(path)
            path.write_bytes(b"first\r\nsecond\r\n")
            self.assertEqual(verifier._normalized_text_hash(path), expected)

    def test_runtime_receipt_tamper_is_rejected(self) -> None:
        receipt = json.loads(
            verifier.RUNTIME_SMOKE_PATH.read_text(encoding="utf-8")
        )
        receipt["training_smoke"]["training"]["budget_steps"] = 129
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "runtime.json"
            path.write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaisesRegex(AssertionError, "receipt hash mismatch"):
                verifier._runtime_receipt({}, path)


if __name__ == "__main__":
    unittest.main()
