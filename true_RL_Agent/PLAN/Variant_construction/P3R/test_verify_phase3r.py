from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import verify_phase3r as verifier


class Phase3RVerifierTests(unittest.TestCase):
    def test_snapshot_tamper_is_rejected(self) -> None:
        tampered = json.loads(verifier.SNAPSHOT_PATH.read_text(encoding="utf-8"))
        tampered["contract"]["protocol_state"] = "active"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "snapshot.json"
            path.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaisesRegex(AssertionError, "receipt hash mismatch"):
                verifier.verify(path)

    def test_malformed_historical_source_entry_is_rejected(self) -> None:
        tampered = json.loads(verifier.SNAPSHOT_PATH.read_text(encoding="utf-8"))
        first_path = next(iter(tampered["product_tree"]["file_hashes"]))
        tampered["product_tree"]["file_hashes"][first_path] = "not-a-digest"
        tampered["product_tree"]["product_tree_hash"] = verifier.canonical_json_hash(
            tampered["product_tree"]["file_hashes"]
        )
        tampered["receipt_hash"] = verifier.canonical_json_hash(
            verifier._without_receipt_hash(tampered)
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "snapshot.json"
            path.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaisesRegex(AssertionError, "provenance entry"):
                verifier.verify(path)

    def test_frozen_historical_evidence_survives_later_protocol_activation(self) -> None:
        result = verifier.verify()
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["protocol_state"], "draft")

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
