from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import verify_phase3a as verifier
from true_rl_agent.evaluation.binding import canonical_json_hash


class Phase3ASnapshotVerifierTests(unittest.TestCase):
    def test_old_snapshot_remains_valid_historical_evidence(self) -> None:
        result = verifier.verify()
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["phase_status"], "no_go")

    def test_strict_go_gate_rejects_the_reference_failure(self) -> None:
        with self.assertRaisesRegex(AssertionError, "not GO"):
            verifier.verify(require_go=True)

    def test_captured_head_is_provenance_not_a_live_gate(self) -> None:
        snapshot = json.loads(verifier.SNAPSHOT_PATH.read_text(encoding="utf-8"))
        snapshot["provenance"]["captured_repository_head"] = "0" * 40
        snapshot["receipt_hash"] = canonical_json_hash(
            verifier._without_receipt_hash(snapshot)
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "snapshot.json"
            path.write_text(json.dumps(snapshot), encoding="utf-8")
            self.assertEqual(verifier.verify(path)["status"], "pass")

if __name__ == "__main__":
    unittest.main()
