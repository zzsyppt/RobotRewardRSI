from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import verify_phase2_snapshot as verifier
from true_rl_agent.evaluation.binding import canonical_json_hash


class Phase2SnapshotVerifierTests(unittest.TestCase):
    def test_frozen_snapshot_verifies(self) -> None:
        result = verifier.verify()
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["support_cardinalities"], [2, 4, 6])

    def test_product_contract_tamper_is_rejected(self) -> None:
        snapshot = copy.deepcopy(verifier.emit_current())
        snapshot["verification_contract"]["legacy_cell"]["support_count"] = 99
        snapshot["receipt_hash"] = canonical_json_hash(verifier._without_hash(snapshot))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "snapshot.json"
            path.write_text(json.dumps(snapshot), encoding="utf-8")
            with self.assertRaisesRegex(AssertionError, "current contract"):
                verifier.verify(path)

    def test_captured_head_is_provenance_not_live_gate(self) -> None:
        snapshot = copy.deepcopy(verifier.emit_current())
        snapshot["provenance"]["captured_repository_head"] = "0" * 40
        snapshot["receipt_hash"] = canonical_json_hash(verifier._without_hash(snapshot))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "snapshot.json"
            path.write_text(json.dumps(snapshot), encoding="utf-8")
            self.assertEqual(verifier.verify(path)["status"], "pass")

    def test_product_hash_is_line_ending_independent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sample.py"
            path.write_bytes(b"first\nsecond\n")
            lf_hash = verifier._normalized_text_hash(path)
            path.write_bytes(b"first\r\nsecond\r\n")
            self.assertEqual(verifier._normalized_text_hash(path), lf_hash)

    def test_registry_derived_boundary_catches_new_concrete_robot_names(self) -> None:
        source = (
            '_HARDCODED = ("pelvis", "left_ankle_pitch_link", '
            '"right_ankle_pitch_link")\n'
        )
        hits = verifier._generic_stair_marker_hits({"synthetic.py": source})
        self.assertEqual(
            hits,
            [
                "synthetic.py:robot_name:left_ankle_pitch_link",
                "synthetic.py:robot_name:pelvis",
                "synthetic.py:robot_name:right_ankle_pitch_link",
            ],
        )


if __name__ == "__main__":
    unittest.main()
