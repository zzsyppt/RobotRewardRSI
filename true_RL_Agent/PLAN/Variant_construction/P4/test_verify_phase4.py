from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

P4_DIR = Path(__file__).resolve().parent
if str(P4_DIR) not in sys.path:
    sys.path.insert(0, str(P4_DIR))

import verify_phase4 as verifier


class Phase4VerifierTests(unittest.TestCase):
    def test_frozen_local_evidence_matches_current_product(self) -> None:
        result = verifier.verify()
        self.assertEqual(result["status"], "pass")
        self.assertTrue(result["loop_verified"])
        self.assertTrue(result["improvement_observed"])
        self.assertEqual(result["runtime"]["status"], "not_rechecked")

    def test_snapshot_tamper_is_rejected(self) -> None:
        snapshot = json.loads(
            verifier.SNAPSHOT_PATH.read_text(encoding="utf-8")
        )
        tampered = copy.deepcopy(snapshot)
        tampered["validation"]["generation_count"] = 4
        with tempfile.TemporaryDirectory(prefix="p4-snapshot-") as raw_directory:
            path = Path(raw_directory) / "snapshot.json"
            path.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaisesRegex(AssertionError, "snapshot hash mismatch"):
                verifier.verify(path)

    def test_product_source_drift_is_rejected(self) -> None:
        original = verifier._normalized_text_hash

        def drift_one(path: Path) -> str:
            if path.name == "generation_session.py":
                return "0" * 64
            return original(path)

        with mock.patch.object(
            verifier,
            "_normalized_text_hash",
            side_effect=drift_one,
        ):
            with self.assertRaisesRegex(
                AssertionError,
                "product source drift: .*generation_session.py",
            ):
                verifier.verify()


if __name__ == "__main__":
    unittest.main()
