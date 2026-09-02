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

    def test_product_source_closure_is_historical_provenance(self) -> None:
        with mock.patch.object(
            verifier,
            "_normalized_text_hash",
            side_effect=AssertionError("current source bytes must not be read"),
        ):
            result = verifier._verify_product_sources(
                json.loads(verifier.SNAPSHOT_PATH.read_text(encoding="utf-8"))
            )
        self.assertEqual(result["identity_semantics"], "historical_provenance")

    def test_malformed_recorded_product_hash_is_rejected(self) -> None:
        with self.assertRaisesRegex(AssertionError, "recorded product source hash"):
            verifier._verify_product_sources(
                {
                    "product_source_hash_mode": "normalized_lf_sha256",
                    "product_source_hashes": {"src/example.py": "not-a-hash"},
                }
            )


if __name__ == "__main__":
    unittest.main()
