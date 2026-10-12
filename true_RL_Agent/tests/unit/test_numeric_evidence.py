from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from true_rl_agent.runner.numeric_evidence import (
    NumericEvidenceError, NumericEvidenceWriter, iter_numeric_evidence,
    load_numeric_evidence_manifest,
)


class NumericEvidenceTests(unittest.TestCase):
    def test_publication_round_trip_and_corrupt_chunk(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "capture"
            writer = NumericEvidenceWriter(root, {"checkpoint": "fixture"})
            arrays = {"state": np.arange(24, dtype=np.float32).reshape(2, 4, 3),
                      "valid": np.array([[True, False], [True, True]])}
            writer.write_chunk(arrays, {"seed_index": 1})
            self.assertFalse((root / "manifest.json").exists())
            ref = writer.complete({"coverage": {"rows": 8}})
            metadata, restored = next(iter_numeric_evidence(ref))
            self.assertEqual(metadata, {"seed_index": 1})
            for name, value in arrays.items():
                np.testing.assert_array_equal(restored[name], value)
                self.assertEqual(restored[name].dtype, value.dtype)
            with self.assertRaisesRegex(NumericEvidenceError, "already published"):
                writer.complete({})
            chunk = root / load_numeric_evidence_manifest(ref)["chunks"][0]["file"]
            chunk.write_bytes(chunk.read_bytes()[:-3])
            with self.assertRaisesRegex(NumericEvidenceError, "hash or size"):
                list(iter_numeric_evidence(ref))

    def test_failed_budget_never_publishes_complete_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "capture"
            writer = NumericEvidenceWriter(root, {}, max_bytes=1)
            with self.assertRaisesRegex(NumericEvidenceError, "byte budget"):
                writer.write_chunk({"state": np.array([1.0])}, {})
            with self.assertRaisesRegex(NumericEvidenceError, "no chunks"):
                writer.complete({})
            self.assertFalse((root / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
