from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from true_rl_agent.tasks.stair_climbing.baseline_adapter import (
    build_stair_runtime_baseline_manifest,
)
from true_rl_agent.tasks.stair_climbing.phase0_contract import (
    CheckpointFileVerification,
    StairPhase0Bundle,
)


class StairBaselineAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = StairPhase0Bundle.load(PROJECT_ROOT)
        cls.report = cls.bundle.load_report(require_admitted=True)

    def test_runtime_manifest_is_derived_from_frozen_phase0_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "official-g1.pt"
            checkpoint_path.write_bytes(b"test-placeholder")
            verification = CheckpointFileVerification(
                path=checkpoint_path.resolve(),
                sha256=self.bundle.baseline.checkpoint_sha256,
                size_bytes=self.bundle.baseline.checkpoint_size_bytes,
            )
            with patch(
                "true_rl_agent.tasks.stair_climbing.baseline_adapter."
                "verify_checkpoint_file",
                return_value=verification,
            ):
                first = build_stair_runtime_baseline_manifest(
                    bundle=self.bundle,
                    checkpoint_path=checkpoint_path,
                )
                second = build_stair_runtime_baseline_manifest(
                    bundle=self.bundle,
                    checkpoint_path=checkpoint_path,
                )

        source_manifest = self.bundle.baseline.to_dict()
        source = source_manifest["source"]
        compatibility = source_manifest["compatibility"]
        self.assertEqual(first, second)
        self.assertEqual(
            first.source_binding.source_manifest_hash,
            self.bundle.baseline.manifest_hash,
        )
        self.assertEqual(
            first.source_binding.trust_anchor_hash,
            self.bundle.descriptor.descriptor_hash,
        )
        self.assertEqual(
            first.source_binding.admission_evidence_hash,
            self.report.report_hash,
        )
        self.assertEqual(first.provenance.source_run_id, source["artifact_ref"])
        self.assertEqual(first.provenance.source_task_id, source["source_task_id"])
        self.assertEqual(first.provenance.created_by, source["provider"])
        self.assertEqual(
            first.compatibility.observation_schema_hash,
            compatibility["policy_observation_schema_hash"],
        )
        self.assertEqual(
            first.to_policy_initialization().baseline.baseline_manifest_hash,
            first.manifest_hash,
        )

    def test_wrong_checkpoint_bytes_are_rejected_during_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "wrong.pt"
            checkpoint_path.write_bytes(b"wrong")

            with self.assertRaisesRegex(
                ValueError,
                "checkpoint size mismatch",
            ):
                build_stair_runtime_baseline_manifest(
                    bundle=self.bundle,
                    checkpoint_path=checkpoint_path,
                )


if __name__ == "__main__":
    unittest.main()
