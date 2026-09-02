from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

P5_DIR = Path(__file__).resolve().parent
if str(P5_DIR) not in sys.path:
    sys.path.insert(0, str(P5_DIR))

import verify_phase5 as verifier


class Phase5VerifierTests(unittest.TestCase):
    def test_foundation_matches_current_product_and_real_runtime(self) -> None:
        result = verifier.verify(foundation_only=True)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["cell"]["cell_variant_id"], "stair.go2_12dof.unified_v2")
        self.assertEqual(
            result["cell"]["training_predecessor"]["cell_variant_id"],
            "stair.go2_12dof.unified_v1",
        )
        self.assertEqual(
            result["cell"]["g1_v2"]["cell_variant_id"],
            "stair.g1_37dof.unified_v2",
        )
        self.assertEqual(
            result["cell"]["g1_v2"]["training_predecessor"]["cell_variant_id"],
            "stair.g1_37dof.unified_v1",
        )
        self.assertEqual(result["cell"]["formal_episode_count"], 1080)
        self.assertEqual(result["runtime"]["oracle_sample_count"], 2880)
        self.assertGreater(
            result["runtime"]["curriculum_activation"]["up_requests"], 0
        )
        self.assertGreater(
            result["runtime"]["curriculum_activation"]["up_requests_c1"], 0
        )
        self.assertEqual(result["genesis_load_only"]["status"], "not_checked")

    def test_stage_tamper_is_rejected(self) -> None:
        original = json.loads(
            verifier.ORACLE_STAGE_RECEIPT.read_text(encoding="utf-8")
        )
        tampered = copy.deepcopy(original)
        tampered["reference_reward_equivalence"][
            "maximum_absolute_total_rate_difference"
        ] = 1.0
        with tempfile.TemporaryDirectory(prefix="p5-stage-") as raw_directory:
            path = Path(raw_directory) / "oracle.json"
            path.write_text(json.dumps(tampered), encoding="utf-8")
            with mock.patch.object(verifier, "ORACLE_STAGE_RECEIPT", path):
                with self.assertRaisesRegex(
                    AssertionError,
                    "Phase 5 oracle stage stage_hash mismatch",
                ):
                    verifier.verify(foundation_only=True)

    def test_later_robot_row_does_not_invalidate_phase5(self) -> None:
        established = verifier.active_unified_stair_cell_variant_ids()
        with mock.patch.object(
            verifier,
            "active_unified_stair_cell_variant_ids",
            return_value=(*established, "stair.future_robot.unified_v1"),
        ):
            result = verifier.verify(foundation_only=True)
        self.assertEqual(result["status"], "pass")

    def test_go_qualification_replays_and_raw_phase_tamper_is_rejected(
        self,
    ) -> None:
        result = verifier.verify()
        self.assertEqual(result["qualification"]["status"], "go")
        self.assertTrue(result["qualification"]["go"])
        self.assertTrue(result["qualification"]["cell_loop_ready"])
        self.assertEqual(result["g1_v2_qualification"]["status"], "go")
        self.assertTrue(result["g1_v2_qualification"]["go"])
        self.assertTrue(result["g1_v2_qualification"]["cell_loop_ready"])
        self.assertEqual(verifier.verify(require_go=True)["status"], "pass")

        public_accept = verifier.QUALIFICATION_PHASE_RECEIPTS[
            verifier.EvaluationPhase.PUBLIC_ACCEPT
        ]
        original = json.loads(public_accept.read_text(encoding="utf-8"))
        tampered = copy.deepcopy(original)
        tampered["phase_result"]["checkpoint_hash"] = "sha256:" + "0" * 64
        with tempfile.TemporaryDirectory(prefix="p5-qualification-") as raw_directory:
            path = Path(raw_directory) / "public_accept.json"
            path.write_text(json.dumps(tampered), encoding="utf-8")
            phase_receipts = dict(verifier.QUALIFICATION_PHASE_RECEIPTS)
            phase_receipts[verifier.EvaluationPhase.PUBLIC_ACCEPT] = path
            with mock.patch.object(
                verifier,
                "QUALIFICATION_PHASE_RECEIPTS",
                phase_receipts,
            ):
                with self.assertRaisesRegex(
                    AssertionError,
                    "public_accept result hash mismatch",
                ):
                    verifier.verify()


if __name__ == "__main__":
    unittest.main()
