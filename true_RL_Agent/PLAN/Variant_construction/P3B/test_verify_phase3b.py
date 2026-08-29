from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

P3B_DIR = Path(__file__).resolve().parent
if str(P3B_DIR) not in sys.path:
    sys.path.insert(0, str(P3B_DIR))

import verify_phase3b as verifier


class Phase3BVerifierTests(unittest.TestCase):
    def test_frozen_snapshot_matches_current_contract(self) -> None:
        result = verifier.verify()
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["protocol_state"], "active")
        self.assertEqual(result["formal_episode_count"], 1080)

    def test_runtime_receipt_tamper_is_rejected(self) -> None:
        receipt = json.loads(
            verifier.RUNTIME_RECEIPT_PATH.read_text(encoding="utf-8")
        )
        receipt["geometry_probe"]["cell_count"] = 8
        with tempfile.TemporaryDirectory() as raw_directory:
            path = Path(raw_directory) / "runtime.json"
            path.write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaisesRegex(AssertionError, "receipt hash mismatch"):
                verifier._validate_receipt(path, verifier.RUNTIME_SCHEMA_VERSION)

    def test_snapshot_tamper_is_rejected(self) -> None:
        snapshot = json.loads(
            verifier.SNAPSHOT_PATH.read_text(encoding="utf-8")
        )
        tampered = copy.deepcopy(snapshot)
        tampered["contract"]["formal_episode_count"] = 1081
        with tempfile.TemporaryDirectory() as raw_directory:
            path = Path(raw_directory) / "snapshot.json"
            path.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaisesRegex(AssertionError, "snapshot hash mismatch"):
                verifier.verify(path)

    def test_prompt_suite_drift_is_rejected(self) -> None:
        config, manifest, plugin = verifier._resolve_cell()
        adapter = plugin.prompt_adapter()
        sections = adapter.build_prompt_capacity_preflight_sections()
        drifted = copy.deepcopy(sections)
        drifted["task"]["fixed_eval_protocol_summary"]["strata"].remove(
            "c2_mechanism_ood"
        )

        class _DriftedAdapter:
            def build_prompt_capacity_preflight_sections(self):
                return drifted

        with mock.patch.object(
            plugin, "prompt_adapter", return_value=_DriftedAdapter()
        ):
            with self.assertRaisesRegex(
                AssertionError, "prompt strata differ from suite"
            ):
                verifier._contract_receipt(config, manifest, plugin)


if __name__ == "__main__":
    unittest.main()
