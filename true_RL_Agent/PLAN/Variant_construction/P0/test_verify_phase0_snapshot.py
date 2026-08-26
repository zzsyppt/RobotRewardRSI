from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


P0_ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "phase0_snapshot_verifier",
    P0_ROOT / "verify_phase0_snapshot.py",
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("unable to load Phase 0 snapshot verifier")
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)


class Phase0SnapshotVerifierTests(unittest.TestCase):
    def test_frozen_snapshot_passes(self) -> None:
        self.assertEqual(VERIFIER.verify()["status"], "pass")

    def test_coherent_runtime_tamper_is_rejected_by_stair_authority(self) -> None:
        snapshot = json.loads(
            VERIFIER.SNAPSHOT_PATH.read_text(encoding="utf-8")
        )
        receipt = json.loads(
            VERIFIER.RUNTIME_RECEIPT_PATH.read_text(encoding="utf-8")
        )
        profile = next(
            item
            for item in receipt["robot_runtime_facts"]
            if item["profile_key"] == "p0.g1_37_stair_execution"
        )
        old_joint = profile["joint_names"][0]
        fake_joint = "FL_hip_joint"
        profile["joint_names"][0] = fake_joint
        default_positions = profile["default_state"]["joint_pos_rad_by_name"]
        default_positions[fake_joint] = default_positions.pop(old_joint)
        for actuator in profile["actuators"]:
            if old_joint not in actuator["joint_names"]:
                continue
            actuator["joint_names"] = [
                fake_joint if name == old_joint else name
                for name in actuator["joint_names"]
            ]
            for values in actuator["parameters_by_joint"].values():
                values[fake_joint] = values.pop(old_joint)
        profile["runtime_facts_hash"] = VERIFIER.canonical_json_hash(
            VERIFIER._without_hash(profile, "runtime_facts_hash")
        )

        binding = next(
            item
            for item in receipt["legacy_cell_binding_facts"]
            if item["cell_key"] == "stair.g1_37dof.clean_legacy_v1"
        )
        binding["control"]["decimation"] = 8
        binding["control"]["control_dt_s"] = 0.04
        binding["binding_facts_hash"] = VERIFIER.canonical_json_hash(
            VERIFIER._without_hash(binding, "binding_facts_hash")
        )
        receipt["receipt_hash"] = VERIFIER.canonical_json_hash(
            VERIFIER._without_hash(receipt, "receipt_hash")
        )

        declared = snapshot["runtime_probe_receipt"]
        declared["receipt_hash"] = receipt["receipt_hash"]
        declared["profile_runtime_hashes"][profile["profile_key"]] = profile[
            "runtime_facts_hash"
        ]
        declared["binding_facts_hashes"][binding["cell_key"]] = binding[
            "binding_facts_hash"
        ]

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            receipt_path = root / "receipt.json"
            receipt_path.write_text(
                json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2)
                + "\n",
                encoding="utf-8",
            )
            declared["file_sha256"] = hashlib.sha256(
                receipt_path.read_bytes()
            ).hexdigest()
            snapshot["snapshot_hash"] = VERIFIER.canonical_json_hash(
                VERIFIER._without_hash(snapshot, "snapshot_hash")
            )
            snapshot_path = root / "snapshot.json"
            snapshot_path.write_text(
                json.dumps(snapshot, ensure_ascii=False, sort_keys=True, indent=2)
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                AssertionError,
                "G1-37 runtime joints versus Stair protocol",
            ):
                VERIFIER.verify(
                    snapshot_path=snapshot_path,
                    receipt_path=receipt_path,
                )


if __name__ == "__main__":
    unittest.main()
