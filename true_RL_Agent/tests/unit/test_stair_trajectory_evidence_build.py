"""The offline entrypoint must publish complete, source-bound artifacts."""
import importlib.util
import json
from pathlib import Path
import unittest

from tests.unit import test_stair_trajectory_input_inspection as fixture
from true_rl_agent.runner.numeric_evidence import NumericEvidenceError

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("build_stair_trajectory_evidence", ROOT / "scripts/build_stair_trajectory_evidence.py")
build = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build)


class EvidenceBuildTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.StairTrajectoryInputInspectionTests.setUpClass()

    def setUp(self):
        self.fixture = fixture.StairTrajectoryInputInspectionTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_selected_evidence_has_resolvable_operands_and_deterministic_bytes(self):
        reference = self.fixture._archive()
        outputs = [self.fixture.root / "evidence_a", self.fixture.root / "evidence_b"]
        manifests = [build.build_evidence(reference, output, "fixture", {(1, 0), (1, 2)}) for output in outputs]
        result = manifests[0]
        self.assertEqual((result["scanned_episodes"], result["scanned_valid_rows"]), (288, 861))
        self.assertEqual((result["emitted_episodes"], result["emitted_valid_rows"]), (2, 4))
        self.assertEqual(result["cohort_numeric_summary"]["outcome_counts"], {"forbidden_contact": 1, "success": 1})
        self.assertEqual(result["evidence"]["sha256"], manifests[1]["evidence"]["sha256"])
        records = [json.loads(line) for line in (outputs[0] / "evidence.jsonl").read_text("utf-8").splitlines()]
        ids = {r["record_id"] for r in records if "record_id" in r}
        self.assertEqual(len(ids), len(records) - 2)
        for record in records:
            if "record_id" not in record:
                continue
            self.assertEqual(record["evidence_ref"]["manifest_sha256"], reference["manifest_sha256"])
            self.assertTrue(all(i in ids for i in record["value"].get("operand_record_ids", [])))
        with self.assertRaises(FileExistsError):
            build.build_evidence(reference, outputs[0], "fixture")

    def test_invalid_late_chunk_never_publishes_complete_manifest(self):
        reference = self.fixture._archive()
        late = Path(reference["manifest_path"]).parent / "chunk_000001.npz"
        late.write_bytes(late.read_bytes() + b"invalid")
        output = self.fixture.root / "rejected"
        with self.assertRaises(NumericEvidenceError):
            build.build_evidence(reference, output, "fixture", {(1, 0)})
        self.assertFalse((output / "manifest.json").exists())

    def test_missing_selection_never_claims_complete_or_falls_back(self):
        output = self.fixture.root / "absent"
        with self.assertRaisesRegex(ValueError, "absent"):
            build.build_evidence(self.fixture._archive(), output, "fixture", {(1, 999)})
        self.assertFalse((output / "manifest.json").exists())

    def test_sealed_source_rejected_before_any_episode_is_emitted(self):
        def mutate(identity, context, chunks):
            identity["splits"] = ["sealed_validation"]
        output = self.fixture.root / "sealed"
        with self.assertRaisesRegex(NumericEvidenceError, "public_dev"):
            build.build_evidence(self.fixture._archive(mutate), output, "fixture")
        self.assertFalse((output / "manifest.json").exists())
        self.assertEqual((output / "evidence.jsonl.partial").stat().st_size, 0)
