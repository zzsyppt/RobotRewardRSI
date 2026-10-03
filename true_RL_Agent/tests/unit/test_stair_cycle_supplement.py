"""Offline source/operand semantics, frozen files and combined byte budget."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

from tests.unit import test_stair_trajectory_report as reports
from tests.unit import test_stair_trajectory_input_inspection as archives
from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.evaluation.process_report import report_bytes
from true_rl_agent.evaluation.trajectory_translation.contracts import stable_id
from true_rl_agent.evaluation.trajectory_translation.cycles import CYCLE_CONFIG
from true_rl_agent.evaluation.trajectory_translation.report import file_sha256
from true_rl_agent.tasks.stair_climbing.trajectory_cycles import build_cycle_supplement, preview_with_cycles
from true_rl_agent.tasks.stair_climbing.trajectory_report import project_trajectory_card, validate_stair_trajectory_report

ROOT = Path(__file__).resolve().parents[2]


class CycleSupplementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        reports.StairTrajectoryReportTests.setUpClass()
        old = reports.StairTrajectoryReportTests
        archives.StairTrajectoryInputInspectionTests.setUpClass()
        cls.fixture = archives.StairTrajectoryInputInspectionTests()
        cls.fixture.setUp()
        def binding(identity, context, chunks):
            identity["evaluation_binding"]["evaluation_binding_hash"] = "c"*64
            identity["evaluation_plan_hash"] = "e"*64
            identity["seed_namespace"] = "cycle_测试"
        cls.reference = cls.fixture._archive(binding)
        cls.directory = cls.fixture.root/"cycles"
        cls.manifest = build_cycle_supplement(cls.reference,cls.directory,source_id="fixture")
        cls.supplement = json.loads((cls.directory/"report.json").read_text(encoding="utf-8"))
        cls.episodes = [json.loads(line) for line in (cls.directory/"cycles.jsonl").read_text(encoding="utf-8").splitlines()]
        source = cls.supplement["source"]
        domain = {**old.domain, "evaluation_job_id": source["trial_id"], "archive_hash": source["manifest_sha256"],
            "projection_hash": source["runtime_projection_hash"], "seed_namespace": source["seed_namespace"],
            **{k: source["evaluation_binding"][k] for k in ("task_protocol_hash", "evaluation_binding_hash")}}
        full = project_trajectory_card(old.card,old.records,domain=domain,card_hash="f"*64)
        cls.base = project_trajectory_card(old.card,old.records,domain=domain,card_hash="f"*64,
                                          max_bytes=full["audit"]["required_report_bytes"])["public_report"]

    @classmethod
    def tearDownClass(cls):
        cls.fixture.doCleanups()
        reports.StairTrajectoryReportTests.tearDownClass()

    def test_shared_budget_optional_unicode_exact_boundary_and_fallback(self):
        preview, receipt = preview_with_cycles(self.base,self.supplement,self.episodes)
        self.assertEqual(preview["base_report"],self.base)
        self.assertEqual(preview["optional_cycle_statements"],self.supplement["optional_statements"])
        self.assertLessEqual(report_bytes(preview),8192)
        first = {**preview, "optional_cycle_statements": self.supplement["optional_statements"][:1]}
        exact = report_bytes(first)
        preview, receipt = preview_with_cycles(self.base,self.supplement,self.episodes,max_bytes=exact)
        self.assertEqual(preview,first)
        self.assertEqual(receipt["combined_bytes"],exact)
        preview, receipt = preview_with_cycles(self.base,self.supplement,self.episodes,max_bytes=exact-1)
        self.assertEqual(preview,self.base)
        self.assertEqual(receipt["included_ids"],[])
        self.assertEqual(receipt["omitted"][0]["id"],first["optional_cycle_statements"][0]["id"])
        self.assertEqual(self.base["domain"]["catalog_version"],"trajectory.p0.measurements.v2")

    def test_mismatched_source_and_production_mislabel_rejected(self):
        for field in ("archive_hash", "checkpoint_hash", "evaluation_job_id", "seed_namespace",
                      "projection_hash", "task_protocol_hash", "evaluation_binding_hash", "evaluation_plan_hash"):
            bad = deepcopy(self.base)
            bad["domain"][field] = "1"*64
            reports.resign(bad)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError,"source differs"):
                preview_with_cycles(bad,self.supplement,self.episodes)
        bad = deepcopy(self.base)
        bad["domain"]["catalog_version"] = "trajectory.p0.measurements.v3"
        with self.assertRaises(ValueError):
            validate_stair_trajectory_report(bad)
        with self.assertRaises(ValueError):
            preview_with_cycles(self.base,self.supplement,self.episodes,max_bytes=8193)

    def test_parent_text_entity_unit_and_coherent_group_swap_rejected(self):
        for swap in ("entity_text", "unit_text", "group_key_id"):
            report = deepcopy(self.supplement)
            statement = report["optional_statements"][0]
            if swap == "unit_text":
                statement["text"] = statement["text"].replace("Period unknown.","Period unknown [ms].")
            else:
                statement["text"] = statement["text"].replace("FL_foot","RR_foot")
            if swap == "group_key_id":
                key = "contact_cycles:robot_0.support.FL_foot"
                group = report["summary"]["groups"].pop(key)
                group["entity_ids"] = ["robot_0.support.RR_foot"]
                key = key.replace("FL_foot","RR_foot")
                report["summary"]["groups"][key] = group
                statement["id"] = "cycle:" + canonical_json_hash([report["binding"],CYCLE_CONFIG,key,group])
            with self.subTest(swap=swap), self.assertRaisesRegex(ValueError,"bound measurement operands"):
                preview_with_cycles(self.base,report,self.episodes)

    def test_record_unit_frame_and_resigned_semantics_rejected(self):
        for field, value in (("unit","milliseconds; phase degrees"),("frame","physical_contact_time")):
            for resign in (False, True):
                episodes = deepcopy(self.episodes)
                for record in episodes[0]["records"]:
                    record[field] = value
                    if resign:
                        record["record_id"] = stable_id("cycle-evidence",record["episode_id"],CYCLE_CONFIG,
                            record["key"],record["entity_ids"],record["value"],record["unit"],record["frame"])
                with self.subTest(field=field,resign=resign), self.assertRaisesRegex(ValueError,"unit/frame/schema"):
                    preview_with_cycles(self.base,self.supplement,episodes)

    def test_source_metadata_and_record_metric_entity_locator_mutations_rejected(self):
        for field in ("trial_id", "runtime_projection_hash", "seed_namespace"):
            report = deepcopy(self.supplement)
            report["source"][field] = "1"*64
            with self.subTest(field=field), self.assertRaisesRegex(ValueError,"numeric manifest"):
                preview_with_cycles(self.base,report,self.episodes)
        for mutation in ("metric", "entity", "source", "clock", "coverage"):
            episodes = deepcopy(self.episodes)
            record = episodes[0]["records"][0]
            if mutation == "metric": record["key"] = "contact_phase"
            elif mutation == "entity": record["entity_ids"] = ["robot_0.support.RR_foot"]
            elif mutation == "source": record["evidence_ref"]["manifest_sha256"] = "1"*64
            elif mutation == "clock": episodes[0]["dt_s"] *= 1000
            else: episodes.pop()
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                preview_with_cycles(self.base,self.supplement,episodes)

    def test_actual_adapter_builds_separate_complete_artifact(self):
        directory, manifest = self.directory, self.manifest
        self.assertEqual(manifest["catalog_version"],"trajectory.p0.measurements.v3")
        self.assertEqual(manifest["base_catalog_version"],"trajectory.p0.measurements.v2")
        self.assertNotIn("runner_seed",self.supplement["source"])
        self.assertEqual(sum(r["episodes"] for r in self.supplement["runs"]),manifest["episode_count"])
        for name,info in manifest["artifacts"].items():
            self.assertEqual(file_sha256(directory/name),info["sha256"])
        with self.assertRaises(FileExistsError):
            build_cycle_supplement(self.reference,directory,source_id="fixture")

    def test_frozen_push_22_files_still_match(self):
        freeze = json.loads((ROOT/"PLAN/feedback_sys/NEW_SYS/P2b1/RULES_CODE_FREEZE.json").read_text(encoding="utf-8"))
        self.assertEqual(len(freeze["files_sha256"]),22)
        for name,wanted in freeze["files_sha256"].items():
            self.assertEqual(file_sha256(ROOT/name),wanted,name)


if __name__ == "__main__":
    unittest.main()
