"""Version-qualified current-only input; no inferred history or lost endpoint."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

import numpy as np

from tests.unit import test_stair_trajectory_input_inspection as fixture
from true_rl_agent.evaluation.trajectory_translation.measurement import measure_episode
from true_rl_agent.evaluation.trajectory_translation.report import file_sha256
from true_rl_agent.evaluation.process_report import report_bytes
from true_rl_agent.runner.numeric_evidence import NumericEvidenceError
from true_rl_agent.tasks.stair_climbing.trajectory_adapter import iter_stair_episodes
from true_rl_agent.tasks.stair_climbing.trajectory_pipeline import build_trial_trajectory_card
from true_rl_agent.tasks.stair_climbing.trajectory_report import build_stair_trajectory_report


def as_v1(identity, context, chunks):
    identity["evaluation_plan_hash"] = "e" * 64
    identity["evaluation_binding"]["evaluation_binding_hash"] = "f" * 64
    context["profile"] = "stair.unified_first_episode.v1"
    del context["contact_force_history"]
    for metadata, arrays in chunks:
        del metadata["transition_header"]["contact_force_history"]
        del arrays["support_contact_forces_w_history"]
        del arrays["forbidden_contact_forces_w_history"]


class StairTrajectoryV1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.StairTrajectoryInputInspectionTests.setUpClass()

    def setUp(self):
        self.fixture = fixture.StairTrajectoryInputInspectionTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_missing_history_is_unknown_and_manager_is_independent(self):
        ref = self.fixture._archive(as_v1)
        e = next(iter_stair_episodes(ref, source_id="v1"))
        before = set(e.arrays)
        records = measure_episode(e)
        self.assertEqual(set(e.arrays), before)
        self.assertFalse(any(k.endswith("_history") for k in e.arrays))
        self.assertEqual(e.source["contact_force_history"]["status"], "not_recorded")
        measures = {r["key"]: r for r in records if r["record_type"] == "measurement"}
        for r in measures.values():
            if r["key"].endswith(".history_diagnostic"):
                v = r["value"]
                self.assertEqual((v["true_count"], v["false_count"], v["unknown_count"]), (0, 0, 1))
                self.assertEqual(v["valid_count"], 0)
                self.assertIsNone(v["fraction"])
                self.assertTrue(r["evidence_ref"]["missing_channels"])
                self.assertFalse(any(c.endswith("_history") for c in r["evidence_ref"]["channels"]))
        history_records = [r for r in records if ".history_any_marker" in r["key"]]
        self.assertTrue(history_records)
        self.assertTrue(all(r["record_type"] == "state_interval" and r["value"]["state"] == "unknown" for r in history_records))
        self.assertEqual(measures["robot_0.body.base.current_marker"]["value"]["true_count"], 0)
        terminal = measures["task_endpoint"]["value"]
        self.assertTrue(terminal["terminal_flags"]["termination_1"])
        self.assertEqual(terminal["outcome"]["primary_reason"], "forbidden_contact")
        self.assertIsNone(terminal["tail_forbidden_history_norms_n"])

    def test_v1_rejects_history_metadata_arrays_header_or_late_drift(self):
        for field in ("metadata", "array", "header", "late_array", "clock", "unknown_profile"):
            def mutate(identity, context, chunks):
                old = deepcopy(context["contact_force_history"])
                h = chunks[0][1]["support_contact_forces_w_history"].copy()
                as_v1(identity, context, chunks)
                if field == "metadata":
                    context["contact_force_history"] = old
                elif field in ("array", "late_array"):
                    i = int(field == "late_array")
                    chunks[i][1]["support_contact_forces_w_history"] = h[:chunks[i][0]["steps"]]
                elif field == "header":
                    chunks[1][0]["transition_header"]["contact_force_history"] = old
                elif field == "clock":
                    context["seeds"][0]["step_dt"] = .04
                else:
                    context["profile"] = "stair.unified_first_episode.v99"
            with self.subTest(field=field), self.assertRaises(NumericEvidenceError):
                next(iter_stair_episodes(self.fixture._archive(mutate), source_id="v1"))

    def test_v2_missing_history_is_rejected_instead_of_downgraded(self):
        def mutate(identity, context, chunks):
            del chunks[1][1]["support_contact_forces_w_history"]
        with self.assertRaises(NumericEvidenceError):
            next(iter_stair_episodes(self.fixture._archive(mutate), source_id="v2"))

    def test_current_contact_does_not_create_history_and_nan_stays_unknown(self):
        ref = self.fixture._archive(as_v1)
        e = list(iter_stair_episodes(ref, source_id="v1"))[2]
        e.arrays["support_contact_forces_w"][:, 0, :] = [[0, 0, 0], [0, 0, 3], [0, 0, 0]]
        e.arrays["support_contact_forces_w"][0, 1, 0] = np.nan
        r = measure_episode(e)
        transitions = [x for x in r if x["key"] == "robot_0.support.FL_foot.marker.transition"]
        self.assertEqual([x["value"]["step"] for x in transitions], [2, 3])
        self.assertFalse(any(x["record_type"] == "event" and "history" in x["key"] for x in r))
        m = {x["key"]: x["value"] for x in r if x["record_type"] == "measurement"}
        self.assertEqual(m["robot_0.support.FR_foot.marker"]["unknown_count"], 1)

    def test_automatic_card_public_budget_missing_slot_and_source_swap(self):
        ref = self.fixture._archive(as_v1)
        identity = json.loads(Path(ref["manifest_path"]).read_text())["identity"]
        card_ref, _ = build_trial_trajectory_card(ref, expected_identity=identity, output_dir=self.fixture.root / "phase")
        full = build_stair_trajectory_report(card_ref, numeric_reference=ref, expected_identity=identity)
        self.assertEqual(len(full["audit"]["required_claim_ids"]), 18)
        self.assertLessEqual(report_bytes(full["public_report"]), 8192)
        text = json.dumps(full["public_report"], ensure_ascii=False)
        self.assertIn("history未采集", text)
        self.assertIn("manager终止信号独立保留", text)
        self.assertIn("该槽不可提供事实", text)
        with self.assertRaisesRegex(ValueError, "mandatory facts exceed"):
            build_stair_trajectory_report(card_ref, numeric_reference=ref, expected_identity=identity,
                                           max_bytes=full["audit"]["required_report_bytes"] - 1)
        directory = Path(card_ref["manifest_path"]).parent
        groups = [json.loads(line) for line in (directory / "evidence.jsonl").read_text(encoding="utf-8").splitlines()]
        all_group = next(r for r in groups if r["record_type"] == "cohort_summary" and r["group"] == "all")
        for key, metric in all_group["metrics"].items():
            if "history_diagnostic" in key:
                self.assertEqual(metric["valid_episode_count"], 0)
                self.assertEqual(metric["unknown_episode_count"], 288)
                self.assertIsNone(metric["episode_equal"]["mean"])
                self.assertEqual(metric["observed_presence"]["valid_denominator"], 0)
                self.assertEqual(metric["observed_presence"]["unknown_episode_count"], 288)
        other = self.fixture._archive(as_v1)
        # Same labels and policy are insufficient: a different source identity is pinned.
        path = Path(other["manifest_path"])
        path.write_bytes(path.read_bytes() + b" ")
        other["manifest_sha256"] = file_sha256(path)
        with self.assertRaisesRegex(ValueError, "source/trial/checkpoint"):
            build_stair_trajectory_report(card_ref, numeric_reference=other, expected_identity=identity)

    def test_all_success_without_manager_trigger_retains_missing_history_witness(self):
        def mutate(identity, context, chunks):
            as_v1(identity, context, chunks)
            for outcome in context["seeds"][0]["outcomes"]:
                outcome.update(success=True, primary_reason="success")
            for _, arrays in chunks:
                for key in ("done_mask", "wrapper_done_mask", "termination_0", "termination_1"):
                    arrays[key][:] = False
        ref = self.fixture._archive(mutate)
        identity = json.loads(Path(ref["manifest_path"]).read_text())["identity"]
        card_ref, _ = build_trial_trajectory_card(ref, expected_identity=identity, output_dir=self.fixture.root / "success")
        full = build_stair_trajectory_report(card_ref, numeric_reference=ref, expected_identity=identity)
        self.assertEqual(len(full["audit"]["required_claim_ids"]), 18)
        text = json.dumps(full["public_report"], ensure_ascii=False)
        self.assertIn("history未采集", text)
        self.assertIn("manager终止信号独立保留", text)
        self.assertLessEqual(report_bytes(full["public_report"]), 8192)


if __name__ == "__main__":
    unittest.main()
