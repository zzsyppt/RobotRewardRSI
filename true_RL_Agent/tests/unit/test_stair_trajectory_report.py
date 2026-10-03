"""P1c source binding, atomic UTF-8 budgets and versioned delivery boundaries."""
from copy import deepcopy
import json
import unittest

from tests.unit import test_stair_trajectory_production as production_fixture
from true_rl_agent.tasks.stair_climbing.trajectory_production import project_production_report
from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.evaluation.process_report import report_bytes, SENTENCE_SECTIONS
from true_rl_agent.tasks.stair_climbing.behavior import project_stair_behavior, compare_stair_behavior
from true_rl_agent.tasks.stair_climbing.trajectory_report import (
    PUBLIC_SCHEMA, MAX_PUBLIC_BYTES, validate_stair_trajectory_report,
    require_trajectory_trial_binding, require_trajectory_phase_binding,
)
from true_rl_agent.tasks.stair_climbing.prompt_adapter import _latest_behavior_feedback, _reference_behavior_feedback


def resign(report):
    report["report_projection"].pop("content_hash", None)
    report["report_projection"]["content_hash"] = canonical_json_hash(report)
    return report


class StairTrajectoryReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        production_fixture.ProductionTrajectoryTests.setUpClass()
        cls.summary = production_fixture.ProductionTrajectoryTests.summary
        cls.domain = deepcopy(production_fixture.ProductionTrajectoryTests.domain)
        cls.domain["evaluation_job_id"] = "generation-1-lane-a-eval-public-dev"
        cls.full = cls.project()
        cls.public = cls.full["public_report"]

    @classmethod
    def tearDownClass(cls):
        production_fixture.ProductionTrajectoryTests.tearDownClass()

    @classmethod
    def project(cls, budget=MAX_PUBLIC_BYTES):
        return project_production_report(cls.domain, cls.summary, max_bytes=budget)

    def trial(self, report=None, name="generation-1-lane-a"):
        report = deepcopy(report or self.public)
        return {"trial_id": name, "lane_id": name.rsplit("-", 1)[-1], "public_dev": {
            "checkpoint_hash": report["domain"]["checkpoint_hash"],
            "evaluation_plan_hash": report["domain"]["evaluation_plan_hash"],
            "public_summary": {"behavior": report}}}

    def test_exact_final_bytes_and_one_byte_below_required_fail(self):
        n = self.full["audit"]["required_report_bytes"]
        tight = self.project(n)
        self.assertEqual(report_bytes(tight["public_report"]), n)
        self.assertEqual(tight["audit"]["included_claim_ids"], tight["audit"]["required_claim_ids"])
        self.assertEqual(tight, self.project(n))
        with self.assertRaisesRegex(ValueError, "mandatory facts exceed"):
            self.project(n-1)
        actual = self.full["audit"]["report_bytes"]
        self.assertEqual(self.project(actual)["public_report"], self.public)
        self.assertGreater(actual, len(json.dumps(self.public, ensure_ascii=False, separators=(",", ":"))))




    def test_source_sibling_job_checkpoint_and_plan_mismatches_rejected(self):
        require_trajectory_trial_binding(self.public, self.trial())
        for field in ("trial_id", "checkpoint_hash", "evaluation_plan_hash"):
            trial = self.trial()
            if field == "trial_id":
                trial[field] = "generation-1-lane-b"
            else:
                trial["public_dev"][field] = "x"*64
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "sibling/trial/phase"):
                require_trajectory_trial_binding(self.public, trial)
        phase = dict(job_id=self.domain["evaluation_job_id"], checkpoint_hash=self.domain["checkpoint_hash"],
                     evaluation_plan_hash=self.domain["evaluation_plan_hash"], evaluation_binding_hash=self.domain["evaluation_binding_hash"],
                     archive_hash=self.domain["archive_hash"], seed_namespace=self.domain["seed_namespace"])
        require_trajectory_phase_binding(self.public, **phase)
        for field in phase:
            bad = {**phase, field: "wrong"}
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "phase source binding"):
                require_trajectory_phase_binding(self.public, **bad)

    def test_wrong_unselected_sibling_is_not_hidden_by_last_lane(self):
        trial = self.trial()
        bad = self.trial(name="generation-1-lane-b")
        with self.assertRaisesRegex(ValueError, "sibling/trial/phase"):
            _latest_behavior_feedback([{"generation": {"trials": [bad, trial]}}])

    def test_last_lane_is_explicitly_separate_from_accepted_lane(self):
        first = self.trial()
        second_report = deepcopy(self.public)
        second_report["domain"]["evaluation_job_id"] = "generation-1-lane-b-eval-public-dev"
        second_report["domain"]["evaluation_plan_hash"] = "9"*64
        resign(second_report)
        second = self.trial(second_report, "generation-1-lane-b")
        result = _latest_behavior_feedback([{"generation": {"trials": [first, second],
            "formal": {"accepted": True, "selected_trial_id": first["trial_id"]}}}])
        self.assertEqual(result["source_trial_id"], second["trial_id"])
        self.assertEqual(result["profile"], second_report)
        self.assertEqual(result["accepted_best_relationship"], "not_established_for_selected_trial")
        self.assertEqual(result["diff"]["status"], "not_available")


    def test_paths_sealed_text_and_changed_sentence_rejected_before_scrubbing(self):
        for text in ("C:\\private\\original.npz", "/private/archive/data.npz", "sealed_validation"):
            bad = deepcopy(self.public)
            bad["report"]["gaps"][0]["text"] = text
            resign(bad)
            with self.subTest(text=text), self.assertRaises(ValueError):
                validate_stair_trajectory_report(bad)
        bad = deepcopy(self.public)
        bad["report"]["process"][0]["text"] = "lost the denominator"
        with self.assertRaisesRegex(ValueError, "content hash"):
            project_stair_behavior(bad)

    def test_diagnostic_source_not_promoted_to_reference_or_round_trial(self):
        observation = {"status": "pass", "metrics": {"behavior": self.public}}
        binding = {"job_id": self.domain["evaluation_job_id"], **{k: self.domain[k] for k in
            ("checkpoint_hash", "evaluation_plan_hash", "evaluation_binding_hash", "archive_hash", "seed_namespace")}}
        anchor = {"anchor_kind": "reference_program", "public_result_hash": canonical_json_hash(observation),
            "public_result": observation, "trajectory_phase_binding": binding}
        result = _reference_behavior_feedback({"initial_anchor_public_evidence": anchor})
        self.assertEqual(result["source_evaluation_job_id"], self.domain["evaluation_job_id"])
        self.assertEqual(result["accepted_best_relationship"], "not_established_by_behavior_card")
        for field in binding:
            bad = deepcopy(anchor)
            bad["trajectory_phase_binding"][field] = "unrelated-source"
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "phase source binding"):
                _reference_behavior_feedback({"initial_anchor_public_evidence": bad})
        bad = deepcopy(anchor)
        bad["public_result_hash"] = "a"*64
        with self.assertRaisesRegex(ValueError, "result hash mismatch"):
            _reference_behavior_feedback({"initial_anchor_public_evidence": bad})
        bad = deepcopy(anchor)
        bad.pop("trajectory_phase_binding")
        with self.assertRaisesRegex(ValueError, "explicit phase binding"):
            _reference_behavior_feedback({"initial_anchor_public_evidence": bad})
        with self.assertRaisesRegex(ValueError, "explicit phase"):
            _latest_behavior_feedback([{"round_id": "unbound-round", "public_result": {"metrics": {"behavior": self.public}}}])


    def test_minimum_core_keeps_entire_report_and_provider_serialization(self):
        from tests.unit.test_prompt_capacity_projection import _base_sections
        from true_rl_agent.tasks.stair_climbing.prompt_adapter import _minimum_core_stair_prompt_sections
        from true_rl_agent.prompting.payload import build_prompt_payload
        from true_rl_agent.prompting.payload import serialize_prompt_payload
        from true_rl_agent.prompting.task_sections import sanitize_task_prompt_sections, validate_provider_task_prompt_receipt
        sections = _base_sections({"behavior": {"profile": self.public}, "verbose_optional": "汉"*30000})
        minimum = _minimum_core_stair_prompt_sections(sections, target_bytes=56*1024)
        clean = sanitize_task_prompt_sections(minimum)
        self.assertEqual(clean["public_feedback_summary"]["behavior"]["profile"], self.public)
        payload = build_prompt_payload(public_context={"task_prompt_sections": clean})
        delivered = json.loads(serialize_prompt_payload(payload))
        self.assertEqual(delivered["public_feedback_summary"]["behavior"]["profile"], self.public)
        validate_provider_task_prompt_receipt(delivered["authoring_context_frame"]["budget_metadata"]["provider_task_sections"], prompt_payload=delivered)

    def test_receipt_counts_and_digests_reject_rehashed_invalid_values(self):
        for key, value in (("claim_count", -1), ("omitted_claim_count", -19), ("included_claim_count", True),
                           ("omitted_claim_ids_hash", "not-a-hash"), ("required_claim_ids_hash", "F"*64)):
            bad = deepcopy(self.public)
            bad["report_projection"][key] = value
            resign(bad)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "counts/digests"):
                project_stair_behavior(bad)


if __name__ == "__main__":
    unittest.main()
