from copy import deepcopy

import unittest

from true_rl_agent.evaluation.process_report import (
    SENTENCE_SECTIONS, fit_process_report_capacity, render_process_report, report_bytes, validate_sentence,
)
from true_rl_agent.evaluation.visual_observer import (
    audit_observer_output, update_visual_projection_count, visual_process_sentences,
)
from tests.unit.test_task_visual_observation import observation, window
from tests.unit.test_visual_review import synthetic_review


def statement(section, *, kind="S5"):
    return dict(id=section, kind=kind, section=section, text=f"Evidence gap for {section}.",
                subject="subject", predicate="coverage", rule_version="fixture.v1",
                scope={"label": "fixture", "numerator": 0, "denominator": 1},
                evidence_refs=["archive:fixture"], gap_type="not_observed", required=True)


class ProcessReportTests(unittest.TestCase):
    def visual_rows(self):
        steps = [1, 20, 40, 80, 160, 320]
        item = window(steps, ({"id": "fixture_predicate", "body": "root", "definition": "Fixture only."},),
                      views=("v0", "v1"))
        raw = {"episodes": [{"alias": "e1", "not_visible": [], "observations": [
            {**observation("fixture_predicate", steps=steps, views=("v0", "v1")),
             "text": "\U00020000" * 200, "uncertainty": "\U00020001" * 120}
            for _ in range(3)]}]}
        audit = audit_observer_output(raw, [item], numeric_check=lambda *_: {"status": "conflict"})
        self.assertTrue(audit["passed"])
        self.assertEqual(len(audit["observations"][0]["image_evidence"]), 12)
        audit = synthetic_review(audit)
        return audit, ([statement(s) for s in SENTENCE_SECTIONS if s != "visual"]
            + visual_process_sentences(audit, episode_labels={"e1": "r1.e0"}, cohort_episode_n=864,
                                       observer_version="fixture.v1", evidence_ref="observer_evidence"))

    def test_legal_unicode_observation_is_omitted_whole_with_conflict_and_coverage_kept(self):
        audit, rows = self.visual_rows()
        before, raw = deepcopy(rows), deepcopy(audit)
        report = render_process_report(envelope={}, sentences=rows, actionability={"scope": "existing"},
            max_bytes=1800, update_projection=update_visual_projection_count)
        self.assertEqual(report["report_projection"]["omitted_statement_ids"], ["visual.observation"])
        self.assertEqual([row["id"] for row in report["report"]["visual"]], ["visual.coverage"])
        coverage = report["report"]["visual"][0]["text"]
        for text in ("3 statements", "1/864 episodes", "0 mechanically invalid", "0 shown", "3 numeric conflicts"):
            self.assertIn(text, coverage)
        self.assertEqual(report["report"]["counterexample"], [{"id": "counterexample", "text": before[2]["text"]}])
        self.assertEqual(report["report"]["gaps"][-1]["id"], "report.capacity_omissions")
        self.assertLessEqual(report_bytes(report), 1800)
        self.assertEqual(rows, before)
        self.assertEqual(audit, raw)
        with self.assertRaisesRegex(ValueError, "byte budget"):
            render_process_report(envelope={}, sentences=rows, actionability={"scope": "existing"},
                max_bytes=200, update_projection=update_visual_projection_count)

    def test_parent_update_reuses_optional_policy_without_clipping_or_duplicate_gaps(self):
        _, rows = self.visual_rows()
        report = render_process_report(envelope={}, sentences=rows, actionability={"scope": "existing"},
            max_bytes=8192, update_projection=update_visual_projection_count)
        budget = report_bytes(report) + 20
        self.assertIn("1 shown", report["report"]["visual"][-1]["text"])
        report["report"]["parent_diff"][0]["text"] += " Parent fixture detail." * 10
        grown = deepcopy(report)
        bounded = fit_process_report_capacity(report, max_bytes=budget, update_projection=update_visual_projection_count)
        self.assertEqual(bounded["report"]["parent_diff"], grown["report"]["parent_diff"])
        self.assertEqual(bounded["report_projection"]["omitted_statement_ids"], ["visual.observation"])
        self.assertIn("0 shown", bounded["report"]["visual"][0]["text"])
        self.assertEqual(fit_process_report_capacity(bounded, max_bytes=budget,
            update_projection=update_visual_projection_count), bounded)
        self.assertEqual(report, grown)
        # Historical views cannot acquire a newly inferred omission policy.
        grown["report_projection"].pop("optional_statement_ids")
        with self.assertRaisesRegex(ValueError, "byte budget"):
            fit_process_report_capacity(grown, max_bytes=budget, update_projection=update_visual_projection_count)

    def test_visual_prose_without_alignment_and_visibility_is_not_admitted(self):
        s = statement("visual", kind="S3")
        s["window"] = {"start": 10, "end": 20}
        with self.assertRaisesRegex(ValueError, "media_identity"):
            validate_sentence(s)
        s.update(media_identity="media:fixture", alignment_status="same_run",
                 evidence_category="visual_only", visibility="partial")
        validate_sentence(s)

    def test_terminal_same_step_is_not_an_earlier_event(self):
        s = statement("process", kind="S1")
        s["window"] = dict(start=10, confirmed=20, end=20, terminal=20, before_terminal=True)
        with self.assertRaisesRegex(ValueError, "same-step"):
            validate_sentence(s)
        s["window"]["confirmed"] = 19
        validate_sentence(s)


    def test_comparison_requires_real_denominators_and_same_rule(self):
        s = statement("parent_diff", kind="S4")
        scope = {"label": "group", "numerator": 2, "denominator": 3, "rule_version": "fixture.v1"}
        s["comparison"] = dict(left=scope, right={**scope, "rule_version": "fixture.v2"}, paired=1, missing=0)
        with self.assertRaisesRegex(ValueError, "identical rule"):
            validate_sentence(s)
        s["comparison"]["right"]["rule_version"] = "fixture.v1"
        validate_sentence(s)
        s["comparison"]["right"] = {"label": "group", "rule_version": "fixture.v1"}
        with self.assertRaisesRegex(ValueError, "denominators"):
            validate_sentence(s)
        s["comparison"]["right"] = dict(scope)
        s["scope"]["numerator"] = 2
        with self.assertRaisesRegex(ValueError, "counts"):
            validate_sentence(s)


    def test_descriptive_term_cannot_be_discriminating_group_statement(self):
        s = statement("counterexample", kind="S2")
        s.update(window={"kind": "group_episodes"}, comparison={"status": "matched"}, descriptive=True)
        with self.assertRaisesRegex(ValueError, "descriptive"):
            validate_sentence(s)
        s["comparison"] = {
            "discriminating": False,
            "left": {"label": "group A", "numerator": 14, "denominator": 14},
            "right": {"label": "group B", "numerator": 13, "denominator": 13},
        }
        s["text"] = "Over the complete windows, group A 14/14; group B 13/13. This occurrence is not discriminating."
        validate_sentence(s)
        s["text"] = "This occurrence is not discriminating."
        with self.assertRaisesRegex(ValueError, "scoped counts"):
            validate_sentence(s)


    def test_budget_cannot_remove_required_counterexample_or_residual(self):
        rows = [statement(s) for s in SENTENCE_SECTIONS]
        optional = {**statement("process"), "id": "extra", "required": False, "text": "detail " * 1000}
        rows.append(optional)
        before = deepcopy(rows)
        kwargs = dict(envelope={"schema_version": "fixture.report.v1"}, actionability={"scope": "existing permissions"})
        report = render_process_report(**kwargs, sentences=rows, max_bytes=2000)
        assert report["report"]["counterexample"] == [{"id": "counterexample", "text": rows[2]["text"]}]
        assert report["report"]["gaps"][0]["id"] == "gaps"
        assert report["report_projection"]["omitted_statement_ids"] == ["extra"]
        assert report_bytes(report) <= 2000
        assert rows == before
        with self.assertRaisesRegex(ValueError, "budget"):
            render_process_report(**kwargs, sentences=rows, max_bytes=100)


    def test_missing_section_is_not_silently_replaced_by_generic_unknown(self):
        rows = [statement(s) for s in SENTENCE_SECTIONS if s != "visual"]
        with self.assertRaisesRegex(ValueError, "missing-section"):
            render_process_report(envelope={}, sentences=rows, actionability={"scope": "existing"}, max_bytes=8192)



    def test_absence_of_observation_is_not_normality(self):
        for text in ("没有异常", "The behavior is normal.", "No anomalies in the unseen interval."):
            with self.subTest(text=text):
                s = statement("gaps")
                s["text"] = text
                with self.assertRaisesRegex(ValueError, "normality"):
                    validate_sentence(s)
