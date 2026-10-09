"""Push v3 descriptions and numerical coverage, with real CPU capture fixtures."""
from __future__ import annotations

import copy
import json
import unittest

from tests.unit import test_push_behavior
from true_rl_agent.evaluation.process_report import validate_sentence, report_bytes
from true_rl_agent.tasks.push_recovery.behavior import project_push_behavior, compare_push_behavior
from true_rl_agent.tasks.push_recovery.process_report import (
    build_push_process_behavior, build_push_residual, sample_push_coverage,
    select_push_process_representatives, PUSH_PROCESS_SCHEMA, PUSH_PROCESS_PUBLIC_SCHEMA,
    _window_description, _recovery_outline, build_push_process_sentences,
)


class PushProcessReportTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_push_behavior.PushBehaviorTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def full(self):
        legacy, _ = self.fixture.velocity_behavior()
        return legacy, build_push_process_behavior(legacy_full=legacy)

    def test_v3_is_separate_and_preserves_v2_projection_and_slots(self):
        legacy, full = self.full()
        before = copy.deepcopy(legacy)
        v2 = project_push_behavior(legacy)
        public = project_push_behavior(full)
        self.assertEqual(full["schema_version"], PUSH_PROCESS_SCHEMA)
        self.assertEqual(public["schema_version"], PUSH_PROCESS_PUBLIC_SCHEMA)
        self.assertEqual(legacy, before)
        self.assertEqual(project_push_behavior(legacy), v2)
        self.assertEqual(project_push_behavior(v2), v2)
        self.assertEqual(project_push_behavior(public), public)
        for row in full["sentences"]:
            validate_sentence(row)
        self.assertEqual(set(public["report"]), {"results", "process", "counterexample", "visual", "gaps", "parent_diff", "actionability"})
        self.assertNotIn("transcripts", public)
        self.assertNotIn("lexicon", public)
        self.assertNotIn("sentences", public)
        cards = {card["id"]: card["measurable_definition"] for card in full["lexicon"]}
        self.assertTrue(all(card.get("units") for card in cards.values()))
        # Composite response cards describe the actual saved sample fields,
        # including optional world-xy measurements rather than a single speed.
        for row in full["transcripts"]:
            for unit in row["recoveries"]:
                process = unit["process"]
                if process.get("status") != "available":
                    continue
                for card_name, field in (("first_response", "first_observed_response"),
                                         ("window_speed_peak", "window_speed_peak"), ("window_last", "window_last")):
                    sample = process.get(field)
                    if sample is not None:
                        self.assertLessEqual(set(sample), set(cards[card_name]["units"]))
                for span in process["stable_spans"]:
                    self.assertEqual(set(span), set(cards["all_conditions_stable_span"]["units"]))
        self.assertLessEqual(report_bytes(public), 12288)
        self.assertEqual(public["report"]["actionability"]["footstep_or_support_change"]["status"], "requires_new_observation")
        with self.assertRaises(ValueError):
            project_push_behavior(full, max_bytes=512)

    def test_no_latch_is_an_unavailable_interval_not_zero_instability(self):
        legacy, _ = self.full()
        row = legacy["transcripts"][0]
        unit = copy.deepcopy(row["recoveries"][0])
        unit["process"]["post_latch"] = {"status": "not_latched", "unstable_spans": [],
            "unstable_steps": 0, "observed_through_step": None, "until": "last_observation"}
        text = _window_description(row, unit)
        self.assertIn("post-latch interval unavailable", text)
        self.assertNotIn("0 unstable observed steps through None", text)

    def test_later_recovery_and_recurrence_survive_first_unlatched_detail(self):
        _, full = self.full()
        identity = full["representative_selection"]["context_episode_id"]
        row = next(r for r in full["transcripts"] if f"r{r['source'] + 1}.e{r['env_id']}" == identity)
        first, second = row["recoveries"]
        first["recovery_control_step"] = None
        first["max_dwell_steps"] = full["definitions"]["required_dwell_steps"] - 1
        first["process"]["post_latch"] = {"status": "not_latched", "unstable_spans": [],
            "unstable_steps": 0, "observed_through_step": None, "until": "next_executed_injection"}
        latch = second["recovery_control_step"]
        self.assertIsNotNone(latch)
        end = row["final_control_step"]
        self.assertLess(latch, end - 3)
        second["process"]["post_latch"] = {"status": "observed", "unstable_spans": [
            {"start_step": end - 3, "end_step_inclusive": end - 2},
            {"start_step": end, "end_step_inclusive": end}],
            "unstable_steps": 3, "observed_through_step": end, "until": "last_observation"}
        before = copy.deepcopy(full["transcripts"])
        full["sentences"] = build_push_process_sentences(full)
        example = next(s for s in full["sentences"] if s["id"] == "push.process.example")
        self.assertTrue(example["required"])
        text = example["text"]
        for phrase in ("condition combination found in", "This episode's sequence only:",
                       "event_0:", "latch not observed", "event_1:", f"latch {latch}",
                       "3 observed steps in 2 interval(s)", f"first {end - 3}-{end - 2}",
                       f"last {end}-{end}", "last interval reaches observation cutoff"):
            self.assertIn(phrase, text)
        self.assertEqual(text.count("Detail event_0:"), 1)
        self.assertNotIn("Detail event_1:", text)
        public = project_push_behavior(full)
        self.assertEqual(public["report"]["process"][0]["text"], text)
        self.assertEqual(full["transcripts"], before)
        self.assertIn("all 2 recovery units outlined", public["report"]["gaps"][-1]["text"])

    def test_outline_distinguishes_unexecuted_invalid_and_unobserved_windows(self):
        legacy, _ = self.full()
        row = copy.deepcopy(legacy["transcripts"][0])
        unit = row["recoveries"][1]
        unit.update(anchor_control_step=None, anchor_verified=False, observed_window_steps=0,
                    recovery_control_step=None, coverage="anchor_not_observed")
        unit["process"] = {"status": "no_verified_observed_window"}
        row["event_conditions"]["executed_events"] = 1
        text = _recovery_outline(row, unit)
        self.assertIn("event_1: not executed", text)
        self.assertIn("scheduled anchor", text)
        self.assertIn("post-latch interval unavailable", text)
        self.assertNotIn("0 observed steps", text)
        row["event_conditions"]["executed_events"] = 2
        unit.update(anchor_control_step=unit["scheduled_anchor_control_step"], coverage="invalid_injection")
        text = _recovery_outline(row, unit)
        self.assertIn("execution recorded; recovery window unavailable", text)
        self.assertNotIn("not executed", text)
        unit.update(anchor_verified=True, coverage="no_observed_window")
        self.assertIn("recovery window unavailable", _recovery_outline(row, unit))

    def test_outline_distinguishes_no_post_latch_samples_from_observed_zero(self):
        legacy, _ = self.full()
        row = legacy["transcripts"][0]
        unit = copy.deepcopy(row["recoveries"][1])
        latch = unit["recovery_control_step"]
        self.assertIsNotNone(latch)
        unit["process"]["post_latch"] = {"status": "observed", "unstable_spans": [],
            "unstable_steps": 0, "observed_through_step": latch, "until": "last_observation"}
        text = _recovery_outline(row, unit)
        self.assertIn(f"latch {latch}", text)
        self.assertIn("No post-latch observation", text)
        self.assertNotIn("0 observed steps", text)
        unit["process"]["post_latch"]["observed_through_step"] = latch + 1
        self.assertIn("0 observed steps", _recovery_outline(row, unit))

    def test_parent_and_unicode_pressure_drop_only_optional_visual_sentence(self):
        from tests.unit.test_process_report import ProcessReportTests
        _, full = self.full()
        public = project_push_behavior(full)
        parent = copy.deepcopy(public)
        parent["binding"]["checkpoint_hash"] = "sha256:" + "1" * 64
        full["parent_diff"] = compare_push_behavior(public, parent,
            expected_parent_checkpoint_hash=parent["binding"]["checkpoint_hash"])
        required = project_push_behavior(full)
        _, visual = ProcessReportTests().visual_rows()
        full["sentences"] = [row for row in full["sentences"] if row["section"] != "visual"] + [
            row for row in visual if row["section"] == "visual"]
        before = copy.deepcopy(full)
        limit = report_bytes(required) + 500
        projected = project_push_behavior(full, max_bytes=limit)
        self.assertEqual(projected["report_projection"]["omitted_statement_ids"], ["visual.observation"])
        for section in ("results", "process", "counterexample", "parent_diff", "actionability"):
            self.assertEqual(projected["report"][section], required["report"][section])
        self.assertEqual(projected["report"]["gaps"][:-1], required["report"]["gaps"])
        self.assertIn("0 shown", projected["report"]["visual"][0]["text"])
        self.assertIn("3 numeric conflicts", projected["report"]["visual"][0]["text"])
        self.assertEqual(project_push_behavior(projected, max_bytes=limit), projected)
        self.assertEqual(full, before)
        self.assertLessEqual(report_bytes(projected), limit)

    def test_residual_window_signature_controls_minimum_and_episode_aggregation(self):
        legacy, _ = self.full()
        failure, success = copy.deepcopy(legacy["transcripts"])
        failure["event_conditions"] = copy.deepcopy(success["event_conditions"])
        failure["recoveries"] = copy.deepcopy(success["recoveries"])
        failure["terminal_label"] = "not_recovered"
        samples = [failure]
        for i in range(3):
            row = copy.deepcopy(success)
            row["env_id"] = 20 + i
            samples.append(row)
        data = {**legacy, "transcripts": samples}
        residual = build_push_residual(data)
        self.assertEqual(residual["episodes"][0]["state"], "negative")
        self.assertTrue(all(unit["success_windows"] == 3 for unit in residual["episodes"][0]["windows"]))
        data["transcripts"] = samples[:3]
        self.assertEqual(build_push_residual(data)["episodes"][0]["state"], "missing")
        data["transcripts"] = samples
        # The same condition set at a different first-violation time bin has
        # distinguishing evidence; another unavailable window cannot erase it.
        unit = samples[0]["recoveries"][0]
        name = next(name for name, spans in unit["condition_violation_spans"].items() if spans)
        unit["condition_violation_spans"][name][0]["start_step"] = unit["anchor_control_step"] + 200
        samples[0]["recoveries"][1]["anchor_verified"] = False
        self.assertEqual(build_push_residual(data)["episodes"][0]["state"], "positive")
        samples[0]["recoveries"] = []
        self.assertEqual(build_push_residual(data)["episodes"][0]["state"], "missing")

    def test_shared_explanations_are_once_without_removing_scope_or_rule_slots(self):
        _, full = self.full()
        text = " ".join(row["text"] for row in full["sentences"])
        self.assertEqual(text.count("matching preserves event counts and nominal intensity"), 1)
        self.assertEqual(text.count("Rule-negative is a rule label, not no behavior"), 1)
        for row in full["sentences"]:
            self.assertEqual(row["rule_version"], "push_process_v3")
            if row["kind"] == "S2":
                self.assertIn("executed event(s)", row["text"])
                self.assertIn("whole observed recovery windows", row["text"])
                self.assertIn("Per-condition occurrences", row["text"])
            if row["id"].startswith("push.gaps.domain."):
                self.assertIn("failed episodes", row["text"])
                self.assertIn("rule-negative", row["text"])
                self.assertIn("unexamined", row["text"])
        # The author sees only this projection, not the full residual contract.
        delivered = json.dumps(project_push_behavior(full)["report"])
        for phrase in ("first-onset bins (<= 0.5, (0.5,2], > 2 s after anchor)",
                       "Compare same stratum, executed-event count and recovery ordinal",
                       "< 3 successful controls", ">= 50% share its exact signature",
                       "Rule-negative is a rule label, not no behavior",
                       "comparison-missing if unverified/unobserved",
                       "missing need not mean lost raw data", "Episode: any positive wins"):
            self.assertIn(phrase, delivered)

    def test_outcome_pathways_preserve_latch_final_state_and_success_counterexample(self):
        _, full = self.full()
        failure, success = copy.deepcopy(full["transcripts"])
        failure.update(terminal_label="not_recovered", all_required_recoveries_latched=True,
                       final_stable=False)
        missed = copy.deepcopy(failure)
        missed.update(env_id=30, all_required_recoveries_latched=False, final_stable=True)
        success.update(terminal_label="success_recovered", all_required_recoveries_latched=True,
                       final_stable=True, unstable_steps_after_all_recoveries=3)
        fall = copy.deepcopy(failure)
        fall.update(env_id=31, terminal_label="fall_side")
        full["transcripts"] = [failure, missed, success, fall]
        sentence = next(row for row in build_push_process_sentences(full)
                        if row["id"] == "push.process.outcome_pathways")
        validate_sentence(sentence)
        self.assertTrue(sentence["required"])
        self.assertIn("1/2 all required recoveries latched, then final-unstable", sentence["text"])
        self.assertIn("1/2 final-stable with at least one required latch absent", sentence["text"])
        self.assertIn("1/1 unstable after all latches, then final-stable", sentence["text"])
        self.assertEqual(sentence["scope"]["numerator"], 3)
        self.assertEqual(sentence["scope"]["denominator"], 4)

    def test_unexecuted_event_does_not_borrow_other_event_count_controls(self):
        legacy, _ = self.full()
        failure, success = copy.deepcopy(legacy["transcripts"])
        failure["event_conditions"]["executed_events"] = 1
        failure["recoveries"][1]["anchor_verified"] = False
        failure["recoveries"][1]["anchor_control_step"] = None
        controls = []
        for i in range(4):
            row = copy.deepcopy(success)
            row["env_id"] = 30 + i
            controls.append(row)
        residual = build_push_residual({**legacy, "transcripts": [failure, *controls]})
        self.assertEqual(residual["episodes"][0]["state"], "missing")
        self.assertEqual(len(residual["episodes"][0]["windows"]), 1)
        self.assertTrue(all(unit["success_windows"] == 0 for unit in residual["episodes"][0]["windows"]))

    def test_coverage_is_replayable_rule_independent_and_unions_steps(self):
        legacy, full = self.full()
        sample = full["coverage_sampling"]
        self.assertEqual(sample_push_coverage(full, full["residual"]), sample)
        shuffled = copy.deepcopy(full)
        shuffled["transcripts"].reverse()
        self.assertEqual(sample_push_coverage(shuffled, full["residual"]), sample)
        self.assertTrue(all(item["sampled"] >= 1 for item in sample["by_stratum_outcome"]))
        for item in sample["per_episode"]:
            windows = [w for w in sample["windows"] if w["episode_id"] == item["episode_id"]]
            steps = {step for window in windows for step in range(window["start_step"], window["end_step"] + 1)}
            self.assertEqual(item["covered_steps"], len(steps))
            self.assertLessEqual(item["covered_steps"], item["total_steps"])
        self.assertEqual(sample["residual_cap_selected"], min(sample["residual_requested"], 4))
        self.assertIn("media", sample["rule"]["scope"])

    def test_modal_combination_precedes_duration_and_ties_are_disclosed(self):
        legacy, _ = self.full()
        base = legacy["transcripts"][0]
        rows = []
        for index in range(5):
            row = copy.deepcopy(base)
            row["env_id"] = index
            row["valid_steps"] = (10, 20, 30, 1000, 2000)[index]
            row["final_control_step"] = 3000
            if index >= 3:
                for unit in row["recoveries"]:
                    for spans in unit["condition_violation_spans"].values():
                        spans.clear()
            rows.append(row)
        altered = copy.deepcopy(legacy)
        altered["transcripts"] = rows
        focus = copy.deepcopy(altered["profile"]["matched_failure_profiles"][0])
        focus["conditions"] = copy.deepcopy(base["event_conditions"])
        focus["failure_label"] = base["terminal_label"]
        altered["profile"]["matched_failure_profiles"] = [focus]
        selected = select_push_process_representatives(altered)
        self.assertEqual(selected["pattern_n"], 3)
        self.assertEqual(selected["group_n"], 5)
        self.assertEqual(selected["failure_episode_id"], f"r{base['source'] + 1}.e1")
        altered["transcripts"] = [rows[0], rows[1], rows[3], rows[4]]
        tied = select_push_process_representatives(altered)
        self.assertTrue(tied["tied"])
        self.assertFalse(tied["majority"])
        self.assertEqual(tied["pattern"], [])

    def test_parent_diff_renders_both_process_denominators_and_replays(self):
        legacy, full = self.full()
        parent = copy.deepcopy(full)
        parent["binding"]["checkpoint_hash"] = "sha256:" + "1" * 64
        public, previous = project_push_behavior(full), project_push_behavior(parent)
        diff = compare_push_behavior(public, previous, expected_parent_checkpoint_hash=previous["binding"]["checkpoint_hash"])
        public["parent_diff"] = diff
        rendered = project_push_behavior(public)
        self.assertIn("all-recovered then final-unstable", rendered["report"]["parent_diff"][0]["text"])
        self.assertEqual(project_push_behavior(rendered), rendered)
        self.assertLessEqual(report_bytes(rendered), 12288)
        self.assertEqual(compare_push_behavior(public, legacy, expected_parent_checkpoint_hash=legacy["binding"]["checkpoint_hash"])["reason"], "behavior_semantics_version_mismatch")

    def test_no_failure_is_zero_failure_residual_not_missing_success(self):
        from true_rl_agent.tasks.push_recovery.behavior import _group, _matched_profiles
        legacy, _ = self.full()
        rows = [row for row in legacy["transcripts"] if row["terminal_label"] == "success_recovered"]
        legacy["transcripts"] = rows
        legacy["profile"]["all"] = _group(rows)
        legacy["profile"]["by_stratum"] = {row["stratum"]: _group([r for r in rows if r["stratum"] == row["stratum"]]) for row in rows}
        legacy["profile"]["matched_failure_profiles"] = _matched_profiles(rows)
        full = build_push_process_behavior(legacy_full=legacy)
        public = project_push_behavior(full)
        self.assertTrue(all(item["failures"] == item["negative"] == item["missing"] == 0 for item in full["residual"]["domains"]))
        self.assertIsNotNone(full["representative_selection"]["success_episode_id"])
        self.assertIn("no_failure_in_cohort_success_context", public["report"]["counterexample"][0]["text"])
        self.assertTrue(any("0/1 failed episodes" in row["text"] for row in public["report"]["process"]))

    def test_full_rows_cards_and_sample_windows_remain_out_of_prompt(self):
        from tests.unit.test_push_recovery_v2_prompt_smoke import _dev
        from tests.fakes.decision_records import valid_accepted_ledger_record
        from true_rl_agent.tasks.push_recovery.config import PushRecoveryTaskConfig
        from true_rl_agent.tasks.push_recovery.prompt_adapter import PushRecoveryPromptAdapter
        _, full = self.full()
        record = valid_accepted_ledger_record(round_id="process", candidate_id="winner", public_score=.75)
        record["public_result"]["metrics"]["public_dev"] = {**_dev(), "behavior": project_push_behavior(full),
            "behavior_full": {**full, "canary": "full-archive-must-not-reach-author"}}
        adapter = PushRecoveryPromptAdapter(PushRecoveryTaskConfig.for_cell("push.go2_12dof.recovery_v2"))
        projected = adapter.project_prompt_public_history(public_records=[record], initial_anchor_public_evidence=None)
        sections = adapter.build_prompt_sections(public_records=projected.records, public_context={})
        serialized = json.dumps(sections)
        self.assertNotIn("full-archive-must-not-reach-author", serialized)
        self.assertNotIn('"behavior_full"', serialized)
        self.assertIn('"report"', serialized)


if __name__ == "__main__":
    unittest.main()
