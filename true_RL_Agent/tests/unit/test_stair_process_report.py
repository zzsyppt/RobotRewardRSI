"""Independent small examples for the Stair N.1/N.2 public contract."""
from copy import deepcopy
import json
import unittest

from true_rl_agent.evaluation.process_report import validate_sentence
from true_rl_agent.tasks.stair_climbing.behavior import project_stair_behavior, compare_stair_behavior, _progress_process
from true_rl_agent.tasks.stair_climbing.process_report import (
    CONTRACT, FULL_SCHEMA, PUBLIC_SCHEMA, PATTERN_TERMS, MAX_PUBLIC_BYTES,
    derive_stair_process_behavior, stair_process_signature, build_stair_residuals,
    select_stair_observation_windows, build_stair_term_cards, _representative,
    record_stair_candidate_term, attach_stair_process_difference,
    _progress_outline,
)
from tests.unit.test_p3r_stair_projection import episode, motion_full, legacy_view, full


def measured(index, *, success=False, cell="cell-a", run=1, stratum="frontier", terms=("tilt",)):
    value = motion_full([episode(index, reason="success" if success else "forbidden_contact", cell=cell, run=run)])["episodes"][0]
    value["stratum"] = stratum
    value["events"] = [{"code": term, "body": "foot" if term == "unloaded_low_motion" else "root",
                        "start": 55, "confirmed": 60, "end": 100} for term in terms if term != "up_z_negative"]
    value["root_motion"]["first_up_z_negative_step"] = 60 if "up_z_negative" in terms else None
    value["before_terminal"] = {"tilt": "tilt" in terms, "slip_any": False,
                                "low_clearance_any": False, "low_progress": "low_progress" in terms,
                                "unloaded_low_motion": "unloaded_low_motion" in terms}
    return value


def source(episodes):
    # motion_full fills a valid v2 envelope but overwrites root measurements;
    # retain the independent episode fixtures after constructing the envelope.
    result = motion_full(episodes)
    result["episodes"] = deepcopy(episodes)
    result["semantics"]["thresholds"]["forbidden_N"] = 1.0
    return result


class StairProcessReportTests(unittest.TestCase):
    def pool(self):
        return source([measured(0), measured(1), measured(2, terms=("low_progress",)),
                       measured(3, success=True), measured(4, success=True), measured(5, success=True)])

    def test_progress_return_and_censored_tail_preserve_different_meanings(self):
        # Independent miniature trajectory: two returns, then an unfinished loss.
        x = [0., .5, .4, .2, .45, .5, .7, .69, .7, .75, .3]
        samples = [[v, 0., -.8 if step == 4 else -.1, 1.] for step, v in enumerate(x, 1)]
        events = [{"code": "low_progress", "body": "root", "start": 3, "end": 4, "confirmed": 3},
                  {"code": "tilt", "body": "root", "start": 4, "end": 5, "confirmed": None}]
        process = _progress_process(samples, events)
        rows = process["excursions"]
        self.assertEqual([(r["peak_step"], r["trough_step"], r["recovery_step"], r["drop_m"])
                          for r in rows], [(2, 4, 6, .3), (7, 8, 9, .01), (10, 11, None, .45)])
        self.assertEqual(rows[0]["attitude_extrema"]["pitch_rad"], [-.8, 4, -.1, 2])
        self.assertEqual(rows[0]["overlapping_root_event_indices"], [0, 1])
        self.assertEqual(events[1]["confirmed"], None)
        self.assertEqual(rows[-1]["observed_end_step"], 11)
        self.assertEqual(rows[-1]["end_x_m"], .3)

    def test_equal_peak_starts_new_excursion_and_no_retreat_is_not_missing(self):
        samples = [[v, 0., 0., 1.] for v in [0., 1., 1., .5, .5, 1., .8]]
        rows = _progress_process(samples, [])["excursions"]
        self.assertEqual([(r["peak_step"], r["trough_step"], r["recovery_step"]) for r in rows],
                         [(3, 4, 6), (6, 7, None)])
        self.assertEqual(_progress_process([[0., 0., 0., 1.]], [])["excursions"], [])
        self.assertEqual(_progress_process([[x, 0., 0., 1.] for x in [0., 0., 1.]], [])["excursions"], [])

    def test_short_progress_retains_unreturned_tail_and_names_omissions(self):
        value = measured(0)
        # Largest two closed excursions must not hide a smaller unfinished one.
        samples = [[v, 0., 0., 1.] for v in [0., 1., .2, 1., 2., 1.5, 2., 3., 2.9]]
        value["progress_process"] = _progress_process(samples, [])
        value["steps"] = 9
        value["root_motion"]["first"][0] = 0.
        value["root_motion"]["last"][0] = 2.9
        text, refs = _progress_outline(value)
        self.assertIn("2/3", text)
        self.assertIn("2->3", text)
        self.assertNotIn("5->6", text)
        self.assertIn("not re-attained through terminal 9", text)
        self.assertNotIn("balance restored", text)
        self.assertTrue(refs[-1].endswith("/excursions/2"))
        del value["progress_process"]
        self.assertIsNone(_progress_outline(value))

    def test_progress_attitude_does_not_hide_large_roll_behind_small_pitch(self):
        value = measured(0)
        value["steps"] = 4
        samples = [[0., 0., -.1, 1.], [1., .1, -.1, 1.],
                   [.8, 3.08, -.2, -.99], [.5, -3.02, -.3, -.95]]
        value["progress_process"] = _progress_process(samples, [])
        text, _ = _progress_outline(value)
        self.assertIn("signed roll [-3.020,3.080], pitch [-0.300,-0.100] rad over 2-4", text)
        self.assertIn("not re-attained through terminal 4", text)
        public = project_stair_behavior(derive_stair_process_behavior(source([value])))
        self.assertIn("Euler angles wrap", public["report"]["results"][0]["text"])

    def test_previous_process_public_is_readable_but_not_silently_compared_as_v2(self):
        current = project_stair_behavior(derive_stair_process_behavior(self.pool()))
        historical = deepcopy(current)
        historical["derivation_version"] = "stair.process_report.v1"
        self.assertEqual(project_stair_behavior(historical), historical)
        self.assertEqual(compare_stair_behavior(historical, current,
            expected_parent_checkpoint_hash=current["domain"]["checkpoint_hash"]),
            {"status": "not_comparable", "reason": "process_report_version_mismatch"})

    def test_full_derivation_preserves_every_old_measurement_and_is_bounded(self):
        old = self.pool()
        snapshot = deepcopy(old)
        derived = derive_stair_process_behavior(old)
        self.assertEqual(old, snapshot)
        self.assertEqual(derived["schema_version"], FULL_SCHEMA)
        for key in old:
            if key != "schema_version":
                self.assertEqual(derived[key], old[key])
        for row in derived["semantic"]["sentences"]:
            validate_sentence(row)
        public = project_stair_behavior(derived)
        self.assertEqual(public["schema_version"], PUBLIC_SCHEMA)
        self.assertEqual(public["lexicon_version"], CONTRACT["lexicon_version"])
        self.assertTrue(all(row["rule_version"] == public["lexicon_version"] for row in derived["semantic"]["sentences"]))
        narrative = " ".join(row["text"] for section, rows in public["report"].items()
                             if section != "actionability" for row in rows)
        self.assertNotIn("Rule stair.process_terms.v1", narrative)
        self.assertEqual(narrative.count("Validated visual observations: 0/"), 1)
        self.assertEqual(narrative.count("sampling is not visual coverage"), 1)
        self.assertEqual(project_stair_behavior(public), public)
        self.assertLessEqual(len(json.dumps(public, separators=(",", ":")).encode()), MAX_PUBLIC_BYTES)
        self.assertNotIn("episodes", public)
        self.assertNotIn("term_cards", public)
        self.assertNotIn("columns", json.dumps(public))
        self.assertEqual(set(public["report"]), {"results", "process", "counterexample", "visual", "gaps", "parent_diff", "actionability"})

    def test_three_residual_states_and_control_precedence(self):
        pool = self.pool()
        pool["episodes"].append(measured(6, cell="no-controls", terms=()))
        result = build_stair_residuals(pool)
        states = {row["episode_id"]: row["status"] for row in result["episodes"]}
        self.assertEqual(states, {"r1.e0": "rule_negative", "r1.e1": "rule_negative", "r1.e2": "rule_positive", "r1.e6": "comparison_missing"})
        self.assertEqual(result["failure_n"], 4)
        self.assertEqual(sum(result[name] for name in ("rule_positive", "rule_negative", "comparison_missing")), 4)
        self.assertEqual([row["success_n"] for row in result["domains"]], [3, 0])

    def test_equal_success_occurrence_is_explicitly_not_discriminating(self):
        old = source([measured(i, success=i >= 9, terms=("low_progress",)) for i in range(11)])
        full = derive_stair_process_behavior(old)
        group = next(row for row in full["semantic"]["sentences"] if row["id"] == "contrast-0")
        self.assertTrue(group["descriptive"])
        self.assertFalse(group["comparison"]["discriminating"])
        self.assertIn("9/9", group["text"])
        self.assertIn("2/2", group["text"])
        self.assertIn("not discriminating", group["text"].lower())
        validate_sentence(group)

    def test_empty_signature_is_negative_only_with_sufficient_controls(self):
        pool = source([measured(0, terms=()), *[measured(i, success=True) for i in range(1, 4)]])
        self.assertEqual(build_stair_residuals(pool)["rule_negative"], 1)
        pool["episodes"].pop()
        self.assertEqual(build_stair_residuals(pool)["comparison_missing"], 1)

    def test_public_report_explains_insufficient_controls_without_calling_measurements_missing(self):
        pool = source([measured(0), measured(1, success=True)])
        derived = derive_stair_process_behavior(pool)
        public = project_stair_behavior(derived)
        contrast = next(row["text"] for row in public["report"]["process"] if row["id"] == "contrast-0")
        residual = next(row["text"] for row in public["report"]["gaps"] if row["id"] == "residual-total")
        self.assertIn("matched success 1/1", contrast)
        self.assertIn("comparison-missing 1/1", contrast)
        self.assertIn("fewer than 3 same-cell/run successes, not missing measurements", residual)
        self.assertIn("empty four-term pattern", residual)
        self.assertIn(">=50%", residual)
        self.assertIn("does not mean no movement", residual)
        self.assertIn("<= 0.5s, <= 2s, > 2s", residual)
        self.assertEqual(derived["semantic"]["residuals"], build_stair_residuals(pool))

    def test_primary_success_example_does_not_hide_other_successful_process_terms(self):
        pool = source([measured(0, terms=("low_progress",)), measured(1, success=True, terms=()),
                       measured(2, success=True, cell="other-cell", terms=("tilt", "low_progress"))])
        public = project_stair_behavior(derive_stair_process_behavior(pool))
        rows = {row["id"]: row["text"] for row in public["report"]["counterexample"]}
        self.assertIn("Same-cell/run success", rows["compatible-success"])
        self.assertIn("not matched controls", rows["unmatched-success-context"])
        self.assertIn("root tilt 1/2", rows["unmatched-success-context"])
        self.assertIn("low root forward progress 1/2", rows["unmatched-success-context"])
        self.assertIn("Counts may overlap", rows["unmatched-success-context"])

    def test_signature_uses_strict_terminal_and_frozen_time_bins(self):
        item = measured(0, terms=PATTERN_TERMS)
        item["events"][0]["confirmed"] = 100
        item["root_motion"]["first_up_z_negative_step"] = 100
        self.assertEqual(stair_process_signature(item), (("low_progress", "le_2s"), ("unloaded_low_motion", "le_2s")))
        item["events"][1]["confirmed"] = 75
        item["dt_s"] = .02
        self.assertIn(("low_progress", "le_0.5s"), stair_process_signature(item))
        item["steps"] = 200
        self.assertIn(("low_progress", "gt_2s"), stair_process_signature(item))

    def test_time_bins_distinguish_same_terms(self):
        pool = self.pool()
        for item in pool["episodes"][3:]:
            item["events"][0]["confirmed"] = 90
        self.assertEqual(build_stair_residuals(pool)["rule_positive"], 3)

    def test_representative_uses_all_four_conditions_then_median_and_identifier(self):
        items = [measured(i, terms=("tilt", "low_progress", "up_z_negative") if i < 3 else ("tilt", "low_progress")) for i in range(5)]
        items[0]["steps"], items[1]["steps"], items[2]["steps"] = 100, 200, 200
        chosen, selection = _representative(items)
        self.assertEqual(chosen["episode_id"], "r1.e1")
        self.assertEqual(selection["pattern_n"], 3)
        self.assertEqual(selection["group_n"], 5)
        self.assertTrue(selection["majority"])
        self.assertTrue(selection["terms"]["up_z_negative"])
        tied = [measured(0, terms=()), measured(1, terms=("unloaded_low_motion",))]
        chosen, selection = _representative(tied)
        self.assertTrue(selection["tie"])
        self.assertFalse(selection["majority"])
        self.assertEqual(chosen["episode_id"], "r1.e0")

    def test_sampling_is_order_independent_clips_unions_and_names_unseen(self):
        pool = source([measured(i, success=i%2 == 0, cell="x" + str(i%3), stratum="s" + str(i%2)) for i in range(12)])
        pool["episodes"][0]["steps"] = 2
        residuals = build_stair_residuals(pool)
        sampling = select_stair_observation_windows(pool, residuals)
        reverse = deepcopy(pool)
        reverse["episodes"].reverse()
        self.assertEqual(sampling, select_stair_observation_windows(reverse, build_stair_residuals(reverse)))
        self.assertEqual(sampling["supplemental_selected_n"], min(4, sampling["residual_eligible_n"]))
        self.assertEqual(len(set(sampling["supplemental_episode_ids"])), sampling["supplemental_selected_n"])
        self.assertTrue(all(row["selected_n"] >= 1 for row in sampling["episode_coverage"]))
        self.assertEqual(sampling["visual_observed_episode_n"], 0)
        for row in sampling["time_coverage"]:
            points = {n for start, end in row["union"] for n in range(start, end+1)}
            self.assertEqual(len(points), row["covered_steps"])
            self.assertLessEqual(max(points), row["total_steps"])
            self.assertGreaterEqual(min(points), 1)
        chosen = set(sampling["selected_episode_ids"])
        unseen = set(sampling["unselected_episode_ids"])
        self.assertFalse(chosen & unseen)
        self.assertEqual(len(chosen | unseen), 12)

    def test_cards_use_actual_development_counts_and_explicit_counterexamples(self):
        cards = {card["id"]: card for card in build_stair_term_cards(self.pool())}
        self.assertEqual(cards["tilt"]["development_discrimination"][0]["failed_present"], 2)
        self.assertEqual(cards["tilt"]["development_discrimination"][0]["success_present"], 3)
        self.assertEqual(cards["tilt"]["hard_counterexample"]["kind"], "observed_success")
        self.assertEqual(cards["loaded_point_motion"]["status"], "formal_descriptive")
        for card in cards.values():
            self.assertTrue(card["definition"]["operands"])
            self.assertTrue(card["does_not_mean"])
            self.assertTrue(card["missing_field_degradation"])
            self.assertTrue(card["hard_counterexample"])

    def test_success_only_keeps_zero_failure_residual_and_missing_sections(self):
        public = project_stair_behavior(derive_stair_process_behavior(source([measured(i, success=True) for i in range(3)])))
        text = json.dumps(public)
        self.assertIn("0 failed episodes; 0 rule-negative, 0 comparison-missing", text)
        self.assertTrue(all(public["report"][key] for key in public["report"]))

    def test_legacy_outputs_and_mixed_comparison_are_unchanged_or_explicit(self):
        old_v1 = legacy_view()
        old_v2 = project_stair_behavior(full([episode(0)]))
        old_v3 = project_stair_behavior(self.pool())
        for old in (old_v1, old_v2, old_v3):
            self.assertEqual(project_stair_behavior(old), old)
        new = project_stair_behavior(derive_stair_process_behavior(self.pool()))
        difference = compare_stair_behavior(old_v3, new, expected_parent_checkpoint_hash="a"*64)
        self.assertEqual(difference["reason"], "process_report_version_mismatch")

    def test_comparison_requires_parent_binding_and_identical_cohort(self):
        parent = project_stair_behavior(derive_stair_process_behavior(self.pool()))
        candidate = deepcopy(parent)
        candidate["domain"]["checkpoint_hash"] = "b"*64
        difference = compare_stair_behavior(parent, candidate, expected_parent_checkpoint_hash="a"*64)
        self.assertEqual(difference["status"], "available")
        self.assertEqual(difference["paired_cell_run_n"], 1)
        self.assertEqual(difference["missing_cell_run_n"], 0)
        self.assertIn(difference["statement"]["predicate"], PATTERN_TERMS)
        validate_sentence(difference["statement"])
        projected = attach_stair_process_difference(candidate, difference)
        self.assertEqual(projected["report"]["parent_diff"][0]["id"], "parent-process-diff")
        candidate["domain"]["cohort_hash"] = "c"*64
        self.assertEqual(compare_stair_behavior(parent, candidate, expected_parent_checkpoint_hash="a"*64)["reason"], "comparison_domain_mismatch")

    def test_candidate_names_are_a_separate_append_only_proposal_ledger(self):
        derived = derive_stair_process_behavior(self.pool())
        prior = deepcopy(derived["semantic"]["residuals"])
        ledger = derived["semantic"]["candidate_lexicon"]
        updated = record_stair_candidate_term(ledger, episode_id="r1.e0", start=2, end=8,
            observer_version="observer.dev.v1", original_name="one side extended")
        self.assertFalse(ledger["entries"])
        self.assertEqual(updated["entries"][0]["status"], "candidate")
        self.assertEqual(updated["entries"][0]["occurrence_count"], 1)
        with self.assertRaisesRegex(ValueError, "merge record"):
            record_stair_candidate_term(updated, episode_id="r1.e1", start=2, end=8,
                observer_version="observer.dev.v1", original_name="asymmetric limbs", normalized_name="one side extended")
        updated = record_stair_candidate_term(updated, episode_id="r1.e1", start=2, end=8,
            observer_version="observer.dev.v1", original_name="asymmetric limbs", normalized_name="one side extended",
            merge_record={"actor": "developer", "reason": "same descriptive meaning"})
        self.assertEqual(updated["entries"][0]["occurrence_count"], 2)
        derived["semantic"]["candidate_lexicon"] = updated
        self.assertEqual(build_stair_residuals(derived), prior)

    def test_required_capacity_failure_cannot_erase_counterexample_or_residual(self):
        huge = self.pool()
        huge["episodes"] = [measured(i, success=i%2 == 0, cell="domain-" + str(i), stratum="stratum-" + str(i)) for i in range(100)]
        with self.assertRaisesRegex(ValueError, "byte budget"):
            project_stair_behavior(derive_stair_process_behavior(huge))


if __name__ == "__main__":
    unittest.main()
