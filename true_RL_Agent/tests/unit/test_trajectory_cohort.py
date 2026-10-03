"""Arithmetic and counterexamples for cohort observations, not just schema."""
from copy import deepcopy
import unittest

import numpy as np

from tests.unit.test_trajectory_measurement import sample_episode, FEET
from true_rl_agent.evaluation.trajectory_translation.cohort import (
    aggregate_cohort, aggregate_metric, compose_activity, observed_presence,
    select_examples, summarize_episode,
)
from true_rl_agent.evaluation.trajectory_translation.measurement import measure_episode


def summary(episode, *, include_activity=True):
    episode.task.update(cell_id="cell-a", stratum_id="stratum-a")
    header = {"episode_id": episode.episode_id, "display_name": f"fixture.{episode.env_index}",
              "source": episode.source, "entities": episode.entities,
              "run_index": episode.run_index, "env_index": episode.env_index, "task": episode.task,
              "coverage": {"valid_rows": len(episode.steps), "dt_s": episode.dt_s,
                           "first_step": int(episode.steps[0]), "last_step": int(episode.steps[-1])}}
    records = {r["key"]: r for r in measure_episode(episode) if r["record_type"] == "measurement"}
    refs = {key: {"record_id": r["record_id"]} for key, r in records.items()}
    return summarize_episode(header, records, refs, compose_activity(episode) if include_activity else None)


class CohortTests(unittest.TestCase):
    def test_activity_longest_run_preserves_alternating_entity_contributions(self):
        e = sample_episode(60)
        e.arrays["support_lin_vel_w"][25:40, 0, 0] = .1
        e.arrays["support_lin_vel_w"][40:, 1, 0] = .1
        d = compose_activity(e)["value"]["low_progress_any_foot_moving"]["longest_true_diagnostics"]
        self.assertEqual(d["interval"]["steps_half_open"], [26, 61])
        self.assertEqual(d["xy_ge_threshold_counts"], [15, 20, 0, 0])
        self.assertEqual(d["same_entity_ge_threshold_every_record"], [False] * 4)
        e.arrays["support_lin_vel_w"][25:, 0] = np.nan
        d = compose_activity(e)["value"]["low_progress_any_foot_moving"]["longest_true_diagnostics"]
        self.assertIsNone(d["same_entity_ge_threshold_every_record"][0])
        self.assertTrue(d["same_entity_ge_threshold_every_record"][1])

    def test_successful_long_mode_witness_and_same_cell_failure_both_selected(self):
        a, b = sample_episode(30), sample_episode(30)
        b.env_index = 1
        b.task["outcome"].update(success=False, primary_reason="timeout")
        for e in (a, b):
            e.arrays["support_contact_forces_w"][:, 0] = 0
        selected = select_examples([summary(a), summary(b)], set())
        self.assertEqual({x["episode_id"] for x in selected}, {a.episode_id, b.episode_id})
        self.assertTrue(any(any(r.startswith("successful_witness_of_mode") for r in x["reasons"]) for x in selected))
        self.assertTrue(any(any(r.startswith("same_cell_failure_witness_of_mode") for r in x["reasons"]) for x in selected))

    def test_force_share_eligibility_uses_p1a_sum_n_not_dt_scaled_integral(self):
        for dt, force, expected in ((.02, 2e-13, .25), (2., 1e-13, None)):
            e = sample_episode(2, dt=dt)
            e.arrays["support_contact_forces_w"][:, :, 2] = force
            for field in ("norm_share", "signed_fz_share"):
                self.assertEqual(summary(e)["metrics"][FEET[0] + "." + field]["value"], expected)

    def test_gap_cannot_prove_episode_absence_but_witness_proves_presence(self):
        e = sample_episode(steps=[1, 3])
        e.arrays["support_contact_forces_w"][:] = 0
        s = summary(e)
        m = s["metrics"][FEET[0] + ".marker"]
        self.assertIsNone(m["presence"])
        self.assertEqual((m["unknown_samples"], m["unobserved_control_steps"]), (0, 1))
        e.arrays["support_contact_forces_w"][1, 0, 2] = 2
        self.assertTrue(summary(e)["metrics"][FEET[0] + ".marker"]["presence"])

    def test_missing_foot_does_not_remove_root_or_other_foot_denominator(self):
        e = sample_episode(30)
        e.arrays["support_contact_forces_w"][:, 0] = np.nan
        s = summary(e)
        root = aggregate_metric([s], "net_progress_m")
        lost = aggregate_metric([s], FEET[0] + ".marker")
        other = aggregate_metric([s], FEET[1] + ".marker")
        self.assertEqual((root["valid_episode_count"], lost["unknown_episode_count"], other["valid_episode_count"]), (1, 1, 1))
        self.assertEqual(lost["observed_presence"]["valid_denominator"], 0)
        self.assertEqual(lost["unknown_samples"], 30)

    def test_norm_and_fz_have_independent_samples_and_episode_denominators(self):
        e = sample_episode(3)
        e.arrays["support_contact_forces_w"][:, 0, 0] = np.nan
        s = summary(e)
        norm = aggregate_metric([s], FEET[0] + ".norm_share")
        fz = aggregate_metric([s], FEET[0] + ".signed_fz_share")
        self.assertEqual((norm["valid_samples"], norm["unknown_episode_count"]), (0, 1))
        self.assertEqual((fz["valid_samples"], fz["valid_episode_count"]), (3, 1))
        self.assertEqual(fz["pooled"]["value"], .25)

    def test_zero_force_is_valid_off_but_share_unknown(self):
        e = sample_episode(3)
        e.arrays["support_contact_forces_w"][:] = 0
        s = summary(e)
        self.assertIs(s["metrics"][FEET[0] + ".marker"]["presence"], False)
        self.assertEqual(s["metrics"][FEET[0] + ".marker"]["value"], 0)
        self.assertIsNone(s["metrics"][FEET[0] + ".norm_share"]["value"])

    def test_unprocessed_unknown_and_negative_are_distinct(self):
        known = summary(sample_episode(2))
        missing = deepcopy(known)
        missing["metrics"].pop(FEET[0] + ".marker")
        unknown = deepcopy(known)
        unknown["metrics"][FEET[0] + ".marker"].update(value=None, presence=None, valid_samples=0, unknown_samples=2, numerator=0, denominator=0)
        stats = aggregate_metric([known, missing, unknown], FEET[0] + ".marker")
        self.assertEqual([stats[k] for k in ("total_episode_count", "valid_episode_count", "unknown_episode_count", "unprocessed_episode_count")], [3, 1, 1, 1])

    def test_different_dt_and_length_distinguish_episode_equal_from_cell_weight(self):
        a, b = sample_episode(2, dt=.1), sample_episode(4, dt=.2)
        a.arrays["support_contact_forces_w"][:] = 0
        stats = aggregate_metric([summary(a), summary(b)], FEET[0] + ".marker")
        self.assertEqual(stats["episode_equal"]["mean"], .5)
        self.assertAlmostEqual(stats["pooled"]["value"], .8)
        self.assertAlmostEqual(stats["pooled"]["denominator"], 1.)

    def test_endpoint_net_remains_valid_while_incomplete_path_is_unknown(self):
        e = sample_episode(steps=[1, 2, 4, 5])
        e.arrays["base_pos_w"][:, 0] = [0, 1, 100, 101]
        s = summary(e)
        self.assertEqual(s["metrics"]["net_progress_m"]["value"], 101)
        self.assertIsNone(s["metrics"]["complete_path_xy_m"]["value"])
        self.assertFalse(s["highlights"]["path_complete"])

    def test_presence_unknown_without_witness_not_negative(self):
        self.assertIsNone(observed_presence(0, 1))
        self.assertTrue(observed_presence(1, 100))
        self.assertIs(observed_presence(0, 0), False)
        e = sample_episode(30)
        e.arrays["base_pos_w"][:, 0] = np.arange(30) * .1
        s = summary(e)
        self.assertEqual(s["metrics"]["low_progress"]["unknown_samples"], 25)
        self.assertIsNone(s["metrics"]["low_progress"]["presence"])

    def test_complete_runs_exclude_left_and_right_censoring_and_keep_one_sample(self):
        e = sample_episode(7)
        e.arrays["support_contact_forces_w"][:, 0, 2] = [0, 0, 2, 0, 2, 0, 0]
        stats = aggregate_metric([summary(e)], FEET[0] + ".marker")["off_runs"]
        self.assertEqual((stats["count"], stats["complete_count"], stats["left_censored_count"], stats["right_censored_count"]), (3, 1, 1, 1))
        self.assertEqual(stats["complete_observed_span_s"]["count"], 1)
        self.assertEqual(stats["complete_observed_span_s"]["median"], 0.)

    def test_same_and_alternating_feet_do_not_become_same_long_state(self):
        e = sample_episode(60)
        e.arrays["support_contact_forces_w"][:30, 0] = 0
        e.arrays["support_contact_forces_w"][30:, 1] = 0
        s = summary(e)
        left = s["metrics"][FEET[0] + ".low_speed_off"]["runs"]["longest"]
        right = s["metrics"][FEET[1] + ".low_speed_off"]["runs"]["longest"]
        self.assertEqual((left["sample_count"], right["sample_count"]), (30, 30))
        self.assertEqual((left["steps_half_open"], right["steps_half_open"]), ([1, 31], [31, 61]))

    def test_low_progress_active_and_quiet_records_are_disjoint_and_xy_strict(self):
        e = sample_episode(60)
        e.arrays["support_lin_vel_w"][25:40, 0] = [.05, 0, 99]
        e.arrays["support_lin_vel_w"][40:, 0] = [.03, 0, 99]
        values = compose_activity(e)["value"]
        active = values["low_progress_any_foot_moving"]
        quiet = values["low_progress_all_feet_low_speed"]
        self.assertEqual((active["true_count"], quiet["true_count"]), (15, 20))
        self.assertEqual(quiet["unknown_count"], 25)
        self.assertEqual(active["unknown_count"], 0)  # Known nonmoving initial rows make conjunction false.

    def test_any_moving_witness_with_other_unknown_all_low_requires_all_feet(self):
        e = sample_episode(30)
        e.arrays["support_lin_vel_w"][25:, 0] = np.nan
        e.arrays["support_lin_vel_w"][25:28, 1, 0] = .1
        v = compose_activity(e)["value"]
        self.assertEqual(v["low_progress_any_foot_moving"]["true_count"], 3)
        self.assertEqual(v["low_progress_any_foot_moving"]["unknown_count"], 2)
        self.assertEqual(v["low_progress_all_feet_low_speed"]["unknown_count"], 27)

    def test_missing_velocity_and_progress_stay_unknown(self):
        e = sample_episode(30)
        del e.arrays["support_lin_vel_w"]
        self.assertEqual(compose_activity(e)["value"]["low_progress_all_feet_low_speed"]["unknown_count"], 30)
        del e.arrays["base_pos_w"]
        self.assertEqual(compose_activity(e)["value"]["low_progress_any_foot_moving"]["unknown_count"], 30)

    def test_current_history_separate_and_inversion_not_upright_foot_label(self):
        e = sample_episode(60)
        e.arrays["support_contact_forces_w"][:, 0] = 0
        e.arrays["support_contact_forces_w_history"][:, :, 0, 2] = 2
        e.arrays["base_quat_w"][10:] = [0, 1, 0, 0]
        s = summary(e)
        self.assertEqual(s["metrics"][FEET[0] + ".marker"]["value"], 0)
        self.assertEqual(s["metrics"][FEET[0] + ".history_diagnostic"]["value"], 1)
        self.assertEqual(s["highlights"]["first_negative"], 11)
        self.assertEqual(s["highlights"][FEET[0] + ".low_speed_off"]["minimum_up_z"], -1)

    def test_all_failure_no_parent_and_exact_cell_facets(self):
        a, b = sample_episode(), sample_episode()
        b.env_index = 1
        for e in (a, b):
            e.task["outcome"].update(success=False, primary_reason="timeout")
        sa, sb = summary(a), summary(b)
        sb["task"]["cell_id"] = "cell-b"
        c = aggregate_cohort([sa, sb], {"manifest_sha256": "a" * 64})
        self.assertEqual(c["groups"]["success"]["episode_count"], 0)
        self.assertEqual(c["groups"]["cell_id:cell-a"]["episode_count"], 1)
        self.assertEqual(c["parent_comparison"]["status"], "unavailable")
        self.assertIsNone(c["groups"]["success"]["metrics"]["net_progress_m"]["episode_equal"]["mean"])

    def test_cross_source_and_duplicate_rejected(self):
        s = summary(sample_episode())
        with self.assertRaisesRegex(ValueError, "duplicate"):
            aggregate_cohort([s, s], {"manifest_sha256": "a" * 64})
        with self.assertRaisesRegex(ValueError, "cross-source"):
            aggregate_cohort([s], {"manifest_sha256": "b" * 64})

    def test_representative_ties_deterministic_and_success_retreat_is_retained(self):
        a, b = sample_episode(), sample_episode()
        b.env_index = 1
        b.arrays["base_pos_w"][:, 0] = [0, 2, 1, 2]
        summaries = [summary(a), summary(b)]
        selected = select_examples(summaries, {a.episode_id})
        self.assertEqual(selected, select_examples(list(reversed(summaries)), {a.episode_id}))
        self.assertTrue(any("largest_observed_successful_retreat_with_return_not_typical" in e["reasons"] for e in selected))
        with self.assertRaisesRegex(ValueError, "absent"):
            select_examples(summaries, {"missing"})
