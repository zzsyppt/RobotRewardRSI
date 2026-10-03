"""Mutation gates for the raw-array audit, independent of card rendering."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

import numpy as np

from tests.unit.test_trajectory_measurement import sample_episode, FEET
from tests.unit.test_trajectory_cohort import summary
from true_rl_agent.evaluation.trajectory_translation.cohort import aggregate_cohort

SPEC = importlib.util.spec_from_file_location("verify_policy_behavior_card", Path(__file__).resolve().parents[2] / "scripts/verify_policy_behavior_card.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class CardRawAuditTests(unittest.TestCase):
    def setUp(self):
        self.episode = sample_episode(60)
        e = self.episode
        e.task["outcome"].update(bin_id="public_dev.stratum-a_level_0", evaluation_cell_id="cell-a")
        e.arrays["episode_length"] = e.steps.astype(float)
        e.arrays["support_lin_vel_w"][25:40, 0, 0] = .1
        e.arrays["support_lin_vel_w"][40:, 1, 0] = .1
        e.arrays["forbidden_contact_forces_w_history"][-1, 0, 0, 2] = 10
        e.arrays["base_quat_w"][45:] = [0, 1, 0, 0]
        self.summary = summary(e)

    def verify_episode(self, s):
        e = self.episode
        return audit.audit_episode(s, e.arrays, e.dt_s, np.zeros(3), e.task["outcome"],
                                   {k: list(v) for k, v in e.entities.items()})

    def test_all_metrics_and_intervals_match_independent_raw_calculations(self):
        result = self.verify_episode(self.summary)
        self.assertEqual(result["metrics_checked"], len(self.summary["metrics"]))

    def test_missing_history_unknown_is_checked_against_absent_raw_arrays(self):
        for name in ("support_contact_forces_w_history", "forbidden_contact_forces_w_history"):
            del self.episode.arrays[name]
        self.episode.source["contact_force_history"] = {"status": "not_recorded"}
        expected = summary(self.episode)
        self.verify_episode(expected)
        key = "robot_0.body.base.history_diagnostic"
        for field, value in (("value", 0.0), ("presence", False), ("unknown_samples", 0)):
            bad = deepcopy(expected)
            bad["metrics"][key][field] = value
            with self.subTest(field=field), self.assertRaises(AssertionError):
                self.verify_episode(bad)

    def test_unknown_progress_window_cannot_be_counted_as_negative(self):
        bad = deepcopy(self.summary)
        bad["metrics"]["low_progress"]["unknown_samples"] = 0
        with self.assertRaisesRegex(AssertionError, "unknown_samples"):
            self.verify_episode(bad)

    def test_current_history_swap_rejected_by_raw_not_claim_template(self):
        bad = deepcopy(self.summary)
        bad["metrics"]["robot_0.body.base.current_marker"] = deepcopy(bad["metrics"]["robot_0.body.base.history_diagnostic"])
        with self.assertRaises(AssertionError):
            self.verify_episode(bad)

    def test_alternating_foot_contributor_count_and_entity_order_mutations(self):
        for field, value in (("xy_ge_threshold_counts", [35, 0, 0, 0]),
                             ("entity_ids", list(reversed(FEET))),
                             ("same_entity_ge_threshold_every_record", [True, False, False, False])):
            bad = deepcopy(self.summary)
            bad["activity"]["value"]["low_progress_any_foot_moving"]["longest_true_diagnostics"][field] = value
            with self.assertRaises(AssertionError):
                self.verify_episode(bad)

    def test_force_share_cannot_be_replaced_by_marker_occupancy(self):
        bad = deepcopy(self.summary)
        bad["metrics"][FEET[0] + ".norm_share"]["value"] = 1.
        with self.assertRaises(AssertionError):
            self.verify_episode(bad)

    def test_group_denominator_unknown_and_membership_mutations(self):
        source = self.episode.source["manifest_sha256"]
        summaries = {self.summary["episode_id"]: self.summary}
        groups = aggregate_cohort([self.summary], {"manifest_sha256": source})["groups"]
        self.assertGreater(audit.audit_groups(groups, summaries, source), 0)
        for field in ("valid_samples", "unknown_samples", "total_episode_count"):
            bad = deepcopy(groups)
            bad["all"]["metrics"]["low_progress"][field] += 1
            with self.assertRaises(AssertionError):
                audit.audit_groups(bad, summaries, source)
        bad = deepcopy(groups)
        bad["all"]["member_ids"].append("unbound-source")
        with self.assertRaises(AssertionError):
            audit.audit_groups(bad, summaries, source)


if __name__ == "__main__":
    unittest.main()
