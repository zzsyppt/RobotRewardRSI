"""Known periods and an independent scalar oracle for local qualification."""
from copy import deepcopy
import unittest

import numpy as np

from tests.unit.trajectory_fixtures import sample_episode
from tests.unit.cycle_oracle import check_periods
from true_rl_agent.evaluation.trajectory_translation.cycles import measure_contact_cycles
from true_rl_agent.tasks.stair_climbing.trajectory_cycles import episode_periods, summarize_periods


def fixture(onsets, *, n=None, width=6, dt=.02):
    n = n or max(max(group) for group in onsets)+width-1
    ep = sample_episode(n, support=tuple("foot"+str(i) for i in range(len(onsets))), dt=dt)
    ep.arrays["support_contact_forces_w"][:] = 0
    for j, group in enumerate(onsets):
        for k in group:
            ep.arrays["support_contact_forces_w"][k-1:min(k-1+width, n), j, 2] = 2
    return ep


class ContactCycleTests(unittest.TestCase):
    def checked(self, ep):
        periods = measure_contact_cycles(ep)
        check_periods(periods, ep.entities["support"], ep.steps,
                      ep.arrays["support_contact_forces_w"], ep.dt_s, ep.task["contact_threshold_n"])
        return periods

    def test_known_period_overlap_and_episode_equal_cohort(self):
        first = self.checked(fixture([[4, 24, 44, 64, 84, 104]]))
        self.assertEqual(first["foot0"], [.4]*5)  # Two windows, five unique cycles.
        second = self.checked(fixture([[4, 44, 84, 124, 164]]))
        summary = summarize_periods([episode_periods(first), episode_periods(second)])
        stats = summary["groups"]["contact_cycles:foot0"]["episode_median_qualified_cycle_period_s"]
        self.assertAlmostEqual(stats["median"], .6)  # Each episode has equal weight.

    def test_cv_and_median_deviation_independent_boundaries(self):
        for lengths, qualified in [([17,17,23,23], True), ([17,17,23,24], False),
                                   ([20,20,20,24], True), ([20,20,20,25], False)]:
            ep = fixture([[4, *list(4+np.cumsum(lengths))]])
            self.assertEqual(self.checked(ep)["foot0"], [v*.02 for v in lengths] if qualified else [])

    def test_single_or_three_cycles_and_endpoint_are_not_invented(self):
        for onsets in ([4,24], [4,24,44,64], [1,24,44,64,84]):
            self.assertEqual(self.checked(fixture([onsets]))["foot0"], [])
        for tail in (1,2,3):
            periods = self.checked(fixture([[4,24,44,64,84]], n=83+tail))["foot0"]
            self.assertEqual(periods, [.4]*4 if tail == 3 else [])

    def test_raw_short_on_off_changes_disqualify_local_windows(self):
        for width in (1, 2):
            for state in (0, 1):
                ep = fixture([[4,24,44,64,84,104]], width=10)
                start = 28 if state == 0 else 38
                ep.arrays["support_contact_forces_w"][start-1:start-1+width, 0, 2] = 2*state
                self.assertEqual(self.checked(ep)["foot0"], [])

    def test_unknown_and_step_gap_never_bridge_cycles(self):
        for invalid in (np.nan, np.inf):
            ep = fixture([[4,24,44,64,84]])
            ep.arrays["support_contact_forces_w"][49,0,0] = invalid
            self.assertEqual(self.checked(ep)["foot0"], [])
        ep = fixture([[4,24,44,64,84]])
        ep.steps[50:] += 1
        self.assertEqual(self.checked(ep)["foot0"], [])

    def test_dt_scaling_and_history_irrelevance(self):
        for dt in (.02, .007):
            ep = fixture([list(range(4,165,20)), list(range(9,170,20))], dt=dt)
            periods = self.checked(ep)
            self.assertAlmostEqual(periods["foot0"][0],20*dt)
            ep.arrays["support_contact_forces_w_history"][:] = 99999
            self.assertEqual(periods, self.checked(ep))
        ep.arrays.pop("support_contact_forces_w_history")
        self.assertEqual(periods, self.checked(ep))

    def test_entity_order_source_identity_and_threshold(self):
        ep = fixture([list(range(4,165,20)), list(range(9,170,20))])
        original = self.checked(ep)
        ep.entities["support"] = tuple(reversed(ep.entities["support"]))
        ep.arrays["support_contact_forces_w"] = ep.arrays["support_contact_forces_w"][:,::-1]
        self.assertEqual(self.checked(ep), original)
        ep.source["manifest_sha256"] = "f"*64
        self.assertEqual(self.checked(ep), original)
        ep.task["contact_threshold_n"] = 2
        self.assertEqual(self.checked(ep), {name: [] for name in ep.entities["support"]})

    def test_oracle_catches_incorrect_period(self):
        ep = fixture([[4,24,44,64,84]])
        bad = deepcopy(self.checked(ep))
        bad["foot0"][0] = .3
        with self.assertRaises(AssertionError):
            check_periods(bad, ep.entities["support"], ep.steps, ep.arrays["support_contact_forces_w"], ep.dt_s, 1.)

    def test_threshold_equality_missing_current_and_equal_occupancy_are_not_rhythm(self):
        ep = fixture([list(range(4,105,20))],n=120,width=10)
        self.assertTrue(self.checked(ep)["foot0"])
        forces = ep.arrays["support_contact_forces_w"]
        count = np.count_nonzero(forces[:,0,2])
        forces[:] = 0
        forces[:count,0,2] = 2
        self.assertEqual(self.checked(ep)["foot0"], [])
        forces[:] = 0
        forces[:,:,0] = 1
        self.assertEqual(self.checked(ep)["foot0"], [])
        ep.arrays.pop("support_contact_forces_w")
        self.assertEqual(measure_contact_cycles(ep)["foot0"], [])


if __name__ == "__main__":
    unittest.main()
