"""CPU examples for post-force recovery and physical/technical failure counts."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from push_force_scoring import force_outcome, force_sequence_outcome


def pulse(start, end, *, completed=True, verified=True):
    return {"start_control_step": start, "end_control_step": end,
            "completed": completed, "injection_verified": verified}


class PushForceScoringTests(unittest.TestCase):
    def test_window_includes_last_step_but_excludes_last_forced_sample(self):
        for stable_steps, expected in (({8, 9, 10}, 10), ({9, 10, 11}, None),
                                       ({5, 6, 7}, None)):
            with self.subTest(stable_steps=stable_steps):
                steps = list(range(1, 12))
                result = force_outcome(
                    steps, [step in stable_steps for step in steps], [pulse(3, 5)], 1,
                    failure_label=None, final_stable=True, window_steps=5, dwell_steps=3,
                )
                self.assertEqual(result["event_recovery_control_steps"], [expected])
                self.assertEqual(result["success"], expected is not None)
                self.assertEqual(result["invalid_injection_count"], 0)

    def test_unstable_samples_or_missing_steps_break_continuous_dwell(self):
        for steps, stable in (([6, 7, 8, 9, 10], [True, True, False, True, True]),
                              ([6, 7, 9, 10], [True] * 4)):
            with self.subTest(steps=steps):
                result = force_outcome(
                    steps, stable, [pulse(3, 5)], 1, failure_label=None,
                    final_stable=True, window_steps=5, dwell_steps=3,
                )
                self.assertFalse(result["success"])
                self.assertEqual(result["recovered_event_count"], 0)

    def test_two_events_require_independent_post_end_dwells(self):
        steps = list(range(1, 12))
        for stable_steps, expected in ((set(range(3, 11)), [5, None]),
                                       (set(range(3, 12)), [5, 11]),
                                       ({9, 10, 11}, [None, 11])):
            with self.subTest(expected=expected):
                result = force_outcome(
                    steps, [step in stable_steps for step in steps],
                    [pulse(1, 2), pulse(7, 8)], 2, failure_label=None,
                    final_stable=True, window_steps=3, dwell_steps=3,
                )
                self.assertEqual(result["event_recovery_control_steps"], expected)
                self.assertEqual(result["success"], all(step is not None for step in expected))

    def test_early_failure_excuses_missing_or_partial_pulses_only(self):
        for events, unexplained_invalid in (([], 2), ([pulse(1, 2)], 1),
                                             ([pulse(1, 2, completed=False)], 2)):
            for label in ("fall_forward", "out_of_bounds", None):
                with self.subTest(events=events, label=label):
                    result = force_outcome(
                        [1, 2], [False, False], events, 2, failure_label=label,
                        final_stable=False, window_steps=3, dwell_steps=2,
                    )
                    self.assertFalse(result["success"])
                    self.assertEqual(result["failure_label"], label or "invalid_push_injection")
                    self.assertEqual(result["recovered_event_count"], 0)
                    self.assertEqual(result["invalid_injection_count"],
                                     0 if label else unexplained_invalid)

    def test_bad_injection_is_counted_even_when_fall_owns_failure_label(self):
        for completed in (True, False):
            for label in ("fall_side", None):
                with self.subTest(completed=completed, label=label):
                    result = force_outcome(
                        [1, 2, 3, 4], [True] * 4,
                        [pulse(1, 2, completed=completed, verified=False)], 1,
                        failure_label=label, final_stable=True, window_steps=2, dwell_steps=2,
                    )
                    self.assertFalse(result["success"])
                    self.assertEqual(result["failure_label"], label or "invalid_push_injection")
                    self.assertEqual(result["invalid_injection_count"], 1)
                    self.assertEqual(result["event_recovery_control_steps"], [None])

    def test_completed_recovery_still_requires_stable_final_state(self):
        result = force_outcome(
            [1, 2, 3, 4, 5], [False, False, True, True, False], [pulse(1, 2)], 1,
            failure_label=None, final_stable=False, window_steps=3, dwell_steps=2,
        )
        self.assertEqual(result["event_recovery_control_steps"], [4])
        self.assertEqual(result["recovered_event_count"], 1)
        self.assertEqual(result["invalid_injection_count"], 0)
        self.assertFalse(result["success"])
        self.assertEqual(result["failure_label"], "not_recovered")


class ForceSequenceTests(unittest.TestCase):
    def score(self, steps, stable, events=None, **overrides):
        kwargs = dict(failure_label=None, final_stable=True, window_steps=5, dwell_steps=3)
        kwargs.update(overrides)
        return force_sequence_outcome(steps, stable,
            [pulse(1, 2), pulse(4, 5)] if events is None else events, 2, **kwargs)

    def test_only_fresh_final_pulse_dwell_counts_and_window_is_inclusive(self):
        steps = list(range(1, 12))
        for good, expected in (({3, 4, 5}, None), ({8, 9, 10}, 10),
                               ({9, 10, 11}, None), ({6, 7, 9, 10}, None)):
            with self.subTest(good=good):
                result = self.score(steps, [s in good for s in steps])
                self.assertEqual(result['sequence_recovery_control_step'], expected)
                self.assertEqual(result['success'], expected is not None)
        self.assertFalse(self.score([6, 7, 9, 10], [True] * 4)['success'])

    def test_physical_early_failure_does_not_hide_bad_injection(self):
        for events, invalid in (([], 0), ([pulse(1, 2, completed=False)], 0),
                                ([pulse(1, 2, completed=False, verified=False)], 1)):
            result = self.score([1, 2], [False, False], events, failure_label='fall_side')
            self.assertFalse(result['success'])
            self.assertEqual(result['failure_label'], 'fall_side')
            self.assertEqual(result['invalid_injection_count'], invalid)
            self.assertFalse(result['sequence_recovered'])
        self.assertGreater(self.score([1, 2], [False, False], [])['invalid_injection_count'], 0)

    def test_complete_recovery_requires_final_stability(self):
        result = self.score(list(range(1, 11)), [True] * 10, final_stable=False)
        self.assertTrue(result['sequence_recovered'])
        self.assertEqual(result['sequence_recovery_control_step'], 8)
        self.assertFalse(result['success'])


if __name__ == "__main__":
    unittest.main()
