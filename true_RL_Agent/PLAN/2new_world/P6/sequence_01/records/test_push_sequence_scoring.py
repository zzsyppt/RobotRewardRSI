"""CPU checks for the declared two-push diagnostic recovery semantics."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from push_sequence_scoring import sequence_outcome


class PushSequenceScoringTests(unittest.TestCase):
    def test_prior_dwell_and_discontinuous_stability_cannot_complete_sequence(self):
        steps = list(range(1, 13))
        # A long dwell spans the first push, but the last push clears it.
        # After the last push there are three stable samples, never consecutive.
        stable = [3 <= step <= 8 or step in (10, 12) for step in steps]
        result = sequence_outcome(
            steps, stable,
            [{"control_step": 3, "injection_verified": True},
             {"control_step": 7, "injection_verified": True}],
            failure_label=None, final_stable=True, window_steps=5, dwell_steps=3,
        )
        self.assertFalse(result["success"])
        self.assertFalse(result["sequence_recovered"])
        self.assertIsNone(result["sequence_recovery_control_step"])
        self.assertEqual(result["invalid_injection_count"], 0)

    def test_continuous_dwell_may_finish_on_inclusive_window_boundary(self):
        steps = list(range(1, 9))
        stable = [step >= 5 for step in steps]
        events = [{"control_step": 2, "injection_verified": True},
                  {"control_step": 5, "injection_verified": True}]
        # Step 5 is the injection sample: only steps 6, 7 and 8 count.
        result = sequence_outcome(
            steps, stable, events, failure_label=None, final_stable=True,
            window_steps=3, dwell_steps=3,
        )
        self.assertTrue(result["success"])
        self.assertTrue(result["sequence_recovered"])
        self.assertEqual(result["sequence_recovery_control_step"], 8)
        self.assertEqual(result["invalid_injection_count"], 0)

        outside = sequence_outcome(
            steps, stable, events, failure_label=None, final_stable=True,
            window_steps=2, dwell_steps=3,
        )
        self.assertFalse(outside["success"])
        self.assertFalse(outside["sequence_recovered"])
        self.assertIsNone(outside["sequence_recovery_control_step"])

    def test_early_failure_preserves_label_but_unexplained_missing_push_is_invalid(self):
        first_push = [{"control_step": 2, "injection_verified": True}]
        for label in ("fall_forward", "out_of_bounds"):
            with self.subTest(failure_label=label):
                result = sequence_outcome(
                    [1, 2, 3, 4], [True, True, False, False], first_push,
                    failure_label=label, final_stable=False,
                    window_steps=5, dwell_steps=2,
                )
                self.assertFalse(result["success"])
                self.assertEqual(result["failure_label"], label)
                self.assertFalse(result["sequence_recovered"])
                self.assertIsNone(result["sequence_recovery_control_step"])
                self.assertEqual(result["invalid_injection_count"], 0)

        missing = sequence_outcome(
            [1, 2, 3, 4], [True, True, True, True], first_push,
            failure_label=None, final_stable=True, window_steps=5, dwell_steps=2,
        )
        self.assertFalse(missing["success"])
        self.assertFalse(missing["sequence_recovered"])
        self.assertEqual(missing["invalid_injection_count"], 1)

        unverified = sequence_outcome(
            [1, 2, 3, 4, 5], [True, True, True, True, True],
            [{"control_step": 2, "injection_verified": True},
             {"control_step": 3, "injection_verified": False}],
            failure_label=None, final_stable=True, window_steps=2, dwell_steps=2,
        )
        self.assertFalse(unverified["success"])
        self.assertEqual(unverified["invalid_injection_count"], 1)

    def test_recovered_sequence_still_requires_stable_final_state(self):
        result = sequence_outcome(
            [1, 2, 3, 4, 5, 6, 7],
            [False, False, False, False, True, True, False],
            [{"control_step": 2, "injection_verified": True},
             {"control_step": 4, "injection_verified": True}],
            failure_label=None, final_stable=False, window_steps=3, dwell_steps=2,
        )
        self.assertTrue(result["sequence_recovered"])
        self.assertEqual(result["sequence_recovery_control_step"], 6)
        self.assertFalse(result["success"])
        self.assertIsNotNone(result["failure_label"])
        self.assertEqual(result["invalid_injection_count"], 0)


if __name__ == "__main__":
    unittest.main()
