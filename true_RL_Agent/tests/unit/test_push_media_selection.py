"""CPU checks for real-event Push media selection and missing-frame behavior."""

from copy import deepcopy
import unittest

from true_rl_agent.tasks.push_recovery.visual_observation import select_push_media_steps


def recovery(index, start, *, peak, last, latch=None, anchor=None, response=None):
    return {
        "identity": f"event_{index}", "anchor_verified": True,
        "anchor_control_step": start if anchor is None else anchor,
        "recovery_control_step": latch,
        "process": {
            "status": "available", "reference_event_index": index,
            "window_speed_peak": {"control_step": peak},
            "first_observed_response": {"control_step": start + 1 if response is None else response},
            "window_last": {"control_step": last},
        },
    }


def episode(*units, final=100, valid=None):
    return {"env_id": 2, "episode_index": 1, "final_control_step": final,
            "valid_steps": final if valid is None else valid, "recoveries": list(units)}


def velocity(*events):
    return {"env_id": 2, "episode_index": 1, "push_events": [
        {"event_index": index, "control_step": start, "injection_verified": verified}
        for index, start, verified in events]}


class PushMediaSelectionTests(unittest.TestCase):
    def select(self, transcript, raw, *, steps=range(1, 101), dt=0.1):
        return select_push_media_steps(transcript, runtime_episode_record=raw,
                                       available_steps=steps, step_dt_s=dt)

    def test_last_verified_actual_injection_wins_not_schedule_or_list_order(self):
        later = recovery(1, 30, peak=36, last=70, latch=45)
        scheduled = recovery(2, 60, peak=61, last=90, latch=80)
        scheduled["scheduled_anchor_control_step"] = 60
        transcript = episode(later, recovery(0, 5, peak=6, last=20, latch=15), scheduled)
        raw = velocity((0, 5, True), (1, 30, True), (2, 60, False))
        original = deepcopy((transcript, raw))
        result = self.select(transcript, raw)
        self.assertEqual(result["selected_steps"], [1, 25, 36, 45, 50, 100])
        self.assertEqual(result["process"]["reference_event_index"], 1)
        self.assertEqual(result["baseline_steps"], [1, 50, 100])
        self.assertEqual(result["context_offset_steps"], 5)
        self.assertEqual((transcript, raw), original)

    def test_force_requires_completed_verified_pulse_and_actual_end_anchor(self):
        transcript = episode(recovery(0, 30, anchor=35, peak=45, last=65, latch=52),
                             recovery(1, 60, anchor=65, peak=67, last=75), final=80)
        raw = {"force_events": [
            {"event_index": 0, "start_control_step": 30, "end_control_step": 35,
             "injection_verified": True, "completed": True},
            {"event_index": 1, "start_control_step": 60, "end_control_step": 65,
             "injection_verified": True, "completed": False},
        ]}
        result = self.select(transcript, raw, steps=range(1, 81))
        self.assertEqual(result["selected_steps"], [1, 25, 40, 45, 52, 80])
        self.assertEqual(result["process"]["source"], "runtime_episode_record.force_events")
        raw["force_events"][0]["end_control_step"] = 34
        self.assertEqual(self.select(transcript, raw, steps=range(1, 81))["process"]["status"],
                         "no_verified_observed_recovery")

    def test_peak_after_latch_uses_window_last_without_claiming_episode_recovery(self):
        transcript = episode(recovery(0, 20, peak=40, latch=30, last=60))
        transcript["terminal_label"] = "not_recovered"
        result = self.select(transcript, velocity((0, 20, True)))
        self.assertEqual(result["process"]["after"], 60)
        self.assertEqual(result["process"]["after_source"], "window_last")
        self.assertEqual(result["selected_steps"], [1, 15, 40, 50, 60, 100])

    def test_half_second_ceil_uses_earlier_available_frame_not_nearest(self):
        transcript = episode(recovery(0, 30, peak=31, latch=80, last=90))
        steps = [1, 8, 12, 14, 20, 31, 50, 80, 100]
        result = self.select(transcript, velocity((0, 30, True)), steps=steps, dt=0.03)
        self.assertEqual(result["context_offset_steps"], 17)
        self.assertEqual(result["process"]["before_target_step"], 13)
        self.assertEqual(result["process"]["before"], 12)
        self.assertEqual(result["selected_steps"], [1, 12, 31, 50, 80, 100])

    def test_missing_event_frames_are_not_snapped_to_nearby_frames(self):
        transcript = episode(recovery(0, 30, peak=34, response=31, latch=77, last=80))
        steps = [1, 24, 31, 35, 50, 78, 80, 100]
        result = self.select(transcript, velocity((0, 30, True)), steps=steps)
        self.assertEqual(result["process"]["middle"], 31)
        self.assertEqual(result["process"]["middle_source"], "first_observed_response")
        self.assertEqual(result["process"]["after"], 80)
        self.assertEqual(result["process"]["after_source"], "window_last")
        self.assertNotIn(35, result["selected_steps"])
        self.assertNotIn(78, result["selected_steps"])
        result = self.select(transcript, velocity((0, 30, True)), steps=[1, 24, 35, 50, 78, 100])
        self.assertIsNone(result["process"]["middle"])
        self.assertIsNone(result["process"]["after"])
        self.assertEqual(result["selected_steps"], [1, 24, 35, 50, 78, 100])
        self.assertEqual(set(result["fill_steps"]), {35, 78})

    def test_before_falls_back_to_earliest_strictly_prior_and_can_be_missing(self):
        transcript = episode(recovery(0, 5, peak=6, latch=9, last=12), final=20)
        result = self.select(transcript, velocity((0, 5, True)), steps=range(1, 21), dt=0.02)
        self.assertEqual(result["process"]["before"], 1)
        self.assertEqual(result["process"]["before_source"], "earliest_available_before_injection")
        transcript = episode(recovery(0, 1, peak=2, latch=3, last=4), final=4)
        result = self.select(transcript, velocity((0, 1, True)), steps=[1, 2, 3, 4])
        self.assertIsNone(result["process"]["before"])
        self.assertEqual(result["selected_steps"], [1, 2, 3, 4])

    def test_no_verified_unit_uses_deterministic_farthest_gap_fill(self):
        transcript = episode(recovery(0, 20, peak=21, latch=30, last=50))
        transcript["recoveries"][0]["anchor_verified"] = False
        result = self.select(transcript, velocity((0, 20, True)))
        self.assertEqual(result["selected_steps"], [1, 13, 25, 50, 75, 100])
        self.assertEqual(result["fill_steps"], [75, 25, 13])
        self.assertIsNone(result["process"]["source"])

    def test_duplicate_actual_event_index_is_not_treated_as_unique_occurrence(self):
        transcript = episode(recovery(0, 20, peak=21, latch=30, last=50))
        result = self.select(transcript, velocity((0, 20, True), (0, 20, True)))
        self.assertIsNone(result["process"]["source"])

    def test_short_shifted_episode_preserves_true_bounds_and_early_midpoint_tie(self):
        result = self.select(episode(final=10, valid=4), velocity(), steps=[7, 8, 9, 10])
        self.assertEqual(result["baseline_steps"], [7, 8, 10])
        self.assertEqual(result["selected_steps"], [7, 8, 9, 10])
        self.assertEqual(result["fill_steps"], [9])
        result = self.select(episode(final=7, valid=1), velocity(), steps=[7])
        self.assertEqual(result["selected_steps"], [7])

    def test_rejects_invalid_time_and_missing_or_noncanonical_available_bounds(self):
        for dt in (0, -0.1, float("nan"), float("inf"), True, "0.02"):
            with self.subTest(dt=dt), self.assertRaises(ValueError):
                self.select(episode(), velocity(), dt=dt)
        for steps in ([], [1, 100, 50], [1, 50, 50, 100], [0, 1, 100],
                      [True, 100], [1, 50.0, 100], [2, 100], [1, 99], [1, 101]):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                self.select(episode(), velocity(), steps=steps)
        with self.assertRaises(ValueError):
            self.select(episode(), {"push_events": [], "force_events": []})
        with self.assertRaises(ValueError):
            self.select(episode(), {"env_id": 9, "push_events": []})


if __name__ == "__main__":
    unittest.main()
