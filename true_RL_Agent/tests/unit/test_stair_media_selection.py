"""Task media sampling preserves whole-episode coverage and actual frame clocks."""

from copy import deepcopy
import unittest

from true_rl_agent.tasks.stair_climbing.visual_observation import select_stair_media_steps


def episode():
    return {"steps": 200, "dt_s": .02, "coverage": {"first_step": 1}, "events": [],
            "root_motion": {}, "anchor": {"status": "not_reached", "step": None}}


class StairMediaSelectionTests(unittest.TestCase):
    def test_body_priority_half_second_context_and_whole_episode_coverage(self):
        data = episode()
        data["events"] = [
            {"code": "low_progress", "body": "root", "start": 10, "confirmed": 10, "end": 180},
            {"code": "tilt", "body": "root", "start": 70, "confirmed": 75, "end": 100}]
        original = deepcopy(data)
        result = select_stair_media_steps(data, available_steps=list(range(1, 201)))
        self.assertEqual(result["baseline_steps"], [1, 100, 200])
        self.assertEqual(result["selected_steps"], [1, 45, 75, 100, 125, 200])
        self.assertEqual(result["process"]["code"], "tilt")
        self.assertEqual(data, original)
        self.assertEqual(result, select_stair_media_steps(data, available_steps=list(range(1, 201))))

    def test_terminal_interval_and_absent_middle_never_invent_frames(self):
        data = episode()
        data["events"] = [{"code": "tilt", "body": "root", "start": 2, "confirmed": 5, "end": 200}]
        available = [1, 2, 20, 80, 100, 140, 200]
        result = select_stair_media_steps(data, available_steps=available)
        self.assertEqual(result["process"]["before"], 1)
        self.assertIsNone(result["process"]["middle"])
        self.assertIsNone(result["process"]["after"])
        self.assertEqual(len(result["selected_steps"]), 6)
        self.assertTrue(set(result["selected_steps"]) <= set(available))
        self.assertTrue({1, 100, 200} <= set(result["selected_steps"]))

    def test_extremum_retreat_anchor_missing_dt_and_short_episode(self):
        data = episode()
        cases = [
            ({"extrema": {"up_z": [.8, 80, 1., 10]}}, {"status": "not_reached"}, "root_motion/extrema/up_z", [55, 80, 105]),
            ({"max_x_retreat": [80, 120, .1]}, {"status": "not_reached"}, "root_motion/max_x_retreat", [55, 120, 145]),
            ({}, {"status": "crossed", "step": 190}, "anchor/step", [165, 190, 200]),
        ]
        for motion, anchor, source, points in cases:
            with self.subTest(source=source):
                result = select_stair_media_steps({**data, "root_motion": motion, "anchor": anchor}, available_steps=list(range(1, 201)))
                self.assertEqual(result["process"]["source"], source)
                self.assertEqual([result["process"][k] for k in ("before", "middle", "after")], points)
        # A four-term event is not fabricated from the up_z extrema fallback.
        self.assertNotIn("code", result["process"])
        for value in (None, 0, float("nan")):
            with self.assertRaisesRegex(ValueError, "actual positive step dt"):
                select_stair_media_steps({**data, "dt_s": value}, available_steps=[1, 200])
        short = select_stair_media_steps({**data, "steps": 3}, available_steps=[1, 2, 3])
        self.assertEqual(short["selected_steps"], [1, 2, 3])
        self.assertIsNone(short["process"]["source"])


if __name__ == "__main__":
    unittest.main()
