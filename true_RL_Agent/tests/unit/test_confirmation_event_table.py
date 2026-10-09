"""The N.5 display table preserves every original event and its list index."""

from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import unittest


HELPERS = Path(__file__).resolve().parents[2] / "PLAN/feedback_sys/P3NEW"
sys.path.insert(0, str(HELPERS))
try:
    spec = importlib.util.spec_from_file_location("confirmation_event_table_packer", HELPERS / "prepare_confirmation_materials.py")
    packer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(packer)
finally:
    sys.path.remove(str(HELPERS))


class ConfirmationEventTableTests(unittest.TestCase):
    def test_roundtrip_order_optional_measured_and_null(self):
        original = {"episode_id": "r1.e1", "steps": 10, "events": [
            {"body": "FL_foot", "code": "contact_instant", "start": 1, "confirmed": None, "end": 2},
            {"body": "root", "code": "tilt", "start": 2, "confirmed": 4, "end": 9, "measured": {"peak": -0.0}},
            {"body": "root", "code": "low_progress", "start": 8, "confirmed": None, "end": 10, "measured": None},
        ]}
        before = deepcopy(original)
        encoded = packer.review_episode_derivation(original, task="stair", material_profile="related_v1")
        self.assertEqual(packer.expand_review_episode_derivation(encoded), original)
        self.assertEqual(original, before)
        self.assertEqual(encoded["events"]["columns"], ["body", "code", "start", "confirmed", "end"])
        self.assertEqual(list(encoded["events"]["measured_by_original_event_index"]), ["1", "2"])
        self.assertIn("/events/i", encoded["events"]["original_locator"])

    def test_old_profile_and_push_are_unchanged(self):
        original = {"events": [{"another_task_field": 1}]}
        for task, profile in [("stair", "original"), ("stair", None), ("push", "related_v1")]:
            with self.subTest(task=task, profile=profile):
                self.assertEqual(packer.review_episode_derivation(original, task=task, material_profile=profile), original)

    def test_new_unrecognized_field_is_not_silently_dropped(self):
        event = {"body": "root", "code": "tilt", "start": 1, "confirmed": 1, "end": 1, "new_fact": 3}
        with self.assertRaises(ValueError):
            packer.review_episode_derivation({"events": [event]}, task="stair", material_profile="related_v1")

    def test_empty_list_and_bad_supplement_index(self):
        encoded = packer.review_episode_derivation({"events": []}, task="stair", material_profile="related_v1")
        self.assertEqual(packer.expand_review_episode_derivation(encoded), {"events": []})
        encoded["events"]["measured_by_original_event_index"] = {"0": 1}
        with self.assertRaises(ValueError):
            packer.expand_review_episode_derivation(encoded)


if __name__ == "__main__":
    unittest.main()
