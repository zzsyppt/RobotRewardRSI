"""Regression gates for the P0 reference generator, including locked raw archives.

Archive tests skip only when the ignored public inputs are absent. The P0 release
gate requires these tests to execute with zero skips; no model calls are involved.
"""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
P0 = ROOT / "PLAN/feedback_sys/NEW_SYS/P0"
SPEC = importlib.util.spec_from_file_location("p0_reference", P0 / "recompute_reference_facts.py")
reference = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reference)


def facts_by_name(episode):
    prefix = episode["episode_id"] + "."
    return {f["fact_id"][len(prefix):]: f for f in episode["facts"]}


class ReferenceSemanticTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def generate(self, force, velocity, quaternion=None, *, legacy_e54=False):
        """Exercise episode_facts itself with a fully specified synthetic episode."""
        n = len(force)
        env = 54 if legacy_e54 else 0
        source_id = "legacy_dev" if legacy_e54 else "synthetic"
        count = env + 1
        quaternion = np.tile([1.0, 0, 0, 0], (n, 1)) if quaternion is None else quaternion
        arrays = {
            "base_pos_w": np.zeros((n, count, 3)),
            "base_quat_w": np.repeat(quaternion[:, None, :], count, axis=1),
            "support_pos_w": np.zeros((n, count, 4, 3)),
            "support_lin_vel_w": np.repeat(velocity[:, None, :, :], count, axis=1),
            "support_contact_forces_w": np.repeat(force[:, None, :, :], count, axis=1),
            "support_contact_forces_w_history": np.zeros((n, count, 3, 4, 3)),
            "forbidden_contact_forces_w": np.zeros((n, count, 1, 3)),
            "forbidden_contact_forces_w_history": np.zeros((n, count, 3, 1, 3)),
            "valid": np.ones((n, count), bool),
            "episode_length": np.repeat(np.arange(1, n + 1)[:, None], count, axis=1),
            "step_index": np.arange(1, n + 1),
            "source_chunk_index": np.zeros(n, dtype=np.int64),
            "source_chunk_row": np.arange(n),
        }
        arrays["support_pos_w"][:, :, :, 1] = 1.0
        for key in ("done_mask", "wrapper_done_mask", "termination_0", "termination_1"):
            arrays[key] = np.zeros((n, count), bool)
        outcome = {"success": False, "episode_steps": n, "primary_reason": "synthetic"}
        source = {
            "source_id": source_id, "lock": {"manifest_sha256": "synthetic-no-physical-claim"},
            "runs": {1: {"arrays": arrays, "static": {}}},
            "seeds": {1: {"outcomes": [outcome] * count}},
            "manifest": {"metadata": {"runtime_projection": {"resolved_roles": {"forbidden_contact": ["base"]}}}},
        }
        selected = {"episode_id": f"{source_id}.r1.e{env}", "run_index": 1, "env_index": env,
                    "archived_outcome": outcome, "selection_reason": "synthetic semantic regression",
                    "required_fact_categories": []}
        return facts_by_name(reference.episode_facts(ROOT, source, selected, Path(self.temp.name)))

    def test_planar_speed_and_strict_force_and_speed_boundaries(self):
        force = np.zeros((4, 4, 3))
        force[:, 1:, 2] = 2.0
        force[:, 0, 2] = [0, 1, np.nextafter(1.0, np.inf), 0]
        velocity = np.tile([0.03, 0, 0.08], (4, 4, 1))
        velocity[3, 0] = [0.05, 0, 0.08]
        facts = self.generate(force, velocity)
        markers = facts["FL_foot.markers"]["value"]
        self.assertEqual(markers["on_count"], 1)
        self.assertEqual(markers["observed_onset_steps"], [3])
        low = facts["FL_foot.low_speed_off"]["value"]["longest"]
        self.assertEqual((low["longest_run"]["first_step"], low["longest_run"]["last_step"]), (1, 2))
        self.assertGreater(low["max_speed_xyz_mps"], 0.05)
        self.assertAlmostEqual(low["max_speed_xy_mps"], 0.03)
        self.assertTrue(low["longest_run"]["left_censored"])
        self.assertFalse(low["longest_run"]["right_censored"])
        # An equality-only episode must have no low-speed run. Checking only
        # the earlier longest run would miss an erroneous <= speed threshold.
        force[:, 0] = 0
        velocity[:, 0] = [0.05, 0, 0.08]
        equal_speed = self.generate(force, velocity)["FL_foot.low_speed_off"]["value"]
        self.assertIsNone(equal_speed["longest"])

    def test_complete_left_and_right_censored_runs_keep_one_sample_event(self):
        force = np.zeros((8, 4, 3))
        force[:, :, 2] = 2
        force[[0, 1, 3, 5, 6, 7], 0] = 0
        facts = self.generate(force, np.tile([0.03, 0, 0.08], (8, 4, 1)))
        left, single, right = facts["FL_foot.markers"]["value"]["off_runs"]
        self.assertEqual((left["first_step"], left["last_step"], left["left_censored"], left["onset_boundary_s"]),
                         (1, 2, True, None))
        self.assertEqual(left["end_boundary_s"], [0.04, 0.06])
        self.assertEqual((single["first_step"], single["sample_count"], single["observed_endpoint_span_s"]), (4, 1, 0))
        self.assertEqual(single["marker_cell_duration_s"], 0.02)
        self.assertFalse(single["left_censored"] or single["right_censored"])
        self.assertEqual(single["onset_boundary_s"], [0.06, 0.08])
        self.assertEqual(single["end_boundary_s"], [0.08, 0.1])
        self.assertEqual((right["first_step"], right["last_step"], right["sample_count"]), (6, 8, 3))
        self.assertTrue(right["right_censored"])
        self.assertIsNone(right["end_boundary_s"])
        self.assertAlmostEqual(right["marker_cell_duration_s"], 0.06)
        self.assertAlmostEqual(right["observed_endpoint_span_s"], 0.04)
        self.assertEqual(facts["FL_foot.low_speed_off"]["value"]["longest"]["longest_run"], right)

    def test_legacy_e54_specific_counterexample_uses_xy_too(self):
        force = np.zeros((300, 4, 3))
        force[:, :, 2] = 2
        force[2:, :2] = 0
        velocity = np.tile([0.03, 0, 0.08], (300, 4, 1))
        q = np.tile([0.0, 1, 0, 0], (300, 1))
        q[0] = [1, 0, 0, 0]
        facts = self.generate(force, velocity, q, legacy_e54=True)
        special = facts["required_inverted_counterexample"]["value"]
        self.assertEqual(special["first_negative_step"], 2)
        for front in special["front_low_speed_off_runs"]:
            run = front["run"]
            self.assertEqual((run["first_step"], run["last_step"], run["sample_count"]), (3, 300, 298))
            self.assertTrue(run["right_censored"])
            self.assertEqual(run, facts[front["entity"] + ".low_speed_off"]["value"]["longest"]["longest_run"])
        velocity[-1, :2] = [0.05, 0, 0.08]
        boundary = self.generate(force, velocity, q, legacy_e54=True)
        for front in boundary["required_inverted_counterexample"]["value"]["front_low_speed_off_runs"]:
            run = front["run"]
            self.assertEqual((run["first_step"], run["last_step"], run["sample_count"]), (3, 299, 297))
            self.assertFalse(run["right_censored"])
            self.assertEqual(run, boundary[front["entity"] + ".low_speed_off"]["value"]["longest"]["longest_run"])

    def test_all_16_combinations_round_trip_with_low_role_index_bit(self):
        # Explicit truth table; integer i has FL as its least significant bit.
        table = np.array([
            [0,0,0,0], [1,0,0,0], [0,1,0,0], [1,1,0,0],
            [0,0,1,0], [1,0,1,0], [0,1,1,0], [1,1,1,0],
            [0,0,0,1], [1,0,0,1], [0,1,0,1], [1,1,0,1],
            [0,0,1,1], [1,0,1,1], [0,1,1,1], [1,1,1,1]], dtype=bool)
        for code, markers in enumerate(table):
            with self.subTest(code=code):
                force = np.zeros((2, 4, 3))
                force[0, :, 2] = markers * 2
                force[1, :, 2] = 2  # ensures nonzero force denominator even for code=0
                combination = self.generate(force, np.zeros_like(force))["contact_combinations"]["value"]
                self.assertEqual(combination["bit_order"], reference.FEET)
                self.assertEqual(combination["bit_weights"], [1, 2, 4, 8])
                self.assertEqual(combination["encoding_version"], "role_index_lsb.v1")
                counts = np.array(combination["counts_0_to_15"])
                counts[15] -= 1
                observed_code = int(np.flatnonzero(counts)[0])
                self.assertEqual(observed_code, code)
                self.assertEqual(int(counts.sum()), 1)
                self.assertEqual([bool((observed_code >> i) & 1) for i in range(4)], markers.tolist())

    def test_normalized_quaternion_controls_sign_and_body_frame_raw_is_diagnostic(self):
        angle = np.arccos(1e-5)
        unit = np.array([np.cos(angle / 2), np.sin(angle / 2), 0, 0])
        q = np.tile(unit * 1.00004, (2, 1))
        force = np.zeros((2, 4, 3))
        force[:, :, 2] = 2
        facts = self.generate(force, np.zeros_like(force), q)
        up = facts["body_up"]["value"]
        self.assertIsNone(up["first_negative_step"])
        self.assertEqual(up["negative_sample_count"], 0)
        self.assertAlmostEqual(up["first_up_z"], 1e-5, places=12)
        self.assertEqual(up["raw_body_up_diagnostic"]["first_negative_step"], 1)
        anchor = facts["body_frame_anchors"]["value"]["first_body_xyz_m"][0]
        np.testing.assert_allclose(anchor, [0, np.cos(angle), -np.sin(angle)], atol=1e-12)
        with self.assertRaisesRegex(ValueError, "Quaternion norm"):
            self.generate(force, np.zeros_like(force), np.tile(unit * 1.0002, (2, 1)))


@unittest.skipUnless((ROOT / "outputs/trajectory_p0/raw/numeric_evidence/manifest.json").is_file(),
                     "P0 locked public archive absent; release gate requires zero skips")
class LockedRawReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        lock = json.loads((P0 / "source_lock.json").read_text(encoding="utf-8"))
        cls.source = reference.load_source(ROOT, next(s for s in lock["sources"] if s["source_id"] == "r4"))
        cls.selection = json.loads((P0 / "e1_selection.json").read_text(encoding="utf-8"))

    def generate(self, episode_id):
        selected = next(e for e in self.selection["episodes"] if e["episode_id"] == episode_id)
        with tempfile.TemporaryDirectory() as directory:
            return facts_by_name(reference.episode_facts(ROOT, self.source, selected, Path(directory)))

    def test_real_r4_e60_fl_planar_run_and_step626_discriminate_xyz(self):
        facts = self.generate("r4.r1.e60")
        run = facts["FL_foot.low_speed_off"]["value"]["longest"]["longest_run"]
        self.assertEqual((run["first_step"], run["last_step"], run["sample_count"]), (214, 637, 424))
        self.assertEqual((run["marker_cell_duration_s"], run["observed_endpoint_span_s"]), (8.48, 8.46))
        self.assertFalse(run["left_censored"])
        self.assertTrue(run["right_censored"])
        self.assertEqual(run["onset_boundary_s"], [4.26, 4.28])
        self.assertIsNone(run["end_boundary_s"])
        a = self.source["runs"][1]["arrays"]
        self.assertEqual((int(a["source_chunk_index"][625]), int(a["source_chunk_row"][625])), (9, 49))
        velocity = a["support_lin_vel_w"][:637, 60, 0].astype(np.float64)
        force = a["support_contact_forces_w"][:637, 60, 0].astype(np.float64)
        np.testing.assert_array_equal(force[625], [0, 0, 0])
        np.testing.assert_array_equal(velocity[625], [0.0415688157081604, 0.016667895019054413, -0.0320286750793457])
        self.assertAlmostEqual(float(np.linalg.norm(velocity[625, :2])), 0.04478599294137869)
        self.assertGreater(float(np.linalg.norm(velocity[625])), 0.05)
        xyz = reference.longest(reference.intervals((np.linalg.norm(force, axis=1) <= 1) &
                                                    (np.linalg.norm(velocity, axis=1) < 0.05)))
        self.assertEqual((xyz["first_step"], xyz["last_step"], xyz["sample_count"]), (475, 625, 151))
        self.assertFalse(xyz["right_censored"])

    def test_real_r4_e35_step14_code11_and_episode_count135(self):
        a = self.source["runs"][1]["arrays"]
        force = a["support_contact_forces_w"][13, 35].astype(np.float64)
        markers = (np.linalg.norm(force, axis=1) > 1).tolist()
        self.assertEqual(markers, [True, True, False, True])
        self.assertEqual(sum(int(on) << i for i, on in enumerate(markers)), 11)
        combinations = self.generate("r4.r1.e35")["contact_combinations"]["value"]
        self.assertEqual(combinations["counts_0_to_15"][11], 135)
        self.assertEqual(combinations["FL_FR_RR_only_count"], 135)
        self.assertNotEqual(combinations["counts_0_to_15"][13], 135)


if __name__ == "__main__":
    unittest.main()
