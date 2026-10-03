from copy import deepcopy
import inspect
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from tests.unit.trajectory_fixtures import sample_episode
from tests.unit.joint_limit_oracle import asset_bounds, check_record
from true_rl_agent.evaluation.trajectory_translation.joint_limits import measure_joint_limits
from true_rl_agent.tasks.stair_climbing.trajectory_joint_limits import ASSET_FILE



def fixture(values, names=("a", "b"), dt=.02):
    ep = sample_episode(len(values), dt=dt)
    ep.entities["joints"] = names
    ep.arrays["joint_pos"] = np.array(values, dtype=float)
    return ep


def bounds():
    return {n: {"lower": lo, "upper": hi, "unit": "rad", "coordinate_status": "qualified_nominal"}
            for n, lo, hi in (("a", 0, 20), ("b", -2, 2))}


class JointLimitTests(unittest.TestCase):
    def checked(self, ep, table=None):
        table = bounds() if table is None else table
        records = measure_joint_limits(ep, table, q_unit="rad", asset_id="synthetic")
        for i, record in enumerate(records):
            b = table[ep.entities["joints"][i]]
            check_record(record, ep.arrays["joint_pos"][:, i], ep.steps, b["lower"], b["upper"], ep.dt_s, ep.entities["joints"][i])
        return records

    def test_exact_bounds_band_midpoint_and_outside_not_clipped(self):
        ep = fixture([[0, -2], [1, 0], [10, 2], [19, 3], [20, -3], [-1, 1], [21, 0]])
        records = self.checked(ep)
        v = records[0]["value"]
        self.assertEqual(v["counts"]["inside_edge_band_rows"], 4)
        self.assertEqual(v["counts"]["outside_rows"], 2)
        self.assertEqual(v["position_fraction_range"], [-.05, 1.05])
        self.assertEqual(v["range_occupancy_fraction"], 1.1)

    def test_band_nextafter_and_outside_nextafter(self):
        ep = fixture([[np.nextafter(1., 0), 0], [1, 0], [np.nextafter(1., 2), 0], [np.nextafter(0., -1), 0]])
        v = self.checked(ep)[0]["value"]
        self.assertEqual(v["counts"]["inside_edge_band_rows"], 2)
        self.assertEqual(v["counts"]["outside_rows"], 1)

    def test_names_reordered_preserve_per_entity_values(self):
        ep = fixture([[.5, -1.99], [5, 1]])
        first = self.checked(ep)
        ep.entities["joints"] = ("b", "a")
        ep.arrays["joint_pos"] = ep.arrays["joint_pos"][:, ::-1]
        other = self.checked(ep)
        for a, b in zip(first, reversed(other)):
            self.assertEqual(a["value"]["q_range_rad"], b["value"]["q_range_rad"])
            self.assertEqual(a["value"]["minimum_signed_margin_rad"], b["value"]["minimum_signed_margin_rad"])

    def test_nan_inf_gap_single_sample_and_unknown(self):
        ep = fixture([[np.nan, -2], [np.inf, np.nan], [10, np.inf]], dt=.1)
        ep.steps = np.array([2, 4, 8])
        first = self.checked(ep)
        self.assertEqual(first[0]["value"]["missing_steps"], 4)
        self.assertEqual(first[0]["value"]["range_occupancy_fraction"], 0)
        self.assertEqual(first[1]["value"]["sample_cell_seconds"]["inside_edge_band"], .1)
        ep.arrays["joint_pos"][:] = np.nan
        self.checked(ep)

    def test_missing_mapping_and_missing_channel_are_unknown(self):
        ep = fixture([[0, 0]])
        records = measure_joint_limits(ep, {}, q_unit="rad", asset_id="unknown")
        self.assertTrue(all(r["value"]["unknown_reason"] == "unproven_coordinate_mapping" for r in records))
        self.assertTrue(all(r["value"]["outside_fraction"] is None for r in records))
        ep.arrays.pop("joint_pos")
        records = measure_joint_limits(ep, bounds(), q_unit="rad", asset_id="synthetic")
        self.assertEqual(records[0]["evidence_ref"]["missing_channels"], ["joint_pos"])

    def test_wrong_units_bad_limits_shape_and_duplicate_names_rejected(self):
        ep = fixture([[0, 0]])
        with self.assertRaises(ValueError):
            measure_joint_limits(ep, bounds(), q_unit="degree", asset_id="x")
        for lo, hi, unit in ((0, 0, "rad"), (2, 1, "rad"), (0, np.inf, "rad"), (0, 20, "degree")):
            b = bounds(); b["a"].update(lower=lo, upper=hi, unit=unit)
            with self.subTest(lo=lo, hi=hi, unit=unit), self.assertRaises(ValueError):
                measure_joint_limits(ep, b, q_unit="rad", asset_id="x")
        ep.arrays["joint_pos"] = np.zeros((1, 3))
        with self.assertRaises(ValueError):
            measure_joint_limits(ep, bounds(), q_unit="rad", asset_id="x")
        ep = fixture([[0, 0]])
        ep.entities["joints"] = ("a", "a")
        with self.assertRaises(ValueError): ep.__post_init__()

    def test_oracle_rejects_entity_unit_frame_even_if_id_resigned(self):
        ep = fixture([[0, 0]])
        record = self.checked(ep)[0]
        for field, value in (("unit", "degrees"), ("frame", "runtime_effective_limits"), ("entity_ids", ["b"])):
            bad = deepcopy(record); bad[field] = value; bad["record_id"] = "resigned"
            with self.subTest(field=field), self.assertRaises(AssertionError):
                check_record(bad, [0], [1], 0, 20, .02, "a")

    def test_asset_independent_usda_frames_and_degrees(self):
        parsed = asset_bounds(json.loads(ASSET_FILE.read_text(encoding="utf-8")))
        self.assertNotEqual(parsed["FL_thigh_joint"], parsed["RL_thigh_joint"])
        self.assertAlmostEqual(parsed["FL_hip_joint"][1], 1.04719988)
        for name, (lo, hi) in parsed.items():
            width = hi-lo
            ep = fixture([[lo], [lo+.05*width], [(lo+hi)/2], [hi-.05*width], [hi]], names=(name,))
            table = {name: {"lower": lo, "upper": hi, "unit": "rad", "coordinate_status": "qualified_nominal"}}
            self.assertEqual(self.checked(ep, table)[0]["value"]["counts"]["inside_edge_band_rows"], 4)





if __name__ == "__main__": unittest.main()
