import json
import unittest

import numpy as np

from tests.unit.trajectory_fixtures import sample_episode
from tests.unit.joint_limit_oracle import asset_bounds, expected, close
from true_rl_agent.evaluation.trajectory_translation.joint_limits import measure_joint_limits, measure_runtime_joint_limits
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
    def test_runtime_columns_are_already_resolved_and_environment_specific(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from true_rl_agent.tasks.stair_climbing.trajectory_capabilities import resolve_capabilities, CAPTURE_PROFILE, CONTRACT_VERSION
        names = ["b", "a"]
        q = {"source":"robot.data.joint_pos[:, resolved.joint_ids]", "unit":"rad",
             "coordinate_basis":"USD_joint_local_frames_no_added_zero_offset", "joint_names":names,
             "articulation_joint_names":["unused","a","b"], "resolved_joint_ids":[2,1], "runtime_projection_hash":"fixture"}
        buffers = {"hard":np.array([[[-1,1],[10,20]],[[-2,2],[0,1]]],dtype=float),
                   "soft":np.array([[[-.5,.5],[12,18]],[[-1,1],[.2,.8]]],dtype=float)}
        mapping = {"hard":"joint_pos_limits", "soft":"soft_joint_pos_limits"}
        runtime = {"joint_names":names, "resolved_joint_ids":[2,1], "axes":["environment","joint","lower_upper"],
            "unit":"rad", "recorded_at":"after_environment_creation_before_first_seed_reset",
            "end_values_equal":False, "end_checked_at":"after_last_scored_episode",
            "coverage":"initial_and_final_snapshots_only; intermediate_changes_not_monitored",
            "values":{mapping[k]:v.tolist() for k,v in buffers.items()},
            "sources":{v:f"robot.data.{v}[:, resolved.joint_ids, :]" for v in mapping.values()}}
        archive = SimpleNamespace(cells=[None,None], projection=SimpleNamespace(joint_names=names,projection_hash="fixture"),
            manifest={"metadata":{"profile":CAPTURE_PROFILE, "runtime_joint_position_limits":runtime,
                "measurement_contract":{"schema_version":CONTRACT_VERSION,"sample_phase":"pre_reset_control_step_end","joint_position":q}}})
        with patch("true_rl_agent.tasks.stair_climbing.trajectory_capabilities.nominal_binding",side_effect=AssertionError("asset dependency")):
            caps,nominal = resolve_capabilities(archive,{"manifest_sha256":"new_fixture"})
        self.assertIsNone(nominal)
        self.assertIs(caps["runtime_joint_limits"]["binding"]["end_values_equal"],False)
        ep = fixture([[1.5,.5],[2.0856,0],[np.nan,1.01]],names=tuple(names)); ep.env_index=1
        got=measure_runtime_joint_limits(ep,buffers)
        self.assertEqual((got["hard"]["b"]["valid"],got["hard"]["b"]["unknown"],got["hard"]["b"]["upper_outside"]),(2,1,1))
        self.assertEqual((got["soft"]["b"]["upper_outside"],got["soft"]["a"]["lower_outside"],got["soft"]["a"]["upper_outside"]),(2,1,1))
        self.assertAlmostEqual(got["hard"]["b"]["minimum_margin"],-.0856)
        runtime["joint_names"] = ["a","b"]
        with self.assertRaisesRegex(ValueError,"order"):
            resolve_capabilities(archive,{"manifest_sha256":"new_fixture"})

    def test_runtime_boundary_encoding_does_not_swallow_real_excess(self):
        ep = fixture([[.1],[.1856],[np.nan]],names=("a",)); ep.env_index=0
        ep.arrays["joint_pos"] = ep.arrays["joint_pos"].astype(np.float32)
        got=measure_runtime_joint_limits(ep,{"hard":np.array([[[0,.1]]])})["hard"]["a"]
        self.assertEqual((got["valid"],got["unknown"],got["upper_outside"]),(2,1,1))
        self.assertAlmostEqual(got["minimum_margin"],-.0856,places=7)
        self.assertLess(got["boundary_tolerance_rad"][1],1e-8)

    def test_runtime_success_and_failure_use_their_own_recorded_point_denominators(self):
        from true_rl_agent.tasks.stair_climbing.trajectory_production import _aggregate_runtime_limits
        rows=[]
        for reason,values in (("success",[1.2]),("success",[0,0,0]),("timeout",[1.2,1.2]),("timeout",[0,0])):
            ep=fixture([[v] for v in values],names=("a",)); ep.env_index=0
            rows.append((reason,measure_runtime_joint_limits(ep,{"hard":np.array([[[-1,1]]])})))
        groups=_aggregate_runtime_limits(rows)["groups"]
        for reason,percent,valid in (("success",25.,4),("timeout",50.,4),("all",37.5,8)):
            fact=groups[reason]["facts"]["hard"]["a"]
            self.assertEqual((fact["outside_percent"]["upper"],fact["valid"]),(percent,valid))

    def checked(self, ep, table=None):
        table = bounds() if table is None else table
        result = measure_joint_limits(ep, table, q_unit="rad")
        self.assertEqual(set(result), set(ep.entities["joints"]))
        for i, name in enumerate(ep.entities["joints"]):
            b = table[name]
            close(result[name], expected(ep.arrays["joint_pos"][:, i], b["lower"], b["upper"]))
        return result

    def test_exact_bounds_band_midpoint_and_outside_not_clipped(self):
        ep = fixture([[0, -2], [1, 0], [10, 2], [19, 3], [20, -3], [-1, 1], [21, 0]])
        records = self.checked(ep)
        v = records["a"]
        self.assertEqual(v["counts"]["inside_edge_band_rows"], 4)
        self.assertEqual(v["counts"]["outside_rows"], 2)
        self.assertEqual(v["minimum_signed_margin_rad"], -1.)
        self.assertEqual((v["counts"]["lower_outside_rows"], v["counts"]["upper_outside_rows"]), (1, 1))

    def test_band_nextafter_and_outside_nextafter(self):
        ep = fixture([[np.nextafter(1., 0), 0], [1, 0], [np.nextafter(1., 2), 0], [np.nextafter(0., -1), 0]])
        v = self.checked(ep)["a"]
        self.assertEqual(v["counts"]["inside_edge_band_rows"], 2)
        self.assertEqual(v["counts"]["outside_rows"], 1)

    def test_names_reordered_preserve_per_entity_values(self):
        ep = fixture([[.5, -1.99], [5, 1]])
        first = self.checked(ep)
        ep.entities["joints"] = ("b", "a")
        ep.arrays["joint_pos"] = ep.arrays["joint_pos"][:, ::-1]
        other = self.checked(ep)
        self.assertEqual(first, other)

    def test_nan_inf_gap_single_sample_and_unknown(self):
        ep = fixture([[np.nan, -2], [np.inf, np.nan], [10, np.inf]], dt=.1)
        ep.steps = np.array([2, 4, 8])
        first = self.checked(ep)
        self.assertEqual(first["a"]["counts"]["unknown_rows"], 2)
        self.assertEqual(first["a"]["minimum_signed_margin_rad"], 10)
        self.assertEqual(first["b"]["counts"]["inside_edge_band_rows"], 1)
        ep.arrays["joint_pos"][:] = np.nan
        self.checked(ep)

    def test_missing_mapping_and_missing_channel_are_unknown(self):
        ep = fixture([[0, 0]])
        result = measure_joint_limits(ep, {}, q_unit="rad")
        self.assertTrue(all(v["counts"]["unknown_rows"] == 1 and v["minimum_signed_margin_rad"] is None for v in result.values()))
        ep.arrays.pop("joint_pos")
        self.assertEqual(measure_joint_limits(ep, bounds(), q_unit="rad"), result)

    def test_wrong_units_bad_limits_shape_and_duplicate_names_rejected(self):
        ep = fixture([[0, 0]])
        with self.assertRaises(ValueError):
            measure_joint_limits(ep, bounds(), q_unit="degree")
        for lo, hi, unit in ((0, 0, "rad"), (2, 1, "rad"), (0, np.inf, "rad"), (0, 20, "degree")):
            b = bounds(); b["a"].update(lower=lo, upper=hi, unit=unit)
            with self.subTest(lo=lo, hi=hi, unit=unit), self.assertRaises(ValueError):
                measure_joint_limits(ep, b, q_unit="rad")
        ep.arrays["joint_pos"] = np.zeros((1, 3))
        with self.assertRaises(ValueError):
            measure_joint_limits(ep, bounds(), q_unit="rad")
        ep = fixture([[0, 0]])
        ep.entities["joints"] = ("a", "a")
        with self.assertRaises(ValueError): ep.__post_init__()

    def test_asset_independent_usda_frames_and_degrees(self):
        parsed = asset_bounds(json.loads(ASSET_FILE.read_text(encoding="utf-8")))
        self.assertNotEqual(parsed["FL_thigh_joint"], parsed["RL_thigh_joint"])
        self.assertAlmostEqual(parsed["FL_hip_joint"][1], 1.04719988)
        for name, (lo, hi) in parsed.items():
            width = hi-lo
            ep = fixture([[lo], [lo+.05*width], [(lo+hi)/2], [hi-.05*width], [hi]], names=(name,))
            table = {name: {"lower": lo, "upper": hi, "unit": "rad", "coordinate_status": "qualified_nominal"}}
            self.assertEqual(self.checked(ep, table)[name]["counts"]["inside_edge_band_rows"], 4)

if __name__ == "__main__": unittest.main()
