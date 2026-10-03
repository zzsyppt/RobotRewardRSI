from copy import deepcopy
import inspect
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from tests.unit.test_trajectory_measurement import sample_episode
from tests.unit.joint_limit_oracle import asset_bounds, check_record
from true_rl_agent.evaluation.trajectory_translation.joint_limits import measure_joint_limits
from true_rl_agent.tasks.stair_climbing.trajectory_joint_limits import (
    ASSET_FILE, nominal_binding, build_joint_limit_supplement, _base_report, summarize, QUALIFIED_SOURCES, _canonical_hash,
)
from true_rl_agent.tasks.stair_climbing.trajectory_adapter import _load_contract
from true_rl_agent.runner.numeric_evidence import NumericEvidenceError
from true_rl_agent.evaluation.process_report import report_bytes
from tests.unit import test_stair_cycle_supplement as cycles
from tests.unit import test_stair_trajectory_report as reports


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

    def test_episode_equal_differs_from_pooled_samples_and_time(self):
        rows = []
        for values, dt in (([[0, 0]], .1), ([[10, 0]]*3, .02)):
            rows.append({"joint_names": ["a", "b"], "records": self.checked(fixture(values, dt=dt))})
        g = summarize(rows)["groups"]["a"]
        self.assertEqual(g["episode_equal"]["inside_edge_band_fraction"]["mean"], .5)
        self.assertEqual(g["pooled_sample_fraction"]["inside_edge_band"], .25)
        self.assertAlmostEqual(g["pooled_sample_cell_time_fraction"]["inside_edge_band"], .625)


class JointLimitIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cycles.CycleSupplementTests.setUpClass()
        cls.f = cycles.CycleSupplementTests
        cls.archive = _load_contract(cls.f.reference)
        cls.source_patch = patch.dict(QUALIFIED_SOURCES, {cls.f.reference["manifest_sha256"]: _canonical_hash(cls.f.base)})
        cls.source_patch.start()

    @classmethod
    def tearDownClass(cls):
        cls.source_patch.stop()
        cycles.CycleSupplementTests.tearDownClass()

    def test_wrong_asset_and_name_mapping_blocked(self):
        for mutation in ("sha", "root", "unresolved"):
            arc = SimpleNamespace(manifest=deepcopy(self.archive.manifest), projection=self.archive.projection)
            c = arc.manifest["metadata"]["resolved_cell_manifest"]["robot_profile"]["asset_closure"]
            if mutation == "sha": c["entries"][0]["sha256"] = "0"*64
            elif mutation == "root": c["root_identity"] = "different.usd"
            else: c["unresolved"] = ["unknown.usd"]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): nominal_binding(arc)

    def test_resigned_base_domain_swap_rejected(self):
        for key in ("archive_hash", "projection_hash", "checkpoint_hash", "evaluation_plan_hash"):
            bad = deepcopy(self.f.base); bad["domain"][key] = "0"*64; reports.resign(bad)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "source/domain"):
                _base_report(bad, self.archive, self.f.reference)
        bad = deepcopy(self.f.base)
        bad["report"]["process"][0]["text"] = "Runtime joint saturation confirmed."
        reports.resign(bad)
        with self.assertRaisesRegex(ValueError, "base content"):
            _base_report(bad, self.archive, self.f.reference)

    def test_build_preview_from_raw_only_and_persist_exact_bytes(self):
        output = self.f.fixture.root/"limits"
        manifest = build_joint_limit_supplement(self.f.reference, output, source_id="fixture", base_public=self.f.base)
        report = json.loads((output/"report.json").read_bytes())
        preview = json.loads((output/"preview.json").read_bytes())
        receipt = json.loads((output/"preview_receipt.json").read_bytes())
        self.assertEqual(receipt["combined_bytes"], (output/"preview.json").stat().st_size)
        self.assertLessEqual(receipt["combined_bytes"], 8192)
        self.assertTrue(receipt["included_ids"])
        self.assertEqual(preview["base_report"], self.f.base)
        expected = {c["id"]: c for c in report["optional_statements"]}
        self.assertEqual(preview["optional_nominal_joint_limit_statements"], [expected[i] for i in receipt["included_ids"]])
        self.assertEqual(manifest["episode_count"], 288)
        self.assertEqual(set(inspect.signature(build_joint_limit_supplement).parameters),
                         {"reference", "output", "source_id", "base_public"})
        # A mutated/resigned sidecar is not a raw archive and cannot enter the build.
        for mutation in ("entity", "unit"):
            bad = deepcopy(report)
            bad["optional_statements"][0]["text"] = mutation
            bad["optional_statements"][0]["id"] = "resigned"
            with self.subTest(mutation=mutation), self.assertRaises(NumericEvidenceError):
                build_joint_limit_supplement(bad, output/mutation, source_id="fixture")

    def test_exact_8192_and_one_byte_over_omit_whole_first_sentence(self):
        original = self.f.fixture.root/"budget_source"
        build_joint_limit_supplement(self.f.reference, original, source_id="fixture", base_public=self.f.base)
        first = json.loads((original/"report.json").read_bytes())["optional_statements"][0]
        candidate = {"schema_version": "stair.nominal_joint_limit_preview.v1", "status": "offline_not_production_delivery",
                     "base_report": self.f.base, "optional_nominal_joint_limit_statements": [first]}
        padding = 8192-report_bytes(candidate)
        self.assertGreater(padding, 0)
        for extra in (0, 1):
            base = deepcopy(self.f.base)
            base["report"]["process"][0]["text"] += "x"*(padding+extra)
            reports.resign(base)
            out = self.f.fixture.root/("budget_"+str(extra))
            with patch.dict(QUALIFIED_SOURCES, {self.f.reference["manifest_sha256"]: _canonical_hash(base)}):
                build_joint_limit_supplement(self.f.reference, out, source_id="fixture", base_public=base)
            receipt = json.loads((out/"preview_receipt.json").read_bytes())
            if extra == 0:
                self.assertEqual(receipt["combined_bytes"], 8192)
                self.assertIn(first["id"], receipt["included_ids"])
            else:
                self.assertIn(first["id"], [x["id"] for x in receipt["omitted"]])
            self.assertEqual(receipt["combined_bytes"], (out/"preview.json").stat().st_size)


if __name__ == "__main__": unittest.main()
