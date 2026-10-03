"""Known truth and a separate scalar oracle for v3 M06 qualification."""
from copy import deepcopy
import unittest

import numpy as np

from tests.unit.test_trajectory_measurement import sample_episode
from tests.unit.cycle_oracle import check_records
from true_rl_agent.evaluation.trajectory_translation.cycles import measure_contact_cycles
from true_rl_agent.tasks.stair_climbing.trajectory_cycles import summarize_cycles


def fixture(onsets, *, n=None, width=6, dt=.02):
    n = n or max(max(group) for group in onsets)+width-1
    ep = sample_episode(n, support=tuple("foot"+str(i) for i in range(len(onsets))), dt=dt)
    ep.arrays["support_contact_forces_w"][:] = 0
    for j, group in enumerate(onsets):
        for k in group:
            ep.arrays["support_contact_forces_w"][k-1:min(k-1+width, n), j, 2] = 2
    return ep


class ContactCycleTests(unittest.TestCase):
    def test_independent_oracle_rejects_unit_frame_even_with_resigned_id(self):
        import hashlib
        import json
        ep = fixture([[4,24,44,64,84], [9,29,49,69,89]])
        for field, value in (("unit", "milliseconds; phase degrees"), ("frame", "physical_contact_time")):
            for resign in (False, True):
                records = measure_contact_cycles(ep)
                for record in records:
                    record[field] = value
                    if resign:
                        record["record_id"] = "cycle-evidence:" + hashlib.sha256(
                            json.dumps(record,sort_keys=True).encode()).hexdigest()
                with self.subTest(field=field,resign=resign), self.assertRaisesRegex(AssertionError,"metadata"):
                    check_records(records,ep.steps,ep.arrays["support_contact_forces_w"],ep.dt_s)

    def checked(self, ep):
        records = measure_contact_cycles(ep)
        check_records(records, ep.steps, ep.arrays["support_contact_forces_w"], ep.dt_s)
        return records

    def test_known_period_overlap_edges_and_episode_equal_cohort(self):
        ep = fixture([[4, 24, 44, 64, 84, 104]])
        records = self.checked(ep)
        v = records[0]["value"]
        self.assertEqual(v["counts"]["qualified_windows"], 2)
        self.assertEqual(v["counts"]["qualified_unique_cycles"], 5)
        self.assertEqual(v["counts"]["covered_unique_edges"], 100)
        self.assertEqual(v["qualified_readout_span_s"], 2)
        self.assertEqual([w["median_period_s"] for w in v["windows"]], [.4, .4])
        ep2 = fixture([[4, 44, 84, 124, 164]])
        second = self.checked(ep2)
        summary = summarize_cycles([{"records": records}, {"records": second}])
        group = summary["groups"]["contact_cycles:foot0"]
        self.assertAlmostEqual(group["episode_median_qualified_cycle_period_s"]["mean"], .6)
        self.assertAlmostEqual(group["pooled_qualified_span_fraction"], 260/(108+168))

    def test_cv_and_median_deviation_independent_boundaries(self):
        for lengths, reason in [([17,17,23,23], None), ([17,17,23,24], "period_cv"),
                                ([20,20,20,24], None), ([20,20,20,25], "period_median_deviation")]:
            ep = fixture([[4, *list(4+np.cumsum(lengths))]])
            w = self.checked(ep)[0]["value"]["windows"][0]
            self.assertEqual(w["reasons"], [] if reason is None else [reason])

    def test_single_or_three_cycles_and_endpoint_are_not_invented(self):
        for onsets in ([4,24], [4,24,44,64]):
            v = self.checked(fixture([onsets]))[0]["value"]
            self.assertEqual(v["counts"]["complete_cycles"], len(onsets)-1)
            self.assertEqual(v["windows"], [])
        for tail in (1,2,3):
            v = self.checked(fixture([[4,24,44,64,84]], n=83+tail))[0]["value"]
            self.assertEqual(v["counts"]["complete_cycles"], 4)
            self.assertEqual(v["counts"]["qualified_windows"], int(tail==3))
        v = self.checked(fixture([[1,24,44,64,84]]))[0]["value"]
        self.assertEqual(v["onsets"][0]["step"], 24)
        self.assertTrue(v["runs"][0]["left_censored"])

    def test_raw_short_on_off_preserved_and_variants_independent(self):
        for width in (1, 2):
            for state in (0, 1):
                ep = fixture([[4,24,44,64,84,104]], width=10)
                start = 28 if state == 0 else 38
                ep.arrays["support_contact_forces_w"][start-1:start-1+width, 0, 2] = 2*state
                v = self.checked(ep)[0]["value"]
                self.assertTrue(any("debounce_sensitive" in w["reasons"] for w in v["windows"]))
                self.assertIn(start, v["debounce_sensitivity"][width-1]["changed_steps"])
                self.assertEqual(len(v["onsets"]), 7)

    def test_unknown_missing_and_gap_never_bridge_cycles(self):
        for invalid in (np.nan, np.inf):
            ep = fixture([[4,24,44,64,84]])
            ep.arrays["support_contact_forces_w"][49,0,0] = invalid
            v = self.checked(ep)[0]["value"]
            self.assertEqual(v["windows"], [])
            self.assertEqual(v["counts"]["unknown_rows"], 1)
        ep = fixture([[4,24,44,64,84]])
        ep.steps[50:] += 1
        v = self.checked(ep)[0]["value"]
        self.assertEqual(v["windows"], [])
        self.assertEqual(v["counts"]["missing_steps"], 1)

    def test_constant_phase_dt_scaling_and_history_irrelevance(self):
        for dt in (.02, .007):
            ep = fixture([list(range(4,165,20)), list(range(9,170,20))], dt=dt)
            records = self.checked(ep)
            phase = records[-1]["value"]["windows"][0]
            self.assertEqual(phase["raw_phase_samples"], [.25]*4)
            self.assertAlmostEqual(phase["mean_phase_turns"], .25)
            self.assertAlmostEqual(phase["circular_resultant"], 1)
            ep.arrays["support_contact_forces_w_history"][:] = 99999
            self.assertEqual(records, self.checked(ep))
        ep.arrays.pop("support_contact_forces_w_history")
        self.assertEqual(records, self.checked(ep))

    def test_phase_multiplicity_zero_wrap_unknown_and_concentration(self):
        ep = fixture([list(range(104,1005,100)), list(range(104,1005,50))], width=20)
        rows = self.checked(ep)[-1]["value"]["windows"]
        self.assertTrue(all("zero_or_multiple_target_onsets" in row["reasons"] for row in rows))
        ep = fixture([list(range(104,1005,100)), [103,205,303,405,503,605,703,805,903,1005]], width=20)
        rows = self.checked(ep)[-1]["value"]["windows"]
        self.assertTrue(all("zero_or_multiple_target_onsets" in row["reasons"] for row in rows))
        for shift in (0, 99):
            ep = fixture([list(range(104,1305,100)), list(range(104+shift,1305+shift,100))], width=20)
            row = self.checked(ep)[-1]["value"]["windows"][0]
            self.assertAlmostEqual(row["mean_phase_turns"], shift/100)
        for shift in (71, 72):
            starts = list(range(104,12105,1000))
            target = [k+500+(-shift if i%2==0 else shift) for i,k in enumerate(starts)]
            ep = fixture([starts,target], width=100)
            row = self.checked(ep)[-1]["value"]["windows"][0]
            self.assertEqual(row["reasons"], [] if shift==71 else ["phase_concentration"])
        ep = fixture([list(range(4,165,20)), list(range(9,170,20))])
        ep.arrays["support_contact_forces_w"][20,1,0] = np.nan
        self.assertIn("target_coverage_unknown", self.checked(ep)[-1]["value"]["windows"][0]["reasons"])

    def test_entity_order_source_identity_and_threshold(self):
        ep = fixture([list(range(4,165,20)), list(range(9,170,20))])
        original = self.checked(ep)
        ep.entities["support"] = tuple(reversed(ep.entities["support"]))
        ep.arrays["support_contact_forces_w"] = ep.arrays["support_contact_forces_w"][:,::-1]
        reordered = self.checked(ep)
        self.assertEqual(reordered[0]["value"], original[1]["value"])
        self.assertAlmostEqual(reordered[-1]["value"]["windows"][0]["mean_phase_turns"], .75)
        ep.source["manifest_sha256"] = "f"*64
        self.assertNotEqual(self.checked(ep)[0]["record_id"], reordered[0]["record_id"])
        ep.task["contact_threshold_n"] = 2
        with self.assertRaises(ValueError):
            measure_contact_cycles(ep)

    def test_phase_left_boundary_requires_preceding_target_readout(self):
        ep = fixture([list(range(20,181,20)), [20, *range(30,191,20)]], width=3)
        ep.arrays["support_contact_forces_w"][18,1,0] = np.nan
        row = self.checked(ep)[-1]["value"]["windows"][0]
        self.assertIn("target_coverage_unknown", row["reasons"])
        self.assertIsNone(row["mean_phase_turns"])

    def test_oracle_catches_self_signed_counterfeit_facts(self):
        ep = fixture([[4,24,44,64,84]])
        records = self.checked(ep)
        bad = deepcopy(records)
        bad[0]["value"]["windows"][0]["median_period_s"] = .3
        with self.assertRaises(AssertionError):
            check_records(bad, ep.steps, ep.arrays["support_contact_forces_w"], ep.dt_s)

    def test_threshold_equality_missing_current_and_equal_occupancy_are_not_rhythm(self):
        ep = fixture([list(range(4,105,20))],n=120,width=10)
        periodic = self.checked(ep)[0]["value"]
        forces = ep.arrays["support_contact_forces_w"]
        count = np.count_nonzero(forces[:,0,2])
        forces[:] = 0
        forces[:count,0,2] = 2
        constant = self.checked(ep)[0]["value"]
        self.assertGreater(periodic["counts"]["qualified_windows"],0)
        self.assertEqual(constant["windows"],[])
        forces[:] = 0
        forces[:,:,0] = 1
        self.assertEqual(self.checked(ep)[0]["value"]["onsets"],[])
        ep.arrays.pop("support_contact_forces_w")
        missing = measure_contact_cycles(ep)[0]
        self.assertEqual(missing["value"]["counts"]["unknown_rows"],120)
        self.assertIsNone(missing["value"]["qualified_span_fraction"])
        self.assertEqual(missing["evidence_ref"]["missing_channels"],["support_contact_forces_w"])


if __name__ == "__main__":
    unittest.main()
