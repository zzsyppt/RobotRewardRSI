"""Known truth and a separate scalar oracle for v3 M06 qualification."""
from copy import deepcopy
import unittest

import numpy as np

from tests.unit.trajectory_fixtures import sample_episode
from tests.unit.cycle_oracle import check_records
from true_rl_agent.evaluation.trajectory_translation.cycles import measure_contact_cycles


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
            self.assertAlmostEqual(records[0]["value"]["windows"][0]["median_period_s"],20*dt)
            ep.arrays["support_contact_forces_w_history"][:] = 99999
            self.assertEqual(records, self.checked(ep))
        ep.arrays.pop("support_contact_forces_w_history")
        self.assertEqual(records, self.checked(ep))


    def test_entity_order_source_identity_and_threshold(self):
        ep = fixture([list(range(4,165,20)), list(range(9,170,20))])
        original = self.checked(ep)
        ep.entities["support"] = tuple(reversed(ep.entities["support"]))
        ep.arrays["support_contact_forces_w"] = ep.arrays["support_contact_forces_w"][:,::-1]
        reordered = self.checked(ep)
        self.assertEqual(reordered[0]["value"], original[1]["value"])
        ep.source["manifest_sha256"] = "f"*64
        self.assertNotEqual(self.checked(ep)[0]["record_id"], reordered[0]["record_id"])
        ep.task["contact_threshold_n"] = 2
        self.assertEqual(measure_contact_cycles(ep)[0]["measurement_config"]["contact_threshold_n"],2)


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
