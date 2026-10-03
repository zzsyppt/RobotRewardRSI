"""Audit-only qualification; production measurements and templates are frozen."""
import importlib.util
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("p2a_audit", ROOT / "PLAN/feedback_sys/NEW_SYS/P2a/audit_replay.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class RetriggerSelectionTests(unittest.TestCase):
    def scan(self, values, steps=None, dt=.02):
        forces = np.zeros((len(values), 1, 3))
        forces[:, 0, 2] = [np.nan if v == -1 else 2*v for v in values]
        return audit.single_current_on_witnesses(
            np.arange(1, len(values)+1) if steps is None else steps, forces, ["FL"], dt)

    def qualified(self, values, steps=None, dt=.02):
        return [w for w in self.scan(values, steps, dt) if w["qualifies_long_off"]]

    def test_exact_threshold_and_censored_offs_keep_complete_one_sample_on(self):
        w, = self.qualified([0]*25 + [1, 0])
        self.assertEqual(w["on_step"], 26)
        self.assertEqual(w["off_before"]["sample_cell_duration_proxy_s"], .5)
        self.assertEqual(w["off_before"]["observed_span_s"], .48)
        self.assertTrue(w["off_before"]["left_censored"])
        self.assertTrue(w["off_after"]["right_censored"])
        self.assertTrue(w["on"]["complete_run"])
        self.assertEqual(w["on"]["observed_span_s"], 0)
        self.assertEqual(w["off_after"]["sample_count"], 1)

    def test_short_before_cannot_use_long_after_or_combined_off_duration(self):
        for values in ([0]*24 + [1] + [0]*100, [0]*14 + [1] + [0]*14):
            with self.subTest(values=len(values)):
                self.assertEqual(len(self.scan(values)), 1)  # Valid short-on regression example.
                self.assertEqual(self.qualified(values), [])

    def test_unknown_does_not_bridge_before_on_or_resumption(self):
        for index in (12, 24, 25, 26):
            values = [0]*25 + [1, 0]
            values[index] = -1
            with self.subTest(index=index):
                self.assertEqual(self.qualified(values), [])

    def test_step_gaps_do_not_bridge_before_on_or_resumption(self):
        for index in (12, 25, 26):
            steps = np.arange(1, 28)
            steps[index:] += 1
            with self.subTest(index=index):
                self.assertEqual(self.qualified([0]*25 + [1, 0], steps), [])

    def test_multiple_on_or_unobserved_bracket_is_not_single_complete_on(self):
        for values in ([0]*25 + [1, 1, 0], [0]*25 + [1], [1] + [0]*25):
            with self.subTest(values=len(values)):
                self.assertEqual(self.qualified(values), [])

    def test_current_threshold_is_strict_and_nonfinite_is_unknown(self):
        force = np.zeros((27, 1, 3))
        force[:, 0, 2] = 1.0  # Finite equality is off, not on.
        force[25, 0, 2] = np.nextafter(1.0, 2.0)
        args = (np.arange(1, 28), force, ["FL"], .02)
        self.assertTrue(audit.single_current_on_witnesses(*args)[0]["qualifies_long_off"])
        force[24, 0, 0] = np.nan
        self.assertEqual(audit.single_current_on_witnesses(*args), [])

    def test_step_precedes_entity_order_and_entity_ties_use_stable_ids(self):
        force = np.zeros((30, 2, 3))
        force[25, 0, 2] = 2; force[26, 1, 2] = 2
        rows = audit.single_current_on_witnesses(np.arange(1, 31), force, ["RR", "FL"], .02)
        self.assertEqual([(r["on_step"], r["entity"]) for r in rows], [(26, "RR"), (27, "FL")])
        force[26, 1, 2] = 0; force[25, 1, 2] = 2
        rows = audit.single_current_on_witnesses(np.arange(1, 31), force, ["RR", "FL"], .02)
        self.assertEqual([r["entity"] for r in rows], ["FL", "RR"])


class SelectionLedgerTests(unittest.TestCase):
    def ledger(self):
        return {"sources": [{"source_id": "k1",
            "counterexamples": {"single_current_retrigger": [1, 0], "manager_without_current": [1, 5]},
            "absent_cases": ["single_current_retrigger", "another_case"],
            "examples": {"r1.e0": {"reasons": ["stratum", "single_current_retrigger"]}}}]}

    def test_legacy_labels_are_corrected_everywhere_without_mutating_original(self):
        frozen = self.ledger(); original = deepcopy(frozen)
        pre = audit.selection_for_verify(frozen, "k1")
        self.assertEqual(frozen, original)
        self.assertEqual(pre["counterexamples"], {"isolated_current_on": [1, 0], "manager_without_current": [1, 5]})
        self.assertEqual(pre["absent_cases"], ["isolated_current_on", "another_case"])
        self.assertEqual(pre["examples"]["r1.e0"]["reasons"], ["stratum", "isolated_current_on"])
        self.assertEqual(pre["long_off_qualification_status"], "not_established_by_original_ledger")

    def test_revised_ledger_keeps_qualified_labels_and_rejects_other_predicate(self):
        frozen = self.ledger(); frozen["retrigger_qualification"] = audit.RETRIGGER_QUALIFICATION
        pre = audit.selection_for_verify(frozen, "k1")
        self.assertEqual(pre["counterexamples"], frozen["sources"][0]["counterexamples"])
        self.assertEqual(pre["long_off_qualification_status"], "evaluated_by_post_review_selector")
        frozen["retrigger_qualification"] = "both sides sum >=0.5s"
        with self.assertRaises(AssertionError):
            audit.selection_for_verify(frozen, "k1")

    def test_cli_passes_selected_ledger_and_create_once_report_path(self):
        with patch.object(audit, "verify") as verify:
            audit.main(["verify", "--source", "k2", "--selection", "corrected.json", "--output", "new_audit.json"])
        verify.assert_called_once_with("k2", Path("corrected.json"), Path("new_audit.json"))
        with self.assertRaises(FileExistsError):
            audit.verify("k1", output=Path(__file__))


if __name__ == "__main__":
    unittest.main()
