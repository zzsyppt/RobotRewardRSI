"""Synthetic CPU checks only; never open N5 captures, answers, or image files."""

from copy import deepcopy
import hashlib
import itertools
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_n5_media_selection as driver
from true_rl_agent.tasks.push_recovery.failure_classifier import (
    classify_recovery_episode, classify_force_recovery_episode, recovery_outcome_class,
)


def candidate(run, env, outcome, eligibility="complete"):
    return {"run_index": run, "env_index": env, "outcome": outcome,
            "eligibility": eligibility, "synthetic_only": True}


def outcome(task, success):
    if task == "stair":
        return {"success": success, "primary_reason": "success" if success else "timeout"}
    metrics = {"control_only": False, "push_events": [{"injection_verified": True, "recovered": success}],
               "planned_push_count": 1, "max_pitch_rad": 0.0, "min_pitch_rad": 0.0,
               "max_abs_roll_rad": 0.0, "min_base_height_m": 0.7,
               "max_xy_drift_m": 0.0, "final_recovery_condition": success}
    classified = classify_recovery_episode(metrics, {"fall_tilt_rad": 1.0, "fall_height_m": 0.3,
                                                     "max_displacement_m": None})
    return {"control_only": False, "success": classified["success"],
            "failure_label": classified["label"]}


def pool(task):
    return [candidate(run, env, outcome(task, success))
            for run, env, success in ((1, 9, True), (1, 10, False), (2, 9, True), (2, 10, False))]


def summary(result):
    fields = ("run_index", "env_index", "outcome_class", "rank_sha256", "selected", "selection_reason")
    return {"selected": [tuple(row[k] for k in fields) for row in result["selected"]],
            "pool": [tuple(row[k] for k in fields) for row in result["pool"]],
            "missing_classes": result["missing_classes"]}


class N5MediaSelectionChecks(unittest.TestCase):
    def setUp(self):
        def forbidden(*args, **kwargs):
            raise AssertionError("Synthetic choose_episodes checks must not read material")
        guard = patch.multiple(driver, read=forbidden, digest=forbidden,
                               load_numeric_evidence_manifest=forbidden,
                               read_native_snapshot_media=forbidden)
        guard.start()
        self.addCleanup(guard.stop)

    def test_exact_frozen_hash_and_one_minimum_per_outcome(self):
        self.assertEqual(driver.RULE, "p3new-n5-observer-episode-v1")
        for task in ("stair", "push"):
            result = driver.choose_episodes(task, pool(task))
            self.assertEqual([r["outcome_class"] for r in result["selected"]], ["success", "failure"])
            self.assertEqual(result["missing_classes"], [])
            for row in result["pool"]:
                text = f"p3new-n5-observer-episode-v1|{task}|{row['run_index']}|{row['env_index']}"
                self.assertEqual(row["rank_sha256"], hashlib.sha256(text.encode()).hexdigest())
            for chosen in result["selected"]:
                ranks = [r["rank_sha256"] for r in result["pool"] if r["outcome_class"] == chosen["outcome_class"]]
                self.assertEqual(chosen["rank_sha256"], min(ranks))
            self.assertEqual(sum(r["selected"] for r in result["pool"]), 2)

    def test_all_input_permutations_and_invented_report_fields_have_no_effect(self):
        for task in ("stair", "push"):
            candidates = pool(task)
            expected = summary(driver.choose_episodes(task, candidates))
            for permutation in itertools.permutations(candidates):
                annotated = deepcopy(list(permutation))
                for i, row in enumerate(annotated):
                    row.update(report_quality=1000 - i, observer_answer="synthetic favorable prose",
                               semantic_review_passed=i % 2 == 0)
                    row["outcome"]["report_quality"] = i
                self.assertEqual(summary(driver.choose_episodes(task, annotated)), expected)

    def test_missing_class_is_not_replaced_with_second_same_class(self):
        for task, success in itertools.product(("stair", "push"), (True, False)):
            candidates = [candidate(run, 9, outcome(task, success)) for run in (1, 2, 3)]
            result = driver.choose_episodes(task, candidates)
            self.assertEqual(len(result["selected"]), 1)
            self.assertEqual(result["missing_classes"], ["failure" if success else "success"])
            self.assertEqual(sum(r["selected"] for r in result["pool"]), 1)

    def test_missing_capture_and_truncated_eligibility_never_fill_slots(self):
        for task in ("stair", "push"):
            candidates = [candidate(1, 9, outcome(task, True), "missing_capture"),
                          candidate(1, 10, outcome(task, False), "truncated")]
            result = driver.choose_episodes(task, candidates)
            self.assertEqual(result["selected"], [])
            self.assertEqual(result["missing_classes"], ["success", "failure"])
            self.assertTrue(all(r["outcome_class"] is None and not r["selected"] for r in result["pool"]))

    def test_push_control_censored_and_unknown_outcomes_are_ineligible(self):
        invalid = [
            {"control_only": True, "success": True, "failure_label": None},
            {"control_only": True, "success": False, "failure_label": "fall_forward"},
            {"control_only": False, "success": False, "failure_label": "censored"},
            {"control_only": False, "success": False, "failure_label": "no_push_control"},
            {"control_only": False, "success": False, "failure_label": None},
            {"control_only": False, "success": True, "failure_label": "fall_forward"},
            {"control_only": False, "success": True, "failure_label": None},
            {"control_only": False, "success": False, "failure_label": "success_recovered"},
            {"control_only": False, "success": False, "failure_label": "new_unknown_outcome"},
            {"success": True, "failure_label": None},
            {"control_only": False, "success": None, "failure_label": None},
        ]
        result = driver.choose_episodes("push", [candidate(1, i, row) for i, row in enumerate(invalid)])
        self.assertEqual(result["selected"], [])
        self.assertEqual(result["missing_classes"], ["success", "failure"])

    def test_force_classifier_outcomes_obey_the_same_persisted_contract(self):
        for recovered in (True, False):
            metrics = {"control_only": False, "force_events": [{"injection_verified": True, "completed": True}],
                       "planned_push_count": 1, "max_pitch_rad": 0.0, "min_pitch_rad": 0.0,
                       "max_abs_roll_rad": 0.0, "min_base_height_m": 0.7,
                       "unplanned_wrench": False, "sequence_recovered": recovered,
                       "final_recovery_condition": recovered}
            classified = classify_force_recovery_episode(metrics, {"fall_tilt_rad": 1.0, "fall_height_m": 0.3})
            recorded = {"control_only": False, "success": classified["success"], "failure_label": classified["label"]}
            self.assertEqual(recovery_outcome_class(recorded), "success" if recovered else "failure")

    def test_stair_unknown_and_contradictory_outcomes_are_ineligible(self):
        invalid = [{}, {"success": None, "primary_reason": "success"},
                   {"success": True, "primary_reason": "timeout"},
                   {"success": False, "primary_reason": "success"},
                   {"success": False, "primary_reason": "censored"},
                   {"success": False, "primary_reason": ""},
                   {"success": False, "primary_reason": None},
                   {"success": 1, "primary_reason": "success"}]
        result = driver.choose_episodes("stair", [candidate(1, i, row) for i, row in enumerate(invalid)])
        self.assertEqual(result["selected"], [])
        self.assertEqual(result["missing_classes"], ["success", "failure"])

    def test_two_runs_same_env_keep_outcome_identity_and_inputs_unchanged(self):
        for task in ("stair", "push"):
            candidates = [candidate(1, 9, outcome(task, True)), candidate(2, 9, outcome(task, False))]
            before = deepcopy(candidates)
            result = driver.choose_episodes(task, candidates)
            self.assertEqual([(r["run_index"], r["env_index"]) for r in result["selected"]], [(1, 9), (2, 9)])
            for chosen, original in zip(result["selected"], before):
                self.assertEqual(chosen["outcome"], original["outcome"])
            self.assertEqual(candidates, before)
            result["selected"][0]["outcome"]["synthetic_mutation"] = True
            self.assertEqual(candidates, before)
        stair = driver.choose_episodes("stair", pool("stair"))["pool"]
        push = driver.choose_episodes("push", pool("push"))["pool"]
        self.assertTrue(all(a["rank_sha256"] != b["rank_sha256"] for a, b in zip(stair, push)))

    def test_duplicate_or_invalid_identity_is_rejected(self):
        row = candidate(1, 9, outcome("stair", True))
        with self.assertRaisesRegex(ValueError, "duplicate pool identity"):
            driver.choose_episodes("stair", [row, deepcopy(row)])
        for run, env in ((0, 9), (True, 9), (1, -1), (1, True), (1, 1.0)):
            with self.subTest(run=run, env=env), self.assertRaises(ValueError):
                driver.choose_episodes("stair", [candidate(run, env, outcome("stair", True))])
        with self.assertRaises(ValueError):
            driver.choose_episodes("other", [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
