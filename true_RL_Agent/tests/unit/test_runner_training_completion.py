"""Completed subprocess evidence can supply a missing formal-training declaration."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.unit import test_push_native_dev_replay as replay_fixtures
from true_rl_agent.agent_core.session_genesis import _validate_reference_train_result
from true_rl_agent.runner.checkpoint import file_sha256
from true_rl_agent.runner.subprocess_runner import SubprocessRunner


class RunnerTrainingCompletionTests(unittest.TestCase):
    def _fixture(self, root, cell):
        _, composition, contract, _, _ = replay_fixtures.fixture(root, cell)
        jobs = composition.backend_adapter.materialize_reference_jobs(
            session_id=contract["session_id"], task_spec=composition.task_plugin.task_spec,
            reference=composition.reference_program_artifact,
            artifact_root=root / "test-reference-source", policy_initialization=composition.policy_initialization,
            backend_execution_environment_manifest_hash=(
                replay_fixtures.TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_MANIFEST_HASH))
        path = Path(jobs.train_spec.metadata["summary_path"])
        summary = json.loads(path.read_text())
        # These are the common production Push/Stair completion fields. Fixture
        # bytes stay CPU-only and are never represented as actual policy weights.
        summary["training"]["budget_accounting"] = dict(summary["training"])
        summary["training"].update(initial_policy_state_hash="a" * 64, latest_checkpoint_iteration=499)
        return composition.runner, jobs.train_spec, path, summary

    def test_missing_marker_is_derived_from_complete_evidence_for_both_cells(self):
        for cell in replay_fixtures.REPLAY.CELLS:
            with self.subTest(cell=cell), tempfile.TemporaryDirectory() as directory, \
                 patch.object(SubprocessRunner, "_run_job", side_effect=AssertionError("retrained")):
                runner, spec, path, summary = self._fixture(Path(directory), cell)
                path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
                original, mtime = path.read_bytes(), path.stat().st_mtime_ns
                first = runner.recover_completed_train(spec)
                second = runner.recover_completed_train(spec)
                _validate_reference_train_result(spec, first)
                self.assertTrue(first.summary["formal_training_executed"])
                evidence = first.summary["runner_training_completion"]
                self.assertEqual(evidence["source_summary_sha256"], file_sha256(path))
                self.assertEqual(evidence["checkpoint_sha256"], first.checkpoint.checksum)
                self.assertEqual(evidence["final_checkpoint_iteration"], 499)
                self.assertEqual(first.summary, second.summary)
                self.assertEqual(path.read_bytes(), original)
                self.assertEqual(path.stat().st_mtime_ns, mtime)
                self.assertNotIn("formal_training_executed", json.loads(path.read_text()))

    def test_explicit_true_and_false_keep_exact_historical_summary(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(SubprocessRunner, "_run_job", side_effect=AssertionError("retrained")):
            runner, spec, path, summary = self._fixture(Path(directory), replay_fixtures.REPLAY.CELLS[0])
            for marker in (True, False):
                with self.subTest(marker=marker):
                    raw = {**summary, "formal_training_executed": marker}
                    path.write_text(json.dumps(raw), encoding="utf-8")
                    result = runner.recover_completed_train(spec)
                    self.assertEqual(result.summary, raw)
                    self.assertNotIn("runner_training_completion", result.summary)
                    if marker:
                        _validate_reference_train_result(spec, result)
                    else:
                        with self.assertRaisesRegex(ValueError, "did not execute formal training"):
                            _validate_reference_train_result(spec, result)

    def test_dry_run_incomplete_budget_final_iteration_or_missing_anchor_never_upgrade(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(SubprocessRunner, "_run_job", side_effect=AssertionError("retrained")):
            runner, spec, path, summary = self._fixture(Path(directory), replay_fixtures.REPLAY.CELLS[1])
            for case in ("dry_run", "budget", "iteration", "initial_hash"):
                with self.subTest(case=case):
                    raw = copy.deepcopy(summary)
                    if case == "dry_run":
                        raw["dry_run"] = True
                    elif case == "budget":
                        raw["training"]["budget_accounting"]["planned_env_steps"] -= 1
                    elif case == "iteration":
                        raw["training"]["latest_checkpoint_iteration"] = 498
                    else:
                        raw["training"].pop("initial_policy_state_hash")
                    path.write_text(json.dumps(raw), encoding="utf-8")
                    result = runner.recover_completed_train(spec)
                    self.assertEqual(result.summary, raw)
                    with self.assertRaisesRegex(ValueError, "did not execute formal training"):
                        _validate_reference_train_result(spec, result)


if __name__ == "__main__":
    unittest.main()
