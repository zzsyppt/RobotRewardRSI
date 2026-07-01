from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from true_rl_agent.policy import BaselinePolicyRef, PolicyInitialization
from true_rl_agent.runner.artifacts import RunnerArtifactLayout
from true_rl_agent.runner.budget import BudgetViolation, enforce_fixed_budget
from true_rl_agent.runner.checkpoint import safe_load_checkpoint, seal_checkpoint
from true_rl_agent.runner.isolation import IsolationPolicy
from true_rl_agent.runner.jobs import dry_run_eval_command, dry_run_train_command
from true_rl_agent.runner.specs import CheckpointRef, EvalJobSpec, TrainJobSpec
from true_rl_agent.safety.eval_seal import build_trusted_eval_seal
from true_rl_agent.safety.invariants import enforce_task_invariants
from true_rl_agent.tasks.base import TaskInvariantSpec


class RunnerJobTests(unittest.TestCase):
    def test_dry_run_commands_distinguish_random_and_warm_start_train_jobs(self) -> None:
        random_spec = TrainJobSpec(
            job_id="train-random",
            task_id="unitree_push_recovery",
            candidate_id="candidate-001",
            policy_initialization=PolicyInitialization.random(),
            budget_steps=128,
        )
        baseline_checkpoint = CheckpointRef(
            uri="baseline://push/policy.pt",
            checksum="sha256:baseline",
            role="warm_start",
        )
        warm_spec = TrainJobSpec(
            job_id="train-warm",
            task_id="unitree_push_recovery",
            candidate_id="candidate-002",
            policy_initialization=PolicyInitialization.warm_start(
                BaselinePolicyRef(
                    baseline_id="push-baseline",
                    task_id="unitree_push_recovery",
                    checkpoint=baseline_checkpoint,
                )
            ),
            budget_steps=128,
        )

        random_command = dry_run_train_command(random_spec)
        warm_command = dry_run_train_command(warm_spec)

        self.assertIn("--init random", random_command)
        self.assertNotIn("--warm-start", random_command)
        self.assertIn("--init warm_start", warm_command)
        self.assertIn("--warm-start baseline://push/policy.pt", warm_command)

    def test_eval_dry_run_command_uses_eval_checkpoint_only(self) -> None:
        eval_spec = EvalJobSpec(
            job_id="eval-001",
            task_id="unitree_push_recovery",
            candidate_id="candidate-001",
            checkpoint=CheckpointRef("artifact://round/best.pt", "sha256:eval", "eval", sealed=True),
            public_seed_count=4,
            trusted_seed_count=8,
        )

        command = dry_run_eval_command(eval_spec)

        self.assertIn("--checkpoint artifact://round/best.pt", command)
        self.assertIn("--public-seeds 4", command)
        self.assertNotIn("trusted_seed_count", command)

    def test_fixed_budget_check_blocks_budget_drift(self) -> None:
        spec = TrainJobSpec(
            job_id="train-001",
            task_id="unitree_push_recovery",
            candidate_id="candidate-001",
            policy_initialization=PolicyInitialization.random(),
            budget_steps=128,
        )

        enforce_fixed_budget(spec, expected_budget_steps=128)
        with self.assertRaises(BudgetViolation):
            enforce_fixed_budget(spec, expected_budget_steps=256)

    def test_artifact_layout_builds_session_round_train_and_eval_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            layout = RunnerArtifactLayout(root=Path(tmp), session_id="session-001")

            self.assertEqual(layout.session_dir.name, "session-001")
            self.assertEqual(layout.round_dir("round-001").name, "round-001")
            self.assertEqual(layout.train_dir("round-001").name, "train")
            self.assertEqual(layout.eval_dir("round-001").name, "eval")

    def test_isolation_policy_reserves_dry_run_subprocess_and_rootfs_modes(self) -> None:
        self.assertEqual(IsolationPolicy().kind, "dry_run")
        self.assertTrue(IsolationPolicy(kind="subprocess", subprocess_boundary=True).subprocess_boundary)
        self.assertEqual(IsolationPolicy(kind="rootfs", rootfs="/runner-rootfs").rootfs, "/runner-rootfs")
        with self.assertRaises(ValueError):
            IsolationPolicy(kind="rootfs")

    def test_checkpoint_sealing_and_safe_load_are_runner_owned(self) -> None:
        train_output = CheckpointRef(
            uri="artifact://round/train/raw.pt",
            checksum="sha256:raw",
            role="train_output",
        )

        sealed = seal_checkpoint(train_output, sealed_uri="artifact://round/checkpoints/best.pt")
        safe_result = safe_load_checkpoint(sealed)
        unsafe_result = safe_load_checkpoint(
            CheckpointRef(uri="artifact://round/unsealed.pt", checksum="sha256:raw", role="eval")
        )

        self.assertEqual(sealed.role, "eval")
        self.assertTrue(sealed.sealed)
        self.assertEqual(safe_result.status, "pass")
        self.assertEqual(safe_result.checkpoint_hash, "sha256:raw")
        self.assertEqual(unsafe_result.status, "fail")

    def test_task_invariant_violation_fails_before_training(self) -> None:
        expected = TaskInvariantSpec({"fall_threshold": 0.30, "episode_length": 512})
        changed = TaskInvariantSpec({"fall_threshold": 0.20, "episode_length": 512})

        with self.assertRaises(ValueError):
            enforce_task_invariants(expected=expected, actual=changed)

    def test_trusted_eval_seal_records_hashes_without_candidate_code(self) -> None:
        eval_spec = EvalJobSpec(
            job_id="eval-001",
            task_id="unitree_push_recovery",
            candidate_id="candidate-001",
            checkpoint=CheckpointRef("artifact://round/best.pt", "sha256:eval", "eval", sealed=True),
            public_seed_count=4,
            trusted_seed_count=8,
        )
        invariants = TaskInvariantSpec({"fall_threshold": 0.30})

        seal = build_trusted_eval_seal(eval_spec=eval_spec, task_invariants=invariants)

        self.assertEqual(len(seal.eval_config_hash), 64)
        self.assertEqual(seal.checkpoint_hash, "sha256:eval")
        self.assertEqual(len(seal.task_invariant_hash), 64)
        self.assertFalse(seal.candidate_code_loaded)


if __name__ == "__main__":
    unittest.main()
