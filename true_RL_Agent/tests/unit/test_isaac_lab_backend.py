from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from true_rl_agent.backends.isaac_lab_rsl_rl import (
    IsaacLabRslRlBackendAdapter,
    IsaacLabRslRlBackendConfig,
)
from true_rl_agent.candidates.schema import CandidateChangeSet, CandidatePackageRef
from true_rl_agent.policy import BaselinePolicyRef, PolicyInitialization
from true_rl_agent.runner.specs import CheckpointRef
from true_rl_agent.tasks.base import TaskInvariantSpec, TaskSpec


class IsaacLabBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        self.adapter = IsaacLabRslRlBackendAdapter(
            IsaacLabRslRlBackendConfig(
                isaac_lab_python="python",
                train_entrypoint="scripts/isaaclab_rsl_rl_train.py",
                eval_entrypoint="scripts/isaaclab_rsl_rl_eval.py",
                default_budget_steps=2048,
                public_seed_count=4,
                trusted_seed_count=8,
            )
        )
        self.task = TaskSpec(
            task_id="unitree_push_recovery",
            name="Unitree Push Recovery",
            supported_policy_initializations=frozenset({"random", "warm_start"}),
            invariants=TaskInvariantSpec({"fall_threshold": 0.30}),
        )
        self.candidate = CandidatePackageRef(
            candidate_id="candidate-001",
            path="artifact://candidate-001",
            change_set=CandidateChangeSet(changed_components=frozenset({"reward_callable"})),
        )

    def test_backend_declares_external_dependencies_and_future_capabilities(self) -> None:
        dependencies = self.adapter.external_dependencies
        capability = self.adapter.future_curriculum_capabilities["external_push_curriculum"]

        self.assertIn("Isaac Lab", dependencies)
        self.assertIn("RSL-RL", dependencies)
        self.assertEqual(self.adapter.backend_spec.backend_id, "isaac_lab_rsl_rl")
        self.assertFalse(capability["effective_now"])

    def test_materializes_random_train_and_checkpoint_eval_cli_specs(self) -> None:
        bundle = self.adapter.materialize_jobs(
            round_id="round-001",
            task_spec=self.task,
            candidate=self.candidate,
            policy_initialization=PolicyInitialization.random(),
        )

        self.assertEqual(bundle.train_spec.policy_initialization.kind, "random")
        self.assertIsNone(bundle.train_spec.training_start_checkpoint())
        self.assertEqual(
            bundle.train_spec.metadata["cli_argv"][:2],
            ["python", "scripts/isaaclab_rsl_rl_train.py"],
        )
        self.assertIn("--init random", bundle.train_spec.metadata["cli_command"])
        self.assertIn("--candidate artifact://candidate-001", bundle.train_spec.metadata["cli_command"])
        self.assertTrue(bundle.train_spec.metadata["summary_path"].endswith("round-001/train/summary.json"))
        self.assertIn(bundle.train_spec.metadata["summary_path"], bundle.train_spec.metadata["cli_argv"])
        self.assertEqual(bundle.train_spec.metadata["sealed_checkpoint_uri"], "artifact://round-001/checkpoints/best.pt")
        self.assertTrue(bundle.train_spec.metadata["checkpoint_output_path"].endswith("round-001/checkpoints/best.pt"))
        self.assertIn(bundle.train_spec.metadata["checkpoint_output_path"], bundle.train_spec.metadata["cli_argv"])
        self.assertIn("--checkpoint", bundle.eval_spec.metadata["cli_command"])
        self.assertEqual(
            bundle.eval_spec.metadata["cli_argv"][:2],
            ["python", "scripts/isaaclab_rsl_rl_eval.py"],
        )
        self.assertIn("--public-accept-seeds", bundle.eval_spec.metadata["cli_argv"])
        self.assertEqual(
            bundle.eval_spec.metadata["cli_argv"][
                bundle.eval_spec.metadata["cli_argv"].index("--public-accept-seeds") + 1
            ],
            "1",
        )
        self.assertEqual(bundle.eval_spec.public_accept_seed_count, 1)
        self.assertEqual(bundle.eval_spec.eval_num_envs, 64)
        self.assertTrue(bundle.eval_spec.metadata["summary_path"].endswith("round-001/eval/summary.json"))
        self.assertIn(bundle.eval_spec.metadata["summary_path"], bundle.eval_spec.metadata["cli_argv"])
        self.assertEqual(bundle.task_invariants, self.task.invariants)

    def test_materializes_warm_start_train_with_explicit_init_checkpoint(self) -> None:
        checkpoint = CheckpointRef(
            uri="baseline://push/policy.pt",
            checksum="sha256:baseline",
            role="warm_start",
            sealed=True,
        )
        init = PolicyInitialization.warm_start(
            BaselinePolicyRef(
                baseline_id="push-baseline",
                task_id="unitree_push_recovery",
                checkpoint=checkpoint,
            )
        )

        bundle = self.adapter.materialize_jobs(
            round_id="round-002",
            task_spec=self.task,
            candidate=self.candidate,
            policy_initialization=init,
        )

        self.assertEqual(bundle.train_spec.training_start_checkpoint().uri, "baseline://push/policy.pt")
        self.assertIn("--init warm_start", bundle.train_spec.metadata["cli_command"])
        self.assertIn("--init-checkpoint baseline://push/policy.pt", bundle.train_spec.metadata["cli_command"])

    def test_structured_compatibility_failure_for_unsupported_task_or_initialization(self) -> None:
        unsupported_task = TaskSpec(
            task_id="stair_climbing",
            name="Stair Climbing",
            supported_policy_initializations=frozenset({"random"}),
            invariants=TaskInvariantSpec({}),
        )

        task_report = self.adapter.check_compatibility(
            task_spec=unsupported_task,
            policy_initialization="random",
        )
        init_report = self.adapter.check_compatibility(
            task_spec=self.task,
            policy_initialization="unsupported_init",
        )

        self.assertFalse(task_report.compatible)
        self.assertIn("unsupported_task:stair_climbing", task_report.reasons)
        self.assertFalse(init_report.compatible)
        self.assertIn("unsupported_policy_initialization:unsupported_init", init_report.reasons)

    def test_materializes_phase8b_full_budget_cli_with_explicit_max_iterations(self) -> None:
        adapter = IsaacLabRslRlBackendAdapter(
            IsaacLabRslRlBackendConfig(
                isaac_lab_python="python",
                train_entrypoint="scripts/isaaclab_rsl_rl_train.py",
                eval_entrypoint="scripts/isaaclab_rsl_rl_eval.py",
                default_budget_steps=49_152_000,
                train_num_envs=4096,
                train_max_iterations=500,
                train_steps_per_iteration=24,
                public_seed_count=4,
                trusted_seed_count=8,
            )
        )
        task = TaskSpec(
            task_id="unitree_push_recovery",
            name="Unitree Push Recovery",
            supported_policy_initializations=frozenset({"random", "warm_start"}),
            invariants=TaskInvariantSpec(
                {
                    "budget_steps": 49_152_000,
                    "train_num_envs": 4096,
                    "train_max_iterations": 500,
                    "train_steps_per_iteration": 24,
                }
            ),
        )

        bundle = adapter.materialize_jobs(
            round_id="round-full-budget",
            task_spec=task,
            candidate=self.candidate,
            policy_initialization=PolicyInitialization.random(),
        )

        argv = bundle.train_spec.metadata["cli_argv"]
        self.assertEqual(bundle.train_spec.budget_steps, 49_152_000)
        self.assertIn("--budget-steps", argv)
        self.assertEqual(argv[argv.index("--budget-steps") + 1], "49152000")
        self.assertIn("--num-envs", argv)
        self.assertEqual(argv[argv.index("--num-envs") + 1], "4096")
        self.assertIn("--max-iterations", argv)
        self.assertEqual(argv[argv.index("--max-iterations") + 1], "500")
        self.assertIn("--num-steps-per-env", argv)
        self.assertEqual(argv[argv.index("--num-steps-per-env") + 1], "24")
        self.assertIn("--max-iterations 500", bundle.train_spec.metadata["cli_command"])
        self.assertIn("--num-steps-per-env 24", bundle.train_spec.metadata["cli_command"])

    def test_rejects_backend_budget_mismatch_against_task_invariant(self) -> None:
        task = TaskSpec(
            task_id="unitree_push_recovery",
            name="Unitree Push Recovery",
            supported_policy_initializations=frozenset({"random", "warm_start"}),
            invariants=TaskInvariantSpec({"budget_steps": 49_152_000}),
        )

        with self.assertRaisesRegex(ValueError, "task_budget_mismatch"):
            self.adapter.materialize_jobs(
                round_id="round-budget-mismatch",
                task_spec=task,
                candidate=self.candidate,
                policy_initialization=PolicyInitialization.random(),
            )


if __name__ == "__main__":
    unittest.main()
