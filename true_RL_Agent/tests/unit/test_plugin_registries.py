from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from true_rl_agent.backends.base import BackendAdapter
from true_rl_agent.backends.registry import BackendRegistry
from true_rl_agent.backends.specs import BackendSpec
from true_rl_agent.candidates.schema import CandidateChangeSet, CandidatePackageRef
from true_rl_agent.policy import PolicyInitialization
from true_rl_agent.runner.specs import CheckpointRef, EvalJobSpec, TrainJobSpec
from true_rl_agent.tasks.base import TaskInvariantSpec, TaskPlugin, TaskSpec
from true_rl_agent.tasks.compatibility import check_task_backend_compatibility
from true_rl_agent.tasks.registry import TaskRegistry


class MinimalTaskPlugin(TaskPlugin):
    @property
    def task_spec(self) -> TaskSpec:
        return TaskSpec(
            task_id="unitree_push_recovery",
            name="Unitree Push Recovery",
            supported_policy_initializations=frozenset({"random", "warm_start"}),
            invariants=TaskInvariantSpec({"fall_threshold": 0.30}),
        )


class MinimalBackendAdapter(BackendAdapter):
    @property
    def backend_spec(self) -> BackendSpec:
        return BackendSpec(
            backend_id="isaac_lab_rsl_rl",
            supported_tasks=frozenset({"unitree_push_recovery"}),
            supported_policy_initializations=frozenset({"random"}),
        )

    def materialize_jobs(
        self,
        *,
        round_id: str,
        task_spec: TaskSpec,
        candidate: CandidatePackageRef,
        policy_initialization: PolicyInitialization,
    ):
        train_spec = TrainJobSpec(
            job_id=f"{round_id}-train",
            task_id=task_spec.task_id,
            candidate_id=candidate.candidate_id,
            policy_initialization=policy_initialization,
            budget_steps=128,
        )
        eval_spec = EvalJobSpec(
            job_id=f"{round_id}-eval",
            task_id=task_spec.task_id,
            candidate_id=candidate.candidate_id,
            checkpoint=CheckpointRef(
                uri=f"artifact://{round_id}/planned-eval.pt",
                checksum="sha256:planned",
                role="eval",
            ),
            public_seed_count=4,
            trusted_seed_count=8,
        )
        return self.job_bundle_type(train_spec=train_spec, eval_spec=eval_spec)


class PluginRegistryTests(unittest.TestCase):
    def test_task_registry_resolves_registered_task_and_rejects_duplicates(self) -> None:
        registry = TaskRegistry()
        plugin = MinimalTaskPlugin()

        registry.register(plugin)

        self.assertIs(registry.get("unitree_push_recovery"), plugin)
        with self.assertRaises(ValueError):
            registry.register(plugin)
        with self.assertRaises(KeyError):
            registry.get("missing-task")

    def test_backend_registry_resolves_registered_backend_and_rejects_duplicates(self) -> None:
        registry = BackendRegistry()
        adapter = MinimalBackendAdapter()

        registry.register(adapter)

        self.assertIs(registry.get("isaac_lab_rsl_rl"), adapter)
        with self.assertRaises(ValueError):
            registry.register(adapter)
        with self.assertRaises(KeyError):
            registry.get("missing-backend")

    def test_compatibility_check_combines_task_and_backend_capabilities(self) -> None:
        task = MinimalTaskPlugin()
        backend = MinimalBackendAdapter()

        random_report = check_task_backend_compatibility(
            task_plugin=task,
            backend_adapter=backend,
            policy_initialization="random",
        )
        warm_start_report = check_task_backend_compatibility(
            task_plugin=task,
            backend_adapter=backend,
            policy_initialization="warm_start",
        )

        self.assertTrue(random_report.compatible)
        self.assertEqual(random_report.reasons, ())
        self.assertFalse(warm_start_report.compatible)
        self.assertIn("backend_does_not_support_policy_initialization:warm_start", warm_start_report.reasons)

    def test_plugin_boundaries_do_not_expose_training_or_decision_authority(self) -> None:
        task = MinimalTaskPlugin()
        backend = MinimalBackendAdapter()

        for forbidden in ("training_command", "train_command", "run_training", "decide", "accept", "reject"):
            self.assertFalse(hasattr(task, forbidden), f"TaskPlugin must not expose {forbidden}")
        for forbidden in ("decide", "accept", "reject", "append_ledger"):
            self.assertFalse(hasattr(backend, forbidden), f"BackendAdapter must not expose {forbidden}")

    def test_backend_adapter_materializes_jobs_without_decision_authority(self) -> None:
        task = MinimalTaskPlugin()
        backend = MinimalBackendAdapter()
        candidate = CandidatePackageRef(
            candidate_id="candidate-001",
            path="artifact://candidate-001",
            change_set=CandidateChangeSet(changed_components=frozenset({"reward_callable"})),
        )

        bundle = backend.materialize_jobs(
            round_id="round-001",
            task_spec=task.task_spec,
            candidate=candidate,
            policy_initialization=PolicyInitialization.random(),
        )

        self.assertEqual(bundle.train_spec.candidate_id, "candidate-001")
        self.assertEqual(bundle.eval_spec.checkpoint.role, "eval")
        self.assertFalse(hasattr(bundle, "decision"))


if __name__ == "__main__":
    unittest.main()
