from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tests.fakes.evaluation import (
    TEST_CAPABILITY_PROFILE,
    make_evaluation_binding,
    make_training_binding,
    make_verified_candidate_artifact,
)
from true_rl_agent.backends.base import BackendAdapter
from true_rl_agent.backends.registry import BackendRegistry
from true_rl_agent.backends.specs import BackendSpec
from true_rl_agent.candidates.artifact import VerifiedCandidateArtifactRef
from true_rl_agent.candidates.schema import CandidateChangeSet
from true_rl_agent.evaluation.subject import ExperimentSubject
from true_rl_agent.policy import PolicyInitialization
from true_rl_agent.runner.specs import (
    CheckpointOutputPlan,
    EvalJobSpec,
    TrainJobSpec,
)
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
            genesis_anchor_policy={
                "random": "bootstrap",
                "warm_start": "policy_baseline",
            },
            candidate_capability_profile=TEST_CAPABILITY_PROFILE,
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

    def evaluation_binding_for(self, task_spec: TaskSpec):
        _ = task_spec
        return make_evaluation_binding(
            public_seed_count=4,
            sealed_validation_seed_count=8,
        )

    def materialize_jobs(
        self,
        *,
        round_id: str,
        task_spec: TaskSpec,
        candidate: VerifiedCandidateArtifactRef,
        policy_initialization: PolicyInitialization,
        evaluation_binding=None,
        expected_initial_policy_state_hash=None,
        backend_execution_environment_manifest_hash=None,
        train_attempt_scope=None,
        eval_attempt_scope=None,
    ):
        _ = (
            expected_initial_policy_state_hash,
            backend_execution_environment_manifest_hash,
        )
        evaluation_binding = evaluation_binding or self.evaluation_binding_for(
            task_spec
        )
        train_spec = TrainJobSpec(
            job_id=f"{round_id}-train",
            task_id=task_spec.task_id,
            subject=ExperimentSubject.candidate(candidate.candidate_id),
            subject_manifest_hash=candidate.subject_manifest_hash,
            program_source_checksum=candidate.program_source_checksum,
            policy_initialization=policy_initialization,
            training_binding=make_training_binding(policy_initialization),
            budget_steps=128,
            execution_device="cuda:0",
            candidate_artifact=candidate,
            attempt_scope=train_attempt_scope,
        )
        eval_spec = EvalJobSpec(
            job_id=f"{round_id}-eval",
            task_id=task_spec.task_id,
            subject=ExperimentSubject.candidate(candidate.candidate_id),
            subject_manifest_hash=candidate.subject_manifest_hash,
            program_source_checksum=candidate.program_source_checksum,
            training_binding_hash=str(
                make_training_binding(policy_initialization).binding_hash
            ),
            capability_profile_hash=str(TEST_CAPABILITY_PROFILE.profile_hash),
            checkpoint=None,
            checkpoint_output_plan=CheckpointOutputPlan(
                uri=f"artifact://{round_id}/planned-eval.pt",
            ),
            public_seed_count=4,
            sealed_validation_seed_count=8,
            evaluation_binding=evaluation_binding,
            execution_device="cuda:0",
            attempt_scope=eval_attempt_scope,
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

        self.assertFalse(
            hasattr(adapter, "requires_verified_candidate_artifact")
        )
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
        candidate = make_verified_candidate_artifact(
            "candidate-001",
            change_set=CandidateChangeSet(
                changed_components=frozenset({"reward_callable"})
            ),
        )

        bundle = backend.materialize_jobs(
            round_id="round-001",
            task_spec=task.task_spec,
            candidate=candidate,
            policy_initialization=PolicyInitialization.random(),
        )

        self.assertEqual(bundle.train_spec.subject.kind, "candidate")
        self.assertEqual(bundle.train_spec.subject.subject_id, "candidate-001")
        self.assertIsNone(bundle.eval_spec.checkpoint)
        self.assertEqual(
            bundle.eval_spec.checkpoint_output_plan.uri,
            "artifact://round-001/planned-eval.pt",
        )
        self.assertFalse(hasattr(bundle, "decision"))


if __name__ == "__main__":
    unittest.main()
