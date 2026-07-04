from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from true_rl_agent.agent_core.decision import DecisionOutcome, decide_candidate
from true_rl_agent.backends.specs import BackendSpec
from true_rl_agent.candidates.schema import CandidateChangeSet, CandidatePackageRef, CandidateSurfacePolicy
from true_rl_agent.curriculum.schema import CurriculumPlan
from true_rl_agent.evaluation.public_result import PublicEvalResult
from true_rl_agent.policy import BaselinePolicyRef, PolicyInitialization
from true_rl_agent.runner.specs import CheckpointRef, EvalJobSpec, TrainJobSpec
from true_rl_agent.tasks.base import TaskInvariantSpec, TaskSpec


class CoreSchemaTests(unittest.TestCase):
    def test_policy_initialization_supports_random_and_warm_start(self) -> None:
        random_init = PolicyInitialization.random()
        self.assertEqual(random_init.kind, "random")
        self.assertIsNone(random_init.baseline)

        warm_start_checkpoint = CheckpointRef(
            uri="baseline://push/random_recovery_seed",
            checksum="sha256:warm-start",
            role="warm_start",
        )
        baseline = BaselinePolicyRef(
            baseline_id="push-recovery-seed",
            task_id="unitree_push_recovery",
            checkpoint=warm_start_checkpoint,
            compatibility_tags=frozenset({"unitree", "push_recovery"}),
        )
        warm_start = PolicyInitialization.warm_start(baseline)
        self.assertEqual(warm_start.kind, "warm_start")
        self.assertIs(warm_start.baseline, baseline)

    def test_train_job_start_checkpoint_is_separate_from_eval_checkpoint(self) -> None:
        train_start = CheckpointRef(
            uri="baseline://push/random_recovery_seed",
            checksum="sha256:warm-start",
            role="warm_start",
        )
        baseline = BaselinePolicyRef(
            baseline_id="push-recovery-seed",
            task_id="unitree_push_recovery",
            checkpoint=train_start,
        )
        train_spec = TrainJobSpec(
            job_id="train-001",
            task_id="unitree_push_recovery",
            candidate_id="candidate-code-level",
            policy_initialization=PolicyInitialization.warm_start(baseline),
            budget_steps=1024,
        )

        eval_checkpoint = CheckpointRef(
            uri="artifact://round-001/checkpoints/best.pt",
            checksum="sha256:eval",
            role="eval",
        )
        eval_spec = EvalJobSpec(
            job_id="eval-001",
            task_id="unitree_push_recovery",
            candidate_id="candidate-code-level",
            checkpoint=eval_checkpoint,
            public_seed_count=8,
            trusted_seed_count=16,
        )

        self.assertEqual(train_spec.training_start_checkpoint().role, "warm_start")
        self.assertEqual(eval_spec.checkpoint.role, "eval")
        self.assertNotEqual(train_spec.training_start_checkpoint(), eval_spec.checkpoint)

    def test_curriculum_plan_is_frozen_runner_owned_or_task_default_only(self) -> None:
        for mode in ("frozen", "runner_owned", "task_default"):
            plan = CurriculumPlan(mode=mode, llm_controlled=False)
            self.assertEqual(plan.mode, mode)
            self.assertFalse(plan.llm_controlled)

        with self.assertRaises(ValueError):
            CurriculumPlan(mode="llm_schedule", llm_controlled=True)

    def test_candidate_surface_policy_distinguishes_code_from_tuning_and_records(self) -> None:
        policy = CandidateSurfacePolicy()

        self.assertEqual(
            policy.classify(CandidateChangeSet(changed_components=frozenset({"reward_weight"}))).classification,
            "parameter_only",
        )
        self.assertFalse(
            policy.classify(CandidateChangeSet(changed_components=frozenset({"reward_weight"}))).eligible_for_policy_accept
        )
        self.assertEqual(
            policy.classify(CandidateChangeSet(changed_components=frozenset({"diagnostic"}))).classification,
            "diagnostic_only",
        )
        self.assertEqual(
            policy.classify(CandidateChangeSet(changed_components=frozenset({"failure_classifier"}))).classification,
            "engineering_only",
        )
        self.assertTrue(
            policy.classify(CandidateChangeSet(changed_components=frozenset({"reward_callable"}))).eligible_for_policy_accept
        )

    def test_task_backend_candidate_and_result_objects_project_public_state(self) -> None:
        invariants = TaskInvariantSpec(
            {
                "fall_threshold": 0.30,
                "episode_length": 512,
                "push_distribution": "runner_owned_target",
            }
        )
        task = TaskSpec(
            task_id="unitree_push_recovery",
            name="Unitree Push Recovery",
            supported_policy_initializations=frozenset({"random", "warm_start"}),
            invariants=invariants,
        )
        backend = BackendSpec(
            backend_id="isaac_lab_rsl_rl",
            supported_tasks=frozenset({"unitree_push_recovery"}),
            supported_policy_initializations=frozenset({"random", "warm_start"}),
        )
        candidate = CandidatePackageRef(
            candidate_id="candidate-001",
            path="artifact://candidate-001",
            change_set=CandidateChangeSet(changed_components=frozenset({"reward_callable"})),
        )
        result = PublicEvalResult(status="pass", public_score=0.25, metrics={"success_rate": 0.25})

        self.assertTrue(task.supports_policy_initialization("random"))
        self.assertTrue(backend.supports(task_id=task.task_id, policy_initialization="warm_start"))
        self.assertEqual(candidate.to_public_dict()["changed_components"], ["reward_callable"])
        self.assertEqual(result.to_public_dict()["public_score"], 0.25)
        self.assertNotIn("trusted_seed_count", result.to_public_dict())

    def test_task_invariant_spec_detects_changes(self) -> None:
        original = TaskInvariantSpec({"fall_threshold": 0.30, "episode_length": 512})
        same = TaskInvariantSpec({"fall_threshold": 0.30, "episode_length": 512})
        changed = TaskInvariantSpec({"fall_threshold": 0.20, "episode_length": 512})

        original.assert_compatible_with(same)
        with self.assertRaises(ValueError):
            original.assert_compatible_with(changed)

    def test_decision_outcomes_are_stable_strings(self) -> None:
        self.assertEqual(DecisionOutcome.POLICY_IMPROVEMENT_ACCEPT.value, "policy_improvement_accept")
        self.assertEqual(DecisionOutcome.DIAGNOSTIC_IMPROVEMENT_RECORD.value, "diagnostic_improvement_record")
        self.assertEqual(DecisionOutcome.ENGINEERING_IMPROVEMENT_RECORD.value, "engineering_improvement_record")
        self.assertEqual(DecisionOutcome.REJECT.value, "reject")

    def test_policy_improvement_requires_strict_public_score_improvement(self) -> None:
        surface = CandidateSurfacePolicy().classify(
            CandidateChangeSet(changed_components=frozenset({"reward_callable"}))
        )

        equal_score = decide_candidate(
            surface=surface,
            public_result=PublicEvalResult(status="pass", public_score=0.0, metrics={}),
            current_best_score=0.0,
            policy_accept_delta=0.0,
        )
        improved_score = decide_candidate(
            surface=surface,
            public_result=PublicEvalResult(status="pass", public_score=0.01, metrics={}),
            current_best_score=0.0,
            policy_accept_delta=0.0,
        )

        self.assertEqual(equal_score.outcome, DecisionOutcome.REJECT)
        self.assertFalse(equal_score.accepted)
        self.assertEqual(improved_score.outcome, DecisionOutcome.POLICY_IMPROVEMENT_ACCEPT)
        self.assertTrue(improved_score.accepted)

    def test_policy_improvement_requires_public_result_gates_to_pass(self) -> None:
        surface = CandidateSurfacePolicy().classify(
            CandidateChangeSet(changed_components=frozenset({"reward_callable"}))
        )

        gated_result = decide_candidate(
            surface=surface,
            public_result=PublicEvalResult(
                status="pass",
                public_score=0.90,
                metrics={
                    "acceptance": {"required_pass_gates": ["motion_quality_gate"]},
                    "motion_quality_gate": {"status": "fail", "reason": "unstable_motion"},
                },
            ),
            current_best_score=0.10,
            policy_accept_delta=0.0,
        )

        self.assertEqual(gated_result.outcome, DecisionOutcome.REJECT)
        self.assertFalse(gated_result.accepted)
        self.assertEqual(gated_result.reason, "required_gate_failed")

    def test_policy_improvement_uses_noise_aware_public_score_delta(self) -> None:
        surface = CandidateSurfacePolicy().classify(
            CandidateChangeSet(changed_components=frozenset({"reward_callable"}))
        )

        noisy_small_delta = decide_candidate(
            surface=surface,
            public_result=PublicEvalResult(
                status="pass",
                public_score=0.53,
                metrics={
                    "acceptance": {
                        "min_delta": 0.05,
                        "score_stats": {"standard_error": 0.01},
                        "best_score_stats": {"standard_error": 0.01},
                    }
                },
            ),
            current_best_score=0.50,
            policy_accept_delta=0.0,
        )
        clear_delta = decide_candidate(
            surface=surface,
            public_result=PublicEvalResult(
                status="pass",
                public_score=0.57,
                metrics={
                    "acceptance": {
                        "min_delta": 0.05,
                        "score_stats": {"standard_error": 0.01},
                        "best_score_stats": {"standard_error": 0.01},
                    }
                },
            ),
            current_best_score=0.50,
            policy_accept_delta=0.0,
        )

        self.assertEqual(noisy_small_delta.outcome, DecisionOutcome.REJECT)
        self.assertEqual(noisy_small_delta.reason, "noise_aware_delta_not_met")
        self.assertEqual(clear_delta.outcome, DecisionOutcome.POLICY_IMPROVEMENT_ACCEPT)
        self.assertEqual(clear_delta.reason, "noise_aware_delta")


if __name__ == "__main__":
    unittest.main()
