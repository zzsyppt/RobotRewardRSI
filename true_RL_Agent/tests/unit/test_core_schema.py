from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tests.fakes.evaluation import (
    TEST_CAPABILITY_PROFILE,
    TEST_PROGRAM_SOURCE_CHECKSUM,
    TEST_SUBJECT_MANIFEST_HASH,
    make_evaluation_binding,
    make_training_binding,
    make_verified_candidate_artifact,
)
from true_rl_agent.agent_core.decision import DecisionOutcome
from true_rl_agent.agent_core.decision import decide_candidate as _decide_candidate
from true_rl_agent.backends.specs import BackendSpec
from true_rl_agent.candidates.schema import (
    CandidateChangeSet,
    CandidatePackageRef,
    CandidateSurfacePolicy,
)
from true_rl_agent.curriculum.schema import CurriculumPlan
from true_rl_agent.evaluation.acceptance_protocol import (
    PAIRED_HEADROOM_DELTA_RULE,
    PUBLIC_SCORE_BINDING_ACCEPTANCE_MEAN,
    AcceptanceProtocol,
)
from true_rl_agent.evaluation.binding import EvaluationBindingV1
from true_rl_agent.evaluation.current_best import CurrentBestEvalStats
from true_rl_agent.evaluation.decision_record_contract import validate_decision_record
from true_rl_agent.evaluation.protocol import EvaluationProtocol
from true_rl_agent.evaluation.public_result import PublicEvalResult
from true_rl_agent.evaluation.subject import ExperimentSubject
from true_rl_agent.policy import BaselinePolicyRef, PolicyInitialization
from true_rl_agent.runner.specs import (
    CheckpointRef,
    EvalJobSpec,
    TrainJobSpec,
    require_execution_device,
)
from true_rl_agent.tasks.base import TaskInvariantSpec, TaskSpec
from true_rl_agent.tasks.push_recovery.candidate_surface import (
    PUSH_RECOVERY_CANDIDATE_SURFACE_POLICY,
)

TEST_DECISION_PROTOCOL = AcceptanceProtocol(
    protocol_id="core_schema_acceptance_v1",
    score_source="public_accept.score_stats.mean",
    min_score_count=3,
    absolute_accept_floor=0.05,
    min_delta=0.05,
    public_score_binding=PUBLIC_SCORE_BINDING_ACCEPTANCE_MEAN,
)
TEST_EVALUATION_PROTOCOL = EvaluationProtocol(
    eval_protocol_version="core_schema_eval_v1",
    public_seed_count=1,
    public_accept_seed_count=3,
    sealed_validation_seed_count=7,
    eval_num_envs=64,
)
TEST_EVALUATION_BINDING = EvaluationBindingV1(
    eval_protocol_hash=TEST_EVALUATION_PROTOCOL.protocol_hash,
    acceptance_protocol_hash=TEST_DECISION_PROTOCOL.protocol_hash,
    task_protocol_hash="f" * 64,
    seed_cohort_hash="1" * 64,
)
TEST_CHECKPOINT_HASH = "sha256:" + "2" * 64
TEST_EVAL_CHECKPOINT_HASH = "sha256:" + "3" * 64
TEST_TRAINING_BINDING_HASH = "4" * 64


def _bootstrap(initial_best_score: float) -> CurrentBestEvalStats:
    return CurrentBestEvalStats.bootstrap(
        initial_best_score=initial_best_score,
        acceptance_protocol=TEST_DECISION_PROTOCOL,
        evaluation_binding=TEST_EVALUATION_BINDING,
        anchor_id="core-schema-bootstrap",
    )


def decide_candidate(**kwargs):
    kwargs.pop("policy_accept_delta", None)
    public_result = kwargs["public_result"]
    metrics = copy.deepcopy(dict(public_result.metrics))
    acceptance = metrics.get("acceptance")
    if isinstance(acceptance, dict):
        acceptance.setdefault("score_source", TEST_DECISION_PROTOCOL.score_source)
        acceptance.setdefault(
            "public_score_binding",
            TEST_DECISION_PROTOCOL.public_score_binding,
        )
        acceptance.setdefault("min_score_count", TEST_DECISION_PROTOCOL.min_score_count)
        acceptance.setdefault(
            "absolute_accept_floor",
            TEST_DECISION_PROTOCOL.absolute_accept_floor,
        )
        acceptance.setdefault("min_delta", TEST_DECISION_PROTOCOL.min_delta)
        acceptance.setdefault(
            "acceptance_protocol_hash",
            TEST_DECISION_PROTOCOL.protocol_hash,
        )
        kwargs["public_result"] = PublicEvalResult(
            status=public_result.status,
            public_score=public_result.public_score,
            metrics=metrics,
        )
    kwargs["acceptance_protocol"] = TEST_DECISION_PROTOCOL
    return _decide_candidate(**kwargs)


class CoreSchemaTests(unittest.TestCase):
    def test_execution_device_requires_canonical_explicit_value(self) -> None:
        for value in ("cpu", "cuda:0", "cuda:12"):
            with self.subTest(valid=value):
                self.assertEqual(require_execution_device(value), value)

        for value in (None, True, 0, "", "cuda", "cuda:-1", "cuda:01", "CPU"):
            with self.subTest(invalid=value):
                with self.assertRaisesRegex(ValueError, "canonical"):
                    require_execution_device(value)

    def _code_surface(self):
        return PUSH_RECOVERY_CANDIDATE_SURFACE_POLICY.classify(
            CandidateChangeSet(changed_components=frozenset({"reward_callable"}))
        )

    def _full_eval_result(
        self,
        *,
        public_score: float,
        score_mean: float | None = None,
        standard_error: float = 0.01,
        protocol_hash: str = TEST_EVALUATION_PROTOCOL.protocol_hash,
        required_gates: list[str] | None = None,
        candidate_id: str = "candidate-001",
        checkpoint_hash: str = TEST_CHECKPOINT_HASH,
        evaluation_binding: EvaluationBindingV1 | None = None,
        acceptance_protocol: AcceptanceProtocol = TEST_DECISION_PROTOCOL,
        score_samples: list[float] | None = None,
    ) -> PublicEvalResult:
        binding = evaluation_binding or EvaluationBindingV1(
            eval_protocol_hash=protocol_hash,
            acceptance_protocol_hash=acceptance_protocol.protocol_hash,
            task_protocol_hash=TEST_EVALUATION_BINDING.task_protocol_hash,
            seed_cohort_hash=TEST_EVALUATION_BINDING.seed_cohort_hash,
        )
        gates = required_gates or [
            "accept_evidence_gate",
            "sealed_validation_gate",
            "budget_accounting",
            "motion_quality_gate",
            "behavior_quality_gate",
        ]
        score_stats = {
            "count": 3,
            "mean": public_score if score_mean is None else score_mean,
            "std": standard_error,
            "standard_error": standard_error,
            "eval_protocol_hash": protocol_hash,
        }
        metrics = {
            "evaluation_binding": binding.to_public_dict(),
            "evaluation_subject": {
                "schema_version": "true_rl_agent.experiment.subject.v2",
                "kind": "candidate",
                "id": candidate_id,
            },
            "subject_manifest_hash": TEST_SUBJECT_MANIFEST_HASH,
            "program_source_checksum": TEST_PROGRAM_SOURCE_CHECKSUM,
            "training_binding_hash": TEST_TRAINING_BINDING_HASH,
            "capability_profile_hash": str(
                TEST_CAPABILITY_PROFILE.profile_hash
            ),
            "public_accept": {
                "score_stats": dict(score_stats),
                "seed_results": [
                    {"run_index": 1, "status": "pass", "success_rate": score_stats["mean"]},
                    {"run_index": 2, "status": "pass", "success_rate": score_stats["mean"]},
                    {"run_index": 3, "status": "pass", "success_rate": score_stats["mean"]},
                ],
            },
            "acceptance": {
                "min_delta": acceptance_protocol.min_delta,
                "absolute_accept_floor": acceptance_protocol.absolute_accept_floor,
                "min_score_count": acceptance_protocol.min_score_count,
                "score_source": acceptance_protocol.score_source,
                "public_score_binding": acceptance_protocol.public_score_binding,
                "acceptance_protocol_hash": acceptance_protocol.protocol_hash,
                "score_stats": dict(score_stats),
                "required_pass_gates": gates,
            },
            "accept_evidence_gate": {"status": "pass"},
            "sealed_validation": _sealed_validation_evidence(
                protocol_hash=protocol_hash,
                score=float(score_stats["mean"]),
                evaluation_binding=binding,
                candidate_id=candidate_id,
                checkpoint_hash=checkpoint_hash,
            ),
            "sealed_validation_gate": _sealed_validation_gate(),
            "budget_accounting": {"status": "pass"},
            "motion_quality_gate": {"status": "pass"},
            "behavior_quality_gate": {"status": "pass"},
        }
        if acceptance_protocol.requires_paired_score_samples:
            metrics["acceptance"]["score_samples"] = list(score_samples or [])
        return PublicEvalResult(status="pass", public_score=public_score, metrics=metrics)

    def _accepted_best(
        self,
        *,
        score: float = 0.50,
        standard_error: float = 0.01,
        acceptance_protocol: AcceptanceProtocol = TEST_DECISION_PROTOCOL,
        evaluation_binding: EvaluationBindingV1 = TEST_EVALUATION_BINDING,
        score_samples: list[float] | None = None,
    ) -> CurrentBestEvalStats:
        public_result = self._full_eval_result(
            public_score=score,
            score_mean=score,
            standard_error=standard_error,
            candidate_id="candidate-best",
            acceptance_protocol=acceptance_protocol,
            evaluation_binding=evaluation_binding,
            score_samples=score_samples,
        )
        return CurrentBestEvalStats.from_public_result(
            round_id="round-best",
            candidate_id="candidate-best",
            public_result=public_result,
            acceptance_protocol=acceptance_protocol,
            subject_manifest_hash=TEST_SUBJECT_MANIFEST_HASH,
            program_source_checksum=TEST_PROGRAM_SOURCE_CHECKSUM,
            training_binding_hash=TEST_TRAINING_BINDING_HASH,
            capability_profile_hash=str(TEST_CAPABILITY_PROFILE.profile_hash),
            checkpoint_hash=TEST_CHECKPOINT_HASH,
            expected_evaluation_binding=evaluation_binding,
        )

    def test_policy_initialization_supports_random_and_warm_start(self) -> None:
        random_init = PolicyInitialization.random()
        self.assertEqual(random_init.kind, "random")
        self.assertIsNone(random_init.baseline)

        warm_start_checkpoint = CheckpointRef(
            uri="baseline://push/random_recovery_seed",
            checksum=TEST_CHECKPOINT_HASH,
            role="warm_start",
            sealed=True,
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
            checksum=TEST_CHECKPOINT_HASH,
            role="warm_start",
            sealed=True,
        )
        baseline = BaselinePolicyRef(
            baseline_id="push-recovery-seed",
            task_id="unitree_push_recovery",
            checkpoint=train_start,
        )
        candidate_artifact = make_verified_candidate_artifact(
            "candidate-code-level"
        )
        train_spec = TrainJobSpec(
            job_id="train-001",
            task_id="unitree_push_recovery",
            subject=ExperimentSubject.candidate("candidate-code-level"),
            subject_manifest_hash=candidate_artifact.subject_manifest_hash,
            program_source_checksum=candidate_artifact.program_source_checksum,
            policy_initialization=PolicyInitialization.warm_start(baseline),
            training_binding=make_training_binding(
                PolicyInitialization.warm_start(baseline),
                budget_steps=1024,
            ),
            budget_steps=1024,
            execution_device="cuda:0",
            candidate_artifact=candidate_artifact,
        )

        eval_checkpoint = CheckpointRef(
            uri="artifact://round-001/checkpoints/best.pt",
            checksum=TEST_EVAL_CHECKPOINT_HASH,
            role="eval",
            sealed=True,
        )
        eval_spec = EvalJobSpec(
            job_id="eval-001",
            task_id="unitree_push_recovery",
            subject=ExperimentSubject.candidate("candidate-code-level"),
            subject_manifest_hash=candidate_artifact.subject_manifest_hash,
            program_source_checksum=candidate_artifact.program_source_checksum,
            training_binding_hash=str(
                make_training_binding(
                    PolicyInitialization.warm_start(baseline),
                    budget_steps=1024,
                ).binding_hash
            ),
            capability_profile_hash=str(TEST_CAPABILITY_PROFILE.profile_hash),
            checkpoint=eval_checkpoint,
            public_seed_count=8,
            sealed_validation_seed_count=16,
            evaluation_binding=make_evaluation_binding(
                public_seed_count=8,
                sealed_validation_seed_count=16,
            ),
            execution_device="cuda:0",
        )

        self.assertEqual(train_spec.training_start_checkpoint().role, "warm_start")
        self.assertEqual(eval_spec.checkpoint.role, "eval")
        self.assertNotEqual(train_spec.training_start_checkpoint(), eval_spec.checkpoint)

    def test_job_specs_reject_non_object_metadata_at_ingestion(self) -> None:
        with self.assertRaisesRegex(ValueError, "train job metadata must be an object"):
            TrainJobSpec(
                job_id="train-invalid-metadata",
                task_id="unitree_push_recovery",
                subject=ExperimentSubject.candidate("candidate"),
                subject_manifest_hash=TEST_SUBJECT_MANIFEST_HASH,
                program_source_checksum=TEST_PROGRAM_SOURCE_CHECKSUM,
                policy_initialization=PolicyInitialization.random(),
                training_binding=make_training_binding(),
                budget_steps=128,
                execution_device="cuda:0",
                metadata=None,  # type: ignore[arg-type]
            )

        with self.assertRaisesRegex(ValueError, "eval job metadata must be an object"):
            EvalJobSpec(
                job_id="eval-invalid-metadata",
                task_id="unitree_push_recovery",
                subject=ExperimentSubject.candidate("candidate"),
                subject_manifest_hash=TEST_SUBJECT_MANIFEST_HASH,
                program_source_checksum=TEST_PROGRAM_SOURCE_CHECKSUM,
                training_binding_hash=str(make_training_binding().binding_hash),
                capability_profile_hash=str(TEST_CAPABILITY_PROFILE.profile_hash),
                checkpoint=CheckpointRef(
                    uri="artifact://candidate/eval.pt",
                    checksum=TEST_EVAL_CHECKPOINT_HASH,
                    role="eval",
                    sealed=True,
                ),
                public_seed_count=1,
                sealed_validation_seed_count=1,
                evaluation_binding=make_evaluation_binding(
                    public_seed_count=1,
                    sealed_validation_seed_count=1,
                ),
                execution_device="cuda:0",
                metadata=None,  # type: ignore[arg-type]
            )

    def test_curriculum_plan_is_frozen_runner_owned_or_task_default_only(self) -> None:
        for mode in ("frozen", "runner_owned", "task_default"):
            plan = CurriculumPlan(mode=mode, llm_controlled=False)
            self.assertEqual(plan.mode, mode)
            self.assertFalse(plan.llm_controlled)

        with self.assertRaises(ValueError):
            CurriculumPlan(mode="llm_schedule", llm_controlled=True)

    def test_candidate_surface_policy_distinguishes_code_from_tuning_and_records(self) -> None:
        policy = PUSH_RECOVERY_CANDIDATE_SURFACE_POLICY

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

    def test_candidate_surface_taxonomy_is_task_supplied_not_common_hardcoded(self) -> None:
        neutral = CandidateSurfacePolicy()
        vision_task = CandidateSurfacePolicy(
            code_level_components=frozenset(
                {"vision_policy", "task_specific_controller"}
            )
        )

        self.assertEqual(
            neutral.classify(
                CandidateChangeSet(frozenset({"recovery_logic"}))
            ).classification,
            "unsupported",
        )
        self.assertTrue(
            vision_task.classify(
                CandidateChangeSet(frozenset({"vision_policy"}))
            ).eligible_for_policy_accept
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
            genesis_anchor_policy={
                "random": "bootstrap",
                "warm_start": "policy_baseline",
            },
            candidate_capability_profile=TEST_CAPABILITY_PROFILE,
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
        equal_score = decide_candidate(
            surface=self._code_surface(),
            public_result=self._full_eval_result(public_score=0.04),
            current_best=_bootstrap(0.0),
            policy_accept_delta=0.0,
        )
        improved_score = decide_candidate(
            surface=self._code_surface(),
            public_result=self._full_eval_result(public_score=0.06),
            current_best=_bootstrap(0.0),
            policy_accept_delta=0.0,
        )

        self.assertEqual(equal_score.outcome, DecisionOutcome.REJECT)
        self.assertFalse(equal_score.accepted)
        self.assertEqual(improved_score.outcome, DecisionOutcome.POLICY_IMPROVEMENT_ACCEPT)
        self.assertTrue(improved_score.accepted)

    def test_policy_improvement_requires_public_result_gates_to_pass(self) -> None:
        gated_result = decide_candidate(
            surface=self._code_surface(),
            public_result=PublicEvalResult(
                status="pass",
                public_score=0.90,
                metrics={
                    "public_accept": {
                        "score_stats": {
                            "count": 3,
                            "mean": 0.90,
                            "standard_error": 0.01,
                            "eval_protocol_hash": "protocol-a",
                        }
                    },
                    "acceptance": {
                        "score_stats": {
                            "count": 3,
                            "mean": 0.90,
                            "standard_error": 0.01,
                            "eval_protocol_hash": "protocol-a",
                        },
                        "required_pass_gates": [
                            "accept_evidence_gate",
                            "budget_accounting",
                            "sealed_validation_gate",
                            "motion_quality_gate",
                        ],
                    },
                    "accept_evidence_gate": {"status": "pass"},
                    "budget_accounting": {"status": "pass"},
                    "sealed_validation_gate": {"status": "pass"},
                    "motion_quality_gate": {"status": "fail", "reason": "unstable_motion"},
                },
            ),
            current_best=_bootstrap(0.10),
            policy_accept_delta=0.0,
        )

        self.assertEqual(gated_result.outcome, DecisionOutcome.REJECT)
        self.assertFalse(gated_result.accepted)
        self.assertEqual(gated_result.reason, "required_gate_failed")

    def test_policy_improvement_requires_public_accept_split_score_stats(self) -> None:
        result = self._full_eval_result(public_score=0.20, score_mean=0.20)
        metrics = dict(result.metrics)
        metrics.pop("public_accept")

        decision = decide_candidate(
            surface=self._code_surface(),
            public_result=PublicEvalResult(
                status=result.status,
                public_score=result.public_score,
                metrics=metrics,
            ),
            current_best=_bootstrap(0.10),
            policy_accept_delta=0.0,
        )

        self.assertEqual(decision.outcome, DecisionOutcome.REJECT)
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.reason, "required_score_missing")

    def test_policy_improvement_does_not_require_task_specific_gates_in_common_logic(self) -> None:
        result = self._full_eval_result(
            public_score=0.20,
            score_mean=0.20,
            required_gates=["accept_evidence_gate", "budget_accounting", "sealed_validation_gate"],
        )
        metrics = dict(result.metrics)
        metrics.pop("motion_quality_gate")
        metrics.pop("behavior_quality_gate")

        decision = decide_candidate(
            surface=self._code_surface(),
            public_result=PublicEvalResult(
                status=result.status,
                public_score=result.public_score,
                metrics=metrics,
            ),
            current_best=_bootstrap(0.10),
            policy_accept_delta=0.0,
        )

        self.assertEqual(decision.outcome, DecisionOutcome.POLICY_IMPROVEMENT_ACCEPT)
        self.assertTrue(decision.accepted)

    def test_policy_improvement_uses_noise_aware_public_score_delta(self) -> None:
        noisy_small_delta = decide_candidate(
            surface=self._code_surface(),
            public_result=self._full_eval_result(public_score=0.53, score_mean=0.53, standard_error=0.01),
            current_best=self._accepted_best(score=0.50, standard_error=0.01),
            policy_accept_delta=0.0,
        )
        clear_delta = decide_candidate(
            surface=self._code_surface(),
            public_result=self._full_eval_result(public_score=0.57, score_mean=0.57, standard_error=0.01),
            current_best=self._accepted_best(score=0.50, standard_error=0.01),
            policy_accept_delta=0.0,
        )

        self.assertEqual(noisy_small_delta.outcome, DecisionOutcome.REJECT)
        self.assertEqual(noisy_small_delta.reason, "noise_aware_delta_not_met")
        self.assertEqual(clear_delta.outcome, DecisionOutcome.POLICY_IMPROVEMENT_ACCEPT)
        self.assertEqual(clear_delta.reason, "noise_aware_delta")

    def test_policy_improvement_uses_paired_headroom_delta(self) -> None:
        protocol = AcceptanceProtocol(
            protocol_id="core_schema_paired_acceptance_v1",
            score_source="public_accept.score_stats.mean",
            min_score_count=3,
            absolute_accept_floor=0.05,
            min_delta=0.0,
            public_score_binding=PUBLIC_SCORE_BINDING_ACCEPTANCE_MEAN,
            decision_rule=PAIRED_HEADROOM_DELTA_RULE,
            score_ceiling=1.0,
            headroom_reduction_fraction=0.2,
            paired_standard_error_multiplier=1.5,
        )
        binding = EvaluationBindingV1(
            eval_protocol_hash=TEST_EVALUATION_PROTOCOL.protocol_hash,
            acceptance_protocol_hash=protocol.protocol_hash,
            task_protocol_hash=TEST_EVALUATION_BINDING.task_protocol_hash,
            seed_cohort_hash=TEST_EVALUATION_BINDING.seed_cohort_hash,
        )
        best_samples = [0.95, 0.95, 0.95]
        candidate_samples = [1.0, 0.95, 1.0]
        candidate_score = sum(candidate_samples) / len(candidate_samples)
        current_best = self._accepted_best(
            score=sum(best_samples) / len(best_samples),
            acceptance_protocol=protocol,
            evaluation_binding=binding,
            score_samples=best_samples,
        )
        decision = _decide_candidate(
            surface=self._code_surface(),
            public_result=self._full_eval_result(
                public_score=candidate_score,
                score_mean=candidate_score,
                acceptance_protocol=protocol,
                evaluation_binding=binding,
                score_samples=candidate_samples,
            ),
            current_best=current_best,
            acceptance_protocol=protocol,
        )

        self.assertEqual(
            decision.outcome,
            DecisionOutcome.POLICY_IMPROVEMENT_ACCEPT,
        )
        self.assertEqual(decision.reason, "paired_headroom_delta")
        self.assertEqual(
            decision.comparison["threshold_components"]["decision_rule"],
            PAIRED_HEADROOM_DELTA_RULE,
        )
        self.assertGreater(
            decision.comparison["delta"],
            decision.comparison["threshold"],
        )

        mismatched_binding = EvaluationBindingV1(
            eval_protocol_hash=binding.eval_protocol_hash,
            acceptance_protocol_hash=binding.acceptance_protocol_hash,
            task_protocol_hash="e" * 64,
            seed_cohort_hash=binding.seed_cohort_hash,
        )
        mismatched = _decide_candidate(
            surface=self._code_surface(),
            public_result=self._full_eval_result(
                public_score=candidate_score,
                score_mean=candidate_score,
                acceptance_protocol=protocol,
                evaluation_binding=mismatched_binding,
                score_samples=candidate_samples,
            ),
            current_best=current_best,
            acceptance_protocol=protocol,
        )
        self.assertEqual(mismatched.reason, "evaluation_binding_mismatch")
        self.assertNotIn("threshold_components", mismatched.comparison)
        self.assertNotIn("delta", mismatched.comparison)

    def test_policy_improvement_rejects_insufficient_accept_evidence_count(self) -> None:
        result = decide_candidate(
            surface=self._code_surface(),
            public_result=PublicEvalResult(
                status="pass",
                public_score=0.95,
                metrics={
                    "public_accept": {
                        "score_stats": {
                            "count": 1,
                            "mean": 0.95,
                            "std": 0.0,
                            "standard_error": 0.0,
                            "eval_protocol_hash": "protocol-a",
                        }
                    },
                    "acceptance": {
                        "min_delta": 0.05,
                        "min_score_count": 3,
                        "score_stats": {
                            "count": 1,
                            "mean": 0.95,
                            "std": 0.0,
                            "standard_error": 0.0,
                            "eval_protocol_hash": "protocol-a",
                        },
                        "required_pass_gates": ["sealed_validation_gate"],
                    },
                    "sealed_validation_gate": _sealed_validation_gate(),
                },
            ),
            current_best=_bootstrap(0.10),
            policy_accept_delta=0.0,
        )

        self.assertEqual(result.outcome, DecisionOutcome.REJECT)
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "insufficient_accept_evidence")

    def test_policy_improvement_rejects_eval_protocol_hash_mismatch_when_stats_are_present(self) -> None:
        result = decide_candidate(
            surface=self._code_surface(),
            public_result=self._full_eval_result(
                public_score=0.95,
                score_mean=0.95,
                protocol_hash="2" * 64,
            ),
            current_best=self._accepted_best(score=0.10),
            policy_accept_delta=0.0,
        )

        self.assertEqual(result.outcome, DecisionOutcome.REJECT)
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "evaluation_binding_mismatch")

    def test_policy_decision_uses_canonical_candidate_protocol_binding(self) -> None:
        mutations = {
            "missing_acceptance_hash": (
                lambda metrics: metrics["acceptance"]["score_stats"].pop(
                    "eval_protocol_hash"
                ),
                "eval_protocol_hash_missing",
            ),
            "missing_public_accept_hash": (
                lambda metrics: metrics["public_accept"]["score_stats"].pop(
                    "eval_protocol_hash"
                ),
                "eval_protocol_hash_missing",
            ),
            "conflicting_hash": (
                lambda metrics: metrics["public_accept"]["score_stats"].__setitem__(
                    "eval_protocol_hash", "protocol-other"
                ),
                "accept_score_stats_untrusted",
            ),
        }
        for label, (mutate, expected_reason) in mutations.items():
            with self.subTest(label=label):
                public_result = self._full_eval_result(
                    public_score=0.90,
                    score_mean=0.90,
                )
                metrics = copy.deepcopy(public_result.metrics)
                mutate(metrics)

                decision = decide_candidate(
                    surface=self._code_surface(),
                    public_result=PublicEvalResult(
                        status=public_result.status,
                        public_score=public_result.public_score,
                        metrics=metrics,
                    ),
                    current_best=_bootstrap(0.10),
                    policy_accept_delta=0.0,
                )

                self.assertFalse(decision.accepted)
                self.assertEqual(decision.reason, expected_reason)

    def test_policy_improvement_compares_canonical_candidate_stats_to_current_best(self) -> None:
        result = decide_candidate(
            surface=self._code_surface(),
            public_result=self._full_eval_result(public_score=0.54, score_mean=0.54),
            current_best=self._accepted_best(score=0.50, standard_error=0.01),
            policy_accept_delta=0.0,
        )

        self.assertEqual(result.outcome, DecisionOutcome.REJECT)
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "noise_aware_delta_not_met")
        self.assertEqual(result.comparison["candidate_score"], 0.54)
        self.assertEqual(result.comparison["current_best_score"], 0.50)
        self.assertEqual(result.comparison["current_best"]["accepted_round_id"], "round-best")

    def test_policy_improvement_rejects_incomplete_sealed_validation_gate(self) -> None:
        public_result = self._full_eval_result(public_score=0.90, score_mean=0.90)
        metrics = dict(public_result.metrics)
        metrics["sealed_validation_gate"] = {"status": "pass"}

        result = decide_candidate(
            surface=self._code_surface(),
            public_result=PublicEvalResult(
                status=public_result.status,
                public_score=public_result.public_score,
                metrics=metrics,
            ),
            current_best=_bootstrap(0.10),
            policy_accept_delta=0.0,
        )

        self.assertEqual(result.outcome, DecisionOutcome.REJECT)
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "required_gate_failed")

    def test_policy_improvement_rejects_sealed_validation_internal_failure(self) -> None:
        public_result = self._full_eval_result(public_score=0.90, score_mean=0.90)
        metrics = dict(public_result.metrics)
        metrics["sealed_validation_gate"] = _sealed_validation_gate(
            checkpoint_safety_status="fail",
            candidate_isolation_status="pass",
        )
        metrics["sealed_validation_gate"]["status"] = "pass"

        result = decide_candidate(
            surface=self._code_surface(),
            public_result=PublicEvalResult(
                status=public_result.status,
                public_score=public_result.public_score,
                metrics=metrics,
            ),
            current_best=_bootstrap(0.10),
            policy_accept_delta=0.0,
        )

        self.assertEqual(result.outcome, DecisionOutcome.REJECT)
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "required_gate_failed")

    def test_policy_improvement_rejects_top_level_eval_candidate_contamination(self) -> None:
        public_result = self._full_eval_result(public_score=0.90, score_mean=0.90)
        metrics = dict(public_result.metrics)
        metrics["candidate_code_loaded_in_eval"] = True

        result = decide_candidate(
            surface=self._code_surface(),
            public_result=PublicEvalResult(
                status=public_result.status,
                public_score=public_result.public_score,
                metrics=metrics,
            ),
            current_best=_bootstrap(0.10),
            policy_accept_delta=0.0,
        )

        self.assertEqual(result.outcome, DecisionOutcome.REJECT)
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "candidate_contamination_detected")

    def test_policy_improvement_records_consumed_gates_and_comparison_detail(self) -> None:
        result = decide_candidate(
            surface=self._code_surface(),
            public_result=self._full_eval_result(public_score=0.58, score_mean=0.58),
            current_best=self._accepted_best(score=0.50, standard_error=0.01),
            policy_accept_delta=0.0,
        )

        public = result.to_public_dict()

        self.assertEqual(result.outcome, DecisionOutcome.POLICY_IMPROVEMENT_ACCEPT)
        self.assertIn("sealed_validation_gate", result.consumed_gates)
        self.assertEqual(result.gate_statuses["sealed_validation_gate"], "pass")
        self.assertEqual(public["comparison"]["threshold"], 0.05)
        self.assertAlmostEqual(public["comparison"]["delta"], 0.08)
        self.assertEqual(public["comparison"]["best_score_stats"]["mean"], 0.50)
        self.assertEqual(
            public["consumed_gates"],
            [
                "accept_evidence_gate",
                "sealed_validation_gate",
                "budget_accounting",
                "motion_quality_gate",
                "behavior_quality_gate",
            ],
        )

    def test_policy_improvement_requires_sealed_validation_gate_to_be_consumed(self) -> None:
        result = decide_candidate(
            surface=self._code_surface(),
            public_result=self._full_eval_result(
                public_score=0.90,
                score_mean=0.90,
                required_gates=["budget_accounting", "motion_quality_gate"],
            ),
            current_best=self._accepted_best(score=0.50),
            policy_accept_delta=0.0,
        )

        self.assertEqual(result.outcome, DecisionOutcome.REJECT)
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "sealed_validation_gate_not_required")

    def test_bootstrap_accept_still_requires_full_eval_gates_and_absolute_floor(self) -> None:
        public_result = self._full_eval_result(
            public_score=0.08,
            score_mean=0.08,
            candidate_id="candidate-bootstrap",
        )
        result = decide_candidate(
            surface=self._code_surface(),
            public_result=public_result,
            current_best=_bootstrap(0.10),
            policy_accept_delta=0.0,
        )

        self.assertEqual(result.outcome, DecisionOutcome.POLICY_IMPROVEMENT_ACCEPT)
        self.assertTrue(result.accepted)
        self.assertEqual(result.reason, "absolute_accept_floor")
        self.assertEqual(result.comparison["mode"], "bootstrap")
        self.assertEqual(result.comparison["absolute_accept_floor"], 0.05)
        self.assertTrue(
            validate_decision_record(
                {
                    "candidate": {
                        "candidate_id": "candidate-bootstrap",
                        "changed_components": ["reward_callable"],
                    },
                    "candidate_surface": {
                        "classification": "code_level",
                        "eligible_for_policy_accept": True,
                        "reason": "code_level_candidate",
                    },
                    "verifier": {"status": "pass"},
                    "jobs": {
                        "train": {
                            "job_id": "bootstrap-train",
                            "subject": ExperimentSubject.candidate(
                                "candidate-bootstrap"
                            ).to_public_dict(),
                            "subject_manifest_hash": TEST_SUBJECT_MANIFEST_HASH,
                            "program_source_checksum": TEST_PROGRAM_SOURCE_CHECKSUM,
                            "training_binding": {
                                "binding_hash": TEST_TRAINING_BINDING_HASH,
                                "capability_profile_hash": str(
                                    TEST_CAPABILITY_PROFILE.profile_hash
                                ),
                            },
                        },
                        "eval": {
                            "job_id": "bootstrap-eval",
                            "subject": ExperimentSubject.candidate(
                                "candidate-bootstrap"
                            ).to_public_dict(),
                            "subject_manifest_hash": TEST_SUBJECT_MANIFEST_HASH,
                            "program_source_checksum": TEST_PROGRAM_SOURCE_CHECKSUM,
                            "training_binding_hash": TEST_TRAINING_BINDING_HASH,
                            "capability_profile_hash": str(
                                TEST_CAPABILITY_PROFILE.profile_hash
                            ),
                            **TEST_EVALUATION_PROTOCOL.to_public_dict(),
                            "checkpoint": {
                                "checksum": TEST_CHECKPOINT_HASH
                            },
                        },
                    },
                    "train_result": {"job_id": "bootstrap-train"},
                    "decision": result.to_public_dict(),
                    "public_result": public_result.to_public_dict(),
                },
                acceptance_protocol=TEST_DECISION_PROTOCOL,
            ).valid
        )

    def test_bootstrap_absolute_floor_uses_canonical_acceptance_score(self) -> None:
        result = decide_candidate(
            surface=self._code_surface(),
            public_result=self._full_eval_result(public_score=0.04, score_mean=0.04),
            current_best=_bootstrap(0.10),
            policy_accept_delta=0.0,
        )

        self.assertEqual(result.outcome, DecisionOutcome.REJECT)
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "absolute_accept_floor_not_met")
        self.assertEqual(result.comparison["candidate_score"], 0.04)


def _sealed_validation_gate(
    *,
    checkpoint_safety_status: str = "pass",
    candidate_isolation_status: str = "pass",
) -> dict[str, object]:
    status = "pass" if checkpoint_safety_status == "pass" and candidate_isolation_status == "pass" else "fail"
    return {
        "status": status,
        "checkpoint_safety_status": checkpoint_safety_status,
        "candidate_isolation_status": candidate_isolation_status,
        "candidate_code_loaded_in_eval": False,
        "candidate_package_loaded_in_eval": False,
        "candidate_path_argument_present": False,
        "eval_protocol_hash_match_status": "pass",
        "evaluation_binding_hash_match_status": "pass",
        "checkpoint_hash_match_status": "pass",
        "holdout_rollout_status": "pass",
        "reason": "sealed_validation_complete",
    }


def _sealed_validation_evidence(
    *,
    protocol_hash: str,
    score: float,
    evaluation_binding: EvaluationBindingV1 = TEST_EVALUATION_BINDING,
    candidate_id: str = "candidate-001",
    checkpoint_hash: str = TEST_CHECKPOINT_HASH,
) -> dict[str, object]:
    return {
        "schema_version": "true_rl_agent.sealed_validation.evidence.v3",
        "status": "pass",
        "reason": "test_holdout_passed",
        "expected_seed_count": 7,
        "observed_seed_count": 7,
        "eval_protocol_hash": protocol_hash,
        "evaluation_binding_hash": evaluation_binding.binding_hash,
        "evaluation_subject": {
            "schema_version": "true_rl_agent.experiment.subject.v2",
            "kind": "candidate",
            "id": candidate_id,
        },
        "subject_manifest_hash": TEST_SUBJECT_MANIFEST_HASH,
        "program_source_checksum": TEST_PROGRAM_SOURCE_CHECKSUM,
        "training_binding_hash": TEST_TRAINING_BINDING_HASH,
        "capability_profile_hash": str(TEST_CAPABILITY_PROFILE.profile_hash),
        "checkpoint_hash": checkpoint_hash,
        "score_stats": {
            "count": 7,
            "mean": score,
            "std": 0.0,
            "standard_error": 0.0,
            "eval_protocol_hash": protocol_hash,
        },
    }


if __name__ == "__main__":
    unittest.main()
