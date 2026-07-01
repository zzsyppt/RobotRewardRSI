from __future__ import annotations

from true_rl_agent.backends.base import BackendAdapter, BackendJobBundle
from true_rl_agent.backends.specs import BackendSpec
from true_rl_agent.candidates.schema import CandidatePackageRef
from true_rl_agent.candidates.verifier import VerificationResult, CandidateVerifier
from true_rl_agent.evaluation.public_result import PublicEvalResult
from true_rl_agent.policy import PolicyInitialization
from true_rl_agent.runner.base import Runner, TrainJobResult
from true_rl_agent.runner.checkpoint import seal_checkpoint
from true_rl_agent.runner.specs import CheckpointRef, EvalJobSpec, TrainJobSpec
from true_rl_agent.tasks.base import TaskInvariantSpec, TaskPlugin, TaskSpec


class FakeTaskPlugin(TaskPlugin):
    @property
    def task_spec(self) -> TaskSpec:
        return TaskSpec(
            task_id="fake_push_recovery",
            name="Fake Push Recovery",
            supported_policy_initializations=frozenset({"random", "warm_start"}),
            invariants=TaskInvariantSpec(
                {
                    "fall_threshold": 0.30,
                    "episode_length": 512,
                    "push_distribution": "fixed_fake",
                }
            ),
        )


class FakeBackendAdapter(BackendAdapter):
    @property
    def backend_spec(self) -> BackendSpec:
        return BackendSpec(
            backend_id="fake_backend",
            supported_tasks=frozenset({"fake_push_recovery"}),
            supported_policy_initializations=frozenset({"random", "warm_start"}),
        )

    def materialize_jobs(
        self,
        *,
        round_id: str,
        task_spec: TaskSpec,
        candidate: CandidatePackageRef,
        policy_initialization: PolicyInitialization,
    ) -> BackendJobBundle:
        metadata = {"fake_public_score": candidate.metadata["fake_public_score"]}
        train_spec = TrainJobSpec(
            job_id=f"{round_id}-train",
            task_id=task_spec.task_id,
            candidate_id=candidate.candidate_id,
            policy_initialization=policy_initialization,
            budget_steps=128,
            metadata=metadata,
        )
        planned_checkpoint = CheckpointRef(
            uri=f"artifact://{round_id}/{candidate.candidate_id}/planned.pt",
            checksum="sha256:planned",
            role="eval",
        )
        eval_spec = EvalJobSpec(
            job_id=f"{round_id}-eval",
            task_id=task_spec.task_id,
            candidate_id=candidate.candidate_id,
            checkpoint=planned_checkpoint,
            public_seed_count=4,
            trusted_seed_count=8,
            metadata=metadata,
        )
        return BackendJobBundle(train_spec=train_spec, eval_spec=eval_spec)


class FakeRunner(Runner):
    def run_train(self, train_spec: TrainJobSpec) -> TrainJobResult:
        raw_checkpoint = CheckpointRef(
            uri=f"artifact://{train_spec.job_id}/best.pt",
            checksum=f"sha256:{train_spec.candidate_id}",
            role="train_output",
        )
        checkpoint = seal_checkpoint(
            raw_checkpoint,
            sealed_uri=f"artifact://{train_spec.job_id}/sealed/best.pt",
        )
        return TrainJobResult(
            job_id=train_spec.job_id,
            checkpoint=checkpoint,
            summary={"formal_training_executed": False, "budget_steps": train_spec.budget_steps},
        )

    def run_eval(self, eval_spec: EvalJobSpec) -> PublicEvalResult:
        score = float(eval_spec.metadata["fake_public_score"])
        return PublicEvalResult(
            status="pass",
            public_score=score,
            metrics={"success_rate": score, "public_seed_count": eval_spec.public_seed_count},
        )


class FakeVerifier(CandidateVerifier):
    def verify(self, candidate: CandidatePackageRef) -> VerificationResult:
        status = str(candidate.metadata.get("verifier_status", "pass"))
        return VerificationResult(status=status, reason="fake_verifier")
