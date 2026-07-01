"""Isaac Lab + RSL-RL backend dry-run adapter."""

from __future__ import annotations

from dataclasses import dataclass

from true_rl_agent.backends.base import BackendAdapter, BackendJobBundle
from true_rl_agent.backends.specs import BackendSpec
from true_rl_agent.candidates.schema import CandidatePackageRef
from true_rl_agent.policy import PolicyInitialization
from true_rl_agent.runner.specs import CheckpointRef, EvalJobSpec, TrainJobSpec
from true_rl_agent.tasks.base import TaskSpec


@dataclass(frozen=True)
class IsaacLabRslRlBackendConfig:
    isaac_lab_python: str
    train_entrypoint: str
    eval_entrypoint: str
    default_budget_steps: int
    public_seed_count: int
    trusted_seed_count: int
    robot: str = "unitree_g1"
    observation_schema_hash: str = "push_recovery_obs_v1"
    action_schema_hash: str = "unitree_actions_v1"
    policy_architecture: str = "rsl_rl_ppo_mlp"


@dataclass(frozen=True)
class BackendCompatibilityReport:
    compatible: bool
    reasons: tuple[str, ...]


class IsaacLabRslRlBackendAdapter(BackendAdapter):
    """Dry-run materializer for the first real backend target."""

    external_dependencies = (
        "Isaac Lab",
        "RSL-RL",
        "Unitree G1/H1 assets",
        "CUDA runtime on the execution server",
    )
    future_curriculum_capabilities = {
        "external_push_curriculum": {"supported": True, "effective_now": False},
        "train_only_dr_scale": {"supported": True, "effective_now": False},
    }

    def __init__(self, config: IsaacLabRslRlBackendConfig) -> None:
        self.config = config

    @property
    def backend_spec(self) -> BackendSpec:
        return BackendSpec(
            backend_id="isaac_lab_rsl_rl",
            supported_tasks=frozenset({"unitree_push_recovery"}),
            supported_policy_initializations=frozenset({"random", "warm_start"}),
            curriculum_capabilities=self.future_curriculum_capabilities,
        )

    def check_compatibility(self, *, task_spec: TaskSpec, policy_initialization: str) -> BackendCompatibilityReport:
        reasons: list[str] = []
        if task_spec.task_id not in self.backend_spec.supported_tasks:
            reasons.append(f"unsupported_task:{task_spec.task_id}")
        if policy_initialization not in self.backend_spec.supported_policy_initializations:
            reasons.append(f"unsupported_policy_initialization:{policy_initialization}")
        return BackendCompatibilityReport(compatible=not reasons, reasons=tuple(reasons))

    def materialize_jobs(
        self,
        *,
        round_id: str,
        task_spec: TaskSpec,
        candidate: CandidatePackageRef,
        policy_initialization: PolicyInitialization,
    ) -> BackendJobBundle:
        report = self.check_compatibility(
            task_spec=task_spec,
            policy_initialization=policy_initialization.kind,
        )
        if not report.compatible:
            raise ValueError(f"backend compatibility failed: {', '.join(report.reasons)}")

        planned_checkpoint = CheckpointRef(
            uri=f"artifact://{round_id}/checkpoints/best.pt",
            checksum=f"sha256:planned-{round_id}-{candidate.candidate_id}",
            role="eval",
        )
        train_spec = TrainJobSpec(
            job_id=f"{round_id}-train",
            task_id=task_spec.task_id,
            candidate_id=candidate.candidate_id,
            policy_initialization=policy_initialization,
            budget_steps=self.config.default_budget_steps,
            metadata={
                "backend_id": self.backend_id,
                "cli_command": self._train_command(
                    round_id=round_id,
                    candidate=candidate,
                    policy_initialization=policy_initialization,
                ),
                "external_dependencies": self.external_dependencies,
                "robot": self.config.robot,
                "observation_schema_hash": self.config.observation_schema_hash,
                "action_schema_hash": self.config.action_schema_hash,
                "policy_architecture": self.config.policy_architecture,
            },
        )
        eval_spec = EvalJobSpec(
            job_id=f"{round_id}-eval",
            task_id=task_spec.task_id,
            candidate_id=candidate.candidate_id,
            checkpoint=planned_checkpoint,
            public_seed_count=self.config.public_seed_count,
            trusted_seed_count=self.config.trusted_seed_count,
            metadata={
                "backend_id": self.backend_id,
                "cli_command": self._eval_command(
                    round_id=round_id,
                    candidate=candidate,
                    checkpoint=planned_checkpoint,
                ),
                "checkpoint_policy": "runner_sealed_train_output_required",
            },
        )
        return BackendJobBundle(
            train_spec=train_spec,
            eval_spec=eval_spec,
            task_invariants=task_spec.invariants,
        )

    def _train_command(
        self,
        *,
        round_id: str,
        candidate: CandidatePackageRef,
        policy_initialization: PolicyInitialization,
    ) -> str:
        parts = [
            "isaaclab-rsl-rl-train",
            f"--python {self.config.isaac_lab_python}",
            f"--entrypoint {self.config.train_entrypoint}",
            f"--round {round_id}",
            f"--candidate {candidate.path}",
            f"--budget-steps {self.config.default_budget_steps}",
            f"--init {policy_initialization.kind}",
        ]
        init_checkpoint = None
        if policy_initialization.baseline is not None:
            init_checkpoint = policy_initialization.baseline.checkpoint
        if init_checkpoint is not None:
            parts.append(f"--init-checkpoint {init_checkpoint.uri}")
        return " ".join(parts)

    def _eval_command(
        self,
        *,
        round_id: str,
        candidate: CandidatePackageRef,
        checkpoint: CheckpointRef,
    ) -> str:
        return " ".join(
            [
                "isaaclab-rsl-rl-eval",
                f"--python {self.config.isaac_lab_python}",
                f"--entrypoint {self.config.eval_entrypoint}",
                f"--round {round_id}",
                f"--candidate {candidate.path}",
                f"--checkpoint {checkpoint.uri}",
                f"--public-seeds {self.config.public_seed_count}",
            ]
        )
