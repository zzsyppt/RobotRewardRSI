"""Isaac Lab + RSL-RL backend job materializer."""

from __future__ import annotations

import posixpath
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
    public_accept_seed_count: int = 1
    robot: str = "unitree_g1"
    observation_schema_hash: str = "push_recovery_obs_v1"
    action_schema_hash: str = "unitree_actions_v1"
    policy_architecture: str = "rsl_rl_ppo_mlp"
    artifact_root: str = "outputs/runner"
    train_profile: str = "strong_push"
    eval_profile: str = "strong_push"
    train_num_envs: int = 256
    train_max_iterations: int | None = None
    train_steps_per_iteration: int | None = None
    eval_num_envs: int = 64
    work_dir: str = "."


@dataclass(frozen=True)
class BackendCompatibilityReport:
    compatible: bool
    reasons: tuple[str, ...]


class IsaacLabRslRlBackendAdapter(BackendAdapter):
    """Materialize fixed train/eval jobs for the first real backend target."""

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
        self._assert_training_budget_matches_task(task_spec)

        planned_checkpoint = CheckpointRef(
            uri=f"artifact://{round_id}/checkpoints/best.pt",
            checksum=f"sha256:planned-{round_id}-{candidate.candidate_id}",
            role="eval",
        )
        train_summary_path = self._summary_path(round_id=round_id, phase="train")
        eval_summary_path = self._summary_path(round_id=round_id, phase="eval")
        checkpoint_output_path = self._checkpoint_output_path(round_id=round_id)
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
                    summary_path=train_summary_path,
                    checkpoint_output_path=checkpoint_output_path,
                ),
                "cli_argv": self._train_argv(
                    round_id=round_id,
                    candidate=candidate,
                    policy_initialization=policy_initialization,
                    summary_path=train_summary_path,
                    checkpoint_output_path=checkpoint_output_path,
                ),
                "summary_path": train_summary_path,
                "sealed_checkpoint_uri": planned_checkpoint.uri,
                "checkpoint_output_path": checkpoint_output_path,
                "work_dir": self.config.work_dir,
                "external_dependencies": self.external_dependencies,
                "robot": self.config.robot,
                "observation_schema_hash": self.config.observation_schema_hash,
                "action_schema_hash": self.config.action_schema_hash,
                "policy_architecture": self.config.policy_architecture,
                "fixed_train_protocol": {
                    **dict(task_spec.fixed_train_protocol),
                    "budget_steps": self.config.default_budget_steps,
                },
            },
        )
        eval_spec = EvalJobSpec(
            job_id=f"{round_id}-eval",
            task_id=task_spec.task_id,
            candidate_id=candidate.candidate_id,
            checkpoint=planned_checkpoint,
            public_seed_count=self.config.public_seed_count,
            trusted_seed_count=self.config.trusted_seed_count,
            public_accept_seed_count=self.config.public_accept_seed_count,
            eval_num_envs=self.config.eval_num_envs,
            metadata={
                "backend_id": self.backend_id,
                "cli_command": self._eval_command(
                    round_id=round_id,
                    candidate=candidate,
                    checkpoint=planned_checkpoint,
                    summary_path=eval_summary_path,
                ),
                "cli_argv": self._eval_argv(
                    round_id=round_id,
                    candidate=candidate,
                    checkpoint=planned_checkpoint,
                    summary_path=eval_summary_path,
                ),
                "summary_path": eval_summary_path,
                "work_dir": self.config.work_dir,
                "checkpoint_policy": "runner_sealed_train_output_required",
                "fixed_eval_protocol": dict(task_spec.fixed_eval_protocol),
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
        summary_path: str | None = None,
        checkpoint_output_path: str | None = None,
    ) -> str:
        return " ".join(
            self._train_argv(
                round_id=round_id,
                candidate=candidate,
                policy_initialization=policy_initialization,
                summary_path=summary_path,
                checkpoint_output_path=checkpoint_output_path,
            )
        )

    def _train_argv(
        self,
        *,
        round_id: str,
        candidate: CandidatePackageRef,
        policy_initialization: PolicyInitialization,
        summary_path: str | None = None,
        checkpoint_output_path: str | None = None,
    ) -> list[str]:
        summary_path = summary_path or self._summary_path(round_id=round_id, phase="train")
        checkpoint_output_path = checkpoint_output_path or self._checkpoint_output_path(round_id=round_id)
        parts = [
            self.config.isaac_lab_python,
            self.config.train_entrypoint,
            "--round",
            round_id,
            "--candidate",
            candidate.path,
            "--budget-steps",
            str(self.config.default_budget_steps),
            "--init",
            policy_initialization.kind,
            "--summary-path",
            summary_path,
            "--checkpoint-output",
            checkpoint_output_path,
            "--profile",
            self.config.train_profile,
            "--num-envs",
            str(self.config.train_num_envs),
        ]
        if self.config.train_steps_per_iteration is not None:
            parts.extend(["--num-steps-per-env", str(self.config.train_steps_per_iteration)])
        if self.config.train_max_iterations is not None:
            parts.extend(["--max-iterations", str(self.config.train_max_iterations)])
        parts.append("--headless")
        init_checkpoint = None
        if policy_initialization.baseline is not None:
            init_checkpoint = policy_initialization.baseline.checkpoint
        if init_checkpoint is not None:
            parts.extend(["--init-checkpoint", init_checkpoint.uri])
        return parts

    def _assert_training_budget_matches_task(self, task_spec: TaskSpec) -> None:
        invariants = dict(task_spec.invariants.values)
        checks = {
            "budget_steps": self.config.default_budget_steps,
            "train_num_envs": self.config.train_num_envs,
            "train_max_iterations": self.config.train_max_iterations,
            "train_steps_per_iteration": self.config.train_steps_per_iteration,
            "eval_num_envs": self.config.eval_num_envs,
            "public_seed_count": self.config.public_seed_count,
            "public_accept_seed_count": self.config.public_accept_seed_count,
        }
        for key, actual in checks.items():
            if key not in invariants or actual is None:
                continue
            expected = invariants[key]
            if int(expected) != int(actual):
                raise ValueError(f"task_budget_mismatch:{key}:expected={expected}:actual={actual}")

    def _eval_command(
        self,
        *,
        round_id: str,
        candidate: CandidatePackageRef,
        checkpoint: CheckpointRef,
        summary_path: str | None = None,
    ) -> str:
        return " ".join(
            self._eval_argv(
                round_id=round_id,
                candidate=candidate,
                checkpoint=checkpoint,
                summary_path=summary_path,
            )
        )

    def _eval_argv(
        self,
        *,
        round_id: str,
        candidate: CandidatePackageRef,
        checkpoint: CheckpointRef,
        summary_path: str | None = None,
    ) -> list[str]:
        summary_path = summary_path or self._summary_path(round_id=round_id, phase="eval")
        return [
            self.config.isaac_lab_python,
            self.config.eval_entrypoint,
            "--round",
            round_id,
            "--candidate",
            candidate.path,
            "--checkpoint",
            checkpoint.uri,
            "--public-seeds",
            str(self.config.public_seed_count),
            "--public-accept-seeds",
            str(self.config.public_accept_seed_count),
            "--summary-path",
            summary_path,
            "--profile",
            self.config.eval_profile,
            "--num-envs",
            str(self.config.eval_num_envs),
            "--headless",
        ]

    def _summary_path(self, *, round_id: str, phase: str) -> str:
        return posixpath.join(self.config.artifact_root.rstrip("/"), round_id, phase, "summary.json")

    def _checkpoint_output_path(self, *, round_id: str) -> str:
        return posixpath.join(self.config.artifact_root.rstrip("/"), round_id, "checkpoints", "best.pt")
