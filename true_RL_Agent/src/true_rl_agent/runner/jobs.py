"""Dry-run runner job command projections."""

from __future__ import annotations

from true_rl_agent.runner.specs import EvalJobSpec, TrainJobSpec


def dry_run_train_command(spec: TrainJobSpec) -> str:
    parts = [
        "true-rl-runner train",
        f"--job {spec.job_id}",
        f"--task {spec.task_id}",
        f"--candidate {spec.candidate_id}",
        f"--budget-steps {spec.budget_steps}",
        f"--init {spec.policy_initialization.kind}",
    ]
    checkpoint = spec.training_start_checkpoint()
    if checkpoint is not None:
        parts.append(f"--warm-start {checkpoint.uri}")
    return " ".join(parts)


def dry_run_eval_command(spec: EvalJobSpec) -> str:
    return " ".join(
        [
            "true-rl-runner eval",
            f"--job {spec.job_id}",
            f"--task {spec.task_id}",
            f"--candidate {spec.candidate_id}",
            f"--checkpoint {spec.checkpoint.uri}",
            f"--public-seeds {spec.public_seed_count}",
            f"--public-accept-seeds {spec.public_accept_seed_count}",
        ]
    )
