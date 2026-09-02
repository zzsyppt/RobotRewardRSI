"""Dry-run runner job command projections."""

from __future__ import annotations

from true_rl_agent.runner.specs import EvaluationJobSpec, TrainJobSpec


def dry_run_train_command(spec: TrainJobSpec) -> str:
    parts = [
        "true-rl-runner train",
        f"--job {spec.job_id}",
        f"--task {spec.task_id}",
        f"--subject-kind {spec.subject.kind}",
        f"--subject-id {spec.subject.subject_id}",
        f"--budget-steps {spec.budget_steps}",
        f"--init {spec.policy_initialization.kind}",
        f"--device {spec.execution_device}",
        f"--training-binding-hash {spec.training_binding.binding_hash}",
        f"--capability-profile-hash {spec.training_binding.capability_profile_hash}",
    ]
    checkpoint = spec.training_start_checkpoint()
    if checkpoint is not None:
        parts.append(f"--warm-start {checkpoint.uri}")
    return " ".join(parts)


def dry_run_eval_command(spec: EvaluationJobSpec) -> str:
    subject = spec.evaluation_subject()
    binding = spec.evaluation_binding
    checkpoint = spec.execution_checkpoint()
    parts = [
        "true-rl-runner eval",
        f"--job {spec.job_id}",
        f"--task {spec.task_id}",
        f"--subject-kind {subject.kind}",
        f"--subject-id {subject.subject_id}",
        f"--checkpoint {checkpoint.uri}",
        f"--checkpoint-sha256 {checkpoint.checksum}",
        f"--public-seeds {spec.public_seed_count}",
        f"--public-accept-seeds {spec.public_accept_seed_count}",
        f"--sealed-validation-seeds {spec.sealed_validation_seed_count}",
        f"--num-envs {spec.eval_num_envs}",
        f"--device {spec.execution_device}",
        f"--eval-protocol-version {spec.eval_protocol_version}",
        f"--task-protocol-hash {binding.task_protocol_hash}",
        f"--acceptance-protocol-hash {binding.acceptance_protocol_hash}",
        f"--seed-cohort-hash {binding.seed_cohort_hash}",
        f"--evaluation-binding-hash {binding.binding_hash}",
    ]
    subject_manifest_hash = spec.evaluation_manifest_hash()
    if subject_manifest_hash is not None:
        parts.append(f"--subject-manifest-hash {subject_manifest_hash}")
    return " ".join(parts)
