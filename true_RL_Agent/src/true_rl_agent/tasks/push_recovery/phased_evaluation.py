"""Push phase packets built from the same rollouts as the full evaluator."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from true_rl_agent.evaluation.binding import canonical_json_hash, require_sha256_digest
from true_rl_agent.evaluation.phases import (
    EVALUATION_PHASE_SUMMARY_SCHEMA_VERSION,
    EvaluationPhase,
)
from true_rl_agent.tasks.push_recovery.metrics import project_push_split

PUSH_PHASE_EVIDENCE_SCHEMA = "true_rl_agent.task.push_recovery.phase_evidence.v1"


def build_push_phase_summary(
    *,
    phase: EvaluationPhase,
    evaluation_plan_hash: str,
    seed_summaries: Sequence[Mapping[str, Any]],
    expected_seed_count: int,
    wall_time_ns: int,
    identity: Mapping[str, Any],
    execution_device: str,
    checkpoint_safety_report: Mapping[str, Any],
    runtime_dynamics_hash: str,
) -> dict[str, Any]:
    """Keep raw task evidence private; do not pretend one phase is a full eval."""

    require_sha256_digest(evaluation_plan_hash, "evaluation plan hash")
    if len(seed_summaries) != expected_seed_count:
        raise ValueError("Push phase seed count mismatch")
    errors: list[dict[str, Any]] = []
    project_push_split(seed_summaries, split_name=phase.value, errors=errors)
    if errors:
        raise ValueError("Push phase contains incomplete rollout evidence")
    from true_rl_agent.tasks.push_recovery.numeric_capture import validate_push_numeric_reference
    for summary in seed_summaries:
        if "numeric_evidence_ref" in summary:
            if phase is not EvaluationPhase.PUBLIC_DEV:
                raise ValueError("Push numeric archives are public_dev-only")
            expected_identity = {key: identity[key] for key in ("job_id", "task_id", "subject_kind", "subject_id",
                "subject_manifest_hash", "evaluation_binding", "backend_execution_environment_verification") if key in identity}
            expected_identity.update(evaluation_plan_hash=evaluation_plan_hash,
                checkpoint_hash=checkpoint_safety_report["checkpoint_hash"], runtime_dynamics_hash=runtime_dynamics_hash)
            validate_push_numeric_reference(summary["numeric_evidence_ref"], summary, expected_identity=expected_identity)
    costs = [summary["execution_cost"] for summary in seed_summaries]
    env_counts = {cost["evaluation_env_count"] for cost in costs}
    if len(env_counts) != 1:
        raise ValueError("Push phase environment counts differ between seeds")
    env_count = env_counts.pop()
    vector_steps = sum(cost["vector_env_step_calls"] for cost in costs)
    task_evidence = {
        "schema_version": PUSH_PHASE_EVIDENCE_SCHEMA,
        "phase": phase.value,
        "seed_summaries": list(seed_summaries),
    }
    return {
        **identity,
        "schema_version": EVALUATION_PHASE_SUMMARY_SCHEMA_VERSION,
        "evaluation_phase": phase.value,
        "evaluation_plan_hash": evaluation_plan_hash,
        "execution_device": execution_device,
        "status": "pass",
        "errors": [],
        "task_evidence": task_evidence,
        "task_evidence_hash": canonical_json_hash(task_evidence),
        "candidate_code_loaded_in_eval": False,
        "candidate_package_loaded": False,
        "candidate_package_loaded_in_eval": False,
        "checkpoint_safety_report": dict(checkpoint_safety_report),
        # This checks the loaded Push configuration, not a terrain preflight.
        "environment_source_preflight": {
            "scope": "push_runtime_configuration",
            "runtime_dynamics_hash": runtime_dynamics_hash,
        },
        "evaluation_execution_cost": {
            "schema_version": "true_rl_agent.evaluation.execution_cost.v1",
            "clock_source": "time.perf_counter_ns",
            "measurement_scope": "split_rollout_and_summary_aggregation",
            "splits": {
                phase.value: {
                    "wall_time_s": wall_time_ns / 1_000_000_000,
                    "seed_count": expected_seed_count,
                    "evaluation_env_count_per_seed": env_count,
                    "vector_env_step_calls": vector_steps,
                    "simulated_env_control_steps": vector_steps * env_count,
                    "closed_first_episode_control_steps": sum(
                        cost["closed_first_episode_control_steps"] for cost in costs
                    ),
                }
            },
        },
    }
