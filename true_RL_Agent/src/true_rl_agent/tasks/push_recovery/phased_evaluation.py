"""Push phase packets built from the same rollouts as the full evaluator."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
import time
from typing import Any

from true_rl_agent.evaluation.binding import canonical_json_hash, require_sha256_digest
from true_rl_agent.json_contract import canonical_json_bytes
from true_rl_agent.fs_contract import atomic_replace_bytes
from true_rl_agent.evaluation.phases import (
    EVALUATION_PHASE_SUMMARY_SCHEMA_VERSION,
    EvaluationPhase,
)
from true_rl_agent.tasks.push_recovery.metrics import project_push_split
from true_rl_agent.runner.specs import EvalPhaseJobSpec

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
    behavior_context: Mapping[str, Any] | None = None,
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
    if behavior_context is not None:
        if phase is not EvaluationPhase.PUBLIC_DEV:
            raise ValueError("Push behavior evidence is public_dev-only")
        from true_rl_agent.tasks.push_recovery.behavior import project_push_behavior
        from true_rl_agent.tasks.push_recovery.process_report import build_push_process_behavior
        for key, value in {"evaluation_plan_hash": evaluation_plan_hash,
                           "checkpoint_hash": checkpoint_safety_report["checkpoint_hash"],
                           "runtime_dynamics_hash": runtime_dynamics_hash,
                           "program_source_checksum": identity["program_source_checksum"]}.items():
            if behavior_context.get(key) != value:
                raise ValueError("Push behavior context differs from the phase identity")
        if all("numeric_evidence_ref" in seed for seed in seed_summaries):
            # The simulator publishes an unfinished handoff. The same runner
            # seals it only after process exit, GPU release and CPU derivation.
            task_evidence["behavior_context"] = dict(behavior_context)
        else:
            task_evidence["behavior"] = project_push_behavior(build_push_process_behavior(seed_summaries, context=behavior_context))
    return {
        **identity,
        "schema_version": EVALUATION_PHASE_SUMMARY_SCHEMA_VERSION,
        "evaluation_phase": phase.value,
        "evaluation_plan_hash": evaluation_plan_hash,
        "execution_device": execution_device,
        "status": "postprocessing" if "behavior_context" in task_evidence else "pass",
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


def finalize_push_eval_phase_summary(
    eval_spec: EvalPhaseJobSpec, summary: Mapping[str, Any], *, output_dir: Path,
    behavior_feedback_profile: str = "legacy_process_v1", trajectory_report_max_bytes: int = 12288,
    expected_seed_count: int | None = None,
) -> dict[str, Any]:
    """Derive one public-dev behavior on CPU before the runner seals it."""
    from true_rl_agent.tasks.push_recovery.behavior import project_push_behavior
    from true_rl_agent.tasks.push_recovery.process_report import build_push_process_behavior
    from true_rl_agent.tasks.push_recovery.numeric_capture import validate_push_numeric_reference

    if summary.get("status") != "postprocessing" or eval_spec.phase is not EvaluationPhase.PUBLIC_DEV:
        raise ValueError("Push behavior finalization requires unfinished public-dev")
    finalized = deepcopy(dict(summary))
    evidence = finalized["task_evidence"]
    context = evidence.pop("behavior_context")
    expected_context = {
        "evaluation_plan_hash": eval_spec.evaluation_plan_hash,
        "checkpoint_hash": eval_spec.execution_checkpoint().checksum,
        "program_source_checksum": eval_spec.program_source_checksum,
        "runtime_dynamics_hash": finalized["environment_source_preflight"]["runtime_dynamics_hash"],
    }
    if any(context.get(key) != value for key, value in expected_context.items()):
        raise ValueError("Push behavior context differs from the phase plan")
    expected_identity = {**eval_spec.job_identity().to_public_dict(), **expected_context,
        "evaluation_binding": eval_spec.evaluation_binding.to_public_dict(),
        "backend_execution_environment_verification": finalized.get("backend_execution_environment_verification")}
    if expected_identity["backend_execution_environment_verification"] is None:
        expected_identity.pop("backend_execution_environment_verification")
    wall_started, cpu_started = time.perf_counter(), time.process_time()
    from .trajectory_pipeline import TRAJECTORY_PROFILE, validate_feedback_profile, build_phase_trajectory
    validate_feedback_profile(behavior_feedback_profile, trajectory_report_max_bytes, context["cell_variant_id"])
    if behavior_feedback_profile == TRAJECTORY_PROFILE:
        # Capture records training lineage; capability is bound by the outer
        # phase/plan receipt, not claimed as a channel in historical archives.
        expected_identity["training_binding_hash"] = eval_spec.training_binding_hash
        if expected_seed_count is None:
            raise ValueError("Push automatic trajectory requires the composition seed count")
        full, evidence["behavior"], evidence["trajectory_build"] = build_phase_trajectory(
            evidence["seed_summaries"], expected_identity=expected_identity, context=context,
            expected_seed_count=expected_seed_count, output_dir=output_dir, max_bytes=trajectory_report_max_bytes)
    else:
        for seed in evidence["seed_summaries"]:
            validate_push_numeric_reference(seed["numeric_evidence_ref"], seed, expected_identity=expected_identity)
        full = build_push_process_behavior(evidence["seed_summaries"], context=context)
        evidence["behavior"] = project_push_behavior(full)
    full_bytes = canonical_json_bytes(full)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_replace_bytes(output_dir / "behavior_full.json", full_bytes)
    evidence["behavior_full"] = {"schema_version": "true_rl_agent.process_full_ref.v1",
        "file": "behavior_full.json", "sha256": canonical_json_hash(full), "size_bytes": len(full_bytes)}
    cost = finalized["evaluation_execution_cost"]["splits"][eval_spec.phase.value]
    cost["postprocessing_wall_time_s"] = time.perf_counter() - wall_started
    cost["postprocessing_cpu_time_s"] = time.process_time() - cpu_started
    cost["full_process_derivation_bytes"] = len(full_bytes)
    cost["public_process_report_bytes"] = len(canonical_json_bytes(evidence["behavior"]))
    finalized["task_evidence_hash"] = canonical_json_hash(evidence)
    finalized["status"] = "pass"
    return finalized
