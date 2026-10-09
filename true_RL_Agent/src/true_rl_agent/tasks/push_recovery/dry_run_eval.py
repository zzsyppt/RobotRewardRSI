"""Push-recovery dry-run eval projection for run audits."""

from __future__ import annotations

from typing import Any, Mapping

from true_rl_agent.evaluation.acceptance_protocol import AcceptanceProtocol
from true_rl_agent.evaluation.public_result import PublicEvalResult
from true_rl_agent.evaluation.result_contract import validate_finite_number
from true_rl_agent.evaluation.sealed_validation import build_sealed_validation_gate
from true_rl_agent.runner.budget import build_budget_accounting
from true_rl_agent.runner.dry_run import DryRunCandidateMetadata
from true_rl_agent.runner.eval_isolation import build_candidate_isolation_gate
from true_rl_agent.runner.specs import EvalJobSpec, TrainJobSpec
from true_rl_agent.tasks.push_recovery.metrics import (
    build_push_public_result_from_summaries,
    build_push_sealed_rollout_evidence,
)


def build_push_dry_run_train_summary(
    train_spec: TrainJobSpec,
    metadata: DryRunCandidateMetadata,
) -> dict[str, Any]:
    components = set(metadata.changed_components)
    code_level_active = bool(components & {"reward_callable", "control_logic", "recovery_logic"})
    diagnostic_active = "diagnostic" in components
    fixed_train_protocol = train_spec.metadata.get("fixed_train_protocol")
    if not isinstance(fixed_train_protocol, Mapping):
        raise ValueError("dry-run train spec requires fixed_train_protocol metadata")
    budget_accounting = build_budget_accounting(
        budget_steps=train_spec.budget_steps,
        num_envs=fixed_train_protocol.get("train_num_envs"),
        num_steps_per_env=fixed_train_protocol.get(
            "train_steps_per_iteration"
        ),
        max_iterations=fixed_train_protocol.get("train_max_iterations"),
    )
    return {
        "dry_run": True,
        "formal_training_executed": False,
        "budget_steps": train_spec.budget_steps,
        "training": {
            "budget_accounting": budget_accounting,
        },
        "runtime_activation_summary": {
            "reward_term_calls": 4 if code_level_active else 0,
            "diagnostic_calls": 1 if diagnostic_active else 0,
            "nonzero_reward_contribution": code_level_active,
            "reward_output_stats": (
                {"mean": 0.4, "finite_fraction": 1.0} if code_level_active else {}
            ),
            "diagnostic_outputs": {},
            "feature_activation_records": [],
            "capability_activation_records": sorted(components),
        },
    }


def build_push_dry_run_public_result(
    eval_spec: EvalJobSpec,
    metadata: DryRunCandidateMetadata,
    *,
    acceptance_protocol: AcceptanceProtocol,
) -> PublicEvalResult:
    score = validate_finite_number(
        metadata.public_score,
        field_name="dry-run public_score",
    )
    sealed_summaries = [
        _rollout_summary(
            score=score,
            failure_breakdown=metadata.failure_breakdown,
            motion_quality=metadata.task_eval_metadata,
            run_index=index,
        )
        for index in range(1, int(eval_spec.sealed_validation_seed_count) + 1)
    ]
    result = build_push_public_result_from_summaries(
        public_dev_summaries=[
            _rollout_summary(
                score=score,
                failure_breakdown=metadata.failure_breakdown,
                motion_quality=metadata.task_eval_metadata,
                run_index=index,
            )
            for index in range(1, eval_spec.public_seed_count + 1)
        ],
        public_accept_summaries=[
            _rollout_summary(
                score=score,
                failure_breakdown=metadata.failure_breakdown,
                motion_quality=metadata.task_eval_metadata,
                run_index=index,
            )
            for index in range(1, eval_spec.public_accept_seed_count + 1)
        ],
        budget_summary=None,
        min_public_accept_count=eval_spec.public_accept_seed_count,
        acceptance_protocol=acceptance_protocol,
        evaluation_protocol=eval_spec.evaluation_protocol(),
    )
    metrics = dict(result.metrics)
    evaluation_protocol = eval_spec.evaluation_protocol()
    sealed_evidence = build_push_sealed_rollout_evidence(
        summaries=sealed_summaries,
        evaluation_protocol=evaluation_protocol,
        checkpoint_hash=eval_spec.execution_checkpoint().checksum,
        evaluation_binding_hash=eval_spec.evaluation_binding.binding_hash,
        evaluation_subject=eval_spec.evaluation_subject(),
        subject_manifest_hash=eval_spec.evaluation_manifest_hash(),
        program_source_checksum=(
            eval_spec.program_source_checksum
            if isinstance(eval_spec, EvalJobSpec)
            else None
        ),
        training_binding_hash=(
            eval_spec.training_binding_hash
            if isinstance(eval_spec, EvalJobSpec)
            else None
        ),
        capability_profile_hash=eval_spec.capability_profile_hash,
        acceptance_protocol=acceptance_protocol,
    )
    metrics["sealed_validation"] = sealed_evidence
    metrics["sealed_validation_gate"] = build_sealed_validation_gate(
        rollout_evidence=sealed_evidence,
        checkpoint_safety_status="pass",
        checkpoint_hash_match_status="pass",
        candidate_isolation_gate=build_candidate_isolation_gate(),
        expected_eval_protocol_hash=evaluation_protocol.protocol_hash,
        expected_evaluation_binding_hash=eval_spec.evaluation_binding.binding_hash,
    )
    acceptance = dict(metrics.get("acceptance", {}))
    gates = list(acceptance.get("required_pass_gates", []))
    if "sealed_validation_gate" not in gates:
        gates.append("sealed_validation_gate")
    acceptance["required_pass_gates"] = gates
    acceptance["missing_required_score"] = False
    metrics["acceptance"] = acceptance
    return PublicEvalResult(status=result.status, public_score=result.public_score, metrics=metrics)


def _rollout_summary(
    *,
    score: float,
    failure_breakdown: Mapping[str, float],
    motion_quality: Mapping[str, Any],
    run_index: int,
) -> dict[str, Any]:
    return {
        "status": "pass",
        "aggregate": {
            "success_rate": float(score),
            "episodes": 8,
            "failure_breakdown": dict(failure_breakdown),
        },
        "secondary_metrics": _secondary_metrics(motion_quality),
        "run_index": int(run_index),
    }


def _secondary_metrics(motion_quality: Mapping[str, Any]) -> dict[str, float]:
    base_height = _float_value(
        motion_quality,
        "mean_min_base_height_m",
        fallback_key="mean_base_height",
        default=0.78,
    )
    tilt = _float_value(motion_quality, "mean_tilt_rad", default=0.05)
    return {
        "mean_episode_steps": _float_value(motion_quality, "mean_episode_steps", default=512.0),
        "mean_return": _float_value(motion_quality, "mean_return", default=1.0),
        "mean_action_l2": _float_value(motion_quality, "mean_action_l2", default=0.4),
        "mean_action_rate_l2": _float_value(motion_quality, "mean_action_rate_l2", default=0.08),
        "mean_min_base_height_m": base_height,
        "mean_max_abs_pitch_rad": _float_value(
            motion_quality,
            "mean_max_abs_pitch_rad",
            default=tilt,
        ),
        "mean_max_abs_roll_rad": _float_value(
            motion_quality,
            "mean_max_abs_roll_rad",
            default=tilt,
        ),
        "mean_final_xy_drift_m": _float_value(
            motion_quality,
            "mean_final_xy_drift_m",
            default=0.18,
        ),
        "mean_final_planar_speed_mps": _float_value(
            motion_quality,
            "mean_final_planar_speed_mps",
            default=0.12,
        ),
        "recovery_window_success_rate": _float_value(
            motion_quality,
            "recovery_window_success_rate",
            default=0.75,
        ),
    }


def _float_value(
    values: Mapping[str, Any],
    key: str,
    *,
    fallback_key: str | None = None,
    default: float,
) -> float:
    raw = values.get(key)
    if not isinstance(raw, int | float) and fallback_key is not None:
        raw = values.get(fallback_key)
    if isinstance(raw, int | float) and not isinstance(raw, bool):
        return float(raw)
    return float(default)
