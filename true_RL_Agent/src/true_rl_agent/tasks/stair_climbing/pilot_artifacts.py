"""Strict Phase D pilot summaries and public-safe selection evidence."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from true_rl_agent.evaluation.binding import (
    canonical_json_hash,
    require_sha256_checksum,
    require_sha256_digest,
)
from true_rl_agent.json_contract import (
    canonical_json_bytes,
    load_strict_json_object,
)
from true_rl_agent.runner.budget import build_budget_accounting
from true_rl_agent.runner.checkpoint import (
    file_sha256,
    validate_checkpoint_safety_report,
)
from true_rl_agent.tasks.stair_climbing.config import StairClimbingTaskConfig
from true_rl_agent.tasks.stair_climbing.episode_evidence import StairEpisodeOutcome
from true_rl_agent.tasks.stair_climbing.isaaclab_execution import (
    StairTrainingExecutionResult,
)
from true_rl_agent.tasks.stair_climbing.local_paths import (
    resolve_existing_directory,
    resolve_existing_file,
)
from true_rl_agent.tasks.stair_climbing.public_projection import (
    build_stair_runtime_activation_gate,
)
from true_rl_agent.tasks.stair_climbing.reference_pilot import (
    assert_stair_pilot_formal_bindings_disjoint,
    build_stair_pilot_config,
    build_stair_pilot_evaluation_binding,
    build_stair_pilot_training_binding,
    build_stair_reference_capability_evidence,
    stair_pilot_training_seed_domain_hash,
    stair_pilot_training_seed_index,
)
from true_rl_agent.tasks.stair_climbing.reference_program import (
    StairOfficialReferenceProgram,
)
from true_rl_agent.tasks.stair_climbing.runtime_profile import (
    STAIR_COMPLETE_REWARD_RUNTIME_STATUS,
)
from true_rl_agent.tasks.stair_climbing.seed_policy import (
    STAIR_PILOT_EVAL_NAMESPACE,
    STAIR_PILOT_TRAIN_NAMESPACE,
)
from true_rl_agent.tasks.stair_climbing.termination_capture import (
    stair_termination_manager_resolution_contract,
)
from true_rl_agent.tasks.stair_climbing.training_distribution import (
    resolved_stair_training_distribution,
    stair_training_distribution_upstream_source_evidence,
)

STAIR_PILOT_TRAIN_SUMMARY_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.phase_d_pilot_train.v3"
)
STAIR_PILOT_EVAL_SUMMARY_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.phase_d_pilot_eval.v3"
)
STAIR_PILOT_SEED_EVIDENCE_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.phase_d_pilot_seed.v3"
)
STAIR_PHASE_D_REPORT_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.phase_d_report.v5"
)
STAIR_PILOT_CURVE_SUMMARY_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.phase_d_curve_diagnostics.v1"
)
STAIR_PHASE_D_SMOKE_REPORT_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.phase_d_execution_smoke.v2"
)

STAIR_PILOT_REQUIRED_SCALARS = (
    "Loss/value_function",
    "Loss/surrogate",
    "Policy/mean_noise_std",
    "Loss/learning_rate",
    "Perf/total_fps",
    "Train/mean_reward",
    "Curriculum/terrain_levels",
    "Train/mean_episode_length",
    "Episode_Termination/base_contact",
)

_BOUNDARIES = {
    "enters_genesis": False,
    "enters_acceptance": False,
    "enters_production_prompt": False,
}


def summarize_stair_pilot_tensorboard(
    log_dir: Path,
    *,
    max_iterations: int,
    diagnostics_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Summarize complete curves without making a budget decision."""

    log_dir = resolve_existing_directory(
        log_dir,
        field_name="stair pilot TensorBoard log_dir",
    )
    _positive_int(max_iterations, "stair pilot max_iterations")
    diagnostics = _validate_curve_diagnostics_contract(
        diagnostics_contract,
        max_iterations=max_iterations,
    )
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
    )

    event_files = sorted(log_dir.rglob("events.out.tfevents.*"))
    if not event_files:
        raise ValueError("stair pilot produced no TensorBoard event file")
    accumulator = EventAccumulator(str(log_dir))
    accumulator.Reload()
    available = tuple(sorted(accumulator.Tags().get("scalars", ())))
    missing = sorted(set(STAIR_PILOT_REQUIRED_SCALARS) - set(available))
    if missing:
        raise ValueError(
            "stair pilot TensorBoard required scalar missing: " + ",".join(missing)
        )
    expected_last_step = max_iterations - 1
    scalars: dict[str, Any] = {}
    for tag in STAIR_PILOT_REQUIRED_SCALARS:
        events = tuple(accumulator.Scalars(tag))
        if not events:
            raise ValueError(f"stair pilot TensorBoard scalar is empty: {tag}")
        steps = tuple(int(event.step) for event in events)
        values = tuple(float(event.value) for event in events)
        nonfinite_count = sum(not math.isfinite(value) for value in values)
        if steps != tuple(range(max_iterations)):
            raise ValueError(
                "stair pilot TensorBoard scalar does not cover every iteration: "
                f"{tag}"
            )
        if nonfinite_count:
            raise ValueError(f"stair pilot TensorBoard scalar is non-finite: {tag}")
        tail = _curve_tail_statistics(
            steps,
            values,
            window_iterations=diagnostics["tail_window_iterations"],
        )
        scalars[tag] = {
            "count": len(events),
            "first_step": steps[0],
            "last_step": steps[-1],
            "first": values[0],
            "last": values[-1],
            "minimum": min(values),
            "maximum": max(values),
            "nonfinite_count": nonfinite_count,
            "tail": tail,
        }
    tail_metrics = {
        tag: {
            "normalized_abs_tail_change": scalars[tag]["tail"][
                "normalized_abs_tail_change"
            ]
        }
        for tag in diagnostics["required_scalar_tags"]
    }
    return {
        "schema_version": STAIR_PILOT_CURVE_SUMMARY_SCHEMA_VERSION,
        "event_file_count": len(event_files),
        "required_scalar_tags": list(STAIR_PILOT_REQUIRED_SCALARS),
        "available_scalar_tags": list(available),
        "expected_last_iteration": expected_last_step,
        "diagnostics_contract": dict(diagnostics),
        "scalars": scalars,
        "tail_metrics": tail_metrics,
        "decision_authority": False,
        "status": "pass",
    }


def build_stair_pilot_train_summary(
    config: StairClimbingTaskConfig,
    *,
    official: StairOfficialReferenceProgram,
    attempt_id: str,
    execution: StairTrainingExecutionResult,
    curve_summary: Mapping[str, Any] | None,
) -> dict[str, Any]:
    _canonical_text(attempt_id, "stair pilot attempt_id")
    binding = build_stair_pilot_training_binding(config)
    training_seed_index = stair_pilot_training_seed_index(config)
    payload = {
        "schema_version": STAIR_PILOT_TRAIN_SUMMARY_SCHEMA_VERSION,
        "status": execution.status,
        "attempt_id": attempt_id,
        "training_seed_index": training_seed_index,
        "draft_protocol_hash": config.protocol_v2.protocol_hash,
        "base_protocol_hash": config.protocol_v1.protocol_hash,
        "budget_steps": config.budget_steps,
        "max_iterations": config.train_max_iterations,
        "initialization": {
            "mode": "random",
            "checkpoint_input": None,
            "seed_namespace": STAIR_PILOT_TRAIN_NAMESPACE,
            "seed_domain_hash": stair_pilot_training_seed_domain_hash(config),
        },
        "training_binding": binding.to_public_dict(),
        "reference_program": official.artifact.to_public_dict(),
        "reward_program": official.reward_program.to_public_dict(),
        "execution": _portable_stair_pilot_execution(
            execution.to_summary_fields()
        ),
        "curve_summary": dict(curve_summary) if curve_summary is not None else None,
        "boundaries": dict(_BOUNDARIES),
    }
    payload["summary_hash"] = canonical_json_hash(payload)
    return payload


def validate_stair_pilot_train_summary(
    config: StairClimbingTaskConfig,
    *,
    official: StairOfficialReferenceProgram,
    summary: Mapping[str, Any],
    checkpoint_path: Path,
    expected_attempt_id: str | None = None,
) -> dict[str, Any]:
    fields = {
        "schema_version",
        "status",
        "attempt_id",
        "training_seed_index",
        "draft_protocol_hash",
        "base_protocol_hash",
        "budget_steps",
        "max_iterations",
        "initialization",
        "training_binding",
        "reference_program",
        "reward_program",
        "execution",
        "curve_summary",
        "boundaries",
        "summary_hash",
    }
    normalized = _strict_fields(summary, fields, "stair pilot train summary")
    if normalized["schema_version"] != STAIR_PILOT_TRAIN_SUMMARY_SCHEMA_VERSION:
        raise ValueError("stair pilot train summary schema mismatch")
    if normalized["status"] != "pass":
        raise ValueError("stair pilot training did not pass")
    attempt_id = _canonical_text(normalized["attempt_id"], "stair pilot attempt_id")
    if expected_attempt_id is not None and attempt_id != expected_attempt_id:
        raise ValueError("stair pilot train attempt_id mismatch")
    if normalized["training_seed_index"] != stair_pilot_training_seed_index(config):
        raise ValueError("stair pilot train training_seed_index mismatch")
    _assert_config_identity(config, normalized)
    if normalized["training_binding"] != build_stair_pilot_training_binding(
        config
    ).to_public_dict():
        raise ValueError("stair pilot training binding mismatch")
    if normalized["reference_program"] != official.artifact.to_public_dict():
        raise ValueError("stair pilot reference artifact mismatch")
    if normalized["reward_program"] != official.reward_program.to_public_dict():
        raise ValueError("stair pilot reward qualification mismatch")
    if normalized["boundaries"] != _BOUNDARIES:
        raise ValueError("stair pilot train boundary flags mismatch")
    initialization = _mapping(normalized["initialization"], "initialization")
    expected_initialization = {
        "mode": "random",
        "checkpoint_input": None,
        "seed_namespace": STAIR_PILOT_TRAIN_NAMESPACE,
        "seed_domain_hash": stair_pilot_training_seed_domain_hash(config),
    }
    if dict(initialization) != expected_initialization:
        raise ValueError("stair pilot initialization evidence mismatch")
    _validate_training_execution(
        config,
        normalized["execution"],
        checkpoint_path=checkpoint_path,
        expected_reward_program=official.reward_program.to_public_dict(),
        expected_workload_kind="production",
        expected_num_envs=config.train_num_envs,
        expected_num_steps_per_env=config.train_steps_per_iteration,
        expected_max_iterations=config.train_max_iterations,
        expected_budget_steps=config.budget_steps,
        portable_paths=True,
    )
    _validate_curve_summary(
        normalized["curve_summary"],
        max_iterations=config.train_max_iterations,
        diagnostics_contract=config.protocol_v2.pilot_training_contract[
            "curve_diagnostics"
        ],
    )
    _validate_declared_hash(normalized, "summary_hash", "stair pilot train")
    return normalized


def build_stair_pilot_eval_summary(
    config: StairClimbingTaskConfig,
    *,
    attempt_id: str,
    checkpoint_evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    _canonical_text(attempt_id, "stair pilot attempt_id")
    evaluation = dict(checkpoint_evaluation)
    payload = {
        "schema_version": STAIR_PILOT_EVAL_SUMMARY_SCHEMA_VERSION,
        "status": "pass",
        "attempt_id": attempt_id,
        "training_seed_index": stair_pilot_training_seed_index(config),
        "draft_protocol_hash": config.protocol_v2.protocol_hash,
        "base_protocol_hash": config.protocol_v1.protocol_hash,
        "budget_steps": config.budget_steps,
        "max_iterations": config.train_max_iterations,
        "training_binding": build_stair_pilot_training_binding(
            config
        ).to_public_dict(),
        "evaluation_binding": build_stair_pilot_evaluation_binding(
            config
        ).to_public_dict(),
        "checkpoint_evaluation": evaluation,
        "evaluation": {
            "split": "public_accept",
            "seed_namespace": STAIR_PILOT_EVAL_NAMESPACE,
            "candidate_or_reference_source_loaded": False,
        },
        "boundaries": dict(_BOUNDARIES),
    }
    payload["summary_hash"] = canonical_json_hash(payload)
    return validate_stair_pilot_eval_summary(
        config,
        summary=payload,
        expected_checkpoint_hash=evaluation["checkpoint_hash"],
        expected_attempt_id=attempt_id,
    )


def validate_stair_pilot_eval_summary(
    config: StairClimbingTaskConfig,
    *,
    summary: Mapping[str, Any],
    expected_checkpoint_hash: str,
    expected_attempt_id: str | None = None,
) -> dict[str, Any]:
    fields = {
        "schema_version",
        "status",
        "attempt_id",
        "training_seed_index",
        "draft_protocol_hash",
        "base_protocol_hash",
        "budget_steps",
        "max_iterations",
        "training_binding",
        "evaluation_binding",
        "checkpoint_evaluation",
        "evaluation",
        "boundaries",
        "summary_hash",
    }
    normalized = _strict_fields(summary, fields, "stair pilot eval summary")
    if normalized["schema_version"] != STAIR_PILOT_EVAL_SUMMARY_SCHEMA_VERSION:
        raise ValueError("stair pilot eval summary schema mismatch")
    if normalized["status"] != "pass":
        raise ValueError("stair pilot evaluation did not pass")
    attempt_id = _canonical_text(normalized["attempt_id"], "stair pilot attempt_id")
    if expected_attempt_id is not None and attempt_id != expected_attempt_id:
        raise ValueError("stair pilot eval attempt_id mismatch")
    if normalized["training_seed_index"] != stair_pilot_training_seed_index(config):
        raise ValueError("stair pilot eval training_seed_index mismatch")
    _assert_config_identity(config, normalized)
    if normalized["training_binding"] != build_stair_pilot_training_binding(
        config
    ).to_public_dict():
        raise ValueError("stair pilot eval training binding mismatch")
    if normalized["evaluation_binding"] != build_stair_pilot_evaluation_binding(
        config
    ).to_public_dict():
        raise ValueError("stair pilot evaluation binding mismatch")
    checkpoint_hash = require_sha256_checksum(
        expected_checkpoint_hash,
        "expected stair pilot checkpoint hash",
    )
    evaluation = _validate_checkpoint_evaluation(
        config,
        normalized["checkpoint_evaluation"],
        expected_iteration=config.train_max_iterations - 1,
        expected_checkpoint_hash=checkpoint_hash,
    )
    expected_evaluation = {
        "split": "public_accept",
        "seed_namespace": STAIR_PILOT_EVAL_NAMESPACE,
        "candidate_or_reference_source_loaded": False,
    }
    if normalized["evaluation"] != expected_evaluation:
        raise ValueError("stair pilot evaluation execution boundary mismatch")
    if normalized["boundaries"] != _BOUNDARIES:
        raise ValueError("stair pilot eval boundary flags mismatch")
    _validate_declared_hash(normalized, "summary_hash", "stair pilot eval")
    normalized["checkpoint_evaluation"] = evaluation
    return normalized


def _validate_checkpoint_evaluation(
    config: StairClimbingTaskConfig,
    value: Any,
    *,
    expected_iteration: int,
    expected_checkpoint_hash: str,
) -> dict[str, Any]:
    fields = {
        "iteration",
        "checkpoint_hash",
        "checkpoint_safety_report",
        "per_seed_outcomes",
        "capability_evidence",
    }
    normalized = _strict_fields(value, fields, "stair checkpoint evaluation")
    if normalized["iteration"] != expected_iteration:
        raise ValueError("stair pilot evaluated checkpoint iteration mismatch")
    checkpoint_hash = require_sha256_checksum(
        normalized["checkpoint_hash"],
        "stair pilot evaluated checkpoint hash",
    )
    if checkpoint_hash != expected_checkpoint_hash:
        raise ValueError("stair pilot evaluated checkpoint hash mismatch")
    validate_checkpoint_safety_report(
        normalized["checkpoint_safety_report"],
        expected_checkpoint_hash=checkpoint_hash,
    )
    outcomes, serialized_outcomes = _validate_pilot_raw_outcomes(
        config,
        normalized["per_seed_outcomes"],
    )
    recomputed_capability = build_stair_reference_capability_evidence(
        config.protocol_v1,
        protocol_v2=config.protocol_v2,
        evaluation_protocol=config.evaluation_protocol(),
        per_seed_outcomes=outcomes,
    )
    if normalized["capability_evidence"] != recomputed_capability:
        raise ValueError("stair pilot capability evidence differs from raw outcomes")
    normalized["per_seed_outcomes"] = serialized_outcomes
    normalized["capability_evidence"] = recomputed_capability
    return normalized


def serialize_stair_pilot_raw_outcomes(
    per_seed_outcomes: Sequence[Sequence[StairEpisodeOutcome]],
) -> list[list[dict[str, Any]]]:
    return [
        [
            {
                "env_index": outcome.env_index,
                "bin_id": outcome.bin_id,
                "episode_steps": outcome.episode_steps,
                "primary_reason": outcome.primary_reason,
                "success": outcome.success,
                "diagnostic_flags": list(outcome.diagnostic_flags),
                "max_progress_m": outcome.max_progress_m,
                "progress_fraction": outcome.progress_fraction,
                "active_terminal_terms": list(outcome.active_terminal_terms),
            }
            for outcome in seed_outcomes
        ]
        for seed_outcomes in per_seed_outcomes
    ]


def _validate_pilot_raw_outcomes(
    config: StairClimbingTaskConfig,
    value: Any,
) -> tuple[tuple[tuple[StairEpisodeOutcome, ...], ...], list[list[dict[str, Any]]]]:
    if not isinstance(value, list) or len(value) != config.public_accept_seed_count:
        raise ValueError("stair pilot raw outcome seed count mismatch")
    outcome_fields = {
        "env_index",
        "bin_id",
        "episode_steps",
        "primary_reason",
        "success",
        "diagnostic_flags",
        "max_progress_m",
        "progress_fraction",
        "active_terminal_terms",
    }
    per_seed_outcomes = []
    for seed_index, raw_seed in enumerate(value):
        if not isinstance(raw_seed, list) or len(raw_seed) != config.eval_num_envs:
            raise ValueError(
                f"stair pilot raw outcome count mismatch for seed {seed_index}"
            )
        outcomes = []
        for env_index, raw in enumerate(raw_seed):
            item = _strict_fields(raw, outcome_fields, "stair pilot raw outcome")
            if item["env_index"] != env_index:
                raise ValueError("stair pilot raw outcome env ordering mismatch")
            diagnostic_flags = item["diagnostic_flags"]
            active_terminal_terms = item["active_terminal_terms"]
            if not isinstance(diagnostic_flags, list) or not isinstance(
                active_terminal_terms,
                list,
            ):
                raise ValueError("stair pilot raw outcome tuple fields mismatch")
            outcomes.append(
                StairEpisodeOutcome(
                    env_index=item["env_index"],
                    bin_id=item["bin_id"],
                    episode_steps=item["episode_steps"],
                    primary_reason=item["primary_reason"],
                    success=item["success"],
                    diagnostic_flags=tuple(diagnostic_flags),
                    max_progress_m=item["max_progress_m"],
                    progress_fraction=item["progress_fraction"],
                    active_terminal_terms=tuple(active_terminal_terms),
                )
            )
        per_seed_outcomes.append(tuple(outcomes))
    normalized = tuple(per_seed_outcomes)
    return normalized, serialize_stair_pilot_raw_outcomes(normalized)


def build_stair_phase_d_smoke_report(
    config: StairClimbingTaskConfig,
    *,
    official: StairOfficialReferenceProgram,
    execution: StairTrainingExecutionResult,
    checkpoint_path: Path,
) -> dict[str, Any]:
    fields = execution.to_summary_fields()
    _validate_training_execution(
        config,
        fields,
        checkpoint_path=checkpoint_path,
        expected_reward_program=official.reward_program.to_public_dict(),
        expected_workload_kind="phase_d_smoke",
        expected_num_envs=16,
        expected_num_steps_per_env=8,
        expected_max_iterations=1,
        expected_budget_steps=128,
        portable_paths=False,
    )
    checkpoint = _mapping(fields["checkpoint"], "smoke checkpoint")
    training = _mapping(fields["training"], "smoke training")
    payload = {
        "schema_version": STAIR_PHASE_D_SMOKE_REPORT_SCHEMA_VERSION,
        "status": "pass",
        "draft_protocol_hash": config.protocol_v2.protocol_hash,
        "base_protocol_hash": config.protocol_v1.protocol_hash,
        "reference_program": official.artifact.to_public_dict(),
        "reward_program": official.reward_program.to_public_dict(),
        "pilot_training_binding": build_stair_pilot_training_binding(
            config
        ).to_public_dict(),
        "initialization": {
            "mode": "random",
            "checkpoint_input": None,
            "seed_namespace": STAIR_PILOT_TRAIN_NAMESPACE,
            "seed_domain_hash": stair_pilot_training_seed_domain_hash(config),
        },
        "execution_core": (
            "true_rl_agent.tasks.stair_climbing.isaaclab_execution."
            "execute_stair_training_smoke"
        ),
        "workload": {
            "kind": "phase_d_smoke",
            "num_envs": 16,
            "num_steps_per_env": 8,
            "max_iterations": 1,
            "budget_steps": 128,
        },
        "execution_evidence": {
            "checkpoint_hash": checkpoint["checksum"],
            "latest_checkpoint_iteration": 0,
            "initial_policy_state_hash": training["initial_policy_state_hash"],
            "budget_accounting": training["budget_accounting"],
            "training_distribution": training["training_distribution"],
            "runner_execution": _project_stair_runner_execution(
                config,
                training["effective_agent_execution"],
            ),
            "runtime_authority": _project_runtime_authority(
                fields["runtime_activation_summary"]
            ),
        },
        "eligibility": {
            "pilot_budget_evidence": False,
            "capability_gate_evidence": False,
            **_BOUNDARIES,
        },
    }
    payload["report_hash"] = canonical_json_hash(payload)
    return validate_stair_phase_d_smoke_report(
        config,
        official=official,
        report=payload,
    )


def validate_stair_phase_d_smoke_report(
    config: StairClimbingTaskConfig,
    *,
    official: StairOfficialReferenceProgram,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    fields = {
        "schema_version",
        "status",
        "draft_protocol_hash",
        "base_protocol_hash",
        "reference_program",
        "reward_program",
        "pilot_training_binding",
        "initialization",
        "execution_core",
        "workload",
        "execution_evidence",
        "eligibility",
        "report_hash",
    }
    normalized = _strict_fields(report, fields, "stair Phase D smoke report")
    if normalized["schema_version"] != STAIR_PHASE_D_SMOKE_REPORT_SCHEMA_VERSION:
        raise ValueError("stair Phase D smoke report schema mismatch")
    if normalized["status"] != "pass":
        raise ValueError("stair Phase D smoke did not pass")
    _assert_config_identity(config, normalized, include_budget=False)
    if normalized["reference_program"] != official.artifact.to_public_dict():
        raise ValueError("stair Phase D smoke reference artifact mismatch")
    if normalized["reward_program"] != official.reward_program.to_public_dict():
        raise ValueError("stair Phase D smoke reward program mismatch")
    if normalized["pilot_training_binding"] != build_stair_pilot_training_binding(
        config
    ).to_public_dict():
        raise ValueError("stair Phase D smoke training binding mismatch")
    expected_initialization = {
        "mode": "random",
        "checkpoint_input": None,
        "seed_namespace": STAIR_PILOT_TRAIN_NAMESPACE,
        "seed_domain_hash": stair_pilot_training_seed_domain_hash(config),
    }
    if normalized["initialization"] != expected_initialization:
        raise ValueError("stair Phase D smoke initialization mismatch")
    if normalized["execution_core"] != (
        "true_rl_agent.tasks.stair_climbing.isaaclab_execution."
        "execute_stair_training_smoke"
    ):
        raise ValueError("stair Phase D smoke execution core mismatch")
    expected_workload = {
        "kind": "phase_d_smoke",
        "num_envs": 16,
        "num_steps_per_env": 8,
        "max_iterations": 1,
        "budget_steps": 128,
    }
    if normalized["workload"] != expected_workload:
        raise ValueError("stair Phase D smoke workload mismatch")
    evidence = _mapping(normalized["execution_evidence"], "execution_evidence")
    if set(evidence) != {
        "checkpoint_hash",
        "latest_checkpoint_iteration",
        "initial_policy_state_hash",
        "budget_accounting",
        "training_distribution",
        "runner_execution",
        "runtime_authority",
    }:
        raise ValueError("stair Phase D smoke execution evidence fields mismatch")
    require_sha256_checksum(
        evidence.get("checkpoint_hash"),
        "stair Phase D smoke checkpoint hash",
    )
    if evidence.get("latest_checkpoint_iteration") != 0:
        raise ValueError("stair Phase D smoke checkpoint iteration mismatch")
    require_sha256_digest(
        evidence.get("initial_policy_state_hash"),
        "stair Phase D smoke initial policy state hash",
    )
    expected_budget = {
        "status": "pass",
        "budget_steps": 128,
        "num_envs": 16,
        "num_steps_per_env": 8,
        "max_iterations": 1,
        "planned_env_steps": 128,
        "config_derived_total_timesteps": 128,
    }
    if evidence.get("budget_accounting") != expected_budget:
        raise ValueError("stair Phase D smoke budget evidence mismatch")
    _validate_training_distribution_evidence(
        config,
        evidence.get("training_distribution"),
    )
    _validate_stair_runner_execution_projection(
        config,
        evidence.get("runner_execution"),
    )
    _validate_runtime_authority_projection(
        config,
        evidence.get("runtime_authority"),
    )
    expected_eligibility = {
        "pilot_budget_evidence": False,
        "capability_gate_evidence": False,
        **_BOUNDARIES,
    }
    if normalized["eligibility"] != expected_eligibility:
        raise ValueError("stair Phase D smoke eligibility boundary mismatch")
    _validate_declared_hash(normalized, "report_hash", "stair Phase D smoke")
    return normalized


def build_stair_pilot_seed_evidence(
    config: StairClimbingTaskConfig,
    *,
    train_summary: Mapping[str, Any],
    eval_summary: Mapping[str, Any],
    train_process: Mapping[str, Any],
    eval_process: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint_hash = require_sha256_checksum(
        _mapping(train_summary["execution"], "execution")["checkpoint"]["checksum"],
        "stair pilot checkpoint hash",
    )
    curve_summary = _mapping(train_summary["curve_summary"], "curve_summary")
    checkpoint_evaluation = _mapping(
        eval_summary["checkpoint_evaluation"],
        "checkpoint_evaluation",
    )
    capability_status = _mapping(
        checkpoint_evaluation["capability_evidence"],
        "capability_evidence",
    )["gate"]["status"]
    if capability_status not in {"pass", "fail"}:
        raise ValueError("stair pilot capability gate status is invalid")
    failed_gates = [] if capability_status == "pass" else ["capability"]
    payload = {
        "schema_version": STAIR_PILOT_SEED_EVIDENCE_SCHEMA_VERSION,
        "status": capability_status,
        "failed_gates": failed_gates,
        "attempt_id": train_summary["attempt_id"],
        "training_seed_index": stair_pilot_training_seed_index(config),
        "budget_steps": config.budget_steps,
        "max_iterations": config.train_max_iterations,
        "train_summary_hash": train_summary["summary_hash"],
        "eval_summary_hash": eval_summary["summary_hash"],
        "checkpoint_hash": checkpoint_hash,
        "training_binding": train_summary["training_binding"],
        "evaluation_binding": build_stair_pilot_evaluation_binding(
            config
        ).to_public_dict(),
        "gate_statuses": {"capability": capability_status},
        "training": {
            "runtime_authority": _project_runtime_authority(
                _mapping(train_summary["execution"], "execution")[
                    "runtime_activation_summary"
                ]
            ),
            "budget_accounting": _mapping(
                _mapping(train_summary["execution"], "execution")["training"],
                "execution.training",
            )["budget_accounting"],
            "latest_checkpoint_iteration": config.train_max_iterations - 1,
            "initial_policy_state_hash": _mapping(
                train_summary["execution"], "execution"
            )["training"]["initial_policy_state_hash"],
            "expected_initial_policy_state_hash": _mapping(
                train_summary["execution"], "execution"
            )["training"]["expected_initial_policy_state_hash"],
            "training_distribution": _mapping(
                train_summary["execution"], "execution"
            )["training"]["training_distribution"],
            "curve_summary": curve_summary,
        },
        "evaluation": {
            "status": "completed",
            "checkpoint_evaluation": checkpoint_evaluation,
        },
        "processes": {
            "train": dict(train_process),
            "eval": dict(eval_process),
        },
    }
    payload["seed_evidence_hash"] = canonical_json_hash(payload)
    return payload


def build_stair_phase_d_report(
    config: StairClimbingTaskConfig,
    *,
    official: StairOfficialReferenceProgram,
    pilot_evidence: Mapping[str, Any],
    event_log_checksum: str,
    environment: Mapping[str, Any],
) -> dict[str, Any]:
    assert_stair_pilot_formal_bindings_disjoint(config)
    event_log_checksum = require_sha256_checksum(
        event_log_checksum,
        "stair pilot event log checksum",
    )
    normalized_evidence = validate_stair_pilot_seed_evidence(
        config.protocol_v2_path,
        pilot_evidence,
    )
    training_contract = config.protocol_v2.pilot_training_contract
    payload = {
        "schema_version": STAIR_PHASE_D_REPORT_SCHEMA_VERSION,
        "status": normalized_evidence["status"],
        "draft_protocol_hash": config.protocol_v2.protocol_hash,
        "base_protocol_hash": config.protocol_v1.protocol_hash,
        "reference_program": official.artifact.to_public_dict(),
        "budget_policy": {
            "kind": training_contract["budget_policy"],
            "max_iterations": config.train_max_iterations,
            "budget_steps": config.budget_steps,
            "budget_optimality_claimed": False,
            "curve_convergence_claimed": False,
            "warm_start_allowed": False,
            "capability_confirmation_checkpoint": "final_iteration",
        },
        "pilot_evidence": normalized_evidence,
        "event_log_checksum": event_log_checksum,
        "environment": dict(environment),
        "boundaries": dict(_BOUNDARIES),
    }
    payload["report_hash"] = canonical_json_hash(payload)
    return validate_stair_phase_d_report(
        config,
        official=official,
        report=payload,
    )


def validate_stair_phase_d_report(
    config: StairClimbingTaskConfig,
    *,
    official: StairOfficialReferenceProgram,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    fields = {
        "schema_version",
        "status",
        "draft_protocol_hash",
        "base_protocol_hash",
        "reference_program",
        "budget_policy",
        "pilot_evidence",
        "event_log_checksum",
        "environment",
        "boundaries",
        "report_hash",
    }
    normalized = _strict_fields(report, fields, "stair Phase D report")
    if normalized["schema_version"] != STAIR_PHASE_D_REPORT_SCHEMA_VERSION:
        raise ValueError("stair Phase D report schema mismatch")
    _assert_config_identity(config, normalized, include_budget=False)
    if normalized["reference_program"] != official.artifact.to_public_dict():
        raise ValueError("stair Phase D reference program mismatch")
    expected_policy = {
        "kind": config.protocol_v2.pilot_training_contract["budget_policy"],
        "max_iterations": config.train_max_iterations,
        "budget_steps": config.budget_steps,
        "budget_optimality_claimed": False,
        "curve_convergence_claimed": False,
        "warm_start_allowed": False,
        "capability_confirmation_checkpoint": "final_iteration",
    }
    if normalized["budget_policy"] != expected_policy:
        raise ValueError("stair Phase D fixed budget policy mismatch")
    evidence = validate_stair_pilot_seed_evidence(
        config.protocol_v2_path,
        normalized["pilot_evidence"],
    )
    if normalized["status"] != evidence["status"]:
        raise ValueError("stair Phase D report status mismatch")
    require_sha256_checksum(
        normalized["event_log_checksum"],
        "stair Phase D event log checksum",
    )
    if not isinstance(normalized["environment"], Mapping):
        raise ValueError("stair Phase D environment evidence must be an object")
    _validate_environment_evidence(normalized["environment"])
    training_distribution = config.protocol_v2.runner_owned_training_distribution
    if training_distribution is None:
        raise ValueError("stair Phase D requires a training distribution contract")
    if normalized["environment"]["upstream_training_distribution_sources"] != (
        stair_training_distribution_upstream_source_evidence(training_distribution)
    ):
        raise ValueError(
            "stair Phase D upstream training distribution source evidence mismatch"
        )
    if normalized["boundaries"] != _BOUNDARIES:
        raise ValueError("stair Phase D boundary flags mismatch")
    _validate_declared_hash(normalized, "report_hash", "stair Phase D report")
    normalized["pilot_evidence"] = evidence
    return normalized


def validate_stair_pilot_seed_evidence(
    protocol_v2_path: Path,
    seed: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the fixed-budget training-seed result from raw evidence."""

    return _validate_seed_evidence(
        resolve_existing_file(
            protocol_v2_path,
            field_name="stair pilot protocol_v2_path",
        ),
        seed,
    )


def validate_stair_phase_d_environment_evidence(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the immutable source/runtime snapshot for one pilot run."""

    normalized = dict(_mapping(value, "stair Phase D environment"))
    _validate_environment_evidence(normalized)
    return normalized


def write_canonical_pilot_json(path: Path, payload: Mapping[str, Any]) -> None:
    if not isinstance(path, Path):
        raise TypeError("stair pilot JSON path must be a Path")
    if path.exists():
        raise FileExistsError(f"stair pilot output must be fresh: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(dict(payload), trailing_newline=True))


def load_canonical_pilot_json(path: Path, *, context: str) -> dict[str, Any]:
    return load_strict_json_object(
        path,
        context=context,
        require_canonical=True,
        trailing_newline=True,
    )


def _validate_seed_evidence(
    protocol_v2_path: Path,
    seed: Any,
) -> dict[str, Any]:
    fields = {
        "schema_version",
        "status",
        "failed_gates",
        "attempt_id",
        "training_seed_index",
        "budget_steps",
        "max_iterations",
        "train_summary_hash",
        "eval_summary_hash",
        "checkpoint_hash",
        "training_binding",
        "evaluation_binding",
        "gate_statuses",
        "training",
        "evaluation",
        "processes",
        "seed_evidence_hash",
    }
    normalized = _strict_fields(seed, fields, "stair pilot seed evidence")
    if normalized["schema_version"] != STAIR_PILOT_SEED_EVIDENCE_SCHEMA_VERSION:
        raise ValueError("stair pilot seed evidence schema mismatch")
    config = build_stair_pilot_config(protocol_v2_path)
    if normalized["training_seed_index"] != 0:
        raise ValueError("stair pilot seed evidence must use the fixed seed index")
    if normalized["budget_steps"] != config.budget_steps:
        raise ValueError("stair pilot seed budget differs from protocol")
    if normalized["max_iterations"] != config.train_max_iterations:
        raise ValueError("stair pilot seed iteration mismatch")
    _canonical_text(normalized["attempt_id"], "stair pilot attempt_id")
    require_sha256_digest(
        normalized["train_summary_hash"],
        "stair pilot train_summary_hash",
    )
    checkpoint_hash = require_sha256_checksum(
        normalized["checkpoint_hash"],
        "stair pilot checkpoint",
    )
    if normalized["training_binding"] != build_stair_pilot_training_binding(
        config
    ).to_public_dict():
        raise ValueError("stair pilot seed training binding mismatch")
    if normalized["evaluation_binding"] != build_stair_pilot_evaluation_binding(
        config
    ).to_public_dict():
        raise ValueError("stair pilot seed evaluation binding mismatch")

    training = _mapping(normalized["training"], "training")
    require_sha256_digest(
        training.get("initial_policy_state_hash"),
        "stair pilot initial policy state hash",
    )
    _validate_training_distribution_evidence(
        config,
        training.get("training_distribution"),
    )
    expected_initial_policy_state_hash = training.get(
        "expected_initial_policy_state_hash"
    )
    if expected_initial_policy_state_hash is not None:
        require_sha256_digest(
            expected_initial_policy_state_hash,
            "stair pilot expected initial policy state hash",
        )
        if expected_initial_policy_state_hash != training["initial_policy_state_hash"]:
            raise ValueError("stair pilot initial policy pairing mismatch")
    if training.get("latest_checkpoint_iteration") != config.train_max_iterations - 1:
        raise ValueError("stair pilot seed checkpoint iteration mismatch")
    budget = _mapping(training.get("budget_accounting"), "budget_accounting")
    if (
        budget.get("status") != "pass"
        or budget.get("planned_env_steps") != config.budget_steps
    ):
        raise ValueError("stair pilot seed budget accounting mismatch")
    _validate_runtime_authority_projection(config, training.get("runtime_authority"))
    _validate_curve_summary(
        training.get("curve_summary"),
        max_iterations=config.train_max_iterations,
        diagnostics_contract=config.protocol_v2.pilot_training_contract[
            "curve_diagnostics"
        ],
    )

    evaluation = _mapping(normalized["evaluation"], "evaluation")
    if set(evaluation) != {"status", "checkpoint_evaluation"}:
        raise ValueError("stair pilot seed evaluation fields mismatch")
    if evaluation["status"] != "completed":
        raise ValueError("stair pilot capability evaluation is incomplete")
    require_sha256_digest(
        normalized["eval_summary_hash"],
        "stair pilot eval_summary_hash",
    )
    checkpoint_evaluation = _validate_checkpoint_evaluation(
        config,
        evaluation["checkpoint_evaluation"],
        expected_iteration=config.train_max_iterations - 1,
        expected_checkpoint_hash=checkpoint_hash,
    )
    capability_status = _mapping(
        checkpoint_evaluation["capability_evidence"],
        "capability_evidence",
    )["gate"]["status"]
    expected_gate_statuses = {"capability": capability_status}
    failed_gates = [] if capability_status == "pass" else ["capability"]
    if normalized["gate_statuses"] != expected_gate_statuses:
        raise ValueError("stair pilot seed gate statuses drifted")
    if normalized["failed_gates"] != failed_gates or normalized["status"] != (
        capability_status
    ):
        raise ValueError("stair pilot seed aggregate status mismatch")

    processes = _mapping(normalized["processes"], "processes")
    if set(processes) != {"train", "eval"}:
        raise ValueError("stair pilot seed process set mismatch")
    _validate_process_completion(processes["train"], phase="train")
    _validate_process_completion(processes["eval"], phase="eval")
    _validate_declared_hash(
        normalized,
        "seed_evidence_hash",
        "stair pilot seed evidence",
    )
    normalized["evaluation"] = {
        "status": "completed",
        "checkpoint_evaluation": checkpoint_evaluation,
    }
    return normalized


def _validate_training_execution(
    config: StairClimbingTaskConfig,
    value: Any,
    *,
    checkpoint_path: Path,
    expected_reward_program: Mapping[str, Any],
    expected_workload_kind: str,
    expected_num_envs: int,
    expected_num_steps_per_env: int,
    expected_max_iterations: int,
    expected_budget_steps: int,
    portable_paths: bool,
) -> None:
    execution = _mapping(value, "execution")
    if execution.get("status") != "pass":
        raise ValueError("stair pilot execution status mismatch")
    if execution.get("training_executed") is not True:
        raise ValueError("stair pilot execution did not train")
    if execution.get("errors") != []:
        raise ValueError("stair pilot execution contains errors")
    checkpoint = _mapping(execution.get("checkpoint"), "execution.checkpoint")
    expected_path = resolve_existing_file(
        checkpoint_path,
        field_name="stair pilot checkpoint path",
    )
    if portable_paths:
        if checkpoint.get("uri") != "artifact-relative://checkpoint.pt":
            raise ValueError("stair pilot checkpoint portable URI mismatch")
        metadata = _mapping(checkpoint.get("metadata"), "checkpoint metadata")
        if set(metadata) != {
            "artifact_relative_path",
            "source_latest_checkpoint_sha256",
        } or metadata["artifact_relative_path"] != "checkpoint.pt":
            raise ValueError("stair pilot checkpoint portable metadata mismatch")
    else:
        actual_path = resolve_existing_file(
            Path(str(checkpoint.get("uri"))),
            field_name="stair pilot declared checkpoint path",
        )
        if actual_path != expected_path:
            raise ValueError("stair pilot checkpoint path mismatch")
    checksum = require_sha256_checksum(
        checkpoint.get("checksum"),
        "stair pilot checkpoint checksum",
    )
    if file_sha256(expected_path) != checksum:
        raise ValueError("stair pilot checkpoint bytes drifted")
    if portable_paths:
        require_sha256_checksum(
            metadata["source_latest_checkpoint_sha256"],
            "stair pilot source checkpoint checksum",
        )
    training = _mapping(execution.get("training"), "execution.training")
    if training.get("workload_kind") != expected_workload_kind:
        raise ValueError("stair training execution workload kind mismatch")
    if training.get("num_envs") != expected_num_envs:
        raise ValueError("stair training execution env count mismatch")
    if training.get("num_steps_per_env") != expected_num_steps_per_env:
        raise ValueError("stair training execution rollout length mismatch")
    if training.get("budget_steps") != expected_budget_steps:
        raise ValueError("stair pilot execution budget mismatch")
    if training.get("max_iterations") != expected_max_iterations:
        raise ValueError("stair pilot execution iteration mismatch")
    if portable_paths and training.get("log_dir") != (
        "artifact-relative://train_run/rsl_rl"
    ):
        raise ValueError("stair pilot TensorBoard portable URI mismatch")
    if training.get("latest_checkpoint_iteration") != expected_max_iterations - 1:
        raise ValueError("stair pilot execution final checkpoint mismatch")
    require_sha256_digest(
        training.get("initial_policy_state_hash"),
        "stair pilot initial policy state hash",
    )
    _validate_training_distribution_evidence(
        config,
        training.get("training_distribution"),
    )
    expected_initial_policy_state_hash = training.get(
        "expected_initial_policy_state_hash"
    )
    if expected_initial_policy_state_hash is not None:
        require_sha256_digest(
            expected_initial_policy_state_hash,
            "stair pilot expected initial policy state hash",
        )
        if expected_initial_policy_state_hash != training["initial_policy_state_hash"]:
            raise ValueError("stair pilot execution initial policy pairing mismatch")
    budget = _mapping(training.get("budget_accounting"), "budget_accounting")
    if budget != build_budget_accounting(
        budget_steps=expected_budget_steps,
        num_envs=expected_num_envs,
        num_steps_per_env=expected_num_steps_per_env,
        max_iterations=expected_max_iterations,
    ):
        raise ValueError("stair pilot budget accounting did not close")
    runtime = _mapping(
        execution.get("runtime_activation_summary"),
        "runtime_activation_summary",
    )
    if runtime.get("program") != expected_reward_program:
        raise ValueError("stair pilot executed reward program identity mismatch")
    subject = _mapping(expected_reward_program.get("subject"), "reward subject")
    runtime_gate = build_stair_runtime_activation_gate(
        config,
        {
            "runtime_activation_summary": runtime,
            "program_source_checksum": expected_reward_program.get(
                "program_source_checksum"
            ),
            "subject_manifest_hash": expected_reward_program.get(
                "subject_manifest_hash"
            ),
            "reward_program_qualification_hash": expected_reward_program.get(
                "qualification_hash"
            ),
            "subject_kind": subject.get("kind"),
            "subject_id": subject.get("id"),
        },
        expected_control_calls=expected_num_steps_per_env * expected_max_iterations,
    )
    if runtime_gate.get("status") != "pass":
        raise ValueError(
            "stair pilot reward runtime authority failed: "
            + ",".join(runtime_gate.get("failed_checks", ()))
        )
    preflight = _mapping(runtime.get("activation_preflight"), "activation_preflight")
    if execution.get("activation_preflight") != preflight:
        raise ValueError("stair pilot activation preflight has two values")


def _portable_stair_pilot_execution(value: Mapping[str, Any]) -> dict[str, Any]:
    execution = dict(_mapping(value, "execution"))
    checkpoint = dict(_mapping(execution.get("checkpoint"), "execution.checkpoint"))
    checksum = require_sha256_checksum(
        checkpoint.get("checksum"),
        "stair pilot checkpoint checksum",
    )
    source_metadata = _mapping(checkpoint.get("metadata"), "checkpoint metadata")
    source_checksum = require_sha256_checksum(
        source_metadata.get("source_latest_checkpoint_sha256", checksum),
        "stair pilot source checkpoint checksum",
    )
    checkpoint["uri"] = "artifact-relative://checkpoint.pt"
    checkpoint["metadata"] = {
        "artifact_relative_path": "checkpoint.pt",
        "source_latest_checkpoint_sha256": source_checksum,
    }
    training = dict(_mapping(execution.get("training"), "execution.training"))
    training["log_dir"] = "artifact-relative://train_run/rsl_rl"
    execution["checkpoint"] = checkpoint
    execution["training"] = training
    return execution


def _project_runtime_authority(value: Any) -> dict[str, Any]:
    runtime = _mapping(value, "runtime_activation_summary")
    program = _mapping(runtime.get("program"), "runtime program")
    preflight = _mapping(runtime.get("activation_preflight"), "activation_preflight")
    return {
        "schema_version": (
            "true_rl_agent.task.stair_climbing.phase_d_runtime_authority.v2"
        ),
        "status": "pass",
        "runtime_status": runtime.get("runtime_status"),
        "dispatcher_calls": runtime.get("dispatcher_calls"),
        "term_calls": runtime.get("term_calls"),
        "declared_term_count": runtime.get("declared_term_count"),
        "input_guard": runtime.get("input_guard"),
        "rng_guard": runtime.get("rng_guard"),
        "source_alias_guard": runtime.get("source_alias_guard"),
        "training_recorder_enabled": runtime.get("training_recorder_enabled"),
        "reward_manager_active_terms": runtime.get("reward_manager_active_terms"),
        "reward_manager_term_weight": runtime.get("reward_manager_term_weight"),
        "termination_manager_resolution": runtime.get(
            "termination_manager_resolution"
        ),
        "program_source_checksum": program.get("program_source_checksum"),
        "program_qualification_hash": program.get("qualification_hash"),
        "activation_preflight_hash": canonical_json_hash(preflight),
        "runtime_evidence_hash": canonical_json_hash(runtime),
    }


def _validate_training_distribution_evidence(
    config: StairClimbingTaskConfig,
    value: Any,
) -> None:
    contract = config.protocol_v2.runner_owned_training_distribution
    if contract is None:
        expected = {
            "schema_version": (
                "true_rl_agent.task.stair_climbing."
                "training_distribution_evidence.legacy_v1"
            ),
            "status": "legacy",
        }
    else:
        expected = dict(
            resolved_stair_training_distribution(
                contract,
                base_protocol=config.protocol_v1.to_dict(),
            )
        )
    if value != expected:
        raise ValueError("stair pilot training distribution evidence mismatch")


def _project_stair_runner_execution(
    config: StairClimbingTaskConfig,
    value: Any,
) -> dict[str, Any]:
    effective = _mapping(value, "effective Stair runner execution")
    execution = config.protocol_v2.to_public_dict()["training_execution"]
    if "observation_groups" not in execution:
        projection = {"runner_contract_mode": "legacy_registry_defaults"}
    else:
        projection = {
            "runner_contract_mode": effective.get("runner_contract_mode"),
            "observation_groups": effective.get("observation_groups"),
            "resolved_observation_groups": effective.get(
                "resolved_observation_groups"
            ),
            "clip_actions": effective.get("clip_actions"),
            "empirical_normalization": effective.get(
                "empirical_normalization"
            ),
            "logger": effective.get("logger"),
            "resume": effective.get("resume"),
        }
    _validate_stair_runner_execution_projection(config, projection)
    return projection


def _validate_stair_runner_execution_projection(
    config: StairClimbingTaskConfig,
    value: Any,
) -> None:
    projection = _mapping(value, "Stair runner execution projection")
    execution = config.protocol_v2.to_public_dict()["training_execution"]
    if "observation_groups" not in execution:
        expected = {"runner_contract_mode": "legacy_registry_defaults"}
    else:
        expected = {
            "runner_contract_mode": "explicit",
            "observation_groups": execution["observation_groups"],
            "resolved_observation_groups": execution["observation_groups"],
            "clip_actions": execution["clip_actions"],
            "empirical_normalization": execution["empirical_normalization"],
            "logger": execution["logger"],
            "resume": execution["resume"],
        }
    if projection != expected:
        raise ValueError("stair pilot RSL-RL runner execution evidence mismatch")


def _validate_runtime_authority_projection(
    config: StairClimbingTaskConfig,
    value: Any,
) -> None:
    runtime = _mapping(value, "runtime_authority")
    fields = {
        "schema_version",
        "status",
        "runtime_status",
        "dispatcher_calls",
        "term_calls",
        "declared_term_count",
        "input_guard",
        "rng_guard",
        "source_alias_guard",
        "training_recorder_enabled",
        "reward_manager_active_terms",
        "reward_manager_term_weight",
        "termination_manager_resolution",
        "program_source_checksum",
        "program_qualification_hash",
        "activation_preflight_hash",
        "runtime_evidence_hash",
    }
    if set(runtime) != fields:
        raise ValueError("stair pilot runtime authority fields mismatch")
    if runtime.get("schema_version") != (
        "true_rl_agent.task.stair_climbing.phase_d_runtime_authority.v2"
    ):
        raise ValueError("stair pilot runtime authority schema mismatch")
    if runtime.get("status") != "pass":
        raise ValueError("stair pilot runtime authority did not pass")
    if runtime.get("runtime_status") != STAIR_COMPLETE_REWARD_RUNTIME_STATUS:
        raise ValueError("stair pilot runtime status mismatch")
    calls = runtime.get("dispatcher_calls")
    terms = runtime.get("declared_term_count")
    if (
        isinstance(calls, bool)
        or not isinstance(calls, int)
        or calls <= 0
        or isinstance(terms, bool)
        or not isinstance(terms, int)
        or terms <= 0
        or runtime.get("term_calls") != calls * terms
    ):
        raise ValueError("stair pilot runtime authority accounting mismatch")
    if any(runtime.get(name) != "pass" for name in ("input_guard", "rng_guard", "source_alias_guard")):
        raise ValueError("stair pilot runtime authority guard failed")
    if runtime.get("training_recorder_enabled") is not False:
        raise ValueError("stair pilot runtime authority enabled training recorder")
    if runtime.get("reward_manager_active_terms") != ["complete_reward"]:
        raise ValueError("stair pilot runtime authority term mismatch")
    if runtime.get("reward_manager_term_weight") != 1.0:
        raise ValueError("stair pilot runtime authority weight mismatch")
    termination_resolution = _mapping(
        runtime.get("termination_manager_resolution"),
        "termination_manager_resolution",
    )
    expected_resolution = stair_termination_manager_resolution_contract(
        config.protocol_v1.to_dict()
    )
    if termination_resolution != expected_resolution:
        raise ValueError("stair pilot termination manager resolution mismatch")
    require_sha256_checksum(
        runtime.get("program_source_checksum"),
        "stair pilot runtime program checksum",
    )
    for name in (
        "program_qualification_hash",
        "activation_preflight_hash",
        "runtime_evidence_hash",
    ):
        require_sha256_digest(runtime.get(name), f"stair pilot runtime {name}")


def _validate_environment_evidence(value: Any) -> None:
    environment = _mapping(value, "stair Phase D environment")
    fields = {
        "schema_version",
        "git_commit",
        "git_worktree_scope",
        "git_worktree_clean",
        "python_version",
        "python_executable_name",
        "package_versions",
        "upstream_training_distribution_sources",
        "hardware",
        "hardware_fingerprint",
    }
    if set(environment) != fields:
        raise ValueError("stair Phase D environment fields mismatch")
    if environment.get("schema_version") != (
        "true_rl_agent.task.stair_climbing.phase_d_environment.v4"
    ):
        raise ValueError("stair Phase D environment schema mismatch")
    git_commit = environment.get("git_commit")
    if (
        not isinstance(git_commit, str)
        or len(git_commit) not in {40, 64}
        or any(char not in "0123456789abcdef" for char in git_commit)
    ):
        raise ValueError("stair Phase D git commit must be a canonical object ID")
    if environment.get("git_worktree_scope") != "true_RL_Agent":
        raise ValueError("stair Phase D git worktree scope mismatch")
    if environment.get("git_worktree_clean") is not True:
        raise ValueError("stair Phase D evidence requires a clean project worktree")
    _canonical_text(environment.get("python_version"), "python_version")
    executable_name = _canonical_text(
        environment.get("python_executable_name"),
        "python_executable_name",
    )
    if Path(executable_name).name != executable_name:
        raise ValueError("stair Phase D python executable name contains a path")
    versions = _mapping(environment.get("package_versions"), "package_versions")
    if set(versions) != {"torch", "isaaclab", "rsl-rl-lib", "tensorboard"}:
        raise ValueError("stair Phase D package version set mismatch")
    for name, version in versions.items():
        if version is not None:
            _canonical_text(version, f"package version {name}")
    upstream_sources = _mapping(
        environment.get("upstream_training_distribution_sources"),
        "upstream training distribution sources",
    )
    if set(upstream_sources) != {
        "schema_version",
        "status",
        "project",
        "revision",
        "source_files",
        "source_manifest_hash",
    }:
        raise ValueError("stair Phase D upstream source evidence fields mismatch")
    if upstream_sources.get("status") != "pass":
        raise ValueError("stair Phase D upstream source evidence did not pass")
    _canonical_text(upstream_sources.get("schema_version"), "upstream source schema")
    for field_name in ("project", "revision"):
        _canonical_text(
            upstream_sources.get(field_name),
            f"upstream source {field_name}",
        )
    source_files = upstream_sources.get("source_files")
    if not isinstance(source_files, list) or not source_files:
        raise ValueError("stair Phase D upstream source file set is empty")
    previous_identity = None
    normalized_sources = []
    for raw in source_files:
        source = _mapping(raw, "upstream source file")
        if set(source) != {
            "import_package",
            "package_relative_path",
            "covered_symbols",
            "checksum",
            "size_bytes",
        }:
            raise ValueError("stair Phase D upstream source file fields mismatch")
        import_package = _canonical_text(
            source.get("import_package"),
            "upstream source import_package",
        )
        relative_path = _canonical_text(
            source.get("package_relative_path"),
            "upstream source package_relative_path",
        )
        relative = Path(relative_path)
        covered_symbols = source.get("covered_symbols")
        if (
            not isinstance(covered_symbols, list)
            or not covered_symbols
            or any(not isinstance(item, str) or not item for item in covered_symbols)
            or covered_symbols != sorted(set(covered_symbols))
        ):
            raise ValueError(
                "stair Phase D upstream source symbol coverage is invalid"
            )
        if (
            relative.is_absolute()
            or relative.as_posix() != relative_path
            or ".." in relative.parts
            or (
                previous_identity is not None
                and (import_package, relative_path) <= previous_identity
            )
        ):
            raise ValueError("stair Phase D upstream source path is not canonical")
        normalized_sources.append(
            {
                "import_package": import_package,
                "package_relative_path": relative_path,
                "covered_symbols": list(covered_symbols),
                "checksum": require_sha256_checksum(
                    source.get("checksum"),
                    "upstream source checksum",
                ),
                "size_bytes": _positive_int(
                    source.get("size_bytes"),
                    "upstream source size",
                ),
            }
        )
        previous_identity = (import_package, relative_path)
    source_identity = {
        "project": upstream_sources["project"],
        "revision": upstream_sources["revision"],
        "source_files": normalized_sources,
    }
    if upstream_sources.get("source_manifest_hash") != canonical_json_hash(
        source_identity
    ):
        raise ValueError("stair Phase D upstream source manifest hash mismatch")
    hardware = _mapping(environment.get("hardware"), "hardware")
    required_hardware = {
        "platform",
        "machine",
        "device",
        "cuda_available",
        "cuda_version",
        "gpu_name",
    }
    if not required_hardware.issubset(hardware) or set(hardware) - (
        required_hardware | {"probe_error"}
    ):
        raise ValueError("stair Phase D hardware evidence fields mismatch")
    for name in ("platform", "machine", "device"):
        _canonical_text(hardware.get(name), f"hardware {name}")
    if hardware.get("cuda_available") is not True:
        raise ValueError("stair Phase D passing evidence requires CUDA")
    for name in ("cuda_version", "gpu_name"):
        _canonical_text(hardware.get(name), f"hardware {name}")
    if "probe_error" in hardware:
        raise ValueError("stair Phase D passing hardware evidence contains an error")
    expected_fingerprint = canonical_json_hash(hardware)
    if environment.get("hardware_fingerprint") != expected_fingerprint:
        raise ValueError("stair Phase D hardware fingerprint mismatch")
    canonical_json_bytes(environment)


def _validate_curve_summary(
    value: Any,
    *,
    max_iterations: int,
    diagnostics_contract: Mapping[str, Any],
) -> None:
    curve = _mapping(value, "curve_summary")
    if set(curve) != {
        "schema_version",
        "event_file_count",
        "required_scalar_tags",
        "available_scalar_tags",
        "expected_last_iteration",
        "diagnostics_contract",
        "scalars",
        "tail_metrics",
        "decision_authority",
        "status",
    }:
        raise ValueError("stair pilot curve diagnostic fields mismatch")
    if curve.get("schema_version") != STAIR_PILOT_CURVE_SUMMARY_SCHEMA_VERSION:
        raise ValueError("stair pilot curve summary schema mismatch")
    if curve.get("status") != "pass":
        raise ValueError("stair pilot curve summary did not pass")
    if curve.get("required_scalar_tags") != list(STAIR_PILOT_REQUIRED_SCALARS):
        raise ValueError("stair pilot required scalar set mismatch")
    if curve.get("expected_last_iteration") != max_iterations - 1:
        raise ValueError("stair pilot curve iteration boundary mismatch")
    diagnostics = _validate_curve_diagnostics_contract(
        diagnostics_contract,
        max_iterations=max_iterations,
    )
    if curve.get("diagnostics_contract") != diagnostics:
        raise ValueError("stair pilot curve diagnostics contract mismatch")
    if curve.get("decision_authority") is not False:
        raise ValueError("stair pilot curve diagnostics gained decision authority")
    scalars = _mapping(curve.get("scalars"), "curve scalars")
    if set(scalars) != set(STAIR_PILOT_REQUIRED_SCALARS):
        raise ValueError("stair pilot curve scalar set mismatch")
    for tag in STAIR_PILOT_REQUIRED_SCALARS:
        item = _mapping(scalars[tag], f"curve scalar {tag}")
        if (
            item.get("count") != max_iterations
            or item.get("first_step") != 0
            or item.get("last_step") != max_iterations - 1
        ):
            raise ValueError(f"stair pilot curve scalar coverage mismatch: {tag}")
        if item.get("nonfinite_count") != 0:
            raise ValueError(f"stair pilot curve scalar is non-finite: {tag}")
        for field_name in ("first", "last", "minimum", "maximum"):
            _finite(item.get(field_name), f"curve scalar {tag}.{field_name}")
        tail = _mapping(item.get("tail"), f"curve scalar {tag}.tail")
        if set(tail) != {
            "count",
            "first_step",
            "last_step",
            "mean",
            "population_stddev",
            "slope_per_iteration",
            "normalized_abs_tail_change",
        }:
            raise ValueError(f"stair pilot curve tail fields mismatch: {tag}")
        if (
            tail.get("count") != diagnostics["tail_window_iterations"]
            or tail.get("first_step")
            != max_iterations - diagnostics["tail_window_iterations"]
            or tail.get("last_step") != max_iterations - 1
        ):
            raise ValueError(f"stair pilot curve tail coverage mismatch: {tag}")
        for field_name in (
            "mean",
            "population_stddev",
            "slope_per_iteration",
            "normalized_abs_tail_change",
        ):
            _finite(tail.get(field_name), f"curve scalar {tag}.tail.{field_name}")
        if tail["population_stddev"] < 0.0 or tail["normalized_abs_tail_change"] < 0.0:
            raise ValueError(f"stair pilot curve tail statistic is negative: {tag}")

    metrics = _mapping(curve.get("tail_metrics"), "curve tail metrics")
    if set(metrics) != set(diagnostics["required_scalar_tags"]):
        raise ValueError("stair pilot curve diagnostic metric set mismatch")
    for tag in diagnostics["required_scalar_tags"]:
        expected = {
            "normalized_abs_tail_change": scalars[tag]["tail"][
                "normalized_abs_tail_change"
            ]
        }
        if metrics[tag] != expected:
            raise ValueError(f"stair pilot curve diagnostic drifted: {tag}")


def _validate_curve_diagnostics_contract(
    value: Mapping[str, Any],
    *,
    max_iterations: int,
) -> dict[str, Any]:
    contract = dict(_mapping(value, "pilot curve diagnostics contract"))
    if set(contract) != {
        "schema_version",
        "tail_window_iterations",
        "normalization",
        "required_scalar_tags",
    }:
        raise ValueError("stair pilot curve diagnostics contract fields mismatch")
    if contract["schema_version"] != (
        "true_rl_agent.task.stair_climbing.pilot_curve_diagnostics.v1"
    ):
        raise ValueError("stair pilot curve diagnostics contract schema mismatch")
    window = _positive_int(
        contract["tail_window_iterations"],
        "stair pilot curve diagnostics window",
    )
    if window >= max_iterations:
        raise ValueError("stair pilot curve diagnostics window exceeds training")
    if contract["normalization"] != (
        "absolute_linear_tail_change_over_population_stddev_with_1e-6_floor"
    ):
        raise ValueError("stair pilot curve diagnostics normalization mismatch")
    tags = contract["required_scalar_tags"]
    if (
        not isinstance(tags, Sequence)
        or isinstance(tags, (str, bytes))
        or list(tags)
        != [
        "Train/mean_reward",
        "Curriculum/terrain_levels",
        ]
    ):
        raise ValueError("stair pilot curve diagnostics scalar tags mismatch")
    contract["required_scalar_tags"] = list(tags)
    return contract


def _curve_tail_statistics(
    steps: Sequence[int],
    values: Sequence[float],
    *,
    window_iterations: int,
) -> dict[str, Any]:
    tail_steps = tuple(float(item) for item in steps[-window_iterations:])
    tail_values = tuple(float(item) for item in values[-window_iterations:])
    count = len(tail_values)
    mean_x = sum(tail_steps) / count
    mean_y = sum(tail_values) / count
    variance_y = sum((item - mean_y) ** 2 for item in tail_values) / count
    denominator = sum((item - mean_x) ** 2 for item in tail_steps)
    if denominator <= 0.0:
        raise ValueError("stair pilot curve diagnostic tail has no iteration span")
    slope = sum(
        (x - mean_x) * (y - mean_y)
        for x, y in zip(tail_steps, tail_values, strict=True)
    ) / denominator
    population_stddev = math.sqrt(variance_y)
    normalized_change = (
        abs(slope)
        * float(count - 1)
        / max(population_stddev, 1.0e-6)
    )
    return {
        "count": count,
        "first_step": int(tail_steps[0]),
        "last_step": int(tail_steps[-1]),
        "mean": mean_y,
        "population_stddev": population_stddev,
        "slope_per_iteration": slope,
        "normalized_abs_tail_change": normalized_change,
    }


def _validate_process_completion(value: Any, *, phase: str) -> dict[str, Any]:
    process = _mapping(value, f"{phase} process")
    expected_fields = {
        "event_id",
        "phase",
        "status",
        "has_exited",
        "exit_code",
        "failure_message",
    }
    if set(process) != expected_fields:
        raise ValueError(f"stair pilot {phase} process fields mismatch")
    if process.get("phase") != phase or process.get("status") != "completed":
        raise ValueError(f"stair pilot {phase} process did not complete")
    if process.get("has_exited") is not True or process.get("exit_code") != 0:
        raise ValueError(f"stair pilot {phase} process exit evidence mismatch")
    if process.get("failure_message") is not None:
        raise ValueError(f"stair pilot {phase} process contains a failure")
    _canonical_text(process.get("event_id"), f"stair pilot {phase} event_id")
    return dict(process)


def _assert_config_identity(
    config: StairClimbingTaskConfig,
    value: Mapping[str, Any],
    *,
    include_budget: bool = True,
) -> None:
    if value.get("draft_protocol_hash") != config.protocol_v2.protocol_hash:
        raise ValueError("stair pilot draft protocol hash mismatch")
    if value.get("base_protocol_hash") != config.protocol_v1.protocol_hash:
        raise ValueError("stair pilot base protocol hash mismatch")
    if include_budget:
        if value.get("budget_steps") != config.budget_steps:
            raise ValueError("stair pilot budget mismatch")
        if value.get("max_iterations") != config.train_max_iterations:
            raise ValueError("stair pilot max_iterations mismatch")


def _validate_declared_hash(
    value: Mapping[str, Any],
    field_name: str,
    context: str,
) -> None:
    declared = require_sha256_digest(value.get(field_name), f"{context} hash")
    expected = canonical_json_hash(
        {key: item for key, item in value.items() if key != field_name}
    )
    if declared != expected:
        raise ValueError(f"{context} hash mismatch")


def _strict_fields(
    value: Any,
    expected: set[str],
    context: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{context} fields mismatch")
    canonical_json_bytes(value)
    return dict(value)


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return value


def _canonical_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be canonical text")
    return value


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _finite(value: Any, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{field_name} must be finite")
    return float(value)


__all__ = [
    "STAIR_PHASE_D_REPORT_SCHEMA_VERSION",
    "STAIR_PHASE_D_SMOKE_REPORT_SCHEMA_VERSION",
    "STAIR_PILOT_CURVE_SUMMARY_SCHEMA_VERSION",
    "STAIR_PILOT_EVAL_SUMMARY_SCHEMA_VERSION",
    "STAIR_PILOT_REQUIRED_SCALARS",
    "STAIR_PILOT_SEED_EVIDENCE_SCHEMA_VERSION",
    "STAIR_PILOT_TRAIN_SUMMARY_SCHEMA_VERSION",
    "build_stair_phase_d_report",
    "build_stair_phase_d_smoke_report",
    "build_stair_pilot_eval_summary",
    "build_stair_pilot_seed_evidence",
    "build_stair_pilot_train_summary",
    "load_canonical_pilot_json",
    "serialize_stair_pilot_raw_outcomes",
    "summarize_stair_pilot_tensorboard",
    "validate_stair_phase_d_report",
    "validate_stair_phase_d_smoke_report",
    "validate_stair_phase_d_environment_evidence",
    "validate_stair_pilot_eval_summary",
    "validate_stair_pilot_seed_evidence",
    "validate_stair_pilot_train_summary",
    "write_canonical_pilot_json",
]
