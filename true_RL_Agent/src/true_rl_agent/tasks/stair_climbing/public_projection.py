"""Task-owned public projection for production Stair evaluation."""

from __future__ import annotations

import math
from typing import Any, Mapping

from true_rl_agent.evaluation.public_result import PublicEvalResult
from true_rl_agent.evaluation.sealed_validation import (
    CANDIDATE_CONTAMINATION_FIELDS,
    SEALED_VALIDATION_EVIDENCE_KEY,
)
from true_rl_agent.runner.budget import build_budget_accounting
from true_rl_agent.tasks.stair_climbing.config import StairClimbingTaskConfig
from true_rl_agent.tasks.stair_climbing.evaluation_contract import (
    validate_stair_split_summary,
)
from true_rl_agent.tasks.stair_climbing.runtime_profile import (
    STAIR_COMPLETE_REWARD_RUNTIME_ACTIVE_STATUS,
)

_TASK_DERIVED_INPUT_KEYS = {
    "stair_evaluation",
    "public_dev",
    "public_accept",
    "prompt_feedback",
    "accept_evidence_gate",
    "budget_accounting",
    "runtime_activation_gate",
    "stair_behavior_gate",
    "acceptance",
}
_RUNNER_PASSTHROUGH_KEYS = {
    "evaluation_binding",
    "evaluation_subject",
    "subject_manifest_hash",
    "program_source_checksum",
    "training_binding_hash",
    "capability_profile_hash",
    "checkpoint_safety_status",
    "checkpoint_safety_report",
    "evidence_manifest",
    SEALED_VALIDATION_EVIDENCE_KEY,
    "sealed_validation_gate",
    *CANDIDATE_CONTAMINATION_FIELDS,
}


def project_stair_public_result(
    *,
    config: StairClimbingTaskConfig,
    public_result: PublicEvalResult,
    train_summary: Mapping[str, Any],
) -> PublicEvalResult:
    """Validate recorder summaries and derive every Stair acceptance input."""

    if not isinstance(config, StairClimbingTaskConfig):
        raise TypeError("stair public projection requires StairClimbingTaskConfig")
    if not isinstance(public_result, PublicEvalResult):
        raise TypeError("stair public projection requires PublicEvalResult")
    if not isinstance(train_summary, Mapping):
        raise TypeError("stair public projection train_summary must be an object")

    incoming = dict(public_result.metrics)
    unsupported = set(incoming) - _TASK_DERIVED_INPUT_KEYS - _RUNNER_PASSTHROUGH_KEYS
    if unsupported:
        raise ValueError(
            "stair eval metrics contain unsupported field(s): "
            + ", ".join(sorted(unsupported))
        )
    raw_splits = incoming.get("stair_evaluation")
    if not isinstance(raw_splits, Mapping) or set(raw_splits) != {
        "public_dev",
        "public_accept",
    }:
        raise ValueError("stair evaluation must contain exactly the two public splits")

    evaluation_protocol = config.evaluation_protocol()
    public_dev = validate_stair_split_summary(
        config.protocol_v1,
        evaluation_protocol=evaluation_protocol,
        split="public_dev",
        summary=_mapping(raw_splits["public_dev"], "stair public_dev summary"),
    )
    public_accept = validate_stair_split_summary(
        config.protocol_v1,
        evaluation_protocol=evaluation_protocol,
        split="public_accept",
        summary=_mapping(raw_splits["public_accept"], "stair public_accept summary"),
    )
    canonical_score = float(public_accept["score_stats"]["mean"])
    if not math.isclose(
        public_result.public_score,
        canonical_score,
        rel_tol=1.0e-9,
        abs_tol=1.0e-9,
    ):
        raise ValueError("stair public score differs from public_accept macro mean")

    budget_gate = _budget_accounting_gate(config, train_summary)
    runtime_gate = _runtime_activation_gate(config)
    behavior_gate = {
        "status": "pass",
        "reason": "task_recorder_public_split_evidence_complete",
        "source": "runner_owned_stair_transition_recorder",
        "validated_splits": ["public_dev", "public_accept"],
        "complete_seven_bin_coverage": True,
        "classification_warning_count": 0,
    }
    accept_gate = {
        "status": "pass",
        "reason": "sufficient_recomputed_public_accept_evidence",
        "observed_count": public_accept["score_stats"]["count"],
        "required_count": config.protocol_v1.acceptance_protocol.min_score_count,
        "statistics_unit": "one_seven_bin_macro_score_per_seed",
    }
    acceptance_protocol = config.protocol_v1.acceptance_protocol
    required_gates = tuple(
        config.protocol_v1.to_dict()["acceptance"]["required_pass_gates"]
    )
    acceptance = {
        "score_source": acceptance_protocol.score_source,
        "public_score_binding": acceptance_protocol.public_score_binding,
        "score_stats": dict(public_accept["score_stats"]),
        "min_score_count": acceptance_protocol.min_score_count,
        "missing_required_score": False,
        "required_pass_gates": list(required_gates),
        "acceptance_protocol_hash": acceptance_protocol.protocol_hash,
        "absolute_accept_floor": acceptance_protocol.absolute_accept_floor,
        "min_delta": acceptance_protocol.min_delta,
    }

    metrics = {
        key: incoming[key]
        for key in _RUNNER_PASSTHROUGH_KEYS
        if key in incoming
    }
    metrics.update(
        {
            "public_dev": public_dev,
            "public_accept": public_accept,
            "prompt_feedback": _prompt_feedback(public_dev),
            "accept_evidence_gate": accept_gate,
            "budget_accounting": budget_gate,
            "runtime_activation_gate": runtime_gate,
            "stair_behavior_gate": behavior_gate,
            "acceptance": acceptance,
        }
    )
    return PublicEvalResult(
        status=public_result.status,
        public_score=canonical_score,
        metrics=metrics,
    )


def _budget_accounting_gate(
    config: StairClimbingTaskConfig,
    train_summary: Mapping[str, Any],
) -> dict[str, Any]:
    expected = build_budget_accounting(
        budget_steps=config.budget_steps,
        num_envs=config.train_num_envs,
        num_steps_per_env=config.train_steps_per_iteration,
        max_iterations=config.train_max_iterations,
    )
    training = train_summary.get("training")
    if not isinstance(training, Mapping):
        return {"status": "fail", "reason": "training_summary_missing"}
    observed = training.get("budget_accounting")
    if not isinstance(observed, Mapping):
        return {"status": "fail", "reason": "budget_accounting_missing"}
    if dict(observed) != expected:
        return {
            "status": "fail",
            "reason": "budget_accounting_differs_from_stair_config",
        }
    return dict(expected)


def _runtime_activation_gate(config: StairClimbingTaskConfig) -> dict[str, Any]:
    runtime_status = config.train_runtime_profile().to_public_dict()[
        "complete_reward_runtime"
    ]
    active = runtime_status == STAIR_COMPLETE_REWARD_RUNTIME_ACTIVE_STATUS
    return {
        "status": "pass" if active else "fail",
        "reason": (
            "pre_reset_complete_reward_runtime_active"
            if active
            else "complete_reward_runtime_not_activated"
        ),
        "runtime_status": runtime_status,
    }


def _prompt_feedback(public_dev: Mapping[str, Any]) -> dict[str, Any]:
    failures = _mapping(
        public_dev.get("primary_failure_breakdown"),
        "stair public failure breakdown",
    )
    dominant_reason, dominant_rate = max(
        failures.items(),
        key=lambda item: (float(item[1]), str(item[0])),
    )
    return {
        "schema_version": "true_rl_agent.task.stair_climbing.prompt_feedback.v1",
        "source_split": "public_dev",
        "dominant_primary_failure": {
            "reason": dominant_reason,
            "rate": float(dominant_rate),
        },
        "diagnostic_flag_rates": dict(
            _mapping(
                public_dev.get("diagnostic_flag_rates"),
                "stair public diagnostic rates",
            )
        ),
        "progress_summary": dict(
            _mapping(
                public_dev.get("progress_summary"),
                "stair public progress summary",
            )
        ),
    }


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return value


__all__ = ["project_stair_public_result"]
