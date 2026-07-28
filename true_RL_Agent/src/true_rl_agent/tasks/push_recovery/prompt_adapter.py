"""Push recovery task-owned prompt semantics."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from statistics import fmean
from typing import Any, Mapping, Sequence

from true_rl_agent.agent_core.context import (
    sanitize_public_context,
    sanitize_public_records_with_policy,
    sanitize_public_value_with_policy,
)
from true_rl_agent.agent_core.public_safety import (
    FORBIDDEN_PUBLIC_KEY_PARTS,
    contains_forbidden_public_value_marker,
    scrub_public_paths,
)
from true_rl_agent.candidates.schedule_contract import REWARD_WEIGHT_SCALE_KEY
from true_rl_agent.prompting.serialization import canonical_prompt_json_bytes
from true_rl_agent.tasks.prompt_adapter import sanitize_task_prompt_sections
from true_rl_agent.tasks.push_recovery.config import PushRecoveryTaskConfig
from true_rl_agent.tasks.push_recovery.metrics import classify_push_prompt_failure
from true_rl_agent.tasks.push_recovery.reward_input_contract import (
    prompt_effective_reward_input_fields,
    public_push_signal_reward_input_fields,
    push_recovery_reward_input_fields,
    reserved_zero_fallback_reward_input_fields,
)

_RUNTIME_DIAGNOSTIC_OUTPUT_MAX_FIELDS = 24
_RUNTIME_DIAGNOSTIC_OUTPUT_VALUE_MAX_KEYS = 8
_RUNTIME_DIAGNOSTIC_OUTPUT_VALUE_MAX_DEPTH = 3
_RUNTIME_DIAGNOSTIC_OUTPUT_TEXT_LIMIT = 160
_RUNTIME_DIAGNOSTIC_OUTPUT_KEY_LIMIT = 96
_RUNTIME_DIAGNOSTIC_OUTPUT_KEY_PREFIX_LIMIT = 40
_RUNTIME_DIAGNOSTIC_OUTPUT_BYTE_BUDGET = 1536
_RUNTIME_REWARD_TERM_MAX_FIELDS = 16
_RUNTIME_REWARD_TERM_BYTE_BUDGET = 2 * 1024
_RUNTIME_DIAGNOSTIC_OUTPUT_STAT_KEYS = (
    "name",
    "num_envs",
    "fields",
    "mean",
    "std",
    "min",
    "max",
    "finite",
    "shape",
    "finite_fraction",
    "nonzero_fraction",
    "nan_fraction",
    "inf_fraction",
    "count",
)

_FIELD_SHAPES: dict[str, list[str]] = {
    "base_height": ["num_envs"],
    "roll": ["num_envs"],
    "pitch": ["num_envs"],
    "base_pos_xy": ["num_envs", "2"],
    "base_lin_vel": ["num_envs", "3"],
    "base_ang_vel": ["num_envs", "3"],
    "projected_gravity": ["num_envs", "3"],
    "commanded_base_vel": ["num_envs", "3"],
    "joint_pos": ["num_envs", "num_joints"],
    "joint_vel": ["num_envs", "num_joints"],
    "joint_pos_error": ["num_envs", "num_joints"],
    "last_action": ["num_envs", "num_actions"],
    "prev_action": ["num_envs", "num_actions"],
    "action_delta": ["num_envs", "num_actions"],
    "push_active": ["num_envs"],
    "push_vec_public": ["num_envs", "2"],
    "progress_fraction": ["num_envs"],
}

_FIELD_METADATA: dict[str, dict[str, Any]] = {
    "base_height": {
        "semantics": "floating base height above terrain or runner reference plane",
        "units": "m",
        "coordinate_frame": "world_z",
        "source_status": "observed_or_zero_fallback",
        "fallback_behavior": "zero fallback when root height is unavailable",
        "candidate_usage_guidance": "use as a stance and fall-risk feature with roll and pitch",
        "known_limitations": "zero fallback can make missing observations look like low height",
    },
    "roll": {
        "semantics": "base roll angle derived from root orientation",
        "units": "rad",
        "coordinate_frame": "base_orientation",
        "source_status": "derived_or_zero_fallback",
        "fallback_behavior": "zero fallback when orientation is unavailable",
        "candidate_usage_guidance": "penalize large absolute roll as a stability risk",
        "known_limitations": "zero fallback can hide missing orientation data",
    },
    "pitch": {
        "semantics": "base pitch angle derived from root orientation",
        "units": "rad",
        "coordinate_frame": "base_orientation",
        "source_status": "derived_or_zero_fallback",
        "fallback_behavior": "zero fallback when orientation is unavailable",
        "candidate_usage_guidance": "penalize large absolute pitch as a forward/backward fall risk",
        "known_limitations": "zero fallback can hide missing orientation data",
    },
    "base_pos_xy": {
        "semantics": "base xy position relative to the environment origin",
        "units": "m",
        "coordinate_frame": "env_origin_relative_world_xy",
        "source_status": "derived_or_zero_fallback",
        "fallback_behavior": "zero vector fallback when base position is unavailable",
        "candidate_usage_guidance": "use for bounded drift diagnostics, not absolute map location",
        "known_limitations": "origin-relative position is not a global navigation target",
    },
    "base_lin_vel": {
        "semantics": "base linear velocity vector as exposed by the runner adapter",
        "units": "m/s",
        "coordinate_frame": "runner_reported_base_or_world_frame",
        "source_status": "observed_or_zero_fallback",
        "fallback_behavior": "zero vector fallback when velocity is unavailable",
        "candidate_usage_guidance": "use for recovery damping and excessive drift penalties",
        "known_limitations": "frame can depend on backend adapter availability",
    },
    "base_ang_vel": {
        "semantics": "base angular velocity vector as exposed by the runner adapter",
        "units": "rad/s",
        "coordinate_frame": "runner_reported_base_or_world_frame",
        "source_status": "observed_or_zero_fallback",
        "fallback_behavior": "zero vector fallback when angular velocity is unavailable",
        "candidate_usage_guidance": "use for rotational damping and instability diagnostics",
        "known_limitations": "frame can depend on backend adapter availability",
    },
    "projected_gravity": {
        "semantics": "gravity direction projected into the base frame",
        "units": "unitless",
        "coordinate_frame": "base_frame",
        "source_status": "derived_or_zero_fallback",
        "fallback_behavior": "zero vector fallback when orientation is unavailable",
        "candidate_usage_guidance": "use as an orientation-stability feature",
        "known_limitations": "zero fallback should not be interpreted as a valid gravity direction",
    },
    "commanded_base_vel": {
        "semantics": "public commanded base velocity vector when a command manager is available",
        "units": "m/s_and_rad/s",
        "coordinate_frame": "command_frame_xy_yaw",
        "source_status": "observed_command_or_zero_fallback",
        "fallback_behavior": "zero vector fallback when command is unavailable",
        "candidate_usage_guidance": "compare actual velocity to command only when command signal is meaningful",
        "known_limitations": "zero command can mean either stand-still command or missing command source",
    },
    "joint_pos": {
        "semantics": "joint position vector in runner joint order",
        "units": "rad",
        "coordinate_frame": "runner_joint_order",
        "source_status": "observed_or_zero_fallback",
        "fallback_behavior": "zero matrix fallback when joint positions are unavailable",
        "candidate_usage_guidance": "use for posture regularization and range diagnostics",
        "known_limitations": "joint order is runner-owned and should not be inferred from names",
    },
    "joint_vel": {
        "semantics": "joint velocity vector in runner joint order",
        "units": "rad/s",
        "coordinate_frame": "runner_joint_order",
        "source_status": "observed_or_zero_fallback",
        "fallback_behavior": "zero matrix fallback when joint velocities are unavailable",
        "candidate_usage_guidance": "use for smoothness and instability penalties",
        "known_limitations": "large penalties can suppress recovery actions",
    },
    "joint_pos_error": {
        "semantics": "joint_pos minus runner default joint position",
        "units": "rad",
        "coordinate_frame": "runner_joint_order",
        "source_status": "derived_or_zero_fallback",
        "fallback_behavior": "zero matrix fallback when default joint position is unavailable",
        "candidate_usage_guidance": "use for soft posture bias, not hard constraints",
        "known_limitations": "default posture is runner-owned and task fixed",
    },
    "last_action": {
        "semantics": "current policy action vector for the step",
        "units": "policy_action_units",
        "coordinate_frame": "runner_action_order",
        "source_status": "observed_current_action",
        "fallback_behavior": "zero fallback when action is unavailable",
        "candidate_usage_guidance": "use for action magnitude diagnostics or gentle regularization",
        "known_limitations": "action units are policy-space values, not joint torques",
    },
    "prev_action": {
        "semantics": "previous policy action vector from runner-owned action history",
        "units": "policy_action_units",
        "coordinate_frame": "runner_action_order",
        "source_status": "runner_derived_or_zero_fallback",
        "fallback_behavior": "zero previous action on first step or reset",
        "candidate_usage_guidance": "use with action_delta for smoothness diagnostics",
        "known_limitations": "reset handling can produce legitimate zeros",
    },
    "action_delta": {
        "semantics": "current action minus previous action",
        "units": "policy_action_units",
        "coordinate_frame": "runner_action_order",
        "source_status": "runner_derived_or_zero_fallback",
        "fallback_behavior": "current action minus zero on first step or reset",
        "candidate_usage_guidance": "use to discourage abrupt control changes without blocking recovery",
        "known_limitations": "over-penalizing action changes can hurt push recovery",
    },
    "push_active": {
        "semantics": (
            "per-environment indicator for a push detected on the current runner step; "
            "it may come from an explicit public event or the current step-to-step public velocity delta"
        ),
        "units": "bool_like",
        "coordinate_frame": "not_applicable",
        "source_status": "observed_public_push_signal_or_public_velocity_delta_with_zero_fallback",
        "fallback_behavior": "zero fallback when no push event signal is available",
        "candidate_usage_guidance": (
            "use only as immediate push-onset context; it is not a guaranteed post-push window. "
            "Use posture, velocity, and position inputs for continuing recovery"
        ),
        "known_limitations": (
            "the indicator is not latched and may be zero on later recovery steps; derived signals can include "
            "abrupt public velocity changes when no explicit push event signal is available"
        ),
    },
    "push_vec_public": {
        "semantics": (
            "current-step public xy velocity delta for a runner-owned push event or observed abrupt base-velocity change"
        ),
        "units": "m/s",
        "coordinate_frame": "world_xy_velocity_delta",
        "source_status": "observed_public_push_signal_or_public_velocity_delta_with_zero_fallback",
        "fallback_behavior": "zero vector fallback when no push event signal is available",
        "candidate_usage_guidance": (
            "use to condition immediate response on an already-observed perturbation; use persistent state inputs "
            "for subsequent recovery steps"
        ),
        "known_limitations": (
            "the vector is not a persistent recovery-window state, is not the runner push distribution, "
            "and does not reveal future pushes"
        ),
    },
    "progress_fraction": {
        "semantics": "runner-owned training progress fraction for the current step",
        "units": "fraction",
        "coordinate_frame": "not_applicable",
        "source_status": "runner_owned",
        "fallback_behavior": "runner supplies a clamped value in [0, 1]",
        "candidate_usage_guidance": "use only for mild annealing inside candidate reward logic",
        "known_limitations": "not an eval metric and not a budget control",
    },
}


@dataclass(frozen=True)
class PushRecoveryPromptAdapter:
    """Task-owned prompt adapter for Unitree push recovery."""

    config: PushRecoveryTaskConfig
    task_id: str = "unitree_push_recovery"

    def build_prompt_sections(
        self,
        *,
        public_records: Sequence[Mapping[str, Any]],
        public_context: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        scrub_policy = self.prompt_public_record_scrub_policy()
        safe_public_records = sanitize_public_records_with_policy(
            public_records,
            blocked_key_parts=scrub_policy["blocked_key_parts"],
            blocked_text_parts=scrub_policy["blocked_text_parts"],
        )
        safe_public_context = sanitize_public_value_with_policy(
            public_context,
            blocked_key_parts=scrub_policy["blocked_key_parts"],
            blocked_text_parts=scrub_policy["blocked_text_parts"],
        )
        if not isinstance(safe_public_context, Mapping):
            safe_public_context = {}
        sections = {
            "task": self._task_section(),
            "reward_inputs_contract": _reward_inputs_contract(),
            "public_eval_contract": _public_eval_contract(),
            "public_feedback_summary": _public_feedback_summary(
                safe_public_records,
                safe_public_context,
            ),
            "runtime_feedback_summary": _runtime_feedback_summary(safe_public_records),
            # Bootstrap examples teach the component API. Once evaluated history
            # exists, the bounded source catalog and experiment memory carry the
            # real candidate lineage, which is both more relevant and avoids
            # repeating static tutorial code beside empirical feedback.
            "known_good_examples": _known_good_examples() if not safe_public_records else [],
            "selection_guidance": _selection_guidance(),
        }
        return sanitize_task_prompt_sections(sections)

    def prompt_public_record_scrub_policy(self) -> Mapping[str, Sequence[str]]:
        reserved_fields = reserved_zero_fallback_reward_input_fields()
        return {
            "blocked_key_parts": reserved_fields,
            "blocked_text_parts": reserved_fields,
        }

    def _task_section(self) -> dict[str, Any]:
        config = self.config
        eval_protocol = config.evaluation_protocol().to_public_dict()
        train_runtime_dynamics = config.train_runtime_dynamics()
        eval_runtime_dynamics = config.eval_runtime_dynamics()
        return {
            "schema_version": "true_rl_agent.task_prompt.push_recovery.v2",
            "task_id": self.task_id,
            "name": "Unitree Push Recovery",
            "prompt_adapter": "push_recovery_task_prompt_adapter",
            "objective": (
                "Improve a Unitree locomotion policy so it remains upright and recovers stable "
                "standing behavior after runner-owned planar x/y pushes."
            ),
            "robot_or_environment": {
                "robot_family": "Unitree humanoid locomotion",
                "setting": "cold-start or warm-start policy trained in the fixed runner protocol",
                "push_setup": "runner-owned fixed target push distribution",
            },
            "policy_initialization": {
                "supported": ["random", "warm_start"],
                "candidate_can_change": False,
            },
            "runner_owned_runtime_dynamics": _runtime_dynamics_prompt_summary(
                train_runtime_dynamics,
                eval_runtime_dynamics,
            ),
            "fixed_train_protocol_summary": {
                "baseline_train_distribution": "runner_owned_target_distribution",
                "push_distribution": "fixed_target_push",
                "budget_steps": config.budget_steps,
                "train_num_envs": config.train_num_envs,
                "train_max_iterations": config.train_max_iterations,
                "train_steps_per_iteration": config.train_steps_per_iteration,
                "runtime_profile": config.train_profile,
                "llm_schedule_control": "not_prompt_writable",
            },
            "fixed_eval_protocol_summary": {
                "eval_protocol": "fixed_public_accept_and_sealed_validation",
                "success_predicate": config.success_predicate,
                "seed_visibility": _prompt_safe_seed_visibility(config.eval_seed_visibility),
                **eval_protocol,
                "runtime_profile": config.eval_profile,
                "checkpoint_policy": "runner_sealed_train_output",
            },
            "candidate_must_not_change": [
                "runner",
                "eval protocol",
                "success metric",
                "training budget",
                "seed policy",
                "checkpoint policy",
                "PPO",
                "task invariants",
                "push distribution",
            ],
        }


def _runtime_dynamics_prompt_summary(
    train: Mapping[str, Any],
    evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    same_dynamics = train.get("runtime_dynamics_hash") == evaluation.get("runtime_dynamics_hash")
    prompt_train = {key: value for key, value in train.items() if key != "runtime_dynamics_hash"}
    prompt_eval = {key: value for key, value in evaluation.items() if key != "runtime_dynamics_hash"}
    summary: dict[str, Any] = {
        "schema_version": "true_rl_agent.task_prompt.push_recovery.runtime_dynamics.v1",
        "train": prompt_train,
        "eval": (
            {
                "same_as_train": True,
                "active_profile": evaluation.get("active_profile"),
            }
            if same_dynamics
            else prompt_eval
        ),
        "reasoning_guidance": [],
    }
    standing = train.get("standing_command")
    push = train.get("push")
    guidance = summary["reasoning_guidance"]
    if isinstance(standing, Mapping) and standing.get("commanded_locomotion_is_zero") is True:
        guidance.append(
            "This is a stand-still recovery task: commanded x velocity, y velocity, and yaw rate are zero. "
            "There is no forward locomotion command to preserve."
        )
    if (
        isinstance(push, Mapping)
        and push.get("active_world_axes") == ["x", "y"]
        and push.get("xy_ranges_equal") is True
    ):
        guidance.append(
            "Runner pushes span symmetric world x and y velocity ranges. Do not assume a forward-only or "
            "lateral-y-only disturbance; prefer isotropic xy mechanisms unless public evidence supports "
            "an axis-specific change."
        )
    return summary


def _reward_inputs_contract() -> dict[str, Any]:
    fields = {}
    reserved_fields = {}
    effective_names = set(prompt_effective_reward_input_fields())
    reserved_names = set(reserved_zero_fallback_reward_input_fields())
    task_field_names = push_recovery_reward_input_fields()
    for name in task_field_names:
        metadata = dict(_FIELD_METADATA[name])
        field_contract = {
            "name": name,
            "shape": list(_FIELD_SHAPES[name]),
            "dtype_or_tensor_kind": "tensor_like_numeric",
            **metadata,
        }
        if name in reserved_names:
            field_contract["prompt_visibility"] = "reserved_not_effective"
            reserved_fields[name] = field_contract
        else:
            fields[name] = field_contract
    return {
        "schema_version": "true_rl_agent.reward_inputs_contract.push_recovery.v1",
        "task_id": "unitree_push_recovery",
        "field_order": [name for name in task_field_names if name in effective_names],
        "fields": fields,
        "public_push_signal_fields": list(public_push_signal_reward_input_fields()),
        "reserved_fields": reserved_fields,
        "reserved_field_policy": {
            "candidate_must_not_use_reserved_fields": bool(reserved_fields),
            "reason": "reserved fields are unavailable for candidate logic; an empty set means all listed fields are prompt-effective",
        },
        "candidate_does_not_receive": [
            "env",
            "runner",
            "optimizer",
            "checkpoint policy state",
            "eval config",
            "success metric internals",
            "unprojected seed identifiers",
        ],
    }


def _prompt_safe_seed_visibility(value: str) -> str:
    if value == "counts_only_no_raw_seed_ids":
        return "counts_only_seed_identifiers_omitted"
    return str(value).replace("raw_seed", "unprojected_seed")


def _public_eval_contract() -> dict[str, Any]:
    return {
        "schema_version": "true_rl_agent.public_eval_contract.push_recovery.v2",
        "result_type": "PublicEvalResult(status, public_score, metrics)",
        "status": {
            "meaning": "public projection status for the eval summary",
            "candidate_can_modify": False,
        },
        "public_score": {
            "meaning": "public push recovery success-rate aggregate selected by the task projection",
            "direction": "higher_is_better",
            "range": [0.0, 1.0],
            "missing_value": "score unavailable when required public split is missing or invalid",
        },
        "metrics": {
            "success_rate": "public success-rate summary; higher means more evaluated episodes recovered",
            "public_dev": "public development split aggregate when multiple summaries are available",
            "public_accept": "public accept split aggregate when present; unprojected seed identifiers are omitted",
            "failure_breakdown": "public failure-mode rates such as fall, push_drift, timeout, or task-specific labels",
            "motion_quality": "public secondary motion summaries such as height, tilt, drift, action magnitude",
            "budget_accounting": "read-only training budget accounting gate",
            "accept_evidence_gate": "read-only public-accept sample-size and uncertainty gate",
            "motion_quality_gate": "read-only public sanity gate for unsafe or degenerate motion",
            "behavior_quality_gate": "read-only public behavior gate for recovery evidence completeness and bounded metrics",
            "sealed_validation_gate": "read-only accept-hard gate summary for checkpoint safety and candidate isolation",
            "prompt_feedback": "task-owned bounded feedback summary derived from public split aggregates",
            "acceptance": "read-only score source and required public gates; candidate cannot edit this",
            "errors": "public projection errors after raw details have been filtered",
        },
        "eval_evidence_structure": {
            "public_dev": {
                "purpose": "diagnostic public split for next-round feedback",
                "prompt_visibility": "aggregate_stats_and_bounded_failure_summary",
                "candidate_can_modify": False,
            },
            "public_accept": {
                "purpose": "runner-owned public comparison split for accept decisions",
                "prompt_visibility": "aggregate_stats_only",
                "candidate_can_modify": False,
            },
            "sealed_validation_gate": {
                "purpose": "accept-hard non-prompt validation gate summarized for feedback",
                "prompt_visibility": "gate_summary_only",
                "candidate_can_modify": False,
                "safe_status_fields": [
                    "status",
                    "checkpoint_safety_status",
                    "candidate_isolation_status",
                    "protocol_match_status",
                ],
            },
            "non_prompt_validation": {
                "purpose": "runner-owned validation evidence that is consumed by decision but not exposed as rollout detail",
                "prompt_visibility": "status_only",
                "candidate_can_modify": False,
            },
        },
        "non_prompt_validation": {
            "sealed_validation_gate": "accept_hard_condition",
            "checkpoint_safety_status": "prompt_visible_status_only",
            "candidate_isolation_status": "prompt_visible_status_only",
            "rollout_detail": "not_prompt_visible",
            "artifact_paths": "not_prompt_visible",
        },
        "candidate_must_not_request_or_modify": [
            "eval_seed",
            "holdout",
            "eval_profile",
            "checkpoint_policy",
            "accept_threshold",
            "public_accept_count",
            "sealed_validation_count",
            "runner_eval_argv",
        ],
        "prompt_history_policy": {
            "retain": [
                "round_id",
                "decision_outcome",
                "public_score",
                "comparison_summary",
                "failed_gates",
                "dominant_failure",
                "sealed_validation_gate_summary",
                "changed_components",
            ],
            "omit": [
                "unprojected_seed_identifiers",
                "per_episode_detail",
                "runner_local_artifact_paths",
                "checkpoint_policy_internals",
                "sealed_rollout_detail",
            ],
        },
        "runner_decision_authority": "runner_owned_read_only",
        "candidate_can_modify_eval": False,
    }


def _public_feedback_summary(
    public_records: Sequence[Mapping[str, Any]],
    public_context: Mapping[str, Any],
) -> dict[str, Any]:
    records = [
        record
        for record in public_records
        if isinstance(record.get("public_result"), Mapping)
        and bool(record.get("public_result"))
    ]
    latest = records[-1] if records else None
    scores = [_public_score(record) for record in records]
    scores = [score for score in scores if score is not None]
    valid_accepted_records = [
        record
        for record in records
        if _decision(record).get("accepted") is True
    ]
    accepted_scores = [
        score
        for record in valid_accepted_records
        for score in [_public_score(record)]
        if score is not None
    ]
    task_summary = public_context.get("task_public_summary")
    context_best = _optional_number(public_context.get("best_public_score"))
    best_accepted_score = max(accepted_scores) if accepted_scores else context_best
    return sanitize_public_context(
        {
            "schema_version": "true_rl_agent.public_feedback_summary.push_recovery.v1",
            "status": "available" if latest is not None else "no_prior_public_feedback",
            "round_count": len(records),
            "accepted_count": len(valid_accepted_records),
            "rejected_count": sum(
                1
                for record in records
                if _decision(record).get("accepted") is False
            ),
            "latest_round": _latest_round_summary(latest),
            "latest_public_score": (
                _public_score(latest)
                if latest is not None
                else None
            ),
            "best_public_score": best_accepted_score,
            "best_accepted_public_score": best_accepted_score,
            "best_observed_public_score": max(scores) if scores else None,
            "mean_public_score": fmean(scores) if scores else None,
            "latest_failure_mode": (
                classify_push_prompt_failure(_metrics(latest)) if latest is not None else "not_available"
            ),
            "eval_feedback": _eval_feedback_summary(records),
            "task_public_summary": task_summary if isinstance(task_summary, Mapping) else None,
        }
    )


def _runtime_feedback_summary(public_records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    latest = next(
        (
            record
            for record in reversed(public_records)
            if isinstance(record.get("runtime_activation_summary"), Mapping)
            or isinstance(record.get("train_result"), Mapping)
            or isinstance(record.get("runtime_activation_gate"), Mapping)
            or isinstance(record.get("mechanism_attribution"), Mapping)
        ),
        None,
    )
    if latest is None:
        return {
            "schema_version": "true_rl_agent.runtime_feedback_summary.push_recovery.v1",
            "status": "no_prior_runtime_feedback",
            "eval_candidate_code_policy": "checkpoint_eval_only; candidate code is not expected to load in eval",
        }
    runtime = _runtime(latest)
    train_summary = _train_summary(latest)
    candidate_summary = _mapping(train_summary.get("candidate"))
    public_metrics = _metrics(latest)
    return sanitize_public_context(
        {
            "schema_version": "true_rl_agent.runtime_feedback_summary.push_recovery.v1",
            "status": "available",
            "latest_round_id": latest.get("round_id"),
            "candidate_package_loaded": _first_present(
                train_summary.get("candidate_package_loaded"),
                train_summary.get("candidate_runtime_enabled"),
                candidate_summary.get("enabled"),
                "unknown",
            ),
            "candidate_code_loaded_in_eval": _first_present(
                train_summary.get("candidate_code_loaded_in_eval"),
                public_metrics.get("candidate_code_loaded_in_eval"),
                False,
            ),
            "eval_candidate_code_policy": "checkpoint_eval_only; candidate code is not expected to load in eval",
            "latest_runtime_activation": {
                "reward_term_calls": int(runtime.get("reward_term_calls", 0) or 0),
                "diagnostic_calls": int(runtime.get("diagnostic_calls", 0) or 0),
                "nonzero_reward_contribution": bool(runtime.get("nonzero_reward_contribution", False)),
                "reward_output_stats": _mapping(runtime.get("reward_output_stats")),
                "reward_term_activation": _runtime_reward_term_activation_summary(
                    runtime.get("component_activation_records")
                ),
                "diagnostic_outputs": _runtime_diagnostic_outputs_summary(runtime.get("diagnostic_outputs")),
                "capability_activation_records": _list_of_strings(runtime.get("capability_activation_records")),
            },
            "runtime_activation_gate": _mapping(latest.get("runtime_activation_gate")),
            "mechanism_attribution": _mapping(latest.get("mechanism_attribution")),
            "runner_error": latest.get("runner_error"),
        }
    )


def _runtime_reward_term_activation_summary(value: Any) -> dict[str, Any]:
    records = [
        item
        for item in (list(value) if isinstance(value, list | tuple) else [])
        if isinstance(item, Mapping) and item.get("component_kind") == "reward_term"
    ]
    shown: list[dict[str, Any]] = []
    for item in records:
        if len(shown) >= _RUNTIME_REWARD_TERM_MAX_FIELDS:
            break
        raw_name = _optional_text(item.get("name")) or "unnamed_reward_term"
        term_id, _ = _compact_runtime_diagnostic_key(
            raw_name,
            raw_visible=False,
            prefix="reward_term",
            index=len(shown) + 1,
        )
        projected: dict[str, Any] = {"term_id": term_id}
        public_label = _safe_runtime_identifier_label(raw_name)
        if public_label is not None:
            projected["public_label"] = public_label
        for key in (
            "observation_count",
            "nonzero_observation_count",
            "nonzero_observation_fraction",
            "max_abs_weighted_contribution",
            "last_weighted_mean",
            "effective_weight",
            "sampling_interval_steps",
        ):
            numeric = _optional_number(item.get(key))
            if numeric is not None:
                projected[key] = numeric
        next_shown = [*shown, projected]
        next_summary = _runtime_reward_term_activation_summary_value(
            total=len(records),
            shown=next_shown,
        )
        if _serialized_size(next_summary) > _RUNTIME_REWARD_TERM_BYTE_BUDGET:
            break
        shown = next_shown
    return _runtime_reward_term_activation_summary_value(total=len(records), shown=shown)


def _runtime_reward_term_activation_summary_value(
    *,
    total: int,
    shown: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": "true_rl_agent.runtime_reward_term_activation_summary.v1",
        "observation_scope": "bounded_periodic_samples_over_training_steps",
        "term_count": total,
        "shown_term_count": len(shown),
        "omitted_term_count": max(0, total - len(shown)),
        "byte_budget": _RUNTIME_REWARD_TERM_BYTE_BUDGET,
        "terms": [dict(item) for item in shown],
    }


def _runtime_diagnostic_outputs_summary(value: Any) -> dict[str, Any]:
    outputs = _mapping(value)
    total = len(outputs)
    fields: dict[str, Any] = {}
    key_metadata: dict[str, Any] = {}
    byte_budget_omitted_count = 0
    for key, item in outputs.items():
        if len(fields) >= _RUNTIME_DIAGNOSTIC_OUTPUT_MAX_FIELDS:
            break
        compact_key, metadata = _compact_runtime_diagnostic_key(
            key,
            raw_visible=False,
            prefix="diagnostic",
            index=len(fields) + 1,
        )
        compact_value = _compact_runtime_diagnostic_value(item)
        next_fields = dict(fields)
        next_fields[compact_key] = compact_value
        next_metadata = dict(key_metadata)
        if metadata is not None:
            next_metadata[compact_key] = metadata
        next_summary = _runtime_diagnostic_outputs_summary_value(
            total=total,
            fields=next_fields,
            key_metadata=next_metadata,
        )
        if _serialized_size(next_summary) > _RUNTIME_DIAGNOSTIC_OUTPUT_BYTE_BUDGET:
            byte_budget_omitted_count = total - len(fields)
            break
        fields = next_fields
        key_metadata = next_metadata
    return _runtime_diagnostic_outputs_summary_value(
        total=total,
        fields=fields,
        key_metadata=key_metadata,
        byte_budget_omitted_count=byte_budget_omitted_count,
    )


def _runtime_diagnostic_outputs_summary_value(
    *,
    total: int,
    fields: Mapping[str, Any],
    key_metadata: Mapping[str, Any],
    byte_budget_omitted_count: int = 0,
) -> dict[str, Any]:
    summary = {
        "schema_version": "true_rl_agent.runtime_diagnostic_outputs_summary.v1",
        "field_count": total,
        "shown_field_count": len(fields),
        "omitted_field_count": max(0, total - len(fields)),
        "byte_budget": _RUNTIME_DIAGNOSTIC_OUTPUT_BYTE_BUDGET,
        "fields": dict(fields),
        "overflow_policy": "preserve_insertion_order_first_fields_with_anonymous_stable_key_hashes_and_counts",
    }
    if key_metadata:
        summary["field_key_metadata"] = dict(key_metadata)
    if byte_budget_omitted_count > 0:
        summary["byte_budget_omitted_field_count"] = int(byte_budget_omitted_count)
    return summary


def _compact_runtime_diagnostic_value(value: Any, *, depth: int = 0) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, str):
        return _bounded_runtime_diagnostic_text(value)
    if isinstance(value, Mapping):
        if depth >= _RUNTIME_DIAGNOSTIC_OUTPUT_VALUE_MAX_DEPTH:
            return {
                "summary": "nested_mapping_omitted",
                "field_count": len(value),
                "max_depth": _RUNTIME_DIAGNOSTIC_OUTPUT_VALUE_MAX_DEPTH,
            }
        compact: dict[str, Any] = {}
        for key in _RUNTIME_DIAGNOSTIC_OUTPUT_STAT_KEYS:
            if key in value:
                if key == "name":
                    label = _safe_runtime_identifier_label(value[key])
                    compact[key] = label if label is not None else "anonymous_diagnostic"
                elif key == "fields" and isinstance(value[key], Mapping):
                    compact[key] = _compact_runtime_diagnostic_metric_fields(
                        value[key],
                        depth=depth + 1,
                    )
                elif key == "shape" and isinstance(value[key], list | tuple):
                    compact[key] = [
                        item
                        for item in list(value[key])[:4]
                        if isinstance(item, int | str) and not isinstance(item, bool)
                    ]
                else:
                    compact[key] = _compact_runtime_diagnostic_value(value[key], depth=depth + 1)
        for key, item in value.items():
            if key in _RUNTIME_DIAGNOSTIC_OUTPUT_STAT_KEYS:
                continue
            key_text, _ = _compact_runtime_diagnostic_key(
                key,
                raw_visible=False,
                prefix="field",
                index=len(compact) + 1,
            )
            if key_text in compact:
                continue
            if len(compact) >= _RUNTIME_DIAGNOSTIC_OUTPUT_VALUE_MAX_KEYS:
                break
            compact[key_text] = _compact_runtime_diagnostic_value(item, depth=depth + 1)
        omitted = max(0, len(value) - len(compact))
        if omitted:
            compact["omitted_field_count"] = omitted
        return compact
    if isinstance(value, list | tuple):
        if depth >= _RUNTIME_DIAGNOSTIC_OUTPUT_VALUE_MAX_DEPTH:
            return {
                "summary": "nested_sequence_omitted",
                "item_count": len(value),
                "max_depth": _RUNTIME_DIAGNOSTIC_OUTPUT_VALUE_MAX_DEPTH,
            }
        shown = [_compact_runtime_diagnostic_value(item, depth=depth + 1) for item in list(value)[:4]]
        if len(value) > len(shown):
            return {
                "shown_items": shown,
                "item_count": len(value),
                "omitted_item_count": len(value) - len(shown),
            }
        return shown
    return _bounded_runtime_diagnostic_text(str(value))


def _compact_runtime_diagnostic_metric_fields(
    value: Mapping[str, Any],
    *,
    depth: int,
) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for raw_key, item in value.items():
        if len(compact) >= _RUNTIME_DIAGNOSTIC_OUTPUT_VALUE_MAX_KEYS:
            break
        key, _ = _compact_runtime_diagnostic_key(
            raw_key,
            raw_visible=False,
            prefix="metric",
            index=len(compact) + 1,
        )
        projected = _compact_runtime_diagnostic_value(item, depth=depth + 1)
        label = _safe_runtime_identifier_label(raw_key)
        if label is not None and isinstance(projected, Mapping):
            projected = {"public_label": label, **dict(projected)}
        compact[key] = projected
    omitted = max(0, len(value) - len(compact))
    if omitted:
        compact["omitted_field_count"] = omitted
    return compact


def _compact_runtime_diagnostic_key(
    value: Any,
    *,
    raw_visible: bool = True,
    prefix: str = "field",
    index: int | None = None,
) -> tuple[str, dict[str, Any] | None]:
    text = _safe_runtime_diagnostic_key_text(value)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if not raw_visible:
        ordinal = max(1, int(index or 1))
        return f"{prefix}_{ordinal:03d}_{digest[:12]}", {
            "original_length": len(text),
            "sha256": digest,
        }
    if len(text) <= _RUNTIME_DIAGNOSTIC_OUTPUT_KEY_LIMIT:
        return text, None
    suffix = f"...#{digest[:12]}"
    prefix_length = min(
        _RUNTIME_DIAGNOSTIC_OUTPUT_KEY_PREFIX_LIMIT,
        max(1, _RUNTIME_DIAGNOSTIC_OUTPUT_KEY_LIMIT - len(suffix)),
    )
    compact = text[:prefix_length] + suffix
    return compact, {
        "original_length": len(text),
        "sha256": digest,
    }


def _safe_runtime_diagnostic_key_text(value: Any) -> str:
    text = scrub_public_paths(" ".join(str(value).split()))
    lowered = text.lower()
    if contains_forbidden_public_value_marker(text) or any(part in lowered for part in FORBIDDEN_PUBLIC_KEY_PARTS):
        return "redacted_diagnostic_key"
    return text or "unnamed_diagnostic"


def _safe_runtime_identifier_label(value: Any) -> str | None:
    text = _safe_runtime_diagnostic_key_text(value)
    if (
        len(text) <= 64
        and text.isascii()
        and text.isidentifier()
        and text != "redacted_diagnostic_key"
    ):
        return text
    return None


def _bounded_runtime_diagnostic_text(value: str) -> str:
    text = scrub_public_paths(" ".join(str(value).split()))
    if contains_forbidden_public_value_marker(text):
        return "[redacted]"
    if len(text) <= _RUNTIME_DIAGNOSTIC_OUTPUT_TEXT_LIMIT:
        return text
    return text[: _RUNTIME_DIAGNOSTIC_OUTPUT_TEXT_LIMIT - 15] + "...[truncated]"


def _serialized_size(value: Mapping[str, Any]) -> int:
    return canonical_prompt_json_bytes(value)


def _known_good_examples() -> list[dict[str, Any]]:
    return [
        {
            "kind": "RewardTerm",
            "name": "upright_velocity_margin",
            "purpose": "shape-safe reward callable using posture and velocity signals",
            "covers_reward_inputs": ["base_height", "roll", "pitch", "base_lin_vel"],
            "components_py_snippet": (
                "from true_rl_agent.candidates.components import RewardInputs, RewardTerm, TrainingComponents\n\n"
                "def upright_velocity_margin(inp: RewardInputs):\n"
                "    tilt = inp.roll * inp.roll + inp.pitch * inp.pitch\n"
                "    return inp.base_height - 0.25 * tilt\n\n"
                "def build_training_components() -> TrainingComponents:\n"
                "    return TrainingComponents(\n"
                "        name='upright_velocity_margin_candidate',\n"
                "        reward_terms=[RewardTerm(name='upright_velocity_margin', weight=1.0, fn=upright_velocity_margin)],\n"
                "        rationale='Reward upright posture using public RewardInputs.',\n"
                "    )"
            ),
            "not_a_reward_menu": True,
        },
        {
            "kind": "DiagnosticTerm",
            "name": "public_tilt_diagnostic",
            "purpose": "public diagnostic summary for future prompt feedback",
            "covers_reward_inputs": ["roll", "pitch"],
            "components_py_snippet": (
                "from true_rl_agent.candidates.components import DiagnosticTerm, TrainingComponents\n\n"
                "def public_tilt_diagnostic(inp):\n"
                "    return {'mean_tilt_proxy': inp.roll * inp.roll + inp.pitch * inp.pitch}\n\n"
                "def build_training_components() -> TrainingComponents:\n"
                "    return TrainingComponents(\n"
                "        name='tilt_diagnostic_candidate',\n"
                "        diagnostics=[DiagnosticTerm(name='public_tilt_diagnostic', fn=public_tilt_diagnostic)],\n"
                "        rationale='Expose a public tilt diagnostic for later rounds.',\n"
                "    )"
            ),
            "not_accept_sufficient": True,
        },
        {
            "kind": "ScheduleTerm",
            "name": f"{REWARD_WEIGHT_SCALE_KEY}_api_reference",
            "purpose": "legal ScheduleTerm API example; parameter-only changes are not accept-sufficient",
            "covers_reward_inputs": [],
            "components_py_snippet": (
                "from true_rl_agent.candidates.components import ScheduleTerm, TrainingComponents\n\n"
                "def build_training_components() -> TrainingComponents:\n"
                "    return TrainingComponents(\n"
                "        name='scale_reference_candidate',\n"
                f"        schedules=[ScheduleTerm(name='scale', key='{REWARD_WEIGHT_SCALE_KEY}', value=1.0)],\n"
                "        rationale='API reference only; combine with code-level reward logic for policy improvement.',\n"
                "    )"
            ),
            "parameter_only_not_accept_sufficient": True,
        },
    ]


def _selection_guidance() -> list[str]:
    return [
        "Prefer small code-level RewardTerm changes over parameter-only changes.",
        "Use public feedback to target real failure modes such as falls, drift after push, timeout, or motion-quality gate failures.",
        "If runtime feedback says reward terms did not run or produced zero contribution, fix loading, shape, finite values, or activation first.",
        "Diagnostics are useful public evidence for later rounds but are not sufficient for policy improvement accept.",
        (
            "Treat push_active and push_vec_public as current-step detected perturbation signals, not a guaranteed "
            "post-push window; use posture, velocity, and position inputs for continuing recovery."
        ),
        "Do not modify eval protocol, success metric, training budget, seed policy, checkpoint policy, runner code, PPO, or task invariants.",
        "Do not depend on filesystem access, unprojected seed values, non-prompt artifacts, environment objects, optimizer state, or eval internals.",
    ]


def _latest_round_summary(
    record: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if record is None:
        return None
    decision = _decision(record)
    candidate = _mapping(record.get("candidate"))
    return {
        "round_id": record.get("round_id"),
        "candidate_id": candidate.get("candidate_id"),
        "changed_components": candidate.get("changed_components"),
        "decision_outcome": decision.get("outcome"),
        "decision_reason": decision.get("reason"),
        "accepted": decision.get("accepted"),
        "surface_classification": _mapping(record.get("candidate_surface")).get("classification"),
    }


def _eval_feedback_summary(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    latest = records[-1] if records else None
    if latest is None:
        return {
            "schema_version": "true_rl_agent.eval_feedback_summary.push_recovery.v1",
            "status": "no_prior_eval_feedback",
        }
    metrics = _metrics(latest)
    return sanitize_public_context(
        {
            "schema_version": "true_rl_agent.eval_feedback_summary.push_recovery.v1",
            "status": "available",
            "latest_round_id": latest.get("round_id"),
            "latest_split_summary": _latest_split_summary(metrics),
            "failed_gate_trend": _failed_gate_trend(records),
            "sealed_validation": _sealed_validation_trend(records),
            "non_prompt_validation": _non_prompt_validation_summary(metrics),
        }
    )


def _latest_split_summary(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "public_dev": _split_score_summary(metrics.get("public_dev")),
        "public_accept": _split_score_summary(metrics.get("public_accept")),
    }


def _split_score_summary(value: Any) -> dict[str, Any]:
    split = _mapping(value)
    stats = _score_stats_summary(split.get("score_stats"))
    return {"status": "available", "score_stats": stats} if stats else {"status": "not_available"}


def _score_stats_summary(value: Any) -> dict[str, Any]:
    stats = _mapping(value)
    if not stats:
        return {}
    summary: dict[str, Any] = {}
    for key in ("count", "episode_count"):
        parsed = _optional_int(stats.get(key))
        if parsed is not None:
            summary[key] = parsed
    for key in ("mean", "std", "standard_error"):
        parsed = _optional_number(stats.get(key))
        if parsed is not None:
            summary[key] = parsed
    return summary


def _failed_gate_trend(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    failed_counts: dict[str, int] = {}
    rounds_with_failures = 0
    latest_failed: list[dict[str, str]] = []
    for record in records:
        failed = _failed_gate_summary(_decision(record).get("gate_statuses"))
        if failed:
            rounds_with_failures += 1
            latest_failed = failed
        for gate in failed:
            name = gate["name"]
            failed_counts[name] = failed_counts.get(name, 0) + 1
    return {
        "rounds_with_failures": rounds_with_failures,
        "failed_counts": dict(sorted(failed_counts.items())),
        "latest_failed_gates": latest_failed,
    }


def _failed_gate_summary(value: Any) -> list[dict[str, str]]:
    statuses = _mapping(value)
    failed: list[dict[str, str]] = []
    for raw_name, raw_status in statuses.items():
        name = _optional_text(raw_name)
        status = _optional_text(raw_status)
        if not name or not status or status == "pass":
            continue
        failed.append({"name": name, "status": status})
        if len(failed) >= 8:
            break
    return failed


def _sealed_validation_trend(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    latest: dict[str, Any] | None = None
    failure_count = 0
    observed_count = 0
    for record in records:
        summary = _sealed_validation_gate_prompt_summary(
            _metrics(record).get("sealed_validation_gate")
        )
        if not summary:
            continue
        observed_count += 1
        latest = summary
        if summary.get("status") != "pass":
            failure_count += 1
    if latest is None:
        return {
            "status": "not_available",
            "observed_count": 0,
            "failure_count": 0,
            "latest": None,
        }
    return {
        "status": latest.get("status"),
        "observed_count": observed_count,
        "failure_count": failure_count,
        "latest": latest,
    }


def _non_prompt_validation_summary(metrics: Mapping[str, Any]) -> dict[str, Any]:
    sealed = _sealed_validation_gate_prompt_summary(metrics.get("sealed_validation_gate"))
    if not sealed:
        return {"status": "not_available"}
    return {
        "status": sealed.get("status"),
        "sealed_validation_gate": sealed.get("status"),
        "checkpoint_safety_status": sealed.get("checkpoint_safety_status"),
        "candidate_isolation_status": sealed.get("candidate_isolation_status"),
        "visibility": "status_only",
    }


def _sealed_validation_gate_prompt_summary(value: Any) -> dict[str, Any] | None:
    gate = _mapping(value)
    if not gate:
        return None
    summary: dict[str, Any] = {}
    for key in (
        "status",
        "reason",
        "checkpoint_safety_status",
        "candidate_isolation_status",
        "protocol_match_status",
        "checkpoint_hash_match_status",
        "eval_protocol_hash_match_status",
    ):
        text = _optional_text(gate.get(key))
        if text is not None:
            summary[key] = text
    return summary or None


def _decision(
    record: Mapping[str, Any],
) -> Mapping[str, Any]:
    return _mapping(record.get("decision"))


def _public_score(
    record: Mapping[str, Any] | None,
) -> float | None:
    if record is None:
        return None
    decision_score = _decision(record).get("public_score")
    if not isinstance(decision_score, bool) and isinstance(
        decision_score,
        int | float,
    ):
        return float(decision_score)
    return None


def _metrics(record: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if record is None:
        return {}
    public_result = _mapping(record.get("public_result"))
    return _mapping(public_result.get("metrics"))


def _runtime(record: Mapping[str, Any]) -> Mapping[str, Any]:
    runtime = _mapping(record.get("runtime_activation_summary"))
    if runtime:
        return runtime
    return _mapping(_train_summary(record).get("runtime_activation_summary"))


def _train_summary(record: Mapping[str, Any]) -> Mapping[str, Any]:
    train_result = _mapping(record.get("train_result"))
    return _mapping(train_result.get("summary"))


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list_of_strings(value: Any) -> list[str]:
    if not isinstance(value, list | tuple):
        return []
    return [str(item) for item in value if isinstance(item, str)]


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    return " ".join(str(value).split())


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _optional_number(value: Any) -> int | float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return None


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None
