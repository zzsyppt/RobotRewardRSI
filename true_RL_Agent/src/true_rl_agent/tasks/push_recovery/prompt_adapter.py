"""Push recovery task-owned prompt semantics."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Any, Mapping, Sequence

from true_rl_agent.agent_core.context import sanitize_public_context
from true_rl_agent.candidates.components import RewardInputs
from true_rl_agent.tasks.prompt_adapter import sanitize_task_prompt_sections
from true_rl_agent.tasks.push_recovery.config import PushRecoveryTaskConfig
from true_rl_agent.tasks.push_recovery.metrics import classify_push_failure


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

_FIELD_METADATA: dict[str, dict[str, str]] = {
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
        "semantics": "bool-like runner-owned indicator for a recent push event",
        "units": "bool_like",
        "coordinate_frame": "not_applicable",
        "source_status": "runner_owned_push_signal_currently_zero_fallback_in_generic_runtime_adapter",
        "fallback_behavior": "zero fallback when no push event signal is available",
        "candidate_usage_guidance": "treat as weak public evidence; do not rely on it as a guaranteed push detector",
        "known_limitations": "current generic runtime path can provide only zero fallback",
    },
    "push_vec_public": {
        "semantics": "public xy velocity delta for a runner-owned push event when instrumented",
        "units": "m/s",
        "coordinate_frame": "world_xy_velocity_delta",
        "source_status": "runner_owned_push_signal_currently_zero_fallback_in_generic_runtime_adapter",
        "fallback_behavior": "zero vector fallback when no push event signal is available",
        "candidate_usage_guidance": "not guaranteed to be nonzero; combine with posture and velocity signals",
        "known_limitations": "current generic runtime path can provide only zero fallback",
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
        sections = {
            "task": self._task_section(),
            "reward_inputs_contract": _reward_inputs_contract(),
            "public_eval_contract": _public_eval_contract(),
            "public_feedback_summary": _public_feedback_summary(public_records, public_context),
            "runtime_feedback_summary": _runtime_feedback_summary(public_records),
            "known_good_examples": _known_good_examples(),
            "selection_guidance": _selection_guidance(),
        }
        return sanitize_task_prompt_sections(sections)

    def _task_section(self) -> dict[str, Any]:
        config = self.config
        return {
            "schema_version": "true_rl_agent.task_prompt.push_recovery.v1",
            "task_id": self.task_id,
            "name": "Unitree Push Recovery",
            "prompt_adapter": "push_recovery_task_prompt_adapter",
            "objective": (
                "Improve a Unitree locomotion policy so it remains upright and recovers stable "
                "velocity after runner-owned lateral pushes."
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
            "fixed_train_protocol_summary": {
                "baseline_train_distribution": "runner_owned_target_distribution",
                "push_distribution": "fixed_target_push",
                "budget_steps": config.budget_steps,
                "train_num_envs": config.train_num_envs,
                "train_max_iterations": config.train_max_iterations,
                "train_steps_per_iteration": config.train_steps_per_iteration,
                "llm_schedule_control": "not_prompt_writable",
            },
            "fixed_eval_protocol_summary": {
                "eval_protocol": "fixed_public_eval_plus_non_prompt_checks",
                "success_predicate": config.success_predicate,
                "seed_visibility": config.eval_seed_visibility,
                "eval_num_envs": config.eval_num_envs,
                "public_seed_count": config.public_seed_count,
                "public_accept_seed_count": config.public_accept_seed_count,
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


def _reward_inputs_contract() -> dict[str, Any]:
    fields = {}
    for name in RewardInputs.field_names():
        metadata = dict(_FIELD_METADATA[name])
        fields[name] = {
            "name": name,
            "shape": list(_FIELD_SHAPES[name]),
            "dtype_or_tensor_kind": "tensor_like_numeric",
            **metadata,
        }
    return {
        "schema_version": "true_rl_agent.reward_inputs_contract.push_recovery.v1",
        "task_id": "unitree_push_recovery",
        "field_order": list(RewardInputs.field_names()),
        "fields": fields,
        "candidate_does_not_receive": [
            "env",
            "runner",
            "optimizer",
            "checkpoint policy state",
            "eval config",
            "success metric internals",
            "raw seed identifiers",
        ],
    }


def _public_eval_contract() -> dict[str, Any]:
    return {
        "schema_version": "true_rl_agent.public_eval_contract.push_recovery.v1",
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
            "public_accept": "public accept split aggregate when present; raw seed identifiers are omitted",
            "failure_breakdown": "public failure-mode rates such as fall, push_drift, timeout, or task-specific labels",
            "motion_quality": "public secondary motion summaries such as height, tilt, drift, action magnitude",
            "budget_accounting": "read-only training budget accounting gate",
            "motion_quality_gate": "read-only public sanity gate for unsafe or degenerate motion",
            "acceptance": "read-only score source and required public gates; candidate cannot edit this",
            "errors": "public projection errors after raw details have been filtered",
        },
        "runner_decision_authority": "runner_owned_read_only",
        "candidate_can_modify_eval": False,
    }


def _public_feedback_summary(
    public_records: Sequence[Mapping[str, Any]],
    public_context: Mapping[str, Any],
) -> dict[str, Any]:
    records = [_record for _record in public_records if _record.get("public_result") is not None]
    latest = records[-1] if records else None
    scores = [_public_score(record) for record in records]
    scores = [score for score in scores if score is not None]
    task_summary = public_context.get("task_public_summary")
    return sanitize_public_context(
        {
            "schema_version": "true_rl_agent.public_feedback_summary.push_recovery.v1",
            "status": "available" if latest is not None else "no_prior_public_feedback",
            "round_count": len(records),
            "accepted_count": sum(1 for record in records if _decision(record).get("accepted") is True),
            "rejected_count": sum(1 for record in records if _decision(record).get("accepted") is False),
            "latest_round": _latest_round_summary(latest),
            "latest_public_score": _public_score(latest) if latest is not None else None,
            "best_public_score": max(scores) if scores else None,
            "mean_public_score": fmean(scores) if scores else None,
            "latest_failure_mode": (
                classify_push_failure(_metrics(latest)) if latest is not None else "not_available"
            ),
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
                "diagnostic_outputs": _mapping(runtime.get("diagnostic_outputs")),
                "capability_activation_records": _list_of_strings(runtime.get("capability_activation_records")),
            },
            "mechanism_attribution": _mapping(latest.get("mechanism_attribution")),
            "runner_error": latest.get("runner_error"),
        }
    )


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
            "name": "reward_weight_scale_api_reference",
            "purpose": "legal ScheduleTerm API example; parameter-only changes are not accept-sufficient",
            "covers_reward_inputs": [],
            "components_py_snippet": (
                "from true_rl_agent.candidates.components import ScheduleTerm, TrainingComponents\n\n"
                "def build_training_components() -> TrainingComponents:\n"
                "    return TrainingComponents(\n"
                "        name='scale_reference_candidate',\n"
                "        schedules=[ScheduleTerm(name='scale', key='reward_weight_scale', value=1.0)],\n"
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
        "Treat push_active and push_vec_public as weak public signals in the current generic runtime path; combine them with posture and velocity features.",
        "Do not modify eval protocol, success metric, training budget, seed policy, checkpoint policy, runner code, PPO, or task invariants.",
        "Do not depend on filesystem access, raw seeds, non-prompt artifacts, environment objects, optimizer state, or eval internals.",
    ]


def _latest_round_summary(record: Mapping[str, Any] | None) -> dict[str, Any] | None:
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


def _decision(record: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(record.get("decision"))


def _public_score(record: Mapping[str, Any] | None) -> float | None:
    if record is None:
        return None
    public_result = _mapping(record.get("public_result"))
    value = public_result.get("public_score")
    if isinstance(value, int | float):
        return float(value)
    decision_score = _decision(record).get("public_score")
    if isinstance(decision_score, int | float):
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


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None
