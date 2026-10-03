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
from true_rl_agent.agent_core.generation_contract import latest_nonaccepted_trial, project_public_trial_reason
from true_rl_agent.agent_core.public_safety import (
    contains_forbidden_public_key_marker,
    contains_forbidden_public_value_marker,
    redact_authorization_credential_markers,
    sanitize_untrusted_public_diagnostic,
    scrub_public_paths,
)
from true_rl_agent.candidates.schedule_contract import REWARD_WEIGHT_SCALE_KEY
from true_rl_agent.candidates.mechanism import RuntimeActivationEvidence
from true_rl_agent.prompting.serialization import canonical_prompt_json_bytes
from true_rl_agent.prompting.capacity import build_prompt_capacity_receipt
from true_rl_agent.tasks.prompt_adapter import (
    MAX_TASK_PROMPT_SECTIONS_BYTES,
    PromptPublicHistoryProjection,
    sanitize_task_prompt_sections,
)
from true_rl_agent.tasks.push_recovery.config import PushRecoveryTaskConfig
from true_rl_agent.tasks.push_recovery.exploration import public_dev_comparison, public_dev_parent_reference
from true_rl_agent.tasks.push_recovery.reward_inputs import PushRecoveryV2RewardInputs
from true_rl_agent.tasks.push_recovery.metrics import classify_push_prompt_failure
from true_rl_agent.tasks.push_recovery.behavior import compare_push_behavior, project_push_behavior
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
        recovery_v2 = self.config.push_protocol is not None
        if recovery_v2:
            public_records = self.project_prompt_public_history(
                public_records=public_records, initial_anchor_public_evidence=None,
            ).records
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
            "reward_inputs_contract": _v2_reward_inputs_contract() if recovery_v2 else _reward_inputs_contract(),
            "public_eval_contract": _v2_public_eval_contract() if recovery_v2 else _public_eval_contract(),
            "public_feedback_summary": _v2_feedback_summary(
                safe_public_records, initial_anchor=_mapping(safe_public_context.get("initial_anchor_public_evidence")),
            ) if recovery_v2 else _public_feedback_summary(
                safe_public_records,
                safe_public_context,
            ),
            "runtime_feedback_summary": _runtime_feedback_summary(safe_public_records),
            # Bootstrap examples teach the component API. Once evaluated history
            # exists, the bounded source catalog and experiment memory carry the
            # real candidate lineage, which is both more relevant and avoids
            # repeating static tutorial code beside empirical feedback.
            "known_good_examples": _known_good_examples() if not recovery_v2 and not safe_public_records else [],
            "selection_guidance": _v2_selection_guidance() if recovery_v2 else _selection_guidance(),
        }
        if self.config.push_evaluation_protocol is not None:
            sections["task"].update(
                schema_version="true_rl_agent.task_prompt.push_recovery.v4",
                objective="Recover stable standing after a complete finite external-force sequence.",
                success_authority="The task controller verifies every planned force pulse, then requires one fresh stable dwell after the last pulse and a stable final state, without a fall. Displacement is diagnostic only. Reward values do not define success.",
            )
            sections["reward_inputs_contract"]["training_scope"] = (
                "These inputs describe the unchanged training perturbations. push_active and push_vec_public report an already executed training velocity increment in m/s; they do not describe evaluation force pulses."
            )
            sections["public_eval_contract"].update(
                schema_version="true_rl_agent.public_eval_contract.push_recovery.v4",
                recovery_evidence_gate="read-only status; validates pulse completion and verification separately from whole-sequence recovery",
                recovery_scope="one fresh recovery window after the last complete planned external-force pulse",
            )
            sections["public_feedback_summary"]["schema_version"] = "true_rl_agent.public_feedback_summary.push_recovery.v3"
            sections["selection_guidance"][-1] = (
                "success_recovered requires every planned force pulse to complete and be verified, followed by a fresh stable dwell after the last pulse and a stable final state without a fall; reward shaping cannot change that definition."
            )
        try:
            return sanitize_task_prompt_sections(sections)
        except ValueError as exc:
            if "task prompt sections exceed" not in str(exc):
                raise
        return sanitize_task_prompt_sections(
            _minimum_core_push_prompt_sections(sections)
        )

    def build_prompt_capacity_preflight_sections(self) -> Mapping[str, Any]:
        """Return Push's task-owned minimum shape for startup validation."""

        sections = self.build_prompt_sections(
            public_records=(),
            public_context={},
        )
        return sanitize_task_prompt_sections(
            _minimum_core_push_prompt_sections(sections)
        )

    def prompt_public_record_scrub_policy(self) -> Mapping[str, Sequence[str]]:
        reserved_fields = reserved_zero_fallback_reward_input_fields()
        return {
            "blocked_key_parts": reserved_fields,
            "blocked_text_parts": reserved_fields,
        }

    def project_prompt_public_history(
        self,
        *,
        public_records: Sequence[Mapping[str, Any]],
        initial_anchor_public_evidence: Mapping[str, Any] | None,
    ) -> PromptPublicHistoryProjection:
        if self.config.push_protocol is not None:
            anchor = _mapping(_mapping(initial_anchor_public_evidence).get("public_result"))
            return PromptPublicHistoryProjection(
                records=tuple(_v2_prompt_record(record) for record in public_records),
                public_score_source="public_dev.score_stats.mean",
                initial_anchor_score=_optional_number(_mapping(
                    _mapping(_mapping(anchor.get("metrics")).get("public_dev")).get("score_stats")
                ).get("mean")),
            )
        return PromptPublicHistoryProjection(
            records=tuple(dict(record) for record in public_records),
        )

    def nonaccepted_authoring_tiebreak_key(
        self,
        public_record: Mapping[str, Any],
    ) -> tuple[float, ...]:
        _ = public_record
        return ()

    def _task_section(self) -> dict[str, Any]:
        config = self.config
        if config.push_protocol is not None:
            return _v2_task_section(config)
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



def _v2_task_section(config: PushRecoveryTaskConfig) -> dict[str, Any]:
    physical = config.push_protocol.resolve(config.push_robot_scale_m)
    return {
        "schema_version": "true_rl_agent.task_prompt.push_recovery.v3",
        "task_id": "unitree_push_recovery",
        "name": "Unitree Complete Push Recovery",
        "objective": "Recover stable standing after every actual runner-injected planar push, including repeated pushes.",
        "robot_or_environment": {"cell_variant_id": config.cell_variant_id, "runtime_adapter_id": config.runtime_adapter_id},
        "candidate_capability_profile": "push_complete_recovery_v2",
        "candidate_capabilities": ["reward_term", "diagnostic_term", "reward_weight_scale_schedule"],
        "reward_program": {
            "authority": "complete_replacement",
            "composition": "All RewardTerm outputs form the complete reward program; upstream rewards are disabled.",
            "integration": "Return per-environment reward rates. The task RewardManager sums weighted rates and multiplies the total by control dt exactly once, before reset. Do not multiply by dt in candidate code.",
            "schedule": "reward_weight_scale changes the program weights only; it cannot change task timing or pushes.",
        },
        "spatial_scale": {key: physical[key] for key in ("robot_scale_m", "velocity_scale_mps", "time_scale_s")},
        "scale_definition": "L is the bound robot's nominal base height in metres; velocity scale=sqrt(g*L), time scale=sqrt(L/g).",
        "fixed_train_protocol_summary": {"budget_steps": config.budget_steps, "train_num_envs": config.train_num_envs,
            "train_max_iterations": config.train_max_iterations, "train_steps_per_iteration": config.train_steps_per_iteration},
        "fixed_eval_protocol_summary": {"success_predicate": config.success_predicate,
            "success_label": "success_recovered", "prompt_score_source": "public_dev.score_stats.mean"},
        "success_authority": "The task controller owns actual injection verification, complete recovery conditions, dwell, repeated-event accounting and success/failure counts. Reward values do not define success.",
        "candidate_must_not_change": ["task controller", "push schedule or distribution", "success predicates or counts",
            "evaluation", "training budget", "seeds", "checkpoints", "PPO", "robot or action ABI"],
    }


def _v2_reward_inputs_contract() -> dict[str, Any]:
    contract = _reward_inputs_contract()
    fields = contract["fields"]
    additions = {
        "nominal_base_height": (["num_envs"], "m", "bound robot nominal base height L; a spatial normalization, not a success threshold"),
        "base_height_relative": (["num_envs"], "unitless", "origin-relative base height divided by nominal_base_height"),
        "base_lin_vel_scaled": (["num_envs", "3"], "unitless", "world-frame linear velocity divided by sqrt(g*L)"),
        "base_ang_vel_scaled": (["num_envs", "3"], "unitless", "world-frame angular velocity multiplied by sqrt(L/g)"),
        "terminated_non_timeout": (["num_envs"], "bool_like", "task termination mask before reset; timeout alone is excluded"),
    }
    for name, (shape, units, semantics) in additions.items():
        fields[name] = {"name": name, "shape": shape, "units": units, "semantics": semantics,
            "dtype_or_tensor_kind": "tensor_like_numeric", "source_status": "task_owned_current_step"}
    fields["base_height"] = {**fields["base_height"], "semantics": "root world z minus environment origin z",
        "source_status": "observed", "fallback_behavior": "required by the V2 runtime",
        "known_limitations": "metres; use base_height_relative to compare robot sizes"}
    for name in ("base_lin_vel", "base_ang_vel"):
        fields[name] = {**fields[name], "coordinate_frame": "robot_base_frame",
            "semantics": "raw SI velocity in the robot base frame; the corresponding scaled field uses world coordinates",
            "known_limitations": "do not mix base-frame raw vectors with world-frame scaled vectors"}
    fields["push_active"] = {"name": "push_active", "shape": ["num_envs"], "units": "bool_like",
        "semantics": "one only on the current step of an actual task-controller injected push; zero otherwise",
        "source_status": "verified_current_step_event", "known_limitations": "not latched through recovery and never inferred from autonomous robot motion"}
    fields["push_vec_public"] = {"name": "push_vec_public", "shape": ["num_envs", "2"], "units": "m/s",
        "coordinate_frame": "world_xy_velocity_increment", "source_status": "verified_current_step_event",
        "semantics": "actual injected planar velocity increment for this step; zero when push_active is zero",
        "known_limitations": "already occurred event only; no future timing, direction, severity or schedule"}
    contract.update(schema_version="true_rl_agent.reward_inputs_contract.push_recovery.v2",
        input_type="PushRecoveryV2RewardInputs", field_order=list(PushRecoveryV2RewardInputs.field_names()))
    return contract


def _v2_public_eval_contract() -> dict[str, Any]:
    return {
        "schema_version": "true_rl_agent.public_eval_contract.push_recovery.v3",
        "score_source": "public_dev.score_stats.mean",
        "public_dev": "canonical task phase summary: recovery success statistics and aggregated recovery_counts",
        "primary_success_label": "success_recovered",
        "recovery_evidence_gate": "read-only status; validates task-owned episode and event counts",
        "sealed_validation_gate": {"prompt_visibility": "status_only"},
        "candidate_can_modify_eval": False,
        "history_policy": "No acceptance scores, sealed rollout details, seed identifiers or arbitrary failure text enter authoring feedback.",
        "trial_memory_semantics": "objective_delta is a descriptive candidate-minus-parent public-dev score; guard_delta is not applicable; runtime_effect is not_assessed because Push does not claim component-level causal attribution.",
    }


def _v2_selection_guidance() -> list[str]:
    return [
        "Supply a complete recovery reward program through RewardTerm callables, with optional DiagnosticTerm and reward_weight_scale ScheduleTerm.",
        "Use morphology-scaled height and world velocities for portable recovery shaping; raw SI joint and action signals retain their declared units.",
        "Target public-dev failures and inspect the same trained program's runtime activation before changing reward logic.",
        "push_active and push_vec_public describe only an already injected current-step event; use posture and velocity for later recovery.",
        "success_recovered requires task-owned complete recovery and dwell after every planned push; reward shaping cannot change that definition.",
    ]


_RECOVERY_COUNT_FIELDS = ("episodes", "successes", "eligible_episode_count", "control_episode_count",
    "planned_push_event_count", "pushed_episode_count", "push_event_count", "recovered_event_count", "invalid_injection_count")
_FORCE_COUNT_FIELDS = tuple(name for name in _RECOVERY_COUNT_FIELDS if name != "recovered_event_count") + (
    "completed_push_event_count", "verified_push_event_count", "sequence_recovered_count")
_RECOVERY_LABELS = ("success_recovered", "not_recovered", "fall_forward", "fall_backward", "fall_side",
    "height_collapse", "out_of_bounds", "invalid_push_injection", "no_push_control")


def _v2_dev_summary(value: Any) -> dict[str, Any]:
    """Compact the canonical project_push_split output; never read another split."""
    split = _mapping(value)
    if not split:
        return {}
    counts, labels, strata = {}, {}, {}
    seeds = split.get("seed_results")
    rows = seeds if isinstance(seeds, list | tuple) else [split]
    for row in rows:
        row = _mapping(row)
        recovery_counts = _mapping(row.get("recovery_counts"))
        force_sequence = "sequence_recovered_count" in recovery_counts
        for source, names, target in (
            (recovery_counts, _FORCE_COUNT_FIELDS if force_sequence else _RECOVERY_COUNT_FIELDS, counts),
            (_mapping(_mapping(row.get("failure_breakdown")).get("counts")),
             tuple(label for label in _RECOVERY_LABELS if label != "out_of_bounds") if force_sequence else _RECOVERY_LABELS, labels),
        ):
            for name in names:
                count = source.get(name)
                if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
                    target[name] = target.get(name, 0) + count
        for name in ("low", "medium", "high", "repeated"):
            stratum = _mapping(_mapping(row.get("stratum_results")).get(name))
            episodes, successes = stratum.get("episodes"), stratum.get("successes")
            if (isinstance(episodes, int) and not isinstance(episodes, bool) and episodes > 0
                    and isinstance(successes, int) and not isinstance(successes, bool) and 0 <= successes <= episodes):
                target = strata.setdefault(name, {"episodes": 0, "successes": 0})
                target["episodes"] += episodes
                target["successes"] += successes
    for stratum in strata.values():
        stratum["n_over_N"] = f"{stratum['successes']}/{stratum['episodes']}"
    result = {"score_stats": _score_stats_summary(split.get("score_stats")), "recovery_counts": counts,
        "stratum_results": strata,
        "failure_breakdown": {"counts": labels, "total": counts.get("episodes", sum(labels.values()))},
        "dominant_outcome": max(labels, key=labels.get) if labels and max(labels.values()) > 0 else "not_available"}
    if isinstance(split.get("behavior"), Mapping):
        result["behavior"] = project_push_behavior(split["behavior"])
    if "description_status" in split:
        result["description_status"] = split["description_status"]
    return result


def _v2_prompt_record(record: Mapping[str, Any]) -> dict[str, Any]:
    _validate_trajectory_record(record)
    # Shared history freezes every field outside these three feedback domains.
    projected = {key: value for key, value in record.items() if key not in {"public_result", "decision", "generation"}}
    metrics = _metrics(record)
    dev = _v2_dev_summary(metrics.get("public_dev"))
    score = _optional_number(_mapping(dev.get("score_stats")).get("mean"))
    decision = _decision(record)
    reason = ("decision_record_invalid" if decision.get("reason") == "decision_record_invalid" else
        "runner_accepted_candidate" if decision.get("accepted") is True else "runner_did_not_accept_candidate")
    projected["decision"] = {"accepted": decision.get("accepted"), "outcome": decision.get("outcome"), "reason": reason,
        "surface_classification": decision.get("surface_classification"),
        "public_score": score, "comparison": {"score_source": "public_dev.score_stats.mean", "candidate_score": score}}
    if isinstance(record.get("public_result"), Mapping):
        projected["public_result"] = {"status": _mapping(record.get("public_result")).get("status"), "public_score": score,
            "metrics": {"public_dev": dev,
                **({"trajectory_phase_binding": metrics["trajectory_phase_binding"]} if "trajectory_phase_binding" in metrics else {}),
                **{key: {"status": _mapping(metrics.get(key)).get("status", "not_available")}
                for key in ("recovery_evidence_gate", "sealed_validation_gate")}}}
    generation = _mapping(record.get("generation"))
    if generation:
        visible = {key: generation[key] for key in ("schema_version", "generation_id", "generation_index", "generation_outcome") if key in generation}
        reference = public_dev_parent_reference(generation)
        if reference:
            visible["public_dev_parent_reference"] = reference
        for section, names in (("selection", ("selected_trial_id",)),
                ("formal", ("accepted", "decision_outcome", "selected_lane_id", "selected_trial_id"))):
            if isinstance(generation.get(section), Mapping):
                visible[section] = {name: generation[section][name] for name in names if name in generation[section]}
        if "formal" in generation and generation["formal"] is None:
            # No formal round means there is no single candidate to attribute
            # to this generation; shared memory keeps the sibling trials instead.
            visible["formal"] = None
        visible["trials"] = []
        for trial in generation.get("trials", ()):
            item = {key: trial[key] for key in ("schema_version", "lane_id", "trial_id", "trial_status", "candidate",
                "candidate_surface", "mechanism_operation_delta", "runtime_activation_summary", "authoring_status", "lane_intent") if key in trial}
            host_reason = project_public_trial_reason(trial.get("status_reason"))
            if host_reason is not None:
                item["status_reason"] = host_reason
            if isinstance(trial.get("train"), Mapping):
                item["train"] = {"completed": True}
            if isinstance(trial.get("public_result"), Mapping):
                item["public_result"] = _v2_prompt_record(trial).get("public_result", {})
            phase = _mapping(trial.get("public_dev"))
            if phase:
                item["public_dev"] = {"public_summary": _v2_dev_summary(phase.get("public_summary")),
                    **{k: phase[k] for k in ("checkpoint_hash", "evaluation_plan_hash") if k in phase}}
                behavior = item["public_dev"]["public_summary"].get("behavior")
                # Consume trusted parent evidence before this task's compact
                # history drops it. Reprojecting a view preserves the bound diff.
                if (behavior is not None and behavior.get("schema_version") != "true_rl_agent.push_recovery.behavior.public.v4"
                        and ("parent_current_best" in generation or "parent_public_dev" in generation)):
                    behavior["parent_diff"] = compare_push_behavior(behavior,
                        _mapping(generation.get("parent_public_dev")).get("behavior"),
                        expected_parent_checkpoint_hash=_mapping(generation.get("parent_current_best")).get("checkpoint_hash"))
                    item["public_dev"]["public_summary"]["behavior"] = project_push_behavior(behavior)
            comparison = public_dev_comparison(generation, trial)
            item["prompt_outcome"] = {"runtime_effect": "not_assessed"}
            if comparison:
                item["prompt_outcome"]["objective_delta"] = comparison["delta"]
            if ("public_result" in projected and _mapping(trial.get("candidate")).get("candidate_id") is not None
                    and _mapping(trial.get("candidate")).get("candidate_id") == _mapping(record.get("candidate")).get("candidate_id")):
                item["public_result"] = projected["public_result"]
            visible["trials"].append(item)
        parent = _mapping(_mapping(generation.get("parent_public_dev")).get("behavior"))
        if parent.get("schema_version") == "true_rl_agent.push_recovery.behavior.public.v4":
            best = _mapping(generation.get("parent_current_best"))
            expected = {"subject_kind": best.get("anchor_kind"), "subject_id": best.get("anchor_id"),
                **{k: best.get(k) for k in ("checkpoint_hash", "subject_manifest_hash", "program_source_checksum",
                                          "training_binding_hash", "evaluation_binding_hash")}}
            if any(parent["domain"].get(k) != v for k, v in expected.items()):
                raise ValueError("Push trajectory generation parent identity mismatch")
            visible["parent_trajectory_source"] = _trajectory_source(parent)
        elif "parent_trajectory_source" in generation:
            visible["parent_trajectory_source"] = generation["parent_trajectory_source"]
        projected["generation"] = visible
    return projected


def _v2_feedback_summary(
    records: Sequence[Mapping[str, Any]], *, initial_anchor: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    for record in records:
        _validate_trajectory_record(record)
    selection, source = _feedback_source(records)
    if source is None:
        feedback = _v2_reference_feedback(initial_anchor)
    else:
        feedback = _v2_trial_feedback(selection, source)
        feedback.update({
            "latest_round": {"round_id": source.get("trial_id", source.get("round_id")),
                "candidate_id": _mapping(source.get("candidate")).get("candidate_id"),
                "status_reason": project_public_trial_reason(source.get("status_reason"))},
            "recovery_evidence_gate": {"status": _mapping(_metrics(source).get("recovery_evidence_gate")).get("status", "not_available")},
            "sealed_validation_gate": {"status": _mapping(_metrics(source).get("sealed_validation_gate")).get("status", "not_available")},
        })
        if selection == "latest_trained_nonaccepted":
            # The sibling's failures cannot describe the accepted parent. Read
            # both from their own dev projection, not the memory-wide aggregate.
            current = _accepted_feedback_source(records)
            current_feedback = (_v2_trial_feedback("accepted_current_best", current)
                if current is not None else _v2_reference_feedback(initial_anchor))
            if current_feedback.get("public_dev"):
                if "behavior" in current_feedback["public_dev"]:
                    prior = current_feedback["public_dev"]["behavior"]
                    current_feedback["public_dev"]["behavior"] = {"status": "omitted", "reason": "behavior_budget_reserved_for_selected_feedback_trial",
                        **({"source": _trajectory_source(prior)} if prior.get("schema_version") == "true_rl_agent.push_recovery.behavior.public.v4" else {})}
                feedback["current_best"] = current_feedback
    dev = feedback.get("public_dev")
    if isinstance(dev, dict) and "behavior" not in dev:
        dev["behavior"] = dev.pop("description_status", {"status":"missing","reason":"not_recorded_in_source_phase"})
    if records and isinstance(dev, dict) and _mapping(dev.get("behavior")).get("schema_version") == "true_rl_agent.push_recovery.behavior.public.v4":
        generation = next((_mapping(r.get("generation")) for r in reversed(records)
            if any(t.get("trial_id") == _mapping(source).get("trial_id")
                   for t in _mapping(r.get("generation")).get("trials", ()))), {})
        feedback["trajectory_sources"] = {"selected_feedback": _trajectory_source(dev["behavior"]),
            "generation_id": generation.get("generation_id"),
            "siblings": [{"trial_id": t.get("trial_id"), "lane_id": t.get("lane_id"),
                           "trial_status": t.get("trial_status"), "source": _trajectory_source(b)}
                         for t in generation.get("trials", ())
                         if (b := _mapping(_mapping(_mapping(t.get("public_dev")).get("public_summary")).get("behavior"))).get("schema_version")
                         == "true_rl_agent.push_recovery.behavior.public.v4"],
            "parent": generation.get("parent_trajectory_source", {"status": "unavailable"})}
    return {"schema_version": "true_rl_agent.public_feedback_summary.push_recovery.v2", **feedback}


def _v2_reference_feedback(initial_anchor: Mapping[str, Any] | None) -> dict[str, Any]:
    anchor = _mapping(initial_anchor)
    dev = _v2_dev_summary(_metrics(anchor).get("public_dev")) if anchor.get("anchor_kind") == "reference_program" else {}
    if _mapping(dev.get("behavior")).get("schema_version") == "true_rl_agent.push_recovery.behavior.public.v4":
        from true_rl_agent.evaluation.binding import canonical_json_hash
        if anchor.get("public_result_hash") != canonical_json_hash(anchor.get("public_result")):
            raise ValueError("Push trajectory reference public result hash mismatch")
        _validate_trajectory_record(anchor)
        if dev["behavior"]["domain"]["subject_kind"] != "reference":
            raise ValueError("Push trajectory reference role mismatch")
    if not dev:
        return {"status": "no_prior_public_feedback"}
    return {"status": "available", "source_selection": "reference_current_best",
        "latest_public_score": _mapping(dev.get("score_stats")).get("mean"),
        "latest_failure_mode": classify_push_prompt_failure({"public_dev": dev}), "public_dev": dev}


def _v2_trial_feedback(selection: str, source: Mapping[str, Any]) -> dict[str, Any]:
    dev = _v2_dev_summary(_mapping(source.get("public_dev")).get("public_summary"))
    if not dev:
        dev = _v2_dev_summary(_metrics(source).get("public_dev"))
    candidate = _mapping(source.get("candidate"))
    return {"status": "available" if dev else "no_public_dev_for_selected_trial", "source_selection": selection,
        "source_round_id": source.get("trial_id", source.get("round_id")), "source_candidate_id": candidate.get("candidate_id"),
        "program_source_checksum": candidate.get("program_source_checksum"),
        "latest_public_score": _mapping(dev.get("score_stats")).get("mean"),
        "latest_failure_mode": classify_push_prompt_failure({"public_dev": dev}),
        "public_dev": dev}


def _trajectory_source(behavior: Mapping[str, Any]) -> dict[str, Any]:
    public = project_push_behavior(behavior)
    return {"card_id": public["card_id"], **{k: public["domain"][k] for k in
        ("source_id", "job_id", "subject_kind", "subject_id", "checkpoint_hash", "evaluation_plan_hash")}}


def _validate_trajectory_record(record: Mapping[str, Any]) -> None:
    from .trajectory_report import PUBLIC_SCHEMA, require_phase_binding, require_trial_binding
    metrics = _metrics(record)
    behavior = _mapping(_mapping(metrics.get("public_dev")).get("behavior"))
    if behavior.get("schema_version") == PUBLIC_SCHEMA:
        binding = metrics.get("trajectory_phase_binding")
        if not isinstance(binding, Mapping):
            raise ValueError("Push trajectory result requires explicit producer phase binding")
        require_phase_binding(behavior, binding)
    for trial in _mapping(record.get("generation")).get("trials", ()):
        behavior = _mapping(_mapping(_mapping(trial.get("public_dev")).get("public_summary")).get("behavior"))
        if behavior.get("schema_version") == PUBLIC_SCHEMA:
            require_trial_binding(behavior, trial)


def _minimum_core_push_prompt_sections(
    sections: Mapping[str, Any],
) -> dict[str, Any]:
    original = dict(sections)
    feedback = _mapping(original.get("public_feedback_summary"))
    minimum_feedback = {
        key: feedback[key]
        for key in (
            "schema_version",
            "status",
            "round_count",
            "accepted_count",
            "rejected_count",
            "latest_round",
            "latest_public_score",
            "best_accepted_public_score",
            "latest_failure_mode",
            "source_selection", "source_round_id", "source_candidate_id", "program_source_checksum",
            "public_dev", "recovery_evidence_gate", "sealed_validation_gate",
            "current_best",
            "trajectory_sources",
        )
        if key in feedback
    }
    if not minimum_feedback.get("latest_round"):
        minimum_feedback["latest_round"] = {"status": "no_closed_outcome"}
    minimum_feedback["projection_level"] = "minimum_core"
    minimum = {
        "task": original.get("task"),
        "reward_inputs_contract": original.get("reward_inputs_contract"),
        "public_eval_contract": original.get("public_eval_contract"),
        "public_feedback_summary": minimum_feedback,
        "runtime_feedback_summary": {
            "schema_version": (
                "true_rl_agent.runtime_feedback_summary.push_recovery.minimum_core.v1"
            ),
            "status": "omitted_for_capacity",
            "reason": "minimum_core_projection",
        },
        "known_good_examples": [],
        "selection_guidance": original.get("selection_guidance", []),
    }
    receipt = build_prompt_capacity_receipt(
        original=original,
        projected_content=minimum,
        projection_id=("push_recovery.minimum_core.v3" if str(feedback.get("schema_version", "")).endswith("push_recovery.v3")
                       else "push_recovery.minimum_core.v2" if str(feedback.get("schema_version", "")).endswith("push_recovery.v2")
                       else "push_recovery.minimum_core.v1"),
        omitted=("aggregate_history_detail", "runtime_activation_detail"),
    )
    minimum_feedback["capacity_projection_receipt"] = receipt
    if canonical_prompt_json_bytes(minimum) > MAX_TASK_PROMPT_SECTIONS_BYTES:
        raise ValueError("Push minimum-core prompt sections exceed shared capacity")
    return minimum


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


def _feedback_source(public_records: Sequence[Mapping[str, Any]], *, include_failed_training_sampling: bool = False) -> tuple[str, Mapping[str, Any] | None]:
    selected = latest_nonaccepted_trial(public_records, include_failed_training_sampling=include_failed_training_sampling)
    if selected is not None:
        if not isinstance(selected[1].get("train", selected[1].get("train_result")), Mapping):
            return "latest_nonaccepted_training_failure", selected[1]
        return "latest_trained_nonaccepted", selected[1]
    return "accepted_current_best", _accepted_feedback_source(public_records)


def _accepted_feedback_source(public_records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    # Select the accepted identity before reading either public or runtime data.
    for record in reversed(public_records):
        if _decision(record).get("accepted") is not True:
            continue
        trials = _mapping(record.get("generation", record)).get("trials", ())
        latest = next((trial for trial in trials
            if trial.get("trial_status") == "accepted"
            and isinstance(trial.get("candidate"), Mapping)
            and isinstance(trial.get("train"), Mapping)), None)
        if not trials and isinstance(record.get("candidate"), Mapping) and isinstance(record.get("train_result"), Mapping):
            latest = record
        if latest is not None:
            return latest
    return None


def _runtime_feedback_summary(public_records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    source_selection, latest = _feedback_source(public_records, include_failed_training_sampling=True)
    if latest is None:
        return {
            "schema_version": "true_rl_agent.runtime_feedback_summary.push_recovery.v2",
            "status": "no_prior_runtime_feedback",
            "eval_candidate_code_policy": "checkpoint_eval_only; candidate code is not expected to load in eval",
        }
    runtime = _runtime(latest)
    measured = RuntimeActivationEvidence.from_mapping(runtime).measurement_summary()
    train_summary = _train_summary(latest)
    candidate = _mapping(latest.get("candidate"))
    source_identity = {
        "source_selection": source_selection,
        "source_round_id": latest.get("trial_id", latest.get("round_id")),
        "source_candidate_id": candidate.get("candidate_id"),
        "program_source_checksum": candidate.get("program_source_checksum"),
        "source_trial_status": latest.get("trial_status"),
    }
    if measured["status"] == "not_available":
        return sanitize_public_context({
            "schema_version": "true_rl_agent.runtime_feedback_summary.push_recovery.v2",
            "status": "not_available",
            "reason": "selected_trial_has_no_runtime_observations",
            **({"latest_runtime_activation": measured} if "training_sampling" in measured else {}),
            **source_identity,
        })
    candidate_summary = _mapping(train_summary.get("candidate"))
    public_metrics = _metrics(latest)
    return sanitize_public_context(
        {
            "schema_version": "true_rl_agent.runtime_feedback_summary.push_recovery.v2",
            "status": "available",
            "reason": "same_trial_training_receipt",
            **source_identity,
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
                **measured,
                "reward_output_stats": _mapping(runtime.get("reward_output_stats")),
                "reward_term_activation": _runtime_reward_term_activation_summary(
                    runtime.get("component_activation_records")
                ),
                "diagnostic_outputs": _runtime_diagnostic_outputs_summary(runtime.get("diagnostic_outputs")),
                "capability_activation_records": _list_of_strings(runtime.get("capability_activation_records")),
            },
            "runtime_activation_gate": _mapping(latest.get("runtime_activation_gate")),
            "mechanism_attribution": _mapping(latest.get("mechanism_attribution")),
            "runner_error": sanitize_untrusted_public_diagnostic(
                latest.get("runner_error")
            ),
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
    if redact_authorization_credential_markers(text) != text:
        return "redacted_diagnostic_key"
    if contains_forbidden_public_value_marker(
        text
    ) or contains_forbidden_public_key_marker(text):
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
    if redact_authorization_credential_markers(text) != text:
        return "[redacted]"
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
