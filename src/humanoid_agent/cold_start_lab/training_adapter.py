"""Phase 8E-3 training-side adapter for callable candidate components.

The adapter is runner-owned.  It may execute candidate reward/loss/diagnostic
callables inside the isolated training process, but candidates never receive
the env, runner, optimizer, checkpoint selector, eval config, or metric code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from humanoid_agent.code_sandbox.components import RewardInputs, TrainingComponents
from humanoid_agent.code_sandbox.loader import CandidateManifest, load_training_components
from humanoid_agent.cold_start_lab.recipe_adapter import (
    DEFAULT_SCHEDULE_ALLOWED_KEYS,
    MAX_DIAGNOSTIC_TERMS,
    MAX_DIAGNOSTIC_FIELDS_PER_TERM,
    MAX_TOTAL_DIAGNOSTIC_FIELDS,
    ComponentContractError,
    build_reward_inputs,
    evaluate_diagnostic_terms,
    guard_vector_value,
    precompute_schedule_table,
    validate_training_components,
)
from humanoid_agent.cold_start_lab.push_signal import BUFFER_ATTR


DEFAULT_SCHEDULE_PROGRESS_POINTS = tuple(index / 8.0 for index in range(9))
# Runner-owned clamp on the candidate curriculum's reward weight scaling.  The
# candidate proposes a schedule; the runner bounds it so one term cannot explode
# the training reward.  This is a training-only lever; eval is unaffected.
REWARD_WEIGHT_SCALE_KEY = "reward_weight_scale"
REWARD_WEIGHT_SCALE_MIN = 0.0
REWARD_WEIGHT_SCALE_MAX = 10.0


PUSH_OR_DR_SCHEDULE_KEYS = frozenset(
    {
        "push_magnitude_scale",
        "push_start_fraction",
        "push_interval_s",
        "domain_randomization_scale",
    }
)

SCHEDULE_CONTROL_DETAILS = {
    "reward_weight_scale": {
        "training_effect": "multiplies candidate reward term weights after guard checks",
        "bounds": [REWARD_WEIGHT_SCALE_MIN, REWARD_WEIGHT_SCALE_MAX],
    },
    "push_magnitude_scale": {
        "training_effect": "not yet connected to env push curriculum",
    },
    "push_start_fraction": {
        "training_effect": "not yet connected to env push curriculum",
    },
    "push_interval_s": {
        "training_effect": "not yet connected to env push curriculum",
    },
    "domain_randomization_scale": {
        "training_effect": "not yet connected to env domain randomization",
    },
}


def training_adapter_runtime_contract() -> dict[str, Any]:
    return {
        "candidate_receives": list(RewardInputs.field_names()),
        "candidate_does_not_receive": [
            "env",
            "runner",
            "optimizer",
            "checkpoint_selector",
            "eval_config",
            "success_metric",
        ],
        "reward_terms_added_to_env_reward": True,
        "loss_terms_guarded_but_not_inserted_into_ppo_optimizer": True,
        "diagnostics_public_numeric_summary_only": True,
        "diagnostic_limits": {
            "max_diagnostic_terms": MAX_DIAGNOSTIC_TERMS,
            "max_fields_per_diagnostic": MAX_DIAGNOSTIC_FIELDS_PER_TERM,
            "max_total_diagnostic_fields": MAX_TOTAL_DIAGNOSTIC_FIELDS,
            "overflow_behavior": "reject_candidate_payload",
        },
        "schedule_drives_reward_weight_scale": True,
        "reward_weight_scale_bounds": [REWARD_WEIGHT_SCALE_MIN, REWARD_WEIGHT_SCALE_MAX],
        "schedule_curriculum_applied_to_env_push_or_dr": False,
        "input_field_status": _input_field_status(),
    }


def control_effectiveness_contract(
    *,
    allowed_schedule_keys: Iterable[str] = DEFAULT_SCHEDULE_ALLOWED_KEYS,
) -> dict[str, Any]:
    allowed_keys = list(allowed_schedule_keys)
    missing = [key for key in allowed_keys if key not in SCHEDULE_CONTROL_DETAILS]
    if missing:
        raise ComponentContractError(f"missing schedule control effectiveness metadata: {missing}")
    runtime_contract = training_adapter_runtime_contract()
    reward_scale_effective = bool(runtime_contract["schedule_drives_reward_weight_scale"])
    push_or_dr_effective = bool(runtime_contract["schedule_curriculum_applied_to_env_push_or_dr"])
    schedule_meta: dict[str, dict[str, Any]] = {}
    for key in allowed_keys:
        meta = dict(SCHEDULE_CONTROL_DETAILS[key])
        if key == REWARD_WEIGHT_SCALE_KEY:
            meta["effectiveness"] = "effective_now" if reward_scale_effective else "accepted_precomputed_not_applied"
        elif key in PUSH_OR_DR_SCHEDULE_KEYS:
            meta["effectiveness"] = "effective_now" if push_or_dr_effective else "accepted_precomputed_not_applied"
        else:
            meta["effectiveness"] = "accepted_precomputed_not_applied"
        schedule_meta[key] = meta
    effective_keys = {
        key: meta
        for key, meta in schedule_meta.items()
        if meta["effectiveness"] == "effective_now"
    }
    accepted_but_ineffective = {
        key: meta
        for key, meta in schedule_meta.items()
        if meta["effectiveness"] != "effective_now"
    }
    return {
        "schema_version": "humanoid_agent.phase8e.control_effectiveness.v1",
        "component_controls": {
            "reward_terms": {
                "allowed": True,
                "effectiveness": "effective_now",
                "training_effect": "guarded vector output is added to env reward",
            },
            "loss_terms": {
                "allowed": True,
                "effectiveness": "guarded_but_not_inserted_into_ppo_optimizer",
                "training_effect": "callable is guard-checked, but no PPO optimizer loss is changed in this phase",
            },
            "schedules": {
                "allowed": True,
                "effectiveness": "key_specific",
                "training_effect": "schedule payloads are precomputed; only effective_keys alter current training",
            },
            "diagnostics": {
                "allowed": True,
                "effectiveness": "public_numeric_summary_only",
                "training_effect": "summarized for feedback, not used as a reward, loss, or decision rule",
                "limits": runtime_contract["diagnostic_limits"],
            },
        },
        "schedule_controls": {
            "allowed_keys": allowed_keys,
            "effective_keys": effective_keys,
            "accepted_but_ineffective_keys": accepted_but_ineffective,
        },
        "primary_mechanism_required": "candidate_must_be_able_to_improve_through_reward_terms_and_reward_weight_scale_alone",
        "runtime_evidence": {
            "source": "Phase8ETrainingAdapter.public_summary.runtime_contract",
            "reward_weight_scale_behavior_probe": "covered_by_phase8e_training_adapter_selftest",
            "schedule_curriculum_applied_to_env_push_or_dr": runtime_contract["schedule_curriculum_applied_to_env_push_or_dr"],
            "loss_terms_inserted_into_ppo_optimizer": False,
        },
        "guidance": [
            "Do not rely on currently ineffective schedule keys as the main causal mechanism.",
            "The candidate must be able to improve through RewardTerm callables and reward_weight_scale alone.",
        ],
    }


def _optional_positive_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except Exception as exc:  # noqa: BLE001
        raise ComponentContractError(f"{name} must be a positive integer") from exc
    if parsed <= 0:
        raise ComponentContractError(f"{name} must be a positive integer")
    return parsed


@dataclass
class Phase8ERuntimeState:
    step_calls: int = 0
    reward_term_calls: int = 0
    loss_term_calls: int = 0
    diagnostic_calls: int = 0
    last_reward_terms: list[dict[str, Any]] = field(default_factory=list)
    last_loss_terms: list[dict[str, Any]] = field(default_factory=list)
    last_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    last_candidate_reward_mean: float | None = None
    last_reward_weight_scale: float = 1.0
    input_provenance: dict[str, str] | None = None
    previous_action: Any = None


class Phase8ETrainingAdapter:
    """Execute candidate callables through the public training contract."""

    def __init__(
        self,
        *,
        manifest: CandidateManifest | None,
        components: TrainingComponents,
        num_envs: int,
        planned_training_steps: int,
        num_joints: int | None = None,
        schedule_table: dict[str, Any] | None = None,
        diagnostic_interval_steps: int = 1,
    ) -> None:
        if num_envs <= 0:
            raise ComponentContractError("num_envs must be positive")
        self.manifest = manifest
        self.components = components
        self.num_envs = int(num_envs)
        self.num_joints = _optional_positive_int(num_joints, "num_joints")
        self.planned_training_steps = max(1, int(planned_training_steps))
        self.schedule_table = schedule_table or {}
        self.diagnostic_interval_steps = max(1, int(diagnostic_interval_steps))
        self.component_summary = validate_training_components(components)
        self.runtime = Phase8ERuntimeState()

    @classmethod
    def from_candidate_dir(
        cls,
        candidate_dir: str | Path,
        *,
        num_envs: int,
        planned_training_steps: int,
        num_joints: int | None = None,
        expected_schema_version: str = "humanoid_agent.cold_start.candidate.v1",
        task_spec: Any = None,
        schedule_progress_points: Iterable[float] = DEFAULT_SCHEDULE_PROGRESS_POINTS,
        schedule_allowed_keys: Iterable[str] = DEFAULT_SCHEDULE_ALLOWED_KEYS,
        diagnostic_interval_steps: int = 1,
    ) -> "Phase8ETrainingAdapter":
        manifest, components = load_training_components(
            candidate_dir,
            expected_schema_version=expected_schema_version,
            allow_llm_authored=True,
            task_spec=task_spec,
            base_env_cfg=None,
            base_agent_cfg=None,
        )
        schedule_table = precompute_schedule_table(
            components,
            progress_points=schedule_progress_points,
            allowed_keys=schedule_allowed_keys,
        )
        return cls(
            manifest=manifest,
            components=components,
            num_envs=num_envs,
            planned_training_steps=planned_training_steps,
            num_joints=num_joints,
            schedule_table=schedule_table,
            diagnostic_interval_steps=diagnostic_interval_steps,
        )

    def public_summary(self) -> dict[str, Any]:
        return {
            "schema_version": "humanoid_agent.phase8e.training_adapter.summary.v1",
            "enabled": True,
            "candidate": self.manifest.to_public_summary() if self.manifest is not None else None,
            "components": self.component_summary,
            "schedule_table": self.schedule_table,
            "runtime_contract": training_adapter_runtime_contract(),
        }

    def runtime_summary(self) -> dict[str, Any]:
        return {
            "schema_version": "humanoid_agent.phase8e.training_adapter.runtime.v1",
            "step_calls": self.runtime.step_calls,
            "reward_term_calls": self.runtime.reward_term_calls,
            "loss_term_calls": self.runtime.loss_term_calls,
            "diagnostic_calls": self.runtime.diagnostic_calls,
            "last_reward_terms": self.runtime.last_reward_terms,
            "last_loss_terms": self.runtime.last_loss_terms,
            "last_diagnostics": self.runtime.last_diagnostics,
            "last_candidate_reward_mean": self.runtime.last_candidate_reward_mean,
            "last_reward_weight_scale": self.runtime.last_reward_weight_scale,
            "input_provenance": self.runtime.input_provenance,
        }

    def apply_step_rewards(self, *, env: Any, actions: Any, rewards: Any, reset_mask: Any = None) -> Any:
        """Return rewards plus guarded candidate RewardTerm contributions."""

        self.runtime.step_calls += 1
        progress = self._current_progress()
        reward_weight_scale = self._current_reward_weight_scale(progress)
        self.runtime.last_reward_weight_scale = reward_weight_scale
        inputs = self._build_inputs(env=env, actions=actions, rewards=rewards, reset_mask=reset_mask)
        candidate_reward = None
        reward_summaries: list[dict[str, Any]] = []
        for term in self.components.reward_terms:
            guarded = guard_vector_value(
                name=term.name,
                kind="reward",
                value=term.fn(inputs),
                num_envs=self.num_envs,
                clip_min=term.clip_min,
                clip_max=term.clip_max,
            )
            effective_weight = float(term.weight) * reward_weight_scale
            value = _scale_vector(_coerce_like(guarded.value, rewards), effective_weight)
            candidate_reward = value if candidate_reward is None else _add_vectors(candidate_reward, value)
            summary = guarded.summary.to_public_summary()
            summary["weight"] = float(term.weight)
            summary["reward_weight_scale"] = reward_weight_scale
            summary["effective_weight"] = effective_weight
            reward_summaries.append(summary)
        self.runtime.reward_term_calls += len(self.components.reward_terms)
        self.runtime.last_reward_terms = reward_summaries

        loss_summaries: list[dict[str, Any]] = []
        for term in self.components.loss_terms:
            guarded = guard_vector_value(
                name=term.name,
                kind="loss",
                value=term.fn(inputs),
                num_envs=self.num_envs,
                clip_min=term.clip_min,
                clip_max=term.clip_max,
            )
            summary = guarded.summary.to_public_summary()
            summary["weight"] = float(term.weight)
            summary["optimizer_insertion"] = "not_enabled_in_phase8e_3"
            loss_summaries.append(summary)
        self.runtime.loss_term_calls += len(self.components.loss_terms)
        self.runtime.last_loss_terms = loss_summaries

        if self.components.diagnostics and self.runtime.step_calls % self.diagnostic_interval_steps == 0:
            diagnostics = evaluate_diagnostic_terms(self.components, inputs, num_envs=self.num_envs)
            self.runtime.diagnostic_calls += len(self.components.diagnostics)
            self.runtime.last_diagnostics = diagnostics

        if candidate_reward is None:
            self.runtime.last_candidate_reward_mean = 0.0
            return rewards
        self.runtime.last_candidate_reward_mean = _vector_mean(candidate_reward)
        return _add_vectors(rewards, candidate_reward)

    def _current_progress(self) -> float:
        return min(1.0, self.runtime.step_calls / self.planned_training_steps)

    def _current_reward_weight_scale(self, progress: float) -> float:
        """Resolve the runner-clamped reward weight scale from the candidate schedule.

        The schedule table is pre-evaluated on a fixed grid (runner-owned).  We
        take the piecewise-constant value at the largest grid point <= progress,
        multiply any per-schedule ``reward_weight_scale`` values, and clamp to the
        runner bounds so the candidate cannot blow up the training reward.
        """

        rows = self.schedule_table.get("rows") if isinstance(self.schedule_table, dict) else None
        if not isinstance(rows, list) or not rows:
            return 1.0
        chosen: dict[str, Any] | None = None
        for row in rows:
            if not isinstance(row, dict):
                continue
            point = row.get("progress_fraction")
            if not isinstance(point, (int, float)):
                continue
            if float(point) <= progress + 1e-9:
                chosen = row
        if chosen is None:
            chosen = rows[0] if isinstance(rows[0], dict) else None
        if chosen is None:
            return 1.0
        scale = 1.0
        found = False
        values = chosen.get("values")
        if isinstance(values, dict):
            for payload in values.values():
                if isinstance(payload, dict) and REWARD_WEIGHT_SCALE_KEY in payload:
                    raw = payload[REWARD_WEIGHT_SCALE_KEY]
                    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                        scale *= float(raw)
                        found = True
        if not found:
            return 1.0
        return max(REWARD_WEIGHT_SCALE_MIN, min(REWARD_WEIGHT_SCALE_MAX, scale))

    def _build_inputs(self, *, env: Any, actions: Any, rewards: Any, reset_mask: Any = None) -> RewardInputs:
        base = _unwrap_env(env)
        robot_data = _robot_data(base)
        progress = self._current_progress()
        height, height_obs = _root_height(robot_data, rewards)
        roll, pitch, rp_obs = _roll_pitch(robot_data, rewards)
        base_pos_xy, base_pos_xy_obs = _base_pos_xy(base, robot_data, rewards)
        lin_vel, lin_obs = _observed_or_zero(_first_attr(robot_data, ("root_lin_vel_b", "root_lin_vel_w", "body_lin_vel_w")), rewards, width=3)
        ang_vel, ang_obs = _observed_or_zero(_first_attr(robot_data, ("root_ang_vel_b", "root_ang_vel_w", "body_ang_vel_w")), rewards, width=3)
        projected_gravity, gravity_obs = _projected_gravity(robot_data, rewards)
        commanded_base_vel, command_obs = _commanded_base_vel(base, rewards)
        num_joints = self.num_joints or _num_joints_from_robot_data(robot_data)
        if self.num_joints is None and num_joints is not None:
            self.num_joints = num_joints
        joint_pos, jpos_obs = _observed_or_zero(_first_attr(robot_data, ("joint_pos",)), rewards, width=num_joints)
        joint_vel, jvel_obs = _observed_or_zero(_first_attr(robot_data, ("joint_vel",)), rewards, width=num_joints)
        joint_pos_error, jerr_obs = _joint_pos_error(robot_data, joint_pos, rewards, num_joints=num_joints)
        last_action, act_obs = _observed_or_zero(actions, rewards)
        prev_action, prev_provenance = _previous_action_or_zero(self.runtime.previous_action, actions, rewards)
        action_delta, delta_provenance = _action_delta_or_zero(actions, prev_action, prev_provenance, rewards)
        self.runtime.previous_action = _mask_reset_actions(_clone_public_value(actions), reset_mask)
        push_active, push_vec_public, push_provenance = _push_signal_or_zero(base, rewards)
        mapping = {
            "base_height": height,
            "roll": roll,
            "pitch": pitch,
            "base_pos_xy": base_pos_xy,
            "base_lin_vel": lin_vel,
            "base_ang_vel": ang_vel,
            "projected_gravity": projected_gravity,
            "commanded_base_vel": commanded_base_vel,
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "joint_pos_error": joint_pos_error,
            "last_action": last_action,
            "prev_action": prev_action,
            "action_delta": action_delta,
            "push_active": push_active,
            "push_vec_public": push_vec_public,
            "progress_fraction": _constant_like(rewards, progress),
        }
        if self.runtime.input_provenance is None:
            # Record once: whether each field carried real env signal or fell back
            # to zero this run.  This makes silent attribute-name mismatches visible
            # instead of the candidate training on zeros while the static contract
            # claims the field is available.
            self.runtime.input_provenance = {
                "base_height": _provenance_tag(height_obs),
                "roll": _provenance_tag(rp_obs),
                "pitch": _provenance_tag(rp_obs),
                "base_pos_xy": _provenance_tag(base_pos_xy_obs),
                "base_lin_vel": _provenance_tag(lin_obs),
                "base_ang_vel": _provenance_tag(ang_obs),
                "projected_gravity": _provenance_tag(gravity_obs),
                "commanded_base_vel": _provenance_tag(command_obs),
                "joint_pos": _provenance_tag(jpos_obs),
                "joint_vel": _provenance_tag(jvel_obs),
                "joint_pos_error": _provenance_tag(jerr_obs),
                "last_action": _provenance_tag(act_obs),
                "prev_action": prev_provenance,
                "action_delta": delta_provenance,
                "push_active": push_provenance,
                "push_vec_public": push_provenance,
                "progress_fraction": "runner_owned",
            }
        elif push_provenance == "runner_owned_push_event_instrumentation":
            self.runtime.input_provenance["push_active"] = push_provenance
            self.runtime.input_provenance["push_vec_public"] = push_provenance
        return build_reward_inputs(mapping)


class Phase8EVecEnvRewardWrapper:
    """Proxy an RSL-RL VecEnv and insert guarded candidate reward terms."""

    def __init__(self, env: Any, adapter: Phase8ETrainingAdapter) -> None:
        self.env = env
        self.phase8e_adapter = adapter
        if self.phase8e_adapter.num_joints is None:
            self.phase8e_adapter.num_joints = _num_joints_from_env(_unwrap_env(env))

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def step(self, actions: Any) -> tuple[Any, ...]:
        result = self.env.step(actions)
        if not isinstance(result, tuple):
            raise ComponentContractError("VecEnv.step must return a tuple")
        reward_index = _find_reward_index(result, self.phase8e_adapter.num_envs)
        rewards = result[reward_index]
        reset_mask = _reset_mask_from_step_result(result, self.phase8e_adapter.num_envs)
        updated_rewards = self.phase8e_adapter.apply_step_rewards(env=self.env, actions=actions, rewards=rewards, reset_mask=reset_mask)
        mutable = list(result)
        mutable[reward_index] = updated_rewards
        return tuple(mutable)


def _find_reward_index(result: tuple[Any, ...], num_envs: int) -> int:
    if len(result) < 2:
        raise ComponentContractError("VecEnv.step result is missing the reward slot")
    reward = result[1]
    if _shape_tuple(reward) != (num_envs,) or _is_bool_vector(reward):
        raise ComponentContractError(
            "RSL-RL VecEnv.step reward slot must be a numeric vector at index 1"
        )
    return 1


def _reset_mask_from_step_result(result: tuple[Any, ...], num_envs: int) -> Any:
    if len(result) >= 5:
        return _combine_reset_masks(result[2], result[3], num_envs)
    if len(result) >= 3 and _shape_tuple(result[2]) == (num_envs,):
        return result[2]
    return None


def _combine_reset_masks(first: Any, second: Any, num_envs: int) -> Any:
    first_torch = _torch_for(first)
    if first_torch is not None:
        second_torch = _torch_for(second)
        if second_torch is not None:
            second_mask = second.bool()
        else:
            second_list = _mask_to_list(second, num_envs)
            if second_list is None:
                return None
            second_mask = first_torch.tensor(second_list, dtype=first_torch.bool, device=first.device)
        return first.bool() | second_mask
    first_list = _mask_to_list(first, num_envs)
    second_list = _mask_to_list(second, num_envs)
    if first_list is None or second_list is None:
        return None
    return [a or b for a, b in zip(first_list, second_list, strict=True)]


def _unwrap_env(env: Any) -> Any:
    current = env
    for _ in range(8):
        unwrapped = getattr(current, "unwrapped", None)
        if unwrapped is None or unwrapped is current:
            return current
        current = unwrapped
    return current


def _robot_data(env: Any) -> Any:
    robot = _robot_object(env)
    return getattr(robot, "data", None)


def _robot_object(env: Any) -> Any:
    scene = getattr(env, "scene", None)
    robot = None
    if scene is not None:
        try:
            robot = scene["robot"]
        except Exception:  # noqa: BLE001 - scene type varies across Isaac tasks.
            robot = getattr(scene, "robot", None)
    if robot is None:
        robot = getattr(env, "robot", None)
    return robot


def _num_joints_from_env(env: Any) -> int | None:
    robot = _robot_object(env)
    inferred = _num_joints_from_robot_data(getattr(robot, "data", None))
    if inferred is not None:
        return inferred
    for attr_name in ("num_joints", "num_dof", "num_actions"):
        raw = getattr(robot, attr_name, None)
        if raw is None:
            continue
        try:
            parsed = int(raw)
        except Exception:  # noqa: BLE001
            continue
        if parsed > 0:
            return parsed
    return None


def _provenance_tag(observed: bool) -> str:
    return "observed" if observed else "zero_fallback"


def _root_height(robot_data: Any, rewards: Any) -> tuple[Any, bool]:
    root_pos = _first_attr(robot_data, ("root_pos_w", "root_link_pos_w"))
    column = _column(root_pos, 2)
    if column is not None:
        return column, True
    return _constant_like(rewards, 0.0), False


def _base_pos_xy(env: Any, robot_data: Any, rewards: Any) -> tuple[Any, bool]:
    root_pos = _first_attr(robot_data, ("root_pos_w", "root_link_pos_w"))
    xy = _matrix_columns(root_pos, 2)
    origins_xy = _env_origins_xy(env)
    if xy is not None and origins_xy is not None:
        relative_xy = _subtract_values(xy, origins_xy)
        if relative_xy is not None:
            return relative_xy, True
    return _zero_matrix_like(rewards, width=2), False


def _env_origins_xy(env: Any) -> Any | None:
    scene = getattr(env, "scene", None)
    origins = getattr(scene, "env_origins", None)
    return _matrix_columns(origins, 2)


def _roll_pitch(robot_data: Any, rewards: Any) -> tuple[Any, Any, bool]:
    quat = _first_attr(robot_data, ("root_quat_w", "root_link_quat_w"))
    torch_mod = _torch_for(quat)
    if torch_mod is None or _shape_tuple(quat)[-1:] != (4,):
        zero = _constant_like(rewards, 0.0)
        return zero, zero, False
    # IsaacLab root quaternions are conventionally wxyz.
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    roll = torch_mod.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch_arg = torch_mod.clamp(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = torch_mod.asin(pitch_arg)
    return roll, pitch, True


def _projected_gravity(robot_data: Any, rewards: Any) -> tuple[Any, bool]:
    quat = _first_attr(robot_data, ("root_quat_w", "root_link_quat_w"))
    torch_mod = _torch_for(quat)
    if torch_mod is None or _shape_tuple(quat)[-1:] != (4,):
        return _zero_matrix_like(rewards, width=3), False
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return torch_mod.stack(
        (
            -2.0 * (x * z - w * y),
            -2.0 * (y * z + w * x),
            -(1.0 - 2.0 * (x * x + y * y)),
        ),
        dim=-1,
    ), True


def _commanded_base_vel(env: Any, rewards: Any) -> tuple[Any, bool]:
    command_manager = getattr(env, "command_manager", None)
    if command_manager is not None:
        get_command = getattr(command_manager, "get_command", None)
        if callable(get_command):
            for name in ("base_velocity", "base_vel"):
                try:
                    value = get_command(name)
                except Exception:  # noqa: BLE001 - Isaac managers vary across tasks.
                    value = None
                observed, ok = _observed_or_zero(value, rewards, width=3)
                if ok:
                    return observed, True
        for name in ("base_velocity", "base_vel", "command"):
            observed, ok = _observed_or_zero(getattr(command_manager, name, None), rewards, width=3)
            if ok:
                return observed, True
    return _zero_matrix_like(rewards, width=3), False


def _joint_pos_error(robot_data: Any, joint_pos: Any, rewards: Any, *, num_joints: int | None = None) -> tuple[Any, bool]:
    default_joint_pos = _first_attr(robot_data, ("default_joint_pos", "default_joint_pos_target"))
    if default_joint_pos is None:
        return _joint_fallback_like(joint_pos, rewards, num_joints=num_joints), False
    default_joint_pos = _broadcast_first_dim(default_joint_pos, _shape_tuple(joint_pos))
    if default_joint_pos is None or _shape_tuple(default_joint_pos) != _shape_tuple(joint_pos):
        return _joint_fallback_like(joint_pos, rewards, num_joints=num_joints), False
    diff = _subtract_values(joint_pos, default_joint_pos)
    if diff is None:
        return _joint_fallback_like(joint_pos, rewards, num_joints=num_joints), False
    return diff, True


def _joint_fallback_like(value: Any, rewards: Any, *, num_joints: int | None) -> Any:
    if num_joints is not None:
        return _zero_matrix_like(rewards, width=num_joints)
    return _zero_like_value(value, rewards)


def _num_joints_from_robot_data(robot_data: Any) -> int | None:
    for attr_name in ("joint_pos", "joint_vel", "default_joint_pos", "default_joint_pos_target"):
        inferred = _trailing_width(_first_attr(robot_data, (attr_name,)))
        if inferred is not None:
            return inferred
    for attr_name in ("num_joints", "num_dof", "num_actions"):
        raw = getattr(robot_data, attr_name, None)
        if raw is None:
            continue
        try:
            parsed = int(raw)
        except Exception:  # noqa: BLE001
            continue
        if parsed > 0:
            return parsed
    return None


def _trailing_width(value: Any) -> int | None:
    shape = _shape_tuple(value)
    if len(shape) >= 2 and shape[-1] > 0:
        return int(shape[-1])
    return None


def _first_attr(obj: Any, names: tuple[str, ...]) -> Any:
    if obj is None:
        return None
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return None


def _column(value: Any, index: int) -> Any | None:
    if value is None:
        return None
    shape = _shape_tuple(value)
    if len(shape) >= 2 and shape[0] > 0 and shape[-1] > index:
        try:
            return value[:, index]
        except Exception:  # noqa: BLE001
            pass
    if isinstance(value, list | tuple) and value and all(isinstance(row, list | tuple) and len(row) > index for row in value):
        return [float(row[index]) for row in value]
    return None


def _matrix_columns(value: Any, width: int) -> Any | None:
    if value is None:
        return None
    shape = _shape_tuple(value)
    if len(shape) >= 2 and shape[0] > 0 and shape[-1] >= width:
        try:
            return value[:, :width]
        except Exception:  # noqa: BLE001
            pass
    if isinstance(value, list | tuple) and value and all(isinstance(row, list | tuple) and len(row) >= width for row in value):
        return [[float(row[index]) for index in range(width)] for row in value]
    return None


def _observed_or_zero(value: Any, rewards: Any, *, width: int | None = None) -> tuple[Any, bool]:
    fallback = _zero_matrix_like(rewards, width=width) if width is not None else _constant_like(rewards, 0.0)
    if value is None:
        return fallback, False
    shape = _shape_tuple(value)
    reward_shape = _shape_tuple(rewards)
    if shape == reward_shape or (len(shape) >= 2 and shape[0] == reward_shape[0]):
        return value, True
    return fallback, False


def _push_signal_or_zero(env: Any, rewards: Any) -> tuple[Any, Any, str]:
    buffer = getattr(env, BUFFER_ATTR, None)
    if buffer is not None and callable(getattr(buffer, "consume", None)):
        try:
            snapshot = buffer.consume(rewards)
            active = snapshot.get("active")
            vector = snapshot.get("vector")
            provenance = str(snapshot.get("provenance", "runner_owned_push_event_instrumentation"))
            if _shape_tuple(active) == _shape_tuple(rewards) and _shape_tuple(vector) == (
                _shape_tuple(rewards)[0],
                2,
            ):
                return active, vector, provenance
        except Exception:  # noqa: BLE001 - reward input construction must fall back safely.
            pass
    return _constant_like(rewards, 0.0), _zero_matrix_like(rewards, width=2), "zero_fallback_no_push_event_observed"


def _constant_like(rewards: Any, value: float) -> Any:
    torch_mod = _torch_for(rewards)
    if torch_mod is not None:
        return torch_mod.full_like(rewards, float(value))
    if isinstance(rewards, list | tuple):
        return [float(value) for _ in rewards]
    return value


def _zero_matrix_like(rewards: Any, *, width: int | None) -> Any:
    if width is None:
        return _constant_like(rewards, 0.0)
    torch_mod = _torch_for(rewards)
    if torch_mod is not None:
        return torch_mod.zeros((int(rewards.shape[0]), int(width)), device=rewards.device, dtype=rewards.dtype)
    if isinstance(rewards, list | tuple):
        return [[0.0 for _ in range(width)] for _ in rewards]
    return [[0.0 for _ in range(width)]]


def _previous_action_or_zero(previous_action: Any, actions: Any, rewards: Any) -> tuple[Any, str]:
    if previous_action is not None:
        return previous_action, "derived_from_action_history"
    if actions is not None:
        return _zero_like_value(actions, rewards), "zero_fallback"
    value, _ = _observed_or_zero(None, rewards)
    return value, "zero_fallback"


def _action_delta_or_zero(actions: Any, previous_action: Any, previous_provenance: str, rewards: Any) -> tuple[Any, str]:
    if actions is None or previous_action is None:
        value, _ = _observed_or_zero(None, rewards)
        return value, "zero_fallback"
    delta = _subtract_values(actions, previous_action)
    if delta is None:
        value, _ = _observed_or_zero(None, rewards)
        return value, "zero_fallback"
    if previous_provenance == "zero_fallback":
        return delta, "derived_from_current_action_and_zero_previous_action"
    return delta, "derived_from_action_history"


def _zero_like_value(value: Any, rewards: Any) -> Any:
    torch_mod = _torch_for(value)
    if torch_mod is not None:
        return torch_mod.zeros_like(value)
    if isinstance(value, list):
        return [_zero_like_value(item, rewards) for item in value]
    if isinstance(value, tuple):
        return tuple(_zero_like_value(item, rewards) for item in value)
    if value is None:
        return _constant_like(rewards, 0.0)
    return 0.0


def _subtract_values(left: Any, right: Any) -> Any | None:
    try:
        if _torch_for(left) is not None:
            return left - right
        if isinstance(left, list | tuple) and isinstance(right, list | tuple) and len(left) == len(right):
            return [_subtract_values(a, b) for a, b in zip(left, right, strict=True)]
        return float(left) - float(right)
    except Exception:  # noqa: BLE001
        return None


def _broadcast_first_dim(value: Any, target_shape: tuple[int, ...]) -> Any | None:
    shape = _shape_tuple(value)
    if shape == target_shape:
        return value
    if len(shape) == len(target_shape) and len(shape) >= 2 and shape[0] == 1 and shape[1:] == target_shape[1:]:
        torch_mod = _torch_for(value)
        if torch_mod is not None:
            return value.expand(target_shape)
        if isinstance(value, list | tuple) and value:
            row = _clone_public_value(value[0])
            return [_clone_public_value(row) for _ in range(target_shape[0])]
    return None


def _mask_reset_actions(value: Any, reset_mask: Any) -> Any:
    if value is None or reset_mask is None:
        return value
    torch_mod = _torch_for(value)
    if torch_mod is not None:
        reset_torch = _torch_for(reset_mask)
        if reset_torch is not None:
            mask = reset_mask.bool()
        else:
            reset_list = _mask_to_list(reset_mask, _shape_tuple(value)[0] if _shape_tuple(value) else 0)
            if reset_list is None:
                return value
            mask = torch_mod.tensor(reset_list, dtype=torch_mod.bool, device=value.device)
        masked = value.detach().clone()
        masked[mask] = 0.0
        return masked
    reset_list = _mask_to_list(reset_mask, len(value) if isinstance(value, list | tuple) else 0)
    if reset_list is None or not isinstance(value, list | tuple):
        return value
    out = []
    for item, reset in zip(value, reset_list, strict=True):
        out.append(_zero_like_value(item, []) if reset else _clone_public_value(item))
    return out


def _mask_to_list(mask: Any, num_envs: int) -> list[bool] | None:
    if mask is None or num_envs <= 0:
        return None
    torch_mod = _torch_for(mask)
    if torch_mod is not None:
        data = mask.detach().cpu().tolist()
    elif callable(getattr(mask, "tolist", None)):
        data = mask.tolist()
    else:
        data = mask
    if not isinstance(data, list | tuple) or len(data) != num_envs:
        return None
    return [bool(item) for item in data]


def _clone_public_value(value: Any) -> Any:
    if value is None:
        return None
    torch_mod = _torch_for(value)
    if torch_mod is not None:
        return value.detach().clone()
    if isinstance(value, list):
        return [_clone_public_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_public_value(item) for item in value)
    return value


BASELINE_REWARD_INPUT_FIELDS = (
    "base_height",
    "roll",
    "pitch",
    "base_lin_vel",
    "base_ang_vel",
    "joint_pos",
    "joint_vel",
    "last_action",
    "push_active",
    "push_vec_public",
    "progress_fraction",
)

FIRST_BATCH_REWARD_INPUT_FIELDS = (
    "base_pos_xy",
    "projected_gravity",
    "commanded_base_vel",
    "prev_action",
    "action_delta",
    "joint_pos_error",
)

PLANNED_PUBLIC_REWARD_INPUT_FIELDS = (
    "feet_contact",
    "feet_pos_z",
    "feet_vel_xy",
    "joint_limit_margin",
)

REWARD_INPUT_FIELD_SHAPES = {
    "base_height": ["N"],
    "roll": ["N"],
    "pitch": ["N"],
    "base_pos_xy": ["N", 2],
    "base_lin_vel": ["N", 3],
    "base_ang_vel": ["N", 3],
    "projected_gravity": ["N", 3],
    "commanded_base_vel": ["N", 3],
    "joint_pos": ["N", "num_joints"],
    "joint_vel": ["N", "num_joints"],
    "joint_pos_error": ["N", "num_joints"],
    "last_action": ["N", "num_actions"],
    "prev_action": ["N", "num_actions"],
    "action_delta": ["N", "num_actions"],
    "push_active": ["N"],
    "push_vec_public": ["N", 2],
    "progress_fraction": ["N"],
}


REWARD_INPUT_FIELD_METADATA = {
    "base_height": {
        "semantics": "base height scalar per env",
        "source_status": "observed_when_root_position_available_else_zero_fallback",
        "provenance": "observed_or_zero_fallback",
        "units": "m",
        "coordinate_frame": "world_z",
        "fallback_behavior": "zero_fallback_when_root_position_missing",
    },
    "roll": {
        "semantics": "base roll angle per env derived from root quaternion",
        "source_status": "derived_from_root_quaternion_when_available_else_zero_fallback",
        "provenance": "derived_from_observed_or_zero_fallback",
        "units": "rad",
        "coordinate_frame": "root_orientation",
        "fallback_behavior": "zero_fallback_when_root_quaternion_missing",
    },
    "pitch": {
        "semantics": "base pitch angle per env derived from root quaternion",
        "source_status": "derived_from_root_quaternion_when_available_else_zero_fallback",
        "provenance": "derived_from_observed_or_zero_fallback",
        "units": "rad",
        "coordinate_frame": "root_orientation",
        "fallback_behavior": "zero_fallback_when_root_quaternion_missing",
    },
    "base_pos_xy": {
        "semantics": "env-origin-relative base xy position per env",
        "source_status": "derived_from_root_pos_w_and_scene_env_origins_when_available_else_zero_fallback",
        "provenance": "derived_from_observed_or_zero_fallback",
        "units": "m",
        "coordinate_frame": "env_origin_relative_world_xy",
        "fallback_behavior": "zero_fallback_when_root_pos_w_or_scene.env_origins_missing",
    },
    "base_lin_vel": {
        "semantics": "base linear velocity vector per env",
        "source_status": "observed_when_robot_velocity_attr_available_else_zero_fallback",
        "provenance": "observed_or_zero_fallback",
        "units": "m/s",
        "coordinate_frame": "base_or_world_as_reported_by_adapter",
        "fallback_behavior": "zero_fallback_when_robot_velocity_attr_missing",
    },
    "base_ang_vel": {
        "semantics": "base angular velocity vector per env",
        "source_status": "observed_when_robot_angular_velocity_attr_available_else_zero_fallback",
        "provenance": "observed_or_zero_fallback",
        "units": "rad/s",
        "coordinate_frame": "base_or_world_as_reported_by_adapter",
        "fallback_behavior": "zero_fallback_when_robot_angular_velocity_attr_missing",
    },
    "projected_gravity": {
        "semantics": "gravity direction projected into the base frame",
        "source_status": "derived_from_root_quaternion_when_available_else_zero_fallback",
        "provenance": "derived_from_observed_or_zero_fallback",
        "units": "unitless",
        "coordinate_frame": "base_frame",
        "fallback_behavior": "zero_vector3_fallback_when_root_quaternion_missing",
    },
    "commanded_base_vel": {
        "semantics": "public commanded base velocity vector per env",
        "source_status": "observed_command_when_command_manager_available_else_zero_fallback",
        "provenance": "observed_or_zero_fallback",
        "units": "m/s_and_rad/s",
        "coordinate_frame": "command_frame_xy_yaw",
        "fallback_behavior": "zero_vector3_fallback_when_base_velocity_command_missing",
    },
    "joint_pos": {
        "semantics": "joint position vector per env in runner joint order",
        "source_status": "observed_when_joint_pos_available_else_zero_fallback",
        "provenance": "observed_or_zero_fallback",
        "units": "rad",
        "coordinate_frame": "joint_order",
        "fallback_behavior": "zero_matrix_[N,num_joints]_when_joint_pos_missing_and_num_joints_known",
    },
    "joint_vel": {
        "semantics": "joint velocity vector per env in runner joint order",
        "source_status": "observed_when_joint_vel_available_else_zero_fallback",
        "provenance": "observed_or_zero_fallback",
        "units": "rad/s",
        "coordinate_frame": "joint_order",
        "fallback_behavior": "zero_matrix_[N,num_joints]_when_joint_vel_missing_and_num_joints_known",
    },
    "joint_pos_error": {
        "semantics": "joint_pos minus default_joint_pos in runner joint order",
        "source_status": "derived_from_joint_pos_and_default_joint_pos_when_available_else_zero_fallback",
        "provenance": "derived_from_observed_or_zero_fallback",
        "units": "rad",
        "coordinate_frame": "joint_order",
        "fallback_behavior": "zero_matrix_[N,num_joints]_when_default_joint_pos_missing_or_not_broadcastable_and_num_joints_known",
    },
    "last_action": {
        "semantics": "current action vector passed to this env step",
        "source_status": "observed_current_action",
        "provenance": "observed",
        "units": "policy_action_units",
        "coordinate_frame": "action_order",
        "fallback_behavior": "zero_fallback_when_actions_missing",
    },
    "prev_action": {
        "semantics": "previous action vector from runner-owned action history",
        "source_status": "runner_owned_action_history",
        "provenance": "derived_from_action_history_or_zero_fallback",
        "units": "policy_action_units",
        "coordinate_frame": "action_order",
        "fallback_behavior": "zero_previous_action_on_first_step_or_after_episode_reset",
    },
    "action_delta": {
        "semantics": "current action minus previous action",
        "source_status": "runner_owned_action_history",
        "provenance": "derived_from_action_history_or_current_action_and_zero_previous_action",
        "units": "policy_action_units",
        "coordinate_frame": "action_order",
        "fallback_behavior": "current_action_minus_zero_previous_on_first_step_or_after_episode_reset",
    },
    "push_active": {
        "semantics": "bool-like indicator for envs that received a runner-owned push event on the latest step",
        "source_status": "runner_owned_push_event_instrumentation",
        "provenance": "runner_wrapped_push_robot_event_or_zero_fallback",
        "units": "bool_like",
        "coordinate_frame": "not_applicable",
        "fallback_behavior": "zero_when_no_push_event_fired_or_push_signal_buffer_unavailable",
    },
    "push_vec_public": {
        "semantics": "public xy velocity delta sampled by the runner-owned push event on the latest step",
        "source_status": "runner_owned_push_event_instrumentation",
        "provenance": "runner_wrapped_push_robot_event_or_zero_fallback",
        "units": "m/s",
        "coordinate_frame": "world_xy_velocity_delta",
        "fallback_behavior": "zero_vector2_when_no_push_event_fired_or_push_signal_buffer_unavailable",
    },
    "progress_fraction": {
        "semantics": "per-env public training progress fraction inside RewardInputs",
        "source_status": "runner_owned_progress",
        "provenance": "runner_owned",
        "units": "fraction",
        "coordinate_frame": "not_applicable",
        "fallback_behavior": "runner_always_supplies_clamped_progress_fraction",
    },
}


def reward_inputs_contract() -> dict[str, Any]:
    field_order = list(RewardInputs.field_names())
    missing = [
        name
        for name in field_order
        if name not in REWARD_INPUT_FIELD_SHAPES or name not in REWARD_INPUT_FIELD_METADATA
    ]
    if missing:
        raise ComponentContractError(f"missing RewardInputs contract metadata: {missing}")
    return {
        "schema_version": "humanoid_agent.phase8e.reward_inputs_contract.v1",
        "field_order": field_order,
        "fields": {
            name: {
                "shape": list(REWARD_INPUT_FIELD_SHAPES[name]),
                "dtype_or_value_kind": "tensor_like_numeric",
                "shape_source": "RewardInputs.field_names + runner_owned_field_shapes",
                **REWARD_INPUT_FIELD_METADATA[name],
            }
            for name in field_order
        },
    }


def reward_inputs_field_expansion_report() -> dict[str, Any]:
    return {
        "schema_version": "humanoid_agent.phase8e.reward_inputs_field_expansion.v1",
        "baseline_fields": list(BASELINE_REWARD_INPUT_FIELDS),
        "first_batch_fields": list(FIRST_BATCH_REWARD_INPUT_FIELDS),
        "implemented_public_fields": list(RewardInputs.field_names()),
        "planned_public_fields": list(PLANNED_PUBLIC_REWARD_INPUT_FIELDS),
        "field_shapes": {name: REWARD_INPUT_FIELD_SHAPES[name] for name in RewardInputs.field_names()},
        "admission_policy": [
            "training_side_public_state_only",
            "runner_constructed_or_runner_derived",
            "shape_declared_before_prompt_exposure",
            "synthetic_fast_validation_required",
            "no_eval_decision_or_checkpoint_policy_state",
            "no_env_runner_optimizer_or_policy_objects",
        ],
        "field_status": _input_field_status(),
    }


def reward_inputs_field_status() -> dict[str, str]:
    return {
        name: f"{meta['source_status']}; fallback={meta['fallback_behavior']}"
        for name, meta in reward_inputs_contract()["fields"].items()
    }


def _input_field_status() -> dict[str, str]:
    return reward_inputs_field_status()


def _coerce_like(value: Any, rewards: Any) -> Any:
    torch_mod = _torch_for(rewards)
    if torch_mod is not None:
        if _torch_for(value) is not None:
            return value.to(device=rewards.device, dtype=rewards.dtype)
        return torch_mod.as_tensor(value, device=rewards.device, dtype=rewards.dtype)
    if isinstance(rewards, list | tuple):
        data = value.tolist() if callable(getattr(value, "tolist", None)) else value
        return [float(item) for item in data]
    return value


def _scale_vector(value: Any, scale: float) -> Any:
    if isinstance(value, list):
        return [float(item) * scale for item in value]
    return value * scale


def _add_vectors(left: Any, right: Any) -> Any:
    if isinstance(left, list) and isinstance(right, list):
        return [float(a) + float(b) for a, b in zip(left, right, strict=True)]
    return left + right


def _vector_mean(value: Any) -> float:
    torch_mod = _torch_for(value)
    if torch_mod is not None:
        return float(value.detach().float().mean().cpu().item())
    data = value.tolist() if callable(getattr(value, "tolist", None)) else value
    if isinstance(data, list | tuple) and data:
        return sum(float(item) for item in data) / len(data)
    try:
        return float(data)
    except (TypeError, ValueError):
        return 0.0


def _shape_tuple(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        if isinstance(value, list | tuple):
            if not value:
                return (0,)
            child_shape = _shape_tuple(value[0])
            if child_shape and all(_shape_tuple(item) == child_shape for item in value[1:]):
                return (len(value), *child_shape)
            return (len(value),)
        return ()
    try:
        return tuple(int(item) for item in shape)
    except (TypeError, ValueError):
        return ()


def _is_bool_vector(value: Any) -> bool:
    torch_mod = _torch_for(value)
    if torch_mod is not None:
        return value.dtype == torch_mod.bool
    if isinstance(value, list | tuple) and value:
        return isinstance(value[0], bool)
    return isinstance(value, bool)


def _torch_for(value: Any) -> Any | None:
    try:
        import torch  # type: ignore[import-not-found]  # noqa: PLC0415
    except Exception:  # noqa: BLE001 - local non-Isaac tests do not install torch.
        return None
    return torch if isinstance(value, torch.Tensor) else None
