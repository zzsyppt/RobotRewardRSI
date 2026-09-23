"""Runtime adapter that connects candidate components to push recovery training."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
from true_rl_agent.backends.isaac_lab_arrays import as_torch, quaternion_wxyz

from true_rl_agent.candidates.artifact import (
    SealedCandidateArtifactRef,
    load_sealed_candidate_artifact,
)
from true_rl_agent.candidates.authoring_policy import require_component_name
from true_rl_agent.candidates.components import TrainingComponents
from true_rl_agent.candidates.loader import (
    MAX_CANDIDATE_EXECUTION_FILE_BYTES,
    CandidateManifest,
    load_candidate_manifest,
    load_training_components,
    load_training_components_source_bytes,
    parse_candidate_manifest_bytes,
)
from true_rl_agent.candidates.package_contract import (
    CandidatePackagePreflight,
    open_preflighted_candidate_file,
    preflight_candidate_package,
)
from true_rl_agent.candidates.schedule_contract import (
    REWARD_WEIGHT_SCALE_KEY,
    effective_schedule_keys,
    schedule_bounds_for_key,
    schedule_key_bounds,
    schedule_runtime_activation_record,
)
from true_rl_agent.candidates.schema import CandidateChangeSet
from true_rl_agent.runner.candidate_failure import (
    is_infrastructure_runtime_exception,
)
from true_rl_agent.tasks.push_recovery.candidate_failure import (
    PushCandidateProgramFailure,
)
from true_rl_agent.tasks.push_recovery.reward_input_contract import (
    prompt_effective_reward_input_fields,
    public_push_signal_reward_input_fields,
    push_recovery_reward_input_fields,
    reserved_zero_fallback_reward_input_fields,
)
from true_rl_agent.tasks.push_recovery.reward_inputs import PushRecoveryRewardInputs, PushRecoveryV2RewardInputs

PUBLIC_PUSH_DELTA_THRESHOLD_MPS = 0.12
PUBLIC_PUSH_ACTIVE_KEYS = (
    "push_active",
    "public_push_active",
    "push_active_public",
    "last_push_active",
)
PUBLIC_PUSH_VEC_KEYS = (
    "push_vec_public",
    "public_push_vec",
    "public_push_vec_xy",
    "last_push_velocity_delta_xy",
    "push_velocity_delta_xy",
)
_CANDIDATE_MANIFEST_PATH = "candidate_manifest.yaml"
_CANDIDATE_COMPONENTS_PATH = "training_code/components.py"


def _load_bound_sealed_components(
    sealed_artifact: SealedCandidateArtifactRef,
) -> tuple[CandidateManifest, TrainingComponents]:
    """Load and execute the exact bytes bound to one verified sealed identity."""

    preflight = preflight_candidate_package(
        sealed_artifact.path,
        sealed_artifact=True,
        enforce_size_limits=False,
    )
    for relative_path in (_CANDIDATE_MANIFEST_PATH, _CANDIDATE_COMPONENTS_PATH):
        state = preflight.file_state(relative_path)
        if state.size_bytes > MAX_CANDIDATE_EXECUTION_FILE_BYTES:
            raise ValueError(
                "candidate execution file exceeds the absolute byte limit: "
                f"{relative_path}"
            )

    manifest_payload = _read_preflighted_execution_file(
        preflight,
        _CANDIDATE_MANIFEST_PATH,
    )
    source_payload = _read_preflighted_execution_file(
        preflight,
        _CANDIDATE_COMPONENTS_PATH,
    )
    manifest = parse_candidate_manifest_bytes(manifest_payload)
    if manifest.candidate_id != sealed_artifact.candidate_id:
        raise ValueError("candidate manifest identity mismatch")
    if len(source_payload) != sealed_artifact.program_source_size_bytes:
        raise ValueError("sealed candidate program source size mismatch before execution")
    source_checksum = "sha256:" + hashlib.sha256(source_payload).hexdigest()
    if source_checksum != sealed_artifact.program_source_checksum:
        raise ValueError(
            "sealed candidate program source checksum mismatch before execution"
        )
    components = load_training_components_source_bytes(
        source_payload,
        module_name=f"true_rl_candidate_{manifest.candidate_id}",
        source_name=str(preflight.file_state(_CANDIDATE_COMPONENTS_PATH).path),
    )
    return manifest, components


def _read_preflighted_execution_file(
    preflight: CandidatePackagePreflight,
    relative_path: str,
) -> bytes:
    state = preflight.file_state(relative_path)
    with open_preflighted_candidate_file(preflight, relative_path) as source:
        payload = source.read(MAX_CANDIDATE_EXECUTION_FILE_BYTES + 1)
    if len(payload) != state.size_bytes:
        raise ValueError(f"candidate execution file size changed: {relative_path}")
    return payload


class RuntimeComponentError(ValueError):
    """Raised when candidate runtime outputs violate the runner contract."""


@dataclass
class RuntimeActivationState:
    step_calls: int = 0
    reward_term_calls: int = 0
    diagnostic_calls: int = 0
    last_reward_terms: list[dict[str, Any]] = field(default_factory=list)
    reward_term_activation: dict[str, dict[str, Any]] = field(default_factory=dict)
    last_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    last_candidate_reward_mean: float = 0.0
    max_abs_candidate_reward: float = 0.0
    max_abs_candidate_reward_frame: dict[str, Any] | None = None
    last_reward_weight_scale: float = 1.0
    # Source availability for the latest reward inputs, not whole-run coverage.
    input_provenance: dict[str, str] | None = None


class CandidateRuntimeAdapter:
    """Execute candidate callables through public training inputs only."""

    def __init__(
        self,
        *,
        manifest: CandidateManifest | None,
        components: TrainingComponents,
        num_envs: int,
        planned_training_steps: int,
        diagnostic_interval_steps: int = 24,
    ) -> None:
        for field_name, value in (
            ("num_envs", num_envs),
            ("planned_training_steps", planned_training_steps),
            ("diagnostic_interval_steps", diagnostic_interval_steps),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise RuntimeComponentError(
                    f"{field_name} must be a positive integer"
                )
        self.manifest = manifest
        self.components = components
        self.num_envs = num_envs
        self.planned_training_steps = planned_training_steps
        self.diagnostic_interval_steps = diagnostic_interval_steps
        self.runtime = RuntimeActivationState()
        self.training_numeric_capture = None
        self._previous_action: Any = None
        self._previous_base_lin_vel_xy: Any = None

    @classmethod
    def from_candidate_dir(
        cls,
        candidate_dir: str | Path,
        *,
        num_envs: int,
        planned_training_steps: int,
        diagnostic_interval_steps: int = 24,
        expected_candidate_id: str | None = None,
        expected_subject_manifest_hash: str | None = None,
        expected_program_source_checksum: str | None = None,
        expected_candidate_artifact_hash: str | None = None,
    ) -> "CandidateRuntimeAdapter":
        candidate_path = Path(candidate_dir)
        expected_identity = (
            expected_subject_manifest_hash,
            expected_program_source_checksum,
            expected_candidate_artifact_hash,
        )
        if any(value is not None for value in expected_identity):
            if expected_candidate_id is None or any(
                value is None for value in expected_identity
            ):
                raise ValueError(
                    "candidate runtime artifact identity must be complete"
                )
            sealed_artifact = load_sealed_candidate_artifact(
                candidate_path,
                change_set=CandidateChangeSet(frozenset()),
                expected_candidate_id=expected_candidate_id,
                expected_artifact_hash=expected_candidate_artifact_hash,
                expected_subject_manifest_hash=expected_subject_manifest_hash,
                expected_program_source_checksum=expected_program_source_checksum,
            )
            manifest, components = _load_bound_sealed_components(sealed_artifact)
        else:
            manifest = load_candidate_manifest(candidate_path)
            components = load_training_components(candidate_path)
        if expected_candidate_id is not None and manifest.candidate_id != expected_candidate_id:
            raise ValueError(
                "candidate manifest identity mismatch: "
                f"expected={expected_candidate_id!r} actual={manifest.candidate_id!r}"
            )
        validate_training_components(components)
        return cls(
            manifest=manifest,
            components=components,
            num_envs=num_envs,
            planned_training_steps=planned_training_steps,
            diagnostic_interval_steps=diagnostic_interval_steps,
        )

    def public_summary(self) -> dict[str, Any]:
        physical = bool(getattr(self, "_physical_program", False))
        return {
            "schema_version": "true_rl_agent.push_recovery.candidate_runtime.summary.v2" if physical else "true_rl_agent.push_recovery.candidate_runtime.summary.v1",
            "enabled": True,
            "candidate": self.manifest.to_public_dict() if self.manifest is not None else None,
            "components": self.components.to_public_summary(),
            "runtime_contract": runtime_contract(physical=physical),
        }

    def bind_training_numeric_capture(self, capture) -> None:
        if self.training_numeric_capture is not None:
            raise ValueError("Push training numeric capture already bound")
        if capture.source_step_offset != self.runtime.step_calls or capture.num_envs != self.num_envs:
            raise ValueError("Push training capture offset or environment count differs")
        self.training_numeric_capture = capture

    def training_numeric_semantics(self, *, control_dt: float) -> dict[str, Any]:
        physical = bool(getattr(self, "_physical_program", False))
        return {
            "raw_term": "guarded_and_clipped_program_term_before_weight",
            "weighted_term": "raw_term_times_declared_weight_times_recorded_schedule_scale",
            "total": "actual_candidate_sum_before_addition_to_input_reward",
            "control_dt_s": control_dt,
            "reward_execution": "complete_reward_rate" if physical else "additive_step_reward",
            "step_reward_relation": (
                "RewardManager multiplies returned_reward rate by control_dt_s once"
                if physical else "returned_reward already is a per_step reward; no further dt multiplication"
            ),
            "terms": [{"index": index, "name": term.name, "weight": float(term.weight),
                       "clip_min": term.clip_min, "clip_max": term.clip_max}
                      for index, term in enumerate(self.components.reward_terms)],
        }

    def enable_physical_program(self) -> None:
        if not self.components.reward_terms:
            raise RuntimeComponentError("Push v2 requires a nonempty full reward program")
        self._physical_program = True

    @classmethod
    def from_reference_program(
        cls, program, *, num_envs: int, planned_training_steps: int,
    ) -> "CandidateRuntimeAdapter":
        from true_rl_agent.training_programs import VerifiedReferenceProgramArtifact

        if not isinstance(program, VerifiedReferenceProgramArtifact):
            raise TypeError("Push reference requires verified source bytes")
        components = load_training_components_source_bytes(
            program.source_bytes,
            module_name=f"true_rl_reference_{program.manifest.reference_id}",
            source_name=program.manifest.source_uri,
        )
        validate_training_components(components)
        return cls(manifest=None, components=components, num_envs=num_envs,
                   planned_training_steps=planned_training_steps)

    def runtime_summary(self) -> dict[str, Any]:
        reward_stats = _stats_from_last_terms(self.runtime.last_reward_terms)
        diagnostic_outputs = _diagnostic_output_map(self.runtime.last_diagnostics)
        return {
            "schema_version": "true_rl_agent.push_recovery.candidate_runtime.v2" if getattr(self, "_physical_program", False) else "true_rl_agent.push_recovery.candidate_runtime.v1",
            "step_calls": self.runtime.step_calls,
            "reward_term_count": len(self.components.reward_terms),
            "diagnostic_term_count": len(self.components.diagnostics),
            "reward_term_calls": self.runtime.reward_term_calls,
            "diagnostic_calls": self.runtime.diagnostic_calls,
            "nonzero_reward_contribution": self.runtime.max_abs_candidate_reward > 1.0e-12,
            "last_candidate_reward_mean": self.runtime.last_candidate_reward_mean,
            "max_abs_candidate_reward": self.runtime.max_abs_candidate_reward,
            "max_abs_candidate_reward_frame": self.runtime.max_abs_candidate_reward_frame,
            "last_reward_weight_scale": self.runtime.last_reward_weight_scale,
            "reward_output_stats": reward_stats,
            "diagnostic_outputs": diagnostic_outputs,
            "last_reward_terms": self.runtime.last_reward_terms,
            "component_activation_records": self._component_activation_records(),
            "input_provenance": self.runtime.input_provenance or {},
            "feature_activation_records": [],
            "capability_activation_records": self._capability_activation_records(),
        }

    def apply_step_rewards(
        self,
        *,
        env: Any,
        actions: Any,
        rewards: Any,
        reset_mask: Any = None,
        step_extras: Mapping[str, Any] | None = None,
    ) -> Any:
        self.runtime.step_calls += 1
        progress = min(1.0, self.runtime.step_calls / self.planned_training_steps)
        reward_weight_scale = self._reward_weight_scale()
        self.runtime.last_reward_weight_scale = reward_weight_scale
        inputs = self._build_inputs(
            env=env,
            actions=actions,
            rewards=rewards,
            reset_mask=reset_mask,
            progress=progress,
            step_extras=step_extras,
        )

        candidate_reward = None
        reward_summaries: list[dict[str, Any]] = []
        weighted_terms: list[Any] = []
        capture = self.training_numeric_capture
        capture_this_step = capture is not None and capture.wants_step(self.runtime.step_calls)
        raw_terms: list[Any] = []
        for term in self.components.reward_terms:
            try:
                raw_value = term.fn(inputs)
            except Exception as exc:
                if is_infrastructure_runtime_exception(exc):
                    raise
                raise PushCandidateProgramFailure() from exc
            try:
                value, summary = guard_vector(
                    raw_value,
                    name=term.name,
                    kind="reward",
                    num_envs=self.num_envs,
                    clip_min=term.clip_min,
                    clip_max=term.clip_max,
                )
            except RuntimeComponentError as exc:
                raise PushCandidateProgramFailure() from exc
            effective_weight = float(term.weight) * reward_weight_scale
            weighted = _scale_like(value, rewards, effective_weight)
            weighted_terms.append(weighted)
            if capture_this_step:
                raw_terms.append(value)
            candidate_reward = (
                weighted
                if candidate_reward is None
                else _add_like(candidate_reward, weighted)
            )
            summary.update(
                {
                    "weight": float(term.weight),
                    "reward_weight_scale": reward_weight_scale,
                    "effective_weight": effective_weight,
                }
            )
            if self.runtime.step_calls % self.diagnostic_interval_steps == 0:
                self._record_reward_term_activation(
                    name=term.name,
                    weighted_value=weighted,
                    effective_weight=effective_weight,
                )
            reward_summaries.append(summary)

        self.runtime.reward_term_calls += len(self.components.reward_terms)
        self.runtime.last_reward_terms = reward_summaries

        if self.components.diagnostics and self.runtime.step_calls % self.diagnostic_interval_steps == 0:
            diagnostics = []
            for term in self.components.diagnostics:
                try:
                    payload = term.fn(inputs)
                except Exception as exc:
                    if is_infrastructure_runtime_exception(exc):
                        raise
                    raise PushCandidateProgramFailure() from exc
                diagnostics.append(
                    _summarize_diagnostic(term.name, payload, self.num_envs)
                )
            self.runtime.diagnostic_calls += len(self.components.diagnostics)
            self.runtime.last_diagnostics = diagnostics

        if candidate_reward is None:
            self.runtime.last_candidate_reward_mean = 0.0
            if capture is not None:
                capture.observe(self.runtime.step_calls,
                    {"input_reward": rewards, "returned_reward": rewards} if capture_this_step else None,
                    metadata={"reward_weight_scale": reward_weight_scale, "reward_term_count": 0})
            return rewards
        self.runtime.last_candidate_reward_mean = _vector_mean(candidate_reward)
        step_abs_max = _vector_abs_max(candidate_reward)
        if step_abs_max > self.runtime.max_abs_candidate_reward:
            self.runtime.max_abs_candidate_reward_frame = self._reward_peak_frame(
                candidate_reward, weighted_terms, reward_summaries,
            )
        self.runtime.max_abs_candidate_reward = max(
            self.runtime.max_abs_candidate_reward,
            step_abs_max,
        )
        returned_reward = _add_like(rewards, candidate_reward)
        if capture is not None:
            arrays = None
            if capture_this_step:
                arrays = {"input_reward": rewards, "candidate_total": candidate_reward,
                          "returned_reward": returned_reward}
                for index, (raw, weighted) in enumerate(zip(raw_terms, weighted_terms, strict=True)):
                    arrays[f"raw_term_{index:03d}"] = raw
                    arrays[f"weighted_term_{index:03d}"] = weighted
            capture.observe(self.runtime.step_calls, arrays,
                            metadata={"reward_weight_scale": reward_weight_scale})
        return returned_reward

    def _reward_peak_frame(
        self,
        candidate_reward: Any,
        weighted_terms: list[Any],
        reward_summaries: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Keep one same-step/env diagnostic, only when the global peak grows."""
        torch_mod = torch_for(candidate_reward)
        if torch_mod is not None:
            env_index = int(candidate_reward.detach().abs().argmax().item())
            values = torch_mod.stack([
                candidate_reward[env_index],
                *(value[env_index] for value in weighted_terms),
            ]).detach().cpu().tolist()
        else:
            env_index = max(range(self.num_envs), key=lambda i: abs(candidate_reward[i]))
            values = [candidate_reward[env_index], *(value[env_index] for value in weighted_terms)]
        return {
            "schema_version": "true_rl_agent.push_recovery.reward_peak_frame.v1",
            "scope": "one_running_absolute_peak_of_candidate_reward",
            "units": (
                "reward_rate_before_control_dt_integration"
                if getattr(self, "_physical_program", False)
                else "legacy_reward_contribution"
            ),
            "control_step_1based": self.runtime.step_calls,
            "env_index_0based": env_index,
            "signed_total": float(values[0]),
            "term_semantics": "signed contributions after clipping and effective weights; summed in runtime dtype and term order for signed_total",
            "terms": [
                {
                    "name": summary["name"],
                    "effective_weight": summary["effective_weight"],
                    "signed_weighted_contribution": float(value),
                }
                for summary, value in zip(reward_summaries, values[1:], strict=True)
            ],
        }

    def _record_reward_term_activation(
        self,
        *,
        name: str,
        weighted_value: Any,
        effective_weight: float,
    ) -> None:
        record = self.runtime.reward_term_activation.setdefault(
            name,
            {
                "name": name,
                "observation_count": 0,
                "nonzero_observation_count": 0,
                "max_abs_weighted_contribution": 0.0,
                "last_weighted_mean": 0.0,
                "effective_weight": float(effective_weight),
                "sampling_interval_steps": self.diagnostic_interval_steps,
            },
        )
        max_abs = _vector_abs_max(weighted_value)
        record["observation_count"] = int(record["observation_count"]) + 1
        if max_abs > 1.0e-12:
            record["nonzero_observation_count"] = int(record["nonzero_observation_count"]) + 1
        record["max_abs_weighted_contribution"] = max(
            float(record["max_abs_weighted_contribution"]),
            float(max_abs),
        )
        record["last_weighted_mean"] = _round(_vector_mean(weighted_value))
        record["effective_weight"] = float(effective_weight)

    def _component_activation_records(self) -> list[dict[str, Any]]:
        summary: list[dict[str, Any]] = []
        for term in self.components.reward_terms:
            record = self.runtime.reward_term_activation.get(term.name)
            if record is None:
                continue
            observation_count = int(record["observation_count"])
            nonzero_observation_count = int(record["nonzero_observation_count"])
            summary.append(
                {
                    **record,
                    "component_kind": "reward_term",
                    "nonzero_observation_fraction": _round(
                        nonzero_observation_count / observation_count
                        if observation_count > 0
                        else 0.0
                    ),
                }
            )
        return summary

    def _reward_weight_scale(self) -> float:
        scale = 1.0
        found = False
        for schedule in self.components.schedules:
            if schedule.key == REWARD_WEIGHT_SCALE_KEY:
                scale *= float(schedule.value)
                found = True
        if not found:
            return 1.0
        bounds = schedule_bounds_for_key(REWARD_WEIGHT_SCALE_KEY)
        if bounds is None:
            return scale
        min_value, max_value = bounds
        return max(min_value, min(max_value, scale))

    def _build_inputs(
        self,
        *,
        env: Any,
        actions: Any,
        rewards: Any,
        reset_mask: Any,
        progress: float,
        step_extras: Mapping[str, Any] | None = None,
    ) -> PushRecoveryRewardInputs:
        base = unwrap_env(env)
        robot_data = robot_data_from_env(base)
        height, height_obs = root_height(robot_data, rewards)
        roll, pitch, rp_obs = roll_pitch(robot_data, rewards)
        base_pos_xy, pos_obs = base_pos_xy_from_env(base, robot_data, rewards)
        lin_vel, lin_obs = observed_or_zero(first_attr(robot_data, ("root_lin_vel_b", "root_lin_vel_w", "body_lin_vel_w")), rewards, width=3)
        ang_vel, ang_obs = observed_or_zero(first_attr(robot_data, ("root_ang_vel_b", "root_ang_vel_w", "body_ang_vel_w")), rewards, width=3)
        projected_gravity, gravity_obs = projected_gravity_from_quat(robot_data, rewards)
        commanded_base_vel, command_obs = commanded_base_vel_from_env(base, rewards)
        joint_pos, jpos_obs = observed_or_zero(first_attr(robot_data, ("joint_pos",)), rewards, width=joint_width(robot_data))
        joint_vel, jvel_obs = observed_or_zero(first_attr(robot_data, ("joint_vel",)), rewards, width=joint_width(robot_data))
        joint_pos_error, jerr_obs = joint_pos_error_from_data(robot_data, joint_pos, rewards)
        last_action, action_obs = observed_or_zero(actions, rewards)
        prev_action, prev_obs = previous_action_or_zero(self._previous_action, actions, rewards)
        action_delta, delta_obs = action_delta_or_zero(actions, prev_action, rewards)
        self._previous_action = mask_reset_actions(clone_public_value(actions), reset_mask)
        current_base_lin_vel_xy = matrix_columns(lin_vel, 2) if lin_obs else None
        push_active, push_vec, push_obs = public_push_signal_from_env(
            env=base,
            step_extras=step_extras,
            rewards=rewards,
            current_base_lin_vel_xy=current_base_lin_vel_xy,
            previous_base_lin_vel_xy=self._previous_base_lin_vel_xy,
        )
        if current_base_lin_vel_xy is not None:
            self._previous_base_lin_vel_xy = mask_reset_actions(
                clone_public_value(current_base_lin_vel_xy),
                reset_mask,
            )
        else:
            self._previous_base_lin_vel_xy = None
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
            "push_vec_public": push_vec,
            "progress_fraction": constant_like(rewards, progress),
        }
        provenance = {
            "base_height": provenance_tag(height_obs),
            "roll": provenance_tag(rp_obs),
            "pitch": provenance_tag(rp_obs),
            "base_pos_xy": provenance_tag(pos_obs),
            "base_lin_vel": provenance_tag(lin_obs),
            "base_ang_vel": provenance_tag(ang_obs),
            "projected_gravity": provenance_tag(gravity_obs),
            "commanded_base_vel": provenance_tag(command_obs),
            "joint_pos": provenance_tag(jpos_obs),
            "joint_vel": provenance_tag(jvel_obs),
            "joint_pos_error": provenance_tag(jerr_obs),
            "last_action": provenance_tag(action_obs),
            "prev_action": provenance_tag(prev_obs),
            "action_delta": provenance_tag(delta_obs and action_obs and prev_obs),
            "push_active": provenance_tag(push_obs),
            "push_vec_public": provenance_tag(push_obs),
            "progress_fraction": "runner_owned",
        }
        # Refresh every field together; the first call's zero-initialized action
        # history must not describe later calls that have observed history.
        self.runtime.input_provenance = provenance
        if getattr(base, "_push_full_program_runtime", False):
            p = base.cfg.push_protocol_parameters
            world_linear = as_torch(robot_data.root_lin_vel_w)
            world_angular = as_torch(robot_data.root_ang_vel_w)
            relative_height = as_torch(robot_data.root_pos_w)[:, 2] - as_torch(base.scene.env_origins)[:, 2]
            mapping.update(
                base_height=relative_height,
                nominal_base_height=constant_like(rewards, p["robot_scale_m"]),
                base_height_relative=relative_height / p["robot_scale_m"],
                base_lin_vel_scaled=world_linear / p["velocity_scale_mps"],
                base_ang_vel_scaled=world_angular * p["time_scale_s"],
                terminated_non_timeout=as_torch(base.termination_manager.terminated).to(dtype=rewards.dtype),
            )
            self.runtime.input_provenance.update({name: "task_owned_bound_scale" for name in (
                "nominal_base_height", "base_height_relative", "base_lin_vel_scaled", "base_ang_vel_scaled")})
            self.runtime.input_provenance["terminated_non_timeout"] = "trusted_pre_reset_termination"
            return PushRecoveryV2RewardInputs(**mapping)
        return PushRecoveryRewardInputs(**mapping)

    def _capability_activation_records(self) -> list[str]:
        records = []
        if self.components.reward_terms:
            records.append("reward_callable")
        if self.components.diagnostics:
            records.append("diagnostic")
        if any(schedule.key == REWARD_WEIGHT_SCALE_KEY for schedule in self.components.schedules):
            record = schedule_runtime_activation_record(REWARD_WEIGHT_SCALE_KEY)
            if record is not None:
                records.append(record)
        return records


class CandidateRewardVecEnvWrapper:
    """Proxy an RSL-RL VecEnv and insert candidate reward contributions."""

    def __init__(self, env: Any, adapter: CandidateRuntimeAdapter) -> None:
        self.env = env
        self.candidate_runtime_adapter = adapter

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def step(self, actions: Any) -> tuple[Any, ...]:
        result = self.env.step(actions)
        if not isinstance(result, tuple):
            raise RuntimeComponentError("VecEnv.step must return a tuple")
        reward_index = find_reward_index(result, self.candidate_runtime_adapter.num_envs)
        rewards = result[reward_index]
        reset_mask = reset_mask_from_step_result(result, self.candidate_runtime_adapter.num_envs)
        updated_rewards = self.candidate_runtime_adapter.apply_step_rewards(
            env=self.env,
            actions=actions,
            rewards=rewards,
            reset_mask=reset_mask,
            step_extras=step_extras_from_step_result(result),
        )
        mutable = list(result)
        mutable[reward_index] = updated_rewards
        return tuple(mutable)


def runtime_contract(*, physical: bool = False) -> dict[str, Any]:
    reserved_zero_fallback_inputs = list(reserved_zero_fallback_reward_input_fields())
    result = {
        "candidate_receives": list(push_recovery_reward_input_fields()),
        "candidate_receives_prompt_effective": list(prompt_effective_reward_input_fields()),
        "public_push_signal_inputs": list(public_push_signal_reward_input_fields()),
        "reserved_zero_fallback_inputs": reserved_zero_fallback_inputs,
        "candidate_does_not_receive": ["env", "runner", "optimizer", "checkpoint_selector", "eval_config", "success_metric"],
        "reward_terms_added_to_env_reward": True,
        "diagnostics_public_numeric_summary_only": True,
        "schedule_controls": {
            "effective_keys": list(effective_schedule_keys()),
            "bounds": schedule_key_bounds(),
        },
    }
    if physical:
        result.update(candidate_receives=list(PushRecoveryV2RewardInputs.field_names()),
            candidate_receives_prompt_effective=list(PushRecoveryV2RewardInputs.field_names()),
            reward_terms_added_to_env_reward=False, reward_mode="complete_program_replacement",
            reward_output="rate", rate_integration="reward_manager_once_by_control_dt",
            transition_source="before_auto_reset", push_signal_source="already_executed_event_only")
    return result


def validate_training_components(components: TrainingComponents) -> None:
    if not isinstance(components, TrainingComponents):
        raise RuntimeComponentError("candidate entrypoint must return TrainingComponents")
    try:
        require_component_name(components.name, "TrainingComponents.name")
    except ValueError as exc:
        raise RuntimeComponentError("TrainingComponents.name is invalid") from exc
    names: set[str] = set()
    for family, terms in (
        ("reward_terms", components.reward_terms),
        ("diagnostics", components.diagnostics),
        ("schedules", components.schedules),
    ):
        for term in terms:
            name = getattr(term, "name", "")
            try:
                require_component_name(name, f"{family} term name")
            except ValueError as exc:
                raise RuntimeComponentError(f"{family} term name is invalid") from exc
            if name in names:
                raise RuntimeComponentError(f"duplicate component name: {name}")
            names.add(name)
    for term in components.reward_terms:
        if not callable(term.fn):
            raise RuntimeComponentError(f"reward term is not callable: {term.name}")
    for term in components.diagnostics:
        if not callable(term.fn):
            raise RuntimeComponentError(f"diagnostic term is not callable: {term.name}")
    for term in components.schedules:
        if term.key != REWARD_WEIGHT_SCALE_KEY:
            raise RuntimeComponentError(f"unsupported schedule key: {term.key}")


def guard_vector(
    value: Any,
    *,
    name: str,
    kind: str,
    num_envs: int,
    clip_min: float | None = None,
    clip_max: float | None = None,
) -> tuple[Any, dict[str, Any]]:
    torch_mod = torch_for(value)
    clip_applied = False
    if torch_mod is not None:
        if tuple(value.shape) != (num_envs,):
            raise RuntimeComponentError(f"{kind} term {name} must return shape ({num_envs},)")
        if not bool(torch_mod.isfinite(value).all().item()):
            raise RuntimeComponentError(f"{kind} term {name} returned non-finite values")
        guarded = value
        if clip_min is not None or clip_max is not None:
            guarded = torch_mod.clamp(
                guarded,
                min=float(clip_min) if clip_min is not None else None,
                max=float(clip_max) if clip_max is not None else None,
            )
            clip_applied = True
        return guarded, {
            "name": name,
            "kind": kind,
            "shape": tuple(guarded.shape),
            "dtype": str(guarded.dtype),
            "num_envs": num_envs,
            "finite": True,
            "clip_min": clip_min,
            "clip_max": clip_max,
            "clip_applied": clip_applied,
            "mean": _round(_vector_mean(guarded)),
        }

    try:
        values = to_float_list(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeComponentError(
            f"{kind} term {name} must return a numeric vector"
        ) from exc
    if len(values) != num_envs:
        raise RuntimeComponentError(f"{kind} term {name} must return {num_envs} values")
    if any(not math.isfinite(item) for item in values):
        raise RuntimeComponentError(f"{kind} term {name} returned non-finite values")
    if clip_min is not None or clip_max is not None:
        low = float(clip_min) if clip_min is not None else None
        high = float(clip_max) if clip_max is not None else None
        values = [min(high, item) if high is not None else item for item in values]
        values = [max(low, item) if low is not None else item for item in values]
        clip_applied = True
    return values, {
        "name": name,
        "kind": kind,
        "shape": (num_envs,),
        "dtype": "float",
        "num_envs": num_envs,
        "finite": True,
        "clip_min": clip_min,
        "clip_max": clip_max,
        "clip_applied": clip_applied,
        "mean": _round(_vector_mean(values)),
    }


def find_reward_index(result: tuple[Any, ...], num_envs: int) -> int:
    if len(result) < 2:
        raise RuntimeComponentError("VecEnv.step result is missing the reward slot")
    reward = result[1]
    if shape_tuple(reward) != (num_envs,) or is_bool_vector(reward):
        raise RuntimeComponentError("VecEnv.step reward slot must be a numeric vector at index 1")
    return 1


def reset_mask_from_step_result(result: tuple[Any, ...], num_envs: int) -> Any:
    if len(result) >= 5:
        return combine_reset_masks(result[2], result[3], num_envs)
    if len(result) >= 3 and shape_tuple(result[2]) == (num_envs,):
        return result[2]
    return None


def step_extras_from_step_result(result: tuple[Any, ...]) -> Mapping[str, Any] | None:
    if result and isinstance(result[-1], Mapping):
        return result[-1]
    return None


def combine_reset_masks(first: Any, second: Any, num_envs: int) -> Any:
    torch_mod = torch_for(first)
    if torch_mod is not None:
        second_mask = second.bool() if torch_for(second) is not None else torch_mod.tensor(to_bool_list(second), dtype=torch_mod.bool, device=first.device)
        return first.bool() | second_mask
    first_list = to_bool_list(first)
    second_list = to_bool_list(second)
    if len(first_list) != num_envs or len(second_list) != num_envs:
        return None
    return [left or right for left, right in zip(first_list, second_list, strict=True)]


def unwrap_env(env: Any) -> Any:
    current = env
    for _ in range(8):
        unwrapped = getattr(current, "unwrapped", None)
        if unwrapped is None or unwrapped is current:
            return current
        current = unwrapped
    return current


def robot_data_from_env(env: Any) -> Any:
    robot = robot_object(env)
    return getattr(robot, "data", None)


def robot_object(env: Any) -> Any:
    scene = getattr(env, "scene", None)
    robot = None
    if scene is not None:
        try:
            robot = scene["robot"]
        except Exception:
            robot = getattr(scene, "robot", None)
    if robot is None:
        robot = getattr(env, "robot", None)
    return robot


def provenance_tag(observed: bool) -> str:
    return "observed" if observed else "zero_fallback"


def root_height(robot_data: Any, rewards: Any) -> tuple[Any, bool]:
    root_pos = first_attr(robot_data, ("root_pos_w", "root_link_pos_w"))
    column = column_value(root_pos, 2)
    if column is not None:
        return column, True
    return constant_like(rewards, 0.0), False


def base_pos_xy_from_env(env: Any, robot_data: Any, rewards: Any) -> tuple[Any, bool]:
    root_pos = first_attr(robot_data, ("root_pos_w", "root_link_pos_w"))
    xy = matrix_columns(root_pos, 2)
    origins = matrix_columns(getattr(getattr(env, "scene", None), "env_origins", None), 2)
    if xy is not None and origins is not None:
        diff = subtract_values(xy, origins)
        if diff is not None:
            return diff, True
    if xy is not None:
        return xy, True
    return zero_matrix_like(rewards, width=2), False


def roll_pitch(robot_data: Any, rewards: Any) -> tuple[Any, Any, bool]:
    quat = first_attr(robot_data, ("root_quat_w", "root_link_quat_w"))
    torch_mod = torch_for(quat)
    if torch_mod is None or shape_tuple(quat)[-1:] != (4,):
        zero = constant_like(rewards, 0.0)
        return zero, zero, False
    w, x, y, z = quaternion_wxyz(quat).unbind(dim=-1)
    roll = torch_mod.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch_arg = torch_mod.clamp(2.0 * (w * y - z * x), -1.0, 1.0)
    return roll, torch_mod.asin(pitch_arg), True


def projected_gravity_from_quat(robot_data: Any, rewards: Any) -> tuple[Any, bool]:
    quat = first_attr(robot_data, ("root_quat_w", "root_link_quat_w"))
    torch_mod = torch_for(quat)
    if torch_mod is None or shape_tuple(quat)[-1:] != (4,):
        return zero_matrix_like(rewards, width=3), False
    w, x, y, z = quaternion_wxyz(quat).unbind(dim=-1)
    return torch_mod.stack(
        (
            -2.0 * (x * z - w * y),
            -2.0 * (y * z + w * x),
            -(1.0 - 2.0 * (x * x + y * y)),
        ),
        dim=-1,
    ), True


def commanded_base_vel_from_env(env: Any, rewards: Any) -> tuple[Any, bool]:
    command_manager = getattr(env, "command_manager", None)
    get_command = getattr(command_manager, "get_command", None)
    if callable(get_command):
        for name in ("base_velocity", "base_vel"):
            try:
                observed, ok = observed_or_zero(get_command(name), rewards, width=3)
            except Exception:
                observed, ok = None, False
            if ok:
                return observed, True
    return zero_matrix_like(rewards, width=3), False


def public_push_signal_from_env(
    *,
    env: Any,
    step_extras: Mapping[str, Any] | None,
    rewards: Any,
    current_base_lin_vel_xy: Any,
    previous_base_lin_vel_xy: Any,
) -> tuple[Any, Any, bool]:
    physical_state = getattr(env, "_push_runtime_state", None)
    if physical_state is not None:
        # Exact, already executed events. Autonomous motion is never an event.
        return physical_state.public_active.clone(), physical_state.public_delta.clone(), True
    sources = (step_extras, env)
    explicit_vec = public_signal_value(sources, PUBLIC_PUSH_VEC_KEYS)
    push_vec = public_push_vec_or_none(explicit_vec, rewards)
    if push_vec is None:
        push_vec = public_velocity_delta_or_none(current_base_lin_vel_xy, previous_base_lin_vel_xy)
        if push_vec is not None:
            push_vec = threshold_public_velocity_delta(push_vec)
    if push_vec is not None:
        explicit_active = public_signal_value(sources, PUBLIC_PUSH_ACTIVE_KEYS)
        active = public_push_active_or_none(explicit_active, rewards)
        if active is None:
            active = push_active_from_public_delta(push_vec, rewards)
        return active, push_vec, True

    explicit_active = public_signal_value(sources, PUBLIC_PUSH_ACTIVE_KEYS)
    active = public_push_active_or_none(explicit_active, rewards)
    if active is not None:
        return active, zero_matrix_like(rewards, width=2), True
    return constant_like(rewards, 0.0), zero_matrix_like(rewards, width=2), False


def public_signal_value(sources: tuple[Any, ...], names: tuple[str, ...]) -> Any | None:
    nested_keys = ("push", "public_push", "push_signal", "events")
    for source in sources:
        if source is None:
            continue
        if isinstance(source, Mapping):
            value = mapping_value(source, names)
            if value is not None:
                return value
            for key in nested_keys:
                nested = source.get(key)
                if isinstance(nested, Mapping):
                    value = mapping_value(nested, names)
                    if value is not None:
                        return value
            continue
        for name in names:
            value = getattr(source, name, None)
            if value is not None:
                return value
    return None


def mapping_value(mapping: Mapping[str, Any], names: tuple[str, ...]) -> Any | None:
    for name in names:
        value = mapping.get(name)
        if value is not None:
            return value
    return None


def public_push_active_or_none(value: Any, rewards: Any) -> Any | None:
    if value is None or shape_tuple(value) != (vector_length(rewards),):
        return None
    torch_mod = torch_for(value)
    if torch_mod is not None:
        reward_torch = torch_for(rewards)
        dtype = rewards.dtype if reward_torch is not None else value.dtype
        device = rewards.device if reward_torch is not None else value.device
        return value.to(device=device, dtype=dtype)
    return [1.0 if bool(item) else 0.0 for item in value]


def public_push_vec_or_none(value: Any, rewards: Any) -> Any | None:
    if value is None or shape_tuple(value) != (vector_length(rewards), 2):
        return None
    torch_mod = torch_for(value)
    if torch_mod is not None:
        reward_torch = torch_for(rewards)
        dtype = rewards.dtype if reward_torch is not None else value.dtype
        device = rewards.device if reward_torch is not None else value.device
        return value.to(device=device, dtype=dtype)
    return [[float(item) for item in row[:2]] for row in value]


def public_velocity_delta_or_none(current_xy: Any, previous_xy: Any) -> Any | None:
    if current_xy is None or previous_xy is None:
        return None
    if shape_tuple(current_xy) != shape_tuple(previous_xy) or shape_tuple(current_xy)[-1:] != (2,):
        return None
    return subtract_values(current_xy, previous_xy)


def threshold_public_velocity_delta(push_vec: Any) -> Any:
    torch_mod = torch_for(push_vec)
    if torch_mod is not None:
        active = torch_mod.linalg.vector_norm(push_vec, dim=1) >= PUBLIC_PUSH_DELTA_THRESHOLD_MPS
        return push_vec * active.to(device=push_vec.device, dtype=push_vec.dtype).unsqueeze(-1)
    rows = []
    for row in push_vec:
        x = float(row[0])
        y = float(row[1])
        if math.sqrt(x * x + y * y) >= PUBLIC_PUSH_DELTA_THRESHOLD_MPS:
            rows.append([x, y])
        else:
            rows.append([0.0, 0.0])
    return rows


def push_active_from_public_delta(push_vec: Any, rewards: Any) -> Any:
    torch_mod = torch_for(push_vec)
    if torch_mod is not None:
        active = torch_mod.linalg.vector_norm(push_vec, dim=1) >= PUBLIC_PUSH_DELTA_THRESHOLD_MPS
        reward_torch = torch_for(rewards)
        dtype = rewards.dtype if reward_torch is not None else push_vec.dtype
        device = rewards.device if reward_torch is not None else push_vec.device
        return active.to(device=device, dtype=dtype)
    active_values = []
    for row in push_vec:
        x = float(row[0])
        y = float(row[1])
        active_values.append(1.0 if math.sqrt(x * x + y * y) >= PUBLIC_PUSH_DELTA_THRESHOLD_MPS else 0.0)
    return active_values


def joint_width(robot_data: Any) -> int | None:
    for attr_name in ("joint_pos", "joint_vel", "default_joint_pos", "default_joint_pos_target"):
        width = trailing_width(first_attr(robot_data, (attr_name,)))
        if width is not None:
            return width
    return None


def joint_pos_error_from_data(robot_data: Any, joint_pos: Any, rewards: Any) -> tuple[Any, bool]:
    default_joint_pos = first_attr(robot_data, ("default_joint_pos", "default_joint_pos_target"))
    if default_joint_pos is None:
        return zero_like_value(joint_pos, rewards), False
    diff = subtract_values(joint_pos, broadcast_first_dim(default_joint_pos, shape_tuple(joint_pos)))
    if diff is None:
        return zero_like_value(joint_pos, rewards), False
    return diff, True


def observed_or_zero(value: Any, rewards: Any, *, width: int | None = None) -> tuple[Any, bool]:
    if value is not None:
        if width is None or shape_tuple(value)[-1:] == (width,):
            return value, True
    if width is not None:
        return zero_matrix_like(rewards, width=width), False
    return zero_like_value(value, rewards), False


def previous_action_or_zero(previous: Any, actions: Any, rewards: Any) -> tuple[Any, bool]:
    if previous is None:
        return zero_like_value(actions, rewards), False
    return previous, True


def action_delta_or_zero(actions: Any, previous: Any, rewards: Any) -> tuple[Any, bool]:
    diff = subtract_values(actions, previous)
    if diff is None:
        return zero_like_value(actions, rewards), False
    return diff, True


def mask_reset_actions(actions: Any, reset_mask: Any) -> Any:
    if reset_mask is None:
        return actions
    torch_mod = torch_for(actions)
    if torch_mod is not None:
        mask = reset_mask.bool() if torch_for(reset_mask) is not None else torch_mod.tensor(to_bool_list(reset_mask), dtype=torch_mod.bool, device=actions.device)
        out = actions.clone()
        out[mask] = 0
        return out
    values = clone_public_value(actions)
    if isinstance(values, list):
        masks = to_bool_list(reset_mask)
        return [zero_like_python(row) if index < len(masks) and masks[index] else row for index, row in enumerate(values)]
    return values


def first_attr(obj: Any, names: tuple[str, ...]) -> Any:
    if obj is None:
        return None
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return as_torch(value)
    return None


def column_value(value: Any, index: int) -> Any | None:
    torch_mod = torch_for(value)
    if torch_mod is not None and len(value.shape) >= 2 and value.shape[1] > index:
        return value[:, index]
    if isinstance(value, list) and value and isinstance(value[0], (list, tuple)) and len(value[0]) > index:
        return [float(row[index]) for row in value]
    return None


def matrix_columns(value: Any, width: int) -> Any | None:
    torch_mod = torch_for(value)
    if torch_mod is not None and len(value.shape) >= 2 and value.shape[1] >= width:
        return value[:, :width]
    if isinstance(value, list) and value and isinstance(value[0], (list, tuple)) and len(value[0]) >= width:
        return [[float(item) for item in row[:width]] for row in value]
    return None


def zero_matrix_like(rewards: Any, *, width: int | None) -> Any:
    width = int(width or 0)
    torch_mod = torch_for(rewards)
    if torch_mod is not None:
        return torch_mod.zeros((int(rewards.shape[0]), width), device=rewards.device, dtype=rewards.dtype)
    return [[0.0 for _ in range(width)] for _ in range(vector_length(rewards))]


def constant_like(rewards: Any, value: float) -> Any:
    torch_mod = torch_for(rewards)
    if torch_mod is not None:
        return torch_mod.full_like(rewards, float(value))
    return [float(value) for _ in range(vector_length(rewards))]


def zero_like_value(value: Any, rewards: Any) -> Any:
    torch_mod = torch_for(value)
    if torch_mod is not None:
        return torch_mod.zeros_like(value)
    shape = shape_tuple(value)
    if len(shape) == 2:
        return [[0.0 for _ in range(shape[1])] for _ in range(shape[0])]
    if len(shape) == 1:
        return [0.0 for _ in range(shape[0])]
    return constant_like(rewards, 0.0)


def subtract_values(left: Any, right: Any) -> Any | None:
    if left is None or right is None:
        return None
    torch_mod = torch_for(left)
    if torch_mod is not None and torch_for(right) is not None:
        if shape_tuple(left) == shape_tuple(right):
            return left - right
        return None
    if isinstance(left, list) and isinstance(right, list) and shape_tuple(left) == shape_tuple(right):
        if left and isinstance(left[0], list):
            return [[float(a) - float(b) for a, b in zip(row_l, row_r, strict=True)] for row_l, row_r in zip(left, right, strict=True)]
        return [float(a) - float(b) for a, b in zip(left, right, strict=True)]
    return None


def broadcast_first_dim(value: Any, target_shape: tuple[int, ...]) -> Any:
    if value is None:
        return None
    torch_mod = torch_for(value)
    if torch_mod is not None:
        if tuple(value.shape) == target_shape:
            return value
        if len(target_shape) == 2 and len(value.shape) == 1 and value.shape[0] == target_shape[1]:
            return value.unsqueeze(0).expand(target_shape)
    if isinstance(value, list) and len(target_shape) == 2:
        if shape_tuple(value) == target_shape:
            return value
        if len(value) == target_shape[1] and not isinstance(value[0], list):
            return [list(value) for _ in range(target_shape[0])]
    return value


def clone_public_value(value: Any) -> Any:
    torch_mod = torch_for(value)
    if torch_mod is not None:
        return value.detach().clone()
    if isinstance(value, list):
        return [clone_public_value(item) for item in value]
    if isinstance(value, tuple):
        return [clone_public_value(item) for item in value]
    return value


def _add_like(left: Any, right: Any) -> Any:
    torch_mod = torch_for(left)
    if torch_mod is not None:
        return left + right
    return [float(a) + float(b) for a, b in zip(to_float_list(left), to_float_list(right), strict=True)]


def _scale_like(value: Any, rewards: Any, scale: float) -> Any:
    torch_mod = torch_for(value)
    if torch_mod is not None:
        return value.to(device=rewards.device, dtype=rewards.dtype) * float(scale) if torch_for(rewards) is not None else value * float(scale)
    return [float(item) * float(scale) for item in to_float_list(value)]


def _vector_mean(value: Any) -> float:
    torch_mod = torch_for(value)
    if torch_mod is not None:
        return float(value.detach().to(dtype=torch_mod.float32).mean().cpu().item())
    values = to_float_list(value)
    return sum(values) / len(values) if values else 0.0


def _vector_abs_max(value: Any) -> float:
    torch_mod = torch_for(value)
    if torch_mod is not None:
        return float(torch_mod.abs(value).max().detach().cpu().item()) if value.numel() else 0.0
    values = to_float_list(value)
    return max((abs(item) for item in values), default=0.0)


def _stats_from_last_terms(terms: list[dict[str, Any]]) -> dict[str, Any]:
    if not terms:
        return {"mean": 0.0, "finite_fraction": 1.0}
    means = [float(term.get("mean", 0.0)) for term in terms]
    return {"mean": _round(sum(means) / len(means)), "finite_fraction": 1.0}


def _diagnostic_output_map(diagnostics: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(item.get("name") or f"diagnostic_{index}"): dict(item)
        for index, item in enumerate(diagnostics)
    }


def _summarize_diagnostic(name: str, payload: Any, num_envs: int) -> dict[str, Any]:
    if isinstance(payload, Mapping):
        fields = {str(key): summarize_public_value(value) for key, value in payload.items()}
    else:
        fields = {"value": summarize_public_value(payload)}
    return {"name": name, "num_envs": num_envs, "fields": fields}


def summarize_public_value(value: Any) -> Any:
    torch_mod = torch_for(value)
    if torch_mod is not None:
        if value.numel() == 0:
            return {"shape": tuple(value.shape), "mean": None}
        finite = bool(torch_mod.isfinite(value).all().item())
        return {"shape": tuple(value.shape), "mean": _round(_vector_mean(value)), "finite": finite}
    if isinstance(value, int | float | bool):
        return value
    if isinstance(value, list | tuple):
        try:
            values = to_float_list(value)
        except Exception:
            return {"type": type(value).__name__, "size": len(value)}
        return {"shape": shape_tuple(value), "mean": _round(sum(values) / len(values)) if values else None, "finite": all(math.isfinite(v) for v in values)}
    return {"type": type(value).__name__}


def torch_for(value: Any) -> Any:
    try:
        import torch  # type: ignore[import-not-found]
    except Exception:
        return None
    return torch if torch.is_tensor(value) else None


def shape_tuple(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            return tuple(int(item) for item in shape)
        except Exception:
            return ()
    if isinstance(value, list | tuple):
        if value and isinstance(value[0], (list, tuple)):
            return (len(value), len(value[0]))
        return (len(value),)
    return ()


def trailing_width(value: Any) -> int | None:
    shape = shape_tuple(value)
    if len(shape) >= 2:
        return shape[-1]
    return None


def vector_length(value: Any) -> int:
    shape = shape_tuple(value)
    if shape:
        return shape[0]
    return 1


def to_float_list(value: Any) -> list[float]:
    torch_mod = torch_for(value)
    if torch_mod is not None:
        flattened = value.detach().cpu().reshape(-1).tolist()
        return [float(item) for item in flattened]
    if isinstance(value, list | tuple):
        if value and isinstance(value[0], (list, tuple)):
            return [float(item) for row in value for item in row]
        return [float(item) for item in value]
    return [float(value)]


def to_bool_list(value: Any) -> list[bool]:
    torch_mod = torch_for(value)
    if torch_mod is not None:
        return [bool(item) for item in value.detach().cpu().reshape(-1).tolist()]
    if isinstance(value, list | tuple):
        return [bool(item) for item in value]
    return [bool(value)]


def is_bool_vector(value: Any) -> bool:
    torch_mod = torch_for(value)
    if torch_mod is not None:
        return str(value.dtype) == "torch.bool"
    return isinstance(value, list) and all(isinstance(item, bool) for item in value)


def zero_like_python(value: Any) -> Any:
    if isinstance(value, list):
        return [zero_like_python(item) for item in value]
    if isinstance(value, tuple):
        return [zero_like_python(item) for item in value]
    return 0.0


def _round(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)
