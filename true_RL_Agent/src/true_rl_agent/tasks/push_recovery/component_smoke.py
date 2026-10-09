"""Push recovery task-owned smoke fixture."""

from __future__ import annotations

from dataclasses import dataclass

from true_rl_agent.tasks.component_smoke import ComponentSmokeFixture
from true_rl_agent.tasks.push_recovery.config import PushRecoveryTaskConfig
from true_rl_agent.tasks.push_recovery.reward_inputs import PushRecoveryRewardInputs, PushRecoveryV2RewardInputs
from true_rl_agent.robots.defaults import build_default_robot_registry


@dataclass(frozen=True)
class PushRecoveryComponentSmokeAdapter:
    """Build CPU Tensor RewardInputs fixtures for push recovery candidate smoke checks."""

    config: PushRecoveryTaskConfig
    task_id: str = "unitree_push_recovery"

    @property
    def candidate_capability_ids(self) -> frozenset[str]:
        if self.config.push_protocol is not None:
            return frozenset({"reward_term", "diagnostic_term", "reward_weight_scale_schedule"})
        return frozenset()

    def build_component_smoke_fixture(self) -> ComponentSmokeFixture:
        num_envs = 4
        num_joints = 6
        num_actions = 6
        recovery_v2 = self.config.push_protocol is not None
        if recovery_v2:
            registry = build_default_robot_registry()
            profile = registry.profile(registry.runtime_adapter(self.config.runtime_adapter_id).robot_profile_id)
            num_joints = num_actions = len(profile.runtime_facts["joint_names"])
        inputs = PushRecoveryRewardInputs(
            base_height=_tensor([0.34, 0.39, 0.42, 0.36]),
            roll=_tensor([0.02, -0.04, 0.08, -0.03]),
            pitch=_tensor([0.03, 0.06, -0.05, 0.01]),
            base_pos_xy=_matrix(num_envs, 2, scale=0.05),
            base_lin_vel=_matrix(num_envs, 3, scale=0.10),
            base_ang_vel=_matrix(num_envs, 3, scale=0.08),
            projected_gravity=_matrix(num_envs, 3, scale=0.02),
            commanded_base_vel=_matrix(num_envs, 3, scale=0.05),
            joint_pos=_matrix(num_envs, num_joints, scale=0.03),
            joint_vel=_matrix(num_envs, num_joints, scale=0.04),
            joint_pos_error=_matrix(num_envs, num_joints, scale=0.02),
            last_action=_matrix(num_envs, num_actions, scale=0.05),
            prev_action=_matrix(num_envs, num_actions, scale=0.04),
            action_delta=_matrix(num_envs, num_actions, scale=0.01),
            push_active=_tensor([0.0, 1.0, 1.0, 0.0]),
            push_vec_public=_matrix(num_envs, 2, scale=0.20),
            progress_fraction=_tensor([0.05, 0.35, 0.65, 0.95]),
        )
        if recovery_v2:
            physical = self.config.push_protocol.resolve(self.config.push_robot_scale_m)
            length = physical["robot_scale_m"]
            relative_height = _tensor([1.0, 0.95, 0.80, 0.40])
            values = {name: getattr(inputs, name) for name in inputs.field_names()}
            values.update(
                base_height=relative_height * length,
                nominal_base_height=_tensor([length] * num_envs),
                base_height_relative=relative_height,
                base_lin_vel_scaled=inputs.base_lin_vel / physical["velocity_scale_mps"],
                base_ang_vel_scaled=inputs.base_ang_vel * physical["time_scale_s"],
                terminated_non_timeout=_tensor([0.0, 0.0, 0.0, 1.0]),
                projected_gravity=_tensor([[0.0, 0.0, -1.0]] * num_envs),
                commanded_base_vel=_tensor([[0.0, 0.0, 0.0]] * num_envs),
                push_vec_public=_tensor([[0.0, 0.0], [0.20, 0.0], [0.0, -0.20], [0.0, 0.0]]),
            )
            inputs = PushRecoveryV2RewardInputs(**values)
        return ComponentSmokeFixture(
            task_id=self.task_id,
            reward_inputs=inputs,
            expected_reward_output_shape=(num_envs,),
            public_summary={
                "fixture_owner": "task_owned_push_recovery",
                "num_envs": num_envs,
                "num_joints": num_joints,
                "num_actions": num_actions,
                "purpose": "authoring_time_reward_callable_smoke_only",
            },
        )


def _matrix(rows: int, cols: int, *, scale: float):
    return _tensor(
        [
            [scale * float((row + 1) * (col + 1)) for col in range(cols)]
            for row in range(rows)
        ]
    )


def _tensor(values):
    """Use the training Tensor API; Torch is required only when building a fixture."""
    import torch

    return torch.tensor(values, dtype=torch.float32, device="cpu")
