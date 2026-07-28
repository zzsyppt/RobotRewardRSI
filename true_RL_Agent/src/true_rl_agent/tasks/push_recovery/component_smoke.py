"""Push recovery task-owned smoke fixture."""

from __future__ import annotations

from dataclasses import dataclass

from true_rl_agent.tasks.component_smoke import ComponentSmokeFixture, SmokeTensor
from true_rl_agent.tasks.push_recovery.config import PushRecoveryTaskConfig
from true_rl_agent.tasks.push_recovery.reward_inputs import PushRecoveryRewardInputs


@dataclass(frozen=True)
class PushRecoveryComponentSmokeAdapter:
    """Build lightweight RewardInputs fixtures for push recovery candidate smoke checks."""

    config: PushRecoveryTaskConfig
    task_id: str = "unitree_push_recovery"

    def build_component_smoke_fixture(self) -> ComponentSmokeFixture:
        num_envs = 4
        num_joints = 6
        num_actions = 6
        return ComponentSmokeFixture(
            task_id=self.task_id,
            reward_inputs=PushRecoveryRewardInputs(
                base_height=SmokeTensor.vector([0.34, 0.39, 0.42, 0.36]),
                roll=SmokeTensor.vector([0.02, -0.04, 0.08, -0.03]),
                pitch=SmokeTensor.vector([0.03, 0.06, -0.05, 0.01]),
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
                push_active=SmokeTensor.vector([0.0, 1.0, 1.0, 0.0]),
                push_vec_public=_matrix(num_envs, 2, scale=0.20),
                progress_fraction=SmokeTensor.vector([0.05, 0.35, 0.65, 0.95]),
            ),
            expected_reward_output_shape=(num_envs,),
            public_summary={
                "fixture_owner": "task_owned_push_recovery",
                "num_envs": num_envs,
                "num_joints": num_joints,
                "num_actions": num_actions,
                "purpose": "authoring_time_reward_callable_smoke_only",
            },
        )


def _matrix(rows: int, cols: int, *, scale: float) -> SmokeTensor:
    return SmokeTensor.matrix(
        [
            [scale * float((row + 1) * (col + 1)) for col in range(cols)]
            for row in range(rows)
        ]
    )
