'''Go2 Stair reward: reference flat-derived terms plus one dense forbidden-contact force penalty.'''

from true_rl_agent.candidates.components import RewardTerm, TrainingComponents


def track_lin_vel_xy_exp(inputs):
    error = (
        inputs.commanded_base_vel[:, :2] - inputs.base_lin_vel[:, :2]
    ).square().sum(dim=1)
    return (-error / 0.25).exp()


def track_ang_vel_z_exp(inputs):
    error = (
        inputs.commanded_base_vel[:, 2] - inputs.base_ang_vel[:, 2]
    ).square()
    return (-error / 0.25).exp()


def ang_vel_xy_l2(inputs):
    return inputs.base_ang_vel[:, :2].square().sum(dim=1)


def flat_orientation_l2(inputs):
    return inputs.projected_gravity[:, :2].square().sum(dim=1)


def dof_torques_l2(inputs):
    return inputs.joint_torque.square().sum(dim=1)


def dof_acc_l2(inputs):
    return inputs.joint_acc.square().sum(dim=1)


def action_rate_l2(inputs):
    return inputs.action_delta.square().sum(dim=1)


def feet_air_time(inputs):
    reward = (
        (inputs.support_last_air_time - 0.5)
        * inputs.support_first_contact.float()
    ).sum(dim=1)
    moving = inputs.commanded_base_vel[:, :2].norm(dim=1) > 0.1
    return reward * moving.float()


def forbidden_contact_force_l2_penalty(inputs):
    return inputs.forbidden_contact_force.square().sum(dim=(-1, -2)).sqrt()


def build_training_components():
    return TrainingComponents(
        name='go2_stair_forbidden_contact_force_penalty_v1',
        reward_terms=[
            RewardTerm(name='track_lin_vel_xy_exp', weight=3.5, fn=track_lin_vel_xy_exp),
            RewardTerm(name='track_ang_vel_z_exp', weight=0.75, fn=track_ang_vel_z_exp),
            RewardTerm(name='ang_vel_xy_l2', weight=-0.05, fn=ang_vel_xy_l2),
            RewardTerm(name='flat_orientation_l2', weight=-2.5, fn=flat_orientation_l2),
            RewardTerm(name='dof_torques_l2', weight=-0.0002, fn=dof_torques_l2),
            RewardTerm(name='dof_acc_l2', weight=-2.5e-7, fn=dof_acc_l2),
            RewardTerm(name='action_rate_l2', weight=-0.01, fn=action_rate_l2),
            RewardTerm(name='feet_air_time', weight=0.25, fn=feet_air_time),
            RewardTerm(
                name='forbidden_contact_force_l2_penalty',
                weight=-0.002,
                fn=forbidden_contact_force_l2_penalty,
                clip_min=0.0,
                clip_max=300.0,
                description='Dense L2 penalty on resolved forbidden-contact force to reduce riser collisions on steep OOD stairs.',
            ),
        ],
        rationale=(
            'Reference anchor go2_stair_official_flat_derived_v1 scores 1.0 on '
            'c0_id/c1_id/c2 but fails on extrapolated steep stairs: forbidden_contact '
            'is the dominant primary failure (joint_ood 0.519, tread_ood 0.333, '
            'height_ood 0.148) and joint_ood cell 3 reaches 0.0 success. This '
            'candidate changes exactly one mechanism: it adds a new dense penalty '
            'term on the L2 norm of the resolved forbidden-contact force '
            '(clip [0,300] N, weight -0.002, bounded weighted contribution -0.6). '
            'All reference terms and weights are preserved verbatim. The hypothesis '
            'is that per-step negative gradient on rising forbidden-contact force '
            'teaches the policy to avoid riser-edge collisions on steep OOD stairs, '
            'reducing pre-arrival terminations without altering in-distribution '
            'contact-free behavior.'
        ),
    )
