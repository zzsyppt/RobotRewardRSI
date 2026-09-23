"""Go2 Stair reward: reference flat-derived terms plus a bounded swing-foot clearance penalty for steep OOD stairs."""

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


def swing_foot_clearance_penalty(inputs):
    base_z = inputs.base_pos_relative_to_terrain_origin[:, 2].unsqueeze(1)
    foot_z = inputs.support_pos_relative_to_terrain_origin[:, :, 2]
    low_clearance = (base_z - foot_z).clamp(min=0.0, max=0.25)
    below = (foot_z - base_z).clamp(min=0.0, max=0.35)
    return low_clearance.square().sum(dim=1) + below.square().sum(dim=1)


def build_training_components():
    return TrainingComponents(
        name='go2_stair_official_flat_derived_swing_foot_clearance_v1',
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
                name='swing_foot_clearance_penalty',
                weight=-4.0,
                fn=swing_foot_clearance_penalty,
                clip_min=0.0,
                clip_max=0.4,
                description='Penalize low swing-foot clearance above the base and feet far below the base during stair ascent.',
            ),
        ],
        rationale=(
            'Anchor is the verified current-best reference go2_stair_official_flat_derived_v1 '
            '(objective 0.9344). All eight reference terms and weights are preserved exactly. '
            'The single mechanism change is the addition of RewardTerm swing_foot_clearance_penalty, '
            'which differs at the code level from the closest prior candidate '
            'stair_foot_clearance_gap_penalty_v1: the rejected term penalized the squared vertical '
            'gap between foot height and base height only in the low-gap band [0, 0.2] m (i.e. feet '
            'hanging just below the base) and left the rising-foot regime unshaped; the new term '
            'additionally penalizes low absolute clearance above the base in [0, 0.25] m and feet '
            'hanging below the base in [0, 0.35] m, with the two bounded squared penalties summed '
            'per foot. Public dev evidence: the rejected gap penalty improved tread_ood success from '
            '0.667 to 0.963 and height_ood success from 0.852 to 0.667 while objective rose only '
            '0.0003 because joint_ood success fell to 0.296 with forbidden_contact at 0.630; the '
            'reference itself fails joint_ood success 0.407 with forbidden_contact 0.519. The '
            'hypothesis is that tall-riser/short-tread extrapolation, not the flat in-distribution '
            'regime, is the primary risk surface, and that a bounded penalty on both low foot '
            'clearance above the base and feet dangling below it teaches the swing foot to reach '
            'the next tread with adequate vertical margin instead of slicing through the riser '
            'plane. This is intended to reduce pre-arrival forbidden-contact terminations on '
            'extrapolated stairs while preserving the reference behavior on in-distribution '
            'terrain.'
        ),
    )
