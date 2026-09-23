import torch

from true_rl_agent.candidates.components import RewardInputs, RewardTerm, TrainingComponents


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
    foot_z = inputs.support_pos_relative_to_terrain_origin[:, :, 2]
    base_z = inputs.base_pos_relative_to_terrain_origin[:, 2].unsqueeze(1)
    gap = foot_z - base_z
    low_gap = gap.clamp(min=0.0, max=0.2)
    # Active only while a foot is airborne or in its first-contact swing phase.
    swing = (inputs.support_first_contact.float() + (1.0 - inputs.support_contact.float())).clamp(max=1.0)
    commanded_forward = (inputs.commanded_base_vel[:, 0] > 0.1).float().unsqueeze(1)
    return (low_gap.square() * swing * commanded_forward).sum(dim=1)


def build_training_components():
    return TrainingComponents(
        name='go2_stair_official_flat_derived_swing_foot_clearance_penalty_v2',
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
                weight=-5.0,
                fn=swing_foot_clearance_penalty,
                clip_min=0.0,
                clip_max=0.2,
                description='Penalize swing feet near the riser plane during ascent on extrapolated stairs.',
            ),
        ],
        rationale=(
            'Anchor is the verified current-best reference go2_stair_official_flat_derived_v1 '
            '(objective 0.9344); all eight reference terms and weights are preserved exactly. '
            'The single mechanism change is the addition of RewardTerm swing_foot_clearance_penalty, '
            'which is a modified variant of the previously evaluated but never accepted '
            'stair_foot_clearance_gap_penalty_v1 (best observed nonaccepted objective 0.93477). '
            'That candidate used the bare vertical gap between foot height and base height for all '
            'feet. This variant instead gates the penalty by swing-flight state: it uses the '
            'per-foot first-contact flag and historical contact forces to target only feet that are '
            'airborne or at the instant of touchdown, while leaving stance feet unpenalized. It also '
            'adds a commanded-forward gate so that the shaping is active only during ascent. '
            'The hypothesis is that the reference fails mainly in scalar OOD (joint_ood success '
            '0.407, tread_ood 0.667, height_ood 0.852) with forbidden contact as the dominant '
            'failure (12.5 percent overall, 51.9 percent in joint_ood), that the prior gap penalty '
            'improved tread_ood (0.667 to 0.963) and raised mean progress (0.9130 to 0.9141) but '
            'did not reduce joint_ood failure, and that this is because penalizing the foot/base '
            'gap for stance feet confuses load-bearing contact. Applying the same low-gap squared '
            'penalty only while a foot is airborne or in its first-contact swing phase teaches '
            'higher swing trajectories on tall/steep extrapolated stairs without penalizing '
            'stance support.'
        ),
    )
