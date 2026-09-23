"""Go2 Stair reward: reference flat-derived terms plus an explicit foot-clearance gap penalty for steep OOD stairs."""

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


def foot_clearance_gap_penalty(inputs):
    foot_z = inputs.support_pos_relative_to_terrain_origin[:, :, 2]
    base_z = inputs.base_pos_relative_to_terrain_origin[:, 2].unsqueeze(1)
    gap = foot_z - base_z
    low_gap = gap.clamp(min=0.0, max=0.2)
    return low_gap.square().sum(dim=1)


def build_training_components():
    return TrainingComponents(
        name='go2_stair_official_flat_derived_foot_clearance_gap_v1',
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
                name='foot_clearance_gap_penalty',
                weight=-5.0,
                fn=foot_clearance_gap_penalty,
                clip_min=0.0,
                clip_max=0.2,
                description='Penalize feet hanging just below the base while ascending steep OOD stairs.',
            ),
        ],
        rationale=(
            'Anchor is the verified current-best reference go2_stair_official_flat_derived_v1 '
            '(objective 0.9344); all eight reference terms and weights are preserved exactly. '
            'The single mechanism change is the addition of RewardTerm foot_clearance_gap_penalty, '
            'which penalizes the squared vertical gap between each foot terrain-relative height and '
            'the base terrain-relative height only in the low-gap band [0, 0.2] m, so feet that are '
            'already below or far below the base are not penalized and the bounded weighted penalty is '
            'at most -0.8 per step. Public dev evidence: the failing cells are tall-riser and '
            'short-tread extrapolations (height_ood success 0.852, joint_ood 0.407, tread_ood 0.667) '
            'and forbidden_contact is the dominant failure (12.5 percent overall, 51.9 percent in '
            'joint_ood), while the reference has no foot-height shaping. This same dominant-failure '
            'diagnosis motivated two rejected candidates: stair_forbidden_contact_force_penalty_v1 '
            '(dense force L2 penalty; forbidden contact dropped but tall-step successes collapsed) and '
            'stair_climb_lift_v1 (positive vertical base-velocity bonus; objective -0.0207). The '
            'proposed term differs at the code level from both: it consumes the foot vertical-gap '
            'field that neither evaluated candidate used, applies a bounded squared penalty instead of '
            'a linear force norm or velocity bonus, and it is inactive on flat ground because the gap '
            'is negative there. The hypothesis is that the policy fails on tall steps because it '
            'slices its swing feet through the riser plane at small vertical clearance; penalizing '
            'the low-gap regime during ascent teaches higher foot trajectories without rewarding '
            'jumping or penalizing contact, which reduces pre-arrival forbidden-contact terminations '
            'on extrapolated stairs while preserving in-distribution contact-free behavior.'
        ),
    )
