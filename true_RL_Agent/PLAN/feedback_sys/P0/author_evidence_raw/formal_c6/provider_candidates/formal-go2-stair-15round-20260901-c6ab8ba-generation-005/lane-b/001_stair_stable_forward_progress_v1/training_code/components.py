# Go2 Stair reward: reference flat-derived terms plus one gated stable forward-progress bonus.

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


def stair_stable_forward_progress_bonus(inputs):
    forward_speed = inputs.base_lin_vel_yaw[:, 0].clamp(min=0.0, max=1.5)
    upright = (inputs.projected_gravity[:, 2] > 0.85).float()
    supported = (inputs.support_contact.float().sum(dim=1) >= 1).float()
    commanded_forward = (inputs.commanded_base_vel[:, 0] > 0.1).float()
    not_forbidden = 1.0 - inputs.forbidden_contact.float()
    return forward_speed * upright * supported * commanded_forward * not_forbidden


def build_training_components():
    return TrainingComponents(
        name='go2_stair_official_flat_derived_stable_forward_progress_v1',
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
                name='stair_stable_forward_progress_bonus',
                weight=0.15,
                fn=stair_stable_forward_progress_bonus,
                clip_min=0.0,
                clip_max=1.0,
                description='Reward commanded forward travel in the yaw frame only while the base is upright, at least one support is in contact, and no forbidden contact has occurred.',
            ),
        ],
        rationale=(
            'Anchor is the verified current-best reference go2_stair_official_flat_derived_v1 (objective 0.9344). All eight reference terms and weights are preserved exactly. The single mechanism change is the addition of RewardTerm stair_stable_forward_progress_bonus, which rewards bounded yaw-frame forward speed only when all four safety gates hold: projected gravity z > 0.85, at least one support-contact sample is active, a forward command is present, and the episode is not already in forbidden contact. The term is clipped to [0.0, 1.0] with weight 0.15, so its maximum per-step contribution is 0.15. Public dev evidence: the reference fails almost entirely in F_scalar (0.705), with joint_ood success 0.407 and goal_not_reached 0.593, tread_ood success 0.667 and goal_not_reached 0.333, and height_ood success 0.852; direct forbidden-contact penalties (stair_forbidden_contact_force_penalty_v1 and stair_forbidden_contact_terminal_penalty_v1) and foot-gap/clearance shaping (stair_foot_clearance_gap_penalty_v1 and stair_step_desc_clearance_v1) were already tried and did not improve the scalar-OOD compromise. This term differs at the code level from the closest prior candidate stair_climb_lift_v1 (which gave an unbounded positive vertical base-velocity bonus and scored -0.0207) by consuming the horizontal yaw-frame forward velocity field instead of vertical base velocity and by applying upright, support, command, and forbidden-contact gates with a bounded clip; it also differs from the rejected foot-clearance mechanisms because it uses no foot or terrain positional inputs and no contact-force penalty. The hypothesis is that the scalar-OOD failures are not caused by insufficient foot clearance alone but by the policy losing forward drive or attempting progress from unstable, unsupported, or tilted states on extrapolated riser and tread combinations; rewarding only stable forward travel shapes the policy to keep ascending through tall, short-tread steps while preserving the reference in-distribution behavior.'
        ),
    )
