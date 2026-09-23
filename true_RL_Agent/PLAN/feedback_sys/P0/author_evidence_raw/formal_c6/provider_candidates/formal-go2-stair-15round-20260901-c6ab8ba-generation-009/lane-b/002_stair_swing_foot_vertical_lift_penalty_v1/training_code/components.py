'''Go2 Stair reward: reference flat-derived terms plus a vertical-lift trip-hazard penalty for OOD riser/tread extrapolation.'''

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


def swing_foot_vertical_lift_penalty(inputs):
    foot_z = inputs.support_pos_relative_to_terrain_origin[:, :, 2]
    base_z = inputs.base_pos_relative_to_terrain_origin[:, 2].unsqueeze(1)
    rel_z = foot_z - base_z
    low_band = ((rel_z > -0.15) & (rel_z < 0.2)).float()
    swing = (~inputs.support_contact).float()
    forward_vel = inputs.support_lin_vel[:, :, 0].clamp(min=0.0)
    vertical_vel = inputs.support_lin_vel[:, :, 2]
    insufficient_lift = (vertical_vel < 0.3).float()
    elevated = (inputs.base_height > 0.1).float()
    before_goal = (inputs.goal_progress_fraction < 1.0).float()
    penalty = (
        low_band * swing * forward_vel * insufficient_lift
    ).sum(dim=1) * elevated * before_goal
    return penalty


def build_training_components():
    return TrainingComponents(
        name='go2_stair_official_flat_derived_vertical_lift_v1',
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
                name='swing_foot_vertical_lift_penalty',
                weight=-0.2,
                fn=swing_foot_vertical_lift_penalty,
                clip_min=0.0,
                clip_max=2.0,
                description='Penalize forward sweep of swing feet near the base plane while the foot is not lifting fast enough, on elevated stair terrain before the goal.',
            ),
        ],
        rationale=(
            'Anchor is the verified current-best reference go2_stair_official_flat_derived_level_sweep_v1 '
            '(objective 0.9483); all eight reference terms and weights are preserved exactly. '
            'The single mechanism change is a new RewardTerm swing_foot_vertical_lift_penalty that '
            'evolves the closest prior accepted candidate stair_swing_foot_level_sweep_penalty_v1 '
            '(objective 0.9483, F_scalar 0.767, residual dominant forbidden_contact 9.7 percent, '
            'joint_ood forbidden_contact 66.7 percent) at the code level: the prior term gated the '
            'penalty by vertical_vel <= 0.2 m/s, while this term uses vertical_vel < 0.3 m/s, so it '
            'penalizes a wider range of insufficient vertical lift while still excluding feet that are '
            'actively lifting fast enough. The raw term is clipped to [0, 2.0] with weight -0.2, so its '
            'weighted contribution is at most -0.4 per step; activation gates (swing contact, '
            'base_height > 0.1, and goal_progress_fraction < 1.0) keep it silent on the flat start, '
            'flat behavior, and after arrival. Hypothesis: the residual OOD forbidden-contact failures '
            'on simultaneous riser/tread extrapolation come from swing feet slicing forward with '
            'insufficient vertical lift; raising the lift threshold from 0.2 to 0.3 m/s teaches more '
            'aggressive vertical-first foot clearance, protecting height_ood, tread_ood, and the '
            'C0/C1 in-distribution guards.'
        ),
    )
