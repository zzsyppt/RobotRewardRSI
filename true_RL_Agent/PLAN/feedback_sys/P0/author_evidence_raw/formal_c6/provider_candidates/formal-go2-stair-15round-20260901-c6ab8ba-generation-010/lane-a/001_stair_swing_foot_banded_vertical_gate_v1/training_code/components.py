'''Go2 Stair reward: reference flat-derived terms plus a two-sided vertical-lift gate penalty for OOD riser/tread extrapolation.'''

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


def swing_foot_banded_vertical_gate_penalty(inputs):
    foot_z = inputs.support_pos_relative_to_terrain_origin[:, :, 2]
    base_z = inputs.base_pos_relative_to_terrain_origin[:, 2].unsqueeze(1)
    rel_z = foot_z - base_z
    low_band = ((rel_z > -0.15) & (rel_z < 0.2)).float()
    swing = (~inputs.support_contact).float()
    forward_vel = inputs.support_lin_vel[:, :, 0].clamp(min=0.0)
    vertical_vel = inputs.support_lin_vel[:, :, 2]
    above_low = (vertical_vel > 0.2).float()
    below_high = (vertical_vel < 0.35).float()
    partial_lift = (above_low * below_high).clamp(min=0.0, max=1.0)
    elevated = (inputs.base_height > 0.1).float()
    before_goal = (inputs.goal_progress_fraction < 1.0).float()
    penalty = (
        low_band * swing * forward_vel * partial_lift
    ).sum(dim=1) * elevated * before_goal
    return penalty


def build_training_components():
    return TrainingComponents(
        name='go2_stair_official_flat_derived_banded_vertical_gate_v1',
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
                name='swing_foot_banded_vertical_gate_penalty',
                weight=-0.2,
                fn=swing_foot_banded_vertical_gate_penalty,
                clip_min=0.0,
                clip_max=2.0,
                description='Penalize forward sweep of swing feet near the base plane only when vertical lift velocity is between 0.2 and 0.35 m/s, on elevated stair terrain before the goal.',
            ),
        ],
        rationale=(
            'Anchor is the verified current-best reference go2_stair_official_flat_derived_level_sweep_v1 '
            '(objective 0.9483); all eight reference terms and weights are preserved exactly. '
            'The single mechanism change is a new RewardTerm swing_foot_banded_vertical_gate_penalty that '
            'evolves the closest prior candidate stair_swing_foot_vertical_lift_penalty_v1 '
            '(objective 0.9179, F_scalar 0.630, joint_ood 0.274, forbidden_contact 16.2 percent, '
            'slip_detected 39.8 percent, dominant failure forbidden_contact) at the code level: the '
            'rejected term penalized every low forward sweep with vertical_vel < 0.3 m/s, which also '
            'punished feet decelerating at the top of their lift and produced broad instability, while '
            'the accepted term used vertical_vel <= 0.2 m/s and missed the 0.2-0.35 intermediate-lift '
            'regime implicated in joint-ood collisions. This term uses a two-sided gate (vertical_vel > '
            '0.2 and vertical_vel < 0.35), penalizing only the intermediate-lift band bounded away from '
            'both decisive lifting and settled/descending phases. The raw term is clipped to [0, 2.0] '
            'with weight -0.2, so its weighted contribution is at most -0.4 per step; activation gates '
            '(swing contact, base_height > 0.1, and goal_progress_fraction < 1.0) keep it silent on '
            'the flat start, flat behavior, and after arrival. Hypothesis: residual joint-ood '
            'forbidden-contact failures come from swing feet that begin lifting but then sweep forward '
            'without reaching a decisively high vertical velocity; restricting the penalty to that '
            'intermediate 0.2-0.35 m/s regime removes the earlier threshold widening that destabilized '
            'the base, so the policy learns clean vertical-first clearance that protects height_ood, '
            'tread_ood, and the C0/C1 in-distribution guards.'
        ),
    )
