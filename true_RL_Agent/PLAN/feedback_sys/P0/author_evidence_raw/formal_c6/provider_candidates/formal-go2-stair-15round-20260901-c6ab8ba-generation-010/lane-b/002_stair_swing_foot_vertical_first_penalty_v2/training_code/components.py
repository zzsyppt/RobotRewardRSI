'''Go2 Stair reward: reference flat-derived terms plus a vertical-first swing-foot trip-hazard penalty for riser/tread OOD extrapolation.'''

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


def swing_foot_vertical_first_penalty(inputs):
    foot_z = inputs.support_pos_relative_to_terrain_origin[:, :, 2]
    base_z = inputs.base_pos_relative_to_terrain_origin[:, 2].unsqueeze(1)
    rel_z = foot_z - base_z
    low_band = ((rel_z > -0.15) & (rel_z < 0.2)).float()
    swing = (~inputs.support_contact).float()
    vertical_vel = inputs.support_lin_vel[:, :, 2]
    forward_vel = inputs.support_lin_vel[:, :, 0].clamp(min=0.0)
    rising = (vertical_vel > 0.0).float()
    insufficient_rise = (vertical_vel < 0.35).float()
    air_time = inputs.support_air_time
    early_swing = ((air_time > 0.02) & (air_time < 0.35)).float()
    still_low = (rel_z <= 0.05).float()
    elevated = (inputs.base_height > 0.1).float()
    before_goal = (inputs.goal_progress_fraction < 1.0).float()
    penalty = (
        low_band * swing * forward_vel * rising * insufficient_rise
        * early_swing * still_low
    ).sum(dim=1) * elevated * before_goal
    return penalty


def build_training_components():
    return TrainingComponents(
        name='go2_stair_official_flat_derived_vertical_first_v2',
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
                name='swing_foot_vertical_first_penalty',
                weight=-0.2,
                fn=swing_foot_vertical_first_penalty,
                clip_min=0.0,
                clip_max=2.0,
                description=(
                    'Penalize horizontal advance of a swing foot only while it is '
                    'airborne in an early-swing window, still at or below the '
                    'departure altitude, and rising with vertical velocity below '
                    '0.35 m/s. It never penalizes late-swing motion or descending '
                    'placement, so productive foot placement is preserved.'
                ),
            ),
        ],
        rationale=(
            'Anchor is the accepted current-best source '
            'go2_stair_official_flat_derived_level_sweep_v1 (objective 0.9483, '
            'joint_ood forbidden_contact 66.7 percent, joint_ood success 0.222). '
            'All eight reference terms and weights are preserved exactly. '
            'The single mechanism change replaces the anchor term '
            'swing_foot_level_sweep_penalty, which penalizes any swing-foot '
            'forward sweep whose vertical velocity is below 0.2, including the '
            'productive later-swing and descending-placement periods, with '
            'swing_foot_vertical_first_penalty v2. Closest prior candidate: '
            'swing_foot_vertical_first_penalty v1 (attempt/latest_failed), which '
            'failed admission smoke because a boolean gate was combined through '
            'bitwise and with already-float tensors before ever running under '
            'evaluation. The exact code-level differences are: (1) v2 composes '
            'every comparison into boolean masks first and converts to float only '
            'for multiplication, so the term executes cleanly; (2) the scrape-prone '
            'slice no longer depends on lagged support_contact window states and '
            'instead directly combines current swing state, a low-band foot altitude '
            'at or below the departure plane, positive but insufficient vertical '
            'velocity (0 to 0.35 m/s), and an explicit early-swing air-time window '
            'from 0.02 to 0.35 s. Hypothesis: OOD forbidden-contact failures are '
            'dominated by feet that advance horizontally while still creeping '
            'upward against the next riser just after takeoff; penalizing only that '
            'early, low, rising, forward-moving slice teaches the policy to lift '
            'first while leaving late swing and descending foot placement '
            'productive, improving height_ood and joint_ood without disturbing '
            'C0/C1 in-distribution locomotion.'
        ),
    )
