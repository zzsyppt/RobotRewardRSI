'''Go2 Stair reward: reference flat-derived terms plus a level-sweep trip-hazard penalty and a complementary positive swing-lift bonus for OOD riser/tread extrapolation.'''

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


def swing_foot_level_sweep_penalty(inputs):
    foot_z = inputs.support_pos_relative_to_terrain_origin[:, :, 2]
    base_z = inputs.base_pos_relative_to_terrain_origin[:, 2].unsqueeze(1)
    rel_z = foot_z - base_z
    low_band = ((rel_z > -0.15) & (rel_z < 0.2)).float()
    swing = (~inputs.support_contact).float()
    forward_vel = inputs.support_lin_vel[:, :, 0].clamp(min=0.0)
    vertical_vel = inputs.support_lin_vel[:, :, 2]
    level_or_down = (vertical_vel <= 0.2).float()
    elevated = (inputs.base_height > 0.1).float()
    before_goal = (inputs.goal_progress_fraction < 1.0).float()
    penalty = (
        low_band * swing * forward_vel * level_or_down
    ).sum(dim=1) * elevated * before_goal
    return penalty


def swing_foot_low_lift_bonus(inputs):
    foot_z = inputs.support_pos_relative_to_terrain_origin[:, :, 2]
    base_z = inputs.base_pos_relative_to_terrain_origin[:, 2].unsqueeze(1)
    rel_z = foot_z - base_z
    low_band = ((rel_z > -0.15) & (rel_z < 0.2)).float()
    swing = (~inputs.support_contact).float()
    lift_vel = inputs.support_lin_vel[:, :, 2].clamp(min=0.0, max=2.0)
    moving = inputs.commanded_base_vel[:, :2].norm(dim=1) > 0.1
    elevated = (inputs.base_height > 0.1).float()
    before_goal = (inputs.goal_progress_fraction < 1.0).float()
    bonus = (low_band * swing * lift_vel).sum(dim=1) * elevated * before_goal
    return bonus * moving.float()


def build_training_components():
    return TrainingComponents(
        name='go2_stair_official_flat_derived_level_sweep_lift_bonus_v1',
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
                name='swing_foot_level_sweep_penalty',
                weight=-0.2,
                fn=swing_foot_level_sweep_penalty,
                clip_min=0.0,
                clip_max=2.0,
                description='Penalize forward sweep of swing feet near the base plane while the foot is not actively lifting, on elevated stair terrain before the goal.',
            ),
            RewardTerm(
                name='swing_foot_low_lift_bonus',
                weight=0.1,
                fn=swing_foot_low_lift_bonus,
                clip_min=0.0,
                clip_max=2.0,
                description='Reward positive vertical lift velocity of swing feet inside the same low-clearance band on elevated stair terrain before the goal.',
            ),
        ],
        rationale=('Anchor is the accepted current-best go2_stair_official_flat_derived_level_sweep_v1 '
                   '(objective 0.9483, j_ood 0.9224); all eight reference terms and weights, and the '
                   'accepted swing_foot_level_sweep_penalty term and its weight, are preserved exactly. '
                   'The single mechanism change is adding a second custom RewardTerm '
                   'swing_foot_low_lift_bonus. The closest evaluated mechanism is the accepted current-best '
                   'level-sweep penalty itself; the exact code-level difference is that the new term '
                   'rewards the positive vertical component of swing-foot velocity inside the same '
                   'low-clearance band instead of only punishing level/downward forward sweeps. It is not '
                   'a variant of the rejected negative swing_foot_vertical_lift_penalty_v1 (objective '
                   '0.9179) because that family added more negative pressure on lift phases and further '
                   'depressed progress, and it is not a reuse of the rejected positive '
                   'blocked_swing_forward_nudge (objective 0.9263, joint_ood slip rate 1.0) because the '
                   'new term has no forward-push component at all: it rewards only upward foot motion '
                   'while the foot is still at base-clearance height. Hypothesis: the accepted penalty '
                   'taught the policy not to slice swing feet level/forward near the base plane, but '
                   'joint_ood still fails with 66.7 percent forbidden contact and 66.7 percent slip '
                   'because it gives no positive learning signal for the required vertical-first lift; '
                   'the failed forward nudge pushed feet into the riser and caused slip/orientation '
                   'instability. A bounded positive bonus on low-band swing vertical velocity directly '
                   'shapes the missing action, complements the unmodified sweep penalty by shortening '
                   'the non-lifting part of the swing, and is silent on the flat start, after goal '
                   'arrival, and when the commanded horizontal speed is negligible. The raw term is '
                   'clipped to [0, 2.0] with weight 0.1, so the weighted contribution is at most +0.2 '
                   'per step and cannot overpower the flat-derived locomotion terms.'),
    )
