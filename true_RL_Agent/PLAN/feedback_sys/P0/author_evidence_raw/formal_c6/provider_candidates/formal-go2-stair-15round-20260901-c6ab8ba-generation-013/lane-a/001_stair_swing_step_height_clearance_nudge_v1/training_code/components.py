'''Go2 Stair reward: reference flat-derived locomotion terms plus a forward-command-active swing-lift step bonus that targets tall-riser/short-tread OOD forbidden-contact failures by intensifying the functional effect of the positive swing-rise mechanism without its late-swing over-lift side effect.'''

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


def forward_step_clearance_bonus(inputs):
    foot_z = inputs.support_pos_relative_to_terrain_origin[:, :, 2]
    base_z = inputs.base_pos_relative_to_terrain_origin[:, 2].unsqueeze(1)
    rel_z = foot_z - base_z
    swing = (~inputs.support_contact).float()
    forward_vel = inputs.support_lin_vel[:, :, 0].clamp(min=0.0)
    stance_present = inputs.support_contact.any(dim=1).float()
    moving_forward_cmd = (inputs.commanded_base_vel[:, 0] > 0.15).float()
    elevated = (inputs.base_height > 0.1).float()
    before_goal = (inputs.goal_progress_fraction < 1.0).float()
    gate = stance_present * moving_forward_cmd * elevated * before_goal
    lifting = (inputs.support_lin_vel[:, :, 2] > 0.15).float()
    low_lift_headroom = (rel_z < 0.22).float()
    bonus = (
        forward_vel * swing * lifting * low_lift_headroom
    ).sum(dim=1) * gate
    return bonus


def build_training_components():
    return TrainingComponents(
        name='go2_stair_official_flat_derived_forward_step_clearance_v1',
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
                name='forward_step_clearance_bonus',
                weight=0.35,
                fn=forward_step_clearance_bonus,
                clip_min=0.0,
                clip_max=2.0,
                description='Reward airborne feet that are forward-swinging and actively lifting below the base plane before the goal on elevated stair terrain, while a stance foot is planted and a forward command is active.',
            ),
        ],
        rationale=(
            'Anchor is the accepted current-best go2_stair_official_flat_derived_level_sweep_v1 '
            '(objective 0.9483; j_ood 0.9224; c0_id=1.0, c1_id=1.0). All eight reference terms '
            'and weights are preserved exactly. The accepted head still loses joint_ood '
            '(score 0.412, success 0.222, forbidden_contact 0.667, slip_detected 0.667), so the '
            'simultaneous tall-riser/short-tread extrapolation mostly fails by riser contact while '
            'forward motion is active. The closest prior positive mechanism is the latest '
            'nonaccepted swing_rise_clearance_bonus_v1 (objective 0.9405, j_ood 0.9108, same guards); '
            'its sole runtime active candidate term shifted the policy: forbidden_contact fell to 0.060 '
            'from 0.097 while its joint_ood forbidden-contact rate improved to 0.407, yet objective '
            'still dropped because height_ood success fell to 0.593 with 0.333 timeout, a sign that '
            'rewarding vertical lift throughout the whole swing corridor (-0.45 to 0.1 m relative '
            'height) produced over-lift or late-swing deceleration. The exact code-level difference is '
            'that this term replaces swing_rise_clearance_bonus with forward_step_clearance_bonus: '
            'the prior term rewarded vertical velocity alone inside (-0.45, 0.1) m; this term instead '
            'scores swing forward speed multiplied by a binary active-lift gate (vertical velocity > '
            '0.15 m/s) and an upper headroom gate (rel_z < 0.22 m), with no lower rel_z cutoff, so a '
            'foot must make simultaneously forward and upward progress from any height while it is '
            'still below base height. This keeps the step-rise timing gradient of the rejected '
            'level-sweep penalty (which cut riser-contact forbidden_contact and defended height_ood at '
            '0.916/0.889) without punishing the forward-swing component needed after the foot clears '
            'the next riser, and it removes the prior reward above base height that encouraged '
            'over-lift. The raw term is clipped to [0, 2.0] with weight 0.35, so its weighted '
            'contribution is at most +0.7 per step; the stance/command/elevated/pre-goal gates keep '
            'it silent during flat start, zero-command flight, and after arrival. Hypothesis: '
            'conditioning lift reward on simultaneous forward swing below the base plane teaches the '
            'policy to rise while advancing over tall risers without sacrificing swing speed or '
            'height_ood cadence, so joint_ood forbidden-contact rises, guard scores hold, and the '
            'objective improves over the accepted head without the swing_rise_clearance_bonus over-lift '
            'profile that lowered the latest evaluation.'
        ),
    )
