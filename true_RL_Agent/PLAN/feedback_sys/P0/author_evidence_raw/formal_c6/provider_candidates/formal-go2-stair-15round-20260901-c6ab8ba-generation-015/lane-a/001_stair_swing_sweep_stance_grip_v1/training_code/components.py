'''Go2 Stair reward: reference flat-derived terms plus a combined swing-sweep and planted-foot grip penalty for OOD riser/tread extrapolation.'''

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


def swing_sweep_stance_grip_penalty(inputs):
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
    sweep_penalty = (
        low_band * swing * forward_vel * level_or_down
    ).sum(dim=1) * elevated * before_goal

    stance = inputs.support_contact.float()
    foot_horiz_speed = inputs.support_lin_vel[:, :, :2].norm(dim=2)
    stance_slip = (stance * foot_horiz_speed).sum(dim=1) * elevated * before_goal
    return sweep_penalty + 0.2 * stance_slip


def build_training_components():
    return TrainingComponents(
        name='go2_stair_official_flat_derived_sweep_stance_grip_v1',
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
                name='swing_sweep_stance_grip_penalty',
                weight=-0.25,
                fn=swing_sweep_stance_grip_penalty,
                clip_min=0.0,
                clip_max=3.0,
                description='Preserve the accepted level-sweep trip-hazard penalty and add a planted-foot horizontal-slip penalty on elevated stair terrain before the goal.',
            ),
        ],
        rationale=(
            'Anchor is the verified current-best stair_swing_foot_level_sweep_penalty_v1 '
            '(objective 0.9483, J_ID 1.0, J_OOD 0.9224, C0-ID 1.0, C1-ID 1.0); all eight '
            'reference terms and the existing swing-level-sweep sub-mechanism are preserved exactly. '
            'The single mechanism change evolves the current wing-foot Level sweep penalty into a '
            'combined swing-sweep plus planted-foot grip penalty at the code level: the old term only '
            'penalized non-lifting forward sweeps while a foot was airborne, and public-dev diagnostics '
            'show the residual scalar-OOD failures concentrate in joint_ood where forbidden_contact is '
            '0.667 and slip_detected is 0.667. The new term adds a stance branch that penalizes '
            'horizontal sliding of currently loaded feet (support_contact true) on elevated stair '
            'terrain before the goal, so the policy is shaped to plant and hold each foot on the '
            'available tread instead of driving the base while a loaded foot is still slipping across '
            'an extrapolated riser/tread edge. The stance branch is scaled at 0.2 inside the raw term '
            'and the whole raw term is clipped to [0, 3.0] with weight -0.25, so the maximum weighted '
            'contribution is -0.75 per step and the activation gates (base_height > 0.1 and '
            'goal_progress_fraction < 1.0) keep the term silent on the flat start and after top '
            'arrival. Hypothesis: simultaneous riser/tread extrapolation fails because loaded feet '
            'slide off narrow or shallow contacts, which then produces forbidden contact on the next '
            'step; penalizing horizontal slip of stance feet while retaining the accepted swing-lift '
            'shaping should improve joint_ood and height_ood without regressing the C0/C1 '
            'in-distribution guards.'
        ),
    )
