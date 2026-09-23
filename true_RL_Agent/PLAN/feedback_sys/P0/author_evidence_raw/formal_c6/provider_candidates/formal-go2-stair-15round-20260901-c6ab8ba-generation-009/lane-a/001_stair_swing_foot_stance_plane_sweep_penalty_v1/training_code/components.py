'''Go2 Stair reward: reference flat-derived terms plus an adaptive stance-plane swing-foot trip-hazard penalty for OOD riser/tread extrapolation.'''

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


def swing_foot_stance_plane_sweep_penalty(inputs):
    foot_z = inputs.support_pos_relative_to_terrain_origin[:, :, 2]
    grounded = inputs.support_contact.float()
    stance_z = foot_z * grounded
    highest_support_z = (
        stance_z - 1000.0 * (1.0 - grounded)
    ).max(dim=1).values
    lowest_support_z = (
        stance_z + 1000.0 * (1.0 - grounded)
    ).min(dim=1).values
    support_spread = highest_support_z - lowest_support_z
    stair_context = (
        (support_spread > 0.05).float()
        * (inputs.base_height > 0.1).float()
    )
    below_clearance = (
        foot_z < (highest_support_z.unsqueeze(1) + 0.04)
    ).float()
    swing = (1.0 - grounded)
    forward_vel = inputs.support_lin_vel[:, :, 0].clamp(min=0.0)
    vertical_vel = inputs.support_lin_vel[:, :, 2]
    level_or_down = (vertical_vel <= 0.25).float()
    before_goal = (inputs.goal_progress_fraction < 1.0).float()
    penalty = (
        below_clearance * swing * forward_vel * level_or_down
    ).sum(dim=1) * stair_context * before_goal
    return penalty


def build_training_components():
    return TrainingComponents(
        name='go2_stair_official_flat_derived_stance_plane_sweep_v1',
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
                name='swing_foot_stance_plane_sweep_penalty',
                weight=-0.2,
                fn=swing_foot_stance_plane_sweep_penalty,
                clip_min=0.0,
                clip_max=2.0,
                description='Penalize forward sweeps of swing feet below the highest grounded support plane while the foot is not actively lifting, only in a multi-level stair context before the goal.',
            ),
        ],
        rationale=(
            'Anchor is the accepted current-best go2_stair_official_flat_derived_level_sweep_v1 '
            '(objective 0.9483); all eight reference terms and weights are preserved exactly. '
            'The single mechanism change replaces the swing_foot_level_sweep_penalty term with '
            'swing_foot_stance_plane_sweep_penalty. The closest prior candidates are the rejected '
            'stair_swing_foot_low_forward_penalty_v1 (objective 0.9527, joint_ood success 0.4074) '
            'and the accepted stair_swing_foot_level_sweep_penalty_v1 (objective 0.9483, joint_ood '
            'success 0.2222). Exact code-level difference: both prior terms define the low-clearance '
            'zone relative to the base plane (foot_z - base_z in [-0.15, 0.2]), which is body-frame '
            'and does not track the actual upper step on large OOD risers; the new term defines the '
            'hazard zone relative to the highest currently grounded support foot height plus a 0.04 m '
            'clearance margin, and it is gated by a stance-height spread > 0.05 m so flat or '
            'near-flat gaits are not penalized. Hypothesis: the residual joint_ood forbidden-contact '
            'failures (0.667 forbidden contact, 0.222 success in the current best) are swing feet '
            'sweeping forward below the next upper step; referencing the actual highest support plane '
            'makes the trip-hazard penalty scale with riser/tread extrapolation while the stance-spread '
            'gate preserves C0/C1 flat behavior. The raw term clips to [0, 2.0] with weight -0.2, so '
            'the weighted contribution is at most -0.4 per step, and all reference weights are '
            'unchanged to protect the named guards.'
        ),
    )
