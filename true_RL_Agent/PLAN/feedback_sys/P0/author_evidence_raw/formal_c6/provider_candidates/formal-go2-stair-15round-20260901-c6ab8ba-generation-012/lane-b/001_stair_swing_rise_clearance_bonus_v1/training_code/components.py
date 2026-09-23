'''Go2 Stair reward: reference flat-derived terms plus a positive swing-rise clearance bonus that targets the tall-riser/short-tread OOD forbidden-contact bottleneck without re-issuing a low-sweep punishment.'''

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


def swing_rise_clearance_bonus(inputs):
    foot_z = inputs.support_pos_relative_to_terrain_origin[:, :, 2]
    base_z = inputs.base_pos_relative_to_terrain_origin[:, 2].unsqueeze(1)
    rel_z = foot_z - base_z
    lift_corridor = ((rel_z > -0.45) & (rel_z < 0.1)).float()
    swing = (~inputs.support_contact).float()
    up_vel = inputs.support_lin_vel[:, :, 2].clamp(min=0.0)
    lift_score = (lift_corridor * swing * up_vel).sum(dim=1)
    stance_present = inputs.support_contact.any(dim=1).float()
    moving_forward = (inputs.commanded_base_vel[:, 0] > 0.15).float()
    elevated = (inputs.base_height > 0.1).float()
    before_goal = (inputs.goal_progress_fraction < 1.0).float()
    gate = stance_present * moving_forward * elevated * before_goal
    return lift_score * gate


def build_training_components():
    return TrainingComponents(
        name='go2_stair_official_flat_derived_swing_rise_clearance_v1',
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
                name='swing_rise_clearance_bonus',
                weight=0.2,
                fn=swing_rise_clearance_bonus,
                clip_min=0.0,
                clip_max=2.5,
                description='Reward airborne feet that are actively rising through the lift corridor on elevated stair terrain before the goal, while at least one foot remains planted and a forward command is active.',
            ),
        ],
        rationale=('Anchor is the accepted current-best go2_stair_official_flat_derived_level_sweep_v1 '
                   '(objective 0.9483, j_ood 0.9224, c0_id 1.0, c1_id 1.0). All eight reference '
                   'terms and weights are preserved exactly. The single mechanism change replaces the '
                   'negative swing_foot_level_sweep_penalty with the positive swing_rise_clearance_bonus. '
                   'The accepted head still loses joint_ood (score 0.412, success 0.222, forbidden '
                   'contact 0.667, slip_detected 0.667): simultaneous tall-riser and short-tread '
                   'extrapolation fails because the swing foot does not gain enough vertical rise before '
                   'the body advances, so the shin strikes the riser. The closest prior positive attempt, '
                   'stair_swing_forward_nudge_success_only_v1, shaped only sustained low-foot contact '
                   'and was effectively silent at runtime (last_mean 6.9e-07), so it provided no '
                   'learning gradient for the lifting phase. The exact code-level difference is that this '
                   'term scores the airborne vertical-velocity component of every foot inside a usable '
                   'mid-extension corridor relative to the base (rel_z in (-0.45, 0.1) m), gated by a '
                   'planted stance, a forward command, elevated terrain, and pre-arrival progress. It is '
                   'positive so it teaches vertical-first foot clearance instead of freezing forward '
                   'motion like the rejected negative variants (0.9183/0.9169). The raw term is clipped '
                   'to [0, 2.5] with weight 0.2, so its weighted contribution is at most +0.5 per step; '
                   'the gate keeps it silent at the goal and during zero-command flight. Hypothesis: '
                   'directly rewarding upward swing velocity in the lift corridor creates a smooth '
                   'gradient that raises the foot trajectory above tall risers during forward progress, '
                   'reducing forbidden-contact failures on joint_ood while preserving the C0/C1 '
                   'in-distribution guards and the C2 mechanism generalization.')
    )
