'''Go2 Stair reward: reference flat-derived terms plus a phase-aware swing vertical-clearance shaping term targeting the residual joint_ood forbidden-contact failures from simultaneous tall-riser and short-tread extrapolation.'''

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


def swing_vertical_clearance_rhythm(inputs):
    foot_z = inputs.support_pos_relative_to_terrain_origin[:, :, 2]
    base_z = inputs.support_pos_relative_to_terrain_origin[:, 0, 2:3]
    rel_z = foot_z - base_z
    above_stance = ((rel_z > 0.05) & (rel_z < 0.55)).float()
    swing = (~inputs.support_contact).float()
    forward = inputs.support_lin_vel[:, :, 0].clamp(min=0.0)
    clearance_score = (above_stance * swing * forward).sum(dim=1)
    stance_present = inputs.support_contact.any(dim=1).float()
    moving_forward = (inputs.commanded_base_vel[:, 0] > 0.15).float()
    elevated = (inputs.base_height > 0.1).float()
    before_goal = (inputs.goal_progress_fraction < 1.0).float()
    gate = stance_present * moving_forward * elevated * before_goal
    return clearance_score * gate


def build_training_components():
    return TrainingComponents(
        name='go2_stair_official_flat_derived_swing_vertical_clearance_rhythm_v1',
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
                name='swing_vertical_clearance_rhythm',
                weight=0.15,
                fn=swing_vertical_clearance_rhythm,
                clip_min=0.0,
                clip_max=2.0,
                description='Reward a swing foot that is held above the stance-foot plane while advancing forward on elevated stair terrain before the goal, gated by a planted stance and active forward command.',
            ),
        ],
        rationale="Anchor is the accepted current-best go2_stair_official_flat_derived_level_sweep_v1 (objective 0.9483, j_ood 0.9224, c0_id 1.0, c1_id 1.0). All eight reference terms and weights are preserved exactly. The single mechanism change: replace the negative swing_foot_level_sweep_penalty with the positive swing_vertical_clearance_rhythm bonus. Accepted head carries joint_ood forbidden-contact failures at 0.667 rate, where tall-riser and short-tread extrapolation requires swing foot apex to clear a high riser while tread is short. Prior attempts with the same climb trajectory are negative evidence: accepted level-sweep penalty stops non-lifting forward sweeps yet leaves vertical apex too low; banded gate penalties (0.9169/0.9179 forbidden-contact 0.176/0.162) and lift-velocity reward variants caused regression (height_ood lost by 0.297 for the rise bonus, joint_ood-only lift reward 0.8998). Closest prior candidate is stair_swing_rise_clearance_bonus_v1 (0.9405), rejected when positive airborne lift-velocity shaping inside a base-relative corridor collapsed height_ood success to 0.593 through over-rewarding foot rise while the body moved quickly down-tread. Code-level difference: this term does not reward vertical velocity at all. It rewards forward progress of a swing foot only when the foot is held above the currently planted stance foot by 0.05-0.55 m, and never uses commanded or base height as a lift target. The stance-referenced corridor removes the base-coupled runaway that penalized backward/settled base trajectories, and omitting the up-velocity factor preserves fast stair climbing cadence while demanding a high-foot trajectory. Raw term clipped to [0, 2.0] with weight 0.15; weighted contribution at most +0.3 per step. Gate requires stance contact, forward command, elevated terrain, and pre-arrival progress, so it is silent on flat terrain, at zero command, and after arrival. Hypothesis: enforcing a high swing-foot floor in world space - not a lift-rate - during forward sweeps raises the foot apex above tall risers without slowing climb cadence, fixing joint_ood forbidden contact while holding the c0_id, c1_id, F_c1, and c2 mechanism guards at 1.0.",
    )
