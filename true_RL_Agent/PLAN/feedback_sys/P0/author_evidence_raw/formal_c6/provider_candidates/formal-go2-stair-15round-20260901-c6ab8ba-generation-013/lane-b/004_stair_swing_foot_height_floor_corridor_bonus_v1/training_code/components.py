'''Go2 Stair reward: reference flat-derived terms plus a saturating gait-phase swing-foot height-floor bonus replacing the failed swing-vertical-clearance rhythm term and the accepted level-sweep penalty.'''

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


def swing_foot_height_floor_corridor(inputs):
    foot_z = inputs.support_pos_relative_to_terrain_origin[:, :, 2]
    base_z = inputs.base_pos_relative_to_terrain_origin[:, 2:3]
    rel_z = foot_z - base_z
    stance = inputs.support_contact.any(dim=1).float()
    swing = (~inputs.support_contact).float()
    in_corridor = ((rel_z > 0.05) & (rel_z < 0.55)).float()
    moving_forward = (inputs.commanded_base_vel[:, 0] > 0.15).float()
    elevated = (inputs.base_height > 0.12).float()
    before_goal = (inputs.goal_progress_fraction < 1.0).float()
    gated = swing * in_corridor * stance * moving_forward * elevated * before_goal
    held_count = gated.sum(dim=1)
    bonus = ((held_count / (held_count + 1.0)) * 2.0).clamp(max=2.0)
    return bonus


def build_training_components():
    return TrainingComponents(
        name='go2_stair_official_flat_derived_swing_foot_height_floor_corridor_v1',
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
                name='swing_foot_height_floor_corridor',
                weight=0.35,
                fn=swing_foot_height_floor_corridor,
                clip_min=0.0,
                clip_max=2.0,
                description='Saturating bonus for holding a non-contact foot inside a height-floor corridor above the base while climbing forward before the goal, gated by a planted stance.',
            ),
        ],
        rationale="Anchor is the accepted current-best go2_stair_official_flat_derived_level_sweep_v1 (objective 0.9483; j_ood 0.9224; c0_id 1.0; c1_id 1.0) with joint_ood forbidden-contact failures at 0.667 on simultaneous tall-riser and short-tread extrapolation. The immediately preceding nonaccepted source go2_stair_official_flat_derived_swing_vertical_clearance_rhythm_v1 replaced the level-sweep penalty with a positive swing_vertical_clearance_rhythm bonus and was rejected. Closest prior candidate is that swing_vertical_clearance_rhythm term. Code-level difference: the prior term rewarded the raw count of corridor-holding swing feet multiplied by positive support-forward velocity and took its reference plane from support member zero, so reward magnitude scaled with swing count and cadence while the reference plane was tied to opaque support-member order; the new term removes the support-velocity product and all support-member indexing, references the base frame, counts corridor-holding swing feet only while at least one stance contact exists, and normalizes that count to a saturating bonus capped at 2.0. It remains silent on flat terrain, at zero forward command, and after arrival. Hypothesis: a bounded stance-gated floor on swing-foot height during forward climbing raises the foot apex enough to clear tall risers on short treads without making swing count, foot forward velocity, or flight cadence the reward driver, thereby fixing joint_ood forbidden contact while preserving the accepted c0_id, c1_id, F_c1, and c2 mechanism guards.",
    )
