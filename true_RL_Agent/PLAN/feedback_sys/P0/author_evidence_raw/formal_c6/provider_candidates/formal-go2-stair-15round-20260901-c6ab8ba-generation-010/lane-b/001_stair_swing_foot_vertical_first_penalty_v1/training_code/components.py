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
    in_contact_previous = inputs.support_contact
    vertical_vel_signed = inputs.support_lin_vel[:, :, 2]
    forward_vel = inputs.support_lin_vel[:, :, 0].clamp(min=0.0)
    airtime_gate = (inputs.support_air_time > 0.02).float()
    recent_contact = in_contact_previous.float()
    caught_low = ((rel_z <= 0.0) & (recent_contact == 1.0)).float()
    low_rising = ((vertical_vel_signed > 0.0) & caught_low & airtime_gate).float()
    elevated = (inputs.base_height > 0.1).float()
    before_goal = (inputs.goal_progress_fraction < 1.0).float()
    penalty = (
        low_band * swing * forward_vel * low_rising
    ).sum(dim=1) * elevated * before_goal
    return penalty


def build_training_components():
    return TrainingComponents(
        name='go2_stair_official_flat_derived_vertical_first_v1',
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
                description='Penalize horizontal advance of a swing foot only while it leaves the step in a low band and is still rising after contact break, so vertical-first foot clearance is enforced without deterring established mid-swing or descending foot placement.',
            ),
        ],
        rationale=(
            'Anchor is the verified current-best accepted source '
            'go2_stair_official_flat_derived_level_sweep_v1 (objective 0.9483, '
            'joint_ood forbidden_contact 66.7 percent, joint_ood success 0.222). '
            'All eight reference terms and weights are preserved exactly. '
            'The single mechanism change replaces the anchor term '
            'swing_foot_level_sweep_penalty (penalizes any swing-foot forward '
            'sweep with vertical_vel <= 0.2) with swing_foot_vertical_first_penalty. '
            'The code-level differences are: (1) the penalty is active only when '
            'the foot just separated from a support (support_contact true) and is '
            'still early in the swing (support_air_time > 0.02); (2) it applies '
            'only while the foot is still low near the departure surface '
            '(foot_z <= base_z) and still rising (vertical_vel > 0); (3) it never '
            'penalizes later swing phase or descending placement, where horizontal '
            'advance is productive. This is distinct from the rejected '
            'swing_foot_vertical_lift_penalty (objective 0.9179) which penalized '
            'low-band forward sweeps whenever vertical_vel < 0.3 without any '
            'takeoff/early-swing gate and drove height_ood forbidden_contact to '
            '48.1 percent and slip to 85.2 percent; the new mechanism therefore '
            'targets the pathological low takeoff slice on the joint riser/tread '
            'extrapolation while preserving late-swing terrain clearance and '
            'placement. The raw term is clipped to [0, 2.0] with weight -0.2, so '
            'its weighted contribution is at most -0.4 per step. Activation gates '
            '(low_band, swing, base_height > 0.1, goal_progress_fraction < 1.0) '
            'keep it silent on the flat start, flat behavior, and after arrival. '
            'Hypothesis: OOD forbidden-contact failures are dominated by feet that '
            'brush the next riser during takeoff because they advance horizontally '
            'before clearing the current edge; a vertical-first early-swing penalty '
            'teaches the policy to lift first, improving height_ood and joint_ood '
            'without disturbing C0/C1 in-distribution locomotion.'
        ),
    )
