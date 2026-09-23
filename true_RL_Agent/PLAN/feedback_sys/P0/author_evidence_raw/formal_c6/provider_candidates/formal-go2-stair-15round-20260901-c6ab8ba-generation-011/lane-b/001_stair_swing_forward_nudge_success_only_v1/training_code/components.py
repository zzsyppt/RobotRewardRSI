'''Go2 Stair reward: reference flat-derived terms plus an activation-balanced forward swing nudge that shortens blocked swing duration on OOD riser/tread extrapolation.'''

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


def ongoing_support_contact(inputs):
    contact_any = inputs.support_contact.any(dim=1)
    contact_time_sum = inputs.support_contact_time.sum(dim=1)
    return (contact_any.float() * contact_time_sum).clamp(min=0.1, max=20.0)


def blocked_swing_forward_nudge(inputs):
    low_band = ((inputs.support_pos_relative_to_terrain_origin[:, :, 2]
                 - inputs.base_pos_relative_to_terrain_origin[:, 2].unsqueeze(1)) < 0.08).float()
    contact = inputs.support_contact.float()
    low_contact = (low_band * contact).any(dim=1).float()
    ongoing_low_contact = low_contact / ongoing_support_contact(inputs)
    fwd_force_trend = inputs.support_contact_force[:, :, 0].clamp(min=0.0) * contact
    nudge = fwd_force_trend.sum(dim=1) * ongoing_low_contact
    elevated = (inputs.base_height > 0.1).float()
    before_goal = (inputs.goal_progress_fraction < 1.0).float()
    return nudge * elevated * before_goal


def build_training_components():
    return TrainingComponents(
        name='go2_stair_official_flat_derived_blocked_swing_nudge_v1',
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
                name='blocked_swing_forward_nudge',
                weight=0.2,
                fn=blocked_swing_forward_nudge,
                clip_min=0.0,
                clip_max=2.0,
                description='Reward persistent low forward contact force on elevated stair terrain before the goal, balanced by total ongoing support contact so the bonus is not a standing or static push reward.',
            ),
        ],
        rationale=('Current best is go2_stair_official_flat_derived_level_sweep_v1 '
                   '(objective 0.9483) with the trip hazard mechanism '
                   'swing_foot_level_sweep_penalty. It reduced forbidden contact in '
                   'height_ood (11.1 percent) and tread_ood (0 percent) but left '
                   'joint_ood blocked: success 0.222 with forbidden_contact 66.7 '
                   'percent and only 0.602 mean progress, and the sibling '
                   'vertical-first variants that kept the negative gate fell '
                   'further (objective 0.9179/0.9169). The negative penalty family '
                   'has therefore produced nonpositive results and is suppressed '
                   'by experiment memory; this candidate is not a punishment '
                   'variant. The closest prior candidate is the current best itself; '
                   'the exact code-level difference is replacing its negative '
                   'swing-foot sweep penalty with a positive activation-ratio '
                   'forward-push nudge that targets the same blocked-swing phase '
                   'on high OOD risers. The reward fires only when at least one '
                   'low foot is currently in sustained contact (rel_z < 0.08), so '
                   'it exists precisely on the OOD failure mode in which a swing '
                   'foot is scraping against the riser instead of clearing it. '
                   'It is normalized by the total ongoing support-contact time and '
                   'weighted +0.2 with clip [0,2], so prolonged static pushing has '
                   'almost no incentive (contact_time grows in the denominator), '
                   'whereas a quick push that begins to move the body is rewarded. '
                   'Forces are raw physical magnitudes; the finite clip_min/max and '
                   'weight make the weighted contribution at most +0.4 per step. '
                   'The positive sign means it also teaches resolvable low-foot '
                   'contact instead of passively freezing or delaying the swing. '
                   'Hypothesis: penalizing low forward motion over-corrected to '
                   'pausing and orientation instability (slip 0.40/0.27 in the two '
                   'negative rejections), while a positive balanced nudge on the '
                   'same low-contact mode clears the riser by pushing upward and '
                   'forward, protecting j_ood and leaving the C0/C1 in-distribution '
                   'locomotion guards unchanged.')
    )
