'''Stair reward: replace swing level-sweep penalty with stance contact-slip penalty to stabilize OOD foot placement.'''

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


def support_contact_slip_penalty(inputs):
    stance = inputs.support_contact.float()
    horizontal_speed = inputs.support_lin_vel[:, :, :2].norm(dim=2)
    excess_speed = (horizontal_speed - 0.3).clamp(min=0.0)
    slip = (stance * excess_speed.square()).sum(dim=1)
    elevated = (inputs.base_height > 0.15).float()
    before_goal = (inputs.goal_progress_fraction < 1.0).float()
    return slip * elevated * before_goal


def build_training_components():
    return TrainingComponents(
        name='go2_stair_official_flat_derived_stance_contact_slip_v1',
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
                name='support_contact_slip_penalty',
                weight=-1.5,
                fn=support_contact_slip_penalty,
                clip_min=0.0,
                clip_max=4.0,
                description='Penalize squared excess horizontal speed above 0.3 m/s for force-loaded feet on elevated stair terrain before the goal; arrests touchdown drift on short OOD treads.',
            ),
        ],
        rationale='''Anchor accepted current best go2_stair_official_flat_derived_level_sweep_v1 (objective 0.9483, joint OOD forbidden contact ~0.667, success ~0.222). The swing level-sweep slot penalizes unloaded low feet that advance without vertical lift but does not shape the foot once loaded; on combined riser/tread OOD it can touch down with residual horizontal momentum and roll/slide off a short tread nose into secondary forbidden contacts. This source preserves all eight reference terms with exact weights and changes only that slot by exchanging the mechanism for support_contact_slip_penalty. Closest prior candidates: swing_foot_level_sweep_penalty (current head) and latest-rejected swing_foot_vertical_clearance_penalty_v1, both swing-phase masks on relative height and vertical velocity. The code-level difference is exact: the new term multiplies the loaded-foot force mask support_contact by squared excess horizontal foot speed above 0.3 m/s, gated by base_height > 0.15 and goal_progress_fraction < 1.0; it contains no low-band, no swing mask, no level_or_down gate, and no first-contact scalar. Hypothesis: teaching loaded feet to arrest lateral/forward drift after touchdown on short steep treads reduces slip/off-course and secondary forbidden-contact excursions in joint_ood and c1 frontier cells without changing in-distribution speed, orientation, torque, acceleration, or action-rate objectives.''',
    )
