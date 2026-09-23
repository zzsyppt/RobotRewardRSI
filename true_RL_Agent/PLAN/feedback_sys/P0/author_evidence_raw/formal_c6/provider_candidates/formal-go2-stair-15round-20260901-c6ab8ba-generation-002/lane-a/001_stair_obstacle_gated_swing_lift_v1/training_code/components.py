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

def foot_swing_lift_on_step(inputs):
    step_signal = (-inputs.height_scan.min(dim=-1).values).clamp(min=0.0, max=1.0)
    upward_rate = inputs.support_lin_vel[:, 2].clamp(min=0.0)
    airborne = (inputs.support_air_time > 0.0).float()
    return (upward_rate * airborne).sum(dim=1) * step_signal

def build_training_components():
    return TrainingComponents(
        name='go2_stair_obstacle_gated_swing_lift_v1',
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
                name='obstacle_gated_swing_lift',
                weight=0.5,
                fn=foot_swing_lift_on_step,
                clip_min=0.0,
                clip_max=1.0,
                description='Rewards upward airborne foot velocity only while the downward height scan sees elevated terrain closer than the flat reference; bounded total weighted contribution +0.5.',
            ),
        ],
        rationale=(
            'Anchor is verified current-best go2_stair_official_flat_derived_v1 '
            '(public dev objective 0.9344; c0_id, c1_id, phase_composition_ood, '
            'c1_conditional_frontier, and c2_mechanism_ood all 1.0). The scalar '
            'OOD family is the only weak axis (F_scalar 0.705): joint_ood 0.493 '
            'with forbidden_contact 0.519, tread_ood 0.734 with forbidden_contact '
            '0.333, and height_ood 0.888 with forbidden_contact 0.148. Tall risers '
            'and short treads therefore fail primarily by foot/body contact with '
            'the next step. Two prior add-on mechanisms were evaluated and '
            'nonpositive: stair_forbidden_contact_force_penalty_v1 added a dense '
            'post-contact force penalty (guard j_ood 0.8776) and stair_climb_lift_v1 '
            'rewarded base upward velocity (guard j_ood 0.8705). This candidate does '
            'not reuse either mechanism. It adds exactly one new mechanism: '
            'obstacle-gated swing lift. The downward height scan is negative for '
            'terrain closer/higher than the flat reference, so -min(height_scan) is '
            'a smooth terrain-proximity gate in [0,1] computed only by an order-invariant '
            'reduction (the ray mapping is not exposed). The term then rewards upward '
            'vertical foot velocity only for airborne feet, shaping leg clearance over '
            'risers before contact. Unlike the force penalty it acts before collision '
            'and does not reward stopping; unlike climb_lift it shapes foot lifting '
            'instead of base bouncing. The term is clipped to [0,1] with weight 0.5, '
            'so its weighted contribution is bounded by +0.5. All eight reference '
            'terms and weights are preserved verbatim.'
        ),
    )
