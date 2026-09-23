'''Morphology-scaled standing recovery reward with a gated capture-step channel.'''

from true_rl_agent.candidates.components import RewardInputs, RewardTerm, TrainingComponents


def capture_step_stance(inputs: RewardInputs):
    tilt = inputs.roll.square() + inputs.pitch.square()
    height_error = (inputs.base_height_relative - 1.0).square()
    stance = (-4.0 * tilt - 4.0 * height_error).exp()
    downward = inputs.base_lin_vel_scaled[:, 2].clamp(max=0.0).abs()
    low_height = (0.8 - inputs.base_height_relative).clamp(min=0.0)
    height_barrier = 4.0 * low_height.square() * (1.0 + downward.clamp(max=2.0))
    g_xy = inputs.projected_gravity[:, :2]
    v_xy = inputs.base_lin_vel[:, :2]
    capture = (v_xy * (-g_xy)).sum(dim=1).clamp(min=-1.0, max=1.0)
    return stance - height_barrier + 0.5 * capture


def stationary_v3(inputs: RewardInputs):
    planar_speed = inputs.base_lin_vel_scaled[:, :2].square().sum(dim=1).sqrt() / 0.08
    angular_speed = inputs.base_ang_vel_scaled.square().sum(dim=1).sqrt() / 0.5
    planar_core = planar_speed.clamp(max=1.0)
    angular_core = angular_speed.clamp(max=1.0)
    planar_penalty = 0.5 * planar_core.square() + planar_speed - planar_core
    angular_penalty = 0.5 * angular_core.square() + angular_speed - angular_core
    return 1.0 - 0.25 * (planar_penalty + angular_penalty)


def joint_pose(inputs: RewardInputs):
    return inputs.joint_pos_error.square().mean(dim=1)


def action_rate(inputs: RewardInputs):
    return inputs.action_delta.square().mean(dim=1)


def fall(inputs: RewardInputs):
    return inputs.terminated_non_timeout.float()


def build_training_components() -> TrainingComponents:
    return TrainingComponents(
        name='push_standing_recovery_capture_step_v1',
        reward_terms=[
            RewardTerm(name='capture_step_stance', weight=3.0, fn=capture_step_stance),
            RewardTerm(name='stationary_v3', weight=1.0, fn=stationary_v3),
            RewardTerm(name='joint_pose', weight=-0.1, fn=joint_pose),
            RewardTerm(name='action_rate', weight=-0.01, fn=action_rate),
            RewardTerm(name='fall', weight=-20.0, fn=fall),
        ],
        rationale='Anchor and evidence. Authoring anchor: accepted current-best push_standing_recovery_height_barrier_v1 (public-dev mean 0.78125). Its one-sided height barrier reduced height_collapse, but the latest nonaccepted sibling push_standing_recovery_impulse_absorb_v1 (public-dev mean 0.8489583333333334; formal_rejected and qualified_not_selected) added a multiplicative tilt-speed-spin barrier and moved the residual failure mass to fall_dominated: per-192 counts fall_backward 7, fall_side 10, height_collapse 7, not_recovered 5, with high stratum 33/48 and repeated stratum 37/48. The multiplicative product is near zero whenever any single risk factor is small, so it suppresses motion only in the joint risk regime, and the remaining failures are falls in which the robot does not move its base under the center of mass. Exact code-level difference from the anchor: exactly one RewardTerm callable is replaced, upright_height_barrier becomes capture_step_stance. The nominal stance reward exp(-4*(roll^2 + pitch^2) - 4*(base_height_relative - 1)^2) and the full height barrier 4*max(0.8 - base_height_relative, 0)^2*(1 + min(max(-v_z, 0), 2)) with v_z = base_lin_vel_scaled[2] are preserved unchanged, so nominal standing, mild crouch, and downward-motion risk are shaped exactly as in the anchor. The new callable adds exactly one mechanism: a capture-step reward computed from base-frame projected_gravity and base-frame base_lin_vel. Let g_xy = projected_gravity[:, :2] and v_xy = base_lin_vel[:, :2]. The added term is 0.5 * clamp((v_xy * (-g_xy)).sum(dim=1), -1, 1). This term is near zero in nominal upright stance because the horizontal gravity projection is near zero, and it becomes positive when the robot is tilted and its base moves in the direction opposite to the horizontal gravity component, i.e. a bounded capture step toward the fall direction. Mechanism hypothesis: the anchor and the rejected impulse_absorb sibling both shape pre-fall risk with penalties that suppress base motion. The residual fall_backward and fall_side failures indicate that the hardest high-stratum and repeated episodes need a controlled step, not only damping. The new channel provides a positive, tilt-gated gradient for that step while leaving the anchor height barrier and the unchanged stationary_v3 damping in place. It does not use base_ang_vel_scaled and does not use the multiplicative risk product of the rejected sibling, so it is a distinct single-component mechanism. Every other callable, weight, diagnostic, and schedule is identical to the anchor. Reward shaping cannot change pulse verification, timing, or the task predicate, and this candidate changes none of them.',
    )
