'''Morphology-scaled standing recovery reward with a one-sided impulse-absorption barrier.'''

from true_rl_agent.candidates.components import RewardInputs, RewardTerm, TrainingComponents


def impulse_absorb(inputs: RewardInputs):
    tilt = inputs.roll.square() + inputs.pitch.square()
    height_error = (inputs.base_height_relative - 1.0).square()
    stance = (-4.0 * tilt - 4.0 * height_error).exp()
    downward = inputs.base_lin_vel_scaled[:, 2].clamp(max=0.0).abs()
    low_height = (0.8 - inputs.base_height_relative).clamp(min=0.0)
    height_barrier = 4.0 * low_height.square() * (1.0 + downward.clamp(max=2.0))
    planar_speed = inputs.base_lin_vel_scaled[:, :2].square().sum(dim=1).sqrt()
    angular_speed = inputs.base_ang_vel_scaled.square().sum(dim=1).sqrt()
    tilt_risk = tilt / (tilt + 0.25)
    speed_risk = planar_speed / (planar_speed + 0.5)
    spin_risk = angular_speed / (angular_speed + 1.0)
    impulse_risk = tilt_risk * speed_risk * spin_risk
    return stance - height_barrier - 2.0 * impulse_risk


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
        name='push_standing_recovery_impulse_absorb_v1',
        reward_terms=[
            RewardTerm(name='impulse_absorb', weight=3.0, fn=impulse_absorb, clip_min=-3.0, description='Upright stance reward with height barrier and a one-sided multiplicative impulse-absorption barrier.'),
            RewardTerm(name='stationary_v3', weight=1.0, fn=stationary_v3),
            RewardTerm(name='joint_pose', weight=-0.1, fn=joint_pose),
            RewardTerm(name='action_rate', weight=-0.01, fn=action_rate),
            RewardTerm(name='fall', weight=-20.0, fn=fall),
        ],
        diagnostics=[],
        schedules=[],
        rationale='Anchor and evidence. Authoring anchor: accepted current-best push_standing_recovery_height_barrier_v1 (public_dev mean 0.78125). Its one-sided, downward-modulated height barrier addressed the height_collapse-dominated failure reported for the closest prior evaluated candidate push_recovery_height_hinge_v4 (public_dev mean 0.703125; per-192 counts 29 height_collapse, 14 fall_backward, 9 fall_side; high push stratum 18/48, repeated stratum 27/48). The latest nonaccepted sibling push_standing_recovery_stance_barriers_v2 (public_dev mean 0.8072916666666666; per-192 counts 28 height_collapse, 5 fall_side, 2 fall_backward, 2 not_recovered; high stratum 22/48, repeated 38/48) added a one-sided additive tilt barrier on top of the same stance and height channels. Its residual failure mass is still height_collapse-dominated, and its high-stratum success (22/48) remains the weakest stratum, so the additive tilt channel did not convert the hardest episodes. Exact code-level difference from the anchor: exactly one RewardTerm callable is replaced, upright_height_barrier becomes impulse_absorb. The nominal stance reward exp(-4*(roll^2 + pitch^2) - 4*(base_height_relative - 1)^2) and the full height barrier 4*max(0.8 - base_height_relative, 0)^2*(1 + min(max(-v_z, 0), 2)) with v_z = base_lin_vel_scaled[2] are preserved unchanged, so nominal standing, mild crouch, and downward-motion risk are shaped exactly as in the anchor. The new callable adds exactly one mechanism: a one-sided multiplicative impulse-absorption barrier built from three bounded saturating risk factors, tilt_risk = tilt/(tilt + 0.25) with tilt = roll^2 + pitch^2, speed_risk = |v_xy|/(|v_xy| + 0.5) using world-frame base_lin_vel_scaled[:, :2], and spin_risk = |omega|/(|omega| + 1.0) using base_ang_vel_scaled, combined as impulse_risk = tilt_risk * speed_risk * spin_risk and subtracted as 2.0 * impulse_risk, with the stance term output bounded below by clip_min = -3.0. Every other callable, weight, diagnostic and schedule is identical to the anchor. Mechanism hypothesis: the anchor and the additive-tilt sibling both apply their pre-fall risk channels as separable one-dimensional penalties, so a robot that is simultaneously tilted, translating, and rotating still receives only the sum of individually bounded terms and the hardest high-stratum episodes remain height_collapse-dominated. The multiplicative conjunction is near zero whenever any single risk factor is small, so nominal stance, mild crouch, and ordinary recovery-step transients are untouched, but it grows sharply only in the joint tilt-plus-translation-plus-rotation regime that immediately precedes the observed collapse, giving policy updates a usable pre-terminal gradient concentrated exactly on the hardest episodes. The replaced callable deliberately does not include planar or angular velocity damping as standalone terms (that stays in the unchanged stationary_v3 callable), so the new mechanism is a conjunctive pre-fall risk channel rather than a fused stationary-recovery composite. Reward shaping cannot change pulse verification, timing, or the success definition, and this candidate changes none of them.',
    )
