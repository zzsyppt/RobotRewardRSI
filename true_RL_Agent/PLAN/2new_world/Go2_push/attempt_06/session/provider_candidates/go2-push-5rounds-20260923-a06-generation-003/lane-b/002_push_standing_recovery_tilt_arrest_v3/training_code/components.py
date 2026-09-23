'''Morphology-scaled standing recovery reward with a one-sided tilt-arrest barrier.'''

from true_rl_agent.candidates.components import RewardInputs, RewardTerm, TrainingComponents


def upright_tilt_arrest(inputs: RewardInputs):
    tilt = inputs.roll.square() + inputs.pitch.square()
    height_error = (inputs.base_height_relative - 1.0).square()
    stance = (-4.0 * tilt - 4.0 * height_error).exp()
    downward = inputs.base_lin_vel_scaled[:, 2].clamp(max=0.0).abs()
    low_height = (0.8 - inputs.base_height_relative).clamp(min=0.0)
    height_barrier = 4.0 * low_height.square() * (1.0 + downward.clamp(max=2.0))
    tilt_excess = (tilt - 0.25).clamp(min=0.0)
    tilt_barrier = 2.0 * tilt_excess + tilt_excess.square()
    return stance - height_barrier - tilt_barrier


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
        name='push_standing_recovery_tilt_arrest_v3',
        reward_terms=[
            RewardTerm(name='upright_tilt_arrest', weight=3.0, fn=upright_tilt_arrest, clip_min=-3.0, description='Upright stance reward with one-sided height and tilt pre-fall barriers.'),
            RewardTerm(name='stationary_v3', weight=1.0, fn=stationary_v3),
            RewardTerm(name='joint_pose', weight=-0.1, fn=joint_pose),
            RewardTerm(name='action_rate', weight=-0.01, fn=action_rate),
            RewardTerm(name='fall', weight=-20.0, fn=fall),
        ],
        diagnostics=[],
        schedules=[],
        rationale='Anchor and evidence. Authoring anchor: accepted current-best push_standing_recovery_height_barrier_v1 (public_dev mean 0.78125). Its one-sided, downward-modulated height barrier addressed the height_collapse-dominated failure reported for the closest prior evaluated candidate push_recovery_height_hinge_v4 (public_dev mean 0.703125; per-192 counts 29 height_collapse, 14 fall_backward, 9 fall_side; high push stratum 18/48, repeated stratum 27/48). The residual failure mass is therefore expected to sit in tilt-dominated fall modes that the exponential stance channel can no longer suppress once it saturates. Exact code-level difference from the anchor: exactly one RewardTerm callable is replaced, upright_height_barrier becomes upright_tilt_arrest. The nominal stance reward exp(-4*(roll^2 + pitch^2) - 4*(base_height_relative - 1)^2) and the full height barrier 4*max(0.8 - base_height_relative, 0)^2*(1 + min(max(-v_z, 0), 2)) with v_z = base_lin_vel_scaled[2] are preserved unchanged, so nominal standing, mild crouch, and downward-motion risk are shaped exactly as in the anchor. The new callable adds exactly one mechanism: a one-sided, non-saturating tilt barrier with tilt = roll^2 + pitch^2, excess = max(tilt - 0.25, 0) and tilt_barrier = 2*excess + excess^2, and the stance term output is bounded below by clip_min = -3.0. Every other callable, weight, diagnostic and schedule is identical to the anchor. Mechanism hypothesis: below about 0.5 rad combined tilt the barrier is exactly zero, so nominal stance and ordinary recovery-step transients are untouched; above that band it contributes a persistent linear-plus-quadratic gradient precisely where exp(-4*tilt) has decayed to a near-zero slope and the -20 fall term fires only at termination, giving policy updates a usable pre-terminal signal to arrest tilt growth and convert high-stratum backward and side falls into recovered episodes. The replaced callable deliberately does not include planar or angular velocity damping (that stays in the unchanged stationary_v3 callable), so the new mechanism is a barrier-shaped pre-fall risk channel rather than a fused stationary-recovery composite. Reward shaping cannot change pulse verification, timing, or the success definition, and this candidate changes none of them.',
    )
