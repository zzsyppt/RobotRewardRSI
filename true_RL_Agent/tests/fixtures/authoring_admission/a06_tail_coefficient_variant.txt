'''Morphology-scaled standing recovery reward with a height barrier and a pre-terminal velocity brake.'''

from true_rl_agent.candidates.components import RewardInputs, RewardTerm, TrainingComponents


def upright_height_barrier(inputs: RewardInputs):
    tilt = inputs.roll.square() + inputs.pitch.square()
    height_error = (inputs.base_height_relative - 1.0).square()
    reward = (-4.0 * tilt - 4.0 * height_error).exp()
    downward = inputs.base_lin_vel_scaled[:, 2].clamp(max=0.0).abs()
    low_height = (0.8 - inputs.base_height_relative).clamp(min=0.0)
    barrier = 4.0 * low_height.square() * (1.0 + downward.clamp(max=2.0))
    return reward - barrier


def stationary_brake(inputs: RewardInputs):
    planar_speed = inputs.base_lin_vel_scaled[:, :2].square().sum(dim=1).sqrt() / 0.08
    angular_speed = inputs.base_ang_vel_scaled.square().sum(dim=1).sqrt() / 0.5
    planar_core = planar_speed.clamp(max=1.0)
    angular_core = angular_speed.clamp(max=1.0)
    planar_penalty = 0.5 * planar_core.square() + planar_speed - planar_core
    angular_penalty = 0.5 * angular_core.square() + angular_speed - angular_core
    brake = 0.5 * (planar_speed - 1.0).clamp(min=0.0) + 0.5 * (angular_speed - 1.0).clamp(min=0.0)
    return 1.0 - 0.25 * (planar_penalty + angular_penalty) - brake


def joint_pose(inputs: RewardInputs):
    return inputs.joint_pos_error.square().mean(dim=1)


def action_rate(inputs: RewardInputs):
    return inputs.action_delta.square().mean(dim=1)


def fall(inputs: RewardInputs):
    return inputs.terminated_non_timeout.float()


def build_training_components() -> TrainingComponents:
    return TrainingComponents(
        name='push_standing_recovery_velocity_brake_v2',
        reward_terms=[
            RewardTerm(name='upright_height_barrier', weight=3.0, fn=upright_height_barrier),
            RewardTerm(name='stationary_brake', weight=1.0, fn=stationary_brake),
            RewardTerm(name='joint_pose', weight=-0.1, fn=joint_pose),
            RewardTerm(name='action_rate', weight=-0.01, fn=action_rate),
            RewardTerm(name='fall', weight=-20.0, fn=fall),
        ],
        diagnostics=[],
        schedules=[],
        rationale='Anchor and evidence. Authoring anchor: accepted current-best push_standing_recovery_height_barrier_v1 (public_dev mean 0.78125), whose one-sided downward-modulated height barrier addressed the height_collapse-dominated failure reported for the closest prior evaluated candidate push_recovery_height_hinge_v4 (public_dev mean 0.703125). The latest nonaccepted candidate push_standing_recovery_stance_barriers_v2 (public_dev mean 0.8072916666666666, formal_rejected) added a one-sided tilt barrier on top of the anchor height barrier; its public-dev failure breakdown still shows 28 height_collapse, 5 fall_side, 2 fall_backward, 2 not_recovered, with the high push stratum at 22/48. The residual failure mass is therefore still dominated by height collapse and side/backward falls that occur while the base is already translating or rotating: the exponential stance channel and the height barrier both act on posture, but neither directly damps the pre-terminal base velocity that carries the robot into the fall. Exact code-level difference from the anchor: exactly one RewardTerm callable is replaced, stationary_v3 becomes stationary_brake. The nominal stance reward exp(-4*(roll^2 + pitch^2) - 4*(base_height_relative - 1)^2) and the full height barrier 4*max(0.8 - base_height_relative, 0)^2*(1 + min(max(-v_z, 0), 2)) with v_z = base_lin_vel_scaled[2] are preserved unchanged, so nominal standing, mild crouch, and downward-motion risk are shaped exactly as in the anchor. The new callable keeps the same saturating planar and angular speed cores and the same 1.0 - 0.25*(planar_penalty + angular_penalty) shape, and adds exactly one mechanism: a one-sided, non-saturating brake on the pre-terminal base velocity magnitude, brake = 0.5*max(planar_speed - 1.0, 0) + 0.5*max(angular_speed - 1.0, 0), subtracted from the stationary output. Every other callable, weight, diagnostic and schedule is identical to the anchor. Mechanism hypothesis: below the saturating cores (planar_speed <= 1.0 and angular_speed <= 1.0) the brake is exactly zero, so nominal stance and ordinary recovery-step transients are untouched; above that band the brake contributes a persistent linear gradient precisely where the saturating cores have flattened to a constant, giving policy updates a usable pre-terminal signal to arrest base translation and rotation before the -20 fall term fires only at termination. This is a velocity-domain brake, deliberately distinct from the tilt-domain barrier of push_standing_recovery_stance_barriers_v2 and from the height-domain barrier of the anchor, so it is not a repeat of either evaluated program. Reward shaping cannot change pulse verification, timing, or the success definition, and this candidate changes none of them.',
    )
