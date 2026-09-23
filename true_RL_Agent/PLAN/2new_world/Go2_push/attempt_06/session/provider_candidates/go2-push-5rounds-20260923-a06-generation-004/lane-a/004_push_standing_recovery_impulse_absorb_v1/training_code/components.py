'''Complete morphology-scaled standing recovery reward rates with impulse-absorbing stance damping.'''

from true_rl_agent.candidates.components import RewardTerm, TrainingComponents


def upright_height_barrier(inputs):
    tilt = inputs.roll.square() + inputs.pitch.square()
    height_error = (inputs.base_height_relative - 1.0).square()
    reward = (-4.0 * tilt - 4.0 * height_error).exp()
    downward = inputs.base_lin_vel_scaled[:, 2].clamp(max=0.0).abs()
    low_height = (0.8 - inputs.base_height_relative).clamp(min=0.0)
    barrier = 4.0 * low_height.square() * (1.0 + downward.clamp(max=2.0))
    return reward - barrier


def impulse_absorb(inputs):
    planar_speed = inputs.base_lin_vel_scaled[:, :2].square().sum(dim=1).sqrt()
    angular_speed = inputs.base_ang_vel_scaled.square().sum(dim=1).sqrt()
    tilt = inputs.roll.square() + inputs.pitch.square()
    height_deficit = (1.0 - inputs.base_height_relative).clamp(min=0.0)
    excitation = planar_speed + 0.5 * angular_speed + 2.0 * tilt + 2.0 * height_deficit
    return -excitation.square()


def joint_pose(inputs):
    return inputs.joint_pos_error.square().mean(dim=1)


def action_rate(inputs):
    return inputs.action_delta.square().mean(dim=1)


def fall(inputs):
    return inputs.terminated_non_timeout.float()


def build_training_components() -> TrainingComponents:
    return TrainingComponents(
        name='push_standing_recovery_impulse_absorb_v1',
        reward_terms=[
            RewardTerm(name='upright_height_barrier', weight=3.0, fn=upright_height_barrier),
            RewardTerm(name='impulse_absorb', weight=-0.5, fn=impulse_absorb),
            RewardTerm(name='joint_pose', weight=-0.1, fn=joint_pose),
            RewardTerm(name='action_rate', weight=-0.01, fn=action_rate),
            RewardTerm(name='fall', weight=-20.0, fn=fall),
        ],
        rationale='Reference current-best push_standing_recovery_height_barrier_v1 (public-dev mean 0.78125) has residual failures dominated by height_collapse (28/192) and a weak high stratum (22/48), i.e. after the largest pulses the base still collapses or drifts instead of settling. Its stationary_v3 term is a piecewise-linear speed penalty that is exactly flat once planar_speed exceeds 0.08 m/s and angular_speed exceeds 0.5 rad/s, so it provides no gradient during the large post-push excursion that precedes those failures. This candidate changes exactly one component mechanism: the stationary_v3 RewardTerm callable is replaced by impulse_absorb. The new callable forms a single bounded excitation scalar from morphology-scaled planar speed, angular speed, tilt, and height deficit, and returns its negative square, so the penalty grows smoothly and keeps a usable gradient throughout the large-excursion regime while remaining near zero at the nominal stance. The upright_height_barrier callable, the joint_pose, action_rate, and fall callables, and every weight are preserved unchanged. The mechanism-level difference from the rejected tilt_arrest_v7 is that this is a velocity/height excitation absorber, not a tilt barrier, and it does not duplicate the height barrier already present in upright_height_barrier.',
    )
