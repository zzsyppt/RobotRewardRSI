'''Complete morphology-scaled standing recovery reward rates with a height barrier.'''

from true_rl_agent.candidates.components import RewardTerm, TrainingComponents


def upright_height_barrier(inputs):
    tilt = inputs.roll.square() + inputs.pitch.square()
    height_error = (inputs.base_height_relative - 1.0).square()
    reward = (-4.0 * tilt - 4.0 * height_error).exp()
    downward = inputs.base_lin_vel_scaled[:, 2].clamp(max=0.0).abs()
    low_height = (0.8 - inputs.base_height_relative).clamp(min=0.0)
    barrier = 4.0 * low_height.square() * (1.0 + downward.clamp(max=2.0))
    return reward - barrier


def stationary_v3(inputs):
    planar_speed = inputs.base_lin_vel_scaled[:, :2].square().sum(dim=1).sqrt() / 0.08
    angular_speed = inputs.base_ang_vel_scaled.square().sum(dim=1).sqrt() / 0.5
    planar_core = planar_speed.clamp(max=1.0)
    angular_core = angular_speed.clamp(max=1.0)
    planar_penalty = 0.5 * planar_core.square() + planar_speed - planar_core
    angular_penalty = 0.5 * angular_core.square() + angular_speed - angular_core
    return 1.0 - 0.25 * (planar_penalty + angular_penalty)


def joint_pose(inputs):
    return inputs.joint_pos_error.square().mean(dim=1)


def action_rate(inputs):
    return inputs.action_delta.square().mean(dim=1)


def fall(inputs):
    return inputs.terminated_non_timeout.float()


def build_training_components() -> TrainingComponents:
    return TrainingComponents(
        name='push_standing_recovery_height_barrier_v1',
        reward_terms=[
            RewardTerm(name='upright_height_barrier', weight=3.0, fn=upright_height_barrier),
            RewardTerm(name='stationary_v3', weight=1.0, fn=stationary_v3),
            RewardTerm(name='joint_pose', weight=-0.1, fn=joint_pose),
            RewardTerm(name='action_rate', weight=-0.01, fn=action_rate),
            RewardTerm(name='fall', weight=-20.0, fn=fall),
        ],
        rationale='Reference current-best push_standing_recovery_stationary_reference_v3 has public-dev failures dominated by height_collapse (26/192) and high-stratum falls (high stratum 19/48 success). Its upright_height term uses exp(-4*tilt -4*height_error^2), which saturates and leaves only a soft penalty as base_height_relative drops. This candidate changes exactly one component mechanism: the upright_height RewardTerm callable is replaced by upright_height_barrier. The new callable keeps the same exponential nominal stance reward and adds a quadratic barrier on (0.8 - base_height_relative) clamped at zero, multiplied by a bounded factor from downward world-frame base_lin_vel_scaled[2]. Every other reference RewardTerm, weight, and callable is preserved. The added barrier directly penalizes the observed pre-fall height collapse and downward motion while leaving nominal standing and mild crouch behavior unchanged.',
    )
