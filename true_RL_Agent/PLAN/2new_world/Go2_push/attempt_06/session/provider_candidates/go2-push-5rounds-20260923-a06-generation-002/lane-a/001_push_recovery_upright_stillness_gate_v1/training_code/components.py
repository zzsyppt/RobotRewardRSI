'''Morphology-scaled standing recovery reward with an upright stillness gate.'''

from true_rl_agent.candidates.components import RewardInputs, RewardTerm, TrainingComponents


def upright_height_barrier(inputs: RewardInputs):
    tilt = inputs.roll.square() + inputs.pitch.square()
    height_error = (inputs.base_height_relative - 1.0).square()
    reward = (-4.0 * tilt - 4.0 * height_error).exp()
    downward = inputs.base_lin_vel_scaled[:, 2].clamp(max=0.0).abs()
    low_height = (0.8 - inputs.base_height_relative).clamp(min=0.0)
    barrier = 4.0 * low_height.square() * (1.0 + downward.clamp(max=2.0))
    return reward - barrier


def stationary_recovery(inputs: RewardInputs):
    tilt = inputs.roll.square() + inputs.pitch.square()
    deficit = (1.0 - inputs.base_height_relative).clamp(min=0.0)
    gate = (-2.0 * tilt - 6.0 * deficit.square()).exp()
    planar_speed = inputs.base_lin_vel_scaled[:, :2].square().sum(dim=1).sqrt() / 0.08
    angular_speed = inputs.base_ang_vel_scaled.square().sum(dim=1).sqrt() / 0.5
    planar_core = planar_speed.clamp(max=1.0)
    angular_core = angular_speed.clamp(max=1.0)
    planar_penalty = 0.5 * planar_core.square() + planar_speed - planar_core
    angular_penalty = 0.5 * angular_core.square() + angular_speed - angular_core
    return gate * (1.0 - 0.25 * (planar_penalty + angular_penalty))


def joint_pose(inputs: RewardInputs):
    return inputs.joint_pos_error.square().mean(dim=1)


def action_rate(inputs: RewardInputs):
    return inputs.action_delta.square().mean(dim=1)


def fall(inputs: RewardInputs):
    return inputs.terminated_non_timeout.float()


def build_training_components() -> TrainingComponents:
    return TrainingComponents(
        name='push_recovery_upright_stillness_gate_v1',
        reward_terms=[
            RewardTerm(name='upright_height_barrier', weight=3.0, fn=upright_height_barrier),
            RewardTerm(name='stationary_recovery', weight=1.0, fn=stationary_recovery),
            RewardTerm(name='joint_pose', weight=-0.1, fn=joint_pose),
            RewardTerm(name='action_rate', weight=-0.01, fn=action_rate),
            RewardTerm(name='fall', weight=-20.0, fn=fall),
        ],
        diagnostics=[],
        schedules=[],
        rationale='Closest prior is the accepted current-best push_standing_recovery_height_barrier_v1. The single changed mechanism is the stance stillness RewardTerm: stationary_v3 is replaced by stationary_recovery. stationary_v3 returns a full +1.0 stillness reward whenever planar and angular speeds are zero, even when the robot is crouched or tilted, and it penalizes planar and angular motion with the same gain throughout recovery. stationary_recovery keeps the identical speed-penalty shape at the upright stance but multiplies the entire term by an upright gate exp(-2*(roll^2+pitch^2) - 6*max(1-base_height_relative,0)^2). At nominal height and level orientation the gate is 1.0, so the final stable-dwell stillness reward is unchanged. During a large-push transient the gate shrinks toward zero, removing the crouch-or-still bonus and reducing the penalty on the stepping and limb motion needed to arrest backward and side momentum. The upright_height_barrier, joint_pose, action_rate, and fall terms and all weights are preserved. This targets the remaining high-stratum and repeated-push falls and the height-collapse failure without changing push timing, success predicates, or the training protocol.',
    )
