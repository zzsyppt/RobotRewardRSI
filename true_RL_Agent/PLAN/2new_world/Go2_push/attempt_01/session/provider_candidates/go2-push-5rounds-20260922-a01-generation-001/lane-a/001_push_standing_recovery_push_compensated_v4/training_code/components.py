'''Complete, morphology-scaled standing recovery reward rates.

The stationary damping term is push-compensated: the task controller reports
the injected planar velocity increment for the current step, and this term
removes that increment from the measured world-frame velocity before applying
the damping penalty. The injected increment is not a policy outcome; residual
motion after the push is still penalized on subsequent steps. All other
mechanisms are unchanged from the reference standing-recovery program.
'''

from true_rl_agent.candidates.components import RewardTerm, TrainingComponents


def upright_height(inputs):
    tilt = inputs.roll.square() + inputs.pitch.square()
    height_error = (inputs.base_height_relative - 1.0).square()
    return (-4.0 * tilt - 4.0 * height_error).exp()


def stationary_push_compensated(inputs):
    gravity = 9.81
    velocity_scale = (gravity * inputs.nominal_base_height).sqrt()
    push_mask = inputs.push_active.float().unsqueeze(1)
    push_scaled = (inputs.push_vec_public / velocity_scale.unsqueeze(1)) * push_mask
    planar_velocity = inputs.base_lin_vel_scaled[:, :2] - push_scaled
    planar_speed = planar_velocity.square().sum(dim=1).sqrt() / 0.08
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


def build_training_components():
    return TrainingComponents(
        name='push_standing_recovery_push_compensated_v4',
        reward_terms=[
            RewardTerm(name='upright_height', weight=3.0, fn=upright_height),
            RewardTerm(name='stationary_push_compensated', weight=1.0, fn=stationary_push_compensated),
            RewardTerm(name='joint_pose', weight=-0.1, fn=joint_pose),
            RewardTerm(name='action_rate', weight=-0.01, fn=action_rate),
            RewardTerm(name='fall', weight=-20.0, fn=fall),
        ],
        rationale='Reference stationary_v3 penalizes total measured planar velocity, including the task controller injected push increment. That increment is not a policy outcome, so the penalty adds uncontrollable negative reward and can discourage recovery motion. This candidate keeps every other mechanism identical and changes only the stationary damping term: it computes the public push increment in the same scaled world-frame units and subtracts it before damping. Residual post-push motion is still penalized on subsequent steps. Hypothesis: removing push-induced velocity from the damping penalty reduces reward noise and improves complete-sequence recovery and the fresh stable dwell.',
    )
