'''Complete, morphology-scaled standing recovery reward rates.

The task dispatcher replaces upstream rewards and integrates these rates once
with control dt. Joint penalties use means so changing joint count does not
silently multiply their weight. Perturbation schedules are not reward inputs.
'''

from true_rl_agent.candidates.components import RewardTerm, TrainingComponents


def upright_height(inputs):
    tilt = inputs.roll.square() + inputs.pitch.square()
    height_error = (inputs.base_height_relative - 1.0).square()
    return (-4.0 * tilt - 4.0 * height_error).exp()


def stable_damping(inputs):
    '''Reward stillness only when the base is upright and near nominal height.

    The height and gravity gates suppress the stillness reward during acute
    recovery, so a collapsed or tilted posture cannot collect the stationary
    bonus and recovery motion is not penalized by this term.
    '''
    height_gate = ((inputs.base_height_relative - 0.5) / 0.5).clamp(min=0.0, max=1.0)
    upright = (-inputs.projected_gravity[:, 2]).clamp(min=0.0, max=1.0)
    upright_gate = ((upright - 0.5) / 0.5).clamp(min=0.0, max=1.0)
    stable_gate = height_gate * upright_gate

    planar_speed = inputs.base_lin_vel_scaled[:, :2].square().sum(dim=1).sqrt() / 0.08
    angular_speed = inputs.base_ang_vel_scaled.square().sum(dim=1).sqrt() / 0.5
    planar_core = planar_speed.clamp(max=1.0)
    angular_core = angular_speed.clamp(max=1.0)
    planar_penalty = 0.5 * planar_core.square() + planar_speed - planar_core
    angular_penalty = 0.5 * angular_core.square() + angular_speed - angular_core

    return stable_gate * (1.0 - 0.25 * (planar_penalty + angular_penalty))


def joint_pose(inputs):
    return inputs.joint_pos_error.square().mean(dim=1)


def action_rate(inputs):
    return inputs.action_delta.square().mean(dim=1)


def fall(inputs):
    return inputs.terminated_non_timeout.float()


def build_training_components():
    return TrainingComponents(
        name='push_standing_recovery_posture_gated_damping_v1',
        reward_terms=[
            RewardTerm(name='upright_height', weight=3.0, fn=upright_height),
            RewardTerm(
                name='stable_damping',
                weight=1.0,
                fn=stable_damping,
                clip_min=0.0,
                clip_max=1.0,
            ),
            RewardTerm(name='joint_pose', weight=-0.1, fn=joint_pose),
            RewardTerm(name='action_rate', weight=-0.01, fn=action_rate),
            RewardTerm(name='fall', weight=-20.0, fn=fall),
        ],
        rationale=(
            'The reference program stationary_v3 term grants full stillness '
            'reward regardless of base height or orientation, so a robot that '
            'collapses and remains still can collect the stationary bonus, and '
            'the same term penalizes the leg motion needed to recover from a '
            'large push. Replace only that mechanism with stable_damping: a '
            'posture-gated stillness reward that multiplies the original '
            'velocity penalty by smooth height and projected-gravity gates. '
            'The gates are zero when the base is low or tilted, so collapsed '
            'postures cannot collect the bonus, while recovery motion during '
            'the acute phase is not penalized. This targets the observed '
            'height-collapse failure mode and the high/repeated push strata '
            'without changing other reward mechanisms, weights, or task '
            'protocol.'
        ),
    )
