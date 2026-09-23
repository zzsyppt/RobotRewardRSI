'''Complete, morphology-scaled standing recovery reward rates with an explicit height-collapse penalty.

The task dispatcher replaces upstream rewards and integrates these rates once
with control dt. Joint penalties use means so changing joint count does not
silently multiply their weight. Perturbation schedules are not reward inputs.
'''

from true_rl_agent.candidates.components import RewardTerm, TrainingComponents


def upright_height(inputs):
    tilt = inputs.roll.square() + inputs.pitch.square()
    height_error = (inputs.base_height_relative - 1.0).square()
    return (-4.0 * tilt - 4.0 * height_error).exp()


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


def height_collapse(inputs):
    deficit = (0.75 - inputs.base_height_relative).clamp(min=0.0)
    tilt = inputs.roll.square() + inputs.pitch.square()
    return deficit.square() * (1.0 + tilt)


def build_training_components():
    return TrainingComponents(
        name='push_standing_recovery_height_margin_v1',
        reward_terms=[
            RewardTerm(name='upright_height', weight=3.0, fn=upright_height),
            RewardTerm(name='stationary_v3', weight=1.0, fn=stationary_v3),
            RewardTerm(name='joint_pose', weight=-0.1, fn=joint_pose),
            RewardTerm(name='action_rate', weight=-0.01, fn=action_rate),
            RewardTerm(name='fall', weight=-20.0, fn=fall),
            RewardTerm(
                name='height_collapse',
                weight=-8.0,
                fn=height_collapse,
                clip_min=0.0,
                clip_max=1.0,
                description='Squared normalized height deficit below 0.75, amplified by tilt.',
            ),
        ],
        rationale='The reference standing-recovery program reaches 0.734 public-dev score but fails mostly on high and repeated perturbations, with height_collapse the largest failure category (26 of 192). The reference only gives a positive exponential height reward, so a collapsed but non-terminated posture is not directly penalized. This candidate preserves every reference reward term and adds one bounded height-collapse penalty: below 0.75 normalized base height it charges the squared height deficit, amplified by tilt. The intended mechanism is an asymmetric, morphology-scaled posture cost that makes severe crouching unattractive while leaving the unchanged upright, damping, joint, action-rate and fall terms to shape normal recovery.',
    )
