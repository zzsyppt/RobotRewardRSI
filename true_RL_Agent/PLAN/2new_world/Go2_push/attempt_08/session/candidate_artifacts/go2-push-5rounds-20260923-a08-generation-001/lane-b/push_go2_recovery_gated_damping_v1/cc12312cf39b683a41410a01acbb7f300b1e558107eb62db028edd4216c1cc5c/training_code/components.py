'''Morphology-scaled recovery reward with stability-gated damping.

The task dispatcher replaces upstream rewards and integrates these rates once
with control dt. This file preserves the reference upright, posture, action
rate, and fall mechanisms and replaces the unconditional velocity penalty with
a gated damping mechanism that relaxes damping during large recovery motion
and adds a low-height, low-speed collapse guard.
'''

from true_rl_agent.candidates.components import RewardTerm, TrainingComponents


def upright_height(inputs):
    tilt = inputs.roll.square() + inputs.pitch.square()
    height_error = (inputs.base_height_relative - 1.0).square()
    return (-4.0 * tilt - 4.0 * height_error).exp()


def recovery_damping(inputs):
    planar_speed = (inputs.base_lin_vel_scaled[:, :2].square().sum(dim=1) + 1e-6).sqrt()
    angular_speed = (inputs.base_ang_vel_scaled.square().sum(dim=1) + 1e-6).sqrt()
    tilt = inputs.roll.square() + inputs.pitch.square()
    height_error = inputs.base_height_relative - 1.0
    uprightness = (-2.0 * tilt).exp()
    height_ok = (-4.0 * height_error.square()).exp()
    stability_gate = 0.2 + 0.8 * uprightness * height_ok
    motion_cost = (planar_speed / 0.35).square() + (angular_speed / 1.0).square()
    damping_reward = 1.0 - 0.4 * stability_gate * motion_cost
    low_height = (0.9 - inputs.base_height_relative).clamp(min=0.0)
    low_speed_gate = (1.0 - planar_speed / 0.3).clamp(min=0.0, max=1.0)
    collapse_penalty = low_height * low_speed_gate
    return damping_reward - 4.0 * collapse_penalty


def joint_pose(inputs):
    return inputs.joint_pos_error.square().mean(dim=1)


def action_rate(inputs):
    return inputs.action_delta.square().mean(dim=1)


def fall(inputs):
    return inputs.terminated_non_timeout.float()


def build_training_components():
    return TrainingComponents(
        name='push_standing_recovery_gated_damping_v1',
        reward_terms=[
            RewardTerm(name='upright_height', weight=3.0, fn=upright_height),
            RewardTerm(name='recovery_damping', weight=1.0, fn=recovery_damping, clip_min=-1.5, clip_max=1.0),
            RewardTerm(name='joint_pose', weight=-0.1, fn=joint_pose),
            RewardTerm(name='action_rate', weight=-0.01, fn=action_rate),
            RewardTerm(name='fall', weight=-20.0, fn=fall),
        ],
        diagnostics=[],
        schedules=[],
        rationale='Reference candidate push_standing_recovery_stationary_reference_v3 uses stationary_v3, an unconditional velocity penalty that is strongest exactly when the robot needs corrective motion after large pushes. This replacement keeps every other reference reward mechanism unchanged and replaces stationary_v3 with recovery_damping: a stability-gated velocity damping term that scales damping down when tilt or height error is large, so recovery motion is not suppressed, and adds a low-height low-speed collapse guard that penalizes a quiescent crouched state after the push sequence. The mechanism targets the public high and repeated strata failures and the dominant height_collapse mode by allowing transient recovery motion while still shaping a tall, low-velocity stable dwell.',
    )
