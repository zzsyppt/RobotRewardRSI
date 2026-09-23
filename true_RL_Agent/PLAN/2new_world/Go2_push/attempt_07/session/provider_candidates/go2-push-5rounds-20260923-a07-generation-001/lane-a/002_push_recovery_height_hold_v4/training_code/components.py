"""Morphology-scaled push-recovery reward rates.

Task dispatcher replaces upstream rewards and integrates these rates once with
control dt. Joint penalties use means so joint count does not scale weights.
"""

from true_rl_agent.candidates.components import (
    DiagnosticTerm,
    RewardTerm,
    ScheduleTerm,
    TrainingComponents,
)


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


def height_hold(inputs):
    deficit = (1.0 - inputs.base_height_relative).clamp(min=0.0)
    return -deficit.square()


def joint_pose(inputs):
    return inputs.joint_pos_error.square().mean(dim=1)


def action_rate(inputs):
    return inputs.action_delta.square().mean(dim=1)


def fall(inputs):
    return inputs.terminated_non_timeout.float()


def height_diag(inputs):
    return inputs.base_height_relative.mean()


def build_training_components():
    return TrainingComponents(
        name="push_recovery_height_hold_v4",
        reward_terms=[
            RewardTerm(name="upright_height", weight=3.0, fn=upright_height),
            RewardTerm(name="stationary_v3", weight=1.0, fn=stationary_v3),
            RewardTerm(name="height_hold", weight=2.0, fn=height_hold),
            RewardTerm(name="joint_pose", weight=-0.1, fn=joint_pose),
            RewardTerm(name="action_rate", weight=-0.01, fn=action_rate),
            RewardTerm(name="fall", weight=-20.0, fn=fall),
        ],
        diagnostics=[DiagnosticTerm(name="height_diag", fn=height_diag)],
        schedules=[ScheduleTerm(name="reward_weight_scale", key="reward_weight_scale", value=1.0)],
        rationale=(
            "Closest prior candidate is the reference push_standing_recovery_stationary_reference_v3 "
            "(public_dev mean 0.734375), whose dominant failure mode is height_collapse (26/192) with "
            "high-stratum and repeated-push losses. The reference upright_height term only rewards being "
            "near nominal height through a bounded exponential, so once the base drops the gradient "
            "vanishes and the policy is not pushed back up. This candidate adds exactly one new reward "
            "callable, height_hold, a one-sided quadratic penalty on the normalized height deficit "
            "(1 - base_height_relative) clamped at zero, weighted +2.0, which supplies a persistent "
            "downward-height gradient that does not saturate. All other reference mechanisms "
            "(upright_height, stationary_v3, joint_pose, action_rate, fall) are preserved unchanged, and "
            "a height_diag diagnostic plus a neutral reward_weight_scale schedule are added for feedback."
        ),
    )
