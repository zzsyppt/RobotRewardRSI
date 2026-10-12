"""Morphology-scaled push-recovery reward rates.

The task dispatcher replaces upstream rewards and integrates these rates once
with control dt. Joint penalties use means so changing joint count does not
silently multiply their weight. Perturbation schedules are not reward inputs.
"""

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


def tilt_damping(inputs):
    tilt = inputs.roll.square() + inputs.pitch.square()
    tilt_rate = inputs.base_ang_vel_scaled[:, :2].square().sum(dim=1)
    return -(tilt * tilt_rate)


def joint_pose(inputs):
    return inputs.joint_pos_error.square().mean(dim=1)


def action_rate(inputs):
    return inputs.action_delta.square().mean(dim=1)


def fall(inputs):
    return inputs.terminated_non_timeout.float()


def build_training_components():
    return TrainingComponents(
        name="push_recovery_tilt_damping_v5",
        reward_terms=[
            RewardTerm(name="upright_height", weight=3.0, fn=upright_height),
            RewardTerm(name="stationary_v3", weight=1.0, fn=stationary_v3),
            RewardTerm(name="tilt_damping", weight=1.0, fn=tilt_damping),
            RewardTerm(name="joint_pose", weight=-0.1, fn=joint_pose),
            RewardTerm(name="action_rate", weight=-0.01, fn=action_rate),
            RewardTerm(name="fall", weight=-20.0, fn=fall),
        ],
        rationale=(
            "Closest prior candidate is the reference push_standing_recovery_stationary_reference_v3 "
            "(public_dev mean 0.734375), whose dominant failure mode is height_collapse with "
            "high-stratum and repeated-push losses. The reference upright_height term rewards being "
            "near nominal height and upright through a bounded exponential, but it is a pure state "
            "potential: it provides no gradient that couples tilt magnitude to angular rate, so a robot "
            "that is already tilting while rotating has no explicit incentive to arrest the rotation "
            "before the tilt grows into a fall. This candidate changes exactly one named canonical "
            "prompt-visible component mechanism relative to the reference: it adds a single new reward "
            "callable, tilt_damping, defined as the negative product of the squared tilt magnitude "
            "(roll^2 + pitch^2) and the squared planar angular rate (base_ang_vel_scaled[:, :2]^2 "
            "summed), weighted +1.0. Because the term is non-positive and vanishes whenever either "
            "tilt or angular rate is zero, it penalizes only the joint condition of being tilted while "
            "rotating, supplying a recovery gradient that pushes the policy to kill angular velocity "
            "during the tilt excursion rather than only to return to nominal height. All other reference "
            "mechanisms (upright_height, stationary_v3, joint_pose, action_rate, fall) are preserved "
            "unchanged, and no diagnostics or schedules are added."
        ),
    )
