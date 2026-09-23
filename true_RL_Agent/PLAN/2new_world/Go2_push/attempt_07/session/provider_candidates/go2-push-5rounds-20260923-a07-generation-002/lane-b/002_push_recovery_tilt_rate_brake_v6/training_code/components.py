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


def tilt_rate_brake(inputs):
    tilt = inputs.roll.square() + inputs.pitch.square()
    tilt_rate = inputs.base_ang_vel_scaled[:, :2].square().sum(dim=1)
    return -(tilt_rate / (1.0 + 8.0 * tilt))


def joint_pose(inputs):
    return inputs.joint_pos_error.square().mean(dim=1)


def action_rate(inputs):
    return inputs.action_delta.square().mean(dim=1)


def fall(inputs):
    return inputs.terminated_non_timeout.float()


def build_training_components():
    return TrainingComponents(
        name="push_recovery_tilt_rate_brake_v6",
        reward_terms=[
            RewardTerm(name="upright_height", weight=3.0, fn=upright_height),
            RewardTerm(name="stationary_v3", weight=1.0, fn=stationary_v3),
            RewardTerm(name="tilt_damping", weight=1.0, fn=tilt_damping),
            RewardTerm(name="tilt_rate_brake", weight=1.0, fn=tilt_rate_brake),
            RewardTerm(name="joint_pose", weight=-0.1, fn=joint_pose),
            RewardTerm(name="action_rate", weight=-0.01, fn=action_rate),
            RewardTerm(name="fall", weight=-20.0, fn=fall),
        ],
        rationale=(
            "Closest prior candidate is the accepted current best push_recovery_tilt_damping_v5 "
            "(public_dev mean 0.8541666666666666; outcomes success_recovered 164, height_collapse 18, "
            "fall_side 5, fall_backward 3, not_recovered 2; low 48/48, medium 48/48, high 29/48, "
            "repeated 39/48). Its dominant residual failure mode is height_collapse, concentrated in "
            "the high and repeated pulse strata, so the remaining gap is the large-pulse recovery "
            "transient rather than the small-pulse regime. The v5 mechanism tilt_damping is the "
            "negative product of squared tilt magnitude and squared planar angular rate, so its "
            "magnitude is proportional to tilt: at small tilt it is nearly silent, and it only becomes "
            "a strong brake once the robot is already substantially tilted, which is late in a large "
            "pulse excursion. This candidate changes exactly one named canonical prompt-visible "
            "component mechanism relative to v5: it adds a single new reward callable, "
            "tilt_rate_brake, defined as the negative planar angular rate magnitude divided by "
            "(1 + 8 * tilt), weighted +1.0. Because the denominator is one at zero tilt and grows with "
            "tilt, this term is a strong angular-rate brake while the robot is still near upright and "
            "is progressively attenuated as tilt grows, so it supplies an early, tilt-gated damping "
            "gradient that arrests rotation before the excursion becomes a collapse, while never "
            "penalizing tilt itself, height, or action magnitude. It is non-positive and vanishes at "
            "zero angular rate, so it cannot reward motion and cannot conflict with the upright_height "
            "stance bonus. All other v5 mechanisms (upright_height, stationary_v3, tilt_damping, "
            "joint_pose, action_rate, fall) are preserved unchanged, and no diagnostics or schedules "
            "are added. Expected outcome: large-pulse excursions are braked earlier, so height_collapse "
            "shrinks and the high and repeated strata improve while the already-solved low and medium "
            "strata are preserved."
        ),
    )
