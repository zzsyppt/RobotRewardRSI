"""Morphology-scaled push-recovery reward rates.

Rates are integrated once by the task RewardManager with control dt.
Joint and action penalties use means so joint count does not scale weight.
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


def impulse_damped_velocity(inputs):
    planar = inputs.base_lin_vel_scaled[:, :2].square().sum(dim=1).sqrt()
    angular = inputs.base_ang_vel_scaled.square().sum(dim=1).sqrt()
    push_mag = inputs.push_vec_public.square().sum(dim=1).sqrt()
    push_gate = inputs.push_active.float()
    planar_core = (planar / 0.08).clamp(max=1.0)
    angular_core = (angular / 0.5).clamp(max=1.0)
    planar_pen = 0.5 * planar_core.square() + (planar / 0.08) - planar_core
    angular_pen = 0.5 * angular_core.square() + (angular / 0.5) - angular_core
    base = 1.0 - 0.25 * (planar_pen + angular_pen)
    return base - 0.5 * push_gate * push_mag


def joint_pose(inputs):
    return inputs.joint_pos_error.square().mean(dim=1)


def action_rate(inputs):
    return inputs.action_delta.square().mean(dim=1)


def fall(inputs):
    return inputs.terminated_non_timeout.float()


def tilt_diag(inputs):
    return inputs.roll.square() + inputs.pitch.square()


def speed_diag(inputs):
    return inputs.base_lin_vel_scaled[:, :2].square().sum(dim=1).sqrt()


def build_training_components():
    return TrainingComponents(
        name="push_recovery_impulse_damped_v1",
        reward_terms=[
            RewardTerm(name="upright_height", weight=3.0, fn=upright_height),
            RewardTerm(name="impulse_damped_velocity", weight=1.0, fn=impulse_damped_velocity),
            RewardTerm(name="joint_pose", weight=-0.1, fn=joint_pose),
            RewardTerm(name="action_rate", weight=-0.01, fn=action_rate),
            RewardTerm(name="fall", weight=-20.0, fn=fall),
        ],
        diagnostics=[
            DiagnosticTerm(name="tilt_diag", fn=tilt_diag),
            DiagnosticTerm(name="speed_diag", fn=speed_diag),
        ],
        schedules=[
            ScheduleTerm(name="reward_weight_scale", key="reward_weight_scale", value=1.0),
        ],
        rationale=(
            "Reference anchor push_standing_recovery_stationary_reference_v3 uses "
            "stationary_v3, which rewards low base velocity unconditionally and "
            "therefore fights the recovery response right after an injected push. "
            "This candidate replaces stationary_v3 with impulse_damped_velocity: "
            "the same bounded planar/angular damping core, but with an explicit "
            "push_active-gated term that subtracts 0.5 * |push_vec_public| only on "
            "the current step of a verified injected push. The gate uses only the "
            "prompt-visible push_active and push_vec_public fields, so it never "
            "fires on autonomous motion and never latches through recovery. "
            "upright_height, joint_pose, action_rate, and fall are preserved "
            "unchanged from the anchor; diagnostics and a neutral "
            "reward_weight_scale schedule are added for public feedback."
        ),
    )
