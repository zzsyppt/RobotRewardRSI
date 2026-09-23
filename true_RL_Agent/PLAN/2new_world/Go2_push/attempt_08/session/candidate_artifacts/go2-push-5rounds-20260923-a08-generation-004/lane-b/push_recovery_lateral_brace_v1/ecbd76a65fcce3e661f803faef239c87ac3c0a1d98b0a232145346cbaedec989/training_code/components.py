"""Morphology-scaled standing recovery reward rates with a lateral brace term.

Stance is scored by a deadbanded, non-saturating potential over relative base
height and tilt. A lateral brace term penalizes sideways tilt and lateral base
speed, targeting the dominant fall_side failures. Joint and action penalties
use means so changing joint count does not silently multiply their weight.
"""

from true_rl_agent.candidates.components import RewardTerm, TrainingComponents


def upright_recovery(inputs):
    height_excess = ((inputs.base_height_relative - 1.0).abs() - 0.05).clamp(min=0.0)
    tilt = inputs.roll.square() + inputs.pitch.square()
    tilt_excess = (tilt - 0.01).clamp(min=0.0)
    penalty = (
        5.0 * height_excess
        + height_excess.square()
        + 2.0 * tilt_excess
        + 2.0 * tilt_excess.square()
    )
    return 1.0 - penalty


def lateral_brace(inputs):
    roll_excess = (inputs.roll.abs() - 0.10).clamp(min=0.0)
    lateral_speed = inputs.base_lin_vel_scaled[:, 1].abs()
    return -(roll_excess + 0.5 * lateral_speed)


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


def build_training_components():
    return TrainingComponents(
        name="push_standing_recovery_lateral_brace_v1",
        reward_terms=[
            RewardTerm(
                name="upright_recovery",
                weight=3.0,
                fn=upright_recovery,
                clip_min=-4.0,
                clip_max=1.0,
                description="Deadbanded non-saturating stance potential over relative base height and tilt.",
            ),
            RewardTerm(
                name="lateral_brace",
                weight=-1.0,
                fn=lateral_brace,
                clip_min=-2.0,
                clip_max=0.0,
                description="Penalty on sideways tilt beyond a deadband and lateral base speed.",
            ),
            RewardTerm(name="stationary_v3", weight=1.0, fn=stationary_v3),
            RewardTerm(name="joint_pose", weight=-0.1, fn=joint_pose),
            RewardTerm(name="action_rate", weight=-0.01, fn=action_rate),
            RewardTerm(name="fall", weight=-20.0, fn=fall),
        ],
        diagnostics=[],
        schedules=[],
        rationale=(
            "Closest prior program: push_recovery_nonsaturating_stance_v1, the accepted current best "
            "(public_dev mean 0.8802). Its dominant failure is fall_dominated, with 15 fall_side versus "
            "1 fall_forward and 3 height_collapse, so sideways toppling is the residual bottleneck. "
            "Prior attempts push_recovery_tilt_aware_stance_v2 (tilt-height coupling) and "
            "push_recovery_vel_damped_stance_v1 (displacement-gated speed damping) were both formally "
            "rejected, so those routes are negative evidence and are not repeated. Sole mechanism "
            "change here: a new RewardTerm lateral_brace is added; upright_recovery, stationary_v3, "
            "joint_pose, action_rate, fall, all weights, signs, clips, and the empty schedule and "
            "diagnostic lists are preserved exactly. The anchor penalizes roll and pitch only through "
            "the symmetric squared tilt sum, which is dominated by whichever axis is larger and gives "
            "no dedicated lateral signal; a base that is tipping sideways can therefore accumulate "
            "roll and lateral drift at little extra cost. lateral_brace returns "
            "-(clamp(|roll| - 0.10, 0) + 0.5 * |v_y_scaled|), a one-sided lateral potential that is "
            "exactly zero inside a 0.10 rad roll deadband and grows linearly with sideways tilt and "
            "lateral base speed. It is asymmetric by construction: it never penalizes forward/backward "
            "motion or pitch, so it cannot suppress the sagittal recovery transient that the anchor "
            "already handles. The clip [-2, 0] with weight -1.0 bounds the weighted contribution, and "
            "the deadband keeps the nominal stance value identical to the anchor. Success definition, "
            "push schedule, and all runner-owned surfaces are untouched."
        ),
    )
