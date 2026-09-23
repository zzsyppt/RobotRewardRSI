"""Morphology-scaled standing recovery reward rates with a roll-dominant lateral brace.

Stance is scored by a deadbanded, non-saturating potential over relative base
height and tilt. The lateral brace penalizes sideways tilt and lateral base
speed, targeting the residual fall_side failures. Joint and action penalties
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


def roll_brace(inputs):
    roll_excess = (inputs.roll.abs() - 0.10).clamp(min=0.0)
    return -roll_excess.square()


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
        name="push_standing_recovery_roll_brace_v2",
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
            RewardTerm(
                name="roll_brace",
                weight=-1.0,
                fn=roll_brace,
                clip_min=-2.0,
                clip_max=0.0,
                description="Quadratic penalty on roll excess beyond a 0.10 rad deadband.",
            ),
            RewardTerm(name="stationary_v3", weight=1.0, fn=stationary_v3),
            RewardTerm(name="joint_pose", weight=-0.1, fn=joint_pose),
            RewardTerm(name="action_rate", weight=-0.01, fn=action_rate),
            RewardTerm(name="fall", weight=-20.0, fn=fall),
        ],
        diagnostics=[],
        schedules=[],
        rationale=(
            "Closest prior program: push_recovery_lateral_brace_v1, the accepted current best "
            "(public_dev mean 0.9115). Its dominant residual failure is fall_dominated with 10 "
            "fall_side and 7 not_recovered, and the high stratum is the weak point (37/48 vs 48/48 "
            "low and medium), so sideways toppling under strong pushes is the bottleneck. The "
            "sibling push_recovery_side_braced_stance_v1 (displacement-gated squared roll) was "
            "qualified_not_selected at 0.8385 with height_collapse_dominated, so gating the lateral "
            "term on displacement is negative evidence and is not repeated. Sole mechanism change "
            "here: a new RewardTerm roll_brace is added; upright_recovery, lateral_brace, "
            "stationary_v3, joint_pose, action_rate, fall, all their weights, signs, clips, and the "
            "empty schedule and diagnostic lists are preserved exactly. The anchor's lateral_brace "
            "is linear in |roll| beyond a 0.10 rad deadband, so its restoring slope is constant and "
            "weak once roll is large; a base tipping sideways under a strong push can accumulate "
            "large roll at only a linear cost while the height branch of upright_recovery dominates "
            "the gradient. roll_brace returns -clamp(|roll| - 0.10, 0, inf)^2, a quadratic lateral "
            "potential that is exactly zero inside the same 0.10 rad deadband and grows with the "
            "square of roll excess, so it adds restoring slope that increases with tilt magnitude "
            "exactly in the large-roll regime where the anchor's linear term is too weak. It is "
            "asymmetric by construction: it never penalizes pitch or forward/backward motion, so it "
            "cannot suppress the sagittal recovery transient. The clip [-2, 0] with weight -1.0 "
            "bounds the weighted contribution, and the shared deadband keeps the nominal stance "
            "value identical to the anchor. Success definition, push schedule, and all runner-owned "
            "surfaces are untouched."
        ),
    )
