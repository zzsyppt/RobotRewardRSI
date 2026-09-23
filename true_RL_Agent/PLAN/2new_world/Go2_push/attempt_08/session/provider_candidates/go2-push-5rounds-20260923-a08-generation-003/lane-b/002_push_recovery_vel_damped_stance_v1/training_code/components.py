"""Morphology-scaled standing recovery reward rates.

Stance is scored by a deadbanded, non-saturating potential over relative base
height and tilt. A separate recovery-damping term penalizes scaled base
velocity only while the base is displaced from the nominal stance, so it
suppresses residual drift and side falls without blocking the recovery
transient. Joint and action penalties use means so changing joint count does
not silently multiply their weight.
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


def recovery_damping(inputs):
    height_excess = ((inputs.base_height_relative - 1.0).abs() - 0.05).clamp(min=0.0)
    tilt = inputs.roll.square() + inputs.pitch.square()
    tilt_excess = (tilt - 0.01).clamp(min=0.0)
    displacement = height_excess + tilt_excess
    gate = (displacement / (displacement + 0.25)).clamp(max=1.0)
    planar_speed = inputs.base_lin_vel_scaled[:, :2].square().sum(dim=1).sqrt()
    angular_speed = inputs.base_ang_vel_scaled.square().sum(dim=1).sqrt()
    return -gate * (planar_speed + 0.5 * angular_speed)


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
        name="push_standing_recovery_vel_damped_stance_v1",
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
                name="recovery_damping",
                weight=-0.5,
                fn=recovery_damping,
                clip_min=-2.0,
                clip_max=0.0,
                description="Displacement-gated penalty on scaled base linear and angular speed.",
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
            "(public_dev mean 0.8802, dominant failure fall_dominated with 15 fall_side and 1 "
            "fall_forward). The sibling attempt push_recovery_tilt_aware_stance_v2 added a 4*e*u "
            "coupling to the same kernel and was formally rejected at 0.5521, so the coupling route is "
            "negative evidence and is not repeated. Sole mechanism change here: a new RewardTerm "
            "recovery_damping is added; the upright_recovery kernel, stationary_v3, joint_pose, "
            "action_rate, fall, all their weights, signs, clips, and the empty schedule and diagnostic "
            "lists are preserved exactly. The anchor shapes only posture (height and tilt) and a "
            "saturating stationary term, so a base that is already displaced can keep a large residual "
            "planar or angular speed at no extra cost; the residual side falls are consistent with that "
            "unpenalized drift. recovery_damping returns "
            "-g * (|v_xy|_scaled + 0.5 * |w|_scaled) with "
            "g = clamp((e + u) / (e + u + 0.25), 0, 1), where e and u are the same deadbanded height "
            "and tilt excesses used by the stance kernel. The gate is near zero at the nominal stance "
            "and approaches one when the base is far from it, so the term is inactive when the robot is "
            "already standing and grows with displacement, penalizing residual drift exactly in the "
            "regime where the anchor's posture-only shaping is weakest. Because the gate is smooth and "
            "bounded, the term never blocks the fast recovery transient: at small displacement the "
            "penalty is a small fraction of the speed, and the clip [-2, 0] keeps the weighted "
            "contribution bounded at weight -0.5. The success definition, push schedule, and all "
            "runner-owned surfaces are untouched."
        ),
    )
