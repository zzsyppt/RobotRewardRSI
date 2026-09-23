"""Morphology-scaled standing recovery reward rates.

Stance is scored by a deadbanded, non-saturating potential over relative base
height and tilt, with a tilt-aware coupling that strengthens the pull-up slope
when the base is both low and tilted. Joint and action penalties use means so
changing joint count does not silently multiply their weight.
"""

from true_rl_agent.candidates.components import RewardTerm, TrainingComponents


def upright_recovery(inputs):
    height_excess = ((inputs.base_height_relative - 1.0).abs() - 0.05).clamp(min=0.0)
    tilt = inputs.roll.square() + inputs.pitch.square()
    tilt_excess = (tilt - 0.01).clamp(min=0.0)
    coupling = height_excess * tilt_excess
    penalty = (
        5.0 * height_excess
        + height_excess.square()
        + 2.0 * tilt_excess
        + 2.0 * tilt_excess.square()
        + 4.0 * coupling
    )
    return 1.0 - penalty


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
        name="push_standing_recovery_tilt_aware_stance_v2",
        reward_terms=[
            RewardTerm(
                name="upright_recovery",
                weight=3.0,
                fn=upright_recovery,
                clip_min=-4.0,
                clip_max=1.0,
                description="Deadbanded non-saturating stance potential with tilt-aware coupling.",
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
            "fall_forward). Sole mechanism change: the upright_recovery kernel gains a tilt-aware "
            "coupling term; every other term, weight, sign, clip, and the empty schedule and "
            "diagnostic lists are preserved. The anchor's kernel is "
            "1 - (5*e + e*e + 2*u + 2*u*u) with e = clamp(abs(base_height_relative - 1) - 0.05, 0) "
            "and u = clamp(roll*roll + pitch*pitch - 0.01, 0). Its height and tilt branches are "
            "additive and independent, so a base that is simultaneously low and tilted receives the "
            "same slope as a base that is only low or only tilted; the residual side falls occur "
            "when the base is both sunk and rolled, where the anchor's pull-up slope is weakest "
            "relative to the destabilizing torque. The replacement adds 4*e*u to the penalty, so the "
            "gradient with respect to height grows with tilt and the gradient with respect to tilt "
            "grows with height. Near the goal (e = 0 or u = 0) the coupling vanishes and the value is "
            "identical to the anchor, so the low and medium strata that already recover keep the same "
            "signal. Far from the goal the coupling adds up to about 4 per unit of e*u, which at "
            "weight 3.0 roughly doubles the effective pull-up slope for a base that is both sunk and "
            "rolled, directly targeting the fall_side mode without changing the success definition."
        ),
    )
