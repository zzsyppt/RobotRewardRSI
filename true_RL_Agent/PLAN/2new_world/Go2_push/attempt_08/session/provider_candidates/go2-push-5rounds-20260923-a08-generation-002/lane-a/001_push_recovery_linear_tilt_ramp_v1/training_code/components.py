"""Complete, morphology-scaled standing recovery reward rates.

Stance is scored by a deadbanded, non-saturating potential over relative base
height and total tilt magnitude. The height branch is linear-dominant, and the
tilt branch keeps the same kernel shape but applies it to the tilt angle itself
so its pull-back slope does not vanish at small lean angles. Joint penalties use
means so changing the joint count does not silently multiply their weight.
Perturbation schedules are not reward inputs.
"""

from true_rl_agent.candidates.components import RewardTerm, TrainingComponents


def upright_recovery_linear_tilt(inputs):
    height_excess = ((inputs.base_height_relative - 1.0).abs() - 0.05).clamp(min=0.0)
    tilt_magnitude = (inputs.roll.square() + inputs.pitch.square() + 1e-6).sqrt()
    tilt_excess = (tilt_magnitude - 0.1).clamp(min=0.0)
    penalty = (
        5.0 * height_excess
        + height_excess.square()
        + 2.0 * tilt_excess
        + 2.0 * tilt_excess.square()
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
        name="push_standing_recovery_linear_tilt_ramp_v1",
        reward_terms=[
            RewardTerm(
                name="upright_recovery_linear_tilt",
                weight=3.0,
                fn=upright_recovery_linear_tilt,
                clip_min=-4.0,
                clip_max=1.0,
                description="Deadbanded non-saturating stance potential over relative base height and tilt magnitude.",
            ),
            RewardTerm(name="stationary_v3", weight=1.0, fn=stationary_v3),
            RewardTerm(name="joint_pose", weight=-0.1, fn=joint_pose),
            RewardTerm(name="action_rate", weight=-0.01, fn=action_rate),
            RewardTerm(name="fall", weight=-20.0, fn=fall),
        ],
        diagnostics=[],
        schedules=[],
        rationale=(
            "Closest prior program: push_recovery_nonsaturating_stance_v1, the accepted current-best "
            "anchor. Sole mechanism change: the tilt branch of the stance potential callable, renamed "
            "upright_recovery_linear_tilt; every other callable body, weight, sign, clip, and the empty "
            "schedule and diagnostic lists are preserved. The anchor scored tilt with "
            "u = clamp(roll*roll + pitch*pitch - 0.01, 0) and penalty 2*u + 2*u*u, so its tilt pull-back "
            "slope is 4*theta*(1 + 2*u) per radian: about 0.4 at the deadband edge and about 0.85 at 0.2 "
            "radians. That leaves the first 6 to 15 degrees of a lean almost free, exactly the window in "
            "which a lateral perturbation has to be arrested. The anchor's residual public-dev failures "
            "are dominated by sideways falls (15 fall_side against 1 fall_forward), consistent with a "
            "tilt branch that stays shallow until the lean has already run away, after which only the "
            "fall penalty remains. The replacement keeps the identical kernel shape and coefficients "
            "(2*x + 2*x*x) but applies them to the tilt-magnitude excess "
            "v = clamp(sqrt(roll*roll + pitch*pitch + 1e-6) - 0.1, 0) instead of to the squared-tilt "
            "excess u, which makes the tilt branch linear-dominant in the lean angle and mirrors the "
            "height branch the previous round already made linear-dominant for the same reason. The "
            "deadband is unchanged in value, since 0.1 radians is exactly the anchor's sqrt(0.01) tilt "
            "wall, and at 0.9 radians the two forms take the same value again, so the near-goal potential "
            "and the deep-lean magnitude are both preserved while the 0.1 to 0.9 radian ramp is strictly "
            "steeper. Concretely the slope becomes 2.0 + 4*v per radian, that is 2.0 versus 0.4 at the "
            "deadband edge and 2.4 versus 0.85 at 0.2 radians, and a 10 to 25 degree lean now costs about "
            "0.5 to 2.7 weighted stance reward per step instead of about 0.13 to 1.3. The term weight, "
            "clip bounds, and the posture, action-rate, velocity-damping and fall terms are untouched, so "
            "the low and medium force strata that already recover keep their existing signal while the "
            "policy receives a graded early pull-back gradient for the leans that dominate the remaining "
            "failures."
        ),
    )
