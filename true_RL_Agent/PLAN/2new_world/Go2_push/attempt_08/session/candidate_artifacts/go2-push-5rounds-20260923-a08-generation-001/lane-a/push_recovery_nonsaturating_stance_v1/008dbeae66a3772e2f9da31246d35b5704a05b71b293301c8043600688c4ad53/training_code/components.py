"""Complete, morphology-scaled standing recovery reward rates.

Stance is scored by a deadbanded, non-saturating potential: outside a small
tolerance around the nominal stance the penalty keeps growing, so a base that
has sunk or tilted far from the nominal pose keeps a strong pull-up slope.
Joint penalties use means so changing joint count does not silently multiply
their weight. Perturbation schedules are not reward inputs.
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
        name="push_standing_recovery_nonsaturating_stance_v1",
        reward_terms=[
            RewardTerm(
                name="upright_recovery",
                weight=3.0,
                fn=upright_recovery,
                clip_min=-4.0,
                clip_max=1.0,
                description="Deadbanded non-saturating stance potential over relative base height and tilt.",
            ),
            RewardTerm(name="stationary_v3", weight=1.0, fn=stationary_v3),
            RewardTerm(name="joint_pose", weight=-0.1, fn=joint_pose),
            RewardTerm(name="action_rate", weight=-0.01, fn=action_rate),
            RewardTerm(name="fall", weight=-20.0, fn=fall),
        ],
        diagnostics=[],
        schedules=[],
        rationale=(
            "Closest prior program: push_standing_recovery_stationary_reference_v3, the current-best "
            "reference anchor. Sole mechanism change: the upright and height reward kernel; every other "
            "term, weight, sign, and the empty schedule and diagnostic lists are preserved. The anchor "
            "scored stance with a saturating Gaussian well over squared tilt and squared relative-height "
            "error at weight 3.0, so its shaping slope decays as the base sinks and a base held at 40-70 "
            "percent height loss still collected nearly the full 1.0-weighted stationary term. The "
            "observed height_collapse_dominated outcome is exactly that shallow optimum: staying low is "
            "cheap and the pull-up slope is too weak to escape. The replacement callable upright_recovery "
            "keeps weight 3.0 and returns 1.0 - (5*e + e*e + 2*u + 2*u*u), where "
            "e = clamp(abs(base_height_relative - 1) - 0.05, 0) and "
            "u = clamp(roll*roll + pitch*pitch - 0.01, 0), with clip_min -4.0 and clip_max 1.0. The "
            "height branch is linear-dominant, so unlike the anchor it never flattens: its slope stays "
            "about 5 per unit of relative height (about 15 per step at weight 3.0) at any depth, and a "
            "base at 30-50 percent relative height now earns roughly -6 to -10 per step instead of about "
            "1, so any upward progress is strictly better than lying low at every height. The 0.05 "
            "height and 0.01 tilt-squared deadbands keep the near-goal value identical to the anchor, so "
            "the low and medium force strata that already recover keep the same signal, while the tilt "
            "branch becomes steeper than the anchor beyond about 30 degrees to also suppress the "
            "residual backward and side falls."
        ),
    )
