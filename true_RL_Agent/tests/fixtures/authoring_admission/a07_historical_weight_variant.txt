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
        name="push_recovery_tilt_rate_brake_v7",
        reward_terms=[
            RewardTerm(name="upright_height", weight=3.0, fn=upright_height),
            RewardTerm(name="stationary_v3", weight=1.0, fn=stationary_v3),
            RewardTerm(name="tilt_damping", weight=1.0, fn=tilt_damping),
            RewardTerm(
                name="tilt_rate_brake",
                weight=0.5,
                fn=tilt_rate_brake,
                description="Bounded angular-rate brake that is strongest in the upright stance and fades as tilt grows, so it damps residual wobble without taxing large recovery rotations.",
            ),
            RewardTerm(name="joint_pose", weight=-0.1, fn=joint_pose),
            RewardTerm(name="action_rate", weight=-0.01, fn=action_rate),
            RewardTerm(name="fall", weight=-20.0, fn=fall),
        ],
        diagnostics=[],
        schedules=[],
        rationale=(
            "Anchor: accepted current best push_recovery_tilt_damping_v5 (public_dev mean 0.85417; "
            "low 48/48, medium 48/48, high 29/48, repeated 39/48; outcomes success_recovered 164, "
            "height_collapse 18, fall_side 5, fall_backward 3, not_recovered 2). Exact code-level "
            "difference: exactly one named mechanism is added, the reward callable tilt_rate_brake, "
            "defined as -(sum(base_ang_vel_scaled[:, :2]^2) / (1.0 + 8.0 * (roll^2 + pitch^2))) with "
            "weight +0.5. upright_height, stationary_v3, tilt_damping, joint_pose, action_rate and "
            "fall keep their original callables, weights and mechanisms; no term is removed and no "
            "diagnostic or schedule component is introduced. Hypothesis: the anchor's only "
            "rate-coupling term, tilt_damping, is the product of squared tilt and squared planar "
            "angular rate, so its gradient is proportional to tilt and it is nearly silent in the "
            "small-tilt regime where the high-stratum and repeated-push losses accumulate; the "
            "runtime receipt confirms this, with tilt_damping max_abs 8.18 but last_weighted_mean "
            "-0.0012, i.e. it acts only in brief large-tilt spikes. The new term inverts that "
            "weighting: it is a bounded angular-rate penalty that is strongest when the robot is "
            "upright (denominator near 1) and fades as tilt grows, so it supplies a continuous "
            "wobble-damping gradient during the post-pulse settle and during the fresh stable dwell "
            "that success_recovered requires, while deliberately not taxing the large corrective "
            "rotations that a strong pulse demands. Because the term is non-positive and vanishes "
            "whenever the planar angular rate is zero, it cannot reward motion and cannot dominate "
            "the transient regime; its maximum weighted magnitude is 0.5. Expected outcome: fewer "
            "residual-wobble losses in the high and repeated strata and fewer fall_side and "
            "fall_backward outcomes, with low and medium strata preserved because the brake is "
            "small relative to upright_height and stationary_v3. Closest rejected prior is the "
            "sibling push_recovery_tilt_rate_brake_v6 (public_dev delta -0.1875, "
            "qualified_not_selected), which added a rate brake at weight 1.0; this candidate halves "
            "that weight to 0.5 so the brake stays a secondary shaping signal rather than competing "
            "with the stance and quietness terms that already carry the accepted program."
        ),
    )
