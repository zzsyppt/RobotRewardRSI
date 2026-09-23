"""Complete, morphology-scaled standing recovery reward rates with dense fall-rate damping.

Stance is scored by a deadbanded, non-saturating potential over relative base
height and tilt, and the world-frame speed magnitudes keep their anchoring
penalty. One dense channel is added: a rate penalty on downward base velocity
and on the horizontal roll/pitch tipping rate, so the transient that ends in a
fall carries a price before the terminal fall term fires. Joint and action
penalties use means so changing joint count does not silently multiply their
weight. Perturbation schedules are not reward inputs.
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


def fall_rate_damping(inputs):
    sink = (0.05 - inputs.base_lin_vel_scaled[:, 2]).clamp(min=0.0)
    tip = (
        inputs.base_ang_vel_scaled[:, :2].square().sum(dim=1).sqrt() / 0.25 - 1.0
    ).clamp(min=0.0)
    return sink + sink.square() + tip + tip.square()


def fall(inputs):
    return inputs.terminated_non_timeout.float()


def build_training_components() -> TrainingComponents:
    return TrainingComponents(
        name="push_standing_recovery_rate_damped_v1",
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
            RewardTerm(
                name="fall_rate_damping",
                weight=-1.0,
                fn=fall_rate_damping,
                clip_min=0.0,
                clip_max=8.0,
                description="Deadbanded dense rate penalty on downward base velocity and horizontal tipping rate.",
            ),
            RewardTerm(name="fall", weight=-20.0, fn=fall),
        ],
        diagnostics=[],
        schedules=[],
        rationale=(
            "Closest prior program: push_recovery_nonsaturating_stance_v1, the accepted current best "
            "(public_dev mean 0.8802; outcome counts success_recovered 169, fall_side 15, not_recovered 4, "
            "height_collapse 3, fall_forward 1; strata low 47/48, medium 43/48, high 37/48, repeated 42/48). "
            "Sole mechanism change: one added RewardTerm, fall_rate_damping, weight -1.0, clip [0.0, 8.0]; "
            "every existing callable, weight, sign, clip and the empty diagnostics and schedules lists are "
            "preserved exactly. The anchor prices posture (relative base height and tilt, plus the deadbands, "
            "in upright_recovery) and world-frame speed magnitude (stationary_v3), so vertical base velocity "
            "never enters the program at all, and the only angular-rate channel is the yaw-mixing world-frame "
            "magnitude inside stationary_v3, whose effective 0.25 factor prices a 2.5 rad/s tipping rate at "
            "0.125 of output, which is nothing next to the -20 terminal fall term. The transient that actually "
            "produces the residual side, forward and low-height failures, the base dropping and rotating faster "
            "than the legs can replace support, therefore carries no dense price before termination. "
            "fall_rate_damping adds that missing channel on already-public inputs: "
            "sink = clamp(0.05 - base_lin_vel_scaled[:,2], min=0) prices downward world-z velocity beyond "
            "about 0.1 m/s, and tip = clamp(norm(base_ang_vel_scaled[:,:2]) / 0.25 - 1.0, min=0) prices the "
            "horizontal roll and pitch tipping rate beyond about 1.25 rad/s while deliberately leaving yaw "
            "unpriced, because spinning in place is not a fall. The cost sink + sink^2 + tip + tip^2 is "
            "identically zero in the near-goal stance where the low and medium strata already recover, grows "
            "smoothly with both the depth and the rate of the deviation, and its 8.0 cap is an outlier bound "
            "that does not bind in the operating range. Both channels are rate-gated rather than posture-gated, "
            "so a base that is displaced or tilted but slow keeps exactly the anchor signal; that is what "
            "distinguishes this from the generation-two height-times-tilt coupling and linear tilt ramp, which "
            "raised the price of being displaced at all and cost the high and repeated strata their recovery "
            "margin, and from the round-one gated planar damping, which only rescaled the existing planar term."
        ),
    )
