"""Complete, morphology-scaled push-recovery reward rates.

The task dispatcher replaces upstream rewards and integrates these rates once
with control dt. Joint penalties use means so changing joint count does not
silently multiply their weight. Perturbation schedules are not reward inputs.
The stance term keeps a non-vanishing slope far from the nominal stance, so a
deep post-push height deficit is never reward-free.
"""

from true_rl_agent.candidates.components import RewardTerm, TrainingComponents


def recovery_posture_margin(inputs):
    # Piecewise stance margin: a quadratic deadband around the nominal stance
    # plus a linear tail, so the recovery gradient never vanishes.
    # Height below nominal is steeper than height above nominal because
    # post-push collapse is the dominant observed failure.
    height_err = inputs.base_height_relative - 1.0
    below = (-height_err).clamp(min=0.0)
    above = height_err.clamp(min=0.0)
    below_head = below.clamp(max=0.08)
    above_head = above.clamp(max=0.12)
    below_cost = 0.5 * below_head.square() / 0.08 + (below - 0.08).clamp(min=0.0)
    above_cost = 0.5 * above_head.square() / 0.12 + (above - 0.12).clamp(min=0.0)
    # The same quadratic-then-linear shape on tilt keeps a slope at large lean.
    tilt = inputs.roll.square() + inputs.pitch.square()
    tilt_head = tilt.clamp(max=0.05)
    tilt_cost = 0.5 * tilt_head.square() / 0.05 + (tilt - 0.05).clamp(min=0.0)
    margin = 1.0 - 1.5 * below_cost - 0.75 * above_cost - 1.5 * tilt_cost
    return margin.clamp(min=-1.0, max=1.0)


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
        name="push_recovery_nonvanishing_stance_margin_v4",
        reward_terms=[
            RewardTerm(
                name="recovery_posture_margin",
                weight=3.0,
                fn=recovery_posture_margin,
                clip_min=-1.0,
                clip_max=1.0,
                description="Bounded stance margin with a quadratic deadband and a linear tail.",
            ),
            RewardTerm(name="stationary_v3", weight=1.0, fn=stationary_v3),
            RewardTerm(name="joint_pose", weight=-0.1, fn=joint_pose),
            RewardTerm(name="action_rate", weight=-0.01, fn=action_rate),
            RewardTerm(name="fall", weight=-20.0, fn=fall),
        ],
        diagnostics=[],
        schedules=[],
        rationale="Canonical hypothesis: the anchor stance reward exp(-4*(tilt^2 + height_err^2)) flattens once a strong push drives the base away from nominal height, so a collapsed stance is nearly reward-free and the climb back is not reinforced, which matches the public failure evidence where height_collapse is the largest failure cluster (26 of 51 non-recoveries) and the high and repeated push strata recover least often (19/48 and 27/48). This candidate changes exactly one mechanism, the reward callable that shapes stance and height: the saturating exponential is replaced by a bounded piecewise quadratic-then-linear margin, quadratic inside a small deadband around nominal height and nominal tilt and strictly linear beyond it, with a steeper coefficient for height below nominal than above, so the gradient toward a nominal stance never vanishes and a deeper crouch is always scored worse than a shallower one. Every other term (stationary_v3, joint_pose, action_rate, fall), every weight and the rest of the component set are preserved from the anchor. Expected effect: a directed, non-saturating pull back to nominal height after the last pulse turns height-collapse endings into fresh stable dwells and lifts success mainly in the high and repeated push strata without disturbing the already-solid mild strata.",
    )
