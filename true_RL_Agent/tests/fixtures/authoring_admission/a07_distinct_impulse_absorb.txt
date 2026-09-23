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


def impulse_absorb(inputs):
    tilt = inputs.roll.square() + inputs.pitch.square()
    height_deficit = (1.0 - inputs.base_height_relative).clamp(min=0.0)
    excursion = tilt + 4.0 * height_deficit.square()
    return -(excursion / (1.0 + 4.0 * excursion))


def joint_pose(inputs):
    return inputs.joint_pos_error.square().mean(dim=1)


def action_rate(inputs):
    return inputs.action_delta.square().mean(dim=1)


def fall(inputs):
    return inputs.terminated_non_timeout.float()


def build_training_components():
    return TrainingComponents(
        name="push_recovery_impulse_absorb_v8",
        reward_terms=[
            RewardTerm(name="upright_height", weight=3.0, fn=upright_height),
            RewardTerm(name="stationary_v3", weight=1.0, fn=stationary_v3),
            RewardTerm(name="tilt_damping", weight=1.0, fn=tilt_damping),
            RewardTerm(name="impulse_absorb", weight=1.0, fn=impulse_absorb),
            RewardTerm(name="joint_pose", weight=-0.1, fn=joint_pose),
            RewardTerm(name="action_rate", weight=-0.01, fn=action_rate),
            RewardTerm(name="fall", weight=-20.0, fn=fall),
        ],
        rationale=(
            "Anchor: accepted current best push_recovery_tilt_damping_v5 (public_dev mean 0.85417; "
            "low 48/48, medium 48/48, high 29/48, repeated 39/48; outcomes success_recovered 164, "
            "height_collapse 18, fall_side 5, fall_backward 3, not_recovered 2). Exact code-level "
            "difference: exactly one named mechanism is added, a new reward callable impulse_absorb "
            "defined as -(E / (1 + 4*E)) with E = (roll^2 + pitch^2) + 4*clamp(1 - base_height_relative, 0)^2, "
            "weighted +1.0. All other anchor mechanisms (upright_height, stationary_v3, tilt_damping, "
            "joint_pose, action_rate, fall) keep their original callables, weights and mechanisms; no "
            "term is removed and no diagnostic or schedule is introduced. Hypothesis: the anchor's "
            "upright_height term is a bounded exponential of the same excursion, so its gradient is "
            "proportional to its own value and collapses toward zero exactly in the deep-tilt, "
            "low-height region where the labelled height_collapse and fall outcomes live; the anchor "
            "therefore has almost no restoring gradient once the robot is already far from nominal, "
            "which matches the high-stratum and repeated-push losses. The new term is a saturating "
            "excursion penalty whose gradient is largest at small excursion and decays only as "
            "1/(1 + 4*E)^2, so it keeps supplying a bounded, non-vanishing restoring force deep into "
            "the excursion instead of fading like the exponential. Because the term is non-positive, "
            "bounded in [-1, 0], and vanishes only at the nominal upright stance, it cannot dominate "
            "the transient regime the way an unbounded quadratic penalty would, and a quiet upright "
            "stance pays essentially nothing. Expected outcome: fewer recoveries lost to unchecked "
            "excursion growth, so high and repeated strata improve while low and medium strata are "
            "preserved because the term is near zero in the tall upright stance those strata already "
            "hold. Closest rejected prior is the sibling push_recovery_tilt_brake_v7 (public_dev delta "
            "-0.1875, historical_exact_program_repeat), which added a tilt-normalized angular-rate "
            "brake -(rate)/(1 + 8*tilt); this candidate differs at the code level by penalizing the "
            "state excursion itself (tilt plus a one-sided height deficit) with a saturating shape "
            "rather than penalizing angular rate, so it acts on the postural error the exponential "
            "stance bonus under-weights instead of on the rate that drives it."
        ),
    )
