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


def tilt_brake(inputs):
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
        name="push_recovery_tilt_brake_v7",
        reward_terms=[
            RewardTerm(name="upright_height", weight=3.0, fn=upright_height),
            RewardTerm(name="stationary_v3", weight=1.0, fn=stationary_v3),
            RewardTerm(name="tilt_damping", weight=1.0, fn=tilt_damping),
            RewardTerm(name="tilt_brake", weight=1.0, fn=tilt_brake),
            RewardTerm(name="joint_pose", weight=-0.1, fn=joint_pose),
            RewardTerm(name="action_rate", weight=-0.01, fn=action_rate),
            RewardTerm(name="fall", weight=-20.0, fn=fall),
        ],
        rationale=(
            "Anchor: accepted current best push_recovery_tilt_damping_v5 (public_dev mean 0.85417; "
            "low 48/48, medium 48/48, high 29/48, repeated 39/48; outcomes success_recovered 164, "
            "height_collapse 18, fall_side 5, fall_backward 3, not_recovered 2). Exact code-level "
            "difference: exactly one named mechanism is added, a new reward callable tilt_brake "
            "defined as -(sum of squared planar angular rate) / (1 + 8 * (roll^2 + pitch^2)), "
            "weighted +1.0. All other anchor mechanisms (upright_height, stationary_v3, "
            "tilt_damping, joint_pose, action_rate, fall) keep their original callables, weights "
            "and mechanisms; no term is removed and no diagnostic or schedule is introduced. "
            "Hypothesis: the anchor's tilt_damping term is the product of squared tilt and squared "
            "angular rate, so its gradient vanishes quadratically as tilt approaches zero; a robot "
            "that has just been pushed and is still nearly upright receives almost no explicit "
            "incentive to arrest the angular velocity that will grow into the tilt excursion, which "
            "is where the high-stratum and repeated-push losses and the side/backward falls sit. "
            "The new term is a tilt-normalized angular-rate brake: it is non-positive, vanishes only "
            "when the planar angular rate is zero, and its magnitude is largest at small tilt, so it "
            "supplies a recovery gradient exactly in the near-upright phase where the anchor's "
            "product term is silent, while the denominator 1 + 8 * tilt attenuates it once the robot "
            "is already tilted so it does not fight the large postural corrections that strong "
            "pulses require. Because the term is bounded by the angular rate itself and is "
            "attenuated by tilt, it cannot dominate the transient regime the way a fixed quadratic "
            "penalty would. Expected outcome: fewer tilt-phase recoveries lost to unchecked angular "
            "velocity, so high and repeated strata improve while low and medium strata are preserved "
            "because a quiet upright stance already has near-zero angular rate and pays almost "
            "nothing. Closest rejected prior is the sibling push_recovery_tilt_rate_brake_v6 "
            "(public_dev delta -0.1875, qualified_not_selected), which added a tilt_rate_brake "
            "term; this candidate differs at the code level by normalizing the squared planar "
            "angular rate by 1 + 8 * tilt instead of using the raw rate, so the brake is strongest "
            "near upright and fades as tilt grows, rather than applying uniformly across the whole "
            "tilt range."
        ),
    )
