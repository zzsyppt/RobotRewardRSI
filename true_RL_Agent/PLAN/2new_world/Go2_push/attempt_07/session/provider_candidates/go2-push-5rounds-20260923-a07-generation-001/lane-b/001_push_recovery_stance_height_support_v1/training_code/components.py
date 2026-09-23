"""Complete, morphology-scaled standing recovery reward rates.

Five mechanisms of the verified standing-recovery reference are preserved with
their original callables and weights: an upright and height bonus, a damping
term, a joint-posture penalty, an action-rate penalty and a fall penalty. One
mechanism is added: a one-sided, morphology-normalized stance-height support
penalty that grows smoothly with stance-height deficit so a crouched
equilibrium is never cheaper than standing tall at nominal height. All
callables return reward rates; the task integrator applies control dt once.
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


def joint_pose(inputs):
    return inputs.joint_pos_error.square().mean(dim=1)


def action_rate(inputs):
    return inputs.action_delta.square().mean(dim=1)


def fall(inputs):
    return inputs.terminated_non_timeout.float()


def stance_height_support(inputs):
    deficit = (1.0 - inputs.base_height_relative).clamp(min=0.0)
    scaled = deficit / 0.30
    return -(scaled * scaled)


def build_training_components():
    return TrainingComponents(
        name="push_recovery_stance_height_support_v1",
        reward_terms=[
            RewardTerm(name="upright_height", weight=3.0, fn=upright_height),
            RewardTerm(name="stationary_v3", weight=1.0, fn=stationary_v3),
            RewardTerm(name="joint_pose", weight=-0.1, fn=joint_pose),
            RewardTerm(name="action_rate", weight=-0.01, fn=action_rate),
            RewardTerm(name="fall", weight=-20.0, fn=fall),
            RewardTerm(
                name="stance_height_support",
                weight=3.0,
                fn=stance_height_support,
                description="One-sided quadratic penalty on stance-height deficit normalized by 0.30 of nominal height.",
            ),
        ],
        diagnostics=[],
        schedules=[],
        rationale="Anchor: verified reference push_standing_recovery_stationary_reference_v3 (public_dev mean 0.734375; outcomes success_recovered 141, height_collapse 26, fall_side 13, fall_backward 11, not_recovered 1; low 48/48, medium 47/48, high 19/48, repeated 27/48). Exact code-level difference: one RewardTerm is added, stance_height_support with weight 3.0 and callable -(clamp(1.0 - base_height_relative, min=0.0) / 0.30)**2, which reads only the current-step base_height_relative public input; all five reference RewardTerms keep their original callables, weights and mechanisms and no other component is added or altered. Hypothesis: the reference pays a bounded saturating exponential stance bonus with maximum 3.0 rate while its damping mechanism rewards near-zero planar and angular speed unconditionally, so absorbing a force pulse in a crouched low-motion posture costs only about 0.44 rate units of that bonus at 80 percent height; the public failure labels show that equilibrium as height_collapse, and the same unconditional speed reward also suppresses the fast postural corrections that large pulses need, consistent with high 19/48 and repeated 27/48. The added term is zero at and above nominal stance height and grows smoothly with morphology-normalized height deficit, calibrated so that a 0.30 deficit contributes -3.0 rate, equal to the entire maximum stance bonus, which makes a crouched manifold strictly worse than nominal stance at every rated deficit while never penalizing upright motion, tilt or action magnitude during a recovery transient. Expected outcome: after the final pulse the policy returns to nominal stance height instead of settling low, so height_collapse outcomes shrink and the high and repeated pulse strata improve while the already-solved low and medium strata are preserved.",
    )
