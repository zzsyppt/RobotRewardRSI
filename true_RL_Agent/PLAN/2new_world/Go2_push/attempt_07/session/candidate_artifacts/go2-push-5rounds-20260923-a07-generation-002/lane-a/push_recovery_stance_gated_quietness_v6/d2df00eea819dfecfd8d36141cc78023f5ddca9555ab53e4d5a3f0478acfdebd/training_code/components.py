"""Stance-gated quietness for morphology-scaled push-recovery reward rates.

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
    quiet = 1.0 - 0.25 * (planar_penalty + angular_penalty)
    tilt = inputs.roll.square() + inputs.pitch.square()
    height_deficit = (1.0 - inputs.base_height_relative).clamp(min=0.0)
    stance_gate = (-4.0 * tilt - 12.0 * height_deficit.square()).exp()
    return stance_gate * quiet


def tilt_damping(inputs):
    tilt = inputs.roll.square() + inputs.pitch.square()
    tilt_rate = inputs.base_ang_vel_scaled[:, :2].square().sum(dim=1)
    return -(tilt * tilt_rate)


def joint_pose(inputs):
    return inputs.joint_pos_error.square().mean(dim=1)


def action_rate(inputs):
    return inputs.action_delta.square().mean(dim=1)


def fall(inputs):
    return inputs.terminated_non_timeout.float()


def build_training_components():
    return TrainingComponents(
        name="push_recovery_stance_gated_quietness_v6",
        reward_terms=[
            RewardTerm(name="upright_height", weight=3.0, fn=upright_height),
            RewardTerm(
                name="stationary_v3",
                weight=1.0,
                fn=stationary_v3,
                description="Quietness bonus multiplied by a bounded stance gate over tilt and height deficit; paid in full only for a tall, upright stance.",
            ),
            RewardTerm(name="tilt_damping", weight=1.0, fn=tilt_damping),
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
            "difference: exactly one named mechanism changes, the bounded quietness reward "
            "stationary_v3; its planar and angular speed core and its weight of 1.0 are unchanged, but "
            "its output is now multiplied by a stance gate "
            "exp(-4.0 * (roll^2 + pitch^2) - 12.0 * clamp(1.0 - base_height_relative, 0.0)^2) computed "
            "from the same public inputs, so the bonus is paid in full only for a tall, upright "
            "stance. upright_height, tilt_damping, joint_pose, action_rate and fall keep their "
            "original callables, weights and mechanisms; no term is added or removed and no "
            "diagnostic or schedule component is introduced. Hypothesis: the anchor pays up to +1.0 "
            "rate for any low-motion state, so a crouched post-push equilibrium collects nearly the "
            "whole quietness bonus while giving up only about 0.44 rate of the exponential stance "
            "bonus at 0.8 relative height; that cheap crouch attractor matches the labelled "
            "height_collapse outcomes, and the same unconditional damping also taxes the fast "
            "postural corrections that the strongest pulses need exactly while the robot is still "
            "tilted, which is where the side and backward falls and the not_recovered outcomes sit. "
            "Multiplying the quietness bonus by a bounded stance gate removes the crouch discount "
            "without adding any penalty: at 0.8 relative height the bonus falls from 1.0 to about "
            "0.62 and the standing restoring force rises from about 4.1 to about 7.1 rate units per "
            "unit of relative height, while at 0.5 rad of tilt the damping pressure is cut to about "
            "0.37 so recovery motion during a strong pulse is not taxed. Expected outcome: fewer "
            "crouch settles after the last pulse and fewer tilt-phase recoveries lost to damping, so "
            "height_collapse and the fall labels shrink while the low and medium strata are preserved "
            "because the gate is close to one in the tall upright stance those strata already hold. "
            "Closest rejected prior is the sibling push_recovery_stance_height_support_v1 (public_dev "
            "0.0, all 192 episodes height_collapse, zero verified push events), which instead added a "
            "separate one-sided quadratic height-deficit penalty of weight 3.0 whose weighted "
            "magnitude reached 15.35; that cost applied throughout deep transient crouches and "
            "dominated the program. This candidate adds no penalty and no new weight, so its maximum "
            "possible effect is the removal of 1.0 rate unit from a quiet crouch and it cannot "
            "dominate the transient regime."
        ),
    )
