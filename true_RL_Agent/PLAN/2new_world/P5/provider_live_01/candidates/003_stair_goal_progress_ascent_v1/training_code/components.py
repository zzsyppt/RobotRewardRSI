"""Complete Stair reward program: reference flat-derived terms plus one dense goal-progress term."""

from true_rl_agent.candidates.components import RewardTerm, TrainingComponents


def track_lin_vel_xy_exp(inputs):
    error = (
        inputs.commanded_base_vel[:, :2] - inputs.base_lin_vel[:, :2]
    ).square().sum(dim=1)
    return (-error / 0.25).exp()


def track_ang_vel_z_exp(inputs):
    error = (
        inputs.commanded_base_vel[:, 2] - inputs.base_ang_vel[:, 2]
    ).square()
    return (-error / 0.25).exp()


def ang_vel_xy_l2(inputs):
    return inputs.base_ang_vel[:, :2].square().sum(dim=1)


def flat_orientation_l2(inputs):
    return inputs.projected_gravity[:, :2].square().sum(dim=1)


def dof_torques_l2(inputs):
    return inputs.joint_torque.square().sum(dim=1)


def dof_acc_l2(inputs):
    return inputs.joint_acc.square().sum(dim=1)


def action_rate_l2(inputs):
    return inputs.action_delta.square().sum(dim=1)


def feet_air_time(inputs):
    reward = (
        (inputs.support_last_air_time - 0.5)
        * inputs.support_first_contact.float()
    ).sum(dim=1)
    moving = inputs.commanded_base_vel[:, :2].norm(dim=1) > 0.1
    return reward * moving.float()


def goal_progress(inputs):
    return inputs.goal_progress_fraction


def build_training_components():
    return TrainingComponents(
        name="stair_goal_progress_ascent_v1",
        reward_terms=[
            RewardTerm(
                name="track_lin_vel_xy_exp", weight=3.5, fn=track_lin_vel_xy_exp
            ),
            RewardTerm(
                name="track_ang_vel_z_exp", weight=0.75, fn=track_ang_vel_z_exp
            ),
            RewardTerm(name="ang_vel_xy_l2", weight=-0.05, fn=ang_vel_xy_l2),
            RewardTerm(
                name="flat_orientation_l2", weight=-2.5, fn=flat_orientation_l2
            ),
            RewardTerm(name="dof_torques_l2", weight=-0.0002, fn=dof_torques_l2),
            RewardTerm(name="dof_acc_l2", weight=-2.5e-7, fn=dof_acc_l2),
            RewardTerm(name="action_rate_l2", weight=-0.01, fn=action_rate_l2),
            RewardTerm(name="feet_air_time", weight=0.25, fn=feet_air_time),
            RewardTerm(
                name="goal_progress",
                weight=1.0,
                fn=goal_progress,
                clip_min=0.0,
                clip_max=1.0,
                description="Dense clamped spatial ascent toward the frozen top goal.",
            ),
        ],
        rationale=(
            "Closest prior candidate is stair_contact_aware_ascent_v1, which "
            "changed three reward mechanisms at once (goal_progress_rate, "
            "support_contact_reward, upright_posture). This candidate isolates "
            "a single mechanism: relative to the accepted reference program "
            "go2-stair-official-flat-derived-v1, the eight reference terms "
            "(track_lin_vel_xy_exp 3.5, track_ang_vel_z_exp 0.75, "
            "ang_vel_xy_l2 -0.05, flat_orientation_l2 -2.5, dof_torques_l2 "
            "-0.0002, dof_acc_l2 -2.5e-7, action_rate_l2 -0.01, feet_air_time "
            "0.25) are reproduced with identical callable bodies, weights and "
            "clipping, and exactly one RewardTerm is added: goal_progress, "
            "weight 1.0, clipped to [0.0, 1.0], whose callable returns "
            "inputs.goal_progress_fraction. The other two terms of the prior "
            "candidate are not carried over, so any outcome change is "
            "attributable to the goal-progress mechanism alone. Hypothesis: "
            "the reference program is a flat-locomotion objective that tracks "
            "commanded body velocity and penalises tilt, torque and action "
            "change, but assigns no value to spatial displacement toward the "
            "frozen top goal. Public failure evidence for the reference is "
            "dominated by forbidden_contact 0.131, concentrated in "
            "c1_conditional_frontier (0.324) and height_ood (0.333), together "
            "with goal_not_reached_detected 0.200: a policy can hold a "
            "commanded body velocity while its base makes little upward "
            "progress, and non-arrival is the largest single source of lost "
            "primary score. J_OOD averages three equally weighted families, "
            "so the c1 conditional-frontier family alone carries one third of "
            "the score, and the scalar families add the height, tread and "
            "joint surfaces where the trained gait is least likely to convert "
            "into real ascent. The added term supplies a dense per-step "
            "gradient that is monotone in the task-owned clamped spatial "
            "progress, crediting every gain between the frozen start and the "
            "frozen top on regular, irregular, high-riser and long-tread "
            "surfaces instead of only at the sparse top-arrival event; "
            "because the field is recomputed every step and is not latched, "
            "sliding back down forfeits the bonus, so the mechanism rewards "
            "sustained ascent rather than a momentary arrival. At weight 1.0 "
            "with a [0, 1] clip the term contributes at most 1.0 times the "
            "control-step duration per step, about 29 percent of the maximum "
            "velocity-tracking contribution, which adds ascent drive on the "
            "out-of-distribution strata without overriding the reference "
            "velocity, posture, torque and smoothness mechanisms that protect "
            "the in-distribution guards. No contact, posture or force term is "
            "introduced, so the frontier contact failures are addressed only "
            "indirectly, through more decisive climbing; this keeps the "
            "experiment to exactly one attributable reward mechanism."
        ),
    )
