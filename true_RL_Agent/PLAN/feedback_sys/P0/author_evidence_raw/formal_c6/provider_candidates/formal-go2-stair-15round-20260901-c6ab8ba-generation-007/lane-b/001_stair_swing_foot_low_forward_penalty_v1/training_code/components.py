"""Go2 Stair reward: reference flat-derived terms plus a swing-foot low-forward trip-hazard penalty for OOD riser/tread extrapolation."""

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


def swing_foot_low_forward_penalty(inputs):
    foot_z = inputs.support_pos_relative_to_terrain_origin[:, :, 2]
    base_z = inputs.base_pos_relative_to_terrain_origin[:, 2].unsqueeze(1)
    rel_z = foot_z - base_z
    low_band = ((rel_z > -0.15) & (rel_z < 0.2)).float()
    swing = (~inputs.support_contact).float()
    forward_vel = inputs.support_lin_vel[:, :, 0].clamp(min=0.0)
    elevated = (inputs.base_height > 0.1).float()
    before_goal = (inputs.goal_progress_fraction < 1.0).float()
    penalty = (low_band * swing * forward_vel).sum(dim=1) * elevated * before_goal
    return penalty


def build_training_components():
    return TrainingComponents(
        name="go2_stair_official_flat_derived_swing_foot_low_forward_v1",
        reward_terms=[
            RewardTerm(name="track_lin_vel_xy_exp", weight=3.5, fn=track_lin_vel_xy_exp),
            RewardTerm(name="track_ang_vel_z_exp", weight=0.75, fn=track_ang_vel_z_exp),
            RewardTerm(name="ang_vel_xy_l2", weight=-0.05, fn=ang_vel_xy_l2),
            RewardTerm(name="flat_orientation_l2", weight=-2.5, fn=flat_orientation_l2),
            RewardTerm(name="dof_torques_l2", weight=-0.0002, fn=dof_torques_l2),
            RewardTerm(name="dof_acc_l2", weight=-2.5e-7, fn=dof_acc_l2),
            RewardTerm(name="action_rate_l2", weight=-0.01, fn=action_rate_l2),
            RewardTerm(name="feet_air_time", weight=0.25, fn=feet_air_time),
            RewardTerm(
                name="swing_foot_low_forward_penalty",
                weight=-0.2,
                fn=swing_foot_low_forward_penalty,
                clip_min=0.0,
                clip_max=2.0,
                description="Penalize forward sweep of swing feet near the base plane before the goal on elevated stairs to reduce riser/nose forbidden-contact.",
            ),
        ],
        rationale=(
            "Anchor is the verified current-best reference go2_stair_official_flat_derived_v1 "
            "(objective 0.9344); all eight reference terms and weights are preserved exactly. "
            "The single mechanism change is the addition of RewardTerm swing_foot_low_forward_penalty, "
            "a trip-hazard shaping term that penalizes the forward world-x velocity of each swing foot only "
            "while that foot is in the low-clearance band around the base plane (foot z minus base z between "
            "-0.15 m and 0.2 m), only while the base is elevated above the start platform "
            "(base_height > 0.1 m), and only before the goal is reached (goal_progress_fraction < 1.0); "
            "the raw term is clipped to [0, 2.0] and weighted -0.2, so its weighted contribution is at most "
            "-0.4 per step. Closest prior candidate: stair_foot_clearance_gap_penalty_v1 "
            "(objective 0.9348, rejected), which also shaped foot height to reduce forbidden contact but "
            "penalized the squared static vertical-gap band [0, 0.2] m of every foot above the base using only "
            "base_pos_relative_to_terrain_origin and support_pos_relative_to_terrain_origin; it improved "
            "tread_ood (0.734 to 0.976) but regressed height_ood (0.888 to 0.718) and left joint_ood near 0.43. "
            "The proposed term differs at the code level: it adds support_lin_vel (forward-velocity factor), "
            "support_contact (swing gate), base_height and goal_progress_fraction (activation gates), and it "
            "penalizes hazardous horizontal slicing motion of low airborne feet instead of a static position "
            "deficit, so feet that are stationary, moving backward, or far below the base during a normal "
            "flat-ground stance are not penalized; the term is inactive on the flat start and on the top "
            "platform. Public dev evidence: the reference dominant failure is forbidden_contact (12.5 percent "
            "overall, 51.9 percent in joint_ood, 33.3 percent in tread_ood) with joint_ood success 0.407. "
            "Two already-evaluated forbidden-contact responses are not reused because the dense force L2 "
            "penalty collapsed tall-step successes (joint success 0.074, objective 0.918) and the terminal "
            "penalty induced orientation instability and insufficient clearance (objective 0.922). Hypothesis: "
            "on extrapolated tall-riser/short-tread stairs the policy advances swing feet through the "
            "riser/nose plane at low vertical clearance; gating a mild penalty on swing feet that move forward "
            "while in the low-clearance band during ascent teaches vertical-first foot lifting before horizontal "
            "sweep, reducing pre-arrival forbidden-contact terminations on scalar-extrapolation strata while the "
            "band and activation gates keep the term silent on flat and post-arrival behavior, protecting the "
            "C0-ID and C1-ID guards."
        ),
    )
