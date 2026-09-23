"""Go2 Stair reward: reference flat-derived terms plus a swing-foot clearance reward for OOD riser/tread extrapolation."""

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


def swing_foot_clearance_reward(inputs):
    foot_z = inputs.support_pos_relative_to_terrain_origin[:, :, 2]
    base_z = inputs.base_pos_relative_to_terrain_origin[:, 2].unsqueeze(1)
    rel_z = foot_z - base_z
    lift = (rel_z - 0.2).clamp(min=0.0)
    swing = (~inputs.support_contact).float()
    elevated = (inputs.base_height > 0.1).float().unsqueeze(1)
    moving = (inputs.commanded_base_vel[:, :2].norm(dim=1) > 0.1).float().unsqueeze(1)
    before_goal = (inputs.goal_progress_fraction < 1.0).float().unsqueeze(1)
    reward = (swing * lift * elevated * moving * before_goal).sum(dim=1)
    return reward


def build_training_components():
    return TrainingComponents(
        name="go2_stair_official_flat_derived_swing_foot_clearance_v1",
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
                name="swing_foot_clearance_reward",
                weight=0.5,
                fn=swing_foot_clearance_reward,
                clip_min=0.0,
                clip_max=1.0,
                description="Reward airborne foot altitude above base plus 0.2 m on stair-centric phases before the goal.",
            ),
        ],
        rationale=(
            "Anchor is the accepted reference current-best go2_stair_official_flat_derived_v1 "
            "(public_dev objective 0.9344, guards J_OOD 0.9017 / C0-ID 1.0 / C1-ID 1.0); all eight "
            "reference RewardTerms and their nonzero weights are preserved exactly. The single mechanism "
            "change is the addition of RewardTerm swing_foot_clearance_reward, which rewards altitude of "
            "airborne feet above the base plane plus 0.2 m, gated by swing (support_contact false), base "
            "elevation above 0.1 m, commanded horizontal motion above 0.1 m/s, and goal_progress_fraction "
            "below 1.0; the raw term is clipped to [0.0, 1.0] and weighted 0.5, so its weighted "
            "contribution is at most +0.5 per step. Closest prior candidate: "
            "stair_foot_clearance_gap_penalty_v1 (objective 0.9348, rejected) penalized the squared static "
            "vertical-gap band [0.0, 0.2] m of every foot above the base using only base and foot world "
            "positions; it improved tread_ood (0.734 to 0.976) but regressed height_ood (0.888 to 0.718) "
            "because tall-riser scalar extrapolation requires foot clearance above 0.2 m, which that band "
            "cap punished. The proposed term differs at the code level: it is a positive altitude reward "
            "rather than a band penalty, so it never caps foot height; it fires only on airborne feet "
            "(~support_contact) instead of all feet; and it adds support_contact, commanded_base_vel, "
            "base_height, and goal_progress_fraction activation gates. Relative to the latest rejected "
            "candidate stair_swing_foot_low_forward_penalty_v1 (objective 0.9527), which penalized the "
            "forward world-x sweep of low swing feet and lifted tread_ood to 0.923 but left joint_ood "
            "success stalled at 0.407 with forbidden_contact rising to 0.593, the proposed term optimizes "
            "vertical clearance rather than horizontal sweep rate: it rewards the altitude signal "
            "(foot_z - base_z) directly with a floor threshold instead of penalizing banded forward "
            "velocity, so the policy is taught the missing vertical-first foot elevation on simultaneous "
            "tall-riser/short-tread extrapolation. Public dev evidence: forbidden_contact is the dominant "
            "reference failure (12.5 percent overall, 51.9 percent in joint_ood, 33.3 percent in "
            "tread_ood, 14.8 percent in height_ood), and joint_ood is the weakest OOD cell. Hypothesis: "
            "rewarding swing-foot altitude above the base floor during stair-centric phases teaches the "
            "policy to clear extrapolated risers and tread noses before horizontal advance, reducing "
            "pre-arrival forbidden-contact terminations on F_scalar strata while the elevation, motion, "
            "swing, and before-goal gates keep the term silent on the flat start, after arrival, and when "
            "idle, protecting the C0-ID and C1-ID guards."
        ),
    )
