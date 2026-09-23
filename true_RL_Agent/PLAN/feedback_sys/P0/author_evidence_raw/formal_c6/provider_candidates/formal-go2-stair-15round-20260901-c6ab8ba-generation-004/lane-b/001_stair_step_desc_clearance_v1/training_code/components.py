import torch

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


def step_descent_clearance_penalty(inputs):
    base_z = inputs.base_pos_relative_to_terrain_origin[:, 2].unsqueeze(1)
    foot_z = inputs.support_pos_relative_to_terrain_origin[:, :, 2]
    deviation = (foot_z - base_z).clamp(min=-0.35, max=0.30)
    return deviation.square().sum(dim=1)


def build_training_components():
    return TrainingComponents(
        name="go2_stair_official_flat_derived_step_descent_clearance_v1",
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
                name="step_descent_clearance_penalty",
                weight=-4.0,
                fn=step_descent_clearance_penalty,
                clip_min=0.0,
                clip_max=0.4,
                description="Penalize the squared vertical deviation of each foot from the base height, discouraging both low foot slices through tall risers and excessive over-lift on extrapolated stairs.",
            ),
        ],
        rationale=(
            "Anchor is the verified current-best reference go2_stair_official_flat_derived_v1 "
            "(objective 0.9344). All eight reference terms and weights are preserved exactly. "
            "The single mechanism change is the addition of RewardTerm "
            "step_descent_clearance_penalty, which differs at the code level from the closest "
            "prior candidate stair_swing_foot_clearance_v1: the rejected candidate penalized a "
            "low-clearance band and a below-base band with two separate maxima and added their "
            "squared penalties, which left feet far above the base unshaped and relied on an "
            "open-ended low clearance clamp; this new term instead uses one continuous squared "
            "deviation of each foot terrain-relative height from the base height on a symmetric "
            "band [-0.35, 0.30] m, so very low feet (riser-plane slicing) and overly high feet "
            "(excessive lift) are both penalized with a single bounded spring. It also differs "
            "from stair_foot_clearance_gap_penalty_v1, which penalized only feet hanging just "
            "below the base in [0, 0.2] m. Public dev evidence: the reference fails mainly in "
            "the scalar OOD family (F_scalar 0.705), with joint_ood success 0.407 and "
            "forbidden_contact 0.519; the rejected gap penalty improved tread_ood success from "
            "0.667 to 0.963 but reduced joint_ood success to 0.296 with forbidden_contact 0.630, "
            "suggesting the below-base-only penalty over-lifted the swing foot. The hypothesis "
            "is that precise foot placement relative to the base height on steep and short-tread "
            "extrapolations matters more than raw clearance, and that a symmetric bounded "
            "quadratic cost teaches a compact stepping pattern which clears the riser plane "
            "without excessive foot elevation, reducing forbidden-contact terminations on "
            "joint_ood while retaining the tread_ood gains."
        ),
    )
