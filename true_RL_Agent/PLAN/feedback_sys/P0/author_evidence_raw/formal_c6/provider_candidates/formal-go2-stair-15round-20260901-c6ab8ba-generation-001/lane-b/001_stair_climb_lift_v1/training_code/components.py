"""Go2 Stair reward: flat-derived reference plus an explicit climb-lift term for steep OOD stairs."""

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


def climb_lift(inputs):
    v_up = inputs.base_lin_vel_yaw[:, 2]
    commanded_forward = inputs.commanded_base_vel[:, 0]
    on_course = inputs.goal_progress_fraction > 0.05
    commanded_advance = commanded_forward > 0.1
    active = (on_course & commanded_advance).float()
    return v_up.clamp(min=0.0) * active


def build_training_components():
    return TrainingComponents(
        name="go2_stair_official_flat_derived_climb_lift_v1",
        reward_terms=[
            RewardTerm(name="track_lin_vel_xy_exp", weight=3.5, fn=track_lin_vel_xy_exp),
            RewardTerm(name="track_ang_vel_z_exp", weight=0.75, fn=track_ang_vel_z_exp),
            RewardTerm(name="ang_vel_xy_l2", weight=-0.05, fn=ang_vel_xy_l2),
            RewardTerm(name="flat_orientation_l2", weight=-2.5, fn=flat_orientation_l2),
            RewardTerm(name="dof_torques_l2", weight=-0.0002, fn=dof_torques_l2),
            RewardTerm(name="dof_acc_l2", weight=-2.5e-7, fn=dof_acc_l2),
            RewardTerm(name="action_rate_l2", weight=-0.01, fn=action_rate_l2),
            RewardTerm(name="feet_air_time", weight=0.25, fn=feet_air_time),
            RewardTerm(name="climb_lift", weight=0.75, fn=climb_lift, clip_min=0.0, clip_max=1.0),
        ],
        rationale=(
            "Anchor is the verified current-best reference go2_stair_official_flat_derived_v1; "
            "all eight reference terms and weights are preserved exactly. The single mechanism "
            "change is the addition of RewardTerm climb_lift, which rewards positive world "
            "vertical base velocity (base_lin_vel_yaw[:, 2]) only while the policy is commanded "
            "forward (commanded_base_vel[:, 0] > 0.1) and the robot is on-course "
            "(goal_progress_fraction > 0.05), with the raw term clipped to [0, 1] so jumping is "
            "never rewarded beyond a 1 m/s lift rate. Public dev evidence: the only failing cells "
            "are tall-riser / short-tread OOD cells (height_ood_03 0.56, joint_ood_01 0.22, "
            "joint_ood_03 0.0, tread_ood_03 0.0) and forbidden_contact is the dominant failure "
            "(12.5% overall, 51.9% in joint_ood). The reference has no upward-motion shaping "
            "(the flat-derived z-velocity penalty was removed for ascent), so the policy has no "
            "signal to lift the body and legs over steep risers; climb_lift is the active "
            "complement of that removed penalty and directly targets the forbidden-contact "
            "mechanism on tall steps without ever penalizing downward motion or acting on flat "
            "ground."
        ),
    )
