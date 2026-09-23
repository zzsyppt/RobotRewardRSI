"""Complete Stair reward: contact-aware ascent shaping on top of the flat-derived base."""

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


def goal_progress_rate(inputs):
    return inputs.goal_progress_fraction


def support_contact_reward(inputs):
    return inputs.support_contact.float().sum(dim=1)


def upright_posture(inputs):
    return -inputs.projected_gravity[:, 2]


def build_training_components():
    return TrainingComponents(
        name="stair_contact_aware_ascent_v1",
        reward_terms=[
            RewardTerm(name="track_lin_vel_xy_exp", weight=3.5, fn=track_lin_vel_xy_exp),
            RewardTerm(name="track_ang_vel_z_exp", weight=0.75, fn=track_ang_vel_z_exp),
            RewardTerm(name="ang_vel_xy_l2", weight=-0.05, fn=ang_vel_xy_l2),
            RewardTerm(name="flat_orientation_l2", weight=-2.5, fn=flat_orientation_l2),
            RewardTerm(name="dof_torques_l2", weight=-0.0002, fn=dof_torques_l2),
            RewardTerm(name="dof_acc_l2", weight=-2.5e-7, fn=dof_acc_l2),
            RewardTerm(name="action_rate_l2", weight=-0.01, fn=action_rate_l2),
            RewardTerm(name="feet_air_time", weight=0.25, fn=feet_air_time),
            RewardTerm(name="goal_progress_rate", weight=1.5, fn=goal_progress_rate, clip_min=0.0, clip_max=1.0),
            RewardTerm(name="support_contact_reward", weight=0.3, fn=support_contact_reward, clip_min=0.0, clip_max=4.0),
            RewardTerm(name="upright_posture", weight=0.5, fn=upright_posture, clip_min=-1.0, clip_max=0.0),
        ],
        rationale=(
            "Closest prior candidate is the reference anchor "
            "go2-stair-official-flat-derived-v1, which scores 0.7467 with "
            "J_OOD 0.7467 and dominant failure forbidden_contact (0.131) "
            "concentrated in c1_conditional_frontier (0.324) and height_ood "
            "(0.333), plus goal_not_reached_detected 0.200. The reference "
            "reward is a flat-locomotion program with no spatial goal signal "
            "and no support-contact shaping, so it cannot distinguish "
            "standing on a step from climbing toward the top. Code-level "
            "difference: I keep all eight reference terms and their exact "
            "weights, and add three new RewardTerms driven only by declared "
            "fields: goal_progress_rate (weight 1.5, clipped [0,1]) rewards "
            "instantaneous clamped spatial progress toward the frozen goal; "
            "support_contact_reward (weight 0.3, clipped [0,4]) rewards "
            "having support effectors in contact, which is required for "
            "stair ascent and is not inherently bad; upright_posture "
            "(weight 0.5, clipped [-1,0]) uses projected_gravity z to "
            "discourage the base_orientation_instability that co-occurs with "
            "forbidden_contact on the frontier. This addresses the observed "
            "failure by giving the policy a dense ascent gradient and a "
            "contact/posture prior instead of relying on velocity tracking "
            "alone, while preserving every reference coefficient so the "
            "c0_id and c1_id guards are not disturbed."
        ),
    )
