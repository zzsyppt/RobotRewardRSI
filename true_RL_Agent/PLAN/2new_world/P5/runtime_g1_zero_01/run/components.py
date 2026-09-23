"""Official Isaac Lab G1 rough reward over profile-declared role groups."""
from true_rl_agent.candidates.components import RewardTerm, TrainingComponents

def termination_penalty(inputs):
    return inputs.terminated_non_timeout.float() + inputs.roll * 0.0

def track_lin_vel_xy_exp(inputs):
    error = (inputs.commanded_base_vel[:, :2] - inputs.base_lin_vel_yaw[:, :2]).square().sum(dim=1)
    return (-error / 0.25).exp()

def track_ang_vel_z_exp(inputs):
    error = (inputs.commanded_base_vel[:, 2] - inputs.base_ang_vel_world[:, 2]).square()
    return (-error / 0.25).exp()

def feet_air_time(inputs):
    in_contact = inputs.support_contact_time > 0.0
    in_mode_time = inputs.support_contact_time.where(in_contact, inputs.support_air_time)
    single_stance = in_contact.int().sum(dim=1) == 1
    reward = (in_mode_time * single_stance.unsqueeze(-1).float()).min(dim=1).values.clamp(max=0.4)
    moving = inputs.commanded_base_vel[:, :2].square().sum(dim=1).sqrt() > 0.1
    return reward * moving.float()

def feet_slide(inputs):
    contact = inputs.support_contact_force_history.norm(dim=-1).max(dim=1).values > 1.0
    planar_speed = inputs.support_lin_vel[:, :, :2].norm(dim=-1)
    return (planar_speed * contact.float()).sum(dim=1)

def dof_pos_limits(inputs):
    lower = inputs.ankle_group_soft_pos_limits[:, :, 0]
    upper = inputs.ankle_group_soft_pos_limits[:, :, 1]
    below = -(inputs.ankle_group_pos - lower).clamp(max=0.0)
    above = (inputs.ankle_group_pos - upper).clamp(min=0.0)
    return (below + above).sum(dim=1)

def joint_deviation_hip(inputs):
    return inputs.hip_yaw_roll_group_pos_error.abs().sum(dim=1)

def joint_deviation_arms(inputs):
    return inputs.arm_group_pos_error.abs().sum(dim=1)

def joint_deviation_fingers(inputs):
    return inputs.finger_group_pos_error.abs().sum(dim=1)

def joint_deviation_torso(inputs):
    return inputs.torso_group_pos_error.abs().sum(dim=1)

def ang_vel_xy_l2(inputs):
    return inputs.base_ang_vel[:, :2].square().sum(dim=1)

def flat_orientation_l2(inputs):
    return inputs.projected_gravity[:, :2].square().sum(dim=1)

def action_rate_l2(inputs):
    return inputs.action_delta.square().sum(dim=1)

def dof_acc_l2(inputs):
    return inputs.hip_knee_group_acc.square().sum(dim=1)

def dof_torques_l2(inputs):
    return inputs.locomotion_group_torque.square().sum(dim=1)

def build_training_components():
    return TrainingComponents(name='official_g1_rough_role_projected_vnext_v1', reward_terms=[RewardTerm(name='termination_penalty', weight=-200.0, fn=termination_penalty), RewardTerm(name='track_lin_vel_xy_exp', weight=1.0, fn=track_lin_vel_xy_exp), RewardTerm(name='track_ang_vel_z_exp', weight=2.0, fn=track_ang_vel_z_exp), RewardTerm(name='feet_air_time', weight=0.25, fn=feet_air_time), RewardTerm(name='feet_slide', weight=-0.1, fn=feet_slide), RewardTerm(name='dof_pos_limits', weight=-1.0, fn=dof_pos_limits), RewardTerm(name='joint_deviation_hip', weight=-0.1, fn=joint_deviation_hip), RewardTerm(name='joint_deviation_arms', weight=-0.1, fn=joint_deviation_arms), RewardTerm(name='joint_deviation_fingers', weight=-0.05, fn=joint_deviation_fingers), RewardTerm(name='joint_deviation_torso', weight=-0.1, fn=joint_deviation_torso), RewardTerm(name='ang_vel_xy_l2', weight=-0.05, fn=ang_vel_xy_l2), RewardTerm(name='flat_orientation_l2', weight=-1.0, fn=flat_orientation_l2), RewardTerm(name='action_rate_l2', weight=-0.005, fn=action_rate_l2), RewardTerm(name='dof_acc_l2', weight=-1.25e-07, fn=dof_acc_l2), RewardTerm(name='dof_torques_l2', weight=-1.5e-07, fn=dof_torques_l2)], rationale='All 15 active Isaac Lab G1 rough locomotion terms are preserved; concrete joint membership comes only from RobotProfile roles.')
