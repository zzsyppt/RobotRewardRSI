"""Isaac Lab Go2 flat reward with two explicit Stair-context adaptations."""
from true_rl_agent.candidates.components import RewardTerm, TrainingComponents

def track_lin_vel_xy_exp(inputs):
    error = (inputs.commanded_base_vel[:, :2] - inputs.base_lin_vel[:, :2]).square().sum(dim=1)
    return (-error / 0.25).exp() + inputs.roll * 0.0

def track_ang_vel_z_exp(inputs):
    error = (inputs.commanded_base_vel[:, 2] - inputs.base_ang_vel[:, 2]).square()
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
    reward = ((inputs.support_last_air_time - 0.5) * inputs.support_first_contact.float()).sum(dim=1)
    moving = inputs.commanded_base_vel[:, :2].norm(dim=1) > 0.1
    return reward * moving.float()

def build_training_components():
    return TrainingComponents(name='go2_stair_official_flat_derived_v1', reward_terms=[RewardTerm(name='track_lin_vel_xy_exp', weight=3.5, fn=track_lin_vel_xy_exp), RewardTerm(name='track_ang_vel_z_exp', weight=0.75, fn=track_ang_vel_z_exp), RewardTerm(name='ang_vel_xy_l2', weight=-0.05, fn=ang_vel_xy_l2), RewardTerm(name='flat_orientation_l2', weight=-2.5, fn=flat_orientation_l2), RewardTerm(name='dof_torques_l2', weight=-0.0002, fn=dof_torques_l2), RewardTerm(name='dof_acc_l2', weight=-2.5e-07, fn=dof_acc_l2), RewardTerm(name='action_rate_l2', weight=-0.01, fn=action_rate_l2), RewardTerm(name='feet_air_time', weight=0.25, fn=feet_air_time)], rationale='The raw functions and six unchanged non-zero weights come from Isaac Lab 2.3.0 UnitreeGo2FlatEnvCfg.  The forward-tracking weight is command-distribution normalized from 1.5 to 3.5, and the vertical-velocity penalty is disabled for Stair ascent.')
