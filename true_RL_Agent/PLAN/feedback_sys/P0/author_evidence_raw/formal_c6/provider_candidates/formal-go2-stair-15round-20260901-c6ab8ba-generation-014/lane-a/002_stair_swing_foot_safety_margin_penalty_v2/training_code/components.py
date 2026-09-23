'''Go2 Stair reward: reference flat-derived terms plus a swing-foot safety-margin penalty for OOD riser/tread extrapolation.'''

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


def swing_foot_safety_margin_penalty(inputs):
    foot_z = inputs.support_pos_relative_to_terrain_origin[:, :, 2]
    base_z = inputs.base_pos_relative_to_terrain_origin[:, 2].unsqueeze(1)
    rel_z = foot_z - base_z
    low_band = ((rel_z > -0.15) & (rel_z < 0.2)).float()
    swing = (~inputs.support_contact).float()
    forward_vel = inputs.support_lin_vel[:, :, 0].clamp(min=0.0)
    vertical_vel = inputs.support_lin_vel[:, :, 2]
    level_or_down = (vertical_vel <= 0.4).float()
    base_z_col = inputs.base_height.unsqueeze(1)
    clearance_gate = (
        (base_z_col - foot_z).clamp(min=0.0) < 0.02
    ).float()
    elevated = (inputs.base_height > 0.1).float()
    before_goal = (inputs.goal_progress_fraction < 1.0).float()
    penalty = (
        low_band * swing * forward_vel * level_or_down * clearance_gate
    ).sum(dim=1) * elevated * before_goal
    return penalty


def build_training_components():
    return TrainingComponents(
        name='go2_stair_flat_derived_swing_foot_safety_margin_v2',
        reward_terms=[
            RewardTerm(name='track_lin_vel_xy_exp', weight=3.5, fn=track_lin_vel_xy_exp),
            RewardTerm(name='track_ang_vel_z_exp', weight=0.75, fn=track_ang_vel_z_exp),
            RewardTerm(name='ang_vel_xy_l2', weight=-0.05, fn=ang_vel_xy_l2),
            RewardTerm(name='flat_orientation_l2', weight=-2.5, fn=flat_orientation_l2),
            RewardTerm(name='dof_torques_l2', weight=-0.0002, fn=dof_torques_l2),
            RewardTerm(name='dof_acc_l2', weight=-2.5e-7, fn=dof_acc_l2),
            RewardTerm(name='action_rate_l2', weight=-0.01, fn=action_rate_l2),
            RewardTerm(name='feet_air_time', weight=0.25, fn=feet_air_time),
            RewardTerm(
                name='swing_foot_safety_margin_penalty',
                weight=-0.2,
                fn=swing_foot_safety_margin_penalty,
                clip_min=0.0,
                clip_max=2.0,
                description='Penalize only swing feet whose heel/ankle path would sacrifice low clearance (foot within 0.02 m of the base plane at the step height) while sweeping forward without active lifting on elevated stair terrain before the goal.',
            ),
        ],
        rationale=(
            'Closest prior candidate: stair_swing_foot_safety_margin_penalty_v1 (attempt/latest_failed), which evolved the accepted current-best stair_swing_foot_level_sweep_penalty_v1 (objective 0.94827; C0/C1 ID guards 1.0; dominant residual failure forbidden_contact concentrated in joint_ood and height_ood) by converting the level-sweep penalty into a true clearance-margin penalty: an elevated-terrain, pre-goal sum over swing feet that (a) sit in the band [-0.15, 0.2] m below/above the base plane (low_band), (b) are not in contact, (c) sweep forward, (d) do not actively lift faster than 0.4 m/s, and (e) retain less than 0.02 m of vertical clearance margin between the per-foot z position and the base-height plane (clearance_gate). '
            'The prior candidate never reached evaluation: component smoke failed on its clearance_gate line because base_height is a per-environment scalar of shape [num_envs,], while foot_z is shape [num_envs, 4]; the raw subtraction broadcast [num_envs] against [num_envs, 4] and raised RuntimeError (tensor a (5) versus tensor b (4) at non-singleton dimension 1). '
            'The exact code-level difference in this candidate is the mechanical broadcast repair: base_height is unsqueezed to a column (base_height.unsqueeze(1), shape [num_envs, 1]) before subtracting foot_z, so clearance_gate evaluates elementwise over the four support members and the penalty returns shape [num_envs,]. '
            'No reward weight, clip, reference term, or mechanistic predicate changes relative to the prior candidate; all eight reference terms and weights from the accepted current-best anchor are preserved exactly. '
            'Hypothesis: residual forbidden contact on simultaneous riser/tread extrapolation is driven by swing feet losing clearance margin against the immediate step surface while sweeping forward; penalizing only margin-violating motion teaches the policy to clear the near surface first and should protect the OOD scalar strata without regressing C0/C1 in-distribution behavior.'
        ),
    )
