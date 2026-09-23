'''Go2 Stair reward: reference flat-derived terms plus an upslope-pitch allowance in flat_orientation_l2.'''
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
    roll_penalty = inputs.projected_gravity[:, 1].square()
    pitch = inputs.projected_gravity[:, 0]
    pitch_high = (pitch - 0.2).clamp(min=0.0)
    pitch_low = (-0.1 - pitch).clamp(min=0.0)
    pitch_penalty = pitch_high.square() + pitch_low.square()
    return roll_penalty + pitch_penalty


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


def build_training_components():
    return TrainingComponents(
        name='go2_stair_official_flat_derived_upslope_pitch_allowance_v1',
        reward_terms=[
            RewardTerm(name='track_lin_vel_xy_exp', weight=3.5, fn=track_lin_vel_xy_exp),
            RewardTerm(name='track_ang_vel_z_exp', weight=0.75, fn=track_ang_vel_z_exp),
            RewardTerm(name='ang_vel_xy_l2', weight=-0.05, fn=ang_vel_xy_l2),
            RewardTerm(name='flat_orientation_l2', weight=-2.5, fn=flat_orientation_l2),
            RewardTerm(name='dof_torques_l2', weight=-0.0002, fn=dof_torques_l2),
            RewardTerm(name='dof_acc_l2', weight=-2.5e-7, fn=dof_acc_l2),
            RewardTerm(name='action_rate_l2', weight=-0.01, fn=action_rate_l2),
            RewardTerm(name='feet_air_time', weight=0.25, fn=feet_air_time),
        ],
        rationale=(
            "Anchor is the verified current-best reference go2_stair_official_flat_derived_v1 "
            "(objective 0.9344, J_ood 0.9017, C0-ID/C1-ID 1.0). All eight reference terms and "
            "weights are preserved exactly; the single mechanism change is to flat_orientation_l2. "
            "The reference term penalizes projected_gravity[:, :2] (roll and any pitch) with the same "
            "quadratic weight -2.5. The new term still penalizes lateral roll with the unchanged "
            "quadratic, but allows a bounded upslope pitch band: projected_gravity[:,0] (~= sin(pitch)) "
            "in [-0.1, 0.2] is not penalized, while pitch beyond 0.2 rad and backward pitch beyond "
            "-0.1 rad are penalized quadratically. This is the only changed RewardTerm. "
            "Public-dev evidence: F_scalar is the only weak family (0.705 vs 1.0 elsewhere) and it "
            "fails almost exclusively by forbidden_contact (joint_ood forbidden_contact 0.519, "
            "success 0.407; tread_ood forbidden_contact 0.333, success 0.667) with elevated "
            "base_orientation_instability_detected on the same strata (joint 0.296, tread 0.333). "
            "Three prior scalar-family attempts changed feet/body vertical clearance or contact "
            "penalties: stair_forbidden_contact_force_penalty_v1 (dense force L2; collapsed joint "
            "success), stair_foot_clearance_gap_penalty_v1 (below-base foot gap; improved tread but "
            "raised joint forbidden_contact to 0.630), and stair_step_desc_clearance_v1 (symmetric "
            "foot-base deviation; collapsed all guards). None shaped sagittal posture. This term "
            "differs at the code level from those because it operates only on projected_gravity: it "
            "lets the torso pitch uphill on steep/short stairs so the body does not slice through the "
            "next riser plane, while the unchanged quadratic roll penalty and the quadratic bound on "
            "excessive pitch bound postural risk. Hypothesis: the reference fails on scalar-"
            "extrapolated stairs because it keeps the trunk level and overreaches with flat foot "
            "trajectories; an upslope-pitch allowance gives a dense gradient toward the climbing "
            "posture that clears high risers and short treads without weakening roll stability or "
            "any in-distribution mechanism."
        ),
    )
