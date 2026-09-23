'''Go2 Stair reward: reference flat-derived terms plus a terminal penalty for the task-owned forbidden-contact failure event.'''

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


def forbidden_contact_terminal_penalty(inputs):
    return inputs.forbidden_contact.float()


def build_training_components():
    return TrainingComponents(
        name='go2_stair_official_flat_derived_forbidden_contact_terminal_v1',
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
                name='forbidden_contact_terminal_penalty',
                weight=-5.0,
                fn=forbidden_contact_terminal_penalty,
                clip_min=0.0,
                clip_max=1.0,
                description=(
                    'Penalize the task-owned forbidden-contact termination event '
                    'to teach avoidance of pre-arrival contact failures on '
                    'extrapolated stairs.'
                ),
            ),
        ],
        rationale=(
            'Anchor is the verified current-best reference '
            'go2_stair_official_flat_derived_v1 (objective 0.9344). All eight '
            'reference terms and weights are preserved exactly. The single '
            'mechanism change is the addition of RewardTerm '
            'forbidden_contact_terminal_penalty, which returns the task-owned '
            'binary forbidden_contact mask (0/1) multiplied by weight -5.0, '
            'with clip bounds [0,1]. It differs at the code level from the '
            'closest prior candidate stair_forbidden_contact_force_penalty_v1: '
            'that candidate subtracted an unclipped dense L2 norm of the '
            'forbidden_contact_force vector at every control step (max_abs '
            '0.6) and collapsed height_ood success to 0.556 and joint_ood to '
            '0.074, while the new term uses only the binary termination mask '
            'and therefore penalizes only the exact terminal event, never '
            'benign contact forces during normal stair contact. Public dev '
            'evidence: the reference dominant failure is forbidden_contact '
            '(12.5% overall; 51.9% in joint_ood, 33.3% in tread_ood, 14.8% in '
            'height_ood), and c0_id/c1_id/c1_conditional_frontier/c2_mechanism_ood '
            'cells already have zero forbidden contact. The hypothesis is that '
            'making the forbidden-contact termination explicitly negative '
            'sharpens the value function avoidance gradient for pre-arrival '
            'contact failures on extrapolated tall-riser/short-tread stairs '
            'without altering behavior in in-distribution cells where the '
            'event never occurs, protecting the c0_id and c1_id guards.'
        ),
    )
