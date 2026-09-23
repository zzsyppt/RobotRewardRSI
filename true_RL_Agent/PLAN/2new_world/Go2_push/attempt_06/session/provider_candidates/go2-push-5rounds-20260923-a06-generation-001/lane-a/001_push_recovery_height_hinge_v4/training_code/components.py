'''Morphology-scaled recovery reward with an explicit anti-collapse stance hinge.'''

from true_rl_agent.candidates.components import RewardInputs, RewardTerm, TrainingComponents


def recovery_stance(inputs: RewardInputs):
    tilt = inputs.roll.square() + inputs.pitch.square()
    height = inputs.base_height_relative
    deficit = (1.0 - height).clamp(min=0.0)
    excess = (height - 1.0).clamp(min=0.0)
    return (-4.0 * tilt).exp() - 6.0 * deficit.square() - 0.5 * excess.square()


def stationary_v3(inputs: RewardInputs):
    planar_speed = inputs.base_lin_vel_scaled[:, :2].square().sum(dim=1).sqrt() / 0.08
    angular_speed = inputs.base_ang_vel_scaled.square().sum(dim=1).sqrt() / 0.5
    planar_core = planar_speed.clamp(max=1.0)
    angular_core = angular_speed.clamp(max=1.0)
    planar_penalty = 0.5 * planar_core.square() + planar_speed - planar_core
    angular_penalty = 0.5 * angular_core.square() + angular_speed - angular_core
    return 1.0 - 0.25 * (planar_penalty + angular_penalty)


def joint_pose(inputs: RewardInputs):
    return inputs.joint_pos_error.square().mean(dim=1)


def action_rate(inputs: RewardInputs):
    return inputs.action_delta.square().mean(dim=1)


def fall(inputs: RewardInputs):
    return inputs.terminated_non_timeout.float()


def build_training_components() -> TrainingComponents:
    return TrainingComponents(
        name='push_recovery_height_hinge_v4',
        reward_terms=[
            RewardTerm(name='recovery_stance', weight=3.0, fn=recovery_stance, clip_min=-4.0, clip_max=1.0, description='Hinge-penalized upright stance to prevent height collapse.'),
            RewardTerm(name='stationary_v3', weight=1.0, fn=stationary_v3),
            RewardTerm(name='joint_pose', weight=-0.1, fn=joint_pose),
            RewardTerm(name='action_rate', weight=-0.01, fn=action_rate),
            RewardTerm(name='fall', weight=-20.0, fn=fall),
        ],
        diagnostics=[],
        schedules=[],
        rationale='Closest prior is current_best push_standing_recovery_stationary_reference_v3. The single changed mechanism is the upright-height reward callable: it keeps the exp(-4*tilt) orientation reward but replaces the symmetric squared height-error with a one-sided hinge on height deficit, clip-bounded at -4, so low-height crouching produces a strong negative gradient and weighted contribution. The stationary, joint-pose, action-rate, and fall terms and all weights are preserved. This targets the public-dev height_collapse_dominated failure without changing task timing, pushes, or protocol.',
    )
