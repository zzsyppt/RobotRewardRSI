'''Complete, morphology-scaled standing recovery reward rates.

The accepted reference program is preserved apart from its single damping
mechanism, which is replaced by a lean-guarded damping cost. The task
dispatcher integrates these rates once with control dt; joint penalties use
means so changing joint count does not silently multiply their weight.
Perturbation schedules are not reward inputs.
'''

from true_rl_agent.candidates.components import RewardTerm, TrainingComponents


def upright_height(inputs):
    tilt = inputs.roll.square() + inputs.pitch.square()
    height_error = (inputs.base_height_relative - 1.0).square()
    return (-4.0 * tilt - 4.0 * height_error).exp()


def lean_guarded_damping(inputs):
    planar_speed = inputs.base_lin_vel_scaled[:, :2].square().sum(dim=1).sqrt() / 0.08
    angular_speed = inputs.base_ang_vel_scaled.square().sum(dim=1).sqrt() / 0.5
    planar_core = 0.5 * planar_speed.square()
    angular_core = 0.5 * angular_speed.square()
    planar_cost = planar_core / (1.0 + planar_core)
    angular_cost = angular_core / (1.0 + angular_core)
    lean = inputs.projected_gravity[:, :2].square().sum(dim=1).sqrt().clamp(max=1.0)
    guard = (1.0 - lean).clamp(min=0.0)
    return 1.0 - 0.25 * (guard * planar_cost + angular_cost)


def joint_pose(inputs):
    return inputs.joint_pos_error.square().mean(dim=1)


def action_rate(inputs):
    return inputs.action_delta.square().mean(dim=1)


def fall(inputs):
    return inputs.terminated_non_timeout.float()


def build_training_components():
    return TrainingComponents(
        name='push_recovery_lean_guarded_damping_v5',
        reward_terms=[
            RewardTerm(name='upright_height', weight=3.0, fn=upright_height),
            RewardTerm(name='lean_guarded_damping', weight=1.0, fn=lean_guarded_damping),
            RewardTerm(name='joint_pose', weight=-0.1, fn=joint_pose),
            RewardTerm(name='action_rate', weight=-0.01, fn=action_rate),
            RewardTerm(name='fall', weight=-20.0, fn=fall),
        ],
        rationale=(
            'Hypothesis. The closest prior program is the accepted current best '
            'push_standing_recovery_bounded_damping_v4, which recovers nearly all '
            'low and medium pushes but still loses high and repeated pushes to '
            'height collapse, side falls and backward falls. Its bounded_damping '
            'callable relaxes the planar damping share only by one minus 0.5 times '
            'the lean magnitude, so at large lean the planar cost is still charged '
            'at half strength exactly when a catch step is needed, and the '
            'remaining charge competes with the posture and termination mechanisms '
            'during the recovery window. Exact code-level difference. The planar '
            'relaxation factor is changed from 1 - 0.5*lean to a full guard '
            '(1 - lean) clamped at zero, so the planar damping cost vanishes '
            'entirely once the projected-gravity lean reaches its clamp, while the '
            'angular damping cost, the rational saturation x/(1+x) on both costs, '
            'the 0.08 and 0.5 normalization radii, and every other mechanism and '
            'weight are preserved unchanged. Prediction. Catch steps are no longer '
            'taxed during large-lean transients, so height collapse and residual '
            'falls should drop in the high and repeated push strata while low and '
            'medium stratum standing behaviour is retained because the small-signal '
            'cost geometry is unchanged.'
        ),
    )
