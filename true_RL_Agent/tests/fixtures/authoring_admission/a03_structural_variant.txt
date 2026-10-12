'''Complete, morphology-scaled push-recovery reward rates.

The dispatcher integrates these rates once with control dt and replaces upstream
rewards. Joint penalties use means so joint count does not silently multiply
their weight. Perturbation schedules are not reward inputs.
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
    lean = inputs.projected_gravity[:, :2]
    lean_mag = lean.square().sum(dim=1).sqrt().clamp(max=1.0)
    unit = 0.08 * (9.81 * inputs.nominal_base_height).clamp(min=1e-6).sqrt()
    outward = (inputs.base_lin_vel[:, :2] * lean).sum(dim=1).clamp(min=0.0) / unit
    outward_core = 0.5 * outward.square()
    outward_cost = outward_core / (1.0 + outward_core)
    catch_relax = 1.0 - 0.5 * lean_mag
    return 1.0 - 0.25 * (catch_relax * planar_cost + angular_cost) - 0.3 * outward_cost


def joint_pose(inputs):
    return inputs.joint_pos_error.square().mean(dim=1)


def action_rate(inputs):
    return inputs.action_delta.square().mean(dim=1)


def fall(inputs):
    return inputs.terminated_non_timeout.float()


def build_training_components():
    return TrainingComponents(
        name='push_recovery_lean_outward_damping_v5',
        reward_terms=[
            RewardTerm(name='upright_height', weight=3.0, fn=upright_height),
            RewardTerm(name='lean_guarded_damping', weight=1.0, fn=lean_guarded_damping),
            RewardTerm(name='joint_pose', weight=-0.1, fn=joint_pose),
            RewardTerm(name='action_rate', weight=-0.01, fn=action_rate),
            RewardTerm(name='fall', weight=-20.0, fn=fall),
        ],
        rationale=(
            'Hypothesis. Closest prior candidate is the accepted current best '
            'push_standing_recovery_bounded_damping_v4 (public_dev mean 0.838542), '
            'which bounded the stationarity cost with a rational saturation plus a '
            'lean catch relax. Public evidence since then: the sibling '
            'push_recovery_nonvanishing_stance_margin_v4 reshaped only the posture '
            'callable into a piecewise quadratic-then-linear margin and scored '
            '0.828125, with height_collapse, fall_backward and fall_side still '
            'dominating, so posture-shape changes are treated as negative evidence '
            'and every posture, joint, action and termination mechanism is '
            'preserved unchanged here. Exact code-level difference. The single '
            'velocity-cost callable bounded_damping becomes lean_guarded_damping: '
            'it keeps the anchor small-signal geometry and rational saturation of '
            'the planar and angular costs, 0.5 times the square of scaled speed '
            'over 0.08 and over 0.5 passed through x divided by one plus x, and the '
            'same catch_relax of one minus 0.5 times the lean magnitude, and adds '
            'one bounded directional cost, the component of the raw base-frame '
            'linear velocity along the horizontal base-frame gravity direction '
            'rectified by clamp min zero so only motion that drives the base '
            'further along its current lean is charged, divided by the same 0.08 '
            'velocity unit expressed as sqrt(9.81 * nominal_base_height) for '
            'morphology portability and saturated by the same rational shape. Both '
            'signals are already robot-base-frame, so no world/base mixing occurs, '
            'and a zero projected-gravity fallback makes the new cost vanish '
            'instead of mis-penalizing. Prediction. Velocity that deepens a lean is '
            'discounted before it becomes a fall while corrective and catch-step '
            'motion keeps the anchor relaxed cost, so fall_backward and fall_side '
            'should drop in the medium, high and repeated push strata without '
            'losing the low stratum recovery and without repeating the rejected '
            'posture-tail mechanism.'
        ),
    )
