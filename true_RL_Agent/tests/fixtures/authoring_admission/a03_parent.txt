'''Complete, morphology-scaled standing recovery reward rates.

The verified reference standing program is preserved apart from its single
stationarity mechanism, which is replaced by a bounded damping cost with the
same small-signal geometry. The task dispatcher integrates these rates once
with control dt; joint penalties use means so changing joint count does not
silently multiply their weight. Perturbation schedules are not reward inputs.
'''

from true_rl_agent.candidates.components import RewardTerm, TrainingComponents


def upright_height(inputs):
    tilt = inputs.roll.square() + inputs.pitch.square()
    height_error = (inputs.base_height_relative - 1.0).square()
    return (-4.0 * tilt - 4.0 * height_error).exp()


def bounded_damping(inputs):
    planar_speed = inputs.base_lin_vel_scaled[:, :2].square().sum(dim=1).sqrt() / 0.08
    angular_speed = inputs.base_ang_vel_scaled.square().sum(dim=1).sqrt() / 0.5
    planar_core = 0.5 * planar_speed.square()
    angular_core = 0.5 * angular_speed.square()
    planar_cost = planar_core / (1.0 + planar_core)
    angular_cost = angular_core / (1.0 + angular_core)
    lean = inputs.projected_gravity[:, :2].square().sum(dim=1).sqrt().clamp(max=1.0)
    catch_relax = 1.0 - 0.5 * lean
    return 1.0 - 0.25 * (catch_relax * planar_cost + angular_cost)


def joint_pose(inputs):
    return inputs.joint_pos_error.square().mean(dim=1)


def action_rate(inputs):
    return inputs.action_delta.square().mean(dim=1)


def fall(inputs):
    return inputs.terminated_non_timeout.float()


def build_training_components():
    return TrainingComponents(
        name='push_standing_recovery_bounded_damping_v4',
        reward_terms=[
            RewardTerm(name='upright_height', weight=3.0, fn=upright_height),
            RewardTerm(name='bounded_damping', weight=1.0, fn=bounded_damping),
            RewardTerm(name='joint_pose', weight=-0.1, fn=joint_pose),
            RewardTerm(name='action_rate', weight=-0.01, fn=action_rate),
            RewardTerm(name='fall', weight=-20.0, fn=fall),
        ],
        rationale=(
            'Hypothesis. The closest prior program is the verified reference '
            'push_standing_recovery_stationary_reference_v3, which recovers almost '
            'every low and medium push but fails most high and repeated pushes as '
            'height collapse, side falls and backward falls. The suspected cause is '
            'its single stationarity callable, which keeps growing without bound '
            'once a push transient passes its clamp radius, so push-induced base '
            'velocity dominates credit assignment and the posture and termination '
            'mechanisms lose influence exactly during the recovery window. Exact '
            'code-level difference. That callable is replaced by bounded_damping, '
            'which keeps the reference small-signal costs, 0.5 times the square of '
            'scaled planar speed over 0.08 and 0.5 times the square of scaled '
            'angular speed over 0.5, but passes each of those costs through the '
            'rational saturation x divided by one plus x so both costs stay bounded '
            'by 1, and it relaxes the planar share by one minus 0.5 times the lean '
            'magnitude read from projected gravity so that a lean-triggered catch '
            'step is not charged the full damping cost. Every other mechanism and '
            'weight is preserved unchanged, including the upright_height posture '
            'reward, joint_pose, action_rate and the fall termination penalty. '
            'Prediction. Height collapse and residual falls should drop in the high '
            'and repeated push strata while low and medium stratum standing '
            'behaviour is retained because the small-signal cost geometry is '
            'unchanged.'
        ),
    )
