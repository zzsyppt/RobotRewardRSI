"""Scale-normalized standing recovery reward rates with push compensation.

The damping term subtracts the planar velocity increment that the task
controller already injected on the current step from the scale-normalized
linear speed before damping, so the external increment is not charged to the
policy. The reported increment is zero on steps without an injected pulse, so
no extra masking is needed, and residual motion after a pulse stays damped on
later steps. The damping core keeps the same soft hinge shape as the verified
standing-recovery program, expressed with elementary elementwise operations.
All other mechanisms are unchanged.
"""

from true_rl_agent.candidates.components import RewardTerm, TrainingComponents


def upright_height(inputs):
    tilt = inputs.roll.square() + inputs.pitch.square()
    height_error = (inputs.base_height_relative - 1.0).square()
    return (-4.0 * tilt - 4.0 * height_error).exp()


def stationary_push_compensated(inputs):
    velocity_scale = (9.81 * inputs.nominal_base_height).sqrt()
    speed = inputs.base_lin_vel_scaled.square().sum(dim=1).sqrt()
    injected = inputs.push_vec_public.square().sum(dim=1).sqrt() / velocity_scale
    residual = speed - injected
    residual = 0.5 * (residual + residual.square().sqrt())
    linear_speed = residual / 0.08
    angular_speed = inputs.base_ang_vel_scaled.square().sum(dim=1).sqrt() / 0.5
    linear_core = 0.5 * (linear_speed + 1.0 - (linear_speed - 1.0).square().sqrt())
    angular_core = 0.5 * (angular_speed + 1.0 - (angular_speed - 1.0).square().sqrt())
    linear_penalty = 0.5 * linear_core.square() + linear_speed - linear_core
    angular_penalty = 0.5 * angular_core.square() + angular_speed - angular_core
    return 1.0 - 0.25 * (linear_penalty + angular_penalty)


def joint_pose(inputs):
    return inputs.joint_pos_error.square().mean(dim=1)


def action_rate(inputs):
    return inputs.action_delta.square().mean(dim=1)


def fall(inputs):
    return inputs.terminated_non_timeout.float()


def build_training_components():
    return TrainingComponents(
        name='push_standing_recovery_push_compensated_v5',
        reward_terms=[
            RewardTerm(name='upright_height', weight=3.0, fn=upright_height),
            RewardTerm(name='stationary_push_compensated', weight=1.0, fn=stationary_push_compensated),
            RewardTerm(name='joint_pose', weight=-0.1, fn=joint_pose),
            RewardTerm(name='action_rate', weight=-0.01, fn=action_rate),
            RewardTerm(name='fall', weight=-20.0, fn=fall),
        ],
        rationale='Closest prior candidate: push_standing_recovery_push_compensated_v4, whose damping term failed the authoring component check because it relied on a shape-expansion op the task fixture cannot execute. Hypothesis unchanged: the planar velocity increment injected by the task controller is an external perturbation rather than a policy outcome, so damping it charges the policy for motion it cannot control, while subtracting the already-applied increment leaves only residual, policy-generated motion damped. Exact code-level difference: the damping term now performs the whole compensation with elementwise arithmetic and dim-1 reductions over the public fields only, with no shape expansion and no indexing, and it expresses the soft hinge damping core as the minimum of the normalized speed and one through elementary operations. Residual motion after a pulse is still damped on subsequent steps, and every other term, weight and mechanism is unchanged. Expected effect: less reward noise on pulse steps and stronger credit assignment for post-pulse recovery and the fresh stable dwell.',
    )
