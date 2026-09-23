'''Morphology-scaled standing recovery reward rates with a lateral brace and a
one-sided lateral tilt-growth brake.

Stance is scored by a deadbanded, non-saturating potential over relative base
height and tilt. A lateral brace penalizes sideways tilt and lateral base speed.
A lateral_tilt_decay term additionally charges only the growing phase of the
sideways tilt, using roll times the morphology-scaled world roll rate past the
same roll deadband, so a decaying corrective roll is never penalized. Joint and
action penalties use means so changing joint count does not silently multiply
their weight.
'''

from true_rl_agent.candidates.components import RewardTerm, TrainingComponents


def upright_recovery(inputs):
    height_excess = ((inputs.base_height_relative - 1.0).abs() - 0.05).clamp(min=0.0)
    tilt = inputs.roll.square() + inputs.pitch.square()
    tilt_excess = (tilt - 0.01).clamp(min=0.0)
    penalty = (
        5.0 * height_excess
        + height_excess.square()
        + 2.0 * tilt_excess
        + 2.0 * tilt_excess.square()
    )
    return 1.0 - penalty


def lateral_brace(inputs):
    roll_excess = (inputs.roll.abs() - 0.10).clamp(min=0.0)
    lateral_speed = inputs.base_lin_vel_scaled[:, 1].abs()
    return -(roll_excess + 0.5 * lateral_speed)


def lateral_tilt_decay(inputs):
    roll = inputs.roll
    level_gate = (roll.abs() - 0.10).clamp(min=0.0, max=0.10) / 0.10
    tilt_growth = (roll * inputs.base_ang_vel_scaled[:, 0]).clamp(min=0.0)
    return -(level_gate * tilt_growth)


def stationary_v3(inputs):
    planar_speed = inputs.base_lin_vel_scaled[:, :2].square().sum(dim=1).sqrt() / 0.08
    angular_speed = inputs.base_ang_vel_scaled.square().sum(dim=1).sqrt() / 0.5
    planar_core = planar_speed.clamp(max=1.0)
    angular_core = angular_speed.clamp(max=1.0)
    planar_penalty = 0.5 * planar_core.square() + planar_speed - planar_core
    angular_penalty = 0.5 * angular_core.square() + angular_speed - angular_core
    return 1.0 - 0.25 * (planar_penalty + angular_penalty)


def joint_pose(inputs):
    return inputs.joint_pos_error.square().mean(dim=1)


def action_rate(inputs):
    return inputs.action_delta.square().mean(dim=1)


def fall(inputs):
    return inputs.terminated_non_timeout.float()


def build_training_components():
    return TrainingComponents(
        name="push_standing_recovery_tilt_decay_v1",
        reward_terms=[
            RewardTerm(
                name="upright_recovery",
                weight=3.0,
                fn=upright_recovery,
                clip_min=-4.0,
                clip_max=1.0,
                description="Deadbanded non-saturating stance potential over relative base height and tilt.",
            ),
            RewardTerm(
                name="lateral_brace",
                weight=-1.0,
                fn=lateral_brace,
                clip_min=-2.0,
                clip_max=0.0,
                description="Penalty on sideways tilt beyond a deadband and lateral base speed.",
            ),
            RewardTerm(
                name="lateral_tilt_decay",
                weight=-1.5,
                fn=lateral_tilt_decay,
                clip_min=-1.0,
                clip_max=0.0,
                description="One-sided penalty on growing sideways tilt past the lateral brace deadband.",
            ),
            RewardTerm(name="stationary_v3", weight=1.0, fn=stationary_v3),
            RewardTerm(name="joint_pose", weight=-0.1, fn=joint_pose),
            RewardTerm(name="action_rate", weight=-0.01, fn=action_rate),
            RewardTerm(name="fall", weight=-20.0, fn=fall),
        ],
        diagnostics=[],
        schedules=[],
        rationale=(
            "Closest prior program: the accepted current best push_recovery_lateral_brace_v1 "
            "(public_dev mean 0.9115; its parent push_recovery_nonsaturating_stance_v1 was 0.8802). "
            "Its public_dev breakdown is 175 success_recovered, 10 fall_side and 7 not_recovered: "
            "after the lateral brace landed, height_collapse and fall_forward vanished completely, "
            "so every residual toppling failure is a sideways roll collapse, concentrated in the "
            "high (37/48) and repeated (42/48) strata. Negative evidence that is not repeated: the "
            "sibling push_recovery_side_braced_stance_v1 (0.8385, height_collapse dominated) and the "
            "earlier push_recovery_tilt_aware_stance_v2 and push_recovery_vel_damped_stance_v1 all "
            "strengthened level or speed penalties in the displacement regime that contains the "
            "recovery transient. Sole mechanism change: one new RewardTerm lateral_tilt_decay is "
            "added; the anchor's upright_recovery kernel, lateral_brace, stationary_v3, joint_pose, "
            "action_rate and fall callables, their weights, signs, clips, term order, and the empty "
            "schedule and diagnostic lists are preserved exactly. lateral_tilt_decay computes "
            "roll_rate = base_ang_vel_scaled[:, 0], the morphology-scaled world-frame roll rate, and "
            "level_gate = clamp((abs(roll) - 0.10) / 0.10, 0, 1), which reuses the accepted brace's "
            "0.10 rad roll deadband, and returns -(level_gate * clamp(roll * roll_rate, 0, inf)). It "
            "is the one-sided derivative of the lateral tilt energy: it charges a state only while "
            "the sideways tilt magnitude is growing, it is exactly zero while the base sits inside "
            "the nominal stance band, and it is exactly zero whenever roll * roll_rate is negative, "
            "that is while the base is already rolling back toward level. The anchor prices the "
            "lateral state with three level quantities, namely the squared tilt branch inside "
            "upright_recovery, the linear roll brace, and the symmetric lateral speed term; none of "
            "them separates a still-growing topple from an already-decaying one, so at the tipping "
            "onset the only lateral gradient available scales with the current roll rather than with "
            "the diverging rate that decides whether the fresh stable dwell is reached before the "
            "base passes the point of no return. The new term supplies exactly that missing "
            "one-sided rate gradient, which shortens the window in which a repeated-pulse sequence "
            "can re-topple a base that is still rolling, and it touches no translational channel, so "
            "it cannot brake the lateral stepping strategy that the accepted speed terms already "
            "bound, which is the failure mode of the refuted speed-damping route. It is also "
            "structurally distinct from the excluded single-component card RewardTerm:roll_brace, "
            "which depends on roll alone as a level brace: this mechanism pairs roll with the "
            "world-frame roll rate and is one-sided in the growth direction, with clip [-1.0, 0.0] "
            "at weight -1.5 bounding the added contribution to 1.5, the same order as the accepted "
            "lateral_brace, so it shapes the tipping onset rather than competing with the -20 fall "
            "penalty. The success definition, the pulse schedule, and all task-owned, "
            "evaluation-owned, checkpoint-owned and budget-owned surfaces are untouched."
        ),
    )
