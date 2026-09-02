"""Callable Phase 8E candidate used to validate the code-level loop."""

from __future__ import annotations

from humanoid_agent.code_sandbox.components import DiagnosticTerm, RewardInputs, RewardTerm, ScheduleTerm, TrainingComponents


def upright_survival_margin(inp: RewardInputs):
    """Reward height margin and upright orientation using public tensors only."""

    return inp.base_height - 0.5 * (inp.roll * inp.roll + inp.pitch * inp.pitch)


def delayed_push_ramp(progress: float) -> dict[str, float]:
    """Ramp public push magnitude after early balance has time to form."""

    if progress < 0.25:
        scale = 0.0
    elif progress > 0.75:
        scale = 1.0
    else:
        scale = (progress - 0.25) / 0.5
    return {"push_magnitude_scale": scale}


def collapse_vs_recovery_margin(inp: RewardInputs) -> dict[str, object]:
    """Public numeric diagnostic: summarize height and upright margin."""

    return {
        "height_margin": inp.base_height - 0.45,
        "upright_penalty": inp.roll * inp.roll + inp.pitch * inp.pitch,
    }


def build_training_components(task_spec, base_env_cfg, base_agent_cfg) -> TrainingComponents:
    _ = (task_spec, base_env_cfg, base_agent_cfg)
    return TrainingComponents(
        name="phase8e_declarative_curriculum_candidate",
        reward_terms=[
            RewardTerm(
                name="upright_survival_margin",
                fn=upright_survival_margin,
                weight=1.0,
                clip_min=-2.0,
                clip_max=2.0,
                description="height and orientation margin during early learning",
            )
        ],
        schedules=[
            ScheduleTerm(
                name="delayed_push_ramp",
                fn=delayed_push_ramp,
                allowed_keys=("push_magnitude_scale",),
                description="public push magnitude scale ramp from 0.25 to 0.75 progress",
            )
        ],
        diagnostics=[
            DiagnosticTerm(
                name="collapse_vs_recovery_margin",
                fn=collapse_vs_recovery_margin,
                description="Track collapse and recovery margins using public training tensors only.",
            )
        ],
        metadata={
            "recipe_family": "standing_push_recovery_curriculum",
            "runner_controlled": True,
            "eval_interface_changed": False,
        },
        rationale="Use a direct reward callable for stance margin and a pre-evaluated push curriculum schedule.",
    )
