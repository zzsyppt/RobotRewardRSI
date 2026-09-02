"""No-op Phase 8A candidate components."""

from __future__ import annotations

from humanoid_agent.code_sandbox.components import TrainingComponents


def build_training_components(task_spec, base_env_cfg, base_agent_cfg) -> TrainingComponents:
    """Return an empty component set.

    The parameters are accepted to exercise the Phase 8A candidate interface.
    This demo intentionally does not mutate them and does not change training.
    """

    _ = (task_spec, base_env_cfg, base_agent_cfg)
    return TrainingComponents(
        name="phase8a_noop_candidate",
        rationale="No-op candidate used only to validate loader/runner plumbing.",
    )
