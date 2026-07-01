"""Deterministic no-op package for Phase 8E provider ingestion."""

from __future__ import annotations

from humanoid_agent.code_sandbox.components import TrainingComponents


def build_training_components(task_spec, base_env_cfg, base_agent_cfg) -> TrainingComponents:
    _ = (task_spec, base_env_cfg, base_agent_cfg)
    return TrainingComponents(
        name="phase8e_provider_dry_run_noop",
        rationale="No-op package for provider ingestion plumbing.",
    )
