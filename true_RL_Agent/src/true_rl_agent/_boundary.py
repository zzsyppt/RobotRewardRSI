"""Runtime boundary declarations for the independent project."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeBoundary:
    """Static project-boundary contract checked by tests and reviews."""

    package_name: str = "true_rl_agent"
    parent_project_runtime_imports_allowed: bool = False
    candidate_acceptance_may_be_tuning_only: bool = False
    llm_controls_curriculum_in_first_version: bool = False


RUNTIME_BOUNDARY = RuntimeBoundary()

