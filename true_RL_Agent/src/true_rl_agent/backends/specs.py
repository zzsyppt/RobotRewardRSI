"""Backend capability declarations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class BackendSpec:
    """Static backend capability contract."""

    backend_id: str
    supported_tasks: frozenset[str]
    supported_policy_initializations: frozenset[str]
    curriculum_capabilities: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    supports_baseline_eval: bool = False

    def supports(self, *, task_id: str, policy_initialization: str) -> bool:
        return (
            task_id in self.supported_tasks
            and policy_initialization in self.supported_policy_initializations
        )

    def curriculum_capability_effective_now(self, capability_id: str) -> bool:
        capability = self.curriculum_capabilities.get(capability_id, {})
        return bool(capability.get("effective_now", False))

    def to_public_curriculum_capabilities(self) -> dict[str, dict[str, Any]]:
        return {key: dict(value) for key, value in self.curriculum_capabilities.items()}
