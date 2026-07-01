"""Backend capability declarations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BackendSpec:
    """Static backend capability contract."""

    backend_id: str
    supported_tasks: frozenset[str]
    supported_policy_initializations: frozenset[str]

    def supports(self, *, task_id: str, policy_initialization: str) -> bool:
        return (
            task_id in self.supported_tasks
            and policy_initialization in self.supported_policy_initializations
        )

