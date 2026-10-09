"""Task-owned component smoke fixture contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

@dataclass(frozen=True)
class ComponentSmokeFixture:
    """Task-owned inputs for authoring-time component smoke checks."""

    task_id: str
    reward_inputs: Any
    expected_reward_output_shape: tuple[int, ...]
    public_summary: Mapping[str, Any] = field(default_factory=dict)


class ComponentSmokeAdapter(Protocol):
    """Optional task-owned smoke fixture provider."""

    task_id: str

    @property
    def candidate_capability_ids(self) -> frozenset[str]:
        """Capabilities that this task can exercise before materialization."""
        ...

    def build_component_smoke_fixture(self) -> ComponentSmokeFixture | None:
        """Return a public-safe fixture, or None to skip smoke for this task."""
        ...


@dataclass(frozen=True)
class NeutralComponentSmokeAdapter:
    """Default no-op smoke adapter for tasks without a fixture contract."""

    task_id: str

    @property
    def candidate_capability_ids(self) -> frozenset[str]:
        return frozenset()

    def build_component_smoke_fixture(self) -> None:
        return None
