"""Task-specific public memory extension interface."""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence


class TaskMemoryAdapter(Protocol):
    """Compress public task experience into next-round public context."""

    task_id: str

    def build_public_summary(self, public_records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        """Return task-specific public context from public records only."""
        ...
