"""Frozen curriculum contract for the first project version."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


ALLOWED_CURRICULUM_MODES = frozenset({"frozen", "runner_owned", "task_default"})


@dataclass(frozen=True)
class CurriculumPlan:
    """Curriculum lineage state without LLM-controlled schedule optimization."""

    mode: str
    llm_controlled: bool = False
    lineage: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode not in ALLOWED_CURRICULUM_MODES:
            raise ValueError(f"unsupported curriculum mode: {self.mode}")
        if self.llm_controlled:
            raise ValueError("first-version curriculum must not be LLM controlled")

    def to_public_dict(self) -> dict[str, Any]:
        return {"mode": self.mode, "llm_controlled": self.llm_controlled, "lineage": dict(self.lineage)}

