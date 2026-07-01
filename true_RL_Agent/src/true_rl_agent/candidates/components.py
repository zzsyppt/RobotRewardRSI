"""Controlled candidate training component hooks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from true_rl_agent.candidates.schema import CandidateChangeSet


@dataclass(frozen=True)
class RewardTerm:
    name: str
    weight: float
    fn: Callable[[Any], Any]


@dataclass(frozen=True)
class DiagnosticTerm:
    name: str
    fn: Callable[[Any], Any]


@dataclass(frozen=True)
class ScheduleTerm:
    name: str
    key: str
    value: float


@dataclass(frozen=True)
class TrainingComponents:
    name: str
    reward_terms: list[RewardTerm] = field(default_factory=list)
    diagnostics: list[DiagnosticTerm] = field(default_factory=list)
    schedules: list[ScheduleTerm] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""

    def is_noop(self) -> bool:
        return not (self.reward_terms or self.diagnostics or self.schedules)

    def to_change_set(self) -> CandidateChangeSet:
        components: set[str] = set()
        if self.reward_terms:
            components.add("reward_callable")
        if self.diagnostics:
            components.add("diagnostic")
        for schedule in self.schedules:
            if schedule.key == "reward_weight_scale":
                components.add("reward_weight")
            else:
                components.add("schedule")
        return CandidateChangeSet(frozenset(components))

    def to_public_summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "is_noop": self.is_noop(),
            "effective_hook_counts": {
                "reward_terms": len(self.reward_terms),
                "diagnostics": len(self.diagnostics),
                "schedules": len(self.schedules),
            },
            "rationale": self.rationale,
        }
