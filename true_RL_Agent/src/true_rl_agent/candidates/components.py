"""Controlled candidate training component hooks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from true_rl_agent.candidates.schema import CandidateChangeSet


@dataclass(frozen=True)
class RewardInputs:
    """Public tensors/vectors exposed to candidate training callables."""

    base_height: Any
    roll: Any
    pitch: Any
    base_pos_xy: Any
    base_lin_vel: Any
    base_ang_vel: Any
    projected_gravity: Any
    commanded_base_vel: Any
    joint_pos: Any
    joint_vel: Any
    joint_pos_error: Any
    last_action: Any
    prev_action: Any
    action_delta: Any
    push_active: Any
    push_vec_public: Any
    progress_fraction: Any

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        return tuple(cls.__dataclass_fields__)


@dataclass(frozen=True)
class RewardTerm:
    name: str
    weight: float
    fn: Callable[[RewardInputs], Any]
    clip_min: float | None = None
    clip_max: float | None = None
    description: str = ""


@dataclass(frozen=True)
class DiagnosticTerm:
    name: str
    fn: Callable[[RewardInputs], Any]
    description: str = ""


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
            "terms": {
                "reward_terms": [_term_public_summary(term) for term in self.reward_terms],
                "diagnostics": [_term_public_summary(term) for term in self.diagnostics],
                "schedules": [_term_public_summary(term) for term in self.schedules],
            },
            "metadata": dict(self.metadata),
            "rationale": self.rationale,
        }


def _term_public_summary(term: Any) -> dict[str, Any]:
    if isinstance(term, RewardTerm):
        return {
            "name": term.name,
            "kind": "RewardTerm",
            "weight": float(term.weight),
            "clip_min": term.clip_min,
            "clip_max": term.clip_max,
            "description": term.description,
        }
    if isinstance(term, DiagnosticTerm):
        return {
            "name": term.name,
            "kind": "DiagnosticTerm",
            "description": term.description,
        }
    if isinstance(term, ScheduleTerm):
        return {
            "name": term.name,
            "kind": "ScheduleTerm",
            "key": term.key,
            "value": term.value,
        }
    return {"name": getattr(term, "name", "<unknown>"), "kind": type(term).__name__}
