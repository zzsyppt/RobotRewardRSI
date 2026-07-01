"""Task data contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

from true_rl_agent.curriculum.schema import CurriculumPlan


@dataclass(frozen=True)
class TaskInvariantSpec:
    """Runner-owned task semantics that candidates must not weaken."""

    values: Mapping[str, Any] = field(default_factory=dict)

    def assert_compatible_with(self, other: "TaskInvariantSpec") -> None:
        if dict(self.values) != dict(other.values):
            raise ValueError("task invariant mismatch")


@dataclass(frozen=True)
class TaskSpec:
    """Public task identity and runner-owned invariant contract."""

    task_id: str
    name: str
    supported_policy_initializations: frozenset[str]
    invariants: TaskInvariantSpec
    fixed_train_protocol: Mapping[str, Any] = field(default_factory=dict)
    fixed_eval_protocol: Mapping[str, Any] = field(default_factory=dict)
    default_curriculum: CurriculumPlan = field(default_factory=lambda: CurriculumPlan(mode="task_default"))

    def supports_policy_initialization(self, kind: str) -> bool:
        return kind in self.supported_policy_initializations


class TaskPlugin(ABC):
    """Task semantic boundary for AgentCore."""

    @property
    @abstractmethod
    def task_spec(self) -> TaskSpec:
        raise NotImplementedError
