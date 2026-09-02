"""Policy initialization contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

PolicyInitializationKind = Literal["random", "warm_start"]


@dataclass(frozen=True)
class BaselinePolicyRef:
    """Reference to a compatible baseline policy used only as a training start."""

    baseline_id: str
    task_id: str
    checkpoint: Any
    compatibility_tags: frozenset[str] = frozenset()
    baseline_manifest_hash: str | None = None


@dataclass(frozen=True)
class PolicyInitialization:
    """Runner-owned policy initialization mode for a training job."""

    kind: PolicyInitializationKind
    baseline: BaselinePolicyRef | None = None

    @staticmethod
    def random() -> "PolicyInitialization":
        return PolicyInitialization(kind="random")

    @staticmethod
    def warm_start(baseline: BaselinePolicyRef) -> "PolicyInitialization":
        return PolicyInitialization(kind="warm_start", baseline=baseline)

    def __post_init__(self) -> None:
        if self.kind == "random" and self.baseline is not None:
            raise ValueError("random initialization cannot include a baseline")
        if self.kind == "warm_start" and self.baseline is None:
            raise ValueError("warm_start initialization requires a baseline")
