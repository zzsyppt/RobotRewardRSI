"""Provider interfaces for replay and dry-run candidate supply."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Protocol

from true_rl_agent.candidates.schema import CandidatePackageRef


class CandidateProvider(Protocol):
    def next_candidate(self, *, public_context: dict[str, Any]) -> CandidatePackageRef:
        raise NotImplementedError


@dataclass
class ReplayProvider:
    candidates: Iterable[CandidatePackageRef]

    def __post_init__(self) -> None:
        self._remaining = list(self.candidates)

    def next_candidate(self, *, public_context: dict[str, Any]) -> CandidatePackageRef:
        _ = public_context
        if not self._remaining:
            raise StopIteration("replay provider has no remaining candidates")
        return self._remaining.pop(0)
