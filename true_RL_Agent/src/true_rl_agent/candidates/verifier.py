"""Candidate verifier interface for replay and fake-round tests."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

from true_rl_agent.candidates.schema import CandidatePackageRef


@dataclass(frozen=True)
class VerificationResult:
    status: str
    reason: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_public_dict(self) -> dict[str, Any]:
        return {"status": self.status, "reason": self.reason, "details": dict(self.details)}


class CandidateVerifier(ABC):
    @abstractmethod
    def verify(self, candidate: CandidatePackageRef) -> VerificationResult:
        raise NotImplementedError
