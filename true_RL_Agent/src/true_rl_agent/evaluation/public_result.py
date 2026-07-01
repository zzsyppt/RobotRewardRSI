"""Public eval result projection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class PublicEvalResult:
    """Candidate-visible eval summary."""

    status: str
    public_score: float
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "public_score": self.public_score,
            "metrics": dict(self.metrics),
        }

