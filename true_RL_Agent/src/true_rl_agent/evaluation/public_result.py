"""Public eval result projection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from true_rl_agent.evaluation.public_projection import assert_public_projection_safe
from true_rl_agent.evaluation.result_contract import validate_public_eval_result_fields


@dataclass(frozen=True)
class PublicEvalResult:
    """Candidate-visible eval summary."""

    status: str
    public_score: float
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        fields = validate_public_eval_result_fields(
            status=self.status,
            public_score=self.public_score,
            metrics=self.metrics,
            field_name="public eval",
        )
        object.__setattr__(self, "status", fields.status)
        object.__setattr__(self, "public_score", fields.public_score)

    def to_public_dict(self) -> dict[str, Any]:
        data = {
            "status": self.status,
            "public_score": self.public_score,
            "metrics": dict(self.metrics),
        }
        assert_public_projection_safe(data)
        return data
