"""Comparable public-accept statistics, with no claim about unexecuted gates."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from true_rl_agent.evaluation.acceptance_protocol import AcceptanceProtocol, validated_score_samples
from true_rl_agent.evaluation.score_stats import validate_score_stats


@dataclass(frozen=True)
class PublicAcceptScores:
    score_stats: Mapping[str, Any]
    score_samples: tuple[float, ...] = ()
    guard_score_stats: Mapping[str, Any] = field(default_factory=dict)
    guard_score_samples: tuple[float, ...] = ()
    named_guard_evidence: Mapping[str, Any] = field(default_factory=dict)

    @property
    def score_mean(self):
        return self.score_stats.get("mean")

    @property
    def standard_error(self):
        return self.score_stats.get("standard_error")

    @property
    def guard_score_mean(self):
        return self.guard_score_stats.get("mean")

    def to_dict(self):
        return {"score_stats": dict(self.score_stats), "score_samples": list(self.score_samples),
            "guard_score_stats": dict(self.guard_score_stats),
            "guard_score_samples": list(self.guard_score_samples),
            "named_guard_evidence": dict(self.named_guard_evidence)}

    @classmethod
    def from_mapping(cls, value, *, protocol: AcceptanceProtocol, eval_protocol_hash: str):
        if not isinstance(value, Mapping) or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("public-accept score fields mismatch")

        def validate(stats, samples, *, paired):
            check = validate_score_stats(stats, minimum_count=protocol.min_score_count,
                require_std=True, require_standard_error=protocol.min_score_count >= 2,
                require_protocol_hash=True)
            if not check.valid or check.eval_protocol_hash != eval_protocol_hash:
                raise ValueError("public-accept statistics are incomplete or differently bound")
            if paired:
                return validated_score_samples(samples, expected_mean=check.mean,
                    protocol=protocol, field_name="public_accept_scores")
            if samples:
                raise ValueError("unpaired score evidence must not declare paired samples")
            return ()

        samples = validate(value["score_stats"], value["score_samples"],
            paired=protocol.requires_paired_score_samples)
        guard_samples = ()
        if protocol.uses_guard_score:
            guard_samples = validate(value["guard_score_stats"], value["guard_score_samples"],
                paired=protocol.requires_paired_guard_score_samples)
        elif value["guard_score_stats"] or value["guard_score_samples"]:
            raise ValueError("public-accept evidence contains an undeclared guard")
        named = value["named_guard_evidence"]
        if not isinstance(named, Mapping) or set(named) != {g.name for g in protocol.named_guards}:
            raise ValueError("public-accept named guard set differs from the protocol")
        for item in named.values():
            validate(item["score_stats"], item["score_samples"], paired=True)
        return cls(value["score_stats"], samples, value["guard_score_stats"], guard_samples, named)
