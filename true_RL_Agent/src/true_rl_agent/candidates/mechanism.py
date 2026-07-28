"""Mechanism evidence gate for anti-tuning checks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from true_rl_agent.candidates.schema import (
    CandidateChangeSet,
    CandidateSurfaceAssessment,
    CandidateSurfacePolicy,
)


@dataclass(frozen=True)
class RuntimeActivationEvidence:
    reward_term_calls: int = 0
    diagnostic_calls: int = 0
    nonzero_reward_contribution: bool = False
    reward_output_stats: Mapping[str, Any] = field(default_factory=dict)
    diagnostic_outputs: Mapping[str, Any] = field(default_factory=dict)
    component_activation_records: tuple[Mapping[str, Any], ...] = ()
    feature_activation_records: tuple[str, ...] = ()
    capability_activation_records: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _nonnegative_integer(self.reward_term_calls, "reward_term_calls")
        _nonnegative_integer(self.diagnostic_calls, "diagnostic_calls")
        if not isinstance(self.nonzero_reward_contribution, bool):
            raise ValueError("nonzero_reward_contribution must be a boolean")
        if not isinstance(self.reward_output_stats, Mapping):
            raise ValueError("reward_output_stats must be an object")
        if not isinstance(self.diagnostic_outputs, Mapping):
            raise ValueError("diagnostic_outputs must be an object")
        _mapping_records(
            self.component_activation_records,
            "component_activation_records",
        )
        _string_records(
            self.feature_activation_records,
            "feature_activation_records",
        )
        _string_records(
            self.capability_activation_records,
            "capability_activation_records",
        )
        object.__setattr__(self, "reward_output_stats", dict(self.reward_output_stats))
        object.__setattr__(self, "diagnostic_outputs", dict(self.diagnostic_outputs))
        object.__setattr__(
            self,
            "component_activation_records",
            tuple(dict(record) for record in self.component_activation_records),
        )
        object.__setattr__(
            self,
            "feature_activation_records",
            tuple(self.feature_activation_records),
        )
        object.__setattr__(
            self,
            "capability_activation_records",
            tuple(self.capability_activation_records),
        )

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "reward_term_calls": self.reward_term_calls,
            "diagnostic_calls": self.diagnostic_calls,
            "nonzero_reward_contribution": self.nonzero_reward_contribution,
            "reward_output_stats": dict(self.reward_output_stats),
            "diagnostic_outputs": dict(self.diagnostic_outputs),
            "component_activation_records": [
                dict(record) for record in self.component_activation_records
            ],
            "feature_activation_records": list(self.feature_activation_records),
            "capability_activation_records": list(self.capability_activation_records),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "RuntimeActivationEvidence":
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise TypeError("runtime activation evidence must be an object")
        reward_output_stats = value.get("reward_output_stats", {})
        diagnostic_outputs = value.get("diagnostic_outputs", {})
        component_activation_records = value.get("component_activation_records", ())
        feature_activation_records = value.get("feature_activation_records", ())
        capability_activation_records = value.get("capability_activation_records", ())
        return cls(
            reward_term_calls=_nonnegative_integer(
                value.get("reward_term_calls", 0),
                "reward_term_calls",
            ),
            diagnostic_calls=_nonnegative_integer(
                value.get("diagnostic_calls", 0),
                "diagnostic_calls",
            ),
            nonzero_reward_contribution=_boolean(
                value.get("nonzero_reward_contribution", False),
                "nonzero_reward_contribution",
            ),
            reward_output_stats=_mapping(reward_output_stats, "reward_output_stats"),
            diagnostic_outputs=_mapping(diagnostic_outputs, "diagnostic_outputs"),
            component_activation_records=_mapping_records(
                component_activation_records,
                "component_activation_records",
            ),
            feature_activation_records=_string_records(
                feature_activation_records,
                "feature_activation_records",
            ),
            capability_activation_records=_string_records(
                capability_activation_records,
                "capability_activation_records",
            ),
        )


def _nonnegative_integer(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _boolean(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean")
    return value


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return value


def _mapping_records(value: Any, field_name: str) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, list | tuple) or any(
        not isinstance(record, Mapping) for record in value
    ):
        raise ValueError(f"{field_name} must contain only objects")
    return tuple(value)


def _string_records(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list | tuple) or any(
        not isinstance(record, str) or not record for record in value
    ):
        raise ValueError(f"{field_name} must contain only non-empty strings")
    return tuple(value)


@dataclass(frozen=True)
class MechanismDecision:
    status: str
    reason: str

    def to_public_dict(self) -> dict[str, str]:
        return {"status": self.status, "reason": self.reason}


class MechanismEvidenceGate:
    def evaluate(
        self,
        *,
        surface: CandidateSurfaceAssessment,
        evidence: RuntimeActivationEvidence,
    ) -> MechanismDecision:
        if surface.classification == "parameter_only":
            return MechanismDecision("reject", "parameter_only_candidate")
        if surface.classification in {"diagnostic_only", "engineering_only"}:
            return MechanismDecision("record_only", f"{surface.classification}_candidate")
        if surface.classification != "code_level":
            return MechanismDecision("reject", surface.reason)
        if evidence.reward_term_calls <= 0 or not evidence.nonzero_reward_contribution:
            return MechanismDecision("attribution_unclear", "runner_activation_missing")
        return MechanismDecision("pass", "runner_activation_observed")


def build_mechanism_attribution_summary(
    *,
    runtime_activation_gate: MechanismDecision,
    ablation_summary: Mapping[str, Any],
) -> dict[str, Any]:
    if runtime_activation_gate.status != "pass":
        reason = runtime_activation_gate.reason
        method = "runtime_activation_gate"
    else:
        reason = "shadow_ablation_not_run"
        method = str(ablation_summary.get("method") or "none")
    return {
        "schema_version": "true_rl_agent.mechanism_attribution.v1",
        "status": "not_attributed",
        "reason": reason,
        "causal_claim": False,
        "runtime_activation_gate": runtime_activation_gate.to_public_dict(),
        "attribution_method": method,
        "shadow_ablation_status": str(ablation_summary.get("status") or "not_run"),
    }


def build_knob_only_shadow_candidate(
    change_set: CandidateChangeSet,
    *,
    surface_policy: CandidateSurfacePolicy,
) -> CandidateChangeSet:
    return surface_policy.parameter_only_subset(change_set)
