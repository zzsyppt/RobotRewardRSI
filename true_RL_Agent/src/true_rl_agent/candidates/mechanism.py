"""Mechanism evidence gate for anti-tuning checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

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
        reward_output_stats = value.get("reward_output_stats") or {}
        diagnostic_outputs = value.get("diagnostic_outputs") or {}
        component_activation_records = value.get("component_activation_records") or ()
        feature_activation_records = value.get("feature_activation_records") or ()
        capability_activation_records = value.get("capability_activation_records") or ()
        return cls(
            reward_term_calls=int(value.get("reward_term_calls", 0)),
            diagnostic_calls=int(value.get("diagnostic_calls", 0)),
            nonzero_reward_contribution=bool(value.get("nonzero_reward_contribution", False)),
            reward_output_stats=dict(reward_output_stats),
            diagnostic_outputs=dict(diagnostic_outputs),
            component_activation_records=tuple(
                dict(record)
                for record in component_activation_records
                if isinstance(record, Mapping)
            ),
            feature_activation_records=tuple(feature_activation_records),
            capability_activation_records=tuple(capability_activation_records),
        )


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
