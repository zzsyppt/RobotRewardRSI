"""Mechanism evidence gate for anti-tuning checks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from true_rl_agent.candidates.schema import (
    CandidateChangeSet,
    CandidateSurfaceAssessment,
    CandidateSurfacePolicy,
)

MECHANISM_ATTRIBUTION_SCHEMA_VERSION = (
    "true_rl_agent.mechanism_attribution.v2"
)


@dataclass(frozen=True)
class RuntimeActivationEvidence:
    # None is unmeasured. A measured zero must survive all public projections.
    reward_term_calls: int | None = None
    diagnostic_calls: int | None = None
    nonzero_reward_contribution: bool | None = None
    reward_output_stats: Mapping[str, Any] = field(default_factory=dict)
    diagnostic_outputs: Mapping[str, Any] = field(default_factory=dict)
    component_activation_records: tuple[Mapping[str, Any], ...] = ()
    feature_activation_records: tuple[str, ...] = ()
    capability_activation_records: tuple[str, ...] = ()
    step_calls: int | None = None
    reward_term_count: int | None = None
    diagnostic_term_count: int | None = None
    training_sampling: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for key in ("reward_term_calls", "diagnostic_calls", "step_calls", "reward_term_count", "diagnostic_term_count"):
            value = getattr(self, key)
            if value is not None:
                _nonnegative_integer(value, key)
        if self.nonzero_reward_contribution is not None and not isinstance(self.nonzero_reward_contribution, bool):
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
        if not isinstance(self.training_sampling, Mapping):
            raise ValueError("training_sampling must be an object")
        if self.training_sampling:
            from true_rl_agent.evaluation.public_projection import project_public_training_sampling
            object.__setattr__(self, "training_sampling", project_public_training_sampling(self.training_sampling))
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

    def measurement_summary(self) -> dict[str, Any]:
        """Available means a record exists; None counters are still unmeasured."""
        measured = {
            key: getattr(self, key)
            for key in ("step_calls", "reward_term_count", "diagnostic_term_count", "reward_term_calls", "diagnostic_calls", "nonzero_reward_contribution")
        }
        available = any(value is not None for value in measured.values()) or any((
            self.reward_output_stats, self.diagnostic_outputs, self.component_activation_records,
            self.feature_activation_records, self.capability_activation_records,
        ))
        result = {"status": "available", **measured} if available else {"status": "not_available"}
        if self.training_sampling:
            result["training_sampling"] = dict(self.training_sampling)
        return result

    def reward_execution_state(self) -> str:
        """Compact observation for history, without claiming causal effect."""
        if self.nonzero_reward_contribution is True:
            return "executed_nonzero"
        if self.reward_term_calls is not None and self.reward_term_calls > 0:
            return (
                "executed_zero" if self.nonzero_reward_contribution is False
                else "executed_value_unknown"
            )
        if self.step_calls == 0:
            return "not_executed"
        if self.reward_term_count == 0:
            return "no_reward_terms"
        # Older receipts zero-filled missing counters without recording whether
        # the adapter stepped or any reward terms existed.
        if (
            self.reward_term_calls == 0
            and self.step_calls is not None and self.step_calls > 0
            and self.reward_term_count is not None and self.reward_term_count > 0
        ):
            return "terms_not_called"
        if self.measurement_summary()["status"] == "not_available":
            return "not_available"
        # A loaded capability, diagnostic call, or false contribution without
        # reward-call evidence cannot prove that a reward term executed.
        return "unknown"

    def to_public_dict(self) -> dict[str, Any]:
        measured = self.measurement_summary()
        if measured["status"] == "not_available":
            return measured
        return {
            **measured,
            "reward_output_stats": dict(self.reward_output_stats),
            "diagnostic_outputs": dict(self.diagnostic_outputs),
            "component_activation_records": [
                dict(record) for record in self.component_activation_records
            ],
            "feature_activation_records": list(self.feature_activation_records),
            "capability_activation_records": list(self.capability_activation_records),
        }

    @classmethod
    def from_train_summary(cls, summary: Mapping[str, Any]) -> "RuntimeActivationEvidence":
        """One receipt-owned path shared by live rounds and generation replay."""
        from true_rl_agent.evaluation.public_projection import project_public_training_sampling
        evidence = cls.from_mapping(summary.get("runtime_activation_summary"))
        training = summary.get("training")
        sampled = training.get("numeric_evidence") if isinstance(training, Mapping) else None
        if sampled is None and evidence.training_sampling:
            sampled = evidence.training_sampling
        return replace(evidence, training_sampling=project_public_training_sampling(sampled))

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
            reward_term_calls=value.get("reward_term_calls"),
            diagnostic_calls=value.get("diagnostic_calls"),
            nonzero_reward_contribution=value.get("nonzero_reward_contribution"),
            step_calls=value.get("step_calls"),
            reward_term_count=value.get("reward_term_count"),
            diagnostic_term_count=value.get("diagnostic_term_count"),
            training_sampling=value.get("training_sampling", {}),
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
        if evidence.reward_term_calls is None or evidence.reward_term_calls <= 0 or not evidence.nonzero_reward_contribution:
            return MechanismDecision("attribution_unclear", "runner_activation_missing")
        return MechanismDecision("pass", "runner_activation_observed")


def build_mechanism_attribution_summary(
    *,
    runtime_activation_gate: MechanismDecision,
    ablation_summary: Mapping[str, Any],
    authoring_mechanism_delta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if runtime_activation_gate.status != "pass":
        reason = runtime_activation_gate.reason
        method = "runtime_activation_gate"
    else:
        reason = "shadow_ablation_not_run"
        method = str(ablation_summary.get("method") or "none")
    result = {
        "schema_version": MECHANISM_ATTRIBUTION_SCHEMA_VERSION,
        "status": "not_attributed",
        "reason": reason,
        "causal_claim": False,
        "runtime_activation_gate": runtime_activation_gate.to_public_dict(),
        "attribution_method": method,
        "shadow_ablation_status": str(ablation_summary.get("status") or "not_run"),
    }
    if authoring_mechanism_delta is not None:
        result["authoring_mechanism_delta"] = dict(authoring_mechanism_delta)
    return result


def build_knob_only_shadow_candidate(
    change_set: CandidateChangeSet,
    *,
    surface_policy: CandidateSurfacePolicy,
) -> CandidateChangeSet:
    return surface_policy.parameter_only_subset(change_set)
