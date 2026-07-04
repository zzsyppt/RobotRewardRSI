"""Candidate package and code-change surface contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class CandidateChangeSet:
    """Structured summary of candidate-owned changes."""

    changed_components: frozenset[str]


@dataclass(frozen=True)
class CodeChangeManifest:
    """Candidate-declared code-change manifest."""

    change_set: CandidateChangeSet
    rationale: str


@dataclass(frozen=True)
class CandidatePackageRef:
    """Reference to a candidate package without loading arbitrary code."""

    candidate_id: str
    path: str
    change_set: CandidateChangeSet
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "changed_components": sorted(self.change_set.changed_components),
        }


@dataclass(frozen=True)
class CandidateSurfaceAssessment:
    classification: str
    eligible_for_policy_accept: bool
    reason: str

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "classification": self.classification,
            "eligible_for_policy_accept": self.eligible_for_policy_accept,
            "reason": self.reason,
        }


class CandidateSurfacePolicy:
    """Classify candidate changes before policy acceptance decisions."""

    PARAMETER_ONLY_COMPONENTS = frozenset(
        {
            "reward_weight",
            "reward_weight_scale",
            "terrain_range",
            "height_range",
            "stair_height_range",
            "ppo_hyperparameter",
            "eval_threshold",
            "budget",
            "seed",
            "schedule",
            "single_stage_numeric_schedule",
        }
    )
    DIAGNOSTIC_COMPONENTS = frozenset({"diagnostic"})
    ENGINEERING_COMPONENTS = frozenset({"failure_classifier", "logging", "public_projection"})
    CODE_LEVEL_COMPONENTS = frozenset({"reward_callable", "control_logic", "recovery_logic"})

    def classify(self, change_set: CandidateChangeSet) -> CandidateSurfaceAssessment:
        components = change_set.changed_components
        if not components:
            return CandidateSurfaceAssessment("no_code_change", False, "no_candidate_changes")
        if components <= self.PARAMETER_ONLY_COMPONENTS:
            return CandidateSurfaceAssessment("parameter_only", False, "parameter_only_candidate")
        if components <= self.DIAGNOSTIC_COMPONENTS:
            return CandidateSurfaceAssessment("diagnostic_only", False, "diagnostic_only_candidate")
        if components <= self.ENGINEERING_COMPONENTS:
            return CandidateSurfaceAssessment("engineering_only", False, "engineering_only_candidate")
        if components & self.CODE_LEVEL_COMPONENTS:
            return CandidateSurfaceAssessment("code_level", True, "code_level_candidate")
        return CandidateSurfaceAssessment("unsupported", False, "unsupported_candidate_surface")
