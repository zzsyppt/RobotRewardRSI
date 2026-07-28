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


@dataclass(frozen=True)
class CandidateSurfacePolicy:
    """Task-supplied component taxonomy with task-neutral classifications."""

    parameter_only_components: frozenset[str] = frozenset()
    diagnostic_components: frozenset[str] = frozenset()
    engineering_components: frozenset[str] = frozenset()
    code_level_components: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        groups = (
            self.parameter_only_components,
            self.diagnostic_components,
            self.engineering_components,
            self.code_level_components,
        )
        for index, group in enumerate(groups):
            if any(group & other for other in groups[index + 1 :]):
                raise ValueError("candidate surface component groups must be disjoint")

    def classify(self, change_set: CandidateChangeSet) -> CandidateSurfaceAssessment:
        components = change_set.changed_components
        if not components:
            return CandidateSurfaceAssessment("no_code_change", False, "no_candidate_changes")
        if components <= self.parameter_only_components:
            return CandidateSurfaceAssessment("parameter_only", False, "parameter_only_candidate")
        if components <= self.diagnostic_components:
            return CandidateSurfaceAssessment("diagnostic_only", False, "diagnostic_only_candidate")
        if components <= self.engineering_components:
            return CandidateSurfaceAssessment("engineering_only", False, "engineering_only_candidate")
        supported_components = frozenset().union(
            self.parameter_only_components,
            self.diagnostic_components,
            self.engineering_components,
            self.code_level_components,
        )
        if components <= supported_components and components & self.code_level_components:
            return CandidateSurfaceAssessment("code_level", True, "code_level_candidate")
        return CandidateSurfaceAssessment("unsupported", False, "unsupported_candidate_surface")

    def parameter_only_subset(self, change_set: CandidateChangeSet) -> CandidateChangeSet:
        return CandidateChangeSet(
            frozenset(change_set.changed_components & self.parameter_only_components)
        )
