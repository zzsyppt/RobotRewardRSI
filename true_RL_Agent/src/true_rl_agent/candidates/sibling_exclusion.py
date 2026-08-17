"""Verifier-owned semantic exclusion between two sibling authoring lanes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from true_rl_agent.candidates.authoring_policy import select_authoring_anchor_source
from true_rl_agent.candidates.component_source_semantics import (
    SingleComponentDeltaComparisonUnavailable,
    compare_single_component_deltas,
    project_single_component_operation_delta,
)
from true_rl_agent.candidates.loader import MAX_CANDIDATE_EXECUTION_FILE_BYTES
from true_rl_agent.candidates.schema import CandidatePackageRef
from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.fs_contract import read_bounded_regular_file
from true_rl_agent.training_programs import CANONICAL_TRAINING_COMPONENTS_PATH


SINGLE_COMPONENT_SEMANTIC_EXCLUSION_SCHEMA_VERSION = (
    "true_rl_agent.single_component_semantic_exclusion.v1"
)


@dataclass(frozen=True)
class SingleComponentSemanticExclusion:
    """Private comparison source plus its deliberately small public projection."""

    comparison_source: str
    comparison_anchor_source: str
    operation_delta: Mapping[str, Any]
    public_projection: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.comparison_source or not self.comparison_anchor_source:
            raise ValueError("sibling exclusion requires both comparison sources")
        if not isinstance(self.operation_delta, Mapping):
            raise TypeError("sibling exclusion operation delta must be an object")
        operation_delta = dict(self.operation_delta)
        projection = _validate_public_projection(self.public_projection)
        component = operation_delta.get("after_component") or operation_delta.get(
            "before_component"
        )
        if (
            operation_delta.get("operation") != projection["operation"]
            or not isinstance(component, Mapping)
            or component.get("mechanism_id")
            != projection["canonical_mechanism_delta"]
        ):
            raise ValueError("sibling exclusion public/private delta mismatch")
        object.__setattr__(self, "operation_delta", MappingProxyType(operation_delta))
        object.__setattr__(self, "public_projection", MappingProxyType(projection))

    @property
    def exclusion_hash(self) -> str:
        return str(self.public_projection["exclusion_hash"])

    @classmethod
    def from_verified_candidate(
        cls,
        *,
        candidate: CandidatePackageRef,
        prompt_payload: Mapping[str, Any],
    ) -> "SingleComponentSemanticExclusion":
        if not isinstance(candidate, CandidatePackageRef):
            raise TypeError("sibling exclusion candidate must be a CandidatePackageRef")
        anchor_source, _, _ = select_authoring_anchor_source(prompt_payload)
        if anchor_source is None:
            raise SingleComponentDeltaComparisonUnavailable(
                "sibling exclusion requires an inline authoring anchor"
            )
        candidate_source = _candidate_source(candidate)
        operation_delta = project_single_component_operation_delta(
            candidate_source=candidate_source,
            anchor_source=anchor_source,
        )
        payload = build_single_component_semantic_exclusion_projection(
            operation_delta
        )
        return cls(
            comparison_source=candidate_source,
            comparison_anchor_source=anchor_source,
            operation_delta=operation_delta,
            public_projection=payload,
        )

    def compare_candidate(
        self,
        *,
        candidate: CandidatePackageRef,
        candidate_prompt_payload: Mapping[str, Any],
    ) -> str | None:
        anchor_source, _, _ = select_authoring_anchor_source(
            candidate_prompt_payload
        )
        if anchor_source is None:
            raise SingleComponentDeltaComparisonUnavailable(
                "sibling candidate has no inline authoring anchor"
            )
        return compare_single_component_deltas(
            candidate_source=_candidate_source(candidate),
            candidate_anchor_source=anchor_source,
            comparison_source=self.comparison_source,
            comparison_anchor_source=self.comparison_anchor_source,
        )


def _candidate_source(candidate: CandidatePackageRef) -> str:
    payload = read_bounded_regular_file(
        Path(candidate.path) / CANONICAL_TRAINING_COMPONENTS_PATH,
        max_bytes=MAX_CANDIDATE_EXECUTION_FILE_BYTES,
        label="sibling candidate training components source",
    )
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SingleComponentDeltaComparisonUnavailable(
            "sibling candidate source is not UTF-8"
        ) from exc


def build_single_component_semantic_exclusion_projection(
    operation_delta: Mapping[str, Any],
) -> dict[str, Any]:
    """Project the one canonical provider-visible exclusion from a verifier delta.

    ``canonical_mechanism_delta`` exposes the component name only as a locator
    for the provider.  It is not the equivalence authority: the blocking
    comparison below remains name-insensitive and uses normalized component
    semantics, so renaming cannot satisfy the exclusion.
    """

    if not isinstance(operation_delta, Mapping):
        raise SingleComponentDeltaComparisonUnavailable(
            "sibling exclusion operation delta is unavailable"
        )
    component = operation_delta.get("after_component") or operation_delta.get(
        "before_component"
    )
    if not isinstance(component, Mapping):
        raise SingleComponentDeltaComparisonUnavailable(
            "sibling exclusion has no changed component"
        )
    mechanism_id = component.get("mechanism_id")
    if not isinstance(mechanism_id, str) or ":" not in mechanism_id:
        raise SingleComponentDeltaComparisonUnavailable(
            "sibling exclusion mechanism identity is unavailable"
        )
    payload = {
        "schema_version": SINGLE_COMPONENT_SEMANTIC_EXCLUSION_SCHEMA_VERSION,
        "operation": operation_delta.get("operation"),
        "canonical_mechanism_delta": mechanism_id,
        "component_kind": mechanism_id.split(":", 1)[0],
        "public_input_dependencies": list(component.get("input_fields", ())),
        "parameter_normalized_semantic_signature": component.get(
            "parameter_normalized_structure_hash"
        ),
        "instruction": (
            "Propose one component mechanism whose normalized callable, input, "
            "and clip structure is substantively different from this exclusion; "
            "renaming or changing only numeric parameters is not distinct."
        ),
    }
    payload["exclusion_hash"] = canonical_json_hash(payload)
    return _validate_public_projection(payload)


def validate_single_component_semantic_exclusion_projection(
    value: Any,
) -> dict[str, Any]:
    """Validate the sole public exclusion schema at a replay boundary."""

    return _validate_public_projection(value)


def _validate_public_projection(value: Any) -> dict[str, Any]:
    fields = {
        "schema_version",
        "operation",
        "canonical_mechanism_delta",
        "component_kind",
        "public_input_dependencies",
        "parameter_normalized_semantic_signature",
        "instruction",
        "exclusion_hash",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("sibling exclusion public projection fields mismatch")
    projection = dict(value)
    if projection.get("schema_version") != (
        SINGLE_COMPONENT_SEMANTIC_EXCLUSION_SCHEMA_VERSION
    ):
        raise ValueError("sibling exclusion public projection schema mismatch")
    expected_hash = projection.pop("exclusion_hash")
    if expected_hash != canonical_json_hash(projection):
        raise ValueError("sibling exclusion public projection hash mismatch")
    projection["exclusion_hash"] = expected_hash
    dependencies = projection.get("public_input_dependencies")
    if not isinstance(dependencies, list) or any(
        not isinstance(item, str) or not item for item in dependencies
    ):
        raise ValueError("sibling exclusion input dependencies are invalid")
    for field_name in (
        "operation",
        "canonical_mechanism_delta",
        "component_kind",
        "parameter_normalized_semantic_signature",
        "instruction",
    ):
        if not isinstance(projection.get(field_name), str) or not projection[
            field_name
        ]:
            raise ValueError(f"sibling exclusion {field_name} is invalid")
    return projection


__all__ = [
    "SINGLE_COMPONENT_SEMANTIC_EXCLUSION_SCHEMA_VERSION",
    "SingleComponentDeltaComparisonUnavailable",
    "SingleComponentSemanticExclusion",
    "build_single_component_semantic_exclusion_projection",
    "validate_single_component_semantic_exclusion_projection",
]
