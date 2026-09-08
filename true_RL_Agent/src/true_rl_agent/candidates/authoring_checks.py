"""Stable public facade for candidate authoring checks."""

from __future__ import annotations

from typing import Any, Mapping

from true_rl_agent.agent_core.source_request_limits import LATEST_FAILED_SOURCE_ID
from true_rl_agent.candidates.artifact import SealedCandidateArtifactRef
from true_rl_agent.candidates.authoring_policy import SINGLE_COMPONENT_DELTA_INSTRUCTION
from true_rl_agent.candidates.authoring_public_result import public_authoring_result
from true_rl_agent.candidates.component_smoke_runtime import (
    COMPONENT_SMOKE_SCHEMA_VERSION,
    DEFAULT_COMPONENT_SMOKE_LOAD_TIMEOUT_S,
    DEFAULT_COMPONENT_SMOKE_TIMEOUT_S,
    DEFAULT_COMPONENT_SMOKE_WORKER_STARTUP_TIMEOUT_S,
)
from true_rl_agent.candidates.component_smoke_runtime import (
    run_component_smoke_check as _run_component_smoke_check,
)
from true_rl_agent.candidates.component_source_semantics import (
    ANTI_DEGENERATE_SCHEMA_VERSION,
    SingleComponentDeltaComparisonUnavailable,
    assess_anti_degenerate_candidate,
    assess_anti_degenerate_source,
    compare_single_component_deltas,
)
from true_rl_agent.candidates.schema import CandidatePackageRef
from true_rl_agent.tasks.component_smoke import ComponentSmokeAdapter

STATIC_AUTHORING_FAILURE_SCHEMA_VERSION = "true_rl_agent.static_authoring_failure.v1"

_public_result = public_authoring_result

__all__ = [
    "ANTI_DEGENERATE_SCHEMA_VERSION",
    "COMPONENT_SMOKE_SCHEMA_VERSION",
    "SingleComponentDeltaComparisonUnavailable",
    "assess_anti_degenerate_candidate",
    "assess_anti_degenerate_source",
    "compare_single_component_deltas",
    "build_anti_degenerate_repair_feedback",
    "build_component_smoke_repair_feedback",
    "run_component_smoke_check",
]

def run_component_smoke_check(
    candidate: CandidatePackageRef | SealedCandidateArtifactRef,
    smoke_adapter: ComponentSmokeAdapter | None,
    *,
    timeout_s: float = DEFAULT_COMPONENT_SMOKE_TIMEOUT_S,
    worker_startup_timeout_s: float = DEFAULT_COMPONENT_SMOKE_WORKER_STARTUP_TIMEOUT_S,
) -> dict[str, Any]:
    """Run the spawn-isolated smoke runtime through the stable facade."""

    return _run_component_smoke_check(
        candidate,
        smoke_adapter,
        timeout_s=timeout_s,
        worker_startup_timeout_s=worker_startup_timeout_s,
        load_timeout_s=DEFAULT_COMPONENT_SMOKE_LOAD_TIMEOUT_S,
    )


def build_component_smoke_repair_feedback(
    *,
    candidate: CandidatePackageRef,
    smoke_result: Mapping[str, Any],
) -> dict[str, Any]:
    advisory = smoke_result.get("status") == "advisory"
    feedback = {
        "schema_version": "true_rl_agent.provider_repair_feedback.v1",
        "status": "repair_suggested" if advisory else "repair_required",
        "previous_candidate_id": candidate.candidate_id,
        "previous_candidate_changed_components": sorted(candidate.change_set.changed_components),
        "previous_candidate_source_logical_id": LATEST_FAILED_SOURCE_ID,
        "previous_candidate_source_delivery": "see_source_catalog",
        "failure_category": (
            "component_smoke_advisory" if advisory else "component_smoke_error"
        ),
        "component_smoke_result": dict(smoke_result),
        "instructions": [
            "Fix only the candidate training_code/components.py so every reward and diagnostic callable executes on the task-owned RewardInputs fixture.",
            "Return vector-like finite reward outputs with the shape described by the prompt reward_inputs_contract.",
            (
                "The callable produced a nonzero smoke signal, but its declared clip_min/clip_max and weight suppressed that signal to zero; correct those declarations if this was not intentional."
                if advisory
                else "Keep each callable finite under the declared component contract."
            ),
            "Diagnostic callables must execute without errors on the same task-owned fixture.",
            "Do not change runner, eval protocol, training budget, seed policy, checkpoints, PPO, or acceptance rules.",
        ],
    }
    return _public_result(feedback)


def build_anti_degenerate_repair_feedback(
    *,
    candidate: CandidatePackageRef,
    assessment: Mapping[str, Any],
) -> dict[str, Any]:
    feedback = {
        "schema_version": "true_rl_agent.provider_repair_feedback.v1",
        "status": "rewrite_required",
        "previous_candidate_id": candidate.candidate_id,
        "previous_candidate_changed_components": sorted(candidate.change_set.changed_components),
        "previous_candidate_source_logical_id": LATEST_FAILED_SOURCE_ID,
        "previous_candidate_source_delivery": "see_source_catalog",
        "failure_category": "authoring_policy_violation",
        "anti_degenerate_assessment": dict(assessment),
        "instructions": [
            "Rewrite the candidate to introduce a real code-level reward mechanism delta.",
            SINGLE_COMPONENT_DELTA_INSTRUCTION,
            "Do not resubmit identical, formatting-only, or rationale-only source.",
            (
                "Do not repeat a previously evaluated mechanism by changing only "
                "metadata strings or numeric constants."
            ),
            "Keep runner, eval, seed, budget, checkpoint, PPO, curriculum, and acceptance rules unchanged.",
        ],
    }
    return _public_result(feedback)


def build_static_authoring_failure(
    *,
    candidate: CandidatePackageRef,
    feedback: Mapping[str, Any],
) -> dict[str, Any]:
    return _public_result(
        {
            "schema_version": STATIC_AUTHORING_FAILURE_SCHEMA_VERSION,
            "candidate_id": candidate.candidate_id,
            "candidate_changed_components": sorted(candidate.change_set.changed_components),
            "failure_category": feedback.get("failure_category", "static_authoring_error"),
            "feedback": dict(feedback),
        }
    )
