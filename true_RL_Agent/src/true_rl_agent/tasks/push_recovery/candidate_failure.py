"""Closed candidate-owned failure vocabulary for Push reward execution."""

from __future__ import annotations

from true_rl_agent.runner.candidate_failure import (
    CandidateFailureContract,
    CandidateFailureEvidence,
    CandidateFailureRule,
)

PUSH_CANDIDATE_FAILURE_CONTRACT = CandidateFailureContract(
    contract_id="true_rl_agent.task.push_recovery.candidate_failure.v1",
    rules=(
        CandidateFailureRule(
            stage="reward_execution",
            reason_codes=("candidate_component_runtime_rejected",),
            requires_initial_policy_state=False,
        ),
    ),
)


class PushCandidateProgramFailure(RuntimeError):
    """A task-classified candidate failure with closed public evidence."""

    def __init__(self) -> None:
        evidence = PUSH_CANDIDATE_FAILURE_CONTRACT.evidence(
            stage="reward_execution",
            reason_code="candidate_component_runtime_rejected",
        )
        self.evidence: CandidateFailureEvidence = evidence
        super().__init__(
            "Push candidate component failed during reward execution"
        )


__all__ = [
    "PUSH_CANDIDATE_FAILURE_CONTRACT",
    "PushCandidateProgramFailure",
]
