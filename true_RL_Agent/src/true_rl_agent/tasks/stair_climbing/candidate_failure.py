"""Closed candidate-owned failure vocabulary for Stair reward execution."""

from __future__ import annotations

from true_rl_agent.runner.candidate_failure import (
    CandidateFailureContract,
    CandidateFailureEvidence,
    CandidateFailureRule,
)

STAIR_CANDIDATE_FAILURE_CONTRACT = CandidateFailureContract(
    contract_id="true_rl_agent.task.stair_climbing.candidate_failure.v1",
    rules=(
        CandidateFailureRule(
            stage="activation_preflight",
            reason_codes=("reward_program_activation_rejected",),
            requires_initial_policy_state=False,
        ),
        CandidateFailureRule(
            stage="ppo_reward_execution",
            reason_codes=(
                "reward_program_output_contract_rejected",
                "reward_program_runtime_rejected",
            ),
            requires_initial_policy_state=True,
        ),
        CandidateFailureRule(
            stage="qualification",
            reason_codes=("reward_program_qualification_rejected",),
            requires_initial_policy_state=False,
        ),
    ),
)


class StairCandidateProgramFailure(RuntimeError):
    """A task-classified candidate failure with no raw exception projection."""

    def __init__(self, evidence: CandidateFailureEvidence) -> None:
        validated = STAIR_CANDIDATE_FAILURE_CONTRACT.validate_evidence(
            evidence.to_public_dict()
        )
        self.evidence = validated
        super().__init__(
            "Stair candidate program failed during "
            f"{validated.failure_stage} ({validated.reason_code})"
        )

    @classmethod
    def qualification(cls) -> "StairCandidateProgramFailure":
        return cls(
            STAIR_CANDIDATE_FAILURE_CONTRACT.evidence(
                stage="qualification",
                reason_code="reward_program_qualification_rejected",
            )
        )

    @classmethod
    def activation_preflight(cls) -> "StairCandidateProgramFailure":
        return cls(
            STAIR_CANDIDATE_FAILURE_CONTRACT.evidence(
                stage="activation_preflight",
                reason_code="reward_program_activation_rejected",
            )
        )

    @classmethod
    def ppo_reward_execution(cls) -> "StairCandidateProgramFailure":
        return cls(
            STAIR_CANDIDATE_FAILURE_CONTRACT.evidence(
                stage="ppo_reward_execution",
                reason_code="reward_program_runtime_rejected",
            )
        )

    @classmethod
    def ppo_reward_output_contract(cls) -> "StairCandidateProgramFailure":
        return cls(
            STAIR_CANDIDATE_FAILURE_CONTRACT.evidence(
                stage="ppo_reward_execution",
                reason_code="reward_program_output_contract_rejected",
            )
        )


__all__ = [
    "STAIR_CANDIDATE_FAILURE_CONTRACT",
    "StairCandidateProgramFailure",
]
