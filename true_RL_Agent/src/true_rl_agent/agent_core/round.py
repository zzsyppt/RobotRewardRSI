"""Single-round orchestration for fake candidate replay."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from true_rl_agent.agent_core.decision import DecisionRecord, decide_candidate, reject_without_eval
from true_rl_agent.agent_core.ledger import append_ledger_record
from true_rl_agent.backends.base import BackendAdapter
from true_rl_agent.candidates.schema import CandidatePackageRef, CandidateSurfacePolicy
from true_rl_agent.candidates.verifier import CandidateVerifier
from true_rl_agent.evaluation.public_result import PublicEvalResult
from true_rl_agent.policy import PolicyInitialization
from true_rl_agent.runner.base import Runner, TrainJobResult
from true_rl_agent.safety.invariants import enforce_task_invariants
from true_rl_agent.tasks.base import TaskPlugin


@dataclass(frozen=True)
class RoundInputs:
    round_id: str
    task_plugin: TaskPlugin
    backend_adapter: BackendAdapter
    runner: Runner
    verifier: CandidateVerifier
    candidate: CandidatePackageRef
    policy_initialization: PolicyInitialization
    ledger_path: Path
    public_context: Mapping[str, Any]
    current_best_score: float
    policy_accept_delta: float


@dataclass(frozen=True)
class RoundResult:
    decision: DecisionRecord
    train_result: TrainJobResult | None
    public_result: PublicEvalResult | None


def run_candidate_round(inputs: RoundInputs) -> RoundResult:
    task_spec = inputs.task_plugin.task_spec
    if not task_spec.supports_policy_initialization(inputs.policy_initialization.kind):
        raise ValueError(f"task does not support policy initialization: {inputs.policy_initialization.kind}")

    surface = CandidateSurfacePolicy().classify(inputs.candidate.change_set)
    verification = inputs.verifier.verify(inputs.candidate)
    jobs = None
    train_result = None
    public_result = None
    executed_eval_spec = None
    if verification.status != "pass":
        decision = reject_without_eval(
            reason="candidate_verifier_failed",
            surface_classification=surface.classification,
        )
    else:
        jobs = inputs.backend_adapter.materialize_jobs(
            round_id=inputs.round_id,
            task_spec=task_spec,
            candidate=inputs.candidate,
            policy_initialization=inputs.policy_initialization,
        )
        try:
            enforce_task_invariants(
                expected=task_spec.invariants,
                actual=jobs.task_invariants or task_spec.invariants,
            )
        except ValueError:
            decision = reject_without_eval(
                reason="task_invariant_violation",
                surface_classification=surface.classification,
            )
        else:
            train_result = inputs.runner.run_train(jobs.train_spec)
            executed_eval_spec = replace(jobs.eval_spec, checkpoint=train_result.checkpoint)
            public_result = inputs.runner.run_eval(executed_eval_spec)
            decision = decide_candidate(
                surface=surface,
                public_result=public_result,
                current_best_score=inputs.current_best_score,
                policy_accept_delta=inputs.policy_accept_delta,
            )

    record = {
        "schema_version": "true_rl_agent.ledger_record.v1",
        "round_id": inputs.round_id,
        "public_context": dict(inputs.public_context),
        "task": {"task_id": task_spec.task_id},
        "backend": {"backend_id": inputs.backend_adapter.backend_id},
        "candidate": inputs.candidate.to_public_dict(),
        "verifier": verification.to_public_dict(),
        "candidate_surface": surface.to_public_dict(),
        "jobs": (
            {
                "train": jobs.train_spec.to_public_dict(),
                "eval": (executed_eval_spec or jobs.eval_spec).to_public_dict(),
            }
            if jobs is not None
            else None
        ),
        "train_result": train_result.to_public_dict() if train_result is not None else None,
        "public_result": public_result.to_public_dict() if public_result is not None else None,
        "decision": decision.to_public_dict(),
        "feedback_safe": {
            "public_only": True,
            "private_eval_used": False,
            "hidden_seed_data_included": False,
        },
    }
    append_ledger_record(inputs.ledger_path, record)
    return RoundResult(decision=decision, train_result=train_result, public_result=public_result)
