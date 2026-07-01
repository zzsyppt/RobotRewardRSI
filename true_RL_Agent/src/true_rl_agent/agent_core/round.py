"""Single-round orchestration for fake candidate replay."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

from true_rl_agent.agent_core.context import sanitize_public_context
from true_rl_agent.agent_core.decision import DecisionOutcome, DecisionRecord, decide_candidate, reject_without_eval
from true_rl_agent.agent_core.ledger import append_ledger_record
from true_rl_agent.backends.base import BackendAdapter
from true_rl_agent.candidates.mechanism import (
    MechanismEvidenceGate,
    RuntimeActivationEvidence,
    build_knob_only_shadow_candidate,
)
from true_rl_agent.candidates.schema import CandidatePackageRef, CandidateSurfacePolicy
from true_rl_agent.candidates.verifier import CandidateVerifier
from true_rl_agent.curriculum.capabilities import CapabilityRegistry, build_default_curriculum_capability_registry
from true_rl_agent.curriculum.compiler import CurriculumCompiler
from true_rl_agent.curriculum.schema import CurriculumPlan
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
    curriculum_plan: CurriculumPlan = field(default_factory=lambda: CurriculumPlan(mode="task_default"))
    curriculum_registry: CapabilityRegistry = field(default_factory=build_default_curriculum_capability_registry)
    curriculum_proposal: Mapping[str, Any] | None = None
    task_default_training_config: Mapping[str, Any] = field(default_factory=dict)


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
    task_default_training_config = dict(inputs.task_default_training_config or task_spec.fixed_train_protocol)
    compiled_curriculum = CurriculumCompiler().compile(
        plan=inputs.curriculum_plan,
        task_default_training_config=task_default_training_config,
        registry=inputs.curriculum_registry,
        llm_proposal=inputs.curriculum_proposal,
    )
    verification = inputs.verifier.verify(inputs.candidate)
    jobs = None
    train_result = None
    public_result = None
    executed_eval_spec = None
    runtime_evidence = RuntimeActivationEvidence()
    mechanism_decision = MechanismEvidenceGate().evaluate(
        change_set=inputs.candidate.change_set,
        evidence=runtime_evidence,
    )
    ablation_summary = _build_ablation_summary(
        candidate=inputs.candidate,
        triggered=False,
        reason="not_evaluated",
    )
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
            runtime_evidence = RuntimeActivationEvidence.from_mapping(
                train_result.summary.get("runtime_activation_summary")
            )
            mechanism_decision = MechanismEvidenceGate().evaluate(
                change_set=inputs.candidate.change_set,
                evidence=runtime_evidence,
            )
            preliminary_decision = decide_candidate(
                surface=surface,
                public_result=public_result,
                current_best_score=inputs.current_best_score,
                policy_accept_delta=inputs.policy_accept_delta,
            )
            potential_accept = preliminary_decision.outcome is DecisionOutcome.POLICY_IMPROVEMENT_ACCEPT
            ablation_summary = _build_ablation_summary(
                candidate=inputs.candidate,
                triggered=potential_accept,
                reason="potential_accept_shadow_check" if potential_accept else "not_potential_accept",
            )
            if potential_accept and mechanism_decision.status != "pass":
                decision = DecisionRecord(
                    DecisionOutcome.REJECT,
                    False,
                    mechanism_decision.reason,
                    surface.classification,
                    public_result.public_score,
                )
            else:
                decision = preliminary_decision

    record = {
        "schema_version": "true_rl_agent.ledger_record.v1",
        "round_id": inputs.round_id,
        "public_context": sanitize_public_context(inputs.public_context),
        "task": {"task_id": task_spec.task_id},
        "backend": {"backend_id": inputs.backend_adapter.backend_id},
        "candidate": inputs.candidate.to_public_dict(),
        "verifier": verification.to_public_dict(),
        "candidate_surface": surface.to_public_dict(),
        "curriculum": compiled_curriculum.to_public_dict(),
        "code_change_classification": surface.to_public_dict(),
        "runtime_activation_summary": runtime_evidence.to_public_dict(),
        "ablation_summary": ablation_summary,
        "mechanism_attribution": mechanism_decision.to_public_dict(),
        "anti_tuning_decision": _build_anti_tuning_decision(
            surface_classification=surface.classification,
            mechanism_status=mechanism_decision.status,
            mechanism_reason=mechanism_decision.reason,
            public_context=inputs.public_context,
        ),
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


def _build_ablation_summary(*, candidate: CandidatePackageRef, triggered: bool, reason: str) -> dict[str, Any]:
    shadow = build_knob_only_shadow_candidate(candidate.change_set)
    return {
        "method": "knob_only_shadow_candidate",
        "triggered": triggered,
        "reason": reason,
        "shadow_changed_components": sorted(shadow.changed_components),
        "status": "static_shadow_generated" if triggered else "not_triggered",
    }


def _build_anti_tuning_decision(
    *,
    surface_classification: str,
    mechanism_status: str,
    mechanism_reason: str,
    public_context: Mapping[str, Any],
) -> dict[str, Any]:
    previous_streak = int(public_context.get("pure_knob_streak", 0) or 0)
    pure_knob_streak = previous_streak + 1 if surface_classification == "parameter_only" else 0
    return {
        "surface_classification": surface_classification,
        "mechanism_status": mechanism_status,
        "mechanism_reason": mechanism_reason,
        "parameter_only_not_accept_eligible": surface_classification == "parameter_only",
        "pure_knob_streak": pure_knob_streak,
    }
