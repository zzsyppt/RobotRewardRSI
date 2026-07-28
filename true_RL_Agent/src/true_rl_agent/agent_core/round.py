"""Single-round orchestration for fake candidate replay."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

from true_rl_agent.agent_core.context import sanitize_public_context
from true_rl_agent.agent_core.decision import (
    DecisionOutcome,
    DecisionRecord,
    decide_candidate,
    reject_without_eval,
)
from true_rl_agent.agent_core.ledger import append_ledger_record
from true_rl_agent.agent_core.source_request_feedback import (
    project_source_request_round_feedback,
)
from true_rl_agent.backends.base import BackendAdapter
from true_rl_agent.candidates.artifact import (
    VerifiedCandidateArtifactRef,
)
from true_rl_agent.candidates.mechanism import (
    MechanismEvidenceGate,
    RuntimeActivationEvidence,
    build_knob_only_shadow_candidate,
    build_mechanism_attribution_summary,
)
from true_rl_agent.candidates.schema import (
    CandidateChangeSet,
    CandidatePackageRef,
    CandidateSurfacePolicy,
)
from true_rl_agent.candidates.verifier import (
    CandidateVerifier,
    seal_and_verify_candidate,
)
from true_rl_agent.curriculum.capabilities import (
    CapabilityRegistry,
    build_default_curriculum_capability_registry,
)
from true_rl_agent.curriculum.compiler import CurriculumCompiler
from true_rl_agent.curriculum.schema import CurriculumPlan
from true_rl_agent.evaluation.acceptance_contract import validate_acceptance_evidence
from true_rl_agent.evaluation.acceptance_protocol import AcceptanceProtocol
from true_rl_agent.evaluation.binding import (
    EvaluationBindingV1,
    attach_evaluation_binding,
    require_sha256_digest,
)
from true_rl_agent.evaluation.current_best import (
    CurrentBestEvalStats,
    attach_eval_protocol_hash,
)
from true_rl_agent.evaluation.decision_record_contract import validate_decision_record
from true_rl_agent.evaluation.public_projection import (
    assert_public_projection_safe,
    project_public_runner_error,
)
from true_rl_agent.evaluation.public_result import PublicEvalResult
from true_rl_agent.memory.records import PrivateArtifactRef, PromptPayloadRef
from true_rl_agent.policy import PolicyInitialization
from true_rl_agent.runner.argv_contract import option_occurrences
from true_rl_agent.runner.base import (
    CandidateRuntimeError,
    Runner,
    RunnerExecutionError,
    TrainJobResult,
)
from true_rl_agent.runner.specs import CheckpointRef, EvalJobSpec
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
    current_best_eval_stats: CurrentBestEvalStats
    session_genesis_hash: str
    evaluation_binding: EvaluationBindingV1
    source_request_feedback: Mapping[str, Any] | None = None
    prompt_payload_ref: PromptPayloadRef | None = None
    curriculum_plan: CurriculumPlan = field(default_factory=lambda: CurriculumPlan(mode="task_default"))
    curriculum_registry: CapabilityRegistry = field(default_factory=build_default_curriculum_capability_registry)
    curriculum_proposal: Mapping[str, Any] | None = None
    task_default_training_config: Mapping[str, Any] = field(default_factory=dict)
    record_candidate_runtime_failures: bool = True
    candidate_artifact_root: Path | None = None


@dataclass(frozen=True)
class RoundResult:
    decision: DecisionRecord
    train_result: TrainJobResult | None
    public_result: PublicEvalResult | None
    round_stage: str


def run_candidate_round(inputs: RoundInputs) -> RoundResult:
    task_spec = inputs.task_plugin.task_spec
    acceptance_protocol = task_spec.acceptance_protocol
    if acceptance_protocol is None:
        raise ValueError("task does not define a fixed acceptance protocol")
    if not task_spec.supports_policy_initialization(inputs.policy_initialization.kind):
        raise ValueError(f"task does not support policy initialization: {inputs.policy_initialization.kind}")
    require_sha256_digest(
        inputs.session_genesis_hash,
        "RoundInputs.session_genesis_hash",
    )
    if not isinstance(inputs.evaluation_binding, EvaluationBindingV1):
        raise TypeError("RoundInputs.evaluation_binding must be EvaluationBindingV1")
    inputs.backend_adapter.validate_session_evaluation_binding(
        task_spec=task_spec,
        evaluation_binding=inputs.evaluation_binding,
    )
    if inputs.current_best_eval_stats.evaluation_binding() != inputs.evaluation_binding:
        raise ValueError("round current-best evaluation binding mismatch")

    candidate: CandidatePackageRef | VerifiedCandidateArtifactRef = inputs.candidate
    artifact_root = inputs.candidate_artifact_root or (
        inputs.ledger_path.parent / "candidate_artifacts"
    )
    artifact_verification = seal_and_verify_candidate(
        inputs.candidate,
        artifact_root=artifact_root,
        verifier=inputs.verifier,
    )
    verification = artifact_verification.result
    if artifact_verification.verified_artifact is not None:
        candidate = artifact_verification.verified_artifact
    else:
        candidate = replace(
            inputs.candidate,
            change_set=CandidateChangeSet(frozenset()),
        )
    surface_policy = inputs.task_plugin.candidate_surface_policy()
    surface = surface_policy.classify(candidate.change_set)
    task_default_training_config = dict(inputs.task_default_training_config or task_spec.fixed_train_protocol)
    compiled_curriculum = CurriculumCompiler().compile(
        plan=inputs.curriculum_plan,
        task_default_training_config=task_default_training_config,
        registry=inputs.curriculum_registry,
        llm_proposal=inputs.curriculum_proposal,
    )
    jobs = None
    train_result = None
    public_result = None
    executed_eval_spec = None
    runner_error = None
    runtime_evidence = RuntimeActivationEvidence()
    runtime_activation_gate = MechanismEvidenceGate().evaluate(
        surface=surface,
        evidence=runtime_evidence,
    )
    ablation_summary = _build_ablation_summary(
        candidate=candidate,
        surface_policy=surface_policy,
        triggered=False,
        reason="not_evaluated",
    )
    mechanism_attribution = build_mechanism_attribution_summary(
        runtime_activation_gate=runtime_activation_gate,
        ablation_summary=ablation_summary,
    )
    if verification.status != "pass":
        decision = reject_without_eval(
            reason="candidate_verifier_failed",
            surface_classification=surface.classification,
        )
    else:
        if not isinstance(candidate, VerifiedCandidateArtifactRef):
            raise RuntimeError(
                "passed candidate verification did not produce an execution capability"
            )
        jobs = inputs.backend_adapter.materialize_jobs(
            round_id=inputs.round_id,
            task_spec=task_spec,
            candidate=candidate,
            policy_initialization=inputs.policy_initialization,
            evaluation_binding=inputs.evaluation_binding,
        )
        if jobs.eval_spec.evaluation_binding != inputs.evaluation_binding:
            raise ValueError("materialized eval job evaluation binding mismatch")
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
            try:
                train_result = inputs.runner.run_train(jobs.train_spec)
                executed_eval_spec = _bind_eval_spec_to_checkpoint(jobs.eval_spec, train_result.checkpoint)
                raw_eval_result = inputs.runner.run_eval(executed_eval_spec)
                try:
                    projected_result = attach_eval_protocol_hash(
                        inputs.task_plugin.project_public_result(
                            public_result=raw_eval_result,
                            train_summary=train_result.summary,
                        ),
                        eval_protocol_hash=executed_eval_spec.evaluation_protocol().protocol_hash,
                    )
                    public_result = _validate_public_eval_result(
                        attach_evaluation_binding(
                            projected_result,
                            binding=executed_eval_spec.evaluation_binding,
                            subject=executed_eval_spec.evaluation_subject(),
                            subject_manifest_hash=(
                                executed_eval_spec.subject_manifest_hash
                            ),
                            program_source_checksum=(
                                executed_eval_spec.program_source_checksum
                            ),
                            training_binding_hash=(
                                executed_eval_spec.training_binding_hash
                            ),
                            capability_profile_hash=(
                                executed_eval_spec.capability_profile_hash
                            ),
                        ),
                        acceptance_protocol=acceptance_protocol,
                    )
                except ValueError as exc:
                    runner_error = project_public_runner_error(exc)
                    public_result = None
                    decision = reject_without_eval(
                        reason="eval_incomplete",
                        surface_classification=surface.classification,
                    )
                else:
                    runtime_evidence = RuntimeActivationEvidence.from_mapping(
                        train_result.summary.get("runtime_activation_summary")
                    )
                    runtime_activation_gate = MechanismEvidenceGate().evaluate(
                        surface=surface,
                        evidence=runtime_evidence,
                    )
                    preliminary_decision = decide_candidate(
                        surface=surface,
                        public_result=public_result,
                        current_best=inputs.current_best_eval_stats,
                        acceptance_protocol=acceptance_protocol,
                    )
                    potential_accept = preliminary_decision.outcome is DecisionOutcome.POLICY_IMPROVEMENT_ACCEPT
                    ablation_summary = _build_ablation_summary(
                        candidate=candidate,
                        surface_policy=surface_policy,
                        triggered=potential_accept,
                        reason="potential_accept_shadow_check" if potential_accept else "not_potential_accept",
                    )
                    mechanism_attribution = build_mechanism_attribution_summary(
                        runtime_activation_gate=runtime_activation_gate,
                        ablation_summary=ablation_summary,
                    )
                    if potential_accept and runtime_activation_gate.status != "pass":
                        decision = DecisionRecord(
                            DecisionOutcome.REJECT,
                            False,
                            runtime_activation_gate.reason,
                            surface.classification,
                            public_result.public_score,
                            preliminary_decision.comparison,
                            preliminary_decision.consumed_gates,
                            preliminary_decision.gate_statuses,
                        )
                    else:
                        decision = preliminary_decision
            except CandidateRuntimeError as exc:
                if not inputs.record_candidate_runtime_failures:
                    raise
                runner_error = project_public_runner_error(exc)
                decision = reject_without_eval(
                    reason="candidate_runtime_failed",
                    surface_classification=surface.classification,
                )
            except (RunnerExecutionError, ValueError, TypeError) as exc:
                if train_result is None:
                    raise
                runner_error = project_public_runner_error(exc)
                decision = reject_without_eval(
                    reason="eval_incomplete",
                    surface_classification=surface.classification,
                )

    private_artifacts = _private_artifacts_for_round(
        candidate=candidate,
        jobs=jobs,
        train_result=train_result,
        executed_eval_spec=executed_eval_spec,
    )
    source_request_feedback = project_source_request_round_feedback(
        inputs.source_request_feedback
    )
    record = {
        "schema_version": "true_rl_agent.ledger_record.v1",
        "round_id": inputs.round_id,
        "session_genesis_hash": inputs.session_genesis_hash,
        "evaluation_binding": inputs.evaluation_binding.to_public_dict(),
        "prompt_payload": (
            inputs.prompt_payload_ref.to_public_dict()
            if inputs.prompt_payload_ref is not None
            else None
        ),
        "public_context": sanitize_public_context(inputs.public_context),
        "source_request_feedback": source_request_feedback,
        "task": {"task_id": task_spec.task_id},
        "acceptance_protocol": acceptance_protocol.to_public_dict(),
        "backend": {"backend_id": inputs.backend_adapter.backend_id},
        "candidate": candidate.to_public_dict(),
        "private_artifacts": [artifact.to_private_dict() for artifact in private_artifacts],
        "verifier": verification.to_public_dict(),
        "candidate_surface": surface.to_public_dict(),
        "curriculum": compiled_curriculum.to_public_dict(),
        "code_change_classification": surface.to_public_dict(),
        "runtime_activation_summary": runtime_evidence.to_public_dict(),
        "runtime_activation_gate": runtime_activation_gate.to_public_dict(),
        "ablation_summary": ablation_summary,
        "mechanism_attribution": mechanism_attribution,
        "anti_tuning_decision": _build_anti_tuning_decision(
            surface_classification=surface.classification,
            runtime_activation_status=runtime_activation_gate.status,
            runtime_activation_reason=runtime_activation_gate.reason,
            mechanism_attribution=mechanism_attribution,
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
        "runner_error": runner_error,
        "decision": decision.to_public_dict(),
        "feedback_safe": {
            "public_only": True,
            "private_eval_used": False,
            "hidden_seed_data_included": False,
        },
    }
    decision_validation = validate_decision_record(
        record,
        acceptance_protocol=acceptance_protocol,
        expected_session_genesis_hash=inputs.session_genesis_hash,
        expected_evaluation_binding=inputs.evaluation_binding,
    )
    decision_validation.raise_for_failure()
    _assert_round_public_sections_safe(record)
    append_ledger_record(inputs.ledger_path, record)
    return RoundResult(
        decision=decision,
        train_result=train_result,
        public_result=public_result,
        round_stage=decision_validation.round_stage.stage,
    )


def _build_ablation_summary(
    *,
    candidate: CandidatePackageRef | VerifiedCandidateArtifactRef,
    surface_policy: CandidateSurfacePolicy,
    triggered: bool,
    reason: str,
) -> dict[str, Any]:
    shadow = build_knob_only_shadow_candidate(
        candidate.change_set,
        surface_policy=surface_policy,
    )
    return {
        "method": "knob_only_shadow_candidate",
        "triggered": triggered,
        "reason": reason,
        "shadow_changed_components": sorted(shadow.changed_components),
        "status": "static_shadow_generated" if triggered else "not_triggered",
    }


def _validate_public_eval_result(
    public_result: PublicEvalResult,
    *,
    acceptance_protocol: AcceptanceProtocol,
) -> PublicEvalResult:
    public_result.to_public_dict()
    if public_result.status != "pass":
        return public_result
    validation = validate_acceptance_evidence(
        public_result.metrics,
        acceptance_protocol=acceptance_protocol,
        public_score=public_result.public_score,
    )
    if not validation.acceptance_protocol_binding_valid:
        detail = ", ".join(
            reason
            for reason in validation.reasons
            if reason.startswith("acceptance_protocol_")
        )
        raise ValueError(
            "public eval acceptance protocol is invalid"
            + (f": {detail}" if detail else "")
        )
    if not validation.score_stats_binding_valid:
        raise ValueError("public eval acceptance score stats are incomplete or inconsistent")
    if not validation.public_score_binding_valid:
        raise ValueError(
            "public eval public_score is not bound to canonical acceptance evidence"
        )
    return public_result


def _private_artifacts_for_round(
    *,
    candidate: CandidatePackageRef | VerifiedCandidateArtifactRef,
    jobs: Any,
    train_result: TrainJobResult | None,
    executed_eval_spec: EvalJobSpec | None,
) -> tuple[PrivateArtifactRef, ...]:
    refs = [
        PrivateArtifactRef(
            kind="candidate_package",
            uri=candidate.path,
            reason="candidate_source_resolution",
        )
    ]
    if jobs is not None:
        refs.extend(_metadata_private_artifacts(jobs.train_spec.metadata, phase="train"))
        eval_spec = executed_eval_spec or jobs.eval_spec
        refs.extend(_metadata_private_artifacts(eval_spec.metadata, phase="eval"))
    if train_result is not None:
        refs.extend(_checkpoint_private_artifacts(train_result.checkpoint))
    return _dedupe_private_artifacts(refs)


def _metadata_private_artifacts(metadata: Mapping[str, Any], *, phase: str) -> list[PrivateArtifactRef]:
    refs: list[PrivateArtifactRef] = []
    summary_path = metadata.get("summary_path")
    if isinstance(summary_path, str) and summary_path:
        refs.append(
            PrivateArtifactRef(
                kind=f"runner_{phase}_summary",
                uri=summary_path,
                reason=f"runner_{phase}_raw_summary",
            )
        )
    work_dir = metadata.get("work_dir")
    if isinstance(work_dir, str) and work_dir:
        refs.append(
            PrivateArtifactRef(
                kind=f"runner_{phase}_work_dir",
                uri=work_dir,
                reason=f"runner_{phase}_execution_directory",
            )
        )
    checkpoint_output = metadata.get("checkpoint_output_path")
    if isinstance(checkpoint_output, str) and checkpoint_output:
        refs.append(
            PrivateArtifactRef(
                kind="runner_checkpoint_output",
                uri=checkpoint_output,
                reason="runner_train_checkpoint_output",
            )
        )
    return refs


def _checkpoint_private_artifacts(checkpoint: CheckpointRef) -> list[PrivateArtifactRef]:
    refs: list[PrivateArtifactRef] = []
    for key, kind, reason in (
        ("local_path", "runner_checkpoint", "runner_local_checkpoint_path"),
        ("source_latest_checkpoint", "runner_checkpoint_source", "runner_source_checkpoint_path"),
    ):
        value = checkpoint.metadata.get(key)
        if isinstance(value, str) and value:
            refs.append(PrivateArtifactRef(kind=kind, uri=value, checksum=checkpoint.checksum, reason=reason))
    return refs


def _dedupe_private_artifacts(refs: list[PrivateArtifactRef]) -> tuple[PrivateArtifactRef, ...]:
    deduped: list[PrivateArtifactRef] = []
    seen: set[tuple[str, str]] = set()
    for ref in refs:
        key = (ref.kind, ref.uri)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ref)
    return tuple(deduped)


def _assert_round_public_sections_safe(record: Mapping[str, Any]) -> None:
    for key in ("jobs", "train_result", "public_result", "runner_error", "decision"):
        value = record.get(key)
        if value is not None:
            assert_public_projection_safe(value)


def _bind_eval_spec_to_checkpoint(eval_spec: EvalJobSpec, checkpoint: CheckpointRef) -> EvalJobSpec:
    if checkpoint.role != "eval" or not checkpoint.sealed:
        raise ValueError(
            "eval binding requires a sealed checkpoint with role='eval'"
        )
    if eval_spec.checkpoint is not None:
        raise ValueError("eval job is already bound to checkpoint bytes")
    checkpoint_output_plan = eval_spec.checkpoint_output_plan
    if checkpoint_output_plan is None:
        raise ValueError("eval job has no checkpoint output plan")
    lineage = checkpoint.metadata
    expected_lineage = {
        "training_subject": eval_spec.subject.to_public_dict(),
        "subject_manifest_hash": eval_spec.subject_manifest_hash,
        "program_source_checksum": eval_spec.program_source_checksum,
        "training_binding_hash": eval_spec.training_binding_hash,
        "capability_profile_hash": eval_spec.capability_profile_hash,
    }
    for field_name, expected in expected_lineage.items():
        if lineage.get(field_name) != expected:
            raise ValueError(
                f"eval checkpoint lineage mismatch for {field_name}"
            )
    old_uri = checkpoint_output_plan.uri
    cli_checkpoint = str(checkpoint.metadata.get("local_path") or checkpoint.uri)
    metadata = dict(eval_spec.metadata)
    cli_argv = metadata.get("cli_argv")
    if cli_argv is None:
        if "cli_command" in metadata:
            raise ValueError(
                "planned eval metadata cannot provide cli_command without cli_argv"
            )
    elif not isinstance(cli_argv, list) or not all(
        isinstance(item, str) for item in cli_argv
    ):
        raise ValueError("planned eval metadata requires structured cli_argv")
    else:
        metadata["cli_argv"] = _bind_eval_cli_argv(
            cli_argv,
            old_checkpoint_uri=old_uri,
            new_checkpoint_uri=cli_checkpoint,
            new_checkpoint_hash=checkpoint.checksum,
        )
        metadata["cli_command"] = " ".join(metadata["cli_argv"])
    return replace(
        eval_spec,
        checkpoint=checkpoint,
        checkpoint_output_plan=None,
        metadata=metadata,
    )


def _bind_eval_cli_argv(
    cli_argv: list[str],
    *,
    old_checkpoint_uri: str,
    new_checkpoint_uri: str,
    new_checkpoint_hash: str,
) -> list[str]:
    checkpoint_positions = option_occurrences(cli_argv, "--checkpoint")
    if len(checkpoint_positions) != 1:
        raise ValueError("planned eval argv requires exactly one --checkpoint")
    checkpoint_index = checkpoint_positions[0]
    if cli_argv[checkpoint_index] != "--checkpoint":
        raise ValueError("planned eval --checkpoint must use a separate value token")
    if checkpoint_index + 1 >= len(cli_argv):
        raise ValueError("planned eval argv --checkpoint is missing its value")
    if cli_argv[checkpoint_index + 1] != old_checkpoint_uri:
        raise ValueError("planned eval argv checkpoint does not match output plan")
    if option_occurrences(cli_argv, "--checkpoint-sha256"):
        raise ValueError("planned eval argv must not declare a checkpoint digest")
    updated = list(cli_argv)
    updated[checkpoint_index + 1] = new_checkpoint_uri
    updated[checkpoint_index + 2 : checkpoint_index + 2] = [
        "--checkpoint-sha256",
        new_checkpoint_hash,
    ]
    return updated


def _build_anti_tuning_decision(
    *,
    surface_classification: str,
    runtime_activation_status: str,
    runtime_activation_reason: str,
    mechanism_attribution: Mapping[str, Any],
    public_context: Mapping[str, Any],
) -> dict[str, Any]:
    previous_streak = int(public_context.get("pure_knob_streak", 0) or 0)
    pure_knob_streak = previous_streak + 1 if surface_classification == "parameter_only" else 0
    return {
        "surface_classification": surface_classification,
        "runtime_activation_status": runtime_activation_status,
        "runtime_activation_reason": runtime_activation_reason,
        "mechanism_attribution_status": mechanism_attribution.get("status"),
        "mechanism_attribution_reason": mechanism_attribution.get("reason"),
        "parameter_only_not_accept_eligible": surface_classification == "parameter_only",
        "pure_knob_streak": pure_knob_streak,
    }
