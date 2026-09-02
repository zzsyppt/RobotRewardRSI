"""Ledger-independent candidate trial execution and width-one commit wrapper."""

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
from true_rl_agent.agent_core.ledger import (
    LEDGER_RECORD_SCHEMA_VERSION,
    NON_PROVIDER_AUTHORING_MODE,
    PROVIDER_AUTHORING_MODE,
    append_ledger_record_once,
    primary_ledger_epoch_hash,
    require_ledger_authoring_mode,
)
from true_rl_agent.agent_core.trial_receipts import (
    TRIAL_EXECUTION_CONTRACT_SCHEMA_VERSION,
    LoadedTrialExecution,
    TrialReceiptJournal,
    TrialReceiptRef,
    load_trial_execution_receipt,
)
from true_rl_agent.backends.base import BackendAdapter
from true_rl_agent.candidates.artifact import (
    SealedCandidateArtifactRef,
    VerifiedCandidateArtifactRef,
)
from true_rl_agent.candidates.mechanism import (
    MechanismEvidenceGate,
    RuntimeActivationEvidence,
    build_knob_only_shadow_candidate,
    build_mechanism_attribution_summary,
)
from true_rl_agent.candidates.authoring_policy import select_authoring_anchor_source
from true_rl_agent.candidates.component_source_semantics import (
    project_single_component_operation_delta,
)
from true_rl_agent.candidates.loader import MAX_CANDIDATE_EXECUTION_FILE_BYTES
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
    require_sha256_checksum,
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
from true_rl_agent.prompt_artifacts import (
    AuthoringEvidenceRef,
    authoring_outcome_from_evidence,
    project_selected_authoring_mechanism_delta,
    validate_ledger_prompt_artifacts,
    verify_authoring_anchor_prompt,
    verify_authoring_candidate_binding,
    verify_authoring_evidence,
)
from true_rl_agent.runner.base import (
    CandidateRuntimeError,
    Runner,
    RunnerExecutionError,
    TrainJobResult,
)
from true_rl_agent.runner.eval_binding import bind_eval_spec_to_checkpoint
from true_rl_agent.runner.job_attempts import JobAttemptScope
from true_rl_agent.runner.specs import CheckpointRef, EvalJobSpec, GpuProcessBinding
from true_rl_agent.fs_contract import read_bounded_regular_file
from true_rl_agent.safety.invariants import enforce_task_invariants
from true_rl_agent.tasks.base import TaskPlugin
from true_rl_agent.training_programs import CANONICAL_TRAINING_COMPONENTS_PATH


@dataclass(frozen=True)
class RoundInputs:
    round_id: str
    task_plugin: TaskPlugin
    backend_adapter: BackendAdapter
    runner: Runner
    candidate_admission_policy: CandidateVerifier
    candidate: CandidatePackageRef
    policy_initialization: PolicyInitialization
    ledger_path: Path
    public_context: Mapping[str, Any]
    current_best_eval_stats: CurrentBestEvalStats
    session_genesis_hash: str
    evaluation_binding: EvaluationBindingV1
    authoring_mode: str
    prompt_payload_ref: PromptPayloadRef | None = None
    authoring_evidence_ref: AuthoringEvidenceRef | None = None
    curriculum_plan: CurriculumPlan = field(default_factory=lambda: CurriculumPlan(mode="task_default"))
    curriculum_registry: CapabilityRegistry = field(default_factory=build_default_curriculum_capability_registry)
    curriculum_proposal: Mapping[str, Any] | None = None
    task_default_training_config: Mapping[str, Any] = field(default_factory=dict)
    record_candidate_runtime_failures: bool = True
    candidate_artifact_root: Path | None = None
    expected_initial_policy_state_hash: str | None = None
    parent_program_source_checksum: str | None = None
    ledger_epoch_hash: str | None = None
    backend_execution_environment_manifest_hash: str | None = None
    trial_receipt_root: Path | None = None
    train_attempt_scope: JobAttemptScope | None = None
    eval_attempt_scope: JobAttemptScope | None = None
    gpu_process_binding: GpuProcessBinding | None = None


@dataclass(frozen=True)
class RoundResult:
    decision: DecisionRecord
    train_result: TrainJobResult | None
    public_result: PublicEvalResult | None
    round_stage: str


@dataclass(frozen=True)
class TrialExecution:
    """A complete, validated round-v7 candidate trial not yet ledger-committed."""

    record: Mapping[str, Any]
    decision: DecisionRecord
    train_result: TrainJobResult | None
    public_result: PublicEvalResult | None
    round_stage: str
    receipt_ref: TrialReceiptRef
    reused_receipt: bool = False


@dataclass(frozen=True)
class PreparedCandidateTrial:
    """Verified/materialized trial state before any train or eval side effect."""

    inputs: RoundInputs
    acceptance_protocol: AcceptanceProtocol
    ledger_epoch_hash: str
    authoring_mode: str
    authoring_mechanism_delta: Mapping[str, Any] | None
    authoring_mechanism_operation_delta: Mapping[str, Any] | None
    source_request_feedback: Mapping[str, Any] | None
    candidate: CandidatePackageRef | VerifiedCandidateArtifactRef
    record_candidate: (
        CandidatePackageRef
        | SealedCandidateArtifactRef
        | VerifiedCandidateArtifactRef
    )
    verification: Any
    surface: Any
    compiled_curriculum: Any
    jobs: Any
    task_invariants_valid: bool
    trial_journal: TrialReceiptJournal
    reused_execution: TrialExecution | None = None

    @property
    def ready_for_training(self) -> bool:
        return (
            self.verification.status == "pass"
            and self.task_invariants_valid
            and self.jobs is not None
            and self.reused_execution is None
        )


@dataclass(frozen=True)
class CandidateTrainExecution:
    prepared: PreparedCandidateTrial
    train_result: TrainJobResult
    train_receipt_ref: TrialReceiptRef
    eval_plan: EvalJobSpec


def run_candidate_round(inputs: RoundInputs) -> RoundResult:
    """Execute and exactly-once commit the legacy width-one round path."""

    execution = execute_candidate_trial(inputs)
    return commit_candidate_trial(inputs, execution)


def prepare_candidate_trial(inputs: RoundInputs) -> PreparedCandidateTrial:
    """Seal, verify, materialize and bind one trial without running a job."""

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
    ledger_epoch_hash = (
        primary_ledger_epoch_hash(inputs.session_genesis_hash)
        if inputs.ledger_epoch_hash is None
        else require_sha256_digest(
            inputs.ledger_epoch_hash,
            "RoundInputs.ledger_epoch_hash",
        )
    )
    if inputs.expected_initial_policy_state_hash is not None:
        require_sha256_digest(
            inputs.expected_initial_policy_state_hash,
            "RoundInputs.expected_initial_policy_state_hash",
        )
    if inputs.backend_execution_environment_manifest_hash is not None:
        require_sha256_digest(
            inputs.backend_execution_environment_manifest_hash,
            "RoundInputs.backend_execution_environment_manifest_hash",
        )
    if not isinstance(inputs.evaluation_binding, EvaluationBindingV1):
        raise TypeError("RoundInputs.evaluation_binding must be EvaluationBindingV1")
    inputs.backend_adapter.validate_session_evaluation_binding(
        task_spec=task_spec,
        evaluation_binding=inputs.evaluation_binding,
    )
    if inputs.current_best_eval_stats.evaluation_binding() != inputs.evaluation_binding:
        raise ValueError("round current-best evaluation binding mismatch")
    if inputs.parent_program_source_checksum is not None:
        require_sha256_checksum(
            inputs.parent_program_source_checksum,
            "RoundInputs.parent_program_source_checksum",
        )
    if (
        inputs.parent_program_source_checksum
        != inputs.current_best_eval_stats.program_source_checksum
    ):
        raise ValueError("round parent program source does not match current-best")
    authoring_mode = require_ledger_authoring_mode(inputs.authoring_mode)
    has_prompt = inputs.prompt_payload_ref is not None
    has_evidence = inputs.authoring_evidence_ref is not None
    if has_prompt != has_evidence:
        raise ValueError(
            "round prompt payload and authoring evidence must be provided together"
        )
    if authoring_mode == PROVIDER_AUTHORING_MODE and not has_prompt:
        raise ValueError(
            "provider-authored round requires prompt payload and authoring evidence"
        )
    if authoring_mode == NON_PROVIDER_AUTHORING_MODE and has_prompt:
        raise ValueError(
            "non-provider round cannot contain provider authoring artifacts"
        )
    authoring_evidence: Mapping[str, Any] | None = None
    authoring_mechanism_delta: Mapping[str, Any] | None = None
    authoring_mechanism_operation_delta: Mapping[str, Any] | None = None
    source_request_feedback: Mapping[str, Any] | None = None
    if has_prompt and has_evidence:
        assert inputs.prompt_payload_ref is not None
        assert inputs.authoring_evidence_ref is not None
        authoring_evidence = verify_authoring_evidence(
            inputs.authoring_evidence_ref,
            expected_round_id=inputs.round_id,
            expected_selected_prompt=inputs.prompt_payload_ref,
        )
        authoring_outcome = authoring_outcome_from_evidence(authoring_evidence)
        if authoring_outcome["stop_reason"] != "candidate_verified":
            raise ValueError(
                "candidate round requires a verified-candidate authoring outcome"
            )
        source_request_feedback = authoring_outcome[
            "source_request_feedback"
        ]

    candidate: CandidatePackageRef | VerifiedCandidateArtifactRef = inputs.candidate
    artifact_root = inputs.candidate_artifact_root or (
        inputs.ledger_path.parent / "candidate_artifacts"
    )
    artifact_verification = seal_and_verify_candidate(
        inputs.candidate,
        artifact_root=artifact_root,
        verifier=inputs.candidate_admission_policy,
    )
    verification = artifact_verification.result
    record_candidate: (
        CandidatePackageRef
        | SealedCandidateArtifactRef
        | VerifiedCandidateArtifactRef
    ) = inputs.candidate
    if artifact_verification.verified_artifact is not None:
        candidate = artifact_verification.verified_artifact
        record_candidate = artifact_verification.verified_artifact
    else:
        if artifact_verification.sealed_artifact is not None:
            record_candidate = artifact_verification.sealed_artifact
        candidate = replace(
            inputs.candidate,
            change_set=CandidateChangeSet(frozenset()),
        )
    if authoring_evidence is not None:
        verify_authoring_candidate_binding(
            authoring_evidence,
            expected_candidate=record_candidate.to_public_dict(),
        )
        authoring_mechanism_delta = project_selected_authoring_mechanism_delta(
            authoring_evidence
        )
        authoring_mechanism_operation_delta = (
            _project_authoring_mechanism_operation_delta(
                anchor_prompt_payload=verify_authoring_anchor_prompt(
                    authoring_evidence
                ),
                candidate=record_candidate,
            )
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
    task_invariants_valid = True
    if verification.status == "pass":
        if not isinstance(candidate, VerifiedCandidateArtifactRef):
            raise RuntimeError(
                "passed candidate verification did not produce an execution capability"
            )
        materialization_kwargs: dict[str, Any] = {}
        if inputs.backend_execution_environment_manifest_hash is not None:
            materialization_kwargs[
                "backend_execution_environment_manifest_hash"
            ] = inputs.backend_execution_environment_manifest_hash
        if inputs.train_attempt_scope is not None:
            materialization_kwargs["train_attempt_scope"] = inputs.train_attempt_scope
        if inputs.eval_attempt_scope is not None:
            materialization_kwargs["eval_attempt_scope"] = inputs.eval_attempt_scope
        if inputs.gpu_process_binding is not None:
            if not isinstance(inputs.gpu_process_binding, GpuProcessBinding):
                raise TypeError("RoundInputs.gpu_process_binding is invalid")
            materialization_kwargs["gpu_process_binding"] = (
                inputs.gpu_process_binding
            )
        jobs = inputs.backend_adapter.materialize_jobs(
            round_id=inputs.round_id,
            task_spec=task_spec,
            candidate=candidate,
            policy_initialization=inputs.policy_initialization,
            evaluation_binding=inputs.evaluation_binding,
            expected_initial_policy_state_hash=(
                inputs.expected_initial_policy_state_hash
            ),
            **materialization_kwargs,
        )
        if jobs.eval_spec.evaluation_binding != inputs.evaluation_binding:
            raise ValueError("materialized eval job evaluation binding mismatch")
        try:
            enforce_task_invariants(
                expected=task_spec.invariants,
                actual=jobs.task_invariants or task_spec.invariants,
            )
        except ValueError:
            task_invariants_valid = False

    trial_journal = _build_trial_receipt_journal(
        inputs=inputs,
        ledger_epoch_hash=ledger_epoch_hash,
        authoring_mode=authoring_mode,
        candidate=record_candidate,
        verification=verification.to_public_dict(),
        surface=surface.to_public_dict(),
        curriculum=compiled_curriculum.to_public_dict(),
        jobs=jobs,
        mechanism_operation_delta=authoring_mechanism_operation_delta,
    )
    trial_journal.bind_contract()
    loaded_execution = trial_journal.load_execution()
    reused_execution = None
    if loaded_execution is not None:
        reused_execution = _trial_execution_from_loaded(
            inputs=inputs,
            acceptance_protocol=acceptance_protocol,
            authoring_mode=authoring_mode,
            loaded=loaded_execution,
        )

    return PreparedCandidateTrial(
        inputs=inputs,
        acceptance_protocol=acceptance_protocol,
        ledger_epoch_hash=ledger_epoch_hash,
        authoring_mode=authoring_mode,
        authoring_mechanism_delta=authoring_mechanism_delta,
        authoring_mechanism_operation_delta=(
            authoring_mechanism_operation_delta
        ),
        source_request_feedback=source_request_feedback,
        candidate=candidate,
        record_candidate=record_candidate,
        verification=verification,
        surface=surface,
        compiled_curriculum=compiled_curriculum,
        jobs=jobs,
        task_invariants_valid=task_invariants_valid,
        trial_journal=trial_journal,
        reused_execution=reused_execution,
    )


def run_prepared_candidate_train(
    prepared: PreparedCandidateTrial,
) -> CandidateTrainExecution:
    """Run or reload the single train node for an eligible prepared trial."""

    if not isinstance(prepared, PreparedCandidateTrial):
        raise TypeError("prepared candidate trial is invalid")
    if not prepared.ready_for_training:
        raise ValueError("prepared candidate trial is not eligible for training")
    assert prepared.jobs is not None
    loaded_train = prepared.trial_journal.load_train(prepared.jobs.train_spec)
    if loaded_train is None:
        train_result = prepared.inputs.runner.run_train(prepared.jobs.train_spec)
        train_receipt_ref = prepared.trial_journal.publish_train(
            prepared.jobs.train_spec,
            train_result,
        )
    else:
        train_result, train_receipt_ref = loaded_train
    eval_plan = bind_eval_spec_to_checkpoint(
        prepared.jobs.eval_spec,
        train_result.checkpoint,
    )
    return CandidateTrainExecution(
        prepared=prepared,
        train_result=train_result,
        train_receipt_ref=train_receipt_ref,
        eval_plan=eval_plan,
    )


def project_candidate_full_eval_result(
    train_execution: CandidateTrainExecution,
    raw_eval_result: PublicEvalResult,
) -> PublicEvalResult:
    """Apply the existing task projector to one complete runner result."""

    if not isinstance(train_execution, CandidateTrainExecution):
        raise TypeError("candidate train execution is invalid")
    prepared = train_execution.prepared
    eval_plan = train_execution.eval_plan
    projected_result = attach_eval_protocol_hash(
        prepared.inputs.task_plugin.project_candidate_public_result(
            public_result=raw_eval_result,
            train_summary=train_execution.train_result.summary,
            mechanism_operation_delta=(
                prepared.authoring_mechanism_operation_delta
            ),
        ),
        eval_protocol_hash=eval_plan.evaluation_protocol().protocol_hash,
    )
    return _validate_public_eval_result(
        attach_evaluation_binding(
            projected_result,
            binding=eval_plan.evaluation_binding,
            subject=eval_plan.evaluation_subject(),
            subject_manifest_hash=eval_plan.subject_manifest_hash,
            program_source_checksum=eval_plan.program_source_checksum,
            training_binding_hash=eval_plan.training_binding_hash,
            capability_profile_hash=eval_plan.capability_profile_hash,
        ),
        acceptance_protocol=prepared.acceptance_protocol,
    )


def finalize_prepared_candidate_trial(
    prepared: PreparedCandidateTrial,
    *,
    train_execution: CandidateTrainExecution | None = None,
    public_result: PublicEvalResult | None = None,
    failure_reason: str | None = None,
    runner_error: Mapping[str, Any] | None = None,
    eval_receipt_ref: TrialReceiptRef | None = None,
) -> TrialExecution:
    """Build the sole round-v7 trial projection from already-owned stage outputs."""

    if not isinstance(prepared, PreparedCandidateTrial):
        raise TypeError("prepared candidate trial is invalid")
    if prepared.reused_execution is not None:
        if any(
            value is not None
            for value in (
                train_execution,
                public_result,
                failure_reason,
                runner_error,
                eval_receipt_ref,
            )
        ):
            raise ValueError("reused trial execution cannot be finalized again")
        return prepared.reused_execution
    if train_execution is not None and train_execution.prepared is not prepared:
        raise ValueError("train execution belongs to a different prepared trial")

    inputs = prepared.inputs
    task_spec = inputs.task_plugin.task_spec
    train_result = (
        train_execution.train_result if train_execution is not None else None
    )
    executed_eval_spec = (
        train_execution.eval_plan if train_execution is not None else None
    )
    train_receipt_ref = (
        train_execution.train_receipt_ref if train_execution is not None else None
    )
    surface = prepared.surface
    surface_policy = inputs.task_plugin.candidate_surface_policy()
    runtime_evidence = RuntimeActivationEvidence()
    if train_result is not None:
        runtime_evidence = RuntimeActivationEvidence.from_mapping(
            train_result.summary.get("runtime_activation_summary")
        )
    runtime_activation_gate = MechanismEvidenceGate().evaluate(
        surface=surface,
        evidence=runtime_evidence,
    )
    ablation_summary = _build_ablation_summary(
        candidate=prepared.candidate,
        surface_policy=surface_policy,
        triggered=False,
        reason="not_evaluated",
    )
    mechanism_attribution = build_mechanism_attribution_summary(
        runtime_activation_gate=runtime_activation_gate,
        ablation_summary=ablation_summary,
        authoring_mechanism_delta=prepared.authoring_mechanism_delta,
    )

    if prepared.verification.status != "pass":
        if train_execution is not None or public_result is not None:
            raise ValueError("failed verifier trial cannot carry execution results")
        decision = reject_without_eval(
            reason="candidate_verifier_failed",
            surface_classification=surface.classification,
        )
    elif not prepared.task_invariants_valid:
        if train_execution is not None or public_result is not None:
            raise ValueError("invariant-failed trial cannot carry execution results")
        decision = reject_without_eval(
            reason="task_invariant_violation",
            surface_classification=surface.classification,
        )
    elif failure_reason is not None:
        if public_result is not None:
            raise ValueError("failed trial cannot carry a public result")
        decision = reject_without_eval(
            reason=failure_reason,
            surface_classification=surface.classification,
        )
    else:
        if train_execution is None or public_result is None:
            raise ValueError("evaluated trial requires train and public results")
        public_result = _validate_public_eval_result(
            public_result,
            acceptance_protocol=prepared.acceptance_protocol,
        )
        preliminary_decision = decide_candidate(
            surface=surface,
            public_result=public_result,
            current_best=inputs.current_best_eval_stats,
            acceptance_protocol=prepared.acceptance_protocol,
        )
        potential_accept = (
            preliminary_decision.outcome
            is DecisionOutcome.POLICY_IMPROVEMENT_ACCEPT
        )
        ablation_summary = _build_ablation_summary(
            candidate=prepared.candidate,
            surface_policy=surface_policy,
            triggered=potential_accept,
            reason=(
                "potential_accept_shadow_check"
                if potential_accept
                else "not_potential_accept"
            ),
        )
        mechanism_attribution = build_mechanism_attribution_summary(
            runtime_activation_gate=runtime_activation_gate,
            ablation_summary=ablation_summary,
            authoring_mechanism_delta=prepared.authoring_mechanism_delta,
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

    private_artifacts = _private_artifacts_for_round(
        candidate=prepared.record_candidate,
        jobs=prepared.jobs,
        train_result=train_result,
        executed_eval_spec=executed_eval_spec,
    )
    private_artifact_dicts = [
        artifact.to_private_dict() for artifact in private_artifacts
    ]
    if inputs.authoring_evidence_ref is not None:
        private_artifact_dicts.insert(
            0,
            inputs.authoring_evidence_ref.to_private_dict(),
        )
    record = {
        "schema_version": LEDGER_RECORD_SCHEMA_VERSION,
        "authoring_mode": prepared.authoring_mode,
        "round_id": inputs.round_id,
        "parent_program_source_checksum": (
            inputs.parent_program_source_checksum
        ),
        "session_genesis_hash": inputs.session_genesis_hash,
        "ledger_epoch_hash": prepared.ledger_epoch_hash,
        "evaluation_binding": inputs.evaluation_binding.to_public_dict(),
        "prompt_payload": (
            inputs.prompt_payload_ref.to_public_dict()
            if inputs.prompt_payload_ref is not None
            else None
        ),
        "public_context": sanitize_public_context(inputs.public_context),
        "source_request_feedback": prepared.source_request_feedback,
        "task": {"task_id": task_spec.task_id},
        "acceptance_protocol": prepared.acceptance_protocol.to_public_dict(),
        "backend": {"backend_id": inputs.backend_adapter.backend_id},
        "candidate": prepared.record_candidate.to_public_dict(),
        "private_artifacts": private_artifact_dicts,
        "verifier": prepared.verification.to_public_dict(),
        "candidate_surface": surface.to_public_dict(),
        "curriculum": prepared.compiled_curriculum.to_public_dict(),
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
                "train": prepared.jobs.train_spec.to_public_dict(),
                "eval": (
                    executed_eval_spec or prepared.jobs.eval_spec
                ).to_public_dict(),
            }
            if prepared.jobs is not None
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
    round_stage = _validate_trial_record(
        inputs=inputs,
        acceptance_protocol=prepared.acceptance_protocol,
        authoring_mode=prepared.authoring_mode,
        record=record,
    )
    receipt_ref = prepared.trial_journal.publish_execution(
        record=record,
        round_stage=round_stage,
        train_receipt_ref=train_receipt_ref,
        eval_receipt_ref=eval_receipt_ref,
        mechanism_operation_delta=prepared.authoring_mechanism_operation_delta,
    )
    return TrialExecution(
        record=record,
        decision=decision,
        train_result=train_result,
        public_result=public_result,
        round_stage=round_stage,
        receipt_ref=receipt_ref,
    )


def execute_candidate_trial(inputs: RoundInputs) -> TrialExecution:
    """Execute one candidate without appending a scientific ledger record."""

    prepared = prepare_candidate_trial(inputs)
    if prepared.reused_execution is not None:
        return prepared.reused_execution
    if prepared.verification.status != "pass" or not prepared.task_invariants_valid:
        return finalize_prepared_candidate_trial(prepared)
    try:
        train_execution = run_prepared_candidate_train(prepared)
    except CandidateRuntimeError as exc:
        if not inputs.record_candidate_runtime_failures:
            raise
        return finalize_prepared_candidate_trial(
            prepared,
            failure_reason="candidate_runtime_failed",
            runner_error=project_public_runner_error(exc),
        )
    except (RunnerExecutionError, ValueError, TypeError):
        raise

    eval_receipt_ref: TrialReceiptRef | None = None
    loaded_eval = prepared.trial_journal.load_eval(train_execution.eval_plan)
    if loaded_eval is not None:
        raw_eval_result, eval_receipt_ref = loaded_eval
    try:
        if loaded_eval is None:
            raw_eval_result = inputs.runner.run_eval(train_execution.eval_plan)
            eval_receipt_ref = prepared.trial_journal.publish_eval(
                train_execution.eval_plan,
                raw_eval_result,
            )
        public_result = project_candidate_full_eval_result(
            train_execution,
            raw_eval_result,
        )
    except CandidateRuntimeError as exc:
        if not inputs.record_candidate_runtime_failures:
            raise
        return finalize_prepared_candidate_trial(
            prepared,
            train_execution=train_execution,
            failure_reason="candidate_runtime_failed",
            runner_error=project_public_runner_error(exc),
            eval_receipt_ref=eval_receipt_ref,
        )
    except (RunnerExecutionError, ValueError, TypeError) as exc:
        return finalize_prepared_candidate_trial(
            prepared,
            train_execution=train_execution,
            failure_reason="eval_incomplete",
            runner_error=project_public_runner_error(exc),
            eval_receipt_ref=eval_receipt_ref,
        )
    return finalize_prepared_candidate_trial(
        prepared,
        train_execution=train_execution,
        public_result=public_result,
        eval_receipt_ref=eval_receipt_ref,
    )


def commit_candidate_trial(
    inputs: RoundInputs,
    execution: TrialExecution,
) -> RoundResult:
    """Validate one final receipt and perform the sole width-one ledger write."""

    if not isinstance(execution, TrialExecution):
        raise TypeError("candidate trial execution is invalid")
    contract, loaded = load_trial_execution_receipt(execution.receipt_ref)
    _validate_trial_contract_for_commit(inputs=inputs, contract=contract)
    if (
        dict(loaded.record) != dict(execution.record)
        or loaded.decision != execution.decision
        or loaded.train_result != execution.train_result
        or loaded.public_result != execution.public_result
        or loaded.round_stage != execution.round_stage
    ):
        raise ValueError("candidate trial execution differs from its final receipt")
    acceptance_protocol = inputs.task_plugin.task_spec.acceptance_protocol
    if acceptance_protocol is None:
        raise ValueError("task does not define a fixed acceptance protocol")
    authoring_mode = require_ledger_authoring_mode(inputs.authoring_mode)
    round_stage = _validate_trial_record(
        inputs=inputs,
        acceptance_protocol=acceptance_protocol,
        authoring_mode=authoring_mode,
        record=loaded.record,
    )
    if round_stage != execution.round_stage:
        raise ValueError("candidate trial round stage changed before commit")
    append_ledger_record_once(inputs.ledger_path, loaded.record)
    return RoundResult(
        decision=loaded.decision,
        train_result=loaded.train_result,
        public_result=loaded.public_result,
        round_stage=loaded.round_stage,
    )


def _validate_trial_contract_for_commit(
    *,
    inputs: RoundInputs,
    contract: Mapping[str, Any],
) -> None:
    expected_epoch_hash = inputs.ledger_epoch_hash or primary_ledger_epoch_hash(
        inputs.session_genesis_hash
    )
    expected_identity = {
        "session_genesis_hash": inputs.session_genesis_hash,
        "ledger_epoch_hash": expected_epoch_hash,
        "round_id": inputs.round_id,
    }
    for field_name, expected in expected_identity.items():
        if contract.get(field_name) != expected:
            raise ValueError(
                f"trial execution contract {field_name} does not match commit inputs"
            )


def _project_authoring_mechanism_operation_delta(
    *,
    anchor_prompt_payload: Mapping[str, Any],
    candidate: (
        CandidatePackageRef
        | SealedCandidateArtifactRef
        | VerifiedCandidateArtifactRef
    ),
) -> Mapping[str, Any] | None:
    anchor_source, anchor_hash, _ = select_authoring_anchor_source(
        anchor_prompt_payload
    )
    if anchor_source is None:
        return None
    if anchor_hash is None:
        raise ValueError("authoring anchor source has no checksum")
    source_bytes = read_bounded_regular_file(
        Path(candidate.path) / CANONICAL_TRAINING_COMPONENTS_PATH,
        max_bytes=MAX_CANDIDATE_EXECUTION_FILE_BYTES,
        label="candidate training components source",
    )
    try:
        candidate_source = source_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("candidate training components source is not UTF-8") from exc
    operation_delta = project_single_component_operation_delta(
        candidate_source=candidate_source,
        anchor_source=anchor_source,
    )
    if operation_delta["anchor_source_checksum"] != f"sha256:{anchor_hash}":
        raise ValueError("mechanism operation delta anchor checksum mismatch")
    expected_candidate_checksum = getattr(
        candidate,
        "program_source_checksum",
        None,
    )
    if (
        expected_candidate_checksum is not None
        and operation_delta["candidate_source_checksum"]
        != expected_candidate_checksum
    ):
        raise ValueError("mechanism operation delta candidate checksum mismatch")
    return operation_delta


def _build_trial_receipt_journal(
    *,
    inputs: RoundInputs,
    ledger_epoch_hash: str,
    authoring_mode: str,
    candidate: (
        CandidatePackageRef
        | SealedCandidateArtifactRef
        | VerifiedCandidateArtifactRef
    ),
    verification: Mapping[str, Any],
    surface: Mapping[str, Any],
    curriculum: Mapping[str, Any],
    jobs: Any,
    mechanism_operation_delta: Mapping[str, Any] | None,
) -> TrialReceiptJournal:
    acceptance_protocol = inputs.task_plugin.task_spec.acceptance_protocol
    if acceptance_protocol is None:
        raise ValueError("task does not define a fixed acceptance protocol")
    authoring_artifacts = None
    if inputs.prompt_payload_ref is not None:
        assert inputs.authoring_evidence_ref is not None
        authoring_artifacts = {
            "prompt_payload": {
                "schema_version": inputs.prompt_payload_ref.schema_version,
                "checksum": inputs.prompt_payload_ref.checksum,
                "size_bytes": inputs.prompt_payload_ref.size_bytes,
            },
            "authoring_evidence": {
                "checksum": inputs.authoring_evidence_ref.checksum,
                "size_bytes": inputs.authoring_evidence_ref.size_bytes,
            },
        }
    contract = {
        "schema_version": TRIAL_EXECUTION_CONTRACT_SCHEMA_VERSION,
        "round_id": inputs.round_id,
        "session_genesis_hash": inputs.session_genesis_hash,
        "ledger_epoch_hash": ledger_epoch_hash,
        "parent_program_source_checksum": (
            inputs.parent_program_source_checksum
        ),
        "task": {"task_id": inputs.task_plugin.task_spec.task_id},
        "backend": {"backend_id": inputs.backend_adapter.backend_id},
        "evaluation_binding": inputs.evaluation_binding.to_public_dict(),
        "acceptance_protocol": acceptance_protocol.to_public_dict(),
        "current_best": inputs.current_best_eval_stats.to_public_dict(),
        "public_context": sanitize_public_context(inputs.public_context),
        "authoring_mode": authoring_mode,
        "authoring_artifacts": authoring_artifacts,
        "candidate": candidate.to_public_dict(),
        "verification": dict(verification),
        "candidate_surface": dict(surface),
        "curriculum": dict(curriculum),
        "jobs": (
            {
                "train": jobs.train_spec.to_public_dict(),
                "eval_plan": jobs.eval_spec.to_public_dict(),
            }
            if jobs is not None
            else None
        ),
        "expected_initial_policy_state_hash": (
            inputs.expected_initial_policy_state_hash
        ),
        "backend_execution_environment_manifest_hash": (
            inputs.backend_execution_environment_manifest_hash
        ),
        "record_candidate_runtime_failures": (
            inputs.record_candidate_runtime_failures
        ),
        "mechanism_operation_delta": (
            dict(mechanism_operation_delta)
            if mechanism_operation_delta is not None
            else None
        ),
    }
    receipt_root = inputs.trial_receipt_root or (
        inputs.ledger_path.parent / "trial_receipts"
    )
    return TrialReceiptJournal(receipt_root, contract=contract)


def _trial_execution_from_loaded(
    *,
    inputs: RoundInputs,
    acceptance_protocol: AcceptanceProtocol,
    authoring_mode: str,
    loaded: LoadedTrialExecution,
) -> TrialExecution:
    round_stage = _validate_trial_record(
        inputs=inputs,
        acceptance_protocol=acceptance_protocol,
        authoring_mode=authoring_mode,
        record=loaded.record,
    )
    if round_stage != loaded.round_stage:
        raise ValueError("reused trial receipt round stage mismatch")
    return TrialExecution(
        record=dict(loaded.record),
        decision=loaded.decision,
        train_result=loaded.train_result,
        public_result=loaded.public_result,
        round_stage=loaded.round_stage,
        receipt_ref=loaded.receipt_ref,
        reused_receipt=True,
    )


def _validate_trial_record(
    *,
    inputs: RoundInputs,
    acceptance_protocol: AcceptanceProtocol,
    authoring_mode: str,
    record: Mapping[str, Any],
) -> str:
    expected_epoch_hash = inputs.ledger_epoch_hash or primary_ledger_epoch_hash(
        inputs.session_genesis_hash
    )
    expected_prompt_payload = (
        inputs.prompt_payload_ref.to_public_dict()
        if inputs.prompt_payload_ref is not None
        else None
    )
    expected_identity = {
        "schema_version": LEDGER_RECORD_SCHEMA_VERSION,
        "authoring_mode": authoring_mode,
        "round_id": inputs.round_id,
        "parent_program_source_checksum": (
            inputs.parent_program_source_checksum
        ),
        "session_genesis_hash": inputs.session_genesis_hash,
        "ledger_epoch_hash": expected_epoch_hash,
        "evaluation_binding": inputs.evaluation_binding.to_public_dict(),
        "prompt_payload": expected_prompt_payload,
        "public_context": sanitize_public_context(inputs.public_context),
        "task": {"task_id": inputs.task_plugin.task_spec.task_id},
        "acceptance_protocol": acceptance_protocol.to_public_dict(),
        "backend": {"backend_id": inputs.backend_adapter.backend_id},
    }
    for field_name, expected in expected_identity.items():
        if record.get(field_name) != expected:
            raise ValueError(
                f"candidate trial record {field_name} does not match its inputs"
            )
    candidate = record.get("candidate")
    if (
        not isinstance(candidate, Mapping)
        or candidate.get("candidate_id") != inputs.candidate.candidate_id
    ):
        raise ValueError("candidate trial record candidate identity mismatch")
    decision_validation = validate_decision_record(
        record,
        acceptance_protocol=acceptance_protocol,
        expected_session_genesis_hash=inputs.session_genesis_hash,
        expected_evaluation_binding=inputs.evaluation_binding,
        expected_initial_policy_state_hash=(
            inputs.expected_initial_policy_state_hash
        ),
    )
    decision_validation.raise_for_failure()
    prompt_artifact_reasons = validate_ledger_prompt_artifacts(
        record,
        expected_authoring_mode=authoring_mode,
    )
    if prompt_artifact_reasons:
        raise ValueError(
            "invalid round prompt artifacts: "
            + ", ".join(prompt_artifact_reasons)
        )
    _assert_round_public_sections_safe(record)
    return decision_validation.round_stage.stage


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
    candidate: (
        CandidatePackageRef
        | SealedCandidateArtifactRef
        | VerifiedCandidateArtifactRef
    ),
    jobs: Any,
    train_result: TrainJobResult | None,
    executed_eval_spec: EvalJobSpec | None,
) -> tuple[PrivateArtifactRef, ...]:
    candidate_artifact_kind = (
        "candidate_artifact"
        if isinstance(
            candidate,
            (SealedCandidateArtifactRef, VerifiedCandidateArtifactRef),
        )
        else "candidate_package"
    )
    refs = [
        PrivateArtifactRef(
            kind=candidate_artifact_kind,
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
