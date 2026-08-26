"""Task-owned Stair phase evidence and full-result assembly."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.evaluation.phases import (
    EVALUATION_PHASE_SUMMARY_SCHEMA_VERSION,
    EvaluationAssembly,
    EvaluationPhase,
    EvaluationPhaseReceiptBinding,
    EvaluationPhaseResult,
)
from true_rl_agent.evaluation.sealed_validation import build_sealed_validation_gate
from true_rl_agent.runner.checkpoint import validate_checkpoint_safety_report
from true_rl_agent.runner.eval_isolation import build_candidate_isolation_gate
from true_rl_agent.runner.job_identity import JobIdentity
from true_rl_agent.runner.specs import EvalJobSpec, EvalPhaseJobSpec
from true_rl_agent.tasks.stair_climbing.config import (
    STAIR_TASK_ID,
    StairClimbingTaskConfig,
)
from true_rl_agent.tasks.stair_climbing.evaluation_contract import (
    build_stair_sealed_rollout_evidence,
    canonical_stair_sealed_summary,
    validate_stair_split_summary,
)
from true_rl_agent.tasks.stair_climbing.generalization_contract import (
    validate_stair_generalization_summary,
)
from true_rl_agent.tasks.stair_climbing.isaaclab_execution import (
    StairEvaluationExecutionResult,
)
from true_rl_agent.tasks.stair_climbing.phase_contract import (
    STAIR_PHASE_EVIDENCE_SCHEMA_VERSION,
)
from true_rl_agent.tasks.stair_climbing.seed_policy import (
    STAIR_FORMAL_EVAL_NAMESPACE,
)
from true_rl_agent.tasks.stair_climbing.selection import (
    StairPublicDevScreenEvidence,
    StairSelectionResult,
    build_stair_public_dev_screen_evidence,
    replay_stair_public_dev_winner,
)


STAIR_FULL_EVAL_SUMMARY_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.eval_summary.v1"
)


def build_stair_eval_phase_summary(
    *,
    config: StairClimbingTaskConfig,
    identity: JobIdentity,
    evaluation_plan_hash: str,
    phase: EvaluationPhase,
    execution: StairEvaluationExecutionResult,
    evaluation_binding: Any,
    checkpoint_safety_report: Mapping[str, Any],
    expected_seed_namespace: str = STAIR_FORMAL_EVAL_NAMESPACE,
) -> dict[str, Any]:
    """Build the private subprocess summary for exactly one Stair split."""

    phase = EvaluationPhase.parse(phase)
    if set(execution.summaries) != {phase.value}:
        raise ValueError("Stair phase execution contains the wrong split set")
    task_evidence = build_stair_phase_task_evidence(
        config=config,
        phase=phase,
        execution=execution,
        expected_seed_namespace=expected_seed_namespace,
    )
    return {
        "schema_version": EVALUATION_PHASE_SUMMARY_SCHEMA_VERSION,
        **identity.to_public_dict(),
        "evaluation_protocol": config.evaluation_protocol().to_public_dict(),
        "evaluation_binding": evaluation_binding.to_public_dict(),
        "evaluation_phase": phase.value,
        "evaluation_plan_hash": evaluation_plan_hash,
        "execution_device": execution.execution_device,
        "backend_execution_environment_verification": dict(
            execution.backend_execution_environment_verification
        ),
        "environment_source_preflight": dict(execution.source_manifest_preflight),
        "evaluation_execution_cost": execution.execution_cost_artifact(),
        "status": "pass",
        "task_evidence": task_evidence,
        "task_evidence_hash": canonical_json_hash(task_evidence),
        "checkpoint_safety_report": dict(checkpoint_safety_report),
        "candidate_code_loaded_in_eval": False,
        "candidate_package_loaded": False,
        "candidate_package_loaded_in_eval": False,
        "candidate_path_argument_present": False,
        "errors": [],
    }


def build_stair_phase_task_evidence(
    *,
    config: StairClimbingTaskConfig,
    phase: EvaluationPhase,
    execution: StairEvaluationExecutionResult,
    expected_seed_namespace: str = STAIR_FORMAL_EVAL_NAMESPACE,
) -> dict[str, Any]:
    phase = EvaluationPhase.parse(phase)
    if set(execution.summaries) != {phase.value}:
        raise ValueError("Stair phase evidence requires exactly its declared split")
    _require_stair_execution_domain(
        config,
        execution,
        phases={phase},
        expected_seed_namespace=expected_seed_namespace,
    )
    raw_summary = execution.public_summaries()[phase.value]
    canonical_summary = _canonical_phase_summary(
        config=config,
        phase=phase,
        summary=raw_summary,
    )
    return {
        "schema_version": STAIR_PHASE_EVIDENCE_SCHEMA_VERSION,
        "phase": phase.value,
        "execution_identity": execution.execution_identity(),
        "summary": canonical_summary,
    }


def build_stair_full_eval_summary(
    *,
    config: StairClimbingTaskConfig,
    identity: JobIdentity,
    evaluation_binding: Any,
    execution: StairEvaluationExecutionResult,
    checkpoint_safety_report: Mapping[str, Any],
    subject_manifest_hash: str,
    program_source_checksum: str,
    training_binding_hash: str,
    capability_profile_hash: str,
) -> dict[str, Any]:
    """Build the legacy full summary from one all-split execution."""

    expected = set(EvaluationPhase)
    _require_stair_execution_domain(config, execution, phases=expected)
    summaries = execution.public_summaries()
    public_dev = _canonical_phase_summary(
        config=config,
        phase=EvaluationPhase.PUBLIC_DEV,
        summary=summaries[EvaluationPhase.PUBLIC_DEV.value],
    )
    public_accept = _canonical_phase_summary(
        config=config,
        phase=EvaluationPhase.PUBLIC_ACCEPT,
        summary=summaries[EvaluationPhase.PUBLIC_ACCEPT.value],
    )
    sealed = _canonical_phase_summary(
        config=config,
        phase=EvaluationPhase.SEALED_VALIDATION,
        summary=summaries[EvaluationPhase.SEALED_VALIDATION.value],
    )
    return _build_complete_stair_eval_summary(
        config=config,
        identity=identity,
        evaluation_binding=evaluation_binding,
        execution_device=execution.execution_device,
        environment_source_preflight=execution.source_manifest_preflight,
        backend_execution_environment_verification=(
            execution.backend_execution_environment_verification
        ),
        evaluation_execution_cost=execution.execution_cost_artifact(),
        checkpoint_safety_report=checkpoint_safety_report,
        evaluation_subject=identity.subject,
        subject_manifest_hash=subject_manifest_hash,
        program_source_checksum=program_source_checksum,
        training_binding_hash=training_binding_hash,
        capability_profile_hash=capability_profile_hash,
        checkpoint_hash=execution.checkpoint_hash,
        public_dev_summary=public_dev,
        public_accept_summary=public_accept,
        sealed_summary=sealed,
    )


def assemble_stair_eval_phase_results(
    *,
    config: StairClimbingTaskConfig,
    eval_plan: EvalJobSpec,
    trial_id: str,
    selection_result: StairSelectionResult,
    selection_screens: Sequence[StairPublicDevScreenEvidence],
    guard_reference: StairPublicDevScreenEvidence,
    phase_results: Sequence[EvaluationPhaseResult],
    phase_receipt_refs: Mapping[EvaluationPhase | str, Mapping[str, Any]],
) -> EvaluationAssembly:
    """Assemble one winner's three validated phases into the normal raw summary."""

    if not isinstance(eval_plan, EvalJobSpec):
        raise TypeError("Stair phase assembly requires an EvalJobSpec plan")
    _require_eval_plan_matches_config(config, eval_plan)
    checkpoint = eval_plan.execution_checkpoint()
    by_phase: dict[EvaluationPhase, EvaluationPhaseResult] = {}
    for result in phase_results:
        if not isinstance(result, EvaluationPhaseResult):
            raise TypeError("Stair phase assembly received an invalid result")
        if result.phase in by_phase:
            raise ValueError("Stair phase assembly contains a duplicate phase")
        _require_phase_result_matches_plan(result, eval_plan)
        by_phase[result.phase] = result
    if set(by_phase) != set(EvaluationPhase):
        raise ValueError("Stair phase assembly requires public-dev, public-accept, and sealed")

    if not isinstance(trial_id, str) or not trial_id:
        raise ValueError("Stair formal assembly trial_id is required")
    if not isinstance(selection_result, StairSelectionResult):
        raise TypeError("Stair formal assembly requires a selection result")
    replayed_selection = replay_stair_public_dev_winner(
        config=config,
        screens=selection_screens,
        guard_reference=guard_reference,
        result_schema_version=selection_result.schema_version,
    )
    if selection_result.contract_hash != replayed_selection.contract_hash:
        raise ValueError("Stair formal assembly selection contract mismatch")
    if replayed_selection != selection_result:
        raise ValueError("Stair formal assembly selection replay mismatch")
    if selection_result.selected_trial_id != trial_id:
        raise ValueError("Stair formal assembly is not bound to the selected trial")
    phase_projection = build_stair_public_dev_screen_evidence(
        config=config,
        trial_id=trial_id,
        phase_result=by_phase[EvaluationPhase.PUBLIC_DEV],
        screen_role="guard_reference",
    )
    selected_matches = [
        screen for screen in selection_screens if screen.trial_id == trial_id
    ]
    if len(selected_matches) != 1:
        raise ValueError("Stair formal assembly selected screen is not unique")
    selected_screen = selected_matches[0]
    selected_phase_fields = selected_screen.to_selector_dict()
    expected_phase_fields = phase_projection.to_selector_dict()
    for field_name in (
        "screen_role",
        "train_receipt_checksum",
        "runtime_effect_evidence",
    ):
        selected_phase_fields.pop(field_name)
        expected_phase_fields.pop(field_name)
    if selected_phase_fields != expected_phase_fields:
        raise ValueError("Stair formal assembly selected screen phase mismatch")
    if selected_screen.evidence_hash not in selection_result.screen_evidence_hashes:
        raise ValueError("Stair formal assembly selection evidence mismatch")

    canonical = {
        phase: canonical_stair_phase_result_summary(
            config=config,
            expected_phase=phase,
            result=by_phase[phase],
        )
        for phase in EvaluationPhase
    }
    first = by_phase[EvaluationPhase.PUBLIC_DEV]
    for other in by_phase.values():
        if (
            dict(other.environment_source_preflight)
            != dict(first.environment_source_preflight)
            or dict(other.backend_execution_environment_verification)
            != dict(first.backend_execution_environment_verification)
            or dict(other.checkpoint_safety_report)
            != dict(first.checkpoint_safety_report)
        ):
            raise ValueError("Stair phase execution environment evidence differs")

    summary = _build_complete_stair_eval_summary(
        config=config,
        identity=eval_plan.job_identity(),
        evaluation_binding=eval_plan.evaluation_binding,
        execution_device=eval_plan.execution_device,
        environment_source_preflight=first.environment_source_preflight,
        backend_execution_environment_verification=(
            first.backend_execution_environment_verification
        ),
        evaluation_execution_cost=_merge_phase_costs(by_phase),
        checkpoint_safety_report=first.checkpoint_safety_report,
        evaluation_subject=eval_plan.subject,
        subject_manifest_hash=eval_plan.subject_manifest_hash,
        program_source_checksum=eval_plan.program_source_checksum,
        training_binding_hash=eval_plan.training_binding_hash,
        capability_profile_hash=eval_plan.capability_profile_hash,
        checkpoint_hash=checkpoint.checksum,
        public_dev_summary=canonical[EvaluationPhase.PUBLIC_DEV],
        public_accept_summary=canonical[EvaluationPhase.PUBLIC_ACCEPT],
        sealed_summary=canonical[EvaluationPhase.SEALED_VALIDATION],
    )
    receipt_refs: dict[EvaluationPhase, Mapping[str, Any]] = {}
    if not isinstance(phase_receipt_refs, Mapping):
        raise TypeError("Stair phase receipt refs must be an object")
    for raw_phase, raw_ref in phase_receipt_refs.items():
        phase = EvaluationPhase.parse(raw_phase)
        if phase in receipt_refs:
            raise ValueError("Stair phase receipt refs contain a duplicate phase")
        if not isinstance(raw_ref, Mapping):
            raise TypeError("Stair phase receipt ref must be an object")
        receipt_refs[phase] = raw_ref
    if set(receipt_refs) != set(EvaluationPhase):
        raise ValueError("Stair assembly requires one receipt ref per phase")

    bindings: list[EvaluationPhaseReceiptBinding] = []
    for phase in EvaluationPhase:
        raw_ref = receipt_refs[phase]
        if set(raw_ref) != {
            "schema_version",
            "kind",
            "path",
            "checksum",
            "size_bytes",
        } or raw_ref.get("kind") != "eval_phase":
            raise ValueError("Stair phase receipt ref contract mismatch")
        result = by_phase[phase]
        phase_spec = EvalPhaseJobSpec(
            job_id=result.job_id,
            evaluation_plan=eval_plan,
            phase=phase,
        )
        bindings.append(
            EvaluationPhaseReceiptBinding(
                phase=phase,
                job_id=result.job_id,
                path=raw_ref.get("path"),
                checksum=raw_ref.get("checksum"),
                size_bytes=raw_ref.get("size_bytes"),
                ref_schema_version=raw_ref.get("schema_version"),
                job_spec_hash=canonical_json_hash(phase_spec.to_public_dict()),
                phase_result_hash=canonical_json_hash(result.to_private_dict()),
                resource_binding_hash=result.resource_binding_hash,
            )
        )
    return EvaluationAssembly(
        evaluation_plan_hash=canonical_json_hash(eval_plan.to_public_dict()),
        selected_trial_id=trial_id,
        selection=selection_result.to_public_dict(),
        phase_receipts=tuple(bindings),
        summary=summary,
    )


def _build_complete_stair_eval_summary(
    *,
    config: StairClimbingTaskConfig,
    identity: JobIdentity,
    evaluation_binding: Any,
    execution_device: str,
    environment_source_preflight: Mapping[str, Any],
    backend_execution_environment_verification: Mapping[str, Any],
    evaluation_execution_cost: Mapping[str, Any],
    checkpoint_safety_report: Mapping[str, Any],
    evaluation_subject: Any,
    subject_manifest_hash: str | None,
    program_source_checksum: str,
    training_binding_hash: str,
    capability_profile_hash: str,
    checkpoint_hash: str,
    public_dev_summary: Mapping[str, Any],
    public_accept_summary: Mapping[str, Any],
    sealed_summary: Mapping[str, Any],
) -> dict[str, Any]:
    evaluation_protocol = config.evaluation_protocol()
    evaluation_profile = config.evaluation_profile_config
    public_objective = public_accept_summary["generalization"]
    sealed_evidence = build_stair_sealed_rollout_evidence(
        config.protocol_v1,
        semantics=config.evaluation_semantics,
        evaluation_protocol=evaluation_protocol,
        summary=sealed_summary,
        checkpoint_hash=checkpoint_hash,
        evaluation_binding_hash=evaluation_binding.binding_hash,
        evaluation_subject=evaluation_subject,
        subject_manifest_hash=subject_manifest_hash,
        program_source_checksum=program_source_checksum,
        training_binding_hash=training_binding_hash,
        capability_profile_hash=capability_profile_hash,
        acceptance_protocol=config.acceptance_protocol,
        evaluation_profile=evaluation_profile,
        public_objective_summary=public_objective,
    )
    checkpoint_report = validate_checkpoint_safety_report(
        checkpoint_safety_report,
        expected_checkpoint_hash=checkpoint_hash,
    )
    sealed_gate = build_sealed_validation_gate(
        rollout_evidence=sealed_evidence,
        checkpoint_safety_status=str(checkpoint_report["status"]),
        checkpoint_hash_match_status=str(
            checkpoint_report["checkpoint_hash_match_status"]
        ),
        candidate_isolation_gate=build_candidate_isolation_gate(),
        expected_eval_protocol_hash=evaluation_protocol.protocol_hash,
        expected_evaluation_binding_hash=evaluation_binding.binding_hash,
    )
    return {
        "schema_version": STAIR_FULL_EVAL_SUMMARY_SCHEMA_VERSION,
        **identity.to_public_dict(),
        "evaluation_protocol": evaluation_protocol.to_public_dict(),
        "evaluation_binding": evaluation_binding.to_public_dict(),
        "execution_device": execution_device,
        "backend_execution_environment_verification": dict(
            backend_execution_environment_verification
        ),
        "evaluation_execution_cost": dict(evaluation_execution_cost),
        "status": "pass",
        "public_score": float(public_accept_summary["anchor"]["score_stats"]["mean"]),
        "metrics": {
            "environment_source_preflight": dict(environment_source_preflight),
            "stair_evaluation": {
                "public_dev": dict(public_dev_summary),
                "public_accept": dict(public_accept_summary),
            },
        },
        "checkpoint_safety_report": checkpoint_report,
        "sealed_validation_summary": dict(sealed_summary),
        "sealed_validation_evidence": sealed_evidence,
        "sealed_validation_gate": sealed_gate,
        "candidate_code_loaded_in_eval": False,
        "candidate_package_loaded": False,
        "candidate_package_loaded_in_eval": False,
        "candidate_path_argument_present": False,
        "errors": [],
    }


def _canonical_phase_summary(
    *,
    config: StairClimbingTaskConfig,
    phase: EvaluationPhase,
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(summary, Mapping):
        raise ValueError("Stair phase summary must be an object")
    if phase is EvaluationPhase.SEALED_VALIDATION:
        raw_generalization = summary.get("generalization")
        if not isinstance(raw_generalization, Mapping):
            raise ValueError("Stair sealed phase lacks generalization evidence")
        return canonical_stair_sealed_summary(
            config.protocol_v1,
            semantics=config.evaluation_semantics,
            evaluation_protocol=config.evaluation_protocol(),
            summary=raw_generalization,
            evaluation_profile=config.evaluation_profile_config,
        )
    if set(summary) != {"anchor", "generalization"}:
        raise ValueError("Stair public phase summary fields mismatch")
    return {
        "anchor": validate_stair_split_summary(
            config.protocol_v1,
            semantics=config.evaluation_semantics,
            evaluation_protocol=config.evaluation_protocol(),
            split=phase.value,
            summary=summary["anchor"],
        ),
        "generalization": validate_stair_generalization_summary(
            profile=config.evaluation_profile,
            semantics=config.evaluation_semantics,
            evaluation_protocol=config.evaluation_protocol(),
            split=phase.value,
            summary=summary["generalization"],
        ),
    }


def canonical_stair_phase_result_summary(
    *,
    config: StairClimbingTaskConfig,
    expected_phase: EvaluationPhase,
    result: EvaluationPhaseResult,
    expected_seed_namespace: str = STAIR_FORMAL_EVAL_NAMESPACE,
) -> dict[str, Any]:
    evidence = result.task_evidence
    if set(evidence) != {"schema_version", "phase", "execution_identity", "summary"}:
        raise ValueError("Stair phase task evidence fields mismatch")
    if (
        evidence.get("schema_version") != STAIR_PHASE_EVIDENCE_SCHEMA_VERSION
        or evidence.get("phase") != expected_phase.value
    ):
        raise ValueError("Stair phase task evidence identity mismatch")
    execution_identity = evidence.get("execution_identity")
    if not isinstance(execution_identity, Mapping):
        raise ValueError("Stair phase execution identity is invalid")
    expected_splits = [expected_phase.value]
    if (
        execution_identity.get("checkpoint_hash") != result.checkpoint_hash
        or execution_identity.get("seed_namespace") != expected_seed_namespace
        or execution_identity.get("seed_cohort_hash")
        != config.seed_resolver().namespace_cohort_hash(expected_seed_namespace)
        or execution_identity.get("splits") != expected_splits
        or execution_identity.get("execution_device") != result.execution_device
        or execution_identity.get("environment_source_preflight")
        != dict(result.environment_source_preflight)
        or execution_identity.get("backend_execution_environment_verification")
        != dict(result.backend_execution_environment_verification)
    ):
        raise ValueError("Stair phase execution identity binding mismatch")
    summary = evidence.get("summary")
    if expected_phase is EvaluationPhase.SEALED_VALIDATION:
        return canonical_stair_sealed_summary(
            config.protocol_v1,
            semantics=config.evaluation_semantics,
            evaluation_protocol=config.evaluation_protocol(),
            summary=summary,
            evaluation_profile=config.evaluation_profile_config,
        )
    return _canonical_phase_summary(
        config=config,
        phase=expected_phase,
        summary=summary,
    )


def _require_phase_result_matches_plan(
    result: EvaluationPhaseResult,
    eval_plan: EvalJobSpec,
) -> None:
    checkpoint = eval_plan.execution_checkpoint()
    expected = {
        "evaluation_plan_hash": canonical_json_hash(eval_plan.to_public_dict()),
        "task_id": eval_plan.task_id,
        "subject": eval_plan.subject,
        "subject_manifest_hash": eval_plan.subject_manifest_hash,
        "program_source_checksum": eval_plan.program_source_checksum,
        "training_binding_hash": eval_plan.training_binding_hash,
        "capability_profile_hash": eval_plan.capability_profile_hash,
        "checkpoint_hash": checkpoint.checksum,
        "eval_protocol_hash": eval_plan.evaluation_protocol().protocol_hash,
        "evaluation_binding_hash": eval_plan.evaluation_binding.binding_hash,
        "execution_device": eval_plan.execution_device,
    }
    if any(getattr(result, name) != value for name, value in expected.items()):
        raise ValueError("Stair phase result differs from its evaluation plan")


def _require_eval_plan_matches_config(
    config: StairClimbingTaskConfig,
    eval_plan: EvalJobSpec,
) -> None:
    protocol = config.evaluation_protocol()
    binding = eval_plan.evaluation_binding
    expected_binding = {
        "eval_protocol_hash": protocol.protocol_hash,
        "acceptance_protocol_hash": config.acceptance_protocol.protocol_hash,
        "task_protocol_hash": config.task_protocol_hash,
        "seed_cohort_hash": config.seed_resolver().namespace_cohort_hash(
            STAIR_FORMAL_EVAL_NAMESPACE
        ),
    }
    if eval_plan.task_id != STAIR_TASK_ID or any(
        getattr(binding, name) != expected
        for name, expected in expected_binding.items()
    ):
        raise ValueError("Stair evaluation plan differs from the active task contract")


def _require_stair_execution_domain(
    config: StairClimbingTaskConfig,
    execution: StairEvaluationExecutionResult,
    *,
    phases: set[EvaluationPhase],
    expected_seed_namespace: str = STAIR_FORMAL_EVAL_NAMESPACE,
) -> None:
    if not isinstance(config, StairClimbingTaskConfig):
        raise TypeError("Stair phased evaluation requires StairClimbingTaskConfig")
    if not isinstance(execution, StairEvaluationExecutionResult):
        raise TypeError("Stair phased evaluation requires an execution result")
    if set(execution.summaries) != {phase.value for phase in phases}:
        raise ValueError("Stair execution split set mismatch")
    if execution.seed_namespace != expected_seed_namespace:
        raise ValueError("Stair phased evaluation seed namespace mismatch")
    expected_cohort = config.seed_resolver().namespace_cohort_hash(
        expected_seed_namespace
    )
    if execution.seed_cohort_hash != expected_cohort:
        raise ValueError("Stair phased evaluation seed cohort mismatch")


def _merge_phase_costs(
    by_phase: Mapping[EvaluationPhase, EvaluationPhaseResult],
) -> dict[str, Any]:
    merged_splits: dict[str, Any] = {}
    common: dict[str, Any] | None = None
    for phase in EvaluationPhase:
        cost = by_phase[phase].execution_cost
        if set(cost) != {"schema_version", "clock_source", "measurement_scope", "splits"}:
            raise ValueError("Stair phase execution cost fields mismatch")
        phase_common = {
            "schema_version": cost["schema_version"],
            "clock_source": cost["clock_source"],
            "measurement_scope": cost["measurement_scope"],
        }
        if common is None:
            common = phase_common
        elif common != phase_common:
            raise ValueError("Stair phase execution cost contracts differ")
        split_costs = cost.get("splits")
        if not isinstance(split_costs, Mapping) or set(split_costs) != {phase.value}:
            raise ValueError("Stair phase execution cost split mismatch")
        merged_splits[phase.value] = dict(split_costs[phase.value])
    assert common is not None
    return {**common, "splits": merged_splits}


__all__ = [
    "STAIR_FULL_EVAL_SUMMARY_SCHEMA_VERSION",
    "STAIR_PHASE_EVIDENCE_SCHEMA_VERSION",
    "assemble_stair_eval_phase_results",
    "build_stair_eval_phase_summary",
    "build_stair_full_eval_summary",
    "build_stair_phase_task_evidence",
    "canonical_stair_phase_result_summary",
]
