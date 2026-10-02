"""Task-owned Stair phase evidence and full-result assembly."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
import time
from typing import Any

from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.evaluation.phase_assembly import (
    bind_phase_receipts,
    merge_phase_costs,
    require_phase_result_matches_plan,
)
from true_rl_agent.evaluation.phases import (
    EVALUATION_PHASE_SUMMARY_SCHEMA_VERSION,
    EvaluationAssembly,
    EvaluationPhase,
    EvaluationPhaseResult,
)
from true_rl_agent.evaluation.sealed_validation import (
    build_sealed_rollout_evidence,
    build_sealed_validation_gate,
)
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
    evaluate_unified_stair_sealed_score_gate,
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
from true_rl_agent.tasks.stair_climbing.unified_scoring import (
    build_unified_reference_score_projection,
    build_reference_objective_family_mask,
    validate_objective_family_mask,
    validate_unified_stair_summary,
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
    """Bind rollout evidence; expensive CPU behavior waits for GPU release."""

    phase = EvaluationPhase.parse(phase)
    if set(execution.summaries) != {phase.value}:
        raise ValueError("Stair phase execution contains the wrong split set")
    task_evidence = build_stair_phase_task_evidence(
        config=config,
        phase=phase,
        execution=execution,
        expected_seed_namespace=expected_seed_namespace,
    )
    if execution.numeric_evidence is not None:
        from true_rl_agent.runner.numeric_evidence import load_numeric_evidence_manifest
        archive = load_numeric_evidence_manifest(execution.numeric_evidence)
        expected = {
            **identity.to_public_dict(),
            "checkpoint_hash": execution.checkpoint_hash,
            "evaluation_plan_hash": evaluation_plan_hash,
            "evaluation_binding": evaluation_binding.to_public_dict(),
            "seed_namespace": expected_seed_namespace,
            "splits": [phase.value],
            "environment_source_preflight": dict(execution.source_manifest_preflight),
            "backend_execution_environment_verification": dict(execution.backend_execution_environment_verification),
        }
        if (phase is not EvaluationPhase.PUBLIC_DEV
                or any(archive["identity"].get(key) != value for key, value in expected.items())
                or archive["metadata"].get("summary") != task_evidence["summary"]):
            raise ValueError("Stair numeric evidence execution binding mismatch")
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
        **({"checkpoint_loading": dict(execution.checkpoint_loading)}
           if execution.checkpoint_loading is not None else {}),
        **({"constructed_learner": dict(execution.constructed_learner)}
           if execution.constructed_learner is not None else {}),
        "status": (
            "postprocessing"
            if execution.numeric_evidence is not None and config.is_unified_protocol
            else "pass"
        ),
        "task_evidence": task_evidence,
        "task_evidence_hash": canonical_json_hash(task_evidence),
        "checkpoint_safety_report": dict(checkpoint_safety_report),
        "candidate_code_loaded_in_eval": False,
        "candidate_package_loaded": False,
        "candidate_package_loaded_in_eval": False,
        "candidate_path_argument_present": False,
        "errors": [],
    }


def finalize_stair_eval_phase_summary(
    eval_spec: EvalPhaseJobSpec,
    summary: Mapping[str, Any],
    *,
    config: StairClimbingTaskConfig,
    output_dir: Path,
) -> dict[str, Any]:
    """Complete the single production behavior derivation on the CPU host.

    The runner invokes this only after its owned process exits and its GPU lease
    is released. Historical completed summaries are not re-analyzed.
    """
    from true_rl_agent.runner.numeric_evidence import load_numeric_evidence_manifest
    from true_rl_agent.tasks.stair_climbing.behavior import (
        project_stair_behavior,
    )
    from true_rl_agent.tasks.stair_climbing.process_report import build_stair_process_behavior

    if summary.get("status") != "postprocessing":
        raise ValueError("Stair finalization requires an unfinished rollout summary")
    if not config.is_unified_protocol or eval_spec.phase is not EvaluationPhase.PUBLIC_DEV:
        raise ValueError("Stair behavior finalization requires unified public-dev")
    finalized = deepcopy(dict(summary))
    evidence = finalized["task_evidence"]
    reference = evidence["numeric_evidence"]
    expected = {
        **eval_spec.job_identity().to_public_dict(),
        "checkpoint_hash": eval_spec.execution_checkpoint().checksum,
        "evaluation_plan_hash": eval_spec.evaluation_plan_hash,
        "evaluation_binding": eval_spec.evaluation_binding.to_public_dict(),
        "seed_namespace": evidence["execution_identity"]["seed_namespace"],
        "splits": [eval_spec.phase.value],
        "environment_source_preflight": finalized["environment_source_preflight"],
        "backend_execution_environment_verification": finalized[
            "backend_execution_environment_verification"
        ],
    }
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    archive = load_numeric_evidence_manifest(reference)
    if archive["metadata"].get("summary") != evidence["summary"]:
        raise ValueError("Stair numeric evidence summary binding mismatch")
    full = build_stair_process_behavior(reference, expected_identity=expected)
    evidence["behavior"] = project_stair_behavior(full)
    from true_rl_agent.json_contract import canonical_json_bytes
    from true_rl_agent.runner.checkpoint import file_sha256
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "behavior_full.json"
    temporary = output_dir / "behavior_full.json.finalizing"
    if path.is_symlink() or temporary.is_symlink():
        raise ValueError("Stair full behavior output must not be a symlink")
    temporary.write_bytes(canonical_json_bytes(full) + b"\n")
    temporary.replace(path)
    evidence["behavior_full"] = {"schema_version": "true_rl_agent.process_full_ref.v1",
        "file": path.name, "sha256": file_sha256(path).removeprefix("sha256:"), "size_bytes": path.stat().st_size}
    split_cost = finalized["evaluation_execution_cost"]["splits"][eval_spec.phase.value]
    split_cost["postprocessing_wall_time_s"] = time.perf_counter() - wall_started
    split_cost["postprocessing_cpu_time_s"] = time.process_time() - cpu_started
    finalized["task_evidence_hash"] = canonical_json_hash(evidence)
    finalized["status"] = "pass"
    return finalized


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
        **({"numeric_evidence": dict(execution.numeric_evidence)} if execution.numeric_evidence is not None else {}),
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
    objective_family_mask: Mapping[str, Any] | None = None,
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
    summary = _build_complete_stair_eval_summary(
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
        objective_family_mask=objective_family_mask,
    )
    if execution.checkpoint_loading is not None:
        summary["checkpoint_loading"] = dict(execution.checkpoint_loading)
    if execution.constructed_learner is not None:
        summary["constructed_learner"] = dict(execution.constructed_learner)
    return summary


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
    objective_family_mask: Mapping[str, Any] | None = None,
) -> EvaluationAssembly:
    """Assemble one winner's three validated phases into the normal raw summary."""

    by_phase, summary = _validated_complete_phase_summary(
        config=config,
        eval_plan=eval_plan,
        phase_results=phase_results,
        objective_family_mask=objective_family_mask,
    )

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
    selected_matches = [
        screen for screen in selection_screens if screen.trial_id == trial_id
    ]
    if len(selected_matches) != 1:
        raise ValueError("Stair formal assembly selected screen is not unique")
    selected_screen = selected_matches[0]
    phase_projection = build_stair_public_dev_screen_evidence(
        config=config,
        trial_id=trial_id,
        phase_result=by_phase[EvaluationPhase.PUBLIC_DEV],
        screen_role="guard_reference",
        objective_family_mask=objective_family_mask,
        statistics_version=selected_screen.generalization_score_stats.get("statistics_version"),
    )
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

    bindings = bind_phase_receipts(
        eval_plan=eval_plan,
        by_phase=by_phase,
        phase_receipt_refs=phase_receipt_refs,
    )
    return EvaluationAssembly(
        evaluation_plan_hash=canonical_json_hash(eval_plan.to_public_dict()),
        selected_trial_id=trial_id,
        selection=selection_result.to_public_dict(),
        phase_receipts=bindings,
        summary=summary,
    )


def assemble_stair_reference_eval_phase_results(
    *,
    config: StairClimbingTaskConfig,
    eval_plan: EvalJobSpec,
    phase_results: Sequence[EvaluationPhaseResult],
    phase_receipt_refs: Mapping[EvaluationPhase | str, Mapping[str, Any]],
) -> EvaluationAssembly:
    """Assemble one reference program's three phases without candidate selection."""

    if not isinstance(eval_plan, EvalJobSpec) or eval_plan.subject.kind != "reference":
        raise TypeError("Stair reference phase assembly requires a reference eval plan")
    by_phase, summary = _validated_complete_phase_summary(
        config=config,
        eval_plan=eval_plan,
        phase_results=phase_results,
        objective_family_mask=None,
    )
    bindings = bind_phase_receipts(
        eval_plan=eval_plan,
        by_phase=by_phase,
        phase_receipt_refs=phase_receipt_refs,
    )
    reference_id = eval_plan.subject.subject_id
    selection = {
        "schema_version": (
            "true_rl_agent.task.stair_climbing.reference_eval_assembly.v1"
        ),
        "selected_trial_id": reference_id,
        "selection_kind": "reference_program_qualification",
        "acceptance_authority": False,
    }
    return EvaluationAssembly(
        evaluation_plan_hash=canonical_json_hash(eval_plan.to_public_dict()),
        selected_trial_id=reference_id,
        selection=selection,
        phase_receipts=bindings,
        summary=summary,
    )


def _validated_complete_phase_summary(
    *,
    config: StairClimbingTaskConfig,
    eval_plan: EvalJobSpec,
    phase_results: Sequence[EvaluationPhaseResult],
    objective_family_mask: Mapping[str, Any] | None,
) -> tuple[dict[EvaluationPhase, EvaluationPhaseResult], dict[str, Any]]:
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
        require_phase_result_matches_plan(result, eval_plan)
        by_phase[result.phase] = result
    if set(by_phase) != set(EvaluationPhase):
        raise ValueError(
            "Stair phase assembly requires public-dev, public-accept, and sealed"
        )
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
        evaluation_execution_cost=merge_phase_costs(by_phase),
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
        objective_family_mask=objective_family_mask,
    )
    if "behavior" in first.task_evidence:
        from true_rl_agent.tasks.stair_climbing.behavior import project_stair_behavior
        summary["metrics"]["behavior"] = project_stair_behavior(first.task_evidence["behavior"])
    return by_phase, summary


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
    objective_family_mask: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if config.is_unified_protocol:
        return _build_complete_unified_stair_eval_summary(
            config=config,
            identity=identity,
            evaluation_binding=evaluation_binding,
            execution_device=execution_device,
            environment_source_preflight=environment_source_preflight,
            backend_execution_environment_verification=(
                backend_execution_environment_verification
            ),
            evaluation_execution_cost=evaluation_execution_cost,
            checkpoint_safety_report=checkpoint_safety_report,
            evaluation_subject=evaluation_subject,
            subject_manifest_hash=subject_manifest_hash,
            program_source_checksum=program_source_checksum,
            training_binding_hash=training_binding_hash,
            capability_profile_hash=capability_profile_hash,
            checkpoint_hash=checkpoint_hash,
            public_dev_summary=public_dev_summary,
            public_accept_summary=public_accept_summary,
            sealed_summary=sealed_summary,
            objective_family_mask=objective_family_mask,
        )
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


def _build_complete_unified_stair_eval_summary(
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
    objective_family_mask: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Assemble the single-cell ID/OOD result without reviving terrain cells."""

    suite = config.ordered_evaluation_suite
    is_reference = getattr(evaluation_subject, "kind", None) == "reference"
    if is_reference:
        derived_mask = build_reference_objective_family_mask(
            suite=suite,
            public_accept_summary=public_accept_summary,
        )
        if objective_family_mask is not None and dict(objective_family_mask) != derived_mask:
            raise ValueError("reference objective family mask must be result-derived")
        objective_family_mask = derived_mask
    elif objective_family_mask is not None:
        objective_family_mask = validate_objective_family_mask(
            suite=suite,
            mask=objective_family_mask,
        )
    else:
        raise ValueError("unified Stair candidate evaluation lacks Genesis mask")
    public_projection = build_unified_reference_score_projection(
        public_accept_summary,
        objective_family_mask=objective_family_mask,
    )
    sealed_projection = build_unified_reference_score_projection(
        sealed_summary,
        objective_family_mask=objective_family_mask,
    )
    public_primary = public_projection["j_primary"]["score_stats"]
    sealed_primary = sealed_projection["j_primary"]["score_stats"]
    sealed_task_summary = {
        **dict(sealed_summary),
        "score_stats": dict(sealed_primary),
        "objective_family_mask_hash": public_projection[
            "objective_family_mask_hash"
        ],
    }
    public_mean = float(public_primary["mean"])
    sealed_mean = float(sealed_primary["mean"])
    passed, reason = evaluate_unified_stair_sealed_score_gate(
        sealed_mean=sealed_mean,
        public_mean=public_mean,
        acceptance_protocol=config.acceptance_protocol,
        evaluation_subject=evaluation_subject,
    )
    evaluation_protocol = config.evaluation_protocol()
    sealed_evidence = build_sealed_rollout_evidence(
        score_stats=sealed_primary,
        expected_seed_count=evaluation_protocol.sealed_validation_seed_count,
        eval_protocol_hash=evaluation_protocol.protocol_hash,
        evaluation_binding_hash=evaluation_binding.binding_hash,
        evaluation_subject=evaluation_subject,
        checkpoint_hash=checkpoint_hash,
        task_validation_status="pass" if passed else "fail",
        reason=reason,
        subject_manifest_hash=subject_manifest_hash,
        program_source_checksum=program_source_checksum,
        training_binding_hash=training_binding_hash,
        capability_profile_hash=capability_profile_hash,
        task_evidence_hash=canonical_json_hash(sealed_task_summary),
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
        "public_score": public_mean,
        "metrics": {
            "environment_source_preflight": dict(environment_source_preflight),
            "stair_evaluation": {
                "public_dev": dict(public_dev_summary),
                "public_accept": dict(public_accept_summary),
            },
            "unified_score": public_projection,
            **(
                {"objective_family_mask": dict(objective_family_mask)}
                if objective_family_mask is not None
                else {}
            ),
        },
        "checkpoint_safety_report": checkpoint_report,
        "sealed_validation_summary": sealed_task_summary,
        "sealed_validation_score": sealed_projection,
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
    if config.is_unified_protocol:
        return validate_unified_stair_summary(
            suite=config.ordered_evaluation_suite,
            semantics=config.evaluation_semantics,
            split=phase.value,
            summary=summary,
        )
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
    required_fields = {"schema_version", "phase", "execution_identity", "summary"}
    if not required_fields.issubset(evidence) or set(evidence) - required_fields - {"numeric_evidence", "behavior", "behavior_full"}:
        raise ValueError("Stair phase task evidence fields mismatch")
    if (
        evidence.get("schema_version") != STAIR_PHASE_EVIDENCE_SCHEMA_VERSION
        or evidence.get("phase") != expected_phase.value
    ):
        raise ValueError("Stair phase task evidence identity mismatch")
    execution_identity = evidence.get("execution_identity")
    if "numeric_evidence" in evidence:
        from true_rl_agent.runner.numeric_evidence import validate_numeric_evidence_reference
        if expected_phase is not EvaluationPhase.PUBLIC_DEV:
            raise ValueError("Stair numeric evidence is public_dev-only")
        # The producer bound the attachment before sealing this phase. Summary
        # replay must not depend on that machine's attachment path remaining online.
        validate_numeric_evidence_reference(evidence["numeric_evidence"])
    if not isinstance(execution_identity, Mapping):
        raise ValueError("Stair phase execution identity is invalid")
    if "behavior" in evidence:
        from true_rl_agent.tasks.stair_climbing.behavior import project_stair_behavior
        if expected_phase is not EvaluationPhase.PUBLIC_DEV or "numeric_evidence" not in evidence:
            raise ValueError("Stair behavior requires public_dev numeric evidence")
        behavior = project_stair_behavior(evidence["behavior"])
        if (behavior["domain"]["checkpoint_hash"] != result.checkpoint_hash
                or behavior["domain"]["archive_hash"] != evidence["numeric_evidence"]["manifest_sha256"]
                or behavior["domain"]["task_protocol_hash"] != config.evaluation_semantics.to_dict()["task_protocol_hash"]):
            raise ValueError("Stair behavior phase binding mismatch")
    if "behavior_full" in evidence:
        reference = evidence["behavior_full"]
        if ("behavior" not in evidence or not isinstance(reference, Mapping)
                or set(reference) != {"schema_version", "file", "sha256", "size_bytes"}
                or reference["schema_version"] != "true_rl_agent.process_full_ref.v1"
                or reference["file"] != "behavior_full.json"
                or not isinstance(reference["sha256"], str) or len(reference["sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in reference["sha256"])
                or type(reference["size_bytes"]) is not int or reference["size_bytes"] <= 0
                or behavior["schema_version"] != "true_rl_agent.stair.behavior.public.v4"):
            raise ValueError("Stair full behavior reference mismatch")
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
    if (
        expected_phase is EvaluationPhase.SEALED_VALIDATION
        and not config.is_unified_protocol
    ):
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


__all__ = [
    "STAIR_FULL_EVAL_SUMMARY_SCHEMA_VERSION",
    "STAIR_PHASE_EVIDENCE_SCHEMA_VERSION",
    "assemble_stair_eval_phase_results",
    "assemble_stair_reference_eval_phase_results",
    "build_stair_eval_phase_summary",
    "build_stair_full_eval_summary",
    "build_stair_phase_task_evidence",
    "canonical_stair_phase_result_summary",
    "finalize_stair_eval_phase_summary",
]
