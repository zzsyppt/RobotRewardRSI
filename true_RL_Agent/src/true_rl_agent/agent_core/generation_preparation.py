"""Bind one formal session from a resolved cell and explicit frozen cost data."""

from __future__ import annotations

from true_rl_agent.agent_core.generation_contract import GENERATION_LEDGER_RECORD_V3_SCHEMA_VERSION

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from true_rl_agent.agent_core.generation_contract import (
    GENERATION_LANE_IDS, GENERATION_SESSION_CONTRACT_V3_SCHEMA_VERSION,
    GenerationLaneAssignment, GenerationSessionContract,
    generation_resource_plan_hash, publish_generation_session_contract,
    load_replay_generation_session_contract,
)
from true_rl_agent.agent_core.generation_cost import FrozenGenerationCostContract
from true_rl_agent.agent_core.completed_reference_training import recover_completed_reference_training
from true_rl_agent.agent_core.generation_finalization import GenerationFinalPlan, publish_final_plan
from true_rl_agent.agent_core.generation_foundation import GenerationSessionFoundation
from true_rl_agent.agent_core.generation_lifecycle import (
    GENERATION_CONTINUATION_POLICY_FILENAME, GENERATION_CONTINUATION_POLICY_V2_SCHEMA_VERSION,
    GenerationContinuationPolicy, publish_generation_continuation_policy,
)
from true_rl_agent.agent_core.generation_operations import OperationAdmission
from true_rl_agent.agent_core.generation_runtime import generation_provider_config, generation_genesis_execution
from true_rl_agent.composition import ResolvedTaskComposition
from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.fs_contract import create_once_bytes
from true_rl_agent.json_contract import canonical_json_bytes, load_strict_json_object
from true_rl_agent.runner.gpu_lease import GpuDeviceIdentity, GpuLeaseManager
from true_rl_agent.runner.specs import GpuProcessBinding
from true_rl_agent.runner.subprocess_runner import SubprocessRunner


@dataclass(frozen=True)
class GenerationLaunchConfig:
    project_root: Path
    output_dir: Path
    lease_root: Path
    provider_env_file: Path | None
    session_id: str
    cell_variant_id: str
    machine_id: str
    gpu0_uuid: str
    gpu1_uuid: str
    max_allocated_active_gpu_hours: float
    max_simulated_control_steps: int
    max_output_bytes: int
    source_reference_genesis_dir: Path | None = None
    source_reference_session_id: str | None = None
    source_reference_training_genesis_dir: Path | None = None
    source_reference_training_session_id: str | None = None
    completed_reference_training_source: Path | None = None
    resume_existing: bool = False
    pause_after_generation: int | None = None
    terminal_audit_only: bool = False
    nvidia_smi_executable: str = "nvidia-smi"
    session_mode: str = "formal_generation_session"
    max_generations: int = 15

    def __post_init__(self) -> None:
        if type(self.max_generations) is not int or self.max_generations <= 0:
            raise ValueError("generation max_generations must be a positive integer")
        for name in ("project_root", "output_dir", "lease_root", "provider_env_file"):
            value = getattr(self, name)
            if name == "provider_env_file" and value is None:
                continue
            if not isinstance(value, Path) or not value.is_absolute():
                raise ValueError(f"generation {name} must be an absolute Path")
        if (self.source_reference_genesis_dir is None) != (self.source_reference_session_id is None):
            raise ValueError("reference import requires both source directory and session identity")
        if (self.source_reference_training_genesis_dir is None) != (self.source_reference_training_session_id is None):
            raise ValueError("reference training reuse requires source directory and session identity")
        if self.source_reference_training_genesis_dir is not None:
            source = self.source_reference_training_genesis_dir
            if not isinstance(source, Path) or not source.is_absolute() or source.resolve() == self.output_dir.resolve():
                raise ValueError("reference training source must be an absolute, different directory")
            if self.imports_reference_genesis:
                raise ValueError("reference training reuse and full Genesis import are mutually exclusive")
        if self.completed_reference_training_source is not None:
            if not self.completed_reference_training_source.is_absolute():
                raise ValueError("completed training descriptor must be an absolute Path")
            if self.imports_reference_genesis or self.source_reference_training_genesis_dir is not None:
                raise ValueError("reference training requires exactly one source")
        if self.pause_after_generation is not None and (
            type(self.pause_after_generation) is not int
            or not 1 <= self.pause_after_generation < self.max_generations
        ):
            raise ValueError("pause must follow a nonterminal complete generation")
        if self.terminal_audit_only and (not self.resume_existing or self.pause_after_generation is not None):
            raise ValueError("terminal audit requires resume without an execution pause")

    @property
    def imports_reference_genesis(self) -> bool:
        return self.source_reference_genesis_dir is not None

    @property
    def reuses_reference_training(self) -> bool:
        return self.source_reference_training_genesis_dir is not None or self.completed_reference_training_source is not None

    @property
    def reference_mode(self) -> str:
        if self.imports_reference_genesis:
            return "import_qualified_genesis"
        return "imports_training_only" if self.reuses_reference_training else "train_and_qualify"


def _json_value(value):
    if isinstance(value, Mapping):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def prepare_generation_foundation(
    config: GenerationLaunchConfig,
    *, composition: ResolvedTaskComposition, source_commit: str,
    inventory: tuple[str, str], cost: FrozenGenerationCostContract,
    final_plan: GenerationFinalPlan, provider_policy: Mapping[str, Any],
) -> GenerationSessionFoundation:
    """Freeze common execution ownership without selecting task/robot semantics.

    Composition and the task's cohort preparation happen before this binding.
    This host owns GPU lanes, continuation, admission and session commitments.
    """
    manifest = composition.resolved_cell_manifest
    if manifest is None or manifest.cell_variant_id != config.cell_variant_id:
        raise ValueError("generation requires the selected resolved cell")
    if composition.genesis_anchor_kind != "reference_program":
        raise ValueError("generation sessions require reference-program Genesis")
    if not isinstance(composition.runner, SubprocessRunner):
        raise TypeError("generation sessions require the production subprocess runner")
    phased = composition.task_plugin.phased_evaluation_adapter()
    if phased is None or phased.parent_public_dev_source != "genesis_receipt":
        raise ValueError("new sessions require phased reference evaluation")
    if cost.scientific_cost_basis != phased.generation_workload(cell_manifest=manifest):
        raise ValueError("session workload differs from the resolved task projection")
    if cost.max_generations != config.max_generations or cost.reference_mode != config.reference_mode:
        raise ValueError("cost topology differs from the requested session")
    if (final_plan.session_id != config.session_id
            or final_plan.task_id != composition.task_id
            or final_plan.task_protocol_hash != phased.task_protocol_hash
            or final_plan.required_closed_generations != config.max_generations
            or final_plan.total_eval_seed_units != cost.final_confirmation_seed_units):
        raise ValueError("final confirmation differs from the frozen session work")
    if (cost.max_allocated_active_gpu_hours > config.max_allocated_active_gpu_hours
            or cost.max_simulated_control_steps > config.max_simulated_control_steps
            or cost.max_output_bytes > config.max_output_bytes):
        raise ValueError("operator limits cannot reserve the requested session")
    budget = cost.provider_budget
    if provider_policy.get("backend", "openai_compatible_chat") != "external_file":
        from true_rl_agent.agent_core.provider_limits import validate_provider_model_context
        model_context = validate_provider_model_context(provider_policy.get("model_context"))
        if budget.input_units_per_attempt_reserve + budget.output_units_per_attempt_cap > model_context["context_window_tokens"]:
            raise ValueError("reserved input and requested output exceed the provider context window")
    generation_provider_config(output_dir=config.output_dir / "provider_candidates",
                               env_file=config.provider_env_file, policy=provider_policy)
    if (provider_policy["max_attempts_per_lane"] != budget.attempts_per_lane_per_generation
            or provider_policy["max_output_tokens"] != budget.output_units_per_attempt_cap
            or provider_policy["input_units_per_attempt_reserve"] != budget.input_units_per_attempt_reserve):
        raise ValueError("provider policy differs from the frozen cost budget")
    if inventory != (config.gpu0_uuid, config.gpu1_uuid) or inventory[0] == inventory[1]:
        raise ValueError("GPU inventory differs from the two requested lanes")

    identities = tuple(GpuDeviceIdentity(f"cuda:{index}", config.machine_id, uuid)
                       for index, uuid in enumerate(inventory))
    manager = GpuLeaseManager(lease_root=config.lease_root,
                             devices={item.execution_device: item for item in identities})
    bindings = tuple(GpuProcessBinding(host_execution_device=item.execution_device,
                                      process_execution_device="cuda:0", resource_binding_hash=item.identity_hash)
                     for item in identities)
    external_waiting = provider_policy.get("backend") == "external_file"
    if config.resume_existing:
        stored_continuation = GenerationContinuationPolicy.from_mapping(
            load_strict_json_object(config.output_dir / GENERATION_CONTINUATION_POLICY_FILENAME,
                                    require_canonical=True, trailing_newline=True)
        )
        # Old sessions retain their frozen reasons; do not opt them into recovery.
        external_waiting = external_waiting and "external_author_waiting" in stored_continuation.allowed_continuation_reasons
    continuation = GenerationContinuationPolicy(
        schema_version=GENERATION_CONTINUATION_POLICY_V2_SCHEMA_VERSION, max_segments=7,
        allowed_continuation_reasons=("operator_pause", "typed_infrastructure_failure", "exception",
                                      "budget_exhausted", "contract_incomplete")
                                     + (("external_author_waiting",) if external_waiting else ()),
        operator_pause_generation_boundaries=tuple(range(1, config.max_generations)),
    )
    if config.resume_existing:
        if not set(stored_continuation.operator_pause_generation_boundaries).issubset(
            continuation.operator_pause_generation_boundaries
        ):
            raise ValueError("stored pause boundaries exceed the generation contract")
        # Reuse the committed subset; publication still verifies every other policy field.
        continuation = replace(continuation,
            operator_pause_generation_boundaries=stored_continuation.operator_pause_generation_boundaries)
    if (config.pause_after_generation is not None
            and config.pause_after_generation not in continuation.operator_pause_generation_boundaries):
        raise ValueError("requested pause is not declared by the frozen continuation policy")
    resource_assignments = tuple(
        GenerationLaneAssignment(
            lane_id=lane, execution_device=binding.process_execution_device,
            resource_binding_hash=binding.resource_binding_hash,
            lane_intent_hash=canonical_json_hash({"purpose": "resource-plan-hash-does-not-include-lane-intent"}),
            gpu_process_binding=binding,
        ) for lane, binding in zip(GENERATION_LANE_IDS, bindings, strict=True)
    )
    authoring_policy = {
        "schema_version": "true_rl_agent.generation_authoring_policy.v2",
        "failed_lane_does_not_cancel_sibling_authoring": False,
        "authoring_order": list(GENERATION_LANE_IDS),
        "sibling_semantic_exclusion": True, "both_candidates_ready_barrier": True,
        "candidate_admission_policy": _json_value(composition.candidate_admission_policy.admission_policy_descriptor().to_public_dict()),
        "authoring_capability_readiness": _json_value(composition.authoring_capability_readiness.to_prompt_mapping()),
    }
    retry_policy = {
        "schema_version": "true_rl_agent.generation_retry_policy.v1",
        "reference_train_retry_entitlement": cost.retry_train_jobs,
        "reference_eval_retry_entitlement": 0,
        "parent_and_public_phase_retry_entitlement": 0,
        "session_wide_sealed_validation_retry_entitlement": 1,
        "eligible_sealed_validation_roles": ["generation_winner", "final_confirmation_genesis", "final_confirmation_final"],
    }
    contract = GenerationSessionContract(
        schema_version=GENERATION_SESSION_CONTRACT_V3_SCHEMA_VERSION,
        ledger_record_schema=GENERATION_LEDGER_RECORD_V3_SCHEMA_VERSION,
        session_mode=config.session_mode, session_id=config.session_id,
        task_id=composition.task_id, backend_id=composition.backend_adapter.backend_id,
        max_generations=config.max_generations, max_trials=2 * config.max_generations,
        max_train_jobs=2 * config.max_generations,
        bound_contract_hashes={
            "authoring_policy": canonical_json_hash(authoring_policy),
            "continuation_policy": continuation.policy_hash,
            "evaluation_cost": canonical_json_hash(phased.cost_contract()),
            "evaluation_protocol": final_plan.evaluation_protocol_hash,
            "exploration_projector": canonical_json_hash(composition.task_memory_adapter.exploration_projector_contract()),
            "final_confirmation_plan": final_plan.plan_hash,
            "provider_policy": canonical_json_hash(provider_policy),
            "resource_plan": generation_resource_plan_hash(resource_assignments),
            "retry_policy": canonical_json_hash(retry_policy),
            "selector": canonical_json_hash(phased.selection_contract()),
            "session_cost": cost.contract_hash,
        },
    )
    cost_path = config.output_dir / "generation_session_cost.json"
    if cost_path.exists():
        stored_cost = FrozenGenerationCostContract.from_mapping(load_strict_json_object(
            cost_path, require_canonical=True, trailing_newline=True,
        ))
        if stored_cost != cost:
            raise ValueError("existing generation cost contract differs")
    elif config.resume_existing:
        raise FileNotFoundError("resumed session is missing its cost contract")
    else:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        create_once_bytes(cost_path, canonical_json_bytes(cost.to_dict(), trailing_newline=True))
    publish_final_plan(config.output_dir, final_plan, require_existing=True)
    publish_generation_continuation_policy(
        config.output_dir, continuation, require_existing=config.resume_existing,
    )
    if config.terminal_audit_only:
        contract, contract_ref = load_replay_generation_session_contract(config.output_dir, contract)
    else:
        contract_ref = publish_generation_session_contract(
            config.output_dir, contract, require_existing=config.resume_existing,
        )
    runner = replace(composition.runner, gpu_lease_manager=manager,
                     attempt_admission_hook=OperationAdmission(
                         output_dir=config.output_dir, session_id=config.session_id,
                         cost_contract=cost, session_contract_hash=contract.contract_hash,
                         termination_grace_s=composition.runner.termination_grace_s,
                     ))
    completed_training = None
    if config.completed_reference_training_source is not None:
        backend = composition.backend_adapter
        evaluation_binding = backend.resolve_session_evaluation_binding(
            session_id=config.session_id, session_dir=config.output_dir,
            task_spec=composition.task_plugin.task_spec, allow_create=not config.resume_existing)
        jobs = backend.materialize_reference_jobs(session_id=config.session_id,
            task_spec=composition.task_plugin.task_spec, reference=composition.reference_program_artifact,
            artifact_root=config.output_dir / "reference_program_artifacts",
            policy_initialization=composition.policy_initialization,
            evaluation_binding=evaluation_binding, gpu_process_binding=bindings[0])
        completed_training = recover_completed_reference_training(config.completed_reference_training_source,
            requested=jobs.train_spec, runner=runner, output_dir=config.output_dir,
            require_existing=config.resume_existing)
    foundation = GenerationSessionFoundation(
        config=config, source_commit=source_commit, composition=composition, runner=runner,
        phased_adapter=phased, continuation_policy=continuation, cost_contract=cost,
        session_contract=contract, session_contract_ref=contract_ref,
        gpu_identities=identities, gpu_bindings=bindings, gpu_lease_manager=manager,
        provider_policy=dict(provider_policy), final_confirmation_policy=final_plan.genesis_commitment(),
        final_confirmation_plan=final_plan,
        completed_reference_training=completed_training,
    )
    if completed_training is not None:
        # This is an execution plan, not a completed scientific Genesis. No
        # environment probe, attempt reservation, training or evaluation runs here.
        execution = generation_genesis_execution(foundation)
        preparation = {"schema_version": "true_rl_agent.reference_genesis_preparation.v1",
            "source_commit": source_commit, "session_id": config.session_id,
            "session_contract_hash": contract.contract_hash, "cost_contract_hash": cost.contract_hash,
            "reference_mode": config.reference_mode,
            "historical_training": completed_training.source_ref(),
            "current_reference_train_attempt": None,
            "qualification_journal_root": str(config.output_dir / "job_attempts"),
            "qualification_scopes": {phase.value: scope.binding_payload()
                for phase, scope in execution["reference_eval_phase_attempt_scopes"].items()}}
        path = config.output_dir / "reference_genesis_preparation.json"
        if path.exists():
            if load_strict_json_object(path) != preparation:
                raise ValueError("reference Genesis preparation differs from the frozen session")
        elif config.resume_existing:
            raise FileNotFoundError("resumed session lacks reference Genesis preparation")
        else:
            create_once_bytes(path, canonical_json_bytes(preparation, trailing_newline=True))
    return foundation
