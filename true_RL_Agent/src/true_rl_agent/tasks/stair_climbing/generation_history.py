"""Read-only reconstruction of frozen Stair generation evidence.

New sessions are owned by agent_core.generation_application. This module retains
only projections needed to audit old report formats; it cannot run a generation.
"""


from __future__ import annotations
import hashlib
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from true_rl_agent.agent_core.generation_foundation import GenerationSessionFoundation
from true_rl_agent.agent_core.generation_cost_journal import (
    cost_receipt_by_hash, generation_cost_receipt_projection, assert_actual_cost_receipt_matches_usage,
)
from true_rl_agent.agent_core.generation_closure import (
    load_existing_generation_execution, publish_or_validate_generation_closure,
)
from true_rl_agent.agent_core.generation_accounting import (
    collect_actual_cost_usage,
    session_retry_usage,
)
from true_rl_agent.agent_core.generation_provider_budget import collect_provider_actuals
from true_rl_agent.agent_core.generation_job_evidence import trial_receipt_ref
from true_rl_agent.agent_core.generation_resources import validate_stored_resource_closure_evidence
from true_rl_agent.agent_core.exploration import GenerationExplorationPlan
from true_rl_agent.agent_core.generation_finalization import (
    select_final_evaluation_targets,
    final_attempt_journal as _formal_final_attempt_journal,
)
from true_rl_agent.agent_core.generation_operations import OperationAdmission
from true_rl_agent.agent_core.generation_contract import (
    GENERATION_LANE_IDS,
    GENERATION_SESSION_CONTRACT_FILENAME,
    GENERATION_SESSION_CONTRACT_V3_SCHEMA_VERSION,
    GenerationLaneAssignment,
    GenerationSessionContract,
    GenerationSessionContractRef,
    generation_resource_plan_hash,
    load_generation_session_contract,
    publish_generation_session_contract,
)
from true_rl_agent.agent_core.generation_lifecycle import (
    GENERATION_CONTINUATION_POLICY_FILENAME,
    GENERATION_CONTINUATION_POLICY_SCHEMA_VERSION,
    GENERATION_CONTINUATION_POLICY_V2_SCHEMA_VERSION,
    GenerationContinuationPolicy,
    GenerationLifecycleAudit,
    GenerationLifecycleIdentity,
    GenerationTerminalBinding,
    audit_generation_lifecycle,
    build_generation_terminal_binding,
    load_generation_continuation_policy,
    publish_generation_continuation_policy,
)
from true_rl_agent.agent_core.generation_receipts import (
    GenerationReceiptRef,
    GenerationReceiptJournal,
)
from true_rl_agent.agent_core.generation_record import load_bound_generation_manifest
from true_rl_agent.agent_core.ledger import (
    PROVIDER_AUTHORING_MODE,
    load_ledger,
    primary_ledger_epoch_hash,
)
from true_rl_agent.agent_core.provider_limits import PROVIDER_ADAPTIVE_REASONING_POLICY_ID
from true_rl_agent.agent_core.session_genesis import (
    SESSION_GENESIS_FILENAME,
    SessionGenesisArtifact,
    VerifiedSessionGenesis,
    prepare_session_genesis,
    session_genesis_identity_hash,
)
from true_rl_agent.agent_core.trial_receipts import load_trial_eval_phase_receipt
from true_rl_agent.composition import (
    ResolvedCellManifestV1,
    ResolvedTaskComposition,
    TaskCompositionContext,
    resolve_task_composition,
)
from true_rl_agent.composition.cell_manifest import (
    RESOLVED_CELL_MANIFEST_FILENAME,
    load_resolved_cell_manifest,
)
from true_rl_agent.evaluation.acceptance_replay import (
    ProjectedPublicHistory,
    project_public_history,
)
from true_rl_agent.evaluation.binding import (
    EvaluationBindingV1,
    canonical_json_hash,
    require_sha256_digest,
)
from true_rl_agent.evaluation.current_best import CurrentBestEvalStats
from true_rl_agent.evaluation.phases import EvaluationPhase
from true_rl_agent.fs_contract import read_bounded_regular_file
from true_rl_agent.json_contract import (
    canonical_json_bytes,
    load_strict_json_object,
    strict_json_object_loads,
)
from true_rl_agent.prompt_artifacts import RoundAuthoringJournal
from true_rl_agent.runner.checkpoint import file_sha256
from true_rl_agent.runner.gpu_process_binding import is_full_gpu_uuid
from true_rl_agent.runner.gpu_lease import GpuDeviceIdentity, GpuLeaseManager
from true_rl_agent.runner.job_attempts import PreGenesisExecutionScope
from true_rl_agent.runner.specs import EvalJobSpec, EvalPhaseJobSpec, GpuProcessBinding
from true_rl_agent.runner.subprocess_runner import SubprocessRunner
from true_rl_agent.tasks.stair_climbing.composition import StairClimbingCompositionProvider
from true_rl_agent.tasks.stair_climbing.config import STAIR_TASK_ID
from true_rl_agent.tasks.stair_climbing.entrypoint_contract import (
    STAIR_EVAL_PURPOSE_FINAL_CONFIRMATION,
)
from true_rl_agent.tasks.stair_climbing.exploration import StairExplorationProjector
from true_rl_agent.tasks.stair_climbing.final_confirmation import (
    bind_stair_generation_final_config,
    STAIR_FINAL_CONFIRMATION_DIRNAME,
    STAIR_FINAL_CONFIRMATION_REQUIRED_GENERATIONS,
    STAIR_GENERATION_REPORT_FILENAME,
    StairFinalConfirmationPlanV2,
    StairFinalConfirmationReservationV2,
    StairFinalConfirmationResultV2,
    load_stair_generation_final_confirmation_plan,
    load_stair_generation_final_confirmation_reservation,
    load_stair_generation_final_confirmation_result,
    stair_final_confirmation_plan_path,
    stair_final_confirmation_reservation_path,
    stair_final_confirmation_result_path,
)
from true_rl_agent.tasks.stair_climbing.generation_authoring_timing import (
    STAIR_GENERATION_AUTHORING_TIMING_DIRNAME,
    replay_stair_generation_authoring_timing,
)
from true_rl_agent.tasks.stair_climbing.row_catalog import (
    DEFAULT_STAIR_CELL_VARIANT_ID,
    active_unified_stair_cell_variant_ids,
)
from true_rl_agent.tasks.stair_climbing.generation_session_cost import (
    DEVELOPMENT_GENERATION_PILOT_MODE,
    DEVELOPMENT_GENERATION_VALIDATION_MODE,
    FORMAL_GENERATION_SESSION_MODE,
    STAIR_FORMAL_GENERATION_PAUSE_BOUNDARIES,
    STAIR_GENERATION_MAX_INFRASTRUCTURE_RETRIES_PER_ELIGIBLE_JOB,
    STAIR_GENERATION_PROVIDER_ATTEMPTS_PER_LANE,
    STAIR_GENERATION_PROVIDER_INPUT_UNITS_PER_ATTEMPT,
    STAIR_GENERATION_PROVIDER_OUTPUT_UNITS_PER_ATTEMPT,
    STAIR_GENERATION_SESSION_COST_FILENAME,
    StairGenerationActualCostJournal,
    StairGenerationActualCostReceiptRef,
    StairGenerationSessionCostContractLike,
    StairGenerationSessionCostContractV2,
    StairGenerationSessionCostContractV3,
    StairGenerationSessionCostContractV4,
    build_stair_generation_scientific_cost_basis,
    load_stair_generation_session_cost_contract,
    replay_stair_generation_combined_cost_chain,
    replay_stair_generation_formal_cost_chain,
)
from true_rl_agent.tasks.stair_climbing.phased_adapter import StairPhasedEvaluationAdapter
from true_rl_agent.tasks.stair_climbing.phased_evaluation import (
    canonical_stair_phase_result_summary,
)
from true_rl_agent.tasks.stair_climbing.unified_scoring import (
    objective_family_mask_from_public_result,
)
from true_rl_agent.tasks.stair_climbing.seed_policy import STAIR_FINAL_CONFIRMATION_EVAL_NAMESPACE
from true_rl_agent.tasks.stair_climbing.source_checkout import require_stair_source_checkout


STAIR_GENERATION_REPORT_SCHEMA_VERSION = (
    "true_rl_agent.stair.generation_session_report.v2"
)


STAIR_GENERATION_VALIDATION_REPORT_SCHEMA_VERSION = (
    "true_rl_agent.stair.generation_session_report.v3"
)


STAIR_GENERATION_FORMAL_REPORT_SCHEMA_VERSION = (
    "true_rl_agent.stair.generation_session_report.v6"
)


STAIR_GENERATION_SOURCE_CHECKOUT_SCHEMA_VERSION = (
    "true_rl_agent.stair.generation_source_checkout.v1"
)


STAIR_GENERATION_SOURCE_CHECKOUT_FILENAME = "generation_source_checkout.json"


_FINAL_SOURCE_CHECKOUT_FILENAME = "source_checkout.json"


_FINAL_SOURCE_CHECKOUT_SCHEMA = "true_rl_agent.stair.finalization_source_checkout.v1"


STAIR_GENERATION_PROVIDER_TIMEOUT_S = 300


STAIR_GENERATION_PROVIDER_MODEL = "deepseek-v4-flash"


STAIR_GENERATION_PROVIDER_REASONING_EFFORT = "max"


STAIR_GENERATION_PROVIDER_BASE_URL = "https://api.deepseek.com"


STAIR_GENERATION_PROVIDER_ENDPOINT = "/chat/completions"


STAIR_GENERATION_PROVIDER_THINKING_DISABLED_TEMPERATURE = 0.2




_SOURCE_COMMIT = re.compile(r"[0-9a-f]{40}\Z")


_P6B_TERMINAL_REPORT_MAX_BYTES = 4 * 1024 * 1024


_GENERATION_SESSION_CONTRACT_MAX_BYTES = 128 * 1024


@dataclass(frozen=True)
class HistoricalStairGenerationConfig:
    """Operator-supplied physical limits; scientific semantics stay fixed."""

    project_root: Path
    output_dir: Path
    lease_root: Path
    session_id: str
    expected_source_commit: str
    machine_id: str
    gpu0_uuid: str
    gpu1_uuid: str
    max_allocated_active_gpu_hours: float
    max_simulated_control_steps: int
    max_output_bytes: int
    source_reference_genesis_dir: Path | None = None
    source_reference_session_id: str | None = None
    cell_variant_id: str = DEFAULT_STAIR_CELL_VARIANT_ID
    nvidia_smi_executable: str = "nvidia-smi"
    resume_existing: bool = False
    session_mode: str = DEVELOPMENT_GENERATION_PILOT_MODE
    pause_after_generation: int | None = None
    terminal_audit_only: bool = False
    final_replication_source: Path | None = None

    def __post_init__(self) -> None:
        if self.final_replication_source is not None and (
            not isinstance(self.final_replication_source, Path)
            or self.session_mode != FORMAL_GENERATION_SESSION_MODE
        ):
            raise ValueError("final replication requires a source path and a formal study")
        for field_name in (
            "project_root",
            "output_dir",
            "lease_root",
        ):
            if not isinstance(getattr(self, field_name), Path):
                raise TypeError(f"{field_name} must be a Path")
        if not self.output_dir.is_absolute() or not self.lease_root.is_absolute():
            raise ValueError("generation output_dir and lease_root must be absolute")
        if (self.source_reference_genesis_dir is None) != (
            self.source_reference_session_id is None
        ):
            raise ValueError(
                "source reference Genesis directory and session_id must be provided together"
            )
        if self.source_reference_genesis_dir is not None:
            if not isinstance(self.source_reference_genesis_dir, Path):
                raise TypeError("source_reference_genesis_dir must be a Path")
            if not self.source_reference_genesis_dir.is_absolute():
                raise ValueError("source_reference_genesis_dir must be absolute")
            if self.source_reference_genesis_dir.resolve() == self.output_dir.resolve():
                raise ValueError(
                    "source reference Genesis directory must differ from output_dir"
                )
            if (
                not isinstance(self.source_reference_session_id, str)
                or not self.source_reference_session_id
            ):
                raise ValueError("source_reference_session_id is required")
        if not isinstance(self.session_id, str) or not self.session_id:
            raise ValueError("generation session_id is required")
        if self.cell_variant_id not in active_unified_stair_cell_variant_ids():
            raise ValueError(
                "generation cell_variant_id must select an active unified Stair row"
            )
        if _SOURCE_COMMIT.fullmatch(self.expected_source_commit) is None:
            raise ValueError("expected_source_commit must be a lowercase Git SHA")
        if not isinstance(self.machine_id, str) or not self.machine_id.strip():
            raise ValueError("generation machine_id is required")
        for field_name in ("gpu0_uuid", "gpu1_uuid"):
            if not is_full_gpu_uuid(getattr(self, field_name)):
                raise ValueError(f"{field_name} must be one full GPU UUID")
        if self.gpu0_uuid == self.gpu1_uuid:
            raise ValueError("generation lanes require two distinct physical GPUs")
        if (
            isinstance(self.max_allocated_active_gpu_hours, bool)
            or not isinstance(self.max_allocated_active_gpu_hours, int | float)
            or float(self.max_allocated_active_gpu_hours) <= 0.0
        ):
            raise ValueError("max_allocated_active_gpu_hours must be positive")
        for field_name in ("max_simulated_control_steps", "max_output_bytes"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        if (
            not isinstance(self.nvidia_smi_executable, str)
            or not self.nvidia_smi_executable.strip()
        ):
            raise ValueError("nvidia_smi_executable is required")
        if not isinstance(self.resume_existing, bool):
            raise TypeError("resume_existing must be boolean")
        if self.session_mode not in {
            DEVELOPMENT_GENERATION_PILOT_MODE,
            DEVELOPMENT_GENERATION_VALIDATION_MODE,
            FORMAL_GENERATION_SESSION_MODE,
        }:
            raise ValueError("generation session mode is unsupported")
        if self.pause_after_generation is not None:
            allowed_boundaries = {
                DEVELOPMENT_GENERATION_VALIDATION_MODE: {1, 2},
                FORMAL_GENERATION_SESSION_MODE: set(
                    STAIR_FORMAL_GENERATION_PAUSE_BOUNDARIES
                ),
            }.get(self.session_mode, set())
            if (
                isinstance(self.pause_after_generation, bool)
                or not isinstance(self.pause_after_generation, int)
                or self.pause_after_generation not in allowed_boundaries
            ):
                raise ValueError(
                    "generation boundary pause is not valid for the session mode"
                )
        if not isinstance(self.terminal_audit_only, bool):
            raise TypeError("terminal_audit_only must be boolean")
        if self.terminal_audit_only and (
            self.session_mode
            not in {
                DEVELOPMENT_GENERATION_VALIDATION_MODE,
                FORMAL_GENERATION_SESSION_MODE,
            }
            or not self.resume_existing
            or self.pause_after_generation is not None
        ):
            raise ValueError(
                "terminal audit requires resume_existing and cannot request a pause"
            )

    @property
    def max_generations(self) -> int:
        return {
            DEVELOPMENT_GENERATION_PILOT_MODE: 1,
            DEVELOPMENT_GENERATION_VALIDATION_MODE: 3,
            FORMAL_GENERATION_SESSION_MODE: (
                STAIR_FINAL_CONFIRMATION_REQUIRED_GENERATIONS
            ),
        }[self.session_mode]

    @property
    def imports_reference_genesis(self) -> bool:
        return self.source_reference_genesis_dir is not None


@dataclass(frozen=True)
class CompletedStairGenerationSession:
    """Read-only typed base closure of one completed three-generation session.

    ``report`` is identity-checked input, not trusted proof of its own claims.
    Callers must independently replay acceptance and terminal/scientific claims.
    """

    output_dir: Path
    source_checkout: Mapping[str, Any]
    resolved_cell_manifest: ResolvedCellManifestV1
    session_genesis: SessionGenesisArtifact
    session_contract: GenerationSessionContract
    cost_contract: StairGenerationSessionCostContractV2
    continuation_policy: GenerationContinuationPolicy
    ledger: tuple[Mapping[str, Any], ...]
    lifecycle: GenerationLifecycleAudit
    actual_cost_receipts: tuple[StairGenerationActualCostReceiptRef, ...]
    generation_execution_receipts: tuple[GenerationReceiptRef, ...]
    report: Mapping[str, Any]


def _load_existing_session_genesis_for_terminal_audit(
    foundation: GenerationSessionFoundation,
) -> VerifiedSessionGenesis:
    genesis = _prepare_foundation_session_genesis(foundation)
    _bind_unified_objective_family_mask(foundation, genesis=genesis)
    return genesis


def _prepare_foundation_session_genesis(
    foundation: GenerationSessionFoundation,
) -> VerifiedSessionGenesis:
    config = foundation.config
    composition = foundation.composition
    snapshot = composition.genesis_identity_snapshot()
    identity_hash = session_genesis_identity_hash(snapshot)
    return prepare_session_genesis(
        output_dir=config.output_dir,
        session_id=config.session_id,
        task_plugin=foundation.composition.task_plugin,
        backend_adapter=foundation.composition.backend_adapter,
        runner=foundation.runner,
        policy_initialization=foundation.composition.policy_initialization,
        genesis_identity_snapshot=snapshot,
        baseline_manifest=foundation.composition.baseline_manifest,
        reference_program_artifact=(foundation.composition.reference_program_artifact),
        bootstrap_initial_best_score=(
            foundation.composition.bootstrap_initial_best_score
        ),
        bootstrap_initial_policy_state_hash=(
            foundation.composition.bootstrap_initial_policy_state_hash
        ),
        bind_backend_execution_environment=True,
        generation_session_contract_ref=foundation.session_contract_ref,
        **_session_genesis_execution_kwargs(
            foundation,
            genesis_identity_hash=identity_hash,
        ),
        require_existing=True,
    )


def _bind_unified_objective_family_mask(
    foundation: GenerationSessionFoundation,
    *,
    genesis: VerifiedSessionGenesis,
) -> None:
    plugin = foundation.composition.task_plugin
    from true_rl_agent.tasks.stair_climbing.config import StairClimbingTaskConfig

    if not isinstance(getattr(plugin, "config", None), StairClimbingTaskConfig):
        return
    if not plugin.config.is_unified_protocol:
        return
    evidence = genesis.initial_anchor_public_evidence()
    public_result = evidence.get("public_result") if isinstance(evidence, Mapping) else None
    mask = objective_family_mask_from_public_result(
        suite=plugin.config.ordered_evaluation_suite,
        public_result=public_result,
    )
    plugin.bind_objective_family_mask(mask)


def _load_foundation(
    config: HistoricalStairGenerationConfig,
    *,
    source_commit: str,
    inventory: tuple[str, str],
) -> GenerationSessionFoundation:
    provider = StairClimbingCompositionProvider()
    values = dict(provider.cli_defaults())
    values["cell_variant_id"] = config.cell_variant_id
    formal_final_plan: StairFinalConfirmationPlanV2 | None = None
    final_plan_path: Path | None = None
    if config.session_mode == FORMAL_GENERATION_SESSION_MODE:
        provisional_composition = resolve_task_composition(
            provider=provider,
            values=values,
            context=TaskCompositionContext(
                project_root=config.project_root,
                output_dir=config.output_dir,
                read_only=True,
            ),
        )
        if provisional_composition.task_plugin.config.protocol_state != "active":
            raise ValueError(
                "Stair unified protocol is draft; formal generation is disabled"
            )
        final_plan_path = stair_final_confirmation_plan_path(config.output_dir)
        formal_final_plan = load_stair_generation_final_confirmation_plan(
            final_plan_path,
            expected_session_id=config.session_id,
            expected_config=provisional_composition.task_plugin.config,
        )
        if config.final_replication_source is not None:
            from true_rl_agent.tasks.stair_climbing.study_holdout import StairStudyHoldout

            source = StairStudyHoldout.load(config.final_replication_source)
            if formal_final_plan.replication_of != source.commitment:
                raise ValueError("Stair resumed final replication source mismatch")
    values.update(
        {
            "mode": config.session_mode,
            "session_id": config.session_id,
            "final_confirmation_plan": final_plan_path,
            "max_generations": config.max_generations,
            "initial_best_score": 0.0,
        }
    )
    composition = resolve_task_composition(
        provider=provider,
        values=values,
        context=TaskCompositionContext(
            project_root=config.project_root,
            output_dir=config.output_dir,
            read_only=True,
        ),
    )
    if composition.resolved_cell_manifest is None:
        raise ValueError("Stair generation requires a resolved cell manifest")
    if composition.resolved_cell_manifest.cell_variant_id != config.cell_variant_id:
        raise ValueError("Stair generation resolved a different robot row")
    if not isinstance(composition.runner, SubprocessRunner):
        raise TypeError("Stair generation requires the production SubprocessRunner")
    if composition.genesis_anchor_kind != "reference_program":
        raise ValueError("Stair generation pilot requires reference-program Genesis")
    identities = (
        GpuDeviceIdentity("cuda:0", config.machine_id, inventory[0]),
        GpuDeviceIdentity("cuda:1", config.machine_id, inventory[1]),
    )
    manager = GpuLeaseManager(
        lease_root=config.lease_root,
        devices={identity.execution_device: identity for identity in identities},
    )
    runner = replace(composition.runner, gpu_lease_manager=manager)
    bindings = tuple(
        GpuProcessBinding(
            host_execution_device=identity.execution_device,
            process_execution_device="cuda:0",
            resource_binding_hash=identity.identity_hash,
        )
        for identity in identities
    )
    assert len(bindings) == 2
    phased_adapter = composition.task_plugin.phased_evaluation_adapter()
    if not isinstance(phased_adapter, StairPhasedEvaluationAdapter):
        raise TypeError("Stair composition returned an invalid phased adapter")
    scientific_cost_basis = build_stair_generation_scientific_cost_basis(
        session_id=config.session_id,
        config=composition.task_plugin.config,
    )
    if (
        scientific_cost_basis.phased_evaluation_cost_contract_hash
        != phased_adapter.cost_contract()["contract_hash"]
    ):
        raise ValueError("Stair phased evaluation cost source drifted")
    # Frozen data, not today's launch defaults, owns historical budget semantics.
    cost = load_stair_generation_session_cost_contract(
        config.output_dir / STAIR_GENERATION_SESSION_COST_FILENAME,
    )
    if (cost.scientific_cost_basis != scientific_cost_basis
            or cost.session_mode != config.session_mode
            or cost.max_generations != config.max_generations):
        raise ValueError("historical session workload differs from the resolved task")
    if (isinstance(cost, StairGenerationSessionCostContractV2)
            and cost.imports_reference_evidence != config.imports_reference_genesis):
        raise ValueError("historical reference-import semantics differ from the request")
    if (float(config.max_allocated_active_gpu_hours) != cost.max_allocated_active_gpu_hours
            or config.max_simulated_control_steps != cost.max_simulated_control_steps
            or config.max_output_bytes != cost.max_output_bytes):
        raise ValueError("historical operator caps differ from the frozen cost contract")
    continuation = GenerationContinuationPolicy(
        schema_version=(
            GENERATION_CONTINUATION_POLICY_V2_SCHEMA_VERSION
            if config.session_mode == FORMAL_GENERATION_SESSION_MODE
            else GENERATION_CONTINUATION_POLICY_SCHEMA_VERSION
        ),
        max_segments=(
            7
            if config.session_mode == FORMAL_GENERATION_SESSION_MODE
            else (
                4
                if config.session_mode == DEVELOPMENT_GENERATION_VALIDATION_MODE
                else 2
            )
        ),
        allowed_continuation_reasons=(
            "operator_pause",
            "typed_infrastructure_failure",
            "exception",
            "budget_exhausted",
            "contract_incomplete",
        ),
        operator_pause_generation_boundaries=(
            STAIR_FORMAL_GENERATION_PAUSE_BOUNDARIES
            if config.session_mode == FORMAL_GENERATION_SESSION_MODE
            else ()
        ),
    )
    publish_generation_continuation_policy(
        config.output_dir, continuation, require_existing=True,
    )
    provider_policy = _provider_policy()
    if config.session_mode == FORMAL_GENERATION_SESSION_MODE:
        if formal_final_plan is None or formal_final_plan.plan_hash is None:
            raise ValueError("formal generation final confirmation plan is missing")
        final_confirmation_policy = formal_final_plan.genesis_commitment()
        final_confirmation_plan_hash = formal_final_plan.plan_hash
    else:
        final_confirmation_policy = {
            "schema_version": (
                "true_rl_agent.stair.generation_final_confirmation_policy.v1"
            ),
            "session_mode": config.session_mode,
            "enabled": False,
            "reason": "development_generation_has_no_terminal_confirmation",
        }
        final_confirmation_plan_hash = canonical_json_hash(
            final_confirmation_policy
        )
    projector_contract = StairExplorationProjector(
        composition.task_plugin.config
    ).contract()
    placeholder_intent_hash = canonical_json_hash(
        {"purpose": "resource-plan-hash-does-not-include-lane-intent"}
    )
    resource_assignments = tuple(
        GenerationLaneAssignment(
            lane_id=lane_id,
            execution_device=binding.process_execution_device,
            resource_binding_hash=binding.resource_binding_hash,
            lane_intent_hash=placeholder_intent_hash,
            gpu_process_binding=binding,
        )
        for lane_id, binding in zip(GENERATION_LANE_IDS, bindings, strict=True)
    )
    session_contract = GenerationSessionContract(
        schema_version=GENERATION_SESSION_CONTRACT_V3_SCHEMA_VERSION,
        session_mode=config.session_mode,
        session_id=config.session_id,
        task_id=composition.task_id,
        backend_id=composition.backend_adapter.backend_id,
        max_generations=config.max_generations,
        max_trials=2 * config.max_generations,
        max_train_jobs=2 * config.max_generations,
        bound_contract_hashes={
            "authoring_policy": canonical_json_hash(_authoring_policy(composition)),
            "continuation_policy": continuation.policy_hash,
            "evaluation_cost": canonical_json_hash(phased_adapter.cost_contract()),
            "evaluation_protocol": (
                composition.task_plugin.config.evaluation_protocol().protocol_hash
            ),
            "exploration_projector": canonical_json_hash(projector_contract),
            "final_confirmation_plan": final_confirmation_plan_hash,
            "provider_policy": canonical_json_hash(provider_policy),
            "resource_plan": generation_resource_plan_hash(
                resource_assignments  # type: ignore[arg-type]
            ),
            "retry_policy": canonical_json_hash(_retry_policy(cost)),
            "selector": canonical_json_hash(phased_adapter.selection_contract()),
            "session_cost": cost.contract_hash,
        },
    )
    session_contract, session_contract_ref = (
        _load_compatible_session_contract(
            config,
            session_contract,
        )
    )
    if isinstance(cost, StairGenerationSessionCostContractV2):
        runner = replace(
            runner,
            attempt_admission_hook=OperationAdmission(
                output_dir=config.output_dir,
                session_id=config.session_id,
                cost_contract=cost,
                session_contract_hash=session_contract.contract_hash,
                termination_grace_s=runner.termination_grace_s,
            ),
        )
    return GenerationSessionFoundation(
        config=config,
        source_commit=source_commit,
        composition=composition,
        runner=runner,
        phased_adapter=phased_adapter,
        continuation_policy=continuation,
        cost_contract=cost,
        session_contract=session_contract,
        session_contract_ref=session_contract_ref,
        gpu_identities=identities,
        gpu_bindings=(bindings[0], bindings[1]),
        gpu_lease_manager=manager,
        provider_policy=provider_policy,
        final_confirmation_policy=final_confirmation_policy,
        final_confirmation_plan=formal_final_plan,
    )


def _load_compatible_session_contract(
    config: HistoricalStairGenerationConfig,
    current: GenerationSessionContract,
) -> tuple[GenerationSessionContract, GenerationSessionContractRef]:
    """Retain one closed P6-B contract across authoring-environment drift.

    The compatibility branch is intentionally task-owned and terminal-only.
    It cannot authorize a fresh or in-progress session, P6-C, or drift in any
    scientific/resource contract other than the environment-derived authoring
    policy identity.  Session Genesis and the terminal report loader then bind
    and replay the exact returned stored-contract reference.
    """

    try:
        ref = publish_generation_session_contract(
            config.output_dir, current, require_existing=True,
        )
    except FileExistsError:
        if (
            not config.resume_existing
            or config.session_mode != DEVELOPMENT_GENERATION_PILOT_MODE
        ):
            raise
        stored, ref = _load_terminal_p6b_session_contract(config)
        _assert_terminal_p6b_contract_compatibility(
            stored=stored,
            current=current,
        )
        return stored, ref
    return current, ref


def _load_terminal_p6b_session_contract(
    config: HistoricalStairGenerationConfig,
) -> tuple[GenerationSessionContract, GenerationSessionContractRef]:
    report_path = config.output_dir / STAIR_GENERATION_REPORT_FILENAME
    report_payload = read_bounded_regular_file(
        report_path,
        max_bytes=_P6B_TERMINAL_REPORT_MAX_BYTES,
        label="terminal P6-B generation report",
    )
    report = strict_json_object_loads(
        report_payload,
        context="terminal P6-B generation report",
    )
    if report_payload != canonical_json_bytes(report, trailing_newline=True):
        raise ValueError("terminal P6-B generation report is not canonical JSON")
    terminal = report.get("terminal_binding")
    checks = report.get("checks")
    if (
        report.get("schema_version") != STAIR_GENERATION_REPORT_SCHEMA_VERSION
        or report.get("session_mode") != DEVELOPMENT_GENERATION_PILOT_MODE
        or report.get("session_id") != config.session_id
        or report.get("output_dir") != str(config.output_dir.resolve())
        or report.get("acceptance_authority") is not False
        or not isinstance(checks, Mapping)
        or not checks
        or not all(value is True for value in checks.values())
        or checks.get("mode_is_development_generation_pilot") is not True
        or checks.get("lifecycle_terminal") is not True
        or checks.get("one_generation_one_ledger_record") is not True
        or not isinstance(terminal, Mapping)
        or terminal.get("schema_version")
        != "true_rl_agent.generation_terminal_binding.v1"
        or terminal.get("acceptance_authority") is not False
        or terminal.get("closed_generation_count") != 1
        or terminal.get("required_generation_count") != 1
    ):
        raise ValueError(
            "stored generation contract compatibility requires a canonical "
            "terminal P6-B v2 report"
        )

    contract_path = config.output_dir / GENERATION_SESSION_CONTRACT_FILENAME
    contract_payload = read_bounded_regular_file(
        contract_path,
        max_bytes=_GENERATION_SESSION_CONTRACT_MAX_BYTES,
        label="stored terminal P6-B generation session contract",
    )
    contract_value = strict_json_object_loads(
        contract_payload,
        context="stored terminal P6-B generation session contract",
    )
    if contract_payload != canonical_json_bytes(
        contract_value,
        trailing_newline=True,
    ):
        raise ValueError(
            "stored terminal P6-B generation session contract is not canonical JSON"
        )
    stored = GenerationSessionContract.from_mapping(contract_value)
    if report.get("generation_session_contract_hash") != stored.contract_hash:
        raise ValueError(
            "terminal P6-B report does not bind the stored generation session contract"
        )
    return stored, GenerationSessionContractRef(
        logical_name=GENERATION_SESSION_CONTRACT_FILENAME,
        checksum="sha256:" + hashlib.sha256(contract_payload).hexdigest(),
        size_bytes=len(contract_payload),
    )


def _assert_terminal_p6b_contract_compatibility(
    *,
    stored: GenerationSessionContract,
    current: GenerationSessionContract,
) -> None:
    identity_fields = (
        "schema_version",
        "session_mode",
        "session_id",
        "task_id",
        "backend_id",
        "width",
        "authoring_order",
        "ledger_record_schema",
        "max_generations",
        "max_trials",
        "max_train_jobs",
    )
    drifted_identity = [
        field_name
        for field_name in identity_fields
        if getattr(stored, field_name) != getattr(current, field_name)
    ]
    if drifted_identity:
        raise ValueError(
            "terminal P6-B stored generation contract identity drifted: "
            + ", ".join(drifted_identity)
        )
    stored_hashes = dict(stored.bound_contract_hashes)
    current_hashes = dict(current.bound_contract_hashes)
    non_authoring_fields = sorted(
        (set(stored_hashes) | set(current_hashes)) - {"authoring_policy"}
    )
    drifted_hashes = [
        field_name
        for field_name in non_authoring_fields
        if stored_hashes.get(field_name) != current_hashes.get(field_name)
    ]
    if drifted_hashes:
        raise ValueError(
            "terminal P6-B stored generation contract bound hashes drifted: "
            + ", ".join(drifted_hashes)
        )


def _formal_lifecycle_identity(
    foundation: GenerationSessionFoundation,
    *,
    genesis: VerifiedSessionGenesis,
) -> GenerationLifecycleIdentity:
    contract = foundation.session_contract
    if (
        contract.session_mode != FORMAL_GENERATION_SESSION_MODE
        or contract.max_generations != STAIR_FINAL_CONFIRMATION_REQUIRED_GENERATIONS
    ):
        raise ValueError("P6-D lifecycle requires the frozen formal contract")
    return _bounded_lifecycle_identity(foundation, genesis=genesis)


def _bounded_lifecycle_identity(
    foundation: GenerationSessionFoundation,
    *,
    genesis: VerifiedSessionGenesis,
) -> GenerationLifecycleIdentity:
    contract = foundation.session_contract
    if (contract.session_mode, contract.max_generations) not in {
        (DEVELOPMENT_GENERATION_VALIDATION_MODE, 3),
        (
            FORMAL_GENERATION_SESSION_MODE,
            STAIR_FINAL_CONFIRMATION_REQUIRED_GENERATIONS,
        ),
    }:
        raise ValueError("bounded lifecycle contract is unsupported")
    return GenerationLifecycleIdentity(
        session_id=foundation.config.session_id,
        session_genesis_hash=genesis.session_genesis_hash,
        generation_session_contract_hash=contract.contract_hash,
        final_confirmation_plan_hash=contract.bound_contract_hashes[
            "final_confirmation_plan"
        ],
        ledger_epoch_hash=primary_ledger_epoch_hash(genesis.session_genesis_hash),
        output_root=str(foundation.config.output_dir.resolve()),
    )


@dataclass(frozen=True)
class _FormalFinalEvaluationPair:
    genesis_plan: EvalJobSpec
    final_plan: EvalJobSpec
    genesis_phase_spec: EvalPhaseJobSpec
    final_phase_spec: EvalPhaseJobSpec
    final_current_best: CurrentBestEvalStats


def _build_formal_final_evaluation_pair(
    foundation: GenerationSessionFoundation,
    *,
    genesis: VerifiedSessionGenesis,
    ledger: list[dict[str, Any]],
    terminal: GenerationTerminalBinding,
) -> _FormalFinalEvaluationPair:
    targets = select_final_evaluation_targets(
        genesis=genesis, ledger=ledger, terminal=terminal,
    )
    genesis_plan, final_target = targets.reference, targets.terminal
    current_best = targets.current_best
    session_id = foundation.config.session_id
    bound_genesis = _bind_formal_final_eval_plan(
        foundation,
        template=genesis_plan,
        target=genesis_plan,
        role="genesis",
        base_job_id=f"{session_id}-final-confirmation-genesis-eval",
    )
    bound_final = _bind_formal_final_eval_plan(
        foundation,
        template=genesis_plan,
        target=final_target,
        role="final",
        base_job_id=f"{session_id}-final-confirmation-final-eval",
    )
    backend = foundation.composition.backend_adapter
    genesis_phase = backend.materialize_eval_phase(
        eval_plan=bound_genesis,
        phase=EvaluationPhase.SEALED_VALIDATION,
    )
    final_phase = backend.materialize_eval_phase(
        eval_plan=bound_final,
        phase=EvaluationPhase.SEALED_VALIDATION,
    )
    return _FormalFinalEvaluationPair(
        genesis_plan=bound_genesis,
        final_plan=bound_final,
        genesis_phase_spec=genesis_phase,
        final_phase_spec=final_phase,
        final_current_best=current_best,
    )


def _formal_final_config(foundation: GenerationSessionFoundation):
    plan = foundation.final_confirmation_plan
    if plan is None:
        raise ValueError("Stair formal final plan is missing")
    return bind_stair_generation_final_config(
        foundation.composition.task_plugin.config,
        plan=plan,
        plan_path=stair_final_confirmation_plan_path(foundation.config.output_dir),
    )


def _bind_formal_final_eval_plan(
    foundation: GenerationSessionFoundation,
    *,
    template: EvalJobSpec,
    target: EvalJobSpec,
    role: str,
    base_job_id: str,
) -> EvalJobSpec:
    if role not in {"genesis", "final"}:
        raise ValueError("P6-D final eval role is invalid")
    plan = foundation.final_confirmation_plan
    if plan is None:
        raise ValueError("P6-D final eval plan is missing")
    checkpoint = target.execution_checkpoint()
    require_sha256_digest(
        target.subject_manifest_hash,
        "P6-D final eval subject manifest hash",
    )
    local_path = checkpoint.metadata.get("local_path")
    if not isinstance(local_path, str) or not local_path:
        raise ValueError("P6-D final eval checkpoint lacks a private local path")
    local_checkpoint = Path(local_path)
    if file_sha256(local_checkpoint) != checkpoint.checksum:
        raise ValueError("P6-D final eval checkpoint checksum mismatch")
    binding = EvaluationBindingV1(
        eval_protocol_hash=template.evaluation_binding.eval_protocol_hash,
        acceptance_protocol_hash=(
            template.evaluation_binding.acceptance_protocol_hash
        ),
        task_protocol_hash=template.evaluation_binding.task_protocol_hash,
        seed_cohort_hash=plan.seed_cohort_hash,
    )
    gpu_binding = foundation.gpu_bindings[0]
    summary_path = (
        foundation.config.output_dir
        / "runner"
        / STAIR_FINAL_CONFIRMATION_DIRNAME
        / role
        / "eval"
        / "summary.json"
    ).resolve(strict=False)
    summary_path_text = summary_path.as_posix()
    if template.metadata.get("stair_eval_purpose") is not None:
        raise ValueError("P6-D final eval template already declares a purpose")
    bound = foundation.composition.backend_adapter.rematerialize_eval_plan(
        eval_plan=replace(
            target,
            job_id=base_job_id,
            evaluation_binding=binding,
            execution_device=gpu_binding.process_execution_device,
            attempt_scope=None,
            gpu_process_binding=gpu_binding,
        ),
        round_id=base_job_id,
        summary_path=summary_path_text,
    )
    argv = list(bound.metadata["cli_argv"])
    if plan.study_holdout_commitment is not None:
        argv.extend([
            "--stair-final-plan",
            stair_final_confirmation_plan_path(foundation.config.output_dir).as_posix(),
        ])
    argv.extend(
        [
            "--stair-eval-purpose",
            STAIR_EVAL_PURPOSE_FINAL_CONFIRMATION,
        ]
    )
    metadata = dict(bound.metadata)
    metadata.update(
        {
            "cli_argv": argv,
            "cli_command": " ".join(argv),
            "summary_path": summary_path_text,
            "stair_eval_purpose": STAIR_EVAL_PURPOSE_FINAL_CONFIRMATION,
        }
    )
    return replace(bound, metadata=metadata)


def _load_formal_source_checkout_identity(
    foundation: GenerationSessionFoundation,
) -> Mapping[str, Any]:
    payload = load_strict_json_object(
        foundation.config.output_dir / STAIR_GENERATION_SOURCE_CHECKOUT_FILENAME,
        context="P6-D source checkout identity",
    )
    if set(payload) != {
        "schema_version",
        "source_commit",
        "source_checkout_hash",
    }:
        raise ValueError("P6-D source checkout identity fields mismatch")
    body = {
        "schema_version": STAIR_GENERATION_SOURCE_CHECKOUT_SCHEMA_VERSION,
        "source_commit": foundation.source_commit,
    }
    if (
        payload.get("schema_version")
        != STAIR_GENERATION_SOURCE_CHECKOUT_SCHEMA_VERSION
        or payload.get("source_commit") != foundation.source_commit
        or payload.get("source_checkout_hash") != canonical_json_hash(body)
    ):
        raise ValueError("P6-D source checkout identity mismatch")
    return payload


def _formal_resume_generation_source(
    config: HistoricalStairGenerationConfig, executable_commit: str,
) -> str:
    """A closed session may use a repaired finalizer, never new generation code."""
    source = load_strict_json_object(
        config.output_dir / STAIR_GENERATION_SOURCE_CHECKOUT_FILENAME,
        context="P6-D original generation source",
    )
    original_commit = source.get("source_commit")
    if original_commit == executable_commit:
        return executable_commit
    # Reuse the launch gate's full lifecycle/cost/ledger inspection. A partial
    # generation or a continuable segment cannot cross the source boundary.
    from .launch_history import (
        STAIR_FORMAL_LAUNCH_SUBJECT_TERMINAL_AUDIT_ONLY,
        STAIR_FORMAL_LAUNCH_SUBJECT_TERMINAL_FINALIZATION,
        inspect_historical_formal_launch_subject,
    )
    subject = inspect_historical_formal_launch_subject(
        formal_output_dir=config.output_dir,
        source_root=config.project_root,
        session_id=config.session_id,
        expected_source_commit=executable_commit,
        subject_mode=(
            STAIR_FORMAL_LAUNCH_SUBJECT_TERMINAL_AUDIT_ONLY
            if config.terminal_audit_only
            else STAIR_FORMAL_LAUNCH_SUBJECT_TERMINAL_FINALIZATION
        ),
    )
    return subject["source_identity"]["source_commit"]


def _formal_final_execution_source(
    foundation: GenerationSessionFoundation,
) -> Mapping[str, Any]:
    """Bind the finalizer separately without rewriting generation provenance."""
    generation_source = _load_formal_source_checkout_identity(foundation)
    path = (
        foundation.config.output_dir / STAIR_FINAL_CONFIRMATION_DIRNAME
        / _FINAL_SOURCE_CHECKOUT_FILENAME
    )
    executable_commit = foundation.config.expected_source_commit
    body = {
        "schema_version": _FINAL_SOURCE_CHECKOUT_SCHEMA,
        "source_commit": executable_commit,
        "generation_source_checkout_hash": generation_source["source_checkout_hash"],
    }
    if path.is_file():
        stored = _load_completed_stair_canonical_json(
            path, context="P6-D finalizer source checkout",
        )
        stored_body = {key: stored.get(key) for key in body}
        if (
            set(stored) != {*body, "source_checkout_hash"}
            or stored_body["schema_version"] != _FINAL_SOURCE_CHECKOUT_SCHEMA
            or _SOURCE_COMMIT.fullmatch(str(stored_body["source_commit"])) is None
            or stored_body["generation_source_checkout_hash"]
            != generation_source["source_checkout_hash"]
            or stored["source_checkout_hash"] != canonical_json_hash(stored_body)
        ):
            raise ValueError("P6-D finalizer source checkout mismatch")
        return stored
    return generation_source


def _build_formal_final_reservation(
    foundation: GenerationSessionFoundation,
    *,
    genesis: VerifiedSessionGenesis,
    plan: StairFinalConfirmationPlanV2,
    pair: _FormalFinalEvaluationPair,
    terminal: GenerationTerminalBinding,
    ledger_checksum: str,
    source_checkout_hash: str,
) -> StairFinalConfirmationReservationV2:
    """Deterministically bind one replayed terminal head to its final eval pair."""

    backend_manifest_hash = genesis.backend_execution_environment_manifest_hash
    if not isinstance(backend_manifest_hash, str):
        raise ValueError("P6-D backend execution environment is unbound")
    genesis_checkpoint = pair.genesis_plan.execution_checkpoint()
    final_checkpoint = pair.final_plan.execution_checkpoint()
    return StairFinalConfirmationReservationV2(
        plan_hash=plan.plan_hash,
        session_id=foundation.config.session_id,
        session_genesis_hash=genesis.session_genesis_hash,
        generation_session_contract_hash=foundation.session_contract.contract_hash,
        generation_cost_contract_hash=foundation.cost_contract.contract_hash,
        ledger_checksum=ledger_checksum,
        terminal_binding=terminal.to_dict(),
        source_checkout_hash=source_checkout_hash,
        backend_execution_environment_manifest_hash=backend_manifest_hash,
        gpu_process_binding=foundation.gpu_bindings[0],
        genesis_job_id=pair.genesis_phase_spec.job_id,
        final_job_id=pair.final_phase_spec.job_id,
        genesis_eval_phase_spec_hash=canonical_json_hash(
            pair.genesis_phase_spec.to_public_dict()
        ),
        final_eval_phase_spec_hash=canonical_json_hash(
            pair.final_phase_spec.to_public_dict()
        ),
        genesis_evaluation_plan_hash=canonical_json_hash(
            pair.genesis_plan.to_public_dict()
        ),
        final_evaluation_plan_hash=canonical_json_hash(
            pair.final_plan.to_public_dict()
        ),
        genesis_checkpoint_hash=genesis_checkpoint.checksum,
        final_checkpoint_hash=final_checkpoint.checksum,
        genesis_source_manifest_hash=require_sha256_digest(
            pair.genesis_plan.subject_manifest_hash,
            "P6-D Genesis source manifest hash",
        ),
        final_source_manifest_hash=require_sha256_digest(
            pair.final_plan.subject_manifest_hash,
            "P6-D final source manifest hash",
        ),
        genesis_training_binding_hash=pair.genesis_plan.training_binding_hash,
        final_training_binding_hash=pair.final_plan.training_binding_hash,
        genesis_capability_profile_hash=pair.genesis_plan.capability_profile_hash,
        final_capability_profile_hash=pair.final_plan.capability_profile_hash,
        final_anchor=pair.final_current_best.to_public_dict(),
    )


def _validate_formal_final_reservation_against_terminal_replay(
    foundation: GenerationSessionFoundation,
    *,
    genesis: VerifiedSessionGenesis,
    ledger: list[dict[str, Any]],
    reservation: StairFinalConfirmationReservationV2,
) -> None:
    """Require the persisted Reservation to match authoritative source replay."""

    plan = foundation.final_confirmation_plan
    if not isinstance(plan, StairFinalConfirmationPlanV2):
        raise ValueError("P6-D terminal audit final Plan is missing")
    terminal = build_generation_terminal_binding(
        ledger_path=foundation.config.output_dir / "ledger.jsonl",
        output_dir=foundation.config.output_dir,
        attempt_journal_root=foundation.config.output_dir / "job_attempts",
        identity=_formal_lifecycle_identity(foundation, genesis=genesis),
        policy=foundation.continuation_policy,
        session_contract=foundation.session_contract,
        initial_current_best=genesis.initial_current_best,
        acceptance_protocol=(
            foundation.composition.task_plugin.task_spec.acceptance_protocol
        ),
        expected_initial_policy_state_hash=(
            genesis.expected_initial_policy_state_hash
        ),
    )
    pair = _build_formal_final_evaluation_pair(
        foundation,
        genesis=genesis,
        ledger=ledger,
        terminal=terminal,
    )
    ledger_checksum = file_sha256(foundation.config.output_dir / "ledger.jsonl")
    if ledger_checksum is None:
        raise ValueError("P6-D terminal audit ledger checksum is unavailable")
    source = _formal_final_execution_source(foundation)
    replayed = _build_formal_final_reservation(
        foundation,
        genesis=genesis,
        plan=plan,
        pair=pair,
        terminal=terminal,
        ledger_checksum=ledger_checksum,
        source_checkout_hash=source["source_checkout_hash"],
    )
    if reservation != replayed:
        raise ValueError(
            "P6-D final Reservation differs from authoritative terminal replay"
        )


def _build_formal_report(
    foundation: GenerationSessionFoundation,
    *,
    genesis: VerifiedSessionGenesis,
    ledger: list[dict[str, Any]],
    terminal_binding: Mapping[str, Any],
    generation_cost_receipt: StairGenerationActualCostReceiptRef,
    final_cost_receipt: StairGenerationActualCostReceiptRef,
    reservation: StairFinalConfirmationReservationV2,
    result: StairFinalConfirmationResultV2,
    resource_evidence: Mapping[str, Any],
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    """Recompute the formal report from ledger, receipts, journals, and files."""

    if created_at_utc is None:
        created_at_utc = datetime.now(timezone.utc).isoformat()
    elif not isinstance(created_at_utc, str) or not created_at_utc:
        raise ValueError("P6-D report timestamp is invalid")
    if len(ledger) != STAIR_FINAL_CONFIRMATION_REQUIRED_GENERATIONS:
        raise ValueError("P6-D report requires exactly fifteen generations")
    plan = foundation.final_confirmation_plan
    if not isinstance(plan, StairFinalConfirmationPlanV2):
        raise ValueError("P6-D report final confirmation Plan is missing")
    ledger_path = foundation.config.output_dir / "ledger.jsonl"
    ledger_checksum = file_sha256(ledger_path)
    if ledger_checksum is None or ledger_checksum != reservation.ledger_checksum:
        raise ValueError("P6-D report ledger checksum differs from Reservation")
    if load_ledger(ledger_path) != ledger:
        raise ValueError("P6-D report ledger snapshot changed during replay")
    if dict(terminal_binding) != dict(reservation.terminal_binding):
        raise ValueError("P6-D report terminal binding differs from Reservation")
    if generation_cost_receipt.payload_hash != terminal_binding.get(
        "cost_chain_head_hash"
    ):
        raise ValueError("P6-D report generation cost head mismatch")
    if (
        final_cost_receipt.payload.get("event_id")
        != "final-confirmation-settlement"
        or final_cost_receipt.payload.get("previous_cost_chain_head_hash")
        != generation_cost_receipt.payload_hash
    ):
        raise ValueError("P6-D report final cost chain binding mismatch")
    usage, charges, open_reservations = collect_actual_cost_usage(
        foundation,
        genesis=genesis,
        ledger=ledger,
    )
    assert_actual_cost_receipt_matches_usage(
        final_cost_receipt,
        usage=usage,
        conservative_charges=charges,
        context="P6-D final report",
    )
    if open_reservations:
        raise ValueError("P6-D report found open attempt reservations")
    lifecycle = audit_generation_lifecycle(
        output_dir=foundation.config.output_dir,
        identity=_formal_lifecycle_identity(foundation, genesis=genesis),
        policy=foundation.continuation_policy,
        require_closed=True,
        require_recovery_attempt_results=True,
    )
    if not isinstance(foundation.cost_contract, StairGenerationSessionCostContractV3):
        raise TypeError("P6-D report requires the v3 cost contract")
    cost_receipts = StairGenerationActualCostJournal(
        foundation.config.output_dir, contract=foundation.cost_contract,
    ).load_snapshot()
    formal_cost_head = replay_stair_generation_formal_cost_chain(
        contract=foundation.cost_contract,
        actual_cost_receipts=tuple(cost_receipts.values()),
        lifecycle=lifecycle,
        require_finalization_settlement=True,
    )
    if formal_cost_head != final_cost_receipt.payload_hash:
        raise ValueError("P6-D report final cost head differs from formal replay")
    topology = _formal_lifecycle_topology(foundation)
    generation_details, generation_workloads = _formal_generation_details(
        foundation,
        genesis=genesis,
        ledger=ledger,
        cost_receipts=cost_receipts,
    )
    generation_outcome_counts = Counter(
        item["generation_outcome"] for item in generation_details
    )
    initial_head_hash = ledger[0]["parent_current_best_hash"]
    final_head_hash = ledger[-1]["resulting_current_best_hash"]
    scientific_output = {
        "generation_outcome_counts": dict(sorted(generation_outcome_counts.items())),
        "completed_scientific_generation_count": (
            _completed_scientific_generation_count(generation_outcome_counts)
        ),
        "accepted_generation_count": generation_outcome_counts["accepted"],
        "head_transition_count": sum(
            record["parent_current_best_hash"]
            != record["resulting_current_best_hash"]
            for record in ledger
        ),
        "initial_head_hash": initial_head_hash,
        "final_head_hash": final_head_hash,
        "head_moved_from_session_start": initial_head_hash != final_head_hash,
    }
    validate_stored_resource_closure_evidence(
        foundation,
        resource_evidence=resource_evidence,
        generation_index=STAIR_FINAL_CONFIRMATION_REQUIRED_GENERATIONS,
        include_final_confirmation=True,
    )
    post_genesis = cost_receipts.get("post-genesis")
    post_parent = cost_receipts.get("post-parent-bootstrap")
    if post_genesis is None or post_parent is None:
        raise ValueError("P6-D report lacks bootstrap cost boundaries")
    parent_workload = _validation_workload_delta(
        post_genesis.payload["actual"],
        post_parent.payload["actual"],
    )
    final_workload = _validation_workload_delta(
        generation_cost_receipt.payload["actual"],
        final_cost_receipt.payload["actual"],
    )
    retry_train_jobs, retry_eval_seed_units = session_retry_usage(foundation)
    final_result_path = stair_final_confirmation_result_path(
        foundation.config.output_dir
    )
    result_checksum = file_sha256(final_result_path)
    if result_checksum is None:
        raise ValueError("P6-D final Result checksum is unavailable")
    _validate_formal_final_result_receipts(
        foundation,
        result=result,
    )
    checks = {
        "mode_is_frozen_formal_fifteen_generation": (
            foundation.session_contract.session_mode
            == FORMAL_GENERATION_SESSION_MODE
            and foundation.session_contract.max_generations
            == STAIR_FINAL_CONFIRMATION_REQUIRED_GENERATIONS
        ),
        "ledger_has_fifteen_contiguous_closed_generations": (
            len(generation_details) == STAIR_FINAL_CONFIRMATION_REQUIRED_GENERATIONS
            and [item["generation_index"] for item in generation_details]
            == list(range(1, STAIR_FINAL_CONFIRMATION_REQUIRED_GENERATIONS + 1))
        ),
        "generation_lifecycle_is_terminal": (
            lifecycle.terminal
            and lifecycle.active_segment_id is None
            and lifecycle.closed_generation_count
            == STAIR_FINAL_CONFIRMATION_REQUIRED_GENERATIONS
            and lifecycle.lifecycle_head_hash
            == terminal_binding.get("lifecycle_head_hash")
            and lifecycle.cost_chain_head_hash
            == generation_cost_receipt.payload_hash
        ),
        "normal_segment_topology_uses_precommitted_boundaries_and_ends_at_15": (
            _formal_normal_closed_generation_boundaries_are_valid(
                topology["normal_closed_generation_boundaries"]
            )
        ),
        "final_plan_reservation_result_are_single_bound_chain": (
            reservation.plan_hash == plan.plan_hash
            and result.plan_hash == plan.plan_hash
            and result.reservation_hash == reservation.reservation_hash
        ),
        "final_result_is_diagnostic_only": (
            result.to_dict()["information_flow"]["acceptance_authority"] is False
            and "accepted" not in result.to_dict()
            and "passed" not in result.to_dict()
        ),
        "final_evals_share_precommitted_cohort_and_gpu": (
            result.seed_cohort_hash == plan.seed_cohort_hash
            and result.genesis_execution.resource_binding_hash
            == reservation.gpu_process_binding.resource_binding_hash
            and result.final_execution.resource_binding_hash
            == reservation.gpu_process_binding.resource_binding_hash
            and reservation.gpu_process_binding.process_execution_device == "cuda:0"
        ),
        "final_confirmation_did_not_change_ledger_or_head": (
            ledger_checksum == reservation.ledger_checksum
            and terminal_binding.get("current_head_hash")
            == canonical_json_hash(ledger[-1]["resulting_current_best"])
        ),
        "all_actual_cost_caps_passed": (
            generation_cost_receipt.payload["cap_checks"]["all_passed"] is True
            and final_cost_receipt.payload["cap_checks"]["all_passed"] is True
        ),
        "all_attempts_and_resources_are_closed": (
            not open_reservations
            and resource_evidence.get("attempt_chain", {}).get(
                "open_reservation_count"
            )
            == 0
            and resource_evidence.get("lease_reacquire", {}).get("passed") is True
        ),
        "imported_reference_has_zero_local_genesis_workload": (
            _imported_reference_has_zero_local_genesis_workload(
                genesis,
                post_genesis.payload["actual"],
            )
        ),
    }
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise ValueError("P6-D formal report checks failed: " + ", ".join(failed))
    provider = collect_provider_actuals(foundation, ledger=ledger)
    history = _generation_history_for_ledger(
        foundation, genesis=genesis, ledger=ledger,
    )
    final_exploration = foundation.composition.task_memory_adapter.build_generation_exploration_plan(
        history, generation_session_contract=foundation.session_contract,
    )
    selector_metrics = _validation_selector_reversal_metrics(
        foundation,
        ledger=ledger,
        final_exploration=final_exploration,
    )
    return {
        "schema_version": STAIR_GENERATION_FORMAL_REPORT_SCHEMA_VERSION,
        "created_at_utc": created_at_utc,
        "session_id": foundation.config.session_id,
        "session_mode": FORMAL_GENERATION_SESSION_MODE,
        "source_commit": foundation.source_commit,
        "session_genesis_hash": genesis.session_genesis_hash,
        "reference_genesis_import_ref": _reference_genesis_import_projection(
            genesis
        ),
        "generation_session_contract_hash": foundation.session_contract.contract_hash,
        "generation_cost_contract_hash": foundation.cost_contract.contract_hash,
        "ledger_checksum": ledger_checksum,
        "terminal_binding": dict(terminal_binding),
        "lifecycle": {
            "continuation_policy_hash": lifecycle.continuation_policy_hash,
            "segment_count": lifecycle.segment_count,
            "closed_generation_count": lifecycle.closed_generation_count,
            "lifecycle_head_hash": lifecycle.lifecycle_head_hash,
            "generation_cost_chain_head_hash": lifecycle.cost_chain_head_hash,
            "terminal": lifecycle.terminal,
            "active_segment_id": lifecycle.active_segment_id,
            "receipt_hashes": list(lifecycle.receipt_hashes),
            "recovery_count": len(lifecycle.recovery_cost_edges),
            "recovery_cost_edges": [list(item) for item in lifecycle.recovery_cost_edges],
            "topology": topology,
        },
        "generations": generation_details,
        "scientific_output": scientific_output,
        "cost": {
            "post_genesis": generation_cost_receipt_projection(post_genesis),
            "parent_bootstrap_workload": parent_workload,
            "generation_workloads": generation_workloads,
            "generation_terminal": generation_cost_receipt_projection(
                generation_cost_receipt
            ),
            "final_confirmation_workload": final_workload,
            "final_terminal": generation_cost_receipt_projection(
                final_cost_receipt
            ),
            "provider": provider,
            "retry_train_jobs": retry_train_jobs,
            "retry_eval_seed_units": retry_eval_seed_units,
            "conservative_charges": list(charges),
        },
        "exploration": _validation_exploration_metrics(
            foundation,
            history=history,
            final_plan=final_exploration,
            ledger=ledger,
            selector_metrics=selector_metrics,
        ),
        "authoring": _validation_authoring_metrics(foundation),
        "final_confirmation": {
            "plan": plan.to_dict(),
            "reservation": reservation.to_dict(),
            "result": result.to_dict(),
            "result_checksum": result_checksum,
            "diagnostic_only": True,
            **(
                {"execution_source": dict(_formal_final_execution_source(foundation))}
                if (foundation.config.output_dir / STAIR_FINAL_CONFIRMATION_DIRNAME
                    / _FINAL_SOURCE_CHECKOUT_FILENAME).is_file()
                else {}
            ),
        },
        "resource_evidence": dict(resource_evidence),
        "resource_evidence_hash": canonical_json_hash(resource_evidence),
        "checks": checks,
        "engineering_status": "go",
        "acceptance_authority": False,
    }


def _completed_scientific_generation_count(
    generation_outcome_counts: Mapping[str, int],
) -> int:
    """Count generations that reached a scientific selection decision."""

    return sum(
        generation_outcome_counts.get(outcome, 0)
        for outcome in ("accepted", "formal_rejected", "no_selection")
    )


def _formal_lifecycle_topology(
    foundation: GenerationSessionFoundation,
) -> dict[str, Any]:
    root = foundation.config.output_dir / "generation_segments"
    segments: list[dict[str, Any]] = []
    if not root.is_dir() or root.is_symlink():
        raise ValueError("P6-D lifecycle segment root is invalid")
    for segment_dir in sorted(root.iterdir()):
        if segment_dir.is_symlink() or not segment_dir.is_dir():
            raise ValueError("P6-D lifecycle segment path is invalid")
        start = load_strict_json_object(
            segment_dir / "start.json",
            context="P6-D lifecycle segment start",
        )
        exit_path = segment_dir / "exit.json"
        exit_record = (
            load_strict_json_object(
                exit_path,
                context="P6-D lifecycle segment exit",
            )
            if exit_path.is_file()
            else None
        )
        segments.append(
            {
                "segment_id": start.get("segment_id"),
                "continuation_index": start.get("continuation_index"),
                "generations_at_start": start.get("generations_at_start"),
                "status": (
                    exit_record.get("status")
                    if isinstance(exit_record, Mapping)
                    else "recovered_unclean"
                ),
                "reason": (
                    exit_record.get("reason")
                    if isinstance(exit_record, Mapping)
                    else "unclean_recovery"
                ),
                "generations_at_end": (
                    exit_record.get("generations_at_end")
                    if isinstance(exit_record, Mapping)
                    else start.get("generations_at_start")
                ),
            }
        )
    normal_boundaries = [
        item["generations_at_end"]
        for item in segments
        if (
            item["status"] == "paused" and item["reason"] == "operator_pause"
        )
        or (item["status"] == "completed" and item["reason"] == "normal")
    ]
    return {
        "segments": segments,
        "normal_closed_generation_boundaries": normal_boundaries,
        "normal_segment_count": len(normal_boundaries),
        "recovery_or_failure_segment_count": len(segments) - len(normal_boundaries),
    }


def _formal_normal_closed_generation_boundaries_are_valid(
    boundaries: Sequence[Any],
) -> bool:
    """Accept any explicitly requested formal pauses plus the terminal boundary."""

    normalized = list(boundaries)
    allowed = {
        *STAIR_FORMAL_GENERATION_PAUSE_BOUNDARIES,
        STAIR_FINAL_CONFIRMATION_REQUIRED_GENERATIONS,
    }
    return (
        bool(normalized)
        and normalized[-1] == STAIR_FINAL_CONFIRMATION_REQUIRED_GENERATIONS
        and normalized == sorted(set(normalized))
        and set(normalized).issubset(allowed)
    )


def _validate_formal_final_result_receipts(
    foundation: GenerationSessionFoundation,
    *,
    result: StairFinalConfirmationResultV2,
) -> None:
    for role, execution in (
        ("genesis", result.genesis_execution),
        ("final", result.final_execution),
    ):
        root = (
            foundation.config.output_dir
            / STAIR_FINAL_CONFIRMATION_DIRNAME
            / "phase_receipts"
            / role
        )
        paths = sorted(root.rglob("eval_phases/sealed_validation.json"))
        if len(paths) != 1:
            raise ValueError(f"P6-D {role} phase receipt is not unique")
        ref = trial_receipt_ref(paths[0], kind="eval_phase")
        if ref.checksum != execution.receipt_checksum:
            raise ValueError(f"P6-D {role} phase receipt checksum mismatch")
        loaded = load_trial_eval_phase_receipt(ref)
        phase_result = loaded.result
        summary = canonical_stair_phase_result_summary(
            config=_formal_final_config(foundation),
            expected_phase=EvaluationPhase.SEALED_VALIDATION,
            result=phase_result,
            expected_seed_namespace=STAIR_FINAL_CONFIRMATION_EVAL_NAMESPACE,
        )
        attempt_journal = _formal_final_attempt_journal(
            foundation,
            logical_job_id=execution.job_id,
        )
        if attempt_journal is None:
            raise ValueError(f"P6-D {role} attempt journal is missing")
        attempt_result = attempt_journal.load_reusable_output_result()
        if (
            loaded.job_spec_hash != execution.eval_phase_spec_hash
            or phase_result.job_id != execution.job_id
            or phase_result.evaluation_plan_hash
            != execution.evaluation_plan_hash
            or phase_result.checkpoint_hash != execution.checkpoint_hash
            or phase_result.resource_binding_hash
            != execution.resource_binding_hash
            or canonical_json_hash(summary)
            != execution.generalization_summary_hash
            or attempt_result is None
            or attempt_result.payload_hash
            != execution.attempt_result_payload_hash
            or attempt_result.payload.get("outcome") != "completed"
            or attempt_result.payload.get("output_hash")
            != canonical_json_hash(phase_result.to_private_dict())
        ):
            raise ValueError(f"P6-D {role} phase execution evidence mismatch")


def _formal_generation_details(
    foundation: GenerationSessionFoundation,
    *,
    genesis: VerifiedSessionGenesis,
    ledger: list[dict[str, Any]],
    cost_receipts: Mapping[str, StairGenerationActualCostReceiptRef],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    post_parent = cost_receipts.get("post-parent-bootstrap")
    if post_parent is None:
        raise ValueError("P6-D report has no parent-bootstrap cost boundary")
    previous_actual = post_parent.payload["actual"]
    details: list[dict[str, Any]] = []
    workloads: list[dict[str, Any]] = []
    for generation_index, record in enumerate(ledger, start=1):
        generation_id = (
            f"{foundation.config.session_id}-generation-{generation_index:03d}"
        )
        if (
            record.get("generation_index") != generation_index
            or record.get("generation_id") != generation_id
        ):
            raise ValueError("P6-D generation identities are not contiguous")
        loaded_record, execution_ref = load_existing_generation_execution(
            foundation,
            record=record,
        )
        cost_ref = cost_receipts.get(
            f"generation-{generation_index:03d}-settlement"
        )
        if cost_ref is None:
            raise ValueError("P6-D generation lacks its cost settlement")
        closure = publish_or_validate_generation_closure(
            foundation,
            ledger=ledger,
            generation_record=loaded_record,
            generation_execution_receipt=execution_ref.to_dict(),
            actual_cost_receipt=cost_ref,
            resource_evidence={},
            segment_id="formal-report-replay",
            require_existing=True,
        )
        workload = _validation_workload_delta(
            previous_actual,
            cost_ref.payload["actual"],
        )
        previous_actual = cost_ref.payload["actual"]
        trials = record.get("trials")
        if not isinstance(trials, list):
            raise ValueError("P6-D generation trials are invalid")
        details.append(
            {
                "generation_id": generation_id,
                "generation_index": generation_index,
                "generation_outcome": record.get("generation_outcome"),
                "parent_current_best_hash": record.get(
                    "parent_current_best_hash"
                ),
                "resulting_current_best_hash": record.get(
                    "resulting_current_best_hash"
                ),
                "selection": _json_value(record.get("selection")),
                "formal": _json_value(record.get("formal")),
                "trial_statuses": [
                    {
                        "lane_id": trial.get("lane_id"),
                        "trial_id": trial.get("trial_id"),
                        "trial_status": trial.get("trial_status"),
                        "status_reason": trial.get("status_reason"),
                    }
                    for trial in trials
                    if isinstance(trial, Mapping)
                ],
                "generation_record_hash": canonical_json_hash(record),
                "generation_execution_receipt": execution_ref.to_dict(),
                "generation_closure": closure,
                "actual_workload_delta": workload,
            }
        )
        workloads.append(
            {
                "generation_index": generation_index,
                "cost_receipt": generation_cost_receipt_projection(cost_ref),
                "workload_delta": workload,
            }
        )
    return details, workloads


def _session_genesis_attempt_scope(**kwargs: Any) -> PreGenesisExecutionScope:
    retries = kwargs.pop("max_infrastructure_retries")
    return PreGenesisExecutionScope(
        **kwargs,
        attempt_budget_units=retries + 1,
        max_infrastructure_retries=retries,
    )


def _session_genesis_execution_kwargs(
    foundation: GenerationSessionFoundation,
    *,
    genesis_identity_hash: str,
) -> dict[str, Any]:
    """Bind either fresh reference work or one verified direct import."""

    config = foundation.config
    if config.imports_reference_genesis:
        assert config.source_reference_genesis_dir is not None
        assert config.source_reference_session_id is not None
        return {
            "source_reference_genesis_dir": config.source_reference_genesis_dir,
            "source_reference_session_id": config.source_reference_session_id,
            "reference_phased_evaluation_adapter": foundation.phased_adapter,
        }
    reference_train_scope = _session_genesis_attempt_scope(
        journal_root=config.output_dir / "job_attempts",
        session_id=config.session_id,
        genesis_identity_hash=genesis_identity_hash,
        session_contract_hash=foundation.session_contract.contract_hash,
        lane_id="lane-a",
        invocation_segment_id="session-genesis",
        budget_contract_hash=foundation.cost_contract.contract_hash,
        resource_binding_hash=foundation.gpu_bindings[0].resource_binding_hash,
        max_infrastructure_retries=(
            STAIR_GENERATION_MAX_INFRASTRUCTURE_RETRIES_PER_ELIGIBLE_JOB
        ),
    )
    return {
        "gpu_process_binding": foundation.gpu_bindings[0],
        "reference_train_attempt_scope": reference_train_scope,
        **_session_genesis_reference_evaluation_kwargs(
            foundation,
            genesis_identity_hash=genesis_identity_hash,
        ),
    }


def _session_genesis_reference_evaluation_kwargs(
    foundation: GenerationSessionFoundation,
    *,
    genesis_identity_hash: str,
) -> dict[str, Any]:
    """Choose the task-owned reference evaluation shape for this protocol."""

    config = foundation.config
    common = {
        "journal_root": config.output_dir / "job_attempts",
        "session_id": config.session_id,
        "genesis_identity_hash": genesis_identity_hash,
        "session_contract_hash": foundation.session_contract.contract_hash,
        "lane_id": "lane-a",
        "budget_contract_hash": foundation.cost_contract.contract_hash,
        "resource_binding_hash": (
            foundation.gpu_bindings[0].resource_binding_hash
        ),
        "max_infrastructure_retries": 0,
    }
    if foundation.composition.task_plugin.config.is_unified_protocol:
        return {
            "reference_phased_evaluation_adapter": foundation.phased_adapter,
            "reference_eval_phase_attempt_scopes": {
                phase: _session_genesis_attempt_scope(
                    **common,
                    invocation_segment_id=(
                        "session-genesis-" + phase.value.replace("_", "-")
                    ),
                )
                for phase in EvaluationPhase
            },
        }
    return {
        "reference_eval_attempt_scope": _session_genesis_attempt_scope(
            **common,
            invocation_segment_id="session-genesis",
        )
    }


def _generation_history_for_ledger(
    foundation: GenerationSessionFoundation,
    *,
    genesis: VerifiedSessionGenesis,
    ledger: list[dict[str, Any]],
) -> ProjectedPublicHistory:
    protocol = foundation.composition.task_plugin.task_spec.acceptance_protocol
    if protocol is None:
        raise ValueError("Stair generation requires an acceptance protocol")
    return project_public_history(
        ledger,
        initial_current_best=genesis.initial_current_best,
        expected_session_genesis_hash=genesis.session_genesis_hash,
        acceptance_protocol=protocol,
        expected_initial_policy_state_hash=(genesis.expected_initial_policy_state_hash),
        expected_authoring_mode=PROVIDER_AUTHORING_MODE,
        expected_ledger_epoch_hash=primary_ledger_epoch_hash(
            genesis.session_genesis_hash
        ),
    )


def _provider_policy() -> dict[str, Any]:
    return {
        "schema_version": "true_rl_agent.stair.generation_provider_policy.v2",
        "base_url": STAIR_GENERATION_PROVIDER_BASE_URL,
        "endpoint": STAIR_GENERATION_PROVIDER_ENDPOINT,
        "model": STAIR_GENERATION_PROVIDER_MODEL,
        "reasoning_mode": "adaptive",
        "reasoning_effort": STAIR_GENERATION_PROVIDER_REASONING_EFFORT,
        "empty_final_reasoning_retry": True,
        "adaptive_reasoning_policy_id": (PROVIDER_ADAPTIVE_REASONING_POLICY_ID),
        "thinking_enabled_request": {
            "thinking_type": "enabled",
            "reasoning_effort": STAIR_GENERATION_PROVIDER_REASONING_EFFORT,
            "temperature_present": False,
        },
        "thinking_disabled_repair_request": {
            "thinking_type": "disabled",
            "reasoning_effort_present": False,
            "temperature": (STAIR_GENERATION_PROVIDER_THINKING_DISABLED_TEMPERATURE),
        },
        "response_format": "json_object",
        "stream": False,
        "max_output_tokens": (STAIR_GENERATION_PROVIDER_OUTPUT_UNITS_PER_ATTEMPT),
        "input_units_per_attempt_reserve": (
            STAIR_GENERATION_PROVIDER_INPUT_UNITS_PER_ATTEMPT
        ),
        "timeout_s": STAIR_GENERATION_PROVIDER_TIMEOUT_S,
        "max_attempts_per_lane": (STAIR_GENERATION_PROVIDER_ATTEMPTS_PER_LANE),
        "max_attempts_per_generation": (
            2 * STAIR_GENERATION_PROVIDER_ATTEMPTS_PER_LANE
        ),
    }


def _authoring_policy(
    composition: ResolvedTaskComposition,
) -> dict[str, Any]:
    admission = composition.candidate_admission_policy.admission_policy_descriptor()
    return {
        "schema_version": "true_rl_agent.stair.generation_authoring_policy.v1",
        "authoring_order": list(GENERATION_LANE_IDS),
        "sibling_semantic_exclusion": True,
        "both_candidates_ready_barrier": True,
        "candidate_admission_policy": _json_value(admission.to_public_dict()),
        "authoring_capability_readiness": _json_value(
            composition.authoring_capability_readiness.to_prompt_mapping()
        ),
    }


def _retry_policy(
    cost_contract: StairGenerationSessionCostContractLike,
) -> dict[str, Any]:
    reference_train_retries = (
        STAIR_GENERATION_MAX_INFRASTRUCTURE_RETRIES_PER_ELIGIBLE_JOB
        if cost_contract.retry_train_jobs > 0
        else 0
    )
    common = {
        "logical_job_max_infrastructure_retries": (
            STAIR_GENERATION_MAX_INFRASTRUCTURE_RETRIES_PER_ELIGIBLE_JOB
        ),
        "reference_train_retry_entitlement": reference_train_retries,
        "reference_eval_retry_entitlement": 0,
        "parent_and_public_phase_retry_entitlement": 0,
        "retry_train_job_reserve": cost_contract.retry_train_jobs,
        "retry_eval_seed_unit_reserve": cost_contract.retry_eval_seed_units,
        "candidate_or_contract_failure_retryable": False,
    }
    if isinstance(cost_contract, StairGenerationSessionCostContractV3):
        return {
            "schema_version": "true_rl_agent.stair.generation_retry_policy.v2",
            **common,
            "session_wide_sealed_validation_retry_entitlement": (
                STAIR_GENERATION_MAX_INFRASTRUCTURE_RETRIES_PER_ELIGIBLE_JOB
            ),
            "eligible_sealed_validation_roles": [
                "generation_winner",
                "final_confirmation_genesis",
                "final_confirmation_final",
            ],
        }
    return {
        "schema_version": "true_rl_agent.stair.generation_retry_policy.v1",
        **common,
        "winner_sealed_validation_retry_entitlement": (
            STAIR_GENERATION_MAX_INFRASTRUCTURE_RETRIES_PER_ELIGIBLE_JOB
        ),
    }


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_value(item) for item in value]
    return value


def _validation_workload_delta(
    previous_actual: Mapping[str, Any],
    current_actual: Mapping[str, Any],
) -> dict[str, int | float]:
    workload_fields = (
        "provider_attempts",
        "provider_request_units",
        "provider_response_units",
        "provider_total_units",
        "job_attempts",
        "train_jobs",
        "train_steps",
        "eval_seed_units",
        "simulated_control_steps",
        "resource_lease_wall_duration_ns",
        "conservatively_charged_gpu_capacity_ns",
        "observed_output_bytes",
    )
    workload_delta: dict[str, int | float] = {}
    for field_name in workload_fields:
        current_value = current_actual.get(field_name)
        previous_value = previous_actual.get(field_name)
        if (
            isinstance(current_value, bool)
            or not isinstance(current_value, int | float)
            or isinstance(previous_value, bool)
            or not isinstance(previous_value, int | float)
            or current_value < previous_value
        ):
            raise ValueError("Stair bounded generation workload counter regressed")
        workload_delta[field_name] = current_value - previous_value
    workload_delta["allocated_active_gpu_hours"] = (
        workload_delta["resource_lease_wall_duration_ns"]
        + workload_delta["conservatively_charged_gpu_capacity_ns"]
    ) / 3_600_000_000_000
    return workload_delta


_IMPORTED_REFERENCE_ZERO_LOCAL_WORKLOAD_FIELDS = (
    "job_attempts",
    "train_jobs",
    "train_steps",
    "eval_seed_units",
    "simulated_control_steps",
    "resource_lease_wall_duration_ns",
)


def _reference_genesis_import_projection(
    genesis: VerifiedSessionGenesis,
) -> dict[str, Any] | None:
    ref = genesis.reference_genesis_import_ref
    return ref.to_dict() if ref is not None else None


def _imported_reference_has_zero_local_genesis_workload(
    genesis: VerifiedSessionGenesis,
    actual: Mapping[str, Any],
) -> bool:
    if genesis.reference_genesis_import_ref is None:
        return True
    return all(
        actual.get(field_name) == 0
        for field_name in _IMPORTED_REFERENCE_ZERO_LOCAL_WORKLOAD_FIELDS
    )


def _validation_exploration_metrics(
    foundation: GenerationSessionFoundation,
    *,
    history: ProjectedPublicHistory,
    final_plan: GenerationExplorationPlan,
    ledger: list[dict[str, Any]],
    selector_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    priority_cards = [
        card
        for card in final_plan.index.cards
        if card.revisit_disposition == "priority"
    ]
    eligible_cards = [
        card
        for card in final_plan.index.cards
        if card.revisit_disposition == "eligible"
    ]
    expected_high_losers: set[tuple[str, str]] = set()
    projector = StairExplorationProjector(
        foundation.composition.task_plugin.config,
        objective_family_mask=(
            foundation.composition.task_plugin.objective_family_mask()
        ),
    )
    for history_record in history.records:
        generation = history_record.get("generation")
        if not isinstance(generation, Mapping):
            continue
        selection = generation.get("selection")
        if not isinstance(selection, Mapping):
            continue
        selected_trial_id = selection.get("selected_trial_id")
        parent = CurrentBestEvalStats.from_mapping(
            generation.get("parent_current_best")
        )
        trials = generation.get("trials")
        if not isinstance(trials, list):
            raise TypeError("Stair bounded generation projected generation trials are invalid")
        for trial in trials:
            if (
                not isinstance(trial, Mapping)
                or trial.get("trial_id") == selected_trial_id
            ):
                continue
            assessment = projector.assess_trial(
                generation=generation,
                trial=trial,
                parent=parent,
            )
            if assessment.value_label == "high":
                expected_high_losers.add(
                    (str(generation["generation_id"]), str(trial["trial_id"]))
                )
    observed_high_losers = {
        (card.generation_id, card.trial_id)
        for card in final_plan.index.cards
        if card.value_label == "high" and card.selection_status != "selected"
    }
    service_facts = list(final_plan.index.revisit_service_facts)
    cards_by_hash = {card.card_hash: card for card in final_plan.index.cards}
    priority_served = sum(
        cards_by_hash[item["source_card_hash"]].revisit_disposition == "priority"
        for item in service_facts
    )
    eligible_served = sum(
        cards_by_hash[item["source_card_hash"]].revisit_disposition == "eligible"
        for item in service_facts
    )
    priority_service_failures = sum(
        item.get("public_evidence_obtained") is not True
        and cards_by_hash[item["source_card_hash"]].revisit_disposition == "priority"
        for item in service_facts
    )
    eligible_service_failures = sum(
        item.get("public_evidence_obtained") is not True
        and cards_by_hash[item["source_card_hash"]].revisit_disposition == "eligible"
        for item in service_facts
    )
    priority_expired = sum(
        item.get("reason") == "stale_after_head_transitions"
        and cards_by_hash[item["card_hash"]].revisit_disposition == "priority"
        for item in final_plan.frontier.ineligible_cards
    )
    eligible_expired = sum(
        item.get("reason") == "stale_after_head_transitions"
        and cards_by_hash[item["card_hash"]].revisit_disposition == "eligible"
        for item in final_plan.frontier.ineligible_cards
    )
    priority_deferred = sum(
        cards_by_hash[item["card_hash"]].revisit_disposition == "priority"
        for item in final_plan.frontier.deferred_cards
    )
    eligible_deferred = sum(
        cards_by_hash[item["card_hash"]].revisit_disposition == "eligible"
        for item in final_plan.frontier.deferred_cards
    )
    negative_keys = {
        card.verifier_equivalence_key
        for card in final_plan.index.cards
        if card.observed_effect == "negative"
        and card.verifier_equivalence_key is not None
    }
    trained_negative_keys: Counter[str] = Counter()
    for record in ledger:
        trials = record.get("trials")
        if not isinstance(trials, list):
            raise TypeError("Stair bounded generation exploration trial list is invalid")
        for trial in trials:
            if not isinstance(trial, Mapping) or trial.get("train") is None:
                continue
            candidate = trial.get("candidate")
            if isinstance(candidate, Mapping):
                key = candidate.get("verifier_equivalence_key")
                if isinstance(key, str) and key in negative_keys:
                    trained_negative_keys[key] += 1
    negative_repeat_count = sum(
        max(0, count - 1) for count in trained_negative_keys.values()
    )
    legacy_direction_consistency = [
        card.public_evidence_summary.get("public_dev", {}).get(
            "direction_consistent_with_accept"
        )
        for card in final_plan.index.cards
        if card.public_evidence_summary.get("status") == "public_complete"
        and isinstance(card.public_evidence_summary.get("public_dev"), Mapping)
    ]
    service_denominator = 0
    service_numerator = 0
    unexpected_revisit_service_count = 0
    service_mismatches: list[dict[str, Any]] = []
    for prefix_height in range(len(ledger)):
        plan = foundation.composition.task_memory_adapter.build_generation_exploration_plan(
            history.prefix(prefix_height),
            generation_session_contract=foundation.session_contract,
        )
        queued_by_hash = {card.card_hash: card for card in plan.index.cards}
        expected_priority_hash = next(
            (
                card_hash
                for card_hash in plan.frontier.queued_card_hashes
                if queued_by_hash[card_hash].revisit_disposition == "priority"
            ),
            None,
        )
        next_generation_index = prefix_height + 1
        actual_service_hashes = [
            item["source_card_hash"]
            for item in service_facts
            if item["generation_index"] == next_generation_index
        ]
        if expected_priority_hash is None:
            unexpected_revisit_service_count += len(actual_service_hashes)
            continue
        service_denominator += 1
        if actual_service_hashes == [expected_priority_hash]:
            service_numerator += 1
        else:
            service_mismatches.append(
                {
                    "generation_index": next_generation_index,
                    "expected_priority_card_hash": expected_priority_hash,
                    "actual_revisit_card_hashes": actual_service_hashes,
                }
            )
    return {
        "priority": {
            "arrivals": len(priority_cards),
            "served": priority_served,
            "service_failures": priority_service_failures,
            "expiry": priority_expired,
            "backlog": sum(
                cards_by_hash[card_hash].revisit_disposition == "priority"
                for card_hash in final_plan.frontier.queued_card_hashes
            ),
            "deferred": priority_deferred,
        },
        "eligible": {
            "arrivals": len(eligible_cards),
            "served": eligible_served,
            "service_failures": eligible_service_failures,
            "expiry": eligible_expired,
            "backlog": sum(
                cards_by_hash[card_hash].revisit_disposition == "eligible"
                for card_hash in final_plan.frontier.queued_card_hashes
            ),
            "deferred": eligible_deferred,
        },
        "high_card_recall": {
            "numerator": len(expected_high_losers & observed_high_losers),
            "denominator": len(expected_high_losers),
            "rate": (
                len(expected_high_losers & observed_high_losers)
                / len(expected_high_losers)
                if expected_high_losers
                else 1.0
            ),
        },
        "revisit_lane_service": {
            "numerator": service_numerator,
            "denominator": service_denominator,
            "rate": (
                service_numerator / service_denominator if service_denominator else None
            ),
            "mismatches": service_mismatches,
            "unexpected_revisit_service_count": (unexpected_revisit_service_count),
        },
        "revisit_yield": {
            "numerator": sum(
                item.get("public_evidence_obtained") is True for item in service_facts
            ),
            "denominator": len(service_facts),
        },
        "negative_equivalent_trained_repeat_count": negative_repeat_count,
        "blocked_verifier_equivalent_key_count": len(
            final_plan.blocked_verifier_equivalence_keys
        ),
        "public_dev_accept_consistency": {
            **(
                dict(selector_metrics["public_dev_accept_consistency"])
                if foundation.composition.task_plugin.config.is_unified_protocol
                else {
                    "numerator": sum(
                        value is True for value in legacy_direction_consistency
                    ),
                    "denominator": len(legacy_direction_consistency),
                    "rate": (
                        sum(
                            value is True
                            for value in legacy_direction_consistency
                        )
                        / len(legacy_direction_consistency)
                        if legacy_direction_consistency
                        else None
                    ),
                }
            ),
        },
    }


def _validation_authoring_metrics(
    foundation: GenerationSessionFoundation,
) -> dict[str, Any]:
    sibling_block_count = 0
    identical_block_count = 0
    parameter_only_block_count = 0
    sibling_repair_opportunities = 0
    sibling_repair_successes = 0
    historical_block_count = 0
    historical_advisory_attempts = 0
    historical_advisory_distinct = 0
    timing_by_generation: list[dict[str, Any]] = []
    measured_closed_authoring_wall_duration_ns = 0
    stopped_timing_attempt_count = 0
    incomplete_timing_attempt_count = 0
    prompt_payload_count = 0
    prompt_payload_bytes = 0
    for generation_index in range(1, foundation.session_contract.max_generations + 1):
        generation_id = (
            f"{foundation.config.session_id}-generation-{generation_index:03d}"
        )
        timing = replay_stair_generation_authoring_timing(
            evidence_root=(
                foundation.config.output_dir / STAIR_GENERATION_AUTHORING_TIMING_DIRNAME
            ),
            generation_id=generation_id,
        )
        generation_measured_closed_ns = timing[
            "measured_closed_attempt_wall_duration_ns"
        ]
        generation_total_ns = timing["total_wall_duration_ns"]
        stopped_count = len(timing["stopped_attempts"])
        incomplete_count = len(timing["incomplete_attempts"])
        measured_closed_authoring_wall_duration_ns += generation_measured_closed_ns
        stopped_timing_attempt_count += stopped_count
        incomplete_timing_attempt_count += incomplete_count
        timing_by_generation.append(
            {
                "generation_id": generation_id,
                "lane_measured_closed_attempt_wall_duration_ns": dict(
                    timing["lane_measured_closed_attempt_wall_duration_ns"]
                ),
                "measured_closed_attempt_wall_duration_ns": (
                    generation_measured_closed_ns
                ),
                "total_wall_duration_ns": generation_total_ns,
                "timing_complete": timing["timing_complete"],
                "stopped_attempt_count": stopped_count,
                "incomplete_attempt_count": incomplete_count,
            }
        )
        for lane_id in GENERATION_LANE_IDS:
            round_dir = (
                foundation.config.output_dir / "authoring" / generation_id / lane_id
            )
            if not round_dir.exists():
                continue
            snapshot = RoundAuthoringJournal(
                round_dir,
                round_id=f"{generation_id}-{lane_id}",
            ).snapshot()
            prompt_payload_count += len(snapshot.prompt_payload_refs)
            prompt_payload_bytes += sum(
                ref.size_bytes for ref in snapshot.prompt_payload_refs
            )
            lane_had_sibling_block = False
            lane_repaired_after_block = False
            for attempt in snapshot.attempts:
                sibling = attempt.get("sibling_semantic_exclusion_assessment")
                if isinstance(sibling, Mapping):
                    if sibling.get("blocking") is True:
                        sibling_block_count += 1
                        lane_had_sibling_block = True
                        equivalence = sibling.get("equivalence")
                        if equivalence == "identical_or_format_only":
                            identical_block_count += 1
                        elif equivalence == "parameter_only_delta":
                            parameter_only_block_count += 1
                    elif lane_had_sibling_block and sibling.get("status") == "pass":
                        lane_repaired_after_block = True
                historical = attempt.get("historical_verifier_exclusion_assessment")
                if isinstance(historical, Mapping):
                    historical_advisory_attempts += 1
                    if historical.get("blocking") is True:
                        historical_block_count += 1
                    elif historical.get("status") == "pass":
                        historical_advisory_distinct += 1
            if lane_had_sibling_block:
                sibling_repair_opportunities += 1
                if lane_repaired_after_block:
                    sibling_repair_successes += 1
    return {
        "sibling_exclusion_block_count": sibling_block_count,
        "same_delta_block_count": identical_block_count,
        "semantic_equivalent_delta_block_count": parameter_only_block_count,
        "sibling_exclusion_repair": {
            "numerator": sibling_repair_successes,
            "denominator": sibling_repair_opportunities,
            "rate": (
                sibling_repair_successes / sibling_repair_opportunities
                if sibling_repair_opportunities
                else None
            ),
        },
        "blocked_historical_verifier_equivalent_attempt_count": (
            historical_block_count
        ),
        "negative_advisory_distinct_hit_rate": {
            "numerator": historical_advisory_distinct,
            "denominator": historical_advisory_attempts,
            "rate": (
                historical_advisory_distinct / historical_advisory_attempts
                if historical_advisory_attempts
                else None
            ),
        },
        "timing": {
            "clock": "process_monotonic_ns",
            "timing_complete": incomplete_timing_attempt_count == 0,
            "measured_closed_attempt_wall_duration_ns": (
                measured_closed_authoring_wall_duration_ns
            ),
            "total_wall_duration_ns": (
                measured_closed_authoring_wall_duration_ns
                if incomplete_timing_attempt_count == 0
                else None
            ),
            "total_wall_duration_seconds": (
                measured_closed_authoring_wall_duration_ns / 1_000_000_000
                if incomplete_timing_attempt_count == 0
                else None
            ),
            "stopped_attempt_count": stopped_timing_attempt_count,
            "incomplete_attempt_count": incomplete_timing_attempt_count,
            "generations": timing_by_generation,
        },
        "prompt_payloads": {
            "source": "all_round_authoring_journal_attempts",
            "count": prompt_payload_count,
            "total_bytes": prompt_payload_bytes,
        },
    }


def _validation_selector_reversal_metrics(
    foundation: GenerationSessionFoundation,
    *,
    ledger: list[dict[str, Any]],
    final_exploration: GenerationExplorationPlan,
) -> dict[str, Any]:
    if foundation.composition.task_plugin.config.is_unified_protocol:
        return _unified_validation_selector_reversal_metrics(
            foundation,
            ledger=ledger,
        )

    cards_by_generation: dict[int, list[Any]] = {}
    for card in final_exploration.index.cards:
        cards_by_generation.setdefault(card.generation_index, []).append(card)
    complete_pair_count = 0
    reversals: list[dict[str, Any]] = []
    for generation_index, cards in sorted(cards_by_generation.items()):
        selected = [card for card in cards if card.selection_status == "selected"]
        losers = [
            card for card in cards if card.selection_status == "qualified_not_selected"
        ]
        if len(selected) != 1 or len(losers) != 1:
            continue
        winner = selected[0]
        loser = losers[0]
        winner_summary = winner.public_evidence_summary
        loser_summary = loser.public_evidence_summary
        if (
            winner_summary.get("status") != "public_complete"
            or loser_summary.get("status") != "public_complete"
        ):
            continue
        winner_objective = winner_summary.get("objective")
        loser_objective = loser_summary.get("objective")
        winner_guard = winner_summary.get("guard")
        loser_guard = loser_summary.get("guard")
        if not all(
            isinstance(item, Mapping)
            for item in (
                winner_objective,
                loser_objective,
                winner_guard,
                loser_guard,
            )
        ):
            raise ValueError("Stair bounded generation selector public comparison is invalid")
        assert isinstance(winner_objective, Mapping)
        assert isinstance(loser_objective, Mapping)
        assert isinstance(winner_guard, Mapping)
        assert isinstance(loser_guard, Mapping)
        numeric_values = (
            winner_objective.get("candidate"),
            loser_objective.get("candidate"),
            winner_objective.get("materiality_threshold"),
            loser_objective.get("materiality_threshold"),
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int | float)
            for value in numeric_values
        ):
            raise ValueError("Stair bounded generation selector objective comparison is invalid")
        winner_score, loser_score, winner_threshold, loser_threshold = (
            float(value) for value in numeric_values
        )
        complete_pair_count += 1
        loser_advantage = loser_score - winner_score
        material_reversal = loser_advantage >= max(
            winner_threshold,
            loser_threshold,
        )
        winner_gate = (
            winner_objective.get("materiality_met") is True
            and winner_guard.get("guard_met") is True
        )
        loser_gate = (
            loser_objective.get("materiality_met") is True
            and loser_guard.get("guard_met") is True
        )
        gate_reversal = not winner_gate and loser_gate
        if material_reversal or gate_reversal:
            reversals.append(
                {
                    "generation_index": generation_index,
                    "selected_trial_id": winner.trial_id,
                    "loser_trial_id": loser.trial_id,
                    "selected_objective": winner_score,
                    "loser_objective": loser_score,
                    "loser_advantage": loser_advantage,
                    "material_reversal": material_reversal,
                    "public_gate_reversal": gate_reversal,
                }
            )
    return {
        "complete_dual_public_generation_count": complete_pair_count,
        "observed_reversal_count": len(reversals),
        "blocking_reversal_count": (len(reversals) if complete_pair_count >= 2 else 0),
        "reversals": reversals,
    }


def _unified_validation_selector_reversal_metrics(
    foundation: GenerationSessionFoundation,
    *,
    ledger: list[dict[str, Any]],
) -> dict[str, Any]:
    """Verify that public-accept is routed only to a public-dev winner."""

    _ = foundation
    violations: list[dict[str, Any]] = []
    selected_confirmation_count = 0
    no_selection_count = 0
    for record in ledger:
        selection = record.get("selection")
        trials = record.get("trials")
        if not isinstance(selection, Mapping) or not isinstance(trials, list):
            continue
        selected_trial_id = selection.get("selected_trial_id")
        confirmed_trial_ids = [
            str(trial.get("trial_id"))
            for trial in trials
            if isinstance(trial, Mapping)
            and trial.get("public_accept") is not None
        ]
        if selected_trial_id is None:
            no_selection_count += 1
            valid = not confirmed_trial_ids
        else:
            valid = confirmed_trial_ids in ([], [str(selected_trial_id)])
            if confirmed_trial_ids == [str(selected_trial_id)]:
                selected_confirmation_count += 1
        if not valid:
            violations.append(
                {
                    "generation_index": record.get("generation_index"),
                    "selected_trial_id": selected_trial_id,
                    "public_accept_trial_ids": confirmed_trial_ids,
                }
            )
    return {
        "status": "pass" if not violations else "fail",
        "comparison_source": "selected_lane_only_public_accept_routing",
        "selected_confirmation_generation_count": selected_confirmation_count,
        "no_selection_generation_count": no_selection_count,
        "routing_violation_count": len(violations),
        "complete_dual_public_generation_count": 0,
        "observed_reversal_count": 0,
        "blocking_reversal_count": 0,
        "public_dev_accept_consistency": {
            "status": "not_collected_by_design",
            "numerator": 0,
            "denominator": 0,
            "rate": None,
        },
        "reversals": [],
        "routing_violations": violations,
    }


def _load_and_validate_existing_formal_report(
    foundation: GenerationSessionFoundation,
    *,
    genesis: VerifiedSessionGenesis,
    cost_journal: StairGenerationActualCostJournal,
) -> dict[str, Any]:
    """Read-only replay of the completed P6-D report and every bound source."""

    path = foundation.config.output_dir / STAIR_GENERATION_REPORT_FILENAME
    report = load_strict_json_object(path, context="P6-D generation session report")
    if path.read_bytes() != canonical_json_bytes(report, trailing_newline=True):
        raise ValueError("P6-D generation report is not canonical JSON")
    if report.get("schema_version") != STAIR_GENERATION_FORMAL_REPORT_SCHEMA_VERSION:
        raise ValueError("P6-D generation report schema mismatch")
    created_at_utc = report.get("created_at_utc")
    if not isinstance(created_at_utc, str) or not created_at_utc:
        raise ValueError("P6-D generation report timestamp is invalid")
    ledger = load_ledger(foundation.config.output_dir / "ledger.jsonl")
    if len(ledger) != STAIR_FINAL_CONFIRMATION_REQUIRED_GENERATIONS:
        raise ValueError("P6-D generation report ledger height mismatch")
    plan = foundation.final_confirmation_plan
    if not isinstance(plan, StairFinalConfirmationPlanV2):
        raise ValueError("P6-D generation report Plan is missing")
    reservation = load_stair_generation_final_confirmation_reservation(
        stair_final_confirmation_reservation_path(foundation.config.output_dir),
        expected_plan=plan,
    )
    result = load_stair_generation_final_confirmation_result(
        stair_final_confirmation_result_path(foundation.config.output_dir),
        plan=plan,
        reservation=reservation,
        config=_formal_final_config(foundation),
        objective_family_mask=(
            foundation.composition.task_plugin.objective_family_mask()
        ),
    )
    generation_cost = cost_receipt_by_hash(
        cost_journal,
        str(reservation.terminal_binding["cost_chain_head_hash"]),
    )
    final_cost = cost_journal.receipt_for_event("final-confirmation-settlement")
    if final_cost is None:
        raise ValueError("P6-D generation report final cost receipt is missing")
    resource_evidence = report.get("resource_evidence")
    if not isinstance(resource_evidence, Mapping):
        raise ValueError("P6-D generation report resource evidence is invalid")
    expected = _build_formal_report(
        foundation,
        genesis=genesis,
        ledger=ledger,
        terminal_binding=reservation.terminal_binding,
        generation_cost_receipt=generation_cost,
        final_cost_receipt=final_cost,
        reservation=reservation,
        result=result,
        resource_evidence=resource_evidence,
        created_at_utc=created_at_utc,
    )
    if report != expected:
        raise ValueError("P6-D generation report differs from recomputation")
    _validate_formal_final_reservation_against_terminal_replay(
        foundation,
        genesis=genesis,
        ledger=ledger,
        reservation=reservation,
    )
    return report


def _verify_source_checkout(
    project_root: Path,
    *,
    expected_commit: str,
) -> str:
    checkout = require_stair_source_checkout(
        project_root,
        expected_commit=expected_commit,
    )
    return checkout.head_commit


def load_completed_stair_generation_session(
    *,
    output_dir: Path,
    expected_session_id: str,
    expected_cell_variant_id: str,
    expected_source_commit: str,
) -> CompletedStairGenerationSession:
    """Replay one completed P6-C session without creating or publishing files.

    This deliberately does not call the live/resume entrypoint or construct a
    ``GenerationSessionFoundation``. It only loads already-published typed
    contracts and replays their identity, lifecycle, receipt and cost chains.
    The terminal report is returned only after identity checks; higher-level
    acceptance, scoring and report-claim replay remain with the verifier.
    """

    if not isinstance(output_dir, Path):
        raise TypeError("completed Stair generation output_dir must be a Path")
    if not output_dir.is_absolute():
        raise ValueError("completed Stair generation output_dir must be absolute")
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise FileNotFoundError(
            "completed Stair generation output_dir must be an existing regular directory"
        )
    resolved_output_dir = output_dir.resolve(strict=True)
    if (
        not isinstance(expected_session_id, str)
        or not expected_session_id
        or expected_session_id != expected_session_id.strip()
    ):
        raise ValueError("completed Stair generation expected_session_id is invalid")
    if (
        not isinstance(expected_cell_variant_id, str)
        or not expected_cell_variant_id
        or expected_cell_variant_id != expected_cell_variant_id.strip()
    ):
        raise ValueError(
            "completed Stair generation expected_cell_variant_id is invalid"
        )
    if (
        not isinstance(expected_source_commit, str)
        or _SOURCE_COMMIT.fullmatch(expected_source_commit) is None
    ):
        raise ValueError("completed Stair generation expected source commit is invalid")

    required_files = (
        STAIR_GENERATION_SOURCE_CHECKOUT_FILENAME,
        GENERATION_SESSION_CONTRACT_FILENAME,
        STAIR_GENERATION_SESSION_COST_FILENAME,
        GENERATION_CONTINUATION_POLICY_FILENAME,
        RESOLVED_CELL_MANIFEST_FILENAME,
        SESSION_GENESIS_FILENAME,
        "ledger.jsonl",
        STAIR_GENERATION_REPORT_FILENAME,
    )
    missing = [
        name
        for name in required_files
        if not _completed_stair_regular_file(resolved_output_dir / name)
    ]
    if missing:
        raise FileNotFoundError(
            "completed Stair generation session is missing required artifacts: "
            + ", ".join(missing)
        )

    source_checkout = _load_completed_stair_source_checkout(
        resolved_output_dir,
        expected_source_commit=expected_source_commit,
    )
    report = _load_completed_stair_canonical_json(
        resolved_output_dir / STAIR_GENERATION_REPORT_FILENAME,
        context="completed Stair generation validation report",
    )
    if (
        report.get("schema_version")
        != STAIR_GENERATION_VALIDATION_REPORT_SCHEMA_VERSION
        or report.get("session_mode")
        != DEVELOPMENT_GENERATION_VALIDATION_MODE
        or report.get("session_id") != expected_session_id
        or report.get("source_commit") != expected_source_commit
        or report.get("output_dir") != str(resolved_output_dir)
        or report.get("acceptance_authority") is not False
    ):
        raise ValueError("completed Stair generation report identity mismatch")

    manifest_path = (resolved_output_dir / RESOLVED_CELL_MANIFEST_FILENAME).resolve(
        strict=True
    )
    resolved_cell_manifest = load_resolved_cell_manifest(
        manifest_path,
        expected_task_id=STAIR_TASK_ID,
    )
    if resolved_cell_manifest.cell_variant_id != expected_cell_variant_id:
        raise ValueError("completed Stair generation robot row mismatch")

    genesis_payload = _load_completed_stair_canonical_json(
        resolved_output_dir / SESSION_GENESIS_FILENAME,
        context="completed Stair generation Session Genesis",
    )
    session_genesis = SessionGenesisArtifact.from_mapping(genesis_payload)
    if (
        session_genesis.session_id != expected_session_id
        or session_genesis.task_id != STAIR_TASK_ID
        or session_genesis.session_genesis_hash
        != report.get("session_genesis_hash")
        or session_genesis.evaluation_binding.task_protocol_hash
        != resolved_cell_manifest.resolved_cell_task_protocol_hash
    ):
        raise ValueError("completed Stair generation Session Genesis identity mismatch")
    contract_ref = session_genesis.generation_session_contract_ref
    if contract_ref is None:
        raise ValueError(
            "completed Stair generation Session Genesis has no generation contract"
        )
    session_contract = load_generation_session_contract(
        resolved_output_dir,
        contract_ref,
    )
    if (
        session_contract.schema_version
        != GENERATION_SESSION_CONTRACT_V3_SCHEMA_VERSION
        or session_contract.session_mode
        != DEVELOPMENT_GENERATION_VALIDATION_MODE
        or session_contract.session_id != expected_session_id
        or session_contract.task_id != STAIR_TASK_ID
        or session_contract.backend_id != session_genesis.backend_id
        or session_contract.max_generations != 3
        or session_contract.bound_contract_hashes.get("evaluation_protocol")
        != session_genesis.evaluation_binding.eval_protocol_hash
        or report.get("generation_session_contract_hash")
        != session_contract.contract_hash
    ):
        raise ValueError("completed Stair generation session contract mismatch")

    cost_contract = load_stair_generation_session_cost_contract(
        resolved_output_dir / STAIR_GENERATION_SESSION_COST_FILENAME
    )
    if (
        type(cost_contract)
        not in {
            StairGenerationSessionCostContractV2,
            StairGenerationSessionCostContractV4,
        }
        or cost_contract.session_mode
        != DEVELOPMENT_GENERATION_VALIDATION_MODE
        or cost_contract.max_generations != 3
        or (
            type(cost_contract) is StairGenerationSessionCostContractV4
            and session_genesis.reference_genesis_import_ref is None
        )
        or session_contract.bound_contract_hashes.get("session_cost")
        != cost_contract.contract_hash
        or session_contract.bound_contract_hashes.get("retry_policy")
        != canonical_json_hash(_retry_policy(cost_contract))
        or report.get("session_cost_contract_hash") != cost_contract.contract_hash
    ):
        raise ValueError("completed Stair generation cost contract mismatch")
    continuation_policy = load_generation_continuation_policy(resolved_output_dir)
    if (
        session_contract.bound_contract_hashes.get("continuation_policy")
        != continuation_policy.policy_hash
    ):
        raise ValueError("completed Stair generation continuation policy mismatch")

    ledger = load_ledger(resolved_output_dir / "ledger.jsonl")
    if len(ledger) != session_contract.max_generations:
        raise ValueError("completed Stair generation ledger height mismatch")
    ledger_epoch_hash = primary_ledger_epoch_hash(
        str(session_genesis.session_genesis_hash)
    )
    execution_receipts: list[GenerationReceiptRef] = []
    for generation_index, record in enumerate(ledger, start=1):
        generation_id = f"{expected_session_id}-generation-{generation_index:03d}"
        if (
            record.get("schema_version") != session_contract.ledger_record_schema
            or record.get("generation_index") != generation_index
            or record.get("generation_id") != generation_id
            or record.get("session_genesis_hash")
            != session_genesis.session_genesis_hash
            or record.get("generation_session_contract_hash")
            != session_contract.contract_hash
            or record.get("ledger_epoch_hash") != ledger_epoch_hash
        ):
            raise ValueError("completed Stair generation ledger identity mismatch")
        generation_manifest = load_bound_generation_manifest(record)
        if (
            generation_manifest.generation_id != generation_id
            or generation_manifest.generation_index != generation_index
            or generation_manifest.session_genesis_hash
            != session_genesis.session_genesis_hash
            or generation_manifest.generation_session_contract_hash
            != session_contract.contract_hash
            or generation_manifest.ledger_epoch_hash != ledger_epoch_hash
            or generation_manifest.evaluation_binding_hash
            != session_genesis.evaluation_binding.binding_hash
            or generation_resource_plan_hash(generation_manifest.lanes)
            != session_contract.bound_contract_hashes.get("resource_plan")
        ):
            raise ValueError(
                "completed Stair generation record/manifest identity mismatch"
            )
        journal = GenerationReceiptJournal(
            resolved_output_dir / "generation_receipts" / generation_id,
            manifest=generation_manifest,
        )
        loaded_execution = journal.load_execution()
        if loaded_execution is None or loaded_execution[0] != record:
            raise ValueError(
                "completed Stair generation execution receipt is missing or drifted"
            )
        execution_ref = loaded_execution[1]
        execution_receipts.append(execution_ref)

    lifecycle_identity = GenerationLifecycleIdentity(
        session_id=expected_session_id,
        session_genesis_hash=str(session_genesis.session_genesis_hash),
        generation_session_contract_hash=session_contract.contract_hash,
        final_confirmation_plan_hash=session_contract.bound_contract_hashes[
            "final_confirmation_plan"
        ],
        ledger_epoch_hash=ledger_epoch_hash,
        output_root=str(resolved_output_dir),
    )
    lifecycle = audit_generation_lifecycle(
        output_dir=resolved_output_dir,
        identity=lifecycle_identity,
        policy=continuation_policy,
        require_closed=True,
    )
    if (
        not lifecycle.terminal
        or lifecycle.active_segment_id is not None
        or lifecycle.closed_generation_count != session_contract.max_generations
    ):
        raise ValueError("completed Stair generation lifecycle is not terminal")

    cost_journal = StairGenerationActualCostJournal(
        resolved_output_dir,
        contract=cost_contract,
    )
    actual_cost_receipts = cost_journal.load_chain()
    if not actual_cost_receipts:
        raise ValueError("completed Stair generation actual-cost chain is missing")
    combined_cost_head = replay_stair_generation_combined_cost_chain(
        contract=cost_contract,
        actual_cost_receipts=actual_cost_receipts,
        lifecycle=lifecycle,
    )
    if combined_cost_head != lifecycle.cost_chain_head_hash:
        raise ValueError("completed Stair generation actual-cost head mismatch")

    return CompletedStairGenerationSession(
        output_dir=resolved_output_dir,
        source_checkout=dict(source_checkout),
        resolved_cell_manifest=resolved_cell_manifest,
        session_genesis=session_genesis,
        session_contract=session_contract,
        cost_contract=cost_contract,
        continuation_policy=continuation_policy,
        ledger=tuple(dict(record) for record in ledger),
        lifecycle=lifecycle,
        actual_cost_receipts=tuple(actual_cost_receipts),
        generation_execution_receipts=tuple(execution_receipts),
        report=dict(report),
    )


def _completed_stair_regular_file(path: Path) -> bool:
    return not path.is_symlink() and path.is_file()


def _load_completed_stair_canonical_json(
    path: Path,
    *,
    context: str,
) -> dict[str, Any]:
    payload = load_strict_json_object(path, context=context)
    if path.read_bytes() != canonical_json_bytes(payload, trailing_newline=True):
        raise ValueError(f"{context} is not canonical JSON")
    return payload


def _load_completed_stair_source_checkout(
    output_dir: Path,
    *,
    expected_source_commit: str,
) -> dict[str, Any]:
    payload = _load_completed_stair_canonical_json(
        output_dir / STAIR_GENERATION_SOURCE_CHECKOUT_FILENAME,
        context="completed Stair generation source checkout",
    )
    body = {
        "schema_version": STAIR_GENERATION_SOURCE_CHECKOUT_SCHEMA_VERSION,
        "source_commit": expected_source_commit,
    }
    if (
        set(payload) != {*body, "source_checkout_hash"}
        or payload.get("schema_version")
        != STAIR_GENERATION_SOURCE_CHECKOUT_SCHEMA_VERSION
        or payload.get("source_commit") != expected_source_commit
        or payload.get("source_checkout_hash") != canonical_json_hash(body)
    ):
        raise ValueError("completed Stair generation source checkout mismatch")
    return payload


def audit_completed_stair_formal_session(config):
    """Recompute an existing 15-generation report without runner/provider work."""
    if not config.resume_existing or not config.terminal_audit_only:
        raise ValueError("historical report auditing requires a completed read-only session")
    if config.session_mode != FORMAL_GENERATION_SESSION_MODE:
        raise ValueError("formal history reader requires the frozen formal session mode")
    report_path = config.output_dir / STAIR_GENERATION_REPORT_FILENAME
    if not report_path.is_file():
        raise FileNotFoundError("historical audit requires the completed formal report")
    source_commit = _verify_source_checkout(config.project_root, expected_commit=config.expected_source_commit)
    source_commit = _formal_resume_generation_source(config, source_commit)
    foundation = _load_foundation(config, source_commit=source_commit,
                                    inventory=(config.gpu0_uuid, config.gpu1_uuid))
    genesis = _load_existing_session_genesis_for_terminal_audit(foundation)
    journal = StairGenerationActualCostJournal(config.output_dir, contract=foundation.cost_contract)
    return _load_and_validate_existing_formal_report(foundation, genesis=genesis, cost_journal=journal)


__all__ = ["HistoricalStairGenerationConfig", "CompletedStairGenerationSession",
           "load_completed_stair_generation_session", "audit_completed_stair_formal_session"]
