"""Production Stair entrypoint for one fixed-width development generation.

The module is deliberately task-owned.  It assembles the existing Genesis,
authoring, generation, exploration, lifecycle, runner, and ledger contracts;
it does not implement another scientific state machine.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import tempfile
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from true_rl_agent.agent_core.authoring_loop import CandidateAuthoringLoop
from true_rl_agent.agent_core.exploration import GenerationExplorationPlan
from true_rl_agent.agent_core.generation import (
    FixedWidthGenerationInputs,
    GenerationLaneRuntime,
    execute_fixed_width_generation,
)
from true_rl_agent.agent_core.generation_contract import (
    GENERATION_LANE_IDS,
    GENERATION_MANIFEST_V2_SCHEMA_VERSION,
    GENERATION_SESSION_CONTRACT_FILENAME,
    GENERATION_SESSION_CONTRACT_V3_SCHEMA_VERSION,
    GenerationLaneAssignment,
    GenerationManifest,
    GenerationSessionContract,
    GenerationSessionContractRef,
    generation_resource_plan_hash,
    generation_training_initialization_hash,
    publish_generation_session_contract,
)
from true_rl_agent.agent_core.generation_lifecycle import (
    GENERATION_CONTINUATION_POLICY_FILENAME,
    GENERATION_CONTINUATION_POLICY_SCHEMA_VERSION,
    GENERATION_CONTINUATION_POLICY_V2_SCHEMA_VERSION,
    GenerationContinuationPolicy,
    GenerationLifecycleAudit,
    GenerationLifecycleIdentity,
    GenerationSegmentJournal,
    GenerationTerminalBinding,
    audit_generation_lifecycle,
    build_generation_continuation_cursor,
    build_generation_terminal_binding,
    publish_generation_continuation_policy,
)
from true_rl_agent.agent_core.generation_record import (
    load_bound_generation_manifest,
    load_bound_generation_trial_train_receipt,
)
from true_rl_agent.agent_core.generation_receipts import (
    GenerationReceiptJournal,
    load_generation_manifest_from_receipt,
)
from true_rl_agent.agent_core.ledger import (
    PROVIDER_AUTHORING_MODE,
    load_ledger,
    primary_ledger_epoch_hash,
)
from true_rl_agent.agent_core.openai_compatible_provider import (
    OpenAICompatibleCandidateProvider,
    OpenAICompatibleProviderConfig,
)
from true_rl_agent.agent_core.provider_limits import (
    PROVIDER_ADAPTIVE_REASONING_POLICY_ID,
)
from true_rl_agent.agent_core.round import RoundInputs
from true_rl_agent.agent_core.session_genesis import (
    REFERENCE_EVAL_ARTIFACT_FILENAME,
    SESSION_GENESIS_FILENAME,
    VerifiedSessionGenesis,
    prepare_session_genesis,
    session_genesis_identity_hash,
)
from true_rl_agent.agent_core.trial_receipts import (
    TRIAL_EXECUTION_CONTRACT_SCHEMA_VERSION,
    TrialReceiptJournal,
    TrialReceiptRef,
    load_trial_eval_phase_receipt,
    load_trial_train_receipt,
)
from true_rl_agent.composition import (
    ResolvedTaskComposition,
    TaskCompositionContext,
    resolve_task_composition,
)
from true_rl_agent.evaluation.acceptance_replay import project_public_history
from true_rl_agent.evaluation.binding import (
    EvaluationBindingV1,
    canonical_json_hash,
    require_sha256_digest,
)
from true_rl_agent.evaluation.current_best import CurrentBestEvalStats
from true_rl_agent.evaluation.phases import EvaluationPhase, EvaluationPhaseResult
from true_rl_agent.fs_contract import create_once_bytes, read_bounded_regular_file
from true_rl_agent.json_contract import (
    canonical_json_bytes,
    load_strict_json_object,
    strict_json_object_loads,
)
from true_rl_agent.memory import ExperimentRecord, PublicContextBuilder
from true_rl_agent.prompt_artifacts import (
    AuthoringEvidenceRef,
    PromptPayloadRef,
    RoundAuthoringJournal,
    verify_authoring_evidence,
    verify_prompt_payload,
)
from true_rl_agent.runner.gpu_lease import (
    GpuDeviceIdentity,
    GpuLeaseHolder,
    GpuLeaseManager,
)
from true_rl_agent.runner.job_attempts import (
    RESOURCE_LEASE_ATTEMPT_RESULT_SCHEMA_VERSION,
    JobAttemptJournal,
    JobAttemptScope,
    PreGenesisExecutionScope,
)
from true_rl_agent.runner.checkpoint import file_sha256
from true_rl_agent.runner.specs import (
    CheckpointRef,
    EvalJobSpec,
    EvalPhaseJobSpec,
    GpuProcessBinding,
)
from true_rl_agent.runner.subprocess_runner import SubprocessRunner
from true_rl_agent.tasks.stair_climbing.composition import (
    StairClimbingCompositionProvider,
)
from true_rl_agent.tasks.stair_climbing.exploration import (
    StairExplorationProjector,
)
from true_rl_agent.tasks.stair_climbing.generation_bootstrap import (
    replay_stair_generation_guard_chain,
    run_genesis_parent_public_dev,
)
from true_rl_agent.tasks.stair_climbing.generation_attempt_scopes import (
    StairGenerationLaneOperationScopes,
    recover_stair_generation_lane_operation_scopes,
    recover_stair_parent_bootstrap_attempt_scope,
)
from true_rl_agent.tasks.stair_climbing.generation_authoring_timing import (
    STAIR_GENERATION_AUTHORING_TIMING_DIRNAME,
    replay_stair_generation_authoring_timing,
    run_stair_generation_authoring_attempt,
)
from true_rl_agent.tasks.stair_climbing.generation_session_cost import (
    DEVELOPMENT_GENERATION_PILOT_MODE,
    DEVELOPMENT_GENERATION_VALIDATION_MODE,
    FORMAL_GENERATION_SESSION_MODE,
    STAIR_GENERATION_FORMAL_RETRY_ALLOCATED_ACTIVE_GPU_HOURS,
    STAIR_GENERATION_FORMAL_RETRY_OUTPUT_BYTES,
    STAIR_GENERATION_FORMAL_SUCCESSFUL_ALLOCATED_ACTIVE_GPU_HOURS,
    STAIR_GENERATION_FORMAL_SUCCESSFUL_OUTPUT_BYTES,
    STAIR_FORMAL_GENERATION_PAUSE_BOUNDARIES,
    STAIR_GENERATION_FINALIZATION_OUTPUT_RESERVE_BYTES,
    STAIR_GENERATION_MAX_INFRASTRUCTURE_RETRIES_PER_ELIGIBLE_JOB,
    STAIR_GENERATION_PROVIDER_ATTEMPTS_PER_LANE,
    STAIR_GENERATION_PROVIDER_INPUT_UNITS_PER_ATTEMPT,
    STAIR_GENERATION_PROVIDER_OUTPUT_UNITS_PER_ATTEMPT,
    STAIR_GENERATION_RETRY_EVAL_SEED_UNITS,
    STAIR_GENERATION_RETRY_TRAIN_JOBS,
    STAIR_GENERATION_SESSION_COST_FILENAME,
    STAIR_GENERATION_VALIDATION_RETRY_ALLOCATED_ACTIVE_GPU_HOURS,
    STAIR_GENERATION_VALIDATION_RETRY_OUTPUT_BYTES,
    STAIR_GENERATION_VALIDATION_SUCCESSFUL_ALLOCATED_ACTIVE_GPU_HOURS,
    STAIR_GENERATION_VALIDATION_SUCCESSFUL_OUTPUT_BYTES,
    StairGenerationActualCostJournal,
    StairGenerationActualCostReceiptRef,
    StairGenerationActualCostUsage,
    StairGenerationActualCostUsageV2,
    StairGenerationOperationCostReserve,
    StairGenerationSessionCostContract,
    StairGenerationSessionCostContractLike,
    StairGenerationSessionCostContractV2,
    StairGenerationSessionCostContractV3,
    build_stair_generation_scientific_cost_basis,
    create_stair_generation_session_cost_contract,
    load_stair_generation_session_cost_contract,
    replay_stair_generation_combined_cost_chain,
    replay_stair_generation_formal_cost_chain,
    stair_formal_generation_operation_cost_reserves,
    stair_generation_operation_cost_reserves,
)
from true_rl_agent.tasks.stair_climbing.final_confirmation import (
    STAIR_FINAL_CONFIRMATION_DIRNAME,
    STAIR_FINAL_CONFIRMATION_PLAN_FILENAME,
    STAIR_FINAL_CONFIRMATION_REQUIRED_GENERATIONS,
    STAIR_GENERATION_REPORT_FILENAME,
    StairFinalConfirmationPhaseExecution,
    StairFinalConfirmationPlanV2,
    StairFinalConfirmationReservationV2,
    StairFinalConfirmationResultV2,
    build_stair_generation_final_confirmation_result,
    create_stair_generation_final_confirmation_plan,
    create_stair_generation_final_confirmation_reservation,
    load_stair_generation_final_confirmation_plan,
    load_stair_generation_final_confirmation_reservation,
    load_stair_generation_final_confirmation_result,
    stair_final_confirmation_plan_path,
    stair_final_confirmation_reservation_path,
    stair_final_confirmation_result_path,
    write_stair_generation_final_confirmation_result,
)
from true_rl_agent.tasks.stair_climbing.gpu_process_binding import (
    validate_stair_gpu_runtime_summary_evidence,
)
from true_rl_agent.tasks.stair_climbing.phased_adapter import (
    StairPhasedEvaluationAdapter,
)
from true_rl_agent.tasks.stair_climbing.phased_evaluation import (
    canonical_stair_phase_result_summary,
)
from true_rl_agent.tasks.stair_climbing.entrypoint_contract import (
    STAIR_EVAL_PURPOSE_FINAL_CONFIRMATION,
)
from true_rl_agent.tasks.stair_climbing.seed_policy import (
    STAIR_FINAL_CONFIRMATION_EVAL_NAMESPACE,
)
from true_rl_agent.tasks.stair_climbing.provider_readiness import (
    STAIR_PROVIDER_READINESS_SCHEMA_VERSION,
    require_stair_provider_readiness,
    validate_stair_provider_credential,
)
from true_rl_agent.tasks.stair_climbing.source_checkout import (
    require_stair_source_checkout,
)

STAIR_GENERATION_REPORT_SCHEMA_VERSION = (
    "true_rl_agent.stair.generation_session_report.v2"
)
STAIR_GENERATION_VALIDATION_REPORT_SCHEMA_VERSION = (
    "true_rl_agent.stair.generation_session_report.v3"
)
STAIR_GENERATION_FORMAL_REPORT_SCHEMA_VERSION = (
    "true_rl_agent.stair.generation_session_report.v6"
)
STAIR_GENERATION_PAUSE_RESULT_SCHEMA_VERSION = (
    "true_rl_agent.stair.generation_pause_result.v1"
)
STAIR_FORMAL_GENERATION_PAUSE_RESULT_SCHEMA_VERSION = (
    "true_rl_agent.stair.generation_pause_result.v2"
)
STAIR_GENERATION_CLOSURE_SCHEMA_VERSION = "true_rl_agent.stair.generation_closure.v1"
STAIR_GENERATION_PREFLIGHT_SCHEMA_VERSION = (
    "true_rl_agent.stair.generation_preflight.v1"
)
STAIR_GENERATION_SOURCE_CHECKOUT_SCHEMA_VERSION = (
    "true_rl_agent.stair.generation_source_checkout.v1"
)
STAIR_GENERATION_SOURCE_CHECKOUT_FILENAME = "generation_source_checkout.json"
STAIR_GENERATION_PROVIDER_TIMEOUT_S = 300
STAIR_GENERATION_PROVIDER_MODEL = "deepseek-v4-flash"
STAIR_GENERATION_PROVIDER_REASONING_EFFORT = "max"
STAIR_GENERATION_PROVIDER_BASE_URL = "https://api.deepseek.com"
STAIR_GENERATION_PROVIDER_ENDPOINT = "/chat/completions"
STAIR_GENERATION_PROVIDER_API_KEY_ENV = "DEEPSEEK_API_KEY"
STAIR_GENERATION_PROVIDER_THINKING_DISABLED_TEMPERATURE = 0.2
_ALLOCATED_GPU_CAPACITY_TIME_BASIS = (
    "sum_of_bound_operation_resource_lease_wall_duration_ns;"
    "not_utilization_weighted;excludes_coordination_only_lease_time"
)
_FULL_GPU_UUID = re.compile(
    r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\Z"
)
_SOURCE_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_PARENT_GPU_REMAP_KEYS = ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES")
_P6B_TERMINAL_REPORT_MAX_BYTES = 4 * 1024 * 1024
_GENERATION_SESSION_CONTRACT_MAX_BYTES = 128 * 1024


@dataclass(frozen=True)
class StairGenerationPilotConfig:
    """Operator-supplied physical limits; scientific semantics stay fixed."""

    project_root: Path
    output_dir: Path
    lease_root: Path
    session_id: str
    expected_source_commit: str
    machine_id: str
    gpu0_uuid: str
    gpu1_uuid: str
    provider_env_file: Path
    max_allocated_active_gpu_hours: float
    max_simulated_control_steps: int
    max_output_bytes: int
    nvidia_smi_executable: str = "nvidia-smi"
    resume_existing: bool = False
    session_mode: str = DEVELOPMENT_GENERATION_PILOT_MODE
    pause_after_generation: int | None = None
    terminal_audit_only: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "project_root",
            "output_dir",
            "lease_root",
            "provider_env_file",
        ):
            if not isinstance(getattr(self, field_name), Path):
                raise TypeError(f"{field_name} must be a Path")
        if not self.output_dir.is_absolute() or not self.lease_root.is_absolute():
            raise ValueError("generation output_dir and lease_root must be absolute")
        if not isinstance(self.session_id, str) or not self.session_id:
            raise ValueError("generation session_id is required")
        if _SOURCE_COMMIT.fullmatch(self.expected_source_commit) is None:
            raise ValueError("expected_source_commit must be a lowercase Git SHA")
        if not isinstance(self.machine_id, str) or not self.machine_id.strip():
            raise ValueError("generation machine_id is required")
        for field_name in ("gpu0_uuid", "gpu1_uuid"):
            if _FULL_GPU_UUID.fullmatch(getattr(self, field_name)) is None:
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


@dataclass(frozen=True)
class StairGenerationSessionFoundation:
    config: StairGenerationPilotConfig
    source_commit: str
    composition: ResolvedTaskComposition
    runner: SubprocessRunner
    phased_adapter: StairPhasedEvaluationAdapter
    continuation_policy: GenerationContinuationPolicy
    cost_contract: StairGenerationSessionCostContractLike
    session_contract: GenerationSessionContract
    session_contract_ref: GenerationSessionContractRef
    gpu_identities: tuple[GpuDeviceIdentity, GpuDeviceIdentity]
    gpu_bindings: tuple[GpuProcessBinding, GpuProcessBinding]
    gpu_lease_manager: GpuLeaseManager
    provider_policy: Mapping[str, Any]
    final_confirmation_policy: Mapping[str, Any]
    final_confirmation_plan: StairFinalConfirmationPlanV2 | None


@dataclass(frozen=True)
class _StairCostGuardedProvider:
    """Apply the frozen session provider cap immediately before transport."""

    delegate: OpenAICompatibleCandidateProvider
    output_dir: Path
    generation_id: str
    lane_id: str
    max_generations: int = 1
    allow_conservative_interrupted_recovery: bool = False

    def next_authoring_response(
        self,
        *,
        prompt_payload: dict[str, Any],
    ) -> Any:
        _assert_provider_request_budget(
            output_dir=self.output_dir,
            generation_id=self.generation_id,
            current_lane_id=self.lane_id,
            current_prompt_payload=prompt_payload,
            max_generations=self.max_generations,
            allow_conservative_interrupted_recovery=(
                self.allow_conservative_interrupted_recovery
            ),
        )
        return self.delegate.next_authoring_response(prompt_payload=prompt_payload)

    def next_candidate(
        self,
        *,
        prompt_payload: dict[str, Any],
    ) -> Any:
        _assert_provider_request_budget(
            output_dir=self.output_dir,
            generation_id=self.generation_id,
            current_lane_id=self.lane_id,
            current_prompt_payload=prompt_payload,
            max_generations=self.max_generations,
            allow_conservative_interrupted_recovery=(
                self.allow_conservative_interrupted_recovery
            ),
        )
        return self.delegate.next_candidate(prompt_payload=prompt_payload)


def preflight_stair_generation_pilot(
    config: StairGenerationPilotConfig,
) -> dict[str, Any]:
    """Validate the production session boundary without starting the session."""

    _reject_parent_gpu_remap()
    if config.resume_existing:
        raise ValueError("preflight-only cannot resume an existing session")
    if config.output_dir.exists() or config.output_dir.is_symlink():
        raise FileExistsError(
            "preflight-only requires a nonexistent generation output_dir"
        )

    source_commit = _verify_source_checkout(
        config.project_root,
        expected_commit=config.expected_source_commit,
    )
    inventory = discover_two_gpu_inventory(
        config.nvidia_smi_executable,
        expected=(config.gpu0_uuid, config.gpu1_uuid),
    )
    _assert_target_gpus_compute_idle(
        config.nvidia_smi_executable,
        expected=(config.gpu0_uuid, config.gpu1_uuid),
        stage="preflight-only",
    )
    if not config.provider_env_file.is_file():
        raise FileNotFoundError("provider env file does not exist")
    _preflight_provider_credential(config.provider_env_file)

    with tempfile.TemporaryDirectory(
        prefix="true-rl-agent-stair-preflight-"
    ) as temporary:
        scratch_root = Path(temporary).resolve()
        scratch_output = scratch_root / "foundation"
        if config.session_mode == DEVELOPMENT_GENERATION_VALIDATION_MODE:
            _publish_or_verify_source_checkout_identity(
                scratch_output,
                source_commit=source_commit,
            )
        foundation = _create_foundation(
            replace(
                config,
                output_dir=scratch_output,
                lease_root=scratch_root / "leases",
                resume_existing=False,
            ),
            source_commit=source_commit,
            inventory=inventory,
        )
        session_genesis_identity_hash(
            foundation.composition.genesis_identity_snapshot()
        )
        persistence_checks = _exercise_preflight_persistence_contracts(
            foundation,
            scratch_root=scratch_root / "persistence",
        )

    if config.output_dir.exists() or config.output_dir.is_symlink():
        raise RuntimeError("preflight-only created or raced with formal output")

    checks = {
        "formal_output_absent_before_and_after": True,
        "source_gpu_and_credential_preflight_read_only": True,
        "scratch_foundation_create_once_verified": True,
        "genesis_identity_snapshot_hashable": True,
        **persistence_checks,
    }
    return {
        "schema_version": STAIR_GENERATION_PREFLIGHT_SCHEMA_VERSION,
        "mode": "preflight-only",
        "checks": checks,
        "side_effects": {
            "formal_output_created": False,
            "gpu_lease_or_runner_job_started": False,
            "provider_request_sent": False,
            "attempt_ledger_genesis_or_cost_reservation_written": False,
            "scratch_retained": False,
        },
    }


def preflight_stair_generation_provider_readiness(
    config: StairGenerationPilotConfig,
) -> dict[str, Any]:
    """Check the fixed provider account without creating formal session state."""

    _reject_parent_gpu_remap()
    if config.resume_existing:
        raise ValueError("provider-readiness-only cannot resume an existing session")
    if config.output_dir.exists() or config.output_dir.is_symlink():
        raise FileExistsError(
            "provider-readiness-only requires a nonexistent generation output_dir"
        )
    source_commit = _verify_source_checkout(
        config.project_root,
        expected_commit=config.expected_source_commit,
    )
    readiness = _require_stair_generation_provider_readiness(config)
    if config.output_dir.exists() or config.output_dir.is_symlink():
        raise RuntimeError(
            "provider-readiness-only created or raced with formal output"
        )
    return {
        "schema_version": STAIR_PROVIDER_READINESS_SCHEMA_VERSION,
        "mode": "provider-readiness-only",
        "session_id": config.session_id,
        "source_commit": source_commit,
        **readiness,
        "checks": {
            "source_checkout_matches_expected_commit": True,
            "formal_output_absent_before_and_after": True,
            **readiness["checks"],
        },
        "side_effects": {
            **readiness["side_effects"],
            "formal_output_created": False,
            "gpu_lease_or_runner_job_started": False,
            "attempt_ledger_genesis_or_cost_reservation_written": False,
        },
    }


def run_stair_generation_pilot(
    config: StairGenerationPilotConfig,
) -> dict[str, Any]:
    """Run or explicitly resume one no-final-confirmation Stair session."""

    _reject_parent_gpu_remap()
    output_exists = config.output_dir.exists()
    if output_exists and not config.resume_existing:
        raise FileExistsError(
            "generation output_dir exists; pass resume_existing explicitly"
        )
    if not output_exists and config.resume_existing:
        raise FileNotFoundError("generation resume output_dir does not exist")
    if config.resume_existing:
        required_foundation = (
            STAIR_GENERATION_SESSION_COST_FILENAME,
            GENERATION_SESSION_CONTRACT_FILENAME,
            GENERATION_CONTINUATION_POLICY_FILENAME,
        )
        if config.session_mode in {
            DEVELOPMENT_GENERATION_VALIDATION_MODE,
            FORMAL_GENERATION_SESSION_MODE,
        }:
            required_foundation = (
                STAIR_GENERATION_SOURCE_CHECKOUT_FILENAME,
                *required_foundation,
            )
        if config.session_mode == FORMAL_GENERATION_SESSION_MODE:
            required_foundation = (
                *required_foundation,
                str(
                    Path(STAIR_FINAL_CONFIRMATION_DIRNAME)
                    / STAIR_FINAL_CONFIRMATION_PLAN_FILENAME
                ),
            )
        missing = [
            name
            for name in required_foundation
            if not (config.output_dir / name).is_file()
        ]
        if missing:
            raise ValueError(
                "generation resume output lacks its create-once foundation: "
                + ", ".join(missing)
            )
    source_commit = _verify_source_checkout(
        config.project_root,
        expected_commit=config.expected_source_commit,
    )
    final_report_path = config.output_dir / STAIR_GENERATION_REPORT_FILENAME
    if (
        config.resume_existing
        and config.session_mode
        in {
            DEVELOPMENT_GENERATION_VALIDATION_MODE,
            FORMAL_GENERATION_SESSION_MODE,
        }
    ):
        _publish_or_verify_source_checkout_identity(
            config.output_dir,
            source_commit=source_commit,
        )
    terminal_validation_report = (
        config.session_mode == DEVELOPMENT_GENERATION_VALIDATION_MODE
        and final_report_path.is_file()
    )
    if (
        config.session_mode == FORMAL_GENERATION_SESSION_MODE
        and final_report_path.is_file()
        and not config.terminal_audit_only
    ):
        raise ValueError(
            "completed P6-D session only permits terminal_audit_only"
        )
    if terminal_validation_report or config.terminal_audit_only:
        foundation = _create_foundation(
            config,
            source_commit=source_commit,
            inventory=(config.gpu0_uuid, config.gpu1_uuid),
        )
        genesis = _load_existing_session_genesis_for_terminal_audit(foundation)
        cost_journal = StairGenerationActualCostJournal(
            config.output_dir,
            contract=foundation.cost_contract,
        )
        if config.session_mode == FORMAL_GENERATION_SESSION_MODE:
            if not final_report_path.is_file():
                raise FileNotFoundError(
                    "P6-D terminal audit requires the completed formal report"
                )
            report = _load_and_validate_existing_formal_report(
                foundation,
                genesis=genesis,
                cost_journal=cost_journal,
            )
        elif terminal_validation_report:
            report = _load_and_validate_existing_validation_report(
                foundation,
                genesis=genesis,
                cost_journal=cost_journal,
                startup_resource_observation=None,
            )
        else:
            report = _finalize_missing_terminal_validation_report(
                foundation,
                genesis=genesis,
                cost_journal=cost_journal,
            )
        if any(key in os.environ for key in _PARENT_GPU_REMAP_KEYS):
            raise RuntimeError(
                "generation orchestration changed the parent GPU environment"
            )
        return report
    if _generation_provider_readiness_required(config):
        _require_stair_generation_provider_readiness(config)
    if (
        not config.resume_existing
        and config.session_mode
        in {
            DEVELOPMENT_GENERATION_VALIDATION_MODE,
            FORMAL_GENERATION_SESSION_MODE,
        }
    ):
        _publish_or_verify_source_checkout_identity(
            config.output_dir,
            source_commit=source_commit,
        )
    inventory = discover_two_gpu_inventory(
        config.nvidia_smi_executable,
        expected=(config.gpu0_uuid, config.gpu1_uuid),
    )
    foundation = _create_foundation(
        config,
        source_commit=source_commit,
        inventory=inventory,
    )
    startup_resource_observation = _assert_target_gpus_compute_idle(
        config.nvidia_smi_executable,
        expected=(config.gpu0_uuid, config.gpu1_uuid),
        stage="startup",
    )
    cost_journal = StairGenerationActualCostJournal(
        config.output_dir,
        contract=foundation.cost_contract,
    )
    cost_journal.publish_reservation()
    genesis_identity_snapshot = foundation.composition.genesis_identity_snapshot()
    genesis_identity_hash = session_genesis_identity_hash(genesis_identity_snapshot)
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
    reference_eval_scope = _session_genesis_attempt_scope(
        journal_root=config.output_dir / "job_attempts",
        session_id=config.session_id,
        genesis_identity_hash=genesis_identity_hash,
        session_contract_hash=foundation.session_contract.contract_hash,
        lane_id="lane-a",
        invocation_segment_id="session-genesis",
        budget_contract_hash=foundation.cost_contract.contract_hash,
        resource_binding_hash=foundation.gpu_bindings[0].resource_binding_hash,
        max_infrastructure_retries=0,
    )
    genesis_exists = (config.output_dir / SESSION_GENESIS_FILENAME).exists()
    if final_report_path.exists() and not genesis_exists:
        raise ValueError("generation report exists without Session Genesis")
    if not genesis_exists:
        if not config.provider_env_file.is_file():
            raise FileNotFoundError("provider env file does not exist")
        _preflight_provider_credential(config.provider_env_file)
    try:
        genesis = prepare_session_genesis(
            output_dir=config.output_dir,
            session_id=config.session_id,
            task_plugin=foundation.composition.task_plugin,
            backend_adapter=foundation.composition.backend_adapter,
            runner=foundation.runner,
            policy_initialization=foundation.composition.policy_initialization,
            genesis_identity_snapshot=genesis_identity_snapshot,
            baseline_manifest=foundation.composition.baseline_manifest,
            reference_program_artifact=(
                foundation.composition.reference_program_artifact
            ),
            bootstrap_initial_best_score=(
                foundation.composition.bootstrap_initial_best_score
            ),
            bootstrap_initial_policy_state_hash=(
                foundation.composition.bootstrap_initial_policy_state_hash
            ),
            bind_backend_execution_environment=True,
            generation_session_contract_ref=foundation.session_contract_ref,
            gpu_process_binding=foundation.gpu_bindings[0],
            reference_train_attempt_scope=reference_train_scope,
            reference_eval_attempt_scope=reference_eval_scope,
            require_existing=final_report_path.exists(),
        )
    except BaseException:
        usage, charges, _ = _collect_actual_cost_usage(
            foundation,
            genesis=None,
            ledger=[],
        )
        cost_journal.publish_snapshot(
            event_id=(f"genesis-failed-{len(cost_journal.load_chain()):04d}"),
            previous_cost_chain_head_hash=_latest_cost_head(cost_journal),
            usage=usage,
            conservative_charges=charges,
            finalization_output_reserve_bytes=0,
        )
        raise
    if final_report_path.exists():
        if config.session_mode == DEVELOPMENT_GENERATION_VALIDATION_MODE:
            report = _load_and_validate_existing_validation_report(
                foundation,
                genesis=genesis,
                cost_journal=cost_journal,
                startup_resource_observation=startup_resource_observation,
            )
        elif config.session_mode == FORMAL_GENERATION_SESSION_MODE:
            report = _load_and_validate_existing_formal_report(
                foundation,
                genesis=genesis,
                cost_journal=cost_journal,
            )
        else:
            report = _load_and_validate_existing_report(
                foundation,
                genesis=genesis,
                cost_journal=cost_journal,
                startup_resource_observation=startup_resource_observation,
            )
        if any(key in os.environ for key in _PARENT_GPU_REMAP_KEYS):
            raise RuntimeError(
                "generation orchestration changed the parent GPU environment"
            )
        return report
    if not config.provider_env_file.is_file():
        raise FileNotFoundError("provider env file does not exist")
    _preflight_provider_credential(config.provider_env_file)
    post_genesis = cost_journal.receipt_for_event("post-genesis")
    if post_genesis is None:
        post_genesis_usage, post_genesis_charges, _ = _collect_actual_cost_usage(
            foundation,
            genesis=genesis,
            ledger=[],
        )
        post_genesis = cost_journal.publish_snapshot(
            event_id="post-genesis",
            previous_cost_chain_head_hash=_latest_cost_head(cost_journal),
            usage=post_genesis_usage,
            conservative_charges=post_genesis_charges,
            finalization_output_reserve_bytes=0,
        )
    if post_genesis.payload["cap_checks"]["all_passed"] is not True:
        raise ValueError("Session Genesis exhausted the Stair cost contract")
    if config.session_mode == DEVELOPMENT_GENERATION_VALIDATION_MODE:
        report = _run_validation_session(
            foundation,
            genesis=genesis,
            cost_journal=cost_journal,
            startup_resource_observation=startup_resource_observation,
        )
    elif config.session_mode == FORMAL_GENERATION_SESSION_MODE:
        report = _run_formal_session(
            foundation,
            genesis=genesis,
            cost_journal=cost_journal,
            startup_resource_observation=startup_resource_observation,
        )
    else:
        report = _run_one_generation(
            foundation,
            genesis=genesis,
            cost_journal=cost_journal,
            startup_resource_observation=startup_resource_observation,
        )
    if any(key in os.environ for key in _PARENT_GPU_REMAP_KEYS):
        raise RuntimeError(
            "generation orchestration changed the parent GPU environment"
        )
    if report.get("schema_version") not in {
        STAIR_GENERATION_PAUSE_RESULT_SCHEMA_VERSION,
        STAIR_FORMAL_GENERATION_PAUSE_RESULT_SCHEMA_VERSION,
    }:
        path = config.output_dir / STAIR_GENERATION_REPORT_FILENAME
        create_once_bytes(
            path,
            canonical_json_bytes(report, trailing_newline=True),
        )
    return report


def _load_existing_session_genesis_for_terminal_audit(
    foundation: StairGenerationSessionFoundation,
) -> VerifiedSessionGenesis:
    config = foundation.config
    snapshot = foundation.composition.genesis_identity_snapshot()
    identity_hash = session_genesis_identity_hash(snapshot)
    reference_train_scope = _session_genesis_attempt_scope(
        journal_root=config.output_dir / "job_attempts",
        session_id=config.session_id,
        genesis_identity_hash=identity_hash,
        session_contract_hash=foundation.session_contract.contract_hash,
        lane_id="lane-a",
        invocation_segment_id="session-genesis",
        budget_contract_hash=foundation.cost_contract.contract_hash,
        resource_binding_hash=foundation.gpu_bindings[0].resource_binding_hash,
        max_infrastructure_retries=(
            STAIR_GENERATION_MAX_INFRASTRUCTURE_RETRIES_PER_ELIGIBLE_JOB
        ),
    )
    reference_eval_scope = _session_genesis_attempt_scope(
        journal_root=config.output_dir / "job_attempts",
        session_id=config.session_id,
        genesis_identity_hash=identity_hash,
        session_contract_hash=foundation.session_contract.contract_hash,
        lane_id="lane-a",
        invocation_segment_id="session-genesis",
        budget_contract_hash=foundation.cost_contract.contract_hash,
        resource_binding_hash=foundation.gpu_bindings[0].resource_binding_hash,
        max_infrastructure_retries=0,
    )
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
        gpu_process_binding=foundation.gpu_bindings[0],
        reference_train_attempt_scope=reference_train_scope,
        reference_eval_attempt_scope=reference_eval_scope,
        require_existing=True,
    )


def discover_two_gpu_inventory(
    executable: str,
    *,
    expected: tuple[str, str],
) -> tuple[str, str]:
    """Read the trusted host inventory and require exactly the frozen two GPUs."""

    completed = subprocess.run(
        [
            executable,
            "--query-gpu=index,uuid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    discovered: dict[int, str] = {}
    for raw_line in completed.stdout.splitlines():
        if not raw_line.strip():
            continue
        parts = [part.strip() for part in raw_line.split(",")]
        if len(parts) != 2 or not parts[0].isdigit():
            raise ValueError("nvidia-smi returned an invalid GPU inventory row")
        index = int(parts[0])
        uuid = parts[1]
        if index in discovered or _FULL_GPU_UUID.fullmatch(uuid) is None:
            raise ValueError("nvidia-smi returned an invalid GPU identity")
        discovered[index] = uuid
    if discovered != {0: expected[0], 1: expected[1]}:
        raise ValueError("host GPU inventory differs from the frozen two-GPU plan")
    return expected


def _assert_target_gpus_compute_idle(
    executable: str,
    *,
    expected: tuple[str, str],
    stage: str,
) -> dict[str, Any]:
    if stage not in {"preflight-only", "startup", "final"}:
        raise ValueError("GPU compute-idle stage is invalid")
    completed = subprocess.run(
        [
            executable,
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    rows: list[dict[str, Any]] = []
    for raw_line in completed.stdout.splitlines():
        if not raw_line.strip():
            continue
        parts = [part.strip() for part in raw_line.split(",")]
        if len(parts) != 2 or _FULL_GPU_UUID.fullmatch(parts[0]) is None:
            raise ValueError("nvidia-smi returned an invalid compute-app row")
        try:
            process_id = int(parts[1])
        except ValueError as exc:
            raise ValueError(
                "nvidia-smi returned a non-numeric compute-app PID"
            ) from exc
        if process_id <= 0:
            raise ValueError("nvidia-smi returned an invalid compute-app PID")
        rows.append({"gpu_uuid": parts[0], "process_id": process_id})
    target_rows = [item for item in rows if item["gpu_uuid"] in expected]
    if target_rows:
        raise ValueError(f"target GPUs are not compute-idle at {stage}")
    return {
        "schema_version": "true_rl_agent.stair.generation_compute_idle.v1",
        "stage": stage,
        "source": "nvidia-smi-query-compute-apps",
        "target_gpu_uuids": list(expected),
        "rows": target_rows,
    }


def _create_foundation(
    config: StairGenerationPilotConfig,
    *,
    source_commit: str,
    inventory: tuple[str, str],
) -> StairGenerationSessionFoundation:
    provider = StairClimbingCompositionProvider()
    values = dict(provider.cli_defaults())
    formal_final_plan: StairFinalConfirmationPlanV2 | None = None
    final_plan_path: Path | None = None
    if config.session_mode == FORMAL_GENERATION_SESSION_MODE:
        provisional_composition = resolve_task_composition(
            provider=provider,
            values=values,
            context=TaskCompositionContext(
                project_root=config.project_root,
                output_dir=config.output_dir,
            ),
        )
        final_plan_path = stair_final_confirmation_plan_path(config.output_dir)
        if final_plan_path.is_file():
            formal_final_plan = load_stair_generation_final_confirmation_plan(
                final_plan_path,
                expected_session_id=config.session_id,
                expected_config=provisional_composition.task_plugin.config,
            )
        else:
            if config.resume_existing:
                raise FileNotFoundError(
                    "formal generation resume lacks final confirmation plan"
                )
            formal_final_plan, final_plan_path = (
                create_stair_generation_final_confirmation_plan(
                    output_dir=config.output_dir,
                    session_id=config.session_id,
                    config=provisional_composition.task_plugin.config,
                )
            )
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
        ),
    )
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
    phased_adapter = StairPhasedEvaluationAdapter(composition.task_plugin.config)
    scientific_cost_basis = build_stair_generation_scientific_cost_basis(
        session_id=config.session_id,
        config=composition.task_plugin.config,
    )
    if (
        scientific_cost_basis.phased_evaluation_cost_contract_hash
        != phased_adapter.cost_contract()["contract_hash"]
    ):
        raise ValueError("Stair phased evaluation cost source drifted")
    if config.session_mode == DEVELOPMENT_GENERATION_VALIDATION_MODE:
        cost: StairGenerationSessionCostContractLike = (
            StairGenerationSessionCostContractV2(
                session_mode=DEVELOPMENT_GENERATION_VALIDATION_MODE,
                max_generations=3,
                scientific_cost_basis=scientific_cost_basis,
                retry_train_jobs=STAIR_GENERATION_RETRY_TRAIN_JOBS,
                retry_eval_seed_units=STAIR_GENERATION_RETRY_EVAL_SEED_UNITS,
                successful_allocated_active_gpu_hours=(
                    STAIR_GENERATION_VALIDATION_SUCCESSFUL_ALLOCATED_ACTIVE_GPU_HOURS
                ),
                retry_allocated_active_gpu_hours=(
                    STAIR_GENERATION_VALIDATION_RETRY_ALLOCATED_ACTIVE_GPU_HOURS
                ),
                successful_output_bytes=(
                    STAIR_GENERATION_VALIDATION_SUCCESSFUL_OUTPUT_BYTES
                ),
                retry_output_bytes=STAIR_GENERATION_VALIDATION_RETRY_OUTPUT_BYTES,
            )
        )
        if (
            float(config.max_allocated_active_gpu_hours)
            != cost.max_allocated_active_gpu_hours
            or config.max_simulated_control_steps != cost.max_simulated_control_steps
            or config.max_output_bytes != cost.max_output_bytes
        ):
            raise ValueError(
                "Stair bounded generation operator caps must exactly match the frozen physical cost contract"
            )
    elif config.session_mode == FORMAL_GENERATION_SESSION_MODE:
        cost = StairGenerationSessionCostContractV3(
            session_mode=FORMAL_GENERATION_SESSION_MODE,
            max_generations=STAIR_FINAL_CONFIRMATION_REQUIRED_GENERATIONS,
            scientific_cost_basis=scientific_cost_basis,
            retry_train_jobs=STAIR_GENERATION_RETRY_TRAIN_JOBS,
            retry_eval_seed_units=STAIR_GENERATION_RETRY_EVAL_SEED_UNITS,
            successful_allocated_active_gpu_hours=(
                STAIR_GENERATION_FORMAL_SUCCESSFUL_ALLOCATED_ACTIVE_GPU_HOURS
            ),
            retry_allocated_active_gpu_hours=(
                STAIR_GENERATION_FORMAL_RETRY_ALLOCATED_ACTIVE_GPU_HOURS
            ),
            successful_output_bytes=(
                STAIR_GENERATION_FORMAL_SUCCESSFUL_OUTPUT_BYTES
            ),
            retry_output_bytes=STAIR_GENERATION_FORMAL_RETRY_OUTPUT_BYTES,
        )
        if (
            float(config.max_allocated_active_gpu_hours)
            != cost.max_allocated_active_gpu_hours
            or config.max_simulated_control_steps != cost.max_simulated_control_steps
            or config.max_output_bytes != cost.max_output_bytes
        ):
            raise ValueError(
                "P6-D operator caps must exactly match the frozen physical cost contract"
            )
    else:
        cost = StairGenerationSessionCostContract(
            session_mode=DEVELOPMENT_GENERATION_PILOT_MODE,
            max_generations=1,
            scientific_cost_basis=scientific_cost_basis,
            retry_train_jobs=STAIR_GENERATION_RETRY_TRAIN_JOBS,
            retry_eval_seed_units=STAIR_GENERATION_RETRY_EVAL_SEED_UNITS,
            max_allocated_active_gpu_hours=(config.max_allocated_active_gpu_hours),
            max_simulated_control_steps=config.max_simulated_control_steps,
            max_output_bytes=config.max_output_bytes,
        )
    if cost.physical_train_step_cap > config.max_simulated_control_steps:
        raise ValueError(
            "simulated-control-step cap cannot be below the physical train-step cap"
        )
    cost_path = config.output_dir / STAIR_GENERATION_SESSION_COST_FILENAME
    if cost_path.exists():
        if load_stair_generation_session_cost_contract(cost_path) != cost:
            raise ValueError("existing Stair session cost contract differs")
    else:
        create_stair_generation_session_cost_contract(config.output_dir, cost)
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
    publish_generation_continuation_policy(config.output_dir, continuation)
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
            "retry_policy": canonical_json_hash(_retry_policy()),
            "selector": canonical_json_hash(phased_adapter.selection_contract()),
            "session_cost": cost.contract_hash,
        },
    )
    session_contract, session_contract_ref = (
        _publish_or_load_terminal_p6b_session_contract(
            config,
            session_contract,
        )
    )
    if isinstance(cost, StairGenerationSessionCostContractV2):
        runner = replace(
            runner,
            attempt_admission_hook=_StairOperationAdmission(
                output_dir=config.output_dir,
                session_id=config.session_id,
                cost_contract=cost,
                session_contract_hash=session_contract.contract_hash,
                termination_grace_s=runner.termination_grace_s,
            ),
        )
    return StairGenerationSessionFoundation(
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


def _publish_or_load_terminal_p6b_session_contract(
    config: StairGenerationPilotConfig,
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
        ref = publish_generation_session_contract(config.output_dir, current)
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
    config: StairGenerationPilotConfig,
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


def _exercise_preflight_persistence_contracts(
    foundation: StairGenerationSessionFoundation,
    *,
    scratch_root: Path,
) -> dict[str, bool]:
    """Exercise the persistence boundaries needed before any formal GPU work."""

    scratch_root.mkdir(parents=True, exist_ok=False)
    binding = foundation.gpu_bindings[0]
    if foundation.config.lease_root.exists() or (
        foundation.config.lease_root.is_symlink()
    ):
        raise ValueError("preflight-only touched its scratch GPU lease root")

    train_spec, train_result, eval_spec = _preflight_reference_v3_round_trip(
        foundation,
        scratch_root=scratch_root / "reference-v3",
    )
    digest = canonical_json_hash({"mode": "preflight-only"})
    manifest = GenerationManifest(
        schema_version=GENERATION_MANIFEST_V2_SCHEMA_VERSION,
        generation_id="preflight-generation-001",
        generation_index=1,
        session_genesis_hash=digest,
        generation_session_contract_hash=(foundation.session_contract.contract_hash),
        ledger_epoch_hash=digest,
        parent_program_source_checksum=train_spec.program_source_checksum,
        parent_checkpoint_hash=train_result.checkpoint.checksum,
        current_best_evidence_hash=digest,
        base_public_context_hash=digest,
        evaluation_binding_hash=eval_spec.evaluation_binding.binding_hash,
        training_initialization_hash=digest,
        guard_reference_evidence_hash=digest,
        guard_reference_phase_receipt_hash=digest,
        lanes=tuple(
            GenerationLaneAssignment(
                lane_id=lane_id,
                execution_device=binding.process_execution_device,
                resource_binding_hash=binding.resource_binding_hash,
                lane_intent_hash=digest,
                gpu_process_binding=binding,
            )
            for lane_id, binding in zip(
                GENERATION_LANE_IDS,
                foundation.gpu_bindings,
                strict=True,
            )
        ),  # type: ignore[arg-type]
    )
    manifest_ref = GenerationReceiptJournal(
        scratch_root / "generation-receipts", manifest=manifest
    ).bind_manifest()
    if load_generation_manifest_from_receipt(manifest_ref) != manifest:
        raise ValueError("preflight generation manifest did not round-trip")

    trial_contract = {
        "schema_version": TRIAL_EXECUTION_CONTRACT_SCHEMA_VERSION,
        "session_genesis_hash": digest,
        "ledger_epoch_hash": digest,
        "round_id": "preflight-trial",
    }
    trial_journal = TrialReceiptJournal(
        scratch_root / "trial-receipts",
        contract=trial_contract,
    )
    trial_journal.bind_contract()
    trial_journal.publish_train(train_spec, train_result)
    loaded_train = trial_journal.load_train(train_spec)
    if loaded_train is None or loaded_train[0] != train_result:
        raise ValueError("preflight trial train result did not round-trip")

    phase_scope = _job_attempt_scope(
        foundation,
        genesis=SimpleNamespace(session_genesis_hash=digest),
        generation_id="preflight-generation-001",
        lane_id=GENERATION_LANE_IDS[0],
        invocation_segment_id="preflight-only",
        binding=binding,
    )
    phase_spec = foundation.composition.backend_adapter.materialize_eval_phase(
        eval_plan=eval_spec,
        phase=EvaluationPhase.PUBLIC_DEV,
        attempt_scope=phase_scope,
    )
    phase_result = EvaluationPhaseResult(
        job_id=phase_spec.job_id,
        task_id=phase_spec.task_id,
        phase=phase_spec.phase,
        evaluation_plan_hash=phase_spec.evaluation_plan_hash,
        subject=phase_spec.subject,
        subject_manifest_hash=phase_spec.subject_manifest_hash,
        program_source_checksum=phase_spec.program_source_checksum,
        training_binding_hash=phase_spec.training_binding_hash,
        capability_profile_hash=phase_spec.capability_profile_hash,
        checkpoint_hash=phase_spec.execution_checkpoint().checksum,
        eval_protocol_hash=phase_spec.evaluation_protocol().protocol_hash,
        evaluation_binding_hash=phase_spec.evaluation_binding.binding_hash,
        execution_device=phase_spec.execution_device,
        resource_binding_hash=binding.resource_binding_hash,
        task_evidence={"preflight_only": True},
        execution_cost={},
        environment_source_preflight={},
        backend_execution_environment_verification={},
        checkpoint_safety_report={},
        gpu_process_binding=binding.to_dict(),
    )
    trial_journal.publish_eval_phase(phase_spec, phase_result)
    loaded_phase = trial_journal.load_eval_phase(phase_spec)
    if loaded_phase is None or loaded_phase[0] != phase_result:
        raise ValueError("preflight eval phase result did not round-trip")
    if phase_scope.journal_root.exists() or phase_scope.journal_root.is_symlink():
        raise ValueError("preflight-only wrote a job-attempt journal")
    return {
        "reference_v3_exact_private_round_trip": True,
        "eval_checkpoint_bind_and_process_input_validated": True,
        "generation_manifest_v2_round_trip": True,
        "trial_train_private_binding_round_trip": True,
        "eval_phase_private_binding_round_trip": True,
        "no_lease_attempt_ledger_genesis_or_cost_reservation_created": True,
    }


def _preflight_reference_v3_round_trip(
    foundation: StairGenerationSessionFoundation,
    *,
    scratch_root: Path,
) -> tuple[Any, Any, Any]:
    """Exercise the exact GPU-bound reference writer/loader boundary."""

    from true_rl_agent.agent_core.session_genesis import (
        GPU_BOUND_REFERENCE_EVAL_ARTIFACT_SCHEMA_VERSION,
        _reference_eval_artifact_payload,
        _reference_eval_spec_from_private_artifact,
        _reference_train_result_from_private_artifact,
        _reference_train_spec_from_private_artifact,
        _job_spec_to_private_dict,
    )
    from true_rl_agent.runner.base import TrainJobResult
    from true_rl_agent.runner.checkpoint import file_sha256
    from true_rl_agent.runner.checkpoint_preflight import (
        validate_checkpoint_process_input,
    )
    from true_rl_agent.runner.eval_binding import bind_eval_spec_to_checkpoint
    from true_rl_agent.runner.specs import CheckpointRef

    reference = foundation.composition.reference_program_artifact
    if reference is None:
        raise ValueError("preflight requires the frozen Stair reference program")
    scratch_root.mkdir(parents=True, exist_ok=False)
    gpu_process_binding = foundation.gpu_bindings[0]
    digest = canonical_json_hash({"mode": "preflight-only"})
    backend = foundation.composition.backend_adapter
    task_spec = foundation.composition.task_plugin.task_spec
    jobs = backend.materialize_reference_jobs(
        session_id=f"{foundation.config.session_id}-preflight-v3",
        task_spec=task_spec,
        reference=reference,
        artifact_root=scratch_root / "reference-program-artifacts",
        policy_initialization=foundation.composition.policy_initialization,
        evaluation_binding=backend.evaluation_binding_for(task_spec),
        backend_execution_environment_manifest_hash=digest,
        gpu_process_binding=gpu_process_binding,
    )
    train_spec = jobs.train_spec
    planned_eval_spec = jobs.eval_spec
    reference_artifact = train_spec.reference_artifact
    training_binding = train_spec.training_binding

    checkpoint_path = (scratch_root / "checkpoint" / "best.pt").resolve()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=False)
    checkpoint_path.write_bytes(b"preflight-reference-checkpoint\n")
    checkpoint_hash = file_sha256(checkpoint_path)
    checkpoint_metadata: dict[str, Any] = {
        "local_path": str(checkpoint_path),
        "training_subject": train_spec.subject.to_public_dict(),
        "subject_manifest_hash": train_spec.subject_manifest_hash,
        "program_source_checksum": train_spec.program_source_checksum,
        "training_binding_hash": training_binding.binding_hash,
        "capability_profile_hash": training_binding.capability_profile_hash,
        "execution_device": train_spec.execution_device,
    }
    checkpoint_metadata["gpu_process_binding"] = gpu_process_binding.to_dict()
    checkpoint = CheckpointRef(
        uri=str(train_spec.metadata["sealed_checkpoint_uri"]),
        checksum=checkpoint_hash,
        role="eval",
        sealed=True,
        metadata=checkpoint_metadata,
    )
    train_result = TrainJobResult(
        job_id=train_spec.job_id,
        subject=train_spec.subject,
        subject_manifest_hash=train_spec.subject_manifest_hash,
        program_source_checksum=train_spec.program_source_checksum,
        training_binding_hash=training_binding.binding_hash,
        capability_profile_hash=training_binding.capability_profile_hash,
        execution_device=train_spec.execution_device,
        checkpoint=checkpoint,
        gpu_process_binding=gpu_process_binding,
        summary={
            "formal_training_executed": True,
            "training": {"initial_policy_state_hash": digest},
        },
    )
    eval_spec = bind_eval_spec_to_checkpoint(planned_eval_spec, checkpoint)
    validate_checkpoint_process_input(
        checkpoint,
        argv=eval_spec.metadata["cli_argv"],
        checkpoint_path_option="--checkpoint",
        checkpoint_checksum_option="--checkpoint-sha256",
        resolved_local_path=checkpoint_path,
    )

    public_result = SimpleNamespace(to_public_dict=lambda: {})
    trusted_eval_seal = SimpleNamespace(to_public_dict=lambda: {})
    qualification = SimpleNamespace(
        gate={},
        private_evidence={},
    )
    payload = _reference_eval_artifact_payload(
        session_id=foundation.config.session_id,
        backend_id=backend.backend_id,
        reference=reference,
        reference_artifact=reference_artifact,
        train_spec=train_spec,
        train_result=train_result,
        eval_spec=eval_spec,
        public_result=public_result,
        trusted_eval_seal=trusted_eval_seal,
        qualification=qualification,
    )
    if (
        payload.get("schema_version")
        != GPU_BOUND_REFERENCE_EVAL_ARTIFACT_SCHEMA_VERSION
    ):
        raise ValueError("preflight reference private schema mismatch")
    artifact_path = scratch_root / "reference-eval-artifact.json"
    create_once_bytes(
        artifact_path,
        canonical_json_bytes(payload, trailing_newline=True),
    )
    loaded_payload = load_strict_json_object(
        artifact_path,
        context="preflight reference private artifact",
        require_canonical=True,
        trailing_newline=True,
    )
    if loaded_payload != payload:
        raise ValueError("preflight reference private artifact changed on reload")
    anchor = SimpleNamespace(
        reference_manifest=reference.manifest,
        training_binding=training_binding,
        initial_policy_state_hash=digest,
        checkpoint_hash=checkpoint.checksum,
    )
    loaded_train_spec = _reference_train_spec_from_private_artifact(
        loaded_payload,
        anchor=anchor,
        reference_artifact=reference_artifact,
    )
    loaded_train_result = _reference_train_result_from_private_artifact(
        loaded_payload,
        anchor=anchor,
        train_spec=loaded_train_spec,
    )
    loaded_eval_spec = _reference_eval_spec_from_private_artifact(
        loaded_payload,
        anchor=anchor,
        evaluation_binding=eval_spec.evaluation_binding,
    )
    if _job_spec_to_private_dict(loaded_train_spec) != _job_spec_to_private_dict(
        train_spec
    ) or canonical_json_bytes(dict(loaded_train_spec.metadata)) != canonical_json_bytes(
        dict(train_spec.metadata)
    ):
        raise ValueError("preflight reference train spec changed on reload")
    if loaded_train_result != train_result:
        raise ValueError("preflight reference train result changed on reload")
    if _job_spec_to_private_dict(loaded_eval_spec) != _job_spec_to_private_dict(
        eval_spec
    ) or canonical_json_bytes(dict(loaded_eval_spec.metadata)) != canonical_json_bytes(
        dict(eval_spec.metadata)
    ):
        raise ValueError("preflight reference eval spec changed on reload")

    private_bindings = (
        loaded_payload["train_job"]["spec"].get("gpu_process_binding"),
        loaded_payload["train_result"].get("gpu_process_binding"),
        loaded_payload["eval_job"]["spec"].get("gpu_process_binding"),
        loaded_payload["train_result"]["checkpoint"]["metadata"].get(
            "gpu_process_binding"
        ),
    )
    if any(item != gpu_process_binding.to_dict() for item in private_bindings):
        raise ValueError("preflight reference v3 lost a private GPU binding")
    return train_spec, train_result, eval_spec


def _run_one_generation(
    foundation: StairGenerationSessionFoundation,
    *,
    genesis: VerifiedSessionGenesis,
    cost_journal: StairGenerationActualCostJournal,
    startup_resource_observation: Mapping[str, Any],
) -> dict[str, Any]:
    config = foundation.config
    composition = foundation.composition
    contract = foundation.session_contract
    ledger_path = config.output_dir / "ledger.jsonl"
    ledger_epoch_hash = primary_ledger_epoch_hash(genesis.session_genesis_hash)
    lifecycle_identity = GenerationLifecycleIdentity(
        session_id=config.session_id,
        session_genesis_hash=genesis.session_genesis_hash,
        generation_session_contract_hash=contract.contract_hash,
        final_confirmation_plan_hash=contract.bound_contract_hashes[
            "final_confirmation_plan"
        ],
        ledger_epoch_hash=ledger_epoch_hash,
        output_root=str(config.output_dir.resolve()),
    )
    existing_ledger = load_ledger(ledger_path)
    lifecycle = audit_generation_lifecycle(
        output_dir=config.output_dir,
        identity=lifecycle_identity,
        policy=foundation.continuation_policy,
        require_closed=False,
    )
    if lifecycle.terminal:
        if len(existing_ledger) != 1 or lifecycle.closed_generation_count != 1:
            raise ValueError("terminal P6-B lifecycle must bind one generation")
        latest_cost = cost_journal.load_chain()[-1]
        if lifecycle.cost_chain_head_hash != latest_cost.payload_hash:
            raise ValueError("terminal lifecycle actual-cost head mismatch")
        record, execution_ref = _load_existing_generation_execution(
            foundation,
            record=existing_ledger[0],
        )
        resource_evidence = _build_resource_closure_evidence(
            foundation,
            genesis=genesis,
            invocation_segment_id="terminal-audit",
            startup_observation=startup_resource_observation,
        )
        terminal = build_generation_terminal_binding(
            ledger_path=ledger_path,
            output_dir=config.output_dir,
            attempt_journal_root=config.output_dir / "job_attempts",
            identity=lifecycle_identity,
            policy=foundation.continuation_policy,
            session_contract=contract,
            initial_current_best=genesis.initial_current_best,
            acceptance_protocol=(composition.task_plugin.task_spec.acceptance_protocol),
        )
        return _build_report(
            foundation,
            genesis=genesis,
            generation_record=record,
            generation_execution_receipt=execution_ref.to_dict(),
            generation_committed_or_reused=True,
            pre_exploration=_generation_exploration_for_ledger(
                foundation,
                genesis=genesis,
                ledger=[],
            ),
            post_exploration=_generation_exploration_for_ledger(
                foundation,
                genesis=genesis,
                ledger=existing_ledger,
            ),
            terminal_binding=terminal.to_dict(),
            ledger=existing_ledger,
            actual_cost_receipt=latest_cost,
            resource_evidence=resource_evidence,
        )
    segment = GenerationSegmentJournal.start(
        output_dir=config.output_dir,
        identity=lifecycle_identity,
        policy=foundation.continuation_policy,
        session_contract=contract,
        closed_generation_count=len(existing_ledger),
        cost_chain_head_hash=_latest_cost_head(cost_journal),
        attempt_journal_root=config.output_dir / "job_attempts",
    )
    segment_closed = False
    try:
        attempt_scope_segment_id = _first_generation_segment_id(config.output_dir)
        if existing_ledger:
            if len(existing_ledger) != 1:
                raise ValueError("P6-B ledger exceeds its single generation")
            generation_record, execution_ref = _load_existing_generation_execution(
                foundation,
                record=existing_ledger[0],
            )
            ledger = existing_ledger
            generation_execution_receipt = execution_ref.to_dict()
            generation_committed_or_reused = True
            exploration = _generation_exploration_for_ledger(
                foundation,
                genesis=genesis,
                ledger=[],
            )
        else:
            bootstrap_scope = _job_attempt_scope(
                foundation,
                genesis=genesis,
                generation_id="generation-parent-public-dev-bootstrap",
                lane_id="lane-a",
                invocation_segment_id=attempt_scope_segment_id,
                binding=foundation.gpu_bindings[0],
            )
            guard = run_genesis_parent_public_dev(
                genesis=genesis,
                adapter=foundation.phased_adapter,
                backend=composition.backend_adapter,
                runner=foundation.runner,
                receipt_root=config.output_dir / "parent_public_dev_receipts",
                generation_session_contract_hash=contract.contract_hash,
                attempt_scope=bootstrap_scope,
            )
            generation_inputs, exploration = _build_generation_inputs(
                foundation,
                genesis=genesis,
                guard=guard,
                segment_id=attempt_scope_segment_id,
            )
            result = execute_fixed_width_generation(generation_inputs)
            ledger = load_ledger(ledger_path)
            if len(ledger) != 1 or not (result.committed or result.reused_execution):
                raise ValueError(
                    "P6-B must create exactly one generation ledger record"
                )
            generation_record = dict(result.record)
            generation_execution_receipt = result.execution_receipt_ref.to_dict()
            generation_committed_or_reused = result.committed or result.reused_execution
        resource_evidence = _build_resource_closure_evidence(
            foundation,
            genesis=genesis,
            invocation_segment_id=segment.segment_id,
            startup_observation=startup_resource_observation,
        )
        usage, conservative_charges, open_reservations = _collect_actual_cost_usage(
            foundation,
            genesis=genesis,
            ledger=ledger,
        )
        actual_cost = cost_journal.publish_snapshot(
            event_id=f"segment-{segment.segment_id}-settlement",
            previous_cost_chain_head_hash=segment.cost_chain_head_hash,
            usage=usage,
            conservative_charges=conservative_charges,
        )
        if open_reservations:
            raise ValueError("generation left an open job-attempt reservation")
        if actual_cost.payload["cap_checks"]["all_passed"] is not True:
            segment.finish(
                status="failed",
                reason="budget_exhausted",
                closed_generation_count=1,
                cost_chain_head_hash=actual_cost.payload_hash,
            )
            segment_closed = True
            raise ValueError("generation exhausted the Stair cost contract")
        segment.finish(
            status="completed",
            reason="normal",
            closed_generation_count=1,
            cost_chain_head_hash=actual_cost.payload_hash,
        )
        segment_closed = True
    except BaseException as exc:
        if not segment_closed:
            try:
                failed_ledger = load_ledger(ledger_path)
                usage, conservative_charges, open_reservations = (
                    _collect_actual_cost_usage(
                        foundation,
                        genesis=genesis,
                        ledger=failed_ledger,
                    )
                )
                failed_cost = cost_journal.publish_snapshot(
                    event_id=f"segment-{segment.segment_id}-failed",
                    previous_cost_chain_head_hash=segment.cost_chain_head_hash,
                    usage=usage,
                    conservative_charges=conservative_charges,
                )
                segment.finish(
                    status="failed",
                    reason="exception",
                    closed_generation_count=len(failed_ledger),
                    cost_chain_head_hash=failed_cost.payload_hash,
                    unclosed_attempt_reservations=open_reservations,
                    exception=exc,
                )
            except Exception as lifecycle_error:  # noqa: BLE001 - preserve original failure
                add_note = getattr(exc, "add_note", None)
                if callable(add_note):
                    add_note(
                        "generation lifecycle exit evidence failed: "
                        + type(lifecycle_error).__name__
                    )
        raise
    terminal = build_generation_terminal_binding(
        ledger_path=ledger_path,
        output_dir=config.output_dir,
        attempt_journal_root=config.output_dir / "job_attempts",
        identity=lifecycle_identity,
        policy=foundation.continuation_policy,
        session_contract=contract,
        initial_current_best=genesis.initial_current_best,
        acceptance_protocol=composition.task_plugin.task_spec.acceptance_protocol,
    )
    post_history = project_public_history(
        ledger,
        initial_current_best=genesis.initial_current_best,
        expected_session_genesis_hash=genesis.session_genesis_hash,
        acceptance_protocol=composition.task_plugin.task_spec.acceptance_protocol,
        expected_initial_policy_state_hash=(genesis.expected_initial_policy_state_hash),
        expected_authoring_mode=PROVIDER_AUTHORING_MODE,
        expected_ledger_epoch_hash=ledger_epoch_hash,
    )
    post_exploration = (
        composition.task_memory_adapter.build_generation_exploration_plan(
            post_history,
            generation_session_contract=contract,
        )
    )
    return _build_report(
        foundation,
        genesis=genesis,
        generation_record=generation_record,
        generation_execution_receipt=generation_execution_receipt,
        generation_committed_or_reused=generation_committed_or_reused,
        pre_exploration=exploration,
        post_exploration=post_exploration,
        terminal_binding=terminal.to_dict(),
        ledger=ledger,
        actual_cost_receipt=actual_cost,
        resource_evidence=resource_evidence,
    )


def _validation_lifecycle_identity(
    foundation: StairGenerationSessionFoundation,
    *,
    genesis: VerifiedSessionGenesis,
) -> GenerationLifecycleIdentity:
    contract = foundation.session_contract
    if (
        contract.session_mode != DEVELOPMENT_GENERATION_VALIDATION_MODE
        or contract.max_generations != 3
    ):
        raise ValueError("Stair bounded generation lifecycle requires the frozen validation contract")
    return _bounded_lifecycle_identity(foundation, genesis=genesis)


def _formal_lifecycle_identity(
    foundation: StairGenerationSessionFoundation,
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
    foundation: StairGenerationSessionFoundation,
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


def _run_validation_session(
    foundation: StairGenerationSessionFoundation,
    *,
    genesis: VerifiedSessionGenesis,
    cost_journal: StairGenerationActualCostJournal,
    startup_resource_observation: Mapping[str, Any],
) -> dict[str, Any]:
    """Run or resume the frozen three-generation Stair validation session."""

    contract = foundation.session_contract
    if (
        contract.session_mode != DEVELOPMENT_GENERATION_VALIDATION_MODE
        or contract.max_generations != 3
    ):
        raise ValueError("Stair bounded generation validation requires the frozen three-generation mode")
    return _run_bounded_generation_session(
        foundation,
        genesis=genesis,
        cost_journal=cost_journal,
        startup_resource_observation=startup_resource_observation,
        lifecycle_identity=_validation_lifecycle_identity(
            foundation,
            genesis=genesis,
        ),
    )


def _run_formal_session(
    foundation: StairGenerationSessionFoundation,
    *,
    genesis: VerifiedSessionGenesis,
    cost_journal: StairGenerationActualCostJournal,
    startup_resource_observation: Mapping[str, Any],
) -> dict[str, Any]:
    """Run, resume, or finalize the frozen fifteen-generation Stair session."""

    contract = foundation.session_contract
    if (
        contract.session_mode != FORMAL_GENERATION_SESSION_MODE
        or contract.max_generations != STAIR_FINAL_CONFIRMATION_REQUIRED_GENERATIONS
        or not isinstance(foundation.cost_contract, StairGenerationSessionCostContractV3)
        or foundation.final_confirmation_plan is None
    ):
        raise ValueError("P6-D formal session foundation is incomplete")
    lifecycle_identity = _formal_lifecycle_identity(
        foundation,
        genesis=genesis,
    )
    lifecycle = audit_generation_lifecycle(
        output_dir=foundation.config.output_dir,
        identity=lifecycle_identity,
        policy=foundation.continuation_policy,
        require_closed=False,
        require_recovery_attempt_results=False,
    )
    if lifecycle.terminal:
        ledger = load_ledger(foundation.config.output_dir / "ledger.jsonl")
        if len(ledger) != STAIR_FINAL_CONFIRMATION_REQUIRED_GENERATIONS:
            raise ValueError("P6-D terminal lifecycle has the wrong ledger height")
        plan = foundation.final_confirmation_plan
        assert isinstance(plan, StairFinalConfirmationPlanV2)
        reservation_path = stair_final_confirmation_reservation_path(
            foundation.config.output_dir
        )
        if reservation_path.is_file():
            reservation = load_stair_generation_final_confirmation_reservation(
                reservation_path,
                expected_plan=plan,
            )
            terminal_binding = dict(reservation.terminal_binding)
        else:
            terminal_binding = build_generation_terminal_binding(
                ledger_path=foundation.config.output_dir / "ledger.jsonl",
                output_dir=foundation.config.output_dir,
                attempt_journal_root=foundation.config.output_dir / "job_attempts",
                identity=lifecycle_identity,
                policy=foundation.continuation_policy,
                session_contract=contract,
                initial_current_best=genesis.initial_current_best,
                acceptance_protocol=(
                    foundation.composition.task_plugin.task_spec.acceptance_protocol
                ),
                expected_initial_policy_state_hash=(
                    genesis.expected_initial_policy_state_hash
                ),
            ).to_dict()
        generation_cost = _cost_receipt_by_hash(
            cost_journal,
            str(terminal_binding["cost_chain_head_hash"]),
        )
        return _finalize_formal_generation_session(
            foundation,
            genesis=genesis,
            ledger=ledger,
            terminal_binding=terminal_binding,
            generation_cost_receipt=generation_cost,
            generation_resource_evidence={},
            cost_journal=cost_journal,
            startup_resource_observation=startup_resource_observation,
        )
    return _run_bounded_generation_session(
        foundation,
        genesis=genesis,
        cost_journal=cost_journal,
        startup_resource_observation=startup_resource_observation,
        lifecycle_identity=lifecycle_identity,
    )


def _cost_receipt_by_hash(
    journal: StairGenerationActualCostJournal,
    payload_hash: str,
) -> StairGenerationActualCostReceiptRef:
    matches = [item for item in journal.load_chain() if item.payload_hash == payload_hash]
    if len(matches) != 1:
        raise ValueError("P6-D cost-chain receipt is not uniquely identified")
    return matches[0]


def _run_bounded_generation_session(
    foundation: StairGenerationSessionFoundation,
    *,
    genesis: VerifiedSessionGenesis,
    cost_journal: StairGenerationActualCostJournal,
    startup_resource_observation: Mapping[str, Any],
    lifecycle_identity: GenerationLifecycleIdentity,
) -> dict[str, Any]:
    """Run the single bounded generation engine shared by P6-C and P6-D."""

    config = foundation.config
    composition = foundation.composition
    contract = foundation.session_contract
    if (contract.session_mode, contract.max_generations) not in {
        (DEVELOPMENT_GENERATION_VALIDATION_MODE, 3),
        (
            FORMAL_GENERATION_SESSION_MODE,
            STAIR_FINAL_CONFIRMATION_REQUIRED_GENERATIONS,
        ),
    }:
        raise ValueError("bounded generation session contract is unsupported")
    ledger_path = config.output_dir / "ledger.jsonl"
    ledger = load_ledger(ledger_path)
    lifecycle = audit_generation_lifecycle(
        output_dir=config.output_dir,
        identity=lifecycle_identity,
        policy=foundation.continuation_policy,
        require_closed=False,
        require_recovery_attempt_results=False,
    )
    combined_cost_head = _replay_validation_combined_cost_chain(
        cost_journal,
        lifecycle,
    )
    requested_closed_boundary = (
        config.pause_after_generation is not None
        and len(ledger) == config.pause_after_generation
    )
    if (
        config.pause_after_generation is not None
        and len(ledger) > config.pause_after_generation
    ):
        raise ValueError("requested generation boundary is already closed")
    if requested_closed_boundary and lifecycle.active_segment_id is None:
        generation_index = len(ledger)
        closure_path = (
            config.output_dir
            / "generation_closures"
            / f"{ledger[-1]['generation_id']}.json"
        )
        if (
            cost_journal.receipt_for_event(
                f"generation-{generation_index:03d}-settlement"
            )
            is not None
            and closure_path.is_file()
            and _validation_boundary_was_already_paused(
                config.output_dir,
                generation_count=generation_index,
            )
        ):
            raise ValueError("requested generation boundary is already closed")
    if lifecycle.terminal:
        if len(ledger) != contract.max_generations:
            raise ValueError("terminal Stair bounded generation lifecycle has the wrong ledger height")
        latest_cost = cost_journal.load_chain()[-1]
        if combined_cost_head != latest_cost.payload_hash:
            raise ValueError("terminal lifecycle actual-cost head mismatch")
        terminal = build_generation_terminal_binding(
            ledger_path=ledger_path,
            output_dir=config.output_dir,
            attempt_journal_root=config.output_dir / "job_attempts",
            identity=lifecycle_identity,
            policy=foundation.continuation_policy,
            session_contract=contract,
            initial_current_best=genesis.initial_current_best,
            acceptance_protocol=(composition.task_plugin.task_spec.acceptance_protocol),
            expected_initial_policy_state_hash=(
                genesis.expected_initial_policy_state_hash
            ),
        )
        resource_evidence = _build_resource_closure_evidence(
            foundation,
            genesis=genesis,
            invocation_segment_id="terminal-audit",
            startup_observation=startup_resource_observation,
            generation_index=len(ledger),
        )
        return _build_bounded_terminal_result(
            foundation,
            genesis=genesis,
            ledger=ledger,
            terminal_binding=terminal.to_dict(),
            actual_cost_receipt=latest_cost,
            resource_evidence=resource_evidence,
            cost_journal=cost_journal,
            startup_resource_observation=startup_resource_observation,
        )

    segment = GenerationSegmentJournal.start(
        output_dir=config.output_dir,
        identity=lifecycle_identity,
        policy=foundation.continuation_policy,
        session_contract=contract,
        closed_generation_count=len(ledger),
        cost_chain_head_hash=combined_cost_head,
        attempt_journal_root=config.output_dir / "job_attempts",
    )
    started_lifecycle = audit_generation_lifecycle(
        output_dir=config.output_dir,
        identity=lifecycle_identity,
        policy=foundation.continuation_policy,
        require_closed=False,
    )
    segment_cost_chain_head = _replay_validation_combined_cost_chain(
        cost_journal,
        started_lifecycle,
    )
    if segment.cost_chain_head_hash != segment_cost_chain_head:
        raise ValueError("Stair bounded generation segment cost head differs from combined replay")
    segment_closed = False
    try:
        bootstrap_scope = _job_attempt_scope(
            foundation,
            genesis=genesis,
            generation_id="generation-parent-public-dev-bootstrap",
            lane_id="lane-a",
            invocation_segment_id=segment.segment_id,
            binding=foundation.gpu_bindings[0],
        )
        bootstrap_scope = recover_stair_parent_bootstrap_attempt_scope(
            attempt_journal_root=config.output_dir / "job_attempts",
            session_id=config.session_id,
            current_invocation_segment_id=segment.segment_id,
            current_scope=bootstrap_scope,
        )
        initial_guard = run_genesis_parent_public_dev(
            genesis=genesis,
            adapter=foundation.phased_adapter,
            backend=composition.backend_adapter,
            runner=foundation.runner,
            receipt_root=config.output_dir / "parent_public_dev_receipts",
            generation_session_contract_hash=contract.contract_hash,
            attempt_scope=bootstrap_scope,
        )
        existing_bootstrap_cost = cost_journal.receipt_for_event(
            "post-parent-bootstrap"
        )
        bootstrap_cost = _publish_or_load_parent_bootstrap_cost_boundary(
            foundation,
            genesis=genesis,
            ledger=ledger,
            cost_journal=cost_journal,
            previous_cost_chain_head_hash=segment_cost_chain_head,
        )
        if existing_bootstrap_cost is None:
            segment_cost_chain_head = bootstrap_cost.payload_hash
        cursor = build_generation_continuation_cursor(
            ledger_path=ledger_path,
            output_dir=config.output_dir,
            identity=lifecycle_identity,
            policy=foundation.continuation_policy,
            session_contract=contract,
            initial_current_best=genesis.initial_current_best,
            acceptance_protocol=(composition.task_plugin.task_spec.acceptance_protocol),
            expected_initial_policy_state_hash=(
                genesis.expected_initial_policy_state_hash
            ),
        )
        parent, guard = replay_stair_generation_guard_chain(
            generation_records=ledger,
            initial_parent=genesis.initial_current_best,
            initial_guard_reference=initial_guard,
            acceptance_protocol=(composition.task_plugin.task_spec.acceptance_protocol),
            adapter=foundation.phased_adapter,
            expected_generation_session_contract_hash=contract.contract_hash,
        )
        if (
            parent != cursor.current_best
            or cursor.next_generation_index != len(ledger) + 1
        ):
            raise ValueError("Stair bounded generation guard replay differs from continuation cursor")

        closures: list[dict[str, Any]] = []
        latest_resource_evidence: Mapping[str, Any] | None = None
        for index, existing_record in enumerate(ledger, start=1):
            loaded_record, execution_ref = _load_existing_generation_execution(
                foundation,
                record=existing_record,
            )
            event_id = f"generation-{index:03d}-settlement"
            cost_ref = cost_journal.receipt_for_event(event_id)
            cost_receipt_existed = cost_ref is not None
            closure_path = (
                config.output_dir
                / "generation_closures"
                / f"{loaded_record['generation_id']}.json"
            )
            if cost_ref is None or not closure_path.is_file():
                if index != len(ledger):
                    raise ValueError(
                        "Stair bounded generation closed generation prefix lacks an automatic closure"
                    )
                latest_resource_evidence = _build_resource_closure_evidence(
                    foundation,
                    genesis=genesis,
                    invocation_segment_id=segment.segment_id,
                    startup_observation=startup_resource_observation,
                    generation_index=index,
                )
                cost_ref, open_reservations = _settle_generation_cost(
                    foundation,
                    genesis=genesis,
                    ledger=ledger,
                    cost_journal=cost_journal,
                    event_id=event_id,
                    previous_cost_chain_head_hash=segment_cost_chain_head,
                )
                if not cost_receipt_existed:
                    segment_cost_chain_head = cost_ref.payload_hash
                if open_reservations:
                    raise ValueError(
                        "Stair bounded generation recovered generation has open job-attempt reservations"
                    )
            assert cost_ref is not None
            closure = _publish_or_validate_generation_closure(
                foundation,
                ledger=ledger,
                generation_record=loaded_record,
                generation_execution_receipt=execution_ref.to_dict(),
                actual_cost_receipt=cost_ref,
                resource_evidence=latest_resource_evidence or {},
                segment_id=segment.segment_id,
            )
            closures.append(closure)

        if requested_closed_boundary:
            if not closures:
                raise ValueError("Stair bounded generation recovered boundary has no completed closure")
            boundary_cost = _current_validation_boundary_cost_receipt(
                foundation,
                genesis=genesis,
                ledger=ledger,
                cost_journal=cost_journal,
                lifecycle=started_lifecycle,
                segment_id=segment.segment_id,
            )
            segment_cost_chain_head = boundary_cost.payload_hash
            segment.finish(
                status="paused",
                reason="operator_pause",
                closed_generation_count=len(ledger),
                cost_chain_head_hash=segment_cost_chain_head,
            )
            segment_closed = True
            return _build_bounded_pause_result(
                foundation,
                genesis=genesis,
                ledger=ledger,
                segment_id=segment.segment_id,
                actual_cost_receipt=boundary_cost,
                closure=closures[-1],
            )

        while len(ledger) < contract.max_generations:
            generation_index = len(ledger) + 1
            _assert_generation_admission(
                foundation,
                genesis=genesis,
                ledger=ledger,
                generation_index=generation_index,
            )
            generation_inputs, _ = _build_generation_inputs(
                foundation,
                genesis=genesis,
                guard=guard,
                segment_id=segment.segment_id,
                generation_index=generation_index,
                parent=parent,
            )
            result = execute_fixed_width_generation(generation_inputs)
            next_ledger = load_ledger(ledger_path)
            if (
                next_ledger[: len(ledger)] != ledger
                or len(next_ledger) != generation_index
                or not (result.committed or result.reused_execution)
            ):
                raise ValueError("bounded generation did not append exactly one record")
            ledger = next_ledger
            generation_record = dict(result.record)
            if generation_record != ledger[-1]:
                raise ValueError("Stair bounded generation execution result differs from its ledger record")
            latest_resource_evidence = _build_resource_closure_evidence(
                foundation,
                genesis=genesis,
                invocation_segment_id=segment.segment_id,
                startup_observation=startup_resource_observation,
                generation_index=generation_index,
            )
            actual_cost, open_reservations = _settle_generation_cost(
                foundation,
                genesis=genesis,
                ledger=ledger,
                cost_journal=cost_journal,
                event_id=f"generation-{generation_index:03d}-settlement",
                previous_cost_chain_head_hash=segment_cost_chain_head,
            )
            segment_cost_chain_head = actual_cost.payload_hash
            if open_reservations:
                raise ValueError("bounded generation left an open attempt reservation")
            if actual_cost.payload["cap_checks"]["all_passed"] is not True:
                segment.finish(
                    status="failed",
                    reason="budget_exhausted",
                    closed_generation_count=len(ledger),
                    cost_chain_head_hash=actual_cost.payload_hash,
                )
                segment_closed = True
                raise ValueError("bounded generation exhausted its frozen cost contract")
            closures.append(
                _publish_or_validate_generation_closure(
                    foundation,
                    ledger=ledger,
                    generation_record=generation_record,
                    generation_execution_receipt=(
                        result.execution_receipt_ref.to_dict()
                    ),
                    actual_cost_receipt=actual_cost,
                    resource_evidence=latest_resource_evidence,
                    segment_id=segment.segment_id,
                )
            )
            cursor = build_generation_continuation_cursor(
                ledger_path=ledger_path,
                output_dir=config.output_dir,
                identity=lifecycle_identity,
                policy=foundation.continuation_policy,
                session_contract=contract,
                initial_current_best=genesis.initial_current_best,
                acceptance_protocol=(
                    composition.task_plugin.task_spec.acceptance_protocol
                ),
                expected_initial_policy_state_hash=(
                    genesis.expected_initial_policy_state_hash
                ),
            )
            parent, guard = replay_stair_generation_guard_chain(
                generation_records=ledger,
                initial_parent=genesis.initial_current_best,
                initial_guard_reference=initial_guard,
                acceptance_protocol=(
                    composition.task_plugin.task_spec.acceptance_protocol
                ),
                adapter=foundation.phased_adapter,
                expected_generation_session_contract_hash=contract.contract_hash,
            )
            if parent != cursor.current_best:
                raise ValueError("Stair bounded generation post-generation head replay diverged")
            if config.pause_after_generation == generation_index:
                boundary_cost = _current_validation_boundary_cost_receipt(
                    foundation,
                    genesis=genesis,
                    ledger=ledger,
                    cost_journal=cost_journal,
                    lifecycle=started_lifecycle,
                    segment_id=segment.segment_id,
                )
                segment_cost_chain_head = boundary_cost.payload_hash
                segment.finish(
                    status="paused",
                    reason="operator_pause",
                    closed_generation_count=len(ledger),
                    cost_chain_head_hash=segment_cost_chain_head,
                )
                segment_closed = True
                return _build_bounded_pause_result(
                    foundation,
                    genesis=genesis,
                    ledger=ledger,
                    segment_id=segment.segment_id,
                    actual_cost_receipt=boundary_cost,
                    closure=closures[-1],
                )

        if latest_resource_evidence is None:
            latest_resource_evidence = _build_resource_closure_evidence(
                foundation,
                genesis=genesis,
                invocation_segment_id=segment.segment_id,
                startup_observation=startup_resource_observation,
                generation_index=len(ledger),
            )
        latest_cost = _current_validation_boundary_cost_receipt(
            foundation,
            genesis=genesis,
            ledger=ledger,
            cost_journal=cost_journal,
            lifecycle=started_lifecycle,
            segment_id=segment.segment_id,
        )
        segment_cost_chain_head = latest_cost.payload_hash
        prospective_terminal = GenerationTerminalBinding(
            identity_hash=lifecycle_identity.identity_hash,
            required_generation_count=contract.max_generations,
            closed_generation_count=len(ledger),
            ledger_snapshot_hash=cursor.ledger_snapshot_hash,
            current_head_hash=cursor.current_head_hash,
            lifecycle_head_hash="0" * 64,
            cost_chain_head_hash=segment_cost_chain_head,
            segment_count=segment.continuation_index + 1,
        )
        _prevalidate_bounded_terminal(
            foundation,
            genesis=genesis,
            ledger=ledger,
            terminal_binding=prospective_terminal.to_dict(),
            actual_cost_receipt=latest_cost,
            resource_evidence=latest_resource_evidence,
        )
        segment.finish(
            status="completed",
            reason="normal",
            closed_generation_count=len(ledger),
            cost_chain_head_hash=segment_cost_chain_head,
        )
        segment_closed = True
    except BaseException as exc:
        if not segment_closed:
            try:
                failed_ledger = load_ledger(ledger_path)
                usage, conservative_charges, open_reservations = (
                    _collect_actual_cost_usage(
                        foundation,
                        genesis=genesis,
                        ledger=failed_ledger,
                    )
                )
                failed_cost = cost_journal.publish_snapshot(
                    event_id=(
                        f"segment-{segment.segment_id}-failed-"
                        f"generation-{len(failed_ledger):03d}"
                    ),
                    previous_cost_chain_head_hash=segment_cost_chain_head,
                    usage=usage,
                    conservative_charges=conservative_charges,
                )
                segment.finish(
                    status="failed",
                    reason="exception",
                    closed_generation_count=len(failed_ledger),
                    cost_chain_head_hash=failed_cost.payload_hash,
                    unclosed_attempt_reservations=open_reservations,
                    exception=exc,
                )
            except Exception as lifecycle_error:  # noqa: BLE001
                add_note = getattr(exc, "add_note", None)
                if callable(add_note):
                    add_note(
                        "generation lifecycle exit evidence failed: "
                        + type(lifecycle_error).__name__
                    )
        raise

    terminal = build_generation_terminal_binding(
        ledger_path=ledger_path,
        output_dir=config.output_dir,
        attempt_journal_root=config.output_dir / "job_attempts",
        identity=lifecycle_identity,
        policy=foundation.continuation_policy,
        session_contract=contract,
        initial_current_best=genesis.initial_current_best,
        acceptance_protocol=composition.task_plugin.task_spec.acceptance_protocol,
        expected_initial_policy_state_hash=(genesis.expected_initial_policy_state_hash),
    )
    return _build_bounded_terminal_result(
        foundation,
        genesis=genesis,
        ledger=ledger,
        terminal_binding=terminal.to_dict(),
        actual_cost_receipt=latest_cost,
        resource_evidence=latest_resource_evidence or {},
        cost_journal=cost_journal,
        startup_resource_observation=startup_resource_observation,
    )


def _build_bounded_pause_result(
    foundation: StairGenerationSessionFoundation,
    *,
    genesis: VerifiedSessionGenesis,
    ledger: list[dict[str, Any]],
    segment_id: str,
    actual_cost_receipt: StairGenerationActualCostReceiptRef,
    closure: Mapping[str, Any],
) -> dict[str, Any]:
    if foundation.session_contract.session_mode == DEVELOPMENT_GENERATION_VALIDATION_MODE:
        return _build_validation_pause_result(
            foundation,
            genesis=genesis,
            ledger=ledger,
            segment_id=segment_id,
            actual_cost_receipt=actual_cost_receipt,
            closure=closure,
        )
    if foundation.session_contract.session_mode == FORMAL_GENERATION_SESSION_MODE:
        return _build_formal_pause_result(
            foundation,
            genesis=genesis,
            ledger=ledger,
            segment_id=segment_id,
            actual_cost_receipt=actual_cost_receipt,
            closure=closure,
        )
    raise ValueError("bounded generation pause mode is unsupported")


def _prevalidate_bounded_terminal(
    foundation: StairGenerationSessionFoundation,
    *,
    genesis: VerifiedSessionGenesis,
    ledger: list[dict[str, Any]],
    terminal_binding: Mapping[str, Any],
    actual_cost_receipt: StairGenerationActualCostReceiptRef,
    resource_evidence: Mapping[str, Any],
) -> None:
    if foundation.session_contract.session_mode == DEVELOPMENT_GENERATION_VALIDATION_MODE:
        _build_validation_report(
            foundation,
            genesis=genesis,
            ledger=ledger,
            terminal_binding=terminal_binding,
            actual_cost_receipt=actual_cost_receipt,
            resource_evidence=resource_evidence,
            created_at_utc="pre-terminal-validation",
        )
        return
    if foundation.session_contract.session_mode != FORMAL_GENERATION_SESSION_MODE:
        raise ValueError("bounded terminal prevalidation mode is unsupported")
    plan = foundation.final_confirmation_plan
    checks = {
        "fifteen_generation_records_are_closed": (
            len(ledger)
            == foundation.session_contract.max_generations
            == STAIR_FINAL_CONFIRMATION_REQUIRED_GENERATIONS
        ),
        "terminal_binding_matches_generation_prefix": (
            terminal_binding.get("required_generation_count")
            == STAIR_FINAL_CONFIRMATION_REQUIRED_GENERATIONS
            and terminal_binding.get("closed_generation_count")
            == STAIR_FINAL_CONFIRMATION_REQUIRED_GENERATIONS
            and terminal_binding.get("cost_chain_head_hash")
            == actual_cost_receipt.payload_hash
        ),
        "generation_cost_caps_passed": (
            actual_cost_receipt.payload["cap_checks"]["all_passed"] is True
        ),
        "final_confirmation_plan_is_genesis_bound": (
            plan is not None
            and plan.plan_hash
            == foundation.session_contract.bound_contract_hashes[
                "final_confirmation_plan"
            ]
        ),
        "generation_resource_closure_is_current": (
            resource_evidence.get("generation_index")
            == STAIR_FINAL_CONFIRMATION_REQUIRED_GENERATIONS
            and resource_evidence.get("attempt_chain", {}).get(
                "open_reservation_count"
            )
            == 0
        ),
    }
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise ValueError(
            "P6-D generation terminal prevalidation failed: " + ", ".join(failed)
        )


def _build_bounded_terminal_result(
    foundation: StairGenerationSessionFoundation,
    *,
    genesis: VerifiedSessionGenesis,
    ledger: list[dict[str, Any]],
    terminal_binding: Mapping[str, Any],
    actual_cost_receipt: StairGenerationActualCostReceiptRef,
    resource_evidence: Mapping[str, Any],
    cost_journal: StairGenerationActualCostJournal,
    startup_resource_observation: Mapping[str, Any],
) -> dict[str, Any]:
    if foundation.session_contract.session_mode == DEVELOPMENT_GENERATION_VALIDATION_MODE:
        return _build_validation_report(
            foundation,
            genesis=genesis,
            ledger=ledger,
            terminal_binding=terminal_binding,
            actual_cost_receipt=actual_cost_receipt,
            resource_evidence=resource_evidence,
        )
    if foundation.session_contract.session_mode == FORMAL_GENERATION_SESSION_MODE:
        return _finalize_formal_generation_session(
            foundation,
            genesis=genesis,
            ledger=ledger,
            terminal_binding=terminal_binding,
            generation_cost_receipt=actual_cost_receipt,
            generation_resource_evidence=resource_evidence,
            cost_journal=cost_journal,
            startup_resource_observation=startup_resource_observation,
        )
    raise ValueError("bounded terminal result mode is unsupported")


@dataclass(frozen=True)
class _FormalFinalEvaluationPair:
    genesis_plan: EvalJobSpec
    final_plan: EvalJobSpec
    genesis_phase_spec: EvalPhaseJobSpec
    final_phase_spec: EvalPhaseJobSpec
    final_current_best: CurrentBestEvalStats


def _finalize_formal_generation_session(
    foundation: StairGenerationSessionFoundation,
    *,
    genesis: VerifiedSessionGenesis,
    ledger: list[dict[str, Any]],
    terminal_binding: Mapping[str, Any],
    generation_cost_receipt: StairGenerationActualCostReceiptRef,
    generation_resource_evidence: Mapping[str, Any],
    cost_journal: StairGenerationActualCostJournal,
    startup_resource_observation: Mapping[str, Any],
) -> dict[str, Any]:
    """Complete diagnostic-only P6-D finalization after the terminal generation."""

    del generation_resource_evidence
    plan = foundation.final_confirmation_plan
    if (
        foundation.session_contract.session_mode != FORMAL_GENERATION_SESSION_MODE
        or foundation.session_contract.max_generations
        != STAIR_FINAL_CONFIRMATION_REQUIRED_GENERATIONS
        or not isinstance(plan, StairFinalConfirmationPlanV2)
        or len(ledger) != STAIR_FINAL_CONFIRMATION_REQUIRED_GENERATIONS
    ):
        raise ValueError("P6-D finalization requires a terminal formal session")
    terminal = GenerationTerminalBinding(
        identity_hash=terminal_binding.get("identity_hash"),
        required_generation_count=terminal_binding.get(
            "required_generation_count"
        ),
        closed_generation_count=terminal_binding.get("closed_generation_count"),
        ledger_snapshot_hash=terminal_binding.get("ledger_snapshot_hash"),
        current_head_hash=terminal_binding.get("current_head_hash"),
        lifecycle_head_hash=terminal_binding.get("lifecycle_head_hash"),
        cost_chain_head_hash=terminal_binding.get("cost_chain_head_hash"),
        segment_count=terminal_binding.get("segment_count"),
    )
    if terminal.to_dict() != dict(terminal_binding):
        raise ValueError("P6-D terminal binding projection mismatch")
    if terminal.cost_chain_head_hash != generation_cost_receipt.payload_hash:
        raise ValueError("P6-D terminal generation cost binding mismatch")

    ledger_path = foundation.config.output_dir / "ledger.jsonl"
    ledger_checksum_before = file_sha256(ledger_path)
    if ledger_checksum_before is None:
        raise ValueError("P6-D terminal ledger checksum is unavailable")
    pair = _build_formal_final_evaluation_pair(
        foundation,
        genesis=genesis,
        ledger=ledger,
        terminal=terminal,
    )
    reservation = _create_or_load_formal_final_reservation(
        foundation,
        genesis=genesis,
        plan=plan,
        pair=pair,
        terminal=terminal,
        ledger_checksum=ledger_checksum_before,
    )
    genesis_execution, genesis_summary = _run_or_load_formal_final_phase(
        foundation,
        genesis=genesis,
        plan=plan,
        reservation=reservation,
        role="genesis",
        eval_plan=pair.genesis_plan,
    )
    final_execution, final_summary = _run_or_load_formal_final_phase(
        foundation,
        genesis=genesis,
        plan=plan,
        reservation=reservation,
        role="final",
        eval_plan=pair.final_plan,
    )
    result = build_stair_generation_final_confirmation_result(
        plan=plan,
        reservation=reservation,
        config=foundation.composition.task_plugin.config,
        genesis_execution=genesis_execution,
        final_execution=final_execution,
        genesis_summary=genesis_summary,
        final_summary=final_summary,
    )
    result_path = stair_final_confirmation_result_path(
        foundation.config.output_dir
    )
    if result_path.is_file():
        stored_result = load_stair_generation_final_confirmation_result(
            result_path,
            plan=plan,
            reservation=reservation,
            config=foundation.composition.task_plugin.config,
        )
        if stored_result != result:
            raise ValueError("P6-D final confirmation Result differs from replay")
    else:
        write_stair_generation_final_confirmation_result(
            output_dir=foundation.config.output_dir,
            result=result,
            plan=plan,
            reservation=reservation,
            config=foundation.composition.task_plugin.config,
        )

    ledger_checksum_after_result = file_sha256(ledger_path)
    if ledger_checksum_after_result != ledger_checksum_before:
        raise ValueError("P6-D final confirmation changed the generation ledger")
    final_cost, open_reservations = _settle_generation_cost(
        foundation,
        genesis=genesis,
        ledger=ledger,
        cost_journal=cost_journal,
        event_id="final-confirmation-settlement",
        previous_cost_chain_head_hash=generation_cost_receipt.payload_hash,
    )
    if open_reservations:
        raise ValueError("P6-D final confirmation left an open attempt reservation")
    if final_cost.payload["cap_checks"]["all_passed"] is not True:
        raise ValueError("P6-D final confirmation exhausted its frozen cost contract")
    final_resource_evidence = _build_resource_closure_evidence(
        foundation,
        genesis=genesis,
        invocation_segment_id="formal-finalization",
        startup_observation=startup_resource_observation,
        generation_index=STAIR_FINAL_CONFIRMATION_REQUIRED_GENERATIONS,
        include_final_confirmation=True,
    )
    report = _build_formal_report(
        foundation,
        genesis=genesis,
        ledger=ledger,
        terminal_binding=terminal.to_dict(),
        generation_cost_receipt=generation_cost_receipt,
        final_cost_receipt=final_cost,
        reservation=reservation,
        result=result,
        resource_evidence=final_resource_evidence,
    )
    current_usage, _, current_open_reservations = _collect_actual_cost_usage(
        foundation,
        genesis=genesis,
        ledger=ledger,
    )
    if current_open_reservations:
        raise ValueError("P6-D report assembly found an open attempt reservation")
    encoded_report = canonical_json_bytes(report, trailing_newline=True)
    accounted_output_bytes = final_cost.payload["actual"][
        "accounted_output_bytes"
    ]
    if (
        current_usage.observed_output_bytes + len(encoded_report)
        > accounted_output_bytes
    ):
        raise ValueError("P6-D report exceeds its settled output reserve")
    return report


def _build_formal_final_evaluation_pair(
    foundation: StairGenerationSessionFoundation,
    *,
    genesis: VerifiedSessionGenesis,
    ledger: list[dict[str, Any]],
    terminal: GenerationTerminalBinding,
) -> _FormalFinalEvaluationPair:
    if not ledger:
        raise ValueError("P6-D final evaluation requires a non-empty ledger")
    current_best = CurrentBestEvalStats.from_mapping(
        ledger[-1].get("resulting_current_best")
    )
    current_best_hash = canonical_json_hash(current_best.to_public_dict())
    if current_best_hash != terminal.current_head_hash:
        raise ValueError("P6-D final current-best differs from terminal head")
    genesis_plan = genesis.reference_eval_plan()
    if genesis_plan is None:
        raise ValueError("P6-D finalization requires the Genesis reference eval plan")
    _require_final_eval_target_matches_current_best(
        genesis_plan,
        current_best=genesis.initial_current_best,
        context="Genesis",
    )
    if current_best.anchor_kind == "reference":
        if current_best != genesis.initial_current_best:
            raise ValueError("P6-D terminal reference head differs from Genesis")
        final_target = genesis_plan
    elif current_best.anchor_kind == "candidate":
        accepted = [
            record
            for record in ledger
            if record.get("generation_outcome") == "accepted"
            and record.get("resulting_current_best_hash") == current_best_hash
        ]
        if len(accepted) != 1:
            raise ValueError("P6-D final candidate head is not uniquely introduced")
        formal = accepted[0].get("formal")
        if not isinstance(formal, Mapping) or formal.get("accepted") is not True:
            raise ValueError("P6-D final candidate lacks accepted formal evidence")
        trial_id = formal.get("selected_trial_id")
        if (
            not isinstance(trial_id, str)
            or trial_id != current_best.accepted_round_id
        ):
            raise ValueError("P6-D final candidate trial identity mismatch")
        loaded_train = load_bound_generation_trial_train_receipt(
            accepted[0],
            trial_id=trial_id,
        )
        train_result = loaded_train.result
        if train_result.initial_policy_state_pairing is not None:
            train_result.initial_policy_state_pairing.require_match()
            pairing_hash = train_result.initial_policy_state_pairing.pairing_hash
        else:
            pairing_hash = None
        final_target = replace(
            genesis_plan,
            subject=train_result.subject,
            subject_manifest_hash=train_result.subject_manifest_hash,
            program_source_checksum=train_result.program_source_checksum,
            training_binding_hash=train_result.training_binding_hash,
            capability_profile_hash=train_result.capability_profile_hash,
            checkpoint=train_result.checkpoint,
            checkpoint_output_plan=None,
            initial_policy_state_pairing_hash=pairing_hash,
        )
        _require_final_eval_target_matches_current_best(
            final_target,
            current_best=current_best,
            context="final candidate",
        )
    else:
        raise ValueError("P6-D terminal head is not an evaluable program anchor")

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


def _require_final_eval_target_matches_current_best(
    target: EvalJobSpec,
    *,
    current_best: CurrentBestEvalStats,
    context: str,
) -> None:
    checkpoint = target.execution_checkpoint()
    expected = {
        "anchor_kind": target.subject.kind,
        "anchor_id": target.subject.subject_id,
        "checkpoint_hash": checkpoint.checksum,
        "subject_manifest_hash": target.subject_manifest_hash,
        "program_source_checksum": target.program_source_checksum,
        "training_binding_hash": target.training_binding_hash,
        "capability_profile_hash": target.capability_profile_hash,
    }
    actual = {
        "anchor_kind": current_best.anchor_kind,
        "anchor_id": current_best.anchor_id,
        "checkpoint_hash": current_best.checkpoint_hash,
        "subject_manifest_hash": current_best.subject_manifest_hash,
        "program_source_checksum": current_best.program_source_checksum,
        "training_binding_hash": current_best.training_binding_hash,
        "capability_profile_hash": current_best.capability_profile_hash,
    }
    if actual != expected:
        raise ValueError(f"P6-D {context} eval target lineage mismatch")


def _bind_formal_final_eval_plan(
    foundation: StairGenerationSessionFoundation,
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
    subject_manifest_hash = require_sha256_digest(
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
    raw_argv = template.metadata.get("cli_argv")
    if not isinstance(raw_argv, list) or not all(
        isinstance(item, str) for item in raw_argv
    ):
        raise ValueError("P6-D final eval template lacks structured argv")
    argv = list(raw_argv)
    replacements = {
        "--job-id": base_job_id,
        "--round": base_job_id,
        "--subject-kind": target.subject.kind,
        "--subject-id": target.subject.subject_id,
        "--subject-manifest-hash": subject_manifest_hash,
        "--program-source-checksum": target.program_source_checksum,
        "--training-binding-hash": target.training_binding_hash,
        "--capability-profile-hash": target.capability_profile_hash,
        "--checkpoint": str(local_checkpoint),
        "--checkpoint-sha256": checkpoint.checksum,
        "--seed-cohort-hash": binding.seed_cohort_hash,
        "--evaluation-binding-hash": binding.binding_hash,
        "--summary-path": summary_path_text,
        "--device": gpu_binding.process_execution_device,
        "--gpu-host-execution-device": gpu_binding.host_execution_device,
        "--gpu-process-execution-device": gpu_binding.process_execution_device,
        "--gpu-resource-binding-hash": gpu_binding.resource_binding_hash,
    }
    for option, value in replacements.items():
        argv = _replace_formal_final_cli_option(argv, option, value)
    if "--stair-eval-purpose" in argv:
        raise ValueError("P6-D final eval template already declares a purpose")
    argv.extend(
        [
            "--stair-eval-purpose",
            STAIR_EVAL_PURPOSE_FINAL_CONFIRMATION,
        ]
    )
    metadata = dict(template.metadata)
    metadata.update(
        {
            "cli_argv": argv,
            "cli_command": " ".join(argv),
            "summary_path": summary_path_text,
            "stair_eval_purpose": STAIR_EVAL_PURPOSE_FINAL_CONFIRMATION,
        }
    )
    return replace(
        target,
        job_id=base_job_id,
        evaluation_binding=binding,
        execution_device=gpu_binding.process_execution_device,
        metadata=metadata,
        attempt_scope=None,
        gpu_process_binding=gpu_binding,
    )


def _replace_formal_final_cli_option(
    argv: list[str],
    option: str,
    value: str,
) -> list[str]:
    positions = [index for index, item in enumerate(argv) if item == option]
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        raise ValueError(f"P6-D final eval requires exactly one {option}")
    updated = list(argv)
    updated[positions[0] + 1] = value
    return updated


def _load_formal_source_checkout_identity(
    foundation: StairGenerationSessionFoundation,
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


def _create_or_load_formal_final_reservation(
    foundation: StairGenerationSessionFoundation,
    *,
    genesis: VerifiedSessionGenesis,
    plan: StairFinalConfirmationPlanV2,
    pair: _FormalFinalEvaluationPair,
    terminal: GenerationTerminalBinding,
    ledger_checksum: str,
) -> StairFinalConfirmationReservationV2:
    source = _load_formal_source_checkout_identity(foundation)
    reservation = _build_formal_final_reservation(
        foundation,
        genesis=genesis,
        plan=plan,
        pair=pair,
        terminal=terminal,
        ledger_checksum=ledger_checksum,
        source_checkout_hash=source["source_checkout_hash"],
    )
    reservation_path = stair_final_confirmation_reservation_path(
        foundation.config.output_dir
    )
    if reservation_path.is_file():
        stored = load_stair_generation_final_confirmation_reservation(
            reservation_path,
            expected_plan=plan,
        )
        if stored != reservation:
            raise ValueError("P6-D final Reservation differs from replay")
        return stored
    stored, _ = create_stair_generation_final_confirmation_reservation(
        output_dir=foundation.config.output_dir,
        plan=plan,
        reservation=reservation,
    )
    return stored


def _build_formal_final_reservation(
    foundation: StairGenerationSessionFoundation,
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
    foundation: StairGenerationSessionFoundation,
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
    source = _load_formal_source_checkout_identity(foundation)
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


def _formal_final_attempt_journal(
    foundation: StairGenerationSessionFoundation,
    *,
    logical_job_id: str,
) -> JobAttemptJournal | None:
    root = foundation.config.output_dir / "job_attempts"
    if not root.exists():
        return None
    journals: dict[Path, JobAttemptJournal] = {}
    for path in sorted(root.glob("*/attempt_*.reservation.json")):
        journal, reservation = JobAttemptJournal.from_reservation_path(path)
        if journal.logical_job_id != logical_job_id:
            continue
        if (
            not isinstance(journal.scope, JobAttemptScope)
            or journal.scope.generation_id != "generation-final-confirmation"
            or journal.scope.lane_id != "lane-a"
            or journal.scope.budget_contract_hash
            != foundation.cost_contract.contract_hash
            or journal.scope.session_contract_hash
            != foundation.session_contract.contract_hash
            or journal.resource_binding_hash
            != foundation.gpu_bindings[0].resource_binding_hash
        ):
            raise ValueError("P6-D final attempt journal binding mismatch")
        _strict_operation_slot(
            session_id=foundation.config.session_id,
            scope=journal.scope,
            logical_job_id=journal.logical_job_id,
            job_kind=journal.job_kind,
            attempt_index=reservation.payload["attempt_index"],
            cost_contract=foundation.cost_contract,
        )
        journals[journal.root] = journal
    if len(journals) > 1:
        raise ValueError("P6-D final job has more than one attempt journal")
    return next(iter(journals.values()), None)


def _formal_final_attempt_scope(
    foundation: StairGenerationSessionFoundation,
    *,
    genesis: VerifiedSessionGenesis,
    logical_job_id: str,
) -> JobAttemptScope:
    existing = _formal_final_attempt_journal(
        foundation,
        logical_job_id=logical_job_id,
    )
    if existing is not None:
        assert isinstance(existing.scope, JobAttemptScope)
        return existing.scope
    return _job_attempt_scope(
        foundation,
        genesis=genesis,
        generation_id="generation-final-confirmation",
        lane_id="lane-a",
        invocation_segment_id="formal-finalization",
        binding=foundation.gpu_bindings[0],
        max_infrastructure_retries=_remaining_sealed_retry_entitlement(foundation),
    )


def _run_or_load_formal_final_phase(
    foundation: StairGenerationSessionFoundation,
    *,
    genesis: VerifiedSessionGenesis,
    plan: StairFinalConfirmationPlanV2,
    reservation: StairFinalConfirmationReservationV2,
    role: str,
    eval_plan: EvalJobSpec,
) -> tuple[StairFinalConfirmationPhaseExecution, Mapping[str, Any]]:
    if role not in {"genesis", "final"}:
        raise ValueError("P6-D final phase role is invalid")
    expected_job_id = (
        reservation.genesis_job_id
        if role == "genesis"
        else reservation.final_job_id
    )
    scope = _formal_final_attempt_scope(
        foundation,
        genesis=genesis,
        logical_job_id=expected_job_id,
    )
    spec = foundation.composition.backend_adapter.materialize_eval_phase(
        eval_plan=eval_plan,
        phase=EvaluationPhase.SEALED_VALIDATION,
        attempt_scope=scope,
    )
    expected_spec_hash = (
        reservation.genesis_eval_phase_spec_hash
        if role == "genesis"
        else reservation.final_eval_phase_spec_hash
    )
    if (
        spec.job_id != expected_job_id
        or canonical_json_hash(spec.to_public_dict()) != expected_spec_hash
    ):
        raise ValueError("P6-D final phase spec differs from Reservation")
    contract = {
        "schema_version": TRIAL_EXECUTION_CONTRACT_SCHEMA_VERSION,
        "session_genesis_hash": genesis.session_genesis_hash,
        "ledger_epoch_hash": primary_ledger_epoch_hash(
            genesis.session_genesis_hash
        ),
        "round_id": spec.job_id,
        "generation_session_contract_hash": foundation.session_contract.contract_hash,
        "final_confirmation_plan_hash": plan.plan_hash,
        "final_confirmation_reservation_hash": reservation.reservation_hash,
        "final_confirmation_role": role,
    }
    journal = TrialReceiptJournal(
        foundation.config.output_dir
        / STAIR_FINAL_CONFIRMATION_DIRNAME
        / "phase_receipts"
        / role,
        contract=contract,
    )
    journal.bind_contract()
    loaded = journal.load_eval_phase(spec)
    if loaded is None:
        phase_result = foundation.runner.run_eval_phase(spec)
        receipt_ref = journal.publish_eval_phase(spec, phase_result)
    else:
        phase_result, receipt_ref = loaded
    reloaded = load_trial_eval_phase_receipt(receipt_ref)
    if reloaded.result != phase_result:
        raise ValueError("P6-D final phase receipt replay mismatch")
    summary = canonical_stair_phase_result_summary(
        config=foundation.composition.task_plugin.config,
        expected_phase=EvaluationPhase.SEALED_VALIDATION,
        result=phase_result,
        expected_seed_namespace=STAIR_FINAL_CONFIRMATION_EVAL_NAMESPACE,
    )
    attempt_journal = _formal_final_attempt_journal(
        foundation,
        logical_job_id=spec.job_id,
    )
    if attempt_journal is None:
        raise ValueError("P6-D final phase has no attempt journal")
    attempt_result = attempt_journal.load_reusable_output_result()
    if (
        attempt_result is None
        or attempt_result.payload.get("outcome") != "completed"
        or attempt_result.payload.get("output_hash")
        != canonical_json_hash(phase_result.to_private_dict())
    ):
        raise ValueError("P6-D final phase attempt result binding mismatch")
    execution = StairFinalConfirmationPhaseExecution(
        role=role,
        job_id=spec.job_id,
        eval_phase_spec_hash=canonical_json_hash(spec.to_public_dict()),
        evaluation_plan_hash=canonical_json_hash(eval_plan.to_public_dict()),
        attempt_result_payload_hash=attempt_result.payload_hash,
        receipt_checksum=receipt_ref.checksum,
        resource_binding_hash=foundation.gpu_bindings[0].resource_binding_hash,
        checkpoint_hash=eval_plan.execution_checkpoint().checksum,
        generalization_summary_hash=canonical_json_hash(summary),
    )
    return execution, summary


def _build_formal_report(
    foundation: StairGenerationSessionFoundation,
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
    usage, charges, open_reservations = _collect_actual_cost_usage(
        foundation,
        genesis=genesis,
        ledger=ledger,
    )
    _assert_actual_cost_receipt_matches_usage(
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
    formal_cost_head = replay_stair_generation_formal_cost_chain(
        contract=foundation.cost_contract,
        actual_cost_receipts=StairGenerationActualCostJournal(
            foundation.config.output_dir,
            contract=foundation.cost_contract,
        ).load_chain(),
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
    _validate_stored_resource_closure_evidence(
        foundation,
        resource_evidence=resource_evidence,
        generation_index=STAIR_FINAL_CONFIRMATION_REQUIRED_GENERATIONS,
        include_final_confirmation=True,
    )
    post_genesis = StairGenerationActualCostJournal(
        foundation.config.output_dir,
        contract=foundation.cost_contract,
    ).receipt_for_event("post-genesis")
    post_parent = StairGenerationActualCostJournal(
        foundation.config.output_dir,
        contract=foundation.cost_contract,
    ).receipt_for_event("post-parent-bootstrap")
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
    retry_train_jobs, retry_eval_seed_units = _session_retry_usage(foundation)
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
        "normal_segment_topology_is_3_6_9_12_15": (
            topology["normal_closed_generation_boundaries"]
            == [
                *STAIR_FORMAL_GENERATION_PAUSE_BOUNDARIES,
                STAIR_FINAL_CONFIRMATION_REQUIRED_GENERATIONS,
            ]
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
    }
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise ValueError("P6-D formal report checks failed: " + ", ".join(failed))
    provider = _provider_actuals(foundation, ledger=ledger)
    return {
        "schema_version": STAIR_GENERATION_FORMAL_REPORT_SCHEMA_VERSION,
        "created_at_utc": created_at_utc,
        "session_id": foundation.config.session_id,
        "session_mode": FORMAL_GENERATION_SESSION_MODE,
        "source_commit": foundation.source_commit,
        "session_genesis_hash": genesis.session_genesis_hash,
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
            "post_genesis": _generation_cost_receipt_projection(post_genesis),
            "parent_bootstrap_workload": parent_workload,
            "generation_workloads": generation_workloads,
            "generation_terminal": _generation_cost_receipt_projection(
                generation_cost_receipt
            ),
            "final_confirmation_workload": final_workload,
            "final_terminal": _generation_cost_receipt_projection(
                final_cost_receipt
            ),
            "provider": provider,
            "retry_train_jobs": retry_train_jobs,
            "retry_eval_seed_units": retry_eval_seed_units,
            "conservative_charges": list(charges),
        },
        "exploration": _validation_exploration_metrics(
            foundation,
            genesis=genesis,
            ledger=ledger,
        ),
        "authoring": _validation_authoring_metrics(foundation),
        "final_confirmation": {
            "plan": plan.to_dict(),
            "reservation": reservation.to_dict(),
            "result": result.to_dict(),
            "result_checksum": result_checksum,
            "diagnostic_only": True,
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
    foundation: StairGenerationSessionFoundation,
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


def _validate_formal_final_result_receipts(
    foundation: StairGenerationSessionFoundation,
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
        ref = _trial_receipt_ref(paths[0], kind="eval_phase")
        if ref.checksum != execution.receipt_checksum:
            raise ValueError(f"P6-D {role} phase receipt checksum mismatch")
        loaded = load_trial_eval_phase_receipt(ref)
        phase_result = loaded.result
        summary = canonical_stair_phase_result_summary(
            config=foundation.composition.task_plugin.config,
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
    foundation: StairGenerationSessionFoundation,
    *,
    genesis: VerifiedSessionGenesis,
    ledger: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cost_journal = StairGenerationActualCostJournal(
        foundation.config.output_dir,
        contract=foundation.cost_contract,
    )
    post_parent = cost_journal.receipt_for_event("post-parent-bootstrap")
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
        loaded_record, execution_ref = _load_existing_generation_execution(
            foundation,
            record=record,
        )
        cost_ref = cost_journal.receipt_for_event(
            f"generation-{generation_index:03d}-settlement"
        )
        if cost_ref is None:
            raise ValueError("P6-D generation lacks its cost settlement")
        closure = _publish_or_validate_generation_closure(
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
                "cost_receipt": _generation_cost_receipt_projection(cost_ref),
                "workload_delta": workload,
            }
        )
    return details, workloads


def _build_formal_pause_result(
    foundation: StairGenerationSessionFoundation,
    *,
    genesis: VerifiedSessionGenesis,
    ledger: list[dict[str, Any]],
    segment_id: str,
    actual_cost_receipt: StairGenerationActualCostReceiptRef,
    closure: Mapping[str, Any],
) -> dict[str, Any]:
    generation_count = len(ledger)
    boundaries = set(
        foundation.continuation_policy.operator_pause_generation_boundaries
    )
    checks = {
        "mode_is_formal_generation_session": (
            foundation.session_contract.session_mode == FORMAL_GENERATION_SESSION_MODE
            and foundation.session_contract.max_generations
            == STAIR_FINAL_CONFIRMATION_REQUIRED_GENERATIONS
        ),
        "generation_boundary_is_precommitted_and_closed": (
            generation_count in boundaries
            and closure.get("generation_index") == generation_count
        ),
        "actual_cost_caps_passed": (
            actual_cost_receipt.payload["cap_checks"]["all_passed"] is True
        ),
        "acceptance_authority_is_false": (closure.get("acceptance_authority") is False),
    }
    if not all(checks.values()):
        raise ValueError("P6-D generation pause checks failed")
    return {
        "schema_version": STAIR_FORMAL_GENERATION_PAUSE_RESULT_SCHEMA_VERSION,
        "session_id": foundation.config.session_id,
        "session_mode": FORMAL_GENERATION_SESSION_MODE,
        "session_genesis_hash": genesis.session_genesis_hash,
        "generation_session_contract_hash": foundation.session_contract.contract_hash,
        "closed_generation_count": generation_count,
        "next_generation_index": generation_count + 1,
        "segment_id": segment_id,
        "ledger_snapshot_hash": canonical_json_hash(ledger),
        "actual_cost_receipt": _generation_cost_receipt_projection(
            actual_cost_receipt
        ),
        "generation_closure": dict(closure),
        "checks": checks,
        "acceptance_authority": False,
    }


def _build_validation_pause_result(
    foundation: StairGenerationSessionFoundation,
    *,
    genesis: VerifiedSessionGenesis,
    ledger: list[dict[str, Any]],
    segment_id: str,
    actual_cost_receipt: StairGenerationActualCostReceiptRef,
    closure: Mapping[str, Any],
) -> dict[str, Any]:
    generation_count = len(ledger)
    checks = {
        "mode_is_development_generation_validation": (
            foundation.session_contract.session_mode
            == DEVELOPMENT_GENERATION_VALIDATION_MODE
        ),
        "generation_boundary_is_closed": (
            generation_count in {1, 2}
            and closure.get("generation_index") == generation_count
        ),
        "actual_cost_caps_passed": (
            actual_cost_receipt.payload["cap_checks"]["all_passed"] is True
        ),
        "acceptance_authority_is_false": (closure.get("acceptance_authority") is False),
    }
    if not all(checks.values()):
        raise ValueError("bounded generation pause checks failed")
    return {
        "schema_version": STAIR_GENERATION_PAUSE_RESULT_SCHEMA_VERSION,
        "session_id": foundation.config.session_id,
        "session_mode": DEVELOPMENT_GENERATION_VALIDATION_MODE,
        "session_genesis_hash": genesis.session_genesis_hash,
        "generation_session_contract_hash": (foundation.session_contract.contract_hash),
        "closed_generation_count": generation_count,
        "next_generation_index": generation_count + 1,
        "segment_id": segment_id,
        "ledger_snapshot_hash": canonical_json_hash(ledger),
        "actual_cost_receipt": _generation_cost_receipt_projection(actual_cost_receipt),
        "generation_closure": dict(closure),
        "checks": checks,
        "acceptance_authority": False,
    }


@dataclass(frozen=True)
class _StairOperationAllocationState:
    reserved_success_slots: frozenset[str]
    reserved_retry_classes: frozenset[str]
    attempt_results: Mapping[str, Mapping[str, Any] | None]
    measured_gpu_capacity_ns: int
    conservatively_charged_gpu_capacity_ns: int
    unmeasured_bound_attempts: int
    open_attempt_output_reserve_bytes: int


def _strict_operation_reserves_by_class(
    cost_contract: StairGenerationSessionCostContractV2,
) -> dict[str, StairGenerationOperationCostReserve]:
    frozen = (
        stair_formal_generation_operation_cost_reserves()
        if isinstance(cost_contract, StairGenerationSessionCostContractV3)
        else stair_generation_operation_cost_reserves()
    )
    result = {item.operation_class: item for item in frozen}
    if len(result) != len(frozen):
        raise ValueError("strict operation reserve classes are duplicated")
    return result


def _strict_operation_slot(
    *,
    session_id: str,
    scope: JobAttemptScope | PreGenesisExecutionScope,
    logical_job_id: str,
    job_kind: str,
    attempt_index: int,
    cost_contract: StairGenerationSessionCostContractV2,
) -> tuple[str, StairGenerationOperationCostReserve]:
    reserves = _strict_operation_reserves_by_class(cost_contract)
    max_generations = cost_contract.max_generations
    expected_retries = 0
    retry_class: str | None = None
    if logical_job_id == f"{session_id}-train":
        if (
            not isinstance(scope, PreGenesisExecutionScope)
            or scope.session_id != session_id
            or scope.lane_id != "lane-a"
            or job_kind != "train"
        ):
            raise ValueError("Stair bounded generation reference-train attempt binding is invalid")
        success_slot = "reference_train"
        success_class = "reference_train"
        retry_class = "reference_train_retry"
        expected_retries = 1
    elif logical_job_id == f"{session_id}-eval":
        if (
            not isinstance(scope, PreGenesisExecutionScope)
            or scope.session_id != session_id
            or scope.lane_id != "lane-a"
            or job_kind != "eval"
        ):
            raise ValueError("Stair bounded generation reference-eval attempt binding is invalid")
        success_slot = "reference_qualification"
        success_class = "reference_qualification"
    elif logical_job_id == f"{session_id}-eval-public-dev":
        if (
            not isinstance(scope, JobAttemptScope)
            or scope.generation_id != "generation-parent-public-dev-bootstrap"
            or scope.lane_id != "lane-a"
            or job_kind != "eval_phase"
        ):
            raise ValueError("Stair bounded generation parent-bootstrap attempt binding is invalid")
        success_slot = "parent_public_dev"
        success_class = "parent_public_dev"
    elif (
        isinstance(cost_contract, StairGenerationSessionCostContractV3)
        and logical_job_id
        in {
            f"{session_id}-final-confirmation-genesis-eval-sealed-validation",
            f"{session_id}-final-confirmation-final-eval-sealed-validation",
        }
    ):
        if (
            not isinstance(scope, JobAttemptScope)
            or scope.generation_id != "generation-final-confirmation"
            or scope.lane_id != "lane-a"
            or job_kind != "eval_phase"
        ):
            raise ValueError("formal final confirmation attempt binding is invalid")
        role = (
            "genesis" if "-genesis-" in logical_job_id else "final"
        )
        success_slot = f"final-confirmation:{role}"
        success_class = "sealed_validation"
        retry_class = "sealed_validation_retry"
        expected_retries = 1
    else:
        match = re.fullmatch(
            re.escape(session_id) + r"-generation-([0-9]{3})-lane-(a|b)-"
            r"(train|eval-public-dev|eval-public-accept|eval-sealed-validation)",
            logical_job_id,
        )
        if match is None or not isinstance(scope, JobAttemptScope):
            raise ValueError("Stair bounded generation attempt has no frozen operation allocation")
        generation_text, lane_suffix, operation_suffix = match.groups()
        generation_index = int(generation_text)
        if not 1 <= generation_index <= max_generations:
            raise ValueError("generation attempt exceeds its frozen session")
        lane_id = f"lane-{lane_suffix}"
        generation_id = f"{session_id}-generation-{generation_index:03d}"
        if scope.generation_id != generation_id or scope.lane_id != lane_id:
            raise ValueError("bounded generation attempt scope is misbound")
        if operation_suffix == "train":
            expected_kind = "train"
            success_class = f"lane_{lane_suffix}_train"
            success_slot = f"generation-{generation_index:03d}:{lane_id}:train"
        elif operation_suffix == "eval-public-dev":
            expected_kind = "eval_phase"
            success_class = f"lane_{lane_suffix}_public_dev"
            success_slot = f"generation-{generation_index:03d}:{lane_id}:public_dev"
        elif operation_suffix == "eval-public-accept":
            expected_kind = "eval_phase"
            success_class = f"lane_{lane_suffix}_public_accept"
            success_slot = f"generation-{generation_index:03d}:{lane_id}:public_accept"
        else:
            expected_kind = "eval_phase"
            success_class = "sealed_validation"
            success_slot = f"generation-{generation_index:03d}:sealed_validation"
            retry_class = "sealed_validation_retry"
            expected_retries = 1
        if job_kind != expected_kind:
            raise ValueError("Stair bounded generation attempt job kind differs from its operation")
    if scope.max_infrastructure_retries != expected_retries:
        raise ValueError("Stair bounded generation attempt retry authority differs from its operation")
    if attempt_index == 0:
        return success_slot, reserves[success_class]
    if attempt_index == 1 and retry_class is not None:
        return retry_class, reserves[retry_class]
    raise ValueError("Stair bounded generation attempt index has no frozen operation allocation")


def _strict_operation_allocation_state(
    *,
    output_dir: Path,
    session_id: str,
    cost_contract: StairGenerationSessionCostContractV2,
    session_contract_hash: str,
) -> _StairOperationAllocationState:
    attempt_root = output_dir / "job_attempts"
    success_slots: set[str] = set()
    retry_classes: set[str] = set()
    results: dict[str, Mapping[str, Any] | None] = {}
    measured_ns = 0
    conservative_ns = 0
    unmeasured = 0
    open_output_reserve = 0
    if not attempt_root.exists():
        return _StairOperationAllocationState(
            reserved_success_slots=frozenset(),
            reserved_retry_classes=frozenset(),
            attempt_results={},
            measured_gpu_capacity_ns=0,
            conservatively_charged_gpu_capacity_ns=0,
            unmeasured_bound_attempts=0,
            open_attempt_output_reserve_bytes=0,
        )
    if attempt_root.is_symlink() or not attempt_root.is_dir():
        raise ValueError("Stair bounded generation job-attempt root is invalid")
    for path in sorted(attempt_root.glob("*/attempt_*.reservation.json")):
        journal, reservation = JobAttemptJournal.from_reservation_path(path)
        scope = journal.scope
        if (
            scope.budget_contract_hash != cost_contract.contract_hash
            or scope.session_contract_hash != session_contract_hash
        ):
            raise ValueError("Stair bounded generation attempt belongs to another budget authority")
        attempt_index = reservation.payload["attempt_index"]
        slot, reserve = _strict_operation_slot(
            session_id=session_id,
            scope=scope,
            logical_job_id=journal.logical_job_id,
            job_kind=journal.job_kind,
            attempt_index=attempt_index,
            cost_contract=cost_contract,
        )
        if reserve.budget_class == "successful_session":
            if slot in success_slots:
                raise ValueError("Stair bounded generation successful operation slot is duplicated")
            success_slots.add(slot)
        else:
            if slot in retry_classes:
                raise ValueError("Stair bounded generation session-wide retry allocation is duplicated")
            retry_classes.add(slot)
        result_path = journal.root / f"attempt_{attempt_index:02d}.result.json"
        result_payload: Mapping[str, Any] | None = None
        if result_path.exists():
            result_payload = journal.load_result(attempt_index).payload
        else:
            open_output_reserve += reserve.output_bytes_per_attempt
        results[slot] = result_payload
        if (
            result_payload is not None
            and result_payload.get("schema_version")
            == RESOURCE_LEASE_ATTEMPT_RESULT_SCHEMA_VERSION
        ):
            measured_ns += result_payload["resource_lease_wall_duration_ns"]
        else:
            unmeasured += 1
            conservative_ns += reserve.allocated_gpu_capacity_ns_per_attempt
    return _StairOperationAllocationState(
        reserved_success_slots=frozenset(success_slots),
        reserved_retry_classes=frozenset(retry_classes),
        attempt_results=results,
        measured_gpu_capacity_ns=measured_ns,
        conservatively_charged_gpu_capacity_ns=conservative_ns,
        unmeasured_bound_attempts=unmeasured,
        open_attempt_output_reserve_bytes=open_output_reserve,
    )


def _strict_success_operation_slots(
    cost_contract: StairGenerationSessionCostContractV2,
) -> dict[str, StairGenerationOperationCostReserve]:
    reserves = _strict_operation_reserves_by_class(cost_contract)
    slots = {
        "reference_train": reserves["reference_train"],
        "reference_qualification": reserves["reference_qualification"],
        "parent_public_dev": reserves["parent_public_dev"],
    }
    max_generations = cost_contract.max_generations
    for generation_index in range(1, max_generations + 1):
        for lane_suffix in ("a", "b"):
            lane_id = f"lane-{lane_suffix}"
            for suffix, operation_class in (
                ("train", f"lane_{lane_suffix}_train"),
                ("public_dev", f"lane_{lane_suffix}_public_dev"),
                ("public_accept", f"lane_{lane_suffix}_public_accept"),
            ):
                slots[f"generation-{generation_index:03d}:{lane_id}:{suffix}"] = (
                    reserves[operation_class]
                )
        slots[f"generation-{generation_index:03d}:sealed_validation"] = reserves[
            "sealed_validation"
        ]
    if isinstance(cost_contract, StairGenerationSessionCostContractV3):
        slots["final-confirmation:genesis"] = reserves["sealed_validation"]
        slots["final-confirmation:final"] = reserves["sealed_validation"]
    expected_counts: dict[str, int] = {}
    for reserve in slots.values():
        expected_counts[reserve.operation_class] = (
            expected_counts.get(reserve.operation_class, 0) + 1
        )
    frozen_counts = {
        reserve.operation_class: reserve.multiplicity
        for reserve in reserves.values()
        if reserve.budget_class == "successful_session"
    }
    if expected_counts != frozen_counts:
        raise ValueError("Stair bounded generation operation slot inventory differs from its cost contract")
    return slots


def _strict_retry_is_still_possible(
    *,
    operation_class: str,
    closed_generation_count: int,
    state: _StairOperationAllocationState,
    cost_contract: StairGenerationSessionCostContractV2,
) -> bool:
    if operation_class == "reference_train_retry":
        if operation_class in state.reserved_retry_classes:
            return False
        result = state.attempt_results.get("reference_train")
        return (
            "reference_train" not in state.reserved_success_slots
            or result is None
            or (
                result.get("outcome") == "infrastructure_failure"
                and result.get("retryable") is True
            )
        )
    if operation_class != "sealed_validation_retry":
        raise ValueError("Stair bounded generation retry operation class is unknown")
    if operation_class in state.reserved_retry_classes:
        return False
    slots = _strict_success_operation_slots(cost_contract)
    future_slot_exists = any(
        slot.endswith(":sealed_validation")
        and slot not in state.reserved_success_slots
        and int(slot.split(":", 1)[0].split("-")[1]) > closed_generation_count
        for slot in slots
    )
    retryable_active_slot_exists = any(
        slot.endswith(":sealed_validation")
        and (
            result is None
            or (
                result.get("outcome") == "infrastructure_failure"
                and result.get("retryable") is True
            )
        )
        for slot, result in state.attempt_results.items()
    )
    final_slot_exists = any(
        slot.startswith("final-confirmation:")
        and slot not in state.reserved_success_slots
        for slot in slots
    )
    return (
        future_slot_exists
        or retryable_active_slot_exists
        or final_slot_exists
    )


def _strict_remaining_operation_reserves(
    *,
    closed_generation_count: int,
    state: _StairOperationAllocationState,
    cost_contract: StairGenerationSessionCostContractV2,
) -> tuple[StairGenerationOperationCostReserve, ...]:
    slots = _strict_success_operation_slots(cost_contract)
    remaining: list[StairGenerationOperationCostReserve] = []
    for slot, reserve in slots.items():
        if slot in state.reserved_success_slots:
            continue
        generation_match = re.match(r"generation-([0-9]{3}):", slot)
        if (
            generation_match is not None
            and int(generation_match.group(1)) <= closed_generation_count
        ):
            continue
        remaining.append(reserve)
    reserves = _strict_operation_reserves_by_class(cost_contract)
    for retry_class in ("reference_train_retry", "sealed_validation_retry"):
        if _strict_retry_is_still_possible(
            operation_class=retry_class,
            closed_generation_count=closed_generation_count,
            state=state,
            cost_contract=cost_contract,
        ):
            remaining.append(reserves[retry_class])
    return tuple(remaining)


def _strict_operation_reservation_checks(
    *,
    output_dir: Path,
    session_id: str,
    cost_contract: StairGenerationSessionCostContractV2,
    session_contract_hash: str,
    ledger: list[dict[str, Any]],
) -> tuple[dict[str, bool], _StairOperationAllocationState]:
    state = _strict_operation_allocation_state(
        output_dir=output_dir,
        session_id=session_id,
        cost_contract=cost_contract,
        session_contract_hash=session_contract_hash,
    )
    remaining = _strict_remaining_operation_reserves(
        closed_generation_count=len(ledger),
        state=state,
        cost_contract=cost_contract,
    )
    output_bytes, _ = _output_tree_size(output_dir)
    checks = {
        "allocated_active_gpu_hours": (
            state.measured_gpu_capacity_ns
            + state.conservatively_charged_gpu_capacity_ns
            + sum(item.allocated_gpu_capacity_ns_per_attempt for item in remaining)
            <= int(cost_contract.max_allocated_active_gpu_hours * 3_600_000_000_000)
        ),
        "output_bytes": (
            output_bytes
            + state.open_attempt_output_reserve_bytes
            + STAIR_GENERATION_FINALIZATION_OUTPUT_RESERVE_BYTES
            + sum(item.output_bytes_per_attempt for item in remaining)
            <= cost_contract.max_output_bytes
        ),
    }
    return checks, state


@dataclass(frozen=True)
class _StairOperationAdmission:
    output_dir: Path
    session_id: str
    cost_contract: StairGenerationSessionCostContractV2
    session_contract_hash: str
    termination_grace_s: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.termination_grace_s, bool)
            or not isinstance(self.termination_grace_s, int)
            or self.termination_grace_s <= 0
        ):
            raise ValueError("Stair bounded generation operation termination grace must be positive")

    def __call__(
        self,
        scope: JobAttemptScope | PreGenesisExecutionScope,
        logical_job_id: str,
        job_kind: str,
        attempt_index: int,
    ) -> int:
        if (
            scope.budget_contract_hash != self.cost_contract.contract_hash
            or scope.session_contract_hash != self.session_contract_hash
        ):
            raise ValueError("Stair bounded generation attempt admission authority is misbound")
        ledger_path = self.output_dir / "ledger.jsonl"
        ledger = load_ledger(ledger_path) if ledger_path.exists() else []
        slot, reserve = _strict_operation_slot(
            session_id=self.session_id,
            scope=scope,
            logical_job_id=logical_job_id,
            job_kind=job_kind,
            attempt_index=attempt_index,
            cost_contract=self.cost_contract,
        )
        checks, state = _strict_operation_reservation_checks(
            output_dir=self.output_dir,
            session_id=self.session_id,
            cost_contract=self.cost_contract,
            session_contract_hash=self.session_contract_hash,
            ledger=ledger,
        )
        if reserve.budget_class == "successful_session":
            if slot in state.reserved_success_slots:
                raise ValueError("Stair bounded generation operation slot was already reserved")
            generation_match = re.match(r"generation-([0-9]{3}):", slot)
            if generation_match is not None and int(generation_match.group(1)) <= len(
                ledger
            ):
                raise ValueError("Stair bounded generation closed generation cannot start new work")
        elif not _strict_retry_is_still_possible(
            operation_class=reserve.operation_class,
            closed_generation_count=len(ledger),
            state=state,
            cost_contract=self.cost_contract,
        ):
            raise ValueError("Stair bounded generation session-wide retry reserve is unavailable")
        if not all(checks.values()):
            failed = sorted(name for name, passed in checks.items() if not passed)
            raise RuntimeError(
                "Stair bounded generation operation admission cannot reserve frozen work: "
                + ", ".join(failed)
            )
        # Timeout cleanup can spend one grace window on TERM, one after KILL,
        # and one proving process-group quiescence.  Keep those windows plus
        # one second inside this operation's frozen one-GPU wall-time slot.
        cleanup_margin_s = 3 * max(self.termination_grace_s, 1) + 1
        slot_wall_time_s = (
            reserve.allocated_gpu_capacity_ns_per_attempt // 1_000_000_000
        )
        timeout_s = slot_wall_time_s - cleanup_margin_s
        if timeout_s <= 0:
            raise RuntimeError("Stair bounded generation operation slot cannot fund process cleanup")
        return timeout_s


def _assert_generation_admission(
    foundation: StairGenerationSessionFoundation,
    *,
    genesis: VerifiedSessionGenesis,
    ledger: list[dict[str, Any]],
    generation_index: int,
) -> None:
    """Reserve remaining worst-case work before a bounded generation starts."""

    if generation_index != len(ledger) + 1:
        raise ValueError("Stair bounded generation admission generation index is not contiguous")
    generation_id = f"{foundation.config.session_id}-generation-{generation_index:03d}"
    usage, _, _ = _collect_actual_cost_usage(
        foundation,
        genesis=genesis,
        ledger=ledger,
    )
    if not isinstance(foundation.cost_contract, StairGenerationSessionCostContractV2):
        raise TypeError("Stair bounded generation admission requires the v2 physical-cost contract")
    operation_checks, _ = _strict_operation_reservation_checks(
        output_dir=foundation.config.output_dir,
        session_id=foundation.config.session_id,
        cost_contract=foundation.cost_contract,
        session_contract_hash=foundation.session_contract.contract_hash,
        ledger=ledger,
    )
    completed_train_jobs, completed_eval_seed_units = (
        _completed_current_generation_operation_cost(
            foundation,
            generation_id=generation_id,
        )
    )
    scientific = foundation.cost_contract.scientific_cost_basis
    remaining_train_jobs = scientific.train_jobs_per_generation - completed_train_jobs
    remaining_eval_seed_units = (
        scientific.eval_seed_units_per_generation - completed_eval_seed_units
    )
    if remaining_train_jobs < 0 or remaining_eval_seed_units < 0:
        raise ValueError("Stair bounded generation partial generation exceeds its scientific plan")
    provider_attempts_in_generation = _generation_provider_attempt_count(
        foundation.config.output_dir,
        generation_id=generation_id,
    )
    provider_attempts_per_generation = (
        len(GENERATION_LANE_IDS) * STAIR_GENERATION_PROVIDER_ATTEMPTS_PER_LANE
    )
    if provider_attempts_in_generation > provider_attempts_per_generation:
        raise ValueError("Stair bounded generation partial generation exceeds its provider plan")
    remaining_provider_attempts = (
        provider_attempts_per_generation - provider_attempts_in_generation
    )
    provider_plan = foundation.cost_contract.to_dict()["provider"]
    remaining_train_steps = remaining_train_jobs * scientific.train_steps_per_job
    remaining_control_steps = remaining_train_steps + (
        remaining_eval_seed_units * scientific.max_eval_control_steps_per_seed_unit
    )
    checks = {
        "provider_attempts": (
            usage.provider_attempts + remaining_provider_attempts
            <= provider_plan["attempt_cap"]
        ),
        "provider_request_units": (
            usage.provider_request_units
            + remaining_provider_attempts
            * STAIR_GENERATION_PROVIDER_INPUT_UNITS_PER_ATTEMPT
            <= provider_plan["input_unit_cap"]
        ),
        "provider_response_units": (
            usage.provider_response_units
            + remaining_provider_attempts
            * STAIR_GENERATION_PROVIDER_OUTPUT_UNITS_PER_ATTEMPT
            <= provider_plan["output_unit_cap"]
        ),
        "provider_total_units": (
            usage.provider_total_units
            + remaining_provider_attempts
            * (
                STAIR_GENERATION_PROVIDER_INPUT_UNITS_PER_ATTEMPT
                + STAIR_GENERATION_PROVIDER_OUTPUT_UNITS_PER_ATTEMPT
            )
            <= provider_plan["total_unit_cap"]
        ),
        "train_jobs": (
            usage.train_jobs + remaining_train_jobs
            <= foundation.cost_contract.physical_train_job_cap
        ),
        "train_steps": (
            usage.train_steps + remaining_train_steps
            <= foundation.cost_contract.physical_train_step_cap
        ),
        "eval_seed_units": (
            usage.eval_seed_units + remaining_eval_seed_units
            <= foundation.cost_contract.physical_eval_seed_unit_cap
        ),
        "simulated_control_steps": (
            usage.simulated_control_steps + remaining_control_steps
            <= foundation.cost_contract.max_simulated_control_steps
        ),
        "allocated_active_gpu_hours": operation_checks["allocated_active_gpu_hours"],
        "output_bytes": operation_checks["output_bytes"],
    }
    _remaining_sealed_retry_entitlement(foundation)
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise RuntimeError(
            "bounded generation admission cannot reserve frozen work: "
            + ", ".join(failed)
        )


def _completed_current_generation_operation_cost(
    foundation: StairGenerationSessionFoundation,
    *,
    generation_id: str,
) -> tuple[int, int]:
    attempt_root = foundation.config.output_dir / "job_attempts"
    completed_train_jobs: set[str] = set()
    completed_eval_seed_units: dict[str, int] = {}
    if not attempt_root.exists():
        return 0, 0
    if attempt_root.is_symlink() or not attempt_root.is_dir():
        raise ValueError("job-attempt root is invalid")
    for path in sorted(attempt_root.glob("*/attempt_*.reservation.json")):
        journal, reservation = JobAttemptJournal.from_reservation_path(path)
        scope = journal.scope
        if (
            not isinstance(scope, JobAttemptScope)
            or scope.generation_id != generation_id
        ):
            continue
        if (
            scope.budget_contract_hash != foundation.cost_contract.contract_hash
            or scope.session_contract_hash != foundation.session_contract.contract_hash
        ):
            raise ValueError("partial generation attempt belongs to another contract")
        result_path = path.with_name(path.name.replace(".reservation.", ".result."))
        if not result_path.is_file():
            continue
        result = journal.load_result(reservation.payload["attempt_index"])
        if result.payload.get("outcome") != "completed":
            continue
        if journal.job_kind == "train":
            if journal.logical_job_id in completed_train_jobs:
                raise ValueError("partial generation train completed more than once")
            completed_train_jobs.add(journal.logical_job_id)
        elif journal.job_kind in {"eval", "eval_phase", "baseline_eval"}:
            if journal.logical_job_id in completed_eval_seed_units:
                raise ValueError("partial generation eval completed more than once")
            completed_eval_seed_units[journal.logical_job_id] = (
                _eval_seed_units_for_job(
                    journal.logical_job_id,
                    journal.job_kind,
                    cost_contract=foundation.cost_contract,
                )
            )
        else:
            raise ValueError("unknown Stair job-attempt kind")
    return len(completed_train_jobs), sum(completed_eval_seed_units.values())


def _generation_provider_attempt_count(
    output_dir: Path,
    *,
    generation_id: str,
) -> int:
    count = 0
    for lane_id in GENERATION_LANE_IDS:
        round_dir = output_dir / "authoring" / generation_id / lane_id
        if not round_dir.exists():
            continue
        if round_dir.is_symlink() or not round_dir.is_dir():
            raise ValueError("provider authoring directory is invalid")
        count += len(
            RoundAuthoringJournal(
                round_dir,
                round_id=f"{generation_id}-{lane_id}",
            )
            .snapshot()
            .attempts
        )
    return count


def _build_generation_inputs(
    foundation: StairGenerationSessionFoundation,
    *,
    genesis: VerifiedSessionGenesis,
    guard: Any,
    segment_id: str,
    generation_index: int = 1,
    parent: Any | None = None,
) -> tuple[FixedWidthGenerationInputs, GenerationExplorationPlan]:
    composition = foundation.composition
    contract = foundation.session_contract
    ledger_path = foundation.config.output_dir / "ledger.jsonl"
    ledger_epoch_hash = primary_ledger_epoch_hash(genesis.session_genesis_hash)
    ledger = load_ledger(ledger_path)
    exploration = _generation_exploration_for_ledger(
        foundation,
        genesis=genesis,
        ledger=ledger,
    )
    records = [ExperimentRecord.from_ledger_record(record) for record in ledger]
    context_builder = PublicContextBuilder(
        task_memory_adapter=composition.task_memory_adapter,
        task_prompt_adapter=composition.task_plugin.prompt_adapter(),
        acceptance_protocol=composition.task_plugin.task_spec.acceptance_protocol,
        initial_current_best=genesis.initial_current_best,
        session_genesis_hash=genesis.session_genesis_hash,
        candidate_capability_profile=(
            composition.task_plugin.task_spec.candidate_capability_profile
        ),
        initial_reference_program_source=genesis.reference_program_source(),
        initial_anchor_public_evidence=genesis.initial_anchor_public_evidence(),
        expected_initial_policy_state_hash=(genesis.expected_initial_policy_state_hash),
        ledger_authoring_mode=PROVIDER_AUTHORING_MODE,
        expected_ledger_epoch_hash=ledger_epoch_hash,
        generation_session_contract=contract,
    )
    base_context = context_builder.build(records)
    if (
        isinstance(generation_index, bool)
        or not isinstance(generation_index, int)
        or generation_index <= 0
        or generation_index > contract.max_generations
    ):
        raise ValueError("generation index exceeds the frozen session contract")
    generation_id = f"{foundation.config.session_id}-generation-{generation_index:03d}"
    generation_scoped_paths = contract.max_generations > 1
    train_scopes = tuple(
        _job_attempt_scope(
            foundation,
            genesis=genesis,
            generation_id=generation_id,
            lane_id=lane_id,
            invocation_segment_id=segment_id,
            binding=binding,
        )
        for lane_id, binding in zip(
            GENERATION_LANE_IDS,
            foundation.gpu_bindings,
            strict=True,
        )
    )
    public_dev_scopes = tuple(
        _job_attempt_scope(
            foundation,
            genesis=genesis,
            generation_id=generation_id,
            lane_id=lane_id,
            invocation_segment_id=segment_id,
            binding=binding,
        )
        for lane_id, binding in zip(
            GENERATION_LANE_IDS,
            foundation.gpu_bindings,
            strict=True,
        )
    )
    public_accept_scopes = tuple(
        _job_attempt_scope(
            foundation,
            genesis=genesis,
            generation_id=generation_id,
            lane_id=lane_id,
            invocation_segment_id=segment_id,
            binding=binding,
        )
        for lane_id, binding in zip(
            GENERATION_LANE_IDS,
            foundation.gpu_bindings,
            strict=True,
        )
    )
    sealed_retry_entitlement = _remaining_sealed_retry_entitlement(foundation)
    sealed_scopes = tuple(
        _job_attempt_scope(
            foundation,
            genesis=genesis,
            generation_id=generation_id,
            lane_id=lane_id,
            invocation_segment_id=segment_id,
            binding=binding,
            max_infrastructure_retries=sealed_retry_entitlement,
        )
        for lane_id, binding in zip(
            GENERATION_LANE_IDS,
            foundation.gpu_bindings,
            strict=True,
        )
    )
    operation_scopes = tuple(
        recover_stair_generation_lane_operation_scopes(
            attempt_journal_root=(foundation.config.output_dir / "job_attempts"),
            generation_id=generation_id,
            lane_id=lane_id,
            current_invocation_segment_id=segment_id,
            current_scopes=StairGenerationLaneOperationScopes(
                train=train_scopes[index],
                public_dev=public_dev_scopes[index],
                public_accept=public_accept_scopes[index],
                sealed_validation=sealed_scopes[index],
            ),
        )
        for index, lane_id in enumerate(GENERATION_LANE_IDS)
    )
    assignments = tuple(
        GenerationLaneAssignment(
            lane_id=lane_id,
            execution_device=binding.process_execution_device,
            resource_binding_hash=binding.resource_binding_hash,
            lane_intent_hash=canonical_json_hash(intent),
            gpu_process_binding=binding,
        )
        for lane_id, binding, intent in zip(
            GENERATION_LANE_IDS,
            foundation.gpu_bindings,
            exploration.lane_intents,
            strict=True,
        )
    )
    if parent is None:
        parent = genesis.initial_current_best
    guard.assert_matches_parent(parent)
    manifest = GenerationManifest(
        schema_version=GENERATION_MANIFEST_V2_SCHEMA_VERSION,
        generation_id=generation_id,
        generation_index=generation_index,
        session_genesis_hash=genesis.session_genesis_hash,
        generation_session_contract_hash=contract.contract_hash,
        ledger_epoch_hash=ledger_epoch_hash,
        parent_program_source_checksum=parent.program_source_checksum,
        parent_checkpoint_hash=parent.checkpoint_hash,
        current_best_evidence_hash=canonical_json_hash(parent.to_public_dict()),
        base_public_context_hash=canonical_json_hash(base_context),
        evaluation_binding_hash=genesis.evaluation_binding.binding_hash,
        training_initialization_hash=generation_training_initialization_hash(
            policy_initialization_kind=composition.policy_initialization.kind,
            training_binding_hash=parent.training_binding_hash,
            expected_initial_policy_state_hash=(
                genesis.expected_initial_policy_state_hash
            ),
        ),
        guard_reference_evidence_hash=guard.evidence.evidence_hash,
        guard_reference_phase_receipt_hash=canonical_json_hash(
            guard.phase_receipt_ref.to_dict()
        ),
        lanes=assignments,  # type: ignore[arg-type]
    )
    providers = {
        lane_id: _StairCostGuardedProvider(
            delegate=OpenAICompatibleCandidateProvider(
                _stair_generation_provider_config(
                    output_dir=(
                        foundation.config.output_dir
                        / "provider_candidates"
                        / (generation_id if generation_scoped_paths else "")
                        / lane_id
                    ),
                    env_file=foundation.config.provider_env_file,
                )
            ),
            output_dir=foundation.config.output_dir,
            generation_id=generation_id,
            lane_id=lane_id,
            max_generations=contract.max_generations,
            allow_conservative_interrupted_recovery=isinstance(
                foundation.cost_contract,
                StairGenerationSessionCostContractV2,
            ),
        )
        for lane_id in GENERATION_LANE_IDS
    }
    readiness = composition.authoring_capability_readiness.to_prompt_mapping()
    smoke_adapter = composition.task_plugin.component_smoke_adapter()

    def author(lane_id: str, exclusion: Any) -> Any:
        lane_index = GENERATION_LANE_IDS.index(lane_id)
        trial_id = f"{generation_id}-{lane_id}"
        authoring_loop = CandidateAuthoringLoop(
            context_builder=context_builder,
            provider=providers[lane_id],
            candidate_admission_policy=composition.candidate_admission_policy,
            round_dir=(
                foundation.config.output_dir / "authoring" / generation_id / lane_id
            ),
            max_attempts=STAIR_GENERATION_PROVIDER_ATTEMPTS_PER_LANE,
            component_smoke_adapter=smoke_adapter,
            authoring_capability_readiness=readiness,
            sibling_semantic_exclusion=exclusion,
            generation_lane_intent=exploration.lane_intents[lane_index],
            blocked_verifier_equivalence_keys=(
                exploration.blocked_verifier_equivalence_keys
            ),
        )
        if not generation_scoped_paths:
            return authoring_loop.run_round(round_id=trial_id, records=records)
        return run_stair_generation_authoring_attempt(
            evidence_root=(
                foundation.config.output_dir / STAIR_GENERATION_AUTHORING_TIMING_DIRNAME
            ),
            generation_id=generation_id,
            lane_id=lane_id,
            author=lambda: authoring_loop.run_round(
                round_id=trial_id,
                records=records,
            ),
        )

    def round_inputs(lane_id: str, authored: Any) -> RoundInputs:
        lane_index = GENERATION_LANE_IDS.index(lane_id)
        binding = foundation.gpu_bindings[lane_index]
        scopes = operation_scopes[lane_index]
        trial_id = f"{generation_id}-{lane_id}"
        if authored.candidate is None:
            raise ValueError("verified generation authoring result has no candidate")
        return RoundInputs(
            round_id=trial_id,
            task_plugin=composition.task_plugin,
            backend_adapter=composition.backend_adapter,
            runner=foundation.runner,
            candidate_admission_policy=composition.candidate_admission_policy,
            candidate=authored.candidate,
            policy_initialization=composition.policy_initialization,
            ledger_path=(
                foundation.config.output_dir
                / "non_authoritative_lane_ledgers"
                / (generation_id if generation_scoped_paths else "")
                / f"{lane_id}.jsonl"
            ),
            public_context=authored.round_public_context,
            current_best_eval_stats=parent,
            session_genesis_hash=genesis.session_genesis_hash,
            evaluation_binding=genesis.evaluation_binding,
            authoring_mode=PROVIDER_AUTHORING_MODE,
            prompt_payload_ref=authored.prompt_payload_ref,
            authoring_evidence_ref=authored.authoring_evidence_ref,
            record_candidate_runtime_failures=True,
            candidate_artifact_root=(
                foundation.config.output_dir
                / "candidate_artifacts"
                / (generation_id if generation_scoped_paths else "")
                / lane_id
            ),
            expected_initial_policy_state_hash=(
                genesis.expected_initial_policy_state_hash
            ),
            parent_program_source_checksum=parent.program_source_checksum,
            ledger_epoch_hash=ledger_epoch_hash,
            backend_execution_environment_manifest_hash=(
                genesis.backend_execution_environment_manifest_hash
            ),
            trial_receipt_root=(
                foundation.config.output_dir
                / "trial_receipts"
                / generation_id
                / lane_id
            ),
            train_attempt_scope=scopes.train,
            eval_attempt_scope=scopes.public_dev,
            gpu_process_binding=binding,
        )

    lanes = tuple(
        GenerationLaneRuntime(
            lane_id=lane_id,
            trial_id=f"{generation_id}-{lane_id}",
            lane_intent=exploration.lane_intents[index],
            lane_intent_hash=canonical_json_hash(exploration.lane_intents[index]),
            author=(lambda exclusion, lane_id=lane_id: author(lane_id, exclusion)),
            build_round_inputs=(
                lambda authored, lane_id=lane_id: round_inputs(lane_id, authored)
            ),
            **operation_scopes[index].to_generation_lane_runtime_kwargs(),
        )
        for index, lane_id in enumerate(GENERATION_LANE_IDS)
    )
    return (
        FixedWidthGenerationInputs(
            manifest=manifest,
            session_contract=contract,
            lanes=lanes,  # type: ignore[arg-type]
            guard_reference=guard,
            phased_adapter=foundation.phased_adapter,
            receipt_root=(
                foundation.config.output_dir / "generation_receipts" / generation_id
            ),
            ledger_path=ledger_path,
            parent_current_best=parent,
            initial_current_best=genesis.initial_current_best,
            acceptance_protocol=composition.task_plugin.task_spec.acceptance_protocol,
        ),
        exploration,
    )


def _job_attempt_scope(
    foundation: StairGenerationSessionFoundation,
    *,
    genesis: VerifiedSessionGenesis,
    generation_id: str,
    lane_id: str,
    invocation_segment_id: str,
    binding: GpuProcessBinding,
    max_infrastructure_retries: int = 0,
) -> JobAttemptScope:
    return JobAttemptScope(
        journal_root=foundation.config.output_dir / "job_attempts",
        session_genesis_hash=genesis.session_genesis_hash,
        session_contract_hash=foundation.session_contract.contract_hash,
        generation_id=generation_id,
        lane_id=lane_id,
        invocation_segment_id=invocation_segment_id,
        budget_contract_hash=foundation.cost_contract.contract_hash,
        attempt_budget_units=max_infrastructure_retries + 1,
        max_infrastructure_retries=max_infrastructure_retries,
        resource_binding_hash=binding.resource_binding_hash,
    )


def _session_genesis_attempt_scope(**kwargs: Any) -> PreGenesisExecutionScope:
    retries = kwargs.pop("max_infrastructure_retries")
    return PreGenesisExecutionScope(
        **kwargs,
        attempt_budget_units=retries + 1,
        max_infrastructure_retries=retries,
    )


def _latest_cost_head(journal: StairGenerationActualCostJournal) -> str:
    chain = journal.load_chain()
    return chain[-1].payload_hash if chain else journal.contract.contract_hash


def _replay_validation_combined_cost_chain(
    cost_journal: StairGenerationActualCostJournal,
    lifecycle: GenerationLifecycleAudit,
) -> str:
    return replay_stair_generation_combined_cost_chain(
        contract=cost_journal.contract,
        actual_cost_receipts=cost_journal.load_chain(),
        lifecycle=lifecycle,
    )


def _remaining_sealed_retry_entitlement(
    foundation: StairGenerationSessionFoundation,
) -> int:
    """Return the one session-wide sealed retry only while it is unconsumed."""

    train_retry_jobs, eval_retry_seed_units = _session_retry_usage(foundation)
    if train_retry_jobs > foundation.cost_contract.retry_train_jobs:
        raise ValueError("Stair session-wide train retry reserve is exhausted")
    if eval_retry_seed_units > foundation.cost_contract.retry_eval_seed_units:
        raise ValueError("Stair session-wide eval retry reserve is exhausted")
    if eval_retry_seed_units not in {
        0,
        foundation.cost_contract.retry_eval_seed_units,
    }:
        raise ValueError("Stair sealed retry reserve was consumed fractionally")
    return (
        STAIR_GENERATION_MAX_INFRASTRUCTURE_RETRIES_PER_ELIGIBLE_JOB
        if eval_retry_seed_units == 0
        else 0
    )


def _session_retry_usage(
    foundation: StairGenerationSessionFoundation,
) -> tuple[int, int]:
    train_retry_jobs = 0
    eval_retry_seed_units = 0
    attempt_root = foundation.config.output_dir / "job_attempts"
    if attempt_root.exists():
        if attempt_root.is_symlink() or not attempt_root.is_dir():
            raise ValueError("job-attempt root is invalid")
        for path in sorted(attempt_root.glob("*/attempt_*.reservation.json")):
            journal, reservation = JobAttemptJournal.from_reservation_path(path)
            scope = journal.scope
            if (
                scope.budget_contract_hash != foundation.cost_contract.contract_hash
                or scope.session_contract_hash
                != foundation.session_contract.contract_hash
            ):
                raise ValueError("job attempt belongs to another retry contract")
            attempt_index = reservation.payload["attempt_index"]
            if attempt_index == 0:
                continue
            if attempt_index != 1:
                raise ValueError("Stair logical job exceeded its retry entitlement")
            if journal.job_kind == "train":
                train_retry_jobs += 1
            elif journal.job_kind in {"eval", "eval_phase", "baseline_eval"}:
                eval_retry_seed_units += _eval_seed_units_for_job(
                    journal.logical_job_id,
                    journal.job_kind,
                    cost_contract=foundation.cost_contract,
                )
            else:
                raise ValueError("unknown Stair job-attempt kind")
    return train_retry_jobs, eval_retry_seed_units


def _first_generation_segment_id(output_dir: Path) -> str:
    root = output_dir / "generation_segments"
    segments = sorted(root.glob("segment-*"))
    if not segments:
        raise ValueError("generation lifecycle has no segment")
    first = segments[0]
    if first.is_symlink() or not first.is_dir():
        raise ValueError("generation segment directory is invalid")
    start = load_strict_json_object(
        first / "start.json",
        context="generation first segment start",
    )
    segment_id = start.get("segment_id")
    if (
        not isinstance(segment_id, str)
        or not segment_id
        or not first.name.endswith(segment_id)
    ):
        raise ValueError("generation first segment identity mismatch")
    return segment_id


def _generation_exploration_for_ledger(
    foundation: StairGenerationSessionFoundation,
    *,
    genesis: VerifiedSessionGenesis,
    ledger: list[dict[str, Any]],
) -> GenerationExplorationPlan:
    protocol = foundation.composition.task_plugin.task_spec.acceptance_protocol
    if protocol is None:
        raise ValueError("Stair generation requires an acceptance protocol")
    history = project_public_history(
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
    return foundation.composition.task_memory_adapter.build_generation_exploration_plan(
        history,
        generation_session_contract=foundation.session_contract,
    )


def _load_existing_generation_execution(
    foundation: StairGenerationSessionFoundation,
    *,
    record: Mapping[str, Any],
) -> tuple[dict[str, Any], Any]:
    manifest = load_bound_generation_manifest(record)
    journal = GenerationReceiptJournal(
        foundation.config.output_dir / "generation_receipts" / manifest.generation_id,
        manifest=manifest,
    )
    loaded = journal.load_execution()
    if loaded is None or loaded[0] != dict(record):
        raise ValueError("terminal generation execution receipt is missing")
    return loaded


def _generation_cost_receipt_projection(
    receipt: StairGenerationActualCostReceiptRef,
) -> dict[str, Any]:
    return {
        "path": str(receipt.path.resolve()),
        "payload_hash": receipt.payload_hash,
        "event_id": receipt.payload["event_id"],
    }


def _publish_or_validate_generation_closure(
    foundation: StairGenerationSessionFoundation,
    *,
    ledger: list[dict[str, Any]],
    generation_record: Mapping[str, Any],
    generation_execution_receipt: Mapping[str, Any],
    actual_cost_receipt: StairGenerationActualCostReceiptRef,
    resource_evidence: Mapping[str, Any],
    segment_id: str,
    require_existing: bool = False,
) -> dict[str, Any]:
    generation_id = generation_record.get("generation_id")
    generation_index = generation_record.get("generation_index")
    if (
        not isinstance(generation_id, str)
        or not generation_id
        or isinstance(generation_index, bool)
        or not isinstance(generation_index, int)
        or generation_index <= 0
        or len(ledger) < generation_index
        or ledger[generation_index - 1] != dict(generation_record)
    ):
        raise ValueError("generation closure record identity is invalid")
    path = (
        foundation.config.output_dir / "generation_closures" / f"{generation_id}.json"
    )
    loaded: dict[str, Any] | None = None
    if path.exists() or path.is_symlink():
        loaded = load_strict_json_object(path, context="Stair generation closure")
        if path.read_bytes() != canonical_json_bytes(loaded, trailing_newline=True):
            raise ValueError("Stair generation closure is not canonical JSON")
        existing_segment = loaded.get("closed_by_segment_id")
        existing_resource = loaded.get("resource_evidence")
        if not isinstance(existing_segment, str) or not existing_segment:
            raise ValueError("Stair generation closure segment is invalid")
        if not isinstance(existing_resource, Mapping):
            raise ValueError("Stair generation closure resource evidence is invalid")
        segment_id = existing_segment
        resource_evidence = existing_resource
    elif require_existing:
        raise FileNotFoundError(
            f"bounded generation closure is missing: {generation_id}"
        )
    _validate_stored_resource_closure_evidence(
        foundation,
        resource_evidence=resource_evidence,
        generation_index=generation_index,
    )
    body = {
        "schema_version": STAIR_GENERATION_CLOSURE_SCHEMA_VERSION,
        "session_id": foundation.config.session_id,
        "generation_id": generation_id,
        "generation_index": generation_index,
        "generation_record_hash": canonical_json_hash(generation_record),
        "generation_execution_receipt": dict(generation_execution_receipt),
        "generation_execution_receipt_hash": canonical_json_hash(
            generation_execution_receipt
        ),
        "ledger_prefix_hash": canonical_json_hash(ledger[:generation_index]),
        "resulting_current_best_hash": generation_record.get(
            "resulting_current_best_hash"
        ),
        "actual_cost_receipt": _generation_cost_receipt_projection(actual_cost_receipt),
        "resource_evidence": dict(resource_evidence),
        "resource_evidence_hash": canonical_json_hash(resource_evidence),
        "closed_by_segment_id": segment_id,
        "acceptance_authority": False,
    }
    payload = {**body, "closure_hash": canonical_json_hash(body)}
    if loaded is not None:
        if loaded != payload:
            raise ValueError("Stair generation closure differs from recomputation")
        return loaded
    path.parent.mkdir(parents=True, exist_ok=True)
    create_once_bytes(path, canonical_json_bytes(payload, trailing_newline=True))
    return payload


def _settle_generation_cost(
    foundation: StairGenerationSessionFoundation,
    *,
    genesis: VerifiedSessionGenesis,
    ledger: list[dict[str, Any]],
    cost_journal: StairGenerationActualCostJournal,
    event_id: str,
    previous_cost_chain_head_hash: str,
) -> tuple[StairGenerationActualCostReceiptRef, tuple[Path, ...]]:
    usage, conservative_charges, open_reservations = _collect_actual_cost_usage(
        foundation,
        genesis=genesis,
        ledger=ledger,
    )
    existing = cost_journal.receipt_for_event(event_id)
    if existing is not None:
        _assert_actual_cost_receipt_matches_usage(
            existing,
            usage=usage,
            conservative_charges=conservative_charges,
            context="generation settlement",
        )
        return existing, open_reservations
    receipt = cost_journal.publish_snapshot(
        event_id=event_id,
        previous_cost_chain_head_hash=previous_cost_chain_head_hash,
        usage=usage,
        conservative_charges=conservative_charges,
    )
    return receipt, open_reservations


def _assert_actual_cost_receipt_matches_usage(
    receipt: StairGenerationActualCostReceiptRef,
    *,
    usage: StairGenerationActualCostUsage,
    conservative_charges: tuple[str, ...],
    context: str,
) -> None:
    actual = receipt.payload["actual"]
    for field_name, current in usage.to_dict().items():
        if field_name in {"observed_output_bytes", "observed_output_files"}:
            continue
        if actual.get(field_name) != current:
            raise ValueError(f"{context} actual cost drifted: {field_name}")
    if usage.observed_output_bytes > actual["accounted_output_bytes"]:
        raise ValueError(f"{context} exceeded its settled output reserve")
    if tuple(receipt.payload["conservative_charges"]) != conservative_charges:
        raise ValueError(f"{context} conservative charges drifted")


def _current_validation_boundary_cost_receipt(
    foundation: StairGenerationSessionFoundation,
    *,
    genesis: VerifiedSessionGenesis,
    ledger: list[dict[str, Any]],
    cost_journal: StairGenerationActualCostJournal,
    lifecycle: GenerationLifecycleAudit,
    segment_id: str,
) -> StairGenerationActualCostReceiptRef:
    """Return an actual receipt at the current bounded-session cost-chain head.

    Generation closures intentionally retain their generation-settlement receipt.
    A pause or terminal boundary instead binds the current segment head, including
    later failure snapshots.  Lifecycle recovery can advance the combined head
    without creating an actual-cost receipt, so that one case is reconciled by a
    segment-unique, create-once snapshot.
    """

    combined_head = _replay_validation_combined_cost_chain(cost_journal, lifecycle)
    chain = cost_journal.load_chain()
    if not chain:
        raise ValueError("Stair bounded generation boundary has no actual-cost chain")
    latest = chain[-1]
    usage, conservative_charges, open_reservations = _collect_actual_cost_usage(
        foundation,
        genesis=genesis,
        ledger=ledger,
    )
    if open_reservations:
        raise ValueError("Stair bounded generation boundary has open job-attempt reservations")

    if latest.payload_hash == combined_head:
        current = latest
    else:
        current = cost_journal.publish_snapshot(
            event_id=f"segment-{segment_id}-boundary-reconciliation",
            previous_cost_chain_head_hash=combined_head,
            usage=usage,
            conservative_charges=conservative_charges,
        )
        if (
            _replay_validation_combined_cost_chain(cost_journal, lifecycle)
            != current.payload_hash
        ):
            raise ValueError(
                "Stair bounded generation boundary reconciliation did not close the cost chain"
            )

    _assert_actual_cost_receipt_matches_usage(
        current,
        usage=usage,
        conservative_charges=conservative_charges,
        context="Stair bounded generation boundary",
    )
    if current.payload["cap_checks"]["all_passed"] is not True:
        raise ValueError("Stair bounded generation boundary exhausted its frozen cost contract")
    return current


def _validation_boundary_was_already_paused(
    output_dir: Path,
    *,
    generation_count: int,
) -> bool:
    """Distinguish a delivered pause from a failed post-generation segment."""

    segment_root = output_dir / "generation_segments"
    segment_dirs = sorted(segment_root.glob("segment-*"))
    if not segment_dirs:
        return False
    exit_path = segment_dirs[-1] / "exit.json"
    if not exit_path.is_file():
        return False
    exit_record = load_strict_json_object(
        exit_path,
        context="Stair bounded generation last segment exit",
    )
    return bool(
        exit_record.get("status") == "paused"
        and exit_record.get("reason") == "operator_pause"
        and exit_record.get("generations_at_end") == generation_count
    )


def _publish_or_load_parent_bootstrap_cost_boundary(
    foundation: StairGenerationSessionFoundation,
    *,
    genesis: VerifiedSessionGenesis,
    ledger: list[dict[str, Any]],
    cost_journal: StairGenerationActualCostJournal,
    previous_cost_chain_head_hash: str,
) -> StairGenerationActualCostReceiptRef:
    """Bind the one session-level parent bootstrap outside generation deltas."""

    event_id = "post-parent-bootstrap"
    existing = cost_journal.receipt_for_event(event_id)
    if existing is not None:
        if existing.payload["cap_checks"]["all_passed"] is not True:
            raise ValueError("Stair bounded generation parent bootstrap exhausted its frozen cost contract")
        return existing
    if ledger:
        raise ValueError(
            "Stair bounded generation closed generation prefix lacks its parent-bootstrap cost boundary"
        )
    usage, conservative_charges, open_reservations = _collect_actual_cost_usage(
        foundation,
        genesis=genesis,
        ledger=[],
    )
    if open_reservations:
        raise ValueError("Stair bounded generation parent bootstrap left an open attempt reservation")
    receipt = cost_journal.publish_snapshot(
        event_id=event_id,
        previous_cost_chain_head_hash=previous_cost_chain_head_hash,
        usage=usage,
        conservative_charges=conservative_charges,
        finalization_output_reserve_bytes=0,
    )
    if receipt.payload["cap_checks"]["all_passed"] is not True:
        raise ValueError("Stair bounded generation parent bootstrap exhausted its frozen cost contract")
    return receipt


def _collect_actual_cost_usage(
    foundation: StairGenerationSessionFoundation,
    *,
    genesis: VerifiedSessionGenesis | None,
    ledger: list[dict[str, Any]],
) -> tuple[StairGenerationActualCostUsage, tuple[str, ...], tuple[Path, ...]]:
    provider = _provider_actuals(foundation, ledger=ledger)
    scientific_cost = foundation.cost_contract.scientific_cost_basis
    if genesis is None:
        train_steps_by_job: dict[str, int] = {}
        eval_cost_by_job: dict[str, tuple[int, int]] = {}
        job_cost_evidence_hash = canonical_json_hash([])
    else:
        (
            train_steps_by_job,
            eval_cost_by_job,
            job_cost_evidence_hash,
        ) = _validated_job_cost_artifacts(foundation, genesis=genesis)
    attempt_root = foundation.config.output_dir / "job_attempts"
    job_attempts = 0
    train_jobs = 0
    train_steps = 0
    eval_seed_units = 0
    eval_control_steps = 0
    lease_wall_ns = 0
    unmeasured_bound_attempts = 0
    conservative: list[str] = list(provider["conservative_charges"])
    open_reservations: list[Path] = []
    used_train_costs: set[str] = set()
    used_eval_costs: set[str] = set()
    if attempt_root.exists():
        if attempt_root.is_symlink() or not attempt_root.is_dir():
            raise ValueError("job-attempt root is invalid")
        for path in sorted(attempt_root.glob("*/attempt_*.reservation.json")):
            journal, reservation = JobAttemptJournal.from_reservation_path(path)
            scope = journal.scope
            if (
                scope.budget_contract_hash != foundation.cost_contract.contract_hash
                or scope.session_contract_hash
                != foundation.session_contract.contract_hash
            ):
                raise ValueError("job attempt belongs to another cost contract")
            result_path = path.with_name(path.name.replace(".reservation.", ".result."))
            if not result_path.exists():
                open_reservations.append(path)
                job_attempts += 1
                if journal.resource_binding_hash is not None:
                    unmeasured_bound_attempts += 1
                    conservative.append(
                        f"open-bound-attempt:{journal.logical_job_id}:"
                        f"{reservation.payload['attempt_index']}"
                    )
                if journal.job_kind == "train":
                    train_jobs += 1
                    train_steps += scientific_cost.train_steps_per_job
                    conservative.append(
                        f"full-train-attempt:{journal.logical_job_id}:"
                        f"{reservation.payload['attempt_index']}"
                    )
                else:
                    seed_units = _eval_seed_units_for_job(
                        journal.logical_job_id,
                        journal.job_kind,
                        cost_contract=foundation.cost_contract,
                    )
                    eval_seed_units += seed_units
                    eval_control_steps += (
                        seed_units
                        * scientific_cost.max_eval_control_steps_per_seed_unit
                    )
                    conservative.append(
                        f"full-eval-attempt:{journal.logical_job_id}:"
                        f"{reservation.payload['attempt_index']}"
                    )
                continue
            result = journal.load_result(reservation.payload["attempt_index"])
            payload = result.payload
            job_attempts += 1
            if journal.resource_binding_hash is not None:
                if (
                    payload.get("schema_version")
                    == RESOURCE_LEASE_ATTEMPT_RESULT_SCHEMA_VERSION
                ):
                    lease_wall_ns += payload["resource_lease_wall_duration_ns"]
                else:
                    unmeasured_bound_attempts += 1
                    conservative.append(
                        f"unmeasured-bound-attempt:{journal.logical_job_id}:"
                        f"{payload['attempt_index']}"
                    )
            if journal.job_kind == "train":
                train_jobs += 1
                observed_steps = train_steps_by_job.get(journal.logical_job_id)
                if payload.get("outcome") == "completed" and observed_steps is not None:
                    if journal.logical_job_id in used_train_costs:
                        raise ValueError("train cost artifact was consumed twice")
                    used_train_costs.add(journal.logical_job_id)
                    train_steps += observed_steps
                else:
                    train_steps += scientific_cost.train_steps_per_job
                    conservative.append(
                        f"full-train-attempt:{journal.logical_job_id}:"
                        f"{payload['attempt_index']}"
                    )
                continue
            if journal.job_kind not in {"eval", "eval_phase", "baseline_eval"}:
                raise ValueError("unknown Stair job-attempt kind")
            seed_units = _eval_seed_units_for_job(
                journal.logical_job_id,
                journal.job_kind,
                cost_contract=foundation.cost_contract,
            )
            eval_seed_units += seed_units
            observed_cost = eval_cost_by_job.get(journal.logical_job_id)
            if payload.get("outcome") == "completed" and observed_cost is not None:
                if journal.logical_job_id in used_eval_costs:
                    raise ValueError("eval cost artifact was consumed twice")
                observed_seed_units, observed_steps = observed_cost
                if observed_seed_units != seed_units:
                    raise ValueError("eval cost seed count differs from attempt plan")
                used_eval_costs.add(journal.logical_job_id)
                eval_control_steps += observed_steps
            else:
                eval_control_steps += (
                    seed_units * scientific_cost.max_eval_control_steps_per_seed_unit
                )
                conservative.append(
                    f"full-eval-attempt:{journal.logical_job_id}:"
                    f"{payload['attempt_index']}"
                )
    unused_train = set(train_steps_by_job) - used_train_costs
    unused_eval = set(eval_cost_by_job) - used_eval_costs
    if unused_train or unused_eval:
        raise ValueError("cost artifacts are not bound to completed job attempts")
    output_bytes, output_files = _output_tree_size(foundation.config.output_dir)
    conservative_gpu_ns = 0
    usage_type: type[StairGenerationActualCostUsage] = StairGenerationActualCostUsage
    if isinstance(foundation.cost_contract, StairGenerationSessionCostContractV2):
        allocation_state = _strict_operation_allocation_state(
            output_dir=foundation.config.output_dir,
            session_id=foundation.config.session_id,
            cost_contract=foundation.cost_contract,
            session_contract_hash=foundation.session_contract.contract_hash,
        )
        if (
            allocation_state.measured_gpu_capacity_ns != lease_wall_ns
            or allocation_state.unmeasured_bound_attempts != unmeasured_bound_attempts
        ):
            raise ValueError("Stair bounded generation operation allocation accounting drifted")
        conservative_gpu_ns = allocation_state.conservatively_charged_gpu_capacity_ns
        usage_type = StairGenerationActualCostUsageV2
    usage_kwargs = {
        "provider_attempts": provider["attempts"],
        "provider_request_units": provider["request_units"],
        "provider_response_units": provider["response_units"],
        "provider_total_units": provider["total_units"],
        "provider_reasoning_units": provider["reasoning_units"],
        "provider_conservative_attempts": provider["conservative_attempts"],
        "job_attempts": job_attempts,
        "train_jobs": train_jobs,
        "train_steps": train_steps,
        "eval_seed_units": eval_seed_units,
        "simulated_control_steps": train_steps + eval_control_steps,
        "resource_lease_wall_duration_ns": lease_wall_ns,
        "unmeasured_bound_attempts": unmeasured_bound_attempts,
        "observed_output_bytes": output_bytes,
        "observed_output_files": output_files,
        "job_cost_evidence_hash": job_cost_evidence_hash,
    }
    if usage_type is StairGenerationActualCostUsageV2:
        usage_kwargs["conservatively_charged_gpu_capacity_ns"] = conservative_gpu_ns
    usage = usage_type(**usage_kwargs)
    return usage, tuple(sorted(set(conservative))), tuple(open_reservations)


def _assert_provider_request_budget(
    *,
    output_dir: Path,
    generation_id: str,
    current_lane_id: str,
    current_prompt_payload: Mapping[str, Any],
    max_generations: int = 1,
    allow_conservative_interrupted_recovery: bool = False,
) -> None:
    if not isinstance(allow_conservative_interrupted_recovery, bool):
        raise TypeError("provider interrupted recovery policy must be boolean")
    if current_lane_id not in GENERATION_LANE_IDS:
        raise ValueError("provider budget gate lane is invalid")
    if (
        isinstance(max_generations, bool)
        or not isinstance(max_generations, int)
        or max_generations <= 0
    ):
        raise ValueError("provider budget generation count is invalid")
    generation_prefix, marker, generation_suffix = generation_id.rpartition(
        "-generation-"
    )
    if (
        marker != "-generation-"
        or not generation_prefix
        or not generation_suffix.isdigit()
        or int(generation_suffix) not in range(1, max_generations + 1)
    ):
        raise ValueError("provider budget generation identity is invalid")
    _assert_expected_authoring_directories(
        output_dir,
        generation_prefix=generation_prefix,
        max_generations=max_generations,
    )
    attempt_count = 0
    request_units = 0
    response_units = 0
    total_units = 0
    current_attempt_seen = False
    attempt_counts_by_generation: dict[str, int] = {}
    for generation_index in range(1, max_generations + 1):
        scanned_generation_id = f"{generation_prefix}-generation-{generation_index:03d}"
        attempt_counts_by_generation[scanned_generation_id] = 0
        for lane_id in GENERATION_LANE_IDS:
            round_id = f"{scanned_generation_id}-{lane_id}"
            round_dir = output_dir / "authoring" / scanned_generation_id / lane_id
            if not round_dir.exists():
                if (
                    scanned_generation_id == generation_id
                    and lane_id == current_lane_id
                ):
                    raise ValueError("current provider journal does not exist")
                continue
            if round_dir.is_symlink() or not round_dir.is_dir():
                raise ValueError("provider budget journal directory is invalid")
            attempts = (
                RoundAuthoringJournal(
                    round_dir,
                    round_id=round_id,
                )
                .snapshot()
                .attempts
            )
            attempt_counts_by_generation[scanned_generation_id] += len(attempts)
            for index, attempt in enumerate(attempts):
                attempt_count += 1
                is_current = (
                    scanned_generation_id == generation_id
                    and lane_id == current_lane_id
                    and index == len(attempts) - 1
                )
                status = attempt.get("status")
                if is_current:
                    if current_attempt_seen or status != "interrupted_after_prompt":
                        raise ValueError(
                            "current provider attempt is not its unique prompt reservation"
                        )
                    prompt_ref = PromptPayloadRef.from_mapping(
                        attempt.get("prompt_payload")
                    )
                    if verify_prompt_payload(prompt_ref) != dict(
                        current_prompt_payload
                    ):
                        raise ValueError("current provider prompt reservation mismatch")
                    current_attempt_seen = True
                    continue
                if status == "interrupted_before_prompt":
                    continue
                if status == "interrupted_after_prompt":
                    if not allow_conservative_interrupted_recovery:
                        raise RuntimeError(
                            "prior provider request has no authoritative usage receipt"
                        )
                    request_units += STAIR_GENERATION_PROVIDER_INPUT_UNITS_PER_ATTEMPT
                    response_units += (
                        STAIR_GENERATION_PROVIDER_OUTPUT_UNITS_PER_ATTEMPT
                    )
                    total_units += (
                        STAIR_GENERATION_PROVIDER_INPUT_UNITS_PER_ATTEMPT
                        + STAIR_GENERATION_PROVIDER_OUTPUT_UNITS_PER_ATTEMPT
                    )
                    continue
                provider_attempt = attempt.get("provider_attempt")
                if status != "completed" or not isinstance(provider_attempt, Mapping):
                    raise ValueError("prior provider attempt state is invalid")
                usage = provider_attempt.get("provider_usage")
                if not isinstance(usage, Mapping) or not all(
                    field_name in usage
                    for field_name in (
                        "request_units",
                        "response_units",
                        "total_units",
                    )
                ):
                    raise RuntimeError(
                        "prior provider request has no authoritative usage receipt"
                    )
                values: list[int] = []
                for field_name in (
                    "request_units",
                    "response_units",
                    "total_units",
                ):
                    value = usage[field_name]
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, int)
                        or value < 0
                    ):
                        raise ValueError("prior provider usage counter is invalid")
                    values.append(value)
                if (
                    values[0] > STAIR_GENERATION_PROVIDER_INPUT_UNITS_PER_ATTEMPT
                    or values[1] > STAIR_GENERATION_PROVIDER_OUTPUT_UNITS_PER_ATTEMPT
                    or values[2]
                    > STAIR_GENERATION_PROVIDER_INPUT_UNITS_PER_ATTEMPT
                    + STAIR_GENERATION_PROVIDER_OUTPUT_UNITS_PER_ATTEMPT
                ):
                    raise RuntimeError("prior provider request exceeded its frozen cap")
                request_units += values[0]
                response_units += values[1]
                total_units += values[2]
    if not current_attempt_seen:
        raise ValueError("current provider prompt reservation was not found")
    provider_attempt_cap = (
        max_generations
        * len(GENERATION_LANE_IDS)
        * STAIR_GENERATION_PROVIDER_ATTEMPTS_PER_LANE
    )
    if attempt_count > provider_attempt_cap:
        raise RuntimeError("Stair provider attempt cap is exhausted")
    per_generation_attempt_cap = (
        len(GENERATION_LANE_IDS) * STAIR_GENERATION_PROVIDER_ATTEMPTS_PER_LANE
    )
    if any(
        count > per_generation_attempt_cap
        for count in attempt_counts_by_generation.values()
    ):
        raise RuntimeError("Stair per-generation provider attempt cap is exhausted")
    current_generation_index = int(generation_suffix)
    if any(
        attempt_counts_by_generation[f"{generation_prefix}-generation-{index:03d}"]
        for index in range(current_generation_index + 1, max_generations + 1)
    ):
        raise ValueError("provider attempts exist ahead of the current generation")
    request_units += STAIR_GENERATION_PROVIDER_INPUT_UNITS_PER_ATTEMPT
    response_units += STAIR_GENERATION_PROVIDER_OUTPUT_UNITS_PER_ATTEMPT
    total_units += (
        STAIR_GENERATION_PROVIDER_INPUT_UNITS_PER_ATTEMPT
        + STAIR_GENERATION_PROVIDER_OUTPUT_UNITS_PER_ATTEMPT
    )
    current_attempt_count = attempt_counts_by_generation[generation_id]
    remaining_generation_attempts = per_generation_attempt_cap - current_attempt_count
    request_units += (
        remaining_generation_attempts
        * STAIR_GENERATION_PROVIDER_INPUT_UNITS_PER_ATTEMPT
    )
    response_units += (
        remaining_generation_attempts
        * STAIR_GENERATION_PROVIDER_OUTPUT_UNITS_PER_ATTEMPT
    )
    total_units += remaining_generation_attempts * (
        STAIR_GENERATION_PROVIDER_INPUT_UNITS_PER_ATTEMPT
        + STAIR_GENERATION_PROVIDER_OUTPUT_UNITS_PER_ATTEMPT
    )
    if (
        request_units
        > provider_attempt_cap * STAIR_GENERATION_PROVIDER_INPUT_UNITS_PER_ATTEMPT
        or response_units
        > provider_attempt_cap * STAIR_GENERATION_PROVIDER_OUTPUT_UNITS_PER_ATTEMPT
        or total_units
        > provider_attempt_cap
        * (
            STAIR_GENERATION_PROVIDER_INPUT_UNITS_PER_ATTEMPT
            + STAIR_GENERATION_PROVIDER_OUTPUT_UNITS_PER_ATTEMPT
        )
    ):
        raise RuntimeError("Stair provider unit cap cannot reserve this request")


def _provider_actuals(
    foundation: StairGenerationSessionFoundation,
    *,
    ledger: list[dict[str, Any]],
) -> dict[str, Any]:
    totals = {
        "attempts": 0,
        "request_units": 0,
        "response_units": 0,
        "total_units": 0,
        "reasoning_units": 0,
        "conservative_attempts": 0,
    }
    charges: list[str] = []
    snapshots: dict[tuple[str, str], Any] = {}
    session_contract = getattr(foundation, "session_contract", None)
    max_generations = getattr(session_contract, "max_generations", 1)
    _assert_expected_authoring_directories(
        foundation.config.output_dir,
        generation_prefix=foundation.config.session_id,
        max_generations=max_generations,
    )
    for generation_index in range(1, max_generations + 1):
        generation_id = (
            f"{foundation.config.session_id}-generation-{generation_index:03d}"
        )
        for lane_id in GENERATION_LANE_IDS:
            round_id = f"{generation_id}-{lane_id}"
            round_dir = (
                foundation.config.output_dir / "authoring" / generation_id / lane_id
            )
            if not round_dir.exists():
                continue
            if round_dir.is_symlink() or not round_dir.is_dir():
                raise ValueError("authoring round directory is invalid")
            snapshot = RoundAuthoringJournal(
                round_dir,
                round_id=round_id,
            ).snapshot()
            snapshots[(generation_id, lane_id)] = snapshot
            for attempt in snapshot.attempts:
                totals["attempts"] += 1
                status = attempt["status"]
                if status == "interrupted_before_prompt":
                    continue
                provider_attempt = attempt.get("provider_attempt")
                if status == "interrupted_after_prompt":
                    _charge_full_provider_reserve(
                        totals,
                        charges,
                        label=f"{round_id}:{attempt['sequence']}",
                    )
                    continue
                if status != "completed" or not isinstance(provider_attempt, Mapping):
                    raise ValueError("authoring attempt status is invalid")
                _accumulate_provider_attempt(
                    totals,
                    charges,
                    provider_attempt=provider_attempt,
                    label=f"{round_id}:{attempt['sequence']}",
                )
    ledger_lanes: set[tuple[str, str]] = set()
    for record in ledger:
        generation_id = record.get("generation_id")
        if not isinstance(generation_id, str) or not generation_id:
            raise TypeError("generation ledger identity is invalid")
        trials = record.get("trials")
        artifacts = record.get("private_artifacts")
        if not isinstance(trials, list) or not isinstance(artifacts, list):
            raise TypeError("generation ledger authoring evidence is invalid")
        for trial in trials:
            if not isinstance(trial, Mapping):
                raise TypeError("generation trial is invalid")
            authoring = trial.get("authoring")
            if authoring is None:
                continue
            if not isinstance(authoring, Mapping):
                raise TypeError("generation trial authoring is invalid")
            lane_id = trial.get("lane_id")
            if not isinstance(lane_id, str):
                raise TypeError("generation trial lane id is invalid")
            ledger_lanes.add((generation_id, lane_id))
            prompt = _unique_private_artifact(
                artifacts,
                kind=f"{lane_id}_prompt_payload",
                checksum=authoring.get("prompt_payload_checksum"),
            )
            evidence = _unique_private_artifact(
                artifacts,
                kind=f"{lane_id}_authoring_evidence",
                checksum=authoring.get("authoring_evidence_checksum"),
            )
            prompt_ref = PromptPayloadRef(
                path=prompt["uri"],
                checksum=prompt["checksum"],
                size_bytes=prompt["size_bytes"],
            )
            evidence_payload = verify_authoring_evidence(
                AuthoringEvidenceRef(
                    path=evidence["uri"],
                    checksum=evidence["checksum"],
                    size_bytes=evidence["size_bytes"],
                ),
                expected_round_id=str(trial.get("trial_id")),
                expected_selected_prompt=prompt_ref,
            )
            snapshot = snapshots.get((generation_id, lane_id))
            if (
                snapshot is None
                or list(snapshot.attempts) != evidence_payload["attempts"]
            ):
                raise ValueError(
                    "ledger authoring evidence differs from its live journal"
                )
    if not ledger_lanes.issubset(snapshots):
        raise ValueError("ledger authoring journal is missing")
    return {**totals, "conservative_charges": charges}


def _assert_expected_authoring_directories(
    output_dir: Path,
    *,
    generation_prefix: str,
    max_generations: int,
) -> None:
    authoring_root = Path(output_dir) / "authoring"
    if not authoring_root.exists() and not authoring_root.is_symlink():
        return
    if authoring_root.is_symlink() or not authoring_root.is_dir():
        raise ValueError("authoring root directory is invalid")
    expected_generations = {
        f"{generation_prefix}-generation-{index:03d}"
        for index in range(1, max_generations + 1)
    }
    for generation_path in authoring_root.iterdir():
        if (
            generation_path.name not in expected_generations
            or generation_path.is_symlink()
            or not generation_path.is_dir()
        ):
            raise ValueError("authoring root contains an unknown generation")
        for lane_path in generation_path.iterdir():
            if (
                lane_path.name not in GENERATION_LANE_IDS
                or lane_path.is_symlink()
                or not lane_path.is_dir()
            ):
                raise ValueError("authoring generation contains an unknown lane")


def _accumulate_provider_attempt(
    totals: dict[str, int],
    charges: list[str],
    *,
    provider_attempt: Mapping[str, Any],
    label: str,
) -> None:
    usage = provider_attempt.get("provider_usage")
    if not isinstance(usage, Mapping) or not all(
        field_name in usage
        for field_name in ("request_units", "response_units", "total_units")
    ):
        _charge_full_provider_reserve(totals, charges, label=label)
        return
    for source, target in (
        ("request_units", "request_units"),
        ("response_units", "response_units"),
        ("total_units", "total_units"),
    ):
        value = usage[source]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("provider usage counter is invalid")
        totals[target] += value
    reasoning = usage.get("reasoning_units", 0)
    if isinstance(reasoning, bool) or not isinstance(reasoning, int) or reasoning < 0:
        raise ValueError("provider reasoning usage is invalid")
    totals["reasoning_units"] += reasoning


def _charge_full_provider_reserve(
    totals: dict[str, int],
    charges: list[str],
    *,
    label: str,
) -> None:
    totals["request_units"] += STAIR_GENERATION_PROVIDER_INPUT_UNITS_PER_ATTEMPT
    totals["response_units"] += STAIR_GENERATION_PROVIDER_OUTPUT_UNITS_PER_ATTEMPT
    totals["total_units"] += (
        STAIR_GENERATION_PROVIDER_INPUT_UNITS_PER_ATTEMPT
        + STAIR_GENERATION_PROVIDER_OUTPUT_UNITS_PER_ATTEMPT
    )
    totals["conservative_attempts"] += 1
    charges.append("provider-full-reserve:" + label)


def _unique_private_artifact(
    artifacts: list[Any],
    *,
    kind: str,
    checksum: Any,
) -> Mapping[str, Any]:
    matches = [
        item
        for item in artifacts
        if isinstance(item, Mapping)
        and item.get("kind") == kind
        and item.get("checksum") == checksum
    ]
    if len(matches) != 1:
        raise ValueError(f"private artifact is not unique: {kind}")
    item = matches[0]
    if not isinstance(item.get("uri"), str) or not isinstance(
        item.get("size_bytes"), int
    ):
        raise TypeError("private artifact reference is invalid")
    return item


def _validated_job_cost_artifacts(
    foundation: StairGenerationSessionFoundation,
    *,
    genesis: VerifiedSessionGenesis,
) -> tuple[dict[str, int], dict[str, tuple[int, int]], str]:
    train_steps: dict[str, int] = {}
    eval_costs: dict[str, tuple[int, int]] = {}
    evidence: list[dict[str, Any]] = []
    private_path = foundation.config.output_dir / REFERENCE_EVAL_ARTIFACT_FILENAME
    private = load_strict_json_object(
        private_path,
        context="reference Genesis private eval artifact",
    )
    train_result = private.get("train_result")
    if not isinstance(train_result, Mapping):
        raise TypeError("reference Genesis train result is invalid")
    train_job_id = str(train_result.get("job_id"))
    _insert_unique_cost(
        train_steps,
        train_job_id,
        _train_planned_steps(
            train_result.get("summary"),
            expected_steps=(
                foundation.cost_contract.scientific_cost_basis.train_steps_per_job
            ),
        ),
    )
    eval_plan = genesis.reference_eval_plan()
    if eval_plan is None:
        raise ValueError("reference Genesis has no eval plan")
    evidence.append(
        _job_cost_evidence_projection(
            foundation,
            path=private_path,
            kind="reference-eval-artifact",
            job_identity={
                "train_job_id": train_job_id,
                "eval_job_id": eval_plan.job_id,
            },
        )
    )
    summary_path = eval_plan.metadata.get("summary_path")
    if not isinstance(summary_path, str) or not summary_path:
        raise ValueError("reference eval plan has no summary path")
    reference_summary_path = Path(summary_path)
    reference_summary = load_strict_json_object(
        reference_summary_path,
        context="reference evaluation summary",
    )
    if reference_summary.get("job_id") != eval_plan.job_id:
        raise ValueError("reference evaluation summary job identity mismatch")
    _insert_unique_cost(
        eval_costs,
        eval_plan.job_id,
        _execution_cost_totals(reference_summary.get("evaluation_execution_cost")),
    )
    evidence.append(
        _job_cost_evidence_projection(
            foundation,
            path=reference_summary_path,
            kind="reference-eval-summary",
            job_identity={"job_id": eval_plan.job_id},
        )
    )
    for root in (
        foundation.config.output_dir / "parent_public_dev_receipts",
        foundation.config.output_dir / "trial_receipts",
        foundation.config.output_dir
        / STAIR_FINAL_CONFIRMATION_DIRNAME
        / "phase_receipts",
    ):
        if not root.exists():
            continue
        if root.is_symlink() or not root.is_dir():
            raise ValueError("trial receipt root is invalid")
        for path in sorted(root.rglob("train.json")):
            ref = _trial_receipt_ref(path, kind="train")
            loaded = load_trial_train_receipt(ref)
            _insert_unique_cost(
                train_steps,
                loaded.result.job_id,
                _train_planned_steps(
                    loaded.result.summary,
                    expected_steps=(
                        foundation.cost_contract.scientific_cost_basis.train_steps_per_job
                    ),
                ),
            )
            evidence.append(
                _job_cost_evidence_projection(
                    foundation,
                    path=path,
                    kind="trial-train-receipt",
                    job_identity={"job_id": loaded.result.job_id},
                )
            )
        for path in sorted(root.rglob("eval_phases/*.json")):
            ref = _trial_receipt_ref(path, kind="eval_phase")
            loaded = load_trial_eval_phase_receipt(ref)
            _insert_unique_cost(
                eval_costs,
                loaded.result.job_id,
                _execution_cost_totals(loaded.result.execution_cost),
            )
            evidence.append(
                _job_cost_evidence_projection(
                    foundation,
                    path=path,
                    kind="trial-eval-phase-receipt",
                    job_identity={"job_id": loaded.result.job_id},
                )
            )
    evidence.sort(key=lambda item: (item["relative_path"], item["kind"]))
    return train_steps, eval_costs, canonical_json_hash(evidence)


def _job_cost_evidence_projection(
    foundation: StairGenerationSessionFoundation,
    *,
    path: Path,
    kind: str,
    job_identity: Mapping[str, str],
) -> dict[str, Any]:
    output_root = foundation.config.output_dir.resolve(strict=True)
    if path.is_symlink() or not path.is_file():
        raise ValueError("job cost evidence path is not a regular file")
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(output_root)
    except ValueError as exc:
        raise ValueError("job cost evidence escaped the session output") from exc
    raw = resolved.read_bytes()
    if not raw or len(raw) > 64 * 1024 * 1024:
        raise ValueError("job cost evidence size is invalid")
    return {
        "kind": kind,
        "relative_path": relative.as_posix(),
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
        "job_identity": dict(job_identity),
    }


def _trial_receipt_ref(path: Path, *, kind: str) -> TrialReceiptRef:
    payload = path.read_bytes()
    if not payload:
        raise ValueError("trial receipt is empty")
    return TrialReceiptRef(
        kind=kind,
        path=str(path.resolve()),
        checksum="sha256:" + hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
    )


def _train_planned_steps(summary: Any, *, expected_steps: int) -> int:
    if (
        isinstance(expected_steps, bool)
        or not isinstance(expected_steps, int)
        or expected_steps <= 0
    ):
        raise ValueError("expected Stair train steps must be positive")
    found: list[Mapping[str, Any]] = []

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            accounting = value.get("budget_accounting")
            if isinstance(accounting, Mapping):
                found.append(accounting)
            for item in value.values():
                visit(item)
        elif isinstance(value, list | tuple):
            for item in value:
                visit(item)

    visit(summary)
    if len(found) != 1 or found[0].get("status") != "pass":
        raise ValueError("train result has no unique passing budget accounting")
    steps = found[0].get("planned_env_steps")
    if steps != expected_steps:
        raise ValueError("Stair train result differs from the fixed budget")
    return steps


def _execution_cost_totals(value: Any) -> tuple[int, int]:
    if not isinstance(value, Mapping):
        raise TypeError("evaluation execution cost is missing")
    splits = value.get("splits")
    if not isinstance(splits, Mapping) or not splits:
        raise ValueError("evaluation execution cost splits are invalid")
    seed_units = 0
    control_steps = 0
    for split in splits.values():
        if not isinstance(split, Mapping):
            raise TypeError("evaluation execution cost split is invalid")
        seeds = split.get("seed_count")
        steps = split.get("simulated_env_control_steps")
        if (
            isinstance(seeds, bool)
            or not isinstance(seeds, int)
            or seeds <= 0
            or isinstance(steps, bool)
            or not isinstance(steps, int)
            or steps <= 0
        ):
            raise ValueError("evaluation execution cost counters are invalid")
        seed_units += seeds
        control_steps += steps
    return seed_units, control_steps


def _eval_seed_units_for_job(
    logical_job_id: str,
    job_kind: str,
    *,
    cost_contract: StairGenerationSessionCostContractLike,
) -> int:
    if not isinstance(cost_contract, StairGenerationSessionCostContractLike):
        raise TypeError("Stair eval seed cost requires the session cost contract")
    basis = cost_contract.scientific_cost_basis
    if job_kind in {"eval", "baseline_eval"}:
        return basis.reference_eval_seed_units
    suffixes = {
        "-public-dev": basis.seed_units_for_phase("public_dev"),
        "-public-accept": basis.seed_units_for_phase("public_accept"),
        "-sealed-validation": basis.seed_units_for_phase("sealed_validation"),
    }
    matches = [
        value for suffix, value in suffixes.items() if logical_job_id.endswith(suffix)
    ]
    if len(matches) != 1:
        raise ValueError("Stair eval-phase job id has an unknown phase")
    return matches[0]


def _insert_unique_cost(target: dict[str, Any], job_id: str, value: Any) -> None:
    if not job_id or job_id == "None" or job_id in target:
        raise ValueError("job cost artifact identity is duplicated")
    target[job_id] = value


def _output_tree_size(root: Path) -> tuple[int, int]:
    root = root.resolve(strict=True)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("generation output root is invalid")
    total_bytes = 0
    file_count = 0
    pending = [root]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.is_symlink():
                    raise ValueError("generation output tree contains a symlink")
                if entry.is_dir(follow_symlinks=False):
                    pending.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    total_bytes += entry.stat(follow_symlinks=False).st_size
                    file_count += 1
                else:
                    raise ValueError("generation output tree contains a special file")
    return total_bytes, file_count


def _build_resource_closure_evidence(
    foundation: StairGenerationSessionFoundation,
    *,
    genesis: VerifiedSessionGenesis,
    invocation_segment_id: str,
    startup_observation: Mapping[str, Any],
    generation_index: int | None = None,
    include_final_confirmation: bool = False,
) -> dict[str, Any]:
    _validate_resource_closure_generation_index(foundation, generation_index)
    if not isinstance(include_final_confirmation, bool):
        raise TypeError("resource closure final-confirmation scope must be boolean")
    expected_startup = {
        "schema_version": "true_rl_agent.stair.generation_compute_idle.v1",
        "stage": "startup",
        "source": "nvidia-smi-query-compute-apps",
        "target_gpu_uuids": [
            foundation.config.gpu0_uuid,
            foundation.config.gpu1_uuid,
        ],
        "rows": [],
    }
    if dict(startup_observation) != expected_startup:
        raise ValueError("generation startup GPU-idle evidence is invalid")
    final_idle = _assert_target_gpus_compute_idle(
        foundation.config.nvidia_smi_executable,
        expected=(
            foundation.config.gpu0_uuid,
            foundation.config.gpu1_uuid,
        ),
        stage="final",
    )
    summaries, completed_bound_job_count = _resource_summary_evidence(
        foundation,
        generation_index=generation_index,
        include_final_confirmation=include_final_confirmation,
    )
    attempt_chain = _attempt_chain_closure_evidence(
        foundation,
        generation_index=generation_index,
        include_final_confirmation=include_final_confirmation,
    )
    if attempt_chain["open_reservation_count"] != 0:
        raise ValueError("generation resource closure has open job attempts")
    reacquired = _reacquire_generation_gpu_leases(
        foundation,
        genesis=genesis,
        invocation_segment_id=invocation_segment_id,
    )
    payload = {
        "schema_version": (
            "true_rl_agent.stair.generation_resource_closure.v1"
            if generation_index is None
            else "true_rl_agent.stair.generation_resource_closure.v2"
        ),
        "startup_compute_idle": expected_startup,
        "final_compute_idle": final_idle,
        "completed_bound_job_count": completed_bound_job_count,
        "summaries": summaries,
        "attempt_chain": attempt_chain,
        "lease_reacquire": reacquired,
        "acceptance_authority": False,
    }
    if generation_index is not None:
        payload["generation_index"] = generation_index
    return payload


def _validate_stored_resource_closure_evidence(
    foundation: StairGenerationSessionFoundation,
    *,
    resource_evidence: Mapping[str, Any],
    generation_index: int | None = None,
    include_final_confirmation: bool = False,
) -> None:
    _validate_resource_closure_generation_index(foundation, generation_index)
    if not isinstance(include_final_confirmation, bool):
        raise TypeError("resource closure final-confirmation scope must be boolean")
    expected_gpu_uuids = [
        foundation.config.gpu0_uuid,
        foundation.config.gpu1_uuid,
    ]
    expected_startup = {
        "schema_version": "true_rl_agent.stair.generation_compute_idle.v1",
        "stage": "startup",
        "source": "nvidia-smi-query-compute-apps",
        "target_gpu_uuids": expected_gpu_uuids,
        "rows": [],
    }
    expected_final = {
        **expected_startup,
        "stage": "final",
    }
    expected_schema = (
        "true_rl_agent.stair.generation_resource_closure.v1"
        if generation_index is None
        else "true_rl_agent.stair.generation_resource_closure.v2"
    )
    if resource_evidence.get("schema_version") != expected_schema:
        raise ValueError("stored Stair bounded generation resource closure schema mismatch")
    if generation_index is not None:
        if set(resource_evidence) != {
            "schema_version",
            "generation_index",
            "startup_compute_idle",
            "final_compute_idle",
            "completed_bound_job_count",
            "summaries",
            "attempt_chain",
            "lease_reacquire",
            "acceptance_authority",
        }:
            raise ValueError("stored Stair bounded generation resource closure fields mismatch")
        if resource_evidence.get("generation_index") != generation_index:
            raise ValueError("stored Stair bounded generation resource closure prefix mismatch")
    if resource_evidence.get("startup_compute_idle") != expected_startup:
        raise ValueError("stored Stair bounded generation startup GPU evidence mismatch")
    if resource_evidence.get("final_compute_idle") != expected_final:
        raise ValueError("stored Stair bounded generation final GPU evidence mismatch")
    summaries, completed_count = _resource_summary_evidence(
        foundation,
        generation_index=generation_index,
        include_final_confirmation=include_final_confirmation,
    )
    if (
        resource_evidence.get("summaries") != summaries
        or resource_evidence.get("completed_bound_job_count") != completed_count
    ):
        raise ValueError("stored Stair bounded generation resource summaries drifted")
    if resource_evidence.get("attempt_chain") != _attempt_chain_closure_evidence(
        foundation,
        generation_index=generation_index,
        include_final_confirmation=include_final_confirmation,
    ):
        raise ValueError("stored Stair bounded generation attempt-chain closure drifted")
    lease_reacquire = resource_evidence.get("lease_reacquire")
    if not isinstance(lease_reacquire, Mapping) or lease_reacquire != {
        "passed": True,
        "simultaneous": True,
        "resource_binding_hashes": [
            binding.resource_binding_hash for binding in foundation.gpu_bindings
        ],
    }:
        raise ValueError("stored Stair bounded generation lease-reacquire evidence mismatch")
    if resource_evidence.get("acceptance_authority") is not False:
        raise ValueError("stored Stair bounded generation resource evidence gained authority")


def _resource_summary_evidence(
    foundation: StairGenerationSessionFoundation,
    *,
    generation_index: int | None = None,
    include_final_confirmation: bool = False,
) -> tuple[list[dict[str, Any]], int]:
    _validate_resource_closure_generation_index(foundation, generation_index)
    if not isinstance(include_final_confirmation, bool):
        raise TypeError("resource summary final-confirmation scope must be boolean")
    completed_jobs = _completed_bound_job_attempts(
        foundation,
        generation_index=generation_index,
        include_final_confirmation=include_final_confirmation,
    )
    prefix_known_jobs = _known_bound_job_attempts(
        foundation,
        generation_index=generation_index,
        include_final_confirmation=include_final_confirmation,
    )
    all_known_jobs = (
        prefix_known_jobs
        if generation_index is None
        else _known_bound_job_attempts(
            foundation,
            generation_index=foundation.session_contract.max_generations,
            include_final_confirmation=True,
        )
    )
    binding_by_hash = {
        binding.resource_binding_hash: (binding, identity.physical_id)
        for binding, identity in zip(
            foundation.gpu_bindings,
            foundation.gpu_identities,
            strict=True,
        )
    }
    artifact_root = foundation.config.output_dir / "runner"
    if completed_jobs and not artifact_root.is_dir():
        raise ValueError("completed bound jobs have no runner artifact root")
    projections: list[dict[str, Any]] = []
    seen_jobs: set[str] = set()
    for path in (
        sorted(artifact_root.rglob("summary.json")) if artifact_root.exists() else []
    ):
        if path.is_symlink() or not path.is_file():
            raise ValueError("runner summary path is not a regular file")
        resolved = path.resolve(strict=True)
        try:
            relative = resolved.relative_to(
                foundation.config.output_dir.resolve(strict=True)
            )
        except ValueError as exc:
            raise ValueError("runner summary escaped the session output") from exc
        raw = resolved.read_bytes()
        if not raw or len(raw) > 64 * 1024 * 1024:
            raise ValueError("runner summary size is invalid")
        summary = load_strict_json_object(
            resolved,
            context="bound Stair runner summary",
        )
        if summary.get("gpu_process_binding") is None:
            continue
        job_id = summary.get("job_id")
        if summary.get("status") != "pass":
            if job_id in completed_jobs:
                raise ValueError("completed bound job has a non-passing summary")
            continue
        if not isinstance(job_id, str):
            raise TypeError("passing bound summary job_id must be a string")
        if job_id not in completed_jobs:
            if job_id in all_known_jobs and job_id not in prefix_known_jobs:
                future_expected = binding_by_hash.get(all_known_jobs[job_id])
                if future_expected is None:
                    raise ValueError("future bound job used an unknown physical GPU")
                future_binding, future_physical_uuid = future_expected
                validate_stair_gpu_runtime_summary_evidence(
                    summary,
                    expected_binding=future_binding,
                    expected_physical_gpu_uuid=future_physical_uuid,
                    require_compute_context_observation=True,
                )
                continue
            raise ValueError("passing bound summary has no completed job attempt")
        if job_id in seen_jobs:
            raise ValueError("bound job has duplicate passing summaries")
        resource_hash = completed_jobs[job_id]
        expected = binding_by_hash.get(resource_hash)
        if expected is None:
            raise ValueError("bound job used an unknown physical GPU")
        binding, physical_uuid = expected
        validated = validate_stair_gpu_runtime_summary_evidence(
            summary,
            expected_binding=binding,
            expected_physical_gpu_uuid=physical_uuid,
            require_compute_context_observation=True,
        )
        process = validated["owned_process_identity"]
        runtime = validated["stair_gpu_runtime_evidence"]
        projections.append(
            {
                "relative_path": relative.as_posix(),
                "raw_sha256": hashlib.sha256(raw).hexdigest(),
                "job_id": job_id,
                "process_id": process["process_id"],
                "process_group_id": process["process_group_id"],
                "gpu_process_binding": validated["gpu_process_binding"],
                "external_compute_context_observation": runtime[
                    "external_compute_context_observation"
                ],
            }
        )
        seen_jobs.add(job_id)
    if seen_jobs != set(completed_jobs):
        raise ValueError("completed bound jobs lack unique resource summaries")
    return projections, len(completed_jobs)


def _completed_bound_job_attempts(
    foundation: StairGenerationSessionFoundation,
    *,
    generation_index: int | None = None,
    include_final_confirmation: bool = False,
) -> dict[str, str]:
    _validate_resource_closure_generation_index(foundation, generation_index)
    root = foundation.config.output_dir / "job_attempts"
    completed: dict[str, str] = {}
    if not root.exists():
        return completed
    for path in sorted(root.glob("*/attempt_*.reservation.json")):
        journal, reservation = JobAttemptJournal.from_reservation_path(path)
        if journal.resource_binding_hash is None:
            continue
        if (
            journal.scope.budget_contract_hash != foundation.cost_contract.contract_hash
            or journal.scope.session_contract_hash
            != foundation.session_contract.contract_hash
        ):
            raise ValueError("resource attempt belongs to another session")
        if not _attempt_is_in_generation_prefix(
            foundation,
            journal=journal,
            reservation=reservation.payload,
            generation_index=generation_index,
            include_final_confirmation=include_final_confirmation,
        ):
            continue
        result_path = path.with_name(path.name.replace(".reservation.", ".result."))
        if not result_path.exists():
            continue
        result = journal.load_result(reservation.payload["attempt_index"])
        if result.payload.get("outcome") != "completed":
            continue
        if journal.logical_job_id in completed:
            raise ValueError("bound logical job completed more than once")
        completed[journal.logical_job_id] = journal.resource_binding_hash
    return completed


def _known_bound_job_attempts(
    foundation: StairGenerationSessionFoundation,
    *,
    generation_index: int | None = None,
    include_final_confirmation: bool = False,
) -> dict[str, str]:
    _validate_resource_closure_generation_index(foundation, generation_index)
    root = foundation.config.output_dir / "job_attempts"
    known: dict[str, str] = {}
    if not root.exists():
        return known
    for path in sorted(root.glob("*/attempt_*.reservation.json")):
        journal, reservation = JobAttemptJournal.from_reservation_path(path)
        if journal.resource_binding_hash is None:
            continue
        if (
            journal.scope.budget_contract_hash != foundation.cost_contract.contract_hash
            or journal.scope.session_contract_hash
            != foundation.session_contract.contract_hash
        ):
            raise ValueError("resource attempt belongs to another session")
        if not _attempt_is_in_generation_prefix(
            foundation,
            journal=journal,
            reservation=reservation.payload,
            generation_index=generation_index,
            include_final_confirmation=include_final_confirmation,
        ):
            continue
        previous = known.get(journal.logical_job_id)
        if previous is not None and previous != journal.resource_binding_hash:
            raise ValueError("bound logical job changed its physical GPU")
        known[journal.logical_job_id] = journal.resource_binding_hash
    return known


def _attempt_chain_closure_evidence(
    foundation: StairGenerationSessionFoundation,
    *,
    generation_index: int | None = None,
    include_final_confirmation: bool = False,
) -> dict[str, Any]:
    _validate_resource_closure_generation_index(foundation, generation_index)
    root = foundation.config.output_dir / "job_attempts"
    reservation_count = 0
    closed_result_count = 0
    open_reservation_count = 0
    if root.exists():
        if root.is_symlink() or not root.is_dir():
            raise ValueError("job-attempt root is invalid")
        for path in sorted(root.glob("*/attempt_*.reservation.json")):
            journal, reservation = JobAttemptJournal.from_reservation_path(path)
            if (
                journal.scope.budget_contract_hash
                != foundation.cost_contract.contract_hash
                or journal.scope.session_contract_hash
                != foundation.session_contract.contract_hash
            ):
                raise ValueError("job attempt belongs to another session")
            if not _attempt_is_in_generation_prefix(
                foundation,
                journal=journal,
                reservation=reservation.payload,
                generation_index=generation_index,
                include_final_confirmation=include_final_confirmation,
            ):
                continue
            reservation_count += 1
            result_path = path.with_name(path.name.replace(".reservation.", ".result."))
            if not result_path.exists():
                open_reservation_count += 1
                continue
            journal.load_result(reservation.payload["attempt_index"])
            closed_result_count += 1
    payload = {
        "schema_version": (
            "true_rl_agent.stair.generation_attempt_chain_closure.v1"
            if generation_index is None
            else "true_rl_agent.stair.generation_attempt_chain_closure.v2"
        ),
        "source": "validated-job-attempt-journal",
        "reservation_count": reservation_count,
        "closed_result_count": closed_result_count,
        "open_reservation_count": open_reservation_count,
    }
    if generation_index is not None:
        payload["generation_index"] = generation_index
    return payload


def _validate_resource_closure_generation_index(
    foundation: StairGenerationSessionFoundation,
    generation_index: int | None,
) -> None:
    if generation_index is None:
        return
    if (
        isinstance(generation_index, bool)
        or not isinstance(generation_index, int)
        or generation_index <= 0
        or generation_index > foundation.session_contract.max_generations
        or foundation.session_contract.session_mode
        not in {
            DEVELOPMENT_GENERATION_VALIDATION_MODE,
            FORMAL_GENERATION_SESSION_MODE,
        }
    ):
        raise ValueError("strict resource closure generation index is invalid")


def _attempt_is_in_generation_prefix(
    foundation: StairGenerationSessionFoundation,
    *,
    journal: JobAttemptJournal,
    reservation: Mapping[str, Any],
    generation_index: int | None,
    include_final_confirmation: bool,
) -> bool:
    if not isinstance(include_final_confirmation, bool):
        raise TypeError("attempt final-confirmation scope must be boolean")
    attempt_index = reservation.get("attempt_index")
    if isinstance(attempt_index, bool) or not isinstance(attempt_index, int):
        raise TypeError("Stair bounded generation attempt index must be an integer")
    _strict_operation_slot(
        session_id=foundation.config.session_id,
        scope=journal.scope,
        logical_job_id=journal.logical_job_id,
        job_kind=journal.job_kind,
        attempt_index=attempt_index,
        cost_contract=foundation.cost_contract,
    )
    if isinstance(journal.scope, PreGenesisExecutionScope):
        return True
    if journal.scope.generation_id == "generation-final-confirmation":
        return include_final_confirmation
    if generation_index is None:
        return True
    if journal.scope.generation_id == "generation-parent-public-dev-bootstrap":
        return True
    match = re.fullmatch(
        re.escape(foundation.config.session_id) + r"-generation-([0-9]{3})",
        journal.scope.generation_id,
    )
    if match is None:
        raise ValueError("strict attempt generation prefix is invalid")
    attempt_generation_index = int(match.group(1))
    if not 1 <= attempt_generation_index <= foundation.session_contract.max_generations:
        raise ValueError("attempt generation exceeds its strict session")
    return attempt_generation_index <= generation_index


def _reacquire_generation_gpu_leases(
    foundation: StairGenerationSessionFoundation,
    *,
    genesis: VerifiedSessionGenesis,
    invocation_segment_id: str,
) -> dict[str, Any]:
    leases: list[Any] = []
    try:
        for lane_id, identity in zip(
            GENERATION_LANE_IDS,
            foundation.gpu_identities,
            strict=True,
        ):
            lease = foundation.gpu_lease_manager.acquire(
                identity.execution_device,
                holder=GpuLeaseHolder(
                    session_genesis_hash=genesis.session_genesis_hash,
                    generation_id=(f"{foundation.config.session_id}-resource-closure"),
                    lane_id=lane_id,
                    invocation_segment_id=invocation_segment_id,
                    logical_job_id=f"resource-closure-{lane_id}",
                ),
            )
            lease.assert_owned()
            leases.append(lease)
        return {
            "passed": True,
            "simultaneous": True,
            "resource_binding_hashes": [
                lease.resource_binding_hash for lease in leases
            ],
        }
    finally:
        for lease in reversed(leases):
            lease.release()


def _stair_generation_provider_config(
    *,
    output_dir: Path,
    env_file: Path,
) -> OpenAICompatibleProviderConfig:
    """Build the task-owned provider config bound by the generation policy."""

    return OpenAICompatibleProviderConfig(
        model=STAIR_GENERATION_PROVIDER_MODEL,
        output_dir=output_dir,
        base_url=STAIR_GENERATION_PROVIDER_BASE_URL,
        endpoint=STAIR_GENERATION_PROVIDER_ENDPOINT,
        api_key_env=STAIR_GENERATION_PROVIDER_API_KEY_ENV,
        env_file=env_file,
        max_tokens=STAIR_GENERATION_PROVIDER_OUTPUT_UNITS_PER_ATTEMPT,
        timeout_s=STAIR_GENERATION_PROVIDER_TIMEOUT_S,
        temperature=STAIR_GENERATION_PROVIDER_THINKING_DISABLED_TEMPERATURE,
        reasoning_mode="adaptive",
        reasoning_effort=STAIR_GENERATION_PROVIDER_REASONING_EFFORT,
        empty_final_reasoning_retry=True,
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


def _retry_policy() -> dict[str, Any]:
    return {
        "schema_version": "true_rl_agent.stair.generation_retry_policy.v1",
        "logical_job_max_infrastructure_retries": (
            STAIR_GENERATION_MAX_INFRASTRUCTURE_RETRIES_PER_ELIGIBLE_JOB
        ),
        "reference_train_retry_entitlement": (
            STAIR_GENERATION_MAX_INFRASTRUCTURE_RETRIES_PER_ELIGIBLE_JOB
        ),
        "reference_eval_retry_entitlement": 0,
        "parent_and_public_phase_retry_entitlement": 0,
        "winner_sealed_validation_retry_entitlement": (
            STAIR_GENERATION_MAX_INFRASTRUCTURE_RETRIES_PER_ELIGIBLE_JOB
        ),
        "retry_train_job_reserve": STAIR_GENERATION_RETRY_TRAIN_JOBS,
        "retry_eval_seed_unit_reserve": (STAIR_GENERATION_RETRY_EVAL_SEED_UNITS),
        "candidate_or_contract_failure_retryable": False,
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


def _build_validation_report(
    foundation: StairGenerationSessionFoundation,
    *,
    genesis: VerifiedSessionGenesis,
    ledger: list[dict[str, Any]],
    terminal_binding: Mapping[str, Any],
    actual_cost_receipt: StairGenerationActualCostReceiptRef,
    resource_evidence: Mapping[str, Any],
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    if created_at_utc is None:
        created_at_utc = datetime.now(timezone.utc).isoformat()
    elif not isinstance(created_at_utc, str) or not created_at_utc:
        raise ValueError("Stair bounded generation report timestamp is invalid")
    if len(ledger) != foundation.session_contract.max_generations:
        raise ValueError("Stair bounded generation report requires its full generation ledger")
    cost_journal = StairGenerationActualCostJournal(
        foundation.config.output_dir,
        contract=foundation.cost_contract,
    )
    generation_details: list[dict[str, Any]] = []
    prompt_payload_bytes = 0
    paired_ready_count = 0
    zero_train_count = 0
    post_genesis_cost = cost_journal.receipt_for_event("post-genesis")
    if post_genesis_cost is None:
        raise ValueError("Stair bounded generation report has no post-Genesis cost boundary")
    post_parent_bootstrap_cost = cost_journal.receipt_for_event("post-parent-bootstrap")
    if post_parent_bootstrap_cost is None:
        raise ValueError("Stair bounded generation report has no post-parent-bootstrap cost boundary")
    parent_bootstrap_workload_delta = _validation_workload_delta(
        post_genesis_cost.payload["actual"],
        post_parent_bootstrap_cost.payload["actual"],
    )
    previous_actual = post_parent_bootstrap_cost.payload["actual"]
    for generation_index, record in enumerate(ledger, start=1):
        expected_id = (
            f"{foundation.config.session_id}-generation-{generation_index:03d}"
        )
        if (
            record.get("generation_index") != generation_index
            or record.get("generation_id") != expected_id
        ):
            raise ValueError("Stair bounded generation report generation identity is non-contiguous")
        loaded_record, execution_ref = _load_existing_generation_execution(
            foundation,
            record=record,
        )
        cost_ref = cost_journal.receipt_for_event(
            f"generation-{generation_index:03d}-settlement"
        )
        if cost_ref is None:
            raise ValueError("bounded generation has no cost settlement")
        closure = _publish_or_validate_generation_closure(
            foundation,
            ledger=ledger,
            generation_record=loaded_record,
            generation_execution_receipt=execution_ref.to_dict(),
            actual_cost_receipt=cost_ref,
            resource_evidence={},
            segment_id="terminal-report-replay",
            require_existing=True,
        )
        current_actual = cost_ref.payload["actual"]
        workload_delta = _validation_workload_delta(
            previous_actual,
            current_actual,
        )
        previous_actual = current_actual
        pre_plan = _generation_exploration_for_ledger(
            foundation,
            genesis=genesis,
            ledger=ledger[: generation_index - 1],
        )
        post_plan = _generation_exploration_for_ledger(
            foundation,
            genesis=genesis,
            ledger=ledger[:generation_index],
        )
        trials = record.get("trials")
        artifacts = record.get("private_artifacts")
        if not isinstance(trials, list) or not isinstance(artifacts, list):
            raise TypeError("bounded generation projection is invalid")
        if len(trials) != len(GENERATION_LANE_IDS):
            raise ValueError("bounded generation does not preserve width two")
        if all(
            isinstance(trial, Mapping)
            and trial.get("train") is not None
            and trial.get("public_dev") is not None
            for trial in trials
        ):
            paired_ready_count += 1
        if all(
            isinstance(trial, Mapping) and trial.get("train") is None
            for trial in trials
        ):
            zero_train_count += 1
        for artifact in artifacts:
            if isinstance(artifact, Mapping) and artifact.get("kind") in {
                "lane-a_prompt_payload",
                "lane-b_prompt_payload",
            }:
                size_bytes = artifact.get("size_bytes")
                if (
                    isinstance(size_bytes, bool)
                    or not isinstance(size_bytes, int)
                    or size_bytes < 0
                ):
                    raise ValueError("Stair bounded generation prompt payload size is invalid")
                prompt_payload_bytes += size_bytes
        generation_details.append(
            {
                "generation_id": expected_id,
                "generation_index": generation_index,
                "generation_outcome": record["generation_outcome"],
                "parent_current_best_hash": record["parent_current_best_hash"],
                "resulting_current_best_hash": record["resulting_current_best_hash"],
                "selected_lane_id": (
                    record.get("selection", {}).get("selected_lane_id")
                    if isinstance(record.get("selection"), Mapping)
                    else None
                ),
                "generation_record_hash": canonical_json_hash(record),
                "generation_execution_receipt": execution_ref.to_dict(),
                "generation_closure": closure,
                "actual_workload_delta": workload_delta,
                "pre_exploration_plan_hash": pre_plan.plan_hash,
                "post_exploration_plan_hash": post_plan.plan_hash,
                "post_exploration_card_count": len(post_plan.index.cards),
                "post_revisit_backlog_count": post_plan.frontier.backlog_count,
                "lane_intent_hashes": [
                    canonical_json_hash(intent) for intent in pre_plan.lane_intents
                ],
            }
        )
    retry_train_jobs, retry_eval_seed_units = _session_retry_usage(foundation)
    final_exploration = _generation_exploration_for_ledger(
        foundation,
        genesis=genesis,
        ledger=ledger,
    )
    exploration_metrics = _validation_exploration_metrics(
        foundation,
        genesis=genesis,
        ledger=ledger,
    )
    authoring_metrics = _validation_authoring_metrics(foundation)
    actual = actual_cost_receipt.payload["actual"]
    bindings = [binding.to_dict() for binding in foundation.gpu_bindings]
    checks = {
        "mode_is_development_generation_validation": (
            foundation.session_contract.session_mode
            == DEVELOPMENT_GENERATION_VALIDATION_MODE
            and foundation.session_contract.max_generations == 3
        ),
        "three_contiguous_generation_records": (
            len(generation_details) == 3
            and [item["generation_index"] for item in generation_details] == [1, 2, 3]
        ),
        "generation_closures_are_non_authoritative": all(
            item["generation_closure"].get("acceptance_authority") is False
            for item in generation_details
        ),
        "two_distinct_host_gpu_bindings": (
            bindings[0]["host_execution_device"] == "cuda:0"
            and bindings[1]["host_execution_device"] == "cuda:1"
            and bindings[0]["resource_binding_hash"]
            != bindings[1]["resource_binding_hash"]
        ),
        "both_children_are_logical_cuda0": all(
            value["process_execution_device"] == "cuda:0" for value in bindings
        ),
        "final_confirmation_disabled": (
            foundation.final_confirmation_policy["enabled"] is False
        ),
        "lifecycle_terminal_at_three": (
            terminal_binding.get("closed_generation_count") == 3
            and terminal_binding.get("required_generation_count") == 3
        ),
        "actual_cost_chain_closed": (
            terminal_binding.get("cost_chain_head_hash")
            == actual_cost_receipt.payload_hash
            and actual_cost_receipt.payload["cap_checks"]["all_passed"] is True
        ),
        "provider_budget_within_session_cap": (
            actual["provider_attempts"] <= foundation.cost_contract.provider_attempt_cap
            and actual["provider_request_units"]
            <= foundation.cost_contract.to_dict()["provider"]["input_unit_cap"]
            and actual["provider_response_units"]
            <= foundation.cost_contract.to_dict()["provider"]["output_unit_cap"]
            and actual["provider_total_units"]
            <= foundation.cost_contract.to_dict()["provider"]["total_unit_cap"]
        ),
        "session_wide_retry_reserve_respected": (
            retry_train_jobs <= foundation.cost_contract.retry_train_jobs
            and retry_eval_seed_units <= foundation.cost_contract.retry_eval_seed_units
        ),
        "external_pid_uuid_and_attempt_chain_closed": (
            resource_evidence.get("startup_compute_idle", {}).get("rows") == []
            and resource_evidence.get("final_compute_idle", {}).get("rows") == []
            and resource_evidence.get("lease_reacquire", {}).get("passed") is True
            and resource_evidence.get("attempt_chain", {}).get("open_reservation_count")
            == 0
        ),
        "parent_gpu_environment_unchanged": not any(
            key in os.environ for key in _PARENT_GPU_REMAP_KEYS
        ),
    }
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise ValueError("Stair bounded generation validation report checks failed: " + ", ".join(failed))
    selector_metrics = _validation_selector_reversal_metrics(final_exploration)
    p6d_entry_gates = {
        "authoring_timing_complete": (
            authoring_metrics["timing"]["timing_complete"] is True
        ),
        "high_card_recall_is_complete": (
            exploration_metrics["high_card_recall"]["denominator"]
            == exploration_metrics["high_card_recall"]["numerator"]
        ),
        "priority_revisit_lane_service_is_complete": (
            exploration_metrics["revisit_lane_service"]["denominator"]
            == exploration_metrics["revisit_lane_service"]["numerator"]
            and not exploration_metrics["revisit_lane_service"]["mismatches"]
            and exploration_metrics["revisit_lane_service"][
                "unexpected_revisit_service_count"
            ]
            == 0
        ),
        "negative_equivalent_trained_repeat_is_zero": (
            exploration_metrics["negative_equivalent_trained_repeat_count"] == 0
        ),
        "selector_material_reversal_absent": (
            selector_metrics["blocking_reversal_count"] == 0
        ),
    }
    return {
        "schema_version": STAIR_GENERATION_VALIDATION_REPORT_SCHEMA_VERSION,
        "session_id": foundation.config.session_id,
        "session_mode": DEVELOPMENT_GENERATION_VALIDATION_MODE,
        "created_at_utc": created_at_utc,
        "source_commit": foundation.source_commit,
        "output_dir": str(foundation.config.output_dir.resolve()),
        "session_genesis_hash": genesis.session_genesis_hash,
        "generation_session_contract_hash": (foundation.session_contract.contract_hash),
        "session_cost_contract_hash": foundation.cost_contract.contract_hash,
        "provider_policy_hash": canonical_json_hash(foundation.provider_policy),
        "allocated_gpu_capacity_time_basis": _ALLOCATED_GPU_CAPACITY_TIME_BASIS,
        "gpu_process_bindings": bindings,
        "generations": generation_details,
        "terminal_binding": dict(terminal_binding),
        "final_confirmation": dict(foundation.final_confirmation_policy),
        "successful_cost_plan": foundation.cost_contract.to_dict()[
            "successful_session"
        ],
        "retry_reserve": foundation.cost_contract.to_dict()["retry_reserve"],
        "physical_cost_cap": foundation.cost_contract.to_dict()["physical_cap"],
        "actual_cost_receipt": {
            **_generation_cost_receipt_projection(actual_cost_receipt),
            "actual": dict(actual),
            "cap_checks": dict(actual_cost_receipt.payload["cap_checks"]),
            "conservative_charges": list(
                actual_cost_receipt.payload["conservative_charges"]
            ),
        },
        "session_metrics": {
            "parent_bootstrap_workload_delta": parent_bootstrap_workload_delta,
            "generation_outcome_counts": dict(
                Counter(item["generation_outcome"] for item in generation_details)
            ),
            "paired_ready_generation_count": paired_ready_count,
            "paired_ready_rate": paired_ready_count / len(generation_details),
            "zero_train_authoring_incomplete_generation_count": zero_train_count,
            "immutable_prompt_payload_bytes": prompt_payload_bytes,
            "retry_train_jobs_consumed": retry_train_jobs,
            "retry_eval_seed_units_consumed": retry_eval_seed_units,
            "additional_eval_seed_units_vs_success_plan": max(
                0,
                actual["eval_seed_units"]
                - foundation.cost_contract.successful_eval_seed_units,
            ),
            "additional_train_jobs_vs_success_plan": max(
                0,
                actual["train_jobs"] - foundation.cost_contract.successful_train_jobs,
            ),
            "final_exploration_card_count": len(final_exploration.index.cards),
            "final_revisit_backlog_count": final_exploration.frontier.backlog_count,
            "exploration": exploration_metrics,
            "authoring": authoring_metrics,
            "selector_reversal": selector_metrics,
        },
        "resource_evidence": dict(resource_evidence),
        "checks": checks,
        "p6d_entry_gates": p6d_entry_gates,
        "acceptance_authority": False,
    }


def _validation_exploration_metrics(
    foundation: StairGenerationSessionFoundation,
    *,
    genesis: VerifiedSessionGenesis,
    ledger: list[dict[str, Any]],
) -> dict[str, Any]:
    final_plan = _generation_exploration_for_ledger(
        foundation,
        genesis=genesis,
        ledger=ledger,
    )
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
    history = project_public_history(
        ledger,
        initial_current_best=genesis.initial_current_best,
        expected_session_genesis_hash=genesis.session_genesis_hash,
        acceptance_protocol=(
            foundation.composition.task_plugin.task_spec.acceptance_protocol
        ),
        expected_initial_policy_state_hash=(genesis.expected_initial_policy_state_hash),
        expected_authoring_mode=PROVIDER_AUTHORING_MODE,
        expected_ledger_epoch_hash=primary_ledger_epoch_hash(
            genesis.session_genesis_hash
        ),
    )
    expected_high_losers: set[tuple[str, str]] = set()
    projector = StairExplorationProjector(foundation.composition.task_plugin.config)
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
    direction_consistency = [
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
        plan = _generation_exploration_for_ledger(
            foundation,
            genesis=genesis,
            ledger=ledger[:prefix_height],
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
            "numerator": sum(value is True for value in direction_consistency),
            "denominator": len(direction_consistency),
            "rate": (
                sum(value is True for value in direction_consistency)
                / len(direction_consistency)
                if direction_consistency
                else None
            ),
        },
    }


def _validation_authoring_metrics(
    foundation: StairGenerationSessionFoundation,
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
    }


def _validation_selector_reversal_metrics(
    final_exploration: GenerationExplorationPlan,
) -> dict[str, Any]:
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


def _build_report(
    foundation: StairGenerationSessionFoundation,
    *,
    genesis: VerifiedSessionGenesis,
    generation_record: Mapping[str, Any],
    generation_execution_receipt: Mapping[str, Any],
    generation_committed_or_reused: bool,
    pre_exploration: GenerationExplorationPlan,
    post_exploration: GenerationExplorationPlan,
    terminal_binding: Mapping[str, Any],
    ledger: list[dict[str, Any]],
    actual_cost_receipt: StairGenerationActualCostReceiptRef,
    resource_evidence: Mapping[str, Any],
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    if created_at_utc is None:
        created_at_utc = datetime.now(timezone.utc).isoformat()
    elif not isinstance(created_at_utc, str) or not created_at_utc:
        raise ValueError("Stair generation report timestamp is invalid")
    bindings = [binding.to_dict() for binding in foundation.gpu_bindings]
    checks = {
        "mode_is_development_generation_pilot": (
            foundation.session_contract.session_mode
            == DEVELOPMENT_GENERATION_PILOT_MODE
        ),
        "fresh_reference_genesis_bound": (
            genesis.artifact.genesis_kind == "reference_program"
            and genesis.reference_eval_plan() is not None
            and genesis.reference_eval_plan().gpu_process_binding
            == foundation.gpu_bindings[0]
        ),
        "two_distinct_host_gpu_bindings": (
            bindings[0]["host_execution_device"] == "cuda:0"
            and bindings[1]["host_execution_device"] == "cuda:1"
            and bindings[0]["resource_binding_hash"]
            != bindings[1]["resource_binding_hash"]
        ),
        "both_children_are_logical_cuda0": all(
            value["process_execution_device"] == "cuda:0" for value in bindings
        ),
        "one_generation_one_ledger_record": len(ledger) == 1,
        "generation_committed_or_exactly_reused": (generation_committed_or_reused),
        "final_confirmation_disabled": (
            foundation.final_confirmation_policy["enabled"] is False
        ),
        "lifecycle_terminal": (terminal_binding.get("closed_generation_count") == 1),
        "actual_cost_chain_closed": (
            terminal_binding.get("cost_chain_head_hash")
            == actual_cost_receipt.payload_hash
            and actual_cost_receipt.payload["cap_checks"]["all_passed"] is True
        ),
        "external_pid_uuid_and_attempt_chain_closed": (
            resource_evidence.get("startup_compute_idle", {}).get("rows") == []
            and resource_evidence.get("final_compute_idle", {}).get("rows") == []
            and resource_evidence.get("lease_reacquire", {}).get("passed") is True
            and resource_evidence.get("completed_bound_job_count")
            == len(resource_evidence.get("summaries", []))
            and resource_evidence.get("attempt_chain", {}).get("open_reservation_count")
            == 0
        ),
        "parent_gpu_environment_unchanged": not any(
            key in os.environ for key in _PARENT_GPU_REMAP_KEYS
        ),
    }
    if not all(checks.values()):
        raise ValueError("Stair generation session report checks failed")
    return {
        "schema_version": STAIR_GENERATION_REPORT_SCHEMA_VERSION,
        "session_id": foundation.config.session_id,
        "session_mode": DEVELOPMENT_GENERATION_PILOT_MODE,
        "created_at_utc": created_at_utc,
        "source_commit": foundation.source_commit,
        "output_dir": str(foundation.config.output_dir.resolve()),
        "session_genesis_hash": genesis.session_genesis_hash,
        "generation_session_contract_hash": (foundation.session_contract.contract_hash),
        "session_cost_contract_hash": foundation.cost_contract.contract_hash,
        "allocated_gpu_capacity_time_basis": (_ALLOCATED_GPU_CAPACITY_TIME_BASIS),
        "provider_policy_hash": canonical_json_hash(foundation.provider_policy),
        "gpu_process_bindings": bindings,
        "generation_id": generation_record["generation_id"],
        "generation_outcome": generation_record["generation_outcome"],
        "generation_execution_receipt": dict(generation_execution_receipt),
        "pre_generation_exploration_plan_hash": pre_exploration.plan_hash,
        "post_generation_exploration_plan_hash": post_exploration.plan_hash,
        "post_generation_exploration_card_count": len(post_exploration.index.cards),
        "post_generation_revisit_backlog_count": (
            post_exploration.frontier.backlog_count
        ),
        "terminal_binding": dict(terminal_binding),
        "final_confirmation": dict(foundation.final_confirmation_policy),
        "successful_cost_plan": foundation.cost_contract.to_dict()[
            "successful_session"
        ],
        "physical_cost_cap": foundation.cost_contract.to_dict()["physical_cap"],
        "actual_cost_receipt": {
            "path": str(actual_cost_receipt.path.resolve()),
            "payload_hash": actual_cost_receipt.payload_hash,
            "event_id": actual_cost_receipt.payload["event_id"],
            "actual": dict(actual_cost_receipt.payload["actual"]),
            "cap_checks": dict(actual_cost_receipt.payload["cap_checks"]),
            "conservative_charges": list(
                actual_cost_receipt.payload["conservative_charges"]
            ),
        },
        "resource_evidence": dict(resource_evidence),
        "checks": checks,
        "acceptance_authority": False,
    }


def _load_and_validate_existing_report(
    foundation: StairGenerationSessionFoundation,
    *,
    genesis: VerifiedSessionGenesis,
    cost_journal: StairGenerationActualCostJournal,
    startup_resource_observation: Mapping[str, Any],
) -> dict[str, Any]:
    path = foundation.config.output_dir / STAIR_GENERATION_REPORT_FILENAME
    report = load_strict_json_object(
        path,
        context="Stair generation session report",
    )
    if path.read_bytes() != canonical_json_bytes(report, trailing_newline=True):
        raise ValueError("Stair generation report is not canonical JSON")
    created_at_utc = report.get("created_at_utc")
    if not isinstance(created_at_utc, str) or not created_at_utc:
        raise ValueError("Stair generation report timestamp is invalid")
    ledger = load_ledger(foundation.config.output_dir / "ledger.jsonl")
    if len(ledger) != 1:
        raise ValueError("Stair generation report ledger height mismatch")
    record, execution_ref = _load_existing_generation_execution(
        foundation,
        record=ledger[0],
    )
    lifecycle_identity = GenerationLifecycleIdentity(
        session_id=foundation.config.session_id,
        session_genesis_hash=genesis.session_genesis_hash,
        generation_session_contract_hash=(foundation.session_contract.contract_hash),
        final_confirmation_plan_hash=foundation.session_contract.bound_contract_hashes[
            "final_confirmation_plan"
        ],
        ledger_epoch_hash=primary_ledger_epoch_hash(genesis.session_genesis_hash),
        output_root=str(foundation.config.output_dir.resolve()),
    )
    terminal = build_generation_terminal_binding(
        ledger_path=foundation.config.output_dir / "ledger.jsonl",
        output_dir=foundation.config.output_dir,
        attempt_journal_root=foundation.config.output_dir / "job_attempts",
        identity=lifecycle_identity,
        policy=foundation.continuation_policy,
        session_contract=foundation.session_contract,
        initial_current_best=genesis.initial_current_best,
        acceptance_protocol=(
            foundation.composition.task_plugin.task_spec.acceptance_protocol
        ),
    )
    chain = cost_journal.load_chain()
    if not chain:
        raise ValueError("Stair generation report has no actual-cost chain")
    cost_ref = chain[-1]
    current_usage, conservative_charges, open_reservations = (
        _collect_actual_cost_usage(
            foundation,
            genesis=genesis,
            ledger=ledger,
        )
    )
    if open_reservations:
        raise ValueError("Stair generation report has open job attempts")
    sealed_actual = cost_ref.payload["actual"]
    for field_name, current in current_usage.to_dict().items():
        if field_name in {"observed_output_bytes", "observed_output_files"}:
            continue
        if sealed_actual.get(field_name) != current:
            raise ValueError(
                f"Stair generation report actual cost drifted: {field_name}"
            )
    if current_usage.observed_output_bytes > sealed_actual["accounted_output_bytes"]:
        raise ValueError("Stair generation report exceeded output reserve")
    current_resource_evidence = _build_resource_closure_evidence(
        foundation,
        genesis=genesis,
        invocation_segment_id="terminal-report-validation",
        startup_observation=startup_resource_observation,
    )
    pre = _generation_exploration_for_ledger(
        foundation,
        genesis=genesis,
        ledger=[],
    )
    post = _generation_exploration_for_ledger(
        foundation,
        genesis=genesis,
        ledger=ledger,
    )
    expected = _build_report(
        foundation,
        genesis=genesis,
        generation_record=record,
        generation_execution_receipt=execution_ref.to_dict(),
        generation_committed_or_reused=True,
        pre_exploration=pre,
        post_exploration=post,
        terminal_binding=terminal.to_dict(),
        ledger=ledger,
        actual_cost_receipt=cost_ref,
        resource_evidence=current_resource_evidence,
        created_at_utc=created_at_utc,
    )
    if report != expected:
        raise ValueError("Stair generation report differs from recomputation")
    return report


def _assert_validation_cost_receipt_is_current(
    foundation: StairGenerationSessionFoundation,
    *,
    genesis: VerifiedSessionGenesis,
    ledger: list[dict[str, Any]],
    cost_journal: StairGenerationActualCostJournal,
    lifecycle: GenerationLifecycleAudit,
    cost_ref: StairGenerationActualCostReceiptRef,
) -> StairGenerationActualCostUsage:
    combined_head = _replay_validation_combined_cost_chain(
        cost_journal,
        lifecycle,
    )
    if combined_head != cost_ref.payload_hash:
        raise ValueError("Stair bounded generation terminal actual-cost chain head mismatch")
    current_usage, conservative_charges, open_reservations = _collect_actual_cost_usage(
        foundation,
        genesis=genesis,
        ledger=ledger,
    )
    if open_reservations:
        raise ValueError("Stair bounded generation terminal state has open job attempts")
    _assert_actual_cost_receipt_matches_usage(
        cost_ref,
        usage=current_usage,
        conservative_charges=conservative_charges,
        context="bounded generation session report",
    )
    return current_usage


def _finalize_missing_terminal_validation_report(
    foundation: StairGenerationSessionFoundation,
    *,
    genesis: VerifiedSessionGenesis,
    cost_journal: StairGenerationActualCostJournal,
) -> dict[str, Any]:
    """Create the report after a crash between terminal closure and report write."""

    report_path = foundation.config.output_dir / STAIR_GENERATION_REPORT_FILENAME
    if report_path.exists() or report_path.is_symlink():
        raise FileExistsError("Stair bounded generation terminal finalization requires a missing report")
    ledger_path = foundation.config.output_dir / "ledger.jsonl"
    ledger = load_ledger(ledger_path)
    if len(ledger) != foundation.session_contract.max_generations:
        raise ValueError("Stair bounded generation terminal finalization ledger height mismatch")
    lifecycle_identity = _validation_lifecycle_identity(
        foundation,
        genesis=genesis,
    )
    lifecycle = audit_generation_lifecycle(
        output_dir=foundation.config.output_dir,
        identity=lifecycle_identity,
        policy=foundation.continuation_policy,
        require_closed=True,
    )
    if not lifecycle.terminal:
        raise ValueError("Stair bounded generation terminal finalization requires a terminal lifecycle")
    chain = cost_journal.load_chain()
    if not chain:
        raise ValueError("Stair bounded generation terminal finalization has no actual-cost chain")
    cost_ref = chain[-1]
    if lifecycle.cost_chain_head_hash != cost_ref.payload_hash:
        raise ValueError("Stair bounded generation terminal finalization cost head mismatch")
    current_usage = _assert_validation_cost_receipt_is_current(
        foundation,
        genesis=genesis,
        ledger=ledger,
        cost_journal=cost_journal,
        lifecycle=lifecycle,
        cost_ref=cost_ref,
    )
    terminal = build_generation_terminal_binding(
        ledger_path=ledger_path,
        output_dir=foundation.config.output_dir,
        attempt_journal_root=foundation.config.output_dir / "job_attempts",
        identity=lifecycle_identity,
        policy=foundation.continuation_policy,
        session_contract=foundation.session_contract,
        initial_current_best=genesis.initial_current_best,
        acceptance_protocol=(
            foundation.composition.task_plugin.task_spec.acceptance_protocol
        ),
        expected_initial_policy_state_hash=(genesis.expected_initial_policy_state_hash),
    )
    lifecycle = audit_generation_lifecycle(
        output_dir=foundation.config.output_dir,
        identity=lifecycle_identity,
        policy=foundation.continuation_policy,
        require_closed=True,
    )
    last_record, execution_ref = _load_existing_generation_execution(
        foundation,
        record=ledger[-1],
    )
    last_generation_cost = cost_journal.receipt_for_event(
        f"generation-{len(ledger):03d}-settlement"
    )
    if last_generation_cost is None:
        raise ValueError(
            "Stair bounded generation terminal finalization has no last-generation cost settlement"
        )
    closure = _publish_or_validate_generation_closure(
        foundation,
        ledger=ledger,
        generation_record=last_record,
        generation_execution_receipt=execution_ref.to_dict(),
        actual_cost_receipt=last_generation_cost,
        resource_evidence={},
        segment_id="terminal-finalize",
        require_existing=True,
    )
    resource_evidence = closure["resource_evidence"]
    _validate_stored_resource_closure_evidence(
        foundation,
        resource_evidence=resource_evidence,
        generation_index=len(ledger),
    )
    report = _build_validation_report(
        foundation,
        genesis=genesis,
        ledger=ledger,
        terminal_binding=terminal.to_dict(),
        actual_cost_receipt=cost_ref,
        resource_evidence=resource_evidence,
    )
    report_payload = canonical_json_bytes(report, trailing_newline=True)
    if (
        current_usage.observed_output_bytes + len(report_payload)
        > cost_ref.payload["actual"]["accounted_output_bytes"]
    ):
        raise ValueError("Stair bounded generation terminal report exceeds its finalization reserve")
    create_once_bytes(report_path, report_payload)
    return _load_and_validate_existing_validation_report(
        foundation,
        genesis=genesis,
        cost_journal=cost_journal,
        startup_resource_observation=None,
    )


def _load_and_validate_existing_validation_report(
    foundation: StairGenerationSessionFoundation,
    *,
    genesis: VerifiedSessionGenesis,
    cost_journal: StairGenerationActualCostJournal,
    startup_resource_observation: Mapping[str, Any] | None,
) -> dict[str, Any]:
    path = foundation.config.output_dir / STAIR_GENERATION_REPORT_FILENAME
    report = load_strict_json_object(
        path,
        context="bounded generation session report",
    )
    if path.read_bytes() != canonical_json_bytes(report, trailing_newline=True):
        raise ValueError("bounded generation report is not canonical JSON")
    if (
        report.get("schema_version")
        != STAIR_GENERATION_VALIDATION_REPORT_SCHEMA_VERSION
    ):
        raise ValueError("bounded generation report schema mismatch")
    created_at_utc = report.get("created_at_utc")
    if not isinstance(created_at_utc, str) or not created_at_utc:
        raise ValueError("bounded generation report timestamp is invalid")
    ledger_path = foundation.config.output_dir / "ledger.jsonl"
    ledger = load_ledger(ledger_path)
    if len(ledger) != foundation.session_contract.max_generations:
        raise ValueError("bounded generation report ledger height mismatch")
    lifecycle_identity = _validation_lifecycle_identity(
        foundation,
        genesis=genesis,
    )
    lifecycle = audit_generation_lifecycle(
        output_dir=foundation.config.output_dir,
        identity=lifecycle_identity,
        policy=foundation.continuation_policy,
        require_closed=True,
    )
    if not lifecycle.terminal:
        raise ValueError("bounded generation report requires a terminal lifecycle")
    terminal = build_generation_terminal_binding(
        ledger_path=ledger_path,
        output_dir=foundation.config.output_dir,
        attempt_journal_root=foundation.config.output_dir / "job_attempts",
        identity=lifecycle_identity,
        policy=foundation.continuation_policy,
        session_contract=foundation.session_contract,
        initial_current_best=genesis.initial_current_best,
        acceptance_protocol=(
            foundation.composition.task_plugin.task_spec.acceptance_protocol
        ),
        expected_initial_policy_state_hash=(genesis.expected_initial_policy_state_hash),
    )
    chain = cost_journal.load_chain()
    if not chain:
        raise ValueError("bounded generation report has no actual-cost chain")
    cost_ref = chain[-1]
    _assert_validation_cost_receipt_is_current(
        foundation,
        genesis=genesis,
        ledger=ledger,
        cost_journal=cost_journal,
        lifecycle=lifecycle,
        cost_ref=cost_ref,
    )
    if startup_resource_observation is None:
        resource_evidence = report.get("resource_evidence")
        if not isinstance(resource_evidence, Mapping):
            raise ValueError("Stair bounded generation report resource evidence is invalid")
        _validate_stored_resource_closure_evidence(
            foundation,
            resource_evidence=resource_evidence,
            generation_index=len(ledger),
        )
    else:
        resource_evidence = _build_resource_closure_evidence(
            foundation,
            genesis=genesis,
            invocation_segment_id="terminal-report-validation",
            startup_observation=startup_resource_observation,
            generation_index=len(ledger),
        )
    expected = _build_validation_report(
        foundation,
        genesis=genesis,
        ledger=ledger,
        terminal_binding=terminal.to_dict(),
        actual_cost_receipt=cost_ref,
        resource_evidence=resource_evidence,
        created_at_utc=created_at_utc,
    )
    if report != expected:
        raise ValueError("bounded generation report differs from recomputation")
    return report


def _load_and_validate_existing_formal_report(
    foundation: StairGenerationSessionFoundation,
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
        config=foundation.composition.task_plugin.config,
    )
    generation_cost = _cost_receipt_by_hash(
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


def _reject_parent_gpu_remap() -> None:
    present = [key for key in _PARENT_GPU_REMAP_KEYS if key in os.environ]
    if present:
        raise ValueError(
            "generation parent environment cannot carry GPU visibility remaps: "
            + ", ".join(present)
        )


def _preflight_provider_credential(env_file: Path) -> None:
    """Validate the fixed secret boundary without network I/O or disclosure."""

    validate_stair_provider_credential(
        env_file=env_file,
        api_key_env=STAIR_GENERATION_PROVIDER_API_KEY_ENV,
    )


def _require_stair_generation_provider_readiness(
    config: StairGenerationPilotConfig,
) -> dict[str, Any]:
    return require_stair_provider_readiness(
        base_url=STAIR_GENERATION_PROVIDER_BASE_URL,
        api_key_env=STAIR_GENERATION_PROVIDER_API_KEY_ENV,
        env_file=config.provider_env_file,
    )


def _generation_provider_readiness_required(
    config: StairGenerationPilotConfig,
) -> bool:
    # P6-D formal launch readiness validates only the local credential boundary.
    # A live balance request is a development-session gate and must not become an
    # unrecorded network side effect of the formal child process.
    if config.session_mode == FORMAL_GENERATION_SESSION_MODE:
        return False
    if not config.resume_existing:
        return True
    if (config.output_dir / STAIR_GENERATION_REPORT_FILENAME).is_file():
        return False
    ledger_path = config.output_dir / "ledger.jsonl"
    if not ledger_path.is_file():
        return True
    ledger_height = len(load_ledger(ledger_path))
    if (
        config.pause_after_generation is not None
        and ledger_height >= config.pause_after_generation
    ):
        return False
    return ledger_height < config.max_generations


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


def _publish_or_verify_source_checkout_identity(
    output_dir: Path,
    *,
    source_commit: str,
) -> dict[str, Any]:
    """Bind one Stair generation output root to its original source checkout."""

    if _SOURCE_COMMIT.fullmatch(source_commit) is None:
        raise ValueError("source checkout identity requires a lowercase Git SHA")
    body = {
        "schema_version": STAIR_GENERATION_SOURCE_CHECKOUT_SCHEMA_VERSION,
        "source_commit": source_commit,
    }
    payload = {
        **body,
        "source_checkout_hash": canonical_json_hash(body),
    }
    encoded = canonical_json_bytes(payload, trailing_newline=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / STAIR_GENERATION_SOURCE_CHECKOUT_FILENAME
    if path.exists() or path.is_symlink():
        observed = read_bounded_regular_file(
            path,
            max_bytes=4096,
            label="Stair generation source checkout identity",
        )
        if observed != encoded:
            raise ValueError(
                "existing Stair generation source checkout identity differs"
            )
    else:
        create_once_bytes(path, encoded)
    return payload


__all__ = [
    "STAIR_GENERATION_PREFLIGHT_SCHEMA_VERSION",
    "STAIR_GENERATION_REPORT_FILENAME",
    "STAIR_GENERATION_REPORT_SCHEMA_VERSION",
    "STAIR_GENERATION_SOURCE_CHECKOUT_FILENAME",
    "STAIR_GENERATION_SOURCE_CHECKOUT_SCHEMA_VERSION",
    "StairGenerationPilotConfig",
    "StairGenerationSessionFoundation",
    "discover_two_gpu_inventory",
    "preflight_stair_generation_pilot",
    "run_stair_generation_pilot",
]
