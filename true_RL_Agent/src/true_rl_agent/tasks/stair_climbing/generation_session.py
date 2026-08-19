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
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
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
    GenerationContinuationPolicy,
    GenerationLifecycleIdentity,
    GenerationSegmentJournal,
    audit_generation_lifecycle,
    build_generation_terminal_binding,
    publish_generation_continuation_policy,
)
from true_rl_agent.agent_core.generation_receipts import GenerationReceiptJournal
from true_rl_agent.agent_core.ledger import (
    PROVIDER_AUTHORING_MODE,
    load_ledger,
    primary_ledger_epoch_hash,
)
from true_rl_agent.agent_core.openai_compatible_provider import (
    OpenAICompatibleCandidateProvider,
    OpenAICompatibleProviderConfig,
    load_env_file,
)
from true_rl_agent.agent_core.round import RoundInputs
from true_rl_agent.agent_core.session_genesis import (
    REFERENCE_EVAL_ARTIFACT_FILENAME,
    SESSION_GENESIS_FILENAME,
    VerifiedSessionGenesis,
    prepare_session_genesis,
)
from true_rl_agent.agent_core.trial_receipts import (
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
from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.fs_contract import create_once_bytes
from true_rl_agent.json_contract import (
    canonical_json_bytes,
    load_strict_json_object,
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
from true_rl_agent.runner.specs import GpuProcessBinding
from true_rl_agent.runner.subprocess_runner import SubprocessRunner
from true_rl_agent.tasks.stair_climbing.composition import (
    StairClimbingCompositionProvider,
)
from true_rl_agent.tasks.stair_climbing.exploration import (
    StairExplorationProjector,
)
from true_rl_agent.tasks.stair_climbing.generation_bootstrap import (
    run_genesis_parent_public_dev,
)
from true_rl_agent.tasks.stair_climbing.generation_session_cost import (
    DEVELOPMENT_GENERATION_PILOT_MODE,
    STAIR_GENERATION_MAX_INFRASTRUCTURE_RETRIES_PER_ELIGIBLE_JOB,
    STAIR_GENERATION_PROVIDER_ATTEMPTS_PER_LANE,
    STAIR_GENERATION_PROVIDER_INPUT_UNITS_PER_ATTEMPT,
    STAIR_GENERATION_PROVIDER_OUTPUT_UNITS_PER_ATTEMPT,
    STAIR_GENERATION_RETRY_EVAL_SEED_UNITS,
    STAIR_GENERATION_RETRY_TRAIN_JOBS,
    STAIR_GENERATION_SESSION_COST_FILENAME,
    StairGenerationActualCostJournal,
    StairGenerationActualCostReceiptRef,
    StairGenerationActualCostUsage,
    StairGenerationSessionCostContract,
    build_stair_generation_scientific_cost_basis,
    create_stair_generation_session_cost_contract,
    load_stair_generation_session_cost_contract,
)
from true_rl_agent.tasks.stair_climbing.gpu_process_binding import (
    validate_stair_gpu_runtime_summary_evidence,
)
from true_rl_agent.tasks.stair_climbing.phased_adapter import (
    StairPhasedEvaluationAdapter,
)

STAIR_GENERATION_REPORT_SCHEMA_VERSION = (
    "true_rl_agent.stair.generation_session_report.v2"
)
STAIR_GENERATION_REPORT_FILENAME = "generation_session_report.json"
STAIR_GENERATION_PROVIDER_TIMEOUT_S = 300
STAIR_GENERATION_PROVIDER_MODEL = "deepseek-v4-flash"
STAIR_GENERATION_PROVIDER_BASE_URL = "https://api.deepseek.com"
STAIR_GENERATION_PROVIDER_ENDPOINT = "/chat/completions"
STAIR_GENERATION_PROVIDER_API_KEY_ENV = "DEEPSEEK_API_KEY"
_ALLOCATED_GPU_CAPACITY_TIME_BASIS = (
    "sum_of_bound_operation_resource_lease_wall_duration_ns;"
    "not_utilization_weighted;excludes_coordination_only_lease_time"
)
_FULL_GPU_UUID = re.compile(
    r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\Z"
)
_SOURCE_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_PARENT_GPU_REMAP_KEYS = ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES")


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


@dataclass(frozen=True)
class StairGenerationSessionFoundation:
    config: StairGenerationPilotConfig
    source_commit: str
    composition: ResolvedTaskComposition
    runner: SubprocessRunner
    phased_adapter: StairPhasedEvaluationAdapter
    continuation_policy: GenerationContinuationPolicy
    cost_contract: StairGenerationSessionCostContract
    session_contract: GenerationSessionContract
    session_contract_ref: GenerationSessionContractRef
    gpu_identities: tuple[GpuDeviceIdentity, GpuDeviceIdentity]
    gpu_bindings: tuple[GpuProcessBinding, GpuProcessBinding]
    gpu_lease_manager: GpuLeaseManager
    provider_policy: Mapping[str, Any]
    final_confirmation_policy: Mapping[str, Any]


@dataclass(frozen=True)
class _StairCostGuardedProvider:
    """Apply the frozen session provider cap immediately before transport."""

    delegate: OpenAICompatibleCandidateProvider
    output_dir: Path
    generation_id: str
    lane_id: str

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
        )
        return self.delegate.next_candidate(prompt_payload=prompt_payload)


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
    genesis_identity_hash = canonical_json_hash(genesis_identity_snapshot)
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
    final_report_path = config.output_dir / STAIR_GENERATION_REPORT_FILENAME
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
    path = config.output_dir / STAIR_GENERATION_REPORT_FILENAME
    create_once_bytes(
        path,
        canonical_json_bytes(report, trailing_newline=True),
    )
    return report


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
    if stage not in {"startup", "final"}:
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
    values.update(
        {
            "mode": DEVELOPMENT_GENERATION_PILOT_MODE,
            "session_id": config.session_id,
            "final_confirmation_plan": None,
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
        max_segments=2,
        allowed_continuation_reasons=(
            "operator_pause",
            "typed_infrastructure_failure",
            "exception",
            "budget_exhausted",
            "contract_incomplete",
        ),
    )
    publish_generation_continuation_policy(config.output_dir, continuation)
    provider_policy = _provider_policy()
    final_confirmation_policy = {
        "schema_version": (
            "true_rl_agent.stair.generation_final_confirmation_policy.v1"
        ),
        "session_mode": DEVELOPMENT_GENERATION_PILOT_MODE,
        "enabled": False,
        "reason": "development_generation_has_no_terminal_confirmation",
    }
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
        session_mode=DEVELOPMENT_GENERATION_PILOT_MODE,
        session_id=config.session_id,
        task_id=composition.task_id,
        backend_id=composition.backend_adapter.backend_id,
        max_generations=1,
        max_trials=2,
        max_train_jobs=2,
        bound_contract_hashes={
            "authoring_policy": canonical_json_hash(_authoring_policy(composition)),
            "continuation_policy": continuation.policy_hash,
            "evaluation_cost": canonical_json_hash(phased_adapter.cost_contract()),
            "evaluation_protocol": (
                composition.task_plugin.config.evaluation_protocol().protocol_hash
            ),
            "exploration_projector": canonical_json_hash(projector_contract),
            "final_confirmation_plan": canonical_json_hash(final_confirmation_policy),
            "provider_policy": canonical_json_hash(provider_policy),
            "resource_plan": generation_resource_plan_hash(
                resource_assignments  # type: ignore[arg-type]
            ),
            "retry_policy": canonical_json_hash(_retry_policy()),
            "selector": canonical_json_hash(phased_adapter.selection_contract()),
            "session_cost": cost.contract_hash,
        },
    )
    session_contract_ref = publish_generation_session_contract(
        config.output_dir,
        session_contract,
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
    )


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


def _build_generation_inputs(
    foundation: StairGenerationSessionFoundation,
    *,
    genesis: VerifiedSessionGenesis,
    guard: Any,
    segment_id: str,
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
    generation_id = f"{foundation.config.session_id}-generation-001"
    lane_scopes = tuple(
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
    sealed_scopes = tuple(
        _job_attempt_scope(
            foundation,
            genesis=genesis,
            generation_id=generation_id,
            lane_id=lane_id,
            invocation_segment_id=segment_id,
            binding=binding,
            max_infrastructure_retries=(
                STAIR_GENERATION_MAX_INFRASTRUCTURE_RETRIES_PER_ELIGIBLE_JOB
            ),
        )
        for lane_id, binding in zip(
            GENERATION_LANE_IDS,
            foundation.gpu_bindings,
            strict=True,
        )
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
    parent = genesis.initial_current_best
    manifest = GenerationManifest(
        schema_version=GENERATION_MANIFEST_V2_SCHEMA_VERSION,
        generation_id=generation_id,
        generation_index=1,
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
                OpenAICompatibleProviderConfig(
                    model=STAIR_GENERATION_PROVIDER_MODEL,
                    output_dir=(
                        foundation.config.output_dir / "provider_candidates" / lane_id
                    ),
                    base_url=STAIR_GENERATION_PROVIDER_BASE_URL,
                    endpoint=STAIR_GENERATION_PROVIDER_ENDPOINT,
                    api_key_env=STAIR_GENERATION_PROVIDER_API_KEY_ENV,
                    env_file=foundation.config.provider_env_file,
                    max_tokens=(STAIR_GENERATION_PROVIDER_OUTPUT_UNITS_PER_ATTEMPT),
                    timeout_s=STAIR_GENERATION_PROVIDER_TIMEOUT_S,
                    reasoning_mode="enabled",
                    reasoning_effort="max",
                )
            ),
            output_dir=foundation.config.output_dir,
            generation_id=generation_id,
            lane_id=lane_id,
        )
        for lane_id in GENERATION_LANE_IDS
    }
    readiness = composition.authoring_capability_readiness.to_prompt_mapping()
    smoke_adapter = composition.task_plugin.component_smoke_adapter()

    def author(lane_id: str, exclusion: Any) -> Any:
        lane_index = GENERATION_LANE_IDS.index(lane_id)
        trial_id = f"{generation_id}-{lane_id}"
        return CandidateAuthoringLoop(
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
        ).run_round(round_id=trial_id, records=records)

    def round_inputs(lane_id: str, authored: Any) -> RoundInputs:
        lane_index = GENERATION_LANE_IDS.index(lane_id)
        binding = foundation.gpu_bindings[lane_index]
        attempt_scope = lane_scopes[lane_index]
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
                foundation.config.output_dir / "candidate_artifacts" / lane_id
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
            train_attempt_scope=attempt_scope,
            eval_attempt_scope=attempt_scope,
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
            attempt_scope=lane_scopes[index],
            sealed_attempt_scope=sealed_scopes[index],
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
    manifest = GenerationManifest.from_mapping(record.get("manifest"))
    journal = GenerationReceiptJournal(
        foundation.config.output_dir / "generation_receipts" / manifest.generation_id,
        manifest=manifest,
    )
    loaded = journal.load_execution()
    if loaded is None or loaded[0] != dict(record):
        raise ValueError("terminal generation execution receipt is missing")
    return loaded


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
    usage = StairGenerationActualCostUsage(
        provider_attempts=provider["attempts"],
        provider_request_units=provider["request_units"],
        provider_response_units=provider["response_units"],
        provider_total_units=provider["total_units"],
        provider_reasoning_units=provider["reasoning_units"],
        provider_conservative_attempts=provider["conservative_attempts"],
        job_attempts=job_attempts,
        train_jobs=train_jobs,
        train_steps=train_steps,
        eval_seed_units=eval_seed_units,
        simulated_control_steps=train_steps + eval_control_steps,
        resource_lease_wall_duration_ns=lease_wall_ns,
        unmeasured_bound_attempts=unmeasured_bound_attempts,
        observed_output_bytes=output_bytes,
        observed_output_files=output_files,
        job_cost_evidence_hash=job_cost_evidence_hash,
    )
    return usage, tuple(sorted(set(conservative))), tuple(open_reservations)


def _assert_provider_request_budget(
    *,
    output_dir: Path,
    generation_id: str,
    current_lane_id: str,
    current_prompt_payload: Mapping[str, Any],
) -> None:
    if current_lane_id not in GENERATION_LANE_IDS:
        raise ValueError("provider budget gate lane is invalid")
    attempt_count = 0
    request_units = 0
    response_units = 0
    total_units = 0
    current_attempt_seen = False
    for lane_id in GENERATION_LANE_IDS:
        round_id = f"{generation_id}-{lane_id}"
        round_dir = output_dir / "authoring" / generation_id / lane_id
        if not round_dir.exists():
            if lane_id == current_lane_id:
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
        for index, attempt in enumerate(attempts):
            attempt_count += 1
            is_current = lane_id == current_lane_id and index == len(attempts) - 1
            status = attempt.get("status")
            if is_current:
                if current_attempt_seen or status != "interrupted_after_prompt":
                    raise ValueError(
                        "current provider attempt is not its unique prompt reservation"
                    )
                prompt_ref = PromptPayloadRef.from_mapping(
                    attempt.get("prompt_payload")
                )
                if verify_prompt_payload(prompt_ref) != dict(current_prompt_payload):
                    raise ValueError("current provider prompt reservation mismatch")
                current_attempt_seen = True
                continue
            if status == "interrupted_before_prompt":
                continue
            if status == "interrupted_after_prompt":
                raise RuntimeError(
                    "prior provider request has no authoritative usage receipt"
                )
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
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError("prior provider usage counter is invalid")
                values.append(value)
            request_units += values[0]
            response_units += values[1]
            total_units += values[2]
    if not current_attempt_seen:
        raise ValueError("current provider prompt reservation was not found")
    provider_attempt_cap = (
        len(GENERATION_LANE_IDS) * STAIR_GENERATION_PROVIDER_ATTEMPTS_PER_LANE
    )
    if attempt_count > provider_attempt_cap:
        raise RuntimeError("Stair provider attempt cap is exhausted")
    request_units += STAIR_GENERATION_PROVIDER_INPUT_UNITS_PER_ATTEMPT
    response_units += STAIR_GENERATION_PROVIDER_OUTPUT_UNITS_PER_ATTEMPT
    total_units += (
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
    snapshots: dict[str, Any] = {}
    generation_id = f"{foundation.config.session_id}-generation-001"
    for lane_id in GENERATION_LANE_IDS:
        round_id = f"{generation_id}-{lane_id}"
        round_dir = foundation.config.output_dir / "authoring" / generation_id / lane_id
        if not round_dir.exists():
            continue
        if round_dir.is_symlink() or not round_dir.is_dir():
            raise ValueError("authoring round directory is invalid")
        snapshot = RoundAuthoringJournal(
            round_dir,
            round_id=round_id,
        ).snapshot()
        snapshots[lane_id] = snapshot
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
    ledger_lanes: set[str] = set()
    for record in ledger:
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
            ledger_lanes.add(lane_id)
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
            snapshot = snapshots.get(lane_id)
            if (
                snapshot is None
                or list(snapshot.attempts) != evidence_payload["attempts"]
            ):
                raise ValueError(
                    "ledger authoring evidence differs from its live journal"
                )
    if ledger and ledger_lanes != set(snapshots):
        raise ValueError("ledger and authoring journal lane sets differ")
    return {**totals, "conservative_charges": charges}


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
    cost_contract: StairGenerationSessionCostContract,
) -> int:
    if not isinstance(cost_contract, StairGenerationSessionCostContract):
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
) -> dict[str, Any]:
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
    summaries, completed_bound_job_count = _resource_summary_evidence(foundation)
    attempt_chain = _attempt_chain_closure_evidence(foundation)
    if attempt_chain["open_reservation_count"] != 0:
        raise ValueError("generation resource closure has open job attempts")
    reacquired = _reacquire_generation_gpu_leases(
        foundation,
        genesis=genesis,
        invocation_segment_id=invocation_segment_id,
    )
    return {
        "schema_version": "true_rl_agent.stair.generation_resource_closure.v1",
        "startup_compute_idle": expected_startup,
        "final_compute_idle": final_idle,
        "completed_bound_job_count": completed_bound_job_count,
        "summaries": summaries,
        "attempt_chain": attempt_chain,
        "lease_reacquire": reacquired,
        "acceptance_authority": False,
    }


def _resource_summary_evidence(
    foundation: StairGenerationSessionFoundation,
) -> tuple[list[dict[str, Any]], int]:
    completed_jobs = _completed_bound_job_attempts(foundation)
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
        if not isinstance(job_id, str) or job_id not in completed_jobs:
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
) -> dict[str, str]:
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


def _attempt_chain_closure_evidence(
    foundation: StairGenerationSessionFoundation,
) -> dict[str, Any]:
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
            reservation_count += 1
            result_path = path.with_name(path.name.replace(".reservation.", ".result."))
            if not result_path.exists():
                open_reservation_count += 1
                continue
            journal.load_result(reservation.payload["attempt_index"])
            closed_result_count += 1
    return {
        "schema_version": "true_rl_agent.stair.generation_attempt_chain_closure.v1",
        "source": "validated-job-attempt-journal",
        "reservation_count": reservation_count,
        "closed_result_count": closed_result_count,
        "open_reservation_count": open_reservation_count,
    }


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


def _provider_policy() -> dict[str, Any]:
    return {
        "schema_version": "true_rl_agent.stair.generation_provider_policy.v1",
        "base_url": STAIR_GENERATION_PROVIDER_BASE_URL,
        "endpoint": STAIR_GENERATION_PROVIDER_ENDPOINT,
        "model": STAIR_GENERATION_PROVIDER_MODEL,
        "reasoning_mode": "enabled",
        "reasoning_effort": "max",
        "temperature_present": False,
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
    current_usage, _, open_reservations = _collect_actual_cost_usage(
        foundation,
        genesis=genesis,
        ledger=ledger,
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


def _reject_parent_gpu_remap() -> None:
    present = [key for key in _PARENT_GPU_REMAP_KEYS if key in os.environ]
    if present:
        raise ValueError(
            "generation parent environment cannot carry GPU visibility remaps: "
            + ", ".join(present)
        )


def _preflight_provider_credential(env_file: Path) -> None:
    """Validate the fixed secret boundary without network I/O or disclosure."""

    value = load_env_file(env_file).get(
        STAIR_GENERATION_PROVIDER_API_KEY_ENV
    ) or os.environ.get(STAIR_GENERATION_PROVIDER_API_KEY_ENV)
    if not isinstance(value, str) or not value:
        raise ValueError("provider credential is missing from its fixed boundary")
    try:
        value.encode("latin-1")
    except UnicodeEncodeError as exc:
        raise ValueError("provider credential is not HTTP-header safe") from exc
    if any(
        character in "\r\n" or ord(character) < 0x20 or 0x7F <= ord(character) <= 0x9F
        for character in value
    ):
        raise ValueError("provider credential is not HTTP-header safe")


def _verify_source_checkout(
    project_root: Path,
    *,
    expected_commit: str,
) -> str:
    root = project_root.resolve()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()
    if commit != expected_commit:
        raise ValueError("source checkout does not match expected commit")
    tracked_status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout
    if tracked_status.strip():
        raise ValueError("source checkout has tracked modifications")
    return commit


__all__ = [
    "STAIR_GENERATION_REPORT_FILENAME",
    "STAIR_GENERATION_REPORT_SCHEMA_VERSION",
    "StairGenerationPilotConfig",
    "StairGenerationSessionFoundation",
    "discover_two_gpu_inventory",
    "run_stair_generation_pilot",
]
