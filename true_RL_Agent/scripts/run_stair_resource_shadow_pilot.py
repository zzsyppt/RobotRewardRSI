#!/usr/bin/env python3
"""Run the fixed, non-scientific Stair P6-A two-GPU resource shadow."""

from __future__ import annotations

import argparse
import concurrent.futures
import errno
import json
import os
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from true_rl_agent.agent_core.generation_contract import (  # noqa: E402
    GenerationLaneAssignment,
)
from true_rl_agent.agent_core.session_genesis import (  # noqa: E402
    SESSION_GENESIS_SCHEMA_VERSION,
    ReferenceProgramGenesisAnchor,
    SessionGenesisArtifact,
)
from true_rl_agent.backends.isaac_lab_execution_environment import (  # noqa: E402
    verify_current_isaac_lab_execution_environment,
)
from true_rl_agent.evaluation.binding import (  # noqa: E402
    canonical_json_hash,
    require_sha256_digest,
)
from true_rl_agent.fs_contract import (  # noqa: E402
    create_once_bytes,
    read_bounded_regular_file,
)
from true_rl_agent.json_contract import (  # noqa: E402
    canonical_json_bytes,
    strict_json_object_loads,
)
from true_rl_agent.runner.gpu_lease import (  # noqa: E402
    GpuLeaseHolder,
    GpuLeaseManager,
)
from true_rl_agent.runner.process_handle import (  # noqa: E402
    OwnedProcessResult,
    run_owned_process,
)
from true_rl_agent.runner.resource_shadow import (  # noqa: E402
    RESOURCE_SHADOW_AUTHORITY_COUNTER_KEYS,
    RESOURCE_SHADOW_PLAN_FILENAME,
    ResourceShadowJob,
    ResourceShadowLaneMeasurement,
    ResourceShadowPlan,
    create_resource_shadow_plan,
    discover_nvidia_gpu_inventory,
    sample_resource_snapshot,
    summarize_resource_shadow,
)
from true_rl_agent.tasks.stair_climbing.config import (  # noqa: E402
    STAIR_PROTOCOL_V2_RELATIVE_PATH,
    STAIR_TASK_ID,
    StairClimbingTaskConfig,
)
from true_rl_agent.tasks.stair_climbing.resource_shadow import (  # noqa: E402
    load_stair_resource_shadow_active_marker,
    load_stair_resource_shadow_summary,
    resolve_stair_resource_shadow_fixture,
    stair_resource_shadow_workload_hash,
)


PILOT_REPORT_SCHEMA_VERSION = "true_rl_agent.p6a_resource_shadow_report.v2"
PILOT_PREFLIGHT_SCHEMA_VERSION = "true_rl_agent.p6a_resource_shadow_preflight.v1"
PILOT_CASE_SCHEMA_VERSION = "true_rl_agent.p6a_resource_shadow_case.v3"
PILOT_FAILURE_SCHEMA_VERSION = "true_rl_agent.p6a_resource_shadow_failure.v1"
JOB_TIMEOUT_SECONDS = 900
TERMINATION_GRACE_SECONDS = 5.0
MIN_FREE_DISK_BYTES = 30 * 1024**3
FORECAST_PEAK_OUTPUT_BYTES = 8 * 1024**3
MIN_AVAILABLE_RAM_KIB = 64 * 1024**2
MIN_ACTIVE_SAMPLE_COUNT = 45
FAULT_REQUIRED_MATCHED_SAMPLES = 3
FAULT_ARM_TIMEOUT_SECONDS = 300.0
SHARED_GPU_LEASE_ROOT = Path("/tmp/true_rl_agent_gpu_leases")


@dataclass(frozen=True)
class _FrozenContext:
    output_dir: Path
    isaac_python: Path
    genesis_path: Path
    genesis: SessionGenesisArtifact
    config: StairClimbingTaskConfig
    fixture_hash: str
    workload_hash: str
    source_commit: str
    environment_manifest_hash: str
    expected_initial_policy_state_hash: str
    preflight: Mapping[str, Any]


@dataclass(frozen=True)
class _OwnedJobOutcome:
    job: ResourceShadowJob
    child_output_dir: Path
    marker_path: Path
    checkpoint_path: Path
    launch_start_ns: int
    launch_end_ns: int
    full_job_wall_seconds: float
    result: OwnedProcessResult | None
    exception_type: str | None
    exception_message: str | None


@dataclass(frozen=True)
class _CaseOutcome:
    case_id: str
    jobs: tuple[ResourceShadowJob, ...]
    job_outcomes: tuple[_OwnedJobOutcome, ...]
    markers: Mapping[str, Mapping[str, Any] | None]
    summaries: Mapping[str, Mapping[str, Any] | None]
    snapshots: tuple[Mapping[str, Any], ...]
    measurements: Mapping[str, ResourceShadowLaneMeasurement]
    valid_jobs: Mapping[str, bool]
    live_group_bindings: Mapping[str, bool]
    unknown_compute_pids: tuple[int, ...]
    fault_injection: Mapping[str, Any] | None
    post_case_compute_apps: tuple[Mapping[str, Any], ...]
    receipt_path: Path


class _Sampler:
    def __init__(self, *, plan: ResourceShadowPlan, output_dir: Path) -> None:
        self._plan = plan
        self._output_dir = output_dir
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._snapshots: list[dict[str, Any]] = []
        self._errors: list[str] = []
        self._thread = threading.Thread(
            target=self._run,
            name="p6a-resource-sampler",
            daemon=False,
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=45.0)
        if self._thread.is_alive():
            raise RuntimeError("resource sampler did not stop")

    def snapshot(self) -> tuple[dict[str, Any], ...]:
        with self._lock:
            return tuple(dict(item) for item in self._snapshots)

    def errors(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._errors)

    def _run(self) -> None:
        next_sample = time.monotonic()
        while not self._stop.is_set():
            try:
                sample = sample_resource_snapshot(
                    inventory=self._plan.gpu_inventory,
                    output_dir=self._output_dir,
                )
            except Exception as exc:  # Operational sampling must not orphan jobs.
                with self._lock:
                    self._errors.append(type(exc).__name__)
                return
            with self._lock:
                self._snapshots.append(sample)
            next_sample += self._plan.sample_interval_seconds
            self._stop.wait(max(0.0, next_sample - time.monotonic()))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--session-genesis", required=True)
    parser.add_argument("--isaac-python", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-gpu0-uuid", required=True)
    parser.add_argument("--expected-gpu1-uuid", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    prepared_output_dir: Path | None = None
    try:
        context, plan = _prepare(args)
        prepared_output_dir = context.output_dir
        report = _run_pilot(context=context, plan=plan)
        _create_once_json(context.output_dir / "resource_shadow_report.json", report)
        print(json.dumps({"go": report["go"], "report": str(context.output_dir / "resource_shadow_report.json")}, sort_keys=True))
        return 0 if report["go"] is True else 1
    except KeyboardInterrupt:
        if prepared_output_dir is not None:
            _publish_failure_if_possible(prepared_output_dir, "KeyboardInterrupt")
        return 130
    except Exception as exc:
        if prepared_output_dir is not None:
            _publish_failure_if_possible(prepared_output_dir, type(exc).__name__)
        print(f"P6-A resource shadow failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def _prepare(args: argparse.Namespace) -> tuple[_FrozenContext, ResourceShadowPlan]:
    if not sys.platform.startswith("linux") or os.name != "posix":
        raise OSError("P6-A resource shadow requires Linux")
    output_dir = _fresh_absolute_path(args.output_dir, "output_dir")
    genesis_path = _existing_absolute_file(args.session_genesis, "session_genesis")
    isaac_python = _existing_absolute_executable(args.isaac_python, "isaac_python")
    if Path(sys.executable).resolve() != isaac_python:
        raise ValueError("coordinator and child must use the same absolute Isaac Python")
    for variable in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"):
        if os.environ.get(variable, "").strip():
            raise ValueError(f"{variable} must be unset for physical ordinal validation")

    expected_commit = _lower_hex_commit(args.expected_source_commit)
    source_commit = _git("rev-parse", "HEAD")
    if source_commit != expected_commit:
        raise ValueError("current source commit differs from the pre-registered commit")
    if _git_status_dirty():
        raise ValueError("tracked worktree must be clean before P6-A")
    upstream_commit = _git("rev-parse", "@{upstream}")
    if upstream_commit != source_commit:
        raise ValueError("local and upstream source commits differ")

    genesis = _load_genesis(genesis_path)
    if genesis.schema_version != SESSION_GENESIS_SCHEMA_VERSION:
        raise ValueError("P6-A requires the frozen width-one Session Genesis v3")
    if genesis.task_id != STAIR_TASK_ID or genesis.backend_id != "isaac_lab_rsl_rl":
        raise ValueError("P6-A Session Genesis task/backend mismatch")
    if not isinstance(genesis.anchor, ReferenceProgramGenesisAnchor):
        raise ValueError("P6-A requires a reference-program Genesis anchor")
    if genesis.backend_execution_environment is None:
        raise ValueError("P6-A Genesis has no backend execution environment")
    environment_hash = require_sha256_digest(
        genesis.backend_execution_environment.get("manifest_hash"),
        "P6-A execution environment manifest hash",
    )
    environment_verification = verify_current_isaac_lab_execution_environment(
        environment_hash
    )
    expected_initial_hash = require_sha256_digest(
        genesis.anchor.initial_policy_state_hash,
        "P6-A expected initial policy state hash",
    )
    config = StairClimbingTaskConfig.defaults(
        PROJECT_ROOT / STAIR_PROTOCOL_V2_RELATIVE_PATH
    )
    if config.task_protocol_hash != genesis.evaluation_binding.task_protocol_hash:
        raise ValueError("current Stair task protocol differs from Session Genesis")
    fixture = resolve_stair_resource_shadow_fixture(
        config,
        expected_initial_policy_state_hash=expected_initial_hash,
        project_root=PROJECT_ROOT,
    )
    if (
        fixture.contract["program_source_checksum"]
        != genesis.anchor.reference_manifest.source_checksum
    ):
        raise ValueError("current official reference source differs from Session Genesis")
    if (
        fixture.contract["agent_configuration_hash"]
        != genesis.anchor.training_binding.agent_configuration_hash
    ):
        raise ValueError("current training agent configuration differs from Session Genesis")

    inventory = discover_nvidia_gpu_inventory()
    if tuple(item.ordinal for item in inventory) != (0, 1):
        raise ValueError("P6-A requires a host with exactly physical GPU ordinals 0 and 1")
    by_ordinal = {item.ordinal: item for item in inventory}
    selected_inventory = (by_ordinal[0], by_ordinal[1])
    expected_uuids = (args.expected_gpu0_uuid, args.expected_gpu1_uuid)
    if tuple(item.uuid for item in selected_inventory) != expected_uuids:
        raise ValueError("physical GPU UUID mapping differs from the pre-registered mapping")
    initial_snapshot = sample_resource_snapshot(inventory=selected_inventory)
    if initial_snapshot["compute_apps"]:
        raise RuntimeError("planned GPUs already have compute processes")
    memory = _memory_fields(initial_snapshot)
    if memory["MemAvailable"] < MIN_AVAILABLE_RAM_KIB:
        raise RuntimeError("P6-A host has less than the pre-registered available RAM")
    disk = shutil.disk_usage(output_dir.parent)
    required_free_disk_bytes = max(
        MIN_FREE_DISK_BYTES,
        3 * FORECAST_PEAK_OUTPUT_BYTES,
    )
    if disk.free < required_free_disk_bytes:
        raise RuntimeError("P6-A output filesystem has insufficient free space")

    lane_intent_hashes = tuple(
        canonical_json_hash(
            {
                "schema_version": "true_rl_agent.p6a_resource_lane_intent.v1",
                "purpose": "resource_shadow_only",
                "lane_id": lane_id,
                "acceptance_authority": False,
            }
        )
        for lane_id in ("lane-a", "lane-b")
    )
    lanes = tuple(
        GenerationLaneAssignment(
            lane_id=lane_id,
            execution_device=inventory_item.execution_device,
            resource_binding_hash=inventory_item.identity_hash,
            lane_intent_hash=lane_intent_hash,
        )
        for lane_id, inventory_item, lane_intent_hash in zip(
            ("lane-a", "lane-b"),
            selected_inventory,
            lane_intent_hashes,
            strict=True,
        )
    )
    pilot_id = "p6a-resource-shadow-" + source_commit[:12]
    preflight = {
        "schema_version": PILOT_PREFLIGHT_SCHEMA_VERSION,
        "pilot_id": pilot_id,
        "source_commit": source_commit,
        "upstream_commit": upstream_commit,
        "session_genesis_path": str(genesis_path),
        "session_genesis_hash": genesis.session_genesis_hash,
        "execution_environment_manifest_hash": environment_hash,
        "execution_environment_verification": dict(environment_verification),
        "initial_policy_state_hash": expected_initial_hash,
        "gpu_inventory": [item.to_dict() for item in selected_inventory],
        "initial_snapshot": initial_snapshot,
        "free_disk_bytes": disk.free,
        "forecast_peak_output_bytes": FORECAST_PEAK_OUTPUT_BYTES,
        "minimum_free_disk_bytes": required_free_disk_bytes,
        "minimum_available_ram_kib": MIN_AVAILABLE_RAM_KIB,
        "minimum_active_sample_count": MIN_ACTIVE_SAMPLE_COUNT,
        "job_timeout_seconds": JOB_TIMEOUT_SECONDS,
        "fault_required_matched_samples": FAULT_REQUIRED_MATCHED_SAMPLES,
        "shared_gpu_lease_root": str(SHARED_GPU_LEASE_ROOT),
        "provider_requests": 0,
        "evaluation_queries": 0,
        "primary_ledger_writes": 0,
        "acceptance_authority": False,
        "ledger_authority": False,
    }

    output_dir.mkdir(parents=False, exist_ok=False)
    _create_once_json(output_dir / "preflight.json", preflight)
    plan = create_resource_shadow_plan(
        output_dir / RESOURCE_SHADOW_PLAN_FILENAME,
        pilot_id=pilot_id,
        source_commit=source_commit,
        session_genesis_hash=require_sha256_digest(
            genesis.session_genesis_hash,
            "P6-A session Genesis hash",
        ),
        execution_environment_manifest_hash=environment_hash,
        fixture_hash=fixture.fixture_hash,
        workload_hash=stair_resource_shadow_workload_hash(),
        lanes=lanes,  # type: ignore[arg-type]
        gpu_inventory=selected_inventory,
    )
    return (
        _FrozenContext(
            output_dir=output_dir,
            isaac_python=isaac_python,
            genesis_path=genesis_path,
            genesis=genesis,
            config=config,
            fixture_hash=fixture.fixture_hash,
            workload_hash=stair_resource_shadow_workload_hash(),
            source_commit=source_commit,
            environment_manifest_hash=environment_hash,
            expected_initial_policy_state_hash=expected_initial_hash,
            preflight=preflight,
        ),
        plan,
    )


def _run_pilot(*, context: _FrozenContext, plan: ResourceShadowPlan) -> dict[str, Any]:
    pilot_started = time.perf_counter()
    lease_manager = GpuLeaseManager(
        lease_root=SHARED_GPU_LEASE_ROOT,
        devices={item.execution_device: item.identity for item in plan.gpu_inventory},
    )
    grouped: dict[str, list[ResourceShadowJob]] = {}
    for job in plan.jobs:
        grouped.setdefault(job.case_id, []).append(job)
    cases: list[_CaseOutcome] = []
    for case_jobs in grouped.values():
        case = _run_case(
            context=context,
            plan=plan,
            lease_manager=lease_manager,
            jobs=tuple(case_jobs),
        )
        cases.append(case)
        if case.jobs[0].mode != "interruption":
            _require_case_execution_valid(case=case, plan=plan)

    single_a: list[ResourceShadowLaneMeasurement] = []
    single_b: list[ResourceShadowLaneMeasurement] = []
    paired_by_index: dict[int, dict[str, ResourceShadowLaneMeasurement]] = {}
    warmup_valid = True
    all_active_sample_counts_valid = True
    all_sampler_errors_absent = True
    all_unknown_pids_absent = True
    fault_case: _CaseOutcome | None = None
    for case in cases:
        all_sampler_errors_absent = all_sampler_errors_absent and not _case_sampler_errors(
            case.receipt_path
        )
        all_unknown_pids_absent = all_unknown_pids_absent and not case.unknown_compute_pids
        if case.jobs[0].mode == "interruption":
            fault_case = case
            continue
        for job in case.jobs:
            measurement = case.measurements[job.job_id]
            if job.warmup:
                warmup_valid = warmup_valid and case.valid_jobs[job.job_id]
                continue
            active_count = _active_sample_count(
                snapshots=case.snapshots,
                marker=case.markers[job.job_id],
                summary=case.summaries[job.job_id],
                expected_uuid=plan.inventory_for_execution_device(
                    job.execution_device
                ).uuid,
            )
            all_active_sample_counts_valid = (
                all_active_sample_counts_valid
                and active_count >= MIN_ACTIVE_SAMPLE_COUNT
            )
            if job.mode == "single" and job.lane_id == "lane-a":
                single_a.append(measurement)
            elif job.mode == "single" and job.lane_id == "lane-b":
                single_b.append(measurement)
            elif job.mode == "paired":
                paired_by_index.setdefault(job.repeat_index, {})[job.lane_id] = measurement
    if fault_case is None or fault_case.fault_injection is None:
        raise RuntimeError("P6-A interruption probe did not close")
    fault_probe = _fault_probe(
        case=fault_case,
        lease_manager=lease_manager,
        plan=plan,
    )
    paired = [
        (paired_by_index[index]["lane-a"], paired_by_index[index]["lane-b"])
        for index in sorted(paired_by_index)
    ]
    core_summary = summarize_resource_shadow(
        plan=plan,
        single_lane_a=single_a,
        single_lane_b=single_b,
        paired=paired,
        fault_probe=fault_probe,
        authority_counters={
            key: 0 for key in RESOURCE_SHADOW_AUTHORITY_COUNTER_KEYS
        },
    )
    all_snapshots = tuple(
        snapshot for case in cases for snapshot in case.snapshots
    )
    resource_observations = _resource_observations(
        snapshots=all_snapshots,
        output_dir=context.output_dir,
        initial_snapshot=context.preflight["initial_snapshot"],
    )
    additional_gates = {
        "all_warmups_valid": warmup_valid,
        "minimum_active_sample_count": all_active_sample_counts_valid,
        "sampler_errors_absent": all_sampler_errors_absent,
        "unknown_compute_pids_absent": all_unknown_pids_absent,
        "all_case_process_group_bindings_valid": all(
            all(case.live_group_bindings.values()) for case in cases
        ),
        "all_case_cross_device_pids_absent": all(
            not measurement.cross_device_pid
            for case in cases
            for measurement in case.measurements.values()
        ),
        "all_case_orphaned_processes_absent": all(
            not measurement.orphaned_process
            for case in cases
            for measurement in case.measurements.values()
        ),
        "all_case_oom_absent": all(
            not measurement.oom
            for case in cases
            for measurement in case.measurements.values()
        ),
        "all_case_post_compute_apps_cleared": all(
            not case.post_case_compute_apps for case in cases
        ),
        "global_swap_growth_within_limit": (
            resource_observations["max_swap_growth_kib"] / 1024.0
            <= plan.thresholds["swap_delta_mib_max"]
        ),
        "peak_output_within_forecast": (
            resource_observations["peak_output_size_bytes"]
            <= context.preflight["forecast_peak_output_bytes"]
        ),
    }
    report = {
        "schema_version": PILOT_REPORT_SCHEMA_VERSION,
        "pilot_id": plan.pilot_id,
        "plan_hash": plan.plan_hash,
        "source_commit": context.source_commit,
        "session_genesis_hash": plan.session_genesis_hash,
        "preflight_path": str(context.output_dir / "preflight.json"),
        "preflight_sha256": _sha256_path(context.output_dir / "preflight.json"),
        "core_summary": core_summary,
        "core_summary_hash": canonical_json_hash(core_summary),
        "additional_gates": additional_gates,
        "resource_observations": resource_observations,
        "timing_observations": _timing_observations(cases),
        "case_receipts": [
            {
                "case_id": case.case_id,
                "path": str(case.receipt_path),
                "sha256": _sha256_path(case.receipt_path),
            }
            for case in cases
        ],
        "go": core_summary["go"] is True and all(additional_gates.values()),
        "operational_only": True,
        "acceptance_authority": False,
        "ledger_authority": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pilot_wall_seconds": time.perf_counter() - pilot_started,
    }
    return report


def _require_case_execution_valid(
    *, case: _CaseOutcome, plan: ResourceShadowPlan
) -> None:
    if _case_sampler_errors(case.receipt_path):
        raise RuntimeError(f"resource sampling failed in {case.case_id}")
    if case.unknown_compute_pids or case.post_case_compute_apps:
        raise RuntimeError(f"unexpected compute process observed in {case.case_id}")
    for job in case.jobs:
        measurement = case.measurements[job.job_id]
        active_count = _active_sample_count(
            snapshots=case.snapshots,
            marker=case.markers[job.job_id],
            summary=case.summaries[job.job_id],
            expected_uuid=plan.inventory_for_execution_device(
                job.execution_device
            ).uuid,
        )
        if (
            not case.valid_jobs[job.job_id]
            or active_count < MIN_ACTIVE_SAMPLE_COUNT
            or measurement.active_seconds
            < plan.thresholds["minimum_measured_active_seconds"]
        ):
            raise RuntimeError(f"resource shadow job was invalid: {job.job_id}")


def _run_case(
    *,
    context: _FrozenContext,
    plan: ResourceShadowPlan,
    lease_manager: GpuLeaseManager,
    jobs: tuple[ResourceShadowJob, ...],
) -> _CaseOutcome:
    if not jobs or len({job.case_id for job in jobs}) != 1:
        raise ValueError("resource shadow case jobs are invalid")
    case_id = jobs[0].case_id
    case_dir = context.output_dir / "cases" / case_id
    case_dir.mkdir(parents=True, exist_ok=False)
    plan_path = context.output_dir / RESOURCE_SHADOW_PLAN_FILENAME
    markers: dict[str, Mapping[str, Any] | None] = {job.job_id: None for job in jobs}
    live_group_bindings: dict[str, bool] = {job.job_id: False for job in jobs}
    sampler: _Sampler | None = None
    fault_injection: dict[str, Any] | None = None
    snapshots: tuple[Mapping[str, Any], ...] = ()
    outcomes: tuple[_OwnedJobOutcome, ...] = ()

    with ExitStack() as lease_stack:
        acquired: dict[str, Any] = {}
        ordered_jobs = sorted(
            jobs,
            key=lambda item: plan.inventory_for_execution_device(
                item.execution_device
            ).physical_resource_hash,
        )
        for job in ordered_jobs:
            lease = lease_stack.enter_context(
                lease_manager.acquire(
                    job.execution_device,
                    holder=GpuLeaseHolder(
                        session_genesis_hash=plan.session_genesis_hash,
                        generation_id=plan.pilot_id,
                        lane_id=job.lane_id,
                        invocation_segment_id=case_id,
                        logical_job_id=job.job_id,
                    ),
                )
            )
            if (
                lease.resource_binding_hash
                != plan.lane_assignments[job.lane_id].resource_binding_hash
            ):
                raise RuntimeError("GPU lease binding differs from the frozen plan")
            acquired[job.job_id] = lease
            _create_once_json(case_dir / "leases" / f"{job.job_id}.json", lease.receipt)

        sampler = _Sampler(plan=plan, output_dir=context.output_dir)
        sampler.start()
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs))
        futures = {
            job.job_id: executor.submit(
                _run_owned_job,
                context=context,
                plan=plan,
                plan_path=plan_path,
                case_dir=case_dir,
                job=job,
            )
            for job in jobs
        }
        try:
            monitor_started = time.monotonic()
            while not all(future.done() for future in futures.values()):
                for job in jobs:
                    marker = _try_load_marker(
                        context=context,
                        plan=plan,
                        case_dir=case_dir,
                        job=job,
                    )
                    if marker is None:
                        continue
                    markers[job.job_id] = marker
                    try:
                        live_group_bindings[job.job_id] = (
                            os.getpgid(int(marker["process_id"]))
                            == int(marker["process_group_id"])
                            == int(marker["process_id"])
                        )
                    except ProcessLookupError:
                        pass
                if jobs[0].mode == "interruption" and fault_injection is None:
                    fault_injection = _maybe_inject_fault(
                        jobs=jobs,
                        markers=markers,
                        snapshots=sampler.snapshot(),
                        plan=plan,
                    )
                    if (
                        fault_injection is None
                        and time.monotonic() - monitor_started
                        > FAULT_ARM_TIMEOUT_SECONDS
                    ):
                        fault_injection = {
                            "armed": False,
                            "signal_sent": False,
                            "reason": "fault_arm_timeout",
                        }
                time.sleep(0.2)
            outcomes = tuple(futures[job.job_id].result() for job in jobs)
        except BaseException:
            _terminate_known_groups(markers.values())
            raise
        finally:
            executor.shutdown(wait=True, cancel_futures=False)
            sampler.stop()
            snapshots = sampler.snapshot()

    post_case = sample_resource_snapshot(
        inventory=plan.gpu_inventory,
        output_dir=context.output_dir,
    )
    post_apps = tuple(post_case["compute_apps"])
    snapshots = (*snapshots, post_case)
    if sampler is None:
        raise RuntimeError("resource sampler was not constructed")
    sampler_errors = sampler.errors()

    outcome_by_job = {outcome.job.job_id: outcome for outcome in outcomes}
    summaries: dict[str, Mapping[str, Any] | None] = {}
    valid_jobs: dict[str, bool] = {}
    measurements: dict[str, ResourceShadowLaneMeasurement] = {}
    for job in jobs:
        outcome = outcome_by_job[job.job_id]
        if markers[job.job_id] is None:
            markers[job.job_id] = _try_load_marker(
                context=context,
                plan=plan,
                case_dir=case_dir,
                job=job,
            )
        summary = _try_load_summary(
            context=context,
            plan=plan,
            outcome=outcome,
            expected_status=None,
        )
        summaries[job.job_id] = summary
        marker = markers[job.job_id]
        cross_device = _cross_device_pid(
            snapshots=snapshots,
            marker=marker,
            expected_uuid=plan.inventory_for_execution_device(
                job.execution_device
            ).uuid,
        )
        group_alive = (
            _process_group_alive(int(marker["process_group_id"]))
            if marker is not None
            else False
        )
        swap_delta = _swap_delta_mib(snapshots)
        start_ns = (
            int(summary["active_start_monotonic_ns"])
            if summary is not None
            else int(marker["active_start_monotonic_ns"])
            if marker is not None
            else outcome.launch_start_ns
        )
        end_ns = (
            int(summary["active_end_monotonic_ns"])
            if summary is not None
            else max(start_ns + 1, outcome.launch_end_ns)
        )
        normal_expected = job.mode != "interruption" or job.lane_id == "lane-b"
        result_ok = (
            outcome.result is not None
            and outcome.result.returncode == 0
            and marker is not None
            and outcome.result.pid == marker["process_id"]
            and summary is not None
            and summary["status"] == "pass"
            and summary["process_id"] == marker["process_id"]
            and summary["process_group_id"] == marker["process_group_id"]
            and summary["active_start_monotonic_ns"]
            == marker["active_start_monotonic_ns"]
        )
        valid = (
            result_ok
            and not sampler_errors
            and not cross_device
            and not group_alive
            and not post_apps
            and live_group_bindings[job.job_id]
        )
        valid_jobs[job.job_id] = valid if normal_expected else not result_ok
        measurements[job.job_id] = ResourceShadowLaneMeasurement(
            job_id=job.job_id,
            case_id=job.case_id,
            repeat_index=job.repeat_index,
            lane_id=job.lane_id,
            active_start_ns=start_ns,
            active_end_ns=end_ns,
            full_job_wall_seconds=outcome.full_job_wall_seconds,
            valid=valid,
            oom=_outcome_has_oom(outcome),
            swap_delta_mib=swap_delta,
            cross_device_pid=cross_device,
            orphaned_process=group_alive,
        )

    known_pids = {
        int(marker["process_id"])
        for marker in markers.values()
        if marker is not None
    }
    first_active_ns = min(
        (
            int(marker["active_start_monotonic_ns"])
            for marker in markers.values()
            if marker is not None
        ),
        default=sys.maxsize,
    )
    unknown_pids = sorted(
        {
            int(app["pid"])
            for snapshot in snapshots
            if int(snapshot["monotonic_ns"]) >= first_active_ns
            for app in snapshot["compute_apps"]
            if int(app["pid"]) not in known_pids
        }
    )
    snapshots_path = case_dir / "resource_snapshots.json"
    _create_once_json(
        snapshots_path,
        {
            "schema_version": "true_rl_agent.p6a_resource_snapshots.v1",
            "plan_hash": plan.plan_hash,
            "case_id": case_id,
            "snapshots": list(snapshots),
            "acceptance_authority": False,
            "ledger_authority": False,
        },
    )
    receipt = {
        "schema_version": PILOT_CASE_SCHEMA_VERSION,
        "plan_hash": plan.plan_hash,
        "case_id": case_id,
        "jobs": [job.to_dict() for job in jobs],
        "job_results": [
            _job_outcome_dict(
                outcome=outcome_by_job[job.job_id],
                marker=markers[job.job_id],
                summary=summaries[job.job_id],
                measurement=measurements[job.job_id],
                valid=valid_jobs[job.job_id],
                active_sample_count=_active_sample_count(
                    snapshots=snapshots,
                    marker=markers[job.job_id],
                    summary=summaries[job.job_id],
                    expected_uuid=plan.inventory_for_execution_device(
                        job.execution_device
                    ).uuid,
                ),
            )
            for job in jobs
        ],
        "resource_snapshots_path": str(snapshots_path),
        "resource_snapshots_sha256": _sha256_path(snapshots_path),
        "resource_snapshot_count": len(snapshots),
        "sampler_errors": list(sampler_errors),
        "unknown_compute_pids": unknown_pids,
        "post_case_compute_apps": list(post_apps),
        "fault_injection": fault_injection,
        "operational_only": True,
        "acceptance_authority": False,
        "ledger_authority": False,
    }
    receipt_path = case_dir / "case_receipt.json"
    _create_once_json(receipt_path, receipt)
    return _CaseOutcome(
        case_id=case_id,
        jobs=jobs,
        job_outcomes=outcomes,
        markers=markers,
        summaries=summaries,
        snapshots=snapshots,
        measurements=measurements,
        valid_jobs=valid_jobs,
        live_group_bindings=live_group_bindings,
        unknown_compute_pids=tuple(unknown_pids),
        fault_injection=fault_injection,
        post_case_compute_apps=post_apps,
        receipt_path=receipt_path,
    )


def _run_owned_job(
    *,
    context: _FrozenContext,
    plan: ResourceShadowPlan,
    plan_path: Path,
    case_dir: Path,
    job: ResourceShadowJob,
) -> _OwnedJobOutcome:
    inventory = plan.inventory_for_execution_device(job.execution_device)
    child_output = case_dir / "jobs" / job.job_id
    marker_path = child_output / "active.json"
    checkpoint_path = child_output / "checkpoint.pt"
    argv = (
        str(context.isaac_python),
        str(PROJECT_ROOT / "scripts" / "isaaclab_stair_resource_shadow_train.py"),
        "--plan",
        str(plan_path),
        "--plan-hash",
        plan.plan_hash,
        "--session-genesis-hash",
        plan.session_genesis_hash,
        "--execution-environment-manifest-hash",
        plan.execution_environment_manifest_hash,
        "--fixture-hash",
        plan.fixture_hash,
        "--workload-hash",
        plan.workload_hash,
        "--resource-plan-hash",
        plan.resource_plan_hash,
        "--expected-initial-policy-state-hash",
        context.expected_initial_policy_state_hash,
        "--job-id",
        job.job_id,
        "--case-id",
        job.case_id,
        "--lane-id",
        job.lane_id,
        "--physical-execution-device",
        job.execution_device,
        "--output-dir",
        str(child_output),
        "--active-marker",
        str(marker_path),
        "--checkpoint-path",
        str(checkpoint_path),
        "--device",
        job.process_execution_device,
        "--headless",
    )
    start_ns = time.monotonic_ns()
    started = time.perf_counter()
    result: OwnedProcessResult | None = None
    exception_type: str | None = None
    exception_message: str | None = None
    try:
        result = run_owned_process(
            argv,
            cwd=PROJECT_ROOT,
            timeout_s=JOB_TIMEOUT_SECONDS,
            log_dir=case_dir / "runner_process" / job.job_id,
            termination_grace_s=TERMINATION_GRACE_SECONDS,
            environment_overrides={
                "CUDA_VISIBLE_DEVICES": inventory.uuid,
                "NVIDIA_VISIBLE_DEVICES": None,
            },
        )
    except Exception as exc:
        exception_type = type(exc).__name__
        exception_message = str(exc)[:512]
    end_ns = time.monotonic_ns()
    return _OwnedJobOutcome(
        job=job,
        child_output_dir=child_output,
        marker_path=marker_path,
        checkpoint_path=checkpoint_path,
        launch_start_ns=start_ns,
        launch_end_ns=end_ns,
        full_job_wall_seconds=time.perf_counter() - started,
        result=result,
        exception_type=exception_type,
        exception_message=exception_message,
    )


def _maybe_inject_fault(
    *,
    jobs: Sequence[ResourceShadowJob],
    markers: Mapping[str, Mapping[str, Any] | None],
    snapshots: Sequence[Mapping[str, Any]],
    plan: ResourceShadowPlan,
) -> dict[str, Any] | None:
    if len(jobs) != 2 or any(markers[job.job_id] is None for job in jobs):
        return None
    matched_counts: dict[str, int] = {}
    for job in jobs:
        marker = markers[job.job_id]
        assert marker is not None
        matched_counts[job.job_id] = sum(
            1
            for snapshot in snapshots
            if int(snapshot["monotonic_ns"])
            >= int(marker["active_start_monotonic_ns"])
            and any(
                int(app["pid"]) == int(marker["process_id"])
                and app["gpu_uuid"]
                == plan.inventory_for_execution_device(job.execution_device).uuid
                for app in snapshot["compute_apps"]
            )
        )
    if any(count < FAULT_REQUIRED_MATCHED_SAMPLES for count in matched_counts.values()):
        return None
    lane_a = next(job for job in jobs if job.lane_id == "lane-a")
    marker = markers[lane_a.job_id]
    assert marker is not None
    pid = int(marker["process_id"])
    pgid = int(marker["process_group_id"])
    if pid != pgid or os.getpgid(pid) != pgid or pgid == os.getpgrp():
        raise RuntimeError("fault target is not the bound, dedicated lane-a process group")
    os.killpg(pgid, signal.SIGTERM)
    return {
        "armed": True,
        "signal_sent": True,
        "signal": "SIGTERM",
        "target_job_id": lane_a.job_id,
        "target_pid": pid,
        "target_process_group_id": pgid,
        "marker_hash": canonical_json_hash(marker),
        "post_active_matched_sample_counts_by_lane": {
            job.lane_id: matched_counts[job.job_id] for job in jobs
        },
        "signaled_at_monotonic_ns": time.monotonic_ns(),
    }


def _try_load_marker(
    *,
    context: _FrozenContext,
    plan: ResourceShadowPlan,
    case_dir: Path,
    job: ResourceShadowJob,
) -> Mapping[str, Any] | None:
    output_dir = case_dir / "jobs" / job.job_id
    marker_path = output_dir / "active.json"
    if not marker_path.exists():
        return None
    return load_stair_resource_shadow_active_marker(
        marker_path,
        output_dir=output_dir,
        expected_plan_hash=plan.plan_hash,
        expected_job_id=job.job_id,
        expected_case_id=job.case_id,
        expected_lane_id=job.lane_id,
        expected_physical_execution_device=job.execution_device,
        expected_process_execution_device=job.process_execution_device,
        expected_physical_gpu_uuid=plan.inventory_for_execution_device(
            job.execution_device
        ).uuid,
        expected_renderer_active_gpu_ordinal=plan.inventory_for_execution_device(
            job.execution_device
        ).ordinal,
    )


def _try_load_summary(
    *,
    context: _FrozenContext,
    plan: ResourceShadowPlan,
    outcome: _OwnedJobOutcome,
    expected_status: str | None,
) -> Mapping[str, Any] | None:
    path = outcome.child_output_dir / "summary.json"
    if not path.exists():
        return None
    return load_stair_resource_shadow_summary(
        path,
        output_dir=outcome.child_output_dir,
        expected_checkpoint_path=outcome.checkpoint_path,
        expected_plan_hash=plan.plan_hash,
        expected_session_genesis_hash=plan.session_genesis_hash,
        expected_execution_environment_manifest_hash=(
            plan.execution_environment_manifest_hash
        ),
        expected_fixture_hash=plan.fixture_hash,
        expected_workload_hash=plan.workload_hash,
        expected_resource_plan_hash=plan.resource_plan_hash,
        expected_initial_policy_state_hash=(
            context.expected_initial_policy_state_hash
        ),
        expected_job_id=outcome.job.job_id,
        expected_case_id=outcome.job.case_id,
        expected_lane_id=outcome.job.lane_id,
        expected_physical_execution_device=outcome.job.execution_device,
        expected_process_execution_device=outcome.job.process_execution_device,
        expected_physical_gpu_uuid=plan.inventory_for_execution_device(
            outcome.job.execution_device
        ).uuid,
        expected_renderer_active_gpu_ordinal=(
            plan.inventory_for_execution_device(outcome.job.execution_device).ordinal
        ),
        expected_status=expected_status,
    )


def _fault_probe(
    *,
    case: _CaseOutcome,
    lease_manager: GpuLeaseManager,
    plan: ResourceShadowPlan,
) -> dict[str, bool]:
    by_lane = {job.lane_id: job for job in case.jobs}
    lane_a = by_lane["lane-a"]
    lane_b = by_lane["lane-b"]
    outcome_a = next(item for item in case.job_outcomes if item.job == lane_a)
    injection = case.fault_injection or {}
    marker_a = case.markers[lane_a.job_id]
    signal_time = injection.get("signaled_at_monotonic_ns")
    matched_counts = injection.get("post_active_matched_sample_counts_by_lane")
    counts_bound = (
        isinstance(matched_counts, Mapping)
        and set(matched_counts) == {"lane-a", "lane-b"}
        and all(
            isinstance(count, int)
            and not isinstance(count, bool)
            and count >= FAULT_REQUIRED_MATCHED_SAMPLES
            for count in matched_counts.values()
        )
    )
    interruption_bound = (
        injection.get("armed") is True
        and injection.get("signal_sent") is True
        and injection.get("signal") == "SIGTERM"
        and injection.get("target_job_id") == lane_a.job_id
        and marker_a is not None
        and injection.get("target_pid") == marker_a["process_id"]
        and injection.get("target_process_group_id")
        == marker_a["process_group_id"]
        and injection.get("marker_hash") == canonical_json_hash(marker_a)
        and isinstance(signal_time, int)
        and not isinstance(signal_time, bool)
        and signal_time >= int(marker_a["active_start_monotonic_ns"])
        and outcome_a.launch_end_ns >= signal_time
        and outcome_a.result is not None
        and outcome_a.result.pid == marker_a["process_id"]
        and case.summaries[lane_a.job_id] is None
        and case.valid_jobs[lane_a.job_id]
        and counts_bound
    )
    return {
        # Isaac installs a SIGTERM handler and can exit cleanly with returncode 0.
        # Interruption authority is the bound signal receipt plus the absence of
        # a normal summary, not the process' chosen numeric exit status.
        "lane_a_interrupted": interruption_bound,
        "lane_b_completed": case.valid_jobs[lane_b.job_id],
        "lane_a_process_group_clean": not case.measurements[
            lane_a.job_id
        ].orphaned_process,
        "lane_b_process_group_clean": not case.measurements[
            lane_b.job_id
        ].orphaned_process,
        "leases_cleared": _leases_reacquirable(
            lease_manager=lease_manager,
            plan=plan,
        ),
        "compute_pids_cleared": not case.post_case_compute_apps,
        "coordinator_closed": True,
    }


def _leases_reacquirable(
    *, lease_manager: GpuLeaseManager, plan: ResourceShadowPlan
) -> bool:
    try:
        with ExitStack() as stack:
            for lane_id in ("lane-a", "lane-b"):
                lane = plan.lane_assignments[lane_id]
                stack.enter_context(
                    lease_manager.acquire(
                        lane.execution_device,
                        holder=GpuLeaseHolder(
                            session_genesis_hash=plan.session_genesis_hash,
                            generation_id=plan.pilot_id,
                            lane_id=lane_id,
                            invocation_segment_id="p6a-cleanup-check",
                            logical_job_id=f"cleanup-{lane_id}",
                        ),
                    )
                )
        return True
    except Exception:
        return False


def _job_outcome_dict(
    *,
    outcome: _OwnedJobOutcome,
    marker: Mapping[str, Any] | None,
    summary: Mapping[str, Any] | None,
    measurement: ResourceShadowLaneMeasurement,
    valid: bool,
    active_sample_count: int,
) -> dict[str, Any]:
    return {
        "job_id": outcome.job.job_id,
        "returncode": outcome.result.returncode if outcome.result else None,
        "process_id": outcome.result.pid if outcome.result else None,
        "stdout_log": str(outcome.result.stdout_log) if outcome.result else None,
        "stderr_log": str(outcome.result.stderr_log) if outcome.result else None,
        "stdout_tail": outcome.result.stdout_tail if outcome.result else None,
        "stderr_tail": outcome.result.stderr_tail if outcome.result else None,
        "exception_type": outcome.exception_type,
        "exception_message": outcome.exception_message,
        "marker_hash": canonical_json_hash(marker) if marker is not None else None,
        "summary_hash": canonical_json_hash(summary) if summary is not None else None,
        "measurement": measurement.to_dict(),
        "active_sample_count": active_sample_count,
        "valid": valid,
    }


def _active_sample_count(
    *,
    snapshots: Sequence[Mapping[str, Any]],
    marker: Mapping[str, Any] | None,
    summary: Mapping[str, Any] | None,
    expected_uuid: str,
) -> int:
    if marker is None or summary is None:
        return 0
    pid = int(marker["process_id"])
    start = int(marker["active_start_monotonic_ns"])
    end = int(summary["active_end_monotonic_ns"])
    return sum(
        1
        for snapshot in snapshots
        if start <= int(snapshot["monotonic_ns"]) <= end
        and any(
            int(app["pid"]) == pid and app["gpu_uuid"] == expected_uuid
            for app in snapshot["compute_apps"]
        )
    )


def _cross_device_pid(
    *,
    snapshots: Sequence[Mapping[str, Any]],
    marker: Mapping[str, Any] | None,
    expected_uuid: str,
) -> bool:
    if marker is None:
        return False
    pid = int(marker["process_id"])
    return any(
        int(app["pid"]) == pid and app["gpu_uuid"] != expected_uuid
        for snapshot in snapshots
        for app in snapshot["compute_apps"]
    )


def _outcome_has_oom(outcome: _OwnedJobOutcome) -> bool:
    text = " ".join(
        item
        for item in (
            outcome.exception_message,
            outcome.result.stdout_tail if outcome.result else None,
            outcome.result.stderr_tail if outcome.result else None,
        )
        if item
    ).lower()
    return "out of memory" in text or "cuda oom" in text


def _swap_delta_mib(snapshots: Sequence[Mapping[str, Any]]) -> float:
    values = [
        int(snapshot["procfs"]["swap_used_kib"])
        for snapshot in snapshots
        if snapshot["procfs"].get("swap_used_kib") is not None
    ]
    if not values:
        return 0.0
    return max(0, max(values) - values[0]) / 1024.0


def _resource_observations(
    *,
    snapshots: Sequence[Mapping[str, Any]],
    output_dir: Path,
    initial_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    if not snapshots:
        raise ValueError("P6-A produced no resource snapshots")
    initial_memory = _memory_fields(initial_snapshot)
    memory_rows = [_memory_fields(item) for item in snapshots]
    min_available = min(row["MemAvailable"] for row in memory_rows)
    max_available_drop = max(0, initial_memory["MemAvailable"] - min_available)
    max_swap = max(row["SwapTotal"] - row["SwapFree"] for row in memory_rows)
    initial_swap = initial_memory["SwapTotal"] - initial_memory["SwapFree"]
    gpu_rows: dict[str, list[Mapping[str, Any]]] = {}
    for snapshot in snapshots:
        for gpu in snapshot["gpus"]:
            gpu_rows.setdefault(str(gpu["uuid"]), []).append(gpu)
    cpu_first = snapshots[0]["procfs"]
    cpu_last = snapshots[-1]["procfs"]
    total_delta = int(cpu_last["cpu_total_ticks"]) - int(cpu_first["cpu_total_ticks"])
    idle_delta = int(cpu_last["cpu_idle_ticks"]) - int(cpu_first["cpu_idle_ticks"])
    cpu_busy = 1.0 - idle_delta / total_delta if total_delta > 0 else None
    final_output_size = _directory_size_bytes(output_dir)
    sampled_output_sizes = [
        int(item["output_dir_size_bytes"])
        for item in snapshots
        if isinstance(item.get("output_dir_size_bytes"), int)
        and not isinstance(item.get("output_dir_size_bytes"), bool)
    ]
    return {
        "sample_count": len(snapshots),
        "min_mem_available_kib": min_available,
        "max_mem_available_drop_kib": max_available_drop,
        "recommended_future_available_ram_gate_kib": int(
            1.2 * max_available_drop + 10 * 1024**2
        ),
        "max_swap_growth_kib": max(0, max_swap - initial_swap),
        "aggregate_cpu_busy_fraction": cpu_busy,
        "disk_read_sector_delta": int(cpu_last["disk_read_sectors_total"])
        - int(cpu_first["disk_read_sectors_total"]),
        "disk_write_sector_delta": int(cpu_last["disk_write_sectors_total"])
        - int(cpu_first["disk_write_sectors_total"]),
        "peak_output_size_bytes": max([final_output_size, *sampled_output_sizes]),
        "final_output_size_bytes": final_output_size,
        "gpus": {
            uuid: {
                "peak_utilization_gpu_percent": max(
                    float(row["utilization_gpu_percent"] or 0.0) for row in rows
                ),
                "peak_memory_used_mib": max(
                    float(row["memory_used_mib"] or 0.0) for row in rows
                ),
                "peak_power_draw_watts": max(
                    float(row["power_draw_watts"] or 0.0) for row in rows
                ),
            }
            for uuid, rows in gpu_rows.items()
        },
    }


def _timing_observations(cases: Sequence[_CaseOutcome]) -> dict[str, Any]:
    measured_singles: dict[str, list[_OwnedJobOutcome]] = {
        "lane-a": [],
        "lane-b": [],
    }
    paired_case_wall: list[float] = []
    for case in cases:
        if case.jobs[0].mode == "paired" and not case.jobs[0].warmup:
            paired_case_wall.append(
                max(item.full_job_wall_seconds for item in case.job_outcomes)
            )
        for outcome in case.job_outcomes:
            if (
                outcome.job.mode == "single"
                and not outcome.job.warmup
                and outcome.job.lane_id in measured_singles
            ):
                measured_singles[outcome.job.lane_id].append(outcome)
    return {
        "lane_a_single_median_full_job_wall_seconds": statistics.median(
            item.full_job_wall_seconds for item in measured_singles["lane-a"]
        ),
        "lane_b_single_median_full_job_wall_seconds": statistics.median(
            item.full_job_wall_seconds for item in measured_singles["lane-b"]
        ),
        "paired_median_case_wall_seconds": statistics.median(paired_case_wall),
    }


def _memory_fields(snapshot: Mapping[str, Any]) -> dict[str, int]:
    memory = snapshot["procfs"]["memory_kib"]
    fields = ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree")
    if any(isinstance(memory.get(key), bool) or not isinstance(memory.get(key), int) for key in fields):
        raise ValueError("resource snapshot has incomplete memory fields")
    return {key: int(memory[key]) for key in fields}


def _case_sampler_errors(path: Path) -> tuple[Any, ...]:
    payload = read_bounded_regular_file(path, max_bytes=4 * 1024**2, label="P6-A case receipt")
    value = strict_json_object_loads(payload, context="P6-A case receipt")
    errors = value.get("sampler_errors")
    return tuple(errors) if isinstance(errors, list) else ("invalid",)


def _terminate_known_groups(markers: Sequence[Mapping[str, Any] | None]) -> None:
    for marker in markers:
        if marker is None:
            continue
        pgid = int(marker["process_group_id"])
        if pgid == os.getpgrp():
            continue
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def _process_group_alive(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            return False
        raise
    return True


def _load_genesis(path: Path) -> SessionGenesisArtifact:
    payload = read_bounded_regular_file(path, max_bytes=2 * 1024**2, label="P6-A Session Genesis")
    value = strict_json_object_loads(payload, context="P6-A Session Genesis")
    if payload != canonical_json_bytes(value, trailing_newline=True):
        raise ValueError("P6-A Session Genesis is not canonical JSON")
    return SessionGenesisArtifact.from_mapping(value)


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
        timeout=30,
    )
    if completed.returncode != 0:
        raise RuntimeError("git preflight failed: " + " ".join(args))
    return completed.stdout.strip()


def _git_status_dirty() -> bool:
    return bool(_git("status", "--porcelain", "--untracked-files=no"))


def _lower_hex_commit(value: Any) -> str:
    if not isinstance(value, str) or len(value) != 40 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError("expected source commit must be a full lowercase git hash")
    return value


def _fresh_absolute_path(value: str, field_name: str) -> Path:
    path = _absolute_path(value, field_name)
    if path.exists():
        raise FileExistsError(f"P6-A {field_name} must be fresh")
    if not path.parent.is_dir():
        raise FileNotFoundError(f"P6-A {field_name} parent does not exist")
    return path


def _existing_absolute_file(value: str, field_name: str) -> Path:
    path = _absolute_path(value, field_name)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"P6-A {field_name} must be an existing regular file")
    return path.resolve(strict=True)


def _existing_absolute_executable(value: str, field_name: str) -> Path:
    path = _absolute_path(value, field_name).resolve(strict=True)
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError(f"P6-A {field_name} must resolve to an executable file")
    return path


def _absolute_path(value: str, field_name: str) -> Path:
    if not isinstance(value, str) or not value or "://" in value:
        raise ValueError(f"P6-A {field_name} must be a local path")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"P6-A {field_name} must be absolute")
    return path.resolve(strict=False)


def _create_once_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    create_once_bytes(path, canonical_json_bytes(dict(payload), trailing_newline=True))


def _publish_failure_if_possible(output_dir: Path, exception_type: str) -> None:
    try:
        resolved = output_dir.resolve(strict=False)
        if not resolved.is_dir():
            return
        target = resolved / "pilot_failure.json"
        if target.exists():
            return
        _create_once_json(
            target,
            {
                "schema_version": PILOT_FAILURE_SCHEMA_VERSION,
                "exception_type": exception_type,
                "status": "incomplete",
                "acceptance_authority": False,
                "ledger_authority": False,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
    except Exception:
        return


def _sha256_path(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_size_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


if __name__ == "__main__":
    raise SystemExit(main())
