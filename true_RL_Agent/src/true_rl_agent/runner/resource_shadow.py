"""Task-neutral contracts and metrics for the fixed width-two resource shadow.

This module deliberately does not launch work or interpret task results.  It
freezes two physical GPU bindings, records one-shot host/resource observations,
and computes the pre-registered operational gates for P6-A.
"""

from __future__ import annotations

import csv
import math
import os
import re
import statistics
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from true_rl_agent.agent_core.generation_contract import (
    GENERATION_LANE_IDS,
    GenerationLaneAssignment,
    generation_resource_plan_hash,
)
from true_rl_agent.evaluation.binding import canonical_json_hash, require_sha256_digest
from true_rl_agent.fs_contract import create_once_bytes, read_bounded_regular_file
from true_rl_agent.json_contract import canonical_json_bytes, strict_json_object_loads
from true_rl_agent.runner.gpu_lease import GpuDeviceIdentity


RESOURCE_SHADOW_PLAN_SCHEMA_VERSION = "true_rl_agent.resource_shadow_plan.v2"
RESOURCE_SHADOW_SNAPSHOT_SCHEMA_VERSION = "true_rl_agent.resource_snapshot.v1"
RESOURCE_SHADOW_SUMMARY_SCHEMA_VERSION = "true_rl_agent.resource_shadow_summary.v2"
RESOURCE_SHADOW_PLAN_FILENAME = "resource_shadow_plan.json"
RESOURCE_SHADOW_WIDTH = 2
_MAX_PLAN_BYTES = 256 * 1024
_HEX_COMMIT = re.compile(r"^[0-9a-f]{7,64}$")

RESOURCE_SHADOW_AUTHORITY_COUNTER_KEYS = (
    "provider_requests",
    "public_eval_queries",
    "sealed_or_final_confirmation_queries",
    "primary_ledger_writes",
    "selector_acceptance_or_head_transitions",
    "exploration_updates",
)

RESOURCE_SHADOW_THRESHOLD_KEYS = (
    "minimum_measured_active_seconds",
    "median_overlap_ratio_min",
    "median_active_slowdown_max",
    "max_active_slowdown",
    "median_paired_gpu_hours_ratio_max",
    "single_baseline_gpu_hours_ratio_max",
    "swap_delta_mib_max",
)

DEFAULT_RESOURCE_SHADOW_THRESHOLDS: Mapping[str, float] = MappingProxyType(
    {
        "minimum_measured_active_seconds": 60.0,
        "median_overlap_ratio_min": 0.90,
        "median_active_slowdown_max": 1.15,
        "max_active_slowdown": 1.25,
        "median_paired_gpu_hours_ratio_max": 1.15,
        "single_baseline_gpu_hours_ratio_max": 2.30,
        "swap_delta_mib_max": 64.0,
    }
)


@dataclass(frozen=True)
class NvidiaGpuInventoryEntry:
    ordinal: int
    uuid: str
    pci_bus_id: str
    name: str
    memory_total_mib: int
    driver_version: str
    machine_id: str

    def __post_init__(self) -> None:
        if isinstance(self.ordinal, bool) or not isinstance(self.ordinal, int) or self.ordinal < 0:
            raise ValueError("NVIDIA GPU ordinal must be a non-negative integer")
        for field_name in (
            "uuid",
            "pci_bus_id",
            "name",
            "driver_version",
            "machine_id",
        ):
            _bounded_text(getattr(self, field_name), f"NVIDIA GPU {field_name}", maximum=256)
        if (
            isinstance(self.memory_total_mib, bool)
            or not isinstance(self.memory_total_mib, int)
            or self.memory_total_mib <= 0
        ):
            raise ValueError("NVIDIA GPU memory_total_mib must be positive")

    @property
    def execution_device(self) -> str:
        return f"cuda:{self.ordinal}"

    @property
    def identity(self) -> GpuDeviceIdentity:
        return GpuDeviceIdentity(
            execution_device=self.execution_device,
            machine_id=self.machine_id,
            physical_id=self.uuid,
        )

    @property
    def identity_hash(self) -> str:
        return self.identity.identity_hash

    @property
    def physical_resource_hash(self) -> str:
        return self.identity.physical_resource_hash

    def to_dict(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "execution_device": self.execution_device,
            "uuid": self.uuid,
            "pci_bus_id": self.pci_bus_id,
            "name": self.name,
            "memory_total_mib": self.memory_total_mib,
            "driver_version": self.driver_version,
            "machine_id": self.machine_id,
            "identity_hash": self.identity_hash,
            "physical_resource_hash": self.physical_resource_hash,
        }

    @classmethod
    def from_mapping(cls, value: Any) -> "NvidiaGpuInventoryEntry":
        mapping = _exact_mapping(
            value,
            {
                "ordinal",
                "execution_device",
                "uuid",
                "pci_bus_id",
                "name",
                "memory_total_mib",
                "driver_version",
                "machine_id",
                "identity_hash",
                "physical_resource_hash",
            },
            "NVIDIA GPU inventory entry",
        )
        entry = cls(
            ordinal=mapping["ordinal"],
            uuid=mapping["uuid"],
            pci_bus_id=mapping["pci_bus_id"],
            name=mapping["name"],
            memory_total_mib=mapping["memory_total_mib"],
            driver_version=mapping["driver_version"],
            machine_id=mapping["machine_id"],
        )
        if mapping["execution_device"] != entry.execution_device:
            raise ValueError("NVIDIA GPU execution-device mapping mismatch")
        if mapping["identity_hash"] != entry.identity_hash:
            raise ValueError("NVIDIA GPU identity hash mismatch")
        if mapping["physical_resource_hash"] != entry.physical_resource_hash:
            raise ValueError("NVIDIA GPU physical-resource hash mismatch")
        return entry


@dataclass(frozen=True)
class ResourceShadowJob:
    job_id: str
    case_id: str
    mode: str
    repeat_index: int
    warmup: bool
    lane_id: str
    execution_device: str
    process_execution_device: str

    def __post_init__(self) -> None:
        _bounded_text(self.job_id, "resource shadow job_id")
        _bounded_text(self.case_id, "resource shadow case_id")
        if self.mode not in {"single", "paired", "interruption"}:
            raise ValueError("resource shadow job mode is invalid")
        if (
            isinstance(self.repeat_index, bool)
            or not isinstance(self.repeat_index, int)
            or self.repeat_index < 1
        ):
            raise ValueError("resource shadow repeat_index must be positive")
        if not isinstance(self.warmup, bool):
            raise TypeError("resource shadow warmup must be boolean")
        if self.lane_id not in GENERATION_LANE_IDS:
            raise ValueError("resource shadow lane_id is invalid")
        expected_device = "cuda:0" if self.lane_id == "lane-a" else "cuda:1"
        if self.execution_device != expected_device:
            raise ValueError("resource shadow lane/device binding mismatch")
        if self.process_execution_device != "cuda:0":
            raise ValueError(
                "resource shadow isolated process device must be logical cuda:0"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "case_id": self.case_id,
            "mode": self.mode,
            "repeat_index": self.repeat_index,
            "warmup": self.warmup,
            "lane_id": self.lane_id,
            "execution_device": self.execution_device,
            "process_execution_device": self.process_execution_device,
        }

    @classmethod
    def from_mapping(cls, value: Any) -> "ResourceShadowJob":
        return cls(**_exact_mapping(value, set(_RESOURCE_SHADOW_JOB_KEYS), "resource shadow job"))


_RESOURCE_SHADOW_JOB_KEYS = (
    "job_id",
    "case_id",
    "mode",
    "repeat_index",
    "warmup",
    "lane_id",
    "execution_device",
    "process_execution_device",
)


def build_resource_shadow_jobs(
    *, warmup_repeats: int = 1, measured_repeats: int = 3
) -> tuple[ResourceShadowJob, ...]:
    """Build the sole supported single/paired/interruption matrix."""

    if warmup_repeats != 1:
        raise ValueError("resource shadow requires exactly one warm-up per mode")
    if measured_repeats != 3:
        raise ValueError("resource shadow requires exactly three measured repeats")
    jobs: list[ResourceShadowJob] = []
    for lane_id, device in (("lane-a", "cuda:0"), ("lane-b", "cuda:1")):
        label = "gpu0" if lane_id == "lane-a" else "gpu1"
        for warmup, repeat_count in ((True, warmup_repeats), (False, measured_repeats)):
            phase = "warmup" if warmup else "measured"
            for repeat_index in range(1, repeat_count + 1):
                case_id = f"single-{label}-{phase}-{repeat_index:02d}"
                jobs.append(
                    ResourceShadowJob(
                        job_id=f"{case_id}-{lane_id}",
                        case_id=case_id,
                        mode="single",
                        repeat_index=repeat_index,
                        warmup=warmup,
                        lane_id=lane_id,
                        execution_device=device,
                        process_execution_device="cuda:0",
                    )
                )
    for warmup, repeat_count in ((True, warmup_repeats), (False, measured_repeats)):
        phase = "warmup" if warmup else "measured"
        for repeat_index in range(1, repeat_count + 1):
            case_id = f"paired-{phase}-{repeat_index:02d}"
            for lane_id, device in (("lane-a", "cuda:0"), ("lane-b", "cuda:1")):
                jobs.append(
                    ResourceShadowJob(
                        job_id=f"{case_id}-{lane_id}",
                        case_id=case_id,
                        mode="paired",
                        repeat_index=repeat_index,
                        warmup=warmup,
                        lane_id=lane_id,
                        execution_device=device,
                        process_execution_device="cuda:0",
                    )
                )
    case_id = "interruption-probe-01"
    for lane_id, device in (("lane-a", "cuda:0"), ("lane-b", "cuda:1")):
        jobs.append(
            ResourceShadowJob(
                job_id=f"{case_id}-{lane_id}",
                case_id=case_id,
                mode="interruption",
                repeat_index=1,
                warmup=False,
                lane_id=lane_id,
                execution_device=device,
                process_execution_device="cuda:0",
            )
        )
    return tuple(jobs)


@dataclass(frozen=True)
class ResourceShadowPlan:
    pilot_id: str
    source_commit: str
    session_genesis_hash: str
    execution_environment_manifest_hash: str
    fixture_hash: str
    workload_hash: str
    lanes: tuple[GenerationLaneAssignment, GenerationLaneAssignment]
    gpu_inventory: tuple[NvidiaGpuInventoryEntry, NvidiaGpuInventoryEntry]
    jobs: tuple[ResourceShadowJob, ...]
    resource_plan_hash: str
    warmup_repeats: int
    measured_repeats: int
    sample_interval_seconds: float
    thresholds: Mapping[str, float]
    authority_counters: Mapping[str, int]
    schema_version: str = RESOURCE_SHADOW_PLAN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RESOURCE_SHADOW_PLAN_SCHEMA_VERSION:
            raise ValueError("resource shadow plan schema mismatch")
        _bounded_text(self.pilot_id, "resource shadow pilot_id")
        if (
            not isinstance(self.source_commit, str)
            or _HEX_COMMIT.fullmatch(self.source_commit) is None
        ):
            raise ValueError("resource shadow source_commit must be a lowercase git hex id")
        for field_name in (
            "session_genesis_hash",
            "execution_environment_manifest_hash",
            "fixture_hash",
            "workload_hash",
            "resource_plan_hash",
        ):
            require_sha256_digest(getattr(self, field_name), f"resource shadow {field_name}")
        if self.warmup_repeats != 1 or self.measured_repeats != 3:
            raise ValueError("resource shadow repeat counts must remain 1 warm-up and 3 measured")
        if not isinstance(self.sample_interval_seconds, int | float) or isinstance(
            self.sample_interval_seconds, bool
        ) or not math.isfinite(float(self.sample_interval_seconds)) or float(
            self.sample_interval_seconds
        ) != 1.0:
            raise ValueError("resource shadow sample interval must be exactly 1.0 second")
        if (
            not isinstance(self.lanes, tuple)
            or len(self.lanes) != RESOURCE_SHADOW_WIDTH
            or tuple(lane.lane_id for lane in self.lanes) != GENERATION_LANE_IDS
            or tuple(lane.execution_device for lane in self.lanes) != ("cuda:0", "cuda:1")
        ):
            raise ValueError("resource shadow lanes must be lane-a/cuda:0 then lane-b/cuda:1")
        if self.resource_plan_hash != generation_resource_plan_hash(self.lanes):
            raise ValueError("resource shadow resource-plan hash mismatch")
        if (
            not isinstance(self.gpu_inventory, tuple)
            or len(self.gpu_inventory) != RESOURCE_SHADOW_WIDTH
            or tuple(item.ordinal for item in self.gpu_inventory) != (0, 1)
        ):
            raise ValueError("resource shadow inventory must contain GPU ordinals 0 and 1")
        if len({item.uuid for item in self.gpu_inventory}) != RESOURCE_SHADOW_WIDTH:
            raise ValueError("resource shadow GPU ordinals alias the same physical UUID")
        if (
            len({item.physical_resource_hash for item in self.gpu_inventory})
            != RESOURCE_SHADOW_WIDTH
        ):
            raise ValueError("resource shadow lanes must bind distinct physical GPUs")
        machine_ids = {item.machine_id for item in self.gpu_inventory}
        if len(machine_ids) != 1:
            raise ValueError("resource shadow GPU inventory must come from one machine")
        for lane, inventory in zip(self.lanes, self.gpu_inventory, strict=True):
            if lane.resource_binding_hash != inventory.identity_hash:
                raise ValueError("resource shadow lane resource binding does not match inventory")
        expected_jobs = build_resource_shadow_jobs(
            warmup_repeats=self.warmup_repeats,
            measured_repeats=self.measured_repeats,
        )
        if tuple(self.jobs) != expected_jobs:
            raise ValueError("resource shadow jobs do not match the fixed run matrix")
        job_ids = [job.job_id for job in self.jobs]
        if len(set(job_ids)) != len(job_ids):
            raise ValueError("resource shadow job ids must be unique")
        thresholds = _validated_thresholds(self.thresholds)
        counters = _validated_authority_counters(self.authority_counters)
        object.__setattr__(self, "thresholds", MappingProxyType(thresholds))
        object.__setattr__(self, "authority_counters", MappingProxyType(counters))

    @property
    def plan_hash(self) -> str:
        return canonical_json_hash(self._hash_payload())

    @property
    def lane_assignments(self) -> Mapping[str, GenerationLaneAssignment]:
        return MappingProxyType({lane.lane_id: lane for lane in self.lanes})

    def job(self, job_id: str) -> ResourceShadowJob:
        matches = [item for item in self.jobs if item.job_id == job_id]
        if len(matches) != 1:
            raise ValueError("resource shadow job_id is unknown or duplicate")
        return matches[0]

    def inventory_for_execution_device(
        self, execution_device: str
    ) -> NvidiaGpuInventoryEntry:
        """Resolve one host-visible CUDA ordinal to its frozen physical GPU."""

        matches = [
            item
            for item in self.gpu_inventory
            if item.execution_device == execution_device
        ]
        if len(matches) != 1:
            raise ValueError(
                "resource shadow execution device has no unique physical GPU"
            )
        return matches[0]

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "pilot_id": self.pilot_id,
            "source_commit": self.source_commit,
            "session_genesis_hash": self.session_genesis_hash,
            "execution_environment_manifest_hash": self.execution_environment_manifest_hash,
            "fixture_hash": self.fixture_hash,
            "workload_hash": self.workload_hash,
            "lanes": [lane.to_dict() for lane in self.lanes],
            "gpu_inventory": [item.to_dict() for item in self.gpu_inventory],
            "jobs": [item.to_dict() for item in self.jobs],
            "resource_plan_hash": self.resource_plan_hash,
            "warmup_repeats": self.warmup_repeats,
            "measured_repeats": self.measured_repeats,
            "sample_interval_seconds": float(self.sample_interval_seconds),
            "thresholds": dict(self.thresholds),
            "authority_counters": dict(self.authority_counters),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._hash_payload(), "plan_hash": self.plan_hash}

    @classmethod
    def from_mapping(cls, value: Any) -> "ResourceShadowPlan":
        mapping = _exact_mapping(
            value,
            {
                "schema_version",
                "pilot_id",
                "source_commit",
                "session_genesis_hash",
                "execution_environment_manifest_hash",
                "fixture_hash",
                "workload_hash",
                "lanes",
                "gpu_inventory",
                "jobs",
                "resource_plan_hash",
                "warmup_repeats",
                "measured_repeats",
                "sample_interval_seconds",
                "thresholds",
                "authority_counters",
                "plan_hash",
            },
            "resource shadow plan",
        )
        raw_lanes = _object_list(mapping["lanes"], "resource shadow lanes")
        raw_inventory = _object_list(mapping["gpu_inventory"], "resource shadow GPU inventory")
        raw_jobs = _object_list(mapping["jobs"], "resource shadow jobs")
        plan = cls(
            schema_version=mapping["schema_version"],
            pilot_id=mapping["pilot_id"],
            source_commit=mapping["source_commit"],
            session_genesis_hash=mapping["session_genesis_hash"],
            execution_environment_manifest_hash=mapping[
                "execution_environment_manifest_hash"
            ],
            fixture_hash=mapping["fixture_hash"],
            workload_hash=mapping["workload_hash"],
            lanes=tuple(GenerationLaneAssignment.from_mapping(item) for item in raw_lanes),
            gpu_inventory=tuple(
                NvidiaGpuInventoryEntry.from_mapping(item) for item in raw_inventory
            ),
            jobs=tuple(ResourceShadowJob.from_mapping(item) for item in raw_jobs),
            resource_plan_hash=mapping["resource_plan_hash"],
            warmup_repeats=mapping["warmup_repeats"],
            measured_repeats=mapping["measured_repeats"],
            sample_interval_seconds=mapping["sample_interval_seconds"],
            thresholds=mapping["thresholds"],
            authority_counters=mapping["authority_counters"],
        )
        if mapping["plan_hash"] != plan.plan_hash:
            raise ValueError("resource shadow plan hash mismatch")
        return plan


def create_resource_shadow_plan(
    path: Path,
    *,
    pilot_id: str,
    source_commit: str,
    session_genesis_hash: str,
    execution_environment_manifest_hash: str,
    fixture_hash: str,
    workload_hash: str,
    lanes: tuple[GenerationLaneAssignment, GenerationLaneAssignment],
    gpu_inventory: tuple[NvidiaGpuInventoryEntry, NvidiaGpuInventoryEntry],
    jobs: tuple[ResourceShadowJob, ...] | None = None,
    warmup_repeats: int = 1,
    measured_repeats: int = 3,
    sample_interval_seconds: float = 1.0,
    thresholds: Mapping[str, float] | None = None,
    authority_counters: Mapping[str, int] | None = None,
) -> ResourceShadowPlan:
    target = Path(path)
    plan = ResourceShadowPlan(
        pilot_id=pilot_id,
        source_commit=source_commit,
        session_genesis_hash=session_genesis_hash,
        execution_environment_manifest_hash=execution_environment_manifest_hash,
        fixture_hash=fixture_hash,
        workload_hash=workload_hash,
        lanes=lanes,
        gpu_inventory=gpu_inventory,
        jobs=jobs
        if jobs is not None
        else build_resource_shadow_jobs(
            warmup_repeats=warmup_repeats, measured_repeats=measured_repeats
        ),
        resource_plan_hash=generation_resource_plan_hash(lanes),
        warmup_repeats=warmup_repeats,
        measured_repeats=measured_repeats,
        sample_interval_seconds=sample_interval_seconds,
        thresholds=(
            DEFAULT_RESOURCE_SHADOW_THRESHOLDS if thresholds is None else thresholds
        ),
        authority_counters=(
            {key: 0 for key in RESOURCE_SHADOW_AUTHORITY_COUNTER_KEYS}
            if authority_counters is None
            else authority_counters
        ),
    )
    payload = canonical_json_bytes(plan.to_dict(), trailing_newline=True)
    if len(payload) > _MAX_PLAN_BYTES:
        raise ValueError("resource shadow plan exceeds its byte limit")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        existing = read_bounded_regular_file(
            target, max_bytes=_MAX_PLAN_BYTES, label="resource shadow plan"
        )
        if existing != payload:
            raise FileExistsError("resource shadow plan already exists with different bytes")
    else:
        create_once_bytes(target, payload)
    return load_resource_shadow_plan(target)


def load_resource_shadow_plan(path: Path) -> ResourceShadowPlan:
    payload = read_bounded_regular_file(
        Path(path), max_bytes=_MAX_PLAN_BYTES, label="resource shadow plan"
    )
    value = strict_json_object_loads(payload, context="resource shadow plan")
    if payload != canonical_json_bytes(value, trailing_newline=True):
        raise ValueError("resource shadow plan is not canonical JSON")
    return ResourceShadowPlan.from_mapping(value)


def require_resource_shadow_child_binding(
    plan: ResourceShadowPlan,
    *,
    plan_hash: str,
    session_genesis_hash: str,
    execution_environment_manifest_hash: str,
    fixture_hash: str,
    workload_hash: str,
    resource_plan_hash: str,
    job_id: str,
    case_id: str,
    lane_id: str,
    physical_execution_device: str,
    process_execution_device: str,
) -> ResourceShadowJob:
    if not isinstance(plan, ResourceShadowPlan):
        raise TypeError("resource shadow child binding requires a plan")
    expected = {
        "plan_hash": plan.plan_hash,
        "session_genesis_hash": plan.session_genesis_hash,
        "execution_environment_manifest_hash": plan.execution_environment_manifest_hash,
        "fixture_hash": plan.fixture_hash,
        "workload_hash": plan.workload_hash,
        "resource_plan_hash": plan.resource_plan_hash,
    }
    supplied = {
        "plan_hash": plan_hash,
        "session_genesis_hash": session_genesis_hash,
        "execution_environment_manifest_hash": execution_environment_manifest_hash,
        "fixture_hash": fixture_hash,
        "workload_hash": workload_hash,
        "resource_plan_hash": resource_plan_hash,
    }
    for field_name, expected_value in expected.items():
        if supplied[field_name] != expected_value:
            raise ValueError(f"resource shadow child {field_name} mismatch")
    job = plan.job(job_id)
    if (
        job.case_id != case_id
        or job.lane_id != lane_id
        or job.execution_device != physical_execution_device
        or job.process_execution_device != process_execution_device
    ):
        raise ValueError("resource shadow child job binding mismatch")
    lane = plan.lane_assignments[lane_id]
    if lane.execution_device != physical_execution_device:
        raise ValueError("resource shadow child lane binding mismatch")
    return job


def parse_nvidia_smi_inventory(
    text: str, *, machine_id: str
) -> tuple[NvidiaGpuInventoryEntry, ...]:
    _bounded_text(machine_id, "machine_id", maximum=256)
    if not isinstance(text, str):
        raise TypeError("nvidia-smi inventory output must be text")
    entries: list[NvidiaGpuInventoryEntry] = []
    for row in csv.reader(text.splitlines(), skipinitialspace=True):
        if not row or all(not item.strip() for item in row):
            continue
        if len(row) != 6:
            raise ValueError("nvidia-smi inventory row must contain six fields")
        ordinal_text, uuid, pci_bus_id, name, memory_text, driver_version = (
            item.strip() for item in row
        )
        try:
            ordinal = int(ordinal_text)
            memory_total_mib = int(float(memory_text))
        except ValueError as exc:
            raise ValueError("nvidia-smi inventory numeric field is invalid") from exc
        entries.append(
            NvidiaGpuInventoryEntry(
                ordinal=ordinal,
                uuid=uuid,
                pci_bus_id=pci_bus_id,
                name=name,
                memory_total_mib=memory_total_mib,
                driver_version=driver_version,
                machine_id=machine_id,
            )
        )
    if not entries:
        raise ValueError("nvidia-smi returned no GPU inventory")
    if len({item.ordinal for item in entries}) != len(entries):
        raise ValueError("nvidia-smi returned duplicate GPU ordinals")
    if len({item.uuid for item in entries}) != len(entries):
        raise ValueError("multiple CUDA ordinals alias the same physical GPU UUID")
    return tuple(sorted(entries, key=lambda item: item.ordinal))


def discover_nvidia_gpu_inventory(
    *,
    nvidia_smi: str = "nvidia-smi",
    machine_id_path: Path = Path("/etc/machine-id"),
) -> tuple[NvidiaGpuInventoryEntry, ...]:
    machine_id = _read_small_text(machine_id_path, "machine id", max_bytes=4096).strip()
    completed = _run_command(
        [
            nvidia_smi,
            "--query-gpu=index,uuid,pci.bus_id,name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    return parse_nvidia_smi_inventory(completed.stdout, machine_id=machine_id)


def sample_resource_snapshot(
    *,
    inventory: Sequence[NvidiaGpuInventoryEntry],
    output_dir: Path | None = None,
    nvidia_smi: str = "nvidia-smi",
    proc_root: Path = Path("/proc"),
) -> dict[str, Any]:
    """Collect one synchronous diagnostic sample; callers own sampling cadence."""

    entries = tuple(inventory)
    if not entries:
        raise ValueError("resource snapshot requires a non-empty GPU inventory")
    if len({item.uuid for item in entries}) != len(entries):
        raise ValueError("resource snapshot inventory aliases a physical GPU")
    gpu_output = _run_command(
        [
            nvidia_smi,
            "--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total,power.draw",
            "--format=csv,noheader,nounits",
        ]
    ).stdout
    apps_output = _run_command(
        [
            nvidia_smi,
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        allow_no_rows=True,
    ).stdout
    gpu_rows = _parse_gpu_sample_rows(gpu_output)
    expected_uuids = {item.uuid for item in entries}
    observed = {item["uuid"] for item in gpu_rows if item["uuid"] in expected_uuids}
    if observed != expected_uuids:
        raise ValueError("resource snapshot did not observe every planned GPU UUID")
    compute_apps = _parse_compute_app_rows(apps_output)
    proc = _sample_procfs(proc_root, compute_apps)
    return {
        "schema_version": RESOURCE_SHADOW_SNAPSHOT_SCHEMA_VERSION,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "monotonic_ns": time.monotonic_ns(),
        "gpus": [item for item in gpu_rows if item["uuid"] in expected_uuids],
        "compute_apps": [item for item in compute_apps if item["gpu_uuid"] in expected_uuids],
        "procfs": proc,
        "output_dir_size_bytes": _directory_size_bytes(output_dir) if output_dir else None,
        "operational_only": True,
        "acceptance_authority": False,
        "ledger_authority": False,
    }


@dataclass(frozen=True)
class ResourceShadowLaneMeasurement:
    job_id: str
    case_id: str
    repeat_index: int
    lane_id: str
    active_start_ns: int
    active_end_ns: int
    full_job_wall_seconds: float
    valid: bool = True
    oom: bool = False
    swap_delta_mib: float = 0.0
    cross_device_pid: bool = False
    orphaned_process: bool = False

    def __post_init__(self) -> None:
        _bounded_text(self.job_id, "resource measurement job_id")
        _bounded_text(self.case_id, "resource measurement case_id")
        if self.lane_id not in GENERATION_LANE_IDS:
            raise ValueError("resource measurement lane_id is invalid")
        if (
            isinstance(self.repeat_index, bool)
            or not isinstance(self.repeat_index, int)
            or self.repeat_index < 1
        ):
            raise ValueError("resource measurement repeat_index must be positive")
        for field_name in ("active_start_ns", "active_end_ns"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"resource measurement {field_name} is invalid")
        if self.active_end_ns <= self.active_start_ns:
            raise ValueError("resource measurement active interval must be positive")
        _finite_nonnegative(self.full_job_wall_seconds, "full_job_wall_seconds")
        _finite_nonnegative(self.swap_delta_mib, "swap_delta_mib")
        if self.full_job_wall_seconds < self.active_seconds:
            raise ValueError(
                "resource measurement full_job_wall_seconds cannot be shorter "
                "than the active interval"
            )
        for field_name in ("valid", "oom", "cross_device_pid", "orphaned_process"):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"resource measurement {field_name} must be boolean")

    @property
    def active_seconds(self) -> float:
        return (self.active_end_ns - self.active_start_ns) / 1_000_000_000.0

    @property
    def allocated_gpu_hours(self) -> float:
        return self.active_seconds / 3600.0

    def to_dict(self) -> dict[str, Any]:
        return {field_name: getattr(self, field_name) for field_name in _MEASUREMENT_KEYS}

    @classmethod
    def from_mapping(cls, value: Any) -> "ResourceShadowLaneMeasurement":
        return cls(**_exact_mapping(value, set(_MEASUREMENT_KEYS), "resource shadow measurement"))


_MEASUREMENT_KEYS = (
    "job_id",
    "case_id",
    "repeat_index",
    "lane_id",
    "active_start_ns",
    "active_end_ns",
    "full_job_wall_seconds",
    "valid",
    "oom",
    "swap_delta_mib",
    "cross_device_pid",
    "orphaned_process",
)


def active_interval_overlap_ratio(
    first: ResourceShadowLaneMeasurement,
    second: ResourceShadowLaneMeasurement,
) -> float:
    intersection_ns = max(
        0,
        min(first.active_end_ns, second.active_end_ns)
        - max(first.active_start_ns, second.active_start_ns),
    )
    return intersection_ns / min(
        first.active_end_ns - first.active_start_ns,
        second.active_end_ns - second.active_start_ns,
    )


def summarize_resource_shadow(
    *,
    plan: ResourceShadowPlan,
    single_lane_a: Sequence[ResourceShadowLaneMeasurement | Mapping[str, Any]],
    single_lane_b: Sequence[ResourceShadowLaneMeasurement | Mapping[str, Any]],
    paired: Sequence[
        tuple[
            ResourceShadowLaneMeasurement | Mapping[str, Any],
            ResourceShadowLaneMeasurement | Mapping[str, Any],
        ]
    ],
    fault_probe: Mapping[str, Any],
    authority_counters: Mapping[str, int],
) -> dict[str, Any]:
    """Purely compute the pre-registered P6-A resource metrics and gates."""

    if not isinstance(plan, ResourceShadowPlan):
        raise TypeError("resource shadow summary requires a plan")
    singles_a = tuple(_measurement(item) for item in single_lane_a)
    singles_b = tuple(_measurement(item) for item in single_lane_b)
    pairs = tuple((_measurement(a), _measurement(b)) for a, b in paired)
    expected_indices = tuple(range(1, plan.measured_repeats + 1))
    _require_measurement_series(singles_a, lane_id="lane-a", expected_indices=expected_indices)
    _require_measurement_series(singles_b, lane_id="lane-b", expected_indices=expected_indices)
    for item in (*singles_a, *singles_b):
        _require_measurement_matches_planned_job(plan, item, expected_mode="single")
    if len(pairs) != plan.measured_repeats:
        raise ValueError("resource shadow paired measurement count mismatch")
    pair_by_index: dict[
        int,
        tuple[ResourceShadowLaneMeasurement, ResourceShadowLaneMeasurement],
    ] = {}
    for first, second in pairs:
        if first.lane_id != "lane-a" or second.lane_id != "lane-b":
            raise ValueError("resource shadow paired measurements must be lane-a then lane-b")
        if first.case_id != second.case_id or first.repeat_index != second.repeat_index:
            raise ValueError("resource shadow paired measurement identity mismatch")
        _require_measurement_matches_planned_job(plan, first, expected_mode="paired")
        _require_measurement_matches_planned_job(plan, second, expected_mode="paired")
        if first.repeat_index in pair_by_index:
            raise ValueError("resource shadow paired repeat is duplicated")
        pair_by_index[first.repeat_index] = (first, second)
    if tuple(sorted(pair_by_index)) != expected_indices:
        raise ValueError("resource shadow paired repeat indices mismatch")
    singles_a_by_index = {item.repeat_index: item for item in singles_a}
    singles_b_by_index = {item.repeat_index: item for item in singles_b}
    overlap: list[float] = []
    slowdown_a: list[float] = []
    slowdown_b: list[float] = []
    matched_gpu_ratio: list[float] = []
    one_single_ratio: list[float] = []
    all_measurements = [*singles_a, *singles_b]
    per_repeat: list[dict[str, Any]] = []
    for index in expected_indices:
        pair_a, pair_b = pair_by_index[index]
        single_a = singles_a_by_index[index]
        single_b = singles_b_by_index[index]
        all_measurements.extend((pair_a, pair_b))
        overlap_value = active_interval_overlap_ratio(pair_a, pair_b)
        slowdown_a_value = pair_a.active_seconds / single_a.active_seconds
        slowdown_b_value = pair_b.active_seconds / single_b.active_seconds
        pair_hours = pair_a.allocated_gpu_hours + pair_b.allocated_gpu_hours
        matched_single_hours = single_a.allocated_gpu_hours + single_b.allocated_gpu_hours
        matched_ratio_value = pair_hours / matched_single_hours
        one_single_ratio_value = pair_hours / (matched_single_hours / 2.0)
        overlap.append(overlap_value)
        slowdown_a.append(slowdown_a_value)
        slowdown_b.append(slowdown_b_value)
        matched_gpu_ratio.append(matched_ratio_value)
        one_single_ratio.append(one_single_ratio_value)
        per_repeat.append(
            {
                "repeat_index": index,
                "overlap_ratio": overlap_value,
                "lane_a_active_slowdown": slowdown_a_value,
                "lane_b_active_slowdown": slowdown_b_value,
                "paired_allocated_gpu_hours": pair_hours,
                "matched_single_allocated_gpu_hours": matched_single_hours,
                "paired_to_matched_single_gpu_hours_ratio": matched_ratio_value,
                "paired_to_one_single_baseline_gpu_hours_ratio": one_single_ratio_value,
            }
        )
    counters = _validated_authority_counters(authority_counters)
    if counters != dict(plan.authority_counters):
        raise ValueError("resource shadow authority counters drifted from the plan")
    fault = _validated_fault_probe(fault_probe)
    thresholds = plan.thresholds
    gates = {
        "measured_repeat_counts": len(singles_a) == len(singles_b) == len(pairs) == 3,
        "all_measurements_valid": all(item.valid for item in all_measurements),
        "minimum_measured_active_duration": all(
            item.active_seconds >= thresholds["minimum_measured_active_seconds"]
            for item in all_measurements
        ),
        "median_overlap": statistics.median(overlap) >= thresholds["median_overlap_ratio_min"],
        "lane_a_median_slowdown": statistics.median(slowdown_a)
        <= thresholds["median_active_slowdown_max"],
        "lane_b_median_slowdown": statistics.median(slowdown_b)
        <= thresholds["median_active_slowdown_max"],
        "max_slowdown": max([*slowdown_a, *slowdown_b]) <= thresholds["max_active_slowdown"],
        "paired_gpu_hours": statistics.median(matched_gpu_ratio)
        <= thresholds["median_paired_gpu_hours_ratio_max"],
        "one_single_baseline_gpu_hours": statistics.median(one_single_ratio)
        <= thresholds["single_baseline_gpu_hours_ratio_max"],
        "no_oom": not any(item.oom for item in all_measurements),
        "swap_delta": max(item.swap_delta_mib for item in all_measurements)
        <= thresholds["swap_delta_mib_max"],
        "no_cross_device_pid": not any(item.cross_device_pid for item in all_measurements),
        "no_orphaned_process": not any(item.orphaned_process for item in all_measurements),
        "fault_isolation_and_cleanup": all(fault.values()),
        "operational_authority_zero": all(value == 0 for value in counters.values()),
    }
    return {
        "schema_version": RESOURCE_SHADOW_SUMMARY_SCHEMA_VERSION,
        "plan_hash": plan.plan_hash,
        "per_repeat": per_repeat,
        "aggregates": {
            "median_overlap_ratio": statistics.median(overlap),
            "lane_a_median_active_slowdown": statistics.median(slowdown_a),
            "lane_b_median_active_slowdown": statistics.median(slowdown_b),
            "max_active_slowdown": max([*slowdown_a, *slowdown_b]),
            "median_paired_to_matched_single_gpu_hours_ratio": statistics.median(
                matched_gpu_ratio
            ),
            "median_paired_to_one_single_baseline_gpu_hours_ratio": statistics.median(
                one_single_ratio
            ),
        },
        "fault_probe": fault,
        "authority_counters": counters,
        "gates": gates,
        "go": all(gates.values()),
        "operational_only": True,
        "acceptance_authority": False,
        "ledger_authority": False,
    }


def _measurement(
    value: ResourceShadowLaneMeasurement | Mapping[str, Any],
) -> ResourceShadowLaneMeasurement:
    return (
        value
        if isinstance(value, ResourceShadowLaneMeasurement)
        else ResourceShadowLaneMeasurement.from_mapping(value)
    )


def _require_measurement_series(
    values: Sequence[ResourceShadowLaneMeasurement],
    *,
    lane_id: str,
    expected_indices: tuple[int, ...],
) -> None:
    if len(values) != len(expected_indices):
        raise ValueError("resource shadow single measurement count mismatch")
    if any(item.lane_id != lane_id for item in values):
        raise ValueError("resource shadow single measurement lane mismatch")
    indices = tuple(sorted(item.repeat_index for item in values))
    if indices != expected_indices:
        raise ValueError("resource shadow single repeat indices mismatch")


def _require_measurement_matches_planned_job(
    plan: ResourceShadowPlan,
    measurement: ResourceShadowLaneMeasurement,
    *,
    expected_mode: str,
) -> None:
    job = plan.job(measurement.job_id)
    if (
        job.case_id != measurement.case_id
        or job.repeat_index != measurement.repeat_index
        or job.lane_id != measurement.lane_id
        or job.mode != expected_mode
        or job.warmup
    ):
        raise ValueError("resource shadow measurement does not match its planned job")


def _validated_fault_probe(value: Mapping[str, Any]) -> dict[str, bool]:
    fields = {
        "lane_a_interrupted",
        "lane_b_completed",
        "lane_a_process_group_clean",
        "lane_b_process_group_clean",
        "leases_cleared",
        "compute_pids_cleared",
        "coordinator_closed",
    }
    mapping = _exact_mapping(value, fields, "resource shadow fault probe")
    if any(not isinstance(item, bool) for item in mapping.values()):
        raise TypeError("resource shadow fault-probe fields must be boolean")
    return mapping


def _validated_authority_counters(value: Mapping[str, int]) -> dict[str, int]:
    mapping = _exact_mapping(
        value, set(RESOURCE_SHADOW_AUTHORITY_COUNTER_KEYS), "resource shadow authority counters"
    )
    for field_name, item in mapping.items():
        if isinstance(item, bool) or not isinstance(item, int) or item != 0:
            raise ValueError(f"resource shadow {field_name} must remain zero")
    return mapping


def _validated_thresholds(value: Mapping[str, float]) -> dict[str, float]:
    mapping = _exact_mapping(
        value,
        set(RESOURCE_SHADOW_THRESHOLD_KEYS),
        "resource shadow thresholds",
    )
    normalized: dict[str, float] = {}
    for field_name, item in mapping.items():
        if (
            isinstance(item, bool)
            or not isinstance(item, int | float)
            or not math.isfinite(float(item))
        ):
            raise ValueError(f"resource shadow threshold {field_name} must be finite")
        normalized[field_name] = float(item)
    if normalized != dict(DEFAULT_RESOURCE_SHADOW_THRESHOLDS):
        raise ValueError("resource shadow thresholds must match the pre-registered P6-A gates")
    return normalized


def _parse_gpu_sample_rows(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in csv.reader(text.splitlines(), skipinitialspace=True):
        if not row or all(not item.strip() for item in row):
            continue
        if len(row) != 6:
            raise ValueError("nvidia-smi GPU sample row must contain six fields")
        index, uuid, utilization, used, total, power = (item.strip() for item in row)
        rows.append(
            {
                "ordinal": int(index),
                "uuid": uuid,
                "utilization_gpu_percent": _float_or_none(utilization),
                "memory_used_mib": _float_or_none(used),
                "memory_total_mib": _float_or_none(total),
                "power_draw_watts": _float_or_none(power),
            }
        )
    return rows


def _parse_compute_app_rows(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in csv.reader(text.splitlines(), skipinitialspace=True):
        if not row or all(not item.strip() for item in row):
            continue
        if len(row) == 1 and row[0].strip().lower().startswith("no running"):
            continue
        if len(row) != 4:
            raise ValueError("nvidia-smi compute-app row must contain four fields")
        gpu_uuid, pid, process_name, used = (item.strip() for item in row)
        rows.append(
            {
                "gpu_uuid": gpu_uuid,
                "pid": int(pid),
                "process_name": process_name,
                "used_gpu_memory_mib": _float_or_none(used),
            }
        )
    return rows


def _sample_procfs(proc_root: Path, compute_apps: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    meminfo: dict[str, int] = {}
    meminfo_text = _read_small_text(
        proc_root / "meminfo",
        "proc meminfo",
        max_bytes=256 * 1024,
    )
    for line in meminfo_text.splitlines():
        key, separator, remainder = line.partition(":")
        if not separator:
            continue
        token = remainder.strip().split(maxsplit=1)[0]
        try:
            meminfo[key] = int(token)
        except ValueError:
            continue
    cpu_line = _read_small_text(
        proc_root / "stat",
        "proc stat",
        max_bytes=1024 * 1024,
    ).splitlines()[0]
    fields = cpu_line.split()
    if not fields or fields[0] != "cpu":
        raise ValueError("proc stat has no aggregate CPU row")
    cpu_ticks = [int(item) for item in fields[1:]]
    disk_read_sectors = 0
    disk_write_sectors = 0
    disk_text = _read_small_text(
        proc_root / "diskstats",
        "proc diskstats",
        max_bytes=4 * 1024 * 1024,
    )
    for line in disk_text.splitlines():
        fields = line.split()
        if len(fields) < 14 or fields[2].startswith(("loop", "ram")):
            continue
        disk_read_sectors += int(fields[5])
        disk_write_sectors += int(fields[9])
    processes: list[dict[str, Any]] = []
    for app in compute_apps:
        pid = app.get("pid")
        if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
            continue
        io_path = proc_root / str(pid) / "io"
        stat_path = proc_root / str(pid) / "stat"
        try:
            io_text = _read_small_text(io_path, f"proc {pid} io", max_bytes=64 * 1024)
            stat_text = _read_small_text(stat_path, f"proc {pid} stat", max_bytes=64 * 1024)
        except (OSError, ValueError):
            # The process can exit between the nvidia-smi sample and the
            # per-process /proc reads.  That is normal sampling churn, not a
            # failure of the host-wide resource observation.
            continue
        io_values: dict[str, int] = {}
        for line in io_text.splitlines():
            key, separator, raw = line.partition(":")
            if separator:
                try:
                    io_values[key] = int(raw.strip())
                except ValueError:
                    pass
        stat_fields = stat_text.split()
        processes.append(
            {
                "pid": pid,
                "utime_ticks": int(stat_fields[13]) if len(stat_fields) > 14 else None,
                "stime_ticks": int(stat_fields[14]) if len(stat_fields) > 14 else None,
                "read_bytes": io_values.get("read_bytes"),
                "write_bytes": io_values.get("write_bytes"),
            }
        )
    swap_total = meminfo.get("SwapTotal")
    swap_free = meminfo.get("SwapFree")
    return {
        "memory_kib": {
            key: meminfo.get(key)
            for key in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree")
        },
        "swap_used_kib": (
            swap_total - swap_free
            if swap_total is not None and swap_free is not None
            else None
        ),
        "cpu_total_ticks": sum(cpu_ticks),
        "cpu_idle_ticks": sum(cpu_ticks[index] for index in (3, 4) if index < len(cpu_ticks)),
        "disk_read_sectors_total": disk_read_sectors,
        "disk_write_sectors_total": disk_write_sectors,
        "compute_processes": processes,
    }


def _run_command(
    command: list[str],
    *,
    allow_no_rows: bool = False,
) -> subprocess.CompletedProcess[str]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired, UnicodeError) as exc:
        raise RuntimeError(f"failed to run resource command: {command[0]}") from exc
    if completed.returncode != 0:
        if allow_no_rows and "No running processes" in completed.stderr:
            return completed
        raise RuntimeError(
            f"resource command failed with exit code {completed.returncode}: {command[0]}"
        )
    return completed


def _read_small_text(path: Path, label: str, *, max_bytes: int) -> str:
    payload = read_bounded_regular_file(Path(path), max_bytes=max_bytes, label=label)
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not UTF-8") from exc


def _directory_size_bytes(path: Path) -> int:
    total = 0
    root = Path(path)
    if not root.exists():
        return 0
    for directory, _, filenames in os.walk(root, followlinks=False):
        for filename in filenames:
            candidate = Path(directory) / filename
            try:
                if candidate.is_symlink() or not candidate.is_file():
                    continue
                total += candidate.stat().st_size
            except OSError:
                continue
    return total


def _float_or_none(value: str) -> float | None:
    if value in {"N/A", "[Not Supported]", "Not Supported"}:
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("nvidia-smi returned a non-finite numeric value")
    return number


def _exact_mapping(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{label} fields mismatch")
    return dict(value)


def _object_list(value: Any, label: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(item, Mapping) for item in value):
        raise ValueError(f"{label} must be an array of objects")
    return value


def _bounded_text(value: Any, label: str, *, maximum: int = 128) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f"{label} must be a non-empty string of at most {maximum} chars")
    return value


def _finite_nonnegative(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise ValueError(f"resource measurement {label} must be finite and non-negative")
    return float(value)


__all__ = [
    "DEFAULT_RESOURCE_SHADOW_THRESHOLDS",
    "RESOURCE_SHADOW_AUTHORITY_COUNTER_KEYS",
    "RESOURCE_SHADOW_PLAN_FILENAME",
    "RESOURCE_SHADOW_PLAN_SCHEMA_VERSION",
    "RESOURCE_SHADOW_SNAPSHOT_SCHEMA_VERSION",
    "RESOURCE_SHADOW_SUMMARY_SCHEMA_VERSION",
    "NvidiaGpuInventoryEntry",
    "ResourceShadowJob",
    "ResourceShadowLaneMeasurement",
    "ResourceShadowPlan",
    "active_interval_overlap_ratio",
    "build_resource_shadow_jobs",
    "create_resource_shadow_plan",
    "discover_nvidia_gpu_inventory",
    "load_resource_shadow_plan",
    "parse_nvidia_smi_inventory",
    "require_resource_shadow_child_binding",
    "sample_resource_snapshot",
    "summarize_resource_shadow",
]
