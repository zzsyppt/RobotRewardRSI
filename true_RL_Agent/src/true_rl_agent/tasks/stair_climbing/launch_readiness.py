"""Task-owned launch-resource readiness for formal Stair generation.

The P6-D mainline produces and consumes only v3 readiness receipts.  The v1
validator remains as the shared strict core used by v3; retired v2 evidence is
audited from its source-bound historical commit rather than current mainline.
Stair-specific thresholds intentionally remain here at the task leaf while
inventory sampling and physical leases reuse runner-owned primitives.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import shutil
import socket
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any

from true_rl_agent.agent_core.generation_contract import (
    GENERATION_SESSION_CONTRACT_FILENAME,
    GenerationSessionContract,
    load_generation_session_contract,
)
from true_rl_agent.agent_core.generation_lifecycle import (
    GENERATION_CONTINUATION_POLICY_FILENAME,
    GENERATION_CONTINUATION_POLICY_V2_SCHEMA_VERSION,
    GenerationLifecycleIdentity,
    audit_generation_lifecycle,
    load_generation_continuation_policy,
)
from true_rl_agent.agent_core.ledger import load_ledger, primary_ledger_epoch_hash
from true_rl_agent.agent_core.session_genesis import (
    SESSION_GENESIS_FILENAME,
    SessionGenesisArtifact,
)
from true_rl_agent.composition.cell_manifest import (
    RESOLVED_CELL_MANIFEST_FILENAME,
    load_resolved_cell_manifest,
)
from true_rl_agent.evaluation.binding import (
    canonical_json_hash,
    require_sha256_checksum,
    require_sha256_digest,
)
from true_rl_agent.fs_contract import create_once_bytes, read_bounded_regular_file
from true_rl_agent.json_contract import canonical_json_bytes, strict_json_object_loads
from true_rl_agent.runner.gpu_lease import GpuLeaseManager, PreGenesisGpuLeaseHolder
from true_rl_agent.runner.resource_shadow import (
    NvidiaGpuInventoryEntry,
    discover_nvidia_gpu_inventory,
    sample_resource_snapshot,
)
from true_rl_agent.tasks.stair_climbing.composition import (
    StairClimbingCompositionConfig,
)
from true_rl_agent.tasks.stair_climbing.config import (
    STAIR_TASK_ID,
    StairClimbingTaskConfig,
)
from true_rl_agent.tasks.stair_climbing.final_confirmation import (
    STAIR_FINAL_CONFIRMATION_DIRNAME,
    STAIR_FINAL_CONFIRMATION_PLAN_FILENAME,
    STAIR_FINAL_CONFIRMATION_REQUIRED_GENERATIONS,
    STAIR_FINAL_CONFIRMATION_RESERVATION_FILENAME,
    STAIR_FINAL_CONFIRMATION_RESULT_FILENAME,
    STAIR_GENERATION_REPORT_FILENAME,
    load_stair_generation_final_confirmation_plan,
    load_stair_generation_final_confirmation_reservation,
    load_stair_generation_final_confirmation_result,
)
from true_rl_agent.tasks.stair_climbing.generation_session_cost import (
    FORMAL_GENERATION_SESSION_MODE,
    STAIR_GENERATION_ACTUAL_COST_DIRNAME,
    STAIR_GENERATION_FORMAL_RETRY_OUTPUT_BYTES,
    STAIR_GENERATION_FORMAL_SUCCESSFUL_OUTPUT_BYTES,
    STAIR_GENERATION_SESSION_COST_FILENAME,
    STAIR_FORMAL_GENERATION_PAUSE_BOUNDARIES,
    StairGenerationActualCostJournal,
    StairGenerationSessionCostContractV3,
    load_stair_generation_session_cost_contract,
    replay_stair_generation_combined_cost_chain,
    replay_stair_generation_formal_cost_chain,
)
from true_rl_agent.tasks.stair_climbing.provider_readiness import (
    validate_stair_provider_credential,
)
from true_rl_agent.tasks.stair_climbing.source_checkout import (
    inspect_stair_source_checkout,
)
from true_rl_agent.tasks.stair_climbing.unified_scoring import (
    objective_family_mask_from_public_result,
)

STAIR_GENERATION_LAUNCH_READINESS_SCHEMA_VERSION = (
    "true_rl_agent.stair.generation_launch_readiness.v1"
)
STAIR_GENERATION_LAUNCH_READINESS_V3_SCHEMA_VERSION = (
    "true_rl_agent.stair.generation_launch_readiness.v3"
)
STAIR_GENERATION_LAUNCH_READINESS_REF_SCHEMA_VERSION = (
    "true_rl_agent.stair.generation_launch_readiness_ref.v1"
)
STAIR_GENERATION_LAUNCH_READINESS_FILENAME = "launch_resource_readiness.json"
DEFAULT_SAMPLE_DURATION_SECONDS = 600.0
DEFAULT_SAMPLE_INTERVAL_SECONDS = 1.0
MAX_SAMPLE_GAP_MULTIPLIER = 2
MAX_READINESS_AGE_SECONDS = 300.0
MAX_READINESS_FUTURE_SKEW_SECONDS = 30.0
MIN_AVAILABLE_RAM_KIB = 25 * 1024 * 1024
MAX_SWAP_GROWTH_KIB = 64 * 1024
MIN_FREE_DISK_BYTES = 30 * 1024**3
OUTPUT_DISK_MULTIPLIER = 3
FORMAL_REMAINING_OUTPUT_SAFETY_MULTIPLIER = 1.20
FORMAL_REMAINING_OUTPUT_SAFETY_NUMERATOR = 6
FORMAL_REMAINING_OUTPUT_SAFETY_DENOMINATOR = 5
FORMAL_DISK_HEADROOM_BYTES = 10 * 1024**3
FORMAL_MAX_OUTPUT_BYTES = (
    STAIR_GENERATION_FORMAL_SUCCESSFUL_OUTPUT_BYTES
    + STAIR_GENERATION_FORMAL_RETRY_OUTPUT_BYTES
)
MAX_READINESS_RECEIPT_BYTES = 8 * 1024 * 1024
MAX_PRELAUNCH_PROBE_EVIDENCE_BYTES = 8 * 1024 * 1024
STAIR_FORMAL_LAUNCH_SUBJECT_FRESH_ABSENT = "formal_fresh_absent"
STAIR_FORMAL_LAUNCH_SUBJECT_NONTERMINAL_RESUME = "formal_nonterminal_resume"
STAIR_FORMAL_LAUNCH_SUBJECT_TERMINAL_FINALIZATION = "formal_terminal_finalization"
STAIR_FORMAL_LAUNCH_SUBJECT_TERMINAL_AUDIT_ONLY = "formal_terminal_audit_only"
STAIR_FORMAL_LAUNCH_SUBJECT_MODES = frozenset(
    {
        STAIR_FORMAL_LAUNCH_SUBJECT_FRESH_ABSENT,
        STAIR_FORMAL_LAUNCH_SUBJECT_NONTERMINAL_RESUME,
        STAIR_FORMAL_LAUNCH_SUBJECT_TERMINAL_FINALIZATION,
        STAIR_FORMAL_LAUNCH_SUBJECT_TERMINAL_AUDIT_ONLY,
    }
)
STAIR_GENERATION_PROVIDER_API_KEY_ENV = "DEEPSEEK_API_KEY"
STAIR_GENERATION_SOURCE_CHECKOUT_FILENAME = "generation_source_checkout.json"
STAIR_GENERATION_SOURCE_CHECKOUT_SCHEMA_VERSION = (
    "true_rl_agent.stair.generation_source_checkout.v1"
)
P6B_ATTEMPT_CHAIN_PROBE_SCHEMA_VERSION = "true_rl_agent.p6b.attempt_chain_probe.v1"
P6B_STAIR_TRAIN_TEARDOWN_LAUNCH_SCHEMA_VERSION = (
    "true_rl_agent.p6b.stair_train_teardown_probe.launch.v1"
)
P6B_STAIR_TRAIN_TEARDOWN_RESULT_SCHEMA_VERSION = (
    "true_rl_agent.p6b.stair_train_teardown_probe.result.v1"
)

_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}\Z")
_GPU_UUID_PATTERN = re.compile(
    r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\Z"
)
_BOOT_ID_PATTERN = re.compile(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\Z")

_ROOT_KEYS = {
    "schema_version",
    "status",
    "binding",
    "source",
    "producer",
    "gpu_inventory",
    "environment",
    "thresholds",
    "observations",
    "checks",
    "authority",
    "receipt_hash",
}
_V3_ROOT_KEYS = (_ROOT_KEYS - {"provider_readiness"}) | {
    "execution",
    "prelaunch_evidence",
    "provider_credential",
    "subject",
}
_BINDING_KEYS = {
    "formal_output_dir",
    "source_root",
    "session_id",
    "source_commit",
    "machine_id",
    "hostname",
    "boot_id",
    "gpu_uuids",
    "lease_root",
    "max_output_bytes",
    "nvidia_smi_executable",
}
_SOURCE_KEYS = {
    "head_commit",
    "tracked_clean_before",
    "tracked_clean_after",
}
_PRODUCER_KEYS = {
    "argv",
    "python_executable",
    "entrypoint",
    "working_directory",
    "process_id",
    "process_group_id",
    "process_session_id",
    "critical_sources",
}
_FILE_EVIDENCE_KEYS = {"path", "sha256", "size_bytes"}
_ENVIRONMENT_KEYS = {
    "cuda_visible_devices_set",
    "nvidia_visible_devices_set",
}
_THRESHOLD_KEYS = {
    "sample_duration_seconds",
    "sample_interval_seconds",
    "maximum_sample_gap_nanoseconds",
    "minimum_available_ram_kib",
    "maximum_swap_growth_kib",
    "minimum_free_disk_bytes",
    "output_disk_multiplier",
    "required_free_disk_bytes",
}
_V3_THRESHOLD_KEYS = {
    "sample_duration_seconds",
    "sample_interval_seconds",
    "maximum_sample_gap_nanoseconds",
    "minimum_available_ram_kib",
    "maximum_swap_growth_kib",
    "minimum_free_disk_bytes",
    "remaining_output_safety_multiplier",
    "fixed_disk_headroom_bytes",
    "current_accounted_output_bytes",
    "remaining_worst_case_output_bytes",
    "required_free_disk_bytes",
}
_OBSERVATION_KEYS = {
    "started_at_utc",
    "completed_at_utc",
    "window_start_monotonic_ns",
    "window_end_monotonic_ns",
    "window_sample_count",
    "maximum_sample_gap_nanoseconds_observed",
    "all_sample_count",
    "samples",
    "minimum_available_ram_kib_observed",
    "initial_swap_used_kib",
    "maximum_swap_used_kib",
    "maximum_swap_growth_kib_observed",
    "free_disk_bytes_before",
    "free_disk_bytes_after",
    "initial_lease_cycle",
    "reacquire_lease_cycle",
    "formal_output_absent_before",
    "formal_output_absent_after",
}
_SAMPLE_KEYS = {
    "sequence",
    "phase",
    "captured_at_utc",
    "monotonic_ns",
    "gpu_uuids",
    "target_compute_app_count",
    "memory_available_kib",
    "swap_used_kib",
    "free_disk_bytes",
}
_LEASE_CYCLE_KEYS = {
    "cycle",
    "lane_ids",
    "execution_devices",
    "gpu_uuids",
    "resource_binding_hashes",
    "physical_resource_hashes",
    "held_sample_sequence",
    "simultaneously_held",
    "released",
}
_CHECK_KEYS = {
    "exact_two_gpu_inventory",
    "no_parent_device_remap",
    "tracked_source_clean_and_head_matches",
    "formal_output_absent_before_and_after",
    "initial_dual_gpu_acquire_release",
    "fixed_sampling_window_completed",
    "target_gpus_compute_idle_throughout",
    "minimum_available_ram_maintained",
    "swap_growth_within_limit",
    "disk_space_sufficient_before_and_after",
    "dual_gpu_reacquire_release",
}
_AUTHORITY_KEYS = {
    "operational_only",
    "provider_requests",
    "gpu_workloads",
    "ledger_writes",
    "scientific_claim_authority",
    "acceptance_authority",
    "ledger_authority",
}
_V3_CHECK_KEYS = (
    _CHECK_KEYS
    - {
        "formal_output_absent_before_and_after",
    }
) | {
    "formal_output_subject_unchanged_before_and_after",
    "production_shaped_attempt_chain_fault_probe_passed",
    "provider_credential_read_only",
    "real_stair_train_teardown_probe_passed",
    "remaining_output_disk_formula_bound",
}
_FORMAL_AUTHORITY_KEYS = {
    "operational_only",
    "provider_readiness_requests",
    "generation_provider_requests",
    "gpu_workloads",
    "formal_output_writes",
    "ledger_writes",
    "scientific_claim_authority",
    "acceptance_authority",
    "ledger_authority",
}
_V3_EXECUTION_KEYS = {
    "critical_sources",
    "formal_generation_command_sha256",
}
_PRELAUNCH_EVIDENCE_KEYS = {
    "attempt_chain_fault_probe_report",
    "stair_train_teardown_result",
}
_ATTEMPT_CHAIN_PROBE_ROOT_KEYS = {
    "schema_version",
    "status",
    "started_utc",
    "finished_utc",
    "command_argv",
    "host",
    "source",
    "owned_python_executable",
    "probe_identity_hash",
    "attempt_chain",
    "process_observations",
    "checks",
    "file_sha256",
}
_ATTEMPT_CHAIN_PROBE_SOURCE_KEYS = {
    "source_commit",
    "tracked_clean",
    "source_file_sha256",
}
_ATTEMPT_CHAIN_PROBE_HOST_KEYS = {
    "hostname",
    "boot_id",
    "kernel",
    "python_executable",
    "python_version",
}
_ATTEMPT_CHAIN_PROBE_CHECK_KEYS = {
    "exactly_two_create_once_results",
    "retry_chain_closed",
    "zero_process_group_residual",
}
_ATTEMPT_CHAIN_PROBE_CHAIN_KEYS = {
    "attempt_result_count",
    "result_files",
    "first_result_hash",
    "retry_previous_result_hash",
    "terminal_chain_head_hash",
    "terminal_output_hash",
    "republish_created",
    "republish_bytes_unchanged",
}
_ATTEMPT_CHAIN_PROBE_OBSERVATION_KEYS = {
    "attempt_index",
    "pid",
    "pgid",
    "grandchild_pid",
    "live_pids_after_return",
    "stdout_log",
    "stderr_log",
}
_STAIR_TEARDOWN_RESULT_ROOT_KEYS = {
    "schema_version",
    "status",
    "created_at_utc",
    "launch_identity_hash",
    "source",
    "host",
    "child",
    "startup_compute_idle",
    "post_exit_compute_idle_samples",
    "process_group_live_members_after_return",
    "detached_processes_referencing_child_output",
    "lease_reacquired",
    "formal_output_absent",
    "launch_checksum",
    "operational_only",
    "scientific_claim_authority",
    "acceptance_authority",
    "ledger_authority",
}
_STAIR_TEARDOWN_SOURCE_KEYS = {"commit", "tracked_clean", "checksums"}
_STAIR_TEARDOWN_HOST_KEYS = {
    "hostname",
    "boot_id",
    "parent_process_id",
    "isaac_python",
    "nvidia_smi_executable",
    "target_gpu",
}
_STAIR_TEARDOWN_LAUNCH_ROOT_KEYS = {
    "schema_version",
    "status",
    "created_at_utc",
    "launch_identity_hash",
    "source",
    "host",
    "probe_argv",
    "formal_output_dir",
    "lease_root",
    "child_argv",
    "child_environment_overrides",
    "timeout_s",
    "termination_grace_s",
    "workload",
    "operational_only",
    "scientific_claim_authority",
    "acceptance_authority",
    "ledger_authority",
    "startup_compute_idle",
    "lease_held",
}
_PROVIDER_CREDENTIAL_KEYS = {
    "schema_version",
    "api_key_env",
    "checks",
    "privacy",
    "side_effects",
}
STAIR_PROVIDER_CREDENTIAL_READINESS_SCHEMA_VERSION = (
    "true_rl_agent.stair.provider_credential_readiness.v1"
)
_RESUME_SUBJECT_KEYS = {
    "subject_mode",
    "source_identity",
    "session_genesis",
    "generation_session_contract",
    "continuation_policy",
    "ledger",
    "lifecycle",
    "cost",
    "snapshot_hash",
}
_V3_RESUME_SUBJECT_KEYS = _RESUME_SUBJECT_KEYS | {
    "attempt_state",
    "final_confirmation",
    "formal_report",
    "current_accounted_output_bytes",
}


@dataclass(frozen=True)
class StairGenerationLaunchReadinessConfig:
    receipt_path: Path
    formal_output_dir: Path
    session_id: str
    source_root: Path
    expected_source_commit: str
    expected_machine_id: str
    expected_gpu_uuids: tuple[str, str]
    lease_root: Path
    max_output_bytes: int
    nvidia_smi_executable: Path | str
    producer_argv: tuple[str, ...]
    sample_duration_seconds: float = DEFAULT_SAMPLE_DURATION_SECONDS
    sample_interval_seconds: float = DEFAULT_SAMPLE_INTERVAL_SECONDS
    machine_id_path: Path = Path("/etc/machine-id")
    boot_id_path: Path = Path("/proc/sys/kernel/random/boot_id")
    proc_meminfo_path: Path = Path("/proc/meminfo")

    def __post_init__(self) -> None:
        for field_name in (
            "receipt_path",
            "formal_output_dir",
            "source_root",
            "lease_root",
            "machine_id_path",
            "boot_id_path",
            "proc_meminfo_path",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, Path):
                raise TypeError(f"Stair launch readiness {field_name} must be a Path")
        source_root = self.source_root.resolve(strict=True)
        if not source_root.is_dir():
            raise ValueError("Stair launch readiness source_root must be a directory")
        expected_module = (
            source_root
            / "src"
            / "true_rl_agent"
            / "tasks"
            / "stair_climbing"
            / "launch_readiness.py"
        ).resolve(strict=True)
        if Path(__file__).resolve() != expected_module:
            raise ValueError(
                "Stair launch readiness source_root does not own the running module"
            )
        nvidia_smi = resolve_stair_nvidia_smi_executable(self.nvidia_smi_executable)
        normalized_paths = {
            "receipt_path": self.receipt_path.resolve(strict=False),
            "formal_output_dir": self.formal_output_dir.resolve(strict=False),
            "source_root": source_root,
            "lease_root": self.lease_root.resolve(strict=False),
            "nvidia_smi_executable": nvidia_smi,
            "machine_id_path": self.machine_id_path.resolve(strict=False),
            "boot_id_path": self.boot_id_path.resolve(strict=False),
            "proc_meminfo_path": self.proc_meminfo_path.resolve(strict=False),
        }
        for field_name, value in normalized_paths.items():
            if not value.is_absolute():
                raise ValueError(
                    f"Stair launch readiness {field_name} must resolve absolutely"
                )
            object.__setattr__(self, field_name, value)
        if _path_is_within(self.receipt_path, self.formal_output_dir):
            raise ValueError(
                "launch readiness receipt must be outside the formal output directory"
            )
        _bounded_text(self.session_id, "Stair launch readiness session_id")
        if _COMMIT_PATTERN.fullmatch(self.expected_source_commit) is None:
            raise ValueError(
                "Stair launch readiness expected_source_commit must be a full lowercase git commit"
            )
        _bounded_text(
            self.expected_machine_id,
            "Stair launch readiness expected_machine_id",
            maximum=256,
        )
        if (
            not isinstance(self.expected_gpu_uuids, tuple)
            or len(self.expected_gpu_uuids) != 2
            or any(
                not isinstance(value, str) or _GPU_UUID_PATTERN.fullmatch(value) is None
                for value in self.expected_gpu_uuids
            )
            or self.expected_gpu_uuids[0] == self.expected_gpu_uuids[1]
        ):
            raise ValueError(
                "Stair launch readiness requires two distinct full GPU UUIDs"
            )
        _positive_int(self.max_output_bytes, "Stair launch readiness max_output_bytes")
        if (
            not isinstance(self.producer_argv, tuple)
            or not self.producer_argv
            or len(self.producer_argv) > 256
            or any(
                not isinstance(value, str) or not value or len(value) > 4096
                for value in self.producer_argv
            )
            or sum(len(value) for value in self.producer_argv) > 64 * 1024
        ):
            raise ValueError("Stair launch readiness producer argv is invalid")
        duration = _finite_number(
            self.sample_duration_seconds,
            "Stair launch readiness sample_duration_seconds",
        )
        interval = _finite_number(
            self.sample_interval_seconds,
            "Stair launch readiness sample_interval_seconds",
        )
        if duration < DEFAULT_SAMPLE_DURATION_SECONDS:
            raise ValueError(
                "Stair launch readiness sampling window must be at least 600s"
            )
        if interval <= 0.0 or interval > DEFAULT_SAMPLE_INTERVAL_SECONDS:
            raise ValueError(
                "Stair launch readiness sampling interval must be at most 1s"
            )
        object.__setattr__(self, "sample_duration_seconds", duration)
        object.__setattr__(self, "sample_interval_seconds", interval)


@dataclass(frozen=True)
class StairGenerationLaunchReadinessConfigV3:
    """Bind one exact P6-D fresh, resume, finalization, or audit subject."""

    base: StairGenerationLaunchReadinessConfig
    subject_mode: str
    provider_env_file: Path
    formal_generation_command_sha256: str
    attempt_chain_fault_probe_report: Path
    stair_train_teardown_result: Path

    def __post_init__(self) -> None:
        if not isinstance(self.base, StairGenerationLaunchReadinessConfig):
            raise TypeError("Stair launch readiness v3 base config is invalid")
        if self.subject_mode not in STAIR_FORMAL_LAUNCH_SUBJECT_MODES:
            raise ValueError("Stair launch readiness v3 subject_mode is invalid")
        if self.base.max_output_bytes != FORMAL_MAX_OUTPUT_BYTES:
            raise ValueError(
                "Stair launch readiness v3 requires the frozen P6-D output cap"
            )
        if not isinstance(self.provider_env_file, Path):
            raise TypeError("Stair launch readiness provider_env_file must be a Path")
        provider_env_file = self.provider_env_file.resolve(strict=True)
        if not provider_env_file.is_file():
            raise ValueError(
                "Stair launch readiness provider_env_file must be a regular file"
            )
        object.__setattr__(self, "provider_env_file", provider_env_file)
        probe_paths: dict[str, Path] = {}
        for field_name in (
            "attempt_chain_fault_probe_report",
            "stair_train_teardown_result",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, Path):
                raise TypeError(f"Stair launch readiness {field_name} must be a Path")
            target = value.resolve(strict=True)
            if not target.is_file():
                raise ValueError(
                    f"Stair launch readiness {field_name} must be a regular file"
                )
            if _path_is_within(target, self.base.formal_output_dir):
                raise ValueError(
                    "Stair launch readiness probe evidence must be outside the "
                    "formal output directory"
                )
            probe_paths[field_name] = target
            object.__setattr__(self, field_name, target)
        if len(set(probe_paths.values())) != len(probe_paths):
            raise ValueError("Stair launch readiness probe evidence paths must differ")
        object.__setattr__(
            self,
            "formal_generation_command_sha256",
            require_sha256_checksum(
                self.formal_generation_command_sha256,
                "Stair formal generation command sha256",
            ),
        )


def resolve_stair_nvidia_smi_executable(value: Path | str) -> Path:
    """Resolve a bare PATH command or an absolute executable, never a relative path."""

    if not isinstance(value, Path | str):
        raise TypeError("Stair nvidia-smi executable must be a path or command name")
    raw = os.fspath(value)
    if not raw:
        raise ValueError("Stair nvidia-smi executable must be non-empty")
    candidate = Path(raw)
    if candidate.is_absolute():
        target = candidate.resolve(strict=True)
    else:
        if "/" in raw or "\\" in raw:
            raise ValueError(
                "Stair nvidia-smi executable must be an absolute path or bare PATH name"
            )
        discovered = shutil.which(raw)
        if discovered is None:
            raise FileNotFoundError(
                f"Stair nvidia-smi executable is unavailable: {raw}"
            )
        target = Path(discovered).resolve(strict=True)
    if not target.is_file():
        raise ValueError("Stair nvidia-smi executable must be a regular file")
    return target


def _producer_identity(
    config: StairGenerationLaunchReadinessConfig,
) -> dict[str, Any]:
    entrypoint = (
        config.source_root / "scripts" / "run_stair_generation_launch_readiness.py"
    ).resolve(strict=True)
    return {
        "argv": list(config.producer_argv),
        "python_executable": _file_evidence(Path(sys.executable).resolve(strict=True)),
        "entrypoint": _file_evidence(entrypoint),
        "working_directory": str(Path.cwd().resolve(strict=True)),
        "process_id": os.getpid(),
        "process_group_id": _process_identity("getpgid"),
        "process_session_id": _process_identity("getsid"),
        "critical_sources": [
            _file_evidence(path)
            for path in _producer_critical_source_paths(config.source_root)
        ],
    }


def _producer_critical_source_paths(
    source_root: Path,
    *,
    strict: bool = True,
) -> tuple[Path, ...]:
    package_root = source_root / "src" / "true_rl_agent"
    paths = (
        (package_root / "tasks" / "stair_climbing" / "launch_readiness.py"),
        package_root / "runner" / "gpu_lease.py",
        package_root / "runner" / "resource_shadow.py",
    )
    return tuple(path.resolve(strict=strict) for path in paths)




def _critical_execution_paths(
    source_root: Path,
    *,
    strict: bool = True,
) -> tuple[Path, ...]:
    package_root = source_root / "src" / "true_rl_agent"
    paths = (
        source_root / "scripts" / "run_stair_generation_launch_readiness.py",
        source_root / "scripts" / "launch_stair_generation_session.py",
        source_root / "scripts" / "run_stair_generation_session.py",
        package_root / "tasks" / "stair_climbing" / "launch_readiness.py",
        package_root / "tasks" / "stair_climbing" / "generation_session.py",
        package_root / "tasks" / "stair_climbing" / "source_checkout.py",
        package_root / "runner" / "gpu_lease.py",
        package_root / "runner" / "resource_shadow.py",
    )
    return tuple(path.resolve(strict=strict) for path in paths)


def _v3_execution_identity(
    source_root: Path,
    *,
    formal_generation_command_sha256: str,
) -> dict[str, Any]:
    return {
        "critical_sources": [
            _file_evidence(path) for path in _v3_critical_execution_paths(source_root)
        ],
        "formal_generation_command_sha256": require_sha256_checksum(
            formal_generation_command_sha256,
            "Stair formal generation command sha256",
        ),
    }


def _v3_critical_execution_paths(
    source_root: Path,
    *,
    strict: bool = True,
) -> tuple[Path, ...]:
    package_root = source_root / "src" / "true_rl_agent"
    paths = (
        *_critical_execution_paths(source_root, strict=False),
        source_root / "scripts" / "isaaclab_rsl_rl_eval.py",
        package_root / "agent_core" / "generation_lifecycle.py",
        package_root / "runner" / "job_attempts.py",
        package_root / "tasks" / "stair_climbing" / "entrypoint_contract.py",
        package_root / "tasks" / "stair_climbing" / "final_confirmation.py",
        package_root / "tasks" / "stair_climbing" / "generation_session_cost.py",
        package_root / "tasks" / "stair_climbing" / "phased_evaluation.py",
    )
    return tuple(path.resolve(strict=strict) for path in paths)




def _process_identity(function_name: str) -> int | None:
    function = getattr(os, function_name, None)
    if function is None:
        return None
    try:
        return int(function(0))
    except OSError:
        return None


def _file_evidence(path: Path) -> dict[str, Any]:
    target = Path(path).resolve(strict=True)
    if not target.is_file():
        raise ValueError(
            "Stair launch readiness evidence source must be a regular file"
        )
    digest = hashlib.sha256()
    size_bytes = 0
    with target.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            size_bytes += len(block)
    return {
        "path": str(target),
        "sha256": "sha256:" + digest.hexdigest(),
        "size_bytes": size_bytes,
    }


def _current_prelaunch_evidence(
    config: StairGenerationLaunchReadinessConfigV3,
    *,
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    attempt_ref, attempt_report = _read_canonical_probe_evidence(
        config.attempt_chain_fault_probe_report,
        label="production-shaped attempt-chain fault probe report",
    )
    teardown_ref, teardown_result = _read_canonical_probe_evidence(
        config.stair_train_teardown_result,
        label="real Stair train teardown probe result",
    )
    _validate_attempt_chain_probe_report(attempt_report, binding=binding)
    _validate_stair_teardown_probe_result(
        teardown_result,
        result_path=Path(teardown_ref["path"]),
        binding=binding,
    )
    return {
        "attempt_chain_fault_probe_report": attempt_ref,
        "stair_train_teardown_result": teardown_ref,
    }


def _validate_prelaunch_evidence_mapping(
    value: Any,
    *,
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    evidence = _exact_mapping(
        value,
        _PRELAUNCH_EVIDENCE_KEYS,
        "readiness prelaunch evidence",
    )
    validated: dict[str, Any] = {}
    reports: dict[str, dict[str, Any]] = {}
    labels = {
        "attempt_chain_fault_probe_report": (
            "production-shaped attempt-chain fault probe report"
        ),
        "stair_train_teardown_result": "real Stair train teardown probe result",
    }
    formal_output_dir = Path(str(binding["formal_output_dir"])).resolve(strict=False)
    for key in sorted(_PRELAUNCH_EVIDENCE_KEYS):
        declared = _validate_file_evidence(evidence[key], f"readiness {labels[key]}")
        current, report = _read_canonical_probe_evidence(
            Path(declared["path"]),
            label=labels[key],
        )
        if current != declared:
            raise ValueError(f"readiness {labels[key]} bytes drifted")
        if _path_is_within(Path(current["path"]), formal_output_dir):
            raise ValueError(
                "readiness probe evidence must be outside the formal output "
                "directory"
            )
        validated[key] = current
        reports[key] = report
    if (
        validated["attempt_chain_fault_probe_report"]["path"]
        == validated["stair_train_teardown_result"]["path"]
    ):
        raise ValueError("readiness probe evidence paths must differ")
    _validate_attempt_chain_probe_report(
        reports["attempt_chain_fault_probe_report"],
        binding=binding,
    )
    _validate_stair_teardown_probe_result(
        reports["stair_train_teardown_result"],
        result_path=Path(validated["stair_train_teardown_result"]["path"]),
        binding=binding,
    )
    return validated


def _read_canonical_probe_evidence(
    path: Path,
    *,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    target = Path(path).resolve(strict=True)
    payload = read_bounded_regular_file(
        target,
        max_bytes=MAX_PRELAUNCH_PROBE_EVIDENCE_BYTES,
        label=label,
    )
    report = strict_json_object_loads(payload, context=label)
    if payload != canonical_json_bytes(report, trailing_newline=True):
        raise ValueError(f"{label} is not canonical JSON")
    if not payload:
        raise ValueError(f"{label} must be non-empty")
    return (
        {
            "path": str(target),
            "sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        },
        report,
    )


def _validate_attempt_chain_probe_report(
    value: Any,
    *,
    binding: Mapping[str, Any],
) -> None:
    report = _exact_mapping(
        value,
        _ATTEMPT_CHAIN_PROBE_ROOT_KEYS,
        "attempt-chain fault probe report",
    )
    if (
        report["schema_version"] != P6B_ATTEMPT_CHAIN_PROBE_SCHEMA_VERSION
        or report["status"] != "pass"
    ):
        raise ValueError("attempt-chain fault probe did not pass")
    _parse_utc_timestamp(
        report["started_utc"], "attempt-chain probe started_utc", maximum=128
    )
    _parse_utc_timestamp(
        report["finished_utc"], "attempt-chain probe finished_utc", maximum=128
    )
    command_argv = report["command_argv"]
    if (
        not isinstance(command_argv, list)
        or not command_argv
        or any(not isinstance(item, str) or not item for item in command_argv)
    ):
        raise ValueError("attempt-chain probe command argv is invalid")
    host = _exact_mapping(
        report["host"],
        _ATTEMPT_CHAIN_PROBE_HOST_KEYS,
        "attempt-chain probe host",
    )
    if host["hostname"] != binding["hostname"] or host["boot_id"] != binding["boot_id"]:
        raise ValueError("attempt-chain probe host binding mismatch")
    source = _exact_mapping(
        report["source"],
        _ATTEMPT_CHAIN_PROBE_SOURCE_KEYS,
        "attempt-chain probe source",
    )
    if (
        source["source_commit"] != binding["source_commit"]
        or source["tracked_clean"] is not True
    ):
        raise ValueError("attempt-chain probe source binding mismatch")
    _validate_checksum_mapping(
        source["source_file_sha256"],
        "attempt-chain probe source checksums",
    )
    owned_python = report["owned_python_executable"]
    if not isinstance(owned_python, str) or not Path(owned_python).is_absolute():
        raise ValueError("attempt-chain probe owned Python path is invalid")
    require_sha256_digest(
        report["probe_identity_hash"], "attempt-chain probe identity hash"
    )

    chain = _exact_mapping(
        report["attempt_chain"],
        _ATTEMPT_CHAIN_PROBE_CHAIN_KEYS,
        "attempt-chain probe chain",
    )
    result_files = chain["result_files"]
    if (
        chain["attempt_result_count"] != 2
        or not isinstance(result_files, list)
        or len(result_files) != 2
        or len(set(result_files)) != 2
        or any(not isinstance(item, str) or not item for item in result_files)
    ):
        raise ValueError("attempt-chain probe did not bind exactly two results")
    first_hash = require_sha256_digest(
        chain["first_result_hash"], "attempt-chain probe first result hash"
    )
    retry_predecessor = require_sha256_digest(
        chain["retry_previous_result_hash"],
        "attempt-chain probe retry predecessor hash",
    )
    terminal_hash = require_sha256_digest(
        chain["terminal_chain_head_hash"],
        "attempt-chain probe terminal chain-head hash",
    )
    require_sha256_digest(
        chain["terminal_output_hash"],
        "attempt-chain probe terminal output hash",
    )
    if first_hash != retry_predecessor or terminal_hash == first_hash:
        raise ValueError("attempt-chain probe retry linkage is invalid")
    if (
        chain["republish_created"] != [False, False]
        or chain["republish_bytes_unchanged"] is not True
    ):
        raise ValueError("attempt-chain probe create-once replay is invalid")

    observations = report["process_observations"]
    if not isinstance(observations, list) or len(observations) != 2:
        raise ValueError("attempt-chain probe process observations are incomplete")
    for index, raw_observation in enumerate(observations):
        observation = _exact_mapping(
            raw_observation,
            _ATTEMPT_CHAIN_PROBE_OBSERVATION_KEYS,
            "attempt-chain probe process observation",
        )
        pid = _positive_int(observation["pid"], "attempt-chain probe process id")
        if (
            observation["attempt_index"] != index
            or observation["pgid"] != pid
            or _positive_int(
                observation["grandchild_pid"],
                "attempt-chain probe grandchild process id",
            )
            == pid
            or observation["live_pids_after_return"] != []
        ):
            raise ValueError("attempt-chain probe process cleanup is invalid")
        for field_name in ("stdout_log", "stderr_log"):
            _bounded_text(
                observation[field_name],
                f"attempt-chain probe {field_name}",
                maximum=4096,
            )
    checks = _exact_mapping(
        report["checks"],
        _ATTEMPT_CHAIN_PROBE_CHECK_KEYS,
        "attempt-chain probe checks",
    )
    if any(checks[key] is not True for key in _ATTEMPT_CHAIN_PROBE_CHECK_KEYS):
        raise ValueError("attempt-chain probe checks did not all pass")
    _validate_checksum_mapping(report["file_sha256"], "attempt-chain probe files")


def _validate_stair_teardown_probe_result(
    value: Any,
    *,
    result_path: Path,
    binding: Mapping[str, Any],
) -> None:
    result = _exact_mapping(
        value,
        _STAIR_TEARDOWN_RESULT_ROOT_KEYS,
        "Stair teardown probe result",
    )
    if (
        result["schema_version"] != P6B_STAIR_TRAIN_TEARDOWN_RESULT_SCHEMA_VERSION
        or result["status"] != "pass"
    ):
        raise ValueError("Stair teardown probe did not pass")
    _parse_utc_timestamp(
        result["created_at_utc"], "Stair teardown result created_at_utc", maximum=128
    )
    launch_identity_hash = require_sha256_digest(
        result["launch_identity_hash"], "Stair teardown launch identity hash"
    )
    source = _exact_mapping(
        result["source"],
        _STAIR_TEARDOWN_SOURCE_KEYS,
        "Stair teardown source",
    )
    if (
        source["commit"] != binding["source_commit"]
        or source["tracked_clean"] is not True
    ):
        raise ValueError("Stair teardown source binding mismatch")
    _validate_checksum_mapping(source["checksums"], "Stair teardown source checksums")

    host = _exact_mapping(
        result["host"],
        _STAIR_TEARDOWN_HOST_KEYS,
        "Stair teardown host",
    )
    if (
        host["hostname"] != binding["hostname"]
        or host["boot_id"] != binding["boot_id"]
        or host["nvidia_smi_executable"] != binding["nvidia_smi_executable"]
    ):
        raise ValueError("Stair teardown host binding mismatch")
    _positive_int(host["parent_process_id"], "Stair teardown parent process id")
    isaac_python = host["isaac_python"]
    if not isinstance(isaac_python, str) or not Path(isaac_python).is_absolute():
        raise ValueError("Stair teardown Isaac Python path is invalid")
    target_gpu = NvidiaGpuInventoryEntry.from_mapping(host["target_gpu"])
    if (
        target_gpu.uuid not in binding["gpu_uuids"]
        or target_gpu.machine_id != binding["machine_id"]
    ):
        raise ValueError("Stair teardown target GPU binding mismatch")

    raw_child = result["child"]
    if not isinstance(raw_child, Mapping):
        raise ValueError("Stair teardown child evidence must be an object")
    child = dict(raw_child)
    process_id = _positive_int(
        child.get("process_id"), "Stair teardown child process id"
    )
    if child.get("process_group_id") != process_id or child.get("returncode") != 0:
        raise ValueError("Stair teardown child normal exit is invalid")
    child_environment = child.get("environment_overrides")
    if not isinstance(child_environment, Mapping) or dict(child_environment) != {
        "CUDA_VISIBLE_DEVICES": target_gpu.uuid,
        "NVIDIA_VISIBLE_DEVICES": None,
    }:
        raise ValueError("Stair teardown child GPU environment is invalid")
    for field_name in (
        "stdout_checksum",
        "stderr_checksum",
        "report_checksum",
        "checkpoint_checksum",
    ):
        require_sha256_checksum(
            child.get(field_name), f"Stair teardown child {field_name}"
        )

    startup_idle = _validate_teardown_idle_sample(
        result["startup_compute_idle"],
        expected_stage="startup",
        expected_gpu_uuid=target_gpu.uuid,
    )
    post_exit = result["post_exit_compute_idle_samples"]
    if not isinstance(post_exit, list) or len(post_exit) != 2:
        raise ValueError("Stair teardown post-exit idle evidence is incomplete")
    for raw_sample, stage in zip(
        post_exit,
        ("post-exit-1", "post-exit-2"),
        strict=True,
    ):
        _validate_teardown_idle_sample(
            raw_sample,
            expected_stage=stage,
            expected_gpu_uuid=target_gpu.uuid,
        )
    if (
        result["process_group_live_members_after_return"] != []
        or result["detached_processes_referencing_child_output"] != []
        or result["lease_reacquired"] is not True
        or result["formal_output_absent"] is not True
    ):
        raise ValueError("Stair teardown cleanup checks did not all pass")
    if {
        key: result[key]
        for key in (
            "operational_only",
            "scientific_claim_authority",
            "acceptance_authority",
            "ledger_authority",
        )
    } != {
        "operational_only": True,
        "scientific_claim_authority": False,
        "acceptance_authority": False,
        "ledger_authority": False,
    }:
        raise ValueError("Stair teardown authority projection is invalid")

    launch_checksum = require_sha256_checksum(
        result["launch_checksum"], "Stair teardown launch checksum"
    )
    launch_ref, launch = _read_canonical_probe_evidence(
        result_path.parent / "launch.json",
        label="Stair teardown launch evidence",
    )
    if launch_ref["sha256"] != launch_checksum:
        raise ValueError("Stair teardown launch evidence bytes drifted")
    launch = _exact_mapping(
        launch,
        _STAIR_TEARDOWN_LAUNCH_ROOT_KEYS,
        "Stair teardown launch evidence",
    )
    if (
        launch["schema_version"] != P6B_STAIR_TRAIN_TEARDOWN_LAUNCH_SCHEMA_VERSION
        or launch["status"] != "launched"
        or launch["launch_identity_hash"] != launch_identity_hash
        or launch["source"] != source
        or launch["host"] != host
        or launch["formal_output_dir"] != binding["formal_output_dir"]
        or launch["lease_root"] != binding["lease_root"]
        or launch["startup_compute_idle"] != startup_idle
        or launch["lease_held"] is not True
    ):
        raise ValueError("Stair teardown launch binding mismatch")
    if {
        key: launch[key]
        for key in (
            "operational_only",
            "scientific_claim_authority",
            "acceptance_authority",
            "ledger_authority",
        )
    } != {
        "operational_only": True,
        "scientific_claim_authority": False,
        "acceptance_authority": False,
        "ledger_authority": False,
    }:
        raise ValueError("Stair teardown launch authority projection is invalid")


def _validate_teardown_idle_sample(
    value: Any,
    *,
    expected_stage: str,
    expected_gpu_uuid: str,
) -> dict[str, Any]:
    sample = _exact_mapping(
        value,
        {"stage", "snapshot"},
        "Stair teardown GPU-idle sample",
    )
    if sample["stage"] != expected_stage or not isinstance(sample["snapshot"], Mapping):
        raise ValueError("Stair teardown GPU-idle sample stage is invalid")
    snapshot = dict(sample["snapshot"])
    gpu_rows = snapshot.get("gpus")
    if (
        snapshot.get("compute_apps") != []
        or not isinstance(gpu_rows, list)
        or {item.get("uuid") for item in gpu_rows if isinstance(item, Mapping)}
        != {expected_gpu_uuid}
    ):
        raise ValueError("Stair teardown target GPU is not compute-idle")
    return {"stage": expected_stage, "snapshot": snapshot}


def _validate_checksum_mapping(value: Any, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{label} must be a non-empty object")
    result: dict[str, str] = {}
    for raw_key, raw_checksum in value.items():
        key = _bounded_text(raw_key, f"{label} path", maximum=4096)
        result[key] = require_sha256_checksum(raw_checksum, f"{label} checksum")
    return result


@dataclass(frozen=True)
class StairGenerationLaunchReadinessReceiptRef:
    path: Path
    sha256: str
    size_bytes: int
    receipt_hash: str
    schema_version: str = STAIR_GENERATION_LAUNCH_READINESS_REF_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != STAIR_GENERATION_LAUNCH_READINESS_REF_SCHEMA_VERSION:
            raise ValueError("Stair launch readiness receipt-ref schema mismatch")
        if not isinstance(self.path, Path) or not self.path.is_absolute():
            raise ValueError("Stair launch readiness receipt-ref path must be absolute")
        require_sha256_checksum(
            self.sha256, "Stair launch readiness receipt-ref sha256"
        )
        _positive_int(self.size_bytes, "Stair launch readiness receipt-ref size_bytes")
        if self.size_bytes > MAX_READINESS_RECEIPT_BYTES:
            raise ValueError("Stair launch readiness receipt-ref size exceeds limit")
        require_sha256_digest(
            self.receipt_hash,
            "Stair launch readiness receipt-ref receipt_hash",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "path": str(self.path),
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "receipt_hash": self.receipt_hash,
        }

    @classmethod
    def from_mapping(cls, value: Any) -> StairGenerationLaunchReadinessReceiptRef:
        mapping = _exact_mapping(
            value,
            {"schema_version", "path", "sha256", "size_bytes", "receipt_hash"},
            "Stair launch readiness receipt-ref",
        )
        path = mapping["path"]
        if not isinstance(path, str):
            raise ValueError("Stair launch readiness receipt-ref path must be text")
        return cls(
            schema_version=mapping["schema_version"],
            path=Path(path),
            sha256=mapping["sha256"],
            size_bytes=mapping["size_bytes"],
            receipt_hash=mapping["receipt_hash"],
        )


@dataclass(frozen=True)
class StairGenerationLaunchReadinessReceipt:
    """A strictly validated, canonical readiness receipt."""

    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        normalized = _validate_receipt_mapping(self.payload)
        object.__setattr__(self, "payload", MappingProxyType(normalized))

    @property
    def receipt_hash(self) -> str:
        value = self.payload["receipt_hash"]
        assert isinstance(value, str)
        return value

    def to_dict(self) -> dict[str, Any]:
        return strict_json_object_loads(
            canonical_json_bytes(dict(self.payload)),
            context="Stair launch readiness receipt",
        )

    def assert_fresh(self, *, now: datetime | None = None) -> None:
        current = datetime.now(timezone.utc) if now is None else now
        if current.tzinfo is None or current.utcoffset() != timezone.utc.utcoffset(
            current
        ):
            raise ValueError("Stair launch readiness freshness time must use UTC")
        observations = self.payload["observations"]
        assert isinstance(observations, Mapping)
        completed_at = _parse_utc_timestamp(
            observations["completed_at_utc"],
            "Stair launch readiness completed_at_utc",
            maximum=128,
        )
        age_seconds = (current - completed_at).total_seconds()
        if age_seconds < -MAX_READINESS_FUTURE_SKEW_SECONDS:
            raise ValueError("Stair launch readiness receipt is from the future")
        if age_seconds > MAX_READINESS_AGE_SECONDS:
            raise ValueError("Stair launch readiness receipt is stale")

    def to_ref(self, path: Path) -> StairGenerationLaunchReadinessReceiptRef:
        target = Path(path).resolve(strict=True)
        payload = read_bounded_regular_file(
            target,
            max_bytes=MAX_READINESS_RECEIPT_BYTES,
            label="Stair launch readiness receipt",
        )
        if payload != canonical_json_bytes(self.to_dict(), trailing_newline=True):
            raise ValueError("Stair launch readiness receipt-ref source bytes mismatch")
        return StairGenerationLaunchReadinessReceiptRef(
            path=target,
            sha256="sha256:" + hashlib.sha256(payload).hexdigest(),
            size_bytes=len(payload),
            receipt_hash=self.receipt_hash,
        )

    @classmethod
    def from_mapping(cls, value: Any) -> StairGenerationLaunchReadinessReceipt:
        return cls(payload=value)


InventoryDiscoverer = Callable[
    [StairGenerationLaunchReadinessConfig], Sequence[NvidiaGpuInventoryEntry]
]
ResourceSampler = Callable[
    [Sequence[NvidiaGpuInventoryEntry], StairGenerationLaunchReadinessConfig],
    Mapping[str, Any],
]
SystemSampler = Callable[[StairGenerationLaunchReadinessConfig], Mapping[str, Any]]
SourceInspector = Callable[[Path], Mapping[str, Any]]
FormalLaunchSubjectInspector = Callable[
    [StairGenerationLaunchReadinessConfigV3], Mapping[str, Any]
]


def _collect_stair_generation_launch_readiness_v3(
    config: StairGenerationLaunchReadinessConfig,
    *,
    v3_config: StairGenerationLaunchReadinessConfigV3,
    inventory_discoverer: InventoryDiscoverer | None = None,
    resource_sampler: ResourceSampler | None = None,
    system_sampler: SystemSampler | None = None,
    source_inspector: SourceInspector | None = None,
    formal_subject_inspector: FormalLaunchSubjectInspector | None = None,
    monotonic_ns: Callable[[], int] = time.monotonic_ns,
    utc_now: Callable[[], str] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    hostname_reader: Callable[[], str] = socket.gethostname,
    environ: Mapping[str, str] | None = None,
) -> StairGenerationLaunchReadinessReceipt:
    if not isinstance(config, StairGenerationLaunchReadinessConfig):
        raise TypeError("Stair launch readiness config is invalid")
    if (
        not isinstance(v3_config, StairGenerationLaunchReadinessConfigV3)
        or v3_config.base != config
    ):
        raise TypeError("Stair launch readiness v3 config is invalid")
    _assert_path_absent(config.receipt_path, "launch readiness receipt")
    environment = os.environ if environ is None else environ
    if not isinstance(environment, Mapping):
        raise TypeError("Stair launch readiness environ must be a mapping")
    remap = {
        "cuda_visible_devices_set": "CUDA_VISIBLE_DEVICES" in environment,
        "nvidia_visible_devices_set": "NVIDIA_VISIBLE_DEVICES" in environment,
    }
    if any(remap.values()):
        raise ValueError("Stair launch readiness parent environment remaps GPU devices")

    inspect_source = _inspect_source if source_inspector is None else source_inspector
    source_before = _validated_source_state(
        inspect_source(config.source_root),
        expected_commit=config.expected_source_commit,
    )
    producer = _producer_identity(config)
    machine_id = _read_identity(config.machine_id_path, "machine id", maximum=256)
    if machine_id != config.expected_machine_id:
        raise ValueError("Stair launch readiness machine identity mismatch")
    hostname = _bounded_text(
        hostname_reader(),
        "Stair launch readiness hostname",
        maximum=256,
    )
    boot_id = _read_identity(config.boot_id_path, "boot id", maximum=128)
    if _BOOT_ID_PATTERN.fullmatch(boot_id) is None:
        raise ValueError("Stair launch readiness boot id is invalid")

    launch_binding = _binding_mapping(
        config,
        machine_id=machine_id,
        hostname=hostname,
        boot_id=boot_id,
    )
    execution_before = _v3_execution_identity(
        config.source_root,
        formal_generation_command_sha256=(
            v3_config.formal_generation_command_sha256
        ),
    )
    prelaunch_evidence_before = _current_prelaunch_evidence(
        v3_config,
        binding=launch_binding,
    )
    inspect_formal_subject = (
        _inspect_v3_launch_subject
        if formal_subject_inspector is None
        else formal_subject_inspector
    )
    subject_before = _validated_v3_subject_mapping(
        inspect_formal_subject(v3_config),
        binding=launch_binding,
    )
    provider_credential = _provider_credential_readiness(v3_config)

    discover = (
        _discover_inventory if inventory_discoverer is None else inventory_discoverer
    )
    inventory = _validated_inventory(
        discover(config),
        expected_machine_id=config.expected_machine_id,
        expected_gpu_uuids=config.expected_gpu_uuids,
    )
    sample_gpu = _sample_gpu_resources if resource_sampler is None else resource_sampler
    sample_system = (
        _sample_system_resources if system_sampler is None else system_sampler
    )
    now_utc = _utc_now if utc_now is None else utc_now
    manager = GpuLeaseManager(
        lease_root=config.lease_root,
        devices={item.execution_device: item.identity for item in inventory},
    )
    current_accounted_output_bytes = _v3_subject_current_accounted_output_bytes(
        subject_before
    )
    remaining_worst_case_output_bytes = (
        FORMAL_MAX_OUTPUT_BYTES - current_accounted_output_bytes
    )
    required_free_disk_bytes = _formal_required_free_disk_bytes(
        current_accounted_output_bytes
    )
    started_at_utc = _bounded_text(
        now_utc(),
        "Stair launch readiness started_at_utc",
        maximum=128,
    )
    samples: list[dict[str, Any]] = []
    initial_swap_used_kib: int | None = None

    def capture(phase: str) -> dict[str, Any]:
        nonlocal initial_swap_used_kib
        _bounded_text(phase, "Stair launch readiness sample phase", maximum=64)
        gpu = sample_gpu(inventory, config)
        system = sample_system(config)
        record = _validated_sample_inputs(
            sequence=len(samples),
            phase=phase,
            captured_at_utc=now_utc(),
            monotonic_ns=monotonic_ns(),
            gpu_snapshot=gpu,
            system_snapshot=system,
            expected_gpu_uuids=config.expected_gpu_uuids,
        )
        if initial_swap_used_kib is None:
            initial_swap_used_kib = record["swap_used_kib"]
        swap_growth = max(0, record["swap_used_kib"] - initial_swap_used_kib)
        if record["target_compute_app_count"] != 0:
            raise RuntimeError(
                "target GPUs are not compute-idle during launch readiness"
            )
        if record["memory_available_kib"] < MIN_AVAILABLE_RAM_KIB:
            raise RuntimeError("Stair launch readiness available RAM is below 25 GiB")
        if swap_growth > MAX_SWAP_GROWTH_KIB:
            raise RuntimeError("Stair launch readiness swap growth exceeds 64 MiB")
        if record["free_disk_bytes"] < required_free_disk_bytes:
            raise RuntimeError("Stair launch readiness output filesystem is too full")
        samples.append(record)
        return record

    first = capture("before-initial-lease")
    initial_cycle = _run_dual_lease_cycle(
        manager=manager,
        config=config,
        inventory=inventory,
        cycle="initial",
        capture=capture,
    )
    capture("after-initial-release")

    window_sample_count = (
        math.ceil(config.sample_duration_seconds / config.sample_interval_seconds) + 1
    )
    sample_interval_ns = round(config.sample_interval_seconds * 1_000_000_000)
    maximum_sample_gap_ns = MAX_SAMPLE_GAP_MULTIPLIER * sample_interval_ns
    window_start_ns: int | None = None
    window_end_ns: int | None = None
    previous_window_sample_ns: int | None = None
    observed_sample_gaps_ns: list[int] = []
    for index in range(window_sample_count):
        if index and window_start_ns is not None:
            deadline_ns = window_start_ns + index * sample_interval_ns
            remaining_ns = deadline_ns - monotonic_ns()
            if remaining_ns > 0:
                sleep(remaining_ns / 1_000_000_000)
        sample = capture("sampling-window")
        if window_start_ns is None:
            window_start_ns = sample["monotonic_ns"]
        if previous_window_sample_ns is not None:
            gap_ns = sample["monotonic_ns"] - previous_window_sample_ns
            if gap_ns <= 0 or gap_ns > maximum_sample_gap_ns:
                raise RuntimeError("Stair launch readiness sampling cadence drifted")
            observed_sample_gaps_ns.append(gap_ns)
        previous_window_sample_ns = sample["monotonic_ns"]
        window_end_ns = sample["monotonic_ns"]
    assert window_start_ns is not None and window_end_ns is not None
    required_window_ns = int(config.sample_duration_seconds * 1_000_000_000)
    if window_end_ns - window_start_ns < required_window_ns:
        raise RuntimeError("Stair launch readiness sampling window ended too early")

    reacquire_cycle = _run_dual_lease_cycle(
        manager=manager,
        config=config,
        inventory=inventory,
        cycle="reacquire",
        capture=capture,
    )
    final = capture("after-reacquire-release")
    subject_after = _validated_v3_subject_mapping(
        inspect_formal_subject(v3_config),
        binding=launch_binding,
    )
    if _provider_credential_readiness(v3_config) != provider_credential:
        raise RuntimeError(
            "provider credential boundary changed during launch readiness"
        )
    if subject_after != subject_before:
        raise RuntimeError(
            "formal output launch subject changed during launch readiness"
        )
    prelaunch_evidence_after = _current_prelaunch_evidence(
        v3_config,
        binding=launch_binding,
    )
    if prelaunch_evidence_after != prelaunch_evidence_before:
        raise RuntimeError(
            "prelaunch probe evidence changed during launch readiness"
        )

    source_after = _validated_source_state(
        inspect_source(config.source_root),
        expected_commit=config.expected_source_commit,
    )
    if source_before != source_after:
        raise RuntimeError("tracked source identity changed during launch readiness")
    current_execution = _v3_execution_identity(
        config.source_root,
        formal_generation_command_sha256=(
            v3_config.formal_generation_command_sha256
        ),
    )
    if current_execution != execution_before:
        raise RuntimeError("critical execution source changed during readiness")
    if _read_identity(config.machine_id_path, "machine id", maximum=256) != machine_id:
        raise RuntimeError("machine identity changed during launch readiness")
    if hostname_reader() != hostname:
        raise RuntimeError("hostname changed during launch readiness")
    if _read_identity(config.boot_id_path, "boot id", maximum=128) != boot_id:
        raise RuntimeError("host boot identity changed during launch readiness")

    assert initial_swap_used_kib is not None
    minimum_available_ram = min(item["memory_available_kib"] for item in samples)
    maximum_swap_used = max(item["swap_used_kib"] for item in samples)
    maximum_swap_growth = max(0, maximum_swap_used - initial_swap_used_kib)
    completed_at_utc = _bounded_text(
        now_utc(),
        "Stair launch readiness completed_at_utc",
        maximum=128,
    )
    output_absent = (
        v3_config.subject_mode == STAIR_FORMAL_LAUNCH_SUBJECT_FRESH_ABSENT
    )
    body = {
        "schema_version": STAIR_GENERATION_LAUNCH_READINESS_V3_SCHEMA_VERSION,
        "status": "pass",
        "binding": _binding_mapping(
            config,
            machine_id=machine_id,
            hostname=hostname,
            boot_id=boot_id,
        ),
        "source": {
            "head_commit": source_before["head_commit"],
            "tracked_clean_before": source_before["tracked_clean"],
            "tracked_clean_after": source_after["tracked_clean"],
        },
        "producer": producer,
        "gpu_inventory": [item.to_dict() for item in inventory],
        "environment": remap,
        "thresholds": {
            "sample_duration_seconds": config.sample_duration_seconds,
            "sample_interval_seconds": config.sample_interval_seconds,
            "maximum_sample_gap_nanoseconds": maximum_sample_gap_ns,
            "minimum_available_ram_kib": MIN_AVAILABLE_RAM_KIB,
            "maximum_swap_growth_kib": MAX_SWAP_GROWTH_KIB,
            "minimum_free_disk_bytes": MIN_FREE_DISK_BYTES,
            "remaining_output_safety_multiplier": (
                FORMAL_REMAINING_OUTPUT_SAFETY_MULTIPLIER
            ),
            "fixed_disk_headroom_bytes": FORMAL_DISK_HEADROOM_BYTES,
            "current_accounted_output_bytes": current_accounted_output_bytes,
            "remaining_worst_case_output_bytes": remaining_worst_case_output_bytes,
            "required_free_disk_bytes": required_free_disk_bytes,
        },
        "observations": {
            "started_at_utc": started_at_utc,
            "completed_at_utc": completed_at_utc,
            "window_start_monotonic_ns": window_start_ns,
            "window_end_monotonic_ns": window_end_ns,
            "window_sample_count": window_sample_count,
            "maximum_sample_gap_nanoseconds_observed": max(observed_sample_gaps_ns),
            "all_sample_count": len(samples),
            "samples": samples,
            "minimum_available_ram_kib_observed": minimum_available_ram,
            "initial_swap_used_kib": initial_swap_used_kib,
            "maximum_swap_used_kib": maximum_swap_used,
            "maximum_swap_growth_kib_observed": maximum_swap_growth,
            "free_disk_bytes_before": first["free_disk_bytes"],
            "free_disk_bytes_after": final["free_disk_bytes"],
            "initial_lease_cycle": initial_cycle,
            "reacquire_lease_cycle": reacquire_cycle,
            "formal_output_absent_before": output_absent,
            "formal_output_absent_after": output_absent,
        },
        "execution": execution_before,
        "prelaunch_evidence": prelaunch_evidence_before,
        "provider_credential": provider_credential,
        "subject": subject_before,
        "checks": {key: True for key in sorted(_V3_CHECK_KEYS)},
        "authority": {
            "operational_only": True,
            "provider_readiness_requests": 0,
            "generation_provider_requests": 0,
            "gpu_workloads": 0,
            "formal_output_writes": 0,
            "ledger_writes": 0,
            "scientific_claim_authority": False,
            "acceptance_authority": False,
            "ledger_authority": False,
        },
    }
    receipt_dict = {**body, "receipt_hash": canonical_json_hash(body)}
    receipt = StairGenerationLaunchReadinessReceipt.from_mapping(receipt_dict)
    encoded = canonical_json_bytes(receipt.to_dict(), trailing_newline=True)
    if len(encoded) > MAX_READINESS_RECEIPT_BYTES:
        raise ValueError("Stair launch readiness receipt exceeds its byte limit")
    config.receipt_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    _assert_path_absent(config.receipt_path, "launch readiness receipt")
    create_once_bytes(config.receipt_path, encoded)
    persisted = read_bounded_regular_file(
        config.receipt_path,
        max_bytes=MAX_READINESS_RECEIPT_BYTES,
        label="Stair launch readiness receipt",
    )
    if persisted != encoded:
        raise ValueError("Stair launch readiness receipt persisted bytes mismatch")
    return StairGenerationLaunchReadinessReceipt.from_mapping(
        strict_json_object_loads(
            persisted,
            context="Stair launch readiness receipt",
        )
    )


def collect_stair_generation_launch_readiness_v3(
    config: StairGenerationLaunchReadinessConfigV3,
    *,
    inventory_discoverer: InventoryDiscoverer | None = None,
    resource_sampler: ResourceSampler | None = None,
    system_sampler: SystemSampler | None = None,
    source_inspector: SourceInspector | None = None,
    formal_subject_inspector: FormalLaunchSubjectInspector | None = None,
    monotonic_ns: Callable[[], int] = time.monotonic_ns,
    utc_now: Callable[[], str] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    hostname_reader: Callable[[], str] = socket.gethostname,
    environ: Mapping[str, str] | None = None,
) -> StairGenerationLaunchReadinessReceipt:
    """Collect one P6-D readiness receipt without network or formal writes."""

    if not isinstance(config, StairGenerationLaunchReadinessConfigV3):
        raise TypeError("Stair launch readiness v3 config is invalid")
    return _collect_stair_generation_launch_readiness_v3(
        config.base,
        v3_config=config,
        inventory_discoverer=inventory_discoverer,
        resource_sampler=resource_sampler,
        system_sampler=system_sampler,
        source_inspector=source_inspector,
        formal_subject_inspector=formal_subject_inspector,
        monotonic_ns=monotonic_ns,
        utc_now=utc_now,
        sleep=sleep,
        hostname_reader=hostname_reader,
        environ=environ,
    )


def load_stair_generation_launch_readiness_receipt(
    path: Path,
    *,
    expected_formal_output_dir: Path,
    expected_source_root: Path,
    expected_session_id: str,
    expected_source_commit: str,
    expected_machine_id: str,
    expected_gpu_uuids: tuple[str, str],
    expected_lease_root: Path,
    expected_max_output_bytes: int,
    expected_nvidia_smi_executable: Path,
    expected_hostname: str | None = None,
    expected_boot_id: str | None = None,
    boot_id_path: Path = Path("/proc/sys/kernel/random/boot_id"),
    expected_subject_mode: str | None = None,
    expected_subject_modes: tuple[str, ...] | None = None,
    expected_formal_generation_command_sha256: str | None = None,
) -> StairGenerationLaunchReadinessReceipt:
    """Load canonical bytes and require every formal-launch binding to match."""

    target = Path(path).resolve(strict=True)
    payload = read_bounded_regular_file(
        target,
        max_bytes=MAX_READINESS_RECEIPT_BYTES,
        label="Stair launch readiness receipt",
    )
    value = strict_json_object_loads(payload, context="Stair launch readiness receipt")
    if payload != canonical_json_bytes(value, trailing_newline=True):
        raise ValueError("Stair launch readiness receipt is not canonical JSON")
    receipt = StairGenerationLaunchReadinessReceipt.from_mapping(value)
    hostname = socket.gethostname() if expected_hostname is None else expected_hostname
    boot_id = (
        _read_identity(Path(boot_id_path), "boot id", maximum=128)
        if expected_boot_id is None
        else expected_boot_id
    )
    expected_binding = {
        "formal_output_dir": str(
            Path(expected_formal_output_dir).resolve(strict=False)
        ),
        "source_root": str(Path(expected_source_root).resolve(strict=True)),
        "session_id": expected_session_id,
        "source_commit": expected_source_commit,
        "machine_id": expected_machine_id,
        "hostname": hostname,
        "boot_id": boot_id,
        "gpu_uuids": list(expected_gpu_uuids),
        "lease_root": str(Path(expected_lease_root).resolve(strict=False)),
        "max_output_bytes": expected_max_output_bytes,
        "nvidia_smi_executable": str(
            Path(expected_nvidia_smi_executable).resolve(strict=True)
        ),
    }
    if receipt.payload["binding"] != expected_binding:
        raise ValueError("Stair launch readiness receipt binding mismatch")
    schema_version = receipt.payload["schema_version"]
    source_root = Path(expected_source_root).resolve(strict=True)
    if expected_subject_mode is not None and expected_subject_modes is not None:
        raise ValueError("launch readiness subject-mode expectations are ambiguous")
    _assert_current_producer_sources(
        receipt.payload["producer"], source_root=source_root
    )
    if schema_version != STAIR_GENERATION_LAUNCH_READINESS_V3_SCHEMA_VERSION:
        raise ValueError("current launch readiness requires the v3 schema")
    expected_command_sha256 = require_sha256_checksum(
        expected_formal_generation_command_sha256,
        "expected formal generation command sha256",
    )
    subject = receipt.payload["subject"]
    assert isinstance(subject, Mapping)
    allowed_modes = (
        tuple(expected_subject_modes)
        if expected_subject_modes is not None
        else ((expected_subject_mode,) if expected_subject_mode is not None else ())
    )
    if (
        not allowed_modes
        or len(set(allowed_modes)) != len(allowed_modes)
        or any(mode not in STAIR_FORMAL_LAUNCH_SUBJECT_MODES for mode in allowed_modes)
    ):
        raise ValueError("v3 launch readiness requires allowed subject modes")
    current_mode = subject["subject_mode"]
    if current_mode not in allowed_modes:
        raise ValueError("Stair launch readiness subject-mode binding mismatch")
    _assert_current_v3_execution_sources(
        receipt.payload["execution"],
        source_root=source_root,
        formal_generation_command_sha256=expected_command_sha256,
    )
    current_subject = _inspect_v3_launch_subject_values(
        formal_output_dir=Path(expected_formal_output_dir).resolve(strict=False),
        source_root=source_root,
        session_id=expected_session_id,
        expected_source_commit=expected_source_commit,
        subject_mode=str(current_mode),
    )
    current_subject = _validated_v3_subject_mapping(
        current_subject,
        binding=expected_binding,
    )
    if current_subject != subject:
        raise ValueError("Stair launch readiness subject binding drifted")
    return receipt


def _binding_mapping(
    config: StairGenerationLaunchReadinessConfig,
    *,
    machine_id: str,
    hostname: str,
    boot_id: str,
) -> dict[str, Any]:
    return {
        "formal_output_dir": str(config.formal_output_dir),
        "source_root": str(config.source_root),
        "session_id": config.session_id,
        "source_commit": config.expected_source_commit,
        "machine_id": machine_id,
        "hostname": hostname,
        "boot_id": boot_id,
        "gpu_uuids": list(config.expected_gpu_uuids),
        "lease_root": str(config.lease_root),
        "max_output_bytes": config.max_output_bytes,
        "nvidia_smi_executable": str(config.nvidia_smi_executable),
    }


def _provider_credential_readiness(
    config: StairGenerationLaunchReadinessConfigV3,
) -> dict[str, Any]:
    validate_stair_provider_credential(
        env_file=config.provider_env_file,
        api_key_env=STAIR_GENERATION_PROVIDER_API_KEY_ENV,
    )
    return {
        "schema_version": STAIR_PROVIDER_CREDENTIAL_READINESS_SCHEMA_VERSION,
        "api_key_env": STAIR_GENERATION_PROVIDER_API_KEY_ENV,
        "checks": {
            "env_file_regular": True,
            "credential_present": True,
            "credential_http_header_safe": True,
        },
        "privacy": {
            "credential_persisted": False,
            "credential_hash_persisted": False,
            "env_file_contents_persisted": False,
        },
        "side_effects": {
            "network_requests": 0,
            "generation_requests": 0,
        },
    }


def _validate_provider_credential_mapping(value: Any) -> dict[str, Any]:
    credential = _exact_mapping(
        value,
        _PROVIDER_CREDENTIAL_KEYS,
        "readiness v3 provider credential",
    )
    if (
        credential["schema_version"]
        != STAIR_PROVIDER_CREDENTIAL_READINESS_SCHEMA_VERSION
        or credential["api_key_env"] != STAIR_GENERATION_PROVIDER_API_KEY_ENV
    ):
        raise ValueError("readiness v3 provider credential identity mismatch")
    if _exact_mapping(
        credential["checks"],
        {"env_file_regular", "credential_present", "credential_http_header_safe"},
        "readiness v3 provider credential checks",
    ) != {
        "env_file_regular": True,
        "credential_present": True,
        "credential_http_header_safe": True,
    }:
        raise ValueError("readiness v3 provider credential checks did not pass")
    if _exact_mapping(
        credential["privacy"],
        {
            "credential_persisted",
            "credential_hash_persisted",
            "env_file_contents_persisted",
        },
        "readiness v3 provider credential privacy",
    ) != {
        "credential_persisted": False,
        "credential_hash_persisted": False,
        "env_file_contents_persisted": False,
    }:
        raise ValueError("readiness v3 provider credential privacy mismatch")
    if _exact_mapping(
        credential["side_effects"],
        {"network_requests", "generation_requests"},
        "readiness v3 provider credential side effects",
    ) != {"network_requests": 0, "generation_requests": 0}:
        raise ValueError("readiness v3 provider credential side effects mismatch")
    return strict_json_object_loads(
        canonical_json_bytes(credential),
        context="readiness v3 provider credential",
    )


def _formal_required_free_disk_bytes(current_accounted_output_bytes: int) -> int:
    current = _nonnegative_int(
        current_accounted_output_bytes,
        "P6-D current accounted output bytes",
    )
    if current > FORMAL_MAX_OUTPUT_BYTES:
        raise ValueError("P6-D current accounted output exceeds its frozen cap")
    remaining = FORMAL_MAX_OUTPUT_BYTES - current
    safety_reserve = (
        FORMAL_REMAINING_OUTPUT_SAFETY_NUMERATOR * remaining
        + FORMAL_REMAINING_OUTPUT_SAFETY_DENOMINATOR
        - 1
    ) // FORMAL_REMAINING_OUTPUT_SAFETY_DENOMINATOR
    return max(MIN_FREE_DISK_BYTES, safety_reserve + FORMAL_DISK_HEADROOM_BYTES)


def _v3_subject_current_accounted_output_bytes(
    subject: Mapping[str, Any] | None,
) -> int:
    if not isinstance(subject, Mapping):
        raise ValueError("readiness v3 subject is unavailable")
    return _nonnegative_int(
        subject.get("current_accounted_output_bytes"),
        "readiness v3 current accounted output bytes",
    )


def _resolved_formal_task_config(
    *,
    output_dir: Path,
    project_root: Path,
) -> StairClimbingTaskConfig:
    manifest = load_resolved_cell_manifest(
        (output_dir / RESOLVED_CELL_MANIFEST_FILENAME).resolve(),
        expected_task_id=STAIR_TASK_ID,
    )
    defaults = StairClimbingCompositionConfig.defaults(project_root=project_root)
    values = defaults.to_task_defaults()
    values["cell_variant_id"] = manifest.cell_variant_id
    composition = StairClimbingCompositionConfig.from_values(
        values
    ).bind_selected_cell_contract(project_root=project_root)
    return composition.to_task_config(
        project_root=project_root,
        resolved_cell_task_protocol_hash=(
            manifest.resolved_cell_task_protocol_hash
        ),
    )




def _inspect_v3_launch_subject(
    config: StairGenerationLaunchReadinessConfigV3,
) -> Mapping[str, Any]:
    return _inspect_v3_launch_subject_values(
        formal_output_dir=config.base.formal_output_dir,
        source_root=config.base.source_root,
        session_id=config.base.session_id,
        expected_source_commit=config.base.expected_source_commit,
        subject_mode=config.subject_mode,
    )


def _inspect_v3_launch_subject_values(
    *,
    formal_output_dir: Path,
    source_root: Path,
    session_id: str,
    expected_source_commit: str,
    subject_mode: str,
) -> dict[str, Any]:
    output_dir = Path(formal_output_dir).resolve(strict=False)
    project_root = Path(source_root).resolve(strict=True)
    if subject_mode == STAIR_FORMAL_LAUNCH_SUBJECT_FRESH_ABSENT:
        _assert_path_absent(output_dir, "formal output")
        return {
            "subject_mode": STAIR_FORMAL_LAUNCH_SUBJECT_FRESH_ABSENT,
            "formal_output_absent": True,
            "current_accounted_output_bytes": 0,
        }
    if subject_mode not in (
        STAIR_FORMAL_LAUNCH_SUBJECT_NONTERMINAL_RESUME,
        STAIR_FORMAL_LAUNCH_SUBJECT_TERMINAL_FINALIZATION,
        STAIR_FORMAL_LAUNCH_SUBJECT_TERMINAL_AUDIT_ONLY,
    ):
        raise ValueError("Stair launch readiness v3 subject_mode is invalid")
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise ValueError(
            "formal resume readiness requires a real formal output directory"
        )

    source_path = output_dir / STAIR_GENERATION_SOURCE_CHECKOUT_FILENAME
    source_evidence, source_payload = _canonical_object_file(
        source_path,
        label="Stair generation source checkout identity",
        max_bytes=4096,
    )
    source_mapping = _exact_mapping(
        source_payload,
        {"schema_version", "source_commit", "source_checkout_hash"},
        "Stair generation source checkout identity",
    )
    source_body = {
        "schema_version": STAIR_GENERATION_SOURCE_CHECKOUT_SCHEMA_VERSION,
        "source_commit": expected_source_commit,
    }
    if (
        source_mapping["schema_version"]
        != STAIR_GENERATION_SOURCE_CHECKOUT_SCHEMA_VERSION
        or source_mapping["source_commit"] != expected_source_commit
        or source_mapping["source_checkout_hash"]
        != canonical_json_hash(source_body)
    ):
        raise ValueError("formal resume source checkout identity mismatch")

    contract_path = output_dir / GENERATION_SESSION_CONTRACT_FILENAME
    contract_evidence, contract_payload = _canonical_object_file(
        contract_path,
        label="generation session contract",
        max_bytes=MAX_READINESS_RECEIPT_BYTES,
    )
    contract = GenerationSessionContract.from_mapping(contract_payload)
    if (
        contract.session_id != session_id
        or contract.session_mode != FORMAL_GENERATION_SESSION_MODE
        or contract.max_generations != STAIR_FINAL_CONFIRMATION_REQUIRED_GENERATIONS
        or contract.task_id != STAIR_TASK_ID
        or contract.backend_id != "isaac_lab_rsl_rl"
    ):
        raise ValueError("formal resume generation contract identity mismatch")

    policy = load_generation_continuation_policy(output_dir)
    if (
        policy.schema_version != GENERATION_CONTINUATION_POLICY_V2_SCHEMA_VERSION
        or policy.max_segments != 7
        or policy.operator_pause_generation_boundaries
        != STAIR_FORMAL_GENERATION_PAUSE_BOUNDARIES
        or policy.policy_hash
        != contract.bound_contract_hashes["continuation_policy"]
    ):
        raise ValueError("formal resume continuation policy mismatch")
    policy_path = output_dir / GENERATION_CONTINUATION_POLICY_FILENAME
    policy_evidence = _authoritative_file_evidence(policy_path)

    task_config = _resolved_formal_task_config(
        output_dir=output_dir,
        project_root=project_root,
    )
    plan_path = (
        output_dir
        / STAIR_FINAL_CONFIRMATION_DIRNAME
        / STAIR_FINAL_CONFIRMATION_PLAN_FILENAME
    )
    plan = load_stair_generation_final_confirmation_plan(
        plan_path,
        expected_session_id=session_id,
        expected_config=task_config,
    )
    if (
        plan.plan_hash is None
        or plan.plan_hash
        != contract.bound_contract_hashes["final_confirmation_plan"]
    ):
        raise ValueError("formal resume final confirmation plan binding mismatch")
    plan_evidence = _authoritative_file_evidence(plan_path)

    cost_contract_path = output_dir / STAIR_GENERATION_SESSION_COST_FILENAME
    cost_contract = load_stair_generation_session_cost_contract(cost_contract_path)
    if (
        not isinstance(cost_contract, StairGenerationSessionCostContractV3)
        or cost_contract.contract_hash
        != contract.bound_contract_hashes["session_cost"]
        or cost_contract.max_output_bytes != FORMAL_MAX_OUTPUT_BYTES
    ):
        raise ValueError("formal resume cost contract binding mismatch")
    cost_contract_evidence = _authoritative_file_evidence(cost_contract_path)
    cost_chain = StairGenerationActualCostJournal(
        output_dir,
        contract=cost_contract,
    ).load_chain()
    cost_receipt_files = [
        {
            **_authoritative_file_evidence(item.path),
            "payload_hash": item.payload_hash,
        }
        for item in cost_chain
    ]
    current_accounted_output_bytes = 0
    if cost_chain:
        actual = cost_chain[-1].payload.get("actual")
        if not isinstance(actual, Mapping):
            raise ValueError("formal resume cost head has no actual usage")
        current_accounted_output_bytes = _nonnegative_int(
            actual.get("accounted_output_bytes"),
            "formal resume current accounted output bytes",
        )
        checks = cost_chain[-1].payload.get("cap_checks")
        if not isinstance(checks, Mapping) or checks.get("all_passed") is not True:
            raise ValueError("formal resume cost head exhausted a frozen cap")
    if current_accounted_output_bytes > FORMAL_MAX_OUTPUT_BYTES:
        raise ValueError("formal resume accounted output exceeds its frozen cap")

    genesis_path = output_dir / SESSION_GENESIS_FILENAME
    genesis_file = _optional_file_evidence(genesis_path)
    genesis: SessionGenesisArtifact | None = None
    if genesis_file["exists"] is True:
        _, genesis_payload = _canonical_object_file(
            genesis_path,
            label="session Genesis",
            max_bytes=MAX_READINESS_RECEIPT_BYTES,
        )
        genesis = SessionGenesisArtifact.from_mapping(genesis_payload)
        if genesis.session_id != session_id or genesis.session_genesis_hash is None:
            raise ValueError("formal resume session Genesis identity mismatch")
        contract_ref = genesis.generation_session_contract_ref
        if contract_ref is None:
            raise ValueError("formal resume session Genesis has no generation contract")
        if load_generation_session_contract(output_dir, contract_ref) != contract:
            raise ValueError("formal resume Genesis generation contract differs")

    ledger_path = output_dir / "ledger.jsonl"
    ledger_records = load_ledger(ledger_path)
    ledger_evidence = _optional_file_evidence(ledger_path)
    ledger_height = len(ledger_records)
    if ledger_height and ledger_evidence["exists"] is not True:
        raise ValueError("formal resume non-empty ledger file is absent")

    if genesis is None:
        if subject_mode != STAIR_FORMAL_LAUNCH_SUBJECT_NONTERMINAL_RESUME:
            raise ValueError("formal terminal readiness requires Session Genesis")
        if ledger_evidence["exists"] is True or ledger_height:
            raise ValueError("pre-Genesis formal resume cannot have a ledger")
        lifecycle_files = _lifecycle_receipt_file_evidence(output_dir)
        if lifecycle_files:
            raise ValueError("pre-Genesis formal resume cannot have lifecycle receipts")
        lifecycle_mapping = {
            "initialized": False,
            "head_hash": None,
            "active": False,
            "active_segment_id": None,
            "segment_count": 0,
            "closed_generation_count": 0,
            "terminal": False,
            "continuation_eligible": True,
            "cost_chain_head_hash": None,
            "receipt_files": [],
        }
    else:
        identity = GenerationLifecycleIdentity(
            session_id=session_id,
            session_genesis_hash=genesis.session_genesis_hash,
            generation_session_contract_hash=contract.contract_hash,
            final_confirmation_plan_hash=plan.plan_hash,
            ledger_epoch_hash=primary_ledger_epoch_hash(
                genesis.session_genesis_hash
            ),
            output_root=str(output_dir),
        )
        lifecycle = audit_generation_lifecycle(
            output_dir=output_dir,
            identity=identity,
            policy=policy,
            require_closed=False,
            require_recovery_attempt_results=False,
        )
        if not cost_chain:
            raise ValueError("post-Genesis formal resume requires an actual-cost chain")
        if lifecycle.terminal:
            replay_stair_generation_formal_cost_chain(
                contract=cost_contract,
                actual_cost_receipts=cost_chain,
                lifecycle=lifecycle,
                require_finalization_settlement=(
                    subject_mode
                    == STAIR_FORMAL_LAUNCH_SUBJECT_TERMINAL_AUDIT_ONLY
                ),
            )
        else:
            replay_stair_generation_combined_cost_chain(
                contract=cost_contract,
                actual_cost_receipts=cost_chain,
                lifecycle=lifecycle,
            )
        lifecycle_start_height = lifecycle.closed_generation_count
        if lifecycle.active_segment_id is None:
            if ledger_height != lifecycle_start_height:
                raise ValueError(
                    "formal resume ledger height and lifecycle height differ"
                )
        elif not (
            lifecycle_start_height
            <= ledger_height
            <= STAIR_FINAL_CONFIRMATION_REQUIRED_GENERATIONS
        ):
            raise ValueError(
                "active formal resume ledger height is outside its lifecycle budget"
            )
        lifecycle_files = _lifecycle_receipt_file_evidence(output_dir)
        if [item["payload_hash"] for item in lifecycle_files] != sorted(
            lifecycle.receipt_hashes
        ):
            raise ValueError("formal resume lifecycle receipt file set mismatch")
        lifecycle_mapping = {
            "initialized": True,
            "head_hash": lifecycle.lifecycle_head_hash,
            "active": lifecycle.active_segment_id is not None,
            "active_segment_id": lifecycle.active_segment_id,
            "segment_count": lifecycle.segment_count,
            "closed_generation_count": lifecycle.closed_generation_count,
            "terminal": lifecycle.terminal,
            "continuation_eligible": lifecycle.continuation_eligible,
            "cost_chain_head_hash": lifecycle.cost_chain_head_hash,
            "receipt_files": lifecycle_files,
        }

    reservation_path = (
        output_dir
        / STAIR_FINAL_CONFIRMATION_DIRNAME
        / STAIR_FINAL_CONFIRMATION_RESERVATION_FILENAME
    )
    reservation_file = _optional_file_evidence(reservation_path)
    reservation = None
    if reservation_file["exists"] is True:
        reservation = load_stair_generation_final_confirmation_reservation(
            reservation_path,
            expected_plan=plan,
        )
    result_path = (
        output_dir
        / STAIR_FINAL_CONFIRMATION_DIRNAME
        / STAIR_FINAL_CONFIRMATION_RESULT_FILENAME
    )
    result_file = _optional_file_evidence(result_path)
    result = None
    if result_file["exists"] is True:
        if reservation is None:
            raise ValueError("formal final result exists without its reservation")
        objective_family_mask = None
        if task_config.is_unified_protocol:
            if genesis is None:
                raise ValueError(
                    "unified formal final result exists without Session Genesis"
                )
            objective_family_mask = objective_family_mask_from_public_result(
                suite=task_config.ordered_evaluation_suite,
                public_result=getattr(genesis.anchor, "public_result", None),
            )
        result = load_stair_generation_final_confirmation_result(
            result_path,
            plan=plan,
            reservation=reservation,
            config=task_config,
            objective_family_mask=objective_family_mask,
        )
    report_path = output_dir / STAIR_GENERATION_REPORT_FILENAME
    report_file = _optional_file_evidence(report_path)

    terminal = bool(lifecycle_mapping["terminal"])
    continuation_eligible = bool(lifecycle_mapping["continuation_eligible"])
    active = bool(lifecycle_mapping["active"])
    if subject_mode == STAIR_FORMAL_LAUNCH_SUBJECT_NONTERMINAL_RESUME:
        if terminal or (not active and not continuation_eligible):
            raise ValueError("formal nonterminal launch subject is not continuable")
        if (
            reservation is not None
            or result is not None
            or report_file["exists"] is True
        ):
            raise ValueError("formal nonterminal launch has terminal artifacts")
    else:
        if (
            not lifecycle_mapping["initialized"]
            or not terminal
            or active
            or continuation_eligible
            or lifecycle_mapping["closed_generation_count"]
            != STAIR_FINAL_CONFIRMATION_REQUIRED_GENERATIONS
            or ledger_height != STAIR_FINAL_CONFIRMATION_REQUIRED_GENERATIONS
        ):
            raise ValueError("formal terminal launch subject is not terminal")
        if subject_mode == STAIR_FORMAL_LAUNCH_SUBJECT_TERMINAL_FINALIZATION:
            if report_file["exists"] is True:
                raise ValueError(
                    "terminal finalization readiness requires the report to be absent"
                )
        elif (
            report_file["exists"] is not True
            or reservation is None
            or result is None
        ):
            raise ValueError(
                "terminal audit-only readiness requires completed final artifacts"
            )

    attempt_state = _formal_attempt_state_evidence(output_dir)
    body = {
        "subject_mode": subject_mode,
        "source_identity": {
            "file": source_evidence,
            "source_commit": expected_source_commit,
            "source_checkout_hash": source_mapping["source_checkout_hash"],
        },
        "session_genesis": {
            "file": genesis_file,
            "session_genesis_hash": (
                genesis.session_genesis_hash if genesis is not None else None
            ),
        },
        "generation_session_contract": {
            "file": contract_evidence,
            "contract_hash": contract.contract_hash,
        },
        "continuation_policy": {
            "file": policy_evidence,
            "policy_hash": policy.policy_hash,
        },
        "ledger": {
            "file": ledger_evidence,
            "height": ledger_height,
            "snapshot_hash": canonical_json_hash(ledger_records),
        },
        "lifecycle": lifecycle_mapping,
        "cost": {
            "contract": {
                "file": cost_contract_evidence,
                "contract_hash": cost_contract.contract_hash,
            },
            "receipt_count": len(cost_chain),
            "head": cost_receipt_files[-1] if cost_receipt_files else None,
            "receipt_files": cost_receipt_files,
        },
        "attempt_state": attempt_state,
        "final_confirmation": {
            "plan": {"file": plan_evidence, "plan_hash": plan.plan_hash},
            "reservation": {
                "file": reservation_file,
                "reservation_hash": (
                    reservation.reservation_hash if reservation is not None else None
                ),
            },
            "result": {
                "file": result_file,
                "result_hash": result.result_hash if result is not None else None,
            },
        },
        "formal_report": report_file,
        "current_accounted_output_bytes": current_accounted_output_bytes,
    }
    return {**body, "snapshot_hash": canonical_json_hash(body)}


def _formal_attempt_state_evidence(output_dir: Path) -> dict[str, Any]:
    roots = tuple(
        output_dir / name
        for name in ("job_attempts", "live_invocations", "authoring")
    )
    files: list[dict[str, Any]] = []
    for root in roots:
        if not root.exists():
            continue
        if root.is_symlink() or not root.is_dir():
            raise ValueError("formal attempt-state root is invalid")
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise ValueError("formal attempt-state path must not be a symlink")
            if path.is_file() and path.suffix in {".json", ".jsonl"}:
                files.append(_file_evidence(path))
    files.sort(key=lambda item: item["path"])
    body = {
        "file_count": len(files),
        "total_size_bytes": sum(item["size_bytes"] for item in files),
        "files": files,
    }
    return {**body, "snapshot_hash": canonical_json_hash(body)}


def _canonical_object_file(
    path: Path,
    *,
    label: str,
    max_bytes: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    evidence = _authoritative_file_evidence(path)
    payload = read_bounded_regular_file(path, max_bytes=max_bytes, label=label)
    value = strict_json_object_loads(payload, context=label)
    if payload != canonical_json_bytes(value, trailing_newline=True):
        raise ValueError(f"{label} is not canonical JSON")
    return evidence, value


def _authoritative_file_evidence(path: Path) -> dict[str, Any]:
    unresolved = Path(path)
    if unresolved.is_symlink():
        raise ValueError("formal-state evidence file must not be a symlink")
    return _file_evidence(unresolved)


def _optional_file_evidence(path: Path) -> dict[str, Any]:
    unresolved = Path(path)
    if unresolved.is_symlink():
        raise ValueError("formal-state optional evidence must not be a symlink")
    absolute = unresolved.resolve(strict=False)
    if not unresolved.exists():
        return {
            "path": str(absolute),
            "exists": False,
            "sha256": None,
            "size_bytes": 0,
        }
    evidence = _authoritative_file_evidence(unresolved)
    return {
        "path": evidence["path"],
        "exists": True,
        **{key: evidence[key] for key in ("sha256", "size_bytes")},
    }


def _lifecycle_receipt_file_evidence(output_dir: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    paths: list[Path] = []
    for dirname in ("generation_segments", "generation_recoveries"):
        root = output_dir / dirname
        if not root.exists():
            continue
        if root.is_symlink() or not root.is_dir():
            raise ValueError("generation lifecycle receipt root is invalid")
        paths.extend(root.rglob("*.json"))
    for path in sorted(paths, key=lambda item: str(item.resolve(strict=False))):
        evidence, payload = _canonical_object_file(
            path,
            label="generation lifecycle receipt",
            max_bytes=MAX_READINESS_RECEIPT_BYTES,
        )
        result.append({**evidence, "payload_hash": canonical_json_hash(payload)})
    return sorted(result, key=lambda item: item["payload_hash"])


def _run_dual_lease_cycle(
    *,
    manager: GpuLeaseManager,
    config: StairGenerationLaunchReadinessConfig,
    inventory: tuple[NvidiaGpuInventoryEntry, NvidiaGpuInventoryEntry],
    cycle: str,
    capture: Callable[[str], dict[str, Any]],
) -> dict[str, Any]:
    binding_hash = canonical_json_hash(
        {
            "schema_version": "true_rl_agent.stair.generation_launch_readiness_binding.v1",
            "session_id": config.session_id,
            "source_commit": config.expected_source_commit,
            "formal_output_dir": str(config.formal_output_dir),
            "gpu_uuids": list(config.expected_gpu_uuids),
            "cycle": cycle,
        }
    )
    leases = []
    with ExitStack() as held:
        for lane_id, item in zip(("lane-a", "lane-b"), inventory, strict=True):
            holder = PreGenesisGpuLeaseHolder(
                session_id=config.session_id,
                genesis_identity_hash=binding_hash,
                session_contract_hash=binding_hash,
                lane_id=lane_id,
                invocation_segment_id="launch-readiness",
                logical_job_id=f"launch-readiness-{cycle}-{lane_id}",
            )
            lease = held.enter_context(
                manager.acquire(item.execution_device, holder=holder)
            )
            lease.assert_owned()
            leases.append(lease)
        held_sample = capture(f"{cycle}-leases-held")
        resource_binding_hashes = [lease.resource_binding_hash for lease in leases]
    return {
        "cycle": cycle,
        "lane_ids": ["lane-a", "lane-b"],
        "execution_devices": [item.execution_device for item in inventory],
        "gpu_uuids": [item.uuid for item in inventory],
        "resource_binding_hashes": resource_binding_hashes,
        "physical_resource_hashes": [item.physical_resource_hash for item in inventory],
        "held_sample_sequence": held_sample["sequence"],
        "simultaneously_held": True,
        "released": True,
    }


def _discover_inventory(
    config: StairGenerationLaunchReadinessConfig,
) -> Sequence[NvidiaGpuInventoryEntry]:
    return discover_nvidia_gpu_inventory(
        nvidia_smi=str(config.nvidia_smi_executable),
        machine_id_path=config.machine_id_path,
    )


def _sample_gpu_resources(
    inventory: Sequence[NvidiaGpuInventoryEntry],
    config: StairGenerationLaunchReadinessConfig,
) -> Mapping[str, Any]:
    return sample_resource_snapshot(
        inventory=inventory,
        nvidia_smi=str(config.nvidia_smi_executable),
    )


def _sample_system_resources(
    config: StairGenerationLaunchReadinessConfig,
) -> Mapping[str, Any]:
    try:
        text = config.proc_meminfo_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise RuntimeError("failed to read host memory information") from exc
    fields: dict[str, int] = {}
    for line in text.splitlines():
        key, separator, remainder = line.partition(":")
        if not separator:
            continue
        token = remainder.strip().split(maxsplit=1)[0]
        try:
            fields[key] = int(token)
        except ValueError:
            continue
    required = ("MemAvailable", "SwapTotal", "SwapFree")
    if any(key not in fields for key in required):
        raise ValueError("host memory information is incomplete")
    if fields["SwapFree"] > fields["SwapTotal"]:
        raise ValueError("host swap information is invalid")
    disk = shutil.disk_usage(config.formal_output_dir.parent)
    return {
        "memory_available_kib": fields["MemAvailable"],
        "swap_used_kib": fields["SwapTotal"] - fields["SwapFree"],
        "free_disk_bytes": disk.free,
    }


def _inspect_source(source_root: Path) -> Mapping[str, Any]:
    state = inspect_stair_source_checkout(source_root)
    return {"head_commit": state.head_commit, "tracked_clean": state.clean}


def _validated_source_state(
    value: Any,
    *,
    expected_commit: str,
) -> dict[str, Any]:
    mapping = _exact_mapping(
        value,
        {"head_commit", "tracked_clean"},
        "Stair launch readiness source state",
    )
    if mapping["head_commit"] != expected_commit:
        raise ValueError("Stair launch readiness source HEAD mismatch")
    if mapping["tracked_clean"] is not True:
        raise ValueError("Stair launch readiness tracked source is dirty")
    return {"head_commit": expected_commit, "tracked_clean": True}


def _validated_inventory(
    value: Any,
    *,
    expected_machine_id: str,
    expected_gpu_uuids: tuple[str, str],
) -> tuple[NvidiaGpuInventoryEntry, NvidiaGpuInventoryEntry]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise TypeError("Stair launch readiness GPU inventory must be a sequence")
    entries = tuple(value)
    if (
        len(entries) != 2
        or any(not isinstance(item, NvidiaGpuInventoryEntry) for item in entries)
        or tuple(item.ordinal for item in entries) != (0, 1)
        or tuple(item.uuid for item in entries) != expected_gpu_uuids
        or any(item.machine_id != expected_machine_id for item in entries)
        or len({item.physical_resource_hash for item in entries}) != 2
    ):
        raise ValueError("Stair launch readiness GPU inventory binding mismatch")
    return entries  # type: ignore[return-value]


def _validated_sample_inputs(
    *,
    sequence: int,
    phase: str,
    captured_at_utc: Any,
    monotonic_ns: Any,
    gpu_snapshot: Any,
    system_snapshot: Any,
    expected_gpu_uuids: tuple[str, str],
) -> dict[str, Any]:
    if not isinstance(gpu_snapshot, Mapping):
        raise ValueError("Stair launch readiness GPU snapshot must be an object")
    raw_gpus = gpu_snapshot.get("gpus")
    raw_apps = gpu_snapshot.get("compute_apps")
    if not isinstance(raw_gpus, list) or not isinstance(raw_apps, list):
        raise ValueError("Stair launch readiness GPU snapshot fields are incomplete")
    observed_uuids: list[str] = []
    for row in raw_gpus:
        if not isinstance(row, Mapping) or not isinstance(row.get("uuid"), str):
            raise ValueError("Stair launch readiness GPU snapshot row is invalid")
        observed_uuids.append(row["uuid"])
    if tuple(observed_uuids) != expected_gpu_uuids:
        raise ValueError("Stair launch readiness GPU snapshot inventory mismatch")
    target_compute_app_count = 0
    for app in raw_apps:
        if not isinstance(app, Mapping):
            raise ValueError("Stair launch readiness compute-app row is invalid")
        gpu_uuid = app.get("gpu_uuid")
        if not isinstance(gpu_uuid, str):
            raise ValueError("Stair launch readiness compute-app GPU UUID is invalid")
        if gpu_uuid in expected_gpu_uuids:
            target_compute_app_count += 1
    system = _exact_mapping(
        system_snapshot,
        {"memory_available_kib", "swap_used_kib", "free_disk_bytes"},
        "Stair launch readiness system snapshot",
    )
    for key in ("memory_available_kib", "swap_used_kib", "free_disk_bytes"):
        _nonnegative_int(system[key], f"Stair launch readiness {key}")
    _nonnegative_int(monotonic_ns, "Stair launch readiness monotonic_ns")
    return {
        "sequence": sequence,
        "phase": phase,
        "captured_at_utc": _bounded_text(
            captured_at_utc,
            "Stair launch readiness captured_at_utc",
            maximum=128,
        ),
        "monotonic_ns": monotonic_ns,
        "gpu_uuids": list(expected_gpu_uuids),
        "target_compute_app_count": target_compute_app_count,
        "memory_available_kib": system["memory_available_kib"],
        "swap_used_kib": system["swap_used_kib"],
        "free_disk_bytes": system["free_disk_bytes"],
    }


def _validate_receipt_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("Stair launch readiness receipt must be an object")
    schema_version = value.get("schema_version")
    if schema_version == STAIR_GENERATION_LAUNCH_READINESS_V3_SCHEMA_VERSION:
        return _validate_receipt_mapping_v3(value)
    raise ValueError("Stair launch readiness receipt schema mismatch")




def _validate_receipt_mapping_v3(value: Any) -> dict[str, Any]:
    root = _exact_mapping(value, _V3_ROOT_KEYS, "Stair launch readiness v3 receipt")
    if root["schema_version"] != STAIR_GENERATION_LAUNCH_READINESS_V3_SCHEMA_VERSION:
        raise ValueError("Stair launch readiness v3 receipt schema mismatch")
    if root["status"] != "pass":
        raise ValueError("Stair launch readiness v3 receipt status must be pass")
    binding = _exact_mapping(root["binding"], _BINDING_KEYS, "readiness binding")
    if binding.get("max_output_bytes") != FORMAL_MAX_OUTPUT_BYTES:
        raise ValueError("readiness v3 output cap binding mismatch")
    execution = _validate_v3_execution_mapping(root["execution"], binding=binding)
    prelaunch_evidence = _validate_prelaunch_evidence_mapping(
        root["prelaunch_evidence"],
        binding=binding,
    )
    credential = _validate_provider_credential_mapping(root["provider_credential"])
    subject = _validated_v3_subject_mapping(root["subject"], binding=binding)
    thresholds = _exact_mapping(
        root["thresholds"],
        _V3_THRESHOLD_KEYS,
        "readiness v3 thresholds",
    )
    current_accounted = _v3_subject_current_accounted_output_bytes(subject)
    remaining = FORMAL_MAX_OUTPUT_BYTES - current_accounted
    expected_required_disk = _formal_required_free_disk_bytes(current_accounted)
    if (
        thresholds["remaining_output_safety_multiplier"]
        != FORMAL_REMAINING_OUTPUT_SAFETY_MULTIPLIER
        or thresholds["fixed_disk_headroom_bytes"] != FORMAL_DISK_HEADROOM_BYTES
        or thresholds["current_accounted_output_bytes"] != current_accounted
        or thresholds["remaining_worst_case_output_bytes"] != remaining
        or thresholds["required_free_disk_bytes"] != expected_required_disk
    ):
        raise ValueError("readiness v3 remaining-output disk formula mismatch")
    observations = _exact_mapping(
        root["observations"], _OBSERVATION_KEYS, "readiness observations"
    )
    checks = _exact_mapping(root["checks"], _V3_CHECK_KEYS, "readiness v3 checks")
    if any(checks[key] is not True for key in _V3_CHECK_KEYS):
        raise ValueError("readiness v3 checks must all pass")
    authority = _exact_mapping(
        root["authority"], _FORMAL_AUTHORITY_KEYS, "readiness v3 authority"
    )
    expected_authority = {
        "operational_only": True,
        "provider_readiness_requests": 0,
        "generation_provider_requests": 0,
        "gpu_workloads": 0,
        "formal_output_writes": 0,
        "ledger_writes": 0,
        "scientific_claim_authority": False,
        "acceptance_authority": False,
        "ledger_authority": False,
    }
    if authority != expected_authority:
        raise ValueError("readiness v3 authority projection is invalid")
    output_absent = (
        subject["subject_mode"] == STAIR_FORMAL_LAUNCH_SUBJECT_FRESH_ABSENT
    )
    if (
        observations["formal_output_absent_before"] is not output_absent
        or observations["formal_output_absent_after"] is not output_absent
    ):
        raise ValueError("readiness v3 formal-output subject evidence mismatch")

    synthetic = dict(root)
    for key in (
        "execution",
        "prelaunch_evidence",
        "provider_credential",
        "subject",
    ):
        del synthetic[key]
    synthetic["schema_version"] = STAIR_GENERATION_LAUNCH_READINESS_SCHEMA_VERSION
    synthetic["thresholds"] = {
        "sample_duration_seconds": thresholds["sample_duration_seconds"],
        "sample_interval_seconds": thresholds["sample_interval_seconds"],
        "maximum_sample_gap_nanoseconds": thresholds[
            "maximum_sample_gap_nanoseconds"
        ],
        "minimum_available_ram_kib": thresholds["minimum_available_ram_kib"],
        "maximum_swap_growth_kib": thresholds["maximum_swap_growth_kib"],
        "minimum_free_disk_bytes": thresholds["minimum_free_disk_bytes"],
        "output_disk_multiplier": OUTPUT_DISK_MULTIPLIER,
        "required_free_disk_bytes": expected_required_disk,
    }
    synthetic["observations"] = {
        **dict(observations),
        "formal_output_absent_before": True,
        "formal_output_absent_after": True,
    }
    synthetic["checks"] = {key: True for key in sorted(_CHECK_KEYS)}
    synthetic["authority"] = {
        "operational_only": True,
        "provider_requests": 0,
        "gpu_workloads": 0,
        "ledger_writes": 0,
        "scientific_claim_authority": False,
        "acceptance_authority": False,
        "ledger_authority": False,
    }
    hash_payload = dict(synthetic)
    del hash_payload["receipt_hash"]
    synthetic["receipt_hash"] = canonical_json_hash(hash_payload)
    _validate_receipt_mapping_v1(
        synthetic,
        expected_required_free_disk_bytes=expected_required_disk,
    )

    declared_hash = require_sha256_digest(
        root["receipt_hash"], "Stair launch readiness v3 receipt_hash"
    )
    actual_hash_payload = dict(root)
    del actual_hash_payload["receipt_hash"]
    if declared_hash != canonical_json_hash(actual_hash_payload):
        raise ValueError("Stair launch readiness v3 receipt hash mismatch")
    root["execution"] = execution
    root["prelaunch_evidence"] = prelaunch_evidence
    root["provider_credential"] = credential
    root["subject"] = subject
    return strict_json_object_loads(
        canonical_json_bytes(root), context="Stair launch readiness v3 receipt"
    )


def _validate_receipt_mapping_v1(
    value: Any,
    *,
    expected_required_free_disk_bytes: int | None = None,
) -> dict[str, Any]:
    root = _exact_mapping(value, _ROOT_KEYS, "Stair launch readiness receipt")
    if root["schema_version"] != STAIR_GENERATION_LAUNCH_READINESS_SCHEMA_VERSION:
        raise ValueError("Stair launch readiness receipt schema mismatch")
    if root["status"] != "pass":
        raise ValueError("Stair launch readiness receipt status must be pass")
    binding = _exact_mapping(root["binding"], _BINDING_KEYS, "readiness binding")
    source = _exact_mapping(root["source"], _SOURCE_KEYS, "readiness source")
    producer = _validate_producer_mapping(root["producer"], binding=binding)
    environment = _exact_mapping(
        root["environment"], _ENVIRONMENT_KEYS, "readiness environment"
    )
    thresholds = _exact_mapping(
        root["thresholds"], _THRESHOLD_KEYS, "readiness thresholds"
    )
    observations = _exact_mapping(
        root["observations"], _OBSERVATION_KEYS, "readiness observations"
    )
    checks = _exact_mapping(root["checks"], _CHECK_KEYS, "readiness checks")
    authority = _exact_mapping(
        root["authority"], _AUTHORITY_KEYS, "readiness authority"
    )

    for key in (
        "formal_output_dir",
        "source_root",
        "lease_root",
        "nvidia_smi_executable",
    ):
        if not isinstance(binding[key], str) or not Path(binding[key]).is_absolute():
            raise ValueError(f"readiness binding {key} must be an absolute path")
    _bounded_text(binding["session_id"], "readiness binding session_id")
    if _COMMIT_PATTERN.fullmatch(str(binding["source_commit"])) is None:
        raise ValueError("readiness binding source_commit is invalid")
    _bounded_text(binding["machine_id"], "readiness binding machine_id", maximum=256)
    _bounded_text(binding["hostname"], "readiness binding hostname", maximum=256)
    if (
        not isinstance(binding["boot_id"], str)
        or _BOOT_ID_PATTERN.fullmatch(binding["boot_id"]) is None
    ):
        raise ValueError("readiness binding boot_id is invalid")
    gpu_uuids = binding["gpu_uuids"]
    if (
        not isinstance(gpu_uuids, list)
        or len(gpu_uuids) != 2
        or any(
            not isinstance(item, str) or _GPU_UUID_PATTERN.fullmatch(item) is None
            for item in gpu_uuids
        )
        or gpu_uuids[0] == gpu_uuids[1]
    ):
        raise ValueError("readiness binding GPU UUIDs are invalid")
    _positive_int(binding["max_output_bytes"], "readiness binding max_output_bytes")
    if (
        source["head_commit"] != binding["source_commit"]
        or source["tracked_clean_before"] is not True
        or source["tracked_clean_after"] is not True
    ):
        raise ValueError("readiness source evidence is invalid")
    if any(environment[key] is not False for key in _ENVIRONMENT_KEYS):
        raise ValueError("readiness environment remap evidence is invalid")

    duration = _finite_number(
        thresholds["sample_duration_seconds"], "readiness sample duration"
    )
    interval = _finite_number(
        thresholds["sample_interval_seconds"], "readiness sample interval"
    )
    if duration < DEFAULT_SAMPLE_DURATION_SECONDS or interval <= 0.0 or interval > 1.0:
        raise ValueError("readiness sampling thresholds are invalid")
    expected_maximum_sample_gap_ns = MAX_SAMPLE_GAP_MULTIPLIER * round(
        interval * 1_000_000_000
    )
    if thresholds["maximum_sample_gap_nanoseconds"] != expected_maximum_sample_gap_ns:
        raise ValueError("readiness sampling-gap threshold mismatch")
    if thresholds["minimum_available_ram_kib"] != MIN_AVAILABLE_RAM_KIB:
        raise ValueError("readiness RAM threshold mismatch")
    if thresholds["maximum_swap_growth_kib"] != MAX_SWAP_GROWTH_KIB:
        raise ValueError("readiness swap threshold mismatch")
    if thresholds["minimum_free_disk_bytes"] != MIN_FREE_DISK_BYTES:
        raise ValueError("readiness disk threshold mismatch")
    if thresholds["output_disk_multiplier"] != OUTPUT_DISK_MULTIPLIER:
        raise ValueError("readiness output disk multiplier mismatch")
    expected_required_disk = (
        max(
            MIN_FREE_DISK_BYTES,
            OUTPUT_DISK_MULTIPLIER * binding["max_output_bytes"],
        )
        if expected_required_free_disk_bytes is None
        else _positive_int(
            expected_required_free_disk_bytes,
            "readiness expected required disk bytes",
        )
    )
    if thresholds["required_free_disk_bytes"] != expected_required_disk:
        raise ValueError("readiness required disk bytes mismatch")

    raw_inventory = root["gpu_inventory"]
    if not isinstance(raw_inventory, list):
        raise ValueError("readiness GPU inventory must be a list")
    inventory = tuple(
        NvidiaGpuInventoryEntry.from_mapping(item) for item in raw_inventory
    )
    _validated_inventory(
        inventory,
        expected_machine_id=binding["machine_id"],
        expected_gpu_uuids=tuple(gpu_uuids),  # type: ignore[arg-type]
    )

    samples = observations["samples"]
    if not isinstance(samples, list) or not samples:
        raise ValueError("readiness samples must be a non-empty list")
    previous_monotonic_ns: int | None = None
    previous_captured_at: datetime | None = None
    for index, sample_value in enumerate(samples):
        sample = _exact_mapping(sample_value, _SAMPLE_KEYS, "readiness sample")
        if sample["sequence"] != index:
            raise ValueError("readiness sample sequence mismatch")
        _bounded_text(sample["phase"], "readiness sample phase", maximum=64)
        captured_at = _parse_utc_timestamp(
            sample["captured_at_utc"], "readiness sample captured_at_utc", maximum=128
        )
        sample_monotonic_ns = _nonnegative_int(
            sample["monotonic_ns"], "readiness sample monotonic_ns"
        )
        if (
            previous_monotonic_ns is not None
            and sample_monotonic_ns < previous_monotonic_ns
        ):
            raise ValueError("readiness sample monotonic sequence regressed")
        if previous_captured_at is not None and captured_at < previous_captured_at:
            raise ValueError("readiness sample UTC sequence regressed")
        previous_monotonic_ns = sample_monotonic_ns
        previous_captured_at = captured_at
        if sample["gpu_uuids"] != gpu_uuids:
            raise ValueError("readiness sample GPU UUID mismatch")
        if sample["target_compute_app_count"] != 0:
            raise ValueError("readiness sample contains a target compute app")
        for key in ("memory_available_kib", "swap_used_kib", "free_disk_bytes"):
            _nonnegative_int(sample[key], f"readiness sample {key}")
    if observations["all_sample_count"] != len(samples):
        raise ValueError("readiness all-sample count mismatch")
    expected_window_count = math.ceil(duration / interval) + 1
    if observations["window_sample_count"] != expected_window_count:
        raise ValueError("readiness window-sample count mismatch")
    expected_phases = [
        "before-initial-lease",
        "initial-leases-held",
        "after-initial-release",
        *(["sampling-window"] * expected_window_count),
        "reacquire-leases-held",
        "after-reacquire-release",
    ]
    if [item["phase"] for item in samples] != expected_phases:
        raise ValueError("readiness sample phase sequence mismatch")
    window_samples = [item for item in samples if item["phase"] == "sampling-window"]
    if len(window_samples) != expected_window_count:
        raise ValueError("readiness sampling-window evidence is incomplete")
    window_start = window_samples[0]["monotonic_ns"]
    window_end = window_samples[-1]["monotonic_ns"]
    window_gaps = [
        current["monotonic_ns"] - previous["monotonic_ns"]
        for previous, current in zip(
            window_samples[:-1], window_samples[1:], strict=True
        )
    ]
    maximum_window_gap_ns = max(window_gaps)
    if (
        observations["window_start_monotonic_ns"] != window_start
        or observations["window_end_monotonic_ns"] != window_end
        or window_end - window_start < int(duration * 1_000_000_000)
        or any(gap <= 0 for gap in window_gaps)
        or maximum_window_gap_ns > expected_maximum_sample_gap_ns
        or observations["maximum_sample_gap_nanoseconds_observed"]
        != maximum_window_gap_ns
    ):
        raise ValueError("readiness sampling-window cadence mismatch")
    min_ram = min(item["memory_available_kib"] for item in samples)
    initial_swap = samples[0]["swap_used_kib"]
    max_swap = max(item["swap_used_kib"] for item in samples)
    max_swap_growth = max(0, max_swap - initial_swap)
    if (
        observations["minimum_available_ram_kib_observed"] != min_ram
        or min_ram < MIN_AVAILABLE_RAM_KIB
        or observations["initial_swap_used_kib"] != initial_swap
        or observations["maximum_swap_used_kib"] != max_swap
        or observations["maximum_swap_growth_kib_observed"] != max_swap_growth
        or max_swap_growth > MAX_SWAP_GROWTH_KIB
        or observations["free_disk_bytes_before"] != samples[0]["free_disk_bytes"]
        or observations["free_disk_bytes_after"] != samples[-1]["free_disk_bytes"]
        or observations["free_disk_bytes_before"] < expected_required_disk
        or observations["free_disk_bytes_after"] < expected_required_disk
    ):
        raise ValueError("readiness system observations are inconsistent")
    if (
        observations["formal_output_absent_before"] is not True
        or observations["formal_output_absent_after"] is not True
    ):
        raise ValueError("readiness formal-output absence evidence is invalid")
    for cycle_name, expected_cycle in (
        ("initial_lease_cycle", "initial"),
        ("reacquire_lease_cycle", "reacquire"),
    ):
        cycle = _exact_mapping(
            observations[cycle_name], _LEASE_CYCLE_KEYS, "readiness lease cycle"
        )
        if (
            cycle["cycle"] != expected_cycle
            or cycle["lane_ids"] != ["lane-a", "lane-b"]
            or cycle["execution_devices"] != ["cuda:0", "cuda:1"]
            or cycle["gpu_uuids"] != gpu_uuids
            or cycle["resource_binding_hashes"]
            != [item.identity_hash for item in inventory]
            or cycle["physical_resource_hashes"]
            != [item.physical_resource_hash for item in inventory]
            or cycle["simultaneously_held"] is not True
            or cycle["released"] is not True
        ):
            raise ValueError("readiness dual-GPU lease-cycle evidence is invalid")
        for hash_key in (
            "resource_binding_hashes",
            "physical_resource_hashes",
        ):
            hashes = cycle[hash_key]
            if not isinstance(hashes, list) or len(hashes) != 2:
                raise ValueError("readiness lease-cycle hashes are invalid")
            for item in hashes:
                require_sha256_digest(item, f"readiness lease-cycle {hash_key}")
        held_sequence = cycle["held_sample_sequence"]
        if (
            isinstance(held_sequence, bool)
            or not isinstance(held_sequence, int)
            or held_sequence < 0
            or held_sequence >= len(samples)
            or samples[held_sequence]["phase"] != f"{expected_cycle}-leases-held"
        ):
            raise ValueError("readiness lease-cycle held sample mismatch")
    if any(checks[key] is not True for key in _CHECK_KEYS):
        raise ValueError("readiness checks must all pass")
    if authority != {
        "operational_only": True,
        "provider_requests": 0,
        "gpu_workloads": 0,
        "ledger_writes": 0,
        "scientific_claim_authority": False,
        "acceptance_authority": False,
        "ledger_authority": False,
    }:
        raise ValueError("readiness authority projection is invalid")
    started_at = _parse_utc_timestamp(
        observations["started_at_utc"], "readiness started_at_utc", maximum=128
    )
    completed_at = _parse_utc_timestamp(
        observations["completed_at_utc"], "readiness completed_at_utc", maximum=128
    )
    first_captured_at = _parse_utc_timestamp(
        samples[0]["captured_at_utc"],
        "readiness first sample captured_at_utc",
        maximum=128,
    )
    last_captured_at = _parse_utc_timestamp(
        samples[-1]["captured_at_utc"],
        "readiness last sample captured_at_utc",
        maximum=128,
    )
    if started_at > first_captured_at or completed_at < last_captured_at:
        raise ValueError("readiness UTC observation bounds mismatch")
    declared_hash = require_sha256_digest(
        root["receipt_hash"], "Stair launch readiness receipt_hash"
    )
    hash_payload = dict(root)
    del hash_payload["receipt_hash"]
    if declared_hash != canonical_json_hash(hash_payload):
        raise ValueError("Stair launch readiness receipt hash mismatch")
    root["producer"] = producer
    return strict_json_object_loads(
        canonical_json_bytes(root), context="Stair launch readiness receipt"
    )


def _validate_producer_mapping(
    value: Any,
    *,
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    producer = _exact_mapping(value, _PRODUCER_KEYS, "readiness producer")
    argv = producer["argv"]
    if (
        not isinstance(argv, list)
        or not argv
        or len(argv) > 256
        or any(
            not isinstance(item, str) or not item or len(item) > 4096 for item in argv
        )
        or sum(len(item) for item in argv) > 64 * 1024
    ):
        raise ValueError("readiness producer argv is invalid")
    source_root_value = binding.get("source_root")
    if (
        not isinstance(source_root_value, str)
        or not Path(source_root_value).is_absolute()
    ):
        raise ValueError("readiness producer source root is invalid")
    source_root = Path(source_root_value).resolve(strict=False)
    python_executable = _validate_file_evidence(
        producer["python_executable"], "readiness producer Python"
    )
    entrypoint = _validate_file_evidence(
        producer["entrypoint"], "readiness producer entrypoint"
    )
    expected_entrypoint = str(
        (source_root / "scripts" / "run_stair_generation_launch_readiness.py").resolve(
            strict=False
        )
    )
    if entrypoint["path"] != expected_entrypoint:
        raise ValueError("readiness producer entrypoint path mismatch")
    working_directory = producer["working_directory"]
    if (
        not isinstance(working_directory, str)
        or not Path(working_directory).is_absolute()
    ):
        raise ValueError("readiness producer working directory must be absolute")
    process_id = _positive_int(producer["process_id"], "readiness producer process_id")
    process_group_id = _optional_positive_int(
        producer["process_group_id"], "readiness producer process_group_id"
    )
    process_session_id = _optional_positive_int(
        producer["process_session_id"], "readiness producer process_session_id"
    )
    raw_critical_sources = producer["critical_sources"]
    if not isinstance(raw_critical_sources, list) or len(raw_critical_sources) != 3:
        raise ValueError("readiness producer critical sources are invalid")
    critical_sources = [
        _validate_file_evidence(item, "readiness producer critical source")
        for item in raw_critical_sources
    ]
    expected_critical_paths = [
        str(path) for path in _producer_critical_source_paths(source_root, strict=False)
    ]
    if [item["path"] for item in critical_sources] != expected_critical_paths:
        raise ValueError("readiness producer critical source paths mismatch")
    return {
        "argv": list(argv),
        "python_executable": python_executable,
        "entrypoint": entrypoint,
        "working_directory": working_directory,
        "process_id": process_id,
        "process_group_id": process_group_id,
        "process_session_id": process_session_id,
        "critical_sources": critical_sources,
    }




def _validate_v3_execution_mapping(
    value: Any,
    *,
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    execution = _exact_mapping(value, _V3_EXECUTION_KEYS, "readiness v3 execution")
    source_root_value = binding.get("source_root")
    if (
        not isinstance(source_root_value, str)
        or not Path(source_root_value).is_absolute()
    ):
        raise ValueError("readiness v3 execution source root is invalid")
    source_root = Path(source_root_value).resolve(strict=False)
    expected_paths = [
        str(path) for path in _v3_critical_execution_paths(source_root, strict=False)
    ]
    raw_sources = execution["critical_sources"]
    if not isinstance(raw_sources, list) or len(raw_sources) != len(expected_paths):
        raise ValueError("readiness v3 critical execution sources are invalid")
    sources = [
        _validate_file_evidence(item, "readiness v3 critical execution source")
        for item in raw_sources
    ]
    if [item["path"] for item in sources] != expected_paths:
        raise ValueError("readiness v3 critical execution source paths mismatch")
    return {
        "critical_sources": sources,
        "formal_generation_command_sha256": require_sha256_checksum(
            execution["formal_generation_command_sha256"],
            "readiness v3 formal generation command sha256",
        ),
    }






def _validated_v3_subject_mapping(
    value: Any,
    *,
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("readiness v3 subject must be an object")
    subject_mode = value.get("subject_mode")
    if subject_mode == STAIR_FORMAL_LAUNCH_SUBJECT_FRESH_ABSENT:
        subject = _exact_mapping(
            value,
            {
                "subject_mode",
                "formal_output_absent",
                "current_accounted_output_bytes",
            },
            "readiness v3 fresh subject",
        )
        if (
            subject["formal_output_absent"] is not True
            or subject["current_accounted_output_bytes"] != 0
        ):
            raise ValueError("readiness v3 fresh subject projection mismatch")
        return subject
    if subject_mode not in (
        STAIR_FORMAL_LAUNCH_SUBJECT_NONTERMINAL_RESUME,
        STAIR_FORMAL_LAUNCH_SUBJECT_TERMINAL_FINALIZATION,
        STAIR_FORMAL_LAUNCH_SUBJECT_TERMINAL_AUDIT_ONLY,
    ):
        raise ValueError("readiness v3 subject mode is invalid")
    subject = _exact_mapping(
        value,
        _V3_RESUME_SUBJECT_KEYS,
        "readiness v3 resume subject",
    )
    output_value = binding.get("formal_output_dir")
    source_commit = binding.get("source_commit")
    if not isinstance(output_value, str) or not Path(output_value).is_absolute():
        raise ValueError("readiness v3 resume output binding is invalid")
    output_dir = Path(output_value).resolve(strict=False)

    source_identity = _exact_mapping(
        subject["source_identity"],
        {"file", "source_commit", "source_checkout_hash"},
        "readiness v3 source identity",
    )
    _require_evidence_path(
        source_identity["file"],
        output_dir / STAIR_GENERATION_SOURCE_CHECKOUT_FILENAME,
        "readiness v3 source identity file",
    )
    if source_identity["source_commit"] != source_commit:
        raise ValueError("readiness v3 source identity commit mismatch")
    require_sha256_digest(
        source_identity["source_checkout_hash"],
        "readiness v3 source checkout hash",
    )

    genesis = _exact_mapping(
        subject["session_genesis"],
        {"file", "session_genesis_hash"},
        "readiness v3 session Genesis",
    )
    genesis_file = _validate_optional_file_evidence(
        genesis["file"],
        expected_path=output_dir / SESSION_GENESIS_FILENAME,
        label="readiness v3 session Genesis file",
    )
    if genesis_file["exists"]:
        require_sha256_digest(
            genesis["session_genesis_hash"],
            "readiness v3 session Genesis hash",
        )
    elif genesis["session_genesis_hash"] is not None:
        raise ValueError("readiness v3 absent Genesis carries a hash")

    contract = _exact_mapping(
        subject["generation_session_contract"],
        {"file", "contract_hash"},
        "readiness v3 generation session contract",
    )
    _require_evidence_path(
        contract["file"],
        output_dir / GENERATION_SESSION_CONTRACT_FILENAME,
        "readiness v3 generation session contract file",
    )
    require_sha256_digest(
        contract["contract_hash"], "readiness v3 generation contract hash"
    )

    continuation = _exact_mapping(
        subject["continuation_policy"],
        {"file", "policy_hash"},
        "readiness v3 continuation policy",
    )
    _require_evidence_path(
        continuation["file"],
        output_dir / GENERATION_CONTINUATION_POLICY_FILENAME,
        "readiness v3 continuation policy file",
    )
    require_sha256_digest(
        continuation["policy_hash"], "readiness v3 continuation policy hash"
    )

    ledger = _exact_mapping(
        subject["ledger"],
        {"file", "height", "snapshot_hash"},
        "readiness v3 ledger",
    )
    ledger_file = _validate_optional_file_evidence(
        ledger["file"],
        expected_path=output_dir / "ledger.jsonl",
        label="readiness v3 ledger file",
    )
    height = _nonnegative_int(ledger["height"], "readiness v3 ledger height")
    if height > STAIR_FINAL_CONFIRMATION_REQUIRED_GENERATIONS:
        raise ValueError("readiness v3 ledger exceeds the terminal generation")
    require_sha256_digest(ledger["snapshot_hash"], "readiness v3 ledger snapshot hash")
    if height and ledger_file["exists"] is not True:
        raise ValueError("readiness v3 non-empty ledger file is absent")

    lifecycle = _validate_v3_lifecycle_subject(
        subject["lifecycle"],
        output_dir=output_dir,
        expected_height=height,
    )
    cost = _validate_v3_cost_subject(subject["cost"], output_dir=output_dir)
    attempt_state = _validate_v3_attempt_state(
        subject["attempt_state"], output_dir=output_dir
    )
    final_confirmation = _validate_v3_final_confirmation_subject(
        subject["final_confirmation"], output_dir=output_dir
    )
    formal_report = _validate_optional_file_evidence(
        subject["formal_report"],
        expected_path=output_dir / STAIR_GENERATION_REPORT_FILENAME,
        label="readiness v3 formal report",
    )
    current_accounted = _nonnegative_int(
        subject["current_accounted_output_bytes"],
        "readiness v3 current accounted output bytes",
    )
    if current_accounted > FORMAL_MAX_OUTPUT_BYTES:
        raise ValueError("readiness v3 accounted output exceeds its frozen cap")
    if cost["current_accounted_output_bytes"] != current_accounted:
        raise ValueError("readiness v3 cost/accounted-output binding mismatch")

    initialized = lifecycle["initialized"]
    terminal = lifecycle["terminal"]
    active = lifecycle["active"]
    continuation_eligible = lifecycle["continuation_eligible"]
    reservation_exists = final_confirmation["reservation"]["file"]["exists"]
    result_exists = final_confirmation["result"]["file"]["exists"]
    report_exists = formal_report["exists"]
    if not initialized and genesis_file["exists"]:
        raise ValueError("readiness v3 initialized lifecycle lacks Genesis")
    if initialized and not genesis_file["exists"]:
        raise ValueError("readiness v3 lifecycle exists before Genesis")
    if subject_mode == STAIR_FORMAL_LAUNCH_SUBJECT_NONTERMINAL_RESUME:
        if terminal or (not active and not continuation_eligible):
            raise ValueError("readiness v3 nonterminal subject is not continuable")
        if reservation_exists or result_exists or report_exists:
            raise ValueError("readiness v3 nonterminal subject has terminal artifacts")
    else:
        if (
            not initialized
            or not terminal
            or active
            or continuation_eligible
            or lifecycle["closed_generation_count"]
            != STAIR_FINAL_CONFIRMATION_REQUIRED_GENERATIONS
            or height != STAIR_FINAL_CONFIRMATION_REQUIRED_GENERATIONS
        ):
            raise ValueError("readiness v3 terminal subject is not terminal")
        if subject_mode == STAIR_FORMAL_LAUNCH_SUBJECT_TERMINAL_FINALIZATION:
            if report_exists:
                raise ValueError("readiness v3 finalization subject already has report")
        elif not (reservation_exists and result_exists and report_exists):
            raise ValueError("readiness v3 audit-only subject is incomplete")

    declared_snapshot_hash = require_sha256_digest(
        subject["snapshot_hash"], "readiness v3 subject snapshot hash"
    )
    snapshot_body = dict(subject)
    del snapshot_body["snapshot_hash"]
    if declared_snapshot_hash != canonical_json_hash(snapshot_body):
        raise ValueError("readiness v3 subject snapshot hash mismatch")
    normalized = dict(subject)
    normalized["lifecycle"] = lifecycle
    normalized["cost"] = {
        key: value
        for key, value in cost.items()
        if key != "current_accounted_output_bytes"
    }
    normalized["attempt_state"] = attempt_state
    normalized["final_confirmation"] = final_confirmation
    normalized["formal_report"] = formal_report
    return strict_json_object_loads(
        canonical_json_bytes(normalized),
        context="readiness v3 resume subject",
    )


def _validate_v3_lifecycle_subject(
    value: Any,
    *,
    output_dir: Path,
    expected_height: int,
) -> dict[str, Any]:
    lifecycle = _exact_mapping(
        value,
        {
            "initialized",
            "head_hash",
            "active",
            "active_segment_id",
            "segment_count",
            "closed_generation_count",
            "terminal",
            "continuation_eligible",
            "cost_chain_head_hash",
            "receipt_files",
        },
        "readiness v3 lifecycle",
    )
    initialized = lifecycle["initialized"]
    if not isinstance(initialized, bool):
        raise ValueError("readiness v3 lifecycle initialized must be boolean")
    head_hash = lifecycle["head_hash"]
    if head_hash is not None:
        require_sha256_digest(head_hash, "readiness v3 lifecycle head hash")
    active = lifecycle["active"]
    active_segment_id = lifecycle["active_segment_id"]
    if not isinstance(active, bool) or (active_segment_id is not None) is not active:
        raise ValueError("readiness v3 lifecycle active state mismatch")
    if active_segment_id is not None:
        _bounded_text(
            active_segment_id,
            "readiness v3 lifecycle active segment id",
            maximum=128,
        )
    segment_count = _nonnegative_int(
        lifecycle["segment_count"], "readiness v3 lifecycle segment count"
    )
    closed_count = _nonnegative_int(
        lifecycle["closed_generation_count"],
        "readiness v3 lifecycle closed generation count",
    )
    if (
        closed_count > STAIR_FINAL_CONFIRMATION_REQUIRED_GENERATIONS
        or (active and closed_count > expected_height)
        or (
        not active and closed_count != expected_height
        )
    ):
        raise ValueError("readiness v3 lifecycle and ledger heights mismatch")
    for key in ("terminal", "continuation_eligible"):
        if not isinstance(lifecycle[key], bool):
            raise ValueError(f"readiness v3 lifecycle {key} must be boolean")
    cost_head = lifecycle["cost_chain_head_hash"]
    if cost_head is not None:
        require_sha256_digest(cost_head, "readiness v3 lifecycle cost-chain head")
    receipt_files = _validate_payload_evidence_list(
        lifecycle["receipt_files"],
        roots=(
            output_dir / "generation_segments",
            output_dir / "generation_recoveries",
        ),
        label="readiness v3 lifecycle receipt",
    )
    if not initialized:
        if (
            head_hash is not None
            or active
            or segment_count
            or closed_count
            or lifecycle["terminal"]
            or lifecycle["continuation_eligible"] is not True
            or cost_head is not None
            or receipt_files
        ):
            raise ValueError("readiness v3 uninitialized lifecycle is not empty")
    elif head_hash is None:
        if segment_count or receipt_files:
            raise ValueError("readiness v3 empty lifecycle has receipt evidence")
    elif not receipt_files or head_hash not in {
        item["payload_hash"] for item in receipt_files
    }:
        raise ValueError("readiness v3 lifecycle head has no file evidence")
    return {**lifecycle, "receipt_files": receipt_files}


def _validate_v3_cost_subject(value: Any, *, output_dir: Path) -> dict[str, Any]:
    cost = _exact_mapping(
        value,
        {"contract", "receipt_count", "head", "receipt_files"},
        "readiness v3 cost",
    )
    contract = _exact_mapping(
        cost["contract"],
        {"file", "contract_hash"},
        "readiness v3 cost contract",
    )
    _require_evidence_path(
        contract["file"],
        output_dir / STAIR_GENERATION_SESSION_COST_FILENAME,
        "readiness v3 cost contract file",
    )
    require_sha256_digest(contract["contract_hash"], "readiness v3 cost contract hash")
    receipt_count = _nonnegative_int(
        cost["receipt_count"], "readiness v3 cost receipt count"
    )
    receipt_files = _validate_payload_evidence_list(
        cost["receipt_files"],
        roots=(output_dir / STAIR_GENERATION_ACTUAL_COST_DIRNAME,),
        label="readiness v3 cost receipt",
    )
    if len(receipt_files) != receipt_count:
        raise ValueError("readiness v3 cost receipt count mismatch")
    if receipt_count:
        head = _validate_payload_file_evidence(
            cost["head"], label="readiness v3 cost head"
        )
        if head != receipt_files[-1]:
            raise ValueError("readiness v3 cost head is not the final receipt")
        _, head_payload = _canonical_object_file(
            Path(head["path"]),
            label="readiness v3 cost head",
            max_bytes=MAX_READINESS_RECEIPT_BYTES,
        )
        actual = head_payload.get("actual")
        if not isinstance(actual, Mapping):
            raise ValueError("readiness v3 cost head actual usage is invalid")
        current_accounted = _nonnegative_int(
            actual.get("accounted_output_bytes"),
            "readiness v3 cost head accounted output bytes",
        )
    else:
        if cost["head"] is not None:
            raise ValueError("readiness v3 empty cost chain has a head")
        head = None
        current_accounted = 0
    return {
        "contract": contract,
        "receipt_count": receipt_count,
        "head": head,
        "receipt_files": receipt_files,
        "current_accounted_output_bytes": current_accounted,
    }


def _validate_v3_attempt_state(value: Any, *, output_dir: Path) -> dict[str, Any]:
    state = _exact_mapping(
        value,
        {"file_count", "total_size_bytes", "files", "snapshot_hash"},
        "readiness v3 attempt state",
    )
    files_value = state["files"]
    if not isinstance(files_value, list):
        raise ValueError("readiness v3 attempt-state files must be an array")
    files = [
        _validate_file_evidence(item, "readiness v3 attempt-state file")
        for item in files_value
    ]
    if [item["path"] for item in files] != sorted(item["path"] for item in files):
        raise ValueError("readiness v3 attempt-state files are not ordered")
    if len({item["path"] for item in files}) != len(files):
        raise ValueError("readiness v3 attempt-state files contain duplicates")
    roots = tuple(
        (output_dir / name).resolve(strict=False)
        for name in ("job_attempts", "live_invocations", "authoring")
    )
    for item in files:
        path = Path(item["path"])
        if path.suffix not in {".json", ".jsonl"} or not any(
            _path_is_within(path, root) for root in roots
        ):
            raise ValueError("readiness v3 attempt-state path is not authoritative")
    if state["file_count"] != len(files) or state["total_size_bytes"] != sum(
        item["size_bytes"] for item in files
    ):
        raise ValueError("readiness v3 attempt-state counts mismatch")
    body = {
        "file_count": len(files),
        "total_size_bytes": state["total_size_bytes"],
        "files": files,
    }
    if require_sha256_digest(
        state["snapshot_hash"], "readiness v3 attempt-state snapshot hash"
    ) != canonical_json_hash(body):
        raise ValueError("readiness v3 attempt-state snapshot hash mismatch")
    return {**body, "snapshot_hash": state["snapshot_hash"]}


def _validate_v3_final_confirmation_subject(
    value: Any,
    *,
    output_dir: Path,
) -> dict[str, Any]:
    final = _exact_mapping(
        value,
        {"plan", "reservation", "result"},
        "readiness v3 final confirmation",
    )
    final_root = output_dir / STAIR_FINAL_CONFIRMATION_DIRNAME
    plan = _exact_mapping(
        final["plan"], {"file", "plan_hash"}, "readiness v3 final plan"
    )
    _require_evidence_path(
        plan["file"],
        final_root / STAIR_FINAL_CONFIRMATION_PLAN_FILENAME,
        "readiness v3 final plan file",
    )
    require_sha256_digest(plan["plan_hash"], "readiness v3 final plan hash")
    normalized: dict[str, Any] = {"plan": plan}
    for key, filename, hash_name in (
        (
            "reservation",
            STAIR_FINAL_CONFIRMATION_RESERVATION_FILENAME,
            "reservation_hash",
        ),
        ("result", STAIR_FINAL_CONFIRMATION_RESULT_FILENAME, "result_hash"),
    ):
        artifact = _exact_mapping(
            final[key], {"file", hash_name}, f"readiness v3 final {key}"
        )
        file_evidence = _validate_optional_file_evidence(
            artifact["file"],
            expected_path=final_root / filename,
            label=f"readiness v3 final {key} file",
        )
        if file_evidence["exists"]:
            require_sha256_digest(
                artifact[hash_name], f"readiness v3 final {key} hash"
            )
        elif artifact[hash_name] is not None:
            raise ValueError(f"readiness v3 absent final {key} carries a hash")
        normalized[key] = {"file": file_evidence, hash_name: artifact[hash_name]}
    if (
        normalized["result"]["file"]["exists"]
        and not normalized["reservation"]["file"]["exists"]
    ):
        raise ValueError("readiness v3 final result exists without reservation")
    return normalized






def _validate_payload_evidence_list(
    value: Any,
    *,
    roots: tuple[Path, ...],
    label: str,
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"{label} files must be a list")
    result = [_validate_payload_file_evidence(item, label=label) for item in value]
    if len({item["path"] for item in result}) != len(result):
        raise ValueError(f"{label} files contain duplicate paths")
    for item in result:
        path = Path(item["path"])
        if not any(_path_is_within(path, root.resolve(strict=False)) for root in roots):
            raise ValueError(f"{label} path is outside its authoritative root")
    return result


def _validate_payload_file_evidence(value: Any, *, label: str) -> dict[str, Any]:
    evidence = _exact_mapping(
        value,
        _FILE_EVIDENCE_KEYS | {"payload_hash"},
        label,
    )
    file_evidence = _validate_file_evidence(
        {key: evidence[key] for key in _FILE_EVIDENCE_KEYS},
        label,
    )
    return {
        **file_evidence,
        "payload_hash": require_sha256_digest(
            evidence["payload_hash"], f"{label} payload hash"
        ),
    }


def _require_evidence_path(value: Any, path: Path, label: str) -> dict[str, Any]:
    evidence = _validate_file_evidence(value, label)
    if evidence["path"] != str(path.resolve(strict=False)):
        raise ValueError(f"{label} path mismatch")
    return evidence


def _validate_optional_file_evidence(
    value: Any,
    *,
    expected_path: Path,
    label: str,
) -> dict[str, Any]:
    evidence = _exact_mapping(
        value,
        {"path", "exists", "sha256", "size_bytes"},
        label,
    )
    if evidence["path"] != str(expected_path.resolve(strict=False)):
        raise ValueError(f"{label} path mismatch")
    if not isinstance(evidence["exists"], bool):
        raise ValueError(f"{label} exists must be boolean")
    size = _nonnegative_int(evidence["size_bytes"], f"{label} size_bytes")
    if evidence["exists"]:
        require_sha256_checksum(evidence["sha256"], f"{label} sha256")
    elif evidence["sha256"] is not None or size != 0:
        raise ValueError(f"{label} absent projection mismatch")
    return evidence


def _validate_file_evidence(value: Any, label: str) -> dict[str, Any]:
    evidence = _exact_mapping(value, _FILE_EVIDENCE_KEYS, label)
    path = evidence["path"]
    if not isinstance(path, str) or not Path(path).is_absolute():
        raise ValueError(f"{label} path must be absolute")
    return {
        "path": path,
        "sha256": require_sha256_checksum(evidence["sha256"], f"{label} sha256"),
        "size_bytes": _positive_int(evidence["size_bytes"], f"{label} size_bytes"),
    }


def _assert_current_producer_sources(
    value: Any,
    *,
    source_root: Path,
) -> None:
    producer = _exact_mapping(value, _PRODUCER_KEYS, "readiness producer")
    python_evidence = _validate_file_evidence(
        producer["python_executable"], "readiness producer Python"
    )
    if _file_evidence(Path(python_evidence["path"])) != python_evidence:
        raise ValueError("readiness producer Python bytes drifted")
    expected_entrypoint = _file_evidence(
        source_root / "scripts" / "run_stair_generation_launch_readiness.py"
    )
    if producer["entrypoint"] != expected_entrypoint:
        raise ValueError("readiness producer entrypoint bytes drifted")
    expected_critical_sources = [
        _file_evidence(path) for path in _producer_critical_source_paths(source_root)
    ]
    if producer["critical_sources"] != expected_critical_sources:
        raise ValueError("readiness producer critical source bytes drifted")




def _assert_current_v3_execution_sources(
    value: Any,
    *,
    source_root: Path,
    formal_generation_command_sha256: str,
) -> None:
    expected = _v3_execution_identity(
        source_root,
        formal_generation_command_sha256=formal_generation_command_sha256,
    )
    if value != expected:
        raise ValueError("readiness v3 execution source bytes drifted")


def _read_identity(path: Path, label: str, *, maximum: int) -> str:
    payload = read_bounded_regular_file(Path(path), max_bytes=maximum, label=label)
    try:
        value = payload.decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not valid UTF-8") from exc
    return _bounded_text(value, label, maximum=maximum)


def _assert_path_absent(path: Path, label: str) -> None:
    try:
        Path(path).lstat()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise ValueError(f"{label} could not be inspected") from exc
    raise FileExistsError(f"{label} must be absent")


def _path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _exact_mapping(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    if set(value) != keys:
        raise ValueError(f"{label} fields mismatch")
    return dict(value)


def _bounded_text(value: Any, label: str, *, maximum: int = 128) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f"{label} must be non-empty text of at most {maximum} chars")
    return value


def _parse_utc_timestamp(value: Any, label: str, *, maximum: int) -> datetime:
    text = _bounded_text(value, label, maximum=maximum)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(f"{label} must use UTC")
    return parsed


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _optional_positive_int(value: Any, label: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, label)


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _finite_number(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "DEFAULT_SAMPLE_DURATION_SECONDS",
    "DEFAULT_SAMPLE_INTERVAL_SECONDS",
    "MAX_READINESS_RECEIPT_BYTES",
    "MAX_SWAP_GROWTH_KIB",
    "MIN_AVAILABLE_RAM_KIB",
    "MIN_FREE_DISK_BYTES",
    "OUTPUT_DISK_MULTIPLIER",
    "STAIR_GENERATION_LAUNCH_READINESS_FILENAME",
    "STAIR_GENERATION_LAUNCH_READINESS_REF_SCHEMA_VERSION",
    "STAIR_GENERATION_LAUNCH_READINESS_V3_SCHEMA_VERSION",
    "STAIR_FORMAL_LAUNCH_SUBJECT_FRESH_ABSENT",
    "STAIR_FORMAL_LAUNCH_SUBJECT_MODES",
    "STAIR_FORMAL_LAUNCH_SUBJECT_NONTERMINAL_RESUME",
    "STAIR_FORMAL_LAUNCH_SUBJECT_TERMINAL_AUDIT_ONLY",
    "STAIR_FORMAL_LAUNCH_SUBJECT_TERMINAL_FINALIZATION",
    "StairGenerationLaunchReadinessConfig",
    "StairGenerationLaunchReadinessConfigV3",
    "StairGenerationLaunchReadinessReceipt",
    "StairGenerationLaunchReadinessReceiptRef",
    "collect_stair_generation_launch_readiness_v3",
    "load_stair_generation_launch_readiness_receipt",
    "resolve_stair_nvidia_smi_executable",
]
