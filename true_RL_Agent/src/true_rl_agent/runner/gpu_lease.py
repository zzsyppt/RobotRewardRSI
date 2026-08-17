"""Task-neutral, process-held leases for trusted physical GPU identities."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from true_rl_agent.agent_core.process_lock import (
    ProcessFileLock,
    ProcessFileLockUnavailableError,
    acquire_process_file_lock,
)
from true_rl_agent.evaluation.binding import canonical_json_hash, require_sha256_digest
from true_rl_agent.json_contract import canonical_json_bytes
from true_rl_agent.runner.specs import require_execution_device


GPU_LEASE_SCHEMA_VERSION = "true_rl_agent.runner.gpu_lease.v1"


@dataclass(frozen=True)
class GpuDeviceIdentity:
    """Trusted mapping from one execution ordinal to one physical GPU."""

    execution_device: str
    machine_id: str
    physical_id: str

    def __post_init__(self) -> None:
        execution_device = require_execution_device(
            self.execution_device,
            "GPU identity execution_device",
        )
        if not execution_device.startswith("cuda:"):
            raise ValueError("GPU identity requires a CUDA execution device")
        _require_bounded_text(self.machine_id, "GPU identity machine_id")
        _require_bounded_text(self.physical_id, "GPU identity physical_id")

    @property
    def identity_hash(self) -> str:
        return canonical_json_hash(
            {
                "schema_version": "true_rl_agent.runner.gpu_device_identity.v1",
                "execution_device": self.execution_device,
                "machine_id": self.machine_id,
                "physical_id": self.physical_id,
            }
        )

    @property
    def physical_resource_hash(self) -> str:
        """Stable lock key independent of process-local CUDA ordinals."""

        return canonical_json_hash(
            {
                "schema_version": "true_rl_agent.runner.physical_gpu_identity.v1",
                "machine_id": self.machine_id,
                "physical_id": self.physical_id,
            }
        )


@dataclass(frozen=True)
class GpuLeaseHolder:
    session_genesis_hash: str
    generation_id: str
    lane_id: str
    invocation_segment_id: str
    logical_job_id: str

    def __post_init__(self) -> None:
        require_sha256_digest(
            self.session_genesis_hash,
            "GPU lease session_genesis_hash",
        )
        for field_name in (
            "generation_id",
            "lane_id",
            "invocation_segment_id",
            "logical_job_id",
        ):
            _require_bounded_text(getattr(self, field_name), f"GPU lease {field_name}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_genesis_hash": self.session_genesis_hash,
            "generation_id": self.generation_id,
            "lane_id": self.lane_id,
            "invocation_segment_id": self.invocation_segment_id,
            "logical_job_id": self.logical_job_id,
        }


class GpuLease:
    """One held GPU lock whose metadata never acts as lock authority."""

    def __init__(
        self,
        *,
        identity: GpuDeviceIdentity,
        holder: GpuLeaseHolder,
        lock: ProcessFileLock,
        receipt: Mapping[str, Any],
    ) -> None:
        self.identity = identity
        self.holder = holder
        self._lock = lock
        self.receipt = MappingProxyType(dict(receipt))

    @property
    def resource_binding_hash(self) -> str:
        return self.identity.identity_hash

    def assert_owned(self) -> None:
        self._lock.assert_owned()

    def release(self) -> None:
        self._lock.release()

    def __enter__(self) -> GpuLease:
        self.assert_owned()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        _ = exc_type, exc_value, traceback
        self.release()


class GpuLeaseManager:
    """Resolve trusted device identities and serialize physical GPU ownership."""

    def __init__(
        self,
        *,
        lease_root: Path,
        devices: Mapping[str, GpuDeviceIdentity],
    ) -> None:
        if not isinstance(lease_root, Path):
            raise TypeError("GPU lease_root must be a Path")
        if not isinstance(devices, Mapping):
            raise TypeError("GPU devices must be a mapping")
        normalized: dict[str, GpuDeviceIdentity] = {}
        for execution_device, identity in devices.items():
            require_execution_device(execution_device, "GPU device mapping key")
            if not isinstance(identity, GpuDeviceIdentity):
                raise TypeError("GPU device mapping values must be GpuDeviceIdentity")
            if identity.execution_device != execution_device:
                raise ValueError("GPU device identity ordinal mismatch")
            normalized[execution_device] = identity
        self.lease_root = lease_root.resolve(strict=False)
        self.devices = MappingProxyType(normalized)

    def identity_for(self, execution_device: str) -> GpuDeviceIdentity | None:
        require_execution_device(execution_device)
        if execution_device == "cpu":
            return None
        try:
            return self.devices[execution_device]
        except KeyError as exc:
            raise ValueError(
                f"no trusted physical identity for {execution_device}"
            ) from exc

    def acquire(
        self,
        execution_device: str,
        *,
        holder: GpuLeaseHolder,
    ) -> GpuLease | _CpuLease:
        identity = self.identity_for(execution_device)
        if identity is None:
            return _CpuLease()
        receipt = {
            "schema_version": GPU_LEASE_SCHEMA_VERSION,
            "identity": {
                "execution_device": identity.execution_device,
                "machine_id": identity.machine_id,
                "physical_id": identity.physical_id,
                "identity_hash": identity.identity_hash,
                "physical_resource_hash": identity.physical_resource_hash,
            },
            "holder": holder.to_dict(),
            "owner_pid": os.getpid(),
            "ledger_authority": False,
        }
        payload = canonical_json_bytes(receipt, trailing_newline=True)
        lock_path = self.lease_root / f"{identity.physical_resource_hash}.lock"
        try:
            lock = acquire_process_file_lock(
                lock_path,
                blocking=False,
                payload=payload,
            )
        except ProcessFileLockUnavailableError as exc:
            raise GpuLeaseUnavailableError(
                f"physical GPU lease is already owned: {identity.physical_id}"
            ) from exc
        return GpuLease(
            identity=identity,
            holder=holder,
            lock=lock,
            receipt=receipt,
        )


class GpuLeaseUnavailableError(RuntimeError):
    """Raised when another process already owns a physical GPU lease."""


class _CpuLease:
    resource_binding_hash: None = None

    def assert_owned(self) -> None:
        return

    def release(self) -> None:
        return

    def __enter__(self) -> _CpuLease:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        _ = exc_type, exc_value, traceback


def _require_bounded_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 128:
        raise ValueError(f"{field_name} must be a non-empty string of at most 128 chars")
    return value


__all__ = [
    "GPU_LEASE_SCHEMA_VERSION",
    "GpuDeviceIdentity",
    "GpuLease",
    "GpuLeaseHolder",
    "GpuLeaseManager",
    "GpuLeaseUnavailableError",
]
