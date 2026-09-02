"""Stair-owned child GPU visibility and renderer binding."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from true_rl_agent.runner.specs import GpuProcessBinding
from true_rl_agent.runner.process_handle import (
    current_owned_process_identity,
    validate_owned_process_identity,
)


STAIR_GPU_RUNTIME_EVIDENCE_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.gpu_runtime_evidence.v2"
)
STAIR_GPU_COMPUTE_CONTEXT_OBSERVATION_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.gpu_compute_context_observation.v1"
)
_FULL_GPU_UUID_PATTERN = re.compile(
    r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\Z"
)
_BINDING_ARGUMENTS = (
    "gpu_host_execution_device",
    "gpu_process_execution_device",
    "gpu_resource_binding_hash",
)


@dataclass(frozen=True)
class StairGpuRuntimeBinding:
    """Validated task runtime evidence for one UUID-isolated Stair child."""

    gpu_process_binding: GpuProcessBinding
    physical_gpu_uuid: str
    renderer_active_gpu_ordinal: int

    def __post_init__(self) -> None:
        if not isinstance(self.gpu_process_binding, GpuProcessBinding):
            raise TypeError("Stair GPU runtime requires a GpuProcessBinding")
        if _FULL_GPU_UUID_PATTERN.fullmatch(self.physical_gpu_uuid) is None:
            raise ValueError("Stair GPU runtime requires one full physical GPU UUID")
        if (
            isinstance(self.renderer_active_gpu_ordinal, bool)
            or not isinstance(self.renderer_active_gpu_ordinal, int)
            or self.renderer_active_gpu_ordinal < 0
        ):
            raise ValueError("Stair renderer active GPU ordinal is invalid")

    def summary_fields(
        self,
        *,
        compute_context_observation: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        process_identity = current_owned_process_identity()
        observation = (
            validate_stair_gpu_compute_context_observation(
                compute_context_observation,
                expected_runtime=self,
                expected_process_id=process_identity["process_id"],
            )
            if compute_context_observation is not None
            else None
        )
        return {
            "gpu_process_binding": self.gpu_process_binding.to_dict(),
            "owned_process_identity": process_identity,
            "stair_gpu_runtime_evidence": {
                "schema_version": STAIR_GPU_RUNTIME_EVIDENCE_SCHEMA_VERSION,
                "cuda_visibility_binding": "single_physical_gpu_uuid",
                "physical_gpu_uuid": self.physical_gpu_uuid,
                "renderer_active_gpu_ordinal": self.renderer_active_gpu_ordinal,
                "renderer_multi_gpu": False,
                "renderer_multi_gpu_auto_enable": False,
                "renderer_multi_gpu_max_gpu_count": 1,
                "nvidia_visible_devices_unset": True,
                "external_compute_context_observation": observation,
            },
        }


def add_stair_gpu_process_binding_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    """Add the optional production GPU binding accepted by Stair children."""

    parser.add_argument("--gpu-host-execution-device")
    parser.add_argument("--gpu-process-execution-device")
    parser.add_argument("--gpu-resource-binding-hash")


def validate_stair_gpu_process_binding_invocation(
    args: argparse.Namespace,
    *,
    environment: Mapping[str, str],
) -> StairGpuRuntimeBinding | None:
    """Validate a bound child before Isaac creates an application context."""

    values = tuple(getattr(args, field_name, None) for field_name in _BINDING_ARGUMENTS)
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError("Stair GPU process binding arguments must be supplied together")

    binding = GpuProcessBinding(
        host_execution_device=values[0],
        process_execution_device=values[1],
        resource_binding_hash=values[2],
    )
    if binding.process_execution_device != "cuda:0":
        raise ValueError("Stair bound child process device must be cuda:0")
    if getattr(args, "device", None) != binding.process_execution_device:
        raise ValueError("Stair logical device does not match GPU process binding")

    visible_gpu = environment.get("CUDA_VISIBLE_DEVICES")
    if not isinstance(visible_gpu, str) or _FULL_GPU_UUID_PATTERN.fullmatch(
        visible_gpu
    ) is None:
        raise ValueError(
            "Stair bound child CUDA_VISIBLE_DEVICES must be one full GPU UUID"
        )
    if "NVIDIA_VISIBLE_DEVICES" in environment:
        raise ValueError(
            "Stair bound child requires NVIDIA_VISIBLE_DEVICES to be unset"
        )

    renderer_ordinal = int(binding.host_execution_device.removeprefix("cuda:"))
    if getattr(args, "kit_args", ""):
        raise ValueError("Stair bound child Kit arguments are task-owned")
    args.kit_args = stair_single_gpu_kit_args(
        renderer_active_gpu_ordinal=renderer_ordinal
    )
    return StairGpuRuntimeBinding(
        gpu_process_binding=binding,
        physical_gpu_uuid=visible_gpu,
        renderer_active_gpu_ordinal=renderer_ordinal,
    )


def stair_single_gpu_kit_args(*, renderer_active_gpu_ordinal: int) -> str:
    """Return the P6-A-validated Kit settings for one isolated Stair child."""

    if (
        isinstance(renderer_active_gpu_ordinal, bool)
        or not isinstance(renderer_active_gpu_ordinal, int)
        or renderer_active_gpu_ordinal < 0
    ):
        raise ValueError("Stair renderer active GPU ordinal is invalid")
    return " ".join(
        (
            "--/renderer/multiGpu/autoEnable=false",
            "--/renderer/multiGpu/maxGpuCount=1",
            f"--/renderer/activeGpu={renderer_active_gpu_ordinal}",
        )
    )


def attach_stair_gpu_runtime_evidence(
    summary: Mapping[str, Any],
    runtime: StairGpuRuntimeBinding | None,
    *,
    compute_context_observation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Attach private task-owned launch evidence without changing legacy summaries."""

    result = dict(summary)
    if runtime is None:
        if compute_context_observation is not None:
            raise ValueError(
                "unbound Stair summary cannot claim a GPU compute observation"
            )
        return result
    if any(
        key in result
        for key in (
            "gpu_process_binding",
            "owned_process_identity",
            "stair_gpu_runtime_evidence",
        )
    ):
        raise ValueError("Stair GPU runtime evidence was already attached")
    result.update(
        runtime.summary_fields(
            compute_context_observation=compute_context_observation,
        )
    )
    return result


def capture_stair_gpu_compute_context(
    runtime: StairGpuRuntimeBinding,
    *,
    nvidia_smi: str = "nvidia-smi",
) -> dict[str, Any]:
    """Take one external PID-to-UUID sample while the Isaac context is alive."""

    if not isinstance(runtime, StairGpuRuntimeBinding):
        raise TypeError("Stair GPU compute observation requires a runtime binding")
    if not isinstance(nvidia_smi, str) or not nvidia_smi.strip():
        raise ValueError("nvidia-smi executable is required")
    identity = current_owned_process_identity()
    completed = subprocess.run(
        [
            nvidia_smi,
            "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    process_rows: list[dict[str, Any]] = []
    for raw_line in completed.stdout.splitlines():
        if not raw_line.strip():
            continue
        parts = [part.strip() for part in raw_line.split(",")]
        if len(parts) != 3:
            raise ValueError("nvidia-smi returned an invalid compute-app row")
        gpu_uuid, raw_pid, raw_memory = parts
        try:
            pid = int(raw_pid)
            used_gpu_memory_mib = int(raw_memory)
        except ValueError as exc:
            raise ValueError(
                "nvidia-smi returned a non-numeric compute-app field"
            ) from exc
        if pid <= 0 or used_gpu_memory_mib < 0:
            raise ValueError("nvidia-smi returned an invalid compute-app value")
        if pid == identity["process_id"]:
            process_rows.append(
                {
                    "gpu_uuid": gpu_uuid,
                    "used_gpu_memory_mib": used_gpu_memory_mib,
                }
            )
    observation = {
        "schema_version": (
            STAIR_GPU_COMPUTE_CONTEXT_OBSERVATION_SCHEMA_VERSION
        ),
        "source": "nvidia-smi-query-compute-apps",
        "monotonic_ns": time.monotonic_ns(),
        "process_id": identity["process_id"],
        "rows": process_rows,
    }
    return validate_stair_gpu_compute_context_observation(
        observation,
        expected_runtime=runtime,
        expected_process_id=identity["process_id"],
    )


def validate_stair_gpu_compute_context_observation(
    value: Mapping[str, Any] | None,
    *,
    expected_runtime: StairGpuRuntimeBinding,
    expected_process_id: int,
) -> dict[str, Any]:
    """Require exactly one observed compute context on the planned UUID."""

    if not isinstance(expected_runtime, StairGpuRuntimeBinding):
        raise TypeError("expected Stair GPU runtime binding is invalid")
    if (
        isinstance(expected_process_id, bool)
        or not isinstance(expected_process_id, int)
        or expected_process_id <= 0
    ):
        raise ValueError("expected Stair process id is invalid")
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "source",
        "monotonic_ns",
        "process_id",
        "rows",
    }:
        raise ValueError("Stair GPU compute observation fields mismatch")
    if (
        value.get("schema_version")
        != STAIR_GPU_COMPUTE_CONTEXT_OBSERVATION_SCHEMA_VERSION
        or value.get("source") != "nvidia-smi-query-compute-apps"
        or value.get("process_id") != expected_process_id
    ):
        raise ValueError("Stair GPU compute observation identity mismatch")
    monotonic_ns = value.get("monotonic_ns")
    if (
        isinstance(monotonic_ns, bool)
        or not isinstance(monotonic_ns, int)
        or monotonic_ns <= 0
    ):
        raise ValueError("Stair GPU compute observation clock is invalid")
    rows = value.get("rows")
    if not isinstance(rows, list) or len(rows) != 1:
        raise ValueError(
            "Stair child must have exactly one observed GPU compute context"
        )
    row = rows[0]
    if not isinstance(row, Mapping) or set(row) != {
        "gpu_uuid",
        "used_gpu_memory_mib",
    }:
        raise ValueError("Stair GPU compute observation row fields mismatch")
    memory = row.get("used_gpu_memory_mib")
    if (
        row.get("gpu_uuid") != expected_runtime.physical_gpu_uuid
        or isinstance(memory, bool)
        or not isinstance(memory, int)
        or memory < 0
    ):
        raise ValueError(
            "Stair child compute context is not isolated to its planned GPU UUID"
        )
    return {
        "schema_version": (
            STAIR_GPU_COMPUTE_CONTEXT_OBSERVATION_SCHEMA_VERSION
        ),
        "source": "nvidia-smi-query-compute-apps",
        "monotonic_ns": monotonic_ns,
        "process_id": expected_process_id,
        "rows": [
            {
                "gpu_uuid": row["gpu_uuid"],
                "used_gpu_memory_mib": memory,
            }
        ],
    }


def validate_stair_gpu_runtime_summary_evidence(
    summary: Mapping[str, Any],
    *,
    expected_binding: GpuProcessBinding,
    expected_physical_gpu_uuid: str,
    require_compute_context_observation: bool,
) -> dict[str, Any]:
    """Validate the receipt-facing Stair process/UUID evidence in one summary."""

    if not isinstance(summary, Mapping):
        raise TypeError("Stair GPU runtime summary must be an object")
    actual_binding = GpuProcessBinding.from_mapping(
        summary.get("gpu_process_binding")
    )
    if actual_binding != expected_binding:
        raise ValueError("Stair summary GPU process binding mismatch")
    process_identity = validate_owned_process_identity(
        summary.get("owned_process_identity"),
        require_process_group_leader=True,
    )
    evidence = summary.get("stair_gpu_runtime_evidence")
    expected_fields = {
        "schema_version",
        "cuda_visibility_binding",
        "physical_gpu_uuid",
        "renderer_active_gpu_ordinal",
        "renderer_multi_gpu",
        "renderer_multi_gpu_auto_enable",
        "renderer_multi_gpu_max_gpu_count",
        "nvidia_visible_devices_unset",
        "external_compute_context_observation",
    }
    if not isinstance(evidence, Mapping) or set(evidence) != expected_fields:
        raise ValueError("Stair GPU runtime evidence fields mismatch")
    renderer_ordinal = int(
        expected_binding.host_execution_device.removeprefix("cuda:")
    )
    expected_values = {
        "schema_version": STAIR_GPU_RUNTIME_EVIDENCE_SCHEMA_VERSION,
        "cuda_visibility_binding": "single_physical_gpu_uuid",
        "physical_gpu_uuid": expected_physical_gpu_uuid,
        "renderer_active_gpu_ordinal": renderer_ordinal,
        "renderer_multi_gpu": False,
        "renderer_multi_gpu_auto_enable": False,
        "renderer_multi_gpu_max_gpu_count": 1,
        "nvidia_visible_devices_unset": True,
    }
    for field_name, expected in expected_values.items():
        if evidence.get(field_name) != expected:
            raise ValueError(f"Stair GPU runtime evidence {field_name} mismatch")
    raw_observation = evidence.get("external_compute_context_observation")
    if raw_observation is None:
        if require_compute_context_observation:
            raise ValueError("Stair summary is missing external PID-to-UUID evidence")
        observation = None
    else:
        observation = validate_stair_gpu_compute_context_observation(
            raw_observation,
            expected_runtime=StairGpuRuntimeBinding(
                gpu_process_binding=expected_binding,
                physical_gpu_uuid=expected_physical_gpu_uuid,
                renderer_active_gpu_ordinal=renderer_ordinal,
            ),
            expected_process_id=process_identity["process_id"],
        )
    return {
        "gpu_process_binding": actual_binding.to_dict(),
        "owned_process_identity": process_identity,
        "stair_gpu_runtime_evidence": {
            **expected_values,
            "external_compute_context_observation": observation,
        },
    }


__all__ = [
    "STAIR_GPU_COMPUTE_CONTEXT_OBSERVATION_SCHEMA_VERSION",
    "STAIR_GPU_RUNTIME_EVIDENCE_SCHEMA_VERSION",
    "StairGpuRuntimeBinding",
    "add_stair_gpu_process_binding_arguments",
    "attach_stair_gpu_runtime_evidence",
    "capture_stair_gpu_compute_context",
    "stair_single_gpu_kit_args",
    "validate_stair_gpu_compute_context_observation",
    "validate_stair_gpu_process_binding_invocation",
    "validate_stair_gpu_runtime_summary_evidence",
]
