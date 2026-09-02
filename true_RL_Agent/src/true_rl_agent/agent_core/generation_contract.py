"""Create-once contracts for one fixed-width generation session.

This module owns task-neutral identity and topology only.  Task selectors and
evaluation semantics remain opaque hashes supplied by the task plugin.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from true_rl_agent.evaluation.binding import (
    canonical_json_hash,
    require_sha256_checksum,
    require_sha256_digest,
)
from true_rl_agent.fs_contract import create_once_bytes, read_bounded_regular_file
from true_rl_agent.json_contract import canonical_json_bytes, strict_json_object_loads
from true_rl_agent.runner.specs import GpuProcessBinding, require_execution_device


GENERATION_SESSION_CONTRACT_SCHEMA_VERSION = (
    "true_rl_agent.generation_session_contract.v2"
)
GENERATION_SESSION_CONTRACT_V3_SCHEMA_VERSION = (
    "true_rl_agent.generation_session_contract.v3"
)
GENERATION_SESSION_CONTRACT_REF_SCHEMA_VERSION = (
    "true_rl_agent.generation_session_contract_ref.v1"
)
GENERATION_MANIFEST_SCHEMA_VERSION = "true_rl_agent.generation_manifest.v1"
GENERATION_MANIFEST_V2_SCHEMA_VERSION = "true_rl_agent.generation_manifest.v2"
GENERATION_LEDGER_RECORD_SCHEMA_VERSION = (
    "true_rl_agent.generation_ledger_record.v1"
)
GENERATION_PUBLIC_HISTORY_PROJECTION_SCHEMA_VERSION = (
    "true_rl_agent.generation_public_history_projection.v3"
)
GENERATION_SESSION_CONTRACT_FILENAME = "generation_session_contract.json"
GENERATION_WIDTH = 2
GENERATION_LANE_IDS = ("lane-a", "lane-b")
GENERATION_SESSION_MODES = frozenset(
    {
        "development_generation_pilot",
        "development_generation_validation",
        "formal_generation_session",
    }
)
MAX_GENERATION_LANE_INTENT_BYTES = 2 * 1024


def generation_training_initialization_hash(
    *,
    policy_initialization_kind: str,
    training_binding_hash: str,
    expected_initial_policy_state_hash: str | None,
) -> str:
    """Hash the candidate-independent initialization frozen before authoring."""

    if policy_initialization_kind not in {"random", "warm_start"}:
        raise ValueError("generation policy initialization kind is invalid")
    require_sha256_digest(
        training_binding_hash,
        "generation training binding hash",
    )
    if expected_initial_policy_state_hash is not None:
        require_sha256_digest(
            expected_initial_policy_state_hash,
            "generation expected initial policy state hash",
        )
    return canonical_json_hash(
        {
            "schema_version": "true_rl_agent.generation_training_initialization.v1",
            "policy_initialization": policy_initialization_kind,
            "training_binding_hash": training_binding_hash,
            "expected_initial_policy_state_hash": expected_initial_policy_state_hash,
        }
    )


def generation_resource_plan_hash(
    lanes: tuple["GenerationLaneAssignment", "GenerationLaneAssignment"],
) -> str:
    """Hash only trusted lane-to-resource assignments, not per-generation intent."""

    if (
        not isinstance(lanes, tuple)
        or len(lanes) != GENERATION_WIDTH
        or tuple(item.lane_id for item in lanes) != GENERATION_LANE_IDS
    ):
        raise ValueError("generation resource plan requires ordered lane-a/lane-b")
    bindings = tuple(item.gpu_process_binding for item in lanes)
    if any(binding is not None for binding in bindings):
        if not all(binding is not None for binding in bindings):
            raise ValueError(
                "generation resource plan cannot mix bound and unbound lanes"
            )
        bound_lanes = tuple(
            binding for binding in bindings if isinstance(binding, GpuProcessBinding)
        )
        if (
            len({binding.host_execution_device for binding in bound_lanes})
            != GENERATION_WIDTH
        ):
            raise ValueError(
                "generation resource plan requires distinct host devices"
            )
        if (
            len({binding.resource_binding_hash for binding in bound_lanes})
            != GENERATION_WIDTH
        ):
            raise ValueError(
                "generation resource plan requires distinct GPU bindings"
            )
        return canonical_json_hash(
            {
                "schema_version": "true_rl_agent.generation_resource_plan.v2",
                "lanes": [
                    {
                        "lane_id": item.lane_id,
                        "gpu_process_binding": item.gpu_process_binding.to_dict(),
                    }
                    for item in lanes
                    if item.gpu_process_binding is not None
                ],
            }
        )
    return canonical_json_hash(
        {
            "schema_version": "true_rl_agent.generation_resource_plan.v1",
            "lanes": [
                {
                    "lane_id": item.lane_id,
                    "execution_device": item.execution_device,
                    "resource_binding_hash": item.resource_binding_hash,
                }
                for item in lanes
            ],
        }
    )


GENERATION_AUTHORING_ORDER = GENERATION_LANE_IDS
GENERATION_TRIAL_STATUSES = frozenset(
    {
        "not_started",
        "candidate_failed",
        "verified_not_run",
        "screened_only",
        "qualified_not_selected",
        "formal_rejected",
        "accepted",
    }
)
_MAX_CONTRACT_BYTES = 128 * 1024
_BOUND_CONTRACT_KEYS = frozenset(
    {
        "authoring_policy",
        "continuation_policy",
        "evaluation_cost",
        "evaluation_protocol",
        "exploration_projector",
        "final_confirmation_plan",
        "provider_policy",
        "resource_plan",
        "retry_policy",
        "selector",
    }
)
_V3_BOUND_CONTRACT_KEYS = _BOUND_CONTRACT_KEYS | {"session_cost"}


@dataclass(frozen=True)
class GenerationSessionContractRef:
    logical_name: str
    checksum: str
    size_bytes: int
    schema_version: str = GENERATION_SESSION_CONTRACT_REF_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != GENERATION_SESSION_CONTRACT_REF_SCHEMA_VERSION:
            raise ValueError("generation session contract ref schema mismatch")
        if self.logical_name != GENERATION_SESSION_CONTRACT_FILENAME:
            raise ValueError("generation session contract ref logical name mismatch")
        require_sha256_checksum(self.checksum, "generation session contract checksum")
        if (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes <= 0
            or self.size_bytes > _MAX_CONTRACT_BYTES
        ):
            raise ValueError("generation session contract ref size is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "logical_name": self.logical_name,
            "checksum": self.checksum,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_mapping(cls, value: Any) -> "GenerationSessionContractRef":
        mapping = _exact_mapping(
            value,
            {"schema_version", "logical_name", "checksum", "size_bytes"},
            "generation session contract ref",
        )
        return cls(
            schema_version=mapping["schema_version"],
            logical_name=mapping["logical_name"],
            checksum=mapping["checksum"],
            size_bytes=mapping["size_bytes"],
        )


@dataclass(frozen=True)
class GenerationSessionContract:
    session_id: str
    task_id: str
    backend_id: str
    max_generations: int
    max_trials: int
    max_train_jobs: int
    bound_contract_hashes: Mapping[str, str]
    authoring_order: tuple[str, str] = GENERATION_AUTHORING_ORDER
    width: int = GENERATION_WIDTH
    ledger_record_schema: str = GENERATION_LEDGER_RECORD_SCHEMA_VERSION
    schema_version: str = GENERATION_SESSION_CONTRACT_SCHEMA_VERSION
    session_mode: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version not in {
            GENERATION_SESSION_CONTRACT_SCHEMA_VERSION,
            GENERATION_SESSION_CONTRACT_V3_SCHEMA_VERSION,
        }:
            raise ValueError("generation session contract schema mismatch")
        if self.schema_version == GENERATION_SESSION_CONTRACT_SCHEMA_VERSION:
            if self.session_mode is not None:
                raise ValueError(
                    "legacy generation session contract cannot carry session_mode"
                )
            expected_bound_contract_keys = _BOUND_CONTRACT_KEYS
        else:
            if self.session_mode not in GENERATION_SESSION_MODES:
                raise ValueError("generation session contract v3 session_mode is invalid")
            expected_bound_contract_keys = _V3_BOUND_CONTRACT_KEYS
        for field_name in ("session_id", "task_id", "backend_id"):
            _bounded_text(getattr(self, field_name), f"generation contract {field_name}")
        if self.width != GENERATION_WIDTH:
            raise ValueError("generation width must be exactly two")
        if tuple(self.authoring_order) != GENERATION_AUTHORING_ORDER:
            raise ValueError("generation authoring order must be lane-a then lane-b")
        if self.ledger_record_schema != GENERATION_LEDGER_RECORD_SCHEMA_VERSION:
            raise ValueError("generation ledger writer schema mismatch")
        for field_name in ("max_generations", "max_trials", "max_train_jobs"):
            _positive_int(getattr(self, field_name), f"generation contract {field_name}")
        if self.max_trials != self.max_generations * self.width:
            raise ValueError("generation max_trials must equal max_generations times width")
        if self.max_train_jobs != self.max_trials:
            raise ValueError("generation max_train_jobs must equal max_trials")
        hashes = dict(self.bound_contract_hashes)
        if set(hashes) != expected_bound_contract_keys:
            raise ValueError("generation bound contract hash fields mismatch")
        for field_name, value in hashes.items():
            require_sha256_digest(value, f"generation {field_name} contract hash")
        object.__setattr__(self, "bound_contract_hashes", MappingProxyType(hashes))

    @property
    def contract_hash(self) -> str:
        return canonical_json_hash(self._hash_payload())

    def _hash_payload(self) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "task_id": self.task_id,
            "backend_id": self.backend_id,
            "width": self.width,
            "authoring_order": list(self.authoring_order),
            "ledger_record_schema": self.ledger_record_schema,
            "max_generations": self.max_generations,
            "max_trials": self.max_trials,
            "max_train_jobs": self.max_train_jobs,
            "bound_contract_hashes": dict(self.bound_contract_hashes),
        }
        if self.schema_version == GENERATION_SESSION_CONTRACT_V3_SCHEMA_VERSION:
            result["session_mode"] = self.session_mode
        return result

    def to_dict(self) -> dict[str, Any]:
        return {**self._hash_payload(), "contract_hash": self.contract_hash}

    @classmethod
    def from_mapping(cls, value: Any) -> "GenerationSessionContract":
        if not isinstance(value, Mapping):
            raise ValueError("generation session contract fields mismatch")
        fields = {
            "schema_version",
            "session_id",
            "task_id",
            "backend_id",
            "width",
            "authoring_order",
            "ledger_record_schema",
            "max_generations",
            "max_trials",
            "max_train_jobs",
            "bound_contract_hashes",
            "contract_hash",
        }
        if value.get("schema_version") == GENERATION_SESSION_CONTRACT_V3_SCHEMA_VERSION:
            fields.add("session_mode")
        mapping = _exact_mapping(
            value,
            fields,
            "generation session contract",
        )
        order = mapping["authoring_order"]
        if not isinstance(order, list) or len(order) != 2:
            raise ValueError("generation authoring order is invalid")
        bound = mapping["bound_contract_hashes"]
        if not isinstance(bound, Mapping):
            raise TypeError("generation bound contract hashes must be an object")
        contract = cls(
            schema_version=mapping["schema_version"],
            session_id=mapping["session_id"],
            task_id=mapping["task_id"],
            backend_id=mapping["backend_id"],
            width=mapping["width"],
            authoring_order=(order[0], order[1]),
            ledger_record_schema=mapping["ledger_record_schema"],
            max_generations=mapping["max_generations"],
            max_trials=mapping["max_trials"],
            max_train_jobs=mapping["max_train_jobs"],
            bound_contract_hashes=dict(bound),
            session_mode=mapping.get("session_mode"),
        )
        if mapping["contract_hash"] != contract.contract_hash:
            raise ValueError("generation session contract hash mismatch")
        return contract


def publish_generation_session_contract(
    output_dir: Path,
    contract: GenerationSessionContract,
) -> GenerationSessionContractRef:
    if not isinstance(contract, GenerationSessionContract):
        raise TypeError("generation session contract is invalid")
    output_dir = Path(output_dir).resolve(strict=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / GENERATION_SESSION_CONTRACT_FILENAME
    payload = canonical_json_bytes(contract.to_dict(), trailing_newline=True)
    if len(payload) > _MAX_CONTRACT_BYTES:
        raise ValueError("generation session contract exceeds its byte limit")
    if path.exists():
        existing = read_bounded_regular_file(
            path,
            max_bytes=_MAX_CONTRACT_BYTES,
            label="generation session contract",
        )
        if existing != payload:
            raise FileExistsError("generation session contract already exists with different bytes")
    else:
        create_once_bytes(path, payload)
    return GenerationSessionContractRef(
        logical_name=GENERATION_SESSION_CONTRACT_FILENAME,
        checksum="sha256:" + hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
    )


def load_generation_session_contract(
    output_dir: Path,
    ref: GenerationSessionContractRef,
) -> GenerationSessionContract:
    if not isinstance(ref, GenerationSessionContractRef):
        raise TypeError("generation session contract ref is invalid")
    path = Path(output_dir).resolve(strict=False) / ref.logical_name
    payload = read_bounded_regular_file(
        path,
        max_bytes=_MAX_CONTRACT_BYTES,
        label="generation session contract",
    )
    if len(payload) != ref.size_bytes:
        raise ValueError("generation session contract size mismatch")
    checksum = "sha256:" + hashlib.sha256(payload).hexdigest()
    if checksum != ref.checksum:
        raise ValueError("generation session contract checksum mismatch")
    value = strict_json_object_loads(payload, context="generation session contract")
    if payload != canonical_json_bytes(value, trailing_newline=True):
        raise ValueError("generation session contract is not canonical JSON")
    return GenerationSessionContract.from_mapping(value)


@dataclass(frozen=True)
class GenerationLaneAssignment:
    lane_id: str
    execution_device: str
    resource_binding_hash: str | None
    lane_intent_hash: str
    gpu_process_binding: GpuProcessBinding | None = None

    def __post_init__(self) -> None:
        if self.lane_id not in GENERATION_LANE_IDS:
            raise ValueError("generation lane_id is invalid")
        require_execution_device(self.execution_device, "generation lane execution_device")
        if self.resource_binding_hash is not None:
            require_sha256_digest(
                self.resource_binding_hash,
                "generation lane resource_binding_hash",
            )
        require_sha256_digest(self.lane_intent_hash, "generation lane_intent_hash")
        if self.gpu_process_binding is not None:
            if not isinstance(self.gpu_process_binding, GpuProcessBinding):
                raise TypeError(
                    "generation lane gpu_process_binding must be a GpuProcessBinding"
                )
            if (
                self.execution_device
                != self.gpu_process_binding.process_execution_device
            ):
                raise ValueError(
                    "generation lane execution_device does not match GPU process binding"
                )
            if (
                self.resource_binding_hash
                != self.gpu_process_binding.resource_binding_hash
            ):
                raise ValueError(
                    "generation lane resource hash does not match GPU process binding"
                )

    def to_dict(self) -> dict[str, Any]:
        result = {
            "lane_id": self.lane_id,
            "execution_device": self.execution_device,
            "resource_binding_hash": self.resource_binding_hash,
            "lane_intent_hash": self.lane_intent_hash,
        }
        if self.gpu_process_binding is not None:
            result["gpu_process_binding"] = self.gpu_process_binding.to_dict()
        return result

    def to_public_dict(self) -> dict[str, Any]:
        """Project one lane without exposing its host-device assignment."""

        result = {
            "lane_id": self.lane_id,
            "execution_device": self.execution_device,
            "resource_binding_hash": self.resource_binding_hash,
            "lane_intent_hash": self.lane_intent_hash,
        }
        if self.gpu_process_binding is not None:
            result["gpu_process_binding"] = (
                self.gpu_process_binding.to_public_dict()
            )
        return result

    @classmethod
    def from_mapping(cls, value: Any) -> "GenerationLaneAssignment":
        fields = {
            "lane_id",
            "execution_device",
            "resource_binding_hash",
            "lane_intent_hash",
        }
        if not isinstance(value, Mapping):
            raise ValueError("generation lane assignment fields mismatch")
        raw = dict(value)
        if set(raw) == fields:
            return cls(**raw)
        if set(raw) != fields | {"gpu_process_binding"}:
            raise ValueError("generation lane assignment fields mismatch")
        raw["gpu_process_binding"] = GpuProcessBinding.from_mapping(
            raw["gpu_process_binding"]
        )
        return cls(**raw)


@dataclass(frozen=True)
class GenerationManifest:
    generation_id: str
    generation_index: int
    session_genesis_hash: str
    generation_session_contract_hash: str
    ledger_epoch_hash: str
    parent_program_source_checksum: str
    parent_checkpoint_hash: str
    current_best_evidence_hash: str
    base_public_context_hash: str
    evaluation_binding_hash: str
    training_initialization_hash: str
    guard_reference_evidence_hash: str
    guard_reference_phase_receipt_hash: str
    lanes: tuple[GenerationLaneAssignment, GenerationLaneAssignment]
    schema_version: str = GENERATION_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version not in {
            GENERATION_MANIFEST_SCHEMA_VERSION,
            GENERATION_MANIFEST_V2_SCHEMA_VERSION,
        }:
            raise ValueError("generation manifest schema mismatch")
        _bounded_text(self.generation_id, "generation_id")
        _positive_int(self.generation_index, "generation_index")
        for field_name in (
            "session_genesis_hash",
            "generation_session_contract_hash",
            "ledger_epoch_hash",
            "current_best_evidence_hash",
            "base_public_context_hash",
            "evaluation_binding_hash",
            "training_initialization_hash",
            "guard_reference_evidence_hash",
            "guard_reference_phase_receipt_hash",
        ):
            require_sha256_digest(getattr(self, field_name), f"generation manifest {field_name}")
        require_sha256_checksum(
            self.parent_program_source_checksum,
            "generation manifest parent_program_source_checksum",
        )
        require_sha256_checksum(
            self.parent_checkpoint_hash,
            "generation manifest parent_checkpoint_hash",
        )
        if (
            not isinstance(self.lanes, tuple)
            or len(self.lanes) != GENERATION_WIDTH
            or tuple(item.lane_id for item in self.lanes) != GENERATION_LANE_IDS
        ):
            raise ValueError("generation manifest lanes must be ordered lane-a/lane-b")
        if self.schema_version == GENERATION_MANIFEST_SCHEMA_VERSION:
            if any(item.gpu_process_binding is not None for item in self.lanes):
                raise ValueError("generation manifest v1 cannot carry GPU process bindings")
            cuda_lanes = [
                item for item in self.lanes if item.execution_device.startswith("cuda:")
            ]
            if cuda_lanes:
                if len(cuda_lanes) != GENERATION_WIDTH:
                    raise ValueError("generation GPU lanes cannot mix CUDA and non-CUDA")
                if any(item.resource_binding_hash is None for item in cuda_lanes):
                    raise ValueError("generation CUDA lanes require resource bindings")
                if len({item.execution_device for item in cuda_lanes}) != GENERATION_WIDTH:
                    raise ValueError("generation CUDA lanes must use distinct devices")
                if len({item.resource_binding_hash for item in cuda_lanes}) != GENERATION_WIDTH:
                    raise ValueError("generation CUDA lanes must bind distinct GPUs")
        else:
            bindings = tuple(item.gpu_process_binding for item in self.lanes)
            if not all(isinstance(item, GpuProcessBinding) for item in bindings):
                raise ValueError(
                    "generation manifest v2 requires every lane GPU process binding"
                )
            bound_lanes = tuple(
                item for item in bindings if isinstance(item, GpuProcessBinding)
            )
            if (
                len({item.host_execution_device for item in bound_lanes})
                != GENERATION_WIDTH
            ):
                raise ValueError(
                    "generation CUDA lanes must use distinct host devices"
                )
            if (
                len({item.resource_binding_hash for item in bound_lanes})
                != GENERATION_WIDTH
            ):
                raise ValueError("generation CUDA lanes must bind distinct GPUs")

    @property
    def manifest_hash(self) -> str:
        return canonical_json_hash(self._hash_payload())

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generation_id": self.generation_id,
            "generation_index": self.generation_index,
            "session_genesis_hash": self.session_genesis_hash,
            "generation_session_contract_hash": self.generation_session_contract_hash,
            "ledger_epoch_hash": self.ledger_epoch_hash,
            "parent_program_source_checksum": self.parent_program_source_checksum,
            "parent_checkpoint_hash": self.parent_checkpoint_hash,
            "current_best_evidence_hash": self.current_best_evidence_hash,
            "base_public_context_hash": self.base_public_context_hash,
            "evaluation_binding_hash": self.evaluation_binding_hash,
            "training_initialization_hash": self.training_initialization_hash,
            "guard_reference_evidence_hash": self.guard_reference_evidence_hash,
            "guard_reference_phase_receipt_hash": (
                self.guard_reference_phase_receipt_hash
            ),
            "lanes": [lane.to_dict() for lane in self.lanes],
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._hash_payload(), "manifest_hash": self.manifest_hash}

    def to_public_dict(self) -> dict[str, Any]:
        """Project the hash-bound manifest without private host topology."""

        return {
            **self._hash_payload(),
            "lanes": [lane.to_public_dict() for lane in self.lanes],
            "manifest_hash": self.manifest_hash,
        }

    @classmethod
    def from_mapping(cls, value: Any) -> "GenerationManifest":
        mapping = _exact_mapping(
            value,
            {
                "schema_version",
                "generation_id",
                "generation_index",
                "session_genesis_hash",
                "generation_session_contract_hash",
                "ledger_epoch_hash",
                "parent_program_source_checksum",
                "parent_checkpoint_hash",
                "current_best_evidence_hash",
                "base_public_context_hash",
                "evaluation_binding_hash",
                "training_initialization_hash",
                "guard_reference_evidence_hash",
                "guard_reference_phase_receipt_hash",
                "lanes",
                "manifest_hash",
            },
            "generation manifest",
        )
        raw_lanes = mapping["lanes"]
        if not isinstance(raw_lanes, list) or len(raw_lanes) != GENERATION_WIDTH:
            raise ValueError("generation manifest lanes are invalid")
        manifest = cls(
            schema_version=mapping["schema_version"],
            generation_id=mapping["generation_id"],
            generation_index=mapping["generation_index"],
            session_genesis_hash=mapping["session_genesis_hash"],
            generation_session_contract_hash=mapping[
                "generation_session_contract_hash"
            ],
            ledger_epoch_hash=mapping["ledger_epoch_hash"],
            parent_program_source_checksum=mapping[
                "parent_program_source_checksum"
            ],
            parent_checkpoint_hash=mapping["parent_checkpoint_hash"],
            current_best_evidence_hash=mapping["current_best_evidence_hash"],
            base_public_context_hash=mapping["base_public_context_hash"],
            evaluation_binding_hash=mapping["evaluation_binding_hash"],
            training_initialization_hash=mapping["training_initialization_hash"],
            guard_reference_evidence_hash=mapping[
                "guard_reference_evidence_hash"
            ],
            guard_reference_phase_receipt_hash=mapping[
                "guard_reference_phase_receipt_hash"
            ],
            lanes=tuple(
                GenerationLaneAssignment.from_mapping(item) for item in raw_lanes
            ),
        )
        if mapping["manifest_hash"] != manifest.manifest_hash:
            raise ValueError("generation manifest hash mismatch")
        return manifest


def require_generation_trial_status(value: Any) -> str:
    if value not in GENERATION_TRIAL_STATUSES:
        raise ValueError("generation trial_status is invalid")
    return str(value)


def require_generation_lane_intent(value: Any) -> dict[str, Any]:
    """Validate one small public directive without interpreting task semantics."""

    if not isinstance(value, Mapping) or not value:
        raise ValueError("generation lane intent must be a non-empty object")
    normalized = dict(value)
    if len(canonical_json_bytes(normalized)) > MAX_GENERATION_LANE_INTENT_BYTES:
        raise ValueError("generation lane intent exceeds its byte limit")
    return normalized


def _exact_mapping(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{label} fields mismatch")
    return dict(value)


def _bounded_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 128:
        raise ValueError(f"{label} must be a non-empty string of at most 128 chars")
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


__all__ = [
    "GENERATION_AUTHORING_ORDER",
    "GENERATION_LANE_IDS",
    "GENERATION_LEDGER_RECORD_SCHEMA_VERSION",
    "GENERATION_PUBLIC_HISTORY_PROJECTION_SCHEMA_VERSION",
    "GENERATION_MANIFEST_SCHEMA_VERSION",
    "GENERATION_MANIFEST_V2_SCHEMA_VERSION",
    "MAX_GENERATION_LANE_INTENT_BYTES",
    "GENERATION_SESSION_CONTRACT_FILENAME",
    "GENERATION_SESSION_CONTRACT_REF_SCHEMA_VERSION",
    "GENERATION_SESSION_CONTRACT_SCHEMA_VERSION",
    "GENERATION_SESSION_CONTRACT_V3_SCHEMA_VERSION",
    "GENERATION_SESSION_MODES",
    "GENERATION_TRIAL_STATUSES",
    "GENERATION_WIDTH",
    "GenerationLaneAssignment",
    "GenerationManifest",
    "GenerationSessionContract",
    "GenerationSessionContractRef",
    "generation_resource_plan_hash",
    "generation_training_initialization_hash",
    "load_generation_session_contract",
    "publish_generation_session_contract",
    "require_generation_lane_intent",
    "require_generation_trial_status",
]
