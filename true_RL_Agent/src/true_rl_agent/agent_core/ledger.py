"""Append-only public ledger helpers."""

from __future__ import annotations

import os
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from true_rl_agent.agent_core.generation_contract import (
    GENERATION_LEDGER_RECORD_SCHEMA_VERSION,
    GENERATION_LEDGER_RECORD_V2_SCHEMA_VERSION,
    GENERATION_LEDGER_RECORD_V3_SCHEMA_VERSION,
)

from true_rl_agent.agent_core.process_lock import acquire_process_file_lock
from true_rl_agent.evaluation.binding import (
    canonical_json_hash,
    require_sha256_digest,
)
from true_rl_agent.fs_contract import atomic_replace_bytes
from true_rl_agent.json_contract import (
    canonical_json_text,
    strict_json_object_loads,
)

LEDGER_RECORD_SCHEMA_V2 = "true_rl_agent.ledger_record.v2"
LEDGER_RECORD_SCHEMA_V3 = "true_rl_agent.ledger_record.v3"
LEDGER_RECORD_SCHEMA_V4 = "true_rl_agent.ledger_record.v4"
LEDGER_RECORD_SCHEMA_V5 = "true_rl_agent.ledger_record.v5"
LEDGER_RECORD_SCHEMA_V6 = "true_rl_agent.ledger_record.v6"
LEDGER_RECORD_SCHEMA_VERSION = "true_rl_agent.ledger_record.v7"
PRIMARY_LEDGER_EPOCH_SCHEMA_VERSION = "true_rl_agent.primary_ledger_epoch.v1"
LEGACY_UNVERSIONED_SCHEMA = "legacy_unversioned"
PROVIDER_AUTHORING_MODE = "provider_authoring"
NON_PROVIDER_AUTHORING_MODE = "non_provider"
LEGACY_UNVERIFIED_AUTHORING_MODE = "legacy_unverified"
LEDGER_AUTHORING_MODES = frozenset(
    {PROVIDER_AUTHORING_MODE, NON_PROVIDER_AUTHORING_MODE}
)


@dataclass(frozen=True)
class LedgerRecordSchemaDescriptor:
    schema_version: str
    authoring_mode: str
    authoring_evidence_verifiable: bool
    ledger_epoch_bound: bool
    sealed_authoring_projection_bound: bool
    mechanism_delta_bound: bool
    round_authoring_anchor_bound: bool
    current: bool
    record_kind: str = "round"


def dispatch_ledger_record_schema(
    record: Mapping[str, Any] | Any,
) -> LedgerRecordSchemaDescriptor:
    """Parse one supported ledger schema without unknown-version fallback."""

    if not isinstance(record, Mapping):
        raise TypeError("ledger record must be an object")
    schema_version = record.get("schema_version")
    if schema_version is None:
        return LedgerRecordSchemaDescriptor(
            schema_version=LEGACY_UNVERSIONED_SCHEMA,
            authoring_mode=NON_PROVIDER_AUTHORING_MODE,
            authoring_evidence_verifiable=True,
            ledger_epoch_bound=False,
            sealed_authoring_projection_bound=False,
            mechanism_delta_bound=False,
            round_authoring_anchor_bound=False,
            current=False,
        )
    if schema_version == LEDGER_RECORD_SCHEMA_V2:
        return LedgerRecordSchemaDescriptor(
            schema_version=LEDGER_RECORD_SCHEMA_V2,
            authoring_mode=LEGACY_UNVERIFIED_AUTHORING_MODE,
            authoring_evidence_verifiable=False,
            ledger_epoch_bound=False,
            sealed_authoring_projection_bound=False,
            mechanism_delta_bound=False,
            round_authoring_anchor_bound=False,
            current=False,
        )
    if schema_version == LEDGER_RECORD_SCHEMA_V3:
        raise ValueError(
            "ledger record v3 is unsupported because authoring mode is ambiguous"
        )
    if schema_version == LEDGER_RECORD_SCHEMA_V4:
        return LedgerRecordSchemaDescriptor(
            schema_version=LEDGER_RECORD_SCHEMA_V4,
            authoring_mode=require_ledger_authoring_mode(
                record.get("authoring_mode")
            ),
            authoring_evidence_verifiable=True,
            ledger_epoch_bound=False,
            sealed_authoring_projection_bound=False,
            mechanism_delta_bound=False,
            round_authoring_anchor_bound=False,
            current=False,
        )
    if schema_version == LEDGER_RECORD_SCHEMA_V5:
        return LedgerRecordSchemaDescriptor(
            schema_version=LEDGER_RECORD_SCHEMA_V5,
            authoring_mode=require_ledger_authoring_mode(
                record.get("authoring_mode")
            ),
            authoring_evidence_verifiable=True,
            ledger_epoch_bound=True,
            sealed_authoring_projection_bound=True,
            mechanism_delta_bound=False,
            round_authoring_anchor_bound=False,
            current=False,
        )
    if schema_version == LEDGER_RECORD_SCHEMA_V6:
        return LedgerRecordSchemaDescriptor(
            schema_version=LEDGER_RECORD_SCHEMA_V6,
            authoring_mode=require_ledger_authoring_mode(
                record.get("authoring_mode")
            ),
            authoring_evidence_verifiable=True,
            ledger_epoch_bound=True,
            sealed_authoring_projection_bound=True,
            mechanism_delta_bound=True,
            round_authoring_anchor_bound=False,
            current=False,
        )
    if schema_version in {GENERATION_LEDGER_RECORD_SCHEMA_VERSION, GENERATION_LEDGER_RECORD_V2_SCHEMA_VERSION, GENERATION_LEDGER_RECORD_V3_SCHEMA_VERSION}:
        return LedgerRecordSchemaDescriptor(
            schema_version=schema_version,
            authoring_mode=PROVIDER_AUTHORING_MODE,
            authoring_evidence_verifiable=True,
            ledger_epoch_bound=True,
            sealed_authoring_projection_bound=True,
            mechanism_delta_bound=True,
            round_authoring_anchor_bound=True,
            current=True,
            record_kind="generation",
        )
    if schema_version != LEDGER_RECORD_SCHEMA_VERSION:
        raise ValueError(f"unsupported ledger record schema: {schema_version!r}")
    return LedgerRecordSchemaDescriptor(
        schema_version=LEDGER_RECORD_SCHEMA_VERSION,
        authoring_mode=require_ledger_authoring_mode(record.get("authoring_mode")),
        authoring_evidence_verifiable=True,
        ledger_epoch_bound=True,
        sealed_authoring_projection_bound=True,
        mechanism_delta_bound=True,
        round_authoring_anchor_bound=True,
        current=True,
    )


def primary_ledger_epoch_hash(session_genesis_hash: str) -> str:
    """Derive the immutable identity of a Genesis-owned primary ledger."""

    return canonical_json_hash(
        {
            "schema_version": PRIMARY_LEDGER_EPOCH_SCHEMA_VERSION,
            "session_genesis_hash": require_sha256_digest(
                session_genesis_hash,
                "primary ledger session Genesis hash",
            ),
        }
    )


def require_ledger_record_epoch_hash(record: Mapping[str, Any]) -> str:
    descriptor = dispatch_ledger_record_schema(record)
    if not descriptor.ledger_epoch_bound:
        raise ValueError("ledger record schema does not bind a ledger epoch")
    return require_sha256_digest(
        record.get("ledger_epoch_hash"),
        "ledger record epoch hash",
    )


def require_ledger_authoring_mode(value: Any) -> str:
    if value not in LEDGER_AUTHORING_MODES:
        raise ValueError(
            "ledger authoring_mode must be provider_authoring or non_provider"
        )
    return str(value)


def is_strict_ledger_record(record: Mapping[str, Any] | Any) -> bool:
    """Return strict scientific-contract status for a supported schema."""

    if not isinstance(record, Mapping):
        return False
    descriptor = dispatch_ledger_record_schema(record)
    return descriptor.schema_version != LEGACY_UNVERSIONED_SCHEMA


def append_ledger_record(path: Path, record: dict[str, Any]) -> None:
    _append_ledger_record(path, record, idempotent_identity=False)


def append_ledger_record_once(
    path: Path,
    record: dict[str, Any],
    *,
    pre_append_validator: Callable[
        [Sequence[Mapping[str, Any]], Mapping[str, Any]], None
    ]
    | None = None,
) -> bool:
    """Append one current scientific record, or verify its prior exact commit.

    Returns ``True`` only when this call appended bytes. A prior byte-equivalent
    record with the same commit-slot identity is a successful no-op; a conflicting
    record is rejected while the ledger write lock is held.
    """

    return _append_ledger_record(
        path,
        record,
        idempotent_identity=True,
        pre_append_validator=pre_append_validator,
    )


def _append_ledger_record(
    path: Path,
    record: dict[str, Any],
    *,
    idempotent_identity: bool,
    pre_append_validator: Callable[
        [Sequence[Mapping[str, Any]], Mapping[str, Any]], None
    ]
    | None = None,
) -> bool:
    descriptor = dispatch_ledger_record_schema(record)
    if not descriptor.current:
        raise ValueError("new ledger records must use the current schema")
    require_ledger_record_epoch_hash(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError("ledger path must be a regular file")
    lock_payload = (
        canonical_json_text(
            {
                "schema_version": "true_rl_agent.ledger_write_lock.v1",
                "process_id": os.getpid(),
                "owner_token": uuid.uuid4().hex,
            }
        )
        + "\n"
    ).encode("utf-8")
    lock_path = path.with_name(f".{path.name}.write.lock")
    with acquire_process_file_lock(
        lock_path,
        blocking=True,
        payload=lock_payload,
    ):
        existing = path.read_bytes() if path.is_file() else b""
        if existing and not existing.endswith(b"\n"):
            raise ValueError("existing ledger has an incomplete final record")
        if existing:
            records = load_ledger(path)
        else:
            records = []
        existing_record_kinds = {
            dispatch_ledger_record_schema(item).record_kind for item in records
        }
        if existing_record_kinds and existing_record_kinds != {
            descriptor.record_kind
        }:
            raise ValueError("ledger record kinds cannot be mixed in one epoch")
        if idempotent_identity:
            identity = _ledger_commit_identity(record)
            matches = [
                item
                for item in records
                if dispatch_ledger_record_schema(item).current
                and _ledger_commit_identity(item) == identity
            ]
            if len(matches) > 1:
                raise ValueError("ledger contains duplicate commit identities")
            if matches:
                if matches[0] != record:
                    first_difference = _first_record_difference(
                        matches[0],
                        record,
                    )
                    raise ValueError(
                        "ledger commit identity conflicts with existing record; "
                        f"first_difference={first_difference}; "
                        "existing_record_hash="
                        f"{canonical_json_hash(matches[0])}; "
                        f"incoming_record_hash={canonical_json_hash(record)}"
                    )
                return False
        if pre_append_validator is not None:
            pre_append_validator(tuple(records), record)
        encoded = (canonical_json_text(record) + "\n").encode("utf-8")
        atomic_replace_bytes(path, existing + encoded)
        return True


def _ledger_commit_identity(record: Mapping[str, Any]) -> tuple[str, str, str, str]:
    descriptor = dispatch_ledger_record_schema(record)
    if not descriptor.current:
        raise ValueError("idempotent commit requires the current ledger schema")
    identity_field = (
        "generation_id" if descriptor.record_kind == "generation" else "round_id"
    )
    record_id = record.get(identity_field)
    if not isinstance(record_id, str) or not record_id:
        raise ValueError(f"idempotent commit requires a {identity_field}")
    session_genesis_hash = require_sha256_digest(
        record.get("session_genesis_hash"),
        "ledger commit session Genesis hash",
    )
    ledger_epoch_hash = require_ledger_record_epoch_hash(record)
    # Candidate and parent are deliberately outside the commit-slot identity.
    # The bound execution receipt must reproduce the same round or generation;
    # any drift conflicts instead of creating a second row in the same slot.
    return (
        descriptor.schema_version,
        session_genesis_hash,
        ledger_epoch_hash,
        record_id,
    )


def _first_record_difference(
    existing: Mapping[str, Any],
    incoming: Mapping[str, Any],
) -> str:
    for field_name in sorted(set(existing).union(incoming)):
        if existing.get(field_name) != incoming.get(field_name):
            return field_name
    return "unknown"


def load_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = strict_json_object_loads(
                line,
                context=f"ledger line {line_number}",
            )
        except ValueError as exc:
            raise ValueError(f"invalid ledger JSON at line {line_number}: {path}") from exc
        try:
            descriptor = dispatch_ledger_record_schema(value)
            if descriptor.schema_version == LEGACY_UNVERSIONED_SCHEMA:
                raise ValueError("persisted ledger record has no schema_version")
            if descriptor.ledger_epoch_bound:
                require_ledger_record_epoch_hash(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid ledger schema at line {line_number}: {path}"
            ) from exc
        records.append(value)
    return records
