"""Append-only lifecycle evidence for one fixed-width generation session.

This module deliberately owns operational continuation only.  It never stores
candidate metrics, decisions, or current-best state; those remain in the
generation ledger and its replay validators.
"""

from __future__ import annotations

import ctypes
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from true_rl_agent.agent_core.generation_contract import (
    GENERATION_LEDGER_RECORD_SCHEMA_VERSION,
    GenerationSessionContract,
)
from true_rl_agent.agent_core.ledger import PROVIDER_AUTHORING_MODE, load_ledger
from true_rl_agent.agent_core.process_lock import ProcessFileLock, acquire_process_file_lock
from true_rl_agent.evaluation.acceptance_protocol import AcceptanceProtocol
from true_rl_agent.evaluation.acceptance_replay import replay_acceptance_ledger
from true_rl_agent.evaluation.binding import canonical_json_hash, require_sha256_digest
from true_rl_agent.evaluation.current_best import CurrentBestEvalStats
from true_rl_agent.evaluation.public_projection import (
    assert_public_projection_safe,
    project_public_runner_error,
)
from true_rl_agent.fs_contract import create_once_bytes, read_bounded_regular_file
from true_rl_agent.json_contract import canonical_json_bytes, strict_json_object_loads
from true_rl_agent.runner.job_attempts import (
    JobAttemptJournal,
    PreGenesisExecutionScope,
)


GENERATION_CONTINUATION_POLICY_SCHEMA_VERSION = (
    "true_rl_agent.generation_continuation_policy.v1"
)
GENERATION_CONTINUATION_POLICY_V2_SCHEMA_VERSION = (
    "true_rl_agent.generation_continuation_policy.v2"
)
GENERATION_SEGMENT_START_SCHEMA_VERSION = (
    "true_rl_agent.generation_segment.start.v2"
)
LEGACY_GENERATION_SEGMENT_EXIT_SCHEMA_VERSION = (
    "true_rl_agent.generation_segment.exit.v1"
)
GENERATION_SEGMENT_EXIT_SCHEMA_VERSION = (
    "true_rl_agent.generation_segment.exit.v2"
)
GENERATION_UNCLEAN_RECOVERY_SCHEMA_VERSION = (
    "true_rl_agent.generation_segment.unclean_recovery.v2"
)
GENERATION_CONTINUATION_POLICY_FILENAME = "generation_continuation_policy.json"
_MAX_RECEIPT_BYTES = 128 * 1024
_MAX_POLICY_BYTES = 64 * 1024
_MAX_SEGMENTS = 32
_MAX_EXCEPTION_TYPE_CHARS = 128
_MAX_EXCEPTION_MESSAGE_CHARS = 512
_SEGMENT_DIR_PATTERN = re.compile(r"segment-(?P<index>[0-9]{4})-(?P<id>[0-9a-f]{32})\Z")
_EXIT_STATUSES = frozenset({"completed", "paused", "failed"})
_EXIT_REASONS = frozenset(
    {
        "normal",
        "operator_pause",
        "typed_infrastructure_failure",
        "exception",
        "budget_exhausted",
        "contract_incomplete",
    }
)


@dataclass(frozen=True)
class GenerationContinuationPolicy:
    max_segments: int
    allowed_continuation_reasons: tuple[str, ...]
    allow_unclean_recovery: bool = True
    max_logical_job_infrastructure_retries: int = 1
    exclusive_session_lease: bool = True
    operator_pause_generation_boundaries: tuple[int, ...] = ()
    schema_version: str = GENERATION_CONTINUATION_POLICY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version not in {
            GENERATION_CONTINUATION_POLICY_SCHEMA_VERSION,
            GENERATION_CONTINUATION_POLICY_V2_SCHEMA_VERSION,
        }:
            raise ValueError("generation continuation policy schema mismatch")
        if (
            isinstance(self.max_segments, bool)
            or not isinstance(self.max_segments, int)
            or not 1 <= self.max_segments <= _MAX_SEGMENTS
        ):
            raise ValueError("generation continuation max_segments is invalid")
        if self.max_logical_job_infrastructure_retries != 1:
            raise ValueError("generation continuation permits exactly one infra retry")
        if self.allow_unclean_recovery is not True:
            raise ValueError("generation continuation must declare unclean recovery")
        if self.exclusive_session_lease is not True:
            raise ValueError("generation continuation requires an exclusive lease")
        reasons = tuple(self.allowed_continuation_reasons)
        if (
            not reasons
            or len(set(reasons)) != len(reasons)
            or any(reason not in _EXIT_REASONS - {"normal"} for reason in reasons)
        ):
            raise ValueError("generation continuation reasons are invalid")
        object.__setattr__(self, "allowed_continuation_reasons", reasons)
        boundaries = tuple(self.operator_pause_generation_boundaries)
        if self.schema_version == GENERATION_CONTINUATION_POLICY_SCHEMA_VERSION:
            if boundaries:
                raise ValueError(
                    "generation continuation v1 cannot declare pause boundaries"
                )
        elif (
            not boundaries
            or tuple(sorted(set(boundaries))) != boundaries
            or any(
                isinstance(boundary, bool)
                or not isinstance(boundary, int)
                or boundary <= 0
                for boundary in boundaries
            )
            or "operator_pause" not in reasons
        ):
            raise ValueError(
                "generation continuation v2 pause boundaries are invalid"
            )
        object.__setattr__(
            self,
            "operator_pause_generation_boundaries",
            boundaries,
        )

    @property
    def policy_hash(self) -> str:
        return canonical_json_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "max_segments": self.max_segments,
            "allowed_continuation_reasons": list(
                self.allowed_continuation_reasons
            ),
            "allow_unclean_recovery": self.allow_unclean_recovery,
            "max_logical_job_infrastructure_retries": (
                self.max_logical_job_infrastructure_retries
            ),
            "exclusive_session_lease": self.exclusive_session_lease,
        }
        if self.schema_version == GENERATION_CONTINUATION_POLICY_V2_SCHEMA_VERSION:
            result["operator_pause_generation_boundaries"] = list(
                self.operator_pause_generation_boundaries
            )
        return result

    @classmethod
    def from_mapping(cls, value: Any) -> "GenerationContinuationPolicy":
        if not isinstance(value, Mapping):
            raise ValueError("generation continuation policy must be a mapping")
        schema_version = value.get("schema_version")
        fields = {
            "schema_version",
            "max_segments",
            "allowed_continuation_reasons",
            "allow_unclean_recovery",
            "max_logical_job_infrastructure_retries",
            "exclusive_session_lease",
        }
        if schema_version == GENERATION_CONTINUATION_POLICY_V2_SCHEMA_VERSION:
            fields.add("operator_pause_generation_boundaries")
        mapping = _exact_mapping(
            value,
            fields,
            "generation continuation policy",
        )
        reasons = mapping["allowed_continuation_reasons"]
        if not isinstance(reasons, list):
            raise ValueError("generation continuation reasons must be a list")
        return cls(
            schema_version=mapping["schema_version"],
            max_segments=mapping["max_segments"],
            allowed_continuation_reasons=tuple(reasons),
            allow_unclean_recovery=mapping["allow_unclean_recovery"],
            max_logical_job_infrastructure_retries=mapping[
                "max_logical_job_infrastructure_retries"
            ],
            exclusive_session_lease=mapping["exclusive_session_lease"],
            operator_pause_generation_boundaries=tuple(
                mapping.get("operator_pause_generation_boundaries", ())
            ),
        )


@dataclass(frozen=True)
class GenerationLifecycleIdentity:
    session_id: str
    session_genesis_hash: str
    generation_session_contract_hash: str
    final_confirmation_plan_hash: str
    ledger_epoch_hash: str
    output_root: str

    def __post_init__(self) -> None:
        _bounded_text(self.session_id, "generation lifecycle session_id")
        for field_name in (
            "session_genesis_hash",
            "generation_session_contract_hash",
            "final_confirmation_plan_hash",
            "ledger_epoch_hash",
        ):
            require_sha256_digest(
                getattr(self, field_name),
                f"generation lifecycle {field_name}",
            )
        root = Path(self.output_root)
        if not root.is_absolute():
            raise ValueError("generation lifecycle output_root must be absolute")
        object.__setattr__(self, "output_root", str(root.resolve(strict=False)))

    @property
    def identity_hash(self) -> str:
        return canonical_json_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "session_genesis_hash": self.session_genesis_hash,
            "generation_session_contract_hash": (
                self.generation_session_contract_hash
            ),
            "final_confirmation_plan_hash": self.final_confirmation_plan_hash,
            "ledger_epoch_hash": self.ledger_epoch_hash,
            "output_root": self.output_root,
        }

    @classmethod
    def from_mapping(cls, value: Any) -> "GenerationLifecycleIdentity":
        return cls(
            **_exact_mapping(
                value,
                {
                    "session_id",
                    "session_genesis_hash",
                    "generation_session_contract_hash",
                    "final_confirmation_plan_hash",
                    "ledger_epoch_hash",
                    "output_root",
                },
                "generation lifecycle identity",
            )
        )


@dataclass(frozen=True)
class GenerationLifecycleAudit:
    identity: GenerationLifecycleIdentity
    continuation_policy_hash: str
    segment_count: int
    closed_generation_count: int
    lifecycle_head_hash: str | None
    cost_chain_head_hash: str
    active_segment_id: str | None
    terminal: bool
    continuation_eligible: bool
    receipt_hashes: tuple[str, ...]
    recovery_cost_edges: tuple[tuple[str, str, str], ...]


@dataclass(frozen=True)
class GenerationContinuationCursor:
    """Replay-owned point from which the next generation may start."""

    current_best: CurrentBestEvalStats
    next_generation_index: int
    ledger_record_count: int
    ledger_snapshot_hash: str
    current_head_hash: str
    lifecycle_head_hash: str | None
    cost_chain_head_hash: str
    active_segment_id: str | None

    def __post_init__(self) -> None:
        if self.next_generation_index != self.ledger_record_count + 1:
            raise ValueError("generation continuation cursor index mismatch")
        for value, label in (
            (self.ledger_snapshot_hash, "cursor ledger snapshot"),
            (self.current_head_hash, "cursor current head"),
            (self.cost_chain_head_hash, "cursor cost chain head"),
        ):
            require_sha256_digest(value, label)
        if self.lifecycle_head_hash is not None:
            require_sha256_digest(
                self.lifecycle_head_hash,
                "cursor lifecycle head",
            )
        if canonical_json_hash(self.current_best.to_public_dict()) != self.current_head_hash:
            raise ValueError("generation continuation cursor current head mismatch")


@dataclass(frozen=True)
class GenerationTerminalBinding:
    """Task-neutral, replayable input for a task-owned terminal reservation."""

    identity_hash: str
    required_generation_count: int
    closed_generation_count: int
    ledger_snapshot_hash: str
    current_head_hash: str
    lifecycle_head_hash: str
    cost_chain_head_hash: str
    segment_count: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.required_generation_count, bool)
            or not isinstance(self.required_generation_count, int)
            or self.required_generation_count <= 0
            or self.closed_generation_count != self.required_generation_count
        ):
            raise ValueError("generation terminal count mismatch")
        if (
            isinstance(self.segment_count, bool)
            or not isinstance(self.segment_count, int)
            or self.segment_count <= 0
        ):
            raise ValueError("generation terminal segment count is invalid")
        for value, label in (
            (self.identity_hash, "terminal identity"),
            (self.ledger_snapshot_hash, "terminal ledger snapshot"),
            (self.current_head_hash, "terminal current head"),
            (self.lifecycle_head_hash, "terminal lifecycle head"),
            (self.cost_chain_head_hash, "terminal cost chain head"),
        ):
            require_sha256_digest(value, label)

    @property
    def binding_hash(self) -> str:
        return canonical_json_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "true_rl_agent.generation_terminal_binding.v1",
            "identity_hash": self.identity_hash,
            "required_generation_count": self.required_generation_count,
            "closed_generation_count": self.closed_generation_count,
            "ledger_snapshot_hash": self.ledger_snapshot_hash,
            "current_head_hash": self.current_head_hash,
            "lifecycle_head_hash": self.lifecycle_head_hash,
            "cost_chain_head_hash": self.cost_chain_head_hash,
            "segment_count": self.segment_count,
            "acceptance_authority": False,
        }


@dataclass
class GenerationSegmentJournal:
    """One process-held segment in a generation session lifecycle chain."""

    output_dir: Path
    identity: GenerationLifecycleIdentity
    policy: GenerationContinuationPolicy
    segment_id: str
    continuation_index: int
    segment_dir: Path
    start_record: Mapping[str, Any]
    start_hash: str
    cost_chain_head_hash: str
    _lease: ProcessFileLock

    @classmethod
    def start(
        cls,
        *,
        output_dir: Path,
        identity: GenerationLifecycleIdentity,
        policy: GenerationContinuationPolicy,
        session_contract: GenerationSessionContract,
        closed_generation_count: int,
        cost_chain_head_hash: str,
        attempt_journal_root: Path,
        process_group_alive: Callable[[int], bool] | None = None,
        recovery_process_incarnation: Callable[[int], str | None] | None = None,
    ) -> "GenerationSegmentJournal":
        output_root = Path(output_dir).resolve(strict=False)
        if str(output_root) != identity.output_root:
            raise ValueError("generation lifecycle output root mismatch")
        _require_nonnegative_int(closed_generation_count, "closed generation count")
        require_sha256_digest(cost_chain_head_hash, "generation cost chain head")
        _require_policy_binding(policy, session_contract, identity)
        output_root.mkdir(parents=True, exist_ok=True)
        lease_id = uuid.uuid4().hex
        lease_payload = canonical_json_bytes(
            {
                "schema_version": "true_rl_agent.generation_session_lease.v1",
                "identity_hash": identity.identity_hash,
                "lease_id": lease_id,
                "process_id": os.getpid(),
            },
            trailing_newline=True,
        )
        lease = acquire_process_file_lock(
            output_root / ".generation_session.lease",
            blocking=False,
            payload=lease_payload,
        )
        try:
            _reconcile_published_recoveries(
                output_dir=output_root,
                identity=identity,
                policy=policy,
            )
            audit = audit_generation_lifecycle(
                output_dir=output_root,
                identity=identity,
                policy=policy,
                require_closed=False,
            )
            if audit.segment_count == 0 and closed_generation_count != 0:
                raise ValueError("generation lifecycle must start from an empty ledger")
            if audit.segment_count > 0:
                if audit.active_segment_id is None:
                    if audit.closed_generation_count != closed_generation_count:
                        raise ValueError("generation lifecycle ledger height mismatch")
                    if audit.cost_chain_head_hash != cost_chain_head_hash:
                        cost_chain_head_hash = _advance_cost_head_across_recovery(
                            output_dir=output_root,
                            identity=identity,
                            policy=policy,
                            lifecycle=audit,
                            supplied_cost_head=cost_chain_head_hash,
                        )
                elif audit.closed_generation_count > closed_generation_count:
                    raise ValueError(
                        "generation unclean recovery ledger height regressed"
                    )
            if (
                audit.segment_count > 0
                and audit.active_segment_id is None
                and not audit.continuation_eligible
            ):
                raise ValueError("generation lifecycle is not continuation eligible")
            previous_hash = audit.lifecycle_head_hash
            if audit.active_segment_id is not None:
                previous_hash, cost_chain_head_hash = _recover_unclean_segment(
                    output_dir=output_root,
                    identity=identity,
                    policy=policy,
                    active_segment_id=audit.active_segment_id,
                    previous_lifecycle_head_hash=previous_hash,
                    closed_generation_count=closed_generation_count,
                    cost_chain_head_hash=cost_chain_head_hash,
                    attempt_journal_root=Path(attempt_journal_root),
                    process_group_alive=(
                        process_group_alive or _process_group_is_alive
                    ),
                    process_incarnation=(
                        recovery_process_incarnation
                        or _process_incarnation_token
                    ),
                )
                audit = audit_generation_lifecycle(
                    output_dir=output_root,
                    identity=identity,
                    policy=policy,
                    require_closed=False,
                )
                if audit.active_segment_id is not None:
                    raise ValueError("generation unclean recovery did not close the chain")
                if audit.lifecycle_head_hash != previous_hash:
                    raise ValueError("generation recovery lifecycle head mismatch")
                if audit.cost_chain_head_hash != cost_chain_head_hash:
                    raise ValueError("generation recovery cost head mismatch")
            continuation_index = audit.segment_count
            if continuation_index >= policy.max_segments:
                raise ValueError("generation continuation segment budget is exhausted")
            segment_id = uuid.uuid4().hex
            segment_dir = (
                output_root
                / "generation_segments"
                / f"segment-{continuation_index:04d}-{segment_id}"
            )
            segment_dir.mkdir(parents=True, exist_ok=False)
            process_id = os.getpid()
            process_incarnation_token = _process_incarnation_token(process_id)
            if process_incarnation_token is None:
                raise ValueError("generation segment process disappeared before start")
            start = {
                "schema_version": GENERATION_SEGMENT_START_SCHEMA_VERSION,
                "identity": identity.to_dict(),
                "continuation_policy_hash": policy.policy_hash,
                "segment_id": segment_id,
                "continuation_index": continuation_index,
                "previous_exit_or_recovery_receipt_hash": previous_hash,
                "process_id": process_id,
                "process_group_id": _current_process_group_id(),
                "process_incarnation_token": process_incarnation_token,
                "lease_id": lease_id,
                "started_at_utc": _utc_now(),
                "generations_at_start": closed_generation_count,
                "cost_chain_head_hash": cost_chain_head_hash,
                "ledger_authority": False,
            }
            _publish_json_once(segment_dir / "start.json", start)
            return cls(
                output_dir=output_root,
                identity=identity,
                policy=policy,
                segment_id=segment_id,
                continuation_index=continuation_index,
                segment_dir=segment_dir,
                start_record=start,
                start_hash=canonical_json_hash(start),
                cost_chain_head_hash=cost_chain_head_hash,
                _lease=lease,
            )
        except BaseException:
            lease.release()
            raise

    def finish(
        self,
        *,
        status: str,
        reason: str,
        closed_generation_count: int,
        cost_chain_head_hash: str,
        unclosed_attempt_reservations: Sequence[Path] = (),
        exception: BaseException | None = None,
    ) -> Path:
        try:
            self._lease.assert_owned()
            if status not in _EXIT_STATUSES or reason not in _EXIT_REASONS:
                raise ValueError("generation segment exit status/reason is invalid")
            if status == "completed" and reason != "normal":
                raise ValueError("completed generation segment must exit normally")
            if status != "completed" and reason == "normal":
                raise ValueError("non-completed generation segment cannot exit normally")
            if (reason == "exception") != (exception is not None):
                raise ValueError(
                    "generation segment exception detail does not match exit reason"
                )
            exception_type, exception_message = _project_exit_exception(exception)
            _require_nonnegative_int(
                closed_generation_count,
                "generation segment exit ledger height",
            )
            if closed_generation_count < self.start_record["generations_at_start"]:
                raise ValueError("generation segment ledger height regressed")
            if (
                reason == "operator_pause"
                and self.policy.schema_version
                == GENERATION_CONTINUATION_POLICY_V2_SCHEMA_VERSION
                and closed_generation_count
                not in self.policy.operator_pause_generation_boundaries
            ):
                raise ValueError(
                    "generation segment pause is not at a precommitted boundary"
                )
            require_sha256_digest(
                cost_chain_head_hash,
                "generation segment exit cost chain head",
            )
            refs = [_reservation_ref(path) for path in unclosed_attempt_reservations]
            if refs:
                raise ValueError(
                    "controlled segment exit cannot leave attempt reservations unclosed"
                )
            continuation_eligible = (
                status in {"paused", "failed"}
                and reason in self.policy.allowed_continuation_reasons
                and self.continuation_index + 1 < self.policy.max_segments
            )
            exit_record = {
                "schema_version": GENERATION_SEGMENT_EXIT_SCHEMA_VERSION,
                "identity": self.identity.to_dict(),
                "continuation_policy_hash": self.policy.policy_hash,
                "segment_id": self.segment_id,
                "continuation_index": self.continuation_index,
                "start_receipt_hash": self.start_hash,
                "status": status,
                "reason": reason,
                "exception_type": exception_type,
                "exception_message": exception_message,
                "exited_at_utc": _utc_now(),
                "generations_at_end": closed_generation_count,
                "unclosed_attempt_reservations": refs,
                "cost_chain_head_hash": cost_chain_head_hash,
                "continuation_eligible": continuation_eligible,
                "ledger_authority": False,
            }
            path = self.segment_dir / "exit.json"
            _publish_json_once(path, exit_record)
            self.cost_chain_head_hash = cost_chain_head_hash
            return path
        finally:
            self._lease.release()

    def release_unclean_for_test(self) -> None:
        """Simulate process death in CPU-only fault-injection tests.

        Production code must never call this method; an actual process death
        releases the OS-held lease without writing an exit receipt.
        """

        self._lease.release()


def publish_generation_continuation_policy(
    output_dir: Path,
    policy: GenerationContinuationPolicy,
) -> Path:
    if not isinstance(policy, GenerationContinuationPolicy):
        raise TypeError("generation continuation policy is invalid")
    path = Path(output_dir).resolve(strict=False) / (
        GENERATION_CONTINUATION_POLICY_FILENAME
    )
    payload = canonical_json_bytes(policy.to_dict(), trailing_newline=True)
    if len(payload) > _MAX_POLICY_BYTES:
        raise ValueError("generation continuation policy exceeds its byte limit")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        existing = read_bounded_regular_file(
            path,
            max_bytes=_MAX_POLICY_BYTES,
            label="generation continuation policy",
        )
        if existing != payload:
            raise FileExistsError(
                "generation continuation policy already exists with different bytes"
            )
        return path
    try:
        create_once_bytes(path, payload)
    except FileExistsError:
        existing = read_bounded_regular_file(
            path,
            max_bytes=_MAX_POLICY_BYTES,
            label="generation continuation policy",
        )
        if existing != payload:
            raise FileExistsError(
                "generation continuation policy already exists with different bytes"
            )
    return path


def load_generation_continuation_policy(
    output_dir: Path,
) -> GenerationContinuationPolicy:
    path = Path(output_dir).resolve(strict=False) / (
        GENERATION_CONTINUATION_POLICY_FILENAME
    )
    value = _load_json(path, max_bytes=_MAX_POLICY_BYTES, label="continuation policy")
    return GenerationContinuationPolicy.from_mapping(value)


def audit_generation_lifecycle(
    *,
    output_dir: Path,
    identity: GenerationLifecycleIdentity,
    policy: GenerationContinuationPolicy,
    require_closed: bool,
    require_recovery_attempt_results: bool = True,
) -> GenerationLifecycleAudit:
    """Verify the complete retained lifecycle chain without trusting a cache.

    Recovery attempt results are required by default.  The sole relaxed use is
    the pre-start inspection performed before the session lease can reconcile a
    recovery receipt whose result publication was interrupted.
    """

    output_root = Path(output_dir).resolve(strict=False)
    if str(output_root) != identity.output_root:
        raise ValueError("generation lifecycle audit output root mismatch")
    segment_root = output_root / "generation_segments"
    recovery_root = output_root / "generation_recoveries"
    segment_dirs = _ordered_segment_dirs(segment_root)
    recoveries = _load_recoveries(recovery_root, identity=identity, policy=policy)
    previous_hash: str | None = None
    cost_head: str | None = None
    generation_height = 0
    active_segment_id: str | None = None
    receipt_hashes: list[str] = []
    recovery_cost_edges: list[tuple[str, str, str]] = []
    used_recovery_starts: set[str] = set()
    continuation_eligible = not segment_dirs
    for expected_index, segment_dir in enumerate(segment_dirs):
        match = _SEGMENT_DIR_PATTERN.fullmatch(segment_dir.name)
        assert match is not None
        if int(match.group("index")) != expected_index:
            raise ValueError("generation lifecycle segment index gap")
        start = _load_json(
            segment_dir / "start.json",
            max_bytes=_MAX_RECEIPT_BYTES,
            label="generation segment start",
        )
        _validate_start(
            start,
            identity=identity,
            policy=policy,
            segment_dir=segment_dir,
            expected_index=expected_index,
            previous_hash=previous_hash,
            expected_generation_height=generation_height,
            expected_cost_head=cost_head,
        )
        start_hash = canonical_json_hash(start)
        receipt_hashes.append(start_hash)
        if cost_head is None:
            cost_head = start["cost_chain_head_hash"]
        exit_path = segment_dir / "exit.json"
        if exit_path.exists():
            exit_record = _load_json(
                exit_path,
                max_bytes=_MAX_RECEIPT_BYTES,
                label="generation segment exit",
            )
            _validate_exit(
                exit_record,
                start=start,
                start_hash=start_hash,
                identity=identity,
                policy=policy,
            )
            previous_hash = canonical_json_hash(exit_record)
            receipt_hashes.append(previous_hash)
            generation_height = exit_record["generations_at_end"]
            cost_head = exit_record["cost_chain_head_hash"]
            continuation_eligible = bool(exit_record["continuation_eligible"])
            if (
                expected_index + 1 < len(segment_dirs)
                and exit_record["continuation_eligible"] is not True
            ):
                raise ValueError("generation lifecycle continued after terminal exit")
            continue
        matches = [
            item
            for item in recoveries
            if item.get("unclean_start_receipt_hash") == start_hash
        ]
        if len(matches) > 1:
            raise ValueError("generation lifecycle has duplicate unclean recoveries")
        if not matches:
            if expected_index != len(segment_dirs) - 1:
                raise ValueError("generation lifecycle has an unclosed interior segment")
            active_segment_id = start["segment_id"]
            previous_hash = start_hash
            generation_height = start["generations_at_start"]
            cost_head = start["cost_chain_head_hash"]
            continuation_eligible = False
            continue
        recovery = matches[0]
        _validate_recovery(
            recovery,
            start=start,
            start_hash=start_hash,
            identity=identity,
            policy=policy,
            require_attempt_results=require_recovery_attempt_results,
        )
        recovery_hash = canonical_json_hash(recovery)
        receipt_hashes.append(recovery_hash)
        recovery_cost_edges.append(
            (
                start["cost_chain_head_hash"],
                recovery["cost_chain_head_hash_before_recovery"],
                recovery["cost_chain_head_hash_after_recovery"],
            )
        )
        used_recovery_starts.add(start_hash)
        previous_hash = recovery_hash
        generation_height = recovery["generations_at_recovery"]
        cost_head = recovery["cost_chain_head_hash_after_recovery"]
        continuation_eligible = expected_index + 1 < policy.max_segments
    if len(used_recovery_starts) != len(recoveries):
        raise ValueError("generation lifecycle contains an orphan recovery receipt")
    if require_closed and active_segment_id is not None:
        raise ValueError("generation lifecycle has an active or unclean segment")
    if cost_head is None:
        cost_head = "0" * 64
    terminal = bool(
        segment_dirs
        and active_segment_id is None
        and _last_exit_is_completed(segment_dirs[-1])
    )
    return GenerationLifecycleAudit(
        identity=identity,
        continuation_policy_hash=policy.policy_hash,
        segment_count=len(segment_dirs),
        closed_generation_count=generation_height,
        lifecycle_head_hash=previous_hash,
        cost_chain_head_hash=cost_head,
        active_segment_id=active_segment_id,
        terminal=terminal,
        continuation_eligible=continuation_eligible,
        receipt_hashes=tuple(receipt_hashes),
        recovery_cost_edges=tuple(recovery_cost_edges),
    )


def build_generation_continuation_cursor(
    *,
    ledger_path: Path,
    output_dir: Path,
    identity: GenerationLifecycleIdentity,
    policy: GenerationContinuationPolicy,
    session_contract: GenerationSessionContract,
    initial_current_best: CurrentBestEvalStats,
    acceptance_protocol: AcceptanceProtocol,
    expected_initial_policy_state_hash: str | None = None,
    expected_authoring_mode: str = PROVIDER_AUTHORING_MODE,
) -> GenerationContinuationCursor:
    """Rebuild the sole continuation point from ledger plus lifecycle facts."""

    _require_policy_binding(policy, session_contract, identity)
    records = load_ledger(Path(ledger_path))
    if len(records) > session_contract.max_generations:
        raise ValueError("generation ledger exceeds its frozen generation budget")
    _require_generation_ledger_identity(records, session_contract=session_contract)
    replay = replay_acceptance_ledger(
        records,
        initial_current_best=initial_current_best,
        expected_session_genesis_hash=identity.session_genesis_hash,
        acceptance_protocol=acceptance_protocol,
        expected_initial_policy_state_hash=expected_initial_policy_state_hash,
        expected_authoring_mode=expected_authoring_mode,
        expected_ledger_epoch_hash=identity.ledger_epoch_hash,
    )
    replay.raise_for_continuation()
    lifecycle = audit_generation_lifecycle(
        output_dir=output_dir,
        identity=identity,
        policy=policy,
        require_closed=False,
    )
    if lifecycle.active_segment_id is None:
        if lifecycle.closed_generation_count != len(records):
            raise ValueError("generation lifecycle/ledger height mismatch")
    elif lifecycle.closed_generation_count > len(records):
        raise ValueError("active generation lifecycle is ahead of the ledger")
    if lifecycle.terminal and len(records) != session_contract.max_generations:
        raise ValueError("generation lifecycle completed before the frozen budget")
    current_head_hash = canonical_json_hash(replay.current_best.to_public_dict())
    return GenerationContinuationCursor(
        current_best=replay.current_best,
        next_generation_index=len(records) + 1,
        ledger_record_count=len(records),
        ledger_snapshot_hash=canonical_json_hash(records),
        current_head_hash=current_head_hash,
        lifecycle_head_hash=lifecycle.lifecycle_head_hash,
        cost_chain_head_hash=lifecycle.cost_chain_head_hash,
        active_segment_id=lifecycle.active_segment_id,
    )


def build_generation_terminal_binding(
    *,
    ledger_path: Path,
    output_dir: Path,
    attempt_journal_root: Path,
    identity: GenerationLifecycleIdentity,
    policy: GenerationContinuationPolicy,
    session_contract: GenerationSessionContract,
    initial_current_best: CurrentBestEvalStats,
    acceptance_protocol: AcceptanceProtocol,
    expected_initial_policy_state_hash: str | None = None,
    expected_authoring_mode: str = PROVIDER_AUTHORING_MODE,
) -> GenerationTerminalBinding:
    """Fail closed unless one fixed session is scientifically and operationally closed."""

    cursor = build_generation_continuation_cursor(
        ledger_path=ledger_path,
        output_dir=output_dir,
        identity=identity,
        policy=policy,
        session_contract=session_contract,
        initial_current_best=initial_current_best,
        acceptance_protocol=acceptance_protocol,
        expected_initial_policy_state_hash=expected_initial_policy_state_hash,
        expected_authoring_mode=expected_authoring_mode,
    )
    lifecycle = audit_generation_lifecycle(
        output_dir=output_dir,
        identity=identity,
        policy=policy,
        require_closed=True,
    )
    if not lifecycle.terminal:
        raise ValueError("generation lifecycle is not terminal-completed")
    if cursor.ledger_record_count != session_contract.max_generations:
        raise ValueError("generation ledger has not closed its frozen generation count")
    _validate_session_attempts_closed(
        Path(attempt_journal_root),
        identity=identity,
    )
    if lifecycle.lifecycle_head_hash is None:
        raise ValueError("generation terminal lifecycle head is missing")
    return GenerationTerminalBinding(
        identity_hash=identity.identity_hash,
        required_generation_count=session_contract.max_generations,
        closed_generation_count=cursor.ledger_record_count,
        ledger_snapshot_hash=cursor.ledger_snapshot_hash,
        current_head_hash=cursor.current_head_hash,
        lifecycle_head_hash=lifecycle.lifecycle_head_hash,
        cost_chain_head_hash=lifecycle.cost_chain_head_hash,
        segment_count=lifecycle.segment_count,
    )


def _require_generation_ledger_identity(
    records: Sequence[Mapping[str, Any]],
    *,
    session_contract: GenerationSessionContract,
) -> None:
    """Keep legacy replay compatibility outside the generation session boundary."""

    for expected_index, record in enumerate(records, start=1):
        if (
            record.get("schema_version") != GENERATION_LEDGER_RECORD_SCHEMA_VERSION
            or record.get("record_kind") != "generation"
        ):
            raise ValueError("generation continuation requires generation ledger records")
        if (
            record.get("generation_session_contract_hash")
            != session_contract.contract_hash
        ):
            raise ValueError("generation continuation session contract mismatch")
        if record.get("generation_index") != expected_index:
            raise ValueError("generation continuation ledger index mismatch")


def _validate_session_attempts_closed(
    root: Path,
    *,
    identity: GenerationLifecycleIdentity,
) -> None:
    if not root.exists():
        return
    if root.is_symlink() or not root.is_dir():
        raise ValueError("attempt journal root must be a regular directory")
    for path in sorted(root.glob("*/attempt_*.reservation.json")):
        journal, reservation = JobAttemptJournal.from_reservation_path(path)
        scope = journal.scope
        if isinstance(scope, PreGenesisExecutionScope):
            continue
        if scope.session_genesis_hash != identity.session_genesis_hash:
            continue
        if scope.session_contract_hash != identity.generation_session_contract_hash:
            raise ValueError("attempt reservation session contract mismatch")
        attempt_index = reservation.payload["attempt_index"]
        result_path = path.with_name(path.name.replace(".reservation.", ".result."))
        if not result_path.exists():
            raise ValueError("generation terminal audit found an unclosed job attempt")
        journal.load_result(attempt_index)


def _recover_unclean_segment(
    *,
    output_dir: Path,
    identity: GenerationLifecycleIdentity,
    policy: GenerationContinuationPolicy,
    active_segment_id: str,
    previous_lifecycle_head_hash: str | None,
    closed_generation_count: int,
    cost_chain_head_hash: str,
    attempt_journal_root: Path,
    process_group_alive: Callable[[int], bool],
    process_incarnation: Callable[[int], str | None],
) -> tuple[str, str]:
    if not policy.allow_unclean_recovery:
        raise ValueError("generation continuation policy forbids unclean recovery")
    segment_dir = next(
        (
            path
            for path in _ordered_segment_dirs(output_dir / "generation_segments")
            if path.name.endswith(active_segment_id)
        ),
        None,
    )
    if segment_dir is None:
        raise ValueError("generation active segment directory is missing")
    start = _load_json(
        segment_dir / "start.json",
        max_bytes=_MAX_RECEIPT_BYTES,
        label="generation segment start",
    )
    start_hash = canonical_json_hash(start)
    if previous_lifecycle_head_hash != start_hash:
        raise ValueError("generation active segment lifecycle head mismatch")
    old_pid = start["process_id"]
    stored_incarnation = start["process_incarnation_token"]
    observed_incarnation = process_incarnation(old_pid)
    if observed_incarnation == stored_incarnation:
        raise ValueError("generation unclean segment process is still alive")
    if observed_incarnation is not None:
        require_sha256_digest(
            observed_incarnation,
            "observed generation process incarnation",
        )
    process_id_reused = observed_incarnation is not None
    old_process_group = start["process_group_id"]
    if not process_id_reused and process_group_alive(old_process_group):
        raise ValueError("generation unclean segment process group is still alive")
    reservations = _unclosed_reservations(
        Path(attempt_journal_root),
        identity=identity,
        segment_id=active_segment_id,
    )
    refs = [_reservation_ref(path) for path in reservations]
    recovery = {
        "schema_version": GENERATION_UNCLEAN_RECOVERY_SCHEMA_VERSION,
        "identity": identity.to_dict(),
        "continuation_policy_hash": policy.policy_hash,
        "recovered_segment_id": active_segment_id,
        "recovered_continuation_index": start["continuation_index"],
        "unclean_start_receipt_hash": start_hash,
        "observed_process_id": old_pid,
        "observed_process_group_id": start["process_group_id"],
        "observed_process_incarnation_token": observed_incarnation,
        "process_id_reused": process_id_reused,
        "recovered_at_utc": _utc_now(),
        "generations_at_recovery": closed_generation_count,
        "abandoned_attempt_reservations": refs,
        "cost_chain_head_hash_before_recovery": cost_chain_head_hash,
        "cost_chain_head_hash_after_recovery": _recovery_cost_head(
            cost_chain_head_hash,
            refs,
        ),
        "ledger_authority": False,
    }
    recovery_root = output_dir / "generation_recoveries"
    recovery_path = recovery_root / (
        f"recovery-{start['continuation_index']:04d}-{active_segment_id}.json"
    )
    _publish_json_once(recovery_path, recovery)
    recovery_hash = canonical_json_hash(recovery)
    for reservation_path in reservations:
        journal, reservation = JobAttemptJournal.from_reservation_path(
            reservation_path
        )
        attempt_index = reservation.payload["attempt_index"]
        journal.finish(
            reservation,
            outcome="infrastructure_failure",
            failure_kind="abandoned_unclean",
            retryable=(
                attempt_index < journal.scope.max_infrastructure_retries
            ),
            output_hash=None,
            returncode=None,
            recovery_receipt_hash=recovery_hash,
        )
    return recovery_hash, recovery["cost_chain_head_hash_after_recovery"]


def _reconcile_published_recoveries(
    *,
    output_dir: Path,
    identity: GenerationLifecycleIdentity,
    policy: GenerationContinuationPolicy,
) -> None:
    """Finish a recovery transaction that crashed after publishing its receipt."""

    recoveries = _load_recoveries(
        output_dir / "generation_recoveries",
        identity=identity,
        policy=policy,
    )
    if not recoveries:
        return
    starts: dict[str, Mapping[str, Any]] = {}
    for segment_dir in _ordered_segment_dirs(output_dir / "generation_segments"):
        start = _load_json(
            segment_dir / "start.json",
            max_bytes=_MAX_RECEIPT_BYTES,
            label="generation segment start",
        )
        starts[canonical_json_hash(start)] = start
    for recovery in recoveries:
        start_hash = recovery.get("unclean_start_receipt_hash")
        start = starts.get(start_hash)
        if start is None:
            raise ValueError("generation recovery has no bound segment start")
        _validate_recovery(
            recovery,
            start=start,
            start_hash=start_hash,
            identity=identity,
            policy=policy,
            require_attempt_results=False,
        )
        recovery_hash = canonical_json_hash(recovery)
        for ref in recovery["abandoned_attempt_reservations"]:
            journal, reservation = JobAttemptJournal.from_reservation_path(
                Path(ref["path"])
            )
            attempt_index = reservation.payload["attempt_index"]
            journal.finish(
                reservation,
                outcome="infrastructure_failure",
                failure_kind="abandoned_unclean",
                retryable=(
                    attempt_index < journal.scope.max_infrastructure_retries
                ),
                output_hash=None,
                returncode=None,
                recovery_receipt_hash=recovery_hash,
            )


def _advance_cost_head_across_recovery(
    *,
    output_dir: Path,
    identity: GenerationLifecycleIdentity,
    policy: GenerationContinuationPolicy,
    lifecycle: GenerationLifecycleAudit,
    supplied_cost_head: str,
) -> str:
    """Accept only the single recovery edge immediately after caller state."""

    recoveries = _load_recoveries(
        output_dir / "generation_recoveries",
        identity=identity,
        policy=policy,
    )
    matches = [
        recovery
        for recovery in recoveries
        if recovery.get("cost_chain_head_hash_before_recovery")
        == supplied_cost_head
        and recovery.get("cost_chain_head_hash_after_recovery")
        == lifecycle.cost_chain_head_hash
        and canonical_json_hash(recovery) == lifecycle.lifecycle_head_hash
    ]
    if len(matches) != 1:
        raise ValueError("generation lifecycle cost chain head mismatch")
    return lifecycle.cost_chain_head_hash


def _unclosed_reservations(
    root: Path,
    *,
    identity: GenerationLifecycleIdentity,
    segment_id: str,
) -> list[Path]:
    if not root.exists():
        return []
    if root.is_symlink() or not root.is_dir():
        raise ValueError("attempt journal root must be a regular directory")
    reservations: list[Path] = []
    for path in sorted(root.glob("*/attempt_*.reservation.json")):
        journal, reservation = JobAttemptJournal.from_reservation_path(path)
        scope = journal.scope
        if isinstance(scope, PreGenesisExecutionScope):
            continue
        if scope.session_genesis_hash != identity.session_genesis_hash:
            continue
        if scope.session_contract_hash != identity.generation_session_contract_hash:
            raise ValueError("attempt reservation session contract mismatch")
        if scope.invocation_segment_id != segment_id:
            continue
        result_path = path.with_name(path.name.replace(".reservation.", ".result."))
        if not result_path.exists():
            reservations.append(reservation.path)
    return reservations


def _reservation_ref(path: Path) -> dict[str, Any]:
    journal, reservation = JobAttemptJournal.from_reservation_path(Path(path))
    del journal
    payload = read_bounded_regular_file(
        reservation.path,
        max_bytes=_MAX_RECEIPT_BYTES,
        label="attempt reservation",
    )
    return {
        "path": str(reservation.path.resolve(strict=True)),
        "reservation_hash": reservation.payload_hash,
        "size_bytes": len(payload),
        "reserved_budget_units": reservation.payload["reserved_budget_units"],
    }


def _validate_start(
    value: Mapping[str, Any],
    *,
    identity: GenerationLifecycleIdentity,
    policy: GenerationContinuationPolicy,
    segment_dir: Path,
    expected_index: int,
    previous_hash: str | None,
    expected_generation_height: int,
    expected_cost_head: str | None,
) -> None:
    mapping = _exact_mapping(
        value,
        {
            "schema_version",
            "identity",
            "continuation_policy_hash",
            "segment_id",
            "continuation_index",
            "previous_exit_or_recovery_receipt_hash",
            "process_id",
            "process_group_id",
            "process_incarnation_token",
            "lease_id",
            "started_at_utc",
            "generations_at_start",
            "cost_chain_head_hash",
            "ledger_authority",
        },
        "generation segment start",
    )
    match = _SEGMENT_DIR_PATTERN.fullmatch(segment_dir.name)
    if (
        mapping["schema_version"] != GENERATION_SEGMENT_START_SCHEMA_VERSION
        or mapping["ledger_authority"] is not False
        or mapping["continuation_index"] != expected_index
        or mapping["segment_id"] != (match.group("id") if match else None)
        or mapping["previous_exit_or_recovery_receipt_hash"] != previous_hash
        or mapping["generations_at_start"] != expected_generation_height
        or GenerationLifecycleIdentity.from_mapping(mapping["identity"]) != identity
        or mapping["continuation_policy_hash"] != policy.policy_hash
    ):
        raise ValueError("generation segment start binding mismatch")
    if expected_cost_head is not None and mapping["cost_chain_head_hash"] != expected_cost_head:
        raise ValueError("generation segment start cost chain mismatch")
    require_sha256_digest(mapping["cost_chain_head_hash"], "segment cost chain head")
    _positive_int(mapping["process_id"], "segment process_id")
    _positive_int(mapping["process_group_id"], "segment process_group_id")
    require_sha256_digest(
        mapping["process_incarnation_token"],
        "segment process incarnation token",
    )
    _canonical_uuid_hex(mapping["lease_id"], "segment lease_id")
    _timestamp(mapping["started_at_utc"], "segment started_at_utc")


def _validate_exit(
    value: Mapping[str, Any],
    *,
    start: Mapping[str, Any],
    start_hash: str,
    identity: GenerationLifecycleIdentity,
    policy: GenerationContinuationPolicy,
) -> None:
    schema_version = value.get("schema_version") if isinstance(value, Mapping) else None
    common_fields = {
        "schema_version",
        "identity",
        "continuation_policy_hash",
        "segment_id",
        "continuation_index",
        "start_receipt_hash",
        "status",
        "reason",
        "exited_at_utc",
        "generations_at_end",
        "unclosed_attempt_reservations",
        "cost_chain_head_hash",
        "continuation_eligible",
        "ledger_authority",
    }
    if schema_version == LEGACY_GENERATION_SEGMENT_EXIT_SCHEMA_VERSION:
        expected_fields = common_fields
    elif schema_version == GENERATION_SEGMENT_EXIT_SCHEMA_VERSION:
        expected_fields = common_fields | {
            "exception_type",
            "exception_message",
        }
    else:
        raise ValueError("generation segment exit schema mismatch")
    mapping = _exact_mapping(
        value,
        expected_fields,
        "generation segment exit",
    )
    if (
        mapping["ledger_authority"] is not False
        or GenerationLifecycleIdentity.from_mapping(mapping["identity"]) != identity
        or mapping["continuation_policy_hash"] != policy.policy_hash
        or mapping["segment_id"] != start["segment_id"]
        or mapping["continuation_index"] != start["continuation_index"]
        or mapping["start_receipt_hash"] != start_hash
        or mapping["status"] not in _EXIT_STATUSES
        or mapping["reason"] not in _EXIT_REASONS
        or mapping["generations_at_end"] < start["generations_at_start"]
        or mapping["unclosed_attempt_reservations"] != []
    ):
        raise ValueError("generation segment exit binding mismatch")
    expected_eligible = (
        mapping["status"] in {"paused", "failed"}
        and mapping["reason"] in policy.allowed_continuation_reasons
        and mapping["continuation_index"] + 1 < policy.max_segments
    )
    if mapping["continuation_eligible"] is not expected_eligible:
        raise ValueError("generation segment continuation eligibility mismatch")
    if (mapping["status"] == "completed") != (mapping["reason"] == "normal"):
        raise ValueError("generation segment completion reason mismatch")
    if schema_version == GENERATION_SEGMENT_EXIT_SCHEMA_VERSION:
        exception_type = mapping["exception_type"]
        exception_message = mapping["exception_message"]
        if mapping["reason"] == "exception":
            if (
                not isinstance(exception_type, str)
                or not exception_type
                or len(exception_type) > _MAX_EXCEPTION_TYPE_CHARS
                or not isinstance(exception_message, str)
                or len(exception_message) > _MAX_EXCEPTION_MESSAGE_CHARS
            ):
                raise ValueError("generation segment exception detail is invalid")
            assert_public_projection_safe(
                {"type": exception_type, "message": exception_message}
            )
        elif exception_type is not None or exception_message is not None:
            raise ValueError(
                "generation segment non-exception exit has exception detail"
            )
    require_sha256_digest(mapping["cost_chain_head_hash"], "segment exit cost head")
    _timestamp(mapping["exited_at_utc"], "segment exited_at_utc")


def _project_exit_exception(
    exception: BaseException | None,
) -> tuple[str | None, str | None]:
    if exception is None:
        return None, None
    if not isinstance(exception, BaseException):
        raise TypeError("generation segment exception must be a BaseException")
    projection = project_public_runner_error(exception)
    exception_type = str(projection["type"])[:_MAX_EXCEPTION_TYPE_CHARS]
    exception_message = " ".join(str(projection["message"]).split())[
        :_MAX_EXCEPTION_MESSAGE_CHARS
    ]
    return exception_type, exception_message


def _validate_recovery(
    value: Mapping[str, Any],
    *,
    start: Mapping[str, Any],
    start_hash: str,
    identity: GenerationLifecycleIdentity,
    policy: GenerationContinuationPolicy,
    require_attempt_results: bool = True,
) -> None:
    mapping = _exact_mapping(
        value,
        {
            "schema_version",
            "identity",
            "continuation_policy_hash",
            "recovered_segment_id",
            "recovered_continuation_index",
            "unclean_start_receipt_hash",
            "observed_process_id",
            "observed_process_group_id",
            "observed_process_incarnation_token",
            "process_id_reused",
            "recovered_at_utc",
            "generations_at_recovery",
            "abandoned_attempt_reservations",
            "cost_chain_head_hash_before_recovery",
            "cost_chain_head_hash_after_recovery",
            "ledger_authority",
        },
        "generation unclean recovery",
    )
    if (
        mapping["schema_version"] != GENERATION_UNCLEAN_RECOVERY_SCHEMA_VERSION
        or mapping["ledger_authority"] is not False
        or GenerationLifecycleIdentity.from_mapping(mapping["identity"]) != identity
        or mapping["continuation_policy_hash"] != policy.policy_hash
        or mapping["recovered_segment_id"] != start["segment_id"]
        or mapping["recovered_continuation_index"] != start["continuation_index"]
        or mapping["unclean_start_receipt_hash"] != start_hash
        or mapping["observed_process_id"] != start["process_id"]
        or mapping["observed_process_group_id"] != start["process_group_id"]
        or mapping["generations_at_recovery"] < start["generations_at_start"]
    ):
        raise ValueError("generation unclean recovery binding mismatch")
    observed_incarnation = mapping["observed_process_incarnation_token"]
    if observed_incarnation is not None:
        require_sha256_digest(
            observed_incarnation,
            "observed generation process incarnation",
        )
    expected_reused = (
        observed_incarnation is not None
        and observed_incarnation != start["process_incarnation_token"]
    )
    if (
        mapping["process_id_reused"] is not expected_reused
        or observed_incarnation == start["process_incarnation_token"]
    ):
        raise ValueError("generation recovery process incarnation mismatch")
    refs = mapping["abandoned_attempt_reservations"]
    if not isinstance(refs, list):
        raise ValueError("generation abandoned attempt refs must be a list")
    for ref in refs:
        _validate_reservation_ref(ref)
        journal, reservation = JobAttemptJournal.from_reservation_path(Path(ref["path"]))
        if reservation.payload_hash != ref["reservation_hash"]:
            raise ValueError("generation abandoned reservation hash mismatch")
        if require_attempt_results:
            result = journal.load_result(reservation.payload["attempt_index"])
            if result.payload.get("recovery_receipt_hash") != canonical_json_hash(mapping):
                raise ValueError("generation recovered attempt result binding mismatch")
    expected_cost = _recovery_cost_head(
        mapping["cost_chain_head_hash_before_recovery"],
        refs,
    )
    if mapping["cost_chain_head_hash_after_recovery"] != expected_cost:
        raise ValueError("generation recovery cost chain mismatch")
    _timestamp(mapping["recovered_at_utc"], "generation recovered_at_utc")


def _load_recoveries(
    root: Path,
    *,
    identity: GenerationLifecycleIdentity,
    policy: GenerationContinuationPolicy,
) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    if root.is_symlink() or not root.is_dir():
        raise ValueError("generation recovery root must be a regular directory")
    values: list[dict[str, Any]] = []
    for path in sorted(root.glob("recovery-*.json")):
        value = _load_json(
            path,
            max_bytes=_MAX_RECEIPT_BYTES,
            label="generation unclean recovery",
        )
        if (
            GenerationLifecycleIdentity.from_mapping(value.get("identity")) != identity
            or value.get("continuation_policy_hash") != policy.policy_hash
        ):
            raise ValueError("generation recovery belongs to another session")
        values.append(value)
    return values


def _ordered_segment_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    if root.is_symlink() or not root.is_dir():
        raise ValueError("generation segment root must be a regular directory")
    entries = sorted(root.iterdir(), key=lambda path: path.name)
    if len(entries) > _MAX_SEGMENTS:
        raise ValueError("generation lifecycle segment count exceeds hard limit")
    for path in entries:
        if path.is_symlink() or not path.is_dir() or _SEGMENT_DIR_PATTERN.fullmatch(path.name) is None:
            raise ValueError("generation segment directory is invalid")
    return entries


def _last_exit_is_completed(segment_dir: Path) -> bool:
    path = segment_dir / "exit.json"
    if not path.exists():
        return False
    value = _load_json(path, max_bytes=_MAX_RECEIPT_BYTES, label="generation segment exit")
    return value.get("status") == "completed"


def _recovery_cost_head(
    previous_hash: str,
    refs: Sequence[Mapping[str, Any]],
) -> str:
    require_sha256_digest(previous_hash, "generation recovery prior cost head")
    if not refs:
        return previous_hash
    return canonical_json_hash(
        {
            "schema_version": "true_rl_agent.generation_recovery_cost_delta.v1",
            "previous_cost_chain_head_hash": previous_hash,
            "abandoned_reservations": [
                {
                    "reservation_hash": ref["reservation_hash"],
                    "reserved_budget_units": ref["reserved_budget_units"],
                }
                for ref in refs
            ],
        }
    )


def _validate_reservation_ref(value: Any) -> Mapping[str, Any]:
    mapping = _exact_mapping(
        value,
        {"path", "reservation_hash", "size_bytes", "reserved_budget_units"},
        "abandoned attempt reservation ref",
    )
    if not isinstance(mapping["path"], str) or not Path(mapping["path"]).is_absolute():
        raise ValueError("abandoned reservation path must be absolute")
    require_sha256_digest(mapping["reservation_hash"], "abandoned reservation hash")
    _positive_int(mapping["size_bytes"], "abandoned reservation size")
    _positive_int(mapping["reserved_budget_units"], "abandoned reserved budget")
    return mapping


def _require_policy_binding(
    policy: GenerationContinuationPolicy,
    contract: GenerationSessionContract,
    identity: GenerationLifecycleIdentity,
) -> None:
    if not isinstance(policy, GenerationContinuationPolicy):
        raise TypeError("generation continuation policy is invalid")
    if not isinstance(contract, GenerationSessionContract):
        raise TypeError("generation session contract is invalid")
    if contract.contract_hash != identity.generation_session_contract_hash:
        raise ValueError("generation lifecycle session contract hash mismatch")
    if contract.session_id != identity.session_id:
        raise ValueError("generation lifecycle session identity mismatch")
    if contract.bound_contract_hashes["continuation_policy"] != policy.policy_hash:
        raise ValueError("generation continuation policy is not contract-bound")
    if (
        contract.bound_contract_hashes["final_confirmation_plan"]
        != identity.final_confirmation_plan_hash
    ):
        raise ValueError("generation final confirmation plan binding mismatch")


def _publish_json_once(path: Path, value: Mapping[str, Any]) -> None:
    payload = canonical_json_bytes(dict(value), trailing_newline=True)
    if len(payload) > _MAX_RECEIPT_BYTES:
        raise ValueError("generation lifecycle receipt exceeds its byte limit")
    path.parent.mkdir(parents=True, exist_ok=True)
    create_once_bytes(path, payload)


def _load_json(path: Path, *, max_bytes: int, label: str) -> dict[str, Any]:
    payload = read_bounded_regular_file(path, max_bytes=max_bytes, label=label)
    value = strict_json_object_loads(payload, context=label)
    if payload != canonical_json_bytes(value, trailing_newline=True):
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _exact_mapping(
    value: Any,
    fields: set[str],
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{label} fields mismatch")
    return dict(value)


def _bounded_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 160:
        raise ValueError(f"{label} must be non-empty and bounded")
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _require_nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _canonical_uuid_hex(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be UUID hex")
    try:
        if uuid.UUID(hex=value).hex != value:
            raise ValueError
    except (AttributeError, ValueError) as exc:
        raise ValueError(f"{label} must be UUID hex") from exc
    return value


def _timestamp(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} is required")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} is invalid") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{label} must include timezone")
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _current_process_group_id() -> int:
    getter = getattr(os, "getpgid", None)
    if callable(getter):
        return int(getter(0))
    return os.getpid()


def _process_incarnation_token(process_id: int) -> str | None:
    """Identify one OS process instance, not merely its reusable PID."""

    _positive_int(process_id, "process_id")
    if os.name == "nt":
        process_query_limited_information = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(
            process_query_limited_information,
            False,
            process_id,
        )
        if not handle:
            error = int(ctypes.windll.kernel32.GetLastError())
            if error == 87:  # ERROR_INVALID_PARAMETER: PID is not present.
                return None
            raise OSError(error, "unable to query Windows process creation time")

        class _FileTime(ctypes.Structure):
            _fields_ = [
                ("low", ctypes.c_uint32),
                ("high", ctypes.c_uint32),
            ]

        creation = _FileTime()
        exit_time = _FileTime()
        kernel = _FileTime()
        user = _FileTime()
        try:
            ok = ctypes.windll.kernel32.GetProcessTimes(
                handle,
                ctypes.byref(creation),
                ctypes.byref(exit_time),
                ctypes.byref(kernel),
                ctypes.byref(user),
            )
            if not ok:
                error = int(ctypes.windll.kernel32.GetLastError())
                raise OSError(error, "unable to read Windows process creation time")
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
        creation_ticks = (int(creation.high) << 32) | int(creation.low)
        return canonical_json_hash(
            {
                "platform": "windows_filetime",
                "process_id": process_id,
                "creation_ticks": creation_ticks,
            }
        )

    proc_root = Path("/proc")
    if not proc_root.is_dir():
        raise RuntimeError(
            "generation continuation requires Linux procfs process identity"
        )
    try:
        boot_id = (proc_root / "sys/kernel/random/boot_id").read_text(
            encoding="utf-8"
        ).strip()
    except OSError as exc:
        raise RuntimeError("Linux boot identity is unavailable") from exc
    try:
        stat = (proc_root / str(process_id) / "stat").read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise RuntimeError("Linux process identity is unreadable") from exc
    right_parenthesis = stat.rfind(")")
    fields_after_comm = (
        stat[right_parenthesis + 2 :].split() if right_parenthesis >= 0 else []
    )
    if not boot_id or len(fields_after_comm) <= 19:
        raise RuntimeError("Linux process identity is malformed")
    try:
        starttime_ticks = int(fields_after_comm[19])
    except ValueError as exc:
        raise RuntimeError("Linux process start time is malformed") from exc
    return canonical_json_hash(
        {
            "platform": "linux_procfs",
            "process_id": process_id,
            "boot_id": boot_id,
            "starttime_ticks": starttime_ticks,
        }
    )


def _pid_is_alive(process_id: int) -> bool:
    _positive_int(process_id, "process_id")
    if os.name != "nt":
        try:
            os.kill(process_id, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True
    process_query_limited_information = 0x1000
    synchronize = 0x00100000
    handle = ctypes.windll.kernel32.OpenProcess(
        process_query_limited_information | synchronize,
        False,
        process_id,
    )
    if not handle:
        return False
    try:
        wait_result = ctypes.windll.kernel32.WaitForSingleObject(handle, 0)
        return wait_result == 0x00000102  # WAIT_TIMEOUT means still active.
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)


def _process_group_is_alive(process_group_id: int) -> bool:
    _positive_int(process_group_id, "process_group_id")
    if os.name == "nt":
        return _pid_is_alive(process_group_id)
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


__all__ = [
    "GENERATION_CONTINUATION_POLICY_FILENAME",
    "GENERATION_CONTINUATION_POLICY_SCHEMA_VERSION",
    "GENERATION_CONTINUATION_POLICY_V2_SCHEMA_VERSION",
    "GENERATION_SEGMENT_EXIT_SCHEMA_VERSION",
    "GENERATION_SEGMENT_START_SCHEMA_VERSION",
    "GENERATION_UNCLEAN_RECOVERY_SCHEMA_VERSION",
    "LEGACY_GENERATION_SEGMENT_EXIT_SCHEMA_VERSION",
    "GenerationContinuationPolicy",
    "GenerationContinuationCursor",
    "GenerationLifecycleAudit",
    "GenerationLifecycleIdentity",
    "GenerationSegmentJournal",
    "GenerationTerminalBinding",
    "audit_generation_lifecycle",
    "build_generation_continuation_cursor",
    "build_generation_terminal_binding",
    "load_generation_continuation_policy",
    "publish_generation_continuation_policy",
]
