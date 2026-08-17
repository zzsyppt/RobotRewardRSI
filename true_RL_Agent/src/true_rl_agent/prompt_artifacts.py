"""Immutable, task-neutral artifacts for provider prompt authoring."""

from __future__ import annotations

import hashlib
import os
import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from true_rl_agent.agent_core.ledger import (
    NON_PROVIDER_AUTHORING_MODE,
    PROVIDER_AUTHORING_MODE,
    dispatch_ledger_record_schema,
    require_ledger_authoring_mode,
)
from true_rl_agent.agent_core.process_lock import (
    ProcessFileLock,
    ProcessFileLockUnavailableError,
    acquire_process_file_lock,
)
from true_rl_agent.agent_core.public_safety import (
    assert_public_only_prompt_payload,
)
from true_rl_agent.agent_core.source_request_feedback import (
    build_source_request_round_feedback,
)
from true_rl_agent.agent_core.source_request_limits import CURRENT_BEST_SOURCE_ID
from true_rl_agent.authoring_artifact_contract import (
    MAX_AUTHORING_ATTEMPT_RESULT_BYTES,
    MAX_AUTHORING_BUDGET_ARTIFACT_BYTES,
    MAX_AUTHORING_EVIDENCE_ABSOLUTE_BYTES,
    authoring_evidence_max_bytes,
    require_authoring_attempt_count,
)
from true_rl_agent.candidates.authoring_policy import (
    ANTI_DEGENERATE_ASSESSMENT_SCHEMA_VERSION,
    CANDIDATE_MECHANISM_DELTA_SCHEMA_VERSION,
    require_canonical_mechanism_delta,
    select_authoring_anchor_source,
)
from true_rl_agent.candidates.mechanism import (
    MECHANISM_ATTRIBUTION_SCHEMA_VERSION,
    MechanismDecision,
    build_mechanism_attribution_summary,
)
from true_rl_agent.evaluation.binding import require_sha256_checksum
from true_rl_agent.fs_contract import create_once_bytes, read_bounded_regular_file
from true_rl_agent.json_contract import (
    canonical_json_bytes,
    strict_json_object_loads,
)
from true_rl_agent.prompt_serialization import (
    MAX_AUTHORING_PROMPT_PAYLOAD_BYTES,
    canonical_prompt_json,
)

PROMPT_PAYLOAD_REF_SCHEMA_VERSION = "true_rl_agent.prompt_payload_ref.v1"
ROUND_AUTHORING_ANCHOR_REF_SCHEMA_VERSION = (
    "true_rl_agent.round_authoring_anchor_ref.v1"
)
ROUND_AUTHORING_ANCHOR_SCHEMA_VERSION = (
    "true_rl_agent.round_authoring_anchor.v1"
)
MAX_ROUND_AUTHORING_ANCHOR_BYTES = 16 * 1024
AUTHORING_EVIDENCE_SCHEMA_V2 = "true_rl_agent.authoring_evidence.v2"
AUTHORING_EVIDENCE_SCHEMA_V3 = "true_rl_agent.authoring_evidence.v3"
AUTHORING_EVIDENCE_SCHEMA_VERSION = "true_rl_agent.authoring_evidence.v4"
AUTHORING_ATTEMPT_RESULT_SCHEMA_VERSION = (
    "true_rl_agent.authoring_attempt_result.v2"
)
AUTHORING_EVIDENCE_KIND = "provider_authoring_evidence"
_ATTEMPT_ID_RE = re.compile(r"^attempt_(\d{6})$")
_AUTHORING_BUDGET_SCHEMA_VERSION = "true_rl_agent.authoring_budget.v1"
_AUTHORING_TERMINAL_OUTCOME_SCHEMA_VERSION = (
    "true_rl_agent.authoring_terminal_outcome.v1"
)
_AUTHORING_STOP_REASONS = frozenset(
    {"candidate_verified", "provider_failed", "static_failed"}
)


@dataclass(frozen=True)
class AuthoringEvidenceSchemaDescriptor:
    schema_version: str
    budget_bound: bool
    terminal_outcome_bound: bool
    mechanism_delta_bound: bool
    round_authoring_anchor_bound: bool
    current: bool


def dispatch_authoring_evidence_schema(
    evidence: Mapping[str, Any] | Any,
) -> AuthoringEvidenceSchemaDescriptor:
    """Parse an explicitly supported sealed-evidence schema."""

    if not isinstance(evidence, Mapping):
        raise TypeError("authoring evidence must be an object")
    schema_version = evidence.get("schema_version")
    if schema_version == AUTHORING_EVIDENCE_SCHEMA_V2:
        return AuthoringEvidenceSchemaDescriptor(
            schema_version=AUTHORING_EVIDENCE_SCHEMA_V2,
            budget_bound=False,
            terminal_outcome_bound=False,
            mechanism_delta_bound=False,
            round_authoring_anchor_bound=False,
            current=False,
        )
    if schema_version == AUTHORING_EVIDENCE_SCHEMA_V3:
        return AuthoringEvidenceSchemaDescriptor(
            schema_version=AUTHORING_EVIDENCE_SCHEMA_V3,
            budget_bound=True,
            terminal_outcome_bound=True,
            mechanism_delta_bound=True,
            round_authoring_anchor_bound=False,
            current=False,
        )
    if schema_version == AUTHORING_EVIDENCE_SCHEMA_VERSION:
        return AuthoringEvidenceSchemaDescriptor(
            schema_version=AUTHORING_EVIDENCE_SCHEMA_VERSION,
            budget_bound=True,
            terminal_outcome_bound=True,
            mechanism_delta_bound=True,
            round_authoring_anchor_bound=True,
            current=True,
        )
    raise ValueError(
        f"unsupported authoring evidence schema: {schema_version!r}"
    )


@dataclass(frozen=True)
class PromptPayloadRef:
    """Identity of the exact canonical bytes sent to one provider attempt."""

    path: str
    checksum: str
    size_bytes: int
    visibility: str = "public_prompt_payload"
    schema_version: str = PROMPT_PAYLOAD_REF_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PROMPT_PAYLOAD_REF_SCHEMA_VERSION:
            raise ValueError("prompt payload ref schema_version mismatch")
        if not isinstance(self.path, str) or not self.path:
            raise ValueError("prompt payload ref path must be non-empty")
        require_sha256_checksum(self.checksum, "prompt payload checksum")
        if (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes <= 0
            or self.size_bytes > MAX_AUTHORING_PROMPT_PAYLOAD_BYTES
        ):
            raise ValueError(
                "prompt payload size_bytes must be within the authoring payload limit"
            )
        if self.visibility != "public_prompt_payload":
            raise ValueError("prompt payload visibility mismatch")

    @classmethod
    def from_mapping(cls, value: Any) -> PromptPayloadRef:
        if not isinstance(value, Mapping):
            raise ValueError("prompt payload ref must be an object")
        expected = {
            "schema_version",
            "path",
            "checksum",
            "size_bytes",
            "visibility",
        }
        if set(value) != expected:
            raise ValueError("prompt payload ref fields mismatch")
        return cls(
            schema_version=value.get("schema_version"),
            path=value.get("path"),
            checksum=value.get("checksum"),
            size_bytes=value.get("size_bytes"),
            visibility=value.get("visibility"),
        )

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "path": self.path,
            "checksum": self.checksum,
            "size_bytes": self.size_bytes,
            "visibility": self.visibility,
        }


@dataclass(frozen=True)
class RoundAuthoringAnchorRef:
    """Runner-private identity of the prompt frozen before the first provider call."""

    path: str
    checksum: str
    size_bytes: int
    visibility: str = "runner_private_authoring_anchor"
    schema_version: str = ROUND_AUTHORING_ANCHOR_REF_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ROUND_AUTHORING_ANCHOR_REF_SCHEMA_VERSION:
            raise ValueError("round authoring anchor ref schema_version mismatch")
        if not isinstance(self.path, str) or not self.path:
            raise ValueError("round authoring anchor ref path must be non-empty")
        require_sha256_checksum(
            self.checksum,
            "round authoring anchor checksum",
        )
        if (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes <= 0
            or self.size_bytes > MAX_ROUND_AUTHORING_ANCHOR_BYTES
        ):
            raise ValueError(
                "round authoring anchor size_bytes must be positive and at most "
                f"{MAX_ROUND_AUTHORING_ANCHOR_BYTES}"
            )
        if self.visibility != "runner_private_authoring_anchor":
            raise ValueError("round authoring anchor visibility mismatch")

    @classmethod
    def from_mapping(cls, value: Any) -> RoundAuthoringAnchorRef:
        if not isinstance(value, Mapping) or set(value) != {
            "schema_version",
            "path",
            "checksum",
            "size_bytes",
            "visibility",
        }:
            raise ValueError("round authoring anchor ref fields mismatch")
        return cls(
            schema_version=value.get("schema_version"),
            path=value.get("path"),
            checksum=value.get("checksum"),
            size_bytes=value.get("size_bytes"),
            visibility=value.get("visibility"),
        )

    def to_private_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "path": self.path,
            "checksum": self.checksum,
            "size_bytes": self.size_bytes,
            "visibility": self.visibility,
        }


@dataclass(frozen=True)
class AuthoringEvidenceRef:
    """Private reference to one immutable round authoring transcript."""

    path: str
    checksum: str
    size_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path:
            raise ValueError("authoring evidence path must be non-empty")
        require_sha256_checksum(self.checksum, "authoring evidence checksum")
        if (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes <= 0
            or self.size_bytes > MAX_AUTHORING_EVIDENCE_ABSOLUTE_BYTES
        ):
            raise ValueError(
                "authoring evidence size_bytes must be within the typed artifact limit"
            )

    @classmethod
    def from_private_mapping(cls, value: Any) -> AuthoringEvidenceRef:
        if not isinstance(value, Mapping):
            raise ValueError("authoring evidence ref must be an object")
        expected = {"kind", "uri", "checksum", "size_bytes", "reason"}
        if set(value) != expected:
            raise ValueError("authoring evidence ref fields mismatch")
        if value.get("kind") != AUTHORING_EVIDENCE_KIND:
            raise ValueError("authoring evidence kind mismatch")
        if value.get("reason") != "provider_authoring_audit":
            raise ValueError("authoring evidence reason mismatch")
        return cls(
            path=value.get("uri"),
            checksum=value.get("checksum"),
            size_bytes=value.get("size_bytes"),
        )

    def to_private_dict(self) -> dict[str, Any]:
        return {
            "kind": AUTHORING_EVIDENCE_KIND,
            "uri": self.path,
            "checksum": self.checksum,
            "size_bytes": self.size_bytes,
            "reason": "provider_authoring_audit",
        }


@dataclass(frozen=True)
class AuthoringAttemptHandle:
    """One reserved append-only provider attempt in a round journal."""

    round_id: str
    attempt_id: str
    sequence: int
    prompt_payload_ref: PromptPayloadRef
    result_path: str


@dataclass(frozen=True)
class AuthoringJournalSnapshot:
    """Validated durable authoring state used to resume one round."""

    attempts: tuple[dict[str, Any], ...]
    completed_attempt_records: tuple[dict[str, Any], ...]
    prompt_payload_refs: tuple[PromptPayloadRef, ...]
    round_authoring_anchor_ref: RoundAuthoringAnchorRef | None
    round_authoring_anchor: Mapping[str, Any] | None
    latest_continuation_state: Mapping[str, Any] | None
    latest_completed_attempt_id: str | None

    @property
    def reserved_attempt_count(self) -> int:
        return len(self.attempts)


@dataclass(frozen=True)
class _LoadedAuthoringAttempt:
    public_record: dict[str, Any]
    prompt_payload_ref: PromptPayloadRef | None
    continuation_state: Mapping[str, Any] | None


class AuthoringJournalInProgressError(RuntimeError):
    """Raised when another process is authoring the same round."""


class RoundAuthoringJournal:
    """Append-only provider-attempt journal that survives process interruption."""

    def __init__(self, round_dir: Path, *, round_id: str) -> None:
        self.round_dir = round_dir.resolve(strict=False)
        self.round_id = _required_text(round_id, "round_id")
        self.attempt_root = self.round_dir / "authoring_attempts"
        self.evidence_root = self.round_dir / "authoring_evidence"

    def acquire_lease(self) -> ProcessFileLock:
        payload = canonical_json_bytes(
            {
                "schema_version": "true_rl_agent.authoring_journal_lease.v1",
                "round_id": self.round_id,
                "process_id": os.getpid(),
                "owner_token": uuid.uuid4().hex,
                "ownership_model": "process_held_os_file_lock",
            },
            trailing_newline=True,
        )
        try:
            return acquire_process_file_lock(
                self.round_dir / ".authoring.lease",
                blocking=False,
                payload=payload,
            )
        except ProcessFileLockUnavailableError as exc:
            raise AuthoringJournalInProgressError(
                f"authoring already in progress for round: {self.round_id}"
            ) from exc

    def snapshot(self) -> AuthoringJournalSnapshot:
        loaded = self._load_attempts()
        has_prompt_attempt = any(
            item.prompt_payload_ref is not None for item in loaded
        )
        round_authoring_anchor_ref: RoundAuthoringAnchorRef | None = None
        round_authoring_anchor: dict[str, Any] | None = None
        if has_prompt_attempt:
            (
                round_authoring_anchor_ref,
                round_authoring_anchor,
            ) = self._load_round_authoring_anchor(required=True)
            _require_anchor_matches_first_attempt(
                round_authoring_anchor,
                loaded,
            )
        elif (self.round_dir / "authoring_anchor.json").exists():
            raise ValueError("round authoring anchor exists without an attempt")
        completed = [
            item
            for item in loaded
            if item.public_record["status"] == "completed"
        ]
        latest = completed[-1] if completed else None
        return AuthoringJournalSnapshot(
            attempts=tuple(dict(item.public_record) for item in loaded),
            completed_attempt_records=tuple(
                dict(item.public_record["provider_attempt"])
                for item in completed
            ),
            prompt_payload_refs=tuple(
                item.prompt_payload_ref
                for item in loaded
                if item.prompt_payload_ref is not None
            ),
            round_authoring_anchor_ref=round_authoring_anchor_ref,
            round_authoring_anchor=round_authoring_anchor,
            latest_continuation_state=(
                dict(latest.continuation_state)
                if latest is not None
                and latest.continuation_state is not None
                else None
            ),
            latest_completed_attempt_id=(
                str(latest.public_record["attempt_id"])
                if latest is not None
                else None
            ),
        )

    def bind_budget(
        self,
        *,
        max_provider_attempts: int,
        max_source_requests_per_round: int,
        max_source_ids_per_request: int,
        max_source_request_bytes_per_round: int,
    ) -> Mapping[str, Any]:
        require_authoring_attempt_count(max_provider_attempts)
        for field_name, value, allow_zero in (
            ("max_provider_attempts", max_provider_attempts, False),
            (
                "max_source_requests_per_round",
                max_source_requests_per_round,
                True,
            ),
            (
                "max_source_ids_per_request",
                max_source_ids_per_request,
                False,
            ),
            (
                "max_source_request_bytes_per_round",
                max_source_request_bytes_per_round,
                True,
            ),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                or (not allow_zero and value == 0)
            ):
                raise ValueError(f"{field_name} has an invalid authoring budget")
        base = {
            "schema_version": _AUTHORING_BUDGET_SCHEMA_VERSION,
            "round_id": self.round_id,
            "max_provider_attempts": max_provider_attempts,
            "max_source_requests_per_round": max_source_requests_per_round,
            "max_source_ids_per_request": max_source_ids_per_request,
            "max_source_request_bytes_per_round": (
                max_source_request_bytes_per_round
            ),
        }
        expected = {
            **base,
            "contract_hash": _checksum(canonical_json_bytes(base)),
        }
        encoded = canonical_json_bytes(expected, trailing_newline=True)
        if len(encoded) > MAX_AUTHORING_BUDGET_ARTIFACT_BYTES:
            raise ValueError("authoring budget exceeds its typed byte limit")
        path = self.round_dir / "authoring_budget.json"
        if not path.exists():
            _create_authoring_artifact_once(
                path,
                encoded,
            )
        actual = self._load_budget_contract()
        if actual != expected:
            raise ValueError("authoring budget differs from the sealed round contract")
        return actual

    def begin_attempt(
        self,
        payload: Mapping[str, Any],
    ) -> AuthoringAttemptHandle:
        """Reserve a new attempt ID and publish its exact provider prompt."""

        budget = self._load_budget_contract()
        loaded_attempts = self._load_attempts()
        has_prompt_attempt = any(
            item.prompt_payload_ref is not None for item in loaded_attempts
        )
        if len(loaded_attempts) >= budget["max_provider_attempts"]:
            raise ValueError("authoring provider budget is exhausted")
        if has_prompt_attempt:
            _, round_authoring_anchor = self._load_round_authoring_anchor(
                required=True
            )
            _require_anchor_matches_first_attempt(
                round_authoring_anchor,
                loaded_attempts,
            )
        elif (self.round_dir / "authoring_anchor.json").exists():
            raise ValueError("round authoring anchor exists without an attempt")
        attempt_dir, sequence = self._reserve_attempt_dir()
        attempt_id = attempt_dir.name
        prompt_ref = publish_prompt_payload(
            attempt_dir / "prompt_payload.json",
            payload,
        )
        if not has_prompt_attempt:
            self._publish_round_authoring_anchor(prompt_ref)
        return AuthoringAttemptHandle(
            round_id=self.round_id,
            attempt_id=attempt_id,
            sequence=sequence,
            prompt_payload_ref=prompt_ref,
            result_path=str(attempt_dir / "attempt_result.json"),
        )

    def complete_attempt(
        self,
        handle: AuthoringAttemptHandle,
        provider_attempt: Mapping[str, Any],
        *,
        continuation_state: Mapping[str, Any],
    ) -> None:
        """Commit one completed provider attempt without mutating prior events."""

        self._validate_handle(handle)
        record = dict(provider_attempt)
        if record.get("round_id") != self.round_id:
            raise ValueError("authoring attempt result round_id mismatch")
        if record.get("attempt_id") != handle.attempt_id:
            raise ValueError("authoring attempt result attempt_id mismatch")
        attempt_number = record.get("attempt")
        if (
            isinstance(attempt_number, bool)
            or not isinstance(attempt_number, int)
            or attempt_number <= 0
        ):
            raise ValueError("authoring provider attempt number must be positive")
        assert_public_only_prompt_payload(record)
        if not isinstance(continuation_state, Mapping):
            raise TypeError("authoring continuation state must be an object")
        private_continuation = dict(continuation_state)
        payload = {
            "schema_version": AUTHORING_ATTEMPT_RESULT_SCHEMA_VERSION,
            "round_id": self.round_id,
            "attempt_id": handle.attempt_id,
            "sequence": handle.sequence,
            "prompt_payload_checksum": handle.prompt_payload_ref.checksum,
            "provider_attempt": record,
            "continuation_state": private_continuation,
        }
        encoded = canonical_json_bytes(payload, trailing_newline=True)
        if len(encoded) > MAX_AUTHORING_ATTEMPT_RESULT_BYTES:
            raise ValueError("authoring attempt result exceeds its typed byte limit")
        _create_authoring_artifact_once(
            Path(handle.result_path),
            encoded,
        )

    def publish_evidence(
        self,
        *,
        selected_attempt_id: str,
        stop_reason: str,
        provider_failure: Mapping[str, Any] | None = None,
        static_authoring_failure: Mapping[str, Any] | None = None,
    ) -> AuthoringEvidenceRef:
        """Seal all completed and interrupted attempts for the round."""

        loaded_attempts = self._load_attempts()
        (
            round_authoring_anchor_ref,
            round_authoring_anchor,
        ) = self._load_round_authoring_anchor(required=True)
        _require_anchor_matches_first_attempt(
            round_authoring_anchor,
            loaded_attempts,
        )
        attempts = [dict(item.public_record) for item in loaded_attempts]
        if not attempts:
            raise ValueError("authoring journal has no attempts")
        selected = next(
            (
                item
                for item in attempts
                if item["attempt_id"] == selected_attempt_id
            ),
            None,
        )
        if selected is None or selected["status"] != "completed":
            raise ValueError("selected authoring attempt is not completed")
        if selected is not attempts[-1]:
            raise ValueError("selected authoring attempt must be the final attempt")
        provider_attempts = [
            dict(item["provider_attempt"])
            for item in attempts
            if item["status"] == "completed"
        ]
        terminal_outcome = _build_authoring_terminal_outcome(
            stop_reason=stop_reason,
            provider_failure=provider_failure,
            static_authoring_failure=static_authoring_failure,
            provider_attempts=provider_attempts,
            selected_attempt_id=selected_attempt_id,
        )
        evidence = {
            "schema_version": AUTHORING_EVIDENCE_SCHEMA_VERSION,
            "round_id": self.round_id,
            "budget_contract": self._load_budget_contract(),
            "round_authoring_anchor": (
                round_authoring_anchor_ref.to_private_dict()
            ),
            "selected_attempt_id": selected_attempt_id,
            "attempts": attempts,
            "terminal_outcome": terminal_outcome,
        }
        evidence_path = (
            self.evidence_root / f"evidence_{uuid.uuid4().hex}.json"
        )
        return _publish_authoring_evidence_payload(evidence_path, evidence)

    def load_round_authoring_anchor(self) -> dict[str, Any]:
        """Load the independently sealed first-prompt anchor contract."""

        _, anchor = self._load_round_authoring_anchor(required=True)
        return anchor

    def _publish_round_authoring_anchor(
        self,
        prompt_ref: PromptPayloadRef,
    ) -> tuple[RoundAuthoringAnchorRef, dict[str, Any]]:
        prompt_payload = verify_prompt_payload(prompt_ref)
        anchor_source, compares_to, comparison_anchor = (
            select_authoring_anchor_source(prompt_payload)
        )
        parent_program_source_checksum = (
            _prompt_parent_program_source_checksum(prompt_payload)
        )
        _validate_round_authoring_anchor_authority(
            prompt_payload=prompt_payload,
            parent_program_source_checksum=parent_program_source_checksum,
            anchor_source=anchor_source,
            compares_to=compares_to,
            comparison_anchor=comparison_anchor,
        )
        base = {
            "schema_version": ROUND_AUTHORING_ANCHOR_SCHEMA_VERSION,
            "round_id": self.round_id,
            "prompt_payload": prompt_ref.to_public_dict(),
            "parent_program_source_checksum": parent_program_source_checksum,
            "comparison_anchor": comparison_anchor,
            "compares_to": compares_to,
        }
        payload = {
            **base,
            "contract_hash": _checksum(canonical_json_bytes(base)),
        }
        path = self.round_dir / "authoring_anchor.json"
        _create_authoring_artifact_once(
            path,
            canonical_json_bytes(payload, trailing_newline=True),
        )
        return self._load_round_authoring_anchor(required=True)

    def _load_round_authoring_anchor(
        self,
        *,
        required: bool,
    ) -> tuple[RoundAuthoringAnchorRef, dict[str, Any]]:
        path = self.round_dir / "authoring_anchor.json"
        if not path.exists():
            if required:
                raise ValueError("round authoring anchor is missing")
            raise FileNotFoundError(path)
        if path.is_symlink() or not path.is_file():
            raise ValueError("round authoring anchor must be a regular file")
        encoded = read_bounded_regular_file(
            path,
            max_bytes=MAX_ROUND_AUTHORING_ANCHOR_BYTES,
            label="round authoring anchor artifact",
        )
        ref = RoundAuthoringAnchorRef(
            path=str(path.resolve(strict=False)),
            checksum=_checksum(encoded),
            size_bytes=len(encoded),
        )
        return ref, verify_round_authoring_anchor(
            ref,
            expected_round_id=self.round_id,
        )

    def _reserve_attempt_dir(self) -> tuple[Path, int]:
        self.attempt_root.mkdir(parents=True, exist_ok=True)
        sequence = 1
        while True:
            attempt_dir = self.attempt_root / f"attempt_{sequence:06d}"
            try:
                attempt_dir.mkdir()
            except FileExistsError:
                sequence += 1
                continue
            return attempt_dir, sequence

    def _load_attempts(self) -> list[_LoadedAuthoringAttempt]:
        if not self.attempt_root.is_dir():
            return []
        indexed: list[tuple[int, Path]] = []
        for child in self.attempt_root.iterdir():
            match = _ATTEMPT_ID_RE.fullmatch(child.name)
            if match is None or child.is_symlink() or not child.is_dir():
                raise ValueError("authoring journal contains an invalid attempt entry")
            indexed.append((int(match.group(1)), child))
        indexed.sort(key=lambda item: item[0])
        if [sequence for sequence, _ in indexed] != list(
            range(1, len(indexed) + 1)
        ):
            raise ValueError("authoring journal attempt sequence is not contiguous")
        return [
            self._load_attempt(attempt_dir, sequence=sequence)
            for sequence, attempt_dir in indexed
        ]

    def _load_attempt(
        self,
        attempt_dir: Path,
        *,
        sequence: int,
    ) -> _LoadedAuthoringAttempt:
        attempt_id = attempt_dir.name
        prompt_path = attempt_dir / "prompt_payload.json"
        result_path = attempt_dir / "attempt_result.json"
        prompt_ref = (
            _prompt_ref_from_existing(prompt_path)
            if prompt_path.is_file()
            else None
        )
        if result_path.is_file() and prompt_ref is None:
            raise ValueError("authoring journal result has no prompt")
        provider_attempt: dict[str, Any] | None = None
        continuation_state: dict[str, Any] | None = None
        if result_path.is_file():
            result = _load_canonical_json_artifact(
                result_path,
                field_name="authoring attempt result",
                max_bytes=MAX_AUTHORING_ATTEMPT_RESULT_BYTES,
            )
            if set(result) != {
                "schema_version",
                "round_id",
                "attempt_id",
                "sequence",
                "prompt_payload_checksum",
                "provider_attempt",
                "continuation_state",
            }:
                raise ValueError("authoring attempt result fields mismatch")
            if (
                result.get("schema_version")
                != AUTHORING_ATTEMPT_RESULT_SCHEMA_VERSION
                or result.get("round_id") != self.round_id
                or result.get("attempt_id") != attempt_id
                or result.get("sequence") != sequence
                or result.get("prompt_payload_checksum") != prompt_ref.checksum
            ):
                raise ValueError("authoring attempt result identity mismatch")
            raw_attempt = result.get("provider_attempt")
            if not isinstance(raw_attempt, Mapping):
                raise ValueError("authoring attempt result payload is invalid")
            provider_attempt = dict(raw_attempt)
            if (
                provider_attempt.get("round_id") != self.round_id
                or provider_attempt.get("attempt_id") != attempt_id
                or provider_attempt.get("attempt") != sequence
            ):
                raise ValueError("authoring provider attempt identity mismatch")
            assert_public_only_prompt_payload(provider_attempt)
            raw_continuation = result.get("continuation_state")
            if not isinstance(raw_continuation, Mapping):
                raise ValueError("authoring continuation state is invalid")
            continuation_state = dict(raw_continuation)
        if provider_attempt is not None:
            status = "completed"
        elif prompt_ref is not None:
            status = "interrupted_after_prompt"
        else:
            status = "interrupted_before_prompt"
        return _LoadedAuthoringAttempt(
            public_record={
                "attempt_id": attempt_id,
                "sequence": sequence,
                "status": status,
                "prompt_payload": (
                    prompt_ref.to_public_dict() if prompt_ref is not None else None
                ),
                "provider_attempt": provider_attempt,
            },
            prompt_payload_ref=prompt_ref,
            continuation_state=continuation_state,
        )

    def _validate_handle(self, handle: AuthoringAttemptHandle) -> None:
        if handle.round_id != self.round_id:
            raise ValueError("authoring attempt handle round mismatch")
        expected_dir = self.attempt_root / handle.attempt_id
        if Path(handle.prompt_payload_ref.path).parent != expected_dir:
            raise ValueError("authoring attempt handle path mismatch")
        verify_prompt_payload(handle.prompt_payload_ref)

    def _load_budget_contract(self) -> dict[str, Any]:
        path = self.round_dir / "authoring_budget.json"
        contract = _load_canonical_json_artifact(
            path,
            field_name="authoring budget contract",
            max_bytes=MAX_AUTHORING_BUDGET_ARTIFACT_BYTES,
        )
        expected_fields = {
            "schema_version",
            "round_id",
            "max_provider_attempts",
            "max_source_requests_per_round",
            "max_source_ids_per_request",
            "max_source_request_bytes_per_round",
            "contract_hash",
        }
        if set(contract) != expected_fields:
            raise ValueError("authoring budget contract fields mismatch")
        base = {
            key: value
            for key, value in contract.items()
            if key != "contract_hash"
        }
        if (
            contract.get("schema_version") != _AUTHORING_BUDGET_SCHEMA_VERSION
            or contract.get("round_id") != self.round_id
            or contract.get("contract_hash")
            != _checksum(canonical_json_bytes(base))
        ):
            raise ValueError("authoring budget contract identity mismatch")
        return contract


def publish_prompt_payload(
    path: Path,
    payload: Mapping[str, Any],
) -> PromptPayloadRef:
    """Atomically create the exact provider-facing prompt bytes once."""

    public_payload = dict(payload)
    assert_public_only_prompt_payload(public_payload)
    encoded = canonical_prompt_json(public_payload).encode("utf-8")
    if len(encoded) > MAX_AUTHORING_PROMPT_PAYLOAD_BYTES:
        raise ValueError("prompt payload exceeds the authoring payload limit")
    resolved = path.resolve(strict=False)
    _create_authoring_artifact_once(resolved, encoded)
    return PromptPayloadRef(
        path=str(resolved),
        checksum=_checksum(encoded),
        size_bytes=len(encoded),
    )


def verify_prompt_payload(ref: PromptPayloadRef) -> dict[str, Any]:
    """Verify a prompt ref against immutable canonical bytes on disk."""

    payload = _verified_artifact_bytes(
        path=Path(ref.path),
        checksum=ref.checksum,
        size_bytes=ref.size_bytes,
        field_name="prompt payload artifact",
        max_size_bytes=MAX_AUTHORING_PROMPT_PAYLOAD_BYTES,
    )
    parsed = strict_json_object_loads(payload, context="prompt payload artifact")
    assert_public_only_prompt_payload(parsed)
    if payload != canonical_prompt_json(parsed).encode("utf-8"):
        raise ValueError("prompt payload artifact is not canonical provider JSON")
    return parsed


def verify_round_authoring_anchor(
    ref: RoundAuthoringAnchorRef,
    *,
    expected_round_id: str,
) -> dict[str, Any]:
    """Verify the independently sealed first-prompt anchor contract."""

    payload = _verified_artifact_bytes(
        path=Path(ref.path),
        checksum=ref.checksum,
        size_bytes=ref.size_bytes,
        field_name="round authoring anchor artifact",
        max_size_bytes=MAX_ROUND_AUTHORING_ANCHOR_BYTES,
    )
    parsed = strict_json_object_loads(
        payload,
        context="round authoring anchor artifact",
    )
    if payload != canonical_json_bytes(parsed, trailing_newline=True):
        raise ValueError("round authoring anchor artifact is not canonical JSON")
    expected_fields = {
        "schema_version",
        "round_id",
        "prompt_payload",
        "parent_program_source_checksum",
        "comparison_anchor",
        "compares_to",
        "contract_hash",
    }
    if set(parsed) != expected_fields:
        raise ValueError("round authoring anchor fields mismatch")
    if (
        parsed.get("schema_version") != ROUND_AUTHORING_ANCHOR_SCHEMA_VERSION
        or parsed.get("round_id") != expected_round_id
    ):
        raise ValueError("round authoring anchor identity mismatch")
    base = {key: value for key, value in parsed.items() if key != "contract_hash"}
    if parsed.get("contract_hash") != _checksum(canonical_json_bytes(base)):
        raise ValueError("round authoring anchor contract hash mismatch")
    prompt_ref = PromptPayloadRef.from_mapping(parsed.get("prompt_payload"))
    prompt_payload = verify_prompt_payload(prompt_ref)
    parent_program_source_checksum = _prompt_parent_program_source_checksum(
        prompt_payload
    )
    anchor_source, compares_to, comparison_anchor = select_authoring_anchor_source(
        prompt_payload
    )
    _validate_round_authoring_anchor_authority(
        prompt_payload=prompt_payload,
        parent_program_source_checksum=parent_program_source_checksum,
        anchor_source=anchor_source,
        compares_to=compares_to,
        comparison_anchor=comparison_anchor,
    )
    if (
        parsed.get("parent_program_source_checksum")
        != parent_program_source_checksum
        or parsed.get("comparison_anchor") != comparison_anchor
        or parsed.get("compares_to") != compares_to
    ):
        raise ValueError("round authoring anchor projection mismatch")
    return parsed


def _require_anchor_matches_first_attempt(
    anchor: Mapping[str, Any],
    attempts: list[_LoadedAuthoringAttempt],
) -> None:
    first_prompt_ref = next(
        (
            item.prompt_payload_ref
            for item in attempts
            if item.prompt_payload_ref is not None
        ),
        None,
    )
    if first_prompt_ref is None:
        raise ValueError("round authoring anchor has no prompt attempt")
    anchor_prompt_ref = PromptPayloadRef.from_mapping(anchor.get("prompt_payload"))
    if anchor_prompt_ref != first_prompt_ref:
        raise ValueError("round authoring anchor does not bind the first prompt")


def verify_authoring_evidence(
    ref: AuthoringEvidenceRef,
    *,
    expected_round_id: str,
    expected_selected_prompt: PromptPayloadRef,
) -> dict[str, Any]:
    """Verify a sealed authoring transcript and every referenced prompt."""

    payload = _verified_artifact_bytes(
        path=Path(ref.path),
        checksum=ref.checksum,
        size_bytes=ref.size_bytes,
        field_name="authoring evidence artifact",
        max_size_bytes=MAX_AUTHORING_EVIDENCE_ABSOLUTE_BYTES,
    )
    parsed = strict_json_object_loads(
        payload,
        context="authoring evidence artifact",
    )
    if payload != canonical_json_bytes(parsed, trailing_newline=True):
        raise ValueError("authoring evidence artifact is not canonical JSON")
    descriptor = dispatch_authoring_evidence_schema(parsed)
    expected_fields = {
        "schema_version",
        "round_id",
        "selected_attempt_id",
        "attempts",
    }
    if descriptor.budget_bound:
        expected_fields.add("budget_contract")
    if descriptor.terminal_outcome_bound:
        expected_fields.add("terminal_outcome")
    if descriptor.round_authoring_anchor_bound:
        expected_fields.add("round_authoring_anchor")
    if set(parsed) != expected_fields:
        raise ValueError("authoring evidence fields mismatch")
    if parsed.get("round_id") != expected_round_id:
        raise ValueError("authoring evidence round_id mismatch")
    budget_contract = (
        _validate_authoring_budget_contract(
            parsed.get("budget_contract"),
            expected_round_id=expected_round_id,
        )
        if descriptor.budget_bound
        else None
    )
    if (
        budget_contract is not None
        and len(payload)
        > authoring_evidence_max_bytes(
            budget_contract["max_provider_attempts"]
        )
    ):
        raise ValueError("authoring evidence exceeds its budget-derived byte limit")
    attempts = parsed.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        raise ValueError("authoring evidence attempts must be non-empty")
    selected_attempt_id = parsed.get("selected_attempt_id")
    if not isinstance(selected_attempt_id, str) or not selected_attempt_id:
        raise ValueError("authoring evidence selected_attempt_id is invalid")
    selected_prompt: PromptPayloadRef | None = None
    first_prompt: PromptPayloadRef | None = None
    round_authoring_anchor: dict[str, Any] | None = None
    if descriptor.round_authoring_anchor_bound:
        try:
            round_authoring_anchor_ref = RoundAuthoringAnchorRef.from_mapping(
                parsed.get("round_authoring_anchor")
            )
            round_authoring_anchor = verify_round_authoring_anchor(
                round_authoring_anchor_ref,
                expected_round_id=expected_round_id,
            )
        except (OSError, TypeError, ValueError) as exc:
            raise ValueError("round authoring anchor is invalid") from exc
        round_parent_checksum = round_authoring_anchor.get(
            "parent_program_source_checksum"
        )
    else:
        round_parent_checksum = None
    completed_attempts: list[dict[str, Any]] = []
    for index, item in enumerate(attempts, start=1):
        if not isinstance(item, Mapping) or set(item) != {
            "attempt_id",
            "sequence",
            "status",
            "prompt_payload",
            "provider_attempt",
        }:
            raise ValueError("authoring evidence attempt fields mismatch")
        attempt_id = item.get("attempt_id")
        if (
            item.get("sequence") != index
            or attempt_id != f"attempt_{index:06d}"
        ):
            raise ValueError("authoring evidence attempt order mismatch")
        status = item.get("status")
        raw_prompt = item.get("prompt_payload")
        raw_attempt = item.get("provider_attempt")
        prompt_ref: PromptPayloadRef | None = None
        verified_prompt: dict[str, Any] | None = None
        if status == "interrupted_before_prompt":
            if raw_prompt is not None or raw_attempt is not None:
                raise ValueError("interrupted-before-prompt evidence is invalid")
        elif status == "interrupted_after_prompt":
            if raw_attempt is not None:
                raise ValueError("interrupted authoring attempt has a result")
            prompt_ref = PromptPayloadRef.from_mapping(raw_prompt)
            verified_prompt = verify_prompt_payload(prompt_ref)
        elif status == "completed":
            prompt_ref = PromptPayloadRef.from_mapping(raw_prompt)
            verified_prompt = verify_prompt_payload(prompt_ref)
            if not isinstance(raw_attempt, Mapping):
                raise ValueError(
                    "authoring evidence provider_attempt must be an object"
                )
            attempt = dict(raw_attempt)
            attempt_number = attempt.get("attempt")
            if (
                attempt.get("round_id") != expected_round_id
                or attempt.get("attempt_id") != attempt_id
                or isinstance(attempt_number, bool)
                or not isinstance(attempt_number, int)
                or (
                    attempt_number != index
                    if descriptor.budget_bound
                    else attempt_number <= 0
                )
            ):
                raise ValueError(
                    "authoring evidence provider attempt identity mismatch"
                )
            assert_public_only_prompt_payload(attempt)
            completed_attempts.append(attempt)
        else:
            raise ValueError("authoring evidence attempt status mismatch")
        if (
            prompt_ref is not None
            and descriptor.round_authoring_anchor_bound
            and _prompt_parent_program_source_checksum(
                verified_prompt if verified_prompt is not None else {}
            )
            != round_parent_checksum
        ):
            raise ValueError("authoring attempts changed round anchor parent")
        if prompt_ref is not None and first_prompt is None:
            first_prompt = prompt_ref
        if attempt_id == selected_attempt_id:
            if status != "completed" or prompt_ref is None:
                raise ValueError("selected authoring attempt is incomplete")
            selected_prompt = prompt_ref
    if selected_prompt is None:
        raise ValueError("selected authoring attempt is missing")
    if attempts[-1].get("attempt_id") != selected_attempt_id:
        raise ValueError("selected authoring attempt must be the final attempt")
    if selected_prompt != expected_selected_prompt:
        raise ValueError("authoring evidence selected prompt mismatch")
    if descriptor.round_authoring_anchor_bound:
        assert round_authoring_anchor is not None
        round_authoring_anchor_prompt = PromptPayloadRef.from_mapping(
            round_authoring_anchor.get("prompt_payload")
        )
        if (
            first_prompt is None
            or round_authoring_anchor_prompt != first_prompt
        ):
            raise ValueError("round authoring prompt anchor mismatch")
    if budget_contract is not None:
        validate_authoring_budget_usage(
            completed_attempts,
            budget_contract=budget_contract,
            expected_round_id=expected_round_id,
            reserved_attempt_count=len(attempts),
        )
    if descriptor.terminal_outcome_bound:
        authoring_outcome_from_evidence(parsed)
    return parsed


def validate_authoring_budget_usage(
    provider_attempts: list[dict[str, Any]],
    *,
    budget_contract: Mapping[str, Any],
    expected_round_id: str,
    reserved_attempt_count: int,
) -> dict[str, int]:
    """Validate durable provider/source usage against one sealed round budget."""

    contract = _validate_authoring_budget_contract(
        budget_contract,
        expected_round_id=expected_round_id,
    )
    if (
        isinstance(reserved_attempt_count, bool)
        or not isinstance(reserved_attempt_count, int)
        or reserved_attempt_count < 0
        or reserved_attempt_count > contract["max_provider_attempts"]
    ):
        raise ValueError("authoring evidence exceeds the provider attempt budget")

    source_request_count = 0
    cumulative_returned_bytes = 0
    if any(not isinstance(item, Mapping) for item in provider_attempts):
        raise ValueError("authoring provider attempt must be an object")
    ordered_attempts = sorted(
        provider_attempts,
        key=lambda item: item.get("attempt", -1),
    )
    seen_attempts: set[int] = set()
    for attempt in ordered_attempts:
        attempt_number = attempt.get("attempt")
        if (
            attempt.get("round_id") != expected_round_id
            or isinstance(attempt_number, bool)
            or not isinstance(attempt_number, int)
            or attempt_number <= 0
            or attempt_number > reserved_attempt_count
            or attempt_number in seen_attempts
        ):
            raise ValueError("authoring provider attempt budget identity mismatch")
        seen_attempts.add(attempt_number)
        source_result = attempt.get("source_request_result")
        if attempt.get("provider_status") != "source_request":
            if source_result is not None:
                raise ValueError(
                    "non-source authoring attempt carries source request usage"
                )
            continue
        if not isinstance(source_result, Mapping) or set(source_result) != {
            "status",
            "returned_logical_ids",
            "denied_logical_ids",
            "returned_bytes",
            "budget",
        }:
            raise ValueError("source request attempt result fields mismatch")
        returned_ids = source_result.get("returned_logical_ids")
        denied_ids = source_result.get("denied_logical_ids")
        returned_bytes = source_result.get("returned_bytes")
        source_budget = source_result.get("budget")
        if (
            not isinstance(returned_ids, list)
            or any(not isinstance(item, str) for item in returned_ids)
            or not isinstance(denied_ids, list)
            or any(not isinstance(item, Mapping) for item in denied_ids)
            or isinstance(returned_bytes, bool)
            or not isinstance(returned_bytes, int)
            or returned_bytes < 0
            or not isinstance(source_budget, Mapping)
            or set(source_budget) != {
                "source_request_count",
                "max_source_requests_per_round",
                "max_source_ids_per_request",
                "cumulative_returned_bytes",
                "max_source_request_bytes_per_round",
            }
        ):
            raise ValueError("source request attempt budget is invalid")
        source_request_count += 1
        cumulative_returned_bytes += returned_bytes
        if (
            source_budget.get("source_request_count") != source_request_count
            or source_budget.get("max_source_requests_per_round")
            != contract["max_source_requests_per_round"]
            or source_budget.get("max_source_ids_per_request")
            != contract["max_source_ids_per_request"]
            or source_budget.get("cumulative_returned_bytes")
            != cumulative_returned_bytes
            or source_budget.get("max_source_request_bytes_per_round")
            != contract["max_source_request_bytes_per_round"]
            or cumulative_returned_bytes
            > contract["max_source_request_bytes_per_round"]
        ):
            raise ValueError("source request usage differs from the sealed budget")
        if len(returned_ids) > contract["max_source_ids_per_request"]:
            raise ValueError("source request returned too many logical IDs")
        if source_request_count > contract["max_source_requests_per_round"] and (
            returned_bytes != 0 or returned_ids
        ):
            raise ValueError("source request returned data after its turn budget")

    return {
        "source_request_count": source_request_count,
        "source_request_bytes_returned": cumulative_returned_bytes,
    }


def _validate_authoring_budget_contract(
    value: Any,
    *,
    expected_round_id: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("authoring budget contract must be an object")
    contract = dict(value)
    expected_fields = {
        "schema_version",
        "round_id",
        "max_provider_attempts",
        "max_source_requests_per_round",
        "max_source_ids_per_request",
        "max_source_request_bytes_per_round",
        "contract_hash",
    }
    if set(contract) != expected_fields:
        raise ValueError("authoring budget contract fields mismatch")
    if (
        contract.get("schema_version") != _AUTHORING_BUDGET_SCHEMA_VERSION
        or contract.get("round_id") != expected_round_id
    ):
        raise ValueError("authoring budget contract identity mismatch")
    for field_name, allow_zero in (
        ("max_provider_attempts", False),
        ("max_source_requests_per_round", True),
        ("max_source_ids_per_request", False),
        ("max_source_request_bytes_per_round", True),
    ):
        item = contract.get(field_name)
        if (
            isinstance(item, bool)
            or not isinstance(item, int)
            or item < 0
            or (not allow_zero and item == 0)
        ):
            raise ValueError("authoring budget contract value is invalid")
    require_authoring_attempt_count(contract["max_provider_attempts"])
    base = {
        key: item
        for key, item in contract.items()
        if key != "contract_hash"
    }
    if contract.get("contract_hash") != _checksum(canonical_json_bytes(base)):
        raise ValueError("authoring budget contract hash mismatch")
    return contract


def authoring_outcome_from_evidence(
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Project the canonical terminal authoring outcome from current evidence."""

    descriptor = dispatch_authoring_evidence_schema(evidence)
    if not descriptor.terminal_outcome_bound:
        raise ValueError("historical authoring evidence has no canonical outcome")
    attempts = evidence.get("attempts")
    selected_attempt_id = evidence.get("selected_attempt_id")
    if not isinstance(attempts, list) or not isinstance(selected_attempt_id, str):
        raise ValueError("authoring evidence terminal outcome inputs are invalid")
    provider_attempts = [
        dict(item["provider_attempt"])
        for item in attempts
        if isinstance(item, Mapping)
        and item.get("status") == "completed"
        and isinstance(item.get("provider_attempt"), Mapping)
    ]
    return _validate_authoring_terminal_outcome(
        evidence.get("terminal_outcome"),
        provider_attempts=provider_attempts,
        selected_attempt_id=selected_attempt_id,
    )


def _build_authoring_terminal_outcome(
    *,
    stop_reason: str,
    provider_failure: Mapping[str, Any] | None,
    static_authoring_failure: Mapping[str, Any] | None,
    provider_attempts: list[dict[str, Any]],
    selected_attempt_id: str,
) -> dict[str, Any]:
    source_request_feedback = build_source_request_round_feedback(
        [
            attempt["source_request_result"]
            for attempt in provider_attempts
            if isinstance(attempt.get("source_request_result"), Mapping)
        ]
    )
    outcome = {
        "schema_version": _AUTHORING_TERMINAL_OUTCOME_SCHEMA_VERSION,
        "stop_reason": stop_reason,
        "provider_failure": (
            dict(provider_failure) if provider_failure is not None else None
        ),
        "static_authoring_failure": (
            dict(static_authoring_failure)
            if static_authoring_failure is not None
            else None
        ),
        "source_request_feedback": source_request_feedback,
    }
    return _validate_authoring_terminal_outcome(
        outcome,
        provider_attempts=provider_attempts,
        selected_attempt_id=selected_attempt_id,
    )


def _validate_authoring_terminal_outcome(
    value: Any,
    *,
    provider_attempts: list[dict[str, Any]],
    selected_attempt_id: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "stop_reason",
        "provider_failure",
        "static_authoring_failure",
        "source_request_feedback",
    }:
        raise ValueError("authoring terminal outcome fields mismatch")
    outcome = dict(value)
    stop_reason = outcome.get("stop_reason")
    if (
        outcome.get("schema_version")
        != _AUTHORING_TERMINAL_OUTCOME_SCHEMA_VERSION
        or stop_reason not in _AUTHORING_STOP_REASONS
    ):
        raise ValueError("authoring terminal outcome identity mismatch")
    selected = next(
        (
            attempt
            for attempt in provider_attempts
            if attempt.get("attempt_id") == selected_attempt_id
        ),
        None,
    )
    if selected is None:
        raise ValueError("authoring terminal outcome selected attempt is missing")

    selected_provider_failure = selected.get("provider_failure")
    provider_failure = outcome.get("provider_failure")
    static_failure = outcome.get("static_authoring_failure")
    if stop_reason == "provider_failed":
        if (
            not isinstance(selected_provider_failure, Mapping)
            or not isinstance(provider_failure, Mapping)
            or dict(provider_failure) != dict(selected_provider_failure)
            or static_failure is not None
        ):
            raise ValueError("provider terminal outcome differs from selected attempt")
    elif stop_reason == "static_failed":
        if provider_failure is not None or selected_provider_failure is not None:
            raise ValueError("static terminal outcome carries a provider failure")
        if not isinstance(static_failure, Mapping):
            raise ValueError("static terminal outcome is missing its failure")
        candidate_id = selected.get("candidate_id")
        if (
            not isinstance(candidate_id, str)
            or static_failure.get("candidate_id") != candidate_id
            or static_failure.get("candidate_changed_components")
            != selected.get("candidate_changed_components")
        ):
            raise ValueError("static terminal outcome candidate mismatch")
    elif (
        provider_failure is not None
        or static_failure is not None
        or selected_provider_failure is not None
    ):
        raise ValueError("verified candidate outcome carries a terminal failure")

    expected_source_feedback = build_source_request_round_feedback(
        [
            attempt["source_request_result"]
            for attempt in provider_attempts
            if isinstance(attempt.get("source_request_result"), Mapping)
        ]
    )
    if outcome.get("source_request_feedback") != expected_source_feedback:
        raise ValueError("authoring source feedback differs from sealed attempts")
    assert_public_only_prompt_payload(outcome)
    return outcome


def verify_authoring_candidate_binding(
    evidence: Mapping[str, Any],
    *,
    expected_candidate: Mapping[str, Any] | None,
) -> None:
    """Bind the selected provider attempt to the candidate used by the runner."""

    attempts = evidence.get("attempts")
    selected_attempt_id = evidence.get("selected_attempt_id")
    if not isinstance(attempts, list) or not isinstance(selected_attempt_id, str):
        raise ValueError("authoring evidence selected candidate identity mismatch")
    selected = next(
        (
            item
            for item in attempts
            if isinstance(item, Mapping)
            and item.get("attempt_id") == selected_attempt_id
        ),
        None,
    )
    provider_attempt = (
        selected.get("provider_attempt")
        if isinstance(selected, Mapping)
        else None
    )
    if not isinstance(provider_attempt, Mapping):
        raise ValueError("authoring evidence selected candidate identity mismatch")

    actual_candidate_id = provider_attempt.get("candidate_id")
    actual_changed_components = provider_attempt.get(
        "candidate_changed_components"
    )
    if expected_candidate is None:
        if actual_candidate_id is not None or actual_changed_components is not None:
            raise ValueError(
                "authoring evidence selected candidate identity mismatch"
            )
        return
    if not isinstance(expected_candidate, Mapping):
        raise ValueError("authoring evidence selected candidate identity mismatch")
    expected_candidate_id = expected_candidate.get("candidate_id")
    expected_changed_components = expected_candidate.get("changed_components")
    if (
        not isinstance(expected_candidate_id, str)
        or not isinstance(expected_changed_components, list)
        or any(not isinstance(item, str) for item in expected_changed_components)
        or actual_candidate_id != expected_candidate_id
        or actual_changed_components != sorted(expected_changed_components)
    ):
        raise ValueError("authoring evidence selected candidate identity mismatch")

    for candidate_field, attempt_field in (
        ("subject_manifest_hash", "candidate_subject_manifest_hash"),
        ("program_source_checksum", "candidate_program_source_checksum"),
        ("program_source_size_bytes", "candidate_program_source_size_bytes"),
    ):
        if (
            candidate_field in expected_candidate
            and provider_attempt.get(attempt_field)
            != expected_candidate.get(candidate_field)
        ):
            raise ValueError(
                "authoring evidence selected candidate identity mismatch"
            )


def project_selected_authoring_mechanism_delta(
    evidence: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Project the verifier-owned canonical delta from the selected attempt."""

    descriptor = dispatch_authoring_evidence_schema(evidence)
    if not descriptor.mechanism_delta_bound:
        raise ValueError("historical authoring evidence has no canonical mechanism delta")
    attempts = evidence.get("attempts")
    selected_attempt_id = evidence.get("selected_attempt_id")
    if not isinstance(attempts, list) or not isinstance(selected_attempt_id, str):
        raise ValueError("authoring evidence selected mechanism delta is invalid")
    selected = next(
        (
            item
            for item in attempts
            if isinstance(item, Mapping)
            and item.get("attempt_id") == selected_attempt_id
        ),
        None,
    )
    provider_attempt = (
        selected.get("provider_attempt")
        if isinstance(selected, Mapping)
        else None
    )
    assessment = (
        provider_attempt.get("anti_degenerate_assessment")
        if isinstance(provider_attempt, Mapping)
        else None
    )
    if assessment is None:
        raise ValueError("selected authoring mechanism assessment is missing")
    if not isinstance(assessment, Mapping):
        raise ValueError("selected authoring mechanism assessment is invalid")
    if (
        assessment.get("schema_version")
        != ANTI_DEGENERATE_ASSESSMENT_SCHEMA_VERSION
    ):
        raise ValueError("selected authoring mechanism assessment schema is invalid")
    try:
        if descriptor.round_authoring_anchor_bound:
            anchor_ref = RoundAuthoringAnchorRef.from_mapping(
                evidence.get("round_authoring_anchor")
            )
            round_authoring_anchor = verify_round_authoring_anchor(
                anchor_ref,
                expected_round_id=_required_text(
                    evidence.get("round_id"),
                    "authoring evidence round_id",
                ),
            )
            raw_prompt_ref = round_authoring_anchor.get("prompt_payload")
        else:
            raw_prompt_ref = selected.get("prompt_payload")
        prompt_ref = PromptPayloadRef.from_mapping(raw_prompt_ref)
        if descriptor.round_authoring_anchor_bound:
            first_prompt_ref = next(
                (
                    PromptPayloadRef.from_mapping(item.get("prompt_payload"))
                    for item in attempts
                    if isinstance(item, Mapping)
                    and item.get("prompt_payload") is not None
                ),
                None,
            )
            if first_prompt_ref is None or prompt_ref != first_prompt_ref:
                raise ValueError("round authoring prompt anchor mismatch")
        anchor_prompt = verify_prompt_payload(prompt_ref)
        _, expected_anchor_hash, expected_anchor_role = (
            select_authoring_anchor_source(anchor_prompt)
        )
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError("round authoring prompt anchor is invalid") from exc
    candidate_program_checksum = provider_attempt.get(
        "candidate_program_source_checksum"
    )
    classification = assessment.get("classification")
    if (
        assessment.get("status") not in {"pass", "warning"}
        or assessment.get("blocking") is not False
        or not isinstance(classification, str)
    ):
        raise ValueError("selected authoring mechanism assessment is not canonical")
    candidate_source_sha256 = assessment.get("candidate_source_sha256")
    if (
        not isinstance(candidate_program_checksum, str)
        or not isinstance(candidate_source_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", candidate_source_sha256)
        or candidate_program_checksum != f"sha256:{candidate_source_sha256}"
    ):
        raise ValueError("authoring mechanism delta source identity mismatch")
    if (
        assessment.get("comparison_anchor") != expected_anchor_role
        or assessment.get("compares_to") != expected_anchor_hash
    ):
        raise ValueError("authoring mechanism delta comparison identity mismatch")
    changed = assessment.get("changed_mechanisms")
    changed_count = assessment.get("changed_mechanism_count")
    if changed in (None, []):
        if (
            classification != "unknown"
            or (
                changed_count is not None
                and (type(changed_count) is not int or changed_count != 0)
            )
        ):
            raise ValueError("empty authoring mechanism assessment is inconsistent")
        if expected_anchor_hash is None:
            if assessment.get("reason") != "authoring_anchor_source_not_inline":
                raise ValueError(
                    "anchor-unavailable candidate has invalid mechanism assessment"
                )
            return None
        raise ValueError("local selected candidate has no canonical mechanism delta")
    try:
        canonical_changed = require_canonical_mechanism_delta(changed)
    except ValueError as exc:
        raise ValueError(
            "selected authoring mechanism assessment is not canonical"
        ) from exc
    if (
        classification
        not in {
            "mechanism_delta_present",
            "parameter_only_delta",
            "rationale_only_delta",
            "parameter_and_rationale_only_delta",
        }
        or type(changed_count) is not int
        or changed_count != len(canonical_changed)
    ):
        raise ValueError("selected authoring mechanism assessment is not canonical")
    compares_to = assessment.get("compares_to")
    if not isinstance(compares_to, str) or not re.fullmatch(r"[0-9a-f]{64}", compares_to):
        raise ValueError("authoring mechanism delta comparison identity is invalid")
    return {
        "schema_version": CANDIDATE_MECHANISM_DELTA_SCHEMA_VERSION,
        "comparison_anchor": assessment.get("comparison_anchor"),
        "compares_to": compares_to,
        "candidate_source_sha256": candidate_source_sha256,
        "changed_mechanisms": list(canonical_changed),
    }


def _prompt_parent_program_source_checksum(
    prompt_payload: Mapping[str, Any],
) -> str | None:
    lineage = prompt_payload.get("candidate_lineage")
    if not isinstance(lineage, Mapping):
        raise TypeError("round authoring prompt has no candidate lineage")
    checksum = lineage.get("parent_program_source_checksum")
    if checksum is None:
        return None
    return require_sha256_checksum(
        checksum,
        "round authoring parent program source checksum",
    )


def _validate_round_authoring_anchor_authority(
    *,
    prompt_payload: Mapping[str, Any],
    parent_program_source_checksum: str | None,
    anchor_source: str | None,
    compares_to: str | None,
    comparison_anchor: str,
) -> None:
    if parent_program_source_checksum is None:
        if comparison_anchor == "none":
            if anchor_source is not None or compares_to is not None:
                raise ValueError("cold-start authoring anchor state is inconsistent")
            return
        if comparison_anchor == "best_observed_nonaccepted":
            if anchor_source is None or compares_to is None:
                raise ValueError("cold-start observed authoring anchor is incomplete")
            return
        raise ValueError(
            "cold-start authoring prompt cannot use a current-best source anchor"
        )
    if anchor_source is None or compares_to is None:
        raise ValueError(
            "round authoring prompt has a parent but no verifiable source anchor"
        )
    if comparison_anchor != "current_best":
        raise ValueError(
            "round authoring parent must use the current-best source anchor"
        )
    lineage = prompt_payload.get("candidate_lineage")
    current_best = (
        lineage.get("current_best")
        if isinstance(lineage, Mapping)
        else None
    )
    if not isinstance(current_best, Mapping):
        raise TypeError("round authoring parent has no current-best lineage")
    lineage_program_checksum = require_sha256_checksum(
        current_best.get("current_head_program_source_checksum"),
        "round authoring current-best program source checksum",
    )
    lineage_prompt_hash = current_best.get(
        "current_head_prompt_content_sha256"
    )
    if (
        not isinstance(lineage_prompt_hash, str)
        or not re.fullmatch(r"[0-9a-f]{64}", lineage_prompt_hash)
    ):
        raise ValueError("round authoring current-best prompt hash is invalid")
    if lineage_program_checksum != parent_program_source_checksum:
        raise ValueError("round authoring parent differs from current-best lineage")
    if lineage_prompt_hash != compares_to:
        raise ValueError(
            "round authoring source anchor differs from current-best lineage"
        )
    if (
        current_best.get("source_logical_id") != CURRENT_BEST_SOURCE_ID
        or current_best.get("source_delivery") != "inline"
        or current_best.get("source_payload_ref") != CURRENT_BEST_SOURCE_ID
        or current_best.get("source_hash_scope")
        != "prompt_visible_sanitized_content"
    ):
        raise ValueError("round authoring current-best source authority is invalid")


def validate_ledger_prompt_artifacts(
    record: Mapping[str, Any],
    *,
    expected_authoring_mode: str,
) -> tuple[str, ...]:
    """Validate authoring mode and artifacts for one continuation record."""

    descriptor = dispatch_ledger_record_schema(record)
    expected_mode = require_ledger_authoring_mode(expected_authoring_mode)
    if not descriptor.authoring_evidence_verifiable:
        if expected_mode == PROVIDER_AUTHORING_MODE:
            return ("authoring_evidence_legacy_unverified",)
        reasons: list[str] = []
        if record.get("prompt_payload") is not None:
            reasons.append("prompt_artifact_ref_unexpected")
        if _authoring_evidence_items(record):
            reasons.append("authoring_evidence_ref_unexpected")
        return tuple(reasons)
    if descriptor.authoring_mode != expected_mode:
        return ("authoring_evidence_mode_mismatch",)
    raw_prompt = record.get("prompt_payload")
    evidence_items = _authoring_evidence_items(record)
    if descriptor.authoring_mode == NON_PROVIDER_AUTHORING_MODE:
        reasons: list[str] = []
        try:
            _require_bound_ledger_mechanism_attribution(
                record,
                expected_authoring_delta=None,
            )
        except (TypeError, ValueError):
            reasons.append("mechanism_attribution_contract_invalid")
        if raw_prompt is not None:
            reasons.append("prompt_artifact_ref_unexpected")
        if evidence_items:
            reasons.append("authoring_evidence_ref_unexpected")
        return tuple(reasons)
    if descriptor.authoring_mode != PROVIDER_AUTHORING_MODE:
        raise ValueError("unsupported ledger authoring mode")
    reasons: list[str] = []
    if raw_prompt is None:
        reasons.append("prompt_artifact_ref_missing")
        if not evidence_items:
            reasons.append("authoring_evidence_ref_missing")
        elif len(evidence_items) > 1:
            reasons.append("authoring_evidence_ref_ambiguous")
        return tuple(reasons)
    try:
        prompt_ref = PromptPayloadRef.from_mapping(raw_prompt)
    except (TypeError, ValueError):
        return ("prompt_artifact_ref_invalid",)
    try:
        verify_prompt_payload(prompt_ref)
    except (OSError, TypeError, ValueError):
        reasons.append("prompt_artifact_content_invalid")
    if len(evidence_items) != 1:
        reasons.append(
            "authoring_evidence_ref_missing"
            if not evidence_items
            else "authoring_evidence_ref_ambiguous"
        )
        return tuple(reasons)
    try:
        evidence_ref = AuthoringEvidenceRef.from_private_mapping(
            evidence_items[0]
        )
        evidence = verify_authoring_evidence(
            evidence_ref,
            expected_round_id=_required_text(record.get("round_id"), "round_id"),
            expected_selected_prompt=prompt_ref,
        )
    except (OSError, TypeError, ValueError):
        reasons.append("authoring_evidence_content_invalid")
    else:
        try:
            _require_ledger_evidence_compatibility(record, evidence)
        except (TypeError, ValueError):
            reasons.append("authoring_evidence_schema_not_current")
        else:
            try:
                verify_authoring_candidate_binding(
                    evidence,
                    expected_candidate=record.get("candidate"),
                )
            except (TypeError, ValueError):
                reasons.append("authoring_evidence_candidate_mismatch")
            else:
                try:
                    _require_current_ledger_authoring_projection(record, evidence)
                except (TypeError, ValueError):
                    reasons.append(
                        "authoring_evidence_outcome_projection_mismatch"
                    )
    return tuple(dict.fromkeys(reasons))


def load_ledger_authoring_evidence(
    record: Mapping[str, Any],
) -> tuple[AuthoringEvidenceRef, dict[str, Any]]:
    """Load one fully verified ledger-bound authoring transcript."""

    descriptor = dispatch_ledger_record_schema(record)
    if descriptor.authoring_mode != PROVIDER_AUTHORING_MODE:
        raise ValueError("ledger record is not provider-authored")
    prompt_ref = PromptPayloadRef.from_mapping(record.get("prompt_payload"))
    evidence_items = _authoring_evidence_items(record)
    if len(evidence_items) != 1:
        raise ValueError("ledger must contain one authoring evidence ref")
    evidence_ref = AuthoringEvidenceRef.from_private_mapping(evidence_items[0])
    evidence = verify_authoring_evidence(
        evidence_ref,
        expected_round_id=_required_text(record.get("round_id"), "round_id"),
        expected_selected_prompt=prompt_ref,
    )
    _require_ledger_evidence_compatibility(record, evidence)
    verify_authoring_candidate_binding(
        evidence,
        expected_candidate=record.get("candidate"),
    )
    _require_current_ledger_authoring_projection(record, evidence)
    return evidence_ref, evidence


def _require_ledger_evidence_compatibility(
    record: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> None:
    ledger_descriptor = dispatch_ledger_record_schema(record)
    evidence_descriptor = dispatch_authoring_evidence_schema(evidence)
    if (
        ledger_descriptor.sealed_authoring_projection_bound
        and not evidence_descriptor.terminal_outcome_bound
    ):
        raise ValueError(
            "ledger schema requires current authoring evidence"
        )
    if (
        ledger_descriptor.authoring_mode == PROVIDER_AUTHORING_MODE
        and ledger_descriptor.round_authoring_anchor_bound
        and not evidence_descriptor.round_authoring_anchor_bound
    ):
        raise ValueError(
            "ledger schema requires round-authoring-anchor-bound evidence"
        )


def _require_current_ledger_authoring_projection(
    record: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> None:
    ledger_descriptor = dispatch_ledger_record_schema(record)
    if not ledger_descriptor.sealed_authoring_projection_bound:
        return
    if "provider_attempt_records" in record:
        raise ValueError("current ledger records cannot duplicate provider attempts")
    outcome = authoring_outcome_from_evidence(evidence)
    if (
        ledger_descriptor.authoring_mode == PROVIDER_AUTHORING_MODE
        and ledger_descriptor.mechanism_delta_bound
    ):
        _require_ledger_authoring_parent_binding(
            record,
            evidence,
            require_source_authority=(
                outcome.get("stop_reason") == "candidate_verified"
            ),
        )
    for ledger_field, outcome_field in (
        ("provider_failure", "provider_failure"),
        ("static_authoring_failure", "static_authoring_failure"),
        ("source_request_feedback", "source_request_feedback"),
    ):
        if record.get(ledger_field) != outcome.get(outcome_field):
            raise ValueError(
                f"ledger {ledger_field} differs from sealed authoring evidence"
            )
    expected_delta = (
        project_selected_authoring_mechanism_delta(evidence)
        if (
            ledger_descriptor.mechanism_delta_bound
            and outcome.get("stop_reason") == "candidate_verified"
        )
        else None
    )
    if (
        ledger_descriptor.authoring_mode == PROVIDER_AUTHORING_MODE
        and ledger_descriptor.mechanism_delta_bound
        and outcome.get("stop_reason") == "candidate_verified"
        and record.get("parent_program_source_checksum") is not None
        and expected_delta is None
    ):
        raise ValueError(
            "provider-authored candidate with a parent must have a canonical mechanism delta"
        )
    if ledger_descriptor.mechanism_delta_bound:
        if outcome.get("stop_reason") == "candidate_verified":
            _require_bound_ledger_mechanism_attribution(
                record,
                expected_authoring_delta=expected_delta,
            )
        else:
            attribution = record.get("mechanism_attribution")
            if attribution is not None:
                raise ValueError("failed authoring record has mechanism attribution")


def _require_ledger_authoring_parent_binding(
    record: Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    require_source_authority: bool,
) -> None:
    if "parent_program_source_checksum" not in record:
        raise ValueError("ledger authoring parent binding is missing")
    evidence_descriptor = dispatch_authoring_evidence_schema(evidence)
    if evidence_descriptor.round_authoring_anchor_bound:
        anchor_ref = RoundAuthoringAnchorRef.from_mapping(
            evidence.get("round_authoring_anchor")
        )
        anchor = verify_round_authoring_anchor(
            anchor_ref,
            expected_round_id=_required_text(
                evidence.get("round_id"),
                "authoring evidence round_id",
            ),
        )
        prompt_ref = PromptPayloadRef.from_mapping(anchor.get("prompt_payload"))
    else:
        attempts = evidence.get("attempts")
        selected_attempt_id = evidence.get("selected_attempt_id")
        if not isinstance(attempts, list) or not isinstance(
            selected_attempt_id,
            str,
        ):
            raise ValueError("authoring evidence selected prompt is invalid")
        selected = next(
            (
                item
                for item in attempts
                if isinstance(item, Mapping)
                and item.get("attempt_id") == selected_attempt_id
            ),
            None,
        )
        if not isinstance(selected, Mapping):
            raise ValueError("authoring evidence selected prompt is missing")
        prompt_ref = PromptPayloadRef.from_mapping(
            selected.get("prompt_payload")
        )
    prompt_payload = verify_prompt_payload(prompt_ref)
    prompt_parent = _prompt_parent_program_source_checksum(prompt_payload)
    if (
        not evidence_descriptor.round_authoring_anchor_bound
        and require_source_authority
    ):
        anchor_source, compares_to, comparison_anchor = (
            select_authoring_anchor_source(prompt_payload)
        )
        _validate_round_authoring_anchor_authority(
            prompt_payload=prompt_payload,
            parent_program_source_checksum=prompt_parent,
            anchor_source=anchor_source,
            compares_to=compares_to,
            comparison_anchor=comparison_anchor,
        )
    if record.get("parent_program_source_checksum") != prompt_parent:
        raise ValueError(
            "ledger parent differs from the authoritative authoring prompt"
        )


def _require_bound_ledger_mechanism_attribution(
    record: Mapping[str, Any],
    *,
    expected_authoring_delta: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    if not dispatch_ledger_record_schema(record).mechanism_delta_bound:
        return {}
    attribution = record.get("mechanism_attribution")
    if (
        not isinstance(attribution, Mapping)
        or attribution.get("schema_version")
        != MECHANISM_ATTRIBUTION_SCHEMA_VERSION
    ):
        raise ValueError("current ledger mechanism attribution is invalid")
    runtime_gate = record.get("runtime_activation_gate")
    ablation_summary = record.get("ablation_summary")
    if not isinstance(runtime_gate, Mapping) or not isinstance(
        ablation_summary,
        Mapping,
    ):
        raise ValueError("current ledger mechanism evidence is invalid")
    gate_status = runtime_gate.get("status")
    gate_reason = runtime_gate.get("reason")
    if (
        not isinstance(gate_status, str)
        or not gate_status
        or not isinstance(gate_reason, str)
        or not gate_reason
    ):
        raise ValueError("current ledger runtime activation gate is invalid")
    expected = build_mechanism_attribution_summary(
        runtime_activation_gate=MechanismDecision(gate_status, gate_reason),
        ablation_summary=ablation_summary,
        authoring_mechanism_delta=expected_authoring_delta,
    )
    if dict(attribution) != expected:
        raise ValueError("current ledger mechanism attribution projection mismatch")
    return attribution


def _authoring_evidence_items(
    record: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    artifacts = record.get("private_artifacts")
    if not isinstance(artifacts, list):
        return []
    return [
        item
        for item in artifacts
        if isinstance(item, Mapping)
        and item.get("kind") == AUTHORING_EVIDENCE_KIND
    ]


def _publish_authoring_evidence_payload(
    path: Path,
    evidence: Mapping[str, Any],
) -> AuthoringEvidenceRef:
    encoded = canonical_json_bytes(dict(evidence), trailing_newline=True)
    budget = evidence.get("budget_contract")
    if not isinstance(budget, Mapping):
        raise ValueError("current authoring evidence requires a budget contract")
    maximum = authoring_evidence_max_bytes(budget.get("max_provider_attempts"))
    if len(encoded) > maximum:
        raise ValueError("authoring evidence exceeds its budget-derived byte limit")
    resolved = path.resolve(strict=False)
    _create_authoring_artifact_once(resolved, encoded)
    return AuthoringEvidenceRef(
        path=str(resolved),
        checksum=_checksum(encoded),
        size_bytes=len(encoded),
    )


def _prompt_ref_from_existing(path: Path) -> PromptPayloadRef:
    if path.is_symlink() or not path.is_file():
        raise ValueError("authoring journal prompt is not a regular file")
    payload = read_bounded_regular_file(
        path,
        max_bytes=MAX_AUTHORING_PROMPT_PAYLOAD_BYTES,
        label="authoring journal prompt",
    )
    parsed = strict_json_object_loads(
        payload,
        context="authoring journal prompt",
    )
    assert_public_only_prompt_payload(parsed)
    if payload != canonical_prompt_json(parsed).encode("utf-8"):
        raise ValueError("authoring journal prompt is not canonical provider JSON")
    return PromptPayloadRef(
        path=str(path.resolve(strict=False)),
        checksum=_checksum(payload),
        size_bytes=len(payload),
    )


def _load_canonical_json_artifact(
    path: Path,
    *,
    field_name: str,
    max_bytes: int,
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{field_name} is not a regular file")
    payload = read_bounded_regular_file(
        path,
        max_bytes=max_bytes,
        label=field_name,
    )
    parsed = strict_json_object_loads(payload, context=field_name)
    if payload != canonical_json_bytes(parsed, trailing_newline=True):
        raise ValueError(f"{field_name} is not canonical JSON")
    return parsed


def _verified_artifact_bytes(
    *,
    path: Path,
    checksum: str,
    size_bytes: int,
    field_name: str,
    max_size_bytes: int | None = None,
) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{field_name} is not a regular file")
    try:
        payload = (
            path.read_bytes()
            if max_size_bytes is None
            else read_bounded_regular_file(
                path,
                max_bytes=max_size_bytes,
                label=field_name,
            )
        )
    except OSError as exc:
        raise ValueError(f"failed to read {field_name}") from exc
    if len(payload) != size_bytes:
        raise ValueError(f"{field_name} size mismatch")
    if _checksum(payload) != checksum:
        raise ValueError(f"{field_name} checksum mismatch")
    return payload


def _create_authoring_artifact_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    create_once_bytes(path, payload)


def _checksum(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _required_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be non-empty")
    return value


__all__ = [
    "AUTHORING_ATTEMPT_RESULT_SCHEMA_VERSION",
    "AUTHORING_EVIDENCE_KIND",
    "AUTHORING_EVIDENCE_SCHEMA_V2",
    "AUTHORING_EVIDENCE_SCHEMA_V3",
    "AUTHORING_EVIDENCE_SCHEMA_VERSION",
    "CANDIDATE_MECHANISM_DELTA_SCHEMA_VERSION",
    "PROMPT_PAYLOAD_REF_SCHEMA_VERSION",
    "ROUND_AUTHORING_ANCHOR_REF_SCHEMA_VERSION",
    "ROUND_AUTHORING_ANCHOR_SCHEMA_VERSION",
    "AuthoringAttemptHandle",
    "AuthoringEvidenceRef",
    "PromptPayloadRef",
    "RoundAuthoringAnchorRef",
    "RoundAuthoringJournal",
    "authoring_outcome_from_evidence",
    "dispatch_authoring_evidence_schema",
    "load_ledger_authoring_evidence",
    "project_selected_authoring_mechanism_delta",
    "publish_prompt_payload",
    "validate_authoring_budget_usage",
    "validate_ledger_prompt_artifacts",
    "verify_authoring_candidate_binding",
    "verify_authoring_evidence",
    "verify_prompt_payload",
    "verify_round_authoring_anchor",
]
