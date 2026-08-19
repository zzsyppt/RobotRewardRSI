"""Create-once logical-job attempt receipts for bounded infrastructure retry."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from true_rl_agent.evaluation.binding import canonical_json_hash, require_sha256_digest
from true_rl_agent.fs_contract import create_once_bytes, read_bounded_regular_file
from true_rl_agent.json_contract import canonical_json_bytes, strict_json_object_loads


ATTEMPT_RESERVATION_SCHEMA_VERSION = "true_rl_agent.runner.attempt_reservation.v1"
ATTEMPT_RESULT_SCHEMA_VERSION = "true_rl_agent.runner.attempt_result.v1"
RECOVERED_ATTEMPT_RESULT_SCHEMA_VERSION = (
    "true_rl_agent.runner.attempt_result.v2"
)
RESOURCE_LEASE_ATTEMPT_RESULT_SCHEMA_VERSION = (
    "true_rl_agent.runner.attempt_result.v3"
)
PRE_GENESIS_EXECUTION_SCOPE_SCHEMA_VERSION = (
    "true_rl_agent.runner.pre_genesis_execution_scope.v1"
)
_ATTEMPT_RECEIPT_MAX_BYTES = 64 * 1024
_EXECUTION_DEVICE_PATTERN = re.compile(r"(?:cpu|cuda:(?:0|[1-9][0-9]*))\Z")
_ATTEMPT_OUTCOMES = frozenset(
    {
        "completed",
        "candidate_failure",
        "contract_failure",
        "infrastructure_failure",
        "interrupted",
        "scientific_failure",
        "unclassified_failure",
    }
)
_ATTEMPT_RESERVATION_FIELDS = frozenset(
    {
        "schema_version",
        "scope",
        "logical_job_id",
        "job_kind",
        "job_input_hash",
        "environment_hash",
        "execution_device",
        "resource_binding_hash",
        "attempt_index",
        "reserved_budget_units",
        "previous_attempt_result_hash",
    }
)
_ATTEMPT_RESULT_FIELDS = frozenset(
    {
        "schema_version",
        "reservation_hash",
        "scope_hash",
        "logical_job_id",
        "job_kind",
        "job_input_hash",
        "environment_hash",
        "execution_device",
        "resource_binding_hash",
        "attempt_index",
        "consumed_budget_units",
        "outcome",
        "failure_kind",
        "retryable",
        "output_hash",
        "returncode",
    }
)
_RECOVERED_ATTEMPT_RESULT_FIELDS = frozenset(
    {*_ATTEMPT_RESULT_FIELDS, "recovery_receipt_hash"}
)
_RESOURCE_LEASE_ATTEMPT_RESULT_FIELDS = frozenset(
    {*_ATTEMPT_RESULT_FIELDS, "resource_lease_wall_duration_ns"}
)


class JobAttemptRecoveryRequiredError(ValueError):
    """An existing reservation may only be closed by lifecycle inspection."""


@dataclass(frozen=True)
class JobAttemptScope:
    """Immutable authority and budget shared by every attempt of one job."""

    journal_root: Path
    session_genesis_hash: str
    session_contract_hash: str
    generation_id: str
    lane_id: str
    invocation_segment_id: str
    budget_contract_hash: str
    attempt_budget_units: int = 2
    max_infrastructure_retries: int = 1
    resource_binding_hash: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.journal_root, Path):
            raise TypeError("attempt journal_root must be a Path")
        for field_name in (
            "session_genesis_hash",
            "session_contract_hash",
            "budget_contract_hash",
        ):
            require_sha256_digest(getattr(self, field_name), f"attempt {field_name}")
        if self.resource_binding_hash is not None:
            require_sha256_digest(
                self.resource_binding_hash,
                "attempt resource_binding_hash",
            )
        for field_name in ("generation_id", "lane_id", "invocation_segment_id"):
            _require_bounded_text(getattr(self, field_name), f"attempt {field_name}")
        if (
            isinstance(self.max_infrastructure_retries, bool)
            or self.max_infrastructure_retries not in {0, 1}
        ):
            raise ValueError("max_infrastructure_retries must be 0 or 1")
        required_units = self.max_infrastructure_retries + 1
        if (
            isinstance(self.attempt_budget_units, bool)
            or not isinstance(self.attempt_budget_units, int)
            or self.attempt_budget_units != required_units
        ):
            raise ValueError(
                "attempt_budget_units must equal one initial attempt plus the retry reserve"
            )

    def binding_payload(self) -> dict[str, Any]:
        return {
            "session_genesis_hash": self.session_genesis_hash,
            "session_contract_hash": self.session_contract_hash,
            "generation_id": self.generation_id,
            "lane_id": self.lane_id,
            "invocation_segment_id": self.invocation_segment_id,
            "budget_contract_hash": self.budget_contract_hash,
            "attempt_budget_units": self.attempt_budget_units,
            "max_infrastructure_retries": self.max_infrastructure_retries,
            "resource_binding_hash": self.resource_binding_hash,
        }

    @classmethod
    def from_binding_payload(
        cls,
        value: Any,
        *,
        journal_root: Path,
    ) -> "JobAttemptScope":
        if not isinstance(value, Mapping) or set(value) != {
            "session_genesis_hash",
            "session_contract_hash",
            "generation_id",
            "lane_id",
            "invocation_segment_id",
            "budget_contract_hash",
            "attempt_budget_units",
            "max_infrastructure_retries",
            "resource_binding_hash",
        }:
            raise ValueError("attempt scope fields mismatch")
        return cls(journal_root=journal_root, **dict(value))


@dataclass(frozen=True)
class PreGenesisExecutionScope:
    """Immutable attempt authority available before Session Genesis exists.

    The identity is limited to inputs that are already frozen before reference
    or baseline execution.  A completed Session Genesis carries the same
    runtime snapshot and generation-session contract, so it transitively binds
    these attempts without inventing a prospective ``session_genesis_hash``.
    """

    journal_root: Path
    session_id: str
    genesis_identity_hash: str
    session_contract_hash: str
    lane_id: str
    invocation_segment_id: str
    budget_contract_hash: str
    attempt_budget_units: int = 2
    max_infrastructure_retries: int = 1
    resource_binding_hash: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.journal_root, Path):
            raise TypeError("pre-Genesis attempt journal_root must be a Path")
        _require_bounded_text(self.session_id, "pre-Genesis attempt session_id")
        for field_name in (
            "genesis_identity_hash",
            "session_contract_hash",
            "budget_contract_hash",
        ):
            require_sha256_digest(
                getattr(self, field_name),
                f"pre-Genesis attempt {field_name}",
            )
        if self.resource_binding_hash is not None:
            require_sha256_digest(
                self.resource_binding_hash,
                "pre-Genesis attempt resource_binding_hash",
            )
        for field_name in ("lane_id", "invocation_segment_id"):
            _require_bounded_text(
                getattr(self, field_name),
                f"pre-Genesis attempt {field_name}",
            )
        if (
            isinstance(self.max_infrastructure_retries, bool)
            or self.max_infrastructure_retries not in {0, 1}
        ):
            raise ValueError("max_infrastructure_retries must be 0 or 1")
        required_units = self.max_infrastructure_retries + 1
        if (
            isinstance(self.attempt_budget_units, bool)
            or not isinstance(self.attempt_budget_units, int)
            or self.attempt_budget_units != required_units
        ):
            raise ValueError(
                "attempt_budget_units must equal one initial attempt plus the retry reserve"
            )

    def binding_payload(self) -> dict[str, Any]:
        return {
            "schema_version": PRE_GENESIS_EXECUTION_SCOPE_SCHEMA_VERSION,
            "session_id": self.session_id,
            "genesis_identity_hash": self.genesis_identity_hash,
            "session_contract_hash": self.session_contract_hash,
            "lane_id": self.lane_id,
            "invocation_segment_id": self.invocation_segment_id,
            "budget_contract_hash": self.budget_contract_hash,
            "attempt_budget_units": self.attempt_budget_units,
            "max_infrastructure_retries": self.max_infrastructure_retries,
            "resource_binding_hash": self.resource_binding_hash,
        }

    @classmethod
    def from_binding_payload(
        cls,
        value: Any,
        *,
        journal_root: Path,
    ) -> "PreGenesisExecutionScope":
        if not isinstance(value, Mapping) or set(value) != {
            "schema_version",
            "session_id",
            "genesis_identity_hash",
            "session_contract_hash",
            "lane_id",
            "invocation_segment_id",
            "budget_contract_hash",
            "attempt_budget_units",
            "max_infrastructure_retries",
            "resource_binding_hash",
        }:
            raise ValueError("pre-Genesis attempt scope fields mismatch")
        if value.get("schema_version") != PRE_GENESIS_EXECUTION_SCOPE_SCHEMA_VERSION:
            raise ValueError("pre-Genesis attempt scope schema mismatch")
        payload = dict(value)
        payload.pop("schema_version")
        return cls(journal_root=journal_root, **payload)


JobAttemptExecutionScope = JobAttemptScope | PreGenesisExecutionScope


def _execution_scope_from_binding_payload(
    value: Any,
    *,
    journal_root: Path,
) -> JobAttemptExecutionScope:
    if (
        isinstance(value, Mapping)
        and value.get("schema_version")
        == PRE_GENESIS_EXECUTION_SCOPE_SCHEMA_VERSION
    ):
        return PreGenesisExecutionScope.from_binding_payload(
            value,
            journal_root=journal_root,
        )
    return JobAttemptScope.from_binding_payload(
        value,
        journal_root=journal_root,
    )


@dataclass(frozen=True)
class JobAttemptReceiptRef:
    path: Path
    payload_hash: str
    payload: Mapping[str, Any]
    created: bool


class JobAttemptJournal:
    """One logical job's append-only reservation/result chain."""

    def __init__(
        self,
        *,
        scope: JobAttemptExecutionScope,
        logical_job_id: str,
        job_kind: str,
        job_input_hash: str,
        environment_hash: str,
        execution_device: str,
        resource_binding_hash: str | None,
    ) -> None:
        if not isinstance(scope, JobAttemptScope | PreGenesisExecutionScope):
            raise TypeError("attempt journal scope is invalid")
        _require_bounded_text(logical_job_id, "logical_job_id")
        if job_kind not in {"train", "eval", "eval_phase", "baseline_eval"}:
            raise ValueError("attempt job_kind is invalid")
        require_sha256_digest(job_input_hash, "attempt job_input_hash")
        require_sha256_digest(environment_hash, "attempt environment_hash")
        _require_execution_device(execution_device)
        if resource_binding_hash is not None:
            require_sha256_digest(
                resource_binding_hash,
                "attempt journal resource_binding_hash",
            )
        if scope.resource_binding_hash != resource_binding_hash:
            raise ValueError("attempt scope resource binding does not match job lease")
        self.scope = scope
        self.logical_job_id = logical_job_id
        self.job_kind = job_kind
        self.job_input_hash = job_input_hash
        self.environment_hash = environment_hash
        self.execution_device = execution_device
        self.resource_binding_hash = resource_binding_hash
        key = canonical_json_hash(
            {
                "scope": scope.binding_payload(),
                "logical_job_id": logical_job_id,
                "job_kind": job_kind,
                "job_input_hash": job_input_hash,
            }
        )
        self.root = scope.journal_root.resolve(strict=False) / key[:24]

    def reserve(
        self,
        attempt_index: int,
        *,
        previous_result: JobAttemptReceiptRef | None,
    ) -> JobAttemptReceiptRef:
        self._require_attempt_index(attempt_index)
        if attempt_index == 0:
            if previous_result is not None:
                raise ValueError("initial attempt cannot reference a previous result")
            previous_hash = None
        else:
            if previous_result is None:
                raise ValueError("retry reservation requires the previous attempt result")
            previous = self._load_and_validate_result(attempt_index - 1)
            if previous.payload_hash != previous_result.payload_hash:
                raise ValueError("retry previous result reference mismatch")
            if previous.payload.get("outcome") != "infrastructure_failure":
                raise ValueError("only an infrastructure failure may authorize retry")
            if previous.payload.get("retryable") is not True:
                raise ValueError("previous infrastructure failure is not retryable")
            previous_hash = previous.payload_hash
        payload = {
            "schema_version": ATTEMPT_RESERVATION_SCHEMA_VERSION,
            "scope": self.scope.binding_payload(),
            "logical_job_id": self.logical_job_id,
            "job_kind": self.job_kind,
            "job_input_hash": self.job_input_hash,
            "environment_hash": self.environment_hash,
            "execution_device": self.execution_device,
            "resource_binding_hash": self.resource_binding_hash,
            "attempt_index": attempt_index,
            "reserved_budget_units": 1,
            "previous_attempt_result_hash": previous_hash,
        }
        return self._publish_or_verify(
            self._reservation_path(attempt_index),
            payload,
            context="attempt reservation",
        )

    def finish(
        self,
        reservation: JobAttemptReceiptRef,
        *,
        outcome: str,
        failure_kind: str | None,
        retryable: bool,
        output_hash: str | None,
        returncode: int | None,
        recovery_receipt_hash: str | None = None,
        resource_lease_wall_duration_ns: int | None = None,
    ) -> JobAttemptReceiptRef:
        if outcome not in _ATTEMPT_OUTCOMES:
            raise ValueError("attempt result outcome is invalid")
        reservation_payload = self._load_receipt(
            reservation.path,
            expected_schema=ATTEMPT_RESERVATION_SCHEMA_VERSION,
            context="attempt reservation",
        )
        if canonical_json_hash(reservation_payload) != reservation.payload_hash:
            raise ValueError("attempt reservation reference hash mismatch")
        attempt_index = reservation_payload.get("attempt_index")
        self._require_attempt_index(attempt_index)
        expected_reservation_path = self._reservation_path(attempt_index)
        if reservation.path.resolve(strict=False) != expected_reservation_path.resolve(
            strict=False
        ):
            raise ValueError("attempt reservation belongs to a different logical job")
        canonical_reservation = self._load_receipt(
            expected_reservation_path,
            expected_schema=ATTEMPT_RESERVATION_SCHEMA_VERSION,
            context="attempt reservation",
        )
        if canonical_reservation != reservation_payload:
            raise ValueError("attempt reservation payload does not match journal authority")
        if retryable and outcome != "infrastructure_failure":
            raise ValueError("only infrastructure failure may be retryable")
        if retryable and attempt_index >= self.scope.max_infrastructure_retries:
            raise ValueError("terminal attempt cannot be marked retryable")
        if output_hash is not None:
            require_sha256_digest(output_hash, "attempt output_hash")
        if returncode is not None and (
            isinstance(returncode, bool) or not isinstance(returncode, int)
        ):
            raise TypeError("attempt returncode must be an integer or null")
        if failure_kind is not None:
            _require_bounded_text(failure_kind, "attempt failure_kind")
        if recovery_receipt_hash is not None:
            require_sha256_digest(
                recovery_receipt_hash,
                "attempt recovery_receipt_hash",
            )
            if (
                outcome != "infrastructure_failure"
                or failure_kind != "abandoned_unclean"
            ):
                raise ValueError(
                    "recovery-bound attempt result must be abandoned_unclean infrastructure failure"
                )
        if resource_lease_wall_duration_ns is not None:
            if (
                isinstance(resource_lease_wall_duration_ns, bool)
                or not isinstance(resource_lease_wall_duration_ns, int)
                or resource_lease_wall_duration_ns <= 0
            ):
                raise ValueError(
                    "resource_lease_wall_duration_ns must be a positive integer"
                )
            if self.resource_binding_hash is None:
                raise ValueError(
                    "resource lease wall duration requires a bound GPU resource"
                )
            if recovery_receipt_hash is not None:
                raise ValueError(
                    "recovered attempt result cannot claim a measured lease duration"
                )
        payload = {
            "schema_version": (
                RECOVERED_ATTEMPT_RESULT_SCHEMA_VERSION
                if recovery_receipt_hash is not None
                else (
                    RESOURCE_LEASE_ATTEMPT_RESULT_SCHEMA_VERSION
                    if resource_lease_wall_duration_ns is not None
                    else ATTEMPT_RESULT_SCHEMA_VERSION
                )
            ),
            "reservation_hash": reservation.payload_hash,
            "scope_hash": canonical_json_hash(self.scope.binding_payload()),
            "logical_job_id": self.logical_job_id,
            "job_kind": self.job_kind,
            "job_input_hash": self.job_input_hash,
            "environment_hash": self.environment_hash,
            "execution_device": self.execution_device,
            "resource_binding_hash": self.resource_binding_hash,
            "attempt_index": attempt_index,
            "consumed_budget_units": 1,
            "outcome": outcome,
            "failure_kind": failure_kind,
            "retryable": retryable,
            "output_hash": output_hash,
            "returncode": returncode,
        }
        if recovery_receipt_hash is not None:
            payload["recovery_receipt_hash"] = recovery_receipt_hash
        if resource_lease_wall_duration_ns is not None:
            payload["resource_lease_wall_duration_ns"] = (
                resource_lease_wall_duration_ns
            )
        return self._publish_or_verify(
            self._result_path(attempt_index),
            payload,
            context="attempt result",
        )

    def load_result(self, attempt_index: int) -> JobAttemptReceiptRef:
        self._require_attempt_index(attempt_index)
        return self._load_and_validate_result(attempt_index)

    def load_reservation(self, attempt_index: int) -> JobAttemptReceiptRef:
        self._require_attempt_index(attempt_index)
        path = self._reservation_path(attempt_index)
        payload = self._load_receipt(
            path,
            expected_schema=ATTEMPT_RESERVATION_SCHEMA_VERSION,
            context="attempt reservation",
        )
        if payload.get("attempt_index") != attempt_index:
            raise ValueError("attempt reservation index mismatch")
        return JobAttemptReceiptRef(
            path=path,
            payload_hash=canonical_json_hash(payload),
            payload=payload,
            created=False,
        )

    def load_reusable_output_result(self) -> JobAttemptReceiptRef | None:
        """Return the sole completed output receipt without authorizing new work.

        This is the recovery boundary for a process that finished a runner job
        and sealed its attempt result, but exited before the higher-level trial
        receipt was published.  Only a result carrying an output hash is
        reusable.  Open reservations and retryable infrastructure failures are
        left to ``begin_or_resume`` and the lifecycle recovery inspector.
        """

        for attempt_index in range(self.scope.max_infrastructure_retries + 1):
            reservation_path = self._reservation_path(attempt_index)
            if not reservation_path.exists():
                return None
            self.load_reservation(attempt_index)
            result_path = self._result_path(attempt_index)
            if not result_path.exists():
                return None
            result = self.load_result(attempt_index)
            outcome = result.payload.get("outcome")
            if outcome in {"completed", "scientific_failure"}:
                require_sha256_digest(
                    result.payload.get("output_hash"),
                    "reusable attempt output_hash",
                )
                for later_index in range(
                    attempt_index + 1,
                    self.scope.max_infrastructure_retries + 1,
                ):
                    if self._reservation_path(later_index).exists():
                        raise ValueError(
                            "terminal attempt output cannot have a later reservation"
                        )
                return result
            if not (
                outcome == "infrastructure_failure"
                and result.payload.get("retryable") is True
            ):
                return None
        return None

    def begin_or_resume(
        self,
    ) -> tuple[int, JobAttemptReceiptRef, JobAttemptReceiptRef | None]:
        """Reserve the first runnable attempt without repeating prior work.

        A reservation without a result is deliberately not repaired here.  It
        can only be closed by the generation lifecycle recovery inspector after
        proving the owning process is gone.  Once that inspector records a
        retryable ``abandoned_unclean`` result, this method consumes the single
        pre-reserved retry instead of creating a fresh attempt chain.
        """

        first_path = self._reservation_path(0)
        if not first_path.exists():
            reservation = self.reserve(0, previous_result=None)
            if not reservation.created:
                raise ValueError("initial attempt reservation raced another writer")
            return 0, reservation, None
        first_reservation = self.load_reservation(0)
        first_result_path = self._result_path(0)
        if not first_result_path.exists():
            raise JobAttemptRecoveryRequiredError(
                "existing attempt reservation requires recovery inspection"
            )
        first_result = self.load_result(0)
        if not (
            first_result.payload.get("outcome") == "infrastructure_failure"
            and first_result.payload.get("retryable") is True
        ):
            raise ValueError("existing logical job already has a terminal result")
        if self.scope.max_infrastructure_retries < 1:
            raise ValueError("existing logical job has no retry reserve")
        retry_path = self._reservation_path(1)
        if retry_path.exists():
            self.load_reservation(1)
            if not self._result_path(1).exists():
                raise JobAttemptRecoveryRequiredError(
                    "existing retry reservation requires recovery inspection"
                )
            self.load_result(1)
            raise ValueError("existing logical job already exhausted its retry")
        retry = self.reserve(1, previous_result=first_result)
        if not retry.created:
            raise ValueError("retry reservation raced another writer")
        return 1, retry, first_result

    @classmethod
    def from_reservation_path(
        cls,
        path: Path,
    ) -> tuple["JobAttemptJournal", JobAttemptReceiptRef]:
        """Reconstruct the sole journal authority named by one reservation."""

        resolved = Path(path).resolve(strict=True)
        payload = cls._load_receipt(
            resolved,
            expected_schema=ATTEMPT_RESERVATION_SCHEMA_VERSION,
            context="attempt reservation",
        )
        scope = _execution_scope_from_binding_payload(
            payload.get("scope"),
            journal_root=resolved.parent.parent,
        )
        journal = cls(
            scope=scope,
            logical_job_id=payload.get("logical_job_id"),
            job_kind=payload.get("job_kind"),
            job_input_hash=payload.get("job_input_hash"),
            environment_hash=payload.get("environment_hash"),
            execution_device=payload.get("execution_device"),
            resource_binding_hash=payload.get("resource_binding_hash"),
        )
        expected_path = journal._reservation_path(payload.get("attempt_index"))
        if resolved != expected_path.resolve(strict=True):
            raise ValueError("attempt reservation path does not match its authority")
        return journal, journal.load_reservation(payload.get("attempt_index"))

    def _load_and_validate_result(self, attempt_index: int) -> JobAttemptReceiptRef:
        path = self._result_path(attempt_index)
        payload = self._load_receipt(
            path,
            expected_schema=None,
            context="attempt result",
        )
        if payload.get("attempt_index") != attempt_index:
            raise ValueError("attempt result index mismatch")
        if payload.get("job_input_hash") != self.job_input_hash:
            raise ValueError("attempt result job input mismatch")
        expected_bindings = {
            "logical_job_id": self.logical_job_id,
            "job_kind": self.job_kind,
            "environment_hash": self.environment_hash,
            "execution_device": self.execution_device,
            "resource_binding_hash": self.resource_binding_hash,
            "consumed_budget_units": 1,
        }
        for field_name, expected in expected_bindings.items():
            if payload.get(field_name) != expected:
                raise ValueError(f"attempt result {field_name} mismatch")
        if payload.get("scope_hash") != canonical_json_hash(
            self.scope.binding_payload()
        ):
            raise ValueError("attempt result scope mismatch")
        reservation = self._load_receipt(
            self._reservation_path(attempt_index),
            expected_schema=ATTEMPT_RESERVATION_SCHEMA_VERSION,
            context="attempt reservation",
        )
        reservation_bindings = {
            "scope": self.scope.binding_payload(),
            "logical_job_id": self.logical_job_id,
            "job_kind": self.job_kind,
            "job_input_hash": self.job_input_hash,
            "environment_hash": self.environment_hash,
            "execution_device": self.execution_device,
            "resource_binding_hash": self.resource_binding_hash,
            "attempt_index": attempt_index,
            "reserved_budget_units": 1,
        }
        for field_name, expected in reservation_bindings.items():
            if reservation.get(field_name) != expected:
                raise ValueError(f"attempt reservation {field_name} mismatch")
        if payload.get("reservation_hash") != canonical_json_hash(reservation):
            raise ValueError("attempt result reservation binding mismatch")
        recovered = payload.get("schema_version") == (
            RECOVERED_ATTEMPT_RESULT_SCHEMA_VERSION
        )
        if recovered:
            require_sha256_digest(
                payload.get("recovery_receipt_hash"),
                "attempt recovery_receipt_hash",
            )
            if (
                payload.get("outcome") != "infrastructure_failure"
                or payload.get("failure_kind") != "abandoned_unclean"
            ):
                raise ValueError("recovered attempt result semantics mismatch")
        measured_resource_lease = payload.get("schema_version") == (
            RESOURCE_LEASE_ATTEMPT_RESULT_SCHEMA_VERSION
        )
        if measured_resource_lease:
            duration_ns = payload.get("resource_lease_wall_duration_ns")
            if (
                isinstance(duration_ns, bool)
                or not isinstance(duration_ns, int)
                or duration_ns <= 0
                or self.resource_binding_hash is None
            ):
                raise ValueError("attempt result resource lease duration is invalid")
        return JobAttemptReceiptRef(
            path=path,
            payload_hash=canonical_json_hash(payload),
            payload=payload,
            created=False,
        )

    def _publish_or_verify(
        self,
        path: Path,
        payload: Mapping[str, Any],
        *,
        context: str,
    ) -> JobAttemptReceiptRef:
        encoded = canonical_json_bytes(dict(payload), trailing_newline=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        created = True
        try:
            create_once_bytes(path, encoded)
        except FileExistsError:
            created = False
            existing = self._load_receipt(
                path,
                expected_schema=str(payload["schema_version"]),
                context=context,
            )
            if existing != dict(payload):
                raise ValueError(f"existing {context} does not match requested payload")
        return JobAttemptReceiptRef(
            path=path,
            payload_hash=canonical_json_hash(payload),
            payload=dict(payload),
            created=created,
        )

    @staticmethod
    def _load_receipt(
        path: Path,
        *,
        expected_schema: str | None,
        context: str,
    ) -> dict[str, Any]:
        payload = read_bounded_regular_file(
            path,
            max_bytes=_ATTEMPT_RECEIPT_MAX_BYTES,
            label=context,
        )
        parsed = strict_json_object_loads(payload, context=context)
        schema = parsed.get("schema_version")
        if expected_schema is not None and schema != expected_schema:
            raise ValueError(f"{context} schema mismatch")
        if schema == ATTEMPT_RESERVATION_SCHEMA_VERSION:
            expected_fields = _ATTEMPT_RESERVATION_FIELDS
        elif schema == ATTEMPT_RESULT_SCHEMA_VERSION:
            expected_fields = _ATTEMPT_RESULT_FIELDS
        elif schema == RECOVERED_ATTEMPT_RESULT_SCHEMA_VERSION:
            expected_fields = _RECOVERED_ATTEMPT_RESULT_FIELDS
        elif schema == RESOURCE_LEASE_ATTEMPT_RESULT_SCHEMA_VERSION:
            expected_fields = _RESOURCE_LEASE_ATTEMPT_RESULT_FIELDS
        else:
            raise ValueError(f"{context} schema mismatch")
        if set(parsed) != expected_fields:
            raise ValueError(f"{context} fields mismatch")
        if canonical_json_bytes(parsed, trailing_newline=True) != payload:
            raise ValueError(f"{context} is not canonical JSON")
        return parsed

    def _require_attempt_index(self, attempt_index: Any) -> int:
        if (
            isinstance(attempt_index, bool)
            or not isinstance(attempt_index, int)
            or attempt_index < 0
            or attempt_index > self.scope.max_infrastructure_retries
            or attempt_index >= self.scope.attempt_budget_units
        ):
            raise ValueError("attempt index exceeds the reserved retry budget")
        return attempt_index

    def _reservation_path(self, attempt_index: int) -> Path:
        return self.root / f"attempt_{attempt_index:02d}.reservation.json"

    def _result_path(self, attempt_index: int) -> Path:
        return self.root / f"attempt_{attempt_index:02d}.result.json"


def _require_bounded_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 128:
        raise ValueError(f"{field_name} must be a non-empty string of at most 128 chars")
    return value


def _require_execution_device(value: Any) -> str:
    if not isinstance(value, str) or _EXECUTION_DEVICE_PATTERN.fullmatch(value) is None:
        raise ValueError("attempt execution_device must be canonical cpu or cuda:<index>")
    return value


__all__ = [
    "ATTEMPT_RESERVATION_SCHEMA_VERSION",
    "ATTEMPT_RESULT_SCHEMA_VERSION",
    "PRE_GENESIS_EXECUTION_SCOPE_SCHEMA_VERSION",
    "RECOVERED_ATTEMPT_RESULT_SCHEMA_VERSION",
    "RESOURCE_LEASE_ATTEMPT_RESULT_SCHEMA_VERSION",
    "JobAttemptExecutionScope",
    "JobAttemptRecoveryRequiredError",
    "JobAttemptJournal",
    "JobAttemptReceiptRef",
    "JobAttemptScope",
    "PreGenesisExecutionScope",
]
