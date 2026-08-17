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
        scope: JobAttemptScope,
        logical_job_id: str,
        job_kind: str,
        job_input_hash: str,
        environment_hash: str,
        execution_device: str,
        resource_binding_hash: str | None,
    ) -> None:
        if not isinstance(scope, JobAttemptScope):
            raise TypeError("attempt journal scope must be a JobAttemptScope")
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
        payload = {
            "schema_version": ATTEMPT_RESULT_SCHEMA_VERSION,
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
        return self._publish_or_verify(
            self._result_path(attempt_index),
            payload,
            context="attempt result",
        )

    def load_result(self, attempt_index: int) -> JobAttemptReceiptRef:
        self._require_attempt_index(attempt_index)
        return self._load_and_validate_result(attempt_index)

    def _load_and_validate_result(self, attempt_index: int) -> JobAttemptReceiptRef:
        path = self._result_path(attempt_index)
        payload = self._load_receipt(
            path,
            expected_schema=ATTEMPT_RESULT_SCHEMA_VERSION,
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
        expected_schema: str,
        context: str,
    ) -> dict[str, Any]:
        payload = read_bounded_regular_file(
            path,
            max_bytes=_ATTEMPT_RECEIPT_MAX_BYTES,
            label=context,
        )
        parsed = strict_json_object_loads(payload, context=context)
        if parsed.get("schema_version") != expected_schema:
            raise ValueError(f"{context} schema mismatch")
        expected_fields = (
            _ATTEMPT_RESERVATION_FIELDS
            if expected_schema == ATTEMPT_RESERVATION_SCHEMA_VERSION
            else _ATTEMPT_RESULT_FIELDS
        )
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
    "JobAttemptJournal",
    "JobAttemptReceiptRef",
    "JobAttemptScope",
]
