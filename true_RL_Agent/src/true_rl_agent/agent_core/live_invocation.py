"""Write-once lifecycle evidence for one live-loop invocation."""

from __future__ import annotations

import hashlib
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from typing import Any, Mapping

from true_rl_agent.evaluation.binding import require_sha256_digest
from true_rl_agent.fs_contract import (
    atomic_replace_bytes,
    create_once_bytes,
    read_bounded_regular_file,
)
from true_rl_agent.json_contract import canonical_json_bytes, strict_json_object_loads

LIVE_INVOCATION_START_SCHEMA_VERSION = "true_rl_agent.live_invocation.start.v1"
LIVE_INVOCATION_EXIT_SCHEMA_VERSION = "true_rl_agent.live_invocation.exit.v1"
LIVE_INVOCATION_EXIT_STATUSES = frozenset({"completed", "interrupted", "failed"})
LIVE_LOOP_SUMMARY_SCHEMA_VERSION = "true_rl_agent.live_loop.summary.v6"
LIVE_LOOP_SUMMARY_FIELDS = frozenset(
    {
        "schema_version",
        "mode",
        "session_id",
        "session_genesis_hash",
        "ledger_epoch_hash",
        "session_genesis_kind",
        "created_at_utc",
        "output_dir",
        "session_genesis_dir",
        "ledger_epoch",
        "ledger_path",
        "rounds_requested",
        "rounds_completed",
        "resumed_from_round_count",
        "rounds_executed_this_invocation",
        "prompt_payload_paths",
        "provider_attempt_count",
        "provider_attempt_count_for_rounds_closed_this_invocation",
        "authoring_evidence_paths",
        "provider_attempt_history_authority",
        "candidate_paths",
        "decision_outcome_counts",
        "runner_candidate_count",
        "trainable_round_count",
        "train_result_count",
        "public_result_count",
        "authored_candidate_count",
        "trainable_candidate_count",
        "trained_candidate_count",
        "full_eval_completed_count",
        "sealed_validation_completed_count",
        "accepted_count",
        "eval_incomplete_count",
        "eval_failure_bucket_counts",
        "provider_failure_count",
        "static_authoring_failure_count",
        "authoring_failure_round_count",
        "round_accounting",
        "final_best_score",
        "final_best_eval_stats",
        "checks",
        "eval_run_audit",
    }
)
LIVE_LOOP_SUMMARY_REQUIRED_CHECKS = frozenset(
    {
        "round_count_matches_request",
        "prompt_payload_count_matches_rounds",
        "authoring_provenance_classified",
        "authoring_evidence_count_matches_bound_rounds",
        "provider_attempt_history_complete",
        "runner_candidate_count_matches_trainable_rounds",
        "authoring_failures_accounted_for",
        "at_least_one_train_result",
        "at_least_one_public_result",
        "eval_run_audit_passed",
    }
)
MAX_LIVE_INVOCATION_RECORD_BYTES = 64 * 1024
# v6 keeps attempt details and prompt refs in the write-once authoring evidence
# referenced by the summary instead of copying them into this projection.
MAX_LIVE_SUMMARY_BYTES = 256 * 1024
MAX_LIVE_SUMMARY_PATH_BYTES = 4096
MAX_LIVE_INVOCATION_COUNT = 256


@dataclass(frozen=True)
class VerifiedCompletedLiveInvocation:
    invocation_id: str
    invocation_dir: Path
    summary: Mapping[str, Any]
    summary_checksum: str
    summary_size_bytes: int


@dataclass(frozen=True)
class LiveInvocationJournal:
    invocation_id: str
    invocation_dir: Path
    session_id: str
    session_genesis_hash: str
    ledger_epoch_hash: str
    rounds_requested: int
    closed_ledger_count_at_start: int
    started_at_utc: str

    @classmethod
    def start(
        cls,
        *,
        output_dir: Path,
        session_id: str,
        session_genesis_hash: str,
        ledger_epoch_hash: str,
        rounds_requested: int,
        closed_ledger_count: int,
    ) -> "LiveInvocationJournal":
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("live invocation session_id must be non-empty")
        if (
            isinstance(rounds_requested, bool)
            or not isinstance(rounds_requested, int)
            or rounds_requested <= 0
        ):
            raise ValueError("live invocation rounds_requested must be positive")
        _require_ledger_count(closed_ledger_count)
        genesis_hash = require_sha256_digest(
            session_genesis_hash,
            "live invocation session Genesis hash",
        )
        epoch_hash = require_sha256_digest(
            ledger_epoch_hash,
            "live invocation ledger epoch hash",
        )
        invocation_id = uuid.uuid4().hex
        invocation_root = output_dir.resolve() / "live_invocations"
        invocation_root.mkdir(parents=True, exist_ok=True)
        if invocation_root.is_symlink() or not invocation_root.is_dir():
            raise ValueError("live invocation root must be a regular directory")
        invocation_dir = invocation_root / invocation_id
        invocation_dir.mkdir(exist_ok=False)
        started_at_utc = _utc_now()
        journal = cls(
            invocation_id=invocation_id,
            invocation_dir=invocation_dir,
            session_id=session_id,
            session_genesis_hash=genesis_hash,
            ledger_epoch_hash=epoch_hash,
            rounds_requested=rounds_requested,
            closed_ledger_count_at_start=closed_ledger_count,
            started_at_utc=started_at_utc,
        )
        _publish_json_once(
            invocation_dir / "start.json",
            {
                "schema_version": LIVE_INVOCATION_START_SCHEMA_VERSION,
                "invocation_id": invocation_id,
                "session_id": session_id,
                "session_genesis_hash": genesis_hash,
                "ledger_epoch_hash": epoch_hash,
                "process_id": os.getpid(),
                "started_at_utc": started_at_utc,
                "rounds_requested": rounds_requested,
                "closed_ledger_count_at_start": closed_ledger_count,
                "ledger_authority": False,
            },
        )
        return journal

    def finish(
        self,
        *,
        status: str,
        termination_kind: str,
        closed_ledger_count: int,
        exception_type: str | None = None,
        live_summary_path: Path | None = None,
    ) -> Path:
        if status not in LIVE_INVOCATION_EXIT_STATUSES:
            raise ValueError("unsupported live invocation exit status")
        if not isinstance(termination_kind, str) or not termination_kind:
            raise ValueError("live invocation termination_kind must be non-empty")
        _require_ledger_count(closed_ledger_count)
        if not (
            self.closed_ledger_count_at_start
            <= closed_ledger_count
            <= self.rounds_requested
        ):
            raise ValueError("live invocation ledger counts are not monotonic")
        if exception_type is not None and (
            not isinstance(exception_type, str) or not exception_type
        ):
            raise ValueError("live invocation exception_type must be non-empty")
        allowed_termination_kinds = {
            "completed": {"normal"},
            "interrupted": {"keyboard_interrupt"},
            "failed": {"exception", "checks_failed"},
        }[status]
        if termination_kind not in allowed_termination_kinds:
            raise ValueError("live invocation status and termination kind mismatch")
        if (termination_kind in {"normal", "checks_failed"}) != (
            exception_type is None
        ):
            raise ValueError("live invocation exception type does not match status")
        if status == "completed" and closed_ledger_count != self.rounds_requested:
            raise ValueError("completed invocation must close all requested rounds")
        summary_ref = self._summary_snapshot_ref(
            status=status,
            termination_kind=termination_kind,
            closed_ledger_count=closed_ledger_count,
            live_summary_path=live_summary_path,
        )
        exit_record: dict[str, Any] = {
            "schema_version": LIVE_INVOCATION_EXIT_SCHEMA_VERSION,
            "invocation_id": self.invocation_id,
            "session_id": self.session_id,
            "session_genesis_hash": self.session_genesis_hash,
            "ledger_epoch_hash": self.ledger_epoch_hash,
            "started_at_utc": self.started_at_utc,
            "exited_at_utc": _utc_now(),
            "status": status,
            "termination_kind": termination_kind,
            "closed_ledger_count_at_exit": closed_ledger_count,
            "ledger_authority": False,
            "partial_rounds_are_not_ledger_records": True,
        }
        if exception_type is not None:
            exit_record["exception_type"] = exception_type
        if summary_ref is not None:
            exit_record["live_summary"] = summary_ref
        path = self.invocation_dir / "exit.json"
        _publish_json_once(path, exit_record)
        return path

    def _summary_snapshot_ref(
        self,
        *,
        status: str,
        termination_kind: str,
        closed_ledger_count: int,
        live_summary_path: Path | None,
    ) -> dict[str, Any] | None:
        summary_required = status == "completed" or termination_kind == "checks_failed"
        if not summary_required:
            if live_summary_path is not None:
                raise ValueError("this invocation exit cannot bind a live summary")
            return None
        if live_summary_path is None:
            raise ValueError("completed invocation must bind its live summary")
        expected = self.invocation_dir.parent.parent / "live_summary.json"
        supplied = Path(os.path.abspath(live_summary_path))
        expected_absolute = Path(os.path.abspath(expected))
        if supplied != expected_absolute:
            raise ValueError("live invocation summary path mismatch")
        payload = read_bounded_regular_file(
            supplied,
            max_bytes=MAX_LIVE_SUMMARY_BYTES,
            label="completed live summary",
        )
        if not payload:
            raise ValueError("completed live summary must be non-empty")
        summary = strict_json_object_loads(payload, context="live summary")
        if payload != canonical_json_bytes(summary, trailing_newline=True):
            raise ValueError("completed live summary is not canonical")
        _validate_live_summary_identity(
            summary,
            session_id=self.session_id,
            session_genesis_hash=self.session_genesis_hash,
            ledger_epoch_hash=self.ledger_epoch_hash,
            rounds_requested=self.rounds_requested,
            rounds_completed=closed_ledger_count,
            resumed_from_round_count=self.closed_ledger_count_at_start,
        )
        checks_passed = all(summary["checks"].values())
        if checks_passed != (status == "completed"):
            raise ValueError("live invocation summary checks do not match exit status")
        snapshot = self.invocation_dir / "live_summary.json"
        create_once_bytes(snapshot, payload)
        return {
            "path": str(snapshot.resolve()),
            "checksum": "sha256:" + hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }


def find_verified_completed_live_invocation(
    *,
    output_dir: Path,
    session_id: str,
    session_genesis_hash: str,
    ledger_epoch_hash: str,
    rounds_requested: int,
) -> VerifiedCompletedLiveInvocation:
    """Find the unique completed invocation bound to the current root summary."""

    output_root = Path(output_dir).resolve(strict=True)
    invocation_root = output_root / "live_invocations"
    if invocation_root.is_symlink() or not invocation_root.is_dir():
        raise ValueError("live invocation root must be a regular directory")
    candidates = sorted(
        islice(invocation_root.iterdir(), MAX_LIVE_INVOCATION_COUNT + 1),
        key=lambda path: path.name,
    )
    if len(candidates) > MAX_LIVE_INVOCATION_COUNT:
        raise ValueError("live invocation count exceeds the verification limit")
    root_summary_payload = read_bounded_regular_file(
        output_root / "live_summary.json",
        max_bytes=MAX_LIVE_SUMMARY_BYTES,
        label="root live summary",
    )
    verified: list[VerifiedCompletedLiveInvocation] = []
    last_rejection: ValueError | None = None
    for candidate in candidates:
        try:
            item = _verify_completed_live_invocation(
                output_root=output_root,
                invocation_dir=candidate,
                session_id=session_id,
                session_genesis_hash=session_genesis_hash,
                ledger_epoch_hash=ledger_epoch_hash,
                rounds_requested=rounds_requested,
                root_summary_payload=root_summary_payload,
            )
        except ValueError as exc:
            last_rejection = exc
            continue
        verified.append(item)
    if not verified:
        raise ValueError(
            "no completed live invocation binds the current summary"
        ) from last_rejection
    if len(verified) != 1:
        raise ValueError("multiple completed live invocations bind the current summary")
    return verified[0]


def _verify_completed_live_invocation(
    *,
    output_root: Path,
    invocation_dir: Path,
    session_id: str,
    session_genesis_hash: str,
    ledger_epoch_hash: str,
    rounds_requested: int,
    root_summary_payload: bytes,
) -> VerifiedCompletedLiveInvocation:
    if invocation_dir.is_symlink() or not invocation_dir.is_dir():
        raise ValueError("live invocation must be a regular directory")
    if invocation_dir.parent.resolve(strict=True) != (
        output_root / "live_invocations"
    ).resolve(strict=True):
        raise ValueError("live invocation escaped its output root")
    invocation_id = invocation_dir.name
    try:
        if uuid.UUID(hex=invocation_id).hex != invocation_id:
            raise ValueError
    except (AttributeError, ValueError) as exc:
        raise ValueError("live invocation id must be canonical UUID hex") from exc

    start_payload = read_bounded_regular_file(
        invocation_dir / "start.json",
        max_bytes=MAX_LIVE_INVOCATION_RECORD_BYTES,
        label="live invocation start",
    )
    exit_payload = read_bounded_regular_file(
        invocation_dir / "exit.json",
        max_bytes=MAX_LIVE_INVOCATION_RECORD_BYTES,
        label="live invocation exit",
    )
    snapshot_payload = read_bounded_regular_file(
        invocation_dir / "live_summary.json",
        max_bytes=MAX_LIVE_SUMMARY_BYTES,
        label="live invocation summary snapshot",
    )
    start = strict_json_object_loads(start_payload, context="live invocation start")
    exit_record = strict_json_object_loads(exit_payload, context="live invocation exit")
    if start_payload != canonical_json_bytes(start, trailing_newline=True):
        raise ValueError("live invocation start is not canonical")
    if exit_payload != canonical_json_bytes(exit_record, trailing_newline=True):
        raise ValueError("live invocation exit is not canonical")
    _validate_completed_invocation_records(
        start=start,
        exit_record=exit_record,
        invocation_id=invocation_id,
        session_id=session_id,
        session_genesis_hash=session_genesis_hash,
        ledger_epoch_hash=ledger_epoch_hash,
        rounds_requested=rounds_requested,
    )
    summary_ref = exit_record["live_summary"]
    snapshot = invocation_dir / "live_summary.json"
    if set(summary_ref) != {"path", "checksum", "size_bytes"}:
        raise ValueError("live invocation summary reference fields mismatch")
    if summary_ref["path"] != str(snapshot.resolve(strict=True)):
        raise ValueError("live invocation summary reference path mismatch")
    if summary_ref["size_bytes"] != len(snapshot_payload):
        raise ValueError("live invocation summary reference size mismatch")
    checksum = "sha256:" + hashlib.sha256(snapshot_payload).hexdigest()
    if summary_ref["checksum"] != checksum:
        raise ValueError("live invocation summary reference checksum mismatch")
    if snapshot_payload != root_summary_payload:
        raise ValueError("live invocation snapshot differs from root live summary")
    summary = strict_json_object_loads(
        snapshot_payload, context="live summary snapshot"
    )
    if snapshot_payload != canonical_json_bytes(summary, trailing_newline=True):
        raise ValueError("live invocation summary snapshot is not canonical")
    _validate_live_summary_identity(
        summary,
        session_id=session_id,
        session_genesis_hash=session_genesis_hash,
        ledger_epoch_hash=ledger_epoch_hash,
        rounds_requested=rounds_requested,
        rounds_completed=rounds_requested,
        resumed_from_round_count=start["closed_ledger_count_at_start"],
    )
    if not all(summary["checks"].values()):
        raise ValueError("completed live invocation summary checks did not pass")
    return VerifiedCompletedLiveInvocation(
        invocation_id=invocation_id,
        invocation_dir=invocation_dir.resolve(strict=True),
        summary=summary,
        summary_checksum=checksum,
        summary_size_bytes=len(snapshot_payload),
    )


def _validate_completed_invocation_records(
    *,
    start: Mapping[str, Any],
    exit_record: Mapping[str, Any],
    invocation_id: str,
    session_id: str,
    session_genesis_hash: str,
    ledger_epoch_hash: str,
    rounds_requested: int,
) -> None:
    start_fields = {
        "schema_version",
        "invocation_id",
        "session_id",
        "session_genesis_hash",
        "ledger_epoch_hash",
        "process_id",
        "started_at_utc",
        "rounds_requested",
        "closed_ledger_count_at_start",
        "ledger_authority",
    }
    exit_fields = {
        "schema_version",
        "invocation_id",
        "session_id",
        "session_genesis_hash",
        "ledger_epoch_hash",
        "started_at_utc",
        "exited_at_utc",
        "status",
        "termination_kind",
        "closed_ledger_count_at_exit",
        "ledger_authority",
        "partial_rounds_are_not_ledger_records",
        "live_summary",
    }
    if set(start) != start_fields or set(exit_record) != exit_fields:
        raise ValueError("completed live invocation record fields mismatch")
    shared_expected = {
        "invocation_id": invocation_id,
        "session_id": session_id,
        "session_genesis_hash": require_sha256_digest(
            session_genesis_hash, "completed invocation session Genesis hash"
        ),
        "ledger_epoch_hash": require_sha256_digest(
            ledger_epoch_hash, "completed invocation ledger epoch hash"
        ),
    }
    if any(start.get(key) != value for key, value in shared_expected.items()):
        raise ValueError("live invocation start identity mismatch")
    if any(exit_record.get(key) != value for key, value in shared_expected.items()):
        raise ValueError("live invocation exit identity mismatch")
    process_id = start.get("process_id")
    closed_at_start = start.get("closed_ledger_count_at_start")
    if (
        start.get("schema_version") != LIVE_INVOCATION_START_SCHEMA_VERSION
        or isinstance(process_id, bool)
        or not isinstance(process_id, int)
        or process_id <= 0
        or not isinstance(start.get("started_at_utc"), str)
        or not start["started_at_utc"]
        or start.get("rounds_requested") != rounds_requested
        or isinstance(closed_at_start, bool)
        or not isinstance(closed_at_start, int)
        or not 0 <= closed_at_start <= rounds_requested
        or start.get("ledger_authority") is not False
    ):
        raise ValueError("live invocation start contract mismatch")
    if (
        exit_record.get("schema_version") != LIVE_INVOCATION_EXIT_SCHEMA_VERSION
        or exit_record.get("started_at_utc") != start["started_at_utc"]
        or not isinstance(exit_record.get("exited_at_utc"), str)
        or not exit_record["exited_at_utc"]
        or exit_record.get("status") != "completed"
        or exit_record.get("termination_kind") != "normal"
        or exit_record.get("closed_ledger_count_at_exit") != rounds_requested
        or exit_record.get("ledger_authority") is not False
        or exit_record.get("partial_rounds_are_not_ledger_records") is not True
        or not isinstance(exit_record.get("live_summary"), Mapping)
    ):
        raise ValueError("live invocation exit contract mismatch")


def _validate_live_summary_identity(
    summary: Mapping[str, Any],
    *,
    session_id: str,
    session_genesis_hash: str,
    ledger_epoch_hash: str,
    rounds_requested: int,
    rounds_completed: int,
    resumed_from_round_count: int,
) -> None:
    checks = summary.get("checks")
    if (
        set(summary) != LIVE_LOOP_SUMMARY_FIELDS
        or summary.get("schema_version") != LIVE_LOOP_SUMMARY_SCHEMA_VERSION
        or summary.get("mode") != "live"
        or summary.get("session_id") != session_id
        or summary.get("session_genesis_hash")
        != require_sha256_digest(
            session_genesis_hash, "live summary session Genesis hash"
        )
        or summary.get("ledger_epoch_hash")
        != require_sha256_digest(ledger_epoch_hash, "live summary ledger epoch hash")
        or summary.get("rounds_requested") != rounds_requested
        or summary.get("rounds_completed") != rounds_completed
        or summary.get("resumed_from_round_count") != resumed_from_round_count
        or not isinstance(checks, Mapping)
        or set(checks) != LIVE_LOOP_SUMMARY_REQUIRED_CHECKS
        or any(type(value) is not bool for value in checks.values())
    ):
        raise ValueError("live invocation summary identity mismatch")
    _validate_live_summary_projection(summary)


def _validate_live_summary_projection(summary: Mapping[str, Any]) -> None:
    rounds = _nonnegative_int(summary.get("rounds_completed"), "rounds_completed")
    prompt_paths = _bounded_path_list(
        summary.get("prompt_payload_paths"),
        "prompt_payload_paths",
        max_items=rounds,
    )
    evidence_paths = _bounded_path_list(
        summary.get("authoring_evidence_paths"),
        "authoring_evidence_paths",
        max_items=rounds,
    )
    attempt_count = _nonnegative_int(
        summary.get("provider_attempt_count"), "provider_attempt_count"
    )
    candidate_paths = _bounded_path_list(
        summary.get("candidate_paths"), "candidate_paths", max_items=rounds
    )
    authority = _mapping(
        summary.get("provider_attempt_history_authority"),
        "provider_attempt_history_authority",
    )
    attempt_count_for_closed_rounds = _nonnegative_int(
        summary.get("provider_attempt_count_for_rounds_closed_this_invocation"),
        "provider_attempt_count_for_rounds_closed_this_invocation",
    )
    if attempt_count_for_closed_rounds > attempt_count:
        raise ValueError("live summary closed-round attempt count exceeds total")
    eval_audit = _mapping(summary.get("eval_run_audit"), "eval_run_audit")
    eval_counts = _mapping(eval_audit.get("counts"), "eval_run_audit.counts")
    eval_checks = _mapping(eval_audit.get("checks"), "eval_run_audit.checks")
    eval_violations = _mapping(
        eval_audit.get("violations"), "eval_run_audit.violations"
    )
    audit_rounds = _list(eval_audit.get("rounds"), "eval_run_audit.rounds")
    if eval_audit.get("schema_version") != "true_rl_agent.eval_run_audit.v1":
        raise ValueError("live summary eval audit schema mismatch")
    if any(type(value) is not bool for value in eval_checks.values()):
        raise ValueError("live summary eval audit checks must be booleans")
    if any(not isinstance(value, list) for value in eval_violations.values()):
        raise ValueError("live summary eval audit violations must be lists")
    if len(audit_rounds) != rounds:
        raise ValueError("live summary eval audit round count mismatch")
    for field_name in ("output_dir", "session_genesis_dir", "ledger_path"):
        _bounded_path(summary.get(field_name), field_name)
    ledger_epoch = summary.get("ledger_epoch")
    if isinstance(ledger_epoch, Mapping) and "manifest_path" in ledger_epoch:
        _bounded_path(ledger_epoch.get("manifest_path"), "ledger_epoch.manifest_path")
    resumed = _nonnegative_int(
        summary.get("resumed_from_round_count"), "resumed_from_round_count"
    )
    executed = _nonnegative_int(
        summary.get("rounds_executed_this_invocation"),
        "rounds_executed_this_invocation",
    )
    if resumed + executed != rounds:
        raise ValueError("live summary invocation round accounting mismatch")

    provider_failures = _nonnegative_int(
        summary.get("provider_failure_count"), "provider_failure_count"
    )
    static_failures = _nonnegative_int(
        summary.get("static_authoring_failure_count"),
        "static_authoring_failure_count",
    )
    authoring_failures = _nonnegative_int(
        summary.get("authoring_failure_round_count"),
        "authoring_failure_round_count",
    )
    trainable_rounds = _nonnegative_int(
        summary.get("trainable_round_count"), "trainable_round_count"
    )
    runner_candidates = _nonnegative_int(
        summary.get("runner_candidate_count"), "runner_candidate_count"
    )
    bound_rounds = _nonnegative_int(
        authority.get("bound_round_count"),
        "provider_attempt_history_authority.bound_round_count",
    )
    evidence_count = _nonnegative_int(
        authority.get("round_evidence_count"),
        "provider_attempt_history_authority.round_evidence_count",
    )
    _nonnegative_int(
        authority.get("interrupted_attempt_count"),
        "provider_attempt_history_authority.interrupted_attempt_count",
    )
    if (
        set(authority)
        != {
            "mode",
            "latest_summary_is_projection",
            "round_evidence_count",
            "bound_round_count",
            "interrupted_attempt_count",
        }
        or authority.get("mode") != "ledger_bound_immutable_round_artifacts"
        or authority.get("latest_summary_is_projection") is not True
        or evidence_count != len(evidence_paths)
    ):
        raise ValueError("live summary authoring history authority mismatch")
    if authoring_failures != provider_failures + static_failures:
        raise ValueError("live summary authoring failure accounting mismatch")
    if runner_candidates != len(candidate_paths):
        raise ValueError("live summary candidate path accounting mismatch")

    count_fields = {
        "authored_candidate_count": "authored_candidate_count",
        "trainable_candidate_count": "trainable_candidate_count",
        "trained_candidate_count": "trained_candidate_count",
        "full_eval_completed_count": "full_eval_completed_count",
        "sealed_validation_completed_count": "sealed_validation_completed_count",
        "accepted_count": "accepted_count",
        "eval_incomplete_count": "eval_incomplete_count",
    }
    for summary_field, audit_field in count_fields.items():
        if _nonnegative_int(summary.get(summary_field), summary_field) != (
            _nonnegative_int(
                eval_counts.get(audit_field), f"eval_run_audit.counts.{audit_field}"
            )
        ):
            raise ValueError(f"live summary {summary_field} differs from eval audit")
    if summary.get("eval_failure_bucket_counts") != eval_audit.get("failure_buckets"):
        raise ValueError("live summary eval failure buckets mismatch")
    outcome_counts = _mapping(
        summary.get("decision_outcome_counts"), "decision_outcome_counts"
    )
    if (
        any(not isinstance(key, str) or not key for key in outcome_counts)
        or any(
            _nonnegative_int(value, f"decision_outcome_counts.{key}") < 0
            for key, value in outcome_counts.items()
        )
        or sum(outcome_counts.values()) != rounds
    ):
        raise ValueError("live summary decision outcome accounting mismatch")

    expected_checks = {
        "round_count_matches_request": rounds == summary.get("rounds_requested"),
        "prompt_payload_count_matches_rounds": len(prompt_paths) == rounds,
        "authoring_provenance_classified": bound_rounds == rounds,
        "authoring_evidence_count_matches_bound_rounds": (
            len(evidence_paths) == bound_rounds
        ),
        "provider_attempt_history_complete": attempt_count >= bound_rounds,
        "runner_candidate_count_matches_trainable_rounds": (
            runner_candidates == trainable_rounds
        ),
        "authoring_failures_accounted_for": (
            rounds == trainable_rounds + authoring_failures
        ),
        "at_least_one_train_result": (
            _nonnegative_int(summary.get("train_result_count"), "train_result_count")
            > 0
        ),
        "at_least_one_public_result": (
            _nonnegative_int(summary.get("public_result_count"), "public_result_count")
            > 0
        ),
        "eval_run_audit_passed": bool(eval_checks) and all(eval_checks.values()),
    }
    if dict(summary["checks"]) != expected_checks:
        raise ValueError("live summary checks differ from their source facts")
    round_accounting = _mapping(summary.get("round_accounting"), "round_accounting")
    if set(round_accounting) != {
        "rounds_completed",
        "runner_candidate_count",
        "trainable_round_count",
        "provider_failure_count",
        "static_authoring_failure_count",
        "candidate_paths_semantics",
    }:
        raise ValueError("live summary round accounting fields mismatch")
    for key, expected in (
        ("rounds_completed", rounds),
        ("runner_candidate_count", runner_candidates),
        ("trainable_round_count", trainable_rounds),
        ("provider_failure_count", provider_failures),
        ("static_authoring_failure_count", static_failures),
    ):
        if round_accounting.get(key) != expected:
            raise ValueError("live summary round accounting mismatch")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    return value


def _bounded_path_list(value: Any, label: str, *, max_items: int) -> list[str]:
    items = _list(value, label)
    if len(items) > max_items:
        raise ValueError(f"{label} has too many entries")
    return [
        _bounded_path(item, f"{label}[{index}]") for index, item in enumerate(items)
    ]


def _bounded_path(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value.encode("utf-8")) > MAX_LIVE_SUMMARY_PATH_BYTES
    ):
        raise ValueError(f"{label} must be a bounded non-empty path")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _require_ledger_count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("closed ledger count must be a non-negative integer")
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _publish_json_once(path: Path, value: dict[str, Any]) -> None:
    payload = canonical_json_bytes(value, trailing_newline=True)
    create_once_bytes(path, payload)


def write_live_summary(path: Path, summary: Mapping[str, Any]) -> None:
    """Publish the bounded root summary after validating its fixed envelope."""

    payload = canonical_json_bytes(dict(summary), trailing_newline=True)
    if len(payload) > MAX_LIVE_SUMMARY_BYTES:
        raise ValueError("live summary exceeds the byte limit")
    target = Path(path)
    if target.parent.is_symlink() or not target.parent.is_dir():
        raise ValueError("live summary parent must be a regular directory")
    atomic_replace_bytes(target, payload)


__all__ = [
    "LIVE_INVOCATION_EXIT_SCHEMA_VERSION",
    "LIVE_INVOCATION_EXIT_STATUSES",
    "LIVE_INVOCATION_START_SCHEMA_VERSION",
    "LIVE_LOOP_SUMMARY_FIELDS",
    "LIVE_LOOP_SUMMARY_REQUIRED_CHECKS",
    "LIVE_LOOP_SUMMARY_SCHEMA_VERSION",
    "MAX_LIVE_INVOCATION_COUNT",
    "MAX_LIVE_INVOCATION_RECORD_BYTES",
    "MAX_LIVE_SUMMARY_BYTES",
    "LiveInvocationJournal",
    "VerifiedCompletedLiveInvocation",
    "find_verified_completed_live_invocation",
    "write_live_summary",
]
