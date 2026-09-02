"""Create-once operational timing evidence for Stair generation authoring.

This module observes one lane-authoring callable in the generation process.  It
does not choose candidates, trigger retries, reserve budget, or interpret the
scientific result.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
import time
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

from true_rl_agent.agent_core.generation_contract import GENERATION_LANE_IDS
from true_rl_agent.fs_contract import create_once_bytes, read_bounded_regular_file
from true_rl_agent.json_contract import canonical_json_bytes, strict_json_object_loads

STAIR_GENERATION_AUTHORING_TIMING_DIRNAME = "generation_authoring_timing"
STAIR_GENERATION_AUTHORING_TIMING_START_SCHEMA_VERSION = (
    "true_rl_agent.stair.generation_authoring_timing.start.v1"
)
STAIR_GENERATION_AUTHORING_TIMING_STOP_SCHEMA_VERSION = (
    "true_rl_agent.stair.generation_authoring_timing.stop.v1"
)
STAIR_GENERATION_AUTHORING_TIMING_SUMMARY_SCHEMA_VERSION = (
    "true_rl_agent.stair.generation_authoring_timing.summary.v1"
)

_MAX_RECEIPT_BYTES = 32 * 1024
_MAX_ATTEMPTS_PER_LANE = 1_000_000
_GENERATION_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_ATTEMPT_DIR_PATTERN = re.compile(r"attempt-(?P<index>[0-9]{6})\Z")
_AUTHORITY = {
    "observer_only": True,
    "authoring_control": False,
    "retry_authority": False,
    "budget_authority": False,
    "scientific_state_authority": False,
}
_START_KEYS = {
    "schema_version",
    "generation_id",
    "lane_id",
    "attempt_id",
    "attempt_index",
    "process_id",
    "started_at_utc",
    "started_monotonic_ns",
    "authority",
}
_STOP_KEYS = {
    "schema_version",
    "generation_id",
    "lane_id",
    "attempt_id",
    "attempt_index",
    "process_id",
    "stopped_at_utc",
    "stopped_monotonic_ns",
    "duration_ns",
    "outcome",
    "exception_type",
    "start_receipt",
    "authority",
}

_ResultT = TypeVar("_ResultT")


def run_stair_generation_authoring_attempt(
    *,
    evidence_root: Path,
    generation_id: str,
    lane_id: str,
    author: Callable[[], _ResultT],
    monotonic_ns: Callable[[], int] = time.monotonic_ns,
    utc_now: Callable[[], str] | None = None,
) -> _ResultT:
    """Run one lane authoring call and retain its operational wall duration."""

    if not callable(author):
        raise TypeError("Stair generation author must be callable")
    if not callable(monotonic_ns):
        raise TypeError("Stair authoring monotonic clock must be callable")
    if utc_now is not None and not callable(utc_now):
        raise TypeError("Stair authoring UTC clock must be callable")
    utc_clock = utc_now or _utc_now
    root = _prepare_evidence_root(evidence_root)
    generation_id = _generation_id(generation_id)
    lane_id = _lane_id(lane_id)
    lane_root = _prepare_lane_root(root, generation_id, lane_id)
    attempt_index, attempt_id, attempt_dir = _allocate_attempt(
        lane_root,
        generation_id=generation_id,
        lane_id=lane_id,
    )

    process_id = os.getpid()
    started_ns = _monotonic_value(
        monotonic_ns(),
        "Stair authoring start monotonic_ns",
    )
    started_at_utc = _utc_value(utc_clock(), "Stair authoring start UTC")
    start_record = {
        "schema_version": STAIR_GENERATION_AUTHORING_TIMING_START_SCHEMA_VERSION,
        "generation_id": generation_id,
        "lane_id": lane_id,
        "attempt_id": attempt_id,
        "attempt_index": attempt_index,
        "process_id": process_id,
        "started_at_utc": started_at_utc,
        "started_monotonic_ns": started_ns,
        "authority": dict(_AUTHORITY),
    }
    start_payload = canonical_json_bytes(start_record, trailing_newline=True)
    create_once_bytes(attempt_dir / "start.json", start_payload)

    try:
        result = author()
    except BaseException as exc:
        try:
            _publish_stop(
                attempt_dir=attempt_dir,
                start_record=start_record,
                start_payload=start_payload,
                outcome="raised",
                exception_type=type(exc).__name__[:128],
                monotonic_ns=monotonic_ns,
                utc_now=utc_clock,
            )
        except BaseException as timing_error:  # noqa: BLE001 - preserve author error
            add_note = getattr(exc, "add_note", None)
            if callable(add_note):
                add_note(
                    "Stair authoring timing stop evidence also failed: "
                    + type(timing_error).__name__
                )
        raise

    _publish_stop(
        attempt_dir=attempt_dir,
        start_record=start_record,
        start_payload=start_payload,
        outcome="returned",
        exception_type=None,
        monotonic_ns=monotonic_ns,
        utc_now=utc_clock,
    )
    return result


def replay_stair_generation_authoring_timing(
    *,
    evidence_root: Path,
    generation_id: str,
) -> dict[str, Any]:
    """Strictly replay one generation without consulting filesystem times."""

    root = _existing_real_directory(
        _absolute_path(evidence_root, "Stair authoring timing evidence_root"),
        "Stair authoring timing evidence root",
    )
    generation_id = _generation_id(generation_id)
    generation_root = _existing_real_directory(
        root / generation_id,
        "Stair authoring timing generation root",
    )
    unknown_lanes = {
        entry.name
        for entry in generation_root.iterdir()
        if entry.name not in GENERATION_LANE_IDS
    }
    if unknown_lanes:
        raise ValueError("Stair authoring timing generation has unknown entries")

    lane_measured_closed = {lane_id: 0 for lane_id in GENERATION_LANE_IDS}
    stopped_attempts: list[dict[str, Any]] = []
    incomplete_attempts: list[dict[str, Any]] = []
    for lane_id in GENERATION_LANE_IDS:
        lane_root = generation_root / lane_id
        if not lane_root.exists() and not lane_root.is_symlink():
            continue
        lane_root = _existing_real_directory(
            lane_root,
            "Stair authoring timing lane root",
        )
        for attempt_index, attempt_dir in _attempt_directories(lane_root):
            start, stop = _load_attempt(
                attempt_dir,
                generation_id=generation_id,
                lane_id=lane_id,
                attempt_index=attempt_index,
            )
            identity = {
                "generation_id": generation_id,
                "lane_id": lane_id,
                "attempt_id": start["attempt_id"],
                "attempt_index": attempt_index,
                "attempt_path": str(attempt_dir.resolve(strict=True)),
                "process_id": start["process_id"],
                "started_at_utc": start["started_at_utc"],
                "started_monotonic_ns": start["started_monotonic_ns"],
            }
            if stop is None:
                incomplete_attempts.append(identity)
                continue
            duration_ns = stop["duration_ns"]
            lane_measured_closed[lane_id] += duration_ns
            stopped_attempts.append(
                {
                    **identity,
                    "stopped_at_utc": stop["stopped_at_utc"],
                    "stopped_monotonic_ns": stop["stopped_monotonic_ns"],
                    "duration_ns": duration_ns,
                    "outcome": stop["outcome"],
                    "exception_type": stop["exception_type"],
                }
            )
    measured_closed = sum(lane_measured_closed.values())
    timing_complete = not incomplete_attempts
    return {
        "schema_version": STAIR_GENERATION_AUTHORING_TIMING_SUMMARY_SCHEMA_VERSION,
        "generation_id": generation_id,
        "lane_measured_closed_attempt_wall_duration_ns": lane_measured_closed,
        "measured_closed_attempt_wall_duration_ns": measured_closed,
        "total_wall_duration_ns": measured_closed if timing_complete else None,
        "timing_complete": timing_complete,
        "stopped_attempts": stopped_attempts,
        "incomplete_attempts": incomplete_attempts,
        "authority": dict(_AUTHORITY),
    }


def _publish_stop(
    *,
    attempt_dir: Path,
    start_record: Mapping[str, Any],
    start_payload: bytes,
    outcome: str,
    exception_type: str | None,
    monotonic_ns: Callable[[], int],
    utc_now: Callable[[], str],
) -> None:
    stopped_ns = _monotonic_value(
        monotonic_ns(),
        "Stair authoring stop monotonic_ns",
    )
    started_ns = start_record["started_monotonic_ns"]
    if stopped_ns < started_ns:
        raise ValueError("Stair authoring monotonic clock regressed")
    if os.getpid() != start_record["process_id"]:
        raise RuntimeError("Stair authoring timing process identity changed")
    stop_record = {
        "schema_version": STAIR_GENERATION_AUTHORING_TIMING_STOP_SCHEMA_VERSION,
        "generation_id": start_record["generation_id"],
        "lane_id": start_record["lane_id"],
        "attempt_id": start_record["attempt_id"],
        "attempt_index": start_record["attempt_index"],
        "process_id": start_record["process_id"],
        "stopped_at_utc": _utc_value(utc_now(), "Stair authoring stop UTC"),
        "stopped_monotonic_ns": stopped_ns,
        "duration_ns": stopped_ns - started_ns,
        "outcome": outcome,
        "exception_type": exception_type,
        "start_receipt": {
            "sha256": "sha256:" + hashlib.sha256(start_payload).hexdigest(),
            "size_bytes": len(start_payload),
        },
        "authority": dict(_AUTHORITY),
    }
    create_once_bytes(
        attempt_dir / "stop.json",
        canonical_json_bytes(stop_record, trailing_newline=True),
    )


def _prepare_evidence_root(path: Path) -> Path:
    root = _absolute_path(path, "Stair authoring timing evidence_root")
    _existing_real_directory(root.parent, "Stair authoring timing parent")
    root.mkdir(mode=0o700, exist_ok=True)
    return _existing_real_directory(root, "Stair authoring timing evidence root")


def _prepare_lane_root(root: Path, generation_id: str, lane_id: str) -> Path:
    generation_root = root / generation_id
    generation_root.mkdir(mode=0o700, exist_ok=True)
    generation_root = _existing_real_directory(
        generation_root,
        "Stair authoring timing generation root",
    )
    lane_root = generation_root / lane_id
    lane_root.mkdir(mode=0o700, exist_ok=True)
    return _existing_real_directory(lane_root, "Stair authoring timing lane root")


def _allocate_attempt(
    lane_root: Path,
    *,
    generation_id: str,
    lane_id: str,
) -> tuple[int, str, Path]:
    attempts = _attempt_directories(lane_root)
    for attempt_index, attempt_dir in attempts:
        _load_attempt(
            attempt_dir,
            generation_id=generation_id,
            lane_id=lane_id,
            attempt_index=attempt_index,
        )
    next_index = len(attempts)
    if next_index >= _MAX_ATTEMPTS_PER_LANE:
        raise ValueError("Stair authoring timing attempt limit is exhausted")
    attempt_id = f"attempt-{next_index:06d}"
    attempt_dir = lane_root / attempt_id
    attempt_dir.mkdir(mode=0o700, exist_ok=False)
    return next_index, attempt_id, attempt_dir


def _attempt_directories(lane_root: Path) -> list[tuple[int, Path]]:
    attempts: list[tuple[int, Path]] = []
    for entry in lane_root.iterdir():
        match = _ATTEMPT_DIR_PATTERN.fullmatch(entry.name)
        if match is None:
            raise ValueError("Stair authoring timing lane has unknown entries")
        attempts.append(
            (
                int(match.group("index")),
                _existing_real_directory(
                    entry,
                    "Stair authoring timing attempt directory",
                ),
            )
        )
    attempts.sort(key=lambda item: item[0])
    if [index for index, _ in attempts] != list(range(len(attempts))):
        raise ValueError("Stair authoring timing attempt indices are not contiguous")
    return attempts


def _load_attempt(
    attempt_dir: Path,
    *,
    generation_id: str,
    lane_id: str,
    attempt_index: int,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    entries = {entry.name for entry in attempt_dir.iterdir()}
    if entries not in ({"start.json"}, {"start.json", "stop.json"}):
        raise ValueError("Stair authoring timing attempt files mismatch")
    start_path = attempt_dir / "start.json"
    start_payload, start = _load_canonical_receipt(start_path, "authoring start")
    _validate_start(
        start,
        generation_id=generation_id,
        lane_id=lane_id,
        attempt_index=attempt_index,
        attempt_id=attempt_dir.name,
    )
    stop_path = attempt_dir / "stop.json"
    if "stop.json" not in entries:
        return start, None
    _, stop = _load_canonical_receipt(stop_path, "authoring stop")
    _validate_stop(stop, start=start, start_payload=start_payload)
    return start, stop


def _validate_start(
    value: Mapping[str, Any],
    *,
    generation_id: str,
    lane_id: str,
    attempt_index: int,
    attempt_id: str,
) -> None:
    _exact_keys(value, _START_KEYS, "Stair authoring timing start")
    if (
        value["schema_version"]
        != STAIR_GENERATION_AUTHORING_TIMING_START_SCHEMA_VERSION
        or value["generation_id"] != generation_id
        or value["lane_id"] != lane_id
        or value["attempt_id"] != attempt_id
        or value["attempt_index"] != attempt_index
    ):
        raise ValueError("Stair authoring timing start identity mismatch")
    _positive_int(value["process_id"], "Stair authoring timing start process_id")
    _monotonic_value(
        value["started_monotonic_ns"],
        "Stair authoring timing start monotonic_ns",
    )
    _utc_value(value["started_at_utc"], "Stair authoring timing start UTC")
    _require_no_authority(value["authority"])


def _validate_stop(
    value: Mapping[str, Any],
    *,
    start: Mapping[str, Any],
    start_payload: bytes,
) -> None:
    _exact_keys(value, _STOP_KEYS, "Stair authoring timing stop")
    if value["schema_version"] != STAIR_GENERATION_AUTHORING_TIMING_STOP_SCHEMA_VERSION:
        raise ValueError("Stair authoring timing stop schema mismatch")
    for field in (
        "generation_id",
        "lane_id",
        "attempt_id",
        "attempt_index",
        "process_id",
    ):
        if value[field] != start[field]:
            raise ValueError("Stair authoring timing stop identity mismatch")
    stopped_ns = _monotonic_value(
        value["stopped_monotonic_ns"],
        "Stair authoring timing stop monotonic_ns",
    )
    duration_ns = _nonnegative_int(
        value["duration_ns"],
        "Stair authoring timing duration_ns",
    )
    if (
        stopped_ns < start["started_monotonic_ns"]
        or duration_ns != stopped_ns - start["started_monotonic_ns"]
    ):
        raise ValueError("Stair authoring timing duration mismatch")
    _utc_value(value["stopped_at_utc"], "Stair authoring timing stop UTC")
    outcome = value["outcome"]
    exception_type = value["exception_type"]
    if outcome == "returned":
        if exception_type is not None:
            raise ValueError("returned Stair authoring timing has an exception")
    elif outcome == "raised":
        _bounded_text(exception_type, "Stair authoring exception_type", maximum=128)
    else:
        raise ValueError("Stair authoring timing outcome is invalid")
    if value["start_receipt"] != {
        "sha256": "sha256:" + hashlib.sha256(start_payload).hexdigest(),
        "size_bytes": len(start_payload),
    }:
        raise ValueError("Stair authoring timing start receipt mismatch")
    _require_no_authority(value["authority"])


def _load_canonical_receipt(
    path: Path,
    label: str,
) -> tuple[bytes, dict[str, Any]]:
    payload = read_bounded_regular_file(
        path,
        max_bytes=_MAX_RECEIPT_BYTES,
        label=f"Stair {label}",
    )
    value = strict_json_object_loads(payload, context=f"Stair {label}")
    if payload != canonical_json_bytes(value, trailing_newline=True):
        raise ValueError(f"Stair {label} is not canonical JSON")
    return payload, value


def _absolute_path(value: Any, label: str) -> Path:
    if not isinstance(value, Path) or not value.is_absolute():
        raise ValueError(f"{label} must be an absolute Path")
    return value


def _existing_real_directory(path: Path, label: str) -> Path:
    try:
        inspected = path.lstat()
    except OSError as exc:
        raise ValueError(f"{label} is unavailable") from exc
    if not stat.S_ISDIR(inspected.st_mode):
        raise ValueError(f"{label} must be a real directory")
    return path.resolve(strict=True)


def _generation_id(value: Any) -> str:
    if not isinstance(value, str) or _GENERATION_ID_PATTERN.fullmatch(value) is None:
        raise ValueError("Stair authoring timing generation_id is invalid")
    return value


def _lane_id(value: Any) -> str:
    if value not in GENERATION_LANE_IDS:
        raise ValueError("Stair authoring timing lane_id is invalid")
    return value


def _monotonic_value(value: Any, label: str) -> int:
    return _nonnegative_int(value, label)


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _bounded_text(value: Any, label: str, *, maximum: int) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise ValueError(f"{label} must be non-empty bounded text")
    return value


def _utc_value(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{label} must be a UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError(f"{label} must be a UTC timestamp") from exc
    if parsed.utcoffset() != timezone.utc.utcoffset(None):
        raise ValueError(f"{label} must be a UTC timestamp")
    return value


def _require_no_authority(value: Any) -> None:
    if value != _AUTHORITY:
        raise ValueError("Stair authoring timing authority projection mismatch")


def _exact_keys(value: Any, fields: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{label} fields mismatch")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


__all__ = [
    "STAIR_GENERATION_AUTHORING_TIMING_DIRNAME",
    "STAIR_GENERATION_AUTHORING_TIMING_START_SCHEMA_VERSION",
    "STAIR_GENERATION_AUTHORING_TIMING_STOP_SCHEMA_VERSION",
    "STAIR_GENERATION_AUTHORING_TIMING_SUMMARY_SCHEMA_VERSION",
    "replay_stair_generation_authoring_timing",
    "run_stair_generation_authoring_attempt",
]
