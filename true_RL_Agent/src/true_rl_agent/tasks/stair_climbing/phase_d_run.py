"""Authoritative replay validation for one completed Stair Phase D pilot run."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from true_rl_agent.json_contract import (
    canonical_json_bytes,
    strict_json_object_loads,
)
from true_rl_agent.runner.checkpoint import file_sha256
from true_rl_agent.tasks.stair_climbing.local_paths import (
    resolve_existing_directory,
    resolve_existing_file,
)
from true_rl_agent.tasks.stair_climbing.pilot_artifacts import (
    load_canonical_pilot_json,
    summarize_stair_pilot_tensorboard,
    validate_stair_phase_d_report,
    validate_stair_pilot_eval_summary,
    validate_stair_pilot_seed_evidence,
    validate_stair_pilot_train_summary,
)
from true_rl_agent.tasks.stair_climbing.reference_pilot import (
    build_stair_pilot_config,
    stair_pilot_training_seed_index,
)
from true_rl_agent.tasks.stair_climbing.reference_program import (
    StairOfficialReferenceProgram,
    resolve_official_stair_reference_program,
)

STAIR_PHASE_D_PROCESS_EVENT_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.phase_d_process_event.v1"
)


def build_stair_phase_d_process_event(
    *,
    attempt_id: str,
    budget_steps: int,
    phase: str,
    status: str,
    has_exited: bool,
    exit_code: int | None,
    failure_message: str | None,
) -> dict[str, Any]:
    """Build one canonical process lifecycle event."""

    if phase not in {"train", "eval"}:
        raise ValueError("stair pilot process phase is invalid")
    if status not in {"started", "completed", "failed"}:
        raise ValueError("stair pilot process status is invalid")
    event = {
        "schema_version": STAIR_PHASE_D_PROCESS_EVENT_SCHEMA_VERSION,
        "event_id": f"{attempt_id}:{phase}:{status}",
        "attempt_id": attempt_id,
        "budget_steps": budget_steps,
        "phase": phase,
        "status": status,
        "occurred_at_utc": datetime.now(timezone.utc).isoformat(),
        "has_exited": has_exited,
        "exit_code": exit_code,
        "failure_message": failure_message,
    }
    validate_stair_phase_d_process_event(event)
    return event


def append_stair_phase_d_process_event(
    path: Path,
    event: Mapping[str, Any],
) -> None:
    """Durably append one validated canonical JSONL event."""

    validate_stair_phase_d_process_event(event)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(canonical_json_bytes(dict(event), trailing_newline=True))
        handle.flush()
        os.fsync(handle.fileno())


def validate_stair_phase_d_event_log(path: Path) -> list[dict[str, Any]]:
    """Validate canonical bytes, unique IDs, and complete process lifecycles."""

    path = resolve_existing_file(path, field_name="stair Phase D event log")
    events: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for line_number, raw_line in enumerate(path.read_bytes().splitlines(), start=1):
        if not raw_line:
            raise ValueError("stair Phase D event log contains an empty line")
        event = strict_json_object_loads(
            raw_line,
            context=f"stair Phase D event line {line_number}",
        )
        if raw_line != canonical_json_bytes(event):
            raise ValueError("stair Phase D event log line is not canonical")
        validate_stair_phase_d_process_event(event)
        if event["event_id"] in seen_ids:
            raise ValueError("stair Phase D event log contains duplicate event_id")
        seen_ids.add(event["event_id"])
        events.append(event)
    if not events:
        raise ValueError("stair Phase D event log is empty")

    lifecycle: dict[tuple[str, str], list[str]] = {}
    for event in events:
        key = (event["attempt_id"], event["phase"])
        lifecycle.setdefault(key, []).append(event["status"])
    for key, statuses in lifecycle.items():
        if len(statuses) != 2 or statuses[0] != "started" or statuses[1] not in {
            "completed",
            "failed",
        }:
            raise ValueError(
                "stair Phase D process lifecycle is incomplete: "
                f"attempt={key[0]} phase={key[1]}"
            )
    return events


def validate_stair_phase_d_process_event(event: Mapping[str, Any]) -> None:
    fields = {
        "schema_version",
        "event_id",
        "attempt_id",
        "budget_steps",
        "phase",
        "status",
        "occurred_at_utc",
        "has_exited",
        "exit_code",
        "failure_message",
    }
    if not isinstance(event, Mapping) or set(event) != fields:
        raise ValueError("stair Phase D process event fields mismatch")
    if event["schema_version"] != STAIR_PHASE_D_PROCESS_EVENT_SCHEMA_VERSION:
        raise ValueError("stair Phase D process event schema mismatch")
    event_id = _canonical_text(event["event_id"], "event_id")
    attempt_id = _canonical_text(event["attempt_id"], "attempt_id")
    if (
        isinstance(event["budget_steps"], bool)
        or not isinstance(event["budget_steps"], int)
        or event["budget_steps"] <= 0
    ):
        raise ValueError("stair Phase D event budget must be positive")
    if event["phase"] not in {"train", "eval"}:
        raise ValueError("stair Phase D event phase mismatch")
    if event["status"] not in {"started", "completed", "failed"}:
        raise ValueError("stair Phase D event status mismatch")
    if event_id != f"{attempt_id}:{event['phase']}:{event['status']}":
        raise ValueError("stair Phase D process event_id binding mismatch")
    _canonical_text(event["occurred_at_utc"], "occurred_at_utc")
    if not isinstance(event["has_exited"], bool):
        raise ValueError("stair Phase D event has_exited must be boolean")
    if event["status"] == "started":
        if event["has_exited"] or event["exit_code"] is not None:
            raise ValueError("stair Phase D started event exit state mismatch")
    elif event["status"] == "completed":
        if event["has_exited"] is not True or event["exit_code"] != 0:
            raise ValueError("stair Phase D completed event exit state mismatch")
        if event["failure_message"] is not None:
            raise ValueError("stair Phase D completed event has failure message")
    elif event["failure_message"] is None:
        raise ValueError("stair Phase D failed event requires failure message")


def reload_completed_stair_pilot_seed(
    *,
    config: Any,
    official: StairOfficialReferenceProgram,
    seed_root: Path,
    events_path: Path,
) -> dict[str, Any]:
    """Rebuild one seed result from raw summaries and checkpoint bytes."""

    seed = validate_stair_pilot_seed_evidence(
        config.protocol_v2_path,
        load_canonical_pilot_json(
            seed_root / "seed_evidence.json",
            context="stair pilot seed evidence",
        ),
    )
    if seed["training_seed_index"] != stair_pilot_training_seed_index(config):
        raise ValueError("stair pilot seed index differs from requested config")
    attempt_id = _canonical_text(seed["attempt_id"], "attempt_id")
    attempt_root = seed_root / "attempts" / attempt_id
    checkpoint_path = attempt_root / "checkpoint.pt"
    train = validate_stair_pilot_train_summary(
        config,
        official=official,
        summary=load_canonical_pilot_json(
            attempt_root / "train_summary.json",
            context="stair pilot train summary",
        ),
        checkpoint_path=checkpoint_path,
        expected_attempt_id=attempt_id,
    )
    replayed_curve_summary = summarize_stair_pilot_tensorboard(
        attempt_root / "train_run" / "rsl_rl",
        max_iterations=config.train_max_iterations,
        diagnostics_contract=config.protocol_v2.pilot_training_contract[
            "curve_diagnostics"
        ],
    )
    if replayed_curve_summary != train["curve_summary"]:
        raise ValueError("stair pilot TensorBoard curve evidence drifted")
    checkpoint_hash = train["execution"]["checkpoint"]["checksum"]
    evaluation = validate_stair_pilot_eval_summary(
        config,
        summary=load_canonical_pilot_json(
            attempt_root / "eval_summary.json",
            context="stair pilot eval summary",
        ),
        expected_checkpoint_hash=checkpoint_hash,
        expected_attempt_id=attempt_id,
    )
    if train["summary_hash"] != seed["train_summary_hash"]:
        raise ValueError("stair pilot resumed train summary hash mismatch")
    if evaluation["summary_hash"] != seed["eval_summary_hash"]:
        raise ValueError("stair pilot resumed eval summary hash mismatch")
    if checkpoint_hash != seed["checkpoint_hash"]:
        raise ValueError("stair pilot resumed checkpoint hash mismatch")

    events = validate_stair_phase_d_event_log(events_path)
    events_by_id = {event["event_id"]: event for event in events}
    for phase, process in seed["processes"].items():
        event = events_by_id.get(process["event_id"])
        if event is None:
            raise ValueError("stair pilot process event is missing")
        expected_event = {
            "attempt_id": attempt_id,
            "budget_steps": config.budget_steps,
            "phase": phase,
            "status": "completed",
            "has_exited": True,
            "exit_code": 0,
            "failure_message": None,
        }
        if any(event.get(key) != value for key, value in expected_event.items()):
            raise ValueError("stair pilot seed process event binding mismatch")
    return seed


def stair_phase_d_pilot_seed_root(*, output_root: Path, config: Any) -> Path:
    """Derive the evidence layout from the protocol-owned iteration count."""

    if not isinstance(output_root, Path):
        raise TypeError("stair Phase D output_root must be a Path")
    max_iterations = getattr(config, "train_max_iterations", None)
    if (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, int)
        or max_iterations <= 0
    ):
        raise ValueError("stair Phase D max_iterations must be positive")
    return output_root / f"pilot-{max_iterations}"


def load_validated_stair_phase_d_run(
    *,
    protocol_path: Path,
    output_root: Path,
    project_root: Path,
) -> dict[str, Any]:
    """Revalidate a completed run from its canonical report and every raw artifact."""

    protocol_path = resolve_existing_file(
        protocol_path,
        field_name="stair Phase D protocol path",
    )
    output_root = resolve_existing_directory(
        output_root,
        field_name="stair Phase D output root",
    )
    project_root = resolve_existing_directory(
        project_root,
        field_name="stair Phase D project root",
    )

    base_config = build_stair_pilot_config(protocol_path)
    official = resolve_official_stair_reference_program(
        base_config,
        project_root=project_root,
    )
    events_path = output_root / "events.jsonl"
    report = validate_stair_phase_d_report(
        base_config,
        official=official,
        report=load_canonical_pilot_json(
            output_root / "phase_d_pilot_report.json",
            context="stair Phase D report",
        ),
    )
    if file_sha256(events_path) != report["event_log_checksum"]:
        raise ValueError("stair Phase D event log changed after report creation")

    reloaded = reload_completed_stair_pilot_seed(
        config=base_config,
        official=official,
        seed_root=stair_phase_d_pilot_seed_root(
            output_root=output_root,
            config=base_config,
        ),
        events_path=events_path,
    )
    if reloaded != report["pilot_evidence"]:
        raise ValueError("stair Phase D report differs from raw pilot artifacts")
    attempt_id = reloaded["attempt_id"]
    expected_event_ids = [
        f"{attempt_id}:train:started",
        f"{attempt_id}:train:completed",
        f"{attempt_id}:eval:started",
        f"{attempt_id}:eval:completed",
    ]
    events = validate_stair_phase_d_event_log(events_path)
    if [event["event_id"] for event in events] != expected_event_ids:
        raise ValueError("stair Phase D event sequence differs from pilot evidence")
    return report


def _canonical_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"stair Phase D {field_name} must be canonical text")
    return value


__all__ = [
    "STAIR_PHASE_D_PROCESS_EVENT_SCHEMA_VERSION",
    "append_stair_phase_d_process_event",
    "build_stair_phase_d_process_event",
    "load_validated_stair_phase_d_run",
    "reload_completed_stair_pilot_seed",
    "stair_phase_d_pilot_seed_root",
    "validate_stair_phase_d_event_log",
    "validate_stair_phase_d_process_event",
]
