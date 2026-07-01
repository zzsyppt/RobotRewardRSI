#!/usr/bin/env python3
"""Self-test for the Phase 8E capped self-evolution loop.

The full round is faked; these tests verify the loop sequencing, the shared
lab_dir (cross-round inheritance/feedback), the round cap, and aggregate hygiene
without launching training.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from humanoid_agent.cold_start_lab.phase8e_evolution_loop import (  # noqa: E402
    Phase8EEvolutionLoopError,
    run_phase8e_evolution_loop,
)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_loop_runs_capped_rounds_with_shared_lab_dir() -> None:
    with tempfile.TemporaryDirectory(prefix="phase8e-evolution-cap-") as tmp:
        root = Path(tmp)
        calls: list[dict[str, Any]] = []

        def full_round_runner(**kwargs: Any) -> tuple[int, dict[str, Any]]:
            calls.append(kwargs)
            return 0, {"status": "reject", "formal_training_executed": True, "summary_file": str(Path(kwargs["run_dir"]) / "s.json")}

        exit_code, aggregate = run_phase8e_evolution_loop(
            config_path=PROJECT_ROOT / "configs" / "cold_start" / "phase8e_loop.yaml",
            run_root=root / "evo",
            max_rounds=3,
            full_round_runner=full_round_runner,
        )
        _assert(exit_code == 0, "loop should exit 0 on normal completion")
        _assert(len(calls) == 3, "loop must run exactly max_rounds full rounds")
        _assert(aggregate["rounds_executed"] == 3, "aggregate must record three rounds")
        _assert(aggregate["accepted_rounds"] == 0, "no rounds accepted in this fake")
        _assert(aggregate["stop_reason"] == "max_rounds_reached", "stop reason should be the cap")

        lab_dirs = {str(call["lab_dir"]) for call in calls}
        _assert(len(lab_dirs) == 1, "every round must share one lab_dir for inheritance")
        run_dirs = [str(call["run_dir"]) for call in calls]
        _assert(len(set(run_dirs)) == 3, "each round must use a distinct run_dir")
        for call in calls:
            _assert(call["append_ledger"] is True, "loop must append the ledger each round")
            _assert(call["inherit_if_accepted"] is True, "loop must enable accepted-recipe inheritance")


def test_loop_tracks_best_accept_and_optional_no_improve_stop() -> None:
    with tempfile.TemporaryDirectory(prefix="phase8e-evolution-improve-") as tmp:
        root = Path(tmp)
        # Round 1 accepts at 0.20, round 2 accepts at 0.10 (no improvement),
        # round 3 would also be no-improvement -> stop_after_no_improve=2 stops at round 3.
        scripted = [("accept", 0.20), ("accept", 0.10), ("accept", 0.10), ("accept", 0.30)]
        seen: list[int] = []

        def full_round_runner(**kwargs: Any) -> tuple[int, dict[str, Any]]:
            index = len(seen)
            seen.append(index)
            status, mean = scripted[index]
            run_dir = Path(kwargs["run_dir"])
            linked_file = run_dir / "linked_round" / "phase8e_linked_round_summary.json"
            linked_file.parent.mkdir(parents=True, exist_ok=True)
            linked_file.write_text(
                json.dumps({"round_summary": {"decision": {"detail": {"candidate_public_accept_mean": mean}}}}),
                encoding="utf-8",
            )
            return 0, {
                "status": status,
                "formal_training_executed": True,
                "summary_file": str(run_dir / "s.json"),
                "stages": {"linked_round": {"summary_file": str(linked_file)}},
            }

        exit_code, aggregate = run_phase8e_evolution_loop(
            config_path=PROJECT_ROOT / "configs" / "cold_start" / "phase8e_loop.yaml",
            run_root=root / "evo",
            max_rounds=4,
            stop_after_no_improve=2,
            full_round_runner=full_round_runner,
        )
        _assert(exit_code == 0, "loop should exit 0")
        _assert(aggregate["best_public_accept_mean"] == 0.20, "best should be the round-1 mean")
        _assert(aggregate["accepted_rounds"] == 3, "three rounds accepted before stop")
        _assert(aggregate["rounds_executed"] == 3, "no-improve stop should trigger at round 3, before round 4")
        _assert(aggregate["stop_reason"] == "no_improve_for_2_rounds", "early stop reason recorded")


def test_aggregate_timestamps_cover_loop_runtime() -> None:
    with tempfile.TemporaryDirectory(prefix="phase8e-evolution-time-") as tmp:
        root = Path(tmp)

        def full_round_runner(**kwargs: Any) -> tuple[int, dict[str, Any]]:
            time.sleep(0.02)
            return 0, {"status": "reject", "formal_training_executed": False, "summary_file": str(Path(kwargs["run_dir"]) / "s.json")}

        _exit_code, aggregate = run_phase8e_evolution_loop(
            config_path=PROJECT_ROOT / "configs" / "cold_start" / "phase8e_loop.yaml",
            run_root=root / "evo",
            max_rounds=1,
            full_round_runner=full_round_runner,
        )
        created = datetime.fromisoformat(aggregate["created_at_utc"])
        completed = datetime.fromisoformat(aggregate["completed_at_utc"])
        _assert(completed >= created, "completed_at_utc must not precede created_at_utc")
        _assert((completed - created).total_seconds() >= 0.01, "created_at_utc must be captured before round execution")


def test_invalid_max_rounds_rejected() -> None:
    try:
        run_phase8e_evolution_loop(
            config_path=PROJECT_ROOT / "configs" / "cold_start" / "phase8e_loop.yaml",
            run_root=Path(tempfile.gettempdir()) / "phase8e_evo_invalid",
            max_rounds=0,
            full_round_runner=lambda **_kw: (0, {"status": "reject"}),
        )
    except Phase8EEvolutionLoopError:
        return
    raise AssertionError("max_rounds=0 should be rejected")


def main() -> int:
    test_loop_runs_capped_rounds_with_shared_lab_dir()
    test_loop_tracks_best_accept_and_optional_no_improve_stop()
    test_aggregate_timestamps_cover_loop_runtime()
    test_invalid_max_rounds_rejected()
    print(json.dumps({"status": "pass", "test": "phase8e_evolution_loop_selftest"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
