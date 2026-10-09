# ruff: noqa: E402
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from true_rl_agent.json_contract import canonical_json_bytes
from true_rl_agent.tasks.stair_climbing.generation_authoring_timing import (
    replay_stair_generation_authoring_timing,
    run_stair_generation_authoring_attempt,
)

GENERATION_ID = "p6c-session-generation-001"


class _Clock:
    def __init__(
        self,
        monotonic_values: list[int],
        utc_values: list[str],
    ) -> None:
        self._monotonic_values = iter(monotonic_values)
        self._utc_values = iter(utc_values)

    def monotonic_ns(self) -> int:
        return next(self._monotonic_values)

    def utc_now(self) -> str:
        return next(self._utc_values)


def _clock(start: int, stop: int, *, minute: int) -> _Clock:
    return _Clock(
        [start, stop],
        [
            f"2026-08-22T00:{minute:02d}:00Z",
            f"2026-08-22T00:{minute:02d}:01Z",
        ],
    )


class StairGenerationAuthoringTimingTests(unittest.TestCase):
    def test_wrapper_preserves_results_and_replay_sums_lane_wall_time(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            evidence_root = Path(tmp).resolve() / "timing"
            lane_a_result = object()
            lane_b_result = object()
            lane_a_clock = _clock(100, 160, minute=1)
            lane_b_clock = _clock(1_000, 1_090, minute=2)

            self.assertIs(
                run_stair_generation_authoring_attempt(
                    evidence_root=evidence_root,
                    generation_id=GENERATION_ID,
                    lane_id="lane-a",
                    author=lambda: lane_a_result,
                    monotonic_ns=lane_a_clock.monotonic_ns,
                    utc_now=lane_a_clock.utc_now,
                ),
                lane_a_result,
            )
            self.assertIs(
                run_stair_generation_authoring_attempt(
                    evidence_root=evidence_root,
                    generation_id=GENERATION_ID,
                    lane_id="lane-b",
                    author=lambda: lane_b_result,
                    monotonic_ns=lane_b_clock.monotonic_ns,
                    utc_now=lane_b_clock.utc_now,
                ),
                lane_b_result,
            )

            generation_root = evidence_root / GENERATION_ID
            os.utime(
                generation_root / "lane-a" / "attempt-000000" / "stop.json",
                ns=(9_000_000_000, 9_000_000_000),
            )
            os.utime(
                generation_root / "lane-b" / "attempt-000000" / "stop.json",
                ns=(1_000_000_000, 1_000_000_000),
            )
            summary = replay_stair_generation_authoring_timing(
                evidence_root=evidence_root,
                generation_id=GENERATION_ID,
            )

            self.assertEqual(
                summary["lane_measured_closed_attempt_wall_duration_ns"],
                {"lane-a": 60, "lane-b": 90},
            )
            self.assertEqual(summary["measured_closed_attempt_wall_duration_ns"], 150)
            self.assertEqual(summary["total_wall_duration_ns"], 150)
            self.assertTrue(summary["timing_complete"])
            self.assertEqual(len(summary["stopped_attempts"]), 2)
            self.assertEqual(summary["incomplete_attempts"], [])
            self.assertTrue(summary["authority"]["observer_only"])
            self.assertFalse(summary["authority"]["authoring_control"])

    def test_raised_callable_is_timed_and_the_same_exception_is_reraised(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            evidence_root = Path(tmp).resolve() / "timing"
            clock = _clock(500, 575, minute=3)
            failure = RuntimeError("provider unavailable")

            def fail():
                raise failure

            with self.assertRaises(RuntimeError) as raised:
                run_stair_generation_authoring_attempt(
                    evidence_root=evidence_root,
                    generation_id=GENERATION_ID,
                    lane_id="lane-a",
                    author=fail,
                    monotonic_ns=clock.monotonic_ns,
                    utc_now=clock.utc_now,
                )

            self.assertIs(raised.exception, failure)
            summary = replay_stair_generation_authoring_timing(
                evidence_root=evidence_root,
                generation_id=GENERATION_ID,
            )
            stopped = summary["stopped_attempts"]
            self.assertEqual(len(stopped), 1)
            self.assertEqual(stopped[0]["outcome"], "raised")
            self.assertEqual(stopped[0]["exception_type"], "RuntimeError")
            self.assertEqual(stopped[0]["duration_ns"], 75)
            self.assertEqual(summary["incomplete_attempts"], [])

    def test_hard_crash_leaves_start_and_resume_uses_a_new_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            evidence_root = Path(tmp).resolve() / "timing"
            environment = dict(os.environ)
            environment["PYTHONPATH"] = os.pathsep.join(
                filter(
                    None,
                    (
                        str(SRC_ROOT),
                        environment.get("PYTHONPATH"),
                    ),
                )
            )
            source = f"""
import os
from pathlib import Path
from true_rl_agent.tasks.stair_climbing.generation_authoring_timing import run_stair_generation_authoring_attempt
run_stair_generation_authoring_attempt(
    evidence_root=Path({str(evidence_root)!r}),
    generation_id={GENERATION_ID!r},
    lane_id='lane-a',
    author=lambda: os._exit(23),
)
"""
            crashed = subprocess.run(
                [sys.executable, "-c", source],
                cwd=PROJECT_ROOT,
                env=environment,
                check=False,
                capture_output=True,
                timeout=10,
            )
            self.assertEqual(crashed.returncode, 23, crashed.stderr.decode())

            lane_root = evidence_root / GENERATION_ID / "lane-a"
            first_start = lane_root / "attempt-000000" / "start.json"
            first_bytes = first_start.read_bytes()
            self.assertFalse((first_start.parent / "stop.json").exists())
            resume_clock = _clock(2_000, 2_125, minute=4)
            self.assertEqual(
                run_stair_generation_authoring_attempt(
                    evidence_root=evidence_root,
                    generation_id=GENERATION_ID,
                    lane_id="lane-a",
                    author=lambda: "resumed",
                    monotonic_ns=resume_clock.monotonic_ns,
                    utc_now=resume_clock.utc_now,
                ),
                "resumed",
            )

            self.assertEqual(first_start.read_bytes(), first_bytes)
            self.assertTrue((lane_root / "attempt-000001" / "start.json").is_file())
            self.assertTrue((lane_root / "attempt-000001" / "stop.json").is_file())
            summary = replay_stair_generation_authoring_timing(
                evidence_root=evidence_root,
                generation_id=GENERATION_ID,
            )
            self.assertEqual(
                summary["lane_measured_closed_attempt_wall_duration_ns"]["lane-a"],
                125,
            )
            self.assertEqual(summary["measured_closed_attempt_wall_duration_ns"], 125)
            self.assertIsNone(summary["total_wall_duration_ns"])
            self.assertFalse(summary["timing_complete"])
            self.assertEqual(len(summary["incomplete_attempts"]), 1)
            self.assertEqual(
                summary["incomplete_attempts"][0]["attempt_id"],
                "attempt-000000",
            )

    def test_strict_replay_rejects_a_tampered_duration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            evidence_root = Path(tmp).resolve() / "timing"
            clock = _clock(100, 150, minute=5)
            run_stair_generation_authoring_attempt(
                evidence_root=evidence_root,
                generation_id=GENERATION_ID,
                lane_id="lane-a",
                author=lambda: None,
                monotonic_ns=clock.monotonic_ns,
                utc_now=clock.utc_now,
            )
            stop_path = (
                evidence_root
                / GENERATION_ID
                / "lane-a"
                / "attempt-000000"
                / "stop.json"
            )
            stop = json.loads(stop_path.read_text(encoding="utf-8"))
            stop["duration_ns"] += 1
            stop_path.write_bytes(canonical_json_bytes(stop, trailing_newline=True))

            with self.assertRaisesRegex(ValueError, "duration mismatch"):
                replay_stair_generation_authoring_timing(
                    evidence_root=evidence_root,
                    generation_id=GENERATION_ID,
                )


if __name__ == "__main__":
    unittest.main()
