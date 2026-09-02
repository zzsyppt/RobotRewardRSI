from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from true_rl_agent.candidates.isolated_worker import (
    IsolatedWorkerStartupError,
    WorkerReady,
    run_ready_worker,
)


def _delayed_worker(
    startup_delay_s: float,
    execution_delay_s: float,
    result_connection,
    start_connection,
) -> None:
    time.sleep(startup_delay_s)
    result_connection.send(WorkerReady())
    if start_connection.recv() != "start":
        return
    time.sleep(execution_delay_s)
    result_connection.send("done")


class IsolatedCandidateWorkerTests(unittest.TestCase):
    def test_startup_time_is_not_charged_to_execution_budget(self) -> None:
        outcome = run_ready_worker(
            target=_delayed_worker,
            args=(0.2, 0.0),
            startup_timeout_s=2.0,
            execution_timeout_s=0.1,
        )

        self.assertTrue(outcome.completed)
        self.assertEqual(outcome.result, "done")

    def test_execution_timeout_starts_after_ready(self) -> None:
        outcome = run_ready_worker(
            target=_delayed_worker,
            args=(0.0, 0.2),
            startup_timeout_s=2.0,
            execution_timeout_s=0.05,
        )

        self.assertFalse(outcome.completed)
        self.assertIsNone(outcome.result)

    def test_startup_timeout_is_an_infrastructure_error(self) -> None:
        with self.assertRaises(IsolatedWorkerStartupError):
            run_ready_worker(
                target=_delayed_worker,
                args=(0.2, 0.0),
                startup_timeout_s=0.05,
                execution_timeout_s=1.0,
            )


if __name__ == "__main__":
    unittest.main()
