from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from true_rl_agent.evaluation.subject import ExperimentSubject  # noqa: E402
from true_rl_agent.runner.job_identity import (  # noqa: E402
    JobIdentity,
    validate_job_summary_identity,
)


class JobIdentityTests(unittest.TestCase):
    def test_identity_round_trips_as_the_canonical_summary_binding(self) -> None:
        identity = JobIdentity(
            "job-1",
            "task-1",
            ExperimentSubject.candidate("candidate-1"),
        )

        validated = validate_job_summary_identity(
            identity.to_public_dict(),
            expected=identity,
            summary_name="train summary",
        )

        self.assertIs(validated, identity)

    def test_identity_rejects_empty_or_non_string_fields(self) -> None:
        for field, value in (
            ("job_id", " "),
            ("job_id", " job-1 "),
            ("task_id", None),
            ("subject", True),
        ):
            with self.subTest(field=field):
                values = {
                    "job_id": "job-1",
                    "task_id": "task-1",
                    "subject": ExperimentSubject.candidate("candidate-1"),
                }
                values[field] = value
                with self.assertRaisesRegex((TypeError, ValueError), field):
                    JobIdentity(**values)

    def test_summary_binding_does_not_coerce_identity_fields(self) -> None:
        identity = JobIdentity(
            "1",
            "task-1",
            ExperimentSubject.candidate("candidate-1"),
        )

        with self.assertRaisesRegex(ValueError, "binding mismatch for job_id"):
            validate_job_summary_identity(
                {
                    "job_id": 1,
                    "task_id": "task-1",
                    "subject_kind": "candidate",
                    "subject_id": "candidate-1",
                },
                expected=identity,
                summary_name="train summary",
            )


if __name__ == "__main__":
    unittest.main()
