"""Version-qualified current-only input; no inferred history or lost endpoint."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

import numpy as np

from tests.unit import stair_archive_fixture as fixture
from true_rl_agent.runner.numeric_evidence import file_sha256
from true_rl_agent.evaluation.process_report import report_bytes
from true_rl_agent.runner.numeric_evidence import NumericEvidenceError
from true_rl_agent.tasks.stair_climbing.trajectory_adapter import iter_stair_episodes


def as_v1(identity, context, chunks):
    identity["evaluation_plan_hash"] = "e" * 64
    identity["evaluation_binding"]["evaluation_binding_hash"] = "f" * 64
    context["profile"] = "stair.unified_first_episode.v1"
    del context["contact_force_history"]
    for metadata, arrays in chunks:
        del metadata["transition_header"]["contact_force_history"]
        del arrays["support_contact_forces_w_history"]
        del arrays["forbidden_contact_forces_w_history"]


class StairTrajectoryV1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.StairArchiveFixture.setUpClass()

    def setUp(self):
        self.fixture = fixture.StairArchiveFixture()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)


    def test_v1_rejects_history_metadata_arrays_header_or_late_drift(self):
        for field in ("metadata", "array", "header", "late_array", "clock", "unknown_profile"):
            def mutate(identity, context, chunks):
                old = deepcopy(context["contact_force_history"])
                h = chunks[0][1]["support_contact_forces_w_history"].copy()
                as_v1(identity, context, chunks)
                if field == "metadata":
                    context["contact_force_history"] = old
                elif field in ("array", "late_array"):
                    i = int(field == "late_array")
                    chunks[i][1]["support_contact_forces_w_history"] = h[:chunks[i][0]["steps"]]
                elif field == "header":
                    chunks[1][0]["transition_header"]["contact_force_history"] = old
                elif field == "clock":
                    context["seeds"][0]["step_dt"] = .04
                else:
                    context["profile"] = "stair.unified_first_episode.v99"
            with self.subTest(field=field), self.assertRaises(NumericEvidenceError):
                next(iter_stair_episodes(self.fixture._archive(mutate), source_id="v1"))

    def test_v2_missing_history_is_rejected_instead_of_downgraded(self):
        def mutate(identity, context, chunks):
            del chunks[1][1]["support_contact_forces_w_history"]
        with self.assertRaises(NumericEvidenceError):
            next(iter_stair_episodes(self.fixture._archive(mutate), source_id="v2"))





if __name__ == "__main__":
    unittest.main()
