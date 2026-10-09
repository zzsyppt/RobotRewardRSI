"""Retention floor from frozen formal c6 and Push public memory projections.

The fixture contains only compact public memory and scalar runtime measurements.
Its source hashes document extraction; tests never read the historical PLAN data.
This exercises the production row encoder and compactor, not receipt replay.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from true_rl_agent.agent_core import experiment_memory as memory  # noqa: E402
from true_rl_agent.prompt_serialization import canonical_prompt_json_bytes  # noqa: E402


class FeedbackMemoryRetentionTests(unittest.TestCase):
    def test_frozen_real_sessions_keep_baseline_history_with_runtime_states(self):
        fixture = json.loads((ROOT / "tests/fixtures/feedback_memory_baseline.json").read_text(encoding="utf-8"))
        self.assertEqual(memory.EXPERIMENT_MEMORY_MAX_SERIALIZED_BYTES, fixture["byte_budget"])
        for case in fixture["cases"]:
            with self.subTest(case=case["id"]):
                baseline = case["memory"]
                self.assertEqual(len(baseline["experiments"]), case["baseline_retained"])
                enriched = copy.deepcopy(baseline)
                session = fixture["sessions"][case["task"]]
                for entry in enriched["experiments"]:
                    generation = session["generations"][str(entry["round_index"])]
                    trials = []
                    for row in entry["trial_outcomes"]:
                        lane = generation["lanes"][row[0]]
                        self.assertEqual(lane["candidate_id"], row[1])
                        trials.append({
                            "lane_id": "lane-" + row[0],
                            "candidate": {"candidate_id": row[1]},
                            "mechanism_operation_delta": {
                                "operation": row[2], "after_component": {"mechanism_id": row[3]},
                            },
                            "trial_status": row[4],
                            "prompt_outcome": {
                                "objective_delta": row[5], "guard_delta": row[6], "runtime_effect": row[7],
                            },
                            "status_reason": row[8],
                            "runtime_activation_summary": fixture["runtime_profiles"][lane["profile"]],
                        })
                    selected = generation["selected_lane"]
                    selected_runtime = (
                        fixture["runtime_profiles"][generation["lanes"][selected]["profile"]]
                        if selected is not None else None
                    )
                    encoded = memory._record_entry({
                        "runtime_activation_summary": selected_runtime,
                        "generation": {"trials": trials},
                    }, round_index=entry["round_index"])
                    # Preserve every original candidate, outcome, delta and reason;
                    # production alone supplies the newly added measurement column.
                    self.assertEqual([r[:9] for r in encoded["trial_outcomes"]], entry["trial_outcomes"])
                    entry["trial_outcomes"] = encoded["trial_outcomes"]
                    if "runtime_observation" in encoded:
                        entry["runtime_observation"] = encoded["runtime_observation"]

                actual = memory.recompact_experiment_memory(enriched)
                retained = actual["experiments"]
                self.assertGreaterEqual(len(retained), case["minimum_retained"])
                self.assertLessEqual(canonical_prompt_json_bytes(actual), fixture["byte_budget"])
                self.assertEqual(actual["omitted_experiment_count"], baseline["omitted_experiment_count"]
                                 + len(baseline["experiments"]) - len(retained))
                shown = {e["round_index"] for e in retained}
                self.assertIn(baseline["experiments"][-1]["round_index"], shown)
                last_accepted = next((e for e in reversed(baseline["experiments"]) if e.get("accepted")), None)
                if last_accepted is not None:
                    self.assertIn(last_accepted["round_index"], shown)
                for entry in retained:
                    self.assertNotIn("runtime_observation", entry)
                    self.assertTrue(all(len(row) == 10 and isinstance(row[9], str)
                                        for row in entry["trial_outcomes"]))
                self.assertEqual(memory.recompact_experiment_memory(actual), actual)


if __name__ == "__main__":
    unittest.main()
