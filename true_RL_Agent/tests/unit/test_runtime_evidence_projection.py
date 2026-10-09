from __future__ import annotations

import copy
from types import SimpleNamespace
import unittest

from true_rl_agent.agent_core.context import _candidate_runtime_activated_in_training
from true_rl_agent.agent_core.experiment_memory import build_experiment_memory, recompact_experiment_memory
from true_rl_agent.candidates.components import RewardTerm, TrainingComponents
from true_rl_agent.candidates.mechanism import RuntimeActivationEvidence
from true_rl_agent.memory.context_builder import PublicContextBuilder
from true_rl_agent.tasks.push_recovery.prompt_adapter import (
    _minimum_core_push_prompt_sections,
    _runtime_feedback_summary,
)
from true_rl_agent.tasks.push_recovery.runtime_adapter import CandidateRuntimeAdapter
from true_rl_agent.tasks.stair_climbing.prompt_adapter import _latest_runtime_feedback


def record(runtime):
    return {
        "round_id": "measured-round",
        "candidate": {"candidate_id": "measured-candidate", "changed_components": []},
        "decision": {"accepted": False, "outcome": "reject", "reason": "eval_incomplete"},
        "train_result": {"summary": {"candidate_package_loaded": True}},
        "runtime_activation_summary": runtime,
    }


class RuntimeEvidenceProjectionTests(unittest.TestCase):
    def test_missing_partial_and_measured_zero_survive_author_and_memory(self):
        cases = (
            None,
            {},
            {"reward_term_calls": 4},
            {"step_calls": 1, "reward_term_count": 0, "reward_term_calls": 0},
            {"step_calls": 0, "reward_term_count": 1, "reward_term_calls": 0},
            {"step_calls": 1, "reward_term_count": 1, "reward_term_calls": 1,
             "nonzero_reward_contribution": False},
        )
        snapshots = []
        for raw in cases:
            with self.subTest(raw=raw):
                evidence = RuntimeActivationEvidence.from_mapping(raw)
                public = evidence.to_public_dict()
                expected = evidence.measurement_summary()
                self.assertEqual(RuntimeActivationEvidence.from_mapping(public).to_public_dict(), public)
                built = PublicContextBuilder().build_prompt_payload([record(public)])
                history = built["public_history"]
                actual = history["recent_outcomes"][-1]["runtime_activation"]
                for key, value in expected.items():
                    self.assertEqual(actual.get(key), value)
                memory = history["experiment_memory"]
                for _ in range(3):
                    memory = recompact_experiment_memory(memory)
                    self.assertEqual(memory["experiments"][-1]["runtime_observation"], evidence.reward_execution_state())
                for project in (_runtime_feedback_summary, _latest_runtime_feedback):
                    feedback = project([record(public)])
                    self.assertEqual(feedback["status"], expected["status"])
                    if expected["status"] == "available":
                        for key, value in expected.items():
                            self.assertEqual(feedback["latest_runtime_activation"].get(key), value)
                snapshots.append(expected)
        self.assertEqual(snapshots[0], {"status": "not_available"})
        self.assertIsNone(snapshots[2]["diagnostic_calls"])
        self.assertIsNone(snapshots[2]["nonzero_reward_contribution"])
        self.assertNotEqual(snapshots[3], snapshots[4])
        self.assertNotEqual(snapshots[4], snapshots[5])

    def test_runtime_producer_distinguishes_no_terms_no_execution_and_zero_value(self):
        empty = CandidateRuntimeAdapter(manifest=None, components=TrainingComponents(name="empty"),
                                        num_envs=1, planned_training_steps=2)
        zero = CandidateRuntimeAdapter(manifest=None,
            components=TrainingComponents(name="zero", reward_terms=[RewardTerm(name="zero", weight=1.0, fn=lambda inputs: [0.0])]),
            num_envs=1, planned_training_steps=2)
        before = RuntimeActivationEvidence.from_mapping(zero.runtime_summary()).measurement_summary()
        for adapter in (empty, zero):
            adapter.apply_step_rewards(env=SimpleNamespace(), actions=[[0.0]], rewards=[0.0])
        no_terms = RuntimeActivationEvidence.from_mapping(empty.runtime_summary()).measurement_summary()
        measured_zero = RuntimeActivationEvidence.from_mapping(zero.runtime_summary()).measurement_summary()
        self.assertEqual((before["step_calls"], before["reward_term_count"]), (0, 1))
        self.assertEqual((no_terms["step_calls"], no_terms["reward_term_count"], no_terms["reward_term_calls"]), (1, 0, 0))
        self.assertEqual((measured_zero["step_calls"], measured_zero["reward_term_calls"]), (1, 1))
        self.assertIs(measured_zero["nonzero_reward_contribution"], False)

    def test_package_loaded_is_not_execution_and_capacity_omission_is_not_missing(self):
        self.assertEqual(_candidate_runtime_activated_in_training(record(None)), "unknown")
        self.assertIs(_candidate_runtime_activated_in_training(record({"step_calls": 0})), False)
        self.assertIs(_candidate_runtime_activated_in_training(record({"step_calls": 1})), True)
        sections = {
            "task": {"task_id": "runtime_probe"}, "reward_inputs_contract": {},
            "public_eval_contract": {}, "public_feedback_summary": {"status": "available"},
            "runtime_feedback_summary": {"status": "not_available"}, "selection_guidance": [],
        }
        original = copy.deepcopy(sections)
        minimum = _minimum_core_push_prompt_sections(sections)
        self.assertEqual(minimum["runtime_feedback_summary"]["status"], "omitted_for_capacity")
        self.assertEqual(sections, original)

    def test_generation_memory_preserves_each_sibling_under_capacity_pressure(self):
        measured = {"step_calls": 1, "reward_term_count": 1, "reward_term_calls": 1,
                    "nonzero_reward_contribution": False}
        expected = RuntimeActivationEvidence.from_mapping(measured).reward_execution_state()
        records = []
        for generation in range(20):
            item = record(None)
            item["generation"] = {"trials": [
                {"lane_id": "lane-a", "candidate": {"candidate_id": f"missing-{generation}"},
                 "trial_status": "train_failed"},
                {"lane_id": "lane-b", "candidate": {"candidate_id": f"zero-{generation}"},
                 "trial_status": "screened_only", "runtime_activation_summary": measured},
            ]}
            records.append(item)
        memory = build_experiment_memory(records)
        self.assertEqual(memory["trial_outcome_columns"][-1], "runtime_observation")
        self.assertGreater(memory["omitted_experiment_count"], 0)
        for _ in range(3):
            memory = recompact_experiment_memory(memory)
            for entry in memory["experiments"]:
                missing, zero = entry["trial_outcomes"]
                generation = entry["round_index"] - 1
                self.assertEqual(missing[:2], ["a", f"missing-{generation}"])
                self.assertEqual(zero[:2], ["b", f"zero-{generation}"])
                self.assertNotIn("runtime_observation", entry)
                self.assertEqual(missing[-1], "not_available")
                self.assertEqual(zero[-1], expected)

    def test_memory_states_do_not_invent_execution_from_partial_records(self):
        cases = (
            (None, "not_available"),
            ({"step_calls": 0, "reward_term_count": 1}, "not_executed"),
            ({"step_calls": 1, "reward_term_count": 0}, "no_reward_terms"),
            ({"step_calls": 1, "reward_term_count": 1, "reward_term_calls": 0}, "terms_not_called"),
            ({"step_calls": 1, "reward_term_calls": 0}, "unknown"),
            ({"reward_term_count": 1, "reward_term_calls": 0}, "unknown"),
            ({"reward_term_calls": 0}, "unknown"),
            ({"reward_term_calls": 4, "nonzero_reward_contribution": False}, "executed_zero"),
            ({"reward_term_calls": 4, "nonzero_reward_contribution": True}, "executed_nonzero"),
            ({"reward_term_calls": 4}, "executed_value_unknown"),
            ({"nonzero_reward_contribution": False}, "unknown"),
            ({"diagnostic_calls": 4}, "unknown"),
            ({"capability_activation_records": ["loaded"]}, "unknown"),
        )
        for raw, state in cases:
            with self.subTest(raw=raw):
                evidence = RuntimeActivationEvidence.from_mapping(raw)
                self.assertEqual(evidence.reward_execution_state(), state)
                item = record(raw)
                item["generation"] = {"trials": [
                    {"lane_id": "lane-a", "candidate": {"candidate_id": "a"},
                     "runtime_activation_summary": raw}
                ]}
                memory = build_experiment_memory([item])
                for _ in range(3):
                    memory = recompact_experiment_memory(memory)
                    self.assertEqual(memory["experiments"][0]["trial_outcomes"][0][-1], state)
                    self.assertNotIn("runtime_observation", memory["experiments"][0])

    def test_old_width_one_zero_filled_record_keeps_reward_execution_unknown(self):
        # Before P1, RuntimeActivationEvidence.to_public_dict emitted these
        # defaults even when no runtime evidence had been supplied.
        legacy_runtime = {
            "reward_term_calls": 0, "diagnostic_calls": 0,
            "nonzero_reward_contribution": False,
            "reward_output_stats": {}, "diagnostic_outputs": {},
            "component_activation_records": [], "feature_activation_records": [],
            "capability_activation_records": [],
        }
        built = PublicContextBuilder().build_prompt_payload([record(legacy_runtime)])
        history = built["public_history"]
        observed = history["recent_outcomes"][0]["runtime_activation"]
        self.assertEqual(observed["reward_term_calls"], 0)
        self.assertIsNone(observed["step_calls"])
        self.assertIsNone(observed["reward_term_count"])
        self.assertEqual(history["experiment_memory"]["experiments"][0]["runtime_observation"], "unknown")

    def test_recompact_preserves_historical_nine_columns_and_current_short_state(self):
        old_row = ["a", "candidate-a", "add", "RewardTerm:reward-a",
                   "qualified_not_selected", 0.1, 0.0, "pass", None]
        current_row = ["b", "candidate-b", "add", "RewardTerm:reward-b",
                       "qualified_not_selected", 0.0, 0.0, "pass", None, "executed_zero"]
        compact = recompact_experiment_memory({"experiments": [
            {"round_index": 1, "trial_outcomes": [old_row, current_row]},
            {"round_index": 2, "runtime_observation": "unknown"},
        ]})
        self.assertEqual(compact["experiments"][0]["trial_outcomes"],
                         [old_row + ["not_available"], current_row])
        self.assertEqual(compact["experiments"][1]["runtime_observation"], "unknown")
        self.assertEqual(recompact_experiment_memory(compact), compact)


if __name__ == "__main__":
    unittest.main()
