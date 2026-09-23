from __future__ import annotations

import copy
import json
import unittest

from tests.fakes.decision_records import valid_accepted_ledger_record
from tests.unit.test_exploration_frontier import _generation, _history, _trial, fake_bootstrap_current_best
from tests.unit.test_push_force_consumers import episode, rollout
from true_rl_agent.agent_core.experiment_memory import build_experiment_memory
from true_rl_agent.agent_core.generation_contract import GENERATION_LEDGER_RECORD_V3_SCHEMA_VERSION
from true_rl_agent.agent_core.ledger import PROVIDER_AUTHORING_MODE
from true_rl_agent.agent_core.exploration import (
    build_exploration_index, build_exploration_prompt_projection, build_revisit_frontier,
)
from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.prompting import assert_public_only_prompt_payload
from true_rl_agent.tasks.push_recovery.config import PushRecoveryTaskConfig
from true_rl_agent.tasks.push_recovery.exploration import PushExplorationProjector
from true_rl_agent.tasks.push_recovery.memory import PushTaskMemoryAdapter
from true_rl_agent.tasks.push_recovery.metrics import project_push_split
from true_rl_agent.tasks.push_recovery.rollout import aggregate_recovery_episodes
from true_rl_agent.tasks.push_recovery.prompt_adapter import (
    PushRecoveryPromptAdapter, _minimum_core_push_prompt_sections,
)


def _dev():
    return project_push_split([rollout()], split_name="public_dev", errors=[])


class PushFeedbackProjectionTests(unittest.TestCase):
    def setUp(self):
        self.config = PushRecoveryTaskConfig.for_cell("push.go2_12dof.recovery_v3")
        self.adapter = PushRecoveryPromptAdapter(self.config)

    def project(self, record):
        return self.adapter.project_prompt_public_history(
            public_records=[record], initial_anchor_public_evidence=None,
        ).records[0]

    def test_trial_memory_uses_measured_parent_dev_and_is_idempotent(self):
        record = valid_accepted_ledger_record(round_id="generation", candidate_id="winner", public_score=.6)
        record["generation"] = {
            "guard_reference": {"score_stats": {"mean": 0.75}, "checkpoint_hash": "private-checkpoint"},
            "parent_current_best": {"public_score": 0.987654321},
            "trials": [{"lane_id": "lane-a", "trial_status": "formal_rejected",
                "candidate": {"candidate_id": "trial-a"}, "train": {},
                "public_dev": {"public_summary": _dev()},
                "public_accept": {"public_summary": {"score_stats": {"mean": 0.987654321}}}}],
        }
        for parent_field in ("guard_reference", "parent_public_dev"):
            with self.subTest(parent_field=parent_field):
                raw = copy.deepcopy(record)
                raw["generation"][parent_field] = raw["generation"].pop("guard_reference")
                projected = self.project(raw)
                self.assertEqual(projected, self.project(projected))
                memory = build_experiment_memory([projected])
                outcome = memory["experiments"][0]["trial_outcomes"][0]
                self.assertEqual(outcome[5:8], [-0.25, None, "not_assessed"])
                self.assertNotIn("0.987654321", json.dumps(projected))
                self.assertNotIn("private-checkpoint", json.dumps(projected))
        record["generation"].pop("guard_reference")
        outcome = self.project(record)["generation"]["trials"][0]["prompt_outcome"]
        self.assertNotIn("objective_delta", outcome)

    def test_no_formal_generation_preserves_sibling_memory_without_false_attribution(self):
        parent = fake_bootstrap_current_best()
        generation = _generation(parent, generation_index=1,
            trial_a=_trial("trial-a", "formal_rejected", seed="1"),
            trial_b=_trial("trial-b", "qualified_not_selected", seed="2"),
            selected_trial_id=None)
        record = {"schema_version": GENERATION_LEDGER_RECORD_V3_SCHEMA_VERSION,
            "record_kind": "generation", "authoring_mode": PROVIDER_AUTHORING_MODE,
            "round_id": "generation-001", "generation": generation,
            "decision": {"accepted": False, "outcome": "generation_head_retained"}}
        for outcome in ("formal_rejected", "no_selection", "candidate_execution_incomplete"):
            with self.subTest(outcome=outcome):
                generation["generation_outcome"] = outcome
                projected = self.project(record)
                self.assertIn("formal", projected["generation"])
                self.assertIsNone(projected["generation"]["formal"])
                self.assertEqual(projected, self.project(projected))
                memory = build_experiment_memory([projected])["experiments"][0]
                self.assertNotIn("candidate_components", memory)
                self.assertEqual([row[1] for row in memory["trial_outcomes"]], ["candidate-a", "candidate-b"])
                self.assertEqual([row[3] for row in memory["trial_outcomes"]], ["RewardTerm:term_1", "RewardTerm:term_2"])
        # A formal record still needs its real attribution; the task must not
        # convert missing evidence into the explicit no-formal-round sentinel.
        generation["formal"] = {}
        with self.assertRaisesRegex(ValueError, "invalid mechanism attribution"):
            build_experiment_memory([self.project(record)])

    def test_task_memory_retains_failure_rates_after_aggregate_projection(self):
        record = {"decision": {"accepted": False}, "public_result": {"metrics": {"public_dev": _dev()}}}
        projected = self.project(record)
        memory = PushTaskMemoryAdapter(config=self.config)
        raw_summary = memory.build_public_summary([record])
        summary = memory.build_public_summary([projected])
        self.assertTrue(raw_summary["latest_failure_breakdown"])
        self.assertEqual(summary["latest_failure_breakdown"], raw_summary["latest_failure_breakdown"])
        self.assertEqual(summary["aggregate_failure_breakdown"], raw_summary["aggregate_failure_breakdown"])
        self.assertEqual(summary["latest_failure_mode"], raw_summary["latest_failure_mode"])
        self.assertEqual(projected, self.project(projected))

    def test_initial_reference_feedback_uses_only_dev_for_both_robots(self):
        dev = _dev()
        for cell in ("push.go2_12dof.recovery_v3", "push.g1_37dof.recovery_v3"):
            adapter = PushRecoveryPromptAdapter(PushRecoveryTaskConfig.for_cell(cell))
            context = {"initial_anchor_public_evidence": {
                "anchor_kind": "reference_program", "public_result_hash": "a" * 64,
                "public_result": {"public_score": 0.987654321, "metrics": {
                    "public_dev": dev,
                    "public_accept": {"canary": "accept-detail-746", "score_stats": {"mean": 0.987654321}},
                    "sealed_validation": {"canary": "sealed-detail-746"},
                    "checkpoint_hash": "private-checkpoint-746",
                }},
            }}
            original = copy.deepcopy(context)
            sections = adapter.build_prompt_sections(public_records=(), public_context=context)
            for value in (sections, _minimum_core_push_prompt_sections(sections)):
                feedback = value["public_feedback_summary"]
                self.assertEqual(feedback["source_selection"], "reference_current_best")
                self.assertEqual(feedback["latest_public_score"], .5)
                self.assertEqual(feedback["public_dev"]["recovery_counts"]["successes"], 1)
                self.assertEqual(feedback["public_dev"]["stratum_results"]["repeated"]["n_over_N"], "1/2")
                self.assertEqual(feedback["public_dev"]["failure_breakdown"]["total"], 2)
                for marker in ("0.987654321", "accept-detail-746", "sealed-detail-746", "private-checkpoint-746"):
                    self.assertNotIn(marker, json.dumps(value))
                assert_public_only_prompt_payload(value)
            self.assertEqual(context, original)

    def test_evaluated_candidate_feedback_takes_precedence_over_reference(self):
        candidate = {"candidate": {"candidate_id": "latest"}, "train_result": {},
            "decision": {"accepted": False}, "public_result": {"metrics": {"public_dev": _dev()}}}
        context = {"initial_anchor_public_evidence": {"anchor_kind": "reference_program",
            "public_result": {"metrics": {"public_dev": {"score_stats": {"mean": .9}}}}}}
        feedback = self.adapter.build_prompt_sections(
            public_records=[candidate], public_context=context,
        )["public_feedback_summary"]
        self.assertEqual(feedback["source_candidate_id"], "latest")
        self.assertEqual(feedback["latest_public_score"], .5)
        self.assertEqual(feedback["current_best"]["source_selection"], "reference_current_best")
        self.assertEqual(feedback["current_best"]["latest_public_score"], .9)

    def test_current_best_dev_keeps_its_own_failure_evidence_and_is_not_duplicated(self):
        raw = rollout()
        rows = [episode(env_id=0), episode(env_id=1)]
        raw.update(per_episode=rows, aggregate=aggregate_recovery_episodes(rows, force_sequence=True))
        best_dev = project_push_split([raw], split_name="public_dev", errors=[])
        best = {"trial_id": "gen-1-b", "trial_status": "accepted", "train": {},
            "candidate": {"candidate_id": "parent", "program_source_checksum": "sha256:parent"},
            "public_dev": {"public_summary": best_dev},
            "public_accept": {"canary": "accept-parent-219"},
            "sealed_validation": {"canary": "sealed-parent-219"}}
        sibling = {"trial_id": "gen-1-a", "trial_status": "qualified_not_selected", "train": {},
            "candidate": {"candidate_id": "sibling", "program_source_checksum": "sha256:sibling"},
            "public_dev": {"public_summary": _dev()}}
        record = valid_accepted_ledger_record(round_id="gen-1", candidate_id="parent", public_score=1.)
        record["generation"] = {"trials": [sibling, best]}
        rejected = {"round_id": "gen-2", "candidate": {"candidate_id": "later-reject"}, "train_result": {},
            "decision": {"accepted": False}, "public_result": {"metrics": {"public_dev": _dev()}}}
        for cell in ("push.go2_12dof.recovery_v3", "push.g1_37dof.recovery_v3"):
            adapter = PushRecoveryPromptAdapter(PushRecoveryTaskConfig.for_cell(cell))
            for records, selected in (([record], "sibling"), ([record, rejected], "later-reject")):
                with self.subTest(cell=cell, selected=selected):
                    sections = adapter.build_prompt_sections(public_records=records, public_context={
                        "task_public_summary": {"latest_failure_breakdown": {"height_collapse": 999}}})
                    for value in (sections, _minimum_core_push_prompt_sections(sections)):
                        feedback = value["public_feedback_summary"]
                        self.assertEqual(feedback["source_candidate_id"], selected)
                        self.assertEqual(feedback["latest_public_score"], .5)
                        parent = feedback["current_best"]
                        self.assertEqual(parent["source_selection"], "accepted_current_best")
                        self.assertEqual(parent["source_round_id"], "gen-1-b")
                        self.assertEqual(parent["source_candidate_id"], "parent")
                        self.assertEqual(parent["program_source_checksum"], "sha256:parent")
                        self.assertEqual(parent["latest_public_score"], 1.)
                        self.assertEqual(parent["public_dev"]["recovery_counts"]["successes"], 2)
                        self.assertEqual(parent["public_dev"]["failure_breakdown"]["counts"], {"success_recovered": 2})
                        self.assertEqual(feedback["public_dev"]["failure_breakdown"]["counts"]["fall_side"], 1)
                        for marker in ("accept-parent-219", "sealed-parent-219", "999"):
                            self.assertNotIn(marker, json.dumps(value))
                        assert_public_only_prompt_payload(value)
            accepted_only = copy.deepcopy(record)
            accepted_only["generation"]["trials"] = [best]
            feedback = adapter.build_prompt_sections(public_records=[accepted_only], public_context={})["public_feedback_summary"]
            self.assertEqual(feedback["source_selection"], "accepted_current_best")
            self.assertEqual(feedback["source_candidate_id"], "parent")
            self.assertNotIn("current_best", feedback)

    def test_failure_mode_reports_failure_even_when_successes_are_the_majority(self):
        raw = rollout()
        rows = [episode(env_id=0), episode(env_id=1), episode(success=False, env_id=2)]
        raw.update(per_episode=rows, aggregate=aggregate_recovery_episodes(rows, force_sequence=True))
        dev = project_push_split([raw], split_name="public_dev", errors=[])
        record = {"candidate": {"candidate_id": "majority-success"}, "train_result": {},
            "decision": {"accepted": False}, "public_result": {"metrics": {"public_dev": dev}}}
        anchor = {"initial_anchor_public_evidence": {"anchor_kind": "reference_program",
            "public_result": {"metrics": {"public_dev": dev}}}}
        for records, context in (([record], {}), ([], anchor)):
            feedback = self.adapter.build_prompt_sections(
                public_records=records, public_context=context,
            )["public_feedback_summary"]
            self.assertEqual(feedback["public_dev"]["dominant_outcome"], "success_recovered")
            self.assertEqual(feedback["latest_failure_mode"], "fall_dominated")
        memory = PushTaskMemoryAdapter().build_public_summary([self.project(record)])
        self.assertEqual(memory["latest_failure_mode"], "fall_dominated")
        self.assertEqual(memory["latest_failure_breakdown"], {"fall_side": .333333})

    def test_exploration_cards_and_memory_never_include_acceptance_scores(self):
        parent = fake_bootstrap_current_best()
        a = _trial("trial-source", "formal_rejected", seed="1")
        b = _trial("trial-b", "qualified_not_selected", seed="2")
        for trial in (a, b):
            trial["public_dev"]["public_summary"] = _dev()
            trial["public_accept"]["public_summary"] = {"score_stats": {"mean": .987654321, "count": 3}}
        generation = _generation(parent, generation_index=1, trial_a=a, trial_b=b, selected_trial_id=None)
        generation["guard_reference"] = {"score_stats": {"mean": .75}}
        projector = PushExplorationProjector(self.config)
        index = build_exploration_index(_history(parent, [generation]), task_projector=projector,
            expected_projector_contract_hash=canonical_json_hash(projector.contract()))
        projection = build_exploration_prompt_projection(index, build_revisit_frontier(index))
        memory = build_experiment_memory([], exploration_projection=projection)
        self.assertEqual(len(index.cards), 2)
        self.assertTrue(index.cards[0].public_evidence_summary["public_dev"])
        self.assertEqual(projection["priority_or_eligible_cards"], [])
        for value in (index.cards[0].public_evidence_summary, projection, memory):
            self.assertNotIn("public_accept", json.dumps(value))
            self.assertNotIn("0.987654321", json.dumps(value))
            assert_public_only_prompt_payload(value)
        self.assertEqual(index.cards[0].public_evidence_summary["objective"]["delta"], -.25)
        self.assertEqual(index.cards[0].observed_effect, "unknown")


if __name__ == "__main__":
    unittest.main()
