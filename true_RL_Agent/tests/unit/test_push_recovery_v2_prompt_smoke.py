from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest

import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tests.fakes.decision_records import (
    TEST_ACCEPTANCE_PROTOCOL, TEST_SESSION_GENESIS_HASH, make_bootstrap_current_best,
    valid_accepted_ledger_record,
)
from true_rl_agent.candidates.component_smoke_runtime import run_component_smoke_check
from true_rl_agent.candidates.prompt_projection import assert_public_only_prompt_payload
from true_rl_agent.candidates.schema import CandidateChangeSet, CandidatePackageRef
from true_rl_agent.memory import PublicContextBuilder
from true_rl_agent.prompting.serialization import canonical_prompt_json_bytes
from true_rl_agent.prompting.task_sections import MAX_TASK_PROMPT_SECTIONS_BYTES
from true_rl_agent.tasks.push_recovery.component_smoke import PushRecoveryComponentSmokeAdapter
from true_rl_agent.tasks.push_recovery.config import PushRecoveryTaskConfig
from true_rl_agent.tasks.push_recovery.metrics import project_push_split
from true_rl_agent.tasks.push_recovery.prompt_adapter import PushRecoveryPromptAdapter, _minimum_core_push_prompt_sections
from true_rl_agent.tasks.push_recovery.reference_program import RECOVERY_REFERENCE_SOURCE
from true_rl_agent.tasks.push_recovery.reward_inputs import PushRecoveryV2RewardInputs


def _dev(successes=3):
    summary = {"schema_version": "true_rl_agent.push_recovery.rollout.v2", "status": "pass",
        "aggregate": {"episodes": 4, "successes": successes, "success_rate": successes / 4,
            "eligible_episode_count": 4, "control_episode_count": 0,
            "planned_push_event_count": 4, "pushed_episode_count": 4, "push_event_count": 4,
            "recovered_event_count": successes, "invalid_injection_count": 0,
            "failure_breakdown": {"total": 4, "counts": {"success_recovered": successes, "not_recovered": 4 - successes}}}}
    errors = []
    projected = project_push_split([summary], split_name="public_dev", errors=errors)
    assert not errors, errors
    return projected


def _candidate(root, source):
    (root / "training_code").mkdir()
    (root / "training_code/components.py").write_text(source, encoding="utf-8")
    (root / "candidate_manifest.yaml").write_text(
        "schema_version: true_rl_agent.candidate.v1\ncandidate_id: push-v2-smoke\n"
        "entrypoint: training_code.components:build_training_components\n", encoding="utf-8")
    return CandidatePackageRef(candidate_id="push-v2-smoke", path=str(root),
        change_set=CandidateChangeSet(frozenset({"reward_callable"})))


class PushRecoveryV2PromptSmokeTests(unittest.TestCase):
    def setUp(self):
        self.config = PushRecoveryTaskConfig.for_cell("push.go2_12dof.recovery_v2")
        self.adapter = PushRecoveryPromptAdapter(self.config)

    def test_v2_contract_matches_runtime_fields_and_reward_authority(self):
        sections = self.adapter.build_prompt_sections(public_records=(), public_context={})
        contract = sections["reward_inputs_contract"]
        self.assertEqual(set(contract["fields"]), set(PushRecoveryV2RewardInputs.field_names()))
        self.assertEqual(contract["field_order"], list(PushRecoveryV2RewardInputs.field_names()))
        self.assertEqual(sections["task"]["candidate_capability_profile"], "push_complete_recovery_v2")
        self.assertEqual(sections["task"]["reward_program"]["authority"], "complete_replacement")
        self.assertIn("exactly once", sections["task"]["reward_program"]["integration"])
        self.assertEqual(sections["known_good_examples"], [])
        raw = json.dumps(sections)
        for obsolete in ("velocity delta", "observed abrupt", "G1", "additive", "0.45", "0.25 * tilt"):
            self.assertNotIn(obsolete, raw)
        assert_public_only_prompt_payload(sections)
        self.assertLess(canonical_prompt_json_bytes(sections), MAX_TASK_PROMPT_SECTIONS_BYTES)

    def test_canonical_dev_counts_success_label_and_source_match_runtime(self):
        a = {"round_id": "old-best", "candidate": {"candidate_id": "A"}, "train_result": {"summary": {}},
            "decision": {"accepted": True}, "public_result": {"metrics": {"public_dev": _dev(4)}}}
        b = {"trial_id": "g2-B", "trial_status": "not_selected", "status_reason": "sibling_not_selected",
            "candidate": {"candidate_id": "B", "program_source_checksum": "sha256:" + "b" * 64}, "train": {},
            "public_dev": {"public_summary": _dev()}, "runtime_activation_summary": {"reward_term_calls": 17}}
        records = [a, {"round_id": "g2", "generation": {"trials": [b]}},
            {"round_id": "g3", "generation": {"trials": [{"trial_status": "authoring_failed"}]}}]
        projected = self.adapter.project_prompt_public_history(public_records=records, initial_anchor_public_evidence=None)
        self.assertEqual(projected.records, self.adapter.project_prompt_public_history(
            public_records=projected.records, initial_anchor_public_evidence=None).records)
        sections = self.adapter.build_prompt_sections(public_records=records, public_context={})
        feedback, runtime = sections["public_feedback_summary"], sections["runtime_feedback_summary"]
        for key in ("source_round_id", "source_candidate_id", "program_source_checksum"):
            self.assertEqual(feedback[key], runtime[key])
        self.assertEqual(feedback["source_candidate_id"], "B")
        self.assertEqual(feedback["latest_public_score"], 0.75)
        self.assertEqual(feedback["latest_failure_mode"], "not_recovered_dominated")
        self.assertEqual(feedback["public_dev"]["recovery_counts"]["recovered_event_count"], 3)
        self.assertEqual(runtime["latest_runtime_activation"]["reward_term_calls"], 17)
        b.pop("public_dev")
        b.pop("runtime_activation_summary")
        sections = self.adapter.build_prompt_sections(public_records=records, public_context={})
        self.assertEqual(sections["public_feedback_summary"]["status"], "no_public_dev_for_selected_trial")
        self.assertEqual(sections["runtime_feedback_summary"]["status"], "not_available")
        fallback = self.adapter.build_prompt_sections(public_records=[a], public_context={})
        self.assertEqual(fallback["public_feedback_summary"]["source_selection"], "accepted_current_best")

    def test_full_provider_and_minimum_core_exclude_accept_and_private_detail(self):
        record = valid_accepted_ledger_record(round_id="accepted", candidate_id="winner", public_score=0.987654321)
        metrics = record["public_result"]["metrics"]
        metrics["public_dev"] = _dev()
        metrics["recovery_evidence_gate"] = {"status": "pass", "failures": [{"reason": "accept-canary-746"}]}
        metrics["sealed_validation_gate"]["reason"] = "sealed-canary-746"
        metrics["public_accept"]["opaque_detail"] = "accept-canary-746"
        metrics["private_debug"] = "private-canary-746"
        original = copy.deepcopy(record)
        payload = PublicContextBuilder(task_prompt_adapter=self.adapter,
            acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL, initial_current_best=make_bootstrap_current_best(),
            session_genesis_hash=TEST_SESSION_GENESIS_HASH).build_prompt_payload([record])
        direct = self.adapter.build_prompt_sections(public_records=[record], public_context={"best_public_score": 0.987654321})
        minimum = _minimum_core_push_prompt_sections(direct)
        for value in (payload, direct, minimum):
            raw = json.dumps(value)
            for forbidden in ("0.987654321", "accept-canary-746", "sealed-canary-746", "private-canary-746"):
                self.assertNotIn(forbidden, raw)
            assert_public_only_prompt_payload(value)
        self.assertEqual(record, original)
        self.assertEqual(direct["public_feedback_summary"]["sealed_validation_gate"], {"status": "pass"})
        self.assertEqual(minimum["public_feedback_summary"]["public_dev"]["recovery_counts"]["successes"], 3)
        self.assertLess(canonical_prompt_json_bytes(minimum), MAX_TASK_PROMPT_SECTIONS_BYTES)

    def test_dev_strata_sum_only_present_integer_counts(self):
        dev = _dev()
        dev["seed_results"][0]["stratum_results"] = {
            "low": {"episodes": 2, "successes": 2}, "high": {"episodes": 2, "successes": 1},
            "medium": {"episodes": None, "successes": None}, "sealed": {"episodes": 746, "successes": 746}}
        dev["seed_results"].append(copy.deepcopy(dev["seed_results"][0]))
        record = {"candidate": {"candidate_id": "strata"}, "train_result": {}, "decision": {"accepted": False},
            "public_result": {"metrics": {"public_dev": dev}}}
        feedback = self.adapter.build_prompt_sections(public_records=[record], public_context={})["public_feedback_summary"]
        self.assertEqual(feedback["public_dev"]["stratum_results"], {
            "low": {"episodes": 4, "successes": 4, "n_over_N": "4/4"},
            "high": {"episodes": 4, "successes": 2, "n_over_N": "2/4"}})
        record["public_result"]["metrics"]["public_dev"] = _dev()
        feedback = self.adapter.build_prompt_sections(public_records=[record], public_context={})["public_feedback_summary"]
        self.assertEqual(feedback["public_dev"]["stratum_results"], {})

    def test_private_trial_reason_and_sealed_fields_are_not_echoed(self):
        trial = {"trial_id": "g1-A", "trial_status": "formal_rejected", "train": {}, "candidate": {"candidate_id": "A"},
            "status_reason": "sealed-canary-746", "public_dev": {"public_summary": _dev()},
            "public_accept": {"public_summary": {"score_stats": {"mean": 0.987654321}}},
            "sealed_validation": {"status": "fail", "reason": "sealed-canary-746"}}
        records = [{"generation": {"trials": [trial], "parent_current_best": {"public_score": 0.987654321}}}]
        value = self.adapter.project_prompt_public_history(public_records=records, initial_anchor_public_evidence=None)
        raw = json.dumps(value.records)
        self.assertNotIn("sealed-canary", raw)
        self.assertNotIn("0.987654321", raw)
        trial["status_reason"] = "candidate_runtime_failed_public_dev"
        sections = self.adapter.build_prompt_sections(public_records=records, public_context={})
        self.assertEqual(sections["public_feedback_summary"]["latest_round"]["status_reason"], "candidate_runtime_failed_public_dev")

    def test_reference_and_candidate_smoke_execute_on_both_robot_dimensions(self):
        reference = (PROJECT_ROOT / RECOVERY_REFERENCE_SOURCE).read_text(encoding="utf-8")
        for cell, joints in (("push.g1_37dof.recovery_v2", 37), ("push.go2_12dof.recovery_v2", 12)):
            with self.subTest(cell=cell):
                adapter = PushRecoveryComponentSmokeAdapter(PushRecoveryTaskConfig.for_cell(cell))
                fixture = adapter.build_component_smoke_fixture()
                self.assertIsInstance(fixture.reward_inputs, PushRecoveryV2RewardInputs)
                self.assertEqual(fixture.reward_inputs.joint_pos.shape, (4, joints))
                self.assertEqual(fixture.reward_inputs.last_action.shape, (4, joints))
                self.assertEqual(fixture.reward_inputs.push_vec_public[0].tolist(), [0.0, 0.0])
                self.assertEqual(adapter.candidate_capability_ids,
                    frozenset({"reward_term", "diagnostic_term", "reward_weight_scale_schedule"}))
                candidate_source = reference.replace("weight=3.0", "weight=2.5").replace(
                    "inputs.joint_pos_error.square().mean(dim=1)", f"inputs.joint_pos_error[:, {joints - 1}].square()")
                for source in (reference, candidate_source):
                    with tempfile.TemporaryDirectory() as tmp:
                        result = run_component_smoke_check(_candidate(Path(tmp), source), adapter)
                    self.assertEqual(result["status"], "pass", result)
                    self.assertEqual(result["checked_reward_terms"], 5)

    def test_smoke_rejects_candidate_using_wrong_robot_dimension(self):
        reference = (PROJECT_ROOT / RECOVERY_REFERENCE_SOURCE).read_text(encoding="utf-8")
        source = reference.replace("inputs.joint_pos_error.square().mean(dim=1)", "inputs.joint_pos_error[:, 36].square()")
        with tempfile.TemporaryDirectory() as tmp:
            result = run_component_smoke_check(_candidate(Path(tmp), source), PushRecoveryComponentSmokeAdapter(self.config))
        self.assertEqual(result["status"], "fail")

    def test_force_smoke_runs_tensor_broadcasting_and_rejects_invalid_outputs(self):
        # The first real Go2 5-round candidate used this valid Tensor expression.
        source = '''
from true_rl_agent.candidates.components import RewardTerm, TrainingComponents

def stationary_push_compensated(inputs):
    velocity_scale = (9.81 * inputs.nominal_base_height).sqrt()
    push_mask = inputs.push_active.float().unsqueeze(1)
    push_scaled = (inputs.push_vec_public / velocity_scale.unsqueeze(1)) * push_mask
    planar_velocity = inputs.base_lin_vel_scaled[:, :2] - push_scaled
    planar_speed = planar_velocity.square().sum(dim=1).sqrt() / 0.08
    angular_speed = inputs.base_ang_vel_scaled.square().sum(dim=1).sqrt() / 0.5
    planar_core = planar_speed.clamp(max=1.0)
    angular_core = angular_speed.clamp(max=1.0)
    planar_penalty = 0.5 * planar_core.square() + planar_speed - planar_core
    angular_penalty = 0.5 * angular_core.square() + angular_speed - angular_core
    return 1.0 - 0.25 * (planar_penalty + angular_penalty)

def build_training_components():
    return TrainingComponents(name="push_compensated", reward_terms=[
        RewardTerm(name="stationary_push_compensated", weight=1.0,
                   fn=stationary_push_compensated)])
'''
        output = "return 1.0 - 0.25 * (planar_penalty + angular_penalty)"
        cases = (
            (source, "pass", "component_smoke_passed"),
            (source.replace(output, "return inputs.joint_pos"), "fail", "reward_output_wrong_shape"),
            (source.replace(output, "return inputs.roll * float('nan')"), "fail", "reward_output_nonfinite"),
        )
        for cell in ("push.go2_12dof.recovery_v3", "push.g1_37dof.recovery_v3"):
            adapter = PushRecoveryComponentSmokeAdapter(PushRecoveryTaskConfig.for_cell(cell))
            inputs = adapter.build_component_smoke_fixture().reward_inputs
            for field in inputs.field_names():
                value = getattr(inputs, field)
                self.assertIsInstance(value, torch.Tensor)
                self.assertEqual(value.device.type, "cpu")
                self.assertEqual(value.dtype, torch.float32)
            for candidate_source, status, reason in cases:
                with self.subTest(cell=cell, reason=reason), tempfile.TemporaryDirectory() as tmp:
                    result = run_component_smoke_check(_candidate(Path(tmp), candidate_source), adapter)
                    self.assertEqual(result["status"], status, result)
                    self.assertEqual(result["reason"], reason, result)

    def test_legacy_contract_and_smoke_remain_legacy(self):
        config = PushRecoveryTaskConfig()
        fields = PushRecoveryPromptAdapter(config).build_prompt_sections(public_records=(), public_context={})["reward_inputs_contract"]["fields"]
        self.assertNotIn("base_height_relative", fields)
        adapter = PushRecoveryComponentSmokeAdapter(config)
        self.assertEqual(adapter.candidate_capability_ids, frozenset())
        self.assertEqual(adapter.build_component_smoke_fixture().reward_inputs.joint_pos.shape, (4, 6))


if __name__ == "__main__":
    unittest.main()
