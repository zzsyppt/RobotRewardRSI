"""P5 evidence tests: physical fixture coverage and same-transition change."""
from __future__ import annotations

import hashlib
import importlib.util
import unittest
from dataclasses import replace
from pathlib import Path

from true_rl_agent.candidates.component_source_semantics import project_single_component_operation_delta
from true_rl_agent.candidates.verifier import StaticCandidateVerifierPolicy
from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.evaluation.subject import ExperimentSubject
from true_rl_agent.tasks.stair_climbing.component_smoke import StairClimbingComponentSmokeAdapter
from true_rl_agent.tasks.stair_climbing.config import StairClimbingTaskConfig
from true_rl_agent.tasks.stair_climbing.plugin import StairClimbingTaskPlugin
from true_rl_agent.tasks.stair_climbing.public_projection import build_stair_candidate_runtime_effect_evidence
from true_rl_agent.tasks.stair_climbing.reward_program import StairCompleteRewardDispatcher, qualify_stair_diagnostic_parent
from true_rl_agent.tasks.stair_climbing.robot_binding import resolve_stair_cell_manifest

ROOT = Path(__file__).resolve().parents[2]
HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "requires target Torch environment")
class StairP5RuntimeTests(unittest.TestCase):
    def test_nominal_fixture_keeps_cross_row_markers_and_real_contact_cases(self):
        import torch
        for filename in ("stair_unified_protocol_v3.json", "stair_go2_12dof_unified_protocol_v3.json"):
            config = StairClimbingTaskConfig.defaults(ROOT / "configs/tasks" / filename)
            manifest = resolve_stair_cell_manifest(cell_variant_id=config.cell_variant_id, config=config,
                                                   plugin=StairClimbingTaskPlugin(config))
            config = replace(config, resolved_cell_task_protocol_hash=manifest.resolved_cell_task_protocol_hash)
            adapter = StairClimbingComponentSmokeAdapter(config, manifest)
            values = adapter._tensor_values()
            self.assertAlmostEqual(float(values["base_height"][0]), adapter.nominal_morphology["support_length_m"], places=6)
            self.assertEqual(len(torch.unique(values["base_height"])), 5)
            self.assertTrue(torch.equal(values["projected_gravity"][0], torch.tensor([0., 0., -1.])))
            self.assertAlmostEqual(float(values["projected_gravity"][4, 2]), 1.0)
            if "support_first_contact" in adapter.contract.field_names:
                self.assertTrue(values["support_first_contact"][1].all())
                self.assertFalse(values["support_first_contact"][0].any())
            self.assertTrue((values["support_air_time"][2] > 0).all())
            support_z = values["support_pos_relative_to_terrain_origin"][0, :, 2]
            self.assertTrue((support_z.abs() < 0.01).all())
            self.assertTrue((values["support_pos_relative_to_terrain_origin"][4, :, 2]
                             > values["base_height"][4]).all())

    def test_modify_absolute_nonzero_does_not_substitute_for_parent_difference(self):
        config = StairClimbingTaskConfig.defaults(ROOT / "configs/tasks/stair_protocol_v2.json")
        adapter = StairClimbingComponentSmokeAdapter(config)
        parent_source = ("from true_rl_agent.candidates.components import RewardTerm, TrainingComponents\n"
            "def probe(inputs):\n    return inputs.base_height\n"
            "def build_training_components():\n    return TrainingComponents(name='fixture_program', reward_terms=[RewardTerm(name='probe', weight=1.0, fn=probe)])\n")
        def qualify(source, name):
            return qualify_stair_diagnostic_parent(contract=adapter.contract,
                subject=ExperimentSubject.candidate(name), subject_manifest_hash=canonical_json_hash({"fixture": name}),
                source_bytes=source.encode(), expected_source_checksum="sha256:" + hashlib.sha256(source.encode()).hexdigest(),
                source_name=name + ".py", static_verifier_policy=StaticCandidateVerifierPolicy(),
                trusted_reference_source_text=parent_source)
        parent = qualify(parent_source, "fixture-parent")
        for expression, expected_status in (("inputs.base_height + inputs.roll * 0.0", "fail"),
                                             ("inputs.base_height + inputs.roll * inputs.roll", "pass")):
            source = parent_source.replace("return inputs.base_height", "return " + expression)
            candidate = qualify(source, "fixture-candidate")
            delta = project_single_component_operation_delta(candidate_source=source, anchor_source=parent_source)
            context_hash = canonical_json_hash({"fixture_delta": delta})
            dispatcher = StairCompleteRewardDispatcher(candidate, num_envs=5)
            dispatcher.bind_runtime_comparison(parent_program=parent, operation_delta=delta, context_hash=context_hash)
            for step in (1, 2):
                dispatcher.compute(adapter.build_reward_inputs(), common_step=step)
            runtime = dispatcher.runtime_summary()
            self.assertTrue(runtime["nonzero_reward_contribution"])
            effect = runtime["operation_effect"]
            self.assertEqual(effect["sampled_env_transitions"], 10)
            self.assertEqual(effect["sampled_control_steps"], 2)
            summary = {"runtime_activation_summary": runtime, "runtime_diagnostic_context_hash": context_hash,
                       "program_source_checksum": candidate.program_source_checksum}
            evidence = build_stair_candidate_runtime_effect_evidence(summary, mechanism_operation_delta=delta)
            self.assertEqual(evidence["status"], expected_status)
            self.assertEqual(effect["nonzero_env_transitions"] == 0, expected_status == "fail")
            plugin = StairClimbingTaskPlugin(config)
            self.assertEqual(plugin.training_runtime_issue(train_summary=summary, mechanism_operation_delta=delta) is None,
                             expected_status == "pass")
            with self.assertRaisesRegex(ValueError, "context mismatch"):
                build_stair_candidate_runtime_effect_evidence({**summary, "runtime_diagnostic_context_hash": "x"},
                                                              mechanism_operation_delta=delta)


if __name__ == "__main__":
    unittest.main()
