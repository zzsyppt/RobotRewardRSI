"""Direct archive, explicit parent and bounded description degradation."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tests.unit import test_stair_trajectory_production as fixture
from tests.unit.trajectory_fixtures import sample_episode
from true_rl_agent.evaluation.phases import EvaluationPhase
from true_rl_agent.evaluation.trajectory_translation.measurement import measure_behavior
from true_rl_agent.runner.numeric_evidence import load_numeric_evidence_manifest, NumericEvidenceError
from true_rl_agent.tasks.stair_climbing import trajectory_pipeline as pipeline
from true_rl_agent.tasks.stair_climbing.phased_evaluation import finalize_stair_eval_phase_summary
from true_rl_agent.tasks.stair_climbing.trajectory_production import project_production_report
from true_rl_agent.tasks.stair_climbing.behavior import comparison_operands, compare_stair_behavior
from true_rl_agent.tasks.stair_climbing.prompt_adapter import _latest_behavior_feedback


class DirectDescriptionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.ProductionTrajectoryTests.setUpClass()
        cls.f = fixture.ProductionTrajectoryTests

    @classmethod
    def tearDownClass(cls):
        cls.f.tearDownClass()

    def finalization(self, budget):
        identity=self.f.identity
        spec=SimpleNamespace(phase=EvaluationPhase.PUBLIC_DEV,
            job_identity=lambda:SimpleNamespace(to_public_dict=lambda:identity),
            execution_checkpoint=lambda:SimpleNamespace(checksum=identity["checkpoint_hash"]),
            evaluation_plan_hash=identity["evaluation_plan_hash"],
            evaluation_binding=SimpleNamespace(to_public_dict=lambda:identity["evaluation_binding"]))
        summary={"status":"postprocessing","task_evidence":{"numeric_evidence":self.f.reference,
            "summary":load_numeric_evidence_manifest(self.f.reference)["metadata"]["summary"],
            "execution_identity":{"seed_namespace":identity["seed_namespace"]}},
            "evaluation_execution_cost":{"splits":{"public_dev":{}}},
            **{k:identity.get(k) for k in ("environment_source_preflight","backend_execution_environment_verification")}}
        before=deepcopy(summary["task_evidence"]["summary"])
        result=finalize_stair_eval_phase_summary(spec,summary,
            config=SimpleNamespace(is_unified_protocol=True,cell_variant_id="stair.go2_12dof.unified_v3"),
            output_dir=self.f.root/"degraded",trajectory_report_max_bytes=budget)
        self.assertEqual(result["status"],"pass")
        self.assertEqual(result["task_evidence"]["summary"],before)
        return result["task_evidence"]

    def test_budget_degradation_preserves_evaluation_and_reaches_feedback(self):
        evidence=self.finalization(128)
        self.assertNotIn("behavior",evidence)
        self.assertNotIn("trajectory_build",evidence)
        self.assertEqual(evidence["description_status"]["reason"],"budget_exceeded")
        self.assertIn("behavior_comparison",evidence)
        shown=_latest_behavior_feedback([{"generation":{"trials":[{"trial_id":"fixture",
            "public_dev":{"public_summary":{"description_status":evidence["description_status"]}}}]}}])
        self.assertEqual(shown["description_status"],evidence["description_status"])

    def test_unsupported_capability_is_bounded_but_identity_is_not_swallowed(self):
        with patch.object(pipeline,"SUPPORTED_CELLS",()):
            evidence=self.finalization(10240)
            self.assertEqual(evidence["description_status"]["reason"],"unsupported_capture_contract")
            with self.assertRaisesRegex(ValueError,"source identity"):
                pipeline.build_phase_description(self.f.reference,expected_identity={**self.f.identity,"checkpoint_hash":"wrong"})
            with patch.object(pipeline,"iter_numeric_evidence",side_effect=NumericEvidenceError("chunk hash mismatch")):
                with self.assertRaisesRegex(NumericEvidenceError,"hash"):
                    self.finalization(128)

    def pair(self):
        left=sample_episode(80);right=sample_episode(80)
        for e,v in ((left,1.),(right,2.)):
            e.task.update(cell_id="same-geometry",stratum_id="fixture")
            e.arrays["base_pos_w"][:,0]=e.times*v
        a,b=[comparison_operands([measure_behavior(e)]) for e in (left,right)]
        domain=deepcopy(self.f.domain)
        parent=project_production_report(domain,self.f.summary)["public_report"]
        candidate=project_production_report({**domain,"checkpoint_hash":"child"},self.f.summary)["public_report"]
        a["domain"]=parent["domain"];b["domain"]=candidate["domain"]
        return parent,candidate,a,b

    def test_explicit_parent_absolute_delta_denominators_and_mismatch(self):
        parent,candidate,a,b=self.pair()
        kwargs=dict(expected_parent_checkpoint_hash=parent["domain"]["checkpoint_hash"],parent_comparison=a,candidate_comparison=b)
        result=compare_stair_behavior(parent,candidate,**kwargs)
        self.assertEqual(result["matched_conditions"],1)
        self.assertTrue(any(r["unit"]=="m/s" and r["parent"][0]==1 and r["candidate"][0]==2 and r["delta"]==1 for r in result["rows"]))
        self.assertTrue(all(r["parent_n"]==r["candidate_n"]==1 for r in result["rows"]))
        self.assertEqual(compare_stair_behavior(None,candidate,**kwargs)["status"],"available")
        with self.assertRaisesRegex(ValueError,"parent checkpoint"):
            compare_stair_behavior(parent,candidate,**{**kwargs,"expected_parent_checkpoint_hash":"sibling"})
        b["descriptors"]["all.speed"]["unit"]="cm/s"
        self.assertEqual(compare_stair_behavior(parent,candidate,**kwargs)["status"],"not_comparable")

    def test_short_or_missing_terminal_window_never_matches_full_window_as_zero(self):
        ep=sample_episode(8);ep.task.update(cell_id="same",stratum_id="fixture")
        ep.arrays["base_pos_w"][:,0]=ep.times
        facts=comparison_operands([measure_behavior(ep)])["conditions"]["same"]["terminal"]["success"]["facts"]
        self.assertEqual(facts["forward_speed"][:2],[None,0])
        self.assertEqual(facts["height"][:2],[None,0])
        self.assertEqual(facts["progress"][:2],[.16,1])

    def test_source_byte_changes_do_not_qualify_or_reject_semantic_contract(self):
        from true_rl_agent.tasks.stair_climbing.trajectory_adapter import _load_contract
        from true_rl_agent.tasks.stair_climbing.trajectory_capabilities import resolve_capabilities
        archive=_load_contract(self.f.reference)
        contract=archive.manifest["metadata"]["measurement_contract"]
        contract["joint_position"]["recorder_source_sha256"]="1"*64
        contract["nominal_terrain"]["source_sha256"]={k:"2"*64 for k in contract["nominal_terrain"]["source_sha256"]}
        self.assertEqual(resolve_capabilities(archive,self.f.reference)[0]["nominal_joint_limits"]["status"],"available")


if __name__ == "__main__":
    unittest.main()
