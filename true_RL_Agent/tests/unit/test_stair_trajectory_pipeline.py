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
from true_rl_agent.tasks.stair_climbing import trajectory_production as production
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

    def finalization(self, budget, *, reference=None):
        reference = self.f.reference if reference is None else reference
        identity=load_numeric_evidence_manifest(reference)["identity"]
        spec=SimpleNamespace(phase=EvaluationPhase.PUBLIC_DEV,
            job_identity=lambda:SimpleNamespace(to_public_dict=lambda:identity),
            execution_checkpoint=lambda:SimpleNamespace(checksum=identity["checkpoint_hash"]),
            evaluation_plan_hash=identity["evaluation_plan_hash"],
            evaluation_binding=SimpleNamespace(to_public_dict=lambda:identity["evaluation_binding"]))
        summary={"status":"postprocessing","task_evidence":{"numeric_evidence":reference,
            "summary":load_numeric_evidence_manifest(reference)["metadata"]["summary"],
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

    def test_budget_is_applied_at_author_boundary_after_evaluation(self):
        evidence=self.finalization(128)
        self.assertIn("behavior",evidence)
        self.assertNotIn("trajectory_build",evidence)
        self.assertNotIn("description_status",evidence)
        self.assertIn("behavior_comparison",evidence)
        from true_rl_agent.tasks.stair_climbing.trajectory_production import author_behavior_payload
        with self.assertRaisesRegex(ValueError,'mandatory facts exceed'):
            author_behavior_payload({'source':{},'profile':evidence['behavior']})

    def test_unsupported_capability_is_bounded_but_identity_is_not_swallowed(self):
        with patch.object(pipeline,"SUPPORTED_CELLS",()):
            evidence=self.finalization(10240)
            self.assertEqual(evidence["description_status"]["reason"],"unsupported_capture_contract")
            with self.assertRaisesRegex(ValueError,"source identity"):
                pipeline.build_phase_description(self.f.reference,expected_identity={**self.f.identity,"checkpoint_hash":"wrong"})
            with patch.object(pipeline,"iter_numeric_evidence",side_effect=NumericEvidenceError("chunk hash mismatch")):
                with self.assertRaisesRegex(NumericEvidenceError,"hash"):
                    self.finalization(128)

    def test_measurement_failure_drains_archive_and_preserves_science(self):
        original = production.iter_stair_episodes
        consumed = []
        def episodes(*args, **kwargs):
            for ep in original(*args, **kwargs):
                consumed.append((ep.run_index, ep.env_index))
                yield ep
        with patch.object(production, "iter_stair_episodes", side_effect=episodes), patch.object(
                production, "measure_behavior", side_effect=ArithmeticError("synthetic measurement failure")) as measure, self.assertLogs(
                "true_rl_agent.tasks.stair_climbing.phased_evaluation", level="WARNING") as logged:
            evidence = self.finalization(10240)
        self.assertIn("ArithmeticError: synthetic measurement failure", "\n".join(logged.output))
        self.assertEqual(len(consumed), len(self.f.fixture.cells))
        measure.assert_called_once()
        self.assertEqual(evidence["description_status"]["reason"], "description_computation_failed")
        self.assertEqual(evidence["description_status"]["stage"], "measurement")
        self.assertEqual(evidence["description_status"]["error_type"], "ArithmeticError")
        self.assertEqual(evidence["description_status"]["error_message"], "synthetic measurement failure")
        self.assertNotIn("behavior", evidence)
        self.assertNotIn("behavior_full", evidence)
        self.assertNotIn("behavior_comparison", evidence)
        shown = _latest_behavior_feedback([{"generation":{"trials":[{"trial_id":"fixture",
            "candidate":{"candidate_id":"fixture"}, "public_dev":{"checkpoint_hash":"checkpoint",
            "public_summary":{"description_status":evidence["description_status"]}}}]}}],
            current_best={"anchor_kind":"candidate", "anchor_id":"fixture", "accepted_round_id":"fixture", "checkpoint_hash":"checkpoint"})
        self.assertEqual(shown["description_status"], evidence["description_status"])

    def test_late_reader_failure_overrides_measurement_degradation(self):
        first = next(production.iter_stair_episodes(self.f.reference, source_id="test"))
        def episodes(*args, **kwargs):
            yield first
            raise NumericEvidenceError("late second-read chunk hash mismatch")
        with patch.object(production, "iter_stair_episodes", side_effect=episodes), patch.object(
                production, "measure_behavior", side_effect=ArithmeticError("measurement failure")) as measure:
            with self.assertRaisesRegex(NumericEvidenceError, "late second-read"):
                self.finalization(10240)
        measure.assert_called_once()

    def test_aggregation_and_render_failures_preserve_science_without_partial_report(self):
        # The second render is the full-report replay before publication. It must
        # not leave the first render's successful full card attached on failure.
        claims = production._behavior_claims
        for stage in ("aggregation", "rendering"):
            calls = []
            def fail_on_replay(summary):
                calls.append(None)
                if len(calls) == 2:
                    raise ValueError("synthetic replay renderer failure")
                return claims(summary)
            target = "aggregate_behavior" if stage == "aggregation" else "_behavior_claims"
            side_effect = KeyError("synthetic aggregate failure") if stage == "aggregation" else fail_on_replay
            with self.subTest(stage=stage), patch.object(production, target, side_effect=side_effect), self.assertLogs(
                    "true_rl_agent.tasks.stair_climbing.phased_evaluation", level="WARNING"):
                evidence = self.finalization(10240)
                self.assertEqual(evidence["description_status"]["reason"], "description_computation_failed")
                self.assertEqual(evidence["description_status"]["stage"], stage)
                self.assertNotIn("behavior", evidence)
                self.assertNotIn("behavior_full", evidence)
                self.assertEqual("behavior_comparison" in evidence, stage == "rendering")
                if stage == "rendering":
                    self.assertEqual(len(calls), 2)

    def test_source_identity_required_channels_and_terrain_binding_stay_strict(self):
        with self.assertRaisesRegex(ValueError, "source identity"):
            pipeline.build_phase_description(self.f.reference,
                expected_identity={**self.f.identity, "checkpoint_hash":"wrong"})
        def missing_channel(identity, context, chunks):
            identity.update(evaluation_plan_hash="e"*64)
            identity["evaluation_binding"]["evaluation_binding_hash"] = "f"*64
            del chunks[1][1]["joint_pos"]
        reference = self.f.fixture._archive(missing_channel)
        with patch.object(production, "measure_behavior") as measure:
            with self.assertRaises(NumericEvidenceError):
                self.finalization(10240, reference=reference)
            measure.assert_not_called()
        episodes = list(production.iter_stair_episodes(self.f.reference, source_id="test"))[:2]
        episodes[1].task["progress_direction_world"] = [0., 1., 0.]
        with patch.object(production, "iter_stair_episodes", return_value=iter(episodes)), patch.object(
                production, "measure_behavior", side_effect=ArithmeticError("measurement failure")) as measure:
            with self.assertRaisesRegex(ValueError, "nominal axes"):
                self.finalization(10240)
            measure.assert_called_once()
        with self.assertRaisesRegex(ValueError, "domain mismatch"):
            production.project_production_report({**self.f.domain, "task_id":"wrong"}, self.f.summary)

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
        self.assertFalse(any(r["condition"] is None and r["metric"] in {"net", "speed"} for r in result["rows"]))
        self.assertIn("根部位置差分水平速度1→2m/s", result["core_text"])
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

        # Prose may describe partial observed windows; comparisons require the
        # complete common window. Its denominator must not inherit prose's.
        from true_rl_agent.evaluation.trajectory_translation.cohort import aggregate_behavior
        episodes = []
        for index, (samples, speed) in enumerate(((80,.2),(8,.8),(80,1.4))):
            source = sample_episode(samples)
            source.env_index = index
            source.task.update(cell_id="same",stratum_id="fixture")
            source.arrays["base_pos_w"][:,0] = source.times*speed
            if index == 2:
                source.arrays["base_pos_w"][-3,0] = float("nan")
            episodes.append(measure_behavior(source))
        prose = aggregate_behavior(episodes)["groups"]["outcome:success"]["facts"]["terminal"]["forward_speed"]
        compared = comparison_operands(episodes)["conditions"]["same"]["terminal"]["success"]["facts"]
        self.assertEqual(prose["valid_episodes"],3)
        self.assertAlmostEqual(prose["distribution"]["median"],.8)
        self.assertEqual(compared["forward_speed"][1],1)
        self.assertAlmostEqual(compared["forward_speed"][0],.2)
        self.assertEqual(compared["height"][1],1)
        self.assertEqual(compared["progress"][1],3)

    def test_difference_ignores_zero_noise_and_keeps_named_cohort_foot_change(self):
        parent,candidate,a,_ = self.pair()
        b = deepcopy(a); b["domain"] = candidate["domain"]
        kwargs = dict(expected_parent_checkpoint_hash=parent["domain"]["checkpoint_hash"],parent_comparison=a,candidate_comparison=b)
        same = compare_stair_behavior(parent,candidate,**kwargs)
        self.assertEqual(same["rows"],[])
        for group in (b["cohort"],*b["conditions"].values()):
            group["all"]["speed"][0] += 9.54e-7
        foot = next(k for k in b["cohort"]["all"] if k.endswith(".contact"))
        a["cohort"]["all"][foot][0] = 20.
        b["cohort"]["all"][foot][0] = 60.
        result = compare_stair_behavior(parent,candidate,**kwargs)
        self.assertEqual(len(result["rows"]),1)
        row = result["rows"][0]
        self.assertEqual((row["condition"],row["metric"]),(None,'support.contact'))
        fact = row["members"][0]
        self.assertEqual((fact["parent"][0],fact["candidate"][0],fact["delta"]),(20.,60.,40.))
        self.assertIn(foot.removesuffix(".contact").rsplit(".",1)[-1],row["text"])
        self.assertEqual(result["composition"]["parent"],result["composition"]["candidate"])

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
