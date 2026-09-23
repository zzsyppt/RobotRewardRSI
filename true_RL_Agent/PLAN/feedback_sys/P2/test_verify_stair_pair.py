"""CPU fixtures for the offline verifier; none are real robot experiments."""
from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
SPEC = importlib.util.spec_from_file_location("verify_stair_pair", Path(__file__).with_name("verify_stair_pair.py"))
verify = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verify)

import torch
from tests.unit.test_stair_numeric_capture import _TwoStepEnv
from true_rl_agent.composition import TaskCompositionContext
from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.evaluation.phases import EvaluationPhase, EvaluationPhaseResult
from true_rl_agent.runner.specs import CheckpointRef
from true_rl_agent.tasks.stair_climbing.composition import StairClimbingCompositionProvider
from true_rl_agent.tasks.stair_climbing.evaluation_runtime import run_stair_split_execution
from true_rl_agent.tasks.stair_climbing.numeric_capture import StairNumericCapture
from true_rl_agent.tasks.stair_climbing.runtime_profile import STAIR_EVAL_TRANSITION_EXTRAS_KEY
from true_rl_agent.tasks.stair_climbing.runtime_projection import stair_runtime_projection_from_manifest
from true_rl_agent.tasks.stair_climbing.seed_policy import STAIR_FORMAL_EVAL_NAMESPACE
from true_rl_agent.tasks.stair_climbing.unified_scoring import build_unified_stair_summary
from true_rl_agent.tasks.stair_climbing.phased_evaluation import STAIR_PHASE_EVIDENCE_SCHEMA_VERSION


class MixedEnv(_TwoStepEnv):
    def step(self, actions):
        obs, reward, done, extras = super().step(actions)
        if self.step_index == 2:
            raw = extras[STAIR_EVAL_TRANSITION_EXTRAS_KEY]
            half = len(self.cells)//2
            for i, cell in enumerate(self.cells[half:], start=half):
                origin = raw["terrain_env_origins_w"][i]
                raw["base_pos_w"][i] = origin + torch.tensor([cell.progress_goal_m+.01, 0., cell.total_height_m+1.])
                raw["support_pos_w"][i] = origin + torch.tensor([cell.progress_goal_m+.01, 0., cell.total_height_m+.02])
                raw["support_contact_forces_w"][i, :, 2] = 40.
            raw["forbidden_contact_forces_w"].zero_()
            raw["termination_masks"][self.term].zero_()
            done.zero_()
        return obs, reward, done, extras


class VerifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        provider = StairClimbingCompositionProvider()
        cls.composition = provider.resolve(values={**provider.cli_defaults(), "cell_variant_id":"stair.go2_12dof.unified_v3"},
            context=TaskCompositionContext(project_root=ROOT, output_dir=cls.root/"composition"))
        c = cls.composition
        config, backend, manifest = c.task_plugin.config, c.backend_adapter, c.resolved_cell_manifest
        projection = stair_runtime_projection_from_manifest(manifest)
        cells = config.ordered_evaluation_suite.private_cell_assignment("public_dev")
        checkpoint = CheckpointRef(uri="artifact://synthetic/not-real.pt", checksum="sha256:"+"c"*64, role="eval", sealed=True)
        jobs = backend.materialize_reference_jobs(session_id="synthetic-offline-fixture", task_spec=c.task_plugin.task_spec,
            reference=c.reference_program_artifact, artifact_root=cls.root/"template", policy_initialization=c.policy_initialization,
            evaluation_binding=backend.evaluation_binding_for(c.task_plugin.task_spec), backend_execution_environment_manifest_hash="e"*64)
        plan = replace(jobs.eval_spec, checkpoint=checkpoint, checkpoint_output_plan=None)
        source = {"status":"pass", "source_manifest_hash":"b"*64, "task_protocol_hash":manifest.resolved_cell_task_protocol_hash}
        environment = {"status":"pass", "backend_id":"isaac_lab_rsl_rl", "expected_manifest_hash":"e"*64, "actual_manifest_hash":"e"*64}
        for label in ("on", "off"):
            directory = cls.root/label
            directory.mkdir()
            phase = backend.materialize_eval_phase(eval_plan=plan, phase=EvaluationPhase.PUBLIC_DEV,
                capture_numeric=label=="on", numeric_evidence_dir=str(directory/"numeric_evidence") if label=="on" else None)
            identity = {**phase.job_identity().to_public_dict(), "subject_kind":phase.subject.kind,"subject_id":phase.subject.subject_id,
                        "program_source_checksum":phase.program_source_checksum,"training_binding_hash":phase.training_binding_hash,
                        "capability_profile_hash":phase.capability_profile_hash,"checkpoint_hash":checkpoint.checksum,
                        "evaluation_plan_hash":phase.evaluation_plan_hash,"evaluation_binding":phase.evaluation_binding.to_public_dict(),
                        "resolved_cell_manifest_hash":manifest.manifest_hash,"seed_namespace":STAIR_FORMAL_EVAL_NAMESPACE,
                        "splits":["public_dev"],"environment_source_preflight":source,"backend_execution_environment_verification":environment}
            cap = StairNumericCapture(directory/"numeric_evidence", identity=identity,config=config,resolved_cell_manifest=manifest) if label=="on" else None
            rollout = run_stair_split_execution(env=MixedEnv(cells,projection,config), inference_policy=lambda x:x,
                protocol=None, semantics=config.evaluation_semantics,seed_resolver=config.seed_resolver(),split="public_dev",
                seed_count=config.public_seed_count,seed_namespace=STAIR_FORMAL_EVAL_NAMESPACE,runtime_projection=projection,numeric_capture=cap)
            summary = build_unified_stair_summary(suite=config.ordered_evaluation_suite,semantics=config.evaluation_semantics,
                                                 split="public_dev",per_seed_outcomes=rollout.per_seed_outcomes)
            evidence = {"schema_version":STAIR_PHASE_EVIDENCE_SCHEMA_VERSION,"phase":"public_dev","summary":summary,
                        "execution_identity":{"checkpoint_hash":checkpoint.checksum,"seed_namespace":STAIR_FORMAL_EVAL_NAMESPACE,
                            "seed_cohort_hash":config.seed_resolver().namespace_cohort_hash(STAIR_FORMAL_EVAL_NAMESPACE),
                            "splits":["public_dev"],"execution_device":phase.execution_device,
                            "environment_source_preflight":source,"backend_execution_environment_verification":environment}}
            if cap:
                evidence["numeric_evidence"] = cap.complete(summary)
            count = sum(o.episode_steps for seed in rollout.per_seed_outcomes for o in seed)
            cost = {"splits":{"public_dev":{"closed_first_episode_control_steps":count,
                    "evaluation_env_count_per_seed":len(cells),"seed_count":config.public_seed_count,
                    "simulated_env_control_steps":len(cells)*2*config.public_seed_count,"vector_env_step_calls":2*config.public_seed_count,
                    "wall_time_s":11. if label=="on" else 10.}}}
            result = EvaluationPhaseResult(job_id=phase.job_id,task_id=phase.task_id,phase=phase.phase,evaluation_plan_hash=phase.evaluation_plan_hash,
                subject=phase.subject,subject_manifest_hash=phase.subject_manifest_hash,program_source_checksum=phase.program_source_checksum,
                training_binding_hash=phase.training_binding_hash,capability_profile_hash=phase.capability_profile_hash,
                checkpoint_hash=checkpoint.checksum,eval_protocol_hash=phase.evaluation_protocol().protocol_hash,
                evaluation_binding_hash=phase.evaluation_binding.binding_hash,execution_device=phase.execution_device,
                resource_binding_hash=None,task_evidence=evidence,execution_cost=cost,environment_source_preflight=source,
                backend_execution_environment_verification=environment,checkpoint_safety_report={}).to_private_dict()
            job = {"spec":phase.to_public_dict(),"metadata":dict(phase.metadata)}
            for name, value in (("public_dev_result.json",result),("public_dev_job.json",job)):
                (directory/name).write_text(json.dumps(value),encoding="utf-8")
            report = {"status":"PASS","mode":"run","capture_numeric":label=="on","phases_executed":["public_dev"],
                      "public_dev_result_sha256":"sha256:"+verify.sha256(directory/"public_dev_result.json"),
                      "job_file_sha256":"sha256:"+verify.sha256(directory/"public_dev_job.json"),
                      "public_dev_job_spec_hash":canonical_json_hash(job["spec"]),"public_dev_result_hash":canonical_json_hash(result),
                      "evaluation_plan_hash":phase.evaluation_plan_hash,"current_manifest_hash":manifest.manifest_hash,
                      "source_training_manifest_hash":manifest.manifest_hash,"source_train_summary_sha256":"sha256:"+"a"*64,
                      "checkpoint_sha256":checkpoint.checksum,"planned_public_dev_episodes":len(cells)*config.public_seed_count,
                      "public_dev_seed_count":config.public_seed_count,"eval_num_envs":len(cells)}
            (directory/"result.json").write_text(json.dumps(report),encoding="utf-8")
            Path(str(directory)+".time").write_text(f"peak_rss_kib={1000000 if label=='on' else 900000} elapsed_s=15.0 exit=0\n")
        cls.args = SimpleNamespace(project_root=ROOT,on_dir=cls.root/"on",off_dir=cls.root/"off")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_persisted_pair_reconstructs_and_keeps_unreached_failures(self):
        report = verify.verify(self.args)
        self.assertEqual(report["status"],"PASS", report["checks"])
        behavior = report["behavior"]
        self.assertEqual(behavior["all_episode_denominator"],864)
        self.assertEqual(behavior["primary_reason_counts"],{"forbidden_contact":432,"success":432})
        failures = [r for r in behavior["episodes"] if r["outcome"]["primary_reason"]=="forbidden_contact"]
        self.assertTrue(all(r["first_riser_root_anchor"]["status"]=="not_reached" for r in failures))
        self.assertTrue(all(r["last_pre_terminal_snapshot"] is None for r in failures))
        self.assertTrue(all(not r["events"]["forbidden_contact_manager_active"]["present_strictly_before_terminal"] for r in failures))
        self.assertEqual(report["archive"]["valid_first_episode_control_steps"],1296)

    def test_missing_measurement_is_inconclusive_not_zero(self):
        path=Path(str(self.root/"on")+".time"); saved=path.read_bytes();path.unlink()
        try:
            report=verify.verify(self.args)
            self.assertEqual(report["status"],"INCONCLUSIVE")
            self.assertIsNone(report["resources"]["extra_peak_rss_kib"])
        finally:
            path.write_bytes(saved)

    def test_resource_threshold_and_archive_corruption_fail(self):
        path=Path(str(self.root/"on")+".time"); saved=path.read_bytes()
        path.write_text("peak_rss_kib=2000000 elapsed_s=15 exit=0\n")
        try:
            self.assertEqual(verify.verify(self.args)["status"],"FAIL")
        finally:
            path.write_bytes(saved)
        chunk=self.root/"on/numeric_evidence/chunk_000000.npz"; saved=chunk.read_bytes();chunk.write_bytes(saved+b"changed")
        try:
            with self.assertRaisesRegex(RuntimeError,"hash or size"):
                verify.verify(self.args)
        finally:
            chunk.write_bytes(saved)

    def test_event_streak_is_global_any_with_ties_and_terminal_exclusion(self):
        config=self.composition.task_plugin.config
        cell=config.ordered_evaluation_suite.private_cell_assignment("public_dev")[0]
        events=verify.EpisodeEvents(cell,config.evaluation_semantics.to_dict()["episode_outcomes"]["signals"],["left","right"],.02)
        for step in range(1,7):
            contact=(step%2==0,step%2==1)
            t=SimpleNamespace(episode_step=step,abs_roll_rad=.8,abs_pitch_rad=0.,support_contacts=contact,
                support_planar_speeds_mps=(1.,1.),support_surface_clearances_m=(0.,0.),forbidden_contact=step==6,
                progress_m=0.,base_clearance_m=.4,done=step==6,termination_terms=("base_contact",) if step==6 else ())
            raw={"base_quat_w":torch.tensor([[1.,0.,0.,0.]]),"support_contact_forces_w":torch.zeros((1,2,3)),"forbidden_contact_forces_w":torch.zeros((1,1,3))}
            events.observe(t,raw,0)
        o=SimpleNamespace(episode_steps=6,env_index=0,evaluation_cell_id=cell.cell_id,to_public_dict=lambda:{"primary_reason":"forbidden_contact"})
        row=events.finish(o,1,123)
        self.assertEqual(row["events"]["slip_condition_any_support"]["first_sustained_step"],5)
        self.assertEqual(row["events"]["low_clearance_condition_any_noncontact_support"]["first_sustained_step"],3)
        self.assertTrue(row["events"]["slip_condition_any_support"]["sustained_strictly_before_terminal"])
        self.assertFalse(row["events"]["forbidden_contact_manager_active"]["present_strictly_before_terminal"])
        self.assertEqual(row["last_pre_terminal_snapshot"]["control_step"],5)


if __name__=="__main__":
    unittest.main()
