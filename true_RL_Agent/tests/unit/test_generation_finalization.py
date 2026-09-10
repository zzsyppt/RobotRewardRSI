"""Preserve final checkpoint/GPU rebinding across the task × robot matrix."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from tests.fakes.backend_environment import use_test_isaac_lab_execution_environment
from tests.integration.test_generation_session import build_fixture_foundation
from true_rl_agent.agent_core.generation_finalization import GenerationFinalPlan, bind_final_evaluation_job
from true_rl_agent.agent_core.generation_preparation import prepare_generation_foundation
from true_rl_agent.evaluation.phases import EvaluationPhase
from true_rl_agent.evaluation.subject import ExperimentSubject
from true_rl_agent.json_contract import load_strict_json_object
from true_rl_agent.runner.checkpoint import file_sha256
from true_rl_agent.runner.eval_binding import bind_eval_spec_to_checkpoint
from true_rl_agent.runner.job_argv import validate_eval_job_argv
from true_rl_agent.runner.specs import CheckpointRef
from true_rl_agent.tasks.push_recovery.composition import PushRecoveryCompositionProvider
from true_rl_agent.tasks.push_recovery.config import PushRecoveryTaskConfig
from true_rl_agent.tasks.push_recovery.generation_finalization import prepare_push_final_plan
from true_rl_agent.tasks.stair_climbing.composition import StairClimbingCompositionProvider
from true_rl_agent.tasks.stair_climbing.config import StairClimbingTaskConfig
from true_rl_agent.tasks.stair_climbing.generation_finalization import (
    prepare_stair_final_plan, read_stair_final_plan, bind_stair_final_config,
)
from true_rl_agent.tasks.stair_climbing.row_catalog import stair_robot_row


ROOT = Path(__file__).resolve().parents[2]


class GenerationFinalizationTests(unittest.TestCase):
    def test_stair_cells_freeze_real_execution_fields_and_reuse_the_same_private_study(self):
        for cell in ("stair.g1_37dof.unified_v3", "stair.go2_12dof.unified_v3"):
            with self.subTest(cell=cell), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                config = StairClimbingTaskConfig.defaults(ROOT / stair_robot_row(cell).protocol_relative_path)
                original_suite_hash = config.ordered_evaluation_suite.suite_hash
                plan = prepare_stair_final_plan(
                    output_dir=root, session_id="shared-final-study", config=config,
                    required_closed_generations=15, resume_existing=False,
                )
                path = root / "final_confirmation" / "plan.json"
                stored = load_strict_json_object(path)
                self.assertEqual(stored["execution"], {
                    "phase": "sealed_validation", "job_roles": ["genesis", "final"],
                    "seed_count_per_job": 7, "total_eval_seed_units": 14,
                })
                self.assertNotIn("evaluation_profile_hash", stored)
                loaded = read_stair_final_plan(path, expected_session_id=plan.session_id, expected_config=config)
                self.assertEqual(loaded, plan)
                bound = bind_stair_final_config(config, plan=loaded, plan_path=path)
                self.assertEqual(bound.ordered_evaluation_suite.suite_hash, plan.task_commitment["evaluation_suite_hash"])
                self.assertEqual(bound.ordered_evaluation_suite.cohort_hash("sealed_validation"),
                                 plan.task_commitment["sealed_cohort_hash"])
                self.assertEqual(config.ordered_evaluation_suite.suite_hash, original_suite_hash)
                self.assertNotEqual(bound.ordered_evaluation_suite.suite_hash, original_suite_hash)
                before = {p.name: p.read_bytes() for p in path.parent.iterdir()}
                resumed = prepare_stair_final_plan(
                    output_dir=root, session_id=plan.session_id, config=config,
                    required_closed_generations=15, resume_existing=True,
                )
                self.assertEqual(resumed, plan)
                self.assertEqual(before, {p.name: p.read_bytes() for p in path.parent.iterdir()})

    def test_push_uses_the_same_plan_shape_without_claiming_new_holdout_generalization(self):
        with tempfile.TemporaryDirectory() as directory:
            config = PushRecoveryTaskConfig(reference_program_genesis=True)
            plan = prepare_push_final_plan(
                output_dir=Path(directory), session_id="shared-push-final", config=config,
                required_closed_generations=15, resume_existing=False,
            )
            self.assertEqual(GenerationFinalPlan.from_mapping(plan.to_dict()), plan)
            self.assertEqual(plan.phase, EvaluationPhase.SEALED_VALIDATION)
            self.assertEqual(plan.total_eval_seed_units, 14)
            self.assertEqual(plan.seed_cohort_hash, config.seed_cohort_hash())
            self.assertFalse(plan.task_commitment["new_seed_generalization_claim"])
            self.assertFalse(plan.genesis_commitment()["feedback_allowed"])

    def test_imported_checkpoint_is_rematerialized_on_the_session_final_lane(self):
        for provider, cell in (
            (StairClimbingCompositionProvider(), "stair.g1_37dof.unified_v3"),
            (StairClimbingCompositionProvider(), "stair.go2_12dof.unified_v3"),
            (PushRecoveryCompositionProvider(), "push.g1_29dof.reference_v1"),
        ):
            with self.subTest(cell=cell), tempfile.TemporaryDirectory() as directory, use_test_isaac_lab_execution_environment():
                root = Path(directory).resolve()
                foundation = build_fixture_foundation(root, provider=provider, cell=cell)
                composition = foundation.composition
                backend = composition.backend_adapter
                checkpoint_path = root / "best.pt"
                checkpoint_path.write_bytes(b"sealed-checkpoint-fixture")
                for previous_gpu in (None, foundation.gpu_bindings[1]):
                    with self.subTest(previous_gpu=previous_gpu):
                        jobs = backend.materialize_reference_jobs(
                            session_id="source-reference", task_spec=composition.task_plugin.task_spec,
                            reference=composition.reference_program_artifact,
                            artifact_root=root / "reference-artifacts",
                            policy_initialization=composition.policy_initialization,
                            backend_execution_environment_manifest_hash="b" * 64,
                            gpu_process_binding=previous_gpu,
                        )
                        spec = jobs.eval_spec
                        checkpoint = CheckpointRef(
                            uri=spec.checkpoint_output_plan.uri, checksum=file_sha256(checkpoint_path),
                            role="eval", sealed=True,
                            metadata={"local_path": str(checkpoint_path),
                                "training_subject": spec.subject.to_public_dict(),
                                "subject_manifest_hash": spec.subject_manifest_hash,
                                "program_source_checksum": spec.program_source_checksum,
                                "training_binding_hash": spec.training_binding_hash,
                                "capability_profile_hash": spec.capability_profile_hash,
                                "execution_device": spec.execution_device,
                                "gpu_process_binding": previous_gpu.to_dict() if previous_gpu else None},
                        )
                        template = bind_eval_spec_to_checkpoint(spec, checkpoint)
                        original = template.to_public_dict()
                        target = replace(template, subject=ExperimentSubject.candidate("accepted-candidate"),
                            subject_manifest_hash="c" * 64, program_source_checksum="sha256:" + "d" * 64,
                            training_binding_hash="e" * 64)
                        bound = bind_final_evaluation_job(foundation, target=target, role="final")
                        phase = backend.materialize_eval_phase(eval_plan=bound, phase=foundation.final_confirmation_plan.phase)
                        argv = phase.metadata["cli_argv"]
                        validate_eval_job_argv(phase, argv)
                        expected = {
                            "--subject-kind": "candidate", "--subject-id": "accepted-candidate",
                            "--subject-manifest-hash": target.subject_manifest_hash,
                            "--program-source-checksum": target.program_source_checksum,
                            "--training-binding-hash": target.training_binding_hash,
                            "--checkpoint-sha256": checkpoint.checksum,
                            "--seed-cohort-hash": foundation.final_confirmation_plan.seed_cohort_hash,
                            "--gpu-host-execution-device": "cuda:0",
                            "--gpu-process-execution-device": "cuda:0",
                            "--gpu-resource-binding-hash": foundation.gpu_bindings[0].resource_binding_hash,
                            "--backend-execution-environment-manifest-hash": "b" * 64,
                        }
                        for option, value in expected.items():
                            self.assertEqual(argv.count(option), 1, option)
                            self.assertEqual(argv[argv.index(option) + 1], value, option)
                        for name in ("subject_manifest_hash", "program_source_checksum", "training_binding_hash", "capability_profile_hash"):
                            self.assertEqual(bound.metadata[name], getattr(target, name))
                        self.assertEqual(bound.metadata["evaluation_subject"], target.subject.to_public_dict())
                        self.assertEqual(template.to_public_dict(), original)
                        self.assertEqual(bound.checkpoint, checkpoint)
                        self.assertIsNone(bound.attempt_scope)

    def test_self_consistent_foreign_workload_cannot_replace_the_cell_contract(self):
        with tempfile.TemporaryDirectory() as directory, use_test_isaac_lab_execution_environment():
            foundation = build_fixture_foundation(Path(directory).resolve(),
                provider=PushRecoveryCompositionProvider(), cell="push.g1_29dof.reference_v1")
            cost = foundation.cost_contract
            foreign = replace(cost, scientific_cost_basis=replace(cost.scientific_cost_basis, task_protocol_hash="f" * 64))
            before = (foundation.config.output_dir / "generation_session_cost.json").read_bytes()
            with self.assertRaisesRegex(ValueError, "workload differs from the resolved task"):
                prepare_generation_foundation(foundation.config, composition=foundation.composition,
                    source_commit=foundation.source_commit,
                    inventory=(foundation.config.gpu0_uuid, foundation.config.gpu1_uuid), cost=foreign,
                    final_plan=foundation.final_confirmation_plan, provider_policy=foundation.provider_policy)
            self.assertEqual((foundation.config.output_dir / "generation_session_cost.json").read_bytes(), before)
