from __future__ import annotations

import copy
import argparse
import ast
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from tests.fakes.backend_environment import (
    TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_MANIFEST_HASH,
    TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_VERIFICATION,
)
from true_rl_agent.agent_core.trial_receipts import TrialReceiptJournal
from true_rl_agent.evaluation.binding import EvaluationBindingV1, canonical_json_hash
from true_rl_agent.evaluation.phases import EvaluationPhase, EvaluationPhaseResult
from true_rl_agent.evaluation.subject import ExperimentSubject
from true_rl_agent.runner.base import TrainJobResult
from true_rl_agent.runner.specs import CheckpointRef, EvalJobSpec, EvalPhaseJobSpec
from true_rl_agent.runner.subprocess_runner import SubprocessRunner
from true_rl_agent.tasks.push_recovery.config import PushRecoveryTaskConfig
from true_rl_agent.tasks.push_recovery.metrics import build_push_public_result_from_summaries
from true_rl_agent.tasks.push_recovery.phased_adapter import PUSH_PHASE_EVIDENCE_SCHEMA
from true_rl_agent.tasks.push_recovery.plugin import PushRecoveryTaskPlugin
from true_rl_agent.tasks.push_recovery.phased_evaluation import build_push_phase_summary
from true_rl_agent.tasks.push_recovery.composition import PushRecoveryCompositionProvider
from true_rl_agent.tasks.push_recovery.robot_binding import PUSH_G1_29DOF_REFERENCE_CELL_VARIANT_ID
from true_rl_agent.tasks.push_recovery.runtime_adapter import CandidateRuntimeAdapter
from true_rl_agent.composition import TaskCompositionContext, resolve_task_composition
from true_rl_agent.runner.budget import build_budget_accounting
from true_rl_agent.runner.gpu_process_binding import (
    add_gpu_process_binding_arguments, validate_gpu_process_binding_invocation,
)
from true_rl_agent.runner.specs import GpuProcessBinding
from true_rl_agent.composition.cell_manifest import add_resolved_cell_manifest_arguments
from true_rl_agent.evaluation.protocol import DEFAULT_EVAL_PROTOCOL_VERSION
from true_rl_agent.policy import PolicyInitialization


class PushPhasedEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.config = PushRecoveryTaskConfig()
        self.plugin = PushRecoveryTaskPlugin(self.config)
        self.adapter = self.plugin.phased_evaluation_adapter()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def plan(self, subject):
        protocol = self.config.evaluation_protocol()
        return EvalJobSpec(
            job_id="eval-" + subject.subject_id, task_id=self.plugin.task_spec.task_id,
            subject=subject, subject_manifest_hash="b" * 64,
            program_source_checksum="sha256:" + "a" * 64, training_binding_hash="c" * 64,
            capability_profile_hash=self.plugin.task_spec.candidate_capability_profile.profile_hash,
            checkpoint=CheckpointRef(uri="artifact://policy/best.pt", checksum="sha256:" + "d" * 64,
                                     role="eval", sealed=True),
            public_seed_count=self.config.public_seed_count,
            public_accept_seed_count=self.config.public_accept_seed_count,
            sealed_validation_seed_count=self.config.sealed_validation_seed_count,
            eval_num_envs=self.config.eval_num_envs, eval_protocol_version=self.config.eval_protocol_version,
            evaluation_binding=EvaluationBindingV1(
                eval_protocol_hash=protocol.protocol_hash,
                acceptance_protocol_hash=self.config.acceptance_protocol().protocol_hash,
                task_protocol_hash=self.plugin.task_spec.resolved_task_protocol_hash(),
                seed_cohort_hash=self.config.seed_cohort_hash(),
            ), execution_device="cuda:0", metadata={
                "backend_id": "isaac_lab_rsl_rl",
                "backend_execution_environment_manifest_hash": TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_MANIFEST_HASH,
            },
        )

    def phase(self, plan, phase, success=0.625):
        checkpoint = plan.execution_checkpoint().checksum
        seed_summary = {"status": "pass", "aggregate": {
            "success_rate": success, "episodes": self.config.eval_num_envs,
            "failure_breakdown": {"recovered": success, "fall": 1 - success},
        }, "secondary_metrics": {
            "mean_action_l2": 0.1, "mean_action_rate_l2": 0.1,
            "mean_min_base_height_m": 0.7, "mean_max_abs_pitch_rad": 0.1,
            "mean_max_abs_roll_rad": 0.1, "mean_final_xy_drift_m": 0.1,
            "mean_final_planar_speed_mps": 0.1, "recovery_window_success_rate": success,
        }}
        return EvaluationPhaseResult(
            job_id=plan.job_id + "-" + phase.value, task_id=plan.task_id, phase=phase,
            evaluation_plan_hash=canonical_json_hash(plan.to_public_dict()), subject=plan.subject,
            subject_manifest_hash=plan.subject_manifest_hash, program_source_checksum=plan.program_source_checksum,
            training_binding_hash=plan.training_binding_hash, capability_profile_hash=plan.capability_profile_hash,
            checkpoint_hash=checkpoint, eval_protocol_hash=plan.evaluation_protocol().protocol_hash,
            evaluation_binding_hash=plan.evaluation_binding.binding_hash,
            execution_device=plan.execution_device, resource_binding_hash=None,
            task_evidence={"schema_version": PUSH_PHASE_EVIDENCE_SCHEMA, "phase": phase.value,
                "seed_summaries": [copy.deepcopy(seed_summary) for _ in range(self.adapter.seed_count(phase))]},
            execution_cost={"schema_version": "test.evaluation_cost.v1", "clock_source": "perf_counter_ns",
                "measurement_scope": "rollout", "splits": {phase.value: {"wall_duration_ns": 100}}},
            environment_source_preflight={"status": "pass"},
            backend_execution_environment_verification=TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_VERIFICATION,
            checkpoint_safety_report={
                "schema_version": "true_rl_agent.checkpoint_safety_report.v1", "status": "pass",
                "checkpoint_hash": checkpoint, "expected_checkpoint_hash": checkpoint, "file_exists": True,
                "byte_read_status": "pass", "weights_only_load_status": "pass", "checkpoint_hash_match_status": "pass",
                "loader_error_class": None, "loader_error_summary": None, "reason": "checkpoint_safe_to_load",
            },
        )

    def screen(self, plan, result, role="candidate"):
        train = None
        if role == "candidate":
            fields = {name: getattr(plan, name) for name in (
                "subject", "subject_manifest_hash", "program_source_checksum", "training_binding_hash",
                "capability_profile_hash", "execution_device",
            )}
            metadata = {**fields, "training_subject": plan.subject.to_public_dict()}
            metadata.pop("subject")
            train = TrainJobResult(job_id="train-" + plan.subject.subject_id, **fields,
                                   checkpoint=replace(plan.checkpoint, metadata=metadata))
        return self.adapter.project_public_dev_screen(
            trial_id=plan.subject.subject_id, phase_result=result, screen_role=role, train_result=train,
            train_receipt_checksum="sha256:" + "e" * 64 if train else None, mechanism_operation_delta=None,
        )

    def receipts(self, plan, results):
        journal = TrialReceiptJournal(self.root / plan.job_id, contract={
            "schema_version": "true_rl_agent.trial_execution_contract.v1", "round_id": plan.subject.subject_id,
            "session_genesis_hash": "7" * 64, "ledger_epoch_hash": "8" * 64,
        })
        journal.bind_contract()
        return {r.phase: journal.publish_eval_phase(EvalPhaseJobSpec(
            job_id=r.job_id, evaluation_plan=plan, phase=r.phase), r).to_dict() for r in results}

    def test_real_push_projection_assembles_receipts_and_reference(self):
        for kind in ("reference", "candidate"):
            with self.subTest(kind=kind):
                plan = self.plan(ExperimentSubject(kind=kind, subject_id=kind))
                results = [self.phase(plan, phase) for phase in EvaluationPhase]
                arguments = dict(eval_plan=plan, phase_results=results, phase_receipt_refs=self.receipts(plan, results))
                if kind == "candidate":
                    sibling = self.plan(ExperimentSubject.candidate("sibling"))
                    guard = self.plan(ExperimentSubject.reference("parent"))
                    screens = [self.screen(plan, results[0]), self.screen(sibling, self.phase(sibling, EvaluationPhase.PUBLIC_DEV, 0.5))]
                    guard_screen = self.screen(guard, self.phase(guard, EvaluationPhase.PUBLIC_DEV), "guard_reference")
                    assembly = self.adapter.assemble_full_summary(**arguments, trial_id=kind,
                        selection_result=self.adapter.select(screens, guard_reference=guard_screen),
                        selection_screens=screens, guard_reference=guard_screen)
                else:
                    assembly = self.adapter.assemble_reference_full_summary(**arguments)
                    self.assertEqual(assembly, self.adapter.assemble_reference_full_summary(**arguments))
                actual = SubprocessRunner(allowed_executables=("python",)).project_assembled_eval_summary(plan, assembly)
                expected = build_push_public_result_from_summaries(
                    public_dev_summaries=results[0].task_evidence["seed_summaries"],
                    public_accept_summaries=results[1].task_evidence["seed_summaries"],
                    acceptance_protocol=self.config.acceptance_protocol(),
                    evaluation_protocol=self.config.evaluation_protocol(),
                )
                self.assertEqual(actual.public_score, expected.public_score)
                for key in ("public_dev", "public_accept", "motion_quality_gate", "behavior_quality_gate", "acceptance"):
                    self.assertEqual(actual.metrics[key], expected.metrics[key])
                self.assertEqual(actual.metrics["sealed_validation_gate"]["status"], "pass")

    def test_selector_uses_dev_behavior_and_stable_ties(self):
        guard = self.plan(ExperimentSubject.reference("parent"))
        guard_screen = self.screen(guard, self.phase(guard, EvaluationPhase.PUBLIC_DEV), "guard_reference")
        plans = [self.plan(ExperimentSubject.candidate(name)) for name in ("b", "a")]
        screens = [self.screen(plan, self.phase(plan, EvaluationPhase.PUBLIC_DEV)) for plan in plans]
        self.assertEqual(self.adapter.select(screens, guard_reference=guard_screen).selected_trial_id, "a")
        screens = []
        for plan in plans:
            result = self.phase(plan, EvaluationPhase.PUBLIC_DEV)
            evidence = copy.deepcopy(dict(result.task_evidence))
            for seed in evidence["seed_summaries"]:
                seed["secondary_metrics"]["mean_min_base_height_m"] = 0.1
            screens.append(self.screen(plan, replace(result, task_evidence=evidence)))
        self.assertIsNone(self.adapter.select(screens, guard_reference=guard_screen).selected_trial_id)
        with self.assertRaisesRegex(ValueError, "public-dev"):
            self.screen(plans[0], self.phase(plans[0], EvaluationPhase.PUBLIC_ACCEPT))
        with self.assertRaisesRegex(ValueError, "sealed"):
            self.adapter.project_public_phase_summary(phase_result=self.phase(plans[0], EvaluationPhase.SEALED_VALIDATION))

    def test_seed_shape_and_plan_identity_are_enforced(self):
        plan = self.plan(ExperimentSubject.reference("parent"))
        results = [self.phase(plan, phase) for phase in EvaluationPhase]
        evidence = copy.deepcopy(dict(results[0].task_evidence))
        evidence["seed_summaries"].pop()
        with self.assertRaisesRegex(ValueError, "seed count"):
            self.adapter.project_public_phase_summary(phase_result=replace(results[0], task_evidence=evidence))
        results[1] = replace(results[1], training_binding_hash="f" * 64)
        with self.assertRaisesRegex(ValueError, "evaluation plan"):
            self.adapter.assemble_reference_full_summary(eval_plan=plan, phase_results=results, phase_receipt_refs={})

    def test_runtime_phase_packet_preserves_rollouts_and_measures_executed_work(self):
        plan = self.plan(ExperimentSubject.reference("parent"))
        for phase in EvaluationPhase:
            result = self.phase(plan, phase)
            summaries = copy.deepcopy(result.task_evidence["seed_summaries"])
            for index, summary in enumerate(summaries):
                summary["execution_cost"] = {
                    "evaluation_env_count": 64,
                    "vector_env_step_calls": 20 + index,
                    "closed_first_episode_control_steps": 64 * 15,
                }
            packet = build_push_phase_summary(
                phase=phase, evaluation_plan_hash=result.evaluation_plan_hash,
                seed_summaries=summaries, expected_seed_count=len(summaries),
                wall_time_ns=2_500_000_000, execution_device="cuda:0",
                identity={"backend_execution_environment_verification":
                          TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_VERIFICATION},
                checkpoint_safety_report=result.checkpoint_safety_report,
                runtime_dynamics_hash="f" * 64,
            )
            parsed = SubprocessRunner(allowed_executables=("python",))._phase_result_from_summary(
                packet, eval_spec=EvalPhaseJobSpec(job_id=result.job_id,
                    evaluation_plan=plan, phase=phase), expected_checkpoint_hash=result.checkpoint_hash,
            )
            self.assertEqual(parsed.task_evidence["seed_summaries"], summaries)
            cost = parsed.execution_cost["splits"][phase.value]
            self.assertEqual(cost["wall_time_s"], 2.5)
            self.assertEqual(cost["simulated_env_control_steps"],
                             64 * sum(20 + i for i in range(len(summaries))))
            self.assertEqual(cost["closed_first_episode_control_steps"], 64 * 15 * len(summaries))
            self.assertNotIn("sealed_validation_gate", packet)

    def test_reference_cell_resolves_actual_program_and_qualifies_existing_task(self):
        self.assertEqual(self.plugin.task_spec.resolved_task_protocol_hash(),
                         "8a2ffa5f0702efa6f7a162723d349214c382772cf9a6b647cc7dfbd117be1ca2")
        provider = PushRecoveryCompositionProvider()
        values = {**provider.cli_defaults(), "cell_variant_id": PUSH_G1_29DOF_REFERENCE_CELL_VARIANT_ID}
        composition = resolve_task_composition(provider=provider, values=values,
            context=TaskCompositionContext(project_root=Path(__file__).resolve().parents[2],
                                           output_dir=self.root / "reference"))
        self.assertEqual(composition.genesis_anchor_kind, "reference_program")
        binding = GpuProcessBinding(host_execution_device="cuda:1",
            process_execution_device="cuda:0", resource_binding_hash="9" * 64)
        jobs = composition.backend_adapter.materialize_reference_jobs(
            session_id="push-reference-bound", task_spec=composition.task_plugin.task_spec,
            reference=composition.reference_program_artifact,
            artifact_root=self.root / "program-artifacts",
            policy_initialization=PolicyInitialization.random(),
            backend_execution_environment_manifest_hash=TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_MANIFEST_HASH,
            gpu_process_binding=binding,
        )
        bound_eval = replace(jobs.eval_spec, checkpoint_output_plan=None, checkpoint=CheckpointRef(
            uri=str(self.root / "reference.pt"), checksum="sha256:" + "d" * 64,
            role="eval", sealed=True))
        for spec in (jobs.train_spec, *(composition.backend_adapter.materialize_eval_phase(
            eval_plan=bound_eval, phase=phase) for phase in EvaluationPhase)):
            argv = list(spec.metadata["cli_argv"])
            if isinstance(spec, EvalPhaseJobSpec):
                # SubprocessRunner supplies the sealed checkpoint checksum at
                # launch, after the backend's plan materialization.
                argv += ["--checkpoint-sha256", spec.execution_checkpoint().checksum]
            # Exercise the real child parser against materialized production
            # argv; only the Isaac app argument registration is replaced.
            script = Path(argv[1])
            tree = ast.parse(script.read_text(encoding="utf-8"))
            parser_function = next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "_build_parser")
            namespace = {"argparse": argparse, "AppLauncher": _ParserAppLauncher,
                "__doc__": "parser contract", "DEFAULT_EVAL_PROTOCOL_VERSION": DEFAULT_EVAL_PROTOCOL_VERSION,
                "add_gpu_process_binding_arguments": add_gpu_process_binding_arguments,
                "add_resolved_cell_manifest_arguments": add_resolved_cell_manifest_arguments}
            exec(compile(ast.Module(body=[parser_function], type_ignores=[]), str(script), "exec"), namespace)
            args = namespace["_build_parser"]().parse_args(argv[2:])
            runtime = validate_gpu_process_binding_invocation(args, environment={
                "CUDA_VISIBLE_DEVICES": "GPU-00000000-0000-0000-0000-000000000001"})
            self.assertEqual(runtime.gpu_process_binding, binding)
            self.assertEqual(args.subject_kind, "reference")
            if isinstance(spec, EvalPhaseJobSpec):
                self.assertEqual(args.evaluation_phase, spec.phase.value)
                self.assertEqual(args.evaluation_plan_hash, spec.evaluation_plan_hash)
        self.plugin = composition.task_plugin
        self.config = self.plugin.config
        self.adapter = self.plugin.phased_evaluation_adapter()
        runtime = CandidateRuntimeAdapter.from_reference_program(
            composition.reference_program_artifact, num_envs=64, planned_training_steps=100)
        self.assertEqual(len(runtime.components.reward_terms), 0)
        plan = self.plan(ExperimentSubject.reference("parent"))
        results = [self.phase(plan, phase) for phase in EvaluationPhase]
        assembly = self.adapter.assemble_reference_full_summary(eval_plan=plan,
            phase_results=results, phase_receipt_refs=self.receipts(plan, results))
        public = SubprocessRunner(allowed_executables=("python",)).project_assembled_eval_summary(plan, assembly)
        budget = build_budget_accounting(budget_steps=self.config.budget_steps,
            num_envs=self.config.train_num_envs, num_steps_per_env=self.config.train_steps_per_iteration,
            max_iterations=self.config.train_max_iterations)
        qualified = self.plugin.qualify_reference_result(public_result=public,
            train_summary={"budget_accounting": budget})
        self.assertTrue(qualified.passed)
        self.assertNotEqual(self.plugin.task_spec.resolved_task_protocol_hash(),
                            PushRecoveryTaskPlugin().task_spec.resolved_task_protocol_hash())
        self.assertFalse(self.plugin.qualify_reference_result(public_result=public,
                                                             train_summary={}).passed)

    def test_exploration_reads_the_shared_public_phase_projection(self):
        from true_rl_agent.tasks.push_recovery.exploration import PushExplorationProjector
        result = self.phase(self.plan(ExperimentSubject.candidate("trial")), EvaluationPhase.PUBLIC_DEV)
        public_summary = self.adapter.project_public_phase_summary(phase_result=result)
        assessment = PushExplorationProjector(self.config).assess_trial(
            generation={}, trial={"public_dev": {"public_summary": public_summary}}, parent=None)
        self.assertEqual(assessment.public_evidence_summary["public_dev"]["mean"], 0.625)
        self.assertEqual(assessment.observed_effect, "unknown")


class _ParserAppLauncher:
    @staticmethod
    def add_app_launcher_args(parser):
        parser.add_argument("--headless", action="store_true")
        parser.add_argument("--device", default="cuda:0")
        parser.add_argument("--kit_args", default="")
