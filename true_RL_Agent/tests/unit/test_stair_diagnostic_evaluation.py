"""Independent public-dev identity without changing formal Stair evidence."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.unit import test_stair_entrypoint_materialization as entry_fixtures
from tests.unit import test_stair_phased_evaluation as phase_fixtures
from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.evaluation.phases import EvaluationPhase
from true_rl_agent.evaluation.subject import ExperimentSubject
from true_rl_agent.runner.job_argv import validate_eval_job_argv
from true_rl_agent.runner.specs import CheckpointRef
from true_rl_agent.tasks.stair_climbing.config import StairClimbingTaskConfig
from true_rl_agent.tasks.stair_climbing.diagnostic_evaluation import (
    prepare_stair_public_dev_diagnostic, project_stair_public_dev_diagnostic,
)
from true_rl_agent.tasks.stair_climbing.entrypoint_contract import validate_stair_eval_invocation
from true_rl_agent.tasks.stair_climbing.phased_evaluation import (
    assemble_stair_reference_eval_phase_results, build_stair_eval_phase_summary,
    canonical_stair_phase_result_summary,
)
from true_rl_agent.tasks.stair_climbing.seed_policy import (
    STAIR_CALIBRATION_EVAL_NAMESPACE, STAIR_FORMAL_EVAL_NAMESPACE,
    STAIR_PILOT_EVAL_NAMESPACE, STAIR_FINAL_CONFIRMATION_EVAL_NAMESPACE,
    STAIR_PUBLIC_DEV_DIAGNOSTIC_EVAL_NAMESPACE,
    diagnostic_seed_namespace, diagnostic_cohort_from_namespace,
    is_public_dev_diagnostic_namespace,
)

ROOT = Path(__file__).resolve().parents[2]
FORMAL_NAMESPACES = (STAIR_FORMAL_EVAL_NAMESPACE, STAIR_PILOT_EVAL_NAMESPACE,
                     STAIR_FINAL_CONFIRMATION_EVAL_NAMESPACE)
NEW_DIAGNOSTIC_COHORT = "n5_r2_20260930"
# Captured before adding diagnostics: hashes include every cohort hash and all
# split-local runner seeds for each historical formal/pilot/final namespace.
HISTORICAL_SEED_OUTPUTS = {
    "stair_protocol_v2.json": "8693e63b95e2b730c4aba0cdee21c0b36b24094ce24fd8ec28ae4e8dbd52d670",
    "stair_unified_protocol_v1.json": "53210c118e57b28dd4973c28c6b90f58c84ee867c2ad48eb3720835817a080c8",
    "stair_unified_protocol_v2.json": "cfe1f347ed8de6befb63cc9cac1f396d4171f88d2b58eaf8dbbb0fa28ec4eb5c",
    "stair_unified_protocol_v3.json": "dabf1eb7a1e86a59972dc85c2af31bfa8544928d26ac09bf6b56e21603f3dfef",
    "stair_go2_12dof_unified_protocol_v1.json": "ceecac0b45225b762120c349bfa1b0b0c53453ed7e52b7390cb70d96e57f8cd9",
    "stair_go2_12dof_unified_protocol_v2.json": "b6c88e8d3178fb028c9217c60e1aac7f617efcab7d944181f1c6430493cf78b9",
    "stair_go2_12dof_unified_protocol_v3.json": "f93f4a25604e4c4fa6b75c38bdaff6ea0a6344e9ce01be21ee2e7da67a153641",
}


class StairDiagnosticEvaluationTests(unittest.TestCase):
    def test_diagnostic_namespace_round_trip_preserves_default(self):
        self.assertEqual(diagnostic_seed_namespace(), STAIR_PUBLIC_DEV_DIAGNOSTIC_EVAL_NAMESPACE)
        for cohort in ("v1", NEW_DIAGNOSTIC_COHORT):
            namespace = diagnostic_seed_namespace(cohort)
            self.assertEqual(namespace, "public_dev_diagnostic_" + cohort)
            self.assertEqual(diagnostic_cohort_from_namespace(namespace), cohort)
            self.assertTrue(is_public_dev_diagnostic_namespace(namespace))
        for namespace in (STAIR_FORMAL_EVAL_NAMESPACE, "public_dev_diagnostic_", "public_dev_diagnostic_../x"):
            self.assertFalse(is_public_dev_diagnostic_namespace(namespace))
            with self.assertRaises(ValueError):
                diagnostic_cohort_from_namespace(namespace)

    def test_diagnostic_numeric_archive_finalizes_and_projects_after_archive_removal(self):
        from tests.unit.test_stair_numeric_capture import _TwoStepEnv
        from true_rl_agent.runner.specs import EvalPhaseJobSpec
        from true_rl_agent.tasks.stair_climbing.evaluation_runtime import run_stair_split_execution
        from true_rl_agent.tasks.stair_climbing.numeric_capture import StairNumericCapture
        from true_rl_agent.tasks.stair_climbing.phased_evaluation import finalize_stair_eval_phase_summary
        from true_rl_agent.tasks.stair_climbing.plugin import StairClimbingTaskPlugin
        from true_rl_agent.tasks.stair_climbing.robot_binding import resolve_stair_cell_manifest, stair_runtime_projection_from_manifest
        from true_rl_agent.tasks.stair_climbing.unified_scoring import build_unified_stair_summary

        config = StairClimbingTaskConfig.defaults(ROOT / "configs/tasks/stair_go2_12dof_unified_protocol_v3.json")
        manifest = resolve_stair_cell_manifest(cell_variant_id=config.cell_variant_id,
            config=config, plugin=StairClimbingTaskPlugin(config))
        config = replace(config, resolved_cell_task_protocol_hash=manifest.resolved_cell_task_protocol_hash)
        projection = stair_runtime_projection_from_manifest(manifest)
        fixture = phase_fixtures.StairPhasedEvaluationTests()
        fixture.config = config
        namespace = STAIR_PUBLIC_DEV_DIAGNOSTIC_EVAL_NAMESPACE
        formal = fixture._eval_plan("numeric-source")
        plan = replace(formal, job_id="numeric-diagnostic", evaluation_binding=replace(formal.evaluation_binding,
            seed_cohort_hash=config.seed_resolver().namespace_cohort_hash(namespace)))
        base = fixture._execution({EvaluationPhase.PUBLIC_DEV: [0] * config.public_seed_count}, seed_namespace=namespace)
        identity = {**plan.job_identity().to_public_dict(), "checkpoint_hash": base.checkpoint_hash,
            "resolved_cell_manifest_hash": manifest.manifest_hash,
            "evaluation_plan_hash": canonical_json_hash(plan.to_public_dict()),
            "evaluation_binding": plan.evaluation_binding.to_public_dict(), "seed_namespace": namespace,
            "splits": ["public_dev"], "environment_source_preflight": dict(base.source_manifest_preflight),
            "backend_execution_environment_verification": dict(base.backend_execution_environment_verification)}
        with tempfile.TemporaryDirectory() as temporary:
            capture = StairNumericCapture(Path(temporary) / "numeric", identity=identity,
                                          config=config, resolved_cell_manifest=manifest)
            execution = run_stair_split_execution(
                env=_TwoStepEnv(config.ordered_evaluation_suite.private_cell_assignment("public_dev"), projection, config),
                inference_policy=lambda value: value, protocol=None, semantics=config.evaluation_semantics,
                seed_resolver=config.seed_resolver(), split="public_dev", seed_count=config.public_seed_count,
                seed_namespace=namespace, runtime_projection=projection, numeric_capture=capture)
            summary = build_unified_stair_summary(suite=config.ordered_evaluation_suite,
                semantics=config.evaluation_semantics, split="public_dev", per_seed_outcomes=execution.per_seed_outcomes)
            reference = capture.complete(summary)
            raw = build_stair_eval_phase_summary(config=config, identity=plan.job_identity(),
                evaluation_plan_hash=identity["evaluation_plan_hash"], phase=EvaluationPhase.PUBLIC_DEV,
                execution=replace(base, summaries={"public_dev": summary},
                    per_seed_outcomes={"public_dev": execution.per_seed_outcomes}, numeric_evidence=reference),
                evaluation_binding=plan.evaluation_binding, checkpoint_safety_report=fixture._checkpoint_report(),
                expected_seed_namespace=namespace)
            self.assertEqual(raw["status"], "postprocessing")
            finished = finalize_stair_eval_phase_summary(EvalPhaseJobSpec(job_id=plan.job_id,
                evaluation_plan=plan, phase=EvaluationPhase.PUBLIC_DEV), raw, config=config, output_dir=Path(temporary))
            self.assertEqual(finished["status"], "pass")
            result = fixture._phase_result(plan=plan, phase=EvaluationPhase.PUBLIC_DEV,
                                          job_id=plan.job_id, summary=finished)
            with patch("true_rl_agent.tasks.stair_climbing.trajectory_production.measure_behavior",
                       side_effect=ArithmeticError("synthetic description failure")), self.assertLogs(
                       "true_rl_agent.tasks.stair_climbing.phased_evaluation", level="WARNING"):
                degraded = finalize_stair_eval_phase_summary(EvalPhaseJobSpec(job_id=plan.job_id,
                    evaluation_plan=plan, phase=EvaluationPhase.PUBLIC_DEV), raw, config=config,
                    output_dir=Path(temporary) / "unavailable")
            self.assertEqual(degraded["status"], "pass")
            self.assertEqual(degraded["task_evidence"]["summary"], raw["task_evidence"]["summary"])
            degraded_result = fixture._phase_result(plan=plan, phase=EvaluationPhase.PUBLIC_DEV,
                                                   job_id=plan.job_id, summary=degraded)
        public = project_stair_public_dev_diagnostic(config=config, eval_plan=plan, phase_result=result)
        self.assertEqual(public["behavior"], finished["task_evidence"]["behavior"])
        unavailable = project_stair_public_dev_diagnostic(config=config, eval_plan=plan, phase_result=degraded_result)
        self.assertEqual(unavailable["description_status"]["reason"], "description_computation_failed")
        self.assertNotIn("behavior", unavailable)
        self.assertEqual({k:v for k,v in unavailable.items() if k != "description_status"},
                         {k:v for k,v in public.items() if k != "behavior"})

    def test_old_outputs_unchanged_and_new_cohort_is_disjoint_public_dev_only(self):
        for name, expected in HISTORICAL_SEED_OUTPUTS.items():
            with self.subTest(protocol=name):
                config = StairClimbingTaskConfig.defaults(ROOT / "configs/tasks" / name)
                resolver = config.seed_resolver()
                historical = {namespace: {
                    "cohort_hash": resolver.namespace_cohort_hash(namespace),
                    "seeds": {split: [resolver.derive_seed(split=split, run_index=i, namespace=namespace)
                        for i in range(1, resolver._split_seed_count(split) + 1)]
                        for split in ("public_dev", "public_accept", "sealed_validation")}}
                    for namespace in FORMAL_NAMESPACES}
                self.assertEqual(canonical_json_hash(historical), expected)
                self.assertEqual(resolver.base_seed, 170100)
                if not config.is_unified_protocol:
                    continue
                namespace = STAIR_PUBLIC_DEV_DIAGNOSTIC_EVAL_NAMESPACE
                fresh = [resolver.derive_seed(split="public_dev", run_index=i, namespace=namespace)
                         for i in range(1, config.public_seed_count + 1)]
                old = {seed for record in historical.values() for seeds in record["seeds"].values() for seed in seeds}
                old.update(resolver.derive_seed(split="calibration", run_index=i,
                    namespace=STAIR_CALIBRATION_EVAL_NAMESPACE)
                    for i in range(1, resolver._split_seed_count("calibration") + 1))
                self.assertEqual(len(fresh), len(set(fresh)))
                self.assertFalse(set(fresh) & old)
                self.assertNotIn(resolver.namespace_cohort_hash(namespace),
                                 [row["cohort_hash"] for row in historical.values()])
                for split in ("public_accept", "sealed_validation", "calibration"):
                    with self.assertRaisesRegex(ValueError, "namespace"):
                        resolver.derive_seed(split=split, run_index=1, namespace=namespace)
                # The new commitment is split-local, not an all-split acceptance cohort.
                self.assertEqual(resolver.namespace_cohort_hash(namespace), canonical_json_hash({
                    "schema_version": "true_rl_agent.task.stair_climbing.ordered_seed_namespace_cohort.v1",
                    "resolver_version": resolver.resolver_version,
                    "suite_hash": resolver.evaluation_suite.source_suite_hash,
                    "task_protocol_hash": resolver.task_protocol_hash,
                    "namespace": namespace,
                    "split_cohort_hashes": {"public_dev": resolver.evaluation_suite.cohort_hash("public_dev")},
                }))
                new_namespace = diagnostic_seed_namespace(NEW_DIAGNOSTIC_COHORT)
                new_seeds = [resolver.derive_seed(split="public_dev", run_index=i, namespace=new_namespace)
                             for i in range(1, config.public_seed_count + 1)]
                if name == "stair_go2_12dof_unified_protocol_v3.json":
                    self.assertEqual(fresh, [1988007583, 1057377687, 1542212762])
                    self.assertEqual(new_seeds, [1945657272, 1544553419, 1187410])
                self.assertEqual(len(new_seeds), len(set(new_seeds)))
                self.assertFalse(set(new_seeds) & (old | set(fresh)))
                self.assertNotEqual(resolver.namespace_cohort_hash(new_namespace),
                                    resolver.namespace_cohort_hash(namespace))
                for split in ("public_accept", "sealed_validation", "calibration"):
                    with self.assertRaisesRegex(ValueError, "namespace"):
                        resolver.derive_seed(split=split, run_index=1, namespace=new_namespace)

    def test_new_cohort_materializes_and_entrypoint_rebuilds_binding(self):
        self.test_materialized_diagnostic_preserves_policy_and_entrypoint_rebuilds_binding(NEW_DIAGNOSTIC_COHORT)

    def test_materialized_diagnostic_preserves_policy_and_entrypoint_rebuilds_binding(self, cohort="v1"):
        fixture = entry_fixtures.StairEntrypointMaterializationTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            composition, jobs = fixture._materialize(root / "formal", "diagnostic-source")
            checkpoint = root / "checkpoint.pt"
            checkpoint.write_bytes(b"already-trained-policy-fixture")
            plan = replace(jobs.eval_spec, checkpoint_output_plan=None,
                checkpoint=CheckpointRef(uri=str(checkpoint),
                    checksum=entry_fixtures.file_sha256(checkpoint), role="eval", sealed=True))
            before = deepcopy(plan.to_public_dict())
            cohort_options = {} if cohort == "v1" else {"diagnostic_cohort": cohort}
            phase = prepare_stair_public_dev_diagnostic(config=composition.task_plugin.config,
                backend_adapter=composition.backend_adapter, formal_eval_plan=plan,
                job_id="independent-diagnostic", round_id="diagnostic",
                summary_path=str(root / "diagnostic/summary.json"), **cohort_options)
            self.assertEqual(before, plan.to_public_dict())
            self.assertEqual(phase.phase, EvaluationPhase.PUBLIC_DEV)
            for name in ("subject", "subject_manifest_hash", "program_source_checksum", "training_binding_hash",
                         "capability_profile_hash", "checkpoint", "execution_device"):
                self.assertEqual(getattr(phase.evaluation_plan, name), getattr(plan, name))
            self.assertEqual(phase.evaluation_plan.evaluation_protocol(), plan.evaluation_protocol())
            self.assertEqual(phase.evaluation_binding.acceptance_protocol_hash, plan.evaluation_binding.acceptance_protocol_hash)
            self.assertEqual(phase.evaluation_binding.task_protocol_hash, plan.evaluation_binding.task_protocol_hash)
            self.assertNotEqual(phase.evaluation_binding.binding_hash, plan.evaluation_binding.binding_hash)
            self.assertNotEqual(phase.evaluation_plan_hash, canonical_json_hash(plan.to_public_dict()))
            validate_eval_job_argv(phase, tuple(phase.metadata["cli_argv"]))
            self.assertEqual("--diagnostic-cohort" in phase.metadata["cli_argv"], cohort != "v1")
            args = entry_fixtures.build_eval_parser(entry_fixtures._FakeAppLauncher).parse_args(
                phase.metadata["cli_argv"][2:])
            kwargs = dict(project_root=ROOT, resolved_cell_manifest=composition.resolved_cell_manifest)
            context = validate_stair_eval_invocation(args, **kwargs)
            self.assertEqual(context.evaluation_binding, phase.evaluation_binding)
            self.assertEqual(context.seed_namespace, diagnostic_seed_namespace(cohort))
            self.assertEqual(context.config.seed_resolver().base_seed, 170100)
            self.assertIsNotNone(phase.metadata["numeric_evidence_dir"])
            if cohort != "v1":
                changed = deepcopy(args)
                changed.stair_eval_purpose = "formal_generation"
                with self.assertRaisesRegex(ValueError, "diagnostic cohort requires"):
                    validate_stair_eval_invocation(changed, **kwargs)
            for other in (None, "public_accept", "sealed_validation"):
                changed = deepcopy(args)
                changed.evaluation_phase = other
                with self.assertRaisesRegex(ValueError, "requires unified public_dev"):
                    validate_stair_eval_invocation(changed, **kwargs)
            changed = deepcopy(args)
            changed.seed += 1
            with self.assertRaisesRegex(ValueError, "seed mismatch"):
                validate_stair_eval_invocation(changed, **kwargs)
            changed = deepcopy(args)
            changed.seed_cohort_hash = plan.evaluation_binding.seed_cohort_hash
            changed.evaluation_binding_hash = plan.evaluation_binding.binding_hash
            with self.assertRaisesRegex(ValueError, "binding differs"):
                validate_stair_eval_invocation(changed, **kwargs)

    def test_new_cohort_projection_requires_explicit_cohort_and_rejects_formal_assembly(self):
        self.test_diagnostic_phase_projection_matches_new_plan_and_cannot_assemble_formal_evidence(NEW_DIAGNOSTIC_COHORT)

    def test_diagnostic_phase_projection_matches_new_plan_and_cannot_assemble_formal_evidence(self, cohort="v1"):
        fixture = phase_fixtures.StairPhasedEvaluationTests()
        fixture.config = StairClimbingTaskConfig.defaults(ROOT / "configs/tasks/stair_go2_12dof_unified_protocol_v3.json")
        config = fixture.config
        formal = replace(fixture._eval_plan("source"), subject=ExperimentSubject.reference("source-reference"))
        namespace = diagnostic_seed_namespace(cohort)
        diagnostic = replace(formal, job_id="diagnostic",
            evaluation_binding=replace(formal.evaluation_binding,
                seed_cohort_hash=config.seed_resolver().namespace_cohort_hash(namespace)))
        execution = fixture._execution({EvaluationPhase.PUBLIC_DEV: [1] * config.public_seed_count},
                                       seed_namespace=namespace)
        summary = build_stair_eval_phase_summary(config=config,
            identity=replace(diagnostic.job_identity(), job_id="diagnostic-public-dev"),
            evaluation_plan_hash=canonical_json_hash(diagnostic.to_public_dict()),
            phase=EvaluationPhase.PUBLIC_DEV, execution=execution,
            evaluation_binding=diagnostic.evaluation_binding,
            checkpoint_safety_report=fixture._checkpoint_report(), expected_seed_namespace=namespace)
        result = fixture._phase_result(plan=diagnostic, phase=EvaluationPhase.PUBLIC_DEV,
                                       job_id="diagnostic-public-dev", summary=summary)
        if cohort != "v1":
            with self.assertRaisesRegex(ValueError, "requires its public_dev domain"):
                project_stair_public_dev_diagnostic(config=config, eval_plan=diagnostic, phase_result=result)
        cohort_options = {} if cohort == "v1" else {"diagnostic_cohort": cohort}
        projected = project_stair_public_dev_diagnostic(config=config, eval_plan=diagnostic,
                                                       phase_result=result, **cohort_options)
        self.assertEqual(projected, summary["task_evidence"]["summary"])
        with self.assertRaisesRegex(ValueError, "phase result differs"):
            project_stair_public_dev_diagnostic(config=config, eval_plan=diagnostic,
                phase_result=replace(result, training_binding_hash="f" * 64), **cohort_options)
        with self.assertRaisesRegex(ValueError, "execution identity binding"):
            canonical_stair_phase_result_summary(config=config, expected_phase=EvaluationPhase.PUBLIC_DEV, result=result)
        # Keeping the old formal plan rejects mixing the fresh diagnostic with
        # old qualification phases, before any completeness/score decision.
        with self.assertRaisesRegex(ValueError, "phase result differs"):
            assemble_stair_reference_eval_phase_results(config=config, eval_plan=formal,
                phase_results=[result], phase_receipt_refs={})
        with self.assertRaisesRegex(ValueError, "active task contract"):
            assemble_stair_reference_eval_phase_results(config=config, eval_plan=diagnostic,
                phase_results=[result], phase_receipt_refs={})


if __name__ == "__main__":
    unittest.main()
