from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from scripts.isaaclab_stair_rsl_rl_eval import _optional_evaluation_phase
from tests.fakes.backend_environment import (
    TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_MANIFEST_HASH,
    TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_VERIFICATION,
)
from tests.fakes.fake_components import FakeTaskPlugin
from true_rl_agent.agent_core.trial_receipts import (
    TrialReceiptJournal,
    load_trial_eval_phase_receipt,
)
from true_rl_agent.agent_core.session_genesis import VerifiedSessionGenesis
from true_rl_agent.backends.base import BackendAdapter
from true_rl_agent.evaluation.binding import EvaluationBindingV1, canonical_json_hash
from true_rl_agent.evaluation.current_best import CurrentBestEvalStats
from true_rl_agent.evaluation.phases import (
    EvaluationAssembly,
    EvaluationPhase,
    EvaluationPhaseResult,
)
from true_rl_agent.evaluation.subject import ExperimentSubject
from true_rl_agent.runner.job_attempts import JobAttemptScope
from true_rl_agent.runner.specs import (
    CheckpointRef,
    EvalJobSpec,
    EvalPhaseJobSpec,
    GpuProcessBinding,
)
from true_rl_agent.runner.base import Runner
from true_rl_agent.runner.subprocess_runner import SubprocessRunner
from true_rl_agent.tasks.stair_climbing.config import (
    STAIR_TASK_ID,
    StairClimbingTaskConfig,
)
from true_rl_agent.tasks.stair_climbing.episode_evidence import StairEpisodeOutcome
from true_rl_agent.tasks.stair_climbing.evaluation_contract import (
    build_stair_evaluation_summary,
)
from true_rl_agent.tasks.stair_climbing.isaaclab_execution import (
    STAIR_EVALUATION_EXECUTION_COST_SCHEMA_VERSION,
    StairEvaluationExecutionResult,
)
from true_rl_agent.tasks.stair_climbing.phased_evaluation import (
    assemble_stair_eval_phase_results,
    build_stair_eval_phase_summary,
    build_stair_full_eval_summary,
)
from true_rl_agent.tasks.stair_climbing.phased_adapter import (
    StairPhasedEvaluationAdapter,
)
from true_rl_agent.tasks.stair_climbing.generation_bootstrap import (
    run_genesis_parent_public_dev,
)
from true_rl_agent.tasks.stair_climbing.phased_cost import (
    STAIR_PHASED_EVAL_COST_SCHEMA_VERSION,
    stair_phased_evaluation_cost_contract,
)
from true_rl_agent.tasks.stair_climbing.plugin import StairClimbingTaskPlugin
from true_rl_agent.tasks.stair_climbing.seed_policy import (
    STAIR_FORMAL_EVAL_NAMESPACE,
)
from true_rl_agent.tasks.stair_climbing.selection import (
    StairPublicDevScreenEvidence,
    StairSelectionResult,
    build_stair_public_dev_screen_evidence,
    select_stair_public_dev_winner,
)
from true_rl_agent.tasks.stair_climbing.training_distribution import (
    STAIR_EXECUTION_SOURCE_PREFLIGHT_SCHEMA_VERSION,
)


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_HASH = "sha256:" + "1" * 64
SUBJECT_MANIFEST_HASH = "2" * 64
PROGRAM_SOURCE_CHECKSUM = "sha256:" + "3" * 64
TRAINING_BINDING_HASH = "4" * 64
CAPABILITY_PROFILE_HASH = "5" * 64


class StairPhasedEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = StairClimbingTaskConfig.defaults(
            ROOT / "configs" / "tasks" / "stair_protocol_v2.json"
        )

    def test_generation_cost_and_genesis_guard_bootstrap_are_separate(self) -> None:
        cost = stair_phased_evaluation_cost_contract(self.config)
        self.assertEqual(
            cost["schema_version"], STAIR_PHASED_EVAL_COST_SCHEMA_VERSION
        )
        self.assertEqual(cost["max_seed_units_per_generation"], 23)
        self.assertEqual(cost["max_session_bootstrap_seed_units"], 3)

        plan = replace(
            self._eval_plan("genesis-reference"),
            subject=ExperimentSubject.reference("genesis-reference"),
        )
        result = self._phase_result_for(
            plan=plan,
            phase=EvaluationPhase.PUBLIC_DEV,
            successes=[4, 4, 4],
        )
        phase_spec = EvalPhaseJobSpec(
            job_id=result.job_id,
            evaluation_plan=plan,
            phase=EvaluationPhase.PUBLIC_DEV,
        )
        parent = Mock(spec=CurrentBestEvalStats)
        parent.anchor_kind = "reference"
        parent.anchor_id = "genesis-reference"
        parent.checkpoint_hash = result.checkpoint_hash
        parent.training_binding_hash = result.training_binding_hash
        parent.capability_profile_hash = result.capability_profile_hash
        parent.evaluation_binding_hash = result.evaluation_binding_hash
        parent.to_public_dict.return_value = {
            "mode": "reference",
            "anchor_kind": "reference",
            "anchor_id": "genesis-reference",
        }
        genesis = Mock(spec=VerifiedSessionGenesis)
        genesis.reference_eval_plan.return_value = plan
        genesis.session_genesis_hash = "7" * 64
        genesis.evaluation_binding = plan.evaluation_binding
        genesis.initial_current_best = parent
        backend = Mock(spec=BackendAdapter)
        backend.materialize_eval_phase.return_value = phase_spec
        runner = Mock(spec=Runner)
        runner.run_eval_phase.return_value = result
        adapter = StairPhasedEvaluationAdapter(self.config)

        with tempfile.TemporaryDirectory() as tmp:
            first = run_genesis_parent_public_dev(
                genesis=genesis,
                adapter=adapter,
                backend=backend,
                runner=runner,
                receipt_root=Path(tmp),
                generation_session_contract_hash="8" * 64,
            )
            second = run_genesis_parent_public_dev(
                genesis=genesis,
                adapter=adapter,
                backend=backend,
                runner=runner,
                receipt_root=Path(tmp),
                generation_session_contract_hash="8" * 64,
            )

        self.assertEqual(first, second)
        self.assertEqual(first.evidence.subject, plan.subject)
        self.assertEqual(runner.run_eval_phase.call_count, 1)
        self.assertEqual(backend.materialize_eval_phase.call_count, 2)

    def test_three_phase_assembly_matches_monolithic_scientific_summary(self) -> None:
        plan = self._eval_plan("candidate-phased")
        split_successes = {
            EvaluationPhase.PUBLIC_DEV: [5, 6, 5],
            EvaluationPhase.PUBLIC_ACCEPT: [6, 5, 6, 5, 6],
            EvaluationPhase.SEALED_VALIDATION: [5, 5, 6, 5, 6, 5, 6],
        }
        full_execution = self._execution(split_successes)
        checkpoint_report = self._checkpoint_report()
        full_summary = build_stair_full_eval_summary(
            config=self.config,
            identity=plan.job_identity(),
            evaluation_binding=plan.evaluation_binding,
            execution=full_execution,
            checkpoint_safety_report=checkpoint_report,
            subject_manifest_hash=plan.subject_manifest_hash,
            program_source_checksum=plan.program_source_checksum,
            training_binding_hash=plan.training_binding_hash,
            capability_profile_hash=plan.capability_profile_hash,
        )

        phase_results = []
        for phase, successes in split_successes.items():
            phase_execution = self._execution({phase: successes})
            summary = build_stair_eval_phase_summary(
                config=self.config,
                identity=replace(plan.job_identity(), job_id=f"phase-{phase.value}"),
                evaluation_plan_hash=canonical_json_hash(plan.to_public_dict()),
                phase=phase,
                execution=phase_execution,
                evaluation_binding=plan.evaluation_binding,
                checkpoint_safety_report=checkpoint_report,
            )
            phase_results.append(
                self._phase_result(
                    plan=plan,
                    phase=phase,
                    job_id=f"phase-{phase.value}",
                    summary=summary,
                )
            )

        selection, selection_screens, guard_reference = self._selection_for_result(
            trial_id="trial-winner",
            result=phase_results[0],
        )
        with tempfile.TemporaryDirectory() as tmp:
            phase_receipts = self._publish_phase_receipts(
                root=Path(tmp),
                plan=plan,
                phase_results=phase_results,
            )
            assembled = assemble_stair_eval_phase_results(
                config=self.config,
                eval_plan=plan,
                trial_id="trial-winner",
                selection_result=selection,
                selection_screens=selection_screens,
                guard_reference=guard_reference,
                phase_results=phase_results,
                phase_receipt_refs=phase_receipts,
            )
            self.assertIsInstance(assembled, EvaluationAssembly)
            self.assertEqual(assembled.summary, full_summary)

            runner = SubprocessRunner(allowed_executables=("python",))
            with self.assertRaisesRegex(TypeError, "EvaluationAssembly"):
                runner.project_assembled_eval_summary(plan, full_summary)
            assembled_result = runner.project_assembled_eval_summary(plan, assembled)
            tampered_binding = replace(
                assembled.phase_receipts[0],
                checksum="sha256:" + "f" * 64,
            )
            tampered_assembly = replace(
                assembled,
                phase_receipts=(
                    tampered_binding,
                    *assembled.phase_receipts[1:],
                ),
                assembly_hash=None,
            )
            with self.assertRaisesRegex(
                Exception,
                "receipt checksum mismatch",
            ):
                runner.project_assembled_eval_summary(plan, tampered_assembly)
        self.assertEqual(assembled_result.public_score, full_summary["public_score"])
        self.assertEqual(
            assembled.summary["sealed_validation_evidence"]["task_evidence_hash"],
            full_summary["sealed_validation_evidence"]["task_evidence_hash"],
        )

    def test_selector_uses_public_dev_only_and_can_return_no_selection(self) -> None:
        guard_reference = self._screen_with_scores(
            "parent", objective=[0.5] * 3, guard=[0.8] * 3
        )
        left = self._screen_with_scores(
            "trial-a", objective=[0.7] * 3, guard=[0.8] * 3
        )
        right = self._screen_with_scores(
            "trial-b", objective=[0.2, 0.3, 0.2], guard=[0.8] * 3
        )
        selected = select_stair_public_dev_winner(
            config=self.config,
            screens=[right, left],
            guard_reference=guard_reference,
        )
        self.assertEqual(selected.selected_trial_id, "trial-a")
        self.assertEqual(selected.reason, "public_dev_objective_rank")
        self.assertEqual(
            selected.ranking_confidence,
            "outside_estimated_noise_band",
        )
        self.assertFalse(selected.to_public_dict()["acceptance_authority"])
        self.assertEqual(
            StairSelectionResult.from_mapping(selected.to_public_dict()),
            selected,
        )

        tie = select_stair_public_dev_winner(
            config=self.config,
            screens=[
                self._screen_with_scores(
                    "trial-c", objective=[0.5] * 3, guard=[0.8] * 3
                ),
                self._screen_with_scores(
                    "trial-d", objective=[0.5] * 3, guard=[0.8] * 3
                ),
            ],
            guard_reference=guard_reference,
        )
        self.assertIsNone(tie.selected_trial_id)
        self.assertEqual(tie.reason, "inconclusive_exact_tie")

    def test_selector_ranks_tiny_gaps_and_filters_clear_guard_regression(self) -> None:
        parent = self._screen_with_scores(
            "parent", objective=[0.5] * 3, guard=[0.8] * 3
        )
        tiny_lead = self._screen_with_scores(
            "tiny-lead",
            objective=[0.500, 0.600, 0.700],
            guard=[0.8] * 3,
        )
        tiny_lag = self._screen_with_scores(
            "tiny-lag",
            objective=[0.499, 0.599, 0.699],
            guard=[0.8] * 3,
        )
        ranked = select_stair_public_dev_winner(
            config=self.config,
            screens=[tiny_lag, tiny_lead],
            guard_reference=parent,
        )
        self.assertEqual(ranked.selected_trial_id, "tiny-lead")
        self.assertEqual(ranked.reason, "public_dev_objective_rank")
        self.assertEqual(
            ranked.ranking_confidence,
            "deterministic_paired_gap",
        )
        noisy_lead = self._screen_with_scores(
            "noisy-lead",
            objective=[0.510, 0.500, 0.495],
            guard=[0.8] * 3,
        )
        noisy_lag = self._screen_with_scores(
            "noisy-lag",
            objective=[0.500, 0.500, 0.500],
            guard=[0.8] * 3,
        )
        within_noise = select_stair_public_dev_winner(
            config=self.config,
            screens=[noisy_lag, noisy_lead],
            guard_reference=parent,
        )
        self.assertEqual(within_noise.selected_trial_id, "noisy-lead")
        self.assertEqual(
            within_noise.ranking_confidence,
            "within_estimated_noise_band",
        )

        unsafe = self._screen_with_scores(
            "unsafe", objective=[0.9] * 3, guard=[0.74] * 3
        )
        safe = self._screen_with_scores(
            "safe", objective=[0.7] * 3, guard=[0.8] * 3
        )
        guard_filtered = select_stair_public_dev_winner(
            config=self.config,
            screens=[unsafe, safe],
            guard_reference=parent,
        )
        self.assertEqual(guard_filtered.selected_trial_id, "safe")
        self.assertEqual(guard_filtered.reason, "public_dev_guard_filter")
        self.assertEqual(guard_filtered.guard_eligible_trial_ids, ("safe",))

        both_unsafe = select_stair_public_dev_winner(
            config=self.config,
            screens=[
                unsafe,
                self._screen_with_scores(
                    "also-unsafe",
                    objective=[0.8] * 3,
                    guard=[0.70] * 3,
                ),
            ],
            guard_reference=parent,
        )
        self.assertIsNone(both_unsafe.selected_trial_id)
        self.assertEqual(both_unsafe.reason, "no_guard_eligible_candidate")

    def test_selector_records_declared_cross_gpu_execution_differences(self) -> None:
        parent = self._screen_with_scores(
            "parent", objective=[0.5] * 3, guard=[0.8] * 3
        )
        left = self._screen_with_scores(
            "left",
            objective=[0.7] * 3,
            guard=[0.8] * 3,
            execution_device="cuda:0",
            resource_binding_hash="a" * 64,
        )
        right = self._screen_with_scores(
            "right",
            objective=[0.6] * 3,
            guard=[0.8] * 3,
            execution_device="cuda:1",
            resource_binding_hash="b" * 64,
        )
        selected = select_stair_public_dev_winner(
            config=self.config,
            screens=[right, left],
            guard_reference=parent,
        )
        self.assertEqual(selected.selected_trial_id, "left")
        self.assertEqual(left.to_selector_dict()["execution_device"], "cuda:0")
        self.assertEqual(right.to_selector_dict()["execution_device"], "cuda:1")
        self.assertNotIn(left.execution_device, left.comparison_domain)
        self.assertNotIn(left.resource_binding_hash, left.comparison_domain)

    def test_phase_binding_and_non_public_dev_selector_inputs_fail_closed(self) -> None:
        plan = self._eval_plan("candidate-binding")
        result = self._phase_result_for(
            plan=plan,
            phase=EvaluationPhase.PUBLIC_ACCEPT,
            successes=[4] * self.config.public_accept_seed_count,
        )
        with self.assertRaisesRegex(ValueError, "public-dev"):
            build_stair_public_dev_screen_evidence(
                config=self.config,
                trial_id="trial-accept",
                phase_result=result,
            )

        phases = [
            self._phase_result_for(
                plan=plan,
                phase=phase,
                successes=[4] * self._seed_count(phase),
            )
            for phase in EvaluationPhase
        ]
        loser_screen = build_stair_public_dev_screen_evidence(
            config=self.config,
            trial_id="trial-loser",
            phase_result=phases[0],
        )
        loser_screens = [
            loser_screen,
            self._screen("trial-better", [7, 7, 7]),
        ]
        loser_guard_reference = self._screen("trial-parent", [4, 4, 4])
        loser_selection = select_stair_public_dev_winner(
            config=self.config,
            screens=loser_screens,
            guard_reference=loser_guard_reference,
        )
        self.assertEqual(loser_selection.selected_trial_id, "trial-better")
        with self.assertRaisesRegex(ValueError, "not bound to the selected trial"):
            assemble_stair_eval_phase_results(
                config=self.config,
                eval_plan=plan,
                trial_id="trial-loser",
                selection_result=loser_selection,
                selection_screens=loser_screens,
                guard_reference=loser_guard_reference,
                phase_results=phases,
                phase_receipt_refs={},
            )
        selection, selection_screens, guard_reference = self._selection_for_result(
            trial_id="trial-binding",
            result=phases[0],
        )
        phases[0] = replace(phases[0], checkpoint_hash="sha256:" + "9" * 64)
        with self.assertRaisesRegex(ValueError, "differs from its evaluation plan"):
            assemble_stair_eval_phase_results(
                config=self.config,
                eval_plan=plan,
                trial_id="trial-binding",
                selection_result=selection,
                selection_screens=selection_screens,
                guard_reference=guard_reference,
                phase_results=phases,
                phase_receipt_refs={},
            )

    def test_phase_cost_is_exactly_the_executed_split(self) -> None:
        plan = self._eval_plan("candidate-cost")
        result = self._phase_result_for(
            plan=plan,
            phase=EvaluationPhase.PUBLIC_DEV,
            successes=[4] * self.config.public_seed_count,
        )
        self.assertEqual(set(result.execution_cost["splits"]), {"public_dev"})
        self.assertEqual(
            result.execution_cost["splits"]["public_dev"]["seed_count"],
            3,
        )
        self.assertEqual(
            result.execution_cost["schema_version"],
            STAIR_EVALUATION_EXECUTION_COST_SCHEMA_VERSION,
        )

    def test_phase_cli_identity_is_all_or_nothing(self) -> None:
        self.assertIsNone(
            _optional_evaluation_phase(
                SimpleNamespace(
                    evaluation_phase=None,
                    evaluation_plan_hash=None,
                )
            )
        )
        self.assertEqual(
            _optional_evaluation_phase(
                SimpleNamespace(
                    evaluation_phase="public_dev",
                    evaluation_plan_hash="a" * 64,
                )
            ),
            EvaluationPhase.PUBLIC_DEV,
        )
        with self.assertRaisesRegex(ValueError, "supplied together"):
            _optional_evaluation_phase(
                SimpleNamespace(
                    evaluation_phase="public_dev",
                    evaluation_plan_hash=None,
                )
            )

    def test_stair_plugin_owns_the_optional_phase_capability(self) -> None:
        adapter = StairClimbingTaskPlugin(
            self.config
        ).phased_evaluation_adapter()
        self.assertIsInstance(adapter, StairPhasedEvaluationAdapter)
        self.assertEqual(
            adapter.selection_contract(),
            adapter.selection_contract(),
        )
        cost = adapter.cost_contract()
        self.assertEqual(cost["max_seed_units_per_generation"], 23)
        self.assertEqual(cost["max_episode_executions_per_generation"], 6440)
        self.assertEqual(
            cost["per_generation_phases"]["sealed_validation"]["max_jobs"],
            1,
        )
        self.assertNotIn("max_seed_units", cost)
        self.assertNotIn("phases", cost)
        self.assertIsNone(FakeTaskPlugin().phased_evaluation_adapter())

    def test_selector_requires_typed_public_dev_evidence(self) -> None:
        screen = self._screen("trial-a", [4, 4, 4])
        with self.assertRaisesRegex(ValueError, "two screen evidences"):
            select_stair_public_dev_winner(
                config=self.config,
                screens=[screen.to_selector_dict(), screen],
                guard_reference=screen,
            )
        sibling = self._screen("trial-b", [3, 3, 3])
        with self.assertRaisesRegex(ValueError, "frozen parent"):
            select_stair_public_dev_winner(
                config=self.config,
                screens=[screen, sibling],
                guard_reference=screen,
            )

    def test_runner_parser_and_phase_receipt_preserve_plan_binding(self) -> None:
        binding = GpuProcessBinding(
            host_execution_device="cuda:1",
            process_execution_device="cuda:0",
            resource_binding_hash="9" * 64,
        )
        plan = replace(
            self._eval_plan("candidate-receipt"),
            gpu_process_binding=binding,
        )
        phase = EvaluationPhase.PUBLIC_DEV
        execution = self._execution(
            {phase: [4] * self.config.public_seed_count}
        )
        phase_spec = EvalPhaseJobSpec(
            job_id=f"{plan.job_id}-public-dev",
            evaluation_plan=plan,
            phase=phase,
            attempt_scope=JobAttemptScope(
                journal_root=Path("unused-attempts"),
                session_genesis_hash="a" * 64,
                session_contract_hash="b" * 64,
                generation_id="generation-001",
                lane_id="lane-b",
                invocation_segment_id="segment-000",
                budget_contract_hash="c" * 64,
                attempt_budget_units=1,
                max_infrastructure_retries=0,
                resource_binding_hash=binding.resource_binding_hash,
            ),
        )
        summary = build_stair_eval_phase_summary(
            config=self.config,
            identity=phase_spec.job_identity(),
            evaluation_plan_hash=phase_spec.evaluation_plan_hash,
            phase=phase,
            execution=execution,
            evaluation_binding=plan.evaluation_binding,
            checkpoint_safety_report=self._checkpoint_report(),
        )
        runner = SubprocessRunner(allowed_executables=("python",))
        parsed = runner._phase_result_from_summary(  # noqa: SLF001
            summary,
            eval_spec=phase_spec,
            expected_checkpoint_hash=CHECKPOINT_HASH,
            actual_gpu_process_binding=binding,
        )

        with tempfile.TemporaryDirectory() as tmp:
            journal = TrialReceiptJournal(
                Path(tmp),
                contract={
                    "schema_version": "true_rl_agent.trial_execution_contract.v1",
                    "session_genesis_hash": "7" * 64,
                    "ledger_epoch_hash": "8" * 64,
                    "round_id": "round-phase-receipt",
                },
            )
            journal.bind_contract()
            ref = journal.publish_eval_phase(phase_spec, parsed)
            loaded = journal.load_eval_phase(phase_spec)
            standalone = load_trial_eval_phase_receipt(ref)

        self.assertEqual(ref.kind, "eval_phase")
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(loaded[0], parsed)
        self.assertEqual(loaded[0].gpu_process_binding, binding.to_dict())
        self.assertEqual(standalone.result, parsed)

        tampered = dict(summary)
        tampered["task_evidence_hash"] = "9" * 64
        with self.assertRaisesRegex(Exception, "task evidence hash mismatch"):
            runner._phase_result_from_summary(  # noqa: SLF001
                tampered,
                eval_spec=phase_spec,
                expected_checkpoint_hash=CHECKPOINT_HASH,
                actual_gpu_process_binding=binding,
            )

    def _screen(self, trial_id: str, successes: list[int]):
        plan = self._eval_plan(f"candidate-{trial_id}")
        result = self._phase_result_for(
            plan=plan,
            phase=EvaluationPhase.PUBLIC_DEV,
            successes=successes,
        )
        return build_stair_public_dev_screen_evidence(
            config=self.config,
            trial_id=trial_id,
            phase_result=result,
        )

    def _selection_for_result(
        self,
        *,
        trial_id: str,
        result: EvaluationPhaseResult,
    ) -> tuple[
        StairSelectionResult,
        list,
        StairPublicDevScreenEvidence,
    ]:
        selected = build_stair_public_dev_screen_evidence(
            config=self.config,
            trial_id=trial_id,
            phase_result=result,
        )
        sibling = self._screen(f"{trial_id}-sibling", [1, 1, 1])
        guard_reference = replace(
            self._screen(f"{trial_id}-parent", [4, 4, 4]),
            anchor_seed_macro_scores=selected.anchor_seed_macro_scores,
        )
        screens = [selected, sibling]
        selection = select_stair_public_dev_winner(
            config=self.config,
            screens=screens,
            guard_reference=guard_reference,
        )
        self.assertEqual(selection.selected_trial_id, trial_id)
        return selection, screens, guard_reference

    def _screen_with_scores(
        self,
        trial_id: str,
        *,
        objective: list[float],
        guard: list[float],
        execution_device: str = "cuda:0",
        resource_binding_hash: str | None = None,
    ):
        return replace(
            self._screen(trial_id, [4, 4, 4]),
            execution_device=execution_device,
            resource_binding_hash=resource_binding_hash,
            anchor_seed_macro_scores=tuple(guard),
            generalization_seed_macro_scores=tuple(objective),
        )

    def _publish_phase_receipts(
        self,
        *,
        root: Path,
        plan: EvalJobSpec,
        phase_results: list[EvaluationPhaseResult],
    ) -> dict[EvaluationPhase, dict]:
        journal = TrialReceiptJournal(
            root,
            contract={
                "schema_version": "true_rl_agent.trial_execution_contract.v1",
                "session_genesis_hash": "7" * 64,
                "ledger_epoch_hash": "8" * 64,
                "round_id": "round-phase-assembly",
            },
        )
        journal.bind_contract()
        refs = {}
        for result in phase_results:
            spec = EvalPhaseJobSpec(
                job_id=result.job_id,
                evaluation_plan=plan,
                phase=result.phase,
            )
            refs[result.phase] = journal.publish_eval_phase(spec, result).to_dict()
        return refs

    def _phase_result_for(
        self,
        *,
        plan: EvalJobSpec,
        phase: EvaluationPhase,
        successes: list[int],
    ) -> EvaluationPhaseResult:
        job_id = f"{plan.job_id}-{phase.value}"
        summary = build_stair_eval_phase_summary(
            config=self.config,
            identity=replace(plan.job_identity(), job_id=job_id),
            evaluation_plan_hash=canonical_json_hash(plan.to_public_dict()),
            phase=phase,
            execution=self._execution({phase: successes}),
            evaluation_binding=plan.evaluation_binding,
            checkpoint_safety_report=self._checkpoint_report(),
        )
        return self._phase_result(
            plan=plan,
            phase=phase,
            job_id=job_id,
            summary=summary,
        )

    def _phase_result(
        self,
        *,
        plan: EvalJobSpec,
        phase: EvaluationPhase,
        job_id: str,
        summary: dict,
    ) -> EvaluationPhaseResult:
        return EvaluationPhaseResult(
            job_id=job_id,
            task_id=plan.task_id,
            phase=phase,
            evaluation_plan_hash=canonical_json_hash(plan.to_public_dict()),
            subject=plan.subject,
            subject_manifest_hash=plan.subject_manifest_hash,
            program_source_checksum=plan.program_source_checksum,
            training_binding_hash=plan.training_binding_hash,
            capability_profile_hash=plan.capability_profile_hash,
            checkpoint_hash=CHECKPOINT_HASH,
            eval_protocol_hash=plan.evaluation_protocol().protocol_hash,
            evaluation_binding_hash=plan.evaluation_binding.binding_hash,
            execution_device=plan.execution_device,
            resource_binding_hash=None,
            task_evidence=summary["task_evidence"],
            execution_cost=summary["evaluation_execution_cost"],
            environment_source_preflight=summary["environment_source_preflight"],
            backend_execution_environment_verification=summary[
                "backend_execution_environment_verification"
            ],
            checkpoint_safety_report=summary["checkpoint_safety_report"],
        )

    def _eval_plan(self, candidate_id: str) -> EvalJobSpec:
        protocol = self.config.evaluation_protocol()
        binding = EvaluationBindingV1(
            eval_protocol_hash=protocol.protocol_hash,
            acceptance_protocol_hash=self.config.acceptance_protocol.protocol_hash,
            task_protocol_hash=self.config.task_protocol_hash,
            seed_cohort_hash=self.config.seed_resolver().namespace_cohort_hash(
                STAIR_FORMAL_EVAL_NAMESPACE
            ),
        )
        return EvalJobSpec(
            job_id=f"eval-{candidate_id}",
            task_id=STAIR_TASK_ID,
            subject=ExperimentSubject.candidate(candidate_id),
            subject_manifest_hash=SUBJECT_MANIFEST_HASH,
            program_source_checksum=PROGRAM_SOURCE_CHECKSUM,
            training_binding_hash=TRAINING_BINDING_HASH,
            capability_profile_hash=CAPABILITY_PROFILE_HASH,
            checkpoint=CheckpointRef(
                uri=f"artifact://{candidate_id}/best.pt",
                checksum=CHECKPOINT_HASH,
                role="eval",
                sealed=True,
            ),
            public_seed_count=self.config.public_seed_count,
            public_accept_seed_count=self.config.public_accept_seed_count,
            sealed_validation_seed_count=self.config.sealed_validation_seed_count,
            eval_num_envs=self.config.eval_num_envs,
            eval_protocol_version=self.config.eval_protocol_version,
            evaluation_binding=binding,
            execution_device="cuda:0",
            metadata={
                "backend_id": "isaac_lab_rsl_rl",
                "backend_execution_environment_manifest_hash": (
                    TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_MANIFEST_HASH
                )
            },
        )

    def _execution(
        self,
        split_successes: dict[EvaluationPhase, list[int]],
    ) -> StairEvaluationExecutionResult:
        summaries = {}
        outcomes = {}
        steps = {}
        walls = {}
        for phase, successes in split_successes.items():
            per_seed = tuple(
                tuple(self._outcomes(successes_per_cell=count))
                for count in successes
            )
            summaries[phase.value] = build_stair_evaluation_summary(
                self.config.protocol_v1,
                semantics=self.config.evaluation_semantics,
                evaluation_protocol=self.config.evaluation_protocol(),
                split=phase.value,
                per_seed_outcomes=per_seed,
                evaluation_profile=self.config.evaluation_profile,
            )
            outcomes[phase.value] = per_seed
            steps[phase.value] = 100 + len(successes)
            walls[phase.value] = float(len(successes))
        return StairEvaluationExecutionResult(
            summaries=summaries,
            per_seed_outcomes=outcomes,
            split_vector_env_step_calls=steps,
            split_wall_times_s=walls,
            checkpoint_hash=CHECKPOINT_HASH,
            seed_namespace=STAIR_FORMAL_EVAL_NAMESPACE,
            seed_cohort_hash=self.config.seed_resolver().namespace_cohort_hash(
                STAIR_FORMAL_EVAL_NAMESPACE
            ),
            source_manifest_preflight={
                "schema_version": STAIR_EXECUTION_SOURCE_PREFLIGHT_SCHEMA_VERSION,
                "status": "pass",
                "task_protocol_hash": self.config.task_protocol_hash,
                "source_manifest_hash": "6" * 64,
            },
            backend_execution_environment_verification=(
                TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_VERIFICATION
            ),
            execution_device="cuda:0",
        )

    def _outcomes(self, *, successes_per_cell: int) -> list[StairEpisodeOutcome]:
        outcomes = []
        for env_index, cell in enumerate(
            self.config.evaluation_profile.private_cell_assignment()
        ):
            success = env_index % self.config.evaluation_profile.episodes_per_seed_per_cell < successes_per_cell
            outcomes.append(
                StairEpisodeOutcome(
                    env_index=env_index,
                    bin_id=cell.bin_id,
                    episode_steps=cell.episode_horizon_control_steps,
                    primary_reason="success" if success else "timeout",
                    success=success,
                    diagnostic_flags=(
                        ("first_valid_top_arrival_achieved",)
                        if success
                        else ("goal_not_reached_detected",)
                    ),
                    max_progress_m=cell.progress_goal_m if success else 0.5,
                    progress_fraction=1.0 if success else 0.1,
                    active_terminal_terms=() if success else ("time_out",),
                    evaluation_cell_id=cell.cell_id,
                )
            )
        return outcomes

    def _checkpoint_report(self) -> dict[str, object]:
        return {
            "schema_version": "true_rl_agent.checkpoint_safety_report.v1",
            "status": "pass",
            "checkpoint_hash": CHECKPOINT_HASH,
            "expected_checkpoint_hash": CHECKPOINT_HASH,
            "file_exists": True,
            "byte_read_status": "pass",
            "weights_only_load_status": "pass",
            "checkpoint_hash_match_status": "pass",
            "loader_error_class": None,
            "loader_error_summary": None,
            "reason": "checkpoint_safe_to_load",
        }

    def _seed_count(self, phase: EvaluationPhase) -> int:
        return {
            EvaluationPhase.PUBLIC_DEV: self.config.public_seed_count,
            EvaluationPhase.PUBLIC_ACCEPT: self.config.public_accept_seed_count,
            EvaluationPhase.SEALED_VALIDATION: self.config.sealed_validation_seed_count,
        }[phase]


if __name__ == "__main__":
    unittest.main()
