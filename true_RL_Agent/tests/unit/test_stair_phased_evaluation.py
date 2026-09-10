from __future__ import annotations

import tempfile
import unittest
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from scripts.isaaclab_stair_rsl_rl_eval import _optional_evaluation_phase

from tests.fakes.backend_environment import (
    TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_MANIFEST_HASH,
    TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_VERIFICATION,
)
from tests.fakes.fake_components import FakeTaskPlugin
from true_rl_agent.agent_core.session_genesis import VerifiedSessionGenesis
from true_rl_agent.agent_core.trial_receipts import (
    TrialReceiptJournal,
    load_trial_eval_phase_receipt,
)
from true_rl_agent.backends.base import BackendAdapter
from true_rl_agent.evaluation.binding import EvaluationBindingV1, canonical_json_hash
from true_rl_agent.evaluation.current_best import CurrentBestEvalStats
from true_rl_agent.evaluation.derived_statistics import DERIVED_STATISTICS_VERSION
from true_rl_agent.evaluation.phases import (
    EvaluationAssembly,
    EvaluationPhase,
    EvaluationPhaseResult,
)
from true_rl_agent.evaluation.subject import ExperimentSubject
from true_rl_agent.runner.base import Runner
from true_rl_agent.runner.job_attempts import JobAttemptScope
from true_rl_agent.runner.specs import (
    CheckpointRef,
    EvalJobSpec,
    EvalPhaseJobSpec,
    GpuProcessBinding,
)
from true_rl_agent.runner.subprocess_runner import SubprocessRunner
from true_rl_agent.tasks.stair_climbing.config import (
    STAIR_TASK_ID,
    StairClimbingTaskConfig,
)
from true_rl_agent.tasks.stair_climbing.episode_evidence import StairEpisodeOutcome
from true_rl_agent.tasks.stair_climbing.evaluation_contract import (
    build_stair_evaluation_summary,
)
from true_rl_agent.agent_core.generation_guard import (
    GenerationGuardReference,
    replay_generation_guard_chain,
    run_genesis_parent_public_dev,
)
from true_rl_agent.tasks.stair_climbing.isaaclab_execution import (
    STAIR_EVALUATION_EXECUTION_COST_SCHEMA_VERSION,
    StairEvaluationExecutionResult,
)
from true_rl_agent.tasks.stair_climbing.phased_adapter import (
    StairPhasedEvaluationAdapter,
)
from true_rl_agent.tasks.stair_climbing.phased_cost import (
    STAIR_PHASED_EVAL_COST_SCHEMA_VERSION,
    stair_phased_evaluation_cost_contract,
)
from true_rl_agent.tasks.stair_climbing.phased_evaluation import (
    assemble_stair_eval_phase_results,
    assemble_stair_reference_eval_phase_results,
    build_stair_eval_phase_summary,
    build_stair_full_eval_summary,
    canonical_stair_phase_result_summary,
)
from true_rl_agent.tasks.stair_climbing.plugin import StairClimbingTaskPlugin
from true_rl_agent.tasks.stair_climbing.public_projection import (
    _validate_unified_sealed_score_verdict,
)
from true_rl_agent.tasks.stair_climbing.robot_binding import resolve_stair_cell_manifest
from true_rl_agent.tasks.stair_climbing.seed_policy import (
    STAIR_FINAL_CONFIRMATION_EVAL_NAMESPACE,
    STAIR_FORMAL_EVAL_NAMESPACE,
)
from true_rl_agent.tasks.stair_climbing.selection import (
    StairPublicDevScreenEvidence,
    StairSelectionResult,
    build_stair_public_dev_screen_evidence,
    select_stair_public_dev_winner,
)
from true_rl_agent.tasks.stair_climbing.unified_scoring import (
    build_unified_stair_summary,
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

    def test_real_v3_public_accept_projection_preserves_distinct_guards_and_paired_order(self):
        from true_rl_agent.tasks.stair_climbing.unified_scoring import build_reference_objective_family_mask

        counts = {"c0_id": 6, "c1_id": 9, "height_ood": 3, "tread_ood": 6,
                  "joint_ood": 9, "c1_conditional_frontier": 6, "c2_mechanism_ood": 3,
                  "phase_composition_ood": 10}
        for stem in ("stair", "stair_go2_12dof"):
            with self.subTest(robot=stem):
                self.config = StairClimbingTaskConfig.defaults(ROOT / f"configs/tasks/{stem}_unified_protocol_v3.json")
                suite = self.config.ordered_evaluation_suite
                self.assertEqual(suite.episodes_per_seed_per_cell, 12)
                reference = self._execution({EvaluationPhase.PUBLIC_ACCEPT: [6] * 5})
                mask = build_reference_objective_family_mask(suite=suite,
                    public_accept_summary=reference.summaries["public_accept"])
                adapter = StairPhasedEvaluationAdapter(self.config, objective_family_mask_provider=lambda: mask)
                original_outcomes = self._unified_outcomes

                def outcomes(*, split, successes_per_cell):
                    occurrences = {}
                    result = []
                    for item in original_outcomes(split=split, successes_per_cell=12):
                        cell = suite.cell(item.evaluation_cell_id)
                        index = occurrences.get(cell.cell_id, 0)
                        occurrences[cell.cell_id] = index + 1
                        if index >= counts[cell.stratum_id] + successes_per_cell:
                            item = replace(item, success=False, primary_reason="timeout",
                                diagnostic_flags=("goal_not_reached_detected",),
                                max_progress_m=0.1 * cell.progress_goal_m, progress_fraction=0.1,
                                active_terminal_terms=("time_out",))
                        result.append(item)
                    return result

                with patch.object(self, "_unified_outcomes", side_effect=outcomes):
                    phase = self._phase_result_for(plan=self._eval_plan("actual-v3-projection"),
                        phase=EvaluationPhase.PUBLIC_ACCEPT, successes=[0, 1, 0, 1, 0])
                scores = adapter.project_public_accept_scores(phase_result=phase)
                # Each success scores 1 and each timeout with 10% progress scores
                # 0.05. Scalar family mean is 6/12, F_c1 is 6/12, F_c2 is 3/12;
                # J_primary=J_OOD is the equal mean of those three families.
                expected_primary = [0.05 + 0.95 * (5 + offset) / 12 for offset in (0, 1, 0, 1, 0)]
                for actual, expected in zip(scores.score_samples, expected_primary):
                    self.assertAlmostEqual(actual, expected, places=12)
                self.assertEqual(set(scores.named_guard_evidence), {"j_ood", "c0_id", "c1_id", "joint_ood", "phase_control"})
                for name, successes in (("c0_id", 6), ("c1_id", 9), ("joint_ood", 9), ("phase_control", 10), ("j_ood", 5)):
                    guard = scores.named_guard_evidence[name]
                    expected = [0.05 + 0.95 * (successes + offset) / 12 for offset in (0, 1, 0, 1, 0)]
                    for actual, value in zip(guard["score_samples"], expected):
                        self.assertAlmostEqual(actual, value, places=12)
                    self.assertAlmostEqual(guard["score_stats"]["mean"], sum(expected) / 5, places=12)
                    self.assertEqual(guard["score_stats"]["eval_protocol_hash"], self.config.evaluation_protocol().protocol_hash)
                self.assertEqual(scores.score_stats["count"], 5)
                self.assertEqual(scores.score_stats["statistics_version"], DERIVED_STATISTICS_VERSION)
                self.assertNotIn("sealed", json.dumps(scores.to_dict()))

    def test_phase_summary_preserves_learner_evidence_outside_task_semantics(self) -> None:
        plan = self._eval_plan("learner-evidence")
        phase = EvaluationPhase.PUBLIC_DEV
        execution = self._execution({phase: [5, 6, 5]})
        learner = {
            "schema_version": "true_rl_agent.rsl_rl_constructed_learner.v1",
            "initial_noise_std": 1.0,
        }
        summaries = [
            build_stair_eval_phase_summary(
                config=self.config,
                identity=plan.job_identity(),
                evaluation_plan_hash=canonical_json_hash(plan.to_public_dict()),
                phase=phase,
                execution=item,
                evaluation_binding=plan.evaluation_binding,
                checkpoint_safety_report=self._checkpoint_report(),
            )
            for item in (execution, replace(execution, constructed_learner=learner))
        ]
        self.assertEqual(summaries[1]["constructed_learner"], learner)
        self.assertEqual(summaries[0]["task_evidence"], summaries[1]["task_evidence"])
        self.assertEqual(summaries[0]["task_evidence_hash"], summaries[1]["task_evidence_hash"])

    def test_frozen_p5_reference_and_current_candidates_share_comparison_statistics(
        self,
    ) -> None:
        fixtures = json.loads(
            (
                ROOT / "tests/fixtures/stair_summaries/p5_legacy_public_dev.json"
            ).read_text(encoding="utf-8")
        )
        for robot, fixture in fixtures.items():
            self.config = StairClimbingTaskConfig.defaults(
                ROOT / "configs/tasks" / fixture["protocol"]
            )
            adapter = StairPhasedEvaluationAdapter(
                self.config, lambda: fixture["objective_family_mask"]
            )
            for old_parent, old_candidates in (
                (True, False),
                (True, True),
                (False, True),
            ):
                with (
                    self.subTest(
                        robot=robot,
                        old_parent=old_parent,
                        old_candidates=old_candidates,
                    ),
                    tempfile.TemporaryDirectory() as tmp,
                ):
                    parent_plan = replace(
                        self._eval_plan("parent"),
                        subject=ExperimentSubject.reference("p5-reference"),
                    )
                    parent_phase = self._phase_result_for(
                        plan=parent_plan,
                        phase=EvaluationPhase.PUBLIC_DEV,
                        successes=[3] * 3,
                    )
                    if old_parent:
                        parent_phase = replace(
                            parent_phase,
                            task_evidence={
                                **parent_phase.task_evidence,
                                "summary": fixture["summary"],
                            },
                        )
                    before = canonical_json_hash(parent_phase.to_private_dict())
                    journal = TrialReceiptJournal(
                        Path(tmp),
                        contract={
                            "schema_version": "true_rl_agent.trial_execution_contract.v1",
                            "round_id": "imported-reference",
                            "session_genesis_hash": "7" * 64,
                            "ledger_epoch_hash": "8" * 64,
                        },
                    )
                    ref = journal.publish_eval_phase(
                        EvalPhaseJobSpec(
                            job_id=parent_phase.job_id,
                            evaluation_plan=parent_plan,
                            phase=EvaluationPhase.PUBLIC_DEV,
                        ),
                        parent_phase,
                    )
                    parent = Mock(
                        spec=CurrentBestEvalStats,
                        anchor_kind="reference",
                        anchor_id="p5-reference",
                        checkpoint_hash=parent_phase.checkpoint_hash,
                        training_binding_hash=parent_phase.training_binding_hash,
                        capability_profile_hash=parent_phase.capability_profile_hash,
                        evaluation_binding_hash=parent_phase.evaluation_binding_hash,
                    )
                    genesis = Mock(spec=VerifiedSessionGenesis)
                    genesis.reference_eval_plan.return_value = parent_plan
                    genesis.evaluation_binding = parent_plan.evaluation_binding
                    genesis.initial_current_best = parent
                    genesis.reference_eval_phase_receipt.return_value = SimpleNamespace(
                        result=parent_phase, receipt_ref=ref
                    )
                    backend, runner = Mock(spec=BackendAdapter), Mock(spec=Runner)
                    guard = run_genesis_parent_public_dev(
                        genesis=genesis,
                        adapter=adapter,
                        backend=backend,
                        runner=runner,
                        receipt_root=Path(tmp) / "unused",
                        generation_session_contract_hash="8" * 64,
                    )
                    backend.materialize_eval_phase.assert_not_called()
                    runner.run_eval_phase.assert_not_called()
                    screens = []
                    for trial_id, count in (("trial-a", 3), ("trial-b", 2)):
                        phase = self._phase_result_for(
                            plan=self._eval_plan(trial_id),
                            phase=EvaluationPhase.PUBLIC_DEV,
                            successes=[count] * 3,
                        )
                        if old_candidates:
                            phase = replace(
                                phase,
                                task_evidence={
                                    **phase.task_evidence,
                                    "summary": fixture["summary"],
                                },
                            )
                        screens.append(
                            self._candidate_screen(
                                build_stair_public_dev_screen_evidence(
                                    config=self.config,
                                    trial_id=trial_id,
                                    phase_result=phase,
                                    screen_role="guard_reference",
                                    objective_family_mask=fixture[
                                        "objective_family_mask"
                                    ],
                                )
                            )
                        )
                    selected = adapter.select(screens, guard_reference=guard.evidence)
                    if not old_candidates:
                        self.assertEqual(selected.selected_trial_id, "trial-a")
                    for screen in (*screens, guard.evidence):
                        self.assertEqual(
                            screen.generalization_score_stats["statistics_version"],
                            DERIVED_STATISTICS_VERSION,
                        )
                        self.assertEqual(
                            screen.anchor_score_stats["statistics_version"],
                            DERIVED_STATISTICS_VERSION,
                        )
                    self.assertEqual(
                        canonical_json_hash(parent_phase.to_private_dict()), before
                    )
                    self.assertEqual(guard.evidence.phase_result_hash, before)
                    mismatched = dict(guard.evidence.generalization_score_stats)
                    mismatched.pop("statistics_version")
                    with self.assertRaisesRegex(
                        ValueError, "statistics versions differ"
                    ):
                        adapter.select(
                            screens,
                            guard_reference=replace(
                                guard.evidence, generalization_score_stats=mismatched
                            ),
                        )

    def test_historical_screen_statistics_replay_through_formal_assembly(self) -> None:
        fixture = json.loads(
            (
                ROOT / "tests/fixtures/stair_summaries/p5_legacy_public_dev.json"
            ).read_text(encoding="utf-8")
        )["go2"]
        self.config = StairClimbingTaskConfig.defaults(
            ROOT / "configs/tasks" / fixture["protocol"]
        )
        mask = fixture["objective_family_mask"]
        plans = {
            name: self._eval_plan(name) for name in ("trial-a", "trial-b", "parent")
        }
        phases = {}
        for name, plan in plans.items():
            phase = self._phase_result_for(
                plan=plan, phase=EvaluationPhase.PUBLIC_DEV, successes=[3] * 3
            )
            phases[name] = replace(
                phase,
                task_evidence={**phase.task_evidence, "summary": fixture["summary"]},
            )
        screens = {
            name: build_stair_public_dev_screen_evidence(
                config=self.config,
                trial_id=name,
                phase_result=phase,
                screen_role="guard_reference",
                objective_family_mask=mask,
                statistics_version=None,
            )
            for name, phase in phases.items()
        }
        candidates = [
            self._candidate_screen(screens["trial-a"]),
            self._candidate_screen(screens["trial-b"], runtime_status="fail"),
        ]
        # Serialized historical screens retain their absent marker and original identity.
        candidates = [
            StairPublicDevScreenEvidence.from_mapping(screen.to_selector_dict())
            for screen in candidates
        ]
        selection = select_stair_public_dev_winner(
            config=self.config, screens=candidates, guard_reference=screens["parent"]
        )
        self.assertEqual(selection.selected_trial_id, "trial-a")
        self.assertNotIn("statistics_version", candidates[0].generalization_score_stats)
        results = [
            phases["trial-a"],
            *[
                self._phase_result_for(
                    plan=plans["trial-a"],
                    phase=phase,
                    successes=[3] * self._seed_count(phase),
                )
                for phase in (
                    EvaluationPhase.PUBLIC_ACCEPT,
                    EvaluationPhase.SEALED_VALIDATION,
                )
            ],
        ]
        with tempfile.TemporaryDirectory() as tmp:
            refs = self._publish_phase_receipts(
                root=Path(tmp), plan=plans["trial-a"], phase_results=results
            )
            assembly = assemble_stair_eval_phase_results(
                config=self.config,
                eval_plan=plans["trial-a"],
                trial_id="trial-a",
                selection_result=selection,
                selection_screens=candidates,
                guard_reference=screens["parent"],
                phase_results=results,
                phase_receipt_refs=refs,
                objective_family_mask=mask,
            )
        self.assertEqual(assembly.selected_trial_id, "trial-a")
        self.assertEqual(assembly.selection, selection.to_public_dict())

    def test_generation_cost_and_genesis_guard_bootstrap_are_separate(self) -> None:
        cost = stair_phased_evaluation_cost_contract(self.config)
        self.assertEqual(cost["schema_version"], STAIR_PHASED_EVAL_COST_SCHEMA_VERSION)
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

    def test_unified_parent_guard_reuses_genesis_public_dev_receipt(self) -> None:
        unified = StairClimbingTaskConfig.defaults(
            ROOT / "configs" / "tasks" / "stair_unified_protocol_v1.json"
        )
        cost = stair_phased_evaluation_cost_contract(unified)
        self.assertEqual(cost["session_bootstrap"], {})
        self.assertEqual(cost["max_session_bootstrap_seed_units"], 0)
        adapter = StairPhasedEvaluationAdapter(unified)
        genesis = Mock(spec=VerifiedSessionGenesis)
        genesis.reference_eval_plan.return_value = Mock(spec=EvalJobSpec)
        genesis.evaluation_binding = SimpleNamespace(
            task_protocol_hash=unified.task_protocol_hash
        )
        loaded = SimpleNamespace(result=object(), receipt_ref=object())
        genesis.reference_eval_phase_receipt.return_value = loaded
        genesis.initial_current_best = object()
        backend = Mock(spec=BackendAdapter)
        runner = Mock(spec=Runner)
        expected = object()

        with patch(
            "true_rl_agent.agent_core.generation_guard."
            "build_generation_guard_reference",
            return_value=expected,
        ) as build:
            actual = run_genesis_parent_public_dev(
                genesis=genesis,
                adapter=adapter,
                backend=backend,
                runner=runner,
                receipt_root=Path("unused"),
                generation_session_contract_hash="8" * 64,
            )

        self.assertIs(actual, expected)
        genesis.reference_eval_phase_receipt.assert_called_once_with(
            EvaluationPhase.PUBLIC_DEV
        )
        backend.materialize_eval_phase.assert_not_called()
        runner.run_eval_phase.assert_not_called()
        build.assert_called_once_with(
            adapter=adapter,
            parent=genesis.initial_current_best,
            trial_id="genesis-parent",
            phase_result=loaded.result,
            phase_receipt_ref=loaded.receipt_ref,
        )

    def test_guard_chain_replay_updates_only_after_accepted_generation(self) -> None:
        initial_parent = Mock(spec=CurrentBestEvalStats)
        accepted_parent = Mock(spec=CurrentBestEvalStats)
        initial_guard = Mock(spec=GenerationGuardReference)
        accepted_guard = Mock(spec=GenerationGuardReference)
        acceptance_protocol = self.config.acceptance_protocol
        adapter = StairPhasedEvaluationAdapter(self.config)
        records = (
            {"generation_outcome": "formal_rejected", "generation_index": 1},
            {
                "generation_outcome": "paired_authoring_incomplete",
                "generation_index": 2,
            },
            {"generation_outcome": "accepted", "generation_index": 3},
            {"generation_outcome": "no_selection", "generation_index": 4},
        )
        expected_contract_hash = "9" * 64
        validator_target = (
            "true_rl_agent.agent_core.generation_guard."
            "validate_generation_ledger_record"
        )
        loader_target = (
            "true_rl_agent.agent_core.generation_guard."
            "load_next_generation_guard_reference"
        )

        with (
            patch(
                validator_target,
                side_effect=(
                    initial_parent,
                    initial_parent,
                    accepted_parent,
                    accepted_parent,
                ),
            ) as validate_record,
            patch(loader_target, return_value=accepted_guard) as load_guard,
        ):
            current_parent, current_guard = replay_generation_guard_chain(
                generation_records=records,
                initial_parent=initial_parent,
                initial_guard_reference=initial_guard,
                acceptance_protocol=acceptance_protocol,
                adapter=adapter,
                expected_generation_session_contract_hash=(expected_contract_hash),
            )

        self.assertIs(current_parent, accepted_parent)
        self.assertIs(current_guard, accepted_guard)
        self.assertEqual(
            [item.args[0] for item in validate_record.call_args_list],
            list(records),
        )
        self.assertEqual(
            [item.kwargs["expected_parent"] for item in validate_record.call_args_list],
            [initial_parent, initial_parent, initial_parent, accepted_parent],
        )
        self.assertEqual(
            {
                item.kwargs["expected_generation_session_contract_hash"]
                for item in validate_record.call_args_list
            },
            {expected_contract_hash},
        )
        self.assertTrue(
            all(
                item.kwargs["acceptance_protocol"] is acceptance_protocol
                for item in validate_record.call_args_list
            )
        )
        load_guard.assert_called_once_with(
            generation_record=records[2],
            previous_parent=initial_parent,
            acceptance_protocol=acceptance_protocol,
            adapter=adapter,
        )
        initial_guard.assert_matches_parent.assert_called_once_with(initial_parent)
        self.assertEqual(
            accepted_guard.assert_matches_parent.call_args_list,
            [call(accepted_parent), call(accepted_parent)],
        )

    def test_guard_chain_replay_rejects_nonaccepted_head_transition(self) -> None:
        initial_parent = Mock(spec=CurrentBestEvalStats)
        changed_parent = Mock(spec=CurrentBestEvalStats)
        initial_guard = Mock(spec=GenerationGuardReference)
        record = {"generation_outcome": "formal_rejected"}
        validator_target = (
            "true_rl_agent.agent_core.generation_guard."
            "validate_generation_ledger_record"
        )
        loader_target = (
            "true_rl_agent.agent_core.generation_guard."
            "load_next_generation_guard_reference"
        )

        with (
            patch(validator_target, return_value=changed_parent),
            patch(loader_target) as load_guard,
            self.assertRaisesRegex(ValueError, "non-accepted generation changed"),
        ):
            replay_generation_guard_chain(
                generation_records=(record,),
                initial_parent=initial_parent,
                initial_guard_reference=initial_guard,
                acceptance_protocol=self.config.acceptance_protocol,
                adapter=StairPhasedEvaluationAdapter(self.config),
            )
        load_guard.assert_not_called()

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
            tampered_screens = list(selection_screens)
            tampered_screens[0] = self._candidate_screen(
                tampered_screens[0],
                runtime_status="fail",
            )
            with self.assertRaisesRegex(ValueError, "selection replay mismatch"):
                assemble_stair_eval_phase_results(
                    config=self.config,
                    eval_plan=plan,
                    trial_id="trial-winner",
                    selection_result=selection,
                    selection_screens=tampered_screens,
                    guard_reference=guard_reference,
                    phase_results=phase_results,
                    phase_receipt_refs=phase_receipts,
                )

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

    def test_reference_three_phase_assembly_needs_no_candidate_selection(self) -> None:
        plan = replace(
            self._eval_plan("reference-phased"),
            subject=ExperimentSubject.reference("reference-phased"),
        )
        split_successes = {
            EvaluationPhase.PUBLIC_DEV: [5, 6, 5],
            EvaluationPhase.PUBLIC_ACCEPT: [6, 5, 6, 5, 6],
            EvaluationPhase.SEALED_VALIDATION: [5, 5, 6, 5, 6, 5, 6],
        }
        checkpoint_report = self._checkpoint_report()
        phase_results = []
        for phase, successes in split_successes.items():
            summary = build_stair_eval_phase_summary(
                config=self.config,
                identity=replace(plan.job_identity(), job_id=f"reference-{phase.value}"),
                evaluation_plan_hash=canonical_json_hash(plan.to_public_dict()),
                phase=phase,
                execution=self._execution({phase: successes}),
                evaluation_binding=plan.evaluation_binding,
                checkpoint_safety_report=checkpoint_report,
            )
            phase_results.append(
                self._phase_result(
                    plan=plan,
                    phase=phase,
                    job_id=f"reference-{phase.value}",
                    summary=summary,
                )
            )
        with tempfile.TemporaryDirectory() as tmp:
            phase_receipts = self._publish_phase_receipts(
                root=Path(tmp),
                plan=plan,
                phase_results=phase_results,
            )
            assembled = assemble_stair_reference_eval_phase_results(
                config=self.config,
                eval_plan=plan,
                phase_results=phase_results,
                phase_receipt_refs=phase_receipts,
            )
            result = SubprocessRunner(
                allowed_executables=("python",)
            ).project_assembled_eval_summary(plan, assembled)
        self.assertEqual(assembled.selected_trial_id, "reference-phased")
        self.assertEqual(
            assembled.selection["selection_kind"],
            "reference_program_qualification",
        )
        self.assertEqual(result.status, "pass")

    def test_reference_does_not_consume_candidate_public_to_sealed_guard(self) -> None:
        self._check_reference_and_candidate_sealed_guard("stair_unified_protocol_v1.json")

    def test_current_robot_rows_enforce_candidate_sealed_consistency(self) -> None:
        for filename in ("stair_unified_protocol_v3.json", "stair_go2_12dof_unified_protocol_v3.json"):
            with self.subTest(protocol=filename):
                self._check_reference_and_candidate_sealed_guard(filename)

    def _check_reference_and_candidate_sealed_guard(self, filename: str) -> None:
        protocol_path = ROOT / "configs/tasks" / filename
        base = StairClimbingTaskConfig.defaults(protocol_path)
        manifest = resolve_stair_cell_manifest(
            cell_variant_id=base.cell_variant_id,
            config=base,
            plugin=StairClimbingTaskPlugin(base),
        )
        self.config = StairClimbingTaskConfig.defaults(
            protocol_path,
            resolved_cell_task_protocol_hash=(
                manifest.resolved_cell_task_protocol_hash
            ),
        )
        reference_plan = replace(
            self._eval_plan("reference-sealed-boundary"),
            subject=ExperimentSubject.reference("reference-sealed-boundary"),
        )
        episodes = self.config.ordered_evaluation_suite.episodes_per_seed_per_cell
        split_successes = {
            EvaluationPhase.PUBLIC_DEV: [2 * episodes // 3] * 3,
            EvaluationPhase.PUBLIC_ACCEPT: [2 * episodes // 3] * 5,
            EvaluationPhase.SEALED_VALIDATION: [episodes // 3] * 7,
        }
        checkpoint_report = self._checkpoint_report()
        phase_results = []
        for phase in EvaluationPhase:
            job_id = f"reference-sealed-boundary-{phase.value}"
            summary = build_stair_eval_phase_summary(
                config=self.config,
                identity=replace(reference_plan.job_identity(), job_id=job_id),
                evaluation_plan_hash=canonical_json_hash(reference_plan.to_public_dict()),
                phase=phase,
                execution=self._execution({phase: split_successes[phase]}),
                evaluation_binding=reference_plan.evaluation_binding,
                checkpoint_safety_report=checkpoint_report,
            )
            phase_results.append(
                self._phase_result(
                    plan=reference_plan,
                    phase=phase,
                    job_id=job_id,
                    summary=summary,
                )
            )

        with tempfile.TemporaryDirectory() as tmp:
            phase_receipts = self._publish_phase_receipts(
                root=Path(tmp),
                plan=reference_plan,
                phase_results=phase_results,
            )
            reference = assemble_stair_reference_eval_phase_results(
                config=self.config,
                eval_plan=reference_plan,
                phase_results=phase_results,
                phase_receipt_refs=phase_receipts,
            ).summary

        reference_evidence = reference["sealed_validation_evidence"]
        self.assertEqual(reference_evidence["status"], "pass")
        self.assertEqual(
            reference_evidence["reason"],
            "unified_stair_reference_sealed_telemetry_complete",
        )
        public_mean = reference["public_score"]
        sealed_mean = reference["sealed_validation_score"]["j_primary"][
            "score_stats"
        ]["mean"]
        self.assertGreater(
            public_mean - sealed_mean,
            self.config.acceptance_protocol.max_public_to_sealed_score_drop,
        )
        _validate_unified_sealed_score_verdict(
            self.config,
            reference_evidence,
            public_mean=public_mean,
        )

        candidate_plan = self._eval_plan("candidate-sealed-boundary")
        candidate = build_stair_full_eval_summary(
            config=self.config,
            identity=candidate_plan.job_identity(),
            evaluation_binding=candidate_plan.evaluation_binding,
            execution=self._execution(split_successes),
            checkpoint_safety_report=checkpoint_report,
            subject_manifest_hash=candidate_plan.subject_manifest_hash,
            program_source_checksum=candidate_plan.program_source_checksum,
            training_binding_hash=candidate_plan.training_binding_hash,
            capability_profile_hash=candidate_plan.capability_profile_hash,
            objective_family_mask=reference["metrics"]["objective_family_mask"],
        )
        self.assertEqual(candidate["sealed_validation_evidence"]["status"], "fail")
        self.assertEqual(candidate["sealed_validation_gate"]["status"], "fail")
        self.assertEqual(
            candidate["sealed_validation_evidence"]["reason"],
            "unified_stair_sealed_public_consistency_failed",
        )
        _validate_unified_sealed_score_verdict(
            self.config,
            candidate["sealed_validation_evidence"],
            public_mean=candidate["public_score"],
        )
        forged = dict(candidate["sealed_validation_evidence"], status="pass",
                      reason="unified_stair_sealed_primary_complete")
        with self.assertRaisesRegex(ValueError, "verdict differs"):
            _validate_unified_sealed_score_verdict(self.config, forged, public_mean=candidate["public_score"])

    def test_selector_uses_public_dev_only_and_can_return_no_selection(self) -> None:
        guard_reference = self._screen_with_scores(
            "parent",
            objective=[0.5] * 3,
            guard=[0.8] * 3,
            screen_role="guard_reference",
        )
        left = self._screen_with_scores("trial-a", objective=[0.7] * 3, guard=[0.8] * 3)
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
        self.assertEqual(selected.paired_statistics_status, "applicable")
        self.assertEqual(
            set(selected.to_public_dict()["paired_statistics"]),
            {
                "status",
                "mean_delta",
                "standard_error",
                "uncertainty_threshold",
            },
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
        self.assertEqual(tie.paired_statistics_status, "applicable")

    def test_selector_ranks_tiny_gaps_and_filters_clear_guard_regression(self) -> None:
        parent = self._screen_with_scores(
            "parent",
            objective=[0.5] * 3,
            guard=[0.8] * 3,
            screen_role="guard_reference",
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
        safe = self._screen_with_scores("safe", objective=[0.7] * 3, guard=[0.8] * 3)
        guard_filtered = select_stair_public_dev_winner(
            config=self.config,
            screens=[unsafe, safe],
            guard_reference=parent,
        )
        self.assertEqual(guard_filtered.selected_trial_id, "safe")
        self.assertEqual(guard_filtered.reason, "public_dev_guard_filter")
        self.assertEqual(guard_filtered.ranking_confidence, "not_applicable")
        self.assertEqual(
            guard_filtered.selection_eligible_trial_ids,
            ("safe",),
        )
        self.assertEqual(guard_filtered.paired_statistics_status, "not_applicable")
        self.assertIsNone(guard_filtered.paired_mean_delta)
        self.assertIsNone(guard_filtered.paired_standard_error)
        self.assertIsNone(guard_filtered.uncertainty_threshold)
        guard_public = guard_filtered.to_public_dict()
        self.assertEqual(
            guard_public["paired_statistics"],
            {"status": "not_applicable"},
        )

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
        self.assertEqual(both_unsafe.paired_statistics_status, "not_applicable")
        self.assertEqual(
            both_unsafe.to_public_dict()["paired_statistics"],
            {"status": "not_applicable"},
        )

    def test_selector_filters_unobserved_runtime_effect_before_ranking(self) -> None:
        parent = self._screen_with_scores(
            "parent",
            objective=[0.5] * 3,
            guard=[0.8] * 3,
            screen_role="guard_reference",
        )
        zero_high = self._screen_with_scores(
            "zero-high",
            objective=[0.9] * 3,
            guard=[0.8] * 3,
            runtime_status="fail",
        )
        live_lower = self._screen_with_scores(
            "live-lower",
            objective=[0.7] * 3,
            guard=[0.8] * 3,
        )
        filtered = select_stair_public_dev_winner(
            config=self.config,
            screens=[zero_high, live_lower],
            guard_reference=parent,
        )
        self.assertEqual(filtered.selected_trial_id, "live-lower")
        self.assertEqual(filtered.reason, "public_dev_runtime_activation_filter")
        self.assertEqual(
            filtered.selection_eligible_trial_ids,
            ("live-lower",),
        )
        self.assertEqual(filtered.paired_statistics_status, "not_applicable")

        both_zero = select_stair_public_dev_winner(
            config=self.config,
            screens=[
                zero_high,
                self._screen_with_scores(
                    "zero-low",
                    objective=[0.6] * 3,
                    guard=[0.8] * 3,
                    runtime_status="fail",
                ),
            ],
            guard_reference=parent,
        )
        self.assertIsNone(both_zero.selected_trial_id)
        self.assertEqual(
            both_zero.reason,
            "no_runtime_activation_eligible_candidate",
        )
        self.assertEqual(both_zero.selection_eligible_trial_ids, ())
        self.assertEqual(
            both_zero.to_public_dict()["paired_statistics"],
            {"status": "not_applicable"},
        )

    def test_selector_records_declared_cross_gpu_execution_differences(self) -> None:
        parent = self._screen_with_scores(
            "parent",
            objective=[0.5] * 3,
            guard=[0.8] * 3,
            screen_role="guard_reference",
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
                screen_role="guard_reference",
            )

        phases = [
            self._phase_result_for(
                plan=plan,
                phase=phase,
                successes=[4] * self._seed_count(phase),
            )
            for phase in EvaluationPhase
        ]
        loser_screen = self._candidate_screen(
            build_stair_public_dev_screen_evidence(
                config=self.config,
                trial_id="trial-loser",
                phase_result=phases[0],
                screen_role="guard_reference",
            )
        )
        loser_screens = [
            loser_screen,
            self._screen("trial-better", [7, 7, 7]),
        ]
        loser_guard_reference = self._screen(
            "trial-parent", [4, 4, 4], screen_role="guard_reference"
        )
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

    def test_canonical_phase_accepts_explicit_final_confirmation_namespace(
        self,
    ) -> None:
        final_binding = EvaluationBindingV1(
            eval_protocol_hash=self.config.evaluation_protocol().protocol_hash,
            acceptance_protocol_hash=self.config.acceptance_protocol.protocol_hash,
            task_protocol_hash=self.config.task_protocol_hash,
            seed_cohort_hash=self.config.seed_resolver().namespace_cohort_hash(
                STAIR_FINAL_CONFIRMATION_EVAL_NAMESPACE
            ),
        )
        plan = replace(
            self._eval_plan("final-confirmation"),
            evaluation_binding=final_binding,
        )
        phase = EvaluationPhase.SEALED_VALIDATION
        execution = self._execution(
            {phase: [5, 5, 6, 5, 6, 5, 6]},
            seed_namespace=STAIR_FINAL_CONFIRMATION_EVAL_NAMESPACE,
        )
        summary = build_stair_eval_phase_summary(
            config=self.config,
            identity=replace(plan.job_identity(), job_id="final-confirmation-phase"),
            evaluation_plan_hash=canonical_json_hash(plan.to_public_dict()),
            phase=phase,
            execution=execution,
            evaluation_binding=final_binding,
            checkpoint_safety_report=self._checkpoint_report(),
            expected_seed_namespace=STAIR_FINAL_CONFIRMATION_EVAL_NAMESPACE,
        )
        result = self._phase_result(
            plan=plan,
            phase=phase,
            job_id="final-confirmation-phase",
            summary=summary,
        )

        canonical = canonical_stair_phase_result_summary(
            config=self.config,
            expected_phase=phase,
            result=result,
            expected_seed_namespace=STAIR_FINAL_CONFIRMATION_EVAL_NAMESPACE,
        )

        self.assertEqual(canonical, summary["task_evidence"]["summary"])
        with self.assertRaisesRegex(ValueError, "execution identity binding"):
            canonical_stair_phase_result_summary(
                config=self.config,
                expected_phase=phase,
                result=result,
            )

    def test_stair_plugin_owns_the_optional_phase_capability(self) -> None:
        adapter = StairClimbingTaskPlugin(self.config).phased_evaluation_adapter()
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
                guard_reference=self._screen(
                    "trial-a", [4, 4, 4], screen_role="guard_reference"
                ),
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
        execution = self._execution({phase: [4] * self.config.public_seed_count})
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

    def _screen(
        self,
        trial_id: str,
        successes: list[int],
        *,
        screen_role: str = "candidate",
        runtime_status: str = "pass",
    ):
        plan = self._eval_plan(f"candidate-{trial_id}")
        result = self._phase_result_for(
            plan=plan,
            phase=EvaluationPhase.PUBLIC_DEV,
            successes=successes,
        )
        screen = build_stair_public_dev_screen_evidence(
            config=self.config,
            trial_id=trial_id,
            phase_result=result,
            screen_role="guard_reference",
        )
        if screen_role == "guard_reference":
            return screen
        return self._candidate_screen(screen, runtime_status=runtime_status)

    def _candidate_screen(
        self,
        screen: StairPublicDevScreenEvidence,
        *,
        runtime_status: str = "pass",
    ) -> StairPublicDevScreenEvidence:
        reason = {
            "pass": "candidate_reward_operation_runtime_effect_observed",
            "fail": "candidate_reward_operation_runtime_effect_not_observed",
        }[runtime_status]
        return replace(
            screen,
            screen_role="candidate",
            train_receipt_checksum=(
                "sha256:" + canonical_json_hash({"trial_id": screen.trial_id})
            ),
            runtime_effect_evidence={
                "schema_version": (
                    "true_rl_agent.task.stair_climbing."
                    "candidate_runtime_effect_evidence.v1"
                ),
                "status": runtime_status,
                "reason": reason,
                "acceptance_authority": False,
            },
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
        selected = self._candidate_screen(
            build_stair_public_dev_screen_evidence(
                config=self.config,
                trial_id=trial_id,
                phase_result=result,
                screen_role="guard_reference",
            )
        )
        sibling = self._screen(f"{trial_id}-sibling", [1, 1, 1])
        guard_reference = replace(
            self._screen(
                f"{trial_id}-parent",
                [4, 4, 4],
                screen_role="guard_reference",
            ),
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
        screen_role: str = "candidate",
        runtime_status: str = "pass",
    ):
        return replace(
            self._screen(
                trial_id,
                [4, 4, 4],
                screen_role=screen_role,
                runtime_status=runtime_status,
            ),
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
                ),
            },
        )

    def _execution(
        self,
        split_successes: dict[EvaluationPhase, list[int]],
        *,
        seed_namespace: str = STAIR_FORMAL_EVAL_NAMESPACE,
    ) -> StairEvaluationExecutionResult:
        summaries = {}
        outcomes = {}
        steps = {}
        walls = {}
        for phase, successes in split_successes.items():
            if self.config.is_unified_protocol:
                per_seed = tuple(
                    tuple(
                        self._unified_outcomes(
                            split=phase.value,
                            successes_per_cell=count,
                        )
                    )
                    for count in successes
                )
                summaries[phase.value] = build_unified_stair_summary(
                    suite=self.config.ordered_evaluation_suite,
                    semantics=self.config.evaluation_semantics,
                    split=phase.value,
                    per_seed_outcomes=per_seed,
                )
            else:
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
            seed_namespace=seed_namespace,
            seed_cohort_hash=self.config.seed_resolver().namespace_cohort_hash(
                seed_namespace
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

    def _unified_outcomes(
        self,
        *,
        split: str,
        successes_per_cell: int,
    ) -> list[StairEpisodeOutcome]:
        outcomes = []
        occurrences: dict[str, int] = {}
        for env_index, cell in enumerate(
            self.config.ordered_evaluation_suite.private_cell_assignment(split)
        ):
            occurrence = occurrences.get(cell.cell_id, 0)
            occurrences[cell.cell_id] = occurrence + 1
            success = occurrence < successes_per_cell
            outcomes.append(
                StairEpisodeOutcome(
                    env_index=env_index,
                    bin_id=cell.cell_id,
                    episode_steps=cell.episode_horizon_control_steps,
                    primary_reason="success" if success else "timeout",
                    success=success,
                    diagnostic_flags=(
                        ("first_valid_top_arrival_achieved",)
                        if success
                        else ("goal_not_reached_detected",)
                    ),
                    max_progress_m=(
                        cell.progress_goal_m if success else 0.1 * cell.progress_goal_m
                    ),
                    progress_fraction=1.0 if success else 0.1,
                    active_terminal_terms=() if success else ("time_out",),
                    evaluation_cell_id=cell.cell_id,
                )
            )
        return outcomes

    def _outcomes(self, *, successes_per_cell: int) -> list[StairEpisodeOutcome]:
        outcomes = []
        for env_index, cell in enumerate(
            self.config.evaluation_profile.private_cell_assignment()
        ):
            success = (
                env_index % self.config.evaluation_profile.episodes_per_seed_per_cell
                < successes_per_cell
            )
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
