from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from true_rl_agent.composition.base import (
    TaskCompositionContext,
)
from true_rl_agent.agent_core.generation_lifecycle import GenerationTerminalBinding
from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.tasks.stair_climbing.composition import (
    StairClimbingCompositionProvider,
)
from true_rl_agent.tasks.stair_climbing.config import StairClimbingTaskConfig
from true_rl_agent.tasks.stair_climbing.episode_evidence import (
    StairEpisodeOutcome,
)
from true_rl_agent.tasks.stair_climbing.final_confirmation import (
    MAX_STAIR_FINAL_CONFIRMATION_PLAN_BYTES,
    MAX_STAIR_FINAL_CONFIRMATION_RESERVATION_BYTES,
    MAX_STAIR_FINAL_CONFIRMATION_RESULT_BYTES,
    STAIR_FINAL_CONFIRMATION_PLAN_V2_SCHEMA_VERSION,
    STAIR_FINAL_CONFIRMATION_REQUIRED_GENERATIONS,
    STAIR_FINAL_CONFIRMATION_REQUIRED_ROUNDS,
    STAIR_FINAL_CONFIRMATION_SPLIT,
    StairFinalConfirmationPlan,
    StairFinalConfirmationPlanV2,
    StairFinalConfirmationPhaseExecution,
    StairFinalConfirmationReservation,
    StairFinalConfirmationReservationV2,
    StairFinalConfirmationResult,
    StairFinalConfirmationResultV2,
    build_stair_final_confirmation_plan,
    build_stair_generation_final_confirmation_plan,
    build_stair_generation_final_confirmation_result,
    build_stair_final_confirmation_result,
    create_stair_final_confirmation_plan,
    create_stair_generation_final_confirmation_plan,
    create_stair_final_confirmation_reservation,
    create_stair_generation_final_confirmation_reservation,
    load_stair_final_confirmation_plan,
    load_stair_generation_final_confirmation_plan,
    load_stair_generation_final_confirmation_result,
    load_stair_final_confirmation_reservation,
    load_stair_generation_final_confirmation_reservation,
    load_stair_final_confirmation_result,
    stair_final_confirmation_plan_path,
    stair_final_confirmation_result_path,
    write_stair_final_confirmation_result,
    write_stair_generation_final_confirmation_result,
)
from true_rl_agent.tasks.stair_climbing.generalization_contract import (
    build_stair_generalization_summary,
)
from true_rl_agent.tasks.stair_climbing.isaaclab_execution import (
    StairEvaluationExecutionResult,
)
from true_rl_agent.tasks.stair_climbing.seed_policy import (
    STAIR_FINAL_CONFIRMATION_EVAL_NAMESPACE,
    STAIR_FORMAL_EVAL_NAMESPACE,
    STAIR_PILOT_EVAL_NAMESPACE,
    assert_seed_namespaces_disjoint,
)
from true_rl_agent.tasks.stair_climbing.session_start import (
    STAIR_REFERENCE_START_MODE,
)
from true_rl_agent.tasks.stair_climbing.training_distribution import (
    STAIR_EXECUTION_SOURCE_PREFLIGHT_SCHEMA_VERSION,
    stair_training_distribution_upstream_source_evidence,
)
from true_rl_agent.runner.specs import GpuProcessBinding

ROOT = Path(__file__).resolve().parents[2]
SHA = "a" * 64
CHECKSUM = "sha256:" + SHA
BACKEND_MANIFEST_HASH = "b" * 64
EXECUTION_DEVICE = "cuda:0"


class StairFinalConfirmationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = StairClimbingTaskConfig.defaults(
            ROOT / "configs" / "tasks" / "stair_protocol_v2.json"
        )
        cls.source_manifest_hash = (
            stair_training_distribution_upstream_source_evidence(
                cls.config.protocol_v2.runner_owned_training_distribution
            )["source_manifest_hash"]
        )

    def test_plan_is_exact_hashed_and_contains_no_raw_seed_values(self) -> None:
        plan = build_stair_final_confirmation_plan(
            session_id="stair-final-confirmation-test",
            config=self.config,
        )
        restored = StairFinalConfirmationPlan.from_mapping(plan.to_dict())
        self.assertEqual(restored, plan)
        self.assertEqual(
            plan.seed_cohort_hash,
            self.config.seed_resolver().namespace_cohort_hash(
                STAIR_FINAL_CONFIRMATION_EVAL_NAMESPACE
            ),
        )
        encoded = json.dumps(plan.to_dict(), sort_keys=True)
        for forbidden in (
            '"base_seed"',
            '"private_base_seed"',
            '"derived_seed"',
            '"seed_values"',
        ):
            self.assertNotIn(forbidden, encoded)
        self.assertFalse(plan.to_dict()["information_flow"]["feedback_allowed"])
        self.assertFalse(
            plan.to_dict()["information_flow"]["acceptance_authority"]
        )
        self.assertFalse(
            plan.to_dict()["information_flow"]["public_history_projection"]
        )

        tampered = copy.deepcopy(plan.to_dict())
        tampered["cohort"]["seed_count"] += 1
        with self.assertRaisesRegex(ValueError, "plan hash mismatch"):
            StairFinalConfirmationPlan.from_mapping(tampered)

    def test_generation_plan_v2_is_pre_genesis_exact_and_diagnostic_only(self) -> None:
        plan = build_stair_generation_final_confirmation_plan(
            session_id="stair-generation-final-plan-test",
            config=self.config,
        )
        restored = StairFinalConfirmationPlanV2.from_mapping(plan.to_dict())

        self.assertEqual(restored, plan)
        self.assertEqual(
            plan.to_dict()["schema_version"],
            STAIR_FINAL_CONFIRMATION_PLAN_V2_SCHEMA_VERSION,
        )
        self.assertEqual(
            plan.to_dict()["required_closed_generations"],
            STAIR_FINAL_CONFIRMATION_REQUIRED_GENERATIONS,
        )
        self.assertNotIn("required_closed_rounds", plan.to_dict())
        self.assertEqual(
            plan.to_dict()["comparison"]["final_role"],
            "formal_final_current_best_after_generation_15",
        )
        self.assertEqual(
            plan.to_dict()["comparison"]["execution_order"],
            ["genesis", "final"],
        )
        self.assertFalse(plan.to_dict()["information_flow"]["feedback_allowed"])
        self.assertFalse(
            plan.to_dict()["information_flow"]["acceptance_authority"]
        )
        self.assertEqual(plan.seed_count, 7)

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "fresh-generation"
            created, path = create_stair_generation_final_confirmation_plan(
                output_dir=output_dir,
                session_id=plan.session_id,
                config=self.config,
            )
            self.assertEqual(created, plan)
            self.assertEqual(
                load_stair_generation_final_confirmation_plan(
                    path,
                    expected_session_id=plan.session_id,
                    expected_config=self.config,
                ),
                plan,
            )
            with self.assertRaisesRegex(ValueError, "mismatch"):
                load_stair_final_confirmation_plan(path)

        tampered = copy.deepcopy(plan.to_dict())
        tampered["information_flow"]["feedback_allowed"] = True
        with self.assertRaisesRegex(ValueError, "information-flow mismatch"):
            StairFinalConfirmationPlanV2.from_mapping(tampered)

    def test_generation_reservation_v2_binds_terminal_specs_and_private_gpu(self) -> None:
        plan = build_stair_generation_final_confirmation_plan(
            session_id="stair-generation-reservation-test",
            config=self.config,
        )
        terminal = GenerationTerminalBinding(
            identity_hash="1" * 64,
            required_generation_count=15,
            closed_generation_count=15,
            ledger_snapshot_hash="2" * 64,
            current_head_hash="3" * 64,
            lifecycle_head_hash="4" * 64,
            cost_chain_head_hash="5" * 64,
            segment_count=5,
        )
        reservation = StairFinalConfirmationReservationV2(
            plan_hash=plan.plan_hash,
            session_id=plan.session_id,
            session_genesis_hash="6" * 64,
            generation_session_contract_hash="7" * 64,
            generation_cost_contract_hash="8" * 64,
            ledger_checksum="sha256:" + "9" * 64,
            terminal_binding=terminal.to_dict(),
            source_checkout_hash="a" * 64,
            backend_execution_environment_manifest_hash="b" * 64,
            gpu_process_binding=GpuProcessBinding(
                host_execution_device="cuda:0",
                process_execution_device="cuda:0",
                resource_binding_hash="c" * 64,
            ),
            genesis_job_id="generation-final-genesis-eval-sealed-validation",
            final_job_id="generation-final-current-eval-sealed-validation",
            genesis_eval_phase_spec_hash="d" * 64,
            final_eval_phase_spec_hash="e" * 64,
            genesis_evaluation_plan_hash="f" * 64,
            final_evaluation_plan_hash="0" * 64,
            genesis_checkpoint_hash="sha256:" + "1" * 64,
            final_checkpoint_hash="sha256:" + "2" * 64,
            genesis_source_manifest_hash="3" * 64,
            final_source_manifest_hash="4" * 64,
            genesis_training_binding_hash="5" * 64,
            final_training_binding_hash="6" * 64,
            genesis_capability_profile_hash="7" * 64,
            final_capability_profile_hash="8" * 64,
            final_anchor={"terminal_head_hash": terminal.current_head_hash},
        )
        restored = StairFinalConfirmationReservationV2.from_mapping(
            reservation.to_dict(),
            expected_plan=plan,
        )
        self.assertEqual(restored, reservation)
        self.assertEqual(
            restored.gpu_process_binding.to_dict(),
            {
                "host_execution_device": "cuda:0",
                "process_execution_device": "cuda:0",
                "resource_binding_hash": "c" * 64,
            },
        )
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "formal"
            created_plan, _ = create_stair_generation_final_confirmation_plan(
                output_dir=output_dir,
                session_id=plan.session_id,
                config=self.config,
            )
            self.assertEqual(created_plan, plan)
            created, path = (
                create_stair_generation_final_confirmation_reservation(
                    output_dir=output_dir,
                    plan=plan,
                    reservation=reservation,
                )
            )
            self.assertEqual(created, reservation)
            self.assertEqual(
                load_stair_generation_final_confirmation_reservation(
                    path,
                    expected_plan=plan,
                ),
                reservation,
            )
            replayed, replayed_path = (
                create_stair_generation_final_confirmation_reservation(
                    output_dir=output_dir,
                    plan=plan,
                    reservation=reservation,
                )
            )
            self.assertEqual((replayed, replayed_path), (reservation, path))

        tampered = copy.deepcopy(reservation.to_dict())
        del tampered["gpu_process_binding"]["host_execution_device"]
        with self.assertRaisesRegex(ValueError, "fields mismatch"):
            StairFinalConfirmationReservationV2.from_mapping(
                tampered,
                expected_plan=plan,
            )

    def test_generation_result_v2_binds_runner_receipts_and_is_diagnostic(self) -> None:
        plan = build_stair_generation_final_confirmation_plan(
            session_id="stair-generation-result-test",
            config=self.config,
        )
        reservation = self._generation_reservation_v2(plan)
        genesis_summary = self._uniform_summary(successes_per_cell=2)
        final_summary = self._uniform_summary(successes_per_cell=3)
        genesis_execution = StairFinalConfirmationPhaseExecution(
            role="genesis",
            job_id=reservation.genesis_job_id,
            eval_phase_spec_hash=reservation.genesis_eval_phase_spec_hash,
            evaluation_plan_hash=reservation.genesis_evaluation_plan_hash,
            attempt_result_payload_hash="1" * 64,
            receipt_checksum="sha256:" + "2" * 64,
            resource_binding_hash=(
                reservation.gpu_process_binding.resource_binding_hash
            ),
            checkpoint_hash=reservation.genesis_checkpoint_hash,
            generalization_summary_hash=canonical_json_hash(genesis_summary),
        )
        final_execution = StairFinalConfirmationPhaseExecution(
            role="final",
            job_id=reservation.final_job_id,
            eval_phase_spec_hash=reservation.final_eval_phase_spec_hash,
            evaluation_plan_hash=reservation.final_evaluation_plan_hash,
            attempt_result_payload_hash="3" * 64,
            receipt_checksum="sha256:" + "4" * 64,
            resource_binding_hash=(
                reservation.gpu_process_binding.resource_binding_hash
            ),
            checkpoint_hash=reservation.final_checkpoint_hash,
            generalization_summary_hash=canonical_json_hash(final_summary),
        )
        result = build_stair_generation_final_confirmation_result(
            plan=plan,
            reservation=reservation,
            config=self.config,
            genesis_execution=genesis_execution,
            final_execution=final_execution,
            genesis_summary=genesis_summary,
            final_summary=final_summary,
        )
        self.assertEqual(len(result.paired_seed_deltas), 7)
        self.assertNotIn("accepted", result.to_dict())
        self.assertNotIn("passed", result.to_dict())
        self.assertFalse(
            result.to_dict()["information_flow"]["acceptance_authority"]
        )
        self.assertEqual(
            StairFinalConfirmationResultV2.from_mapping(
                result.to_dict(),
                plan=plan,
                reservation=reservation,
                config=self.config,
            ),
            result,
        )

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "formal"
            create_stair_generation_final_confirmation_plan(
                output_dir=output_dir,
                session_id=plan.session_id,
                config=self.config,
            )
            create_stair_generation_final_confirmation_reservation(
                output_dir=output_dir,
                plan=plan,
                reservation=reservation,
            )
            path = write_stair_generation_final_confirmation_result(
                output_dir=output_dir,
                result=result,
                plan=plan,
                reservation=reservation,
                config=self.config,
            )
            self.assertEqual(
                load_stair_generation_final_confirmation_result(
                    path,
                    plan=plan,
                    reservation=reservation,
                    config=self.config,
                ),
                result,
            )
            with self.assertRaises(FileExistsError):
                write_stair_generation_final_confirmation_result(
                    output_dir=output_dir,
                    result=result,
                    plan=plan,
                    reservation=reservation,
                    config=self.config,
                )

        wrong = copy.deepcopy(genesis_execution.to_dict())
        wrong["eval_phase_spec_hash"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "execution binding mismatch"):
            build_stair_generation_final_confirmation_result(
                plan=plan,
                reservation=reservation,
                config=self.config,
                genesis_execution=(
                    StairFinalConfirmationPhaseExecution.from_mapping(wrong)
                ),
                final_execution=final_execution,
                genesis_summary=genesis_summary,
                final_summary=final_summary,
            )

    def test_plan_is_create_once_and_must_precede_genesis_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "fresh"
            plan, path = create_stair_final_confirmation_plan(
                output_dir=output_dir,
                session_id="stair-final-create-once",
                config=self.config,
            )
            self.assertEqual(path, stair_final_confirmation_plan_path(output_dir))
            self.assertEqual(
                load_stair_final_confirmation_plan(
                    path,
                    expected_session_id=plan.session_id,
                    expected_config=self.config,
                ),
                plan,
            )
            with self.assertRaisesRegex(FileExistsError, "create-once"):
                create_stair_final_confirmation_plan(
                    output_dir=output_dir,
                    session_id=plan.session_id,
                    config=self.config,
                )

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            (output_dir / "session_genesis.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "before Genesis/state"):
                create_stair_final_confirmation_plan(
                    output_dir=output_dir,
                    session_id="stair-plan-too-late",
                    config=self.config,
                )

    def test_confirmation_namespace_is_new_and_old_commitments_are_stable(self) -> None:
        resolver = self.config.seed_resolver()
        self.assertEqual(
            resolver.namespace_cohort_hash(STAIR_FORMAL_EVAL_NAMESPACE),
            "2057585bf7148c4d7462a90729afdfda84aaad0cc4c89b1ecae8f2c05210f362",
        )
        self.assertEqual(
            resolver.namespace_cohort_hash(STAIR_PILOT_EVAL_NAMESPACE),
            "12aa0a4fa550e90203ddee8d1545fea917f48cb01e1b1181dd33bf46fd588f9d",
        )
        self.assertEqual(
            resolver.namespace_cohort_hash(
                STAIR_FINAL_CONFIRMATION_EVAL_NAMESPACE
            ),
            "00c8f4a1a0aeb43a121ccf463c8554ca2501300afd7ac73e6f9bfaf9686064b3",
        )
        assert_seed_namespaces_disjoint(resolver)
        all_values = {}
        for namespace in (
            STAIR_FORMAL_EVAL_NAMESPACE,
            STAIR_PILOT_EVAL_NAMESPACE,
            STAIR_FINAL_CONFIRMATION_EVAL_NAMESPACE,
        ):
            all_values[namespace] = {
                resolver.derive_seed(
                    split=split,
                    run_index=index,
                    namespace=namespace,
                )
                for split, count in (
                    ("public_dev", self.config.public_seed_count),
                    ("public_accept", self.config.public_accept_seed_count),
                    (
                        "sealed_validation",
                        self.config.sealed_validation_seed_count,
                    ),
                )
                for index in range(1, count + 1)
            }
        names = tuple(all_values)
        for index, left in enumerate(names):
            for right in names[index + 1 :]:
                self.assertFalse(all_values[left] & all_values[right])

    def test_live_requires_plan_but_nonlive_composition_does_not(self) -> None:
        provider = StairClimbingCompositionProvider()
        values = dict(provider.cli_defaults())
        values["start_mode"] = STAIR_REFERENCE_START_MODE
        composition = provider.resolve(
            values=values,
            context=TaskCompositionContext(
                project_root=ROOT,
                output_dir=ROOT / "outputs" / "final-confirmation-dry-test",
            ),
        )
        self.assertNotIn(
            "final_confirmation_precommit", composition.task_defaults
        )
        values.update(
            {
                "mode": "live",
                "rounds": STAIR_FINAL_CONFIRMATION_REQUIRED_ROUNDS,
                "session_id": "stair-missing-final-plan",
                "final_confirmation_plan": None,
            }
        )
        with self.assertRaisesRegex(ValueError, "requires --final-confirmation-plan"):
            provider.resolve(
                values=values,
                context=TaskCompositionContext(
                    project_root=ROOT,
                    output_dir=ROOT / "outputs" / "final-confirmation-live-test",
                ),
            )

    def test_reservation_is_fail_closed_single_use_and_result_is_non_authoritative(self) -> None:
        plan = build_stair_final_confirmation_plan(
            session_id="stair-final-single-use",
            config=self.config,
        )
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            reservation, _ = create_stair_final_confirmation_reservation(
                output_dir=output_dir,
                plan=plan,
                session_genesis_hash=SHA,
                ledger_checksum=CHECKSUM,
                genesis_checkpoint_hash=CHECKSUM,
                final_checkpoint_hash=CHECKSUM,
                source_manifest_hash=self.source_manifest_hash,
                backend_execution_environment_manifest_hash=(
                    BACKEND_MANIFEST_HASH
                ),
                execution_device=EXECUTION_DEVICE,
                final_anchor={"anchor_kind": "candidate", "round_id": "round-15"},
            )
            with self.assertRaisesRegex(FileExistsError, "single-use"):
                create_stair_final_confirmation_reservation(
                    output_dir=output_dir,
                    plan=plan,
                    session_genesis_hash=SHA,
                    ledger_checksum=CHECKSUM,
                    genesis_checkpoint_hash=CHECKSUM,
                    final_checkpoint_hash=CHECKSUM,
                    source_manifest_hash=self.source_manifest_hash,
                    backend_execution_environment_manifest_hash=(
                        BACKEND_MANIFEST_HASH
                    ),
                    execution_device=EXECUTION_DEVICE,
                    final_anchor={"anchor_kind": "candidate"},
                )
            genesis_summary = self._uniform_summary(successes_per_cell=4)
            final_summary = self._uniform_summary(successes_per_cell=5)
            result = build_stair_final_confirmation_result(
                plan=plan,
                reservation=reservation,
                config=self.config,
                genesis_execution=self._execution(
                    summary=genesis_summary,
                    checkpoint_hash=reservation.genesis_checkpoint_hash,
                ),
                final_execution=self._execution(
                    summary=final_summary,
                    checkpoint_hash=reservation.final_checkpoint_hash,
                ),
            )
            self.assertAlmostEqual(result.paired_mean, 0.125)
            self.assertEqual(result.paired_standard_error, 0.0)
            self.assertEqual(
                result.to_dict()["information_flow"]["result_role"],
                "diagnostic_final_generalization_confirmation_only",
            )
            self.assertFalse(
                result.to_dict()["information_flow"]["acceptance_authority"]
            )
            self.assertFalse(result.to_dict()["information_flow"]["feedback_allowed"])
            result_path = write_stair_final_confirmation_result(
                output_dir=output_dir,
                result=result,
                plan=plan,
                reservation=reservation,
                config=self.config,
            )
            self.assertEqual(
                result_path,
                stair_final_confirmation_result_path(output_dir),
            )
            self.assertFalse((output_dir / "ledger.jsonl").exists())
            self.assertFalse((output_dir / "public_history.json").exists())
            self.assertEqual(
                load_stair_final_confirmation_result(
                    result_path,
                    plan=plan,
                    reservation=reservation,
                    config=self.config,
                ),
                result,
            )
            with self.assertRaises(FileExistsError):
                write_stair_final_confirmation_result(
                    output_dir=output_dir,
                    result=result,
                    plan=plan,
                    reservation=reservation,
                    config=self.config,
                )

    def test_result_recomputes_full_summary_pairing_and_bindings(self) -> None:
        plan = build_stair_final_confirmation_plan(
            session_id="stair-final-recompute",
            config=self.config,
        )
        reservation = StairFinalConfirmationReservation(
            plan_hash=plan.plan_hash,
            session_id=plan.session_id,
            session_genesis_hash=SHA,
            ledger_checksum=CHECKSUM,
            genesis_checkpoint_hash=CHECKSUM,
            final_checkpoint_hash="sha256:" + "b" * 64,
            source_manifest_hash=self.source_manifest_hash,
            backend_execution_environment_manifest_hash=BACKEND_MANIFEST_HASH,
            execution_device=EXECUTION_DEVICE,
            final_anchor={"anchor_kind": "candidate", "round_id": "round-15"},
        )
        result = build_stair_final_confirmation_result(
            plan=plan,
            reservation=reservation,
            config=self.config,
            genesis_execution=self._execution(
                summary=self._uniform_summary(successes_per_cell=4),
                checkpoint_hash=reservation.genesis_checkpoint_hash,
            ),
            final_execution=self._execution(
                summary=self._uniform_summary(successes_per_cell=5),
                checkpoint_hash=reservation.final_checkpoint_hash,
            ),
        )
        payload = result.to_dict()
        self.assertEqual(
            payload["schema_version"],
            "true_rl_agent.task.stair_climbing.final_confirmation_result.v1",
        )
        self.assertNotIn("seed_macro_scores", payload["genesis"])
        self.assertEqual(
            payload["genesis"]["generalization_summary"]["cell_count"],
            35,
        )
        restored = StairFinalConfirmationResult.from_mapping(
            payload,
            plan=plan,
            reservation=reservation,
            config=self.config,
        )
        self.assertEqual(restored, result)

        for mutate in (
            lambda item: item["final"]["generalization_summary"][
                "seed_summaries"
            ][0]["primary_outcome_counts_by_cell"][
                self.config.evaluation_profile.cells[0].cell_id
            ].update({"success": 0, "timeout": 8}),
            lambda item: item["paired"]["seed_deltas"].__setitem__(0, 0.0),
            lambda item: item["genesis"].update(
                {"generalization_summary_hash": "c" * 64}
            ),
            lambda item: item["final"]["execution_identity"].update(
                {"checkpoint_hash": "sha256:" + "c" * 64}
            ),
            lambda item: item["genesis"]["execution_identity"].update(
                {"seed_namespace": "wrong_confirmation_namespace"}
            ),
            lambda item: item["final"]["execution_identity"].update(
                {"seed_cohort_hash": "c" * 64}
            ),
            lambda item: item["genesis"]["execution_identity"][
                "environment_source_preflight"
            ].update({"source_manifest_hash": "c" * 64}),
            lambda item: item["final"]["execution_identity"][
                "backend_execution_environment_verification"
            ].update(
                {
                    "expected_manifest_hash": "c" * 64,
                    "actual_manifest_hash": "c" * 64,
                }
            ),
            lambda item: item["final"]["execution_identity"].update(
                {"execution_device": "cpu"}
            ),
            lambda item: item.update({"reservation_hash": "d" * 64}),
        ):
            tampered = copy.deepcopy(payload)
            mutate(tampered)
            unsigned = {key: value for key, value in tampered.items() if key != "result_hash"}
            tampered["result_hash"] = canonical_json_hash(unsigned)
            with self.assertRaises(ValueError):
                StairFinalConfirmationResult.from_mapping(
                    tampered,
                    plan=plan,
                    reservation=reservation,
                    config=self.config,
                )

    def test_result_builder_rejects_wrong_actual_execution_identity(self) -> None:
        plan = build_stair_final_confirmation_plan(
            session_id="stair-final-execution-binding",
            config=self.config,
        )
        reservation = StairFinalConfirmationReservation(
            plan_hash=plan.plan_hash,
            session_id=plan.session_id,
            session_genesis_hash=SHA,
            ledger_checksum=CHECKSUM,
            genesis_checkpoint_hash=CHECKSUM,
            final_checkpoint_hash="sha256:" + "d" * 64,
            source_manifest_hash=self.source_manifest_hash,
            backend_execution_environment_manifest_hash=BACKEND_MANIFEST_HASH,
            execution_device=EXECUTION_DEVICE,
            final_anchor={"anchor_kind": "candidate", "round_id": "round-15"},
        )
        summary = self._uniform_summary(successes_per_cell=4)
        valid_final = self._execution(
            summary=summary,
            checkpoint_hash=reservation.final_checkpoint_hash,
        )
        wrong_identity_executions = (
            self._execution(
                summary=summary,
                checkpoint_hash="sha256:" + "e" * 64,
            ),
            self._execution(
                summary=summary,
                checkpoint_hash=reservation.genesis_checkpoint_hash,
                seed_namespace="formal_reference_candidate_eval",
                seed_cohort_hash="e" * 64,
            ),
            self._execution(
                summary=summary,
                checkpoint_hash=reservation.genesis_checkpoint_hash,
                seed_cohort_hash="e" * 64,
            ),
            self._execution(
                summary=summary,
                checkpoint_hash=reservation.genesis_checkpoint_hash,
                source_manifest_hash="e" * 64,
            ),
            self._execution(
                summary=summary,
                checkpoint_hash=reservation.genesis_checkpoint_hash,
                backend_manifest_hash="e" * 64,
            ),
            self._execution(
                summary=summary,
                checkpoint_hash=reservation.genesis_checkpoint_hash,
                execution_device="cpu",
            ),
        )
        for execution in wrong_identity_executions:
            with self.subTest(identity=execution.execution_identity()):
                with self.assertRaises(ValueError):
                    build_stair_final_confirmation_result(
                        plan=plan,
                        reservation=reservation,
                        config=self.config,
                        genesis_execution=execution,
                        final_execution=valid_final,
                    )

    def test_task_artifact_loaders_reject_oversize_before_json_parse(self) -> None:
        plan = build_stair_final_confirmation_plan(
            session_id="stair-final-bounded-load",
            config=self.config,
        )
        reservation = StairFinalConfirmationReservation(
            plan_hash=plan.plan_hash,
            session_id=plan.session_id,
            session_genesis_hash=SHA,
            ledger_checksum=CHECKSUM,
            genesis_checkpoint_hash=CHECKSUM,
            final_checkpoint_hash=CHECKSUM,
            source_manifest_hash=self.source_manifest_hash,
            backend_execution_environment_manifest_hash=BACKEND_MANIFEST_HASH,
            execution_device=EXECUTION_DEVICE,
            final_anchor={"anchor_kind": "reference"},
        )
        with tempfile.TemporaryDirectory() as tmp:
            plan_path = Path(tmp) / "plan.json"
            plan_path.write_bytes(b"x" * (MAX_STAIR_FINAL_CONFIRMATION_PLAN_BYTES + 1))
            with self.assertRaisesRegex(ValueError, "exceeds"):
                load_stair_final_confirmation_plan(plan_path)
            reservation_path = Path(tmp) / "reservation.json"
            reservation_path.write_bytes(
                b"x" * (MAX_STAIR_FINAL_CONFIRMATION_RESERVATION_BYTES + 1)
            )
            with self.assertRaisesRegex(ValueError, "exceeds"):
                load_stair_final_confirmation_reservation(
                    reservation_path,
                    expected_plan=plan,
                )
            result_path = Path(tmp) / "result.json"
            result_path.write_bytes(
                b"x" * (MAX_STAIR_FINAL_CONFIRMATION_RESULT_BYTES + 1)
            )
            with self.assertRaisesRegex(ValueError, "exceeds"):
                load_stair_final_confirmation_result(
                    result_path,
                    plan=plan,
                    reservation=reservation,
                    config=self.config,
                )

    def _execution(
        self,
        *,
        summary: dict[str, object],
        checkpoint_hash: str,
        seed_namespace: str = STAIR_FINAL_CONFIRMATION_EVAL_NAMESPACE,
        seed_cohort_hash: str | None = None,
        source_manifest_hash: str | None = None,
        backend_manifest_hash: str = BACKEND_MANIFEST_HASH,
        execution_device: str = EXECUTION_DEVICE,
    ) -> StairEvaluationExecutionResult:
        return StairEvaluationExecutionResult(
            summaries={
                STAIR_FINAL_CONFIRMATION_SPLIT: {"generalization": summary}
            },
            per_seed_outcomes={
                STAIR_FINAL_CONFIRMATION_SPLIT: (
                    (SimpleNamespace(episode_steps=1),),
                )
            },
            split_vector_env_step_calls={STAIR_FINAL_CONFIRMATION_SPLIT: 1},
            split_wall_times_s={STAIR_FINAL_CONFIRMATION_SPLIT: 0.1},
            checkpoint_hash=checkpoint_hash,
            seed_namespace=seed_namespace,
            seed_cohort_hash=(
                seed_cohort_hash
                or self.config.seed_resolver().namespace_cohort_hash(seed_namespace)
            ),
            source_manifest_preflight={
                "schema_version": STAIR_EXECUTION_SOURCE_PREFLIGHT_SCHEMA_VERSION,
                "status": "pass",
                "task_protocol_hash": self.config.task_protocol_hash,
                "source_manifest_hash": (
                    source_manifest_hash or self.source_manifest_hash
                ),
            },
            backend_execution_environment_verification={
                "schema_version": (
                    "true_rl_agent.backend_execution_environment_verification.v1"
                ),
                "backend_id": "isaac_lab_rsl_rl",
                "status": "pass",
                "expected_manifest_hash": backend_manifest_hash,
                "actual_manifest_hash": backend_manifest_hash,
            },
            execution_device=execution_device,
        )

    def _generation_reservation_v2(
        self,
        plan: StairFinalConfirmationPlanV2,
    ) -> StairFinalConfirmationReservationV2:
        terminal = GenerationTerminalBinding(
            identity_hash="1" * 64,
            required_generation_count=15,
            closed_generation_count=15,
            ledger_snapshot_hash="2" * 64,
            current_head_hash="3" * 64,
            lifecycle_head_hash="4" * 64,
            cost_chain_head_hash="5" * 64,
            segment_count=5,
        )
        return StairFinalConfirmationReservationV2(
            plan_hash=plan.plan_hash,
            session_id=plan.session_id,
            session_genesis_hash="6" * 64,
            generation_session_contract_hash="7" * 64,
            generation_cost_contract_hash="8" * 64,
            ledger_checksum="sha256:" + "9" * 64,
            terminal_binding=terminal.to_dict(),
            source_checkout_hash="a" * 64,
            backend_execution_environment_manifest_hash="b" * 64,
            gpu_process_binding=GpuProcessBinding(
                host_execution_device="cuda:0",
                process_execution_device="cuda:0",
                resource_binding_hash="c" * 64,
            ),
            genesis_job_id="generation-final-genesis-eval-sealed-validation",
            final_job_id="generation-final-current-eval-sealed-validation",
            genesis_eval_phase_spec_hash="d" * 64,
            final_eval_phase_spec_hash="e" * 64,
            genesis_evaluation_plan_hash="f" * 64,
            final_evaluation_plan_hash="0" * 64,
            genesis_checkpoint_hash="sha256:" + "1" * 64,
            final_checkpoint_hash="sha256:" + "2" * 64,
            genesis_source_manifest_hash="3" * 64,
            final_source_manifest_hash="4" * 64,
            genesis_training_binding_hash="5" * 64,
            final_training_binding_hash="6" * 64,
            genesis_capability_profile_hash="7" * 64,
            final_capability_profile_hash="8" * 64,
            final_anchor={"terminal_head_hash": terminal.current_head_hash},
        )

    def _uniform_summary(self, *, successes_per_cell: int) -> dict[str, object]:
        per_seed = []
        profile = self.config.evaluation_profile
        for _ in range(self.config.sealed_validation_seed_count):
            outcomes = []
            for env_index, cell in enumerate(profile.private_cell_assignment()):
                success = env_index % profile.episodes_per_seed_per_cell < successes_per_cell
                outcomes.append(
                    StairEpisodeOutcome(
                        env_index=env_index,
                        bin_id=cell.bin_id,
                        episode_steps=cell.episode_horizon_control_steps,
                        primary_reason="success" if success else "timeout",
                        success=success,
                        diagnostic_flags=(
                            ("first_valid_top_arrival_achieved",) if success else ()
                        ),
                        max_progress_m=cell.progress_goal_m if success else 0.5,
                        progress_fraction=1.0 if success else 0.1,
                        active_terminal_terms=() if success else ("time_out",),
                        evaluation_cell_id=cell.cell_id,
                    )
                )
            per_seed.append(outcomes)
        return build_stair_generalization_summary(
            profile=profile,
            semantics=self.config.evaluation_semantics,
            evaluation_protocol=self.config.evaluation_protocol(),
            split="sealed_validation",
            per_seed_outcomes=per_seed,
        )


if __name__ == "__main__":
    unittest.main()
