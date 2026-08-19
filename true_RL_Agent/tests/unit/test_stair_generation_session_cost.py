from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from true_rl_agent.tasks.stair_climbing.config import StairClimbingTaskConfig
from true_rl_agent.tasks.stair_climbing.generation_session_cost import (
    DEVELOPMENT_GENERATION_PILOT_MODE,
    DEVELOPMENT_GENERATION_VALIDATION_MODE,
    FORMAL_GENERATION_SESSION_MODE,
    StairGenerationActualCostJournal,
    StairGenerationActualCostUsage,
    StairGenerationSessionCostContract,
    build_stair_generation_scientific_cost_basis,
    create_stair_generation_session_cost_contract,
    load_stair_generation_session_cost_contract,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _scientific_cost_basis():
    config = StairClimbingTaskConfig.defaults(
        PROJECT_ROOT / "configs" / "tasks" / "stair_protocol_v2.json"
    )
    return build_stair_generation_scientific_cost_basis(
        session_id="cost-contract-test",
        config=config,
    )


def _contract(
    mode: str = DEVELOPMENT_GENERATION_PILOT_MODE,
    generations: int = 1,
    *,
    retry_train_jobs: int = 1,
    retry_eval_seed_units: int = 7,
) -> StairGenerationSessionCostContract:
    return StairGenerationSessionCostContract(
        session_mode=mode,
        max_generations=generations,
        scientific_cost_basis=_scientific_cost_basis(),
        retry_train_jobs=retry_train_jobs,
        retry_eval_seed_units=retry_eval_seed_units,
        max_allocated_active_gpu_hours=12.0,
        max_simulated_control_steps=850_000_000,
        max_output_bytes=16 * 1024**3,
    )


class StairGenerationSessionCostTests(unittest.TestCase):
    def test_scientific_cost_basis_projects_existing_stair_contracts(self) -> None:
        basis = _scientific_cost_basis()

        self.assertEqual(basis.train_steps_per_job, 196_608_000)
        self.assertEqual(basis.reference_eval_seed_units, 15)
        self.assertEqual(basis.parent_bootstrap_seed_units, 3)
        self.assertEqual(basis.eval_seed_units_per_generation, 23)
        self.assertEqual(basis.episodes_per_seed_unit, 280)
        self.assertEqual(basis.max_eval_control_steps_per_seed_unit, 198_800)
        self.assertEqual(basis.final_confirmation_seed_units, 14)

    def test_pilot_counts_include_genesis_bootstrap_and_retry(self) -> None:
        contract = _contract()
        self.assertEqual(contract.successful_train_jobs, 3)
        self.assertEqual(contract.successful_train_steps, 589_824_000)
        self.assertEqual(contract.successful_eval_seed_units, 41)
        self.assertEqual(contract.physical_train_job_cap, 4)
        self.assertEqual(contract.physical_train_step_cap, 786_432_000)
        self.assertEqual(contract.physical_eval_seed_unit_cap, 48)
        self.assertEqual(contract.to_dict()["successful_session"]["episodes"], 11_480)
        self.assertEqual(contract.to_dict()["physical_cap"]["episodes"], 13_440)

    def test_three_and_fifteen_generation_success_totals_are_exact(self) -> None:
        three = _contract(
            DEVELOPMENT_GENERATION_VALIDATION_MODE,
            3,
            retry_train_jobs=0,
            retry_eval_seed_units=0,
        )
        formal = _contract(
            FORMAL_GENERATION_SESSION_MODE,
            15,
            retry_train_jobs=0,
            retry_eval_seed_units=0,
        )
        self.assertEqual(
            (
                three.successful_train_jobs,
                three.successful_train_steps,
                three.successful_eval_seed_units,
            ),
            (7, 1_376_256_000, 87),
        )
        self.assertEqual(
            (
                formal.successful_train_jobs,
                formal.successful_train_steps,
                formal.successful_eval_seed_units,
            ),
            (31, 6_094_848_000, 377),
        )

    def test_mode_generation_mismatch_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not match session mode"):
            _contract(DEVELOPMENT_GENERATION_PILOT_MODE, 3)

    def test_create_once_load_and_tamper_are_strict(self) -> None:
        contract = _contract()
        with tempfile.TemporaryDirectory() as tmp:
            path = create_stair_generation_session_cost_contract(
                Path(tmp),
                contract,
            )
            self.assertEqual(
                load_stair_generation_session_cost_contract(path),
                contract,
            )
            payload = contract.to_dict()
            payload["physical_cap"]["train_jobs"] += 1
            with self.assertRaisesRegex(ValueError, "projection mismatch"):
                StairGenerationSessionCostContract.from_mapping(payload)

    def test_actual_cost_chain_reserves_then_settles_measured_usage(self) -> None:
        contract = _contract()
        with tempfile.TemporaryDirectory() as tmp:
            journal = StairGenerationActualCostJournal(
                Path(tmp),
                contract=contract,
            )
            reservation = journal.publish_reservation()
            usage = StairGenerationActualCostUsage(
                provider_attempts=2,
                provider_request_units=400_000,
                provider_response_units=30_000,
                provider_total_units=430_000,
                provider_reasoning_units=20_000,
                job_attempts=8,
                train_jobs=3,
                train_steps=589_824_000,
                eval_seed_units=41,
                simulated_control_steps=598_000_000,
                resource_lease_wall_duration_ns=7_200_000_000_000,
                observed_output_bytes=1024,
                observed_output_files=8,
            )
            settled = journal.publish_snapshot(
                event_id="segment-abc-settled",
                previous_cost_chain_head_hash=reservation.payload_hash,
                usage=usage,
            )

            self.assertTrue(settled.payload["cap_checks"]["all_passed"])
            self.assertEqual(len(journal.load_chain()), 2)
            self.assertEqual(
                settled.payload["previous_task_cost_receipt_hash"],
                reservation.payload_hash,
            )
            self.assertEqual(
                journal.publish_snapshot(
                    event_id="segment-abc-settled",
                    previous_cost_chain_head_hash=reservation.payload_hash,
                    usage=usage,
                ).payload_hash,
                settled.payload_hash,
            )
            with self.assertRaisesRegex(
                ValueError,
                "differs from recomputation",
            ):
                journal.publish_snapshot(
                    event_id="segment-abc-settled",
                    previous_cost_chain_head_hash=reservation.payload_hash,
                    usage=StairGenerationActualCostUsage(
                        provider_attempts=3,
                    ),
                )

    def test_unmeasured_bound_attempt_blocks_actual_cost_closure(self) -> None:
        contract = _contract()
        with tempfile.TemporaryDirectory() as tmp:
            journal = StairGenerationActualCostJournal(
                Path(tmp),
                contract=contract,
            )
            reservation = journal.publish_reservation()
            settled = journal.publish_snapshot(
                event_id="segment-abc-failed",
                previous_cost_chain_head_hash=reservation.payload_hash,
                usage=StairGenerationActualCostUsage(
                    unmeasured_bound_attempts=1,
                ),
                conservative_charges=("unmeasured-bound-attempt:test:0",),
            )

            self.assertFalse(settled.payload["cap_checks"]["all_passed"])
            self.assertFalse(
                settled.payload["cap_checks"]["resource_lease_measurement_complete"]
            )


if __name__ == "__main__":
    unittest.main()
