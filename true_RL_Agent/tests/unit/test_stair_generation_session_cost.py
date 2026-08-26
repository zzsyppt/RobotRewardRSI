from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from true_rl_agent.json_contract import canonical_json_bytes
from true_rl_agent.tasks.stair_climbing.config import StairClimbingTaskConfig
from true_rl_agent.tasks.stair_climbing.generation_session_cost import (
    DEVELOPMENT_GENERATION_PILOT_MODE,
    DEVELOPMENT_GENERATION_VALIDATION_MODE,
    FORMAL_GENERATION_SESSION_MODE,
    STAIR_GENERATION_FORMAL_RETRY_ALLOCATED_ACTIVE_GPU_HOURS,
    STAIR_GENERATION_FORMAL_RETRY_OUTPUT_BYTES,
    STAIR_GENERATION_FORMAL_SUCCESSFUL_ALLOCATED_ACTIVE_GPU_HOURS,
    STAIR_GENERATION_FORMAL_SUCCESSFUL_OUTPUT_BYTES,
    STAIR_GENERATION_SESSION_COST_SCHEMA_VERSION,
    STAIR_GENERATION_SESSION_COST_V2_SCHEMA_VERSION,
    STAIR_GENERATION_SESSION_COST_V3_SCHEMA_VERSION,
    STAIR_GENERATION_VALIDATION_RETRY_ALLOCATED_ACTIVE_GPU_HOURS,
    STAIR_GENERATION_VALIDATION_RETRY_OUTPUT_BYTES,
    STAIR_GENERATION_VALIDATION_SUCCESSFUL_ALLOCATED_ACTIVE_GPU_HOURS,
    STAIR_GENERATION_VALIDATION_SUCCESSFUL_OUTPUT_BYTES,
    StairGenerationActualCostJournal,
    StairGenerationActualCostReceiptRef,
    StairGenerationActualCostUsage,
    StairGenerationSessionCostContract,
    StairGenerationSessionCostContractV2,
    StairGenerationSessionCostContractV3,
    build_stair_generation_scientific_cost_basis,
    create_stair_generation_session_cost_contract,
    load_stair_generation_session_cost_contract,
    replay_stair_generation_formal_cost_chain,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
P6B_HISTORICAL_COST_CONTRACT_HASH = (
    "347f901d3b079399a085d7c47811451f9f6a782a07040a7839f704e2cff020aa"
)
CURRENT_V1_COST_CONTRACT_HASH = (
    "2beeea51e1bbf899cb0fb83f2c65c83ddb1c719a89426316d93bebdb6f19b46e"
)


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


def _validation_contract_v2() -> StairGenerationSessionCostContractV2:
    return StairGenerationSessionCostContractV2(
        session_mode=DEVELOPMENT_GENERATION_VALIDATION_MODE,
        max_generations=3,
        scientific_cost_basis=_scientific_cost_basis(),
        retry_train_jobs=1,
        retry_eval_seed_units=7,
        successful_allocated_active_gpu_hours=(
            STAIR_GENERATION_VALIDATION_SUCCESSFUL_ALLOCATED_ACTIVE_GPU_HOURS
        ),
        retry_allocated_active_gpu_hours=(
            STAIR_GENERATION_VALIDATION_RETRY_ALLOCATED_ACTIVE_GPU_HOURS
        ),
        successful_output_bytes=(STAIR_GENERATION_VALIDATION_SUCCESSFUL_OUTPUT_BYTES),
        retry_output_bytes=STAIR_GENERATION_VALIDATION_RETRY_OUTPUT_BYTES,
    )


def _formal_contract_v3() -> StairGenerationSessionCostContractV3:
    return StairGenerationSessionCostContractV3(
        session_mode=FORMAL_GENERATION_SESSION_MODE,
        max_generations=15,
        scientific_cost_basis=_scientific_cost_basis(),
        retry_train_jobs=1,
        retry_eval_seed_units=7,
        successful_allocated_active_gpu_hours=(
            STAIR_GENERATION_FORMAL_SUCCESSFUL_ALLOCATED_ACTIVE_GPU_HOURS
        ),
        retry_allocated_active_gpu_hours=(
            STAIR_GENERATION_FORMAL_RETRY_ALLOCATED_ACTIVE_GPU_HOURS
        ),
        successful_output_bytes=STAIR_GENERATION_FORMAL_SUCCESSFUL_OUTPUT_BYTES,
        retry_output_bytes=STAIR_GENERATION_FORMAL_RETRY_OUTPUT_BYTES,
    )


class StairGenerationSessionCostTests(unittest.TestCase):
    def test_formal_cost_replay_allows_exactly_one_terminal_settlement(self) -> None:
        contract = _formal_contract_v3()
        generation_head = "1" * 64
        final_head = "2" * 64
        generation = StairGenerationActualCostReceiptRef(
            path=Path("generation.json"),
            payload_hash=generation_head,
            payload={
                "event_id": "generation-015-settlement",
                "previous_cost_chain_head_hash": contract.contract_hash,
            },
        )
        final = StairGenerationActualCostReceiptRef(
            path=Path("final.json"),
            payload_hash=final_head,
            payload={
                "event_id": "final-confirmation-settlement",
                "previous_cost_chain_head_hash": generation_head,
            },
        )
        lifecycle = SimpleNamespace(
            terminal=True,
            cost_chain_head_hash=generation_head,
            recovery_cost_edges=(),
            segment_count=5,
            active_segment_id=None,
        )

        self.assertEqual(
            replay_stair_generation_formal_cost_chain(
                contract=contract,
                actual_cost_receipts=(generation, final),
                lifecycle=lifecycle,
                require_finalization_settlement=True,
            ),
            final_head,
        )
        with self.assertRaisesRegex(ValueError, "lacks finalization settlement"):
            replay_stair_generation_formal_cost_chain(
                contract=contract,
                actual_cost_receipts=(generation,),
                lifecycle=lifecycle,
                require_finalization_settlement=True,
            )
        bad_final = StairGenerationActualCostReceiptRef(
            path=Path("bad-final.json"),
            payload_hash="3" * 64,
            payload={
                "event_id": "unexpected-settlement",
                "previous_cost_chain_head_hash": generation_head,
            },
        )
        with self.assertRaisesRegex(ValueError, "finalization cost edge"):
            replay_stair_generation_formal_cost_chain(
                contract=contract,
                actual_cost_receipts=(generation, bad_final),
                lifecycle=lifecycle,
                require_finalization_settlement=False,
            )

    def test_formal_v3_exact_totals_operation_evidence_and_round_trip(self) -> None:
        contract = _formal_contract_v3()
        payload = contract.to_dict()

        self.assertEqual(
            payload["schema_version"],
            STAIR_GENERATION_SESSION_COST_V3_SCHEMA_VERSION,
        )
        self.assertEqual(
            (
                payload["successful_session"]["train_jobs"],
                payload["successful_session"]["train_steps"],
                payload["successful_session"]["eval_seed_units"],
                payload["successful_session"]["episodes"],
                payload["successful_session"]["simulated_control_steps"],
            ),
            (31, 6_094_848_000, 377, 105_560, 6_169_795_600),
        )
        self.assertEqual(
            payload["successful_session"]["final_confirmation_seed_units"],
            14,
        )
        self.assertEqual(
            payload["physical_cap"],
            {
                "train_jobs": 32,
                "train_steps": 6_291_456_000,
                "eval_seed_units": 384,
                "episodes": 107_520,
                "allocated_active_gpu_hours": 50.5,
                "simulated_control_steps": 6_367_795_200,
                "output_bytes": 14_495_514_624,
                "conservative_eval_control_steps_per_seed_unit": 198_800,
            },
        )
        self.assertEqual(
            payload["provider"]["attempt_cap"],
            150,
        )
        self.assertEqual(payload["provider"]["input_unit_cap"], 45_000_000)
        self.assertEqual(payload["provider"]["output_unit_cap"], 4_915_200)
        self.assertEqual(payload["provider"]["total_unit_cap"], 49_915_200)
        operation = payload["operation_reserves"]
        self.assertEqual(
            operation["source"],
            {
                "session_id": "p6c-415bfad-20260823T042422Z",
                "source_commit": "415bfad6eb6e85c6cf0cf94851c6b39d1067116b",
                "g4_audit_sha256": (
                    "ad691299c0ca5b08383f984a68f514d987c1831cdd3b24d367a9ca34d7266a93"
                ),
            },
        )
        self.assertEqual(
            operation["totals"]["successful_session"][
                "reserved_allocated_gpu_capacity_ns"
            ],
            175_397_494_998_204,
        )
        self.assertEqual(
            operation["totals"]["successful_session"]["reserved_output_bytes"],
            12_957_346_478,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = create_stair_generation_session_cost_contract(
                Path(tmp),
                contract,
            )
            loaded = load_stair_generation_session_cost_contract(path)
            self.assertIs(type(loaded), StairGenerationSessionCostContractV3)
            self.assertEqual(loaded, contract)

        tampered = contract.to_dict()
        tampered["successful_session"]["final_confirmation_seed_units"] = 0
        with self.assertRaisesRegex(ValueError, "v3 projection mismatch"):
            StairGenerationSessionCostContractV3.from_mapping(tampered)

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

    def test_current_v1_cost_contract_tracks_current_task_identity(
        self,
    ) -> None:
        contract = StairGenerationSessionCostContract(
            session_mode=DEVELOPMENT_GENERATION_PILOT_MODE,
            max_generations=1,
            scientific_cost_basis=_scientific_cost_basis(),
            retry_train_jobs=1,
            retry_eval_seed_units=7,
            max_allocated_active_gpu_hours=8.0,
            max_simulated_control_steps=795_974_400,
            max_output_bytes=8 * 1024**3,
        )
        self.assertEqual(
            contract.schema_version,
            STAIR_GENERATION_SESSION_COST_SCHEMA_VERSION,
        )
        self.assertEqual(contract.contract_hash, CURRENT_V1_COST_CONTRACT_HASH)
        self.assertNotEqual(
            contract.contract_hash,
            P6B_HISTORICAL_COST_CONTRACT_HASH,
        )
        self.assertEqual(
            set(StairGenerationSessionCostContract.__dataclass_fields__),
            {
                "session_mode",
                "max_generations",
                "scientific_cost_basis",
                "retry_train_jobs",
                "retry_eval_seed_units",
                "max_allocated_active_gpu_hours",
                "max_simulated_control_steps",
                "max_output_bytes",
                "schema_version",
            },
        )
        encoded = canonical_json_bytes(contract.to_dict(), trailing_newline=True)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "generation_session_cost.json"
            path.write_bytes(encoded)
            loaded = load_stair_generation_session_cost_contract(path)
            self.assertIs(type(loaded), StairGenerationSessionCostContract)
            self.assertEqual(loaded, contract)
            self.assertEqual(path.read_bytes(), encoded)

    def test_validation_v2_exact_projection_round_trip_and_physical_derivation(
        self,
    ) -> None:
        contract = _validation_contract_v2()
        payload = contract.to_dict()

        self.assertEqual(
            payload["schema_version"],
            STAIR_GENERATION_SESSION_COST_V2_SCHEMA_VERSION,
        )
        self.assertEqual(
            payload["successful_session"],
            {
                "reference_qualification": {
                    "train_jobs": 1,
                    "train_steps": 196_608_000,
                    "eval_seed_units": 15,
                },
                "parent_public_dev_bootstrap_seed_units": 3,
                "generation": {
                    "train_jobs": 6,
                    "train_steps": 1_179_648_000,
                    "eval_seed_units": 69,
                },
                "final_confirmation_seed_units": 0,
                "train_jobs": 7,
                "train_steps": 1_376_256_000,
                "eval_seed_units": 87,
                "episodes": 24_360,
                "simulated_control_steps": 1_393_551_600,
                "allocated_active_gpu_hours": 11.0,
                "output_bytes": 3_221_225_472,
            },
        )
        self.assertEqual(
            payload["retry_reserve"],
            {
                "train_jobs": 1,
                "train_steps": 196_608_000,
                "eval_seed_units": 7,
                "episodes": 1_960,
                "simulated_control_steps": 197_999_600,
                "allocated_active_gpu_hours": 1.5,
                "output_bytes": 536_870_912,
            },
        )
        self.assertEqual(
            payload["physical_cap"],
            {
                "train_jobs": 8,
                "train_steps": 1_572_864_000,
                "eval_seed_units": 94,
                "episodes": 26_320,
                "allocated_active_gpu_hours": 12.5,
                "simulated_control_steps": 1_591_551_200,
                "output_bytes": 3_758_096_384,
                "conservative_eval_control_steps_per_seed_unit": 198_800,
            },
        )
        self.assertEqual(
            payload["provider"],
            {
                "generation_width": 2,
                "attempts_per_lane_per_generation": 5,
                "attempt_cap": 30,
                "input_units_per_attempt_reserve": 300_000,
                "output_units_per_attempt_cap": 32_768,
                "input_unit_cap": 9_000_000,
                "output_unit_cap": 983_040,
                "total_unit_cap": 9_983_040,
                "money_governor": "waived_by_user_20260820",
            },
        )
        self.assertEqual(
            payload["physical_cap"]["train_jobs"],
            payload["successful_session"]["train_jobs"]
            + payload["retry_reserve"]["train_jobs"],
        )
        self.assertEqual(
            payload["physical_cap"]["simulated_control_steps"],
            payload["successful_session"]["simulated_control_steps"]
            + payload["retry_reserve"]["simulated_control_steps"],
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = create_stair_generation_session_cost_contract(
                Path(tmp),
                contract,
            )
            loaded = load_stair_generation_session_cost_contract(path)
            self.assertIs(type(loaded), StairGenerationSessionCostContractV2)
            self.assertEqual(loaded, contract)

    def test_v2_tamper_and_cross_schema_projection_are_rejected(self) -> None:
        contract = _validation_contract_v2()
        tampered = contract.to_dict()
        tampered["physical_cap"]["train_jobs"] += 1
        with self.assertRaisesRegex(ValueError, "v2 projection mismatch"):
            StairGenerationSessionCostContractV2.from_mapping(tampered)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "generation_session_cost.json"
            mixed = contract.to_dict()
            mixed["schema_version"] = STAIR_GENERATION_SESSION_COST_SCHEMA_VERSION
            path.write_bytes(canonical_json_bytes(mixed, trailing_newline=True))
            with self.assertRaisesRegex(ValueError, "projection mismatch"):
                load_stair_generation_session_cost_contract(path)

            mixed = _contract().to_dict()
            mixed["schema_version"] = STAIR_GENERATION_SESSION_COST_V2_SCHEMA_VERSION
            path.write_bytes(canonical_json_bytes(mixed, trailing_newline=True))
            with self.assertRaises(ValueError):
                load_stair_generation_session_cost_contract(path)

            mixed["schema_version"] = "true_rl_agent.unknown.v1"
            path.write_bytes(canonical_json_bytes(mixed, trailing_newline=True))
            with self.assertRaisesRegex(ValueError, "schema is unsupported"):
                load_stair_generation_session_cost_contract(path)

    def test_v2_actual_cost_journal_uses_single_session_physical_cap(self) -> None:
        contract = _validation_contract_v2()
        with tempfile.TemporaryDirectory() as tmp:
            journal = StairGenerationActualCostJournal(
                Path(tmp),
                contract=contract,
            )
            reservation = journal.publish_reservation()
            settled = journal.publish_snapshot(
                event_id="session-physical-cap",
                previous_cost_chain_head_hash=reservation.payload_hash,
                usage=StairGenerationActualCostUsage(
                    provider_attempts=30,
                    provider_request_units=9_000_000,
                    provider_response_units=983_040,
                    provider_total_units=9_983_040,
                    train_jobs=8,
                    train_steps=1_572_864_000,
                    eval_seed_units=94,
                    simulated_control_steps=1_591_551_200,
                    resource_lease_wall_duration_ns=45_000_000_000_000,
                ),
                finalization_output_reserve_bytes=0,
            )

            self.assertEqual(
                settled.payload["reservation"]["allocated_gpu_capacity_hours_cap"],
                12.5,
            )
            self.assertTrue(settled.payload["cap_checks"]["all_passed"])

            exceeded = journal.publish_snapshot(
                event_id="session-physical-cap-exceeded",
                previous_cost_chain_head_hash=settled.payload_hash,
                usage=StairGenerationActualCostUsage(
                    train_jobs=9,
                    train_steps=1_572_864_001,
                    eval_seed_units=95,
                    simulated_control_steps=1_591_551_201,
                ),
                finalization_output_reserve_bytes=0,
            )
            self.assertFalse(exceeded.payload["cap_checks"]["all_passed"])

    def test_actual_cost_usage_schema_remains_v1_exact(self) -> None:
        self.assertEqual(
            tuple(StairGenerationActualCostUsage.__dataclass_fields__),
            (
                "provider_attempts",
                "provider_request_units",
                "provider_response_units",
                "provider_total_units",
                "provider_reasoning_units",
                "provider_conservative_attempts",
                "job_attempts",
                "train_jobs",
                "train_steps",
                "eval_seed_units",
                "simulated_control_steps",
                "resource_lease_wall_duration_ns",
                "unmeasured_bound_attempts",
                "observed_output_bytes",
                "observed_output_files",
                "job_cost_evidence_hash",
            ),
        )

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
