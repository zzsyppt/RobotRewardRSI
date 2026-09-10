from __future__ import annotations

import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from true_rl_agent.runner.base import RunnerFailureKind
from true_rl_agent.runner.gpu_lease import GpuDeviceIdentity, GpuLeaseManager
from true_rl_agent.runner.job_attempts import (
    JobAttemptJournal,
    JobAttemptScope,
    PreGenesisExecutionScope,
)
from true_rl_agent.runner.subprocess_runner import SubprocessRunner
from true_rl_agent.agent_core import generation_operations as session_module
from true_rl_agent.tasks.stair_climbing.generation_history import _retry_policy
from true_rl_agent.tasks.stair_climbing.config import StairClimbingTaskConfig
from true_rl_agent.tasks.stair_climbing.generation_session_cost import (
    DEVELOPMENT_GENERATION_PILOT_MODE,
    DEVELOPMENT_GENERATION_VALIDATION_MODE,
    FORMAL_GENERATION_SESSION_MODE,
    STAIR_GENERATION_FORMAL_RETRY_ALLOCATED_ACTIVE_GPU_HOURS,
    STAIR_GENERATION_FORMAL_RETRY_OUTPUT_BYTES,
    STAIR_GENERATION_FORMAL_SUCCESSFUL_ALLOCATED_ACTIVE_GPU_HOURS,
    STAIR_GENERATION_FORMAL_SUCCESSFUL_OUTPUT_BYTES,
    STAIR_GENERATION_FINALIZATION_OUTPUT_RESERVE_BYTES,
    STAIR_GENERATION_IMPORTED_REFERENCE_VALIDATION_RETRY_ALLOCATED_ACTIVE_GPU_HOURS,
    STAIR_GENERATION_IMPORTED_REFERENCE_VALIDATION_RETRY_OUTPUT_BYTES,
    STAIR_GENERATION_IMPORTED_REFERENCE_VALIDATION_SUCCESSFUL_ALLOCATED_ACTIVE_GPU_HOURS,
    STAIR_GENERATION_IMPORTED_REFERENCE_VALIDATION_SUCCESSFUL_OUTPUT_BYTES,
    STAIR_GENERATION_IMPORTED_REFERENCE_FORMAL_RETRY_ALLOCATED_ACTIVE_GPU_HOURS,
    STAIR_GENERATION_IMPORTED_REFERENCE_FORMAL_RETRY_OUTPUT_BYTES,
    STAIR_GENERATION_IMPORTED_REFERENCE_FORMAL_SUCCESSFUL_ALLOCATED_ACTIVE_GPU_HOURS,
    STAIR_GENERATION_IMPORTED_REFERENCE_FORMAL_SUCCESSFUL_OUTPUT_BYTES,
    STAIR_GENERATION_VALIDATION_RETRY_ALLOCATED_ACTIVE_GPU_HOURS,
    STAIR_GENERATION_VALIDATION_RETRY_OUTPUT_BYTES,
    STAIR_GENERATION_VALIDATION_SUCCESSFUL_ALLOCATED_ACTIVE_GPU_HOURS,
    STAIR_GENERATION_VALIDATION_SUCCESSFUL_OUTPUT_BYTES,
    StairGenerationSessionCostContract,
    StairGenerationSessionCostContractV2,
    StairGenerationSessionCostContractV3,
    StairGenerationSessionCostContractV4,
    StairGenerationSessionCostContractV5,
    build_stair_generation_scientific_cost_basis,
    stair_generation_operation_cost_reserves,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SESSION_ID = "operation-admission-test"
SESSION_CONTRACT_HASH = "a" * 64
GENESIS_IDENTITY_HASH = "b" * 64
JOB_INPUT_HASH = "c" * 64
ENVIRONMENT_HASH = "d" * 64
OUTPUT_HASH = "e" * 64
RECOVERY_HASH = "f" * 64


def _scientific_cost_basis():
    config = StairClimbingTaskConfig.defaults(
        PROJECT_ROOT / "configs" / "tasks" / "stair_protocol_v2.json"
    )
    return build_stair_generation_scientific_cost_basis(
        session_id=SESSION_ID,
        config=config,
    )


def _cost_contract() -> StairGenerationSessionCostContractV2:
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
        successful_output_bytes=STAIR_GENERATION_VALIDATION_SUCCESSFUL_OUTPUT_BYTES,
        retry_output_bytes=STAIR_GENERATION_VALIDATION_RETRY_OUTPUT_BYTES,
    )


def _formal_cost_contract() -> StairGenerationSessionCostContractV3:
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


def _imported_reference_cost_contract() -> StairGenerationSessionCostContractV4:
    return StairGenerationSessionCostContractV4(
        session_mode=DEVELOPMENT_GENERATION_VALIDATION_MODE,
        max_generations=3,
        scientific_cost_basis=replace(
            _scientific_cost_basis(),
            parent_bootstrap_seed_units=0,
        ),
        retry_train_jobs=0,
        retry_eval_seed_units=7,
        successful_allocated_active_gpu_hours=(
            STAIR_GENERATION_IMPORTED_REFERENCE_VALIDATION_SUCCESSFUL_ALLOCATED_ACTIVE_GPU_HOURS
        ),
        retry_allocated_active_gpu_hours=(
            STAIR_GENERATION_IMPORTED_REFERENCE_VALIDATION_RETRY_ALLOCATED_ACTIVE_GPU_HOURS
        ),
        successful_output_bytes=(
            STAIR_GENERATION_IMPORTED_REFERENCE_VALIDATION_SUCCESSFUL_OUTPUT_BYTES
        ),
        retry_output_bytes=(
            STAIR_GENERATION_IMPORTED_REFERENCE_VALIDATION_RETRY_OUTPUT_BYTES
        ),
        candidate_train_process_timeout_s=7_200,
        termination_cleanup_margin_s=16,
    )


def _imported_reference_formal_cost_contract(
) -> StairGenerationSessionCostContractV5:
    return StairGenerationSessionCostContractV5(
        session_mode=FORMAL_GENERATION_SESSION_MODE,
        max_generations=15,
        scientific_cost_basis=replace(
            _scientific_cost_basis(),
            parent_bootstrap_seed_units=0,
        ),
        retry_train_jobs=0,
        retry_eval_seed_units=7,
        successful_allocated_active_gpu_hours=(
            STAIR_GENERATION_IMPORTED_REFERENCE_FORMAL_SUCCESSFUL_ALLOCATED_ACTIVE_GPU_HOURS
        ),
        retry_allocated_active_gpu_hours=(
            STAIR_GENERATION_IMPORTED_REFERENCE_FORMAL_RETRY_ALLOCATED_ACTIVE_GPU_HOURS
        ),
        successful_output_bytes=(
            STAIR_GENERATION_IMPORTED_REFERENCE_FORMAL_SUCCESSFUL_OUTPUT_BYTES
        ),
        retry_output_bytes=(
            STAIR_GENERATION_IMPORTED_REFERENCE_FORMAL_RETRY_OUTPUT_BYTES
        ),
        candidate_train_process_timeout_s=7_200,
        termination_cleanup_margin_s=16,
    )


def _pre_genesis_scope(
    root: Path,
    *,
    contract: StairGenerationSessionCostContractV2,
    resource_binding_hash: str,
    retries: int,
) -> PreGenesisExecutionScope:
    return PreGenesisExecutionScope(
        journal_root=root / "job_attempts",
        session_id=SESSION_ID,
        genesis_identity_hash=GENESIS_IDENTITY_HASH,
        session_contract_hash=SESSION_CONTRACT_HASH,
        lane_id="lane-a",
        invocation_segment_id="session-genesis",
        budget_contract_hash=contract.contract_hash,
        attempt_budget_units=retries + 1,
        max_infrastructure_retries=retries,
        resource_binding_hash=resource_binding_hash,
    )


def _generation_scope(
    root: Path,
    *,
    contract: StairGenerationSessionCostContractV2,
    resource_binding_hash: str,
    lane_id: str,
    retries: int,
    generation_index: int = 1,
) -> JobAttemptScope:
    return JobAttemptScope(
        journal_root=root / "job_attempts",
        session_genesis_hash="1" * 64,
        session_contract_hash=SESSION_CONTRACT_HASH,
        generation_id=f"{SESSION_ID}-generation-{generation_index:03d}",
        lane_id=lane_id,
        invocation_segment_id="segment-000",
        budget_contract_hash=contract.contract_hash,
        attempt_budget_units=retries + 1,
        max_infrastructure_retries=retries,
        resource_binding_hash=resource_binding_hash,
    )


def _final_confirmation_scope(
    root: Path,
    *,
    contract: StairGenerationSessionCostContractV3,
    resource_binding_hash: str,
    retries: int = 1,
) -> JobAttemptScope:
    return JobAttemptScope(
        journal_root=root / "job_attempts",
        session_genesis_hash="1" * 64,
        session_contract_hash=SESSION_CONTRACT_HASH,
        generation_id="generation-final-confirmation",
        lane_id="lane-a",
        invocation_segment_id="final-confirmation",
        budget_contract_hash=contract.contract_hash,
        attempt_budget_units=retries + 1,
        max_infrastructure_retries=retries,
        resource_binding_hash=resource_binding_hash,
    )


def _journal(
    *,
    scope: JobAttemptScope | PreGenesisExecutionScope,
    logical_job_id: str,
    job_kind: str,
) -> JobAttemptJournal:
    return JobAttemptJournal(
        scope=scope,
        logical_job_id=logical_job_id,
        job_kind=job_kind,
        job_input_hash=JOB_INPUT_HASH,
        environment_hash=ENVIRONMENT_HASH,
        execution_device="cuda:0",
        resource_binding_hash=scope.resource_binding_hash,
    )


class _AttemptSpec:
    def __init__(self, scope: PreGenesisExecutionScope) -> None:
        self.job_id = f"{SESSION_ID}-train"
        self.execution_device = "cuda:0"
        self.attempt_scope = scope
        self.metadata = {"backend_id": "test"}

    def to_public_dict(self) -> dict[str, str]:
        return {"job_id": self.job_id, "execution_device": self.execution_device}


class StairGenerationOperationAdmissionTests(unittest.TestCase):
    def test_retry_policy_derives_imported_reference_train_reserve_from_cost(self) -> None:
        legacy = _retry_policy(_cost_contract())
        imported = _retry_policy(_imported_reference_cost_contract())

        self.assertEqual(legacy["reference_train_retry_entitlement"], 1)
        self.assertEqual(legacy["retry_train_job_reserve"], 1)
        self.assertEqual(imported["reference_train_retry_entitlement"], 0)
        self.assertEqual(imported["retry_train_job_reserve"], 0)
        self.assertEqual(imported["retry_eval_seed_unit_reserve"], 7)

    def test_formal_retry_policy_binds_session_wide_sealed_roles(self) -> None:
        for contract in (
            _formal_cost_contract(),
            _imported_reference_formal_cost_contract(),
        ):
            policy = _retry_policy(contract)
            self.assertEqual(
                policy["schema_version"],
                "true_rl_agent.stair.generation_retry_policy.v2",
            )
            self.assertEqual(
                policy["session_wide_sealed_validation_retry_entitlement"],
                1,
            )
            self.assertEqual(
                policy["eligible_sealed_validation_roles"],
                [
                    "generation_winner",
                    "final_confirmation_genesis",
                    "final_confirmation_final",
                ],
            )
            self.assertNotIn(
                "winner_sealed_validation_retry_entitlement",
                policy,
            )

    def test_imported_reference_v4_uses_protocol_timeout_for_candidate_train(
        self,
    ) -> None:
        contract = _imported_reference_cost_contract()
        slots = session_module._strict_success_operation_slots(contract)
        self.assertEqual(len(slots), 21)
        self.assertFalse(any(slot.startswith("reference_") for slot in slots))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            identity = GpuDeviceIdentity("cuda:0", "machine", "GPU-TEST")
            scope = _generation_scope(
                root,
                contract=contract,
                resource_binding_hash=identity.identity_hash,
                lane_id="lane-a",
                retries=0,
            )
            admission = session_module.OperationAdmission(
                output_dir=root,
                session_id=SESSION_ID,
                cost_contract=contract,
                session_contract_hash=SESSION_CONTRACT_HASH,
                termination_grace_s=5,
            )
            logical_train_id = f"{SESSION_ID}-generation-001-lane-a-train"
            self.assertIsNone(admission(scope, logical_train_id, "train", 0))

            runner = SubprocessRunner(
                allowed_executables=(sys.executable,),
                timeout_s=7_200,
                attempt_admission_hook=admission,
            )
            self.assertEqual(
                runner._admit_attempt(
                    scope=scope,
                    logical_job_id=logical_train_id,
                    job_kind="train",
                    attempt_index=0,
                ),
                7_200,
            )
            eval_timeout = admission(
                scope,
                f"{SESSION_ID}-generation-001-lane-a-eval-public-dev",
                "eval_phase",
                0,
            )
            self.assertIsInstance(eval_timeout, int)
            self.assertGreater(eval_timeout, 0)
            self.assertLess(eval_timeout, 7_200)

            pre_genesis = _pre_genesis_scope(
                root,
                contract=contract,
                resource_binding_hash=identity.identity_hash,
                retries=0,
            )
            with self.assertRaisesRegex(ValueError, "cannot execute pre-Genesis"):
                admission(
                    pre_genesis,
                    f"{SESSION_ID}-train",
                    "train",
                    0,
                )

    def test_formal_v3_admits_generation_fifteen_and_final_confirmation(self) -> None:
        contract = _formal_cost_contract()
        slots = session_module._strict_success_operation_slots(contract)
        self.assertEqual(len(slots), 110)
        self.assertIn("generation-015:lane-b:public_accept", slots)
        self.assertIn("final-confirmation:genesis", slots)
        self.assertIn("final-confirmation:final", slots)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            identity = GpuDeviceIdentity("cuda:0", "machine", "GPU-TEST")
            generation_scope = _generation_scope(
                root,
                contract=contract,
                resource_binding_hash=identity.identity_hash,
                lane_id="lane-b",
                retries=0,
                generation_index=15,
            )
            slot, _ = session_module.read_operation_slot(
                session_id=SESSION_ID,
                scope=generation_scope,
                logical_job_id=(
                    f"{SESSION_ID}-generation-015-lane-b-eval-public-accept"
                ),
                job_kind="eval_phase",
                attempt_index=0,
                cost_contract=contract,
            )
            self.assertEqual(slot, "generation-015:lane-b:public_accept")

            final_scope = _final_confirmation_scope(
                root,
                contract=contract,
                resource_binding_hash=identity.identity_hash,
            )
            admission = session_module.OperationAdmission(
                output_dir=root,
                session_id=SESSION_ID,
                cost_contract=contract,
                session_contract_hash=SESSION_CONTRACT_HASH,
                termination_grace_s=5,
            )
            timeout_s = admission(
                final_scope,
                f"{SESSION_ID}-final-confirmation-final-eval-sealed-validation",
                "eval_phase",
                0,
            )
            self.assertGreater(timeout_s, 0)

    def test_imported_reference_formal_v5_has_only_executable_slots(self) -> None:
        contract = _imported_reference_formal_cost_contract()
        slots = session_module._strict_success_operation_slots(contract)
        self.assertEqual(len(slots), 92)
        self.assertFalse(any(slot.startswith("reference_") for slot in slots))
        self.assertNotIn("generation-015:lane-b:public_accept", slots)
        self.assertIn("generation-015:selected:public_accept", slots)
        self.assertIn("final-confirmation:genesis", slots)
        self.assertIn("final-confirmation:final", slots)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            identity = GpuDeviceIdentity("cuda:0", "machine", "GPU-TEST")
            generation_scope = _generation_scope(
                root,
                contract=contract,
                resource_binding_hash=identity.identity_hash,
                lane_id="lane-a",
                retries=0,
            )
            admission = session_module.OperationAdmission(
                output_dir=root,
                session_id=SESSION_ID,
                cost_contract=contract,
                session_contract_hash=SESSION_CONTRACT_HASH,
                termination_grace_s=5,
            )
            logical_train_id = f"{SESSION_ID}-generation-001-lane-a-train"
            self.assertIsNone(
                admission(generation_scope, logical_train_id, "train", 0)
            )
            for lane_id in ("lane-a", "lane-b"):
                lane_scope = _generation_scope(
                    root,
                    contract=contract,
                    resource_binding_hash=identity.identity_hash,
                    lane_id=lane_id,
                    retries=0,
                )
                lane_suffix = lane_id.removeprefix("lane-")
                self.assertEqual(
                    admission(
                        lane_scope,
                        (
                            f"{SESSION_ID}-generation-001-lane-{lane_suffix}-"
                            "eval-public-dev"
                        ),
                        "eval_phase",
                        0,
                    ),
                    420,
                )
                selected_slot, selected_reserve = (
                    session_module.read_operation_slot(
                        session_id=SESSION_ID,
                        scope=lane_scope,
                        logical_job_id=(
                            f"{SESSION_ID}-generation-001-lane-{lane_suffix}-"
                            "eval-public-accept"
                        ),
                        job_kind="eval_phase",
                        attempt_index=0,
                        cost_contract=contract,
                    )
                )
                self.assertEqual(
                    selected_slot,
                    "generation-001:selected:public_accept",
                )
                self.assertEqual(
                    selected_reserve.operation_class,
                    "selected_public_accept",
                )

            pre_genesis = _pre_genesis_scope(
                root,
                contract=contract,
                resource_binding_hash=identity.identity_hash,
                retries=0,
            )
            with self.assertRaisesRegex(ValueError, "cannot execute pre-Genesis"):
                admission(pre_genesis, f"{SESSION_ID}-train", "train", 0)

            final_scope = _final_confirmation_scope(
                root,
                contract=contract,
                resource_binding_hash=identity.identity_hash,
            )
            self.assertGreater(
                admission(
                    final_scope,
                    f"{SESSION_ID}-final-confirmation-final-eval-sealed-validation",
                    "eval_phase",
                    0,
                ),
                0,
            )

    def test_final_confirmation_failure_retains_sealed_retry_entitlement(self) -> None:
        contract = _imported_reference_formal_cost_contract()
        slots = session_module._strict_success_operation_slots(contract)
        final_slot = "final-confirmation:final"
        retryable_failure = {
            "outcome": "infrastructure_failure",
            "retryable": True,
        }
        state = session_module.OperationAllocationState(
            reserved_success_slots=frozenset(slots),
            reserved_retry_classes=frozenset(),
            attempt_results={final_slot: retryable_failure},
            measured_gpu_capacity_ns=0,
            conservatively_charged_gpu_capacity_ns=0,
            unmeasured_bound_attempts=0,
            open_attempt_output_reserve_bytes=0,
        )
        self.assertTrue(
            session_module._strict_retry_is_still_possible(
                operation_class="sealed_validation_retry",
                closed_generation_count=15,
                state=state,
                cost_contract=contract,
            )
        )
        self.assertFalse(
            session_module._strict_retry_is_still_possible(
                operation_class="sealed_validation_retry",
                closed_generation_count=15,
                state=replace(
                    state,
                    attempt_results={
                        final_slot: {"outcome": "completed", "retryable": False}
                    },
                ),
                cost_contract=contract,
            )
        )
        self.assertFalse(
            session_module._strict_retry_is_still_possible(
                operation_class="sealed_validation_retry",
                closed_generation_count=15,
                state=replace(
                    state,
                    reserved_retry_classes=frozenset(
                        {"sealed_validation_retry"}
                    ),
                ),
                cost_contract=contract,
            )
        )

    def test_sealed_success_remains_admissible_after_shared_retry_is_spent(
        self,
    ) -> None:
        contract = _imported_reference_formal_cost_contract()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            identity = GpuDeviceIdentity("cuda:0", "machine", "GPU-TEST")
            admission = session_module.OperationAdmission(
                output_dir=root,
                session_id=SESSION_ID,
                cost_contract=contract,
                session_contract_hash=SESSION_CONTRACT_HASH,
                termination_grace_s=5,
            )
            cases = (
                (
                    _generation_scope(
                        root,
                        contract=contract,
                        resource_binding_hash=identity.identity_hash,
                        lane_id="lane-a",
                        retries=0,
                        generation_index=2,
                    ),
                    f"{SESSION_ID}-generation-002-lane-a-eval-sealed-validation",
                ),
                (
                    _final_confirmation_scope(
                        root,
                        contract=contract,
                        resource_binding_hash=identity.identity_hash,
                        retries=0,
                    ),
                    f"{SESSION_ID}-final-confirmation-final-eval-sealed-validation",
                ),
            )
            for scope, logical_job_id in cases:
                self.assertGreater(
                    admission(scope, logical_job_id, "eval_phase", 0),
                    0,
                )
                with self.assertRaisesRegex(
                    ValueError,
                    "attempt index has no frozen operation allocation",
                ):
                    admission(scope, logical_job_id, "eval_phase", 1)

    def test_strict_contract_rejects_out_of_range_or_unplanned_final_work(
        self,
    ) -> None:
        formal = _formal_cost_contract()
        validation = _cost_contract()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            identity = GpuDeviceIdentity("cuda:0", "machine", "GPU-TEST")
            generation_sixteen = _generation_scope(
                root,
                contract=formal,
                resource_binding_hash=identity.identity_hash,
                lane_id="lane-a",
                retries=0,
                generation_index=16,
            )
            with self.assertRaisesRegex(ValueError, "exceeds its frozen session"):
                session_module.read_operation_slot(
                    session_id=SESSION_ID,
                    scope=generation_sixteen,
                    logical_job_id=f"{SESSION_ID}-generation-016-lane-a-train",
                    job_kind="train",
                    attempt_index=0,
                    cost_contract=formal,
                )

            validation_final_scope = JobAttemptScope(
                journal_root=root / "job_attempts",
                session_genesis_hash="1" * 64,
                session_contract_hash=SESSION_CONTRACT_HASH,
                generation_id="generation-final-confirmation",
                lane_id="lane-a",
                invocation_segment_id="final-confirmation",
                budget_contract_hash=validation.contract_hash,
                attempt_budget_units=2,
                max_infrastructure_retries=1,
                resource_binding_hash=identity.identity_hash,
            )
            with self.assertRaisesRegex(ValueError, "no frozen operation allocation"):
                session_module.read_operation_slot(
                    session_id=SESSION_ID,
                    scope=validation_final_scope,
                    logical_job_id=(
                        f"{SESSION_ID}-final-confirmation-final-"
                        "eval-sealed-validation"
                    ),
                    job_kind="eval_phase",
                    attempt_index=0,
                    cost_contract=validation,
                )

    def test_frozen_reserve_table_round_trip_and_v1_projection_are_stable(self) -> None:
        contract = _cost_contract()
        payload = contract.to_dict()
        plan = payload["operation_reserves"]
        reserves = stair_generation_operation_cost_reserves()

        self.assertEqual(
            [reserve.operation_class for reserve in reserves],
            [
                "reference_train",
                "reference_qualification",
                "parent_public_dev",
                "lane_a_train",
                "lane_b_train",
                "lane_a_public_dev",
                "lane_b_public_dev",
                "lane_a_public_accept",
                "lane_b_public_accept",
                "sealed_validation",
                "reference_train_retry",
                "sealed_validation_retry",
            ],
        )
        self.assertEqual(plan["derivation"]["headroom_numerator"], 5)
        self.assertEqual(plan["derivation"]["headroom_denominator"], 4)
        for reserve in reserves:
            self.assertEqual(
                reserve.allocated_gpu_capacity_ns_per_attempt,
                (reserve.source_allocated_gpu_capacity_ns * 5 + 3) // 4,
            )
            self.assertEqual(
                reserve.output_bytes_per_attempt,
                (reserve.source_output_bytes * 5 + 3) // 4,
            )
        self.assertEqual(
            plan["totals"]["successful_session"],
            {
                "source_allocated_gpu_capacity_ns": 31_127_832_110_058,
                "source_output_bytes": 2_338_080_326,
                "reserved_allocated_gpu_capacity_ns": 38_909_790_137_584,
                "reserved_output_bytes": 2_922_600_413,
                "source_non_job_output_bytes": 3_100_927,
                "source_total_output_bytes": 2_341_181_253,
            },
        )
        self.assertEqual(
            plan["totals"]["retry_reserve"],
            {
                "source_allocated_gpu_capacity_ns": 4_172_484_142_958,
                "source_output_bytes": 326_023_821,
                "reserved_allocated_gpu_capacity_ns": 5_215_605_178_699,
                "reserved_output_bytes": 407_529_777,
            },
        )
        self.assertEqual(
            StairGenerationSessionCostContractV2.from_mapping(payload),
            contract,
        )

        legacy = StairGenerationSessionCostContract(
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
        self.assertNotIn("operation_reserves", legacy.to_dict())
        self.assertEqual(
            StairGenerationSessionCostContract.from_mapping(legacy.to_dict()),
            legacy,
        )

    def test_reference_initial_and_internal_retry_use_distinct_allocations(
        self,
    ) -> None:
        contract = _cost_contract()
        reserves = {
            item.operation_class: item
            for item in stair_generation_operation_cost_reserves()
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            identity = GpuDeviceIdentity("cuda:0", "machine", "GPU-TEST")
            scope = _pre_genesis_scope(
                root,
                contract=contract,
                resource_binding_hash=identity.identity_hash,
                retries=1,
            )
            journal = _journal(
                scope=scope,
                logical_job_id=f"{SESSION_ID}-train",
                job_kind="train",
            )
            admission = session_module.OperationAdmission(
                output_dir=root,
                session_id=SESSION_ID,
                cost_contract=contract,
                session_contract_hash=SESSION_CONTRACT_HASH,
                termination_grace_s=5,
            )

            initial_index, initial_previous = journal.next_runnable_attempt()
            initial_timeout_s = admission(
                scope,
                journal.logical_job_id,
                journal.job_kind,
                initial_index,
            )
            initial = journal.reserve(initial_index, previous_result=initial_previous)
            initial_result = journal.finish(
                initial,
                outcome="infrastructure_failure",
                failure_kind=RunnerFailureKind.SUMMARY_MISSING.value,
                retryable=True,
                output_hash=None,
                returncode=None,
            )
            retry_index, retry_previous = journal.next_runnable_attempt()
            self.assertEqual((initial_index, retry_index), (0, 1))
            self.assertEqual(retry_previous.payload_hash, initial_result.payload_hash)
            retry_timeout_s = admission(
                scope,
                journal.logical_job_id,
                journal.job_kind,
                retry_index,
            )
            journal.reserve(retry_index, previous_result=retry_previous)

            self.assertEqual((initial_timeout_s, retry_timeout_s), (4375, 4653))
            self.assertLess(
                initial_timeout_s,
                reserves["reference_train"].allocated_gpu_capacity_ns_per_attempt
                // 1_000_000_000,
            )
            self.assertLess(
                retry_timeout_s,
                reserves["reference_train_retry"].allocated_gpu_capacity_ns_per_attempt
                // 1_000_000_000,
            )

            state = session_module.read_operation_allocation_state(
                output_dir=root,
                session_id=SESSION_ID,
                cost_contract=contract,
                session_contract_hash=SESSION_CONTRACT_HASH,
            )
            self.assertEqual(state.reserved_success_slots, {"reference_train"})
            self.assertEqual(
                state.reserved_retry_classes,
                {"reference_train_retry"},
            )
            self.assertEqual(
                state.conservatively_charged_gpu_capacity_ns,
                reserves["reference_train"].allocated_gpu_capacity_ns_per_attempt
                + reserves[
                    "reference_train_retry"
                ].allocated_gpu_capacity_ns_per_attempt,
            )

    def test_gpu_and_output_caps_reject_before_a_new_reservation(self) -> None:
        contract = _cost_contract()
        reserves = {
            item.operation_class: item
            for item in stair_generation_operation_cost_reserves()
        }
        for cap_name in ("allocated_active_gpu_hours", "output_bytes"):
            with self.subTest(cap_name=cap_name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                identity = GpuDeviceIdentity("cuda:0", "machine", "GPU-TEST")
                target_scope = _pre_genesis_scope(
                    root,
                    contract=contract,
                    resource_binding_hash=identity.identity_hash,
                    retries=1,
                )
                if cap_name == "allocated_active_gpu_hours":
                    prior_scope = _pre_genesis_scope(
                        root,
                        contract=contract,
                        resource_binding_hash=identity.identity_hash,
                        retries=0,
                    )
                    prior = _journal(
                        scope=prior_scope,
                        logical_job_id=f"{SESSION_ID}-eval",
                        job_kind="eval",
                    )
                    reservation = prior.reserve(0, previous_result=None)
                    headroom = contract.to_dict()["operation_reserves"][
                        "physical_unallocated_headroom"
                    ]["allocated_gpu_capacity_ns"]
                    prior.finish(
                        reservation,
                        outcome="completed",
                        failure_kind=None,
                        retryable=False,
                        output_hash=OUTPUT_HASH,
                        returncode=0,
                        resource_lease_wall_duration_ns=(
                            reserves[
                                "reference_qualification"
                            ].allocated_gpu_capacity_ns_per_attempt
                            + headroom
                            + 1
                        ),
                    )
                    size_patch = patch.object(
                        session_module,
                        "output_tree_size",
                        wraps=session_module.output_tree_size,
                    )
                else:
                    remaining_output = sum(
                        item.output_bytes for item in reserves.values()
                    )
                    near_cap = (
                        contract.max_output_bytes
                        - STAIR_GENERATION_FINALIZATION_OUTPUT_RESERVE_BYTES
                        - remaining_output
                        + 1
                    )
                    size_patch = patch.object(
                        session_module,
                        "output_tree_size",
                        return_value=(near_cap, 1),
                    )
                admission = session_module.OperationAdmission(
                    output_dir=root,
                    session_id=SESSION_ID,
                    cost_contract=contract,
                    session_contract_hash=SESSION_CONTRACT_HASH,
                    termination_grace_s=5,
                )
                runner = SubprocessRunner(
                    allowed_executables=(sys.executable,),
                    timeout_s=10,
                    gpu_lease_manager=GpuLeaseManager(
                        lease_root=root / "gpu_leases",
                        devices={"cuda:0": identity},
                    ),
                    attempt_admission_hook=admission,
                )
                reservations_before = list(
                    (root / "job_attempts").rglob("attempt_*.reservation.json")
                )
                with size_patch, self.assertRaisesRegex(RuntimeError, cap_name):
                    runner._execute_with_attempts(
                        _AttemptSpec(target_scope),
                        job_kind="train",
                        operation=lambda log_dir, execution_context, timeout_s: (
                            self.fail("operation started after admission rejection")
                        ),
                    )
                reservations_after = list(
                    (root / "job_attempts").rglob("attempt_*.reservation.json")
                )
                self.assertEqual(reservations_after, reservations_before)

    def test_recovered_unmeasured_attempt_charges_only_its_operation_capacity(
        self,
    ) -> None:
        contract = _cost_contract()
        reserve = {
            item.operation_class: item
            for item in stair_generation_operation_cost_reserves()
        }["reference_train"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            identity = GpuDeviceIdentity("cuda:0", "machine", "GPU-TEST")
            scope = _pre_genesis_scope(
                root,
                contract=contract,
                resource_binding_hash=identity.identity_hash,
                retries=1,
            )
            journal = _journal(
                scope=scope,
                logical_job_id=f"{SESSION_ID}-train",
                job_kind="train",
            )
            reservation = journal.reserve(0, previous_result=None)
            journal.finish(
                reservation,
                outcome="infrastructure_failure",
                failure_kind="abandoned_unclean",
                retryable=True,
                output_hash=None,
                returncode=None,
                recovery_receipt_hash=RECOVERY_HASH,
            )

            state = session_module.read_operation_allocation_state(
                output_dir=root,
                session_id=SESSION_ID,
                cost_contract=contract,
                session_contract_hash=SESSION_CONTRACT_HASH,
            )
            self.assertEqual(state.measured_gpu_capacity_ns, 0)
            self.assertEqual(state.unmeasured_bound_attempts, 1)
            self.assertEqual(
                state.conservatively_charged_gpu_capacity_ns,
                reserve.allocated_gpu_capacity_ns_per_attempt,
            )
            self.assertLess(
                state.conservatively_charged_gpu_capacity_ns,
                int(contract.max_allocated_active_gpu_hours * 3_600_000_000_000),
            )

    def test_closed_generation_and_duplicate_sealed_slot_are_rejected(self) -> None:
        contract = _cost_contract()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            identity = GpuDeviceIdentity("cuda:0", "machine", "GPU-TEST")
            admission = session_module.OperationAdmission(
                output_dir=root,
                session_id=SESSION_ID,
                cost_contract=contract,
                session_contract_hash=SESSION_CONTRACT_HASH,
                termination_grace_s=5,
            )
            closed_scope = _generation_scope(
                root,
                contract=contract,
                resource_binding_hash=identity.identity_hash,
                lane_id="lane-a",
                retries=0,
            )
            (root / "ledger.jsonl").touch()
            with (
                patch.object(session_module, "load_ledger", return_value=[{}]),
                self.assertRaisesRegex(ValueError, "closed generation"),
            ):
                admission(
                    closed_scope,
                    f"{SESSION_ID}-generation-001-lane-a-train",
                    "train",
                    0,
                )
            self.assertEqual(
                list((root / "job_attempts").rglob("attempt_*.reservation.json")),
                [],
            )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            identity = GpuDeviceIdentity("cuda:0", "machine", "GPU-TEST")
            first_scope = _generation_scope(
                root,
                contract=contract,
                resource_binding_hash=identity.identity_hash,
                lane_id="lane-a",
                retries=1,
            )
            first = _journal(
                scope=first_scope,
                logical_job_id=(
                    f"{SESSION_ID}-generation-001-lane-a-eval-sealed-validation"
                ),
                job_kind="eval_phase",
            )
            first.reserve(0, previous_result=None)
            second_scope = _generation_scope(
                root,
                contract=contract,
                resource_binding_hash=identity.identity_hash,
                lane_id="lane-b",
                retries=1,
            )
            admission = session_module.OperationAdmission(
                output_dir=root,
                session_id=SESSION_ID,
                cost_contract=contract,
                session_contract_hash=SESSION_CONTRACT_HASH,
                termination_grace_s=5,
            )
            with self.assertRaisesRegex(ValueError, "already reserved"):
                admission(
                    second_scope,
                    f"{SESSION_ID}-generation-001-lane-b-eval-sealed-validation",
                    "eval_phase",
                    0,
                )
            self.assertEqual(
                len(list(root.rglob("attempt_*.reservation.json"))),
                1,
            )


if __name__ == "__main__":
    unittest.main()
