from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from true_rl_agent.agent_core.generation_cost import (
    FrozenGenerationCostContract, GenerationProviderBudget, OperationCostReserve,
    generation_operation_layout,
)
from true_rl_agent.agent_core.generation_cost_journal import (
    GenerationActualCostJournal, GenerationActualCostUsage,
)
from true_rl_agent.agent_core.generation_operations import _strict_success_operation_slots
from true_rl_agent.agent_core.generation_preparation import GenerationLaunchConfig, prepare_generation_foundation
from true_rl_agent.composition import TaskCompositionContext, resolve_task_composition
from tests.fakes.backend_environment import use_test_isaac_lab_execution_environment
from true_rl_agent.tasks.stair_climbing.composition import StairClimbingCompositionProvider
from true_rl_agent.tasks.push_recovery.composition import PushRecoveryCompositionProvider
from true_rl_agent.tasks.push_recovery.config import PushRecoveryTaskConfig
from true_rl_agent.tasks.push_recovery.plugin import PushRecoveryTaskPlugin
from true_rl_agent.tasks.push_recovery.robot_binding import (
    PUSH_G1_29DOF_REFERENCE_CELL_VARIANT_ID, resolve_push_cell_manifest,
)
from true_rl_agent.tasks.stair_climbing.config import StairClimbingTaskConfig
from true_rl_agent.tasks.stair_climbing.plugin import StairClimbingTaskPlugin
from true_rl_agent.tasks.stair_climbing.robot_binding import resolve_stair_cell_manifest
from true_rl_agent.tasks.stair_climbing.row_catalog import stair_robot_row


ROOT = Path(__file__).resolve().parents[2]


def cell_workloads():
    for cell in ("stair.g1_37dof.unified_v3", "stair.go2_12dof.unified_v3"):
        config = StairClimbingTaskConfig.defaults(ROOT / stair_robot_row(cell).protocol_relative_path)
        plugin = StairClimbingTaskPlugin(config)
        manifest = resolve_stair_cell_manifest(cell_variant_id=cell, config=config, plugin=plugin)
        yield cell, plugin.phased_evaluation_adapter().generation_workload(cell_manifest=manifest)
    config = PushRecoveryTaskConfig(reference_program_genesis=True)
    plugin = PushRecoveryTaskPlugin(config)
    manifest = resolve_push_cell_manifest(
        cell_variant_id=PUSH_G1_29DOF_REFERENCE_CELL_VARIANT_ID, config=config, plugin=plugin,
    )
    yield manifest.cell_variant_id, plugin.phased_evaluation_adapter().generation_workload(cell_manifest=manifest)


def fixture_provider_policy():
    return {"max_attempts_per_lane": 3, "max_output_tokens": 2000,
            "input_units_per_attempt_reserve": 10000, "api_key_env": "FIXTURE_KEY",
            "base_url": "https://fixture.invalid", "endpoint": "/chat/completions",
            "model": "fixture", "timeout_s": 30, "temperature": 0.6,
            "reasoning_mode": "adaptive", "reasoning_effort": "high",
            "empty_final_reasoning_retry": True,
            "model_context": {"context_window_tokens": 16384, "max_output_tokens": 4096,
                "tokenizer_count_source": "provider_usage_prompt_tokens",
                "specification_source": "fixture model contract"}}


def fixture_cost(workload, *, imports_reference=False, max_generations=15):
    layout = generation_operation_layout(max_generations=max_generations, imports_reference=imports_reference, final_jobs=2)
    sources = {
        name: OperationCostReserve.from_source(
            operation_class=name, budget_class=kind, multiplicity=1,
            source_allocated_gpu_capacity_ns=10_000_000_000,
            source_output_bytes=100_000, headroom_numerator=5, headroom_denominator=4,
        )
        for name, (kind, count) in layout.items()
    }
    return FrozenGenerationCostContract.freeze(
        max_generations=max_generations, workload=workload,
        provider_budget=GenerationProviderBudget(3, 10000, 2000),
        operation_sources=sources, imports_reference=imports_reference,
        final_confirmation=True, finalization_output_reserve_bytes=1_000_000,
        session_output_reserve_bytes=20_000_000,
        headroom_numerator=5, headroom_denominator=4,
    )


class GenerationCostContractTests(unittest.TestCase):
    def test_three_cells_use_the_same_foundation_without_changing_their_scientific_identity(self):
        cases = (
            (StairClimbingCompositionProvider(), "stair.g1_37dof.unified_v3"),
            (StairClimbingCompositionProvider(), "stair.go2_12dof.unified_v3"),
            (PushRecoveryCompositionProvider(), PUSH_G1_29DOF_REFERENCE_CELL_VARIANT_ID),
        )
        gpu_uuids = ("GPU-11111111-1111-1111-1111-111111111111", "GPU-22222222-2222-2222-2222-222222222222")
        for provider, cell in cases:
            with self.subTest(cell=cell), tempfile.TemporaryDirectory() as directory, use_test_isaac_lab_execution_environment():
                root = Path(directory).resolve()
                context = TaskCompositionContext(project_root=ROOT, output_dir=root / "session")
                values = {**provider.cli_defaults(), "cell_variant_id": cell}
                composition = resolve_task_composition(provider=provider, values=values, context=context)
                original_task_hash = composition.task_plugin.task_spec.resolved_task_protocol_hash()
                original_cell_hash = composition.resolved_cell_manifest.manifest_hash
                final_plan = composition.task_plugin.prepare_final_confirmation(
                    output_dir=context.output_dir, session_id="shared-foundation-study",
                    required_closed_generations=15, resume_existing=False,
                )
                values.update({"mode": "formal_generation_session", "session_id": final_plan.session_id,
                               "max_generations": 15,
                               "final_confirmation_plan": context.output_dir / "final_confirmation" / "plan.json"})
                composition = resolve_task_composition(provider=provider, values=values, context=context)
                self.assertEqual(composition.task_plugin.task_spec.resolved_task_protocol_hash(), original_task_hash)
                self.assertEqual(composition.resolved_cell_manifest.manifest_hash, original_cell_hash)
                work = composition.task_plugin.phased_evaluation_adapter().generation_workload(
                    cell_manifest=composition.resolved_cell_manifest,
                )
                cost = fixture_cost(work)
                launch = GenerationLaunchConfig(
                    project_root=ROOT, output_dir=context.output_dir, lease_root=root / "leases",
                    provider_env_file=root / "provider.env", session_id=final_plan.session_id,
                    cell_variant_id=cell, machine_id="fixture", gpu0_uuid=gpu_uuids[0], gpu1_uuid=gpu_uuids[1],
                    max_allocated_active_gpu_hours=cost.max_allocated_active_gpu_hours,
                    max_simulated_control_steps=cost.max_simulated_control_steps, max_output_bytes=cost.max_output_bytes,
                )
                foundation = prepare_generation_foundation(
                    launch, composition=composition, source_commit="a" * 40, inventory=gpu_uuids,
                    cost=cost, final_plan=final_plan,
                    provider_policy=fixture_provider_policy(),
                )
                self.assertIsNotNone(foundation.runner.attempt_admission_hook)
                self.assertEqual(foundation.cost_contract.contract_hash, cost.contract_hash)
                self.assertEqual(foundation.session_contract.task_id, provider.task_id)
                self.assertEqual(foundation.session_contract.bound_contract_hashes["final_confirmation_plan"], final_plan.plan_hash)

    def test_real_task_robot_workloads_share_one_operation_and_receipt_contract(self):
        for cell, work in cell_workloads():
            with self.subTest(cell=cell), tempfile.TemporaryDirectory() as directory:
                cost = fixture_cost(work)
                restored = FrozenGenerationCostContract.from_mapping(cost.to_dict())
                self.assertEqual(restored, cost)
                slots = _strict_success_operation_slots(restored)
                self.assertEqual(len(slots), 96)  # 4 reference + 15 * 6 + 2 final jobs.
                self.assertEqual(sum(slot.startswith("generation-") and slot.endswith(":public_accept")
                                     for slot in slots), 15)
                self.assertEqual(cost.physical_train_job_cap, 32)  # Includes one reference retry.
                self.assertEqual(work.eval_seed_units_per_generation, 18 if cell.startswith("stair.") else 16)
                journal = GenerationActualCostJournal(Path(directory), contract=restored)
                receipt = journal.publish_reservation()
                self.assertTrue(receipt.payload["cap_checks"]["all_passed"])
                self.assertEqual(journal.load_chain(), [receipt])
                snapshot = journal.load_snapshot()
                next_receipt = journal.publish_snapshot(
                    event_id="post-genesis", previous_cost_chain_head_hash=receipt.payload_hash,
                    usage=GenerationActualCostUsage(), finalization_output_reserve_bytes=0,
                )
                self.assertNotIn("post-genesis", snapshot)
                self.assertEqual(journal.load_snapshot()["post-genesis"], next_receipt)

    def test_imported_reference_removes_only_local_reference_work_and_retry(self):
        for cell, work in cell_workloads():
            with self.subTest(cell=cell):
                local, imported = fixture_cost(work), fixture_cost(work, imports_reference=True)
                self.assertEqual(local.physical_train_job_cap - imported.physical_train_job_cap, 2)
                self.assertEqual(local.physical_eval_seed_unit_cap - imported.physical_eval_seed_unit_cap,
                                 work.reference_eval_seed_units)
                self.assertEqual(len(_strict_success_operation_slots(imported)), 92)
                self.assertNotEqual(local.contract_hash, imported.contract_hash)

    def test_push_eval_work_uses_frozen_control_period_and_runtime_episode_horizon(self):
        workloads = dict(cell_workloads())
        push = workloads[PUSH_G1_29DOF_REFERENCE_CELL_VARIANT_ID]
        self.assertEqual(push.max_eval_control_steps_per_seed_unit, 64 * 200)
        cost = fixture_cost(push)
        changed = replace(cost, scientific_cost_basis=replace(push, max_eval_control_steps_per_seed_unit=64 * 201))
        self.assertNotEqual(cost.contract_hash, changed.contract_hash)
        self.assertEqual(changed.max_simulated_control_steps - cost.max_simulated_control_steps,
                         64 * cost.physical_eval_seed_unit_cap)


if __name__ == "__main__":
    unittest.main()
