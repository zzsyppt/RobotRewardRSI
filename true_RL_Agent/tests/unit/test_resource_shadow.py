from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from true_rl_agent.agent_core.generation_contract import GenerationLaneAssignment
from true_rl_agent.json_contract import canonical_json_bytes
from true_rl_agent.runner.resource_shadow import (
    RESOURCE_SHADOW_AUTHORITY_COUNTER_KEYS,
    NvidiaGpuInventoryEntry,
    ResourceShadowLaneMeasurement,
    create_resource_shadow_plan,
    load_resource_shadow_plan,
    parse_nvidia_smi_inventory,
    require_resource_shadow_child_binding,
    summarize_resource_shadow,
)


class ResourceShadowTests(unittest.TestCase):
    def _inventory(
        self,
    ) -> tuple[NvidiaGpuInventoryEntry, NvidiaGpuInventoryEntry]:
        entries = parse_nvidia_smi_inventory(
            "\n".join(
                (
                    "0, GPU-aaaaaaaa, 00000000:01:00.0, NVIDIA RTX 5090, 32607, 575.64.03",
                    "1, GPU-bbbbbbbb, 00000000:02:00.0, NVIDIA RTX 5090, 32607, 575.64.03",
                )
            ),
            machine_id="machine-identity",
        )
        return entries[0], entries[1]

    def _lanes(
        self,
        inventory: tuple[NvidiaGpuInventoryEntry, NvidiaGpuInventoryEntry],
    ) -> tuple[GenerationLaneAssignment, GenerationLaneAssignment]:
        return (
            GenerationLaneAssignment(
                lane_id="lane-a",
                execution_device="cuda:0",
                resource_binding_hash=inventory[0].identity_hash,
                lane_intent_hash="a" * 64,
            ),
            GenerationLaneAssignment(
                lane_id="lane-b",
                execution_device="cuda:1",
                resource_binding_hash=inventory[1].identity_hash,
                lane_intent_hash="b" * 64,
            ),
        )

    def _create_plan(self, path: Path, *, fixture_hash: str = "3" * 64):
        inventory = self._inventory()
        return create_resource_shadow_plan(
            path,
            pilot_id="p6a-pilot",
            source_commit="8" * 40,
            session_genesis_hash="1" * 64,
            execution_environment_manifest_hash="2" * 64,
            fixture_hash=fixture_hash,
            workload_hash="4" * 64,
            lanes=self._lanes(inventory),
            gpu_inventory=inventory,
        )

    def test_nvidia_inventory_parses_physical_identity_and_rejects_alias(self) -> None:
        inventory = self._inventory()
        self.assertEqual(
            tuple(item.execution_device for item in inventory),
            ("cuda:0", "cuda:1"),
        )
        self.assertEqual(inventory[0].memory_total_mib, 32607)
        self.assertEqual(inventory[0].driver_version, "575.64.03")
        self.assertNotEqual(
            inventory[0].physical_resource_hash,
            inventory[1].physical_resource_hash,
        )

        alias = "\n".join(
            (
                "0, GPU-same, 00000000:01:00.0, NVIDIA RTX 5090, 32607, 575.64.03",
                "1, GPU-same, 00000000:02:00.0, NVIDIA RTX 5090, 32607, 575.64.03",
            )
        )
        with self.assertRaisesRegex(ValueError, "alias"):
            parse_nvidia_smi_inventory(alias, machine_id="machine-identity")

    def test_plan_is_create_once_canonical_and_child_binding_is_exact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "resource_shadow_plan.json"
            plan = self._create_plan(path)
            replay = self._create_plan(path)
            self.assertEqual(replay, plan)
            self.assertEqual(load_resource_shadow_plan(path), plan)

            inventory = self._inventory()
            changed_driver_inventory = (
                replace(inventory[0], driver_version="576.01"),
                replace(inventory[1], driver_version="576.01"),
            )
            changed_driver_plan = create_resource_shadow_plan(
                Path(tmp) / "changed_driver_plan.json",
                pilot_id="p6a-pilot",
                source_commit="8" * 40,
                session_genesis_hash="1" * 64,
                execution_environment_manifest_hash="2" * 64,
                fixture_hash="3" * 64,
                workload_hash="4" * 64,
                lanes=self._lanes(changed_driver_inventory),
                gpu_inventory=changed_driver_inventory,
            )
            self.assertNotEqual(changed_driver_plan.plan_hash, plan.plan_hash)

            job = plan.job("paired-measured-01-lane-a")
            self.assertEqual(job.process_execution_device, "cuda:0")
            bound = require_resource_shadow_child_binding(
                plan,
                plan_hash=plan.plan_hash,
                session_genesis_hash=plan.session_genesis_hash,
                execution_environment_manifest_hash=(
                    plan.execution_environment_manifest_hash
                ),
                fixture_hash=plan.fixture_hash,
                workload_hash=plan.workload_hash,
                resource_plan_hash=plan.resource_plan_hash,
                job_id=job.job_id,
                case_id=job.case_id,
                lane_id=job.lane_id,
                physical_execution_device=job.execution_device,
                process_execution_device=job.process_execution_device,
            )
            self.assertEqual(bound, job)
            with self.assertRaisesRegex(ValueError, "job binding"):
                require_resource_shadow_child_binding(
                    plan,
                    plan_hash=plan.plan_hash,
                    session_genesis_hash=plan.session_genesis_hash,
                    execution_environment_manifest_hash=(
                        plan.execution_environment_manifest_hash
                    ),
                    fixture_hash=plan.fixture_hash,
                    workload_hash=plan.workload_hash,
                    resource_plan_hash=plan.resource_plan_hash,
                    job_id=job.job_id,
                    case_id=job.case_id,
                    lane_id=job.lane_id,
                    physical_execution_device="cuda:1",
                    process_execution_device=job.process_execution_device,
                )

            with self.assertRaises(FileExistsError):
                self._create_plan(path, fixture_hash="9" * 64)

            raw = json.loads(path.read_text(encoding="utf-8"))
            raw["sample_interval_seconds"] = 2.0
            path.write_bytes(canonical_json_bytes(raw, trailing_newline=True))
            with self.assertRaises(ValueError):
                load_resource_shadow_plan(path)

    def test_plan_rejects_same_physical_uuid_even_with_distinct_ordinals(self) -> None:
        first, _ = self._inventory()
        alias = NvidiaGpuInventoryEntry(
            ordinal=1,
            uuid=first.uuid,
            pci_bus_id="00000000:02:00.0",
            name=first.name,
            memory_total_mib=first.memory_total_mib,
            driver_version=first.driver_version,
            machine_id=first.machine_id,
        )
        inventory = (first, alias)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "same physical UUID"):
                create_resource_shadow_plan(
                    Path(tmp) / "resource_shadow_plan.json",
                    pilot_id="p6a-pilot",
                    source_commit="8" * 40,
                    session_genesis_hash="1" * 64,
                    execution_environment_manifest_hash="2" * 64,
                    fixture_hash="3" * 64,
                    workload_hash="4" * 64,
                    lanes=self._lanes(inventory),
                    gpu_inventory=inventory,
                )

    def test_metrics_and_operational_gates_are_pure_and_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plan = self._create_plan(Path(tmp) / "resource_shadow_plan.json")
            singles_a = tuple(
                self._measurement(
                    lane_id="lane-a",
                    repeat_index=index,
                    case_id=f"single-gpu0-measured-{index:02d}",
                    start_ns=index * 1_000_000_000,
                    seconds=100.0,
                )
                for index in range(1, 4)
            )
            singles_b = tuple(
                self._measurement(
                    lane_id="lane-b",
                    repeat_index=index,
                    case_id=f"single-gpu1-measured-{index:02d}",
                    start_ns=index * 1_000_000_000,
                    seconds=100.0,
                )
                for index in range(1, 4)
            )
            pairs = tuple(
                (
                    self._measurement(
                        lane_id="lane-a",
                        repeat_index=index,
                        case_id=f"paired-measured-{index:02d}",
                        start_ns=index * 1_000_000_000,
                        seconds=110.0,
                    ),
                    self._measurement(
                        lane_id="lane-b",
                        repeat_index=index,
                        case_id=f"paired-measured-{index:02d}",
                        start_ns=index * 1_000_000_000 + 5_000_000_000,
                        seconds=110.0,
                    ),
                )
                for index in range(1, 4)
            )
            counters = {key: 0 for key in RESOURCE_SHADOW_AUTHORITY_COUNTER_KEYS}
            fault = {
                "lane_a_interrupted": True,
                "lane_b_completed": True,
                "lane_a_process_group_clean": True,
                "lane_b_process_group_clean": True,
                "leases_cleared": True,
                "compute_pids_cleared": True,
                "coordinator_closed": True,
            }
            summary = summarize_resource_shadow(
                plan=plan,
                single_lane_a=singles_a,
                single_lane_b=singles_b,
                paired=pairs,
                fault_probe=fault,
                authority_counters=counters,
            )
            self.assertTrue(summary["go"])
            self.assertTrue(summary["gates"]["minimum_measured_active_duration"])
            self.assertAlmostEqual(summary["aggregates"]["median_overlap_ratio"], 105 / 110)
            self.assertAlmostEqual(
                summary["aggregates"]["median_paired_to_matched_single_gpu_hours_ratio"],
                1.1,
            )
            self.assertAlmostEqual(
                summary["aggregates"]["median_paired_to_one_single_baseline_gpu_hours_ratio"],
                2.2,
            )

            bad_fault = dict(fault)
            bad_fault["lane_b_completed"] = False
            rejected = summarize_resource_shadow(
                plan=plan,
                single_lane_a=singles_a,
                single_lane_b=singles_b,
                paired=pairs,
                fault_probe=bad_fault,
                authority_counters=counters,
            )
            self.assertFalse(rejected["go"])
            self.assertFalse(rejected["gates"]["fault_isolation_and_cleanup"])

            short_pairs = list(pairs)
            short_pairs[0] = (
                self._measurement(
                    lane_id="lane-a",
                    repeat_index=1,
                    case_id="paired-measured-01",
                    start_ns=1_000_000_000,
                    seconds=59.0,
                ),
                short_pairs[0][1],
            )
            too_short = summarize_resource_shadow(
                plan=plan,
                single_lane_a=singles_a,
                single_lane_b=singles_b,
                paired=short_pairs,
                fault_probe=fault,
                authority_counters=counters,
            )
            self.assertFalse(too_short["go"])
            self.assertFalse(
                too_short["gates"]["minimum_measured_active_duration"]
            )

            bad_counters = dict(counters)
            bad_counters["provider_requests"] = 1
            with self.assertRaisesRegex(ValueError, "provider_requests"):
                summarize_resource_shadow(
                    plan=plan,
                    single_lane_a=singles_a,
                    single_lane_b=singles_b,
                    paired=pairs,
                    fault_probe=fault,
                    authority_counters=bad_counters,
                )

    @staticmethod
    def _measurement(
        *,
        lane_id: str,
        repeat_index: int,
        case_id: str,
        start_ns: int,
        seconds: float,
    ) -> ResourceShadowLaneMeasurement:
        return ResourceShadowLaneMeasurement(
            job_id=f"{case_id}-{lane_id}",
            case_id=case_id,
            repeat_index=repeat_index,
            lane_id=lane_id,
            active_start_ns=start_ns,
            active_end_ns=start_ns + int(seconds * 1_000_000_000),
            full_job_wall_seconds=seconds + 4.0,
        )


if __name__ == "__main__":
    unittest.main()
