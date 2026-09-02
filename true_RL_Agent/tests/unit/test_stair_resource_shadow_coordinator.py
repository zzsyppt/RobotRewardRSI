from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from true_rl_agent.agent_core.generation_contract import GenerationLaneAssignment
from true_rl_agent.runner.process_handle import OwnedProcessResult
from true_rl_agent.runner.resource_shadow import (
    RESOURCE_SHADOW_PLAN_FILENAME,
    create_resource_shadow_plan,
    parse_nvidia_smi_inventory,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class StairResourceShadowCoordinatorTests(unittest.TestCase):
    def test_owned_job_binds_physical_uuid_to_logical_cuda_zero(self) -> None:
        coordinator = _load_entrypoint()
        inventory = parse_nvidia_smi_inventory(
            "\n".join(
                (
                    "0, GPU-aaaaaaaa, 00000000:01:00.0, NVIDIA RTX 5090, 32607, 575.64.03",
                    "1, GPU-bbbbbbbb, 00000000:02:00.0, NVIDIA RTX 5090, 32607, 575.64.03",
                )
            ),
            machine_id="test-machine",
        )
        lanes = (
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
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            plan = create_resource_shadow_plan(
                root / RESOURCE_SHADOW_PLAN_FILENAME,
                pilot_id="p6a-device-binding",
                source_commit="8" * 40,
                session_genesis_hash="1" * 64,
                execution_environment_manifest_hash="2" * 64,
                fixture_hash="3" * 64,
                workload_hash="4" * 64,
                lanes=lanes,
                gpu_inventory=inventory,
            )
            job = plan.job("single-gpu1-warmup-01-lane-b")
            context = SimpleNamespace(
                isaac_python=Path(sys.executable).resolve(),
                expected_initial_policy_state_hash="5" * 64,
            )
            result = OwnedProcessResult(
                returncode=0,
                pid=123,
                stdout_tail="",
                stderr_tail="",
                stdout_log=root / "stdout.log",
                stderr_log=root / "stderr.log",
            )
            with patch.object(
                coordinator, "run_owned_process", return_value=result
            ) as run:
                coordinator._run_owned_job(
                    context=context,
                    plan=plan,
                    plan_path=root / RESOURCE_SHADOW_PLAN_FILENAME,
                    case_dir=root / "case",
                    job=job,
                )

        argv = tuple(run.call_args.args[0])
        self.assertEqual(
            run.call_args.kwargs["environment_overrides"],
            {
                "CUDA_VISIBLE_DEVICES": "GPU-bbbbbbbb",
                "NVIDIA_VISIBLE_DEVICES": None,
            },
        )
        self.assertEqual(argv[argv.index("--physical-execution-device") + 1], "cuda:1")
        self.assertEqual(argv[argv.index("--device") + 1], "cuda:0")

    def test_resource_observations_keep_sampled_peak_output(self) -> None:
        coordinator = _load_entrypoint()
        procfs = {
            "memory_kib": {
                "MemTotal": 1_000_000,
                "MemAvailable": 900_000,
                "SwapTotal": 100_000,
                "SwapFree": 99_000,
            },
            "cpu_total_ticks": 1_000,
            "cpu_idle_ticks": 600,
            "disk_read_sectors_total": 100,
            "disk_write_sectors_total": 200,
        }
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            (output_dir / "final.bin").write_bytes(b"x" * 20)
            snapshots = (
                {
                    "procfs": procfs,
                    "gpus": [_gpu_sample()],
                    "output_dir_size_bytes": 30,
                },
                {
                    "procfs": {
                        **procfs,
                        "cpu_total_ticks": 1_100,
                        "cpu_idle_ticks": 650,
                    },
                    "gpus": [_gpu_sample()],
                    "output_dir_size_bytes": 10,
                },
            )
            observations = coordinator._resource_observations(
                snapshots=snapshots,
                output_dir=output_dir,
                initial_snapshot=snapshots[0],
            )
        self.assertEqual(observations["peak_output_size_bytes"], 30)
        self.assertEqual(observations["final_output_size_bytes"], 20)

    def test_fault_probe_uses_post_marker_signal_evidence_not_exit_code(self) -> None:
        coordinator = _load_entrypoint()
        lane_a = SimpleNamespace(
            lane_id="lane-a",
            job_id="interruption-probe-01-lane-a",
            execution_device="cuda:0",
        )
        lane_b = SimpleNamespace(
            lane_id="lane-b",
            job_id="interruption-probe-01-lane-b",
            execution_device="cuda:1",
        )
        marker_a = {
            "process_id": 4100,
            "process_group_id": 4100,
            "active_start_monotonic_ns": 1_000,
        }
        marker_b = {
            "process_id": 4200,
            "process_group_id": 4200,
            "active_start_monotonic_ns": 1_000,
        }
        plan = SimpleNamespace(
            inventory_for_execution_device=lambda device: SimpleNamespace(
                uuid="GPU-aaaaaaaa" if device == "cuda:0" else "GPU-bbbbbbbb"
            )
        )
        pre_marker_snapshots = tuple(
            {
                "monotonic_ns": tick,
                "compute_apps": [
                    {"pid": 4100, "gpu_uuid": "GPU-aaaaaaaa"},
                    {"pid": 4200, "gpu_uuid": "GPU-bbbbbbbb"},
                ],
            }
            for tick in (100, 200, 300)
        )
        with patch.object(
            coordinator.os, "getpgid", return_value=4100, create=True
        ), patch.object(
            coordinator.os, "getpgrp", return_value=9999, create=True
        ), patch.object(coordinator.os, "killpg", create=True) as killpg:
            self.assertIsNone(
                coordinator._maybe_inject_fault(
                    jobs=(lane_a, lane_b),
                    markers={lane_a.job_id: marker_a, lane_b.job_id: marker_b},
                    snapshots=pre_marker_snapshots,
                    plan=plan,
                )
            )
            post_marker_snapshots = pre_marker_snapshots + tuple(
                {
                    "monotonic_ns": tick,
                    "compute_apps": [
                        {"pid": 4100, "gpu_uuid": "GPU-aaaaaaaa"},
                        {"pid": 4200, "gpu_uuid": "GPU-bbbbbbbb"},
                    ],
                }
                for tick in (1_100, 2_100, 3_100)
            )
            injection = coordinator._maybe_inject_fault(
                jobs=(lane_a, lane_b),
                markers={lane_a.job_id: marker_a, lane_b.job_id: marker_b},
                snapshots=post_marker_snapshots,
                plan=plan,
            )
        self.assertIsNotNone(injection)
        killpg.assert_called_once_with(4100, coordinator.signal.SIGTERM)
        assert injection is not None
        self.assertEqual(
            injection["post_active_matched_sample_counts_by_lane"],
            {
                "lane-a": coordinator.FAULT_REQUIRED_MATCHED_SAMPLES,
                "lane-b": coordinator.FAULT_REQUIRED_MATCHED_SAMPLES,
            },
        )
        self.assertNotIn("matched_sample_count_before_signal", injection)

        result_a = OwnedProcessResult(
            returncode=0,
            pid=4100,
            stdout_tail="",
            stderr_tail="",
            stdout_log=Path("lane-a.stdout"),
            stderr_log=Path("lane-a.stderr"),
        )
        case = SimpleNamespace(
            jobs=(lane_a, lane_b),
            job_outcomes=(
                SimpleNamespace(
                    job=lane_a,
                    result=result_a,
                    launch_end_ns=injection["signaled_at_monotonic_ns"] + 1,
                ),
            ),
            fault_injection=injection,
            markers={lane_a.job_id: marker_a, lane_b.job_id: marker_b},
            summaries={lane_a.job_id: None, lane_b.job_id: {"status": "pass"}},
            valid_jobs={lane_a.job_id: True, lane_b.job_id: True},
            measurements={
                lane_a.job_id: SimpleNamespace(orphaned_process=False),
                lane_b.job_id: SimpleNamespace(orphaned_process=False),
            },
            post_case_compute_apps=(),
        )
        with patch.object(coordinator, "_leases_reacquirable", return_value=True):
            fault = coordinator._fault_probe(
                case=case,
                lease_manager=object(),
                plan=plan,
            )
        self.assertTrue(all(fault.values()))

    def test_paired_case_acquires_both_leases_and_receipts_before_spawn(self) -> None:
        coordinator = _load_entrypoint()
        inventory = parse_nvidia_smi_inventory(
            "\n".join(
                (
                    "0, GPU-aaaaaaaa, 00000000:01:00.0, NVIDIA RTX 5090, 32607, 575.64.03",
                    "1, GPU-bbbbbbbb, 00000000:02:00.0, NVIDIA RTX 5090, 32607, 575.64.03",
                )
            ),
            machine_id="test-machine",
        )
        lanes = (
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

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp).resolve()
            plan = create_resource_shadow_plan(
                output_dir / RESOURCE_SHADOW_PLAN_FILENAME,
                pilot_id="p6a-test-pilot",
                source_commit="8" * 40,
                session_genesis_hash="1" * 64,
                execution_environment_manifest_hash="2" * 64,
                fixture_hash="3" * 64,
                workload_hash="4" * 64,
                lanes=lanes,
                gpu_inventory=inventory,
            )
            jobs = tuple(
                job for job in plan.jobs if job.case_id == "paired-warmup-01"
            )
            self.assertEqual(
                tuple((job.lane_id, job.execution_device) for job in jobs),
                (("lane-a", "cuda:0"), ("lane-b", "cuda:1")),
            )

            events: list[str] = []
            acquired: dict[str, object] = {}
            markers_seen: set[str] = set()
            both_markers_seen = threading.Event()
            case_dir = output_dir / "cases" / "paired-warmup-01"

            class _Lease:
                def __init__(
                    self,
                    device: str,
                    binding_hash: str,
                    holder: object,
                ) -> None:
                    self.device = device
                    self.resource_binding_hash = binding_hash
                    self.receipt = {
                        "schema_version": "test.gpu_lease.v1",
                        "device": device,
                        "holder": holder.to_dict(),
                        "ledger_authority": False,
                    }

                def __enter__(self):
                    events.append(f"lease-enter:{self.device}")
                    return self

                def __exit__(self, exc_type, exc_value, traceback) -> None:
                    _ = exc_type, exc_value, traceback
                    events.append(f"lease-exit:{self.device}")

            class _LeaseManager:
                def acquire(self, device: str, *, holder: object):
                    events.append(f"lease-acquire:{device}")
                    acquired[device] = holder
                    binding_hash = plan.lane_assignments[
                        "lane-a" if device == "cuda:0" else "lane-b"
                    ].resource_binding_hash
                    return _Lease(device, binding_hash, holder)

            class _Sampler:
                def __init__(self, *, plan: object, output_dir: Path) -> None:
                    _ = plan, output_dir

                def start(self) -> None:
                    events.append("sampler-start")

                def stop(self) -> None:
                    events.append("sampler-stop")

                def snapshot(self):
                    return (
                        {
                            "monotonic_ns": 150,
                            "compute_apps": [],
                        },
                    )

                def errors(self):
                    return ()

            def fake_marker(*, job, **kwargs):
                _ = kwargs
                markers_seen.add(job.job_id)
                if len(markers_seen) == len(jobs):
                    both_markers_seen.set()
                pid = 4100 if job.lane_id == "lane-a" else 4200
                return {
                    "process_id": pid,
                    "process_group_id": pid,
                    "active_start_monotonic_ns": 100,
                }

            def fake_owned_job(*, job, **kwargs):
                _ = kwargs
                self.assertTrue(both_markers_seen.wait(timeout=2.0))
                for planned_job in jobs:
                    lease_receipt = json.loads(
                        (
                            case_dir / "leases" / f"{planned_job.job_id}.json"
                        ).read_text(encoding="utf-8")
                    )
                    self.assertEqual(
                        lease_receipt["holder"]["logical_job_id"],
                        planned_job.job_id,
                    )
                    self.assertEqual(
                        lease_receipt["holder"]["lane_id"],
                        planned_job.lane_id,
                    )
                    self.assertFalse(lease_receipt["ledger_authority"])
                self.assertEqual(set(acquired), {"cuda:0", "cuda:1"})
                events.append(f"spawn:{job.execution_device}")
                pid = 4100 if job.lane_id == "lane-a" else 4200
                return coordinator._OwnedJobOutcome(
                    job=job,
                    child_output_dir=case_dir / "jobs" / job.job_id,
                    marker_path=case_dir / "jobs" / job.job_id / "active.json",
                    checkpoint_path=case_dir / "jobs" / job.job_id / "checkpoint.pt",
                    launch_start_ns=90,
                    launch_end_ns=61_000_000_100,
                    full_job_wall_seconds=61.0,
                    result=OwnedProcessResult(
                        returncode=0,
                        pid=pid,
                        stdout_tail="",
                        stderr_tail="",
                        stdout_log=case_dir / f"{job.job_id}.stdout.log",
                        stderr_log=case_dir / f"{job.job_id}.stderr.log",
                    ),
                    exception_type=None,
                    exception_message=None,
                )

            def fake_summary(*, outcome, **kwargs):
                _ = kwargs
                pid = 4100 if outcome.job.lane_id == "lane-a" else 4200
                return {
                    "status": "pass",
                    "process_id": pid,
                    "process_group_id": pid,
                    "active_start_monotonic_ns": 100,
                    "active_end_monotonic_ns": 61_000_000_100,
                    "job_id": outcome.job.job_id,
                }

            context = SimpleNamespace(
                output_dir=output_dir,
                expected_initial_policy_state_hash="5" * 64,
            )
            post_sample = {"monotonic_ns": 62_000_000_100, "compute_apps": []}
            with patch.object(coordinator, "_Sampler", _Sampler), patch.object(
                coordinator, "_run_owned_job", side_effect=fake_owned_job
            ), patch.object(
                coordinator, "_try_load_marker", side_effect=fake_marker
            ), patch.object(
                coordinator, "_try_load_summary", side_effect=fake_summary
            ), patch.object(
                coordinator, "sample_resource_snapshot", return_value=post_sample
            ), patch.object(
                coordinator, "_cross_device_pid", return_value=False
            ), patch.object(
                coordinator, "_process_group_alive", return_value=False
            ), patch.object(
                coordinator, "_swap_delta_mib", return_value=0.0
            ), patch.object(
                coordinator, "_active_sample_count", return_value=61
            ), patch.object(
                coordinator.os,
                "getpgid",
                side_effect=lambda pid: pid,
                create=True,
            ):
                outcome = coordinator._run_case(
                    context=context,
                    plan=plan,
                    lease_manager=_LeaseManager(),
                    jobs=jobs,
                )

            first_spawn = min(
                index
                for index, event in enumerate(events)
                if event.startswith("spawn:")
            )
            self.assertTrue(
                all(
                    events.index(f"lease-enter:{device}") < first_spawn
                    for device in ("cuda:0", "cuda:1")
                )
            )
            self.assertTrue(all(outcome.valid_jobs.values()))
            receipt = json.loads(outcome.receipt_path.read_text(encoding="utf-8"))
            self.assertEqual(len(receipt["jobs"]), 2)
            self.assertEqual(len(receipt["job_results"]), 2)
            self.assertFalse(receipt["acceptance_authority"])
            self.assertFalse(receipt["ledger_authority"])


def _load_entrypoint():
    path = PROJECT_ROOT / "scripts" / "run_stair_resource_shadow_pilot.py"
    module_name = "test_run_stair_resource_shadow_pilot"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load Stair resource-shadow coordinator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _gpu_sample() -> dict[str, object]:
    return {
        "uuid": "GPU-aaaaaaaa",
        "utilization_gpu_percent": 10.0,
        "memory_used_mib": 100.0,
        "power_draw_watts": 50.0,
    }


if __name__ == "__main__":
    unittest.main()
