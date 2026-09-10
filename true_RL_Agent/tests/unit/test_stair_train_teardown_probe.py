from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import patch

from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.json_contract import strict_json_object_loads
from true_rl_agent.runner.checkpoint import file_sha256
from true_rl_agent.runner.gpu_lease import GpuLeaseManager
from true_rl_agent.runner.process_handle import OwnedProcessResult
from true_rl_agent.runner.resource_shadow import NvidiaGpuInventoryEntry
from true_rl_agent.runner.specs import GpuProcessBinding

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROBE_PATH = PROJECT_ROOT / "scripts/run_stair_train_teardown_probe.py"
SMOKE_PATH = PROJECT_ROOT / "scripts/isaaclab_stair_rsl_rl_smoke.py"
COMMIT = "a" * 40
GPU_UUID = "GPU-11111111-1111-1111-1111-111111111111"
CHILD_PID = 4242


def _load_probe() -> ModuleType:
    spec = importlib.util.spec_from_file_location("stair_teardown_probe_test", PROBE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_json(path: Path) -> dict[str, Any]:
    return strict_json_object_loads(path.read_bytes(), context=path.name)


def _write_bound_smoke_report(
    probe: ModuleType,
    *,
    report_path: Path,
    checkpoint_path: Path,
    binding: GpuProcessBinding,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(b"bounded-real-smoke-checkpoint")
    checkpoint_checksum = file_sha256(checkpoint_path)
    report: dict[str, Any] = {
        "schema_version": probe.SMOKE_SCHEMA,
        "status": "pass",
        "training": dict(probe.WORKLOAD),
        "reward_manager_active_terms": ["complete_reward"],
        "runtime": {"dispatcher_calls": 9},
        "checkpoint": {"path": str(checkpoint_path), "checksum": checkpoint_checksum},
        "gpu_process_binding": binding.to_dict(),
        "owned_process_identity": {
            "schema_version": "true_rl_agent.runner.owned_process_identity.v1",
            "process_id": CHILD_PID,
            "process_group_id": CHILD_PID,
        },
        "gpu_runtime_evidence": {
            "schema_version": "true_rl_agent.runner.gpu_runtime_evidence.v1",
            "cuda_visibility_binding": "single_physical_gpu_uuid",
            "physical_gpu_uuid": GPU_UUID,
            "renderer_active_gpu_ordinal": 1,
            "renderer_multi_gpu": False,
            "renderer_multi_gpu_auto_enable": False,
            "renderer_multi_gpu_max_gpu_count": 1,
            "nvidia_visible_devices_unset": True,
            "external_compute_context_observation": {
                "schema_version": (
                    "true_rl_agent.task.stair_climbing."
                    "gpu_compute_context_observation.v1"
                ),
                "source": "nvidia-smi-query-compute-apps",
                "monotonic_ns": 123,
                "process_id": CHILD_PID,
                "rows": [{"gpu_uuid": GPU_UUID, "used_gpu_memory_mib": 64}],
            },
        },
    }
    report["report_hash"] = canonical_json_hash(report)
    report_path.write_text(json.dumps(report), encoding="utf-8")


def _configure_probe(
    stack: ExitStack,
    root: Path,
    probe: ModuleType,
    *,
    child_failure: bool = False,
) -> tuple[list[str], NvidiaGpuInventoryEntry, list[str]]:
    python = Path(sys.executable).resolve(strict=True)
    target = NvidiaGpuInventoryEntry(
        ordinal=1,
        uuid=GPU_UUID,
        pci_bus_id="00000000:02:00.0",
        name="test-gpu",
        memory_total_mib=24576,
        driver_version="test-driver",
        machine_id="test-machine",
    )
    samples: list[str] = []
    stack.enter_context(patch.object(probe.sys, "platform", "linux"))
    stack.enter_context(patch.dict(probe.os.environ, {}, clear=True))
    stack.enter_context(
        patch.object(
            probe, "_linux_boot_id", return_value="22222222-2222-2222-2222-222222222222"
        )
    )
    source = lambda expected: {  # noqa: E731 - compact fixed test fixture
        "commit": expected,
        "tracked_clean": True,
        "checksums": {
            path: "sha256:" + "1" * 64 for path in probe.CRITICAL_SOURCE_PATHS
        },
    }
    stack.enter_context(patch.object(probe, "_source_identity", side_effect=source))
    stack.enter_context(
        patch.object(probe, "discover_nvidia_gpu_inventory", return_value=(target,))
    )

    def fake_snapshot(**kwargs: Any) -> dict[str, Any]:
        if kwargs["inventory"] != (target,):
            raise AssertionError("idle snapshot used the wrong GPU")
        samples.append(str(kwargs["output_dir"]))
        return {"compute_apps": [], "gpus": [{"uuid": GPU_UUID}], "operational_only": True}

    def fake_run(argv: list[str], **kwargs: Any) -> OwnedProcessResult:
        if kwargs["cwd"] != PROJECT_ROOT or kwargs["timeout_s"] != 900:
            raise AssertionError("owned process launch contract drifted")
        if kwargs["termination_grace_s"] != 5.0:
            raise AssertionError("owned process cleanup grace drifted")
        expected_env = {"CUDA_VISIBLE_DEVICES": GPU_UUID, "NVIDIA_VISIBLE_DEVICES": None}
        if kwargs["environment_overrides"] != expected_env:
            raise AssertionError("child GPU visibility drifted")
        if Path(argv[0]) != python or Path(argv[2]) != SMOKE_PATH.resolve(strict=True):
            raise AssertionError("child executable drifted")
        if argv[argv.index("--device") + 1] != "cuda:0" or "--headless" not in argv:
            raise AssertionError("child logical-device arguments drifted")
        if child_failure:
            raise RuntimeError("synthetic owned-child failure")
        report_path = Path(argv[argv.index("--output") + 1])
        checkpoint_path = Path(argv[argv.index("--checkpoint-output") + 1])
        binding = GpuProcessBinding(
            host_execution_device=argv[argv.index("--gpu-host-execution-device") + 1],
            process_execution_device=argv[argv.index("--gpu-process-execution-device") + 1],
            resource_binding_hash=argv[argv.index("--gpu-resource-binding-hash") + 1],
        )
        _write_bound_smoke_report(
            probe,
            report_path=report_path,
            checkpoint_path=checkpoint_path,
            binding=binding,
        )
        log_dir = Path(kwargs["log_dir"])
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout, stderr = log_dir / "stdout.log", log_dir / "stderr.log"
        stdout.write_text("pass\n", encoding="utf-8")
        stderr.write_bytes(b"")
        return OwnedProcessResult(0, CHILD_PID, "pass", "", stdout, stderr)

    stack.enter_context(patch.object(probe, "sample_resource_snapshot", side_effect=fake_snapshot))
    stack.enter_context(patch.object(probe, "run_owned_process", side_effect=fake_run))
    stack.enter_context(patch.object(probe, "_linux_live_process_group_members", return_value=()))
    stack.enter_context(patch.object(probe, "_detached_processes_referencing", return_value=()))
    stack.enter_context(patch.object(probe.time, "sleep", return_value=None))
    (root / "leases").mkdir()
    argv = [
        "--output-dir", str(root / "probe"),
        "--formal-output-dir", str(root / "formal"),
        "--isaac-python", str(python),
        "--expected-source-commit", COMMIT,
        "--host-execution-device", "cuda:1",
        "--gpu-uuid", GPU_UUID,
        "--nvidia-smi-executable", str(python),
        "--lease-root", str(root / "leases"),
    ]
    return argv, target, samples


class StairTrainTeardownProbeTests(unittest.TestCase):
    def test_real_smoke_owner_path_closes_all_launch_gates(self) -> None:
        probe = _load_probe()
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory).resolve()
            argv, _, samples = _configure_probe(stack, root, probe)
            self.assertEqual(probe.main(argv), 0)
            launch = _read_json(root / "probe/launch.json")
            result = _read_json(root / "probe/result.json")
            self.assertFalse((root / "probe/failure.json").exists())
            self.assertFalse((root / "formal").exists())
            self.assertEqual(len(samples), 3)
            self.assertTrue(launch["lease_held"])
            self.assertEqual(
                launch["host"]["boot_id"], "22222222-2222-2222-2222-222222222222"
            )
            self.assertEqual(Path(launch["probe_argv"][0]), Path(sys.executable).resolve())
            self.assertIn("--lease-root", launch["probe_argv"])
            self.assertEqual(
                launch["child_environment_overrides"]["CUDA_VISIBLE_DEVICES"],
                GPU_UUID,
            )
            self.assertEqual(result["child"]["process_id"], CHILD_PID)
            self.assertEqual(result["child"]["workload"], probe.WORKLOAD)
            self.assertTrue(result["child"]["reward_runtime_activated"])
            self.assertTrue(result["lease_reacquired"])
            self.assertFalse(result["acceptance_authority"])

    def test_child_failure_is_create_once_and_releases_lease(self) -> None:
        probe = _load_probe()
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory).resolve()
            argv, target, _ = _configure_probe(stack, root, probe, child_failure=True)
            self.assertEqual(probe.main(argv), 1)
            launch = _read_json(root / "probe/launch.json")
            failure = _read_json(root / "probe/failure.json")
            self.assertFalse((root / "probe/result.json").exists())
            self.assertEqual(failure["exception_type"], "RuntimeError")
            self.assertTrue(failure["formal_output_absent_at_failure"])
            manager = GpuLeaseManager(
                lease_root=root / "leases", devices={"cuda:1": target.identity}
            )
            holder = probe._holder(
                launch["launch_identity_hash"], segment="test-reacquire"
            )
            with manager.acquire("cuda:1", holder=holder) as lease:
                lease.assert_owned()

    def test_shared_lease_root_must_preexist_and_be_disjoint(self) -> None:
        probe = _load_probe()
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory).resolve()
            argv, _, _ = _configure_probe(stack, root, probe)
            (root / "leases").rmdir()
            self.assertEqual(probe.main(argv), 1)
            failure = _read_json(root / "probe/failure.json")
            self.assertEqual(failure["exception_type"], "FileNotFoundError")
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory).resolve()
            argv, _, _ = _configure_probe(stack, root, probe)
            argv[argv.index("--lease-root") + 1] = str(root)
            self.assertEqual(probe.main(argv), 1)
            failure = _read_json(root / "probe/failure.json")
            self.assertIn("disjoint directory", failure["exception_message"])

    def test_detached_scan_and_smoke_binding_branch_are_explicit(self) -> None:
        probe = _load_probe()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            marker = root / "child"
            for pid, argv in (("101", b"python\0unrelated\0"), ("202", bytes(marker))):
                process_dir = root / "proc" / pid
                process_dir.mkdir(parents=True)
                (process_dir / "cmdline").write_bytes(argv)
            self.assertEqual(
                probe._detached_processes_referencing(marker, proc_root=root / "proc"),
                (202,),
            )
            self.assertTrue(probe._paths_overlap(root / "a", root / "a/formal"))
        smoke = SMOKE_PATH.read_text(encoding="utf-8")
        self.assertLess(
            smoke.index("gpu_runtime_binding = validate_gpu_process_binding_invocation"),
            smoke.index("simulation_app ="),
        )
        self.assertIn("if gpu_runtime_binding is None", smoke)
        self.assertIn("AppLauncher(args, multi_gpu=False).app", smoke)
        self.assertIn(
            "if gpu_runtime_binding is not None:\n        report = attach_gpu_runtime_evidence",
            smoke,
        )
        parent = PROBE_PATH.read_text(encoding="utf-8")
        self.assertIn("isaaclab_stair_rsl_rl_smoke.py", parent)
        self.assertNotIn("AppLauncher", parent)
        self.assertNotIn("OnPolicyRunner", parent)


if __name__ == "__main__":
    unittest.main()
