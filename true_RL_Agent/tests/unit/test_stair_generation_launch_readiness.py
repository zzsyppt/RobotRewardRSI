from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.json_contract import canonical_json_bytes
from true_rl_agent.runner.gpu_lease import GpuLeaseManager, PreGenesisGpuLeaseHolder
from true_rl_agent.runner.resource_shadow import NvidiaGpuInventoryEntry
from true_rl_agent.tasks.stair_climbing.launch_context import (
    STAIR_GENERATION_LAUNCH_START_PATH_ENV,
    STAIR_GENERATION_LAUNCH_START_SCHEMA_VERSION,
    STAIR_GENERATION_LAUNCH_START_SHA256_ENV,
    consume_stair_generation_launch_context,
    stair_generation_command_hash,
)
from true_rl_agent.tasks.stair_climbing.launch_readiness import (
    DEFAULT_SAMPLE_DURATION_SECONDS,
    DEFAULT_SAMPLE_INTERVAL_SECONDS,
    FORMAL_MAX_OUTPUT_BYTES,
    STAIR_FORMAL_LAUNCH_SUBJECT_FRESH_ABSENT,
    STAIR_FORMAL_LAUNCH_SUBJECT_NONTERMINAL_RESUME,
    STAIR_FORMAL_LAUNCH_SUBJECT_TERMINAL_AUDIT_ONLY,
    STAIR_FORMAL_LAUNCH_SUBJECT_TERMINAL_FINALIZATION,
    StairGenerationLaunchReadinessConfig,
    StairGenerationLaunchReadinessConfigV3,
    collect_stair_generation_launch_readiness_v3,
    load_stair_generation_launch_readiness_receipt,
    resolve_stair_nvidia_smi_executable,
)

_COMMIT = "a" * 40
_MACHINE_ID = "machine-readiness-identity"
_BOOT_ID = "11111111-2222-3333-4444-555555555555"
_GPU_UUIDS = (
    "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
    "GPU-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
)
_ROOT = Path(__file__).resolve().parents[2]


class _FakeClock:
    def __init__(self) -> None:
        self.nanoseconds = 10_000_000_000
        self.utc_calls = 0

    def monotonic_ns(self) -> int:
        return self.nanoseconds

    def sleep(self, seconds: float) -> None:
        self.nanoseconds += int(seconds * 1_000_000_000)

    def utc_now(self) -> str:
        self.utc_calls += 1
        return f"2026-08-22T00:00:00.{self.utc_calls:06d}+00:00"


class StairGenerationLaunchReadinessTests(unittest.TestCase):
    def _inventory(
        self,
    ) -> tuple[NvidiaGpuInventoryEntry, NvidiaGpuInventoryEntry]:
        return (
            NvidiaGpuInventoryEntry(
                ordinal=0,
                uuid=_GPU_UUIDS[0],
                pci_bus_id="00000000:01:00.0",
                name="test GPU 0",
                memory_total_mib=32_000,
                driver_version="test-driver",
                machine_id=_MACHINE_ID,
            ),
            NvidiaGpuInventoryEntry(
                ordinal=1,
                uuid=_GPU_UUIDS[1],
                pci_bus_id="00000000:02:00.0",
                name="test GPU 1",
                memory_total_mib=32_000,
                driver_version="test-driver",
                machine_id=_MACHINE_ID,
            ),
        )

    def test_config_rejects_foreign_source_root_even_with_copied_module(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            config = self._config(root)
            foreign_root = root / "foreign-checkout"
            foreign_module = (
                foreign_root
                / "src"
                / "true_rl_agent"
                / "tasks"
                / "stair_climbing"
                / "launch_readiness.py"
            )
            foreign_module.parent.mkdir(parents=True)
            foreign_module.write_bytes(
                (
                    _ROOT
                    / "src"
                    / "true_rl_agent"
                    / "tasks"
                    / "stair_climbing"
                    / "launch_readiness.py"
                ).read_bytes()
            )

            with self.assertRaisesRegex(ValueError, "does not own the running module"):
                replace(config, source_root=foreign_root)

    def _config(self, root: Path) -> StairGenerationLaunchReadinessConfig:
        nvidia_smi = root / "nvidia-smi"
        nvidia_smi.write_bytes(b"test executable\n")
        machine_id = root / "machine-id"
        machine_id.write_text(_MACHINE_ID + "\n", encoding="utf-8")
        boot_id = root / "boot-id"
        boot_id.write_text(_BOOT_ID + "\n", encoding="utf-8")
        meminfo = root / "meminfo"
        meminfo.write_text(
            "MemAvailable: 30000000 kB\nSwapTotal: 100000 kB\nSwapFree: 100000 kB\n",
            encoding="utf-8",
        )
        return StairGenerationLaunchReadinessConfig(
            receipt_path=root / "evidence" / "readiness.json",
            formal_output_dir=root / "formal-output",
            session_id="p6c-session",
            source_root=_ROOT,
            expected_source_commit=_COMMIT,
            expected_machine_id=_MACHINE_ID,
            expected_gpu_uuids=_GPU_UUIDS,
            lease_root=root / "leases",
            max_output_bytes=8 * 1024**3,
            nvidia_smi_executable=nvidia_smi,
            producer_argv=(
                sys.executable,
                str(_ROOT / "scripts" / "run_stair_generation_launch_readiness.py"),
                "--unit-test",
            ),
            machine_id_path=machine_id,
            boot_id_path=boot_id,
            proc_meminfo_path=meminfo,
        )

    def _config_v3(
        self,
        root: Path,
        *,
        subject_mode: str,
    ) -> StairGenerationLaunchReadinessConfigV3:
        base = replace(
            self._config(root),
            session_id="p6d-session",
            max_output_bytes=FORMAL_MAX_OUTPUT_BYTES,
        )
        provider_env = root / "provider.env"
        provider_env.write_text("DEEPSEEK_API_KEY=unit-test-secret\n", encoding="utf-8")
        attempt_report, teardown_result = self._write_prelaunch_probe_evidence(
            root,
            base=base,
        )
        return StairGenerationLaunchReadinessConfigV3(
            base=base,
            subject_mode=subject_mode,
            provider_env_file=provider_env,
            formal_generation_command_sha256="sha256:" + "7" * 64,
            attempt_chain_fault_probe_report=attempt_report,
            stair_train_teardown_result=teardown_result,
        )

    def _write_prelaunch_probe_evidence(
        self,
        root: Path,
        *,
        base: StairGenerationLaunchReadinessConfig,
    ) -> tuple[Path, Path]:
        attempt_dir = root / "prelaunch" / "attempt-chain"
        attempt_dir.mkdir(parents=True)
        attempt_report = attempt_dir / "report.json"
        attempt_report.write_bytes(
            canonical_json_bytes(
                {
                    "schema_version": "true_rl_agent.p6b.attempt_chain_probe.v1",
                    "status": "pass",
                    "started_utc": "2026-08-22T00:00:00Z",
                    "finished_utc": "2026-08-22T00:00:01Z",
                    "command_argv": [sys.executable, "attempt-probe.py"],
                    "host": {
                        "hostname": "test-host",
                        "boot_id": _BOOT_ID,
                        "kernel": {},
                        "python_executable": sys.executable,
                        "python_version": sys.version,
                    },
                    "source": {
                        "source_commit": _COMMIT,
                        "tracked_clean": True,
                        "source_file_sha256": {
                            "scripts/run_p6b_attempt_chain_fault_probe.py": (
                                "sha256:" + "1" * 64
                            )
                        },
                    },
                    "owned_python_executable": sys.executable,
                    "probe_identity_hash": "2" * 64,
                    "attempt_chain": {
                        "attempt_result_count": 2,
                        "result_files": [
                            "chain/attempt_0.result.json",
                            "chain/attempt_1.result.json",
                        ],
                        "first_result_hash": "3" * 64,
                        "retry_previous_result_hash": "3" * 64,
                        "terminal_chain_head_hash": "4" * 64,
                        "terminal_output_hash": "5" * 64,
                        "republish_created": [False, False],
                        "republish_bytes_unchanged": True,
                    },
                    "process_observations": [
                        {
                            "attempt_index": index,
                            "pid": 100 + index,
                            "pgid": 100 + index,
                            "grandchild_pid": 200 + index,
                            "live_pids_after_return": [],
                            "stdout_log": f"attempt_{index}/stdout.log",
                            "stderr_log": f"attempt_{index}/stderr.log",
                        }
                        for index in range(2)
                    ],
                    "checks": {
                        "exactly_two_create_once_results": True,
                        "retry_chain_closed": True,
                        "zero_process_group_residual": True,
                    },
                    "file_sha256": {"attempts/result.json": "sha256:" + "6" * 64},
                },
                trailing_newline=True,
            )
        )

        teardown_dir = root / "prelaunch" / "teardown"
        teardown_dir.mkdir(parents=True)
        teardown_result = teardown_dir / "result.json"
        target_gpu = self._inventory()[0].to_dict()
        source = {
            "commit": _COMMIT,
            "tracked_clean": True,
            "checksums": {
                "scripts/run_stair_train_teardown_probe.py": "sha256:" + "7" * 64
            },
        }
        host = {
            "hostname": "test-host",
            "boot_id": _BOOT_ID,
            "parent_process_id": 90,
            "isaac_python": sys.executable,
            "nvidia_smi_executable": str(base.nvidia_smi_executable),
            "target_gpu": target_gpu,
        }

        def idle(stage: str) -> dict[str, object]:
            return {
                "stage": stage,
                "snapshot": {
                    "gpus": [{"uuid": _GPU_UUIDS[0]}],
                    "compute_apps": [],
                },
            }

        launch_identity_hash = "8" * 64
        startup_idle = idle("startup")
        launch = {
            "schema_version": (
                "true_rl_agent.p6b.stair_train_teardown_probe.launch.v1"
            ),
            "status": "launched",
            "created_at_utc": "2026-08-22T00:00:00Z",
            "launch_identity_hash": launch_identity_hash,
            "source": source,
            "host": host,
            "probe_argv": [sys.executable, "teardown-probe.py"],
            "formal_output_dir": str(base.formal_output_dir),
            "lease_root": str(base.lease_root),
            "child_argv": [sys.executable, "stair-smoke.py"],
            "child_environment_overrides": {
                "CUDA_VISIBLE_DEVICES": _GPU_UUIDS[0],
                "NVIDIA_VISIBLE_DEVICES": None,
            },
            "timeout_s": 900,
            "termination_grace_s": 5.0,
            "workload": {
                "num_envs": 16,
                "num_steps_per_env": 8,
                "max_iterations": 1,
                "env_steps": 128,
            },
            "operational_only": True,
            "scientific_claim_authority": False,
            "acceptance_authority": False,
            "ledger_authority": False,
            "startup_compute_idle": startup_idle,
            "lease_held": True,
        }
        launch_bytes = canonical_json_bytes(launch, trailing_newline=True)
        (teardown_dir / "launch.json").write_bytes(launch_bytes)
        teardown_result.write_bytes(
            canonical_json_bytes(
                {
                    "schema_version": (
                        "true_rl_agent.p6b.stair_train_teardown_probe.result.v1"
                    ),
                    "status": "pass",
                    "created_at_utc": "2026-08-22T00:00:02Z",
                    "launch_identity_hash": launch_identity_hash,
                    "source": source,
                    "host": host,
                    "child": {
                        "argv": [sys.executable, "stair-smoke.py"],
                        "environment_overrides": {
                            "CUDA_VISIBLE_DEVICES": _GPU_UUIDS[0],
                            "NVIDIA_VISIBLE_DEVICES": None,
                        },
                        "process_id": 101,
                        "process_group_id": 101,
                        "returncode": 0,
                        "stdout_log": "stdout.log",
                        "stdout_checksum": "sha256:" + "9" * 64,
                        "stderr_log": "stderr.log",
                        "stderr_checksum": "sha256:" + "a" * 64,
                        "smoke_report": "smoke_report.json",
                        "checkpoint": "checkpoint.pt",
                        "report_hash": "b" * 64,
                        "report_checksum": "sha256:" + "c" * 64,
                        "checkpoint_checksum": "sha256:" + "d" * 64,
                        "workload": dict(launch["workload"]),
                        "reward_runtime_activated": True,
                        "dispatcher_calls": 1,
                        "gpu_runtime_evidence": {},
                    },
                    "startup_compute_idle": startup_idle,
                    "post_exit_compute_idle_samples": [
                        idle("post-exit-1"),
                        idle("post-exit-2"),
                    ],
                    "process_group_live_members_after_return": [],
                    "detached_processes_referencing_child_output": [],
                    "lease_reacquired": True,
                    "formal_output_absent": True,
                    "launch_checksum": (
                        "sha256:" + hashlib.sha256(launch_bytes).hexdigest()
                    ),
                    "operational_only": True,
                    "scientific_claim_authority": False,
                    "acceptance_authority": False,
                    "ledger_authority": False,
                },
                trailing_newline=True,
            )
        )
        return attempt_report, teardown_result

    @staticmethod
    def _source(_root: Path) -> dict[str, object]:
        return {"head_commit": _COMMIT, "tracked_clean": True}

    @staticmethod
    def _formal_subject(output_dir: Path, *, subject_mode: str) -> dict[str, object]:
        output_dir = output_dir.resolve()

        def evidence(path: Path, marker: str = "1") -> dict[str, object]:
            return {
                "path": str(path.resolve(strict=False)),
                "sha256": "sha256:" + marker * 64,
                "size_bytes": 1,
            }

        def optional(path: Path, *, exists: bool, marker: str) -> dict[str, object]:
            return {
                "path": str(path.resolve(strict=False)),
                "exists": exists,
                "sha256": "sha256:" + marker * 64 if exists else None,
                "size_bytes": 1 if exists else 0,
            }

        terminal = subject_mode in {
            STAIR_FORMAL_LAUNCH_SUBJECT_TERMINAL_FINALIZATION,
            STAIR_FORMAL_LAUNCH_SUBJECT_TERMINAL_AUDIT_ONLY,
        }
        audit_only = subject_mode == STAIR_FORMAL_LAUNCH_SUBJECT_TERMINAL_AUDIT_ONLY
        current_accounted = 2 * 1024**3 if terminal else 1024**3
        cost_path = output_dir / "generation_actual_cost" / "0000-test.json"
        cost_path.parent.mkdir(parents=True, exist_ok=True)
        cost_payload = {"actual": {"accounted_output_bytes": current_accounted}}
        cost_bytes = canonical_json_bytes(cost_payload, trailing_newline=True)
        cost_path.write_bytes(cost_bytes)
        cost_evidence = {
            "path": str(cost_path.resolve()),
            "sha256": "sha256:" + hashlib.sha256(cost_bytes).hexdigest(),
            "size_bytes": len(cost_bytes),
            "payload_hash": canonical_json_hash(cost_payload),
        }
        final_root = output_dir / "final_confirmation"
        lifecycle_receipt = {
            **evidence(
                output_dir / "generation_segments" / "segment-0001" / "exit.json",
                "a",
            ),
            "payload_hash": "b" * 64,
        }
        attempt_body = {"file_count": 0, "total_size_bytes": 0, "files": []}
        body = {
            "subject_mode": subject_mode,
            "source_identity": {
                "file": evidence(output_dir / "generation_source_checkout.json", "2"),
                "source_commit": _COMMIT,
                "source_checkout_hash": "3" * 64,
            },
            "session_genesis": {
                "file": optional(
                    output_dir / "session_genesis.json", exists=True, marker="4"
                ),
                "session_genesis_hash": "5" * 64,
            },
            "generation_session_contract": {
                "file": evidence(output_dir / "generation_session_contract.json", "6"),
                "contract_hash": "7" * 64,
            },
            "continuation_policy": {
                "file": evidence(
                    output_dir / "generation_continuation_policy.json", "8"
                ),
                "policy_hash": "9" * 64,
            },
            "ledger": {
                "file": optional(
                    output_dir / "ledger.jsonl", exists=terminal, marker="a"
                ),
                "height": 15 if terminal else 0,
                "snapshot_hash": "b" * 64,
            },
            "lifecycle": {
                "initialized": True,
                "head_hash": "b" * 64 if terminal else None,
                "active": False,
                "active_segment_id": None,
                "segment_count": 5 if terminal else 0,
                "closed_generation_count": 15 if terminal else 0,
                "terminal": terminal,
                "continuation_eligible": not terminal,
                "cost_chain_head_hash": "c" * 64,
                "receipt_files": [lifecycle_receipt] if terminal else [],
            },
            "cost": {
                "contract": {
                    "file": evidence(output_dir / "generation_session_cost.json", "d"),
                    "contract_hash": "e" * 64,
                },
                "receipt_count": 1,
                "head": cost_evidence,
                "receipt_files": [cost_evidence],
            },
            "attempt_state": {
                **attempt_body,
                "snapshot_hash": canonical_json_hash(attempt_body),
            },
            "final_confirmation": {
                "plan": {
                    "file": evidence(final_root / "plan.json", "f"),
                    "plan_hash": "1" * 64,
                },
                "reservation": {
                    "file": optional(
                        final_root / "reservation.json",
                        exists=audit_only,
                        marker="2",
                    ),
                    "reservation_hash": "3" * 64 if audit_only else None,
                },
                "result": {
                    "file": optional(
                        final_root / "result.json", exists=audit_only, marker="4"
                    ),
                    "result_hash": "5" * 64 if audit_only else None,
                },
            },
            "formal_report": optional(
                output_dir / "generation_session_report.json",
                exists=audit_only,
                marker="6",
            ),
            "current_accounted_output_bytes": current_accounted,
        }
        return {**body, "snapshot_hash": canonical_json_hash(body)}

    def _collect_v3(
        self,
        config: StairGenerationLaunchReadinessConfigV3,
        *,
        formal_subject_inspector=None,
        busy_call: int | None = None,
        stall_call: int | None = None,
    ):
        clock = _FakeClock()
        call_count = 0

        def resource_sampler(inventory, _config):
            nonlocal call_count
            call_count += 1
            if call_count == stall_call:
                clock.nanoseconds += 3_000_000_000
            return {
                "gpus": [{"uuid": item.uuid} for item in inventory],
                "compute_apps": (
                    [{"gpu_uuid": _GPU_UUIDS[0], "pid": 123}]
                    if call_count == busy_call
                    else []
                ),
            }

        return collect_stair_generation_launch_readiness_v3(
            config,
            inventory_discoverer=lambda _config: self._inventory(),
            resource_sampler=resource_sampler,
            system_sampler=lambda _config: {
                "memory_available_kib": 30 * 1024 * 1024,
                "swap_used_kib": 0,
                "free_disk_bytes": 40 * 1024**3,
            },
            source_inspector=self._source,
            formal_subject_inspector=formal_subject_inspector,
            monotonic_ns=clock.monotonic_ns,
            utc_now=clock.utc_now,
            sleep=clock.sleep,
            hostname_reader=lambda: "test-host",
            environ={},
        )

    @staticmethod
    def _loader_kwargs(
        config: StairGenerationLaunchReadinessConfig,
    ) -> dict[str, object]:
        return {
            "expected_formal_output_dir": config.formal_output_dir,
            "expected_source_root": config.source_root,
            "expected_session_id": config.session_id,
            "expected_source_commit": config.expected_source_commit,
            "expected_machine_id": config.expected_machine_id,
            "expected_gpu_uuids": config.expected_gpu_uuids,
            "expected_lease_root": config.lease_root,
            "expected_max_output_bytes": config.max_output_bytes,
            "expected_nvidia_smi_executable": config.nvidia_smi_executable,
            "expected_hostname": "test-host",
            "expected_boot_id": _BOOT_ID,
        }

    def test_collects_canonical_bound_receipt_and_strict_ref(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_v3 = self._config_v3(
                Path(tmp),
                subject_mode=STAIR_FORMAL_LAUNCH_SUBJECT_FRESH_ABSENT,
            )
            config = config_v3.base
            receipt = self._collect_v3(config_v3)

            self.assertEqual(
                receipt.payload["observations"]["window_sample_count"],
                601,
            )
            self.assertEqual(receipt.payload["observations"]["all_sample_count"], 606)
            self.assertFalse(config.formal_output_dir.exists())
            self.assertEqual(
                receipt.payload["producer"]["argv"],
                list(config.producer_argv),
            )
            self.assertEqual(
                receipt.payload["producer"]["entrypoint"]["path"],
                str(
                    (
                        _ROOT / "scripts" / "run_stair_generation_launch_readiness.py"
                    ).resolve()
                ),
            )
            self.assertNotIn(
                "must-not-persist", config.receipt_path.read_text(encoding="utf-8")
            )
            self.assertEqual(
                config.receipt_path.read_bytes(),
                canonical_json_bytes(receipt.to_dict(), trailing_newline=True),
            )
            loaded = load_stair_generation_launch_readiness_receipt(
                config.receipt_path,
                expected_formal_output_dir=config.formal_output_dir,
                expected_source_root=config.source_root,
                expected_session_id=config.session_id,
                expected_source_commit=config.expected_source_commit,
                expected_machine_id=config.expected_machine_id,
                expected_gpu_uuids=config.expected_gpu_uuids,
                expected_lease_root=config.lease_root,
                expected_max_output_bytes=config.max_output_bytes,
                expected_nvidia_smi_executable=config.nvidia_smi_executable,
                expected_hostname="test-host",
                expected_boot_id=_BOOT_ID,
                expected_subject_mode=STAIR_FORMAL_LAUNCH_SUBJECT_FRESH_ABSENT,
                expected_formal_generation_command_sha256=(
                    config_v3.formal_generation_command_sha256
                ),
            )
            self.assertEqual(loaded.receipt_hash, receipt.receipt_hash)
            receipt_ref = receipt.to_ref(config.receipt_path)
            self.assertEqual(receipt_ref.receipt_hash, receipt.receipt_hash)
            self.assertEqual(receipt_ref.path, config.receipt_path)
            self.assertGreater(receipt_ref.size_bytes, 0)
            self.assertTrue(receipt_ref.sha256.startswith("sha256:"))
            completed_at = datetime.fromisoformat(
                receipt.payload["observations"]["completed_at_utc"]
            )
            receipt.assert_fresh(now=completed_at + timedelta(seconds=299))
            with self.assertRaisesRegex(ValueError, "stale"):
                receipt.assert_fresh(now=completed_at + timedelta(seconds=301))
            with self.assertRaisesRegex(FileExistsError, "receipt must be absent"):
                self._collect_v3(config_v3)

    def test_formal_child_consumes_canonical_launcher_and_readiness_context(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            process_argv = [
                sys.executable,
                "-u",
                str(_ROOT / "scripts" / "run_stair_generation_session.py"),
                "--unit-test",
            ]
            config_v3 = replace(
                self._config_v3(
                    Path(tmp),
                    subject_mode=STAIR_FORMAL_LAUNCH_SUBJECT_FRESH_ABSENT,
                ),
                formal_generation_command_sha256=stair_generation_command_hash(
                    process_argv
                ),
            )
            config = config_v3.base
            receipt = self._collect_v3(config_v3)
            readiness_ref = receipt.to_ref(config.receipt_path).to_dict()
            start_path = Path(tmp) / "attempt" / "start.json"
            start_path.parent.mkdir()
            start = {
                "schema_version": STAIR_GENERATION_LAUNCH_START_SCHEMA_VERSION,
                "attempt_id": "attempt-test",
                "started_at_utc": "2026-08-22T00:00:00Z",
                "launcher": {
                    "argv": [sys.executable, "launcher.py"],
                    "cwd": str(_ROOT),
                    "process_id": 1234,
                    "process_group_id": 1234,
                    "session_id": 1234,
                    "python_executable": sys.executable,
                },
                "child": {
                    "argv": process_argv,
                    "command_sha256": stair_generation_command_hash(process_argv),
                    "cwd": str(_ROOT),
                    "timeout_s": 60,
                },
                "host": {
                    "hostname": "test-host",
                    "boot_id": _BOOT_ID,
                    "machine_id": _MACHINE_ID,
                },
                "source": {
                    "commit": _COMMIT,
                    "tracked_clean": True,
                    "files": {"launch.py": {"sha256": "sha256:" + "0" * 64}},
                },
                "launch_readiness_receipt": readiness_ref,
            }
            start_bytes = canonical_json_bytes(start, trailing_newline=True)
            start_path.write_bytes(start_bytes)
            environment = {
                STAIR_GENERATION_LAUNCH_START_PATH_ENV: str(start_path),
                STAIR_GENERATION_LAUNCH_START_SHA256_ENV: (
                    "sha256:" + hashlib.sha256(start_bytes).hexdigest()
                ),
            }

            with (
                mock.patch(
                    "true_rl_agent.tasks.stair_climbing.launch_context.socket.gethostname",
                    return_value="test-host",
                ),
                mock.patch(
                    "true_rl_agent.tasks.stair_climbing.launch_readiness."
                    "StairGenerationLaunchReadinessReceipt.assert_fresh"
                ),
            ):
                loaded = consume_stair_generation_launch_context(
                    process_argv=process_argv,
                    project_root=_ROOT,
                    formal_output_dir=config.formal_output_dir,
                    session_id=config.session_id,
                    expected_source_commit=config.expected_source_commit,
                    expected_machine_id=config.expected_machine_id,
                    expected_gpu_uuids=config.expected_gpu_uuids,
                    lease_root=config.lease_root,
                    max_output_bytes=config.max_output_bytes,
                    nvidia_smi_executable=config.nvidia_smi_executable,
                    helper_mode=False,
                    resume_existing=False,
                    formal_session=True,
                    environ=environment,
                    expected_launcher_parent_pid=1234,
                )

            self.assertEqual(loaded, start)
            self.assertEqual(environment, {})
            self.assertFalse(config.formal_output_dir.exists())

    def test_failure_releases_both_leases_and_never_creates_formal_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_v3 = self._config_v3(
                Path(tmp),
                subject_mode=STAIR_FORMAL_LAUNCH_SUBJECT_FRESH_ABSENT,
            )
            config = config_v3.base
            with self.assertRaisesRegex(RuntimeError, "not compute-idle"):
                self._collect_v3(config_v3, busy_call=2)
            self.assertFalse(config.receipt_path.exists())
            self.assertFalse(config.formal_output_dir.exists())

            manager = GpuLeaseManager(
                lease_root=config.lease_root,
                devices={
                    item.execution_device: item.identity for item in self._inventory()
                },
            )
            binding_hash = canonical_json_hash({"failure_cleanup_probe": True})
            leases = []
            try:
                for lane_id, item in zip(
                    ("lane-a", "lane-b"), self._inventory(), strict=True
                ):
                    leases.append(
                        manager.acquire(
                            item.execution_device,
                            holder=PreGenesisGpuLeaseHolder(
                                session_id=config.session_id,
                                genesis_identity_hash=binding_hash,
                                session_contract_hash=binding_hash,
                                lane_id=lane_id,
                                invocation_segment_id="test-cleanup",
                                logical_job_id=f"cleanup-{lane_id}",
                            ),
                        )
                    )
                for lease in leases:
                    lease.assert_owned()
            finally:
                for lease in reversed(leases):
                    lease.release()

    def test_sampling_gap_and_relative_executable_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_v3 = self._config_v3(
                Path(tmp),
                subject_mode=STAIR_FORMAL_LAUNCH_SUBJECT_FRESH_ABSENT,
            )
            config = config_v3.base
            with self.assertRaisesRegex(RuntimeError, "cadence drifted"):
                self._collect_v3(config_v3, stall_call=10)
            self.assertFalse(config.receipt_path.exists())

            with mock.patch(
                "true_rl_agent.tasks.stair_climbing.launch_readiness.shutil.which",
                return_value=str(config.nvidia_smi_executable),
            ):
                self.assertEqual(
                    resolve_stair_nvidia_smi_executable("nvidia-smi"),
                    config.nvidia_smi_executable,
                )
            with self.assertRaisesRegex(ValueError, "absolute path or bare"):
                resolve_stair_nvidia_smi_executable("./nvidia-smi")

    def test_loader_rejects_hash_fields_and_every_expected_binding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_v3 = self._config_v3(
                Path(tmp),
                subject_mode=STAIR_FORMAL_LAUNCH_SUBJECT_FRESH_ABSENT,
            )
            config = config_v3.base
            receipt = self._collect_v3(config_v3)
            original = receipt.to_dict()

            def load(path: Path, **overrides):
                arguments = {
                    "expected_formal_output_dir": config.formal_output_dir,
                    "expected_source_root": config.source_root,
                    "expected_session_id": config.session_id,
                    "expected_source_commit": config.expected_source_commit,
                    "expected_machine_id": config.expected_machine_id,
                    "expected_gpu_uuids": config.expected_gpu_uuids,
                    "expected_lease_root": config.lease_root,
                    "expected_max_output_bytes": config.max_output_bytes,
                    "expected_nvidia_smi_executable": config.nvidia_smi_executable,
                    "expected_hostname": "test-host",
                    "expected_boot_id": _BOOT_ID,
                    "expected_subject_mode": (
                        STAIR_FORMAL_LAUNCH_SUBJECT_FRESH_ABSENT
                    ),
                    "expected_formal_generation_command_sha256": (
                        config_v3.formal_generation_command_sha256
                    ),
                }
                arguments.update(overrides)
                return load_stair_generation_launch_readiness_receipt(path, **arguments)

            noncanonical = Path(tmp) / "noncanonical.json"
            noncanonical.write_text(json.dumps(original, indent=2), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not canonical"):
                load(noncanonical)

            bad_hash = Path(tmp) / "bad-hash.json"
            tampered = json.loads(json.dumps(original))
            tampered["source"]["tracked_clean_after"] = False
            bad_hash.write_bytes(canonical_json_bytes(tampered, trailing_newline=True))
            with self.assertRaises(ValueError):
                load(bad_hash)

            extra_field = Path(tmp) / "extra-field.json"
            tampered = json.loads(json.dumps(original))
            tampered["unexpected"] = True
            extra_field.write_bytes(
                canonical_json_bytes(tampered, trailing_newline=True)
            )
            with self.assertRaisesRegex(ValueError, "fields mismatch"):
                load(extra_field)

            mismatch_cases = {
                "formal output": {
                    "expected_formal_output_dir": Path(tmp) / "different-output"
                },
                "session": {"expected_session_id": "different-session"},
                "source": {"expected_source_commit": "b" * 40},
                "machine": {"expected_machine_id": "different-machine"},
                "GPU": {
                    "expected_gpu_uuids": (
                        _GPU_UUIDS[0],
                        "GPU-cccccccc-cccc-cccc-cccc-cccccccccccc",
                    )
                },
                "lease": {"expected_lease_root": Path(tmp) / "different-leases"},
                "output cap": {
                    "expected_max_output_bytes": config.max_output_bytes + 1
                },
                "nvidia-smi": {
                    "expected_nvidia_smi_executable": config.machine_id_path
                },
                "hostname": {"expected_hostname": "different-host"},
                "boot": {"expected_boot_id": "99999999-8888-7777-6666-555555555555"},
            }
            for label, overrides in mismatch_cases.items():
                with self.subTest(label=label):
                    with self.assertRaisesRegex(ValueError, "binding mismatch"):
                        load(config.receipt_path, **overrides)

    def test_v3_fresh_is_local_only_and_uses_remaining_output_formula(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config_v3(
                Path(tmp),
                subject_mode=STAIR_FORMAL_LAUNCH_SUBJECT_FRESH_ABSENT,
            )
            receipt = self._collect_v3(config)

            self.assertEqual(
                receipt.payload["schema_version"],
                "true_rl_agent.stair.generation_launch_readiness.v3",
            )
            self.assertEqual(
                receipt.payload["subject"],
                {
                    "subject_mode": STAIR_FORMAL_LAUNCH_SUBJECT_FRESH_ABSENT,
                    "formal_output_absent": True,
                    "current_accounted_output_bytes": 0,
                },
            )
            thresholds = receipt.payload["thresholds"]
            self.assertEqual(
                thresholds["remaining_worst_case_output_bytes"],
                FORMAL_MAX_OUTPUT_BYTES,
            )
            self.assertEqual(
                thresholds["required_free_disk_bytes"],
                30 * 1024**3,
            )
            self.assertEqual(
                receipt.payload["authority"]["provider_readiness_requests"], 0
            )
            self.assertEqual(
                receipt.payload["provider_credential"]["side_effects"][
                    "network_requests"
                ],
                0,
            )
            persisted = config.base.receipt_path.read_text(encoding="utf-8")
            self.assertNotIn("unit-test-secret", persisted)
            self.assertNotIn("provider.env", persisted)
            loaded = load_stair_generation_launch_readiness_receipt(
                config.base.receipt_path,
                **self._loader_kwargs(config.base),
                expected_subject_mode=STAIR_FORMAL_LAUNCH_SUBJECT_FRESH_ABSENT,
                expected_formal_generation_command_sha256=(
                    config.formal_generation_command_sha256
                ),
            )
            self.assertEqual(loaded.receipt_hash, receipt.receipt_hash)

            tampered = receipt.to_dict()
            tampered["thresholds"]["remaining_worst_case_output_bytes"] -= 1
            body = dict(tampered)
            del body["receipt_hash"]
            tampered["receipt_hash"] = canonical_json_hash(body)
            tampered_path = Path(tmp) / "tampered-v3.json"
            tampered_path.write_bytes(
                canonical_json_bytes(tampered, trailing_newline=True)
            )
            with self.assertRaisesRegex(ValueError, "disk formula"):
                load_stair_generation_launch_readiness_receipt(
                    tampered_path,
                    **self._loader_kwargs(config.base),
                    expected_subject_mode=(
                        STAIR_FORMAL_LAUNCH_SUBJECT_FRESH_ABSENT
                    ),
                    expected_formal_generation_command_sha256=(
                        config.formal_generation_command_sha256
                    ),
                )

    def test_v3_binds_nonterminal_finalization_and_audit_only_subjects(self) -> None:
        modes = (
            STAIR_FORMAL_LAUNCH_SUBJECT_NONTERMINAL_RESUME,
            STAIR_FORMAL_LAUNCH_SUBJECT_TERMINAL_FINALIZATION,
            STAIR_FORMAL_LAUNCH_SUBJECT_TERMINAL_AUDIT_ONLY,
        )
        for mode in modes:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                config = self._config_v3(root, subject_mode=mode)
                output_dir = config.base.formal_output_dir
                output_dir.mkdir()
                subject = self._formal_subject(output_dir, subject_mode=mode)
                before = {
                    path.relative_to(output_dir): path.read_bytes()
                    for path in output_dir.rglob("*")
                    if path.is_file()
                }
                receipt = self._collect_v3(
                    config,
                    formal_subject_inspector=lambda _config, value=subject: value,
                )
                after = {
                    path.relative_to(output_dir): path.read_bytes()
                    for path in output_dir.rglob("*")
                    if path.is_file()
                }
                self.assertEqual(after, before)
                self.assertEqual(receipt.payload["subject"]["subject_mode"], mode)
                self.assertFalse(
                    receipt.payload["observations"]["formal_output_absent_before"]
                )
                current = subject["current_accounted_output_bytes"]
                self.assertEqual(
                    receipt.payload["thresholds"][
                        "remaining_worst_case_output_bytes"
                    ],
                    FORMAL_MAX_OUTPUT_BYTES - current,
                )
                with mock.patch(
                    "true_rl_agent.tasks.stair_climbing.launch_readiness."
                    "_inspect_v3_launch_subject_values",
                    return_value=subject,
                ):
                    loaded = load_stair_generation_launch_readiness_receipt(
                        config.base.receipt_path,
                        **self._loader_kwargs(config.base),
                        expected_subject_mode=mode,
                        expected_formal_generation_command_sha256=(
                            config.formal_generation_command_sha256
                        ),
                    )
                self.assertEqual(loaded.receipt_hash, receipt.receipt_hash)

    def test_v3_rejects_missing_credential_and_wrong_frozen_output_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self._config_v3(
                root,
                subject_mode=STAIR_FORMAL_LAUNCH_SUBJECT_FRESH_ABSENT,
            )
            config.provider_env_file.write_text("OTHER=value\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "credential is missing"):
                self._collect_v3(config)
            self.assertFalse(config.base.receipt_path.exists())

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self._config_v3(
                root,
                subject_mode=STAIR_FORMAL_LAUNCH_SUBJECT_FRESH_ABSENT,
            )
            with self.assertRaisesRegex(ValueError, "frozen P6-D output cap"):
                replace(
                    config,
                    base=replace(
                        config.base,
                        max_output_bytes=FORMAL_MAX_OUTPUT_BYTES - 1,
                    ),
                )

    def test_cli_is_v3_only_and_defaults_to_ten_minutes_at_one_second(self) -> None:
        script = (
            Path(__file__).resolve().parents[2]
            / "scripts"
            / "run_stair_generation_launch_readiness.py"
        )
        spec = importlib.util.spec_from_file_location(
            "stair_launch_readiness_cli", script
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        args = module._parse_args(
            [
                "--receipt-path",
                "receipt.json",
                "--formal-output-dir",
                "formal",
                "--session-id",
                "session",
                "--expected-source-commit",
                _COMMIT,
                "--expected-machine-id",
                _MACHINE_ID,
                "--gpu0-uuid",
                _GPU_UUIDS[0],
                "--gpu1-uuid",
                _GPU_UUIDS[1],
                "--lease-root",
                "leases",
                "--max-output-bytes",
                "1",
                "--nvidia-smi-executable",
                "nvidia-smi",
                "--subject-mode",
                "formal_fresh_absent",
                "--provider-env-file",
                "provider.env",
                "--formal-generation-command-sha256",
                "sha256:" + "9" * 64,
                "--attempt-chain-fault-probe-report",
                "attempt-chain/report.json",
                "--stair-train-teardown-result",
                "teardown/result.json",
            ]
        )
        self.assertEqual(args.sample_duration_seconds, DEFAULT_SAMPLE_DURATION_SECONDS)
        self.assertEqual(args.sample_interval_seconds, DEFAULT_SAMPLE_INTERVAL_SECONDS)
        self.assertEqual(args.subject_mode, "formal_fresh_absent")
        self.assertEqual(args.provider_env_file, Path("provider.env"))
        self.assertEqual(
            args.formal_generation_command_sha256,
            "sha256:" + "9" * 64,
        )
        self.assertEqual(
            args.attempt_chain_fault_probe_report,
            Path("attempt-chain/report.json"),
        )
        self.assertEqual(
            args.stair_train_teardown_result,
            Path("teardown/result.json"),
        )


if __name__ == "__main__":
    unittest.main()
