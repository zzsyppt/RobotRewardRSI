from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from true_rl_agent.evaluation.binding import canonical_json_hash  # noqa: E402
from true_rl_agent.json_contract import canonical_json_bytes  # noqa: E402
from true_rl_agent.runner.base import (  # noqa: E402
    RunnerExecutionError,
    RunnerFailureKind,
)
from true_rl_agent.runner.gpu_lease import (  # noqa: E402
    GPU_LEASE_SCHEMA_VERSION,
    PRE_GENESIS_GPU_LEASE_SCHEMA_VERSION,
    GpuDeviceIdentity,
    GpuLeaseHolder,
    GpuLeaseManager,
    GpuLeaseUnavailableError,
    PreGenesisGpuLeaseHolder,
)
from true_rl_agent.runner.job_argv import (  # noqa: E402
    validate_single_device_job_argv,
)
from true_rl_agent.runner.job_attempts import (  # noqa: E402
    PRE_GENESIS_EXECUTION_SCOPE_SCHEMA_VERSION,
    RESOURCE_LEASE_ATTEMPT_RESULT_SCHEMA_VERSION,
    JobAttemptJournal,
    JobAttemptScope,
    PreGenesisExecutionScope,
)
from true_rl_agent.runner.process_handle import (  # noqa: E402
    OwnedProcessLaunchError,
    OwnedProcessTimeout,
    _cleanup_preserving_exception,
    run_owned_process,
)
from true_rl_agent.runner.subprocess_runner import SubprocessRunner  # noqa: E402


_DIGEST_A = "a" * 64
_DIGEST_B = "b" * 64
_DIGEST_C = "c" * 64
_DIGEST_D = "d" * 64


class _FakeResult:
    def __init__(self, value: str) -> None:
        self.value = value

    def to_public_dict(self) -> dict[str, str]:
        return {"value": self.value}


class _FakeSpec:
    def __init__(
        self,
        scope: JobAttemptScope,
        *,
        execution_device: str = "cpu",
    ) -> None:
        self.job_id = "logical-job"
        self.execution_device = execution_device
        self.attempt_scope = scope
        self.metadata = {"backend_id": "fake"}

    def to_public_dict(self) -> dict[str, str]:
        return {"job_id": self.job_id, "execution_device": self.execution_device}


def _scope(root: Path, *, generation_id: str = "generation-001") -> JobAttemptScope:
    return JobAttemptScope(
        journal_root=root,
        session_genesis_hash=_DIGEST_A,
        session_contract_hash=_DIGEST_B,
        generation_id=generation_id,
        lane_id="lane-a",
        invocation_segment_id="segment-000",
        budget_contract_hash=_DIGEST_C,
        attempt_budget_units=2,
        max_infrastructure_retries=1,
    )


def _journal(root: Path, *, generation_id: str = "generation-001") -> JobAttemptJournal:
    return JobAttemptJournal(
        scope=_scope(root, generation_id=generation_id),
        logical_job_id="logical-job",
        job_kind="train",
        job_input_hash=_DIGEST_D,
        environment_hash=_DIGEST_A,
        execution_device="cpu",
        resource_binding_hash=None,
    )


def _pre_genesis_scope(root: Path) -> PreGenesisExecutionScope:
    return PreGenesisExecutionScope(
        journal_root=root,
        session_id="fresh-session",
        genesis_identity_hash=_DIGEST_A,
        session_contract_hash=_DIGEST_B,
        lane_id="lane-a",
        invocation_segment_id="session-genesis",
        budget_contract_hash=_DIGEST_C,
        attempt_budget_units=2,
        max_infrastructure_retries=1,
    )


def _holder(*, logical_job_id: str = "logical-job") -> GpuLeaseHolder:
    return GpuLeaseHolder(
        session_genesis_hash=_DIGEST_A,
        generation_id="generation-001",
        lane_id="lane-a",
        invocation_segment_id="segment-000",
        logical_job_id=logical_job_id,
    )


def _pid_is_alive(pid: int) -> bool:
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    handle = kernel32.OpenProcess(0x1000, False, pid)
    if not handle:
        return False
    try:
        exit_code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return False
        return exit_code.value == 259
    finally:
        kernel32.CloseHandle(handle)


class RunnerProcessHandleTests(unittest.TestCase):
    def test_environment_overrides_are_scoped_to_owned_child(self) -> None:
        override_key = "TRUE_RL_AGENT_TEST_PROCESS_OVERRIDE"
        removed_key = "TRUE_RL_AGENT_TEST_PROCESS_REMOVED"
        overrides = {override_key: "child-value", removed_key: None}
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
            os.environ,
            {override_key: "parent-value", removed_key: "parent-value"},
        ):
            result = run_owned_process(
                [
                    sys.executable,
                    "-c",
                    (
                        "import json, os; "
                        f"print(json.dumps({{{override_key!r}: os.environ.get({override_key!r}), "
                        f"{removed_key!r}: os.environ.get({removed_key!r})}}))"
                    ),
                ],
                cwd=ROOT,
                timeout_s=10,
                log_dir=Path(tmp),
                environment_overrides=overrides,
            )

            self.assertEqual(
                json.loads(result.stdout_tail),
                {override_key: "child-value", removed_key: None},
            )
            self.assertEqual(os.environ[override_key], "parent-value")
            self.assertEqual(os.environ[removed_key], "parent-value")
            self.assertEqual(
                overrides,
                {override_key: "child-value", removed_key: None},
            )

    def test_normal_process_streams_to_logs_and_returns_bounded_tails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_owned_process(
                [
                    sys.executable,
                    "-c",
                    "import sys; print('A' * 20000); print('stderr-ok', file=sys.stderr)",
                ],
                cwd=ROOT,
                timeout_s=10,
                log_dir=Path(tmp),
                termination_grace_s=1,
            )

            self.assertEqual(result.returncode, 0)
            self.assertLessEqual(len(result.stdout_tail.encode("utf-8")), 16 * 1024)
            self.assertIn("stderr-ok", result.stderr_tail)
            self.assertGreater(result.stdout_log.stat().st_size, 16 * 1024)

    def test_timeout_cleans_direct_child_and_grandchild(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            child_pid_path = root / "child.pid"
            parent_source = (
                "import pathlib, subprocess, sys, time; "
                "child=subprocess.Popen([sys.executable, '-c', "
                "'import time; time.sleep(60)']); "
                "pathlib.Path(sys.argv[1]).write_text(str(child.pid), encoding='utf-8'); "
                "time.sleep(60)"
            )
            with self.assertRaises(OwnedProcessTimeout):
                run_owned_process(
                    [sys.executable, "-c", parent_source, str(child_pid_path)],
                    cwd=ROOT,
                    timeout_s=1,
                    log_dir=root / "logs",
                    termination_grace_s=0.2,
                )
            self.assertTrue(child_pid_path.is_file())
            child_pid = int(child_pid_path.read_text(encoding="utf-8"))
            deadline = time.monotonic() + 3.0
            while _pid_is_alive(child_pid) and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertFalse(_pid_is_alive(child_pid))

    def test_launch_failure_is_typed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(OwnedProcessLaunchError):
                run_owned_process(
                    [str(Path(tmp) / "missing-executable")],
                    cwd=ROOT,
                    timeout_s=1,
                    log_dir=Path(tmp) / "logs",
                )

    def test_cleanup_failure_does_not_replace_original_without_add_note(self) -> None:
        class NoNoteError(RuntimeError):
            add_note = None

        original = NoNoteError("original")
        with mock.patch(
            "true_rl_agent.runner.process_handle._terminate_process_tree",
            side_effect=OSError("cleanup"),
        ):
            _cleanup_preserving_exception(
                mock.Mock(),
                windows_job=None,
                grace_s=0,
                original_exception=original,
            )
        self.assertEqual(str(original), "original")

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux parent-death contract")
    def test_linux_parent_death_signal_stops_owned_job(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target_pid_path = root / "target.pid"
            target_source = (
                "import os, pathlib, sys, time; "
                "pathlib.Path(sys.argv[1]).write_text(str(os.getpid()), encoding='utf-8'); "
                "time.sleep(60)"
            )
            supervisor_source = f"""
import sys
from pathlib import Path
from true_rl_agent.runner.process_handle import run_owned_process
run_owned_process(
    [sys.executable, '-c', {target_source!r}, {str(target_pid_path)!r}],
    cwd=Path({str(ROOT)!r}),
    timeout_s=60,
    log_dir=Path({str(root / 'logs')!r}),
)
"""
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(SRC)
            supervisor = subprocess.Popen(
                [sys.executable, "-c", supervisor_source],
                cwd=ROOT,
                env=environment,
            )
            try:
                deadline = time.monotonic() + 10
                while not target_pid_path.is_file() and time.monotonic() < deadline:
                    time.sleep(0.05)
                self.assertTrue(target_pid_path.is_file())
                target_pid = int(target_pid_path.read_text(encoding="utf-8"))
                supervisor.kill()
                supervisor.wait(timeout=10)
                deadline = time.monotonic() + 5
                while _pid_is_alive(target_pid) and time.monotonic() < deadline:
                    time.sleep(0.05)
                self.assertFalse(_pid_is_alive(target_pid))
            finally:
                if supervisor.poll() is None:
                    supervisor.kill()
                    supervisor.wait(timeout=10)


class JobAttemptJournalTests(unittest.TestCase):
    def test_pre_genesis_scope_round_trips_without_changing_legacy_projection(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy = _scope(root)
            self.assertEqual(
                legacy.binding_payload(),
                {
                    "session_genesis_hash": _DIGEST_A,
                    "session_contract_hash": _DIGEST_B,
                    "generation_id": "generation-001",
                    "lane_id": "lane-a",
                    "invocation_segment_id": "segment-000",
                    "budget_contract_hash": _DIGEST_C,
                    "attempt_budget_units": 2,
                    "max_infrastructure_retries": 1,
                    "resource_binding_hash": None,
                },
            )

            scope = _pre_genesis_scope(root)
            journal = JobAttemptJournal(
                scope=scope,
                logical_job_id="reference-train",
                job_kind="train",
                job_input_hash=_DIGEST_D,
                environment_hash=_DIGEST_A,
                execution_device="cpu",
                resource_binding_hash=None,
            )
            reservation = journal.reserve(0, previous_result=None)
            reconstructed, loaded = JobAttemptJournal.from_reservation_path(
                reservation.path
            )

            self.assertEqual(reconstructed.scope, scope)
            self.assertEqual(loaded.payload_hash, reservation.payload_hash)
            self.assertEqual(
                reservation.payload["scope"]["schema_version"],
                PRE_GENESIS_EXECUTION_SCOPE_SCHEMA_VERSION,
            )
            self.assertNotIn("session_genesis_hash", reservation.payload["scope"])
            self.assertNotIn("generation_id", reservation.payload["scope"])

    def test_retry_chain_is_create_once_and_hash_bound(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            journal = _journal(Path(tmp))
            reservation0 = journal.reserve(0, previous_result=None)
            result0 = journal.finish(
                reservation0,
                outcome="infrastructure_failure",
                failure_kind=RunnerFailureKind.SUMMARY_MISSING.value,
                retryable=True,
                output_hash=None,
                returncode=1,
            )
            reservation1 = journal.reserve(1, previous_result=result0)
            result1 = journal.finish(
                reservation1,
                outcome="completed",
                failure_kind=None,
                retryable=False,
                output_hash=_DIGEST_D,
                returncode=0,
            )

            self.assertEqual(
                reservation1.payload["previous_attempt_result_hash"],
                result0.payload_hash,
            )
            self.assertEqual(journal.load_result(1).payload_hash, result1.payload_hash)
            self.assertEqual(
                reservation0.payload_hash,
                canonical_json_hash(reservation0.payload),
            )
            self.assertEqual(
                journal.load_reusable_output_result().payload_hash,
                result1.payload_hash,
            )
            self.assertEqual(
                journal.reserve(0, previous_result=None).payload_hash,
                reservation0.payload_hash,
            )
            self.assertFalse(journal.reserve(0, previous_result=None).created)

    def test_bound_gpu_attempt_records_validated_lease_wall_duration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scope = JobAttemptScope(
                journal_root=root,
                session_genesis_hash=_DIGEST_A,
                session_contract_hash=_DIGEST_B,
                generation_id="generation-001",
                lane_id="lane-a",
                invocation_segment_id="segment-000",
                budget_contract_hash=_DIGEST_C,
                attempt_budget_units=1,
                max_infrastructure_retries=0,
                resource_binding_hash=_DIGEST_D,
            )
            journal = JobAttemptJournal(
                scope=scope,
                logical_job_id="gpu-job",
                job_kind="train",
                job_input_hash=_DIGEST_A,
                environment_hash=_DIGEST_B,
                execution_device="cuda:0",
                resource_binding_hash=_DIGEST_D,
            )
            reservation = journal.reserve(0, previous_result=None)
            result = journal.finish(
                reservation,
                outcome="completed",
                failure_kind=None,
                retryable=False,
                output_hash=_DIGEST_C,
                returncode=0,
                resource_lease_wall_duration_ns=123,
            )

            self.assertEqual(
                result.payload["schema_version"],
                RESOURCE_LEASE_ATTEMPT_RESULT_SCHEMA_VERSION,
            )
            self.assertEqual(
                journal.load_result(0).payload["resource_lease_wall_duration_ns"],
                123,
            )

    def test_non_retryable_result_and_open_reservation_cannot_authorize_retry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            journal = _journal(Path(tmp))
            reservation = journal.reserve(0, previous_result=None)
            with self.assertRaisesRegex(ValueError, "requires the previous"):
                journal.reserve(1, previous_result=None)
            result = journal.finish(
                reservation,
                outcome="contract_failure",
                failure_kind=RunnerFailureKind.CONTRACT_INVALID.value,
                retryable=False,
                output_hash=None,
                returncode=None,
            )
            with self.assertRaisesRegex(ValueError, "infrastructure failure"):
                journal.reserve(1, previous_result=result)

    def test_foreign_reservation_cannot_close_another_logical_job(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = _journal(root, generation_id="generation-001")
            second = _journal(root, generation_id="generation-002")
            foreign = first.reserve(0, previous_result=None)
            with self.assertRaisesRegex(ValueError, "different logical job"):
                second.finish(
                    foreign,
                    outcome="completed",
                    failure_kind=None,
                    retryable=False,
                    output_hash=_DIGEST_D,
                    returncode=0,
                )

    def test_attempt_result_unknown_field_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            journal = _journal(Path(tmp))
            reservation = journal.reserve(0, previous_result=None)
            result = journal.finish(
                reservation,
                outcome="completed",
                failure_kind=None,
                retryable=False,
                output_hash=_DIGEST_D,
                returncode=0,
            )
            tampered = dict(result.payload)
            tampered["unexpected"] = True
            result.path.write_bytes(
                canonical_json_bytes(tampered, trailing_newline=True)
            )
            with self.assertRaisesRegex(ValueError, "fields mismatch"):
                journal.load_result(0)

    def test_runner_retries_only_a_registered_infrastructure_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scope = _scope(Path(tmp))
            spec = _FakeSpec(scope)
            runner = SubprocessRunner((sys.executable,), timeout_s=10)
            calls = 0

            def operation(log_dir: Path | None, execution_context) -> _FakeResult:
                nonlocal calls
                self.assertIsNotNone(log_dir)
                self.assertIsNone(execution_context)
                calls += 1
                if calls == 1:
                    raise RunnerExecutionError(
                        "summary absent",
                        failure_kind=RunnerFailureKind.SUMMARY_MISSING,
                    )
                return _FakeResult("ok")

            result = runner._execute_with_attempts(
                spec,
                job_kind="train",
                operation=operation,
            )
            self.assertEqual(result.to_public_dict(), {"value": "ok"})
            self.assertEqual(calls, 2)
            receipt_paths = sorted(Path(tmp).rglob("attempt_*.result.json"))
            self.assertEqual(len(receipt_paths), 2)
            payloads = [json.loads(path.read_text(encoding="utf-8")) for path in receipt_paths]
            self.assertEqual(
                [payload["outcome"] for payload in payloads],
                ["infrastructure_failure", "completed"],
            )

    def test_runner_derives_pre_genesis_gpu_lease_holder_from_attempt_scope(
        self,
    ) -> None:
        class RecordingGpuLeaseManager(GpuLeaseManager):
            last_holder = None

            def acquire(self, execution_device, *, holder):
                self.last_holder = holder
                return super().acquire(execution_device, holder=holder)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            identity = GpuDeviceIdentity("cuda:0", "machine", "GPU-AAA")
            manager = RecordingGpuLeaseManager(
                lease_root=root / "leases",
                devices={"cuda:0": identity},
            )
            scope = PreGenesisExecutionScope(
                journal_root=root / "attempts",
                session_id="fresh-session",
                genesis_identity_hash=_DIGEST_A,
                session_contract_hash=_DIGEST_B,
                lane_id="lane-a",
                invocation_segment_id="session-genesis",
                budget_contract_hash=_DIGEST_C,
                attempt_budget_units=1,
                max_infrastructure_retries=0,
                resource_binding_hash=identity.identity_hash,
            )
            spec = _FakeSpec(scope, execution_device="cuda:0")
            runner = SubprocessRunner(
                (sys.executable,),
                timeout_s=10,
                gpu_lease_manager=manager,
            )

            result = runner._execute_with_attempts(
                spec,
                job_kind="train",
                operation=lambda log_dir, execution_context: _FakeResult("ok"),
            )

            self.assertEqual(result.to_public_dict(), {"value": "ok"})
            self.assertIsInstance(manager.last_holder, PreGenesisGpuLeaseHolder)
            self.assertEqual(manager.last_holder.session_id, "fresh-session")
            self.assertEqual(
                manager.last_holder.genesis_identity_hash,
                _DIGEST_A,
            )

    def test_runner_does_not_retry_contract_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            spec = _FakeSpec(_scope(Path(tmp)))
            runner = SubprocessRunner((sys.executable,), timeout_s=10)
            calls = 0

            def operation(log_dir: Path | None, execution_context) -> _FakeResult:
                nonlocal calls
                self.assertIsNone(execution_context)
                calls += 1
                raise RunnerExecutionError(
                    "bad binding",
                    failure_kind=RunnerFailureKind.CONTRACT_INVALID,
                )

            with self.assertRaises(RunnerExecutionError):
                runner._execute_with_attempts(
                    spec,
                    job_kind="train",
                    operation=operation,
                )
            self.assertEqual(calls, 1)
            self.assertEqual(len(list(Path(tmp).rglob("attempt_*.reservation.json"))), 1)

    def test_runner_does_not_reexecute_an_existing_attempt_without_inspector(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            spec = _FakeSpec(_scope(Path(tmp)))
            runner = SubprocessRunner((sys.executable,), timeout_s=10)
            journal = JobAttemptJournal(
                scope=spec.attempt_scope,
                logical_job_id=spec.job_id,
                job_kind="train",
                job_input_hash=runner._job_input_hash(spec, job_kind="train"),
                environment_hash=runner._job_environment_hash(spec),
                execution_device="cpu",
                resource_binding_hash=None,
            )
            journal.reserve(0, previous_result=None)
            calls = 0

            def operation(log_dir: Path | None, execution_context) -> _FakeResult:
                nonlocal calls
                self.assertIsNone(execution_context)
                calls += 1
                return _FakeResult("should-not-run")

            with self.assertRaisesRegex(
                RunnerExecutionError,
                "requires recovery inspection",
            ):
                runner._execute_with_attempts(
                    spec,
                    job_kind="train",
                    operation=operation,
                )
            self.assertEqual(calls, 0)

    def test_runner_reuses_a_hash_bound_completed_output_without_reexecution(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            spec = _FakeSpec(_scope(Path(tmp)))
            runner = SubprocessRunner((sys.executable,), timeout_s=10)
            execution_calls = 0
            recovery_calls = 0

            def operation(log_dir: Path | None, execution_context) -> _FakeResult:
                nonlocal execution_calls
                self.assertIsNotNone(log_dir)
                self.assertIsNone(execution_context)
                execution_calls += 1
                return _FakeResult("sealed-output")

            def recover(execution_context) -> _FakeResult:
                nonlocal recovery_calls
                self.assertIsNone(execution_context)
                recovery_calls += 1
                return _FakeResult("sealed-output")

            first = runner._execute_with_attempts(
                spec,
                job_kind="train",
                operation=operation,
                recover_completed=recover,
            )
            second = runner._execute_with_attempts(
                spec,
                job_kind="train",
                operation=operation,
                recover_completed=recover,
            )

            self.assertEqual(first.to_public_dict(), second.to_public_dict())
            self.assertEqual(execution_calls, 1)
            self.assertEqual(recovery_calls, 1)
            with self.assertRaises(RunnerExecutionError) as raised:
                runner._execute_with_attempts(
                    spec,
                    job_kind="train",
                    operation=operation,
                    recover_completed=lambda execution_context: _FakeResult(
                        "tampered-output"
                    ),
                )
            self.assertEqual(
                raised.exception.failure_kind,
                RunnerFailureKind.CONTRACT_INVALID,
            )
            self.assertEqual(execution_calls, 1)


class GpuLeaseTests(unittest.TestCase):
    def test_pre_genesis_holder_uses_v2_without_changing_v1_holder_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            identity = GpuDeviceIdentity("cuda:0", "machine", "GPU-AAA")
            manager = GpuLeaseManager(
                lease_root=Path(tmp),
                devices={"cuda:0": identity},
            )
            legacy = manager.acquire("cuda:0", holder=_holder())
            try:
                self.assertEqual(
                    legacy.receipt["schema_version"],
                    GPU_LEASE_SCHEMA_VERSION,
                )
                self.assertEqual(
                    legacy.receipt["holder"],
                    {
                        "session_genesis_hash": _DIGEST_A,
                        "generation_id": "generation-001",
                        "lane_id": "lane-a",
                        "invocation_segment_id": "segment-000",
                        "logical_job_id": "logical-job",
                    },
                )
            finally:
                legacy.release()

            pre_genesis = manager.acquire(
                "cuda:0",
                holder=PreGenesisGpuLeaseHolder(
                    session_id="fresh-session",
                    genesis_identity_hash=_DIGEST_A,
                    session_contract_hash=_DIGEST_B,
                    lane_id="lane-a",
                    invocation_segment_id="session-genesis",
                    logical_job_id="reference-train",
                ),
            )
            try:
                self.assertEqual(
                    pre_genesis.receipt["schema_version"],
                    PRE_GENESIS_GPU_LEASE_SCHEMA_VERSION,
                )
                self.assertNotIn(
                    "session_genesis_hash",
                    pre_genesis.receipt["holder"],
                )
                self.assertEqual(
                    pre_genesis.receipt["holder"]["genesis_identity_hash"],
                    _DIGEST_A,
                )
            finally:
                pre_genesis.release()

    def test_same_physical_gpu_aliases_conflict_but_distinct_gpus_do_not(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cuda0 = GpuDeviceIdentity("cuda:0", "machine", "GPU-AAA")
            cuda1_alias = GpuDeviceIdentity("cuda:1", "machine", "GPU-AAA")
            cuda2 = GpuDeviceIdentity("cuda:2", "machine", "GPU-BBB")
            manager = GpuLeaseManager(
                lease_root=root,
                devices={"cuda:0": cuda0, "cuda:1": cuda1_alias, "cuda:2": cuda2},
            )
            self.assertNotEqual(cuda0.identity_hash, cuda1_alias.identity_hash)
            self.assertEqual(
                cuda0.physical_resource_hash,
                cuda1_alias.physical_resource_hash,
            )
            lease0 = manager.acquire("cuda:0", holder=_holder(logical_job_id="a"))
            try:
                with self.assertRaises(GpuLeaseUnavailableError):
                    manager.acquire("cuda:1", holder=_holder(logical_job_id="b"))
                lease2 = manager.acquire("cuda:2", holder=_holder(logical_job_id="c"))
                lease2.release()
            finally:
                lease0.release()
            manager.acquire("cuda:1", holder=_holder(logical_job_id="d")).release()

    def test_cross_process_gpu_lease_conflicts_and_recovers_after_abnormal_exit(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ready = root / "ready"
            code = f"""
import os
import time
from pathlib import Path
from true_rl_agent.runner.gpu_lease import GpuDeviceIdentity, GpuLeaseHolder, GpuLeaseManager
manager = GpuLeaseManager(
    lease_root=Path({str(root)!r}),
    devices={{'cuda:0': GpuDeviceIdentity('cuda:0', 'machine', 'GPU-AAA')}},
)
lease = manager.acquire(
    'cuda:0',
    holder=GpuLeaseHolder(
        session_genesis_hash={'a' * 64!r},
        generation_id='generation-001',
        lane_id='lane-a',
        invocation_segment_id='segment-000',
        logical_job_id='child-job',
    ),
)
Path({str(ready)!r}).write_text('ready', encoding='utf-8')
time.sleep(60)
"""
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(SRC)
            child = subprocess.Popen(
                [sys.executable, "-c", code],
                cwd=ROOT,
                env=environment,
            )
            manager = GpuLeaseManager(
                lease_root=root,
                devices={
                    "cuda:0": GpuDeviceIdentity("cuda:0", "machine", "GPU-AAA")
                },
            )
            try:
                deadline = time.monotonic() + 10
                while not ready.is_file() and time.monotonic() < deadline:
                    time.sleep(0.05)
                self.assertTrue(ready.is_file())
                with self.assertRaises(GpuLeaseUnavailableError):
                    manager.acquire("cuda:0", holder=_holder())
            finally:
                if child.poll() is None:
                    child.kill()
                child.wait(timeout=10)
            deadline = time.monotonic() + 3
            while True:
                try:
                    recovered = manager.acquire("cuda:0", holder=_holder())
                    break
                except GpuLeaseUnavailableError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(0.05)
            recovered.release()

    def test_attempt_scope_cannot_drift_from_trusted_gpu_binding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            identity = GpuDeviceIdentity("cuda:0", "machine", "GPU-AAA")
            manager = GpuLeaseManager(
                lease_root=root / "leases",
                devices={"cuda:0": identity},
            )
            mismatched_scope = JobAttemptScope(
                journal_root=root / "attempts",
                session_genesis_hash=_DIGEST_A,
                session_contract_hash=_DIGEST_B,
                generation_id="generation-001",
                lane_id="lane-a",
                invocation_segment_id="segment-000",
                budget_contract_hash=_DIGEST_C,
                attempt_budget_units=2,
                max_infrastructure_retries=1,
                resource_binding_hash=_DIGEST_D,
            )
            spec = _FakeSpec(mismatched_scope, execution_device="cuda:0")
            calls = 0

            def operation(log_dir: Path | None, execution_context) -> _FakeResult:
                nonlocal calls
                self.assertIsNone(execution_context)
                calls += 1
                return _FakeResult("should-not-run")

            with self.assertRaisesRegex(
                RunnerExecutionError,
                "attempt evidence contract is invalid",
            ):
                SubprocessRunner(
                    (sys.executable,),
                    timeout_s=10,
                    gpu_lease_manager=manager,
                )._execute_with_attempts(
                    spec,
                    job_kind="train",
                    operation=operation,
                )
            self.assertEqual(calls, 0)


class RunnerSingleDeviceContractTests(unittest.TestCase):
    def test_distributed_and_multi_gpu_flags_are_rejected(self) -> None:
        for argv in (
            ["python", "train.py", "--distributed"],
            ["python", "train.py", "--multi-gpu"],
            ["python", "train.py", "--world-size=2"],
            ["python", "train.py", "--local_rank", "1"],
        ):
            with self.subTest(argv=argv):
                with self.assertRaisesRegex(ValueError, "distributed or multi-GPU"):
                    validate_single_device_job_argv(argv)

        validate_single_device_job_argv(
            ["python", "train.py", "--device", "cuda:0"]
        )


if __name__ == "__main__":
    unittest.main()
