from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

from scripts.launch_stair_generation_session import (
    ATTEMPT_EXIT_SCHEMA_VERSION,
    ATTEMPT_START_SCHEMA_VERSION,
    P6D_READINESS_CONSUMPTION_SCHEMA_VERSION,
    _consume_p6d_readiness_once,
    _source_identity,
    _formal_generation_identity,
    main,
)

from true_rl_agent.runner.process_handle import (
    OwnedProcessCleanupIncomplete,
    OwnedProcessResult,
    _linux_live_process_group_members,
)
from true_rl_agent.tasks.stair_climbing.launch_context import (
    STAIR_GENERATION_LAUNCH_START_PATH_ENV,
    STAIR_GENERATION_LAUNCH_START_SHA256_ENV,
)
from true_rl_agent.tasks.stair_climbing.launch_readiness import (
    STAIR_FORMAL_LAUNCH_SUBJECT_FRESH_ABSENT,
    STAIR_FORMAL_LAUNCH_SUBJECT_NONTERMINAL_RESUME,
    STAIR_FORMAL_LAUNCH_SUBJECT_TERMINAL_AUDIT_ONLY,
    STAIR_FORMAL_LAUNCH_SUBJECT_TERMINAL_FINALIZATION,
    STAIR_GENERATION_LAUNCH_READINESS_V3_SCHEMA_VERSION,
)

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
COMMIT = "a" * 40
GPU_A = "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
GPU_B = "GPU-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
BOOT_ID = "11111111-2222-3333-4444-555555555555"
class StairGenerationLauncherTests(unittest.TestCase):
    def test_p6d_readiness_consumption_is_create_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            receipt_path = Path(tmp) / "readiness.json"
            receipt_path.write_text("{}\n", encoding="utf-8")
            receipt_ref = {
                "schema_version": "true_rl_agent.stair.generation_launch_readiness_ref.v1",
                "path": str(receipt_path),
                "sha256": "sha256:" + "1" * 64,
                "size_bytes": 3,
                "receipt_hash": "2" * 64,
            }
            marker = _consume_p6d_readiness_once(
                receipt_path=receipt_path,
                receipt_ref=receipt_ref,
                formal_generation_command_sha256="sha256:" + "3" * 64,
            )
            payload = json.loads(marker.read_text(encoding="utf-8"))
            self.assertEqual(
                payload["schema_version"],
                P6D_READINESS_CONSUMPTION_SCHEMA_VERSION,
            )
            self.assertEqual(payload["readiness_receipt"], receipt_ref)
            self.assertEqual(payload["formal_output_writes"], 0)
            self.assertFalse(payload["acceptance_authority"])
            with self.assertRaisesRegex(FileExistsError, "already consumed"):
                _consume_p6d_readiness_once(
                    receipt_path=receipt_path,
                    receipt_ref=receipt_ref,
                    formal_generation_command_sha256="sha256:" + "3" * 64,
                )

    def test_p6d_launcher_selects_exact_v3_subject_modes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            with mock.patch("sys.stderr"), self.assertRaises(SystemExit):
                _formal_generation_identity(
                    [*self._formal_args(root), "--validation-session"]
                )
            args = list(self._formal_args(root))
            replacements = {
                "--max-allocated-active-gpu-hours": "50.5",
                "--max-simulated-control-steps": "6367795200",
                "--max-output-bytes": "14495514624",
            }
            for option, value in replacements.items():
                args[args.index(option) + 1] = value
            fresh = _formal_generation_identity(args)
            assert fresh is not None
            self.assertEqual(
                fresh["launch_subject_modes"],
                (STAIR_FORMAL_LAUNCH_SUBJECT_FRESH_ABSENT,),
            )
            resumed = _formal_generation_identity([*args, "--resume-existing"])
            assert resumed is not None
            self.assertEqual(
                resumed["launch_subject_modes"],
                (
                    STAIR_FORMAL_LAUNCH_SUBJECT_NONTERMINAL_RESUME,
                    STAIR_FORMAL_LAUNCH_SUBJECT_TERMINAL_FINALIZATION,
                ),
            )
            audit = _formal_generation_identity(
                [*args, "--resume-existing", "--terminal-audit-only"]
            )
            assert audit is not None
            self.assertEqual(
                audit["launch_subject_modes"],
                (STAIR_FORMAL_LAUNCH_SUBJECT_TERMINAL_AUDIT_ONLY,),
            )

    def test_source_identity_treats_untracked_execution_source_as_dirty(self) -> None:
        checkout = mock.Mock(head_commit=COMMIT, clean=False)
        with mock.patch(
            "scripts.launch_stair_generation_session.inspect_stair_source_checkout",
            return_value=checkout,
        ) as inspect_checkout:
            source = _source_identity()

        self.assertEqual(source["commit"], COMMIT)
        self.assertIs(source["tracked_clean"], False)
        inspect_checkout.assert_called_once_with(ROOT)

    def test_each_attempt_gets_an_exclusive_log_set_and_exit_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            attempt_parent = (Path(tmp) / "launcher-attempts").resolve()
            first_status, first_report = self._launch(
                attempt_parent,
                "--help",
            )
            first_attempt_path = Path(first_report["attempt_path"])
            first_bytes = {
                path.relative_to(first_attempt_path): path.read_bytes()
                for path in first_attempt_path.rglob("*")
                if path.is_file()
            }

            second_status, second_report = self._launch(
                attempt_parent,
                "--help",
            )

            self.assertEqual((first_status, second_status), (0, 0))
            self.assertNotEqual(first_report["attempt_id"], second_report["attempt_id"])
            self.assertEqual(
                {
                    path.relative_to(first_attempt_path): path.read_bytes()
                    for path in first_attempt_path.rglob("*")
                    if path.is_file()
                },
                first_bytes,
            )
            self.assertEqual(
                len([path for path in attempt_parent.iterdir() if path.is_dir()]),
                2,
            )
            for report in (first_report, second_report):
                receipt = json.loads(
                    Path(report["exit_receipt_path"]).read_text(encoding="utf-8")
                )
                self.assertEqual(receipt["schema_version"], ATTEMPT_EXIT_SCHEMA_VERSION)
                self.assertEqual(receipt["status"], "completed")
                self.assertEqual(receipt["returncode"], 0)
                self.assertIsInstance(receipt["process_id"], int)
                start_ref = receipt["start_receipt"]
                start_path = Path(start_ref["path"])
                start_bytes = start_path.read_bytes()
                self.assertEqual(start_ref["size_bytes"], len(start_bytes))
                self.assertEqual(
                    start_ref["sha256"],
                    "sha256:" + hashlib.sha256(start_bytes).hexdigest(),
                )
                start = json.loads(start_bytes)
                self.assertEqual(start["schema_version"], ATTEMPT_START_SCHEMA_VERSION)
                self.assertEqual(start["attempt_id"], receipt["attempt_id"])
                self.assertEqual(
                    start["child"]["command_sha256"], receipt["command_sha256"]
                )
                self.assertIsNone(start["launch_readiness_receipt"])
                for key in ("launcher", "stdout", "stderr"):
                    log_path = Path(receipt["logs"][key]["path"])
                    self.assertEqual(
                        receipt["logs"][key]["sha256"],
                        "sha256:" + hashlib.sha256(log_path.read_bytes()).hexdigest(),
                    )
                launcher_text = Path(report["launcher_log_path"]).read_text(
                    encoding="utf-8"
                )
                self.assertIn("launcher_attempt_started", launcher_text)
                self.assertIn("launcher_attempt_finished", launcher_text)
                child_text = Path(report["stdout_log_path"]).read_text(
                    encoding="utf-8"
                ) + Path(report["stderr_log_path"]).read_text(encoding="utf-8")
                self.assertIn("--preflight-only", child_text)

    def test_child_failure_is_preserved_in_its_own_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            status, report = self._launch(
                (Path(tmp) / "attempts").resolve(),
                "--definitely-invalid-option",
            )
            receipt = json.loads(
                Path(report["exit_receipt_path"]).read_text(encoding="utf-8")
            )

            self.assertEqual(status, 2)
            self.assertEqual(receipt["status"], "failed")
            self.assertEqual(receipt["returncode"], 2)
            self.assertIn(
                "error:",
                Path(report["stderr_log_path"]).read_text(encoding="utf-8"),
            )

    def test_child_python_traceback_is_preserved_verbatim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            helper = root / "failing-generation.py"
            helper.write_text(
                "raise RuntimeError('unique-generation-traceback')\n",
                encoding="utf-8",
            )
            with mock.patch(
                "scripts.launch_stair_generation_session.GENERATION_ENTRYPOINT",
                helper,
            ):
                status, report = self._launch(
                    root / "attempts",
                    "--unused-helper-argument",
                )

            self.assertEqual(status, 1)
            stderr = Path(report["stderr_log_path"]).read_text(encoding="utf-8")
            self.assertIn("Traceback (most recent call last)", stderr)
            self.assertIn("RuntimeError: unique-generation-traceback", stderr)
            receipt = json.loads(
                Path(report["exit_receipt_path"]).read_text(encoding="utf-8")
            )
            self.assertEqual(receipt["status"], "failed")
            self.assertEqual(receipt["returncode"], 1)

    def test_cleanup_incomplete_receipt_does_not_claim_stable_child_logs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stdout = StringIO()
            failure = OwnedProcessCleanupIncomplete(76543, (12345,))

            def fail_after_logs(*_args, **kwargs):
                log_dir = Path(kwargs["log_dir"])
                log_dir.mkdir(parents=True, exist_ok=False)
                (log_dir / "stdout.log").write_bytes(b"possibly-live-stdout\n")
                (log_dir / "stderr.log").write_bytes(b"possibly-live-stderr\n")
                raise failure

            with (
                mock.patch(
                    "scripts.launch_stair_generation_session.run_owned_process",
                    side_effect=fail_after_logs,
                ),
                redirect_stdout(stdout),
                self.assertRaises(OwnedProcessCleanupIncomplete) as raised,
            ):
                main(
                    [
                        "--python-executable",
                        sys.executable,
                        "--attempt-dir",
                        str((Path(tmp) / "attempts").resolve()),
                        "--timeout-s",
                        "30",
                        "--",
                        "--help",
                    ]
                )

            self.assertIs(raised.exception, failure)
            report = json.loads(stdout.getvalue())
            receipt = json.loads(
                Path(report["exit_receipt_path"]).read_text(encoding="utf-8")
            )
            self.assertEqual(receipt["status"], "cleanup_incomplete")
            self.assertIsNone(receipt["returncode"])
            for key in ("stdout", "stderr"):
                self.assertTrue(Path(receipt["logs"][key]["path"]).is_file())
                self.assertIsNone(receipt["logs"][key]["sha256"])
            launcher_log = receipt["logs"]["launcher"]
            launcher_text = Path(launcher_log["path"]).read_text(encoding="utf-8")
            self.assertIn("Traceback (most recent call last)", launcher_text)
            self.assertIn("OwnedProcessCleanupIncomplete", launcher_text)
            self.assertEqual(
                launcher_log["sha256"],
                "sha256:"
                + hashlib.sha256(Path(launcher_log["path"]).read_bytes()).hexdigest(),
            )

    def test_fresh_formal_launch_requires_readiness_before_attempt_side_effects(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            attempt_parent = root / "attempts"
            with (
                mock.patch(
                    "scripts.launch_stair_generation_session.run_owned_process"
                ) as spawn,
                self.assertRaisesRegex(ValueError, "readiness-receipt is required"),
            ):
                main(self._launcher_args(attempt_parent, *self._formal_args(root)))

            spawn.assert_not_called()
            self.assertFalse(attempt_parent.exists())
            self.assertFalse((root / "formal-output").exists())

    def test_nonterminal_resume_cannot_bypass_launch_readiness(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            attempt_parent = root / "attempts"
            (root / "formal-output").mkdir()
            generation_args = list(self._formal_args(root))
            replacements = {
                "--max-allocated-active-gpu-hours": "50.5",
                "--max-simulated-control-steps": "6367795200",
                "--max-output-bytes": "14495514624",
            }
            for option, value in replacements.items():
                generation_args[generation_args.index(option) + 1] = value

            with (
                mock.patch(
                    "scripts.launch_stair_generation_session.run_owned_process"
                ) as spawn,
                self.assertRaisesRegex(ValueError, "readiness-receipt is required"),
            ):
                main(
                    self._launcher_args(
                        attempt_parent,
                        *generation_args,
                        "--resume-existing",
                    )
                )

            spawn.assert_not_called()
            self.assertFalse(attempt_parent.exists())

    def test_formal_attempt_directory_must_be_outside_formal_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            attempt_parent = root / "formal-output" / "launcher-attempts"

            with (
                mock.patch(
                    "scripts.launch_stair_generation_session.run_owned_process"
                ) as spawn,
                self.assertRaisesRegex(ValueError, "must be outside output"),
            ):
                main(self._launcher_args(attempt_parent, *self._formal_args(root)))

            spawn.assert_not_called()
            self.assertFalse((root / "formal-output").exists())

    def test_fresh_formal_launch_binds_validated_readiness_and_full_start_argv(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            attempt_parent = root / "attempts"
            readiness_path = root / "readiness.json"
            readiness_path.write_text("{}\n", encoding="utf-8")
            provider_env = root / "provider.env"
            provider_secret = "provider-secret-must-not-enter-receipts"
            provider_env.write_text(
                f"DEEPSEEK_API_KEY={provider_secret}\n",
                encoding="utf-8",
            )
            readiness_ref = {
                "schema_version": (
                    "true_rl_agent.stair.generation_launch_readiness_ref.v1"
                ),
                "path": str(readiness_path),
                "sha256": "sha256:" + "1" * 64,
                "size_bytes": 123,
                "receipt_hash": "2" * 64,
            }
            readiness = mock.Mock()
            readiness.payload = {
                "schema_version": (
                    STAIR_GENERATION_LAUNCH_READINESS_V3_SCHEMA_VERSION
                )
            }
            readiness.assert_fresh = mock.Mock()
            readiness.to_ref.return_value.to_dict.return_value = readiness_ref
            generation_args = self._formal_args(root, provider_env=provider_env)
            launcher_args = self._launcher_args(
                attempt_parent,
                "--launch-readiness-receipt",
                str(readiness_path),
                *generation_args,
            )

            def complete_after_start(command, **kwargs):
                consumption_path = Path(str(readiness_path) + ".consumed.json")
                self.assertTrue(consumption_path.is_file())
                consumption = json.loads(
                    consumption_path.read_text(encoding="utf-8")
                )
                self.assertEqual(
                    consumption["schema_version"],
                    P6D_READINESS_CONSUMPTION_SCHEMA_VERSION,
                )
                self.assertEqual(consumption["readiness_receipt"], readiness_ref)
                attempts = [path for path in attempt_parent.iterdir() if path.is_dir()]
                self.assertEqual(len(attempts), 1)
                start_path = attempts[0] / "start.json"
                self.assertTrue(start_path.is_file())
                start = json.loads(start_path.read_text(encoding="utf-8"))
                self.assertEqual(start["launch_readiness_receipt"], readiness_ref)
                self.assertEqual(
                    start["launcher"]["argv"],
                    [
                        str(Path(sys.executable).resolve()),
                        str(
                            (
                                ROOT / "scripts" / "launch_stair_generation_session.py"
                            ).resolve()
                        ),
                        *launcher_args,
                    ],
                )
                self.assertEqual(start["child"]["argv"], list(command))
                self.assertEqual(
                    kwargs["environment_overrides"][
                        STAIR_GENERATION_LAUNCH_START_PATH_ENV
                    ],
                    str(start_path),
                )
                self.assertEqual(
                    kwargs["environment_overrides"][
                        STAIR_GENERATION_LAUNCH_START_SHA256_ENV
                    ],
                    "sha256:" + hashlib.sha256(start_path.read_bytes()).hexdigest(),
                )
                self.assertNotIn(
                    provider_secret, start_path.read_text(encoding="utf-8")
                )
                log_dir = Path(kwargs["log_dir"])
                log_dir.mkdir(parents=True, exist_ok=False)
                stdout_log = log_dir / "stdout.log"
                stderr_log = log_dir / "stderr.log"
                stdout_log.write_text("completed\n", encoding="utf-8")
                stderr_log.write_text("", encoding="utf-8")
                return OwnedProcessResult(
                    returncode=0,
                    pid=43210,
                    stdout_tail="completed\n",
                    stderr_tail="",
                    stdout_log=stdout_log,
                    stderr_log=stderr_log,
                )

            stdout = StringIO()
            with (
                mock.patch(
                    "scripts.launch_stair_generation_session._source_identity",
                    return_value={
                        "commit": COMMIT,
                        "tracked_clean": True,
                        "files": {},
                    },
                ),
                mock.patch(
                    "scripts.launch_stair_generation_session._host_identity",
                    return_value={
                        "hostname": "test-host",
                        "boot_id": BOOT_ID,
                        "machine_id": "test-machine",
                    },
                ),
                mock.patch(
                    "scripts.launch_stair_generation_session.load_stair_generation_launch_readiness_receipt",
                    return_value=readiness,
                ) as load_readiness,
                mock.patch(
                    "scripts.launch_stair_generation_session.run_owned_process",
                    side_effect=complete_after_start,
                ),
                redirect_stdout(stdout),
            ):
                status = main(launcher_args)

            self.assertEqual(status, 0)
            report = json.loads(stdout.getvalue())
            receipt = json.loads(
                Path(report["exit_receipt_path"]).read_text(encoding="utf-8")
            )
            self.assertFalse(
                (Path(report["attempt_path"]) / "runtime_monitor").exists()
            )
            self.assertEqual(
                receipt["start_receipt"]["path"], report["start_receipt_path"]
            )
            readiness.to_ref.assert_called_once_with(readiness_path)
            readiness.assert_fresh.assert_called_once_with()
            load_readiness.assert_called_once()
            loader_kwargs = load_readiness.call_args.kwargs
            self.assertEqual(
                loader_kwargs["expected_formal_output_dir"], root / "formal-output"
            )
            self.assertEqual(loader_kwargs["expected_source_commit"], COMMIT)
            self.assertEqual(loader_kwargs["expected_gpu_uuids"], (GPU_A, GPU_B))
            self.assertEqual(loader_kwargs["expected_hostname"], "test-host")
            self.assertEqual(loader_kwargs["expected_boot_id"], BOOT_ID)

    def test_invalid_readiness_binding_does_not_create_attempt_or_spawn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            attempt_parent = root / "attempts"
            with (
                mock.patch(
                    "scripts.launch_stair_generation_session._source_identity",
                    return_value={
                        "commit": COMMIT,
                        "tracked_clean": True,
                        "files": {},
                    },
                ),
                mock.patch(
                    "scripts.launch_stair_generation_session._host_identity",
                    return_value={
                        "hostname": "test-host",
                        "boot_id": BOOT_ID,
                        "machine_id": "test-machine",
                    },
                ),
                mock.patch(
                    "scripts.launch_stair_generation_session.load_stair_generation_launch_readiness_receipt",
                    side_effect=ValueError("binding mismatch"),
                ),
                mock.patch(
                    "scripts.launch_stair_generation_session.run_owned_process"
                ) as spawn,
                self.assertRaisesRegex(ValueError, "binding mismatch"),
            ):
                main(
                    self._launcher_args(
                        attempt_parent,
                        "--launch-readiness-receipt",
                        str(root / "wrong-readiness.json"),
                        *self._formal_args(root),
                    )
                )

            spawn.assert_not_called()
            self.assertFalse(attempt_parent.exists())

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux signal contract")
    def test_sigterm_cleans_child_group_before_publishing_stable_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            helper_path = root / "generation-helper.py"
            child_pid_path = root / "child.pid"
            grandchild_pid_path = root / "grandchild.pid"
            ready_path = root / "ready"
            helper_path.write_text(
                "import os,pathlib,signal,subprocess,sys,time\n"
                'grandchild_source = ("import signal,time; "\n'
                '    "signal.signal(signal.SIGTERM, signal.SIG_IGN); "\n'
                '    "time.sleep(60)")\n'
                "grandchild=subprocess.Popen([sys.executable, '-c', grandchild_source])\n"
                "pathlib.Path(sys.argv[1]).write_text(str(os.getpid()), encoding='utf-8')\n"
                "pathlib.Path(sys.argv[2]).write_text(str(grandchild.pid), encoding='utf-8')\n"
                "pathlib.Path(sys.argv[3]).write_text('ready', encoding='utf-8')\n"
                "time.sleep(60)\n",
                encoding="utf-8",
            )
            attempt_parent = root / "attempts"
            supervisor_source = f"""
import sys
from pathlib import Path
import scripts.launch_stair_generation_session as launcher
launcher.GENERATION_ENTRYPOINT = Path({str(helper_path)!r})
raise SystemExit(launcher.main([
    '--python-executable', sys.executable,
    '--attempt-dir', {str(attempt_parent)!r},
    '--timeout-s', '60', '--',
    {str(child_pid_path)!r}, {str(grandchild_pid_path)!r}, {str(ready_path)!r},
]))
"""
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(SRC)
            supervisor = subprocess.Popen(
                [sys.executable, "-c", supervisor_source],
                cwd=ROOT,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            child_pid: int | None = None
            try:
                deadline = time.monotonic() + 10.0
                while not ready_path.is_file() and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertTrue(ready_path.is_file())
                child_pid = int(child_pid_path.read_text(encoding="utf-8"))
                grandchild_pid = int(grandchild_pid_path.read_text(encoding="utf-8"))

                os.killpg(supervisor.pid, signal.SIGTERM)
                stdout, stderr = supervisor.communicate(timeout=20)

                self.assertEqual(supervisor.returncode, 143, stderr)
                report = json.loads(stdout.strip().splitlines()[-1])
                receipt_path = Path(report["exit_receipt_path"])
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                self.assertEqual(receipt["status"], "terminated")
                self.assertEqual(receipt["returncode"], 143)
                self.assertNotIn(
                    child_pid,
                    _linux_live_process_group_members(child_pid),
                )
                self.assertNotIn(
                    grandchild_pid,
                    _linux_live_process_group_members(child_pid),
                )
                stable_bytes = {
                    item["path"]: Path(item["path"]).read_bytes()
                    for item in receipt["logs"].values()
                }
                time.sleep(0.1)
                for path, payload in stable_bytes.items():
                    self.assertEqual(Path(path).read_bytes(), payload)
                    self.assertEqual(
                        "sha256:" + hashlib.sha256(payload).hexdigest(),
                        next(
                            item["sha256"]
                            for item in receipt["logs"].values()
                            if item["path"] == path
                        ),
                    )
            finally:
                if supervisor.poll() is None:
                    os.killpg(supervisor.pid, signal.SIGKILL)
                    supervisor.wait(timeout=10)
                if child_pid is not None:
                    try:
                        os.killpg(child_pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    def _launch(
        self,
        attempt_parent: Path,
        *generation_args: str,
    ) -> tuple[int, dict[str, object]]:
        stdout = StringIO()
        with redirect_stdout(stdout):
            status = main(
                [
                    "--python-executable",
                    sys.executable,
                    "--attempt-dir",
                    str(attempt_parent),
                    "--timeout-s",
                    "30",
                    "--",
                    *generation_args,
                ]
            )
        return status, json.loads(stdout.getvalue())

    @staticmethod
    def _launcher_args(attempt_parent: Path, *generation_args: str) -> list[str]:
        launcher_args = [
            "--python-executable",
            sys.executable,
            "--attempt-dir",
            str(attempt_parent),
            "--timeout-s",
            "30",
        ]
        if generation_args[:1] == ("--launch-readiness-receipt",):
            launcher_args.extend(generation_args[:2])
            generation_args = generation_args[2:]
        return [*launcher_args, "--", *generation_args]

    @staticmethod
    def _formal_args(
        root: Path,
        *,
        provider_env: Path | None = None,
    ) -> tuple[str, ...]:
        return (
            "--formal-session",
            "--output-dir",
            str(root / "formal-output"),
            "--session-id",
            "p6c-session",
            "--expected-source-commit",
            COMMIT,
            "--machine-id",
            "test-machine",
            "--gpu0-uuid",
            GPU_A,
            "--gpu1-uuid",
            GPU_B,
            "--provider-env-file",
            str(provider_env or (root / "provider.env")),
            "--lease-root",
            str(root / "leases"),
            "--max-allocated-active-gpu-hours",
            "12",
            "--max-simulated-control-steps",
            "1000000000",
            "--max-output-bytes",
            str(8 * 1024**3),
            "--nvidia-smi-executable",
            sys.executable,
        )


if __name__ == "__main__":
    unittest.main()
