from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from scripts.run_p6b_attempt_chain_fault_probe import (
    PROJECT_ROOT,
    _TIMEOUT_S,
    _run_probe,
    _sha256_file,
    _source_identity,
    main,
)
from true_rl_agent.json_contract import canonical_json_bytes
from true_rl_agent.runner.subprocess_runner import SubprocessRunner

_SCRIPT = "scripts/run_p6b_attempt_chain_fault_probe.py"


def _source_stub() -> dict[str, object]:
    return {
        "source_commit": "a" * 40,
        "tracked_clean": True,
        "source_file_sha256": {_SCRIPT: "sha256:" + "b" * 64},
    }


class P6BAttemptChainFaultProbeTests(unittest.TestCase):
    def test_source_identity_records_head_clean_state_and_script_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            git_root = Path(tmp) / "repo"
            project_root = git_root / "project"
            script = project_root / _SCRIPT
            script.parent.mkdir(parents=True)
            script.write_bytes((PROJECT_ROOT / _SCRIPT).read_bytes())

            def git(*args: str) -> None:
                subprocess.run(
                    ["git", "-C", str(git_root), *args],
                    capture_output=True,
                    text=True,
                    timeout=15,
                    check=True,
                )

            git("init", "--quiet")
            for key, value in (
                ("core.autocrlf", "false"),
                ("user.name", "probe-test"),
                ("user.email", "probe-test@example.invalid"),
            ):
                git("config", key, value)
            git("add", f"project/{_SCRIPT}")
            git("commit", "--quiet", "-m", "fixture")
            (project_root / "untracked-output").write_text("allowed", encoding="utf-8")

            with (
                mock.patch(
                    "scripts.run_p6b_attempt_chain_fault_probe.PROJECT_ROOT",
                    project_root,
                ),
                mock.patch(
                    "scripts.run_p6b_attempt_chain_fault_probe._SOURCE_FILES",
                    (_SCRIPT,),
                ),
            ):
                source = _source_identity()
                self.assertTrue(source["tracked_clean"])
                self.assertRegex(source["source_commit"], r"^[0-9a-f]{40}$")
                self.assertEqual(
                    source["source_file_sha256"][_SCRIPT], _sha256_file(script)
                )
                script.write_text("tracked drift\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "tracked-clean"):
                    _source_identity()

    def test_existing_output_is_rejected_before_probe_work(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = (Path(tmp) / "existing").resolve()
            output.mkdir()
            with mock.patch.object(sys, "platform", "linux"):
                with self.assertRaisesRegex(FileExistsError, "already exists"):
                    main(
                        [
                            "--output",
                            str(output),
                            "--python-executable",
                            str(Path(sys.executable).resolve()),
                        ]
                    )

    def test_probe_passes_attempt_timeout_to_owned_process_runner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = (Path(tmp) / "evidence").resolve()
            output.mkdir()
            observed_timeouts: list[int | None] = []

            def run_job(
                _runner: SubprocessRunner,
                _argv: list[str],
                _metadata: object,
                *,
                job_id: str,
                process_log_dir: Path,
                attempt_timeout_s: int | None,
                execution_context: object,
            ) -> SimpleNamespace:
                self.assertEqual(job_id, "p6b-attempt-chain-probe")
                self.assertIsNone(execution_context)
                attempt_index = len(observed_timeouts)
                observed_timeouts.append(attempt_timeout_s)
                process_log_dir.mkdir(parents=True, exist_ok=True)
                process_id = 40_000 + attempt_index
                identity = {
                    "attempt_index": attempt_index,
                    "pid": process_id,
                    "pgid": process_id,
                    "grandchild_pid": process_id + 100,
                    "grandchild_pgid": process_id,
                }
                state_dir = output / "state"
                (state_dir / f"attempt_{attempt_index}.identity.json").write_bytes(
                    canonical_json_bytes(identity, trailing_newline=True)
                )
                if attempt_index == 1:
                    (state_dir / "summary.json").write_bytes(
                        canonical_json_bytes(
                            {"attempt_index": 1, "status": "completed"},
                            trailing_newline=True,
                        )
                    )
                return SimpleNamespace(
                    returncode=0,
                    pid=process_id,
                    stdout_log=process_log_dir / "stdout.log",
                    stderr_log=process_log_dir / "stderr.log",
                )

            with (
                mock.patch.object(
                    SubprocessRunner,
                    "_run_job",
                    autospec=True,
                    side_effect=run_job,
                ),
                mock.patch(
                    "scripts.run_p6b_attempt_chain_fault_probe."
                    "_linux_live_process_group_members",
                    return_value=(),
                ),
                mock.patch(
                    "scripts.run_p6b_attempt_chain_fault_probe._host_identity",
                    return_value={"boot_id": "cross-platform-test"},
                ),
            ):
                report = _run_probe(
                    output,
                    Path(sys.executable).resolve(),
                    ["probe-test"],
                    _source_stub(),
                )

            self.assertEqual(observed_timeouts, [_TIMEOUT_S, _TIMEOUT_S])
            self.assertEqual(report["status"], "pass")
            self.assertEqual(report["attempt_chain"]["attempt_result_count"], 2)

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux production probe")
    def test_real_retry_chain_and_owned_process_groups_are_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = (Path(tmp) / "evidence").resolve()
            raw_args = [
                "--output",
                str(output),
                "--python-executable",
                str(Path(sys.executable).resolve()),
            ]
            stdout = StringIO()
            with mock.patch(
                "scripts.run_p6b_attempt_chain_fault_probe._source_identity",
                return_value=_source_stub(),
            ), redirect_stdout(stdout):
                self.assertEqual(main(raw_args), 0)

            launch = json.loads(stdout.getvalue())
            report = json.loads(
                Path(launch["report_path"]).read_text(encoding="utf-8")
            )
            chain = report["attempt_chain"]
            self.assertEqual(report["status"], "pass")
            self.assertEqual(
                report["command_argv"],
                [
                    str(Path(sys.executable).resolve()),
                    str((PROJECT_ROOT / _SCRIPT).resolve()),
                    *raw_args,
                ],
            )
            self.assertTrue(report["host"]["boot_id"])
            self.assertEqual(chain["attempt_result_count"], 2)
            self.assertEqual(len(chain["result_files"]), 2)
            self.assertEqual(
                chain["first_result_hash"], chain["retry_previous_result_hash"]
            )
            self.assertTrue(chain["terminal_chain_head_hash"])
            self.assertEqual(chain["republish_created"], [False, False])
            self.assertTrue(chain["republish_bytes_unchanged"])
            observations = report["process_observations"]
            self.assertEqual([item["attempt_index"] for item in observations], [0, 1])
            self.assertTrue(
                all(
                    item["pid"] == item["pgid"]
                    and not item["live_pids_after_return"]
                    for item in observations
                )
            )
            self.assertTrue(all(report["checks"].values()))
            self.assertTrue(report["file_sha256"])


if __name__ == "__main__":
    unittest.main()
