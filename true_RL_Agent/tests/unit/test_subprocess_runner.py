from __future__ import annotations

import json
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from true_rl_agent.evaluation.public_result import PublicEvalResult
from true_rl_agent.policy import PolicyInitialization
from true_rl_agent.runner.base import CandidateRuntimeError
from true_rl_agent.runner.specs import CheckpointRef, EvalJobSpec, TrainJobSpec
from true_rl_agent.runner.subprocess_runner import RunnerExecutionError, SubprocessRunner


class SubprocessRunnerTests(unittest.TestCase):
    def test_subprocess_runner_executes_train_and_eval_from_structured_argv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_summary = root / "train_summary.json"
            eval_summary = root / "eval_summary.json"
            train_code = textwrap.dedent(
                f"""
                import json
                from pathlib import Path
                Path({str(train_summary)!r}).write_text(json.dumps({{
                    "checkpoint": {{
                        "uri": "artifact://train/raw.pt",
                        "checksum": "sha256:train",
                        "role": "train_output"
                    }},
                    "runtime_activation_summary": {{
                        "reward_term_calls": 3,
                        "nonzero_reward_contribution": True
                    }}
                }}))
                """
            )
            eval_code = textwrap.dedent(
                f"""
                import json
                from pathlib import Path
                Path({str(eval_summary)!r}).write_text(json.dumps({{
                    "status": "pass",
                    "public_score": 0.42,
                    "metrics": {{"success_rate": 0.42}},
                    "candidate_code_loaded_in_eval": False,
                    "candidate_package_loaded": False
                }}))
                """
            )
            runner = SubprocessRunner(allowed_executables=(sys.executable,), timeout_s=10)
            train_spec = TrainJobSpec(
                job_id="train-001",
                task_id="unitree_push_recovery",
                candidate_id="candidate-code",
                policy_initialization=PolicyInitialization.random(),
                budget_steps=128,
                metadata={
                    "cli_argv": [sys.executable, "-c", train_code],
                    "summary_path": str(train_summary),
                    "sealed_checkpoint_uri": "artifact://train/sealed/best.pt",
                },
            )
            eval_spec = EvalJobSpec(
                job_id="eval-001",
                task_id="unitree_push_recovery",
                candidate_id="candidate-code",
                checkpoint=CheckpointRef("artifact://train/sealed/best.pt", "sha256:train", "eval", sealed=True),
                public_seed_count=3,
                trusted_seed_count=7,
                metadata={"cli_argv": [sys.executable, "-c", eval_code], "summary_path": str(eval_summary)},
            )

            train_result = runner.run_train(train_spec)
            public_result = runner.run_eval(eval_spec)

        self.assertEqual(train_result.checkpoint.role, "eval")
        self.assertTrue(train_result.checkpoint.sealed)
        self.assertEqual(train_result.checkpoint.uri, "artifact://train/sealed/best.pt")
        self.assertEqual(train_result.summary["runtime_activation_summary"]["reward_term_calls"], 3)
        self.assertIsInstance(public_result, PublicEvalResult)
        self.assertEqual(public_result.public_score, 0.42)
        self.assertEqual(public_result.metrics["success_rate"], 0.42)
        self.assertIs(public_result.metrics["candidate_code_loaded_in_eval"], False)
        self.assertIs(public_result.metrics["candidate_package_loaded"], False)

    def test_subprocess_runner_rejects_shell_strings_and_unapproved_executables(self) -> None:
        runner = SubprocessRunner(allowed_executables=(sys.executable,), timeout_s=1)
        string_command_spec = TrainJobSpec(
            job_id="train-string",
            task_id="unitree_push_recovery",
            candidate_id="candidate",
            policy_initialization=PolicyInitialization.random(),
            budget_steps=128,
            metadata={"cli_argv": "python -c pass", "summary_path": "summary.json"},
        )
        unapproved_spec = TrainJobSpec(
            job_id="train-bad-exec",
            task_id="unitree_push_recovery",
            candidate_id="candidate",
            policy_initialization=PolicyInitialization.random(),
            budget_steps=128,
            metadata={"cli_argv": ["/bin/sh", "-c", "echo bad"], "summary_path": "summary.json"},
        )

        with self.assertRaisesRegex(RunnerExecutionError, "cli_argv must be list"):
            runner.run_train(string_command_spec)
        with self.assertRaisesRegex(RunnerExecutionError, "executable is not allowlisted"):
            runner.run_train(unapproved_spec)

    def test_subprocess_runner_reports_missing_summary_as_execution_error(self) -> None:
        runner = SubprocessRunner(allowed_executables=(sys.executable,), timeout_s=10)
        with tempfile.TemporaryDirectory() as tmp:
            missing_summary = Path(tmp) / "missing.json"
            train_spec = TrainJobSpec(
                job_id="train-missing-summary",
                task_id="unitree_push_recovery",
                candidate_id="candidate",
                policy_initialization=PolicyInitialization.random(),
                budget_steps=128,
                metadata={"cli_argv": [sys.executable, "-c", "pass"], "summary_path": str(missing_summary)},
            )

            with self.assertRaisesRegex(RunnerExecutionError, "summary file was not created"):
                runner.run_train(train_spec)

    def test_subprocess_runner_reports_failed_train_summary_before_checkpoint_parse(self) -> None:
        runner = SubprocessRunner(allowed_executables=(sys.executable,), timeout_s=10)
        with tempfile.TemporaryDirectory() as tmp:
            summary = Path(tmp) / "summary.json"
            failure_code = textwrap.dedent(
                f"""
                import json
                from pathlib import Path
                Path({str(summary)!r}).write_text(json.dumps({{
                    "status": "fail",
                    "errors": [{{"type": "RuntimeError", "message": "Boolean value of Tensor with more than one value is ambiguous"}}]
                }}))
                """
            )
            train_spec = TrainJobSpec(
                job_id="train-runtime-fail",
                task_id="unitree_push_recovery",
                candidate_id="candidate",
                policy_initialization=PolicyInitialization.random(),
                budget_steps=128,
                metadata={"cli_argv": [sys.executable, "-c", failure_code], "summary_path": str(summary)},
            )

            with self.assertRaises(CandidateRuntimeError) as context:
                runner.run_train(train_spec)

        message = str(context.exception)
        self.assertIn("train summary status fail", message)
        self.assertIn("Boolean value of Tensor", message)
        self.assertNotIn("missing checkpoint", message)

    def test_subprocess_runner_redacts_secrets_from_failed_process_output(self) -> None:
        runner = SubprocessRunner(allowed_executables=(sys.executable,), timeout_s=10)
        with tempfile.TemporaryDirectory() as tmp:
            missing_summary = Path(tmp) / "summary.json"
            failure_code = textwrap.dedent(
                """
                import sys
                print("TOKEN=plainsecretvalue")
                print("Authorization: Bearer plainbearer", file=sys.stderr)
                raise SystemExit(2)
                """
            )
            train_spec = TrainJobSpec(
                job_id="train-secret-output",
                task_id="unitree_push_recovery",
                candidate_id="candidate",
                policy_initialization=PolicyInitialization.random(),
                budget_steps=128,
                metadata={"cli_argv": [sys.executable, "-c", failure_code], "summary_path": str(missing_summary)},
            )

            with self.assertRaises(RunnerExecutionError) as context:
                runner.run_train(train_spec)

        message = str(context.exception)
        self.assertNotIn("plainsecretvalue", message)
        self.assertNotIn("plainbearer", message)
        self.assertIn("TOKEN=REDACTED", message)
        self.assertIn("Authorization: Bearer REDACTED", message)


if __name__ == "__main__":
    unittest.main()
