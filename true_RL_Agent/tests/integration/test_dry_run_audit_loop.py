from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from true_rl_agent.agent_core.ledger import load_ledger
from true_rl_agent.audit import run_loop_audit


class DryRunAuditLoopTests(unittest.TestCase):
    def test_fifteen_round_audit_saves_public_prompts_and_blocks_tuning_accepts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary = run_loop_audit(output_dir=Path(tmp), rounds=15, session_id="audit-test")
            ledger = load_ledger(Path(tmp) / "ledger.jsonl")

        self.assertTrue(all(summary["checks"].values()), summary)
        self.assertEqual(summary["rounds_completed"], 15)
        self.assertEqual(len(summary["prompt_payload_paths"]), 15)
        self.assertGreater(summary["decision_outcome_counts"].get("policy_improvement_accept", 0), 0)
        self.assertEqual(summary["violations"]["parameter_only_accepted_rounds"], [])
        self.assertEqual(summary["violations"]["curriculum_llm_controlled_rounds"], [])
        self.assertEqual(summary["violations"]["train_llm_schedule_rounds"], [])
        self.assertEqual(len(ledger), 15)
        self.assertTrue(
            all(record["curriculum"]["llm_controlled"] is False for record in ledger)
        )
        self.assertTrue(
            all(
                record["decision"]["accepted"] is False
                for record in ledger
                if record["candidate_surface"]["classification"] == "parameter_only"
            )
        )
        for record, prompt_path in zip(ledger, summary["prompt_payload_paths"], strict=True):
            self.assertEqual(record["prompt_payload"]["path"], prompt_path)
            self.assertEqual(record["prompt_payload"]["visibility"], "public_prompt_payload")

    def test_cli_entrypoints_are_implemented_without_subprocess(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            run_loop = _load_script("run_loop.py")
            with redirect_stdout(io.StringIO()) as loop_stdout:
                loop_status = run_loop.main(
                    ["--rounds", "4", "--session-id", "cli-loop", "--output-dir", str(tmp_path / "loop")]
                )

            run_round = _load_script("run_round.py")
            with redirect_stdout(io.StringIO()) as round_stdout:
                round_status = run_round.main(
                    ["--session-id", "cli-round", "--output-dir", str(tmp_path / "round")]
                )

            materialize = _load_script("materialize_job.py")
            with redirect_stdout(io.StringIO()) as materialize_stdout:
                materialize_status = materialize.main(
                    ["--round-id", "cli-materialize", "--candidate-id", "candidate-code"]
                )

        self.assertEqual(loop_status, 0)
        self.assertEqual(round_status, 0)
        self.assertEqual(materialize_status, 0)
        self.assertTrue(json.loads(loop_stdout.getvalue())["round_count_matches_request"])
        self.assertTrue(json.loads(round_stdout.getvalue())["round_count_matches_request"])
        self.assertEqual(json.loads(materialize_stdout.getvalue())["train"]["candidate_id"], "candidate-code")

    def test_audit_refuses_existing_ledger_and_materialize_component_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            run_loop_audit(output_dir=tmp_path / "audit", rounds=1, session_id="first")

            with self.assertRaises(FileExistsError):
                run_loop_audit(output_dir=tmp_path / "audit", rounds=1, session_id="second")

            materialize = _load_script("materialize_job.py")
            with redirect_stdout(io.StringIO()) as materialize_stdout:
                status = materialize.main(
                    [
                        "--round-id",
                        "explicit-component",
                        "--candidate-id",
                        "candidate-diagnostic",
                        "--component",
                        "diagnostic",
                    ]
                )
            payload = json.loads(materialize_stdout.getvalue())

        self.assertEqual(status, 0)
        self.assertEqual(payload["train"]["candidate_id"], "candidate-diagnostic")
        self.assertEqual(payload["candidate_changed_components"], ["diagnostic"])
        self.assertNotIn("reward_callable", json.dumps(payload, sort_keys=True))

    def test_live_loop_cli_builds_provider_and_runner_config_without_starting_jobs(self) -> None:
        run_loop = _load_script("run_loop.py")
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "live"
            env_file = Path(tmp) / "deepseek.env"
            env_file.write_text("DEEPSEEK_API_KEY=test-secret\n", encoding="utf-8")
            args = run_loop._parse_args(
                [
                    "--mode",
                    "live",
                    "--rounds",
                    "15",
                    "--session-id",
                    "live-cli",
                    "--output-dir",
                    str(output_dir),
                    "--provider-base-url",
                    "https://api.deepseek.com",
                    "--provider-model",
                    "deepseek-v4-pro",
                    "--provider-api-key-env",
                    "DEEPSEEK_API_KEY",
                    "--provider-env-file",
                    str(env_file),
                    "--isaac-python",
                    "/data/yuyangliu/envs/isaaclab/bin/python",
                    "--budget-steps",
                    "192",
                    "--train-num-envs",
                    "8",
                    "--train-max-iterations",
                    "1",
                    "--train-steps-per-iteration",
                    "24",
                    "--eval-num-envs",
                    "8",
                ]
            )
            provider_config = run_loop._build_provider_config(args, output_dir)
            backend_config = run_loop._build_backend_config(args, output_dir)
            runner = run_loop._build_runner(args)

        self.assertEqual(provider_config.model, "deepseek-v4-pro")
        self.assertEqual(provider_config.api_key_env, "DEEPSEEK_API_KEY")
        self.assertEqual(provider_config.env_file, env_file)
        self.assertEqual(provider_config.output_dir, output_dir / "candidates")
        self.assertEqual(provider_config.max_tokens, 8192)
        self.assertEqual(args.provider_max_attempts, 2)
        self.assertEqual(backend_config.default_budget_steps, 192)
        self.assertEqual(backend_config.train_num_envs, 8)
        self.assertEqual(backend_config.train_max_iterations, 1)
        self.assertEqual(backend_config.eval_num_envs, 8)
        self.assertEqual(backend_config.artifact_root, str(output_dir / "runner"))
        self.assertEqual(runner.allowed_executables, ("/data/yuyangliu/envs/isaaclab/bin/python",))

    def test_live_loop_cli_defaults_to_phase8b_full_training_budget(self) -> None:
        run_loop = _load_script("run_loop.py")
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "live"
            args = run_loop._parse_args(["--mode", "live", "--output-dir", str(output_dir)])
            backend_config = run_loop._build_backend_config(args, output_dir)
            runner = run_loop._build_runner(args)

        self.assertEqual(backend_config.default_budget_steps, 49_152_000)
        self.assertEqual(backend_config.train_num_envs, 4096)
        self.assertEqual(backend_config.train_max_iterations, 500)
        self.assertEqual(backend_config.eval_num_envs, 64)
        self.assertEqual(backend_config.public_seed_count, 3)
        self.assertEqual(backend_config.public_accept_seed_count, 1)
        self.assertEqual(runner.timeout_s, 7200)

    def test_live_loop_cli_rejects_inconsistent_training_budget(self) -> None:
        run_loop = _load_script("run_loop.py")

        with self.assertRaises(SystemExit):
            run_loop._parse_args(
                [
                    "--mode",
                    "live",
                    "--budget-steps",
                    "64",
                    "--train-num-envs",
                    "8",
                    "--train-max-iterations",
                    "500",
                    "--train-steps-per-iteration",
                    "24",
                ]
            )

    def test_dry_run_audit_cli_does_not_validate_live_training_budget(self) -> None:
        run_loop = _load_script("run_loop.py")

        args = run_loop._parse_args(["--mode", "dry-run-audit", "--budget-steps", "64"])

        self.assertEqual(args.mode, "dry-run-audit")
        self.assertEqual(args.budget_steps, 64)


def _load_script(filename: str):
    script_path = PROJECT_ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(f"true_rl_agent_script_{filename}", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    unittest.main()
