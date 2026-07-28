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
from true_rl_agent.composition import (
    TaskCompositionContext,
    resolve_task_composition,
)
from true_rl_agent.tasks.push_recovery.composition import (
    PushRecoveryCompositionProvider,
)
from true_rl_agent.tasks.push_recovery.loop_audit import run_push_recovery_loop_audit


class DryRunAuditLoopTests(unittest.TestCase):
    def test_fifteen_round_audit_saves_public_prompts_and_blocks_tuning_accepts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            summary = run_push_recovery_loop_audit(
                composition=_resolve_push_composition(output_dir),
                output_dir=output_dir,
                rounds=15,
                session_id="audit-test",
            )
            ledger = load_ledger(output_dir / "ledger.jsonl")
            prompt_payload_text = "\n".join(
                Path(path).read_text(encoding="utf-8") for path in summary["prompt_payload_paths"]
            )

        self.assertTrue(all(summary["checks"].values()), summary)
        self.assertEqual(
            summary["schema_version"],
            "true_rl_agent.push_recovery.loop_audit.v1",
        )
        self.assertEqual(summary["rounds_completed"], 15)
        self.assertEqual(len(summary["prompt_payload_paths"]), 15)
        self.assertGreater(summary["decision_outcome_counts"].get("policy_improvement_accept", 0), 0)
        eval_audit = summary["eval_run_audit"]
        self.assertTrue(all(eval_audit["checks"].values()), eval_audit)
        self.assertEqual(summary["authored_candidate_count"], 15)
        self.assertEqual(summary["trainable_candidate_count"], 15)
        self.assertEqual(summary["trained_candidate_count"], 15)
        self.assertEqual(summary["full_eval_completed_count"], 15)
        self.assertEqual(summary["sealed_validation_completed_count"], 15)
        self.assertGreater(summary["accepted_count"], 0)
        self.assertEqual(summary["eval_incomplete_count"], 0)
        self.assertEqual(eval_audit["counts"]["trainable_candidate_count"], 15)
        self.assertEqual(eval_audit["counts"]["full_eval_completed_count"], 15)
        self.assertEqual(eval_audit["counts"]["sealed_validation_completed_count"], 15)
        self.assertNotIn("push_recovery_margin", prompt_payload_text)
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
            metrics = record["public_result"]["metrics"]
            self.assertIn("public_dev", metrics)
            self.assertIn("public_accept", metrics)
            self.assertIn("sealed_validation_gate", metrics)
        accepted_records = [record for record in ledger if record["decision"]["accepted"] is True]
        self.assertTrue(accepted_records)
        for record in accepted_records:
            for gate_name in (
                "budget_accounting",
                "motion_quality_gate",
                "behavior_quality_gate",
                "sealed_validation_gate",
            ):
                self.assertIn(gate_name, record["decision"]["consumed_gates"])

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
            candidate_path = _write_materialize_candidate(
                tmp_path,
                "candidate-code",
                component="reward_callable",
            )
            with redirect_stdout(io.StringIO()) as materialize_stdout:
                materialize_status = materialize.main(
                    [
                        "--round-id",
                        "cli-materialize",
                        "--candidate-id",
                        "candidate-code",
                        "--candidate-path",
                        str(candidate_path),
                        "--output-dir",
                        str(tmp_path / "materialize"),
                    ]
                )

        self.assertEqual(loop_status, 0)
        self.assertEqual(round_status, 0)
        self.assertEqual(materialize_status, 0)
        self.assertTrue(json.loads(loop_stdout.getvalue())["round_count_matches_request"])
        self.assertTrue(json.loads(round_stdout.getvalue())["round_count_matches_request"])
        self.assertEqual(
            json.loads(materialize_stdout.getvalue())["train"]["subject"]["id"],
            "candidate-code",
        )

    def test_audit_refuses_existing_ledger_and_materialize_component_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            audit_output = tmp_path / "audit"
            composition = _resolve_push_composition(audit_output)
            run_push_recovery_loop_audit(
                composition=composition,
                output_dir=audit_output,
                rounds=1,
                session_id="first",
            )

            with self.assertRaises(FileExistsError):
                run_push_recovery_loop_audit(
                    composition=composition,
                    output_dir=audit_output,
                    rounds=1,
                    session_id="second",
                )

            materialize = _load_script("materialize_job.py")
            candidate_path = _write_materialize_candidate(
                tmp_path,
                "candidate-diagnostic",
                component="diagnostic",
            )
            with redirect_stdout(io.StringIO()) as materialize_stdout:
                status = materialize.main(
                    [
                        "--round-id",
                        "explicit-component",
                        "--candidate-id",
                        "candidate-diagnostic",
                        "--candidate-path",
                        str(candidate_path),
                        "--component",
                        "diagnostic",
                        "--output-dir",
                        str(tmp_path / "materialize"),
                    ]
                )
            payload = json.loads(materialize_stdout.getvalue())

        self.assertEqual(status, 0)
        self.assertEqual(
            payload["train"]["subject"]["id"],
            "candidate-diagnostic",
        )
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
            composition = run_loop._resolve_composition(args, output_dir)
            backend_config = composition.backend_adapter.config
            runner = composition.runner

        self.assertEqual(provider_config.model, "deepseek-v4-pro")
        self.assertEqual(provider_config.api_key_env, "DEEPSEEK_API_KEY")
        self.assertEqual(provider_config.env_file, env_file)
        self.assertEqual(provider_config.output_dir, output_dir / "candidates")
        self.assertEqual(provider_config.max_tokens, 8192)
        self.assertEqual(args.provider_max_attempts, run_loop.DEFAULT_MAX_PROVIDER_ATTEMPTS)
        self.assertEqual(args.provider_max_attempts, 5)
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
            composition = run_loop._resolve_composition(args, output_dir)
            backend_config = composition.backend_adapter.config
            runner = composition.runner

        self.assertEqual(backend_config.default_budget_steps, 49_152_000)
        self.assertEqual(backend_config.train_num_envs, 4096)
        self.assertEqual(backend_config.train_max_iterations, 500)
        self.assertEqual(backend_config.eval_num_envs, 64)
        self.assertEqual(backend_config.public_seed_count, 3)
        self.assertEqual(backend_config.public_accept_seed_count, 3)
        self.assertEqual(runner.timeout_s, 7200)

    def test_run_loop_modes_reject_inconsistent_training_budget(self) -> None:
        run_loop = _load_script("run_loop.py")
        invalid_budget_args = [
            "--budget-steps",
            "65",
            "--train-num-envs",
            "8",
            "--train-max-iterations",
            "1",
            "--train-steps-per-iteration",
            "8",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            for mode in ("live", "dry-run-audit"):
                with self.subTest(mode=mode):
                    with self.assertRaisesRegex(
                        ValueError,
                        "planned env steps mismatch",
                    ):
                        run_loop.main(
                            [
                                "--mode",
                                mode,
                                "--output-dir",
                                str(Path(tmp) / mode),
                                *invalid_budget_args,
                            ]
                        )

    def test_run_round_rejects_inconsistent_training_budget(self) -> None:
        run_round = _load_script("run_round.py")
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(
                ValueError,
                "planned env steps mismatch",
            ):
                run_round.main(
                    [
                        "--output-dir",
                        tmp,
                        "--budget-steps",
                        "65",
                        "--train-num-envs",
                        "8",
                        "--train-max-iterations",
                        "1",
                        "--train-steps-per-iteration",
                        "8",
                    ]
                )

    def test_dry_run_cli_entrypoints_materialize_the_resolved_task_snapshot(
        self,
    ) -> None:
        explicit_runtime_args = [
            "--budget-steps",
            "128",
            "--train-num-envs",
            "8",
            "--train-max-iterations",
            "2",
            "--train-steps-per-iteration",
            "8",
            "--public-seed-count",
            "2",
            "--public-accept-seed-count",
            "2",
            "--sealed-validation-seed-count",
            "2",
            "--eval-num-envs",
            "2",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            run_loop = _load_script("run_loop.py")
            loop_output = tmp_path / "loop"
            with redirect_stdout(io.StringIO()):
                loop_status = run_loop.main(
                    [
                        "--rounds",
                        "1",
                        "--session-id",
                        "resolved-loop",
                        "--output-dir",
                        str(loop_output),
                        *explicit_runtime_args,
                    ]
                )

            run_round = _load_script("run_round.py")
            round_output = tmp_path / "round"
            with redirect_stdout(io.StringIO()):
                round_status = run_round.main(
                    [
                        "--session-id",
                        "resolved-round",
                        "--output-dir",
                        str(round_output),
                        *explicit_runtime_args,
                    ]
                )

            records = (
                load_ledger(loop_output / "ledger.jsonl"),
                load_ledger(round_output / "ledger.jsonl"),
            )

        self.assertEqual(loop_status, 0)
        self.assertEqual(round_status, 0)
        for ledger in records:
            self.assertEqual(len(ledger), 1)
            jobs = ledger[0]["jobs"]
            self.assertEqual(jobs["train"]["budget_steps"], 128)
            self.assertEqual(jobs["eval"]["public_seed_count"], 2)
            self.assertEqual(jobs["eval"]["public_accept_seed_count"], 2)
            self.assertEqual(
                jobs["eval"]["sealed_validation_seed_count"],
                2,
            )
            self.assertEqual(jobs["eval"]["eval_num_envs"], 2)
            accounting = ledger[0]["train_result"]["summary"]["training"][
                "budget_accounting"
            ]
            self.assertEqual(
                accounting,
                {
                    "status": "pass",
                    "budget_steps": 128,
                    "num_envs": 8,
                    "num_steps_per_env": 8,
                    "max_iterations": 2,
                    "planned_env_steps": 128,
                    "config_derived_total_timesteps": 128,
                },
            )


def _write_materialize_candidate(
    root: Path,
    candidate_id: str,
    *,
    component: str,
) -> Path:
    candidate = root / "materialize-candidates" / candidate_id
    training_code = candidate / "training_code"
    training_code.mkdir(parents=True)
    (candidate / "candidate_manifest.yaml").write_text(
        "schema_version: true_rl_agent.candidate.v1\n"
        f"candidate_id: {candidate_id}\n"
        "entrypoint: training_code.components:build_training_components\n",
        encoding="utf-8",
    )
    if component == "diagnostic":
        source = (
            "from true_rl_agent.candidates.components import DiagnosticTerm, TrainingComponents\n\n"
            "def diagnostic(inputs):\n"
            "    return {'base_height': inputs.base_height}\n\n"
            "def build_training_components():\n"
            "    return TrainingComponents(\n"
            "        name='candidate-diagnostic',\n"
            "        diagnostics=[DiagnosticTerm(name='height', fn=diagnostic)],\n"
            "    )\n"
        )
    else:
        source = (
            "from true_rl_agent.candidates.components import RewardTerm, TrainingComponents\n\n"
            "def reward(inputs):\n"
            "    return inputs.base_height\n\n"
            "def build_training_components():\n"
            "    return TrainingComponents(\n"
            "        name='candidate-code',\n"
            "        reward_terms=[RewardTerm(name='height', weight=1.0, fn=reward)],\n"
            "    )\n"
        )
    (training_code / "components.py").write_text(source, encoding="utf-8")
    return candidate


def _load_script(filename: str):
    script_path = PROJECT_ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(f"true_rl_agent_script_{filename}", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_push_composition(output_dir: Path):
    provider = PushRecoveryCompositionProvider()
    return resolve_task_composition(
        provider=provider,
        values=dict(provider.cli_defaults()),
        context=TaskCompositionContext(
            project_root=PROJECT_ROOT,
            output_dir=output_dir,
        ),
    )


if __name__ == "__main__":
    unittest.main()
