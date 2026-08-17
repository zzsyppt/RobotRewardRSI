from __future__ import annotations

import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts import (  # noqa: E402
    materialize_stair_job,
    run_loop,
    run_stair_cold_start,
    run_stair_live,
)

from true_rl_agent.composition.defaults import (  # noqa: E402
    build_application_composition_registry,
)
from true_rl_agent.tasks.push_recovery.composition import (  # noqa: E402
    PUSH_RECOVERY_TASK_ID,
)
from true_rl_agent.tasks.stair_climbing.config import (  # noqa: E402
    STAIR_TASK_ID,
    StairClimbingTaskConfig,
)
from true_rl_agent.tasks.stair_climbing.final_confirmation import (  # noqa: E402
    STAIR_FINAL_CONFIRMATION_REQUIRED_ROUNDS,
    create_stair_final_confirmation_plan,
)
from true_rl_agent.tasks.stair_climbing.session_start import (  # noqa: E402
    STAIR_REWARD_COLD_START_MODE,
)


class StairProductionEntrypointTests(unittest.TestCase):
    def test_cold_start_wrapper_owns_mode_and_allows_new_genesis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            run_stair_cold_start,
            "run_generic_loop",
            return_value=0,
        ) as generic:
            exit_code = run_stair_cold_start.main(
                ["--rounds", "15", "--session-id", "stair-cold-session"]
            )

        self.assertEqual(exit_code, 0)
        argv = generic.call_args.args[0]
        self.assertEqual(
            argv[:6],
            [
                "--task-id",
                STAIR_TASK_ID,
                "--mode",
                "live",
                "--start-mode",
                STAIR_REWARD_COLD_START_MODE,
            ],
        )
        self.assertIs(
            generic.call_args.kwargs["require_existing_genesis"],
            False,
        )
        parsed = run_loop._parse_args(
            argv,
            registry=generic.call_args.kwargs["registry"],
        )
        self.assertEqual(parsed.start_mode, STAIR_REWARD_COLD_START_MODE)
        self.assertEqual(parsed.rounds, 15)
        composition = run_loop._resolve_composition(parsed, Path(tmp))
        self.assertEqual(composition.genesis_anchor_kind, "bootstrap")
        self.assertNotIn(
            "final_confirmation_precommit", composition.task_defaults
        )

    def test_cold_start_wrapper_rejects_owned_options(self) -> None:
        for argv in (
            ["--mode", "dry-run-audit"],
            [f"--task-id={PUSH_RECOVERY_TASK_ID}"],
            ["--start-mode", "reference"],
        ):
            with self.subTest(argv=argv), self.assertRaisesRegex(
                SystemExit,
                "owned by the Stair cold-start entrypoint",
            ):
                run_stair_cold_start.main(argv)

    def test_live_wrapper_uses_production_registry_and_fixed_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "fresh-formal"
            session_id = "formal-stair-session"
            config = StairClimbingTaskConfig.defaults(
                PROJECT_ROOT / "configs" / "tasks" / "stair_protocol_v2.json"
            )
            plan, plan_path = create_stair_final_confirmation_plan(
                output_dir=output_dir,
                session_id=session_id,
                config=config,
            )
            with patch.object(
                run_stair_live,
                "run_generic_loop",
                return_value=0,
            ) as generic:
                exit_code = run_stair_live.main(
                    [
                        "--rounds",
                        str(STAIR_FINAL_CONFIRMATION_REQUIRED_ROUNDS),
                        "--session-id",
                        session_id,
                        "--output-dir",
                        str(output_dir),
                        "--final-confirmation-plan",
                        str(plan_path),
                        "--execution-device",
                        "cuda:0",
                    ]
                )

            self.assertEqual(exit_code, 0)
            argv = generic.call_args.args[0]
            registry = generic.call_args.kwargs["registry"]
            require_existing_genesis = generic.call_args.kwargs[
                "require_existing_genesis"
            ]
            summary_validator = generic.call_args.kwargs[
                "live_summary_validator"
            ]
            self.assertEqual(argv.count("--task-id"), 1)
            self.assertEqual(argv.count("--mode"), 1)
            self.assertEqual(
                argv[:4], ["--task-id", STAIR_TASK_ID, "--mode", "live"]
            )
            self.assertEqual(
                registry.task_ids(),
                (PUSH_RECOVERY_TASK_ID, STAIR_TASK_ID),
            )
            self.assertIs(require_existing_genesis, True)
            self.assertIs(
                summary_validator,
                run_stair_live._require_closed_stair_live_summary,
            )

            parsed = run_loop._parse_args(argv, registry=registry)
            self.assertEqual(parsed.task_id, STAIR_TASK_ID)
            self.assertEqual(parsed.mode, "live")
            self.assertEqual(parsed.rounds, STAIR_FINAL_CONFIRMATION_REQUIRED_ROUNDS)
            self.assertEqual(parsed.session_id, session_id)
            self.assertEqual(parsed.output_dir, output_dir)
            self.assertEqual(parsed.final_confirmation_plan, plan_path)
            self.assertEqual(parsed.execution_device, "cuda:0")
            self.assertIsNone(parsed.session_genesis_dir)
            self.assertIsNone(parsed.ledger_epoch_id)
            self.assertEqual(parsed.train_max_iterations, 2000)
            self.assertEqual(parsed.budget_steps, 4096 * 24 * 2000)
            self.assertEqual(parsed.provider_model, "deepseek-v4-flash")
            self.assertEqual(parsed.provider_reasoning_mode, "adaptive")
            self.assertEqual(parsed.provider_reasoning_effort, "max")

            composition = run_loop._resolve_composition(parsed, output_dir)
            self.assertEqual(
                composition.task_defaults["final_confirmation_precommit"],
                plan.genesis_commitment(),
            )

    def test_wrapper_rejects_caller_owned_mode_or_task(self) -> None:
        for argv in (
            ["--mode", "dry-run-audit"],
            [f"--task-id={PUSH_RECOVERY_TASK_ID}"],
            ["--start-mode", STAIR_REWARD_COLD_START_MODE],
            ["--session-genesis-dir", "external-genesis"],
            ["--session-genesis-dir=external-genesis"],
            ["--ledger-epoch-id", "non-primary-epoch"],
            ["--ledger-epoch-id=non-primary-epoch"],
        ):
            with self.subTest(argv=argv), self.assertRaisesRegex(
                SystemExit,
                "owned by the Stair live entrypoint",
            ):
                run_stair_live.main(argv)

    def test_wrapper_closes_rejected_rounds_and_all_successful_evaluations(self) -> None:
        complete = {
            "rounds_requested": 3,
            "rounds_completed": 3,
            "trained_candidate_count": 2,
            "full_eval_completed_count": 2,
            "sealed_validation_completed_count": 2,
        }
        run_stair_live._require_closed_stair_live_summary(complete)

        for field, invalid in (
            ("rounds_completed", 2),
            ("trained_candidate_count", 4),
            ("full_eval_completed_count", 1),
            ("sealed_validation_completed_count", 1),
        ):
            with self.subTest(field=field):
                incomplete = {**complete, field: invalid}
                with self.assertRaisesRegex(
                    ValueError,
                    "Stair live",
                ):
                    run_stair_live._require_closed_stair_live_summary(
                        incomplete
                    )

    def test_materializer_uses_production_registry_and_candidate_kind(
        self,
    ) -> None:
        with patch.object(
            materialize_stair_job,
            "materialize_generic_job",
            return_value=0,
        ) as generic:
            exit_code = materialize_stair_job.main(
                ["--candidate-path", "candidate"]
            )

        self.assertEqual(exit_code, 0)
        argv = generic.call_args.args[0]
        registry = generic.call_args.kwargs["registry"]
        self.assertEqual(
            argv[:4],
            ["--task-id", STAIR_TASK_ID, "--job-kind", "candidate"],
        )
        self.assertEqual(
            registry.task_ids(),
            (PUSH_RECOVERY_TASK_ID, STAIR_TASK_ID),
        )

    def test_materializer_wrapper_rejects_owned_options(self) -> None:
        for argv in (
            ["--job-kind", "baseline-eval"],
            [f"--task-id={PUSH_RECOVERY_TASK_ID}"],
        ):
            with self.subTest(argv=argv), self.assertRaisesRegex(
                SystemExit,
                "owned by the Stair materializer entrypoint",
            ):
                materialize_stair_job.main(argv)

    def test_default_production_registry_includes_stair_but_keeps_push_default(
        self,
    ) -> None:
        registry = build_application_composition_registry()
        self.assertEqual(registry.default_task_id, PUSH_RECOVERY_TASK_ID)
        self.assertEqual(
            registry.task_ids(),
            (PUSH_RECOVERY_TASK_ID, STAIR_TASK_ID),
        )
        parsed = run_loop._parse_args([], registry=registry)
        self.assertEqual(parsed.task_id, PUSH_RECOVERY_TASK_ID)
        self.assertFalse(hasattr(parsed, "start_mode"))
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            run_loop._parse_args(
                ["--start-mode", STAIR_REWARD_COLD_START_MODE],
                registry=registry,
            )

if __name__ == "__main__":
    unittest.main()
