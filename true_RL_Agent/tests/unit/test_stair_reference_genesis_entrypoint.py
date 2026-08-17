from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.create_stair_reference_genesis import main

from tests.fakes.backend_environment import (
    use_test_isaac_lab_execution_environment,
)
from tests.unit.test_reference_genesis import _reference_fixture
from true_rl_agent.agent_core.session_genesis import SESSION_GENESIS_SCHEMA_VERSION
from true_rl_agent.policy import PolicyInitialization
from true_rl_agent.tasks.stair_climbing.config import StairClimbingTaskConfig
from true_rl_agent.tasks.stair_climbing.final_confirmation import (
    create_stair_final_confirmation_plan,
)
from true_rl_agent.tasks.stair_climbing.reference_program import (
    resolve_official_stair_reference_program,
)


class StairReferenceGenesisEntrypointTests(unittest.TestCase):
    def test_preflight_binds_final_budget_without_genesis_or_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            stdout = io.StringIO()
            with (
                patch(
                    "true_rl_agent.tasks.stair_climbing.reference_program."
                    "resolve_official_stair_reference_program",
                    wraps=resolve_official_stair_reference_program,
                ) as resolver,
                use_test_isaac_lab_execution_environment(),
                redirect_stdout(stdout),
            ):
                status = main(
                    [
                        "--mode",
                        "preflight",
                        "--session-id",
                        "stair-reference-preflight",
                        "--output-dir",
                        str(output_dir),
                        "--execution-device",
                        "cuda:1",
                    ]
                )
            payload = json.loads(stdout.getvalue())
            config = StairClimbingTaskConfig.defaults(
                PROJECT_ROOT / "configs" / "tasks" / "stair_protocol_v2.json"
            )

            self.assertEqual(status, 0)
            self.assertEqual(payload["status"], "pass")
            self.assertEqual(payload["execution_device"], "cuda:1")
            self.assertEqual(payload["budget_steps"], config.budget_steps)
            self.assertEqual(
                payload["max_iterations"],
                config.train_max_iterations,
            )
            self.assertEqual(payload["train_subject"], payload["eval_subject"])
            self.assertEqual(payload["train_subject"]["kind"], "reference")
            self.assertFalse((output_dir / "session_genesis.json").exists())
            self.assertFalse((output_dir / "ledger.jsonl").exists())
            resolver.assert_called_once()

    def test_formal_entrypoint_does_not_expose_budget_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stderr = io.StringIO()
            with redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                main(
                    [
                        "--mode",
                        "preflight",
                        "--session-id",
                        "stair-reference-fixed-budget",
                        "--output-dir",
                        tmp,
                        "--train-max-iterations",
                        "1",
                    ]
                )

            self.assertEqual(raised.exception.code, 2)
            self.assertIn("unrecognized arguments", stderr.getvalue())

    def test_load_only_requires_existing_genesis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            with self.assertRaisesRegex(
                ValueError,
                "load-only mode requires an existing session Genesis",
            ):
                main(
                    [
                        "--mode",
                        "load-only",
                        "--session-id",
                        "stair-reference-missing",
                        "--output-dir",
                        str(output_dir),
                        "--execution-device",
                        "cuda:1",
                    ]
                )

            self.assertFalse((output_dir / "session_genesis.json").exists())
            self.assertFalse((output_dir / "ledger.jsonl").exists())

    def test_create_requires_the_pre_genesis_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(
            ValueError,
            "requires --final-confirmation-plan",
        ):
            main(
                [
                    "--mode",
                    "create",
                    "--session-id",
                    "stair-reference-create-no-plan",
                    "--output-dir",
                    tmp,
                ]
            )

    def test_current_v3_uncommitted_load_only_without_plan_is_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            fixture = _reference_fixture(output_dir)
            genesis_path = output_dir / "session_genesis.json"
            ledger_path = output_dir / "ledger.jsonl"
            genesis_before = genesis_path.read_bytes()
            genesis_mtime_before = genesis_path.stat().st_mtime_ns
            genesis_payload = json.loads(genesis_before)
            self.assertEqual(
                genesis_payload["schema_version"],
                SESSION_GENESIS_SCHEMA_VERSION,
            )
            self.assertFalse(ledger_path.exists())
            self.assertNotIn(
                "final_confirmation_precommit",
                json.dumps(fixture["genesis_identity_snapshot"], sort_keys=True),
            )
            composition = SimpleNamespace(
                task_plugin=fixture["task"],
                backend_adapter=fixture["backend"],
                runner=fixture["runner"],
                policy_initialization=PolicyInitialization.random(),
                reference_program_artifact=fixture["reference"],
                task_id=fixture["task"].task_spec.task_id,
                genesis_identity_snapshot=(
                    lambda: fixture["genesis_identity_snapshot"]
                ),
            )
            stdout = io.StringIO()

            with (
                patch(
                    "scripts.create_stair_reference_genesis.resolve_task_composition",
                    return_value=composition,
                ),
                redirect_stdout(stdout),
            ):
                status = main(
                    [
                        "--mode",
                        "load-only",
                        "--session-id",
                        "reference-session",
                        "--output-dir",
                        str(output_dir),
                    ]
                )

            payload = json.loads(stdout.getvalue())
            self.assertEqual(status, 0)
            self.assertEqual(
                payload["load_only_evidence"]["reference_source_status"],
                "pass",
            )
            self.assertEqual(payload["load_only_evidence"]["runner_call_count"], 0)
            self.assertEqual(payload["load_only_evidence"]["provider_call_count"], 0)
            self.assertEqual(payload["ledger_state"], "absent")
            self.assertEqual(payload["ledger_record_count"], 0)
            self.assertEqual(
                payload["load_only_evidence"]["replay_ledger_record_count"],
                0,
            )
            self.assertEqual(
                payload["load_only_evidence"]["genesis_checksum_before"],
                payload["load_only_evidence"]["genesis_checksum_after"],
            )
            self.assertEqual(
                payload["load_only_evidence"]["genesis_mtime_ns_before"],
                payload["load_only_evidence"]["genesis_mtime_ns_after"],
            )
            self.assertEqual(genesis_path.read_bytes(), genesis_before)
            self.assertEqual(genesis_path.stat().st_mtime_ns, genesis_mtime_before)
            self.assertFalse(ledger_path.exists())
            self.assertEqual(
                (fixture["runner"].train_calls, fixture["runner"].eval_calls),
                (1, 1),
            )

    def test_committed_load_only_without_plan_fails_identity_verification(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            config = StairClimbingTaskConfig.defaults(
                PROJECT_ROOT / "configs" / "tasks" / "stair_protocol_v2.json"
            )
            _, plan_path = create_stair_final_confirmation_plan(
                output_dir=output_dir,
                session_id="stair-reference-committed",
                config=config,
            )
            (output_dir / "session_genesis.json").write_bytes(b"committed fixture")

            def reject_legacy_identity(**kwargs):
                self.assertNotIn(
                    "final_confirmation_precommit",
                    kwargs["genesis_identity_snapshot"]["task_defaults"],
                )
                raise ValueError("session Genesis identity drift")

            with (
                patch(
                    "scripts.create_stair_reference_genesis.prepare_session_genesis",
                    side_effect=reject_legacy_identity,
                ) as prepare,
                self.assertRaisesRegex(ValueError, "session Genesis identity drift"),
            ):
                main(
                    [
                        "--mode",
                        "load-only",
                        "--session-id",
                        "stair-reference-committed",
                        "--output-dir",
                        str(output_dir),
                    ]
                )

            prepare.assert_called_once()
            self.assertTrue(plan_path.is_file())


if __name__ == "__main__":
    unittest.main()
