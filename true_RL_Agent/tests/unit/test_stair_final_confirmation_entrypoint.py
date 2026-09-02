from __future__ import annotations

import contextlib
import importlib.util
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterator
from unittest.mock import Mock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from true_rl_agent.agent_core.live_invocation import (  # noqa: E402
    LIVE_LOOP_SUMMARY_REQUIRED_CHECKS,
    LIVE_LOOP_SUMMARY_SCHEMA_VERSION,
    LiveInvocationJournal,
    write_live_summary,
)
from true_rl_agent.tasks.stair_climbing.config import (  # noqa: E402
    StairClimbingTaskConfig,
)
from true_rl_agent.tasks.stair_climbing.final_confirmation import (  # noqa: E402
    STAIR_FINAL_CONFIRMATION_REQUIRED_ROUNDS,
    build_stair_final_confirmation_plan,
    load_stair_final_confirmation_reservation,
    stair_final_confirmation_reservation_path,
)

SESSION_ID = "stair-final-entrypoint-test"
SESSION_GENESIS_HASH = "a" * 64
LEDGER_EPOCH_HASH = "b" * 64
SOURCE_MANIFEST_HASH = "c" * 64
BACKEND_MANIFEST_HASH = "d" * 64
GENESIS_CHECKPOINT_HASH = "sha256:" + "e" * 64
FINAL_CHECKPOINT_HASH = "sha256:" + "f" * 64
LEDGER_CHECKSUM = "sha256:" + "1" * 64


class _FakeAppLauncher:
    @staticmethod
    def add_app_launcher_args(parser: Any) -> None:
        _ = parser

    def __init__(self, args: Any) -> None:
        _ = args
        self.app = object()


class _FailingAppLauncher(_FakeAppLauncher):
    def __init__(self, args: Any) -> None:
        _ = args
        raise RuntimeError("AppLauncher construction failed")


class StairFinalConfirmationEntrypointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.script = _load_entrypoint()
        cls.config = StairClimbingTaskConfig.defaults(
            PROJECT_ROOT / "configs" / "tasks" / "stair_protocol_v2.json"
        )

    def test_completed_receipt_is_rejected_when_ledger_recomputation_differs(
        self,
    ) -> None:
        receipt_audit = _eval_run_audit()
        receipt_best = {
            "anchor_kind": "candidate",
            "accepted_round_id": "round-14",
            "public_accept": 0.42,
        }
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            journal = _write_completed_invocation(
                output_dir=output_dir,
                eval_run_audit=receipt_audit,
                final_best_eval_stats=receipt_best,
                final_best_score=0.42,
            )
            immutable_snapshot = journal.invocation_dir / "live_summary.json"
            snapshot_bytes = immutable_snapshot.read_bytes()
            records = [
                {"round_id": f"round-{index:02d}"}
                for index in range(1, STAIR_FINAL_CONFIRMATION_REQUIRED_ROUNDS + 1)
            ]

            ledger_recomputed_audit = _eval_run_audit()
            ledger_recomputed_audit["failure_buckets"] = {"runner_failure": 1}
            with self.assertRaisesRegex(
                ValueError,
                "live summary eval audit differs from the ledger",
            ):
                self.script._assert_completed_live_run(
                    output_dir=output_dir,
                    session_id=SESSION_ID,
                    session_genesis_hash=SESSION_GENESIS_HASH,
                    expected_ledger_epoch_hash=LEDGER_EPOCH_HASH,
                    records=records,
                    expected_eval_run_audit=ledger_recomputed_audit,
                    expected_final_best_eval_stats=receipt_best,
                    expected_final_best_score=0.42,
                )

            ledger_recomputed_best = {
                **receipt_best,
                "accepted_round_id": "round-15",
            }
            with self.assertRaisesRegex(
                ValueError,
                "live summary current-best differs from the ledger",
            ):
                self.script._assert_completed_live_run(
                    output_dir=output_dir,
                    session_id=SESSION_ID,
                    session_genesis_hash=SESSION_GENESIS_HASH,
                    expected_ledger_epoch_hash=LEDGER_EPOCH_HASH,
                    records=records,
                    expected_eval_run_audit=receipt_audit,
                    expected_final_best_eval_stats=ledger_recomputed_best,
                    expected_final_best_score=0.42,
                )

            with self.assertRaisesRegex(
                ValueError,
                "live summary best score differs from the ledger",
            ):
                self.script._assert_completed_live_run(
                    output_dir=output_dir,
                    session_id=SESSION_ID,
                    session_genesis_hash=SESSION_GENESIS_HASH,
                    expected_ledger_epoch_hash=LEDGER_EPOCH_HASH,
                    records=records,
                    expected_eval_run_audit=receipt_audit,
                    expected_final_best_eval_stats=receipt_best,
                    expected_final_best_score=0.43,
                )

            self.assertEqual(immutable_snapshot.read_bytes(), snapshot_bytes)

    def test_completed_receipt_rejects_zero_trained_rounds(self) -> None:
        empty_audit = _eval_run_audit(trained_rounds=0)
        reference_best = {
            "anchor_kind": "reference",
            "accepted_round_id": None,
            "public_accept": 0.40,
        }
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            _write_completed_invocation(
                output_dir=output_dir,
                eval_run_audit=empty_audit,
                final_best_eval_stats=reference_best,
                final_best_score=0.40,
            )
            records = [
                {"round_id": f"round-{index:02d}"}
                for index in range(
                    1,
                    STAIR_FINAL_CONFIRMATION_REQUIRED_ROUNDS + 1,
                )
            ]

            with self.assertRaisesRegex(
                ValueError,
                "requires complete train/eval accounting",
            ):
                self.script._assert_completed_live_run(
                    output_dir=output_dir,
                    session_id=SESSION_ID,
                    session_genesis_hash=SESSION_GENESIS_HASH,
                    expected_ledger_epoch_hash=LEDGER_EPOCH_HASH,
                    records=records,
                    expected_eval_run_audit=empty_audit,
                    expected_final_best_eval_stats=reference_best,
                    expected_final_best_score=0.40,
                )

    def test_completed_receipt_accepts_one_invocation_from_empty_ledger(self) -> None:
        audit = _eval_run_audit()
        best = {
            "anchor_kind": "candidate",
            "accepted_round_id": "round-15",
            "public_accept": 0.42,
        }
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            _write_completed_invocation(
                output_dir=output_dir,
                eval_run_audit=audit,
                final_best_eval_stats=best,
                final_best_score=0.42,
            )

            self.script._assert_completed_live_run(
                output_dir=output_dir,
                session_id=SESSION_ID,
                session_genesis_hash=SESSION_GENESIS_HASH,
                expected_ledger_epoch_hash=LEDGER_EPOCH_HASH,
                records=[
                    {"round_id": f"round-{index:02d}"}
                    for index in range(
                        1,
                        STAIR_FINAL_CONFIRMATION_REQUIRED_ROUNDS + 1,
                    )
                ],
                expected_eval_run_audit=audit,
                expected_final_best_eval_stats=best,
                expected_final_best_score=0.42,
            )

    def test_completed_receipt_rejects_any_prior_live_invocation(self) -> None:
        audit = _eval_run_audit()
        best = {
            "anchor_kind": "candidate",
            "accepted_round_id": "round-15",
            "public_accept": 0.42,
        }
        prior_exits = (
            ("interrupted", "keyboard_interrupt", "KeyboardInterrupt"),
            ("failed", "exception", "RuntimeError"),
        )
        for status, termination_kind, exception_type in prior_exits:
            with self.subTest(status=status), tempfile.TemporaryDirectory() as tmp:
                output_dir = Path(tmp)
                prior = LiveInvocationJournal.start(
                    output_dir=output_dir,
                    session_id=SESSION_ID,
                    session_genesis_hash=SESSION_GENESIS_HASH,
                    ledger_epoch_hash=LEDGER_EPOCH_HASH,
                    rounds_requested=STAIR_FINAL_CONFIRMATION_REQUIRED_ROUNDS,
                    closed_ledger_count=0,
                )
                prior.finish(
                    status=status,
                    termination_kind=termination_kind,
                    closed_ledger_count=0,
                    exception_type=exception_type,
                )
                _write_completed_invocation(
                    output_dir=output_dir,
                    eval_run_audit=audit,
                    final_best_eval_stats=best,
                    final_best_score=0.42,
                )

                with self.assertRaisesRegex(
                    ValueError,
                    "exactly one live invocation",
                ):
                    self.script._assert_completed_live_run(
                        output_dir=output_dir,
                        session_id=SESSION_ID,
                        session_genesis_hash=SESSION_GENESIS_HASH,
                        expected_ledger_epoch_hash=LEDGER_EPOCH_HASH,
                        records=[
                            {"round_id": f"round-{index:02d}"}
                            for index in range(
                                1,
                                STAIR_FINAL_CONFIRMATION_REQUIRED_ROUNDS + 1,
                            )
                        ],
                        expected_eval_run_audit=audit,
                        expected_final_best_eval_stats=best,
                        expected_final_best_score=0.42,
                    )

    def test_completed_receipt_rejects_resumed_final_invocation(self) -> None:
        audit = _eval_run_audit()
        best = {
            "anchor_kind": "candidate",
            "accepted_round_id": "round-15",
            "public_accept": 0.42,
        }
        for resumed_from_round_count in (14, 15):
            with (
                self.subTest(
                    resumed_from_round_count=resumed_from_round_count,
                ),
                tempfile.TemporaryDirectory() as tmp,
            ):
                output_dir = Path(tmp)
                _write_completed_invocation(
                    output_dir=output_dir,
                    eval_run_audit=audit,
                    final_best_eval_stats=best,
                    final_best_score=0.42,
                    resumed_from_round_count=resumed_from_round_count,
                )

                with self.assertRaisesRegex(
                    ValueError,
                    "one live invocation from an empty ledger",
                ):
                    self.script._assert_completed_live_run(
                        output_dir=output_dir,
                        session_id=SESSION_ID,
                        session_genesis_hash=SESSION_GENESIS_HASH,
                        expected_ledger_epoch_hash=LEDGER_EPOCH_HASH,
                        records=[
                            {"round_id": f"round-{index:02d}"}
                            for index in range(
                                1,
                                STAIR_FINAL_CONFIRMATION_REQUIRED_ROUNDS + 1,
                            )
                        ],
                        expected_eval_run_audit=audit,
                        expected_final_best_eval_stats=best,
                        expected_final_best_score=0.42,
                    )

    def test_candidate_checkpoint_resolution_uses_bound_private_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            genesis_path = root / "genesis.pt"
            final_path = root / "candidate.pt"
            genesis_path.write_bytes(b"genesis")
            final_path.write_bytes(b"candidate")
            genesis = SimpleNamespace(
                artifact=SimpleNamespace(
                    anchor=SimpleNamespace(
                        checkpoint_artifact=SimpleNamespace(uri=str(genesis_path)),
                        checkpoint_hash=GENESIS_CHECKPOINT_HASH,
                    )
                )
            )
            current_best = SimpleNamespace(
                checkpoint_hash=FINAL_CHECKPOINT_HASH,
                anchor_kind="candidate",
                accepted_round_id="round-03",
                comparison_identity=lambda: {
                    "anchor_kind": "candidate",
                    "accepted_round_id": "round-03",
                },
            )
            public_uri = "artifact://round-03/checkpoints/best.pt"
            records = [
                {
                    "round_id": "round-03",
                    "train_result": {
                        "checkpoint": {
                            "checksum": FINAL_CHECKPOINT_HASH,
                            "uri": public_uri,
                        }
                    },
                    "private_artifacts": [
                        {
                            "kind": "runner_checkpoint",
                            "uri": str(final_path),
                            "checksum": FINAL_CHECKPOINT_HASH,
                            "reason": "runner_local_checkpoint_path",
                        }
                    ],
                }
            ]

            with patch.object(
                self.script,
                "ReferenceProgramGenesisAnchor",
                SimpleNamespace,
            ):
                resolved = self.script._resolve_checkpoint_pair(
                    genesis=genesis,
                    records=records,
                    final_current_best=current_best,
                )

            self.assertEqual(resolved.genesis_path, genesis_path.resolve())
            self.assertEqual(resolved.final_path, final_path.resolve())
            self.assertEqual(resolved.final_hash, FINAL_CHECKPOINT_HASH)
            self.assertEqual(
                records[0]["train_result"]["checkpoint"]["uri"],
                public_uri,
            )

    def test_candidate_checkpoint_resolution_rejects_unbound_private_path(
        self,
    ) -> None:
        private_ref = {
            "kind": "runner_checkpoint",
            "uri": "unused.pt",
            "checksum": FINAL_CHECKPOINT_HASH,
            "reason": "runner_local_checkpoint_path",
        }
        cases = {
            "missing": None,
            "wrong_checksum": [
                {
                    **private_ref,
                    "checksum": GENESIS_CHECKPOINT_HASH,
                }
            ],
            "duplicate": [dict(private_ref), dict(private_ref)],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            genesis_path = root / "genesis.pt"
            genesis_path.write_bytes(b"genesis")
            genesis = SimpleNamespace(
                artifact=SimpleNamespace(
                    anchor=SimpleNamespace(
                        checkpoint_artifact=SimpleNamespace(uri=str(genesis_path)),
                        checkpoint_hash=GENESIS_CHECKPOINT_HASH,
                    )
                )
            )
            current_best = SimpleNamespace(
                checkpoint_hash=FINAL_CHECKPOINT_HASH,
                anchor_kind="candidate",
                accepted_round_id="round-03",
                comparison_identity=lambda: {"anchor_kind": "candidate"},
            )
            for name, artifacts in cases.items():
                record = {
                    "round_id": "round-03",
                    "train_result": {
                        "checkpoint": {
                            "checksum": FINAL_CHECKPOINT_HASH,
                            "uri": "artifact://round-03/checkpoints/best.pt",
                        }
                    },
                }
                if artifacts is not None:
                    record["private_artifacts"] = artifacts
                with (
                    self.subTest(name=name),
                    patch.object(
                        self.script,
                        "ReferenceProgramGenesisAnchor",
                        SimpleNamespace,
                    ),
                    self.assertRaisesRegex(
                        ValueError,
                        "private artifacts are missing"
                        if name == "missing"
                        else "exactly one bound local checkpoint artifact",
                    ),
                ):
                    self.script._resolve_checkpoint_pair(
                        genesis=genesis,
                        records=[record],
                        final_current_best=current_best,
                    )

    def test_reference_checkpoint_resolution_needs_no_private_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            genesis_path = Path(tmp) / "genesis.pt"
            genesis_path.write_bytes(b"genesis")
            genesis = SimpleNamespace(
                artifact=SimpleNamespace(
                    anchor=SimpleNamespace(
                        checkpoint_artifact=SimpleNamespace(uri=str(genesis_path)),
                        checkpoint_hash=GENESIS_CHECKPOINT_HASH,
                    )
                )
            )
            current_best = SimpleNamespace(
                checkpoint_hash=GENESIS_CHECKPOINT_HASH,
                anchor_kind="reference",
                accepted_round_id=None,
                comparison_identity=lambda: {"anchor_kind": "reference"},
            )

            with patch.object(
                self.script,
                "ReferenceProgramGenesisAnchor",
                SimpleNamespace,
            ):
                resolved = self.script._resolve_checkpoint_pair(
                    genesis=genesis,
                    records=[],
                    final_current_best=current_best,
                )

            self.assertEqual(resolved.final_path, genesis_path.resolve())
            self.assertEqual(resolved.final_hash, GENESIS_CHECKPOINT_HASH)

    def test_main_passes_the_genesis_bound_epoch_hash_into_replay(self) -> None:
        replay_calls: list[dict[str, Any]] = []

        def replay_spy(*args: Any, **kwargs: Any) -> Any:
            _ = args
            replay_calls.append(kwargs)
            return _successful_replay()

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            with _patched_ready_main(
                self.script,
                output_dir=output_dir,
                config=self.config,
                replay_function=replay_spy,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "AppLauncher construction failed",
                ):
                    self.script.main(
                        _entrypoint_argv(output_dir),
                        app_launcher_class=_FailingAppLauncher,
                    )

        self.assertEqual(len(replay_calls), 1)
        self.assertEqual(
            replay_calls[0]["expected_ledger_epoch_hash"],
            LEDGER_EPOCH_HASH,
        )

    def test_wrong_python_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            current = root / "current-python.exe"
            expected = root / "genesis-bound-python.exe"
            current.write_bytes(b"current")
            expected.write_bytes(b"expected")

            with self.assertRaisesRegex(
                ValueError,
                "Genesis-bound Isaac Python",
            ):
                self.script._require_bound_isaac_python(
                    current_python=current,
                    expected_python=expected,
                )

    def test_app_launcher_failure_precedes_durable_reservation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            with _patched_ready_main(
                self.script,
                output_dir=output_dir,
                config=self.config,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "AppLauncher construction failed",
                ):
                    self.script.main(
                        _entrypoint_argv(output_dir),
                        app_launcher_class=_FailingAppLauncher,
                    )

            self.assertFalse(
                stair_final_confirmation_reservation_path(output_dir).exists()
            )

    def test_first_evaluation_failure_preserves_durable_reservation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            with _patched_ready_main(
                self.script,
                output_dir=output_dir,
                config=self.config,
            ) as fixture:
                with patch.object(
                    self.script,
                    "_run_checkpoint_confirmation",
                    side_effect=RuntimeError("first confirmation evaluation failed"),
                ) as run_confirmation:
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "first confirmation evaluation failed",
                    ):
                        self.script.main(
                            _entrypoint_argv(output_dir),
                            app_launcher_class=_FakeAppLauncher,
                        )

            self.assertEqual(run_confirmation.call_count, 1)
            self.assertEqual(
                run_confirmation.call_args.kwargs["checkpoint_path"],
                fixture.checkpoints.genesis_path,
            )
            reservation_path = stair_final_confirmation_reservation_path(output_dir)
            self.assertTrue(reservation_path.is_file())
            reservation = load_stair_final_confirmation_reservation(
                reservation_path,
                expected_plan=fixture.plan,
            )
            self.assertEqual(reservation.session_genesis_hash, SESSION_GENESIS_HASH)
            self.assertEqual(reservation.ledger_checksum, LEDGER_CHECKSUM)


@contextlib.contextmanager
def _patched_ready_main(
    script: Any,
    *,
    output_dir: Path,
    config: StairClimbingTaskConfig,
    replay_function: Callable[..., Any] | None = None,
) -> Iterator[SimpleNamespace]:
    plan = build_stair_final_confirmation_plan(
        session_id=SESSION_ID,
        config=config,
    )
    backend_config = SimpleNamespace(
        isaac_lab_python=sys.executable,
        execution_device="cuda:0",
    )
    composition = SimpleNamespace(
        task_plugin=SimpleNamespace(config=config),
        backend_adapter=SimpleNamespace(config=backend_config),
        runner=object(),
        policy_initialization=object(),
        genesis_identity_snapshot=lambda: {},
        reference_program_artifact=object(),
    )
    genesis = SimpleNamespace(
        session_genesis_hash=SESSION_GENESIS_HASH,
        initial_current_best=object(),
        expected_initial_policy_state_hash="2" * 64,
        backend_execution_environment={"manifest_hash": BACKEND_MANIFEST_HASH},
    )
    lease = SimpleNamespace(assert_owned=Mock())
    checkpoints = script._CheckpointPair(
        genesis_path=output_dir / "genesis.pt",
        genesis_hash=GENESIS_CHECKPOINT_HASH,
        final_path=output_dir / "final.pt",
        final_hash=FINAL_CHECKPOINT_HASH,
        final_anchor={"anchor_kind": "candidate", "round_id": "round-15"},
    )
    source_preflight = SimpleNamespace(source_manifest_hash=SOURCE_MANIFEST_HASH)
    replay_function = replay_function or (lambda *args, **kwargs: _successful_replay())
    records = [
        {"round_id": f"round-{index:02d}"}
        for index in range(1, STAIR_FINAL_CONFIRMATION_REQUIRED_ROUNDS + 1)
    ]

    with ExitStack() as stack:
        stack.enter_context(
            patch.object(
                script,
                "resolve_task_composition",
                return_value=composition,
            )
        )
        stack.enter_context(
            patch.object(
                script,
                "load_stair_final_confirmation_plan",
                return_value=plan,
            )
        )
        stack.enter_context(
            patch.object(script, "prepare_session_genesis", return_value=genesis)
        )
        stack.enter_context(
            patch.object(
                script,
                "primary_ledger_epoch_hash",
                return_value=LEDGER_EPOCH_HASH,
            )
        )
        stack.enter_context(
            patch.object(
                script,
                "acquire_live_session_lease",
                return_value=contextlib.nullcontext(lease),
            )
        )
        stack.enter_context(patch.object(script, "load_ledger", return_value=records))
        stack.enter_context(
            patch.object(script, "replay_acceptance_ledger", new=replay_function)
        )
        stack.enter_context(
            patch.object(script, "build_eval_run_audit", return_value=_eval_run_audit())
        )
        stack.enter_context(patch.object(script, "_assert_completed_live_run"))
        stack.enter_context(
            patch.object(script, "_resolve_checkpoint_pair", return_value=checkpoints)
        )
        stack.enter_context(patch.object(script, "_require_safe_checkpoint"))
        stack.enter_context(
            patch.object(
                script,
                "preflight_stair_execution_source_manifest",
                return_value=source_preflight,
            )
        )
        stack.enter_context(
            patch.object(
                script,
                "verify_current_isaac_lab_execution_environment",
                return_value={
                    "actual_manifest_hash": BACKEND_MANIFEST_HASH,
                    "expected_manifest_hash": BACKEND_MANIFEST_HASH,
                    "status": "pass",
                },
            )
        )
        stack.enter_context(patch.object(script, "_require_bound_isaac_python"))
        stack.enter_context(
            patch.object(script, "file_sha256", return_value=LEDGER_CHECKSUM)
        )
        yield SimpleNamespace(
            plan=plan,
            composition=composition,
            genesis=genesis,
            checkpoints=checkpoints,
        )


def _successful_replay() -> SimpleNamespace:
    current_best = SimpleNamespace(
        to_public_dict=lambda: {"anchor_kind": "candidate"},
        score_for_comparison=lambda: 0.42,
    )
    return SimpleNamespace(
        current_best=current_best,
        raise_for_continuation=lambda: None,
    )


def _entrypoint_argv(output_dir: Path) -> list[str]:
    return [
        "--output-dir",
        str(output_dir),
        "--session-id",
        SESSION_ID,
        "--final-confirmation-plan",
        str(output_dir / "final_confirmation" / "plan.json"),
    ]


def _write_completed_invocation(
    *,
    output_dir: Path,
    eval_run_audit: dict[str, Any],
    final_best_eval_stats: dict[str, Any],
    final_best_score: float,
    resumed_from_round_count: int = 0,
) -> LiveInvocationJournal:
    journal = LiveInvocationJournal.start(
        output_dir=output_dir,
        session_id=SESSION_ID,
        session_genesis_hash=SESSION_GENESIS_HASH,
        ledger_epoch_hash=LEDGER_EPOCH_HASH,
        rounds_requested=STAIR_FINAL_CONFIRMATION_REQUIRED_ROUNDS,
        closed_ledger_count=resumed_from_round_count,
    )
    summary_path = output_dir / "live_summary.json"
    write_live_summary(
        summary_path,
        _live_summary(
            eval_run_audit=eval_run_audit,
            final_best_eval_stats=final_best_eval_stats,
            final_best_score=final_best_score,
            resumed_from_round_count=resumed_from_round_count,
        ),
    )
    journal.finish(
        status="completed",
        termination_kind="normal",
        closed_ledger_count=STAIR_FINAL_CONFIRMATION_REQUIRED_ROUNDS,
        live_summary_path=summary_path,
    )
    return journal


def _live_summary(
    *,
    eval_run_audit: dict[str, Any],
    final_best_eval_stats: dict[str, Any],
    final_best_score: float,
    resumed_from_round_count: int = 0,
) -> dict[str, Any]:
    rounds = STAIR_FINAL_CONFIRMATION_REQUIRED_ROUNDS
    audit_counts = eval_run_audit["counts"]
    checks = {name: True for name in LIVE_LOOP_SUMMARY_REQUIRED_CHECKS}
    return {
        "schema_version": LIVE_LOOP_SUMMARY_SCHEMA_VERSION,
        "mode": "live",
        "session_id": SESSION_ID,
        "session_genesis_hash": SESSION_GENESIS_HASH,
        "ledger_epoch_hash": LEDGER_EPOCH_HASH,
        "session_genesis_kind": "reference",
        "created_at_utc": "2026-08-13T00:00:00+00:00",
        "output_dir": "test-output",
        "session_genesis_dir": "test-output",
        "ledger_epoch": None,
        "ledger_path": "test-output/ledger.jsonl",
        "rounds_requested": rounds,
        "rounds_completed": rounds,
        "resumed_from_round_count": resumed_from_round_count,
        "rounds_executed_this_invocation": rounds - resumed_from_round_count,
        "prompt_payload_paths": ["prompt.json"] * rounds,
        "provider_attempt_count": rounds,
        "provider_attempt_count_for_rounds_closed_this_invocation": (
            rounds - resumed_from_round_count
        ),
        "authoring_evidence_paths": ["evidence.json"] * rounds,
        "provider_attempt_history_authority": {
            "mode": "ledger_bound_immutable_round_artifacts",
            "latest_summary_is_projection": True,
            "round_evidence_count": rounds,
            "bound_round_count": rounds,
            "interrupted_attempt_count": 0,
        },
        "candidate_paths": ["candidate"] * rounds,
        "decision_outcome_counts": {"reject": rounds},
        "runner_candidate_count": rounds,
        "trainable_round_count": rounds,
        "train_result_count": rounds,
        "public_result_count": rounds,
        "authored_candidate_count": audit_counts["authored_candidate_count"],
        "trainable_candidate_count": audit_counts["trainable_candidate_count"],
        "trained_candidate_count": audit_counts["trained_candidate_count"],
        "full_eval_completed_count": audit_counts["full_eval_completed_count"],
        "sealed_validation_completed_count": audit_counts[
            "sealed_validation_completed_count"
        ],
        "accepted_count": audit_counts["accepted_count"],
        "eval_incomplete_count": audit_counts["eval_incomplete_count"],
        "eval_failure_bucket_counts": eval_run_audit["failure_buckets"],
        "provider_failure_count": 0,
        "static_authoring_failure_count": 0,
        "authoring_failure_round_count": 0,
        "round_accounting": {
            "rounds_completed": rounds,
            "runner_candidate_count": rounds,
            "trainable_round_count": rounds,
            "provider_failure_count": 0,
            "static_authoring_failure_count": 0,
            "candidate_paths_semantics": "test fixture",
        },
        "final_best_score": final_best_score,
        "final_best_eval_stats": final_best_eval_stats,
        "checks": checks,
        "eval_run_audit": eval_run_audit,
    }


def _eval_run_audit(*, trained_rounds: int | None = None) -> dict[str, Any]:
    rounds = STAIR_FINAL_CONFIRMATION_REQUIRED_ROUNDS
    completed = rounds if trained_rounds is None else trained_rounds
    return {
        "schema_version": "true_rl_agent.eval_run_audit.v1",
        "counts": {
            "authored_candidate_count": rounds,
            "trainable_candidate_count": rounds,
            "trained_candidate_count": completed,
            "full_eval_completed_count": completed,
            "sealed_validation_completed_count": completed,
            "accepted_count": 0,
            "eval_incomplete_count": 0,
        },
        "failure_buckets": {},
        "checks": {"complete": True},
        "violations": {"incomplete": []},
        "rounds": [
            {"round_id": f"round-{index:02d}"}
            for index in range(1, rounds + 1)
        ],
    }


def _load_entrypoint() -> Any:
    path = PROJECT_ROOT / "scripts" / "run_stair_final_confirmation.py"
    module_name = "test_run_stair_final_confirmation_entrypoint"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load Stair final confirmation entrypoint: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    unittest.main()
