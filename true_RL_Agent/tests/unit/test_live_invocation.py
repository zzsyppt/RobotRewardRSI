from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from true_rl_agent.agent_core.live_invocation import (  # noqa: E402
    LIVE_INVOCATION_EXIT_SCHEMA_VERSION,
    LIVE_INVOCATION_START_SCHEMA_VERSION,
    LIVE_LOOP_SUMMARY_REQUIRED_CHECKS,
    LIVE_LOOP_SUMMARY_SCHEMA_VERSION,
    MAX_LIVE_INVOCATION_COUNT,
    MAX_LIVE_SUMMARY_BYTES,
    LiveInvocationJournal,
    find_verified_completed_live_invocation,
    write_live_summary,
)


def _summary_bytes(
    *,
    rounds_requested: int,
    rounds_completed: int,
    checks_passed: bool,
    marker: str = "",
    resumed_from_round_count: int = 0,
) -> bytes:
    checks = {name: True for name in LIVE_LOOP_SUMMARY_REQUIRED_CHECKS}
    checks["eval_run_audit_passed"] = checks_passed
    from true_rl_agent.json_contract import canonical_json_bytes

    return canonical_json_bytes(
        {
            "schema_version": LIVE_LOOP_SUMMARY_SCHEMA_VERSION,
            "mode": "live",
            "session_id": "session-test",
            "session_genesis_hash": "a" * 64,
            "ledger_epoch_hash": "b" * 64,
            "session_genesis_kind": "reference",
            "created_at_utc": marker or "2026-08-13T00:00:00+00:00",
            "output_dir": "test-output",
            "session_genesis_dir": "test-output",
            "ledger_epoch": None,
            "ledger_path": "test-output/ledger.jsonl",
            "rounds_requested": rounds_requested,
            "rounds_completed": rounds_completed,
            "resumed_from_round_count": resumed_from_round_count,
            "rounds_executed_this_invocation": (
                rounds_completed - resumed_from_round_count
            ),
            "prompt_payload_paths": ["prompt.json"] * rounds_completed,
            "provider_attempt_count": rounds_completed,
            "provider_attempt_count_for_rounds_closed_this_invocation": 0,
            "authoring_evidence_paths": ["evidence.json"] * rounds_completed,
            "provider_attempt_history_authority": {
                "mode": "ledger_bound_immutable_round_artifacts",
                "latest_summary_is_projection": True,
                "round_evidence_count": rounds_completed,
                "bound_round_count": rounds_completed,
                "interrupted_attempt_count": 0,
            },
            "candidate_paths": ["candidate"] * rounds_completed,
            "decision_outcome_counts": {"reject": rounds_completed},
            "runner_candidate_count": rounds_completed,
            "trainable_round_count": rounds_completed,
            "train_result_count": rounds_completed,
            "public_result_count": rounds_completed,
            "authored_candidate_count": rounds_completed,
            "trainable_candidate_count": rounds_completed,
            "trained_candidate_count": rounds_completed,
            "full_eval_completed_count": rounds_completed,
            "sealed_validation_completed_count": rounds_completed,
            "accepted_count": 0,
            "eval_incomplete_count": 0,
            "eval_failure_bucket_counts": {},
            "provider_failure_count": 0,
            "static_authoring_failure_count": 0,
            "authoring_failure_round_count": 0,
            "round_accounting": {
                "rounds_completed": rounds_completed,
                "runner_candidate_count": rounds_completed,
                "trainable_round_count": rounds_completed,
                "provider_failure_count": 0,
                "static_authoring_failure_count": 0,
                "candidate_paths_semantics": "test",
            },
            "final_best_score": 0.0,
            "final_best_eval_stats": {},
            "checks": checks,
            "eval_run_audit": {
                "schema_version": "true_rl_agent.eval_run_audit.v1",
                "counts": {
                    "authored_candidate_count": rounds_completed,
                    "trainable_candidate_count": rounds_completed,
                    "trained_candidate_count": rounds_completed,
                    "full_eval_completed_count": rounds_completed,
                    "sealed_validation_completed_count": rounds_completed,
                    "accepted_count": 0,
                    "eval_incomplete_count": 0,
                },
                "failure_buckets": {},
                "checks": {"complete": checks_passed},
                "violations": {"incomplete": [] if checks_passed else ["round"]},
                "rounds": [{}] * rounds_completed,
            },
        },
        trailing_newline=True,
    )


class LiveInvocationJournalTests(unittest.TestCase):
    def test_start_and_exit_are_write_once_non_ledger_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            journal = LiveInvocationJournal.start(
                output_dir=Path(tmp),
                session_id="session-test",
                session_genesis_hash="a" * 64,
                ledger_epoch_hash="b" * 64,
                rounds_requested=15,
                closed_ledger_count=3,
            )
            journal.finish(
                status="failed",
                termination_kind="exception",
                closed_ledger_count=4,
                exception_type="RuntimeError",
            )
            start = json.loads(
                (journal.invocation_dir / "start.json").read_text(encoding="utf-8")
            )
            exit_record = json.loads(
                (journal.invocation_dir / "exit.json").read_text(encoding="utf-8")
            )

            self.assertEqual(
                start["schema_version"],
                LIVE_INVOCATION_START_SCHEMA_VERSION,
            )
            self.assertEqual(start["closed_ledger_count_at_start"], 3)
            self.assertFalse(start["ledger_authority"])
            self.assertEqual(
                exit_record["schema_version"],
                LIVE_INVOCATION_EXIT_SCHEMA_VERSION,
            )
            self.assertEqual(exit_record["status"], "failed")
            self.assertEqual(exit_record["closed_ledger_count_at_exit"], 4)
            self.assertFalse(exit_record["ledger_authority"])
            self.assertTrue(exit_record["partial_rounds_are_not_ledger_records"])
            with self.assertRaises(FileExistsError):
                journal.finish(
                    status="failed",
                    termination_kind="exception",
                    closed_ledger_count=4,
                    exception_type="RuntimeError",
                )

    def test_status_and_termination_kind_must_agree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            journal = LiveInvocationJournal.start(
                output_dir=Path(tmp),
                session_id="session-test",
                session_genesis_hash="a" * 64,
                ledger_epoch_hash="b" * 64,
                rounds_requested=1,
                closed_ledger_count=0,
            )
            with self.assertRaisesRegex(ValueError, "termination kind mismatch"):
                journal.finish(
                    status="completed",
                    termination_kind="exception",
                    closed_ledger_count=0,
                )

    def test_completed_exit_binds_the_exact_live_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            journal = LiveInvocationJournal.start(
                output_dir=output_dir,
                session_id="session-test",
                session_genesis_hash="a" * 64,
                ledger_epoch_hash="b" * 64,
                rounds_requested=1,
                closed_ledger_count=0,
            )
            summary_path = output_dir / "live_summary.json"
            payload = _summary_bytes(
                rounds_requested=1,
                rounds_completed=1,
                checks_passed=True,
            )
            summary_path.write_bytes(payload)
            journal.finish(
                status="completed",
                termination_kind="normal",
                closed_ledger_count=1,
                live_summary_path=summary_path,
            )
            exit_record = json.loads(
                (journal.invocation_dir / "exit.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                exit_record["live_summary"],
                {
                    "path": str(
                        (journal.invocation_dir / "live_summary.json").resolve()
                    ),
                    "checksum": ("sha256:" + hashlib.sha256(payload).hexdigest()),
                    "size_bytes": len(payload),
                },
            )
            self.assertEqual(
                (journal.invocation_dir / "live_summary.json").read_bytes(),
                payload,
            )

    def test_checks_failure_is_not_recorded_as_completed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            journal = LiveInvocationJournal.start(
                output_dir=output_dir,
                session_id="session-test",
                session_genesis_hash="a" * 64,
                ledger_epoch_hash="b" * 64,
                rounds_requested=1,
                closed_ledger_count=0,
            )
            summary_path = output_dir / "live_summary.json"
            summary_path.write_bytes(
                _summary_bytes(
                    rounds_requested=1,
                    rounds_completed=1,
                    checks_passed=False,
                )
            )
            journal.finish(
                status="failed",
                termination_kind="checks_failed",
                closed_ledger_count=1,
                live_summary_path=summary_path,
            )
            exit_record = json.loads(
                (journal.invocation_dir / "exit.json").read_text(encoding="utf-8")
            )
            self.assertEqual(exit_record["status"], "failed")
            self.assertEqual(exit_record["termination_kind"], "checks_failed")
            self.assertNotIn("exception_type", exit_record)

    def test_resume_cannot_overwrite_prior_invocation_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            root_summary = output_dir / "live_summary.json"
            first = LiveInvocationJournal.start(
                output_dir=output_dir,
                session_id="session-test",
                session_genesis_hash="a" * 64,
                ledger_epoch_hash="b" * 64,
                rounds_requested=2,
                closed_ledger_count=0,
            )
            first_payload = _summary_bytes(
                rounds_requested=2,
                rounds_completed=2,
                checks_passed=True,
                marker="first",
            )
            root_summary.write_bytes(first_payload)
            first.finish(
                status="completed",
                termination_kind="normal",
                closed_ledger_count=2,
                live_summary_path=root_summary,
            )
            first_snapshot = first.invocation_dir / "live_summary.json"

            second = LiveInvocationJournal.start(
                output_dir=output_dir,
                session_id="session-test",
                session_genesis_hash="a" * 64,
                ledger_epoch_hash="b" * 64,
                rounds_requested=2,
                closed_ledger_count=2,
            )
            second_payload = _summary_bytes(
                rounds_requested=2,
                rounds_completed=2,
                checks_passed=True,
                marker="second",
                resumed_from_round_count=2,
            )
            root_summary.write_bytes(second_payload)
            second.finish(
                status="completed",
                termination_kind="normal",
                closed_ledger_count=2,
                live_summary_path=root_summary,
            )

            self.assertEqual(first_snapshot.read_bytes(), first_payload)
            self.assertEqual(
                (second.invocation_dir / "live_summary.json").read_bytes(),
                second_payload,
            )

    def test_completed_receipt_verifier_binds_start_exit_snapshot_and_root(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            journal = LiveInvocationJournal.start(
                output_dir=output_dir,
                session_id="session-test",
                session_genesis_hash="a" * 64,
                ledger_epoch_hash="b" * 64,
                rounds_requested=1,
                closed_ledger_count=0,
            )
            payload = _summary_bytes(
                rounds_requested=1,
                rounds_completed=1,
                checks_passed=True,
            )
            summary_path = output_dir / "live_summary.json"
            summary_path.write_bytes(payload)
            journal.finish(
                status="completed",
                termination_kind="normal",
                closed_ledger_count=1,
                live_summary_path=summary_path,
            )

            verified = find_verified_completed_live_invocation(
                output_dir=output_dir,
                session_id="session-test",
                session_genesis_hash="a" * 64,
                ledger_epoch_hash="b" * 64,
                rounds_requested=1,
            )
            self.assertEqual(verified.invocation_id, journal.invocation_id)
            self.assertEqual(verified.summary_size_bytes, len(payload))
            self.assertEqual(
                verified.summary_checksum,
                "sha256:" + hashlib.sha256(payload).hexdigest(),
            )

            summary_path.write_bytes(payload + b" ")
            with self.assertRaisesRegex(ValueError, "no completed live"):
                find_verified_completed_live_invocation(
                    output_dir=output_dir,
                    session_id="session-test",
                    session_genesis_hash="a" * 64,
                    ledger_epoch_hash="b" * 64,
                    rounds_requested=1,
                )

    def test_completed_receipt_verifier_rejects_forged_minimal_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            invocation_dir = output_dir / "live_invocations" / ("a" * 32)
            invocation_dir.mkdir(parents=True)
            (output_dir / "live_summary.json").write_text(
                '{"checks":{"x":true}}', encoding="utf-8"
            )
            (invocation_dir / "start.json").write_text("{}", encoding="utf-8")
            (invocation_dir / "exit.json").write_text(
                json.dumps(
                    {
                        "session_id": "session-test",
                        "session_genesis_hash": "a" * 64,
                        "ledger_epoch_hash": "b" * 64,
                        "status": "completed",
                    }
                ),
                encoding="utf-8",
            )
            (invocation_dir / "live_summary.json").write_text(
                '{"checks":{"x":true}}', encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "no completed live"):
                find_verified_completed_live_invocation(
                    output_dir=output_dir,
                    session_id="session-test",
                    session_genesis_hash="a" * 64,
                    ledger_epoch_hash="b" * 64,
                    rounds_requested=1,
                )

    def test_writer_recomputes_each_live_summary_check(self) -> None:
        for check_name in sorted(LIVE_LOOP_SUMMARY_REQUIRED_CHECKS):
            with (
                self.subTest(check_name=check_name),
                tempfile.TemporaryDirectory() as tmp,
            ):
                output_dir = Path(tmp)
                journal = LiveInvocationJournal.start(
                    output_dir=output_dir,
                    session_id="session-test",
                    session_genesis_hash="a" * 64,
                    ledger_epoch_hash="b" * 64,
                    rounds_requested=1,
                    closed_ledger_count=0,
                )
                summary = json.loads(
                    _summary_bytes(
                        rounds_requested=1,
                        rounds_completed=1,
                        checks_passed=True,
                    )
                )
                summary["checks"][check_name] = False
                summary_path = output_dir / "live_summary.json"
                write_live_summary(summary_path, summary)
                with self.assertRaisesRegex(ValueError, "source facts"):
                    journal.finish(
                        status="completed",
                        termination_kind="normal",
                        closed_ledger_count=1,
                        live_summary_path=summary_path,
                    )

    def test_live_summary_rejects_unbound_authority_fields_and_paths(self) -> None:
        base = json.loads(
            _summary_bytes(
                rounds_requested=1,
                rounds_completed=1,
                checks_passed=True,
            )
        )
        variants = []
        extra_authority = json.loads(json.dumps(base))
        extra_authority["provider_attempt_history_authority"]["extra"] = True
        variants.append(extra_authority)
        negative_interrupted = json.loads(json.dumps(base))
        negative_interrupted["provider_attempt_history_authority"][
            "interrupted_attempt_count"
        ] = -1
        variants.append(negative_interrupted)
        negative_outcome = json.loads(json.dumps(base))
        negative_outcome["decision_outcome_counts"] = {"reject": -1}
        variants.append(negative_outcome)
        incomplete_outcomes = json.loads(json.dumps(base))
        incomplete_outcomes["decision_outcome_counts"] = {}
        variants.append(incomplete_outcomes)
        oversize_path = json.loads(json.dumps(base))
        oversize_path["prompt_payload_paths"] = ["x" * 4097]
        variants.append(oversize_path)
        for summary in variants:
            with (
                self.subTest(summary=summary),
                tempfile.TemporaryDirectory() as tmp,
            ):
                output_dir = Path(tmp)
                journal = LiveInvocationJournal.start(
                    output_dir=output_dir,
                    session_id="session-test",
                    session_genesis_hash="a" * 64,
                    ledger_epoch_hash="b" * 64,
                    rounds_requested=1,
                    closed_ledger_count=0,
                )
                summary_path = output_dir / "live_summary.json"
                write_live_summary(summary_path, summary)
                with self.assertRaises(ValueError):
                    journal.finish(
                        status="completed",
                        termination_kind="normal",
                        closed_ledger_count=1,
                        live_summary_path=summary_path,
                    )

    def test_live_summary_read_and_invocation_enumeration_are_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            output_dir.mkdir(exist_ok=True)
            (output_dir / "live_summary.json").write_bytes(
                b"{" + b" " * MAX_LIVE_SUMMARY_BYTES + b"}"
            )
            (output_dir / "live_invocations").mkdir()
            with self.assertRaisesRegex(ValueError, "exceeds maximum size"):
                find_verified_completed_live_invocation(
                    output_dir=output_dir,
                    session_id="session-test",
                    session_genesis_hash="a" * 64,
                    ledger_epoch_hash="b" * 64,
                    rounds_requested=1,
                )

    def test_live_summary_writer_rejects_oversize_before_creating_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary_path = Path(tmp) / "live_summary.json"
            with self.assertRaisesRegex(ValueError, "byte limit"):
                write_live_summary(
                    summary_path,
                    {"oversize": "x" * MAX_LIVE_SUMMARY_BYTES},
                )
            self.assertFalse(summary_path.exists())

    def test_live_summary_writer_replaces_symlink_without_touching_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            external = root / "external.json"
            external.write_text("preserve-me", encoding="utf-8")
            summary_path = root / "live_summary.json"
            try:
                summary_path.symlink_to(external)
            except OSError as exc:
                self.skipTest(f"symlinks unavailable: {exc}")
            write_live_summary(summary_path, {"safe": True})
            self.assertEqual(external.read_text(encoding="utf-8"), "preserve-me")
            self.assertFalse(summary_path.is_symlink())
            self.assertEqual(
                json.loads(summary_path.read_text(encoding="utf-8")),
                {"safe": True},
            )

    def test_live_invocation_enumeration_is_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            invocation_root = output_dir / "live_invocations"
            invocation_root.mkdir(parents=True)
            for index in range(MAX_LIVE_INVOCATION_COUNT + 1):
                (invocation_root / f"{index:032x}").mkdir()
            (output_dir / "live_summary.json").write_bytes(b"{}")
            with self.assertRaisesRegex(ValueError, "count exceeds"):
                find_verified_completed_live_invocation(
                    output_dir=output_dir,
                    session_id="session-test",
                    session_genesis_hash="a" * 64,
                    ledger_epoch_hash="b" * 64,
                    rounds_requested=1,
                )


if __name__ == "__main__":
    unittest.main()
