from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from true_rl_agent.json_contract import canonical_json_bytes, strict_json_object_loads
from true_rl_agent.runner.job_attempts import JobAttemptJournal, JobAttemptScope
from true_rl_agent.tasks.stair_climbing.generation_attempt_scopes import (
    StairGenerationLaneOperationScopes,
    recover_stair_generation_lane_operation_scopes,
    recover_stair_parent_bootstrap_attempt_scope,
)

_SESSION_GENESIS_HASH = "1" * 64
_SESSION_CONTRACT_HASH = "2" * 64
_BUDGET_CONTRACT_HASH = "3" * 64
_RESOURCE_BINDING_HASH = "4" * 64
_GENERATION_ID = "stair-session-generation-002"
_LANE_ID = "lane-a"


def _scope(
    root: Path,
    *,
    segment_id: str,
    max_infrastructure_retries: int = 0,
    **overrides,
) -> JobAttemptScope:
    values = {
        "journal_root": root,
        "session_genesis_hash": _SESSION_GENESIS_HASH,
        "session_contract_hash": _SESSION_CONTRACT_HASH,
        "generation_id": _GENERATION_ID,
        "lane_id": _LANE_ID,
        "invocation_segment_id": segment_id,
        "budget_contract_hash": _BUDGET_CONTRACT_HASH,
        "attempt_budget_units": max_infrastructure_retries + 1,
        "max_infrastructure_retries": max_infrastructure_retries,
        "resource_binding_hash": _RESOURCE_BINDING_HASH,
    }
    values.update(overrides)
    return JobAttemptScope(**values)


def _scopes(root: Path, *, segment_id: str) -> StairGenerationLaneOperationScopes:
    base = _scope(root, segment_id=segment_id)
    return StairGenerationLaneOperationScopes(
        train=base,
        public_dev=base,
        public_accept=base,
        sealed_validation=replace(
            base,
            attempt_budget_units=2,
            max_infrastructure_retries=1,
        ),
    )


def _job_identity(operation: str) -> tuple[str, str]:
    trial_id = f"{_GENERATION_ID}-{_LANE_ID}"
    if operation == "train":
        return f"{trial_id}-train", "train"
    return (
        f"{trial_id}-eval-{operation.replace('_', '-')}",
        "eval_phase",
    )


def _journal(
    scope: JobAttemptScope,
    *,
    operation: str,
    job_input_hash: str = "5" * 64,
    job_kind: str | None = None,
) -> JobAttemptJournal:
    logical_job_id, expected_kind = _job_identity(operation)
    return JobAttemptJournal(
        scope=scope,
        logical_job_id=logical_job_id,
        job_kind=job_kind or expected_kind,
        job_input_hash=job_input_hash,
        environment_hash="6" * 64,
        execution_device="cuda:0",
        resource_binding_hash=scope.resource_binding_hash,
    )


class StairGenerationAttemptScopeTests(unittest.TestCase):
    def test_parent_bootstrap_new_work_uses_current_segment_and_started_work_keeps_old(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "attempts"
            current = _scope(
                root,
                segment_id="segment-current",
                generation_id="generation-parent-public-dev-bootstrap",
            )
            self.assertEqual(
                recover_stair_parent_bootstrap_attempt_scope(
                    attempt_journal_root=root,
                    session_id="stair-session",
                    current_invocation_segment_id="segment-current",
                    current_scope=current,
                ),
                current,
            )

            old = replace(current, invocation_segment_id="segment-old")
            journal = JobAttemptJournal(
                scope=old,
                logical_job_id="stair-session-eval-public-dev",
                job_kind="eval_phase",
                job_input_hash="7" * 64,
                environment_hash="8" * 64,
                execution_device="cuda:0",
                resource_binding_hash=old.resource_binding_hash,
            )
            journal.reserve(0, previous_result=None)

            self.assertEqual(
                recover_stair_parent_bootstrap_attempt_scope(
                    attempt_journal_root=root,
                    session_id="stair-session",
                    current_invocation_segment_id="segment-current",
                    current_scope=current,
                ),
                old,
            )

    def test_started_operations_keep_old_segment_and_new_operations_use_current(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "attempts"
            old = _scopes(root, segment_id="segment-001")
            current = _scopes(root, segment_id="segment-002")
            train = _journal(old.train, operation="train")
            train.reserve(0, previous_result=None)
            public_dev = _journal(old.public_dev, operation="public_dev")
            public_dev_reservation = public_dev.reserve(0, previous_result=None)
            public_dev.finish(
                public_dev_reservation,
                outcome="completed",
                failure_kind=None,
                retryable=False,
                output_hash="7" * 64,
                returncode=None,
            )

            resolved = recover_stair_generation_lane_operation_scopes(
                attempt_journal_root=root,
                generation_id=_GENERATION_ID,
                lane_id=_LANE_ID,
                current_invocation_segment_id="segment-002",
                current_scopes=current,
            )

            self.assertEqual(resolved.train, old.train)
            self.assertEqual(resolved.public_dev, old.public_dev)
            self.assertEqual(resolved.public_accept, current.public_accept)
            self.assertEqual(
                resolved.sealed_validation,
                current.sealed_validation,
            )
            self.assertEqual(
                resolved.to_generation_lane_runtime_kwargs(),
                {
                    "attempt_scope": old.train,
                    "public_dev_attempt_scope": old.public_dev,
                    "public_accept_attempt_scope": current.public_accept,
                    "sealed_attempt_scope": current.sealed_validation,
                },
            )
            raw = public_dev_reservation.path.read_bytes()
            payload = strict_json_object_loads(
                raw,
                context="test attempt reservation",
            )
            self.assertEqual(raw, canonical_json_bytes(payload, trailing_newline=True))

    def test_no_started_operation_returns_current_scopes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "attempts"
            current = _scopes(root, segment_id="segment-current")

            resolved = recover_stair_generation_lane_operation_scopes(
                attempt_journal_root=root,
                generation_id=_GENERATION_ID,
                lane_id=_LANE_ID,
                current_invocation_segment_id="segment-current",
                current_scopes=current,
            )

            self.assertEqual(resolved, current)
            self.assertFalse(root.exists())
            with self.assertRaisesRegex(ValueError, "lane identity mismatch"):
                recover_stair_generation_lane_operation_scopes(
                    attempt_journal_root=root,
                    generation_id=_GENERATION_ID,
                    lane_id=_LANE_ID,
                    current_invocation_segment_id="different-segment",
                    current_scopes=current,
                )

    def test_retry_reservations_in_one_journal_are_one_started_operation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "attempts"
            old_train = _scope(
                root,
                segment_id="segment-old",
                max_infrastructure_retries=1,
            )
            current_base = _scopes(root, segment_id="segment-current")
            current = replace(
                current_base,
                train=replace(
                    current_base.train,
                    attempt_budget_units=2,
                    max_infrastructure_retries=1,
                ),
            )
            journal = _journal(old_train, operation="train")
            first = journal.reserve(0, previous_result=None)
            first_result = journal.finish(
                first,
                outcome="infrastructure_failure",
                failure_kind="process_failed",
                retryable=True,
                output_hash=None,
                returncode=1,
            )
            journal.reserve(1, previous_result=first_result)

            resolved = recover_stair_generation_lane_operation_scopes(
                attempt_journal_root=root,
                generation_id=_GENERATION_ID,
                lane_id=_LANE_ID,
                current_invocation_segment_id="segment-current",
                current_scopes=current,
            )

            self.assertEqual(resolved.train, old_train)

    def test_multiple_attempt_journals_for_one_operation_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "attempts"
            current = _scopes(root, segment_id="segment-current")
            for segment_id, job_input_hash in (
                ("segment-old-a", "a" * 64),
                ("segment-old-b", "b" * 64),
            ):
                journal = _journal(
                    _scope(root, segment_id=segment_id),
                    operation="public_accept",
                    job_input_hash=job_input_hash,
                )
                journal.reserve(0, previous_result=None)

            with self.assertRaisesRegex(
                ValueError,
                "public_accept operation has multiple attempt journals",
            ):
                recover_stair_generation_lane_operation_scopes(
                    attempt_journal_root=root,
                    generation_id=_GENERATION_ID,
                    lane_id=_LANE_ID,
                    current_invocation_segment_id="segment-current",
                    current_scopes=current,
                )

    def test_cross_contract_and_resource_scopes_are_rejected(self) -> None:
        variants = {
            "session_contract_hash": "8" * 64,
            "budget_contract_hash": "9" * 64,
            "resource_binding_hash": "a" * 64,
            "generation_id": "different-generation",
            "lane_id": "lane-b",
        }
        for field_name, value in variants.items():
            with (
                self.subTest(field_name=field_name),
                tempfile.TemporaryDirectory() as tmp,
            ):
                root = Path(tmp) / "attempts"
                current = _scopes(root, segment_id="segment-current")
                mismatched = _scope(
                    root,
                    segment_id="segment-old",
                    **{field_name: value},
                )
                journal = _journal(mismatched, operation="sealed_validation")
                journal.reserve(0, previous_result=None)

                with self.assertRaisesRegex(ValueError, f"{field_name} mismatch"):
                    recover_stair_generation_lane_operation_scopes(
                        attempt_journal_root=root,
                        generation_id=_GENERATION_ID,
                        lane_id=_LANE_ID,
                        current_invocation_segment_id="segment-current",
                        current_scopes=current,
                    )

    def test_phase_job_kind_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "attempts"
            current = _scopes(root, segment_id="segment-current")
            journal = _journal(
                _scope(root, segment_id="segment-old"),
                operation="public_dev",
                job_kind="train",
            )
            journal.reserve(0, previous_result=None)

            with self.assertRaisesRegex(ValueError, "job identity mismatch"):
                recover_stair_generation_lane_operation_scopes(
                    attempt_journal_root=root,
                    generation_id=_GENERATION_ID,
                    lane_id=_LANE_ID,
                    current_invocation_segment_id="segment-current",
                    current_scopes=current,
                )


if __name__ == "__main__":
    unittest.main()
