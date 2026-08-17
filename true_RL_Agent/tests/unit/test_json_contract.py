from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from true_rl_agent.agent_core.ledger import (
    LEDGER_RECORD_SCHEMA_V2,
    LEDGER_RECORD_SCHEMA_V3,
    LEDGER_RECORD_SCHEMA_V4,
    LEDGER_RECORD_SCHEMA_V6,
    LEDGER_RECORD_SCHEMA_VERSION,
    NON_PROVIDER_AUTHORING_MODE,
    append_ledger_record,
    append_ledger_record_once,
    dispatch_ledger_record_schema,
    load_ledger,
)
from true_rl_agent.json_contract import (
    canonical_json_bytes,
    load_strict_json_object,
    strict_json_object_loads,
)

TEST_LEDGER_EPOCH_HASH = "e" * 64


class StrictJsonContractTests(unittest.TestCase):
    def test_rejects_nested_duplicate_keys(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate key: subject_id"):
            strict_json_object_loads(
                b'{"subject":{"subject_id":"forged","subject_id":"actual"}}'
            )

    def test_canonical_file_contract_compares_exact_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.json"
            value = {"schema_version": "v1", "value": 1}
            path.write_bytes(b'{ "schema_version": "v1", "value": 1 }\n')

            with self.assertRaisesRegex(ValueError, "not canonical JSON"):
                load_strict_json_object(
                    path,
                    require_canonical=True,
                    trailing_newline=True,
                )

            path.write_bytes(canonical_json_bytes(value, trailing_newline=True))
            self.assertEqual(
                load_strict_json_object(
                    path,
                    require_canonical=True,
                    trailing_newline=True,
                ),
                value,
            )

    def test_ledger_rejects_duplicate_identity_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.jsonl"
            path.write_bytes(
                b'{"round_id":"forged","round_id":"actual"}\n'
            )

            with self.assertRaisesRegex(ValueError, "invalid ledger JSON at line 1"):
                load_ledger(path)

    def test_ledger_schema_dispatch_rejects_unknown_or_unversioned_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.jsonl"
            for record in (
                {},
                {"schema_version": LEDGER_RECORD_SCHEMA_V3},
                {"schema_version": "true_rl_agent.ledger_record.v999"},
            ):
                with self.subTest(record=record):
                    path.write_bytes(
                        canonical_json_bytes(record, trailing_newline=True)
                    )
                    with self.assertRaisesRegex(ValueError, "invalid ledger schema"):
                        load_ledger(path)

            released = {"schema_version": LEDGER_RECORD_SCHEMA_V2}
            path.write_bytes(canonical_json_bytes(released, trailing_newline=True))
            self.assertEqual(load_ledger(path), [released])
            with self.assertRaisesRegex(ValueError, "current schema"):
                append_ledger_record(path, released)

            historical = {
                "schema_version": LEDGER_RECORD_SCHEMA_V4,
                "authoring_mode": NON_PROVIDER_AUTHORING_MODE,
            }
            path.write_bytes(
                canonical_json_bytes(historical, trailing_newline=True)
            )
            self.assertEqual(load_ledger(path), [historical])
            with self.assertRaisesRegex(ValueError, "current schema"):
                append_ledger_record(path, historical)

            historical_v6 = {
                "schema_version": LEDGER_RECORD_SCHEMA_V6,
                "authoring_mode": NON_PROVIDER_AUTHORING_MODE,
                "ledger_epoch_hash": TEST_LEDGER_EPOCH_HASH,
            }
            path.write_bytes(
                canonical_json_bytes(historical_v6, trailing_newline=True)
            )
            self.assertEqual(load_ledger(path), [historical_v6])
            with self.assertRaisesRegex(ValueError, "current schema"):
                append_ledger_record(path, historical_v6)

    def test_v7_is_the_only_round_authoring_anchor_bound_ledger_schema(
        self,
    ) -> None:
        v6 = dispatch_ledger_record_schema(
            {
                "schema_version": LEDGER_RECORD_SCHEMA_V6,
                "authoring_mode": NON_PROVIDER_AUTHORING_MODE,
            }
        )
        v7 = dispatch_ledger_record_schema(
            {
                "schema_version": LEDGER_RECORD_SCHEMA_VERSION,
                "authoring_mode": NON_PROVIDER_AUTHORING_MODE,
            }
        )

        self.assertFalse(v6.current)
        self.assertTrue(v6.mechanism_delta_bound)
        self.assertFalse(v6.round_authoring_anchor_bound)
        self.assertTrue(v7.current)
        self.assertTrue(v7.round_authoring_anchor_bound)

    def test_current_ledger_record_requires_epoch_hash(self) -> None:
        record = {
            "schema_version": LEDGER_RECORD_SCHEMA_VERSION,
            "authoring_mode": NON_PROVIDER_AUTHORING_MODE,
            "round_id": "round-001",
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.jsonl"
            with self.assertRaisesRegex(ValueError, "epoch hash"):
                append_ledger_record(path, record)
            path.write_bytes(canonical_json_bytes(record, trailing_newline=True))
            with self.assertRaisesRegex(ValueError, "invalid ledger schema"):
                load_ledger(path)

    def test_ledger_append_is_atomic_when_replace_fails(self) -> None:
        first = {
            "schema_version": LEDGER_RECORD_SCHEMA_VERSION,
            "authoring_mode": NON_PROVIDER_AUTHORING_MODE,
            "ledger_epoch_hash": TEST_LEDGER_EPOCH_HASH,
            "round_id": "round-001",
        }
        second = {**first, "round_id": "round-002"}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.jsonl"
            append_ledger_record(path, first)
            original = path.read_bytes()

            with patch(
                "true_rl_agent.agent_core.ledger.os.replace",
                side_effect=OSError("injected replace failure"),
            ):
                with self.assertRaisesRegex(OSError, "injected replace failure"):
                    append_ledger_record(path, second)

            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(load_ledger(path), [first])

    def test_idempotent_commit_conflict_reports_first_difference_and_hashes(
        self,
    ) -> None:
        first = {
            "schema_version": LEDGER_RECORD_SCHEMA_VERSION,
            "authoring_mode": NON_PROVIDER_AUTHORING_MODE,
            "session_genesis_hash": "a" * 64,
            "ledger_epoch_hash": TEST_LEDGER_EPOCH_HASH,
            "round_id": "round-001",
            "candidate": {"candidate_id": "first"},
        }
        conflicting = {
            **first,
            "candidate": {"candidate_id": "different"},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.jsonl"
            self.assertTrue(append_ledger_record_once(path, first))

            with self.assertRaisesRegex(
                ValueError,
                "first_difference=candidate; existing_record_hash=.*; "
                "incoming_record_hash=",
            ):
                append_ledger_record_once(path, conflicting)

            self.assertEqual(load_ledger(path), [first])

    def test_concurrent_ledger_appends_are_serialized_without_lost_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.jsonl"

            def append(index: int) -> None:
                append_ledger_record(
                    path,
                    {
                        "schema_version": LEDGER_RECORD_SCHEMA_VERSION,
                        "authoring_mode": NON_PROVIDER_AUTHORING_MODE,
                        "ledger_epoch_hash": TEST_LEDGER_EPOCH_HASH,
                        "round_id": f"round-{index:03d}",
                    },
                )

            with ThreadPoolExecutor(max_workers=40) as executor:
                list(executor.map(append, range(40)))

            records = load_ledger(path)

        self.assertEqual(len(records), 40)
        self.assertEqual(
            {record["round_id"] for record in records},
            {f"round-{index:03d}" for index in range(40)},
        )

    def test_cross_process_ledger_appends_are_serialized(self) -> None:
        script = """
import sys
from pathlib import Path
from true_rl_agent.agent_core.ledger import (
    LEDGER_RECORD_SCHEMA_VERSION,
    NON_PROVIDER_AUTHORING_MODE,
    append_ledger_record,
)
append_ledger_record(
    Path(sys.argv[1]),
    {
        "schema_version": LEDGER_RECORD_SCHEMA_VERSION,
        "authoring_mode": NON_PROVIDER_AUTHORING_MODE,
        "ledger_epoch_hash": "e" * 64,
        "round_id": sys.argv[2],
    },
)
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.jsonl"
            environment = dict(os.environ)
            environment["PYTHONPATH"] = os.pathsep.join(
                (
                    str(SRC_ROOT),
                    str(PROJECT_ROOT),
                    environment.get("PYTHONPATH", ""),
                )
            )
            processes = [
                subprocess.Popen(
                    [
                        sys.executable,
                        "-c",
                        script,
                        str(path),
                        f"round-{index:03d}",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=environment,
                )
                for index in range(12)
            ]
            completions = [process.communicate(timeout=30) for process in processes]
            for process, (stdout, stderr) in zip(processes, completions, strict=True):
                self.assertEqual(
                    process.returncode,
                    0,
                    msg=f"stdout={stdout}\nstderr={stderr}",
                )
            records = load_ledger(path)

        self.assertEqual(len(records), 12)
        self.assertEqual(
            {record["round_id"] for record in records},
            {f"round-{index:03d}" for index in range(12)},
        )


if __name__ == "__main__":
    unittest.main()
