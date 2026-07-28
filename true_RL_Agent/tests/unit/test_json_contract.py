from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from true_rl_agent.agent_core.ledger import load_ledger
from true_rl_agent.json_contract import (
    canonical_json_bytes,
    load_strict_json_object,
    strict_json_object_loads,
)


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


if __name__ == "__main__":
    unittest.main()
