from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from true_rl_agent.tasks.push_recovery.seed_policy import derive_runner_seed, public_seed_record


class PushRecoverySeedPolicyTests(unittest.TestCase):
    def test_split_run_seed_derivation_is_deterministic_and_split_specific(self) -> None:
        dev_seed = derive_runner_seed(base_seed=42, split_name="public_dev", run_index=1)
        accept_seed = derive_runner_seed(base_seed=42, split_name="public_accept", run_index=1)

        self.assertEqual(dev_seed, derive_runner_seed(base_seed=42, split_name="public_dev", run_index=1))
        self.assertNotEqual(dev_seed, accept_seed)
        self.assertNotEqual(dev_seed, derive_runner_seed(base_seed=42, split_name="public_dev", run_index=2))

    def test_public_seed_record_exposes_count_and_index_without_raw_seed(self) -> None:
        record = public_seed_record(split_name="public_accept", run_index=1, split_seed_count=3)

        self.assertEqual(record["split"], "public_accept")
        self.assertEqual(record["run_index"], 1)
        self.assertEqual(record["split_seed_count"], 3)
        self.assertNotIn("seed", record)
        self.assertNotIn("raw_seed", record)


if __name__ == "__main__":
    unittest.main()
