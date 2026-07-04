from __future__ import annotations

import io
import json
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.materialize_job import main


class MaterializeJobScriptTests(unittest.TestCase):
    def test_materialized_job_exports_runner_execution_argv(self) -> None:
        output = io.StringIO()

        with redirect_stdout(output):
            exit_code = main(
                [
                    "--round-id",
                    "round-001",
                    "--candidate-id",
                    "candidate-001",
                    "--candidate-path",
                    "artifact://candidate-001",
                ]
            )

        payload = json.loads(output.getvalue())
        runner_execution = payload["runner_execution"]

        self.assertEqual(exit_code, 0)
        self.assertTrue(runner_execution["train"]["cli_argv"][0].endswith("/bin/python"))
        self.assertEqual(runner_execution["train"]["cli_argv"][1], "scripts/isaaclab_rsl_rl_train.py")
        self.assertIn("artifact://candidate-001", runner_execution["train"]["cli_argv"])
        self.assertTrue(runner_execution["train"]["summary_path"].endswith("round-001/train/summary.json"))
        self.assertTrue(runner_execution["train"]["checkpoint_output_path"].endswith("round-001/checkpoints/best.pt"))
        self.assertTrue(runner_execution["eval"]["cli_argv"][0].endswith("/bin/python"))
        self.assertEqual(runner_execution["eval"]["cli_argv"][1], "scripts/isaaclab_rsl_rl_eval.py")
        self.assertIn("artifact://round-001/checkpoints/best.pt", runner_execution["eval"]["cli_argv"])
        self.assertTrue(runner_execution["eval"]["summary_path"].endswith("round-001/eval/summary.json"))
        self.assertNotIn("trusted_seed_count", json.dumps(runner_execution))


if __name__ == "__main__":
    unittest.main()
