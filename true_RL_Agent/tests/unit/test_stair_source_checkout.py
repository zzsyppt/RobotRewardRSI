from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from true_rl_agent.tasks.stair_climbing.source_checkout import (
    inspect_stair_source_checkout,
    require_stair_source_checkout,
)


class StairSourceCheckoutTests(unittest.TestCase):
    def test_tracked_modification_is_rejected_separately(self) -> None:
        with self._repository() as root:
            commit = self._head(root)
            (root / "tracked.txt").write_text("modified\n", encoding="utf-8")

            checkout = inspect_stair_source_checkout(root)

            self.assertTrue(checkout.tracked_modifications)
            self.assertEqual(checkout.untracked_execution_sources, ())
            with self.assertRaisesRegex(ValueError, "tracked modifications"):
                require_stair_source_checkout(root, expected_commit=commit)

    def test_each_untracked_execution_source_root_is_rejected(self) -> None:
        with self._repository() as root:
            commit = self._head(root)
            expected = (
                "configs/untracked.json",
                "scripts/untracked.py",
                "src/untracked.py",
            )
            for relative_path in expected:
                path = root / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("untracked\n", encoding="utf-8")

            checkout = inspect_stair_source_checkout(root)

            self.assertFalse(checkout.tracked_modifications)
            self.assertEqual(checkout.untracked_execution_sources, expected)
            self.assertFalse(checkout.clean)
            with self.assertRaisesRegex(ValueError, "untracked execution source"):
                require_stair_source_checkout(root, expected_commit=commit)

    def test_untracked_outputs_and_tests_do_not_dirty_execution_source(self) -> None:
        with self._repository() as root:
            commit = self._head(root)
            for relative_path in (
                "outputs/result.json",
                "tests/local_diagnostic.py",
            ):
                path = root / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("untracked\n", encoding="utf-8")

            checkout = require_stair_source_checkout(
                root,
                expected_commit=commit,
            )

            self.assertTrue(checkout.clean)
            self.assertEqual(checkout.untracked_execution_sources, ())

    def test_commit_drift_is_rejected_separately(self) -> None:
        with self._repository() as root:
            with self.assertRaisesRegex(ValueError, "commit drift"):
                require_stair_source_checkout(
                    root,
                    expected_commit="f" * 40,
                )

    def _repository(self):
        return _TemporaryGitRepository()

    def _head(self, root: Path) -> str:
        return _git(root, "rev-parse", "HEAD").decode("ascii").strip()


class _TemporaryGitRepository:
    def __init__(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()

    def __enter__(self) -> Path:
        root = Path(self._temporary_directory.__enter__()).resolve()
        _git(root, "init", "--quiet")
        _git(root, "config", "user.name", "Source Checkout Test")
        _git(root, "config", "user.email", "source-checkout-test@example.invalid")
        (root / "tracked.txt").write_text("tracked\n", encoding="utf-8")
        _git(root, "add", "tracked.txt")
        _git(root, "commit", "--quiet", "-m", "initial")
        return root

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._temporary_directory.__exit__(exc_type, exc_value, traceback)


def _git(root: Path, *arguments: str) -> bytes:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        stdin=subprocess.DEVNULL,
        capture_output=True,
    ).stdout


if __name__ == "__main__":
    unittest.main()
