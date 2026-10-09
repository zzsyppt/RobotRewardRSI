from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from true_rl_agent.tasks.stair_climbing.config import StairClimbingTaskConfig
from true_rl_agent.tasks.stair_climbing.local_paths import (
    resolve_existing_directory,
    resolve_existing_file,
    resolve_fresh_path,
    resolve_local_path,
)

PROTOCOL_PATH = ROOT / "configs" / "tasks" / "stair_protocol_v2_draft_r2.json"


class StairLocalPathTests(unittest.TestCase):
    def test_regular_inputs_and_fresh_outputs_resolve_normally(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.json"
            source.write_text("{}", encoding="ascii")

            self.assertEqual(
                resolve_existing_file(source, field_name="input file"),
                source.resolve(),
            )
            self.assertEqual(
                resolve_existing_directory(root, field_name="input directory"),
                root.resolve(),
            )
            self.assertEqual(
                resolve_fresh_path(root / "output.json", field_name="output"),
                (root / "output.json").resolve(),
            )

    def test_existing_inputs_reject_leaf_symlinks_before_resolve(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target_file = root / "target.json"
            target_file.write_text("{}", encoding="ascii")
            file_link = root / "file-link.json"
            directory_link = root / "directory-link"
            self._symlink_or_skip(file_link, target_file)
            self._symlink_or_skip(directory_link, root, target_is_directory=True)

            with self.assertRaisesRegex(ValueError, "must not be a symlink"):
                resolve_existing_file(file_link, field_name="input file")
            with self.assertRaisesRegex(ValueError, "must not be a symlink"):
                resolve_existing_directory(directory_link, field_name="input directory")

    def test_output_guards_reject_dangling_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            link = root / "output.json"
            self._symlink_or_skip(link, root / "missing-target.json")

            with self.assertRaisesRegex(ValueError, "must not be a symlink"):
                resolve_local_path(link, field_name="output")
            with self.assertRaisesRegex(FileExistsError, "must be fresh"):
                resolve_fresh_path(link, field_name="output")

    def test_stair_config_rejects_protocol_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            link = Path(tmp) / "stair_protocol_v2_draft_r2.json"
            self._symlink_or_skip(link, PROTOCOL_PATH)

            with self.assertRaisesRegex(ValueError, "must not be a symlink"):
                StairClimbingTaskConfig.defaults(link)

    def _symlink_or_skip(
        self,
        link: Path,
        target: Path,
        *,
        target_is_directory: bool = False,
    ) -> None:
        try:
            link.symlink_to(target, target_is_directory=target_is_directory)
        except (NotImplementedError, OSError) as exc:
            self.skipTest(f"symlink creation is unavailable: {exc}")


if __name__ == "__main__":
    unittest.main()
