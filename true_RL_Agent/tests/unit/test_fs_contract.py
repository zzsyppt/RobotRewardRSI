from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from true_rl_agent.fs_contract import (
    atomic_replace_bytes,
    create_once_bytes,
    read_bounded_regular_file,
)


class FileSystemContractTests(unittest.TestCase):
    def test_bounded_reader_accepts_exact_limit_and_rejects_larger_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "artifact.bin"
            artifact.write_bytes(b"1234")
            self.assertEqual(
                read_bounded_regular_file(
                    artifact,
                    max_bytes=4,
                    label="test artifact",
                ),
                b"1234",
            )

            artifact.write_bytes(b"12345")
            with self.assertRaisesRegex(ValueError, "exceeds maximum size"):
                read_bounded_regular_file(
                    artifact,
                    max_bytes=4,
                    label="test artifact",
                )
            with self.assertRaisesRegex(ValueError, "regular file"):
                read_bounded_regular_file(
                    root,
                    max_bytes=4,
                    label="test directory",
                )

    def test_bounded_reader_rejects_final_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "target.bin"
            target.write_bytes(b"safe")
            link = root / "link.bin"
            try:
                link.symlink_to(target)
            except OSError as exc:
                self.skipTest(f"symlinks unavailable: {exc}")
            with self.assertRaisesRegex(ValueError, "regular file"):
                read_bounded_regular_file(
                    link,
                    max_bytes=4,
                    label="test symlink",
                )

    def test_create_once_preserves_first_payload_and_cleans_temporary_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "artifact.bin"
            create_once_bytes(artifact, b"first")
            with self.assertRaises(FileExistsError):
                create_once_bytes(artifact, b"second")
            self.assertEqual(artifact.read_bytes(), b"first")
            self.assertEqual(list(root.iterdir()), [artifact])

    def test_atomic_replace_replaces_existing_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "artifact.bin"
            artifact.write_bytes(b"first")
            atomic_replace_bytes(artifact, b"second")
            self.assertEqual(artifact.read_bytes(), b"second")
            self.assertEqual(list(root.iterdir()), [artifact])

    def test_publication_uses_short_temporary_name_near_windows_path_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target_name = "artifact.json"
            desired_target_length = 236
            segment_length = (
                desired_target_length
                - len(str(root))
                - len(target_name)
                - 2
            )
            if segment_length <= 0 or segment_length > 240:
                self.skipTest("temporary root cannot form the path-length regression")
            parent = root / ("d" * segment_length)
            parent.mkdir()
            artifact = parent / target_name
            legacy_temporary = artifact.with_name(
                f".{artifact.name}.{'0' * 32}.tmp"
            )
            self.assertEqual(len(str(artifact)), desired_target_length)
            self.assertGreater(len(str(legacy_temporary)), 260)

            create_once_bytes(artifact, b"first")
            atomic_replace_bytes(artifact, b"second")
            self.assertEqual(artifact.read_bytes(), b"second")
            self.assertEqual(list(parent.iterdir()), [artifact])


if __name__ == "__main__":
    unittest.main()
