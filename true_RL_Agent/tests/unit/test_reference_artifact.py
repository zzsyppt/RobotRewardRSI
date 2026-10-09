from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tests.fakes.reference_programs import make_verified_stair_reference
from true_rl_agent.reference_artifact import (
    load_reference_program_artifact,
    materialize_reference_program_artifact,
)


class ReferenceArtifactTests(unittest.TestCase):
    def _program(self, root: Path):
        return make_verified_stair_reference(
            root / "source",
            task_id="unitree_stair_climbing",
            capability_profile_hash="a" * 64,
        )

    def test_repeated_materialization_is_content_addressed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            program = self._program(root)
            first = materialize_reference_program_artifact(
                program,
                artifact_root=root / "artifacts",
            )
            second = materialize_reference_program_artifact(
                program,
                artifact_root=root / "artifacts",
            )

            self.assertEqual(first, second)
            self.assertEqual(first.program, program)
            self.assertEqual(Path(first.path).parent.name, program.manifest.reference_id)

    def test_source_byte_drift_and_extra_files_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            program = self._program(root)
            artifact = materialize_reference_program_artifact(
                program,
                artifact_root=root / "artifacts",
            )
            source = Path(artifact.path) / "training_code" / "components.py"
            source.write_bytes(source.read_bytes() + b"\n# drift\n")
            with self.assertRaisesRegex(ValueError, "file evidence mismatch"):
                load_reference_program_artifact(
                    artifact.path,
                    expected_program=program,
                    expected_artifact_hash=artifact.artifact_hash,
                )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            program = self._program(root)
            artifact = materialize_reference_program_artifact(
                program,
                artifact_root=root / "artifacts",
            )
            (Path(artifact.path) / "unexpected.txt").write_text(
                "not part of the artifact",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "only canonical source bytes"):
                load_reference_program_artifact(
                    artifact.path,
                    expected_program=program,
                    expected_artifact_hash=artifact.artifact_hash,
                )

    def test_symlink_root_is_rejected_before_publication(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            program = self._program(root)
            real_root = root / "real-artifacts"
            real_root.mkdir()
            linked_root = root / "linked-artifacts"
            try:
                os.symlink(real_root, linked_root, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"directory symlinks are unavailable: {exc}")

            with self.assertRaisesRegex(ValueError, "root must not be a symlink"):
                materialize_reference_program_artifact(
                    program,
                    artifact_root=linked_root,
                )


if __name__ == "__main__":
    unittest.main()
