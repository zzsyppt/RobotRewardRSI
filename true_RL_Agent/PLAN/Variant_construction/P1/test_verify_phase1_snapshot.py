from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


P1_ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "phase1_snapshot_verifier",
    P1_ROOT / "verify_phase1_snapshot.py",
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("unable to load Phase 1 snapshot verifier")
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    }


class Phase1SnapshotVerifierTests(unittest.TestCase):
    def test_frozen_snapshot_passes(self) -> None:
        self.assertEqual(VERIFIER.verify()["status"], "pass")

    def test_git_round_trip_preserves_all_evidence_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary) / "repository"
            repository.mkdir()
            shutil.copy2(VERIFIER.PROJECT_ROOT / ".gitattributes", repository)
            evidence = repository / "PLAN/Variant_construction/P1/PUSH_SHARED_LOOP_DRY_RUN"
            shutil.copytree(VERIFIER.EVIDENCE_ROOT, evidence)
            expected = _tree_hashes(evidence)

            subprocess.run(["git", "init", "-q", str(repository)], check=True)
            subprocess.run(
                ["git", "-C", str(repository), "config", "core.autocrlf", "true"],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(repository), "add", ".gitattributes", "PLAN"],
                check=True,
            )
            shutil.rmtree(evidence)
            subprocess.run(
                ["git", "-C", str(repository), "checkout-index", "--all", "--force"],
                check=True,
            )

            self.assertEqual(_tree_hashes(evidence), expected)
            self.assertEqual(VERIFIER.verify(evidence_root=evidence)["status"], "pass")

    def test_semantic_evidence_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            evidence = Path(temporary) / "evidence"
            shutil.copytree(VERIFIER.EVIDENCE_ROOT, evidence)
            audit_path = evidence / "audit_summary.json"
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            audit["accepted_count"] = 1
            audit_path.write_text(
                json.dumps(audit, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            with self.assertRaisesRegex(AssertionError, "P1 evidence entries"):
                VERIFIER.verify(evidence_root=evidence)

    def test_frozen_product_tree_internal_tamper_is_rejected(self) -> None:
        snapshot = json.loads(VERIFIER.SNAPSHOT_PATH.read_text(encoding="utf-8"))
        snapshot["product_tree"]["file_count"] += 1
        snapshot["receipt_hash"] = VERIFIER.canonical_json_hash(
            VERIFIER._without_hash(snapshot, "receipt_hash")
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "snapshot.json"
            path.write_text(json.dumps(snapshot), encoding="utf-8")
            with self.assertRaisesRegex(AssertionError, "frozen product file count"):
                VERIFIER.verify(snapshot_path=path)


if __name__ == "__main__":
    unittest.main()
