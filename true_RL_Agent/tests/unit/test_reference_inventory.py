from __future__ import annotations

import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ReferenceInventoryTests(unittest.TestCase):
    def test_reference_inventory_covers_phase1_required_extraction_map(self) -> None:
        inventory_path = PROJECT_ROOT / "docs" / "reference_inventory.md"
        text = inventory_path.read_text(encoding="utf-8")

        required_terms = [
            "candidate schema",
            "verifier",
            "safe checkpoint",
            "ledger",
            "public projection",
            "training adapter",
            "push-specific",
            "random-only",
            "copy target",
            "design-only reference",
            "codex",
        ]
        missing = [term for term in required_terms if term not in text.lower()]
        self.assertFalse(missing, f"reference inventory is missing required terms: {missing}")

    def test_reference_inventory_keeps_runtime_dependency_boundary_explicit(self) -> None:
        inventory_path = PROJECT_ROOT / "docs" / "reference_inventory.md"
        text = inventory_path.read_text(encoding="utf-8").lower()

        self.assertIn("must not import", text)
        self.assertIn("must not call parent scripts", text)
        self.assertIn("must not require codex", text)


if __name__ == "__main__":
    unittest.main()

