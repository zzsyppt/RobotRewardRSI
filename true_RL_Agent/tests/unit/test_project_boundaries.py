from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"


def _text_files_under(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return [
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix in {".py", ".toml", ".yaml", ".yml", ".json", ".md"}
    ]


class ProjectBoundaryTests(unittest.TestCase):
    def test_project_metadata_declares_independent_package(self) -> None:
        pyproject_path = PROJECT_ROOT / "pyproject.toml"
        self.assertTrue(pyproject_path.is_file(), "true_RL_Agent must be an installable project")

        pyproject_text = pyproject_path.read_text(encoding="utf-8")
        self.assertIn('name = "true-rl-agent"', pyproject_text)
        self.assertIn('requires-python = ">=3.10"', pyproject_text)
        self.assertIn("dependencies = []", pyproject_text)
        self.assertNotIn("humanoid_agent", pyproject_text)
        self.assertNotIn("humanoid-agent", pyproject_text.replace('name = "true-rl-agent"', ""))
        self.assertIn('"" = "src"', pyproject_text)

    def test_package_imports_from_local_src_without_parent_project_imports(self) -> None:
        sys.path.insert(0, str(SRC_ROOT))
        try:
            before = {
                name
                for name in sys.modules
                if name == "humanoid_agent" or name.startswith("humanoid_agent.")
            }
            module = importlib.import_module("true_rl_agent")
            after = {
                name
                for name in sys.modules
                if name == "humanoid_agent" or name.startswith("humanoid_agent.")
            }
        finally:
            sys.path.remove(str(SRC_ROOT))

        self.assertEqual(module.__name__, "true_rl_agent")
        self.assertEqual(after, before)

    def test_runtime_tree_does_not_reference_parent_project(self) -> None:
        forbidden_tokens = [
            "humanoid_agent",
            "D:\\humanoid_agent",
            "/data/yuyangliu/AGENT",
            "../src/humanoid_agent",
            "..\\src\\humanoid_agent",
        ]
        scanned_roots = [
            PROJECT_ROOT / "src",
            PROJECT_ROOT / "scripts",
            PROJECT_ROOT / "configs",
        ]
        scanned_files = [path for root in scanned_roots for path in _text_files_under(root)]
        self.assertTrue(scanned_files, "runtime tree must contain source, script, or config files")

        violations: list[str] = []
        for path in scanned_files:
            text = path.read_text(encoding="utf-8")
            for token in forbidden_tokens:
                if token in text:
                    violations.append(f"{path.relative_to(PROJECT_ROOT)} contains {token!r}")

        self.assertFalse(violations, "\n".join(violations))
