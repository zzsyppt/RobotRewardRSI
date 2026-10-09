from __future__ import annotations

import ast
import importlib
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
PACKAGE_ROOT = SRC_ROOT / "true_rl_agent"


GENERIC_PACKAGE_NAMES = (
    "agent_core",
    "prompting",
    "memory",
    "evaluation",
    "runner",
    "audit",
)
PUSH_AUDIT_LITERALS = (
    "push_recovery",
    "push_drift",
    "recovery_logic",
    "mean_base_height",
    "mean_tilt_rad",
)
STAIR_AUDIT_LITERALS = (
    "unitree_stair_climbing",
    "one_way_ascending_stairs",
    "stair_010",
    "terrain_progress_fraction",
    "insufficient_clearance",
    "toe_collision",
    "stair_height_range",
)


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

    def test_generic_packages_do_not_import_concrete_task_packages(self) -> None:
        concrete_task_names = {
            path.name
            for path in (PACKAGE_ROOT / "tasks").iterdir()
            if path.is_dir() and (path / "__init__.py").exists()
        }
        violations: list[str] = []

        for path in _generic_python_files():
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if _imports_concrete_task(alias.name, concrete_task_names):
                            violations.append(_violation(path, node.lineno, alias.name))
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    if _imports_concrete_task(module, concrete_task_names):
                        violations.append(_violation(path, node.lineno, module))
                    if module == "true_rl_agent.tasks":
                        for alias in node.names:
                            if alias.name in concrete_task_names:
                                violations.append(
                                    _violation(
                                        path,
                                        node.lineno,
                                        f"{module}.{alias.name}",
                                    )
                                )

        self.assertEqual(violations, [], "\n".join(violations))

    def test_generic_packages_do_not_own_push_audit_literals(self) -> None:
        violations: list[str] = []
        for path in _generic_python_files():
            text = path.read_text(encoding="utf-8").lower()
            for literal in PUSH_AUDIT_LITERALS:
                if literal in text:
                    violations.append(f"{path.relative_to(PROJECT_ROOT)} contains {literal}")

        self.assertEqual(violations, [], "\n".join(violations))

    def test_generic_packages_do_not_own_stair_semantics(self) -> None:
        violations: list[str] = []
        for path in _generic_python_files():
            text = path.read_text(encoding="utf-8").lower()
            for literal in STAIR_AUDIT_LITERALS:
                if literal in text:
                    violations.append(f"{path.relative_to(PROJECT_ROOT)} contains {literal}")

        self.assertEqual(violations, [], "\n".join(violations))

    def test_push_recovery_package_does_not_own_stair_semantics(self) -> None:
        violations: list[str] = []
        push_root = PACKAGE_ROOT / "tasks" / "push_recovery"
        for path in sorted(push_root.rglob("*.py")):
            text = path.read_text(encoding="utf-8").lower()
            for literal in STAIR_AUDIT_LITERALS:
                if literal in text:
                    violations.append(
                        f"{path.relative_to(PROJECT_ROOT)} contains {literal}"
                    )

        self.assertEqual(violations, [], "\n".join(violations))

    def test_orchestration_does_not_read_legacy_provider_context(self) -> None:
        paths = (
            PACKAGE_ROOT / "agent_core" / "authoring_loop.py",
            PACKAGE_ROOT / "agent_core" / "live_loop.py",
            PACKAGE_ROOT / "agent_core" / "generation.py",
            PACKAGE_ROOT / "tasks" / "push_recovery" / "loop_audit.py",
        )
        violations: list[str] = []
        for path in paths:
            self.assertTrue(path.exists(), str(path))
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if _reads_legacy_public_context(node):
                    violations.append(_violation(path, node.lineno, "public_context read"))

        self.assertEqual(violations, [], "\n".join(violations))


def _generic_python_files() -> list[Path]:
    files: list[Path] = []
    for name in GENERIC_PACKAGE_NAMES:
        root = PACKAGE_ROOT / name
        if root.exists():
            files.extend(sorted(root.rglob("*.py")))
    return files


def _imports_concrete_task(module: str, concrete_task_names: set[str]) -> bool:
    prefix = "true_rl_agent.tasks."
    if not module.startswith(prefix):
        return False
    task_name = module[len(prefix) :].split(".", 1)[0]
    return task_name in concrete_task_names


def _reads_legacy_public_context(node: ast.AST) -> bool:
    if isinstance(node, ast.Subscript):
        return isinstance(node.slice, ast.Constant) and node.slice.value == "public_context"
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        return False
    return (
        node.func.attr == "get"
        and bool(node.args)
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "public_context"
    )


def _violation(path: Path, line: int, detail: str) -> str:
    return f"{path.relative_to(PROJECT_ROOT)}:{line}: {detail}"


if __name__ == "__main__":
    unittest.main()
