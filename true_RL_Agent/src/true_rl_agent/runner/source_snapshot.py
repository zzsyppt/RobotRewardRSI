"""Path-independent content identity of the project's executable source closure."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def project_source_snapshot(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve(strict=True)
    files: dict[str, str] = {}
    for relative_root in ("src/true_rl_agent", "scripts", "configs"):
        source_root = root / relative_root
        if not source_root.is_dir():
            raise ValueError(f"project source root is missing: {relative_root}")
        for path in sorted(source_root.rglob("*")):
            if "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            if path.is_file():
                path.resolve(strict=True).relative_to(root)
                files[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"schema_version": "true_rl_agent.project_source_snapshot.v1", "files": files}


def running_project_source_snapshot() -> dict[str, Any]:
    return project_source_snapshot(Path(__file__).resolve().parents[3])
