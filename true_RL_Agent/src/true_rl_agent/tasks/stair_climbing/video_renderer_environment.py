"""Project-owned isolated dependency environment for Stair video replay."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RENDERER_ENVIRONMENT_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.video_renderer_environment.v1"
)
RENDERER_ENVIRONMENT_VARIABLE = "TRUE_RL_AGENT_STAIR_RENDERER_ENVIRONMENT"
RENDERER_REQUIREMENTS_RELATIVE_PATH = Path(
    "requirements/stair_checkpoint_video_renderer.txt"
)
RENDERER_CACHE_RELATIVE_PATH = Path(
    "outputs/runtime_dependencies/stair_checkpoint_video_renderer"
)
_MARKER_FILENAME = "renderer_environment.json"
_IMPORT_NAMES = {
    "imageio": "imageio",
    "imageio-ffmpeg": "imageio_ffmpeg",
    "numpy": "numpy",
    "opencv-python-headless": "cv2",
    "pillow": "PIL",
    "pybullet": "pybullet",
}


@dataclass(frozen=True)
class StairVideoRendererEnvironment:
    project_root: Path
    target_dir: Path
    python_executable: Path
    environment_id: str
    requirements_path: Path
    requirements_checksum: str
    package_versions: Mapping[str, str]

    @property
    def marker_path(self) -> Path:
        return self.target_dir / _MARKER_FILENAME

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "schema_version": RENDERER_ENVIRONMENT_SCHEMA_VERSION,
            "environment_id": self.environment_id,
            "dependency_lock": {
                "path": RENDERER_REQUIREMENTS_RELATIVE_PATH.as_posix(),
                "checksum": self.requirements_checksum,
            },
            "packages": dict(sorted(self.package_versions.items())),
            "python": {
                "implementation": sys.implementation.name,
                "version": _python_version(),
            },
            "platform": {
                "system": platform.system().lower(),
                "machine": platform.machine().lower(),
            },
            "isolation": {
                "kind": "project_owned_content_addressed_venv",
                "cache_path": RENDERER_CACHE_RELATIVE_PATH.as_posix(),
                "isolated_python_mode": True,
                "system_site_packages": False,
                "user_site_disabled": True,
                "global_environment_modified": False,
            },
        }


def build_renderer_environment(
    project_root: Path,
) -> StairVideoRendererEnvironment:
    """Bind the exact renderer lock to this interpreter and host ABI."""

    root = Path(project_root).resolve(strict=True)
    requirements_path = (root / RENDERER_REQUIREMENTS_RELATIVE_PATH).resolve(
        strict=True
    )
    if root not in requirements_path.parents:
        raise ValueError("Stair renderer dependency lock escaped the project root")
    requirements_bytes = requirements_path.read_bytes()
    requirements_checksum = f"sha256:{hashlib.sha256(requirements_bytes).hexdigest()}"
    packages = _parse_exact_requirements(requirements_bytes.decode("utf-8"))
    identity = json.dumps(
        {
            "schema": RENDERER_ENVIRONMENT_SCHEMA_VERSION,
            "lock": requirements_checksum,
            "python": [sys.implementation.name, _python_version()],
            "platform": [platform.system().lower(), platform.machine().lower()],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    environment_id = hashlib.sha256(identity).hexdigest()[:24]
    cache_root = (root / RENDERER_CACHE_RELATIVE_PATH).resolve()
    if root not in cache_root.parents:
        raise ValueError("Stair renderer cache escaped the project root")
    target_dir = cache_root / environment_id
    python_relative = (
        Path("Scripts/python.exe") if os.name == "nt" else Path("bin/python")
    )
    return StairVideoRendererEnvironment(
        project_root=root,
        target_dir=target_dir,
        python_executable=target_dir / python_relative,
        environment_id=environment_id,
        requirements_path=requirements_path,
        requirements_checksum=requirements_checksum,
        package_versions=packages,
    )


def ensure_renderer_environment(
    project_root: Path,
) -> StairVideoRendererEnvironment:
    """Provision or verify the content-addressed project venv."""

    environment = build_renderer_environment(project_root)
    if _environment_is_ready(environment):
        return environment
    _remove_generated_target(environment)
    environment.target_dir.parent.mkdir(parents=True, exist_ok=True)
    clean_env = dict(os.environ)
    clean_env.pop("PYTHONPATH", None)
    clean_env["PYTHONNOUSERSITE"] = "1"
    clean_env["PIP_CONFIG_FILE"] = os.devnull
    try:
        _run_checked(
            [sys.executable, "-m", "venv", str(environment.target_dir)],
            cwd=environment.project_root,
            env=clean_env,
            operation="creation",
        )
        _run_checked(
            [
                str(environment.python_executable),
                "-m",
                "pip",
                "install",
                "--isolated",
                "--disable-pip-version-check",
                "--no-input",
                "--only-binary=:all:",
                "--requirement",
                str(environment.requirements_path),
            ],
            cwd=environment.project_root,
            env=clean_env,
            operation="dependency installation",
        )
        _probe_environment(environment)
        environment.marker_path.write_text(
            json.dumps(
                _marker_payload(environment),
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        if not _environment_is_ready(environment):
            raise RuntimeError("Stair renderer environment failed post-install verification")
        return environment
    except Exception:
        _remove_generated_target(environment)
        raise


def renderer_worker_command(
    environment: StairVideoRendererEnvironment,
    script_path: Path,
    arguments: list[str],
) -> list[str]:
    return [
        str(environment.python_executable),
        "-I",
        str(Path(script_path).resolve(strict=True)),
        *arguments,
    ]


def renderer_worker_environment(
    environment: StairVideoRendererEnvironment,
) -> dict[str, str]:
    """Return a worker environment with no caller-controlled import path."""

    result = dict(os.environ)
    result.pop("PYTHONPATH", None)
    result["PYTHONNOUSERSITE"] = "1"
    result[RENDERER_ENVIRONMENT_VARIABLE] = str(environment.marker_path)
    return result


def require_active_renderer_environment(
    project_root: Path,
) -> StairVideoRendererEnvironment:
    """Fail closed unless this process is the isolated managed render worker."""

    environment = build_renderer_environment(project_root)
    raw_marker_path = os.environ.get(RENDERER_ENVIRONMENT_VARIABLE)
    if not sys.flags.isolated:
        raise RuntimeError("Stair renderer worker must run with Python -I")
    if not raw_marker_path or Path(raw_marker_path).resolve() != environment.marker_path:
        raise RuntimeError("Stair renderer worker environment identity drifted")
    if Path(sys.prefix).resolve() != environment.target_dir:
        raise RuntimeError("Stair renderer worker is not running from its managed venv")
    if not _venv_excludes_system_site(environment):
        raise RuntimeError("Stair renderer venv permits system site-packages")
    if _read_marker(environment.marker_path) != _marker_payload(environment):
        raise RuntimeError("Stair renderer worker marker does not match its lock")
    if _current_package_versions(environment) != dict(environment.package_versions):
        raise RuntimeError("Stair renderer worker package versions drifted")
    return environment


def _environment_is_ready(environment: StairVideoRendererEnvironment) -> bool:
    if (
        not environment.target_dir.is_dir()
        or environment.target_dir.is_symlink()
        or not environment.python_executable.is_file()
        or not _venv_excludes_system_site(environment)
        or _read_marker(environment.marker_path) != _marker_payload(environment)
    ):
        return False
    try:
        _probe_environment(environment)
    except RuntimeError:
        return False
    return True


def _probe_environment(environment: StairVideoRendererEnvironment) -> None:
    probe = subprocess.run(
        [
            str(environment.python_executable),
            "-I",
            "-c",
            _PROBE_SOURCE,
            json.dumps(dict(environment.package_versions), sort_keys=True),
            json.dumps(_IMPORT_NAMES, sort_keys=True),
            str(environment.target_dir),
        ],
        cwd=environment.target_dir,
        env={**os.environ, "PYTHONNOUSERSITE": "1"},
        check=False,
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        detail = (probe.stderr or probe.stdout).strip()[-2000:]
        raise RuntimeError("Stair renderer dependency probe failed: " + detail)
    lines = [line for line in probe.stdout.splitlines() if line.strip()]
    try:
        payload = json.loads(lines[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise RuntimeError("Stair renderer dependency probe output is invalid") from exc
    expected = dict(environment.package_versions)
    if not isinstance(payload, Mapping) or payload.get("packages") != expected:
        raise RuntimeError("Stair renderer dependency versions do not match the lock")
    if payload.get("isolated") is not True or payload.get("prefix") != str(
        environment.target_dir
    ):
        raise RuntimeError("Stair renderer dependency probe was not isolated")


def _current_package_versions(
    environment: StairVideoRendererEnvironment,
) -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution, expected_version in environment.package_versions.items():
        importlib.import_module(_IMPORT_NAMES[distribution])
        actual_version = importlib.metadata.version(distribution)
        if actual_version != expected_version:
            raise RuntimeError(
                f"Stair renderer package {distribution} differs from its lock"
            )
        versions[distribution] = actual_version
    return dict(sorted(versions.items()))


def _parse_exact_requirements(content: str) -> dict[str, str]:
    packages: dict[str, str] = {}
    for line_number, raw_line in enumerate(content.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.count("==") != 1:
            raise ValueError(
                f"Stair renderer requirement line {line_number} is not exact"
            )
        name, version = (part.strip() for part in line.split("==", maxsplit=1))
        name = name.lower()
        if name not in _IMPORT_NAMES or not version or name in packages:
            raise ValueError(
                f"Stair renderer requirement line {line_number} is unsupported"
            )
        packages[name] = version
    if set(packages) != set(_IMPORT_NAMES):
        raise ValueError("Stair renderer dependency lock is incomplete")
    return dict(sorted(packages.items()))


def _marker_payload(environment: StairVideoRendererEnvironment) -> dict[str, Any]:
    return {
        "schema_version": RENDERER_ENVIRONMENT_SCHEMA_VERSION,
        "environment_id": environment.environment_id,
        "requirements_checksum": environment.requirements_checksum,
        "python_implementation": sys.implementation.name,
        "python_version": _python_version(),
        "platform_system": platform.system().lower(),
        "platform_machine": platform.machine().lower(),
        "package_versions": dict(sorted(environment.package_versions.items())),
    }


def _read_marker(path: Path) -> Mapping[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, Mapping) else None


def _venv_excludes_system_site(
    environment: StairVideoRendererEnvironment,
) -> bool:
    try:
        lines = (environment.target_dir / "pyvenv.cfg").read_text(
            encoding="utf-8"
        ).splitlines()
    except OSError:
        return False
    normalized = {line.strip().lower() for line in lines}
    return "include-system-site-packages = false" in normalized


def _remove_generated_target(environment: StairVideoRendererEnvironment) -> None:
    target = environment.target_dir
    if not target.exists() and not target.is_symlink():
        return
    expected_parent = (
        environment.project_root / RENDERER_CACHE_RELATIVE_PATH
    ).resolve()
    if target.parent != expected_parent or target.name != environment.environment_id:
        raise RuntimeError("Refusing to remove an unexpected renderer target")
    target.unlink() if target.is_symlink() else shutil.rmtree(target)


def _run_checked(
    command: list[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    operation: str,
) -> None:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=dict(env),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()[-2000:]
        raise RuntimeError(f"Stair renderer {operation} failed: {detail}")


def _python_version() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


_PROBE_SOURCE = r"""
import importlib
import importlib.metadata
import json
import pathlib
import sys
expected, imports = json.loads(sys.argv[1]), json.loads(sys.argv[2])
target = pathlib.Path(sys.argv[3]).resolve()
versions = {}
for distribution, version in expected.items():
    module = importlib.import_module(imports[distribution])
    if target not in pathlib.Path(module.__file__).resolve().parents:
        raise RuntimeError(f"{distribution} resolved outside managed venv")
    versions[distribution] = importlib.metadata.version(distribution)
    if versions[distribution] != version:
        raise RuntimeError(f"{distribution} version drifted")
print(json.dumps({"isolated": bool(sys.flags.isolated),
                  "prefix": str(pathlib.Path(sys.prefix).resolve()),
                  "packages": versions}, sort_keys=True))
"""
