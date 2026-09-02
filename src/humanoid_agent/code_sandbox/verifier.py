"""Phase 8D code-level candidate verifier.

This verifier is deliberately a second line of defense on top of Phase 8C OS
isolation.  It rejects obvious unsafe candidate packages and then loads the
candidate components in the isolated Phase 8C child process.  It does not run a
full Isaac/RSL-RL training loop; that remains a later integration step.
"""

from __future__ import annotations

import ast
import json
import os
import re
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .isolated_loader import load_mapping, run_isolated_component_load
from .isolation import probe_isolation_capabilities
from .loader import CandidateLoadError, candidate_package_fingerprint, load_candidate_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class VerifierIssue:
    severity: str
    code: str
    message: str
    path: str | None = None
    line: int | None = None
    detail: Any = None

    def to_summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
        }
        if self.path is not None:
            out["path"] = self.path
        if self.line is not None:
            out["line"] = self.line
        if self.detail is not None:
            out["detail"] = self.detail
        return out


def verify_candidate_package(
    *,
    candidate_dir: Path,
    config_path: Path,
    run_dir: Path,
    require_server_enforcement: bool = False,
    python_executable: str | None = None,
) -> dict[str, Any]:
    """Verify a code-level candidate package using static and dynamic checks."""

    started = time.time()
    config = load_mapping(config_path)
    static_cfg = _mapping(config.get("static", {}), "static")
    dynamic_cfg = _mapping(config.get("dynamic", {}), "dynamic")
    candidate_dir = candidate_dir.resolve()
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    static_issues: list[VerifierIssue] = []
    manifest = None
    fingerprint: dict[str, Any] | None = None
    static_issues.extend(_verify_file_inventory(candidate_dir, static_cfg))
    manifest_path = candidate_dir / "candidate_manifest.yaml"
    manifest_text, manifest_read_issue = _read_candidate_text(manifest_path, candidate_dir=candidate_dir, cfg=static_cfg)
    if manifest_read_issue is not None:
        static_issues.append(
            VerifierIssue(
                "error",
                "manifest_invalid",
                "candidate manifest could not be read safely",
                manifest_path.relative_to(candidate_dir).as_posix(),
                detail=manifest_read_issue.to_summary(),
            )
        )
    try:
        if manifest_text is not None:
            manifest = load_candidate_manifest(
                candidate_dir,
                expected_schema_version=static_cfg.get("allowed_schema_version") or "humanoid_agent.cold_start.candidate.v1",
            )
            fingerprint = candidate_package_fingerprint(manifest)
        elif manifest_read_issue is None:
            static_issues.append(
                VerifierIssue(
                    "error",
                    "manifest_invalid",
                    "candidate manifest is not a readable regular file",
                    manifest_path.relative_to(candidate_dir).as_posix(),
                )
            )
    except Exception as exc:  # noqa: BLE001
        static_issues.append(VerifierIssue("error", "manifest_invalid", "candidate manifest is invalid", str(candidate_dir), detail=str(exc)))
    if manifest is not None:
        static_issues.extend(_verify_manifest_policy(manifest.path / "candidate_manifest.yaml", static_cfg))
    static_issues.extend(_verify_python_files(candidate_dir, static_cfg))
    static_issues.extend(_verify_text_files(candidate_dir, static_cfg))

    dynamic_summary: dict[str, Any] | None = None
    dynamic_issues: list[VerifierIssue] = []
    capabilities = probe_isolation_capabilities()
    static_status = _status(static_issues)
    if static_status == "pass":
        dynamic_summary, dynamic_issues = _run_dynamic_checks(
            candidate_dir=candidate_dir,
            config=config,
            config_path=config_path,
            run_dir=run_dir / "dynamic",
            dynamic_cfg=dynamic_cfg,
            static_cfg=static_cfg,
            capabilities=capabilities,
            require_server_enforcement=require_server_enforcement,
            python_executable=python_executable,
        )

    all_issues = [*static_issues, *dynamic_issues]
    return {
        "schema_version": str(config.get("summary_schema_version", "humanoid_agent.phase8d.verifier.summary.v1")),
        "phase": str(config.get("phase", "8D-code-verifier")),
        "status": _status(all_issues),
        "candidate_dir": str(candidate_dir),
        "config_path": str(config_path),
        "run_dir": str(run_dir),
        "fingerprint": fingerprint,
        "capabilities": capabilities,
        "checks": {
            "static": {"status": static_status, "issues": [issue.to_summary() for issue in static_issues]},
            "dynamic": {
                "status": "skipped" if dynamic_summary is None and static_status != "pass" else _status(dynamic_issues),
                "issues": [issue.to_summary() for issue in dynamic_issues],
            },
        },
        "dynamic_summary": dynamic_summary,
        "issues": [issue.to_summary() for issue in all_issues],
        "elapsed_seconds": time.time() - started,
    }


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _run_dynamic_checks(
    *,
    candidate_dir: Path,
    config: dict[str, Any],
    config_path: Path,
    run_dir: Path,
    dynamic_cfg: dict[str, Any],
    static_cfg: dict[str, Any],
    capabilities: dict[str, Any],
    require_server_enforcement: bool,
    python_executable: str | None,
) -> tuple[dict[str, Any] | None, list[VerifierIssue]]:
    issues: list[VerifierIssue] = []
    hard_supported = bool(
        capabilities.get("network_namespace_supported")
        and capabilities.get("pid_namespace_supported")
        and capabilities.get("landlock_abi_version")
    )
    if not hard_supported:
        if require_server_enforcement or bool(dynamic_cfg.get("require_phase8c_isolation", True)):
            issues.append(
                VerifierIssue(
                    "error",
                    "phase8c_isolation_unavailable",
                    "Phase 8C isolation is required but not available on this host",
                    str(config_path),
                    detail=capabilities,
                )
            )
            return None, issues
        return {"status": "skipped", "reason": "Phase 8C isolation unavailable on this host"}, issues

    phase8c_config_path = Path(str(config["phase8c_isolation_config"]))
    if not phase8c_config_path.is_absolute():
        phase8c_config_path = PROJECT_ROOT / phase8c_config_path
    isolation_config = load_mapping(phase8c_config_path.resolve())
    exit_code, isolated_summary = run_isolated_component_load(
        candidate_dir=candidate_dir,
        run_dir=run_dir,
        config=isolation_config,
        allow_llm_authored=bool(dynamic_cfg.get("allow_llm_authored_in_isolated_verifier", True)),
        require_noop_components=bool(dynamic_cfg.get("require_noop_components", False)),
        fast_validation=dynamic_cfg.get("fast_validation") if isinstance(dynamic_cfg.get("fast_validation"), dict) else None,
        python_executable=python_executable or dynamic_cfg.get("python_executable"),
    )
    if exit_code != 0 or isolated_summary.get("status") != "pass":
        fast_validation = {}
        child_summary = isolated_summary.get("child_summary")
        if isinstance(child_summary, dict) and isinstance(child_summary.get("fast_validation"), dict):
            fast_validation = child_summary["fast_validation"]
        issue_code = (
            "isolated_dynamic_fast_validation_failed"
            if fast_validation.get("status") == "fail"
            else "isolated_component_load_failed"
        )
        issues.append(
            VerifierIssue(
                "error",
                issue_code,
                "candidate failed isolated dynamic fast validation"
                if issue_code == "isolated_dynamic_fast_validation_failed"
                else "candidate failed isolated Phase 8C component loading",
                detail=fast_validation or isolated_summary.get("errors") or isolated_summary.get("command_result"),
            )
        )
    if bool(dynamic_cfg.get("scan_public_component_summary", True)):
        public_payload = {}
        child_summary = isolated_summary.get("child_summary")
        if isinstance(child_summary, dict):
            public_payload = {
                "candidate": child_summary.get("candidate"),
                "components": child_summary.get("components"),
            }
        issues.extend(_verify_public_payload(public_payload, static_cfg, path="isolated_child_public_summary"))
    return isolated_summary, issues


def _verify_file_inventory(candidate_dir: Path, cfg: dict[str, Any]) -> list[VerifierIssue]:
    issues: list[VerifierIssue] = []
    if not candidate_dir.is_dir():
        return [VerifierIssue("error", "candidate_dir_missing", "candidate directory does not exist", str(candidate_dir))]

    files: list[Path] = []
    total_bytes = 0
    max_files = int(cfg.get("max_files", 64))
    max_file_bytes = int(cfg.get("max_file_bytes", 262144))
    max_total_bytes = int(cfg.get("max_total_bytes", 1048576))
    allowed_top = set(cfg.get("allowed_top_level_entries", []))
    allowed_suffixes = set(cfg.get("allowed_file_suffixes", []))
    forbidden_parts = set(cfg.get("forbidden_path_parts", []))

    for path in _iter_candidate_paths(candidate_dir):
        relative = path.relative_to(candidate_dir)
        relative_text = relative.as_posix()
        parts = set(relative.parts)
        if "__pycache__" in parts or path.suffix == ".pyc":
            continue
        try:
            stat_result = path.lstat()
            mode = stat_result.st_mode
        except OSError as exc:
            issues.append(VerifierIssue("error", "candidate_file_stat_failed", "candidate file could not be stat'ed", relative_text, detail=str(exc)))
            continue
        if path.is_symlink():
            issues.append(VerifierIssue("error", "symlink_not_allowed", "candidate package must not contain symlinks", relative_text))
            continue
        if not stat.S_ISDIR(mode) and not stat.S_ISREG(mode):
            issues.append(VerifierIssue("error", "special_file_not_allowed", "candidate package must not contain special files", relative_text))
            continue
        if parts & forbidden_parts:
            issues.append(VerifierIssue("error", "forbidden_path_part", "candidate path contains forbidden cache/control part", relative_text))
        if relative.parts and allowed_top and relative.parts[0] not in allowed_top:
            issues.append(VerifierIssue("error", "top_level_entry_not_allowed", "candidate top-level entry is not allowed", relative_text))
        if stat.S_ISREG(mode):
            files.append(path)
            size = stat_result.st_size
            total_bytes += size
            if size > max_file_bytes:
                issues.append(VerifierIssue("error", "file_too_large", "candidate file exceeds max_file_bytes", relative_text, detail=size))
            if path.suffix not in allowed_suffixes:
                issues.append(VerifierIssue("error", "file_suffix_not_allowed", "candidate file suffix is not allowed", relative_text))
    if len(files) > max_files:
        issues.append(VerifierIssue("error", "too_many_files", "candidate package exceeds max_files", detail=len(files)))
    if total_bytes > max_total_bytes:
        issues.append(VerifierIssue("error", "package_too_large", "candidate package exceeds max_total_bytes", detail=total_bytes))
    return issues


def _verify_manifest_policy(manifest_path: Path, cfg: dict[str, Any]) -> list[VerifierIssue]:
    issues: list[VerifierIssue] = []
    try:
        manifest = load_mapping(manifest_path)
    except Exception as exc:  # noqa: BLE001
        return [VerifierIssue("error", "manifest_parse_failed", "manifest could not be parsed", str(manifest_path), detail=str(exc))]
    forbidden = set(cfg.get("forbidden_manifest_capabilities", []))
    for key in ("capabilities", "requested_capabilities", "permissions"):
        value = manifest.get(key)
        requested = _capability_names(value)
        for capability in sorted(requested & forbidden):
            issues.append(
                VerifierIssue(
                    "error",
                    "forbidden_manifest_capability",
                    "candidate requested a forbidden capability",
                    str(manifest_path),
                    detail={"field": key, "capability": capability},
                )
            )
    issues.extend(_verify_public_payload({"manifest": manifest}, cfg, path=str(manifest_path)))
    return issues


def _verify_python_files(candidate_dir: Path, cfg: dict[str, Any]) -> list[VerifierIssue]:
    issues: list[VerifierIssue] = []
    training_root = candidate_dir / "training_code"
    try:
        training_root_mode = training_root.lstat().st_mode
    except OSError:
        return [VerifierIssue("error", "training_code_missing", "candidate package must contain training_code/", str(training_root))]
    if stat.S_ISLNK(training_root_mode):
        return []
    if not stat.S_ISDIR(training_root_mode):
        return [VerifierIssue("error", "training_code_missing", "candidate package must contain training_code/", str(training_root))]
    for path in _iter_candidate_paths(training_root):
        if path.suffix != ".py":
            continue
        relative = path.relative_to(candidate_dir).as_posix()
        text, read_issue = _read_candidate_text(path, candidate_dir=candidate_dir, cfg=cfg)
        if read_issue is not None:
            issues.append(read_issue)
            continue
        if text is None:
            continue
        try:
            tree = ast.parse(text, filename=relative)
        except SyntaxError as exc:
            issues.append(VerifierIssue("error", "python_syntax_error", "candidate python file has syntax error", relative, exc.lineno, str(exc)))
            continue
        except (ValueError, TypeError, RecursionError, MemoryError) as exc:
            issues.append(VerifierIssue("error", "python_parse_failed", "candidate python file could not be parsed safely", relative, detail=f"{type(exc).__name__}: {exc}"))
            continue
        issues.extend(_scan_ast(tree, relative, cfg))
    return issues


def _verify_text_files(candidate_dir: Path, cfg: dict[str, Any]) -> list[VerifierIssue]:
    """Scan all candidate text artifacts, not only Python AST and manifest."""

    issues: list[VerifierIssue] = []
    suffixes = set(cfg.get("content_scan_suffixes", cfg.get("allowed_file_suffixes", [])))
    forbidden_numbers = {int(value) for value in cfg.get("forbidden_numeric_tokens", [])}
    seed_re = _forbidden_number_regex(forbidden_numbers)
    forbidden_text = [str(value) for value in cfg.get("forbidden_string_tokens", [])]

    for path in _iter_candidate_paths(candidate_dir):
        if path.suffix not in suffixes:
            continue
        try:
            if not stat.S_ISREG(path.lstat().st_mode):
                continue
        except OSError:
            continue
        text, read_issue = _read_candidate_text(path, candidate_dir=candidate_dir, cfg=cfg)
        if read_issue is not None:
            issues.append(read_issue)
            continue
        if text is None:
            continue
        relative = path.relative_to(candidate_dir).as_posix()
        issues.extend(_text_issues(text, forbidden_text, seed_re, relative, None, "forbidden_content_token"))
    return issues


def _scan_ast(tree: ast.AST, path: str, cfg: dict[str, Any]) -> list[VerifierIssue]:
    issues: list[VerifierIssue] = []
    forbidden_import_roots = set(cfg.get("forbidden_import_roots", []))
    forbidden_call_names = set(cfg.get("forbidden_call_names", []))
    forbidden_numbers = {int(value) for value in cfg.get("forbidden_numeric_tokens", [])}
    forbidden_text = [str(value) for value in cfg.get("forbidden_string_tokens", [])]
    seed_re = _forbidden_number_regex(forbidden_numbers)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root in forbidden_import_roots:
                    issues.append(VerifierIssue("error", "forbidden_import", "candidate imported a forbidden module", path, node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if root in forbidden_import_roots:
                issues.append(VerifierIssue("error", "forbidden_import", "candidate imported from a forbidden module", path, node.lineno, node.module))
        elif isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name in forbidden_call_names:
                issues.append(VerifierIssue("error", "forbidden_call", "candidate called a forbidden builtin", path, node.lineno, name))
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, str):
                issues.extend(_text_issues(node.value, forbidden_text, seed_re, path, getattr(node, "lineno", None), "forbidden_literal"))
            elif isinstance(node.value, int) and node.value in forbidden_numbers:
                issues.append(VerifierIssue("error", "forbidden_numeric_token", "candidate hard-coded protected eval seed", path, getattr(node, "lineno", None), node.value))
        elif isinstance(node, ast.Name):
            issues.extend(_text_issues(node.id, forbidden_text, seed_re, path, getattr(node, "lineno", None), "forbidden_identifier"))
        elif isinstance(node, ast.Attribute):
            issues.extend(_text_issues(node.attr, forbidden_text, seed_re, path, getattr(node, "lineno", None), "forbidden_identifier"))
    return issues


def _verify_public_payload(payload: Any, cfg: dict[str, Any], *, path: str) -> list[VerifierIssue]:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    seed_re = _forbidden_number_regex({int(value) for value in cfg.get("forbidden_numeric_tokens", [])})
    return _text_issues(text, [str(value) for value in cfg.get("forbidden_string_tokens", [])], seed_re, path, None, "forbidden_public_output")


def _text_issues(
    text: str,
    forbidden_text: list[str],
    seed_re: re.Pattern[str],
    path: str,
    line: int | None,
    code: str,
) -> list[VerifierIssue]:
    lowered = text.lower()
    issues: list[VerifierIssue] = []
    for token in forbidden_text:
        if token.lower() in lowered:
            issues.append(VerifierIssue("error", code, "candidate text contains a forbidden protected token", path, line, token))
    match = seed_re.search(text)
    if match:
        issues.append(VerifierIssue("error", "forbidden_numeric_token", "candidate text contains a protected eval seed token", path, line, match.group(0)))
    return issues


def _capability_names(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {value}
    if isinstance(value, list):
        return {str(item) for item in value}
    if isinstance(value, dict):
        return {str(key) for key in value}
    return {str(value)}


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _forbidden_number_regex(numbers: set[int]) -> re.Pattern[str]:
    if not numbers:
        return re.compile(r"(?!x)x")
    joined = "|".join(re.escape(str(number)) for number in sorted(numbers))
    return re.compile(rf"(?<![\d.])(?:{joined})(?![\d.])")


def _iter_candidate_paths(root: Path) -> list[Path]:
    """Enumerate candidate entries without following symlinked directories."""

    if not root.exists():
        return []
    paths: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        current = Path(dirpath)
        kept_dirs: list[str] = []
        for name in sorted(dirnames):
            child = current / name
            paths.append(child)
            try:
                if child.is_symlink():
                    continue
                if not stat.S_ISDIR(child.lstat().st_mode):
                    continue
            except OSError:
                continue
            kept_dirs.append(name)
        dirnames[:] = kept_dirs
        for name in sorted(filenames):
            paths.append(current / name)
    return sorted(paths)


def _read_candidate_text(path: Path, *, candidate_dir: Path, cfg: dict[str, Any]) -> tuple[str | None, VerifierIssue | None]:
    """Read a regular candidate text file without following symlinks.

    Parent-side verifier code runs outside the 8C sandbox, so it must not follow
    candidate symlinks or block on special files while scanning content.
    """

    relative = path.relative_to(candidate_dir).as_posix()
    parts = set(path.relative_to(candidate_dir).parts)
    if "__pycache__" in parts or path.suffix == ".pyc":
        return None, None
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        return None, VerifierIssue("error", "candidate_file_stat_failed", "candidate file could not be stat'ed", relative, detail=str(exc))
    if stat.S_ISLNK(mode):
        return None, None
    if not stat.S_ISREG(mode):
        return None, VerifierIssue("error", "special_file_not_allowed", "candidate package must not contain special files", relative)

    max_file_bytes = int(cfg.get("max_file_bytes", 262144))
    try:
        if os.name == "posix":
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(path, flags)
            try:
                data = os.read(fd, max_file_bytes + 1)
            finally:
                os.close(fd)
        else:
            with path.open("rb") as handle:
                data = handle.read(max_file_bytes + 1)
    except OSError as exc:
        return None, VerifierIssue("error", "candidate_file_read_failed", "candidate file could not be read safely", relative, detail=str(exc))
    if len(data) > max_file_bytes:
        return None, VerifierIssue("error", "file_too_large", "candidate file exceeds max_file_bytes", relative, detail=len(data))
    try:
        return data.decode("utf-8"), None
    except UnicodeDecodeError as exc:
        return None, VerifierIssue("error", "candidate_text_decode_failed", "candidate text file is not valid UTF-8", relative, detail=str(exc))


def _status(issues: list[VerifierIssue]) -> str:
    return "fail" if any(issue.severity == "error" for issue in issues) else "pass"


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{name} config must be a mapping")
    return value
