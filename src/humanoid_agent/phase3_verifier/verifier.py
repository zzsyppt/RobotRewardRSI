"""Phase 3 candidate-patch verifier.

The verifier is intentionally config-driven. Future Agent loops should be able
to add allowed config knobs without rewriting the verifier, while protected
evaluation contracts remain locked by data-key checks.
"""

from __future__ import annotations

import fnmatch
import importlib
import json
import math
import os
import re
import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from humanoid_agent.phase1_push_recovery.config import stable_json_dumps


MISSING = object()
NUMERIC_STRING_RE = re.compile(r"^[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?$")


@dataclass(frozen=True)
class Issue:
    level: str
    code: str
    message: str
    path: str | None = None
    key_path: str | None = None
    expected: Any = None
    actual: Any = None

    def to_dict(self) -> dict[str, Any]:
        data = {
            "level": self.level,
            "code": self.code,
            "message": self.message,
        }
        if self.path is not None:
            data["path"] = self.path
        if self.key_path is not None:
            data["key_path"] = self.key_path
        if self.expected is not None:
            data["expected"] = _json_safe(self.expected)
        if self.actual is not None:
            data["actual"] = _json_safe(self.actual)
        return data


def load_verifier_config(path: str | Path) -> dict[str, Any]:
    data = _load_structured_file(Path(path))
    if not isinstance(data, dict):
        raise ValueError(f"Verifier config must be a mapping: {path}")
    return data


def load_verifier_config_from_git(repo_root: str | Path, base_ref: str, config_path: str | Path) -> dict[str, Any]:
    """Load verifier policy from a trusted Git base ref.

    Candidate patches are allowed to edit files under ``configs/`` in general,
    so loading the verifier policy from the worktree would let a malicious or
    mistaken patch delete the verifier's own guards before verification. Static
    and full verification therefore read this file from ``base_ref``.
    """

    repo_root = Path(repo_root)
    normalized = _normalize_path(config_path)
    blob = subprocess.run(
        ["git", "show", f"{base_ref}:{normalized}"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if blob.returncode != 0:
        raise RuntimeError(
            blob.stderr.strip()
            or f"Could not load trusted verifier config from {base_ref}:{normalized}"
        )
    data = _parse_structured_text(normalized, blob.stdout)
    if not isinstance(data, dict):
        raise ValueError(f"Trusted verifier config must be a mapping: {base_ref}:{normalized}")
    return data


def run_static_verification(
    repo_root: str | Path,
    settings: dict[str, Any],
    *,
    base_ref: str = "HEAD",
    include_untracked: bool = True,
    check_imports: bool = True,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    changed_files = list_changed_files(repo_root, base_ref=base_ref, include_untracked=include_untracked)
    structured_paths = sorted(_structured_paths(repo_root, settings, changed_files))
    current_docs, parse_issues = _load_current_documents(repo_root, structured_paths)
    base_docs = _load_base_documents(repo_root, base_ref, structured_paths)
    result = verify_static_documents(
        base_docs=base_docs,
        current_docs=current_docs,
        changed_files=changed_files,
        settings=settings,
        repo_root=repo_root,
        parse_issues=parse_issues,
        check_imports=check_imports,
    )
    result["base_ref"] = base_ref
    result["changed_files"] = changed_files
    result["structured_files"] = structured_paths
    return result


def verify_static_documents(
    *,
    base_docs: dict[str, Any],
    current_docs: dict[str, Any],
    changed_files: list[str],
    settings: dict[str, Any],
    repo_root: str | Path = ".",
    parse_issues: list[Issue] | None = None,
    check_imports: bool = False,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    issues: list[Issue] = list(parse_issues or [])
    checks: list[dict[str, Any]] = []

    changed_issues = _verify_changed_files(changed_files, settings)
    issues.extend(changed_issues)
    checks.append({"name": "diff_policy", "status": _status_from_issues(changed_issues)})

    protected_issues = _verify_protected_keys(base_docs, current_docs, settings)
    issues.extend(protected_issues)
    checks.append({"name": "protected_yaml_keys", "status": _status_from_issues(protected_issues)})

    finite_issues = _verify_finite_documents(current_docs, settings)
    issues.extend(finite_issues)
    checks.append({"name": "finite_numeric_config", "status": _status_from_issues(finite_issues)})

    semantic_issues = _verify_phase2_semantics(current_docs, settings)
    issues.extend(semantic_issues)
    checks.append({"name": "phase2_semantics", "status": _status_from_issues(semantic_issues)})

    import_issues: list[Issue] = []
    if check_imports:
        import_issues = _verify_python_imports(repo_root, settings)
        issues.extend(import_issues)
    checks.append(
        {
            "name": "python_imports",
            "status": _status_from_issues(import_issues) if check_imports else "skipped",
        }
    )

    return _build_result("static", checks, issues)


def run_smoke_verification(
    repo_root: str | Path,
    settings: dict[str, Any],
    *,
    run_dir: str | Path,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    smoke_cfg = settings.get("smoke", {})
    summary_file = run_dir / "phase3_random_action_smoke_summary.json"
    log_file = run_dir / "phase3_random_action_smoke.log"

    command = [
        sys.executable,
        str(repo_root / smoke_cfg.get("script", "scripts/phase1_push_recovery_eval.py")),
        "--headless",
        "--device",
        str(smoke_cfg.get("device", "cuda:0")),
        "--config",
        str(smoke_cfg.get("config", "configs/phase1_push_recovery.yaml")),
        "--profile",
        str(smoke_cfg.get("profile", "no_push")),
        "--policy",
        str(smoke_cfg.get("policy", "random")),
        "--seed",
        str(smoke_cfg.get("seed", 303)),
        "--num-envs",
        str(smoke_cfg.get("num_envs", 16)),
        "--max-steps",
        str(smoke_cfg.get("max_steps", 128)),
        "--summary-file",
        str(summary_file),
    ]

    env = os.environ.copy()
    src_path = str(repo_root / "src")
    env["PYTHONPATH"] = src_path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    timeout_s = int(smoke_cfg.get("timeout_seconds", 600))
    with log_file.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            cwd=repo_root,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_s,
            check=False,
        )

    issues: list[Issue] = []
    checks: list[dict[str, Any]] = []
    if completed.returncode != 0:
        issues.append(
            Issue(
                "error",
                "smoke_command_failed",
                f"Smoke command exited with code {completed.returncode}",
                path=str(log_file),
            )
        )
    checks.append({"name": "smoke_command", "status": "pass" if completed.returncode == 0 else "fail"})

    summary: dict[str, Any] | None = None
    if not summary_file.exists():
        issues.append(Issue("error", "smoke_summary_missing", "Smoke summary file was not created", path=str(summary_file)))
    else:
        try:
            with summary_file.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if not isinstance(loaded, dict):
                raise ValueError("summary root is not a JSON object")
            summary = loaded
        except Exception as exc:  # noqa: BLE001
            issues.append(Issue("error", "smoke_summary_invalid_json", str(exc), path=str(summary_file)))

    if summary is not None:
        issues.extend(_verify_smoke_summary(summary, settings, summary_file))
    checks.append({"name": "smoke_summary_contract", "status": _status_from_issues(issues)})

    result = _build_result("smoke", checks, issues)
    result["run_dir"] = str(run_dir)
    result["smoke_summary_file"] = str(summary_file)
    result["smoke_log_file"] = str(log_file)
    result["smoke_command"] = command
    return result


def list_changed_files(repo_root: Path, *, base_ref: str, include_untracked: bool) -> list[str]:
    diff_cmd = ["git", "diff", "--name-only", "--diff-filter=ACDMRTUXB", base_ref, "--"]
    diff = subprocess.run(diff_cmd, cwd=repo_root, text=True, capture_output=True, check=False)
    if diff.returncode != 0:
        raise RuntimeError(diff.stderr.strip() or f"git diff failed for {base_ref}")
    files = {_normalize_path(line) for line in diff.stdout.splitlines() if line.strip()}

    if include_untracked:
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        if untracked.returncode != 0:
            raise RuntimeError(untracked.stderr.strip() or "git ls-files failed")
        files.update(_normalize_path(line) for line in untracked.stdout.splitlines() if line.strip())
    return sorted(files)


def _verify_changed_files(changed_files: list[str], settings: dict[str, Any]) -> list[Issue]:
    policy = settings.get("diff_policy", {})
    ignored = policy.get("ignored_globs", [])
    allowed = policy.get("allowed_globs", [])
    forbidden = policy.get("forbidden_globs", [])
    issues: list[Issue] = []
    for raw_path in changed_files:
        path = _normalize_path(raw_path)
        if _matches_any(path, ignored):
            continue
        if _matches_any(path, forbidden):
            issues.append(Issue("error", "forbidden_file_changed", "Candidate modified a protected file", path=path))
            continue
        if allowed and not _matches_any(path, allowed):
            issues.append(Issue("error", "file_not_allowed", "Candidate modified a path outside the allowed scope", path=path))
    return issues


def _verify_protected_keys(
    base_docs: dict[str, Any],
    current_docs: dict[str, Any],
    settings: dict[str, Any],
) -> list[Issue]:
    issues: list[Issue] = []
    for path, key_paths in settings.get("protected_keys", {}).items():
        base_doc = base_docs.get(path, MISSING)
        if base_doc is MISSING:
            continue
        if path not in current_docs:
            issues.append(
                Issue(
                    "error",
                    "protected_file_missing",
                    "Candidate deleted or hid a protected structured file",
                    path=path,
                )
            )
            continue
        current_doc = current_docs[path]
        for key_path in key_paths:
            expected = _get_key_path(base_doc, key_path)
            actual = _get_key_path(current_doc, key_path)
            if not _values_equal(expected, actual):
                issues.append(
                    Issue(
                        "error",
                        "protected_key_changed",
                        "Candidate changed a protected evaluation/budget key",
                        path=path,
                        key_path=key_path,
                        expected=expected,
                        actual=actual,
                    )
                )
    return issues


def _verify_finite_documents(current_docs: dict[str, Any], settings: dict[str, Any]) -> list[Issue]:
    patterns = settings.get("finite_numeric_globs", [])
    issues: list[Issue] = []
    for path, doc in current_docs.items():
        if not _matches_any(path, patterns):
            continue
        for key_path, value in _iter_nonfinite(doc):
            issues.append(
                Issue(
                    "error",
                    "non_finite_numeric",
                    "Config contains NaN or Infinity",
                    path=path,
                    key_path=key_path,
                    actual=value,
                )
            )
        for key_path, value in _iter_numeric_strings(doc):
            issues.append(
                Issue(
                    "error",
                    "numeric_string_config_value",
                    "Config numeric value was parsed as a string; use a YAML numeric scalar instead",
                    path=path,
                    key_path=key_path,
                    expected="numeric scalar",
                    actual=value,
                )
            )
    return issues


def _verify_phase2_semantics(current_docs: dict[str, Any], settings: dict[str, Any]) -> list[Issue]:
    doc = current_docs.get("configs/phase2_baseline.yaml")
    if not isinstance(doc, dict):
        return []
    issues: list[Issue] = []
    training = doc.get("training", {})
    evaluation = doc.get("evaluation", {})
    if int(training.get("num_envs", 0)) <= 0:
        issues.append(Issue("error", "invalid_training_num_envs", "training.num_envs must be positive", "configs/phase2_baseline.yaml", "training.num_envs"))
    if int(training.get("max_iterations", 0)) <= 0:
        issues.append(Issue("error", "invalid_training_iterations", "training.max_iterations must be positive", "configs/phase2_baseline.yaml", "training.max_iterations"))
    if not isinstance(training.get("seeds"), list) or not training["seeds"]:
        issues.append(Issue("error", "invalid_training_seeds", "training.seeds must be a non-empty list", "configs/phase2_baseline.yaml", "training.seeds"))
    target_range = evaluation.get("target_success_rate_range")
    if (
        not isinstance(target_range, list)
        or len(target_range) != 2
        or not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in target_range)
        or not (0.0 <= float(target_range[0]) <= float(target_range[1]) <= 1.0)
    ):
        issues.append(
            Issue(
                "error",
                "invalid_target_success_range",
                "evaluation.target_success_rate_range must be an ordered [low, high] in [0, 1]",
                "configs/phase2_baseline.yaml",
                "evaluation.target_success_rate_range",
            )
        )

    allowed_ppo = settings.get("phase2_semantics", {}).get("allowed_ppo_keys", {})
    for name, value in doc.get("ppo", {}).items():
        rule = allowed_ppo.get(name)
        if rule is None:
            issues.append(Issue("error", "ppo_key_not_allowed", "PPO key is outside the verifier allowlist", "configs/phase2_baseline.yaml", f"ppo.{name}", actual=value))
            continue
        issues.extend(_verify_ppo_value(name, value, rule))
    return issues


def _verify_ppo_value(name: str, value: Any, rule: dict[str, Any]) -> list[Issue]:
    if value is None:
        return []
    key_path = f"ppo.{name}"
    value_type = rule.get("type")
    if value_type == "optional_bool":
        if not isinstance(value, bool):
            return [Issue("error", "invalid_ppo_value", "PPO field must be null or bool", "configs/phase2_baseline.yaml", key_path, actual=value)]
        return []
    if value_type == "positive_int":
        if isinstance(value, bool) or not isinstance(value, int):
            return [Issue("error", "invalid_ppo_value", "PPO field must be an integer", "configs/phase2_baseline.yaml", key_path, actual=value)]
        min_value = int(rule.get("min", 1))
        max_value = int(rule.get("max", 2**31 - 1))
        if not (min_value <= value <= max_value):
            return [Issue("error", "invalid_ppo_value", f"PPO integer must be in [{min_value}, {max_value}]", "configs/phase2_baseline.yaml", key_path, actual=value)]
        return []
    if value_type == "finite_number":
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            return [Issue("error", "invalid_ppo_value", "PPO field must be a finite number", "configs/phase2_baseline.yaml", key_path, actual=value)]
        min_value = float(rule.get("min", float("-inf")))
        max_value = float(rule.get("max", float("inf")))
        if not (min_value <= float(value) <= max_value):
            return [Issue("error", "invalid_ppo_value", f"PPO number must be in [{min_value}, {max_value}]", "configs/phase2_baseline.yaml", key_path, actual=value)]
        return []
    return [Issue("error", "invalid_verifier_config", f"Unknown PPO verifier rule type: {value_type}", "configs/phase3_verifier.yaml", key_path)]


def _verify_python_imports(repo_root: Path, settings: dict[str, Any]) -> list[Issue]:
    issues: list[Issue] = []
    src_path = str(repo_root / "src")
    added = False
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
        added = True
    try:
        for module_name in settings.get("python_imports", []):
            try:
                importlib.import_module(module_name)
            except Exception as exc:  # noqa: BLE001
                issues.append(Issue("error", "python_import_failed", f"Could not import {module_name}: {exc}", key_path=module_name))
    finally:
        if added:
            try:
                sys.path.remove(src_path)
            except ValueError:
                pass
    return issues


def _verify_smoke_summary(summary: dict[str, Any], settings: dict[str, Any], summary_file: Path) -> list[Issue]:
    smoke_cfg = settings.get("smoke", {})
    issues: list[Issue] = []
    if summary.get("status") != "pass":
        issues.append(Issue("error", "smoke_summary_status_failed", "Smoke summary status is not pass", path=str(summary_file), actual=summary.get("status")))
    evidence = summary.get("evidence", {})
    if not bool(evidence.get("reset_ok", False)):
        issues.append(Issue("error", "smoke_reset_failed", "Environment reset did not succeed", path=str(summary_file), key_path="evidence.reset_ok"))
    if bool(smoke_cfg.get("require_reward_finite", True)) and not bool(evidence.get("reward_finite", False)):
        issues.append(Issue("error", "smoke_reward_not_finite", "Reward was not finite during smoke rollout", path=str(summary_file), key_path="evidence.reward_finite"))
    if bool(smoke_cfg.get("require_observation_finite", True)) and not bool(evidence.get("observation_finite", False)):
        issues.append(Issue("error", "smoke_observation_not_finite", "Observation was not finite during smoke rollout", path=str(summary_file), key_path="evidence.observation_finite"))
    env = summary.get("environment", {})
    expected_steps = int(smoke_cfg.get("max_steps", 0))
    if expected_steps and int(env.get("max_steps", -1)) != expected_steps:
        issues.append(Issue("error", "smoke_wrong_step_budget", "Smoke ran with unexpected max_steps", path=str(summary_file), key_path="environment.max_steps", expected=expected_steps, actual=env.get("max_steps")))
    expected_envs = int(smoke_cfg.get("num_envs", 0))
    aggregate = summary.get("aggregate", {})
    if expected_envs and int(aggregate.get("episodes", -1)) != expected_envs:
        issues.append(Issue("error", "smoke_wrong_env_count", "Smoke ran with unexpected num_envs", path=str(summary_file), key_path="aggregate.episodes", expected=expected_envs, actual=aggregate.get("episodes")))
    mean_steps = summary.get("secondary_metrics", {}).get("mean_episode_steps")
    min_mean_steps = float(smoke_cfg.get("min_mean_episode_steps", 0))
    if mean_steps is None or float(mean_steps) < min_mean_steps:
        issues.append(Issue("error", "smoke_early_mass_termination", "Random-action smoke ended too early", path=str(summary_file), key_path="secondary_metrics.mean_episode_steps", expected=f">={min_mean_steps}", actual=mean_steps))
    return issues


def _structured_paths(repo_root: Path, settings: dict[str, Any], changed_files: list[str]) -> set[str]:
    paths: set[str] = set()
    structured = settings.get("structured_files", {})
    for pattern in structured.get("yaml_globs", []) + structured.get("json_globs", []):
        paths.update(_normalize_path(path.relative_to(repo_root)) for path in repo_root.glob(pattern) if path.is_file())
    paths.update(_normalize_path(path) for path in settings.get("protected_keys", {}))
    for path in changed_files:
        if Path(path).suffix.lower() in {".yaml", ".yml", ".json"}:
            paths.add(_normalize_path(path))
    return paths


def _load_current_documents(repo_root: Path, paths: list[str]) -> tuple[dict[str, Any], list[Issue]]:
    docs: dict[str, Any] = {}
    issues: list[Issue] = []
    for path in paths:
        full_path = repo_root / path
        if not full_path.exists():
            continue
        try:
            docs[path] = _load_structured_file(full_path)
        except Exception as exc:  # noqa: BLE001
            issues.append(Issue("error", "structured_parse_failed", str(exc), path=path))
    return docs, issues


def _load_base_documents(repo_root: Path, base_ref: str, paths: list[str]) -> dict[str, Any]:
    docs: dict[str, Any] = {}
    for path in paths:
        blob = subprocess.run(
            ["git", "show", f"{base_ref}:{path}"],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        if blob.returncode != 0:
            continue
        try:
            docs[path] = _parse_structured_text(path, blob.stdout)
        except Exception:
            continue
    return docs


def _load_structured_file(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    return _parse_structured_text(str(path), text)


def _parse_structured_text(path: str, text: str) -> Any:
    suffix = Path(path).suffix.lower()
    if suffix in {".yaml", ".yml"}:
        return _parse_yaml_text(text)
    if suffix == ".json":
        return json.loads(text)
    raise ValueError(f"Unsupported structured file type: {path}")


def _parse_yaml_text(text: str) -> Any:
    try:
        import yaml

        return yaml.safe_load(text)
    except ModuleNotFoundError:
        return _parse_simple_yaml(text)


def _parse_simple_yaml(text: str) -> Any:
    """Parse the small YAML subset used by project configs.

    This fallback exists so static verifier tests can run in minimal local
    Python runtimes without PyYAML. The IsaacLab server environment still uses
    PyYAML. Supported subset: indentation-based mappings, inline scalar lists,
    block scalar strings, comments, null/bool/int/float/string scalars.
    """

    lines = _preprocess_simple_yaml(text)
    if not lines:
        return None
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any] | list[Any]]] = [(-1, root)]
    for index, (indent, content) in enumerate(lines):
        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not stack:
            raise ValueError(f"Invalid YAML indentation near line: {content}")
        parent = stack[-1][1]
        if content.startswith("- "):
            if not isinstance(parent, list):
                raise ValueError(f"Unexpected YAML list item outside a list: {content}")
            parent.append(_parse_simple_yaml_scalar(content[2:].strip()))
            continue
        if ":" not in content:
            raise ValueError(f"Unsupported YAML line: {content}")
        if not isinstance(parent, dict):
            raise ValueError(f"Unexpected YAML mapping entry inside a scalar list: {content}")
        key, raw_value = content.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if raw_value in {">", "|"}:
            parent[key] = ""
            continue
        if raw_value == "":
            child: dict[str, Any] | list[Any]
            child = [] if _next_child_is_list(lines, index, indent) else {}
            parent[key] = child
            stack.append((indent, child))
            continue
        parent[key] = _parse_simple_yaml_scalar(raw_value)
    return root


def _next_child_is_list(lines: list[tuple[int, str]], index: int, indent: int) -> bool:
    for child_indent, child_content in lines[index + 1 :]:
        if child_indent <= indent:
            return False
        return child_content.startswith("- ")
    return False


def _preprocess_simple_yaml(text: str) -> list[tuple[int, str]]:
    result: list[tuple[int, str]] = []
    skip_block_indent: int | None = None
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        stripped = raw.strip()
        if skip_block_indent is not None:
            if indent > skip_block_indent:
                continue
            skip_block_indent = None
        content = _strip_yaml_comment(stripped)
        if not content:
            continue
        result.append((indent, content))
        if content.endswith(": >") or content.endswith(": |") or content.endswith(": >-") or content.endswith(": |-"):
            skip_block_indent = indent
    return result


def _strip_yaml_comment(value: str) -> str:
    in_single = False
    in_double = False
    for index, char in enumerate(value):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            if index == 0 or value[index - 1].isspace():
                return value[:index].rstrip()
    return value


def _parse_simple_yaml_scalar(value: str) -> Any:
    if value in {"null", "Null", "NULL", "~"}:
        return None
    if value in {"true", "True", "TRUE"}:
        return True
    if value in {"false", "False", "FALSE"}:
        return False
    if value in {".nan", ".NaN", ".NAN"}:
        return float("nan")
    if value in {".inf", ".Inf", ".INF", "+.inf", "+.Inf", "+.INF"}:
        return float("inf")
    if value in {"-.inf", "-.Inf", "-.INF"}:
        return float("-inf")
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_simple_yaml_scalar(item.strip()) for item in inner.split(",")]
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        if not any(marker in value for marker in (".", "e", "E")):
            return int(value)
        return float(value)
    except ValueError:
        return value


def _get_key_path(doc: Any, key_path: str) -> Any:
    current = doc
    for part in key_path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return MISSING
    return current


def _values_equal(left: Any, right: Any) -> bool:
    return stable_json_dumps(_json_safe(left)) == stable_json_dumps(_json_safe(right))


def _iter_nonfinite(value: Any, prefix: str = "") -> list[tuple[str, float]]:
    issues: list[tuple[str, float]] = []
    if isinstance(value, bool) or value is None:
        return issues
    if isinstance(value, float):
        if not math.isfinite(value):
            issues.append((prefix or "<root>", value))
    elif isinstance(value, int):
        return issues
    elif isinstance(value, dict):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            issues.extend(_iter_nonfinite(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            child = f"{prefix}[{index}]" if prefix else f"[{index}]"
            issues.extend(_iter_nonfinite(item, child))
    return issues


def _iter_numeric_strings(value: Any, prefix: str = "") -> list[tuple[str, str]]:
    issues: list[tuple[str, str]] = []
    if isinstance(value, str):
        text = value.strip()
        if text == value and NUMERIC_STRING_RE.match(text):
            issues.append((prefix or "<root>", value))
    elif isinstance(value, dict):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            issues.extend(_iter_numeric_strings(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            child = f"{prefix}[{index}]" if prefix else f"[{index}]"
            issues.extend(_iter_numeric_strings(item, child))
    return issues


def _matches_any(path: str, patterns: list[str]) -> bool:
    path = _normalize_path(path)
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def _normalize_path(path: str | Path) -> str:
    return str(path).replace("\\", "/").strip("/")


def _json_safe(value: Any) -> Any:
    if value is MISSING:
        return "<missing>"
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return deepcopy(value)


def _status_from_issues(issues: list[Issue]) -> str:
    return "fail" if any(issue.level == "error" for issue in issues) else "pass"


def _build_result(mode: str, checks: list[dict[str, Any]], issues: list[Issue]) -> dict[str, Any]:
    status = _status_from_issues(issues)
    return {
        "schema_version": "humanoid_agent.phase3.verifier.summary.v1",
        "status": status,
        "mode": mode,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "issues": [issue.to_dict() for issue in issues],
    }
