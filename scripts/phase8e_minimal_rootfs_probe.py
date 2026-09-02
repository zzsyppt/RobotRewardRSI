#!/usr/bin/env python3
"""Probe a namespace + minimal-rootfs GPU training boundary for Phase 8E.

Run this script inside an unprivileged user+mount+network namespace, e.g.:

    unshare -Urmn python scripts/phase8e_minimal_rootfs_probe.py ...

The script builds a tmpfs root, bind-mounts only the paths needed for CUDA and
candidate component loading, pivots into that root, detaches the old root, then
checks CUDA and visibility of protected paths.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


HOST_PROJECT_ROOT = Path(__file__).resolve().parents[1]

_PROTECTED_NUMBER_RE = re.compile(r"(?<![\d.])(?:1001|2001)(?![\d.])")
_PROTECTED_STRING_TOKENS = ("iid_hidden", "iid-hidden", "hidden_audit", "hidden_eval")
_ORIGINAL_JSON_DUMPS = json.dumps
_ORIGINAL_OS_CLOSE = os.close
_ORIGINAL_OS_EXIT = os._exit
_ORIGINAL_OS_MAKEDIRS = os.makedirs
_ORIGINAL_OS_OPEN = os.open
_ORIGINAL_OS_REPLACE = os.replace
_ORIGINAL_OS_WRITE = os.write


def _contains_protected_tokens(text: str) -> bool:
    lowered = text.lower()
    if any(token in lowered for token in _PROTECTED_STRING_TOKENS):
        return True
    return bool(_PROTECTED_NUMBER_RE.search(text))


def _sanitize_training_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Strip eval-only sections so candidates never see eval seeds/iid_hidden.

    Training needs only the training profile, reward, curriculum, and PPO config.
    The ``evaluation`` block (public/iid_hidden seeds) and eval-only push profiles
    are removed so the candidate-readable training config carries no protected
    eval tokens.  This keeps the true code-level path intact (the candidate still
    writes/runs training code) while closing the eval-seed leak.
    """

    config = copy.deepcopy(raw)
    config.pop("evaluation", None)
    profiles = config.get("profiles")
    if isinstance(profiles, dict):
        config["profiles"] = {
            name: body
            for name, body in profiles.items()
            if not any(token in name.lower() for token in ("public", "iid_hidden", "hidden"))
        }
    return config


def _write_sanitized_training_config(source: Path, target: Path) -> None:
    raw = _parse_yaml_text(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError(f"training config must be a mapping: {source}")
    # JSON is valid YAML, so the training script can still read this with
    # yaml.safe_load in the Isaac environment while local selftests avoid a
    # hard PyYAML dependency.
    text = json.dumps(_sanitize_training_config(raw), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if _contains_protected_tokens(text):
        raise RuntimeError(f"sanitized training config still contains protected eval tokens: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def _parse_yaml_text(text: str) -> Any:
    try:
        import yaml  # type: ignore[import-not-found]  # noqa: PLC0415

        return yaml.safe_load(text)
    except ModuleNotFoundError:
        return _parse_simple_yaml(text)


def _parse_simple_yaml(text: str) -> Any:
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
        if raw_value in {">", "|", ">-", "|-"}:
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
        elif char == "#" and not in_single and not in_double and (index == 0 or value[index - 1].isspace()):
            return value[:index].rstrip()
    return value


def _parse_simple_yaml_scalar(value: str) -> Any:
    if value in {"null", "Null", "NULL", "~"}:
        return None
    if value in {"true", "True", "TRUE"}:
        return True
    if value in {"false", "False", "FALSE"}:
        return False
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


def _scan_runner_inputs() -> dict[str, Any]:
    directory = Path("/runner_inputs")
    if not directory.is_dir():
        return {"present": False, "all_clean": True, "files": {}}
    files: dict[str, Any] = {}
    all_clean = True
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        clean = not _contains_protected_tokens(text)
        files[path.name] = {"clean": clean}
        if not clean:
            all_clean = False
    return {"present": True, "all_clean": all_clean, "files": files}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--summary-file", default="minimal_rootfs_summary.json")
    parser.add_argument("--hidden-probe-file", default=None)
    parser.add_argument("--repo-root", default=str(HOST_PROJECT_ROOT))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--expected-schema-version", default="humanoid_agent.cold_start.candidate.v1")
    parser.add_argument("--train-start-smoke", action="store_true")
    parser.add_argument("--train-script", default="scripts/phase1_push_recovery_train_smoke.py")
    parser.add_argument("--phase1-config", default="configs/phase1_push_recovery.yaml")
    parser.add_argument("--phase2-config", default="configs/phase2_baseline.yaml")
    parser.add_argument("--task", default="Unitree-G1-29dof-Velocity")
    parser.add_argument("--profile", default="phase2_train")
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--max-iterations", type=int, default=1)
    parser.add_argument("--seed", type=int, default=801)
    parser.add_argument("--train-timeout-seconds", type=int, default=1800)
    return parser


def _run(command: list[str]) -> None:
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


def _target(rootfs: Path, absolute_target: str | Path) -> Path:
    target = Path(absolute_target)
    if not target.is_absolute():
        raise RuntimeError(f"target must be absolute: {target}")
    return rootfs / str(target).lstrip("/")


def _ensure_mountpoint(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    if source.is_dir():
        target.mkdir(parents=True, exist_ok=True)
    else:
        target.touch(exist_ok=True)


def _bind(source: Path, target: Path, *, readonly: bool, recursive: bool = True) -> None:
    source = source.resolve()
    _ensure_mountpoint(source, target)
    _run(["mount", "--rbind" if recursive else "--bind", str(source), str(target)])
    if readonly:
        _run(["mount", "-o", "remount,bind,ro", str(target)])


def _mount_proc(target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    _run(["mount", "-t", "proc", "proc", str(target)])


def _mount_tmpfs(target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    _run(["mount", "-t", "tmpfs", "tmpfs", str(target)])


def _attempt(fn: Callable[[], Any]) -> dict[str, Any]:
    try:
        value = fn()
    except Exception as exc:  # noqa: BLE001 - probe should classify all failures.
        return {"status": "blocked", "error_type": type(exc).__name__}
    return {"status": "unexpected_success", "value": str(value)[:200]}


def _network_probe() -> dict[str, Any]:
    def connect() -> str:
        with socket.create_connection(("203.0.113.1", 9), timeout=1.0):
            return "connected"

    return _attempt(connect)


def _secret_env_present() -> list[str]:
    return [
        key
        for key in sorted(os.environ)
        if any(token in key.upper() for token in ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL"))
    ]


def _cuda_probe(device: str) -> dict[str, Any]:
    try:
        import torch  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        return {"status": "fail", "stage": "import_torch", "error_type": type(exc).__name__, "message": str(exc)}
    try:
        available = bool(torch.cuda.is_available())
        device_count = int(torch.cuda.device_count())
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "fail",
            "stage": "cuda_query",
            "torch_version": getattr(torch, "__version__", None),
            "torch_cuda_runtime": getattr(torch.version, "cuda", None),
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
    if not available or device_count < 1:
        return {
            "status": "fail",
            "stage": "cuda_available",
            "torch_version": getattr(torch, "__version__", None),
            "torch_cuda_runtime": getattr(torch.version, "cuda", None),
            "cuda_available": available,
            "device_count": device_count,
        }
    try:
        tensor = torch.ones((1,), device=device)
        value = float((tensor + 1).detach().cpu().item())
        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "fail",
            "stage": "cuda_tensor",
            "torch_version": getattr(torch, "__version__", None),
            "torch_cuda_runtime": getattr(torch.version, "cuda", None),
            "cuda_available": available,
            "device_count": device_count,
            "device": device,
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
    return {
        "status": "pass",
        "stage": "cuda_tensor",
        "torch_version": getattr(torch, "__version__", None),
        "torch_cuda_runtime": getattr(torch.version, "cuda", None),
        "cuda_available": available,
        "device_count": device_count,
        "device": device,
        "tensor_value": value,
    }


def _prepare_rootfs(
    *,
    rootfs: Path,
    candidate_dir: Path,
    run_dir: Path,
    host_repo_root: Path,
    train_script: Path,
    phase1_config: Path,
    phase2_config: Path,
) -> None:
    _run(["mount", "--make-rprivate", "/"])
    _mount_tmpfs(rootfs)

    # System/runtime mounts.
    for path in (Path("/usr"), Path("/bin"), Path("/lib"), Path("/lib64")):
        if path.exists():
            _bind(path, _target(rootfs, path), readonly=True)
    if Path("/etc/ld.so.cache").exists():
        _bind(Path("/etc/ld.so.cache"), _target(rootfs, "/etc/ld.so.cache"), readonly=True, recursive=False)
    if Path("/sys").exists():
        _bind(Path("/sys"), _target(rootfs, "/sys"), readonly=True)
    _bind(Path("/dev"), _target(rootfs, "/dev"), readonly=False)
    if (_target(rootfs, "/dev/shm")).exists():
        _mount_tmpfs(_target(rootfs, "/dev/shm"))
    _mount_proc(_target(rootfs, "/proc"))
    _mount_tmpfs(_target(rootfs, "/tmp"))

    # Read-only parent for /data/yuyangliu, so candidates cannot recreate hidden
    # siblings such as /data/yuyangliu/AGENT or /data/yuyangliu/secrets.
    data_anchor = run_dir / "anchors" / "data_yuyangliu"
    envs_root = Path(sys.prefix).resolve()
    vendor_root = Path("/data/yuyangliu/vendor")
    try:
        envs_mount_relative = envs_root.relative_to("/data/yuyangliu")
    except ValueError:
        envs_mount_relative = Path("envs") / envs_root.name
    (data_anchor / envs_mount_relative).mkdir(parents=True, exist_ok=True)
    if vendor_root.exists():
        (data_anchor / "vendor").mkdir(parents=True, exist_ok=True)
    _bind(data_anchor, _target(rootfs, "/data/yuyangliu"), readonly=True)
    _bind(envs_root, _target(rootfs, envs_root), readonly=True)
    if vendor_root.exists():
        _bind(vendor_root, _target(rootfs, vendor_root), readonly=True)

    # Candidate-facing API and package.
    repo_anchor = run_dir / "anchors" / "repo"
    (repo_anchor / "src").mkdir(parents=True, exist_ok=True)
    _bind(repo_anchor, _target(rootfs, "/repo"), readonly=True)
    _bind(host_repo_root / "src", _target(rootfs, "/repo/src"), readonly=True)
    runner_scripts_anchor = run_dir / "anchors" / "runner_scripts"
    runner_inputs_anchor = run_dir / "anchors" / "runner_inputs"
    runner_scripts_anchor.mkdir(parents=True, exist_ok=True)
    runner_inputs_anchor.mkdir(parents=True, exist_ok=True)
    (runner_scripts_anchor / "train_start_smoke.py").touch(exist_ok=True)
    # Bind TRAINING-ONLY sanitized configs: binding the raw configs would re-open
    # the D1 leak, because phase2_baseline.yaml carries the protected eval seeds
    # (1001/2001) and iid_hidden definitions.  Training does not need them.
    _write_sanitized_training_config(phase1_config, runner_inputs_anchor / "phase1_push_recovery.yaml")
    _write_sanitized_training_config(phase2_config, runner_inputs_anchor / "phase2_baseline.yaml")
    _bind(runner_scripts_anchor, _target(rootfs, "/runner_scripts"), readonly=True)
    _bind(runner_inputs_anchor, _target(rootfs, "/runner_inputs"), readonly=True)
    _bind(train_script, _target(rootfs, "/runner_scripts/train_start_smoke.py"), readonly=True, recursive=False)
    _bind(candidate_dir, _target(rootfs, "/candidate"), readonly=True)
    output_dir = run_dir / "candidate_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    _bind(output_dir, _target(rootfs, "/candidate_output"), readonly=False)
    runner_artifacts_dir = run_dir / "runner_artifacts"
    runner_artifacts_dir.mkdir(parents=True, exist_ok=True)
    _bind(runner_artifacts_dir, _target(rootfs, "/runner_artifacts"), readonly=False)


def _pivot_into(rootfs: Path) -> None:
    oldroot = rootfs / "oldroot"
    oldroot.mkdir(parents=True, exist_ok=True)
    os.chdir(rootfs)
    _run(["pivot_root", ".", "oldroot"])
    os.chdir("/")
    _run(["umount", "-l", "/oldroot"])
    try:
        Path("/oldroot").rmdir()
    except OSError:
        pass


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _write_authoritative_summary(path: Path, summary: dict[str, Any]) -> None:
    _ORIGINAL_OS_MAKEDIRS(str(path.parent), exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = (_ORIGINAL_JSON_DUMPS(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    fd = _ORIGINAL_OS_OPEN(str(temp_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = _ORIGINAL_OS_WRITE(fd, view)
            view = view[written:]
    finally:
        _ORIGINAL_OS_CLOSE(fd)
    _ORIGINAL_OS_REPLACE(str(temp_path), str(path))


def _write_status_line(summary_file: Path, status: str) -> None:
    line = _ORIGINAL_JSON_DUMPS({"summary_file": str(summary_file), "status": status}, ensure_ascii=False) + "\n"
    _ORIGINAL_OS_WRITE(1, line.encode("utf-8"))


def _run_train_start_smoke(args: argparse.Namespace) -> dict[str, Any]:
    train_run_dir = Path("/candidate_output/train_start_smoke_public")
    summary_file = Path("/runner_artifacts/train_start_smoke_summary.json")
    log_file = Path("/runner_artifacts/train_start_smoke.log")
    runner_artifacts_dir = Path("/runner_artifacts/train_start_smoke")
    train_run_dir.mkdir(parents=True, exist_ok=True)
    runner_artifacts_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "/runner_scripts/train_start_smoke.py",
        "--headless",
        "--device",
        args.device,
        "--task",
        args.task,
        "--seed",
        str(args.seed),
        "--profile",
        args.profile,
        "--num-envs",
        str(args.num_envs),
        "--max-iterations",
        str(args.max_iterations),
        "--config",
        "/runner_inputs/phase1_push_recovery.yaml",
        "--phase2-config",
        "/runner_inputs/phase2_baseline.yaml",
        "--run-dir",
        str(train_run_dir),
        "--summary-file",
        str(summary_file),
        "--runner-artifacts-dir",
        str(runner_artifacts_dir),
        "--candidate-dir",
        "/candidate",
    ]
    started_at = datetime.now(timezone.utc)
    timed_out = False
    with log_file.open("w", encoding="utf-8") as log:
        try:
            completed = subprocess.run(
                command,
                cwd="/candidate_output",
                env=os.environ.copy(),
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=args.train_timeout_seconds,
                check=False,
            )
            exit_code = completed.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            exit_code = 124
    summary_load_error = None
    if summary_file.is_file():
        try:
            summary = _load_json(summary_file)
        except Exception as exc:  # noqa: BLE001 - malformed child summary is a candidate/recipe reject.
            summary = {}
            summary_load_error = {"type": type(exc).__name__, "message": str(exc)}
    else:
        summary = {}
    latest_checkpoint = summary.get("training", {}).get("latest_checkpoint") if isinstance(summary.get("training"), dict) else None
    source_latest_checkpoint = (
        summary.get("training", {}).get("source_latest_checkpoint") if isinstance(summary.get("training"), dict) else None
    )
    latest_checkpoint_sha256 = (
        summary.get("training", {}).get("latest_checkpoint_sha256") if isinstance(summary.get("training"), dict) else None
    )
    trusted_latest_checkpoint = _resolve_train_start_checkpoint(latest_checkpoint)
    checkpoint_present = trusted_latest_checkpoint is not None and trusted_latest_checkpoint.is_file()
    checkpoint_safety = _checkpoint_safety_report(trusted_latest_checkpoint) if checkpoint_present else None
    checkpoint_safe_load = bool(checkpoint_safety and checkpoint_safety.get("safe_load") is True)
    status = (
        "pass"
        if exit_code == 0 and summary.get("status") == "pass" and checkpoint_present and checkpoint_safe_load
        else "reject"
    )
    reject_reason = None
    if status != "pass":
        if timed_out:
            reject_reason = "train_start_timeout"
        elif exit_code != 0:
            reject_reason = "train_start_subprocess_failed"
        elif not summary_file.is_file():
            reject_reason = "train_start_missing_summary"
        elif summary_load_error is not None:
            reject_reason = "train_start_invalid_summary"
        elif latest_checkpoint and trusted_latest_checkpoint is None:
            reject_reason = "train_start_untrusted_checkpoint_path"
        elif not checkpoint_present:
            reject_reason = "train_start_missing_checkpoint"
        elif not checkpoint_safe_load:
            reject_reason = "train_start_checkpoint_failed_safe_load_gate"
        else:
            reject_reason = "train_start_summary_not_pass"
    return {
        "status": status,
        "reject_reason": reject_reason,
        "command": command,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "elapsed_seconds": (datetime.now(timezone.utc) - started_at).total_seconds(),
        "log_file": str(log_file),
        "summary_file": str(summary_file),
        "summary_load_error": summary_load_error,
        "runner_artifacts_dir": str(runner_artifacts_dir),
        "summary_status": summary.get("status"),
        "source_latest_checkpoint": source_latest_checkpoint,
        "latest_checkpoint": latest_checkpoint,
        "trusted_latest_checkpoint": str(trusted_latest_checkpoint) if trusted_latest_checkpoint is not None else None,
        "latest_checkpoint_sha256": latest_checkpoint_sha256,
        "checkpoint_present": checkpoint_present,
        "checkpoint_safe_load": checkpoint_safe_load,
        "checkpoint_safety": checkpoint_safety,
        "training_error": _public_training_error(summary.get("errors")),
        "num_envs": args.num_envs,
        "max_iterations": args.max_iterations,
    }


def _public_training_error(errors: Any) -> dict[str, Any] | None:
    """Sanitized first training exception so the next round can learn the cause.

    The inner train script records the real exception (e.g. which candidate
    reward callable raised); we surface a truncated type+message so the LLM can
    self-correct.  Eval seeds/secrets are scrubbed downstream by the public-only
    assertion before this reaches any prompt.
    """

    if not isinstance(errors, list) or not errors:
        return None
    first = errors[0]
    if not isinstance(first, dict):
        return None
    return {"type": str(first.get("type", "Error")), "message": str(first.get("message", ""))[:400]}


def _resolve_train_start_checkpoint(path: Any) -> Path | None:
    """Return the only train-start checkpoint path the parent may inspect."""

    if not isinstance(path, str) or not path:
        return None
    from humanoid_agent.immutable_runner.train_start_paths import resolve_train_start_checkpoint_path  # noqa: PLC0415

    return resolve_train_start_checkpoint_path(path, Path("/runner_artifacts"))


def _checkpoint_safety_report(checkpoint: Path) -> dict[str, Any]:
    """Return the trusted safe-load report before this process loads candidate code.

    ``phase1_push_recovery_train_smoke.py`` executes candidate callables, so its
    own checkpoint-safety field is advisory only.  The minimal-rootfs probe is
    the parent process of that train-start child; when called before component
    loading, this helper has not executed candidate code and can authoritatively
    reject checkpoints that would execute pickle payloads in trusted eval.
    """

    from humanoid_agent.immutable_runner.safe_checkpoint import checkpoint_safety_report  # noqa: PLC0415

    return checkpoint_safety_report(checkpoint)


def main() -> int:
    args = _build_parser().parse_args()
    candidate_dir = Path(args.candidate_dir).resolve()
    run_dir = Path(args.run_dir).resolve()
    host_repo_root = Path(args.repo_root).resolve()
    train_script = (host_repo_root / args.train_script).resolve()
    phase1_config = (host_repo_root / args.phase1_config).resolve()
    phase2_config = (host_repo_root / args.phase2_config).resolve()
    hidden_probe_file = Path(args.hidden_probe_file).resolve() if args.hidden_probe_file else None
    run_dir.mkdir(parents=True, exist_ok=True)
    rootfs = run_dir / "rootfs"

    try:
        _prepare_rootfs(
            rootfs=rootfs,
            candidate_dir=candidate_dir,
            run_dir=run_dir,
            host_repo_root=host_repo_root,
            train_script=train_script,
            phase1_config=phase1_config,
            phase2_config=phase2_config,
        )
        _pivot_into(rootfs)
        os.environ["PYTHONPATH"] = "/repo/src"
        os.environ["HOME"] = "/candidate_output/home"
        os.environ["XDG_CACHE_HOME"] = "/candidate_output/home/.cache"
        os.environ["XDG_CONFIG_HOME"] = "/candidate_output/home/.config"
        os.environ["XDG_STATE_HOME"] = "/candidate_output/home/.local/state"
        os.environ["PIP_CACHE_DIR"] = "/candidate_output/home/.cache/pip"
        os.environ["CONDA_PKGS_DIRS"] = "/candidate_output/home/.cache/conda"
        system_lib_dir = Path("/data/yuyangliu/vendor/system-libs/root/usr/lib/x86_64-linux-gnu")
        if system_lib_dir.is_dir():
            current_ld = os.environ.get("LD_LIBRARY_PATH", "")
            os.environ["LD_LIBRARY_PATH"] = str(system_lib_dir) + (os.pathsep + current_ld if current_ld else "")
        unitree_ros_dir = Path("/data/yuyangliu/vendor/unitree_ros")
        if unitree_ros_dir.is_dir():
            os.environ["UNITREE_ROS_DIR"] = str(unitree_ros_dir)
        Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
        Path(os.environ["XDG_CONFIG_HOME"]).mkdir(parents=True, exist_ok=True)
        Path(os.environ["XDG_STATE_HOME"]).mkdir(parents=True, exist_ok=True)
        if "/repo/src" not in sys.path:
            sys.path.insert(0, "/repo/src")

        repo_config = host_repo_root / "configs" / "phase2_baseline.yaml"
        repo_write_target = host_repo_root / "phase8e_minimal_rootfs_repo_write_probe.tmp"
        hidden_read = (
            _attempt(lambda: hidden_probe_file.read_text(encoding="utf-8"))
            if hidden_probe_file is not None
            else {"status": "not_configured"}
        )
        repo_config_read = _attempt(lambda: repo_config.read_text(encoding="utf-8"))
        repo_write = _attempt(lambda: repo_write_target.write_text("bad", encoding="utf-8"))
        candidate_output_write = _attempt(lambda: Path("/candidate_output/probe.txt").write_text("ok", encoding="utf-8"))
        network_connect = _network_probe()
        cuda = _cuda_probe(args.device)
        train_start = _run_train_start_smoke(args) if args.train_start_smoke else {"status": "not_requested"}
        runner_inputs_scan = _scan_runner_inputs()
        secret_keys = _secret_env_present()

        # Load the candidate package only after the train-start child has exited
        # and the parent-side checkpoint safe-load gate has run.  Candidate code
        # is arbitrary Python; executing it before the gate would let it monkeypatch
        # the parent probe process and forge checkpoint-safety helpers.
        from humanoid_agent.code_sandbox.loader import load_training_components  # noqa: PLC0415

        manifest, components = load_training_components(
            "/candidate",
            expected_schema_version=args.expected_schema_version,
            allow_llm_authored=True,
            task_spec={"phase": "8E-minimal-rootfs-probe"},
            base_env_cfg=None,
            base_agent_cfg=None,
        )
        checks = {
            "secret_env_absent": not secret_keys,
            "hidden_file_read_blocked": hidden_read["status"] == "blocked" if hidden_probe_file is not None else True,
            "repo_config_read_blocked": repo_config_read["status"] == "blocked",
            "repo_hidden_config_read_blocked": repo_config_read["status"] == "blocked",
            "repo_write_blocked": repo_write["status"] == "blocked",
            "candidate_output_write_allowed": candidate_output_write["status"] == "unexpected_success",
            "network_connect_blocked": network_connect["status"] == "blocked",
            "runner_inputs_no_protected_tokens": runner_inputs_scan["all_clean"],
            "cuda_training_smoke_passed": cuda["status"] == "pass",
            "train_start_smoke_passed": train_start["status"] == "pass" if args.train_start_smoke else True,
            "train_start_checkpoint_present": bool(train_start.get("checkpoint_present")) if args.train_start_smoke else True,
            "train_start_checkpoint_safe_load": bool(train_start.get("checkpoint_safe_load")) if args.train_start_smoke else True,
            "oldroot_detached": not Path("/oldroot").exists(),
        }
        base_checks = {key: value for key, value in checks.items() if not key.startswith("train_start_")}
        if not all(base_checks.values()):
            status = "fail"
        elif args.train_start_smoke and train_start["status"] != "pass":
            status = "reject"
        else:
            status = "pass"
        summary = {
            "schema_version": "humanoid_agent.phase8e.minimal_rootfs_probe.v1",
            "status": status,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "process_role": "isolated_training_process",
            "candidate": manifest.to_public_summary(),
            "components": components.to_public_summary(),
            "checks": checks,
            "probes": {
                "secret_keys": secret_keys,
                "hidden_read": hidden_read,
                "repo_config_read": repo_config_read,
                "repo_write": repo_write,
                "candidate_output_write": candidate_output_write,
                "network_connect": network_connect,
                "cuda": cuda,
                "train_start": train_start,
                "runner_inputs": runner_inputs_scan,
            },
            "mount_policy": {
                "mode": "namespace_minimal_rootfs_pivot_root",
                "repo_src_bound_readonly": True,
                "candidate_bound_readonly": True,
                "candidate_output_bound_readwrite": True,
                "runner_artifacts_bound_readwrite": True,
                "train_start_authoritative_artifacts_in_runner_artifacts": True,
                "runner_inputs_training_only_sanitized": True,
                "host_repo_root_bound": False,
                "host_secrets_bound": False,
                "vendor_bound_readonly": Path("/data/yuyangliu/vendor").exists(),
                "landlock_used": False,
            },
        }
        exit_code = 0 if status == "pass" else 1
    except Exception as exc:  # noqa: BLE001
        summary = {
            "schema_version": "humanoid_agent.phase8e.minimal_rootfs_probe.v1",
            "status": "fail",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
        exit_code = 1

    output = Path("/candidate_output") / args.summary_file if Path("/candidate_output").exists() else run_dir / args.summary_file
    _write_authoritative_summary(output, summary)
    _write_status_line(output, str(summary["status"]))
    return exit_code


if __name__ == "__main__":
    _ORIGINAL_OS_EXIT(main())
