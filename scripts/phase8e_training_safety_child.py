#!/usr/bin/env python3
"""Phase 8E-1 isolated training-process safety probe.

This is not a formal trainer.  It is a small child process that applies the
same OS-level boundary intended for future true code-level training, then loads
candidate code and records whether the process can read hidden data, write the
repo, see secrets, or reach the network.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from humanoid_agent.code_sandbox.isolation import (  # noqa: E402
    IsolationError,
    apply_landlock,
    default_training_landlock_read_paths,
    default_training_landlock_read_write_paths,
)
from humanoid_agent.code_sandbox.loader import load_training_components  # noqa: E402


_ORIGINAL_JSON_DUMPS = json.dumps
_ORIGINAL_OS_CLOSE = os.close
_ORIGINAL_OS_EXIT = os._exit
_ORIGINAL_OS_MAKEDIRS = os.makedirs
_ORIGINAL_OS_OPEN = os.open
_ORIGINAL_OS_REPLACE = os.replace
_ORIGINAL_OS_WRITE = os.write


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--summary-file", required=True)
    parser.add_argument("--repo-root", default=str(PROJECT_ROOT))
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--expected-schema-version", default="humanoid_agent.cold_start.candidate.v1")
    parser.add_argument("--hidden-probe-file", default=None)
    parser.add_argument("--landlock", action="store_true")
    parser.add_argument("--allow-cuda-device-paths", action="store_true")
    parser.add_argument("--cuda-probe", action="store_true")
    parser.add_argument("--cuda-device", default="cuda:0")
    return parser


def _summary(status: str, **kwargs: Any) -> dict[str, Any]:
    return {
        "schema_version": "humanoid_agent.phase8e.training_safety_child.summary.v1",
        "status": status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        **kwargs,
    }


def _attempt(fn: Callable[[], Any]) -> dict[str, Any]:
    try:
        value = fn()
    except Exception as exc:  # noqa: BLE001 - probe should classify all failures.
        return {"status": "blocked", "error_type": type(exc).__name__}
    return {"status": "unexpected_success", "value": str(value)[:200]}


def _secret_env_present() -> list[str]:
    return [
        key
        for key in sorted(os.environ)
        if any(token in key.upper() for token in ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL"))
    ]


def _network_probe() -> dict[str, Any]:
    def connect() -> str:
        with socket.create_connection(("203.0.113.1", 9), timeout=1.0):
            return "connected"

    return _attempt(connect)


def _cuda_probe(device: str) -> dict[str, Any]:
    try:
        import torch  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001 - probe must report import failures.
        return {"status": "fail", "stage": "import_torch", "error_type": type(exc).__name__, "message": str(exc)}

    try:
        available = bool(torch.cuda.is_available())
        device_count = int(torch.cuda.device_count())
    except Exception as exc:  # noqa: BLE001 - CUDA query itself can fail under isolation.
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
    except Exception as exc:  # noqa: BLE001 - tensor allocation is the real training boundary smoke.
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


def main() -> int:
    args = _build_parser().parse_args()
    candidate_dir = Path(args.candidate_dir).resolve()
    summary_file = Path(args.summary_file).resolve()
    repo_root = Path(args.repo_root).resolve()
    run_dir = Path(args.run_dir).resolve()
    hidden_probe_file = Path(args.hidden_probe_file).resolve() if args.hidden_probe_file else None
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    isolation: dict[str, Any] = {
        "landlock_requested": bool(args.landlock),
        "landlock": {"enabled": False},
        "cwd": os.getcwd(),
    }
    try:
        if args.landlock:
            isolation["landlock"] = apply_landlock(
                read_paths=default_training_landlock_read_paths(
                    repo_root=repo_root,
                    candidate_dir=candidate_dir,
                    allow_cuda_device_paths=bool(args.allow_cuda_device_paths),
                ),
                read_write_paths=default_training_landlock_read_write_paths(run_dir=run_dir),
            )

        manifest, components = load_training_components(
            candidate_dir,
            expected_schema_version=args.expected_schema_version,
            allow_llm_authored=True,
            task_spec={"phase": "8E-1-training-safety-probe"},
            base_env_cfg=None,
            base_agent_cfg=None,
        )
        candidate_output_write = _attempt(lambda: (Path.cwd() / "candidate_output_probe.txt").write_text("ok", encoding="utf-8"))
        repo_write_target = repo_root / "phase8e_training_safety_repo_write_probe.tmp"
        repo_write = _attempt(lambda: repo_write_target.write_text("bad", encoding="utf-8"))
        hidden_read = (
            _attempt(lambda: hidden_probe_file.read_text(encoding="utf-8"))
            if hidden_probe_file is not None
            else {"status": "not_configured"}
        )
        repo_hidden_config_read = _attempt(lambda: (repo_root / "configs" / "phase2_baseline.yaml").read_text(encoding="utf-8"))
        network_connect = _network_probe()
        cuda_training_smoke = _cuda_probe(args.cuda_device) if args.cuda_probe else {"status": "not_requested"}
        secret_keys = _secret_env_present()
        checks = {
            "secret_env_absent": not secret_keys,
            "hidden_file_read_blocked": hidden_read["status"] == "blocked" if hidden_probe_file is not None else True,
            "repo_hidden_config_read_blocked": repo_hidden_config_read["status"] == "blocked",
            "repo_write_blocked": repo_write["status"] == "blocked",
            "candidate_output_write_allowed": candidate_output_write["status"] == "unexpected_success",
            "network_connect_blocked": network_connect["status"] == "blocked",
        }
        if args.cuda_probe:
            checks["cuda_training_smoke_passed"] = cuda_training_smoke["status"] == "pass"
        status = "pass" if all(checks.values()) else "reject"
        summary = _summary(
            status,
            process_role="isolated_training_process",
            isolation=isolation,
            candidate=manifest.to_public_summary(),
            components=components.to_public_summary(),
            probes={
                "secret_keys": secret_keys,
                "hidden_read": hidden_read,
                "repo_hidden_config_read": repo_hidden_config_read,
                "repo_write": repo_write,
                "candidate_output_write": candidate_output_write,
                "network_connect": network_connect,
                "cuda_training_smoke": cuda_training_smoke,
            },
            checks=checks,
            public_training_artifacts={
                "checkpoint_contract": "checkpoint_weights_only",
                "public_log_contract": "public_training_log_only",
                "candidate_output_dir": str(run_dir),
            },
            errors=[],
        )
        exit_code = 0 if status == "pass" else 1
    except Exception as exc:  # noqa: BLE001 - child must return structured rejection.
        summary = _summary(
            "fail" if isinstance(exc, IsolationError) else "reject",
            process_role="isolated_training_process",
            isolation=isolation,
            candidate={"path": str(candidate_dir)},
            errors=[{"type": type(exc).__name__, "message": str(exc)}],
        )
        exit_code = 1

    _write_authoritative_summary(summary_file, summary)
    _write_status_line(summary_file, str(summary["status"]))
    return exit_code


if __name__ == "__main__":
    _ORIGINAL_OS_EXIT(main())
