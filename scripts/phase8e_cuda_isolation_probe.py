#!/usr/bin/env python3
"""Probe whether CUDA works inside the candidate-training isolation boundary."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from humanoid_agent.code_sandbox.isolation import (  # noqa: E402
    apply_landlock,
    default_training_landlock_read_paths,
    default_training_landlock_read_write_paths,
    probe_isolation_capabilities,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["bare", "landlock"], default="bare")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--candidate-dir", default="examples/cold_start_candidates/noop_candidate")
    parser.add_argument("--run-dir", default="/tmp/phase8e_cuda_isolation_probe")
    parser.add_argument("--allow-cuda-device-paths", action="store_true")
    return parser


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


def main() -> int:
    args = _build_parser().parse_args()
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    capabilities = probe_isolation_capabilities()
    landlock_summary: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    try:
        if args.mode == "landlock":
            landlock_summary = apply_landlock(
                read_paths=default_training_landlock_read_paths(
                    repo_root=PROJECT_ROOT,
                    candidate_dir=(PROJECT_ROOT / args.candidate_dir).resolve(),
                    allow_cuda_device_paths=args.allow_cuda_device_paths,
                ),
                read_write_paths=default_training_landlock_read_write_paths(run_dir=run_dir),
            )
    except Exception as exc:  # noqa: BLE001
        error = {"stage": "apply_landlock", "error_type": type(exc).__name__, "message": str(exc)}

    cuda_summary = {"status": "not_run"} if error else _cuda_probe(args.device)
    status = "pass" if error is None and cuda_summary.get("status") == "pass" else "fail"
    summary = {
        "schema_version": "humanoid_agent.phase8e.cuda_isolation_probe.v1",
        "status": status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "device": args.device,
        "capabilities": capabilities,
        "landlock": landlock_summary,
        "cuda": cuda_summary,
        "error": error,
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
