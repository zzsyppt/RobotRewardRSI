#!/usr/bin/env python3
from __future__ import annotations

import json
import platform
import sys

import torch


def main() -> int:
    result: dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "devices": [],
    }

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        print(json.dumps(result, indent=2))
        return 1

    devices: list[dict[str, object]] = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        with torch.cuda.device(index):
            left = torch.randn((2048, 2048), device="cuda")
            right = torch.randn((2048, 2048), device="cuda")
            output = left @ right
            torch.cuda.synchronize()
            finite = bool(torch.isfinite(output).all().item())
        devices.append(
            {
                "index": index,
                "name": props.name,
                "compute_capability": [props.major, props.minor],
                "total_memory_bytes": props.total_memory,
                "matmul_finite": finite,
            }
        )

    result["devices"] = devices
    print(json.dumps(result, indent=2))
    return 0 if all(bool(device["matmul_finite"]) for device in devices) else 1


if __name__ == "__main__":
    raise SystemExit(main())
