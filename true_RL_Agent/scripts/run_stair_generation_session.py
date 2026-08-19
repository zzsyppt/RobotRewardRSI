"""Run the fixed P6-B Stair development-generation pilot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from true_rl_agent.tasks.stair_climbing.generation_session import (
    StairGenerationPilotConfig,
    run_stair_generation_pilot,
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = StairGenerationPilotConfig(
        project_root=PROJECT_ROOT,
        output_dir=args.output_dir.resolve(),
        lease_root=args.lease_root.resolve(),
        session_id=args.session_id,
        expected_source_commit=args.expected_source_commit,
        machine_id=args.machine_id,
        gpu0_uuid=args.gpu0_uuid,
        gpu1_uuid=args.gpu1_uuid,
        provider_env_file=args.provider_env_file.resolve(),
        max_allocated_active_gpu_hours=args.max_allocated_active_gpu_hours,
        max_simulated_control_steps=args.max_simulated_control_steps,
        max_output_bytes=args.max_output_bytes,
        nvidia_smi_executable=args.nvidia_smi_executable,
        resume_existing=args.resume_existing,
    )
    report = run_stair_generation_pilot(config)
    print(json.dumps(report["checks"], indent=2, sort_keys=True))
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one fresh, fixed-provider, dual-GPU Stair generation pilot."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--machine-id", required=True)
    parser.add_argument("--gpu0-uuid", required=True)
    parser.add_argument("--gpu1-uuid", required=True)
    parser.add_argument("--provider-env-file", type=Path, required=True)
    parser.add_argument(
        "--lease-root",
        type=Path,
        default=Path("/tmp/true_rl_agent_gpu_leases"),
    )
    parser.add_argument(
        "--max-allocated-active-gpu-hours",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--max-simulated-control-steps",
        type=int,
        required=True,
    )
    parser.add_argument("--max-output-bytes", type=int, required=True)
    parser.add_argument("--nvidia-smi-executable", default="nvidia-smi")
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="Resume the exact existing session/output instead of creating one.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
