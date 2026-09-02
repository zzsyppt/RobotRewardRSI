#!/usr/bin/env python3
# ruff: noqa: E402
"""Collect one immutable launch-resource readiness receipt for Stair."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from true_rl_agent.json_contract import canonical_json_bytes
from true_rl_agent.tasks.stair_climbing.launch_readiness import (
    DEFAULT_SAMPLE_DURATION_SECONDS,
    DEFAULT_SAMPLE_INTERVAL_SECONDS,
    StairGenerationLaunchReadinessConfig,
    StairGenerationLaunchReadinessConfigV3,
    STAIR_FORMAL_LAUNCH_SUBJECT_MODES,
    collect_stair_generation_launch_readiness_v3,
)


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = _parse_args(raw_argv)
    base_config = StairGenerationLaunchReadinessConfig(
        receipt_path=args.receipt_path,
        formal_output_dir=args.formal_output_dir,
        session_id=args.session_id,
        source_root=PROJECT_ROOT,
        expected_source_commit=args.expected_source_commit,
        expected_machine_id=args.expected_machine_id,
        expected_gpu_uuids=(args.gpu0_uuid, args.gpu1_uuid),
        lease_root=args.lease_root,
        max_output_bytes=args.max_output_bytes,
        nvidia_smi_executable=args.nvidia_smi_executable,
        producer_argv=_producer_argv(raw_argv, explicit=argv is not None),
        sample_duration_seconds=args.sample_duration_seconds,
        sample_interval_seconds=args.sample_interval_seconds,
        machine_id_path=args.machine_id_path,
        boot_id_path=args.boot_id_path,
        proc_meminfo_path=args.proc_meminfo_path,
    )
    receipt = collect_stair_generation_launch_readiness_v3(
        StairGenerationLaunchReadinessConfigV3(
            base=base_config,
            subject_mode=args.subject_mode,
            provider_env_file=args.provider_env_file,
            formal_generation_command_sha256=(
                args.formal_generation_command_sha256
            ),
            attempt_chain_fault_probe_report=(
                args.attempt_chain_fault_probe_report
            ),
            stair_train_teardown_result=args.stair_train_teardown_result,
        )
    )
    sys.stdout.buffer.write(
        canonical_json_bytes(
            receipt.to_ref(base_config.receipt_path).to_dict(), trailing_newline=True
        )
    )
    return 0


def _producer_argv(raw_argv: list[str], *, explicit: bool) -> tuple[str, ...]:
    if not explicit:
        original = getattr(sys, "orig_argv", None)
        if isinstance(original, list) and original:
            return tuple(str(value) for value in original)
    return (
        str(Path(sys.executable).resolve()),
        str(Path(__file__).resolve()),
        *raw_argv,
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt-path", type=Path, required=True)
    parser.add_argument("--formal-output-dir", type=Path, required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-machine-id", required=True)
    parser.add_argument("--gpu0-uuid", required=True)
    parser.add_argument("--gpu1-uuid", required=True)
    parser.add_argument("--lease-root", type=Path, required=True)
    parser.add_argument("--max-output-bytes", type=int, required=True)
    parser.add_argument("--nvidia-smi-executable", required=True)
    parser.add_argument(
        "--subject-mode",
        choices=tuple(sorted(STAIR_FORMAL_LAUNCH_SUBJECT_MODES)),
        required=True,
        help="emit the matching P6-D formal v3 subject contract",
    )
    parser.add_argument(
        "--provider-env-file",
        type=Path,
        required=True,
        help="credential boundary used by the v3 local-only credential check",
    )
    parser.add_argument(
        "--formal-generation-command-sha256",
        required=True,
        help="sha256 binding of the exact tracked launcher child command",
    )
    parser.add_argument(
        "--attempt-chain-fault-probe-report",
        type=Path,
        required=True,
        help="production-shaped Linux attempt-chain probe report.json",
    )
    parser.add_argument(
        "--stair-train-teardown-result",
        type=Path,
        required=True,
        help="short real Stair train teardown result.json",
    )
    parser.add_argument(
        "--sample-duration-seconds",
        type=float,
        default=DEFAULT_SAMPLE_DURATION_SECONDS,
    )
    parser.add_argument(
        "--sample-interval-seconds",
        type=float,
        default=DEFAULT_SAMPLE_INTERVAL_SECONDS,
    )
    parser.add_argument("--machine-id-path", type=Path, default=Path("/etc/machine-id"))
    parser.add_argument(
        "--boot-id-path",
        type=Path,
        default=Path("/proc/sys/kernel/random/boot_id"),
    )
    parser.add_argument("--proc-meminfo-path", type=Path, default=Path("/proc/meminfo"))
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
