"""Run the frozen Stair formal generation session or read-only helpers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from true_rl_agent.tasks.stair_climbing.generation_session import (  # noqa: E402
    StairGenerationPilotConfig,
    preflight_stair_generation_pilot,
    preflight_stair_generation_provider_readiness,
    run_stair_generation_pilot,
)
from true_rl_agent.tasks.stair_climbing.generation_session_cost import (  # noqa: E402
    DEVELOPMENT_GENERATION_PILOT_MODE,
    FORMAL_GENERATION_SESSION_MODE,
    STAIR_FORMAL_GENERATION_PAUSE_BOUNDARIES,
)
from true_rl_agent.tasks.stair_climbing.final_confirmation import (  # noqa: E402
    STAIR_FINAL_CONFIRMATION_REQUIRED_GENERATIONS,
)
from true_rl_agent.tasks.stair_climbing.launch_context import (  # noqa: E402
    consume_stair_generation_launch_context,
)


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_stair_generation_session_args(raw_argv)
    consume_stair_generation_launch_context(
        process_argv=_process_argv(raw_argv, explicit=argv is not None),
        project_root=PROJECT_ROOT,
        formal_output_dir=args.output_dir,
        session_id=args.session_id,
        expected_source_commit=args.expected_source_commit,
        expected_machine_id=args.machine_id,
        expected_gpu_uuids=(args.gpu0_uuid, args.gpu1_uuid),
        lease_root=args.lease_root,
        max_output_bytes=args.max_output_bytes,
        nvidia_smi_executable=args.nvidia_smi_executable,
        helper_mode=(
            args.preflight_only
            or args.provider_readiness_only
            or (args.terminal_audit_only and not args.formal_session)
        ),
        resume_existing=args.resume_existing,
        formal_session=args.formal_session,
        terminal_audit_only=args.terminal_audit_only,
    )
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
        session_mode=(
            FORMAL_GENERATION_SESSION_MODE
            if args.formal_session
            else DEVELOPMENT_GENERATION_PILOT_MODE
        ),
        pause_after_generation=args.pause_after_generation,
        terminal_audit_only=args.terminal_audit_only,
    )
    if args.preflight_only:
        report = preflight_stair_generation_pilot(config)
    elif args.provider_readiness_only:
        report = preflight_stair_generation_provider_readiness(config)
    else:
        report = run_stair_generation_pilot(config)
    printable = (
        report
        if args.provider_readiness_only or args.terminal_audit_only
        else report["checks"]
    )
    print(json.dumps(printable, indent=2, sort_keys=True))
    return 0


def parse_stair_generation_session_args(
    argv: list[str] | None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen dual-GPU P6-D formal session or one of its "
            "read-only helper modes."
        ),
        allow_abbrev=False,
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
        "--formal-session",
        action="store_true",
        help=(
            f"Run the fixed {STAIR_FINAL_CONFIRMATION_REQUIRED_GENERATIONS}-generation "
            "formal P6-D session; generation count "
            "and lifecycle boundaries are not caller-configurable."
        ),
    )
    parser.add_argument(
        "--pause-after-generation",
        type=int,
        choices=STAIR_FORMAL_GENERATION_PAUSE_BOUNDARIES,
        help="Pause only at a boundary precommitted by the selected session mode.",
    )
    parser.add_argument(
        "--terminal-audit-only",
        action="store_true",
        help=(
            "Offline-finalize a missing report for an already terminal formal session, "
            "or strictly audit its existing report, without new provider/GPU work."
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--resume-existing",
        action="store_true",
        help="Resume the exact existing session/output instead of creating one.",
    )
    mode.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Verify source, GPU, credential, and persistence contracts without "
            "creating the formal output or starting any session work."
        ),
    )
    mode.add_argument(
        "--provider-readiness-only",
        action="store_true",
        help=(
            "Check the fixed provider account without creating formal output, "
            "starting GPU work, or sending a generation request."
        ),
    )
    args = parser.parse_args(argv)
    helper_mode = args.preflight_only or args.provider_readiness_only
    if not args.formal_session and not helper_mode:
        parser.error("--formal-session is required for a runnable session")
    if args.pause_after_generation is not None:
        if not args.formal_session:
            parser.error(
                "--pause-after-generation requires a multi-generation session"
            )
        if helper_mode:
            parser.error(
                "--pause-after-generation cannot be combined with a helper mode"
            )
    if args.terminal_audit_only:
        if helper_mode:
            parser.error("--terminal-audit-only cannot be combined with a helper mode")
        if not args.formal_session:
            parser.error(
                "--terminal-audit-only requires a multi-generation session"
            )
        if not args.resume_existing:
            parser.error("--terminal-audit-only requires --resume-existing")
        if args.pause_after_generation is not None:
            parser.error(
                "--terminal-audit-only cannot be combined with a generation pause"
            )
    return args


def _process_argv(raw_argv: list[str], *, explicit: bool) -> list[str]:
    if not explicit:
        original = getattr(sys, "orig_argv", None)
        if isinstance(original, list) and original:
            return [str(value) for value in original]
    return [
        str(Path(sys.executable).resolve()),
        "-u",
        str(Path(__file__).resolve()),
        *raw_argv,
    ]


if __name__ == "__main__":
    raise SystemExit(main())
