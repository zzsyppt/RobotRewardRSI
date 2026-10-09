#!/usr/bin/env python3
"""Run one non-authoritative Stair P6-A resource-shadow training lane."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from true_rl_agent.backends.isaac_lab_execution_environment import (  # noqa: E402
    verify_current_isaac_lab_execution_environment,
)
from true_rl_agent.evaluation.binding import (  # noqa: E402
    require_sha256_digest,
)
from true_rl_agent.fs_contract import create_once_bytes  # noqa: E402
from true_rl_agent.json_contract import canonical_json_bytes  # noqa: E402
from true_rl_agent.runner.isaac_app_lifecycle import (  # noqa: E402
    close_isaac_app_on_success,
)
from true_rl_agent.runner.specs import require_execution_device  # noqa: E402
from true_rl_agent.tasks.stair_climbing.config import (  # noqa: E402
    STAIR_PROTOCOL_V2_RELATIVE_PATH,
    StairClimbingTaskConfig,
)
from true_rl_agent.tasks.stair_climbing.isaaclab_execution import (  # noqa: E402
    VerifiedStairExecutionSourceManifest,
    preflight_stair_execution_source_manifest,
)
from true_rl_agent.tasks.stair_climbing.resource_shadow import (  # noqa: E402
    STAIR_RESOURCE_SHADOW_ACTIVE_MARKER_SCHEMA_VERSION,
    STAIR_RESOURCE_SHADOW_SUMMARY_SCHEMA_VERSION,
    STAIR_RESOURCE_SHADOW_WORKLOAD,
    StairResourceShadowFixture,
    execute_stair_resource_shadow_training,
    resolve_stair_resource_shadow_fixture,
    stair_resource_shadow_kit_args,
    stair_resource_shadow_workload_contract,
    stair_resource_shadow_workload_hash,
)

@dataclass(frozen=True)
class _ShadowChildContext:
    plan: Any
    plan_path: Path
    output_dir: Path
    active_marker: Path
    checkpoint_path: Path
    summary_path: Path
    config: StairClimbingTaskConfig
    fixture: StairResourceShadowFixture
    source_manifest: VerifiedStairExecutionSourceManifest
    environment_verification: Mapping[str, str]
    expected_initial_policy_state_hash: str
    physical_execution_device: str
    process_execution_device: str
    physical_gpu_uuid: str
    renderer_active_gpu_ordinal: int


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    try:
        from isaaclab.app import AppLauncher
    except ModuleNotFoundError:
        parser.add_argument("--headless", action="store_true")
        parser.add_argument("--device", required=True)
    else:
        AppLauncher.add_app_launcher_args(parser)
        device_action = next(
            action for action in parser._actions if action.dest == "device"
        )
        device_action.required = True
        device_action.default = None
    # IsaacLab inspects the parser while registering its own optional flags.
    # Add this script's required bindings afterwards so that parser construction
    # remains side-effect free for programmatic callers and unit tests.
    for option in (
        "--plan",
        "--plan-hash",
        "--session-genesis-hash",
        "--execution-environment-manifest-hash",
        "--fixture-hash",
        "--workload-hash",
        "--resource-plan-hash",
        "--expected-initial-policy-state-hash",
        "--job-id",
        "--case-id",
        "--lane-id",
        "--physical-execution-device",
        "--output-dir",
        "--active-marker",
        "--checkpoint-path",
    ):
        parser.add_argument(option, required=True)
    parser.set_defaults(headless=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        context = _validate_invocation(args)
    except (TypeError, ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    from isaaclab.app import AppLauncher

    simulation_app = AppLauncher(
        args,
        multi_gpu=STAIR_RESOURCE_SHADOW_WORKLOAD.renderer_multi_gpu,
    ).app
    exit_code = _run_shadow_lane(args=args, context=context)
    return close_isaac_app_on_success(simulation_app, exit_code=exit_code)


def _validate_invocation(args: argparse.Namespace) -> _ShadowChildContext:
    from true_rl_agent.runner.resource_shadow import (
        load_resource_shadow_plan,
        require_resource_shadow_child_binding,
    )

    if os.name != "posix":
        raise OSError("Stair resource-shadow child requires Linux process groups")
    if args.headless is not True:
        raise ValueError("Stair resource-shadow training must be headless")
    if getattr(args, "kit_args", ""):
        raise ValueError("Stair resource-shadow Kit arguments are task-owned")
    args.device = require_execution_device(
        args.device,
        "Stair resource-shadow process execution device",
    )
    physical_execution_device = require_execution_device(
        args.physical_execution_device,
        "Stair resource-shadow physical execution device",
    )
    if args.device != STAIR_RESOURCE_SHADOW_WORKLOAD.process_execution_device:
        raise ValueError(
            "Stair resource-shadow process must use isolated logical cuda:0"
        )
    plan_path = _existing_absolute_file(args.plan, "plan")
    output_dir = _absolute_path(args.output_dir, "output_dir")
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError("Stair resource-shadow output_dir must be a directory")
    active_marker = _controlled_output_path(
        args.active_marker,
        output_dir=output_dir,
        field_name="active_marker",
    )
    checkpoint_path = _controlled_output_path(
        args.checkpoint_path,
        output_dir=output_dir,
        field_name="checkpoint_path",
    )
    summary_path = output_dir / "summary.json"
    owned_paths = (
        (active_marker, "active_marker"),
        (checkpoint_path, "checkpoint_path"),
        (summary_path, "summary_path"),
        (output_dir / "run", "run_dir"),
    )
    if len({path for path, _ in owned_paths}) != len(owned_paths):
        raise ValueError("Stair resource-shadow output paths must be distinct")
    for path, field_name in owned_paths:
        if path.exists():
            raise FileExistsError(f"Stair resource-shadow {field_name} must be fresh")

    plan_hash = require_sha256_digest(args.plan_hash, "resource-shadow plan hash")
    session_genesis_hash = require_sha256_digest(
        args.session_genesis_hash,
        "resource-shadow session Genesis hash",
    )
    environment_hash = require_sha256_digest(
        args.execution_environment_manifest_hash,
        "resource-shadow execution environment manifest hash",
    )
    fixture_hash = require_sha256_digest(
        args.fixture_hash,
        "resource-shadow fixture hash",
    )
    workload_hash = require_sha256_digest(
        args.workload_hash,
        "resource-shadow workload hash",
    )
    resource_plan_hash = require_sha256_digest(
        args.resource_plan_hash,
        "resource-shadow resource plan hash",
    )
    expected_initial_hash = require_sha256_digest(
        args.expected_initial_policy_state_hash,
        "resource-shadow expected initial policy state hash",
    )
    for value, field_name in (
        (args.job_id, "job_id"),
        (args.case_id, "case_id"),
        (args.lane_id, "lane_id"),
    ):
        _canonical_identifier(value, field_name)

    plan = load_resource_shadow_plan(plan_path)
    job = require_resource_shadow_child_binding(
        plan,
        plan_hash=plan_hash,
        session_genesis_hash=session_genesis_hash,
        execution_environment_manifest_hash=environment_hash,
        fixture_hash=fixture_hash,
        workload_hash=workload_hash,
        resource_plan_hash=resource_plan_hash,
        job_id=args.job_id,
        case_id=args.case_id,
        lane_id=args.lane_id,
        physical_execution_device=physical_execution_device,
        process_execution_device=args.device,
    )
    inventory = plan.inventory_for_execution_device(job.execution_device)
    visible_gpu = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_gpu != inventory.uuid:
        raise ValueError(
            "Stair resource-shadow CUDA visibility does not match the planned GPU UUID"
        )
    if os.environ.get("NVIDIA_VISIBLE_DEVICES"):
        raise ValueError(
            "Stair resource-shadow child requires NVIDIA_VISIBLE_DEVICES to be unset"
        )
    args.kit_args = stair_resource_shadow_kit_args(
        renderer_active_gpu_ordinal=inventory.ordinal
    )

    config = StairClimbingTaskConfig.defaults(
        PROJECT_ROOT / STAIR_PROTOCOL_V2_RELATIVE_PATH
    )
    fixture = resolve_stair_resource_shadow_fixture(
        config,
        expected_initial_policy_state_hash=expected_initial_hash,
        project_root=PROJECT_ROOT,
    )
    if fixture.fixture_hash != fixture_hash:
        raise ValueError("Stair resource-shadow fixture differs from the frozen plan")
    if stair_resource_shadow_workload_hash() != workload_hash:
        raise ValueError("Stair resource-shadow workload differs from the frozen plan")
    source_manifest = preflight_stair_execution_source_manifest(config)
    environment_verification = verify_current_isaac_lab_execution_environment(
        environment_hash
    )
    return _ShadowChildContext(
        plan=plan,
        plan_path=plan_path,
        output_dir=output_dir,
        active_marker=active_marker,
        checkpoint_path=checkpoint_path,
        summary_path=summary_path,
        config=config,
        fixture=fixture,
        source_manifest=source_manifest,
        environment_verification=environment_verification,
        expected_initial_policy_state_hash=expected_initial_hash,
        physical_execution_device=physical_execution_device,
        process_execution_device=args.device,
        physical_gpu_uuid=inventory.uuid,
        renderer_active_gpu_ordinal=inventory.ordinal,
    )


def _run_shadow_lane(
    *,
    args: argparse.Namespace,
    context: _ShadowChildContext,
) -> int:
    context.output_dir.mkdir(parents=True, exist_ok=True)
    active_start_ns = time.monotonic_ns()
    pid = os.getpid()
    process_group_id = os.getpgid(pid)
    active_marker = {
        "schema_version": STAIR_RESOURCE_SHADOW_ACTIVE_MARKER_SCHEMA_VERSION,
        "plan_hash": args.plan_hash,
        "job_id": args.job_id,
        "case_id": args.case_id,
        "lane_id": args.lane_id,
        "physical_execution_device": context.physical_execution_device,
        "process_execution_device": context.process_execution_device,
        "physical_gpu_uuid": context.physical_gpu_uuid,
        "renderer_active_gpu_ordinal": context.renderer_active_gpu_ordinal,
        "process_id": pid,
        "process_group_id": process_group_id,
        "active_start_monotonic_ns": active_start_ns,
        "acceptance_authority": False,
        "ledger_authority": False,
    }
    _create_once_json(context.active_marker, active_marker)

    result = execute_stair_resource_shadow_training(
        config=context.config,
        source_manifest_preflight=context.source_manifest,
        agent_configuration=context.config.training_agent_configuration(),
        reward_program=context.fixture.reward_program,
        device=args.device,
        run_dir=context.output_dir / "run",
        checkpoint_output=context.checkpoint_path,
        run_name=f"p6a_{args.case_id}_{args.lane_id}",
        expected_initial_policy_state_hash=(
            context.expected_initial_policy_state_hash
        ),
    )
    # The unique training body closes its env in a finally block before returning.
    active_end_ns = time.monotonic_ns()
    fields = result.to_summary_fields()
    training = fields.get("training")
    training_mapping = dict(training) if isinstance(training, Mapping) else {}
    summary = {
        **fields,
        "schema_version": STAIR_RESOURCE_SHADOW_SUMMARY_SCHEMA_VERSION,
        "plan_hash": args.plan_hash,
        "session_genesis_hash": args.session_genesis_hash,
        "execution_environment_manifest_hash": (
            args.execution_environment_manifest_hash
        ),
        "fixture_hash": args.fixture_hash,
        "workload_hash": args.workload_hash,
        "resource_plan_hash": args.resource_plan_hash,
        "job_id": args.job_id,
        "case_id": args.case_id,
        "lane_id": args.lane_id,
        "physical_execution_device": context.physical_execution_device,
        "process_execution_device": context.process_execution_device,
        "physical_gpu_uuid": context.physical_gpu_uuid,
        "renderer_active_gpu_ordinal": context.renderer_active_gpu_ordinal,
        "process_id": pid,
        "process_group_id": process_group_id,
        "active_start_monotonic_ns": active_start_ns,
        "active_end_monotonic_ns": active_end_ns,
        "active_duration_ns": active_end_ns - active_start_ns,
        "expected_initial_policy_state_hash": (
            context.expected_initial_policy_state_hash
        ),
        "initial_policy_state_hash": training_mapping.get(
            "initial_policy_state_hash"
        ),
        "initial_policy_state_match": training_mapping.get(
            "initial_policy_state_match"
        ),
        "workload": stair_resource_shadow_workload_contract(),
        "fixture": context.fixture.to_contract_dict(),
        "environment_source_preflight": (
            context.source_manifest.to_summary_dict()
        ),
        "execution_environment_verification": dict(
            context.environment_verification
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "acceptance_authority": False,
        "ledger_authority": False,
    }
    _create_once_json(context.summary_path, summary)
    print(
        json.dumps(
            {
                "summary_path": str(context.summary_path),
                "status": result.status,
            },
            sort_keys=True,
        )
    )
    return 0 if result.status == "pass" else 1


def _existing_absolute_file(value: str, field_name: str) -> Path:
    if not isinstance(value, str) or not value or "://" in value:
        raise ValueError(f"Stair resource-shadow {field_name} must be a local path")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"Stair resource-shadow {field_name} must be absolute")
    path.lstat()
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Stair resource-shadow {field_name} must be a regular file")
    return path.resolve(strict=True)


def _absolute_path(value: str, field_name: str) -> Path:
    if not isinstance(value, str) or not value or "://" in value:
        raise ValueError(f"Stair resource-shadow {field_name} must be a local path")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"Stair resource-shadow {field_name} must be absolute")
    return path.resolve()


def _controlled_output_path(
    value: str,
    *,
    output_dir: Path,
    field_name: str,
) -> Path:
    path = _absolute_path(value, field_name)
    try:
        relative = path.relative_to(output_dir)
    except ValueError as exc:
        raise ValueError(
            f"Stair resource-shadow {field_name} must stay under output_dir"
        ) from exc
    if not relative.parts:
        raise ValueError(f"Stair resource-shadow {field_name} cannot be output_dir")
    return path


def _canonical_identifier(value: Any, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 256
        or any(ord(character) < 0x20 for character in value)
        or "/" in value
        or "\\" in value
    ):
        raise ValueError(f"Stair resource-shadow {field_name} is invalid")
    return value


def _create_once_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = canonical_json_bytes(dict(payload), trailing_newline=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    create_once_bytes(path, encoded)


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
