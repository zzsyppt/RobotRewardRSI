#!/usr/bin/env python3
"""Replay a completed Push reference through the public-dev runner, with optional numeric capture.

All source paths are explicit inputs. This command never trains, qualifies a
reference, or runs holdouts. Run mode acquires the existing GPU lease before
environment inspection. Independent archive reduction runs separately on CPU.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import time
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from true_rl_agent.composition import TaskCompositionContext
from true_rl_agent.composition.cell_manifest import load_resolved_cell_manifest
from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.evaluation.phases import EvaluationPhase
from true_rl_agent.runner.checkpoint import file_sha256
from true_rl_agent.runner.job_argv import validate_eval_job_argv
from true_rl_agent.runner.gpu_lease import GpuLeaseManager, PreGenesisGpuLeaseHolder
from true_rl_agent.runner.resource_shadow import discover_nvidia_gpu_inventory
from true_rl_agent.runner.specs import CheckpointRef
from true_rl_agent.tasks.push_recovery.composition import PushRecoveryCompositionProvider
from true_rl_agent.tasks.push_recovery.config import PushRecoveryTaskConfig
from true_rl_agent.tasks.push_recovery.checkpoint_admission import validate_push_reference_evaluation_source


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--cell", required=True, choices=tuple(
        f"push.{robot}.recovery_{version}" for robot in ("g1_37dof", "go2_12dof") for version in ("v2", "v3")))
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-summary", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("preflight", "run"), required=True)
    parser.add_argument("--capture", choices=("on", "off"), required=True)
    parser.add_argument("--isaac-python", default=sys.executable)
    parser.add_argument("--eval-base-seed", type=int, default=1500000000)
    parser.add_argument("--timeout-s", type=int, default=1800)
    parser.add_argument("--numeric-evidence-max-bytes", type=int, default=2 * 1024**3)
    return parser


def prepare(args: argparse.Namespace):
    trained = json.loads(args.checkpoint_summary.read_text(encoding="utf-8"))
    if trained.get("status") != "pass" or trained.get("subject_kind") != "reference":
        raise ValueError("Push replay requires a completed reference training receipt")
    old = load_resolved_cell_manifest(args.source_manifest.resolve(), expected_manifest_hash=trained["resolved_cell_manifest_hash"])
    source_config = PushRecoveryTaskConfig.from_manifest(old)
    provider = PushRecoveryCompositionProvider()
    values = {**provider.cli_defaults(), "cell_variant_id": args.cell,
        "execution_device": args.device, "isaac_python": args.isaac_python, "runner_timeout_s": args.timeout_s,
        "push_protocol": source_config.push_protocol.to_dict(), "reference_program_id": trained["subject_id"],
        "eval_base_seed": args.eval_base_seed}
    if args.cell.endswith(".recovery_v2"):
        values["reference_admission_policy"] = None
    composition = provider.resolve(values=values,
        context=TaskCompositionContext(project_root=PROJECT_ROOT, output_dir=args.output / "composition"))
    backend, config = composition.backend_adapter, composition.task_plugin.config
    before, after = old.to_dict(), composition.resolved_cell_manifest.to_dict()
    for field in ("robot_profile", "runtime_adapter"):
        if before[field] != after[field]:
            raise ValueError("Push evaluation changed the trained " + field)
    for field in ("action_content_hash", "observation_content_hash", "policy_io_schema_hash", "agent_configuration_hash", "sensor_schema_hash"):
        if before["contract_hashes"][field] != after["contract_hashes"][field]:
            raise ValueError("Push evaluation changed the trained policy ABI: " + field)
    if before["robot_task_binding"]["task_projection"]["push_protocol"] != after["robot_task_binding"]["task_projection"]["push_protocol"]:
        raise ValueError("Push evaluation changed the source training protocol")
    reference = composition.reference_program_artifact
    source_binding_hash = validate_push_reference_evaluation_source(old, composition.resolved_cell_manifest, trained)
    if trained["training"]["budget_accounting"]["planned_env_steps"] != config.budget_steps:
        raise ValueError("Push replay training budget differs from the completed receipt")
    checksum = trained["checkpoint"]["checksum"]
    if args.mode == "run" and file_sha256(args.checkpoint) != checksum:
        raise ValueError("checkpoint bytes differ from the completed training receipt")
    environment_hash = None
    if args.mode == "run":
        environment_hash = backend.resolve_session_execution_environment(session_id="push-numeric-replay",
            session_dir=args.output, allow_create=True)["manifest_hash"]
    jobs = backend.materialize_reference_jobs(session_id="push-numeric-replay", task_spec=composition.task_plugin.task_spec,
        reference=reference, artifact_root=args.output / "reference_source", policy_initialization=composition.policy_initialization,
        evaluation_binding=backend.evaluation_binding_for(composition.task_plugin.task_spec),
        backend_execution_environment_manifest_hash=environment_hash)
    plan = replace(jobs.eval_spec, training_binding_hash=source_binding_hash,
        program_source_checksum=trained["program_source_checksum"], subject_manifest_hash=trained["subject_manifest_hash"],
        checkpoint=CheckpointRef(uri="artifact://push-numeric-replay/best.pt", checksum=checksum,
        role="eval", sealed=True, metadata={"local_path": str(args.checkpoint.resolve()),
            "resolved_cell_manifest_hash": old.manifest_hash, "source_resolved_manifest_path": str(args.source_manifest.resolve()),
            "source_training_summary_path": str(args.checkpoint_summary.resolve()),
            "source_training_summary_sha256": file_sha256(args.checkpoint_summary)}), checkpoint_output_plan=None)
    plan = backend.rematerialize_eval_plan(eval_plan=plan, round_id="push-numeric-replay",
        summary_path=(args.output / "evaluation/summary.json").as_posix())
    phase = backend.materialize_eval_phase(eval_plan=plan, phase=EvaluationPhase.PUBLIC_DEV,
        capture_numeric=args.capture == "on",
        numeric_evidence_dir=(args.output / "numeric").as_posix() if args.capture == "on" else None,
        numeric_evidence_max_bytes=args.numeric_evidence_max_bytes)
    validate_eval_job_argv(phase, phase.metadata["cli_argv"])
    composition.runner.eval_checkpoint_admission_hook(phase.execution_checkpoint(), composition.resolved_cell_manifest, tuple(phase.metadata["cli_argv"]))
    # Keep original source identity. A preparation cannot claim runner admission;
    # the existing execution-time checkpoint gate still decides it.
    report = {"source_training_manifest_hash": old.manifest_hash,
        "evaluation_manifest_hash": composition.resolved_cell_manifest.manifest_hash,
        "training_summary_sha256": file_sha256(args.checkpoint_summary), "source_manifest_sha256": file_sha256(args.source_manifest),
        "checkpoint_sha256": checksum, "checkpoint_bytes_checked": args.mode == "run",
        "preserved_training_binding_hash": plan.training_binding_hash, "same_robot_and_policy_abi": True,
        "same_training_protocol": True, "execution_environment_manifest_hash": environment_hash,
        "evaluation_plan_hash": phase.evaluation_plan_hash, "job_spec_hash": canonical_json_hash(phase.to_public_dict()),
        "public_dev_seeds": plan.public_seed_count, "eval_num_envs": plan.eval_num_envs,
        "checkpoint_runner_admission": "not_executed"}
    return composition, phase, report


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.numeric_evidence_max_bytes <= 0 or args.timeout_s <= 0:
        raise ValueError("numeric evidence byte limit and timeout must be positive")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": "true_rl_agent.push_numeric_replay.v1", "status": "RUNNING", "mode": args.mode,
        "capture": args.capture, "cell": args.cell, "started_at_unix": time.time(), "train_jobs": 0,
        "accept_jobs": 0, "sealed_jobs": 0, "provider_requests": 0, "qualification_claimed": False}
    def write(name, value):
        (args.output / name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    resources = ExitStack()
    lease_started = None
    try:
        if args.mode == "run":
            if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
                raise ValueError("use an unmasked host CUDA ordinal for the production runner")
            execution_identity = canonical_json_hash({"purpose": "Push numeric replay", "output": str(args.output),
                "source_train_summary_sha256": file_sha256(args.checkpoint_summary), "cell": args.cell, "capture": args.capture})
            holder = PreGenesisGpuLeaseHolder(session_id="push-numeric-replay", genesis_identity_hash=execution_identity,
                session_contract_hash=execution_identity, lane_id=args.device, invocation_segment_id=execution_identity[:16],
                logical_job_id="push-numeric-replay")
            manager = GpuLeaseManager(lease_root=Path("/tmp/true_rl_agent_gpu_leases"),
                devices={item.execution_device: item.identity for item in discover_nvidia_gpu_inventory()})
            lease = resources.enter_context(manager.acquire(args.device, holder=holder))
            lease_started = time.monotonic()
            report["gpu_lease"] = dict(lease.receipt)
        composition, phase, identity = prepare(args)
        report.update(identity)
        write("public_dev_job.json", phase.to_public_dict())
        if args.mode == "preflight":
            report["status"] = "PREPARED"
            return 0
        write("result.json", report)
        result = composition.runner.run_eval_phase(phase)
        write("public_dev_result.json", result.to_private_dict())
        raw = json.loads(Path(phase.metadata["summary_path"]).read_text(encoding="utf-8"))
        if file_sha256(args.checkpoint) != report["checkpoint_sha256"]:
            raise ValueError("checkpoint bytes changed during replay")
        if args.capture == "on":
            write("numeric_references.json", [seed["numeric_evidence_ref"] for seed in raw["task_evidence"]["seed_summaries"]])
            report["numeric_recompute_status"] = "pending_separate_cpu_step"
        report.update(status="PASS", checkpoint_runner_admission="passed",
            public_dev=composition.task_plugin.phased_evaluation_adapter().project_public_phase_summary(phase_result=result),
            evaluation_execution_cost=dict(result.execution_cost))
        return 0
    except Exception:
        report.update(status="FAIL", error=traceback.format_exc())
        traceback.print_exc()
        return 1
    finally:
        if lease_started is not None:
            report["gpu_capacity_seconds"] = time.monotonic() - lease_started
        resources.close()
        report["ended_at_unix"] = time.time()
        write("result.json", report)
        print(json.dumps({"status": report["status"], "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
