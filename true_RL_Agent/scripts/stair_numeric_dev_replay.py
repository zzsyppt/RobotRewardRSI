#!/usr/bin/env python3
"""Replay a completed Stair checkpoint on the selected fixed public-dev cohort.

Inputs are read-only. Capture on/off uses the same production phase runner;
this command never trains, qualifies a reference, or executes acceptance/holdout.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from true_rl_agent.composition import TaskCompositionContext
from true_rl_agent.evaluation.binding import canonical_json_hash, require_sha256_checksum, require_sha256_digest
from true_rl_agent.evaluation.phases import EvaluationPhase
from true_rl_agent.evaluation.subject import ExperimentSubject
from true_rl_agent.fs_contract import read_bounded_regular_file
from true_rl_agent.json_contract import canonical_json_bytes, strict_json_object_loads
from true_rl_agent.runner.budget import build_budget_accounting
from true_rl_agent.runner.checkpoint import file_sha256
from true_rl_agent.runner.checkpoint_preflight import validate_checkpoint_input_artifact
from true_rl_agent.runner.job_argv import validate_eval_job_argv
from true_rl_agent.runner.job_attempts import PreGenesisExecutionScope
from true_rl_agent.runner.native_video import add_native_video_arguments, native_video_argv_from_args
from true_rl_agent.runner.gpu_lease import GpuLeaseManager, PreGenesisGpuLeaseHolder
from true_rl_agent.runner.resource_shadow import discover_nvidia_gpu_inventory
from true_rl_agent.runner.specs import CheckpointRef, GpuProcessBinding, require_eval_phase_result_matches_spec
from true_rl_agent.tasks.stair_climbing.composition import StairClimbingCompositionProvider
from true_rl_agent.tasks.stair_climbing.diagnostic_evaluation import (
    prepare_stair_public_dev_diagnostic, project_stair_public_dev_diagnostic,
)
from true_rl_agent.tasks.stair_climbing.row_catalog import stair_robot_row
from true_rl_agent.tasks.stair_climbing.trajectory_pipeline import LEGACY_PROFILE, PROFILES
from true_rl_agent.tasks.stair_climbing.unified_protocol import StairUnifiedProtocol


def _write_json(path: Path, value, *, create_once: bool = False) -> None:
    with path.open("xb" if create_once else "wb") as stream:
        stream.write(canonical_json_bytes(value, trailing_newline=True))


def _completed_source(summary: dict, *, composition, checkpoint_path: Path) -> CheckpointRef:
    """Validate retained training facts without relabelling their cell or author."""
    config = composition.task_plugin.config
    binding = config.formal_training_binding()
    if (summary.get("status") != "pass" or summary.get("formal_training_executed") is not True
            or summary.get("dry_run", False) is not False or summary.get("errors", [])):
        raise ValueError("source summary must declare completed formal training without errors")
    if summary.get("task_id") != composition.task_id:
        raise ValueError("source training belongs to another task")
    for field in ("training_binding_hash", "capability_profile_hash", "policy_architecture_hash",
                  "agent_configuration_hash", "optimizer_training_profile_hash",
                  "initialization_seed_schedule_hash"):
        expected = binding.binding_hash if field == "training_binding_hash" else getattr(binding, field)
        if summary.get(field) != expected:
            raise ValueError("source training binding mismatch for " + field)
    training = summary.get("training")
    if not isinstance(training, dict):
        raise ValueError("source training completion is missing")
    expected_budget = build_budget_accounting(
        budget_steps=config.budget_steps, num_envs=config.train_num_envs,
        num_steps_per_env=config.train_steps_per_iteration, max_iterations=config.train_max_iterations,
    )
    for field in ("budget_steps", "num_envs", "num_steps_per_env", "max_iterations"):
        if type(training.get(field)) is not int or training[field] != expected_budget[field]:
            raise ValueError("source training budget mismatch for " + field)
    if training.get("budget_accounting") != expected_budget:
        raise ValueError("source training budget accounting mismatch")
    if type(training.get("latest_checkpoint_iteration")) is not int or (
        training["latest_checkpoint_iteration"] != config.train_max_iterations - 1
    ):
        raise ValueError("source training did not reach the final checkpoint iteration")
    require_sha256_digest(training.get("initial_policy_state_hash"), "source initial policy state hash")
    if training.get("execution_contract") != config.training_execution_contract():
        raise ValueError("source training execution contract mismatch")
    source_hash = summary.get("resolved_cell_manifest_hash")
    source = next((m for m in composition.runner.eval_checkpoint_admission_hook.manifests
                   if m.manifest_hash == source_hash), None)
    if source is None or summary.get("resolved_cell_identity") != source.summary_identity():
        raise ValueError("source training manifest/identity is not a verified ancestor")
    source_preflight = summary.get("environment_source_preflight", {})
    if (source_preflight.get("status") != "pass"
            or source_preflight.get("task_protocol_hash") != source.resolved_cell_task_protocol_hash):
        raise ValueError("source training environment preflight mismatch")
    require_sha256_digest(source_preflight.get("source_manifest_hash"), "source execution source manifest hash")
    environment = summary.get("backend_execution_environment_verification", {})
    if (environment.get("status") != "pass" or environment.get("backend_id") != "isaac_lab_rsl_rl"
            or environment.get("actual_manifest_hash") != environment.get("expected_manifest_hash")):
        raise ValueError("source training backend environment verification mismatch")
    require_sha256_digest(environment.get("actual_manifest_hash"), "source backend environment manifest hash")
    original = summary.get("checkpoint", {})
    if original.get("role") not in {"train_output", "eval"}:
        raise ValueError("source checkpoint is not a completed train output")
    subject = ExperimentSubject(kind=summary.get("subject_kind"), subject_id=summary.get("subject_id"))
    if subject.kind not in {"candidate", "reference"}:
        raise ValueError("source must be a trained candidate or reference")
    if subject.kind == "reference" or summary.get("subject_manifest_hash") is not None:
        require_sha256_digest(summary.get("subject_manifest_hash"), "source subject manifest hash")
    require_sha256_checksum(summary.get("program_source_checksum"), "source program checksum")
    if not isinstance(summary.get("job_id"), str) or not summary["job_id"]:
        raise ValueError("source train job id is missing")
    checkpoint = CheckpointRef(
        uri=original.get("uri"), checksum=original.get("checksum"), role="eval", sealed=True,
        metadata={
            **dict(original.get("metadata", {})), "local_path": str(checkpoint_path),
            "resolved_cell_manifest_hash": source_hash,
            "resolved_cell_identity": dict(summary["resolved_cell_identity"]),
            "training_binding_hash": summary["training_binding_hash"],
            "training_subject": subject.to_public_dict(),
            "subject_manifest_hash": summary.get("subject_manifest_hash"),
            "program_source_checksum": summary.get("program_source_checksum"),
            "capability_profile_hash": summary["capability_profile_hash"],
            "source_train_job_id": summary.get("job_id"),
        },
    )
    latest_hash = checkpoint.metadata.get("source_latest_checkpoint_sha256")
    if latest_hash is not None and latest_hash != checkpoint.checksum:
        raise ValueError("source selected checkpoint differs from final training checkpoint")
    validate_checkpoint_input_artifact(checkpoint, local_path=checkpoint_path)
    return checkpoint


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-summary", "--checkpoint-summary", dest="train_summary", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", required=True, help="Host CUDA ordinal; do not pre-mask CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--isaac-python", default=sys.executable)
    parser.add_argument("--mode", choices=("preflight", "run"), required=True)
    parser.add_argument("--capture-numeric", choices=("on", "off"), required=True)
    parser.add_argument("--cohort", choices=("formal", "diagnostic"), default="formal")
    parser.add_argument("--diagnostic-cohort", default="v1")
    parser.add_argument("--numeric-evidence-max-bytes", type=int, default=2 * 1024**3)
    parser.add_argument("--timeout-s", type=int, default=36000)
    parser.add_argument("--behavior-feedback-profile", choices=PROFILES, default=LEGACY_PROFILE,
                        help="Automatic per-archive CPU trajectory cards for supported Stair Go2 numeric inputs.")
    parser.add_argument("--trajectory-report-max-bytes", type=int, default=8192)
    add_native_video_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    from true_rl_agent.tasks.stair_climbing.seed_policy import diagnostic_seed_namespace

    diagnostic_namespace = diagnostic_seed_namespace(args.diagnostic_cohort)
    if args.cohort != "diagnostic" and args.diagnostic_cohort != "v1":
        raise ValueError("Stair diagnostic cohort requires --cohort diagnostic")
    if args.numeric_evidence_max_bytes <= 0 or args.timeout_s <= 0:
        raise ValueError("numeric evidence byte limit and timeout must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "purpose": ("Stair same-checkpoint fixed public-dev replay" if args.cohort == "formal"
                    else "Stair same-checkpoint independent public-dev diagnostic"), "status": "PREPARING",
        "cohort": args.cohort,
        "mode": args.mode, "capture_numeric": args.capture_numeric == "on", "started_at_unix": time.time(),
        "train_jobs": 0, "accept_jobs": 0, "sealed_jobs": 0, "final_jobs": 0,
        "provider_requests": 0, "reference_qualification_performed": False, "phases_executed": [],
    }
    if args.diagnostic_cohort != "v1":
        report.update(diagnostic_cohort=args.diagnostic_cohort, seed_namespace=diagnostic_namespace)
    attempt_scope = None
    phase_invoked = False
    probe_lease_seconds = 0.0
    try:
        protocol_path = args.protocol.resolve(strict=True)
        protocol = StairUnifiedProtocol.load(protocol_path)
        row = stair_robot_row(protocol.cell_variant_id)
        if protocol_path != (PROJECT_ROOT / row.protocol_relative_path).resolve() or not row.active:
            raise ValueError("--protocol must select a registered active protocol in this checkout")
        provider = StairClimbingCompositionProvider()
        composition = provider.resolve(
            values={**provider.cli_defaults(), "cell_variant_id": protocol.cell_variant_id,
                    "execution_device": args.device, "isaac_python": args.isaac_python,
                    "runner_timeout_s": args.timeout_s,
                    "behavior_feedback_profile": args.behavior_feedback_profile,
                    "trajectory_report_max_bytes": args.trajectory_report_max_bytes},
            context=TaskCompositionContext(project_root=PROJECT_ROOT, output_dir=output / "composition"),
        )
        backend, config = composition.backend_adapter, composition.task_plugin.config
        summary_path = args.train_summary.absolute()
        source_bytes = read_bounded_regular_file(summary_path, max_bytes=16 * 1024**2, label="source train summary")
        summary = strict_json_object_loads(source_bytes, context="source train summary")
        checkpoint = _completed_source(summary, composition=composition, checkpoint_path=args.checkpoint.absolute())
        source_summary_hash = "sha256:" + hashlib.sha256(source_bytes).hexdigest()
        session_id = "stair-dev-replay-" + checkpoint.checksum.removeprefix("sha256:")[:16]
        if args.cohort == "diagnostic":
            session_id += "-diagnostic"
        environment_hash = None
        gpu_process_binding = None
        if args.mode == "run":
            if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
                raise ValueError("use an unmasked host CUDA ordinal for the production runner")
            execution_identity = canonical_json_hash({
                "purpose": report["purpose"], "checkpoint": checkpoint.checksum,
                "source_train_summary_sha256": source_summary_hash,
                "current_manifest_hash": composition.resolved_cell_manifest.manifest_hash,
                "output": str(output), "capture_numeric": report["capture_numeric"],
            })
            holder = PreGenesisGpuLeaseHolder(
                session_id=session_id, genesis_identity_hash=execution_identity,
                session_contract_hash=execution_identity, lane_id=args.device,
                invocation_segment_id=execution_identity[:16], logical_job_id=session_id,
            )
            manager = GpuLeaseManager(
                lease_root=Path("/tmp/true_rl_agent_gpu_leases"),
                devices={item.execution_device: item.identity for item in discover_nvidia_gpu_inventory()},
            )
            gpu_identity = manager.identity_for(args.device)
            gpu_process_binding = GpuProcessBinding(
                host_execution_device=args.device, process_execution_device="cuda:0",
                resource_binding_hash=gpu_identity.identity_hash,
            )
            with manager.acquire(args.device, holder=holder) as lease:
                probe_started = time.monotonic()
                report["environment_probe_gpu_lease"] = dict(lease.receipt)
                try:
                    environment = backend.resolve_session_execution_environment(
                        session_id=session_id, session_dir=output, allow_create=True,
                    )
                finally:
                    probe_lease_seconds = time.monotonic() - probe_started
            environment_hash = environment["manifest_hash"]
            attempt_scope = PreGenesisExecutionScope(
                journal_root=output / "job_attempts", session_id=session_id,
                genesis_identity_hash=execution_identity, session_contract_hash=execution_identity,
                lane_id=args.device, invocation_segment_id=execution_identity[:16],
                budget_contract_hash=canonical_json_hash({"timeout_s": args.timeout_s, "attempts": 1}),
                attempt_budget_units=1, max_infrastructure_retries=0,
                resource_binding_hash=gpu_identity.identity_hash,
            )
            composition = replace(composition, runner=replace(composition.runner, gpu_lease_manager=manager))
        jobs = backend.materialize_reference_jobs(
            session_id=session_id, task_spec=composition.task_plugin.task_spec,
            reference=composition.reference_program_artifact, artifact_root=output / "eval_template",
            policy_initialization=composition.policy_initialization,
            evaluation_binding=backend.evaluation_binding_for(composition.task_plugin.task_spec),
            backend_execution_environment_manifest_hash=environment_hash,
            gpu_process_binding=gpu_process_binding,
        )
        # The template supplies current evaluation structure, never source authorship.
        plan = replace(
            jobs.eval_spec, checkpoint=checkpoint, checkpoint_output_plan=None,
            subject=ExperimentSubject(kind=summary["subject_kind"], subject_id=summary["subject_id"]),
            subject_manifest_hash=summary.get("subject_manifest_hash"),
            program_source_checksum=summary["program_source_checksum"],
            training_binding_hash=summary["training_binding_hash"],
            capability_profile_hash=summary["capability_profile_hash"],
        )
        plan = backend.rematerialize_eval_plan(
            eval_plan=plan, round_id=session_id, summary_path=(output / "evaluation/summary.json").as_posix(),
        )
        phase_options = dict(
            attempt_scope=attempt_scope, capture_numeric=args.capture_numeric == "on",
            numeric_evidence_dir=(output / "numeric_evidence").as_posix() if args.capture_numeric == "on" else None,
            numeric_evidence_max_bytes=args.numeric_evidence_max_bytes,
        )
        if args.cohort == "diagnostic":
            phase = prepare_stair_public_dev_diagnostic(
                config=config, backend_adapter=backend, formal_eval_plan=plan,
                job_id=plan.job_id + "-diagnostic", round_id=session_id,
                summary_path=(output / "evaluation/summary.json").as_posix(),
                diagnostic_cohort=args.diagnostic_cohort, **phase_options)
        else:
            phase = backend.materialize_eval_phase(
                eval_plan=plan, phase=EvaluationPhase.PUBLIC_DEV, **phase_options)
        video_argv = native_video_argv_from_args(args)
        if video_argv:
            metadata = dict(phase.metadata)
            metadata["cli_argv"] = [*metadata["cli_argv"], *video_argv]
            metadata["cli_command"] = " ".join(metadata["cli_argv"])
            phase = replace(phase, metadata=metadata)
        validate_eval_job_argv(phase, phase.metadata["cli_argv"])
        composition.runner._validate_checkpoint_before_launch(
            checkpoint=checkpoint, metadata=phase.metadata, job_id=phase.job_id,
            argv=phase.metadata["cli_argv"], checkpoint_path_option="--checkpoint",
            checkpoint_checksum_option="--checkpoint-sha256",
        )
        job_path = output / "public_dev_job.json"
        _write_json(job_path, {"spec": phase.to_public_dict(), "metadata": dict(phase.metadata)}, create_once=True)
        report.update(
            cell=protocol.cell_variant_id, current_manifest_hash=composition.resolved_cell_manifest.manifest_hash,
            source_training_manifest_hash=summary["resolved_cell_manifest_hash"],
            source_training_cell_identity=summary["resolved_cell_identity"],
            source_train_summary_path=str(summary_path), source_train_summary_sha256=source_summary_hash,
            checkpoint_sha256=checkpoint.checksum, checkpoint_bytes_checked=True,
            preserved_training_binding_hash=plan.training_binding_hash, source_subject=plan.subject.to_public_dict(),
            evaluation_plan_hash=phase.evaluation_plan_hash,
            evaluation_binding=phase.evaluation_binding.to_public_dict(),
            public_dev_job_spec_hash=canonical_json_hash(phase.to_public_dict()), job_file_sha256=file_sha256(job_path),
            public_dev_seed_count=plan.public_seed_count, eval_num_envs=plan.eval_num_envs,
            planned_public_dev_episodes=plan.public_seed_count * plan.eval_num_envs,
            backend_execution_environment_manifest_hash=environment_hash,
        )
        if args.mode == "preflight":
            report["status"] = "PREFLIGHT_PASS"
        else:
            report["status"] = "RUNNING"
            _write_json(output / "result.json", report)
            phase_invoked = True
            result = composition.runner.run_eval_phase(phase)
            require_eval_phase_result_matches_spec(result, phase)
            public_summary = (
                project_stair_public_dev_diagnostic(config=config,
                    eval_plan=phase.evaluation_plan, phase_result=result,
                    diagnostic_cohort=args.diagnostic_cohort)
                if args.cohort == "diagnostic" else
                composition.task_plugin.phased_evaluation_adapter().project_public_phase_summary(phase_result=result)
            )
            result_path = output / "public_dev_result.json"
            _write_json(result_path, result.to_private_dict(), create_once=True)
            report.update(
                status="PASS", phases_executed=[result.phase.value],
                public_dev_result_sha256=file_sha256(result_path),
                public_dev_result_hash=canonical_json_hash(result.to_private_dict()),
                public_dev_summary_hash=canonical_json_hash(public_summary),
                task_projection_validated=True,
                evaluation_execution_cost=dict(result.execution_cost),
            )
            if "numeric_evidence" in result.task_evidence:
                report["numeric_evidence_ref"] = dict(result.task_evidence["numeric_evidence"])
        if file_sha256(args.checkpoint.absolute()) != checkpoint.checksum:
            raise ValueError("checkpoint bytes changed during replay")
        if summary_path.read_bytes() != source_bytes:
            raise ValueError("source train summary changed during replay")
        return 0
    except Exception:
        report.update(status="FAIL", error=traceback.format_exc())
        traceback.print_exc()
        return 1
    finally:
        report["environment_probe_gpu_lease_wall_duration_s"] = probe_lease_seconds
        # The runner's phase receipt stops GPU timing before CPU finalization.
        # Missing phase timing remains unknown, never a zero-duration success.
        phase_lease_seconds = 0.0 if not phase_invoked else None
        if attempt_scope is not None:
            receipts = list(attempt_scope.journal_root.rglob("attempt_*.result.json"))
            report["phase_attempt_receipts"] = [str(path) for path in receipts]
            durations = [json.loads(path.read_text(encoding="utf-8")).get("resource_lease_wall_duration_ns")
                         for path in receipts]
            if durations and all(value is not None for value in durations):
                phase_lease_seconds = sum(durations) / 1e9
        report["eval_phase_gpu_lease_wall_duration_s"] = phase_lease_seconds
        report["gpu_lease_wall_duration_s"] = (
            probe_lease_seconds + phase_lease_seconds if phase_lease_seconds is not None else None)
        report["ended_at_unix"] = time.time()
        _write_json(output / "result.json", report)
        print(json.dumps({"status": report["status"], "output": str(output)}), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
