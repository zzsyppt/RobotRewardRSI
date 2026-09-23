"""One fixed stationary-v3 development training, followed only by public_dev.

Run under P6/run_owned_probe.py with a 3600-second total cell timeout.
Preflight samples the target Python environment but launches no train/eval job.
Each invocation requires a new output directory; there is no retry/resume path.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from true_rl_agent.composition import TaskCompositionContext
from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.evaluation.phases import EvaluationPhase
from true_rl_agent.runner.checkpoint import file_sha256
from true_rl_agent.runner.eval_binding import bind_eval_spec_to_checkpoint
from true_rl_agent.runner.job_argv import validate_eval_job_argv, validate_train_job_argv
from true_rl_agent.tasks.push_recovery.composition import PushRecoveryCompositionProvider
from true_rl_agent.training_programs import resolve_reference_program_artifact

CELLS = ("push.g1_37dof.recovery_v2", "push.go2_12dof.recovery_v2")
REFERENCE_ID = "push_standing_recovery_stationary_reference_v3"
SOURCE = ROOT / "PLAN/2new_world/P6/dev_reference_stationary_v3/training_code/components.py"
BUDGET = {"train_num_envs": 4096, "train_steps_per_iteration": 24, "train_max_iterations": 500,
          "budget_steps": 49152000, "train_seed": 42, "eval_num_envs": 64,
          "public_seed_count": 3, "eval_base_seed": 42, "runner_timeout_s": 3600}


def write(output, name, value):
    (output / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def job_record(spec):
    # Retain the existing spec and its actual execution metadata/argv.
    return {"spec": spec.to_public_dict(), "metadata": dict(spec.metadata),
            "argv": list(spec.metadata["cli_argv"])}


def build_composition(args):
    provider = PushRecoveryCompositionProvider()
    return provider.resolve(values={**provider.cli_defaults(), **BUDGET,
        "cell_variant_id": args.cell, "execution_device": args.device,
        "isaac_python": args.isaac_python},
        context=TaskCompositionContext(project_root=ROOT, output_dir=args.output / "composition"))


def resolve_program(composition, expected_checksum):
    task = composition.task_plugin.task_spec
    reference = resolve_reference_program_artifact(SOURCE, reference_id=REFERENCE_ID,
        task_id=task.task_id, authorship_origin="llm_generated", upstream_project="true_RL_Agent",
        upstream_revision="standing_recovery_v2_stationary_only_v3",
        upstream_source_path=SOURCE.relative_to(ROOT).as_posix(),
        capability_profile_hash=task.candidate_capability_profile.profile_hash)
    if reference.manifest.source_checksum != expected_checksum:
        raise ValueError("development reference source differs from the declared bytes")
    return reference


def materialize(composition, reference, *, output, session_id, environment_hash):
    backend, task = composition.backend_adapter, composition.task_plugin.task_spec
    jobs = backend.materialize_reference_jobs(session_id=session_id, task_spec=task,
        reference=reference, artifact_root=output / "reference_artifacts",
        policy_initialization=composition.policy_initialization,
        evaluation_binding=backend.evaluation_binding_for(task),
        backend_execution_environment_manifest_hash=environment_hash)
    train, plan = jobs.train_spec, jobs.eval_spec
    if (train.subject.subject_id != REFERENCE_ID or plan.subject != train.subject
            or train.program_source_checksum != reference.manifest.source_checksum
            or plan.program_source_checksum != train.program_source_checksum
            or train.subject_manifest_hash != reference.manifest.manifest_hash
            or plan.subject_manifest_hash != train.subject_manifest_hash
            or train.budget_steps != BUDGET["budget_steps"]
            or train.policy_initialization.kind != "random" or train.training_start_checkpoint() is not None
            or train.attempt_scope is not None or plan.attempt_scope is not None):
        raise ValueError("custom reference identity, random initialization or fixed budget mismatch")
    validate_train_job_argv(train, train.metadata["cli_argv"])
    validate_eval_job_argv(plan, plan.metadata["cli_argv"])
    return jobs


def public_dev(composition, jobs, checkpoint):
    plan = bind_eval_spec_to_checkpoint(jobs.eval_spec, checkpoint)
    phase = composition.backend_adapter.materialize_eval_phase(eval_plan=plan, phase=EvaluationPhase.PUBLIC_DEV)
    validate_eval_job_argv(phase, phase.metadata["cli_argv"])
    if phase.phase != EvaluationPhase.PUBLIC_DEV or phase.attempt_scope is not None:
        raise ValueError("development driver permits only one public_dev invocation without retries")
    return phase


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--mode", choices=("preflight", "run"), required=True)
    parser.add_argument("--cell", choices=CELLS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--isaac-python", default=sys.executable)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--expected-environment-hash",
                        help="Required for run; take the manifest hash from target preflight.")
    args = parser.parse_args(argv)
    if args.mode == "run" and not args.expected_environment_hash:
        parser.error("run requires --expected-environment-hash from target preflight")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    label = "g1" if args.cell == CELLS[0] else "go2"
    session_id = "p6-stationary-v3-dev-" + label
    report = {"status": "RUNNING", "development_only": True, "enters_qualification": False,
              "mode": args.mode, "cell": args.cell, "session_id": session_id,
              "budget": BUDGET, "train_jobs_invoked": 0, "train_jobs_completed": 0,
              "phases_invoked": [], "phases_executed": [],
              "infrastructure_retries": 0, "total_cell_wall_limit_s": 3600,
              "started_at_unix": time.time()}
    try:
        if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
            raise ValueError("outer lease uses host device ordinals; CUDA_VISIBLE_DEVICES must be unset")
        composition = build_composition(args)
        reference = resolve_program(composition, args.expected_source_sha256)
        backend = composition.backend_adapter
        environment = backend.resolve_session_execution_environment(
            session_id=session_id, session_dir=args.output, allow_create=True)
        if args.expected_environment_hash and environment["manifest_hash"] != args.expected_environment_hash:
            raise ValueError("target execution environment differs from the declared preflight")
        write(args.output, "backend_execution_environment.json", environment)
        write(args.output, "reference_program.json", reference.to_public_dict())
        write(args.output, "source_receipt.json", {"driver_path": str(Path(__file__).resolve()),
            "driver_sha256": file_sha256(Path(__file__)), "source_path": str(SOURCE),
            "source_sha256": reference.manifest.source_checksum,
            "reference_id": REFERENCE_ID, "reference_manifest_hash": reference.manifest.manifest_hash})
        jobs = materialize(composition, reference, output=args.output, session_id=session_id,
                           environment_hash=environment["manifest_hash"])
        write(args.output, "reference_train_job.json", job_record(jobs.train_spec))
        write(args.output, "evaluation_plan_pending_checkpoint.json", job_record(jobs.eval_spec))
        report.update(reference_id=REFERENCE_ID, program_source_checksum=reference.manifest.source_checksum,
            subject_manifest_hash=jobs.train_spec.subject_manifest_hash,
            resolved_cell_manifest_hash=composition.resolved_cell_manifest.manifest_hash,
            training_binding_hash=jobs.train_spec.training_binding.binding_hash,
            execution_environment_manifest_hash=environment["manifest_hash"],
            train_job_spec_hash=canonical_json_hash(jobs.train_spec.to_public_dict()))
        if args.mode == "preflight":
            report.update(status="PREPARED", checkpoint_state="pending_training",
                          planned_phase=EvaluationPhase.PUBLIC_DEV.value)
            return 0
        report["train_jobs_invoked"] = 1
        write(args.output, "result.json", report)
        trained = composition.runner.run_train(jobs.train_spec)
        write(args.output, "train_result.json", trained.to_public_dict())
        report["train_jobs_completed"] = 1
        checkpoint_path = Path(trained.checkpoint.metadata["local_path"])
        if file_sha256(checkpoint_path) != trained.checkpoint.checksum:
            raise ValueError("trained checkpoint bytes changed after runner verification")
        phase = public_dev(composition, jobs, trained.checkpoint)
        write(args.output, "public_dev_job.json", job_record(phase))
        report.update(checkpoint_sha256=trained.checkpoint.checksum,
                      public_dev_job_spec_hash=canonical_json_hash(phase.to_public_dict()))
        report["phases_invoked"] = [EvaluationPhase.PUBLIC_DEV.value]
        write(args.output, "result.json", report)
        result = composition.runner.run_eval_phase(phase)
        write(args.output, "public_dev_result.json", result.to_private_dict())
        public = composition.task_plugin.phased_evaluation_adapter().project_public_phase_summary(phase_result=result)
        report.update(status="PASS", phases_executed=[result.phase.value], public_dev=public,
                      evaluation_execution_cost=dict(result.execution_cost))
        return 0
    except Exception:
        report.update(status="FAIL", error=traceback.format_exc())
        traceback.print_exc()
        return 1
    finally:
        report["ended_at_unix"] = time.time()
        write(args.output, "result.json", report)
        print(json.dumps({"status": report["status"], "output": str(args.output)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
