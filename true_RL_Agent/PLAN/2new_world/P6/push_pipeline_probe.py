"""Diagnostic Push train + public-dev jobs through the production pipeline.

Run execution through P6/run_owned_probe.py, which owns the GPU lease. This
script never qualifies a reference, creates Genesis, or queries another phase.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from true_rl_agent.composition import TaskCompositionContext
from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.evaluation.phases import EvaluationPhase
from true_rl_agent.runner.checkpoint import file_sha256
from true_rl_agent.runner.eval_binding import bind_eval_spec_to_checkpoint
from true_rl_agent.runner.job_argv import (
    validate_eval_job_argv,
    validate_train_job_argv,
)
from true_rl_agent.tasks.push_recovery.composition import (
    PushRecoveryCompositionProvider,
)
from true_rl_agent.tasks.push_recovery.config import (
    PUSH_ACTIVE_CELLS,
)

CELLS = PUSH_ACTIVE_CELLS
TRAIN_NUM_ENVS, TRAIN_STEPS_PER_ITERATION, TRAIN_ITERATIONS = 64, 24, 2
TRAIN_SEED, DEV_SEED_COUNT, EVAL_NUM_ENVS = 731, 1, 64


def build_composition(*, cell, output, device, isaac_python, timeout_s):
    provider = PushRecoveryCompositionProvider()
    values = {**provider.cli_defaults(), "cell_variant_id": cell,
              "execution_device": device, "isaac_python": isaac_python,
              "runner_timeout_s": timeout_s,
              "train_num_envs": TRAIN_NUM_ENVS,
              "train_steps_per_iteration": TRAIN_STEPS_PER_ITERATION,
              "train_max_iterations": TRAIN_ITERATIONS,
              "budget_steps": TRAIN_NUM_ENVS * TRAIN_STEPS_PER_ITERATION * TRAIN_ITERATIONS,
              "train_seed": TRAIN_SEED, "eval_num_envs": EVAL_NUM_ENVS,
              "public_seed_count": DEV_SEED_COUNT}
    return provider.resolve(values=values,
        context=TaskCompositionContext(project_root=ROOT, output_dir=output / "composition"))


def materialize_jobs(composition, *, output, environment_hash=None):
    backend = composition.backend_adapter
    jobs = backend.materialize_reference_jobs(
        session_id="p6-push-pipeline", task_spec=composition.task_plugin.task_spec,
        reference=composition.reference_program_artifact,
        artifact_root=output / "reference_program_artifacts",
        policy_initialization=composition.policy_initialization,
        evaluation_binding=backend.evaluation_binding_for(composition.task_plugin.task_spec),
        backend_execution_environment_manifest_hash=environment_hash)
    validate_train_job_argv(jobs.train_spec, jobs.train_spec.metadata["cli_argv"])
    validate_eval_job_argv(jobs.eval_spec, jobs.eval_spec.metadata["cli_argv"])
    return jobs


def materialize_public_dev(composition, jobs, checkpoint):
    plan = bind_eval_spec_to_checkpoint(jobs.eval_spec, checkpoint)
    phase = composition.backend_adapter.materialize_eval_phase(
        eval_plan=plan, phase=EvaluationPhase.PUBLIC_DEV)
    validate_eval_job_argv(phase, phase.metadata["cli_argv"])
    return phase


def _write(output, name, payload):
    (output / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--cell", choices=CELLS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--mode", choices=("preflight", "run"), required=True)
    parser.add_argument("--isaac-python", default=sys.executable)
    parser.add_argument("--timeout", type=int, default=1800,
                        help="Maximum seconds for each of the two diagnostic jobs.")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"purpose": "P6 diagnostic production pipeline only", "mode": args.mode,
              "cell": args.cell, "started_at_unix": time.time(),
              "status": "RUNNING", "phases_executed": [],
              "reference_qualification_performed": False, "genesis_created": False,
              "provider_requests": 0, "infrastructure_retries": 0}
    try:
        composition = build_composition(cell=args.cell, output=output, device=args.device,
                                        isaac_python=args.isaac_python, timeout_s=args.timeout)
        config = composition.task_plugin.config
        manifest = composition.resolved_cell_manifest
        workload = composition.task_plugin.phased_evaluation_adapter().generation_workload(cell_manifest=manifest)
        report.update(manifest_hash=manifest.manifest_hash,
            task_protocol_hash=composition.task_plugin.task_spec.resolved_task_protocol_hash(),
            budget={"train_num_envs": config.train_num_envs,
                    "train_steps_per_iteration": config.train_steps_per_iteration,
                    "train_max_iterations": config.train_max_iterations,
                    "train_control_steps": config.budget_steps, "train_seed": config.train_seed,
                    "dev_seed_count": config.public_seed_count, "eval_num_envs": config.eval_num_envs,
                    "eval_base_seed": config.eval_base_seed,
                    "max_dev_control_steps": workload.max_eval_control_steps_per_seed_unit,
                    "runner_timeout_s_per_job": composition.runner.timeout_s})
        environment_hash = None
        if args.mode == "run":
            if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
                raise ValueError("pipeline probe requires CUDA_VISIBLE_DEVICES unset to preserve the outer lease's host ordinal")
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            environment = composition.backend_adapter.resolve_session_execution_environment(
                session_id="p6-push-pipeline", session_dir=output, allow_create=True)
            environment_hash = environment["manifest_hash"]
            report["execution_environment"] = environment
        jobs = materialize_jobs(composition, output=output, environment_hash=environment_hash)
        _write(output, "train_job.json", jobs.train_spec.to_public_dict())
        _write(output, "evaluation_plan_pending_checkpoint.json", jobs.eval_spec.to_public_dict())
        report["train_job_spec_hash"] = canonical_json_hash(jobs.train_spec.to_public_dict())
        if args.mode == "preflight":
            report.update(status="PREPARED", checkpoint_state="pending_training",
                          planned_phase=EvaluationPhase.PUBLIC_DEV.value)
            return 0
        _write(output, "result.json", report)
        trained = composition.runner.run_train(jobs.train_spec)
        checkpoint_path = Path(trained.checkpoint.metadata["local_path"])
        if file_sha256(checkpoint_path) != trained.checkpoint.checksum:
            raise ValueError("pipeline train checkpoint bytes changed after runner sealing")
        _write(output, "train_result.json", trained.to_public_dict())
        report["checkpoint_sha256"] = trained.checkpoint.checksum
        phase = materialize_public_dev(composition, jobs, trained.checkpoint)
        _write(output, "public_dev_job.json", phase.to_public_dict())
        report["public_dev_job_spec_hash"] = canonical_json_hash(phase.to_public_dict())
        _write(output, "result.json", report)
        result = composition.runner.run_eval_phase(phase)
        _write(output, "public_dev_result.json", result.to_private_dict())
        public = composition.task_plugin.phased_evaluation_adapter().project_public_phase_summary(phase_result=result)
        report.update(status="PASS", phases_executed=[result.phase.value],
                      public_dev=public, evaluation_execution_cost=dict(result.execution_cost))
        return 0
    except Exception:  # noqa: BLE001 - Persist the diagnostic failure without promoting an artifact.
        report.update(status="FAIL", error=traceback.format_exc())
        traceback.print_exc()
        return 1
    finally:
        report["ended_at_unix"] = time.time()
        _write(output, "result.json", report)
        print(json.dumps({"status": report["status"], "output": str(output)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
