"""Historical P6 Stair V3 regression; P6/run_owned_probe owns its GPU lease.

For current same-checkpoint public-dev replay use scripts/stair_numeric_dev_replay.py,
which validates completed source training and preserves its actual subject/binding.
This script remains only for interpreting/reproducing the historical P6 experiment.

This reuses the P3 evaluation-only successor admission and production phase runner.
It never executes training, acceptance, sealed/final evaluation, or qualification.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
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
from true_rl_agent.runner.job_argv import validate_eval_job_argv
from true_rl_agent.runner.specs import CheckpointRef
from true_rl_agent.tasks.stair_climbing.composition import StairClimbingCompositionProvider


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--cell", choices=("stair.g1_37dof.unified_v3", "stair.go2_12dof.unified_v3"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-summary", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("preflight", "run"), required=True)
    parser.add_argument("--isaac-python", default=sys.executable)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"purpose": "P6 same-checkpoint current Stair V3 public-dev regression", "cell": args.cell,
        "status": "RUNNING", "mode": args.mode, "started_at_unix": time.time(),
        "train_jobs": 0, "accept_jobs": 0, "sealed_jobs": 0, "final_jobs": 0,
        "provider_requests": 0, "reference_qualification_performed": False, "phases_executed": []}
    def write(name, value):
        (output / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        provider = StairClimbingCompositionProvider()
        composition = provider.resolve(values={**provider.cli_defaults(), "cell_variant_id": args.cell,
            "execution_device": args.device, "isaac_python": args.isaac_python, "runner_timeout_s": 1800},
            context=TaskCompositionContext(project_root=ROOT, output_dir=output / "composition"))
        backend, config = composition.backend_adapter, composition.task_plugin.config
        trained = json.loads(args.checkpoint_summary.read_text(encoding="utf-8"))
        checksum = trained["checkpoint"]["checksum"]
        if trained["status"] != "pass" or trained["training"]["budget_accounting"]["planned_env_steps"] != config.budget_steps:
            raise ValueError("source training receipt is not the preserved completed training budget")
        environment_hash = None
        if args.mode == "run":
            if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
                raise ValueError("outer lease requires CUDA_VISIBLE_DEVICES unset and the host CUDA ordinal")
            if file_sha256(args.checkpoint) != checksum:
                raise ValueError("checkpoint differs from its frozen training receipt")
            environment = backend.resolve_session_execution_environment(session_id="p6-stair-dev-regression",
                session_dir=output, allow_create=True)
            environment_hash = environment["manifest_hash"]
            report["execution_environment_manifest_hash"] = environment_hash
        jobs = backend.materialize_reference_jobs(session_id="p6-stair-dev-regression",
            task_spec=composition.task_plugin.task_spec, reference=composition.reference_program_artifact,
            artifact_root=output / "reference_source", policy_initialization=composition.policy_initialization,
            evaluation_binding=backend.evaluation_binding_for(composition.task_plugin.task_spec),
            backend_execution_environment_manifest_hash=environment_hash)
        if trained["training_binding_hash"] != jobs.eval_spec.training_binding_hash:
            raise ValueError("V3 evaluation-only successor changed the checkpoint's training binding")
        # P3's existing successor path retains the source cell identity, never relabels it as V3.
        plan = replace(jobs.eval_spec, checkpoint=CheckpointRef(uri="artifact://p6-stair-dev-regression/best.pt",
            checksum=checksum, role="eval", sealed=True, metadata={"local_path": str(args.checkpoint.resolve()),
                "resolved_cell_manifest_hash": trained["resolved_cell_manifest_hash"]}), checkpoint_output_plan=None)
        plan = backend.rematerialize_eval_plan(eval_plan=plan, round_id="p6-stair-dev-regression",
            summary_path=(output / "evaluation/summary.json").as_posix())
        phase = backend.materialize_eval_phase(eval_plan=plan, phase=EvaluationPhase.PUBLIC_DEV)
        validate_eval_job_argv(phase, phase.metadata["cli_argv"])
        write("public_dev_job.json", phase.to_public_dict())
        report.update(current_manifest_hash=composition.resolved_cell_manifest.manifest_hash,
            source_training_manifest_hash=trained["resolved_cell_manifest_hash"],
            training_summary_sha256=file_sha256(args.checkpoint_summary), checkpoint_sha256=checksum,
            checkpoint_bytes_checked=args.mode == "run", preserved_training_binding_hash=plan.training_binding_hash,
            evaluation_plan_hash=phase.evaluation_plan_hash, public_dev_job_spec_hash=canonical_json_hash(phase.to_public_dict()),
            public_dev_seeds=plan.public_seed_count, eval_num_envs=plan.eval_num_envs, timeout_s=1800)
        if args.mode == "preflight":
            report["status"] = "PREPARED"
            return 0
        write("result.json", report)
        result = composition.runner.run_eval_phase(phase)
        write("public_dev_result.json", result.to_private_dict())
        argv = phase.metadata["cli_argv"]
        raw = json.loads(Path(argv[argv.index("--summary-path") + 1]).read_text(encoding="utf-8"))
        if file_sha256(args.checkpoint) != checksum:
            raise ValueError("checkpoint bytes changed during public-dev regression")
        report.update(status="PASS", phases_executed=[result.phase.value],
            constructed_learner=raw["constructed_learner"], checkpoint_loading=raw["checkpoint_loading"],
            checkpoint_safety_report=raw["checkpoint_safety_report"],
            public_dev=composition.task_plugin.phased_evaluation_adapter().project_public_phase_summary(phase_result=result),
            evaluation_execution_cost=dict(result.execution_cost))
        return 0
    except Exception:
        report.update(status="FAIL", error=traceback.format_exc())
        traceback.print_exc()
        return 1
    finally:
        report["ended_at_unix"] = time.time()
        write("result.json", report)
        print(json.dumps({"status": report["status"], "output": str(output)}), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
