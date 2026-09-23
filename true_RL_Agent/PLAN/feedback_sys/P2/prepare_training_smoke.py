"""CPU-only preparation of current G1/Go2 smoke inputs using task composition.

Run with the target Isaac Python after synchronizing the complete source tree.
This creates no simulator, GPU job, model call, evaluation, or qualification.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", choices=("stair", "push"), required=True)
    parser.add_argument("--cell", help="Current task-matching G1/Go2 v3 cell; defaults to Go2")
    parser.add_argument("--isaac-python", default=sys.executable)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--candidate-dir", type=Path, help="Optional intact saved candidate package; default is official reference")
    args = parser.parse_args(argv)
    allowed_cells = {"stair": ("stair.g1_37dof.unified_v3", "stair.go2_12dof.unified_v3"),
                     "push": ("push.g1_37dof.recovery_v3", "push.go2_12dof.recovery_v3")}
    cell = args.cell or allowed_cells[args.task][1]
    if cell not in allowed_cells[args.task]:
        parser.error("--cell must be a current G1/Go2 v3 cell for the selected task")
    root = args.project_root.resolve(strict=True)
    sys.path.insert(0, str(root / "src"))
    from true_rl_agent.composition import TaskCompositionContext
    from true_rl_agent.runner.checkpoint import file_sha256
    from true_rl_agent.runner.job_argv import validate_train_job_argv
    from true_rl_agent.tasks.push_recovery.composition import PushRecoveryCompositionProvider
    from true_rl_agent.tasks.stair_climbing.composition import StairClimbingCompositionProvider

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    def write(name, value):
        (output / name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    provider = StairClimbingCompositionProvider() if args.task == "stair" else PushRecoveryCompositionProvider()
    values = {**provider.cli_defaults(), "cell_variant_id": cell,
        "isaac_python": args.isaac_python, "execution_device": args.device, "runner_timeout_s": 900}
    if args.task == "push":
        values.update(budget_steps=384, train_num_envs=16, train_max_iterations=3, train_steps_per_iteration=8)
    composition = provider.resolve(values=values, context=TaskCompositionContext(project_root=root, output_dir=output / "composition"))
    backend = composition.backend_adapter
    session = "training-numeric-smoke-" + args.task
    environment = backend.resolve_session_execution_environment(session_id=session, session_dir=output, allow_create=True)
    common = dict(task_spec=composition.task_plugin.task_spec, policy_initialization=composition.policy_initialization,
        evaluation_binding=backend.evaluation_binding_for(composition.task_plugin.task_spec),
        backend_execution_environment_manifest_hash=environment["manifest_hash"])
    if args.candidate_dir is None:
        jobs = backend.materialize_reference_jobs(session_id=session, reference=composition.reference_program_artifact,
            artifact_root=output / "reference_source", **common)
    else:
        from true_rl_agent.candidates.loader import load_candidate_manifest
        from true_rl_agent.candidates.schema import CandidatePackageRef, CandidateChangeSet
        from true_rl_agent.candidates.verifier import seal_and_verify_candidate
        candidate_dir = args.candidate_dir.resolve(strict=True)
        manifest = load_candidate_manifest(candidate_dir)
        candidate = CandidatePackageRef(candidate_id=manifest.candidate_id, path=str(candidate_dir),
                                        change_set=CandidateChangeSet(frozenset()))
        verified = seal_and_verify_candidate(candidate, artifact_root=output / "candidate_source",
                                             verifier=composition.candidate_admission_policy)
        write("candidate_verification.json", verified.result.to_public_dict())
        jobs = backend.materialize_jobs(round_id=session, candidate=verified.require_verified(), **common)
    train = jobs.train_spec
    train_argv = list(train.metadata["cli_argv"])
    validate_train_job_argv(train, train_argv)
    write("execution_environment_manifest.json", environment)
    write("train_spec.json", train.to_public_dict())
    write("train_argv.json", train_argv)
    write("preparation.json", {"status": "PREPARED_NOT_EXECUTED", "task": args.task,
        "project_root": str(root), "cell_variant_id": composition.resolved_cell_manifest.cell_variant_id,
        "subject": train.subject.to_public_dict(), "subject_manifest_hash": train.subject_manifest_hash,
        "program_source_checksum": train.program_source_checksum, "training_binding_hash": train.training_binding.binding_hash,
        "resolved_cell_manifest_hash": composition.resolved_cell_manifest.manifest_hash,
        "execution_environment_manifest_hash": environment["manifest_hash"], "train_argv_sha256": file_sha256(output / "train_argv.json"),
        "input_contract_budget_steps": train.budget_steps, "executed_smoke_budget_steps": 128 if args.task == "stair" else 384,
        "train_process_starts": 0, "simulator_starts": 0, "provider_requests": 0,
        "policy_qualification": False, "formal_training_claim": False,
        "stair_input_contract_is_production_but_smoke_executor_is_bounded": args.task == "stair"})
    print(str(output / "train_argv.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
