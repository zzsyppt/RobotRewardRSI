"""P5 short PPO diagnostic; no formal training or acceptance claim."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
import traceback
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))


def main():
    from isaaclab.app import AppLauncher
    from true_rl_agent.candidates.component_source_semantics import project_single_component_operation_delta
    from true_rl_agent.candidates.verifier import StaticCandidateVerifierPolicy
    from true_rl_agent.evaluation.binding import canonical_json_hash
    from true_rl_agent.evaluation.subject import ExperimentSubject
    from true_rl_agent.runner.isaac_app_lifecycle import bind_isaac_log_path, close_isaac_app_on_success
    from true_rl_agent.tasks.stair_climbing.config import StairClimbingTaskConfig
    from true_rl_agent.tasks.stair_climbing.plugin import StairClimbingTaskPlugin
    from true_rl_agent.tasks.stair_climbing.robot_binding import resolve_stair_cell_manifest
    from true_rl_agent.tasks.stair_climbing.reference_program import resolve_official_stair_reference_program
    from true_rl_agent.tasks.stair_climbing.reward_program import qualify_stair_diagnostic_parent
    from true_rl_agent.tasks.stair_climbing.training_diagnostics import bind_stair_training_diagnostics, load_stair_training_comparison
    from true_rl_agent.tasks.stair_climbing.isaaclab_execution import execute_stair_training_smoke, preflight_stair_execution_source_manifest
    from true_rl_agent.runner.specs import TrainJobSpec
    from true_rl_agent.tasks.stair_climbing.composition import StairClimbingCompositionProvider
    from true_rl_agent.composition.base import TaskCompositionContext
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", choices=("go2", "g1"), required=True)
    parser.add_argument("--effect", choices=("zero", "nonzero"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(headless=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    filename = "stair_go2_12dof_unified_protocol_v3.json" if args.robot == "go2" else "stair_unified_protocol_v3.json"
    config = StairClimbingTaskConfig.defaults(ROOT / "configs/tasks" / filename)
    cell = resolve_stair_cell_manifest(cell_variant_id=config.cell_variant_id, config=config, plugin=StairClimbingTaskPlugin(config))
    config = replace(config, resolved_cell_task_protocol_hash=cell.resolved_cell_task_protocol_hash)
    official = resolve_official_stair_reference_program(config, project_root=ROOT, resolved_cell_manifest=cell)
    from true_rl_agent.tasks.stair_climbing.reference_program import resolve_official_stair_reference_artifact
    reference, _ = resolve_official_stair_reference_artifact(config, project_root=ROOT, resolved_cell_manifest=cell)
    tree = ast.parse(reference.source_text)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name != "build_training_components")
    ret = next(n for n in ast.walk(fn) if isinstance(n, ast.Return))
    extra = "inputs.roll * 0.0" if args.effect == "zero" else "inputs.roll * inputs.roll"
    ret.value = ast.BinOp(left=ret.value, op=ast.Add(), right=ast.parse(extra, mode="eval").body)
    source = ast.unparse(ast.fix_missing_locations(tree)) + "\n"
    (output / "components.py").write_text(source)
    delta = project_single_component_operation_delta(candidate_source=source, anchor_source=reference.source_text)
    candidate = qualify_stair_diagnostic_parent(contract=official.reward_program.contract,
        subject=ExperimentSubject.candidate("p5-runtime-" + args.effect),
        subject_manifest_hash=canonical_json_hash({"purpose": "P5 diagnostic", "source": source}),
        source_bytes=source.encode(), expected_source_checksum="sha256:" + hashlib.sha256(source.encode()).hexdigest(),
        source_name=str(output / "components.py"), static_verifier_policy=StaticCandidateVerifierPolicy(),
        trusted_reference_source_text=reference.source_text)
    # Exercise the production sidecar writer and loader against a materialized job.
    provider = StairClimbingCompositionProvider()
    composition = provider.resolve(values={**provider.cli_defaults(), "cell_variant_id": config.cell_variant_id},
        context=TaskCompositionContext(project_root=ROOT, output_dir=output / "composition"))
    from true_rl_agent.backends.isaac_lab_execution_environment import build_current_isaac_lab_execution_environment_manifest
    environment = build_current_isaac_lab_execution_environment_manifest()
    jobs = composition.backend_adapter.materialize_reference_jobs(session_id=output.name,
        task_spec=composition.task_plugin.task_spec, reference=reference, artifact_root=output / "reference",
        policy_initialization=composition.policy_initialization,
        evaluation_binding=composition.backend_adapter.evaluation_binding_for(composition.task_plugin.task_spec),
        backend_execution_environment_manifest_hash=environment.manifest_hash)
    spec: TrainJobSpec = bind_stair_training_diagnostics(jobs.train_spec, parent_source=reference.source_text,
        parent_evidence={"subject_manifest_hash": reference.manifest.manifest_hash,
            "program_source_checksum": reference.manifest.source_checksum, "anchor_id": reference.manifest.reference_id},
        mechanism_operation_delta=delta)
    argv = spec.metadata["cli_argv"]
    comparison = load_stair_training_comparison(argv[argv.index("--runtime-comparison-json") + 1], spec.runtime_diagnostic_context_hash,
        context=SimpleNamespace(config=config, resolved_cell_manifest=cell,
            candidate_execution_authorization=SimpleNamespace(static_verifier_policy=StaticCandidateVerifierPolicy())),
        candidate_program=candidate)
    report = {"purpose": "P5 short diagnostic; no acceptance or reference promotion", "robot": args.robot,
        "effect": args.effect, "delta": delta, "runtime_diagnostic_context_hash": spec.runtime_diagnostic_context_hash,
        "environment": environment.to_dict()}
    bind_isaac_log_path(args, output / "result.json")
    app = AppLauncher(args, multi_gpu=False).app
    code = 1
    try:
        result = execute_stair_training_smoke(config=config, source_manifest_preflight=preflight_stair_execution_source_manifest(config),
            agent_configuration=config.training_agent_configuration(), reward_program=candidate, device=args.device,
            run_dir=output / "run", checkpoint_output=output / "diagnostic.pt", run_name=output.name,
            resolved_cell_manifest=cell, runtime_comparison=comparison)
        summary = {**result.to_summary_fields(), "runtime_diagnostic_context_hash": spec.runtime_diagnostic_context_hash,
                   "program_source_checksum": candidate.program_source_checksum}
        issue = StairClimbingTaskPlugin(config, resolved_cell_manifest=cell).training_runtime_issue(
            train_summary=summary, mechanism_operation_delta=delta)
        report.update(training=summary, task_stop_issue=issue)
        assert result.status == "pass"
        assert (issue is not None) == (args.effect == "zero")
        report["status"] = "PASS"
        code = 0
    except Exception:
        report.update(status="FAIL", error=traceback.format_exc())
        traceback.print_exc()
    finally:
        (output / "result.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
        sys.exit(close_isaac_app_on_success(app, exit_code=code))


if __name__ == "__main__":
    main()
